# -*- coding: utf-8 -*-
"""Correctness and effectiveness tests for the Green-Jacobi preconditioner.

Two things must hold before any speed claim means anything:

  1. On a HOMOGENEOUS body the Jacobi scale d(x) is uniform, so D^-1/2 is a
     scalar and Green-Jacobi must collapse onto the plain Green preconditioner
     -> M^-1 A is the identity on the compatible subspace.
  2. Its output must stay compatible. D^-1/2 is a pointwise scaling and does
     NOT preserve compatibility, which is why the implementation re-projects;
     this checks the re-projection actually works.

Then the point of the exercise: does it cut iterations at high contrast?

    python3 test_green_jacobi.py [--quick]
"""
import argparse
import contextlib
import os
import shutil
import tempfile
import time

import numpy as np
import scipy.sparse.linalg as spla

from project_paths import CHARGES_DIR, ensure_import_paths, project_path

ensure_import_paths()
import fg.mxfft as mx  # noqa: E402  (needs ensure_import_paths first)
from fg.preconditioning import (  # noqa: E402
    apply_green_jacobi_preconditioner,
    build_Ghat4,
    build_green_jacobi_symbol,
    local_jacobi_scale,
)

N = 31
SCRATCH = os.path.join(tempfile.gettempdir(), "gj_test")
HOMO = os.path.join(SCRATCH, "charge_homogeneous.txt")
VOXEL = project_path("3D_samples", "voxels", "1_voxel.npz")

# Matrix phase shared by every charge file written here: Neo-Hookean, E=10,
# nu=0.48. Contrast is set purely by the filler Young's modulus.
MATRIX = "1.0\t10\t0.48\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n"
TAIL = ("#charge dF\n1.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n"
        "#type\n0.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\n")


def write_homogeneous():
    os.makedirs(SCRATCH, exist_ok=True)
    with open(HOMO, "w") as fh:
        fh.write("#homogeneous: both phases identical\n")
        fh.write(MATRIX)
        fh.write(MATRIX)
        fh.write(TAIL)
    return HOMO


def hetero_charge(contrast):
    """Charge file for a given filler/matrix stiffness ratio.

    Prefers a committed bench_c<contrast>.txt so the well-worn contrasts stay
    reproducible, and otherwise writes an equivalent file into the scratch
    directory -- which is what lets an arbitrary --contrast work.
    """
    committed = os.path.join(CHARGES_DIR, "bench_c{}.txt".format(contrast))
    if os.path.exists(committed):
        return committed

    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, "charge_c{}.txt".format(contrast))
    with open(path, "w") as fh:
        fh.write("#generated: filler/matrix stiffness contrast {}\n".format(contrast))
        fh.write(MATRIX)
        fh.write("1.0\t{:g}\t0.30\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n".format(
            10.0*float(contrast)))
        fh.write(TAIL)
    return path


captured = {}


class Abort(Exception):
    pass


class Shim:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def gmres(self, **kw):
        captured.update(kw)
        raise Abort


def capture(charge, precond, reference="matrix"):
    captured.clear()
    mx.sp = Shim(spla)
    prob = mx.FFTSolver(VOXEL, charge_path=charge,
                        output_path=os.path.join(SCRATCH, "cap"), N=N, output_name=".")
    try:
        prob.calculate(incre_list=[0.1], preconditioner=precond,
                       reference=reference, precond_restrict=True, forcing="fixed",
                       inner_rtol=1e-6)
    except Abort:
        pass
    mx.sp = spla
    return captured["A"], captured["M"], captured["b"], prob.pb.stress_control


def identity_error_and_compatibility(charge, precond, label):
    A, M, b, sc = capture(charge, precond)
    Ghat4 = build_Ghat4(N, sc, 3)
    num_up = 9*N**3
    axes = (-3, -2, -1)
    import scipy.fft
    fft = lambda x: np.fft.fftshift(scipy.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    ifft = lambda x: np.fft.fftshift(scipy.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    G = lambda a: np.real(ifft(mx.ddot42(Ghat4, fft(a))))

    rng = np.random.default_rng(0)
    x = rng.standard_normal(A.shape[0])
    x[:num_up] = G(x[:num_up].reshape(3, 3, N, N, N)).reshape(-1)

    y = M.matvec(A.matvec(x))
    ident = np.linalg.norm(y - x)/np.linalg.norm(x)

    out = M.matvec(x)
    f = out[:num_up].reshape(3, 3, N, N, N)
    incompat = np.linalg.norm(f - G(f))/max(np.linalg.norm(f), 1e-300)
    print("  {:<28} ||M^-1 A x - x||/||x|| = {:.3e}    output incompatible = {:.3e}"
          .format(label, ident, incompat))
    return ident, incompat


def solve_run(tag, charge, precond, reference, increments, restart=400):
    path = os.path.join(SCRATCH, tag)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    prob = mx.FFTSolver(VOXEL, charge_path=charge,
                        output_path=path, N=N, output_name=".")
    t0 = time.time()
    with open(os.path.join(path, "log"), "w") as fh, contextlib.redirect_stdout(fh):
        prob.calculate(incre_list=[0.1]*increments,
                       preconditioner=precond, reference=reference,
                       precond_restrict=True, forcing="eisenstat_walker",
                       max_gmres_iter=20000, gmres_restart=restart)
    st = prob.solver_stats
    inc = st["increments"]
    kry = [k for i in inc for k in i["krylov_iterations"]]
    return dict(status=st["status"], krylov=sum(kry),
                newton=sum(i["newton_iterations"] for i in inc),
                wall=time.time()-t0, per_solve=kry[:6],
                P11=float(np.array(prob.Ps)[-1][0, 0]) if prob.Ps else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the solver runs")
    ap.add_argument("--contrast", default="500")
    ap.add_argument("--increments", type=int, default=1)
    args = ap.parse_args()

    homo = write_homogeneous()
    hetero = hetero_charge(args.contrast)

    print("\n=== 1. HOMOGENEOUS body: Green-Jacobi must collapse onto Green ===")
    ident_g, _ = identity_error_and_compatibility(homo, "green", "Green")
    ident_gj, incompat_gj = identity_error_and_compatibility(homo, "green_jacobi", "Green-Jacobi")
    assert ident_gj < 1e-9, ("Green-Jacobi is NOT the operator inverse on a "
                             "homogeneous body ({:.2e}) - construction is wrong".format(ident_gj))
    assert incompat_gj < 1e-9, ("Green-Jacobi output left the compatible subspace "
                                "({:.2e}) - the re-projection is not working".format(incompat_gj))
    print("  PASS: collapses onto Green, and its output stays compatible")

    print("\n=== 2. HETEROGENEOUS body, contrast {}: does it stay compatible? ==="
          .format(args.contrast))
    _, incompat_h = identity_error_and_compatibility(hetero, "green_jacobi", "Green-Jacobi")
    assert incompat_h < 1e-9, incompat_h
    print("  PASS: output stays compatible at high contrast too")

    if args.quick:
        print("\n--quick: solver runs skipped")
        return

    print("\n=== 3. Does it actually cut iterations? (contrast {}, {} increment(s)) ==="
          .format(args.contrast, args.increments))
    print("%-16s %-26s %8s %7s %8s %14s" % ("preconditioner", "status", "krylov", "newton", "wall", "P11"))
    print("-" * 88, flush=True)
    out = {}
    for precond, label in (("green", "Green"), ("green_jacobi", "Green-Jacobi")):
        r = solve_run(label.replace("-", "_"), hetero, precond, "matrix", args.increments)
        out[label] = r
        print("%-16s %-26s %8d %7d %7.0fs %14s" % (
            label, r["status"], r["krylov"], r["newton"], r["wall"],
            "%.6f" % r["P11"] if r["P11"] else "-"), flush=True)
        print("%-16s   per-solve %s" % ("", r["per_solve"]), flush=True)

    g, gj = out["Green"], out["Green-Jacobi"]
    if gj["krylov"]:
        print("\n  Green-Jacobi vs Green: %.2fx fewer Krylov iterations, %.2fx wall"
              % (g["krylov"]/gj["krylov"], g["wall"]/max(gj["wall"], 1e-9)))
    if g["P11"] and gj["P11"]:
        print("  P11 agreement: %.2e relative  (a preconditioner must not change the answer)"
              % (abs(gj["P11"] - g["P11"])/abs(g["P11"])))


if __name__ == "__main__":
    main()
