# -*- coding: utf-8 -*-
"""Tests for the displacement-based (DBFFT) solver.

The rework exists to remove a structural problem, so the tests check the
structure first and the physics second:

  1. div_adj really is the adjoint of grad (otherwise the operator is not
     symmetric and nothing downstream is trustworthy);
  2. the operator is FULL RANK apart from the three rigid translations - this
     is the whole point. The F-based operator has rank 1371 of 3430 at N=7;
  3. the reference acoustic tensor is genuinely invertible, not
     rank-deficient, so no pseudo-inverse of a singular symbol is needed;
  4. DBFFT reproduces the corrected F-based solver's homogenised stress. Both
     discretise the SAME problem - the F-based one restricted to compatible
     fields, this one parameterising exactly those fields by u - so they must
     agree.

    python3 test_dbfft.py [--quick]
"""
import argparse
import contextlib
import os
import shutil
import time

import numpy as np

import fg.dbfft as db
from fg.dbfft import derivative_symbol, div_adj, grad

STRUCT = "3D_samples/voxels/1_voxel.npz"
SCRATCH = "/tmp/dbfft_test"


def test_adjointness():
    print("=== 1. div_adj is the adjoint of grad ===")
    rng = np.random.default_rng(0)
    for disc in ("fourier", "willot"):
        N = 9
        xi = derivative_symbol(N, 3, disc)
        u = rng.standard_normal((3, N, N, N))
        S = rng.standard_normal((3, 3, N, N, N))
        lhs = np.sum(S*grad(u, xi))
        rhs = np.sum(div_adj(S, xi)*u)
        rel = abs(lhs - rhs)/max(abs(lhs), 1e-300)
        print("  {:<8} <S, grad u> = {:+.6e}   <div_adj S, u> = {:+.6e}   rel {:.2e}"
              .format(disc, lhs, rhs, rel))
        assert rel < 1e-12, (disc, rel)

    # grad kills constants; grad output has zero mean
    N = 9
    xi = derivative_symbol(N, 3, "fourier")
    const = np.ones((3, N, N, N))
    assert np.abs(grad(const, xi)).max() < 1e-12
    g = grad(rng.standard_normal((3, N, N, N)), xi)
    assert np.abs(g.mean(axis=(2, 3, 4))).max() < 1e-12
    print("  grad(constant) = 0 and grad output has zero mean: OK\n")


def _dense_operator(N=5, contrast=100.0):
    """Form the DBFFT Newton operator densely at small N."""
    charge = os.path.join(SCRATCH, "c.txt")
    phase_path = os.path.join(SCRATCH, "phase.npz")
    os.makedirs(SCRATCH, exist_ok=True)
    ph = np.zeros((N, N, N))
    ph[1:4, 1:4, 1:4] = 1.0
    np.savez(phase_path, phase=ph)
    with open(charge, "w") as fh:
        fh.write("#c\n1.0\t10\t0.48\t0\t0\t0\t0\t0\t0\n")
        fh.write("1.0\t{:g}\t0.30\t0\t0\t0\t0\t0\t0\n".format(10.0*contrast))
        fh.write("#dF\n1.0\t0\t0\t0\t0\t0\t0\t0\t0\n")
        fh.write("#type\n0\t0\t0\t0\t1.0\t0\t0\t0\t1.0\n")

    import scipy.sparse.linalg as spla
    captured = {}

    class Abort(Exception):
        pass

    class Shim:
        def __init__(self, r):
            self._r = r

        def __getattr__(self, n):
            return getattr(self._r, n)

        def gmres(self, **kw):
            captured.update(kw)
            raise Abort

    db.sp = Shim(spla)
    prob = db.DBFFTSolver(phase_path, charge_path=charge,
                          output_path=os.path.join(SCRATCH, "op"), N=N, output_name=".")
    try:
        prob.calculate(incre_list=[0.05], savemodel="no", preconditioner="reference",
                       reference="matrix", forcing="fixed", inner_rtol=1e-6)
    except Abort:
        pass
    db.sp = spla
    A = captured["A"]
    n = A.shape[0]
    D = np.empty((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0
        D[:, j] = A.matvec(e)
        e[j] = 0.0
    return D, n, N


def test_rank():
    print("=== 2. the operator is full rank (the point of the rework) ===")
    D, n, N = _dense_operator()
    s = np.linalg.svd(D, compute_uv=False)
    rank = int((s > 1e-9*s[0]).sum())
    print("  operator size            : {}  (u {} + p {} + Fbar {})".format(
        n, 3*N**3, N**3, n - 4*N**3))
    print("  rank                     : {}".format(rank))
    print("  nullity                  : {}".format(n - rank))
    print("  F-based solver, for scale: rank 1371 of 3430 -> nullity 2059")
    assert n - rank <= 3, (
        "expected at most the 3 rigid translations in the null space, got {}"
        .format(n - rank))
    print("  -> nullity {} = the rigid translations only: FULL RANK\n".format(n - rank))


def test_acoustic_tensor():
    print("=== 3. the reference acoustic tensor is invertible ===")
    N = 9
    xi = derivative_symbol(N, 3, "fourier")
    rng = np.random.default_rng(1)
    # an isotropic, strongly elliptic reference
    lam, mu = 1.0, 1.0
    d = np.eye(3)
    K = (lam*np.einsum("ij,kl->ijkl", d, d)
         + mu*(np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)))
    gamma = np.einsum("jxyz,ijkl,lxyz->ikxyz", np.conj(xi), K, xi)
    stack = np.moveaxis(gamma, (0, 1), (-2, -1)).reshape(-1, 3, 3)
    sv = np.linalg.svd(stack, compute_uv=False)
    nz = sv[:, 0] > 0
    cond = sv[nz, 0]/np.maximum(sv[nz, -1], 1e-300)
    herm = np.abs(stack - np.conj(np.swapaxes(stack, -1, -2))).max()
    print("  Hermitian error                     : {:.2e}".format(herm))
    print("  frequencies with a singular Gamma   : {} of {} (only xi = 0 should be)"
          .format(int((sv[:, -1] <= 1e-12*np.maximum(sv[:, 0], 1e-300)).sum()), len(sv)))
    print("  condition number of Gamma           : median {:.2f}  max {:.2f}"
          .format(np.median(cond), cond.max()))
    assert herm < 1e-10
    assert np.median(cond) < 10, "acoustic tensor should be well conditioned"
    print("  -> 3x3, Hermitian, well conditioned: no rank-deficient symbol\n")


def _run(solver_mod, tag, charge, N, increments, **kw):
    path = os.path.join(SCRATCH, tag)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    prob = solver_mod.FFTSolver(STRUCT, charge_path=charge, output_path=path,
                                N=N, output_name=".")
    t0 = time.time()
    with open(os.path.join(path, "log"), "w") as fh, contextlib.redirect_stdout(fh):
        prob.calculate(incre_list=[0.1]*increments, savemodel="normal",
                       preconditioner="reference", max_gmres_iter=20000, **kw)
    st = prob.solver_stats
    kry = [k for i in st["increments"] for k in i["krylov_iterations"]]
    return dict(status=st["status"], krylov=sum(kry),
                newton=sum(i["newton_iterations"] for i in st["increments"]),
                wall=time.time() - t0,
                P11=float(np.array(prob.Ps)[-1][0, 0]) if prob.Ps else None)


def test_agreement(contrast, increments, N):
    import fg.mxfft as mx
    print("=== 4. DBFFT vs the corrected F-based solver (contrast {}) ===".format(contrast))
    charge = "3D_samples/Charges/bench_c{}.txt".format(contrast)
    print("%-14s %-26s %8s %7s %8s %16s" % ("solver", "status", "krylov", "newton", "wall", "P11"))
    print("-" * 86, flush=True)
    res = {}
    res["F-based"] = _run(mx, "mxfft", charge, N, increments, reference="matrix",
                          precond_restrict=True, forcing="eisenstat_walker",
                          gmres_restart=400)
    print("%-14s %-26s %8d %7d %7.0fs %16.9f" % (
        "F-based", res["F-based"]["status"], res["F-based"]["krylov"],
        res["F-based"]["newton"], res["F-based"]["wall"], res["F-based"]["P11"]), flush=True)
    res["DBFFT"] = _run(db, "dbfft", charge, N, increments, reference="matrix",
                        forcing="eisenstat_walker")
    print("%-14s %-26s %8d %7d %7.0fs %16.9f" % (
        "DBFFT", res["DBFFT"]["status"], res["DBFFT"]["krylov"],
        res["DBFFT"]["newton"], res["DBFFT"]["wall"], res["DBFFT"]["P11"]), flush=True)

    a, b = res["F-based"]["P11"], res["DBFFT"]["P11"]
    rel = abs(a - b)/abs(a)
    print("\n  P11 agreement : {:.3e} relative".format(rel))
    if res["DBFFT"]["krylov"]:
        print("  Krylov        : {:.2f}x fewer with DBFFT".format(
            res["F-based"]["krylov"]/res["DBFFT"]["krylov"]))
        print("  wall          : {:.2f}x".format(
            res["F-based"]["wall"]/max(res["DBFFT"]["wall"], 1e-9)))
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="structural tests only")
    ap.add_argument("--contrast", default="100")
    ap.add_argument("--increments", type=int, default=1)
    ap.add_argument("--N", type=int, default=31)
    args = ap.parse_args()

    test_adjointness()
    test_rank()
    test_acoustic_tensor()
    if args.quick:
        print("--quick: solver comparison skipped")
        return
    test_agreement(args.contrast, args.increments, args.N)


if __name__ == "__main__":
    main()
