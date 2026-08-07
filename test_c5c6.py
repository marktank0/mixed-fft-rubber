# -*- coding: utf-8 -*-
"""Correctness tests for C5 (inexact Newton) and C6 (reference tangent).

Two levels:

  unit      `reference_average` against explicit per-phase computation, and
            the Eisenstat-Walker forcing term against its definition.
  solver    the new code configured as `forcing="fixed", reference="mean"`
            must reproduce the pre-change solver BITWISE, and every C5/C6
            variant must agree with it on P11 to the solver tolerance.

Run from the repository root:

    FFT_WORKERS=2 python3 test_c5c6.py [--quick]

`--quick` skips the solver-level runs (which take a few minutes).
See docs/inexact_newton_and_reference_tangent.md.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np

from fg.mxfft import eisenstat_walker_forcing
from fg.preconditioning import REFERENCE_MODES, reference_average

from benchmark_suite import ensure_baseline   # pins the solver AND its changed deps


# --------------------------------------------------------------- unit tests
def test_reference_average():
    rng = np.random.default_rng(0)
    N = 5
    phase = (rng.random((N, N, N)) < 0.3).astype(float)
    mask_a, mask_b = (phase == 0), (phase == 1)

    K4 = rng.random((3, 3, 3, 3, N, N, N))
    JF = rng.random((3, 3, N, N, N))
    KI = rng.random((N, N, N))

    # ranks: the grid is always the trailing three axes
    for field, want in ((K4, (3, 3, 3, 3)), (JF, (3, 3)), (KI, ())):
        for mode in REFERENCE_MODES:
            assert np.shape(reference_average(field, mode, mask_a, mask_b)) == want

    # "mean" must reproduce the pre-C6 hard-coded expression bitwise
    assert np.array_equal(reference_average(K4, "mean", mask_a, mask_b), np.mean(K4, axis=(4, 5, 6)))
    assert np.array_equal(reference_average(JF, "mean", mask_a, mask_b), np.mean(JF, axis=(2, 3, 4)))
    assert np.array_equal(reference_average(KI, "mean", mask_a, mask_b), np.mean(KI))

    ma = K4[..., mask_a].mean(axis=-1)
    mb = K4[..., mask_b].mean(axis=-1)
    assert np.allclose(reference_average(K4, "matrix", mask_a, mask_b), ma)
    assert np.allclose(reference_average(K4, "mid", mask_a, mask_b), 0.5*(ma + mb))

    # "mean" is volume-fraction weighted; "mid" deliberately is not
    phi = mask_b.mean()
    assert np.allclose(reference_average(K4, "mean", mask_a, mask_b), (1-phi)*ma + phi*mb)

    # a single-phase (unfilled) cell falls back to the whole-cell average
    empty = np.zeros((N, N, N), dtype=bool)
    full = np.ones((N, N, N), dtype=bool)
    for mode in REFERENCE_MODES:
        assert np.allclose(reference_average(K4, mode, full, empty), np.mean(K4, axis=(4, 5, 6)))

    try:
        reference_average(K4, "bogus", mask_a, mask_b)
        raise AssertionError("bad reference mode should raise")
    except ValueError:
        pass
    print("reference_average OK")


def test_forcing():
    f = lambda rn, ro, ep: eisenstat_walker_forcing(rn, ro, ep, gamma=0.9, alpha=2.0,
                                                    eta_min=1e-3, eta_max=1e-2)
    assert f(1e-4, 1e-1, 1e-2) == 1e-3      # quadratic drop -> floor
    assert f(1.0, 1.0, 1e-2) == 1e-2        # no progress -> ceiling
    assert f(1.0, 0.0, 1e-2) == 1e-2        # degenerate inputs do not divide by zero
    assert f(np.nan, 1.0, 1e-2) == 1e-2

    rng = np.random.default_rng(0)
    for _ in range(2000):
        rn, ro, ep = rng.random(3)*np.array([1e-2, 1.0, 1e-2])
        assert 1e-3 <= f(rn, ro, ep) <= 1e-2

    # unclamped, matches Eisenstat-Walker choice 2 with and without the safeguard
    wide = dict(gamma=0.9, alpha=2.0, eta_min=1e-12, eta_max=1-1e-12)
    assert np.isclose(eisenstat_walker_forcing(0.3, 1.0, 0.5, **wide),
                      max(0.9*0.3**2, 0.9*0.5**2))      # safeguard active
    assert np.isclose(eisenstat_walker_forcing(0.3, 1.0, 0.01, **wide),
                      0.9*0.3**2)                        # safeguard inactive
    print("eisenstat_walker_forcing OK")


# ------------------------------------------------------------- solver tests
def test_solver(out_root, structure, charge, N, incre):
    ensure_baseline()
    import fg.mxfft as new
    import fg._baseline.mxfft as base

    def run(tag, module, **kw):
        path = os.path.join(out_root, tag)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        prob = module.FFTSolver(structure, charge_path=charge, output_path=path,
                                N=N, output_name=".")
        t0 = time.time()
        prob.calculate(incre_list=incre, savemodel="normal",
                       preconditioner="reference", **kw)
        st = json.load(open(os.path.join(prob.output_path, "solver_stats.json")))
        kry = [k for i in st["increments"] for k in i["krylov_iterations"]]
        return (np.array(prob.Ps), np.array(prob.Fs), sum(kry), time.time()-t0)

    # ---- C5 regression, against the pre-change solver.
    # This must use precond_restrict=False: the preconditioner fix changes the
    # converged answer (the old symbol left the compatible subspace), so the
    # only meaningful bitwise comparison is against the old preconditioner.
    ref_P, ref_F, ref_k, ref_t = run("baseline", base)
    fix_P, fix_F, fix_k, _ = run("fixed", new, forcing="fixed", inner_rtol=1.e-6,
                                 reference="mean", precond_restrict=False)

    assert np.array_equal(fix_P, ref_P) and np.array_equal(fix_F, ref_F), \
        "forcing='fixed' + precond_restrict=False must reproduce the pre-change solver bitwise"
    assert fix_k == ref_k
    print("forcing='fixed', precond_restrict=False reproduces baseline BITWISE"
          " ({} Krylov its) OK".format(ref_k))

    scale = np.max(np.abs(ref_P))
    for tag, kw in (("EW/mean",   dict(reference="mean")),
                    ("EW/matrix", dict(reference="matrix")),
                    ("EW/mid",    dict(reference="mid"))):
        P, _F, k, t = run(tag.replace("/", "_"), new, forcing="eisenstat_walker",
                          precond_restrict=False, **kw)
        # normalise by the tensor scale: the stress-controlled components are
        # driven to ~0 by construction, so a componentwise ratio is meaningless
        err = np.max(np.abs(P - ref_P))/scale
        p11 = np.max(np.abs(P[:, 0, 0] - ref_P[:, 0, 0])/np.abs(ref_P[:, 0, 0]))
        print("{:<10} krylov {:>5} ({:.2f}x)  wall {:>6.1f}s  relerr(P) {:.1e}  relerr(P11) {:.1e}"
              .format(tag, k, ref_k/k, t, err, p11))
        # with the OLD preconditioner the reference mode still perturbs the
        # answer, hence the looser limit off "mean" - that is the defect the
        # fix removes, and is asserted away below
        limit = 1e-4 if kw["reference"] == "mean" else 5e-3
        assert err < limit and p11 < limit, (tag, err, p11)
    print("C5 preserves the baseline answer at fixed preconditioner OK")

    # ---- the preconditioner fix: the reference tangent must NOT change the answer
    fixed = {}
    for ref in ("mean", "matrix", "mid"):
        P, _F, k, t = run("restrict_" + ref, new, forcing="eisenstat_walker",
                          reference=ref, precond_restrict=True)
        fixed[ref] = P
        print("restrict/{:<7} krylov {:>5}  wall {:>6.1f}s  P11 {:.10f}"
              .format(ref, k, t, P[-1, 0, 0]))
    base_p11 = fixed["mean"][:, 0, 0]
    for ref in ("matrix", "mid"):
        spread = np.max(np.abs(fixed[ref][:, 0, 0] - base_p11)/np.abs(base_p11))
        assert spread < 1e-7, (ref, spread)
    print("with precond_restrict=True the reference tangent does not change the"
          " converged answer (spread < 1e-7) OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="unit tests only")
    ap.add_argument("--N", type=int, default=31)
    ap.add_argument("--structure", default="3D_samples/voxels/1_voxel.npz")
    ap.add_argument("--charge", default="3D_samples/Charges/Neo_1.0_E10-1000.txt")
    ap.add_argument("--increments", type=int, default=3)
    ap.add_argument("--out", default="Results/test_c5c6")
    args = ap.parse_args()

    test_reference_average()
    test_forcing()
    if args.quick:
        print("\n--quick: solver-level tests skipped")
        return
    test_solver(args.out, args.structure, args.charge, args.N, [0.1]*args.increments)
    print("\nALL C5/C6 TESTS PASSED")


if __name__ == "__main__":
    main()
