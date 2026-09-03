# -*- coding: utf-8 -*-
"""Contrast-ladder benchmark for C5 (inexact Newton) and C6 (reference tangent).

Compares the pre-change solver against Eisenstat-Walker forcing terms and the
three reference-tangent modes, across a ladder of filler/matrix stiffness
contrasts. Reports total Krylov iterations, Newton iterations and wall time.

Runnable from any working directory:

    FFT_WORKERS=2 python3 benchmark_c5c6.py [--N 31] [--out DIR]

The pre-change solver is materialised from git as
`FFT_simulation/fg/_mxfft_baseline.py`
(gitignored) so the comparison is against real committed code, not a
reimplementation. See docs/inexact_newton_and_reference_tangent.md.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from project_paths import CHARGES_DIR, ensure_import_paths, FG_DIR, PROJECT_ROOT, results_path, VOXELS_DIR

ensure_import_paths()

BASELINE_MODULE = os.path.join(FG_DIR, "_mxfft_baseline.py")
BASELINE_REV = "HEAD"

CHARGE_TEMPLATE = """#first two lines: model:---0)model num 1) p1.. 2)p2...
1.0\t10\t0.48\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
1.0\t{E:g}\t0.30\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
#charge dF
1.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
#(Charge type) P-1 or F-0: 0 = control this component by Fij, 1 = control this component by average Pij
0.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0
"""

# (label, filler Young's modulus); matrix is fixed at E = 10
LADDER = [("10", 100.0), ("100", 1000.0), ("500", 5000.0), ("1000", 10000.0)]


def _solver_source(rev):
    """`git show` the mixed solver out of `rev`, wherever fg/ lived back then.

    The solver moved to FFT_simulation/fg/ partway through the project's
    history, so the path is a property of the commit being read, not of the
    current tree.
    """
    tried = ("FFT_simulation/fg/mxfft.py", "fg/mxfft.py")
    for path in tried:
        try:
            src = subprocess.check_output(
                ["git", "show", "{}:{}".format(rev, path)],
                cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        return src, path
    raise RuntimeError("no mixed solver in {} at any of {}".format(rev, tried))


def ensure_baseline(rev=BASELINE_REV):
    """Materialise the pre-change solver from git so we compare against real code."""
    if os.path.exists(BASELINE_MODULE):
        return
    src, in_commit = _solver_source(rev)
    with open(BASELINE_MODULE, "wb") as fh:
        fh.write(src)
    print("materialised baseline solver from {}:{}".format(rev, in_commit))


def ensure_charge(contrast_label, E, charge_dir):
    path = os.path.join(charge_dir, "bench_c{}.txt".format(contrast_label))
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(CHARGE_TEMPLATE.format(E=E))
    return path


def run(tag, module, structure, charge, N, incre, out_dir, **kw):
    path = os.path.join(out_dir, tag)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    prob = module.FFTSolver(structure, charge_path=charge, output_path=path,
                            N=N, output_name=".")
    # the pinned baseline solver predates both changes: it still calls the
    # Green preconditioner "reference", and still needs savemodel to write
    # output.csv (the live solver always saves)
    base = "_baseline" in module.__name__
    precond = "reference" if base else "green"
    if base:
        kw = dict(kw, savemodel="normal")
    t0 = time.time()
    try:
        prob.calculate(incre_list=incre, preconditioner=precond, **kw)
    except Exception as exc:                       # keep the sweep going
        return {"status": "ERROR:" + type(exc).__name__, "wall": time.time()-t0,
                "krylov": None, "newton": None, "P11": None, "cuts": None}
    wall = time.time() - t0
    st = json.load(open(os.path.join(prob.output_path, "solver_stats.json")))
    kry = [k for inc in st["increments"] for k in inc["krylov_iterations"]]
    return {
        "status": st["status"], "wall": wall, "krylov": sum(kry),
        "newton": sum(i["newton_iterations"] for i in st["increments"]),
        "P11": float(np.array(prob.Ps)[-1][0, 0]) if prob.Ps else None,
        "cuts": st["step_cuts"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=31)
    ap.add_argument("--structure", default=os.path.join(VOXELS_DIR, "1_voxel.npz"))
    ap.add_argument("--charge-dir", default=CHARGES_DIR)
    ap.add_argument("--increments", type=int, default=3)
    ap.add_argument("--out", default=results_path("bench_c5c6"))
    ap.add_argument("--contrasts", nargs="*", default=[c for c, _ in LADDER])
    args = ap.parse_args()

    ensure_baseline()
    sys.path.insert(0, os.getcwd())
    import fg.mxfft as new
    import fg._mxfft_baseline as base

    incre = [0.1]*args.increments
    ladder = [(c, E) for c, E in LADDER if c in args.contrasts]

    configs = [
        ("baseline",       base, {}),
        ("C5 (EW), mean",  new, dict(forcing="eisenstat_walker", reference="mean")),
        ("C5+C6, matrix",  new, dict(forcing="eisenstat_walker", reference="matrix")),
        ("C5+C6, mid",     new, dict(forcing="eisenstat_walker", reference="mid")),
        ("C6 only, matrix", new, dict(forcing="fixed", inner_rtol=1e-6, reference="matrix")),
        ("C6 only, mid",   new, dict(forcing="fixed", inner_rtol=1e-6, reference="mid")),
    ]

    rows = {}
    for cname, E in ladder:
        charge = ensure_charge(cname, E, args.charge_dir)
        for label, mod, kw in configs:
            tag = "c{}_{}".format(cname, label.replace(" ", "").replace(",", "_").replace("+", ""))
            print("\n>>> contrast {} | {}".format(cname, label), flush=True)
            rows[(cname, label)] = run(tag, mod, args.structure, charge,
                                       args.N, incre, args.out, **kw)

    print("\n\n" + "=" * 92)
    print("C5/C6 BENCHMARK   structure={}  N={}  {} increments of 0.1"
          .format(os.path.basename(args.structure), args.N, args.increments))
    print("=" * 92)
    for cname, _E in ladder:
        base_row = rows[(cname, "baseline")]
        print("\ncontrast {}".format(cname))
        print("{:<18}{:>9}{:>9}{:>10}{:>10}{:>14}".format(
            "config", "krylov", "newton", "wall(s)", "speedup", "P11"))
        print("-" * 72)
        for label, _m, _k in configs:
            r = rows[(cname, label)]
            if r["krylov"] is None:
                print("{:<18}{:>9}{:>9}{:>10.1f}{:>10}{:>14}".format(
                    label, "-", "-", r["wall"], "-", r["status"][:13]))
                continue
            sp = base_row["krylov"]/r["krylov"] if base_row["krylov"] else float("nan")
            note = "" if r["cuts"] == 0 else "  cuts={}".format(r["cuts"])
            print("{:<18}{:>9}{:>9}{:>10.1f}{:>9.2f}x{:>14.6f}{}".format(
                label, r["krylov"], r["newton"], r["wall"], sp, r["P11"], note))
    print("=" * 92)

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "benchmark.json")
    json.dump({"{}|{}".format(k[0], k[1]): v for k, v in rows.items()},
              open(dest, "w"), indent=1)
    print("raw results:", dest)


if __name__ == "__main__":
    main()
