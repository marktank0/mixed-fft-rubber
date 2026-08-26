# -*- coding: utf-8 -*-
"""Benchmark the spectral (Moulinec-Suquet) vs Willot rotated discretization.

Reports cost (Krylov iterations, wall time) and two accuracy diagnostics
across a filler/matrix contrast ladder:

  filler strain   ||F - I|| averaged over filler voxels. As contrast -> inf
                  the filler becomes rigid, so the true value tends to zero;
                  a smaller value is a more accurate discrete solution.
  field extremes  min/max of local F11. Gibbs ringing from the truncated
                  Fourier basis shows up as over/undershoot at interfaces.

Runnable from any working directory:

    FFT_WORKERS=2 python3 benchmark_discretization.py

See docs/discretization.md.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np

from project_paths import CHARGES_DIR, ensure_import_paths, results_path, VOXELS_DIR

ensure_import_paths()

import fg.mxfft as mx

CHARGE_TEMPLATE = """#first two lines: model:---0)model num 1) p1.. 2)p2...
1.0\t10\t0.48\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
1.0\t{E:g}\t0.30\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
#charge dF
1.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
#(Charge type) P-1 or F-0: 0 = control this component by Fij, 1 = control this component by average Pij
0.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0
"""

LADDER = [("10", 100.0), ("100", 1000.0), ("500", 5000.0), ("1000", 10000.0)]

_captured = {}
_orig_solution_fields = mx.solution_fields


def _spy(F, P, phase, pressure=None):
    _captured["F"] = np.array(F, copy=True)
    return _orig_solution_fields(F, P, phase, pressure=pressure)


mx.solution_fields = _spy


def ensure_charge(label, E, charge_dir):
    path = os.path.join(charge_dir, "bench_c{}.txt".format(label))
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(CHARGE_TEMPLATE.format(E=E))
    return path


def run(tag, structure, charge, N, incre, out_dir, mask_f, **kw):
    path = os.path.join(out_dir, tag)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    prob = mx.FFTSolver(structure, charge_path=charge, output_path=path, N=N, output_name=".")
    t0 = time.time()
    try:
        prob.calculate(incre_list=incre, savemodel="normal", preconditioner="reference",
                       reference="mean", forcing="eisenstat_walker",
                       save_fields=True, **kw)
    except Exception as exc:
        return {"status": "ERROR:" + type(exc).__name__, "wall": time.time()-t0,
                "krylov": None, "newton": None}
    wall = time.time() - t0
    st = json.load(open(os.path.join(prob.output_path, "solver_stats.json")))
    kry = [k for i in st["increments"] for k in i["krylov_iterations"]]

    F = _captured["F"]
    eye = np.eye(3)[:, :, None, None, None]
    dev = np.sqrt(((F - eye)**2).sum(axis=(0, 1)))         # ||F - I||_F per voxel
    return {
        "status": st["status"], "wall": wall, "krylov": sum(kry),
        "newton": sum(i["newton_iterations"] for i in st["increments"]),
        "P11": float(np.array(prob.Ps)[-1][0, 0]),
        "filler_strain": float(dev[mask_f].mean()),
        "F11_min": float(F[0, 0].min()), "F11_max": float(F[0, 0].max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=31)
    ap.add_argument("--structure", default=os.path.join(VOXELS_DIR, "1_voxel.npz"))
    ap.add_argument("--charge-dir", default=CHARGES_DIR)
    ap.add_argument("--increments", type=int, default=3)
    ap.add_argument("--out", default=results_path("bench_discretization"))
    args = ap.parse_args()

    phase = np.load(args.structure)["phase"]
    mask_f = phase == 1
    incre = [0.1]*args.increments
    print("structure {}  N={}  filler fraction {:.3f}".format(
        os.path.basename(args.structure), args.N, phase.mean()))

    rows = {}
    for label, E in LADDER:
        charge = ensure_charge(label, E, args.charge_dir)
        for disc in ("fourier", "willot"):
            print("\n>>> contrast {} | {}".format(label, disc), flush=True)
            rows[(label, disc)] = run("c{}_{}".format(label, disc), args.structure,
                                      charge, args.N, incre, args.out, mask_f,
                                      discretization=disc)

    print("\n\n" + "=" * 100)
    print("DISCRETIZATION BENCHMARK  N={}  {} increments  (both runs use C5/EW + reference=mean)"
          .format(args.N, args.increments))
    print("=" * 100)
    print("{:<10}{:<10}{:>9}{:>8}{:>9}{:>12}{:>15}{:>11}{:>11}".format(
        "contrast", "scheme", "krylov", "newton", "wall(s)", "P11",
        "filler strain", "F11 min", "F11 max"))
    print("-" * 100)
    for label, _E in LADDER:
        for disc in ("fourier", "willot"):
            r = rows[(label, disc)]
            if r["krylov"] is None:
                print("{:<10}{:<10}{:>9}{:>8}{:>9.1f}{:>12}".format(
                    label, disc, "-", "-", r["wall"], r["status"][:11]))
                continue
            print("{:<10}{:<10}{:>9}{:>8}{:>9.1f}{:>12.6f}{:>15.6f}{:>11.6f}{:>11.6f}".format(
                label, disc, r["krylov"], r["newton"], r["wall"], r["P11"],
                r["filler_strain"], r["F11_min"], r["F11_max"]))
        print("-" * 100)
    print("=" * 100)

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "benchmark.json")
    json.dump({"{}|{}".format(k[0], k[1]): v for k, v in rows.items()}, open(dest, "w"), indent=1)
    print("raw results:", dest)


if __name__ == "__main__":
    main()
