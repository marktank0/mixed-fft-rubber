# -*- coding: utf-8 -*-
"""Scan the solver outputs and write CNN/dataset_index.csv.

For every ``<case>_voxel_output/output.csv`` this reads the F11-P11 curve,
checks that it reached ``config.TARGET_F11``, linearly interpolates the label
at that stretch and pairs it with the matching structure ``.npz``.

Doing this once up front keeps training I/O to one small CSV plus the voxel
files, instead of re-parsing 120 solver logs every epoch.

    python CNN/build_dataset.py
"""

import argparse
import csv
import os

import numpy as np

import config


def case_names(results_dir):
    """Case names (without the ``_voxel_output`` suffix), sorted."""
    suffix = "_voxel_output"
    names = [d[: -len(suffix)] for d in os.listdir(results_dir)
             if d.endswith(suffix) and
             os.path.isdir(os.path.join(results_dir, d))]
    return sorted(names)


def read_curve(output_csv):
    """Return the (F11, target) curve of one solver run.

    ``output.csv`` is written with ~3 significant digits, which is plenty for
    interpolating between increments of 0.025 in F11.
    """
    table = np.genfromtxt(output_csv, delimiter=",", names=True)
    f11 = np.atleast_1d(table["F11"]).astype(float)
    target = np.atleast_1d(table[config.TARGET_COLUMN]).astype(float)
    return f11, target


def interpolate_at(f11, values, f_target):
    """Linear interpolation of `values` at ``F11 = f_target``.

    The solver steps F11 monotonically upwards, so a plain ``np.interp`` is
    well defined; the caller guarantees ``f11.max() >= f_target``.
    """
    return float(np.interp(f_target, f11, values))


def read_metadata(case_dir):
    """Filler volume fraction and solver status, if run_metadata.txt exists.

    Roughly half of the runs were cut short by the walltime and never got a
    metadata file written, so both fields are best-effort only.
    """
    path = os.path.join(case_dir, "run_metadata.txt")
    fraction, status = "", ""
    if not os.path.exists(path):
        return fraction, status
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Filler volume fraction:"):
                fraction = line.split(":", 1)[1].strip()
            elif line.startswith("Solver status:"):
                status = line.split(":", 1)[1].strip()
    return fraction, status


def read_manifest_phr(manifest_csv):
    """Map case name -> phr from the microstructure manifest."""
    phr = {}
    if not os.path.exists(manifest_csv):
        return phr
    with open(manifest_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("name"):
                phr[row["name"]] = row.get("phr", "")
    return phr


def build(results_dir, structure_dir, target_f11, verbose=True):
    """Return (rows, skipped) for every case under `results_dir`."""
    phr_by_name = read_manifest_phr(config.STRUCTURE_MANIFEST)
    rows, skipped = [], []

    for name in case_names(results_dir):
        case_dir = os.path.join(results_dir, name + "_voxel_output")
        output_csv = os.path.join(case_dir, "output.csv")
        structure = os.path.join(structure_dir, name + "_voxel.npz")

        if not os.path.exists(output_csv):
            skipped.append((name, "no output.csv"))
            continue
        if not os.path.exists(structure):
            skipped.append((name, "no structure npz"))
            continue

        f11, values = read_curve(output_csv)
        if f11.size < 2:
            skipped.append((name, "curve too short"))
            continue

        max_f11 = float(f11.max())
        if max_f11 < config.MIN_F11:
            skipped.append((name, "max F11 %.3f < %.3f" % (max_f11, target_f11)))
            continue

        fraction, status = read_metadata(case_dir)
        rows.append({
            "name": name,
            "structure_npz": os.path.relpath(structure, config.PROJECT_ROOT),
            "target_f11": "%.4f" % target_f11,
            "target": "%.6e" % interpolate_at(f11, values, target_f11),
            "max_f11": "%.4f" % max_f11,
            "n_increments": str(f11.size),
            "phr": phr_by_name.get(name, ""),
            "filler_fraction": fraction,
            "solver_status": status,
        })

    if verbose:
        print("kept %d cases, skipped %d" % (len(rows), len(skipped)))
        for name, reason in skipped:
            print("  skipped %-28s %s" % (name, reason))
    return rows, skipped


def write_index(rows, path):
    fields = ["name", "structure_npz", "target_f11", "target", "max_f11",
              "n_increments", "phr", "filler_fraction", "solver_status"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    parser.add_argument("--structure-dir", default=config.STRUCTURE_DIR)
    parser.add_argument("--target-f11", type=float, default=config.TARGET_F11)
    parser.add_argument("--out", default=config.INDEX_CSV)
    args = parser.parse_args()

    rows, _ = build(args.results_dir, args.structure_dir, args.target_f11)
    if not rows:
        raise SystemExit("no usable cases found under %s" % args.results_dir)

    write_index(rows, args.out)
    targets = np.array([float(r["target"]) for r in rows])
    print("wrote %s" % args.out)
    print("%s at F11 = %.2f:  mean %.4f  std %.4f  min %.4f  max %.4f"
          % (config.TARGET_COLUMN, args.target_f11, targets.mean(),
             targets.std(), targets.min(), targets.max()))


if __name__ == "__main__":
    main()
