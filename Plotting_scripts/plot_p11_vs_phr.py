# -*- coding: utf-8 -*-
"""Plot P11 stress versus structure PHR for a batch of FFT simulations.

For every ``phr_*_output`` folder in a results directory this reads
``output.csv`` and extracts the P11 stress at a set of target F11 strains
(F11 deformation gradient = 1 + strain). It then plots P11 (y) against the
structure's PHR (x), using a different marker for each strain level.

Runs whose solver failed are included by default: their curve up to the last
increment they managed is still valid, and a failed run that got past the
requested strain has a perfectly good P11 there. Pass ``--ignore-failed`` to
leave them out.

Usage:
    python plot_p11_vs_phr.py
    python plot_p11_vs_phr.py Results/30_struct_test_run
    python plot_p11_vs_phr.py <results_dir> --strains 0.3 0.6 1.0 --out plot.png
    python plot_p11_vs_phr.py <results_dir> --strains 0.3 0.6 1.0 --out plot.png --ignore-failed      # To ignore the failed runs
"""

import argparse
import csv
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = os.path.join("Results", "30_struct_test_run")
DEFAULT_STRAINS = [0.3, 0.6, 1.0]
# Markers cycled per strain level (extended if more strains are requested).
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# PHR is embedded in the folder name, e.g. "phr_11.86_i180_sample6_voxel_output".
_PHR_RE = re.compile(r"phr_(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_phr(folder_name):
    match = _PHR_RE.search(folder_name)
    return float(match.group(1)) if match else None


def solver_failed(folder):
    """True if solver_stats.json marks this run as failed."""
    stats_path = os.path.join(folder, "solver_stats.json")
    if not os.path.isfile(stats_path):
        return False
    with open(stats_path) as file:
        stats = json.load(file)
    return stats.get("status") == "failed"


def load_f11_p11(csv_path):
    """Return (F11 array, P11 array) from an output.csv."""
    with open(csv_path, newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        idx = {name.strip(): i for i, name in enumerate(header)}
        if "F11" not in idx or "P11" not in idx:
            raise ValueError("output.csv missing F11/P11 columns: {}".format(csv_path))
        f11, p11 = [], []
        for row in reader:
            if not row:
                continue
            f11.append(float(row[idx["F11"]]))
            p11.append(float(row[idx["P11"]]))
    return np.asarray(f11), np.asarray(p11)


def p11_at_strain(f11, p11, strain, tol=0.05):
    """P11 at the row whose F11 is nearest to (1 + strain); None if none close."""
    target = 1.0 + strain
    i = int(np.argmin(np.abs(f11 - target)))
    if abs(f11[i] - target) > tol:
        return None
    return float(p11[i])


def collect(results_dir, strains, ignore_failed):
    """Gather {strain: [(phr, P11), ...]} across all result folders."""
    data = {strain: [] for strain in strains}
    folders = sorted(
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
    )
    used = 0
    for name in folders:
        csv_path = os.path.join(results_dir, name, "output.csv")
        if not os.path.isfile(csv_path):
            continue
        if ignore_failed:
            if solver_failed(os.path.join(results_dir, name)):
                print("skip (solver failed): {}".format(name))
                continue
        phr = parse_phr(name)
        if phr is None:
            print("skip (no phr in name): {}".format(name))
            continue
        f11, p11 = load_f11_p11(csv_path)
        used += 1
        for strain in strains:
            value = p11_at_strain(f11, p11, strain)
            if value is None:
                print("warn: no F11~{:.2f} in {}".format(1.0 + strain, name))
            else:
                data[strain].append((phr, value))
    print("read {} simulation(s) from {}".format(used, results_dir))
    return data


def make_plot(data, strains, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, strain in enumerate(strains):
        points = sorted(data[strain])
        if not points:
            continue
        phr = [p[0] for p in points]
        p11 = [p[1] for p in points]
        ax.scatter(
            phr, p11,
            marker=_MARKERS[i % len(_MARKERS)],
            s=55,
            edgecolors="black",
            linewidths=0.5,
            label="F11 = {:g}  (F11 grad = {:g})".format(strain, 1.0 + strain),
        )

    ax.set_xlabel("PHR")
    ax.set_ylabel(r"$P_{11}$ stress")
    ax.set_title("P11 vs PHR at fixed F11 strains")
    ax.grid(True, alpha=0.3)
    ax.legend(title="F11 strain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print("saved plot to {}".format(out_path))


def main():
    parser = argparse.ArgumentParser(description="Plot P11 vs PHR at fixed F11 strains.")
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Folder containing the phr_*_output simulation folders.")
    parser.add_argument("--strains", nargs="+", type=float, default=DEFAULT_STRAINS,
                        help="F11 strain levels (deformation gradient = 1 + strain).")
    parser.add_argument("--out", default=None, help="Output image path (PNG).")
    # Both flags share a dest, so both carry the default: argparse applies them
    # in order and the later action would otherwise overwrite the earlier one.
    parser.add_argument("--ignore-failed", dest="ignore_failed", action="store_true",
                        default=False,
                        help="Skip runs whose solver_stats.json says failed.")
    parser.add_argument("--include-failed", dest="ignore_failed", action="store_false",
                        default=False,
                        help="Also plot runs whose solver_stats.json says failed "
                             "(default).")
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.results_dir, "P11_vs_phr.png")
    data = collect(args.results_dir, args.strains, args.ignore_failed)
    make_plot(data, args.strains, out_path)


if __name__ == "__main__":
    main()
