# -*- coding: utf-8 -*-
"""Compare solver settings on one and the same set of structures.

Every sub-directory of the test directory is one solver setting, run over the
identical structures, so the only thing that changes between the series is the
solver configuration. Two figures come out of it:

  * max F11 vs PHR   - how far up the load path each setting managed to get
  * cost per load step vs PHR - what that progress cost per unit of load path

Points are coloured per solver setting.

Raw wall-clock time is not comparable between runs here: nearly every job was
cut off by the queue rather than by convergence, so the total mostly measures
the wall-clock limit. The second figure therefore normalises by how far the run
actually got, in units of the nominal 0.025 load step:

    steps = (max F11 - 1) / 0.025
    cost  = total solve time / steps

so a setting that crawls up the load path scores badly even when it burnt the
same wall clock as one that raced up it. Step cuts make the solver's internal
increments smaller than 0.025, and this measure folds that cost in: a run that
had to halve its steps pays for the extra increments without covering extra
load path.

Total solve time comes from ``run_metadata.txt`` ("Run time seconds") where that
file exists. The no_preconditioner runs were written without it, so their total
is the sum of the per-increment ``time_seconds`` in ``solver_stats.json``
instead. Those two are not quite the same quantity - the metadata figure
includes setup and IO, the summed one does not - so ``--time-source stats``
sums the increments for every run, which is apples-to-apples at the cost of
ignoring overhead.

Usage:
    python Plotting_scripts/plot_solver_setting_comparison.py
    python Plotting_scripts/plot_solver_setting_comparison.py <test_dir> --show
    python Plotting_scripts/plot_solver_setting_comparison.py --outdir Plots
    python Plotting_scripts/plot_solver_setting_comparison.py --time-source stats
"""

import argparse
import csv
import json
import os
import re

import matplotlib.pyplot as plt

DEFAULT_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Results", "HPC", "solver_setting_test",
)

# One (colour, marker) pair per solver setting; both differ so the series stay
# readable in greyscale print too.
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]

_PHR_RE = re.compile(r"phr_([0-9]+(?:\.[0-9]+)?)_")

# Nominal load step of the prescribed increment list; the solver may cut below
# it, but it stays the unit the runs are compared in.
NOMINAL_STEP = 0.025


def _max_f11(csv_path):
    """Largest F11 in a run's output.csv, or None if it holds no data rows."""
    best = None
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["F11"])
            except (TypeError, ValueError):
                continue
            if best is None or value > best:
                best = value
    return best


def _metadata_time(metadata_path):
    with open(metadata_path) as handle:
        for line in handle:
            if line.startswith("Run time seconds:"):
                return float(line.split(":", 1)[1])
    return None


def _stats_time(stats_path):
    """Fallback total: the per-increment solve times added up."""
    with open(stats_path) as handle:
        stats = json.load(handle)
    total = 0.0
    for increment in stats.get("increments", []):
        total += float(increment.get("time_seconds") or 0.0)
    return total


def _solve_time(run_dir, time_source):
    if time_source == "stats":
        stats_path = os.path.join(run_dir, "solver_stats.json")
        return _stats_time(stats_path) if os.path.exists(stats_path) else None
    metadata_path = os.path.join(run_dir, "run_metadata.txt")
    if os.path.exists(metadata_path):
        seconds = _metadata_time(metadata_path)
        if seconds is not None:
            return seconds
    stats_path = os.path.join(run_dir, "solver_stats.json")
    if os.path.exists(stats_path):
        return _stats_time(stats_path)
    return None


def load_runs(test_dir, time_source="metadata"):
    """One record per (solver setting, structure) with its PHR, F11 and time.

    ``time_per_step`` is the total time divided by the number of nominal 0.025
    load steps the run covered, and is None for a run that never got anywhere.
    """
    runs = []
    for setting in sorted(os.listdir(test_dir)):
        setting_dir = os.path.join(test_dir, setting)
        if not os.path.isdir(setting_dir):
            continue
        for name in sorted(os.listdir(setting_dir)):
            run_dir = os.path.join(setting_dir, name)
            csv_path = os.path.join(run_dir, "output.csv")
            match = _PHR_RE.match(name)
            if not match or not os.path.exists(csv_path):
                continue
            max_f11 = _max_f11(csv_path)
            if max_f11 is None:
                continue
            seconds = _solve_time(run_dir, time_source)
            steps = (max_f11 - 1.0) / NOMINAL_STEP
            runs.append({
                "setting": setting,
                "phr": float(match.group(1)),
                "max_F11": max_f11,
                "steps": steps,
                "time_seconds": seconds,
                "time_per_step": (
                    seconds / steps if seconds is not None and steps > 0 else None
                ),
            })
    return runs


def _scatter(runs, key, ylabel, title, args, out_name, log_y=False):
    settings = sorted({run["setting"] for run in runs})

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for index, setting in enumerate(settings):
        series = sorted(
            (run for run in runs
             if run["setting"] == setting and run[key] is not None),
            key=lambda run: run["phr"],
        )
        if not series:
            continue
        color = _COLORS[index % len(_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        phr = [run["phr"] for run in series]
        values = [run[key] for run in series]
        if args.lines:
            ax.plot(phr, values, color=color, linewidth=1.0, alpha=0.35, zorder=1)
        ax.scatter(
            phr, values, s=args.marker_size, marker=marker, color=color,
            edgecolors=color, linewidths=1.4, zorder=3, label=setting,
        )

    ax.set_xlabel("PHR")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Solver setting", fontsize=8, title_fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(args.outdir, out_name)
    fig.savefig(out_path, dpi=args.dpi)
    print("wrote {}".format(out_path))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_dir", nargs="?", default=DEFAULT_TEST_DIR)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--figsize", nargs=2, type=float, default=[8.0, 5.0])
    parser.add_argument("--marker-size", type=float, default=45.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lines", action="store_true",
                        help="connect the points of each setting with a line")
    parser.add_argument("--log-time", action="store_true",
                        help="log scale on the solve-time axis")
    parser.add_argument("--time-source", choices=["metadata", "stats"],
                        default="metadata",
                        help="'metadata' uses run_metadata.txt where present "
                             "(summed solver_stats.json otherwise); 'stats' "
                             "sums solver_stats.json for every run")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    runs = load_runs(args.test_dir, args.time_source)
    if not runs:
        raise SystemExit("no runs found under {}".format(args.test_dir))
    if args.outdir and not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    _scatter(
        runs, "max_F11", r"max $F_{11}$ reached",
        "Load path reached per solver setting", args,
        "solver_setting_max_F11_vs_phr.png",
    )
    _scatter(
        runs, "time_per_step",
        "solve time per {:g} load step [s]".format(NOMINAL_STEP),
        "Cost per unit of load path per solver setting", args,
        "solver_setting_time_per_step_vs_phr.png", log_y=args.log_time,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
