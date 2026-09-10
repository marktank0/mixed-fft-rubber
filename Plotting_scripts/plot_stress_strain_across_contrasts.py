# -*- coding: utf-8 -*-
"""Plot one structure's stress-strain curve at every contrast of a sweep.

Point the script at a single run folder -- e.g.
``Results/HPC/Results/contrast_sweep/E10-500/phr_25.29_id22_voxel_output`` --
and it finds the same structure in every other contrast folder of that sweep
and draws all of their P11-vs-F11 curves in one figure, one colour per
filler/matrix contrast.

The curves are the raw solver output: P11 is the volume-averaged first
Piola-Kirchhoff (nominal) stress, F11 the prescribed stretch, and the
undeformed state (F11 = 1, P11 = 0) is prepended so every curve starts at the
origin. ``--true-stress`` plots the Cauchy stress sigma_11 = J^-1 (P F^T)_11
instead.

Each curve simply ends where its run did. A run that completed the load path
gets a filled end marker, one that stopped early a hollow one -- at high
contrast the solver gives up long before the end, and the curve length is the
solver's limit rather than the material's.

Usage:
    python Plotting_scripts/plot_stress_strain_across_contrasts.py
    python Plotting_scripts/plot_stress_strain_across_contrasts.py <run_dir> --show
    python Plotting_scripts/plot_stress_strain_across_contrasts.py <sweep_dir> \
        --structure phr_25.29_id22_voxel_output
    python Plotting_scripts/plot_stress_strain_across_contrasts.py <run_dir> --matrix
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from contrast_sweep_data import (
    contrast_from_dirname, load_run, matrix_p11, parse_phr, results_path,
)

DEFAULT_RUN_DIR = results_path("HPC", "Results", "contrast_sweep", "E10-500",
                               "phr_25.29_id22_voxel_output")

# One (colour, marker) pair per contrast; both differ so the curves stay
# readable in greyscale print too.
_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#7f7f7f",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def resolve_target(path, structure):
    """(sweep_dir, structure folder name) from a run folder or a sweep folder."""
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        raise SystemExit("No such directory: {}".format(path))
    if structure:
        return path, structure
    # A run folder sits at <sweep>/<contrast>/<structure>; a sweep folder needs
    # --structure to say which of its structures to follow.
    if parse_phr(os.path.basename(path)) is None:
        raise SystemExit(
            "{} is not a run folder; pass --structure to name one inside it".format(path)
        )
    return os.path.dirname(os.path.dirname(path)), os.path.basename(path)


def load_curves(sweep_dir, structure):
    """One loaded run per contrast folder that holds this structure."""
    contrast_dirs = sorted(
        name for name in os.listdir(sweep_dir)
        if contrast_from_dirname(name) is not None
        and os.path.isdir(os.path.join(sweep_dir, name))
    )
    if not contrast_dirs:
        raise SystemExit("No contrast folders (E<matrix>-<filler>) in {}".format(sweep_dir))

    runs = []
    for name in contrast_dirs:
        run_dir = os.path.join(sweep_dir, name, structure)
        if not os.path.isdir(run_dir):
            print("skip (structure absent): {}".format(name))
            continue
        run = load_run(run_dir)
        if run is None:
            print("skip (no output.csv): {}".format(name))
            continue
        # A single run folder has no siblings to inherit from, so a run whose
        # metadata is missing falls back to the contrast in the folder name.
        if run["contrast"] is None:
            run["contrast"] = contrast_from_dirname(name)
        runs.append(run)

    if not runs:
        raise SystemExit("{} not found in any contrast folder of {}".format(
            structure, sweep_dir))
    runs.sort(key=lambda run: run["contrast"])
    print("read {} contrast(s) for {}".format(len(runs), structure))
    return runs


def curve(run, key):
    """(F11, stress) starting from the undeformed state."""
    stress = run[key]
    if stress is None:
        return None, None
    return np.concatenate(([1.0], run["f11"])), np.concatenate(([0.0], stress))


def make_plot(runs, structure, args):
    key = "sigma11" if args.true_stress else "p11"
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for index, run in enumerate(runs):
        f11, stress = curve(run, key)
        if f11 is None:
            print("skip (no {} available): contrast {:g}".format(key, run["contrast"]))
            continue
        color = _COLORS[index % len(_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        ax.plot(f11, stress, color=color, linewidth=1.6, zorder=2,
                label=r"$E_f/E_m$ = {:g}".format(run["contrast"]))
        if args.markers:
            ax.plot(f11[1:], stress[1:], linestyle="none", marker=marker,
                    markersize=3.5, color=color, alpha=0.7, zorder=3)
        # The end of the curve is where this run stopped, so say whether that was
        # the end of the load path or the solver giving up.
        ax.plot([f11[-1]], [stress[-1]], marker=marker, markersize=8, color=color,
                markerfacecolor=color if run["complete"] else "none",
                markeredgewidth=1.4, linestyle="none", zorder=4)

    if args.matrix:
        # The neat matrix under the same load case: everything above it is what
        # the filler contributes.
        model, young, poisson = runs[0]["matrix"] or ("1", 10.0, 0.48)
        limit = max(float(run["f11"][-1]) for run in runs)
        grid = np.linspace(1.0, limit, 60)
        reference = [0.0] + [matrix_p11(value, model, young, poisson)
                             for value in grid[1:]]
        ax.plot(grid, reference, color="0.35", linestyle="--", linewidth=1.3,
                zorder=1, label="neat matrix (E = {:g})".format(young))

    ax.set_xlabel(r"stretch  $F_{11}$")
    ax.set_ylabel(r"true stress  $\sigma_{11}$  [MPa]" if args.true_stress
                  else r"nominal stress  $P_{11}$  [MPa]")
    phr = parse_phr(structure)
    subject = "phr {:g}".format(phr) if phr is not None else structure
    ax.set_title(args.title or "Stress-strain of {} across contrasts".format(subject))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1.0)
    ax.set_ylim(bottom=0.0)

    legend = ax.legend(title="contrast", loc="center left", bbox_to_anchor=(1.01, 0.5),
                       fontsize=9, title_fontsize=9, frameon=False)
    legend._legend_box.align = "left"
    fig.text(0.5, -0.02, "curve ends where the run stopped: filled end marker = "
                         "completed the load path, hollow = solver stopped early",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("saved {}".format(args.out))
    if args.show:
        plt.show()
    plt.close(fig)


def print_table(runs, key):
    measure = "sigma11" if key == "sigma11" else "P11"
    header = "{:>10}{:>8}{:>12}{:>14}{:>28}".format(
        "contrast", "steps", "max F11", "max " + measure, "status")
    print()
    print(header)
    print("-" * len(header))
    for run in runs:
        stress = run[key]
        print("{:>10.0f}{:>8}{:>12.4f}{:>14}{:>28}".format(
            run["contrast"], len(run["f11"]), 1.0 + run["max_strain"],
            "-" if stress is None else "{:.4f}".format(float(stress[-1])),
            run["status"]))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot one structure's stress-strain curve at every contrast "
                    "of a sweep."
    )
    parser.add_argument("path", nargs="?", default=DEFAULT_RUN_DIR,
                        help="A run folder inside one contrast of the sweep, or the "
                             "sweep folder together with --structure "
                             "(default: %(default)s).")
    parser.add_argument("--structure", default=None,
                        help="Structure folder name, when `path` is the sweep folder.")
    parser.add_argument("--true-stress", action="store_true",
                        help="Plot the Cauchy stress sigma_11 instead of nominal P11.")
    parser.add_argument("--matrix", action="store_true",
                        help="Also draw the neat-matrix curve as a reference.")
    parser.add_argument("--no-markers", dest="markers", action="store_false", default=True,
                        help="Draw the curves as plain lines, without per-increment dots.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: stress_strain_<structure>.png in "
                             "the sweep folder).")
    parser.add_argument("--show", action="store_true", help="Also open the figure window.")
    parser.add_argument("--figsize", nargs=2, type=float, default=[9.0, 6.0],
                        metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    sweep_dir, structure = resolve_target(args.path, args.structure)
    runs = load_curves(sweep_dir, structure)
    key = "sigma11" if args.true_stress else "p11"
    print_table(runs, key)

    suffix = "_true" if args.true_stress else ""
    args.out = args.out or os.path.join(
        sweep_dir, "stress_strain_{}{}.png".format(structure, suffix))
    make_plot(runs, structure, args)


if __name__ == "__main__":
    main()
