# -*- coding: utf-8 -*-
"""Plot tensile strength against structure PHR, for one or several result folders.

x = PHR, y = tensile strength; one colour per input folder, so a single folder
gives one contrast and several folders compare contrasts on the same axes.

WHAT "TENSILE STRENGTH" MEANS HERE
----------------------------------
``output.csv`` stores P11, the volume-averaged *first Piola-Kirchhoff* stress --
force per **undeformed** area. That is the quantity an experimental tensile
curve plots, so the engineering stress needs no conversion; ``--true-stress``
switches to the Cauchy stress sigma_11 = J^-1 (P F^T)_11 instead.

There is no failure stress in this data, for two independent reasons:

* the material is an incompressible neo-Hookean with no damage or failure model,
  and the P11 curves rise monotonically -- there is no stress peak to call a
  strength;
* the runs stop wherever the solver gave up (or the job was killed), at very
  different stretches, so "the stress at the last increment" would measure the
  solver, not the structure.

So strength is taken as **the nominal stress at one fixed reference stretch,
the same for every run and every folder** -- a "stress at X % strain" measure,
which is what these curves support. The default is 25 % strain
(F11 = 1.25); ``--strain`` sets any other level, ``--strain common`` uses the
largest stretch every plotted run reached, and ``--strain last`` takes each
run's own final increment (solver-limited: reported, and flagged in the title,
for comparison only).

Runs that stopped before the reference stretch are listed and left out rather
than being silently dropped.

Usage:
    python Plotting_scripts/plot_tensile_strength_vs_phr.py
    python Plotting_scripts/plot_tensile_strength_vs_phr.py <dir> --strain 0.4
    python Plotting_scripts/plot_tensile_strength_vs_phr.py <dir1> <dir2> --fit
    python Plotting_scripts/plot_tensile_strength_vs_phr.py <dir> --true-stress --show
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from contrast_sweep_data import load_sweep, results_path, stress_at_strain

DEFAULT_STRAIN = 0.25
DEFAULT_DIR = results_path("HPC", "Results", "cases120_63voxel_10contrast")

# One (colour, marker) pair per folder; both differ so the series stay readable
# in greyscale print too.
_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#7f7f7f",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def series_label(directory, runs, explicit):
    """Legend entry: the contrast when the runs agree on one, else the folder."""
    if explicit:
        return explicit
    contrasts = {run["contrast"] for run in runs}
    if len(contrasts) == 1:
        return r"$E_f/E_m$ = {:g}".format(contrasts.pop())
    return os.path.basename(os.path.normpath(directory))


def load_series(args):
    """[{label, dir, runs}] in the order the folders were given."""
    labels = args.labels or []
    if labels and len(labels) != len(args.result_dirs):
        raise SystemExit("--labels needs one label per folder ({} given, {} folders)"
                         .format(len(labels), len(args.result_dirs)))
    series = []
    for index, directory in enumerate(args.result_dirs):
        runs = load_sweep(directory)
        series.append({
            "label": series_label(directory, runs, labels[index] if labels else None),
            "dir": directory,
            "runs": runs,
        })
    return series


def resolve_strain(series, requested):
    """The reference stretch: a number, 'common', or 'last' (per-run, no target)."""
    if requested == "last":
        return None
    if requested != "common":
        return float(requested)
    strain = min(run["max_strain"] for entry in series for run in entry["runs"])
    print("common strain across every folder: {:.4f}".format(strain))
    return strain


def collect(series, strain, key):
    """Fill each entry with points [(phr, strength)] and the runs left out."""
    for entry in series:
        points, skipped = [], []
        for run in entry["runs"]:
            if strain is None:  # --strain last: each run's own final increment
                stress = run[key]
                value = None if stress is None else float(stress[-1])
            else:
                value = stress_at_strain(run, strain, key)
            if value is None:
                skipped.append(run)
            else:
                points.append((run["phr"], value, run["complete"]))
        entry["points"] = sorted(points)
        entry["skipped"] = skipped
    return series


def make_plot(series, strain, args):
    stress_label = (r"true stress  $\sigma_{11}$  [MPa]" if args.true_stress
                    else r"nominal stress  $P_{11}$  [MPa]")

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for index, entry in enumerate(series):
        if not entry["points"]:
            continue
        color = _COLORS[index % len(_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        phr = [point[0] for point in entry["points"]]
        strength = [point[1] for point in entry["points"]]
        ax.scatter(
            phr, strength, s=args.marker_size, marker=marker,
            facecolors=[color if point[2] else "none" for point in entry["points"]],
            # No alpha= here: it would override the transparent "none" facecolor
            # of a hollow marker and fill it in black.
            edgecolors=color, linewidths=1.2, zorder=3,
            label=entry["label"],
        )
        if args.fit and len(phr) > 1:
            # A least-squares line through a cloud of ~120 structures: the point
            # is the trend with PHR, not any individual microstructure.
            slope, intercept = np.polyfit(phr, strength, 1)
            span = np.array([min(phr), max(phr)])
            ax.plot(span, slope * span + intercept, color=color, linewidth=1.2,
                    alpha=0.6, zorder=2)
            print("{}: fit  strength = {:.4f} * phr + {:.4f}".format(
                entry["label"], slope, intercept))

    ax.set_xlabel("PHR")
    ax.set_ylabel(stress_label)
    if strain is None:
        title = ("Stress at the last increment reached vs PHR "
                 "(solver-limited, not a strength)")
    else:
        title = "Tensile strength vs PHR at strain {:g} ($F_{{11}}$ = {:g})".format(
            strain, 1.0 + strain)
    ax.set_title(args.title or title)
    ax.grid(True, alpha=0.3)

    legend = ax.legend(title="contrast", loc="center left", bbox_to_anchor=(1.01, 0.5),
                       fontsize=9, title_fontsize=9, frameon=False)
    legend._legend_box.align = "left"
    fig.text(0.5, -0.02, "filled marker = solver completed the load path, "
                         "hollow = stopped early", ha="center", fontsize=8, color="0.35")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("saved {}".format(args.out))
    if args.show:
        plt.show()
    plt.close(fig)


def print_summary(series, strain, key):
    measure = "sigma11" if key == "sigma11" else "P11"
    where = ("each run's last increment" if strain is None
             else "F11 = {:g}".format(1.0 + strain))
    print("\n{} at {}".format(measure, where))
    header = "{:<28}{:>7}{:>10}{:>10}{:>10}{:>10}".format(
        "series", "n", "min", "median", "max", "skipped")
    print(header)
    print("-" * len(header))
    for entry in series:
        values = np.array([point[1] for point in entry["points"]])
        if values.size == 0:
            print("{:<28}{:>7}{:>10}{:>10}{:>10}{:>10}".format(
                entry["label"], 0, "-", "-", "-", len(entry["skipped"])))
            continue
        print("{:<28}{:>7}{:>10.3f}{:>10.3f}{:>10.3f}{:>10}".format(
            entry["label"], values.size, values.min(), float(np.median(values)),
            values.max(), len(entry["skipped"])))

    for entry in series:
        if not entry["skipped"]:
            continue
        print("\nexcluded from {}, stopped before the reference stretch:".format(
            entry["label"]))
        for run in entry["skipped"]:
            print("  phr {:>6.2f}  reached F11 = {:.4f}  ({})".format(
                run["phr"], 1.0 + run["max_strain"], run["status"]))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot tensile strength vs PHR for one or more result folders.",
        epilog="Strength is the nominal stress P11 at a fixed reference stretch; "
               "see the module docstring for why the last increment is not one.",
    )
    parser.add_argument("result_dirs", nargs="*", default=[DEFAULT_DIR], metavar="DIR",
                        help="One or more result folders, each plotted in its own "
                             "colour (default: %(default)s).")
    parser.add_argument("--strain", default=str(DEFAULT_STRAIN),
                        help="Reference strain F11 - 1 (default: %(default)s), or "
                             "'common' for the largest strain every run reached, or "
                             "'last' for each run's own final increment.")
    parser.add_argument("--true-stress", action="store_true",
                        help="Use the Cauchy stress sigma_11 instead of nominal P11.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend entry per folder; defaults to its contrast.")
    parser.add_argument("--fit", action="store_true",
                        help="Draw a least-squares line through each folder's points.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: tensile_strength_vs_phr.png in "
                             "the first folder).")
    parser.add_argument("--show", action="store_true", help="Also open the figure window.")
    parser.add_argument("--marker-size", type=float, default=45.0)
    parser.add_argument("--figsize", nargs=2, type=float, default=[9.0, 6.0],
                        metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    key = "sigma11" if args.true_stress else "p11"
    series = load_series(args)
    strain = resolve_strain(series, args.strain)
    collect(series, strain, key)
    if not any(entry["points"] for entry in series):
        raise SystemExit("No run reached the reference stretch")

    suffix = "" if args.strain in ("common", "last") else "_strain{:g}".format(strain)
    if args.true_stress:
        suffix = "_true" + suffix
    args.out = args.out or os.path.join(
        args.result_dirs[0], "tensile_strength_vs_phr{}.png".format(suffix))
    print_summary(series, strain, key)
    make_plot(series, strain, args)


if __name__ == "__main__":
    main()
