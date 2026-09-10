# -*- coding: utf-8 -*-
"""Scatter the reinforcement of a contrast sweep against the contrast.

One marker per simulation: x = filler/matrix contrast on a log axis,
y = reinforcement, one colour per structure PHR.

Reinforcement is the macroscopic stress of the filled structure divided by the
stress of the neat matrix at the same stretch,

    R(F11) = P11_composite(F11) / P11_matrix(F11),

so R = 1 means "no stiffer than unfilled rubber". The reference is not read
from a file: it is computed from the same constitutive routine the solver used
(``FFT_simulation/fg/constitutive_incompressible/<model>.py``) for the
homogeneous uniaxial state the charge file prescribes, without any filler.

Because every run is compared at one common stretch, the strain has to be low
enough that the runs which stalled early still reach it -- most high-contrast
runs stop well before the end of the load path. ``--strain common`` picks the
largest strain every plotted run actually reached.

Usage:
    python Plotting_scripts/plot_reinforcement_vs_contrast.py
    python Plotting_scripts/plot_reinforcement_vs_contrast.py --strain 0.1
    python Plotting_scripts/plot_reinforcement_vs_contrast.py --strain common --show
    python Plotting_scripts/plot_reinforcement_vs_contrast.py <sweep_dir> --raw-p11
"""

import argparse
import os

import matplotlib.pyplot as plt

from contrast_sweep_data import (
    DEFAULT_SWEEP_DIR, load_sweep, reinforcement_rows, resolve_strain,
)

DEFAULT_STRAIN = 0.05

# One (colour, marker) pair per PHR; both differ so the series stay readable in
# greyscale print too.
_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#b5651d",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def dodge(contrast, index, count, strength):
    """Spread the PHRs of one contrast along the log axis so markers stay visible."""
    if count < 2 or strength <= 0:
        return contrast
    offset = (index - (count - 1) / 2.0) / (count - 1)
    return contrast * (10.0 ** (offset * strength))


def make_plot(rows, strain, args):
    phrs = sorted({row["phr"] for row in rows})
    y_key = "p11" if args.raw_p11 else "reinforcement"

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for index, phr in enumerate(phrs):
        series = sorted(
            (row for row in rows if row["phr"] == phr), key=lambda row: row["contrast"]
        )
        color = _COLORS[index % len(_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        x = [dodge(row["contrast"], index, len(phrs), args.dodge) for row in series]
        y = [row[y_key] for row in series]
        if args.lines:
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.35, zorder=1)
        ax.scatter(
            x, y, s=args.marker_size, marker=marker,
            facecolors=[color if row["complete"] else "none" for row in series],
            edgecolors=color, linewidths=1.4, zorder=3,
            label="{:g}".format(phr),
        )

    if not args.raw_p11:
        low, high = ax.get_ylim()
        # Only annotate R = 1 when it is on the axis; otherwise the label floats
        # underneath the frame.
        if low <= 1.0 <= high:
            ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
            ax.text(0.995, 1.0, " neat matrix", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color="0.4")

    contrasts = sorted({row["contrast"] for row in rows})
    ax.set_xscale("log")
    ax.set_xticks(contrasts)
    ax.set_xticklabels(["{:g}".format(contrast) for contrast in contrasts])
    ax.minorticks_off()
    ax.set_xlabel(r"filler/matrix contrast  $E_f/E_m$")
    ax.set_ylabel(r"$P_{11}$  [MPa]" if args.raw_p11
                  else r"reinforcement  $P_{11}/P_{11}^{\mathrm{matrix}}$")
    ax.set_title(args.title or "Reinforcement vs contrast at strain {:g} "
                               "($F_{{11}}$ = {:g})".format(strain, 1.0 + strain))
    ax.grid(True, which="major", alpha=0.25)

    legend = ax.legend(title="PHR", loc="center left", bbox_to_anchor=(1.01, 0.5),
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


def print_table(rows, skipped, strain):
    model, young, poisson = rows[0]["matrix"]
    print("\nneat-matrix reference: model {}, E = {:g}, nu = {:g}".format(
        model, young, poisson))
    print("P11_matrix(F11 = {:g}) = {:.6f}".format(
        1.0 + strain, rows[0]["matrix_p11"]))

    contrasts = sorted({row["contrast"] for row in rows})
    phrs = sorted({row["phr"] for row in rows})
    lookup = {(row["phr"], row["contrast"]): row for row in rows}

    header = "{:>8}".format("phr") + "".join(
        "{:>12}".format("c{:g}".format(contrast)) for contrast in contrasts
    )
    print("\nreinforcement at strain {:g}".format(strain))
    print(header)
    print("-" * len(header))
    for phr in phrs:
        cells = []
        for contrast in contrasts:
            row = lookup.get((phr, contrast))
            cells.append("{:>12}".format(
                "-" if row is None else "{:.3f}".format(row["reinforcement"])
            ))
        print("{:>8.2f}".format(phr) + "".join(cells))
    if skipped:
        print("\nexcluded, stopped before strain {:g}:".format(strain))
        for run in skipped:
            print("  {:<12} phr {:>6.2f}  reached {:.4f}".format(
                run["label"], run["phr"], run["max_strain"]))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Scatter reinforcement vs contrast (log x), coloured by PHR."
    )
    parser.add_argument("sweep_dir", nargs="?", default=DEFAULT_SWEEP_DIR,
                        help="Contrast-sweep directory (default: %(default)s).")
    parser.add_argument("--strain", default=str(DEFAULT_STRAIN),
                        help="Strain F11 - 1 to compare at, or 'common' for the "
                             "largest strain every run reached (default: %(default)s).")
    parser.add_argument("--raw-p11", action="store_true",
                        help="Plot P11 itself instead of the ratio to the neat matrix.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: reinforcement_vs_contrast.png "
                             "in sweep_dir).")
    parser.add_argument("--show", action="store_true", help="Also open the figure window.")
    parser.add_argument("--no-lines", dest="lines", action="store_false", default=True,
                        help="Do not connect the points of one PHR.")
    parser.add_argument("--dodge", type=float, default=0.03,
                        help="Horizontal spread of overlapping PHRs, in log10 decades "
                             "(0 disables; default %(default)s).")
    parser.add_argument("--marker-size", type=float, default=55.0)
    parser.add_argument("--figsize", nargs=2, type=float, default=[9.0, 6.0],
                        metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    runs = load_sweep(args.sweep_dir)
    strain = resolve_strain(runs, args.strain)
    rows, skipped = reinforcement_rows(runs, strain)
    if not rows:
        raise SystemExit("No run reached strain {:g}".format(strain))

    suffix = "" if args.strain == "common" else "_strain{:g}".format(strain)
    args.out = args.out or os.path.join(
        args.sweep_dir, "reinforcement_vs_contrast{}.png".format(suffix))
    print_table(rows, skipped, strain)
    make_plot(rows, strain, args)


if __name__ == "__main__":
    main()
