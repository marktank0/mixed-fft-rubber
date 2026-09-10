# -*- coding: utf-8 -*-
"""Plot the maximum strain every run reached against the filler/matrix contrast.

The transpose of ``plot_max_strain_vs_phr.py``: x = contrast on a log axis,
y = largest F11 - 1 the solver got to before it stopped, one colour per
structure PHR.

A run that finished the whole load path sits at the top of the axis (the
prescribed maximum, 1.0 for the small-step sweep) and is drawn filled; a run
that gave up early sits where it stalled and is drawn hollow, so "converged all
the way" and "happened to stall at the last increment" stay distinguishable.

Usage:
    python Plotting_scripts/plot_max_strain_vs_contrast.py
    python Plotting_scripts/plot_max_strain_vs_contrast.py <sweep_dir> --show
    python Plotting_scripts/plot_max_strain_vs_contrast.py --stretch --out max_F11.png
"""

import argparse
import os

import matplotlib.pyplot as plt

from contrast_sweep_data import DEFAULT_SWEEP_DIR, load_sweep

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


def make_plot(runs, args):
    phrs = sorted({run["phr"] for run in runs})
    offset = 1.0 if args.stretch else 0.0

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for index, phr in enumerate(phrs):
        series = sorted(
            (run for run in runs if run["phr"] == phr), key=lambda run: run["contrast"]
        )
        color = _COLORS[index % len(_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        x = [dodge(run["contrast"], index, len(phrs), args.dodge) for run in series]
        y = [run["max_strain"] + offset for run in series]
        if args.lines:
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.35, zorder=1)
        ax.scatter(
            x, y, s=args.marker_size, marker=marker,
            facecolors=[color if run["complete"] else "none" for run in series],
            edgecolors=color, linewidths=1.4, zorder=3,
            label="{:g}".format(phr),
        )

    contrasts = sorted({run["contrast"] for run in runs})
    ax.set_xscale("log")
    ax.set_xticks(contrasts)
    ax.set_xticklabels(["{:g}".format(contrast) for contrast in contrasts])
    ax.minorticks_off()
    ax.set_xlabel(r"filler/matrix contrast  $E_f/E_m$")
    ax.set_ylabel(r"max $F_{11}$ reached" if args.stretch
                  else r"max strain reached  $F_{11} - 1$")
    ax.set_title(args.title or "Maximum strain reached vs contrast, per PHR")
    ax.grid(True, which="major", alpha=0.25)
    ax.set_ylim(bottom=offset)
    if args.log_y:
        ax.set_yscale("log")

    # The prescribed end of the load path: everything below it is a run that
    # stopped early, which is the whole point of the figure.
    target = max(run["max_strain"] for run in runs) + offset
    ax.axhline(target, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(0.995, target, "full load path ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.4")

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


def print_table(runs):
    contrasts = sorted({run["contrast"] for run in runs})
    phrs = sorted({run["phr"] for run in runs})
    lookup = {(run["contrast"], run["phr"]): run for run in runs}

    header = "{:>10}".format("contrast") + "".join(
        "{:>12}".format("phr{:g}".format(phr)) for phr in phrs
    )
    print("\nmax strain reached (* = stopped before the end of the load path)")
    print(header)
    print("-" * len(header))
    for contrast in contrasts:
        cells = []
        for phr in phrs:
            run = lookup.get((contrast, phr))
            if run is None:
                cells.append("{:>12}".format("-"))
            else:
                flag = "" if run["complete"] else "*"
                cells.append("{:>12}".format("{:.3f}{}".format(run["max_strain"], flag)))
        print("{:>10.0f}".format(contrast) + "".join(cells))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot the maximum strain reached vs contrast (log x), "
                    "coloured by PHR."
    )
    parser.add_argument("sweep_dir", nargs="?", default=DEFAULT_SWEEP_DIR,
                        help="Contrast-sweep directory (default: %(default)s).")
    parser.add_argument("--stretch", action="store_true",
                        help="Plot F11 itself instead of the strain F11 - 1.")
    parser.add_argument("--log-y", action="store_true",
                        help="Log the strain axis too; the strains span a decade.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: max_strain_vs_contrast.png "
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

    args.out = args.out or os.path.join(args.sweep_dir, "max_strain_vs_contrast.png")
    runs = load_sweep(args.sweep_dir)
    print_table(runs)
    make_plot(runs, args)


if __name__ == "__main__":
    main()
