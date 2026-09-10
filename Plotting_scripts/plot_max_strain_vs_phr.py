# -*- coding: utf-8 -*-
"""Plot the maximum strain every run reached against its structure PHR.

One point per simulation of a contrast sweep: x = PHR of the structure,
y = largest F11 - 1 the solver got to before it stopped, one colour per
filler/matrix contrast.

A run that finished the whole load path sits at the top of the axis (the
prescribed maximum, 1.0 for the small-step sweep) and is drawn filled; a run
that gave up early sits where it stalled and is drawn hollow, so "converged all
the way" and "happened to stall at the last increment" stay distinguishable.

Usage:
    python Plotting_scripts/plot_max_strain_vs_phr.py
    python Plotting_scripts/plot_max_strain_vs_phr.py <sweep_dir> --show
    python Plotting_scripts/plot_max_strain_vs_phr.py <sweep_dir> --out max_F11.png
    python Plotting_scripts/plot_max_strain_vs_phr.py <sweep_dir> --stretch --out max_F11.png

"""

import argparse
import os

import matplotlib.pyplot as plt

from contrast_sweep_data import DEFAULT_SWEEP_DIR, load_sweep

# One (colour, marker) pair per contrast; both differ so the series stay
# readable in greyscale print too.
_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#7f7f7f",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def make_plot(runs, args):
    contrasts = sorted({run["contrast"] for run in runs})
    offset = 1.0 if args.stretch else 0.0

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for index, contrast in enumerate(contrasts):
        series = sorted(
            (run for run in runs if run["contrast"] == contrast),
            key=lambda run: run["phr"],
        )
        color = _COLORS[index % len(_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        phr = [run["phr"] for run in series]
        strain = [run["max_strain"] + offset for run in series]
        if args.lines:
            ax.plot(phr, strain, color=color, linewidth=1.0, alpha=0.35, zorder=1)
        ax.scatter(
            phr, strain, s=args.marker_size, marker=marker,
            facecolors=[color if run["complete"] else "none" for run in series],
            edgecolors=color, linewidths=1.4, zorder=3,
            label=r"$E_f/E_m$ = {:g}".format(contrast),
        )

    ax.set_xlabel("PHR")
    ax.set_ylabel(r"max $F_{11}$ reached" if args.stretch
                  else r"max strain reached  $F_{11} - 1$")
    ax.set_title(args.title or "Maximum strain reached vs PHR, per contrast")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=offset)

    # The prescribed end of the load path: everything below it is a run that
    # stopped early, which is the whole point of the figure.
    target = max(run["max_strain"] for run in runs) + offset
    ax.axhline(target, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(0.995, target, "full load path ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.4")

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


def print_table(runs):
    contrasts = sorted({run["contrast"] for run in runs})
    phrs = sorted({run["phr"] for run in runs})
    lookup = {(run["contrast"], run["phr"]): run for run in runs}

    header = "{:>8}".format("phr") + "".join(
        "{:>12}".format("c{:g}".format(contrast)) for contrast in contrasts
    )
    print("\nmax strain reached (* = stopped before the end of the load path)")
    print(header)
    print("-" * len(header))
    for phr in phrs:
        cells = []
        for contrast in contrasts:
            run = lookup.get((contrast, phr))
            if run is None:
                cells.append("{:>12}".format("-"))
            else:
                flag = "" if run["complete"] else "*"
                cells.append("{:>12}".format("{:.3f}{}".format(run["max_strain"], flag)))
        print("{:>8.2f}".format(phr) + "".join(cells))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot the maximum strain reached vs PHR, coloured by contrast."
    )
    parser.add_argument("sweep_dir", nargs="?", default=DEFAULT_SWEEP_DIR,
                        help="Contrast-sweep directory (default: %(default)s).")
    parser.add_argument("--stretch", action="store_true",
                        help="Plot F11 itself instead of the strain F11 - 1.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: max_strain_vs_phr.png in sweep_dir).")
    parser.add_argument("--show", action="store_true", help="Also open the figure window.")
    parser.add_argument("--no-lines", dest="lines", action="store_false", default=True,
                        help="Do not connect the points of one contrast.")
    parser.add_argument("--marker-size", type=float, default=55.0)
    parser.add_argument("--figsize", nargs=2, type=float, default=[9.0, 6.0],
                        metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    args.out = args.out or os.path.join(args.sweep_dir, "max_strain_vs_phr.png")
    runs = load_sweep(args.sweep_dir)
    print_table(runs)
    make_plot(runs, args)


if __name__ == "__main__":
    main()
