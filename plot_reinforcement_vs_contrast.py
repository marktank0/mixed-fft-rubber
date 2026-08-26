# -*- coding: utf-8 -*-
"""Plot reinforcement versus filler/matrix contrast for a benchmark-suite run.

Every (contrast, solver-config) point of a ``benchmark_suite.py`` output directory is
drawn as one marker: y = reinforcement, x = contrast on a log axis, with a distinct
colour *and* symbol per solver configuration.

Reinforcement is the macroscopic stress of the filled structure divided by the stress of
the neat matrix at the same stretch,

    R(F11) = P11_composite(F11) / P11_matrix(F11),

so R = 1 means "no stiffer than unfilled rubber". The neat-matrix reference is not taken
from a file: it is computed from the same constitutive routine the solver uses
(``FFT_simulation/fg/constitutive_incompressible/<model>.py``) by solving the homogeneous uniaxial state
F = diag(F11, l, l), P22 = P33 = 0 together with the mixed-formulation pressure condition
J - 1 = y/D -- i.e. exactly the load case the charge file prescribes, without any filler.

Usage:
    python plot_reinforcement_vs_contrast.py
    python plot_reinforcement_vs_contrast.py Results/benchmark_suite --show
    python plot_reinforcement_vs_contrast.py --increment 0 --out r_at_first_step.png
    python plot_reinforcement_vs_contrast.py --raw-p11        # plot P11 instead of the ratio
"""

import argparse
import csv
import importlib
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

from project_paths import CHARGES_DIR, ensure_import_paths, results_path

ensure_import_paths()

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_RESULTS_DIR = os.path.join("Results", "benchmark_suite")

# Fallbacks used only when the run's charge file cannot be found on this machine.
# They mirror benchmark_suite.CHARGE_TEMPLATE / E_MATRIX (matrix = phase 0).
FALLBACK_MATRIX_MODEL = "1.py"
FALLBACK_MATRIX_E = 10.0
FALLBACK_MATRIX_NU = 0.48

# One (colour, marker) pair per solver config; both differ between configs so the points
# stay distinguishable in greyscale print as well.
_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#b5651d", "#00688b",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p", "8"]

# Preference order when the same (structure, contrast, config) was run more than once.
_STATUS_RANK = {"converged": 0, "converged_with_step_cuts": 1, "failed": 2}


# --------------------------------------------------------------------- matrix reference
def load_umat_field(model_name):
    """Load the batched ``umat_field`` of a constitutive model, as mxfft.py does."""
    module = importlib.import_module(
        "fg.constitutive_incompressible.{}".format(os.path.splitext(model_name)[0])
    )
    return module.umat_field


def matrix_p11(f11, model_name, young, poisson):
    """P11 of the *unfilled* matrix at stretch f11 under the benchmark load case.

    Solves the homogeneous uniaxial state -- F = diag(f11, l, l), lateral faces
    traction free (P22 = 0), pressure y tied to the volume change by the mixed
    formulation's constraint 1 - J + y/D = 0 (the same residual mxfft.py assembles).
    """
    umat_field = load_umat_field(model_name)
    D = None if poisson == 0.5 else young / 3.0 / (1 - 2 * poisson)
    kappa_inv = 0.0 if D is None else 1.0 / D

    def residual(unknowns):
        lateral, pressure = unknowns
        f = np.diag([f11, lateral, lateral])[None]
        P, _, _, _ = umat_field(f, np.array([pressure]), [young, poisson], need_tangent=False)
        J = float(np.linalg.det(f[0]))
        return [P[0, 1, 1], 1.0 - J + pressure * kappa_inv]

    guess = [1.0 / np.sqrt(f11), 0.0]
    lateral, pressure = fsolve(residual, guess, full_output=False)
    f = np.diag([f11, lateral, lateral])[None]
    P, _, _, _ = umat_field(f, np.array([pressure]), [young, poisson], need_tangent=False)
    if abs(P[0, 1, 1]) > 1e-6 * max(1.0, abs(P[0, 0, 0])):
        raise RuntimeError("neat-matrix solve did not reach a traction-free lateral state")
    return float(P[0, 0, 0])


def read_charge_matrix_params(charge_path):
    """(model_name, E, nu) of phase 0 (the matrix) from a charge file, or None."""
    if not charge_path or not os.path.isfile(charge_path):
        return None
    with open(charge_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            return "{}.py".format(int(float(fields[0]))), float(fields[1]), float(fields[2])
    return None


def resolve_matrix_params(args, contrast):
    """Matrix (model, E, nu): explicit flags win, then the run's charge file, then defaults."""
    if args.matrix_e is not None and args.matrix_nu is not None:
        return args.matrix_model, args.matrix_e, args.matrix_nu

    candidates = []
    if args.charge:
        candidates.append(args.charge)
    candidates.append(os.path.join(REPO_ROOT, args.charge_dir, "bench_c{}.txt".format(contrast)))
    for path in candidates:
        params = read_charge_matrix_params(path)
        if params is not None:
            return params
    return FALLBACK_MATRIX_MODEL, FALLBACK_MATRIX_E, FALLBACK_MATRIX_NU


# ------------------------------------------------------------------------- run records
def load_records(results_dir):
    """All benchmark records, deduplicated, preferring converged over failed reruns."""
    jsonl_path = os.path.join(results_dir, "results.jsonl")
    csv_path = os.path.join(results_dir, "summary.csv")

    records = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif os.path.isfile(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                row["contrast"] = float(row["contrast"])
                row["P11"] = float(row["P11"]) if row.get("P11") else None
                row["F11"] = float(row["F11"]) if row.get("F11") else None
                records.append(row)
    else:
        raise SystemExit("No results.jsonl or summary.csv in {}".format(results_dir))

    best = {}
    for record in records:
        key = (record["structure"], float(record["contrast"]), record["config"])
        rank = _STATUS_RANK.get(record.get("status"), 3)
        if key not in best or rank <= _STATUS_RANK.get(best[key].get("status"), 3):
            best[key] = record
    return list(best.values())


def run_curve(results_dir, record):
    """(F11 array, P11 array) over the increments of one run.

    The record's P11_curve is used whenever it exists: the per-run output.csv stores only
    3 significant digits *truncated* (a final F11 of 1.2999999 is written as "1.29e+00"),
    which is enough to shift a reinforcement ratio by a few percent. The load ramps
    linearly from 1 in equal increments, so the intermediate F11 values are exact.
    """
    curve = record.get("P11_curve")
    if curve and record.get("F11") is not None:
        p11 = np.asarray(curve, dtype=float)
        f11 = 1.0 + (float(record["F11"]) - 1.0) * np.arange(1, len(p11) + 1) / len(p11)
        return f11, p11

    # Fallback (e.g. a summary.csv-only directory): the low-precision per-run curve.
    out_csv = os.path.join(
        results_dir, "runs", record["structure"],
        "c{:g}".format(float(record["contrast"])), record["config"], "output.csv",
    )
    if os.path.isfile(out_csv):
        with open(out_csv, newline="") as fh:
            f11, p11 = [], []
            for row in csv.DictReader(fh):
                f11.append(float(row["F11"]))
                p11.append(float(row["P11"]))
        if f11:
            return np.asarray(f11), np.asarray(p11)

    if record.get("P11") is not None and record.get("F11") is not None:
        return np.asarray([float(record["F11"])]), np.asarray([float(record["P11"])])
    return None, None


def point_at_increment(f11, p11, increment):
    if f11 is None or len(f11) == 0:
        return None, None
    index = increment if increment >= 0 else len(f11) + increment
    if not 0 <= index < len(f11):
        return None, None
    return float(f11[index]), float(p11[index])


def collect(args):
    """Rows of {config, contrast, f11, p11, reinforcement, status} plus the failures."""
    results_dir = args.results_dir
    records = load_records(results_dir)
    if args.structure:
        records = [r for r in records if r["structure"] == args.structure]
    structures = sorted({r["structure"] for r in records})
    if len(structures) > 1 and not args.structure:
        print("Note: {} structures present ({}); plotting all of them. "
              "Use --structure to pick one.".format(len(structures), ", ".join(structures)))

    reference_cache = {}
    rows, failures = [], []
    for record in records:
        contrast = float(record["contrast"])
        f11_curve, p11_curve = run_curve(results_dir, record)
        f11, p11 = point_at_increment(f11_curve, p11_curve, args.increment)
        if p11 is None or record.get("status") == "failed":
            failures.append({
                "structure": record["structure"], "config": record["config"],
                "contrast": contrast, "status": record.get("status", "?"),
            })
            continue

        key = (contrast, round(f11, 9))
        if key not in reference_cache:
            model, young, poisson = resolve_matrix_params(args, contrast)
            if args.matrix_p11 is not None:
                reference_cache[key] = (args.matrix_p11, (model, young, poisson))
            else:
                reference_cache[key] = (
                    matrix_p11(f11, model, young, poisson), (model, young, poisson),
                )
        reference, matrix_params = reference_cache[key]

        rows.append({
            "structure": record["structure"], "config": record["config"],
            "family": record.get("family", ""), "contrast": contrast,
            "f11": f11, "p11": p11, "matrix_p11": reference,
            "reinforcement": p11 / reference, "matrix_params": matrix_params,
            "status": record.get("status", "?"),
        })
    return rows, failures


# ------------------------------------------------------------------------------- plot
def dodge(contrast, index, n_configs, strength):
    """Spread the configs of one contrast slightly along the log axis.

    Several configs land on identical stresses (that is the point of the study), so
    without a dodge their markers hide each other completely.
    """
    if n_configs < 2 or strength <= 0:
        return contrast
    offset = (index - (n_configs - 1) / 2.0) / (n_configs - 1)
    return contrast * (10.0 ** (offset * strength))


def make_plot(rows, failures, args):
    configs = sorted({row["config"] for row in rows})
    if args.config_order:
        ordered = [c for c in args.config_order if c in configs]
        configs = ordered + [c for c in configs if c not in ordered]
    style = {
        config: (_COLORS[i % len(_COLORS)], _MARKERS[i % len(_MARKERS)])
        for i, config in enumerate(configs)
    }

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    y_key = "p11" if args.raw_p11 else "reinforcement"

    for index, config in enumerate(configs):
        series = sorted(
            (r for r in rows if r["config"] == config), key=lambda r: r["contrast"]
        )
        color, marker = style[config]
        x = [dodge(r["contrast"], index, len(configs), args.dodge) for r in series]
        y = [r[y_key] for r in series]
        if args.lines:
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.35, zorder=1)
        # Step-cut runs converged only after the solver retreated: hollow marker.
        face = [
            color if r["status"] == "converged" else "none" for r in series
        ]
        ax.scatter(
            x, y, s=args.marker_size, marker=marker, facecolors=face, edgecolors=color,
            linewidths=1.4, label=config, zorder=3,
        )

    if args.mark_failed and failures:
        # Failures get their own band below the data: drawn at the current bottom they
        # would read as data points.
        low, high = ax.get_ylim()
        span = high - low
        band = low - 0.10 * span
        for index, config in enumerate(configs):
            for failure in failures:
                if failure["config"] != config:
                    continue
                color = style[config][0]
                ax.scatter(
                    [dodge(failure["contrast"], index, len(configs), args.dodge)], [band],
                    s=args.marker_size * 0.6, marker="x", color=color, alpha=0.75,
                    linewidths=1.2, zorder=2,
                )
        ax.set_ylim(band - 0.06 * span, high)
        ax.text(
            0.01, band, "did not converge  ", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8, color="0.35",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )

    if not args.raw_p11:
        if args.unity:
            ax.set_ylim(min(ax.get_ylim()[0], 0.99), ax.get_ylim()[1])
        low, high = ax.get_ylim()
        # Only annotate R = 1 when it is actually on the axis; otherwise the label would
        # float underneath the frame.
        if low <= 1.0 <= high:
            ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
            ax.text(
                0.995, 1.0, " neat matrix", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="0.4",
            )

    contrasts = sorted({row["contrast"] for row in rows} | {f["contrast"] for f in failures})
    ax.set_xscale("log")
    ax.set_xticks(contrasts)
    ax.set_xticklabels(["{:g}".format(c) for c in contrasts])
    ax.minorticks_off()
    ax.set_xlabel(r"filler/matrix contrast  $E_f/E_m$")
    ax.set_ylabel(r"$P_{11}$  [MPa]" if args.raw_p11
                  else r"reinforcement  $P_{11}/P_{11}^{\mathrm{matrix}}$")

    f11_values = sorted({round(row["f11"], 4) for row in rows})
    strain_note = "F11 = {}".format(", ".join("{:g}".format(v) for v in f11_values))
    title = args.title or "Reinforcement vs contrast ({})".format(strain_note)
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.25)

    ax.legend(
        title="solver config", loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=8, title_fontsize=9, frameon=False,
    )
    fig.tight_layout()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print("Saved {}".format(out_path))
    if args.show:
        plt.show()
    plt.close(fig)


def print_table(rows, failures):
    if not rows:
        return
    configs = sorted({row["config"] for row in rows})
    contrasts = sorted({row["contrast"] for row in rows})
    lookup = {(row["config"], row["contrast"]): row for row in rows}

    model, young, poisson = rows[0]["matrix_params"]
    print("Neat-matrix reference: model {}, E = {:g}, nu = {:g}".format(model, young, poisson))
    for f11 in sorted({round(row["f11"], 6) for row in rows}):
        reference = next(r["matrix_p11"] for r in rows if round(r["f11"], 6) == f11)
        print("  P11_matrix(F11 = {:g}) = {:.6f}".format(f11, reference))
    print()
    header = "{:<20}".format("config") + "".join("{:>12}".format("c{:g}".format(c)) for c in contrasts)
    print(header)
    print("-" * len(header))
    for config in configs:
        cells = []
        for contrast in contrasts:
            row = lookup.get((config, contrast))
            if row is None:
                cells.append("{:>12}".format("failed"))
            else:
                flag = "" if row["status"] == "converged" else "*"
                cells.append("{:>12}".format("{:.3f}{}".format(row["reinforcement"], flag)))
        print("{:<20}".format(config) + "".join(cells))
    print("\n* converged only with step cuts")
    if failures:
        print("Excluded (no usable P11): " + ", ".join(
            sorted("{}@c{:g}".format(f["config"], f["contrast"]) for f in failures)
        ))


def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                   help="benchmark_suite output directory (default: %(default)s)")
    p.add_argument("--structure", default=None, help="Only plot this structure.")
    p.add_argument("--increment", type=int, default=-1,
                   help="Which load increment to plot; -1 = last (default), 0 = first.")
    p.add_argument("--out", default=results_path("benchmark_suite",
                                                "reinforcement_vs_contrast.png"))
    p.add_argument("--show", action="store_true", help="Also open the figure window.")
    p.add_argument("--raw-p11", action="store_true",
                   help="Plot P11 itself instead of the ratio to the neat matrix.")

    p.add_argument("--charge-dir", default=CHARGES_DIR,
                   help="Where to look for the run's bench_c<contrast>.txt charge file.")
    p.add_argument("--charge", default=None, help="Explicit charge file for the matrix phase.")
    p.add_argument("--matrix-model", default=FALLBACK_MATRIX_MODEL,
                   help="Constitutive model file of the matrix (default: %(default)s).")
    p.add_argument("--matrix-e", type=float, default=None, help="Matrix Young's modulus.")
    p.add_argument("--matrix-nu", type=float, default=None, help="Matrix Poisson ratio.")
    p.add_argument("--matrix-p11", type=float, default=None,
                   help="Skip the neat-matrix solve and use this reference stress directly.")

    p.add_argument("--config-order", nargs="*", default=None,
                   help="Legend/colour order; unlisted configs follow alphabetically.")
    p.add_argument("--no-lines", dest="lines", action="store_false", default=True,
                   help="Do not connect the points of a config.")
    p.add_argument("--dodge", type=float, default=0.07,
                   help="Horizontal spread of overlapping configs, in log10 decades "
                        "(0 disables; default %(default)s).")
    p.add_argument("--unity", action="store_true",
                   help="Force the y axis to include the neat-matrix level R = 1.")
    p.add_argument("--mark-failed", action="store_true",
                   help="Put a grey x at the bottom axis where a config failed to converge.")
    p.add_argument("--marker-size", type=float, default=55.0)
    p.add_argument("--figsize", nargs=2, type=float, default=[9.0, 6.0], metavar=("W", "H"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title", default=None)
    return p


def main(cli_args=None):
    args = _build_parser().parse_args(cli_args)
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(REPO_ROOT, args.results_dir)

    rows, failures = collect(args)
    if not rows:
        raise SystemExit("No usable runs found in {}".format(args.results_dir))
    print_table(rows, failures)
    make_plot(rows, failures, args)


if __name__ == "__main__":
    main()
