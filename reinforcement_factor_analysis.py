# -*- coding: utf-8 -*-
"""Compute the small-strain reinforcement factor of a batch of FFT simulations
and compare it to the Guth-Gold and Mori-Tanaka analytical predictions.

Each ``phr_*_id*_voxel_output`` folder holds a uniaxial-tension run (F11
controlled, P22 = P33 = 0) of a Neo-Hookean matrix/filler RVE. Because the
constitutive model is calibrated so that its (E, poisson) parameters equal
the exact small-strain Young's modulus / Poisson ratio, the initial slope of
the nominal-stress vs. engineering-strain curve

    dP11/d(F11-1) at F11 -> 1

is the composite's effective Young's modulus E_eff under uniaxial stress.
The reinforcement factor is RF = E_eff / E_matrix.

Usage:
    python reinforcement_factor_analysis.py
    python reinforcement_factor_analysis.py Results/50_improved_struct_v4
    python reinforcement_factor_analysis.py <results_dir> --fit-points 3 --out rf.png
"""

import argparse
import csv
import importlib.util
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import brentq

DEFAULT_RESULTS_DIR = os.path.join("Results", "50_improved_struct_v4")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_PHR_RE = re.compile(r"phr_(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_ID_RE = re.compile(r"_id(\d+)_", re.IGNORECASE)
_VF_RE = re.compile(r"Filler volume fraction:\s*([0-9.eE+-]+)")
_MATRIX_RE = re.compile(
    r"Phase\s+\d+\s*\(matrix\):\s*model\s+(\d+),\s*E\s+([0-9.eE+-]+),\s*poisson\s+([0-9.eE+-]+)"
)
_FILLER_RE = re.compile(
    r"Phase\s+\d+\s*\(filler\):\s*model\s+(\d+),\s*E\s+([0-9.eE+-]+),\s*poisson\s+([0-9.eE+-]+)"
)

_UMAT_CACHE = {}


def load_umat(model_num):
    """Dynamically load the umat(f, parameters) function for a constitutive model."""
    if model_num not in _UMAT_CACHE:
        path = os.path.join(REPO_ROOT, "fg", "constitutive", "{}.py".format(model_num))
        spec = importlib.util.spec_from_file_location("umat_model_{}".format(model_num), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _UMAT_CACHE[model_num] = mod.umat
    return _UMAT_CACHE[model_num]


def analytic_uniaxial_p11(f11_values, model_num, e, nu):
    """Nominal P11 for a homogeneous block of this material under the same
    uniaxial-stress condition used in the FFT charges (P22 = P33 = 0,
    transverse isotropy => F22 = F33 = lam), solved via the model's own umat."""
    umat = load_umat(model_num)

    def p22_residual(lam, f11):
        f = np.diag([f11, lam, lam])
        p, _ = umat(f, [e, nu])
        return p[1, 1]

    p11_values = np.empty_like(f11_values, dtype=float)
    lam_guess = 1.0
    for i, f11 in enumerate(f11_values):
        lo, hi = 0.05, max(2.0, lam_guess * 1.5)
        lam = brentq(lambda lam: p22_residual(lam, f11), lo, hi, xtol=1e-13)
        f = np.diag([f11, lam, lam])
        p, _ = umat(f, [e, nu])
        p11_values[i] = p[0, 0]
        lam_guess = lam
    return p11_values


def parse_phr(folder_name):
    match = _PHR_RE.search(folder_name)
    return float(match.group(1)) if match else None


def parse_id(folder_name):
    match = _ID_RE.search(folder_name)
    return int(match.group(1)) if match else None


def read_metadata(meta_path):
    """Return (phi, matrix_model, E_matrix, nu_matrix, E_filler, nu_filler) or None."""
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        text = f.read()
    vf_match = _VF_RE.search(text)
    matrix_match = _MATRIX_RE.search(text)
    filler_match = _FILLER_RE.search(text)
    if not (vf_match and matrix_match and filler_match):
        return None
    phi = float(vf_match.group(1))
    matrix_model = int(matrix_match.group(1))
    e_m, nu_m = float(matrix_match.group(2)), float(matrix_match.group(3))
    e_f, nu_f = float(filler_match.group(2)), float(filler_match.group(3))
    return phi, matrix_model, e_m, nu_m, e_f, nu_f


def read_solver_stats(folder):
    stats_path = os.path.join(folder, "solver_stats.json")
    if not os.path.isfile(stats_path):
        return None
    with open(stats_path) as f:
        return json.load(f)


def n_converged_increments(stats):
    """Number of leading increments marked converged (contiguous from the start)."""
    count = 0
    for inc in stats.get("increments", []):
        if not inc.get("converged"):
            break
        count += 1
    return count


def parse_resolved_config(results_dir):
    """Map case_name -> {structure_path, charge_path, phase_key, matrix_phase,
    filler_phase} from resolved_config.yaml. Insertion order follows the yaml
    ``cases`` list, so ``next(iter(...))`` gives the sweep's first case."""
    cfg_path = os.path.join(results_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    out = {}
    for case in cfg.get("cases", []):
        out[case["case_name"]] = {
            "structure_path": case["structure_path"],
            "charge_path": case.get("charge_path"),
            "phase_key": case.get("phase_key", "phase"),
            "matrix_phase": case.get("matrix_phase", 0),
            "filler_phase": case.get("filler_phase", 1),
        }
    return out


def local_repo_path(remote_path):
    """Map a remote (container) path onto this checkout, using the path
    segment after the repo-root marker 'mixed-fft-rubber/'."""
    marker = "mixed-fft-rubber/"
    posix_path = remote_path.replace("\\", "/")
    idx = posix_path.find(marker)
    if idx == -1:
        return remote_path
    relative = posix_path[idx + len(marker):]
    return os.path.join(REPO_ROOT, *relative.split("/"))


def compute_phi_from_npz(npz_path, phase_key, filler_phase):
    data = np.load(npz_path)
    phase = data[phase_key]
    return float(np.count_nonzero(phase == filler_phase)) / phase.size


def parse_charge_material(charge_path, matrix_phase=0, filler_phase=1):
    """Read (model_num, E, poisson) per phase from a charge .txt file.

    Format: a '#'-commented header line, then one data line per phase
    (columns: model_num, p1=E, p2=poisson, ...), in phase-index order.
    This is available regardless of whether any run in the sweep finished,
    unlike run_metadata.txt (only written on full completion).
    """
    with open(charge_path) as f:
        data_lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    n_needed = max(matrix_phase, filler_phase) + 1
    phase_rows = [
        [float(x) for x in data_lines[i].split()]
        for i in range(n_needed)
    ]
    m = phase_rows[matrix_phase]
    f_ = phase_rows[filler_phase]
    return int(m[0]), m[1], m[2], int(f_[0]), f_[1], f_[2]


def material_defaults_from_config(results_dir, config_map):
    """Best-effort (matrix_model, e_m, nu_m, e_f, nu_f) from the sweep's charge
    file, using the first case in resolved_config.yaml. Returns None if
    resolved_config.yaml is absent or the charge file can't be found/parsed."""
    if not config_map:
        return None
    first_entry = next(iter(config_map.values()))
    charge_path = first_entry.get("charge_path")
    if not charge_path:
        return None
    local_charge = local_repo_path(charge_path)
    if not os.path.isfile(local_charge):
        return None
    matrix_model, e_m, nu_m, _, e_f, nu_f = parse_charge_material(
        local_charge, first_entry["matrix_phase"], first_entry["filler_phase"])
    return matrix_model, e_m, nu_m, e_f, nu_f


def load_f11_p11(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name.strip(): i for i, name in enumerate(header)}
        f11, p11 = [], []
        for row in reader:
            if not row:
                continue
            f11.append(float(row[idx["F11"]]))
            p11.append(float(row[idx["P11"]]))
    return np.asarray(f11), np.asarray(p11)


def initial_modulus(f11, p11, n_fit_points):
    """Least-squares slope of P11 vs (F11-1) through the origin, using the
    n_fit_points smallest-strain samples (secant modulus if n_fit_points=1)."""
    order = np.argsort(f11)
    strain = (f11[order] - 1.0)[:n_fit_points]
    stress = p11[order][:n_fit_points]
    return float(np.sum(strain * stress) / np.sum(strain * strain))


def mori_tanaka_moduli(phi, e_m, nu_m, e_f, nu_f):
    """Effective (K, G) of a two-phase isotropic composite with spherical
    inclusions, Mori-Tanaka estimate (Benveniste 1987 closed form)."""
    k_m = e_m / (3.0 * (1.0 - 2.0 * nu_m))
    g_m = e_m / (2.0 * (1.0 + nu_m))
    k_f = e_f / (3.0 * (1.0 - 2.0 * nu_f))
    g_f = e_f / (2.0 * (1.0 + nu_f))

    k_mt = k_m + (phi * (k_f - k_m) * (3 * k_m + 4 * g_m)) / (
        3 * k_m + 4 * g_m + 3 * (1 - phi) * (k_f - k_m)
    )
    g_mt = g_m + (phi * (g_f - g_m) * 5 * g_m * (3 * k_m + 4 * g_m)) / (
        5 * g_m * (3 * k_m + 4 * g_m) + 6 * (1 - phi) * (g_f - g_m) * (k_m + 2 * g_m)
    )
    return k_mt, g_mt


def youngs_modulus(k, g):
    return 9.0 * k * g / (3.0 * k + g)


def guth_gold(phi):
    return 1.0 + 2.5 * phi + 14.1 * phi ** 2


def collect(results_dir, n_fit_points):
    rows = []
    skipped = []
    material_defaults = None  # (matrix_model, e_m, nu_m, e_f, nu_f), constant across the sweep
    config_map = parse_resolved_config(results_dir)

    folders = sorted(
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
    )

    # Material parameters needed as a fallback for partial runs (missing
    # run_metadata.txt, which is only written on full completion). Prefer the
    # sweep's charge file via resolved_config.yaml -- available even if zero
    # runs finished -- and fall back to any completed run's metadata.
    material_defaults = material_defaults_from_config(results_dir, config_map)
    if material_defaults is None:
        for name in folders:
            meta = read_metadata(os.path.join(results_dir, name, "run_metadata.txt"))
            if meta is not None:
                _, matrix_model, e_m, nu_m, e_f, nu_f = meta
                material_defaults = (matrix_model, e_m, nu_m, e_f, nu_f)
                break

    for name in folders:
        folder = os.path.join(results_dir, name)
        csv_path = os.path.join(folder, "output.csv")
        meta_path = os.path.join(folder, "run_metadata.txt")
        if not os.path.isfile(csv_path):
            continue

        stats = read_solver_stats(folder)
        status = stats.get("status") if stats else None
        partial = False

        meta = read_metadata(meta_path)
        if meta is not None and status == "converged":
            phi, matrix_model, e_m, nu_m, e_f, nu_f = meta
            usable_rows = None  # every row in output.csv is a converged increment
        elif status == "in_progress" and stats is not None:
            # Not all increments finished, but the ones that did are usable:
            # each converged increment's row was written before the solver
            # moved to (and got stuck on) the next one.
            usable_rows = n_converged_increments(stats)
            if usable_rows < n_fit_points:
                skipped.append((name, "only {} converged increment(s)".format(usable_rows)))
                continue
            case_name = name[:-len("_output")] if name.endswith("_output") else name
            cfg_entry = config_map.get(case_name)
            if cfg_entry is None:
                skipped.append((name, "in_progress run with no resolved_config.yaml entry"))
                continue
            if material_defaults is None:
                skipped.append((name, "in_progress run but no reference material params found"))
                continue
            npz_path = local_repo_path(cfg_entry["structure_path"])
            if not os.path.isfile(npz_path):
                skipped.append((name, "in_progress run, structure npz not found at {}".format(npz_path)))
                continue
            phi = compute_phi_from_npz(npz_path, cfg_entry["phase_key"], cfg_entry["filler_phase"])
            matrix_model, e_m, nu_m, e_f, nu_f = material_defaults
            partial = True
        else:
            skipped.append((name, "solver status = {!r}".format(status)))
            continue

        phr = parse_phr(name)
        struct_id = parse_id(name)
        f11, p11 = load_f11_p11(csv_path)
        if usable_rows is not None:
            f11, p11 = f11[:usable_rows], p11[:usable_rows]
        if len(f11) < n_fit_points:
            skipped.append((name, "fewer than {} usable data rows".format(n_fit_points)))
            continue

        # Composite secant/fit modulus over the n smallest-strain samples.
        e_eff_fit = initial_modulus(f11, p11, n_fit_points)
        e_eff_secant = initial_modulus(f11, p11, 1)

        # Strain-matched analytic pure-matrix baseline: same F11 samples, same
        # umat, solved under the same P22=P33=0 uniaxial condition. This
        # isolates the microstructural (filler) reinforcement from the matrix's
        # own hyperelastic softening at finite strain (the smallest sampled
        # strain here is 10%, not infinitesimal).
        order = np.argsort(f11)
        f11_fit = f11[order][:n_fit_points]
        p11_matrix_fit = analytic_uniaxial_p11(f11_fit, matrix_model, e_m, nu_m)
        e_matrix_fit = initial_modulus(f11_fit, p11_matrix_fit, n_fit_points)

        k_mt, g_mt = mori_tanaka_moduli(phi, e_m, nu_m, e_f, nu_f)
        e_mt = youngs_modulus(k_mt, g_mt)

        rows.append({
            "name": name,
            "phr": phr,
            "id": struct_id,
            "phi": phi,
            "partial": partial,
            "e_matrix": e_m,
            "e_matrix_fit": e_matrix_fit,
            "rf_sim": e_eff_fit / e_matrix_fit,
            "rf_sim_vs_E0": e_eff_fit / e_m,
            "rf_sim_secant": e_eff_secant / e_m,
            "rf_gg": guth_gold(phi),
            "rf_mt": e_mt / e_m,
        })

    rows.sort(key=lambda r: r["phi"])
    return rows, skipped, material_defaults


def write_table(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "phr", "id", "phi", "partial_run",
            "RF_sim", "RF_sim_vs_E0", "RF_guth_gold", "RF_mori_tanaka",
            "RF_sim/RF_gg", "RF_sim/RF_mt",
        ])
        for r in rows:
            writer.writerow([
                r["name"], "{:.2f}".format(r["phr"]) if r["phr"] is not None else "",
                r["id"], "{:.6f}".format(r["phi"]), r["partial"],
                "{:.4f}".format(r["rf_sim"]), "{:.4f}".format(r["rf_sim_vs_E0"]),
                "{:.4f}".format(r["rf_gg"]), "{:.4f}".format(r["rf_mt"]),
                "{:.4f}".format(r["rf_sim"] / r["rf_gg"]),
                "{:.4f}".format(r["rf_sim"] / r["rf_mt"]),
            ])


def make_plot(rows, e_m, nu_m, e_f, nu_f, out_path):
    full = [r for r in rows if not r["partial"]]
    partial = [r for r in rows if r["partial"]]
    phi_all = np.array([r["phi"] for r in rows])

    phi_curve = np.linspace(0.0, max(phi_all.max() * 1.05, 0.01), 200)
    rf_gg_curve = guth_gold(phi_curve)
    k_mt, g_mt = mori_tanaka_moduli(phi_curve, e_m, nu_m, e_f, nu_f)
    rf_mt_curve = youngs_modulus(k_mt, g_mt) / e_m

    fig, ax = plt.subplots(figsize=(8, 6))
    if full:
        ax.scatter([r["phi"] for r in full], [r["rf_sim"] for r in full],
                   s=45, color="black", edgecolors="white", linewidths=0.5,
                   zorder=3, label="FFT simulation (per structure)")
    if partial:
        ax.scatter([r["phi"] for r in partial], [r["rf_sim"] for r in partial],
                   s=60, marker="^", color="crimson", edgecolors="white",
                   linewidths=0.5, zorder=3,
                   label="FFT simulation (partial run, fit unaffected)")
    ax.plot(phi_curve, rf_gg_curve, "--", color="tab:orange", linewidth=2,
            label="Guth-Gold")
    ax.plot(phi_curve, rf_mt_curve, "-", color="tab:blue", linewidth=2,
            label="Mori-Tanaka")

    ax.set_xlabel(r"Filler volume fraction $\phi$")
    ax.set_ylabel(r"Reinforcement factor $E_{eff}/E_{matrix}$")
    ax.set_title("Reinforcement factor vs. filler content\n(E_matrix={:g}, E_filler={:g}, contrast={:g}x)"
                 .format(e_m, e_f, e_f / e_m))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print("saved plot to {}".format(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulated reinforcement factor to Guth-Gold and Mori-Tanaka.")
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fit-points", type=int, default=3,
                         help="Number of smallest-strain points used for the "
                              "initial-modulus line fit (default 3).")
    parser.add_argument("--out", default=None, help="Output plot path (PNG).")
    parser.add_argument("--csv-out", default=None, help="Output table path (CSV).")
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.results_dir, "reinforcement_factor_comparison.png")
    csv_out = args.csv_out or os.path.join(args.results_dir, "reinforcement_factor_table.csv")

    rows, skipped, material_defaults = collect(args.results_dir, args.fit_points)
    if not rows:
        raise SystemExit("No usable structures found in {}".format(args.results_dir))
    if material_defaults is None:
        raise SystemExit("Could not determine matrix/filler material parameters "
                          "from any run_metadata.txt in {}".format(args.results_dir))
    matrix_model, e_m, nu_m, e_f, nu_f = material_defaults

    n_partial = sum(1 for r in rows if r["partial"])
    print("Used {} structures ({} fully converged, {} partial runs) "
          "(phi = {:.4f} - {:.4f})".format(
              len(rows), len(rows) - n_partial, n_partial, rows[0]["phi"], rows[-1]["phi"]))
    if n_partial:
        print("Partial runs (last increment did not finish, but the first "
              "{} increments used for the fit converged fine):".format(args.fit_points))
        for r in rows:
            if r["partial"]:
                print("  {}".format(r["name"]))
    if skipped:
        print("Skipped {} folder(s):".format(len(skipped)))
        for name, reason in skipped:
            print("  {} ({})".format(name, reason))

    print("\nRF_sim uses a strain-matched analytic pure-matrix baseline "
          "(same F11 samples run through the matrix's own umat under the "
          "same P22=P33=0 uniaxial condition), so it isolates filler "
          "reinforcement from the matrix's own finite-strain (10-30%) "
          "hyperelastic softening. Guth-Gold and Mori-Tanaka are "
          "infinitesimal-strain linear-elastic predictions.")

    print("\n{:<28s} {:>7s} {:>8s} {:>9s} {:>9s} {:>9s}".format(
        "structure", "phr", "phi", "RF_sim", "RF_GG", "RF_MT"))
    for r in rows:
        print("{:<28s} {:>7.2f} {:>8.4f} {:>9.4f} {:>9.4f} {:>9.4f}".format(
            r["name"], r["phr"], r["phi"], r["rf_sim"], r["rf_gg"], r["rf_mt"]))

    write_table(rows, csv_out)
    print("\nsaved table to {}".format(csv_out))

    make_plot(rows, e_m, nu_m, e_f, nu_f, out_path)


if __name__ == "__main__":
    main()
