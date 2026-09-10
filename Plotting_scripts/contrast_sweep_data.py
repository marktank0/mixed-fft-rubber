# -*- coding: utf-8 -*-
"""Reading a contrast-sweep results tree.

Both plotting scripts in this folder read the same layout, written by
``FFT_simulation/contrast_sweep.py``::

    <sweep_dir>/
        E10-100/                              <- one folder per filler modulus
            phr_15.65_id6_voxel_output/
                output.csv                    <- F11..F33, P11..P33 per increment
                solver_stats.json             <- status + per-increment load steps
                run_metadata.txt              <- phase moduli, structure statistics
            ...
        E10-250/
        ...

Two things are worth knowing about the raw files.

*output.csv stores 3 significant digits*, truncated, so a final F11 of 1.9999
is written as ``1.99e+00``. ``solver_stats.json`` records the exact load of
every increment (``load_start + step``), so the F11 column is rebuilt from it
whenever the two agree on the number of increments.

*Only converged increments are written.* A run that failed part way through
still has a complete, usable curve up to the last increment it managed, which
is exactly the "how far did this get" quantity the max-strain plot shows.
"""

import csv
import importlib
import json
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from project_paths import ensure_import_paths, results_path  # noqa: E402

# The sweep as it was run on the HPC and copied back into the repo.
DEFAULT_SWEEP_DIR = results_path("HPC", "Results", "contrast_sweep_small_step")

# PHR is embedded in the folder name, e.g. "phr_15.65_id6_voxel_output".
_PHR_RE = re.compile(r"phr_(-?\d+(?:\.\d+)?)", re.IGNORECASE)
# Contrast folders are named after the two moduli, e.g. "E10-2500".
_CONTRAST_DIR_RE = re.compile(r"^E(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$", re.IGNORECASE)
_PHASE_RE = re.compile(
    r"Phase\s+(\d+)\s+\((matrix|filler)\):\s*model\s*(\S+?),\s*E\s*([\d.eE+-]+),"
    r"\s*poisson\s*([\d.eE+-]+)"
)
_FILLER_FRACTION_RE = re.compile(r"Filler volume fraction:\s*([\d.eE+-]+)")

FULLY_CONVERGED = ("converged", "converged_with_step_cuts")

# Matrix phase (model, E, nu), used only when run_metadata.txt is missing.
FALLBACK_MATRIX = ("1", 10.0, 0.48)


# ------------------------------------------------------------------ single run
def parse_phr(name):
    match = _PHR_RE.search(name)
    return float(match.group(1)) if match else None


def read_metadata(run_dir):
    """{matrix: (model, E, nu), filler: (...), volume_fraction: float} or {}."""
    path = os.path.join(run_dir, "run_metadata.txt")
    if not os.path.isfile(path):
        return {}
    meta = {}
    with open(path) as handle:
        text = handle.read()
    for _, role, model, young, poisson in _PHASE_RE.findall(text):
        meta[role] = (model, float(young), float(poisson))
    fraction = _FILLER_FRACTION_RE.search(text)
    if fraction:
        meta["volume_fraction"] = float(fraction.group(1))
    return meta


def read_solver_stats(run_dir):
    path = os.path.join(run_dir, "solver_stats.json")
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def read_output_csv(run_dir):
    """(F11, P11) arrays from output.csv; (None, None) when it is missing/empty."""
    path = os.path.join(run_dir, "output.csv")
    if not os.path.isfile(path):
        return None, None
    with open(path, newline="") as handle:
        f11, p11 = [], []
        for row in csv.DictReader(handle):
            if not row or not row.get("F11"):
                continue
            f11.append(float(row["F11"]))
            p11.append(float(row["P11"]))
    if not f11:
        return None, None
    return np.asarray(f11), np.asarray(p11)


_TENSOR_COLUMNS = ["{}{}{}".format(name, i, j)
                   for name in ("F", "P") for i in (1, 2, 3) for j in (1, 2, 3)]


def read_output_tensors(run_dir):
    """(F, P) as (n, 3, 3) arrays from output.csv, or (None, None).

    Needed for the Cauchy stress; the F11/P11 pair on its own cannot give J.
    """
    path = os.path.join(run_dir, "output.csv")
    if not os.path.isfile(path):
        return None, None
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(
            column not in reader.fieldnames for column in _TENSOR_COLUMNS
        ):
            return None, None
        values = [[float(row[column]) for column in _TENSOR_COLUMNS] for row in reader
                  if row and row.get("F11")]
    if not values:
        return None, None
    values = np.asarray(values)
    return values[:, :9].reshape(-1, 3, 3), values[:, 9:].reshape(-1, 3, 3)


def cauchy_11(f, p):
    """sigma_11 = J^-1 (P F^T)_11, the true stress along the tensile axis."""
    j = np.linalg.det(f)
    return np.einsum("vk,vk->v", p[:, 0, :], f[:, 0, :]) / j


def converged_loads(stats):
    """Cumulative load (= F11 - 1) after every increment the solver accepted."""
    if not stats:
        return []
    return [
        float(increment["load_start"]) + float(increment["step"])
        for increment in stats.get("increments", [])
        if increment.get("converged")
    ]


def nominal_checkpoints(stats, count):
    """The `count` loads at which whole prescribed increments completed, or None.

    ``converged_loads`` lists every sub-step the solver accepted, so after a step
    cut it holds far more entries than output.csv has rows -- the CSV gets one
    row per *prescribed* increment. The sub-steps of a cut increment sum back to
    the nominal step, so the rows are the loads sitting on the nominal grid.
    """
    increments = (stats or {}).get("increments", [])
    if not increments:
        return None
    nominal = max(float(increment["step"]) for increment in increments)
    if nominal <= 0:
        return None
    checkpoints = [
        load for load in converged_loads(stats)
        if abs(load / nominal - round(load / nominal)) < 1e-6
    ]
    # Only trust the reconstruction when it lands on exactly the rows the CSV
    # has; anything else means the run did not use one uniform increment.
    return checkpoints if len(checkpoints) == count else None


def load_run(run_dir):
    """One simulation as a dict, or None when it holds no usable curve.

    Keys: phr, dir, status, max_strain, complete, f11, p11, sigma11, matrix,
    filler, contrast, volume_fraction.
    """
    stats = read_solver_stats(run_dir)
    f11, p11 = read_output_csv(run_dir)
    if f11 is None:
        return None

    # The 3-digit F11 column is replaced by the exact loads whenever they can be
    # matched to the CSV's rows one for one -- directly, or, after step cuts, by
    # the loads that complete a whole prescribed increment. Without this the x
    # values of a 0.025 ramp alternate 0.02 / 0.03 while P11 rises smoothly,
    # which shows up as a visible zig-zag in a stress-strain curve.
    loads = converged_loads(stats)
    if len(loads) != len(f11):
        loads = nominal_checkpoints(stats, len(f11)) or []
    if len(loads) == len(f11):
        f11 = 1.0 + np.asarray(loads)

    # The true stress needs the whole tensor, and J is a small number built from
    # the CSV's 3-digit lateral stretches, so sigma11 carries their precision.
    f_tensor, p_tensor = read_output_tensors(run_dir)
    if f_tensor is not None and len(f_tensor) == len(f11):
        f_tensor[:, 0, 0] = f11
        sigma11 = cauchy_11(f_tensor, p_tensor)
    else:
        sigma11 = None

    meta = read_metadata(run_dir)
    matrix = meta.get("matrix")
    filler = meta.get("filler")
    contrast = filler[1] / matrix[1] if matrix and filler and matrix[1] else None
    status = (stats or {}).get("status", "unknown")

    return {
        "phr": parse_phr(os.path.basename(run_dir)),
        "dir": run_dir,
        "status": status,
        "max_strain": float(f11[-1]) - 1.0,
        "complete": status in FULLY_CONVERGED,
        "f11": f11,
        "p11": p11,
        "sigma11": sigma11,
        "matrix": matrix,
        "filler": filler,
        "contrast": contrast,
        "volume_fraction": meta.get("volume_fraction"),
    }


# ------------------------------------------------------------------ whole sweep
def contrast_from_dirname(name):
    match = _CONTRAST_DIR_RE.match(name)
    if not match:
        return None
    matrix_e, filler_e = float(match.group(1)), float(match.group(2))
    return filler_e / matrix_e if matrix_e else None


def load_sweep(sweep_dir):
    """Every run of a sweep as a flat list, sorted by (contrast, phr).

    Accepts either the sweep root (a folder of contrast folders) or a single
    contrast folder, so one contrast can be inspected on its own.
    """
    if not os.path.isdir(sweep_dir):
        raise SystemExit("No such directory: {}".format(sweep_dir))

    entries = sorted(
        name for name in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, name))
    )
    contrast_dirs = [name for name in entries if _CONTRAST_DIR_RE.match(name)]
    if not contrast_dirs and any(parse_phr(name) is not None for name in entries):
        contrast_dirs = [""]  # sweep_dir is itself one contrast folder

    runs = []
    for contrast_dir in contrast_dirs:
        parent = os.path.join(sweep_dir, contrast_dir)
        group = []
        for name in sorted(os.listdir(parent)):
            run_dir = os.path.join(parent, name)
            if not os.path.isdir(run_dir) or parse_phr(name) is None:
                continue
            run = load_run(run_dir)
            if run is None:
                print("skip (no output.csv): {}".format(os.path.join(contrast_dir, name)))
                continue
            run["label"] = contrast_dir or os.path.basename(sweep_dir)
            group.append(run)

        # run_metadata.txt is written when a run finishes, so a run that was
        # killed part way through has none and arrives here without a contrast
        # or matrix phase. Its curve is still good, so it inherits what the rest
        # of its folder agrees on rather than being thrown away.
        for key in ("contrast", "matrix"):
            known = {run[key] for run in group if run[key] is not None}
            shared = known.pop() if len(known) == 1 else None
            if key == "contrast" and shared is None:
                shared = contrast_from_dirname(contrast_dir)
            for run in group:
                if run[key] is None:
                    run[key] = shared

        for run in group:
            if run["contrast"] is None:
                print("skip (no contrast): {}".format(
                    os.path.join(contrast_dir, os.path.basename(run["dir"]))))
                continue
            runs.append(run)

    if not runs:
        raise SystemExit("No simulation folders found under {}".format(sweep_dir))
    runs.sort(key=lambda run: (run["contrast"], run["phr"]))
    print("read {} run(s) from {}".format(len(runs), sweep_dir))
    return runs


# ------------------------------------------------------- neat-matrix reference
def _analytic_matrix_p11(f11, young, poisson):
    """Incompressible neo-Hookean fallback: P11 = 2 C1 (F11 - 1/F11^2)."""
    c1 = young / 4.0 / (1.0 + poisson)
    return 2.0 * c1 * (f11 - 1.0 / f11 ** 2)


def matrix_p11(f11, model, young, poisson):
    """P11 of the *unfilled* matrix at stretch f11 under the sweep's load case.

    Solves the homogeneous uniaxial state -- F = diag(f11, l, l), lateral faces
    traction free (P22 = 0), pressure y tied to the volume change by the mixed
    formulation's constraint 1 - J + y/D = 0 -- with the same constitutive
    routine the solver used, so the ratio taken against it compares like with
    like.
    """
    try:
        ensure_import_paths()
        from scipy.optimize import fsolve
        module = importlib.import_module(
            "fg.constitutive_incompressible.{}".format(str(model).split(".")[0])
        )
        umat_field = module.umat_field
    except Exception as error:  # solver half not importable on this machine
        print("warn: using the analytic neat-matrix stress ({})".format(error))
        return float(_analytic_matrix_p11(f11, young, poisson))

    kappa_inv = 0.0 if poisson == 0.5 else 3.0 * (1 - 2 * poisson) / young

    def residual(unknowns):
        lateral, pressure = unknowns
        f = np.diag([f11, lateral, lateral])[None]
        P, _, _, _ = umat_field(f, np.array([pressure]), [young, poisson],
                                need_tangent=False)
        return [P[0, 1, 1], 1.0 - float(np.linalg.det(f[0])) + pressure * kappa_inv]

    lateral, pressure = fsolve(residual, [1.0 / np.sqrt(f11), 0.0])
    f = np.diag([f11, lateral, lateral])[None]
    P, _, _, _ = umat_field(f, np.array([pressure]), [young, poisson], need_tangent=False)
    if abs(P[0, 1, 1]) > 1e-6 * max(1.0, abs(P[0, 0, 0])):
        raise RuntimeError("neat-matrix solve did not reach a traction-free state")
    return float(P[0, 0, 0])


def stress_at_strain(run, strain, key="p11"):
    """run[key] interpolated at F11 = 1 + strain, or None if the run stopped earlier.

    `key` is "p11" for the nominal (first Piola-Kirchhoff) stress -- force per
    undeformed area, the quantity an experimental tensile curve plots -- or
    "sigma11" for the true (Cauchy) stress. The undeformed state
    (F11 = 1, stress = 0) is prepended so a target below the first increment is
    interpolated rather than dropped.
    """
    stress = run.get(key)
    if stress is None:
        return None
    f11 = np.concatenate(([1.0], run["f11"]))
    stress = np.concatenate(([0.0], stress))
    target = 1.0 + strain
    if target > f11[-1] + 1e-9:
        return None
    return float(np.interp(target, f11, stress))


def p11_at_strain(run, strain):
    """P11 interpolated at F11 = 1 + strain, or None if the run stopped earlier."""
    return stress_at_strain(run, strain, "p11")


def resolve_strain(runs, requested):
    """The stretch level to compare at: a number, or the largest common one."""
    if requested != "common":
        return float(requested)
    strain = min(run["max_strain"] for run in runs)
    print("common strain across all {} runs: {:.4f}".format(len(runs), strain))
    return strain


def reinforcement_rows(runs, strain):
    """(rows, skipped) with the reinforcement of every run at one stretch.

    A row is {phr, contrast, label, p11, matrix_p11, reinforcement, complete,
    matrix}; the runs that stopped before `strain` are returned separately
    instead of being silently dropped.
    """
    reference_cache = {}
    rows, skipped = [], []
    for run in runs:
        p11 = p11_at_strain(run, strain)
        if p11 is None:
            skipped.append(run)
            continue
        model, young, poisson = run["matrix"] or FALLBACK_MATRIX
        key = (model, young, poisson)
        if key not in reference_cache:
            reference_cache[key] = matrix_p11(1.0 + strain, model, young, poisson)
        reference = reference_cache[key]
        rows.append({
            "phr": run["phr"], "contrast": run["contrast"], "label": run["label"],
            "p11": p11, "matrix_p11": reference, "reinforcement": p11 / reference,
            "complete": run["complete"], "matrix": key,
        })
    return rows, skipped
