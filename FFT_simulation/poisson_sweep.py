# -*- coding: utf-8 -*-
"""Run one microstructure across a ladder of Poisson's ratios (mixed formulation only).

This validates the mixed deformation-pressure formulation against

    M. Wang, K. Zhang, C. Chen, "A mixed FFT-Galerkin based approach for
    incompressible or slightly compressible hyperelastic solids under finite
    deformation", CMAME 396 (2022) 115092.

The load case is the paper's uniaxial traction, Eq. (61): F11 is prescribed
while the transverse faces are traction free (P22 = P33 = 0). Run on the
two-inclusion cell of the paper's Fig. 1 with the materials of its Tables 1-2,
this reproduces Fig. 4 - P11 against nu, with nu = 0.5 solved by the fully
incompressible branch (1/kappa = 0) of the same formulation - and, over the
wider nu ladder of Table 1, the iteration-count and kappa/mu panels of Fig. 2.

The paper's Fig. 2 proper is simple shear, which this solver cannot start:
det(I + gamma*e21) = 1 exactly, so the volumetric residual of the first
increment is zero, its Newton reference collapses to tol_abs and the line
search rejects every backtrack. Uniaxial traction has no such degeneracy
(det(F) = 1.2 at the first trial state) and tests the same thing - the
behaviour of the formulation as nu -> 0.5.

    python FFT_simulation/poisson_sweep.py FFT_simulation/Run_configs/poisson_sweep.yaml
    python FFT_simulation/poisson_sweep.py FFT_simulation/Run_configs/poisson_sweep.yaml --collect

The config is an ordinary run config (same `base_path`, `experiment`,
`execution`, `defaults` sections as batch_run.py) plus a `poisson_sweep:`
section that replaces `batch:`/`cases:`/`sweep:`.

Cases are laid out one directory per Poisson's ratio:

    <output_root>/nu0.499/<structure stem>_output/

`--collect` then walks those directories and writes `poisson_sweep.csv` and
`poisson_sweep.png` into <output_root>.
"""

import argparse
import json
import os

import numpy as np

import _bootstrap  # noqa: F401  (puts the repo root and FFT_simulation on sys.path)
from project_paths import CHARGES_DIR

import batch_run
from simulation_config import (
    ConfigError,
    SUPPORTED_ON_EXISTING,
    load_config,
    resolve_base_path,
    resolve_path,
)

# Same layout as the hand-written charge files: two material lines (soft, hard),
# the macroscopic load, and the per-component F/P control mask.
CHARGE_TEMPLATE = (
    "#first two lines: model:---0)model num 1) E-modulus 2) Poisson-ratio"
    " 3) Gamma (mooney-rivlin only)\n"
    "{model:.1f}\t{soft_e:g}\t{soft_nu:.6f}\t{gamma:g}\t0.0\t0.0\t0.0\t0.0\t0.0\n"
    "{model:.1f}\t{hard_e:g}\t{hard_nu:.6f}\t{gamma:g}\t0.0\t0.0\t0.0\t0.0\t0.0\n"
    "#charge dF (row-major F11 F12 F13 F21 F22 F23 F31 F32 F33)\n"
    "{dF}\n"
    "#(Charge type) P-1 or F-0: 0 = control this component by Fij, "
    "1 = control this component by average Pij\n"
    "{mask}\n"
)

MODEL_PREFIX = {1.0: "Neo", 2.0: "Mooney"}

# Fig. 1 of the paper: the unit cell [-0.5, 0.5]^3 at 63^3 voxels holding a
# sphere (X1-0.1)^2 + X2^2 + (X3-0.1)^2 <= 0.35^2 and an ellipsoid
# (X1+0.2)^2 + X2^2/4 + (X3+0.2)^2 <= 0.2^2 (semi-axes 0.2, 0.4, 0.2).
GEOMETRY_DEFAULTS = {
    "N": 63,
    "cell_size": 1.0,
    "sphere_center": (0.1, 0.0, 0.1),
    "sphere_radius": 0.35,
    "ellipsoid_center": (-0.2, 0.0, -0.2),
    "ellipsoid_semi_axes": (0.2, 0.4, 0.2),
}

SWEEP_DEFAULTS = {
    "model": 1.0,
    "soft_e": 0.26,
    "hard_e": 2.6,
    "gamma": 0.0,
    "stretch": 1.2,
}

# Table 1 (the wide ladder) extended with nu = 0.5, the incompressible point of
# Table 2. The four values of Fig. 4 - 0.49, 0.496, 0.499, 0.5 - are a subset.
POISSON_DEFAULT = [0.4, 0.42, 0.44, 0.46, 0.48, 0.49, 0.496, 0.499, 0.5]


# --- config ------------------------------------------------------------------

def sweep_settings(config):
    """Read and validate the `poisson_sweep:` section."""
    sweep = config.get("poisson_sweep")
    if not sweep:
        raise ConfigError(
            "poisson_sweep.py needs a top-level 'poisson_sweep:' section; use "
            "batch_run.py for a plain batch and contrast_sweep.py for a "
            "contrast ladder.")

    settings = dict(SWEEP_DEFAULTS)
    for key in SWEEP_DEFAULTS:
        if sweep.get(key) is not None:
            settings[key] = float(sweep[key])

    if settings["stretch"] <= 1.0:
        raise ConfigError(
            "poisson_sweep.stretch is the prescribed F11 and must exceed 1; "
            "got {}".format(settings["stretch"]))

    poisson = sweep.get("poisson", POISSON_DEFAULT)
    if not poisson:
        raise ConfigError("poisson_sweep.poisson must list at least one Poisson's ratio.")
    settings["poisson"] = [float(value) for value in poisson]
    for nu in settings["poisson"]:
        # nu = 0.5 is legal and means fully incompressible: the constitutive
        # model then sets 1/kappa = 0, which is the incompressible branch of
        # the mixed formulation (the fourth point of the paper's Fig. 4).
        if not 0.0 <= nu <= 0.5:
            raise ConfigError(
                "poisson_sweep.poisson values must lie in [0, 0.5]; got {}".format(nu))

    axis = int(sweep.get("axis", 1))
    if axis not in (1, 2, 3):
        raise ConfigError("poisson_sweep.axis must be 1, 2 or 3; got {!r}".format(axis))
    settings["axis"] = axis

    settings["charge_dir"] = sweep.get("charge_dir")
    settings["structure"] = sweep.get("structure")
    if not settings["structure"]:
        raise ConfigError("poisson_sweep.structure must name the .npz cell to solve.")

    geometry = dict(GEOMETRY_DEFAULTS)
    given = sweep.get("geometry") or {}
    if not isinstance(given, dict):
        raise ConfigError("poisson_sweep.geometry must be a mapping.")
    for key, value in given.items():
        if key not in GEOMETRY_DEFAULTS:
            raise ConfigError("Unknown poisson_sweep.geometry key {!r}".format(key))
        geometry[key] = value
    geometry["N"] = int(geometry["N"])
    geometry["cell_size"] = float(geometry["cell_size"])
    geometry["sphere_radius"] = float(geometry["sphere_radius"])
    for key in ("sphere_center", "ellipsoid_center", "ellipsoid_semi_axes"):
        geometry[key] = np.asarray(geometry[key], dtype=float)
        if geometry[key].shape != (3,):
            raise ConfigError("poisson_sweep.geometry.{} needs three numbers.".format(key))
    settings["geometry"] = geometry

    return settings


# --- geometry -----------------------------------------------------------------

def build_cell(geometry, matrix_phase=0, filler_phase=1):
    """Voxelize the paper's two-inclusion cell.

    A voxel belongs to the hard phase when its center lies inside the sphere or
    the ellipsoid. Offsets are wrapped to the nearest periodic image, so an
    inclusion that pokes through a face reappears on the opposite one - the
    cell the FFT solver actually assumes.
    """
    N = geometry["N"]
    length = geometry["cell_size"]
    spacing = length/N
    origin = -0.5*length

    axis = origin + (np.arange(N) + 0.5)*spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=0)   # (3,N,N,N)

    def offset(center):
        d = grid - np.asarray(center, dtype=float)[:, None, None, None]
        return d - length*np.round(d/length)          # minimum image

    ds = offset(geometry["sphere_center"])
    inside = np.einsum("i...,i...->...", ds, ds) <= geometry["sphere_radius"]**2

    de = offset(geometry["ellipsoid_center"])/geometry["ellipsoid_semi_axes"][:, None, None, None]
    inside |= np.einsum("i...,i...->...", de, de) <= 1.0

    phase = np.full((N, N, N), matrix_phase, dtype=np.uint8)
    phase[inside] = filler_phase
    return phase, float(spacing), float(origin), float(length)


def ensure_structure(settings, path):
    """Return the cell .npz, writing it if it is missing. (path, created)."""
    geometry = settings["geometry"]
    N = geometry["N"]

    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as data:
            if "phase" not in data.files:
                raise ConfigError("{} has no 'phase' array.".format(path))
            shape = data["phase"].shape
        if shape != (N, N, N):
            raise ConfigError(
                "{} holds a {} phase array but the sweep asks for N={}. Delete it "
                "or point poisson_sweep.structure elsewhere.".format(path, shape, N))
        return path, False

    phase, spacing, origin, length = build_cell(geometry)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        phase=phase,
        voxel_size=np.array([spacing]*3, dtype=np.float32),
        origin=np.array([origin]*3, dtype=np.float32),
        box_size=np.array([length]*3, dtype=np.float32),
        notes=np.array(["Wang, Zhang and Chen (2022) CMAME 115092, Fig. 1 cell"]),
        format=np.array(["voxel_phase_v1"]),
    )
    return path, True


# --- charge files -------------------------------------------------------------

def axial_component(settings):
    """Two-digit label of the driven diagonal component, e.g. '11'."""
    return "{0}{0}".format(settings["axis"])


def transverse_components(settings):
    """Two-digit labels of the two traction-free diagonal components."""
    return ["{0}{0}".format(k) for k in (1, 2, 3) if k != settings["axis"]]


def charge_name(settings, nu):
    prefix = MODEL_PREFIX.get(settings["model"], "model{:g}".format(settings["model"]))
    return "{}_uniax{:g}_F{}_E{:g}-{:g}_nu{:g}.txt".format(
        prefix, settings["stretch"], axial_component(settings),
        settings["soft_e"], settings["hard_e"], nu)


def poisson_tag(nu):
    """Directory name for one rung of the ladder, e.g. 'nu0.499'."""
    return "nu{:g}".format(nu)


def load_vectors(settings):
    """Row-major 9-vectors of the prescribed dF and of the F/P control mask.

    Eq. (61): F11 is prescribed (as the increment stretch - 1 away from the
    identity) and the transverse faces are traction free, so P22 and P33 are
    the stress-controlled components. Everything else is F controlled at zero,
    which holds the off-diagonals of F at the identity.
    """
    k = settings["axis"] - 1
    dF = np.zeros(9)
    dF[4*k] = settings["stretch"] - 1.0
    mask = np.zeros(9)
    for other in range(3):
        if other != k:
            mask[4*other] = 1.0
    return dF, mask


def _expected_rows(settings, nu):
    dF, mask = load_vectors(settings)
    return np.array([
        [settings["model"], settings["soft_e"], nu, settings["gamma"], 0, 0, 0, 0, 0],
        [settings["model"], settings["hard_e"], nu, settings["gamma"], 0, 0, 0, 0, 0],
        list(dF),
        list(mask),
    ], dtype=float)


def ensure_charge(settings, nu, charge_dir):
    """Return the charge file for one nu, writing it if it is missing.

    An existing file of that name is reused only if it really describes the same
    materials and load case; a name collision with different physics is an error
    rather than a silently wrong run.
    """
    path = os.path.join(charge_dir, charge_name(settings, nu))
    expected = _expected_rows(settings, nu)

    if os.path.exists(path):
        found = np.loadtxt(path)
        if found.shape != expected.shape or not np.allclose(found, expected):
            raise ConfigError(
                "{} already exists but does not match the requested sweep "
                "(soft E={:g}, hard E={:g}, nu={:g}, F{}={:g}). Rename or delete "
                "it, or point poisson_sweep.charge_dir elsewhere.".format(
                    path, settings["soft_e"], settings["hard_e"], nu,
                    axial_component(settings), settings["stretch"]))
        return path, False

    os.makedirs(charge_dir, exist_ok=True)
    dF, mask = load_vectors(settings)
    text = CHARGE_TEMPLATE.format(
        model=settings["model"], gamma=settings["gamma"],
        soft_e=settings["soft_e"], soft_nu=nu,
        hard_e=settings["hard_e"], hard_nu=nu,
        dF="\t".join("{:g}".format(value) for value in dF),
        mask="\t".join("{:.1f}".format(value) for value in mask),
    )
    tmp = path + ".tmp{}".format(os.getpid())
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return path, True


# --- expansion ----------------------------------------------------------------

def _output_root(config):
    experiment = config.get("experiment", {}) or {}
    output_root = experiment.get("output_root")
    if not output_root:
        raise ConfigError("experiment.output_root is required for a Poisson sweep.")
    return output_root


def expand_sweep(config, base_path_override=None, poisson_filter=None):
    """Turn a `poisson_sweep:` config into an ordinary `cases:` config.

    Returns (expanded_config, plan); the expanded config is handed to batch_run,
    so the sweep shares every path rule, defaults merge and execution setting
    with a normal batch run.
    """
    settings = sweep_settings(config)
    base_path = resolve_base_path(config, base_path_override=base_path_override)

    poisson_values = settings["poisson"]
    if poisson_filter:
        wanted = [float(value) for value in poisson_filter]
        missing = [value for value in wanted if value not in poisson_values]
        if missing:
            raise ConfigError(
                "--poisson {} not in poisson_sweep.poisson {}".format(missing, poisson_values))
        poisson_values = wanted

    charge_dir = settings["charge_dir"]
    charge_dir = resolve_path(charge_dir, base_path) if charge_dir else CHARGES_DIR

    structure, structure_created = ensure_structure(
        settings, resolve_path(settings["structure"], base_path))
    stem = os.path.splitext(os.path.basename(structure))[0]
    output_root = _output_root(config)

    # The grid the solver uses comes from solver.N, while the cell was
    # voxelized at geometry.N - a mismatch is a hard error deep inside
    # load_phase, so catch it here where the message can say which knob to turn.
    solver_N = int(((config.get("defaults", {}) or {}).get("solver", {}) or {}).get("N", 31))
    if solver_N != settings["geometry"]["N"]:
        raise ConfigError(
            "defaults.solver.N is {} but poisson_sweep.geometry.N is {}; the "
            "solver grid must match the voxelized cell.".format(
                solver_N, settings["geometry"]["N"]))

    cases = []
    charges = []
    for nu in poisson_values:
        charge, created = ensure_charge(settings, nu, charge_dir)
        charges.append((nu, charge, created))
        cases.append({
            "name": "{}__{}".format(stem, poisson_tag(nu)),
            "structure_path": structure,
            "charge": {"path": charge},
            "output_path": os.path.join(output_root, poisson_tag(nu)),
            "output_name": "{}_output".format(stem),
        })

    expanded = dict(config)
    expanded["run"] = dict(config.get("run", {}) or {}, mode="cases")
    expanded["cases"] = cases
    expanded.pop("batch", None)
    expanded.pop("sweep", None)

    plan = {
        "structure": structure,
        "structure_created": structure_created,
        "charges": charges,
        "settings": settings,
        "base_path": base_path,
    }
    return expanded, plan


def print_plan(plan):
    settings = plan["settings"]
    axial = axial_component(settings)
    print("poisson sweep: 1 structure x {} Poisson ratios = {} runs".format(
        len(plan["charges"]), len(plan["charges"])))
    print("  cell      {} {}".format(
        plan["structure"], "(generated)" if plan["structure_created"] else "(existing)"))
    print("  load      uniaxial traction, F{} = {:g}, P{} = P{} = 0".format(
        axial, settings["stretch"], *transverse_components(settings)))
    print("  materials soft E={:g} MPa, hard E={:g} MPa (contrast {:g})".format(
        settings["soft_e"], settings["hard_e"], settings["hard_e"]/settings["soft_e"]))
    for nu, charge, created in plan["charges"]:
        print("    nu={:<6g} kappa/mu {:>10.4g}  {} {}".format(
            nu, kappa_over_mu(nu), os.path.basename(charge),
            "(generated)" if created else "(existing)"))


# --- collection ---------------------------------------------------------------

def kappa_over_mu(nu):
    """(E/3(1-2nu)) / (E/2(1+nu)) - independent of E, so both phases share it."""
    if nu >= 0.5:
        return np.inf
    return 2.0*(1.0 + nu)/(3.0*(1.0 - 2.0*nu))


def _read_last_row(output_csv):
    with open(output_csv) as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if len(lines) < 2:
        return None, 0
    header = lines[0].split(",")
    values = [float(value) for value in lines[-1].split(",")]
    return dict(zip(header, values)), len(lines) - 1


def _read_stats(stats_path):
    if not os.path.exists(stats_path):
        return None, 0, 0
    with open(stats_path) as fh:
        stats = json.load(fh)
    krylov = sum(sum(inc.get("krylov_iterations", [])) for inc in stats.get("increments", []))
    newton = sum(inc.get("newton_iterations", 0) for inc in stats.get("increments", []))
    return stats.get("status"), int(krylov), int(newton)


def collect(config, base_path_override=None):
    """Gather every finished case into poisson_sweep.csv / .png under output_root."""
    settings = sweep_settings(config)
    base_path = resolve_base_path(config, base_path_override=base_path_override)
    output_root = resolve_path(_output_root(config), base_path)
    stem = os.path.splitext(os.path.basename(
        resolve_path(settings["structure"], base_path)))[0]

    axial = axial_component(settings)
    p_axial = "P{}".format(axial)
    f_axial = "F{}".format(axial)
    # The transverse components are prescribed to be traction free, so the two
    # P values below are a direct check that the stress control actually held.
    p_free = ["P{}".format(comp) for comp in transverse_components(settings)]

    rows = []
    for nu in settings["poisson"]:
        case_dir = os.path.join(output_root, poisson_tag(nu), "{}_output".format(stem))
        output_csv = os.path.join(case_dir, "output.csv")
        if not os.path.exists(output_csv):
            print("missing (not run yet): {}".format(output_csv))
            continue
        last, n_increments = _read_last_row(output_csv)
        if last is None:
            print("empty output.csv: {}".format(output_csv))
            continue
        status, krylov, newton = _read_stats(os.path.join(case_dir, "solver_stats.json"))
        row = {
            "nu": nu,
            "kappa_over_mu": kappa_over_mu(nu),
            f_axial: last[f_axial],
            p_axial: last[p_axial],
            "krylov_iterations": krylov,
            "newton_iterations": newton,
            "increments_written": n_increments,
            "status": status or "unknown",
        }
        for key in p_free:
            row[key] = last[key]
        rows.append(row)

    if not rows:
        raise SystemExit(
            "Nothing to collect under {} - run the sweep first.".format(output_root))

    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, "poisson_sweep.csv")
    fields = (["nu", "kappa_over_mu", f_axial, p_axial] + p_free
              + ["krylov_iterations", "newton_iterations", "increments_written", "status"])
    with open(csv_path, "w", newline="") as fh:
        fh.write(",".join(fields) + "\n")
        for row in rows:
            fh.write(",".join(
                row[key] if isinstance(row[key], str) else repr(row[key])
                for key in fields) + "\n")
    print("wrote {}".format(csv_path))

    png_path = _plot(rows, output_root, settings)
    if png_path:
        print("wrote {}".format(png_path))

    print("")
    print(" nu      kappa/mu   {:>12} {:>10}  {:>8}  status".format(
        p_axial, p_free[0], "krylov"))
    for row in rows:
        print(" {:<7g} {:>9.4g}  {:>12.5g} {:>10.2e}  {:>8d}  {}".format(
            row["nu"], row["kappa_over_mu"], row[p_axial], row[p_free[0]],
            row["krylov_iterations"], row["status"]))

    _report_incompressible_limit(rows, p_axial)
    return rows


def _report_incompressible_limit(rows, p_axial):
    """The paper's own accuracy check: P11(nu=0.499) vs P11(nu=0.5), 0.15%."""
    by_nu = {row["nu"]: row for row in rows if row["status"] == "converged"}
    if 0.499 not in by_nu or 0.5 not in by_nu:
        return
    near, incompressible = by_nu[0.499][p_axial], by_nu[0.5][p_axial]
    if incompressible == 0.0:
        return
    print("")
    print(" {0}(nu=0.499) = {1:.6g},  {0}(nu=0.5) = {2:.6g}".format(
        p_axial, near, incompressible))
    print(" relative difference {:.3g}%   (paper reports 0.15%)".format(
        100.0*abs(near - incompressible)/abs(incompressible)))


def _plot(rows, output_root, settings):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping poisson_sweep.png")
        return None

    axial = axial_component(settings)
    p_axial = "P{}".format(axial)
    ok = [row for row in rows if row["status"] == "converged"]
    bad = [row for row in rows if row["status"] != "converged"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    def series(ax, key):
        if ok:
            ax.plot([r["nu"] for r in ok], [r[key] for r in ok],
                    "o-", color="tab:blue", mfc="none", ms=7,
                    label="mixed formulation")
        if bad:
            ax.plot([r["nu"] for r in bad], [r[key] for r in bad],
                    "x", color="tab:red", ms=8, label="not converged")
        ax.set_xlabel("Poisson's ratio $\\nu$")
        ax.grid(alpha=0.3)

    series(axes[0], p_axial)
    # nu = 0.5 is solved by a different branch of the formulation (1/kappa = 0),
    # which is the whole point of the comparison in the paper's Fig. 4.
    incompressible = [r for r in ok if r["nu"] >= 0.5]
    if incompressible:
        axes[0].plot([r["nu"] for r in incompressible],
                     [r[p_axial] for r in incompressible],
                     "*", color="tab:red", ms=14, label="incompressible ($\\nu = 0.5$)")
    axes[0].set_ylabel("average $P_{{{}}}$ (MPa)".format(axial))
    axes[0].set_title("(a) average $P_{{{}}}$ at $F_{{{}}} = {:g}$".format(
        axial, axial, settings["stretch"]))
    axes[0].legend(loc="best", fontsize=9)

    series(axes[1], "krylov_iterations")
    axes[1].set_ylabel("total Krylov iterations")
    axes[1].set_title("(b) linear-solver iterations")

    finite = [r for r in rows if np.isfinite(r["kappa_over_mu"])]
    axes[2].plot([r["nu"] for r in finite], [r["kappa_over_mu"] for r in finite],
                 "o-", color="tab:green", mfc="none", ms=7)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Poisson's ratio $\\nu$")
    axes[2].set_ylabel("$\\kappa/\\mu$")
    axes[2].set_title("(c) $\\kappa/\\mu$ ratio")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle("Uniaxial traction, mixed formulation - Wang, Zhang and Chen "
                 "(2022) Fig. 4, E {:g}/{:g} MPa".format(
                     settings["soft_e"], settings["hard_e"]))
    fig.tight_layout()
    png_path = os.path.join(output_root, "poisson_sweep.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path


# --- cli ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Poisson's-ratio sweep (mixed formulation) from a YAML config.")
    parser.add_argument("config", help="Path to a YAML config with a 'poisson_sweep:' section.")
    parser.add_argument("--base-path", help="Override config base_path, useful on servers.")
    parser.add_argument("--max-workers", type=int, help="Override execution.max_workers.")
    parser.add_argument("--poisson", type=float, nargs="*",
                        help="Run only these Poisson ratios (subset of poisson_sweep.poisson).")
    parser.add_argument("--on-existing", choices=SUPPORTED_ON_EXISTING,
                        help="Override execution.on_existing.")
    parser.add_argument("--terminal-output", action="store_true",
                        help="Print solver progress here instead of each case's run.log.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the expanded runs without solving anything.")
    parser.add_argument("--collect", action="store_true",
                        help="Do not solve; gather finished runs into the csv / png.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.collect:
        collect(config, base_path_override=args.base_path)
        return

    expanded, plan = expand_sweep(config, base_path_override=args.base_path,
                                  poisson_filter=args.poisson)
    print_plan(plan)

    batch_run.run_from_config(
        args.config,
        config=expanded,
        base_path_override=args.base_path,
        max_workers_override=args.max_workers,
        log_to_file_override=False if args.terminal_output else None,
        on_existing_override=args.on_existing,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("")
        print("collecting results...")
        collect(config, base_path_override=args.base_path)


if __name__ == "__main__":
    main()
