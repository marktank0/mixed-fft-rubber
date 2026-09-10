# -*- coding: utf-8 -*-
"""Run a set of microstructures across a ladder of filler/matrix contrasts.

Every structure is run at every contrast, all cases in parallel, from a single
YAML config:

    python FFT_simulation/contrast_sweep.py FFT_simulation/Run_configs/contrast_sweep.yaml

The config is an ordinary run config (same `defaults`, `execution`, `phases`,
`outputs` sections as batch_run.py) plus a `sweep:` section that replaces
`batch:`/`cases:`. The contrast ladder is expressed as a list of filler Young's
moduli against a fixed matrix; one charge file per filler modulus is generated
in Run_configs/Charges/ if it does not exist yet, following the existing
`Neo_<strain>_E<matrix>-<filler>.txt` naming. An existing file of that name is
reused, but only after checking that it really describes the material pairing
the sweep asks for.

Cases are laid out one directory per contrast:

    <output_root>/E10-250/phr_20.49_id20_voxel_output/

so a whole contrast can be plotted in one call:

    python plot_p11_vs_phr.py Results/contrast_sweep/E10-250
"""

import argparse
import os

import numpy as np

import _bootstrap  # noqa: F401  (puts the repo root and FFT_simulation on sys.path)
from project_paths import CHARGES_DIR

import batch_run
from simulation_config import (
    ConfigError,
    SUPPORTED_ON_EXISTING,
    _build_batch_cases,
    load_config,
    resolve_base_path,
    resolve_path,
)

# Same layout as the hand-written charge files: two material lines (matrix,
# filler), the macroscopic load, and the per-component F/P control mask.
CHARGE_TEMPLATE = (
    "#first two lines: model:---0)model num 1) p1.. 2)p2...\n"
    "{model:.1f}\t{matrix_e:g}\t{matrix_nu:.2f}\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n"
    "{model:.1f}\t{filler_e:g}\t{filler_nu:.2f}\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n"
    "#charge dF\n"
    "{strain:.1f}\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n"
    "#(Charge type) P-1 or F-0: 0 = control this component by Fij, "
    "1 = control this component by average Pij\n"
    "0.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\n"
)

MODEL_PREFIX = {1.0: "Neo", 2.0: "Mooney"}

SWEEP_DEFAULTS = {
    "model": 1.0,
    "matrix_e": 10.0,
    "matrix_nu": 0.48,
    "filler_nu": 0.30,
    "strain": 1.0,
}


def sweep_settings(config):
    """Read and validate the `sweep:` section."""
    sweep = config.get("sweep")
    if not sweep:
        raise ConfigError(
            "contrast_sweep.py needs a top-level 'sweep:' section "
            "(filler_e + structures); use batch_run.py for a plain batch config."
        )

    settings = dict(SWEEP_DEFAULTS)
    for key in SWEEP_DEFAULTS:
        if sweep.get(key) is not None:
            settings[key] = float(sweep[key])

    filler_e = sweep.get("filler_e")
    if not filler_e:
        raise ConfigError("sweep.filler_e must list at least one filler Young's modulus.")
    settings["filler_e"] = [float(value) for value in filler_e]
    settings["charge_dir"] = sweep.get("charge_dir")
    settings["structures"] = sweep.get("structures")
    if not settings["structures"]:
        raise ConfigError("sweep.structures must be a list of .npz paths or a {glob: ...} mapping.")
    return settings


def charge_name(settings, filler_e):
    prefix = MODEL_PREFIX.get(settings["model"], "model{:g}".format(settings["model"]))
    # matches the existing hand-written files: Neo_1.0_E10-1000.txt
    return "{}_{:.1f}_E{:g}-{:g}.txt".format(
        prefix, settings["strain"], settings["matrix_e"], filler_e)


def contrast_tag(settings, filler_e):
    """Directory name for one rung of the ladder, e.g. 'E10-250'."""
    return "E{:g}-{:g}".format(settings["matrix_e"], filler_e)


def _expected_rows(settings, filler_e):
    return np.array([
        [settings["model"], settings["matrix_e"], settings["matrix_nu"], 0, 0, 0, 0, 0, 0],
        [settings["model"], filler_e, settings["filler_nu"], 0, 0, 0, 0, 0, 0],
        [settings["strain"], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=float)


def ensure_charge(settings, filler_e, charge_dir):
    """Return the charge file for one contrast, writing it if it is missing.

    A file that already exists is reused only if it describes the same material
    pairing and load case; a name collision with different physics is an error
    rather than a silently wrong run.
    """
    path = os.path.join(charge_dir, charge_name(settings, filler_e))
    expected = _expected_rows(settings, filler_e)

    if os.path.exists(path):
        found = np.loadtxt(path)
        if found.shape != expected.shape or not np.allclose(found, expected):
            raise ConfigError(
                "{} already exists but does not match the requested sweep "
                "(matrix E={:g} nu={:g}, filler E={:g} nu={:g}, strain {:g}). "
                "Rename or delete it, or point sweep.charge_dir elsewhere.".format(
                    path, settings["matrix_e"], settings["matrix_nu"],
                    filler_e, settings["filler_nu"], settings["strain"]))
        return path, False

    os.makedirs(charge_dir, exist_ok=True)
    text = CHARGE_TEMPLATE.format(
        model=settings["model"], matrix_e=settings["matrix_e"], matrix_nu=settings["matrix_nu"],
        filler_e=filler_e, filler_nu=settings["filler_nu"], strain=settings["strain"])
    tmp = path + ".tmp{}".format(os.getpid())
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return path, True


def sweep_structures(config, settings, base_path):
    """Absolute paths of the microstructures to sweep, in run order."""
    structures = settings["structures"]

    if isinstance(structures, dict):
        # same keys as batch.structures: glob / include / exclude / sort
        shim = {"batch": {"structures": structures}}
        return [case["structure_path"] for case in _build_batch_cases(shim, base_path)]

    if not isinstance(structures, list):
        raise ConfigError("sweep.structures must be a list or a {glob: ...} mapping.")

    paths = []
    for entry in structures:
        path = resolve_path(entry, base_path)
        if not os.path.isfile(path):
            raise ConfigError("sweep structure does not exist: {}".format(path))
        paths.append(path)
    return paths


def expand_sweep(config, base_path_override=None, filler_e_filter=None):
    """Turn a `sweep:` config into an ordinary `cases:` config.

    Returns (expanded_config, plan); the expanded config is handed to
    batch_run, so the sweep shares every path rule, defaults merge and
    execution setting with a normal batch run.
    """
    settings = sweep_settings(config)
    base_path = resolve_base_path(config, base_path_override=base_path_override)

    filler_values = settings["filler_e"]
    if filler_e_filter:
        wanted = [float(value) for value in filler_e_filter]
        missing = [value for value in wanted if value not in filler_values]
        if missing:
            raise ConfigError(
                "--filler-e {} not in sweep.filler_e {}".format(missing, filler_values))
        filler_values = wanted

    charge_dir = settings["charge_dir"]
    charge_dir = resolve_path(charge_dir, base_path) if charge_dir else CHARGES_DIR

    structures = sweep_structures(config, settings, base_path)

    experiment = config.get("experiment", {}) or {}
    output_root = experiment.get("output_root")
    if not output_root:
        raise ConfigError("experiment.output_root is required for a sweep.")

    cases = []
    charges = []
    for filler_e in filler_values:
        charge, created = ensure_charge(settings, filler_e, charge_dir)
        charges.append((filler_e, charge, created))
        tag = contrast_tag(settings, filler_e)
        for structure in structures:
            stem = os.path.splitext(os.path.basename(structure))[0]
            cases.append({
                "name": "{}__{}".format(stem, tag),
                "structure_path": structure,
                "charge": {"path": charge},
                # one directory per contrast, one run folder per structure
                "output_path": os.path.join(output_root, tag),
                "output_name": "{}_output".format(stem),
            })

    expanded = dict(config)
    expanded["run"] = dict(config.get("run", {}) or {}, mode="cases")
    expanded["cases"] = cases
    expanded.pop("batch", None)

    plan = {
        "structures": structures,
        "charges": charges,
        "cases": len(cases),
        "matrix_e": settings["matrix_e"],
        "base_path": base_path,
    }
    return expanded, plan


def print_plan(plan):
    print("sweep: {} structures x {} contrasts = {} runs".format(
        len(plan["structures"]), len(plan["charges"]), plan["cases"]))
    for filler_e, charge, created in plan["charges"]:
        print("  filler E={:<8g} contrast {:<7g} {} {}".format(
            filler_e, filler_e/plan["matrix_e"], os.path.basename(charge),
            "(generated)" if created else "(existing)"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run a contrast sweep from a YAML sweep config.")
    parser.add_argument("config", help="Path to a YAML config with a 'sweep:' section.")
    parser.add_argument("--base-path", help="Override config base_path, useful on servers.")
    parser.add_argument("--max-workers", type=int, help="Override execution.max_workers.")
    parser.add_argument("--filler-e", type=float, nargs="*",
                        help="Run only these filler moduli (subset of sweep.filler_e).")
    parser.add_argument("--on-existing", choices=SUPPORTED_ON_EXISTING,
                        help="Override execution.on_existing.")
    parser.add_argument("--terminal-output", action="store_true",
                        help="Print solver progress here instead of each case's run.log.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the expanded runs without solving anything.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    expanded, plan = expand_sweep(config, base_path_override=args.base_path,
                                  filler_e_filter=args.filler_e)
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


if __name__ == "__main__":
    main()
