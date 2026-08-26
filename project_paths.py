# -*- coding: utf-8 -*-
"""Canonical locations of everything in this repository.

The solver, its runners and its run configs live in ``FFT_simulation/``, while
the microstructures, the charge-independent tooling and the results stay at the
repository root. Both halves import each other, and the solver is launched from
several different working directories (repo root, FFT_simulation/, a batch
worker, a server job), so *nothing* may rely on the current working directory.

Every path here is derived from this file's own location, and
``ensure_import_paths()`` puts both halves of the repo on ``sys.path`` so that
``import fg`` (in FFT_simulation) and ``import simulation_config`` (at the root)
both work no matter where Python was started.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- the solver half -------------------------------------------------------
FFT_SIMULATION_DIR = os.path.join(PROJECT_ROOT, "FFT_simulation")
FG_DIR = os.path.join(FFT_SIMULATION_DIR, "fg")
RUN_CONFIGS_DIR = os.path.join(FFT_SIMULATION_DIR, "Run_configs")
CHARGES_DIR = os.path.join(RUN_CONFIGS_DIR, "Charges")

# --- the data / output half ------------------------------------------------
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "3D_samples")
VOXELS_DIR = os.path.join(SAMPLES_DIR, "voxels")
MICROSTRUCTURE_DIR = os.path.join(PROJECT_ROOT, "microstructure_generation")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results")


def ensure_import_paths():
    """Make both the repo root and FFT_simulation/ importable.

    Idempotent, and safe to call from a worker process: the entries are
    absolute, so a spawned child inherits usable paths.
    """
    for path in (FFT_SIMULATION_DIR, PROJECT_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    return sys.path


def project_path(*parts):
    """Join `parts` onto the repository root."""
    return os.path.join(PROJECT_ROOT, *parts)


def charge_path(name):
    """Absolute path of a charge file in FFT_simulation/Run_configs/Charges."""
    return os.path.join(CHARGES_DIR, name)


def results_path(*parts):
    """Absolute path inside the repository-root Results/ directory."""
    return os.path.join(RESULTS_DIR, *parts)


def run_config_path(name):
    """Absolute path of a YAML run config in FFT_simulation/Run_configs."""
    return os.path.join(RUN_CONFIGS_DIR, name)
