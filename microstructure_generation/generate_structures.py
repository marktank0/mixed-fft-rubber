# -*- coding: utf-8 -*-
"""YAML-driven runner for generating many 3D filler microstructures.

Reads a generation config (generation_phr_sweep.yaml), expands it into
per-structure specs, generates them (optionally in parallel), records the
resulting PHR of each, and writes a manifest.csv.

Usage:
    python generate_structures.py generation_phr_sweep.yaml
    python generate_structures.py <config> --dry-run
    python generate_structures.py <config> --max-workers 8 --mode cases
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Generation code lives here in microstructure_generation/; shared viewers live
# at the repo root. Put both on the path so worker imports resolve in the main
# process and in spawned worker processes alike.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
for _path in (SCRIPT_DIR, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generation_config import (
    SUPPORTED_MODES,
    build_specs,
    get_execution_settings,
    get_phr_settings,
    load_config,
    preview_name,
)

# Columns written to manifest.csv, in order.
_PARAM_COLUMNS = [
    "box_size",
    "model1.intensity",
    "model1.radius",
    "model2.intensity",
    "model2.min_r1",
    "model2.max_r1",
    "model3.intensity",
    "model3.radius",
]
_MANIFEST_COLUMNS = (
    ["index", "name", "seed", "status"]
    + _PARAM_COLUMNS
    + [
        "n_points",
        "n_final",
        "n_removed_floaters",
        "n_filled_voxels",
        "filler_fraction",
        "phr",
        "spheres_npz",
        "voxel_npz",
    ]
)


def _resolve_output_dir(output_dir):
    """Resolve output_dir the same way combined_particle_models does (vs repo root)."""
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.normpath(os.path.join(REPO_ROOT, output_dir))


def _get_dotted(data, dotted_key):
    node = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _run_one(job):
    """Worker: generate one structure, name it by PHR, save it. Runs in a subprocess.

    PHR, the final filename and the on_existing check all happen inside
    generate_and_save because the PHR (and thus the name) is only known after
    the structure has been generated and voxelized.
    """
    from combined_particle_models import generate_and_save

    spec = job["spec"]
    result = generate_and_save(
        spec,
        return_meshes=False,
        verbose=True,
        filler_density=job["filler_density"],
        rubber_density=job["rubber_density"],
        on_existing=job["on_existing"],
    )

    row = {
        "index": job["index"],
        "name": result["name"],
        "seed": result["seed"],
        "status": result["status"],
        "n_points": result["n_points"],
        "n_final": result["n_final"],
        "n_removed_floaters": result.get("n_removed_floaters", 0),
        "n_filled_voxels": result.get("n_filled_voxels", 0),
        "filler_fraction": result["filler_fraction"],
        "phr": result["phr"],
        "spheres_npz": result["saved_paths"].get("spheres_npz"),
        "voxel_npz": result["saved_paths"].get("voxel_npz"),
    }
    for column in _PARAM_COLUMNS:
        row[column] = _get_dotted(spec, column)
    return row


def _write_manifest(output_dir, rows):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    rows = sorted(rows, key=lambda row: row["index"])
    with open(manifest_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in _MANIFEST_COLUMNS})
    return manifest_path


def run_batch(jobs, max_workers):
    rows = []
    workers = min(max_workers, len(jobs))
    print("Generating {} structures with {} worker(s).".format(len(jobs), workers))

    if workers <= 1:
        for job in jobs:
            rows.append(_run_one(job))
        return rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def run_from_config(config_path, mode_override=None, max_workers_override=None, dry_run=False):
    config = load_config(config_path)
    execution = get_execution_settings(config, max_workers_override=max_workers_override)
    phr_settings = get_phr_settings(config)
    specs = build_specs(config, mode_override=mode_override)

    output_dir = _resolve_output_dir(config["output_dir"])

    if dry_run:
        mode = mode_override or (config.get("run", {}) or {}).get("mode", "sweep")
        print("mode: {}".format(mode))
        print("output_dir: {}".format(output_dir))
        print("max_workers: {}".format(execution["max_workers"]))
        print("on_existing: {}".format(execution["on_existing"]))
        print("structures: {}".format(len(specs)))
        for index, spec in enumerate(specs):
            print("- {:04d} {} (seed={})".format(index, preview_name(spec), spec.get("seed")))
        return []

    jobs = [
        {
            "index": index,
            "spec": spec,
            "filler_density": phr_settings["filler_density"],
            "rubber_density": phr_settings["rubber_density"],
            "on_existing": execution["on_existing"],
        }
        for index, spec in enumerate(specs)
    ]

    rows = run_batch(jobs, execution["max_workers"])

    if execution["save_manifest"] and rows:
        manifest_path = _write_manifest(output_dir, rows)
        print("manifest written to {}".format(manifest_path))

    generated = sum(1 for row in rows if row["status"] == "generated")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    print("Done: {} generated, {} skipped.".format(generated, skipped))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D microstructures from a YAML config.")
    parser.add_argument("config", help="Path to a generation YAML config.")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=None, help="Override run.mode.")
    parser.add_argument("--max-workers", type=int, help="Override execution.max_workers.")
    parser.add_argument("--dry-run", action="store_true", help="List the planned structures without generating.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_from_config(
        args.config,
        mode_override=args.mode,
        max_workers_override=args.max_workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
