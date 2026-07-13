#!/usr/bin/env python
"""
Calculate the PHR of a saved 3D microstructure NPZ file.

Supported NPZ formats:
- voxel phase files with a `phase` array
- sphere parameter files with `centers` and `radii` arrays
"""

import argparse
import os

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

DEFAULT_INPUT_PATH = os.path.join(
    REPO_ROOT,
    "Structures",
    "3D_structures",
    "Voxolized",
    "1_spheres_voxel.npz",
)

# Densities are only used as a ratio, so any consistent units are fine.
DEFAULT_FILLER_DENSITY = 1.8
DEFAULT_RUBBER_DENSITY = 0.92
DEFAULT_PHASE_KEY = "phase"


def resolve_input_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def scalar_from_npz(data, key, default=None):
    if key not in data.files:
        return default
    values = np.asarray(data[key], dtype=float).reshape(-1)
    if values.size == 0:
        return default
    return float(values[0])


def voxel_volume_from_npz(data, phase_key):
    phase = np.asarray(data[phase_key])
    if phase.ndim != 3:
        raise ValueError("Voxel phase array must be 3D, got shape %s." % (phase.shape,))

    filler_cells = int(np.count_nonzero(phase > 0.5))
    filler_fraction = float(filler_cells) / float(phase.size)

    box_size = scalar_from_npz(data, "box_size")
    if box_size is not None:
        total_volume = box_size ** 3
    elif "voxel_size" in data.files:
        voxel_size = np.asarray(data["voxel_size"], dtype=float).reshape(-1)
        if voxel_size.size == 1:
            voxel_volume = voxel_size[0] ** 3
        elif voxel_size.size == 3:
            voxel_volume = float(np.prod(voxel_size))
        else:
            raise ValueError("voxel_size must have 1 or 3 entries.")
        total_volume = float(phase.size) * voxel_volume
    else:
        total_volume = 1.0

    filler_volume = filler_fraction * total_volume
    return {
        "kind": "voxel phase",
        "total_volume": total_volume,
        "filler_volume": filler_volume,
        "filler_count": filler_cells,
        "total_count": int(phase.size),
    }


def sphere_volume_from_npz(data):
    centers = np.asarray(data["centers"], dtype=float)
    radii = np.asarray(data["radii"], dtype=float).reshape(-1)

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3), got %s." % (centers.shape,))
    if radii.size != centers.shape[0]:
        raise ValueError("centers and radii length mismatch: %d vs %d." % (centers.shape[0], radii.size))
    if np.any(radii < 0.0):
        raise ValueError("Sphere radii must be non-negative.")

    box_size = scalar_from_npz(data, "box_size")
    if box_size is None:
        raise ValueError("Sphere NPZ files need a box_size entry to calculate PHR.")

    total_volume = box_size ** 3
    filler_volume = float(np.sum((4.0 / 3.0) * np.pi * radii ** 3))
    return {
        "kind": "sphere parameters",
        "total_volume": total_volume,
        "filler_volume": filler_volume,
        "filler_count": int(radii.size),
        "total_count": None,
    }


def load_structure_volumes(path, phase_key=DEFAULT_PHASE_KEY):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    if phase_key in keys:
        return voxel_volume_from_npz(data, phase_key)
    if "centers" in keys and "radii" in keys:
        return sphere_volume_from_npz(data)

    raise KeyError(
        "Unsupported NPZ format. Expected either a '%s' array or ['centers', 'radii']. Available keys: %s"
        % (phase_key, sorted(data.files))
    )


def calculate_phr(filler_volume, total_volume, filler_density, rubber_density):
    rubber_volume = total_volume - filler_volume
    if rubber_volume <= 0.0:
        raise ValueError(
            "Rubber volume is not positive. Check the input structure or density/volume assumptions."
        )
    return 100.0 * (filler_volume * filler_density) / (rubber_volume * rubber_density)


def build_parser():
    parser = argparse.ArgumentParser(description="Calculate PHR for a voxel or sphere NPZ structure.")
    parser.add_argument(
        "--input",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help="Path to the NPZ file. Defaults to DEFAULT_INPUT_PATH in this script.",
    )
    parser.add_argument("--phase-key", default=DEFAULT_PHASE_KEY, help="Phase array key for voxel NPZ files.")
    parser.add_argument(
        "--filler-density",
        type=float,
        default=DEFAULT_FILLER_DENSITY,
        help="Filler density used for mass conversion.",
    )
    parser.add_argument(
        "--rubber-density",
        type=float,
        default=DEFAULT_RUBBER_DENSITY,
        help="Rubber density used for mass conversion.",
    )
    return parser


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    input_path = resolve_input_path(args.input)

    volumes = load_structure_volumes(input_path, phase_key=args.phase_key)
    total_volume = volumes["total_volume"]
    filler_volume = volumes["filler_volume"]
    filler_fraction = filler_volume / total_volume
    phr = calculate_phr(
        filler_volume=filler_volume,
        total_volume=total_volume,
        filler_density=args.filler_density,
        rubber_density=args.rubber_density,
    )

    print("Input: %s" % input_path)
    print("NPZ type: %s" % volumes["kind"])
    print("Filler volume: %.8g" % filler_volume)
    print("Total volume: %.8g" % total_volume)
    print("Filler volume fraction: %.8g" % filler_fraction)
    print("Filler density: %.8g" % args.filler_density)
    print("Rubber density: %.8g" % args.rubber_density)
    if volumes["total_count"] is not None:
        print("Filled voxels: %d / %d" % (volumes["filler_count"], volumes["total_count"]))
    else:
        print("Spheres: %d" % volumes["filler_count"])
    print("PHR: %.8g" % phr)


if __name__ == "__main__":
    main()
