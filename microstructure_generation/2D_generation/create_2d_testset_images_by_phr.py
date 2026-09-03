#!/usr/bin/env python
"""
Generate 2D slice images from TestSet 3D structures and prefix each image with
its own 2D PHR.
"""

import argparse
import csv
import os
import sys

import numpy as np

# This script lives in microstructure_generation/2D_generation/, so the
# repository root is two levels up (the generation package is one).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(GENERATION_DIR, os.pardir))
# Put both on sys.path so the sibling-module imports below resolve whether
# this file is run as a script or imported.
for _path in (SCRIPT_DIR, GENERATION_DIR, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# calculate_phr lives one level up, in microstructure_generation/.
try:
    from microstructure_generation.calculate_phr import (
        DEFAULT_FILLER_DENSITY,
        DEFAULT_RUBBER_DENSITY,
        calculate_phr,
    )
except ImportError:
    from calculate_phr import (
        DEFAULT_FILLER_DENSITY,
        DEFAULT_RUBBER_DENSITY,
        calculate_phr,
    )

# cross_section_generator sits next to this file in 2D_generation/, which
# can never be a package name (it starts with a digit), so there is no
# package-qualified form for it - it is always imported from SCRIPT_DIR.
from cross_section_generator import (
    DEFAULT_AXIS,
    DEFAULT_NUM_SECTIONS,
    DEFAULT_RESOLUTION,
    DEFAULT_SPHERE_NPZ_GRID,
    generate_cross_sections,
    generate_cross_sections_from_phase,
    load_structure_for_cross_sections,
    resize_sections,
)


DEFAULT_INPUT_DIR = os.path.join(REPO_ROOT, "Structures", "3D_structures", "Spheres")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "Structures", "2D_images")
DEFAULT_MANIFEST = os.path.join(DEFAULT_OUTPUT_DIR, "image_manifest.csv")


def resolve_input_dir(input_dir):
    if os.path.isdir(input_dir):
        return input_dir
    return input_dir


def list_npz_files(input_dir):
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(".npz") and os.path.isfile(os.path.join(input_dir, name))
    ]


def phr_prefix(phr):
    rounded = float("%.2g" % float(phr))
    return ("%.2f" % rounded).zfill(6)


def calculate_2d_phr(section, filler_density, rubber_density):
    binary = np.asarray(section, dtype=bool)
    filler_area = float(np.count_nonzero(binary))
    total_area = float(binary.size)
    return calculate_phr(
        filler_volume=filler_area,
        total_volume=total_area,
        filler_density=filler_density,
        rubber_density=rubber_density,
    )


def save_section_image(section, output_path):
    from PIL import Image

    image_array = np.where(np.asarray(section, dtype=bool), 255, 0).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L").convert("1")
    image.save(output_path)


def generate_sections_for_structure(npz_path, args):
    source_kind, data, structure_name = load_structure_for_cross_sections(
        npz_path,
        sphere_npz_grid=tuple(args.sphere_npz_grid),
    )

    if source_kind == "particles":
        sections = generate_cross_sections(
            data,
            num_sections=args.num_sections,
            axis=args.axis,
            resolution=args.resolution,
        )
    else:
        sections = generate_cross_sections_from_phase(
            data,
            num_sections=args.num_sections,
            axis=args.axis,
        )
        sections = resize_sections(sections, args.resolution)

    return structure_name, sections


def write_manifest(rows, manifest_path):
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "source_structure",
                "section_index",
                "phr",
                "phr_prefix",
                "filler_density",
                "rubber_density",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args):
    input_dir = resolve_input_dir(os.path.abspath(args.input_dir))
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    npz_files = list_npz_files(input_dir)
    if not npz_files:
        raise IOError("No .npz files found in %s" % input_dir)

    manifest_rows = []
    for npz_index, npz_path in enumerate(npz_files, start=1):
        print("[%d/%d] %s" % (npz_index, len(npz_files), os.path.basename(npz_path)))
        structure_name, sections = generate_sections_for_structure(npz_path, args)

        for section_index, section in enumerate(sections):
            phr = calculate_2d_phr(
                section,
                filler_density=args.filler_density,
                rubber_density=args.rubber_density,
            )
            prefix = phr_prefix(phr)
            image_name = "%sphr_%s_section_%03d.png" % (prefix, structure_name, section_index)
            output_path = os.path.join(output_dir, image_name)
            save_section_image(section, output_path)
            manifest_rows.append(
                {
                    "image_name": image_name,
                    "source_structure": os.path.basename(npz_path),
                    "section_index": section_index,
                    "phr": phr,
                    "phr_prefix": prefix,
                    "filler_density": args.filler_density,
                    "rubber_density": args.rubber_density,
                }
            )

    write_manifest(manifest_rows, os.path.abspath(args.manifest))
    print("Wrote %d images to %s" % (len(manifest_rows), output_dir))
    print("Wrote manifest: %s" % os.path.abspath(args.manifest))


def build_parser():
    parser = argparse.ArgumentParser(description="Create 2D test images from 3D NPZ structures with 2D PHR prefixes.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--num-sections", type=int, default=DEFAULT_NUM_SECTIONS)
    parser.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_AXIS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--sphere-npz-grid", nargs=3, type=int, default=list(DEFAULT_SPHERE_NPZ_GRID))
    parser.add_argument("--filler-density", type=float, default=DEFAULT_FILLER_DENSITY)
    parser.add_argument("--rubber-density", type=float, default=DEFAULT_RUBBER_DENSITY)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
