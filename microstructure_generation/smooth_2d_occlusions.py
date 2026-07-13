#!/usr/bin/env python
"""
Post-process 2D binary microstructure masks by filling narrow polymer gaps and
small polymer occlusions inside filler aggregates.

The project image convention is:
- white pixels: filler
- black pixels: polymer matrix

This utility targets thin black polymer slits between neighboring filler pixels
and black connected components that are fully enclosed by white filler. Those
features can create severe local stress concentrations in 2D FFT simulations
even when they are tiny image-generation artifacts.
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy import ndimage


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

DEFAULT_INPUT_DIR = os.path.join(REPO_ROOT, "Structures", "2D_test_set_smoothed")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "Structures", "2D_test_set_occlusion_smoothed")
DEFAULT_MANIFEST = os.path.join(DEFAULT_OUTPUT_DIR, "occlusion_smoothing_manifest.csv")
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_THRESHOLD = 127
DEFAULT_MAX_HOLE_AREA = 512
DEFAULT_CLOSING_RADIUS = 1


def resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def list_image_files(input_dir):
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, name))
        and name.lower().endswith(DEFAULT_IMAGE_EXTENSIONS)
    ]


def square_structure(radius):
    radius = int(radius)
    if radius <= 0:
        return None
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def load_filler_mask(image_path, threshold=DEFAULT_THRESHOLD, invert_phase=False):
    gray = np.asarray(Image.open(image_path).convert("L"))
    filler = gray >= int(threshold)
    if invert_phase:
        filler = ~filler
    return filler


def save_filler_mask(filler, output_path):
    image_array = np.where(np.asarray(filler, dtype=bool), 255, 0).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L").convert("1")
    image.save(output_path)


def fill_enclosed_polymer_occlusions(filler, max_hole_area=None):
    """
    Fill black polymer components that do not touch the image boundary.

    Args:
        filler: Boolean image where True means filler.
        max_hole_area: Maximum enclosed polymer component area to fill. Use
            None to fill every enclosed polymer component.

    Returns:
        tuple: (smoothed_filler, stats)
    """
    filler = np.asarray(filler, dtype=bool)
    polymer = ~filler
    labels, label_count = ndimage.label(polymer, structure=np.ones((3, 3), dtype=bool))
    if label_count == 0:
        return filler.copy(), {
            "filled_occlusion_count": 0,
            "filled_occlusion_pixels": 0,
            "largest_filled_occlusion": 0,
        }

    border_labels = set(
        np.unique(
            np.concatenate(
                [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
            )
        ).tolist()
    )

    fill_mask = np.zeros_like(filler, dtype=bool)
    filled_areas = []
    for label_id in range(1, label_count + 1):
        if label_id in border_labels:
            continue
        component = labels == label_id
        area = int(np.count_nonzero(component))
        if max_hole_area is None or area <= max_hole_area:
            fill_mask |= component
            filled_areas.append(area)

    smoothed = filler | fill_mask
    return smoothed, {
        "filled_occlusion_count": len(filled_areas),
        "filled_occlusion_pixels": int(np.count_nonzero(fill_mask)),
        "largest_filled_occlusion": max(filled_areas) if filled_areas else 0,
    }


def close_narrow_polymer_gaps(filler, closing_radius):
    """
    Add filler into narrow black slits without removing any existing filler.

    A radius-1 square closing fills black gaps roughly one to two pixels wide.
    The final union is intentional: classical binary closing can remove small
    white details, but this post-process should only add filler.
    """
    filler = np.asarray(filler, dtype=bool)
    if closing_radius <= 0:
        return filler.copy(), {"gap_closed_pixels": 0}

    closed = ndimage.binary_closing(
        filler,
        structure=square_structure(closing_radius),
    )
    gap_mask = closed & ~filler
    return filler | gap_mask, {"gap_closed_pixels": int(np.count_nonzero(gap_mask))}


def smooth_filler_mask(filler, max_hole_area, closing_radius=0):
    smoothed = np.asarray(filler, dtype=bool).copy()
    smoothed, gap_stats = close_narrow_polymer_gaps(smoothed, closing_radius)

    smoothed, occlusion_stats = fill_enclosed_polymer_occlusions(
        smoothed,
        max_hole_area=max_hole_area,
    )
    stats = {}
    stats.update(gap_stats)
    stats.update(occlusion_stats)
    return smoothed, stats


def calculate_change_stats(original, smoothed):
    original = np.asarray(original, dtype=bool)
    smoothed = np.asarray(smoothed, dtype=bool)
    added = smoothed & ~original
    removed = original & ~smoothed
    return {
        "original_filler_fraction": float(np.mean(original)),
        "smoothed_filler_fraction": float(np.mean(smoothed)),
        "changed_pixels": int(np.count_nonzero(original != smoothed)),
        "added_filler_pixels": int(np.count_nonzero(added)),
        "removed_filler_pixels": int(np.count_nonzero(removed)),
    }


def write_manifest(rows, manifest_path):
    fieldnames = [
        "image_name",
        "original_filler_fraction",
        "smoothed_filler_fraction",
        "changed_pixels",
        "added_filler_pixels",
        "removed_filler_pixels",
        "gap_closed_pixels",
        "filled_occlusion_count",
        "filled_occlusion_pixels",
        "largest_filled_occlusion",
        "max_hole_area",
        "closing_radius",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args):
    input_dir = resolve_repo_path(args.input_dir)
    output_dir = resolve_repo_path(args.output_dir)
    manifest_path = resolve_repo_path(args.manifest)
    os.makedirs(output_dir, exist_ok=True)

    image_files = list_image_files(input_dir)
    if not image_files:
        raise IOError("No images found in %s" % input_dir)

    max_hole_area = None if args.fill_all_enclosed else int(args.max_hole_area)
    rows = []
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        filler = load_filler_mask(
            image_path,
            threshold=args.threshold,
            invert_phase=args.invert_phase,
        )
        smoothed, occlusion_stats = smooth_filler_mask(
            filler,
            max_hole_area=max_hole_area,
            closing_radius=args.closing_radius,
        )

        output_path = os.path.join(output_dir, image_name)
        save_filler_mask(smoothed, output_path)

        row = {"image_name": image_name}
        row.update(calculate_change_stats(filler, smoothed))
        row.update(occlusion_stats)
        row["max_hole_area"] = "all" if max_hole_area is None else max_hole_area
        row["closing_radius"] = int(args.closing_radius)
        rows.append(row)

        print(
            "%s: closed %d gap px, filled %d enclosed occlusions (%d px), changed %d px"
            % (
                image_name,
                row["gap_closed_pixels"],
                row["filled_occlusion_count"],
                row["filled_occlusion_pixels"],
                row["changed_pixels"],
            )
        )

    write_manifest(rows, manifest_path)
    print("Wrote %d images to %s" % (len(rows), output_dir))
    print("Wrote manifest: %s" % manifest_path)
    return rows


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fill narrow polymer gaps and enclosed occlusions in binary 2D microstructure masks."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--invert-phase", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=DEFAULT_MAX_HOLE_AREA,
        help="Largest enclosed polymer component, in pixels, to fill.",
    )
    parser.add_argument(
        "--fill-all-enclosed",
        action="store_true",
        help="Fill every enclosed polymer component regardless of area.",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=DEFAULT_CLOSING_RADIUS,
        help="Additive square closing radius before occlusion filling; 1 closes roughly 1-2 px black slits, 0 disables it.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
