#!/usr/bin/env python
"""
Generate 2D cross-section images from particle meshes, voxel NPZ files, or
sphere-parameter NPZ files.
"""

import argparse
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

try:
    from microstructure_generation.file_viewers.view_npz_3d import load_npz_content
except ImportError:
    from file_viewers.view_npz_3d import load_npz_content

DEFAULT_STRUCTURE_INPUT = "Structures/3D_structures/Spheres/1_spheres.npz"
DEFAULT_AXIS = "z"
DEFAULT_RESOLUTION = 256
DEFAULT_NUM_SECTIONS = 10
DEFAULT_SPHERE_NPZ_GRID = (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION, DEFAULT_RESOLUTION)
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "cross_sections")


def resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def _sanitize_grid_shape(voxel_grid_shape):
    if len(voxel_grid_shape) != 3:
        raise ValueError("voxel_grid_shape must have 3 entries (Nx, Ny, Nz).")
    nx, ny, nz = [int(v) for v in voxel_grid_shape]
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("voxel_grid_shape values must all be positive.")
    return nx, ny, nz


def voxelize_spheres_to_phase(sphere_centers, sphere_radii, box_size, voxel_grid_shape):
    centers = np.asarray(sphere_centers, dtype=float)
    radii = np.asarray(sphere_radii, dtype=float).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("sphere_centers must have shape (N, 3).")
    if radii.size != centers.shape[0]:
        raise ValueError("sphere_radii must have length N.")

    nx, ny, nz = _sanitize_grid_shape(voxel_grid_shape)
    box_size = float(box_size)
    phase = np.zeros((nx, ny, nz), dtype=np.uint8)
    if centers.shape[0] == 0:
        return phase

    dx = box_size / float(nx)
    dy = box_size / float(ny)
    dz = box_size / float(nz)
    x_coords = (np.arange(nx, dtype=float) + 0.5) * dx
    y_coords = (np.arange(ny, dtype=float) + 0.5) * dy
    z_coords = (np.arange(nz, dtype=float) + 0.5) * dz

    for center, radius in zip(centers, radii):
        cx, cy, cz = center
        radius_sq = float(radius * radius)

        ix0 = max(0, int(np.floor((cx - radius) / dx)))
        iy0 = max(0, int(np.floor((cy - radius) / dy)))
        iz0 = max(0, int(np.floor((cz - radius) / dz)))
        ix1 = min(nx - 1, int(np.floor((cx + radius) / dx)))
        iy1 = min(ny - 1, int(np.floor((cy + radius) / dy)))
        iz1 = min(nz - 1, int(np.floor((cz + radius) / dz)))

        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue

        local_x = x_coords[ix0 : ix1 + 1] - cx
        local_y = y_coords[iy0 : iy1 + 1] - cy
        local_z = z_coords[iz0 : iz1 + 1] - cz
        dist_sq = (
            local_x[:, None, None] ** 2
            + local_y[None, :, None] ** 2
            + local_z[None, None, :] ** 2
        )
        phase[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] |= (
            dist_sq <= radius_sq
        ).astype(np.uint8)

    return phase


def generate_cross_sections(particles, num_sections, axis="z", resolution=500):
    """
    Generate projected section images from particle meshes.

    This is the original mesh-based workflow and expects PyVista particle meshes.
    """
    if not particles:
        return []

    all_bounds = np.array([particle.bounds for particle in particles])
    bounds = [
        np.min(all_bounds[:, 0]), np.max(all_bounds[:, 1]),
        np.min(all_bounds[:, 2]), np.max(all_bounds[:, 3]),
        np.min(all_bounds[:, 4]), np.max(all_bounds[:, 5]),
    ]

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    min_val = bounds[axis_index * 2]
    max_val = bounds[axis_index * 2 + 1]
    section_boundaries = np.linspace(min_val, max_val, num_sections + 1)

    sections = []
    for i in range(num_sections):
        try:
            section_min = section_boundaries[i]
            section_max = section_boundaries[i + 1]
            section_image = np.zeros((resolution, resolution), dtype=bool)

            for particle in particles:
                points = particle.points
                mask = np.ones(len(points), dtype=bool)
                mask &= points[:, axis_index] >= section_min
                mask &= points[:, axis_index] <= section_max

                if not np.any(mask):
                    continue

                section_points = points[mask]
                if axis == "x":
                    proj_coords = section_points[:, [1, 2]]
                    x_grid = np.linspace(bounds[2], bounds[3], resolution)
                    y_grid = np.linspace(bounds[4], bounds[5], resolution)
                elif axis == "y":
                    proj_coords = section_points[:, [0, 2]]
                    x_grid = np.linspace(bounds[0], bounds[1], resolution)
                    y_grid = np.linspace(bounds[4], bounds[5], resolution)
                else:
                    proj_coords = section_points[:, [0, 1]]
                    x_grid = np.linspace(bounds[0], bounds[1], resolution)
                    y_grid = np.linspace(bounds[2], bounds[3], resolution)

                x_pixels = np.interp(proj_coords[:, 0], x_grid, np.arange(resolution))
                y_pixels = np.interp(proj_coords[:, 1], y_grid, np.arange(resolution))
                x_pixels = np.clip(np.round(x_pixels).astype(int), 0, resolution - 1)
                y_pixels = np.clip(np.round(y_pixels).astype(int), 0, resolution - 1)
                section_image[y_pixels, x_pixels] = True

            from scipy import ndimage

            section_image = ndimage.binary_dilation(section_image, structure=np.ones((3, 3)))
            sections.append(section_image)
        except Exception as e:
            print(f"Warning: Error generating section {i}: {str(e)}")
            sections.append(np.zeros((resolution, resolution), dtype=bool))

    return sections


def generate_cross_sections_from_phase(phase, num_sections, axis="z"):
    """
    Generate 2D section images directly from a 3D voxel phase array.

    Each output image is a max projection over one slab along the chosen axis.
    """
    phase = np.asarray(phase, dtype=bool)
    if phase.ndim != 3:
        raise ValueError(f"phase must be 3D, got shape {phase.shape}")

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    axis_size = phase.shape[axis_index]
    section_boundaries = np.linspace(0, axis_size, num_sections + 1, dtype=int)

    sections = []
    for i in range(num_sections):
        start = section_boundaries[i]
        end = section_boundaries[i + 1]
        if end <= start:
            end = min(axis_size, start + 1)

        if axis == "x":
            slab = phase[start:end, :, :]
        elif axis == "y":
            slab = phase[:, start:end, :]
        else:
            slab = phase[:, :, start:end]

        if slab.size == 0:
            if axis == "x":
                section = np.zeros((phase.shape[1], phase.shape[2]), dtype=bool)
            elif axis == "y":
                section = np.zeros((phase.shape[0], phase.shape[2]), dtype=bool)
            else:
                section = np.zeros((phase.shape[0], phase.shape[1]), dtype=bool)
        else:
            section = np.any(slab, axis=axis_index)

        sections.append(section)

    return sections


def resize_sections(sections, resolution):
    """
    Resize boolean section images to a square output resolution.
    """
    from scipy import ndimage

    resized = []
    for section in sections:
        section = np.asarray(section, dtype=bool)
        if section.shape == (resolution, resolution):
            resized.append(section)
            continue

        zoom_factors = (
            resolution / float(section.shape[0]),
            resolution / float(section.shape[1]),
        )
        resized_section = ndimage.zoom(section.astype(float), zoom=zoom_factors, order=0) > 0.5
        resized.append(resized_section)

    return resized


def load_structure_for_cross_sections(structure_input, sphere_npz_grid=DEFAULT_SPHERE_NPZ_GRID):
    """
    Load either:
    - a saved particle structure name / particle `.vtp` set
    - a voxel `.npz`
    - a sphere-parameter `.npz`

    Returns:
        tuple: (source_kind, data, structure_name)
    """
    structure_input = (
        os.path.abspath(structure_input)
        if structure_input.lower().endswith(".npz")
        else structure_input
    )

    if isinstance(structure_input, str) and structure_input.lower().endswith(".npz"):
        content_kind, content, meta = load_npz_content(structure_input)
        structure_name = os.path.splitext(os.path.basename(structure_input))[0]

        if content_kind == "phase":
            return "phase", np.asarray(content, dtype=bool), structure_name

        centers, radii = content
        box_size = float(np.asarray(meta.get("box_size", [1.0]), dtype=float).reshape(-1)[0])
        phase = voxelize_spheres_to_phase(
            sphere_centers=centers,
            sphere_radii=radii,
            box_size=box_size,
            voxel_grid_shape=sphere_npz_grid,
        )
        return "phase", phase.astype(bool), structure_name

    try:
        from microstructure_generation.combined_particle_models import load_particle_structure
    except ImportError:
        from combined_particle_models import load_particle_structure

    particles, _, _ = load_particle_structure(structure_input)
    return "particles", particles, structure_input


def save_cross_sections(sections, base_path, structure_name, create_subfolder=True):
    """
    Save the cross-sectional images.
    """
    from PIL import Image

    if create_subfolder:
        output_dir = os.path.join(base_path, structure_name)
    else:
        output_dir = base_path

    os.makedirs(output_dir, exist_ok=True)

    for i, section in enumerate(sections):
        try:
            # Save exact-resolution 1-bit images:
            # filler/structure -> white, background -> black
            binary_section = np.asarray(section, dtype=bool)
            image_array = np.where(binary_section, 255, 0).astype(np.uint8)
            image = Image.fromarray(image_array, mode="L").convert("1")
            filename = os.path.join(output_dir, f"section_{i:03d}.png")
            image.save(filename)
        except Exception as e:
            print(f"Warning: Error saving section {i}: {str(e)}")


def run(args):
    structure_input = resolve_repo_path(args.structure_input)
    output_dir = resolve_repo_path(args.output_dir)
    sphere_npz_grid = tuple(args.sphere_npz_grid)

    source_kind, data, structure_name = load_structure_for_cross_sections(
        structure_input,
        sphere_npz_grid=sphere_npz_grid,
    )

    if source_kind == "particles" and not data:
        print(f"Could not load structure: {structure_input}")
        return []

    if args.num_sections <= 0:
        raise ValueError("num_sections must be positive.")

    os.makedirs(output_dir, exist_ok=True)

    print("Generating sections...")
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

    print("Saving sections...")
    save_cross_sections(
        sections,
        output_dir,
        structure_name=structure_name,
        create_subfolder=True,
    )
    print("Wrote %d sections to %s" % (len(sections), output_dir))
    return sections


def build_parser():
    parser = argparse.ArgumentParser(description="Generate 2D cross-section images from a 3D structure.")
    parser.add_argument("--structure-input", default=DEFAULT_STRUCTURE_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-sections", type=int, default=DEFAULT_NUM_SECTIONS)
    parser.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_AXIS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--sphere-npz-grid", nargs=3, type=int, default=list(DEFAULT_SPHERE_NPZ_GRID))
    return parser


def main(cli_args=None):
    run(build_parser().parse_args(cli_args))


if __name__ == "__main__":
    main()
