# -*- coding: utf-8 -*-
"""Voxelize sphere-parameter NPZ files into phase NPZ files."""

import os

import numpy as np

from project_paths import SAMPLES_DIR


def default_voxel_output_path(spheres_path, N):
    folder = os.path.dirname(spheres_path)
    base = os.path.splitext(os.path.basename(spheres_path))[0]
    if base.endswith("_spheres"):
        base = base[:-len("_spheres")]
    return os.path.join(folder, "{}_voxel_N{}.npz".format(base, N))


def load_spheres(spheres_path):
    with np.load(spheres_path, allow_pickle=False) as data:
        return {
            "centers": np.array(data["centers"], dtype=float, copy=True),
            "radii": np.array(data["radii"], dtype=float, copy=True),
            "origin": np.array(data["origin"], dtype=float, copy=True),
            "box_size": np.array(data["box_size"], dtype=float, copy=True),
            "notes": np.array(data["notes"], copy=True) if "notes" in data.files else np.array([""]),
        }


def voxelize_spheres(
    spheres_path,
    N,
    save=False,
    output_path=None,
    matrix_phase=0,
    filler_phase=1,
):
    """Voxelize spheres and optionally save a solver-ready NPZ.

    A voxel is assigned to the filler phase when its center lies inside at least
    one sphere. This matches the existing `3D_samples/voxels` convention.
    """
    spheres = load_spheres(spheres_path)
    centers = spheres["centers"]
    radii = spheres["radii"]
    origin = spheres["origin"]
    box_size = spheres["box_size"]
    length = float(box_size[0])
    voxel_size = np.array([length/N, length/N, length/N], dtype=np.float32)

    axes = [origin[i] + (np.arange(N) + 0.5)*voxel_size[i] for i in range(3)]
    X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")

    phase = np.full((N, N, N), matrix_phase, dtype=np.uint8)
    filler_mask = np.zeros((N, N, N), dtype=bool)
    for center, radius in zip(centers, radii):
        dx = X - center[0]
        dy = Y - center[1]
        dz = Z - center[2]
        filler_mask |= (dx*dx + dy*dy + dz*dz) <= radius*radius
    phase[filler_mask] = filler_phase

    result = {
        "phase": phase,
        "voxel_size": voxel_size,
        "origin": origin.astype(np.float32),
        "box_size": box_size.astype(np.float32),
        "notes": spheres["notes"],
        "format": np.array(["voxel_phase_v1"]),
    }

    if save:
        save_path = output_path or default_voxel_output_path(spheres_path, N)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.savez(save_path, **result)
        result["output_path"] = save_path

    return result


def voxelize_and_save(spheres_path, N, output_path=None, matrix_phase=0, filler_phase=1):
    result = voxelize_spheres(
        spheres_path,
        N,
        save=True,
        output_path=output_path,
        matrix_phase=matrix_phase,
        filler_phase=filler_phase,
    )
    return result["output_path"]


if __name__ == "__main__":
    path = voxelize_and_save(os.path.join(SAMPLES_DIR, "spheres", "1_spheres.npz"), N=31)
    print(path)
