# -*- coding: utf-8 -*-
"""Single-case traction runner."""

from run_case import run_case
from sphere_voxelizer import voxelize_and_save
import os

BASE_PATH = "/home/jovyan/mixed-fft-rubber"

SPHERES_PATH = "3D_samples/spheres/1_spheres.npz"
N_voxels = 31
VOXEL_PATH = "3D_samples/voxels/1_voxel.npz"

VOXEL_PATH = voxelize_and_save(
    SPHERES_PATH,
    N=N_voxels,
    output_path=VOXEL_PATH,
)


CASE = {
    "structure_path": os.path.join(BASE_PATH, VOXEL_PATH),
    "charge_path": "3D_samples/Charges/Neo_1.0_E10-100x.txt",
    "output_path": os.path.join(BASE_PATH, r"Results"),
    "N": N_voxels,
    "incre_list": [0.1]*10,
    "preconditioner": "reference",
    "diagnostics": False,
    "save_plots": True,
    "save_fields": True,
    "field_filename": "fields.vti",
    "plot_dpi": 200,
    "matrix_phase": 0,
    "filler_phase": 1,
}


if __name__ == "__main__":
    run_case(CASE)
