# -*- coding: utf-8 -*-
"""Single-case traction runner."""

from run_case import run_case
from sphere_voxelizer import voxelize_and_save

SPHERES_PATH = "3D_samples/spheres/1_spheres.npz"
N_voxels = 63
VOXEL_PATH = "3D_samples/voxels_63/1_voxel_N63.npz"

VOXEL_PATH = voxelize_and_save(
    SPHERES_PATH,
    N=N_voxels,
    output_path=VOXEL_PATH,
)

CASE = {
    "structure_path": VOXEL_PATH,
    "charge_path": r"3D_samples\Charges\Neo_1.0_E10-100x.txt",
    "output_path": r"Results",
    "N": N_voxels,
    "incre_list": [0.10]*10,
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
