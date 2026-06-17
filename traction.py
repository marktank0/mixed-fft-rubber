# -*- coding: utf-8 -*-
"""Single-case traction runner."""

from run_case import run_case


CASE = {
    "structure_path": r"C:/Coding/mixed-fft-master/3D_samples/voxels/1_voxel.npz",
    "charge_path": r"3D_samples\Charges\Neo_1.0_E10x.txt",
    "output_path": r"Results",
    "N": 31,
    "incre_list": [0.05]*20,
    "preconditioner": "reference",
    "diagnostics": False,
    "save_plots": True,
    "plot_dpi": 200,
    "matrix_phase": 0,
    "filler_phase": 1,
}


if __name__ == "__main__":
    run_case(CASE)
