# -*- coding: utf-8 -*-
"""
Created on Fri May  7 03:40:50 2021

@author: WANG Mingchuan
"""

import numpy as np
from fg.mxfft import *
from result_plots import save_result_plots
from run_metadata import write_run_metadata
import time
import os
#
SAVE_PLOTS = True
PLOT_DPI = 200
N = 31
PRECONDITIONER = "reference"
DIAGNOSTICS = False
MATRIX_PHASE = 0
FILLER_PHASE = 1


# Create cases -------------------------------------------------------------
STRUCTURE_DIR = "3D_samples/voxels"
CHARGE_PATH = "3D_samples/Charges/Strain_1.0_E10x.txt"
OUTPUT_PATH = "Results/Benchmark_v1"

CASES = [
    {
        "structure_path": os.path.join(STRUCTURE_DIR, filename),
        "charge_path": CHARGE_PATH,
        "output_path": OUTPUT_PATH,
    }
    for filename in sorted(os.listdir(STRUCTURE_DIR))
    if filename.lower().endswith(".npz")
]
# ---------------------------------------------------------------------------
for case in CASES:
    structure_path = case["structure_path"]
    charge_path = case["charge_path"]
    output_path = case.get("output_path")

    print(structure_path)
    t1 = time.time()
    prob = FFTSolver(
        structure_path,
        charge_path=charge_path,
        output_path=output_path,
        N=N,
    )
    #
    incre_list=[0.1]*10
    prob.calculate(incre_list=incre_list,savemodel="normal", preconditioner=PRECONDITIONER, diagnostics=DIAGNOSTICS)
    solve_time = time.time() - t1
    plot_files = []
    if SAVE_PLOTS:
        plot_files = save_result_plots(prob.output_path, dpi=PLOT_DPI)
        print("plots are saved...")
        for plot_file in plot_files:
            print(plot_file)
    t2 = time.time()
    metadata_file = write_run_metadata(
        prob.output_path,
        structure_path=structure_path,
        charge_path=charge_path,
        run_time_seconds=solve_time,
        N=N,
        incre_list=incre_list,
        preconditioner=PRECONDITIONER,
        diagnostics=DIAGNOSTICS,
        matrix_phase=MATRIX_PHASE,
        filler_phase=FILLER_PHASE,
        plot_files=plot_files,
    )
    print("run metadata is saved...")
    print(metadata_file)
    print("finish!")
    print(t2-t1)