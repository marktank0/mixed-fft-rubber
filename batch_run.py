# -*- coding: utf-8 -*-
"""Parallel batch runner for many independent FFT simulations."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

from run_case import run_case


MAX_WORKERS = 3

STRUCTURE_DIR = "3D_samples/voxels"
CHARGE_PATH = "3D_samples/Charges/Strain_1.0_E10x.txt"
OUTPUT_PATH = "Results/Benchmark_v2"

CASE_SETTINGS = {
    "charge_path": CHARGE_PATH,
    "output_path": OUTPUT_PATH,
    "N": 31,
    "incre_list": [0.1]*10,
    "preconditioner": "reference",
    "diagnostics": False,
    "save_plots": True,
    "plot_dpi": 200,
    "matrix_phase": 0,
    "filler_phase": 1,
    "log_to_file": True,
}


def build_cases():
    return [
        dict(CASE_SETTINGS, structure_path=os.path.join(STRUCTURE_DIR, filename))
        for filename in sorted(os.listdir(STRUCTURE_DIR))
        if filename.lower().endswith(".npz")
    ]


def run_batch(cases, max_workers):
    if not cases:
        print("No cases to run.")
        return []

    actual_workers = min(max_workers, len(cases))
    print("Running {} cases with {} workers.".format(len(cases), actual_workers))

    results = []
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = [executor.submit(run_case, case) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("finished {} -> {}".format(result["structure_path"], result["output_path"]))
    return results


if __name__ == "__main__":
    run_batch(build_cases(), MAX_WORKERS)
