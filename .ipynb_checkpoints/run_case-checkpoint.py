# -*- coding: utf-8 -*-
"""Reusable single-case runner for FFT simulations."""

import contextlib
import os
import time

from fg.io_paths import ensure_output_path, output_run_path
from fg.mxfft import FFTSolver
from result_plots import save_result_plots
from run_metadata import write_run_metadata


def _case_value(case, key, default):
    return case[key] if key in case else default


def _run_case_impl(case):
    structure_path = case["structure_path"]
    charge_path = case.get("charge_path")
    output_path = case.get("output_path")
    N = _case_value(case, "N", 31)
    incre_list = _case_value(case, "incre_list", [0.1]*10)
    preconditioner = _case_value(case, "preconditioner", "reference")
    diagnostics = _case_value(case, "diagnostics", False)
    savemodel = _case_value(case, "savemodel", "normal")
    save_plots = _case_value(case, "save_plots", True)
    plot_dpi = _case_value(case, "plot_dpi", 200)
    matrix_phase = _case_value(case, "matrix_phase", 0)
    filler_phase = _case_value(case, "filler_phase", 1)
    save_fields = _case_value(case, "save_fields", False)
    field_filename = _case_value(case, "field_filename", "fields.vti")

    print(structure_path)
    start_time = time.time()
    prob = FFTSolver(
        structure_path,
        charge_path=charge_path,
        output_path=output_path,
        N=N,
    )

    prob.calculate(
        incre_list=incre_list,
        savemodel=savemodel,
        preconditioner=preconditioner,
        diagnostics=diagnostics,
        save_fields=save_fields,
        field_filename=field_filename,
    )
    solve_time = time.time() - start_time

    plot_files = []
    if save_plots:
        plot_files = save_result_plots(prob.output_path, dpi=plot_dpi)
        print("plots are saved...")
        for plot_file in plot_files:
            print(plot_file)

    metadata_file = write_run_metadata(
        prob.output_path,
        structure_path=structure_path,
        charge_path=prob.charge_path,
        run_time_seconds=solve_time,
        N=N,
        incre_list=incre_list,
        preconditioner=preconditioner,
        diagnostics=diagnostics,
        matrix_phase=matrix_phase,
        filler_phase=filler_phase,
        plot_files=plot_files,
    )
    print("run metadata is saved...")
    print(metadata_file)
    print("finish!")
    print(time.time() - start_time)

    return {
        "structure_path": structure_path,
        "charge_path": prob.charge_path,
        "output_path": prob.output_path,
        "metadata_file": metadata_file,
        "plot_files": plot_files,
        "run_time_seconds": solve_time,
    }


def run_case(case):
    """Run one simulation case and return a small result summary."""
    if not case.get("log_to_file", False):
        return _run_case_impl(case)

    output_path = ensure_output_path(output_run_path(case["structure_path"], case.get("output_path")))
    log_path = os.path.join(output_path, "run.log")
    with open(log_path, "w") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            result = _run_case_impl(case)
    result["log_file"] = log_path
    return result
