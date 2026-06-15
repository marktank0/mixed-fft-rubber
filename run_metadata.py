# -*- coding: utf-8 -*-
"""Run metadata helpers for archiving solver runs."""

import os
from datetime import datetime

import numpy as np


COMPONENTS = ("11", "12", "13", "21", "22", "23", "31", "32", "33")


def _format_list(values):
    return ", ".join(str(value) for value in values)


def _phase_file_from_structure(structure_path):
    if os.path.isfile(structure_path):
        return structure_path

    npz_files = [
        os.path.join(structure_path, name)
        for name in os.listdir(structure_path)
        if name.lower().endswith(".npz")
    ]
    if len(npz_files) == 1:
        return npz_files[0]

    phase_txt = os.path.join(structure_path, "phase.txt")
    if os.path.exists(phase_txt):
        return phase_txt

    if len(npz_files) > 1:
        raise ValueError("Multiple .npz files found in {}; pass a specific structure file".format(structure_path))
    raise FileNotFoundError("No phase .npz file or phase.txt found in {}".format(structure_path))


def load_phase_array(structure_path, phase_key="phase"):
    phase_file = _phase_file_from_structure(structure_path)
    if phase_file.lower().endswith(".npz"):
        with np.load(phase_file, allow_pickle=False) as data:
            if phase_key not in data.files:
                raise KeyError("Missing phase key '{}' in {}".format(phase_key, phase_file))
            phase = np.array(data[phase_key], copy=False)
    else:
        phase = np.loadtxt(phase_file)
    return phase, phase_file


def calculate_volume_fraction(structure_path, filler_phase=1, phase_key="phase"):
    phase, phase_file = load_phase_array(structure_path, phase_key=phase_key)
    filler_count = int(np.count_nonzero(phase == filler_phase))
    total_count = int(phase.size)
    return {
        "phase_file": phase_file,
        "filler_phase": filler_phase,
        "filler_voxels": filler_count,
        "total_voxels": total_count,
        "volume_fraction": filler_count/float(total_count),
    }


def load_charge_metadata(charge_path):
    data = np.loadtxt(charge_path)
    phase0 = data[0, :]
    phase1 = data[1, :]
    dF = data[2, :]
    charge_type = data[3, :]
    return {
        "phase_rows": {
            0: {
                "model": int(phase0[0]),
                "parameters": phase0[1:].tolist(),
                "E": float(phase0[1]),
                "poisson": float(phase0[2]),
            },
            1: {
                "model": int(phase1[0]),
                "parameters": phase1[1:].tolist(),
                "E": float(phase1[1]),
                "poisson": float(phase1[2]),
            },
        },
        "dF": dF.tolist(),
        "charge_type": charge_type.tolist(),
    }


def write_run_metadata(
    output_path,
    structure_path,
    charge_path,
    run_time_seconds,
    N,
    incre_list,
    preconditioner,
    diagnostics=False,
    phase_key="phase",
    matrix_phase=0,
    filler_phase=1,
    plot_files=None,
    filename="run_metadata.txt",
):
    os.makedirs(output_path, exist_ok=True)

    charge = load_charge_metadata(charge_path)
    volume = calculate_volume_fraction(structure_path, filler_phase=filler_phase, phase_key=phase_key)
    matrix = charge["phase_rows"][matrix_phase]
    filler = charge["phase_rows"][filler_phase]
    charge_modes = [
        "P{} controlled".format(component) if value > 0.5 else "F{} controlled".format(component)
        for component, value in zip(COMPONENTS, charge["charge_type"])
    ]

    outfile = os.path.join(output_path, filename)
    with open(outfile, "w") as file:
        file.write("Created: {}\n".format(datetime.now().isoformat(timespec="seconds")))
        file.write("Run time seconds: {:.3f}\n\n".format(run_time_seconds))

        file.write("Input files\n")
        file.write("-----------\n")
        file.write("Structure path: {}\n".format(structure_path))
        file.write("Charge path: {}\n\n".format(charge_path))

        file.write("Material phases\n")
        file.write("---------------\n")
        for phase_id in sorted(charge["phase_rows"]):
            row = charge["phase_rows"][phase_id]
            role = []
            if phase_id == matrix_phase:
                role.append("matrix")
            if phase_id == filler_phase:
                role.append("filler")
            role_text = " ({})".format(", ".join(role)) if role else ""
            file.write("Phase {}{}: model {}, E {}, poisson {}\n".format(
                phase_id,
                role_text,
                row["model"],
                row["E"],
                row["poisson"],
            ))
        file.write("\n")

        file.write("Structure statistics\n")
        file.write("--------------------\n")
        file.write("Filler phase: {}\n".format(volume["filler_phase"]))
        file.write("Filler voxels: {}\n".format(volume["filler_voxels"]))
        file.write("Total voxels: {}\n".format(volume["total_voxels"]))
        file.write("Filler volume fraction: {:.6f}\n\n".format(volume["volume_fraction"]))

        file.write("Solver settings\n")
        file.write("---------------\n")
        file.write("Grid N: {}\n".format(N))
        file.write("Increments: [{}]\n".format(_format_list(incre_list)))
        file.write("Preconditioner: {}\n".format(preconditioner))
        file.write("Diagnostics: {}\n\n".format(diagnostics))

        file.write("Charge\n")
        file.write("------\n")
        file.write("dF: [{}]\n".format(_format_list(charge["dF"])))
        file.write("Control type: [{}]\n\n".format(_format_list(charge_modes)))

    return outfile
