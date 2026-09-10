# -*- coding: utf-8 -*-
"""Small path helpers for solver inputs and outputs."""

import os


def default_charge_path(structure_path):
    base_dir = structure_path if os.path.isdir(structure_path) else os.path.dirname(structure_path)
    return os.path.join(base_dir, "charge.txt")


def phase_source(structure_path, phase_path=None):
    if phase_path is not None:
        base_dir = structure_path if os.path.isdir(structure_path) else os.path.dirname(structure_path)
        return base_dir, phase_path

    if os.path.isfile(structure_path):
        return os.path.dirname(structure_path), structure_path

    return structure_path, None


def output_run_path(structure_path, output_path=None, output_name=None):
    normalized = os.path.normpath(structure_path)
    parent = output_path if output_path is not None else os.path.dirname(normalized)
    if output_name is None:
        name = os.path.basename(normalized)
        if os.path.isfile(normalized):
            name = os.path.splitext(name)[0]
        output_name = "{}_output".format(name)
    return os.path.join(parent, output_name)


def ensure_output_path(output_path):
    os.makedirs(output_path, exist_ok=True)
    return output_path
