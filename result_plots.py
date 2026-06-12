# -*- coding: utf-8 -*-
"""Plot helpers for solver output CSV files."""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_output_csv(csv_path):
    """Load the solver output CSV as a structured NumPy array."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError("No data found in {}".format(csv_path))
    return np.atleast_1d(data)


def plot_stress_strain(data, outfile, dpi=200):
    """Save P11 versus engineering strain F11 - 1."""
    strain = np.insert(data["F11"] - 1.0, 0, 0.0)
    stress = np.insert(data["P11"], 0, 0.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(strain, stress, marker="o", linewidth=1.8)
    ax.set_xlabel("Engineering strain F11 - 1")
    ax.set_ylabel("Stress P11")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_transverse_stretches(data, outfile, dpi=200):
    """Save F22 and F33 versus F11."""
    f11 = np.insert(data["F11"], 0, 1.0)
    f22 = np.insert(data["F22"], 0, 1.0)
    f33 = np.insert(data["F33"], 0, 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(f11, f22, marker="o", linewidth=1.8, label="F22")
    ax.plot(f11, f33, marker="s", linewidth=1.8, label="F33")
    ax.set_xlabel("F11")
    ax.set_ylabel("Transverse stretch")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def save_result_plots(path, csv_name="output.csv", dpi=200):
    """Create the standard result plots next to the solver output CSV."""
    csv_path = os.path.join(path, csv_name)
    data = load_output_csv(csv_path)

    stress_strain_path = os.path.join(path, "stress_strain.png")
    transverse_path = os.path.join(path, "F11_vs_F22_F33.png")

    plot_stress_strain(data, stress_strain_path, dpi=dpi)
    plot_transverse_stretches(data, transverse_path, dpi=dpi)

    return stress_strain_path, transverse_path
