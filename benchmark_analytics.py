# -*- coding: utf-8 -*-
"""Analytics plots for benchmark result folders."""

import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


VOLUME_FRACTION_PATTERN = re.compile(r"Filler volume fraction:\s*([0-9.eE+-]+)")


def read_volume_fraction(metadata_path):
    with open(metadata_path) as file:
        text = file.read()
    match = VOLUME_FRACTION_PATTERN.search(text)
    if not match:
        raise ValueError("No filler volume fraction found in {}".format(metadata_path))
    return float(match.group(1))


def stress_at_f11(output_csv, target_f11, endpoint_tolerance=0.02):
    data = np.genfromtxt(output_csv, delimiter=",", names=True)
    data = np.atleast_1d(data)
    order = np.argsort(data["F11"])
    f11 = data["F11"][order]
    p11 = data["P11"][order]

    if target_f11 < f11[0]:
        if f11[0] - target_f11 <= endpoint_tolerance:
            return float(p11[0]), float(f11[0])
        raise ValueError("{} is below F11 range in {}".format(target_f11, output_csv))

    if target_f11 > f11[-1]:
        if target_f11 - f11[-1] <= endpoint_tolerance:
            return float(p11[-1]), float(f11[-1])
        raise ValueError("{} is above F11 range in {}".format(target_f11, output_csv))

    return float(np.interp(target_f11, f11, p11)), float(target_f11)


def collect_stress_volume_data(results_dir, target_f11_values=(1.20, 1.50, 2.00)):
    rows = []
    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        if not os.path.isdir(run_dir):
            continue

        metadata_path = os.path.join(run_dir, "run_metadata.txt")
        output_csv = os.path.join(run_dir, "output.csv")
        if not os.path.exists(metadata_path) or not os.path.exists(output_csv):
            continue

        volume_fraction = read_volume_fraction(metadata_path)
        row = {
            "run": name,
            "volume_fraction": volume_fraction,
        }
        for target in target_f11_values:
            stress, used_f11 = stress_at_f11(output_csv, target)
            key = "{:.2f}".format(target)
            row["P11_at_F11_{}".format(key)] = stress
            row["used_F11_for_{}".format(key)] = used_f11
        rows.append(row)

    return sorted(rows, key=lambda item: item["volume_fraction"])


def save_summary_csv(rows, outfile, target_f11_values):
    header = ["run", "volume_fraction"]
    for target in target_f11_values:
        key = "{:.2f}".format(target)
        header.extend(["P11_at_F11_{}".format(key), "used_F11_for_{}".format(key)])

    with open(outfile, "w") as file:
        file.write(",".join(header) + "\n")
        for row in rows:
            values = []
            for column in header:
                value = row[column]
                if isinstance(value, float):
                    values.append("{:.8g}".format(value))
                else:
                    values.append(str(value))
            file.write(",".join(values) + "\n")


def plot_stress_vs_volume_fraction(rows, outfile, target_f11_values):
    volume_fraction = np.array([row["volume_fraction"] for row in rows])

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    for target in target_f11_values:
        key = "{:.2f}".format(target)
        stress = np.array([row["P11_at_F11_{}".format(key)] for row in rows])
        ax.plot(volume_fraction, stress, marker="o", linewidth=1.8, label="F11 = {:.2f}".format(target))

    ax.set_xlabel("Filler volume fraction")
    ax.set_ylabel("Stress P11")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(title="Sample point")
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def make_benchmark_stress_volume_plot(
    results_dir="Results/Benchmark_v1",
    target_f11_values=(1.20, 1.50, 2.00),
):
    rows = collect_stress_volume_data(results_dir, target_f11_values)
    if not rows:
        raise ValueError("No complete benchmark runs found in {}".format(results_dir))

    summary_csv = os.path.join(results_dir, "stress_vs_volume_fraction.csv")
    plot_path = os.path.join(results_dir, "stress_vs_volume_fraction.png")
    save_summary_csv(rows, summary_csv, target_f11_values)
    plot_stress_vs_volume_fraction(rows, plot_path, target_f11_values)
    return plot_path, summary_csv, rows


if __name__ == "__main__":
    plot_path, summary_csv, rows = make_benchmark_stress_volume_plot()
    print(plot_path)
    print(summary_csv)
    print("runs:", len(rows))
