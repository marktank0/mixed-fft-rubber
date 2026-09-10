# -*- coding: utf-8 -*-
"""Plot CNN training and validation loss from training_history.csv."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot train_loss and val_loss from a training history CSV."
    )
    parser.add_argument("csv", nargs="?", default="training_history.csv")
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_name(
        "{}_loss.png".format(csv_path.stem)
    )

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    data = np.atleast_1d(data)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(data["epoch"], data["train_loss"], label="Training loss", linewidth=1.8)
    ax.plot(data["epoch"], data["val_loss"], label="Validation loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CNN Training History")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("saved {}".format(out_path))


if __name__ == "__main__":
    main()
