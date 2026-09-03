# -*- coding: utf-8 -*-
"""Score a trained checkpoint on its validation split and plot the parity.

    python CNN/evaluate.py --run-dir CNN/runs/<name>

Reads split.csv from the run directory so the evaluation sees exactly the
cases the model was never trained on.
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import MicrostructureDataset, load_index
from model import StressCNN
from train import physical_metrics, pick_device


def rows_from_split(split_csv, index_rows, split="val"):
    """Pick the index rows whose names appear under `split` in split.csv."""
    by_name = {row["name"]: row for row in index_rows}
    with open(split_csv, "r", encoding="utf-8", newline="") as handle:
        names = [r["name"] for r in csv.DictReader(handle)
                 if r["split"] == split]
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError("split lists cases absent from the index: %s"
                       % ", ".join(missing))
    return [by_name[n] for n in names]


def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def parity_plot(names, pred, true, path):
    """Predicted vs. true stress; saved next to the checkpoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = float(min(pred.min(), true.min()))
    hi = float(max(pred.max(), true.max()))
    pad = 0.05 * (hi - lo or 1.0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1,
            label="ideal")
    ax.scatter(true, pred, s=28, alpha=0.8)
    ax.set_xlabel("FFT %s at F11 = %.2f" % (config.TARGET_COLUMN,
                                           config.TARGET_F11))
    ax.set_ylabel("CNN prediction")
    ax.set_title("Validation parity (%d cases)" % len(names))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--index", default=config.INDEX_CSV)
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mean, std = ckpt["target_mean"], ckpt["target_std"]

    rows = rows_from_split(os.path.join(args.run_dir, "split.csv"),
                           load_index(args.index), args.split)
    dataset = MicrostructureDataset(rows, target_mean=mean, target_std=std,
                                    augment=False)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)

    device = pick_device(args.device)
    model = StressCNN().to(device)
    model.load_state_dict(ckpt["model_state"])

    pred_std, true_std = predict(model, loader, device)
    metrics = physical_metrics(pred_std, true_std, mean, std)
    pred = pred_std * std + mean
    true = true_std * std + mean

    print("%s split, %d cases (checkpoint from epoch %s)"
          % (args.split, len(rows), ckpt.get("epoch", "?")))
    print("MAE %.4f   RMSE %.4f   R2 %.4f"
          % (metrics["mae"], metrics["rmse"], metrics["r2"]))
    print()
    print("%-28s %10s %10s %10s" % ("case", "true", "pred", "error"))
    for row, t, p in zip(rows, true, pred):
        print("%-28s %10.4f %10.4f %10.4f" % (row["name"], t, p, p - t))

    out_csv = os.path.join(args.run_dir, "predictions_%s.csv" % args.split)
    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "true", "pred", "error"])
        for row, t, p in zip(rows, true, pred):
            writer.writerow([row["name"], "%.6e" % t, "%.6e" % p,
                             "%.6e" % (p - t)])

    plot_path = os.path.join(args.run_dir, "parity_%s.png" % args.split)
    parity_plot([r["name"] for r in rows], pred, true, plot_path)
    print("\nwrote %s and %s" % (out_csv, plot_path))


if __name__ == "__main__":
    main()
