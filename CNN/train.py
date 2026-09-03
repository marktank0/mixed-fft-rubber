# -*- coding: utf-8 -*-
"""Train the 3D CNN that predicts P11 at F11 = 1.3 from a microstructure.

    python CNN/build_dataset.py      # once, builds dataset_index.csv
    python CNN/train.py              # then train

Checkpoints, the per-epoch history and the exact split land in CNN/runs/<name>.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import MicrostructureDataset, load_index, make_split
from model import StressCNN, count_parameters


def pick_device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over `loader`; training when an optimizer is given.

    Returns the mean loss and the predictions/targets in standardised units,
    so the caller can de-standardise them for reporting.
    """
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    preds, trues = [], []

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    return total / max(n, 1), np.concatenate(preds), np.concatenate(trues)


def physical_metrics(pred_std, true_std, mean, std):
    """MAE, RMSE and R^2 in the physical units of the target."""
    pred = pred_std * std + mean
    true = true_std * std + mean
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def save_split(path, train_rows, val_rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "name", "target"])
        for tag, rows in (("train", train_rows), ("val", val_rows)):
            for row in rows:
                writer.writerow([tag, row["name"], row["target"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=config.INDEX_CSV)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float,
                        default=config.WEIGHT_DECAY)
    parser.add_argument("--n-train", type=int, default=config.N_TRAIN)
    parser.add_argument("--n-val", type=int, default=config.N_VAL)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = load_index(args.index)
    train_rows, val_rows = make_split(rows, args.n_train, args.n_val, args.seed)
    if len(val_rows) < args.n_val:
        print("note: only %d cases passed the F11 >= %.2f filter, so the "
              "split is %d/%d instead of %d/%d"
              % (len(rows), config.MIN_F11, len(train_rows), len(val_rows),
                 args.n_train, args.n_val))
    print("train %d cases, validate on %d" % (len(train_rows), len(val_rows)))

    # Standardise the target with training statistics only.
    train_targets = np.array([float(r["target"]) for r in train_rows])
    mean, std = float(train_targets.mean()), float(train_targets.std())
    std = std or 1.0
    print("target %s: train mean %.4f, std %.4f"
          % (config.TARGET_COLUMN, mean, std))

    augment = config.AUGMENT and not args.no_augment
    train_set = MicrostructureDataset(train_rows, target_mean=mean,
                                      target_std=std, augment=augment)
    val_set = MicrostructureDataset(val_rows, target_mean=mean,
                                    target_std=std, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS)

    device = pick_device(args.device)
    model = StressCNN().to(device)
    print("model on %s, %d trainable parameters"
          % (device, count_parameters(model)))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.RUN_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_split(os.path.join(run_dir, "split.csv"), train_rows, val_rows)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as fh:
        json.dump({**vars(args), "augment": augment, "target_mean": mean,
                   "target_std": std}, fh, indent=2)

    history, best_val = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_pred, tr_true = run_epoch(model, train_loader, device,
                                              criterion, optimizer)
        va_loss, va_pred, va_true = run_epoch(model, val_loader, device,
                                              criterion)
        scheduler.step()

        tr_metrics = physical_metrics(tr_pred, tr_true, mean, std)
        va_metrics = physical_metrics(va_pred, va_true, mean, std)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": va_loss,
                        **{"train_" + k: v for k, v in tr_metrics.items()},
                        **{"val_" + k: v for k, v in va_metrics.items()}})

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "target_mean": mean, "target_std": std,
                        "val_loss": va_loss, "val_metrics": va_metrics},
                       os.path.join(run_dir, "best.pt"))
            marker = "  *"
        else:
            marker = ""

        print("epoch %3d/%d  train %.4f (MAE %.4f)  val %.4f "
              "(MAE %.4f, R2 %.3f)%s"
              % (epoch, args.epochs, tr_loss, tr_metrics["mae"], va_loss,
                 va_metrics["mae"], va_metrics["r2"], marker))

    torch.save({"epoch": args.epochs, "model_state": model.state_dict(),
                "target_mean": mean, "target_std": std},
               os.path.join(run_dir, "last.pt"))
    with open(os.path.join(run_dir, "history.csv"), "w", encoding="utf-8",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print("best validation loss %.4f; run written to %s" % (best_val, run_dir))


if __name__ == "__main__":
    main()
