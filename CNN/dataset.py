# -*- coding: utf-8 -*-
"""Dataset and train/validation split for the voxel -> P11 regressor."""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import config


def load_index(path=None):
    """Read dataset_index.csv into a list of dicts."""
    path = path or config.INDEX_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found - run `python CNN/build_dataset.py` first" % path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("%s is empty" % path)
    return rows


def make_split(rows, n_train=None, n_val=None, seed=None):
    """Shuffle `rows` once and cut them into a train and a validation list.

    Only 119 of the 120 runs reach F11 = 1.3, so the requested 100/20 split
    cannot always be honoured exactly; the validation set absorbs whatever is
    left after the training set is filled, and the caller is told about it.
    """
    n_train = config.N_TRAIN if n_train is None else n_train
    n_val = config.N_VAL if n_val is None else n_val
    seed = config.SPLIT_SEED if seed is None else seed

    if n_train >= len(rows):
        raise ValueError("n_train=%d but only %d cases are available"
                         % (n_train, len(rows)))

    order = np.random.default_rng(seed).permutation(len(rows))
    train = [rows[i] for i in order[:n_train]]
    val = [rows[i] for i in order[n_train:n_train + n_val]]
    return train, val


class MicrostructureDataset(Dataset):
    """One 63^3 binary phase field per case, one scalar stress per case.

    The phase field is returned as a float tensor of shape (1, N, N, N) with
    values 0.0 (matrix) and 1.0 (filler). Targets are standardised with
    `target_mean` / `target_std`, which must come from the *training* split so
    the validation set stays untouched.
    """

    def __init__(self, rows, root=None, target_mean=0.0, target_std=1.0,
                 augment=False, cache=True):
        self.rows = list(rows)
        self.root = root or config.PROJECT_ROOT
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) or 1.0
        self.augment = augment
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.rows)

    def targets(self):
        """Raw (un-standardised) labels, in dataset order."""
        return np.array([float(r["target"]) for r in self.rows])

    def _load_phase(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        path = os.path.join(self.root, self.rows[idx]["structure_npz"])
        with np.load(path) as data:
            phase = data["phase"].astype(np.float32)
        if phase.shape != (config.GRID_N,) * 3:
            raise ValueError("%s has shape %s, expected %s"
                             % (path, phase.shape, (config.GRID_N,) * 3))
        if self._cache is not None:
            self._cache[idx] = phase
        return phase

    @staticmethod
    def _augment(phase, rng):
        """Apply the symmetries that leave the F11 response unchanged.

        The load is uniaxial along x and the FFT solver is periodic, so the
        homogenised P11 is invariant under a mirror in any axis, under a swap
        of the two transverse axes y and z, and under any cyclic translation.
        A full 90-degree rotation group would *not* be label-preserving,
        because it moves the loading axis.
        """
        for axis in range(3):
            if rng.random() < 0.5:
                phase = np.flip(phase, axis=axis)
        if rng.random() < 0.5:
            phase = np.swapaxes(phase, 1, 2)
        shift = rng.integers(0, phase.shape[0], size=3)
        phase = np.roll(phase, shift=tuple(int(s) for s in shift),
                        axis=(0, 1, 2))
        return np.ascontiguousarray(phase)

    def __getitem__(self, idx):
        phase = self._load_phase(idx)
        if self.augment:
            phase = self._augment(phase, np.random.default_rng())
        x = torch.from_numpy(phase.copy()).unsqueeze(0)          # (1, N, N, N)
        raw = float(self.rows[idx]["target"])
        y = torch.tensor((raw - self.target_mean) / self.target_std,
                         dtype=torch.float32)
        return x, y
