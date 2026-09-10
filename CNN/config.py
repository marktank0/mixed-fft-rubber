# -*- coding: utf-8 -*-
"""Paths and hyperparameters for the 3D-CNN stress regressor.

The network maps a 63x63x63 binary microstructure (0 = matrix, 1 = filler) to
the first Piola-Kirchhoff stress P11 that the FFT solver reached at a uniaxial
stretch of F11 = 1.3 (30% engineering strain).

Every path is derived from this file's own location, so the scripts run the
same from the repository root, from CNN/ or from a batch worker.
"""

import os

CNN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CNN_DIR)

# --- data ------------------------------------------------------------------
# Voxelised microstructures: <STRUCTURE_DIR>/<case>_voxel.npz, key "phase".
STRUCTURE_DIR = os.path.join(
    PROJECT_ROOT, "microstructure_generation", "3D_samples",
    "improved_struct_v4", "voxel_structures",
)
STRUCTURE_MANIFEST = os.path.join(
    PROJECT_ROOT, "microstructure_generation", "3D_samples",
    "improved_struct_v4", "manifest.csv",
)
# Solver output: <RESULTS_DIR>/<case>_voxel_output/output.csv
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "Results", "HPC", "Results", "cases120_63voxel_10contrast",
)

# Case index written by build_dataset.py and read by train.py.
INDEX_CSV = os.path.join(CNN_DIR, "dataset_index.csv")
RUN_DIR = os.path.join(CNN_DIR, "runs")

# --- label -----------------------------------------------------------------
GRID_N = 63                # voxels per side
TARGET_F11 = 1.3           # stretch at which the label is read off
TARGET_COLUMN = "P11"      # column of output.csv used as the label
# A case is kept only if its stress-strain curve actually reached TARGET_F11;
# anything short of it would need extrapolation.
MIN_F11 = TARGET_F11

# --- split -----------------------------------------------------------------
# 120 cases exist; 119 reach F11 = 1.3, so the requested 100/20 split becomes
# 100/19. build_dataset.py reports the exact numbers, and make_split() below
# clamps the validation set to whatever is left over.
N_TRAIN = 100
N_VAL = 20
SPLIT_SEED = 0

# --- training --------------------------------------------------------------
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16         # channel width of the first conv block
DROPOUT = 0.1
NUM_WORKERS = 0            # >0 needs the spawn guard on Windows; 0 is safe
AUGMENT = True             # flips + y/z swap, see dataset.py
DEVICE = "cuda"            # falls back to CPU automatically when unavailable
