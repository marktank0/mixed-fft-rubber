# 3D CNN: microstructure -> stress at 30% strain

Predicts the first Piola-Kirchhoff stress `P11` that the FFT solver reached at
a uniaxial stretch of `F11 = 1.3` (30% engineering strain), directly from the
63x63x63 binary voxel structure.

## Data

| | |
|---|---|
| Structures | `microstructure_generation/3D_samples/improved_struct_v4/voxel_structures/<case>_voxel.npz`, key `phase`, `uint8` (0 = matrix, 1 = filler), shape 63^3 |
| Simulations | `Results/HPC/Results/cases120_63voxel_10contrast/<case>_voxel_output/output.csv`, columns `F11..F33, P11..P33` |
| Contrast | E_matrix 10, E_filler 100 (10x), Neo-Hookean |

The label is `np.interp` of `P11` onto `F11 = 1.3`. The solver steps `F11`
monotonically in increments of 0.025, so 1.3 is an increment boundary and the
interpolation is essentially exact.

**Filter:** only cases whose curve actually reached `F11 >= 1.3` are used —
119 of the 120 runs qualify (one stopped at `F11 = 1.29`).

**Split:** 100 train / 19 validation. The 100/20 you asked for needs 120 usable
cases; after the strain filter there are 119, so the validation set gets the 19
that remain. `train.py` prints this when it happens. Set `--n-train 99` if you
would rather keep a round 20 for validation.

Roughly half the runs have no `run_metadata.txt` (walltime kills), and the ones
that do all report `Solver status: failed` — that flag is recorded in the index
as a column but is *not* used for filtering, since those runs still produced
valid stress-strain data well past 30% strain. Add a filter on it in
`build_dataset.py` if you decide otherwise.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, target definition, split sizes, hyperparameters |
| `build_dataset.py` | Scans the solver outputs, writes `dataset_index.csv` (one row per usable case, with the interpolated label) |
| `dataset.py` | `MicrostructureDataset` + the train/val split |
| `model.py` | `StressCNN` — four conv blocks, global average pool, scalar head |
| `train.py` | Training loop, checkpoints and history into `runs/<name>/` |
| `evaluate.py` | Scores a checkpoint on its held-out split, writes a parity plot |

## Usage

Build the index once:

```bash
python CNN/build_dataset.py
```

Train:

```bash
python CNN/train.py --epochs 200 --batch-size 4
```

Evaluate a finished run:

```bash
python CNN/evaluate.py --run-dir CNN/runs/20260903_120000
```

## Model

Input `(B, 1, 63, 63, 63)`. Four blocks of `conv3x3x3 -> BN -> ReLU` twice then
`maxpool2`, widths 16/32/64/128, taking 63 -> 31 -> 15 -> 7 -> 3. A global
average pool then a 2-layer head gives one scalar. ~1.1M parameters.

Targets are standardised with **training-set** mean and std; the constants are
stored in the checkpoint so `evaluate.py` reports errors in physical units.

## Augmentation

With uniaxial loading along x and a periodic FFT solver, the homogenised `P11`
is invariant under: a mirror in any axis, a swap of the two transverse axes
y and z, and any cyclic translation. `dataset.py` samples from exactly that
group. A full 90-degree rotation group would **not** be valid here, because it
moves the loading axis onto a different structure direction.

## Notes before you run this

- **PyTorch is not installed in this environment.** `pip install torch` (a CUDA
  build if you want the GPU) before running anything here.
- 119 samples is a very small dataset for a 3D CNN. Expect the augmentation and
  the weight decay to matter more than the architecture. If validation R^2 stays
  poor, compare against the trivial baseline of predicting from filler volume
  fraction alone — the index carries `phr` and `filler_fraction` columns for
  exactly that check.
- All 119 structures cached as float32 is ~119 MB in RAM, which is why
  `MicrostructureDataset` caches by default. Set `cache=False` if that is
  tight; storing them as `uint8` and converting per batch would cut it by 4x.
