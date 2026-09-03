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
| `train.py` | Training loop, checkpoints and history into `runs/<name>/`; single-GPU or DDP |
| `dist_utils.py` | Rank bookkeeping, device selection and cross-rank reductions for DDP |
| `sweep.py` | Runs one independent training per GPU, in parallel |
| `evaluate.py` | Scores a checkpoint on its held-out split, writes a parity plot |
| `run_train_gpu.sbatch` | SLURM job for a multi-GPU node (`MODE=sweep` or `MODE=ddp`) |

## Usage

Build the index once (CPU only, no torch needed):

```bash
python CNN/build_dataset.py
```

Train on one GPU:

```bash
python CNN/train.py --epochs 200 --batch-size 4
```

Evaluate a finished run:

```bash
python CNN/evaluate.py --run-dir CNN/runs/20260903_120000
```

## Using more than one GPU

There are two modes, and at this dataset size they are not equally good.

**`sweep.py` — one whole model per GPU (recommended).** Four GPUs run four
independent trainings on four different split seeds, in the same wall time one
run would take. With only 19 validation cases a single score is very noisy, so
the spread across seeds is the number you should actually trust.

```bash
python CNN/sweep.py --gpus 0,1,2,3 --seeds 0,1,2,3 -- --epochs 200
```

Each run writes `CNN/runs/<tag>_seed<N>/` plus a matching `.log`. Fewer GPUs
than seeds is fine — runs queue onto GPUs as they free up.

**`torchrun` — one model split across GPUs (DDP).** Useful if you grow the
dataset or the model; not much help on 100 training cases.

```bash
torchrun --standalone --nproc_per_node=4 CNN/train.py --epochs 200 --batch-size 4
```

Under DDP:

- `--batch-size` is **per GPU**. Four GPUs at `--batch-size 4` is an effective
  batch of 16, so consider `--scale-lr` (multiplies the LR by the world size).
- BatchNorm is converted to `SyncBatchNorm`, because a per-GPU batch of 4 gives
  useless per-device statistics. `--no-sync-bn` opts out.
- Validation runs on rank 0 over the *whole* validation set. Sharding it would
  make `DistributedSampler` pad with duplicate cases and quietly bias the
  metrics — with 19 cases over 4 ranks that is a real distortion, not a
  rounding error.
- Only rank 0 prints, checkpoints and writes `history.csv`.
- `train.py` warns if the training set is too small for the number of ranks.

On SLURM, submit the provided job script:

```bash
sbatch --export=MODE=sweep CNN/run_train_gpu.sbatch
```

Check the partition name and `--gres` line in that file against your cluster
before the first submit — they are guesses based on your existing
`run_fix_c5_c6.sbatch`, which is CPU-only.

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

- **PyTorch is not installed on this machine**, so everything torch-related is
  syntax-checked but not executed. The data pipeline (`build_dataset.py`) *has*
  been run against the real files. Install a CUDA build of torch on the server
  before the first training run, and treat the first DDP launch as a smoke test
  (`--epochs 2`) rather than a full run.
- 119 samples is a very small dataset for a 3D CNN. Expect the augmentation and
  the weight decay to matter more than the architecture. If validation R^2 stays
  poor, compare against the trivial baseline of predicting from filler volume
  fraction alone — the index carries `phr` and `filler_fraction` columns for
  exactly that check.
- All 119 structures cached as float32 is ~119 MB in RAM, which is why
  `MicrostructureDataset` caches by default. Set `cache=False` if that is
  tight; storing them as `uint8` and converting per batch would cut it by 4x.
