# 3D Microstructure Generation

Generates 3D rubber filler microstructures for the FFT solver. A structure is
built from three stochastic models, saved as a voxel grid (and/or sphere
parameters), and its PHR is recorded.

Two ways to run:

- **Batch (recommended):** describe everything in a YAML file and generate many
  structures at once, in parallel. See [Batch generation](#batch-generation).
- **Single run:** the `combined_particle_models.py` CLI generates one structure
  and can open a viewer. See [Single structure (CLI)](#single-structure-cli).

## What it builds

The final filler structure is assembled from three Boolean models:

1. **Model 1** — small candidate filler spheres (the actual filler particles).
2. **Model 2** — randomly oriented spheroids defining the region where filler is
   *allowed*.
3. **Model 3** — larger spheres that *remove* filler again inside Model 2.

The result is: `final filler = Model 1 spheres inside Model 2 but outside Model 3`.

### PHR is an output, not an input

PHR (parts filler per hundred rubber) cannot be set directly. It is computed
from the filler volume fraction *after* voxelization, using the densities you
provide (`calculate_phr.py`). You steer PHR indirectly — mainly through
`model1.intensity` (more candidate spheres → more filler → higher PHR). The
batch runner records the resulting PHR of every structure so you can select by
it afterward.

## Batch generation

Run a config with the runner (from the repository root, using the project venv):

```powershell
.venv\Scripts\python microstructure_generation\generate_structures.py microstructure_generation\Run_configs\generation_phr_sweep.yaml
```

Useful flags:

- `--dry-run` — list the planned structures without generating anything.
- `--max-workers N` — override `execution.max_workers`.
- `--mode sweep|cases` — override `run.mode`.

### The config file

`Run_configs/generation_phr_sweep.yaml` is the annotated template. Key sections:

- **`output_dir`** — where structures are written (relative to the repo root, or
  absolute).
- **`defaults`** — the baseline parameters for *every* structure (box size, the
  three models, voxel grid, which outputs to save). A sweep axis or a case only
  overrides the fields it names; everything else is inherited.
- **`phr`** — `filler_density` and `rubber_density` used to compute PHR.
- **`execution`** — `max_workers` (parallel structures), `on_existing`
  (`skip` | `overwrite` | `error`), and `save_manifest`.
- **`run.mode`** — `sweep` or `cases`.

**Sweep mode** takes the cartesian product of the axes and generates
`replicates` structures per point. An axis value is either a
`{start, stop, step}` range (inclusive) or a list of explicit values:

```yaml
sweep:
  axes:
    model1.intensity: { start: 180, stop: 220, step: 20 }   # 3 values
  replicates: 10                                             # 3 x 10 = 30 structures
  seed:
    strategy: sequential   # sequential | fixed | random
    start_seed: 1000
  naming:
    template: "phr_{phr}_i{value}_s{sample}"
```

- **Seeds** — `sequential` gives structure *k* the seed `start_seed + k` (fully
  reproducible); `fixed` uses one seed for all; `random` draws a fresh seed per
  structure. The actual seed is recorded in the manifest.
- **Naming** — `naming.template` builds the filename. Placeholders:
  `{phr}` (actual PHR, 2 decimals), `{value}` (first axis value), `{sample}`
  (1-based replicate), `{index}` (1-based global index), `{seed}`, and any axis
  as `{model1_intensity}` (dots → underscores). Because `{phr}` is only known
  after generation, the rest of the name must be unique on its own — include
  `{index}`, or `{value}`+`{sample}` as above.

**Cases mode** (`run.mode: cases`) generates an explicit list instead of a
sweep, each entry overriding what it names:

```yaml
cases:
  - name: hand_tuned_low_phr
    seed: 7
    model1: { intensity: 90 }
```

### Outputs

Under `output_dir`:

- `sphere_structures/<name>_spheres.npz` — compact `centers`, `radii`, `origin`,
  `box_size`.
- `voxel_structures/<name>_voxel.npz` — solver-ready `phase`, `voxel_size`,
  `origin`, `box_size`.
- `manifest.csv` — one row per structure: name, seed, status, all parameters,
  `filler_fraction`, `phr`, and the output paths. This is your lookup table for
  selecting structures by PHR.

The `.npz` files represent the full simulation box, not just the occupied region.

> Note: with a `{phr}`-based name and `on_existing: skip`, a re-run still
> regenerates each structure (deterministically from its seed) before it can
> tell the file already exists — so `skip` avoids re-writing, not re-computing.

## Single structure (CLI)

`combined_particle_models.py` generates one structure and can visualize it.
Run it from the repository root (so the shared `file_viewers` package resolves):

```powershell
.venv\Scripts\python microstructure_generation\combined_particle_models.py --filename-base my_structure --seed 41
```

Defaults live in the `Default run settings` block at the top of the script.
Common flags: `--box-size`, `--model1-intensity`, `--model1-radius`,
`--model2-intensity`, `--model2-min-r1`, `--model2-max-r1`, `--model3-intensity`,
`--model3-radius`, `--seed`, `--voxel-grid NX NY NZ`, `--notes`, and
`--no-visualize`. Run with `--help` for the full list. Like the batch runner,
sphere and voxel outputs go into the `sphere_structures/` and `voxel_structures/`
sub-folders.

## Files

- `generate_structures.py` — YAML batch runner (parallel, writes the manifest).
- `generation_config.py` — loads the YAML and expands it into per-structure specs.
- `Run_configs/` — the generation YAML configs; `generation_phr_sweep.yaml`
  is the annotated example / template.
- `combined_particle_models.py` — core generation pipeline and single-run CLI.
- `calculate_phr.py` — PHR calculation from a saved `.npz`.
- `boolean_sphere_models/` — the three Boolean models and their shared base.
- `2D_generation/` — the 2D tooling: `cross_section_generator.py` (render 2D
  section images from a saved structure),
  `create_2d_testset_images_by_phr.py` and `smooth_2d_occlusions.py`.
