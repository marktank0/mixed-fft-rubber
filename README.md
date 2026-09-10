# MixedFFT — FFT homogenisation of filled rubber

An FFT-based solver for the mechanical response of a representative volume
element (RVE) at finite strain. The production solver is the **mixed**
FFT-Galerkin formulation (deformation gradient *and* pressure as unknowns),
which stays well behaved for incompressible and nearly incompressible
hyperelastic phases. A classical strain-only FFT-Galerkin solver is also
included (`fg/fft.py`) but is legacy and not used by any runner.

Based on *"A mixed FFT-based approach for incompressible or slightly
compressible hyperelastic solids under finite deformation"* (WANG Mingchuan).

---

## The two main components

The project is two pipelines that run back to back:

```
  1. microstructure_generation/        2. FFT_simulation/
     packs spheres into an RVE     ->     solves the mechanical response
     and voxelises it                     of each voxel structure
           |                                        |
           v                                        v
     *_voxel.npz (N^3 phase array)          Results/<experiment>/<case>/
                                            output.csv, solver_stats.json
```

**1. Microstructure generation** builds filled-rubber RVEs by Boolean sphere
packing at a target PHR (parts per hundred rubber), then voxelises them into an
`N x N x N` binary phase array saved as `.npz`.

**2. FFT simulation** takes those `.npz` structures plus a *charge* file (the
material pair and the load case) and solves for the homogenised stress
response, writing an averaged stress-strain history per run.

Both are driven by YAML configs and both run many cases in parallel.

---

## Setup

Python 3.9+ with:

| package | needed for |
|---|---|
| `numpy`, `scipy` | the solver (required) |
| `PyYAML` | reading configs — optional, a built-in fallback parser covers the subset used here |
| `matplotlib` | the result plots and `Plotting_scripts/` |
| `pyvista`, `trimesh` | **microstructure generation only** |
| `Pillow` | the 2D image tooling in `microstructure_generation/2D_generation/` |
| `torch` | the CNN surrogate in `CNN/` only |

```bash
pip install numpy scipy pyyaml matplotlib pyvista trimesh pillow
```

Every script resolves its inputs through `project_paths.py`, so runs can be
started from any working directory.

---

## 1. Generating microstructures

**Step 1 — pick or write a config.** They live in
`microstructure_generation/Run_configs/`. `generation_phr_sweep.yaml` is the
annotated template; it sets `output_dir`, the PHR sweep axis, the box size, the
sphere models and the voxel grid.

**Step 2 — preview the plan.** This lists every structure that would be built,
without generating anything:

```bash
python microstructure_generation/generate_structures.py microstructure_generation/Run_configs/generation_phr_sweep.yaml --dry-run
```

**Step 3 — generate.**

```bash
python microstructure_generation/generate_structures.py microstructure_generation/Run_configs/generation_phr_sweep.yaml
```

Useful flags: `--max-workers N` (parallel structures), `--mode sweep|cases`.

**Step 4 — collect the output.** Everything lands under the config's
`output_dir` (relative to the repository root):

```
microstructure_generation/3D_samples/<name>/
    voxel_structures/      *_voxel.npz  <- the input to the FFT solver
    manifest.csv           one row per structure, with its achieved PHR
    sphere_structures/     the centres+radii packings (only if
                           defaults.save_spheres_npz is on)
```

Which sub-folders appear depends on the `save_*` flags in the config;
`voxel_structures/` is the one the FFT solver consumes.

See `microstructure_generation/README.md` for the full config reference.

---

## 2. Running FFT simulations

**Step 1 — pick or write a run config.** They live in
`FFT_simulation/Run_configs/`.
**`full_template.yaml` documents every available setting**, with the valid
options next to each line — start there. The key blocks are:

- `experiment.output_root` — where results go
- `run.mode` — `cases` (an explicit list) or `batch` (glob a directory of `.npz`)
- `execution.max_workers` — how many cases solve in parallel
- `defaults.solver` — grid size `N`, load `increments`, `preconditioner`, tolerances
- `defaults.charge.path` — the material pair and load case (see below)

**Step 2 — preview the plan.** This prints every resolved case and its output
path without solving anything:

```bash
python FFT_simulation/batch_run.py FFT_simulation/Run_configs/50_improved_structure.yaml --dry-run
```

**Step 3 — run.**

```bash
python FFT_simulation/batch_run.py FFT_simulation/Run_configs/50_improved_structure.yaml
```

Useful flags: `--max-workers N`, `--on-existing skip|overwrite|error`,
`--terminal-output` (print solver progress here instead of to each case's
`run.log`), `--base-path` (repoint the whole run, e.g. on a server).

**Step 4 — collect the output.** One directory per case under
`experiment.output_root`:

```
Results/<experiment>/
    resolved_config.yaml       exactly what was run
    <structure>_output/
        output.csv             averaged F and P at every increment boundary
        solver_stats.json      convergence history, status, timings
        run_metadata.txt       inputs, settings, wall time
        run.log                solver progress (when log_to_file is on)
        stress_strain.png      P11 vs F11
        F11_vs_F22_F33.png     transverse contraction
        fields.vti             local fields for ParaView (if save_fields: true)
```

**To sweep contrast** (every structure at every filler/matrix stiffness ratio),
use the dedicated runner — it generates the charge file per contrast for you:

```bash
python FFT_simulation/contrast_sweep.py FFT_simulation/Run_configs/contrast_sweep.yaml
```

Relative paths inside a run config (`structure_path`, `charge.path`,
`output_root`, `batch.structures.glob`) resolve against the **repository root**
— not the config file, and not the working directory.

---

## Defining the problem: structure and charge

### Structure (the phase field)

Normally a `.npz` from the generator, holding an `N x N x N` array of 0.0
(matrix) and 1.0 (filler) under the key `phase`. A legacy `phase.txt` — the
same array flattened to one value per line — is also accepted. Note that a
63x63x63 phase is 250 047 values.

The solver hard-codes phase `0` as material A (matrix) and `1` as material B
(filler); any other value is an error.

### Charge (materials + load case)

A four-row text file, by convention in `FFT_simulation/Run_configs/Charges/`:

```
#first two lines: model:---0)model num 1) p1.. 2)p2...
#First line; Phase 0 (matrix): model - E - Poisson ratio - padding
1.0	10.0	0.48	0.0	0.0	0.0	0.0	0.0	0.0
#Second line; Phase 1 (filler): model - E - Poisson ratio - padding
1.0	1000.0	0.30	0.0	0.0	0.0	0.0	0.0	0.0
#charge dF: total macroscopic deformation, here F11 = 1.0 (100% strain)
1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
#(Charge type) P-1 or F-0: 0 = control this component by Fij, 1 = by average Pij
0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	1.0
```

The leading number on each material line selects the constitutive model in
`FFT_simulation/fg/constitutive_incompressible/<n>.py`:

| model | law |
|---|---|
| `1.0` | Neo-Hookean |
| `2.0` | Mooney-Rivlin |

The macroscopic load is split into steps by `solver.increments` in the run
config; the solver cuts sub-steps automatically if an increment fails to
converge.

---

## Solver options worth knowing

Set under `defaults.solver` in a run config; `full_template.yaml` documents all
of them.

| setting | options | meaning |
|---|---|---|
| `preconditioner` | `green`, `green_jacobi`, `gmres`, `none` | `green` is the production choice: the Green operator built on a homogeneous reference tangent, applied through GMRES. (It was called `reference` before; the old spelling still loads.) |
| `reference` | `mean`, `matrix`, `mid` | *which* tangent the Green symbol linearises around — unrelated to `preconditioner`. `matrix` is best at high contrast. |
| `discretization` | `fourier`, `willot` | spectral derivative, or Willot's rotated finite-difference scheme (less ringing at sharp interfaces) |
| `forcing` | `eisenstat_walker`, `fixed` | inexact Newton: loosen the inner solve early, tighten as the residual falls |
| `gmres_restart` | integer | Krylov basis length — the dominant memory cost |

---

## Using the solver directly from Python

For one-off work, bypass the YAML layer:

```python
import project_paths; project_paths.ensure_import_paths()
from fg.mxfft import FFTSolver

prob = FFTSolver(
    "path/to/structure_voxel.npz",
    charge_path="FFT_simulation/Run_configs/Charges/Neo_1.0_E10-1000.txt",
    output_path="Results/scratch",
    N=63,
)
# incre_list is the load path: [1.0] is one 100% step,
# [0.25]*4 is four steps of 25%.
prob.calculate(incre_list=[0.1]*10, preconditioner="green", reference="matrix")

print(prob.Ps[-1])   # averaged 1st Piola-Kirchhoff stress at the last increment
print(prob.Fs[-1])   # averaged deformation gradient
```

`output.csv` and `solver_stats.json` are always written to the output path.

---

## Visualisation

Local field export is **built in** — no external VTK library needed. Set
`outputs.save_fields: true` in a run config (or `save_fields=True` on
`calculate`) and the solver writes `fields.vti`, holding all nine components of
F and P plus the phase and the pressure, ready to open in ParaView.

For homogenised results, `Plotting_scripts/` renders stress vs. PHR,
reinforcement vs. contrast and similar curves directly from a results
directory:

```bash
python Plotting_scripts/plot_p11_vs_phr.py Results/contrast_sweep/E10-250
```

---

## Repository layout

```
FFT_simulation/
    fg/                     the solvers, preconditioners, constitutive models
                            and VTI export (all located relative to this
                            package, so the working directory never matters)
    batch_run.py            YAML-driven runner (one or many cases)
    contrast_sweep.py       every structure x every filler/matrix contrast
    run_case.py             one simulation case, start to finish
    benchmark_suite.py      solver-improvement benchmark sweep
    Run_configs/*.yaml      run configs (full_template.yaml documents them all)
    Run_configs/Charges/    charge files (material pair + load case)

microstructure_generation/
    generate_structures.py  YAML batch runner for structure generation
    generation_config.py    YAML -> per-structure specs
    combined_particle_models.py   the generation pipeline and single-run CLI
    calculate_phr.py        PHR of a saved .npz
    Run_configs/*.yaml      generation configs
    boolean_sphere_models/  the Boolean sphere models
    2D_generation/          2D cross-section / image tooling
    file_viewers/           .npz inspection helpers
    3D_samples/             generated structures (the solver's input)

3D_samples/                 small hand-made test structures and charges
Results/                    all run output, one directory per experiment
Plotting_scripts/           figures from a results directory
CNN/                        3D CNN surrogate: voxel structure -> P11 at 30% strain
docs/                       solver theory, benchmarks and investigation notes

project_paths.py            canonical location of everything above
simulation_config.py        run YAML -> resolved case dictionaries
result_plots.py             per-case plots
run_metadata.py             per-case metadata
```

---

## Further reading

`docs/` covers the solver work in depth:

- `solver_improvements.md` — robustness features and every solver setting
- `green_reference_preconditioning.md` — the Green preconditioner and the
  compatible-subspace fix
- `inexact_newton_and_reference_tangent.md` — Eisenstat-Walker forcing and the
  reference tangent modes
- `discretization.md` — the Willot scheme
- `benchmarking.md` — how the benchmark suite compares against the pre-change
  solver
- `structure-generation.md` — the generation and 2D tooling
