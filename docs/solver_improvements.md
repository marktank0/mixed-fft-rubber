# Mixed FFT Solver Improvements: What Changed, How It Works, and Why

This document describes every change applied to the mixed FFT solver
(`FFT_simulation/fg/mxfft.py` and supporting files) during the high-contrast convergence
work, the mechanism behind each change, the theory it rests on, and the
validation that was performed. The original planning document with the
full failure-mode diagnosis and literature survey is
`docs/high_contrast_convergence_plan.md`; this file documents what was
actually implemented.

Motivation in one paragraph: the solver originally converged well at a
filler/matrix stiffness contrast of ~10x but failed, silently corrupted
results, or took hours at the contrasts (100x-1000x) needed to represent
carbon black in rubber. The changes below made the solver (a) fast enough
to iterate with (vectorization, ~5x wall time at N=31 and the enabler for
N=63 grids), (b) robust at high contrast (globalized Newton with load-step
cutting: a 500x-contrast case that previously died now converges), and
(c) honest (a run can no longer save non-equilibrated states as if they
were results; every case carries an explicit converged/failed status, and
partial results survive interruption).

---

## 1. Vectorization of the hot loops

### 1.1 Batched constitutive evaluation (`umat_field`)

**Files:** `FFT_simulation/fg/constitutive_incompressible/1.py` (Neo-Hookean),
`FFT_simulation/fg/constitutive_incompressible/2.py` (Mooney-Rivlin), `FFT_simulation/fg/mxfft.py`.

**Before.** `constitutive()` looped over all N^3 voxels in Python
(29,791 at N=31; 250,047 at N=63), calling `umat(f, p, params)` per voxel.
Each call did several 3x3 einsums, a `det`, and an `inv` on tiny arrays,
so Python and NumPy call overhead dominated the cost, not arithmetic.

**After.** Each constitutive file exposes

```python
umat_field(f, yl, parameters, need_tangent=True)
# f: (m,3,3) deformation gradients, yl: (m,) pressures
# returns P (m,3,3), K4 (m,3,3,3,3) or None, JFmT (m,3,3), kappa_inv (float)
```

which evaluates all voxels of one phase in single batched NumPy calls:
`np.linalg.det` and `np.linalg.inv` operate natively on stacks of
matrices, and every per-voxel einsum becomes one einsum with a leading
voxel axis (e.g. `k01[i,j,m,n] = f[i,j] finv[n,m]` becomes
`np.einsum("vij,vnm->vijmn", f, finv)`). The solver splits the grid into
a matrix mask and a filler mask (`phase == 0` / `phase == 1`), gathers
each phase's voxels into an `(m,3,3)` stack, calls `umat_field` once per
phase, and scatters the results back into the full-field arrays. The
Mooney-Rivlin tangent, previously an 81-iteration `itertools.product`
loop per voxel, was rewritten as closed-form batched einsums of the same
`dM` expression documented in
`docs/mooney_rivlin_incompressible_implementation.md`.

The original per-voxel `umat(f, yl, parameters)` signature is kept as a
thin wrapper around `umat_field`, so the standard solver (`FFT_simulation/fg/fft.py`)
and older validation scripts continue to work unchanged.

`need_tangent=False` skips the tangent `K4` (by far the most expensive
output) and returns it as `None`; the line search (Section 3) uses this
because it only needs the stress to evaluate the residual.

**Theory/correctness.** This is a re-expression of identical arithmetic;
only the summation order changes, so agreement is at machine-precision
level. Verified: batched vs original per-voxel implementations agree to
~1e-13 max component error over random states for both models, all
parameter branches (including nu = 0.5).

### 1.2 Vectorized projection-operator assembly (`build_Ghat4`)

The Green-projection symbol Ghat4 was assembled with a Python loop over
81 * N^3 entries (~2.4M iterations at N=31). It is now built from outer
products of the frequency grid: with `q` the (3,N,N,N) frequency field,

```
QQ[j,m] = q_j q_m / |q|^2        (zero frequency handled separately)
Ghat4[i,j,l,m] = delta(i,l) * QQ[j,m]
```

The zero-frequency mode keeps the original special handling: it is zeroed
except for identity entries on the stress-controlled components, which is
what enforces the macroscopic mixed F/P control. Verified to be exactly
equal (0.0 difference) to the original loop for several N and
stress-control sets.

### 1.3 Field averages

`FFTSolver.__average` was a Python triple loop over voxels; it is now
`A2.mean(axis=(2,3,4))`.

### 1.4 FFT backend and shift hygiene

The transforms were `np.fft.fftn/ifftn` wrapped in `fftshift/ifftshift`
applied to **all** axes, including the 3x3 (or 10-component) leading
axes. Shifting a component axis before the transform and unshifting it
after is the identity (fftshift and ifftshift are exact inverses per
axis, and the FFT never touches those axes), so those shifts were pure
wasted array copies. Two changes:

- shifts are now applied to the spatial axes only (provably identical
  output);
- the transforms use `scipy.fft.fftn/ifftn` with `workers=-1`
  (multithreaded), in both `FFT_simulation/fg/mxfft.py` and `FFT_simulation/fg/preconditioning.py`.

**Why this matters.** After 1.1-1.3, profiling showed the FFTs dominate:
at N=31 one Krylov iteration cost ~85 ms in the operator (`KdX`) plus
~96 ms in the reference-preconditioner apply, almost all of it FFT work.
This is the correct cost floor for an FFT-based solver - the remaining
optimizations must reduce the *number* of Krylov iterations (see
Section 8), not their price.

**Measured effect of Section 1 in total:** a 3-increment contrast-10 case
at N=31 went from 97 s to 37 s wall time with results identical to
1e-14 relative; constitutive assembly of the full N=31 grid costs
~128 ms with tangent, ~26 ms stress-only.

---

## 2. Residual-based convergence and fail-loud error handling

**File:** `FFT_simulation/fg/mxfft.py` (`newton_increment`).

**Before.** The Newton loop stopped when the *step size* was small:
`norm(dF)/norm(F) < 5e-5`. If the inner Krylov solver returned a
not-converged flag, the code `break`-ed out of the Newton loop and fell
through to saving the average stress of a **non-equilibrated** field into
`output.csv`, indistinguishable from a valid result, and the next
increment continued from that bad state.

**After.** Convergence is judged on the true nonlinear residual `b` -
the quantity the discretized problem is defined by: the G-projected
stress imbalance `G(TbarP - P)` (equilibrium + macroscopic control) and
the pressure-equation residual `1 - J + p/kappa` (incompressibility).
Each block is normalized by its own norm at the start of the
sub-increment and both must fall below `tol_rel` (default 1e-5, with an
absolute floor `tol_abs = 1e-10`):

```
|b_F|  <= max(tol_rel * |b_F,0|,  tol_abs)   and
|b_p|  <= max(tol_rel * |b_p,0|,  tol_abs)
```

Per-block normalization matters because the two blocks have different
physical units; a single mixed norm lets one block's scale mask
non-convergence of the other - a known pitfall in mixed/saddle-point
systems.

**Theory.** Stopping on the residual is the textbook Newton criterion.
The step-size test is specifically dangerous in the high-contrast regime
because a *stalled* Krylov solve returns a small (wrong) step - so the
old test could report convergence precisely when the linear solver had
failed.

A Krylov failure is now never accepted: when GMRES fails to converge
within the configurable iteration cap (`max_gmres_iter`, Section 6), the
sub-increment is marked failed, which triggers load-step cutting
(Section 4). If cutting is exhausted the case ends with status
`"failed"`; nothing non-equilibrated is ever written as a result.

The Newton iteration count itself is deliberately *not* capped: once
each linear solve converges, Newton steps keep making progress (each
accepted step must strictly decrease the residual), so the failure mode
at high contrast is the linear solve, not the outer iteration. If Newton
ever stagnates, the line search's strict-decrease requirement fails and
routes the sub-increment into the same step-cutting path, so termination
is still guaranteed without an explicit cap.

Note on SciPy semantics: `scipy.sparse.linalg.gmres`'s `maxiter`
argument counts *restart cycles*, not iterations. The solver therefore
takes `max_gmres_iter` as a total inner-iteration budget (the same count
printed as `gmres iter N` in the log) and converts it to whole restart
cycles internally, so the effective cap is `max_gmres_iter` rounded up
to a multiple of the restart length.

**Calibration/validation.** With `tol_rel = 1e-5` the contrast-10
regression case reproduces the pre-change results to 1e-14 relative with
the same iteration counts. A synthetic failure test (GMRES crippled to
`maxiter=1`) confirms: step cuts happen, the case is marked failed, zero
corrupt rows are saved, and `plot_p11_vs_phr.py` skips the folder.

---

## 3. Globalized Newton: det(F) guard and backtracking line search

**File:** `FFT_simulation/fg/mxfft.py` (`newton_increment`).

**Before.** The full Newton step was always applied (`F += dF`,
`p += dp`). At high contrast, thin matrix ligaments between stiff
particles carry strongly amplified strain; a full step from a poor
predictor can drive some voxel to `det F <= 0`, after which the stored
energy is undefined (`J^(-2/3)`, `inv(F)` on an inverted state produce
NaNs) and the run is unrecoverable.

**After.** Every Newton update is safeguarded. For the solved direction
`(dF, dp)`, step lengths `alpha = 1, 1/2, 1/4, ...` (up to
`max_backtracks = 8` halvings) are tried until both conditions hold:

1. **Domain guard:** `min_voxels det(F + alpha dF)` stays above 5% of the
   current `min_voxels det(F)` (a relative floor, so genuinely extreme
   but valid states are not blocked). This is the finite-strain analogue
   of keeping the iterate inside the domain of the energy functional.
2. **Sufficient decrease:** the block-normalized residual norm (the same
   merit quantity as Section 2) strictly decreases.

Each trial costs one stress-only constitutive evaluation
(`need_tangent=False`, Section 1.1) plus one residual assembly - tens of
milliseconds at N=31. If no trial succeeds, the sub-increment fails and
the load step is cut (Section 4).

**Theory.** Backtracking line search on the residual norm is the
standard globalization of Newton's method (Nocedal & Wright ch. 11;
Deuflhard). It does not change the solution being sought - only the path
to it - and Newton's local quadratic convergence is untouched because
near the solution the full step `alpha = 1` always satisfies the
decrease condition. Residual-norm backtracking is not a *globally*
convergent scheme for nonsymmetric systems in theory (the residual can be
nonmonotone near saddle points), which is why it is paired with load-step
cutting: the practical contract is "either progress, or a smaller load
step", the same combination FE codes use for finite-strain problems.

**Measured.** At contrast 10, `alpha = 1` is accepted on every step
(the safeguard never activates - important, because frequent activation
at low contrast would indicate a tangent inconsistency, not a hard
problem). At contrast 500, the line search takes `alpha = 0.5` on the
first Newton step of each increment - exactly the mechanism that
previously produced inverted voxels - and full steps afterwards.

---

## 4. Adaptive load stepping with rollback

**File:** `FFT_simulation/fg/mxfft.py` (increment loop in `calculate`).

**Before.** The load path was a fixed list of increments (e.g. ten steps
of 0.1 in F11). Whether a 0.1 step lies inside Newton's convergence
basin depends on contrast and on the particular microstructure; one hard
increment killed the entire case, and the only workaround was manually
running everything with tiny increments (e.g. 40 x 0.025), paying that
cost even where it was unnecessary.

**After.** The prescribed increments become *targets*. Each target is
reached by sub-steps:

- The Newton solve operates on trial copies; the last accepted state
  `(F, p)` is kept, so a failed sub-increment **rolls back** cleanly.
- On failure (Newton cap, Krylov failure, or line-search dead end) the
  sub-step is halved and retried, down to `min_substep_ratio` (default
  1/16) of the original increment; below that the case stops with status
  `"failed"`.
- After two consecutive easy sub-increments (<= 4 Newton iterations) the
  sub-step grows by 1.5x, capped at the original increment size.
- **Output contract preserved:** results are recorded **only at the
  original increment boundaries** (F11 = 1.1, 1.2, ...), never at
  interior sub-steps, so `output.csv`, the plotting scripts, and all
  downstream analysis are unaffected by how the solver sub-stepped.
- The stress-control target `TbarP` is always computed from the
  *accumulated* load factor, so mixed F/P control scales consistently
  under sub-stepping.

**Theory.** Hyperelasticity is path-independent: the converged state at
a load level does not depend on the sub-steps used to reach it (unlike
plasticity), so sub-stepping is purely a solver aid and results at the
reported boundaries are unchanged up to solver tolerance. One caveat is
non-uniqueness: finite-strain problems can in principle have multiple
equilibria (microscopic buckling of stiff networks), where different
step sequences could select different branches; at moderate filler
fractions under tension this is unlikely, but it is the reason
path-independence should be spot-checked (same case, two different
initial step sizes) when moving to strongly networked microstructures.

---

## 5. Observability, incremental saving, and live logs

**Files:** `FFT_simulation/fg/mxfft.py`, `FFT_simulation/run_case.py`, `run_metadata.py`,
`plot_p11_vs_phr.py`, `simulation_config.py`.

- **`solver_stats.json`** is written next to `output.csv` for every run:
  per sub-increment it records the step size, load interval, Newton
  iteration count, Krylov iterations per Newton step, line-search alphas,
  residual histories, wall time, failure reason (if any); plus totals
  (step cuts) and the final status. This is how "where do the iterations
  go" questions are answered with data instead of guesses.
- **Status flag.** Every case ends as `"converged"`,
  `"converged_with_step_cuts"`, or `"failed"`; while running it is
  `"in_progress"`. The status is written into `solver_stats.json` and
  `run_metadata.txt`, and `plot_p11_vs_phr.py` skips folders whose status
  is `"failed"` so contaminated points can never silently enter a plot.
- **Incremental saving.** `output.csv` and `solver_stats.json` are
  rewritten after **every completed increment boundary**, not only at the
  end of the run. A case that fails - or a process killed hours into a
  batch - keeps everything up to its last converged increment.
  (Previously, an interrupted run saved nothing at all.)
- **Live log.** `run.log` (used when `log_to_file` is on) is now opened
  line-buffered, so every printed line lands on disk immediately and the
  log can be followed while the case runs. Previously the file was fully
  buffered and typically appeared only when the case finished.

---

## 6. Solver settings (YAML)

The robustness knobs are plumbed through `simulation_config.py` ->
`FFT_simulation/run_case.py` -> `FFTSolver.calculate`. In a run config, under the
`solver:` block of a case (or `defaults:`):

| setting | default | meaning |
|---|---|---|
| `max_gmres_iter` | 1000 | total GMRES iteration cap per linear solve (rounded up to whole restart cycles); hitting it fails the sub-increment (then: step cut, or stop if cuts exhausted). Newton iterations are not capped - the linear solve is what stops converging at high contrast |
| `min_substep_ratio` | 1/16 | smallest allowed sub-step as a fraction of the original increment; set `1.0` to disable retries entirely, i.e. "first failure stops the case" |
| `tol_rel` | 1e-5 | relative residual tolerance per block (Section 2) |
| `gmres_restart` | auto | GMRES restart length; auto = memory-aware (Section 7) |
| `preconditioner` | `green` | `green` (the Green preconditioner, called `reference` before this rename - the old spelling is still accepted), `green_jacobi`, `gmres`, or none |

All results up to the last converged increment are saved in every
stopping scenario (cap exceeded, cuts exhausted, external kill).

---

## 7. Memory-aware GMRES restart length

**File:** `FFT_simulation/fg/mxfft.py`.

The GMRES restart length was hardcoded at 100. The Krylov basis is
`(restart+1) x (10 N^3)` doubles: ~240 MB at N=31 but **1.9 GiB at
N=63** - a guaranteed out-of-memory crash on a 16 GB machine, discovered
the first time the vectorized solver made N=63 runs practical. The
restart length now defaults to a memory-aware value capping the basis at
~800 MB (still 100 at N=31, ~40 at N=63) and can be overridden per run
via `solver.gmres_restart` (worth raising on machines with more RAM,
since shorter restarts can slow GMRES convergence on hard systems).

---

## 8. What was measured, and what remains open

Validation summary (all on the same structure/charge unless noted):

| check | result |
|---|---|
| batched vs original constitutive models (random states, both models) | <= ~1e-13 |
| vectorized vs original Ghat4 | exact (0.0) |
| full solver vs pre-change baseline, contrast 10, N=31 | 1e-14 relative, alpha=1 every step |
| wall time, 3 increments, contrast 10, N=31 | 97 s -> 37 s |
| contrast 500 (E 10 vs 5000), N=31, 3 increments | converges, 0 step cuts, 6 Newton its/increment, alpha=0.5 twice per increment, 130-240 GMRES its/solve, ~190 s/increment |
| old-code 40x0.025-increment run vs new solver, contrast 500 | P11 agree within 0.7% |
| synthetic Krylov failure | step cuts -> status `failed`, no corrupt rows, plot skips folder |
| incremental save + live log, N=63 | log grows during run; `output.csv` contains increment 1 while increment 2 still running; adaptive restart = 39, no OOM |
| interrupted N=63 run (session killed mid-increment) | `output.csv` retained the completed increment; stats retained with status `in_progress` |

**Known limits / next levers** (documented in detail in the plan file):

- ~~inexact Newton inner tolerances~~ and ~~a configurable reference
  tangent~~ are **now implemented** (C5, C6). See
  `docs/inexact_newton_and_reference_tangent.md`. C5 is the win:
  Eisenstat-Walker forcing terms cut Krylov iterations 2.29x and wall time
  ~2.5x at contrast 100 with the Newton count unchanged. C6 did *not* pay
  off at contrast 100 / phi = 0.09 - the matrix-phase reference was ~10 %
  worse than the volume average - so the default reference is unchanged.
- The assumption stated here previously, that "a preconditioner and an
  inner tolerance change the iteration, not the fixed point", holds for the
  inner tolerance but **is false for the preconditioner as implemented**.
  The Green symbol is built as `G K_0`, whose pseudo-inverse does not map
  into `range(G)` unless `K_0` is isotropic, so the preconditioner takes
  GMRES out of the compatible subspace and different reference tangents
  converge to different states (P11 spread ~2.8e-3 relative at residual
  ~1e-12). **This has now been fixed**: the symbol is built as the reference
  operator restricted to the compatible subspace. An unpreconditioned control
  confirms the corrected answer to 9 significant figures, and the pre-fix
  result was wrong by **+17.4 % in P11**. Every run made with
  `preconditioner="reference"` is affected and needs regenerating; set
  `precond_restrict=False` to reproduce the old numbers. See
  `docs/green_reference_preconditioning.md`.
- Physics interpretation of results (Mori-Tanaka vs Guth-Gold, effective
  volume fraction, aggregate morphology, percolation) is a
  microstructure question, not a solver question; see the discussion at
  the end of `docs/high_contrast_convergence_plan.md`.

---

## 9. Files touched

| file | change |
|---|---|
| `FFT_simulation/fg/constitutive_incompressible/1.py` | batched `umat_field` (Neo-Hookean); `umat` kept as wrapper |
| `FFT_simulation/fg/constitutive_incompressible/2.py` | batched `umat_field` (Mooney-Rivlin, closed-form tangent); `umat` kept as wrapper |
| `FFT_simulation/fg/mxfft.py` | vectorized Ghat4/constitutive/averages; scipy.fft backend; residual-based Newton with det-guard + line search; adaptive sub-stepping with rollback; statuses; `solver_stats.json`; incremental saving; memory-aware GMRES restart; new `calculate()` parameters |
| `FFT_simulation/fg/preconditioning.py` | multithreaded scipy.fft, spatial-axes-only shifts; `reference_average()` for the C6 reference-tangent modes |
| `FFT_simulation/run_case.py` | passes new solver settings; line-buffered `run.log`; passes solver status to metadata |
| `run_metadata.py` | writes `Solver status:` line |
| `simulation_config.py` | exposes `max_newton`, `min_substep_ratio`, `tol_rel`, `gmres_restart`, and the C5/C6 keys `reference`, `forcing`, `inner_rtol`, `eta_max`, `eta_min` in the YAML schema |
| `plot_p11_vs_phr.py` | skips cases whose `solver_stats.json` status is `failed` |

## 10. References

- Nocedal, Wright, *Numerical Optimization*, ch. 11 (globalized Newton).
- Deuflhard, *Newton Methods for Nonlinear Problems*.
- Eisenstat, Walker (1996), *Choosing the forcing terms in an inexact
  Newton method*, SIAM J. Sci. Comput. (next lever, not yet implemented)
- Kabel, Boehlke, Schneider (2014), *Efficient fixed point and
  Newton-Krylov solvers for FFT-based homogenization of elasticity at
  large deformations*, Comput. Mech.
- Schneider (2021), *A review of nonlinear FFT-based computational
  homogenization methods*, Acta Mechanica.
- Zeman, Vondrejc, Novak, Marek (2010), JCP; Vondrejc, Zeman, Marek
  (2014), CMAME (FFT-Galerkin + Krylov acceleration).
- `docs/high_contrast_convergence_plan.md` (this repository): full
  failure-mode diagnosis (D1-D8), plan (C0-C8), and measured results.
