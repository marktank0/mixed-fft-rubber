# High-Contrast Convergence and Performance Plan

Status: **C1-C4 implemented on the `FFT_for_higher_contrast` branch**
(vectorization, residual-based convergence with fail-loud handling,
det-guard + line search, adaptive load stepping). C1 additionally moved the
FFTs to multithreaded `scipy.fft` and restricted the fftshifts to the
spatial axes after profiling showed the FFTs dominate once the constitutive
loops are vectorized. C0 landed in reduced form (`solver_stats.json` +
solver status in `run_metadata.txt`).

**C5 and C6 are now implemented** - see
`docs/inexact_newton_and_reference_tangent.md` for theory, implementation
and measurements. Summary: C5 (Eisenstat-Walker forcing terms) gives
**2.29x fewer Krylov iterations and ~2.5x lower wall time at contrast 100
with the Newton count unchanged**, and `forcing="fixed"` reproduces the
pre-change solver bitwise. C6 (configurable reference tangent) is
implemented and measured, but **did not help** at contrast 100 /
phi = 0.09: the matrix-phase reference was ~10 % *worse* than the volume
average, so D5's expectation is not confirmed at that contrast and the
default stays `"mean"`.

Validating C6 also uncovered a **pre-existing defect in the Green
preconditioner**: because the symbol is built as `G K_0` rather than
`G K_0 G`, the preconditioner maps iterates out of the compatible subspace,
so different reference tangents converge to genuinely different states
(~2.8e-3 relative in P11) with fully converged residuals. See section 5 of
the C5/C6 note. A tested one-line fix exists but has not been applied
because it changes every result produced with `preconditioner="reference"`.

C7 and C8 remain open.

Measured after C1-C4 (structure phr_18.45, N = 31, reference
preconditioner): contrast 10, 3 increments: 37 s, results identical to the
pre-change solver to 1e-14. Contrast 500 (E 10 vs 5000), 3 increments:
**converges** with zero load-step cuts, 6 Newton iterations per increment
(line search takes alpha = 0.5 twice per increment, alpha = 1 otherwise),
130-240 GMRES iterations per solve, ~190 s per increment (~30 min for a
full 10-increment case). The high Krylov counts confirm D5/D7: C5 and C6
are the next levers to bring 500x down to minutes per case.

Target: the mixed FFT solver in `FFT_simulation/fg/mxfft.py` (and, where changes are shared,
`FFT_simulation/fg/fft.py`), the constitutive files in `FFT_simulation/fg/constitutive_incompressible/`, and
the batch layer (`FFT_simulation/run_case.py`, `simulation_config.py`, `run_metadata.py`).

## 1. Goal

The 30-structure PHR sweep (`Results/30_struct_test_run`) used a filler/matrix
stiffness contrast of 10x (E = 100 vs 10). The resulting P11-vs-PHR relation is
nearly linear, which was shown (against Mori-Tanaka and Guth-Gold estimates) to
be the *correct* homogenization answer at that contrast. The non-linear
reinforcement reported in the rubber literature requires a much stiffer filler
(carbon black: ~10-80 GPa vs ~1-10 MPa for the matrix, i.e. contrast 10^3-10^4).

Raising the contrast currently makes runs fail to converge, or take hours.
This plan makes the solver:

1. robust at contrast up to ~10^3 (no silent failures, graceful step cutting),
2. fast enough that a 30-case sweep at contrast ~10^2-10^3 finishes in
   minutes-per-case at N = 31,
3. instrumented, so we can *see* where iterations go and verify each claim
   below instead of trusting it.

## 2. Diagnosis: why high contrast fails in this specific code

These are the concrete failure modes, in decreasing order of severity.

### D1. Silent acceptance of failed solves

`FFT_simulation/fg/mxfft.py` (Newton loop): if the Krylov solver returns `flag > 0`
(not converged), the code does `break` and falls through to saving the
average stress **of a non-equilibrated field**. The increment loop then
continues from that state. Consequences:

- corrupted points can enter `output.csv` indistinguishable from good ones;
- the next increment starts from a bad state, so one failure cascades.

Any high-contrast result produced so far that hit this path is suspect.
This is a correctness bug independent of performance.

### D2. Full Newton steps with no globalization

The update is always `F += dF`, `YALI += dYL` with step length 1. At high
contrast, thin matrix ligaments between stiff particles carry strongly
amplified strain. A full Newton step from a poor starting point can push a
voxel to `det F <= 0`; the constitutive law then evaluates `J**(-2/3)` and
`inv(F)` on an inverted state and produces NaN/garbage, after which nothing
recovers. There is no `det F > 0` check anywhere. Newton's local convergence
theory only guarantees convergence near the solution; globalization (line
search or step control) is the standard remedy (Nocedal & Wright, ch. 11;
Deuflhard, *Newton Methods for Nonlinear Problems*).

### D3. Fixed load increments

`incre_list` is fixed (ten steps of 0.1 up to F11 = 2.0). Whether 0.1 is
inside Newton's convergence basin depends on contrast and microstructure;
at contrast 10^3 it often is not. There is no retry-with-smaller-step
mechanism, so a single hard increment kills the whole case.

### D4. Step-size-based convergence test

Convergence is declared when `norm(dF)/norm(F) < 5e-5`. A stalled Krylov
solve returns a small (wrong) `dX`, which *passes* this test - the solver can
report convergence exactly when the linear solver failed. The true nonlinear
residual `b` (which the code already assembles: `G(TbarP - P)` and the
incompressibility residual) is never used as the stopping criterion.

### D5. Reference preconditioner degrades with contrast

The reference tangent is the arithmetic voxel average `K_ref = mean(K4)`.
With filler tangent ~10^3 x matrix and 15 % filler, `K_ref` is ~10^2 x the
matrix tangent. The preconditioned operator is then far from identity on
~85 % of the volume (matrix voxels), and GMRES iteration counts grow roughly
linearly with contrast instead of the ~sqrt(contrast) achievable with a good
reference (see Section 3).

### D6. Cost per iteration is dominated by Python loops

- `constitutive()` loops over all N^3 = 29,791 voxels in Python, calling
  `umat` per voxel (each doing several 3x3 einsums, `det`, `inv`).
- `__average` is a Python triple loop.
- `Ghat4` assembly loops over 81 * N^3 ~ 2.4M Python iterations.

Measured: ~200 s per case at contrast 10, where a well-vectorized NumPy
implementation of the same operations is expected to run the whole case in
tens of seconds. High contrast needs *more* iterations; at the current cost
per iteration that means hours.

(Measured after implementation: with the constitutive loops vectorized the
FFTs dominate; one Krylov iteration at N = 31 is ~85 ms matvec + ~96 ms
preconditioner apply, nearly all `np.fft` cost. Moving to multithreaded
`scipy.fft` and spatial-axes-only shifts brought the 3-increment
contrast-10 case from 97 s to 37 s.)

### D7. Fixed, tight inner tolerance

Every Krylov solve runs to `rtol = 1e-6`, even on the first Newton iteration
of an increment where the linearization error dwarfs 1e-6. Inexact Newton
theory (Eisenstat & Walker 1996; Dembo, Eisenstat & Steihaug 1982) says the
inner tolerance only needs to shrink as the outer residual shrinks; solving
to 1e-6 far from the solution buys nothing but wasted GMRES iterations.
Gelebart & Mondon-Cancel (2013) and Kabel, Bohlke & Schneider (2014)
demonstrated exactly this for FFT-based Newton-Krylov at finite strain:
loose inner tolerances (1e-1 to 1e-3 early) preserve the Newton path and cut
total FFT counts by large factors.

### D8. CG on a nonsymmetric system (default path)

The default (`preconditioner=None`) path solves the mixed saddle-point system
with `scipy.sparse.linalg.cg`. The mixed Jacobian is not symmetric (the
repository's own `docs/green_reference_preconditioning.md` notes this), so CG
carries no convergence guarantee here; it works at low contrast by luck.
The batch configs already use GMRES via `preconditioner: reference`, so this
mostly matters for anyone running defaults - but it should be fixed or at
least loudly documented while we are touching the loop.

## 3. Literature basis and what it predicts

- **Contrast sensitivity of FFT schemes.** The basic Moulinec-Suquet fixed
  point converges in O(contrast) iterations and diverges in the rigid limit;
  Krylov-accelerated Galerkin schemes (Zeman et al. 2010; Vondrejc, Zeman &
  Marek 2014) reduce this to roughly O(sqrt(contrast)) with a reasonable
  reference. Schneider's review (*A review of nonlinear FFT-based
  computational homogenization methods*, 2021) surveys this and identifies
  Newton-Krylov with inexact inner solves as among the most efficient
  approaches for finite-strain problems. Practical implication: going from
  contrast 10 to 10^3 should cost ~10x more Krylov iterations, not ~100x,
  *if* the preconditioner and tolerances are right. Infinite (truly rigid)
  contrast remains out of reach; we do not need it (see C8).

- **Inexact Newton.** Eisenstat & Walker (1996) give forcing-term choices
  that provably retain superlinear/quadratic outer convergence. Kabel et al.
  (2014) applied this in FFT homogenization at large deformations.

- **Globalized Newton.** Backtracking line search on the residual norm is
  textbook (Nocedal & Wright; Deuflhard) and is standard in FE codes for
  hyperelasticity, combined with load-step bisection. For hyperelasticity
  specifically, a step-length cap that keeps `det F > 0` everywhere is a
  well-known additional guard (it is the finite-strain analogue of keeping
  the iterate inside the domain of the energy).

- **Reference-medium choice.** For the *fixed-point* scheme the optimal
  reference is the average of extremal moduli (Moulinec & Suquet 1998). For
  Green-preconditioned Krylov methods the reference enters as a
  preconditioner; theory (and Ladecky et al.'s Green-Jacobi analysis) shows
  the preconditioned spectrum is bounded by the local-to-reference moduli
  ratios, so a reference close to the *majority phase* (matrix) conditions
  the matrix voxels well and concentrates the difficulty in the filler
  voxels, which GMRES handles as a low-ish-rank perturbation at moderate
  volume fractions. Which choice wins at phi ~ 0.05-0.19 and contrast 10^2-10^3
  is an empirical question - hence C6 makes it configurable and measures.

- **Saturation toward the rigid limit.** Hashin-Shtrikman/Mori-Tanaka type
  estimates approach their rigid-inclusion asymptote once the phase contrast
  exceeds ~10^2-10^3; beyond that the effective response changes by percent-level
  amounts. So the *physics* goal (visible non-linear reinforcement vs phr)
  does not require contrast 10^4. C8 quantifies where saturation happens for
  our microstructures and picks the cheapest sufficient contrast.

- **Resolution caveat.** At N = 31 the inter-particle matrix ligaments are
  1-2 voxels. At high contrast the local solution develops strong gradients
  there; voxelized interfaces also produce Gibbs-type oscillations. Kabel,
  Merkert & Schneider (2015, composite voxels) and the discretization
  literature (rotated staggered grids, finite-difference/Willot
  discretizations) address this. We will *measure* the resolution error
  (N = 31 vs 63 spot check) rather than assume it away; if it is large, the
  fallback is running production at N = 63 with the (now much cheaper)
  vectorized solver, not implementing composite voxels (deferred, Section 7).

## 4. Planned changes

Each change lists: what, why, expected improvement, correctness argument,
risks, and how we validate it. Order = implementation order.

### C0. Instrumentation and observability (prerequisite for honest evaluation)

**What.** Per increment, record: wall time, Newton iterations, Krylov
iterations per Newton step, final nonlinear residual, number of step cuts,
min `det F` over the grid. Write to the per-case log and a machine-readable
`solver_stats.json` next to `output.csv`; summarize in `run_metadata.txt`.
Add the failure status of the case (converged / converged-after-cuts /
failed) to `run_metadata.txt`.

**Why.** Every claim in C1-C8 ("fewer iterations", "no silent failures")
must be checkable. Currently only ad-hoc prints exist.

**Correctness.** Pure observation; no numerical behavior change.

**Risks.** None material. Slight log volume increase.

**Validate.** Rerun one existing case; `output.csv` must be identical;
stats file appears and is self-consistent with the printed log.

*Implemented (reduced): `solver_stats.json` with per-sub-increment Newton
iterations, Krylov counts, line-search alphas, residual histories, step
cuts and status; solver status line in `run_metadata.txt`.*

### C1. Vectorize the hot loops (speed; no behavior change intended)

**What.**

1. Rewrite `FFT_simulation/fg/constitutive_incompressible/1.py` and `2.py` to accept the
   full field `F (3,3,N,N,N)`, `p (N,N,N)` and return full-field `P`, `K4`,
   `JFmT`, `kappa_inv`. NumPy's `det`/`inv` are batched (operate on
   stacks `(...,3,3)`); every per-voxel einsum becomes one batched einsum
   with grid axes appended. Phase dispatch becomes two masked evaluations
   (matrix voxels / filler voxels) instead of an if-per-voxel.
2. Replace `FFTSolver.__average` with `A2.mean(axis=(2,3,4))`.
3. Vectorize the `Ghat4` assembly (outer products of the frequency grid;
   no Python loop over voxels).
4. Keep the existing per-voxel `umat` signature working (a thin wrapper or a
   `umat_field` alongside `umat`) so old scripts and the finite-difference
   validation in `docs/mooney_rivlin_incompressible_implementation.md`
   still run.

**Why.** D6. This is the enabling change: robustness (C2-C4) *spends*
iterations to buy reliability; that is only affordable if iterations are
cheap. It also makes the N = 63 resolution check (C8) feasible.

**Expected.** Assembly cost down ~50-100x (Python-loop overhead dominates
at 3x3 sizes); whole-case wall time at contrast 10 from ~200 s to ~10-30 s.
The Krylov matvec (`KdX`) is already vectorized; after C1 the FFTs and the
9x9xN^3 einsum in `KdX` become the floor, which is the correct floor for an
FFT solver.

**Correctness argument.** This is a re-expression of identical arithmetic;
only summation order changes, so differences are at roundoff level.
`2.py`'s Mooney-Rivlin tangent loop (`itertools.product` over 81 index
combinations per voxel) vectorizes to closed-form batched einsums that were
already derived in the implementation doc.

**Risks / critical view.**
- Memory: `K4` is already stored full-field (81 * N^3 doubles ~ 19 MB at
  N = 31, ~160 MB at N = 63) - unchanged. The transient batched einsum
  temporaries are of the same order. Fine at N <= 63.
- The masked two-phase evaluation computes nothing extra, but subtle
  indexing bugs are possible (e.g. mask alignment with `(3,3,N,N,N)`
  layout). Mitigated by the regression test below.

**Validate.** For random `F`, `p` states: max abs difference between old
per-voxel and new field implementation of `P`, `K4`, `JFmT` below 1e-12
(relative). Then one full case (contrast 10, existing structure): final
`Ps`/`Fs` match a full-precision baseline captured with the old code.

*Implemented. Measured: umat equivalence to ~1e-13; solver regression vs
old-code baseline 1e-14 relative; constitutive assembly 128 ms full grid
(tangent) / 26 ms (stress only) at N = 31. Additionally the FFTs were moved
to multithreaded `scipy.fft` (shifts on spatial axes only, provably
identical); 3-increment contrast-10 case: 97 s -> 37 s.*

### C2. Residual-based convergence + fail-loud error handling

**What.**

1. Convergence test on the true nonlinear residual: converged when
   `norm(b) <= max(tol_rel * norm(b_first_iter_of_increment), tol_abs)`,
   with `tol_rel = 1e-5` (calibrated in validation so that contrast-10
   results match current ones). Keep the step-norm check only as a
   secondary diagnostic printed to the log, not as the stopping rule.
2. If the Krylov solver returns `flag > 0`: do **not** accept the state.
   Trigger the step-cut path of C4. If cuts are exhausted, mark the case
   failed, write the failure into `run_metadata.txt` and `solver_stats.json`,
   and stop the case (batch layer already isolates cases from each other).
3. `plot_p11_vs_phr.py` and the batch summary must skip failed cases
   (status field from C0) instead of silently plotting them.

**Why.** D1, D4. Declaring convergence off `norm(dF)` is exactly wrong in
the regime we are entering (stalled linear solves make `dF` small).

**Expected.** No performance change at low contrast; at high contrast this
converts wrong-answers into either recovered runs (via C4) or clearly
flagged failures.

**Correctness argument.** `b` *is* the discrete equilibrium +
incompressibility residual the method is defined by (G-projected stress
imbalance, and `1 - J + p/kappa`); driving it to a small relative value is
the textbook Newton stopping criterion. The two blocks have different
physical units, so the implementation normalizes each block by its own
first-iteration norm and requires both to converge (avoids one block's
scale masking the other - this is a known pitfall of naive mixed-system
norms).

**Risks / critical view.**
- The relative-to-first-iterate normalization makes the effective tolerance
  increment-dependent; near-zero first residuals (tiny increments) need the
  absolute floor `tol_abs` to avoid over-solving. Both knobs must be logged.
- Changing the stopping rule *will* change results at the last digit.
  Calibrate `tol_rel` once against the contrast-10 baseline sweep.

**Validate.** Contrast-10 case: P11(F11) curve matches baseline within 1e-4
relative at every increment; Newton iteration counts within +-1 of before.
Synthetic failure test: force `maxiter=2` in GMRES and confirm the case is
marked failed and excluded from plots, not silently saved.

*Implemented: per-block residual convergence (`tol_rel=1e-5`, `tol_abs=1e-10`),
Krylov failure -> step cut -> loud failure status; plot script skips
status "failed" folders.*

### C3. det(F) guard + backtracking line search (globalization)

**What.** After solving for `dX = (dF, dp)`:

1. Compute the largest step `alpha <= 1` such that
   `min_voxels det(F + alpha dF)` stays above `0.05 * min_voxels det(F)`
   (cheap: batched det on the trial field).
2. Backtracking line search on the block-normalized residual norm:
   accept the first halving of `alpha` that decreases it. Each trial costs
   one stress-only constitutive evaluation + one residual assembly (cheap
   after C1).
3. If no trial decreases the residual, treat as a failed Newton step ->
   C4 step cut.

**Why.** D2. This is the single change most responsible for "does not
converge at all" -> "converges slowly at worst". Inverted voxels are
irrecoverable; a residual-increasing full step near a strong nonlinearity
is common at high contrast.

**Expected.** At contrast 10: line search accepts `alpha = 1` essentially
always (validate this - if not, something else changed). At contrast
10^2-10^3: occasional `alpha < 1` early in increments, dramatic reduction in
NaN blow-ups.

**Correctness argument.** Line search does not change the fixed point being
sought, only the path; accepted states always have `det F > 0` so the
stored energy is well-defined. Simple residual-decrease backtracking is not
globally convergent in theory (the residual norm is a nonsmooth merit
function for nonsymmetric systems, and saddle-point residuals can be
nonmonotone), but combined with step cutting (C4) the practical outcome is
"either progress or a smaller load step", which is the standard and
sufficient behavior in FE practice.

**Risks / critical view.**
- Cost: a few extra stress-only constitutive evaluations per Newton step in
  the worst case. After C1 each is ~tens of ms at N = 31; acceptable.
- A too-aggressive `det F` floor can stall progress in genuinely extreme
  states; the floor is relative (5 % of current min det), not absolute, to
  avoid this.
- Danger of hiding bugs: if the line search *frequently* activates at low
  contrast, that indicates a tangent inconsistency, not a hard problem -
  C0's stats make this visible.

**Validate.** Contrast-10 baseline unchanged (alpha = 1 acceptance rate
>= 99 %). Contrast-100+ single case: converges with logged alpha history;
final residual meets tolerance.

*Implemented (single safeguarded backtracking loop, max 8 halvings,
det floor checked per trial). Contrast-10 regression: alpha = 1 accepted
on every step, results identical to baseline.*

### C4. Adaptive load stepping with rollback

**What.** Wrap the increment loop:

1. Keep `(F, YALI)` copies of the last accepted increment (memory:
   10 * N^3 doubles ~ 2.4 MB at N = 31 - negligible).
2. On increment failure (Newton not converged in `max_newton = 15`
   iterations, Krylov failure after C2, or line-search dead end after C3):
   roll back, halve the sub-increment, retry. Minimum sub-increment
   `inc_min = inc/16`; below that, the case is marked failed.
3. After two consecutive easy sub-increments (Newton <= 4 iterations),
   grow the sub-increment by 1.5x, capped so that sub-steps always land
   exactly on the original `incre_list` boundaries (0.1, 0.2, ... 1.0).
4. **Output contract preserved:** `output.csv` rows are written only at the
   original increment boundaries, never at interior sub-steps. This keeps
   `plot_p11_vs_phr.py` (which matches F11 = 1.3, 1.6, 2.0 within
   tol = 0.05) and all downstream analysis working unchanged.

**Why.** D3. Fixed 0.1 steps are either wastefully small (low contrast) or
fatally large (high contrast); which one varies per microstructure sample,
and a 30-case batch cannot be hand-tuned per case.

**Expected.** High-contrast cases that currently die complete with a
moderate number of extra sub-steps. Low-contrast runtime roughly unchanged
(possibly slightly faster if step growth merges increments - it must not,
per the output contract, beyond increment boundaries).

**Correctness argument.** Hyperelasticity is path-independent: the converged
state at a load level does not depend on the sub-stepping used to reach it
(unlike plasticity). Sub-stepping is therefore purely a solver aid; results
at the reported boundaries are unchanged up to solver tolerance.

**Risks / critical view.**
- Non-uniqueness caveat: finite-strain problems can have multiple equilibria
  (microscopic instabilities/buckling of stiff networks). Different step
  sequences could in principle land on different branches. At phi <= 0.19
  and tension-dominated loading this is unlikely, but C8's ladder runs the
  same case with different step sequences as a sanity check.
- Interaction with the stress-controlled components: `TbarP` scales with
  accumulated `inc_tol`; sub-stepping must scale it consistently (it
  already does, being computed from `inc_tol` - keep it that way and test).

**Validate.** Contrast-10 case with forced tiny steps (`inc_min` sweep):
identical boundary outputs. Contrast-300 case: completes; compare P11 at
boundaries between a run with initial step 0.1 and one with 0.05 - must
agree within solver tolerance.

*Implemented (rollback via non-mutating Newton solves on trial states;
growth 1.5x after 2 easy sub-increments capped at the original increment
size and boundary; failure below inc/16).*

### C5. Inexact Newton (adaptive inner tolerance)

**What.** Replace the fixed `rtol = 1e-6` by an Eisenstat-Walker-style
forcing term: `eta_k = min(eta_max, c * (norm(b_k)/norm(b_{k-1}))^2)` with
`eta_max = 1e-2`, plus the safeguard from EW choice 2, floored at
`eta_min = 1e-3` normally, and tightened (`1e-3 * eta`) on the final expected
iteration (when `norm(b_k)` is within ~1 order of the stopping tolerance) so
the last solve doesn't limit the achievable outer residual. `atol` stays at
a small floor (1e-10) to avoid infinite work on near-zero right-hand sides.

**Why.** D7. At high contrast each GMRES iteration is expensive (many are
needed); not over-solving early Newton steps is the cheapest big win after
C1. This is precisely the configuration Kabel et al. (2014) found effective
for finite-strain FFT homogenization.

**Expected.** 2-5x reduction in total Krylov iterations per increment at
high contrast; negligible change in Newton iteration counts (theory: EW
forcing preserves superlinear convergence).

**Correctness argument.** Inexact Newton theory (Dembo-Eisenstat-Steihaug;
Eisenstat-Walker) guarantees the outer iteration still converges to the same
solution provided `eta_k < 1`, and the *final* accuracy is governed by the
outer stopping rule (C2), which is unchanged. The linear solutions being
less accurate mid-iteration does not bias the converged state.

**Risks / critical view.**
- With a marginal preconditioner, a loose early solve can produce a poor
  direction and trigger the line search more often. The stats from C0 will
  show whether total work actually decreased; if not, fall back to a fixed
  modest `rtol = 1e-4` (still better than 1e-6).
- GMRES `rtol` in SciPy is relative to `norm(b)` (preconditioned norm for
  left preconditioning) - the diagnostics (`print_linear_diagnostics`)
  already distinguish true vs preconditioned residuals; keep using them
  during calibration.

**Validate.** Contrast-10 and contrast-100 single cases: same converged
boundary outputs (within outer tolerance), total Krylov iterations reduced;
Newton iterations not increased by more than ~20 %.

*Implemented. Eisenstat-Walker choice 2 (gamma = 0.9, alpha = 2) on the
block-normalized merit function, clamped to [1e-3, 1e-2], reset per
sub-increment; `forcing="fixed"` restores the old fixed `rtol`. Measured at
contrast 100, N = 31: 1024 -> 447 Krylov iterations (2.29x), 104 s -> 42 s,
Newton count unchanged at 12, `forcing="fixed"` bitwise-identical to the
pre-change solver. The final-iteration tightening sketched above was
deliberately not implemented - the measured residual histories reach ~1e-12
in five Newton steps, so eta_min = 1e-3 is never the binding constraint.
Full theory and measurements: `docs/inexact_newton_and_reference_tangent.md`.*

### C6. Configurable reference tangent for the Green preconditioner

**What.** Add a `reference` option to the solver/config:
`"mean"` (current arithmetic voxel average), `"matrix"` (mean over matrix
voxels only), `"mid"` (average of the two phase means). Same for the
`kappa_inv` and `JFmT` references. Batch config exposes it; C0 logs which
one is used. Run a 3x3 matrix (three references x contrast {10, 100, 1000})
on one structure and pick the production default from measured GMRES counts.

**Why.** D5. The arithmetic mean is provably a poor reference for the
majority phase at high contrast. Literature does not give a definitive
answer for the *mixed* (F, p) saddle system - the repository's own
`green_reference_preconditioning.md` labels the preconditioner experimental -
so this must be settled empirically, cheaply, with a switch.

**Expected.** At contrast >= 100, `"matrix"` or `"mid"` should cut GMRES
iterations substantially (plausibly 2-10x based on the small-strain Green
preconditioner literature); at contrast 10 differences should be minor.

**Correctness argument.** A preconditioner changes the iteration, not the
solution: any nonsingular (on the relevant subspace) M yields the same
converged `dX` up to the linear tolerance. The zero-frequency handling
(stress-controlled free components) is reference-independent and stays
as is. The pseudo-inverse construction already guards rank deficiency.

**Risks / critical view.**
- A matrix-only reference makes the preconditioned filler blocks *worse*
  conditioned locally; if filler volume fraction were high this could
  backfire - which is why this is measured, not assumed. At phi <= 0.19 the
  risk is low.
- If none of the three references tames contrast 10^3, the fallback is the
  Green-Jacobi form `D^{-1/2} G0^+ D^{-1/2}` (Ladecky et al.) - deferred
  (Section 7) because it needs new per-voxel scaling plumbing and its
  saddle-point behavior is unproven; we should not build it before knowing
  the simple reference swap is insufficient.

**Validate.** GMRES-count table (reference x contrast) from C0 stats on one
structure; identical converged outputs across references (within linear
tolerance) as a preconditioner-correctness check.

*Implemented as `reference` = `"mean"` | `"matrix"` | `"mid"`
(`reference_average()` in `FFT_simulation/fg/preconditioning.py`), applied to K_0, H_0 and
alpha_0 in the mixed solver and to K_0 in the standard solver. Default stays
`"mean"`.*

***The expectation in D5 was not confirmed at contrast 100.*** *At
phi = 0.089, the matrix-phase reference needed ~10 % **more** Krylov
iterations than the volume average (1130 vs 1024 at fixed inner tolerance);
`"mid"` was equivalent to `"mean"`. Whether D5's argument bites at contrast
>= 1e3 is still open.*

***The preconditioner-correctness check above FAILED, and the cause is a
pre-existing defect, not C6.*** *The three references converge to different
states (P11 spread ~2.8e-3 relative) even at `tol_rel = 1e-9` with residuals
at ~1e-12. The symbol is built as `G K_0`, whose pseudo-inverse has range
`K_0^T range(G)`; that equals `range(G)` only for isotropic K_0, so the
preconditioner carries GMRES iterates out of the compatible subspace and the
incompatible content is invisible to the G-projected F-residual. Building the
symbol as `G K_0 G` fixes it (measured: incompatible content 8.2e-1 ->
4.9e-13) but changes every historical `preconditioner="reference"` result, so
it is left for the project owner to decide. Details in section 5 of
`docs/inexact_newton_and_reference_tangent.md`.*

### C7. Make the default linear solver honest

**What.** Route the mixed system to GMRES whenever `preconditioner=None`
as well (keep CG available behind an explicit `linear_solver="cg"` escape
hatch for reproducing old runs), and say so in the docs.

**Why.** D8. CG on a nonsymmetric indefinite system has no guarantee; it is
a latent foot-gun for anyone running defaults, and its failure mode
(silently wrong search directions) is exactly the kind we are eliminating.

**Expected.** No change for the batch workflow (already GMRES). Slightly
higher memory for default users (GMRES restart basis).

**Risks.** Behavior change for anyone relying on the old default; mitigated
by the escape hatch and a changelog note in the docs.

**Validate.** One case run with old-CG vs new-GMRES defaults at contrast 10:
same converged outputs.

### C8. Contrast ladder study (physics; decides production settings)

**What.** With C0-C6 in place, on 3 structures (low/mid/high phr):

1. Run contrast in {10, 30, 100, 300, 1000} (E_filler in {100 ... 10000},
   matrix fixed), new charge files alongside the existing ones.
2. Plot reinforcement ratio P11/P11_matrix at F11 = 1.3, 1.6, 2.0 vs
   contrast; identify the saturation knee (expected ~10^2-10^3 from
   Hashin-Shtrikman-type arguments).
3. Pick the production contrast: smallest value within ~3 % of the
   contrast-1000 response.
4. Resolution spot check: rerun one high-phr structure at the chosen
   contrast with N = 63 (structures are generated at higher resolution and
   re-voxelized; if only N = 31 voxelizations exist, regenerate one - the
   generation pipeline supports it). Report the N-sensitivity of P11.
5. Re-run the same case with two different initial step sizes as the
   path-independence sanity check from C4.

**Why.** This converts "we need high contrast" into a measured, minimal
requirement, which is the main runtime lever of all: contrast we do not
simulate costs nothing.

**Expected outcome / what improvement looks like.** The final PHR sweep at
the chosen contrast shows visibly super-linear P11 vs phr (approaching the
Guth-Gold-like curvature documented in the earlier analysis), with per-case
runtimes of minutes at N = 31, and the N = 63 check bounding the
discretization error (if it exceeds ~5-10 %, production moves to N = 63 and
the runtime budget is re-evaluated - feasible only because of C1).

**Risks / critical view.**
- Even at saturated contrast, a two-phase hyperelastic RVE reproduces only
  hydrodynamic reinforcement. Filler-network/bound-rubber effects (Payne
  effect etc.) are physically absent; expectations for matching experimental
  data should be set accordingly. This is a modeling limit, not a solver
  defect, and no solver change fixes it.
- 1-2-voxel ligaments at N = 31 with contrast 10^3 will show the largest
  discrepancy vs N = 63; if the spot check is bad, the honest answer is
  higher N, not creative averaging.

## 5. What we expect to see improve (acceptance criteria)

| Metric (from C0 stats) | Now | Target after plan |
|---|---|---|
| Wall time / case, contrast 10, N = 31 | ~200 s | <= 30 s |
| Contrast 100 case | often fails / very slow | converges, <= ~3 min |
| Contrast 1000 case | fails | converges (with step cuts), <= ~15 min |
| Silent bad results | possible (D1) | impossible: status flag per case |
| Krylov iters vs contrast | ~linear growth | clearly sublinear with best reference |
| Baseline regression (contrast 10 sweep) | - | P11 curves match current results <= 1e-4 rel |
| Physics outcome | linear P11 vs phr | superlinear P11 vs phr at production contrast |

## 6. Implementation order and effort

1. **C0 + C1** (instrumentation, vectorization) - mechanical, testable,
   everything else depends on the speed. Includes the regression harness.
2. **C2 + C3 + C4** (residual stopping, fail-loud, line search, adaptive
   steps) - one coherent change to the Newton/increment loop; land together
   with the synthetic-failure tests.
3. **C5 + C6 + C7** (inexact Newton, reference options, honest defaults) -
   independent small changes, each calibrated with the C0 stats.
4. **C8** (contrast ladder + resolution check) - runs, not code; produces
   the production configuration and an updated results note.

## 7. Considered and deferred / rejected

- **Green-Jacobi preconditioner** (Ladecky et al.): deferred until C6 shows
  the plain reference swap insufficient at the production contrast. Reason:
  extra complexity, unproven on the mixed saddle system, and C6 may make it
  unnecessary.
- **Displacement-based reformulation (DBFFT, Lucarini & Segurado 2019):**
  would give a symmetric positive system and better-understood
  preconditioning, but is a rewrite of the solver core, abandoning the
  paper's mixed formulation this repository exists to use. Out of scope.
- **Composite voxels (Kabel et al. 2015):** best-in-class fix for interface
  resolution at high contrast, but touches structure generation and the
  constitutive layer; the N = 63 fallback is simpler and the vectorized
  solver makes it affordable. Revisit only if C8's resolution check fails
  badly at N = 63 too.
- **Numba/C acceleration:** unnecessary once NumPy-vectorized; adds a
  dependency and build friction for ~2x at best on top of C1.
- **True rigid fillers (constraint formulation):** not needed given
  contrast saturation (C8); would require a different formulation entirely.
- **Anderson acceleration / quasi-Newton on the fixed point:** the
  literature (Schneider review, Chen et al.) shows benefits mainly for
  polarization/basic schemes; we already have a Newton-Krylov structure,
  which dominates when tangents are available (they are).

## 8. References

- Moulinec, Suquet (1998). *A numerical method for computing the overall
  response of nonlinear composites with complex microstructure.* CMAME.
- Zeman, Vondrejc, Novak, Marek (2010). *Accelerating a FFT-based solver ...
  by conjugate gradients.* JCP.
- Vondrejc, Zeman, Marek (2014). *An FFT-based Galerkin method ...* CMAME.
- Gelebart, Mondon-Cancel (2013). *Non-linear extension of FFT-based methods
  accelerated by conjugate gradients ...* Comput. Mater. Sci.
- Kabel, Bohlke, Schneider (2014). *Efficient fixed point and Newton-Krylov
  solvers for FFT-based homogenization of elasticity at large deformations.*
  Comput. Mech.
- Kabel, Merkert, Schneider (2015). *Use of composite voxels in FFT-based
  homogenization.* CMAME.
- Eisenstat, Walker (1996). *Choosing the forcing terms in an inexact Newton
  method.* SIAM J. Sci. Comput.
- Dembo, Eisenstat, Steihaug (1982). *Inexact Newton methods.* SINUM.
- Schneider (2021). *A review of nonlinear FFT-based computational
  homogenization methods.* Acta Mechanica.
- Lucarini, Segurado (2019). *DBFFT: A displacement based FFT approach ...*
- Ladecky et al. *Jacobi-accelerated FFT-based solver for smooth
  high-contrast data* (see `docs/green_reference_preconditioning.md`).
- Nocedal, Wright. *Numerical Optimization*, ch. 11 (globalized Newton).
- Deuflhard. *Newton Methods for Nonlinear Problems* (affine-invariant
  globalization, step-size control).
