# Solver Work: What Changed, Why, and What the Evidence Says

A single record of the solver investigation on branches
`claude/fft-homogenization-contrast-62pk68` and
`claude/dbfft-displacement-rework`. Every number quoted here was measured; where
a measurement later turned out to be unreliable, that is stated rather than
quietly dropped (Section 10).

Detail lives in the topic documents: `green_reference_preconditioning.md`, `discretization.md`, `benchmarking.md`,
`inexact_newton_and_reference_tangent.md`, `high_contrast_convergence_plan.md`,
`solver_improvements.md`, and — on the DBFFT branch — `dbfft_rework.md`.

Layout note: the solver lives under `FFT_simulation/fg/`; the test scripts sit
at the repository root.

---

## 1. The short version

The work started as "make the solver faster at high filler/matrix contrast".
It found something more important on the way.

| # | finding | status |
|---|---|---|
| 1 | **The Green preconditioner was producing wrong answers** — every result made with `preconditioner="reference"` is off by +8 to +23 %, growing with contrast | **fixed and validated** |
| 2 | Inexact Newton (C5) is a real 2.1–2.6× speed-up, and in the corrected solver it is the difference between converging and not | implemented, default on |
| 3 | The reference tangent (C6) matters after the fix; `matrix` is the best mode | implemented, default unchanged (`mean`) |
| 4 | Willot's discretization does not reduce runtime — it is an accuracy intervention, not a convergence one | implemented, default off |
| 5 | Green-Jacobi does not transfer to this formulation | implemented, measured, **negative result** |
| 6 | The corrected solver sits at the Green preconditioner's published **O(contrast)** ceiling | measured, λ_max ≈ 572 at contrast 500 |
| 7 | The displacement-based reformulation (DBFFT) removes the structural defect | structurally validated, **not working yet** |

**The one thing that matters for the paper**: results produced with the old
preconditioner are wrong, and the error grows with contrast. Everything else is
performance engineering.

---

## 2. The preconditioner defect (the important one)

### 2.1 What was wrong

The mixed solver's unknown is the deformation gradient `F`, which must satisfy a
*compatibility* constraint — the fluctuation has to be a gradient field. The
operator is built so that its deformation row always emits `G(...)`, i.e. it
only ever produces compatible output.

Formed densely at N=7 (contrast 100, φ = 7.9 %) and decomposed exactly:

| quantity | value |
|---|---|
| operator size | 3430 (F-block 3087 + p-block 343) |
| **rank** | **1371** |
| **nullity** | **2059** |
| dim of the compatible subspace | 1371 |
| rank of the operator restricted to that subspace | **1371 of 1371 — nonsingular** |
| smallest singular value of the restriction | 7.03e-3 (largest 4.97e2) |

Three consequences, all exact rather than statistical:

1. The operator is **massively singular** on the full space.
2. Restricted to compatible fields it is **exactly nonsingular** — so the
   physical problem is well-posed with a unique solution. The formulation is
   sound.
3. Since the restriction is injective, `null(A) ∩ compatible = {0}`: the null
   space is **entirely non-physical**.

Correctness therefore depends on the iterates never leaving the compatible
subspace. Unpreconditioned Krylov does this automatically. **The preconditioner
broke exactly that property.** Feeding compatible basis vectors through `M⁻¹`:

```
||M⁻¹v − Π M⁻¹v|| / ||M⁻¹v||    min 1.96e-01   median 5.94e-01   max 6.85e-01
```

The cause is algebraic: the symbol was `Ĝ K₀`, whose pseudo-inverse has range
`K₀ᵀ range(Ĝ)`, which equals `range(Ĝ)` only for isotropic `K₀`. A finite-strain
tangent never is. So the converged step was *(unique physical solution) +
(arbitrary null-space component)*, with the preconditioner choosing which.

### 2.2 The fix

The symbol is now the reference operator **restricted to the compatible
subspace**, `Π A₀ Π` with `Π = diag(Ĝ, I)`:

```
[ Ĝ K₀ Ĝ    Ĝ H₀  ]
[ H₀ᵀ Ĝ     −α₀   ]
```

Both the deformation block *and* the pressure row are projected — the pressure
row was an easy thing to miss. `precond_restrict=False` restores the old symbol
for reproducing pre-fix numbers.

### 2.3 Four independent validations

**(a) The leakage is gone.**

| | leakage |
|---|---|
| before | median 5.94e-01 |
| after | median **1.84e-15** |

**(b) The reference tangent no longer changes the answer.** This is the
defining property of a preconditioner — it must change the iteration, never the
result. Contrast 100, N=31, `tol_rel=1e-9`:

| reference | before | after |
|---|---|---|
| mean | 1.0835510581 | **1.2726045390** |
| matrix | 1.0804848613 | **1.2726045390** |
| mid | 1.0837578465 | **1.2726045390** |
| spread | 3.02e-03 | **3.98e-12** |

and the converged fluctuation field is compatible again (6.4e-01 → ~2.5e-13).

**(c) An independent control says which answer is right.** Unpreconditioned
GMRES provably cannot leave the compatible subspace, so it is unbiased:

| run | P11 | incompatible | Krylov | wall |
|---|---|---|---|---|
| unpreconditioned (control) | 1.2726045390 | 1.5e-11 | 6314 | 1256 s |
| preconditioned, **fixed** | 1.2726045376 | 2.7e-13 | **217** | **23 s** |
| preconditioned, pre-fix | 1.0835510573 | 6.4e-01 | 130 | 15 s |

The fix agrees with the control to **9 significant figures** while being a 29×
iteration / 55× wall-time accelerator over no preconditioning.

**(d) The sharpest test: on a homogeneous body the restricted symbol *is* the
operator inverse.** If `K(x) = K₀` everywhere then `Ĝ K₀ Ĝ` is the operator
restricted to the compatible subspace, so `M⁻¹A` must act as the identity.
Measured on a truly homogeneous cell (identical E *and* Poisson ratio in both
phases):

| body | symbol | ‖M⁻¹Ax − x‖/‖x‖ |
|---|---|---|
| homogeneous | unrestricted (pre-fix) | 7.06e-01 |
| homogeneous | **restricted (fixed)** | **6.71e-15** |
| heterogeneous, contrast 100 | unrestricted | 1.83 |
| heterogeneous, contrast 100 | restricted | 3.83e+01 |

Only the restricted symbol has the defining property. This test does not depend
on any claim about which answer is correct, which is what makes it decisive.
Asserted in `test_preconditioner.py`.

### 2.4 Impact on existing results

| contrast | legacy P11 | corrected P11 | shift |
|---|---|---|---|
| 10 | 2.659503 | 2.887333 | **+8.6 %** |
| 100 | 2.751416 | 3.278909 | **+19.2 %** |
| 500 | 2.788222 | 3.425616 | **+22.9 %** |

*(N=31, structure `1_voxel`, 3 increments.)*

**Every run made with `preconditioner="reference"` is affected, including the
production sweeps in `Results/`.** The error is not tolerance-level and it grows
with contrast — worst exactly where the physics of interest lives. Affected
sweeps need regenerating; the magnitude depends on contrast, filler fraction and
strain, so re-measure per sweep rather than assuming a single factor.

The cost of correctness is that the restricted symbol is a weaker accelerator
(Section 6).

---

## 3. C5 — inexact Newton (Eisenstat–Walker)

**What.** The inner GMRES tolerance was a fixed `1e-6` on every Newton step.
It is now an Eisenstat–Walker "choice 2" forcing term,
`η_k = γ(‖r_k‖/‖r_{k-1}‖)^α` with the standard safeguard, clamped to
`[1e-3, 1e-2]`. Solving early Newton steps to 1e-6 buys nothing but iterations.

**Measured** (legacy family, N=31, 3 increments, structure `1_voxel`):

| contrast | baseline | C5 | speed-up |
|---|---|---|---|
| 10 | 246 | 115 | **2.14×** |
| 100 | 1024 | 447 | **2.29×** |
| 500 | 4554 | 1736 | **2.62×** |
| 1000 | 6578 | 2904 | **2.27×** |
| 2500 | 22709 | 9148 | **2.48×** |

Newton counts were essentially unchanged, which is the point — the forcing
terms preserve the Newton path.

**In the corrected solver C5 is not an optimisation, it is a requirement.** At
contrast 100 with three increments, `FIX` alone (fixed 1e-6 inner tolerance)
hits the iteration cap and the case *fails*; `FIX+C5` converges in 1388 Krylov
iterations. The restricted symbol is a weaker accelerator, so demanding 1e-6
stagnates; adaptive forcing never asks for accuracy the preconditioner cannot
cheaply deliver.

**Regression**: `forcing="fixed", inner_rtol=1e-6, precond_restrict=False`
reproduces the pre-change solver **bitwise** (`test_c5c6.py`).

---

## 4. C6 — configurable reference tangent

**What.** The reference tangent `K₀` was hard-coded to the arithmetic voxel
average. It is now selectable: `mean` (unchanged default), `matrix` (average
over matrix voxels only), `mid` (unweighted average of the two phase means, the
finite-strain analogue of the classical `(C_min+C_max)/2`).

**Why it should matter.** At contrast 100 with φ = 8.9 %, `mean` gives
`K₀ ≈ 10 K_matrix` — the average is dominated by the stiff phase, so *every*
voxel is badly preconditioned. `matrix` preconditions the 91 % of the cell that
is matrix near-exactly and leaves the filler as a low-volume-fraction
perturbation. This is argument **D5** from the original plan document.

**Measured, clean**: contrast 100, corrected family, 3 increments —
`FIX+C5` 1388 Krylov vs `FIX+C5+C6-matrix` 1145, i.e. **1.21×**, with Newton
15 → 13. Neither run was truncated.

**Measured, less reliable**: at contrast 500 the same pair gave 34818 vs 15431
(2.26×) with Newton 105 → 47 and step cuts 6 → 2 — but *both* runs hit the
iteration cap, so the ratio is not trustworthy. The Newton and step-cut counts
are less affected by truncation and do suggest a real benefit at high contrast;
it needs re-measuring with the raised cap.

**Verdict history is worth recording**: an early reading said C6 gave nothing
and should be deleted. That reading came from measurements taken under the
*buggy* preconditioner, where changing the reference changed which wrong answer
you converged to — so comparing iteration counts across references was
comparing convergence to different targets. Keeping it was the right call.

Default remains `mean` because `reference_average(..., "mean")` is **bitwise
identical** to the old hard-coded expression, so C6 is inert unless selected.

---

## 5. Willot's rotated discretization

**What the code had.** `build_Ghat4` used raw wave numbers in
`δ_il q_j q_m/|q|²` — the exact trigonometric derivative, i.e. the
Moulinec–Suquet/FFT-Galerkin spectral discretization. Most accurate for smooth
fields, worst possible for a two-phase composite: a truncated Fourier basis
cannot represent the strain jump at an interface, so the discrete solution
carries Gibbs oscillations whose amplitude grows with contrast.

**What was added.** Willot (2015)'s rotated scheme — a forward finite difference
along the voxel diagonal, averaged over the transverse directions — selectable
via `discretization: fourier | willot`.

**Verification** (`test_discretization.py`) checked the *defining properties*
rather than trusting the formula:

| check | result |
|---|---|
| `fourier` reproduces the previous `build_Ghat4` | **bitwise** |
| `fourier` reproduces the old `FFT_simulation/fg/fft.py` inline loop | **bitwise** |
| real-space stencil of ξ̃₁ | **8 voxels on a 2×2×2 cube, values ±1, sign splitting along one axis** |
| Ĝ Hermitian / idempotent | 0.0 / 3.3e-16 |
| Ĝ fixes Willot-gradient fields | 4.8e-16 |
| vs a *Fourier*-gradient field | 2.7e-01 — the schemes genuinely differ |

The stencil test is the real proof: a finite difference *must* have local
support, and it came out as exactly the diagonal forward difference.

**Result: no runtime benefit.** Total Krylov iterations (N=31, 3 increments):

| contrast | spectral | Willot | change |
|---|---|---|---|
| 10 | 115 | 115 | 0.0 % |
| 100 | 447 | 457 | +2.2 % |
| 500 | 1736 | 1727 | −0.5 % |
| 1000 | 2904 | 3120 | +7.4 % |

Iteration-neutral, and each iteration is *more* expensive because `Ghat4`
becomes complex (162 → 324 MB at N=63). This is expected: Willot's scheme is an
**accuracy** intervention, not a convergence one. The convergence benefit
reported in the literature is largely for the *basic fixed-point* iteration,
whose contraction rate depends directly on the operator spectrum; with
Newton–Krylov plus a Green preconditioner the conditioning is already the
preconditioner's job.

Its accuracy case remains undemonstrated here — that needs a resolution study
(N=31 vs N=63), which was out of budget. Two projectors agree to ~1e-16 at low
frequency and differ only at high frequency, so the homogenised stress barely
moves (+1.6 % at contrast 10, +2.3 % at contrast 100 with compatible fields on
both sides) while local fields near interfaces would move much more.

---

## 6. Why the corrected solver is slower, and what was ruled out

The corrected preconditioner is materially more expensive at high contrast. Six
configuration suspects were eliminated **by measurement**, at contrast 500,
N=31, `reference="matrix"`:

| suspect | measurement | verdict |
|---|---|---|
| Krylov iteration cap | see Section 8 | real, but a *benchmark* artifact |
| pseudo-inverse / `rcond` | clean rank 4 per frequency, ~1e14 gap to discarded modes; retained-block condition 17 (restricted) vs 3 (unrestricted) | healthy |
| GMRES restart length | 100 → 2986, **400 → 1960**, 1000 → 3685 Krylov | worth setting 400; ~1.5×, not the gap |
| inner tolerance too tight | tightening to `[1e-5,1e-3]` made the first solve run **7830 iterations without reaching 6.1e-3** | the loose default is a *mitigation*, not the cause |
| outer tolerance too tight | `tol_rel` 1e-3 vs 1e-5: 1629 vs 1960 Krylov, P11 differs 1.6e-4 | only 1.2× |
| the Krylov algorithm | see below | not a solver-choice problem |

**The Krylov race.** Five structurally unrelated methods on one real captured
system at contrast 500, ~1500 matvec budget each:

| algorithm | matvecs | true residual |
|---|---|---|
| gmres(100) | 1515 | 6.82e-03 |
| gmres(300) | 1505 | 6.30e-03 |
| lgmres | 1209 | 7.09e-03 |
| gcrotmk | 1110 | 6.91e-03 |
| bicgstab | 1500 | 1.03e-02 |

All plateau in the same band — and `gmres(100)` at **6060** matvecs also gave
6.82e-03, identical to its 1515-matvec value. That is a *floor*, not slow
convergence. When changing the algorithm changes nothing, the operator is the
problem.

**Residual anatomy.** Dissecting the leftover residual at the plateau:

```
true relative residual              6.818e-03
  carried by the F-block            6.818e-03   (p-block only 9.6e-05)
  incompatible fraction of r_F      4.2e-14     -> fully reachable by A
  zero-frequency share              0.001%      -> not a macroscopic-mode bug
  ||M⁻¹r||/||M⁻¹b|| vs ||r||/||b||  ratio 1.39  -> M⁻¹ is NOT blind to it
```

Compatible, reachable, spread across frequencies, plainly visible to the
preconditioner — and GMRES still cannot reduce it. **No structural defect
remains to find.**

**What it actually is.** The eigenvalues confirm the textbook result:

| quantity | measured |
|---|---|
| λ_max(M⁻¹A) at contrast 500 | **≈ 572** |

Green-preconditioner theory bounds the preconditioned spectrum by the *local
ratios* `K(x)/K₀`, so a single homogeneous reference gives a condition number of
order the phase contrast. 572 for χ = 500 reproduces that to within 15 %. The
method is behaving exactly as published; it has an O(contrast) ceiling. The
legacy preconditioner appeared immune only because it was preconditioning a
different, easier operator — and converging to the wrong answer.

A related check: the reference tangent `K₀` is itself **indefinite** (2 of 9
eigenvalues negative), which is normal for a finite-strain ∂P/∂F — but the
restricted symbol `Ĝ K₀ Ĝ` is **sign-definite** across all 806 sampled
frequencies, so the projection rescues definiteness. Not a problem.

---

## 7. Green-Jacobi — implemented, and a negative result

Ladecký et al. (arXiv:2508.02613) introduce Green-Jacobi,
`M⁻¹ = D^{-1/2} G₀⁻¹ D^{-1/2}`, precisely because the plain Green preconditioner
degrades on high-contrast data. Implemented as
`preconditioner="green_jacobi"`, with `d(x)` the Frobenius norm of the local
tangent (the operator is matrix-free, so there is no assembled diagonal) and the
reference built from the scaled tangent `K(x)/d(x)`.

**Correctly built** (`test_green_jacobi.py`):

| check | result |
|---|---|
| collapses onto Green on a homogeneous body (d uniform ⇒ D^{-1/2} is a scalar) | 1.57e-12 |
| output stays compatible, homogeneous | 4.7e-16 |
| output stays compatible, contrast 500 | 4.7e-16 |

**It does not help.** At contrast 500 its *first solve alone* exceeded 1950
iterations without converging, against **1960 Krylov for the entire Green run**
(9 Newton steps).

There is a structural reason, and it is the useful part of the result:

1. Ladecký et al. solve a **displacement/FE** system in which every field is
   admissible. `D` is the genuine diagonal of an assembled stiffness matrix — a
   *local* quantity.
2. Our unknown is the deformation gradient, whose operator is `Ĝ·K` with **Ĝ a
   global projector**. The diagonal of that operator is not `K(x)`, so no purely
   local `d(x)` is its Jacobi diagonal.
3. `D^{-1/2}` is a pointwise scaling that **does not preserve compatibility**, so
   the result must be re-projected — costing an extra FFT round trip and
   partially undoing the scaling just applied.

**The compatibility constraint is the root cause of both problems**: it makes
the operator singular *and* blocks the published high-contrast preconditioner.
That is what motivated Section 9.

---

## 8. Benchmark infrastructure

`FFT_simulation/benchmark_suite.py` measures every change against the pre-change solver across
a contrast ladder, on a many-core machine: one single-threaded process per
(structure, contrast, config), results streamed to `results.jsonl` for crash-safe
resume, per-run logs so 100 parallel runs do not interleave.

Two bugs were found **in the benchmark itself**, both instructive:

**(a) The baseline was not a baseline.** The pinned pre-change `FFT_simulation/fg/mxfft.py`
imports its symbol builder from the *live* `FFT_simulation/fg/preconditioning.py` — so after
the preconditioner fix, "baseline" was silently running the **corrected**
preconditioner. Fixed by pinning `preconditioning.py` too and rewriting the
pinned solver's imports. Verified: the pinned baseline now reproduces the
historical per-solve counts exactly (`[62, 81, 84, 84, 70, 88, 91, 94, 77, 94,
98, 101]`, 1024 total, 0 cuts). The `baseline` vs `baseline-new` pair is kept in
the config matrix as a permanent self-check for this class of mistake.

**(b) The iteration cap silently truncated every high-contrast run.**
`--max-gmres-iter` defaulted to 1000 — ample for the legacy preconditioner (only
**1 of 35** legacy runs ever reached it) but far too tight for the corrected one.
A capped solve is treated as *failed*, which cuts the load step, which exhausts
the sub-step budget, which reports the case as `failed`. This looked like a
robustness collapse of the fix and was purely the cap. Every corrected run at
contrast ≥ 500 in the first full sweep was affected.

Fixed three ways: the default is now 20000 (a safety valve, not a tuning knob);
each run records `cap_hit`, marked `**!**` in `summary.md` with the speed-up
suppressed to `n/a` when either side is truncated; and `--timeout` stops one
pathological point holding up a sweep.

Also: `--workers` defaulted to `min(8, cpu_count())`, silently capping a
100-core server at 8. Now `len(os.sched_getaffinity(0))`, which respects
cpuset limits that `os.cpu_count()` ignores inside a container.

**Memory, not cores, is the binding constraint** at N=63. The GMRES restart
basis is `restart × 10 N³ × 8 B` — 8.0 GB per worker at restart 400. A
100-worker sweep then needs ~865 GB. The script estimates this against
`MemAvailable` and warns.

---

## 9. DBFFT — the displacement-based rework (branch: `claude/dbfft-displacement-rework`)

**Status: structurally validated, not converging. Do not use for production.**

Writing `F = F̄ + grad u` makes compatibility automatic, which removes the root
cause identified in Sections 2 and 7 at the source.

**Verified** (`test_dbfft.py`, plus a finite-difference Jacobian check):

| check | result |
|---|---|
| `div_adj` is the exact adjoint of `grad` | 1.9e-16 (spectral), 6.4e-16 (Willot) |
| **operator rank at N=5** | **499 of 502 — nullity 3** |
| nullity is exactly the rigid translations | yes |
| reference acoustic tensor Hermitian | 0.0 |
| singular only at ξ = 0 | 1 of 729 frequencies |
| **condition number of Γ** | **3.00 median and max** |
| Jacobian vs finite differences | u 1.1e-09, p 1.9e-10, F̄ 1.0e-09 |

Direct comparison:

| | F-based | DBFFT |
|---|---|---|
| nullity | **2059** of 3430 | **3** of 502 |
| reference symbol | rank 4 of 10, needs pinv, condition 17 | **3×3 Hermitian, condition 3.00** |

One real bug was found and fixed during implementation: `div_adj` carries a
factor `|ξ| ~ πN`, so the raw equilibrium residual is ~1000× the
incompressibility residual at N=31. Unscaled, GMRES satisfies equilibrium,
ignores incompressibility, and **line search fails at every step size down to
0.003**. Nondimensionalising (equations ÷ σ, pressure carried as σp̃) fixed the
direction.

**What does not work.** At contrast 100, N=31, one increment: the F-based solver
finishes in 43 s (305 Krylov, 4 Newton); DBFFT did not complete a single Newton
step in 10 minutes. The Jacobian is confirmed correct, so the remaining problem
is the linear solve / preconditioner effectiveness. Leads are recorded in
`dbfft_rework.md`.

**Important expectation management.** DBFFT's operator is
`div_adj ∘ K(x) ∘ grad`, preconditioned by `div_adj ∘ K₀ ∘ grad`. The
generalised eigenvalue problem gives a spectrum bounded by **the same local
ratios `K(x)/K₀`** as the F-based formulation — so **DBFFT alone hits the same
O(contrast) ceiling.** The condition-3.00 acoustic tensor is a property of the
*reference* operator, not a measure of how well it preconditions a heterogeneous
body. What DBFFT buys is (i) correctness without the restriction, (ii) full rank
and ~4× fewer unknowns, and (iii) **making Green-Jacobi applicable as
published** — and only (iii) attacks the ceiling.

**Acceptance criterion**, since both formulations discretise the same problem:
DBFFT must reproduce the corrected F-based P11 — 1.272604498 for one increment
at contrast 100; 2.887333 / 3.278909 / 3.425616 at contrast 10 / 100 / 500 for
three increments. A mismatch means the reformulation is wrong, not merely slow.

---

## 10. Measurements that turned out to be unreliable

Recorded so they are not cited later:

| claim | why it is wrong |
|---|---|
| "null(A) contains compatible directions too" | the test measured overlap of individual SVD basis vectors with the subspace, which is not the same as membership. Superseded by the injectivity of the restriction, which proves `null(A) ∩ compatible = {0}` |
| "restarted-GMRES stagnation explains the slowness" | `lgmres` and `gcrotmk`, both designed to repair exactly that, plateau at the same residual |
| "C5's forcing window is miscalibrated for the corrected preconditioner" | tightening it made things much worse — 7830 iterations without reaching 6.1e-3 |
| "32× penalty at contrast 500" | computed from cap-truncated runs. The clean single-increment measurement is 1960 Krylov |
| "C5 is worth 18.8× / 22.65× in the corrected family" | computed against a `FIX` baseline that hit the cap and failed. The defensible statement is qualitative: `FIX` fails, `FIX+C5` converges |
| "C6-matrix is worth 2.26× at contrast 500" | both sides were cap-truncated. The clean number is 1.21× at contrast 100 |
| "Green-Jacobi's approximate-inverse quality is worse (103 vs 38)" | measured at different contrasts (500 vs 100); not comparable. The defensible comparison is the first-solve iteration count at matched contrast |

---

## 11. Test inventory

| file | covers |
|---|---|
| `test_c5c6.py` | `reference_average` against explicit per-phase computation; Eisenstat–Walker forcing against its definition; **bitwise** reproduction of the pre-change solver; reference-mode independence after the fix (< 1e-7) |
| `test_preconditioner.py` | the homogeneous-body identity test — the decisive correctness proof for the fix |
| `test_discretization.py` | bitwise reproduction of both previous `Ghat4` implementations; the real-space stencil; Hermitian/idempotent/reality/projection properties |
| `test_green_jacobi.py` | collapse onto Green on a homogeneous body; compatibility of the output; head-to-head iteration counts |
| `test_dbfft.py` | grad/div adjointness; operator rank; acoustic-tensor conditioning; agreement with the corrected F-based solver |
| `FFT_simulation/benchmark_suite.py` | the full contrast ladder across all configurations, with truncation flagging |

---

## 12. Recommendations

1. **Regenerate any results produced with `preconditioner="reference"`** before
   the fix. This is the finding with real consequences.
2. **Production configuration**: `precond_restrict: true`, `forcing:
   eisenstat_walker` (η ∈ [1e-3, 1e-2]), `reference: matrix`,
   `discretization: fourier`, `gmres_restart` 300–400. Contrast 100 at N=63 is
   comfortable.
3. **Watch `cap_hit`** in any benchmark output. A truncated run's cost is a
   lower bound, not a measurement.
4. **Run C8 from the plan document before more solver work.** It argues
   Hashin–Shtrikman-type estimates saturate at contrast 10²–10³ and prescribes
   running the cheapest sufficient contrast. If saturation lands at 100–500, the
   expensive regime is not needed and the remaining solver work is moot. That is
   a handful of cheap runs against an open-ended engineering effort.
5. Only if C8 shows contrast ≥ 10³ is genuinely required is the DBFFT →
   Green-Jacobi chain worth finishing — and it is a two-step bet with uncertain
   payoff, since the published Green-Jacobi results are for linear-elastic SPD
   systems and ours is finite-strain saddle-point.

---

## 13. References

- Moulinec, H., Suquet, P. (1998). CMAME 157, 69–94.
- Zeman, J., Vondřejc, J., Novák, J., Marek, I. (2010). *Accelerating a
  FFT-based solver … by conjugate gradients.* JCP.
- Vondřejc, J., Zeman, J., Marek, I. (2014). *An FFT-based Galerkin method for
  homogenization of periodic media.*
- Willot, F. (2015). *Fourier-based schemes for computing the mechanical
  response of composites with accurate local fields.* C. R. Mécanique 343, 232–245.
- Eisenstat, S., Walker, H. (1996). *Choosing the forcing terms in an inexact
  Newton method.* SIAM J. Sci. Comput. 17(1), 16–32.
- Lucarini, S., Segurado, J. (2019). *DBFFT: a displacement based FFT approach
  for non-linear homogenization of the mechanical behavior.* Int. J. Eng. Sci.
- Ladecký, M., Pultarová, I., Bignonnet, F., Jödicke, I., Zeman, J.,
  Pastewka, L. *Jacobi-accelerated FFT-based solver for smooth high-contrast
  data.* arXiv:2508.02613.
- Ladecký, M. et al. *An optimal preconditioned FFT-accelerated finite element
  solver for homogenization.* Appl. Math. Comput. (2023).
- Schneider, M. (2021). *A review of nonlinear FFT-based computational
  homogenization methods.* Acta Mechanica.
