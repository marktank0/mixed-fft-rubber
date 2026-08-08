# Green / Reference-Operator Preconditioning

This note records the mathematical basis for a Green/reference-operator preconditioner for this repository's FFT-Galerkin solvers.

## Literature Base

The relevant line of work is:

- Zeman, Vondrejc, Novak, Marek, "Accelerating a FFT-based solver for numerical homogenization of periodic media by conjugate gradients" (2010): recasts the FFT cell problem as a linear system that can be solved by Krylov methods and reports better high-contrast convergence than fixed-point schemes.
- Vondrejc, Zeman, Marek, "An FFT-based Galerkin Method for Homogenization of Periodic Media" (2014): gives the variational FFT-Galerkin structure used by this repository's standard solver.
- Ladecky et al., "Jacobi-accelerated FFT-based solver for smooth high-contrast data" (2026): discusses three preconditioners: Green, Jacobi, and Green-Jacobi. The important construction for this repository is the Green preconditioner, i.e. the inverse or pseudo-inverse of a homogeneous reference operator in Fourier space.
- Lucarini and Segurado, "DBFFT: A displacement based FFT approach for non-linear homogenization of the mechanical behavior" (2019): shows that changing the unknown to displacement yields a full-rank Hermitian system where Krylov preconditioners are more natural than in strain/deformation-gradient FFT systems.

The key warning from the literature is that a local diagonal/Jacobi idea alone is not the same as a Green-Jacobi preconditioner. Green-Jacobi has the operator form

$$
M_{GJ}^{-1} = D^{-1/2} G_0^{-1} D^{-1/2},
$$

where \(G_0^{-1}\) is the Green/reference inverse and \(D\) is a diagonal extracted from the actual system. The implementation below uses only the Green/reference part.

## Current Standard Solver

The standard finite-deformation FFT-Galerkin solver linearizes the equilibrium equation as

$$
\mathcal{G}\left(\mathbb{K} : \Delta F\right) = R_F,
$$

where:

- \(F\) is the local deformation-gradient field,
- \(\mathbb{K} = \partial P / \partial F\) is the local material tangent,
- \(P\) is the first Piola-Kirchhoff stress,
- \(\mathcal{G}\) is the compatible projection operator implemented with FFTs,
- \(R_F = \mathcal{G}(\bar{P} - P)\) in the code.

The code applies the matrix-free operator

$$
A_F \Delta F = \mathcal{G}\left(\mathbb{K} : \Delta F\right).
$$

In Fourier space, for a homogeneous reference tangent \(\mathbb{K}_0\), the reference operator has symbol

$$
\widehat{A}_{F,0}(\xi)_{ijmn}
=
\widehat{\mathcal{G}}_{ijab}(\xi)\,\mathbb{K}_{0,abmn}.
$$

A Green/reference preconditioner applies the pseudo-inverse of this symbol:

$$
M_F^{-1} r
=
\mathcal{F}^{-1}
\left[
\widehat{A}_{F,0}(\xi)^{+}\,\widehat{r}(\xi)
\right],
$$

where \(+\) denotes the Moore-Penrose pseudo-inverse. The pseudo-inverse is needed because projection-based FFT-Galerkin operators are rank-deficient in some modes, especially at zero frequency.

In the implementation, \(\mathbb{K}_0\) is chosen as the current volume average of the voxel tangents:

$$
\mathbb{K}_0 = \langle \mathbb{K}(x) \rangle.
$$

The zero-frequency mode needs special handling because the charge file fixes some macroscopic \(F_{ij}\) components and frees others through macroscopic \(P_{ij}\) control. The implementation therefore restricts the zero-mode pseudo-inverse to the free stress-controlled components only. This prevents the preconditioner from changing prescribed macroscopic deformation components such as a fixed \(F_{11}\).

## Current Mixed Solver

The mixed solver uses deformation gradient and pressure as unknowns:

$$
x =
\begin{bmatrix}
\Delta F \\
\Delta p
\end{bmatrix}.
$$

Its Newton linearization is

$$
\begin{bmatrix}
\mathcal{G}\mathbb{K} & \mathcal{G}(J F^{-T}) \\
J F^{-T} : (\cdot) & -\kappa^{-1}
\end{bmatrix}
\begin{bmatrix}
\Delta F \\
\Delta p
\end{bmatrix}
=
\begin{bmatrix}
R_F \\
R_p
\end{bmatrix}.
$$

The code implements this as:

$$
R_F = \mathcal{G}(\bar{P}-P),
$$

$$
R_p = 1 - J + p\,\kappa^{-1},
$$

and the matrix-vector product:

$$
\Delta R_F =
\mathcal{G}\left(\mathbb{K}:\Delta F + JF^{-T}\Delta p\right),
$$

$$
\Delta R_p =
JF^{-T}:\Delta F - \kappa^{-1}\Delta p.
$$

For a homogeneous reference state

$$
\mathbb{K}_0 = \langle \mathbb{K} \rangle,\qquad
H_0 = \langle JF^{-T} \rangle,\qquad
\alpha_0 = \langle \kappa^{-1} \rangle,
$$

the Fourier-space reference block is

$$
\widehat{A}_{M,0}(\xi)
=
\begin{bmatrix}
\widehat{\mathcal{G}}(\xi)\mathbb{K}_0
&
\widehat{\mathcal{G}}(\xi)H_0
\\
H_0^T
&
-\alpha_0
\end{bmatrix}.
$$

The reference preconditioner is then

$$
M_M^{-1} r
=
\mathcal{F}^{-1}
\left[
\widehat{A}_{M,0}(\xi)^{+}\,\widehat{r}(\xi)
\right].
$$

At zero frequency, the mixed preconditioner uses the same free stress-controlled deformation components, plus the pressure component. Fixed macroscopic \(F_{ij}\) components are removed from the zero-mode reference inverse.

## Known Defect: the preconditioner does not preserve the compatible subspace

**This is a confirmed defect, established by exact rank analysis.** It affects
every run made with `preconditioner="reference"`.

### The proof

The mixed Newton operator \(A\) was formed densely at \(N=7\) (contrast 100,
\(\phi = 7.9\,\%\)) directly from `fg/mxfft.py`, and its singular values
computed exactly:

| quantity | value |
|---|---|
| operator size | 3430 (F-block 3087 + p-block 343) |
| rank(\(A\)) | **1371** |
| dim null(\(A\)) | **2059** |
| dim of the compatible subspace (F 1028 + p 343) | **1371** |
| rank of \(A\) restricted to the compatible subspace | **1371 of 1371 — nonsingular** |
| smallest singular value of that restriction | 7.03e-3 (largest 4.97e2) |

This says three things precisely:

1. \(A\) as implemented is **massively singular** — a 2059-dimensional null
   space out of 3430. That is expected: the F-row of \(A\) is
   \(\mathcal{G}(\dots)\), so it only ever produces compatible output, giving
   `rank(G)` \(\approx 3N^3\) equations for \(9N^3\) F-unknowns.
2. Restricted to the compatible subspace, \(A\) is **exactly nonsingular**
   (rank 1371 of 1371). The physical problem is therefore well-posed with a
   *unique* solution — the formulation is sound.
3. Since the restriction is injective,
   \(\mathrm{null}(A) \cap \{\text{compatible}\} = \{0\}\). The null space is
   entirely non-physical: it consists of incompatible F fields.

So the method's correctness rests on the iterates never leaving the compatible
subspace. Unpreconditioned Krylov satisfies this automatically: \(b\)'s F-block
is \(\mathcal{G}\)-projected (measured: incompatible content 4.4e-16) and every
application of \(A\) returns \(\mathcal{G}(\dots)\), so the whole Krylov space
\(\mathrm{span}\{b, Ab, A^2b, \dots\}\) stays compatible.

**The preconditioner breaks exactly this property.** Measured on the same
system, feeding basis vectors of the compatible subspace through \(M^{-1}\):

```
||M^-1 v - Pi(M^-1 v)|| / ||M^-1 v||   over compatible v
    min 1.96e-01   median 5.94e-01   max 6.85e-01
```

So \(M^{-1}\) moves a *majority* of each compatible vector out of the subspace,
into directions that lie in \(\mathrm{null}(A)\) — directions the residual
cannot see and which are not part of any physical deformation.

\(\widehat{\mathcal{G}}\) is the orthogonal projector onto compatible
(gradient) fields, so a converged fluctuation field must satisfy
\(\mathcal{G}(\delta F) = \delta F\). With left preconditioning the GMRES
iterates lie in \(\mathrm{span}\{M^{-1}b, M^{-1}AM^{-1}b, \dots\}\). The
symbol implemented above is \(\widehat{\mathcal{G}}\mathbb{K}_0\), whose
pseudo-inverse has range

$$
\mathrm{range}\left[(\widehat{\mathcal{G}}\mathbb{K}_0)^{+}\right]
= \mathbb{K}_0^{T}\,\mathrm{range}(\widehat{\mathcal{G}}),
$$

which equals \(\mathrm{range}(\widehat{\mathcal{G}})\) only when
\(\mathbb{K}_0 \propto I\). A finite-strain hyperelastic tangent never is, so
\(M^{-1}\) carries the iterates *out* of the compatible subspace. The
incompatible content is invisible to the \(R_F\) residual block (which is
itself \(\mathcal{G}\)-projected) but still contributes to the
volume-averaged stress.

Combining this with the rank analysis gives the complete mechanism: the
converged \(\delta X\) is *(the unique physical solution) + (an arbitrary
component of \(\mathrm{null}(A)\))*, and the preconditioner decides which null
component you get. That is precisely why the three reference tangents produce
three different answers while all three drive the residual to \(10^{-12}\).

Measured by feeding a compatible field through the preconditioner:

```
input field                                  incompatible content = 3.6e-16
K_ref isotropic    -> M^-1 output            incompatible content = 3.9e-16
K_ref anisotropic  -> M^-1 output            incompatible content = 8.2e-01
```

Consequence at the solver level (contrast 100, N = 31, `tol_rel = 1e-9`,
nonlinear residual driven to ~1e-12 in every case):

| reference | P11 | incompatible content of converged field |
|---|---|---|
| mean | 1.0835510581 | 0.640 |
| matrix | 1.0804848613 | 0.646 |
| mid | 1.0837578465 | 0.640 |

The values are stable to nine digits within each mode and do not move when
`tol_rel` is tightened, so this is not a tolerance effect: the reference
choice selects which solution the iteration lands on.

### The fix (APPLIED)

The symbol is now built as the reference operator **restricted to the
compatible subspace**, \(\Pi A_0 \Pi\) with \(\Pi = \mathrm{diag}(\widehat{\mathcal{G}}, I)\):

$$
\widehat{A}_{M,0}(\xi)
=
\begin{bmatrix}
\widehat{\mathcal{G}}\mathbb{K}_0\widehat{\mathcal{G}}
&
\widehat{\mathcal{G}}H_0
\\
H_0^{T}\widehat{\mathcal{G}}
&
-\alpha_0
\end{bmatrix},
\qquad
\widehat{A}_{F,0}(\xi) = \widehat{\mathcal{G}}\mathbb{K}_0\widehat{\mathcal{G}} .
$$

Both the deformation block and the pressure row are projected. Since
\(\widehat{\mathcal{G}}\) is a symmetric projector,
\(\mathrm{range}\left[(\Pi A_0 \Pi)^{+}\right] \subseteq \mathrm{range}(\Pi)\),
so the preconditioner keeps the Krylov space inside the subspace on which the
operator is nonsingular — and GMRES therefore converges to the *unique*
physical solution.

`precond_restrict=False` (solver argument / YAML `solver.precond_restrict`)
restores the old symbol, and exists only to reproduce results produced before
the fix.

### Validation

**1. The preconditioner now preserves the subspace.** Same probe as above:

| | leakage \(\lVert M^{-1}v - \Pi M^{-1}v\rVert / \lVert M^{-1}v\rVert\) |
|---|---|
| before | min 1.96e-01, median 5.94e-01, max 6.85e-01 |
| after | min 1.53e-15, median 1.84e-15, max 2.14e-15 |

**2. The reference tangent no longer changes the answer** — the defining
property of a preconditioner. Contrast 100, \(N=31\), `tol_rel=1e-9`:

| reference | before: \(\bar P_{11}\) | after: \(\bar P_{11}\) |
|---|---|---|
| mean | 1.0835510581 | **1.2726045390** |
| matrix | 1.0804848613 | **1.2726045390** |
| mid | 1.0837578465 | **1.2726045390** |
| spread | 3.02e-03 | **3.98e-12** |

and the converged fluctuation field is compatible again (incompatible content
6.4e-01 → ~2.5e-13).

**3. An independent control confirms which answer is correct.**
Unpreconditioned GMRES provably cannot leave the compatible subspace (\(b\)'s
F-block is \(\mathcal{G}\)-projected and every application of \(A\)
re-projects), so it is an unbiased reference. Contrast 100, \(N=31\),
`tol_rel=1e-7`:

| run | \(\bar P_{11}\) | incompatible | Krylov | wall |
|---|---|---|---|---|
| unpreconditioned (control) | 1.2726045390 | 1.5e-11 | 6314 | 1256 s |
| preconditioned, **fixed** | 1.2726045376 | 2.7e-13 | **217** | **23 s** |
| preconditioned, pre-fix | 1.0835510573 | 6.4e-01 | 130 | 15 s |

The fixed preconditioner agrees with the control to **9 significant figures**
while being a 29x iteration / 55x wall-time accelerator over no
preconditioning. The pre-fix result is wrong by **+17.4 %** in
\(\bar P_{11}\).

**4. The decisive test: on a homogeneous body the restricted symbol IS the
operator inverse.**

This is the sharpest available check, and it does not depend on any claim
about which answer is correct. If \(\mathbb{K}(x) = \mathbb{K}_0\) everywhere,
then \(\widehat{\mathcal{G}}\mathbb{K}_0\widehat{\mathcal{G}}\) *is* the
operator restricted to the compatible subspace, so \(M^{-1}A\) must act as the
identity there. Measured on a truly homogeneous cell (both phases given
identical E and identical Poisson ratio), with a random compatible test vector:

| body | symbol | \(\lVert M^{-1}Ax - x\rVert/\lVert x\rVert\) |
|---|---|---|
| homogeneous | unrestricted (pre-fix) | 7.06e-01 |
| homogeneous | **restricted (fixed)** | **6.71e-15** |
| heterogeneous, contrast 100 | unrestricted (pre-fix) | 1.83 |
| heterogeneous, contrast 100 | restricted (fixed) | 3.83e+01 |

Only the restricted symbol has the defining property. `test_preconditioner.py`
asserts this.

The last row is the cost of correctness, and explains the runtime regression
below: on a *heterogeneous* body the restricted symbol is a much poorer
approximate inverse than the unrestricted one appeared to be. That is not a
defect. The restricted preconditioner is approximating the inverse of the real,
ill-conditioned compatible problem, whose conditioning scales with phase
contrast — the classical FFT-homogenization difficulty. The unrestricted symbol
looked better behaved only because it was preconditioning a different and
easier operator, and converging to the wrong answer.

### Performance consequence, and what to do about it

Correcting the preconditioner makes the solver substantially more expensive,
and the gap grows with contrast. Measured (N=31, 3 increments, structure
`1_voxel`, total Krylov iterations):

| contrast | legacy (C5) | corrected (C5) | corrected, `reference="matrix"` |
|---|---|---|---|
| 10 | 115 | 131 | 112 |
| 100 | 447 | 1388 | 1145 |
| 500 | 1736 | 34818* | 15431* |

\* truncated by the benchmark's 1000-iteration cap; see below.

Two things follow.

**The reference tangent now matters, a lot.** With `reference="mean"` at
contrast 100 and \(\phi = 8.9\,\%\), \(\mathbb{K}_0 \approx 10\,\mathbb{K}_\text{matrix}\)
— the average is dominated by the stiff phase, so *every* voxel is badly
preconditioned. With `reference="matrix"` the 91 % of the cell that is matrix
is preconditioned near-exactly and only the filler is left as a
low-volume-fraction perturbation. Measured gain in the corrected family:
1.21x at contrast 100, **2.26x at contrast 500** (Newton 105 -> 47, step cuts
6 -> 2). This is exactly the D5 argument from the plan document, and it only
became visible once the preconditioner was solving the right problem.

**Iteration caps sized for the legacy preconditioner are now far too tight.**
The corrected solver needs several hundred to a few thousand Krylov iterations
per solve at contrast >= 500, against ~85 for the legacy one. A cap of 1000
(the production default) makes every solve "fail", which triggers load-step
cutting, which exhausts the sub-step budget and reports the case as failed —
a cascade that looks like a robustness collapse but is purely the cap.

The open lever is a better preconditioner, not reverting: `reference="matrix"`
as the production default, and the Green-Jacobi form
\(D^{-1/2}G_0^{-1}D^{-1/2}\) noted in the literature section, which is
designed for exactly this situation and remains unimplemented.

### Impact on existing results

**Every run produced with `preconditioner="reference"` is affected**, which
includes the production sweeps in `Results/`. The error is not a small
tolerance-level perturbation: it was 17 % in \(\bar P_{11}\) on the case
measured. Affected runs need to be regenerated. The magnitude will depend on
contrast, filler fraction and strain level, so it should be re-measured per
sweep rather than assumed to be 17 %.

The cost of the fix is that the restricted symbol is a slightly weaker
accelerator: on the contrast-100 case the Krylov count per solve rose from
~130-190 to ~220-290. That is the price of solving the right problem.

## Important Caveats

This is an experimental preconditioner for this repository, not a fully validated Green-Jacobi implementation.

The literature-backed Green and Green-Jacobi preconditioners are most natural for symmetric positive-definite displacement-based systems or small-strain linear systems. This repository's mixed finite-deformation system is a saddle-point-like deformation-pressure system. The original paper also notes that the mixed tangent matrix is not strictly symmetric and leaves rigorous well-posedness and preconditioning analysis as future work.

For that reason:

- no preconditioner remains the default,
- `preconditioner="reference"` must be requested explicitly,
- the standard solver applies the reference preconditioner through SciPy CG,
- the mixed solver applies the reference preconditioner through SciPy GMRES because the pressure block makes the reference system indefinite,
- the mixed solver also accepts `preconditioner="gmres"` as a diagnostic mode: GMRES without the Green/reference preconditioner,
- the implementation uses pseudo-inverses and should be treated as experimental,
- results should always be compared against `preconditioner=None`.

## Implementation Mapping

The implementation uses matrix-free SciPy `LinearOperator` objects:

```python
Aop = LinearOperator(..., matvec=KdX)
Mop = LinearOperator(..., matvec=apply_reference_preconditioner)
sp.cg(A=Aop, M=Mop, b=b, ...)
```

For the mixed solver, the corresponding explicit reference-preconditioned path is:

```python
sp.gmres(A=Aop, M=Mop, b=b, ...)
```

For comparison testing, the mixed solver also allows:

```python
sp.gmres(A=Aop, M=None, b=b, ...)
```

through `preconditioner="gmres"`. The default mixed path is unchanged and still uses CG with no preconditioner. The Green/reference preconditioner is assembled once per Newton iteration from the current averaged tangent fields and then applied inside every GMRES iteration when `preconditioner="reference"` is used.

For the standard solver:

$$
\widehat{A}_{F,0}(\xi)^+
\in \mathbb{R}^{9\times 9}
$$

is stored for every frequency.

For the mixed solver:

$$
\widehat{A}_{M,0}(\xi)^+
\in \mathbb{R}^{10\times 10}
$$

is stored for every frequency.

This preserves the FFT character of the method, but it increases memory use. For a \(63^3\) grid, the mixed preconditioner symbol stores \(10\times10\times63^3\) floating-point values, about 200 MB in `float64`, before FFT work arrays.
