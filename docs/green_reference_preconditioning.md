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

**This is a measured defect, not a theoretical concern.** It affects every run
made with `preconditioner="reference"`.

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

### Candidate fix (tested, not applied)

Build the symbol as the restriction of the operator to the compatible
subspace,

$$
\widehat{A}_{F,0}(\xi) = \widehat{\mathcal{G}}(\xi)\,\mathbb{K}_0\,\widehat{\mathcal{G}}(\xi),
$$

so that, \(\widehat{\mathcal{G}}\) being a symmetric projector, the
pseudo-inverse's range is contained in
\(\mathrm{range}(\widehat{\mathcal{G}})\). Measured on the same probe:

```
current  G*K    -> output incompatible = 8.2e-01
fixed    G*K*G  -> output incompatible = 4.9e-13
```

The mixed symbol needs the same treatment on its \(H_0^{T}\) row. This has
not been applied because it changes every historical result produced with
`preconditioner="reference"`; that is a call for the project owner. Until it
is resolved, `reference="mean"` is the only mode with an established results
history and remains the default.

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
