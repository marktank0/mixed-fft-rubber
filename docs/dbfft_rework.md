# DBFFT: Displacement-Based Reformulation

**Status: structurally validated, not yet converging efficiently. Work in
progress on branch `claude/dbfft-displacement-rework`. Do not use for
production results.**

---

## 1. Why this rework exists

The F-based mixed solver has a structural problem that three separate
investigations all trace back to one cause: the unknown is the deformation
gradient, which must satisfy a compatibility constraint.

| consequence | evidence |
|---|---|
| the operator is singular off the compatible subspace | rank 1371 of 3430 at N=7; nullity 2059 |
| the preconditioner had to be restricted to keep iterates inside it | otherwise it converged to the wrong answer (+17.4 % in P11) |
| the restricted preconditioner has an O(contrast) condition number | \(\lambda_\max(M^{-1}A) \approx 572\) at contrast 500 |
| the published high-contrast preconditioner (Green-Jacobi) does not transfer | \(D^{-1/2}\) is a pointwise scaling and destroys compatibility |

See `docs/green_reference_preconditioning.md` for all four in detail.

Writing \(F = \bar F + \nabla u\) makes compatibility automatic. Every \(u\) is
admissible, so the constraint - and everything that follows from it -
disappears.

---

## 2. Formulation

Unknown vector: `[ u (3N³) | p (N³) | free Fbar components ]`

- **Gradient** \(F_{ij} = \partial u_i/\partial x_j\), applied spectrally with
  the true complex derivative symbol \(\xi_j\). Unlike the F-based projector,
  where the factor \(i\) and any common scale cancel in
  \(\xi_j\xi_m/|\xi|^2\), here \(\xi\) and \(\bar\xi\) are applied separately,
  so both are kept: \(\xi_j = 2\pi i k_j\) for the spectral scheme, and
  Willot's rotated difference including its \(1/(4h)\) factor.
- **Equilibrium residual** \(r_i = \overline{\xi_j} P_{ij}\), which is the
  exact adjoint of the gradient - this is what makes the discrete operator
  symmetric for a symmetric tangent.
- **Incompressibility** \(1 - J + p\kappa^{-1}\), unchanged from the F-based
  solver.
- **Macroscopic control**: the stress-controlled \(\bar F\) components are
  carried as extra unknowns; the strain-controlled ones are prescribed.

**Preconditioner**: the reference acoustic tensor
\(\Gamma_{ik} = \overline{\xi_j}\,\mathbb{K}^0_{ijkl}\,\xi_l\), a 3×3 Hermitian
block per frequency, extended to 4×4 with the pressure coupling.

**Block scaling.** `div_adj` carries a factor \(|\xi| \sim \pi N\), so the raw
equilibrium residual is ~1000× the incompressibility residual at N=31. Left
unscaled, GMRES minimises the total norm, satisfies equilibrium and ignores
incompressibility, and the resulting Newton direction fails line search at
*every* step size. The equilibrium and macroscopic-stress equations are
therefore divided by a stress scale \(\sigma = \langle\|\mathbb{K}\|\rangle\),
and the pressure is carried as \(p = \sigma\tilde p\).

---

## 3. What is verified

`test_dbfft.py`:

| check | result |
|---|---|
| `div_adj` is the exact adjoint of `grad` (spectral) | 1.9e-16 |
| `div_adj` is the exact adjoint of `grad` (Willot) | 6.4e-16 |
| `grad(constant) = 0`, `grad` output has zero mean | exact |
| **operator rank at N=5** | **499 of 502 — nullity 3** |
| nullity is exactly the three rigid translations | yes |
| reference acoustic tensor Hermitian | error 0.0 |
| frequencies with singular \(\Gamma\) | 1 of 729 (only \(\xi = 0\), as it must be) |
| condition number of \(\Gamma\) | **3.00 median, 3.00 max** |

and separately, by finite differences against the residual:

| block | matvec vs \(-\partial R/\partial X\) |
|---|---|
| u | 1.10e-09 |
| p | 1.92e-10 |
| Fbar | 1.04e-09 |

**The structural claim is proven.** Compare directly:

| | F-based | DBFFT |
|---|---|---|
| operator size (N=5/7 scale) | 3430 | 502 |
| nullity | 2059 | **3** |
| reference symbol | rank 4 of 10, needs a pseudo-inverse, condition 17 | **3×3 Hermitian, condition 3.00** |

---

## 4. What does NOT work yet

**The solver does not converge efficiently.** At contrast 100, N=31, one
increment, the F-based solver finishes in 43 s (305 Krylov, 4 Newton). DBFFT
had not completed a single Newton step after 10 minutes.

Before the block scaling of section 2 it failed line search at every step size
down to 0.003 - that is fixed, and the Jacobian is confirmed correct, so the
remaining problem is in the linear solve or the preconditioner's effectiveness,
not the formulation.

Leads, in the order worth trying:

1. **Verify the preconditioner actually preconditions.** Measure
   \(\|M^{-1}Ax - x\|/\|x\|\) as was done for the F-based solver. The 4×4
   block spans \(\Gamma \sim 40\!-\!8900\) against a pressure entry
   \(\kappa^{-1}\sigma \sim 0.06\), a condition number around \(10^5\) per
   frequency; the pressure row/column probably needs its own scaling rather
   than sharing \(\sigma\).
2. **Check the zero-frequency handling.** The preconditioner sets the entire
   \(\xi = 0\) block to zero, which leaves the macroscopic pressure mode and
   the `Fbar` unknowns completely unpreconditioned. In the F-based solver that
   mode was handled explicitly via `_constrain_zero_mode`.
3. **Per-frequency equilibration** of the 4×4 block before inversion, which
   would handle both of the above automatically.
4. Only then benchmark against the F-based solver.

---

## 5. The validation target

DBFFT and the *corrected* F-based solver discretise the **same problem** - the
F-based one restricted to compatible fields, this one parameterising exactly
those fields by \(u\). They must therefore agree on the homogenised stress.
Reference values already measured with the corrected F-based solver
(structure `1_voxel`, N=31, `reference="matrix"`):

| contrast | P11 |
|---|---|
| 10 | 2.887333 |
| 100 | 3.278909 |
| 500 | 3.425616 |

and for one increment at contrast 100, P11 = 1.272604498.

Agreement to solver tolerance is the acceptance criterion; a mismatch means
the reformulation is wrong, not merely slow.

---

## 6. References

- Lucarini, S., Segurado, J. (2019). *DBFFT: a displacement based FFT approach
  for non-linear homogenization of the mechanical behavior.* Int. J. Eng. Sci.
- Ladecký, M., Pultarová, I., Bignonnet, F., Jödicke, I., Zeman, J.,
  Pastewka, L. *Jacobi-accelerated FFT-based solver for smooth high-contrast
  data.* arXiv:2508.02613.
- Willot, F. (2015). *Fourier-based schemes for computing the mechanical
  response of composites with accurate local fields.* C. R. Mécanique.
