# Reference Preconditioner Discrepancy Investigation

This note is deliberately investigative. The current `reference` mode is fast and useful, so the goal is not to change it aggressively. The goal is to explain why `preconditioner="reference"` gives a slightly different final stress than both `preconditioner=None` and `preconditioner="gmres"`.

Observed result from the larger traction-style run:

- `none`/CG and `gmres` both give final \(P_{11} \approx 3.57\).
- `reference` gives final \(P_{11} \approx 3.30\).
- The difference appears both with 4 increments and 8 increments.
- Therefore, this is probably not random iteration noise and probably not just the CG-vs-GMRES solver choice.

## Current Solver Modes

In `FFT_simulation/fg/mxfft.py`, the current mixed solver has three relevant modes:

$$
\texttt{none}: \quad A \Delta x = b \quad \text{solved by CG}
$$

$$
\texttt{gmres}: \quad A \Delta x = b \quad \text{solved by GMRES}
$$

$$
\texttt{reference}: \quad A \Delta x = b \quad \text{solved by left-preconditioned GMRES with } M^{-1}
$$

where

$$
\Delta x =
\begin{bmatrix}
\Delta F \\
\Delta p
\end{bmatrix}.
$$

The matrix-vector product is the same physical Newton linearization:

$$
\Delta R_F =
\mathcal{G}\left(\mathbb{K}:\Delta F + JF^{-T}\Delta p\right),
$$

$$
\Delta R_p =
JF^{-T}:\Delta F - \kappa^{-1}\Delta p.
$$

The reference mode adds the Fourier-space preconditioner

$$
M^{-1} r =
\mathcal{F}^{-1}
\left[
\widehat{A}_{M,0}(\xi)^+ \widehat{r}(\xi)
\right].
$$

If the preconditioner is mathematically harmless and the linear solve is converged to the same true residual, all three modes should converge to the same Newton update. Since `gmres` agrees with `none`, the discrepancy is most likely in the `reference` path or in how the nonlinear outer loop accepts the `reference` path.

## Hypothesis 1: The Preconditioner Changes the Effective Linear System

Status after diagnostics: the simple form of this hypothesis is now unlikely.

The intended use of SciPy GMRES is left preconditioning:

$$
M^{-1}A\Delta x = M^{-1}b.
$$

This should not change the exact solution if \(M^{-1}\) is a good, fixed, linear operator and the linear system is solved tightly enough. But the implemented \(M^{-1}\) uses pseudo-inverses frequency by frequency:

$$
\widehat{A}_{M,0}(\xi)^+.
$$

Pseudo-inverses can remove null-space components. If those removed components matter for the Newton correction, then `reference` may not be solving exactly the same linear system as `gmres`.

### Test

For the same Newton state \(F,p\), compute:

1. `dX_gmres` from `preconditioner="gmres"`.
2. `dX_reference` from `preconditioner="reference"`.
3. The true residuals:

$$
\|A dX_\mathrm{gmres} - b\|,
\qquad
\|A dX_\mathrm{reference} - b\|.
$$

If the reference solution has a much larger true residual while GMRES reports convergence, the preconditioner is probably filtering or distorting part of the system.

### Diagnostic Result

The diagnostic flag was run on a representative small solve. The observed residuals were of the same order for the unpreconditioned and reference paths:

```text
none/gmres true residual total:      about 1e-6 to 3e-8
reference true residual total:       about 5e-7 to 1e-8
reference preconditioned residual:   about 1e-7 to 1e-9
```

This means the reference path is not merely making the preconditioned residual look small while leaving the original equation \(A\Delta x=b\) badly unsolved. The true residual of the original system is also small.

Therefore, the stress difference is probably not caused by the preconditioner hiding a poor linear solve. A more likely variant is that the mixed linear system is underdetermined or nearly singular in some modes, so two different Krylov/preconditioner paths can return different valid or nearly valid Newton updates.

## Hypothesis 2: The Zero-Frequency Constraint Is Still Not Quite Right

The reference preconditioner currently restricts the zero Fourier mode to:

$$
\{F_{ij}\text{ components controlled by }P_{ij}\} \cup \{p\}.
$$

This was added to stop the preconditioner from changing fixed macroscopic deformation components. It appears to preserve fixed \(F_{11}\) in smoke tests.

But the mixed zero mode is subtle. At \(\xi=0\), the projection operator \(\mathcal{G}\) is not the same as nonzero modes. The charge file enforces some average stress constraints, and the pressure equation enforces volume locally. A wrong zero-mode block can bias the average transverse response and therefore change \(P_{11}\).

### Test

Log the average deformation components after each increment and after each Newton iteration:

$$
\langle F_{11}\rangle,\quad
\langle F_{22}\rangle,\quad
\langle F_{33}\rangle.
$$

Also log the controlled average stresses:

$$
\langle P_{22}\rangle,\quad
\langle P_{33}\rangle.
$$

If `reference` reaches a different \(\langle F_{22}\rangle,\langle F_{33}\rangle\) while still satisfying \(P_{22}\approx P_{33}\approx 0\), the result may be a valid nearby solution. If it does not satisfy the controlled stresses as well, the zero-mode handling is suspect.

## Hypothesis 3: The Outer Newton Stop Uses Step Size, Not Equilibrium Residual

The current outer convergence check is:

$$
\frac{\|\Delta F\|}{\|F\|} < 10^{-5}.
$$

This checks the update size. It does not directly check the actual nonlinear residual:

$$
\|b\| =
\left\|
\begin{bmatrix}
\mathcal{G}(\bar{P}-P) \\
1-J+p\kappa^{-1}
\end{bmatrix}
\right\|.
$$

A preconditioned solve can produce a smaller \(\Delta F\) earlier, so it may satisfy the current stopping criterion at a different true equilibrium residual.

### Test

Do not change the stopping rule yet. Only print, for every outer Newton iteration:

$$
\frac{\|b\|}{\max(\|b_0\|,1)},\qquad
\frac{\|\Delta F\|}{\|F\|},\qquad
\frac{\|\Delta p\|}{\max(\|p\|,1)}.
$$

Compare `none`, `gmres`, and `reference` at the moment each accepts the Newton step. If `reference` stops with a noticeably larger \(\|b\|\), the difference is probably acceptance-related rather than a wrong constitutive result.

## Hypothesis 4: GMRES Convergence Is Being Judged on the Preconditioned Residual

For `reference`, the GMRES callback prints the preconditioned residual norm. That residual is not necessarily the same as the true residual:

$$
\|M^{-1}(b-Ax)\| \neq \|b-Ax\|.
$$

SciPy GMRES should still use the true residual for final convergence, but the printed progress can make the solve look better than the true physical residual.

### Test

After every linear solve, print both:

$$
\frac{\|b-A\Delta x\|}{\max(\|b\|,1)}
$$

and, when `reference` is active:

$$
\frac{\|M^{-1}(b-A\Delta x)\|}{\max(\|M^{-1}b\|,1)}.
$$

If the printed GMRES residual is tiny but the true residual is not, then our progress output is misleading and the GMRES tolerance may need to be interpreted differently.

## Hypothesis 5: The Reference Operator Is Too Crude for Strongly Nonlinear Contrast

The preconditioner uses volume averages:

$$
\mathbb{K}_0 = \langle \mathbb{K}\rangle,\qquad
H_0 = \langle JF^{-T}\rangle,\qquad
\alpha_0 = \langle \kappa^{-1}\rangle.
$$

For high-contrast filler/matrix problems, this homogeneous reference block may be a poor approximation of the actual tangent in some modes. A poor preconditioner should normally slow convergence, not change the exact answer, but combined with pseudo-inverse filtering and loose outer acceptance it could steer the nonlinear path.

### Test

Run the same final load with three reference choices:

1. arithmetic averages, current behavior;
2. phase-majority/matrix reference values;
3. harmonic or compliance-like average for the compressibility term.

This test should come after the residual diagnostics above. It is more invasive and should not be first.

## Hypothesis 6: Pressure Scaling Makes the Mixed Block Numerically Unbalanced

The mixed vector combines deformation-gradient components and pressure components:

$$
\Delta x =
\begin{bmatrix}
\Delta F \\
\Delta p
\end{bmatrix}.
$$

These components may have very different natural scales. The reference preconditioner directly mixes them in a \(10\times 10\) frequency block. If the pressure row/column is poorly scaled, GMRES may converge in a norm that underweights a physically important part of the residual.

### Test

Print residual blocks separately:

$$
\|b_F\|,\qquad \|b_p\|.
$$

Also print linear residual blocks after the solve:

$$
\|(A\Delta x-b)_F\|,\qquad
\|(A\Delta x-b)_p\|.
$$

If `reference` has a much worse pressure-block residual but a good combined residual, pressure scaling is likely involved.

## Suggested Investigation Order

1. Add diagnostic logging only, behind a flag such as `diagnostics=True`.
2. Compare `none`, `gmres`, and `reference` on the same 4-step run.
3. Record after every Newton step:
   - true linear residual after the Krylov solve,
   - preconditioned linear residual for `reference`,
   - nonlinear residual \(\|b\|\),
   - residual split into \(F\)-block and pressure-block,
   - \(\|\Delta F\|/\|F\|\),
   - \(\|\Delta p\|/\max(\|p\|,1)\),
   - average \(F\) and average \(P\).
4. If the true linear residuals match but nonlinear residuals differ, inspect outer stopping.
5. If the reference true linear residual is worse, inspect the preconditioner null-space and zero-frequency block.
6. Only after that, test alternative reference operators or pressure scaling.

## Current Best Guess

Updated after residual diagnostics: the strongest current guess is now:

1. `reference` and `gmres` both solve the original linear residual to a similar tolerance;
2. the mixed linear problem may still allow different update directions, especially in zero-frequency or pressure-coupled modes;
3. the resulting Newton path differs enough to produce a different transverse relaxation and therefore a different \(P_{11}\);
4. the zero-frequency mixed pressure/stress block remains the most suspicious place to inspect if this is resumed later.

This problem is parked for now. If resumed later, the next safest diagnostic is a same-state update comparison: freeze one Newton state and solve the exact same \(A\Delta x=b\) with both `gmres` and `reference`, then compare \(\Delta F\), \(\Delta p\), and especially the mean components of \(\Delta F\).

## Diagnostic Flag Added

The mixed solver now accepts:

```python
prob.calculate(..., preconditioner=None, diagnostics=True)
prob.calculate(..., preconditioner="gmres", diagnostics=True)
prob.calculate(..., preconditioner="reference", diagnostics=True)
```

When enabled, the solver prints after every linear Krylov solve:

```text
linear true residual total ... F-block ... p-block ...
```

This is the residual of the original, unpreconditioned equation:

$$
\frac{\|b-A\Delta x\|}{\max(\|b\|,1)}.
$$

For `preconditioner="reference"`, it also prints:

```text
linear preconditioned residual total ... F-block ... p-block ...
```

This is:

$$
\frac{\|M^{-1}(b-A\Delta x)\|}{\max(\|M^{-1}b\|,1)}.
$$

Interpretation:

- if `reference` had a small preconditioned residual but a much larger true residual than `gmres`, hypothesis 1 would be supported;
- the observed diagnostic result was instead that `reference` and `gmres` have similar true residuals, so the discrepancy is probably not caused by the preconditioner hiding an unsolved original linear residual.
