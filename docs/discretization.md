# Discretization of the Green Operator: Spectral vs Willot's Rotated Scheme

This note records what discretization the solvers use, why the spectral choice
is a poor fit for high-contrast two-phase microstructures, and the
implementation and verification of Willot's (2015) rotated finite-difference
scheme as an alternative.

---

## 1. What the code used before

Both solvers built the Green projection symbol from the raw wave numbers:

```python
freq = np.arange(-(N-1)/2., +(N+1)/2.)
q    = meshgrid(freq, freq, freq)
Ghat4[i,j,l,m] = delta(i,l) * q[j]*q[m] / (q.q)
```

That is

$$
\widehat{\mathcal{G}}_{ijlm}(\xi) = \delta_{il}\,\frac{\xi_j \xi_m}{|\xi|^2},
$$

i.e. **the exact trigonometric derivative** $\tilde{\xi}_j = i\xi_j$ — the
Moulinec–Suquet / FFT-Galerkin spectral discretization of the de
Geus/Vondřejc/Zeman lineage. (The array holds integer wave numbers rather than
`2*pi*fftfreq(n)`; the constant $2\pi/L$ and the factor $i$ both cancel in
$\xi_j\xi_m/|\xi|^2$, so it is the same operator.)

It is the most accurate choice for smooth fields and the worst for a
two-phase composite: **a truncated Fourier basis cannot represent a
discontinuity.** At a filler/matrix interface the strain genuinely jumps, and
the jump amplitude grows with the contrast, so the discrete solution carries
Gibbs oscillations radiating from every interface. This is a *representation*
error — no preconditioner or Krylov acceleration removes it, because the
solver is converging accurately to a discrete solution that is itself
polluted.

---

## 2. Willot's rotated scheme

Willot (2015) replaces the exact derivative with a forward finite difference
taken along the voxel *diagonal* — a forward difference in the differentiated
direction, averaged over the two transverse directions. In 3D, with voxel size
$h$ and $\varphi_j = e^{i\xi_j h}$,

$$
\tilde{\xi}_1 = \frac{1}{4h}(\varphi_1 - 1)(\varphi_2 + 1)(\varphi_3 + 1),
$$

and cyclically for $\tilde{\xi}_2, \tilde{\xi}_3$. The projector becomes

$$
\widehat{\mathcal{G}}_{ijlm}(\xi) = \delta_{il}\,
\frac{\tilde{\xi}_j\, \overline{\tilde{\xi}_m}}{|\tilde{\xi}|^2},
\qquad |\tilde{\xi}|^2 = \sum_m |\tilde{\xi}_m|^2 .
$$

The conjugate on the second index is what keeps the operator Hermitian and
idempotent for a complex symbol; with the real spectral $\xi$ it reduces to the
previous expression.

Two simplifications matter for the implementation. The prefactor $1/(4h)$ is a
common **real** scale across all components, and any half-voxel centring
convention is a common **phase** — both cancel identically in
$\tilde{\xi}_j\overline{\tilde{\xi}_m}/|\tilde{\xi}|^2$. So the code uses the
unnormalised, uncentred form and the projector is unaffected.

Because $\tilde{\xi}_j$ is degree one in each $\varphi_m$, its inverse
transform is supported on a single $2\times2\times2$ voxel block — the
defining property of a finite difference, and the reason the scheme represents
interfaces without ringing.

---

## 3. Implementation

`build_Ghat4(N, stress_control, ndim, discretization)` now lives in
`FFT_simulation/fg/preconditioning.py` and is shared by both solvers; `_wave_vectors()`
returns the derivative symbol for the selected scheme (a **real** array for
`"fourier"`, so the default path is arithmetically untouched, and a complex
array for `"willot"`).

Consequences handled:

- `Ghat4` becomes `complex128` under Willot, which doubles its memory
  (162 MB → 324 MB at $N=63$) and makes the projection contraction more
  expensive. This is the main cost of the scheme.
- `build_mixed_reference_symbol` previously hard-coded `dtype=float`; it now
  takes the dtype from `Ghat4`, so the preconditioner works with a complex
  symbol.
- The preconditioner's output stays exactly real: $\widehat{\mathcal{G}}$
  satisfies $\widehat{\mathcal{G}}(-\xi) = \overline{\widehat{\mathcal{G}}(\xi)}$
  and `pinv` commutes with conjugation, so the symbol is conjugate-symmetric
  and the inverse transform is real. The `.real` in
  `apply_*_preconditioner` is exact, not a truncation.
- `FFT_simulation/fg/fft.py`'s duplicated inline $O(81 N^3)$ Python loop for `Ghat4` was
  replaced by the shared builder (verified bitwise identical).

Select with `discretization: fourier | willot` in the YAML `solver:` block, or
`calculate(discretization=...)`. Default is `"fourier"` — unchanged behaviour.
The chosen scheme is recorded in `solver_stats.json`.

---

## 4. Verification

`test_discretization.py` checks the properties that define the scheme, rather than
trusting the formula:

| check | result |
|---|---|
| `"fourier"` reproduces the previous `build_Ghat4` | **bitwise identical** |
| `"fourier"` reproduces the old `FFT_simulation/fg/fft.py` inline loop | **bitwise identical** |
| real-space stencil of $\tilde{\xi}_1$ | 8 non-zero voxels on a $2\times2\times2$ cube, values $\pm1$, sign splitting along axis 0 only — i.e. a forward difference averaged over the 4 transverse corners |
| $\widehat{\mathcal{G}}$ Hermitian | error 0.0 |
| $\widehat{\mathcal{G}}$ idempotent | error 3.3e-16 |
| $\mathcal{G}(\text{real field})$ is real | imaginary part 2.0e-16 relative |
| $\mathcal{G}$ fixes Willot-gradient fields | 4.8e-16 |
| $\mathcal{G}_\text{willot}$ vs a *Fourier*-gradient field | 2.7e-01 — the schemes genuinely differ |
| $\mathcal{G}_\text{willot} \to \mathcal{G}_\text{fourier}$ at the lowest modes | agree to ~1e-16 for every $N$ |

The last row is worth noting: the two projectors coincide to machine precision
at low frequency and differ only at high frequency. That is exactly the
intended behaviour — the projector depends only on the *direction* of
$\tilde{\xi}$, which is parallel to $\xi$ in the long-wavelength limit, so the
schemes are identical where the fields are smooth and differ only where the
ringing lives.

---

## 5. Results

See section 6 for the measured contrast ladder. Two caveats govern how far
these can be pushed:

**The measured numbers below predate the preconditioner fix.** They were
produced while `preconditioner="reference"` still left the compatible subspace
(see `docs/green_reference_preconditioning.md`), so *both* schemes converged to
a physical solution plus an arbitrary incompatible component. The **iteration
counts remain a fair comparison** — both schemes ran under the same
preconditioner — but the **accuracy columns are not meaningful** and the
comparison should be regenerated now that `precond_restrict=True` is the
default.

**A definitive accuracy verdict needs a resolution study.** The standard
demonstration that Willot's scheme is more accurate is that its local fields
converge faster under mesh refinement. That requires $N=63$ runs, which were
out of budget on the current hardware.

---

## 6. Measured

Structure `1_voxel.npz`, \(N=31\), \(\phi = 8.9\,\%\), 3 increments of 0.1,
C5 forcing terms, `reference="mean"`. Total Krylov iterations over all Newton
steps (pre-fix preconditioner — see the caveat in section 5):

| contrast | spectral | Willot | change |
|---|---|---|---|
| 10 | 115 | 115 | 0.0 % |
| 100 | 447 | 457 | +2.2 % |
| 500 | 1736 | 1727 | −0.5 % |
| 1000 | 2904 | 3120 | +7.4 % |

Newton counts were identical at every contrast.

**Willot's scheme does not reduce solver cost.** It is iteration-neutral to
slightly worse across two decades of contrast, and each iteration is *more*
expensive because `Ghat4` is complex (double the memory, a costlier
contraction). Net, it is modestly slower.

This is the expected result, and not a failure of the scheme. Willot's rotated
difference is an **accuracy** intervention, not a convergence one: it changes
*which discrete problem* is solved, not how fast the solver reaches it. The
convergence benefit reported in the literature is largely for the *basic
(fixed-point) Moulinec-Suquet iteration*, whose contraction rate depends
directly on the operator's spectrum. This repository solves with
Newton-Krylov plus a Green preconditioner, where the conditioning is already
handled by the preconditioner — so improving the discretization's spectrum
buys little.

The case for adopting Willot here therefore has to be made on local-field
accuracy (no Gibbs ringing at interfaces), not on runtime, and that case is
**not yet demonstrated in this repository** — see the two caveats in
section 5.

---

## 7. References

- Willot, F. (2015). *Fourier-based schemes for computing the mechanical
  response of composites with accurate local fields.* Comptes Rendus
  Mécanique 343(3), 232–245.
- Moulinec, H., Suquet, P. (1998). *A numerical method for computing the
  overall response of nonlinear composites with complex microstructure.*
  CMAME 157, 69–94.
- Vondřejc, J., Zeman, J., Marek, I. (2014). *An FFT-based Galerkin method for
  homogenization of periodic media.* Comput. Math. Appl. 68, 156–173.
- de Geus, T., Vondřejc, J., Zeman, J., Peerlings, R., Geers, M. (2017).
  *Finite strain FFT-based non-linear solvers made simple.* CMAME 318,
  412–430.
- Schneider, M., Ospald, F., Kabel, M. (2016). *Computational homogenization
  of elasticity on a staggered grid.* Int. J. Numer. Meth. Engng 105, 693–720.
- Lucarini, S., Segurado, J. (2019). *On the accuracy of spectral solvers for
  micromechanics based fatigue modeling.* Comput. Mech. 63, 365–382.
