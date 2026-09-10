# Inexact Newton Forcing Terms and the Reference Tangent (C5, C6)

This note documents two solver changes to `FFT_simulation/fg/mxfft.py` — items **C5**
(inexact Newton with Eisenstat–Walker forcing terms) and **C6**
(configurable reference tangent for the Green preconditioner) of
`docs/high_contrast_convergence_plan.md`. It records the theory, the
implementation mapping, and the measured results, at the level of detail
needed to write the method up.

Both changes target the same quantity — the number of Krylov (GMRES)
iterations spent per load increment at high filler/matrix stiffness
contrast — and neither changes the equations being solved.

---

## 1. Problem setting

The mixed FFT-Galerkin solver treats the deformation gradient $F$ and the
hydrostatic pressure $p$ as independent unknowns. Each load increment is
solved by Newton's method on the residual

$$
R(F,p)
=
\begin{bmatrix}
R_F \\ R_p
\end{bmatrix}
=
\begin{bmatrix}
\mathcal{G}\!\left(\bar{P} - P(F,p)\right) \\[2pt]
1 - J(F) + \kappa^{-1} p
\end{bmatrix},
\qquad J = \det F ,
$$

with the Newton system

$$
\underbrace{
\begin{bmatrix}
\mathcal{G}\,\mathbb{K} & \mathcal{G}\,(JF^{-T}) \\
JF^{-T}\!:\!(\cdot) & -\kappa^{-1}
\end{bmatrix}}_{\textstyle A_k}
\begin{bmatrix}
\Delta F \\ \Delta p
\end{bmatrix}
= R(F_k,p_k),
\qquad
\mathbb{K} = \frac{\partial P}{\partial F} .
$$

$A_k$ is applied matrix-free (`KdX` in `FFT_simulation/fg/mxfft.py`); each application
costs one $\mathcal{G}$ evaluation, i.e. 9 forward and 9 inverse FFTs on the
$N^3$ grid, plus the $9\times 9\times N^3$ tangent contraction. The Krylov
iteration count is therefore an almost exact proxy for cost.

The two blocks carry different physical units (stress; dimensionless
volumetric error), so throughout this note the scalar residual measure is
the **block-normalised merit function** already used by the convergence
test and the line search,

$$
r_k
=
\sqrt{
\left(\frac{\lVert R_F^{(k)}\rVert}{\lVert R_F^{(0)}\rVert}\right)^{2}
+
\left(\frac{\lVert R_p^{(k)}\rVert}{\lVert R_p^{(0)}\rVert}\right)^{2}
} ,
$$

where $(0)$ denotes the first iterate of the current sub-increment. Using a
naive joint norm instead would let the larger-scaled block mask the other.

---

## 2. C5 — Inexact Newton with Eisenstat–Walker forcing terms

### 2.1 Theory

An *inexact* Newton method does not solve the Newton system exactly. It
accepts any step $\Delta x_k$ satisfying

$$
\bigl\lVert R(x_k) - A_k \Delta x_k \bigr\rVert \;\le\; \eta_k \,\bigl\lVert R(x_k) \bigr\rVert ,
\qquad \eta_k \in [0,1),
$$

where $\eta_k$ is the *forcing term*. Dembo, Eisenstat and Steihaug (1982)
showed that the outer iteration retains

* local **linear** convergence when $\eta_k \le \eta_{\max} < 1$,
* **superlinear** convergence when $\eta_k \to 0$,
* **quadratic** convergence when $\eta_k = O\!\left(\lVert R(x_k)\rVert\right)$.

The converged solution is unaffected: the fixed point is defined by
$R(x)=0$ and by the *outer* stopping rule, not by how accurately the
intermediate linear systems were solved. This is what makes C5 safe — it
changes only the path, and its cost.

The practical question is how to pick $\eta_k$. Solving to a fixed tight
tolerance (the previous behaviour, $\eta_k = 10^{-6}$) is wasteful whenever
the linearisation error dominates, which is precisely the situation early in
a Newton sequence: the extra digits of the linear solve are spent resolving a
direction that is about to be discarded. This is *oversolving*.

Eisenstat and Walker (1996) proposed choosing $\eta_k$ from the observed
nonlinear convergence. Their **choice 2** is

$$
\eta_k = \gamma \left(\frac{\lVert R(x_k)\rVert}{\lVert R(x_{k-1})\rVert}\right)^{\alpha},
\qquad \gamma \in (0,1], \quad \alpha \in (1,2],
$$

with the safeguard

$$
\eta_k \leftarrow \max\!\left(\eta_k,\; \gamma\,\eta_{k-1}^{\alpha}\right)
\quad\text{whenever}\quad \gamma\,\eta_{k-1}^{\alpha} > 0.1 ,
$$

which prevents the forcing sequence from collapsing after a single
unusually good step (and thereby oversolving the next one). The rationale is
that the residual reduction actually achieved by the previous step is the
best available estimate of how good the local linear model is; the inner
tolerance tracks it.

We use $\gamma = 0.9$, $\alpha = 2$, and clamp

$$
\eta_k \in [\eta_{\min},\,\eta_{\max}] = [10^{-3},\,10^{-2}].
$$

**Why the clamp is safe.** The outer stopping rule is
$r_k \le \texttt{tol\_rel} = 10^{-5}$ per block. With $\eta_{\min}=10^{-3}$
each Newton step still delivers at least three orders of residual reduction
from the linear solve alone, so the inner tolerance is never the binding
constraint on the achievable outer residual. This is confirmed by the
measured residual histories in §4.2, which reach $\sim 10^{-12}$ relative in
five Newton steps. Consequently the aggressive final-iteration tightening
sketched in the original plan (C5, "tighten to $10^{-3}\eta$ on the last
step") was **not** implemented: it would reintroduce exactly the expensive
solve that C5 exists to remove, and the measurements show it is unnecessary.

### 2.2 Implementation

| item | location |
|---|---|
| forcing-term update | `eisenstat_walker_forcing()`, `FFT_simulation/fg/mxfft.py` |
| per-solve tolerance plumbed into GMRES/CG | `solve_linear(b, rtol)`, `FFT_simulation/fg/mxfft.py` |
| forcing-term state across Newton steps | `newton_increment()`, `FFT_simulation/fg/mxfft.py` |
| logged per Newton step | `solver_stats.json` → `increments[].forcing_terms` |

The forcing term is reset to $\eta_{\max}$ at the start of every
sub-increment, because a sub-increment begins from a new trial state with no
residual history. After each accepted Newton step the ratio $r_k/r_{k-1}$ is
formed from the merit values *before* and *after* the update, and
`eisenstat_walker_forcing` returns the tolerance for the next solve.

Note that the residual ratio is taken from the **damped** step actually
accepted by the line search ($x_{k+1} = x_k + \alpha\,\Delta x_k$), not from
the undamped Newton step. Eisenstat–Walker theory is stated for full steps;
using realised residual norms under globalisation is the standard practice in
globalised inexact-Newton implementations (Pernice & Walker 1998; Kelley
1995) and is the conservative choice here, since a damped step yields a
larger ratio and hence a *looser* — never a spuriously tight — next
tolerance.

Setting `forcing="fixed"` restores the previous behaviour exactly and is
used to reproduce pre-change runs (verified bitwise, §4.1).

### 2.3 A caveat worth stating in a write-up

SciPy's `gmres` with a left preconditioner $M$ measures its relative
tolerance on the **preconditioned** residual,

$$
\bigl\lVert M^{-1}(b - A\,\Delta x) \bigr\rVert \le \eta_k \bigl\lVert M^{-1} b \bigr\rVert ,
$$

whereas the Eisenstat–Walker condition is stated for the true residual. The
two differ by factors bounded through the conditioning of $M$. This makes
$\eta_k$ a proxy rather than a guarantee. It does not affect the converged
solution (the outer stopping rule is enforced on the true nonlinear
residual), and `diagnostics=True` prints both the true and the preconditioned
linear residuals so the discrepancy can be inspected for a given case.

---

## 3. C6 — Configurable reference tangent

### 3.1 Theory

The Green (reference-operator) preconditioner replaces the heterogeneous
tangent by a homogeneous reference $\mathbb{K}_0$ and inverts the resulting
symbol in Fourier space,

$$
M^{-1} r = \mathcal{F}^{-1}\!\left[\widehat{A}_0(\xi)^{+}\,\hat{r}(\xi)\right],
\qquad
\widehat{A}_0(\xi) =
\begin{bmatrix}
\widehat{\mathcal{G}}(\xi)\mathbb{K}_0 & \widehat{\mathcal{G}}(\xi)H_0 \\
H_0^{T} & -\alpha_0
\end{bmatrix},
$$

with $H_0 = \langle JF^{-T}\rangle$ and $\alpha_0 = \langle \kappa^{-1}\rangle$.
The quality of the preconditioner is governed by how close
$\mathbb{K}(x)\,\mathbb{K}_0^{-1}$ is to the identity across the cell.

Previously $\mathbb{K}_0$ was hard-coded to the arithmetic voxel average
$\langle \mathbb{K}\rangle$. At contrast $c$ and filler volume fraction
$\phi$ this is

$$
\langle \mathbb{K}\rangle = (1-\phi)\,\langle\mathbb{K}\rangle_{\mathrm{m}} + \phi\,\langle\mathbb{K}\rangle_{\mathrm{f}} ,
$$

which for $\phi = 0.1$ and $c = 10^3$ is roughly $100\times$ the matrix
tangent — so the preconditioned operator is far from the identity on ~90 % of
the volume. This is diagnosis **D5** of the plan. Three references are now
selectable:

| `reference` | $\mathbb{K}_0$ | rationale |
|---|---|---|
| `"mean"` | $\langle\mathbb{K}\rangle$ | previous behaviour; volume-fraction weighted |
| `"matrix"` | $\langle\mathbb{K}\rangle_{\mathrm{m}}$ | near-exact on the majority phase; concentrates the difficulty in the filler voxels |
| `"mid"` | $\tfrac{1}{2}\left(\langle\mathbb{K}\rangle_{\mathrm{m}} + \langle\mathbb{K}\rangle_{\mathrm{f}}\right)$ | finite-strain analogue of the Moulinec–Suquet $\tfrac{1}{2}(\mathbb{C}_{\min}+\mathbb{C}_{\max})$ reference |

`"mid"` is *not* volume weighted, which is what distinguishes it from
`"mean"`; for the fixed-point scheme the extremal average is the classical
optimum (Moulinec & Suquet 1998), and it is included to test whether that
carries over to the Krylov/preconditioner setting.

### 3.2 Implementation

`reference_average()` in `FFT_simulation/fg/preconditioning.py` computes the reference from
a per-voxel field under any of the three modes. Because the spatial grid
occupies the trailing three axes, the same function serves the rank-7
tangent $\mathbb{K}$ `(3,3,3,3,N,N,N)`, the rank-5 field $JF^{-T}$
`(3,3,N,N,N)` and the scalar $\kappa^{-1}$ `(N,N,N)` without special-casing.
Empty phase masks are skipped, so an unfilled (single-phase) cell falls back
to the whole-cell average under every mode.

Both solvers consume it: `FFT_simulation/fg/mxfft.py` (`solve_linear`) for
$\mathbb{K}_0, H_0, \alpha_0$, and `FFT_simulation/fg/fft.py` for $\mathbb{K}_0$. The
default is `"mean"`, i.e. **unchanged** behaviour — see §5 before selecting
anything else.

---

## 4. Measured results

Test case: `3D_samples/voxels/1_voxel.npz`, $N=31$, $\phi = 8.9\,\%$,
incompressible Neo-Hookean, contrast $c = 100$ ($E = 10$ vs $1000$),
three increments of $0.1$, `preconditioner="reference"`, uniaxial $F_{11}$
with $P_{22}, P_{33}$ stress-controlled. Krylov counts are summed over all
Newton steps of all increments.

### 4.1 C5 and C6, isolated and combined

| run | forcing | reference | Krylov | Newton | wall (s) | speed-up |
|---|---|---|---|---|---|---|
| baseline (pre-change code) | fixed $10^{-6}$ | mean | 1024 | 12 | 104–118 | 1.00× |
| `forcing="fixed"` | fixed $10^{-6}$ | mean | 1024 | 12 | 104 | 1.00× |
| **`forcing="eisenstat_walker"`** | **EW** | **mean** | **447** | **12** | **42** | **2.29×** |
| C5 + C6 | EW | matrix | 505 | 12 | 47 | 2.03× |
| C5 + C6 | EW | mid | 501 | 12 | 45 | 2.04× |
| C6 alone | fixed $10^{-6}$ | matrix | 1130 | 12 | 125 | 0.91× |

Three things to note:

1. **C5 delivers 2.29× fewer Krylov iterations and ~2.5× lower wall time**,
   with the Newton iteration count *unchanged at 12*. That is exactly the
   behaviour inexact-Newton theory predicts: the forcing terms preserve the
   Newton path while removing oversolving. Per-solve counts fall from
   `[62, 81, 84, 84, 70, 88, …]` to `[28, 29, 28, 45, 31, 29, …]`.
2. `forcing="fixed"` reproduces the pre-change solver **bitwise** on both
   $\bar{P}$ and $\bar{F}$, so the old behaviour is exactly recoverable.
3. **C6 did not help at this contrast.** The matrix-phase reference is ~10 %
   *worse* than the volume average ($1130$ vs $1024$ Krylov iterations at
   fixed inner tolerance). The plan's D5 expectation — that a matrix-phase
   reference would cut iteration counts substantially — is **not confirmed**
   at $c = 100$, $\phi \approx 0.09$. It remains to be tested at $c \ge 10^3$,
   which is where D5's argument actually bites.

The realised forcing sequence per increment is
$\eta = [10^{-2},\ \approx 5\times 10^{-3},\ 10^{-2},\ 10^{-3}]$: the second
step relaxes after a large residual drop, the third tightens again as
progress slows, and the final step hits the floor.

### 4.2 Residual histories (contrast 100, one increment, `tol_rel=1e-9`)

```
reference=mean    resF/resF0  1.00e+00  8.73e-02  2.33e-03  2.54e-05  2.30e-08  5.74e-12
reference=matrix  resF/resF0  1.00e+00  8.54e-02  2.13e-03  1.11e-05  2.86e-09  2.41e-12
```

Both reach ~$10^{-12}$ relative in five Newton steps under EW forcing,
confirming that $\eta_{\min}=10^{-3}$ does not limit the attainable outer
residual.

---

## 5. Important finding: the reference preconditioner leaves the compatible subspace

While validating C6 we found a **pre-existing defect in the Green
preconditioner** (`FFT_simulation/fg/preconditioning.py`), independent of C5 and C6 but
exposed by them. It should be resolved before `reference` is used as a
production knob, and it bears on the interpretation of any result produced
with `preconditioner="reference"`.

### 5.1 Symptom

At `tol_rel=1e-9`, where both runs drive the nonlinear residual to
$\sim 10^{-12}$, the three reference modes converge to **different** states:

| reference | $\bar{P}_{11}$ | $\lVert \delta F - \mathcal{G}(\delta F)\rVert / \lVert \delta F\rVert$ |
|---|---|---|
| mean | 1.0835510581 | 0.640 |
| matrix | 1.0804848613 | 0.646 |
| mid | 1.0837578465 | 0.640 |

The values are stable to nine digits within each mode and do not move as
`tol_rel` is tightened from $10^{-5}$ to $10^{-9}$, so this is *not* a
tolerance effect. The spread in $\bar P_{11}$ is $\approx 2.8\times10^{-3}$
relative.

### 5.2 Cause

$\mathcal{G}$ is the orthogonal projector onto compatible (gradient) fields;
a converged fluctuation field must satisfy $\mathcal{G}(\delta F)=\delta F$.
The last column above shows it does not.

With left preconditioning the GMRES iterates lie in
$\mathrm{span}\{M^{-1}b,\ M^{-1}AM^{-1}b,\dots\}$. The implemented symbol is
$\widehat{\mathcal{G}}\mathbb{K}_0$, whose pseudo-inverse has range
$\mathbb{K}_0^{T}\,\mathrm{range}(\widehat{\mathcal{G}})$ — equal to
$\mathrm{range}(\widehat{\mathcal{G}})$ only when $\mathbb{K}_0 \propto I$.
For a finite-strain hyperelastic tangent it never is, so $M^{-1}$ carries the
iterates *out* of the compatible subspace, and the incompatible content is
invisible to the $R_F$ block (which is itself $\mathcal{G}$-projected) while
still contributing to the volume-averaged stress.

Verified directly and algebraically — feeding a compatible field through the
preconditioner:

```
input                                        incompatible content = 3.6e-16
K_ref isotropic  -> M^-1 output              incompatible content = 3.9e-16
K_ref anisotropic-> M^-1 output              incompatible content = 8.2e-01
```

This is the concrete form of the rank-deficiency warning already recorded in
`docs/green_reference_preconditioning.md` (§"Important Caveats"), and it is
the reason Lucarini & Segurado (2019) advocate a displacement-based unknown,
which yields a full-rank system.

### 5.3 Candidate fix (tested, not applied)

Building the symbol as the *restriction* of the operator to the compatible
subspace, $\widehat{\mathcal{G}}\,\mathbb{K}_0\,\widehat{\mathcal{G}}$
instead of $\widehat{\mathcal{G}}\,\mathbb{K}_0$, makes the pseudo-inverse's
range a subset of $\mathrm{range}(\widehat{\mathcal{G}})$, because
$\widehat{\mathcal{G}}$ is a symmetric projector. Measured on the same probe:

```
current  G*K    -> output incompatible = 8.2e-01
fixed    G*K*G  -> output incompatible = 4.9e-13
```

This has **not** been applied, because it changes every result previously
produced with `preconditioner="reference"` and that is a call for the project
owner. Until it is resolved, `reference="mean"` is the only mode with an
established results history, which is why it remains the default.

---

## 6. Configuration

YAML (`FFT_simulation/Run_configs/*.yaml`), under `defaults.solver` or per case:

```yaml
solver:
  reference: mean              # mean | matrix | mid
  forcing: eisenstat_walker    # eisenstat_walker | fixed
  inner_rtol: 1.0e-6           # used only when forcing: fixed
  eta_max: 1.0e-2              # loosest inner tolerance
  eta_min: 1.0e-3              # tightest inner tolerance
```

Python:

```python
prob.calculate(incre_list=[0.1]*10, preconditioner="reference",
               reference="mean", forcing="eisenstat_walker",
               eta_max=1.e-2, eta_min=1.e-3)
```

To reproduce pre-change runs exactly: `forcing="fixed"`, `inner_rtol=1.e-6`,
`reference="mean"`.

`solver_stats.json` records `reference`, `forcing`, `inner_rtol`, `eta_min`,
`eta_max`, and the realised `forcing_terms` per sub-increment.

The Eisenstat–Walker exponents themselves, `ew_gamma` ($\gamma = 0.9$) and
`ew_alpha` ($\alpha = 2$), are arguments of `calculate()` but are not exposed
in the YAML schema — they are held at their standard literature values, and
$\eta_{\min}$/$\eta_{\max}$ are the knobs that actually govern the cost. Pass
them directly to `calculate()` if a sensitivity study needs them.

---

## 7. Scope

C5 is implemented in the mixed solver (`FFT_simulation/fg/mxfft.py`) only. The standard
solver (`FFT_simulation/fg/fft.py`) still declares convergence on the step norm
$\lVert\Delta F\rVert/\lVert F\rVert$ rather than on the true residual
(diagnosis D4), so it has no residual sequence from which to form a forcing
term; adding one there requires C2 to be ported first. C6 is implemented in
both.

---

## 8. References

- Dembo, R., Eisenstat, S., Steihaug, T. (1982). *Inexact Newton methods.*
  SIAM J. Numer. Anal. 19(2), 400–408.
- Eisenstat, S., Walker, H. (1996). *Choosing the forcing terms in an inexact
  Newton method.* SIAM J. Sci. Comput. 17(1), 16–32.
- Pernice, M., Walker, H. (1998). *NITSOL: a Newton iterative solver for
  nonlinear systems.* SIAM J. Sci. Comput. 19(1), 302–318.
- Kelley, C. T. (1995). *Iterative Methods for Linear and Nonlinear
  Equations.* SIAM.
- Kabel, M., Böhlke, T., Schneider, M. (2014). *Efficient fixed point and
  Newton–Krylov solvers for FFT-based homogenization of elasticity at large
  deformations.* Comput. Mech. 54, 1497–1514.
- Gélébart, L., Mondon-Cancel, R. (2013). *Non-linear extension of FFT-based
  methods accelerated by conjugate gradients.* Comput. Mater. Sci. 77,
  430–439.
- Moulinec, H., Suquet, P. (1998). *A numerical method for computing the
  overall response of nonlinear composites with complex microstructure.*
  CMAME 157, 69–94.
- Zeman, J., Vondřejc, J., Novák, J., Marek, I. (2010). *Accelerating a
  FFT-based solver for numerical homogenization of periodic media by
  conjugate gradients.* J. Comput. Phys. 229, 8065–8071.
- Vondřejc, J., Zeman, J., Marek, I. (2014). *An FFT-based Galerkin method
  for homogenization of periodic media.* Comput. Math. Appl. 68, 156–173.
- Schneider, M. (2021). *A review of nonlinear FFT-based computational
  homogenization methods.* Acta Mech. 232, 2051–2100.
- Lucarini, S., Segurado, J. (2019). *DBFFT: A displacement based FFT
  approach for non-linear homogenization of the mechanical behavior.*
  Int. J. Eng. Sci. 144, 103131.
