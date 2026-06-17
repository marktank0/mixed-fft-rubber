# Incompressible Mooney-Rivlin UMAT Plan

This note derives the incompressible Mooney-Rivlin model in the exact form needed by `fg/constitutive_incompressible/2.py`.

The current mixed FFT solver expects each incompressible constitutive file to expose:

```python
def umat(f, yl, parameters):
    return p3x3, k3x3x3x3, JFmT, kappa_inv
```

where:

- `f` is the local deformation gradient \(F\),
- `yl` is the pressure-like mixed variable \(p\),
- `p3x3` is the first Piola-Kirchhoff stress \(P\),
- `k3x3x3x3` is the tangent \(\partial P_{ij}/\partial F_{mn}\),
- `JFmT` is \(J F^{-T}\), the derivative of \(J\) with respect to \(F\),
- `kappa_inv` is \(1/\kappa\), used in the mixed pressure equation.

## Literature Base

The Mooney-Rivlin model is a classical two-invariant hyperelastic model for rubber-like materials. The original model traces back to Mooney's large-deformation rubber elasticity work and Rivlin's invariant formulation. In the modern notation used by finite-strain implementations, the incompressible or nearly incompressible form is usually written as an isochoric invariant model:

$$
W_\mathrm{iso}
=
C_{10}(\bar I_1 - 3)
+
C_{01}(\bar I_2 - 3).
$$

Useful sources for this implementation:

- Mooney, "A theory of large elastic deformation" (1940), original phenomenological large-deformation rubber model.
- Rivlin, "Large elastic deformations of isotropic materials. IV. Further developments of the general theory" (1948), invariant-based finite elasticity foundation.
- Cipolatti, Liu, and Rincon, "Mathematical analysis of successive linear approximation for Mooney-Rivlin material model in finite elasticity" (2011), discusses finite-elasticity boundary value problems for nearly incompressible Mooney-Rivlin materials.
- Shojaei and Yavari, "Compatible-Strain Mixed Finite Element Methods for 3D Compressible and Incompressible Nonlinear Elasticity" (2019), gives a mixed incompressible nonlinear-elasticity setting with deformation gradient, first Piola-Kirchhoff stress, and pressure-like fields. That is close in spirit to this repository's mixed FFT formulation.
- Shontz and Vavasis, "A Robust Solution Procedure for Hyperelastic Solids with Large Boundary Deformation" (2006), uses compressible Mooney-Rivlin finite-strain solids in Newton-type large-deformation computations.

The important implementation point is that Mooney-Rivlin adds an \(I_2\) term to Neo-Hookean. If \(C_{01}=0\), the model reduces to the existing Neo-Hookean form in `1.py`.

## Kinematics

Use:

$$
F_{iJ}, \qquad
J = \det F, \qquad
F^{-T}_{iJ} = F^{-1}_{J i}.
$$

The right Cauchy-Green tensor is:

$$
C = F^T F,
\qquad
C_{IJ}=F_{kI}F_{kJ}.
$$

The invariants are:

$$
I_1 = \mathrm{tr}(C) = F:F,
$$

$$
I_2
=
\frac{1}{2}\left(I_1^2-\mathrm{tr}(C^2)\right).
$$

The isochoric invariants are:

$$
\bar I_1 = J^{-2/3} I_1,
\qquad
\bar I_2 = J^{-4/3} I_2.
$$

## Energy

For the mixed incompressible solver we use the same split as the current Neo-Hookean file:

$$
W(F,p)
=
C_{10}(\bar I_1 - 3)
+
C_{01}(\bar I_2 - 3)
+
p(J-1),
$$

with optional near-incompressible regularization through \(1/\kappa\) in the pressure equation, exactly like `1.py`.

The pressure sign follows the current implementation:

$$
P = P_\mathrm{iso} + p J F^{-T}.
$$

## Parameter Convention

To keep the charge-file layout compatible with the rest of the repository, model `2.py` should keep:

```text
model_number, E, poisson, gamma, ...
```

where:

- `E = parameters[0]`,
- `poisson = parameters[1]`,
- `gamma = parameters[2]` if present, otherwise a default such as `0.5`.

The small-strain shear modulus is:

$$
\mu = \frac{E}{2(1+\nu)}.
$$

For Mooney-Rivlin:

$$
\mu = 2(C_{10}+C_{01}).
$$

So a convenient split is:

$$
C_{10} = (1-\gamma)\frac{\mu}{2},
\qquad
C_{01} = \gamma\frac{\mu}{2}.
$$

Equivalently:

$$
C_{10} = (1-\gamma)\frac{E}{4(1+\nu)},
\qquad
C_{01} = \gamma\frac{E}{4(1+\nu)}.
$$

This has two useful properties:

- the first two charge parameters still mean `E` and `poisson`,
- setting \(\gamma=0\) gives the existing incompressible Neo-Hookean deviatoric stress.

As in `1.py`:

$$
\kappa^{-1}
=
\begin{cases}
0, & \nu = 0.5,\\
\left(\dfrac{E}{3(1-2\nu)}\right)^{-1}, & \nu \ne 0.5.
\end{cases}
$$

## First Piola-Kirchhoff Stress

The Neo-Hookean part is already in `1.py`:

$$
P^{(1)}_{iJ}
=
2 C_{10} J^{-2/3}
\left(
F_{iJ}
-
\frac{1}{3} I_1 F^{-T}_{iJ}
\right).
$$

The Mooney-Rivlin \(I_2\) part is:

$$
P^{(2)}_{iJ}
=
2 C_{01} J^{-4/3}
\left[
I_1 F_{iJ}
-
(FC)_{iJ}
-
\frac{2}{3} I_2 F^{-T}_{iJ}
\right].
$$

The total mixed stress is:

$$
P_{iJ}
=
P^{(1)}_{iJ}
+
P^{(2)}_{iJ}
+
p J F^{-T}_{iJ}.
$$

This is the value returned as `p3x3`.

## Tangent for the Neo-Hookean Part

The existing `1.py` tangent can be reused structurally with \(C_1\) replaced by \(C_{10}\):

$$
\mathbb{K}^{(1)}_{iJmN}
=
\frac{\partial P^{(1)}_{iJ}}{\partial F_{mN}}.
$$

In code terms, this is the current block:

```python
-4/3*C10*J23*k01
+2*C10*J23*k02
+4*C10/9*J23*I1*k03
-4*C10/3*J23*k04
+2*C10/3*J23*I1*k05
```

plus the pressure tangent:

```python
yl*J*k06 - yl*J*k07
```

where the `k0*` arrays are already defined in `1.py`.

## Tangent for the \(I_2\) Part

Define:

$$
q = J^{-4/3},
$$

and

$$
M_{iJ}
=
I_1 F_{iJ}
-
(FC)_{iJ}
-
\frac{2}{3}I_2 F^{-T}_{iJ}.
$$

Then:

$$
P^{(2)}_{iJ}
=
2 C_{01} q M_{iJ}.
$$

The tangent is:

$$
\mathbb{K}^{(2)}_{iJmN}
=
2C_{01}q
\left[
\frac{\partial M_{iJ}}{\partial F_{mN}}
-
\frac{4}{3}F^{-T}_{mN}M_{iJ}
\right].
$$

The derivative of \(M\) is:

$$
\frac{\partial M_{iJ}}{\partial F_{mN}}
=
2F_{mN}F_{iJ}
+
I_1\delta_{im}\delta_{JN}
-
\delta_{im}C_{NJ}
-
F_{iN}F_{mJ}
-
\delta_{JN}(FF^T)_{im}
$$

$$
\qquad
-
\frac{4}{3}
\left[
I_1F_{mN}-(FC)_{mN}
\right]
F^{-T}_{iJ}
+
\frac{2}{3}I_2
F^{-T}_{mJ}F^{-T}_{iN}.
$$

This is the main new analytic block needed in `2.py`.

## Pressure Coupling

The pressure part remains exactly the same as in `1.py`:

$$
P^\mathrm{pressure}_{iJ} = p J F^{-T}_{iJ}.
$$

The derivative with respect to pressure is:

$$
\frac{\partial P_{iJ}}{\partial p}
=
J F^{-T}_{iJ}.
$$

So `JFmT` remains:

```python
JFmT = J*finv.T
```

The derivative with respect to \(F\) is also the current pressure tangent:

$$
pJ
\left(
F^{-T}_{mN}F^{-T}_{iJ}
-
F^{-T}_{mJ}F^{-T}_{iN}
\right),
$$

which is already encoded in `1.py` as:

```python
yl*J*k06 - yl*J*k07
```

## Implementation Sketch

The `2.py` implementation does the following:

1. read `young`, `poisson`, and optional `gamma`;
2. compute:

   $$
   C_{10}=(1-\gamma)\frac{E}{4(1+\nu)},\qquad
   C_{01}=\gamma\frac{E}{4(1+\nu)};
   $$

3. compute \(J\), \(F^{-1}\), \(F^{-T}\), \(C\), \(FF^T\), \(I_1\), \(I_2\);
4. compute \(P^{(1)}\), \(P^{(2)}\), and pressure stress;
5. compute \(\mathbb{K}^{(1)}\), \(\mathbb{K}^{(2)}\), and pressure tangent;
6. return:

```python
return p3x3, k3x3x3x3, JFmT, kappa_inv
```

## Validation Result

The implementation was checked in two ways:

1. With \(\gamma=0\), `2.py` reduces to `1.py`:

```text
stress diff 5.249e-16
tangent diff 2.778e-15
JFmT diff 0.0
kappa diff 0.0
```

2. With \(\gamma=0.25\), the analytic tangent was compared against centered finite differences:

```text
finite-difference tangent absolute error 2.412e-09
finite-difference tangent relative error 1.665e-10
max absolute component error 1.064e-09
```

These checks verify the local constitutive implementation. A tiny mixed FFT solve is still recommended before large batch usage.

## Validation Plan

Before trusting the model in full FFT runs:

1. Set \(\gamma=0\) and compare `2.py` against `1.py` for random positive-determinant \(F\). Stress and tangent should match to roundoff.
2. Finite-difference check the tangent:

   $$
   \mathbb{K}_{iJmN}
   \approx
   \frac{P_{iJ}(F+\epsilon E^{mN})-P_{iJ}(F-\epsilon E^{mN})}{2\epsilon}.
   $$

3. Check uniaxial incompressible response against the known Mooney-Rivlin engineering stress:

   $$
   P_{11}^{eng}
   =
   \left(2C_{10}+\frac{2C_{01}}{\lambda}\right)
   \left(\lambda-\lambda^{-2}\right),
   $$

   for \(F=\mathrm{diag}(\lambda,\lambda^{-1/2},\lambda^{-1/2})\) and transverse traction-free pressure.
4. Run one tiny \(3^3\) mixed solve before using it in the large batch runner.

## Expected Effect Compared With Neo-Hookean

Mooney-Rivlin adds curvature through the second invariant. It may fit rubber-like response over larger strain ranges than Neo-Hookean, but it also makes the tangent more complex. Numerically, it is not guaranteed to converge more easily. If \(C_{01}\) is moderate and the tangent is implemented consistently, it should behave similarly to Neo-Hookean. If \(C_{01}\) is too large or the tangent is inconsistent, Newton/GMRES may become less stable.

## Sources used

- Cipolatti, Liu, Rincon 2011; https://arxiv.org/abs/1107.2705
- Shojaei and Yavari 2019; https://arxiv.org/abs/1906.10741
- Angoshtari 2019; https://arxiv.org/abs/1910.13485 
- Shontz and Vavasis 2006; https://arxiv.org/abs/cs/0609001

