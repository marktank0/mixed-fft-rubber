# -*- coding: utf-8 -*-
"""Reference-operator preconditioners for FFT-Galerkin linear solves."""

import os

import numpy as np
import scipy.fft

_AXES = (1, 2, 3)
_FFT_WORKERS = int(os.environ.get("FFT_WORKERS", "1"))

REFERENCE_MODES = ("mean", "matrix", "mid")
DISCRETIZATIONS = ("fourier", "willot")


def _wave_vectors(N, ndim, discretization):
    """Derivative symbol xi_j of the chosen discretization, on the centred grid.

    "fourier" - the exact trigonometric derivative. The projector only ever
        uses xi_j xi_m*/|xi|^2, so the raw (real) wave numbers k_j stand in for
        i*k_j: the factors of i cancel. This is the Moulinec-Suquet /
        FFT-Galerkin spectral discretization, and is returned as a real array
        so the default path is arithmetically identical to before.

    "willot" - Willot (2015) rotated scheme: a forward finite difference taken
        along the voxel diagonal, averaged over the transverse directions,

            xi_j  ~  (e^{i k_j h} - 1) * prod_{m != j} (e^{i k_m h} + 1),

        with h = 1/N the voxel size. The 1/(4h)^1 prefactor and any half-voxel
        centring phase are omitted: both are a common real scale / common phase
        across all components, and both cancel identically in
        xi_j xi_m*/|xi|^2. Expanding for small k reproduces i*k_j, so the
        scheme is consistent.
    """
    if discretization not in DISCRETIZATIONS:
        raise ValueError(
            "Unknown discretization {!r}; use one of {}.".format(discretization, DISCRETIZATIONS)
        )

    k = np.arange(-(N-1)/2., +(N+1)/2.)
    grid = np.stack(np.meshgrid(*([k]*ndim), indexing="ij"))

    if discretization == "fourier":
        return grid

    phase = np.exp(2j*np.pi*grid/float(N))          # e^{i k_j h}, h = 1/N
    xi = np.empty(grid.shape, dtype=complex)
    for j in range(ndim):
        term = phase[j] - 1.0
        for m in range(ndim):
            if m != j:
                term = term*(phase[m] + 1.0)
        xi[j] = term
    return xi


def build_Ghat4(N, stress_control, ndim=3, discretization="fourier"):
    """Green (compatibility) projection symbol for the chosen discretization.

    Ghat_ijlm = delta_il * xi_j * conj(xi_m) / |xi|^2, which projects onto
    fields that are gradients *with respect to the same derivative symbol*.
    It is Hermitian and idempotent for any xi, and satisfies
    Ghat(-k) = conj(Ghat(k)), so G(real field) stays real.
    """
    xi = _wave_vectors(N, ndim, discretization)
    xi_c = np.conj(xi)                              # a no-op for the real "fourier" grid
    xi2 = np.real(np.einsum("k...,k...->...", xi, xi_c))

    zero_freq = xi2 <= 1.0e-12*max(float(xi2.max()), 1.0)
    xi2safe = np.where(zero_freq, 1.0, xi2)
    QQ = np.einsum("j...,m...->jm...", xi, xi_c)/xi2safe
    Ghat4 = np.einsum("il,jm...->ijlm...", np.eye(ndim), QQ)

    Ghat4[:, :, :, :, zero_freq] = 0.0
    for (i, j) in stress_control:                   # zero freq. -> mean
        Ghat4[i, j, i, j, zero_freq] = 1.0
    return Ghat4


def reference_average(field, mode="mean", matrix_mask=None, filler_mask=None):
    """Homogeneous reference value of a per-voxel field.

    The grid occupies the trailing three axes, so this works unchanged for
    the tangent ``K4`` (3,3,3,3,N,N,N), ``JFmT`` (3,3,N,N,N) and the scalar
    ``kappa_inv`` (N,N,N).

    mode:
      "mean"   volume (voxel) average over the whole cell - the classical
               choice, but at high contrast it is dominated by the stiff
               phase even at low filler fractions.
      "matrix" average over matrix voxels only, so the preconditioner is
               near-exact on the majority phase.
      "mid"    unweighted average of the two phase averages; the
               finite-strain analogue of the (C_min + C_max)/2 reference.

    Empty phase masks are skipped, so a single-phase (unfilled) cell falls
    back to the whole-cell average under every mode.
    """
    field = np.asarray(field)
    grid_axes = (-3, -2, -1)

    if mode == "mean":
        return field.mean(axis=grid_axes)

    if mode not in REFERENCE_MODES:
        raise ValueError(
            "Unknown reference {!r}; use one of {}.".format(mode, REFERENCE_MODES)
        )

    masks = (matrix_mask,) if mode == "matrix" else (matrix_mask, filler_mask)
    phase_means = [
        field[..., mask].mean(axis=-1)
        for mask in masks
        if mask is not None and mask.any()
    ]
    if not phase_means:
        return field.mean(axis=grid_axes)
    return sum(phase_means)/len(phase_means)


def _fft_field(field):
    return np.fft.fftshift(
        scipy.fft.fftn(np.fft.ifftshift(field, axes=_AXES), axes=_AXES, workers=_FFT_WORKERS),
        axes=_AXES,
    )


def _ifft_field(field_hat):
    return np.fft.fftshift(
        scipy.fft.ifftn(np.fft.ifftshift(field_hat, axes=_AXES), axes=_AXES, workers=_FFT_WORKERS),
        axes=_AXES,
    ).real


def _pinv_symbol(symbol, rcond):
    n_comp = symbol.shape[0]
    spatial_shape = symbol.shape[2:]
    symbol_stack = np.moveaxis(symbol, (0, 1), (-2, -1)).reshape(-1, n_comp, n_comp)
    pinv_stack = np.linalg.pinv(symbol_stack, rcond=rcond)
    return np.moveaxis(pinv_stack.reshape(*spatial_shape, n_comp, n_comp), (-2, -1), (0, 1))


def _constrain_zero_mode(inv_symbol, symbol, free_components, rcond):
    if free_components is None:
        return inv_symbol

    center = inv_symbol.shape[-1]//2
    constrained = np.zeros_like(inv_symbol[:, :, center, center, center])
    free = np.array(free_components, dtype=int)
    if free.size:
        zero_symbol = symbol[:, :, center, center, center]
        free_symbol = zero_symbol[np.ix_(free, free)]
        constrained[np.ix_(free, free)] = np.linalg.pinv(free_symbol, rcond=rcond)
    inv_symbol[:, :, center, center, center] = constrained
    return inv_symbol


def build_standard_reference_symbol(Ghat4, K_ref, rcond=1.0e-10, zero_mode_free_components=None,
                                    restrict_to_compatible=True):
    """Build the Fourier-space pseudo-inverse of the reference operator.

    With restrict_to_compatible (the default) the symbol is the reference
    operator *restricted* to the compatible subspace, Ghat K_ref Ghat. Because
    Ghat is a symmetric projector this makes range(symbol^+) a subset of
    range(Ghat), so the preconditioner cannot carry Krylov iterates out of the
    compatible subspace. See docs/green_reference_preconditioning.md.

    restrict_to_compatible=False restores the previous Ghat K_ref symbol and
    exists only to reproduce results produced before the fix.
    """
    N = Ghat4.shape[-1]
    G9 = Ghat4.reshape(9, 9, N, N, N)
    K9 = K_ref.reshape(9, 9)
    if restrict_to_compatible:
        symbol = np.einsum("abxyz,bc,cdxyz->adxyz", G9, K9, G9)
    else:
        symbol = np.einsum("abxyz,bc->acxyz", G9, K9)
    inv_symbol = _pinv_symbol(symbol, rcond)
    return _constrain_zero_mode(inv_symbol, symbol, zero_mode_free_components, rcond)


def apply_standard_reference_preconditioner(vec, inv_symbol):
    """Apply the standard-solver reference preconditioner to a flat vector."""
    N = inv_symbol.shape[-1]
    field = vec.reshape(9, N, N, N)
    field_hat = _fft_field(field)
    result_hat = np.einsum("abxyz,bxyz->axyz", inv_symbol, field_hat)
    return _ifft_field(result_hat).reshape(-1)


def build_mixed_reference_symbol(
    Ghat4,
    K_ref,
    J_ref,
    kappa_inv_ref,
    rcond=1.0e-10,
    zero_mode_free_components=None,
    restrict_to_compatible=True,
):
    """Build the Fourier-space pseudo-inverse of the mixed reference block.

    With restrict_to_compatible (the default) the symbol is the reference
    operator restricted to the compatible subspace, Pi A0 Pi with
    Pi = diag(Ghat, I): the deformation block is sandwiched as Ghat K0 Ghat and
    the pressure row sees only the compatible part of the deformation block.
    range(symbol^+) is then contained in range(Pi), so the preconditioner keeps
    Krylov iterates inside the subspace on which the operator is nonsingular.
    See docs/green_reference_preconditioning.md.

    restrict_to_compatible=False restores the previous symbol and exists only
    to reproduce results produced before the fix.
    """
    N = Ghat4.shape[-1]
    G9 = Ghat4.reshape(9, 9, N, N, N)
    K9 = K_ref.reshape(9, 9)
    J9 = J_ref.reshape(9)

    # dtype follows Ghat4: complex for discretizations with a complex symbol
    dtype = np.result_type(G9.dtype, K9.dtype, J9.dtype, float)
    symbol = np.zeros((10, 10, N, N, N), dtype=dtype)
    if restrict_to_compatible:
        symbol[:9, :9] = np.einsum("abxyz,bc,cdxyz->adxyz", G9, K9, G9)
        symbol[9, :9] = np.einsum("b,baxyz->axyz", J9, G9)
    else:
        symbol[:9, :9] = np.einsum("abxyz,bc->acxyz", G9, K9)
        symbol[9, :9] = J9[:, None, None, None]
    symbol[:9, 9] = np.einsum("abxyz,b->axyz", G9, J9)
    symbol[9, 9] = -float(kappa_inv_ref)

    inv_symbol = _pinv_symbol(symbol, rcond)
    return _constrain_zero_mode(inv_symbol, symbol, zero_mode_free_components, rcond)


def local_jacobi_scale(K4, floor=1.0e-12):
    """Per-voxel scalar stiffness magnitude d(x): the Jacobi diagonal proxy.

    The operator is matrix-free, so there is no assembled diagonal to extract.
    The Frobenius norm of the local tangent is the natural stand-in: it tracks
    the local stiffness, which is what sets the local-to-reference ratios that
    bound the preconditioned spectrum.
    """
    d = np.sqrt(np.einsum("ijkl...,ijkl...->...", K4, K4))
    dmax = float(d.max())
    return np.maximum(d, floor*dmax if dmax > 0.0 else 1.0)


def build_green_jacobi_symbol(Ghat4, K4_field, JFmT_field, kappa_inv_field, d,
                              reference="mean", matrix_mask=None, filler_mask=None,
                              rcond=1.0e-10, zero_mode_free_components=None):
    """Green-Jacobi: the Green symbol built from the Jacobi-SCALED material.

    Ladecky et al. (arXiv:2508.02613) precondition with

        M^-1 = D^-1/2 G0^-1 D^-1/2,

    where D is the diagonal of the actual system. The point is that the plain
    Green preconditioner's spectrum is bounded by the local ratios K(x)/K0, so
    its condition number grows like the phase contrast; dividing out the local
    stiffness first makes those ratios O(1).

    The reference here is therefore built from the *scaled* tangent K(x)/d(x)
    rather than from K(x), so that it is the reference for the scaled problem.

    IMPORTANT deviation from the published form: that work solves a
    displacement/FE system in which every field is admissible. This solver's
    unknown is the deformation gradient, which must stay compatible, and
    D^-1/2 is a real-space pointwise scaling that does NOT preserve
    compatibility. `apply_green_jacobi_preconditioner` therefore re-projects
    afterwards - see the note there.
    """
    inv_d = 1.0/d
    K_ref = reference_average(K4_field*inv_d, reference, matrix_mask, filler_mask)
    J_ref = reference_average(JFmT_field/np.sqrt(d), reference, matrix_mask, filler_mask)
    kappa_ref = float(reference_average(kappa_inv_field, reference, matrix_mask, filler_mask))
    return build_mixed_reference_symbol(
        Ghat4, K_ref, J_ref, kappa_ref, rcond=rcond,
        zero_mode_free_components=zero_mode_free_components,
        restrict_to_compatible=True,
    )


def apply_green_jacobi_preconditioner(vec, inv_symbol, d, Ghat4):
    """Apply  Pi . D^-1/2 . (Ghat K0~ Ghat)^+ . D^-1/2  to a flat vector.

    The trailing projection Pi is not in the published Green-Jacobi operator.
    It is required here because the pointwise D^-1/2 scaling takes a compatible
    field out of the compatible subspace, and this solver's operator is
    singular off that subspace (docs/green_reference_preconditioning.md). Cost:
    one extra FFT round trip per preconditioner application.
    """
    N = inv_symbol.shape[-1]
    num_up = 9*N**3
    scale = 1.0/np.sqrt(d)

    out = vec.copy()
    F = out[:num_up].reshape(3, 3, N, N, N)*scale
    out[:num_up] = F.reshape(-1)

    field_hat = _fft_field(out.reshape(10, N, N, N))
    result = _ifft_field(np.einsum("abxyz,bxyz->axyz", inv_symbol, field_hat)).reshape(-1)

    F = result[:num_up].reshape(3, 3, N, N, N)*scale
    # re-project onto compatible fields
    Fh = _fft_field_4(F)
    F = _ifft_field_4(np.einsum("ijklxyz,klxyz->ijxyz", Ghat4, Fh))
    result[:num_up] = F.reshape(-1)
    return result


_AXES4 = (2, 3, 4)


def _fft_field_4(field):
    return np.fft.fftshift(
        scipy.fft.fftn(np.fft.ifftshift(field, axes=_AXES4), axes=_AXES4, workers=_FFT_WORKERS),
        axes=_AXES4,
    )


def _ifft_field_4(field_hat):
    return np.fft.fftshift(
        scipy.fft.ifftn(np.fft.ifftshift(field_hat, axes=_AXES4), axes=_AXES4, workers=_FFT_WORKERS),
        axes=_AXES4,
    ).real


def apply_mixed_reference_preconditioner(vec, inv_symbol):
    """Apply the mixed-solver reference preconditioner to a flat vector."""
    N = inv_symbol.shape[-1]
    field = vec.reshape(10, N, N, N)
    field_hat = _fft_field(field)
    result_hat = np.einsum("abxyz,bxyz->axyz", inv_symbol, field_hat)
    return _ifft_field(result_hat).reshape(-1)
