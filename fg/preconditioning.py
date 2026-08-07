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


def apply_mixed_reference_preconditioner(vec, inv_symbol):
    """Apply the mixed-solver reference preconditioner to a flat vector."""
    N = inv_symbol.shape[-1]
    field = vec.reshape(10, N, N, N)
    field_hat = _fft_field(field)
    result_hat = np.einsum("abxyz,bxyz->axyz", inv_symbol, field_hat)
    return _ifft_field(result_hat).reshape(-1)
