# -*- coding: utf-8 -*-
"""Reference-operator preconditioners for FFT-Galerkin linear solves."""

import os

import numpy as np
import scipy.fft

_AXES = (1, 2, 3)
_FFT_WORKERS = int(os.environ.get("FFT_WORKERS", "1"))

REFERENCE_MODES = ("mean", "matrix", "mid")


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


def build_standard_reference_symbol(Ghat4, K_ref, rcond=1.0e-10, zero_mode_free_components=None):
    """Build the Fourier-space pseudo-inverse of Ghat:K_ref."""
    N = Ghat4.shape[-1]
    G9 = Ghat4.reshape(9, 9, N, N, N)
    K9 = K_ref.reshape(9, 9)
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
):
    """Build the Fourier-space pseudo-inverse of the mixed reference block."""
    N = Ghat4.shape[-1]
    G9 = Ghat4.reshape(9, 9, N, N, N)
    K9 = K_ref.reshape(9, 9)
    J9 = J_ref.reshape(9)

    symbol = np.zeros((10, 10, N, N, N), dtype=float)
    symbol[:9, :9] = np.einsum("abxyz,bc->acxyz", G9, K9)
    symbol[:9, 9] = np.einsum("abxyz,b->axyz", G9, J9)
    symbol[9, :9] = J9[:, None, None, None]
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
