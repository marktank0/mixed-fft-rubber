# -*- coding: utf-8 -*-
"""Reference-operator preconditioners for FFT-Galerkin linear solves."""

import numpy as np


def _fft_field(field):
    shape = field.shape[1:]
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(field), shape, axes=(1, 2, 3)))


def _ifft_field(field_hat):
    shape = field_hat.shape[1:]
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(field_hat), shape, axes=(1, 2, 3))).real


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
