# -*- coding: utf-8 -*-
"""
Created on Mon June 15 21:23:55 2026

@author: Mark Tankink

incompressible Mooney-Rivlin
"""

import numpy as np


def umat_field(f, yl, parameters, need_tangent=True):
    """Incompressible Mooney-Rivlin, batched over voxels.

    f -- deformation gradients, shape (m,3,3)
    yl -- pressures, shape (m,)
    parameters[0] -- Young's modulus used to set the small-strain shear modulus
    parameters[1] -- Poisson ratio used for the pressure regularization
    parameters[2] -- gamma split between C10 and C01; gamma=0 gives Neo-Hookean
    need_tangent -- skip the tangent (returned as None) when False

    returns P (m,3,3), K4 (m,3,3,3,3) or None, JFmT (m,3,3), kappa_inv (float)
    """
    young = parameters[0]
    poisson = parameters[1]
    gamma = parameters[2] if len(parameters) > 2 else 0.5

    mu = young/(2.0*(1.0 + poisson))
    C10 = (1.0 - gamma)*mu/2.0
    C01 = gamma*mu/2.0

    if poisson == 0.5:
        kappa_inv = 0.0
    else:
        D = young/3.0/(1.0 - 2.0*poisson)
        kappa_inv = 1.0/D

    J = np.linalg.det(f)
    J23 = J**(-2.0/3.0)
    J43 = J**(-4.0/3.0)
    finv = np.linalg.inv(f)
    finvT = np.swapaxes(finv, -1, -2)
    fT = np.swapaxes(f, -1, -2)

    C = fT @ f
    B = f @ fT
    FC = f @ C
    I1 = np.einsum("vij,vij->v", f, f)
    I2 = 0.5*(I1*I1 - np.einsum("vij,vji->v", C, C))

    p1 = (2.0*C10*J23)[:, None, None]*f - (2.0*C10*J23*I1/3.0)[:, None, None]*finvT
    M = I1[:, None, None]*f - FC - (2.0*I2/3.0)[:, None, None]*finvT
    p2 = (2.0*C01*J43)[:, None, None]*M
    P = p1 + p2 + (yl*J)[:, None, None]*finvT

    JFmT = J[:, None, None]*finvT

    if not need_tangent:
        return P, None, JFmT, kappa_inv

    # Neo-Hookean part plus pressure part, same blocks as in 1.py:
    # k01[i,j,m,n] = f[i,j] finv[n,m]
    # k02[i,j,m,n] = d(i,m) d(j,n)
    # k03[i,j,m,n] = finvT[i,j] finvT[m,n]      (= k06)
    # k04[i,j,m,n] = finvT[i,j] f[m,n]
    # k05[i,j,m,n] = finvT[m,j] finvT[i,n]      (= k07)
    i = np.eye(3)
    k02 = np.einsum("im,jn->ijmn", i, i)

    c01 = -4.0*C10*J23/3.0
    c02 = 2.0*C10*J23
    c03 = 4.0*C10*J23*I1/9.0 + yl*J
    c04 = -4.0*C10*J23/3.0
    c05 = 2.0*C10*J23*I1/3.0 - yl*J

    K4 = np.einsum("v,vij,vnm->vijmn", c01, f, finv)
    K4 += c02[:, None, None, None, None]*k02[None]
    K4 += np.einsum("v,vij,vmn->vijmn", c03, finvT, finvT)
    K4 += np.einsum("v,vij,vmn->vijmn", c04, finvT, f)
    K4 += np.einsum("v,vmj,vin->vijmn", c05, finvT, finvT)

    # Mooney (I2) part, batched form of the per-voxel dM expression:
    # dM[a,b,m,n] = 2 f[m,n] f[a,b] + I1 d(a,m) d(b,n) - d(a,m) C[n,b]
    #               - f[a,n] f[m,b] - d(b,n) B[a,m]
    #               - 4/3 (I1 f - FC)[m,n] finvT[a,b]
    #               + 2/3 I2 finvT[m,b] finvT[a,n]
    G1 = I1[:, None, None]*f - FC
    dM = 2.0*np.einsum("vmn,vab->vabmn", f, f)
    dM += np.einsum("v,am,bn->vabmn", I1, i, i)
    dM -= np.einsum("am,vnb->vabmn", i, C)
    dM -= np.einsum("van,vmb->vabmn", f, f)
    dM -= np.einsum("bn,vam->vabmn", i, B)
    dM -= (4.0/3.0)*np.einsum("vmn,vab->vabmn", G1, finvT)
    dM += (2.0/3.0)*np.einsum("v,vmb,van->vabmn", I2, finvT, finvT)

    dM -= (4.0/3.0)*np.einsum("vmn,vab->vabmn", finvT, M)
    K4 += (2.0*C01*J43)[:, None, None, None, None]*dM

    return P, K4, JFmT, kappa_inv


def umat(f, yl, parameters):
    """Incompressible Mooney-Rivlin, single voxel (see umat_field)."""
    P, K4, JFmT, kappa_inv = umat_field(
        np.asarray(f)[None, :, :], np.array([yl], dtype=float), parameters,
    )
    return P[0], K4[0], JFmT[0], kappa_inv
