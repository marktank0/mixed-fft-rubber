# -*- coding: utf-8 -*-
"""
Created on Fri May  7 04:23:55 2021

@author: WANG Mingchuan

incompressible
"""

import numpy as np
#
#
def umat_field(f, yl, parameters, need_tangent=True):
    """incompressible Neo-Hookean, batched over voxels
        f -- deformation gradients, shape (m,3,3)
        yl -- pressures, shape (m,)
        need_tangent -- skip the tangent (returned as None) when False
    returns P (m,3,3), K4 (m,3,3,3,3) or None, JFmT (m,3,3), kappa_inv (float)
    """
    young = parameters[0]
    poisson = parameters[1]
    #
    C1 = young/4./(1+poisson)
    #
    if poisson == 0.5:
        kappa_inv = 0.0
    else:
        D  = young/3./(1-2*poisson)
        kappa_inv = 1./D
    #
    J = np.linalg.det(f)
    J23 = J**(-2/3)
    finv = np.linalg.inv(f)
    finvT = np.swapaxes(finv, -1, -2)
    I1 = np.einsum("vij,vij->v", f, f)
    #
    P = (2*C1*J23)[:, None, None]*f \
        - (2*C1/3*J23*I1)[:, None, None]*finvT \
        + (yl*J)[:, None, None]*finvT
    #
    JFmT = J[:, None, None]*finvT
    #
    if not need_tangent:
        return P, None, JFmT, kappa_inv
    #
    # per-voxel blocks (batched forms of the k01..k07 blocks below):
    # k01[i,j,m,n] = f[i,j]  finv[n,m]
    # k02[i,j,m,n] = d(i,m) d(j,n)
    # k03[i,j,m,n] = finvT[i,j] finvT[m,n]      (= k06)
    # k04[i,j,m,n] = finvT[i,j] f[m,n]
    # k05[i,j,m,n] = finvT[m,j] finvT[i,n]      (= k07)
    i = np.eye(3)
    k02 = np.einsum("im,jn->ijmn", i, i)
    #
    c01 = -4/3*C1*J23
    c02 = 2*C1*J23
    c03 = 4*C1/9*J23*I1 + yl*J
    c04 = -4/3*C1*J23
    c05 = 2*C1/3*J23*I1 - yl*J
    #
    K4 = np.einsum("v,vij,vnm->vijmn", c01, f, finv)
    K4 += c02[:, None, None, None, None]*k02[None]
    K4 += np.einsum("v,vij,vmn->vijmn", c03, finvT, finvT)
    K4 += np.einsum("v,vij,vmn->vijmn", c04, finvT, f)
    K4 += np.einsum("v,vmj,vin->vijmn", c05, finvT, finvT)
    #
    return P, K4, JFmT, kappa_inv


def umat(f,yl,parameters):
    """incompressible Neo-Hookean
        f -- deformation gradient
        yl -- pressure
    """
    P, K4, JFmT, kappa_inv = umat_field(
        np.asarray(f)[None, :, :], np.array([yl], dtype=float), parameters,
    )
    return P[0], K4[0], JFmT[0], kappa_inv
