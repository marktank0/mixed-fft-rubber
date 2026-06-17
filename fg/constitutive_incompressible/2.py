# -*- coding: utf-8 -*-
"""
Created on Mon June 15 21:23:55 2026

@author: Mark Tankink

incompressible
"""

import itertools

import numpy as np


delta = lambda i,j: float(i==j)


def umat(f,yl,parameters):
    """Incompressible Mooney-Rivlin.

    parameters[0] -- Young's modulus used to set the small-strain shear modulus
    parameters[1] -- Poisson ratio used for the pressure regularization
    parameters[2] -- gamma split between C10 and C01; gamma=0 gives Neo-Hookean
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
    finvT = finv.T

    C = f.T @ f
    B = f @ f.T
    FC = f @ C
    I1 = np.einsum("ij,ij->", f, f)
    I2 = 0.5*(I1*I1 - np.einsum("ij,ji->", C, C))

    p1 = 2.0*C10*J23*(f - I1*finvT/3.0)
    M = I1*f - FC - 2.0*I2*finvT/3.0
    p2 = 2.0*C01*J43*M
    p3x3 = p1 + p2 + yl*J*finvT

    i = np.eye(3)
    k01 = np.einsum("ij,nm->ijmn", f, finv)
    k02 = np.einsum("im,jn->ijmn", i, i)
    k03 = np.einsum("ji,nm->ijmn", finv, finv)
    k04 = np.einsum("ji,mn->ijmn", finv, f)
    k05 = np.einsum("jm,ni->ijmn", finv, finv)
    k06 = k03
    k07 = k05

    k_neohookean = (
        -4.0*C10*J23*k01/3.0
        + 2.0*C10*J23*k02
        + 4.0*C10*J23*I1*k03/9.0
        - 4.0*C10*J23*k04/3.0
        + 2.0*C10*J23*I1*k05/3.0
    )

    k_mooney = np.zeros((3,3,3,3))
    for a,b,m,n in itertools.product(range(3), repeat=4):
        dM = (
            2.0*f[m,n]*f[a,b]
            + I1*delta(a,m)*delta(b,n)
            - delta(a,m)*C[n,b]
            - f[a,n]*f[m,b]
            - delta(b,n)*B[a,m]
            - 4.0*(I1*f[m,n] - FC[m,n])*finvT[a,b]/3.0
            + 2.0*I2*finvT[m,b]*finvT[a,n]/3.0
        )
        k_mooney[a,b,m,n] = 2.0*C01*J43*(dM - 4.0*finvT[m,n]*M[a,b]/3.0)

    k_pressure = yl*J*k06 - yl*J*k07
    k3x3x3x3 = k_neohookean + k_mooney + k_pressure

    JFmT = J*finvT
    return p3x3, k3x3x3x3, JFmT, kappa_inv
