# -*- coding: utf-8 -*-
"""
Created on Fri May  7 04:23:55 2021

@author: WANG Mingchuan

incompressible
"""

import numpy as np
#
#
def umat(f,yl,parameters):
    """incompressible Neo-Hookean
        f -- deformation gradient
        yl -- pressure
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
    I1 = np.einsum("ij,ij->",f,f)
    #
    p3x3 = 2*C1*J**(-2/3)*f - 2*C1/3*J**(-2/3)*I1*finv.T+yl*J*finv.T
    #
    i = np.eye(3)
    k01 = np.einsum("ij,nm->ijmn",f,finv)
    k02 = np.einsum("im,jn->ijmn",i,i)
    k03 = np.einsum("ji,nm->ijmn",finv,finv)
    k04 = np.einsum("ji,mn->ijmn",finv,f)
    k05 = np.einsum("jm,ni->ijmn",finv,finv)
    k06 = k03
    k07 = k05
    #
    k3x3x3x3 = -4/3*C1*J23*k01+2*C1*J23*k02+4*C1/9*J23*I1*k03-4*C1/3*J23*k04 \
                            +2*C1/3*J23*I1*k05+ yl*J*k06-yl*J*k07
    JFmT = J*finv.T
    #
    return p3x3, k3x3x3x3, JFmT, kappa_inv