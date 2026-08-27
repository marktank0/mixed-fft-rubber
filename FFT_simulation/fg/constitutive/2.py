# -*- coding: utf-8 -*-
"""
Created on Fri May  7 04:23:55 2021

@author: WANG Mingchuan
"""

import numpy as np
#
#
def umat(f,parameters):
    """ Neo-Hookean"""
    young = parameters[0]
    poisson = parameters[1]
    #
    C1 = young/4./(1+poisson)
    #
    D  = young/3./(1-2*poisson)
    #
    J = np.linalg.det(f)
    J23 = J**(-2/3)
    finv = np.linalg.inv(f)
    I1 = np.einsum("ij,ij->",f,f)
    #
    p3x3 = 2*C1*J**(-2/3)*f - 2*C1/3*J**(-2/3)*I1*finv.T+D*(J-1)*finv.T
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
                            +2*C1/3*J23*I1*k05+D*J*k06-D*(J-1)*k07
    return p3x3,k3x3x3x3