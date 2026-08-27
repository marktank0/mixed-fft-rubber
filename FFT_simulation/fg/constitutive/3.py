# -*- coding: utf-8 -*-
"""
Created on Fri May  7 04:23:55 2021

@author: WANG Mingchuan
"""

import numpy as np
#
#
def umat(f0,f1,sig,state,dtime,mps):
    """ iso strain"""
    df = f1 - f0
    fm = 0.5*(f0 + f1)
    finv = np.linalg.inf(fm)
    #
    L = df @ finv
    D = 0.5*(L + L.T)
    W = L - D
    #
    E = mps[0]
    nu = mps[1]
    #
    lam = E*nu/(1 + nu)/(1 - 2*nu)
    G   = E/2/(1 + nu)
    #
    #
    eye = np.eye(3)
    dr  = np.linalg.inv(eye + 0.5*W) @ (eye - 0.5*W)
    #
    tolerance = 1.0e-5
    niteration = 20
    #
    stiff = np.array([[lam+2*G,lam,lam,0,0,0],
                  [lam,lam+2*G,lam,0,0,0],
                  [lam,lam,lam+2*G,0,0,0],
                  [0,0,0,G,0,0],
                  [0,0,0,0,G,0],
                  [0,0,0,0,0,G]])
    #
    sigJ = np.zeros(6)
    Dvec = mat3x3tovec6(D,style='eps')
    sigJ = stiff @ Dvec
    sigJ = vec6tomat3x3(sigJ)
    #
    sig = dr @ sig @ dr.T + sigJ
    sig_vm  = calmises(sig)
    #
    #current state
    r = state[0]
    kappa_ini = mps[2]
    A      = mps[3]
    kappa, dkappa = calkappa(r, kappa_ini, A)
    #
    kappa0 = kappa
    #
    ddsdde = stiff
    #
    convergeflag = True
    #
    if sig_vm > (1.0+tolerance)*kappa0:
        #plasticity
        #return-mapping begins
        trsig = np.trace(sig)
        sigs  = sig - trsig/3*eye
        #
        #norm
        flow = 1.0/sig_vm*sigs
        #
        #newton-raphson
        dr = 0.0
        convergeflag = False
        for i in range(niteration):
            res = sig_vm - 3.*G*dr - kappa
            dr  = dr + res/(3.*G + dkappa)
            rtemp = r + dr
            kappa, dkappa = calkappa(rtemp, kappa_ini, A)
            #
            if abs(res) < tolerance*kappa0:
                convergeflag = True
                break
        #
        #update variables
        sig = kappa*flow + trsig/3.*eye
        r   = r + dr
        #
        efflam = E/(1-2*nu)/3-2/3*G*kappa/sig_vm
        effnu = G*kappa/sig_vm
        coef = dkappa*3*E/(3*E+2*dkappa*(1+nu))-3*effnu
        #
        ddsdde = np.array([[efflam+2*effnu,efflam,efflam,0,0,0],
                           [efflam,efflam+2*effnu,efflam,0,0,0],
                           [efflam,efflam,efflam+2*effnu,0,0,0],
                           [0,0,0,effnu,0,0],
                           [0,0,0,0,effnu,0],
                           [0,0,0,0,0,effnu]])
        flowvec = mat3x3tovec6(flow)
        for k in range(6):
            for l in range(6):
                ddsdde[k,l] = ddsdde[k,l] + coef*flowvec[k]*flowvec[l]
        #
        state_up = np.zeros(state.shape)
        state_up[0] = r
        #
    p3x3 = J*sig @ np.linalg.inv(f1).T
    k3x3x3x3 = calK(ddsdde)
        #

    return p3x3,k3x3x3x3

def vec9tomat3x3(vec):
    """
    """
    mat = np.array([[vec[0], vec[1], vec[2]],
                    [vec[3], vec[4], vec[5]],
                    [vec[6], vec[7], vec[8]]])
    return mat

def mat3x3tovec9(mat):
    """
    """
    vec = np.array([mat[0,0],mat[0,1],mat[0,2],mat[1,0],mat[1,1],mat[1,2],\
                    mat[2,0],mat[2,1],mat[2,2]])
    return vec
#
def vec6tomat3x3(vec,style = 'sig'):
    """
    """
    if style == 'sig':
        coef = 1.0
    elif style == 'eps':
        coef = 2.0
    else:
        return None
        print("wrong input")
    #
    mat = np.array([[vec[0],vec[3]/coef,vec[4]/coef],
                    [vec[3]/coef,vec[1],vec[5]/coef],
                    [vec[4]/coef,vec[5]/coef,vec[2]]])
    return mat
#
def mat3x3tovec6(mat,style = 'sig'):
    """
    """
    if style == 'sig':
        coef = 1.0
    elif style == 'eps':
        coef = 2.0
    else:
        return None
        print("wrong input")
    #
    vec = np.array([mat[0,0],mat[1,1],mat[2,2],\
                    coef*mat[0,1],coef*mat[0,2],coef*mat[1,2]])
    return vec

def calmises(sig):
    """
    calculated von mises stress
    """
    sigm = (sig[0,0] - sig[1,1])**2 \
         + (sig[2,2] - sig[1,1])**2 \
         + (sig[0,0] - sig[2,2])**2 \
         + 6.0*(sig[0,1]**2+sig[1,2]**2+sig[0,2]**2)
    sigm = 0.5*sigm
    sigm = np.sqrt(sigm)
    return sigm

def calkappa(r, *parameters):
    kappa = parameters[0] + parameters[1]*r
    dkappa = parameters[1]
    return kappa, dkappa
    
def calK(ddsdde, f, tau):
    #
    J = np.linalg.det(f)
    #
    
    
    
    
    
    
    
    