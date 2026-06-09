# -*- coding: utf-8 -*-
"""
Created on Fri May  7 03:17:38 2021

@author: WANG Mingchuan

code for <<A mixed FFT-based approach for incompressible
or slightly compressible hyperelastic solids under finite deformation>> 

"""

import numpy as np
# import numba as nb
import time
import scipy.sparse.linalg as sp
import itertools
import importlib.util
import os


class Problem:
    """ problem definitions """
    def __init__(self, path):
        """ """
        file = os.path.join(path, "charge.txt")
        #file = path + "charge.txt"
        chargedata = np.loadtxt(file)
        #
        self.dF = chargedata[2,:]
        self.dP = self.dF.copy()
        #
        charge_type = chargedata[3,:]
        #
        self.stress_control = []
        for i in range(3):
            for j in range(3):
                #stress control is 1.0
                if charge_type[3*i+j] > 0.5:
                    self.stress_control.append((i,j))
                else:
                    self.dP[3*i+j] = 0.0
        #
        self.model_a_name, self.model_a_para = self.__deal_with_model(chargedata[0,:])
        self.model_b_name, self.model_b_para = self.__deal_with_model(chargedata[1,:])
        #
    def __deal_with_model(self,modelparameters):
        """ """
        model_num = modelparameters[0]
        model_num = int(model_num)
        model_name = "{}.py".format(model_num)
        #
        model_para = modelparameters[1:]
        #
        return model_name, model_para
        
    
#-----------------------------------------useful functions
ddot42 = lambda A4,B2: np.einsum('ijklxyz,klxyz  ->ijxyz  ',A4,B2)
ddot22 = lambda A2,B2: np.einsum("ijxyz,ijxyz    ->xyz    ",A2,B2)
dot22  = lambda A2,B2: np.einsum("iaxyz,ajxyz    ->ijxyz  ",A2,B2)
t22    = lambda A2   : np.einsum("ijxyz -> jixyz",A2)

delta  = lambda i,j: float(i==j)            # Dirac delta function


def mat2vec9(mat):
    return np.array([mat[0,0],mat[0,1],mat[0,2],\
                     mat[1,0],mat[1,1],mat[1,2],\
                     mat[2,0],mat[2,1],mat[2,2]])
def vec2mat9(vec):
    return np.array([[vec[0],vec[1],vec[2]],
                     [vec[3],vec[4],vec[5]],
                     [vec[6],vec[7],vec[8]]])


def load_umat(module_path):
    module_name = "_fg_umat_{}".format(os.path.basename(module_path).replace(".", "_"))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.umat
#---------------------------------------------------------
    
    
    
class FFTSolver:
    """  """
    def __init__(self, path, N = 31):
        """ """
        self.pb = Problem(path)
        self.path = path
        #
        phasefile = os.path.join(path, "phase.txt")
        #phasefile = path + "phase.txt"
        phase = np.loadtxt(phasefile)
        self.phase = phase.reshape([N,N,N])
        self.N = N
        #
        self.iter_num = 0
        #
        print("---------------------------------------------")
        print("-- FFT based solver, by WANG Mingchuan")
        print("---------------------------------------------")
        #
    #
    def __average(self, A2):
        N = self.N
        avg = np.zeros((3,3))
        for x,y,z in itertools.product(range(N),repeat=3):
            avg += A2[:,:,x,y,z]
        avg = avg/(N*N*N)
        return avg
    #\
    def __counter(self,dX):
        self.iter_num += 1
    
    def calculate(self,increment = 10, incre_list=[], savemodel="no", give_Ghat=False, Ghat_given=[]):
        """ """
        #
        ndim = 3
        N = self.N
        #
        if not len(incre_list):
            incre_list = [0.1 for k in range(increment-1)]
        #
        eyeMat = np.einsum("ij,xyz->ijxyz",np.eye(ndim),np.ones([N,N,N]))
        #
        if give_Ghat:
            Ghat4 = Ghat_given
        else:
            freq   = np.arange(-(N-1)/2.,+(N+1)/2.)        # coordinate axis -> freq. axis
            Ghat4  = np.zeros([ndim,ndim,ndim,ndim,N,N,N]) # zero initialize
            # - compute
            for i,j,l,m in itertools.product(range(ndim),repeat=4):
                for x,y,z    in itertools.product(range(N),   repeat=3):
                    q = np.array([freq[x], freq[y], freq[z]])  # frequency vector
                    if not q.dot(q) == 0:                      # zero freq. -> mean
                        Ghat4[i,j,l,m,x,y,z] = delta(i,l)*q[j]*q[m]/(q.dot(q))
                    else:
                        if (i,j) in self.pb.stress_control:
                            Ghat4[i,j,l,m,x,y,z] = delta(i,l)*delta(j,m)
        print("Ghat4 is formed...")
        #
        fft    = lambda x  : np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[N,N,N]))
        ifft   = lambda x  : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[N,N,N]))
        #
        # functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
        G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)))).reshape(-1)
        K_dF   = lambda dFm: ddot42(K4,dFm.reshape(ndim,ndim,N,N,N))
        G_K_dF = lambda dFm: G(K_dF(dFm))
        #
        phase = self.phase
        #
        parameter_a = self.pb.model_a_para
        parameter_b = self.pb.model_b_para
        #
        #consti_a_path = "fg/constitutive/"+self.pb.model_a_name
        consti_a_path = os.path.join("fg/constitutive",self.pb.model_a_name)
        consti_a = load_umat(consti_a_path)
        #consti_b_path = "fg/constitutive/"+self.pb.model_b_name
        consti_b_path = os.path.join("fg/constitutive",self.pb.model_b_name)
        consti_b = load_umat(consti_b_path)
        #
        def constitutive(F):
            P = np.zeros([ndim,ndim,N,N,N])
            K4 = np.zeros([ndim,ndim,ndim,ndim,N,N,N])
            #
            for x,y,z in itertools.product(range(N),repeat=3):
                f = F[:,:,x,y,z]
                #
                if phase[x,y,z] == 0:
                    p,k4 = consti_a(f,parameter_a)
                    #
                elif phase[x,y,z] == 1:
                    p,k4 = consti_b(f,parameter_b)
                #
                P[:,:,x,y,z] = p
                #
                K4[:,:,:,:,x,y,z] = k4
            return P,K4
        
        
        # set macroscopic loading----------------------------
        DbarF_total = np.zeros([ndim,ndim,N,N,N])
        #
        DbarF_total[0,0] = self.pb.dF[0]
        DbarF_total[0,1] = self.pb.dF[1]
        DbarF_total[0,2] = self.pb.dF[2]
        DbarF_total[1,0] = self.pb.dF[3]
        DbarF_total[1,1] = self.pb.dF[4]
        DbarF_total[1,2] = self.pb.dF[5]
        DbarF_total[2,0] = self.pb.dF[6]
        DbarF_total[2,1] = self.pb.dF[7]
        DbarF_total[2,2] = self.pb.dF[8]
        #
        DbarP_total = DbarF_total.copy()
        #
        DbarP_total[0,0] = self.pb.dP[0]
        DbarP_total[0,1] = self.pb.dP[1]
        DbarP_total[0,2] = self.pb.dP[2]
        DbarP_total[1,0] = self.pb.dP[3]
        DbarP_total[1,1] = self.pb.dP[4]
        DbarP_total[1,2] = self.pb.dP[5]
        DbarP_total[2,0] = self.pb.dP[6]
        DbarP_total[2,1] = self.pb.dP[7]
        DbarP_total[2,2] = self.pb.dP[8]
        #
        #
        self.Ps = []
        self.Fs = []
        
        #
        F = np.array(eyeMat,copy=True)
        #P,K4  = constitutive(F)
        inc_tol = 0.0
        for inc in incre_list:
            print("this increment is {} ------------------------".format(inc))
            inc_tol = inc_tol + inc
            DbarF = inc*DbarF_total
            #DbarP = inc*DbarP_total
            TbarP = inc_tol*DbarP_total
            t1 = time.time()
            #
            F     += DbarF
            P,K4  = constitutive(F)
            #b     = G(DbarP)-G_K_dF(DbarF)
            b     = G(TbarP - P)
            Fn    = np.linalg.norm(F)
            iiter = 0
            #
            #print("iteration begins...")
            while iiter < 100:
                self.iter_num = 0
                #begin the iteration
                dFm,flag = sp.cg(rtol=1.e-8, atol=0.0,
                  A = sp.LinearOperator(shape=(F.size,F.size),matvec=G_K_dF,dtype='float'),
                  b = b, callback=self.__counter, maxiter = 1000, 
                )                                        # solve linear system using CG
                #print(flag)
                if flag > 0:
                    print(flag)
                    break
                F    += dFm.reshape(ndim,ndim,N,N,N)     #
                P,K4  = constitutive(F)                  # new residual stress and tangent
                b     = G(TbarP - P)
                print("res {:.3e} iter times {}".format(np.linalg.norm(dFm)/Fn, self.iter_num))
                if np.linalg.norm(dFm)/Fn<1.e-5 and iiter>0: 
                    break # check convergence
                iiter += 1
            #
            t2 = time.time()
            print("time this step...{}".format(t2-t1))
            #
            #save average stress and strain
            Pavg = self.__average(P)
            self.Ps.append(Pavg)
            print("now P is...")
            print(Pavg)
            Favg = self.__average(F)
            self.Fs.append(Favg)
            print("now F is...")
            print(Favg)
            #
        #-------------------------------post 
        print("finish!")
        #
        if savemodel == "normal" or savemodel == "both":
            #save Fs and Ps
            self.__save_F_P(self.path)
            print("F and P are saved")
    #=============================================================================
    
    def __save_F_P(self,path):
        #
        num = len(self.Ps)
        #
        Fs = np.zeros([num,9])
        Ps = np.zeros([num,9])
        #
        for i in range(num):
            Fvec = mat2vec9(self.Fs[i])
            Pvec = mat2vec9(self.Ps[i])
            #
            Fs[i,:] = Fvec
            Ps[i,:] = Pvec
        #
        Ffile = os.path.join(path, "F.txt")
        #Ffile = path + "F.txt"
        np.savetxt(Ffile, Fs)
        #
        Pfile = os.path.join(path, "P.txt")
        #Pfile = path + "P.txt"
        np.savetxt(Pfile, Ps)
        
        
                
        
        
        
        
        
        
        
        
        
        
        
