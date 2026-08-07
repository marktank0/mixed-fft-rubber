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

from fg.io_paths import (
    default_charge_path,
    ensure_output_path,
    output_run_path,
    phase_source,
)
from fg.preconditioning import (
    REFERENCE_MODES,
    apply_standard_reference_preconditioner,
    build_standard_reference_symbol,
    reference_average,
)
from fg.vtk_export import save_vti_cell_fields, solution_fields


class Problem:
    """ problem definitions """
    def __init__(self, path):
        """ """
        file = path if os.path.isfile(path) else os.path.join(path, "charge.txt")
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


def load_phase(path, N, phase_path=None, phase_key="phase"):
    if phase_path is None:
        npz_files = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".npz")
        ]
        phase_txt = os.path.join(path, "phase.txt")
        if len(npz_files) == 1:
            phase_path = npz_files[0]
        elif os.path.exists(phase_txt):
            phase_path = phase_txt
        elif len(npz_files) > 1:
            raise ValueError("Multiple .npz files found in {}; pass phase_path explicitly".format(path))
        else:
            raise FileNotFoundError("No phase .npz file or phase.txt found in {}".format(path))

    if phase_path.lower().endswith(".npz"):
        with np.load(phase_path, allow_pickle=False) as data:
            if phase_key not in data.files:
                raise KeyError("Missing phase key '{}' in {}".format(phase_key, phase_path))
            phase = np.array(data[phase_key], dtype=float, copy=True)
    else:
        phase = np.loadtxt(phase_path)

    if phase.ndim == 1:
        phase = phase.reshape([N,N,N])
    elif phase.shape != (N,N,N):
        raise ValueError("Phase shape {} does not match N={} for {}".format(phase.shape, N, phase_path))

    return phase.astype(float, copy=False), phase_path


def format_scientific_cut(value, decimals=2):
    if not np.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0.00e+00"

    sign = "-" if value < 0.0 else ""
    value_abs = abs(value)
    exponent = int(np.floor(np.log10(value_abs)))
    mantissa = value_abs/(10.0**exponent)
    factor = 10.0**decimals
    mantissa = np.floor(mantissa*factor)/factor
    return "{}{:.{}f}e{:+03d}".format(sign, mantissa, decimals, exponent)


def save_output_csv(path, output, header):
    ensure_output_path(path)
    outfile = os.path.join(path, "output.csv")
    with open(outfile, "w", newline="") as file:
        file.write(header + "\n")
        for row in output:
            file.write(",".join(format_scientific_cut(value) for value in row) + "\n")


#---------------------------------------------------------
    
    
    
class FFTSolver:
    """  """
    def __init__(self, structure_path, charge_path = None, output_path = None, N = 31, phase_path = None, phase_key = "phase"):
        """ """
        self.structure_path = structure_path
        self.charge_path = charge_path or default_charge_path(structure_path)
        self.output_path = ensure_output_path(output_run_path(structure_path, output_path))
        self.path = self.output_path
        self.pb = Problem(self.charge_path)
        #
        phase_dir, selected_phase_path = phase_source(structure_path, phase_path)
        self.phase, self.phase_path = load_phase(phase_dir, N, selected_phase_path, phase_key)
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

    def __progress_counter(self, matvec, rhs, label="cg", interval=50):
        rhs_norm = max(np.linalg.norm(rhs), 1.0)

        def callback(xk):
            self.iter_num += 1
            if self.iter_num % interval == 0:
                residual = np.linalg.norm(rhs - matvec(xk))/rhs_norm
                print("{} iter {} linear residual {:.3e}".format(label, self.iter_num, residual))

        return callback
    
    def calculate(
        self,
        increment = 10,
        incre_list=[],
        savemodel="no",
        give_Ghat=False,
        Ghat_given=[],
        preconditioner=None,
        save_fields=False,
        field_filename="fields.vti",
        reference="mean",
    ):
        """ """
        #
        if preconditioner not in (None, "none", "reference"):
            raise ValueError("Unknown preconditioner {!r}; use None or 'reference'.".format(preconditioner))
        if reference not in REFERENCE_MODES:
            raise ValueError("Unknown reference {!r}; use one of {}.".format(reference, REFERENCE_MODES))
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
        mask_a = (phase == 0)
        mask_b = (phase == 1)
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
                cg_callback = self.__progress_counter(G_K_dF, b, label="cg")
                Aop = sp.LinearOperator(shape=(F.size,F.size),matvec=G_K_dF,dtype='float')
                Mop = None
                if preconditioner == "reference":
                    K_ref = reference_average(K4, reference, mask_a, mask_b)
                    zero_mode_free = [3*i + j for i,j in self.pb.stress_control]
                    inv_symbol = build_standard_reference_symbol(
                        Ghat4, K_ref, zero_mode_free_components=zero_mode_free
                    )
                    Mop = sp.LinearOperator(
                        shape=(F.size,F.size),
                        matvec=lambda vec: apply_standard_reference_preconditioner(vec, inv_symbol),
                        dtype='float',
                    )
                #begin the iteration
                dFm,flag = sp.cg(rtol=1.e-8, atol=0.0,
                  A = Aop, b = b, M = Mop, callback=cg_callback, maxiter = 1000, 
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
            print("F and P are saved to output.csv")
        if save_fields:
            field_file = save_vti_cell_fields(
                self.path,
                solution_fields(F, P, phase),
                filename=field_filename,
            )
            print("local fields are saved to {}".format(field_file))
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
        output = np.hstack((Fs, Ps))
        header = ",".join([
            "F11","F12","F13","F21","F22","F23","F31","F32","F33",
            "P11","P12","P13","P21","P22","P23","P31","P32","P33",
        ])
        save_output_csv(path, output, header)
        
        
                
        
        
        
        
        
        
        
        
        
        
        
