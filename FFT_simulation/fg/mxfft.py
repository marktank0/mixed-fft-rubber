# -*- coding: utf-8 -*-
"""
Created on Fri May  7 03:17:38 2021

@author: WANG Mingchuan

code for <<A mixed FFT-based approach for incompressible
or slightly compressible hyperelastic solids under finite deformation>>


Mixed FFT implementation
"""

import numpy as np
# import numba as nb
import json
import time
import scipy.fft
import scipy.sparse.linalg as sp
import importlib.util
import os

from fg.io_paths import (
    default_charge_path,
    ensure_output_path,
    output_run_path,
    phase_source,
)
from fg.preconditioning import (
    DISCRETIZATIONS,
    REFERENCE_MODES,
    apply_green_jacobi_preconditioner,
    apply_mixed_reference_preconditioner,
    build_Ghat4,
    build_green_jacobi_symbol,
    build_mixed_reference_symbol,
    local_jacobi_scale,
    reference_average,
)
from fg.vtk_export import save_vti_cell_fields, solution_fields

_FFT_WORKERS = int(os.environ.get("FFT_WORKERS", "1"))

class Problem:
    """ problem definitions """
    def __init__(self, path):
        """ """
        chargedata = np.loadtxt(path)
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


# The constitutive model files ship with the solver, so they are located
# relative to this package - never relative to the working directory, which
# differs between the repo root, FFT_simulation/ and a batch worker.
FG_DIR = os.path.dirname(os.path.abspath(__file__))
CONSTITUTIVE_DIR = os.path.join(FG_DIR, "constitutive_incompressible")


def load_umat(module_path, function_name="umat"):
    module_name = "_fg_umat_{}".format(os.path.basename(module_path).replace(".", "_"))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


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


def eisenstat_walker_forcing(res_new, res_old, eta_prev,
                             gamma=0.9, alpha=2.0, eta_min=1.e-3, eta_max=1.e-2):
    """Eisenstat-Walker "choice 2" forcing term for the next inexact Newton solve.

    eta_k = gamma*(||r_k||/||r_{k-1}||)**alpha, with the standard safeguard
    that stops the sequence from dropping too fast after one lucky step
    (applied only while gamma*eta_{k-1}**alpha exceeds 0.1), then clamped to
    [eta_min, eta_max].

    Eisenstat & Walker (1996), SIAM J. Sci. Comput. 17(1), 16-32.
    """
    if not (res_old > 0.0) or not np.isfinite(res_new):
        return float(eta_max)

    eta = gamma*(res_new/res_old)**alpha
    safeguard = gamma*eta_prev**alpha
    if safeguard > 0.1:
        eta = max(eta, safeguard)
    return float(min(eta_max, max(eta_min, eta)))


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
    def __init__(self, structure_path, charge_path = None, output_path = None, N = 31, phase_path = None, phase_key = "phase", output_name = None):
        """ """
        self.structure_path = structure_path
        self.charge_path = charge_path or default_charge_path(structure_path)
        self.output_path = ensure_output_path(output_run_path(structure_path, output_path, output_name))
        self.path = self.output_path
        self.pb = Problem(self.charge_path)
        #
        phase_dir, selected_phase_path = phase_source(structure_path, phase_path)
        self.phase, self.phase_path = load_phase(phase_dir, N, selected_phase_path, phase_key)
        self.N = N
        #
        self.iter_num = 0
        self.solver_status = "not_run"
        self.solver_stats = {}
        print("---------------------------------------------")
        print("-- mixed FFT based solver, by WANG Mingchuan")
        print("---------------------------------------------")
        #
    #
    def __average(self, A2):
        return A2.mean(axis=(2,3,4))
    #
    def __counter(self,dX):
        self.iter_num += 1

    def __progress_counter(self, matvec, rhs, label="cg", interval=10):
        rhs_norm = max(np.linalg.norm(rhs), 1.0)

        def callback(xk):
            self.iter_num += 1
            if self.iter_num % interval == 0:
                residual = np.linalg.norm(rhs - matvec(xk))/rhs_norm
                print("{} iter {} linear residual {:.3e}".format(label, self.iter_num, residual))

        return callback

    def __gmres_progress_counter(self, label="gmres", interval=10):
        def callback(residual):
            self.iter_num += 1
            if self.iter_num % interval == 0:
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
        diagnostics=False,
        save_fields=False,
        field_filename="fields.vti",
        tol_rel=1.e-5,
        tol_abs=1.e-10,
        max_gmres_iter=1000,
        max_backtracks=8,
        min_substep_ratio=1.0/16.0,
        gmres_restart=None,
        reference="mean",
        forcing="eisenstat_walker",
        inner_rtol=1.e-6,
        eta_max=1.e-2,
        eta_min=1.e-3,
        ew_gamma=0.9,
        ew_alpha=2.0,
        discretization="fourier",
        precond_restrict=True,
    ):
        """ """
        #
        if preconditioner not in (None, "none", "gmres", "reference", "green_jacobi"):
            raise ValueError(
                "Unknown preconditioner {!r}; use None, 'gmres', 'reference' or "
                "'green_jacobi'.".format(preconditioner))
        if reference not in REFERENCE_MODES:
            raise ValueError("Unknown reference {!r}; use one of {}.".format(reference, REFERENCE_MODES))
        if forcing not in ("fixed", "eisenstat_walker"):
            raise ValueError("Unknown forcing {!r}; use 'fixed' or 'eisenstat_walker'.".format(forcing))
        if not 0.0 < eta_min <= eta_max < 1.0:
            raise ValueError("Require 0 < eta_min <= eta_max < 1; got {} and {}.".format(eta_min, eta_max))
        if discretization not in DISCRETIZATIONS:
            raise ValueError("Unknown discretization {!r}; use one of {}.".format(discretization, DISCRETIZATIONS))
        #
        ndim = 3
        N = self.N
        num_up = ndim*ndim*N*N*N
        num_down = N*N*N
        num_tol = num_up + num_down
        #
        if gmres_restart is None:
            # cap the GMRES basis at ~800 MB; 100 at N=31, ~40 at N=63
            gmres_restart = int(min(100, max(20, 8.e8/(8.0*num_tol))))
        print("gmres restart length: {}".format(gmres_restart))
        #
        if not len(incre_list):
            incre_list = [0.1 for k in range(increment-1)]
        #
        eyeMat = np.einsum("ij,xyz->ijxyz",np.eye(ndim),np.ones([N,N,N]))
        #
        if give_Ghat:
            Ghat4 = Ghat_given
        else:
            Ghat4 = build_Ghat4(N, self.pb.stress_control, ndim, discretization)
        print("Ghat4 is formed ({} discretization)...".format(discretization))
        #
        # shifts act on the spatial axes only (component axes are untouched
        # by the transform, so shifting them cancelled out anyway)
        axes = (-3, -2, -1)
        fft    = lambda x  : np.fft.fftshift(scipy.fft.fftn (np.fft.ifftshift(x, axes=axes), axes=axes, workers=_FFT_WORKERS), axes=axes)
        ifft   = lambda x  : np.fft.fftshift(scipy.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, workers=_FFT_WORKERS), axes=axes)
        #
        G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)))).reshape(-1)
        #
        phase = self.phase
        #
        parameter_a = self.pb.model_a_para
        parameter_b = self.pb.model_b_para
        #
        consti_a_path = os.path.join(CONSTITUTIVE_DIR,self.pb.model_a_name)
        consti_a = load_umat(consti_a_path, "umat_field")
        consti_b_path = os.path.join(CONSTITUTIVE_DIR,self.pb.model_b_name)
        consti_b = load_umat(consti_b_path, "umat_field")
        #
        mask_a = (phase == 0)
        mask_b = (phase == 1)
        if not np.all(mask_a | mask_b):
            raise ValueError("Phase field contains values other than 0 and 1")
        #
        def constitutive(F, Yali, need_tangent=True):
            P = np.zeros([ndim,ndim,N,N,N])
            K4 = np.zeros([ndim,ndim,ndim,ndim,N,N,N]) if need_tangent else None
            JFinv = np.zeros([ndim,ndim,N,N,N])
            Kappa_inv = np.zeros([N,N,N])
            #
            for mask, consti, para in ((mask_a, consti_a, parameter_a),
                                       (mask_b, consti_b, parameter_b)):
                if not mask.any():
                    continue
                fv = np.moveaxis(F[:, :, mask], -1, 0)          # (m,3,3)
                ylv = Yali[mask]
                p, k4, jfmt, kappainv = consti(fv, ylv, para, need_tangent=need_tangent)
                P[:, :, mask] = np.moveaxis(p, 0, -1)
                if need_tangent:
                    K4[:, :, :, :, mask] = np.moveaxis(k4, 0, -1)
                JFinv[:, :, mask] = np.moveaxis(jfmt, 0, -1)
                Kappa_inv[mask] = kappainv
            return P, K4, JFinv, Kappa_inv

        def det_field(F):
            """det(F) per voxel, shape (N^3,)."""
            return np.linalg.det(np.moveaxis(F.reshape(ndim, ndim, -1), -1, 0))

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
        def calb(F,P,YALI,Kappa_inv,TbarP):
            """ to calculate the b in KdX = b """
            b = np.zeros(num_tol)
            b[0:num_up] = G(TbarP - P)
            J = det_field(F)
            b[num_up:num_tol+1] = 1.0 - J + (YALI*Kappa_inv).reshape(-1)
            #
            return b

        # current tangent fields used by the matrix-free operator; rebound
        # by the Newton loop after every accepted step
        state = {"K4": None, "JFmT": None, "Kappa_inv": None}

        def KdX(dX):
            """ A(X) """
            K4 = state["K4"]
            JFmT = state["JFmT"]
            Kappa_inv = state["Kappa_inv"]
            #
            dFv, dpv = np.split(dX, [num_up])
            dF = dFv.reshape([ndim,ndim,N,N,N])
            dp = dpv.reshape([N,N,N])
            #
            #tmp1: K:dF + JFmT.dp
            tmp1a = ddot42(K4, dF)          #K4:dF
            tmp1b = np.einsum("ijxyz,xyz->ijxyz",JFmT, dp)      #JFmT.dp
            tmp1mat = tmp1a + tmp1b
            tmp1 = G(tmp1mat)
            #
            #tmp2: JFmT:dF - 1/kappa*dp
            tmp2a = ddot22(JFmT, dF)
            tmp2b = Kappa_inv*dp
            tmp2mat = tmp2a - tmp2b
            tmp2 = tmp2mat.reshape(-1)
            #
            result = np.hstack((tmp1,tmp2))
            #
            return result

        def print_linear_diagnostics(dX, Mop, b):
            """Print original-system and preconditioned linear residuals."""
            residual = b - KdX(dX)
            rhs_norm = max(np.linalg.norm(b), 1.0)
            true_total = np.linalg.norm(residual)/rhs_norm
            true_F = np.linalg.norm(residual[0:num_up])/max(np.linalg.norm(b[0:num_up]), 1.0)
            true_p = np.linalg.norm(residual[num_up:num_tol])/max(np.linalg.norm(b[num_up:num_tol]), 1.0)
            print(
                "linear true residual total {:.3e} F-block {:.3e} p-block {:.3e}"
                .format(true_total, true_F, true_p)
            )

            if Mop is not None:
                prec_residual = Mop.matvec(residual)
                prec_rhs = Mop.matvec(b)
                prec_total = np.linalg.norm(prec_residual)/max(np.linalg.norm(prec_rhs), 1.0)
                prec_F = np.linalg.norm(prec_residual[0:num_up])/max(np.linalg.norm(prec_rhs[0:num_up]), 1.0)
                prec_p = np.linalg.norm(prec_residual[num_up:num_tol])/max(np.linalg.norm(prec_rhs[num_up:num_tol]), 1.0)
                print(
                    "linear preconditioned residual total {:.3e} F-block {:.3e} p-block {:.3e}"
                    .format(prec_total, prec_F, prec_p)
                )

        def solve_linear(b, rtol):
            """One preconditioned Krylov solve of KdX = b to relative tolerance rtol."""
            self.iter_num = 0
            Aop = sp.LinearOperator(shape=(num_tol,num_tol),matvec=KdX,dtype='float')
            Mop = None
            if preconditioner == "green_jacobi":
                # Green-Jacobi: divide out the local stiffness before applying
                # the Green symbol, so the local-to-reference ratios that bound
                # the preconditioned spectrum become O(1) instead of O(contrast)
                d = local_jacobi_scale(state["K4"])
                zero_mode_free = [3*i + j for i, j in self.pb.stress_control] + [9]
                inv_symbol = build_green_jacobi_symbol(
                    Ghat4, state["K4"], state["JFmT"], state["Kappa_inv"], d,
                    reference=reference, matrix_mask=mask_a, filler_mask=mask_b,
                    zero_mode_free_components=zero_mode_free,
                )
                Mop = sp.LinearOperator(
                    shape=(num_tol, num_tol),
                    matvec=lambda vec: apply_green_jacobi_preconditioner(vec, inv_symbol, d, Ghat4),
                    dtype='float',
                )
            if preconditioner == "reference":
                K_ref = reference_average(state["K4"], reference, mask_a, mask_b)
                J_ref = reference_average(state["JFmT"], reference, mask_a, mask_b)
                kappa_inv_ref = float(reference_average(state["Kappa_inv"], reference, mask_a, mask_b))
                zero_mode_free = [3*i + j for i,j in self.pb.stress_control] + [9]
                inv_symbol = build_mixed_reference_symbol(
                    Ghat4, K_ref, J_ref, kappa_inv_ref,
                    zero_mode_free_components=zero_mode_free,
                    restrict_to_compatible=precond_restrict,
                )
                Mop = sp.LinearOperator(
                    shape=(num_tol,num_tol),
                    matvec=lambda vec: apply_mixed_reference_preconditioner(vec, inv_symbol),
                    dtype='float',
                )
            if preconditioner in ("gmres", "reference", "green_jacobi"):
                gmres_callback = self.__gmres_progress_counter(label="gmres")
                # scipy's maxiter counts restart cycles, not iterations, so
                # convert the total-iteration cap to whole cycles
                restart_cycles = max(1, int(np.ceil(max_gmres_iter/float(gmres_restart))))
                dX,flag = sp.gmres(rtol=rtol, atol=1.e-10,
                  A = Aop, b = b, M = Mop, callback=gmres_callback,
                  callback_type="pr_norm", restart=gmres_restart,
                  maxiter=restart_cycles,
                )
            else:
                cg_callback = self.__progress_counter(KdX, b, label="cg")
                dX,flag = sp.cg(rtol=rtol, atol=1.e-10,
                  A = Aop, b = b, callback=cg_callback,
                  maxiter=max_gmres_iter,
                )
            if diagnostics:
                print_linear_diagnostics(dX, Mop, b)
            return dX, flag, Mop

        def newton_increment(F, YALI, TbarP):
            """Newton solve at fixed load TbarP starting from (F, YALI).

            Convergence is judged on the true nonlinear residual b, each block
            relative to its norm at the start of the sub-increment. Every step
            is safeguarded: det(F) must stay positive and the residual must
            decrease (backtracking line search), otherwise the sub-increment
            is reported as failed so the caller can cut the load step.
            Returns (converged, F, YALI, P, stats).
            """
            F = F.copy()
            YALI = YALI.copy()
            P, K4, JFmT, Kappa_inv = constitutive(F, YALI)
            state.update(K4=K4, JFmT=JFmT, Kappa_inv=Kappa_inv)
            b = calb(F, P, YALI, Kappa_inv, TbarP)
            #
            res_F = np.linalg.norm(b[0:num_up])
            res_p = np.linalg.norm(b[num_up:num_tol])
            ref_F = max(res_F, tol_abs)
            ref_p = max(res_p, tol_abs)
            merit = np.hypot(res_F/ref_F, res_p/ref_p)
            #
            Fn = np.linalg.norm(F)
            #
            stats = {
                "newton_iterations": 0,
                "krylov_iterations": [],
                "forcing_terms": [],
                "alphas": [],
                "residual_F": [res_F],
                "residual_p": [res_p],
                "fail_reason": None,
            }

            def is_converged():
                return (res_F <= max(tol_rel*ref_F, tol_abs)
                        and res_p <= max(tol_rel*ref_p, tol_abs))

            # inexact-Newton forcing term: the first solve of a sub-increment
            # has no residual history, so it starts at the loosest allowed
            # tolerance and tightens as the Newton residual falls
            eta = eta_max if forcing == "eisenstat_walker" else inner_rtol

            while not is_converged():
                eta_used = eta
                stats["forcing_terms"].append(eta_used)
                dX, flag, Mop = solve_linear(b, eta_used)
                stats["krylov_iterations"].append(self.iter_num)
                if flag > 0:
                    stats["fail_reason"] = "gmres_iteration_cap"
                    print("linear solver hit the iteration cap ({} iterations, flag {})".format(
                        self.iter_num, flag))
                    return False, F, YALI, P, stats
                dFm, dYL = np.split(dX, [num_up])
                dF = dFm.reshape(ndim,ndim,N,N,N)
                dp = dYL.reshape(N,N,N)
                #
                # safeguarded update: keep det(F) positive and require a
                # residual decrease, halving the step length otherwise
                J_floor = 0.05*det_field(F).min()
                alpha = 1.0
                accepted = False
                for _ in range(max_backtracks):
                    F_t = F + alpha*dF
                    if det_field(F_t).min() <= J_floor:
                        alpha *= 0.5
                        continue
                    YALI_t = YALI + alpha*dp
                    P_t, _, _, _ = constitutive(F_t, YALI_t, need_tangent=False)
                    b_t = calb(F_t, P_t, YALI_t, Kappa_inv, TbarP)
                    res_F_t = np.linalg.norm(b_t[0:num_up])
                    res_p_t = np.linalg.norm(b_t[num_up:num_tol])
                    merit_t = np.hypot(res_F_t/ref_F, res_p_t/ref_p)
                    if merit_t < merit:
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    stats["fail_reason"] = "line_search"
                    print("line search failed (last alpha {:.3e})".format(alpha))
                    return False, F, YALI, P, stats
                #
                F = F_t
                YALI = YALI_t
                b = b_t
                merit_prev = merit
                res_F, res_p, merit = res_F_t, res_p_t, merit_t
                P, K4, JFmT, Kappa_inv = constitutive(F, YALI)
                state.update(K4=K4, JFmT=JFmT, Kappa_inv=Kappa_inv)
                #
                # next inner tolerance from the achieved outer residual drop
                if forcing == "eisenstat_walker":
                    eta = eisenstat_walker_forcing(
                        merit, merit_prev, eta,
                        gamma=ew_gamma, alpha=ew_alpha,
                        eta_min=eta_min, eta_max=eta_max,
                    )
                #
                stats["newton_iterations"] += 1
                stats["alphas"].append(alpha)
                stats["residual_F"].append(res_F)
                stats["residual_p"].append(res_p)
                print("res {:.3e} (F-block {:.3e} p-block {:.3e}) alpha {:.3g} eta {:.2e} iter times {}".format(
                    alpha*np.linalg.norm(dFm)/Fn, res_F/ref_F, res_p/ref_p, alpha, eta_used, self.iter_num))
            #
            return True, F, YALI, P, stats

        #
        F = np.array(eyeMat,copy=True)
        YALI = np.zeros([N,N,N])
        #
        self.solver_status = "in_progress"
        self.solver_stats = {
            "status": "in_progress",
            "tol_rel": tol_rel,
            "tol_abs": tol_abs,
            "max_gmres_iter": max_gmres_iter,
            "preconditioner": preconditioner,
            "discretization": discretization,
            "precond_restrict": precond_restrict,
            "reference": reference,
            "forcing": forcing,
            "inner_rtol": inner_rtol if forcing == "fixed" else None,
            "eta_min": eta_min if forcing == "eisenstat_walker" else None,
            "eta_max": eta_max if forcing == "eisenstat_walker" else None,
            "step_cuts": 0,
            "increments": [],
        }
        failed = False
        #
        inc_tol = 0.0
        targets = np.cumsum(incre_list)
        step = float(incre_list[0])
        easy_streak = 0
        for k in range(len(targets)):
            target = float(targets[k])
            inc_full = float(incre_list[k])
            step = min(step, inc_full)
            P_boundary = None
            #
            while inc_tol < target - 1.e-12:
                step = min(step, target - inc_tol)
                print("this increment is {} (target {}) ------------------------".format(step, target))
                t1 = time.time()
                #
                F_trial = F + step*DbarF_total
                TbarP = (inc_tol + step)*DbarP_total
                #
                converged, F_new, YALI_new, P_new, inc_stats = newton_increment(F_trial, YALI, TbarP)
                #
                inc_stats["step"] = step
                inc_stats["load_start"] = inc_tol
                inc_stats["converged"] = converged
                inc_stats["time_seconds"] = time.time() - t1
                self.solver_stats["increments"].append(inc_stats)
                #
                if converged:
                    F = F_new
                    YALI = YALI_new
                    P_boundary = P_new
                    inc_tol += step
                    if inc_stats["newton_iterations"] <= 4:
                        easy_streak += 1
                    else:
                        easy_streak = 0
                    if easy_streak >= 2:
                        step = min(step*1.5, inc_full)
                    print("time this step...{}".format(inc_stats["time_seconds"]))
                else:
                    # roll back to the last accepted state and cut the load step
                    easy_streak = 0
                    step *= 0.5
                    self.solver_stats["step_cuts"] += 1
                    print("increment failed ({}); cutting step to {:.4g}".format(
                        inc_stats["fail_reason"], step))
                    if step < inc_full*min_substep_ratio - 1.e-15:
                        failed = True
                        break
            #
            if failed:
                break
            #
            #save average stress and strain at the original increment boundary
            Pavg = self.__average(P_boundary)
            self.Ps.append(Pavg)
            print("now P is...")
            print(Pavg)
            Favg = self.__average(F)
            self.Fs.append(Favg)
            print("now F is...")
            print(Favg)
            #
            # incremental save so an interrupted or failed run keeps
            # everything up to the last completed increment boundary
            if savemodel == "normal" or savemodel == "both":
                self.__save_F_P(self.path)
                self.__save_stats(self.path)
                print("intermediate results saved ({} increments in output.csv)".format(len(self.Ps)))
            #
        #-------------------------------post
        if failed:
            self.solver_status = "failed"
            print("SOLVER FAILED: no convergence at load {:.4g} even with the".format(inc_tol))
            print("minimum sub-step; results are only saved up to the last")
            print("converged increment boundary.")
        elif self.solver_stats["step_cuts"] > 0:
            self.solver_status = "converged_with_step_cuts"
            print("finish! (with {} load-step cuts)".format(self.solver_stats["step_cuts"]))
        else:
            self.solver_status = "converged"
            print("finish!")
        self.solver_stats["status"] = self.solver_status
        self.__save_stats(self.path)
        print("solver stats are saved to solver_stats.json")
        #
        # keep the final local fields reachable by callers: the averages alone
        # cannot show whether F is a compatible (gradient) field, which is the
        # diagnostic that distinguishes a physical solution from one polluted
        # with null-space content (see docs/green_reference_preconditioning.md)
        self.F_final = F
        self.P_final = P_boundary
        self.pressure_final = YALI
        #
        #
        if savemodel == "normal" or savemodel == "both":
            #save Fs and Ps
            self.__save_F_P(self.path)
            print("F and P are saved to output.csv")
        if save_fields:
            field_file = save_vti_cell_fields(
                self.path,
                solution_fields(F, P_boundary if P_boundary is not None else np.zeros_like(F), phase, pressure=YALI),
                filename=field_filename,
            )
            print("local fields are saved to {}".format(field_file))
    #=============================================================================

    def __save_stats(self, path):
        ensure_output_path(path)
        outfile = os.path.join(path, "solver_stats.json")
        with open(outfile, "w") as file:
            json.dump(self.solver_stats, file, indent=1, default=float)

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
