# -*- coding: utf-8 -*-
"""Displacement-based mixed FFT solver (DBFFT).

Reformulates the mixed problem with the displacement fluctuation u as the
primary unknown instead of the deformation gradient F:

    F(x) = Fbar + grad u(x)

The point is structural, not cosmetic. In the F-based formulation the unknown
must satisfy a compatibility constraint, the operator is therefore singular
off the compatible subspace (rank 1371 of 3430 at N=7), and the preconditioner
has to be restricted to keep Krylov iterates inside that subspace - which
costs a factor O(contrast) in conditioning and blocks the published
high-contrast preconditioners. See docs/green_reference_preconditioning.md.

Writing F as a gradient makes compatibility automatic:

  * every u is admissible, so the operator is full rank (up to the 3 rigid
    translations, which are removed by fixing <u> = 0);
  * the reference operator per frequency is the acoustic tensor
    Gamma_ik = conj(xi_j) K0_ijkl xi_l, a 3x3 Hermitian block that is
    invertible for a strongly elliptic reference - no pseudo-inverse of a
    rank-deficient symbol;
  * the unknown count drops from 9 N^3 to 3 N^3 (+ N^3 pressures).

Lucarini & Segurado (2019), "DBFFT: a displacement based FFT approach for
non-linear homogenization of the mechanical behavior", Int. J. Eng. Sci.

Unknown vector layout:  [ u (3 N^3) | p (N^3) | free Fbar components ]
"""

import importlib.util
import json
import os
import time

import numpy as np
import scipy.fft
import scipy.sparse.linalg as sp

from fg.io_paths import (
    default_charge_path,
    ensure_output_path,
    output_run_path,
    phase_source,
)
from fg.mxfft import (
    Problem,
    eisenstat_walker_forcing,
    format_scientific_cut,
    load_phase,
    load_umat,
    mat2vec9,
    save_output_csv,
)
from fg.preconditioning import DISCRETIZATIONS, REFERENCE_MODES, reference_average
from fg.vtk_export import save_vti_cell_fields, solution_fields

_FFT_WORKERS = int(os.environ.get("FFT_WORKERS", "1"))
_AXES = (-3, -2, -1)


# --------------------------------------------------------------- transforms
def _fft(x):
    return np.fft.fftshift(
        scipy.fft.fftn(np.fft.ifftshift(x, axes=_AXES), axes=_AXES, workers=_FFT_WORKERS),
        axes=_AXES)


def _ifft(x):
    return np.fft.fftshift(
        scipy.fft.ifftn(np.fft.ifftshift(x, axes=_AXES), axes=_AXES, workers=_FFT_WORKERS),
        axes=_AXES)


def derivative_symbol(N, ndim=3, discretization="fourier"):
    """The TRUE (complex) derivative symbol xi_j, with the cell length L = 1.

    Unlike the projection operator in the F-based solver - where the factor i
    and any common scale cancel in xi_j xi_m / |xi|^2 - the displacement
    formulation applies xi_j and conj(xi_j) separately, so both the factor i
    and the physical scaling matter and are kept here.

    Satisfies xi(-k) = conj(xi(k)), so grad/div of a real field stay real.
    """
    if discretization not in DISCRETIZATIONS:
        raise ValueError("Unknown discretization {!r}; use one of {}.".format(
            discretization, DISCRETIZATIONS))

    k = np.arange(-(N - 1)/2., +(N + 1)/2.)
    grid = np.stack(np.meshgrid(*([k]*ndim), indexing="ij"))

    if discretization == "fourier":
        return 2j*np.pi*grid                      # d/dx_j  ->  i (2 pi k_j)

    h = 1.0/N
    phase = np.exp(2j*np.pi*grid/float(N))
    xi = np.empty(grid.shape, dtype=complex)
    for j in range(ndim):
        term = phase[j] - 1.0
        for m in range(ndim):
            if m != j:
                term = term*(phase[m] + 1.0)
        xi[j] = term/(4.0*h)                      # Willot rotated difference
    return xi


def grad(u, xi):
    """F_ij = d u_i / d x_j, spectrally. Zero mean by construction."""
    uh = _fft(u)
    return np.real(_ifft(np.einsum("ixyz,jxyz->ijxyz", uh, xi)))


def div_adj(S, xi):
    """The adjoint of grad: r_i = conj(xi_j) S_ij.

    <S, grad u> = <div_adj S, u> exactly, which is what makes the discrete
    equilibrium operator symmetric for a symmetric tangent.
    """
    Sh = _fft(S)
    return np.real(_ifft(np.einsum("jxyz,ijxyz->ixyz", np.conj(xi), Sh)))


# ------------------------------------------------------------ preconditioner
def build_acoustic_preconditioner(xi, K_ref, J_ref, kappa_inv_ref, rcond=1.0e-10):
    """Inverse of the reference operator, per frequency.

    For each xi the reference block is

        [ Gamma_ik   c_i    ]     Gamma_ik = conj(xi_j) K0_ijkl xi_l
        [ c_k^H     -alpha0 ]     c_i      = conj(xi_j) H0_ij

    Gamma is the acoustic tensor of the reference material: 3x3, Hermitian for
    a tangent with major symmetry, and positive definite whenever the reference
    is strongly elliptic. That is the whole advantage of this formulation - the
    block is genuinely invertible rather than rank-deficient.
    """
    ndim, N = xi.shape[0], xi.shape[-1]
    K = K_ref.reshape(ndim, ndim, ndim, ndim)
    xic = np.conj(xi)

    gamma = np.einsum("jxyz,ijkl,lxyz->ikxyz", xic, K, xi)
    coupling = np.einsum("jxyz,ij->ixyz", xic, J_ref.reshape(ndim, ndim))

    n = ndim + 1
    block = np.zeros((n, n) + (N,)*ndim, dtype=complex)
    block[:ndim, :ndim] = gamma
    block[:ndim, ndim] = coupling
    block[ndim, :ndim] = np.conj(coupling)
    block[ndim, ndim] = -float(kappa_inv_ref)

    stack = np.moveaxis(block, (0, 1), (-2, -1)).reshape(-1, n, n)
    inv = np.linalg.pinv(stack, rcond=rcond)
    inv = np.moveaxis(inv.reshape((N,)*ndim + (n, n)), (-2, -1), (0, 1))

    # zero frequency: xi = 0, so Gamma vanishes. <u> is pinned to 0 and the
    # macroscopic pressure mode is left to the Krylov solver.
    centre = (slice(None), slice(None)) + (N//2,)*ndim
    inv[centre] = 0.0
    return inv


def apply_acoustic_preconditioner(u_flat, p_flat, inv_block, ndim=3):
    N = inv_block.shape[-1]
    stacked = np.concatenate([u_flat.reshape(ndim, N, N, N),
                              p_flat.reshape(1, N, N, N)], axis=0)
    hat = _fft(stacked)
    out = np.real(_ifft(np.einsum("abxyz,bxyz->axyz", inv_block, hat)))
    return out[:ndim].reshape(-1), out[ndim].reshape(-1)


# ------------------------------------------------------------------- solver
class DBFFTSolver:
    """Displacement-based mixed FFT solver."""

    def __init__(self, structure_path, charge_path=None, output_path=None, N=31,
                 phase_path=None, phase_key="phase", output_name=None):
        self.structure_path = structure_path
        self.charge_path = charge_path or default_charge_path(structure_path)
        self.output_path = ensure_output_path(
            output_run_path(structure_path, output_path, output_name))
        self.path = self.output_path
        self.pb = Problem(self.charge_path)
        phase_dir, selected = phase_source(structure_path, phase_path)
        self.phase, self.phase_path = load_phase(phase_dir, N, selected, phase_key)
        self.N = N
        self.iter_num = 0
        self.solver_status = "not_run"
        self.solver_stats = {}
        print("---------------------------------------------")
        print("-- displacement-based mixed FFT solver (DBFFT)")
        print("---------------------------------------------")

    def calculate(self, increment=10, incre_list=(), savemodel="no",
                  preconditioner="reference", diagnostics=False, save_fields=False,
                  field_filename="fields.vti", tol_rel=1.e-5, tol_abs=1.e-10,
                  max_gmres_iter=20000, max_backtracks=8, min_substep_ratio=1.0/16.0,
                  gmres_restart=None, reference="mean", forcing="eisenstat_walker",
                  inner_rtol=1.e-6, eta_max=1.e-2, eta_min=1.e-3, ew_gamma=0.9,
                  ew_alpha=2.0, discretization="fourier"):
        if reference not in REFERENCE_MODES:
            raise ValueError("Unknown reference {!r}".format(reference))
        if forcing not in ("fixed", "eisenstat_walker"):
            raise ValueError("Unknown forcing {!r}".format(forcing))

        ndim, N = 3, self.N
        n_u, n_p = ndim*N**3, N**3
        free = [(i, j) for (i, j) in self.pb.stress_control]
        n_f = len(free)
        n_tot = n_u + n_p + n_f
        if not len(incre_list):
            incre_list = [0.1]*(increment - 1)
        if gmres_restart is None:
            gmres_restart = int(min(200, max(20, 8.e8/(8.0*n_tot))))
        print("gmres restart length: {}".format(gmres_restart))

        xi = derivative_symbol(N, ndim, discretization)
        print("derivative symbol built ({} discretization)".format(discretization))

        phase = self.phase
        mask_a, mask_b = (phase == 0), (phase == 1)
        if not np.all(mask_a | mask_b):
            raise ValueError("Phase field contains values other than 0 and 1")

        consti_a = load_umat(os.path.join("fg/constitutive_incompressible",
                                          self.pb.model_a_name), "umat_field")
        consti_b = load_umat(os.path.join("fg/constitutive_incompressible",
                                          self.pb.model_b_name), "umat_field")
        par_a, par_b = self.pb.model_a_para, self.pb.model_b_para

        def constitutive(F, p, need_tangent=True):
            P = np.zeros([ndim, ndim, N, N, N])
            K4 = np.zeros([ndim]*4 + [N, N, N]) if need_tangent else None
            JFinv = np.zeros([ndim, ndim, N, N, N])
            kap = np.zeros([N, N, N])
            for mask, cons, para in ((mask_a, consti_a, par_a), (mask_b, consti_b, par_b)):
                if not mask.any():
                    continue
                fv = np.moveaxis(F[:, :, mask], -1, 0)
                pp, k4, jf, ki = cons(fv, p[mask], para, need_tangent=need_tangent)
                P[:, :, mask] = np.moveaxis(pp, 0, -1)
                if need_tangent:
                    K4[:, :, :, :, mask] = np.moveaxis(k4, 0, -1)
                JFinv[:, :, mask] = np.moveaxis(jf, 0, -1)
                kap[mask] = ki
            return P, K4, JFinv, kap

        det = lambda F: np.linalg.det(np.moveaxis(F.reshape(ndim, ndim, -1), -1, 0))

        dFbar = np.zeros((ndim, ndim))
        dPbar = np.zeros((ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                dFbar[i, j] = self.pb.dF[ndim*i + j]
                dPbar[i, j] = self.pb.dP[ndim*i + j]

        # Block scaling. div_adj carries a factor |xi| ~ 2 pi N/2, so the raw
        # equilibrium residual is ~1000x the incompressibility residual at
        # N=31. GMRES minimises the total norm, so without scaling it
        # satisfies equilibrium and ignores incompressibility, producing a
        # useless Newton direction. Dividing the equilibrium (and macroscopic
        # stress) equations by a stress scale sigma, and carrying the pressure
        # as p = sigma * p_tilde, makes every block O(1).
        state = {"sigma": 1.0}

        def stress_scale(K4):
            s = float(np.sqrt(np.einsum("ijkl...,ijkl...->...", K4, K4).mean()))
            return s if s > 0.0 else 1.0

        def assemble_F(u, Fbar):
            return Fbar[:, :, None, None, None] + grad(u, xi)

        def residual(u, p, Fbar, Pbar_target):
            F = assemble_F(u, Fbar)
            P, K4, JFmT, kap = constitutive(F, p)
            state.update(K4=K4, JFmT=JFmT, kap=kap, F=F, P=P)
            sigma = state["sigma"]
            r = np.empty(n_tot)
            r[:n_u] = -div_adj(P, xi).reshape(-1)/sigma
            r[n_u:n_u + n_p] = (1.0 - det(F).reshape(N, N, N) + p*kap).reshape(-1)
            Pavg = P.mean(axis=(2, 3, 4))
            for a, (i, j) in enumerate(free):
                r[n_u + n_p + a] = (Pbar_target[i, j] - Pavg[i, j])/sigma
            return r

        def matvec(dX):
            K4, JFmT, kap = state["K4"], state["JFmT"], state["kap"]
            sigma = state["sigma"]
            du = dX[:n_u].reshape(ndim, N, N, N)
            dp = dX[n_u:n_u + n_p].reshape(N, N, N)*sigma      # p = sigma * p_tilde
            dFb = np.zeros((ndim, ndim))
            for a, (i, j) in enumerate(free):
                dFb[i, j] = dX[n_u + n_p + a]

            dF = grad(du, xi) + dFb[:, :, None, None, None]
            dP = np.einsum("ijklxyz,klxyz->ijxyz", K4, dF) \
                + np.einsum("ijxyz,xyz->ijxyz", JFmT, dp)

            out = np.empty(n_tot)
            out[:n_u] = div_adj(dP, xi).reshape(-1)/sigma
            out[n_u:n_u + n_p] = (np.einsum("ijxyz,ijxyz->xyz", JFmT, dF)
                                  - kap*dp).reshape(-1)
            dPavg = dP.mean(axis=(2, 3, 4))
            for a, (i, j) in enumerate(free):
                out[n_u + n_p + a] = dPavg[i, j]/sigma
            return out

        def solve_linear(b, rtol):
            self.iter_num = 0
            Aop = sp.LinearOperator((n_tot, n_tot), matvec=matvec, dtype=float)
            Mop = None
            if preconditioner == "reference":
                K_ref = reference_average(state["K4"], reference, mask_a, mask_b)
                J_ref = reference_average(state["JFmT"], reference, mask_a, mask_b)
                kap_ref = float(reference_average(state["kap"], reference, mask_a, mask_b))
                sigma = state["sigma"]
                inv_block = build_acoustic_preconditioner(
                    xi, K_ref/sigma, J_ref, kap_ref*sigma)

                def apply_M(vec):
                    out = vec.copy()
                    u_out, p_out = apply_acoustic_preconditioner(
                        vec[:n_u], vec[n_u:n_u + n_p], inv_block, ndim)
                    out[:n_u] = u_out
                    out[n_u:n_u + n_p] = p_out
                    return out

                Mop = sp.LinearOperator((n_tot, n_tot), matvec=apply_M, dtype=float)

            def cb(_r):
                self.iter_num += 1

            cycles = max(1, int(np.ceil(max_gmres_iter/float(gmres_restart))))
            dX, flag = sp.gmres(A=Aop, b=b, M=Mop, rtol=rtol, atol=1.e-10,
                                restart=gmres_restart, maxiter=cycles,
                                callback=cb, callback_type="pr_norm")
            return dX, flag

        def newton(u0, p0, Fbar0, Pbar_target):
            u, p, Fbar = u0.copy(), p0.copy(), Fbar0.copy()
            # fix the stress scale for this sub-increment so the merit function
            # stays comparable across line-search trials
            state["sigma"] = 1.0
            residual(u, p, Fbar, Pbar_target)
            state["sigma"] = stress_scale(state["K4"])
            r = residual(u, p, Fbar, Pbar_target)
            n_eq = np.linalg.norm(r[:n_u])
            n_ic = np.linalg.norm(r[n_u:n_u + n_p])
            ref_eq, ref_ic = max(n_eq, tol_abs), max(n_ic, tol_abs)
            merit = np.hypot(n_eq/ref_eq, n_ic/ref_ic)
            stats = {"newton_iterations": 0, "krylov_iterations": [], "alphas": [],
                     "forcing_terms": [], "residual_eq": [n_eq], "residual_ic": [n_ic],
                     "fail_reason": None}
            eta = eta_max if forcing == "eisenstat_walker" else inner_rtol

            def converged():
                return (n_eq <= max(tol_rel*ref_eq, tol_abs)
                        and n_ic <= max(tol_rel*ref_ic, tol_abs))

            while not converged():
                stats["forcing_terms"].append(eta)
                dX, flag = solve_linear(r, eta)
                stats["krylov_iterations"].append(self.iter_num)
                if flag > 0:
                    stats["fail_reason"] = "gmres_iteration_cap"
                    return False, u, p, Fbar, stats

                du = dX[:n_u].reshape(ndim, N, N, N)
                du -= du.mean(axis=(1, 2, 3))[:, None, None, None]   # kill translation
                dp = dX[n_u:n_u + n_p].reshape(N, N, N)*state["sigma"]
                dFb = np.zeros((ndim, ndim))
                for a, (i, j) in enumerate(free):
                    dFb[i, j] = dX[n_u + n_p + a]

                J_floor = 0.05*det(state["F"]).min()
                alpha, accepted = 1.0, False
                for _ in range(max_backtracks):
                    ut, pt, Fbt = u + alpha*du, p + alpha*dp, Fbar + alpha*dFb
                    if det(assemble_F(ut, Fbt)).min() <= J_floor:
                        alpha *= 0.5
                        continue
                    rt = residual(ut, pt, Fbt, Pbar_target)
                    e, c = np.linalg.norm(rt[:n_u]), np.linalg.norm(rt[n_u:n_u + n_p])
                    mt = np.hypot(e/ref_eq, c/ref_ic)
                    if mt < merit:
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    stats["fail_reason"] = "line_search"
                    return False, u, p, Fbar, stats

                u, p, Fbar, r = ut, pt, Fbt, rt
                merit_prev, merit = merit, mt
                n_eq, n_ic = e, c
                residual(u, p, Fbar, Pbar_target)          # refresh the tangent
                if forcing == "eisenstat_walker":
                    eta = eisenstat_walker_forcing(merit, merit_prev, eta,
                                                   gamma=ew_gamma, alpha=ew_alpha,
                                                   eta_min=eta_min, eta_max=eta_max)
                stats["newton_iterations"] += 1
                stats["alphas"].append(alpha)
                stats["residual_eq"].append(n_eq)
                stats["residual_ic"].append(n_ic)
                print("res eq {:.3e} ic {:.3e} alpha {:.3g} eta {:.2e} krylov {}".format(
                    n_eq/ref_eq, n_ic/ref_ic, alpha, eta, self.iter_num))
            return True, u, p, Fbar, stats

        # ---- load stepping
        u = np.zeros((ndim, N, N, N))
        p = np.zeros((N, N, N))
        Fbar = np.eye(ndim)
        self.Ps, self.Fs = [], []
        self.solver_status = "in_progress"
        self.solver_stats = {"status": "in_progress", "tol_rel": tol_rel,
                             "max_gmres_iter": max_gmres_iter, "formulation": "dbfft",
                             "preconditioner": preconditioner, "reference": reference,
                             "discretization": discretization, "forcing": forcing,
                             "step_cuts": 0, "increments": []}

        inc_tol, failed = 0.0, False
        targets = np.cumsum(incre_list)
        step, easy = float(incre_list[0]), 0
        for k in range(len(targets)):
            target, inc_full = float(targets[k]), float(incre_list[k])
            step = min(step, inc_full)
            while inc_tol < target - 1.e-12:
                step = min(step, target - inc_tol)
                print("increment {} (target {}) ---------------".format(step, target))
                t0 = time.time()
                Fbar_trial = Fbar.copy()
                for i in range(ndim):
                    for j in range(ndim):
                        if (i, j) not in free:
                            Fbar_trial[i, j] = Fbar[i, j] + step*dFbar[i, j]
                ok, u_n, p_n, Fb_n, st = newton(u, p, Fbar_trial,
                                                (inc_tol + step)*dPbar)
                st.update(step=step, load_start=inc_tol, converged=ok,
                          time_seconds=time.time() - t0)
                self.solver_stats["increments"].append(st)
                if ok:
                    u, p, Fbar = u_n, p_n, Fb_n
                    inc_tol += step
                    easy = easy + 1 if st["newton_iterations"] <= 4 else 0
                    if easy >= 2:
                        step = min(step*1.5, inc_full)
                    print("time this step... {:.1f}".format(st["time_seconds"]))
                else:
                    easy = 0
                    step *= 0.5
                    self.solver_stats["step_cuts"] += 1
                    print("increment failed ({}); cutting to {:.4g}".format(
                        st["fail_reason"], step))
                    if step < inc_full*min_substep_ratio - 1.e-15:
                        failed = True
                        break
            if failed:
                break
            F = assemble_F(u, Fbar)
            P, _, _, _ = constitutive(F, p, need_tangent=False)
            self.Ps.append(P.mean(axis=(2, 3, 4)))
            self.Fs.append(F.mean(axis=(2, 3, 4)))
            print("now P is...\n{}".format(self.Ps[-1]))
            print("now F is...\n{}".format(self.Fs[-1]))
            if savemodel in ("normal", "both"):
                self.__save(self.path)

        self.solver_status = ("failed" if failed else
                              "converged_with_step_cuts" if self.solver_stats["step_cuts"]
                              else "converged")
        self.solver_stats["status"] = self.solver_status
        self.F_final = assemble_F(u, Fbar)
        self.P_final = state.get("P")
        self.pressure_final = p
        self.__save(self.path)
        print("finish! ({})".format(self.solver_status))
        if save_fields:
            save_vti_cell_fields(self.path, solution_fields(
                self.F_final, self.P_final, phase, pressure=p), filename=field_filename)

    def __save(self, path):
        ensure_output_path(path)
        with open(os.path.join(path, "solver_stats.json"), "w") as fh:
            json.dump(self.solver_stats, fh, indent=1, default=float)
        if not self.Ps:
            return
        n = len(self.Ps)
        out = np.hstack((np.array([mat2vec9(f) for f in self.Fs]).reshape(n, 9),
                         np.array([mat2vec9(p) for p in self.Ps]).reshape(n, 9)))
        header = ",".join(["F11", "F12", "F13", "F21", "F22", "F23", "F31", "F32", "F33",
                           "P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"])
        save_output_csv(path, out, header)


FFTSolver = DBFFTSolver          # drop-in name for the benchmark harness
