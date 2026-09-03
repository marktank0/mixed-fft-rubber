"""Exact rank analysis of the mixed Newton operator (preconditioner diagnostic).

Forms the real operator A from FFT_simulation/fg/mxfft.py densely at N=7 and answers, exactly:

  - is A singular, and how large is its null space?
  - is the null space made of incompatible (non-gradient) F fields?
  - is A nonsingular once restricted to the compatible subspace?
  - does the reference preconditioner map the compatible subspace to itself?
"""
import os
import tempfile

import numpy as np
import scipy.fft
import scipy.sparse.linalg as spla

from project_paths import charge_path, ensure_import_paths

ensure_import_paths()

import fg.mxfft as mx

N = 7
CHARGE = charge_path("Neo_1.0_E10-1000.txt")            # contrast 100
PHASE = os.path.join(tempfile.mkdtemp(prefix="fg_rank_"), "small_phase.npz")

rng = np.random.default_rng(0)
phase = np.zeros((N, N, N))
phase[2:5, 2:5, 2:5] = 1.0                      # a filler block, phi = 27/343 = 7.9%
np.savez(PHASE, phase=phase)
print("N = {}, filler fraction = {:.3f}".format(N, phase.mean()))

captured = {}


class Abort(Exception):
    pass


class Shim:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def gmres(self, **kw):
        captured.update(kw)
        raise Abort


mx.sp = Shim(spla)
prob = mx.FFTSolver(PHASE, charge_path=CHARGE, output_path=os.path.dirname(PHASE), N=N, output_name=".")
try:
    prob.calculate(incre_list=[0.05], preconditioner="green",
                   reference="mean", forcing="fixed", inner_rtol=1e-6)
except Abort:
    pass

A = captured["A"]
M = captured["M"]
n = A.shape[0]
num_up = 9*N**3
print("operator size {} (F-block {}, p-block {})".format(n, num_up, N**3))

# ---- densify A and M
def densify(op):
    D = np.empty((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0
        D[:, j] = op.matvec(e)
        e[j] = 0.0
    return D


print("densifying A ...", flush=True)
Ad = densify(A)
print("densifying M^-1 ...", flush=True)
Md = densify(M)

# ---- compatible-subspace projector on the F-block
axes = (-3, -2, -1)
fft = lambda x: np.fft.fftshift(scipy.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
ifft = lambda x: np.fft.fftshift(scipy.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
Ghat4 = mx.build_Ghat4(N, prob.pb.stress_control, 3)
Gop = lambda A2: np.real(ifft(mx.ddot42(Ghat4, fft(A2))))


def Pi(vec):
    """Project the F-block onto compatible fields; leave the p-block alone."""
    out = vec.copy()
    out[:num_up] = Gop(vec[:num_up].reshape(3, 3, N, N, N)).reshape(-1)
    return out


print("densifying the compatibility projector ...", flush=True)
Pid = densify(spla.LinearOperator((n, n), matvec=Pi, dtype=float))

tol = 1e-9
sA = np.linalg.svd(Ad, compute_uv=False)
rank_A = int((sA > tol*sA[0]).sum())
sP = np.linalg.svd(Pid, compute_uv=False)
dim_compat = int((sP > tol*sP[0]).sum())

print("\n" + "=" * 68)
print("rank(A)                         = {:6d}  of {}".format(rank_A, n))
print("dim null(A)                     = {:6d}".format(n - rank_A))
print("dim of compatible subspace      = {:6d}  (F-block {} + p-block {})".format(
    dim_compat, dim_compat - N**3, N**3))
print("=" * 68)

# ---- is A nonsingular ON the compatible subspace?
U, s, Vt = np.linalg.svd(Pid)
basis = U[:, :dim_compat]                     # orthonormal basis of the compatible subspace
A_restricted = basis.T @ Ad @ basis
sR = np.linalg.svd(A_restricted, compute_uv=False)
rank_R = int((sR > tol*sR[0]).sum())
print("rank(A restricted to compatible subspace) = {} of {}  -> {}".format(
    rank_R, dim_compat, "NONSINGULAR" if rank_R == dim_compat else "singular"))
print("  smallest singular value of restriction  = {:.3e} (largest {:.3e})".format(sR[-1], sR[0]))

# ---- does null(A) meet the compatible subspace?
# NB: it is NOT enough to look at individual SVD null-basis vectors - an
# arbitrary basis of null(A) can have a large projection onto the compatible
# subspace without any of it LYING in the subspace. The right test is whether
# A is injective there, i.e. whether the restriction is full rank.
if rank_R == dim_compat:
    print("\nnull(A) ∩ compatible = {0}  (the restriction above is injective)")
    print("  -> the null space is entirely NON-PHYSICAL: incompatible F fields")
else:
    print("\nnull(A) ∩ compatible has dimension {} -> the physical problem is"
          " itself under-determined".format(dim_compat - rank_R))

# ---- does M^-1 preserve the compatible subspace?
print("\ndoes M^-1 map the compatible subspace into itself?")
leak = []
for k in range(0, dim_compat, max(1, dim_compat//60)):
    v = basis[:, k]
    w = Md @ v
    leak.append(np.linalg.norm(w - Pi(w))/max(np.linalg.norm(w), 1e-300))
leak = np.array(leak)
print("  ||M^-1 v - Pi(M^-1 v)|| / ||M^-1 v||  over compatible v:")
print("    min {:.3e}  median {:.3e}  max {:.3e}".format(leak.min(), np.median(leak), leak.max()))
print("  -> M^-1 LEAVES the compatible subspace" if np.median(leak) > 1e-8
      else "  -> M^-1 preserves the compatible subspace")
