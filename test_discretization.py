"""Validate the Willot rotated-scheme projector against its defining properties."""
import itertools
import numpy as np
import scipy.fft

from fg.preconditioning import build_Ghat4, _wave_vectors

N = 9
ndim = 3
SC = [(1, 1), (2, 2)]
axes = (-3, -2, -1)
fft = lambda x: np.fft.fftshift(scipy.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
ifft = lambda x: np.fft.fftshift(scipy.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
ddot42 = lambda A4, B2: np.einsum('ijklxyz,klxyz->ijxyz', A4, B2)

Gf = build_Ghat4(N, SC, ndim, "fourier")
Gw = build_Ghat4(N, SC, ndim, "willot")
print("fourier dtype {}   willot dtype {}".format(Gf.dtype, Gw.dtype))

# ---- 1. the fourier path must reproduce the previous implementation bitwise
freq = np.arange(-(N-1)/2., +(N+1)/2.)
q = np.stack(np.meshgrid(freq, freq, freq, indexing="ij"))
q2 = np.einsum("kxyz,kxyz->xyz", q, q)
zf = (q2 == 0)
q2safe = np.where(zf, 1.0, q2)
QQ = np.einsum("jxyz,mxyz->jmxyz", q, q)/q2safe
ref = np.einsum("il,jmxyz->ijlmxyz", np.eye(ndim), QQ)
ref[:, :, :, :, zf] = 0.0
for (i, j) in SC:
    ref[i, j, i, j, zf] = 1.0
assert np.array_equal(Gf, ref), "fourier path changed!"
print("1. fourier reproduces the previous build_Ghat4 BITWISE: OK")

# ---- 1b. and the old fft.py inline loop, too
ref2 = np.zeros([ndim]*4 + [N, N, N])
for i, j, l, m in itertools.product(range(ndim), repeat=4):
    for x, y, z in itertools.product(range(N), repeat=3):
        qq = np.array([freq[x], freq[y], freq[z]])
        if not qq.dot(qq) == 0:
            ref2[i, j, l, m, x, y, z] = float(i == l)*qq[j]*qq[m]/(qq.dot(qq))
        elif (i, j) in SC:
            ref2[i, j, l, m, x, y, z] = float(i == l)*float(j == m)
assert np.array_equal(Gf, ref2), "differs from the old fft.py inline loop!"
print("1b. matches the old fft.py inline loop BITWISE: OK")

# ---- 2. the real-space stencil IS a rotated finite difference
# xi_j is degree 1 in each e^{i k_m h}, so its inverse transform must be
# supported on a single 2x2x2 voxel block: +1 on the four corners ahead in j,
# -1 on the four behind. That is Willot's diagonal forward difference.
xi = _wave_vectors(N, ndim, "willot")
k = np.stack(np.meshgrid(freq, freq, freq, indexing="ij"))
stencil = np.fft.ifftn(np.fft.ifftshift(xi[0], axes=(0, 1, 2)), axes=(0, 1, 2))
stencil = np.real_if_close(stencil, tol=1e6)
stencil = stencil/np.abs(stencil).max()                  # coefficients up to scale
support = np.argwhere(np.abs(stencil) > 1e-9)
vals = np.array([stencil[tuple(p)].real for p in support])
offs = [tuple(int(v) if int(v) <= 1 else int(v)-N for v in p) for p in support]
print("2. real-space stencil of xi_1: {} non-zero voxels".format(len(support)))
print("   offsets (wrapped) {}".format(sorted(offs)))
print("   values            {}".format(sorted(set(np.round(vals, 9)))))
assert len(support) == 8, "a finite difference must have local (2x2x2) support"
assert {tuple(sorted(set(c))) for c in zip(*offs)} == {(-1, 0)}, "support is not a 2x2x2 cube"
assert set(np.round(vals, 9)) == {-1.0, 1.0}
plus = {o[0] for o, v in zip(offs, vals) if v > 0}
minus = {o[0] for o, v in zip(offs, vals) if v < 0}
assert len(plus) == 1 and len(minus) == 1 and plus != minus, \
    "the sign must split along axis 0 only"
print("   -> +1 on the 4 corners at x-offset {}, -1 at {}: a forward difference"
      " along axis 0 averaged over the 4 transverse corners: OK".format(plus.pop(), minus.pop()))

# ---- 2b. the PROJECTOR converges to the spectral one as k*h -> 0
print("2b. ||G_willot - G_fourier|| at the lowest frequencies, refining the grid:")
prev = None
for Nn in (9, 17, 33, 65):
    a = build_Ghat4(Nn, SC, ndim, "fourier")
    b = build_Ghat4(Nn, SC, ndim, "willot")
    c = Nn//2
    sl = (slice(None),)*4 + (slice(c-1, c+2),)*3      # the 3x3x3 lowest modes
    d = np.abs(b[sl] - a[sl]).max()
    print("     N = {:>3}  max|dG| = {:.4e}{}".format(
        Nn, d, "" if prev is None else "   ratio {:.2f}".format(d/prev)))
    prev = d
assert prev < 1e-2, "projector should converge to the spectral one under refinement"

# ---- 3. Hermitian and idempotent (as a 9x9 matrix per frequency)
G9w = Gw.reshape(9, 9, N, N, N)
herm = np.abs(G9w - np.conj(np.swapaxes(G9w, 0, 1))).max()
idem = np.abs(np.einsum("abxyz,bcxyz->acxyz", G9w, G9w) - G9w).max()
print("3. Hermitian error {:.3e}   idempotency error {:.3e}".format(herm, idem))
assert herm < 1e-12 and idem < 1e-12

# ---- 4. conjugate symmetry -> G maps real fields to real fields
rng = np.random.default_rng(0)
r = rng.standard_normal((3, 3, N, N, N))
out = ifft(ddot42(Gw, fft(r)))
print("4. imaginary part of G(real field): {:.3e} (relative {:.3e})".format(
    np.abs(out.imag).max(), np.abs(out.imag).max()/np.abs(out.real).max()))
assert np.abs(out.imag).max()/np.abs(out.real).max() < 1e-12

# ---- 5. projects Willot-gradient fields onto themselves
u = rng.standard_normal((3, N, N, N))
uh = fft(u)
Fh = np.einsum("ixyz,jxyz->ijxyz", uh, xi)          # F_ij = xi_j u_i  (Willot gradient)
Fh[:, :, (np.abs(k).sum(axis=0) == 0)] = 0.0        # drop the zero mode
Fw = np.real(ifft(Fh))
GFw = np.real(ifft(ddot42(Gw, fft(Fw))))
print("5. ||G(F) - F||/||F|| for a Willot-gradient field = {:.3e}".format(
    np.linalg.norm(GFw - Fw)/np.linalg.norm(Fw)))
assert np.linalg.norm(GFw - Fw)/np.linalg.norm(Fw) < 1e-12

# ---- 6. a Fourier-gradient field is NOT a Willot-gradient field (schemes differ)
Fh2 = 1j*np.einsum("ixyz,jxyz->ijxyz", uh, k)
Fh2[:, :, (np.abs(k).sum(axis=0) == 0)] = 0.0
Ff = np.real(ifft(Fh2))
print("6. ||G_willot(F_fourier) - F_fourier||/||F|| = {:.3e}  (nonzero: the discretizations differ)".format(
    np.linalg.norm(np.real(ifft(ddot42(Gw, fft(Ff)))) - Ff)/np.linalg.norm(Ff)))

print("\nALL WILLOT PROJECTOR CHECKS PASSED")
