"""Decisive correctness test for the preconditioner fix.

On a HOMOGENEOUS body the reference tangent equals the true tangent, so the
restricted symbol Ghat K0 Ghat IS the operator restricted to the compatible
subspace. Therefore M^-1 A must act as the identity there, and GMRES must
converge in ~1 iteration. The unrestricted symbol Ghat K0 is not the operator
and cannot have this property.

This does not depend on any claim about which answer is right - it is a
statement about the preconditioner that must hold if the fix is correct.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "FFT_WORKERS"):
    os.environ.setdefault(v, "1")

import tempfile

import numpy as np
import scipy.fft
import scipy.sparse.linalg as spla

from project_paths import charge_path, ensure_import_paths, project_path

ensure_import_paths()
import fg.mxfft as mx  # noqa: E402  (needs ensure_import_paths first)

N = 31
SCRATCH = os.path.join(tempfile.gettempdir(), "precond_test")
os.makedirs(SCRATCH, exist_ok=True)
HOMO = os.path.join(SCRATCH, "charge_homogeneous.txt")
VOXEL = project_path("3D_samples", "voxels", "1_voxel.npz")

# identical E *and* identical Poisson ratio in both phases -> truly homogeneous
with open(HOMO, "w") as fh:
    fh.write("#homogeneous: both phases identical\n")
    fh.write("1.0\t10\t0.48\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n")
    fh.write("1.0\t10\t0.48\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n")
    fh.write("#charge dF\n")
    fh.write("1.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n")
    fh.write("#type\n")
    fh.write("0.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\n")

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

axes = (-3, -2, -1)
fft = lambda x: np.fft.fftshift(scipy.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
ifft = lambda x: np.fft.fftshift(scipy.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
num_up = 9*N**3


def capture(restrict, charge):
    captured.clear()
    prob = mx.FFTSolver(VOXEL, charge_path=charge,
                        output_path=os.path.join(SCRATCH, "homchk"), N=N, output_name=".")
    try:
        prob.calculate(incre_list=[0.1], preconditioner="green",
                       reference="mean", forcing="fixed", precond_restrict=restrict)
    except Abort:
        pass
    return captured["A"], captured["M"], captured["b"], prob.pb.stress_control


def report(label, charge):
    print("\n" + "=" * 74)
    print(label)
    print("=" * 74)
    errors = {}
    for restrict in (False, True):
        A, M, b, sc = capture(restrict, charge)
        Ghat4 = mx.build_Ghat4(N, sc, 3)
        G = lambda a: np.real(ifft(mx.ddot42(Ghat4, fft(a))))

        # a random COMPATIBLE test vector
        rng = np.random.default_rng(0)
        x = rng.standard_normal(A.shape[0])
        f = G(x[:num_up].reshape(3, 3, N, N, N))
        x[:num_up] = f.reshape(-1)

        y = M.matvec(A.matvec(x))
        identity_err = np.linalg.norm(y - x)/np.linalg.norm(x)

        n_iter = [0]
        def cb(_r):
            n_iter[0] += 1
        sol, flag = spla.gmres(A=A, b=b, M=M, rtol=1e-8, atol=1e-14,
                               restart=200, maxiter=20, callback=cb,
                               callback_type="pr_norm")
        true_res = np.linalg.norm(b - A.matvec(sol))/np.linalg.norm(b)
        print("  {:<22}  ||M^-1 A x - x||/||x|| = {:.3e}   GMRES its = {:<5} flag={} true res={:.2e}"
              .format("restricted (FIXED)" if restrict else "unrestricted (pre-fix)",
                      identity_err, n_iter[0], flag, true_res))
        errors[restrict] = identity_err
    return errors


homo = report("HOMOGENEOUS body  (K(x) = K0 exactly -> M^-1 A must BE the identity)", HOMO)
report("HETEROGENEOUS body, contrast 100  (M^-1 A is only an approximation)",
       charge_path("bench_c100.txt"))

assert homo[True] < 1e-10, (
    "restricted symbol is NOT the operator inverse on a homogeneous body "
    "({:.2e}) - the preconditioner fix is wrong".format(homo[True]))
assert homo[False] > 1e-3, (
    "unrestricted symbol unexpectedly behaves like the inverse; the test has "
    "lost its discriminating power")
print("\nPASS: on a homogeneous body the restricted symbol is the exact inverse")
print("      on the compatible subspace ({:.2e}), the unrestricted one is not ({:.2e})."
      .format(homo[True], homo[False]))
