# -*- coding: utf-8 -*-
"""Benchmark suite for the high-contrast solver work.

Measures every improvement made to the mixed FFT solver against the
pre-change solver, across a filler/matrix stiffness contrast ladder, and
produces one coherent overview of what each change costs and buys.

    C5   inexact Newton with Eisenstat-Walker forcing terms
    C6   configurable reference tangent (mean | matrix | mid)
    FIX  Green preconditioner restricted to the compatible subspace
    W    Willot rotated discretization instead of the spectral one

Designed for a many-core server: every (structure, contrast, config) run is an
independent single-threaded process, scheduled over a process pool. Results are
appended to a JSONL file as they complete, so an interrupted sweep loses
nothing and can be resumed.

    python3 benchmark_suite.py                    # uses every usable core
    python3 benchmark_suite.py --resume           # continue an interrupted sweep
    python3 benchmark_suite.py --quick            # smoke test
    python3 benchmark_suite.py --dry-run          # show the plan

IMPORTANT - two families of configuration are reported separately:

  legacy      runs with the pre-fix preconditioner. These converge to a
              solution polluted with null-space content, i.e. the WRONG
              answer. They are included only to reproduce and quantify the
              historical behaviour.
  corrected   runs with the preconditioner fix. These are the numbers to use.

Speed-ups are only ever computed within a family, because the two families do
not solve the same problem. See docs/benchmarking.md.
"""

# Thread limits must be set before numpy/scipy are imported anywhere, so that
# each worker stays single-threaded and the pool is not oversubscribed.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "FFT_WORKERS"):
    os.environ.setdefault(_var, "1")

import argparse
import contextlib
import glob
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import _bootstrap  # noqa: F401  (puts the repo root and FFT_simulation on sys.path)
from project_paths import CHARGES_DIR, FG_DIR, PROJECT_ROOT, results_path, VOXELS_DIR

# Written next to the live solver, not into the working directory, so a sweep
# started from anywhere reuses the same materialised baseline.
BASELINE_PKG = os.path.join(FG_DIR, "_baseline")
BASELINE_REV = "3672bcdc5a654cb4911f3697641e0b600bba2cb7"   # last commit before this work

# Files that must be pinned together with the baseline solver, because the
# current versions differ in ways that would change the baseline's behaviour.
# fg/preconditioning.py is the critical one: the preconditioner fix lives
# there, so a pinned mxfft.py importing the *live* module would silently run
# with the corrected preconditioner and stop being a baseline at all.
BASELINE_PINNED = ("mxfft.py", "preconditioning.py")

CHARGE_TEMPLATE = """#first two lines: model:---0)model num 1) p1.. 2)p2...
1.0\t{Em:g}\t0.48\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
1.0\t{Ef:g}\t0.30\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
#charge dF
1.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0
#(Charge type) P-1 or F-0: 0 = control this component by Fij, 1 = control this component by average Pij
0.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0
"""

E_MATRIX = 10.0
CONTRASTS = [10, 100, 500, 1000, 2500]

# name -> (solver, kwargs, family). "base" is the pinned pre-change solver.
# Every "legacy" entry keeps precond_restrict=False so it reproduces the
# historical (incorrect) preconditioner; "corrected" entries switch it on.
CONFIGS = [
    # ---- legacy family: reference is "baseline"
    ("baseline",          "base", {},                                                                        "legacy"),
    ("baseline-new",      "new",  dict(forcing="fixed"),                                                      "legacy"),
    ("C5",                "new",  dict(forcing="eisenstat_walker"),                                           "legacy"),
    ("C6-matrix",         "new",  dict(forcing="fixed", reference="matrix"),                                  "legacy"),
    ("C6-mid",            "new",  dict(forcing="fixed", reference="mid"),                                     "legacy"),
    ("C5+C6-matrix",      "new",  dict(forcing="eisenstat_walker", reference="matrix"),                       "legacy"),
    ("C5+C6-mid",         "new",  dict(forcing="eisenstat_walker", reference="mid"),                          "legacy"),
    # ---- corrected family: reference is "FIX"
    ("FIX",               "new",  dict(forcing="fixed"),                                                      "corrected"),
    ("FIX+C5",            "new",  dict(forcing="eisenstat_walker"),                                           "corrected"),
    ("FIX+C5+C6-matrix",  "new",  dict(forcing="eisenstat_walker", reference="matrix"),                       "corrected"),
    ("FIX+C5+C6-mid",     "new",  dict(forcing="eisenstat_walker", reference="mid"),                          "corrected"),
    ("FIX+C5+Willot",     "new",  dict(forcing="eisenstat_walker", discretization="willot"),                  "corrected"),
    ("FIX+C5+Willot+C6m", "new",  dict(forcing="eisenstat_walker", discretization="willot", reference="matrix"), "corrected"),
    # ---- ground truth: no preconditioner, so the iterates provably stay in the
    #      compatible subspace. Expensive; enable with --control.
    ("control-unprecond", "new",  dict(forcing="eisenstat_walker", preconditioner="gmres"),                   "control"),
]

FAMILY_REFERENCE = {"legacy": "baseline", "corrected": "FIX", "control": "control-unprecond"}


# --------------------------------------------------------------------------- setup
def ensure_baseline(rev=BASELINE_REV):
    """Materialise the pre-change solver from git as a self-contained package.

    The solver and every module it depends on that has since changed are taken
    from `rev`, and the pinned solver's imports are redirected at the pinned
    copies, so "baseline" really is the code that produced the existing
    results. Written atomically: many workers may reach this concurrently.
    """
    marker = os.path.join(BASELINE_PKG, ".rev")
    if os.path.exists(marker):
        with open(marker) as fh:
            if fh.read().strip() == rev:
                return BASELINE_PKG

    staging = BASELINE_PKG + ".tmp{}".format(os.getpid())
    os.makedirs(staging, exist_ok=True)
    open(os.path.join(staging, "__init__.py"), "w").close()

    pinned = {name.rsplit(".", 1)[0] for name in BASELINE_PINNED}
    for name in BASELINE_PINNED:
        # the pinned rev predates the FFT_simulation/ move, so fg/ was at the
        # repository root back then; the path below is a path *inside that
        # commit*, not on disk.
        src = subprocess.check_output(
            ["git", "show", "{}:fg/{}".format(rev, name)], cwd=PROJECT_ROOT).decode()
        # point the pinned solver at its pinned dependencies, not the live ones
        for module in pinned:
            src = src.replace("from fg.{} import".format(module),
                              "from fg._baseline.{} import".format(module))
        with open(os.path.join(staging, name), "w") as fh:
            fh.write(src)

    with open(os.path.join(staging, ".rev"), "w") as fh:
        fh.write(rev + "\n")

    if os.path.exists(BASELINE_PKG):
        import shutil
        shutil.rmtree(BASELINE_PKG, ignore_errors=True)
    try:
        os.rename(staging, BASELINE_PKG)
    except OSError:                       # another worker won the race
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
    return BASELINE_PKG


def estimate_memory_mb(N, gmres_restart, willot=False):
    """Peak resident memory of a single run, in MB.

    The big arrays are the projection symbol Ghat4 (81 N^3, doubled if the
    discretization is complex), the tangent K4 (81 N^3), the preconditioner
    symbol (100 N^3, doubled if complex) and the GMRES restart basis
    (restart * 10 N^3). This matters: at N=63 one run needs well over a
    gigabyte, so 100 concurrent workers is a memory decision, not just a
    core-count decision.
    """
    cells = float(N)**3
    complex_factor = 2.0 if willot else 1.0
    ghat = 81*cells*8*complex_factor
    k4 = 81*cells*8
    symbol = 100*cells*8*complex_factor
    basis = float(gmres_restart)*10*cells*8
    workspace = 30*cells*16
    return (ghat + k4 + symbol + basis + workspace)/1e6


def default_workers():
    """Usable cores. sched_getaffinity respects cpuset/affinity limits, which
    os.cpu_count() ignores - it reports the host's cores even inside a
    container with a smaller quota."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def available_memory_mb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1])/1024.0
    except Exception:
        pass
    return None


def ensure_charge(contrast, charge_dir):
    os.makedirs(charge_dir, exist_ok=True)
    path = os.path.join(charge_dir, "bench_c{}.txt".format(contrast))
    if not os.path.exists(path):
        tmp = path + ".tmp{}".format(os.getpid())
        with open(tmp, "w") as fh:
            fh.write(CHARGE_TEMPLATE.format(Em=E_MATRIX, Ef=E_MATRIX*contrast))
        os.replace(tmp, path)
    return path


def git_revision():
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode().strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ------------------------------------------------------------------- measurement
def incompatible_fraction(F, stress_control, discretization):
    """How much of the converged fluctuation field is NOT a gradient field.

    G is the projector onto compatible fields, so a physical solution satisfies
    G(dF) = dF and this returns ~0. A large value means the solve left the
    compatible subspace and the field is not a deformation of anything.
    """
    import scipy.fft
    from fg.preconditioning import build_Ghat4

    N = F.shape[-1]
    axes = (-3, -2, -1)
    Ghat4 = build_Ghat4(N, stress_control, 3, discretization)
    fluct = F - F.mean(axis=(2, 3, 4))[:, :, None, None, None]
    hat = np.fft.fftshift(scipy.fft.fftn(np.fft.ifftshift(fluct, axes=axes), axes=axes), axes=axes)
    proj = np.fft.fftshift(scipy.fft.ifftn(
        np.fft.ifftshift(np.einsum("ijklxyz,klxyz->ijxyz", Ghat4, hat), axes=axes),
        axes=axes), axes=axes).real
    denom = np.linalg.norm(fluct)
    return float(np.linalg.norm(fluct - proj)/denom) if denom > 0 else 0.0


def run_one(task):
    """Execute a single (structure, contrast, config) benchmark point."""
    started = time.time()
    record = {
        "structure": task["structure_name"], "contrast": task["contrast"],
        "config": task["config"], "family": task["family"], "N": task["N"],
        "increments": task["increments"],
    }
    out_dir = task["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    try:
        if task["solver"] == "base":
            ensure_baseline()
            import fg._baseline.mxfft as solver_mod
        else:
            import fg.mxfft as solver_mod

        kwargs = dict(task["kwargs"])
        # the pinned pre-change solver predates every one of these arguments
        if task["solver"] == "base":
            kwargs = {}

        with open(log_path, "w", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                prob = solver_mod.FFTSolver(
                    task["structure"], charge_path=task["charge"],
                    output_path=out_dir, N=task["N"], output_name=".",
                )
                prob.calculate(
                    incre_list=[0.1]*task["increments"], savemodel="normal",
                    preconditioner=kwargs.pop("preconditioner", "reference"),
                    max_gmres_iter=task["max_gmres_iter"],
                    gmres_restart=task["gmres_restart"],
                    **kwargs
                )

        stats = prob.solver_stats
        kry = [k for inc in stats.get("increments", []) for k in inc.get("krylov_iterations", [])]
        record.update({
            "status": stats.get("status", "unknown"),
            "krylov_total": int(sum(kry)),
            "krylov_max_per_solve": int(max(kry)) if kry else 0,
            "newton_total": int(sum(i.get("newton_iterations", 0) for i in stats.get("increments", []))),
            "step_cuts": stats.get("step_cuts", 0),
            "solver_seconds": float(sum(i.get("time_seconds", 0.0) for i in stats.get("increments", []))),
            "P11": float(np.array(prob.Ps)[-1][0, 0]) if getattr(prob, "Ps", None) else None,
            "P11_curve": [float(p[0, 0]) for p in prob.Ps] if getattr(prob, "Ps", None) else [],
            "F11": float(np.array(prob.Fs)[-1][0, 0]) if getattr(prob, "Fs", None) else None,
        })

        F = getattr(prob, "F_final", None)
        if F is not None:
            record["incompatible"] = incompatible_fraction(
                F, prob.pb.stress_control, task["kwargs"].get("discretization", "fourier"))
            eye = np.eye(3)[:, :, None, None, None]
            dev = np.sqrt(((F - eye)**2).sum(axis=(0, 1)))
            record["filler_strain"] = float(dev[task_mask(task)].mean()) if task_mask(task) is not None else None
            record["F11_min"] = float(F[0, 0].min())
            record["F11_max"] = float(F[0, 0].max())
    except Exception as exc:
        record.update({"status": "ERROR", "error": "{}: {}".format(type(exc).__name__, exc),
                       "traceback": traceback.format_exc()})
        with open(log_path, "a") as log:
            log.write("\n" + traceback.format_exc())

    record["wall_seconds"] = time.time() - started
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    return record


_MASK_CACHE = {}


def task_mask(task):
    """Boolean filler mask for the task's structure (cached per process)."""
    path = task["structure"]
    if path not in _MASK_CACHE:
        try:
            with np.load(path, allow_pickle=False) as data:
                _MASK_CACHE[path] = np.array(data["phase"]) == 1
        except Exception:
            _MASK_CACHE[path] = None
    return _MASK_CACHE[path]


# ----------------------------------------------------------------------- planning
def build_tasks(args):
    structures = sorted(glob.glob(args.structures))
    if not structures:
        raise SystemExit("no structures matched {!r}".format(args.structures))

    wanted = set(args.configs) if args.configs else None
    configs = [c for c in CONFIGS if c[3] != "control" or args.control]
    if wanted:
        configs = [c for c in configs if c[0] in wanted]
    if not configs:
        raise SystemExit("no configs selected")

    tasks = []
    for structure in structures:
        name = os.path.splitext(os.path.basename(structure))[0]
        for contrast in args.contrasts:
            charge = ensure_charge(contrast, args.charge_dir)
            for cfg_name, solver, kwargs, family in configs:
                out_dir = os.path.join(args.out, "runs", name, "c{}".format(contrast), cfg_name)
                tasks.append({
                    "structure": structure, "structure_name": name,
                    "contrast": contrast, "charge": charge,
                    "config": cfg_name, "solver": solver, "family": family,
                    "kwargs": dict(kwargs, precond_restrict=(family != "legacy")) if solver == "new" else {},
                    "N": args.N, "increments": args.increments,
                    "max_gmres_iter": (args.control_max_gmres_iter if family == "control"
                                       else args.max_gmres_iter),
                    "out_dir": out_dir,
                    "gmres_restart": args.gmres_restart,
                })
    # longest first: the highest contrasts dominate the critical path
    tasks.sort(key=lambda t: (-t["contrast"], t["config"]))
    return tasks


def already_done(task):
    path = os.path.join(task["out_dir"], "result.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            return json.load(fh).get("status") not in (None, "ERROR")
    except Exception:
        return False


# ------------------------------------------------------------------------ reporting
def load_results(out):
    results = []
    for path in glob.glob(os.path.join(out, "runs", "*", "*", "*", "result.json")):
        try:
            with open(path) as fh:
                results.append(json.load(fh))
        except Exception:
            pass
    return results


def summarise(results, out, contrasts):
    if not results:
        print("no results to summarise")
        return

    by = {}
    for r in results:
        by.setdefault((r["structure"], r["contrast"], r["config"]), r)

    structures = sorted({r["structure"] for r in results})
    lines = ["# Solver Improvement Benchmark", "",
             "Krylov = total Krylov iterations over every Newton step of every increment.",
             "Speed-up is within a family only (legacy and corrected do not solve the",
             "same problem: the legacy preconditioner converges to a polluted solution).",
             "`incompat` is the non-gradient fraction of the converged fluctuation field;",
             "it should be ~0 for a physical solution.", ""]

    for structure in structures:
        lines += ["## structure: {}".format(structure), ""]
        for family in ("legacy", "corrected", "control"):
            fam_cfgs = [c[0] for c in CONFIGS if c[3] == family]
            present = [c for c in fam_cfgs if any((structure, ct, c) in by for ct in contrasts)]
            if not present:
                continue
            lines += ["### {} family".format(family), "",
                      "| contrast | config | status | Krylov | speedup | Newton | wall (s) | P11 | incompat |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for contrast in contrasts:
                ref = by.get((structure, contrast, FAMILY_REFERENCE.get(family)))
                for cfg in present:
                    r = by.get((structure, contrast, cfg))
                    if r is None:
                        continue
                    if r.get("status") == "ERROR":
                        lines.append("| {} | {} | ERROR | | | | {:.0f} | | |".format(
                            contrast, cfg, r.get("wall_seconds", 0)))
                        continue
                    sp = ""
                    if ref and ref.get("krylov_total") and r.get("krylov_total"):
                        sp = "{:.2f}x".format(ref["krylov_total"]/r["krylov_total"])
                    lines.append("| {} | {} | {} | {} | {} | {} | {:.0f} | {} | {} |".format(
                        contrast, cfg, r.get("status", "?"), r.get("krylov_total", ""), sp,
                        r.get("newton_total", ""), r.get("wall_seconds", 0),
                        "{:.6f}".format(r["P11"]) if r.get("P11") is not None else "",
                        "{:.1e}".format(r["incompatible"]) if r.get("incompatible") is not None else ""))
            lines.append("")

    md = os.path.join(out, "summary.md")
    with open(md, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    cols = ["structure", "contrast", "config", "family", "status", "krylov_total",
            "krylov_max_per_solve", "newton_total", "step_cuts", "wall_seconds",
            "solver_seconds", "P11", "F11", "incompatible", "filler_strain",
            "F11_min", "F11_max", "N", "increments"]
    csv_path = os.path.join(out, "summary.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(results, key=lambda x: (x["structure"], x["contrast"], x["config"])):
            fh.write(",".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols) + "\n")

    print("\n".join(lines))
    print("\nwrote {}\n      {}".format(md, csv_path))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", default=os.path.join(VOXELS_DIR, "1_voxel.npz"),
                    help="glob of .npz microstructures (default: one test structure)")
    ap.add_argument("--contrasts", type=int, nargs="*", default=CONTRASTS)
    ap.add_argument("--configs", nargs="*", help="subset of config names to run")
    ap.add_argument("--N", type=int, default=31)
    ap.add_argument("--increments", type=int, default=3)
    ap.add_argument("--max-gmres-iter", type=int, default=1000,
                    help="Krylov iteration cap per solve (default 1000, matching "
                         "production). On hitting it the solver cuts the load step "
                         "and retries, which is the intended recovery - raising it "
                         "lets a stagnating solve grind instead.")
    ap.add_argument("--control-max-gmres-iter", type=int, default=50000,
                    help="separate, much larger cap for the unpreconditioned control, "
                         "which legitimately needs thousands of iterations")
    ap.add_argument("--gmres-restart", type=int, default=None,
                    help="GMRES restart length. Caps the Krylov basis, which is the "
                         "largest per-worker allocation at high N (default: solver's "
                         "own memory-aware choice, ~40 at N=63 = 800 MB per worker)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel runs (default: all usable cores). Each run is "
                         "single-threaded, so this is the real parallelism.")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 contrasts, 1 increment, core configs only")
    ap.add_argument("--out", default=results_path("benchmark_suite"))
    ap.add_argument("--charge-dir", default=CHARGES_DIR)
    ap.add_argument("--control", action="store_true",
                    help="also run the unpreconditioned ground-truth config (slow)")
    ap.add_argument("--resume", action="store_true", help="skip runs that already completed")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true",
                    help="re-generate summary.md/summary.csv from existing results")
    args = ap.parse_args()

    if args.quick:
        args.contrasts = [10, 100]
        args.increments = 1
        if not args.configs:
            args.configs = ["baseline", "baseline-new", "C5", "FIX", "FIX+C5", "FIX+C5+Willot"]

    os.makedirs(args.out, exist_ok=True)

    if args.summary_only:
        summarise(load_results(args.out), args.out, args.contrasts)
        return

    ensure_baseline()
    tasks = build_tasks(args)

    if args.resume:
        before = len(tasks)
        tasks = [t for t in tasks if not already_done(t)]
        print("resume: {} of {} runs already complete".format(before - len(tasks), before))

    manifest = {
        "git_revision": git_revision(), "baseline_revision": BASELINE_REV,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "N": args.N,
        "increments": args.increments, "contrasts": args.contrasts,
        "structures": sorted(glob.glob(args.structures)), "workers": args.workers,
        "runs_planned": len(tasks),
        "configs": [{"name": n, "solver": s, "kwargs": {k: str(v) for k, v in kw.items()},
                     "family": f} for n, s, kw, f in CONFIGS],
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    if args.dry_run:
        print(json.dumps(manifest, indent=1))
        for t in tasks:
            print("  {:>6}x  {:<20}  -> {}".format(t["contrast"], t["config"], t["out_dir"]))
        print("\n{} runs planned".format(len(tasks)))
        return

    if not tasks:
        print("nothing to run")
        summarise(load_results(args.out), args.out, args.contrasts)
        return

    workers = max(1, min(args.workers, len(tasks)))

    # memory is the real constraint on a many-core box, not cores
    restart = args.gmres_restart or int(min(100, max(20, 8.e8/(8.0*10*args.N**3))))
    per_run = estimate_memory_mb(args.N, restart, willot=True)
    total = per_run*workers
    avail = available_memory_mb()
    print("estimated peak memory: {:.0f} MB per run x {} workers = {:.1f} GB"
          .format(per_run, workers, total/1000.0))
    if avail is not None:
        print("available memory: {:.1f} GB".format(avail/1000.0))
        if total > 0.85*avail:
            safe = max(1, int(0.85*avail/per_run))
            print("\n*** WARNING: this will likely exhaust memory. Use --workers {} "
                  "or fewer,\n    or lower --gmres-restart / --N. Continuing anyway "
                  "in 10 s; Ctrl-C to abort. ***\n".format(safe))
            time.sleep(10)

    print("running {} benchmark points on {} workers (1 thread each)".format(len(tasks), workers))
    print("results stream to {}".format(os.path.join(args.out, "results.jsonl")))

    jsonl = open(os.path.join(args.out, "results.jsonl"), "a", buffering=1)
    done = 0
    t0 = time.time()
    try:
        if workers == 1:
            for task in tasks:
                rec = run_one(task)
                done += 1
                jsonl.write(json.dumps(rec) + "\n")
                print("[{}/{}] {:>6}x {:<20} {:<10} krylov={} {:.0f}s".format(
                    done, len(tasks), rec["contrast"], rec["config"],
                    rec.get("status", "?"), rec.get("krylov_total", "-"), rec["wall_seconds"]),
                    flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_one, t): t for t in tasks}
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:       # a worker died outright
                        rec = {"structure": task["structure_name"], "contrast": task["contrast"],
                               "config": task["config"], "family": task["family"],
                               "status": "ERROR", "error": repr(exc), "wall_seconds": 0.0}
                    done += 1
                    jsonl.write(json.dumps(rec) + "\n")
                    print("[{}/{}] {:>6}x {:<20} {:<10} krylov={} {:.0f}s".format(
                        done, len(tasks), rec["contrast"], rec["config"],
                        rec.get("status", "?"), rec.get("krylov_total", "-"),
                        rec.get("wall_seconds", 0)), flush=True)
    finally:
        jsonl.close()

    print("\nall runs finished in {:.0f} s".format(time.time() - t0))
    summarise(load_results(args.out), args.out, args.contrasts)


if __name__ == "__main__":
    main()
