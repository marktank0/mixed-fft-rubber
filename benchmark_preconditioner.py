# -*- coding: utf-8 -*-
"""Is the Green preconditioner actually worth it? Head-to-head, same everything else.

The Green/reference preconditioner costs roughly one extra FFT round trip per
Krylov iteration - at N=31 its apply is ~63 ms against ~59 ms for the operator
matvec itself, so it very nearly DOUBLES the cost of an iteration. It only pays
off if it more than halves the iteration count. That trade swings with contrast
(a low-contrast system is already well conditioned, so there are fewer
iterations to save) and with grid size (the symbol build and its pseudo-inverse
grow as N^3), so it is worth measuring at your actual operating point rather
than assuming the contrast-100 numbers carry over.

Three solvers, identical in every other respect:

  reference   Green/reference preconditioner, applied through GMRES  (production)
  gmres       NO preconditioner, same GMRES                          (the bare control)
  cg          NO preconditioner, and the solver falls back to CG. Note this is
              not an apples-to-apples comparison - CG is for symmetric positive
              definite systems and the mixed tangent is a nonsymmetric saddle
              point - but it is what preconditioner=None does today, and its
              iterations are much cheaper, so it is worth seeing.

Each (structure, solver) runs in its own subprocess with its own output
directory, so nothing overwrites anything and one blow-up cannot take down the
sweep. Runs are SEQUENTIAL by default because wall-clock numbers from
contending workers are not comparable.

Examples
--------
    # 5 structures spread across filler fraction, contrast 50, N read from the file
    python benchmark_preconditioner.py

    # the case you actually care about: contrast 10
    python benchmark_preconditioner.py --charge Neo_1.0_E10-100.txt

    # N=63
    python benchmark_preconditioner.py --structures "3D_samples/voxels_63/*.npz"

    # short and cheap while you check it does what you want
    python benchmark_preconditioner.py --count 2 --increments 2 --timeout 600
"""
import argparse
import csv
import glob as globmod
import json
import multiprocessing
import os
import sys
import time
import traceback

from project_paths import (
    CHARGES_DIR,
    MICROSTRUCTURE_DIR,
    PROJECT_ROOT,
    charge_path,
    ensure_import_paths,
    results_path,
)

DEFAULT_STRUCTURES = os.path.join(
    MICROSTRUCTURE_DIR, "3D_samples", "improved_struct_v4", "voxel_structures", "*.npz"
)

SOLVERS = {
    # label       preconditioner kwarg passed to FFTSolver.calculate
    "reference": "reference",
    "gmres": "gmres",
    "cg": None,
}


# --------------------------------------------------------------------------
# structure selection
# --------------------------------------------------------------------------

def grid_size(path, phase_key="phase"):
    """Grid size N stored in a structure file, or None if it is flat/unknown."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if phase_key not in data.files:
            return None, None
        phase = data[phase_key]
    if phase.ndim == 3 and phase.shape[0] == phase.shape[1] == phase.shape[2]:
        return int(phase.shape[0]), float(phase.mean())
    return None, float(phase.mean())


def pick_structures(pattern, count):
    """`count` structures spread evenly across the sorted list, not the first N.

    Sorting is by name, and these sets are named by filler content (phr_...),
    so an even spread samples the whole volume-fraction range instead of
    clustering at the easy end.
    """
    paths = sorted(globmod.glob(pattern))
    if not paths:
        raise SystemExit("no structures matched {!r}".format(pattern))
    if count >= len(paths):
        return paths
    step = (len(paths) - 1)/float(count - 1) if count > 1 else 0.0
    return [paths[int(round(i*step))] for i in range(count)]


# --------------------------------------------------------------------------
# one case, in its own process
# --------------------------------------------------------------------------

def run_case(args, structure, solver, out_dir, result_path):
    """Run one (structure, solver) and write a JSON result. Executed in a child."""
    ensure_import_paths()
    import fg.mxfft as mx

    record = {
        "structure": os.path.basename(structure),
        "solver": solver,
        "status": "ERROR",
        "krylov_total": None,
        "krylov_max_per_solve": None,
        "newton_total": None,
        "wall_seconds": None,
        "solver_seconds": None,
        "step_cuts": None,
        "F11": None,
        "P11": None,
        "cap_hit": None,
        "error": None,
    }

    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "w")
    stdout = sys.stdout
    try:
        prob = mx.FFTSolver(structure, charge_path=args.charge, output_path=out_dir,
                            N=args.n, output_name=".")
        kwargs = dict(
            incre_list=[args.step]*args.increments,
            savemodel="normal",
            preconditioner=SOLVERS[solver],
            tol_rel=args.tol_rel,
            max_gmres_iter=args.max_gmres_iter,
            gmres_restart=args.gmres_restart,
            min_substep_ratio=args.min_substep_ratio,
            forcing=args.forcing,
            inner_rtol=args.inner_rtol,
            eta_min=args.eta_min,
            eta_max=args.eta_max,
        )
        if solver == "reference":
            kwargs.update(reference=args.reference, precond_restrict=True,
                          discretization=args.discretization)
        t0 = time.time()
        sys.stdout = log
        prob.calculate(**kwargs)
        sys.stdout = stdout
        record["wall_seconds"] = time.time() - t0

        stats = prob.solver_stats
        incs = stats.get("increments", [])
        per_solve = [k for i in incs for k in i.get("krylov_iterations", [])]
        record.update(
            status=stats.get("status", "?"),
            krylov_total=int(sum(per_solve)),
            krylov_max_per_solve=int(max(per_solve)) if per_solve else 0,
            newton_total=int(sum(len(i.get("krylov_iterations", [])) for i in incs)),
            solver_seconds=float(sum(i.get("time_seconds", 0.0) for i in incs)),
            step_cuts=int(stats.get("step_cuts", 0)),
            cap_hit=bool(per_solve) and max(per_solve) >= args.max_gmres_iter,
        )
        if incs:
            reached = max(i["load_start"] + (i["step"] if i["converged"] else 0.0) for i in incs)
            record["F11"] = 1.0 + reached
        if getattr(prob, "Ps", None):
            record["P11"] = float(prob.Ps[-1][0, 0])
    except Exception:
        sys.stdout = stdout
        record["error"] = traceback.format_exc(limit=6)
    finally:
        sys.stdout = stdout
        log.close()

    with open(result_path, "w") as fh:
        json.dump(record, fh, indent=1)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def summarise(records, out_root, args):
    by = {(r["structure"], r["solver"]): r for r in records}
    structures = []
    for r in records:
        if r["structure"] not in structures:
            structures.append(r["structure"])

    def fmt(v, spec="{}"):
        return "-" if v is None else spec.format(v)

    lines = []
    lines.append("# Preconditioner head-to-head")
    lines.append("")
    lines.append("charge `{}`  |  N={}  |  {} increments of {}  |  tol_rel {:g}".format(
        os.path.basename(args.charge), args.n, args.increments, args.step, args.tol_rel))
    lines.append("max_gmres_iter {}  |  gmres_restart {}  |  reference `{}`  |  forcing `{}`".format(
        args.max_gmres_iter, args.gmres_restart or "auto", args.reference, args.forcing))
    lines.append("")
    lines.append("`!` marks a run in which some solve hit the iteration cap: its cost is a")
    lines.append("LOWER BOUND, not a measurement, because a capped solve is treated as a")
    lines.append("failed increment and triggers a load-step cut.")
    lines.append("")
    header = ("| structure | solver | status | Krylov | max/solve | Newton | wall (s) | "
              "s/iter | cuts | F11 | P11 |")
    lines.append(header)
    lines.append("|" + "---|"*11)
    for s in structures:
        for solver in args.solvers:
            r = by.get((s, solver))
            if r is None:
                continue
            if r["error"]:
                lines.append("| {} | {} | ERROR | | | | | | | | |".format(s, solver))
                continue
            spi = (r["wall_seconds"]/r["krylov_total"]) if r["krylov_total"] else None
            lines.append("| {} | {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                s, solver, r["status"], " **!**" if r["cap_hit"] else "",
                fmt(r["krylov_total"]), fmt(r["krylov_max_per_solve"]), fmt(r["newton_total"]),
                fmt(r["wall_seconds"], "{:.1f}"), fmt(spi, "{:.3f}"), fmt(r["step_cuts"]),
                fmt(r["F11"], "{:.3f}"), fmt(r["P11"], "{:.6f}")))
    lines.append("")

    # the actual question: is the preconditioner worth it, per structure?
    if "reference" in args.solvers and "gmres" in args.solvers:
        lines.append("## Green preconditioner vs the bare solver (GMRES, no preconditioner)")
        lines.append("")
        lines.append("| structure | Krylov speed-up | wall speed-up | same P11? |")
        lines.append("|---|---|---|---|")
        for s in structures:
            a, b = by.get((s, "reference")), by.get((s, "gmres"))
            if not a or not b or a["error"] or b["error"]:
                continue
            if not a["krylov_total"] or not b["krylov_total"]:
                continue
            note = ""
            if a["cap_hit"] or b["cap_hit"]:
                note = " (truncated - not a measurement)"
            same = "-"
            if a["P11"] is not None and b["P11"] is not None:
                rel = abs(a["P11"] - b["P11"])/max(abs(b["P11"]), 1e-30)
                same = "yes ({:.1e})".format(rel) if rel < 1e-4 else "**NO ({:.1e})**".format(rel)
            lines.append("| {} | {:.2f}x | {:.2f}x{} | {} |".format(
                s, b["krylov_total"]/a["krylov_total"],
                b["wall_seconds"]/a["wall_seconds"], note, same))
        lines.append("")
        lines.append("A wall speed-up below 1.0 means the preconditioner is COSTING you time:")
        lines.append("it roughly doubles the price of an iteration, so it has to more than")
        lines.append("halve the iteration count to break even.")
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(out_root, "summary.md"), "w") as fh:
        fh.write(text + "\n")

    cols = ["structure", "solver", "status", "krylov_total", "krylov_max_per_solve",
            "newton_total", "wall_seconds", "solver_seconds", "step_cuts", "F11", "P11",
            "cap_hit", "error"]
    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structures", default=DEFAULT_STRUCTURES,
                   help="glob for structure .npz files (default: smaller_structures_v2)")
    p.add_argument("--count", type=int, default=5,
                   help="how many structures, spread evenly across the sorted list (default 5)")
    p.add_argument("--n", type=int, default=None,
                   help="grid size; default is read from the structure file")
    p.add_argument("--charge", default="Neo_1.0_E10-500.txt",
                   help="charge file name in Run_configs/Charges, or a path "
                        "(default Neo_1.0_E10-500.txt = contrast 50; "
                        "Neo_1.0_E10-100.txt = contrast 10)")
    p.add_argument("--solvers", default="reference,gmres",
                   help="comma list from {reference,gmres,cg} (default reference,gmres)")
    p.add_argument("--increments", type=int, default=4, help="number of load increments")
    p.add_argument("--step", type=float, default=0.05, help="size of each load increment")
    p.add_argument("--tol-rel", type=float, default=1.0e-5, dest="tol_rel")
    p.add_argument("--max-gmres-iter", type=int, default=10000, dest="max_gmres_iter")
    p.add_argument("--gmres-restart", type=int, default=None, dest="gmres_restart",
                   help="default: the solver's own memory-aware choice")
    p.add_argument("--min-substep-ratio", type=float, default=1.0/16.0, dest="min_substep_ratio")
    p.add_argument("--reference", default="matrix", choices=("mean", "matrix", "mid"),
                   help="reference tangent for the Green preconditioner")
    p.add_argument("--discretization", default="fourier", choices=("fourier", "willot"))
    p.add_argument("--forcing", default="eisenstat_walker", choices=("eisenstat_walker", "fixed"))
    p.add_argument("--inner-rtol", type=float, default=1.0e-6, dest="inner_rtol")
    p.add_argument("--eta-min", type=float, default=1.0e-3, dest="eta_min")
    p.add_argument("--eta-max", type=float, default=1.0e-2, dest="eta_max")
    p.add_argument("--timeout", type=float, default=None,
                   help="seconds per case before it is killed and marked TIMEOUT")
    p.add_argument("--fft-workers", type=int, default=1, dest="fft_workers",
                   help="threads for the FFTs (default 1, so timings are comparable)")
    p.add_argument("--out", default=None, help="output directory (default Results/preconditioner_ab)")
    args = p.parse_args()

    args.solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    bad = [s for s in args.solvers if s not in SOLVERS]
    if bad:
        raise SystemExit("unknown solver(s) {}; choose from {}".format(bad, sorted(SOLVERS)))

    if os.path.sep in args.charge or os.path.exists(args.charge):
        args.charge = os.path.abspath(args.charge)
    else:
        args.charge = charge_path(args.charge)
    if not os.path.exists(args.charge):
        raise SystemExit("charge file not found: {}\navailable:\n  {}".format(
            args.charge, "\n  ".join(sorted(os.listdir(CHARGES_DIR)))))

    structures = pick_structures(args.structures, args.count)

    sizes = {}
    for s in structures:
        n, vf = grid_size(s)
        sizes[s] = (n, vf)
    if args.n is None:
        found = sorted({n for n, _ in sizes.values() if n})
        if len(found) != 1:
            raise SystemExit(
                "structures have mixed or unknown grid sizes {}; pass --n explicitly".format(found))
        args.n = found[0]
    mismatch = [os.path.basename(s) for s, (n, _) in sizes.items() if n and n != args.n]
    if mismatch:
        raise SystemExit("--n {} does not match these structures: {}".format(args.n, mismatch))

    out_root = args.out or results_path("preconditioner_ab")
    os.makedirs(out_root, exist_ok=True)

    # one thread everywhere, so wall-clock numbers mean something
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["FFT_WORKERS"] = str(args.fft_workers)

    print("preconditioner head-to-head")
    print("  charge      {}".format(os.path.basename(args.charge)))
    print("  N           {}".format(args.n))
    print("  loading     {} increments of {} -> F11 up to {:.3f}".format(
        args.increments, args.step, 1.0 + args.increments*args.step))
    print("  solvers     {}".format(", ".join(args.solvers)))
    print("  output      {}".format(out_root))
    print("  structures:")
    for s in structures:
        print("    {:44s} vf {:.4f}".format(os.path.basename(s), sizes[s][1]))
    print("\n  runs are sequential; wall times from contending processes are not comparable.\n")

    records = []
    total = len(structures)*len(args.solvers)
    done = 0
    t_start = time.time()
    for structure in structures:
        stem = os.path.splitext(os.path.basename(structure))[0]
        for solver in args.solvers:
            done += 1
            out_dir = os.path.join(out_root, stem, solver)
            os.makedirs(out_dir, exist_ok=True)
            result_path = os.path.join(out_dir, "result.json")
            print("[{}/{}] {:38s} {:10s} ".format(done, total, stem, solver), end="", flush=True)

            t0 = time.time()
            proc = multiprocessing.Process(
                target=run_case, args=(args, structure, solver, out_dir, result_path))
            proc.start()
            proc.join(args.timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
                rec = {"structure": os.path.basename(structure), "solver": solver,
                       "status": "TIMEOUT", "wall_seconds": time.time() - t0,
                       "krylov_total": None, "krylov_max_per_solve": None, "newton_total": None,
                       "solver_seconds": None, "step_cuts": None, "F11": None, "P11": None,
                       "cap_hit": None, "error": "timeout after {:g}s".format(args.timeout)}
                print("TIMEOUT after {:.0f}s".format(rec["wall_seconds"]))
            elif os.path.exists(result_path):
                with open(result_path) as fh:
                    rec = json.load(fh)
                if rec.get("error"):
                    print("ERROR (see {})".format(result_path))
                else:
                    print("{:24s} Krylov {:7d}  Newton {:3d}  {:8.1f}s".format(
                        rec["status"], rec["krylov_total"], rec["newton_total"],
                        rec["wall_seconds"]))
            else:
                rec = {"structure": os.path.basename(structure), "solver": solver,
                       "status": "CRASHED", "wall_seconds": time.time() - t0,
                       "krylov_total": None, "krylov_max_per_solve": None, "newton_total": None,
                       "solver_seconds": None, "step_cuts": None, "F11": None, "P11": None,
                       "cap_hit": None, "error": "child exited without writing a result"}
                print("CRASHED (exit {})".format(proc.exitcode))
            records.append(rec)

    print("\ntotal {:.0f}s\n".format(time.time() - t_start))
    print(summarise(records, out_root, args))
    print("written to {}".format(os.path.join(out_root, "summary.md")))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
