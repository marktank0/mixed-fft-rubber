# Benchmarking the Solver Improvements

`FFT_simulation/benchmark_suite.py` measures every change made during the high-contrast work
against the pre-change solver, across a filler/matrix contrast ladder, and
produces one table you can read end to end.

---

## 1. What it measures

| tag | change |
|---|---|
| **C5** | inexact Newton, Eisenstat–Walker forcing terms (`forcing`) |
| **C6** | configurable reference tangent (`reference` = mean / matrix / mid) |
| **FIX** | Green preconditioner restricted to the compatible subspace (`precond_restrict`) |
| **Willot** | rotated finite-difference discretization (`discretization`) |

The pre-change solver is not reimplemented — it is materialised straight out of
git (rev `3672bcd`, the last commit before this work) into `FFT_simulation/fg/_baseline/`, so
"baseline" really is the code that produced the existing results.

**Both `FFT_simulation/fg/mxfft.py` and `FFT_simulation/fg/preconditioning.py` are pinned**, and the pinned
solver's imports are rewritten to point at the pinned copies. This matters:
the preconditioner fix lives in `preconditioning.py`, so a pinned solver
importing the *live* module would silently run with the corrected
preconditioner and stop being a baseline at all. (That bug was present in an
earlier draft of this script and is exactly what the `baseline` vs
`baseline-new` self-check below catches.)

---

## 2. The one thing to understand before reading the output

Runs are split into **two families that do not solve the same problem**:

- **legacy** — `precond_restrict=False`, i.e. the historical preconditioner.
  These converge to a solution polluted with null-space content: *the wrong
  answer* (see `docs/green_reference_preconditioning.md`). They are included
  only to reproduce and quantify the historical behaviour.
- **corrected** — `precond_restrict=True`. **These are the numbers to use.**

Speed-ups are computed **within a family only**. A "speed-up" measured across
the boundary would be comparing the cost of reaching two different answers,
which is meaningless.

The `incompat` column is what tells them apart: it is the non-gradient fraction
of the converged fluctuation field. A physical solution has `incompat ≈ 0`
(~1e-13); legacy runs land around 6e-1.

There is also an optional **control** family (`--control`): unpreconditioned
GMRES, which provably cannot leave the compatible subspace and is therefore
unbiased ground truth. It is slow — roughly 30× the iterations — so it is off
by default. Use it when you want to certify an answer rather than compare cost.

### Truncated runs: the `**!**` flag

A solve that hits the Krylov iteration cap is treated by the solver as a
*failed* solve: it cuts the load step and retries. So a run in which any solve
hit the cap is **truncated** — its iteration count is a lower bound, not a
measurement, and a speed-up computed from it is meaningless. The suite records
`cap_hit` per run, marks such rows `**!**` in `summary.md`, and prints `n/a`
for the speed-up whenever either side of the comparison is truncated.

**This is the single most important thing to check before trusting a number.**
It is also a trap the earlier version of this suite fell into: `--max-gmres-iter`
defaulted to 1000, which is ample for the legacy preconditioner (only 1 of 35
legacy runs ever reached it) but far too tight for the corrected one, which
needs hundreds to thousands of iterations per solve at contrast >= 500. Every
corrected run at contrast >= 500 was silently truncated, producing a cascade —
capped solve -> load-step cut -> sub-step budget exhausted -> case reported
`failed` — that looked like a robustness collapse but was purely the cap.

The default is now 20000, i.e. a safety valve rather than a tuning knob. Pair
it with `--timeout` on a large sweep so one pathological point cannot hold up
the rest (such a run is recorded with status `TIMEOUT`).

Two built-in self-checks are worth knowing about:

- `baseline` vs `baseline-new` should agree **bitwise**. They are the pinned
  old solver and the current solver configured to reproduce it. A difference
  means the refactor changed the legacy path.
- Within the corrected family, `FIX+C5`, `FIX+C5+C6-matrix` and `FIX+C5+C6-mid`
  should agree in `P11` to ~1e-9. The reference tangent is a preconditioner
  choice; after the fix it must not move the answer.

---

## 3. Running it

```bash
# full sweep - defaults to every usable core
python3 FFT_simulation/benchmark_suite.py

# continue after an interruption (nothing is recomputed)
python3 FFT_simulation/benchmark_suite.py --resume

# smoke test first - a few minutes
python3 FFT_simulation/benchmark_suite.py --quick --out Results/bench_smoke

# see the plan without running anything
python3 FFT_simulation/benchmark_suite.py --dry-run

# rebuild the tables from results already on disk
python3 FFT_simulation/benchmark_suite.py --summary-only
```

`--workers` defaults to `len(os.sched_getaffinity(0))`, which respects
cpuset/affinity limits (`os.cpu_count()` would report the host's cores even
inside a container with a smaller quota). Override it if you want to leave
headroom on a shared machine.

Useful options: `--structures 'path/to/*.npz'` (a glob — run several
microstructures for spread), `--contrasts 10 100 1000`, `--configs FIX FIX+C5`,
`--N`, `--increments`, `--control`, `--gmres-restart`, `--max-gmres-iter`,
`--timeout`.

### Memory is the binding constraint, not cores

Each run is a single-threaded process, so 100 workers means 100 concurrent
solves — but each one holds several large arrays: the projection symbol
`Ghat4` (81·N³, **doubled** under Willot because the symbol is complex), the
tangent `K4` (81·N³), the preconditioner symbol (100·N³, also doubled), and the
GMRES restart basis (`restart`·10·N³).

Rough peak per run:

| N | per run | 100 workers |
|---|---|---|
| 31 | ~0.15 GB | ~15 GB |
| 63 | ~1.8 GB | ~180 GB |

At N=63 a 100-worker sweep needs on the order of 180 GB. The script estimates
this, compares against `MemAvailable`, prints a recommended worker count and
pauses 10 s before continuing if it looks unsafe. If you are memory-bound, drop
`--gmres-restart` (the basis is often the single largest allocation — 800 MB
per worker at the N=63 default) before dropping workers.

Thread limits (`OMP_NUM_THREADS` etc. and `FFT_WORKERS`) are set to 1 at the
top of the module, before NumPy is imported, so workers never oversubscribe.
Do not override them upward while also using many workers.

---

## 4. Output layout

```
Results/benchmark_suite/
  manifest.json      git revision, baseline revision, full config matrix, plan
  results.jsonl      one line per completed run, appended live
  summary.md         the human-readable tables
  summary.csv        tidy long-format table, one row per run
  runs/<structure>/c<contrast>/<config>/
      output.csv         average P and F per increment (solver's own output)
      solver_stats.json  per-increment Newton/Krylov counts, forcing terms, timings
      run.log            full solver log for that run
      result.json        the extracted metrics
```

`results.jsonl` is appended as each run finishes and `result.json` is written
per run, so a sweep that dies halfway keeps everything completed so far;
`--resume` picks up from there. Because each run logs to its own `run.log`,
100 parallel runs do not interleave their output.

Recorded per run: status, `cap_hit`, total and per-solve Krylov counts, Newton
count, step cuts, wall and solver time, the full `P11` curve, `incompat`, mean
filler strain, and `F11` min/max.

---

## 5. Interpreting cost

**Krylov iterations are the reliable cost metric.** One iteration is one
application of the operator (9 forward + 9 inverse FFTs on the grid, plus the
tangent contraction) and, when preconditioned, one preconditioner apply. Wall
time is only comparable when the machine is not oversubscribed — if workers
exceed available cores, or memory pressure causes swapping, wall times become
noise while iteration counts stay meaningful.

Newton counts are worth watching alongside: C5 is supposed to cut Krylov
iterations *without* changing the Newton path, so a large jump in Newton count
would signal that the forcing terms are too loose.

---

## 6. What the suite currently shows

Smoke sweep (`--quick`, structure `1_voxel`, N=31, 1 increment) — the numbers
are small but the pattern is the point:

| family | contrast | config | Krylov | speed-up | P11 | incompat |
|---|---|---|---|---|---|---|
| legacy | 10 | baseline | 76 | 1.00x | 1.040961 | — |
| legacy | 10 | baseline-new | 76 | 1.00x | 1.040961 | 6.4e-01 |
| legacy | 10 | C5 | 36 | 2.11x | 1.040961 | 6.4e-01 |
| legacy | 100 | baseline | 311 | 1.00x | 1.083563 | — |
| legacy | 100 | C5 | 130 | 2.39x | 1.083551 | 6.4e-01 |
| corrected | 10 | FIX | 82 | 1.00x | 1.118882 | 1.3e-14 |
| corrected | 10 | FIX+C5 | 41 | 2.00x | 1.118882 | 4.9e-15 |
| corrected | 10 | FIX+C5+Willot | 42 | 1.95x | 1.136561 | 3.9e-15 |
| corrected | 100 | FIX | 4916 (step cuts) | 1.00x | 1.272605 | 2.2e-13 |
| corrected | 100 | **FIX+C5** | **217** | **22.65x** | 1.272605 | 3.2e-13 |
| corrected | 100 | FIX+C5+Willot | 226 | 21.75x | 1.302381 | 2.7e-13 |

Three things worth carrying into a write-up:

1. **`baseline` and `baseline-new` agree exactly**, which is the self-check
   that the refactor left the legacy path alone.

2. **C5 matters far more after the preconditioner fix, not less.** In the
   legacy family it is worth ~2.1-2.4x. In the corrected family at contrast
   100 it is worth **22x**. The reason: the restricted symbol is a weaker
   accelerator, so a fixed inner tolerance of 1e-6 stagnates, hits the
   iteration cap and forces load-step cuts (15 Newton steps instead of 4).
   Adaptive forcing terms never ask for accuracy the preconditioner cannot
   cheaply deliver, so they sidestep the stagnation entirely. C5 and FIX are
   complementary: the fix makes the answer right, and C5 is what makes the
   corrected solver affordable.

3. **The `incompat` column separates the families cleanly** (6.4e-01 vs
   ~1e-13), and the corrected P11 is 7.5 % higher at contrast 10 and 17.4 %
   higher at contrast 100 than the legacy value.

Willot remains iteration-neutral (+2 % / +4 %) but now, with compatible
fields on both sides, its effect on the homogenized stress is measurable
cleanly for the first time: +1.6 % at contrast 10, +2.3 % at contrast 100.

---

## 7. Caveats

- Willot doubles the memory of two of the largest arrays and makes the
  projection contraction more expensive, so its wall time per iteration is
  higher than the spectral scheme's even when iteration counts match.
- A discretization's *accuracy* cannot be judged from this suite alone. The
  standard demonstration is faster local-field convergence under mesh
  refinement, which needs a resolution study (`--N 31` vs `--N 63` on the same
  structure) rather than a single-N comparison.
- The contrast ladder generates its own charge files
  (`FFT_simulation/Run_configs/Charges/bench_c<contrast>.txt`) with the matrix fixed at
  E = 10 and Poisson ratios 0.48 / 0.30. Change `E_MATRIX` / `CHARGE_TEMPLATE`
  in the script if a different material pairing is wanted.
