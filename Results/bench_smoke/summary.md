# Solver Improvement Benchmark

Krylov = total Krylov iterations over every Newton step of every increment.
Speed-up is within a family only (legacy and corrected do not solve the
same problem: the legacy preconditioner converges to a polluted solution).
`incompat` is the non-gradient fraction of the converged fluctuation field;
it should be ~0 for a physical solution.

## structure: 1_voxel

### legacy family

| contrast | config | status | Krylov | speedup | Newton | wall (s) | P11 | incompat |
|---|---|---|---|---|---|---|---|---|
| 10 | baseline | converged | 76 | 1.00x | 3 | 8 | 1.040961 |  |
| 10 | baseline-new | converged | 76 | 1.00x | 3 | 8 | 1.040961 | 6.4e-01 |
| 10 | C5 | converged | 36 | 2.11x | 3 | 4 | 1.040961 | 6.4e-01 |
| 100 | baseline | converged | 311 | 1.00x | 4 | 35 | 1.083563 |  |
| 100 | baseline-new | converged | 311 | 1.00x | 4 | 35 | 1.083563 | 6.4e-01 |
| 100 | C5 | converged | 130 | 2.39x | 4 | 14 | 1.083551 | 6.4e-01 |

### corrected family

| contrast | config | status | Krylov | speedup | Newton | wall (s) | P11 | incompat |
|---|---|---|---|---|---|---|---|---|
| 10 | FIX | converged | 82 | 1.00x | 3 | 8 | 1.118882 | 1.3e-14 |
| 10 | FIX+C5 | converged | 41 | 2.00x | 3 | 5 | 1.118882 | 4.9e-15 |
| 10 | FIX+C5+Willot | converged | 42 | 1.95x | 3 | 9 | 1.136561 | 3.9e-15 |
| 100 | FIX | converged_with_step_cuts | 4916 | 1.00x | 15 | 564 | 1.272605 | 2.2e-13 |
| 100 | FIX+C5 | converged | 217 | 22.65x | 4 | 23 | 1.272605 | 3.2e-13 |
| 100 | FIX+C5+Willot | converged | 226 | 21.75x | 4 | 27 | 1.302381 | 2.7e-13 |

