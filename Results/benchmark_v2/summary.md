# Solver Improvement Benchmark

Krylov = total Krylov iterations over every Newton step of every increment.
Speed-up is within a family only (legacy and corrected do not solve the
same problem: the legacy preconditioner converges to a polluted solution).
`incompat` is the non-gradient fraction of the converged fluctuation field;
it should be ~0 for a physical solution.

**!** marks a run in which some solve hit the Krylov iteration cap. Its
cost is a LOWER BOUND, not a measurement: the solver treats a capped
solve as failed and cuts the load step, so the run is truncated and any
speed-up computed from it is meaningless. Re-run those points with a
larger --max-gmres-iter.

## structure: 1_voxel

### legacy family

| contrast | config | status | Krylov | speedup | Newton | wall (s) | P11 | incompat |
|---|---|---|---|---|---|---|---|---|
| 10 | baseline | converged | 246 | 1.00x | 9 | 77 | 2.659503 |  |
| 10 | baseline-new | converged | 246 | 1.00x | 9 | 79 | 2.659503 | 5.9e-01 |
| 10 | C5 | converged | 115 | 2.14x | 9 | 41 | 2.659501 | 5.9e-01 |
| 10 | C6-matrix | converged | 253 | 0.97x | 9 | 76 | 2.662412 | 5.8e-01 |
| 10 | C6-mid | converged | 242 | 1.02x | 9 | 72 | 2.657317 | 6.0e-01 |
| 10 | C5+C6-matrix | converged | 114 | 2.16x | 9 | 41 | 2.662411 | 5.8e-01 |
| 10 | C5+C6-mid | converged | 103 | 2.39x | 9 | 37 | 2.657264 | 6.0e-01 |
| 100 | baseline | converged | 1024 | 1.00x | 12 | 286 | 2.751416 |  |
| 100 | baseline-new | converged | 1024 | 1.00x | 12 | 257 | 2.751416 | 6.0e-01 |
| 100 | C5 | converged | 447 | 2.29x | 12 | 114 | 2.751447 | 6.0e-01 |
| 100 | C6-matrix | converged | 1130 | 0.91x | 12 | 285 | 2.747228 | 5.8e-01 |
| 100 | C6-mid | converged | 1028 | 1.00x | 12 | 254 | 2.750971 | 6.0e-01 |
| 100 | C5+C6-matrix | converged | 505 | 2.03x | 12 | 125 | 2.747483 | 5.8e-01 |
| 100 | C5+C6-mid | converged | 501 | 2.04x | 12 | 126 | 2.750995 | 6.0e-01 |
| 500 | baseline | converged | 4554 | 1.00x | 24 | 797 | 2.788222 |  |
| 500 | baseline-new | converged | 4554 | 1.00x | 24 | 774 | 2.788222 | 6.0e-01 |
| 500 | C5 | converged | 1736 | 2.62x | 21 | 364 | 2.788252 | 6.0e-01 |
| 500 | C6-matrix | converged | 5104 | 0.89x | 21 | 835 | 2.760619 | 5.9e-01 |
| 500 | C6-mid | converged | 3179 | 1.43x | 17 | 624 | 2.778284 | 6.1e-01 |
| 500 | C5+C6-matrix | converged | 1787 | 2.55x | 22 | 406 | 2.760773 | 5.9e-01 |
| 500 | C5+C6-mid | converged | 1706 | 2.67x | 21 | 370 | 2.787622 | 6.1e-01 |
| 1000 | baseline | converged | 6578 | 1.00x | 24 | 999 | 2.802905 |  |
| 1000 | baseline-new | converged | 6578 | 1.00x | 24 | 1007 | 2.802905 | 6.1e-01 |
| 1000 | C5 | converged | 2904 | 2.27x | 25 | 551 | 2.801153 | 6.1e-01 |
| 1000 | C6-matrix | converged | 8458 | 0.78x | 24 | 1184 | 2.761604 | 6.1e-01 |
| 1000 | C6-mid | converged | 6622 | 0.99x | 24 | 1003 | 2.802464 | 6.1e-01 |
| 1000 | C5+C6-matrix | converged | 2424 | 2.71x | 24 | 532 | 2.765214 | 6.1e-01 |
| 1000 | C5+C6-mid | converged | 2640 | 2.49x | 25 | 517 | 2.802712 | 6.1e-01 |
| 2500 | baseline | converged | 22709 | 1.00x | 47 | 2490 | 2.827520 |  |
| 2500 | baseline-new | converged | 22709 | 1.00x | 47 | 2464 | 2.827520 | 6.1e-01 |
| 2500 | C5 | converged | 9148 | 2.48x | 56 | 1277 | 2.819600 | 6.1e-01 |
| 2500 | C6-matrix | converged_with_step_cuts | 38410 **!** | n/a | 39 | 3753 | 2.774524 | 5.9e-01 |
| 2500 | C6-mid | converged | 22012 | 1.03x | 47 | 2424 | 2.828528 | 6.1e-01 |
| 2500 | C5+C6-matrix | converged | 3785 | 6.00x | 25 | 673 | 2.776274 | 6.5e-01 |
| 2500 | C5+C6-mid | converged | 8732 | 2.60x | 55 | 1203 | 2.829217 | 6.1e-01 |

### corrected family

| contrast | config | status | Krylov | speedup | Newton | wall (s) | P11 | incompat |
|---|---|---|---|---|---|---|---|---|
| 10 | FIX | converged | 273 | 1.00x | 9 | 87 | 2.887333 | 9.6e-15 |
| 10 | FIX+C5 | converged | 131 | 2.08x | 9 | 50 | 2.887333 | 2.9e-15 |
| 10 | FIX+C5+C6-matrix | converged | 112 | 2.44x | 9 | 46 | 2.887333 | 4.1e-15 |
| 10 | FIX+C5+C6-mid | converged | 126 | 2.17x | 9 | 50 | 2.887333 | 3.9e-14 |
| 10 | FIX+C5+Willot | converged | 134 | 2.04x | 9 | 75 | 2.928567 | 4.2e-15 |
| 10 | FIX+C5+Willot+C6m | converged | 113 | 2.42x | 9 | 77 | 2.928567 | 4.2e-15 |
| 100 | FIX | converged_with_step_cuts | 135238 **!** | n/a | 71 | 11349 | 3.277065 | 1.5e-13 |
| 100 | FIX+C5 | converged | 1388 | n/a | 15 | 325 | 3.278909 | 2.7e-13 |
| 100 | FIX+C5+C6-matrix | converged | 1145 | n/a | 13 | 302 | 3.278910 | 1.5e-13 |
| 100 | FIX+C5+C6-mid | converged | 1481 | n/a | 13 | 337 | 3.278910 | 6.8e-13 |
| 100 | FIX+C5+Willot | converged_with_step_cuts | 2518 | n/a | 23 | 535 | 3.349075 | 2.4e-13 |
| 100 | FIX+C5+Willot+C6m | converged_with_step_cuts | 2675 | n/a | 24 | 570 | 3.349075 | 1.4e-13 |
| 500 | FIX | failed | 103859 **!** | n/a | 6 | 9188 |  | 5.2e-12 |
| 500 | FIX+C5 | converged_with_step_cuts | 106271 **!** | n/a | 58 | 9206 | 3.430619 | 1.5e-12 |
| 500 | FIX+C5+C6-matrix | converged_with_step_cuts | 56935 **!** | n/a | 50 | 5266 | 3.425617 | 9.7e-13 |
| 500 | FIX+C5+C6-mid | converged_with_step_cuts | 106701 **!** | n/a | 65 | 9236 | 3.430618 | 5.2e-12 |
| 500 | FIX+C5+Willot | converged_with_step_cuts | 118028 **!** | n/a | 75 | 9912 | 3.519534 | 1.8e-12 |
| 500 | FIX+C5+Willot+C6m | converged_with_step_cuts | 89309 **!** | n/a | 54 | 7634 | 3.519535 | 9.2e-13 |
| 1000 | FIX | failed | 100000 **!** | n/a | 0 | 8929 |  | 0.0e+00 |
| 1000 | FIX+C5 | failed | 130013 **!** | n/a | 29 | 10996 |  | 6.3e-12 |
| 1000 | FIX+C5+C6-matrix | failed | 176786 **!** | n/a | 80 | 13910 | 2.493501 | 1.7e-12 |
| 1000 | FIX+C5+C6-mid | failed | 207039 **!** | n/a | 114 | 15717 | 2.493494 | 7.8e-12 |
| 1000 | FIX+C5+Willot | failed | 161971 **!** | n/a | 91 | 12719 | 2.568400 | 3.1e-12 |
| 1000 | FIX+C5+Willot+C6m | failed | 163549 **!** | n/a | 60 | 12784 | 2.568405 | 1.5e-12 |
| 2500 | FIX | failed | 100000 **!** | n/a | 0 | 8940 |  | 0.0e+00 |
| 2500 | FIX+C5 | failed | 118489 **!** | n/a | 24 | 10096 |  | 2.0e-11 |
| 2500 | FIX+C5+C6-matrix | failed | 132854 **!** | n/a | 31 | 11154 |  | 2.7e-12 |
| 2500 | FIX+C5+C6-mid | failed | 149125 **!** | n/a | 40 | 12518 |  | 5.9e-11 |
| 2500 | FIX+C5+Willot | failed | 117376 **!** | n/a | 28 | 9712 |  | 1.7e-11 |
| 2500 | FIX+C5+Willot+C6m | failed | 171707 **!** | n/a | 44 | 13159 | 1.423734 | 3.4e-12 |

