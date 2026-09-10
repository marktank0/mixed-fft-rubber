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
| 10 | baseline | converged | 246 | 1.00x | 9 | 76 | 2.659503 |  |
| 10 | baseline-new | converged | 246 | 1.00x | 9 | 83 | 2.659503 | 5.9e-01 |
| 10 | C5 | converged | 115 | 2.14x | 9 | 45 | 2.659501 | 5.9e-01 |
| 10 | C6-matrix | converged | 253 | 0.97x | 9 | 77 | 2.662412 | 5.8e-01 |
| 10 | C6-mid | converged | 242 | 1.02x | 9 | 72 | 2.657317 | 6.0e-01 |
| 10 | C5+C6-matrix | converged | 114 | 2.16x | 9 | 40 | 2.662411 | 5.8e-01 |
| 10 | C5+C6-mid | converged | 103 | 2.39x | 9 | 51 | 2.657264 | 6.0e-01 |
| 100 | baseline | converged | 1024 | 1.00x | 12 | 290 | 2.751416 |  |
| 100 | baseline-new | converged | 1024 | 1.00x | 12 | 300 | 2.751416 | 6.0e-01 |
| 100 | C5 | converged | 447 | 2.29x | 12 | 114 | 2.751447 | 6.0e-01 |
| 100 | C6-matrix | converged | 1130 | 0.91x | 12 | 281 | 2.747228 | 5.8e-01 |
| 100 | C6-mid | converged | 1028 | 1.00x | 12 | 287 | 2.750971 | 6.0e-01 |
| 100 | C5+C6-matrix | converged | 505 | 2.03x | 12 | 142 | 2.747483 | 5.8e-01 |
| 100 | C5+C6-mid | converged | 501 | 2.04x | 12 | 132 | 2.750995 | 6.0e-01 |
| 500 | baseline | converged | 4554 | 1.00x | 24 | 762 | 2.788222 |  |
| 500 | baseline-new | converged | 4554 | 1.00x | 24 | 793 | 2.788222 | 6.0e-01 |
| 500 | C5 | converged | 1736 | 2.62x | 21 | 387 | 2.788252 | 6.0e-01 |
| 500 | C6-matrix | converged | 5104 | 0.89x | 21 | 861 | 2.760619 | 5.9e-01 |
| 500 | C6-mid | converged | 3179 | 1.43x | 17 | 617 | 2.778284 | 6.1e-01 |
| 500 | C5+C6-matrix | converged | 1787 | 2.55x | 22 | 390 | 2.760773 | 5.9e-01 |
| 500 | C5+C6-mid | converged | 1706 | 2.67x | 21 | 367 | 2.787622 | 6.1e-01 |
| 1000 | baseline | converged | 6578 | 1.00x | 24 | 991 | 2.802905 |  |
| 1000 | baseline-new | converged | 6578 | 1.00x | 24 | 995 | 2.802905 | 6.1e-01 |
| 1000 | C5 | converged | 2904 | 2.27x | 25 | 576 | 2.801153 | 6.1e-01 |
| 1000 | C6-matrix | converged | 8458 | 0.78x | 24 | 1138 | 2.761604 | 6.1e-01 |
| 1000 | C6-mid | converged | 6622 | 0.99x | 24 | 988 | 2.802464 | 6.1e-01 |
| 1000 | C5+C6-matrix | converged | 2424 | 2.71x | 24 | 511 | 2.765214 | 6.1e-01 |
| 1000 | C5+C6-mid | converged | 2640 | 2.49x | 25 | 502 | 2.802712 | 6.1e-01 |
| 2500 | baseline | converged | 22709 | 1.00x | 47 | 2198 | 2.827520 |  |
| 2500 | baseline-new | converged | 22709 | 1.00x | 47 | 2199 | 2.827520 | 6.1e-01 |
| 2500 | C5 | converged | 9148 | 2.48x | 56 | 1217 | 2.819600 | 6.1e-01 |
| 2500 | C6-matrix | converged_with_step_cuts | 19410 | 1.17x | 39 | 1970 | 2.774524 | 5.9e-01 |
| 2500 | C6-mid | converged | 22012 | 1.03x | 47 | 2161 | 2.828528 | 6.1e-01 |
| 2500 | C5+C6-matrix | converged | 3785 | 6.00x | 25 | 693 | 2.776274 | 6.5e-01 |
| 2500 | C5+C6-mid | converged | 8732 | 2.60x | 55 | 1179 | 2.829217 | 6.1e-01 |

### corrected family

| contrast | config | status | Krylov | speedup | Newton | wall (s) | P11 | incompat |
|---|---|---|---|---|---|---|---|---|
| 10 | FIX | converged | 273 | 1.00x | 9 | 85 | 2.887333 | 9.6e-15 |
| 10 | FIX+C5 | converged | 131 | 2.08x | 9 | 51 | 2.887333 | 2.9e-15 |
| 10 | FIX+C5+C6-matrix | converged | 112 | 2.44x | 9 | 50 | 2.887333 | 4.1e-15 |
| 10 | FIX+C5+C6-mid | converged | 126 | 2.17x | 9 | 52 | 2.887333 | 3.9e-14 |
| 10 | FIX+C5+Willot | converged | 134 | 2.04x | 9 | 76 | 2.928567 | 4.2e-15 |
| 10 | FIX+C5+Willot+C6m | converged | 113 | 2.42x | 9 | 75 | 2.928567 | 4.2e-15 |
| 100 | FIX | failed | 21465 | 1.00x | 82 | 2100 | 2.346093 | 1.4e-13 |
| 100 | FIX+C5 | converged | 1388 | 15.46x | 15 | 324 | 3.278909 | 2.7e-13 |
| 100 | FIX+C5+C6-matrix | converged | 1145 | 18.75x | 13 | 303 | 3.278910 | 1.5e-13 |
| 100 | FIX+C5+C6-mid | converged | 1481 | 14.49x | 13 | 347 | 3.278910 | 6.8e-13 |
| 100 | FIX+C5+Willot | converged_with_step_cuts | 2518 | 8.52x | 23 | 527 | 3.349075 | 2.4e-13 |
| 100 | FIX+C5+Willot+C6m | converged_with_step_cuts | 2675 | 8.02x | 24 | 573 | 3.349075 | 1.4e-13 |
| 500 | FIX | failed | 4997 | 1.00x | 0 | 856 |  | 0.0e+00 |
| 500 | FIX+C5 | converged_with_step_cuts | 34818 | 0.14x | 105 | 2943 | 3.425616 | 1.4e-12 |
| 500 | FIX+C5+C6-matrix | converged_with_step_cuts | 15431 | 0.32x | 47 | 1707 | 3.425617 | 9.7e-13 |
| 500 | FIX+C5+C6-mid | converged_with_step_cuts | 35949 | 0.14x | 106 | 3026 | 3.425612 | 1.7e-12 |
| 500 | FIX+C5+Willot | converged_with_step_cuts | 29767 | 0.17x | 87 | 2627 | 3.519535 | 1.9e-12 |
| 500 | FIX+C5+Willot+C6m | converged_with_step_cuts | 17663 | 0.28x | 46 | 1890 | 3.519535 | 1.3e-12 |
| 1000 | FIX | failed | 5000 | 1.00x | 0 | 856 |  | 0.0e+00 |
| 1000 | FIX+C5 | failed | 9703 | 0.52x | 17 | 1272 |  | 6.3e-12 |
| 1000 | FIX+C5+C6-matrix | failed | 13302 | 0.38x | 27 | 1547 |  | 3.2e-12 |
| 1000 | FIX+C5+C6-mid | failed | 8960 | 0.56x | 14 | 1199 |  | 2.5e-11 |
| 1000 | FIX+C5+Willot | failed | 12123 | 0.41x | 25 | 1459 | 1.396973 | 5.9e-12 |
| 1000 | FIX+C5+Willot+C6m | failed | 11436 | 0.44x | 23 | 1397 | 1.396971 | 3.2e-12 |
| 2500 | FIX | failed | 5000 | 1.00x | 0 | 846 |  | 0.0e+00 |
| 2500 | FIX+C5 | failed | 10904 | 0.46x | 16 | 1359 |  | 1.7e-11 |
| 2500 | FIX+C5+C6-matrix | failed | 6640 | 0.75x | 7 | 977 |  | 0.0e+00 |
| 2500 | FIX+C5+C6-mid | failed | 12312 | 0.41x | 19 | 1470 |  | 1.1e-10 |
| 2500 | FIX+C5+Willot | failed | 13280 | 0.38x | 23 | 1556 |  | 1.9e-11 |
| 2500 | FIX+C5+Willot+C6m | failed | 9350 | 0.53x | 12 | 1233 |  | 0.0e+00 |

