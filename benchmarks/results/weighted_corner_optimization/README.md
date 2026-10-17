# Isolated weighted-density optimization measurements

`before_*` uses the committed weighted implementation at `2216707d`.
`after_*` uses the optimized assembly in this branch, with the same Julia and
dependency environment. Geometry: `MushroomBilliard(0.4,1.0; R=1.0)`, ppw 10,
1176 physical-boundary nodes and 588 unknowns per reflection sector near k=100.

- `*_micro_*`: five warmed repetitions of fresh source setup and warm matrix
  assembly in one sector. Matrix coefficients were saved temporarily and every
  coefficient was compared; `after_micro_agreement.csv` records the differences.
- `*_full_*`: three warmed complete Beyn/CORK repetitions, fresh mesh/cache per
  solve, both parities, all accepted vectors retained. Every root is checked
  against the separately refined ppw-30, doubled-source mushroom reference.
- `*_cork_recheck_*`: a separate five-repetition before/after CORK follow-up.
  The original three-repeat CORK cohort varied between processes, so this
  follow-up and the original measurements are both retained. No filtering or
  search settings changed. These five-repeat medians supply the final CORK row.
- `optimization_runtime.csv`: the paired medians used by the runtime figure.
  Source/matrix rows refer to one sector; full-solve rows sum both sectors.
- `iteration_audit.csv`: a diagnostic with unchanged CORK settings. Both
  implementations stop at Krylov dimension 550, after 55 operator applications,
  retaining all 6 odd and 9 even roots. Its contended elapsed times are excluded.
- `special_function_audit.txt`: the 4,200-argument audit for each AMOS pair family,
  including public-wrapper error and underflow behavior. The grid comprises 240
  log-spaced positive real parts from 1e-14 to 1e3, each at imaginary parts
  -0.2, -1e-8, 0, 1e-8 and 0.2, plus 3,000 random points with log-uniform real
  parts and imaginary parts in [-0.2,0.2], using MersenneTwister seed 7301.

Machine: Apple M3 Max, Julia 1.11.9, eight Julia threads, OpenBLAS ILP64.
Microbenchmarks use one BLAS thread; full solvers retain the existing BLAS/LAPACK
environment with budget 12. No other numerical jobs ran during isolated timing.
Compilation, plotting, and selected-state SVD are excluded from full-solve times.

The maximum root difference is about 1.947e-9 in every full-solve cohort;
all expected roots are retained. See `docs/src/weighted_corner_optimization.md`
for the derivation, convergence figures, limitations and reproduction commands.
