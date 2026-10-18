# Weighted-density isolation and second optimization pass

Before: immutable source snapshot of `f4124ce4070443d776cd9b45109044cd7c8ab9c0`.
After: the feature isolation, real-argument Taylor reuse and projection changes
committed with this directory. Geometry dependency:
`BilliardGeometry.jl` at `423daff43f7ab4dd55f5ae411db40bf4ef3929b9`.

Machine: Apple M3 Max, Julia 1.11.9, eight Julia threads, OpenBLAS ILP64.
Isolated measurements ran sequentially without competing numerical jobs.
Micro/projection measurements use one BLAS thread; full solvers retain their
existing LAPACK budget of 12. Compilation and selected-state SVD are excluded
from the full contour-solve measurements.

## Files and interpretation

- `original_source_footprint.csv`: original means a source file already present
  at base `3c41a3e9fd9e0f04d826eaa49d88bbaa2cd328e7`. New feature files and
  documentation are excluded. Modified original files decrease from 16 to 8;
  added lines in those files decrease from 489 to 84. Total feature code is
  not smaller: detailed specialized methods now live in separate feature files.
- `*_micro_*`: five warmed repetitions of source setup and matrix construction
  for one mushroom reflection sector. All 19 normalized Taylor matrices and
  the direct complex matrix agree bit for bit with the snapshot. Taylor-18
  assembly decreases from 0.103111209 to 0.099656667 seconds (3.4%). The other
  microbenchmark differences are too small to establish a further improvement.
- `*_geometry_state_medians.csv`: eleven warmed repetitions. Mesh construction
  decreases from 0.000469708 to 0.000357750 seconds (23.8%); projecting the
  physical density into its independent source samples decreases from
  0.000984667 to 0.000410459 seconds (58.3%).
- `geometry_state_agreement.csv`: the separate geometry/state fixture audit.
  Normalization weights and source amplitudes are unchanged; arc lengths differ
  by at most 8.9e-16 and momentum/Husimi relative differences stay below 2e-16.
  This audit's contended timings are not included in the performance tables.
- `*_full_*`: five warmed complete solves, fresh mesh/source cache per trial,
  both mushroom reflection sectors, every accepted vector retained. Geometry is
  MushroomBilliard(0.4,1.0; R=1.0), ppw 10, 1176 full-boundary nodes, 588
  unknowns per sector, k in [99.8,100.2]. All 15 roots survive and agree with
  the separately refined reference to about 1.95e-9. Beyn uses 40 contour nodes;
  the remaining solver settings are recorded in the settings files. Combined
  medians: Beyn 5.5564 -> 5.6547 seconds, CORK 2.5529 -> 2.5442 seconds.
  These results do not demonstrate an additional full-solve speedup.
- `angle_audit/`: 35 final checks at seven rational-pi angles, ppw 12 and 24,
  plus doubled-source weighted audits at ppw 12. All 21 weighted solves recover
  the complete exact spectrum. Every returned root and count is identical to
  the corresponding previously archived angle case. The one legacy count
  mismatch at pi/2, ppw 12 is preserved and explicitly reported. The maximum
  weighted ppw-24 complex error is 1.22e-13; source doubling changes roots by
  at most 1.00e-13. These runtime columns are not used for speed claims.
  The degree is fixed at 18 through the scalar contour API: default rank and
  direct-residual acceptance remains active, but automatic Taylor-degree
  selection is not exercised. The settings wording explicitly records this.

No binary benchmark fixtures or plot metadata are committed. The existing
105-case convergence archive and its PDF figures remain the complete ppw
sweep; this 35-case subset verifies that the new organization preserves them.

## Reproduction

Run `benchmarks/weighted_corner_optimization.jl` separately in the before and
after environments, with `CORNER_OPT_BASELINE` pointing at the before output
for the after run. Use `CORNER_OPT_REPETITIONS=5`.

Run `benchmarks/weighted_corner_postprocessing.jl OUTPUT` with
`CORNER_GEOMETRY_BASELINE` pointing at the before output for the after run.
It writes the temporary physical-projection fixture and its agreement CSV.

For complete solves run `benchmarks/weighted_corner_runtime.jl OUTPUT` with
`CORNER_RUNTIME_KINDS=weighted`, `CORNER_RUNTIME_METHODS=beyn,cork`,
`CORNER_RUNTIME_PARITIES=-1,1`, `CORNER_RUNTIME_REPETITIONS=5`,
`CORNER_RUNTIME_ISOLATED=true` and `CORNER_RUNTIME_REFERENCES` pointing at the
mesh- and source-refined mushroom reference directory described in
`docs/src/weighted_corner_results.md`.

For the final angle audit run `benchmarks/weighted_corner_angles.jl OUTPUT`
with `ANGLE_BENCH_PPWS=12,24` and `ANGLE_BENCH_BASELINE` pointing at the earlier
full angle output directory. Temporary binary cases are needed for direct
cross-run comparison; raw complex roots are also retained here as CSV.

Regression validation: 327 weighted checks (including 12 new dispatch and
representation checks), 276 existing fast geometry/solver checks, 110 boundary
checks and 7 symmetry checks: 720 passes, zero failures. Existing baseline
exclusions in the fast suite are unchanged; this is not a claim that every
legacy test entry point passes. Method ambiguity detection returns none.
