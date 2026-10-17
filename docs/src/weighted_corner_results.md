# Mushroom convergence and runtime

These measurements use the existing
`BilliardGeometry.MushroomBilliard(0.4,1.0; R=1.0)`: cap radius 1, full stem
width 0.8 and height 1, with Dirichlet boundary conditions. Both reflection
sectors are included. The physical boundary has eight original curve segments;
transparent decomposition interfaces are excluded.

The geometry dependency was checked at
[BilliardGeometry revision 423daff4](https://github.com/Quantum-Chaos-Julia/BilliardGeometry.jl/commit/423daff43f7ab4dd55f5ae411db40bf4ef3929b9)
on its `migrate_solvers` branch. No geometry implementation or new package
dependency was added to QuantumBilliards.

## Accuracy and root counts

The windows are ``29.8<k<30.2`` and ``99.8<k<100.2``. Reference calculations
increase weighted ppw from 24 to 30 and double the source integration counts.
Their maximum complex root drift is ``7.5\times10^{-13}`` near 30 and
``4.7\times10^{-13}`` near 100. These are numerical references, not exact
solutions. The independent circle test uses a Bessel-function eigenvalue and
its normalized interior field.

Every weighted test at ppw 8, 10, 12, 15 and 18 retains all **4 roots** near 30
and all **15 roots** near 100. The table reports maximum complex root
difference, including both real and imaginary parts, over both sectors.
Node counts refer to the full physical boundary; symmetry halves the matrix
dimension here.

| Window centre | Requested ppw | Full nodes | Maximum complex root difference | Maximum sampled field-shape difference |
|---|---:|---:|---:|---:|
| 30 | 8 | 480 | ``3.15\times10^{-9}`` | ``2.49\times10^{-7}`` |
| 30 | 12 | 528 | ``3.15\times10^{-9}`` | ``2.49\times10^{-7}`` |
| 30 | 15 | 600 | ``1.42\times10^{-10}`` | ``1.27\times10^{-8}`` |
| 30 | 18 | 672 | ``5.95\times10^{-12}`` | ``1.07\times10^{-9}`` |
| 100 | 8 | 984 | ``2.53\times10^{-7}`` | ``5.88\times10^{-6}`` |
| 100 | 10 | 1176 | ``1.95\times10^{-9}`` | ``5.19\times10^{-7}`` |
| 100 | 12 | 1416 | ``6.60\times10^{-11}`` | ``6.20\times10^{-9}`` |
| 100 | 15 | 1752 | ``3.92\times10^{-12}`` | ``1.91\times10^{-10}`` |
| 100 | 18 | 2088 | ``3.13\times10^{-13}`` | ``2.40\times10^{-11}`` |

The low-band plateau at ppw 8–12 reflects the fixed starting corner panels:
adding some ordinary nodes does not yet refine the dominant corner error.
The policy rounds complete panels and includes a corner floor, so requested
ppw is not identical to the actual node count.

Field-shape differences use 49 interior samples across the stem and cap,
including samples near the cap rim to capture strongly localized states.
One complex scale is fitted to remove the arbitrary eigenvector amplitude
and phase. These are sampled shape checks, not global norm or pointwise error
bounds. The values above use the contour vectors directly; the selected-state
SVD audit is saved separately. Doubling source integration holds the solved
density fixed and checks a different error from mesh refinement.

Near ``k=100``, old global Kress grading with order 4 and ppw 24 uses 2736 nodes
and reaches roughly ``4.3\times10^{-10}`` complex root error. Weighted ppw 12
uses 1416 nodes, **48.2% fewer**, and is more accurate in this test. For the
``10^{-8}`` target, weighted ppw 10 uses 1176 nodes, **57.0% fewer**. Legacy
ppw 18 slightly misses that target; ppw 20 also misses it, so the runtime
comparison uses the smallest tested passing legacy mesh, ppw 24. Graded-grid
convergence is not monotone, and these tests do not identify a universal
minimal ppw.

## Warmed full-solve runtime

The timing window is centred at ``k=100`` with width 0.4. Times below combine
both sectors, retaining all **6 odd plus 9 even roots**, at complex reference
error below ``10^{-8}``. They are medians of three warmed repetitions on an
Apple M3 Max, Julia 1.11.9, eight Julia threads, OpenBLAS ILP64 and a LAPACK
thread budget of 12. Competing numerical jobs were stopped during these runs.

| Method | Legacy q4, ppw 24 | Weighted, ppw 10 | Runtime reduction |
|---|---:|---:|---:|
| Beyn | 13.699 s | 7.633 s | 44.3% |
| CORK | 4.564 s | 3.577 s | 21.6% |

Each repetition builds a fresh mesh and source cache. Totals include source-rule
preparation, matrix/polynomial assembly, factorization, root extraction,
unchanged acceptance checks and returned vectors. Compilation, resolution
selection, wavefunction plotting and selected-state SVD are excluded. Beyn uses
40 contour nodes, probe size 24 and Taylor degree 18. CORK retains its default
polynomial degree 12 and search budget. Maximum complex reference errors in
the timed cases are ``1.95\times10^{-9}`` weighted and ``4.33\times10^{-10}``
legacy. Their acceptance thresholds were not loosened.

Smaller matrices do not translate into the full cubic-factorization saving:
weighted product integration does more work per entry, and validation and root
extraction also contribute. The measured speedup is **1.80× for Beyn** and
**1.28× for CORK** for this fixed accuracy target and machine.

## Reproducibility and validation

The implementation is opt-in through `grading=WeightedCornerGrading()`; see
[the derivation and API](weighted_corners.md). Reproduce the convergence grid
with `benchmarks/weighted_corners.jl`, then use
`benchmarks/weighted_corner_runtime.jl` for isolated timing. The benchmark
scripts retain every complex root and report count/accuracy failures directly.

Small CSV tables are archived in `benchmarks/results/weighted_corners`:
`convergence.csv`, `roots.csv`, `field_targets.csv`, `state_convergence.csv`,
and the `runtime_*.csv` tables. Accuracy-grid timings are explicitly marked
contended; only the separate isolated runtime tables support the speed claims.
Beyn and CORK expose different residual indicators, so their residual columns
must not be compared as if they were the same norm.

The feature checks cover Gaussian/Jacobi/logarithmic moments, physical-density
cardinals, actual-node symmetry, direct/Taylor/CORK matrix agreement, expanded
BIM derivatives, an independent circle eigenstate, all mushroom reference root
counts, source refinement, and physical-density reconstruction/diagnostics.
Selected weighted states use LAPACK SVD: the audit exposed a failed iterative
singular-vector calculation with a misleadingly tiny singular value. The
regression now checks the actual matrix-vector residual as well.

Validation completed with **131 new feature checks**, **276 existing fast
regression checks** and **117 boundary/symmetry compatibility checks** passing
(**524 checks** in total).

The existing full test entry point also contains two stale baseline API tests
(`compute_psi(::SpectralData,...)` and the old DLP constructor without a
billiard argument). Their failures were reproduced on unmodified
`migrate_solvers`; they are outside this feature and are left unchanged.
