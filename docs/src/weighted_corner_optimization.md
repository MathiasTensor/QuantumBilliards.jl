# Weighted-density optimization and angle convergence

The optimization changes only `src/solvers/weighted_corner_assembly.jl` in
production code. The original grading policies and solver algorithms are
unchanged. It keeps the same physical normal density, Gaussian orders,
logarithmic product integration, automatic subdivision, close-corner geometry
and eigenvalue acceptance checks. The [density derivation](weighted_corners.md)
explains why this is an interpolation and integration change, rather than a
diagonal scaling of the old matrix.

## What is reused

A reference Gaussian rule depends on its order and endpoint exponent, not on
the target, spectral parameter or physical panel. Its nodes and ordinary and
logarithmic weights are now cached as read-only arrays. Every physical rule
still owns its mapped coordinates and density-cardinal matrices. Reflecting a
right endpoint therefore cannot reverse a shared rule used elsewhere.

Independent targets prepare their source rules in parallel. The source cache
is locked, read-only during that parallel loop, and updated serially after all
workers finish. The arithmetic for each original line, circle or general curve
is specialized on its concrete type before entering the source-sample loops.
The stable small-chord formulas and their roundoff guards are retained.

For one panel and one Taylor order, the source integration has the form

```math
M_{\ell j}=\sum_s K_{\ell s}W_{sj}
             +\sum_s L_{\ell s}W^{\log}_{sj}.
```

Here ``K`` is the ordinary kernel coefficient, ``L`` its local logarithmic
coefficient, and the real matrices ``W`` and ``W^{\log}`` already include the
weighted density cardinals, source Jacobian and quadrature weights. Packing
the real and imaginary parts of ``K`` and ``L`` into successive rows turns
these sums into small real BLAS matrix products. Scratch belongs to each row
chunk; source rules remain unchanged. Assembly preserves the caller's BLAS
thread setting. On this machine, budgets of one and eight give the same
performance for these small products.

For direct matrix evaluation only, an exactly zero target-normal projection
makes the ``H_1`` and ``J_1`` terms vanish. Those special-function calls can
then be omitted. No small-value threshold is introduced. Taylor transport
continues to evaluate ``H_1`` because it seeds the derivative of ``H_0``.

For complex arguments, orders zero and one are now requested together from
the same AMOS library used by SpecialFunctions. One call shares initialization
and argument reduction; two small stack buffers replace repeated temporary
allocations. Both Hankel and Bessel ``J`` pairs use their unscaled definitions.
The real-argument Bessels path is retained. The wrapper follows the existing
AMOS status policy and falls back to the previous public single-order calls
on other statuses, preserving their exceptions. Its private Fortran interface
is documented in the function docstring and covered by value, underflow and
error-propagation regression tests. No asymptotic approximation is added.

## Isolated assembly measurements

The comparison uses the committed weighted implementation at `2216707d`
and the optimized implementation, with the same dependency environment.
The mushroom has cap radius 1, stem half-width 0.4 and height 1. At requested
ppw 10 and ``k\approx100``, there are 1176 full-boundary unknowns and 588
unknowns in each reflection sector. Times are medians of five warmed trials
on an Apple M3 Max, Julia 1.11.9, eight Julia threads and OpenBLAS ILP64.
Other numerical benchmarks were stopped for these measurements.

| Operation, one sector | Before | After | Time reduction |
|---|---:|---:|---:|
| Fresh source-plan preparation | 0.1602 s | 0.01365 s | 91.5% |
| Warm Taylor matrices, degree 18 | 0.1655 s | 0.1072 s | 35.2% |
| Warm independent matrix at ``100+0.07i`` | 0.1760 s | 0.09663 s | 45.1% |

Source-plan allocation falls from 209.8 MB to 92.1 MB. These are allocated
bytes, not peak resident memory. Taylor allocation changes from 106.2 MB to
106.7 MB because of the private packed blocks; direct allocation decreases
from 97.75 MB to 12.53 MB, an 87.2% reduction.

Every retained Taylor matrix and the direct complex matrix were compared with
the immutable pre-optimization snapshot. The largest relative Frobenius
difference is ``4.77\times10^{-16}``; the largest absolute entry difference
is ``1.20\times10^{-14}``. These are changes in summation order at roundoff
level. They do not replace the separate eigenvalue convergence checks.

The assembly benchmark separates fresh source setup from matrices on an
already prepared mesh. Full-solve totals include both and cannot be obtained
by applying the assembly percentage to the complete solve.

## Complete mushroom solves

The same ppw-10 meshes were timed with both reflection sectors and all 15
roots in ``99.8<k<100.2``. Each warmed repetition constructs a fresh
mesh and source cache and returns the accepted eigenvectors. Compilation and
interior plotting are excluded. Beyn retains its 40 contour nodes and degree-18
Taylor expansion; CORK retains its default degree and search budget. Dense
factorizations and the original root validation algorithms are unchanged.

| Method, both sectors | Before | After | Interpretation |
|---|---:|---:|---|
| Beyn, 3 repetitions | 7.701 s | 6.034 s | 21.6% lower median |
| CORK, 5 repetitions | 2.747 s | 2.486 s | 9.5% lower median |

Every warmup and timed solve returns all 6 odd and 9 even roots. The maximum
complex difference from the separately refined mushroom reference is
``1.947\times10^{-9}`` both before and after. Preparation is a small part of
this complete solve. The paired complex evaluations also accelerate Beyn's
independent direct residual checks; its total speedup is smaller than the
isolated assembly speedup.

CORK is reported from a separate five-repetition follow-up after the original
three-repetition measurements varied substantially between processes (3.109 s
before versus 3.325 s after). Both cohorts are retained in the raw data; the
follow-up uses the same settings and fresh-cache timing protocol. A diagnostic
run confirms the same Krylov dimension 550, 55 operator applications, two
stability checks and passing edge checks in both versions and both sectors.
There is no additional Krylov work introduced by the optimization. These are
machine-specific medians, rather than a universal speedup guarantee.

The [runtime figure](figures/weighted_corner_optimization_runtime.pdf)
shows these complete solves alongside the isolated assembly operations.

## Independent angle references

Each test geometry is a radius-one circular sector,

```math
0<r<1,\qquad 0<\theta<\alpha,
```

with Dirichlet conditions on both straight walls and the circular arc. Its
separated eigenfunctions and exact wavenumbers are

```math
\psi_{m,n}(r,\theta)=J_{m\pi/\alpha}(k_{m,n}r)
                        \sin(m\pi\theta/\alpha),\qquad
k_{m,n}=j_{m\pi/\alpha,n},\quad m,n\ge1.
```

The varied origin angles are ``\pi/3,\pi/2,2\pi/3,3\pi/4,4\pi/3,3\pi/2``
and ``7\pi/4``. Each sector also has two right-angle joins at the arc ends.
The benchmark builds these boundaries from existing BilliardGeometry lines
and arcs. The solver detects every angle from the tangents; it receives no
manually supplied singular exponent.

The contour is centred on ``j_{\pi/\alpha,9}`` and has diameter 0.4. All
angular orders that can contribute roots are enumerated independently. Two
Bessel sign scans, with spacings 0.05 and 0.025, must agree before Float64
bisection references are accepted. Multiplicities and complex numerical
parts are retained. Beyn uses 40 contour nodes, probe size 24 and initial
Taylor degree 18, with unchanged rank and direct nonlinear-residual filters.
This benchmark uses the scalar `solve_vectors` interface, which retains the
specified degree rather than automatically tuning it. The independent Bessel
references check the resulting complete spectra.

Requested ppw values are 6, 8, 10, 12, 15, 18 and 24, for both weighted density
panels and the previous global Kress grading with order 4. The tables record
actual node counts as well: the weighted corner floor can make several
requested ppw values produce the same mesh. A small imaginary part is not
sufficient evidence of accuracy, so the primary error is

```math
\max_j |k_j-k_j^{\mathrm{exact}}|,
```

over every root in the window. A missing or extra root is a failed count,
never a point removed from the error curve.

Leading weighting does not guarantee an analytic remainder. For example,
the angular-ground density on a straight sector wall is

```math
q(r)\propto J_\lambda(kr)/r
 =r^{\lambda-1}(c_0+c_1r^2+c_2r^4+\cdots),\qquad \lambda=\pi/\alpha.
```

Under ``r\propto t^p``, the remainder contains ``t^{2p}``. That power is
fractional for some tested angles. More general corners can also have higher
powers and logarithmic terms. The plots therefore measure convergence rather
than assume one universal ppw saving for every angle.

## Measured angle convergence

All **49 weighted base solves** and **7 doubled-source audits** recover the
complete exact spectrum in their contours. The legacy comparison has **10
count mismatches** among its 49 base solves. Those are marked explicitly in
the figures, including cases with extra roots. A correct count alone also
does not establish accuracy: some low-resolution legacy solves return the
expected number of poorly placed roots.

At requested ppw 24 the maximum complex root errors are:

| Origin angle | Weighted | Legacy global grading, q4 |
|---|---:|---:|
| ``\pi/3`` | ``1.21\times10^{-13}`` | ``1.07\times10^{-13}`` |
| ``\pi/2`` | ``3.55\times10^{-14}`` | ``1.24\times10^{-13}`` |
| ``2\pi/3`` | ``5.35\times10^{-14}`` | ``6.76\times10^{-14}`` |
| ``3\pi/4`` | ``3.21\times10^{-14}`` | ``1.99\times10^{-12}`` |
| ``4\pi/3`` | ``4.03\times10^{-14}`` | ``1.64\times10^{-6}`` |
| ``3\pi/2`` | ``4.29\times10^{-14}`` | ``7.53\times10^{-6}`` |
| ``7\pi/4`` | ``6.72\times10^{-14}`` | ``8.81\times10^{-4}`` |

These tests show the largest benefit at the reentrant corners. Both methods
reach floating-point-scale error at the sharper convex corners when sufficiently
resolved; weighting does not promise a lower error than roundoff.

For the ``10^{-8}`` complex-root target, the smallest passing weighted meshes
among the tested ppw values are:

| Origin angle | Requested ppw | Actual weighted nodes | Smallest passing legacy nodes in this sweep |
|---|---:|---:|---:|
| ``\pi/3`` | 15 | 264 | 376 |
| ``\pi/2`` | 15 | 276 | 420 |
| ``2\pi/3`` | 15 | 312 | 470 |
| ``3\pi/4`` | 10 | 240 | 496 |
| ``4\pi/3`` | 8 | 288 | Target not reached through ppw 24 |
| ``3\pi/2`` | 6 | 264 | Target not reached through ppw 24 |
| ``7\pi/4`` | 8 | 336 | Target not reached through ppw 24 |

Thus the measured node reduction is **30–52%** in the four convex comparisons
where both methods reach the target. This is a comparison of tested meshes,
not a search for globally minimal resolutions. The reentrant cases establish
a stronger advantage within this grid, but an exact passing legacy ppw has
not been measured. The corner floor explains why requested ppw 6 or 8 can
still provide hundreds of weighted unknowns.

Doubling only the source integration counts at ppw 12 changes the roots by at
most ``1.00\times10^{-13}``, retaining every root. This checks source integration
independently of density resolution. All angle-sweep runtime columns are marked
contended and are excluded from performance claims.

The figures include the purpose, exact reference, varied corner and interpretation
on the page itself:

- [Complex root error versus ppw](figures/weighted_corner_angles_exact_error.pdf).
- [Imaginary part versus ppw](figures/weighted_corner_angles_imaginary.pdf).
- [Complex root error versus actual matrix size](figures/weighted_corner_angles_nodes.pdf).

## Reproduction

Run `benchmarks/weighted_corner_optimization.jl` once in each checkout's
environment, with no competing numerical jobs. Set `CORNER_OPT_BASELINE` to
the first output directory for the optimized run to compare every coefficient.
The binary matrix fixtures are temporary and are not archived in the repository.

Use `benchmarks/weighted_corner_runtime.jl` with
`CORNER_RUNTIME_KINDS=weighted` for complete Beyn/CORK solves at fixed ppw 10.
Reference roots come from the separately mesh- and source-refined mushroom
calculation described in [the original feature results](weighted_corner_results.md).

Run `benchmarks/weighted_corner_angles.jl` for the exact-reference ppw sweep,
then `benchmarks/plot_weighted_corner_angles.py` for the figures. Small raw CSV
tables are kept under `benchmarks/results/weighted_corner_optimization` and
`benchmarks/results/weighted_corner_angles`. The figures folder contains PDF
plots only; preview images and binary benchmark caches are kept elsewhere.

The weighted regression file includes 162 Hankel/Bessel sequence checks against
the public SpecialFunctions calls, alongside quadrature, density, symmetry,
matrix and eigenstate checks. Existing all-geometry, boundary and symmetry
suites check compatibility with the original route. The separate
4,200-argument audit for each special-function family
also checks small and large arguments, both signs of the imaginary part,
underflow, and preservation of public exceptions.

Final validation passes **315 weighted checks**, **276 existing all-geometry
checks** and **117 boundary/symmetry checks**: **708 distinct checks** in total.
The original full test entry point's two stale baseline API tests remain as
described in the earlier feature results; they were not changed by this work.
