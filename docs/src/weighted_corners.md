# Weighted corner densities

`WeightedCornerGrading()` adds local Gaussian panels for the physical boundary
normal derivative. It is an opt-in CFIE discretization; the existing periodic
Kress policies and their defaults are unchanged. It uses the geometry already
provided by BilliardGeometry, including `MushroomBilliard`.

```julia
using QuantumBilliards, BilliardGeometry

billiard = MushroomBilliard(0.4, 1.0; R=1.0) # half stem width, stem height
kernel = CombinedFieldIntegralEquationSolver(12.0, billiard;
    grading=WeightedCornerGrading(), sym_characters=(-1,))
solver = BeynSolver(kernel; nq=40) # or CORKSolver(kernel)
points = evaluate_points(kernel, billiard, 30.2)
ks, residuals, vectors = solve_vectors(solver, points, 30.0, 0.4)
```

The last two numbers define a spectral window centred at 30 with width 0.4.
Solve the `(+1,)` sector as well for the full mushroom spectrum. An empty
`sym_characters` tuple selects the full matrix without symmetry reduction.

## Which density is weighted?

For a Dirichlet wedge with interior angle ``\alpha``, separation of variables
gives the leading local eigenfunction and normal derivative

```math
\psi(r,\theta)\sim a r^\lambda\sin(\lambda\theta),\qquad
q(r)=\partial_n\psi(r)\sim r^{\lambda-1}v(r),\qquad
\lambda=\frac{\pi}{\alpha}.
```

The coefficient ``a`` and the remaining amplitude are determined by the global
eigenstate. The code does not guess them. At a mushroom shoulder,
``\alpha=3\pi/2`` and ``q\sim r^{-1/3}``.

The original combined-potential CFIE has a different unknown, ``\mu``, and
uses ``\psi=-2(D+ikS)\mu``. Multiplying its columns by the power appropriate
to ``q`` would weight the wrong density. Instead this policy selects

```math
G_k(x,y)=\frac{i}{4}H_0^{(1)}(k|x-y|),\quad
\psi=S_kq,\quad
B(k)q=\bigl(I-2K'_k-2ikS_k\bigr)q=0,
```

where ``K'_k`` differentiates the **target** normal. At a Dirichlet eigenvalue,
``S_kq=0`` and ``K'_kq=q/2``, so the combined equation has the same physical
Dirichlet spectrum. The coupling term removes the real-axis exterior Robin
ambiguity. It is an analytic target-normal kernel, not a conjugate transpose
of a matrix evaluated at a complex wavenumber.

## How the power enters the discretization

On a corner span, let the distance from the corner be proportional to
``\xi^p``, with ``\xi=t`` or ``1-t``. The interpolated amplitude is
``v=q/\xi^{p(\lambda-1)}``, while stored unknowns remain the physical nodal
values ``q_j``. Their source cardinal functions are

```math
b_j(t)=L_j(t)\frac{\xi(t)^{p(\lambda-1)}}
                          {\xi(t_j)^{p(\lambda-1)}},\qquad
q(t)\simeq\sum_j b_j(t)q_j.
```

The jump identity remains the nodal identity because ``b_j(t_i)=\delta_{ij}``.
This changes the approximation space and source integration; it is not merely
an invertible diagonal rescaling of the old matrix.

For a reentrant corner the code chooses ``p\lambda=2``. Consequently
``q\,ds`` is a smooth geometric factor times ``\xi v\,dt``. The mushroom has
``p=3`` and ``q=t^{-1}v``. At convex corners, ``p=\lceil3\lambda\rceil/\lambda``
keeps the mapped geometry regular and the source power integral. Higher powers
and logarithms can remain in ``v``; this is the leading-power treatment, not
RCIP or a precomputed full charge basis.

Corners are detected from the signed turn between endpoint tangents on
`full_boundary(billiard)`. Transparent interfaces and symmetry walls are
excluded by BilliardGeometry's physical-boundary reconstruction. Thus the
mushroom's geometric decomposition does not introduce artificial corners.
The corner map is only a coordinate change; the original curves are retained.

## Integration, reconstruction and cost

The starting choices are 12 density nodes on an ordinary panel, 16 on each
corner amplitude panel, and two computational subpanels per corner span.
Source integration uses at least 32 ordinary or 96 corner nodes. These are
independent kernel samples, not matrix unknowns. Short/near intervals are
subdivided automatically, and self interactions use logarithmic product
weights. The source geometry is reused across all contour nodes and spectral
Taylor coefficients; well-separated targets share their source rules.

Very close corner nodes also require stable geometry. Subtracting two absolute
positions can introduce a false normal component into a nearly tangent chord.
For a chart offset ``\delta``, the code instead uses

```math
\Delta\gamma=\delta\int_0^1\gamma'(u-t\delta)\,dt,\qquad
n_u\cdot\Delta\gamma=-\delta^2\int_0^1(1-t)n_u\cdot\gamma''(u-t\delta)\,dt.
```

Lines and circles have exact versions; interactions across a shared vertex
retain the small offsets from that vertex before adding any absolute position.
This avoids magnifying geometric roundoff in the target-normal kernel.

All weighted starting choices and roundoff guards are named and documented at
the tops of `weighted_corner_geometry.jl` and `weighted_corner_assembly.jl`.
The original periodic grading implementation remains in `boundarygrading.jl`.
Normally only ppw is chosen by the user.
Integration counts can be increased independently through the policy constructor
for convergence studies, without increasing the number of density unknowns.
The actual node count includes complete panels and a corner minimum, so it can
exceed the nominal ppw budget, particularly at low ``k``. Compare actual
`length(points)` and `boundary_matrix_size(kernel, points)`, not ppw alone.

Beyn, CORK and expanded BIM reuse the existing Helmholtz Taylor recurrences and
root extraction code. Their residual filters and acceptance tolerances are
unchanged. The optional periodic Chebyshev lookup does not apply to these
panels; that option falls back to the accurate weighted direct evaluation.
Assembly remains quadratic in the number of unknowns at fixed local orders;
dense factorizations retain their cubic cost. Smaller matrices therefore help
most once factorization dominates, although each matrix entry has more source
integration work than a basic periodic Nyström entry.

`BIMEigenstate.vec` stores physical ``q`` on this grid. `wavefunction` integrates
``S_kq`` through the same weighted source cardinals, and `boundary_function`
recovers and Rellich-normalizes ``q`` directly. Computing a selected state with
`compute_eigenstate`/`solve_state`, or a tension with `solve`/`solve_vect`, uses
LAPACK SVD on the weighted matrix: iterative smallest-singular triplets can
fail to converge for a corner-localized state. Beyn and CORK retain their
contour vectors without adding a dense SVD for every root. The boundary norm uses a Jacobi
rule for ``|q|^2ds``, whose endpoint exponent is
``2p\lambda-p-1>-1``. `points.ds` contains the compensated positive Jacobi
weights: useful on the density grid, but less accurate for a bare geometric
perimeter estimate. Use the original curves for length/area. Husimi and boundary-momentum projections
use the independent **linear-density** source rules rather than these norm
weights. Reflection/rotation reduction uses the actual Gaussian panel-node
permutation, and reconstruction restores every symmetry image.

Currently the weighted route supports Float64 CFIE on one ordered closed
physical boundary component, including composite-domain mushrooms. Cusps,
disconnected components and mixing physical-density and legacy layer-density
blocks in `CompositeBIMSolver` are rejected explicitly.

## Reproduce the checks

Run `test/weightedcornertests.jl` for independent Gaussian/logarithmic moments,
weighted cardinals, actual-node symmetry folding, direct/Taylor/CORK agreement,
a circle eigenvalue and field checked against Bessel functions, and all
mushroom roots in both tested reference windows and reflection sectors.

`benchmarks/weighted_corners.jl` compares ppw 8, 10, 12, 15 and 18 in windows
centred at ``k=30`` and ``k=100``. It saves every complex root, residual, actual
matrix size, mesh-reference drift, field-shape drift and source-refinement
error. References increase both mesh and source resolution; they are numerical
references, not exact solutions. Both root counts and the real-part drift
matter: a small imaginary part alone does not certify discretization accuracy.
Use its timing flag only with competing CPU work stopped, and repeat warmed
solves before drawing speed conclusions.

The [measured mushroom convergence and runtime](weighted_corner_results.md)
reports the root counts, actual node savings, field checks and warmed timings.
