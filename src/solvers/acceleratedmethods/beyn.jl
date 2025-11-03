#=
#############################################
########### BEYN CONTOUR METHOD #############
#############################################

MAIN REFERENCE: Beyn, Wolf-Jurgen, An integral method for solving nonlinear eigenvalue problems, 2018

- A Beyn-type contour-integral method for nonlinear eigenproblems T(k)φ=0 arising from BIM.
- On each disk Γ: center k0, radius R, we build
A0 = (1/2πi)∮ T(z)^{-1}V dz,  A1 = (1/2πi)∮ z T(z)^{-1}V dz
then project with the rank-revealing SVD of A0 to a small dense B, and solve eigen!(B).
- Returned eigenpairs are filtered: (i) |k−k0|≤R and (ii) residual ‖T(k)φ‖ below a tolerance.

Workflow - short

- Call compute_spectrum with the BIM solver and try first with default kwargs. If a lot of warnings about a potential eigenvalue with tension just a bit below tolerance is printed then increase nq until they disappear.

Workflow - long:

1) Choose geometry, basis, and symmetry:
- Build billiard, basis, and (optional) symmetry for your problem.
- Construct the BIM solver: solver=BoundaryIntegralMethod(…, billiard, symmetry=…).

2) Plan windows (disks) that each contain ≈m levels:
- Call intervals=plan_weyl_windows(billiard,k1,k2; m, Rmax, fundamental).
- Convert to disks: (k0,R)=beyn_disks_from_windows(intervals).
- R is automatically ≤Rmax, so each disk’s radius is controlled.

3) Collocate once per window:
- For each k0[i], call pts[i]=evaluate_points(solver,billiard,real(k0[i])).

4) Pick numerical contour nodes (geometry dependent):
- nq : number of contour nodes (trapezoid on circle). Start at 30–60; increase if residuals are large.
- r : probe rank (≥ expected #levels per disk). Start at m+10…m+20.
- svd_tol: SVD cutoff on Σ(A0). 1e-12…1e-14 is safe; inspect Σ tail in solve_INFO.
- res_tol: residual tolerance. 1e-10…1e-12 typical; tighten if needed.

5) Sanity-check one disk before the sweep:
- Run solve_INFO on the last (or a representative) disk with nq-10, nq, nq+10.
- Inspect:
* singular values Σ(A0) and detected rank,
* kept/dropped counts and residuals,
* whether increasing nq reduces residuals by orders of magnitude.
- If residuals are too big or spurious roots appear, increase nq (and, if m is larger, also r).

6) Solve all disks:
- Use solve_vect or the high-level compute_spectrum to get ks and densities Φ.

7) Compute “tensions” (post-process, scale-invariant):
- Use compute_normalized_tensions to get  t_i = ‖T(k_i)φ_i‖ / (‖T(k_i)‖ · ‖φ_i‖)  (default 1-norm).

Practical guidance

- If you increase m (more levels per disk), increase both r (≥ m by a margin) and usually nq.
- Non-analytic boundaries converge slower with nq; expect to use higher nq and/or smaller R.
- Use solve_INFO logs to diagnose: Σ(A0) gaps, rank rk, counts kept/dropped, and residual histogram.
- Typical robust defaults: m≈8–12, Rmax≈0.5, nq≈64–96, r≈m+15, svd_tol≈1e-13, res_tol≈1e-10.
- For very high k or intricate geometries, start conservative (smaller R, larger nq) and relax if safe.

=#

#####################################
#### CONSTRUCTORS FOR COMPLEX ks ####
#####################################

# Add the 2D Helmholtz double-layer kernel contribution to M[i,j].
# Discrete collocation (row = target i, column = source j).
#
# Indices / geometry:
#   i     – target (row / collocation) index
#   j     – source (column / integration) index
#   xi,yi – target coordinates (point i)
#   xj,yj – source  coordinates (point j)
#   nxi,nyi – unit outward normal at the target point i
#   nxj,nyj – unit outward normal at the source point j (unused in this kernel)
#
# Physics:
#   k     – complex wavenumber on the contour
#   pref  – prefactor for the DLP kernel; for 2D Helmholtz with G = (i/4)H0^(1)(kr),
#           ∂G/∂n = (ik/4) ( (x−y)·n / r ) H1^(1)(kr). Here we use pref = -im*k/2 to match
#           the BIM’s normalization.
#
# Numerics:
#   tol2  – distance^2 threshold; if |x_i - x_j|^2 ≤ tol2 treat as near-self and
#           return false so the caller can handle the diagonal/near-singular term.
#   scale – optional symmetry/sign scaling (default 1).
#
# Returns:
#   true  – contribution added to M[i,j]
#   false – skipped (caller should add diagonal correction, e.g. κ/(2π), outside)
@inline function _add_pair_default_complex!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},tol2::T,pref::Complex{T};scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real}
    dx=xi-xj;dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    if d2<=tol2
        return false
    end
    d=sqrt(d2)
    invd=inv(d)
    h=pref*SpecialFunctions.hankelh1(1,k*d)
    @inbounds begin
        M[i,j]+=scale*((nxi*dx+nyi*dy)*invd)*h
    end
    return true
end

# Add a custom kernel contribution to M[i,j] using a user supplied callback.
#
# Indices / geometry:
#   i,j       – target (row) and source (column) indices, respectively
#   xi,yi     – target coordinates (point i)
#   xj,yj     – source  coordinates (point j)
#   nxi,nyi   – unit outward normal at the target i
#   nxj,nyj   – unit outward normal at the source j
#
# Kernel:
#   kernel_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k) must return the complex-valued
#   kernel entry K_ij for the chosen BIE at complex k.
#
# Scaling:
#   scale – optional multiplicative factor (used by symmetry images/signs). This is handled internally by compute_kernel_matrix_complex_k functions.
#
# Effect:
#   M[i,j] += scale * kernel_fun(...)
#   Returns true unconditionally (no near-singular short-circuit here).
@inline function _add_pair_custom_complex!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::Complex{T},kernel_fun;scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real}
    val_ij=kernel_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)*scale
    @inbounds begin
        M[i,j]+=val_ij
    end
    return true
end

# Build the complex-k boundary integral operator matrix K for a *single* boundary
# (no explicit symmetry images). Supports either:
#   - default double-layer kernel (fast path, triangular fill + mirror),
#   - or a user-provided `kernel_fun` (full N×N fill).
#
# Inputs:
#   K          - Matrix{Complex}: working buffer Fredholm kernel for reuse
#   bp         – BoundaryPointsBIM: holds xy, normals, curvature κ, and panel ds
#   k          – complex wavenumber on the Beyn contour
#   multithreaded – toggle threaded loops (via @use_threads)
#   kernel_fun – :default for built-in DLP; or a Function callback for custom kernels
#
# Numerics/constants:
#   tol2 = (eps(T))^2 – near-self threshold on squared distance
#   pref = -im*k/2    – prefactor matching the chosen DLP normalization
#
# Strategy (default):
#   * Upper-triangular loop j=1:i, then mirror to (j,i) to fill the matrix.
#   * Off-diagonal: add DLP using target normal at i and H1^(1)(k r).
#   * Diagonal  : when r≈0 (d2≤tol2), insert κ[i]/(2π).
#
# Strategy (custom):
#   * Full N×N loop, each entry via `_add_pair_custom_complex!`.
#
# Output:
#   K::Matrix{Complex{T}} – the assembled kernel (before ds-weighting / identity).
function compute_kernel_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    xy=bp.xy;nrm=bp.normal;κ=bp.curvature;N=length(xy)
    xs=getindex.(xy,1);ys=getindex.(xy,2);nx=getindex.(nrm,1);ny=getindex.(nrm,2)
    tol2=(eps(T))^2;pref=-im*k/2
    if kernel_fun===:default
        @use_threads multithreading=multithreaded for i in 1:N
            xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
            @inbounds for j in 1:i
                dx=xi-xs[j];dy=yi-ys[j];d2=muladd(dx,dx,dy*dy)
                if d2≤tol2
                    K[i,j]=Complex{T}(κ[i]/TWO_PI)
                else
                    d=sqrt(d2);invd=inv(d);h=pref*SpecialFunctions.hankelh1(1,k*d)
                    K[i,j]=(nxi*dx+nyi*dy)*invd*h
                    if i!=j
                        K[j,i]=(nx[j]*(-dx)+ny[j]*(-dy))*invd*h
                    end
                end
            end
        end
    else
        @use_threads multithreading=multithreaded for i in 1:N
            xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
            @inbounds for j in 1:N
                xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
                K[i,j]=kernel_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)
            end
        end
    end
    return nothing
end

# Build the complex-k boundary integral operator matrix K with symmetry images.
# This augments the direct kernel with reflected source contributions according to
# the provided symmetry list (x-axis, y-axis, or origin reflections), including the
# correct parity signs per symmetry.
#
# Inputs:
#   K          - Matrix{Complex}: working buffer Fredholm kernel for reuse
#   bp        – BoundaryPointsBIM (xy, normals, curvature κ, shifts of symmetry axes)
#   symmetry  – Vector of symmetry descriptors (e.g., X/Y/XYReflection with parity)
#   k         – complex wavenumber
#   multithreaded – threading toggle
#   kernel_fun – :default DLP, or custom kernel callback
#
# Reflection controls:
#   add_x,add_y,add_xy – which images to add (x-reflect, y-reflect, both)
#   sxgn, sygn, sxy    – corresponding parity factors ±1 (from `parity`)
#   shift_x, shift_y   – axis shifts of the geometry for correct mirror positions
#
# Strategy:
#   For each target i and source j:
#     - Add direct pair (default/custom).
#     - If add_x:   add source reflected across y-axis:  x -> 2*shift_x - x, y→y, scaled by sxgn.
#     - If add_y:   add source reflected across x-axis:  x -> x, y→2*shift_y - y, scaled by sygn.
#     - If add_xy:  add source reflected across both axes, scaled by sxy.
#   Near-diagonal handling (default only): if the *direct* pair is near, caller adds κ/(2π).
#
# Output:
#   K::Matrix{Complex{T}} fully populated with symmetry images included.
function compute_kernel_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},symmetry::Vector{Any},k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature 
    N=length(xy)
    tol2=(eps(T))^2
    pref=-im*k/2
    add_x=false;add_y=false;add_xy=false # true if the symmetry is present
    sxgn=one(T);sygn=one(T);sxy=one(T) # the scalings +/- depending on the symmetry considerations
    shift_x=bp.shift_x;shift_y=bp.shift_y # the reflection axes shifts from billiard geometry
    have_rot=false
    nrot=1;mrot=0
    cx=zero(T);cy=zero(T)
    @inbounds for s in symmetry # symmetry here is always != nothing
        if hasproperty(s,:axis)
            if s.axis==:y_axis;add_x=true;sxgn=(s.parity==-1 ? -one(T) : one(T)); end
            if s.axis==:x_axis;add_y=true;sygn=(s.parity==-1 ? -one(T) : one(T)); end
            if s.axis==:origin
                add_x=true;add_y=true;add_xy=true
                sxgn=(s.parity[1]==-1 ? -one(T) : one(T))
                sygn=(s.parity[2]==-1 ? -one(T) : one(T))
                sxy=sxgn*sygn
            end
        elseif s isa Rotation
            have_rot=true
            nrot=s.n
            mrot=mod(s.m,nrot)
            cx,cy=s.center
        end
    end
    if have_rot
        ctab,stab,χ=_rotation_tables(T,nrot,mrot)
    end
    isdef=(kernel_fun===:default)
    @use_threads multithreading=multithreaded for i in 1:N # make if instead of elseif since can have >1 symmetry
        xi=xy[i][1]; yi=xy[i][2]; nxi=nrm[i][1]; nyi=nrm[i][2] # i is the target, j is the source
        @inbounds for j in 1:N # since it has non-trivial symmetry we have to do both loops over all indices, not just the upper triangular
            xj=xy[j][1];yj=xy[j][2];nxj=nrm[j][1];nyj=nrm[j][2]
            if isdef
                ok=_add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref)
                if !ok; K[i,j]+=Complex(κ[i]/TWO_PI); end
            else
                _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,kernel_fun)
            end
            if add_x # reflect only over the x axis
                xr=_x_reflect(xj,shift_x);yr=yj
                if isdef 
                    _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxgn)
                else
                    nxjr,nyjr=_x_reflect_normal(nxj,nyj) # the custom kernels might be functions of source normals which actually change under symmetries!
                    _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxjr,nyjr,k,kernel_fun;scale=sxgn)
                end
            end
            if add_y # reflect only over the y axis
                xr=xj;yr=_y_reflect(yj,shift_y)
                if isdef
                    _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sygn)
                else
                    nxjr,nyjr=_y_reflect_normal(nxj,nyj)
                    _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxjr,nyjr,k,kernel_fun;scale=sygn)
                end
            end
            if add_xy # reflect over both the axes
                xr=_x_reflect(xj,shift_x);yr=_y_reflect(yj,shift_y)
                if isdef 
                    _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxy)
                else
                    nxjr,nyjr=_xy_reflect_normal(nxj,nyj)
                    _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxjr,nyjr,k,kernel_fun;scale=sxy)
                end
            end
            if have_rot
                @inbounds for l in 1:nrot-1 # l=0 is the direct term we already added; add l=1..nrot-1
                    cl=ctab[l+1];sl=stab[l+1]
                    xr,yr=_rot_point(xj,yj,cx,cy,cl,sl)
                    phase=χ[l+1]  # e^{i 2π m l / n}, rotations due to being 1d-irreps have real characters
                    if isdef
                        _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=phase)
                    else
                        nxjr,nyjr=_rot_vec(nxj,nyj,cl,sl) # rotate the normals if custom kernel due to potential source normal dependance
                        _add_pair_custom_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxjr,nyjr,k,kernel_fun;scale=phase)
                    end
                end
            end
        end
    end
    return nothing
end

# Assemble the full Fredholm operator A(k) for the DLP formulation at complex k:
#   A(k) = I - K(k) D,   where D = diag(ds) applies panel quadrature weights (right scaling).
#
# Inputs:
#   K          - Matrix{Complex}: working buffer Fredholm matrix for reuse, constructed from the kernel
#   bp        – BoundaryPointsBIM (xy, normals, curvature, panel lengths ds, symmetry shifts)
#   symmetry  – nothing -> no images; Vector -> include symmetry images in K(k)
#   k         – complex wavenumber
#   multithreaded, kernel_fun – passed to compute_kernel_matrix_complex_k(...)
#
# Steps:
#   1) Build kernel matrix K (with/without symmetry) at k.
#   2) Right-scale by panel lengths: for each column j, K[:,j] *= ds[j].
#   3) Form A := -K  and add identity on the diagonal -> A = I - K.
#   4) Change numerical zeros to 0 via filter_matrix!.
#
# Output:
#   A::Matrix{Complex{T}} ready for use in Beyn contour solves (T(z) ≡ A(z)).
function fredholm_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},k::Complex{T};multithreaded::Bool=true,kernel_fun::Union{Symbol,Function}=:default) where {T<:Real}
    if isnothing(symmetry)
        compute_kernel_matrix_complex_k!(K,bp,k;multithreaded=multithreaded,kernel_fun=kernel_fun)
    else
        compute_kernel_matrix_complex_k!(K,bp,symmetry,k,multithreaded=multithreaded,kernel_fun=kernel_fun)
    end
    ds=bp.ds
    oneK=one(eltype(K)) 
    @inbounds for j in axes(K,2) 
        @views K[:,j].*=-ds[j] 
        K[j,j]+=oneK
    end
    filter_matrix!(K)
    return nothing
end

#################
#### HELPERS ####
#################

# ΔN(k,Δk) = N(k+Δk) - N(k)
# Uses Weyl’s law as a fast estimator of eigenvalue count N(k).
# Returns the estimated number of levels in the interval [k, k+Δk].
function delta_weyl(billiard::Bi,k::T,Δk::T;fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    # Use Weyl’s law as a fast estimator of counting function N(k); difference gives # levels in [k,k+Δk]
    weyl_law(k+Δk,billiard;fundamental=fundamental)-weyl_law(k,billiard;fundamental=fundamental)
end

# initial_step_from_dos
# Compute an initial guess for Δk from the density of states ρ(k).
# Formula: Δk₀ ≈ m / ρ(k). 
# Ensures Δk₀ is not too small by flooring with min_step.
function initial_step_from_dos(billiard::Bi,k::T,m::Int;fundamental::Bool=true,min_step::Real=1e-6) where {T<:Real,Bi<:AbsBilliard}
    ρ=max(dos_weyl(k,billiard;fundamental=fundamental),1e-12) # estimate DOS ρ(k); clamp to avoid division by tiny numbers
    max(m/ρ,min_step) # target m levels -> Δk≈m/ρ; enforce a minimum to avoid Δk≈0 in sparse regions
end

# grow_upper_bound
# Start from Δk₀ and geometrically increase Δk until ΔN(k,Δk) ≥ m.
# Capped by the remaining interval length. 
# Returns (Δk, success_flag), where success_flag -> true if m levels reached.
function grow_upper_bound(billiard::Bi,k::T,m,Δk0::T,remaining::T;fundamental::Bool=true,max_grows::Int=60) where {T<:Real,Bi<:AbsBilliard}
    Δk=min(remaining,max(Δk0,eps(k))) # start from Δk0 but: (i) not below machine eps, (ii) not above remaining span
    for _ in 1:max_grows
        if delta_weyl(billiard,k,Δk;fundamental=fundamental)≥m-eps();return Δk,true end # stop once we bracket ≥m levels
        if Δk≥0.999999*remaining;return remaining,false end # cannot grow further—signal that we hit the cap
        Δk=min(remaining,2Δk) # after cap whether we met target
    end
    return Δk,delta_weyl(billiard,k,Δk;fundamental=fundamental)≥m
end

# bisect_for_delta_k
# Given bracket [lo,hi] where ΔN(lo) < m ≤ ΔN(hi), perform bisection to find Δk ≈ m levels.
# Stops when tolerance in level count is satisfied or bracket is sufficiently small.
# Returns the Δk that yields ΔN(k,Δk) ≈ m.
function bisect_for_delta_k(billiard::Bi,k::T,m,lo::T,hi::T;fundamental::Bool=true,tol_levels=0.1,maxit::Int=50) where {T<:Real,Bi<:AbsBilliard}
    @assert hi>lo
    Nlo=delta_weyl(billiard,k,lo;fundamental=fundamental) # evaluate count at lo
    if abs(Nlo-m)≤tol_levels;return lo end # early accept if already within tolerance
    Nhi=delta_weyl(billiard,k,hi;fundamental=fundamental) # evaluate count at hi
    if abs(Nhi-m)≤tol_levels;return hi end #  early accept at hi
    @assert Nhi≥m
    for _ in 1:maxit
        mid=0.5*(lo+hi)  # bisection on Δk
        Nmid=delta_weyl(billiard,k,mid;fundamental=fundamental)  # count at midpoint
        if abs(Nmid-m)≤tol_levels;return mid end # tolerance satisfied
        if Nmid<m;lo=mid else hi=mid end # shrink bracket toward target
        if hi-lo≤max(1e-12,1e-9*max(1.0,k));return 0.5*(lo+hi) end # stop when bracket is tiny (abs or relative to k)
    end
    return 0.5*(lo+hi) # fallback: return midpoint of final bracket
end

# plan_weyl_windows
# Cover interval [k1,k2] with windows each containing ≈ m eigenvalues by Weyl estimate.
# Each window length is capped at ≤ 2*Rmax (so that disk radius R ≤ Rmax).
# Uses: initial_step_from_dos, grow_upper_bound, bisect_for_delta_k.
# Returns a list of (kL,kR) windows.
function plan_weyl_windows(billiard::Bi,k1::T,k2::T; m::Int=10, Rmax::Real=1.0,
                           fundamental::Bool=true, tol_levels=0.1, maxit::Int=50) where {T<:Real,Bi<:AbsBilliard}
    iv=Vector{Tuple{T,T}}() # accumulator of (kL,kR) windows
    k=k1 # left cursor that advances to cover [k1,k2]
    while k<k2-eps() # loop until we reach the right end
        rem_raw=k2-k # remaining span
        rem=rem_raw>2*Rmax ? T(2*Rmax) : rem_raw # enforce max window length 2 * Rmax (so disk radius R≤Rmax)
        if delta_weyl(billiard,k,rem;fundamental=fundamental)≤m+tol_levels && rem==rem_raw
            push!(iv,(k,k2)); break # if tail holds ≤m levels and fits entirely, close with a single window
        end
        Δk0=initial_step_from_dos(billiard,k,m;fundamental=fundamental) # DOS-based initial guess for ~m levels
        hi,ok=grow_upper_bound(billiard,k,m,Δk0,rem;fundamental=fundamental) # geometrically grow Δk until we bracket ≥m or hit cap
        if !ok
            push!(iv,(k,k+hi)); k+=hi; continue # couldn’t hit ≥m due to length cap: accept best effort and advance
        end
        Δk=bisect_for_delta_k(billiard,k,m,0.0,hi;fundamental=fundamental,tol_levels=tol_levels,maxit=maxit) # refine to ≈m levels
        kR=min(k+Δk,k+rem) # still respect max window length
        push!(iv,(k,kR)); k=kR # record window and advance cursor
    end
    return iv # list of windows covering [k1,k2]
end

# beyn_disks_from_windows
# Convert a list of windows [kL,kR] into Beyn method disks:
# center k0 -> midpoint of window, radius R -> half the window length.
# Guarantees R ≤ Rmax because windows were capped.
# Returns (k0,R) arrays for use in contour integration.
function beyn_disks_from_windows(iv::Vector{Tuple{T,T}}) where {T<:Real}
    k0=Vector{Complex{T}}(undef,length(iv)); R=Vector{T}(undef,length(iv)) # disk centers (complex, imag=0) and radii
    @inbounds for (i,(kL,kR)) in pairs(iv) # center = midpoint of window -> matches Beyn circle center
        k0[i]=complex(0.5*(kL+kR)); R[i]=0.5*(kR-kL) # radius = half window length → guarantees |window| ≤ 2Rmax -> R≤Rmax
    end
    return k0,R
end

#################################
#### SOLVERS FOR BEYN METHOD ####
#################################

# Constructs the buffer matrices for Beyn's method
#
# Inputs:
#   - T: Type of the elements (Real or Complex)
#   - N: Size of the matrices
#   - r: Number of rows in the buffer matrices
#   - rng: Random number generator
#
# Outputs:
#   - V::Matrix{Complex{T}}: Buffer matrix for the right-hand side
#   - X::Matrix{Complex{T}}: Buffer matrix for the solution
#   - A0::Matrix{Complex{T}}: Buffer matrix for the first moment
#   - A1::Matrix{Complex{T}}: Buffer matrix for the second moment
function beyn_buffer_matrices(::Type{T},N::Int64,r::Int64,rng::G) where {T<:Real,G}
    V=randn(rng,Complex{T},N,r) # best leave as Complex even for Real problems to avoid issues in ldiv!
    X=similar(V)
    A0=zeros(Complex{T},N,r)
    A1=zeros(Complex{T},N,r)
    return V,X,A0,A1
end

# construct the B matrix as described in Beyn's paper using the Chebyshev Hankel evaluations to circumvent allocations for complex argument Hankel functions. For high k this is unavoidable.
#
# Inputs:
#   - solver::BoundaryIntegralMethod: The BIM solver object
#   - pts::BoundaryPointsBIM: The boundary points
#   - N::Int: Size of the Fredholm matrices
#   - k0::Complex{T}: Center of the contour
#   - R::T: Radius of the contour
#   - nq::Int: Number of quadrature points on the contour (if analytic boundary one can use less due to spectral convergence)
#   - r::Int: number of expected eigenvalues inside + some padding to their number
#   - svd_tol::Real: Tolerance for the SVD truncation
#   - rng::AbstractRNG: Random number generator
#   - use_chebyshev::Bool: Whether to use the Chebyshev Hankel evaluation or the standard one (for low k one can use no interpolations)
#   - n_panels::Int: Number of Chebyshev panels to use (try 2000 for k≈3000 and rmax≈4.0)
#   - M::Int: Degree of Chebyshev polynomials to use (try 200 for k≈3000 and rmax≈4.0)
#   - info::Bool: Whether to print info messages
#   - kernel_fun::Union{Symbol,Function}: Kernel function to use (:default for DLP, or custom function)
#   - multithreaded::Bool: Whether to use multithreading in the kernel matrix assembly
#
# Notes:
#   1) Forms A0 = (1/2πi)∮ T(z)^{-1} V dz and A1 = (1/2πi)∮ z T(z)^{-1} V dz via LU solves.
#   2) Rank rk determined by Σ[i] ≥ svd_tol (strict absolute threshold).
#   3) If rk == 0, return empty matrices to signal “no roots in window”.
function construct_B_matrix(solver::BoundaryIntegralMethod,pts::BoundaryPointsBIM,N::Int,k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0),use_chebyshev=true,n_panels=2000,M=300,info::Bool=false,kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {T<:Real}
    θ=range(zero(T),TWO_PI;length=nq+1)[1:end-1] # remove last point
    ej=cis.(θ) # unit circle points
    zj=k0.+R.*ej # contour points
    wj=(R/nq).*ej # contour weights
    rmin,rmax=estimate_rmin_rmax(pts,solver.symmetry) # estimate geometry extents for hankel plan creation on panels
    info && println("Estimated rmin=$(rmin), rmax=$(rmax) for Chebyshev basis setup")
    plans=Vector{ChebHankelPlanH1x}(undef,length(zj))
    Threads.@threads for i in eachindex(plans) # precompute plans for all contour points. This creates for each zj[i] a piecewise Chebyshev approximation of H1x(z) on [rmin,rmax]
        plans[i]=plan_h1x(zj[i],rmin,rmax,npanels=n_panels,M=M,grading=:geometric,geo_ratio=1.013)
    end
    #TODO Make the Fredholm matrices working buffers from outside to prevent large allocations in a loop (only for RAM critical applications since this is a small part of the actual execution time)
    begin # allocate fredholm matrices buffer and the moment matrices for Beyn
        Tbufs1=[zeros(Complex{T},N,N) for _ in 1:nq] 
        V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    end
    if use_chebyshev # use the chebyshev hankel evaluations for matrix construction. This is faster for large k values where standard hankel evaluations are slow and allocate a lot.
        @blas_1 @time "DLP chebyshev" begin
            compute_kernel_matrices_DLP_chebyshev!(Tbufs1,pts,solver.symmetry,plans;multithreaded=multithreaded,kernel_fun=kernel_fun)   
            assemble_fredholm_matrices!(Tbufs1,pts)
        end
    else
        @blas_1 @time "DLP" begin # use standard hankel evaluations for matrix construction. This is faster for small k values where chebyshev interpolation overhead is not worth it.
            @inbounds for j in eachindex(zj)
                fredholm_matrix_complex_k!(Tbufs1[j],pts,solver.symmetry,zj[j],multithreaded=multithreaded,kernel_fun=kernel_fun) 
            end
        end
    end
    # Now perform the Beyn contour integrations to form A0 and A1. To do this we need to solve T(zj) X = V for each zj and accumulate A0 += wj[j] * X, A1 += wj[j] * zj[j] * X. So as the first step we LU factor all T(zj) matrices to get the Fj factors which are used for ldiv! to efficiently solve the systems.
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs1[1];check=false) # just to get the type
    Fs=Vector{typeof(F1)}(undef,nq)
    Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq # LU factor all T(zj) matrices
        Fs[j]=lu!(Tbufs1[j];check=false)
    end
    xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:) # vector views for BLAS.axpy! operations, to avoid allocations in the loop via reshaping the matrices each time in the loop
    begin
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)
            ldiv!(X,Fs[j],V) # make efficient inverse
            BLAS.axpy!(wj[j],xv,a0v) # A0 += wj[j] * X
            BLAS.axpy!(wj[j]*zj[j],xv,a1v) # A1 += wj[j] * zj[j] * X
        end
    end
    @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false) # thin SVD of A0, revealing rank. The singular values > svd_tol correspond to eigenvalues. If all sv > svd_tol then maybe increase r (expected eigenvalue count) or reduce R (contour around k0), but if increasing r careful with nq. Check ref. section 3 eq. 22
    rk=count(>=(svd_tol),Σ) # filter out those that correspond to actual eigenvalues
    if rk==0 # if nothing found early return
        return Matrix{Complex{T}}(undef,N,0),Matrix{Complex{T}}(undef,N,0)
    end
    if rk==r # increase the eigvals dimension for A0 and reasonably adjust svd_tol if needed!
        r_tmp=r+r
        while r_tmp<N # do again the ldiv + axpy accumulation with larger r until some sv < svd_tol. This does not require another Fredholm matrix construction since the same T(zj) can be used for larger r.
            V,X,A0,A1=beyn_buffer_matrices(T,N,r_tmp,rng)
            xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:)
            @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)  
                ldiv!(X,Fs[j],V)
                BLAS.axpy!(wj[j],xv,a0v)
                BLAS.axpy!(wj[j]*zj[j],xv,a1v)
            end
            U,Σ,W=svd!(A0;full=false)
            rk=count(>=(svd_tol),Σ)
            rk<r_tmp && break
            r_tmp+=r
            r_tmp>N && throw(ArgumentError("r > N is impossible: requested r=$(r_tmp), N=$(N)"));break
        end
    end
    Uk=@view U[:,1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Wk=@view W[:,1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Σk=@view Σ[1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    # form B = adjoint(U) * A1 * W * Σ^{-1} as in the reference, p14, integral algorithm 1
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1,Wk)  # tmp := A1 * Wk, not weighted by inverse diagonal Σk
    @inbounds @simd for j in 1:rk # right-divide by diagonal Σk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp) # B := Uk'*tmp, the final step
    return B,Uk
end

# Performs one contour integration of the Beyn algorithm for a
# single circular window centered at k with radius dk.
# Constructs the B matrix via contour integration (using Chebyshev
# Hankel acceleration if enabled), computes its eigenvalues λ and
# eigenvectors Y, and returns both the spectral data and internal
# matrices needed for residual testing.
#
# Inputs:
#   - solver::BoundaryIntegralMethod : BIM solver object
#   - basis::Ba                      : Hankel basis type
#   - pts::BoundaryPointsBIM         : Boundary discretization
#   - k::Complex{T}                  : Center of contour
#   - dk::T                          : Radius of contour
#
# Keyword options:
#   - nq::Int          : # of quadrature nodes on contour (≥15 recommended)
#   - r::Int           : # of random probe vectors (≥ expected # of roots)
#   - svd_tol::Real    : SVD truncation tolerance for rank detection
#   - res_tol::Real    : residual tolerance (not used directly here)
#   - rng              : random generator
#   - use_chebyshev    : use Chebyshev Hankel evaluation (fast at large k)
#   - n_panels, M      : Chebyshev grid parameters
#   - kernel_fun       : kernel function to use (:default for DLP)
#   - multithreaded    : whether to use multithreading in kernel assembly
#
# Outputs:
#   - λ::Vector{Complex{T}}  : eigenvalues inside the contour
#   - Uk::Matrix{Complex{T}} : left singular vectors (A0 basis)
#   - Y::Matrix{Complex{T}}  : eigenvectors of B (small matrix)
#   - k::Complex{T}          : center of window
#   - dk::T                  : radius of window
#   - pts::BoundaryPointsBIM : geometry of this window
#
# Notes:
#   - This function does not check residuals or remove spurious λ.
#     Use `residual_and_norm_select` afterwards for filtering.
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::Complex{T},dk::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),auto_discard_spurious::Bool=true,use_chebyshev::Bool=true,n_panels=2000,M=300) where {Ba<:AbstractHankelBasis} where {T<:Real}
    N=length(pts.xy)
    B,Uk=construct_B_matrix(solver,pts,N,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,multithreaded=multithreaded,kernel_fun=kernel_fun) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[],Uk,Matrix{Complex{T}}(undef,0,0),k,dk,pts
    end
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    # Now form only relevant cols of Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    #println("Eigenvalues found in window k0=$(k), R=$(dk): ",λ)
    return λ,Uk,Y,k,dk,pts
end

# Lightweight version of `solve_vect` returning only eigenvalues λ.
# Intended for diagnostic or quick scans, not for production use.
#
# WARNING:
#  - Does not compute residuals or discard spurious eigenvalues.
#  - Always verify λ with `residual_and_norm_select` before using
#     them in statistics or physical interpretation.
# -------------------------------------------------------------

function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::Complex{T},dk::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),auto_discard_spurious::Bool=true,use_chebyshev::Bool=true,n_panels::Int=2000,M::Int=300) where {Ba<:AbstractHankelBasis} where {T<:Real}
    N=length(pts.xy)
    B,Uk=construct_B_matrix(solver,pts,N,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,multithreaded=multithreaded,kernel_fun=kernel_fun) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[],Uk,Matrix{Complex{T}}(undef,0,0),k,dk,pts
    end
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    # Now form only relevant cols of Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    #println("Eigenvalues found in window k0=$(k), R=$(dk): ",λ)
    return λ
end

# One Beyn solve with more information (and slower) than solve_vect. Provides useful information about residuals and automatic spurious eigenvalue discarding and sheds a light on the internal workings of the algorithm and if we are using a large enough nq for the contour integration.
# NOTE: Does not use chebyshev hankel evaluations for now, only standard hankel evaluations for clarity.
#
# Inputs:
#   - solver::BoundaryIntegralMethod: The BIM solver object
#   - basis::Ba: The basis type
#   - pts::BoundaryPointsBIM: The boundary points
#   - k0::Complex{T}: Center of the contour
#   - R::T: Radius of the contour
#   - kernel_fun::Union{Symbol,Function}: The kernel function to use (:default for DLP)
#   - multithreaded::Bool: Whether to use multithreading
#   - nq::Int: Number of quadrature points on the contour
#   - r::Int: number of expected eigenvalues inside + some padding to their number
#   - svd_tol::Real: Tolerance for the SVD truncation
#   - res_tol::Real: Tolerance for the residual ||A(k)v(k)||
#   - rng::AbstractRNG: Random number generator
#   - use_adaptive_svd_tol::Bool: Whether to use adaptive svd tolerance based on maximum singular value
#   - auto_discard_spurious::Bool: Whether to automatically discard spurious eigenvalues based on residuals
#
# Outputs:
#   - λ::Vector{Complex{T}}: The eigenvalues found inside the contour
#   - Phi::Matrix{Complex{T}}: The eigenvectors corresponding to the eigenvalues
#   - tens::Vector{T}: The residuals ||A(k)v(k)||
function solve_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k0::Complex{T},R::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=48,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0),use_adaptive_svd_tol=false,auto_discard_spurious=false) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    Tbuf=zeros(Complex{T},N,N)  # workspace allocation for contour additions
    fun=(A,z)->fredholm_matrix_complex_k!(A,pts,solver.symmetry,z;multithreaded=multithreaded,kernel_fun=kernel_fun) # function to build fredholm matrix at complex k, this version does not use chebyshev hankel evaluations
    θ=range(zero(T),TWO_PI;length=nq+1);θ=θ[1:end-1];ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej # contour points and weights
    N=size(Tbuf,1);V=randn(rng,Complex{T},N,r);A0=zeros(Complex{T},N,r);A1=zeros(Complex{T},N,r);X=similar(V) # buffer matrices for Beyn
    @info "beyn:start" k0=k0 R=R nq=nq N=N r=r
    @time "Fredholm + lu! + ldiv! + 2*axpy!" begin
        begin
            @inbounds for j in eachindex(zj)
                fill!(Tbuf,zero(eltype(Tbuf)))
                @blas_1 fun(Tbuf,zj[j])
                @blas_multi MAX_BLAS_THREADS F=lu!(Tbuf,check=false)
                ldiv!(X,F,V)
                α0=wj[j];α1=wj[j]*zj[j]
                BLAS.axpy!(α0,vec(X),vec(A0))
                BLAS.axpy!(α1,vec(X),vec(A1)) # blas multi up to here
            end
        end
    end
    @time "SVD" @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    println("Singular values (<1e-10 tail inspection): ",Σ)
    rk=0
    svd_tol=use_adaptive_svd_tol ? maximum(Σ)*1e-15 : svd_tol
    @inbounds for i in eachindex(Σ)
        if Σ[i]≥svd_tol
            rk+=1
        else
            break
        end
    end
    rk==r && @warn "All singular values are above svd_tol = $(svd_tol), r = $(r) needs to be increased" # in the actual implementation where B matrix is constructed this will increase r by a fixed amount and do the procedure again until we have some singular values under tolerance!
    rk==0 && return Complex{T}[],Matrix{Complex{T}}(undef,N,0),T[]
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    @time "eigen" @blas_multi_then_1 MAX_BLAS_THREADS ev=eigen!(B)
    λ=ev.values;Y=ev.vectors;Phi=Uk*Y
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,size(Phi,1))
    dropped_out=0
    dropped_res=0
    res_keep=T[]
    begin 
        @inbounds for j in eachindex(λ)
            d=abs(λ[j]-k0)
            if d>R
                keep[j]=false
                dropped_out+=1
                continue
            end
            fill!(Tbuf,zero(eltype(Tbuf))) 
            fun(Tbuf,λ[j])
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(ybuf,Tbuf,@view(Phi[:,j]))
            @info "k=$((λ[j])) ||A(k)v(k)|| = $(norm(ybuf)) < $res_tol"
            ybuf_norm=norm(ybuf)
            if auto_discard_spurious
                if ybuf_norm≥res_tol
                    keep[j]=false
                    dropped_res+=1
                    if ybuf_norm>1e-8
                        if ybuf_norm>1e-6 # heuristic for when usually it is spurious sqrt(eps())
                            @warn "k=$((λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
                        else # gray zone
                            @warn "k=$((λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , most probably eigenvalue but too low nq" 
                        end
                    else
                        @warn "k=$((λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
                    end
                    continue
                end
            end
            push!(tens,ybuf_norm)
            push!(res_keep,norm(ybuf))
        end
        kept=count(keep)
        if kept>0
            @info "STATUS: " kept=kept dropped_outside=dropped_out dropped_residual=dropped_res max_residual=maximum(res_keep)
        else
            @info "STATUS: " kept=0 dropped_outside=dropped_out dropped_residual=dropped_res
        end
    end
    return λ[keep],Phi[:,keep],tens
end

#####################################
#### RESIDUAL CALCULATION HELPER ####
#####################################

# Computes the residuals ||A(k)φ|| and normalized residuals for a set of eigenpairs (λ,φ) obtained from Beyn's method. 
# Inputs:
#   - solver::BoundaryIntegralMethod: The BIM solver object
#   - λ::AbstractVector{Complex{T}}: Eigenvalues obtained from Beyn
#   - Uk::AbstractMatrix{Complex{T}}: Left singular vectors from Beyn's method
#   - Y::AbstractMatrix{Complex{T}}: Eigenvectors from the small B
#   - k0::Complex{T}: Center of the contour
#   - R::T: Radius of the contour
#   - pts::BoundaryPointsBIM{T}: Boundary points of the domain
#   - symmetry::Union{Vector{Any},Nothing}: Symmetry information for the BIM
#   - kernel_fun::Union{Symbol,Function}=:default: Kernel function to use
#   - res_tol::T: Residual tolerance for discarding spurious eigenvalues
#   - matnorm::Symbol=:one: Matrix norm to use for normalization (:one, :two, :inf)
#   - epss::Real=1e-15: Small value to avoid division by zero
#   - auto_discard_spurious::Bool=true: Whether to automatically discard spurious eigenvalues based on residuals
#   - collect_logs::Bool=false: Whether to collect logs of the selection process
#   - use_chebyshev::Bool=true: Whether to use Chebyshev Hankel evaluations
#   - n_panels::Int=2000: Number of Chebyshev panels
#   - M::Int=300: Degree of Chebyshev polynomials
# Outputs:
#   - idx::Vector{Int}: Indices of kept eigenpairs
#   - Φ_kept::AbstractMatrix{Complex{T}}: Kept eigenvectors
#   - tens::Vector{T}: Residual norms ||Aφ||
#   - tensN::Vector{T}: Normalized residuals
#   - logs::Vector{String}: Logs of the selection process (if collect_logs is true)
function residual_and_norm_select(solver::BoundaryIntegralMethod,λ::AbstractVector{Complex{T}},Uk::AbstractMatrix{Complex{T}},Y::AbstractMatrix{Complex{T}},k0::Complex{T},R::T,pts::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing};kernel_fun::Union{Symbol,Function}=:default,res_tol::T,matnorm::Symbol=:one,epss::Real=1e-15,auto_discard_spurious::Bool=true,collect_logs::Bool=false,use_chebyshev::Bool=true,n_panels=2000,M=300,multithreaded::Bool=true) where {T<:Real}
    N,rk=size(Uk) # N: size of Fredholm matrices, rk: number of eigenpairs
    Φtmp=Matrix{Complex{T}}(undef,N,rk)  # buffer for Φ_j = Uk*Y[:,j]
    y=Vector{Complex{T}}(undef,N) # temporary vector for Aφ
    keep=falses(rk) # which eigenpairs to keep
    tens=Vector{T}(undef,rk) # residual norms ||A*φ||
    tensN=Vector{T}(undef,rk) # normalized tensions
    logs=collect_logs ? String[] : nothing # optional logging
    if !use_chebyshev
        A_buf=Matrix{Complex{T}}(undef,N,N) # regular fredholm matrix buffer
    else
        Tbufs=[zeros(Complex{T},N,N) for _ in 1:length(λ)] # buffer for contour Fredholm matrices
        plans=Vector{ChebHankelPlanH1x}(undef,length(λ)) # chebyshev hankel plans for all λ
        rmin,rmax=estimate_rmin_rmax(pts,symmetry) # estimate geometry extents for hankel plan creation on panels
        Threads.@threads for i in eachindex(plans);plans[i]=plan_h1x(λ[i],rmin,rmax,npanels=n_panels,M=M,grading=:geometric,geo_ratio=1.013);end # precompute plans for all λ in parallel, inexpensive
        compute_kernel_matrices_DLP_chebyshev!(Tbufs,pts,solver.symmetry,plans;multithreaded=multithreaded,kernel_fun=kernel_fun) # compute all fredholm matrices at once using chebyshev (faster for large k but really memory intensive) #TODO make Tbufs input to avoid large allocations in loops somehow
        assemble_fredholm_matrices!(Tbufs,pts) # assemble fredholm matrices from kernel matrices (I - ds * K)
    end
    vecnorm=matnorm===:one ? (v->norm(v,1)) : matnorm===:two ? (v->norm(v)) : (v->norm(v,Inf)) # vector norm for normalization choice
    @inbounds for j in 1:rk
        λj=λ[j]
        abs(λj-k0)>R && (tens[j]=NaN;tensN[j]=NaN;continue)  # skip out-of-contour eigenvalues
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(@view(Φtmp[:,j]),Uk,@view(Y[:,j])) # Φ_j = Uk * Y[:,j]
        if !use_chebyshev
            fill!(A_buf, zero(eltype(A_buf))) # reset buffer
            fredholm_matrix_complex_k!(A_buf,pts,symmetry,complex(λj);multithreaded=multithreaded,kernel_fun=kernel_fun) # construct fredholm matrix at λj without chebyshev
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(y,A_buf,@view(Φtmp[:,j]))  # y = A(k)Φ
        else
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(y,Tbufs[j],@view(Φtmp[:,j]))  # y = A(k)Φ with chebyshev, same as above but using precomputed Tbufs
        end
        rj=norm(y) # residual norm ||Aφ||
        tens[j]=rj # store residual norm
        nA=if !use_chebyshev
            matnorm===:one ? opnorm(A_buf,1) : matnorm===:two ? opnorm(A_buf,2) : opnorm(A_buf,Inf) # norm of A(k) without chebyshev
        else
            matnorm===:one ? opnorm(Tbufs[j], 1) : matnorm===:two ? opnorm(Tbufs[j], 2) : opnorm(Tbufs[j], Inf) # norm of A(k) with chebyshev
        end
        φn=vecnorm(@view(Φtmp[:,j])) # norm of φ for normalization
        yn=vecnorm(y) # norm of Aφ for normalization with the above choice
        tensN[j]=yn/(nA*(φn+epss)+epss)
        if auto_discard_spurious && rj≥res_tol
            collect_logs && push!(logs,"λ=$(λj) ||Aφ||=$(rj) > $res_tol → DROP")
        else
            keep[j]=true
            collect_logs && push!(logs,"λ=$(λj) ||Aφ||=$(rj) < $res_tol ← KEEP")
        end
    end
    idx=findall(keep) # indices of kept eigenpairs
    Φ_kept=isempty(idx) ? Matrix{Complex{T}}(undef,N,0) : Φtmp[:,idx]  # kept subset of Φ
    return idx,Φ_kept,tens[idx],tensN[idx],(collect_logs ? logs : String[])
end

########################
#### HIGH LEVEL API ####
########################

# Compute all eigenpairs in [k1,k2] via a two-phase Beyn workflow:
#   Phase 1: (per Weyl-balanced disk) build Fredholm matrices along the contour,
#            run Beyn to get provisional eigenvalues λ and subspaces (Uk, Y).
#   Phase 2: validate each λ by computing residuals ||A(λ)φ|| (φ=Uk*Y[:,j]),
#            keep those inside the disk with residual < res_tol.
#
# Inputs:
#   - solver::BoundaryIntegralMethod: The BIM solver object
#   - basis::Ba: The basis type
#   - billiard::Bi: The billiard domain
#   - k1::T: Lower bound of the wavenumber interval
#   - k2::T: Upper bound of the wavenumber interval
#   - m::Int=10: Wanted number of eigenvalues per Weyl window
#   - Rmax::T=one(T): Maximum radius of the Weyl windows (careful not to choose m too large to go over the max windows size)
#   - nq::Int=48: Number of quadrature points on the contour
#   - r::Int=m+15: Number of expected eigenvalues inside each contour +
#   padding to avoid rank saturation
#   - fundamental::Bool=true: Whether to use the fundamental domain for symmetry
#   - svd_tol::Real=1e-12: Tolerance for the SVD truncation in Beyn
#   - res_tol::Real=1e-9: Residual tolerance for discarding
#   - spurious::Bool=true: Whether to discard spurious eigenvalues
#   - multithreaded_matrix::Bool=false: Whether to use multithreading in matrix assembly
#   - use_adaptive_svd_tol::Bool=false: Whether to use adaptive svd tolerance based on maximum singular value
#   - use_chebyshev::Bool=true: Whether to use Chebyshev Hankel evaluations
#   - n_panels::Int=2000: Number of Chebyshev panels, does not affect performance much
#   - M::Int=300: Degree of Chebyshev polynomials, does not affect performance much
#   - kernel_fun::Union{Symbol,Function}=:default: Kernel function to use (:default for DLP)
#   - do_INFO::Bool=true: Whether to run a diagnostic Beyn solve on the last window
#
# Returns:
#   ks      :: Vector{T}                   – kept real wavenumbers Re(λ)
#   tens    :: Vector{T}                   – raw residuals ||A(λ)φ||
#   us      :: Vector{Vector{Complex{T}}}  – kept DLP densities φ (one per eigenvalue)
#   pts     :: Vector{BoundaryPointsBIM{T}}– matching boundary points object per φ
#   tensN   :: Vector{T}                   – normalized residuals (scale-free)
#
# Notes:
#   • m controls the expected #eigs per Weyl window; Rmax caps disk radius.
#   • nq should not be tiny (spectral conv. on analytic boundaries, but use ≥15).
#   • r is the probe rank for Beyn (auto-bumped internally if saturated).
#   • use_chebyshev turns on Chebyshev Hankel evaluation (faster at large k).
function compute_spectrum(solver::BoundaryIntegralMethod,basis::Ba,billiard::Bi,k1::T,k2::T;m::Int=10,Rmax::T=one(T),nq::Int=48,r::Int=m+15,fundamental::Bool=true,svd_tol::Real=1e-12,res_tol::Real=1e-9,auto_discard_spurious::Bool=true,multithreaded_matrix::Bool=true,use_adaptive_svd_tol::Bool=false,use_chebyshev::Bool=true,n_panels=2000,M=300,kernel_fun::Union{Symbol,Function}=:default,do_INFO::Bool=true) where {T<:Real,Bi<:AbsBilliard,Ba<:AbstractHankelBasis}
    @time "weyl windows" intervals=plan_weyl_windows(billiard,k1,k2;m=m,fundamental=fundamental,Rmax=Rmax)
    kL2,kR2=intervals[end-1]
    kL3,kR3=intervals[end]
    len3=kR3-kL3
    if len3≤max(100*eps(k2),1e-9*max(k2,1.0)) # hack to not make the final interval tiny
        if (kR3-kL2)≤2*Rmax+10*eps(k2)
            intervals[end-1]=(kL2,kR3)
            pop!(intervals)
        else
            pop!(intervals)
        end
    end
    @time "beyn disks" k0,R=beyn_disks_from_windows(intervals)
    println("Number of intervals: ",length(intervals)); println("Average R: ",sum(R)/length(R))
    all_pts_bim=Vector{BoundaryPointsBIM{T}}(undef,length(k0))
    @time "Point evaluation" for i in eachindex(k0)
        all_pts_bim[i]=evaluate_points(solver,billiard,real(k0[i]))
    end
    # preallocations of the residual calculation 
    @time "Preallocations 1" begin
        λs=Vector{Vector{Complex{T}}}(undef,length(k0))
        Uks=Vector{Matrix{Complex{T}}}(undef,length(k0))
        Ys=Vector{Matrix{Complex{T}}}(undef,length(k0))
        k0s=Vector{Complex{T}}(undef,length(k0))
        Rs=Vector{T}(undef,length(k0))
    end
    nq≤15 && @error "Do not use less than 15 contour nodes"
    if do_INFO
            @time "solve_INFO last disk" begin # solve last window to check the residuals and imaginary parts of the eigenvalues. They should be around 1e-5 to get all valid statistics, so b might need to increase to achieve it
            _=solve_INFO(solver,basis,all_pts_bim[end],complex(k0[end]),R[end];nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,use_adaptive_svd_tol=use_adaptive_svd_tol,multithreaded=multithreaded_matrix,kernel_fun=kernel_fun)
        end
    end
    p=Progress(length(k0),1)
    @time "Beyn pass (all disks)" begin
        @inbounds for i in eachindex(k0)
            λ,Uk,Y,κ,δ,pts=solve_vect(solver,basis,all_pts_bim[i],complex(k0[i]),R[i];nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),auto_discard_spurious=auto_discard_spurious,multithreaded=multithreaded_matrix,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M,kernel_fun=kernel_fun)
            λs[i]=λ;Uks[i]=Uk;Ys[i]=Y;k0s[i]=κ;Rs[i]=δ
            all_pts_bim[i]=pts 
            next!(p)
        end
    end
    # second pass: residuals + normalized tensions + kept Φ (DLP densities)
    @time "Preparing residuals and tensions storage" begin
        ks_list =Vector{Vector{T}}(undef,length(k0))
        tens_list=Vector{Vector{T}}(undef,length(k0)) # raw ||A(λ)Φ||
        tensN_list=Vector{Vector{T}}(undef,length(k0)) # normalized tensions
        phi_list=Vector{Matrix{Complex{T}}}(undef,length(k0)) # kept Φ per window
    end
    @time "Residuals/tensions pass" begin
        @inbounds @showprogress desc="Computing residuals and tensions" for i in eachindex(k0)
            isempty(λs[i]) && (ks_list[i]=T[];tens_list[i]=T[];tensN_list[i]=T[];phi_list[i]=Matrix{Complex{T}}(undef,length(all_pts_bim[i].xy),0); continue)
            idx,Φ_kept,traw,tnorm,_=residual_and_norm_select(solver,λs[i],Uks[i],Ys[i],k0s[i],Rs[i],all_pts_bim[i],solver.symmetry;kernel_fun=:default,res_tol=T(res_tol),matnorm=:one,epss=1e-15,auto_discard_spurious=auto_discard_spurious,collect_logs=false,use_chebyshev=use_chebyshev,n_panels=n_panels,M=M)
            ks_list[i]=real.(λs[i][idx]) # selected real eigen-wavenumbers
            tens_list[i]=traw # raw residual norms
            tensN_list[i]=tnorm # normalized tensions
            phi_list[i]=Matrix(Φ_kept) # kept DLP columns
        end
    end
    # flatten outputs (+ keep pts per column) and return
    nw=length(phi_list)
    # get number of eigenfunctions found in each window (columns of Φ)
    n_by_win=Vector{Int}(undef,nw);@inbounds for i in 1:nw;n_by_win[i]=size(phi_list[i],2);end
    # compute prefix offsets so we can index into flattened arrays directly
    offs=zeros(Int,nw);@inbounds for i in 2:nw
        offs[i]=offs[i-1]+n_by_win[i-1]
    end
    ntot=offs[end]+n_by_win[end] # total number of eigenpairs found
    ks_all=Vector{T}(undef,ntot)
    tens_all=Vector{T}(undef,ntot)
    tensN_all=Vector{T}(undef,ntot)
    us_all=Vector{Vector{Complex{T}}}(undef,ntot)
    pts_all=Vector{BoundaryPointsBIM{T}}(undef,ntot)
    Threads.@threads for i in 1:nw
        n=n_by_win[i] # number of eigenvalues in this window
        n==0 && continue # skip empty windows (no eigenvalues)
        off=offs[i];ksi=ks_list[i];tr=tens_list[i];tn=tensN_list[i];Φ=phi_list[i];pts=all_pts_bim[i]
        @inbounds for j in 1:n
            ks_all[off+j]=ksi[j]
            tens_all[off+j]=tr[j]
            tensN_all[off+j]=tn[j]
            us_all[off+j]=vec(@view Φ[:,j]) # eigenvector (DLP density)
            pts_all[off+j]=pts
        end
    end
    return ks_all,tens_all,us_all,pts_all,tensN_all
end