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
    filter_matrix!(K)
    ds=bp.ds
    oneK=one(eltype(K)) 
    @inbounds for j in axes(K,2) 
        @views K[:,j].*=-ds[j] 
        K[j,j]+=oneK
    end
    return nothing
end

#################
#### HELPERS ####
#################

# ΔN(k,Δk) = N(k+Δk) - N(k)
# Uses Weyl’s law as a fast estimator of eigenvalue count N(k).
# Returns the estimated number of levels in the interval [k, k+Δk].
function delta_weyl(billiard::Bi,k::T,Δk::T;fundamental::Bool=true) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
    # Use Weyl’s law as a fast estimator of counting function N(k); difference gives # levels in [k,k+Δk]
    weyl_law(k+Δk,billiard;fundamental=fundamental)-weyl_law(k,billiard;fundamental=fundamental)
end

# initial_step_from_dos
# Compute an initial guess for Δk from the density of states ρ(k).
# Formula: Δk₀ ≈ m / ρ(k). 
# Ensures Δk₀ is not too small by flooring with min_step.
function initial_step_from_dos(billiard::Bi,k::T,m::Int;fundamental::Bool=true,min_step::Real=1e-6) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
    ρ=max(dos_weyl(k,billiard;fundamental=fundamental),1e-12) # estimate DOS ρ(k); clamp to avoid division by tiny numbers
    max(m/ρ,min_step) # target m levels -> Δk≈m/ρ; enforce a minimum to avoid Δk≈0 in sparse regions
end

# grow_upper_bound
# Start from Δk₀ and geometrically increase Δk until ΔN(k,Δk) ≥ m.
# Capped by the remaining interval length. 
# Returns (Δk, success_flag), where success_flag -> true if m levels reached.
function grow_upper_bound(billiard::Bi,k::T,m,Δk0::T,remaining::T;fundamental::Bool=true,max_grows::Int=60) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
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
function bisect_for_delta_k(billiard::Bi,k::T,m,lo::T,hi::T;fundamental::Bool=true,tol_levels=0.1,maxit::Int=50) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
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
                           fundamental::Bool=true, tol_levels=0.1, maxit::Int=50) where {T<:Real,Bi<:QuantumBilliards.AbsBilliard}
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

# construct_B_matrix
# Build Beyn projected pencil B and left basis Uk from contour integrals.
# Inputs:
#   f   -> ::Fu Function that inplace constructs Fredholm matrix and writes into a matrix buffer
#   Tbuf -> Matrix{Complex} working buffer for contour additions
#   k0  -> disk center (complex, imag≈0)
#   R   -> disk radius
#   nq  -> # trapezoid nodes on Γ
#   r   -> # random probe columns V (≥ expected eigenvalues inside Γ)
#   svd_tol -> absolute SVD cutoff on A0 singular values
# Output:
#   B  -> small rk×rk dense matrix U* A1 * W * Σ^{-1}
#   Uk -> N×rk basis spanning Ran(A0) for retained singular directions
# Notes:
#   1) Forms A0 = (1/2πi)∮ T(z)^{-1} V dz and A1 = (1/2πi)∮ z T(z)^{-1} V dz via LU solves.
#   2) Rank rk determined by Σ[i] ≥ svd_tol (strict absolute threshold).
#   3) If rk == 0, return empty matrices to signal “no roots in window”.
function construct_B_matrix(f::Fu,Tbuf::Matrix{Complex{T}},k0::Complex{T},R::T;nq::Int=32,r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0),use_adaptive_svd_tol=false) where {T<:Real,Fu<:Function}
    # Reference: Beyn, Wolf-Jurgen, An integral method for solving nonlinear eigenvalue problems, 2018, especially Integral algorithm 1 on p14
    # quadrature nodes/weights on contor Γ 
    θ=range(zero(T),TWO_PI;length=nq+1) # the angles that form the complex circle, equally spaced since curvature zero for trapezoidal rule
    θ=θ[1:end-1] # make sure we start at 0 -> 2*pi
    ej=cis.(θ) # e^{iθ} via cis, infinitesimal contribution to speed
    zj=k0.+R.*ej # the actual complex nodes where to take the ks, we choose center around k0
    wj=(R/nq).*ej # Δz/(2π*i) absorbed in weighting as per eq 30/31 in Section 3 in ref.
    N=size(Tbuf,1) # as per integral alogorithm 1 in refe
    V=randn(rng,Complex{T},N,r) # random matrix reused in inner accumulator loop. It is needed not to miss instances of eigenvectors by the operator being orthogonal to them
    A0=zeros(Complex{T},N,r) # the spectral operator A0 = 1 / (2*π*i) * ∮ T^{-1}(z) * V dz
    A1=zeros(Complex{T},N,r) # the spectral operator A1 = 1 / (2*π*i) * ∮ T^{-1}(z) * V * z dz
    X=similar(V) # RHS workspace for sequential LU decomposition at every zj
    # contour accumulation: A0 += wj * (T(zj) \ V), A1 += (wj*zj) * (T(zj) \ V), instead of forming the inverse directly we create a LU factorization object and use ldiv! on it to get the same algebraic operation
    @fastmath begin # cond(Tz) # actually of the real axis the condition numbers of Fredholm A matrix improve greatly!
        @inbounds for j in eachindex(zj)
            fill!(Tbuf,zero(eltype(Tbuf))) # reset the buffer vals
            f(Tbuf,zj[j]) # construct fredholm matrix
            F=lu!(Tbuf,check=false) # LU for the ldiv!
            ldiv!(X,F,V) # make efficient inverse
            α0=wj[j] # 1 / (2*π*i) weight for A0
            α1=wj[j]*zj[j] # 1 / (2*π*i) * z weight for A1
            BLAS.axpy!(α0,vec(X),vec(A0)) # A0 += α0 * X
            BLAS.axpy!(α1,vec(X),vec(A1)) # A1 += α1 * X
        end
    end
    U,Σ,W=svd!(A0;full=false) # thin SVD of A0, revealing rank. The singular values > svd_tol correspond to eigenvalues. If all sv > svd_tol then maybe increase r (expected eigenvalue count) or reduce R (contour around k0), but if increasing r careful with nq. Check ref. section 3 eq. 22
    rk=0
    svd_tol=use_adaptive_svd_tol ? maximum(Σ)*1e-15 : svd_tol
    @inbounds for i in eachindex(Σ)
        if Σ[i]≥svd_tol # filter out those that correspond to actual eigenvalues
            rk+=1 # to determine how big we must construct the matrices below
        else
            break
        end
    end
    if rk==r # increase the eigvals dimension for A0. This should never happen but due to larger oscilations in Weyl's law at higher k this might happen. Has not been found to happen in testing at k=1500.0
        r+r>N && throw(ArgumentError("r > N is impossible: requested r=$(r+r), N=$(N)"))
        return construct_B_matrix(f,Tbuf,k0,R,nq=nq,r=r+r,svd_tol=svd_tol,rng=rng)
    end
    rk==0 && return Matrix{Complex{T}}(undef,N,0),Matrix{Complex{T}}(undef,N,0) # if nothing found early return
    Uk=@view U[:,1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Wk=@view W[:,1:rk]  # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    Σk=@view Σ[1:rk] # take the relevant ones corresponding to eigenvalues as in Integral algorithm 1 on p14 of ref
    # form B = adjoint(U) * A1 * W * Σ^{-1} as in the reference, p14, integral algorithm 1
    tmp=Matrix{Complex{T}}(undef,N,rk)
    mul!(tmp,A1,Wk) # tmp := A1 * Wk, not weighted by inverse diagonal Σk
    @inbounds for j in 1:rk # right-divide by diagonal Σk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    mul!(B,adjoint(Uk),tmp) # B := Uk'*tmp, the final step
    return B,Uk
end

# solve_vect
# Beyn solve that also returns boundary densities (columns of Φ).
# Inputs:
#   solver,basis,pts -> geometry/collocation for building T(k)
#   k      -> disk center
#   dk     -> disk radius
#   nq,r   -> contour quadrature and probe size
#   svd_tol -> absolute SVD threshold for rank reveal
#   res_tol -> residual filter threshold
# Output:
#   λ_keep -> eigenvalue estimates inside disk passing residual check
#   Φ_keep -> corresponding boundary densities (N×nkeep)
#   tens - radii   -> “tension” proxy |λ - k| for kept roots
# Steps:
#   1) construct_B_matrix -> B,Uk
#   2) eigen!(B) -> (λ,Y)
#   3) Φ = Uk * Y
#   4) Filter:
#        geometry -> |λ - k| ≤ dk
#        residual -> ||T(λ) Φ_j|| < res_tol
#   5) tens[j] = ||A(k)v(k)||
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::Complex{T},dk::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol=1e-14,res_tol=1e-8,rng=MersenneTwister(0),auto_discard_spurious=true,use_adaptive_svd_tol=false) where {Ba<:AbstractHankelBasis} where {T<:Real}
    N=length(pts.xy)
    Tbuf=zeros(Complex{T},N,N)  # workspace allocation for contour additions
    fun=(A,z)->fredholm_matrix_complex_k!(A,pts,solver.symmetry,z;multithreaded=multithreaded,kernel_fun=kernel_fun)
    B,Uk=construct_B_matrix(fun,Tbuf,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng,use_adaptive_svd_tol=use_adaptive_svd_tol) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[],Matrix{Complex{T}}(undef,size(Uk,1),0),T[]
    end
    λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    Phi=Uk*Y # Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,length(Phi[:,1])) # all DLP density operators the have same length
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k) # take only those found in the radius R where we have the expected eigenvalues for which r was used
        if d>dk
            keep[j]=false
            continue
        end
        fill!(Tbuf,zero(eltype(Tbuf))) # zero because K is accumulated in symmetry path, reuse from B matrix construction
        fun(Tbuf,λ[j]) # build A(λ[j]) into Tbuf
        mul!(ybuf,Tbuf,@view(Phi[:,j])) # ybuf = T(λ_j)*φ_j, this is a measure of how well we solve the original problem T(λ)v(λ) = 0
        ybuf_norm=norm(ybuf)
        if auto_discard_spurious
            if ybuf_norm≥res_tol # residual criterion, ybuf should be on the order of 1e-13 - 1e-14 for both the imaginary and real part. If larger than that nq must be increased. Check for a small segment with sweep methods like psm/bim/dm at the end of the wanted spectrum to determime of nq is enough for the whole spectrum. If nq large enough use it for whole spectrum
                keep[j]=false
                if ybuf_norm>1e-8
                    if ybuf_norm>1e-6 # heuristic for when usually it is spurious sqrt(eps())
                        @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
                    else # gray zone
                        @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , most probably eigenvalue but too low nq" 
                    end
                else
                    @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
                end
                continue
            end
        end
        push!(tens,ybuf_norm)
    end
    return λ[keep],Phi[:,keep],tens # eigenvalues, DLP density function, "tension - difference from ||A(k)v(k)||, determines badness since for analytic domain it has exponential convergence with exponent nq * N where N is the Fredholm matrix dimension (check ref Abstract)"
end

# Same as solve_vect but returns only eigenvalues and tensions (no Φ).
# Inputs/outputs and filters mirror solve_vect.
# Use when densities are not needed to reduce memory/IO.
function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::Complex{T},dk::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=32,r::Int=48,svd_tol=1e-14,res_tol=1e-8,rng=MersenneTwister(0),auto_discard_spurious=true,use_adaptive_svd_tol=false) where {Ba<:AbstractHankelBasis} where {T<:Real}
    N=length(pts.xy)
    Tbuf=zeros(Complex{T},N,N)  # workspace allocation for contour additions
    fun=(A,z)->fredholm_matrix_complex_k!(A,pts,solver.symmetry,z;multithreaded=multithreaded,kernel_fun=kernel_fun)
    B,Uk=construct_B_matrix(fun,Tbuf,k,dk,nq=nq,r=r,svd_tol=svd_tol,rng=rng,use_adaptive_svd_tol=use_adaptive_svd_tol) # here is where the core of the algorithm is found. Constructs B from step 5 in ref p.14
    if isempty(B) # rk==0
        @info "no_roots_in_window" k0=k R=dk nq=nq svd_tol=svd_tol
        return Complex{T}[],Matrix{Complex{T}}(undef,size(Uk,1),0),T[]
    end
    λ,Y=eigen!(B) # small dense eigendecomposition to get eigenvalues λ are the eigenvalues and v(λ) are the eigenvectors
    Phi=Uk*Y # Φ = U * Y since A0 = U Σ W*, we have A0 * W Σ^{-1} Y = U Y. Each column is now an eigenvector of of T(λ)v(λ) = 0. This is the second layer potential boundary operator now!
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,length(Phi[:,1])) # all DLP density operators the have same length
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k) # take only those found in the radius R where we have the expected eigenvalues for which r was used
        if d>dk
            keep[j]=false
            continue
        end
        fill!(Tbuf,zero(eltype(Tbuf))) # zero because K is accumulated in symmetry path, reuse from B matrix construction
        fun(Tbuf,λ[j]) # build A(λ[j]) into Tbuf
        mul!(ybuf,Tbuf,@view(Phi[:,j])) # ybuf = T(λ_j)*φ_j, this is a measure of how well we solve the original problem T(λ)v(λ) = 0
        ybuf_norm=norm(ybuf)
        if auto_discard_spurious
            if ybuf_norm≥res_tol # residual criterion, ybuf should be on the order of 1e-13 - 1e-14 for both the imaginary and real part. If larger than that nq must be increased. Check for a small segment with sweep methods like psm/bim/dm at the end of the wanted spectrum to determime of nq is enough for the whole spectrum. If nq large enough use it for whole spectrum
                keep[j]=false
                if ybuf_norm>1e-8
                    if ybuf_norm>1e-6 # heuristic for when usually it is spurious sqrt(eps())
                        @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
                    else # gray zone
                        @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , most probably eigenvalue but too low nq" 
                    end
                else
                    @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
                end
                continue
            end
        end
        push!(tens,ybuf_norm)
    end
    return λ[keep],tens # eigenvalues, DLP density function, "tension - difference from ||A(k)v(k)||, determines badness since for analytic domain it has exponential convergence with exponent nq * N where N is the Fredholm matrix dimension (check ref Abstract)"
end

# solve_INFO
# Diagnostic Beyn solve with @info instrumentation.
# Emits:
#   beyn:start -> k0, R, nq, N, r
#   SVD        -> singular values (inspect tail around svd_tol)
#   eigen      -> dense eigentime
#   per-root   -> k, ||T(k)v|| and status vs res_tol
#   STATUS     -> kept, dropped_outside, dropped_residual, max_residual
# Returns:
#   λ_keep, Φ_keep, tens (same as solve_vect)
# Notes:
#   Use to decide nq, r, and svd_tol per geometry/k-band before serious runs.
function solve_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k0::Complex{T},R::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,nq::Int=48,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0),use_adaptive_svd_tol=false) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    Tbuf=zeros(Complex{T},N,N)  # workspace allocation for contour additions
    fun=(A,z)->fredholm_matrix_complex_k!(A,pts,solver.symmetry,z;multithreaded=multithreaded,kernel_fun=kernel_fun)
    θ=range(zero(T),TWO_PI;length=nq+1);θ=θ[1:end-1];ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej
    N=size(Tbuf,1);V=randn(rng,Complex{T},N,r);A0=zeros(Complex{T},N,r);A1=zeros(Complex{T},N,r);X=similar(V)
    @info "beyn:start" k0=k0 R=R nq=nq N=N r=r
    p=Progress(length(zj),1)
    @inbounds for j in eachindex(zj)
        fill!(Tbuf,zero(eltype(Tbuf)))
        fun(Tbuf,zj[j]) 
        F=lu!(Tbuf,check=false)
        ldiv!(X,F,V)
        α0=wj[j];α1=wj[j]*zj[j]
        BLAS.axpy!(α0,vec(X),vec(A0));BLAS.axpy!(α1,vec(X),vec(A1))
        next!(p)
    end
    @time "SVD" U,Σ,W=svd!(A0;full=false)
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
    mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    mul!(B,adjoint(Uk),tmp)
    @time "eigen" ev=eigen!(B)
    λ=ev.values;Y=ev.vectors;Phi=Uk*Y
    keep=trues(length(λ))
    tens=Vector{T}()
    ybuf=Vector{Complex{T}}(undef,size(Phi,1))
    dropped_out=0
    dropped_res=0
    res_keep=T[]
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k0)
        if d>R
            keep[j]=false
            dropped_out+=1
            continue
        end
        fill!(Tbuf,zero(eltype(Tbuf))) 
        fun(Tbuf,λ[j])
        mul!(ybuf,Tbuf,@view(Phi[:,j]))
        @info "k=$(real(λ[j])) ||A(k)v(k)|| = $(norm(ybuf)) < $res_tol"
        ybuf_norm=norm(ybuf)
        if ybuf_norm≥res_tol
            keep[j]=false
            dropped_res+=1
            if ybuf_norm>1e-8
                if ybuf_norm>1e-6 # heuristic for when usually it is spurious sqrt(eps())
                    @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , definitely spurious" 
                else # gray zone
                    @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , most probably eigenvalue but too low nq" 
                end
            else
                @warn "k=$(real(λ[j])) ||A(k)v(k)|| = $(ybuf_norm) > $res_tol , could be spurious or try increasing nq (usually spurious) or lowering residual tolerance" 
            end
            continue
        end
        push!(tens,d)
        push!(res_keep,norm(ybuf))
    end
    kept=count(keep)
    if kept>0
        @info "STATUS: " kept=kept dropped_outside=dropped_out dropped_residual=dropped_res max_residual=maximum(res_keep)
    else
        @info "STATUS: " kept=0 dropped_outside=dropped_out dropped_residual=dropped_res
    end
    return λ[keep],Phi[:,keep],tens
end

####################
#### HIGH LEVEL ####
####################

# computes tensions based on the one or two norm of the matrix operators and scaled to prevent really small norms of ||A(λ)v(λ)||_{1/2}. So it computes it as:
# t_{i} = ||A(λ_i)v(λ_i)||_{1/2} / (||A(λ_i)||_{1/2} * ||v(λ_i)||_{1/2}) with some padding epss in denominator to prevent near zero norms
function compute_normalized_tensions(solver::BoundaryIntegralMethod,pts_all::Vector{BoundaryPointsBIM{T}},ks_all::AbstractVector{T},us_all::Vector{<:AbstractVector{<:Number}};kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true,matnorm::Symbol=:one,epss::Real=1e-15) where {T<:Real}
    @assert length(ks_all)==length(us_all)==length(pts_all)
    tens=Vector{T}(undef,length(ks_all))
    Threads.@threads for i in eachindex(ks_all)
        k=complex(ks_all[i]) # λ_i
        φ=Complex{T}.(us_all[i]) # v_i
        pts=pts_all[i]
        A=fredholm_matrix_complex_k(pts,solver.symmetry,k;multithreaded=multithreaded,kernel_fun=kernel_fun) # get the A(λ_i)
        y=A*φ # A(λ_i)*v(λ_i)
        # there is a choice of operator 1 or 2 norm for the num and denom
        nA=(matnorm==:one ? opnorm(A,1) : matnorm==:two ? opnorm(A,2) : opnorm(A,Inf))
        tens[i]=norm(y,1)/(nA*(norm(φ,1)+epss)+epss) # the tension
    end
    return tens
end

# compute_spectrum
# High-level driver for Beyn over [k1,k2] using Weyl-guided disks. It has no kwargs for custom potentials as it uses weyl's law for the helmholtz kernel under the hood. The user can create his own version of compute_spectrum with the expected number of eigenvalues.
# Steps:
#   1) plan_weyl_windows -> windows of ≈ m levels, with length ≤ 2*Rmax
#   2) beyn_disks_from_windows -> (k0,R) per window
#   3) evaluate_points at each center to freeze collocation
#   4) sanity-check nq via three INFO runs at nq-10, nq, nq+10
#   5) Threads.@threads over windows:
#        solve_vect -> (λ, Φ) per disk
#   6) flatten -> ks_all, us_all, pts_all (no overlap merging needed)
#   7) compute_normalized_tensions -> tens_all (default 1-norm scaling)
# Returns:
#   ks_all, tens_all, us_all, pts_all, tens_normalized_all (ready for boundary/wavefunction use). tens_normalized_all are useful when we have the whole spectrum and can manually inspect the spurious solutions (if any are found)
# Notes:
#   - @error if nq ≤ 10 (insufficient contour resolution).
#   - r typically set to m+15 for headroom; increase nq with m if needed.
function compute_spectrum(solver::BoundaryIntegralMethod,basis::Ba,billiard::Bi,k1::T,k2::T;m::Int=10,Rmax=1.0,nq=48,r=m+15,fundamental=true,svd_tol=1e-14,res_tol=1e-9,auto_discard_spurious=true,multithreaded_matrix=false,multithreaded_ks=true,use_adaptive_svd_tol=false) where {T<:Real,Bi<:AbsBilliard,Ba<:AbstractHankelBasis}
    # Plan how many intervals we will have with the radii and the centers of the radii
    intervals=plan_weyl_windows(billiard,k1,k2;m=m,fundamental=fundamental,Rmax=Rmax)
    k0,R=beyn_disks_from_windows(intervals)
    println("Number of intervals: ",length(intervals))
    println("Final R: ",R[end])
    println("Starting R: ",R[1])
    all_pts_bim = Vector{BoundaryPointsBIM{Float64}}(undef,length(k0))
    for i in eachindex(k0) # calculating collocation points
        all_pts_bim[i]=evaluate_points(solver,billiard,real(k0[i]))
    end
    ks_list=Vector{Vector{T}}(undef,length(k0)) # preallocate ks
    tens_list=Vector{Vector{T}}(undef,length(k0)) # preallocate unnormalized tensions
    phi_list=Vector{Matrix{Complex{T}}}(undef,length(k0)) # preallocate columns of DLPs for each k in ks
    nq≤15 && @error "Do not use less than 15 contour nodes"
    # (LEGACY) do a test solve to see if nq is large enough and inspect the desired tolerances. Also do +/- 10 in nq to see how it varies. If any warnings about solutions that say that it could be a true eigenvalue then that one could be lost and q higher nq might solve it.
    ks,Phi,tens=solve_INFO(solver,basis,all_pts_bim[end],complex(k0[end]),R[end];nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,use_adaptive_svd_tol=use_adaptive_svd_tol)
    ks_list[end]=real.(ks)
    tens_list[end]=tens
    phi_list[end]=Matrix(Phi)
    p=Progress(length(k0),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(k0)[1:end-1]
        ks,Phi,tens=solve_vect(solver,basis,all_pts_bim[i],complex(k0[i]),R[i],nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,auto_discard_spurious=auto_discard_spurious,multithreaded=multithreaded_matrix,use_adaptive_svd_tol=use_adaptive_svd_tol) # we do not need radii in this computation
        ks_list[i]=real.(ks) 
        tens_list[i]=tens # already real
        phi_list[i]=Matrix(Phi)
        next!(p)
    end
    # Now do merging so to get correct types. There are no overlaps here so no need to call overlap_and_merge!
    ks_all=T[];tens_all=T[];us_all=Vector{Complex{T}}[];pts_all=BoundaryPointsBIM{T}[]
    for i in eachindex(k0)
        ks_i=ks_list[i];tens_i=tens_list[i];Phi_i=phi_list[i] 
        n_i=isnothing(Phi_i) ? 0 : size(Phi_i,2) # in case nothing is found so it does not throw error
        for j in 1:n_i
            push!(ks_all,ks_i[j])
            push!(tens_all,tens_i[j])
            push!(us_all,vec(@view Phi_i[:,j]))
            push!(pts_all,all_pts_bim[i]) # repeat same pts for each eigvals in this window
        end
    end
    tens_normalized_all=compute_normalized_tensions(solver,pts_all,ks_all,us_all;matnorm=:one) # use the 1-norm but arbitrary
    return ks_all,tens_all,us_all,pts_all,tens_normalized_all # for wavefunction construction we need ks, us_all, pts_all. The us_all are DLP densities and can be used in boundary_function_BIM function to correctly symmetrize it!
end