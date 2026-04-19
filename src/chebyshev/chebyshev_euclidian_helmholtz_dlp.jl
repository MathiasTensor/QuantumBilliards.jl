#################################################################################
# Contains efficient constructors and evaluators for piecewise-Chebyshev planed
# approximations of the scaled Hankel function H1x(z) = exp(-i z) * H1^(1)(z) 
# for multiple complex wavenumbers k simultaneously. It does this by storing
# multiple Chebyshev plans (one per k) and evaluating them in a single pass
# through the r values needed (not needing to reevaluate the geometries each new pass).
# This is used in the BEYN_CHEBYSHEV method implementation to speed up matrix constructions.
# The accuracy of the final matrices as compared to SpecialFunctions.jl is on the order of 1e-14
# in the relative Frobenius norm, even for large k values (k≈3000 and rmax≈4.0 giving the largest errors
# in the fourth quadrant of the complex plane where there is also the largest condition number of
# the matrix cond≈O(10^4) and absolute error can max out even at 1e-12 due to exponential growth
# there around the roots while relative error is ≈1e-14). There are otherwiese many sources of error
# (comprising of matrix constructions + lu! + ldiv! + axpy! + SVD + eigen)

# API
# compute_kernel_matrices_DLP_chebyshev! - main matrix construction function for DLP kernel with multiple k values using Chebyshev plans.
#   -> depending on the symmetry of the geometry, it calls different accumulation helpers for default or custom kernels.
#
# assemble_fredholm_matrices! - wrapper around compute_kernel_matrices_DLP_chebyshev! to assemble the full Fredholm matrices 
#      adding the identity and the weight corrections if present.
#
# estimate_rmin_rmax - helper function to estimate suitable rmin and rmax values for Chebyshev plans based on geometry size.
#      Critical to evaluate the chebyshev tables with enough panels and polynomial degree to reach high accuracy in the highly 
#      oscillatory regime of large k*r values.
# M0/8/4/26

const TWO_PI=2*pi
const INV_TWO_PI=inv(TWO_PI)

###########################
#### WITHOUT CHEBYSHEV ####
###########################

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
        M[i,j]+=scale*((nxj*dx+nyj*dy)*invd)*h
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
#   bp         – BoundaryPoints{T}: holds xy, normals, curvature κ, and panel ds
#   k          – complex wavenumber on the Beyn contour
#   multithreaded – toggle threaded loops (via @use_threads)
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
function compute_kernel_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},k::Complex{T};multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy;nrm=bp.normal;κ=bp.curvature;N=length(xy)
    xs=getindex.(xy,1);ys=getindex.(xy,2);nx=getindex.(nrm,1);ny=getindex.(nrm,2)
    tol2=(eps(T))^2;pref=im*k/2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i
            dx=xi-xs[j];dy=yi-ys[j];d2=muladd(dx,dx,dy*dy)
            if d2≤tol2
                K[i,j]= -Complex{T}(κ[i]/TWO_PI)
            else
                d=sqrt(d2);invd=inv(d);h=pref*SpecialFunctions.hankelh1(1,k*d)
                K[i,j]=(nx[j]*dx+ny[j]*dy)*invd*h
                if i!=j
                    K[j,i]=(nx[i]*(-dx)+ny[i]*(-dy))*invd*h
                end
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
#   bp        – BoundaryPoints{T} (xy, normals, curvature κ, shifts of symmetry axes)
#   symmetry  – Vector of symmetry descriptors (e.g., X/Y/XYReflection with parity)
#   k         – complex wavenumber
#   multithreaded – threading toggle
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
function compute_kernel_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::Complex{T};multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature 
    N=length(xy)
    tol2=(eps(T))^2
    pref=im*k/2
    add_x=false;add_y=false;add_xy=false # true if the symmetry is present
    sxgn=one(T);sygn=one(T);sxy=one(T) # the scalings +/- depending on the symmetry considerations
    shift_x=bp.shift_x;shift_y=bp.shift_y # the reflection axes shifts from billiard geometry
    have_rot=false
    nrot=1;mrot=0
    cx=zero(T);cy=zero(T)
    s=symmetry
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
    if have_rot
        ctab,stab,χ=_rotation_tables(T,nrot,mrot)
    end
    @use_threads multithreading=multithreaded for i in 1:N # make if instead of elseif since can have >1 symmetry
        xi=xy[i][1]; yi=xy[i][2]; nxi=nrm[i][1]; nyi=nrm[i][2] # i is the target, j is the source
        @inbounds for j in 1:N # since it has non-trivial symmetry we have to do both loops over all indices, not just the upper triangular
            xj=xy[j][1];yj=xy[j][2];nxj=nrm[j][1];nyj=nrm[j][2]
            ok=_add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref)
            if !ok; K[i,j]+= -Complex(κ[i]/TWO_PI); end
            if add_x # reflect only over the x axis
                xr=_x_reflect(xj,shift_x);yr=yj
                nxr=-nxj
                nyr=nyj
                _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=sxgn)
            end
            if add_y # reflect only over the y axis
                xr=xj;yr=_y_reflect(yj,shift_y)
                nxr=nxj
                nyr=-nyj
                _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=sygn)
            end
            if add_xy # reflect over both the axes
                xr=_x_reflect(xj,shift_x);yr=_y_reflect(yj,shift_y)
                nxr=-nxj
                nyr=-nyj
                _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=sxy)
            end
            if have_rot
                @inbounds for l in 1:nrot-1 # l=0 is the direct term we already added; add l=1..nrot-1
                    cl=ctab[l+1];sl=stab[l+1]
                    xr,yr=_rot_point(xj,yj,cx,cy,cl,sl)
                    nxr=cl*nxj-sl*nyj
                    nyr=sl*nxj+cl*nyj
                    phase=χ[l+1]  # e^{i 2π m l / n}, rotations due to being 1d-irreps have real characters
                    _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=phase)
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
#   bp        – BoundaryPoints (xy, normals, curvature, panel lengths ds, symmetry shifts)
#   symmetry  – nothing -> no images; otherwise, a Rotation or Reflection
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
function fredholm_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::Complex{T};multithreaded::Bool=true) where {T<:Real}
    if isnothing(symmetry)
        compute_kernel_matrix_complex_k!(K,bp,k;multithreaded=multithreaded)
    else
        compute_kernel_matrix_complex_k!(K,bp,symmetry,k,multithreaded=multithreaded)
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

#########################################
#### CHEBYSHEV PLANS FOR MULTIPLE KS ####
#########################################

#################################################################################
# Swithcing logic when to start chebyshev interpolation. 
# Below the cutoff, we use the small-argument series expansions for the Hankel functions instead of the Chebyshev evaluation
# (or in the intermadiate region use either small z or direct evaluation to not have to use too high degree poly)
# since the Chebyshev approximation is not accurate near zero due to the singularity. 

# LEGACY FOR COMPATIBILITY
@inline function panel_t_invsqrt_h1x(pl::ChebHankelPlanH1x,r::Float64)
    invsqrt=inv(sqrt(r))
    if r<pl.rmin
        return Int32(0),0.0,invsqrt
    else
        p=_find_panel(pl,r)
        P=pl.panels[p]
        t=(2*r-(P.b+P.a))/(P.b-P.a)
        return Int32(p),t,invsqrt
    end
end

# For any hankel plan - not just h1x like above (not legacy)
@inline function panel_t_h(pl::ChebHankelPlanH,r::Float64)
    if r<pl.rmin
        return Int32(0),0.0
    else
        p=_find_panel(pl,r)
        P=pl.panels[p]
        t=(2*r-(P.b+P.a))/(P.b-P.a)
        return Int32(p),t
    end
end

@inline function _kernel_triplet_from_hankels(k::T,r::T,H0::ComplexF64,H1::ComplexF64,H2::ComplexF64) where {T<:Real}
    kd=k*r
    hK=Complex{T}(zero(T),k/2)*H1
    hdK=Complex{T}(zero(T),-k/2)*r*H0
    hddK=Complex{T}(zero(T),inv(2*k))*((-2+kd*kd)*H1+kd*H2)
    return hK,hdK,hddK
end

####################### MODULAR ACCUMULATION HELPERS (DEFAULT) ###################

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that does not containts symmetry. 
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r (i.e., multiplied by -0.5i*k)
# Operation performed:
#   K[i,j] += ((nxi*dx+nyi*dy)/r) * h
#   and mirror: K[j,i] with target normal at j and displacement (x_j - x_i)=(-dx,-dy).
# Note: if i==j, only the first accumulation is done since diagonal is already handled separately. This part
# is missing in the sym! verison since the diagonal is not handled there because when adding all contribution the diagonal
# is well-behaved and does not need special treatment.
#################################################################################
@inline function _accum_dlp_default_nosym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T}) where {T<:Real}
    dot_src_j=(nxj*dx+nyj*dy)*invr
    @inbounds K[i,j]+=dot_src_j*h
    if i!=j
        dot_src_i=(nxi*(-dx)+nyi*(-dy))*invr
        @inbounds K[j,i]+=dot_src_i*h
    end
    return nothing
end

#################################################################################
# Accumulate the DLP kernel contribution and its first two k-derivatives for one
# non-symmetric pair (i,j), assuming the special-function factors have already
# been evaluated.
#
# Inputs:
#   K,dK,ddK - destination matrices
#   i,j      - target/source indices
#   nxi,nyi  - target normal components
#   nxj,nyj  - source normal components
#   dx,dy    - displacement x_i - x_j
#   invr     - 1 / |x_i - x_j|
#   hK       - scalar kernel factor for K
#   hdK      - scalar kernel factor for dK
#   hddK     - scalar kernel factor for ddK
#
# Operation:
#   Adds to (i,j), and mirrors to (j,i) using the target normal at i.
#
# Note:
#   The diagonal handling is done outside this helper. This function assumes a
#   regular off-diagonal pair.
#################################################################################
@inline function _accum_dlp_triplet_nosym!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},i::Int,j::Int,nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,hK::Complex{T},hdK::Complex{T},hddK::Complex{T}) where {T<:Real}
    dotj=(nxj*dx+nyj*dy)*invr
    @inbounds begin
        K[i,j]+=dotj*hK;dK[i,j]+=dotj*hdK;ddK[i,j]+=dotj*hddK
    end
    if i!=j
        doti=(nxi*(-dx)+nyi*(-dy))*invr
        @inbounds begin
            K[j,i]+=doti*hK;dK[j,i]+=doti*hdK;ddK[j,i]+=doti*hddK
        end
    end
    return nothing
end

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that containts symmetry.
# Inputs:
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r (i.e., multiplied by -0.5i*k)
# Operation performed:
#   K[i,j] += ((nxi*dx+nyi*dy)/r) * h
#   and mirror: K[j,i] with target normal at j and displacement (x_j - x_i)=(-dx,-dy).
# Note: if i==j, only the first accumulation is done since diagonal is already handled separately. This part
# is missing in the sym! verison since the diagonal is not handled there because when adding all contribution the diagonal
# is well-behaved and does not need special treatment.
#################################################################################
@inline function _accum_dlp_default_sym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T}) where {T<:Real}
    dot_src_j=(nxj*dx+nyj*dy)*invr
    @inbounds K[i,j]+=dot_src_j*h
    return nothing
end

#################################################################################
# Accumulate the DLP kernel contribution and its first two k-derivatives for one
# symmetry-image pair (i,j), assuming the special-function factors have already
# been evaluated.
#
# Inputs:
#   K,dK,ddK - destination matrices
#   i,j      - target/source indices
#   nxj,nyj  - image-source normal components
#   dx,dy    - displacement x_i - x_j^(image)
#   invr     - 1 / |x_i - x_j^(image)|
#   hK       - scalar kernel factor for K
#   hdK      - scalar kernel factor for dK
#   hddK     - scalar kernel factor for ddK
#
# Operation:
#   Adds only to entry (i,j). No mirroring is done here, because symmetry-image
#   contributions are not symmetric under the same triangular shortcut as the
#   direct no-symmetry term.
#################################################################################
@inline function _accum_dlp_triplet_sym!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},i::Int,j::Int,nxj::T,nyj::T,dx::T,dy::T,invr::T,hK::Complex{T},hdK::Complex{T},hddK::Complex{T}) where {T<:Real}
    dotj=(nxj*dx+nyj*dy)*invr
    @inbounds begin
        K[i,j]+=dotj*hK;dK[i,j]+=dotj*hdK;ddK[i,j]+=dotj*hddK
    end
    return nothing
end

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that does not containts symmetry. This is just a wrapper that accepts seperately the hankel value and the prefactor.
# Inputs:
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r.
#   scale: the prefactor computed separately (-im*k/2)
#################################################################################
@inline function _accum_dlp_default_nosym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,
nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T},scale)::Nothing where {T<:Real}
    _accum_dlp_default_nosym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,scale*h); return nothing
end

#################################################################################
# Accumulate one default-kernel contribution into K for a given preweighted Hankel from h1_multi_ks_at_r! for
# a geometry that does contain symmetry. This is just a wrapper that accepts seperately the hankel value and the prefactor.
# Inputs:
#   K: matrix to accumulate into
#   i,j: target and source indices
#   nxi,nyi: components of target normal at i
#   nxj,nyj: components of source normal at j
#   dx,dy: displacement components from source j to target i
#   invr: 1/r where r is the distance from source j to target i
#   h: preweighted Hankel function value at r.
#   scale: the prefactor computed separately (-im*k/2)
#################################################################################
@inline function _accum_dlp_default_sym!(K::AbstractMatrix{Complex{T}},i::Int,j::Int,
nxi::T,nyi::T,nxj::T,nyj::T,dx::T,dy::T,invr::T,h::Complex{T},scale)::Nothing where {T<:Real}
    _accum_dlp_default_sym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,scale*h); return nothing
end

################################################
############# DERIVATIVE PATHWAY ###############
################################################

#################################################################################
# Thread-local workspace for the derivative Chebyshev pathway.
#
# Purpose:
#   Avoid repeated allocations inside the multi-k derivative assembly loops.
#   Each thread gets:
#     - one 2-vector `pt_tls` for reflected/rotated image coordinates,
#     - one buffer `h0_tls` for H₀^(1)(k_m r),
#     - one buffer `h1_tls` for H₁^(1)(k_m r),
#     - one buffer `h2_tls` for H₂^(1)(k_m r).
#
# The derivative route needs all three Hankel orders because:
#   K    uses H₁,
#   dK   uses H₀,
#   ddK  uses H₁ and H₂.
#
# Inputs to constructor:
#   T   - real scalar type used for geometry coordinates
#   Mk  - number of contour points / plans
#   nth - number of thread-local copies to allocate
#
# Output:
#   DLPDerivChebWorkspace{T}
#################################################################################
struct DLPDerivChebWorkspace{T<:Real}
    pt_tls::Vector{Vector{T}}
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
    h2_tls::Vector{Vector{ComplexF64}}
end

function DLPDerivChebWorkspace(::Type{T},Mk::Int,nth::Int=Threads.nthreads()) where {T<:Real}
    return DLPDerivChebWorkspace{T}([zeros(T,2) for _ in 1:nth],
        [Vector{ComplexF64}(undef,Mk) for _ in 1:nth],
        [Vector{ComplexF64}(undef,Mk) for _ in 1:nth],
        [Vector{ComplexF64}(undef,Mk) for _ in 1:nth])
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPoints containing the boundary points and normals
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multithreading or not
#################################################################################
function _all_k_nosymm_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans);N=length(bp.xy);tol2=(eps(T))^2
    pref=Vector{Complex{T}}(undef,Mk);ab=Vector{NTuple{2,Float64}}(undef,Mk)
    @inbounds for m in 1:Mk
        km=plans[m].k
        pref[m]=Complex{T}(0,0.5)*km
        ab[m]=(Float64(real(km)),Float64(imag(km)))
    end
    nth=Threads.nthreads()
    phases_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    hvals_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();phases=phases_tls[tid];hvals=hvals_tls[tid]
        @inbounds for j in 1:i
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j] # no temporaries
            dx=xi-xj;dy=yi-yj;d2=muladd(dx,dx,dy*dy) # this is the hypotenuse squared
            if d2≤tol2
                val= -Complex{T}(bp.curvature[i]*INV_TWO_PI) # diagonal limit for non-symmetric DLP
                @inbounds begin
                    @inbounds for m in 1:Mk
                        Ks[m][i,j]=val
                        if i!=j;Ks[m][j,i]=val;end
                    end
                end
            else
                r=sqrt(d2);invr=inv(r) # compute r and 1/r for that matrix entry
                p,t,invsqrt=panel_t_invsqrt_h1x(plans[1],Float64(r))
                h1_multi_ks_at_r!(hvals,phases,plans,p,t,invsqrt,Float64(r),ab)
                @inbounds for m in 1:Mk # accumulate into each matrix on the contour the respective contribution
                    h=pref[m]*hvals[m] # preweight hankel with -im*k/2
                    _accum_dlp_default_nosym!(Ks[m],i,j,nxi,nyi,nxj,nyj,dx,dy,invr,h)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct kernel matrices K, dK, ddK for all contour points using the
# derivative Chebyshev pathway, without symmetry.
#
# Inputs:
#   Ks,dKs,ddKs - vectors of matrices to fill, one triple per contour point
#   bp          - BoundaryPoints containing geometry and curvature
#   plans0      - vector of Chebyshev plans for H₀^(1)(k_m r)
#   plans1      - vector of Chebyshev plans for H₁^(1)(k_m r)
#   multithreaded - whether to use threaded loops
#   ws          - optional derivative Chebyshev workspace
#
# Strategy:
#   * Loop once through all geometric distances r = |x_i - x_j|.
#   * Evaluate H₀, H₁, H₂ for all k_m in one pass at each r.
#   * Form
#         K    from H₁,
#         dK   from H₀,
#         ddK  from H₁ and H₂,
#     and accumulate into every matrix triple.
#   * On the diagonal, insert the standard curvature correction into K only.
#
# Output:
#   nothing (fills Ks, dKs, ddKs in place).
#################################################################################
function _all_k_nosymm_DLP_chebyshev_derivatives!(Ks::Vector{Matrix{Complex{T}}},dKs::Vector{Matrix{Complex{T}}},ddKs::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH};multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    Mk=length(plans0);N=length(bp.xy);tol2=(eps(T))^2
    @assert length(plans1)==Mk && length(Ks)==Mk && length(dKs)==Mk && length(ddKs)==Mk
    for m in 1:Mk
        fill!(Ks[m],zero(eltype(Ks[m])));fill!(dKs[m],zero(eltype(dKs[m])));fill!(ddKs[m],zero(eltype(ddKs[m])))
    end
    kvec=Vector{ComplexF64}(undef,Mk)
    pref0=Vector{ComplexF64}(undef,Mk)
    pref1=Vector{ComplexF64}(undef,Mk)
    pref2=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        km=ComplexF64(plans0[m].k)
        kvec[m]=km
        pref0[m]=0.5im*km
        pref1[m]=-0.5im*km
        pref2[m]=0.5im/km
    end
    local_ws=isnothing(ws) ? DLPDerivChebWorkspace(T,Mk) : ws
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();h0vals=local_ws.h0_tls[tid];h1vals=local_ws.h1_tls[tid];h2vals=local_ws.h2_tls[tid]
        @inbounds for j in 1:i
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j]
            dx=xi-xj;dy=yi-yj;d2=muladd(dx,dx,dy*dy)
            if d2<=tol2
                val=-Complex{T}(bp.curvature[i]*INV_TWO_PI)
                for m in 1:Mk
                    Ks[m][i,j]=val
                    if i!=j;Ks[m][j,i]=val;end
                end
            else
                r=sqrt(d2);invr=inv(r);pidx,t=panel_t_h(plans0[1],Float64(r))
                h0_h1_h2_multi_ks_at_r!(h0vals,h1vals,h2vals,plans0,plans1,pidx,t,Float64(r))
                @inbounds for m in 1:Mk
                    kd=kvec[m]*r
                    hK=pref0[m]*h1vals[m]
                    hdK=pref1[m]*r*h0vals[m]
                    hddK=pref2[m]*((-2+kd*kd)*h1vals[m]+kd*h2vals[m])
                    _accum_dlp_triplet_nosym!(Ks[m],dKs[m],ddKs[m],i,j,nxi,nyi,nxj,nyj,dx,dy,invr,hK,hdK,hddK)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the upper triangle and mirrors it to the lower triangle and used for a single matrix.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPoints containing the boundary points and normals
#   plan: ChebHankelPlanH1x containing the k value and chebyshev tables
#   multithreaded: whether to use multithreading or not
#################################################################################
function _one_k_nosymm_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    tol2=(eps(T))^2
    k=plan.k
    pref=Complex{T}(0,0.5)*k
    a,b=Float64(real(k)),Float64(imag(k))
    fill!(K,zero(eltype(K)))
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        @inbounds for j in 1:i
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j];dx=xi-xj;dy=yi-yj;d2=muladd(dx,dx,dy*dy)
            if d2≤tol2
                val= -Complex{T}(bp.curvature[i]*INV_TWO_PI);K[i,j]=val;if i!=j;K[j,i]=val;end
            else
                r=sqrt(d2);invr=inv(r)
                p,t,invsqrt=panel_t_invsqrt_h1x(plan,Float64(r))
                h=pref*h1_at_r(plan,p,t,invsqrt,Float64(r),a,b)
                _accum_dlp_default_nosym!(K,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,h)
            end
        end
    end
    return nothing
end

#################################################################################
# Construct kernel matrices K, dK, ddK for one contour point using the
# derivative Chebyshev pathway, without symmetry.
#
# Inputs:
#   K,dK,ddK   - destination matrices
#   bp         - BoundaryPoints containing geometry and curvature
#   plan0      - Chebyshev plan for H₀^(1)(k r)
#   plan1      - Chebyshev plan for H₁^(1)(k r)
#   multithreaded - whether to use threaded loops
#
# Strategy:
#   * Loop over the upper triangle j=1:i and mirror to (j,i).
#   * Evaluate H₀, H₁, H₂ at each distance r.
#   * Form the kernel and derivative prefactors directly from the Hankel values.
#   * Insert curvature correction on the diagonal of K only.
#
# Output:
#   nothing (fills K, dK, ddK in place).
#################################################################################
function _one_k_nosymm_DLP_chebyshev_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy);tol2=(eps(T))^2;k=ComplexF64(plan0.k)
    fill!(K,zero(eltype(K)));fill!(dK,zero(eltype(dK)));fill!(ddK,zero(eltype(ddK)))
    pref0=0.5im*k;pref1=-0.5im*k;pref2=0.5im/k
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        @inbounds for j in 1:i
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j]
            dx=xi-xj;dy=yi-yj;d2=muladd(dx,dx,dy*dy)
            if d2<=tol2
                val=-Complex{T}(bp.curvature[i]*INV_TWO_PI);K[i,j]=val
                if i!=j;K[j,i]=val;end
            else
                r=sqrt(d2);invr=inv(r);pidx,t=panel_t_h(plan0,Float64(r))
                H0,H1,H2=h0_h1_h2_at_r(plan0,plan1,pidx,t,Float64(r));kd=k*r
                hK=pref0*H1;hdK=pref1*r*H0;hddK=pref2*((-2+kd*kd)*H1+kd*H2)
                _accum_dlp_triplet_nosym!(K,dK,ddK,i,j,nxi,nyi,nxj,nyj,dx,dy,invr,hK,hdK,hddK)
            end
        end
    end
    return nothing
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations and reflection symmetry.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPoints containing the boundary points and normals  
#   sym: Reflection symmetry object
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multhreading or not
#################################################################################
function _all_k_reflection_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Reflection,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_chebyshev!(Ks,bp,plans;multithreaded)
    Mk=length(plans)
    N=length(bp.xy)
    tol2=(eps(T))^2
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    pref=Vector{Complex{T}}(undef,Mk)
    ab=Vector{NTuple{2,Float64}}(undef,Mk)
    @inbounds for m in 1:Mk
        km=plans[m].k
        pref[m]=Complex{T}(0,0.5)*km
        ab[m]=(Float64(real(km)),Float64(imag(km)))
    end
    nth=Threads.nthreads()
    phases_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    hvals_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    pt_tls=[zeros(T,2) for _ in 1:nth]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        phases=phases_tls[tid]
        hvals=hvals_tls[tid]
        pt=pt_tls[tid]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            nxj,nyj=bp.normal[j]
            @inbounds for (op,scale) in ops
                if op==1
                    x_reflect_point!(pt,xj,yj,shift_x)
                    nxr=-nxj
                    nyr=nyj
                elseif op==2
                    y_reflect_point!(pt,xj,yj,shift_y)
                    nxr=nxj
                    nyr=-nyj
                else
                    xy_reflect_point!(pt,xj,yj,shift_x,shift_y)
                    nxr=-nxj
                    nyr=-nyj
                end
                dx=xi-pt[1]
                dy=yi-pt[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2)
                    invr=inv(r)
                    p,t,invsqrt=panel_t_invsqrt_h1x(plans[1],Float64(r))
                    h1_multi_ks_at_r!(hvals,phases,plans,p,t,invsqrt,Float64(r),ab)
                    @inbounds for m in 1:Mk
                        h=scale*pref[m]*hvals[m]
                        _accum_dlp_default_sym!(Ks[m],i,j,nxi,nyi,nxr,nyr,dx,dy,invr,h)
                    end
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct kernel matrices K, dK, ddK for all contour points using the
# derivative Chebyshev pathway with reflection symmetry.
#
# Inputs:
#   Ks,dKs,ddKs - vectors of matrices to fill
#   bp          - BoundaryPoints containing geometry and symmetry shifts
#   sym         - Reflection symmetry descriptor
#   plans0      - vector of Chebyshev plans for H₀^(1)
#   plans1      - vector of Chebyshev plans for H₁^(1)
#   multithreaded - whether to use threaded loops
#   ws          - optional derivative Chebyshev workspace
#
# Strategy:
#   * First assemble the direct no-symmetry contribution.
#   * Then add reflected image contributions for each active reflection branch.
#   * For each reflected source position, evaluate H₀,H₁,H₂ for all k_m and
#     accumulate K,dK,ddK with the appropriate parity scale.
#
# Output:
#   nothing (fills Ks, dKs, ddKs in place).
#################################################################################
function _all_k_reflection_DLP_chebyshev_derivatives!(Ks::Vector{Matrix{Complex{T}}},dKs::Vector{Matrix{Complex{T}}},ddKs::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Reflection,plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH};multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    _all_k_nosymm_DLP_chebyshev_derivatives!(Ks,dKs,ddKs,bp,plans0,plans1;multithreaded=multithreaded,ws=ws)
    Mk=length(plans0);N=length(bp.xy);tol2=(eps(T))^2
    shift_x=bp.shift_x;shift_y=bp.shift_y;ops=_reflect_ops_and_scales(T,sym)
    kvec=Vector{ComplexF64}(undef,Mk);pref0=Vector{ComplexF64}(undef,Mk);pref1=Vector{ComplexF64}(undef,Mk);pref2=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        km=ComplexF64(plans0[m].k)
        kvec[m]=km;pref0[m]=0.5im*km;pref1[m]=-0.5im*km;pref2[m]=0.5im/km
    end
    local_ws=isnothing(ws) ? DLPDerivChebWorkspace(T,Mk) : ws
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();pt=local_ws.pt_tls[tid];h0vals=local_ws.h0_tls[tid];h1vals=local_ws.h1_tls[tid];h2vals=local_ws.h2_tls[tid]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j]
            @inbounds for (op,scale) in ops
                if op==1
                    x_reflect_point!(pt,xj,yj,shift_x)
                    nxr=-nxj
                    nyr=nyj
                elseif op==2
                    y_reflect_point!(pt,xj,yj,shift_y)
                    nxr=nxj
                    nyr=-nyj
                else
                    xy_reflect_point!(pt,xj,yj,shift_x,shift_y)
                    nxr=-nxj
                    nyr=-nyj
                end
                dx=xi-pt[1]
                dy=yi-pt[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r);pidx,t=panel_t_h(plans0[1],Float64(r))
                    h0_h1_h2_multi_ks_at_r!(h0vals,h1vals,h2vals,plans0,plans1,pidx,t,Float64(r))
                    @inbounds for m in 1:Mk
                        kd=kvec[m]*r
                        hK=scale*pref0[m]*h1vals[m]
                        hdK=scale*pref1[m]*r*h0vals[m]
                        hddK=scale*pref2[m]*((-2+kd*kd)*h1vals[m]+kd*h2vals[m])
                        _accum_dlp_triplet_sym!(Ks[m],dKs[m],ddKs[m],i,j,nxr,nyr,dx,dy,invr,hK,hdK,hddK)
                    end
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the whole matrix since custom kernels may not be symmetric.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPoints containing the boundary points and normals  
#   sym: Reflection symmetry object 
#   plan: ChebHankelPlanH1x containing the k value and chebyshev tables
#   multithreaded: whether to use multhreading or not
#################################################################################
function _one_k_reflection_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},sym::Reflection,plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev!(K,bp,plan;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    k=plan.k
    pref=Complex{T}(0,0.5)*k
    a,b=Float64(real(k)),Float64(imag(k))
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    pt=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        pt_local=pt
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            nxj,nyj=bp.normal[j]
            @inbounds for (op,scale_r) in ops
                if op==1
                    x_reflect_point!(pt_local,xj,yj,shift_x)
                    nxr=-nxj
                    nyr=nyj
                elseif op==2
                    y_reflect_point!(pt_local,xj,yj,shift_y)
                    nxr=nxj
                    nyr=-nyj
                else
                    xy_reflect_point!(pt_local,xj,yj,shift_x,shift_y)
                    nxr=-nxj
                    nyr=-nyj
                end
                dx=xi-pt_local[1]
                dy=yi-pt_local[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2)
                    invr=inv(r)
                    p,t,invsqrt=panel_t_invsqrt_h1x(plan,Float64(r))
                    h=Complex{T}(scale_r,zero(T))*pref*h1_at_r(plan,p,t,invsqrt,Float64(r),a,b)
                    _accum_dlp_default_sym!(K,i,j,nxi,nyi,nxr,nyr,dx,dy,invr,h)
                end
            end
        end
    end
    return K
end

#################################################################################
# Construct kernel matrices K, dK, ddK for one contour point using the
# derivative Chebyshev pathway with reflection symmetry.
#
# Inputs:
#   K,dK,ddK   - destination matrices
#   bp         - BoundaryPoints containing geometry and symmetry shifts
#   sym        - Reflection symmetry descriptor
#   plan0      - Chebyshev plan for H₀^(1)(k r)
#   plan1      - Chebyshev plan for H₁^(1)(k r)
#   multithreaded - whether to use threaded loops
#   ws         - optional derivative Chebyshev workspace
#
# Strategy:
#   * First assemble the direct contribution.
#   * Then add reflected image contributions using the transformed image points
#     and reflected normals.
#   * For each reflected pair, evaluate H₀,H₁,H₂ and accumulate K,dK,ddK with
#     the reflection parity factor.
#
# Output:
#   nothing (fills K, dK, ddK in place).
#################################################################################
function _one_k_reflection_DLP_chebyshev_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},sym::Reflection,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH;multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev_derivatives!(K,dK,ddK,bp,plan0,plan1;multithreaded=multithreaded)
    N=length(bp.xy);tol2=(eps(T))^2;k=ComplexF64(plan0.k)
    pref0=0.5im*k;pref1=-0.5im*k;pref2=0.5im/k
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    ops=_reflect_ops_and_scales(T,sym)
    local_ws=isnothing(ws) ? DLPDerivChebWorkspace(T,1) : ws
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        tid=Threads.threadid();pt=local_ws.pt_tls[tid]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            nxj,nyj=bp.normal[j]
            @inbounds for (op,scale) in ops
                if op==1
                    x_reflect_point!(pt,xj,yj,shift_x);nxr=-nxj;nyr=nyj
                elseif op==2
                    y_reflect_point!(pt,xj,yj,shift_y);nxr=nxj;nyr=-nyj
                else
                    xy_reflect_point!(pt,xj,yj,shift_x,shift_y);nxr=-nxj;nyr=-nyj
                end
                dx=xi-pt[1]
                dy=yi-pt[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r);pidx,t=panel_t_h(plan0,Float64(r))
                    H0,H1,H2=h0_h1_h2_at_r(plan0,plan1,pidx,t,Float64(r));kd=k*r
                    hK=scale*pref0*H1;hdK=scale*pref1*r*H0;hddK=scale*pref2*((-2+kd*kd)*H1+kd*H2)
                    _accum_dlp_triplet_sym!(K,dK,ddK,i,j,nxr,nyr,dx,dy,invr,hK,hdK,hddK)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct matrices along complex ks as defined in the plans for which chebyshev interpolations are already precomputed.
# this is optimized for hankel type complex k contour evaluations and rotational symmetry.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPoints containing the boundary points and normals  
#   sym: Rotation symmetry object
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multhreading or not
#################################################################################
function _all_k_rotation_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Rotation,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_DLP_chebyshev!(Ks,bp,plans;multithreaded)
    Mk=length(plans)
    N=length(bp.xy)
    tol2=(eps(T))^2
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pref=Vector{Complex{T}}(undef,Mk)
    ab=Vector{NTuple{2,Float64}}(undef,Mk)
    @inbounds for m in 1:Mk
        km=plans[m].k
        pref[m]=Complex{T}(0,0.5)*km
        ab[m]=(Float64(real(km)),Float64(imag(km)))
    end
    nth=Threads.nthreads()
    phases_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    hvals_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:nth]
    pt_tls=[zeros(T,2) for _ in 1:nth]
    pans_ref=plans[1].panels
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        tid=Threads.threadid()
        phases=phases_tls[tid]
        hvals=hvals_tls[tid]
        pt=pt_tls[tid]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            nxj,nyj=bp.normal[j]
            @inbounds for l in 2:sym.n
                cl=ctab[l]
                sl=stab[l]
                rot_point!(pt,xj,yj,cx,cy,cl,sl)
                nxr=cl*nxj-sl*nyj
                nyr=sl*nxj+cl*nyj
                dx=xi-pt[1]
                dy=yi-pt[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2)
                    invr=inv(r)
                    p,t,invsqrt=panel_t_invsqrt_h1x(plans[1],Float64(r))
                    h1_multi_ks_at_r!(hvals,phases,plans,p,t,invsqrt,Float64(r),ab)
                    χl=χ[l]
                    @inbounds for m in 1:Mk
                        h=χl*pref[m]*hvals[m]
                        _accum_dlp_default_sym!(Ks[m],i,j,nxi,nyi,nxr,nyr,dx,dy,invr,h)
                    end
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct kernel matrices K, dK, ddK for all contour points using the
# derivative Chebyshev pathway with rotational symmetry.
#
# Inputs:
#   Ks,dKs,ddKs - vectors of matrices to fill
#   bp          - BoundaryPoints containing geometry
#   sym         - Rotation symmetry descriptor
#   plans0      - vector of Chebyshev plans for H₀^(1)
#   plans1      - vector of Chebyshev plans for H₁^(1)
#   multithreaded - whether to use threaded loops
#   ws          - optional derivative Chebyshev workspace
#
# Strategy:
#   * First assemble the direct no-symmetry contribution.
#   * Then add all nontrivial rotated image contributions l=1,...,n-1.
#   * For each rotated source point, evaluate H₀,H₁,H₂ for all k_m and
#     accumulate K,dK,ddK with the corresponding rotational character χ_l.
#
# Output:
#   nothing (fills Ks, dKs, ddKs in place).
#################################################################################
function _all_k_rotation_DLP_chebyshev_derivatives!(Ks::Vector{Matrix{Complex{T}}},dKs::Vector{Matrix{Complex{T}}},ddKs::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Rotation,plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH};multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    _all_k_nosymm_DLP_chebyshev_derivatives!(Ks,dKs,ddKs,bp,plans0,plans1;multithreaded=multithreaded,ws=ws)
    Mk=length(plans0);N=length(bp.xy);tol2=(eps(T))^2
    cx,cy=sym.center;ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    kvec=Vector{ComplexF64}(undef,Mk);pref0=Vector{ComplexF64}(undef,Mk);pref1=Vector{ComplexF64}(undef,Mk);pref2=Vector{ComplexF64}(undef,Mk)
    @inbounds for m in 1:Mk
        km=ComplexF64(plans0[m].k)
        kvec[m]=km;pref0[m]=0.5im*km;pref1[m]=-0.5im*km;pref2[m]=0.5im/km
    end
    local_ws=isnothing(ws) ? DLPDerivChebWorkspace(T,Mk) : ws
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i];nxi,nyi=bp.normal[i]
        tid=Threads.threadid();pt=local_ws.pt_tls[tid];h0vals=local_ws.h0_tls[tid];h1vals=local_ws.h1_tls[tid];h2vals=local_ws.h2_tls[tid]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j]
            @inbounds for l in 2:sym.n
                cl=ctab[l];sl=stab[l]
                rot_point!(pt,xj,yj,cx,cy,cl,sl)
                nxr=cl*nxj-sl*nyj;nyr=sl*nxj+cl*nyj
                dx=xi-pt[1]
                dy=yi-pt[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r);pidx,t=panel_t_h(plans0[1],Float64(r))
                    h0_h1_h2_multi_ks_at_r!(h0vals,h1vals,h2vals,plans0,plans1,pidx,t,Float64(r))
                    χl=χ[l]
                    @inbounds for m in 1:Mk
                        kd=kvec[m]*r
                        hK=χl*pref0[m]*h1vals[m]
                        hdK=χl*pref1[m]*r*h0vals[m]
                        hddK=χl*pref2[m]*((-2+kd*kd)*h1vals[m]+kd*h2vals[m])
                        _accum_dlp_triplet_sym!(Ks[m],dKs[m],ddKs[m],i,j,nxr,nyr,dx,dy,invr,hK,hdK,hddK)
                    end
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Construct a single matrix at a complex k as defined in the plan for which chebyshev interpolations are already precomputed. This one only fills the whole matrix since custom kernels may not be symmetric.
# Inputs:
#   K: matrix to fill
#   bp: BoundaryPoints containing the boundary points and normals  
#   sym: Rotation symmetry object 
#   plan: ChebHankelPlanH1x containing the k value and chebyshev tables
#   multithreaded: whether to use multhreading or not
#################################################################################
function _one_k_rotation_DLP_chebyshev!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},sym::Rotation,plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev!(K,bp,plan;multithreaded)
    N=length(bp.xy)
    tol2=(eps(T))^2
    k=plan.k
    pref=Complex{T}(0,0.5)*k
    a,b=Float64(real(k)),Float64(imag(k))
    pans=plan.panels
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    pt=[zero(T),zero(T)]
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        nxi,nyi=bp.normal[i]
        pt_local=pt
        @inbounds for j in 1:N
            xj,yj=bp.xy[j]
            nxj,nyj=bp.normal[j]
            @inbounds for l in 2:sym.n
                cl=ctab[l]
                sl=stab[l]
                rot_point!(pt_local,xj,yj,cx,cy,cl,sl)
                nxr=cl*nxj-sl*nyj
                nyr=sl*nxj+cl*nyj
                dx=xi-pt_local[1]
                dy=yi-pt_local[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2)
                    invr=inv(r)
                    p,t,invsqrt=panel_t_invsqrt_h1x(plan,Float64(r))
                    h=χ[l]*pref*h1_at_r(plan,p,t,invsqrt,Float64(r),a,b)
                    _accum_dlp_default_sym!(K,i,j,nxi,nyi,nxr,nyr,dx,dy,invr,h)
                end
            end
        end
    end
    return K
end

#################################################################################
# Construct kernel matrices K, dK, ddK for one contour point using the
# derivative Chebyshev pathway with rotational symmetry.
#
# Inputs:
#   K,dK,ddK   - destination matrices
#   bp         - BoundaryPoints containing geometry
#   sym        - Rotation symmetry descriptor
#   plan0      - Chebyshev plan for H₀^(1)(k r)
#   plan1      - Chebyshev plan for H₁^(1)(k r)
#   multithreaded - whether to use threaded loops
#   ws         - optional derivative Chebyshev workspace
#
# Strategy:
#   * First assemble the direct contribution.
#   * Then add all nontrivial rotated image contributions.
#   * For each rotated source point, evaluate H₀,H₁,H₂ and accumulate K,dK,ddK
#     using the rotational character factor χ_l.
#
# Output:
#   nothing (fills K, dK, ddK in place).
#################################################################################
function _one_k_rotation_DLP_chebyshev_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},sym::Rotation,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH;multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    _one_k_nosymm_DLP_chebyshev_derivatives!(K,dK,ddK,bp,plan0,plan1;multithreaded=multithreaded)
    N=length(bp.xy);tol2=(eps(T))^2;k=ComplexF64(plan0.k)
    pref0=0.5im*k;pref1=-0.5im*k;pref2=0.5im/k
    cx,cy=sym.center;ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    local_ws=isnothing(ws) ? DLPDerivChebWorkspace(T,1) : ws
    @use_threads multithreading=multithreaded for i in 1:N
        xi,yi=bp.xy[i]
        tid=Threads.threadid();pt=local_ws.pt_tls[tid]
        @inbounds for j in 1:N
            xj,yj=bp.xy[j];nxj,nyj=bp.normal[j]
            @inbounds for l in 2:sym.n
                cl=ctab[l]
                sl=stab[l]
                rot_point!(pt,xj,yj,cx,cy,cl,sl)
                nxr=cl*nxj-sl*nyj
                nyr=sl*nxj+cl*nyj
                dx=xi-pt[1]
                dy=yi-pt[2]
                d2=muladd(dx,dx,dy*dy)
                if d2>tol2
                    r=sqrt(d2);invr=inv(r);pidx,t=panel_t_h(plan0,Float64(r))
                    H0,H1,H2=h0_h1_h2_at_r(plan0,plan1,pidx,t,Float64(r));kd=k*r
                    hK=χ[l]*pref0*H1;hdK=χ[l]*pref1*r*H0;hddK=χ[l]*pref2*((-2+kd*kd)*H1+kd*H2)
                    _accum_dlp_triplet_sym!(K,dK,ddK,i,j,nxr,nyr,dx,dy,invr,hK,hdK,hddK)
                end
            end
        end
    end
    return nothing
end

#################################################################################
# Main interface to compute double-layer potential kernel matrices using chebyshev-hankel plans.
# Inputs:
#   Ks: vector of matrices to fill, one for each complex k in plans. This should be preallocated outside
#   bp: BoundaryPoints containing the boundary points and normals  
#   symmetry: either nothing, or a symmetry object (Reflection or Rotation)
#   plans: vector of ChebHankelPlanH1x containing the k values and chebyshev tables for each k
#   multithreaded: whether to use multhreading or not
#################################################################################
function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},symmetry,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    if symmetry===nothing
        return compute_kernel_matrices_DLP_chebyshev!(Ks,bp,plans;multithreaded)
    else
        try 
            compute_kernel_matrices_DLP_chebyshev!(Ks,bp,symmetry,plans;multithreaded)
        catch _
            error("Error computing kernel matrices with symmetry $(symmetry): ")
            
        end
    end
end

##################################################################################
# Internal dispatchers for different symmetry cases
##################################################################################

function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    return _all_k_nosymm_DLP_chebyshev!(Ks,bp,plans;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    return _one_k_nosymm_DLP_chebyshev!(K,bp,plan;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},sym::Reflection,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    return _all_k_reflection_DLP_chebyshev!(Ks,bp,sym,plans;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},sym::Reflection,plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    return _one_k_reflection_DLP_chebyshev!(K,bp,sym,plan;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},
sym::Rotation,plans::Vector{ChebHankelPlanH1x};multithreaded::Bool=true) where {T<:Real}
    return _all_k_rotation_DLP_chebyshev!(Ks,bp,sym,plans;multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},
sym::Rotation,plan::ChebHankelPlanH1x;multithreaded::Bool=true) where {T<:Real}
    return _one_k_rotation_DLP_chebyshev!(K,bp,sym,plan;multithreaded)
end

#################################################################################
# Assemble Fredholm matrices from kernel matrices by applying quadrature weights and adding identity.
# Inputs:
#   Ks: vector of matrices to modify in place
#   bp: BoundaryPoints containing the boundary points and quadrature weights
#################################################################################
function assemble_fredholm_matrices!(Ks::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T}) where {T<:Real}
    ds=bp.ds
    N=length(ds)
    Mk=length(Ks)
    @inbounds Threads.@threads for m in 1:Mk
        K=Ks[m]
        for j in 1:N
            s=-ds[j]
            @views K[:,j].*=s
        end
        for i in 1:N
            K[i,i]+=one(eltype(K))
        end
        filter_matrix!(K)
    end
    return nothing
end

#################################################################################
# Assemble Fredholm matrix from kernel matrix by applying quadrature weights and adding identity.
# Inputs:
#   Ks: single matrix to modify in place
#   bp: BoundaryPoints containing the boundary points and quadrature weights
#################################################################################
function assemble_fredholm_matrices!(K::Matrix{Complex{T}},bp::BoundaryPoints{T}) where {T<:Real}
    ds=bp.ds
    N=length(ds)
    @inbounds for j in 1:N
        @views K[:,j].*= -ds[j]
    end
    @inbounds for i in 1:N
        K[i,i]+=one(eltype(K))
    end
    filter_matrix!(K)
    return nothing
end

#############################################################
################### DERIVATIVE PATHWAY ######################
#############################################################

function compute_kernel_matrices_DLP_chebyshev_derivatives!(Ks::Vector{Matrix{Complex{T}}},dKs::Vector{Matrix{Complex{T}}},ddKs::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T},symmetry,plans0::Vector{ChebHankelPlanH},plans1::Vector{ChebHankelPlanH};multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    if symmetry===nothing
        return _all_k_nosymm_DLP_chebyshev_derivatives!(Ks,dKs,ddKs,bp,plans0,plans1;multithreaded=multithreaded,ws=ws)
    elseif symmetry isa Reflection
        return _all_k_reflection_DLP_chebyshev_derivatives!(Ks,dKs,ddKs,bp,symmetry,plans0,plans1;multithreaded=multithreaded,ws=ws)
    elseif symmetry isa Rotation
        return _all_k_rotation_DLP_chebyshev_derivatives!(Ks,dKs,ddKs,bp,symmetry,plans0,plans1;multithreaded=multithreaded,ws=ws)
    else
        error("Unsupported symmetry $(typeof(symmetry))")
    end
end

function compute_kernel_matrices_DLP_chebyshev_derivatives!(K::Matrix{Complex{T}},dK::Matrix{Complex{T}},ddK::Matrix{Complex{T}},
    bp::BoundaryPoints{T},plan0::ChebHankelPlanH,plan1::ChebHankelPlanH;multithreaded::Bool=true) where {T<:Real}
    return _one_k_nosymm_DLP_chebyshev_derivatives!(K,dK,ddK,bp,plan0,plan1;multithreaded=multithreaded)
end

function compute_kernel_matrices_DLP_chebyshev_derivatives!(K::Matrix{Complex{T}},dK::Matrix{Complex{T}},ddK::Matrix{Complex{T}},
    bp::BoundaryPoints{T},sym::Reflection,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH;
    multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    return _one_k_reflection_DLP_chebyshev_derivatives!(K,dK,ddK,bp,sym,plan0,plan1;multithreaded=multithreaded,ws=ws)
end

function compute_kernel_matrices_DLP_chebyshev_derivatives!(K::Matrix{Complex{T}},dK::Matrix{Complex{T}},ddK::Matrix{Complex{T}},
    bp::BoundaryPoints{T},sym::Rotation,plan0::ChebHankelPlanH,plan1::ChebHankelPlanH;
    multithreaded::Bool=true,ws::Union{Nothing,DLPDerivChebWorkspace{T}}=nothing) where {T<:Real}
    return _one_k_rotation_DLP_chebyshev_derivatives!(K,dK,ddK,bp,sym,plan0,plan1;multithreaded=multithreaded,ws=ws)
end

#################################################################################
# Assemble Fredholm matrices and their first two derivatives from kernel matrices
# by applying quadrature weights and adding the identity shift to K only.
#
# Inputs:
#   Ks,dKs,ddKs - vectors of kernel matrices to modify in place
#   bp          - BoundaryPoints containing quadrature weights ds
#
# Operation:
#   * multiply each column j by -ds[j] in all three matrix families,
#   * add identity only to K,
#   * filter numerical zeros in K,dK,ddK.
#
# Output:
#   nothing (modifies Ks, dKs, ddKs in place).
#################################################################################
function assemble_fredholm_matrices_with_derivatives!(Ks::Vector{Matrix{Complex{T}}},dKs::Vector{Matrix{Complex{T}}},ddKs::Vector{Matrix{Complex{T}}},bp::BoundaryPoints{T}) where {T<:Real}
    ds=bp.ds;N=length(ds);Mk=length(Ks)
    @inbounds Threads.@threads for m in 1:Mk
        K=Ks[m];dK=dKs[m];ddK=ddKs[m]
        for j in 1:N
            s=-ds[j]
            @views K[:,j].*=s;@views dK[:,j].*=s;@views ddK[:,j].*=s
        end
        for i in 1:N
            K[i,i]+=one(eltype(K))
        end
        filter_matrix!(K);filter_matrix!(dK);filter_matrix!(ddK)
    end
    return nothing
end

#################################################################################
# Assemble one Fredholm matrix and its first two derivatives from kernel matrices
# by applying quadrature weights and adding the identity shift to K only.
#
# Inputs:
#   K,dK,ddK - kernel matrices to modify in place
#   bp       - BoundaryPoints containing quadrature weights ds
#
# Operation:
#   * multiply each column j by -ds[j] in K,dK,ddK,
#   * add identity only to K,
#   * filter numerical zeros in all three matrices.
#
# Output:
#   nothing (modifies K, dK, ddK in place).
#################################################################################
function assemble_fredholm_matrices_with_derivatives!(K::Matrix{Complex{T}},dK::Matrix{Complex{T}},ddK::Matrix{Complex{T}},bp::BoundaryPoints{T}) where {T<:Real}
    ds=bp.ds;N=length(ds)
    @inbounds for j in 1:N
        s=-ds[j]
        @views K[:,j].*=s;@views dK[:,j].*=s;@views ddK[:,j].*=s
    end
    @inbounds for i in 1:N
        K[i,i]+=one(eltype(K))
    end
    filter_matrix!(K);filter_matrix!(dK);filter_matrix!(ddK)
    return nothing
end

#################################################################################

"""
    construct_boundary_matrices!(Tbufs::Vector{Matrix{Complex{T}}},solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},
    zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}

Construct the BIM Fredholm matrices for a vector of complex contour points.

This is the top-level matrix-construction for each contour point `zj[m]` where it
fills `Tbufs[m]` with the corresponding Fredholm matrix

    T(zj[m]) = I - K(zj[m]),

where `K(z)` is the Nyström discretization of the Helmholtz double-layer kernel
at the complex parameter `z`.

When `use_chebyshev=true`, the construction uses piecewise-Chebyshev plans for
the scaled Hankel function

    H1x(z) = exp(-i z) H₁^(1)(z),

which are precomputed independently for each contour point and then evaluated in
a single pass through the geometric distances appearing in the matrix entries.

# Inputs
- `Tbufs`: Preallocated vector of matrices to fill, one for each contour point in `zj`.
- `solver::BoundaryIntegralMethod`: Solver configuration containing symmetry information.
- `pts::BoundaryPoints{T}`: Boundary discretization containing points, normals, curvature, and weights.
- `zj::AbstractVector{Complex{T}}`: Complex contour points at which to assemble the operator family.

# Keyword arguments
- `multithreaded::Bool=true`: Whether to use threaded assembly loops.
- `use_chebyshev::Bool=true`: If `true`, use piecewise-Chebyshev interpolation of the scaled Hankel function for matrix construction. If `false`, use direct special-function evaluation instead.
- `n_panels::Int=15000`: Number of radial panels used in the Chebyshev plans.
- `M::Int=5`: Chebyshev polynomial degree per panel.
- `timeit::Bool=false`: Whether to print timing information for plan construction and assembly.

# Returns
- `nothing`
  The matrices are written in place into `Tbufs`.
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{Complex{T}}},solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},
zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}
    rmin,rmax=estimate_rmin_rmax(pts,solver.symmetry) # estimate geometry extents for hankel plan creation on panels
    plans=Vector{ChebHankelPlanH1x}(undef,length(zj))
    @benchit timeit=timeit "DLP plans" Threads.@threads for i in eachindex(plans) # precompute plans for all contour points. This creates for each zj[i] a piecewise Chebyshev approximation of H1x(z) on [rmin,rmax]
        plans[i]=plan_h1x(zj[i],rmin,rmax,npanels=n_panels,M=M,grading=:uniform)
    end
    if use_chebyshev # use the chebyshev hankel evaluations for matrix construction. This is faster for large k values where standard hankel evaluations are slow and allocate a lot.
        @blas_1 begin
            @benchit timeit=timeit "DLP Chebyshev" compute_kernel_matrices_DLP_chebyshev!(Tbufs,pts,solver.symmetry,plans;multithreaded=multithreaded)   
            @benchit timeit=timeit "Assemble matrices" assemble_fredholm_matrices!(Tbufs,pts)
        end
    else
        @blas_1 begin # use standard hankel evaluations for matrix construction. This is faster for small k values where chebyshev interpolation overhead is not worth it.
            @benchit timeit=timeit "DLP complex k" @inbounds for j in eachindex(zj)
                fredholm_matrix_complex_k!(Tbufs[j],pts,solver.symmetry,zj[j],multithreaded=multithreaded) 
            end
        end
    end
    return nothing
end

"""

    construct_boundary_matrices_with_derivatives!(Tbufs::Vector{Matrix{Complex{T}}},dTbufs::Vector{Matrix{Complex{T}}},ddTbufs::Vector{Matrix{Complex{T}}},solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}}; multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}

Construct the BIM Fredholm matrices and their first two derivatives for a vector of complex contour points.
It fills, for each contour point `zj[m]`:
- `Tbufs[m]`   with the Fredholm matrix `T(zj[m])`,
- `dTbufs[m]`  with the first derivative `T'(zj[m])`,
- `ddTbufs[m]` with the second derivative `T''(zj[m])`.

When `use_chebyshev=true`, the construction uses piecewise-Chebyshev plans for
the unscaled Hankel functions `H₀^(1)` and `H₁^(1)` on the radial interval
estimated from the geometry. The second-order Hankel term `H₂^(1)` is then
recovered away from the small-argument regime by the standard recurrence

    H₂^(1)(z) = (2/z) H₁^(1)(z) - H₀^(1)(z)

Note: The direct route for complex ks is not yet supported!

# Inputs
- `Tbufs`: Preallocated vector of matrices to hold the Fredholm matrices.
- `dTbufs`: Preallocated vector of matrices to hold the first derivatives.
- `ddTbufs`: Preallocated vector of matrices to hold the second derivatives.
- `solver::BoundaryIntegralMethod`: Solver configuration containing symmetry information.
- `pts::BoundaryPoints{T}`: Boundary discretization containing points, normals, curvature, and weights.
- `zj::AbstractVector{Complex{T}}`: Complex k points at which to assemble the operator family.

# Keyword arguments
- `multithreaded::Bool=true`: Whether to use threaded assembly loops.
- `use_chebyshev::Bool=true`: If `true`, use piecewise-Chebyshev Hankel interpolation for matrix construction. If `false`, use the direct special-function route instead.
- `n_panels::Int=15000`: Number of radial panels used in the Chebyshev plans.
- `M::Int=5`: Chebyshev polynomial degree per panel.
- `timeit::Bool=false`: Whether to print timing information for plan construction and assembly.

# Returns
- `nothing`
  The matrices are written in place into:
  - `Tbufs`,
  - `dTbufs`,
  - `ddTbufs`.
"""
function construct_boundary_matrices_with_derivatives!(Tbufs::Vector{Matrix{Complex{T}}},dTbufs::Vector{Matrix{Complex{T}}},ddTbufs::Vector{Matrix{Complex{T}}},solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels::Int=15000,M::Int=5,timeit::Bool=false) where {T<:Real}
    if use_chebyshev
        rmin,rmax=estimate_rmin_rmax(pts,solver.symmetry)
        plans0=Vector{ChebHankelPlanH}(undef,length(zj))
        plans1=Vector{ChebHankelPlanH}(undef,length(zj))
        @benchit timeit=timeit "DLP deriv plans" Threads.@threads for i in eachindex(zj)
            plans0[i]=plan_h(0,1,ComplexF64(zj[i]),rmin,rmax;npanels=n_panels,M=M,grading=:uniform)
            plans1[i]=plan_h(1,1,ComplexF64(zj[i]),rmin,rmax;npanels=n_panels,M=M,grading=:uniform)
        end
        ws=DLPDerivChebWorkspace(T,length(zj))
        @blas_1 begin
            @benchit timeit=timeit "DLP deriv Chebyshev" compute_kernel_matrices_DLP_chebyshev_derivatives!(Tbufs,dTbufs,ddTbufs,pts,solver.symmetry,plans0,plans1;multithreaded=multithreaded,ws=ws)
            @benchit timeit=timeit "Assemble deriv matrices" assemble_fredholm_matrices_with_derivatives!(Tbufs,dTbufs,ddTbufs,pts)
        end
    else
        @error("Not currently implemented")
    end
    return nothing
end