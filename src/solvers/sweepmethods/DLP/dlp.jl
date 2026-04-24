const TWO_PI=2*pi
const FOUR_PI=4*pi

#TODO Baoling Xie and Jun Lai, A Singularity Guided Nyström Method for Elastostatics on Two Dimensional Domains with Corners, arXiv:2512.18208, 2025

"""
    BoundaryIntegralMethod{T,Sym} <: SweepSolver

Configuration object for the standard boundary integral method (BIM) Fredholm
formulation based on the direct Helmholtz double-layer kernel.

This solver is the “plain” boundary-integral implementation in the library:
it does not use Kress logarithmic splitting, Alpert correction, or any special
corner quadrature. Instead, it assembles the Fredholm second-kind matrix
directly from the default 2D Helmholtz double-layer kernel, using the sampled
boundary points, their normals, curvatures, and arc-length weights.

Mathematically, the assembled operator is of the form

    A(k) = I - K(k),

where K(k) is the Nyström discretization of the boundary double-layer operator
for the interior Helmholtz Dirichlet problem. In this implementation, symmetry
images can be incorporated directly into the kernel before the Fredholm shift
by the identity.

# Fields
- `dim_scaling_factor::T`:
  Compatibility field for the generic solver infrastructure. The plain BIM is a
  boundary-only method and has no separate interior basis dimension, but the
  field is kept so that refinement and sweep code can treat all solvers through
  a common interface.
- `pts_scaling_factor::Vector{T}`:
  Boundary-resolution scaling factors. For each boundary component, the number
  of quadrature nodes is chosen roughly as

      N ≈ k * L * b / (2π),

  where `L` is the component length and `b` is the corresponding scaling factor.
- `sampler::Vector`:
  Sampling rules used on each boundary component. These determine how the
  parameter values are chosen before geometric quantities such as points,
  normals, curvature, and arc-length weights are computed.
- `eps::T`:
  Numerical tolerance placeholder.
- `min_dim::Int64`:
  Compatibility field mirroring the other solvers.
- `min_pts::Int64`:
  Minimum number of boundary points per component.
- `symmetry::Sym`:
  Optional symmetry descriptor. If provided, the boundary points may be taken
  from a desymmetrized boundary, and kernel assembly may add reflected or
  rotated image contributions with the appropriate symmetry factors.

# Limitations
Because this is the direct method, near-singular and singular behavior is
handled only through the built-in diagonal curvature correction and the raw
Nyström quadrature. For high precision on difficult geometries, especially
cornered ones, the Kress- or Alpert-corrected variants are usually preferable.
"""
struct BoundaryIntegralMethod{T<:Real,Sym}<:SweepSolver 
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 
    min_pts::Int64
    symmetry::Sym
end

"""
    struct AbstractHankelBasis <: AbsBasis

Compatibility placeholder.
"""
struct AbstractHankelBasis <: AbsBasis end

"""
    resize_basis(basis::Ba, billiard::Bi, dim::Int, k::Real) -> AbstractHankelBasis

Compatibility placeholder.
"""
function resize_basis(basis::Ba,billiard::Bi,dim::Int,k) where {Ba<:AbstractHankelBasis, Bi<:AbsBilliard}
    return AbstractHankelBasis()
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real, Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return BoundaryIntegralMethod{T,Sym}(1.0,bs,sampler,eps(T),min_pts,min_pts,symmetry)
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},samplers::Vector,billiard::Bi;min_pts=20,symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real, Bi<:AbsBilliard} 
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    Sym=typeof(symmetry)
    return BoundaryIntegralMethod{T,Sym}(1.0,bs,samplers,eps(T),min_pts,min_pts,symmetry)
end

function _boundary_curves_for_solver(billiard::Bi,solver::BoundaryIntegralMethod) where {Bi<:AbsBilliard}
    if isnothing(solver.symmetry)
        hasproperty(billiard,:full_boundary) && return getfield(billiard,:full_boundary)
    end
    hasproperty(billiard,:desymmetrized_full_boundary) && return getfield(billiard,:desymmetrized_full_boundary)
    hasproperty(billiard,:full_boundary) && return getfield(billiard,:full_boundary)
    error("No usable boundary field found in $(typeof(billiard))")
end

"""
    evaluate_points(solver::BoundaryIntegralMethod, billiard, k)

Construct the boundary discretization used by the plain boundary integral
method at wavenumber `k`.
Unlike the Kress-based methods, this function does not build a special
logarithmic-correction discretization. It simply samples the boundary according
to the chosen sampler and converts those samples into a `BoundaryPoints`
object.

# Discretization strategy
For each real boundary curve component:
1. determine its length `L`,
2. choose the number of points approximately as

       N ≈ k * L * b / (2π),

   where `b` is the component-specific entry of `pts_scaling_factor`,
3. enforce `N ≥ min_pts`,
4. sample the parameter nodes using the component’s sampler,
5. evaluate:
   - curve points,
   - outward normals,
   - curvature,
   - cumulative arclength and local increments.

The per-point quadrature weights stored in `bp.ds` are the local arc-length
increments along the sampled boundary and are later used to turn the kernel
matrix into the Fredholm matrix.

# Arguments
- `solver::BoundaryIntegralMethod`:
  Solver configuration containing sampling and scaling information.
- `billiard::AbsBilliard`:
  Geometry to discretize.
- `k`:
  Wavenumber controlling the sampling density.

# Returns
- `BoundaryPoints{T}`

  containing:
  - `xy`: sampled boundary coordinates,
  - `normal`: outward unit normals,
  - `curvature`: curvature values,
  - `ds`: arc-length weights,
  - `shift_x`, `shift_y`: axis offsets used by symmetry-image formulas when
    relevant.

# Notes
- If the billiard provides axis-shift information via `x_axis` or `y_axis`,
  those values are stored in the returned `BoundaryPoints`.
"""
function evaluate_points(solver::BoundaryIntegralMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=_boundary_curves_for_solver(billiard,solver)
    type=eltype(solver.pts_scaling_factor)
    xy_all=Vector{SVector{2,type}}()
    normal_all=Vector{SVector{2,type}}()
    s_all=Vector{type}()
    kappa_all=Vector{type}()
    w_all=Vector{type}()
    soff=zero(type)
    for i in eachindex(curves)
        crv=curves[i]
        if typeof(crv)<:AbsRealCurve
            L=crv.length
            N=max(solver.min_pts,round(Int,k*L*bs[i]/(2*pi)))
            sampler=samplers[i]
            t,dt=sampler isa PolarSampler ? sample_points(sampler,crv,N) : sample_points(sampler,N)
            s=arc_length(crv,t)
            ds=diff(s)
            append!(ds,L+s[1]-s[end])
            append!(s_all,s.-s[1].+soff)
            soff+=L
            append!(xy_all,curve(crv,t))
            append!(normal_all,normal_vec(crv,t))
            append!(kappa_all,curvature(crv,t))
            append!(w_all,ds)
        end
    end
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : type(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : type(0.0)
    return BoundaryPoints{type}(xy_all,normal_all,s_all,w_all,Vector{type}(),Vector{type}(),kappa_all,Vector{SVector{2,type}}(),shift_x,shift_y)
end

"""
    default_helmholtz_kernel_matrix(bp, k; multithreaded=true)

Assemble the raw 2D Helmholtz double-layer kernel matrix on the sampled boundary,
without symmetry images and before multiplication by arc-length weights or
addition of the identity.

Mathematical meaning
--------------------
For the interior Dirichlet Helmholtz problem, the double-layer kernel is

    K(x,y;k) = (i k / 2) * cosφ * H₁^(1)(k r),

in the normalization used by this code, where:
- `r = |x - y|`,
- `cosφ = n_y · (x - y) / r`,
- `n_y` is the outward normal at the source point y,
- `H₁^(1)` is the Hankel function of the first kind of order 1.

Thus, for i ≠ j, the matrix entry represents the source-normal derivative of the
free-space Green function evaluated between boundary points i and j.

Diagonal treatment
------------------
On the diagonal the weak singularity is replaced by the limit

    -κ / (2π),

where `κ` is the boundary curvature at the target/source point. This is the
standard diagonal correction for the direct DLP discretization.

# Arguments
- `bp::BoundaryPoints{T}`:
  Boundary discretization containing points, normals, and curvature.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to thread the pairwise assembly loops.

# Returns
- `Matrix{Complex{T}}`

# Important note
This function assembles only the geometric/kernel part. It does not yet:
- multiply by arc-length weights `ds`,
- apply the Fredholm sign,
- add the identity.
- those steps happen later in `fredholm_matrix!`.
"""
function default_helmholtz_kernel_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    curvatures=bp.curvature
    N=length(xy)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    tol=eps(T)
    pref=Complex{T}(zero(T),k/2) # im*k/2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i # symmetric hankel part
            dx=xi-xs[j];dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy)) # an efficient dy^2+dx*dx
            if d<tol
                M[i,j]= -Complex(curvatures[i]/TWO_PI) 
            else
                invd=inv(d)
                cos_phi=(nx[j]*dx+ny[j]*dy)*invd
                hankel=pref*Bessels.hankelh1(1,k*d)
                M[i,j]=cos_phi*hankel
            end
            if i!=j
                cos_phi_symmetric=(nx[i]*(-dx)+ny[i]*(-dy))*invd # Hankel is symmetric, but cos_phi is not; compute explicitly for M[j, i]
                M[j,i]=cos_phi_symmetric*hankel
            end
        end
    end
    filter_matrix!(M)
    return M
end

"""
    default_helmholtz_kernel_derivative_matrix(bp, k; multithreaded=true)

Assemble the first derivative with respect to k of the raw Helmholtz
double-layer kernel matrix.


    K(x,y;k) = (i k / 2) * cosφ * H₁^(1)(k r),

this function assembles

    ∂K/∂k.

Using the Bessel/Hankel recurrence identities, the derivative simplifies in the
chosen normalization to an expression proportional to

    -(i k / 2) * r * H₀^(1)(k r),

times the same geometric cosine factor.

# Arguments
- `bp::BoundaryPoints{T}`:
  Boundary nodes, normals, and related geometry.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to use threaded assembly.

# Returns
- `Matrix{Complex{T}}`
"""
function default_helmholtz_kernel_derivative_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(xy)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    pref=Complex{T}(zero(T),-k/2)
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:(i-1)
            dx=xi-xs[j];dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            invd=inv(d)
            cos_phi=(nx[j]*dx+ny[j]*dy)*invd
            hankel=pref*d*Bessels.hankelh1(0,k*d)
            M[i,j]=cos_phi*hankel
            cos_phi_symmetric=(nxi*(-dx)+nyi*(-dy))*invd
            M[j,i]=cos_phi_symmetric*hankel
        end
    end
    M[diagind(M)].=Complex(zero(T),zero(T))
    filter_matrix!(M)
    return M
end

"""
    default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPoints{T}, k::T; multithreaded::Bool=true) -> Matrix{Complex{T}}

Assemble the second derivative with respect to k of the raw Helmholtz
double-layer kernel matrix.

This function computes

    ∂²K/∂k²

for the default double-layer Helmholtz kernel. In the normalization of the code,
the off-diagonal expression is a cosine-factor times a combination of
Hankel functions of orders 1 and 2, namely the result of differentiating

    (i k / 2) * H₁^(1)(k r)

twice with respect to k.

# Arguments
- `bp::BoundaryPoints{T}`:
  Boundary discretization containing points and normals.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to thread the assembly loop.

# Returns
- `Matrix{Complex{T}}`

# Notes
As with the first derivative, this is a raw kernel-derivative matrix, not yet
the derivative of the Fredholm operator. The latter is built later by
`fredholm_matrix_with_derivatives!`.
"""
@inline function default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(xy)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    pref=Complex{T}(zero(T),inv(2*k)) # im/(2*k)
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i]; yi=ys[i]
        nxi=nx[i]; nyi=ny[i]
        @inbounds for j in 1:(i-1)
            dx=xi-xs[j]
            dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            invd=inv(d)
            cos_phi=(nx[j]*dx+ny[j]*dy)*invd
            hankel=pref*((-2+(k*d)^2)*Bessels.hankelh1(1,k*d)+k*d*Bessels.hankelh1(2,k*d))
            M[i,j]=cos_phi*hankel
            cos_phi_symmetric=(nxi*(-dx)+nyi*(-dy))*invd
            M[j,i]=cos_phi_symmetric*hankel
        end
    end
    M[diagind(M)].=Complex(zero(T),zero(T))
    filter_matrix!(M)
    return M
end

"""
    compute_kernel_matrix(bp, k; multithreaded=true)

Convenience wrapper returning the raw default Helmholtz double-layer kernel
matrix without symmetry. This is the allocation-returning counterpart of `compute_kernel_matrix!`. It
simply calls the default direct-kernel assembly and returns the resulting dense
matrix.

# Arguments
- `bp::BoundaryPoints{T}`:
  Boundary discretization.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to thread assembly.

# Returns
- `Matrix{Complex{T}}`
"""
function compute_kernel_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return default_helmholtz_kernel_matrix(bp,k;multithreaded=multithreaded)
end


"""
    _add_pair_default!(M, i, j, xi, yi, nxi, nyi, xj, yj, nxj, nyj, k, tol2, pref; scale=1)

Internal low-level helper adding the default off-diagonal double-layer kernel
contribution for one ordered pair `(i,j)`.

This helper exists to avoid repeated temporary allocations and repeated
high-level dispatch inside tight kernel-assembly loops. It computes the raw
double-layer contribution from source point j to target point i and adds it
directly into `M[i,j]`.

Mathematical meaning
--------------------
For a non-singular pair, it adds

    scale * ((n_j · (x_i - x_j)) / r) * pref * H₁^(1)(k r),

where:
- `r = |x_i - x_j|`,
- `n_j` is the source normal,
- `pref` is typically `(i k / 2)` in the current normalization,
- `scale` is an optional symmetry factor.

No diagonal curvature correction is applied here. If the pair is too close,
the function returns `false` and leaves the singular/diagonal handling to the
caller.

# Arguments
- `M::AbstractMatrix{Complex{T}}`:
  Destination matrix.
- `i, j::Int`:
  Target and source indices.
- `xi, yi, nxi, nyi`:
  Target coordinates and target normal components.
- `xj, yj, nxj, nyj`:
  Source coordinates and source normal components.
- `k::T`:
  Real wavenumber.
- `tol2::T`:
  Squared distance threshold below which the pair is considered singular or too
  close to evaluate with the regular formula.
- `pref::Complex{T}`:
  Scalar prefactor already containing the chosen kernel normalization.
- `scale`:
  Optional real or complex multiplicative factor, used for symmetry images.

# Returns
- `Bool`:
  `true` if a regular off-diagonal contribution was added,
  `false` if `r² <= tol2`.
"""
@inline function _add_pair_default!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::T,tol2::T,pref::Complex{T};scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real}
    dx=xi-xj;dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    if d2<=tol2
        return false
    end
    d=sqrt(d2)
    invd=inv(d)
    h=pref*Bessels.hankelh1(1,k*d)
    @inbounds begin
       M[i,j]+=scale*((nxj*dx+nyj*dy)*invd)*h
    end
    return true
end

"""
    _add_pair3_no_symmetry_default!(K, dK, ddK, i, j, xi, yi, nxi, nyi, xj, yj, nxj, nyj, κi, k, tol2; scale=1)

Internal low-level helper adding the default kernel and its first two
k-derivatives for one pair `(i,j)` without symmetry images.

This is the core pairwise building block used by
`compute_kernel_matrix_with_derivatives!`. It computes, in one shot:
- the raw kernel contribution,
- the first derivative with respect to k,
- the second derivative with respect to k.

# Arguments
- `K, dK, ddK`:
  Destination matrices for the kernel and its first two derivatives.
- `i, j::Int`:
  Pair indices.
- `xi, yi, nxi, nyi`:
  Target coordinates and target normal.
- `xj, yj, nxj, nyj`:
  Source coordinates and source normal.
- `κi::T`:
  Curvature at target point i, used only for the diagonal limit.
- `k::T`:
  Real wavenumber.
- `tol2::T`:
  Squared distance threshold.
- `scale`:
  Optional real or complex multiplier.

# Returns
- `Bool`:
  `false` for the diagonal/self case,
  `true` for a regular off-diagonal pair.
"""
@inline function _add_pair3_no_symmetry_default!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real,C<:Complex}
    dx=xi-xj;dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    if i==j # when we have no symmetry safely modify the Diagonal elements, otherwise the d^2<tol2 check in the symmetry version 
        @inbounds K[i,j]+= -scale*Complex(κi/TWO_PI)
        return false
    end
    d=sqrt(d2);invd=inv(d);kd=k*d
    c_ij=(nxj*dx+nyj*dy)*invd # cosϕ[i,j]
    c_ji=(nxi*(-dx)+nyi*(-dy))*invd # cosϕ[j,i]
    H0,H1,H2=Bessels.besselh(0:2,1,kd) # allocates a 3-vector, but this is the biggest efficiency due to reccurence
    pref=Complex{T}(zero(T),k/2)  # base: (im*k/2) * H1(kd)
    pref2=Complex{T}(zero(T),inv(2*k)) # second derivative prefix
    hK=pref*H1 # base val (before cosϕ) # (im*k/2)*H1(kd)
    hdK=pref*d*H0  # first derivative val (im*k/2)*d*H0(kd)
    hddK=pref2*((-2+kd*kd)*H1+kd*H2) # second derivative:  im/(2k) * [ (-2 + (kd)^2) H1(kd) + kd H2(kd) ]
    @inbounds begin 
        K[i,j]+=scale*(c_ij*hK)
        dK[i,j]+=scale*(c_ij*hdK)
        ddK[i,j]+=scale*(c_ij*hddK)
        K[j,i]+=scale*(c_ji*hK) # i!=j since otherwise it would terminate in the i==j check
        dK[j,i]+=scale*(c_ji*hdK)
        ddK[j,i]+=scale*(c_ji*hddK)
    end
    return true
end

"""
    _add_pair3_image_default!(K, dK, ddK, i, j, xi, yi, nxi, nyi, xjr, yjr, nxjr, nyjr, κi, k, tol2; scale=1)

Internal low-level helper adding the default kernel and derivative contribution
from a symmetry image of the source point. This function is the symmetry-image analogue of
`_add_pair3_no_symmetry_default!`. It is used when reflections or rotations are
active and the source point j must contribute not only directly, but also via
its transformed images.

Unlike the no-symmetry version:
- no curvature term is ever added here,
- if the image happens to coincide with the target up to tolerance, the
  contribution is simply skipped.

The function evaluates the same default kernel and its derivatives as in the
base case, but using the transformed image coordinates `(xjr, yjr)` and the
image normal `(nxjr, nyjr)`, multiplied by a symmetry scale factor.

Typical scale factors are:
- `±1` for reflection parity,
- `exp(i 2π m l / n)` for rotational characters.

# Arguments
- `K, dK, ddK`:
  Destination matrices.
- `i, j::Int`:
  Target and original source indices.
- `xi, yi, nxi, nyi`:
  Target data.
- `xjr, yjr, nxjr, nyjr`:
  Image source coordinates and image normal.
- `κi::T`:
  Included for interface consistency; not used here for image contributions.
- `k::T`:
  Real wavenumber.
- `tol2::T`:
  Squared coincidence threshold.
- `scale`:
  Symmetry multiplier.

# Returns
- `Bool`:
  `true` if the image contribution was added,
  `false` if it was skipped because the image-target distance was too small.
"""
@inline function _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κi,k,tol2;scale=one(eltype(K))) 
    dx=xi-xjr
    dy=yi-yjr
    d2=muladd(dx,dx,dy*dy)
    d2<=tol2 && return false
    d=sqrt(d2)
    invd=inv(d)
    kd=k*d
    cij=(nxjr*dx+nyjr*dy)*invd
    H0,H1,H2=Bessels.besselh(0:2,1,kd)
    pref=Complex(zero(k),k/2)
    pref2=Complex(zero(k),inv(2*k))
    hK=pref*H1
    hdK=pref*d*H0
    hddK=pref2*((-2+kd*kd)*H1+kd*H2)
    K[i,j]+=scale*(cij*hK);dK[i,j]+=scale*(cij*hdK);ddK[i,j]+=scale*(cij*hddK)
    return true
end

function compute_kernel_matrix!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(K,Complex{T}(zero(T),zero(T)))
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature
    N=length(xy)
    tol2=(eps(T))^2
    pref=Complex{T}(0,k/2)
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xy[i][1]
        yi=xy[i][2]
        nxi=nrm[i][1]
        nyi=nrm[i][2]
        @inbounds for j in 1:N
            xj=xy[j][1]
            yj=xy[j][2]
            nxj=nrm[j][1]
            nyj=nrm[j][2]
            ok=_add_pair_default!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref)
            if !ok
                K[i,j]+= -Complex(κ[i]/TWO_PI)
            end
        end
    end
    return K
end

function compute_kernel_matrix!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(K,Complex{T}(zero(T),zero(T)))
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature
    N=length(xy)
    tol2=(eps(T))^2
    pref=Complex{T}(0,k/2)
    add_x=false
    add_y=false
    add_xy=false
    sxgn=one(T)
    sygn=one(T)
    sxy=one(T)
    shift_x=bp.shift_x
    shift_y=bp.shift_y
    have_rot=false
    nrot=1
    mrot=0
    cx=zero(T)
    cy=zero(T)
    s=symmetry
    if hasproperty(s,:axis)
        if s.axis==:y_axis
            add_x=true
            sxgn=(s.parity==-1 ? -one(T) : one(T))
        end
        if s.axis==:x_axis
            add_y=true
            sygn=(s.parity==-1 ? -one(T) : one(T))
        end
        if s.axis==:origin
            add_x=true
            add_y=true
            add_xy=true
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
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xy[i][1]
        yi=xy[i][2]
        nxi=nrm[i][1]
        nyi=nrm[i][2]
        @inbounds for j in 1:N
            xj=xy[j][1]
            yj=xy[j][2]
            nxj=nrm[j][1]
            nyj=nrm[j][2]
            ok=_add_pair_default!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref)
            if !ok
                K[i,j]+= -Complex(κ[i]/TWO_PI)
            end
            if add_x
                xr=_x_reflect(xj,shift_x)
                yr=yj
                nxr=-nxj
                nyr=nyj
                _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=sxgn)
            end
            if add_y
                xr=xj
                yr=_y_reflect(yj,shift_y)
                nxr=nxj
                nyr=-nyj
                _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=sygn)
            end
            if add_xy
                xr=_x_reflect(xj,shift_x)
                yr=_y_reflect(yj,shift_y)
                nxr=-nxj
                nyr=-nyj
                _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=sxy)
            end
            if have_rot
                @inbounds for l in 1:nrot-1
                    cl=ctab[l+1]
                    sl=stab[l+1]
                    xr,yr=_rot_point(xj,yj,cx,cy,cl,sl)
                    nxr=cl*nxj-sl*nyj
                    nyr=sl*nxj+cl*nyj
                    phase=χ[l+1]
                    _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxr,nyr,k,tol2,pref;scale=phase)
                end
            end
        end
    end
    return K
end


"""
    compute_kernel_matrix_with_derivatives!(K, dK, ddK, bp, k; multithreaded=true)

Assemble in-place the raw default kernel matrix and its first two derivatives
with respect to k, without symmetry.
It fills:
- `K`   with the raw double-layer kernel,
- `dK`  with ∂K/∂k,
- `ddK` with ∂²K/∂k².

# Arguments
- `K, dK, ddK::AbstractMatrix{Complex{T}}`:
  Destination matrices.
- `bp::BoundaryPoints{T}`:
  Boundary discretization.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to use threaded assembly.

# Returns
- `(K, dK, ddK)`, each modified in place.

# Notes
This function assembles raw kernel quantities only. To obtain the Fredholm matrix
and its derivatives, use `fredholm_matrix_with_derivatives!`.
"""
function compute_kernel_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    fill!(K,Complex{T}(zero(T),zero(T)))
    fill!(dK,Complex{T}(zero(T),zero(T)))
    fill!(ddK,Complex{T}(zero(T),zero(T)))
    xs=getindex.(bp.xy,1);ys=getindex.(bp.xy,2)
    nx=getindex.(bp.normal,1);ny=getindex.(bp.normal,2)
    κ=bp.curvature
    tol2=(eps(T))^2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i
            xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
            _add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2)
        end
    end
    return K,dK,ddK
end

function compute_kernel_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    fill!(K,Complex{T}(zero(T),zero(T)))
    fill!(dK,Complex{T}(zero(T),zero(T)))
    fill!(ddK,Complex{T}(zero(T),zero(T)))
    xs=getindex.(bp.xy,1);ys=getindex.(bp.xy,2)
    nx=getindex.(bp.normal,1);ny=getindex.(bp.normal,2)
    κ=bp.curvature
    tol2=(eps(T))^2
    shift_x=bp.shift_x;shift_y=bp.shift_y
    add_x=false;add_y=false;add_xy=false
    sxgn=one(T);sygn=one(T);sxy=one(T)
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
    @use_threads multithreading=multithreaded for i in 1:N
            xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i] # i is the target, j is the source
            @inbounds for j in 1:N
                xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
                # base (upper triangle only; mirrors into [j,i]; curvature on diag)
                if j<=i
                    _add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2)
                end
                if add_x
                    xjr=_x_reflect(xj,shift_x);yjr=yj
                    nxjr,nyjr=_x_reflect_normal(nxj,nyj)
                    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2;scale=sxgn)
                end
                if add_y
                    xjr=xj;yjr=_y_reflect(yj,shift_y)
                    nxjr,nyjr=_y_reflect_normal(nxj,nyj)
                    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2;scale=sygn)
                end
                if add_xy
                    xjr=_x_reflect(xj,shift_x);yjr=_y_reflect(yj,shift_y)
                    nxjr,nyjr=_xy_reflect_normal(nxj,nyj)
                    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2;scale=sxy)
                end
                if have_rot
                    @inbounds for l in 1:nrot-1
                        cl=ctab[l+1];sl=stab[l+1]
                        xjr,yjr=_rot_point(xj,yj,cx,cy,cl,sl)
                        nxjr,nyjr=_rot_vec(nxj,nyj,cl,sl)
                        phase=χ[l+1]
                        _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2;scale=phase)
                    end
                end
            end
        end
    return K,dK,ddK
end

"""
    fredholm_matrix!(K, bp, symmetry, k; multithreaded=true)

Assemble in-place the Fredholm second-kind matrix used by the plain boundary
integral method. Thus the returned matrix is the actual Fredholm matrix used in solves.

# Arguments
- `K::AbstractMatrix{Complex{T}}`:
  Preallocated destination matrix.
- `bp::BoundaryPoints{T}`:
  Boundary discretization including `ds`.
- `symmetry`:
  Optional symmetry descriptor; may be `nothing`.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to thread the underlying kernel assembly.

# Returns
- `K`, modified in place to contain the Fredholm matrix.
"""
function fredholm_matrix!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    isnothing(symmetry) ? compute_kernel_matrix!(K,bp,k;multithreaded=multithreaded) : compute_kernel_matrix!(K,bp,symmetry,k;multithreaded=multithreaded)
    ds=bp.ds
    @inbounds for j in 1:length(ds)
        @views K[:,j].*=ds[j]
    end
    K.*=-one(T)
    @inbounds for i in axes(K,1)
        K[i,i]+=one(T)
    end
    return K
end

"""
    fredholm_matrix_with_derivatives!(K, dK, ddK, bp, symmetry, k; multithreaded=true)

Assemble in-place the Fredholm matrix and its first two derivatives with respect
to k. The returned matrices are the actual Fredholm matrix and its derivatives used in solves.

# Arguments
- `K, dK, ddK::AbstractMatrix{Complex{T}}`:
  Destination matrices for the Fredholm matrix and its first two derivatives.
- `bp::BoundaryPoints{T}`:
  Boundary discretization.
- `symmetry`:
  Optional symmetry descriptor, possibly `nothing`.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to thread the kernel assembly.

# Returns
- `(K, dK, ddK)`, modified in place.
"""
function fredholm_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(symmetry)
        K,dK,ddK=compute_kernel_matrix_with_derivatives!(K,dK,ddK,bp,k;multithreaded=multithreaded)
    else
        K,dK,ddK=compute_kernel_matrix_with_derivatives!(K,dK,ddK,bp,symmetry,k;multithreaded=multithreaded)
    end
    ds=bp.ds
    @inbounds for j in 1:length(ds)
        @views K[:,j].*=ds[j]
        @views dK[:,j].*=ds[j]
        @views ddK[:,j].*=ds[j]
    end
    K.*=-one(T);dK.*=-one(T);ddK.*=-one(T)
    @inbounds for i in axes(K,1) # only the regular kernel has +I, for others vanishes due to taking the derivative
        K[i,i]+=one(T)
    end
    return K,dK,ddK
end

function compute_kernel_matrix(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    K=Matrix{Complex{T}}(undef,N,N)
    compute_kernel_matrix!(K,bp,symmetry,k;multithreaded=multithreaded)
    return K
end

function fredholm_matrix(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    K=Matrix{Complex{T}}(undef,N,N)
    fredholm_matrix!(K,bp,symmetry,k;multithreaded=multithreaded)
    return K
end

function fredholm_matrix_with_derivatives(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp.xy)
    K=Matrix{Complex{T}}(undef,N,N)
    dK=Matrix{Complex{T}}(undef,N,N)
    ddK=Matrix{Complex{T}}(undef,N,N)
    fredholm_matrix_with_derivatives!(K,dK,ddK,bp,symmetry,k;multithreaded=multithreaded)
    return K,dK,ddK
end

"""
    construct_matrices!(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, A, pts, k; multithreaded=true)
    construct_matrices(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, pts, k; multithreaded=true)
    construct_matrices!(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, A, dA, ddA, pts, k; multithreaded=true)
    construct_matrices(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, pts, k, A, dA, ddA; multithreaded=true)
    construct_matrices(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, pts, k; multithreaded=true)

High-level BIM assembly interface for the Fredholm matrix and, optionally, its
first two derivatives with respect to k.

# Overloads
- `construct_matrices!(..., A, pts, k; ...)`
  In-place assembly of the Fredholm matrix into a preallocated buffer.
- `construct_matrices(..., pts, k; ...)`
  Allocation-returning version of the above.
- `construct_matrices!(..., A, dA, ddA, pts, k; ...)`
  In-place assembly of the Fredholm matrix and its first two derivatives.
- `construct_matrices(..., pts, k, A, dA, ddA; ...)`
  Reuse externally allocated buffers and return them.
- `construct_matrices(..., pts, k; ...)` with three return matrices
  Allocating version for matrix-plus-derivatives.

# Arguments
- `solver::BoundaryIntegralMethod`
- `basis::AbstractHankelBasis`
  Placeholder basis object included for interface compatibility.
- `A, dA, ddA`:
  Destination matrices when using in-place forms.
- `pts::BoundaryPoints{T}`:
  Boundary discretization.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Whether to thread the underlying kernel assembly.

# Returns
- Single-matrix forms return `A`.
- Derivative forms return `(A, dA, ddA)`.

# Notes
The `basis` argument is not mathematically used by the direct BIM itself, but it
is kept so this solver can participate in the same interface as the other
spectral methods.
"""
function construct_matrices!(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 fredholm_matrix!(A,pts,solver.symmetry,k;multithreaded=multithreaded)
    return A
end
function construct_matrices!(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 fredholm_matrix_with_derivatives!(A,dA,ddA,pts,solver.symmetry,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k::T,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}};multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    dA=Matrix{Complex{T}}(undef,N,N)
    ddA=Matrix{Complex{T}}(undef,N,N)
    construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    return A,dA,ddA
end

"""
    solve(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, pts, k; multithreaded=true, use_krylov=true, which=:det_argmin)
    solve(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, A, pts, k; multithreaded=true, use_krylov=true, which=:det_argmin)

High-level scalar solver interface for the plain boundary integral method.

# Overloads
- `solve(..., pts, k; ...)`
  Allocates a fresh matrix, assembles the Fredholm operator, and evaluates the
  requested scalar quantity.
- `solve(..., A, pts, k; ...)`
  Reuses a caller-provided matrix buffer `A` to avoid repeated allocations in
  sweeps or local optimization.

# Arguments
- `solver::BoundaryIntegralMethod`
- `basis::AbstractHankelBasis`
  Placeholder basis for interface compatibility.
- `A::AbstractMatrix{Complex{T}}`
  Optional preallocated Fredholm matrix buffer.
- `pts::BoundaryPoints{T}`:
  Boundary discretization.
- `k`:
  Real wavenumber.
- `multithreaded::Bool=true`
  Passed to matrix assembly.
- `use_krylov::Bool=true`
  Forwarded to the scalar-reduction backend.
- `which::Symbol=:det_argmin`
  Selects the returned scalar diagnostic, depending on the backend. Typical
  choices include:
  - `:svd`
  - `:det`
  - `:det_argmin`

# Returns
- A scalar spectral diagnostic, whose exact meaning depends on `which`.
"""
function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve_vect(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, pts, k; multithreaded=true)
    solve_vect(solver::BoundaryIntegralMethod, basis::AbstractHankelBasis, A, pts, k; multithreaded=true)
    solve_vect(solver::BoundaryIntegralMethod, billiard, basis::AbstractHankelBasis, ks::Vector{T}; multithreaded=true)

Compute the smallest singular value of the BIM Fredholm matrix together with the
associated right singular vector.

    A = U Σ V*,

identify the smallest singular value `σ_min`, and return:
- `σ_min`,
- the corresponding right singular vector.

# Overloads
1. `solve_vect(..., pts, k; ...)`
   Allocates the matrix, assembles it, computes the SVD.
2. `solve_vect(..., A, pts, k; ...)`
   Reuses a preallocated matrix buffer.
3. `solve_vect(..., billiard, basis, ks; ...)`
   Convenience batched form over a vector of wavenumbers. For each k it:
   - builds the boundary discretization,
   - computes the smallest singular vector,
   - stores both the vector and the discretization.

# Arguments
- `solver::BoundaryIntegralMethod`
- `basis::AbstractHankelBasis`
- `A::AbstractMatrix{Complex{T}}`
  Optional preallocated Fredholm buffer.
- `pts::BoundaryPoints{T}`
  Boundary discretization.
- `billiard`
  Needed only by the vector-of-k overload to generate discretizations.
- `ks::Vector{T}`
  Wavenumbers for the batched variant.
- `multithreaded::Bool=true`
  Passed to assembly.

# Returns
Single-k overloads:
- `(σ_min, u_min)`

Vector-of-k overload:
- `(us_all, pts_all)`

  where:
  - `us_all[i]` is the smallest right singular vector at `ks[i]`,
  - `pts_all[i]` is the corresponding boundary discretization.
"""
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],conj.(Vt[idx,:])
end

function solve_vect(solver::BoundaryIntegralMethod,billiard::Bi,basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    us_all=Vector{Vector{eltype(complex(ks[1]))}}(undef,length(ks))
    pts_all=Vector{BoundaryPoints{eltype(ks[1])}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];multithreaded=multithreaded)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

# INTERNAL - only for testing performance of the solve workflow, not for actual use in the solver interface
function solve_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    s_constr=time()
    @info "constructing Fredholm matrix A..."
    @blas_1 construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
    @info "Condition number of A for svd: $(cond(A))"
    e_constr=time()
    s_svd=time()
    mu=@svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
    e_svd=time()
    total_time=(e_svd-s_svd)+(e_constr-s_constr)
    @info "Total solve time for test k: $(total_time)"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("Fredholm matrix A construction: $(100*(e_constr-s_constr)/total_time) %")
    println("SVD: $(100*(e_svd-s_svd)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return mu
end


