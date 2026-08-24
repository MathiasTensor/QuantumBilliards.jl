const TWO_PI=2*pi

#TODO Baoling Xie and Jun Lai, A Singularity Guided Nyström Method for Elastostatics on Two Dimensional Domains with Corners, arXiv:2512.18208, 2025

"""
    BoundaryIntegralMethod{T,Sym} <: SweepSolver

Configuration object for the standard boundary integral method (BIM) Fredholm
formulation based on the direct Helmholtz double-layer kernel.

## Description
This solver is the plain boundary-integral implementation in the library. It
does not use Kress logarithmic splitting, Alpert correction, or any special
corner quadrature. Instead, it assembles the Fredholm second-kind matrix
directly from the default two-dimensional Helmholtz double-layer kernel using
the sampled boundary points, outward normals, curvatures and arc-length weights.

Mathematically, the assembled operator is

    A(k)=I-K(k),

where `K(k)` is the Nyström discretization of the boundary double-layer operator
for the interior Helmholtz Dirichlet problem. Symmetry images may be incorporated
directly into the kernel before the Fredholm shift by the identity.

## Attributes
* `dim_scaling_factor`: Compatibility field for the generic solver infrastructure.
* `pts_scaling_factor`: Boundary-resolution scaling factors.
* `sampler`: Sampling rules used on the boundary curves.
* `eps`: Numerical tolerance placeholder.
* `min_dim`: Compatibility field mirroring the other solvers.
* `min_pts`: Minimum number of boundary points per component.
* `symmetry`: Optional symmetry descriptor.

## Notes
For each boundary curve of length `L`, the nominal number of points is chosen as

    N ≈ k*L*b/(2π),

where `b` is the corresponding entry of `pts_scaling_factor`.

Because this is the direct method, near-singular and singular behavior is
handled only through the built-in diagonal curvature correction and the raw
Nyström quadrature. For high precision on difficult geometries, especially
cornered ones, corrected quadrature variants are generally preferable.
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
    AbstractHankelBasis <: AbsBasis

Compatibility placeholder used by the direct boundary integral solver.

## Description
The plain boundary integral method does not employ an explicit basis expansion.
This type is retained so that the solver conforms to the common solver/basis
interface used throughout the package.
"""
struct AbstractHankelBasis <: AbsBasis end

"""
    resize_basis(basis::Ba,billiard::Bi,dim::Int,k) where {Ba<:AbstractHankelBasis,Bi<:AbsBilliard} → basis::AbstractHankelBasis

Returns an empty [`AbstractHankelBasis`](@ref) compatibility object.

## Arguments
* `basis`: Existing basis placeholder.
* `billiard`: Billiard geometry.
* `dim`: Requested basis dimension.
* `k`: Wavenumber.

## Returns
* `basis`: A new [`AbstractHankelBasis`](@ref) placeholder.
"""
function resize_basis(basis::Ba,billiard::Bi,dim::Int,k) where {Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    return AbstractHankelBasis()
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return BoundaryIntegralMethod{T,Sym}(one(T),bs,sampler,eps(T),min_pts,min_pts,symmetry)
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},samplers::Vector,billiard::Bi;min_pts=20,symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    Sym=typeof(symmetry)
    return BoundaryIntegralMethod{T,Sym}(one(T),bs,samplers,eps(T),min_pts,min_pts,symmetry)
end

"""
    evaluate_points(solver::BoundaryIntegralMethod,billiard::Bi,k) where {Bi<:AbsBilliard} → pts::BoundaryPoints

Constructs the boundary discretization used by the plain boundary integral
method at wavenumber `k`.

## Description
Unlike the Kress-based methods, this function does not build a special
logarithmic-correction discretization. It samples the active real boundary
curves and stores the resulting geometry in a [`BoundaryPoints`](@ref) instance.

For each real boundary curve of length `L`, the number of points is chosen
approximately as

    N ≈ k*L*b/(2π),

where `b` is the corresponding entry of `solver.pts_scaling_factor`, subject to

    N ≥ solver.min_pts.

The local arc-length coordinates of successive curves are shifted so that the
stored coordinate `s` is continuous over the concatenated boundary.

The generic [`boundary_coords`](@ref) geometry evaluator is reused to construct
the boundary coordinates, outward normals, arc-length coordinates and
arc-length quadrature elements.

## Arguments
* `solver`: Boundary integral solver configuration.
* `billiard`: Billiard geometry to discretize.
* `k`: Wavenumber controlling the boundary resolution.

## Returns
* `pts`: A [`BoundaryPoints`](@ref) instance with `xy`, `normal`, `s`, `ds`,
  `curvature`, `shift_x` and `shift_y` populated.
"""
function evaluate_points(solver::BoundaryIntegralMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=isnothing(solver.symmetry) ? billiard.full_boundary : billiard.desymmetrized_full_boundary
    T=eltype(solver.pts_scaling_factor)
    xy_all=Vector{SVector{2,T}}()
    normal_all=Vector{SVector{2,T}}()
    s_all=Vector{T}()
    curvature_all=Vector{T}()
    ds_all=Vector{T}()
    soff=zero(T)
    for i in eachindex(curves)
        crv=curves[i]
        if crv isa AbsRealCurve
            L=crv.length
            N=max(solver.min_pts,round(Int,k*L*bs[i]/TWO_PI))
            sampler=samplers[i]
            t,dt=sampler isa PolarSampler ? sample_points(sampler,crv,N) : sample_points(sampler,N)
            xy,normal,s,ds=boundary_coords(crv,t,dt)
            append!(xy_all,xy)
            append!(normal_all,normal)
            append!(s_all,s.-s[1].+soff)
            append!(curvature_all,curvature(crv,t))
            append!(ds_all,ds)
            soff+=L
        end
    end
    shift_x=hasproperty(billiard,:x_axis) ? T(billiard.x_axis) : zero(T)
    shift_y=hasproperty(billiard,:y_axis) ? T(billiard.y_axis) : zero(T)
    return BoundaryPoints(xy_all;normal=normal_all,s=s_all,ds=ds_all,curvature=curvature_all,shift_x=shift_x,shift_y=shift_y)
end

"""
    default_helmholtz_kernel_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real} → K::Matrix{Complex{T}}

Assembles the raw two-dimensional Helmholtz double-layer kernel matrix.

## Description
For the normalization used here,

    K(x,y;k)=(i k/2) cosφ H₁^(1)(k r),

where

    r=|x-y|,
    cosφ=n_y⋅(x-y)/r.

For coincident target and source points, the diagonal limit is

    -κ/(2π),

where `κ` is the boundary curvature.

The returned matrix contains only the raw kernel. Arc-length quadrature weights,
the Fredholm sign and the identity shift are applied later by
[`fredholm_matrix!`](@ref).

## Arguments
* `bp`: Boundary discretization containing points, outward normals and curvature.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `K`: Dense raw Helmholtz double-layer kernel matrix.
"""
function default_helmholtz_kernel_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    curvatures=bp.curvature
    N=length(bp)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    tol=eps(T)
    pref=Complex{T}(zero(T),k/2)
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i
            dx=xi-xs[j];dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            if d<tol
                M[i,j]=-Complex(curvatures[i]/TWO_PI)
            else
                invd=inv(d)
                cos_phi=(nx[j]*dx+ny[j]*dy)*invd
                hankel=pref*Bessels.hankelh1(1,k*d)
                M[i,j]=cos_phi*hankel
                if i!=j
                    cos_phi_symmetric=(nxi*(-dx)+nyi*(-dy))*invd
                    M[j,i]=cos_phi_symmetric*hankel
                end
            end
        end
    end
    filter_matrix!(M)
    return M
end

"""
    default_helmholtz_kernel_derivative_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real} → dK::Matrix{Complex{T}}

Assembles the first derivative with respect to `k` of the raw Helmholtz
double-layer kernel matrix.

## Description
For

    K(x,y;k)=(i k/2) cosφ H₁^(1)(k r),

the first derivative is

    ∂K/∂k=(i k r/2) cosφ H₀^(1)(k r).

The diagonal derivative vanishes because the diagonal curvature limit is
independent of `k`.

## Arguments
* `bp`: Boundary discretization containing points and outward normals.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `dK`: First derivative of the raw DLP kernel matrix.
"""
function default_helmholtz_kernel_derivative_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(bp)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    pref=Complex{T}(zero(T),k/2)
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i-1
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
    default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real} → ddK::Matrix{Complex{T}}

Assembles the second derivative with respect to `k` of the raw Helmholtz
double-layer kernel matrix.

## Description
For the normalization used here,

    ∂²K/∂k² =
        cosφ * i/(2k) *
        [(-2+(kr)²)H₁^(1)(kr)+kr H₂^(1)(kr)].

The diagonal second derivative vanishes because the curvature diagonal limit of
the raw kernel is independent of `k`.

## Arguments
* `bp`: Boundary discretization containing points and outward normals.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `ddK`: Second derivative of the raw DLP kernel matrix.
"""
@inline function default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(bp)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    pref=Complex{T}(zero(T),inv(2*k))
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i-1
            dx=xi-xs[j]
            dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            invd=inv(d)
            kd=k*d
            cos_phi=(nx[j]*dx+ny[j]*dy)*invd
            hankel=pref*((-2+kd*kd)*Bessels.hankelh1(1,kd)+kd*Bessels.hankelh1(2,kd))
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
    compute_kernel_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real} → K::Matrix{Complex{T}}

Allocates and assembles the raw default Helmholtz double-layer kernel matrix
without symmetry.

## Arguments
* `bp`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `K`: Raw Helmholtz double-layer kernel matrix.
"""
function compute_kernel_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp)
    K=Matrix{Complex{T}}(undef,N,N)
    compute_kernel_matrix!(K,bp,k;multithreaded=multithreaded)
    return K
end

"""
    _add_pair_default!(M,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref;scale=1) → flag::Bool

Adds one regular off-diagonal Helmholtz double-layer kernel contribution.

## Description
For a regular target-source pair, the function adds

    scale*((n_j⋅(x_i-x_j))/r)*pref*H₁^(1)(kr)

to `M[i,j]`.

If the squared target-source distance does not exceed `tol2`, no contribution is
added and the function returns `false`, allowing the caller to insert the
appropriate diagonal limit.

## Arguments
* `M`: Destination matrix.
* `i`: Target index.
* `j`: Source index.
* `xi`, `yi`: Target coordinates.
* `nxi`, `nyi`: Target normal components.
* `xj`, `yj`: Source coordinates.
* `nxj`, `nyj`: Source normal components.
* `k`: Real wavenumber.
* `tol2`: Squared coincidence tolerance.
* `pref`: Kernel prefactor.

## Keyword arguments
* `scale`: Optional symmetry or parity factor.

## Returns
* `flag`: `true` when a regular contribution was added, `false` otherwise.
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
    @inbounds M[i,j]+=scale*((nxj*dx+nyj*dy)*invd)*h
    return true
end

"""
    _add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κi,k,tol2;scale=1) → flag::Bool

Adds the raw DLP kernel and its first two wavenumber derivatives for one direct
target-source pair.

## Description
For a regular pair the scalar special-function factors are

    K    : (i k/2)H₁^(1)(kr),
    dK   : (i k r/2)H₀^(1)(kr),
    ddK  : i/(2k)[(-2+(kr)²)H₁^(1)(kr)+kr H₂^(1)(kr)].

The geometric factor is the source-normal cosine

    n_j⋅(x_i-x_j)/r.

Because direct no-symmetry entries may be assembled triangularly, the
corresponding `(j,i)` contribution is also accumulated.

For `i==j`, only the raw-kernel diagonal limit

    -κ/(2π)

is inserted. The first and second derivative diagonal terms vanish.

## Arguments
* `K`: Destination matrix for the raw DLP kernel.
* `dK`: Destination matrix for the first derivative.
* `ddK`: Destination matrix for the second derivative.
* `i`: Target index.
* `j`: Source index.
* `xi`, `yi`: Target coordinates.
* `nxi`, `nyi`: Target normal components.
* `xj`, `yj`: Source coordinates.
* `nxj`, `nyj`: Source normal components.
* `κi`: Curvature at the target point.
* `k`: Real wavenumber.
* `tol2`: Squared coincidence tolerance.

## Keyword arguments
* `scale`: Optional multiplicative factor.

## Returns
* `flag`: `false` for the diagonal contribution and `true` for a regular pair.
"""
@inline function _add_pair3_no_symmetry_default!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real,C<:Complex}
    dx=xi-xj;dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    if i==j
        @inbounds K[i,j]+=-scale*Complex(κi/TWO_PI)
        return false
    end
    d=sqrt(d2);invd=inv(d);kd=k*d
    c_ij=(nxj*dx+nyj*dy)*invd
    c_ji=(nxi*(-dx)+nyi*(-dy))*invd
    H0,H1,H2=Bessels.besselh(0:2,1,kd)
    pref=Complex{T}(zero(T),k/2)
    pref2=Complex{T}(zero(T),inv(2*k))
    hK=pref*H1
    hdK=pref*d*H0
    hddK=pref2*((-2+kd*kd)*H1+kd*H2)
    @inbounds begin
        K[i,j]+=scale*(c_ij*hK)
        dK[i,j]+=scale*(c_ij*hdK)
        ddK[i,j]+=scale*(c_ij*hddK)
        K[j,i]+=scale*(c_ji*hK)
        dK[j,i]+=scale*(c_ji*hdK)
        ddK[j,i]+=scale*(c_ji*hddK)
    end
    return true
end

"""
    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κi,k,tol2;scale=1) → flag::Bool

Adds the DLP kernel and its first two derivatives from a transformed symmetry
image of the source point.

## Description
The function evaluates the same raw DLP kernel and derivative formulas as the
direct contribution, but using image coordinates `(xjr,yjr)` and image normal
`(nxjr,nyjr)`.

No curvature correction is applied to image contributions. If an image point
coincides with the target to within `tol2`, the contribution is skipped.

## Arguments
* `K`: Destination matrix for the raw DLP kernel.
* `dK`: Destination matrix for the first derivative.
* `ddK`: Destination matrix for the second derivative.
* `i`: Target index.
* `j`: Original source index.
* `xi`, `yi`: Target coordinates.
* `nxi`, `nyi`: Target normal components.
* `xjr`, `yjr`: Symmetry-image source coordinates.
* `nxjr`, `nyjr`: Symmetry-image source normal.
* `κi`: Curvature argument retained for interface consistency.
* `k`: Real wavenumber.
* `tol2`: Squared coincidence tolerance.

## Keyword arguments
* `scale`: Symmetry parity or rotational character factor.

## Returns
* `flag`: `true` if the image contribution was added, `false` if it was skipped.
"""
@inline function _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κi,k,tol2;scale=one(eltype(K)))
    dx=xi-xjr
    dy=yi-yjr
    d2=muladd(dx,dx,dy*dy)
    d2<=tol2&&return false
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

"""
    compute_kernel_matrix!(K,bp,k;multithreaded=true) → K

Assembles the raw Helmholtz double-layer kernel matrix in place without
symmetry images.

## Arguments
* `K`: Preallocated destination matrix.
* `bp`: Boundary discretization containing coordinates, normals and curvature.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `K`: The modified raw DLP kernel matrix.
"""
function compute_kernel_matrix!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(K,Complex{T}(zero(T),zero(T)))
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature
    N=length(bp)
    tol2=eps(T)^2
    pref=Complex{T}(zero(T),k/2)
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
                K[i,j]+=-Complex(κ[i]/TWO_PI)
            end
        end
    end
    return K
end

"""
    compute_kernel_matrix!(K,bp,symmetry,k;multithreaded=true) → K

Assembles the raw Helmholtz double-layer kernel matrix in place including
symmetry-image contributions.

## Description
The direct DLP contribution is assembled for every target-source pair.
Depending on `symmetry`, reflected or rotated image contributions are then
added with the corresponding parity or character factors.

Supported symmetry pathways include reflections across the x-axis, y-axis,
both axes, and finite rotations.

## Arguments
* `K`: Preallocated destination matrix.
* `bp`: Boundary discretization containing coordinates, normals, curvature and symmetry shifts.
* `symmetry`: Reflection or rotation symmetry descriptor.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `K`: Raw DLP kernel matrix including symmetry images.
"""
function compute_kernel_matrix!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    fill!(K,Complex{T}(zero(T),zero(T)))
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature
    N=length(bp)
    tol2=eps(T)^2
    pref=Complex{T}(zero(T),k/2)
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
            sxgn=s.parity==-1 ? -one(T) : one(T)
        end
        if s.axis==:x_axis
            add_y=true
            sygn=s.parity==-1 ? -one(T) : one(T)
        end
        if s.axis==:origin
            add_x=true
            add_y=true
            add_xy=true
            sxgn=s.parity[1]==-1 ? -one(T) : one(T)
            sygn=s.parity[2]==-1 ? -one(T) : one(T)
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
                K[i,j]+=-Complex(κ[i]/TWO_PI)
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
    compute_kernel_matrix_with_derivatives!(K,dK,ddK,bp,k;multithreaded=true) → K,dK,ddK

Assembles the raw DLP kernel and its first two derivatives in place without
symmetry.

## Arguments
* `K`: Destination matrix for the raw DLP kernel.
* `dK`: Destination matrix for the first wavenumber derivative.
* `ddK`: Destination matrix for the second wavenumber derivative.
* `bp`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded assembly.

## Returns
* `K`: Raw DLP kernel matrix.
* `dK`: First derivative of the raw kernel matrix.
* `ddK`: Second derivative of the raw kernel matrix.
"""
function compute_kernel_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp)
    fill!(K,Complex{T}(zero(T),zero(T)))
    fill!(dK,Complex{T}(zero(T),zero(T)))
    fill!(ddK,Complex{T}(zero(T),zero(T)))
    xs=getindex.(bp.xy,1);ys=getindex.(bp.xy,2)
    nx=getindex.(bp.normal,1);ny=getindex.(bp.normal,2)
    κ=bp.curvature
    tol2=eps(T)^2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i
            xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
            _add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2)
        end
    end
    return K,dK,ddK
end

"""
    compute_kernel_matrix_with_derivatives!(K,dK,ddK,bp,symmetry,k;multithreaded=true) → K,dK,ddK

Assembles the raw DLP kernel and its first two derivatives in place including
symmetry-image contributions.

## Description
The direct contribution is assembled triangularly and mirrored. Reflection or
rotation image contributions are then added independently for every ordered
target-source pair.

## Arguments
* `K`: Destination matrix for the raw DLP kernel.
* `dK`: Destination matrix for the first derivative.
* `ddK`: Destination matrix for the second derivative.
* `bp`: Boundary discretization.
* `symmetry`: Reflection or rotation symmetry descriptor.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded assembly.

## Returns
* `K`: Raw DLP kernel matrix.
* `dK`: First derivative.
* `ddK`: Second derivative.
"""
function compute_kernel_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp)
    fill!(K,Complex{T}(zero(T),zero(T)))
    fill!(dK,Complex{T}(zero(T),zero(T)))
    fill!(ddK,Complex{T}(zero(T),zero(T)))
    xs=getindex.(bp.xy,1);ys=getindex.(bp.xy,2)
    nx=getindex.(bp.normal,1);ny=getindex.(bp.normal,2)
    κ=bp.curvature
    tol2=eps(T)^2
    shift_x=bp.shift_x;shift_y=bp.shift_y
    add_x=false;add_y=false;add_xy=false
    sxgn=one(T);sygn=one(T);sxy=one(T)
    have_rot=false
    nrot=1;mrot=0
    cx=zero(T);cy=zero(T)
    s=symmetry
    if hasproperty(s,:axis)
        if s.axis==:y_axis;add_x=true;sxgn=s.parity==-1 ? -one(T) : one(T);end
        if s.axis==:x_axis;add_y=true;sygn=s.parity==-1 ? -one(T) : one(T);end
        if s.axis==:origin
            add_x=true;add_y=true;add_xy=true
            sxgn=s.parity[1]==-1 ? -one(T) : one(T)
            sygn=s.parity[2]==-1 ? -one(T) : one(T)
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
        xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:N
            xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
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
    fredholm_matrix!(K,bp,symmetry,k;multithreaded=true) → K

Assembles the Fredholm second-kind matrix in place.

## Description
Starting from the raw DLP kernel, the function applies the source quadrature
weights and forms

    A(k)=I-K(k)W,

where

    W=diag(bp.ds).

## Arguments
* `K`: Preallocated destination matrix.
* `bp`: Boundary discretization containing the arc-length quadrature weights.
* `symmetry`: Optional symmetry descriptor or `nothing`.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `K`: The modified matrix containing the Fredholm operator.
"""
function fredholm_matrix!(K::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    isnothing(symmetry) ? compute_kernel_matrix!(K,bp,k;multithreaded=multithreaded) : compute_kernel_matrix!(K,bp,symmetry,k;multithreaded=multithreaded)
    ds=bp.ds
    @inbounds for j in eachindex(ds)
        @views K[:,j].*=ds[j]
    end
    K.*=-one(T)
    @inbounds for i in axes(K,1)
        K[i,i]+=one(T)
    end
    return K
end

"""
    fredholm_matrix_with_derivatives!(K,dK,ddK,bp,symmetry,k;multithreaded=true) → K,dK,ddK

Assembles the Fredholm matrix and its first two wavenumber derivatives in place.

## Description
The raw kernel matrices are first assembled and right-scaled by the arc-length
quadrature weights. The Fredholm sign is applied to all three matrices, while
the identity contribution is added only to `K`:

    A(k)=I-K(k)W,
    A'(k)=-K'(k)W,
    A''(k)=-K''(k)W.

## Arguments
* `K`: Destination matrix for the Fredholm operator.
* `dK`: Destination matrix for its first derivative.
* `ddK`: Destination matrix for its second derivative.
* `bp`: Boundary discretization.
* `symmetry`: Optional symmetry descriptor or `nothing`.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `K`: Fredholm matrix.
* `dK`: First derivative of the Fredholm matrix.
* `ddK`: Second derivative of the Fredholm matrix.
"""
function fredholm_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(symmetry)
        compute_kernel_matrix_with_derivatives!(K,dK,ddK,bp,k;multithreaded=multithreaded)
    else
        compute_kernel_matrix_with_derivatives!(K,dK,ddK,bp,symmetry,k;multithreaded=multithreaded)
    end
    ds=bp.ds
    @inbounds for j in eachindex(ds)
        @views K[:,j].*=ds[j]
        @views dK[:,j].*=ds[j]
        @views ddK[:,j].*=ds[j]
    end
    K.*=-one(T);dK.*=-one(T);ddK.*=-one(T)
    @inbounds for i in axes(K,1)
        K[i,i]+=one(T)
    end
    return K,dK,ddK
end

"""
    compute_kernel_matrix(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real} → K::Matrix{Complex{T}}

Allocates and assembles the raw DLP kernel matrix with the supplied symmetry.

## Arguments
* `bp`: Boundary discretization.
* `symmetry`: Symmetry descriptor.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded assembly.

## Returns
* `K`: Raw DLP kernel matrix.
"""
function compute_kernel_matrix(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp)
    K=Matrix{Complex{T}}(undef,N,N)
    compute_kernel_matrix!(K,bp,symmetry,k;multithreaded=multithreaded)
    return K
end

"""
    fredholm_matrix(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real} → K::Matrix{Complex{T}}

Allocates and assembles the BIM Fredholm matrix.

## Arguments
* `bp`: Boundary discretization.
* `symmetry`: Optional symmetry descriptor.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded assembly.

## Returns
* `K`: Dense Fredholm matrix.
"""
function fredholm_matrix(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp)
    K=Matrix{Complex{T}}(undef,N,N)
    fredholm_matrix!(K,bp,symmetry,k;multithreaded=multithreaded)
    return K
end

"""
    fredholm_matrix_with_derivatives(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real} → K,dK,ddK

Allocates and assembles the BIM Fredholm matrix and its first two derivatives.

## Arguments
* `bp`: Boundary discretization.
* `symmetry`: Optional symmetry descriptor.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded assembly.

## Returns
* `K`: Fredholm matrix.
* `dK`: First derivative.
* `ddK`: Second derivative.
"""
function fredholm_matrix_with_derivatives(bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(bp)
    K=Matrix{Complex{T}}(undef,N,N)
    dK=Matrix{Complex{T}}(undef,N,N)
    ddK=Matrix{Complex{T}}(undef,N,N)
    fredholm_matrix_with_derivatives!(K,dK,ddK,bp,symmetry,k;multithreaded=multithreaded)
    return K,dK,ddK
end

#########################################
########## NEEDED FOR HUSIMIS ###########
#########################################

"""
    adjoint_fredholm_matrix!(A,D,bp,symmetry,k;multithreaded=true) → A

Assembles the discrete adjoint Fredholm matrix used for boundary-function and
Husimi postprocessing.

## Description
If `D` denotes the quadrature-weighted source-normal double-layer matrix, the
discrete adjoint is formed as

    K'=W⁻¹DᵀW,

where

    W=diag(bp.ds).

The corresponding adjoint Fredholm operator is

    A=I-K'.

The right null vector of this matrix corresponds directly to the boundary
normal derivative used in Husimi and boundary-function postprocessing.

## Arguments
* `A`: Preallocated destination matrix for the adjoint Fredholm operator.
* `D`: Preallocated workspace for the direct DLP matrix.
* `bp`: Boundary discretization.
* `symmetry`: Optional symmetry descriptor or `nothing`.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `A`: The assembled adjoint Fredholm matrix.
"""
function adjoint_fredholm_matrix!(A::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry,k::T;multithreaded::Bool=true) where {T<:Real}
    isnothing(symmetry) ? compute_kernel_matrix!(D,bp,k;multithreaded=multithreaded) : compute_kernel_matrix!(D,bp,symmetry,k;multithreaded=multithreaded)
    ds=bp.ds
    @inbounds for j in eachindex(ds)
        @views D[:,j].*=ds[j]
    end
    @inbounds for i in axes(A,1),j in axes(A,2)
        A[i,j]=-D[j,i]*ds[j]/ds[i]
    end
    @inbounds for i in axes(A,1)
        A[i,i]+=one(Complex{T})
    end
    return A
end

"""
    smallest_nullvec_krylov!(A;nev=1,tol=1e-12,maxiter=2000,krylovdim=40) → σ,u,info

Computes a smallest-null-vector proxy by applying a Krylov eigensolver to the
inverse of `A`.

## Description
The matrix is LU-factorized once and the inverse action

    x ↦ A⁻¹x

is represented as a `LinearMap`. The largest-magnitude eigenpair of `A⁻¹`
corresponds to the eigenvalue of `A` nearest zero.

The returned scalar

    σ=1/|μ|

is therefore a smallest-eigenvalue/singular-value proxy, while `u` is the
associated normalized vector.

## Arguments
* `A`: Matrix whose near-null vector is sought.

## Keyword arguments
* `nev`: Number of Krylov eigenpairs requested.
* `tol`: Krylov convergence tolerance.
* `maxiter`: Maximum number of Krylov iterations.
* `krylovdim`: Dimension of the Krylov subspace.

## Returns
* `σ`: Smallest-eigenvalue/singular-value proxy.
* `u`: Normalized approximate null vector.
* `info`: Krylov solver diagnostic information.
"""
function smallest_nullvec_krylov!(A::AbstractMatrix{Complex{T}};nev::Int=1,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real}
    n=size(A,1)
    F=lu!(A)
    function op!(y,x)
        copyto!(y,x)
        ldiv!(F,y)
        return y
    end
    C=LinearMaps.LinearMap{eltype(A)}(op!,n,n;ismutating=true)
    μs,vecs,info=eigsolve(C,n,nev,:LM;tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    μ=μs[1]
    u=vecs[1]./norm(vecs[1])
    σ=inv(abs(μ))
    return σ,u,info
end

#########################################

"""
    construct_matrices!(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,A,pts,k;multithreaded=true) → A

Assembles the BIM Fredholm matrix into a preallocated matrix.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `A`: Preallocated destination matrix.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `A`: Assembled Fredholm matrix.
"""
function construct_matrices!(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 fredholm_matrix!(A,pts,solver.symmetry,k;multithreaded=multithreaded)
    return A
end

"""
    construct_matrices!(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,A,dA,ddA,pts,k;multithreaded=true) → A,dA,ddA

Assembles the BIM Fredholm matrix and its first two derivatives into
preallocated matrices.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `A`: Destination matrix for the Fredholm operator.
* `dA`: Destination matrix for the first derivative.
* `ddA`: Destination matrix for the second derivative.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `A`: Fredholm matrix.
* `dA`: First derivative.
* `ddA`: Second derivative.
"""
function construct_matrices!(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 fredholm_matrix_with_derivatives!(A,dA,ddA,pts,solver.symmetry,k;multithreaded=multithreaded)
    return A,dA,ddA
end

"""
    construct_matrices(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,pts,k,A,dA,ddA;multithreaded=true) → A,dA,ddA

Reuses externally allocated buffers to assemble the BIM Fredholm matrix and its
first two derivatives.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.
* `A`: Destination matrix for the Fredholm operator.
* `dA`: Destination matrix for the first derivative.
* `ddA`: Destination matrix for the second derivative.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `A`: Fredholm matrix.
* `dA`: First derivative.
* `ddA`: Second derivative.
"""
function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k::T,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}};multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    return A,dA,ddA
end

"""
    construct_matrices(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,pts,k;multithreaded=true) → A,dA,ddA

Allocates and assembles the BIM Fredholm matrix and its first two derivatives.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded kernel assembly.

## Returns
* `A`: Fredholm matrix.
* `dA`: First derivative.
* `ddA`: Second derivative.
"""
function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts)
    A=Matrix{Complex{T}}(undef,N,N)
    dA=Matrix{Complex{T}}(undef,N,N)
    ddA=Matrix{Complex{T}}(undef,N,N)
    construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded)
    return A,dA,ddA
end

"""
    solve(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,pts,k;multithreaded=true,use_krylov=true,which=:det_argmin)

Evaluates the selected scalar spectral diagnostic of the BIM Fredholm matrix.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded matrix assembly.
* `use_krylov`: Whether the scalar-reduction backend should use its Krylov path.
* `which`: Scalar diagnostic requested by the backend.

## Returns
* Scalar spectral diagnostic selected by `which`.
"""
function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts)
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,A,pts,k;multithreaded=true,use_krylov=true,which=:det_argmin)

Evaluates the selected scalar spectral diagnostic while reusing a preallocated
Fredholm matrix.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `A`: Preallocated Fredholm matrix buffer.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded matrix assembly.
* `use_krylov`: Whether the scalar-reduction backend should use its Krylov path.
* `which`: Scalar diagnostic requested by the backend.

## Returns
* Scalar spectral diagnostic selected by `which`.
"""
function solve(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve_vect(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,pts,k;multithreaded=true,tol=1e-12,maxiter=2000,krylovdim=40) → σ,u

Computes a near-null vector of the adjoint BIM Fredholm matrix.

## Description
The adjoint Fredholm matrix is assembled and the Krylov inverse-eigenvalue
procedure [`smallest_nullvec_krylov!`](@ref) is used to obtain the smallest
spectral proxy and associated boundary normal-derivative vector.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded matrix assembly.
* `tol`: Krylov convergence tolerance.
* `maxiter`: Maximum number of Krylov iterations.
* `krylovdim`: Krylov subspace dimension.

## Returns
* `σ`: Smallest-eigenvalue/singular-value proxy.
* `u`: Associated normalized near-null vector.
"""
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts)
    A=Matrix{Complex{T}}(undef,N,N)
    D=similar(A)
    @blas_1 adjoint_fredholm_matrix!(A,D,pts,solver.symmetry,k;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

"""
    solve_vect(solver::BoundaryIntegralMethod,basis::AbstractHankelBasis,A,pts,k;multithreaded=true,tol=1e-12,maxiter=2000,krylovdim=40) → σ,u

Computes a near-null vector of the adjoint BIM Fredholm matrix while reusing a
preallocated matrix buffer.

## Arguments
* `solver`: Boundary integral solver configuration.
* `basis`: Compatibility basis placeholder.
* `A`: Preallocated adjoint Fredholm matrix buffer.
* `pts`: Boundary discretization.
* `k`: Real wavenumber.

## Keyword arguments
* `multithreaded`: Whether to use threaded matrix assembly.
* `tol`: Krylov convergence tolerance.
* `maxiter`: Maximum number of Krylov iterations.
* `krylovdim`: Krylov subspace dimension.

## Returns
* `σ`: Smallest-eigenvalue/singular-value proxy.
* `u`: Associated normalized near-null vector.
"""
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {Ba<:AbstractHankelBasis,T<:Real}
    D=similar(A)
    @blas_1 adjoint_fredholm_matrix!(A,D,pts,solver.symmetry,k;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

# INTERNAL - only for testing performance of the solve workflow, not for actual use in the solver interface
function solve_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts)
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