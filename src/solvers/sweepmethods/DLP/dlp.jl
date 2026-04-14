using LinearAlgebra, StaticArrays, TimerOutputs, Bessels
const TO=TimerOutput()
const TWO_PI=2*pi
const FOUR_PI=4*pi

"""
    struct BoundaryIntegralMethod{T<:Real}

Represents the configuration for the boundary integral method.

# Fields
- `dim_scaling_factor::T`: Scaling factor for the boundary dimensions (compatibility).
- `pts_scaling_factor::Vector{T}`: Scaling factors for the boundary points.
- `sampler::Vector`: Sampling strategy for the boundary points.
- `eps::T`: Numerical tolerance.
- `min_dim::Int64`: Minimum dimensions (compatibility field).
- `min_pts::Int64`: Minimum points for evaluation.
- `symmetry::Sym`: Symmetry for the configuration - nothing,Reflection,Rotation
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


### STANDARD BIM ###

"""
    BoundaryIntegralMethod(pts_scaling_factor, billiard::Bi; min_pts=20, symmetries=Nothing, x_bc=:D, y_bc=:D) -> BoundaryIntegralMethod

Creates a boundary integral method solver configuration.

# Arguments
- `pts_scaling_factor::Union{T,Vector{T}}`: Scaling factors for the boundary points.
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `min_pts::Int`: Minimum number of boundary points (default: 20).
- `symmetries::Union{Vector{Any},Nothing}`: Symmetry definitions (-1 Dirichlet for the given axis, 1 otherwise).

# Returns
- `BoundaryIntegralMethod`: Constructed solver configuration.
"""
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
    evaluate_points(solver::BoundaryIntegralMethod, billiard::Bi, k::Real) -> BoundaryPoints

Evaluates the boundary points and associated properties for the given solver and billiard.

# Arguments
- `solver::BoundaryIntegralMethod`: Boundary integral method configuration.
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `k::Real`: Wavenumber.

# Returns
- `BoundaryPoints{T}`: Evaluated boundary points and properties.
"""
function evaluate_points(solver::BoundaryIntegralMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=_boundary_curves_for_solver(billiard,solver)
    type=eltype(solver.pts_scaling_factor)
    xy_all=Vector{SVector{2,type}}()
    normal_all=Vector{SVector{2,type}}()
    kappa_all=Vector{type}()
    w_all=Vector{type}()
    for i in eachindex(curves)
        crv=curves[i]
        if typeof(crv)<:AbsRealCurve
            L=crv.length
            N=max(solver.min_pts,round(Int,k*L*bs[i]/(2*pi)))
            sampler=samplers[i]
            if sampler isa PolarSampler
                t,dt=sample_points(sampler,crv,N)
            else
                t,dt=sample_points(sampler,N)
            end
            s=arc_length(crv,t)
            ds=diff(s)
            append!(ds,L+s[1]-s[end]) # add the last difference as we have 1 less element. Add L to s[1] so we can logically subtract s[end]
            xy=curve(crv,t)
            normal=normal_vec(crv,t)
            kappa=curvature(crv,t)
            append!(xy_all,xy)
            append!(normal_all,normal)
            append!(kappa_all,kappa)
            append!(w_all,ds)
        end
    end
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : type(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : type(0.0)
    return BoundaryPoints{type}(xy_all,normal_all,Vector{type}(),w_all,Vector{type}(),Vector{type}(),kappa_all,Vector{SVector{2,type}}(),shift_x,shift_y)
end

"""
    default_helmholtz_kernel_matrix(bp::BoundaryPoints{T}, k::T) -> Matrix{Complex{T}}

Computes the Helmholtz kernel matrix for the given boundary points using the matrix-based approach.

# Arguments
- `bp::BoundaryPoints{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: A matrix where each element corresponds to the Helmholtz kernel between boundary points, incorporating curvature for singular cases.
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
                M[i,j]= Complex(curvatures[i]/TWO_PI) 
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
    default_helmholtz_kernel_derivative_matrix(bp::BoundaryPoints{T}, k::T) -> Matrix{Complex{T}}

Constructs the first derivative (with respect to `k`) of the 2D Helmholtz kernel for all pairs of points
in the boundary `bp`.

where:
- `r` is the distance between points `i` and `j`,
- `cos(φᵢ)` is `(nᵢ · (pᵢ - pⱼ)) / r`, using the normal at `pᵢ`,
- `H₀^(1)` is the Hankel function of the first kind, order 0.

# Arguments
- `bp::BoundaryPoints{T}`: A set of boundary points, including `(x, y)` coordinates and normals.
- `k::T`: Wavenumber, a real value.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: An `N×N` matrix, where `N` is the number of boundary points. The element `(i,j)`
  is the derivative of the Helmholtz kernel with respect to `k` between points `i` and `j`. Diagonal
  entries where `distance < eps(T)` are set to zero.
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
    default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPoints{T}, k::T)
        -> Matrix{Complex{T}}

Constructs the second derivative (with respect to `k`) of the 2D Helmholtz kernel *for all pairs* of points
in the boundary `bp`. Each entry `(i, j)` in the returned matrix corresponds to

    cos(φᵢ) * ( im/(2*k) ) * [ ...combination of HankelH1(1) and HankelH1(2)... ]

where `cos(φᵢ) = (nᵢ · (pᵢ - pⱼ)) / r` and `r` is the distance between boundary points `pᵢ` and `pⱼ`.
The exact Hankel expression matches the partial derivative:
    
    (d²/dk²) of [ cos(φᵢ) * H₀^(1)(k*r)* ... ].

# Arguments
- `bp::BoundaryPoints{T}`: Boundary points, containing `(x, y)` and normals.
- `k::T`: Wavenumber, real.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: An `N×N` matrix, where each entry is the second derivative of the Helmholtz kernel
  wrt. `k` between boundary points `i` and `j`. If `distance < eps(T)`, the entry is set to zero.
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
    compute_kernel_matrix(bp::BoundaryPoints{T}, k::T; kernel_fun::Union{Symbol, Function}=:default) -> Matrix{Complex{T}}

Computes the kernel matrix for the given boundary points using the specified kernel function w/ NO symmetry.

# Arguments
- `bp::BoundaryPoints{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The computed kernel matrix.
"""
function compute_kernel_matrix(bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return default_helmholtz_kernel_matrix(bp,k;multithreaded=multithreaded)
end

"""
    @inline function add_pair_default!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::T,tol2::T,pref::Complex{T};scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real} -> Bool

Compute and add the default Helmholtz double-layer contribution for the pair (i,j).
Writes scalars directly into `M` (both M[i,j] and M[j,i] if i≠j). Returns `true`
if the pair was non-singular (distance² > tol2), `false` otherwise.
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
    _add_pair3_no_symmetry_default!(
        K::AbstractMatrix{C},
        dK::AbstractMatrix{C},
        ddK::AbstractMatrix{C},
        i::Int, j::Int,
        xi::T, yi::T, nxi::T, nyi::T,
        xj::T, yj::T, nxj::T, nyj::T,
        κi::T, k::T, tol2::T;
        scale::Union{T,Complex{T}} = one(Complex{T})
    )::Bool where {T<:Real, C<:Complex}

Add the default 2D Helmholtz double-layer contribution for the pair `(i,j)` without symmetry images, together with
its first and second derivatives w.r.t. `k`. On the diagonal (`i==j`) the curvature term `κi/(2π)` is added to `K`
and the function returns `false`. For off-diagonal pairs, the routine fills both directions `(i,j)` and `(j,i)`.

The default kernel and its k-derivatives are:
- `K:   cosφ * (-im*k/2) * H₁^{(1)}(k r)`
- `dK:  cosφ * (-im*k/2) * r * H₀^{(1)}(k r)`
- `ddK: cosφ * (im/(2k)) * [ (-2 + (k r)^2) H₁^{(1)}(k r) + (k r) H₂^{(1)}(k r) ]`

where `r = ‖(xi,yi)-(xj,yj)‖` and `cosφ = (nxi,nyi)⋅((xi,yi)-(xj,yj))/r`.

# Arguments
- `K, dK, ddK`: `AbstractMatrix{C}` – destination matrices for kernel, first and second k-derivatives.
- `i, j`: `Int` – target and source indices.
- `xi, yi`: `T` – target coordinates.
- `nxi, nyi`: `T` – target outward unit normal.
- `xj, yj`: `T` – source coordinates.
- `nxj, nyj`: `T` – source unit normal.
- `κi`: `T` – boundary curvature at target point `i`.
- `k`: `T` – (real) wavenumber.
- `tol2`: `T` – squared distance threshold; pairs with `r^2 ≤ tol2` are treated as singular.
- `scale`: `Union{T,Complex{T}}` – multiplicative factor (e.g., parity `±1` or a symmetry character). Defaults to `1+0im`.

# Returns
- `Bool`: `false` for diagonal/self (`i==j`, curvature term added to `K[i,i]`), `true` for regular off-diagonal pairs.
"""
@inline function _add_pair3_no_symmetry_default!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real,C<:Complex}
    dx=xi-xj;dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    if i==j # when we have no symmetry safely modify the Diagonal elements, otherwise the d^2<tol2 check in the symmetry version 
        @inbounds K[i,j]+=-scale*Complex(κi/TWO_PI)
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
    _add_pair3_image_default!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xjr::T,yjr::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::Union{T,Complex{T}}=one(Complex{T}))::Bool where {T<:Real}

Add the **default** Helmholtz double-layer kernel (and its k-derivatives) for a **symmetry image** of the source,
i.e. the source at `(xjr, yjr)` obtained by reflection/rotation of `(xj, yj)`. No curvature is added for coincident
image pairs; if `‖(xi,yi)-(xjr,yjr)‖^2 ≤ tol2`, the contribution is skipped and the function returns `false`.

See `_add_pair3_no_symmetry_default!` for the exact kernel forms.

# Arguments
- `K, dK, ddK`: `AbstractMatrix{Complex{T}}` – destination matrices.
- `i, j`: `Int` – target and (original) source indices.
- `xi, yi, nxi, nyi`: `T` – target coordinates and outward unit normal.
- `xjr, yjr`: `T` – **image** source coordinates (already reflected/rotated).
- `nxj, nyj`: `T` – original source outward normal (unused by default kernel for the `(i,j)` entry).
- `κi`: `T` – curvature at target `i` (not used here since no diagonal/image curvature is added).
- `k`: `T` – wavenumber.
- `tol2`: `T` – squared distance tolerance for skipping coincident image pairs.
- `scale`: `Union{T,Complex{T}}` – symmetry factor (parity `±1` or rotation character `e^{iθ}`).

# Returns
- `Bool`: `true` if the image contribution was added; `false` if it was skipped due to `r^2 ≤ tol2`.
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
    @inbounds(K[i,j]+=scale*(cij*hK);dK[i,j]+=scale*(cij*hdK);ddK[i,j]+=scale*(cij*hddK))
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
                K[i,j]+= Complex(κ[i]/TWO_PI)
            end
        end
    end
    return K
end

"""
    compute_kernel_matrix(bp::BoundaryPoints{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T; multithreaded::Bool=true) -> Matrix{Complex{T}}

Computes the kernel matrix for the given boundary points with symmetry reflections applied.

# Arguments
- `K::AbstractMatrix{Complex{T}}`: Destination matrix for the kernel values.
- `bp::BoundaryPoints{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `symmetry::Sym`: Symmetry to apply.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The computed kernel matrix with symmetry reflections applied.
"""
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
                K[i,j]+= Complex(κ[i]/TWO_PI) 
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
    compute_kernel_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real} -> Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}

Build the kernel matrix `K` and its first/second derivatives w.r.t. `k` without symmetry images for a set of
boundary points.
# Arguments
- `K`: `AbstractMatrix{Complex{T}}` – destination matrix for kernel values.
- `dK`: `AbstractMatrix{Complex{T}}` – destination matrix for first derivatives w.r.t. `k`.
- `ddK`: `AbstractMatrix{Complex{T}}` – destination matrix for second derivatives w.r.t. `k`.
- `bp`: `BoundaryPoints{T}` – boundary data (points `xy`, normals, curvature `κ`, and arc-length weights `ds`).
- `k`: `T` – (real) wavenumber about which derivatives are taken.
- `multithreaded`: `Bool` – enable threaded assembly.

# Returns
- `K`:   `Matrix{Complex{T}}` – kernel matrix.
- `dK`:  `Matrix{Complex{T}}` – first derivative w.r.t. `k`.
- `ddK`: `Matrix{Complex{T}}` – second derivative w.r.t. `k`.

Notes:
- Diagonal entries of `K` receive the curvature term `κ/(2π)`; `dK` and `ddK` have zero diagonals for the default kernels.
- Off-diagonal entries fill both `(i,j)` and `(j,i)` for the default kernels to account for different normals.
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

"""
    compute_kernel_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},bp::BoundaryPoints{T},symmetry::Sym,k::T;multithreaded::Bool=true) where {T<:Real, Sym<:AbsSymmetry} -> Tuple{Matrix{Complex{T}},Matrix{Complex{T}},Matrix{Complex{T}}}

- Reflections with fields `axis` (`:x_axis`, `:y_axis`, or `:origin`) and `parity` (`±1` for single-axis or a length-2
  tuple for `:origin`). These contribute with scale factors `sxgn`, `sygn`, or `sxy = sxgn*sygn`.
- `Rotation` symmetries with fields `n::Int` (order), `m::Int` (representation index, taken `mod n`), and
  `center::Tuple{T,T}`. Images are added for `l=1,…,n-1` by rotating the source and multiplying by the character
  `χ_l = exp(im * 2π * m * l / n)`.

For the **default** kernels the source normal of images is not transformed (the DLP at `(i,j)` uses the **target**
normal). For **custom** kernels, the image source normal is reflected/rotated before calling the user callbacks.

# Arguments
- `K`: `AbstractMatrix{Complex{T}}` – destination matrix for kernel values.
- `dK`: `AbstractMatrix{Complex{T}}` – destination matrix for first derivatives w.r.t. `k`.
- `ddK`: `AbstractMatrix{Complex{T}}` – destination matrix for second derivatives w.r.t. `k`.
- `bp`: `BoundaryPoints{T}` – boundary data (points `xy`, normals, curvature `κ`, arc-length weights `ds`,
  and shifts `shift_x`, `shift_y` for reflection axes).
- `symmetry`: `Sym` – symmetry descriptor (reflections and/or `Rotation`).
- `k`: `T` – wavenumber.
- `multithreaded`: `Bool` – enable threaded assembly.

# Returns
- `K`:   `Matrix{Complex{T}}` – kernel matrix including image contributions.
- `dK`:  `Matrix{Complex{T}}` – first derivative w.r.t. `k`.
- `ddK`: `Matrix{Complex{T}}` – second derivative w.r.t. `k`.

Notes:
- Image self-pairs are **not** given curvature; if an image falls within `tol2` the contribution is skipped.
- Reflection scales are real (`±1`); rotation scales are unit-modulus complex characters `χ_l`.
"""
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
                # reflected legs (always full j=1:N; never add curvature)
                if add_x
                    xjr=_x_reflect(xj,shift_x);yjr=yj
                    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sxgn)
                end
                if add_y
                    xjr=xj;yjr=_y_reflect(yj,shift_y)
                    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sygn)
                end
                if add_xy
                    xjr=_x_reflect(xj,shift_x);yjr=_y_reflect(yj,shift_y)
                    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sxy)
                end
                if have_rot
                @inbounds for l in 1:nrot-1 # l=0 is the direct term we already added; add l=1..nrot-1
                    cl=ctab[l+1];sl=stab[l+1]
                    xjr,yjr=_rot_point(xj,yj,cx,cy,cl,sl)
                    phase=χ[l+1]  # e^{i 2π m l / n}, reflections due to being 1d-irreps have real characters
                    _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=phase)
                end
            end
            end
        end
    return K,dK,ddK
end

"""
    fredholm_matrix!(K::AbstractMatrix{Complex{T}}, bp::BoundaryPoints{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T; multithreaded::Bool=true)

Constructs the Fredholm matrix for the boundary integral method using preallocated matrices.

# Arguments
- `K::AbstractMatrix{Complex{T}}`: Destination matrix for the Fredholm operator.
- `bp::BoundaryPoints{T}`: Boundary points structure containing the source points, normal vectors, curvatures, and differential arc lengths.
- `symmetry::Sym`: Symmetry to apply.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The constructed Fredholm matrix, incorporating differential arc lengths and symmetry reflections.
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
    fredholm_matrix_with_derivatives!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}}, bp::BoundaryPoints{T},symmetry::Sym,k::T;multithreaded::Bool=true) where {T<:Real}

Build the Fredholm matrix `A` and it's derivative matrices `dA/dk & d^2A/dk^2`.

# Arguments
- `K::AbstractMatrix{Complex{T}}`: Destination matrix for the Fredholm operator.
- `dK::AbstractMatrix{Complex{T}}`: Destination matrix for the first derivative w.r.t. `k`.
- `ddK::AbstractMatrix{Complex{T}}`: Destination matrix for the second derivative w.r.t. `k`.
- `bp::BoundaryPoints{T}`: Boundary points with `(x,y)`, normals, and `ds`.
- `symmetry::Sym`: Symmetry to apply.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Tuple{Matrix{Complex{T}},Matrix{Complex{T}},Matrix{Complex{T}}}`: The 3`N×N` matrices representing the Fredholm matrix and it's first and second derivative, respectively.
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

function construct_matrices!(solver::BoundaryIntegralMethod,basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    @blas_1 fredholm_matrix!(A,pts,solver.symmetry,k;multithreaded=multithreaded)
    return A
end
function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    construct_matrices!(solver,basis,A,pts,k;multithreaded=multithreaded)
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

function solve_vect(solver::BoundaryIntegralMethod,billiard::Bi,basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPoints{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];multithreaded=multithreaded)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

