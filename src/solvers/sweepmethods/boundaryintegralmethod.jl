using LinearAlgebra, StaticArrays, TimerOutputs, Bessels
const TO=TimerOutput()
const TWO_PI=2*pi

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
- `symmetry::Union{Vector{Any},Nothing}`: Symmetry for the configuration.
"""
struct BoundaryIntegralMethod{T} <: SweepSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 
    min_pts::Int64
    symmetry::Union{Vector{Any},Nothing}
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
function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,symmetry::Union{Vector{Any},Nothing}=nothing) where {T<:Real, Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return BoundaryIntegralMethod{T}(1.0,bs,sampler,eps(T),min_pts,min_pts,symmetry)
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},samplers::Vector,billiard::Bi;min_pts=20,symmetry::Union{Vector{Any},Nothing}=nothing) where {T<:Real, Bi<:AbsBilliard} 
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return BoundaryIntegralMethod{T}(1.0,bs,samplers,eps(T),min_pts,min_pts,symmetry)
end

function _boundary_curves_for_solver(billiard::Bi,solver::BoundaryIntegralMethod) where {Bi<:AbsBilliard}
    if solver.symmetry === nothing
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
            N=max(solver.min_pts,round(Int,k*L*bs[i]/(2*pi))) #TODO preallocate cheap for loop for non-dynamic allocation
            sampler=samplers[i]
            if crv isa PolarSegment
                if sampler isa PolarSampler
                    t,dt=sample_points(sampler,crv,N)
                else
                    t,dt=sample_points(sampler,N)
                end
                s=arc_length(crv,t)
                ds=diff(s)
                append!(ds,L+s[1]-s[end]) # add the last difference as we have 1 less element. Add L to s[1] so we can logically subtract s[end]
            else
                t,dt=sample_points(sampler,N)
                ds=L.*dt
            end
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

#### NEW MATRIX CODE, SLIGHTLY FASTER UTILIZING THE DEFAULT KERNEL'S FUNCTION HANKEL FUNCTION SYMMETRY ####

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
    pref=Complex{T}(zero(T),-k/2) # -im*k/2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i # symmetric hankel part
            dx=xi-xs[j];dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy)) # an efficient dy^2+dx*dx
            if d<tol
                M[i,j]=Complex(curvatures[i]/TWO_PI)
            else
                invd=inv(d)
                cos_phi=(nxi*dx+nyi*dy)*invd
                hankel=pref*Bessels.hankelh1(1,k*d)
                M[i,j]=cos_phi*hankel
            end
            if i!=j
                cos_phi_symmetric=(nx[j]*(-dx)+ny[j]*(-dy))*invd # Hankel is symmetric, but cos_phi is not; compute explicitly for M[j, i]
                M[j,i]=cos_phi_symmetric*hankel
            end
        end
    end
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
        M[i,j]+=scale*((nxi*dx+nyi*dy)*invd)*h
    end
    return true
end

"""
    compute_kernel_matrix(bp::BoundaryPoints{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T; multithreaded::Bool=true) -> Matrix{Complex{T}}

Computes the kernel matrix for the given boundary points with symmetry reflections applied.

# Arguments
- `bp::BoundaryPoints{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `symmetry::Vector{Any}`: Symmetry to apply.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The computed kernel matrix with symmetry reflections applied.
"""
function compute_kernel_matrix(bp::BoundaryPoints{T},symmetry::Vector{Any},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature
    N=length(xy)
    K=zeros(Complex{T},N,N)
    tol2=(eps(T))^2
    pref=Complex{T}(0,-k/2)
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
    @inbounds for s in symmetry
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
                K[i,j]+=Complex(κ[i]/TWO_PI)
            end
            if add_x
                xr=_x_reflect(xj,shift_x)
                yr=yj
                _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxgn)
            end
            if add_y
                xr=xj
                yr=_y_reflect(yj,shift_y)
                _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sygn)
            end
            if add_xy
                xr=_x_reflect(xj,shift_x)
                yr=_y_reflect(yj,shift_y)
                _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxy)
            end
            if have_rot
                @inbounds for l in 1:nrot-1
                    cl=ctab[l+1]
                    sl=stab[l+1]
                    xr,yr=_rot_point(xj,yj,cx,cy,cl,sl)
                    phase=χ[l+1]
                    _add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=phase)
                end
            end
        end
    end
    return K
end

"""
    fredholm_matrix(bp::BoundaryPoints{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T; multithreaded::Bool=true) -> Matrix{Complex{T}}

Constructs the Fredholm matrix for the boundary integral method using the computed kernel matrix.

# Arguments
- `bp::BoundaryPoints{T}`: Boundary points structure containing the source points, normal vectors, curvatures, and differential arc lengths.
- `symmetry::Vector{Any}`: Symmetry to apply.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The constructed Fredholm matrix, incorporating differential arc lengths and symmetry reflections.
"""
function fredholm_matrix(bp::BoundaryPoints{T},symmetry::Union{Vector{Any},Nothing},k::T;multithreaded::Bool=true) where {T<:Real}
    K=isnothing(symmetry) ?
        compute_kernel_matrix(bp,k;multithreaded=multithreaded) :
        compute_kernel_matrix(bp,symmetry,k;multithreaded=multithreaded)
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
    construct_matrices(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPoints, k::T) -> Matrix{Complex{T}}

Constructs the Fredholm matrix using the solver, basis, and boundary points for the boundary integral method.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPoints{T}`: Boundary points structure containing source points, normal vectors, curvatures, and differential arc lengths.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The constructed Fredholm matrix.
"""
function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    return @blas_1 fredholm_matrix(pts,solver.symmetry,k;multithreaded=multithreaded)
end

function solve_full(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mu=svdvals(A) # Arpack's version of svd for computing only the smallest singular value should be better but is non-reentrant
    return mu[end]
end

"""
    solve(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPoints{T}, k::T; multithreaded::Bool=true, use_krylov::Bool=true) -> T

Computes the smallest singular value of the Fredholm matrix for a given configuration.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPoints{T}`: Boundary points structure containing source points, normal vectors, curvatures, and differential arc lengths.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `use_krylov::Bool=true`: Large speedups in singular value/vector calculation. If anomalies in result are present set this flag to `False`.

# Returns
- `T`: The smallest singular value of the Fredholm matrix.
"""
function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true,use_krylov::Bool=true) where {Ba<:AbstractHankelBasis}
    if use_krylov
        return solve_krylov(solver,basis,pts,k,multithreaded=multithreaded)
    else
        return solve_full(solver,basis,pts,k,multithreaded=multithreaded)
    end
end

# INTERNAL BENCHMARKS
function solve_full_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    s_constr=time()
    @info "constructing Fredholm matrix A..."
    A=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @info "Condition number of A for svd: $(cond(A))"
    e_constr=time()
    @info "SVD..."
    s_svd=time()
    @blas_multi_then_1 MAX_BLAS_THREADS mu=svdvals(A)
    e_svd=time()
    total_time=(e_svd-s_svd)+(e_constr-s_constr)
    @info "Total solve time for test k: $(total_time)"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("Fredholm matrix A construction: $(100*(e_constr-s_constr)/total_time) %")
    println("SVD: $(100*(e_svd-s_svd)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return mu[end]
end

function solve_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true,use_krylov::Bool=true) where {Ba<:AbstractHankelBasis}
    if use_krylov
        return solve_krylov_INFO(solver,basis,pts,k,multithreaded=multithreaded)
    else
        return solve_full_INFO(solver,basis,pts,k,multithreaded=multithreaded)
    end
end

function solve_vect_full(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A) # do NOT use svd with DivideAndConquer() here b/c singular matrix!!!
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=Vt[idx,:]
    u_mu=real.(u_mu)
    return mu,u_mu
end

"""
    solve_vect(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPoints{T}, k::T; multithreaded::Bool=true, use_krylov::Bool=true) -> Tuple{T, Vector{T}}

Computes the smallest singular value and its corresponding singular vector for the Fredholm matrix.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPoints{T}`: Boundary points structure containing source points, normal vectors, curvatures, and differential arc lengths.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `use_krylov::Bool=true`: Large speedups in singular value/vector calculation. If anomalies in result are present set this flag to `False`.

# Returns
- `Tuple{T, Vector{T}}`: A tuple containing:
  - `T`: The smallest singular value of the Fredholm matrix.
  - `Vector{T}`: The corresponding singular vector.
"""
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true,use_krylov::Bool=true) where {Ba<:AbstractHankelBasis}
    if use_krylov
        return solve_vect_krylov(solver,basis,pts,k,multithreaded=multithreaded)
    else
        return solve_vect_full(solver,basis,pts,k,multithreaded=multithreaded)
    end
    
end

"""
    solve_eigenvectors_BIM(solver::BoundaryIntegralMethod, billiard::Bi, basis::Ba, ks::Vector{T}; multithreaded::Bool=true, use_krylov::Bool=true) -> Tuple{Vector{Vector{T}}, Vector{BoundaryPoints}}

Computes the eigenvectors of the boundary integral method for a range of wave numbers.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `basis::Ba<:AbstractHankelBasis`: The basis function used for solving the eigenvalue problem.
- `ks::Vector{T}`: A vector of wave numbers `k` for which to compute the eigenvectors.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `use_krylov::Bool=true`: Large speedups in singular value/vector calculation. If anomalies in result are present set this flag to `False`.

# Returns
- `Tuple{Vector{Vector{T}}, Vector{BoundaryPoints}}`:
  - `Vector{Vector{T}}`: A vector containing the eigenvectors for each wave number in `ks`.
  - `Vector{BoundaryPoints}`: A vector of `BoundaryPoints` objects, representing the boundary points used for each wave number in `ks`.
"""
function solve_eigenvectors_BIM(solver::BoundaryIntegralMethod,billiard::Bi,basis::Ba,ks::Vector{T};multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPoints{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];multithreaded=multithreaded,use_krylov=use_krylov)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

#########################################################
#### CONSTRUCTORS FOR COMPLEX ks - FOR BEYN's METHOD ####
#########################################################

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
    tol2=(eps(T))^2;pref=-im*k/2
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
function compute_kernel_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},symmetry::Vector{Any},k::Complex{T};multithreaded::Bool=true) where {T<:Real}
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
    @use_threads multithreading=multithreaded for i in 1:N # make if instead of elseif since can have >1 symmetry
        xi=xy[i][1]; yi=xy[i][2]; nxi=nrm[i][1]; nyi=nrm[i][2] # i is the target, j is the source
        @inbounds for j in 1:N # since it has non-trivial symmetry we have to do both loops over all indices, not just the upper triangular
            xj=xy[j][1];yj=xy[j][2];nxj=nrm[j][1];nyj=nrm[j][2]
            ok=_add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref)
            if !ok; K[i,j]+=Complex(κ[i]/TWO_PI); end
            if add_x # reflect only over the x axis
                xr=_x_reflect(xj,shift_x);yr=yj
                _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxgn)
            end
            if add_y # reflect only over the y axis
                xr=xj;yr=_y_reflect(yj,shift_y)
                _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sygn)
                end
            end
            if add_xy # reflect over both the axes
                xr=_x_reflect(xj,shift_x);yr=_y_reflect(yj,shift_y)
                _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxy)
            end
            if have_rot
                @inbounds for l in 1:nrot-1 # l=0 is the direct term we already added; add l=1..nrot-1
                    cl=ctab[l+1];sl=stab[l+1]
                    xr,yr=_rot_point(xj,yj,cx,cy,cl,sl)
                    phase=χ[l+1]  # e^{i 2π m l / n}, rotations due to being 1d-irreps have real characters
                    _add_pair_default_complex!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=phase)
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
function fredholm_matrix_complex_k!(K::Matrix{Complex{T}},bp::BoundaryPoints{T},symmetry::Union{Vector{Any},Nothing},k::Complex{T};multithreaded::Bool=true) where {T<:Real}
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
