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
    struct BoundaryPointsBIM{T<:Real}

Represents the boundary points used in the method.

# Fields
- `xy::Vector{SVector{2,T}}`: Coordinates of the boundary points.
- `normal::Vector{SVector{2,T}}`: Normal vectors at the boundary points.
- `curvature::Vector{T}`: Curvatures at the boundary points.
- `ds::Vector{T}`: Arc lengths between consecutive boundary points.
- `shift_x::T`: x axis shift for the given geometry (contained in the billiard struct).
- `shift_y::T`: y axis shift for the given geometry (contained in the billiard struct).
"""
struct BoundaryPointsBIM{T} <: AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    curvature::Vector{T}
    ds::Vector{T}
    shift_x::T
    shift_y::T
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

"""
    BoundaryPointsMethod_to_BoundaryPoints(pts::BoundaryPointsBIM{T}) where {T<:Real}

Converts a `BoundaryPointsBIM` object to a `BoundaryPoints` object.

# Arguments
- `pts::BoundaryPointsBIM{T}`: An object containing:
  - `xy::Vector{SVector{2, T}}`: Coordinates of the boundary points.
  - `normal::Vector{SVector{2, T}}`: Normal vectors at the boundary points.
  - `ds::Vector{T}`: Integration weights (arc length differences between points).

# Returns
- `BoundaryPoints{T}`: An object containing:
  - `xy::Vector{SVector{2, T}}`: Coordinates of the boundary points.
  - `normal::Vector{SVector{2, T}}`: Normal vectors at the boundary points.
  - `s::Vector{T}`: Arc length coordinates (cumulative sum of `ds`).
  - `ds::Vector{T}`: diff(s).
"""
function BoundaryPointsMethod_to_BoundaryPoints(pts::BoundaryPointsBIM{T}) where {T<:Real}
    xy=pts.xy
    normal=pts.normal
    ds=pts.ds
    s=cumsum(ds)
    return BoundaryPoints{T}(xy,normal,s,ds)
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

"""
    evaluate_points(solver::BoundaryIntegralMethod, billiard::Bi, k::Real) -> BoundaryPointsBIM

Evaluates the boundary points and associated properties for the given solver and billiard.

# Arguments
- `solver::BoundaryIntegralMethod`: Boundary integral method configuration.
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `k::Real`: Wavenumber.

# Returns
- `BoundaryPointsBIM`: Evaluated boundary points and properties.
"""
function evaluate_points(solver::BoundaryIntegralMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=billiard.desymmetrized_full_boundary
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
    return BoundaryPointsBIM{type}(xy_all,normal_all,kappa_all,w_all,shift_x,shift_y)
end

#### NEW MATRIX CODE, SLIGHTLY FASTER UTILIZING THE DEFAULT KERNEL'S FUNCTION HANKEL FUNCTION SYMMETRY ####

"""
    default_helmholtz_kernel_matrix(bp::BoundaryPointsBIM{T}, k::T) -> Matrix{Complex{T}}

Computes the Helmholtz kernel matrix for the given boundary points using the matrix-based approach.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: A matrix where each element corresponds to the Helmholtz kernel between boundary points, incorporating curvature for singular cases.
"""
function default_helmholtz_kernel_matrix(bp::BoundaryPointsBIM{T},k::T;multithreaded::Bool=true) where {T<:Real}
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
    default_helmholtz_kernel_matrix(bp_s::BoundaryPointsBIM{T}, xy_t::Vector{SVector{2, T}}, k::T) -> Matrix{Complex{T}}

Computes the Helmholtz kernel matrix for interactions between source boundary points and a target point set.

# Arguments
- `bp_s::BoundaryPointsBIM{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `xy_t::Vector{SVector{2, T}}`: Target points for evaluating the kernel matrix.
- `k::T`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: A matrix where each element corresponds to the Helmholtz kernel between source and target points, incorporating curvature for singular cases.
"""
function default_helmholtz_kernel_matrix(bp_s::BoundaryPointsBIM{T},xy_t::Vector{SVector{2,T}},k::T;multithreaded::Bool=true) where {T<:Real}
    xy_s=bp_s.xy
    normals=bp_s.normal
    curvatures=bp_s.curvature
    N=length(xy_s)
    M=Matrix{Complex{T}}(undef,N,N)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    x_s=getindex.(xy_s,1)
    y_s=getindex.(xy_s,2)
    x_t=getindex.(xy_t,1)
    y_t=getindex.(xy_t,2)
    tol=eps(T)
    pref=Complex{T}(zero(T),-k/2) # -im*k/2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=x_s[i];yi=y_s[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:N
            dx=xi-x_t[j];dy=yi-y_t[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            if d<tol
                M[i,j]=Complex(curvatures[i]/TWO_PI)
            else
                invd=inv(d)
                cos_phi=(nxi*dx+nyi*dy)*invd
                hankel=pref*Bessels.hankelh1(1,k*d)
                M[i,j]=cos_phi*hankel
            end
        end
    end
    filter_matrix!(M)
    return M
end

"""
    compute_kernel_matrix(bp::BoundaryPointsBIM{T}, k::T; kernel_fun::Union{Symbol, Function}=:default) -> Matrix{Complex{T}}

Computes the kernel matrix for the given boundary points using the specified kernel function w/ NO symmetry.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use. Defaults to `:default` (Helmholtz kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The computed kernel matrix.
"""
function compute_kernel_matrix(bp::BoundaryPointsBIM{T},k::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {T<:Real}
    if kernel_fun==:default
        return default_helmholtz_kernel_matrix(bp,k;multithreaded=multithreaded)
    else
        return kernel_fun(bp,k)
    end
end

"""
    @inline function add_pair_default!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::T,tol2::T,pref::Complex{T};scale::T=one(T)) where {T<:Real} -> Bool

Compute and add the default Helmholtz double-layer contribution for the pair (i,j).
Writes scalars directly into `M` (both M[i,j] and M[j,i] if i≠j). Returns `true`
if the pair was non-singular (distance² > tol2), `false` otherwise.
"""
@inline function add_pair_default!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::T,tol2::T,pref::Complex{T};scale::T=one(T)) where {T<:Real}
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
    @inline add_pair_custom!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::T,kernel_fun;scale::T=one(T)) where {T<:AbstractFloat} -> Bool

Like `add_pair_default!` but uses a user-supplied kernel evaluator:
`kernel_fun(i,j, xi,yi,nxi,nyi, xj,yj,nxj,nyj, k) :: Complex`.
"""
@inline function add_pair_custom!(M::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::T,kernel_fun;scale::T=one(T)) where {T<:Real}
    val_ij=kernel_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)*scale
    @inbounds begin
        M[i,j]+=val_ij
    end
    return true
end

# X,Y Reflections over potentially shifted axis (depeneding of the shift_x/shift_y of the billiard geometry)
@inline _x_reflect(x::T,sx::T) where {T<:Real}=(2*sx-x)
@inline _y_reflect(y::T,sy::T) where {T<:Real}=(2*sy-y)

"""
    compute_kernel_matrix(bp::BoundaryPointsBIM{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T; kernel_fun::Union{Symbol, Function}=:default) -> Matrix{Complex{T}}

Computes the kernel matrix for the given boundary points with symmetry reflections applied.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points structure containing the source points, normal vectors, and curvatures.
- `symmetry::Vector{Any}`: Symmetry to apply.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use, must have the signature: `kernel_fun(i,j, xi,yi,nxi,nyi, xj,yj,nxj,nyj, k) :: Complex`. Defaults to `:default` (Helmholtz kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The computed kernel matrix with symmetry reflections applied.
"""
function compute_kernel_matrix(bp::BoundaryPointsBIM{T},symmetry::Vector{Any},k::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    nrm=bp.normal
    κ=bp.curvature 
    N=length(xy)
    K=zeros(Complex{T},N,N)
    tol2=(eps(T))^2
    pref=Complex{T}(0,-k/2)
    add_x=false;add_y=false;add_xy=false # true if the symmetry is present
    sxgn=one(T);sygn=one(T);sxy=one(T) # the scalings +/- depending on the symmetry considerations
    shift_x=bp.shift_x;shift_y=bp.shift_y # the reflection axes shifts from billiard geometry
    @inbounds for s in symmetry # symmetry here is always != nothing
        if s.axis==:y_axis;add_x=true;sxgn=(s.parity==-1 ? -one(T) : one(T)); end
        if s.axis==:x_axis;add_y=true;sygn=(s.parity==-1 ? -one(T) : one(T)); end
        if s.axis==:origin
            add_x=true;add_y=true;add_xy=true
            sxgn=(s.parity[1]==-1 ? -one(T) : one(T))
            sygn=(s.parity[2]==-1 ? -one(T) : one(T))
            sxy=sxgn*sygn
        end
    end
    isdef=(kernel_fun===:default) # we only have predefined the default kernel's add_pair! matrix builders. For other kernels it can be different and built with add_pair_custom!
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xy[i][1]; yi=xy[i][2]; nxi=nrm[i][1]; nyi=nrm[i][2]
        @inbounds for j in 1:N # since it has non-trivial symmetry we have to do both loops over all indices, not just the upper triangular
            xj=xy[j][1];yj=xy[j][2];nxj=nrm[j][1];nyj=nrm[j][2]
            if isdef
                ok=add_pair_default!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,tol2,pref)
                if !ok; K[i,j]+=Complex(κ[i]/TWO_PI); end
            else
                add_pair_custom!(K,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k,kernel_fun)
            end
            if add_x # reflect only over the x axis
                xr=_x_reflect(xj,shift_x);yr=yj
                isdef ? add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxgn) :
                        add_pair_custom!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,kernel_fun;scale=sxgn)
            end
            if add_y # reflect only over the y axis
                xr=xj;yr=_y_reflect(yj,shift_y)
                isdef ? add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sygn) :
                        add_pair_custom!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,kernel_fun;scale=sygn)
            end
            if add_xy # reflect over both the axes
                xr=_x_reflect(xj,shift_x);yr=_y_reflect(yj,shift_y)
                isdef ? add_pair_default!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,tol2,pref;scale=sxy) :
                        add_pair_custom!(K,i,j,xi,yi,nxi,nyi,xr,yr,nxj,nyj,k,kernel_fun;scale=sxy)
            end
        end
    end
    return K
end

"""
    fredholm_matrix(bp::BoundaryPointsBIM{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T; kernel_fun::Union{Symbol, Function}=:default) -> Matrix{Complex{T}}

Constructs the Fredholm matrix for the boundary integral method using the computed kernel matrix.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points structure containing the source points, normal vectors, curvatures, and differential arc lengths.
- `symmetry::Vector{Any}`: Symmetry to apply.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use, must have the signature `kernel_fun(i,j, xi,yi,nxi,nyi, xj,yj,nxj,nyj, k) :: Complex`. Defaults to `:default` (Helmholtz kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The constructed Fredholm matrix, incorporating differential arc lengths and symmetry reflections.
"""
function fredholm_matrix(bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},k::T;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {T<:Real}
    K=isnothing(symmetry) ?
        compute_kernel_matrix(bp,k;kernel_fun=kernel_fun,multithreaded=multithreaded) :
        compute_kernel_matrix(bp,symmetry,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
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
    construct_matrices(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM, k::T; kernel_fun::Union{Symbol, Function}=:default) -> Matrix{Complex{T}}

Constructs the Fredholm matrix using the solver, basis, and boundary points for the boundary integral method.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: Boundary points structure containing source points, normal vectors, curvatures, and differential arc lengths.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use. Defaults to `:default` (Helmholtz kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: The constructed Fredholm matrix.
"""
function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    return fredholm_matrix(pts,solver.symmetry,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
end

"""
    solve(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM{T}, k::T; kernel_fun::Union{Symbol, Function}=:default) -> T

Computes the smallest singular value of the Fredholm matrix for a given configuration.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: Boundary points structure containing source points, normal vectors, curvatures, and differential arc lengths.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use. Defaults to `:default` (Helmholtz kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `T`: The smallest singular value of the Fredholm matrix.
"""
function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    mu=svdvals(A) # Arpack's version of svd for computing only the smallest singular value should be better but is non-reentrant
    return mu[end]
end

# INTERNAL BENCHMARKS
function solve_INFO(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    s_constr=time()
    @info "constructing Fredholm matrix A..."
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    @info "Condition number of A for svd: $(cond(A))"
    e_constr=time()
    @info "SVD..."
    s_svd=time()
    mu=svdvals(A)
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

"""
    solve_vect(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM{T}, k::T; kernel_fun::Union{Symbol, Function}=:default) -> Tuple{T, Vector{T}}

Computes the smallest singular value and its corresponding singular vector for the Fredholm matrix.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: Boundary points structure containing source points, normal vectors, curvatures, and differential arc lengths.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use. Defaults to `:default` (Helmholtz kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Tuple{T, Vector{T}}`: A tuple containing:
  - `T`: The smallest singular value of the Fredholm matrix.
  - `Vector{T}`: The corresponding singular vector.
"""
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A) # do NOT use svd with DivideAndConquer() here b/c singular matrix!!!
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=Vt[idx,:]
    u_mu=real.(u_mu)
    return mu,u_mu
end

"""
    solve_eigenvectors_BIM(solver::BoundaryIntegralMethod, billiard::Bi, basis::Ba, ks::Vector{T}; kernel_fun=default_helmholtz_kernel_matrix) -> Tuple{Vector{Vector{T}}, Vector{BoundaryPointsBIM}}

Computes the eigenvectors of the boundary integral method for a range of wave numbers.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `basis::Ba<:AbstractHankelBasis`: The basis function used for solving the eigenvalue problem.
- `ks::Vector{T}`: A vector of wave numbers `k` for which to compute the eigenvectors.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use. Defaults to `:default` (Helmholtz kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Tuple{Vector{Vector{T}}, Vector{BoundaryPointsBIM}}`:
  - `Vector{Vector{T}}`: A vector containing the eigenvectors for each wave number in `ks`.
  - `Vector{BoundaryPointsBIM}`: A vector of `BoundaryPointsBIM` objects, representing the boundary points used for each wave number in `ks`.
"""
function solve_eigenvectors_BIM(solver::BoundaryIntegralMethod,billiard::Bi,basis::Ba,ks::Vector{T};kernel_fun=:default,multithreaded::Bool=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPointsBIM{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];kernel_fun=kernel_fun,multithreaded=multithreaded)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

#### BENCHMARKS ####

@timeit TO "evaluate_points" function evaluate_points_timed(solver::BoundaryIntegralMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    return evaluate_points(solver,billiard,k)
end

@timeit TO "construct_matrices" function construct_matrices_timed(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=:default,multithreaded::Bool=true) where {Ba<:AbsBasis}
    return construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
end

@timeit TO "SVD" function svdvals_timed(A)
    return svdvals(A)
end

function solve_timed(solver::BoundaryIntegralMethod,billiard::Bi,k::Real;kernel_fun=:default,multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts=evaluate_points_timed(solver,billiard,k)
    A=construct_matrices_timed(solver,AbstractHankelBasis(),pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    σs=svdvals_timed(A)
    show(TO)
    reset_timer!(TO)
end

### HELPERS FOR FINDING THE PEAKS ###

"""
    find_peaks(x::Vector{T}, y::Vector{T}; threshold=200.0) where {T<:Real}

Finds the x-coordinates of local maxima in the `y` vector that are greater than the specified `threshold`.

# Arguments
- `x::Vector{T}`: The x-coordinates corresponding to the y-values.
- `y::Vector{T}`: The y-values to search for peaks.
- `threshold::Union{T, Vector{T}}`: Minimum value a peak must exceed to be considered.
  Can be a scalar (applied to all peaks) or a vector (element-wise comparison).
  Default is 200.0.

# Returns
- `Vector{T}`: A vector of x-coordinates where peaks are located.
"""
function find_peaks(x::Vector{T}, y::Vector{T}; threshold::Union{T,Vector{T}}=200.0) where {T<:Real}
    peaks=T[]
    threshold_vec=length(threshold)==1 ? fill(threshold,length(x)) : threshold
    for i in 2:length(y)-1
        if y[i]>y[i-1] && y[i]>y[i+1] && y[i]>threshold_vec[i]
            push!(peaks,x[i])
        end
    end
    return peaks
end

"""
    bim_second_derivative(x::Vector{T}, y::Vector{T}) where {T<:Real}

Computes the second derivative of `y` with respect to `x` using finite differences between the xs in `x`.

# Arguments
- `x::Vector{T}`: The x-coordinates of the data.
- `y::Vector{T}`: The y-values of the data.

# Returns
- `Vector{T}`: Midpoints of the x-values for the second derivative.
- `Vector{T}`: The second derivative of `y` with respect to `x`.
"""
function bim_second_derivative(x::Vector{T}, y::Vector{T}) where {T<:Real}
    first_grad=diff(y)./diff(x)
    first_mid_x=@. (x[1:end-1]+x[2:end])/2
    second_grad=diff(first_grad)./diff(first_mid_x)
    second_mid_x=@. (first_mid_x[1:end-1]+first_mid_x[2:end])/2
    return second_mid_x,second_grad
end

"""
    get_eigenvalues(k_range::Vector{T}, tens::Vector{T}; threshold=200.0) where {T<:Real}

Finds peaks in the second derivative of the logarithm of `tens` with respect to `k_range`. These peaks are as precise as the k step that was chosen in `k_range`.

# Arguments
- `k_range::Vector{T}`: The range of `k` values.
- `tens::Vector{T}`: The tension values.
- `threshold::Real`: Minimum value a peak in the second derivative gradient must exceed. Default is 200.0.

# Returns
- `Vector{T}`: The `k_range` values where peaks in the second derivative gradient are found.
"""
function get_eigenvalues(k_range::Vector{T}, tens::Vector{T}; threshold=200.0) where {T<:Real}
    mid_x,gradient=bim_second_derivative(k_range,log10.(tens))
    return find_peaks(mid_x,gradient;threshold=threshold)
end
