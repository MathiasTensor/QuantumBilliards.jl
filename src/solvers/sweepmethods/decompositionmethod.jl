
struct DecompositionMethodSolver{T} <: SweepSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end


function DecompositionMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim = 100, min_pts = 500) where T<:Real 
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    sampler = [GaussLegendreNodes()]
return DecompositionMethodSolver(d, bs, sampler, eps(T), min_dim, min_pts)
end

function DecompositionMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{AbsSampler}; min_dim = 100, min_pts = 500) where {T<:Real} 
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    return DecompositionMethodSolver(d, bs, samplers, eps(T), min_dim, min_pts)
end


function evaluate_points(solver::DecompositionMethodSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    bs, samplers = adjust_scaling_and_samplers(solver, billiard)
    curves = get_boundary_curves(billiard)
    type = eltype(solver.pts_scaling_factor)
    Ns = _determine_bp_sizes(curves, bs, k)
    M = length(Ns)
    xy_all = Vector{Vector{SVector{2,type}}}(undef, M)
    normal_all = Vector{Vector{SVector{2,type}}}(undef, M)
    ds_all = Vector{Vector{type}}(undef, M)
    w_n_all = Vector{Vector{type}}(undef, M)

    for i in eachindex(curves)
        crv = curves[i]
        L = crv.length
        sampler = samplers[i]
        t, dt = sample_points(sampler, Ns[i])
        ds = L*dt #this needs modification!!!
        xy = curve(crv,t)
        normal = domain_gradient_vector(crv, xy)
        normal .= normal./norm(normal)
        rn = dot.(xy, normal)
        xy_all[i] = xy
        normal_all[i] = normal
        ds_all[i] = ds  
        w_n_all[i] =(ds.*rn)./(2.0*k.^2)         
    end
    return BoundaryPoints(vcat(xy_all...);normal = vcat(normal_all...),  w_dm = vcat(w_n_all...), ds = vcat(ds_all...))
end


function construct_matrices(solver::DecompositionMethodSolver,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    @timeit_debug "construct_matrices" begin
        xy=pts.xy
        w=pts.ds
        wn=pts.w_dm
        N=basis.dim
        M = length(xy)
        nsym=isnothing(basis.symmetries) ? one(eltype(w)) : one(eltype(w))*(length(basis.symmetries)+1)

        @debug "Matrix construction started" N M k nsym
        @timeit_debug "basis_and_gradient_matrices" begin
            # the alogrithm consctructs B and the normal derivative Bn with syrk to minimize the allocation cost. It does this with the trick of putting sqrt(w_n) into both the rows of B and the rows of B' so that we can use syrk on sqrt(W)*B to get B'*(W*B) without forming W*B as a temporary matrix (posible b/c W is diagonal)
            @blas_1 B,dX,dY=basis_and_gradient_matrices(basis,k,xy;multithreaded)
            # Form F = B'*(W*B) by inplace scaling the rows of B by sqrt(w) (inplace to B) and use syrk to perform the Rank-k update of a symmetric matrix
            _scale_rows_sqrtw!(B,w,nsym) #  trick of putting sqrt(w_n) into the rows of the transposed and original B to get (sqrt(W)*B)' * (sqrt(W)*B) so we can use syrk on sqrt(W)*B
        end
        @debug "Basis and gradient matrix computed" size=size(B) 
        @timeit_debug "compute_F" begin
            F=Matrix{eltype(B)}(undef,N,N) # preallocate F
            @blas_multi_then_1 MAX_BLAS_THREADS BLAS.syrk!('U','T',one(eltype(B)),B,zero(eltype(B)),F) # F[u ∈ upper]+=1.0*B'*B, no need to fill(F,0) since the additive constant in C is 0
            _symmetrize_from_upper!(F) # since we chose "U" in syrk, we need to mirror upper -> lower
            # Build Bn into dX: dX <- nx*dX + ny*dY 
            _build_Bn_inplace!(dX,dY,pts.normal)
            # Form G = Bn'*(Wn*Bn) by first scaling the rows of Bn (dX) by sqrt(w_n) (inplace to dX) and use syrk to perform the Rank-k update of a symmetric matrix
            _scale_rows_sqrtw!(dX,wn,nsym) # like for F form sqrt(Wn*Bn) with row scaling with the same trick of putting sqrt(w_n) into the rows of the transposed and original dX to get (sqrt(Wn)*Bn)' * (sqrt(Wn)*Bn) so we can use syrk on dX
        end
        @debug "F computed" size=size(F)
        @timeit_debug "compute_G" begin
            G=Matrix{eltype(B)}(undef,N,N) # preallocate G, no need to fill with zeros since we use zero(eltype(B)) for the additive constant in syrk
            @blas_multi_then_1 MAX_BLAS_THREADS BLAS.syrk!('U','T',one(eltype(B)),dX,zero(eltype(B)),G) # G[u ∈ upper]+=1.0*dX'*dX where dX is now sqrt(Wn)*Bn due to inplace scalings above
            _symmetrize_from_upper!(G) # since we chose "U" in syrk, we need to mirror upper -> lower
        end
        @debug "G computed" size=size(F)
        return F,G
    end    
end


function solve(solver::DecompositionMethodSolver,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    F,G=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eigvals(Symmetric(F),Symmetric(G);eps=solver.eps)
    lam0=mu[end]
    t=1.0/lam0
    return t
end

function solve(solver::DecompositionMethodSolver,F,G)
    @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eigvals(Symmetric(F),Symmetric(G);eps=solver.eps)
    lam0=mu[end]
    t=1.0/lam0
    return t
end

function solve_vect(solver::DecompositionMethodSolver,basis::AbsBasis,pts::BoundaryPoints,k;multithreaded::Bool=true)
    F,G=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @blas_multi MAX_BLAS_THREADS mu,Z,C=generalized_eigen(Symmetric(F),Symmetric(G);eps=solver.eps)
    x=Z[:,end]
    @blas_multi_then_1 MAX_BLAS_THREADS x=C*x #transform into original basis, BLAS mul
    lam0=mu[end]
    t=1.0/lam0
    return t,x./sqrt(lam0)
end
