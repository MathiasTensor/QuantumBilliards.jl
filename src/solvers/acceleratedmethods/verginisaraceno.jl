
mutable struct VerginiSaracenoSolver{T} <: AcceleratedSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end

function VerginiSaracenoSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim = 100, min_pts = 500) where T<:Real 
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    sampler = [GaussLegendreNodes()]
    timer = TimerOutput()
return VerginiSaracenoSolver(d, bs, sampler, eps(T), min_dim, min_pts)
end

function VerginiSaracenoSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{AbsSampler}; min_dim = 100, min_pts = 500) where {T<:Real} 
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    timer = TimerOutput()
    return VerginiSaracenoSolver(d, bs, samplers, eps(T), min_dim, min_pts)
end

function evaluate_points(solver::VerginiSaracenoSolver,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves = get_boundary_curves(billiard)
    type = eltype(solver.pts_scaling_factor)
    Ns = _determine_bp_sizes(curves, bs, k)
    M = length(Ns)
    xy_all = Vector{Vector{SVector{2,type}}}(undef, M)
    w_all = Vector{Vector{type}}(undef, M)

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
        w = ds ./ rn
        xy_all[i] = xy
        w_all[i] = w       
    end
    return BoundaryPoints(vcat(xy_all...);w_vs = vcat(w_all...))
end


# Your function with timing and debug info
function construct_matrices(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k; 
                            multithreaded = true) where {Ba<:AbsBasis}    
    @timeit_debug "construct_matrices" begin
        xy = pts.xy
        w = pts.w_vs
        N = basis.dim
        M = length(xy)
        nsym = one(eltype(w)) * (isnothing(basis.symmetries) ? 1 : length(basis.symmetries) + 1)
        
        @debug "Matrix construction started" N M k nsym
        
        # Compute basis matrix G
        @timeit_debug "basis_matrix" begin
            G = basis_matrix(basis, k, xy; multithreaded)
        end
        @debug "Basis matrix computed" size=size(G) 
        
        # Compute derivative matrix dG
        @timeit_debug "dk_matrix" begin
            dG = dk_matrix(basis, k, xy; multithreaded)
        end
        @debug "Derivative matrix computed" size=size(dG)
        
        # Compute F = G' * W * G
        @timeit_debug "compute_F" begin
            _scale_rows_sqrtw!(G, w, nsym)
            F = Matrix{eltype(G)}(undef, N, N)
            @blas_multi MAX_BLAS_THREADS BLAS.syrk!('U', 'T', one(eltype(G)), G, zero(eltype(G)), F)
            _symmetrize_from_upper!(F)
        end
        @debug "F computed" size=size(F)
        
        # Compute Fk = G' * W * dG + dG' * W * G
        @timeit_debug "compute_Fk" begin
            _scale_rows_sqrtw!(dG, w, nsym)
            Fk = Matrix{eltype(G)}(undef, N, N)
            @blas_multi_then_1 MAX_BLAS_THREADS BLAS.syr2k!('U', 'T', one(eltype(G)), G, dG, zero(eltype(G)), Fk)
            _symmetrize_from_upper!(Fk)
        end
        @debug "Fk computed" size=size(Fk) 
        
        return F, Fk
    end
end

function sm_results(mu,k)
    ks = k .- 2 ./mu .+ 2/k ./(mu.^2) 
    ten = 2.0 .*(2.0 ./ mu).^2
    return ks, ten
end

#=
function sm_vects_results(mu,k)
    ks = k .- 2 ./mu .+ 2/k ./(mu.^2) 
    ten = 2.0 .*(2.0 ./ mu).^2
    #does not sort the results
    return ks, ten
end
=#
function solve(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k, dk; multithreaded = true) where {Ba<:AbsBasis}
    F, Fk = construct_matrices(solver, basis, pts, k; multithreaded)
    mu = generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    p = sortperm(ks)
    return ks[p], ten[p]
end

function solve(solver::VerginiSaracenoSolver,F,Fk, k, dk)
    #F, Fk = construct_matrices(solver, basis, pts, k)
    @blas_multi_then_1 MAX_BLAS_THREADS mu = generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    p = sortperm(ks)
    return ks[p], ten[p]
end

function solve_vectors(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k, dk; multithreaded = true) where {Ba<:AbsBasis}
    F, Fk = construct_matrices(solver, basis, pts, k; multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mu, Z, C = generalized_eigen(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    Z = Z[:,idx]
    X = C*Z #transform into original basis 
    X = (sqrt.(ten))' .* X
    p = sortperm(ks)
    return  ks[p], ten[p], X[:,p]
end

