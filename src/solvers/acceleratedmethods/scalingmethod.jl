
abstract type AbsScalingMethod <: AcceleratedSolver 
end
struct ScalingMethodA{T} <: AbsScalingMethod where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end


function ScalingMethodA(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim = 100, min_pts = 500) where T<:Real 
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    sampler = [GaussLegendreNodes()]
return ScalingMethodA(d, bs, sampler, eps(T), min_dim, min_pts)
end

function ScalingMethodA(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{AbsSampler}; min_dim = 100, min_pts = 500) where {T<:Real} 
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    return ScalingMethodA(d, bs, samplers, eps(T), min_dim, min_pts)
end

struct BoundaryPointsSM{T} <: AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}}
    w::Vector{T}
end

function evaluate_points(solver::AbsScalingMethod, billiard::Bi, k) where {Bi<:AbsBilliard}
    bs, samplers = adjust_scaling_and_samplers(solver, billiard)
    curves = get_boundary_curves(billiard)
    type = eltype(solver.pts_scaling_factor)
    xy_all = Vector{SVector{2,type}}()
    w_all = Vector{type}()
    
    for i in eachindex(curves)
        crv = curves[i]
        L = crv.length
        N = max(solver.min_pts,round(Int, k*L*bs[i]/(2*pi)))
        sampler = samplers[i]
        t, dt = sample_points(sampler, N)
        
        ds = L*dt #this needs modification!!!
        xy = curve(crv,t)
        g = domain_gradient_vector(crv, xy)
        normal =  g./norm(g)
        rn = dot.(xy, normal)
        w = ds ./ rn
        append!(xy_all, xy)
        append!(w_all, w)
        
    end
    return BoundaryPointsSM{type}(xy_all, w_all)
end

function construct_matrices(solver::ScalingMethodA, basis::Ba, pts::BoundaryPointsSM, k; multithreaded = true) where {Ba<:AbsBasis}
    xy = pts.xy
    w = pts.w
    symmetries=basis.symmetries
    if ~isnothing(symmetries)
        n = (length(symmetries)+1.0)
        w = w.*n
    end
    N = basis.dim
    #basis matrix
    B = basis_matrix(basis, k, xy; multithreaded)
    type = eltype(B)
    F = zeros(type,(N,N))
    Fk = similar(F)
    T = (w .* B) #reused later
    mul!(F,B',T) #boundary norm matrix
    #reuse B
    B = dk_matrix(basis,k, xy; multithreaded)
    mul!(Fk,B',T) #B is now derivative matrix
    #symmetrize matrix
    Fk = Fk + Fk' 
    return F, Fk    
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
function solve(solver::AbsScalingMethod, basis::Ba, pts::BoundaryPointsSM, k, dk; multithreaded = true) where {Ba<:AbsBasis}
    F, Fk = construct_matrices(solver, basis, pts, k; multithreaded)
    mu = generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    p = sortperm(ks)
    return ks[p], ten[p]
end

function solve(solver::AbsScalingMethod,F,Fk, k, dk)
    #F, Fk = construct_matrices(solver, basis, pts, k)
    mu = generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    p = sortperm(ks)
    return ks[p], ten[p]
end

function solve_vectors(solver::AbsScalingMethod, basis::Ba, pts::BoundaryPointsSM, k, dk; multithreaded = true) where {Ba<:AbsBasis}
    F, Fk = construct_matrices(solver, basis, pts, k; multithreaded)
    mu, Z, C = generalized_eigen(Symmetric(F),Symmetric(Fk);eps=solver.eps)
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

