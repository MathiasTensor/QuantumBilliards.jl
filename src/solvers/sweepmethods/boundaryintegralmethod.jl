using LinearAlgebra, StaticArrays, TimerOutputs, Bessels

struct SymmetryRuleBIM{T<:Real}
    symmetry_type::Symbol         # :x, :y, :xy, or :nothing
    x_bc::Symbol                  # :D (Dirichlet) or :N (Neumann)
    y_bc::Symbol                  # :D (Dirichlet) or :N (Neumann)
    shift_x::T
    shift_y::T
end

struct BoundaryIntegralMethod{T} <: SweepSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 #for compatibiliy remove later
    min_pts::Int64
    rule::SymmetryRuleBIM
end

struct BoundaryPointsBIM{T} <: AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    curvature::Vector{T}
    ds::Vector{T}
end

struct AbstractHankelBasis <: AbsBasis end

function resize_basis(basis::Ba, billiard::Bi, dim::Int, k) where {Ba<:AbstractHankelBasis, Bi<:AbsBilliard}
    return AbstractHankelBasis()
end

function SymmetryRuleBIM(billiard::Bi; symmetries::Union{Vector{Any},Nothing}=nothing, x_bc=:D, y_bc=:D) where {Bi <: AbsBilliard}
    T = eltype([hasproperty(billiard, :x_axis) ? billiard.x_axis : 0.0, hasproperty(billiard, :y_axis) ? billiard.y_axis : 0.0])
    shift_x = hasproperty(billiard, :x_axis) ? billiard.x_axis : T(0.0)
    shift_y = hasproperty(billiard, :y_axis) ? billiard.y_axis : T(0.0)
    if hasproperty(billiard, :x_axis)
        shift_x = billiard.x_axis
    end
    if hasproperty(billiard, :y_axis)
        shift_y = billiard.y_axis
    end
    if isnothing(symmetries)
        return SymmetryRuleBIM{T}(:nothing,x_bc,y_bc,shift_x,shift_y)
    end
    if length(symmetries) > 1
        throw(ArgumentError("There should be only 1 symmetry"))
    end
    symmetries = symmetries[1]
    if symmetries isa Reflection
        if symmetries.axis == :y_axis
            return SymmetryRuleBIM{T}(:x,x_bc,y_bc,shift_x,shift_y)
        elseif symmetries.axis == :x_axis
            return SymmetryRuleBIM{T}(:y,x_bc,y_bc,shift_x,shift_y)
        elseif symmetries.axis == :origin
            return SymmetryRuleBIM{T}(:xy,x_bc,y_bc,shift_x,shift_y)
        else
            error("Unknown reflection axis: $(symmetries.axis)")
        end
    else
        error("Unsupported symmetry type: $(typeof(symmetries))")
    end
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}}, billiard::Bi; min_pts = 20, symmetries::Union{Vector{Any},Nothing}=nothing, x_bc=:D, y_bc=:D) where {T<:Real, Bi<:AbsBilliard}
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    sampler = [GaussLegendreNodes()]
    return BoundaryIntegralMethod{T}(1.0, bs, sampler, eps(T), min_pts, min_pts, SymmetryRuleBIM(billiard, symmetries=symmetries, x_bc=x_bc, y_bc=y_bc))
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector, billiard::Bi; min_pts = 20, symmetries::Union{Vector{Any},Nothing}=nothing, x_bc=:D, y_bc=:D) where {T<:Real, Bi<:AbsBilliard} 
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    return BoundaryIntegralMethod{T}(1.0, bs, samplers, eps(T), min_pts, min_pts, SymmetryRuleBIM(billiard, symmetries=symmetries, x_bc=x_bc, y_bc=y_bc))
end

function evaluate_points(solver::BoundaryIntegralMethod, billiard::Bi, k) where {Bi<:AbsBilliard}
    bs, samplers = adjust_scaling_and_samplers(solver, billiard)
    curves = billiard.fundamental_boundary
    type = eltype(solver.pts_scaling_factor)
    
    xy_all = Vector{SVector{2,type}}()
    normal_all = Vector{SVector{2,type}}()
    kappa_all = Vector{type}()
    w_all = Vector{type}()
   
    for i in eachindex(curves)
        crv = curves[i]
        if typeof(crv) <: AbsRealCurve
            L = crv.length
            N = max(solver.min_pts,round(Int, k*L*bs[i]/(2*pi)))
            sampler = samplers[i]
            if crv isa PolarSegment
                if sampler isa PolarSampler
                    t, dt = sample_points(sampler, crv, N)
                else
                    t, dt = sample_points(sampler, N)
                end
                s = arc_length(crv,t)
                ds = diff(s)
                append!(ds, L + s[1] - s[end]) # add the last difference as we have 1 less element. Add L to s[1] so we can logically subtract s[end]
            else
                t, dt = sample_points(sampler,N)
                ds = L.*dt
            end
            xy = curve(crv,t)
            normal = normal_vec(crv,t)
            kappa = curvature(crv,t)
            append!(xy_all, xy)
            append!(normal_all, normal)
            append!(kappa_all, kappa)
            append!(w_all, ds)
        end
    end
    return BoundaryPointsBIM{type}(xy_all,normal_all,kappa_all,w_all)
end

function apply_reflection(p::SVector{2, T}, rule::SymmetryRuleBIM{T}) where {T}
    shift_x, shift_y = rule.shift_x, rule.shift_y
    if rule.symmetry_type == :x
        return SVector(2*shift_x - p[1], p[2])
    elseif rule.symmetry_type == :y
        return SVector(p[1], 2*shift_y - p[2])
    elseif rule.symmetry_type == :xy
        return SVector(2*shift_x - p[1], 2*shift_y - p[2])
    else
        return p  # :nothing or no symmetry
    end
end

function compute_hankel(distance12::T, k::T) where {T<:Real}
    if abs(distance12::T) < eps(T) # Avoid division by zero
        return Complex(eps(T), eps(T))  
    end
    return Bessels.hankelh1(1, k * distance12::T)
end

function compute_cos_phi(dx12::T, dy12::T, normal1::SVector{2, T}, p1_curvature::T) where {T<:Real}
    distance12 = hypot(dx12, dy12)
    if distance12 < eps(T)
        return p1_curvature / (2.0 * π)
    else
        return (normal1[1] * dx12 + normal1[2] * dy12) / distance12
    end
end

# Currently not needed
function greens_function(distance12::T, k::T) where {T<:Real}
    if abs(distance12) < eps(T) # handle singularity
        return 0.0 + 0.0im  
    end
    return -im * k / 4 * Bessels.hankelh1(0, k * distance12)
end

function default_helmholtz_kernel(p1::SVector{2, T}, p2::SVector{2, T}, normal1::SVector{2, T}, k::T, p1_curvature::T) where {T<:Real}
    dx, dy = p1[1] - p2[1], p1[2] - p2[2]
    distance12 = hypot(dx, dy)
    return abs(distance12) < eps(T) ? Complex(1/(2*pi)*p1_curvature) : -im * k / 2.0 * compute_cos_phi(dx, dy, normal1, p1_curvature) * compute_hankel(distance12, k)
end

function compute_kernel_batch(p1::SVector{2, T}, p2_points::Vector{SVector{2, T}}, normal1::SVector{2, T}, curvature1::T, rule::SymmetryRuleBIM{T}, k::T) where {T<:Real}
    N = length(p2_points)
    kernel_values = Vector{Complex{T}}(undef, N)

    # Precompute reflected points if applicable (when :xy we need to always compute the :x and :y reflections)
    reflected_p2_x = rule.symmetry_type in [:x, :xy] ?
        [apply_reflection(p2, rule) for p2 in p2_points] : nothing
    reflected_p2_y = rule.symmetry_type in [:y, :xy] ?
        [apply_reflection(p2, SymmetryRuleBIM(:y, rule.x_bc, rule.y_bc, rule.shift_x, rule.shift_y)) for p2 in p2_points] : nothing
    reflected_p2_xy = rule.symmetry_type == :xy ?
        [apply_reflection(p2, rule) for p2 in p2_points] : nothing

    # Compute kernel values
    for j in 1:N
        p2 = p2_points[j]
        base_kernel = default_helmholtz_kernel(p1, p2, normal1, k, curvature1)
        kernel_value = base_kernel

        # Incorporate reflections
        if rule.symmetry_type == :x && reflected_p2_x !== nothing
            reflected_p2 = reflected_p2_x[j]
            if rule.x_bc == :D
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2, normal1, k, curvature1)
            elseif rule.x_bc == :N
                kernel_value += default_helmholtz_kernel(p1, reflected_p2, normal1, k, curvature1)
            end
        end

        if rule.symmetry_type == :y && reflected_p2_y !== nothing
            reflected_p2 = reflected_p2_y[j]
            if rule.y_bc == :D
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2, normal1, k, curvature1)
            elseif rule.y_bc == :N
                kernel_value += default_helmholtz_kernel(p1, reflected_p2, normal1, k, curvature1)
            end
        end

        if rule.symmetry_type == :xy && reflected_p2_xy !== nothing
            reflected_p2x = reflected_p2_x[j]
            reflected_p2y = reflected_p2_y[j]
            reflected_p2xy = reflected_p2_xy[j]
            if rule.x_bc == :D && rule.y_bc == :D
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2x, normal1, k, curvature1)
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2y, normal1, k, curvature1)
                kernel_value += default_helmholtz_kernel(p1, reflected_p2xy, normal1, k, curvature1)
            elseif rule.x_bc == :D && rule.y_bc == :N
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2x, normal1, k, curvature1)
                kernel_value += default_helmholtz_kernel(p1, reflected_p2y, normal1, k, curvature1)
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2xy, normal1, k, curvature1)
            elseif rule.x_bc == :N && rule.y_bc == :D
                kernel_value += default_helmholtz_kernel(p1, reflected_p2x, normal1, k, curvature1)
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2y, normal1, k, curvature1)
                kernel_value -= default_helmholtz_kernel(p1, reflected_p2xy, normal1, k, curvature1)
            elseif rule.x_bc == :N && rule.y_bc == :N
                kernel_value += default_helmholtz_kernel(p1, reflected_p2x, normal1, k, curvature1)
                kernel_value += default_helmholtz_kernel(p1, reflected_p2y, normal1, k, curvature1)
                kernel_value += default_helmholtz_kernel(p1, reflected_p2xy, normal1, k, curvature1)
            end
        end

        kernel_values[j] = kernel_value
    end
    return kernel_values
end

function fredholm_matrix(boundary_points::BoundaryPointsBIM{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T) where {T<:Real}
    xy_points = boundary_points.xy
    normals = boundary_points.normal
    curvatures = boundary_points.curvature
    ds = boundary_points.ds
    N = length(xy_points)

    # Initialize Fredholm matrix
    fredholm_matrix = Matrix{Complex{T}}(I, N, N)

    # Compute rows of the matrix
    for i in 1:N
        p1 = xy_points[i]
        normal1 = normals[i]
        curvature1 = curvatures[i]
        ds1 = ds[i]

        # Compute kernel for all p2 values for the current p1
        kernel_row = compute_kernel_batch(p1, xy_points, normal1, curvature1, symmetry_rule, k)

        # Update the matrix row
        fredholm_matrix[i, :] -= ds1 .* kernel_row
    end

    return fredholm_matrix
end


# high level

function construct_matrices(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM, k) where {Ba<:AbstractHankelBasis}
    return fredholm_matrix(pts, solver.rule, k)
end

function solve(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM, k) where {Ba<:AbstractHankelBasis}
    A = construct_matrices(solver, basis, pts, k)
    mu = svdvals(A)
    lam0 = mu[end]
    t = lam0
    return  t
end










#=
function evaluate_points(solver::BoundaryIntegralMethod, billiard::Bi, k) where {Bi<:AbsBilliard}
    bs, samplers = adjust_scaling_and_samplers(solver, billiard)
    curves = billiard.fundamental_boundary
    type = eltype(solver.pts_scaling_factor)
    
    xy_all = Vector{SVector{2,type}}()
    normal_all = Vector{SVector{2,type}}()
    kappa_all = Vector{type}()
    w_all = Vector{type}()
   
    for i in eachindex(curves)
        crv = curves[i]
        if typeof(crv) <: AbsRealCurve
            L = crv.length
            N = max(solver.min_pts,round(Int, k*L*bs[i]/(2*pi)))
            sampler = samplers[i]
            t, dt = sample_points(sampler,N)
            ds = L*dt #modify this
            xy = curve(crv,t)
            normal = normal_vec(crv,t)
            kappa = curvature(crv,t)
            append!(xy_all, xy)
            append!(normal_all, normal)
            append!(kappa_all, kappa)
            append!(w_all, ds)
        end
    end

    return BoundaryPointsBIM{type}(xy_all,normal_all,kappa_all,w_all)
end
=#

#=
function construct_matrices_benchmark(solver::DecompositionMethod, basis::Ba, pts::BoundaryPointsDM, k) where {Ba<:AbsBasis}
    to = TimerOutput()
    w = pts.w
    w_n = pts.w_n
    symmetries=basis.symmetries
    if ~isnothing(symmetries)
        norm = (length(symmetries)+1.0)
        w = w.*norm
        w_n = w_n.*norm
    end
    #basis and gradient matrices
    @timeit to "basis_and_gradient_matrices" B, dX, dY = basis_and_gradient_matrices(basis, k, pts.xy)
    N = basis.dim
    type = eltype(B)
    F = zeros(type,(N,N))
    G = similar(F)
    
    @timeit to "F construction" begin 
        @timeit to "weights" T = (w .* B) #reused later
        @timeit to "product" mul!(F,B',T) #boundary norm matrix
    end

    @timeit to "G construction" begin 
        @timeit to "normal derivative" nx = getindex.(pts.normal,1)
        @timeit to "normal derivative" ny = getindex.(pts.normal,2)
        #inplace modifications
        @timeit to "normal derivative" dX = nx .* dX 
        @timeit to "normal derivative" dY = ny .* dY
        #reuse B
        @timeit to "normal derivative" B = dX .+ dY
    
    #B is now normal derivative matrix (u function)
        @timeit to "weights" T = (w_n .* B) #apply integration weights
        @timeit to "product" mul!(G,B',T)#norm matrix
    end
    print_timer(to)
    return F, G    
end
=#

#=
function construct_matrices(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM, k) where {Ba<:AbsFundamentalBasis}
    #basis and gradient matrices
    symmetries = basis.symmetries
    xy = pts.xy
    w = pts.ds
    kappa = complex(pts.curvature)

    if ~isnothing(symmetries)
        norm = (length(symmetries)+1.0)
        w = w.*norm
    end
    
    dX, dY = greens_gradient(basis, k, xy, xy)#; return_diagonal=false)
    nx = getindex.(pts.normal,1)
    ny = getindex.(pts.normal,2)
    #inplace modifications
    dX = nx .* dX 
    dY = ny .* dY
    Q = @. -2.0 * dX + dY
    Q[diagind(Q)] = @. kappa/(2.0*pi)
    A = I - w .* Q #apply integrration weights and subtract from identity matrix
    return A   
end

=#


#=
function solve(solver::DecompositionMethod,F,G)
    #F, G = construct_matrices(solver, basis, pts, k)
    mu = generalized_eigvals(Symmetric(F),Symmetric(G);eps=solver.eps)
    lam0 = mu[end]
    t = 1.0/lam0
    return  t
end

function solve_vect(solver::DecompositionMethod,basis::AbsBasis, pts::BoundaryPointsDM, k)
    F, G = construct_matrices(solver, basis, pts, k)
    mu, Z, C = generalized_eigen(Symmetric(F),Symmetric(G);eps=solver.eps)
    x = Z[:,end]
    x = C*x #transform into original basis 
    lam0 = mu[end]
    t = 1.0/lam0
    return  t, x./sqrt(lam0)
end
=#