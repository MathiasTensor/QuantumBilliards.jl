using LinearAlgebra, StaticArrays, TimerOutputs, Bessels

"""
    struct SymmetryRuleBIM{T<:Real}

Represents symmetry rules for the boundary integral method.

# Fields
- `symmetry_type::Symbol`: Type of symmetry (:x, :y, :xy, or :nothing).
- `x_bc::Symbol`: Boundary condition on the x-axis (:D for Dirichlet, :N for Neumann).
- `y_bc::Symbol`: Boundary condition on the y-axis (:D for Dirichlet, :N for Neumann).
- `shift_x::T`: Shift along the x-axis.
- `shift_y::T`: Shift along the y-axis.
"""
struct SymmetryRuleBIM{T<:Real}
    symmetry_type::Symbol        
    x_bc::Symbol                  
    y_bc::Symbol                
    shift_x::T
    shift_y::T
end


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
- `rule::SymmetryRuleBIM`: Symmetry rule for the configuration.
"""
struct BoundaryIntegralMethod{T} <: SweepSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 
    min_pts::Int64
    rule::SymmetryRuleBIM
end

"""
    struct BoundaryPointsBIM{T<:Real}

Represents the boundary points used in the method.

# Fields
- `xy::Vector{SVector{2,T}}`: Coordinates of the boundary points.
- `normal::Vector{SVector{2,T}}`: Normal vectors at the boundary points.
- `curvature::Vector{T}`: Curvatures at the boundary points.
- `ds::Vector{T}`: Arc lengths between consecutive boundary points.
"""
struct BoundaryPointsBIM{T} <: AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    curvature::Vector{T}
    ds::Vector{T}
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
    SymmetryRuleBIM(billiard::Bi; symmetries=Nothing, x_bc=:D, y_bc=:D) -> SymmetryRuleBIM

Constructs a `SymmetryRuleBIM` based on the billiard's properties and symmetry configuration.

# Arguments
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `symmetries::Union{Vector{Any},Nothing}`: Symmetry definitions (optional).
- `x_bc::Symbol`: Boundary condition on the x-axis (:D for Dirichlet, :N for Neumann).
- `y_bc::Symbol`: Boundary condition on the y-axis (:D for Dirichlet, :N for Neumann).

# Returns
- `SymmetryRuleBIM`: Constructed symmetry rule.
"""
function SymmetryRuleBIM(billiard::Bi;symmetries::Union{Vector{Any},Nothing}=nothing,x_bc=:D,y_bc=:D) where {Bi <: AbsBilliard}
    T = eltype([hasproperty(billiard,:x_axis) ? billiard.x_axis : 0.0, hasproperty(billiard,:y_axis) ? billiard.y_axis : 0.0])
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    if hasproperty(billiard,:x_axis)
        shift_x=billiard.x_axis
    end
    if hasproperty(billiard,:y_axis)
        shift_y=billiard.y_axis
    end
    if isnothing(symmetries)
        return SymmetryRuleBIM{T}(:nothing,x_bc,y_bc,shift_x,shift_y)
    end
    if length(symmetries)>1
        throw(ArgumentError("There should be only 1 symmetry"))
    end
    symmetries=symmetries[1]
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

"""
    BoundaryIntegralMethod(pts_scaling_factor, billiard::Bi; min_pts=20, symmetries=Nothing, x_bc=:D, y_bc=:D) -> BoundaryIntegralMethod

Creates a boundary integral method solver configuration.

# Arguments
- `pts_scaling_factor::Union{T,Vector{T}}`: Scaling factors for the boundary points.
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `min_pts::Int`: Minimum number of boundary points (default: 20).
- `symmetries::Union{Vector{Any},Nothing}`: Symmetry definitions (optional).
- `x_bc::Symbol`: Boundary condition on the x-axis (:D for Dirichlet, :N for Neumann).
- `y_bc::Symbol`: Boundary condition on the y-axis (:D for Dirichlet, :N for Neumann).

# Returns
- `BoundaryIntegralMethod`: Constructed solver configuration.
"""
function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,symmetries::Union{Vector{Any},Nothing}=nothing,x_bc=:D,y_bc=:D) where {T<:Real, Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[GaussLegendreNodes()]
    return BoundaryIntegralMethod{T}(1.0,bs,sampler,eps(T),min_pts,min_pts,SymmetryRuleBIM(billiard,symmetries=symmetries,x_bc=x_bc,y_bc=y_bc))
end

function BoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},samplers::Vector,billiard::Bi;min_pts = 20,symmetries::Union{Vector{Any},Nothing}=nothing, x_bc=:D,y_bc=:D) where {T<:Real, Bi<:AbsBilliard} 
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return BoundaryIntegralMethod{T}(1.0,bs,samplers,eps(T),min_pts,min_pts,SymmetryRuleBIM(billiard,symmetries=symmetries,x_bc=x_bc,y_bc=y_bc))
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
    bs,samplers=adjust_scaling_and_samplers(solver, billiard)
    curves=billiard.fundamental_boundary
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
    return BoundaryPointsBIM{type}(xy_all,normal_all,kappa_all,w_all)
end

"""
    apply_reflection(p::SVector{2,T}, rule::SymmetryRuleBIM{T}) -> SVector{2,T}

Applies symmetry reflection rules to a point.

# Arguments
- `p::SVector{2,T}`: Original point.
- `rule::SymmetryRuleBIM{T}`: Symmetry rule to apply.

# Returns
- `SVector{2,T}`: Reflected point.
"""
function apply_reflection(p::SVector{2,T},rule::SymmetryRuleBIM{T}) where {T}
    shift_x, shift_y = rule.shift_x, rule.shift_y
    if rule.symmetry_type==:x
        return SVector(2*shift_x-p[1],p[2])
    elseif rule.symmetry_type==:y
        return SVector(p[1],2*shift_y-p[2])
    elseif rule.symmetry_type==:xy
        return SVector(2*shift_x-p[1],2*shift_y-p[2])
    else
        return p
    end
end

"""
    compute_hankel(distance12::T, k::T) -> Complex{T}

Computes the Hankel function of the first kind for the given distance and wavenumber.

# Arguments
- `distance12::T`: Distance between two points.
- `k::T`: Wavenumber.

# Returns
- `Complex{T}`: Hankel function value. Handles singularity by returning a small complex value.
"""
function compute_hankel(distance12::T,k::T) where {T<:Real}
    if abs(distance12::T)<eps(T) # Avoid division by zero
        return Complex(eps(T),eps(T))  
    end
    return Bessels.hankelh1(1,k*distance12::T)
end

"""
    compute_cos_phi(dx12::T, dy12::T, normal1::SVector{2,T}, p1_curvature::T) -> T

Computes the cosine of the angle φ between the normal vector and the vector connecting two points.

# Arguments
- `dx12::T`: x-component of the vector between two points.
- `dy12::T`: y-component of the vector between two points.
- `normal1::SVector{2,T}`: Normal vector at the first point.
- `p1_curvature::T`: Curvature at the first point.

# Returns
- `T`: Computed cos(φ) value. Handles singularity by using the curvature term.
"""
function compute_cos_phi(dx12::T,dy12::T,normal1::SVector{2,T},p1_curvature::T) where {T<:Real}
    distance12=hypot(dx12,dy12)
    if distance12<eps(T)
        return p1_curvature/(2.0*π)
    else
        return (normal1[1]*dx12+normal1[2]*dy12)/distance12
    end
end

"""
    greens_function(distance12::T, k::T) -> Complex{T}

Computes the Green's function for the given distance and wavenumber.

# Arguments
- `distance12::T`: Distance between two points.
- `k::T`: Wavenumber.

# Returns
- `Complex{T}`: Green's function value. Handles singularity by suppressing the term.
"""
function greens_function(distance12::T, k::T) where {T<:Real}
    if abs(distance12)<eps(T) # handle singularity, this is supressed by the cosphi term
        return 0.0+0.0im  
    end
    return -im*k/4*Bessels.hankelh1(0,k*distance12)
end

"""
    default_helmholtz_kernel(p1::SVector{2,T}, p2::SVector{2,T}, normal1::SVector{2,T}, k::T, p1_curvature::T) -> Complex{T}

Computes the Helmholtz kernel for the given points and properties.

# Arguments
- `p1::SVector{2,T}`: First point.
- `p2::SVector{2,T}`: Second point.
- `normal1::SVector{2,T}`: Normal vector at the first point.
- `k::T`: Wavenumber.
- `p1_curvature::T`: Curvature at the first point.

# Returns
- `Complex{T}`: Computed Helmholtz kernel. Handles singularity using curvature.
"""
function default_helmholtz_kernel(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T},k::T,p1_curvature::T) where {T<:Real}
    dx, dy = p1[1]-p2[1], p1[2]-p2[2]
    distance12=hypot(dx,dy)
    return abs(distance12)<eps(T) ? Complex(1/(2*pi)*p1_curvature) : -im*k/2.0*compute_cos_phi(dx,dy,normal1,p1_curvature)*compute_hankel(distance12,k)
end

"""
    compute_kernel(
        p1::SVector{2,T}, p2::SVector{2,T}, 
        normal1::SVector{2,T}, curvature1::T, 
        reflected_p2_x::Union{SVector{2,T}, Nothing}, 
        reflected_p2_y::Union{SVector{2,T}, Nothing}, 
        reflected_p2_xy::Union{SVector{2,T}, Nothing}, 
        rule::SymmetryRuleBIM{T}, k::T
    ) -> Complex{T}

Computes the kernel value for a given pair of points, incorporating symmetry reflections.

# Arguments
- `p1::SVector{2,T}`: First point.
- `p2::SVector{2,T}`: Second point.
- `normal1::SVector{2,T}`: Normal vector at the first point.
- `curvature1::T`: Curvature at the first point.
- `reflected_p2_x::Union{SVector{2,T}, Nothing}`: Reflected second point across the x-axis (if applicable).
- `reflected_p2_y::Union{SVector{2,T}, Nothing}`: Reflected second point across the y-axis (if applicable).
- `reflected_p2_xy::Union{SVector{2,T}, Nothing}`: Reflected second point across both axes (if applicable).
- `rule::SymmetryRuleBIM{T}`: Symmetry rule to apply.
- `k::T`: Wavenumber.

# Returns
- `Complex{T}`: Computed kernel value.
"""
function compute_kernel(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T}, curvature1::T,reflected_p2_x::Union{SVector{2,T}, Nothing},reflected_p2_y::Union{SVector{2,T}, Nothing},reflected_p2_xy::Union{SVector{2,T}, Nothing},rule::SymmetryRuleBIM{T}, k::T) where {T<:Real}
    kernel_value=default_helmholtz_kernel(p1,p2,normal1,k,curvature1) # Base kernel computation
    if !isnothing(reflected_p2_x) # Handle x-reflection
        if rule.x_bc==:D
            kernel_value-=default_helmholtz_kernel(p1,reflected_p2_x,normal1,k,curvature1)
        elseif rule.x_bc==:N
            kernel_value+=default_helmholtz_kernel(p1,reflected_p2_x,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_y) # Handle y-reflection
        if rule.y_bc==:D
            kernel_value-=default_helmholtz_kernel(p1,reflected_p2_y,normal1,k,curvature1)
        elseif rule.y_bc==:N
            kernel_value+=default_helmholtz_kernel(p1,reflected_p2_y,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_xy) # Handle xy-reflection
        if rule.x_bc==:D && rule.y_bc==:D
            kernel_value+=default_helmholtz_kernel(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:D && rule.y_bc==:N
            kernel_value-=default_helmholtz_kernel(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:D
            kernel_value-=default_helmholtz_kernel(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:N
            kernel_value+=default_helmholtz_kernel(p1,reflected_p2_xy,normal1,k,curvature1)
        end
    end
    return kernel_value
end

"""
    fredholm_matrix(boundary_points::BoundaryPointsBIM, symmetry_rule::SymmetryRuleBIM, k::Real) -> Matrix{Complex{T}}

Constructs the Fredholm matrix for the boundary integral method.

# Arguments
- `boundary_points::BoundaryPointsBIM`: Evaluated boundary points.
- `symmetry_rule::SymmetryRuleBIM`: Symmetry rule for the boundary points.
- `k::Real`: Wavenumber.

# Returns
- `Matrix{Complex{T}}`: Constructed Fredholm matrix.
"""
function fredholm_matrix(boundary_points::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T) where {T<:Real}
    xy_points=boundary_points.xy
    normals=boundary_points.normal
    curvatures=boundary_points.curvature
    ds=boundary_points.ds
    N=length(xy_points)
    reflected_points_x = symmetry_rule.symmetry_type in [:x,:xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:x,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_y = symmetry_rule.symmetry_type in [:y,:xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:y,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_xy = symmetry_rule.symmetry_type==:xy ?
        [apply_reflection(p,symmetry_rule) for p in xy_points] : nothing
    fredholm_matrix = Matrix{Complex{T}}(I,N,N)
    Threads.@threads for i in 1:N
        p1=xy_points[i]
        normal1=normals[i]
        curvature1=curvatures[i]
        ds1=ds[i]
        for j in 1:N
            p2=xy_points[j]
            kernel_value=compute_kernel(
                p1,p2,normal1,curvature1,
                isnothing(reflected_points_x) ? nothing : reflected_points_x[j],
                isnothing(reflected_points_y) ? nothing : reflected_points_y[j],
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k)
            fredholm_matrix[i, j]-=ds1*kernel_value
        end
    end
    return fredholm_matrix
end

# high level

function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM, k) where {Ba<:AbstractHankelBasis}
    return fredholm_matrix(pts,solver.rule,k)
end

function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM, k) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k)
    mu=svdvals(A)
    return mu[end]
end

"""
    create_fredholm_movie!(
        k_range::Vector{T}, billiard::Bi; 
        symmetries::Union{Vector{Any}, Nothing} = nothing, 
        b::T = 15.0, 
        sampler = [GaussLegendreNodes()], 
        output_path::String = "fredholm_movie.mp4"
    ) where {T <: Real, Bi <: AbsBilliard}

Creates an animated movie of Fredholm matrices over a range of `k` values for a given billiard geometry. USE ONLY FOR A NAROW k RANGE SINCE MATRIX DIM DOES NOT YET CHANGE.

# Arguments
- `k_range::Vector{T}`: A vector of wavenumbers (`k`) over which the Fredholm matrix is computed.
- `billiard::Bi`: The billiard geometry (must be a subtype of `AbsBilliard`).
- `symmetries::Union{Vector{Any}, Nothing}`: Optional symmetries to apply to the billiard. Default is `nothing`.
- `b::T`: Scaling factor for the boundary integral method. Default is `15.0`.
- `sampler`: Sampling strategy for the boundary points. Default is `[GaussLegendreNodes()]`.
- `output_path::String`: Path to save the generated animation file. Default is `"fredholm_movie.mp4"`.

# Returns
- None.
"""
function create_fredholm_movie!(k_range::Vector{T}, billiard::Bi; symmetries::Union{Vector{Any},Nothing}=nothing, b=15.0, sampler=[GaussLegendreNodes()], output_path::String="fredholm_movie.mp4") where {T<:Real,Bi<:AbsBilliard}
    symmetryBIM=QuantumBilliards.SymmetryRuleBIM(billiard; symmetries=symmetries)
    sample_k=first(k_range)
    bim_solver = QuantumBilliards.BoundaryIntegralMethod(b,sampler,billiard;symmetries=symmetries)
    pts=QuantumBilliards.evaluate_points(bim_solver,billiard,sample_k)
    fredholm_sample=QuantumBilliards.fredholm_matrix(pts,symmetryBIM,sample_k)
    fig=Figure(resolution =(1500, 1500))
    ax_real=Axis(fig[1,1][1,1],title="real(Fredholm) over k",xlabel="Index (i)",ylabel="Index (j)",aspect=DataAspect())
    ax_imag=Axis(fig[1,1][1,2],title="imag(Fredholm) over k",xlabel="Index (i)",ylabel="Index (j)",aspect=DataAspect())
    heatmap_real=real.(fredholm_sample)
    heatmap_imag=imag.(fredholm_sample)
    heatmap_plot_real=heatmap!(ax_real,heatmap_real)
    heatmap_plot_imag=heatmap!(ax_imag,heatmap_imag)
    Colorbar(fig[1,1][1,1][1,2],heatmap_plot_real)
    Colorbar(fig[1,1][1,2][1,2],heatmap_plot_imag)
    record(fig,output_path,k_range;framerate=30) do k
        pts=QuantumBilliards.evaluate_points(bim_solver,billiard,k)
        fredholm=QuantumBilliards.fredholm_matrix(pts, symmetryBIM, k)
        heatmap_plot_real[1]=real.(fredholm)  # Update 
        heatmap_plot_imag[1]=imag.(fredholm)  # Update
        ax_real.title="real at k = $(round(k,digits=4))" 
        ax_imag.title="imag at k = $(round(k,digits=4))"
    end
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












#### TESTING #####

function test_cos_phi_matrix(solver::BoundaryIntegralMethod, billiard::Bi; k=50) where {Bi<:AbsBilliard}
    # Evaluate boundary points, normals, and curvatures
    boundary_points = evaluate_points(solver,billiard,k)

    # Extract boundary information
    xy_points = boundary_points.xy
    normals = boundary_points.normal
    curvatures = boundary_points.curvature
    n = length(xy_points)

    # Initialize cosPhi matrix
    cos_phi_matrix = Matrix{Float64}(undef, n, n)

    # Compute cosPhi for all point pairs
    for i in 1:n
        for j in 1:n
            p1, p2 = xy_points[i], xy_points[j]
            dx, dy = p1[1] - p2[1], p1[2] - p2[2]
            cos_phi_matrix[i, j] = compute_cos_phi(dx, dy, normals[i], curvatures[i])
        end
    end

    # Visualize the cosPhi matrix
    fig = Figure(resolution=(800, 800))
    ax = Axis(fig[1, 1], title="cos(ϕ) Matrix for Quarter-Circle Billiard, diag=$(cos_phi_matrix[1,1])", xlabel="Boundary Point Index (p2)", ylabel="Boundary Point Index (p1)")
    hm = heatmap!(ax, cos_phi_matrix; colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="cos(ϕ)")
    return fig
end


function test_hankel(solver::BoundaryIntegralMethod, billiard::Bi; k=50) where {Bi<:AbsBilliard}
    # Evaluate points on the billiard's boundary
    boundary_points = evaluate_points(solver, billiard, k)
    xy_points = boundary_points.xy
    N = length(xy_points)

    # Create the Hankel matrix
    hankel_matrix = zeros(ComplexF64, N, N)
    for i in 1:N
        for j in 1:N
            dx = xy_points[i][1] - xy_points[j][1]
            dy = xy_points[i][2] - xy_points[j][2]
            distance = hypot(dx, dy)
            hankel_matrix[i, j] = abs(distance) < eps(Float64) ? 0.0 + 0.0im : Bessels.hankelh1(0, k * distance)
        end
    end

    # Plot the magnitude of the Hankel matrix
    fig = Figure()
    ax = Axis(fig[1,1][1,1],
        title = "|H₀⁽¹⁾(k·distance)|",
        xlabel = "Boundary Point Index (p2)",
        ylabel = "Boundary Point Index (p1)"
    )
    hmap=heatmap!(ax, abs.(hankel_matrix))
    Colorbar(fig[1,1][1,2], hmap)
    return fig
end

function test_reflections(
    boundary_points::BoundaryPointsBIM{T},
    symmetry_rule::SymmetryRuleBIM{T}
) where {T<:Real}
    original_points = boundary_points.xy

    # Compute reflected points
    reflected_x = symmetry_rule.symmetry_type in [:x, :xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:x, symmetry_rule.x_bc, symmetry_rule.y_bc, symmetry_rule.shift_x, symmetry_rule.shift_y)) for p in original_points] : []
    reflected_y = symmetry_rule.symmetry_type in [:y, :xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:y, symmetry_rule.x_bc, symmetry_rule.y_bc, symmetry_rule.shift_x, symmetry_rule.shift_y)) for p in original_points] : []
    reflected_xy = symmetry_rule.symmetry_type == :xy ?
        [apply_reflection(p, symmetry_rule) for p in original_points] : []

    # Plot original and reflected points
    fig = Figure(resolution=(800, 800))
    ax = Axis(fig[1, 1], aspect=1)
    
    # Original points
    scatter!(ax, [p[1] for p in original_points], [p[2] for p in original_points], label="Original Points", color=:blue)

    # Reflected across x-axis
    if !isempty(reflected_x)
        println("reflected x is not empy")
        scatter!(ax, [p[1] for p in reflected_x], [p[2] for p in reflected_x], label="Reflected (x-axis)", color=:red)
    end

    # Reflected across y-axis
    if !isempty(reflected_y)
        println("reflected y is not empty")
        scatter!(ax, [p[1] for p in reflected_y], [p[2] for p in reflected_y], label="Reflected (y-axis)", color=:green)
    end

    # Reflected across both axes
    if !isempty(reflected_xy)
        println("reflected xy is not empty")
        scatter!(ax, [p[1] for p in reflected_xy], [p[2] for p in reflected_xy], label="Reflected (xy-axis)", color=:purple)
    end

    axislegend(ax, position=:rt)
    return fig
end