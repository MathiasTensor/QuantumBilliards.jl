using LinearAlgebra, StaticArrays, TimerOutputs, Bessels

#### REGULAR BIM ####

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

### HELPERS ###

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
    SymmetryRuleBIM_to_Symmetry(rule::SymmetryRuleBIM) :: Union{Nothing, Vector{Any}}

Converts a `SymmetryRuleBIM` object into the corresponding symmetry transformation(s).

# Arguments
- `rule::SymmetryRuleBIM`: The symmetry rule containing the symmetry type and boundary conditions.

# Returns
- `nothing`: If there is no symmetry.
- `Vector{Any}`: A vector of symmetry transformations (e.g., `XReflection`, `YReflection`, `XYReflection`).

# Errors
Raises an error for unsupported boundary conditions or unknown symmetry types.
"""
function SymmetryRuleBIM_to_Symmetry(rule::SymmetryRuleBIM)
    if rule.symmetry_type==:nothing
        return nothing
    elseif rule.symmetry_type==:x
        if rule.x_bc==:D
            return Vector{Any}([XReflection(-1)])
        elseif rule.x_bc==:N
            return Vector{Any}([XReflection(1)])
        else
            error("Unsupported boundary condition: $(rule.x_bc)")
        end
    elseif rule.symmetry_type==:y
        if rule.y_bc==:D
            return Vector{Any}([YReflection(-1)])
        elseif rule.y_bc==:N
            return Vector{Any}([YReflection(1)])
        else
            error("Unsupported boundary condition: $(rule.x_bc)")
        end
    elseif rule.symmetry_type==:xy
        if rule.x_bc==:D && rule.y_bc==:D
            return Vector{Any}([XYReflection(-1,-1)])
        elseif rule.x_bc ==:N && rule.y_bc==:D
            return Vector{Any}([XYReflection(1,-1)])
        elseif rule.x_bc==:D && rule.y_bc==:N
            return Vector{Any}([XYReflection(-1,1)])
        elseif rule.x_bc==:N && rule.y_bc==:N
            return Vector{Any}([XYReflection(1,1)])
        else
            error("Unsupported boundary condition: $(rule.x_bc)")
        end
    else
        error("Unknown symmetry type: $(rule.symmetry_type)")
    end
end

"""
    Symmetry_to_SymmetryBIM(symmetry::Union{Vector{Any}, Nothing}, billiard::Bi) -> SymmetryRuleBIM

Converts a symmetry object into a `SymmetryRuleBIM`.

# Arguments
- `symmetry::Union{Vector{Any}, Nothing}`: A vector containing a single symmetry (e.g., a `Reflection`) or `nothing` if no symmetry is provided.
- `billiard::Bi`: The billiard configuration, a subtype of `AbsBilliard`.

# Returns
- `SymmetryRuleBIM{T}`: A `SymmetryRuleBIM` object encoding the symmetry type, boundary conditions, and axis shifts for the billiard.

# Errors
- Throws an error if the symmetry vector contains more than one element.
- Throws an error for unsupported symmetry types, axes, or parity values.
"""
function Symmetry_to_SymmetryBIM(symmetry::Union{Vector{Any},Nothing},billiard::Bi) where {Bi<:AbsBilliard}
    T = eltype([hasproperty(billiard,:x_axis) ? billiard.x_axis : 0.0, hasproperty(billiard,:y_axis) ? billiard.y_axis : 0.0])
    shift_x=hasproperty(billiard,:x_axis) ? billiard.x_axis : T(0.0)
    shift_y=hasproperty(billiard,:y_axis) ? billiard.y_axis : T(0.0)
    if isnothing(symmetry)
        return SymmetryRuleBIM{T}(:nothing,:D,:D,0.0,0.0)
    end
    if length(symmetry)>1
        error("Only 1 symmetry supported")
    end
    if !(symmetry[1] isa Reflection)
        error("Unsupported symmetry type: $(typeof(sym))")
    end
    if symmetry[1].axis==:y_axis
        if symmetry[1].parity==-1
            return SymmetryRuleBIM(:x,:D,:D,shift_x,shift_y)
        elseif symmetry[1].parity==1
            return SymmetryRuleBIM(:x,:N,:D,shift_x,shift_y)
        else
            error("Unsupported symmetry parity: $(symmetry[1].parity)")
        end
    elseif symmetry[1].axis==:x_axis
        if symmetry[1].parity==-1
            return SymmetryRuleBIM(:y,:D,:D,shift_x,shift_y)
        elseif symmetry[1].parity==1
            return SymmetryRuleBIM(:y,:D,:N,shift_x,shift_y)
        end
    elseif symmetry[1].axis==:origin
        if symmetry[1].parity==[-1,-1]
            return SymmetryRuleBIM(:xy,:D,:D,shift_x,shift_y)
        elseif symmetry[1].parity==[-1,1]
            return SymmetryRuleBIM(:xy,:D,:N,shift_x,shift_y)
        elseif symmetry[1].parity==[1,-1]
            return SymmetryRuleBIM(:xy,:N,:D,shift_x,shift_y)
        elseif symmetry[1].parity==[1,1]
            return SymmetryRuleBIM(:xy,:N,:N,shift_x,shift_y)
        else
            error("Unsupported symmetry parity: $(symmetry[1].parity)")
        end
    else
        error("Unknown symmetry axis: $(symmetry[1].axis)")
    end
end

"""
    BoundaryPointsBIM_to_BoundaryPoints(pts::BoundaryPointsBIM{T}) where {T<:Real}

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
function BoundaryPointsBIM_to_BoundaryPoints(pts::BoundaryPointsBIM{T}) where {T<:Real}
    xy=pts.xy
    normal=pts.normal
    ds=pts.ds
    s=cumsum(ds)
    return BoundaryPoints{T}(xy,normal,s,ds)
end

### STANDARD ###

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

#### BIM - MAIN

"""
    construct_matrices(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM{T}, k::Real) -> Matrix{Complex{T}}

Constructs the Fredholm matrix for the given boundary integral method, basis, and boundary points.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: The boundary points structure with type `T`.
- `k::Real`: The wave number.

# Returns
- `Matrix{Complex{T}}`: The constructed Fredholm matrix.
"""
function construct_matrices(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM, k) where {Ba<:AbstractHankelBasis}
    return fredholm_matrix(pts,solver.rule,k)
end

"""
    solve(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM{T}, k::Real) -> T

Computes the smallest singular value of the Fredholm matrix.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: The boundary points structure with type `T`.
- `k::Real`: The wave number.

# Returns
- `T`: The smallest singular value of the matrix.
"""
function solve(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM, k) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k)
    mu=svdvals(A)
    return mu[end]
end

"""
    solve_vect(solver::BoundaryIntegralMethod, basis::Ba, pts::BoundaryPointsBIM{T}, k::Real) -> Tuple{T, Vector{T}}

Computes the smallest singular value and its corresponding singular vector.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: The boundary points structure with type `T`.
- `k::Real`: The wave number.

# Returns
- `Tuple{T, Vector{T}}`: A tuple containing the smallest singular value and the corresponding singular vector.
"""
function solve_vect(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k)
    F=svd(A)
    mu=F.S[end]
    u_mu=F.Vt[end,:]  # Last row of Vt corresponds to smallest singular value
    u_mu=real.(u_mu)
    u_mu.=ifelse.(abs.(u_mu).<eps(eltype(u_mu)),zero(eltype(u_mu)),u_mu)
    return mu,u_mu
end

"""
    solve_eigenvectors_BIM(solver::BoundaryIntegralMethod, basis::Ba, ks::Vector) -> Tuple{Vector{Vector{T}}, Vector{BoundaryPointsBIM}} where {Ba<:AbstractHankelBasis, T<:Real}

Solve for the eigenvectors of the boundary integral method (BIM) for a range of wave numbers `ks`. These wave numbers should be the actual eigenvalues of the billiard since no check is done in the function if the smallest singular value for that k in ks is really the locally smallest one.

# Arguments
- `solver::BoundaryIntegralMethod`: The boundary integral method solver used to compute the eigenvectors.
- `billiard::Bi`: Billiard configuration (subtype of `AbsBilliard`).
- `basis::Ba<:AbstractHankelBasis`: The basis functions used for solving the eigenvalue problem.
- `ks::Vector{T}`: A vector of wave numbers `k` for which to compute the eigenvectors.

# Returns
- `Vector{Vector{T}}`: A vector of eigenvectors, one for each wave number in `ks`.
- `Vector{BoundaryPointsBIM}`: A vector of `BoundaryPointsBIM` objects, containing the boundary points used for each wave number in `ks`.
"""
function solve_eigenvectors_BIM(solver::BoundaryIntegralMethod,billiard::Bi,basis::Ba,ks::Vector{T}) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPointsBIM{eltype(ks)}}(undef,length(ks))
    Threads.@threads for i in eachindex(ks)
        pts=evaluate_points(solver,billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i])
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end













#### EXPANDED BIM ####

"""
    struct ExpandedBoundaryIntegralMethod{T<:Real}

Represents the configuration for the expanded boundary integral method.

# Fields
- `dim_scaling_factor::T`: Scaling factor for the boundary dimensions (compatibility).
- `pts_scaling_factor::Vector{T}`: Scaling factors for the boundary points.
- `sampler::Vector`: Sampling strategy for the boundary points.
- `eps::T`: Numerical tolerance.
- `min_dim::Int64`: Minimum dimensions (compatibility field).
- `min_pts::Int64`: Minimum points for evaluation.
- `rule::SymmetryRuleBIM`: Symmetry rule for the configuration.
"""
struct ExpandedBoundaryIntegralMethod{T} <: AcceleratedSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 
    min_pts::Int64
    rule::SymmetryRuleBIM
end

function ExpandedBoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,symmetries::Union{Vector{Any},Nothing}=nothing,x_bc=:D,y_bc=:D) where {T<:Real, Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[GaussLegendreNodes()]
    return ExpandedBoundaryIntegralMethod{T}(1.0,bs,sampler,eps(T),min_pts,min_pts,SymmetryRuleBIM(billiard,symmetries=symmetries,x_bc=x_bc,y_bc=y_bc))
end

function ExpandedBoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},samplers::Vector,billiard::Bi;min_pts = 20,symmetries::Union{Vector{Any},Nothing}=nothing, x_bc=:D,y_bc=:D) where {T<:Real, Bi<:AbsBilliard} 
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return ExpandedBoundaryIntegralMethod{T}(1.0,bs,samplers,eps(T),min_pts,min_pts,SymmetryRuleBIM(billiard,symmetries=symmetries,x_bc=x_bc,y_bc=y_bc))
end

"""
    default_helmholtz_kernel_first_derivative(
        p1::SVector{2,T}, p2::SVector{2,T}, 
        normal1::SVector{2,T}, k::T, p1_curvature::T
    ) -> Complex{T}

Computes the first derivative of the Helmholtz kernel with respect to the wavenumber.

# Arguments
- `p1::SVector{2,T}`: First point in the domain.
- `p2::SVector{2,T}`: Second point in the domain.
- `normal1::SVector{2,T}`: Normal vector at `p1`.
- `k::T`: Wavenumber.
- `p1_curvature::T`: Curvature at `p1`.

# Returns
- `Complex{T}`: First derivative of the Helmholtz kernel. Zero when the points coincide.
"""
function default_helmholtz_kernel_first_derivative(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T},k::T,p1_curvature::T) where {T<:Real}
    dx, dy = p1[1]-p2[1], p1[2]-p2[2]
    distance12=hypot(dx,dy)
    if abs(distance12)<eps(T)
        return Complex(0.0,0.0)  # First derivative is zero when r -> 0
    end
    return -im/2*distance12*compute_cos_phi(dx,dy,normal1,p1_curvature)*Bessels.hankelh1(0,k*distance12) 
end

"""
    default_helmholtz_kernel_second_derivative(
        p1::SVector{2,T}, p2::SVector{2,T}, 
        normal1::SVector{2,T}, k::T, p1_curvature::T
    ) -> Complex{T}

Computes the second derivative of the Helmholtz kernel with respect to the wavenumber.

# Arguments
- `p1::SVector{2,T}`: First point in the domain.
- `p2::SVector{2,T}`: Second point in the domain.
- `normal1::SVector{2,T}`: Normal vector at `p1`.
- `k::T`: Wavenumber.
- `p1_curvature::T`: Curvature at `p1`.

# Returns
- `Complex{T}`: Second derivative of the Helmholtz kernel. Zero when the points coincide.
"""
function default_helmholtz_kernel_second_derivative(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T},k::T,p1_curvature::T) where {T<:Real}
    dx, dy = p1[1]-p2[1], p1[2]-p2[2]
    distance12=hypot(dx,dy)
    if abs(distance12)<eps(T)
        return Complex(0.0,0.0)  # Second derivative is zero when r -> 0
    end
    return im/(2*k)*compute_cos_phi(dx,dy,normal1,p1_curvature)*((-2+(k*distance12)^2)*Bessels.hankelh1(1,k*distance12)+k*distance12*Bessels.hankelh1(2,k*distance12))
end

"""
    compute_kernel_derivative(
        p1::SVector{2,T}, 
        p2::SVector{2,T}, 
        normal1::SVector{2,T}, 
        curvature1::T, 
        reflected_p2_x::Union{SVector{2,T}, Nothing}, 
        reflected_p2_y::Union{SVector{2,T}, Nothing}, 
        reflected_p2_xy::Union{SVector{2,T}, Nothing}, 
        rule::SymmetryRuleBIM{T}, 
        k::T
    ) -> Complex{T}

Computes the first derivative of the kernel, incorporating symmetry reflections.

# Arguments
- `p1::SVector{2,T}`: First point in the domain.
- `p2::SVector{2,T}`: Second point in the domain.
- `normal1::SVector{2,T}`: Normal vector at `p1`.
- `curvature1::T`: Curvature at `p1`.
- `reflected_p2_x::Union{SVector{2,T}, Nothing}`: Reflected second point across the x-axis, if applicable.
- `reflected_p2_y::Union{SVector{2,T}, Nothing}`: Reflected second point across the y-axis, if applicable.
- `reflected_p2_xy::Union{SVector{2,T}, Nothing}`: Reflected second point across both axes, if applicable.
- `rule::SymmetryRuleBIM{T}`: Symmetry rule for the kernel.
- `k::T`: Wavenumber.

# Returns
- `Complex{T}`: First derivative of the kernel, with symmetry reflections applied.
"""
function compute_kernel_derivative(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T}, curvature1::T,reflected_p2_x::Union{SVector{2,T}, Nothing},reflected_p2_y::Union{SVector{2,T}, Nothing},reflected_p2_xy::Union{SVector{2,T}, Nothing},rule::SymmetryRuleBIM{T}, k::T) where {T<:Real}
    kernel_value=default_helmholtz_kernel_first_derivative(p1,p2,normal1,k,curvature1) # Base kernel computation
    if !isnothing(reflected_p2_x) # Handle x-reflection
        if rule.x_bc==:D
            kernel_value-=default_helmholtz_kernel_first_derivative(p1,reflected_p2_x,normal1,k,curvature1)
        elseif rule.x_bc==:N
            kernel_value+=default_helmholtz_kernel_first_derivative(p1,reflected_p2_x,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_y) # Handle y-reflection
        if rule.y_bc==:D
            kernel_value-=default_helmholtz_kernel_first_derivative(p1,reflected_p2_y,normal1,k,curvature1)
        elseif rule.y_bc==:N
            kernel_value+=default_helmholtz_kernel_first_derivative(p1,reflected_p2_y,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_xy) # Handle xy-reflection
        if rule.x_bc==:D && rule.y_bc==:D
            kernel_value+=default_helmholtz_kernel_first_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:D && rule.y_bc==:N
            kernel_value-=default_helmholtz_kernel_first_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:D
            kernel_value-=default_helmholtz_kernel_first_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:N
            kernel_value+=default_helmholtz_kernel_first_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        end
    end
    return kernel_value
end

"""
    compute_kernel_second_derivative(
        p1::SVector{2,T}, 
        p2::SVector{2,T}, 
        normal1::SVector{2,T}, 
        curvature1::T, 
        reflected_p2_x::Union{SVector{2,T}, Nothing}, 
        reflected_p2_y::Union{SVector{2,T}, Nothing}, 
        reflected_p2_xy::Union{SVector{2,T}, Nothing}, 
        rule::SymmetryRuleBIM{T}, 
        k::T
    ) -> Complex{T}

Computes the second derivative of the kernel, incorporating symmetry reflections.

# Arguments
- `p1::SVector{2,T}`: First point in the domain.
- `p2::SVector{2,T}`: Second point in the domain.
- `normal1::SVector{2,T}`: Normal vector at `p1`.
- `curvature1::T`: Curvature at `p1`.
- `reflected_p2_x::Union{SVector{2,T}, Nothing}`: Reflected second point across the x-axis, if applicable.
- `reflected_p2_y::Union{SVector{2,T}, Nothing}`: Reflected second point across the y-axis, if applicable.
- `reflected_p2_xy::Union{SVector{2,T}, Nothing}`: Reflected second point across both axes, if applicable.
- `rule::SymmetryRuleBIM{T}`: Symmetry rule for the kernel.
- `k::T`: Wavenumber.

# Returns
- `Complex{T}`: Second derivative of the kernel, with symmetry reflections applied.
"""
function compute_kernel_second_derivative(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T}, curvature1::T,reflected_p2_x::Union{SVector{2,T}, Nothing},reflected_p2_y::Union{SVector{2,T}, Nothing},reflected_p2_xy::Union{SVector{2,T}, Nothing},rule::SymmetryRuleBIM{T}, k::T) where {T<:Real}
    kernel_value=default_helmholtz_kernel_second_derivative(p1,p2,normal1,k,curvature1) # Base kernel computation
    if !isnothing(reflected_p2_x) # Handle x-reflection
        if rule.x_bc==:D
            kernel_value-=default_helmholtz_kernel_second_derivative(p1,reflected_p2_x,normal1,k,curvature1)
        elseif rule.x_bc==:N
            kernel_value+=default_helmholtz_kernel_second_derivative(p1,reflected_p2_x,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_y) # Handle y-reflection
        if rule.y_bc==:D
            kernel_value-=default_helmholtz_kernel_second_derivative(p1,reflected_p2_y,normal1,k,curvature1)
        elseif rule.y_bc==:N
            kernel_value+=default_helmholtz_kernel_second_derivative(p1,reflected_p2_y,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_xy) # Handle xy-reflection
        if rule.x_bc==:D && rule.y_bc==:D
            kernel_value+=default_helmholtz_kernel_second_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:D && rule.y_bc==:N
            kernel_value-=default_helmholtz_kernel_second_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:D
            kernel_value-=default_helmholtz_kernel_second_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:N
            kernel_value+=default_helmholtz_kernel_second_derivative(p1,reflected_p2_xy,normal1,k,curvature1)
        end
    end
    return kernel_value
end

"""
    fredholm_matrix_derivative(
        boundary_points::BoundaryPointsBIM{T}, 
        symmetry_rule::SymmetryRuleBIM{T}, 
        k::T
    ) -> Matrix{Complex{T}}

Constructs the derivative of the Fredholm matrix with respect to the wavenumber.

# Arguments
- `boundary_points::BoundaryPointsBIM{T}`: Evaluated boundary points.
- `symmetry_rule::SymmetryRuleBIM{T}`: Symmetry rules for the matrix construction.
- `k::T`: Wavenumber.

# Returns
- `Matrix{Complex{T}}`: First derivative of the Fredholm matrix.
"""
function fredholm_matrix_derivative(boundary_points::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T) where {T<:Real}
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
            kernel_value=compute_kernel_derivative(
                p1,p2,normal1,curvature1,
                isnothing(reflected_points_x) ? nothing : reflected_points_x[j],
                isnothing(reflected_points_y) ? nothing : reflected_points_y[j],
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k)
            fredholm_matrix[i, j]-=ds1*kernel_value
        end
    end
    return fredholm_matrix
end

"""
    fredholm_matrix_second_derivative(
        boundary_points::BoundaryPointsBIM{T}, 
        symmetry_rule::SymmetryRuleBIM{T}, 
        k::T
    ) -> Matrix{Complex{T}}

Constructs the second derivative of the Fredholm matrix with respect to the wavenumber.

# Arguments
- `boundary_points::BoundaryPointsBIM{T}`: Evaluated boundary points.
- `symmetry_rule::SymmetryRuleBIM{T}`: Symmetry rules for the matrix construction.
- `k::T`: Wavenumber.

# Returns
- `Matrix{Complex{T}}`: Second derivative of the Fredholm matrix.
"""
function fredholm_matrix_second_derivative(boundary_points::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T) where {T<:Real}
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
            kernel_value=compute_kernel_second_derivative(
                p1,p2,normal1,curvature1,
                isnothing(reflected_points_x) ? nothing : reflected_points_x[j],
                isnothing(reflected_points_y) ? nothing : reflected_points_y[j],
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k)
            fredholm_matrix[i, j]-=ds1*kernel_value
        end
    end
    return fredholm_matrix
end

"""
    construct_matrices(
        solver::ExpandedBoundaryIntegralMethod, 
        basis::Ba, 
        pts::BoundaryPointsBIM{T}, 
        k::Real
    ) -> Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}

Constructs the Fredholm matrix and its derivatives for the expanded boundary integral method.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: Boundary points in BIM representation.
- `k::Real`: Wavenumber.

# Returns
- `Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}`:
  - Fredholm matrix.
  - First derivative of the Fredholm matrix.
  - Second derivative of the Fredholm matrix.
"""
function construct_matrices(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k) where {Ba<:AbstractHankelBasis}
    A=fredholm_matrix(pts,solver.rule,k)
    dA=fredholm_matrix_derivative(pts,solver.rule,k)
    ddA=fredholm_matrix_second_derivative(pts,solver.rule,k)
    return A,dA,ddA
end

"""
    solve(
        solver::ExpandedBoundaryIntegralMethod, 
        basis::Ba, 
        pts::BoundaryPointsBIM{T}, 
        k::Real, dk::Real; 
        eps::Real = 1e-15
    ) -> Vector{T}

Computes the corrected k0 for a given wavenumber range using the expanded boundary integral method. This is done in an interval [k-dk,k+dk]

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: Boundary points in BIM representation.
- `k::Real`: Central wavenumber for corrections.
- `dk::Real`: Correction range for wavenumber.
- `eps::Real`: Tolerance for filtering valid eigenvalues.

# Returns
- `Vector{T}`: Corrected eigenvalues within the specified range.
"""
function solve(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;eps=1e-15) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k)
    λ,VR,VL=generalized_eigen_all(A,dA)
    valid=(abs.(λ).>eps) .& (abs.(λ).<dk) .& (imag.(λ).<sqrt(eps))
    if !any(valid) 
        return T[]
    end
    λ=λ[valid]
    VR=VR[:,valid] # already normalized
    VL=VL[:,valid] # already normalized
    corr_1=-λ # consistency with taylor expansion expression A * u = - λ * B * u
    numerators=real.(sum(conj(VL).*(dA*VR),dims=1))  # Vectorized dot products
    denominators=real.(sum(conj(VL).*(ddA*VR),dims=1))
    corr_2=-0.5*corr_1.^2 .* real.(numerators./denominators)
    λ=k.+corr_1.+corr_2
    idxs=(k-dk.<λ).&(λ.<k+dk)
    if !any(valid) 
        return T[]
    end
    return λ[idxs]
end











 #### USEFUL ####

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
