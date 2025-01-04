using LinearAlgebra, StaticArrays, TimerOutputs, Bessels

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
    return -im*k/2*distance12*compute_cos_phi(dx,dy,normal1,p1_curvature)*Bessels.hankelh1(0,k*distance12) 
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
        k::T;
        kernel_der_fun::Function=default_helmholtz_kernel_first_derivative
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
- `kernel_der_fun::Function=default_helmholtz_kernel_first_derivative`: Function to use for the derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_first_derivative`).

# Returns
- `Complex{T}`: First derivative of the kernel, with symmetry reflections applied.
"""
function compute_kernel_derivative(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T}, curvature1::T,reflected_p2_x::Union{SVector{2,T}, Nothing},reflected_p2_y::Union{SVector{2,T}, Nothing},reflected_p2_xy::Union{SVector{2,T}, Nothing},rule::SymmetryRuleBIM{T}, k::T; kernel_der_fun=default_helmholtz_kernel_first_derivative) where {T<:Real}
    kernel_value=kernel_der_fun(p1,p2,normal1,k,curvature1) # Base kernel computation
    if !isnothing(reflected_p2_x) # Handle x-reflection
        if rule.x_bc==:D
            kernel_value-=kernel_der_fun(p1,reflected_p2_x,normal1,k,curvature1)
        elseif rule.x_bc==:N
            kernel_value+=kernel_der_fun(p1,reflected_p2_x,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_y) # Handle y-reflection
        if rule.y_bc==:D
            kernel_value-=kernel_der_fun(p1,reflected_p2_y,normal1,k,curvature1)
        elseif rule.y_bc==:N
            kernel_value+=kernel_der_fun(p1,reflected_p2_y,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_xy) # Handle xy-reflection
        if rule.x_bc==:D && rule.y_bc==:D
            kernel_value+=kernel_der_fun(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:D && rule.y_bc==:N
            kernel_value-=kernel_der_fun(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:D
            kernel_value-=kernel_der_fun(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:N
            kernel_value+=kernel_der_fun(p1,reflected_p2_xy,normal1,k,curvature1)
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
        k::T;
        kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative
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
- `kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative`: Function to use for the 2nd derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_second_derivative`).

# Returns
- `Complex{T}`: Second derivative of the kernel, with symmetry reflections applied.
"""
function compute_kernel_second_derivative(p1::SVector{2,T},p2::SVector{2,T},normal1::SVector{2,T}, curvature1::T,reflected_p2_x::Union{SVector{2,T}, Nothing},reflected_p2_y::Union{SVector{2,T}, Nothing},reflected_p2_xy::Union{SVector{2,T}, Nothing},rule::SymmetryRuleBIM{T}, k::T; kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {T<:Real}
    kernel_value=kernel_der2_fun(p1,p2,normal1,k,curvature1) # Base kernel computation
    if !isnothing(reflected_p2_x) # Handle x-reflection
        if rule.x_bc==:D
            kernel_value-=kernel_der2_fun(p1,reflected_p2_x,normal1,k,curvature1)
        elseif rule.x_bc==:N
            kernel_value+=kernel_der2_fun(p1,reflected_p2_x,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_y) # Handle y-reflection
        if rule.y_bc==:D
            kernel_value-=kernel_der2_fun(p1,reflected_p2_y,normal1,k,curvature1)
        elseif rule.y_bc==:N
            kernel_value+=kernel_der2_fun(p1,reflected_p2_y,normal1,k,curvature1)
        end
    end
    if !isnothing(reflected_p2_xy) # Handle xy-reflection
        if rule.x_bc==:D && rule.y_bc==:D
            kernel_value+=kernel_der2_fun(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:D && rule.y_bc==:N
            kernel_value-=kernel_der2_fun(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:D
            kernel_value-=kernel_der2_fun(p1,reflected_p2_xy,normal1,k,curvature1)
        elseif rule.x_bc==:N && rule.y_bc==:N
            kernel_value+=kernel_der2_fun(p1,reflected_p2_xy,normal1,k,curvature1)
        end
    end
    return kernel_value
end

"""
    fredholm_matrix_derivative(
        boundary_points::BoundaryPointsBIM{T}, 
        symmetry_rule::SymmetryRuleBIM{T}, 
        k::T;
        kernel_der_fun::Function=default_helmholtz_kernel_first_derivative
    ) -> Matrix{Complex{T}}

Constructs the derivative of the Fredholm matrix with respect to the wavenumber.

# Arguments
- `boundary_points::BoundaryPointsBIM{T}`: Evaluated boundary points.
- `symmetry_rule::SymmetryRuleBIM{T}`: Symmetry rules for the matrix construction.
- `k::T`: Wavenumber.
- `kernel_der_fun::Function=default_helmholtz_kernel_first_derivative`: Function to use for the 1st derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_first_derivative`)

# Returns
- `Matrix{Complex{T}}`: First derivative of the Fredholm matrix.
"""
function fredholm_matrix_derivative(boundary_points::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T;kernel_der_fun=default_helmholtz_kernel_first_derivative) where {T<:Real}
    xy_points=boundary_points.xy
    normals=boundary_points.normal
    curvatures=boundary_points.curvature
    ds=boundary_points.ds
    N=length(xy_points)
    reflected_points_x=symmetry_rule.symmetry_type in [:x,:xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:x,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_y=symmetry_rule.symmetry_type in [:y,:xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:y,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_xy=symmetry_rule.symmetry_type==:xy ?
        [apply_reflection(p,symmetry_rule) for p in xy_points] : nothing
    fredholm_matrix=fill(Complex(0.0,0.0),N,N)
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
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k;kernel_der_fun=kernel_der_fun)
            fredholm_matrix[i, j]-=ds1*kernel_value
        end
    end
    return fredholm_matrix
end

"""
    fredholm_matrix_second_derivative(
        boundary_points::BoundaryPointsBIM{T}, 
        symmetry_rule::SymmetryRuleBIM{T}, 
        k::T;
        kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative
    ) -> Matrix{Complex{T}}

Constructs the second derivative of the Fredholm matrix with respect to the wavenumber.

# Arguments
- `boundary_points::BoundaryPointsBIM{T}`: Evaluated boundary points.
- `symmetry_rule::SymmetryRuleBIM{T}`: Symmetry rules for the matrix construction.
- `k::T`: Wavenumber.
- `kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative`: Function to use for the 2nd derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_second_derivative`).

# Returns
- `Matrix{Complex{T}}`: Second derivative of the Fredholm matrix.
"""
function fredholm_matrix_second_derivative(boundary_points::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T;kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {T<:Real}
    xy_points=boundary_points.xy
    normals=boundary_points.normal
    curvatures=boundary_points.curvature
    ds=boundary_points.ds
    N=length(xy_points)
    reflected_points_x=symmetry_rule.symmetry_type in [:x,:xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:x,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_y=symmetry_rule.symmetry_type in [:y,:xy] ?
        [apply_reflection(p, SymmetryRuleBIM(:y,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_xy=symmetry_rule.symmetry_type==:xy ?
        [apply_reflection(p,symmetry_rule) for p in xy_points] : nothing
    fredholm_matrix=fill(Complex(0.0,0.0),N,N)
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
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k;kernel_der2_fun=kernel_der2_fun)
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
        k::Real;
        kernel_fun::Function=default_helmholtz_kernel,
        kernel_der_fun::Function=default_helmholtz_kernel_first_derivative,
        kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative
    ) -> Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}

Constructs the Fredholm matrix and its derivatives for the expanded boundary integral method.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: Boundary points in BIM representation.
- `k::Real`: Wavenumber.
- `kernel_fun::Function`: Function to use for the kernel computation (default for free 2D particle: `default_helmholtz_kernel`).
- `kernel_der_fun::Function=default_helmholtz_kernel_first_derivative`: Function to use for the 1st derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_first_derivative`).
- `kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative`: Function to use for the 2nd derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_second_derivative`).

# Returns
- `Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}`:
  - Fredholm matrix.
  - First derivative of the Fredholm matrix.
  - Second derivative of the Fredholm matrix.
"""
function construct_matrices(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {Ba<:AbstractHankelBasis}
    A=fredholm_matrix(pts,solver.rule,k;kernel_fun=kernel_fun)
    dA=fredholm_matrix_derivative(pts,solver.rule,k;kernel_der_fun=kernel_der_fun)
    ddA=fredholm_matrix_second_derivative(pts,solver.rule,k;kernel_der2_fun=kernel_der2_fun)
    return A,dA,ddA
end

"""
    solve(
        solver::ExpandedBoundaryIntegralMethod, 
        basis::Ba, pts::BoundaryPointsBIM{T}, 
        k::Real,
        dk::Real; 
        eps::Real = 1e-15,
        kernel_fun::Function=default_helmholtz_kernel,
        kernel_der_fun::Function=default_helmholtz_kernel_first_derivative,
        kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative) -> Vector{T}

Computes the corrected k0 for a given wavenumber range using the expanded boundary integral method. This is done in an interval [k-dk,k+dk]

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `pts::BoundaryPointsBIM{T}`: Boundary points in BIM representation.
- `k::Real`: Central wavenumber for corrections.
- `dk::Real`: Correction range for wavenumber.
- `use_lapack_raw::Bool=false`: Use the ggev LAPACK function directly without Julia's eigen(A,B) wrapper for it. Might provide speed-up for certain situations (small matrices...)
- `kernel_fun::Function`: Function to use for the kernel computation (default for free 2D particle: `default_helmholtz_kernel`).
- `kernel_der_fun::Function=default_helmholtz_kernel_first_derivative`: Function to use for the 1st derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_first_derivative`).
- `kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative`: Function to use for the 2nd derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_second_derivative`).

# Returns
- `Vector{T}`: Corrected eigenvalues within the specified range.
"""
function solve(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,kernel_der_fun=kernel_der_fun,kernel_der2_fun=kernel_der2_fun)
    if use_lapack_raw
        λ,VR,VL=generalized_eigen_all_LAPACK_LEGACY(A,dA)
    else
        λ,VR,VL=generalized_eigen_all(A,dA)
    end
    T=eltype(real.(λ))
    valid=(real.(λ).<dk) .& (abs.(imag.(λ).<dk)) # WRONG BUT WHY ???
    #valid=abs.(λ).<dk
    if !any(valid)
        return Vector{T}(),Vector{T}() # early termination
    end
    λ=real.(λ[valid])
    VR=VR[:,valid]
    VL=VL[:,valid]
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        r_dA=similar(v_right)
        r_ddA=similar(v_right)
        mul!(r_ddA,ddA,v_right)
        mul!(r_dA,dA,v_right)
        numerator=real(dot(v_left,r_ddA)) # v_left' * (ddA * v_right)
        denominator=real(dot(v_left,r_dA)) # v_left' * (dA * v_right)
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*numerator/denominator
    end
    λ_corrected=k.+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    idxs=((k-dk).<λ_corrected) .& (λ_corrected.<(k+dk)) # Filter corrected eigenvalues within [k - dk, k + dk]
    if !any(idxs)
        return Vector{T}(), Vector{T}()
    end
    return λ_corrected[idxs],tens[idxs]
end







