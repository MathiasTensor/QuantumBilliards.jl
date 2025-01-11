using LinearAlgebra, StaticArrays, TimerOutputs, Bessels, ProgressMeter

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
    sampler=[LinearNodes()]
    return ExpandedBoundaryIntegralMethod{T}(1.0,bs,sampler,eps(T),min_pts,min_pts,SymmetryRuleBIM(billiard,symmetries=symmetries,x_bc=x_bc,y_bc=y_bc))
end

function ExpandedBoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},samplers::Vector,billiard::Bi;min_pts = 20,symmetries::Union{Vector{Any},Nothing}=nothing, x_bc=:D,y_bc=:D) where {T<:Real, Bi<:AbsBilliard} 
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return ExpandedBoundaryIntegralMethod{T}(1.0,bs,samplers,eps(T),min_pts,min_pts,SymmetryRuleBIM(billiard,symmetries=symmetries,x_bc=x_bc,y_bc=y_bc))
end

#### NEW MATRIX APPROACH FOR FASTER CODE #### 

"""
    default_helmholtz_kernel_derivative_matrix(bp::BoundaryPointsBIM{T}, k::T) -> Matrix{Complex{T}}

Constructs the first derivative (with respect to `k`) of the 2D Helmholtz kernel *for all pairs* of points
in the boundary `bp`. Each entry `(i, j)` in the returned matrix corresponds to the derivative of

    cos(φᵢ) * (-im*k/2)*r * H₀^(1)(k*r)

where:
- `r` is the distance between points `i` and `j`,
- `cos(φᵢ)` is `(nᵢ · (pᵢ - pⱼ)) / r`, using the normal at `pᵢ`,
- `H₀^(1)` is the Hankel function of the first kind, order 0.

# Arguments
- `bp::BoundaryPointsBIM{T}`: A set of boundary points, including `(x, y)` coordinates and normals.
- `k::T`: Wavenumber, a real value.

# Returns
- `Matrix{Complex{T}}`: An `N×N` matrix, where `N` is the number of boundary points. The element `(i,j)`
  is the derivative of the Helmholtz kernel with respect to `k` between points `i` and `j`. Diagonal
  entries where `distance < eps(T)` are set to zero.

**Note**: For `(i ≠ j)`, a mirrored computation is performed for entry `(j,i)` using the normal at `pⱼ`.
Hence, the matrix is typically *not* symmetric, because `cos(φ)` depends on the normal at the source row.
"""
@inline function default_helmholtz_kernel_derivative_matrix(bp::BoundaryPointsBIM{T},k::T) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(xy)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    dx=xs.-xs'
    dy=ys.-ys'
    distances=hypot.(dx,dy)
    @inbounds for i in 1:N
        for j in 1:i # symmetric hankel part
            #dx,dy=xy[i][1]-xy[j][1],xy[i][2]-xy[j][2]
            #distance=hypot(dx,dy)
            distance=distances[i,j]
            if distance<eps(T)
                M[i,j]=Complex(T(0.0),T(0.0))
            else
                #cos_phi=(normals[i][1]*dx+normals[i][2]*dy)/distance
                cos_phi=(normals[i][1]*dx[i,j]+normals[i][2]*dy[i,j])/distance
                hankel=-im*k/2.0*distance*Bessels.hankelh1(0,k*distance)
                M[i,j]=cos_phi*hankel
            end
            if i!=j
                #cos_phi_symmetric=(normals[j][1]*(-dx)+normals[j][2]*(-dy))/distance
                cos_phi_symmetric=(normals[j][1]*(-dx[i,j])+normals[j][2]*(-dy[i,j]))/distance # Hankel is symmetric, but cos_phi is not; compute explicitly for M[j, i]
                M[j,i]=cos_phi_symmetric*hankel
            end
        end
    end
    return M
end

"""
    default_helmholtz_kernel_derivative_matrix(bp_s::BoundaryPointsBIM{T}, xy_t::Vector{SVector{2,T}}, k::T)
        -> Matrix{Complex{T}}

Constructs the first derivative (with respect to `k`) of the 2D Helmholtz kernel between each boundary point
in `bp_s` and a separate list of target points `xy_t`. This is similar to the single-argument version, except
the second set of points is taken from `xy_t` rather than the boundary itself.

# Arguments
- `bp_s::BoundaryPointsBIM{T}`: Source boundary points, which provide `(x, y)` and normal vectors.
- `xy_t::Vector{SVector{2,T}}`: Target points, typically a different set of points in the plane.
- `k::T`: Wavenumber, a real value.

# Returns
- `Matrix{Complex{T}}`: An `N×M` matrix if `bp_s` has `N` points and `xy_t` has `M` points. The element `(i,j)`
  is the derivative of the Helmholtz kernel w.r.t. `k` between source point `i` and target point `j`.
  If the distance is `< eps(T)`, the entry is set to zero.

**Note**: This routine does not rely on reflection logic. It simply computes the derivative
for each `(source_i, target_j)` pair based on the distance and the dot product with `normalᵢ`.
"""
@inline function default_helmholtz_kernel_derivative_matrix(bp_s::BoundaryPointsBIM{T},xy_t::Vector{SVector{2,T}},k::T) where {T<:Real}
    xy_s=bp_s.xy
    normals=bp_s.normal
    N=length(xy_s)
    M=Matrix{Complex{T}}(undef,N,N)
    x_s=getindex.(xy_s,1)
    y_s=getindex.(xy_s,2)
    x_t=getindex.(xy_t,1)
    y_t=getindex.(xy_t,2)
    dx=x_s.-x_t'
    dy= y_s.-y_t'
    distances=hypot.(dx,dy)
    @inbounds for i in 1:N
        for j in 1:N
            #dx,dy=xy_s[i][1]-xy_t[j][1],xy_s[i][2]-xy_t[j][2]
            #distance=hypot(dx,dy)
            distance=distances[i,j]
            if distance<eps(T)
                M[i,j]=Complex(T(0.0),T(0.0))
            else
                #cos_phi=(normals[i][1]*dx+normals[i][2]*dy)/distance
                cos_phi=(normals[i][1]*dx[i,j]+normals[i][2]*dy[i,j])/distance
                hankel=-im*k/2.0*distance*Bessels.hankelh1(0,k*distance)
                M[i,j]=cos_phi*hankel
            end
        end
    end
    return M
end

"""
    default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPointsBIM{T}, k::T)
        -> Matrix{Complex{T}}

Constructs the second derivative (with respect to `k`) of the 2D Helmholtz kernel *for all pairs* of points
in the boundary `bp`. Each entry `(i, j)` in the returned matrix corresponds to

    cos(φᵢ) * ( im/(2*k) ) * [ ...combination of HankelH1(1) and HankelH1(2)... ]

where `cos(φᵢ) = (nᵢ · (pᵢ - pⱼ)) / r` and `r` is the distance between boundary points `pᵢ` and `pⱼ`.
The exact Hankel expression matches the partial derivative:
    
    (d²/dk²) of [ cos(φᵢ) * H₀^(1)(k*r)* ... ].

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points, containing `(x, y)` and normals.
- `k::T`: Wavenumber, real.

# Returns
- `Matrix{Complex{T}}`: An `N×N` matrix, where each entry is the second derivative of the Helmholtz kernel
  wrt. `k` between boundary points `i` and `j`. If `distance < eps(T)`, the entry is set to zero.

**Note**: Similar to the first-derivative matrix, the factor `cos(φᵢ)` uses the normal at the source
row `i`, so the matrix is not necessarily symmetric unless the geometry enforces it.
"""
@inline function default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPointsBIM{T},k::T) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(xy)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    dx=xs.-xs'
    dy=ys.-ys'
    distances=hypot.(dx,dy)
    @inbounds for i in 1:N
        for j in 1:i # symmetric hankel part
            #dx,dy=xy[i][1]-xy[j][1],xy[i][2]-xy[j][2]
            #distance=hypot(dx,dy)
            distance=distances[i,j]
            if distance<eps(T)
                M[i,j]=Complex(T(0.0),T(0.0))
            else
                #cos_phi=(normals[i][1]*dx+normals[i][2]*dy)/distance
                cos_phi=(normals[i][1]*dx[i,j]+normals[i][2]*dy[i,j])/distance
                hankel=im/(2*k)*((-2+(k*distance)^2)*Bessels.hankelh1(1,k*distance)+k*distance*Bessels.hankelh1(2,k*distance))
                M[i,j]=cos_phi*hankel
            end
            if i!=j
                #cos_phi_symmetric=(normals[j][1]*(-dx)+normals[j][2]*(-dy))/distance # Hankel is symmetric, but cos_phi is not; compute explicitly for M[j, i]
                cos_phi_symmetric=(normals[j][1]*(-dx[i,j])+normals[j][2]*(-dy[i,j]))/distance
                M[j,i]=cos_phi_symmetric*hankel
            end
        end
    end
    return M
end

"""
    default_helmholtz_kernel_second_derivative_matrix(bp_s::BoundaryPointsBIM{T},
                                                      xy_t::Vector{SVector{2,T}},
                                                      k::T) -> Matrix{Complex{T}}

Constructs the second derivative (with respect to `k`) of the 2D Helmholtz kernel between each source point
in `bp_s` and a separate list of target points `xy_t`. The entry `(i, j)` in the returned `N×M` matrix is
the second derivative of the Helmholtz kernel, computed using the normal at source `i` and the distance
to target `j`.

# Arguments
- `bp_s::BoundaryPointsBIM{T}`: Source boundary points with `(x, y)` and normal vectors.
- `xy_t::Vector{SVector{2,T}}`: A vector of target points in the plane.
- `k::T`: Wavenumber, real.

# Returns
- `Matrix{Complex{T}}`: If there are `N` source points and `M` target points, returns `N×M`. Any pair
  whose distance is `< eps(T)` yields a zero entry.

**Note**: This function does not include reflection or symmetry corrections. It purely evaluates
the second derivative of the kernel formula at each `(source_i, target_j)`.
"""
@inline function default_helmholtz_kernel_second_derivative_matrix(bp_s::BoundaryPointsBIM{T},xy_t::Vector{SVector{2,T}},k::T) where {T<:Real}
    xy_s=bp_s.xy
    normals=bp_s.normal
    N=length(xy_s)
    M=Matrix{Complex{T}}(undef,N,N)
    x_s=getindex.(xy_s,1)
    y_s=getindex.(xy_s,2)
    x_t=getindex.(xy_t,1)
    y_t=getindex.(xy_t,2)
    dx=x_s.-x_t'
    dy= y_s.-y_t'
    distances=hypot.(dx,dy)
    @inbounds for i in 1:N
        for j in 1:N
            #dx,dy=xy_s[i][1]-xy_t[j][1],xy_s[i][2]-xy_t[j][2]
            #distance=hypot(dx,dy)
            distance=distances[i,j]
            if distance<eps(T)
                M[i,j]=Complex(T(0.0),T(0.0))
            else
                #cos_phi=(normals[i][1]*dx+normals[i][2]*dy)/distance
                cos_phi=(normals[i][1]*dx[i,j]+normals[i][2]*dy[i,j])/distance
                hankel=im/(2*k)*((-2+(k*distance)^2)*Bessels.hankelh1(1,k*distance)+k*distance*Bessels.hankelh1(2,k*distance))
                M[i,j]=cos_phi*hankel
            end
        end
    end
    return M
end

"""
    compute_kernel_der_matrix(bp::BoundaryPointsBIM{T}, k::T; kernel_fun=:first) -> Matrix{Complex{T}}

Compute the Helmholtz derivative kernel matrix (first or second derivative w.r.t. `k`) for boundary points
`bp`, using one of the default matrix functions or a custom function. If `kernel_fun = :first`, it returns
the first derivative matrix; if `kernel_fun = :second`, it returns the second derivative matrix; otherwise,
`kernel_fun` should be a custom function `(bp, k) -> Matrix{Complex{T}}`.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points with `(x, y)` and normal vectors.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol,Function} = :first`: Either `:first`, `:second`, or a custom function.

# Returns
- `Matrix{Complex{T}}`: The `N×N` matrix with the derivative of the Helmholtz kernel w.r.t. `k`.

**Note**: This version does not apply reflection or boundary-condition logic; it’s purely the direct
derivative matrix for the given set of points.
"""
function compute_kernel_der_matrix(bp::BoundaryPointsBIM{T},k::T;kernel_fun::Union{Symbol,Function}=:first) where {T<:Real}
    if kernel_fun==:first
        return default_helmholtz_kernel_derivative_matrix(bp,k)
    elseif kernel_fun==:second
        return default_helmholtz_kernel_second_derivative_matrix(bp,k)
    else
        return kernel_fun(bp,k)
    end
end

"""
    compute_kernel_der_matrix(bp::BoundaryPointsBIM{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T;
                              kernel_fun=:first) -> Matrix{Complex{T}}

Compute the Helmholtz derivative kernel matrix (first or second derivative w.r.t. `k`) for boundary points
`bp`, incorporating reflection logic based on `symmetry_rule`. This builds a “base” derivative matrix,
then constructs reflected submatrices (if `symmetry_type` is `:x`, `:y`, or `:xy`) and applies add/sub
depending on boundary conditions (`:D` vs. `:N`).

# Arguments
- `bp::BoundaryPointsBIM{T}`: The boundary points, with positions and normals.
- `symmetry_rule::SymmetryRuleBIM{T}`: Reflection/boundary-condition info (like `:x_bc`, `:y_bc`).
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol,Function}=:first`: Either `:first`, `:second`, or a custom function that
  returns a matrix for the derivative kernel.

# Returns
- `Matrix{Complex{T}}`: The derivative kernel matrix, including reflection corrections. Size is `N×N`
  if there are `N` boundary points in `bp`.

**Note**: Each reflection (x, y, or xy) is computed as a separate NxN matrix, then added or subtracted
from the base matrix according to the boundary condition (Dirichlet or Neumann). The final result
matches the “pointwise reflection” logic from the legacy code.
"""
function compute_kernel_der_matrix(bp::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T;kernel_fun::Union{Symbol,Function}=:first) where {T<:Real}
    xy_points=bp.xy
    reflected_points_x=symmetry_rule.symmetry_type in [:x,:xy] ?
        [apply_reflection(p,SymmetryRuleBIM(:x,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_y=symmetry_rule.symmetry_type in [:y,:xy] ?
        [apply_reflection(p,SymmetryRuleBIM(:y,symmetry_rule.x_bc,symmetry_rule.y_bc,symmetry_rule.shift_x,symmetry_rule.shift_y)) for p in xy_points] : nothing
    reflected_points_xy=symmetry_rule.symmetry_type==:xy ?
        [apply_reflection(p,symmetry_rule) for p in xy_points] : nothing
    function use_kernel(bp,k)
        if kernel_fun==:first
            kernel=default_helmholtz_kernel_derivative_matrix(bp,k) # starting der kernel where no reflections
        elseif kernel_fun==:second
            kernel=default_helmholtz_kernel_second_derivative_matrix(bp,k) # starting der2 kernel where no reflections
        else
            kernel=kernel_fun(bp,k) # starting kernel where no reflections
        end
    end
    function use_kernel(bp,refl_pts,k)
        if kernel_fun==:first
            kernel=default_helmholtz_kernel_derivative_matrix(bp,refl_pts,k) # starting der kernel where no reflections
        elseif kernel_fun==:second
            kernel=default_helmholtz_kernel_second_derivative_matrix(bp,refl_pts,k) # starting der2 kernel where no reflections
        else
            kernel=kernel_fun(bp,refl_pts,k) # starting kernel where no reflections
        end
    end
    kernel_val=use_kernel(bp,k) # base 
    if symmetry_rule.symmetry_type in [:x,:y,:xy]
        if symmetry_rule.symmetry_type in [:x,:xy]  # Reflection across x-axis
            reflected_kernel_x=use_kernel(bp,reflected_points_x,k)
            if symmetry_rule.x_bc==:D # Adjust kernel based on boundary condition
                kernel_val.-=reflected_kernel_x
            elseif symmetry_rule.x_bc==:N
                kernel_val.+=reflected_kernel_x
            end
        end
        if symmetry_rule.symmetry_type in [:y,:xy]
            reflected_kernel_y=use_kernel(bp,reflected_points_y,k)
            if symmetry_rule.y_bc==:D # # Adjust kernel based on boundary condition
                kernel_val.-=reflected_kernel_y
            elseif symmetry_rule.y_bc==:N
                kernel_val.+=reflected_kernel_y
            end
        end
        if symmetry_rule.symmetry_type==:xy # both x and y additional logic
            reflected_kernel_xy=use_kernel(bp,reflected_points_xy,k)
            if symmetry_rule.x_bc==:D && symmetry_rule.y_bc==:D # Adjust kernel based on boundary conditions
                kernel_val.+=reflected_kernel_xy
            elseif symmetry_rule.x_bc==:D && symmetry_rule.y_bc==:N
                kernel_val.-=reflected_kernel_xy
            elseif symmetry_rule.x_bc==:N && symmetry_rule.y_bc==:D
                kernel_val.-=reflected_kernel_xy
            elseif symmetry_rule.x_bc==:N && symmetry_rule.y_bc==:N
                kernel_val.+=reflected_kernel_xy
            end
        end
    end
    return kernel_val
end

"""
    fredholm_matrix_der(bp::BoundaryPointsBIM{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T;
                        kernel_fun=:first) -> Matrix{Complex{T}}

Build a Fredholm derivative matrix `dA/dk` by combining the derivative kernel matrix and the boundary
elements `ds`. Specifically, it computes

    - ( derivative_kernel_matrix ) .* ds'

where `ds'` is a broadcast of the boundary segment lengths onto each column.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points with `(x,y)`, normals, and `ds`.
- `symmetry_rule::SymmetryRuleBIM{T}`: Optional reflection/boundary condition rules.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Symbol,Function}=:first`: If `:first`, uses the first derivative matrix;
  if `:second`, uses the second derivative matrix; else a custom function.

# Returns
- `Matrix{Complex{T}}`: The `N×N` derivative Fredholm matrix, i.e. `- K_der .* ds'`.

**Note**: This function is analogous to `fredholm_matrix_derivative` in the legacy code, except it
uses a matrix approach to compute all pairwise derivatives at once.
"""
function fredholm_matrix_der(bp::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T;kernel_fun::Union{Symbol,Function}=:first) where {T<:Real}
    kernel_matrix=isnothing(symmetry_rule) ?
        compute_kernel_der_matrix(bp,k;kernel_fun=kernel_fun) :
        compute_kernel_der_matrix(bp,symmetry_rule,k;kernel_fun=kernel_fun)
    ds=bp.ds
    N=length(ds)
    fredholm_der_matrix=-kernel_matrix.*ds'
    return fredholm_der_matrix
end

"""
    all_fredholm_associated_matrices(
        bp::BoundaryPointsBIM{T}, symmetry_rule::SymmetryRuleBIM{T}, k::T;
        kernel_fun::Union{Tuple,Function} = (:default, :first, :second)
    ) -> (Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}})

Construct the full Fredholm matrix and its first and second derivatives w.r.t. the wavenumber,
using a matrix approach. Specifically:
  1. `kernel_fun[1]` -> the base Helmholtz kernel,
  2. `kernel_fun[2]` -> the first derivative wrt `k`,
  3. `kernel_fun[3]` -> the second derivative wrt `k`.

Applies reflection logic if `symmetry_rule` is non-nothing. Then multiplies by `ds` and inserts
the identity on the diagonal for the base matrix.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points (positions, normals, ds).
- `symmetry_rule::SymmetryRuleBIM{T}`: Reflection/boundary condition rules.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Tuple,Function} = (:default, :first, :second)`: The base kernel,
  plus derivative kernels. If a tuple of symbols, each is looked up in the default
  matrix computations. If functions, they should each produce an `N×N` matrix.

# Returns
- `fredholm_matrix::Matrix{Complex{T}}`: The base matrix `I - kernel .* ds'`.
- `fredholm_der_matrix::Matrix{Complex{T}}`: The first derivative matrix `- kernel_der .* ds'`.
- `fredholm_der2_matrix::Matrix{Complex{T}}`: The second derivative matrix `- kernel_der2 .* ds'`.
"""
function all_fredholm_associated_matrices(bp::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {T<:Real}
    if isnothing(symmetry_rule)
        kernel_matrix=compute_kernel_matrix(bp,k;kernel_fun=kernel_fun[1])
        kernel_der_matrix=compute_kernel_der_matrix(bp,k;kernel_fun=kernel_fun[2])
        kernel_der2_matrix=compute_kernel_der_matrix(bp,k;kernel_fun=kernel_fun[3])
    else
        kernel_matrix=compute_kernel_matrix(bp,symmetry_rule,k;kernel_fun=kernel_fun[1])
        kernel_der_matrix=compute_kernel_der_matrix(bp,symmetry_rule,k;kernel_fun=kernel_fun[2])
        kernel_der2_matrix=compute_kernel_der_matrix(bp,symmetry_rule,k;kernel_fun=kernel_fun[3])
    end
    ds=bp.ds
    N=length(ds)
    fredholm_matrix=Diagonal(ones(Complex{T},N))-kernel_matrix.*ds'
    fredholm_der_matrix=-kernel_der_matrix.*ds'
    fredholm_der2_matrix=-kernel_der2_matrix.*ds'
    return fredholm_matrix,fredholm_der_matrix,fredholm_der2_matrix
end

"""
    construct_matrices(
        solver::ExpandedBoundaryIntegralMethod, 
        basis::Ba, 
        pts::BoundaryPointsBIM{T}, 
        k::T;
        kernel_fun::Union{Tuple,Function} = (:default, :first, :second)
    ) -> (Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}})

High-level routine that builds the Fredholm matrix and its first/second derivatives
for the given boundary `pts` and wavenumber `k`, relying on the matrix-based approach
in `all_fredholm_associated_matrices`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: An EBIM solver configuration (its `rule` is used).
- `basis::Ba`: The basis function type (not used directly here, but part of the pipeline).
- `pts::BoundaryPointsBIM{T}`: Boundary points with geometry data.
- `k::T`: The wavenumber.
- `kernel_fun::Union{Tuple,Function} = (:default, :first, :second)`: A triple specifying
  (base kernel, first derivative kernel, second derivative kernel).

# Returns
- `(A, dA, ddA)::Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}`:
  - `A`: The Fredholm matrix.
  - `dA`: The first derivative wrt `k`.
  - `ddA`: The second derivative wrt `k`.
"""
function construct_matrices(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::T;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {Ba<:AbstractHankelBasis,T<:Real}
    return all_fredholm_associated_matrices(pts,solver.rule,k;kernel_fun=kernel_fun)
end

"""
    solve(
        solver::ExpandedBoundaryIntegralMethod, 
        basis::Ba, 
        pts::BoundaryPointsBIM{T}, 
        k::T, 
        dk::T;
        use_lapack_raw::Bool=false,
        kernel_fun::Union{Tuple,Function} = (:default, :first, :second)
    ) -> (Vector{T}, Vector{T})

Compute approximate "corrected" eigenvalues near the wavenumber `k`, using the expanded boundary integral
method. The routine builds `(A, dA, ddA)` via `construct_matrices`, then solves the generalized eigenvalue
problem `A*x = λ * dA * x`. It filters those eigenvalues whose real part lies in `(-dk, dk)`, then applies
a second-order correction with `ddA`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: EBIM configuration.
- `basis::Ba`: The basis type (unused directly here, but part of the solver pipeline).
- `pts::BoundaryPointsBIM{T}`: Boundary points.
- `k::T`: Central wavenumber.
- `dk::T`: Half-width of the search interval in real and imaginary parts of `λ`.
- `use_lapack_raw::Bool=false`: If true, call a direct LAPACK routine for `A,dA` eigen solves.
- `kernel_fun::Union{Tuple,Function} = (:default,:first,:second)`: The base kernel and its derivatives.

# Returns
- `(λ_corrected::Vector{T}, tension::Vector{T})`: The "corrected" wavenumbers (`k + corrections`)
  for the valid solutions, and a tension measure (`abs(corrections)`).

**Note**: The corrections are computed from the first- and second-order expansions in terms of `λ[i]`,
with final `k_corrected = k + corr₁ + corr₂`.
"""
function solve(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun)
    if use_lapack_raw
        λ,VR,VL=generalized_eigen_all_LAPACK_LEGACY(A,dA)
    else
        λ,VR,VL=generalized_eigen_all(A,dA)
    end
    T=eltype(real.(λ))
    valid=(abs.(real.(λ)).<dk) .& (abs.(imag.(λ)).<dk) # use (-dk,dk) × (-dk,dk) instead of disc of radius dk
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
        numerator=transpose(v_left)*ddA*v_right
        denominator=transpose(v_left)*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    λ_corrected=k.+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    return λ_corrected,tens
end


#### DEBUGGING TOOLS ####

"""
    solve_DEBUG_w_1st_order_corrections(
        solver::ExpandedBoundaryIntegralMethod,
        basis::Ba,
        pts::BoundaryPointsBIM,
        k;
        kernel_fun=(:default, :first, :second)
    ) -> (Vector{T}, Vector{T})

A debug routine that solves the generalized eigenproblem `(A, dA)` at wavenumber `k`, extracts eigenvalues
`λ`, then applies *only first-order corrections* `corr₁ = -λ[i]`. The returned `k + corr₁` is the
first-order guess for each root. The tension is `abs(corr₁)`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: EBIM solver config.
- `basis::Ba`: Basis function type.
- `pts::BoundaryPointsBIM`: Boundary geometry.
- `k`: The wavenumber for which we do `(A, dA)`.
- `kernel_fun`: A triple or custom functions specifying the base kernel, first derivative, etc.

# Returns
- `(λ_corrected, tens)::(Vector{T}, Vector{T})`: The first-order corrected k-values and their "tensions".

**Note**: No second derivative `ddA` is used here; purely `A,dA`.
"""
function solve_DEBUG_w_1st_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=(:default,:first,:second)) where {Ba<:AbstractHankelBasis}
    A,dA,_=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun)
    F=eigen(A,dA)
    λ=F.values
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    T=eltype(real.(λ))
    λ=real.(λ)
    corr_1=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        corr_1[i]=-λ[i]
    end
    λ_corrected=k.+corr_1
    tens=abs.(corr_1)
    return λ_corrected,tens
end

"""
    solve_DEBUG_w_2nd_order_corrections(
        solver::ExpandedBoundaryIntegralMethod,
        basis::Ba,
        pts::BoundaryPointsBIM,
        k;
        kernel_fun=(:default, :first, :second)
    ) -> (Vector{T}, Vector{T}, Vector{T}, Vector{T})

A debug routine that solves the generalized eigenproblem `(A, dA)` at wavenumber `k`, then applies
**both first- and second-order** corrections to refine the approximate roots. Specifically,
it extracts λ from `A*x = λ dA*x`, then does:

  corr₁[i] = -λ[i]
  corr₂[i] = -0.5 * corr₁[i]^2 * real( (v_leftᵀ ddA v_right) / (v_leftᵀ dA v_right) )

Hence two sets of corrected wavenumbers: `k + corr₁` and `k + corr₁ + corr₂`. Tensions are `|corr₁|`
and `|corr₁ + corr₂|`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The EBIM solver config.
- `basis::Ba`: Basis function type.
- `pts::BoundaryPointsBIM`: Boundary geometry.
- `k`: Wavenumber for the eigenproblem.
- `kernel_fun`: A triple `(base, first, second)` or custom functions for kernel & derivatives.

# Returns
- `(λ_corrected_1, tens_1, λ_corrected_2, tens_2)`: 
   1. `λ_corrected_1 = k + corr₁` (1st-order),
   2. `tens_1 = abs(corr₁)`,
   3. `λ_corrected_2 = k + corr₁ + corr₂` (2nd-order),
   4. `tens_2 = abs(corr₁ + corr₂)`.
"""
function solve_DEBUG_w_2nd_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=(:default,:first,:second)) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun)
    λ,VR,VL=generalized_eigen_all(A,dA)
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    T=eltype(real.(λ))
    λ=real.(λ)
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        numerator=transpose(v_left)*ddA*v_right
        denominator=transpose(v_left)*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    λ_corrected_1=k.+corr_1
    λ_corrected_2=λ_corrected_1.+corr_2
    tens_1=abs.(corr_1)
    tens_2=abs.(corr_1.+corr_2)
    return λ_corrected_1,tens_1,λ_corrected_2,tens_2
end

### HELPERS ###

"""
    ebim_inv_diff(kvals::Vector{T}) where {T<:Real}

Computes the inverse of the differences between consecutive elements in `kvals`. This inverts the small differences between the ks very close to the correct eigenvalues and serves as a visual aid or potential criteria for finding missing levels.

# Arguments
- `kvals::Vector{T}`: A vector of values for which differences are calculated.

# Returns
- `Vector{T}`: The `kvals` vector excluding its last element.
- `Vector{T}`: The inverse of the differences between consecutive elements in `kvals`.
"""
function ebim_inv_diff(kvals::Vector{T}) where {T<:Real}
    kvals_diff=diff(kvals)
    kvals=kvals[1:end-1]
    return kvals,T(1.0)./kvals_diff
end

"""
    visualize_ebim_sweep(solver::ExpandedBoundaryIntegralMethod, basis::Ba, billiard::Bi, k1, k2; 
                         dk=(k)->(0.05*k^(-1/3))) where {Ba<:AbstractHankelBasis, Bi<:AbsBilliard}

Debugging Function to sweep through a range of `k` values and evaluate the smallest tension for each `k` using the EBIM method. This function identifies corrected `k` values based on the generalized eigenvalue problem and associated tensions, collecting those with the smallest tensions for further analysis.

# Usage
hankel_basis=AbstractHankelBasis()
@time ks_debug,tens_debug,ks_debug_small,tens_debug_small=QuantumBilliards.visualize_ebim_sweep(ebim_solver,hankel_basis,billiard,k1,k2;dk=dk)
scatter!(ax,ks_debug,log10.(tens_debug), color=:blue, marker=:xcross)
-> This gives a sequence of points that fall on a vertical line when close to an actual eigenvalue. 

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration for the EBIM method.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `billiard::Bi`: The billiard geometry, a subtype of `AbsBilliard`.
- `k1`: The initial value of `k` for the sweep.
- `k2`: The final value of `k` for the sweep.
- `dk::Function`: A function defining the step size as a function of `k` (default: `(k) -> (0.05 * k^(-1/3))`).

# Returns
- `Vector{T}`: All corrected `k` values with low tensions throughout the sweep (`ks_all`).
- `Vector{T}`: Inverse tension corresponding to `ks_all` (`tens_all`), which represent the inverse distances between consecutive `ks_all`. Aa large number indicates that we are probably close to an eigenvalue since solution of the ebim sweep tend to accumulate there.
"""
function visualize_ebim_sweep(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1,k2;dk=(k)->(0.05*k^(-1/3))) where {Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    k=k1
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    T=eltype(k1)
    ks_all_1=T[]
    ks_all_2=T[]
    tens_all_1=T[]
    tens_all_2=T[]
    ks=T[] # these are the evaluation points
    push!(ks,k1)
    k=k1
    while k<k2
        k+=dk(k)
        push!(ks,k)
    end
    @showprogress desc="EBIM smallest tens..." for k in ks
        pts=evaluate_points(bim_solver,billiard,k)
        ks1,tens1,ks2,tens2=solve_DEBUG_w_2nd_order_corrections(solver,basis,pts,k)
        idx1=findmin(tens1)[2]
        idx2=findmin(tens2)[2]
        if log10(tens1[idx1])<0.0
            push!(ks_all_1,ks1[idx1])
            push!(tens_all_1,tens1[idx1])     
        end
        if log10(tens2[idx2])<0.0
            push!(ks_all_2,ks2[idx2])
            push!(tens_all_2,tens2[idx2])
        end
    end
    _,logtens_1=ebim_inv_diff(ks_all_1)
    _,logtens_2=ebim_inv_diff(ks_all_2)
    idxs1=findall(x->x>0.0,logtens_1)
    idxs2=findall(x->x>0.0,logtens_2)
    logtens_1=logtens_1[idxs1]
    logtens_2=logtens_2[idxs2]
    ks_all_1=ks_all_1[idxs1]
    ks_all_2=ks_all_2[idxs2]
    return ks_all_1,logtens_1, ks_all_2,logtens_2
end






















#### LEGACY (WORKING) CODE ####

#=
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
    all_fredholm_associated_matrices(
        boundary_points::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T;kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {T<:Real} -> Matrix{Complex{T}}

Constructs the Fredholm matrix, the derivative and the second derivative with respect to the wavenumber.

# Arguments
- `boundary_points::BoundaryPointsBIM{T}`: Evaluated boundary points.
- `symmetry_rule::SymmetryRuleBIM{T}`: Symmetry rules for the matrix construction.
- `k::T`: Wavenumber.
- `kernel_fun::Function`: Function to use for the kernel computation (default for free 2D particle: `default_helmholtz_kernel`).
- `kernel_der_fun::Function=default_helmholtz_kernel_first_derivative`: Function to use for the 1st derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_first_derivative`).
- `kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative`: Function to use for the 2nd derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_second_derivative`).

# Returns
- `Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}`:
  - Fredholm matrix.
  - First derivative of the Fredholm matrix.
  - Second derivative of the Fredholm matrix.
"""
function all_fredholm_associated_matrices(boundary_points::BoundaryPointsBIM{T},symmetry_rule::SymmetryRuleBIM{T},k::T;kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {T<:Real}
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
    fredholm_matrix=Matrix{Complex{T}}(I,N,N)
    fredholm_matrix_der1=fill(Complex(0.0,0.0),N,N)
    fredholm_matrix_der2=fill(Complex(0.0,0.0),N,N)
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
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k;kernel_fun=kernel_fun)
            kernel_value_der=compute_kernel_derivative(
                p1,p2,normal1,curvature1,
                isnothing(reflected_points_x) ? nothing : reflected_points_x[j],
                isnothing(reflected_points_y) ? nothing : reflected_points_y[j],
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k;kernel_der_fun=kernel_der_fun)
            kernel_value_der2=compute_kernel_second_derivative(
                p1,p2,normal1,curvature1,
                isnothing(reflected_points_x) ? nothing : reflected_points_x[j],
                isnothing(reflected_points_y) ? nothing : reflected_points_y[j],
                isnothing(reflected_points_xy) ? nothing : reflected_points_xy[j],symmetry_rule,k;kernel_der2_fun=kernel_der2_fun)
            fredholm_matrix[i, j]-=ds1*kernel_value
            fredholm_matrix_der1[i, j]=-ds1*kernel_value_der
            fredholm_matrix_der2[i, j]=-ds1*kernel_value_der2
        end
    end
    return fredholm_matrix,fredholm_matrix_der1,fredholm_matrix_der2
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
    return all_fredholm_associated_matrices(pts,solver.rule,k;kernel_fun=kernel_fun,kernel_der_fun=kernel_der_fun,kernel_der2_fun=kernel_der2_fun)
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
    valid=(abs.(real.(λ)).<dk) .& (abs.(imag.(λ)).<dk) # use (-dk,dk) × (-dk,dk) instead of disc of radius dk
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
        numerator=transpose(v_left)*ddA*v_right
        denominator=transpose(v_left)*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    λ_corrected=k.+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    return λ_corrected,tens
end

### DEBUGGING TOOLS ###

"""
     solve_DEBUG_w_1st_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {Ba<:AbstractHankelBasis}

FIRST ORDER CORRECTIONS
Debugging function that solves for a given `k` the generalized eigenvalue problem `A * x = λ * (dA/dk) * x` for the `λs` which it then adds to the initial `k` at which the `eigen(A,dA/dk)` was performed and the associated tension (defined as the absolute value of `λ` as the quality of the correction).

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`. This is a placeholder, just use AbstractHankelBasis() for this input.
- `pts::BoundaryPointsBIM`: Boundary points in BIM representation. Use the evaluate_points with the regular BIM solver.
- `k`: The k for which we do the generalized eigenvalue problem.
- `kernel_fun::Function`: Function to use for the kernel computation (default for free 2D particle: `default_helmholtz_kernel`).
- `kernel_der_fun::Function=default_helmholtz_kernel_first_derivative`: Function to use for the 1st derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_first_derivative`).
- `kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative`: Function to use for the 2nd derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_second_derivative`).

# Returns
- `Vector{T}`: Corrected ks without further restrictions.
- `Vector{T}`: Associated tensions for the corrections to k.
"""
function solve_DEBUG_w_1st_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {Ba<:AbstractHankelBasis}
    A,dA,_=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,kernel_der_fun=kernel_der_fun,kernel_der2_fun=kernel_der2_fun)
    F=eigen(A,dA)
    λ=F.values
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    T=eltype(real.(λ))
    λ=real.(λ)
    corr_1=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        corr_1[i]=-λ[i]
    end
    λ_corrected=k.+corr_1
    tens=abs.(corr_1)
    return λ_corrected,tens
end

"""
     solve_DEBUG_w_2nd_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {Ba<:AbstractHankelBasis}

FIRST ORDER CORRECTIONS
Debugging function that solves for a given `k` the generalized eigenvalue problem `A * x = λ * (dA/dk) * x` for the `λs` which it then adds to the initial `k` at which the `eigen(A,dA/dk)` was performed and the associated tension (defined as the absolute value of `λ` as the quality of the correction).

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`. This is a placeholder, just use AbstractHankelBasis() for this input.
- `pts::BoundaryPointsBIM`: Boundary points in BIM representation. Use the evaluate_points with the regular BIM solver.
- `k`: The k for which we do the generalized eigenvalue problem.
- `kernel_fun::Function`: Function to use for the kernel computation (default for free 2D particle: `default_helmholtz_kernel`).
- `kernel_der_fun::Function=default_helmholtz_kernel_first_derivative`: Function to use for the 1st derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_first_derivative`).
- `kernel_der2_fun::Function=default_helmholtz_kernel_second_derivative`: Function to use for the 2nd derivative of the kernel computation (default for free 2D particle: `default_helmholtz_kernel_second_derivative`).

# Returns
- `Vector{T}`: First order corrections to k.
- `Vector{T}`: Associated tensions for the 1st order corrections to k.
- `Vector{T}`: 1st+2nd order corrections to k.
- `Vector{T}`: Associated tensions for the 1st+2nd order corrections to k.
"""
function solve_DEBUG_w_2nd_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=default_helmholtz_kernel,kernel_der_fun=default_helmholtz_kernel_first_derivative,kernel_der2_fun=default_helmholtz_kernel_second_derivative) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,kernel_der_fun=kernel_der_fun,kernel_der2_fun=kernel_der2_fun)
    λ,VR,VL=generalized_eigen_all(A,dA)
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    T=eltype(real.(λ))
    λ=real.(λ)
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        numerator=transpose(v_left)*ddA*v_right
        denominator=transpose(v_left)*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    λ_corrected_1=k.+corr_1
    λ_corrected_2=λ_corrected_1.+corr_2
    tens_1=abs.(corr_1)
    tens_2=abs.(corr_1.+corr_2)
    return λ_corrected_1,tens_1,λ_corrected_2,tens_2
end

=#







