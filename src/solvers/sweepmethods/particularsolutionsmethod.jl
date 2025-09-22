using LinearAlgebra, StaticArrays, TimerOutputs

"""
    struct ParticularSolutionsMethod{T} <: SweepSolver where {T<:Real}

A data structure to hold configuration parameters for the "Particular Solutions Method."
This method typically uses a scaled number of boundary and interior points based on
the wavenumber `k`. It also defines sampling strategies and minimal dimension settings.

# Fields
- `dim_scaling_factor::T`: A real scale factor for matrix dimension or basis dimension.
- `pts_scaling_factor::Vector{T}`: A list of scaling factors for boundary points across different curves.
- `int_pts_scaling_factor::T`: Scaling factor for interior points.
- `sampler::Vector`: A list of samplers (e.g. Gauss-Legendre) for boundary integration.
- `eps::T`: A small numeric epsilon, set to `eps(T)`.
- `min_dim::Int64`: The minimal dimension of a matrix or basis.
- `min_pts::Int64`: The minimal number of boundary points.
- `min_int_pts::Int64`: The minimal number of interior points.
"""
struct ParticularSolutionsMethod{T} <: SweepSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    int_pts_scaling_factor::T
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
    min_int_pts::Int64
end

"""
    ParticularSolutionsMethod(
        dim_scaling_factor::T,
        pts_scaling_factor::Union{T, Vector{T}},
        int_pts_scaling_factor::T;
        min_dim::Int=100,
        min_pts::Int=500,
        min_int_pts::Int=500
    ) -> ParticularSolutionsMethod{T}

Construct a `ParticularSolutionsMethod{T}` object with default Gauss-Legendre samplers.

# Arguments
- `dim_scaling_factor::T`: A scaling factor that controls basis dimension.
- `pts_scaling_factor::Union{T, Vector{T}}`: A scaling factor that controls final matrix dimension for boundary point discretization.
- `int_pts_scaling_factor::T`: Scaling factor for interior points. Usually use the same as the boundary point scaling factor.
- `min_dim::Int`: The minimal dimension used in computations (default 100).
- `min_pts::Int`: The minimal number of boundary points (default 500).
- `min_int_pts::Int`: The minimal number of interior points (default 500).

# Returns
- `ParticularSolutionsMethod{T}`: A new solver configuration with a Gauss-Legendre sampler
  in the `sampler` field, and `eps(T)` as the numeric epsilon.
"""
function ParticularSolutionsMethod(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, int_pts_scaling_factor::T; min_dim = 100, min_pts = 500, min_int_pts=500) where T<:Real 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[GaussLegendreNodes()]
    return ParticularSolutionsMethod(d,bs,int_pts_scaling_factor,sampler,eps(T),min_dim,min_pts,min_int_pts)
end

"""
    ParticularSolutionsMethod(
        dim_scaling_factor::T,
        pts_scaling_factor::Union{T,Vector{T}},
        int_pts_scaling_factor::T,
        samplers::Vector{Sam};
        min_dim::Int=100,
        min_pts::Int=500,
        min_int_pts=500
    ) where {T<:Real, Sam<:AbsSampler}

Construct a `ParticularSolutionsMethod{T}` object with user-provided samplers.

# Arguments
- `dim_scaling_factor::T`: A scaling factor that controls basis dimension.
- `pts_scaling_factor::Union{T, Vector{T}}`: A scaling factor that controls final matrix dimension for boundary point discretization.
- `int_pts_scaling_factor::T`: Scaling factor for interior points. Usually use the same as the boundary point scaling factor.
- `samplers::Vector{Sam}`: A vector of samplers (e.g. `GaussLegendreNodes()`, `PolarSampler()`, etc.).
- `min_dim::Int`: Minimal dimension (default 100).
- `min_pts::Int`: Minimal boundary points (default 500).
- `min_int_pts::Int`: Minimal interior points (default 500).

# Returns
- `ParticularSolutionsMethod{T}`: A solver configuration with the given `samplers` and other settings.
"""
function ParticularSolutionsMethod(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}},int_pts_scaling_factor::T,samplers::Vector{Sam};min_dim = 100,min_pts = 500,min_int_pts=500) where {T<:Real,Sam<:AbsSampler} 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return ParticularSolutionsMethod(d,bs,int_pts_scaling_factor,samplers,eps(T),min_dim,min_pts,min_int_pts)
end

"""
    struct PointsPSM{T} <: AbsPoints where {T<:Real}

Holds boundary and interior points `(x, y)` for the Particular Solutions Method.

# Fields
- `xy_boundary::Vector{SVector{2,T}}`: A list of 2D boundary points.
- `xy_interior::Vector{SVector{2,T}}`: A list of 2D interior points.
"""
struct PointsPSM{T} <: AbsPoints where {T<:Real}
    xy_boundary::Vector{SVector{2,T}}
    xy_interior::Vector{SVector{2,T}} #normal vectors in points
end

"""
    evaluate_points(solver::ParticularSolutionsMethod, billiard::Bi, k)

Generate boundary and interior points for the Particular Solutions Method. Uses the solver's scaling
factors and minimal point constraints to decide how many points to sample on each boundary segment
and how many interior points to sample.

# Arguments
- `solver::ParticularSolutionsMethod`: The PSM solver configuration.
- `billiard::Bi<:AbsBilliard`: A geometry or domain object holding boundary curves.
- `k::Real`: Wavenumber, used to scale the number of points (e.g. `k * L * scaling / (2π)`).

# Returns
- `PointsPSM{T}`: A struct containing:
  - `xy_boundary`: All boundary points appended from each curve.
  - `xy_interior`: Randomly sampled interior points.
"""
function evaluate_points(solver::ParticularSolutionsMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    b_int=solver.int_pts_scaling_factor
    curves=billiard.fundamental_boundary
    type=eltype(solver.pts_scaling_factor)
    xy_all=Vector{SVector{2,type}}()
    xy_int_all=Vector{SVector{2,type}}()
    for i in eachindex(curves)
        crv=curves[i]
        if typeof(crv)<:AbsRealCurve
            L=crv.length
            N=max(solver.min_pts,round(Int,k*L*bs[i]/(2*pi)))
            sampler=samplers[i]
            if crv isa PolarSegment && sampler isa PolarSampler
                t,dt=sample_points(sampler,crv,N)
            else
                t,dt=sample_points(sampler,N)
            end
            xy=curve(crv,t)
            append!(xy_all,xy)
        end
    end
    L=billiard.length
    M=max(solver.min_int_pts,round(Int,k*L*b_int/(2*pi)))
    xy_int_all=random_interior_points(billiard,M)
    return PointsPSM{type}(xy_all,xy_int_all)
end

"""
    construct_matrices_benchmark(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Construct the basis matrices for boundary points and interior points, with timing information.
This is a benchmarking variant that uses a `TimerOutput` to measure the time spent creating each
matrix. It prints the timings at the end.

# Arguments
- `solver::ParticularSolutionsMethod`: Solver config, specifying scaling factors.
- `basis::Ba<:AbsBasis`: A basis to evaluate (e.g. FourierBessel basis).
- `pts::PointsPSM`: The boundary/interior points to evaluate.
- `k::Real`: Wavenumber for the basis.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(B, B_int)`: A pair of matrices:
  - `B::Matrix`: The basis matrix evaluated at boundary points.
  - `B_int::Matrix`: The basis matrix evaluated at interior points.
"""
function construct_matrices_benchmark(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    to=TimerOutput()
    pts_bd=pts.xy_boundary
    pts_int=pts.xy_interior
    #basis and gradient matrices
    @timeit to "basis_matrices" begin
        @timeit to "boundary" B=basis_matrix(basis,k,pts_bd;multithreaded=multithreaded)
        @timeit to "interior" B_int=basis_matrix(basis,k,pts_int;multithreaded=multithreaded)
    end
    print_timer(to)
    return B,B_int  
end

"""
     construct_matrices(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Construct two matrices for the Particular Solutions Method: one for boundary points, one for interior points.
These represent the basis functions evaluated at the domain's boundary and interior.

# Arguments
- `solver::ParticularSolutionsMethod`: The PSM solver config.
- `basis::Ba<:AbsBasis`: A basis type implementing `basis_matrix(...)`.
- `pts::PointsPSM`: Contains boundary (`xy_boundary`) and interior (`xy_interior`) points.
- `k::Real`: Wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(B, B_int)`: 
  - `B::Matrix`: Basis matrix at boundary points.
  - `B_int::Matrix`: Basis matrix at interior points.
"""
function construct_matrices(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    pts_bd=pts.xy_boundary
    pts_int=pts.xy_interior
    B=basis_matrix(basis,k,pts_bd;multithreaded=multithreaded)
    B_int=basis_matrix(basis,k,pts_int;multithreaded=multithreaded)
    return B,B_int  
end


"""
    solve(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Solve the Particular Solutions Method by constructing `(B, B_int)` and computing a measure
(e.g. minimum singular value) that indicates how well the interior and boundary constraints
are satisfied via the `GeneralizedSVD` object. For mode reading it is based on the paper:
https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_112.pdf.

The idea is to represent the minimization of the boundary tension of the wavefunctions as a generalized eigenvalue problem `F * x = λ * G * x` as per the afformentioned paper.

# Arguments
- `solver::ParticularSolutionsMethod`: PSM solver config.
- `basis::Ba<:AbsBasis`: The basis (e.g. a trigonometric or radial basis).
- `pts::PointsPSM`: Boundary and interior points for evaluation.
- `k::Real`: Wavenumber or frequency-like parameter.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `::Real`: The minimum generalized singular value (or similar measure). Lower values can
  indicate a better "fit" to the PDE boundary conditions.
"""
function solve(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    B,B_int=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    solution=svdvals(B,B_int)
    return minimum(solution)
end

# INTERNAL
function solve_INFO(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    s=time()
    s_constr=time()
    @info "Constructing matrices"
    @time B,B_int=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @info "Conditioning cond(B) = $(cond(B)), cond(B_int) = $(cond(B_int))"
    e_constr=time()
    s_svd=time()
    @info "SVD values"
    @time solution=svdvals(B,B_int)
    e_svd=time()
    e=time()
    total_time=e-s
    @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("B & B_int construction: $(100*(e_constr-s_constr)/total_time) %")
    println("SVD: $(100*(e_svd-s_svd)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return minimum(solution)
end

"""
    solve(solver::ParticularSolutionsMethod,B::M,B_int::M) where {M<:AbstractMatrix}

A lower-level solver method that accepts already-constructed matrices `B` and `B_int` rather
than building them anew. Returns the minimum of `svdvals(B, B_int)`.

# Arguments
- `solver::ParticularSolutionsMethod`: PSM solver config (though this might be unused except for clarity).
- `B::AbstractMatrix{<:Complex}`: Basis matrix for boundary points.
- `B_int::AbstractMatrix{<:Complex}`: Basis matrix for interior points.

# Returns
- `::Real`: The minimum singular value from `svdvals(B, B_int)`.
"""
function solve(solver::ParticularSolutionsMethod,B::M,B_int::M) where {M<:AbstractMatrix}
    solution=svdvals(B,B_int)
    return minimum(solution)
end

"""
    solve_vect(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Compute the generalized singular value decomposition (SVD) for the boundary and interior matrices `(B, B_int)`,
and return both the smallest singular value and the associated vector `X` in the decomposition. 

# Arguments
- `solver::ParticularSolutionsMethod`: PSM solver configuration.
- `basis::Ba<:AbsBasis`: The basis object with a `basis_matrix` method.
- `pts::PointsPSM`: Points on boundary and interior.
- `k::Real`: Wavenumber/frequency parameter.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `σ_min::Real`: The smallest generalized singular value in the range.
- `x_min::AbstractVector{<:Complex}`: The vector corresponding to that smallest singular value,
  typically from the decomposition `H[idx, :]`.

# Methodology
1. **Matrix Construction**:
    - Construct matrices `B` and `B_int` from the solver, basis, points, and wavenumber `k`.
2. **GSVD Decomposition**:
    - Perform the generalized singular value decomposition (GSVD) of `B` and `B_int`, resulting in matrices `F.R - left singular vectors of B`, `F.Q - right singular vectors of B_int` (from the factorization object `F=GeneralizedSVD`), and singular values stored in `F.a` and `F.b` (similarly how we have to construct the generalized eigenvalues in ggev3! via λ=α./β in decompositions.jl here). Then we form the product matrix `H=F.R*F.Q'` which reconstructs the generalized eigenvectors.
3. **Compute Generalized Singular Values**:
    - Indices `idx=1:F.k+F.l` specify all the relevant generalized singular values (since we could have artefacts of overdetermined systems or numerical padding).
    - Generalized singular values are computed as the element-wise ratio `sv=F.a[idx]./F.b[idx]` as afformentioned above.
5. **Select the Optimal Solution**:
    - Take the smallest generalized singular value (`sv_min`) and its corresponding singular vector (`X_min`).
"""
function solve_vect(solver::ParticularSolutionsMethod,basis::Ba,pts::PointsPSM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    B,B_int=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    #=
    F=svd(B,B_int)
    H=F.R*F.Q'
    idx=1:F.k+F.l #inidices containing the singular values we need
    sv=F.a[idx]./F.b[idx] #generalized singular values
    X=H[idx,:]
    i_min=argmin(sv)
    return sv[i_min],X[i_min,:] 
    =#
    F=svd!(B,B_int) 
    n=size(B,2) 
    kF,lF=F.k,F.l
    idx=kF+1:kF+lF
    σ=F.a[idx]./F.b[idx]  
    jhat=idx[argmin(σ)]  
    jR=jhat-kF  
    J=n-(kF+lF)+1:n 
    ev=zeros(eltype(B),lF)
    ev[jR]=1
    z=F.R\ev 
    c=F.Q[:,J]*z # c is the eigenvector
    return minimum(σ),c
end