using LinearAlgebra, StaticArrays, TimerOutputs

"""
    struct DecompositionMethod{T} <: SweepSolver where {T<:Real}

A solver configuration for a "Decomposition Method" that uses boundary integral for decomposition 
techniques. It containts scaling factors that control the number of boundary points as a function of the wavenumber `k`, and importantly the `samplers`.

# Fields
- `dim_scaling_factor::T`: A real scale factor for the dimension of the basis functions.
- `pts_scaling_factor::Vector{T}`: A list of scaling factors for boundary points (one per boundary curve).
- `sampler::Vector`: A list of sampling strategies (e.g. `GaussLegendreNodes(), LinearNodes(), PolarSampler()`) to discretize each curve.
- `eps::T`: A small tolerance (set to `eps(T)`) for the numerical nullspace when solving the generalized eigenvalue problem.
- `min_dim::Int64`: Minimal dimension for the decomposition (often a fallback).
- `min_pts::Int64`: Minimal number of boundary points to sample along each curve.
"""
struct DecompositionMethod{T} <: SweepSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end

"""
    DecompositionMethod(
        dim_scaling_factor::T,
        pts_scaling_factor::Union{T,Vector{T}};
        min_dim::Int=100,
        min_pts::Int=500
    ) -> DecompositionMethod{T}

Construct a `DecompositionMethod{T}` solver configuration with default Gauss-Legendre samplers.

# Arguments
- `dim_scaling_factor::T`: A real scale factor for the dimension of the basis functions.
- `pts_scaling_factor::Vector{T}`: A list of scaling factors for boundary points (one per boundary curve).
- `min_dim::Int`: The minimal dimension fallback (default=100).
- `min_pts::Int`: The minimal number of boundary points for each curve (default=500).

# Returns
- `DecompositionMethod{T}`: A solver with `eps(T)`, a default `GaussLegendreNodes()` sampler,
  and your specified scaling factors.
"""
function DecompositionMethod(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}};min_dim = 100,min_pts=500) where T<:Real 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[GaussLegendreNodes()]
return DecompositionMethod(d,bs,sampler,eps(T),min_dim,min_pts)
end

"""
    DecompositionMethod(
        dim_scaling_factor::T,
        pts_scaling_factor::Union{T,Vector{T}},
        samplers::Vector{Sam};
        min_dim::Int=100,
        min_pts::Int=500
    ) where {T<:Real, Sam<:AbsSampler}

Construct a `DecompositionMethod{T}` solver configuration using a custom vector of `samplers`.

# Arguments
- `dim_scaling_factor::T`: A real scale factor for the dimension of the basis functions.
- `pts_scaling_factor::Vector{T}`: A list of scaling factors for boundary points (one per boundary curve).
- `samplers::Vector{Sam}`: A vector of user-defined samplers (e.g. `GaussLegendreNodes()`, `PolarSampler()`, etc.).
- `min_dim::Int`: Minimal dimension fallback (default=100).
- `min_pts::Int`: Minimal number of boundary points (default=500).

# Returns
- `DecompositionMethod{T}`: A solver object storing the given samplers and the specified scale factors,
  with an internal `eps(T)`.
"""
function DecompositionMethod(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}},samplers::Vector{Sam};min_dim=100,min_pts=500) where {T<:Real,Sam<:AbsSampler} 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return DecompositionMethod(d,bs,samplers,eps(T),min_dim,min_pts)
end

"""
    struct BoundaryPointsDM{T} <: AbsPoints where {T<:Real}

Holds boundary data for the Decomposition Method, including boundary coordinates, normals,
and weighting arrays.

# Fields
- `xy::Vector{SVector{2,T}}`: Boundary points `(x, y)`.
- `normal::Vector{SVector{2,T}}`: Normal vectors at each boundary point.
- `w::Vector{T}`: Integration weights or "tension" weights for each point.
- `w_n::Vector{T}`: Additional normalization or weighted factor, defined as `rn=dot.(xy,normal); w_n=(w.*rn)./(2.0*k.^2)` as per the literature (link in `solve` function).
"""
struct BoundaryPointsDM{T} <: AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}} #normal vectors in points
    w::Vector{T} # tension weights
    w_n::Vector{T} #normalization weights
end

"""
    evaluate_points(solver::DecompositionMethod, billiard::Bi, k) -> BoundaryPointsDM{T}

Generate boundary points and associated normals/weights for the Decomposition Method. It:
1. Determines how many points to sample on each boundary curve, based on `k` and `pts_scaling_factor`.
2. Samples those points (e.g. via Gauss-Legendre or a polar sampler).
3. Computes normals, integration weights `w=ds`, and a secondary weight array involing the normals rn=dot.(xy,normal) and `w_n=(w.*rn)./(2.0*k.^2)`.

# Arguments
- `solver::DecompositionMethod`: The solver configuration, specifying scaling factors and minimal points.
- `billiard::Bi<:AbsBilliard`: The domain or geometry containing boundary curves.
- `k::Real`: The wavenumber/frequency-like parameter used to scale the number of points on each boundary.

# Returns
- `BoundaryPointsDM{T}`: A struct containing:
  - `xy`: All boundary points,
  - `normal`: Corresponding normal vectors,
  - `w`: Primary integration weights `w=ds`,
  - `w_n`: Auxiliary normalization weights `w_n=(w.*rn)./(2.0*k.^2)`.
"""
function evaluate_points(solver::DecompositionMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=billiard.fundamental_boundary
    type=eltype(solver.pts_scaling_factor)
    xy_all=Vector{SVector{2,type}}()
    normal_all=Vector{SVector{2,type}}()
    w_all=Vector{type}()
    w_n_all=Vector{type}()
    for i in eachindex(curves)
        crv=curves[i]
        if typeof(crv) <: AbsRealCurve
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
            rn=dot.(xy,normal)
            w=ds
            w_n=(w.*rn)./(2.0*k.^2) 
            append!(xy_all,xy)
            append!(normal_all,normal)
            append!(w_all,w)
            append!(w_n_all,w_n)
        end
    end
    return BoundaryPointsDM{type}(xy_all,normal_all, w_all, w_n_all)
end

"""
    construct_matrices_benchmark(
        solver::DecompositionMethod,
        basis::Ba,
        pts::BoundaryPointsDM,
        k
    ) -> (F, G)

Construct two matrices `F` and `G` for the Decomposition Method, with timing output. Uses `TimerOutput`
to benchmark the following:
1. Building basis and gradient matrices from the given `basis` at wavenumber `k`.
2. Applying boundary weights to form the final matrices `F` and `G`.

Further reading on why the helmholtz problem w/ Dirichlet BCs reduces to finding the maxima of a generalized eigevalues problem with F and G matrices see Barnett's PhD thesis chapter 5:
https://users.flatironinstitute.org/~ahb/thesis_html/node58.html

# Arguments
- `solver::DecompositionMethod`: The solver config for dimension scaling and minimal points.
- `basis::Ba<:AbsBasis`: A basis implementing `basis_and_gradient_matrices(...)`.
- `pts::BoundaryPointsDM`: Boundary data (points, normals, weights).
- `k::Real`: Wavenumber/frequency parameter.

# Returns
- `(F, G)`: Two matrices that represent boundary constraints (`F`) and normal derivative constraints (`G`).
  They are used later to solve a generalized eigenvalue problem.

**Note**: Prints out timing results of each sub-step via `print_timer(to)`.
"""
function construct_matrices_benchmark(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k) where {Ba<:AbsBasis}
    to=TimerOutput()
    w=pts.w
    w_n=pts.w_n
    symmetries=basis.symmetries
    if ~isnothing(symmetries)
        norm=(length(symmetries)+1.0)
        w=w.*norm
        w_n=w_n.*norm
    end
    #basis and gradient matrices
    @timeit to "basis_and_gradient_matrices" B,dX,dY=basis_and_gradient_matrices(basis,k,pts.xy)
    N=basis.dim
    type=eltype(B)
    F=zeros(type,(N,N))
    G=similar(F)
    @timeit to "F construction" begin 
        @timeit to "weights" T=(w.*B) #reused later
        @timeit to "product" mul!(F,B',T) #boundary norm matrix
    end

    @timeit to "G construction" begin 
        @timeit to "normal derivative" nx=getindex.(pts.normal,1)
        @timeit to "normal derivative" ny=getindex.(pts.normal,2)
        #inplace modifications
        @timeit to "normal derivative" dX=nx.*dX 
        @timeit to "normal derivative" dY=ny.*dY
        #reuse B
        @timeit to "normal derivative" B=dX.+dY
        #B is now normal derivative matrix (u function)
        @timeit to "weights" T=(w_n.*B) #apply integration weights
        @timeit to "product" mul!(G,B',T)#norm matrix
    end
    print_timer(to)
    return F,G    
end

"""
    construct_matrices(
        solver::DecompositionMethod,
        basis::Ba,
        pts::BoundaryPointsDM,
        k
    ) -> (F, G)

Construct two matrices `F` and `G` for the Decomposition Method.

Further reading on why the helmholtz problem w/ Dirichlet BCs reduces to finding the maxima of a generalized eigevalues problem with F and G matrices see Barnett's PhD thesis chapter 5:
https://users.flatironinstitute.org/~ahb/thesis_html/node58.html

# Arguments
- `solver::DecompositionMethod`: Contains scaling factors, minimal points, etc.
- `basis::Ba<:AbsBasis`: Provides a method `basis_and_gradient_matrices(basis, k, points)`.
- `pts::BoundaryPointsDM`: Boundary points, normals, and weights.
- `k::Real`: The wavenumber used for constructing the basis and derivatives.

# Returns
- `(F, G)`: 
  - `F`: Bboundary "norm" matrix with weighting `w`.
  - `G`: A normal-derivative "norm" matrix with weighting `w_n`.

**Note**: After this step, `(F, G)` can be used in a generalized eigenvalue problem.
"""
function construct_matrices(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k) where {Ba<:AbsBasis}
    #basis and gradient matrices
    w=pts.w
    w_n=pts.w_n
    symmetries=basis.symmetries
    if ~isnothing(symmetries)
        norm=(length(symmetries)+1.0)
        w=w.*norm
        w_n=w_n.*norm
    end
    B,dX,dY=basis_and_gradient_matrices(basis,k,pts.xy)
    type=eltype(B)
    N=basis.dim
    F=zeros(type,(N,N))
    G=similar(F)
    #apply weights
    T=(w.*B) #reused later
    mul!(F,B',T) #boundary norm matrix
    nx=getindex.(pts.normal,1)
    ny=getindex.(pts.normal,2)
    #inplace modifications
    dX=nx.*dX 
    dY=ny.*dY
    #reuse B
    B=dX.+dY
    #B is now normal derivative matrix (u function)
    T=(w_n.*B) #apply integration weights
    mul!(G,B',T)#norm matrix
    return F,G      
end

"""
    solve(solver::DecompositionMethod, basis::Ba, pts::BoundaryPointsDM, k) -> Float64

Solve the generalized eigenvalue problem given by the matrices `(F, G)` for the largest eigenvalue
`λ[end]`, then return its reciprocal. `t = 1 / λ[end]` represents the smallest possible reciprocal of the largest eigenvalue of the generalizedc eigenvalue problem and is thus a marker of tension. So this is what must be minimized in the sweep.
Further reading:
https://users.flatironinstitute.org/~ahb/thesis_html/node58.html

# Arguments
- `solver::DecompositionMethod`: The solver config, including `eps` for numerical tolerances.
- `basis::Ba<:AbsBasis`: Basis used to build matrices.
- `pts::BoundaryPointsDM`: Boundary data with points, normals, weights.
- `k::Real`: Wavenumber or frequency-like parameter.

# Returns
- `t::Float64`: The reciprocal of the largest eigenvalue from `generalized_eigvals(F, G)`, i.e. `1 / λ[end]`.
"""
function solve(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k) where {Ba<:AbsBasis}
    F,G=construct_matrices(solver,basis,pts,k)
    mu=generalized_eigvals(Symmetric(F),Symmetric(G);eps=solver.eps)
    lam0=mu[end]
    t=1.0/lam0
    return  t
end

"""
    solve(solver::DecompositionMethod, basis::Ba, pts::BoundaryPointsDM, k) -> Float64

Solve the generalized eigenvalue problem given by the matrices `(F, G)` for the largest eigenvalue. This implementation does not internally construct the F and G matrices but must be provided. Should be avoided in favor of `solve(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k)`
`λ[end]`, then return its reciprocal. `t = 1 / λ[end]` represents the smallest possible reciprocal of the largest eigenvalue of the generalizedc eigenvalue problem and is thus a marker of tension. So this is what must be minimized in the sweep.
Further reading:
https://users.flatironinstitute.org/~ahb/thesis_html/node58.html

# Arguments
- `solver::DecompositionMethod`: The solver config, including `eps` for numerical tolerances.
- `F::Matrix`: The boundary matrix.
- `G::Matrix`: The normal-derivative matrix.

# Returns
- `t::Float64`: The reciprocal of the largest eigenvalue from `generalized_eigvals(F, G)`, i.e. `1 / λ[end]`.
"""
function solve(solver::DecompositionMethod,F,G)
    mu=generalized_eigvals(Symmetric(F),Symmetric(G);eps=solver.eps)
    lam0=mu[end]
    t=1.0/lam0
    return t
end

"""
    solve_vect(
        solver::DecompositionMethod,
        basis::AbsBasis,
        pts::BoundaryPointsDM,
        k
    ) -> (t::Float64, x::AbstractVector)

Similar to `solve`, but also returns the associated eigenvector. This function:
1. Constructs `(F, G)`.
2. Computes the generalized eigenpairs `mu, Z, C` via `generalized_eigen(Symmetric(F), Symmetric(G))` where mu is the generalized eigenvalue, Z is the right eigenvector of the problem (column wise for each eigenvalue) and C is the transformation matrix after discsrding the numerical nullspace as per Barnett's PhD thesis. Further reading:
https://users.flatironinstitute.org/~ahb/thesis_html/node58.html
3. Extracts the largest eigenvalue `mu[end]` and the corresponding eigenvector (transformed back with `C*x`).
4. Returns the tension `t = 1 / mu[end]` and a scaled vector `x ./ sqrt(mu[end])` that represents the coefficients of expansion in the used basis for F and G.

# Arguments
- `solver::DecompositionMethod`: The solver config.
- `basis::AbsBasis`: Basis object used in matrix construction.
- `pts::BoundaryPointsDM`: Boundary data with points, normals, weights.
- `k::Real`: Wavenumber or parameter.

# Returns
- `(t, x)`: 
  - `t = 1 / mu[end]`: The reciprocal of the largest eigenvalue. 
  - `x = x_vector ./ sqrt(mu[end])`: The corresponding eigenvector in the original basis.
"""
function solve_vect(solver::DecompositionMethod,basis::AbsBasis,pts::BoundaryPointsDM,k)
    F,G=construct_matrices(solver,basis,pts,k)
    mu,Z,C=generalized_eigen(Symmetric(F),Symmetric(G);eps=solver.eps)
    x=Z[:,end]
    x=C*x #transform into original basis 
    lam0=mu[end]
    t=1.0/lam0
    return t,x./sqrt(lam0)
end
