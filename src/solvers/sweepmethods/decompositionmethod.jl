using LinearAlgebra, StaticArrays, TimerOutputs
#TODO Logging.jl -> @debug
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
    BoundaryPointsMethod_to_BoundaryPoints(pts::BoundaryPointsDM{T}) where {T<:Real}

Convert a `BoundaryPointsDM{T}` to a `BoundaryPoints{T}` by extracting the coordinates, normals, arc-lengths, and integration weights.

# Arguments
- `pts::BoundaryPointsDM{T}`: The boundary points data structure containing coordinates, normals, and weights.

# Returns
- `BoundaryPoints{T}`: A new boundary points structure with the same coordinates, normals, arc-lengths, and integration weights.
"""
function BoundaryPointsMethod_to_BoundaryPoints(pts::BoundaryPointsDM{T}) where {T<:Real}
    xy=pts.xy
    normal=pts.normal
    ds=pts.w
    s=cumsum(ds)
    return BoundaryPoints{T}(xy,normal,s,ds)
end

"""
    construct_matrices_benchmark(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

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
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(F, G)`: Two matrices that represent boundary constraints (`F`) and normal derivative constraints (`G`).
  They are used later to solve a generalized eigenvalue problem.

**Note**: Prints out timing results of each sub-step via `print_timer(to)`.
"""
function construct_matrices_benchmark(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    t0=time()
    xy=pts.xy;w=pts.w;wn=pts.w_n;N=basis.dim
    nsym=isnothing(basis.symmetries) ? one(eltype(w)) : one(eltype(w))*(length(basis.symmetries)+1)
    t=time();B,dX,dY=basis_and_gradient_matrices(basis,k,xy;multithreaded)
    @info "basis_and_gradient_matrices" elapsed=(time()-t) sizeB=size(B) sizeDX=size(dX) sizeDY=size(dY)
    t=time()
    _scale_rows_sqrtw!(B,w,nsym)
    F=Matrix{eltype(B)}(undef,N,N)
    BLAS.syrk!('U','T',one(eltype(B)),B,zero(eltype(B)),F)
    _symmetrize_from_upper!(F)
    @info "F build (syrk)" elapsed=(time()-t)
    t=time()
    _build_Bn_inplace!(dX,dY,pts.normal)
    _scale_rows_sqrtw!(dX,wn,nsym)
    G=Matrix{eltype(B)}(undef,N,N)
    BLAS.syrk!('U','T',one(eltype(B)),dX,zero(eltype(B)),G)
    _symmetrize_from_upper!(G)
    @info "G build (normal+syrk)" elapsed=(time()-t)
    @info "construct_matrices_new_INFO total" elapsed=(time()-t0)
    return F,G  
end

"""
    construct_matrices(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Construct two matrices `F` and `G` for the Decomposition Method.

Further reading on why the helmholtz problem w/ Dirichlet BCs reduces to finding the maxima of a generalized eigevalues problem with F and G matrices see Barnett's PhD thesis chapter 5:
https://users.flatironinstitute.org/~ahb/thesis_html/node58.html

# Arguments
- `solver::DecompositionMethod`: Contains scaling factors, minimal points, etc.
- `basis::Ba<:AbsBasis`: Provides a method `basis_and_gradient_matrices(basis, k, points)`.
- `pts::BoundaryPointsDM`: Boundary points, normals, and weights.
- `k::Real`: The wavenumber used for constructing the basis and derivatives.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(F, G)`: 
  - `F`: Bboundary "norm" matrix with weighting `w`.
  - `G`: A normal-derivative "norm" matrix with weighting `w_n`.

**Note**: After this step, `(F, G)` can be used in a generalized eigenvalue problem.
"""
function construct_matrices(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    xy=pts.xy
    w=pts.w
    wn=pts.w_n
    N=basis.dim
    nsym=isnothing(basis.symmetries) ? one(eltype(w)) : one(eltype(w))*(length(basis.symmetries)+1)
    # the alogrithm consctructs B and the normal derivative Bn with syrk to minimize the allocation cost. It does this with the trick of putting sqrt(w_n) into both the rows of B and the rows of B' so that we can use syrk on sqrt(W)*B to get B'*(W*B) without forming W*B as a temporary matrix (posible b/c W is diagonal)
    B,dX,dY=basis_and_gradient_matrices(basis,k,xy;multithreaded)
    # Form F = B'*(W*B) by inplace scaling the rows of B by sqrt(w) (inplace to B) and use syrk to perform the Rank-k update of a symmetric matrix
    _scale_rows_sqrtw!(B,w,nsym) #  trick of putting sqrt(w_n) into the rows of the transposed and original B to get (sqrt(W)*B)' * (sqrt(W)*B) so we can use syrk on sqrt(W)*B
    F=Matrix{eltype(B)}(undef,N,N) # preallocate F
    BLAS.syrk!('U','T',one(eltype(B)),B,zero(eltype(B)),F) # F[u ∈ upper]+=1.0*B'*B, no need to fill(F,0) since the additive constant in C is 0
    _symmetrize_from_upper!(F) # since we chose "U" in syrk, we need to mirror upper -> lower
    # Build Bn into dX: dX <- nx*dX + ny*dY 
    _build_Bn_inplace!(dX,dY,pts.normal)
    # Form G = Bn'*(Wn*Bn) by first scaling the rows of Bn (dX) by sqrt(w_n) (inplace to dX) and use syrk to perform the Rank-k update of a symmetric matrix
    _scale_rows_sqrtw!(dX,wn,nsym) # like for F form sqrt(Wn*Bn) with row scaling with the same trick of putting sqrt(w_n) into the rows of the transposed and original dX to get (sqrt(Wn)*Bn)' * (sqrt(Wn)*Bn) so we can use syrk on dX
    G=Matrix{eltype(B)}(undef,N,N) # preallocate G, no need to fill with zeros since we use zero(eltype(B)) for the additive constant in syrk
    BLAS.syrk!('U','T',one(eltype(B)),dX,zero(eltype(B)),G) # G[u ∈ upper]+=1.0*dX'*dX where dX is now sqrt(Wn)*Bn due to inplace scalings above
    _symmetrize_from_upper!(G) # since we chose "U" in syrk, we need to mirror upper -> lower
    return F,G    
end

"""
    solve(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Solve the generalized eigenvalue problem given by the matrices `(F, G)` for the largest eigenvalue
`λ[end]`, then return its reciprocal. `t = 1 / λ[end]` represents the smallest possible reciprocal of the largest eigenvalue of the generalizedc eigenvalue problem and is thus a marker of tension. So this is what must be minimized in the sweep.
Further reading:
https://users.flatironinstitute.org/~ahb/thesis_html/node58.html

# Arguments
- `solver::DecompositionMethod`: The solver config, including `eps` for numerical tolerances.
- `basis::Ba<:AbsBasis`: Basis used to build matrices.
- `pts::BoundaryPointsDM`: Boundary data with points, normals, weights.
- `k::Real`: Wavenumber or frequency-like parameter.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `t::Float64`: The reciprocal of the largest eigenvalue from `generalized_eigvals(F, G)`, i.e. `1 / λ[end]`.
"""
function solve(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    F,G=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    mu=generalized_eigvals(Symmetric(F),Symmetric(G);eps=solver.eps)
    lam0=mu[end]
    t=1.0/lam0
    return t
end

# INTERNAL BENCHMARKS
function solve_INFO(solver::DecompositionMethod,basis::Ba,pts::BoundaryPointsDM,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    s_constr=time()
    @info "Constructing F,G for Fx=λGx..."
    @time F,G=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @info "Conditioning: cond(F) = $(cond(F)), cond(G) = $(cond(G))"
    e_constr=time()
    @info "Removing numerical nullspace of ill conditioned F and eigenvalue problem..."
    s_reg=time()
    @time d,S=eigen(Symmetric(F))
    e_reg=time()
    @info "Smallest & Largest eigval: $(extrema(d))"
    @info "Nullspace removal with criteria eigval > $(solver.eps*maximum(d))"
    idx=d.>solver.eps*maximum(d)
    @info "Dim of num Nullspace: $(count(!,idx))" # counts the number of falses = dim of nullspace
    q=1.0./sqrt.(d[idx])
    C=@view S[:,idx]
    C_scaled=C.*q'
    n=size(C_scaled,2)
    tmp=Matrix{eltype(G)}(undef,size(G,1),n)
    E=Matrix{eltype(G)}(undef,n,n)
    mul!(tmp,G,C_scaled)
    mul!(E,C_scaled',tmp)
    @warn "Final eigenvalue problem with new condition number: $(cond(E)) and reduced dimension $(size(E))"
    s_fin=time()
    @time mu=eigvals(Symmetric(E))
    e_fin=time()
    lam0=mu[end]
    t=1.0/lam0
    total_time=(e_fin-s_fin)+(e_reg-s_reg)+(e_constr-s_constr)
    @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("F & G construction: $(100*(e_constr-s_constr)/total_time) %")
    println("Nullspace removal: $(100*(e_reg-s_reg)/total_time) %")
    println("Final eigen problem: $(100*(e_fin-s_fin)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return t
end

"""
    solve(solver::DecompositionMethod,F::Matrix,G::Matrix)

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
    solve_vect(solver::DecompositionMethod,basis::AbsBasis,pts::BoundaryPointsDM,k;multithreaded::Bool=true)

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
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(t, x)`: 
  - `t = 1 / mu[end]`: The reciprocal of the largest eigenvalue. 
  - `x = x_vector ./ sqrt(mu[end])`: The corresponding eigenvector in the original basis.
"""
function solve_vect(solver::DecompositionMethod,basis::AbsBasis,pts::BoundaryPointsDM,k;multithreaded::Bool=true)
    F,G=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    mu,Z,C=generalized_eigen(Symmetric(F),Symmetric(G);eps=solver.eps)
    x=Z[:,end]
    x=C*x #transform into original basis 
    lam0=mu[end]
    t=1.0/lam0
    return t,x./sqrt(lam0)
end
