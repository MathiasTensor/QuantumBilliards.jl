################################################################################
# WEIGHTED PHYSICAL DENSITIES IN THE EXISTING SPECTRAL SOLVERS
#
# The ordinary CFIE represents ψ=-2(D+ikS)μ. Its unknown μ is a combined-layer
# density. WeightedCornerGrading instead represents ψ=S q, where q=∂ₙψ, and
# uses the adjoint equation
#
#                         B(k)q = [I-2K′(k)-2ikS(k)]q = 0.
#
# K′ differentiates the target normal without complex conjugation. Thus B(k)
# remains analytic in k, as required by Beyn, CORK and expanded BIM. The known
# wedge power q∼r^(π/α-1) belongs to this physical density, not to μ.
#
# weighted_corner_geometry.jl builds the angle-adapted interpolation space;
# weighted_corner_assembly.jl integrates its source cardinals and returns
# normalized Taylor matrices Bₗ=B⁽ˡ⁾(k₀)/l!. The methods below connect these
# same matrices to the existing spectral algorithms by solver-type dispatch.
# No alternate contour extraction, acceptance tolerance or root filter is
# introduced. The original periodic solver implementations are unchanged.
# A solver and its points must use the same density formulation; mismatched
# weighted and periodic discretizations are rejected before kernel assembly.
#
# Gaussian density nodes are not uniform periodic samples. Symmetry therefore
# acts on their actual coordinates through _fold_boundary(...,pts,...).
# Finally, a selected physical density is recovered by a dense LAPACK SVD:
# a small unconverged Krylov singular value alone did not certify its vector
# near corner-localized states. Contour extraction adds no per-root SVD.
################################################################################

const WeightedCFIE = CombinedFieldIntegralEquationSolver{T,G} where {T<:Real,G<:WeightedCornerGrading}

"""
    _cfie_evaluate_points(solver::WeightedCFIE, grading::WeightedCornerGrading,
        components::Vector, k::Real) → points::BoundaryPoints

Build local Gaussian nodes for the physical density `q=∂ₙψ`. The geometry
builder detects tangent angles, chooses their power coordinates, and retains
separate density and source-integration resolutions. `components` contains
the complete physical boundary; `k` sets the requested wavelength resolution.
"""
_cfie_evaluate_points(solver::WeightedCFIE, ::WeightedCornerGrading, components::Vector, k::T) where {T<:Real} = _weighted_corner_points(solver,components,k)

"""
    boundary_matrix_size(solver::WeightedCFIE, pts::BoundaryPoints) → size::Int

Count the weighted density unknowns after any symmetry reduction. Symmetry
orbits are constructed from the actual Gaussian nodes in `pts`, rather than
from the indices of a periodic grid.
"""
function boundary_matrix_size(solver::WeightedCFIE, pts::BoundaryPoints)
    solver.symmetry === nothing && return boundary_matrix_size(pts)
    return fundamental_size(_fold_boundary(_bim_numeric_type(solver),solver.billiard,
        pts,solver.symmetry,solver.sym_characters))
end

"""
    symmetrize_layer_density(solver::WeightedCFIE, density::AbstractVector{N},
        pts::BoundaryPoints{T}, billiard::AbsBilliard) → full_density::AbstractVector

Expand physical `q=∂ₙψ` using the character-weighted permutation of the actual
Gaussian nodes. An already full-length `density` is returned unchanged;
otherwise its length must equal the symmetry-reduced matrix dimension.
The output has one value per physical boundary node in `pts`.
"""
function symmetrize_layer_density(solver::WeightedCFIE, density::AbstractVector{N}, pts::BoundaryPoints{T}, billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull = length(pts)
    length(density) == Nfull && return density
    solver.symmetry === nothing && throw(DimensionMismatch("Boundary data has length $(length(density)); expected full length $Nfull because no symmetry is active"))
    orbits = _fold_boundary(T,billiard,pts,solver.symmetry,solver.sym_characters)
    Nred = fundamental_size(orbits)
    length(density) == Nred || throw(DimensionMismatch("Boundary data has length $(length(density)); expected reduced $Nred or full $Nfull"))
    full = Vector{promote_type(N,Complex{T})}(undef,Nfull)
    @inbounds for q in eachindex(full)
        full[q] = orbits.phase[q]*density[orbits.orbit_of[q]]
    end
    return full
end

"""
    construct_matrices(solver::WeightedCFIE, pts::BoundaryPoints, k::Number;
        multithreaded::Bool=true) → B::Matrix{ComplexF64}

Assemble `B(k)=I-2K′(k)-2ikS(k)` for physical `q=∂ₙψ`, using the local
weighted source cardinals in `pts`. `K′` uses the target normal; its analytic
continuation in complex `k` is not a conjugate transpose. The nodal identity
and the actual-node symmetry reduction are included. `multithreaded` enables
parallel construction while preserving the configured quadrature orders.
"""
construct_matrices(solver::WeightedCFIE, pts::BoundaryPoints, k; multithreaded::Bool=true) =
    first(_weighted_corner_matrices(solver,pts,k,0;multithreaded))

"""
    solve(solver::WeightedCFIE, pts::BoundaryPoints, k::Number;
        multithreaded::Bool=true, use_krylov::Bool=true) → tension::Float64

Return the smallest singular value of the weighted physical-density matrix.
LAPACK SVD is used for either value of the compatibility keyword `use_krylov`:
unconverged iterative singular triplets can give a false zero for this density
space. `multithreaded` controls matrix construction, not the SVD accuracy.
"""
function solve(solver::WeightedCFIE, pts::BoundaryPoints, k; multithreaded::Bool=true, use_krylov::Bool=true)
    A = construct_matrices(solver,pts,k;multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS values = svdvals(A)
    return values[end]
end

"""
    solve_vect(solver::WeightedCFIE, pts::BoundaryPoints, k::Number;
        multithreaded::Bool=true) → (tension::Float64, q::Vector{ComplexF64})

Recover the smallest right singular vector by LAPACK SVD. The vector stores
physical normal-density samples `q=∂ₙψ` on the full or symmetry-reduced nodes;
it has Euclidean unit norm before physical Rellich normalization. `tension`
is the corresponding smallest singular value of `B(k)`.
"""
function solve_vect(solver::WeightedCFIE, pts::BoundaryPoints, k; multithreaded::Bool=true)
    A = construct_matrices(solver,pts,k;multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS factorization = svd(A)
    return factorization.S[end],conj.(@view factorization.Vt[end,:])
end

"""
    solve_state(solver::WeightedCFIE, pts::BoundaryPoints{T}, k::Real,
        billiard::AbsBilliard; multithreaded::Bool=true)
        → (tension::T, q::Vector{Complex{T}}, u::Vector{Complex{T}}, norm::T)

Recover a selected physical normal density by SVD, restore its full-boundary
symmetry, then set `u=q/sqrt(norm)`. Here
`norm=∫(x⋅n)|q|² ds/(2k²)` is the Rellich normalization, evaluated with the
owned Jacobi norm weights in `pts.density_space`. The returned `q` is unnormalized;
`u` is the normalized boundary derivative. No combined-layer differentiation
is required because this formulation already solves for `∂ₙψ`.
"""
function solve_state(solver::WeightedCFIE, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    kT = T(k)
    tension,reduced = solve_vect(solver,pts,kT;multithreaded)
    density = symmetrize_layer_density(solver,reduced,pts,billiard)
    normalization = _rellich(pts,density,kT)
    return tension,density,density./sqrt(normalization),normalization
end

"""
    tune_cheb_config(solver::WeightedCFIE, pts::BoundaryPoints, k::Number,
        cfg::ChebyshevConfig) → cfg::ChebyshevConfig

Return `cfg` unchanged: weighted panels use their own local product rules,
not the periodic Kress kernel tables. The Chebyshev-enabled contour interface
therefore evaluates direct analytic matrices for this density formulation.
"""
tune_cheb_config(::WeightedCFIE, ::BoundaryPoints, k, cfg::ChebyshevConfig) = cfg

"""
    _construct_matrices_multi_k_cheb(solver::WeightedCFIE,
        pts::BoundaryPoints{T}, ks::Vector{ComplexF64}, cfg::ChebyshevConfig;
        multithreaded::Bool=true) → matrices::Vector{Matrix{ComplexF64}}

Evaluate the physical-density operator directly at every contour node in
`ks`, using its local source rules. Periodic Kress Chebyshev tables do not
apply to these panels; `cfg` is accepted for interface compatibility.
`multithreaded` controls the independent source-target construction.
"""
function _construct_matrices_multi_k_cheb(solver::WeightedCFIE, pts::BoundaryPoints{T}, ks::Vector{ComplexF64}, ::ChebyshevConfig; multithreaded::Bool=true) where {T<:Real}
    return [construct_matrices(solver,pts,k;multithreaded) for k in ks]
end

"""
    _horner(coefficients::AbstractVector{<:AbstractMatrix}, entry::Int, δ::Number)

Evaluate one linear-indexed entry of `Σₗ coefficients[l+1]*δ^l` without
allocating matrix temporaries. The read-only matrices contain normalized
Taylor coefficients; the returned scalar permits real or complex `δ=k-k₀`.
"""
@inline function _horner(coefficients::AbstractVector{<:AbstractMatrix}, entry::Int, δ::Number)
    value = coefficients[end][entry]
    @inbounds for l in length(coefficients)-1:-1:1
        value = muladd(δ,value,coefficients[l][entry])
    end
    return value
end

"""
    _horner_with_12_derivatives(coefficients::AbstractVector{<:AbstractMatrix},
        entry::Int, δ::Number) → (value, first, second)

Evaluate one matrix entry and its first two wavenumber derivatives. The
updates `ddv←δ*ddv+2dv`, `dv←δ*dv+v`, `v←δ*v+a` use the preceding values,
so normalized Taylor powers need no factorial conversion. Read-only inputs
and degrees zero or one are supported, with exact zero higher derivatives.
"""
@inline function _horner_with_12_derivatives(coefficients::AbstractVector{<:AbstractMatrix}, entry::Int, δ::Number)
    value = coefficients[end][entry]
    first = zero(value); second = zero(value)
    @inbounds for l in length(coefficients)-1:-1:1
        second = muladd(δ,second,2first)
        first = muladd(δ,first,value)
        value = muladd(δ,value,coefficients[l][entry])
    end
    return value,first,second
end

"""
    _construct_matrices_multi_k_taylor(solver::WeightedCFIE,
        pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number},
        p::Int; multithreaded::Bool=true) → matrices::Vector{Matrix{ComplexF64}}

Assemble the normalized Taylor matrices once and evaluate `B(k)` at every
requested real or complex `k` by allocation-free scalar Horner contraction.
`p` is the retained degree; `multithreaded` controls construction and the
independent matrix evaluations. Local integration and symmetry are already
included in each Taylor coefficient.
"""
function _construct_matrices_multi_k_taylor(solver::WeightedCFIE, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; multithreaded::Bool=true)::Vector{Matrix{ComplexF64}}
    coefficients = _weighted_corner_matrices(solver,pts,k0,p;multithreaded)
    values = [similar(first(coefficients)) for _ in ks]
    @use_threads multithreading=multithreaded for q in eachindex(ks)
        @inbounds for entry in eachindex(values[q])
            values[q][entry] = _horner(coefficients,entry,ks[q]-k0)
        end
    end
    return values
end

"""
    _construct_matrices_multi_k_taylor_derivatives(solver::WeightedCFIE,
        pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Real},
        p::Int; multithreaded::Bool=true) → (values, firsts, seconds)

Evaluate the same weighted Taylor polynomial and its first two wavenumber
derivatives at `ks`. Each output is a vector of `Matrix{ComplexF64}`; entries
with the same index refer to the same wavenumber. Source quadrature, degree
`p` and parallel scheduling match the value-only Taylor method.
"""
function _construct_matrices_multi_k_taylor_derivatives(solver::WeightedCFIE, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Real}, p::Int; multithreaded::Bool=true)
    coefficients = _weighted_corner_matrices(solver,pts,k0,p;multithreaded)
    values = [similar(first(coefficients)) for _ in ks]
    firsts = [similar(first(coefficients)) for _ in ks]
    seconds = [similar(first(coefficients)) for _ in ks]
    @use_threads multithreading=multithreaded for q in eachindex(ks)
        @inbounds for entry in eachindex(values[q])
            values[q][entry],firsts[q][entry],seconds[q][entry] =
                _horner_with_12_derivatives(coefficients,entry,ks[q]-k0)
        end
    end
    return values,firsts,seconds
end

"""
    _ebim_construct_matrices(solver::WeightedCFIE, pts::BoundaryPoints{T},
        k::Real; multithreaded::Bool=true) → (B, first, second)

Assemble `B(k)`, `B′(k)` and `B″(k)` for expanded BIM. The source recurrence
returns normalized Taylor matrices `Bₗ=B⁽ˡ⁾/l!`, so the second derivative is
`2B₂`. All outputs are full or symmetry-reduced `Matrix{ComplexF64}`;
`multithreaded` enables parallel source integration.
"""
function _ebim_construct_matrices(solver::WeightedCFIE, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    coefficients = _weighted_corner_matrices(solver,pts,k,2;multithreaded)
    return coefficients[1],coefficients[2],2coefficients[3]
end

"""
    EBIMTaylorCache(solver::WeightedCFIE, pts::BoundaryPoints{Float64},
        k0::Float64, p::Int; multithreaded::Bool=true) → cache::EBIMTaylorCache

Store the physical-density coefficients `B⁽ˡ⁾(k0)/l!` in the existing EBIM
cache layout, with coefficient order in its third dimension. `p≥2` retains
the two derivatives used by expanded BIM. The existing EBIM evaluation and
eigensolver then apply unchanged to this operator.
"""
function EBIMTaylorCache(solver::WeightedCFIE, pts::BoundaryPoints{Float64}, k0::Float64, p::Int; multithreaded::Bool=true)::EBIMTaylorCache
    p >= 2 || throw(ArgumentError("EBIM Taylor degree must be at least 2"))
    matrices = _weighted_corner_matrices(solver,pts,k0,p;multithreaded)
    return EBIMTaylorCache(k0,p,cat(matrices...;dims=3))
end

"""
    build_cork_polynomial(solver::WeightedCFIE, pts::BoundaryPoints,
        k0::Float64, Δ::Float64, p::Int; multithreaded::Bool=true)
        → (polynomial::CORKPolynomial, seconds::Float64)

Convert the normalized weighted Taylor matrices to the existing CORK layout:
`B(k0+Δt)≈Σⱼ CⱼTⱼ(t)`, with `Δ` the polynomial half-width and `p` its degree.
`power_to_cheb` supplies the scalar transformation; complete matrix blocks
are accumulated with BLAS. The returned time covers coefficient construction
and conversion, leaving the original CORK extraction and filtering unchanged.
"""
function build_cork_polynomial(solver::WeightedCFIE, pts::BoundaryPoints, k0::Float64, Δ::Float64, p::Int; multithreaded::Bool=true)::Tuple{CORKPolynomial,Float64}
    started = time()
    coefficients = _weighted_corner_matrices(solver,pts,k0,p;multithreaded)
    N = size(first(coefficients),1)
    C = power_to_cheb(p,Δ)
    B = zeros(ComplexF64,(p+1)*N,N)
    for j in 0:p, l in j:2:p
        axpy!(C[j+1,l+1],coefficients[l+1],@view(B[j*N+1:(j+1)*N,:]))
    end
    return CORKPolynomial(B,k0,Δ,p,N),time()-started
end
