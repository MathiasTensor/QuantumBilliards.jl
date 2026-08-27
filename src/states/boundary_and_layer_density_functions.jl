"""
    regularize!(u::AbstractVector{T}) where {T<:Number} → Nothing

Regularize a boundary function in place by replacing `NaN` entries with the
average of their two neighboring boundary values.
The boundary data are interpreted periodically, so the neighbors of the first
entry are the last and second entries, while the neighbors of the last entry
are the penultimate and first entries.

## Arguments
* `u::AbstractVector{T}`: Boundary-function values to regularize.

## Returns
* `nothing`: `u` is modified in place.
"""
@inline function regularize!(u::AbstractVector{T}) where {T<:Number}
    idx=findall(isnan,u)
    N=length(u)
    for i in idx
        im=i==1 ? N : i-1
        ip=i==N ? 1 : i+1
        u[i]=(u[ip]+u[im])/2
    end
    return nothing
end

"""
    _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {T<:Real,N<:Number} → T

Compute the Rellich normalization integral for a Dirichlet eigenfunction from
its boundary normal derivative.

For a Dirichlet eigenfunction the Rellich identity gives

    ∫_Ω |ψ|² dx = (1/(2k²)) ∫_∂Ω (x ⋅ n)|∂ₙψ|² ds.

The physical boundary quadrature weights are taken directly from `pts.ds`.

## Arguments
* `pts::BoundaryPoints{T}`: Boundary points containing positions, outward normals and physical quadrature weights.
* `u::AbstractVector{N}`: Boundary normal derivative values `∂ₙψ`.
* `k::T`: Wavenumber.

## Returns
* `norm::T`: Approximation of the interior norm `∫Ω|ψ|²dx`.
"""
function _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {T<:Real,N<:Number}
    acc=zero(T)
    @inbounds @simd for i in eachindex(u)
        n=pts.normal[i]
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i]
        acc+=w*abs2(u[i])
    end
    return acc/(2*k^2)
end

###########################################################################
################ BOUNDARY FUNCTION FOR BASIS TYPE SOLVERS #################
###########################################################################

"""
    boundary_function(state::Eigenstate;b::Real=5.0) → Tuple

Construct the Rellich-normalized boundary normal derivative of a basis-expanded
eigenstate.

The symmetry-adapted basis is evaluated directly on the physical fundamental
boundary and the basis gradients are contracted with the outward normal,

    u(s) = ∂ₙψ(s) = nₓ(s)∂ₓψ(s) + nᵧ(s)∂ᵧψ(s).

It is then fully symmetry-expanded to the complete physical boundary.

This method is intended for basis-type solvers such as
`VerginiSaracenoSolver`, `DecompositionMethodSolver` and
`ParticularSolutionsMethod`.

## Arguments
* `state::Eigenstate`: Basis-expanded eigenstate.

## Keyword Arguments
* `b::Real=5.0`: Boundary sampling density in points per de Broglie wavelength.

## Returns
* `u::Vector`: Rellich-normalized boundary normal derivative.
* `pts::BoundaryPoints`: Full-boundary discretization corresponding to `u`.
"""
function boundary_function(state::Eigenstate;b::Real=5.0)
    vec=state.vec
    k=state.k
    k_basis=state.k_basis
    new_basis=state.basis
    billiard=state.billiard
    T=eltype(vec)
    boundary=BilliardGeometry.get_boundary_curves_with_ignored(billiard)
    crv_lengths=[crv.length for crv in boundary]
    sampler=FourierNodes([2,3,5],crv_lengths) 
    L=CompositeCurve(boundary).length
    N=max(round(Int,k*L*b/(2π)),512)
    pts=boundary_coords(billiard,sampler,N)
    @blas_1 dX,dY=gradient_matrices(new_basis,k_basis,pts.xy) # ∂xϕ, ∂yϕ evaluated on pts.xy 
    M=size(dX,1)
    tX=Vector{T}(undef,M) # tX = (∂xϕ)(x_i), always real since the basis is real.
    tY=Vector{T}(undef,M) # tY = (∂yϕ)(x_i), always real since the basis is real.
    u=Vector{T}(undef,M) # u  = ∂nϕ(x_i), always real since the basis is real and the normal is real.
    # 2 GEMVs into empty then fuse normal-combination: ∂_n ϕ = nx ∂_x ϕ + ny ∂_y ϕ
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(tX,dX,vec) # tX = dX*vec
        mul!(tY,dY,vec) # tY = dY*vec
    end
    @fastmath @inbounds @simd for i in 1:M # fuse u = nx.*tX .+ ny.*tY in one loop
        n=pts.normal[i]
        u[i]=muladd(n[2],tY[i],n[1]*tX[i]) # u = n_x tX + n_y tY via muladd
    end
    regularize!(u)
    pts=apply_symmetries_to_boundary_points(pts,new_basis.symmetries)
    u=apply_symmetries_to_boundary_function(u,new_basis.symmetries)
    nrlz=_rellich(pts,u,k) # Rellich boundary norm: ∫ |u|^2 (n·x) ds / (2k^2) no temps
    @blas_1 return u./sqrt(nrlz),pts
end

"""
    boundary_function(state_data::StateData,billiard::Bi,basis::Ba;b::Real=5.0) where {Bi<:AbsBilliard,Ba<:AbsBasis} → Tuple

Construct boundary functions and [`BoundaryPoints`](@ref) discretizations for
all states stored in `state_data`. This struct comes from `VerginiSaracenoSolver`'s `compute_spectrum` method.

For every stored state, the basis is resized to the corresponding wavenumber,
an [`Eigenstate`](@ref) is constructed, and the single-state
[`boundary_function`](@ref) method is used.

## Arguments
* `state_data::StateData`: Stored wavenumbers, tensions and basis coefficients.
* `billiard::Bi`: Billiard geometry.
* `basis::Ba`: Basis used for the stored states.

## Keyword Arguments
* `b::Real=5.0`: Boundary sampling density.

## Returns
* `ks`: Wavenumbers for successfully constructed states.
* `us_all`: Boundary functions corresponding to `ks`.
* `pts_all`: Boundary discretizations corresponding to `ks`.
"""
function boundary_function(state_data::StateData,billiard::Bi,basis::Ba;b::Real=5.0) where {Bi<:AbsBilliard,Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPoints{eltype(ks)}}(undef,length(ks))
    valid_indices=fill(true,length(ks))
    progress=Progress(length(ks);desc="Constructing the u(s)...")
    for i in eachindex(ks)
        try
            vec=X[i]
            dim=length(vec)
            dim=rescale_dimension(basis,dim)
            new_basis=resize_basis(basis,billiard,dim,ks[i])
            state=Eigenstate(ks[i],vec,tens[i],new_basis,billiard)
            u,pts=boundary_function(state;b=b)
            us_all[i]=u
            pts_all[i]=pts
        catch e
            println("Error while constructing the u(s) for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(progress)
    end
    ks=ks[valid_indices]
    us_all=us_all[valid_indices]
    pts_all=pts_all[valid_indices]
    return ks,us_all,pts_all
end

#########################################################################################
################################ MOMENTUM FUNCTION ######################################
#########################################################################################

"""
    momentum_function(u::AbstractVector{U},s::AbstractVector{T},ds::AbstractVector{T};rtol::Real=100eps(T)) where {U<:Number,T<:Real} → Tuple

Compute the one-sided boundary momentum distribution.

## Arguments
* `u::AbstractVector{U}`: Boundary-function values.
* `s::AbstractVector{T}`: Physical arclength coordinates.
* `ds::AbstractVector{T}`: Physical quadrature weights.

## Keyword Arguments
* `rtol::Real=100eps(T)`: Relative tolerance used to detect uniform spacing.

## Returns
* `power`: One-sided boundary momentum weight `|c_m|²`.
* `ks`: Corresponding angular boundary wavenumbers.
"""
function momentum_function(u::AbstractVector{U},s::AbstractVector{T},ds::AbstractVector{T};rtol::Real=100*eps(T)) where {U<:Number,T<:Real}
    N=length(u)
    N==length(s)==length(ds)||throw(DimensionMismatch("u, s and ds must have equal length"))
    # Check whether the physical arclength nodes are uniformly spaced.
    # If so, the Fourier coefficients can be computed with FFTW.
    Δs=s[2]-s[1]
    uniform=all(isapprox(s[i+1]-s[i],Δs;rtol=rtol,atol=eps(T)*max(one(T),abs(Δs))) for i in 2:N-1)
    if uniform && U<:Real
        # Uniform real data: use the one-sided real FFT. Division by N gives
        # the same Fourier-integral normalization as the nonuniform branch.
        fu=FFTW.rfft(u)./N
        ks=FFTW.rfftfreq(N,inv(Δs)).*(2*pi)
        power=abs2.(fu)
        # Negative modes have equal power; double positive modes except Nyquist.
        lastdouble=iseven(N) ? length(power)-1 : length(power)
        lastdouble>=2 && (power[2:lastdouble].*=2)
        return power,ks
    end
    # Nonuniform arclength nodes: evaluate c_m=(1/L)∫u(s)exp(-ik_m*s)ds
    # directly with the physical quadrature weights ds.
    L=sum(ds)
    M=div(N,2) # same number of nonnegative modes as rfft
    ks=T.(2*pi/L.*(0:M))
    power=Vector{T}(undef,M+1)
    @fastmath @inbounds for m in 0:M
        km=ks[m+1]
        acc=zero(Complex{T})
        @simd for j in eachindex(u)
            acc+=u[j]*cis(-km*s[j])*ds[j]
        end
        power[m+1]=abs2(acc/L)
    end
    # For real u, negative modes contain the same power as positive modes.
    # Double positive frequencies, except the Nyquist mode for even N.
    if U<:Real
        lastdouble=iseven(N) ? M : M+1
        lastdouble>=2 && (power[2:lastdouble].*=2)
    end
    return power,ks
end

function momentum_function(u::AbstractVector{U},pts::BoundaryPoints{T};kwargs...) where {U<:Number,T<:Real}
    return momentum_function(u,pts.s,pts.ds;kwargs...)
end

"""
    momentum_function(state::S;b::Real = 5.0) where {S<:AbsState} → (power::Vector, ks::Vector)

Computes the momentum-space representation of an eigenstate's boundary
function directly from `state`, by combining [`boundary_function`](@ref) and
[`momentum_function(u, s)`](@ref momentum_function).

## Arguments
* `state`: The eigenstate for which the boundary momentum function is computed.

## Keyword arguments
* `b::Real = 5.0`: Oversampling factor passed to [`boundary_function`](@ref)
  controlling the boundary point density.

## Returns
* `power`: The normalized boundary momentum weight `|c_m|²`.
* `ks`: The angular wavenumbers corresponding to `power`.
"""
function momentum_function(state::Eigenstate;b::Real=5.0)
    u,pts=boundary_function(state;b=b)
    return momentum_function(u,pts)
end

#########################################################################################################
################### BoundaryIntegralMethod, DLP_kress and DLP_kress_global_corners ######################
#########################################################################################################

#######################################################################
###################### SYMMETRY ORBIT EXPANSION #######################
#######################################################################

"""
    symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand symmetry-reduced boundary data onto the complete physical boundary.

`pts` already represents the full physical boundary. If `layer_density` already
has full-boundary length, it is returned unchanged. Otherwise the active
symmetry orbit map is used to expand the reduced data.
"""
function symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull=length(pts)
    length(layer_density)==Nfull&&return pts,layer_density
    isnothing(solver.symmetry)&&throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    Ifund,full_to_fund,full_to_scale,_,_=symmetry_index_orbits(T,pts,solver.symmetry,billiard)
    Nred=length(Ifund)
    length(layer_density)==Nred||throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S=promote_type(N,Complex{T})
    full_data=Vector{S}(undef,Nfull)
    @inbounds for q in 1:Nfull
        full_data[q]=full_to_scale[q]*layer_density[full_to_fund[q]]
    end
    return pts,full_data
end

"""
    symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Batch version of [`symmetrize_layer_density`](@ref) for `BoundaryIntegralMethod`.
"""
function symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=symmetrize_layer_density(solver,layer_density[i],pts[i],billiard)
    end
    return pts_all,us_all
end

"""
    symmetrize_layer_density(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand symmetry-reduced DLP-Kress boundary data onto the complete physical
boundary. Full-length input is returned silently unchanged.
"""
function symmetrize_layer_density(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull=length(pts)
    length(layer_density)==Nfull&&return pts,layer_density
    isnothing(solver.symmetry)&&throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    Ifund,full_to_fund,full_to_scale,_,_=symmetry_index_orbits(T,pts,solver.symmetry,billiard)
    Nred=length(Ifund)
    length(layer_density)==Nred||throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S=promote_type(N,Complex{T})
    full_data=Vector{S}(undef,Nfull)
    @inbounds for q in 1:Nfull
        full_data[q]=full_to_scale[q]*layer_density[full_to_fund[q]]
    end
    return pts,full_data
end

"""
    symmetrize_layer_density(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Batch version of DLP-Kress symmetry expansion.
"""
function symmetrize_layer_density(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=symmetrize_layer_density(solver,layer_density[i],pts[i],billiard)
    end
    return pts_all,us_all
end

"""
    symmetrize_layer_density(solver::Union{CFIE_kress,CFIE_kress_global_corners,CFIE_kress_corners},layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand symmetry-reduced CFIE-Kress boundary data onto the complete physical
boundary. Full-length input is returned silently unchanged.
"""
function symmetrize_layer_density(solver::Union{CFIE_kress,CFIE_kress_global_corners,CFIE_kress_corners},layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull=length(pts)
    length(layer_density)==Nfull&&return pts,layer_density
    isnothing(solver.symmetry)&&throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    ws=build_cfie_kress_workspace(solver,pts)
    Nred=ws.Nred
    length(layer_density)==Nred||throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S=promote_type(N,Complex{T})
    full_data=Vector{S}(undef,ws.Ntot)
    @inbounds for g in 1:ws.Ntot
        full_data[g]=ws.full_to_scale[g]*layer_density[ws.full_to_fund[g]]
    end
    return pts,full_data
end

# Internal workspace overload used when the CFIE workspace already exists.
function symmetrize_layer_density(solver::Union{CFIE_kress,CFIE_kress_global_corners,CFIE_kress_corners},layer_density::AbstractVector{N},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T}) where {N<:Number,T<:Real}
    Nfull=ws.Ntot
    length(layer_density)==Nfull&&return pts,layer_density
    isnothing(solver.symmetry)&&throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    Nred=ws.Nred
    length(layer_density)==Nred||throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S=promote_type(N,Complex{T})
    full_data=Vector{S}(undef,Nfull)
    @inbounds for g in 1:Nfull
        full_data[g]=ws.full_to_scale[g]*layer_density[ws.full_to_fund[g]]
    end
    return pts,full_data
end

"""
    symmetrize_layer_density(solver::Union{CFIE_kress,CFIE_kress_global_corners,CFIE_kress_corners},layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Batch version of CFIE-Kress symmetry expansion.
"""
function symmetrize_layer_density(solver::Union{CFIE_kress,CFIE_kress_global_corners,CFIE_kress_corners},layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=symmetrize_layer_density(solver,layer_density[i],pts[i],billiard)
    end
    return pts_all,us_all
end

#######################################################################
######################## RELLICH NORMALIZATION ########################
#######################################################################

"""
    boundary_function(solver::BoundaryIntegralMethod,layer_density::AbstractVector{N},pts::BoundaryPoints{T},k::T,billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand the boundary density to the full physical boundary when necessary and
Rellich-normalize it.
"""
function boundary_function(solver::BoundaryIntegralMethod,layer_density::AbstractVector{N},pts::BoundaryPoints{T},k::T,billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts,layer_density=symmetrize_layer_density(solver,layer_density,pts,billiard)
    nrlz=_rellich(pts,layer_density,k)
    return pts,layer_density./sqrt(nrlz)
end

"""
    boundary_function(solver::BoundaryIntegralMethod,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Batch boundary-function construction for `BoundaryIntegralMethod`.
"""
function boundary_function(solver::BoundaryIntegralMethod,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=boundary_function(solver,layer_density[i],pts[i],ks[i],billiard)
    end
    return pts_all,us_all
end

"""
    boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi,k::T) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand a DLP-Kress density to the full physical boundary when necessary and
Rellich-normalize it.
"""
function boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi,k::T) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts,layer_density=symmetrize_layer_density(solver,layer_density,pts,billiard)
    nrlz=_rellich(pts,layer_density,k)
    return pts,layer_density./sqrt(nrlz)
end

"""
    boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Batch DLP-Kress boundary-function construction.
"""
function boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=boundary_function(solver,layer_density[i],pts[i],billiard,ks[i])
    end
    return pts_all,us_all
end

#####################################################
################### CFIE_kress ######################
#####################################################

# TODO Corner-graded Kress meshes remain periodic in the computational
# variable, but the grading map has strongly nonuniform speed near corners.
# We need more verification of spectral tangential differentiation and the regularity of the
# transformed CFIE density before enabling this pathway.

"""
    periodic_derivative_t(f::AbstractVector{Complex{T}}) where {T<:Real} → Vector{Complex{T}}

Compute the derivative with respect to the periodic computational
parameter `t`.

For equispaced periodic nodes

    tⱼ = 2πj/N,

a sampled periodic function has the Fourier representation

    f(t) = Σₘ f̂ₘ exp(imt),

so that

    ∂ₜf(t) = Σₘ im f̂ₘ exp(imt).

For even `N`, the Nyquist mode is assigned zero derivative.

## Arguments
* `f::AbstractVector{Complex{T}}`: Periodic samples on an equispaced computational grid.

## Returns
* `df::Vector{Complex{T}}`: Spectral derivative at the same nodes.
"""
function periodic_derivative_t(f::AbstractVector{Complex{T}}) where {T<:Real}
    N=length(f)
    F=FFTW.fft(f) # to get the Fourier coefficients f̂_k
    kvec=iseven(N) ? vcat(0:N÷2-1,0,-N÷2+1:-1) : vcat(0:(N-1)÷2,-(N-1)÷2:-1)
    return FFTW.ifft((im.*T.(kvec)).*F)
end

"""
    tangential_derivative_density(pts::BoundaryPoints{T},μ::AbstractVector{Complex{T}}) where {T<:Real} → Vector{Complex{T}}

Compute the tangential derivative `∂ₛμ` on one periodic Kress boundary.
If the boundary is parameterized by the periodic computational variable `t`,
then

    ∂ₛμ = (1/|γ'(t)|)∂ₜμ.

The computational derivative is evaluated spectrally with
[`periodic_derivative_t`](@ref), while the physical speed is obtained from the
stored boundary tangent.

## Arguments
* `pts::BoundaryPoints{T}`: Periodic boundary discretization.
* `μ::AbstractVector{Complex{T}}`: Density sampled at `pts`.

## Returns
* `dμds::Vector{Complex{T}}`: tangential derivative.
"""
function tangential_derivative_density(pts::BoundaryPoints{T},μ::AbstractVector{Complex{T}}) where {T<:Real}
    pts.is_periodic||error("Only works for periodic components.")
    speed=hypot.(getindex.(pts.tangent,1),getindex.(pts.tangent,2))
    dμ_dt=periodic_derivative_t(μ)
    return dμ_dt./speed
end

"""
    _slp_self_kress_component(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::BoundaryPoints{T},σ::AbstractVector{Complex{T}},k::T,Rblock::AbstractMatrix{T},G::BoundaryGeomCache{T}) where {T<:Real} → Vector{Complex{T}}

Apply the Helmholtz single-layer operator on one periodic boundary using the
Kress logarithmic singularity split. For

    (Sσ)(x) = ∫_∂Ω Φₖ(x,y)σ(y) ds_y,
    Φₖ(x,y) = (i/4)H₀⁽¹⁾(k|x-y|),

the same-boundary logarithmic singularity is written as

    Φₖ(γ(t),γ(τ))
        = m₁(t,τ) log|2sin((t-τ)/2)| + m₂(t,τ),

where `m₂` is smooth. `Rblock` supplies the Kress product-integration weights
for the logarithmic part, while `pts.ws` supplies the periodic quadrature
weights for the smooth remainder.

## Arguments
* `solver`: CFIE-Kress solver.
* `pts::BoundaryPoints{T}`: Periodic boundary discretization.
* `σ::AbstractVector{Complex{T}}`: Single-layer source density.
* `k::T`: Wavenumber.
* `Rblock::AbstractMatrix{T}`: Kress logarithmic quadrature matrix.
* `G::BoundaryGeomCache{T}`: Precomputed geometric quantities.

## Returns
* `Sσ::Vector{Complex{T}}`: Single-layer field evaluated on the boundary.
"""
function _slp_self_kress_component(solver::Union{CFIE_kress},pts::BoundaryPoints{T},σ::AbstractVector{Complex{T}},k::T,Rblock::AbstractMatrix{T},G::BoundaryGeomCache{T}) where {T<:Real}
    N=length(pts.xy)
    length(σ)==N||error("σ length mismatch in _slp_self_kress_component")
    pts.is_periodic||error("_slp_self_kress_component only supports periodic.")
    speed=G.speed
    logterm=G.logterm
    R=G.R
    ws=pts.ws
    out=Vector{Complex{T}}(undef,N)
    @inbounds for i in 1:N
        acc=zero(Complex{T})
        for j in 1:N
            sj=speed[j]
            if i==j
                m1=-inv_two_pi*sj
                m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*sj^2))*sj
                sval=Complex{T}(Rblock[i,j]*m1,zero(T))+ws[j]*m2
                acc+=sval*σ[j]
            else
                r=R[i,j]
                h0=H(0,k*r)
                j0=real(h0)
                m1=-inv_two_pi*j0*sj
                m2=Complex{T}(0,one(T)/2)*h0*sj-m1*logterm[i,j]
                sval=Rblock[i,j]*m1+ws[j]*m2
                acc+=sval*σ[j]
            end
        end
        out[i]=acc
    end
    return out
end

"""
    slp_boundary_kress(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::BoundaryPoints{T},σ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real} → Vector{Complex{T}}

Apply the on-boundary Helmholtz single-layer operator on the single periodic boundary.
This is the single-boundary wrapper around [`_slp_self_kress_component`](@ref).
"""
function slp_boundary_kress(solver::Union{CFIE_kress},pts::BoundaryPoints{T},σ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    return _slp_self_kress_component(solver,pts,σ,k,ws.Rmat,ws.Gs[1])
end

"""
    hypersingular_maue_kress(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},μ::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T) where {T<:Real} → Vector{Complex{T}}

Apply the hypersingular operator to a CFIE density using Maue regularization.
Direct quadrature of the normal derivative of the double-layer potential is
hypersingular. Maue's identity rewrites this action using only weakly singular
single-layer operators:

    Nμ = ∂ₛS(∂ₛμ) + k² n ⋅ S(nμ).

In Cartesian components,

    Nμ = ∂ₛS(∂ₛμ) + k²[nₓS(nₓμ) + nᵧS(nᵧμ)].

The function therefore:

1. computes `∂ₛμ` spectrally,
2. evaluates `S(∂ₛμ)`,
3. evaluates `S(nₓμ)` and `S(nᵧμ)`,
4. differentiates the first field tangentially,
5. combines the three terms according to Maue's identity.

## Arguments
* `solver`: CFIE-Kress solver.
* `μ::AbstractVector{Complex{T}}`: CFIE density on the full boundary.
* `pts::BoundaryPoints{T}`: Periodic full-boundary discretization.
* `ws::CFIEKressWorkspace{T}`: Precomputed Kress workspace.
* `k::T`: Wavenumber.

## Returns
* `Nμ::Vector{Complex{T}}`: Hypersingular action on `μ`.
"""
function hypersingular_maue_kress(solver::Union{CFIE_kress},μ::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    dμds=tangential_derivative_density(pts,μ)
    nx=getindex.(pts.normal,1)
    ny=getindex.(pts.normal,2)
    N=length(μ)
    σx=Vector{Complex{T}}(undef,N)
    σy=Vector{Complex{T}}(undef,N)
    @inbounds for j in 1:N
        σx[j]=nx[j]*μ[j]
        σy[j]=ny[j]*μ[j]
    end
    S_dμds=slp_boundary_kress(solver,pts,dμds,ws,k)
    Sx=slp_boundary_kress(solver,pts,σx,ws,k)
    Sy=slp_boundary_kress(solver,pts,σy,ws,k)
    T1=collect(tangential_derivative_density(pts,S_dμds))
    T2=Vector{Complex{T}}(undef,N)
    @inbounds for i in 1:N
        T2[i]=k^2*(nx[i]*Sx[i]+ny[i]*Sy[i])
    end
    return T1+T2
end

"""
    boundary_function_hypersingular_part(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T) where {T<:Real} → Vector{Complex{T}}

Return the hypersingular contribution `Nμ` required for CFIE boundary-function recovery.
The sign and factor multiplying this contribution are applied only in the final [`boundary_function`](@ref) assembly.
"""
function boundary_function_hypersingular_part(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    return hypersingular_maue_kress(solver,layer_density,pts,ws,k)
end

"""
    boundary_function_hypersingular_part(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},k::T) where {T<:Real} → Vector{Complex{T}}

Workspace-building convenience overload for
[`boundary_function_hypersingular_part`](@ref).
"""
function boundary_function_hypersingular_part(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},k::T) where {T<:Real}
    ws=build_cfie_kress_workspace(solver,pts)
    return boundary_function_hypersingular_part(solver,layer_density,pts,ws,k)
end

"""
    construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},G::BoundaryGeomCache{T},P::BoundaryPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real} → AbstractMatrix{Complex{T}}

Assemble the pure double-layer Nyström matrix `D(k)` on the single periodic
CFIE-Kress boundary. The Helmholtz double-layer operator is

    (Dμ)(x) = ∫_∂Ω ∂ₙ_y Φₖ(x,y) μ(y) ds_y.

The same-boundary weak singularity is evaluated using the Kress logarithmic
split already used in the CFIE matrix assembly.

The diagonal uses the analytic curvature limit.

## Arguments
* `solver`: CFIE-Kress solver.
* `D::AbstractMatrix{Complex{T}}`: Output matrix.
* `pts::BoundaryPoints{T}`: Periodic boundary discretization.
* `Rmat::AbstractMatrix{T}`: Kress logarithmic quadrature matrix.
* `G::BoundaryGeomCache{T}`: Cached geometry.
* `P::BoundaryPanelArrays{T}`: Cached panel arrays.
* `k::T`: Wavenumber.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable threaded off-diagonal assembly.

## Returns
* `D`: The mutated double-layer matrix.
"""
function construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},Rmat::AbstractMatrix{T},G::BoundaryGeomCache{T},P::BoundaryPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    N=length(P.X)
    @inbounds for i in 1:N
        D[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        wj=pts.ws[j]
        @inbounds for i in 1:(j-1)
            wi=pts.ws[i]
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            _,h1=hankel_pair01(k*r)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            D[i,j]=Rmat[i,j]*l1_ij+wj*l2_ij
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            D[j,i]=Rmat[j,i]*l1_ji+wi*l2_ji
        end
    end
    return D
end

"""
    construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real} → AbstractMatrix{Complex{T}}

Workspace overload of [`construct_cfie_kress_dlp_matrix!`](@ref).
"""
function construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return construct_cfie_kress_dlp_matrix!(solver,D,pts,ws.Rmat,ws.Gs[1],ws.parr[1],k;multithreaded=multithreaded)
end

"""
    construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real} → AbstractMatrix{Complex{T}}

Convenience overload that constructs the CFIE-Kress workspace internally.
"""
function construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress},D::AbstractMatrix{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_kress_workspace(solver,pts)
    return construct_cfie_kress_dlp_matrix!(solver,D,pts,ws,k;multithreaded=multithreaded)
end

"""
    adjoint_K_from_dlp_matrix(D::AbstractMatrix{Complex{T}},ds::AbstractVector{T}) where {T<:Real} → Matrix{Complex{T}}

Construct the discrete adjoint double-layer matrix `K'` from the double-layer
Nyström matrix `D`.

For the quadrature-weighted boundary pairing,

    ⟨f,g⟩ₕ = fᵀWg,    W = diag(ds),

the discrete formal adjoint is

    K' = W⁻¹DᵀW.

No complex conjugation is used here: this is the formal transpose required by
the boundary-integral kernel relation.

## Arguments
* `D::AbstractMatrix{Complex{T}}`: Discrete double-layer matrix.
* `ds::AbstractVector{T}`: Physical boundary quadrature weights.

## Returns
* `Kp::Matrix{Complex{T}}`: Discrete adjoint double-layer matrix.
"""
function adjoint_K_from_dlp_matrix(D::AbstractMatrix{Complex{T}},ds::AbstractVector{T}) where {T<:Real}
    N=length(ds)
    Kp=Matrix{Complex{T}}(undef,N,N)
    @inbounds for i in 1:N
        invdsi=inv(ds[i])
        for j in 1:N
            Kp[i,j]=D[j,i]*ds[j]*invdsi
        end
    end
    return Kp
end

"""
    adjoint_K_action_from_dlp_matrix(D::AbstractMatrix{Complex{T}},μ::AbstractVector{Complex{T}},ds::AbstractVector{T}) where {T<:Real} → Vector{Complex{T}}

Apply the discrete adjoint double-layer operator without explicitly forming `K'`.

Using

    K'μ = W⁻¹DᵀWμ,

the method first forms `Wμ`, applies `transpose(D)`, and finally divides by the
target quadrature weights.

## Arguments
* `D::AbstractMatrix{Complex{T}}`: Discrete double-layer matrix.
* `μ::AbstractVector{Complex{T}}`: Density.
* `ds::AbstractVector{T}`: Physical boundary quadrature weights.

## Returns
* `Kpμ::Vector{Complex{T}}`: Adjoint double-layer action.
"""
function adjoint_K_action_from_dlp_matrix(D::AbstractMatrix{Complex{T}},μ::AbstractVector{Complex{T}},ds::AbstractVector{T}) where {T<:Real}
    N=length(ds)
    tmp=Vector{Complex{T}}(undef,N)
    @inbounds for j in 1:N
        tmp[j]=ds[j]*μ[j]
    end
    v=transpose(D)*tmp
    @inbounds for i in 1:N
        v[i]/=ds[i]
    end
    return v
end

"""
    cfie_kress_adjoint_K_action(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real} → Vector{Complex{T}}

Compute the `K'μ` contribution required for CFIE boundary-function recovery.

The pure DLP matrix is assembled at `k`, after which the weighted transpose
identity

    K'μ = W⁻¹DᵀWμ

is applied using the physical boundary quadrature weights `pts.ds`.

This method returns only `K'μ`; any jump term belonging to the normal derivative
of the single layer is inserted in the final CFIE boundary-function formula.

## Arguments
* `solver`: CFIE-Kress solver.
* `layer_density::AbstractVector{Complex{T}}`: Full-boundary CFIE density.
* `pts::BoundaryPoints{T}`: Full-boundary discretization.
* `ws::CFIEKressWorkspace{T}`: CFIE-Kress workspace.
* `k::T`: Wavenumber.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable threaded DLP assembly.

## Returns
* `Kpμ::Vector{Complex{T}}`: Adjoint double-layer action.
"""
function cfie_kress_adjoint_K_action(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    N=length(pts)
    length(layer_density)==N||error("layer_density length mismatch in cfie_kress_adjoint_K_action")
    D=Matrix{Complex{T}}(undef,N,N)
    construct_cfie_kress_dlp_matrix!(solver,D,pts,ws,k;multithreaded=multithreaded)
    return adjoint_K_action_from_dlp_matrix(D,layer_density,pts.ds)
end

"""
    cfie_kress_adjoint_K_action(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real} → Vector{Complex{T}}

Convenience overload that constructs the CFIE-Kress workspace internally.
"""
function cfie_kress_adjoint_K_action(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_kress_workspace(solver,pts)
    return cfie_kress_adjoint_K_action(solver,layer_density,pts,ws,k;multithreaded=multithreaded)
end

"""
    boundary_function(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real} → Tuple

Recover and Rellich-normalize the physical boundary normal derivative from a
CFIE-Kress layer density.

The input density may be either symmetry-reduced or already defined on the full
physical boundary. The symmetry-orbit expansion is applied automatically:
full-length input is accepted silently unchanged, while reduced input is
expanded through the maps stored in `ws`.

The CFIE implementation uses doubled single- and double-layer operators. With
the corresponding doubled hypersingular and adjoint double-layer operators,
the boundary normal derivative can be taken, up to an irrelevant common
normalization factor, as

    u = -N*μ - i*k*(μ + K'*μ).

The hypersingular action is evaluated through Maue's identity,

    N*μ = ∂ₛ S(∂ₛμ) + k^2*n⋅S(n*μ),

while the adjoint double-layer action is obtained from the discrete
double-layer matrix using

    K' = W^(-1)*transpose(D)*W,
    W = diag(ds).

Here `N`, `S`, `D` and `K'` denote the doubled operators used internally by
the CFIE-Kress discretization. The resulting boundary function is normalized
with the Rellich identity.

## Arguments
* `solver::Union{CFIE_kress}`: CFIE-Kress solver.
* `layer_density::AbstractVector{Complex{T}}`: Reduced or full-boundary CFIE layer density.
* `pts::BoundaryPoints{T}`: Full periodic boundary discretization.
* `ws::CFIEKressWorkspace{T}`: Precomputed CFIE-Kress workspace.
* `k::T`: Wavenumber.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable threaded DLP assembly for the `K'` action.

## Returns
* `pts::BoundaryPoints{T}`: Input full-boundary discretization.
* `u::Vector{Complex{T}}`: Rellich-normalized physical boundary normal derivative.
"""
function boundary_function(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    pts,μ=symmetrize_layer_density(solver,layer_density,pts,ws)
    Nμ=hypersingular_maue_kress(solver,μ,pts,ws,k)
    Kpμ=cfie_kress_adjoint_K_action(solver,μ,pts,ws,k;multithreaded=multithreaded)
    u=-Nμ-Complex{T}(0,k).*(μ+Kpμ)
    nrlz=_rellich(pts,u,k)
    return pts,u./sqrt(nrlz)
end

"""
    boundary_function(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},billiard::Bi,k::T;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard} → Tuple

Workspace-building convenience overload for CFIE-Kress boundary-function
recovery.

The input density may be either symmetry-reduced or already defined on the full
physical boundary. Symmetry expansion is handled internally by the workspace
overload, so callers do not need to invoke [`symmetrize_layer_density`](@ref)
explicitly.

The `billiard` argument is retained for consistency with the other
boundary-integral boundary-function interfaces.
"""
function boundary_function(solver::Union{CFIE_kress},layer_density::AbstractVector{Complex{T}},pts::BoundaryPoints{T},billiard::Bi,k::T;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    ws=build_cfie_kress_workspace(solver,pts)
    return boundary_function(solver,layer_density,pts,ws,k;multithreaded=multithreaded)
end

"""
    boundary_function(solver::Union{CFIE_kress},layer_density::AbstractVector{<:AbstractVector{Complex{T}}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard} → Tuple

Batch CFIE-Kress boundary-function recovery for multiple wavenumbers.

Each state has its own boundary discretization and CFIE-Kress workspace. Every
input density may independently be either symmetry-reduced or already defined
on the full physical boundary. Reduced data are expanded automatically before
the physical normal derivative is reconstructed and Rellich-normalized.

## Arguments
* `solver::Union{CFIE_kress}`: CFIE-Kress solver.
* `layer_density`: Reduced or full CFIE layer densities.
* `pts`: Full physical boundary discretizations.
* `billiard::Bi`: Billiard geometry.
* `ks`: Wavenumbers corresponding to the densities.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable threaded DLP assembly for each state.

## Returns
* `pts`: Input boundary discretizations.
* `us::Vector{Vector{Complex{T}}}`: Physical boundary functions.
"""
function boundary_function(solver::Union{CFIE_kress},layer_density::AbstractVector{<:AbstractVector{Complex{T}}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us=Vector{Vector{Complex{T}}}(undef,length(layer_density))
    for i in eachindex(layer_density)
        ws=build_cfie_kress_workspace(solver,pts[i])
        pts_all[i],us[i]=boundary_function(solver,layer_density[i],pts[i],ws,ks[i];multithreaded=multithreaded)
    end
    return pts_all,us
end
