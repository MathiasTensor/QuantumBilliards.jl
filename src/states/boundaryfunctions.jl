#########################################################################################
################## HELPERS FOR SYMMETRIZATION OF THE BOUNDARY FUNCTION ##################
#########################################################################################

"""
    regularize!(u::Vector{T}) where {T<:Real}

Regularizes the boudnary function to get rid of potential artifacts in them. This is needed otherwise the construction of the objects from the boundary function (wavefunctions, husimi, wigner...) might fail. It approximates the NaNs by using the average of the neighboring values.

# Arguments
- `u::Vector{T}`: the boundary function vector to be regularized.

# Returns
- `Nothing`: inplace modification.
"""
@inline function regularize!(u)
    idx=findall(isnan,u)
    for i in idx
        if i!=1
            u[i]=(u[i+1]+u[i-1])/2.0
        else
            u[i]=(u[i+1]+u[end])/2.0
        end
    end
end

"""
    _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {N<:Number,T<:Real}

Compute the Rellich normalization integral estimate for a boundary density `u`
interpreted as the normal derivative of a Dirichlet eigenfunction.

Uses the identity

    ∫Ω |ψ|² dx = 1/(2k²) ∫∂Ω (x⋅n) |∂ₙψ|² ds

with the quadrature data stored in `pts`.

# Arguments
- `pts`: Boundary points containing `xy`, outward `normal`, and quadrature weights `ds`.
- `u`: Boundary density values at `pts.xy` <-> `∂ₙψ(x,y)`.
- `k`: Wavenumber.

# Returns
- Approximation of `∫Ω |ψ|² dx`.
"""
function _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {N<:Number,T<:Real}
    acc=zero(T)
    @inbounds @simd for i in eachindex(u)
        n=pts.normal[i]
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i]
        acc+=w*abs2(u[i])
    end
    return acc/(2*k^2)
end

"""
    _rellich(pts::BoundaryPointsCFIE{T},u::AbstractVector{N},k::T) where {N<:Number,T<:Real}

Compute the Rellich normalization integral estimate for CFIE boundary data.

The outward normal is reconstructed from the stored tangent as

    n = (t₂,-t₁)

and the returned value approximates

    ∫Ω |ψ|² dx = 1/(2k²) ∫∂Ω (x⋅n) |∂ₙψ|² ds.

For multiply connected domains, call this on all boundary components and sum the
signed contributions. Hole components must use the outward normal of the domain.

# Arguments
- `pts`: CFIE boundary points containing `xy`, `tangent`, and quadrature weights `ds`.
- `u`: Boundary density values at `pts.xy` <-> `∂ₙψ(x,y)`.
- `k`: Wavenumber.

# Returns
- Approximation of `∫Ω |ψ|² dx` for this component.
"""
function _rellich(pts::BoundaryPointsCFIE{T},u::AbstractVector{N},k::T) where {N<:Number,T<:Real}
    acc=zero(T)
    @inbounds @simd for i in eachindex(u)
        t=pts.tangent[i]
        sp=hypot(t[1],t[2])
        n=SVector(t[2]/sp,-t[1]/sp)
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i]
        acc+=w*abs2(u[i])
    end
    return acc/(2*k^2)
end

"""
    _rellich(comps::Vector{BoundaryPointsCFIE{T}}, u::AbstractVector{N}, k::T) where {N<:Number,T<:Real}

Compute the Rellich normalization integral for a multiply connected domain
represented by several CFIE boundary components.

This evaluates the identity

    ∫Ω |ψ|² dx = 1/(2k²) ∫∂Ω (x⋅n) |∂ₙψ|² ds

by summing contributions from all boundary components

    Ω = ∂Ω₁ ∪ ∂Ω₂ ∪ ... ∪ ∂Ωₘ

where ∂Ω₁ is the outer boundary and subsequent ∂Ωᵢ are holes. `u` is interpreted as the normal derivative `∂ₙψ` on the full boundary.

# Arguments
- `comps`: Vector of CFIE boundary components (`BoundaryPointsCFIE`), ordered
  such that the first component is the outer boundary and subsequent components
  are holes.
- `u`: Concatenated boundary density values corresponding to all components,
  ordered consistently with `comps`.
- `k`: Wavenumber.

# Returns
- Approximation of `∫Ω |ψ|² dx` over the full (possibly multiply connected) domain.
"""
function _rellich(comps::Vector{BoundaryPointsCFIE{T}},u::AbstractVector{N},k::T) where {N<:Number,T<:Real}
     length(u)==sum(length(c.xy) for c in comps) || error("u length does not match concatenated CFIE boundary components")
    # require one object per connected component
    length(unique(getfield.(comps,:compid)))==length(comps) || error("_rellich(comps, u, k) expects one BoundaryPointsCFIE per connected component")
    acc=zero(T)
    p=1
    for c in comps
        n=length(c.xy)
        acc+=_rellich(c,@view(u[p:p+n-1]),k)
        p+=n
    end
    return acc
end

"""
    shift_starting_arclength(billiard::Bi, u::Vector, pts::BoundaryPoints) where {Bi<:AbsBilliard}

When constructing the boundary function u we need to sometimes shift the starting point on the full boundary. 
This is a helper function that shifts all the fields in the BoundaryPoints struct such that all of them are 
shifted by the shift_s field in the billiard struct.

Note: This is called after apply_symmetries_to_boundary_points and apply_symmetries_to_boundary_function, 
so the u_bundle is already symmetrized and the pts are already symmetrized. Otherwise the shifting will happen
on the fundamental domain and when symmetrizing the results will be wrong.

# Arguments
- `billiard::Bi<:AbsBilliard`: the billiard object.
- `u::AbstractVector{U<:Real}`: the boundary function vector obtained from the full_boundary. Can be either complex or real
- `pts::BoundaryPoints`: the struct containing the boundary points on the full_boundary.

# Returns
- `BoundaryPoints`: the struct containing the boundary points on the full_boundary with the starting point shifted.
- `u::Vector`: the boundary function vector that is correctly shifted.
"""
function shift_starting_arclength(billiard::Bi,u::AbstractVector{U},pts::BoundaryPoints{T}) where {Bi<:AbsBilliard,T<:Real,U<:Number}
    if hasproperty(billiard,:shift_s)
        shift_s=billiard.shift_s
        L_effective=maximum(pts.s)
        # Find the index of the point where `s` is closest to `s_shift`
        start_index=argmin(abs.(pts.s.-shift_s))
        # Reorder all fields so that `start_index` becomes the first point
        shifted_s=circshift(pts.s,-start_index+1)
        shifted_u=circshift(u,-start_index+1)
        shifted_xy=circshift(pts.xy,-start_index+1)
        shifted_normal=circshift(pts.normal,-start_index+1)
        shifted_ds=circshift(pts.ds,-start_index+1)
        # Wrap around the `s` values to maintain continuity
        s_offset=shifted_s[1]
        shifted_s.=shifted_s.-s_offset  # Subtract the first value to make it zero
        shifted_s.=mod.(shifted_s,L_effective)  # Wrap around to maintain continuity
        pts=BoundaryPoints(shifted_xy,shifted_normal,shifted_s,shifted_ds)
        u=shifted_u
    end
    return pts,u
end

"""
    shift_starting_arclength(billiard::Bi, u_bundle::Matrix, pts::BoundaryPoints) where {Bi<:AbsBilliard}

Wrapper for the arclength shift when we have a matrix (columnwise) of us as the input to the boundary_function. 
This is the an efficient way to handle many boundary u functions.

Note: This is called after apply_symmetries_to_boundary_points and apply_symmetries_to_boundary_function, 
so the u_bundle is already symmetrized and the pts are already symmetrized. Otherwise the shifting will happen
on the fundamental domain and when symmetrizing the results will be wrong.

# Arguments
- `billiard::Bi<:AbsBilliard`: the billiard object.
- `u_bundle::AbstractMatrix{U}`: the boundary function matrix obtained from the full_boundary. Can be either real or complex.
- `pts::BoundaryPoints`: the struct containing the boundary points on the full_boundary.

# Returns
- `BoundaryPoints`: the struct containing the boundary points on the full_boundary with the starting point shifted.
- `u_bundle::Matrix`: the boundary function matrix that is shifted.
"""
function shift_starting_arclength(billiard::Bi,u_bundle::AbstractMatrix{U},pts::BoundaryPoints) where {Bi<:AbsBilliard,U<:Number}
    if hasproperty(billiard,:shift_s)
        shift_s=billiard.shift_s
        L_effective=maximum(pts.s)
        start_index=argmin(abs.(pts.s.-shift_s))
        shifted_s=circshift(pts.s,-start_index+1)
        shifted_xy=circshift(pts.xy,-start_index+1)
        shifted_normal=circshift(pts.normal,-start_index+1)
        shifted_ds=circshift(pts.ds,-start_index+1)
        s_offset=shifted_s[1]
        shifted_s.=shifted_s.-s_offset
        shifted_s.=mod.(shifted_s,L_effective)
        shifted_u_bundle=similar(u_bundle)
        for i in axes(u_bundle,2)
            shifted_u_bundle[:,i].=circshift(u_bundle[:,i],-start_index+1)
        end
        pts=BoundaryPoints(shifted_xy,shifted_normal,shifted_s,shifted_ds)
        return pts,shifted_u_bundle
    else
        return pts,u_bundle
    end
end

##########################
#### BASIS RESCALINGS ####
##########################

#TODO Really annoying hack
"""
    rescale_dimension(basis::Ba, dim::T) where {T<:Real, Ba<:AbsBasis}

Helper/hack function to rescale the dimension of RealPlaneWaves struct b/c due to 
the basis unique parity_pattern function which rescales the basis in a way that it 
multiplies it with the length of possible parity combinations. 
Since we require one in the fundamental domain for boundary function construction this effectively divides 
the dimension of the basis with the parity combination length.

# Arguments
- `basis::Ba`: the basis object.
- `dim::Integer`: the dimension to rescale.

# Returns
- `dim::Integer`: the rescaled dimension.
"""
@inline function rescale_dimension(basis::Ba,dim::Integer) where {Ba<:AbsBasis}
    main_basis=basis isa CompositeBasis ? basis.main : basis
    if main_basis isa RealPlaneWaves # hack since RealPlaneWaves have problems
        if isnothing(main_basis.symmetries)
            dim=Int(dim/4) # always works by construction of parity_pattern
        elseif (main_basis.symmetries isa Reflection)
            if (main_basis.symmetries.axis==:x_axis) || (main_basis.symmetries.axis==:y_axis)
                dim=Int(dim/2) # always works by construction of parity_pattern
            end
        end
    else
        # Add custom logic here for basis which have symmetry adapted dimension scaling
    end
    return dim
end

# Helper to get the dimension of the evanescent basis, default to 0 if not a CompositeBasis
@inline function get_evanescent_dim_in_CompositeBasis(basis::AbsBasis)
    return basis isa CompositeBasis ? basis.evanescent.dim : 0
end

###########################################################################
################ BOUNDARY FUNCTION FOR BASIS TYPE SOLVERS #################
###########################################################################

"""
    boundary_function(state::S; b=5.0) where {S<:AbsState}

Low-level function that constructs the boundary function and the boundary points `BoundaryPoints` on the `desymmetrized_full_boundary` to which symmetries are being applied.
This effectively constructs the boundary function on the whole boundary through applying symmetries to the `desymmetrized_full_boundary`.
It also constructs the norm of the boundary function on the whole boundary (after symmetry application) as norm = ∮u(s)⟨r(s),n(s)⟩ds.

Is is used by `ParticularSolutionsMethod`,`DecompositionMethod` and `VerginiSaraceno` solvers.
To construct a state use the results from `solve_vect` to get the vector of coefficients in a given basis
and then construct an `Eigenstate` object with those coefficients and the associated `k` and `basis`. 
Then this `Eigenstate` object can be used as the input to this function to get the boundary function, arclength values.

Example: 

```julia
d=7.0
b=12.0
k=10.0 # not really an eigenvalue, but needs to be an eigenvalue otherwise the coefficients just give nonsense ofc. 
solver=ParticularSolutionsMethod(d,b,b) # or DecompositionMethod(d,b) or VerginiSaraceno(d,b)
billiard,basis=make_stadium_and_basis(1.0)
pts=evaluate_points(solver,billiard,1.0)
ten,vec=solve_vect(solver,basis,pts,k)
state=Eigenstate(k,vec,ten,basis,billiard)
u,pts,norm=boundary_function(state,b=b)
```

# Arguments
- `state<:AbsState`: typically the eigenstate associated with a particular solution to the problem on the fundamental boundary.
- `b=5.0`: optional parameter for point scaling in the construction of evaluation points on the boundary.

# Returns
- `u::Vector{<:Real}`: the boundary function evaluated at the points on the boundary.
- `pts::BoundaryPoints`: the boundary points corresponding to the `u`. 
"""
function boundary_function(state::S;b=5.0) where {S<:AbsState}
    vec=state.vec
    k=state.k
    k_basis=state.k_basis
    new_basis=state.basis
    billiard=state.billiard
    T=eltype(vec)
    boundary=billiard.fundamental_boundary
    crv_lengths=[crv.length for crv in boundary if crv isa AbsRealCurve]
    sampler=FourierNodes([2,3,5],collect(crv_lengths))
    L=billiard.length
    N=max(round(Int,k*L*b/(2π)),512)
    pts=boundary_coords_fourier(billiard,sampler,N)
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
    pts=apply_symmetries_to_boundary_points(pts,new_basis.symmetries,billiard)
    u=apply_symmetries_to_boundary_function(u,new_basis.symmetries)
    pts,u=shift_starting_arclength(billiard,u,pts)
    nrlz=_rellich(pts,u,k) # Rellich boundary norm: ∫ |u|^2 (n·x) ds / (2k^2) no temps
    @blas_1 return u./sqrt(nrlz),pts
end

"""
    boundary_function(state_data::StateData, billiard::Bi, basis::Ba; b=5.0) where {Bi<:AbsBilliard, Ba<:AbsBasis}

Computes the boundary functions us and the `BoundaryPoints` from which we can construct the wavefunction object 
using the boundary integral definition.

Note: This is a higher-level wrapper around the `boundary_function` that takes in the `StateData` object 
therefore it is only ever used from the `compute_eigenstate` function with the `VerginiSaraceno` solver. 
It is not used in the `ParticularSolutionsMethod` or `DecompositionMethod` solvers.
# Arguments
`state_data::StateData`: The state data object that contains the `ks` and the `vec` from which we can construct the boundary points and the boundary function.
`billiard<:AbsBilliard`: The geometry of the billiard.
`basis<::AbsBilliard`: The basis from which we construct the wavefunction and the `vec` basis coefficients.

# Returns
`ks::Vector`: The vector of eigenvalues. A convenience return.
`us::Vector{Vector}`: The vector of boundary functions (Vector) for each k that is a solution.
`pts_all::Vector{BoundaryPoints}`: A struct that contains the positions for which the u(s) (boundary function was evaluated) {pts.xy}, the arclengths that corresponds to these points {pts.s}, the normal vectors for the points we use {pts.normal} and the differences between the arclengths {pts.ds}. This is for every k in ks.
"""
function boundary_function(state_data::StateData,billiard::Bi,basis::Ba;b=5.0) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks)) # always real since reflection are 1d irreps
    pts_all=Vector{BoundaryPoints{eltype(ks)}}(undef,length(ks))
    valid_indices=fill(true,length(ks))
    progress=Progress(length(ks);desc="Constructing the u(s)...")
    for i in eachindex(ks) 
        #try # the @. macro can faill in gradient_matrices when multithreading
            vec=X[i] # vector of vectors
            dim=length(vec)
            dim=dim-get_evanescent_dim_in_CompositeBasis(basis) # this is to prevent double counting the EWP
            dim=rescale_dimension(basis,dim)
            new_basis=resize_basis(basis,billiard,dim,ks[i]) # dim here should be for the main function
            state=Eigenstate(ks[i],vec,tens[i],new_basis,billiard)
            u,pts=boundary_function(state;b=b) # pts is BoundaryPoints and has information on ds and x
            us_all[i]=u
            pts_all[i]=pts
        #catch e
        #    println("Error while constructing the u(s) for k = $(ks[i]): $e")
        #    valid_indices[i]=false
        #end
        next!(progress)
    end
    ks=ks[valid_indices]
    us_all=us_all[valid_indices]
    pts_all=pts_all[valid_indices]
    return ks,us_all,pts_all
end

"""
    momentum_function(u, s)

Computes the momentum distribution function for the boundary function `u` defined with arclengths `s`. This function uses the Fast Fourier Transform (FFT) to compute the Discrete Fourier Transform 1D. This returns the plot of the DFT (y-axis) and the corresponding frequencies (ks) for each pt in y-axis.

# Arguments
- `u`: A vector representing the boundary function (Wavefunction normal derivative) on the boundary.
- `s`: A vector representing the arclengths (positions) on the boundary where the normal derivative was calculated.

# Returns
- `abs2.(fu)/length(fu)::Vector`: A vector corresponding the the DFT.
- `ks::Vector`: The associated frequencies.
"""
function momentum_function(u,s)
    fu=rfft(u)
    sr=1.0/diff(s)[1]
    ks=rfftfreq(length(s),sr).*(2*pi)
    return abs2.(fu)/length(fu),ks
end

"""
    momentum_function(state::S; b=5.0) where {S<:AbsState}

Computes the momentum distribution function for a given quantum state object. The wavefunction's normal derivative and its corresponding arclength positions are extracted from the state. This function internally calls `boundary_function` to extract the boundary function `u` and arclength positions `s` for the given quantum state.


# Arguments
- `state::S`: A quantum state object of type `<:AbsState`.
- `b::Real`: Number of basis functions used to compute the wavefunction on the boundary (default: `5.0`).

# Returns
- `Vector`: The DFT amplitudes (`abs2.(fu)/length(fu)`), corresponding to the momentum distribution.
- `Vector`: The frequencies (`ks`) associated with the DFT components.
"""
function momentum_function(state::S;b=5.0) where {S<:AbsState}
    u,s=boundary_function(state;b=b)
    return momentum_function(u,s)
end

#########################################################################################################
################### BoundaryIntegralMethod, DLP_kress and DLP_kress_global_corners ######################
#########################################################################################################

#######################################################################
########## Symmetrize layer potentials and boundary points ############
#######################################################################

"""
    symmetrize_layer_potential(solver::BoundaryIntegralMethod,layer_pot::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) -> Tuple{BoundaryPoints{T}, Vector{N}} where {N<:Number,T<:Real,Bi<:AbsBilliard}

Expand a `BoundaryIntegralMethod` layer potential from the symmetry-reduced
boundary to the full boundary.
"""
function symmetrize_layer_potential(solver::BoundaryIntegralMethod,layer_pot::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    # naive DLP has method of images / irrep projections so it works with fundamental domains.
    # Therefore we need to apply the symmetries to the boundary points and the boundary function to get the full boundary function on the full boundary.
    pts=apply_symmetries_to_boundary_points(pts,solver.symmetry,billiard)
    u=apply_symmetries_to_boundary_function(layer_pot,solver.symmetry)
    return pts,u
end

"""
    symmetrize_layer_potential(solver::BoundaryIntegralMethod,layer_pot::Vector{<:AbstractVector{N}},pts::Vector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) -> Tuple{Vector{BoundaryPoints{T}}, Vector{Vector{N}}} where {N<:Number,T<:Real,Bi<:AbsBilliard}

Batch version of `symmetrize_layer_potential` for `BoundaryIntegralMethod`.
"""
function symmetrize_layer_potential(solver::BoundaryIntegralMethod,layer_pot::Vector{<:AbstractVector{N}},pts::Vector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    us_all=Vector{Vector{N}}(undef,length(pts))
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    @use_threads multithreading=multithreaded for i in eachindex(pts)
        pts_all[i]=apply_symmetries_to_boundary_points(pts[i],solver.symmetry,billiard)
        us_all[i]=apply_symmetries_to_boundary_function(layer_pot[i],solver.symmetry)
    end
    return pts_all,us_all
end

"""
    symmetrize_layer_potential(solver::Union{DLP_kress,DLP_kress_global_corners,CFIE_kress,CFIE_kress_global_corners,CFIE_kress_corners,CFIE_alpert},layer_pot,pts,billiard::Bi) -> Tuple{typeof(pts), typeof(layer_pot)} where {Bi<:AbsBilliard}

No-op symmetrization for full-boundary Kress/CFIE discretizations. The return is the same as input.
"""
function symmetrize_layer_potential(solver::Union{DLP_kress,DLP_kress_global_corners,CFIE_kress,CFIE_kress_global_corners,CFIE_kress_corners,CFIE_alpert},layer_pot,pts,billiard::Bi) where {Bi<:AbsBilliard}
    # boudnary is always full due to Kress splitting, therefore there is no symetrization that needs to be done
    return pts,layer_pot
end

#######################################################################

"""
    boundary_function(solver::BoundaryIntegralMethod,layer_pot::AbstractVector{N},pts::BoundaryPoints{T},k::T,billiard::Bi,) -> Tuple{BoundaryPoints{T}, Vector{N}} where {N<:Number,T<:Real,Bi<:AbsBilliard}

Normalize a `BoundaryIntegralMethod` boundary density by the Rellich norm.
"""
function boundary_function(solver::BoundaryIntegralMethod,layer_pot::AbstractVector{N},pts::BoundaryPoints,k::T,billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    nrlz=_rellich(pts,layer_pot,k)
    return pts,layer_pot./sqrt(nrlz)
end

"""
    boundary_function(solver::BoundaryIntegralMethod,layer_pot::Vector{<:AbstractVector{N}},pts::Vector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) -> Tuple{Vector{BoundaryPoints{T}}, Vector{Vector{N}}} where {N<:Number,T<:Real,Bi<:AbsBilliard}

Batch Rellich-normalization for `BoundaryIntegralMethod` boundary densities.
"""
function boundary_function(solver::BoundaryIntegralMethod,layer_pot::Vector{<:AbstractVector{N}},pts::Vector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    us_all=Vector{Vector{N}}(undef,length(pts))
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    @use_threads multithreading=multithreaded for i in eachindex(pts)
        pts_all[i]=pts[i]
        nrlz=_rellich(pts[i],layer_pot[i],ks[i])
        us_all[i]=layer_pot[i]./sqrt(nrlz)
    end
    return pts_all,us_all
end

"""
    boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_pot::AbstractVector{N},pts::BoundaryPointsCFIE{T},billiard::Bi,k::T) -> Tuple{BoundaryPointsCFIE{T}, Vector{N}} where {N<:Number,T<:Real,Bi<:AbsBilliard}

Return the Rellich-normalized DLP-Kress boundary function on a full boundary.
"""
function boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_pot::AbstractVector{N},pts::BoundaryPointsCFIE{T},billiard::Bi,k::T) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    # for DLP kress and DLP_kress_global_corners the boundary function is the layer potential itself and
    # also the boudnary is always full due to Kress splitting, therefore there is no symetrization that needs to be done
    # normalize with Rellich
    nrlz=_rellich(pts,layer_pot,k)
    return pts,layer_pot./sqrt(nrlz)
end

"""
    boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_pot::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPointsCFIE},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) -> Tuple{Vector{typeof(pts[1])}, Vector{typeof(layer_pot[1])}} where {N<:Number,T<:Real,Bi<:AbsBilliard}

Batch Rellich-normalization for DLP-Kress boundary functions.
"""
function boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_pot::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPointsCFIE},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {N<:Number,Bi<:AbsBilliard,T<:Real}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    layer_pot_all=Vector{typeof(layer_pot[1])}(undef,length(pts))
    @use_threads multithreading=multithreaded for i in eachindex(pts)
        pts_all[i]=pts[i]
        nrlz=_rellich(pts[i],layer_pot[i],ks[i])
        layer_pot_all[i]=layer_pot[i]./sqrt(nrlz)
    end
    return pts_all,layer_pot_all
end

####################################################################################################
################### CFIE_kress, CFIE_kress_corners, CFIE_kress_global_corners ######################
####################################################################################################


#    split_cfie_density_by_component(comps, μ)
#
# Split one global CFIE density vector into one density vector per connected
# boundary component. For the Kress-based CFIE solvers, the unknown boundary density is stored as one
# global vector
#
#    μ = [μ₁; μ₂; ...; μ_nc],
#
# where:
# - `nc` is the number of connected boundary components,
# - `μ_a` is the restriction of the density to component `Γ_a`,
# - the global ordering follows exactly the same component-by-component
#  concatenation used in matrix assembly.
#
# If component `Γ_a` is discretized by `N_a` points, then the global vector has
# length
#
#    N_tot = N₁ + N₂ + ... + N_nc,
#
# and this function recovers the block decomposition
#
#    μ_a, a = 1, ..., nc.
#
# Inputs
# - `comps::Vector{BoundaryPointsCFIE{T}}`:
#  The CFIE boundary discretization, one entry per connected component, in the
#  same order used for the global matrix.
# - `μ::AbstractVector{Complex{T}}`:
#  The global concatenated CFIE density.
#
#Returns
#- `out::Vector{Vector{Complex{T}}}`: such that `out[a]` is the slice of `μ` corresponding to the `a`-th connected component.
function split_cfie_density_by_component(comps::Vector{BoundaryPointsCFIE{T}},μ::AbstractVector{Complex{T}}) where {T<:Real}
    out=Vector{Vector{Complex{T}}}(undef,length(comps))
    p=1
    for a in eachindex(comps)
        N=length(comps[a].xy)
        out[a]=collect(μ[p:p+N-1])
        p+=N
    end
    return out
end

"""
    periodic_derivative_t(f::AbstractVector{Complex{T}}) where {T<:Real}

Compute the derivative of a periodic function sampled on an equispaced grid.
Let `f` be a periodic function defined on the interval `[0, 2π)` and sampled at
equispaced nodes

    t_j = 2π j / N,   j = 0, ..., N-1.

The discrete vector `f[j] ≈ f(t_j)` represents a periodic function, and its
derivative with respect to the periodic parameter `t` can be computed via the
Fourier series representation

    f(t) = Σ_k  f̂_k e^{i k t}.

Then

    ∂_t f(t) = Σ_k  (i k) f̂_k e^{i k t}.

# Returns
- (∂_t f)(t_j)::Vector{Complex{T}}: the spectral derivative of `f` evaluated at the same grid points `t_j`. The output is a vector of the same length as `f`, where each entry approximates the derivative of `f` at the corresponding grid point. In other words, if `f[j] ≈ f(t_j)`, then `(∂_t f)[j] ≈ (∂_t f)(t_j)` for each `j = 0, ..., N-1`. The output is generally complex-valued, even if the input `f` is real-valued, due to the nature of the Fourier differentiation.
"""
function periodic_derivative_t(f::AbstractVector{Complex{T}}) where {T<:Real}
    N=length(f)
    F=fft(f) # to get the Fourier coefficients f̂_k
    kvec=iseven(N) ? vcat(0:N÷2-1,0,-N÷2+1:-1) : vcat(0:(N-1)÷2,-(N-1)÷2:-1) # wave numbers k for the Fourier modes, ordered according to the output of fft. If even N, the zero frequency is followed by positive frequencies up to N/2-1, then the Nyquist frequency (which is zero for the derivative), and then negative frequencies from -N/2+1 to -1. If odd N, the zero frequency is followed by positive frequencies up to (N-1)/2, and then negative frequencies from -(N-1)/2 to -1.
    return ifft((im.*T.(kvec)).*F) # ∂_t f(t) = Σ_k  (i k) f̂_k e^{i k t} where f̂_k is the Fourier transform 
end

# tangential_derivative_density(pts, μ)
#
# Compute the tangential derivative ∂_s μ on one periodic CFIE-Kress component.
#
# Mathematical meaning:
# The boundary is parameterized by a periodic variable t (or σ in the graded case),
# and μ is sampled at those nodes. The physical derivative along arc-length is
#
#     ∂_s μ = (1 / |γ'(t)|) ∂_t μ.
#
# Here:
# - ∂_t μ is computed spectrally via periodic_derivative_t,
# - |γ'(t)| is the local speed reconstructed from the stored tangent.
#
# Assumptions:
# - pts.is_periodic == true (Kress periodic discretization),
# - nodes are equispaced in the periodic computational variable.
#
# Not valid for non-periodic (e.g. Alpert) discretizations.
function tangential_derivative_density(pts::BoundaryPointsCFIE{T},μ::AbstractVector{Complex{T}}) where {T<:Real}
    pts.is_periodic || error("Only works for periodic components - Kress works fine, Alpert not implemented.")
    _,_,speed=component_normals(pts)
    dμ_dt=periodic_derivative_t(μ)
    return dμ_dt./speed
end

# tangential_derivative_density_components(comps, μ)
#
# Compute the tangential derivative ∂_s μ componentwise for a global CFIE density.
# This version handles holes and multiple connected components by calling tangential_derivative_density on each component separately.
#
# Mathematical meaning:
# The global density is stored as a concatenation
#
#     μ = [μ₁; μ₂; ...; μ_nc],
#
# where each μ_a lives on one periodic boundary component Γ_a. Since the
# tangential derivative ∂_s acts independently on each component, we compute
#
#     (∂_s μ)_a = ∂_s (μ_a)
#
# using the periodic Kress differentiation on each Γ_a separately.
function tangential_derivative_density_components(comps::Vector{BoundaryPointsCFIE{T}},μ::AbstractVector{Complex{T}}) where {T<:Real}
    μ_comps=split_cfie_density_by_component(comps,μ)
    out=Vector{Vector{Complex{T}}}(undef,length(comps))
    for a in eachindex(comps)
        out[a]=collect(tangential_derivative_density(comps[a],μ_comps[a]))
    end
    return out
end

# _slp_self_kress_component(solver, pts, σ, k, Rblock, G)
#
# Apply the on-boundary single-layer operator Sσ on one periodic component
# using the Kress logarithmic singularity splitting.
#
# The Helmholtz single-layer operator is
#
#     (Sσ)(x) = ∫_Γ Φ_k(x,y) σ(y) ds_y,
#
# with Φ_k(x,y) = (i/4) H_0^(1)(k|x-y|).
#
# On a periodic Kress discretization, the kernel is decomposed as
#
#     Φ_k(x,y) = m1(x,y) log|sin((t-s)/2)| + m2(x,y),
#
# where:
# - m1 captures the universal logarithmic singularity,
# - m2 is smooth.
function _slp_self_kress_component(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::BoundaryPointsCFIE{T},σ::AbstractVector{Complex{T}},k::T,Rblock::AbstractMatrix{T},G::CFIEGeomCache{T}) where {T<:Real}
    N=length(pts.xy)
    length(σ)==N || error("σ length mismatch in _slp_self_kress_component")
    pts.is_periodic || error("_slp_self_kress_component only supports periodic Kress components.")
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

# slp_boundary_kress(solver, comps, σ, ws, k)
#
# Apply the on-boundary single-layer operator S to a global CFIE density σ on
# a multi-component periodic Kress geometry.
#
# Mathematical meaning:
# If the boundary is a disjoint union
#
#     Γ = Γ₁ ∪ Γ₂ ∪ ... ∪ Γ_m,
#
# then the boundary single-layer operator splits into component blocks
#
#     (Sσ)|_{Γ_a} = Σ_b S_ab σ_b.
#
# For same-component interactions a=b, the kernel is weakly singular and must be
# evaluated with the Kress logarithmic correction.
#
# For cross-component interactions a≠b, the kernel is smooth, since distinct
# components stay a positive distance apart, so ordinary quadrature is enough.
function slp_boundary_kress(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},comps::Vector{BoundaryPointsCFIE{T}},σ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    σ_comps=split_cfie_density_by_component(comps,σ)
    out_parts=Vector{Vector{Complex{T}}}(undef,length(comps))
    for a in eachindex(comps)
        ra=ws.offs[a]:(ws.offs[a+1]-1)
        Raa=@view ws.Rmat[ra,ra]
        Sa=_slp_self_kress_component(solver,comps[a],σ_comps[a],k,Raa,ws.Gs[a])
        ta_xy=comps[a].xy
        @inbounds for b in eachindex(comps)
            b==a && continue
            src=comps[b]
            σb=σ_comps[b]
            for i in eachindex(ta_xy)
                xi=ta_xy[i][1]
                yi=ta_xy[i][2]
                acc=zero(Complex{T})
                for j in eachindex(src.xy)
                    q=src.xy[j]
                    r=hypot(xi-q[1],yi-q[2])
                    acc+=Φ_helmholtz(k,r)*σb[j]*src.ds[j]
                end
                Sa[i]+=acc
            end
        end
        out_parts[a]=Sa
    end
    return vcat(out_parts...)
end

# hypersingular_maue_kress(solver, μ, comps, ws, k)
# REF: Kress, A COLLOCATION METHOD FOR A HYPERSINGULAR BOUNDARY INTEGRAL EQUATION VIA TRIGONOMETRIC DIFFERENTIATION, JOURNAL OF INTEGRAL EQUATIONS AND APPLICATIONS Volume 26, Number 2, Summer 2014
# REF: Kress, "Boundary Integral Equations in Time-Harmonic Acoustic Scattering", Eq. 2.7, page 233, 1991
#
# Compute the Maue-regularized hypersingular action Nμ for the periodic
# CFIE-Kress family using Maue's formula (check 1st reference).
#
# Mathematical meaning:
# The hypersingular operator N is the normal derivative of the double-layer
# potential taken on the boundary. Direct evaluation is difficult because its
# kernel is strongly singular. Maue's identity rewrites it in the regularized
# form
#
#     Nμ = ∂_s S(∂_s μ) + k^2 n · S(n μ),
#
# where:
# - S is the boundary single-layer operator,
# - ∂_s is the tangential derivative along the boundary,
# - n is the outward unit normal.
#
# In two dimensions this means that the hypersingular action can be computed
# from weakly singular single-layer evaluations plus tangential differentiation,
# avoiding direct hypersingular quadrature.
#
# What this routine does:
# 1. Split the global density μ into connected components.
# 2. Compute ∂_s μ componentwise.
# 3. Form the auxiliary single-layer source densities
#
#       σx = n_x μ,
#       σy = n_y μ.
#
# 4. Evaluate the three boundary single-layer fields
#
#       S(∂_s μ),  S(n_x μ),  S(n_y μ).
#
# 5. On each component, compute
#
#       T1 = ∂_s S(∂_s μ),
#       T2 = k^2 (n_x S(n_x μ) + n_y S(n_y μ)).
#
# 6. Return
#       Nμ = T1 + T2
function hypersingular_maue_kress(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},μ::AbstractVector{Complex{T}},comps::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    dμds_comps=tangential_derivative_density_components(comps,μ)
    dμds=vcat(dμds_comps...)
    src=flatten_cfie_components(comps)
    Ntot=length(μ)
    σx=Vector{Complex{T}}(undef,Ntot)
    σy=Vector{Complex{T}}(undef,Ntot)
    @inbounds for j in 1:Ntot
        σx[j]=src.nx[j]*μ[j]
        σy[j]=src.ny[j]*μ[j]
    end
    S_dμds=slp_boundary_kress(solver,comps,dμds,ws,k)
    Sx=slp_boundary_kress(solver,comps,σx,ws,k)
    Sy=slp_boundary_kress(solver,comps,σy,ws,k)
    S_dμds_comps=split_cfie_density_by_component(comps,S_dμds)
    Sx_comps=split_cfie_density_by_component(comps,Sx)
    Sy_comps=split_cfie_density_by_component(comps,Sy)
    out_parts=Vector{Vector{Complex{T}}}(undef,length(comps))
    for a in eachindex(comps)
        pts=comps[a]
        nx,ny,_=component_normals(pts)
        T1=collect(tangential_derivative_density(pts,S_dμds_comps[a]))
        Nloc=length(pts.xy)
        T2=Vector{Complex{T}}(undef,Nloc)
        @inbounds for i in 1:Nloc
            T2[i]=k^2*(nx[i]*Sx_comps[a][i]+ny[i]*Sy_comps[a][i])
        end
        out_parts[a]=T1+T2
    end
    return vcat(out_parts...)
end

# boundary_function_hypersingular_part(solver, layer_pot, comps, ws, k)
#
# Return the hypersingular part of the recovered physical boundary function for
# the periodic CFIE-Kress family.
#
# Mathematical meaning:
# If the CFIE wavefunction is represented as
#
#     ψ = -(D + i k S) μ,
#
# then its boundary normal derivative contains the contribution
#
#     -Nμ,
#
# where N is the hypersingular operator associated with the double-layer part.
#
# This helper isolates exactly that piece:
#
#     boundary_function_hypersingular_part = Nμ.
#
# It is therefore just a thin semantic wrapper around
# `hypersingular_maue_kress`, but the wrapper is useful because it makes the
# later boundary-function formula easier to read and documents the physical role
# of this term.
#
# Notes:
# The actual minus sign is applied later in `boundary_function`, since there we
# assemble the full identity
#
#     ∂_n ψ = -Nμ - i k (-1/2 I + K') μ
function boundary_function_hypersingular_part(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_pot::AbstractVector{Complex{T}},comps::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    return hypersingular_maue_kress(solver,layer_pot,comps,ws,k)
end

# boundary_function_hypersingular_part(solver, layer_pot, comps, k)
#
# Convenience wrapper for `boundary_function_hypersingular_part` that builds the
# CFIE-Kress workspace internally.
#
# Use this form when the hypersingular part is needed only once. For repeated
# evaluations at the same discretization, the workspace-taking overload is more efficient
function boundary_function_hypersingular_part(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_pot::AbstractVector{Complex{T}},comps::Vector{BoundaryPointsCFIE{T}},k::T) where {T<:Real}
    ws=build_cfie_kress_workspace(solver,comps)
    return boundary_function_hypersingular_part(solver,layer_pot,comps,ws,k)
end

# construct_cfie_kress_dlp_matrix!(solver, D, pts, Rmat, Gs, parr, offs, k; multithreaded=true)
#
# Assemble the pure double-layer part D(k) of the periodic CFIE-Kress operator
# on a multiply connected periodic boundary.
#
#     (Dμ)(x) = ∫_Γ ∂_{n_y} G_k(x,y) μ(y) ds_y
#
# discretized on all components.
#
# Structure of the assembly:
#
# 1. Same-component interactions:
#    For x,y on the same periodic component Γ_a, the weak logarithmic
#    singularity is handled by the Kress decomposition
#
#        kernel = logarithmic part + smooth remainder.
#
#    On each diagonal block this gives entries of the form
#
#        D_ij = R_ij l1_ij + w_j l2_ij,
#
#    with:
#    - `Rmat` carrying the universal periodic logarithmic quadrature,
#    - `l1_ij` the coefficient of the logarithmic part,
#    - `l2_ij` the smooth remainder,
#    - `w_j = pts[a].ws[j]`.
#
#    The diagonal entries are filled from the curvature limit already used in
#    the CFIE assembly:
#
#        D_ii = w_i κ_i.
#
# 2. Cross-component interactions:
#    For x ∈ Γ_a, y ∈ Γ_b with a ≠ b, the kernel is smooth, so no Kress
#    correction is needed. Those blocks are evaluated directly with standard
#    quadrature:
#
#        D_ij = w_j (i k / 2) H_1^(1)(k r) inner / r.
function construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},Gs::Vector{CFIEGeomCache{T}},parr::Vector{CFIEPanelArrays{T}},offs::Vector{Int},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    nc=length(pts)
    for a in 1:nc
        pa=pts[a]
        Ga=Gs[a]
        Pa=parr[a]
        Na=length(Pa.X)
        ra=offs[a]:(offs[a+1]-1)
        @inbounds for i in 1:Na
            gi=ra[i]
            D[gi,gi]=Complex{T}(pa.ws[i]*Ga.kappa[i],zero(T))
        end
        @use_threads multithreading=(multithreaded && Na>=32) for j in 2:Na
            gj=ra[j]
            wj=pa.ws[j]
            @inbounds for i in 1:(j-1)
                gi=ra[i]
                wi=pa.ws[i]
                r=Ga.R[i,j]
                invr=Ga.invR[i,j]
                lt=Ga.logterm[i,j]
                inn_ij=Ga.inner[i,j]
                inn_ji=Ga.inner[j,i]
                _,h1=hankel_pair01(k*r)
                j1=real(h1)
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                D[gi,gj]=Rmat[gi,gj]*l1_ij+wj*l2_ij
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                D[gj,gi]=Rmat[gj,gi]*l1_ji+wi*l2_ji
            end
        end
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        pa=pts[a]
        pb=pts[b]
        Pa=parr[a]
        Pb=parr[b]
        Na=length(Pa.X)
        Nb=length(Pb.X)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=Pa.X; Ya=Pa.Y
        Xb=Pb.X; Yb=Pb.Y
        dXb=Pb.dX; dYb=Pb.dY
        wb=pb.ws
        @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            @inbounds for j in 1:Nb
                gj=rb[j]
                dx=xi-Xb[j]
                dy=yi-Yb[j]
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                invr=inv(r)
                inn=dYb[j]*dx-dXb[j]*dy
                _,h1=hankel_pair01(k*r)
                D[gi,gj]=wb[j]*(Complex{T}(0,k/2)*inn*h1*invr)
            end
        end
    end
    return D
end

# construct_cfie_kress_dlp_matrix!(solver, D, pts, ws, k; multithreaded=true)
#
# Workspace wrapper for `construct_cfie_kress_dlp_matrix!`.
function construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    return construct_cfie_kress_dlp_matrix!(solver,D,pts,ws.Rmat,ws.Gs,ws.parr,ws.offs,k;multithreaded=multithreaded)
end

# construct_cfie_kress_dlp_matrix!(solver, D, pts, k; multithreaded=true)
#
# Convenience wrapper for `construct_cfie_kress_dlp_matrix!` that first builds
# the required CFIE-Kress workspace internally. Useful when the DLP matrix is needed only once.
function construct_cfie_kress_dlp_matrix!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_kress_workspace(solver,pts)
    return construct_cfie_kress_dlp_matrix!(solver,D,pts,ws,k;multithreaded=multithreaded)
end

# adjoint_K_from_dlp_matrix(D, ds)
#
# Construct the discrete adjoint double-layer matrix K' from the discrete
# double-layer matrix D using the weighted transpose identity.
# On the boundary, the continuous adjoint double-layer operator K' is the
# adjoint of K with respect to the L²(ds) pairing. In a Nyström discretization
# with quadrature weights ds_j, this means that the discrete adjoint is not the
# plain transpose of D, but the weighted transpose
#
#     K' = W^{-1} Dᵀ W,
#
# where
#
#     W = diag(ds_1, ..., ds_N).
#
# Equivalently, entrywise,
#
#     K'_{ij} = D_{ji} ds_j / ds_i.
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

# adjoint_K_action_from_dlp_matrix(D, μ, ds)
#
# Apply the discrete adjoint double-layer operator K' to a vector μ using the
# weighted transpose identity, without explicitly forming K'.
#
# Mathematical meaning:
# In the Nyström discretization associated with physical arc-length weights
# `ds`, the adjoint double-layer operator satisfies
#
#     K' = W^{-1} Dᵀ W,
#
# where W = diag(ds).
#
# Therefore its action on a vector μ is
#
#     K'μ = W^{-1} Dᵀ (W μ).
#
# The action form is cheaper and numerically cleaner than explicitly building
# the full matrix K'.
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

# cfie_kress_adjoint_K_action(solver, layer_pot, pts, ws, k; multithreaded=true)
#
# Compute the adjoint double-layer action K'μ needed in the CFIE boundary
# function recovery formula for the periodic Kress family.
#
# Mathematical meaning:
# If the CFIE wavefunction is written as
#
#     ψ = -(D + i k S) μ,
#
# then taking the interior normal derivative of the single-layer part produces
#
#     ∂_n^- Sμ = (-1/2 I + K') μ
#
# under the outward-normal convention used here.
#
# Therefore the final physical boundary function requires K'μ. This helper
# computes it in three steps:
#
# 1. assemble the pure DLP matrix D(k) on the full multi-component boundary,
# 2. flatten the physical arc-length weights ds,
# 3. apply the weighted transpose identity
#
#        K'μ = W^{-1} Dᵀ W μ.
#
# Notes:
# This function does not include the jump term `-μ/2`; it returns only the K'
# part. The jump is inserted later in `boundary_function`.
function cfie_kress_adjoint_K_action(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_pot::AbstractVector{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    Ntot=ws.Ntot
    length(layer_pot)==Ntot || error("layer_pot length mismatch in cfie_kress_adjoint_K_action")
    D=Matrix{Complex{T}}(undef,Ntot,Ntot)
    construct_cfie_kress_dlp_matrix!(solver,D,pts,ws,k;multithreaded=multithreaded)
    ds=flatten_cfie_ds(pts)
    return adjoint_K_action_from_dlp_matrix(D,layer_pot,ds)
end

# cfie_kress_adjoint_K_action(solver, layer_pot, pts, k; multithreaded=true)
#
# Convenience wrapper for `cfie_kress_adjoint_K_action` that builds the
# CFIE-Kress workspace internally. Useful when need only one application of adjoint K'.
function cfie_kress_adjoint_K_action(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_pot::AbstractVector{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_kress_workspace(solver,pts)
    return cfie_kress_adjoint_K_action(solver,layer_pot,pts,ws,k;multithreaded=multithreaded)
end

"""
    boundary_function(solver, layer_pot, pts, ws, k; multithreaded=true)

Construct the Dirichlet boundary function

    u = ∂_n ψ |_{∂Ω}

from the CFIE layer density for the periodic Kress family.
In this implementation the reconstructed wavefunction is represented as

    ψ = -(D + i k S) μ,

where:
- `D` is the double-layer potential,
- `S` is the single-layer potential,
- `μ` is the CFIE layer density.

Taking the interior normal derivative gives

    ∂_n ψ = -Nμ - i k ∂_n^- Sμ.

Using the standard jump relation for the single-layer operator with outward
normal,

    ∂_n^- Sμ = (-1/2 I + K') μ,

we obtain

    u = ∂_n ψ
      = -Nμ - i k (-1/2 I + K') μ.

1. Compute the hypersingular contribution

       Nμ = hypersingular_maue_kress(...)

   using Maue regularization:

       Nμ = ∂_s S(∂_s μ) + k^2 n · S(n μ)

2. Compute the adjoint double-layer action

       K'μ

   via the weighted transpose of the DLP matrix.

3. Combine the terms as

       u = -Nμ - i k (-μ/2 + K'μ).

Inputs
------
- `solver`: One of the periodic CFIE-Kress solvers, with or without corners.
- `layer_pot`: Global CFIE density μ.
- `pts`: Boundary discretization (one `BoundaryPointsCFIE` per connected component).
- `ws`: Precomputed CFIE Kress workspace.
- `k`: Wavenumber.
- `multithreaded`: Enables threaded DLP assembly for K'.
Output
-------
    - `pts::Vector{BoundaryPointsCFIE{T}}`: the input boundary discretization, returned for convenience.
    - `u ≈ ∂_n ψ ::Vector{Complex{T}}` at all boundary nodes, in the same ordering as `layer_pot`.
"""
function boundary_function(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_pot::AbstractVector{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    Nμ=hypersingular_maue_kress(solver,layer_pot,pts,ws,k)
    Kpμ=cfie_kress_adjoint_K_action(solver,layer_pot,pts,ws,k;multithreaded=multithreaded)
    u= -2*Nμ-Complex{T}(0,k).*(layer_pot+Kpμ)
    nrlz=_rellich(pts,u,k)
    return pts,u./sqrt(nrlz)
end

"""
    boundary_function(solver, layer_pot, pts, k; multithreaded=true)

Recover the physical boundary function ∂_n ψ from the CFIE layer density,
building the CFIE-Kress workspace internally.

This is a convenience wrapper around `boundary_function(solver, layer_pot, pts, ws, k)`.
"""
function boundary_function(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_pot::AbstractVector{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},billiard::Bi,k::T;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    ws=build_cfie_kress_workspace(solver,pts)
    return boundary_function(solver,layer_pot,pts,ws,k;multithreaded=multithreaded)
end

"""
    boundary_function(solver, layer_pot, pts, ks; multithreaded=true)

Recover the physical boundary function ∂_n ψ for multiple wavenumbers, building the CFIE-Kress workspace internally for each.
This is a convenience wrapper around `boundary_function(solver, layer_pot, pts, ws, k)` that loops over multiple wavenumbers. The output is a vector of boundary functions, one per wavenumber.
"""
function boundary_function(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},layer_pot::AbstractVector{<:AbstractVector{Complex{T}}},pts::AbstractVector{<:AbstractVector{BoundaryPointsCFIE{T}}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    us=Vector{Vector{Complex{T}}}(undef,length(pts))
    for i in eachindex(ks)
        ws=build_cfie_kress_workspace(solver,pts[i])
        _,u=boundary_function(solver,layer_pot[i],pts[i],ws,ks[i];multithreaded=multithreaded)
        us[i]=u
    end
    return pts,us
end
