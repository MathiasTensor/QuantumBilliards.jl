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
Then this `Eigenstate` object can be used as the input to this function to get the boundary function, arclength values, and norm.

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
- `norm<:Real`: the norm of the boundary function on the whole boundary.
"""
function boundary_function(state::S;b=5.0) where {S<:AbsState}
    vec=state.vec
    k=state.k
    k_basis=state.k_basis
    new_basis=state.basis
    billiard=state.billiard
    T=eltype(vec)
    boundary=billiard.desymmetrized_full_boundary
    crv_lengths=(crv.length for crv in boundary)
    sampler=FourierNodes([2,3,5],collect(crv_lengths))
    L=billiard.length
    N=max(round(Int,k*L*b/(2π)),512)
    pts=boundary_coords_desymmetrized_full_boundary(billiard,sampler,N)
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
    acc=zero(T)
    @inbounds @simd for i in eachindex(u) # boundary norm: ∫ |u|^2 (n·x) ds / (2k^2) no temps
        n=pts.normal[i]
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i] # w_i = (n·x) ds
        acc+=w*abs2(u[i]) # accumulate w_i*|u_i|^2
    end
    norm=acc/(2*k^2)
    @blas_1 return u,pts,norm
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
        try # the @. macro can faill in gradient_matrices when multithreading
            vec=X[i] # vector of vectors
            dim=length(vec)
            dim=dim-get_evanescent_dim_in_CompositeBasis(basis) # this is to prevent double counting the EWP
            dim=rescale_dimension(basis,dim)
            new_basis=resize_basis(basis,billiard,dim,ks[i]) # dim here should be for the main function
            state=Eigenstate(ks[i],vec,tens[i],new_basis,billiard)
            u,pts,_=boundary_function(state;b=b) # pts is BoundaryPoints and has information on ds and x
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

#########################################################################################################
################### BoundaryIntegralMethod, DLP_kress and DLP_kress_global_corners ######################
#########################################################################################################

# single vec per pts
function boundary_function(solver::BoundaryIntegralMethod,layer_pot::AbstractVector{N},pts::BoundaryPoints,billiard::Bi) where {N<:Number,Bi<:AbsBilliard}
    pts=apply_symmetries_to_boundary_points(pts,solver.symmetries,billiard)
    u=apply_symmetries_to_boundary_function(layer_pot,solver.symmetries)
    return pts,u
end

# multi vec per non-common pts
function boundary_function(solver::BoundaryIntegralMethod,layer_pot::Vector{<:AbstractVector{N}},pts::Vector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    us_all=Vector{Vector{N}}(undef,length(pts))
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    @use_threads multithreading=multithreaded for i in eachindex(pts)
        pts_all[i]=apply_symmetries_to_boundary_points(pts[i],solver.symmetries,billiard)
        us_all[i]=apply_symmetries_to_boundary_function(layer_pot[i],solver.symmetries)
    end
    return pts_all,us_all
end

function boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_pot::AbstractVector{N},pts::BoundaryPointsCFIE{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts=apply_symmetries_to_boundary_points(pts,solver.symmetries,billiard)
    layer_pot=apply_symmetries_to_boundary_function(layer_pot,solver.symmetries)
    return pts,layer_pot
end

function boundary_function(solver::Union{DLP_kress,DLP_kress_global_corners},layer_pot::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPointsCFIE},billiard::Bi;multithreaded::Bool=true) where {N<:Number,Bi<:AbsBilliard}
    pts_all=Vector{typeof(apply_symmetries_to_boundary_points(pts[1],solver.symmetries,billiard))}(undef,length(pts))
    layer_pot_all=Vector{typeof(apply_symmetries_to_boundary_function(layer_pot[1],solver.symmetries))}(undef,length(pts))
    @use_threads multithreading=multithreaded for i in eachindex(pts)
        pts_all[i]=apply_symmetries_to_boundary_points(pts[i],solver.symmetries,billiard)
        layer_pot_all[i]=apply_symmetries_to_boundary_function(layer_pot[i],solver.symmetries)
    end
    return pts_all,layer_pot_all
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
    u,s,norm=boundary_function(state;b=b)
    return momentum_function(u,s)
end

