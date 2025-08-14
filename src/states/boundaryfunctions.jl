using FFTW, SpecialFunctions, JLD2, ProgressMeter

#this takes care of singular points
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

When constructing the boundary function u we need to sometimes shift the starting point on the full boundary. This is a helper function that shifts all the fields in the BoundaryPoints struct such that all of them are shifted by the shift_s field in the billiard struct.

# Arguments
- `billiard::Bi<:AbsBilliard`: the billiard object.
- `u::Vector`: the boundary function vector obtained from the ull_boundary.
- `pts::BoundaryPoints`: the struct containing the boundary points on the full_boundary.

# Returns
- `BoundaryPoints`: the struct containing the boundary points on the full_boundary with the starting point shifted.
- `u::Vector`: the boundary function vector that is correctly shifted.
"""
function shift_starting_arclength(billiard::Bi,u::Vector{T},pts::BoundaryPoints{T}) where {Bi<:AbsBilliard, T<:Real}
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
        shifted_s .= shifted_s .- s_offset  # Subtract the first value to make it zero
        shifted_s .= mod.(shifted_s,L_effective)  # Wrap around to maintain continuity
        pts=BoundaryPoints(shifted_xy,shifted_normal,shifted_s,shifted_ds)
        u=shifted_u
    end
    return pts,u
end

# Wrapper for multi u case
"""
    shift_starting_arclength(billiard::Bi, u_bundle::Matrix, pts::BoundaryPoints) where {Bi<:AbsBilliard}

Wrapper for the arclength shift when we have a StateBundle as the input to the boundary_function. This is the an efficient way to handle many boundary u functions.

# Arguments
- `billiard::Bi<:AbsBilliard`: the billiard object.
- `u_bundle::Matrix`: the boundary function matrix obtained from the StateBundle input.
- `pts::BoundaryPoints`: the struct containing the boundary points on the full_boundary.

# Returns
- `BoundaryPoints`: the struct containing the boundary points on the full_boundary with the starting point shifted.
- `u_bundle::Matrix`: the boundary function matrix that is shifted.
"""
function shift_starting_arclength(billiard::Bi,u_bundle::Matrix,pts::BoundaryPoints) where {Bi<:AbsBilliard}
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
        for i in 1:size(u_bundle, 2)
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

"""
    rescale_dimension(basis::Ba, dim::T) where {T<:Real, Ba<:AbsBasis}

Helper/hack function to rescaled the dimension of RealPlaneWaves struct b/c due to the basis unique parity_pattern function which rescales the basis in a way that it multiplies it with the length of possible parity combinations. Since we require one in the fundamental domain for boundary function construction this effectively divides the dimension of the basis with the parity combination length.

# Arguments
- `basis::Ba`: the basis object.
- `dim::Integer`: the dimension to rescale.

# Returns
- `dim::Integer`: the rescaled dimension.
"""
@inline function rescale_dimension(basis::Ba,dim::Integer) where {Ba<:AbsBasis}
    main_basis=basis isa CompositeBasis ? basis.main : basis
    if main_basis isa RealPlaneWaves # hack since RealPlaneWaves have problems
        if isnothing(main_basis.symmetries[1])
            dim=Int(dim/4) # always works by construction of parity_pattern
        elseif (main_basis.symmetries[1] isa Reflection)
            if (main_basis.symmetries[1].axis==:x_axis) || (main_basis.symmetries[1].axis==:y_axis)
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

##########################

"""
    boundary_function(state::S; b=5.0) where {S<:AbsState}

Low-level function that constructs the boundary function and it's associated arclength `s` values alond the `desymmetrized_full_boundary` to which symmetries are being applied. This effectively constructs the boundary function on the whole boundary through applying symmetries to the `desymmetrized_full_boundary`. It also constructs the norm of the boundary function on the whole boundary (after symmetry application) as norm = ∮u(s)⟨r(s),n(s)⟩ds.

# Arguments
- `state<:AbsState`: typically the eigenstate associated with a particular solution to the problem on the fundamental boundary.
- `b=5.0`: optional parameter for point scaling in the construction of evaluation points on the boundary.

# Returns
- `u::Vector{<:Real}`: the boundary function evaluated at the points on the boundary.
- `pts.s::Vector{<:Real}`: the arclength s values along the boundary.
- `norm<:Real`: the norm of the boundary function on the whole boundary.
"""
function boundary_function(state::S;b=5.0) where {S<:AbsState}
    vec=state.vec
    k=state.k
    k_basis=state.k_basis
    new_basis=state.basis
    billiard=state.billiard
    type=eltype(vec)
    boundary=billiard.desymmetrized_full_boundary
    crv_lengths=[crv.length for crv in boundary]
    sampler=FourierNodes([2,3,5],crv_lengths)
    L=billiard.length
    N=max(round(Int,k*L*b/(2*pi)),512)
    pts=boundary_coords_desymmetrized_full_boundary(billiard,sampler,N)
    dX,dY=gradient_matrices(new_basis,k_basis,pts.xy)
    nx=getindex.(pts.normal,1)
    ny=getindex.(pts.normal,2)
    U=similar(dX)
    U.=nx.*dX.+ny.*dY
    u::Vector{type}=U*vec
    regularize!(u)
    pts=apply_symmetries_to_boundary_points(pts,new_basis.symmetries,billiard)
    u=apply_symmetries_to_boundary_function(u,new_basis.symmetries)
    pts,u=shift_starting_arclength(billiard,u,pts)
    w=dot.(pts.normal,pts.xy).*pts.ds
    integrand=abs2.(u).*w
    norm=sum(integrand)/(2*k^2)
    return u,pts.s::Vector{type},norm
end

"""
    boundary_function(state::S; b=5.0) where {S<:AbsState}

Multi-solution version of the commonly used `state` version of the `boundary_function`.
Low-level function that constructs the boundary function and it's associated arclength `s` values alond the `desymmetrized_full_boundary` to which symmetries are being applied. This effectively constructs the boundary function on the whole boundary through applying symmetries to the `desymmetrized_full_boundary`. It also constructs the norm of the boundary function on the whole boundary (after symmetry application) as norm = ∮u(s)⟨r(s),n(s)⟩ds.

# Arguments
- `state<:AbsState`: typically the eigenstate associated with a particular solution to the problem on the fundamental boundary.
- `b=5.0`: optional parameter for point scaling in the construction of evaluation points on the boundary.

# Returns
- `u::Vector{<:Real}`: the boundary function evaluated at the points on the boundary.
- `pts.s::Vector{<:Real}`: the arclength s values along the boundary.
- `norm<:Real`: the norm of the boundary function on the whole boundary.
"""
function boundary_function(state_bundle::S;b=5.0) where {S<:EigenstateBundle}
    X=state_bundle.X
    k_basis=state_bundle.k_basis
    ks=state_bundle.ks
    new_basis=state_bundle.basis
    billiard=state_bundle.billiard 
    type=eltype(X)
    boundary=billiard.desymmetrized_full_boundary
    crv_lengths=[crv.length for crv in boundary]
    sampler=FourierNodes([2,3,5],crv_lengths)
    L=billiard.length
    N=max(round(Int,k_basis*L*b/(2*pi)),512)
    pts=boundary_coords_desymmetrized_full_boundary(billiard,sampler,N)
    dX,dY=gradient_matrices(new_basis,k_basis,pts.xy)
    nx=getindex.(pts.normal,1)
    ny=getindex.(pts.normal,2)
    U=similar(dX)
    U.=nx.*dX.+ny.*dY
    u_bundle::Matrix{type}=U*X
    pts=apply_symmetries_to_boundary_points(pts,new_basis.symmetries,billiard)
    for u in eachcol(u_bundle)
        regularize!(u)
        u=apply_symmetries_to_boundary_function(u,new_basis.symmetries)
    end
    pts,u_bundle=shift_starting_arclength(billiard,u_bundle,pts)
    w=dot.(pts.normal,pts.xy).*pts.ds
    norms=[sum(abs2.(u_bundle[:,i]).* w)/(2*ks[i]^2) for i in eachindex(ks)]
    us::Vector{Vector{type}} = [u for u in eachcol(u_bundle)]
    return us,pts.s,norms
end

### NEW ### -> for the saving of the boundary functions of StateData construct a StateData wrapper for the Eigenstate version of boundary_function
"""
    boundary_function(state_data::StateData, billiard::Bi, basis::Ba; b=5.0) :: Tuple{Vector{T}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{T}} where {T, Bi<:AbsBilliard, Ba<:AbsBasis}

High level wrapper for the `Eigenstate` version of the `boundary_function`. This one is useful if we save the data from a version of `compute_spectrum` as a `StateData` object which automatically saves not just the `ks` and `tens` but also the vector of coefficients for the basis expansion of the wavefunction. This drastically saves time for computing the Husimi functions.

# Arguments
- `state_data::StateData`: An object of type `StateData` containing the `ks`, `tens`, and `X` coefficients for the basis expansion of the wavefunction.
- `billiard::Bi`: An object of type `Bi` representing the billiard geometry.
- `basis::Ba`: An object of type `Ba` representing the basis functions.


# Returns
- `ks`: A vector of the wave numbers.
- `us`: A vector of vectors containing the boundary functions (the u functions). Each inner vector corresponds to a wave number `ks[i]`.
- `s_vals`: A vector of vectors containing the positions of the boundary points (the s values). Each inner vector corresponds to a wave number `ks[i]`.
- `norms`: A vector of the norms of the boundary functions (the u functions). Each element corresponds to a wave number `ks[i]`.
"""
function boundary_function(state_data::StateData,billiard::Bi,basis::Ba;b=5.0) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    us=Vector{Vector{eltype(ks)}}(undef,length(ks))
    s_vals=Vector{Vector{eltype(ks)}}(undef,length(ks))
    norms=Vector{eltype(ks)}(undef,length(ks))
    valid_indices=fill(true,length(ks))
    progress=Progress(length(ks);desc="Constructing the u(s)...")
    for i in eachindex(ks) 
        try # the @. macro can faill in gradient_matrices when multithreading
            vec=X[i] # vector of vectors
            dim=length(vec)
            dim=dim-get_evanescent_dim_in_CompositeBasis(basis) # this is to prevent double counting the EWP
            dim=rescale_dimension(basis,dim)
            new_basis=resize_basis(basis,billiard,dim,ks[i])  # dim here should be for the main function
            state=Eigenstate(ks[i],vec,tens[i],new_basis,billiard)
            u,s,norm=boundary_function(state;b=b)
            us[i]=u
            s_vals[i]=s
            norms[i]=norm
        catch e
            println("Error while constructing the u(s) for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(progress)
    end
    ks=ks[valid_indices]
    us=us[valid_indices]
    s_vals=s_vals[valid_indices]
    norms=norms[valid_indices]
    return ks,us,s_vals,norms
end

"""
    boundary_function_with_points(state_data::StateData, billiard::Bi, basis::Ba; b=5.0) where {Bi<:AbsBilliard, Ba<:AbsBasis}

Computes the boundary functions us and the `BoundaryPoints` from which we can construct the wavefunction object using the boundary integtral definition.

# Arguments
`state_data::StateData`: The state data object that contains the `ks` and the `vec` from which we cna contruct the boundary points and the boundary function.
`billiard<:AbsBilliard`: The geometry of the billiard.
`basis<::AbsBilliard`: The basis of from which we contruct the wavefunction and the `vec` object.

# Returns
`ks::Vector`: The vector of eigenvalues. A convenience return.
`us::Vector{Vector}`: The vector of boundary functions (Vector) for each k that is a solution.
`pts_all::Vector{BoundaryPoints}`: A struct that contains the positions for which the u(s) (boundary function was evaluated) {pts.xy}, the arclengths that corresponds to these points {pts.s}, the normal vectors for the points we use {pts.normal} and the differences between the arclengths {pts.ds}. This is for every k in ks.
"""
function boundary_function_with_points(state_data::StateData,billiard::Bi,basis::Ba;b=5.0) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPoints{eltype(ks)}}(undef,length(ks))
    valid_indices=fill(true,length(ks))
    progress=Progress(length(ks);desc="Constructing the u(s)...")
    Threads.@threads for i in eachindex(ks) 
        try # the @. macro can faill in gradient_matrices when multithreading
            vec=X[i] # vector of vectors
            dim=length(vec)
            dim=dim-get_evanescent_dim_in_CompositeBasis(basis) # this is to prevent double counting the EWP
            dim=rescale_dimension(basis,dim)
            new_basis=resize_basis(basis,billiard,dim,ks[i]) # dim here should be for the main function
            state=Eigenstate(ks[i],vec,tens[i],new_basis,billiard)
            u,pts,_=setup_momentum_density(state;b=b) # pts is BoundaryPoints and has information on ds and x
            us_all[i]=u
            pts_all[i]=pts
        catch e
            println("Error while constructing the u(s) for k = $(ks[i])")
            valid_indices[i]=false
        end
        next!(progress)
    end
    ks=ks[valid_indices]
    us_all=us_all[valid_indices]
    pts_all=pts_all[valid_indices]
    return ks,us_all,pts_all
end

### BIM ###

"""
    boundary_function_BIM(solver::BoundaryIntegralMethod, u::Vector{T}, pts::BoundaryPointsBIM{T}, billiard::Bi) -> Tuple{BoundaryPoints{T}, Vector{T}} where {T<:Real, Bi<:AbsBilliard}

Processes the boundary function and associated boundary points by applying symmetries and shifting the starting point of the arclengths if necessary.

# Arguments
- `solver::BoundaryIntegralMethod`: The solver that holds the symmetry information.
- `u::Vector{T}`: The boundary function values.
- `pts::BoundaryPointsBIM{T}`: The boundary points in BIM representation.
- `billiard::Bi`: The billiard object defining the geometry.

# Returns
- `BoundaryPoints{T}`: The processed boundary points after applying symmetries and shifting the arclengths.
- `Vector{T}`: The processed boundary function.
"""
function boundary_function_BIM(solver::BoundaryIntegralMethod{T}, u::Vector{T}, pts::BoundaryPointsBIM{T}, billiard::Bi) where {T<:Real,Bi<:AbsBilliard}
    symmetries=SymmetryRuleBIM_to_Symmetry(solver.rule) 
    pts=BoundaryPointsMethod_to_BoundaryPoints(pts)
    regularize!(u)
    pts=apply_symmetries_to_boundary_points(pts,symmetries,billiard)
    u=apply_symmetries_to_boundary_function(Vector{Float64}(u),symmetries)
    pts,u=shift_starting_arclength(billiard,Vector{Float64}(u),pts)
    return pts,u::Vector{Float64}
end

"""
    boundary_function_BIM(solver::BoundaryIntegralMethod, us_all::Vector{Vector{T}}, pts_all::Vector{BoundaryPointsBIM{T}}, billiard::Bi) 
        -> Tuple{Vector{BoundaryPoints{T}}, Vector{Vector{T}}} where {T<:Real, Bi<:AbsBilliard}

Processes multiple boundary functions and their associated boundary points by applying symmetries and shifting the starting point of the arclengths if necessary.

# Arguments
- `solver::BoundaryIntegralMethod`: The solver that holds the symmetry information.
- `us_all::Vector{Vector{T}}`: A vector of boundary function values, where each element corresponds to a set of points in `pts_all`.
- `pts_all::Vector{BoundaryPointsBIM{T}}`: A vector of boundary points in BIM representation.
- `billiard::Bi`: The billiard object defining the geometry.

# Returns
- `Vector{BoundaryPoints{T}}`: A vector of processed boundary points after applying symmetries and shifting the arclengths.
- `Vector{Vector{T}}`: A vector of processed boundary functions corresponding to the processed boundary points.
"""
function boundary_function_BIM(solver::BoundaryIntegralMethod{T}, us_all::Vector{Vector{T}}, pts_all::Vector{BoundaryPointsBIM{T}}, billiard::Bi) where {T<:Real,Bi<:AbsBilliard}
    pts_ret=Vector{BoundaryPoints{T}}(undef,length(us_all))
    us_ret=Vector{Vector{T}}(undef,length(us_all))
    @showprogress for i in eachindex(us_all) 
        pts,u=boundary_function_BIM(solver,us_all[i],pts_all[i],billiard)
        pts_ret[i]=pts
        us_ret[i]=u
    end
    return pts_ret,us_ret
end

"""
    save_boundary_function!(ks, us, s_vals; filename::String="boundary_values.jld2")

Saves the results of the boundary_function with the `StateData` input. Primarly useful for creating efficient input to the husimi function constructor.

# Arguments
- `ks::Vector{Float64}`: A vector of the wave numbers.
- `us::Vector{Vector}`: A vector of vectors containing the boundary functions (the u functions). Each inner vector corresponds to a wave number `ks[i]`.
- `s_vals::Vector{Vector}`: A vector of vectors containing the positions of the boundary points (the s values). Each inner vector corresponds to a wave number `ks[i]`.
- `filename::String`: The name of the jld2 file to save the boundary values to. Default is "boundary_values.jld2".

# Returns
- `Nothing`
"""
function save_boundary_function!(ks, us, s_vals; filename::String="boundary_values.jld2")
    @save filename ks us s_vals
end

"""
     save_BoundaryPoints(ks::Vector, vec_bd_points::Vector{BoundaryPoints}, us::Vector{Vector}; filename::String="boundary_points.jld2") where {Bi<:AbsBilliard}

Saves the Vector of BoundaryPoints structs together us and with ks for convenience. This makes it easier to construct the wavefunction via the boundary integral for which we need the boundary points, the arclength differences and the arclengths. For this it needs to construct the BoundaryPoints via the Scaling Method.

```julia
struct BoundaryPoints{T} <: (AbsPoints where T <: Real)
    xy::Vector{SVector{2, T}}
    normal::Vector{SVector{2, T}}
    s::Vector{T}
    ds::Vector{T}
end
```

# Arguments
- `ks::Vector{Float64}`: A vector of the eigenvalues.
- `vec_bd_points::Vector{BoundaryPoints}`: A vector of BoundaryPoints structs. Each BoundaryPoints struct corresponds to a wave number `ks[i]`.
- `us::Vector{Vector}`: A vector of vectors containing the boundary functions (the u functions). Each inner vector corresponds to a wave number `ks[i]`.

# Returns
- `Nothing`
"""
function save_BoundaryPoints!(ks::Vector{T}, vec_bd_points::Vector{BoundaryPoints{T}}, us::Vector{Vector{T}}; filename::String="boundary_points.jld2") where {T<:Real}
    @save filename ks vec_bd_points us
end

"""
    read_BoundaryPoints(filename::String="boundary_points.jld2")

Read the saved Vector{BoundaryPoints} for later construction of the wavefunction.

```julia
struct BoundaryPoints{T} <: (AbsPoints where T <: Real)
    xy::Vector{SVector{2, T}}
    normal::Vector{SVector{2, T}}
    s::Vector{T}
    ds::Vector{T}
end
```

# Arguments
- `filename::String`: The name of the jld2 file to load the boundary points from. Default is "boundary_points.jld2".

# Returns
- `ks::Vector{<:Real}`: A vector of the wave numbers.
- `vec_bd_points::Vector{BoundaryPoints}`: A vector of BoundaryPoints structs. Each BoundaryPoints struct corresponds to a wave number `ks[i]`.
- `us::Vector{Vector}`: A vector of vectors containing the boundary functions (the u functions). Each inner vector corresponds to a wave number `ks[i]`.
"""
function read_BoundaryPoints(filename::String="boundary_points.jld2")
    @load filename ks vec_bd_points us
    return ks, vec_bd_points, us
end

"""
    read_boundary_function(filename::String="boundary_values.jld2")

Load the boundary function from a jld2 file.
# Arguments
- `filename::String`: The name of the jld2 file to load the boundary values from. Default is "boundary_values.jld2".

# Returns
- `ks::Vector{Float64}`: A vector of the wave numbers. 
- `us::Vector{Vector}`: A vector of vectors containing the boundary functions (the u functions). Each inner vector corresponds to a wave number `ks[i]`.
- `s_vals::Vector{Vector}`: A vector of vectors containing the positions of the boundary points (the s values). Each inner vector corresponds to a wave number `ks[i]`.
"""
function read_boundary_function(filename::String="boundary_values.jld2")
    @load filename ks us s_vals
    return ks, us, s_vals
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
    u,s,norm=boundary_function(state;b)
    return momentum_function(u,s)
end

#this can be optimized by usinf FFTW plans
"""
    momentum_function(state_bundle::S; b=5.0) where {S<:EigenstateBundle}

Computes the momentum distribution functions for all eigenstates in a given bundle of quantum states. Each eigenstate in the bundle has its momentum distribution computed, and the results are returned as a vector of momentum distributions.

# Arguments
- `state_bundle::S`: A bundle of quantum eigenstates of type `<:EigenstateBundle` which is a collection of states `<:AbsState`
- `b::Real`: Number of basis functions used to compute the wavefunctions on the boundary (default: `5.0`).

# Returns
- `Vector{Vector}`: A vector of DFT amplitude vectors, one for each eigenstate in the bundle.
- `Vector`: A vector of frequencies (`ks`) associated with the DFT components (shared across all eigenstates since marker on x-axis (not function of unique u just the length of s)).
"""
function momentum_function(state_bundle::S;b=5.0) where {S<:EigenstateBundle}
    us,s,norms=boundary_function(state_bundle;b)
    mf,ks=momentum_function(us[1],s)
    type=eltype(mf)
    mfs::Vector{Vector{type}}=[mf]
    for i in eachindex(us)[2:end]
        mf,ks=momentum_function(us[i],s)
        push!(mfs,mf)
    end
    return mfs,ks
end

# Helper for momentum function calculations. We need this one since we will require much point information like xy, s,...
"""
    setup_momentum_density(state::S; b::Float64=5.0) where {S<:AbsState}

Prepares the necessary data for computing the momentum density from a given state.

# Arguments
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.

# Returns
- `u_values`: A vector of type `Vector{T}` containing the computed eigenvector components.
- `pts::BoundaryPoints`: A structure containing boundary points and related information.
- `k`: The wave number extracted from the state.

# Description
This function extracts the eigenvector components (`u_values`), boundary points (`pts`), and wave number (`k`) from the provided `state`. It sets up the necessary variables for computing the radially and angularly integrated momentum densities.
The function internally calls `boundary_coords` to obtain the boundary coordinates and uses `gradient_matrices` to compute the derivatives required for `u_values`.

# Notes
- The parameter `b` affects the number of boundary points used in the computation. A higher value results in more points.
- Ensure that the `state` object contains the necessary attributes (`vec`, `k`, `k_basis`, `basis`, `billiard`).
"""
function setup_momentum_density(state::S;b::Float64=5.0) where {S<:AbsState}
    vec=state.vec
    k=state.k
    k_basis=state.k_basis
    new_basis=state.basis
    billiard=state.billiard
    type=eltype(vec)
    boundary=billiard.desymmetrized_full_boundary
    crv_lengths=[crv.length for crv in boundary]
    sampler=FourierNodes([2,3,5],crv_lengths)
    L=billiard.length
    N=max(round(Int,k*L*b/(2*pi)),512)
    pts=boundary_coords_desymmetrized_full_boundary(billiard,sampler,N)
    dX,dY=gradient_matrices(new_basis,k_basis,pts.xy)
    nx=getindex.(pts.normal,1)
    ny=getindex.(pts.normal,2)
    dX=nx.*dX
    dY=ny.*dY
    U=dX.+dY
    u=U*vec
    regularize!(u)
    pts=apply_symmetries_to_boundary_points(pts,new_basis.symmetries,billiard)
    u=apply_symmetries_to_boundary_function(u,new_basis.symmetries)
    pts,u=shift_starting_arclength(billiard,u,pts)
    return u,pts,k
end

function momentum_representation_of_wavefunction(us::Vector{T},pts::BP,k::T) :: Function where {T<:Real,BP<:AbsPoints}
    xy=pts.xy
    ds=pts.ds
    N=length(xy)
    k2=k^2
    function mom(p::SVector{2,T}) # ψ̂(p) from boundary data (Backer Eq. 11 off-shell; Eq. 13 on shell)
        p2=dot(p,p)
        sumI=zero(Complex{T})
        @inbounds for j in 1:N  # I(p) = ∫ u(s) e^{-i p·q(s)} ds  ≈ Σ u_j e^{-i p·q_j} ds_j
            phase=-(p[1]*xy[j][1]+p[2]*xy[j][2])
            sumI+=us[j]*exp(im*phase)*ds[j]
        end
        if abs(p2-k2)>sqrt(eps(T))
            return (one(T)/(2*pi))*sumI/(p2-k2) # we are in the ok zone
        else # on-shell
            sumOn=zero(Complex{T})
            @inbounds for j in 1:N # on-shell: ψ̂(p) = -(i/(4π k^2)) ∫ e^{-i p·q}(p·q) u(s) ds
                pq=p[1]*xy[j][1]+p[2]*xy[j][2]
                sumOn+=us[j]*exp(-im*pq)*pq*ds[j]
            end
            return -(im/(4*pi*k2))*sumOn
        end
    end
    return mom
end

# I(φ) (Eq. 25; real-u case drops the imaginary term)
function computeRadiallyIntegratedDensity(us::Vector{T},pts::BP,k::T) :: Function where {T<:Real,BP<:AbsPoints}
    xy=pts.xy
    ds=pts.ds
    N=length(xy)
    k2=k^2
    ε=sqrt(eps(T))
    # for the constant term -(1/(2k^2)) (∫ u ds)^2, vanishes for odd-symmetry classes
    s0=zero(T)
    @inbounds for j in 1:N
        s0+=us[j]*ds[j]
    end
    const_term=(s0^2)/(2*k2)
    function I_phi(phi::T)
        c=cos(phi);s=sin(phi)
        acc=zero(T)
        @inbounds for i in 1:N
            xi=xy[i];wi=ds[i];ui=us[i]
            for j in 1:N
                dx=xi[1]-xy[j][1];dy=xi[2]-xy[j][2]
                α=abs(c*dx+s*dy)
                x=α*k
                x=abs(x)<ε ? ε : x  # avoid 0 in special functions
                f=sin(x)*cosint(x)-cos(x)*sinint(x) # Eq. (24)
                acc+=ui*us[j]*wi*ds[j]*f
            end
        end
        return (one(T)/(8*pi^2))*acc-const_term
    end
    return I_phi
end

# R(r) (Eq. 28 general; Eq. 31 at r≈k)
function computeAngularIntegratedMomentumDensity(us::Vector{T},pts::BP,k::T) :: Function where {T<:Real,BP<:AbsPoints}
    xy=pts.xy
    ds=pts.ds
    N=length(xy)
    k2=k^2
    ε=sqrt(eps(T))
    function R_r(r::T)
        if abs(r-k)<ε
            acc=zero(T)
            @inbounds for i in 1:N
                xi=xy[i];wi=ds[i];ui=us[i]
                for j in 1:N
                    dx=xi[1]-xy[j][1];dy=xi[2]-xy[j][2]
                    d=hypot(dx,dy)
                    J0=Bessels.besselj(0,d*r)
                    J2=Bessels.besselj(2,d*r)
                    acc+=ui*us[j]*wi*ds[j]*(d^2*0.5*(J2-J0))
                end
            end
            return (one(T)/(16*pi*k))*acc # Eq. (31)
        else
            acc=zero(T)
            @inbounds for i in 1:N
                xi=xy[i];wi=ds[i];ui=us[i]
                for j in 1:N
                    d=hypot(xi[1]-xy[j][1],xi[2]-xy[j][2])
                    acc+=ui*us[j]*wi*ds[j]*Bessels.besselj(0,d*r)
                end
            end
            return (one(T)/(2*pi))*(r/((r^2-k2)^2))*acc  # Eq. (28)
        end
    end
    return R_r
end

##############################
# ψ̂(p) from boundary data   #
##############################
function momentum_representation_of_wavefunction_NEW(us::Vector{T}, pts::BP, k::T) :: Function where {T<:Real, BP<:AbsPoints}
        # precompute coordinates and weights
        N  = length(pts.xy)
        x  = Vector{T}(undef, N)
        y  = Vector{T}(undef, N)
        w  = Vector{T}(undef, N)          # w_j = u_j * ds_j
        @inbounds for j in 1:N
            x[j] = pts.xy[j][1]
            y[j] = pts.xy[j][2]
            w[j] = us[j]*pts.ds[j]
        end
        k2   = k*k
        thr  = sqrt(eps(T))
        inv2π = inv(T(2π))
        I_T   = complex(zero(T), one(T))  # 0 + 1im of type T
    
        function mom(p::SVector{2,T})
            px, py = p
            # I(p) = ∫ u(s) e^{-i p·q} ds  ≈  Σ w_j cis(-p·q_j)
            accI = zero(Complex{T})
            @inbounds @simd for j in 1:N
                accI += w[j] * cis(-(muladd(px, x[j], py*y[j])))
            end
            p2 = muladd(px, px, py*py)
            δ  = p2 - k2
            if abs(δ) > thr                          # off shell (Bäcker Eq. 11)
                return inv2π * accI / δ
            else                                     # on shell (Eq. 13)
                accOn = zero(Complex{T})
                @inbounds @simd for j in 1:N
                    pq = muladd(px, x[j], py*y[j])
                    accOn += us[j]*pts.ds[j] * cis(-pq) * pq
                end
                return -(I_T/(T(4π)*k2)) * accOn
            end
        end
        return mom
    end
    
    
    ########################################################
    # I(φ): radially integrated momentum density (Eq. 24–25)
    ########################################################
    function computeRadiallyIntegratedDensity_NEW(us::Vector{T}, pts::BP, k::T) :: Function where {T<:Real, BP<:AbsPoints}
        N  = length(pts.xy)
        x  = Vector{T}(undef, N)
        y  = Vector{T}(undef, N)
        w  = Vector{T}(undef, N)          # w_i = u_i * ds_i
        @inbounds for i in 1:N
            x[i] = pts.xy[i][1]
            y[i] = pts.xy[i][2]
            w[i] = us[i]*pts.ds[i]
        end
        # constant term  (∫u ds)^2/(2k^2)
        s0  = sum(w)
        const_term = (s0*s0) / (T(2)*k*k)
        thr = sqrt(eps(T))
        inv8π2 = inv(T(8π^2))
    
        function I_phi(φ::T)
            c, s = cos(φ), sin(φ)
            acc  = zero(T)
            # use upper triangle + symmetry factor 2 to halve work
            @inbounds for i in 1:N
                xi, yi, wi = x[i], y[i], w[i]
                # diagonal (i==i) contributes once
                αii = abs(c*0 + s*0)
                xii = max(abs(αii*k), thr)
                acc += wi*wi * (sin(xii)*cosint(xii) - cos(xii)*sinint(xii))
                for j in (i+1):N
                    α  = abs(muladd(c, xi - x[j], s*(yi - y[j])))
                    z  = max(abs(α*k), thr)
                    f  = sin(z)*cosint(z) - cos(z)*sinint(z)
                    acc += T(2)*wi*w[j]*f  # symmetry (i,j) + (j,i)
                end
            end
            return inv8π2*acc - const_term
        end
        return I_phi
    end
    
    
    #################################################################
    # R(r): angularly integrated momentum density (Eq. 28 / Eq. 31) #
    #################################################################
    function computeAngularIntegratedMomentumDensity_NEW(us::Vector{T}, pts::BP, k::T) :: Function where {T<:Real, BP<:AbsPoints}
        N  = length(pts.xy)
        x  = Vector{T}(undef, N)
        y  = Vector{T}(undef, N)
        w  = Vector{T}(undef, N)          # w_i = u_i * ds_i
        @inbounds for i in 1:N
            x[i] = pts.xy[i][1]
            y[i] = pts.xy[i][2]
            w[i] = us[i]*pts.ds[i]
        end
        k2    = k*k
        thr   = sqrt(eps(T))
        inv2π = inv(T(2π))
    
        function R_r(r::T)
            if abs(r - k) < thr               # near shell (Eq. 31)
                acc = zero(T)
                @inbounds for i in 1:N
                    xi, yi, wi = x[i], y[i], w[i]
                    # diagonal
                    z = zero(T)
                    J0 = Bessels.besselj(0, z) # = 1
                    J2 = Bessels.besselj(2, z) # = 0
                    acc += wi*wi * (zero(T)^2 * T(0.5) * (J2 - J0)) # = 0
                    for j in (i+1):N
                        dx = xi - x[j]; dy = yi - y[j]
                        d  = hypot(dx, dy)
                        z  = d*r
                        J0 = Bessels.besselj(0, z)
                        J2 = Bessels.besselj(2, z)
                        acc += T(2)*wi*w[j] * (d^2 * T(0.5) * (J2 - J0))
                    end
                end
                return acc / (T(16π)*k)
            else                               # general r (Eq. 28)
                acc = zero(T)
                @inbounds for i in 1:N
                    xi, yi, wi = x[i], y[i], w[i]
                    acc += wi*wi * Bessels.besselj(0, zero(T))  # = wi^2 * 1
                    for j in (i+1):N
                        d = hypot(xi - x[j], yi - y[j])
                        acc += T(2)*wi*w[j] * Bessels.besselj(0, d*r)
                    end
                end
                return inv2π * (r/((r*r - k2)^2)) * acc
            end
        end
        return R_r
    end


