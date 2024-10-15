#include("../abstracttypes.jl")
#include("../utils/billiardutils.jl")
#include("../utils/gridutils.jl")
#include("../solvers/matrixconstructors.jl")
using FFTW, SpecialFunctions, JLD2

#this takes care of singular points
function regularize!(u)
    idx = findall(isnan, u)
    for i in idx
        if i != 1
            u[i] = (u[i+1] + u[i-1])/2.0
        else
            u[i] = (u[i+1] + u[end])/2.0
        end
    end
end

function boundary_function(state::S; b=5.0) where {S<:AbsState}
    let vec = state.vec, k = state.k, k_basis = state.k_basis, new_basis = state.basis, billiard=state.billiard
        type = eltype(vec)
        boundary = billiard.full_boundary
        crv_lengths = [crv.length for crv in boundary]
        sampler = FourierNodes([2,3,5],crv_lengths)
        L = billiard.length
        N = max(round(Int, k*L*b/(2*pi)), 512)
        pts = boundary_coords(billiard, sampler, N)
        dX, dY = gradient_matrices(new_basis, k_basis, pts.xy)
        nx = getindex.(pts.normal,1)
        ny = getindex.(pts.normal,2)
        dX = nx .* dX 
        dY = ny .* dY
        U::Array{type,2} = dX .+ dY
        u::Vector{type} = U * vec
        regularize!(u)
        #compute the boundary norm
        w = dot.(pts.normal, pts.xy) .* pts.ds
        integrand = abs2.(u) .* w
        norm = sum(integrand)/(2*k^2)
        #println(norm)
        return u, pts.s::Vector{type}, norm
    end
end

function boundary_function(state_bundle::S; b=5.0) where {S<:EigenstateBundle}
    let X = state_bundle.X, k_basis = state_bundle.k_basis, ks = state_bundle.ks, new_basis = state_bundle.basis, billiard=state_bundle.billiard 
        type = eltype(X)
        boundary = billiard.full_boundary
        crv_lengths = [crv.length for crv in boundary]
        sampler = FourierNodes([2,3,5],crv_lengths)
        L = billiard.length
        N = max(round(Int, k_basis*L*b/(2*pi)), 512)
        pts = boundary_coords(billiard, sampler, N)
        dX, dY = gradient_matrices(new_basis, k_basis, pts.xy)
        nx = getindex.(pts.normal,1)
        ny = getindex.(pts.normal,2)
        dX = nx .* dX 
        dY = ny .* dY
        U::Array{type,2} = dX .+ dY
        u_bundle::Matrix{type} = U * X
        for u in eachcol(u_bundle)
            regularize!(u)
        end
        #compute the boundary norm
        w = dot.(pts.normal, pts.xy) .* pts.ds
        norms = [sum(abs2.(u_bundle[:,i]) .* w)/(2*ks[i]^2) for i in eachindex(ks)]
        #println(norm)
        us::Vector{Vector{type}} = [u for u in eachcol(u_bundle)]
        return us, pts.s, norms
    end
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
function boundary_function(state_data::StateData, billiard::Bi, basis::Ba; b=5.0) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks = state_data.ks
    tens = state_data.tens
    X = state_data.X
    us = Vector{Vector{eltype(ks)}}(undef, length(ks))
    s_vals = Vector{Vector{eltype(ks)}}(undef, length(ks))
    norms = Vector{eltype(ks)}(undef, length(ks))
    for i in eachindex(ks) 
        vec = X[i] # vector of vectors
        dim = length(vec)
        new_basis = resize_basis(basis, billiard, dim, ks[i])
        state = Eigenstate(ks[i], vec, tens[i], new_basis, billiard)
        u, s, norm = boundary_function(state; b=b)
        us[i] = u
        s_vals[i] = s
        norms[i] = norm
    end
    return ks, us, s_vals, norms
end

"""
    save_boundary_function(us, s_vals; filename::String="boundary_values.jld2")

Saves the results of the boundary_function with the `StateData` input. Primarly useful for creating efficient input to the husimi function constructor.

# Arguments
- `us::Vector{Vector}`: A vector of vectors containing the boundary functions (the u functions). Each inner vector corresponds to a wave number `ks[i]`.
- `s_vals::Vector{Vector}`: A vector of vectors containing the positions of the boundary points (the s values). Each inner vector corresponds to a wave number `ks[i]`.
- `filename::String`: The name of the jld2 file to save the boundary values to. Default is "boundary_values.jld2".

# Returns
- `Nothing`
"""
function save_boundary_function(us, s_vals; filename::String="boundary_values.jld2")
    @save filename us s_vals
end

"""
    read_boundary_function(filename::String="boundary_values.jld2")

Load the boundary function from a jld2 file.
# Arguments
- `filename::String`: The name of the jld2 file to load the boundary values from. Default is "boundary_values.jld2".

# Returns
- `us::Vector{Vector}`: A vector of vectors containing the boundary functions (the u functions). Each inner vector corresponds to a wave number `ks[i]`.
- `s_vals::Vector{Vector}`: A vector of vectors containing the positions of the boundary points (the s values). Each inner vector corresponds to a wave number `ks[i]`.
"""
function read_boundary_function(filename::String="boundary_values.jld2")
    @load filename us s_vals
    return us, s_vals
end

function momentum_function(u,s)
    fu = rfft(u)
    sr = 1.0/diff(s)[1]
    ks = rfftfreq(length(s),sr).*(2*pi)
    return abs2.(fu)/length(fu), ks
end

function momentum_function(state::S; b=5.0) where {S<:AbsState}
    u, s, norm = boundary_function(state; b)
    return momentum_function(u,s)
end

#this can be optimized by usinf FFTW plans
function momentum_function(state_bundle::S; b=5.0) where {S<:EigenstateBundle}
    us, s, norms = boundary_function(state_bundle; b)
    mf, ks = momentum_function(us[1],s)
    type = eltype(mf)
    mfs::Vector{Vector{type}} = [mf]
    for i in 2:length(us)
        mf, ks = momentum_function(us[i],s)
        push!(mfs,mf)
    end
    return mfs, ks
end






###### ADDITIONS ########










# Helper for momentum function calculations. We need this one since we will require much point information like xy, s,...
"""
    setup_momentum_density(state::S; b::Float64=5.0) where {S<:AbsState}

Prepares the necessary data for computing the momentum density from a given state.

# Arguments
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.

# Returns
- `u_values`: A vector of type `Vector{T}` containing the computed eigenvector components.
- `pts`: A structure containing boundary points and related information.
- `k`: The wave number extracted from the state.

# Description
This function extracts the eigenvector components (`u_values`), boundary points (`pts`), and wave number (`k`) from the provided `state`. It sets up the necessary variables for computing the radially and angularly integrated momentum densities.
The function internally calls `boundary_coords` to obtain the boundary coordinates and uses `gradient_matrices` to compute the derivatives required for `u_values`.

# Notes
- The parameter `b` affects the number of boundary points used in the computation. A higher value results in more points.
- Ensure that the `state` object contains the necessary attributes (`vec`, `k`, `k_basis`, `basis`, `billiard`).
"""
function setup_momentum_density(state::S; b::Float64=5.0) where {S<:AbsState}
    let vec = state.vec, k = state.k, k_basis = state.k_basis, new_basis = state.basis, billiard=state.billiard
        type = eltype(vec)
        boundary = billiard.full_boundary
        crv_lengths = [crv.length for crv in boundary]
        sampler = FourierNodes([2,3,5], crv_lengths)
        L = billiard.length
        N = max(round(Int, k*L*b/(2*pi)), 512)
        # Call boundary_coords to get pts
        pts = boundary_coords(billiard, sampler, N)
        # Compute U as in boundary_function
        dX, dY = gradient_matrices(new_basis, k_basis, pts.xy)
        nx = getindex.(pts.normal,1)
        ny = getindex.(pts.normal,2)
        dX = nx .* dX
        dY = ny .* dY
        U = dX .+ dY
        u_values = U * vec
        regularize!(u_values)
        return u_values, pts, k
    end
end

"""
    momentum_representation_of_state(state::S; b::Float64=5.0) :: Function where {S<:AbsState}

Returns a function that computes the momentum representation of a given quantum state `state` for any momentum vector `p`.

# Arguments
- `state::S`: The quantum state for which the momentum representation is computed. The type `S` is a subtype of `AbsState`.
- `b::Float64`: A parameter controlling the resolution and width of the momentum density (default = 5.0).

# Returns
- A function `mom(p::SVector{2, <:Real})` that computes the momentum representation for a momentum vector `p`.

"""
function momentum_representation_of_state(state::S; b::Float64=5.0) :: Function where {S<:AbsState}
    u_values, pts, k = setup_momentum_density(state; b)
    T = eltype(u_values)
    pts_coords = pts.xy  # Assuming pts.xy is already Vector{SVector{2, T}}
    num_points = length(pts_coords)
    
    function mom(p::SVector) # p = (px, py)
        local_sum = zero(Complex{T})
        k_squared = k^2
        p_squared = norm(p)^2
        
        if abs(p_squared - k_squared) > sqrt(eps(T))
            # Far from energy shell
            for i in 1:num_points
                local_sum += u_values[i] * exp(im*(pts_coords[i][1]*p[1] + pts_coords[i][2]*p[2]))
            end
            return 1/(p_squared - k_squared) * (1/(2*pi)) * local_sum
        else
            # Near energy shell, use approximation by Backer
            for i in 1:num_points
                phase_term = pts_coords[i][1]*p[1] + pts_coords[i][2]*p[2]
                local_sum += u_values[i] * exp(im*phase_term) * phase_term
            end
            return -im/(4*pi*k_squared) * local_sum
        end
    end
    return mom
end

"""
    computeRadiallyIntegratedDensityFromState(state::S; b::Float64=5.0) where {S<:AbsState}

Computes the radially integrated momentum density function `I(φ)` from a given state.

# Arguments
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.

# Returns
- A function `I_phi(φ::T)` that computes the radially integrated momentum density at a given angle `φ`.

# Description
This function calculates the radially integrated momentum density based on the eigenvector components and boundary points extracted from the `state`. It returns a function `I_phi(φ)` that computes the density at any given angle `φ`.
"""
function computeRadiallyIntegratedDensityFromState(state::S; b::Float64=5.0) :: Function where {S<:AbsState}
    u_values, pts, k = setup_momentum_density(state; b)
    T = eltype(u_values)
    pts_coords = pts.xy  # Assuming pts.xy is already Vector{SVector{2, T}}
    num_points = length(pts_coords)
    function I_phi(phi)
        I_phi_array = zeros(T, Threads.nthreads())
        p = k
        Threads.@threads for i in 1:num_points
            thread_id = Threads.threadid() # for indexing threads. Maybe not necessary since we just take the sum in the end
            I_phi_i = zero(T)
            for j in 1:num_points
                delta_x = pts_coords[i][1] - pts_coords[j][1]
                delta_y = pts_coords[i][2] - pts_coords[j][2]
                alpha = abs(cos(phi) * delta_x + sin(phi) * delta_y)
                x = alpha * p
                if abs(x) < sqrt(eps(T))
                    x = sqrt(eps(T))
                end
                Si_x = sinint(x)
                Ci_x = cosint(x)
                f_x = sin(x) * Ci_x - cos(x) * Si_x
                I_phi_i += f_x * u_values[i] * u_values[j]
            end
            I_phi_array[thread_id] += I_phi_i
        end
        I_phi_total = sum(I_phi_array)
        return abs((one(T) / (T(8) * T(pi)^2)) * I_phi_total)
    end
    return I_phi
end

"""
    computeAngularIntegratedMomentumDensityFromState(state::S; b::Float64=5.0) where {S<:AbsState}

Computes the angularly integrated momentum density function `R(r)` from a given state.

# Arguments
- `state::S`: An instance of a subtype of `AbsState`, representing the quantum state.
- `b::Float64=5.0`: An optional parameter controlling the number of boundary points. Defaults to `5.0`.

# Returns
- A function `R_r(r::T)` that computes the angularly integrated momentum density at a given radius `r`.

# Description
This function calculates the angularly integrated momentum density based on the eigenvector components and boundary points extracted from the `state`. It returns a function `R_r(r)` that computes the density at any given radius `r`.
"""
function computeAngularIntegratedMomentumDensityFromState(state::S; b::Float64=5.0) :: Function where {S<:AbsState}
    u_values, pts, k = setup_momentum_density(state; b)
    T = eltype(u_values)
    pts_coords = pts.xy  # Assuming pts.xy is already Vector{SVector{2, T}}
    k_squared = k^2
    epsilon = sqrt(eps(T))
    num_points = length(pts_coords)
    function R_r(r)
        if abs(r - sqrt(k_squared)) < epsilon
            # This just uses Backer's first approximation to the R(r) at the wavefunctions's k value
            R_r_array = zeros(T, Threads.nthreads())
            Threads.@threads for i in 1:num_points
                thread_id = Threads.threadid()
                R_r_i = zero(T)
                for j in 1:num_points
                    delta_x = pts_coords[i][1] - pts_coords[j][1]
                    delta_y = pts_coords[i][2] - pts_coords[j][2]
                    distance = hypot(delta_x, delta_y)
                    J0_value = Bessels.besselj(0, distance * r)
                    J2_value = Bessels.besselj(2, distance * r)
                    R_r_i += u_values[i] * u_values[j] * distance^2 * 0.5 * (J2_value - J0_value)
                end
                R_r_array[thread_id] += R_r_i
            end
        return 1/(16*pi*k) * sum(R_r_array)
        end
        R_r_array = zeros(T, Threads.nthreads())
        Threads.@threads for i in 1:num_points
            thread_id = Threads.threadid()
            R_r_i = zero(T)
            for j in 1:num_points
                delta_x = pts_coords[i][1] - pts_coords[j][1]
                delta_y = pts_coords[i][2] - pts_coords[j][2]
                distance = hypot(delta_x, delta_y)
                J0_value = Bessels.besselj(0, distance * r)
                R_r_i += u_values[i] * u_values[j] * J0_value
            end
            R_r_array[thread_id] += R_r_i
        end
        return (r / (r^2 - k_squared)^2) * sum(R_r_array)
    end
    return R_r
end



