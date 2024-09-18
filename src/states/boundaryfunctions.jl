#include("../abstracttypes.jl")
#include("../utils/billiardutils.jl")
#include("../utils/gridutils.jl")
#include("../solvers/matrixconstructors.jl")
using FFTW, SpecialFunctions

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


# Helper for momentum function calculations
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


function computeRadiallyIntegratedDensityFromState(state::S; b::Float64=5.0) where {S<:AbsState}
    # Set up the necessary variables
    u_values, pts, k = setup_momentum_density(state; b)
    T = eltype(u_values)
    pts_coords = pts.xy  # Assuming pts.xy is already Vector{SVector{2, T}}
    num_points = length(pts_coords)
    function I_phi(phi::T)
        I_phi_array = zeros(T, nthreads())
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
        return (one(T) / (T(8) * T(pi)^2)) * I_phi_total
    end
    return I_phi
end