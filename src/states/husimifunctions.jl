#include("../abstracttypes.jl")

using CircularArrays
using JLD2
using LinearAlgebra

function antisym_vec(x)
    v = reverse(-x[2:end])
    return append!(v,x)
end

function husimi_function(k,u,s,L; c = 10.0, w = 7.0)
    #c density of points in coherent state peak, w width in units of sigma
    #L is the boundary length for periodization
    #compute coherrent state weights
    N = length(s)
    sig = one(k)/sqrt(k) #width of the gaussian
    x = s[s.<=w*sig]
    idx = length(x) #do not change order here
    x = antisym_vec(x)
    a = one(k)/(2*pi*sqrt(pi*k)) #normalization factor in this version Hsimi is not noramlized to 1
    ds = (x[end]-x[1])/length(x) #integration weigth
    uc = CircularVector(u) #allows circular indexing
    gauss = @. exp(-k/2*x^2)*ds
    gauss_l = @. exp(-k/2*(x+L)^2)*ds
    gauss_r = @. exp(-k/2*(x-L)^2)*ds
    #construct evaluation points in p coordinate
    ps = collect(range(0.0,1.0,step = sig/c))
    #construct evaluation points in q coordinate
    q_stride = length(s[s.<=sig/c])
    q_idx = collect(1:q_stride:N)
    push!(q_idx,N) #add last point
    qs = s[q_idx]
    #println(length(qs))
    H = zeros(typeof(k),length(qs),length(ps))
    for i in eachindex(ps)   
        cs = @. exp(im*ps[i]*k*x)*gauss + exp(im*ps[i]*k*(x+L))*gauss_l + exp(im*ps[i]*k*(x-L))*gauss_r#imag part of coherent state
        for j in eachindex(q_idx)
            u_w = uc[q_idx[j]-idx+1:q_idx[j]+idx-1] #window with relevant values of u
            h = sum(cs.*u_w)
            #hi = sum(ci.*u_w)
            H[j,i] = a*abs2(h)
        end
    end

    ps = antisym_vec(ps)
    H_ref = reverse(H[:, 2:end]; dims=2)
    H = hcat(H_ref,H)
     
    return H, qs, ps    
end

function husimi_function(state::S;  b = 5.0, c = 10.0, w = 7.0) where {S<:AbsState}
    L = state.billiard.length
    k = state.k
    u, s, norm = boundary_function(state; b=b)
    return husimi_function(k,u,s,L; c = c, w = w)
end

function husimi_function(state_bundle::S;  b = 5.0, c = 10.0, w = 7.0) where {S<:EigenstateBundle}
    L = state_bundle.billiard.length
    ks = state_bundle.ks
    us, s, norm = boundary_function(state_bundle; b=b)
    H, qs, ps = husimi_function(ks[1],us[1],s,L; c = c, w = w)
    type = eltype(H)
    Hs::Vector{Matrix{type}} = [H]
    for i in 2:length(ks)
        H, qs, ps = husimi_function(ks[i],us[i],s,L; c = c, w = w)
        push!(Hs,H)
    end
    return Hs, qs, ps
end
#=
function coherent(q,p,k,s,L,m::Int)
    let x = s-q+m*L
        a = (k/pi)^0.25
        ft = exp(im*p*k*x) 
        gauss = exp(-k/2*x^2)
        return a*ft*gauss
    end 
end

function coherent(q,p,k,s,L,m::Int,b::Complex)
    let x = s-q+m*L
        a = (k*imag(b)/pi)^0.25
        ft = exp(im*p*k*x) 
        fb = exp(im*real(b)/2*k*x^2)
        gauss = exp(-imag(b)/2*k*x^2)
        return a*ft*fb*gauss
    end 
end
=#

### NEW ###

"""
    husimiAtPoint(k::T,s::Vector{T},u::Vector{T},L::T,q::T,p::T) where {T<:Real}

Calculates the Poincaré-Husimi function at point (q, p) in the quantum phase space.

Arguments:
- `k::T`: Wavenumber of the eigenstate.
- `s::Vector{T}`: Array of points on the boundary.
- `u::Vector{T}`: Array of boundary function values.
- `L::T`: Total length of the boundary (maximum(s)).
- `q::T`: Position coordinate in phase space.
- `p::T`: Momentum coordinate in phase space.

Returns:
- `T`: Husimi function value at (q, p).
"""
function husimiAtPoint(k::T,s::Vector{T},u::Vector{T},L::T,q::T,p::T) where {T<:Real}
    # original algorithm by Benjamin Batistić in python (https://github.com/clozej/quantum_billiards/blob/crt_public/src/CoreModules/HusimiFunctionsOld.py)
    ss = s.-q
    width = 4/sqrt(k)
    indx = findall(x->abs(x)<width, ss)
    si = ss[indx]
    ui = u[indx]
    ds = diff(s)  # length N-1
    ds = vcat(ds,L+s[1]-s[end]) # add Nth
    dsi = ds[indx]
    w = sqrt(sqrt(k/π)).*exp.(-0.5*k*si.*si).*dsi
    cr = w.*cos.(k*p*si)  # Coherent state real part
    ci = w.*sin.(k*p*si)  # Coherent state imaginary part
    h = dot(cr-im*ci,ui)  # Husimi integral (minus because of conjugation)
    return abs2(h)/(2*π*k) # not the actual normalization
end

"""
    husimiOnGrid(k::T,s::Vector{T},u::Vector{T},L::T,nx::Integer,ny::Integer) where {T<:Real}

Evaluates the Poincaré-Husimi function on a grid defined by the sizes nx for q and ny for p. The grids are then automatically generated from 0 -> L and -1 -> 1.

Arguments:
- `k::T`: Wavenumber of the eigenstate.
- `s::Vector{T}`: Array of points on the boundary.
- `u::Vector{T}`: Array of boundary function values.
- `L::T`: Total length of the boundary (maximum(s)).
- `nx::Integer`: Number of grid points in the position coordinate (q).
- `ny::Integer`: Number of grid points in the momentum coordinate (p).

Returns:
- `H::Matrix{T}`: Husimi function matrix of size (ny, nx).
- `qs::Vector{T}`: Array of q values used in the grid.
- `ps::Vector{T}`: Array of p values used in the grid.
"""
function husimiOnGrid(k::T,s::Vector{T},u::Vector{T},L::T,nx::Integer,ny::Integer) where {T<:Real}
    qs = range(0.0,stop=L,length = nx)
    ps = range(-1.0,stop=1.0,length = ny)
    H = zeros(T, nx, ny)
    
    Threads.@threads for idx_p in eachindex(ps)
        for idx_q in eachindex(qs)
            H[idx_q, idx_p] = husimiAtPoint(k,s,u,L,qs[idx_q],ps[idx_p])
        end
    end
    return H./sum(H),qs,ps
end

"""
    husimiOnGridOptimized(k::T, s::Vector{T}, u::Vector{T}, L::T, nx::Integer, ny::Integer) where {T<:Real}

Evaluates the Poincaré-Husimi function on a grid using vectorized operations.

Arguments:
- `k::T`: Wavenumber of the eigenstate.
- `s::Vector{T}`: Array of points on the boundary.
- `u::Vector{T}`: Array of boundary function values.
- `L::T`: Total length of the boundary (maximum(s)).
- `nx::Integer`: Number of grid points in the position coordinate (q).
- `ny::Integer`: Number of grid points in the momentum coordinate (p).

Returns:
- `H::Matrix{T}`: Husimi function matrix of size (ny, nx).
- `qs::Vector{T}`: Array of q values used in the grid.
- `ps::Vector{T}`: Array of p values used in the grid.
"""
function husimiOnGridOptimized(k::T, s::Vector{T}, u::Vector{T}, L::T, nx::Integer, ny::Integer) where {T<:Real}
    qs = range(0.0,stop=L,length=nx)
    ps = range(-1.0,stop=1.0,length=ny)
    N = length(s)
    ds = diff(s) # integration weights ds
    ds = vcat(ds,L+s[1]-s[end]) # ds has length N
    sqrt_k_pi = sqrt(k/π)
    norm_factor = sqrt(sqrt_k_pi)
    width = 4/sqrt(k)
    H = zeros(T,ny,nx)
    Threads.@threads for i_q = 1:nx
        q = qs[i_q]
        si = similar(s)
        w = similar(s)
        cr = similar(s)
        ci = similar(s)
        idx_start = searchsortedfirst(s,q-width) # Find indices within the window 
        idx_end = searchsortedlast(s,q+width)
        len_window = idx_end - idx_start + 1
        @views s_window = s[idx_start:idx_end] # Extract windowed data
        @views ui_window = u[idx_start:idx_end]
        @views dsi_window = ds[idx_start:idx_end]
        si_window = si[1:len_window]
        @inbounds @. si_window = s_window - q
        @inbounds @. w[1:len_window] = norm_factor*exp.(-0.5*k*si_window.^2).*dsi_window
        for i_p = 1:ny
            p = ps[i_p]
            kp = k*p
            @inbounds @. cr[1:len_window] = w[1:len_window].*cos.(kp.*si_window)
            @inbounds @. ci[1:len_window] = w[1:len_window].*sin.(kp.*si_window)
            h_real = @inbounds sum(cr[1:len_window].*ui_window)
            h_imag = -@inbounds sum(ci[1:len_window].*ui_window)  # Negative due to conjugation
            @inbounds H[i_p,i_q] = (h_real^2+h_imag^2)/(2*π*k)
        end
    end
    return H'./sum(H),qs,ps
end

"""
    function husimi_functions_from_StateData(state_data::StateData, billiard::Bi, basis::Ba;  b = 5.0, c = 10.0, w = 7.0) :: Tuple{Vector{Matrix{T}}, Vector{Vector{T}}, Vector{Vector{T}}} where {T<:Real, Bi<:AbsBilliard, Ba<:AbsBasis}

High level wrapper for the construction of the husimi functions from StateData which we compute as we run the compute_spectrum so we do not have to compute the eigenstate for each k in the eigenvalues we get from the spectrum.

# Arguments
- `state_data`: A `StateData` object containing the state data.
- `billiard`: A `Bi` object representing the billiard.
- `basis`: A `Ba` object representing the basis.
- Comment: `c` density of points in coherent state peak, `w` width in units of sigma.

# Returns
- `Hs_return::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps_return::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs_return::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function husimi_functions_from_StateData(state_data::StateData, billiard::Bi, basis::Ba;  b = 5.0, c = 10.0, w = 7.0) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    L = billiard.length
    Hs_return = Vector{Matrix}(undef, length(ks))
    ps_return = Vector{Vector}(undef, length(ks))
    qs_return = Vector{Vector}(undef, length(ks))
    ks, us, s_vals, _ = boundary_function(state_data, billiard, basis; b=b)
    p = Progress(length(ks); desc="Constructing husimi matrices...")
    Threads.@threads for i in eachindex(ks) 
        H, qs, ps = husimi_function(ks[i], us[i], s_vals[i], L; c=c, w=w)
        Hs_return[i] = H
        ps_return[i] = ps
        qs_return[i] = qs
        next!(p)
    end
    return Hs_return, ps_return, qs_return
end

"""
    husimi_functions_from_boundary_functions(ks::Vector, us::Vector{Vector}, s_vals::Vector{Vector}, billiard::Bi; c = 10.0, w = 7.0)

An efficient way to ge the husimi functions from the stored `ks`, `us`, `s_vals` that we can save after doing the version of `compute_spectrum` with the `StateData`.

# Arguments
- `ks::Vector`: A vector of eigenvalues.
- `us::Vector{Vector}`: A vector of vectors representing the boundary functions.
- `s_vals::Vector{Vector}`: A vector of vectors representing the evaluation points in s coordinate.

# Returns
- `Hs_return::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps_return::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs_return::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function husimi_functions_from_boundary_functions(ks, us, s_vals, billiard::Bi; c = 10.0, w = 7.0) where {Bi<:AbsBilliard}
    L = billiard.length
    Hs_return = Vector{Matrix}(undef, length(ks))
    ps_return = Vector{Vector}(undef, length(ks))
    qs_return = Vector{Vector}(undef, length(ks))
    p = Progress(length(ks); desc="Constructing husimi matrices...")
    Threads.@threads for i in eachindex(ks) 
        H, qs, ps = husimi_function(ks[i], us[i], s_vals[i], L; c=c, w=w)
        Hs_return[i] = H
        ps_return[i] = ps
        qs_return[i] = qs
        next!(p)
    end
    return Hs_return, ps_return, qs_return
end

"""
    husimi_functions_from_us_and_boundary_points(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi) where {Bi<:AbsBilliard,T<:Real}

Efficient way to construct the husimi functions (`Vector{Matrix}`) and their grids from the boundary function values along with the vector of `BoundaryPoints` whic containt the .s field which gives the the arclengths.

# Arguments
- `ks::Vector{T}`: A vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: A vector of vectors representing the boundary function values.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: A vector of `BoundaryPoints` objects.

# Returns
- `Hs::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function husimi_functions_from_us_and_boundary_points(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi) where {Bi<:AbsBilliard,T<:Real}
    vec_of_s_vals = [bdPoints.s for bdPoints in vec_bdPoints]
    Hs_list, ps_list, qs_list = husimi_functions_from_boundary_functions(ks, vec_us, vec_of_s_vals, billiard)
    return Hs_list, ps_list, qs_list
end

"""
    save_husimi_functions(Hs::Vector{Matrix}, ps::Vector{Vector}, qs::Vector{Vector}; filename::String="husimi.jld2")

Saves the husimi functions (the matrices and the qs and ps vector that accompany it for projections to classical phase space) to the filename using JLD2.

# Arguments
- `Hs::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
- `filename::String=husimi.jld2`: The name of the file to save the data to (must be .jld2)

# Returns
- `Nothing`
"""
function save_husimi_functions!(Hs::Vector, ps::Vector, qs::Vector; filename::String="husimi.jld2")
    @save filename Hs ps qs
end

"""
    load_husimi_functions(filename::String)

Loads the husimi functions (the matrices and the qs and ps vector that accompany it for projections to classical phase space) from the filename using JLD2.

# Arguments
- `filename::String`: The name of the file to load the data from (must be .jld2)

# Returns
- `Hs::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function load_husimi_functions(filename::String)
    @load filename Hs ps qs
    return Hs, ps, qs
end
