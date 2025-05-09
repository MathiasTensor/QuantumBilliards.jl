using CircularArrays
using JLD2
using LinearAlgebra

"""
    antisym_vec(x::AbstractVector)

Creates an antisymmetric vector by reversing the input vector `x`, negating the reversed values, and appending them to `x`. Used in the original husimi_function construction.

# Arguments
- `x::Vector`: Starting half of the final vector.

# Returns
- `Vector`: An antisymmetric vector constructed from `x`.
"""
function antisym_vec(x)
    v=reverse(-x[2:end])
    return append!(v,x)
end

"""
    husimi_function(k, u, s, L; c = 10.0, w = 7.0)

Calculates the Husimi function on a grid defined by the boundary `s`. The logic is that from the arclengths `s` we construct the qs with the help of the width of the Gaussian at that k (width = 1/√k) and the parameter c (which determines how many q evaluations we will have per this width at k -> defines the step 1/(√k*c) so we will have tham many more q evaluations in peak). Analogously we do for the ps, where the step size (matrix size in ps direction) is range(0.0,1.0,step=1/(√k*c)) (we do only for half since symmetric with p=0 implies we can use antisym_vec(ps) to recreate the whole -1.0 to 1.0 range while also symmetrizing the Husimi function that is constructed from p=0.0 to 1.0 with the logic perscribed above). The w (how many sigmas we will take) is used in construction of the ds summation weights.
Comment: Due to the Poincare map being an involution, i.e. (ξ,p) →(ξ,−p) we construct only the p=0 to p=1 matrix and then via symmetry automatically obtain the p=-1 to p=0 also.
Comment: Original algorithm by Črt Lozej

# Arguments
- `k<:Real`: Wavenumber of the eigenstate.
- `u::Vector{<:Real}`: Array of boundary function values.
- `s::Vector{<:Real}`: Array of boundary points.
- `L<:Real`: Total length of the billiard boundary.
- `c<:Real`: Density of points in the coherent state peak (default: `10.0`).
- `w<:Real`: Width in units of `σ` (default: `7.0`).

# Returns
- `H::Matrix`: Husimi function matrix.
- `qs::Vector`: Array of position coordinates on the grid.
- `ps::Vector`: Array of momentum coordinates on the grid.
"""
function husimi_function(k,u,s,L; c = 10.0, w = 7.0)
    #c density of points in coherent state peak, w width in units of sigma
    #L is the boundary length for periodization
    #compute coherrent state weights
    N=length(s)
    sig=one(k)/sqrt(k) # width of the Gaussian
    x=s[s.<=w*sig]
    idx=length(x) # do not change order here
    x=antisym_vec(x)
    a=one(k)/(2*pi*sqrt(pi*k)) # normalization factor (Husimi not normalized to 1)
    ds=(x[end]-x[1])/length(x) # integration weight
    uc=CircularVector(u) # allows circular indexing
    gauss=@. exp(-k/2*x^2)*ds
    gauss_l=@. exp(-k/2*(x+L)^2)*ds
    gauss_r=@. exp(-k/2*(x-L)^2)*ds
    ps=collect(range(0.0,1.0,step=sig/c)) # evaluation points in p coordinate
    q_stride=length(s[s.<=sig/c])
    q_idx=collect(1:q_stride:N)
    push!(q_idx,N) # add last point
    qs=s[q_idx] # evaluation points in q coordinate
    H=zeros(typeof(k),length(qs),length(ps))
    for i in eachindex(ps)
        cs=@. exp(im*ps[i]*k*x)*gauss + exp(im*ps[i]*k*(x+L))*gauss_l + exp(im*ps[i]*k*(x-L))*gauss_r
        for j in eachindex(q_idx)
            u_w=uc[q_idx[j]-idx+1:q_idx[j]+idx-1] # window with relevant values of u
            h=sum(cs.*u_w)
            H[j,i]=a*abs2(h)
        end
    end
    ps=antisym_vec(ps)
    H_ref=reverse(H[:,2:end];dims=2)
    H=hcat(H_ref,H)
    return H,qs,ps
end

"""
    husimi_function(state::S; b = 5.0, c = 10.0, w = 7.0) where {S<:AbsState}

Calculates the Husimi function for a billiard eigenstate. Wrapper for lower level husimi_function.

# Arguments
- `state<:AbsState`: An eigenstate of the billiard system (contains the basis, billiard and k importantly)
- `b`: Parameter for the boundary function computation (default: `5.0`).
- `c`: Density of points in the coherent state peak (default: `10.0`).
- `w`: Width in units of `σ` (default: `7.0`).

# Returns
- `H::Matrix`: Husimi function matrix.
- `qs::Vector`: Array of position coordinates on the grid.
- `ps::Vector`: Array of momentum coordinates on the grid.
"""
function husimi_function(state::S;b=5.0,c=10.0,w=7.0) where {S<:AbsState}
    L=state.billiard.length
    k=state.k
    u,s,norm=boundary_function(state;b=b)
    return husimi_function(k,u,s,L;c=c,w=w)
end

"""
    husimi_function(state_bundle::S; b = 5.0, c = 10.0, w = 7.0) where {S<:EigenstateBundle}

Calculates the Husimi function for a batch of billiard eigenstates that came from a single Scaling Method evaluation, therefore the expansion coefficients in the basis have same length. Wrapper for lower level husimi_function.

# Arguments
- `state<:EigenstateBundle`: An eigenstate of the billiard system (contains the basis, billiard and k importantly)
- `b`: Parameter for the boundary function computation (default: `5.0`).
- `c`: Density of points in the coherent state peak (default: `10.0`).
- `w`: Width in units of `σ` (default: `7.0`).

# Returns
- `H::Matrix`: Husimi function matrix.
- `qs::Vector`: Array of position coordinates on the grid.
- `ps::Vector`: Array of momentum coordinates on the grid.
"""
function husimi_function(state_bundle::S;b=5.0,c=10.0,w=7.0) where {S<:EigenstateBundle}
    L=state_bundle.billiard.length
    ks=state_bundle.ks
    us,s,norm=boundary_function(state_bundle; b=b)
    H,qs,ps=husimi_function(ks[1],us[1],s,L;c=c,w=w)
    type=eltype(H)
    valid_indices=fill(true,length(ks))
    Hs=Vector{Matrix{type}}(undef,length(ks))
    Hs[1]=H
    for i in eachindex(ks)[2:end] # no need for multithreading here
        try
            H,qs,ps=husimi_function(ks[i],us[i],s,L;c=c,w=w)
            Hs[i]=H
        catch e
            println("Error while constructing Husimi for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
    end
    Hs=Hs[valid_indices]
    return Hs,qs,ps
end

### NEW ###

"""
    husimiAtPoint_LEGACY(k::T,s::Vector{T},u::Vector{T},L::T,q::T,p::T) where {T<:Real}

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
function husimiAtPoint_LEGACY(k::T,s::Vector{T},u::Vector{T},L::T,q::T,p::T) where {T<:Real}
    # original algorithm by Benjamin Batistić in python (https://github.com/clozej/quantum_billiards/blob/crt_public/src/CoreModules/HusimiFunctionsOld.py)
    ss=s.-q
    width=4/sqrt(k)
    indx=findall(x->abs(x)<width,ss)
    si=ss[indx]
    ui=u[indx]
    ds=diff(s)  # length N-1
    ds=vcat(ds,L+s[1]-s[end]) # add Nth
    dsi=ds[indx]
    w=sqrt(sqrt(k/π)).*exp.(-0.5*k*si.*si).*dsi
    cr=w.*cos.(k*p*si)  # Coherent state real part
    ci=w.*sin.(k*p*si)  # Coherent state imaginary part
    h=dot(cr-im*ci,ui)  # Husimi integral (minus because of conjugation)
    return abs2(h)/(2*π*k) # not the actual normalization
end

"""
    husimiOnGrid_LEGACY(k::T,s::Vector{T},u::Vector{T},L::T,nx::Integer,ny::Integer) where {T<:Real}

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
function husimiOnGrid_LEGACY(k::T,s::Vector{T},u::Vector{T},L::T,nx::Integer,ny::Integer) where {T<:Real}
    qs=range(0.0,stop=L,length=nx)
    ps=range(-1.0,stop=1.0,length=ny)
    H=zeros(T,nx,ny)
    Threads.@threads for idx_p in eachindex(ps)
        for idx_q in eachindex(qs)
            H[idx_q,idx_p]=husimiAtPoint_LEGACY(k,s,u,L,qs[idx_q],ps[idx_p])
        end
    end
    return H./sum(H),qs,ps
end

"""
    husimiOnGrid(k::T, s::Vector{T}, u::Vector{T}, L::T, nx::Integer, ny::Integer) where {T<:Real}

Evaluates the Poincaré-Husimi function on a grid using vectorized operations in a thread-safe manner since only a single thread work on a column of the Husimi matrix.
Due to the Poincare map being an involution, i.e. (ξ,p) →(ξ,−p) we construct only the p=0 to p=1 matrix and then via symmetry automatically obtain the p=-1 to p=0 also.

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
    qs=range(0.0,stop=L,length=nx)
    ps_pos=range(0.0,stop=1.0,length=ceil(Int,ny/2))
    ps_full=vcat(-reverse(ps_pos[2:end]),ps_pos)
    ds=vcat(diff(s),L+s[1]-s[end])
    nf=sqrt(sqrt(k/pi))
    width=4/sqrt(k)
    two_pi_k=2*pi*k
    Hp=zeros(T,length(ps_pos),nx)
    Threads.@threads for iq in 1:nx
        q=qs[iq]
        lo=searchsortedfirst(s,q-width)
        hi=searchsortedlast(s,q+width)
        @views s_win=s[lo:hi]
        @views u_win=u[lo:hi]
        @views ds_win=ds[lo:hi]
        si_win=@. s_win-q
        @fastmath @inbounds begin
            w=@. nf*exp(-0.5*k*si_win^2)*ds_win
            for (ip,p) in enumerate(ps_pos)
                kp=k*p
                sinbuf,cosbuf=sincos(kp.*si_win) # matrix [length(si_win),1]
                hr=sum(w.*cosbuf.*u_win)
                hi=-sum(w.*sinbuf.*u_win)
                Hp[ip,iq]=(hr^2+hi^2)/(two_pi_k)
            end
        end
    end
    H=vcat(reverse(Hp;dims=1),Hp[2:end,:])'
    return H./sum(H),qs,ps_full
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
function husimi_functions_from_StateData(state_data::StateData,billiard::Bi,basis::Ba;b=5.0,c=10.0,w=7.0) where {Bi<:AbsBilliard,Ba<:AbsBasis}
    L=billiard.length
    ks=state_data.ks
    valid_indices=fill(true,length(ks))
    Hs_return=Vector{Matrix}(undef,length(ks))
    ps_return=Vector{Vector}(undef,length(ks))
    qs_return=Vector{Vector}(undef,length(ks))
    ks,us,s_vals,_ = boundary_function(state_data,billiard,basis;b=b)
    p=Progress(length(ks);desc="Constructing husimi matrices, N=$(length(ks))")
    Threads.@threads for i in eachindex(ks) 
        try
            H,qs,ps=husimi_function(ks[i],us[i],s_vals[i],L;c=c,w=w)
            Hs_return[i]=H
            ps_return[i]=ps
            qs_return[i]=qs
        catch e
            println("Error while constructing Husimi for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(p)
    end
    Hs_return=Hs_return[valid_indices]
    ps_return=ps_return[valid_indices]
    qs_return=qs_return[valid_indices]
    return Hs_return,ps_return,qs_return
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
    L=billiard.length
    valid_indices=fill(true,length(ks))
    Hs_return=Vector{Matrix}(undef,length(ks))
    ps_return=Vector{Vector}(undef,length(ks))
    qs_return=Vector{Vector}(undef,length(ks))
    p=Progress(length(ks);desc="Constructing husimi matrices, N=$(length(ks))")
    Threads.@threads for i in eachindex(ks) 
        try
            H,qs,ps=husimi_function(ks[i],us[i],s_vals[i],L;c=c,w=w)
            Hs_return[i]=H
            ps_return[i]=ps
            qs_return[i]=qs
        catch e
            println("Error while constructing Husimi for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(p)
    end
    Hs_return=Hs_return[valid_indices]
    ps_return=ps_return[valid_indices]
    qs_return=qs_return[valid_indices]
    return Hs_return,ps_return,qs_return
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
    vec_of_s_vals=[bdPoints.s for bdPoints in vec_bdPoints]
    Hs_list,ps_list,qs_list=husimi_functions_from_boundary_functions(ks,vec_us,vec_of_s_vals,billiard)
    return Hs_list,ps_list,qs_list
end

"""
    husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi, nx::Integer, ny::Integer) where {Bi<:AbsBilliard,T<:Real}

Efficient way to construct the husimi functions (`Vector{Matrix}`) on a common grid of `size(nx,ny)` from the boundary function values along with the vector of `BoundaryPoints` whic containt the .s field which gives the the arclengths.

# Arguments
- `ks::Vector{T}`: A vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: A vector of vectors representing the boundary function values.
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: A vector of `BoundaryPoints` objects.
- `billiard::Bi`: The billiard geometry for the total length. Faster than calling `maximum(s)`.
- `nx::Interger`: The size of linearly spaced q grid.
- `ny::Interger`: The size of linearly spaced p grid.

# Returns
- `Hs_list::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{T}`: A vector representing the evaluation points in p coordinate (same for all husimi matrices).
- `qs::Vector{T}`: A vector representing the evaluation points in q coordinate (same for all husimi matrices).
"""
function husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_bdPoints::Vector{BoundaryPoints{T}}, billiard::Bi, nx::Integer, ny::Integer) where {Bi<:AbsBilliard,T<:Real}
    L=billiard.length
    valid_indices=fill(true,length(ks))
    vec_of_s_vals=[bdPoints.s for bdPoints in vec_bdPoints]
    Hs_list=Vector{Matrix{T}}(undef,length(ks))
    H,qs,ps=husimiOnGrid(ks[1],vec_of_s_vals[1],vec_us[1],L,nx,ny)
    Hs_list[1]=H
    p=Progress(length(ks);desc="Constructing husimi matrices, N=$(length(ks))")
    Threads.@threads for i in eachindex(ks)[2:end]
        try
            H,_,_=husimiOnGrid(ks[i],vec_of_s_vals[i],vec_us[i],L,nx,ny)
            Hs_list[i]=H
        catch e
            println("Error while constructing Husimi for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(p)
    end
    Hs_list=Hs_list[valid_indices]
    ps=collect(ps)
    qs=collect(qs)
    return Hs_list,ps,qs
end

"""
    husimi_functions_from_us_and_arclengths_FIXED_GRID(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_of_s_vals::Vector{Vector{T}}, billiard::Bi, nx::Integer, ny::Integer) where {Bi<:AbsBilliard,T<:Real}

Efficient way to construct the husimi functions (`Vector{Matrix}`) on a common grid of `size(nx,ny)` from the arclength values along with the the boundary functions that corresponds to them.

# Arguments
- `ks::Vector{T}`: A vector of eigenvalues.
- `vec_us::Vector{Vector{T}}`: A vector of vectors representing the boundary function values.
- `vec_of_s_vals::Vector{Vector{T}}`: A vector of arclength values that correspond to the boundary function values.
- `billiard::Bi`: The billiard geometry for the total length.
- `nx::Interger`: The size of linearly spaced q grid.
- `ny::Interger`: The size of linearly spaced p grid.

# Returns
- `Hs_list::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{T}`: A vector representing the evaluation points in p coordinate (same for all husimi matrices).
- `qs::Vector{T}`: A vector representing the evaluation points in q coordinate (same for all husimi matrices).
"""
function husimi_functions_from_us_and_arclengths_FIXED_GRID(ks::Vector{T}, vec_us::Vector{Vector{T}}, vec_of_s_vals::Vector{Vector{T}}, billiard::Bi, nx::Integer, ny::Integer) where {Bi<:AbsBilliard,T<:Real}
    L=billiard.length
    valid_indices=fill(true,length(ks))
    Hs_list=Vector{Matrix{T}}(undef,length(ks))
    H,qs,ps=husimiOnGrid(ks[1],vec_of_s_vals[1],vec_us[1],L,nx,ny)
    Hs_list[1]=H
    p=Progress(length(ks);desc="Constructing husimi matrices, N=$(length(ks))")
    Threads.@threads for i in eachindex(ks)[2:end]
        try
            H,_,_=husimiOnGrid(ks[i],vec_of_s_vals[i],vec_us[i],L,nx,ny)
            Hs_list[i]=H
        catch e
            println("Error while constructing Husimi for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(p)
    end
    Hs_list=Hs_list[valid_indices]
    ps=collect(ps)
    qs=collect(qs)
    return Hs_list,ps,qs
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
    return Hs,ps,qs
end

"""
    save_ks_and_husimi_functions!(filename::String,ks::Vector{T},Hs::Vector{Matrix{T}},ps::Vector{Vector{T}},qs::Vector{Vector{T}}) where {T<:Real}

Saves the ks with their corresponding Husimi functions.
# Arguments
- `filename::String`: The name of the file to save the data to (must be .jld2)
- `ks::Vector{T}`: A vector of eigenvalues.
- `Hs::Vector{Matrix{T}}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{Vector{T}}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs::Vector{Vector{T}}`: A vector of vectors representing the evaluation points in q coordinate.

# Returns
- `Nothing`
"""
function save_ks_and_husimi_functions!(filename::String,ks::Vector{T},Hs::Vector,ps::Vector,qs::Vector) where {T<:Real}
    @time @save filename ks Hs ps qs
end

"""
    read_ks_and_husimi_functions(filename::String)

Loads the ks with their corresponding Husimi functions with their grids.

# Arguments
- `filename::String`: The name of the file to load the data from (must be .jld2)

# Returns
- `ks::Vector{<:Real}`: A vector of eigenvalues.
- `Hs::Vector{Matrix{<:Real}}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{Vector{<:Real}}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs::Vector{Vector{<:Real}}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function read_ks_and_husimi_functions(filename::String)
    @time @load filename ks Hs ps qs
    return ks,Hs,ps,qs
end


