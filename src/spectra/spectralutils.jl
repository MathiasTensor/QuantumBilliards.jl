using ProgressMeter

###########################
######### HELPER ##########
###########################

"""
    is_equal(x::T, dx::T, y::T, dy::T) -> Bool where {T<:Real}

Check if two wavenumbers with their respective tensions overlap. The function constructs intervals around each wavenumber based on the given tensions and checks for overlap.

# Arguments
- `x::T` : The first wavenumber.
- `dx::T` : tension associated with the first wavenumber.
- `y::T` : The second wavenumber.
- `dy::T` : The tension associated with the second wavenumber.

# Returns
`Bool` : `true` if the intervals `[x-dx, x+dx]` and `[y-dy, y+dy]` overlap, `false` otherwise.
"""
function is_equal(x::T,dx::T,y::T,dy::T) :: Bool where {T<:Real}
    x_lower=x-dx
    x_upper=x+dx
    y_lower=y-dy
    y_upper=y+dy
    # Check if the intervals overlap
    return max(x_lower,y_lower)<=min(x_upper,y_upper)
end

"""
    match_wavenumbers(ks_l::Vector{T}, ts_l::Vector{T}, ks_r::Vector{T}, ts_r::Vector{T}) -> Tuple{Vector{T}, Vector{T}, Vector{Bool}} where {T<:Real}

Match wavenumbers and tensions from two sorted lists (`ks_l` and `ks_r`). The function ensures that overlapping wavenumbers (as determined by `is_equal`) are merged, keeping the one with the smaller tension. If no overlap exists, wavenumbers are appended in order of magnitude.

# Arguments
- `ks_l::Vector{T}` : List of wavenumbers from the left list.
- `ts_l::Vector{T}` : List of tensions from the left list.
- `ks_r::Vector{T}` : List of wavenumbers from the right list.
- `ts_r::Vector{T}` : List of tensions from the right list.

# Returns
- `ks::Vector{T}` : List of merged wavenumbers.
- `ts::Vector{T}` : List of merged tensions corresponding to the wavenumbers.
control::Vector{Bool} : A boolean vector indicating whether a merged wavenumber resulted from overlap between `ks_l` and `ks_r`.
"""
function match_wavenumbers(ks_l::Vector{T},ts_l::Vector{T},ks_r::Vector{T},ts_r::Vector{T}) where {T<:Real}
    i=1
    j=1
    ks=T[]
    ts=T[]
    control=Bool[]
    while i<=length(ks_l) && j<=length(ks_r)
        x,dx=ks_l[i],ts_l[i]
        y,dy=ks_r[j],ts_r[j]
        if is_equal(x,dx,y,dy)
            if dx<dy
                push!(ks,x); push!(ts,dx)
            else
                push!(ks,y); push!(ts,dy)
            end
            push!(control,true)
            i+=1
            j+=1
        elseif x<y
            push!(ks,x); push!(ts,dx); push!(control,false)
            i+=1
        else
            push!(ks,y); push!(ts,dy); push!(control,false)
            j+=1
        end
    end
    while i<=length(ks_l)
        push!(ks,ks_l[i]); push!(ts,ts_l[i]); push!(control,false)
        i+=1
    end
    while j<=length(ks_r)
        push!(ks,ks_r[j]); push!(ts,ts_r[j]); push!(control,false)
        j+=1
    end
    return ks,ts,control
end

"""
    match_wavenumbers_with_X(ks_l::Vector, ts_l::Vector, X_l::Vector{Vector}, ks_r::Vector, ts_r::Vector, X_r::Vector{Vector}) -> Tuple{Vector, Vector, Vector{Vector}, Vector{Bool}}

Match wavenumbers and tensions from two input lists, taking into account their respective tensions. If there is any overlap (`is_equal` is called) between the `ks_l` and `ks_r` then we choose the one to push into the `ks` those that have the lowest tension (as more accurate). Otherwise we append those that are smaller of the two. In this way we glue together the `ks_l` and `ks_r` to ks such that it has the smallest tensions and smallest `k` of the two closest to one another.

#Arguments
ks_l::Vector{<:Real} : List of wavenumbers from the left
ts_l::Vector{<:Real} : List of wavenumbers from the right
X_l::Vector{Vector{<:Real}} : List of vectors of vectors of the left sample -> the solve_vector sols for each k in ks_l
ks_r::Vector{<:Real} : List of wavenumbers from the right
ts_r::Vector{<:Real} : List of tensions from the right
X_r::Vector{Vector{<:Real}} : List of vectors of vectors of the right sample -> the solve_vector sols for each k in ks_r

#Returns
ks::Vector{<:Real} : List of merged wavenumbers that match 
ts::Vector{<:Real} : List of merged tensions that 
X_list::Vector{Vector{<:Real}} : List of vectors of vectors of the matched/merged ks
control::Vector{Bool} : List of boolean values indicating whether there was an overlap and we had to choose based on tension value to merge
"""
function match_wavenumbers_with_X(ks_l::Vector{T},ts_l::Vector{T},X_l::Vector{Vector{T}},ks_r::Vector{T},ts_r::Vector{T},X_r::Vector{Vector{T}}) where {T<:Real}
    i=1
    j=1
    ks=T[]
    ts=T[]
    Xs=Vector{Vector{T}}()
    control=Bool[]
    while i<=length(ks_l) && j<=length(ks_r)
        x,dx,Xx=ks_l[i],ts_l[i],X_l[i]
        y,dy,Xy=ks_r[j],ts_r[j],X_r[j]
        if is_equal(x,dx,y,dy)
            if dx<dy
                push!(ks,x); push!(ts,dx); push!(Xs,Xx)
            else
                push!(ks,y); push!(ts,dy); push!(Xs,Xy)
            end
            push!(control,true)
            i+=1
            j+=1
        elseif x<y
            push!(ks,x); push!(ts,dx); push!(Xs,Xx); push!(control,false)
            i+=1
        else
            push!(ks,y); push!(ts,dy); push!(Xs,Xy); push!(control,false)
            j+=1
        end
    end
    while i<=length(ks_l)
        push!(ks,ks_l[i]); push!(ts,ts_l[i]); push!(Xs,X_l[i]); push!(control,false)
        i+=1
    end
    while j<=length(ks_r)
        push!(ks,ks_r[j]); push!(ts,ts_r[j]); push!(Xs,X_r[j]); push!(control,false)
        j+=1
    end
    return ks,ts,Xs,control
end

"""
    overlap_and_merge!(k_left::Vector{T}, ten_left::Vector{T}, k_right::Vector{T}, ten_right::Vector{T}, control_left::Vector{Bool}, kl::T, kr::T; tol::T=1e-3) :: Nothing where {T<:Real}

This function merges two sets of wavenumber data (`k_left`, `ten_left`) and (`k_right`, `ten_right`) that may have overlapping wavenumbers in the interval `[kl - tol, kr + tol]`. 
- It ensures that overlapping wavenumbers are merged, preferring the data with the smaller tension when overlaps occur. 
- Non-overlapping wavenumbers from the right interval (`k_right`) are appended to the left interval (`k_left`) in order.

# Arguments
- `k_left::Vector{T}`: Vector of wavenumbers from the left interval.
- `ten_left::Vector{T}`: Vector of tensions corresponding to `k_left`.
- `k_right::Vector{T}`: Vector of wavenumbers from the right interval.
- `ten_right::Vector{T}`: Vector of tensions corresponding to `k_right`.
- `control_left::Vector{Bool}`: Vector indicating whether each wavenumber in `k_left` was merged (`true`) or not (`false`).
- `kl::T`: Left boundary of the overlapping interval.
- `kr::T`: Right boundary of the overlapping interval.
- `tol::T`: Tolerance for determining overlaps (default: `1e-3`).

# Returns
- `Nothing`: The function modifies `k_left`, `ten_left`, and `control_left` in place.
"""
function overlap_and_merge!(k_left::Vector{T},ten_left::Vector{T},k_right::Vector{T},ten_right::Vector{T},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real}
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(control_left,fill(false,length(k_right)))
        return nothing
    end
    isempty(k_right) && return nothing
    idx_l=(k_left.>(kl-tol)) .&& (k_left.<(kr+tol))
    idx_r=(k_right.>(kl-tol)) .&& (k_right.<(kr+tol))
    ks_l=k_left[idx_l]
    ts_l=ten_left[idx_l]
    ks_r=k_right[idx_r]
    ts_r=ten_right[idx_r]
    ks,ts,control=match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    del_l=findall(idx_l)
    deleteat!(k_left,del_l)
    deleteat!(ten_left,del_l)
    deleteat!(control_left,del_l)
    append!(k_left,ks)
    append!(ten_left,ts)
    append!(control_left,control)
    fl=findlast(idx_r)
    idx_last=isnothing(fl) ? 1 : fl+1
    append!(k_left,k_right[idx_last:end])
    append!(ten_left,ten_right[idx_last:end])
    append!(control_left,fill(false,length(k_right[idx_last:end])))
    return nothing
end

"""
    overlap_and_merge_state!(k_left::Vector, ten_left::Vector, X_left::Vector{Vector}, k_right::Vector, ten_right::Vector, X_right::Vector{Vector}, control_left::Vector{Bool}, kl::T, kr::T; tol::Float64=1e-3) :: Nothing where {T<:Real}

This function merges two sets of wavenumber data (`k_left`, `ten_left`, `X_left`) and (`k_right`, `ten_right`, `X_right`) that may have overlapping wavenumbers in the interval `[kl - tol, kr + tol]`. It ensures that each wavenumber and its associated data are only included once in the merged result, preferring the data with the smaller tension when overlaps occur.

# Arguments
- `k_left::Vector`: Vector of wavenumbers from the left interval.
- `ten_left::Vector`: Vector of tensions corresponding to `k_left`.
- `X_left::Vector{Vector}`: Vector of eigenvectors corresponding to `k_left`.
- `k_right::Vector`: Vector of wavenumbers from the right interval.
- `ten_right::Vector`: Vector of tensions corresponding to `k_right`.
- `X_right::Vector{Vector}`: Vector of eigenvectors corresponding to `k_right`.
- `control_left::Vector{Bool}`: Vector indicating whether each wavenumber in `k_left` was merged (`true`) or not (`false`).
- `kl::T`: Left boundary of the overlapping interval.
- `kr::T`: Right boundary of the overlapping interval.
- `tol::Float64`: Tolerance for determining overlaps (default: `1e-3`).

# Returns
- `Nothing`: The function modifies `k_left`, `ten_left`, `X_left`, and `control_left` in place.
"""
function overlap_and_merge_state!(k_left::AbstractVector{T},ten_left::AbstractVector{T},X_left::Vector{Vector{T}},k_right::AbstractVector{T},ten_right::AbstractVector{T},X_right::Vector{Vector{T}},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real}
    # Check if intervals are empty
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(X_left,X_right)
        append!(control_left,[false for _ in 1:length(k_right)])
        return nothing
    end
    if isempty(k_right)
        return nothing
    end
    # Find overlaps in interval [kl - tol, kr + tol]
    idx_l=k_left.>(kl - tol) .& (k_left.<(kr+tol))
    idx_r=k_right.>(kl - tol) .& (k_right.<(kr+tol))
    # Extract overlapping data
    ks_l=k_left[idx_l]
    ts_l=ten_left[idx_l]
    Xs_l=X_left[idx_l]
    ks_r=k_right[idx_r]
    ts_r=ten_right[idx_r]
    Xs_r=X_right[idx_r]
    # Check if wavenumbers match in overlap interval
    ks,ts,Xs,control=match_wavenumbers_with_X(ks_l,ts_l,Xs_l,ks_r,ts_r,Xs_r)
    # For all those that matched we put them in the location where there was ambiguity (overlap) for merging ks_r into ks_l. So we delete all the k_left that were in the overlap interval and merged the matched results into the correct location where we deleted from idx_l
    deleteat!(k_left,findall(idx_l))
    append!(k_left,ks)
    deleteat!(ten_left,findall(idx_l))
    append!(ten_left,ts)
    deleteat!(X_left,findall(idx_l))
    append!(X_left,Xs)
    deleteat!(control_left,findall(idx_l))
    append!(control_left,control)
    # After we are done with in-tolerance ks safely append what is let to the final results. This is the part where
    fl=findlast(idx_r)
    idx_last=isnothing(fl) ? 1 : fl + 1
    append!(k_left,k_right[idx_last:end])
    append!(ten_left,ten_right[idx_last:end])
    append!(X_left,X_right[idx_last:end])
    append!(control_left,[false for _ in idx_last:length(k_right)])
end

"""
    compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T;tol::T=T(1e-4),N_expect=1,dk_threshold::T=T(0.05),fundamental::Bool=true,multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true,cholesky::Bool=true) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}

Computes the spectrum over a range of wavenumbers `[k1, k2]` using the given solver, basis, and billiard, returning the merged `StateData` containing wavenumbers, tensions, and eigenvectors. MAIN ONE -> for both eigenvalues and husimi/wavefunctions since the expansion coefficients of the basis for the k are saved

# Arguments
- `solver`: The solver used to compute the spectrum.
- `basis`: The basis set used in computations.
- `billiard`: The billiard domain for the problem.
- `k1`, `k2`: The starting and ending wavenumbers of the spectrum range.
- `tol`: Tolerance for computations (default: `1e-4`).
- `N_expect`: Expected number of eigenvalues per interval (default: `3`).
- `dk_threshold`: Maximum allowed interval size for `dk` (default: `0.05`).
- `fundamental`: Whether to use fundamental domain properties (default: `true`).
- `multithreaded_matrices::Bool=false`: If the matrix construction should be multithreaded for the basis and gradient matrices. Very dependant on the k grid and the basis choice to determine the optimal choice for what to multithread.
- `multithreaded_ks::Bool=true`: If the k loop is multithreaded. This is usually the best choice since matrix construction for small k is not as costly.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
- `state-res::StateData{T,T}`: A struct containing:
    - `ks::Vector{T}`: Vector of computed wavenumbers.
    - `X::Vector{T}`: Matrix where each column are the coefficients for the basis expansion for the same indexed eigenvalue.
    - `tens::Vector{T}`: Vector of tensions for each eigenvalue
- `control::Vector{Bool}`: Vector signifying if the eigenvalues at that indexed was compared to another and merged.
"""
function compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T;tol::T=T(1e-4),N_expect=1,dk_threshold::T=T(0.05),fundamental::Bool=true,multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true,cholesky::Bool=false) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}
    k_vals=T[]
    dk_vals=T[]
    k0=k1
    A_fund=billiard.area_fundamental
    L_fund=billiard.length_fundamental
    A_full=billiard.area
    L_full=billiard.length
    while k0<k2
        dk=fundamental ? N_expect/(A_fund*k0/(2π)-L_fund/(4π)) : N_expect/(A_full*k0/(2π)-L_full/(4π))
        dk=abs(dk)
        dk=min(dk,dk_threshold)
        push!(k_vals,k0)
        push!(dk_vals,dk)
        k0+=dk
    end
    @info "Scaling Method w/ StateData..."
    println("min/max dk: ",extrema(dk_vals))
    println("Total intervals: ",length(k_vals))
    all_states=Vector{StateData{T,T}}(undef,length(k_vals))
    all_states[end]=solve_state_data_bundle_with_INFO(solver,basis,billiard,k_vals[end],dk_vals[end]+tol;multithreaded=multithreaded_matrices)
    @info "Multithreading loop? $(multithreaded_ks), multithreading matrix construction? $(multithreaded_matrices)"
    p=Progress(length(k_vals),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(k_vals)[1:end-1]
        ki=k_vals[i]
        dki=dk_vals[i]
        all_states[i]=solve_state_data_bundle(solver,basis,billiard,ki,dki+tol;multithreaded=multithreaded_matrices,cholesky=cholesky)
        next!(p)
    end
    println("Merging intervals...")
    state_res=all_states[1]
    control=[false for _ in 1:length(state_res.ks)]
    p=Progress(length(all_states)-1,1)
    for i in 2:length(all_states)
        overlap_and_merge_state!(
            state_res.ks,state_res.tens,state_res.X,
            all_states[i].ks,all_states[i].tens,all_states[i].X,
            control,k_vals[i-1],k_vals[i];tol=tol)
        next!(p)
    end
    return state_res,control
end

"""
    compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T,dk::T;tol::T=T(1e-4),multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}

Compute the spectrum of a billiard system over a fixed-resolution wavenumber grid `[k1, k2]` using a specified step size `dk`, and return eigenstates (wavenumbers, tensions, basis coefficients) in a merged `StateData` structure.

This is the fixed-interval version of the Vergini–Saraceno-style solver that **also captures the eigenvectors / basis expansion coefficients** for each wavenumber.
After solving, results are merged using `overlap_and_merge_state!`, keeping lower-tension solutions when overlaps occur.

# Arguments
- `solver::Sol`: Spectral solver (e.g., scaling method), subtype of `AcceleratedSolver`.
- `basis::Ba`: Basis object compatible with the billiard geometry, subtype of `AbsBasis`.
- `billiard::Bi`: The geometry (domain), subtype of `AbsBilliard`.
- `k1::T`, `k2::T`: Start and end values of the wavenumber range.
- `dk::T`: Fixed step size between wavenumber intervals.
- `tol::T=1e-4`: Tolerance for merging results in overlapping intervals.
- `multithreaded_matrices::Bool=false`: Enable multithreading for matrix assembly.
- `multithreaded_ks::Bool=true`: Enable multithreading over `k` intervals.
- `cholesky::Bool=false`: Use Cholesky decomposition for solving linear systems.

# Returns
- `state_res::StateData{T,T}`: Struct holding merged eigenvalues, tensions, and eigenvectors.
  - `state_res.ks`: Final merged wavenumbers.
  - `state_res.tens`: Tension values associated with each wavenumber.
  - `state_res.X`: Basis coefficient vectors for each eigenvalue.
- `control::Vector{Bool}`: Vector indicating if an eigenvalue was involved in overlap resolution (`true`) or added uniquely (`false`).
"""
function compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T,dk::T;tol::T=T(1e-4),multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true,cholesky::Bool=false) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}
    k_vals=collect(range(k1,k2,step=dk))
    @info "Scaling Method w/ StateData..."
    println("Total intervals: ",length(k_vals))
    all_states=Vector{StateData{T,T}}(undef,length(k_vals))
    all_states[end]=solve_state_data_bundle_with_INFO(solver,basis,billiard,k_vals[end],dk+tol;multithreaded=multithreaded_matrices)
    @info "Multithreading loop? $(multithreaded_ks), multithreading matrix construction? $(multithreaded_matrices)"
    p=Progress(length(k_vals),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(k_vals)[1:end-1]
        ki=k_vals[i]
        all_states[i]=solve_state_data_bundle(solver,basis,billiard,ki,dk+tol;multithreaded=multithreaded_matrices,cholesky=cholesky)
        next!(p)
    end
    println("Merging intervals...")
    state_res=all_states[1]
    control=[false for _ in 1:length(state_res.ks)]
    p=Progress(length(all_states)-1,1)
    for i in 2:length(all_states)
        overlap_and_merge_state!(
            state_res.ks,state_res.tens,state_res.X,
            all_states[i].ks,all_states[i].tens,all_states[i].X,
            control,k_vals[i-1],k_vals[i];tol=tol)
        next!(p)
    end
    return state_res,control
end

"""
    compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,N1::Int,N2::Int;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental::Bool=true,multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbsBasis,Bi<:AbsBilliard}

Computes the spectrum over a range of wavenumbers defined by the bracketing interval of their state number `[N1, N2]` using the given solver, basis, and billiard, returning the merged `StateData` containing wavenumbers, tensions, and eigenvectors. MAIN ONE -> for both eigenvalues and husimi/wavefunctions since the expansion coefficients of the basis for the k are saved. This one is just a wrapper function for the k version of this function.

# Arguments
- `solver`: The solver used to compute the spectrum.
- `basis`: The basis set used in computations.
- `billiard`: The billiard domain for the problem.
- `N1::Int`, `N2::Int`: The starting and ending state numbers that will be translated to their corresponding eigenvalues via Weyl's law.
- `tol`: Tolerance for computations (default: `1e-4`).
- `N_expect`: Expected number of eigenvalues per interval (default: `3`).
- `dk_threshold`: Maximum allowed interval size for `dk` (default: `0.05`).
- `fundamental`: Whether to use fundamental domain properties (default: `true`).
- `multithreaded_matrices::Bool=false`: If the matrix construction should be multithreaded for the basis and gradient matrices. Very dependant on the k grid and the basis choice to determine the optimal choice for what to multithread.
- `multithreaded_ks::Bool=true`: If the k loop is multithreaded. This is usually the best choice since matrix construction for small k is not as costly.
- `cholesky::Bool=false`: Use Cholesky decomposition for solving linear systems (default: `false`).

# Returns
- `state-res::StateData{T,T}`: A struct containing:
    - `ks::Vector{T}`: Vector of computed wavenumbers.
    - `X::Vector{T}`: Matrix where each column are the coefficients for the basis expansion for the same indexed eigenvalue.
    - `tens::Vector{T}`: Vector of tensions for each eigenvalue
- `control::Vector{Bool}`: Vector signifying if the eigenvalues at that indexed was compared to another and merged.
"""
function compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,N1::Int,N2::Int;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental::Bool=true,multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true,cholesky::Bool=false) where {Ba<:AbsBasis,Bi<:AbsBilliard}
    k1=k_at_state(N1,billiard;fundamental=fundamental)
    k2=k_at_state(N2,billiard;fundamental=fundamental)
    println("k1 = $(k1), k2 = $(k2)")
    return compute_spectrum_with_state_scaling_method(solver,basis,billiard,k1,k2,tol=tol,N_expect=N_expect,dk_threshold=dk_threshold,fundamental=fundamental,multithreaded_matrices=multithreaded_matrices,multithreaded_ks=multithreaded_ks,cholesky=cholesky)
end


