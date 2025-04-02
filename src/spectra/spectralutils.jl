#include("../abstracttypes.jl")
include("../states/eigenstates.jl")
using ProgressMeter


################################################################
############## OVERLAP AND MERGE ALGORITHM  ####################
################################################################
"""
    is_equal(x::T, dx::T, y::T, dy::T) -> Bool where {T<:Real}

Check if two wavenumbers with their respective tensions overlap. The function constructs intervals around each wavenumber based on the given tensions and checks for overlap.

# Arguments
x::T : The first wavenumber.
dx::T : tension associated with the first wavenumber.
y::T : The second wavenumber.
dy::T : The tension associated with the second wavenumber.

# Returns
Bool : `true` if the intervals `[x-dx, x+dx]` and `[y-dy, y+dy]` overlap, `false` otherwise.
"""
function is_equal(x::T, dx::T, y::T, dy::T) :: Bool where {T<:Real}
    # Define the intervals
    x_lower=x-dx
    x_upper=x+dx
    y_lower=y-dy
    y_upper=y+dy
    # Check if the intervals overlap
    return max(x_lower,y_lower) <= min(x_upper,y_upper)
end

"""
    match_wavenumbers(ks_l::Vector{T}, ts_l::Vector{T}, ks_r::Vector{T}, ts_r::Vector{T}) -> Tuple{Vector{T}, Vector{T}, Vector{Bool}} where {T<:Real}

Match wavenumbers and tensions from two sorted lists (`ks_l` and `ks_r`). The function ensures that overlapping wavenumbers (as determined by `is_equal`) are merged, keeping the one with the smaller tension. If no overlap exists, wavenumbers are appended in order of magnitude.

# Arguments
ks_l::Vector{T} : List of wavenumbers from the left list.
ts_l::Vector{T} : List of tensions from the left list.
ks_r::Vector{T} : List of wavenumbers from the right list.
ts_r::Vector{T} : List of tensions from the right list.

# Returns
ks::Vector{T} : List of merged wavenumbers.
ts::Vector{T} : List of merged tensions corresponding to the wavenumbers.
control::Vector{Bool} : A boolean vector indicating whether a merged wavenumber resulted from overlap between `ks_l` and `ks_r`.
"""
function match_wavenumbers(ks_l::Vector, ts_l::Vector, ks_r::Vector, ts_r::Vector)
    #vectors ks_l and_ks_r must be sorted
    i=j=1 #counting index
    control = Vector{Bool}()#control bits
    ks = Vector{eltype(ks_l)}()#final wavenumbers
    ts = Vector{eltype(ts_l)}()#final tensions
    while i <= length(ks_l) && j <= length(ks_r)
        x,dx=ks_l[i],ts_l[i]
        y,dy=ks_r[j],ts_r[j]
        if is_equal(x,dx,y,dy) #check equality with errorbars
            i+=1 
            j+=1
            if dx<dy
                push!(ks,x)
                push!(ts,dx)
                push!(control,true)
            else
                push!(ks,y)
                push!(ts,dy)
                push!(control,true)
            end
        elseif x<y
            i+=1
            push!(ks,x)
            push!(ts,dx)
            push!(control,false)
        else 
            j+=1
            push!(ks,y)
            push!(ts,dy)
            push!(control,false)
        end
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
function match_wavenumbers_with_X(ks_l::Vector, ts_l::Vector, X_l, ks_r::Vector, ts_r::Vector, X_r)
    i=j=1
    ks=Vector{eltype(ks_l)}() # final wavenumbers
    ts=Vector{eltype(ts_l)}() # final tensions
    X_list=Vector{Vector{Float64}}() # final vectors
    control=Bool[]
    while i <= length(ks_l) && j <= length(ks_r)
        x,dx,Xx=ks_l[i],ts_l[i],X_l[i]
        y,dy,Xy=ks_r[j],ts_r[j],X_r[j]
        if is_equal(x,dx,y,dy)
            # Choose which to keep based on tension
            i+=1
            j+=1
            if dx<dy
                push!(ks,x); 
                push!(ts,dx); 
                push!(X_list,Xx); 
                push!(control,true)
            else
                push!(ks,y); 
                push!(ts,dy); 
                push!(X_list,Xy); 
                push!(control,true)
            end
        elseif x<y
            push!(ks,x); 
            push!(ts,dx); 
            push!(X_list,Xx); 
            push!(control,false)
            i+=1
        else
            push!(ks,y); 
            push!(ts,dy); 
            push!(X_list,Xy); 
            push!(control,false)
            j+=1
        end
    end
    return ks,ts,X_list,control
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
function overlap_and_merge!(k_left::Vector, ten_left::Vector, k_right::Vector, ten_right::Vector, control_left::Vector{Bool}, kl::T, kr::T; tol=1e-3) where {T<:Real}
    #check if intervals are empty 
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(control_left,[false for i in 1:length(k_right)])
        return nothing #return short circuits further evaluation
    end
    #if right is empty just skip the mergeing
    if isempty(k_right)
        return nothing
    end
    #find overlaps in interval [k1,k2]
    idx_l=k_left.>(kl-tol) .&& k_left.<(kr+tol)
    idx_r=k_right.>(kl-tol) .&& k_right.<(kr+tol)
    ks_l,ts_l,ks_r,ts_r=k_left[idx_l],ten_left[idx_l],k_right[idx_r],ten_right[idx_r]
    #check if wavnumbers match in overlap interval
    ks,ts,control=match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    deleteat!(k_left,idx_l)
    append!(k_left,ks)
    deleteat!(ten_left,idx_l)
    append!(ten_left,ts)
    deleteat!(control_left,idx_l)
    append!(control_left,control)
    fl=findlast(idx_r)
    idx_last=isnothing(fl) ? 1 : fl + 1
    append!(k_left,k_right[idx_last:end])
    append!(ten_left,ten_right[idx_last:end])
    append!(control_left,[false for i in idx_last:length(k_right)])
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
function overlap_and_merge_state!(k_left::Vector, ten_left::Vector, X_left, k_right::Vector, ten_right::Vector, X_right, control_left::Vector{Bool}, kl::T, kr::T; tol=1e-3) where {T<:Real}
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
    idx_l=k_left.>(kl - tol) .&& k_left.<(kr+tol)
    idx_r=k_right.>(kl - tol) .&& k_right.<(kr+tol)
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

################################################################
####################### LEGACY  ################################
################################################################

function compute_spectrum_LEGACY(solver::Sol,basis::Ba,billiard::Bi,k1::T,k2::T,dk::T;tol=1e-4) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Bi<:AbsBilliard,T<:Real}
    k0=k1
    num_intervals=ceil(Int,(k2-k1)/dk)
    println("Scaling Method...")
    p=Progress(num_intervals,1)
    #initial computation
    k_res,ten_res=solve_spectrum(solver,basis,billiard,k0,dk+tol)
    control=[false for i in 1:length(k_res)]
    while k0<k2
        println("Doing interval: [$(k0), $(k0+dk)]")
        k0+=dk
        k_new,ten_new=solve_spectrum(solver,basis,billiard,k0,dk+tol)
        overlap_and_merge!(k_res,ten_res,k_new,ten_new,control,k0-dk,k0;tol=tol)
        next!(p)
    end
    return k_res,ten_res,control
end

################################################################
################## VERGINI - SARACENO  #########################
################################################################

"""
    function compute_spectrum(solver::Sol,basis::Ba,billiard::Bi,k1::T,k2::T;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental=true) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Bi<:AbsBilliard,T<:Real}

Computes the spectrum over a range of wavenumbers `[k1, k2]` using the given solver, basis, and billiard. Returns the computed eigenvalues and tensions.

# Arguments
- `solver<:AcceleratedSolver`: The solver used to compute the spectrum.
- `basis<:AbsBasis`: The basis set used for computations. Needs to be comtaptible with the geometry of the problem.
- `billiard::Bi`: The billiard domain for the problem.
- `k1::T`, `k2::T`: The starting and ending wavenumbers of the spectrum range.
- `tol`: Additional padding for merging algorithm (default: `1e-4`).
- `N_expect`: Expected number of eigenvalues per interval (default: `1`).
- `dk_threshold`: Maximum allowed interval size for `dk` (default: `0.05`).
- `fundamental`: Whether to use fundamental domain properties (default: `true`).

# Returns
- A tuple `(k_res, ten_res, control)`:
    - `k_res::Vector{T}`: Vector of computed wavenumbers.
    - `ten_res::Vector{T}`: Vector of corresponding tensions.
    - `control::Vector{Bool}`: Vector indicating whether each wavenumber was compared and merged (`true`) with tension comparisons or not (`false`).
"""
function compute_spectrum(solver::Sol,basis::Ba,billiard::Bi,k1::T,k2::T;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental=true) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Bi<:AbsBilliard,T<:Real}
    # Estimate the number of intervals and store the dk values
    println("Starting spectrum computation...")
    k0=k1
    dk_values=[]
    # Fill out the intervals
    while k0<k2
        if fundamental
            dk=N_expect/(billiard.area_fundamental*k0/(2*pi)-billiard.length_fundamental/(4*pi))
        else
            dk=N_expect/(billiard.area*k0/(2*pi)-billiard.length/(4*pi))
        end
        if dk<0.0
            dk= -dk
        end
        if dk>dk_threshold # For small k this limits the size of the interval
            dk=dk_threshold
        end
        push!(dk_values,dk)
        k0+=dk
    end
    println("min/max dk value: ", extrema(dk_values))
    # Initialize the progress bar with estimated number of intervals
    println("Scaling Method...")
    p=Progress(length(dk_values),1)

    k0=k1
    println("Initial solve...")
    @time k_res,ten_res=solve_spectrum(solver,basis,billiard,k0,dk_values[1]+tol)
    control=[false for i in 1:length(k_res)]
    for i in eachindex(dk_values)
        dk=dk_values[i]
        k0+=dk
        println("Doing k: ", k0, " to: ", k0+dk+tol)
        @time k_new,ten_new=solve_spectrum(solver,basis,billiard,k0,dk+tol)
        overlap_and_merge!(k_res,ten_res,k_new,ten_new,control,k0-dk,k0;tol=tol)
        next!(p)
    end
    return k_res,ten_res,control
end

"""
    compute_spectrum(solver::Sol,basis::Ba,billiard::Bi,N1::Int,N2::Int;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental=true) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Bi<:AbsBilliard}

Computes the spectrum for a range of states `[N1, N2]` using the given solver, basis, and billiard. Translates the state numbers to wavenumbers using Weyl's law and then computes the spectrum via the k version of this function.

# Arguments
- `solver<:AcceleratedSolver`: The solver used to compute the spectrum.
- `basis<:AbsBasis`: The basis set used for computations. Needs to be comtaptible with the geometry of the problem.
- `billiard::Bi`: The billiard domain for the problem.
- `N1::Int`, `N2::Int`: The starting and ending state numbers that will be translated to their corresponding eigenvalues via Weyl's law.
- `tol`: Additional padding for merging algorithm (default: `1e-4`).
- `N_expect`: Expected number of eigenvalues per interval (default: `1`).
- `dk_threshold`: Maximum allowed interval size for `dk` (default: `0.05`).
- `fundamental`: Whether to use fundamental domain properties (default: `true`).

# Returns
- A tuple `(k_res, ten_res, control)`:
    - `k_res::Vector{T}`: Vector of computed wavenumbers.
    - `ten_res::Vector{T}`: Vector of corresponding tensions.
    - `control::Vector{Bool}`: Vector indicating whether each wavenumber was compared and merged (`true`) with tension comparisons or not (`false`).
"""
function compute_spectrum(solver::Sol,basis::Ba,billiard::Bi,N1::Int,N2::Int;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental=true) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Bi<:AbsBilliard}
    # get the k1 and k2 from the N1 and N2
    k1=k_at_state(N1,billiard;fundamental=fundamental)
    k2=k_at_state(N2,billiard;fundamental=fundamental)
    println("k1 = $(k1), k2 = $(k2)")
    # Call the main with k
    k_res,ten_res,control=compute_spectrum(solver,basis,billiard,k1,k2;tol=tol,N_expect=N_expect,dk_threshold=dk_threshold,fundamental=fundamental)
    return k_res,ten_res,control
end

"""
    compute_spectrum_with_state(solver, basis, billiard, k1, k2; tol=1e-4, N_expect=3, dk_threshold=0.05, fundamental=true)

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

# Returns
    - `k_res::Vector{T}`: Vector of computed wavenumbers.
    - `ten_res::Vector{T}`: Vector of corresponding tensions.
    - `control::Vector{Bool}`: Vector indicating whether each wavenumber was compared and merged (`true`) with tension comparisons or not (`false`).
"""
function compute_spectrum_with_state(solver::Sol,basis::Ba,billiard::Bi,k1::T,k2::T;tol::T=T(1e-4),N_expect::Int=1,dk_threshold::T=T(0.05),fundamental::Bool=true) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Bi<:AbsBilliard,T<:Real}
    # Estimate the number of intervals and store the dk values
    k0=k1
    dk_values=[]
    A_fund=billiard.area_fundamental;L_fund=billiard.length_fundamental;A_full=billiard.area;L_full=billiard.length
    while k0<k2
        if fundamental
            dk=N_expect/(A_fund*k0/(2*pi)-L_fund/(4*pi)) # weyl estimate
        else
            dk=N_expect/(A_full*k0/(2*pi)-L_full/(4*pi))
        end
        dk<0.0 ? -dk : dk
        dk>dk_threshold ? dk=dk_threshold : nothing # For small k this limits the size of the interval
        push!(dk_values,dk)
        k0+=dk
    end
    println("min/max dk value: ",extrema(dk_values))
    # Initialize the progress bar with estimated number of intervals
    println("Scaling Method w/ StateData...")
    p=Progress(length(dk_values),1)
    println("Total number of eigenvalue problems to solve... ",length(dk_values))
    # Actual computation using precomputed dk values
    k0=k1
    state_res::StateData{T,T}=solve_state_data_bundle_with_INFO(solver,basis,billiard,k0,dk_values[1]+tol)
    control::Vector{Bool}=[false for _ in 1:length(state_res.ks)]
    for i in eachindex(dk_values)[2:end]
        dk=dk_values[i]
        k0+=dk
        state_new::StateData{T,T}=solve_state_data_bundle(solver,basis,billiard,k0,dk+tol)
        # Merge the new state into the accumulated state
        overlap_and_merge_state!(state_res.ks,state_res.tens,state_res.X,state_new.ks,state_new.tens,state_new.X,control,k0-dk,k0;tol=tol)
        next!(p)
    end
    return state_res,control
end

"""
    compute_spectrum_with_state(solver, basis, billiard, k1, k2; tol=1e-4, N_expect=3, dk_threshold=0.05, fundamental=true)

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

# Returns
    - `k_res::Vector{T}`: Vector of computed wavenumbers.
    - `ten_res::Vector{T}`: Vector of corresponding tensions.
    - `control::Vector{Bool}`: Vector indicating whether each wavenumber was compared and merged (`true`) with tension comparisons or not (`false`).
"""
function compute_spectrum_with_state(solver::Sol,basis::Ba,billiard::Bi,N1::Int,N2::Int;tol=1e-4,N_expect::Int=1,dk_threshold=0.05,fundamental::Bool=true) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Bi<:AbsBilliard}
    k1=k_at_state(N1,billiard;fundamental=fundamental)
    k2=k_at_state(N2,billiard;fundamental=fundamental)
    println("k1 = $(k1), k2 = $(k2)")
    compute_spectrum_with_state(solver,basis,billiard,k1,k2,tol=tol,N_expect=N_expect,dk_threshold=dk_threshold,fundamental=fundamental)
end

########################
#### IN PREPARATION ####
########################

"""
    compute_spectrum_optimized(k1::T, k2::T, basis::Ba, billiard::Bi; N_expect::Integer=3, dk_threshold=T(0.05), tol::T=T(1e-4), d0::T=T(1.0), b0::T=T(2.0),partitions::Integer=10, samplers::Vector{Sam}=[GaussLegendreNodes()], fundamental::Bool=true, display_basis_matrices=false) where {T<:Real,Bi<:AbsBilliard,Ba<:AbsBasis,Sam<:AbsSampler}

Compute the eigenvalue spectrum for a given billiard system, optimizing over intervals defined by a dynamical solver.
The function partitions the interval `[k1, k2]` into subintervals and constructs solvers dynamically for each subinterval. A step size (`dk`) for wavenumbers is computed using the Weyl estimate and adjusted based on the `dk_threshold`. For each subinterval, eigenvalues and their corresponding tensions are computed iteratively. Overlapping eigenvalues are merged to ensure continuity, and results are filtered to retain eigenvalues within the target interval. The method supports computation on either the fundamental or full billiard domain.

# Arguments
- `k1::T`: The lower bound of the wavenumber interval (`k`).
- `k2::T`: The upper bound of the wavenumber interval (`k`).
- `basis::Ba`: The basis used for constructing the matrices, which must be of type `AbsBasis`.
- `billiard::Bi`: The billiard system, which must be of type `AbsBilliard`.
- `N_expect::Integer=3`: The number of eigenvalues expected per interval for the Weyl estimate.
- `dk_threshold::T=T(0.05)`: The maximum allowable step size for the wavenumber (`dk`).
- `tol::T=T(1e-4)`: The tolerance used for numerical operations, such as merging eigenvalues.
- `partitions::Integer=10`: The number of partitions for dividing the interval `[k1, k2]`.
- `samplers::Vector{Sam}=[GaussLegendreNodes()]`: A vector of samplers used for quadrature, with elements of type `AbsSampler`.
- `fundamental::Bool=true`: If `true`, use the fundamental domain properties (area and length) for the Weyl estimate.
- `display_basis_matrices::Bool=false`: Whether to display the construced basis with the optimal d.
- `d0::T`: Starting basis dimension scaling factor.
- `b0::T`: Starting basis length scaling factor.

# Returns
- `Tuple{Vector{T}, Vector{T}, Vector{Bool}}`: A tuple containing:
  - A vector of eigenvalues (`ks`).
  - A vector of eigenvalue tensions (`tensions`).
  - A vector of control flags (`controls`).
"""
function compute_spectrum_optimized(k1::T,k2::T,basis::Ba,billiard::Bi;N_expect::Integer=1,dk_threshold=T(0.05),tol::T=T(1e-4),d0::T=T(1.0),b0::T=T(2.0),partitions::Integer=10,samplers::Vector{Sam}=[GaussLegendreNodes()],fundamental::Bool=true,display_basis_matrices=false) where {T<:Real,Bi<:AbsBilliard,Ba<:AbsBasis,Sam<:AbsSampler}
    solvers,intervals=dynamical_solver_construction(k1,k2,basis,billiard;return_benchmarked_matrices=false,display_benchmarked_matrices=display_basis_matrices,partitions=partitions,samplers=samplers,solver_type=:Accelerated,print_params=false,d0=d0,b0=b0)
    for (i,solver) in enumerate(solvers)
        println("Solver $i d: ", solver.dim_scaling_factor)
        println("Solver $i b: ", solver.pts_scaling_factor)
    end
    dk_vals_all=Vector{Vector{T}}(undef,length(intervals)) # contains all the dk values for each interval in that is part of intervals
    ks=Vector{Vector{T}}(undef,length(intervals))
    tensions=Vector{Vector{T}}(undef,length(intervals))
    controls=Vector{Vector{T}}(undef,length(intervals))
    A_fund=billiard.area_fundamental
    L_fund=billiard.length_fundamental
    A_full=billiard.area
    L_full=billiard.length
    for (i,interval) in enumerate(intervals)
        k1,k2=interval # interval::Tuple{T,T}
        k0=k1
        dk_values=Vector{T}() # length not known
        while k0<k2
            if fundamental
                dk=N_expect/(A_fund*k0/(2*pi)-L_fund/(4*pi)) # weyl estimate
            else
                dk=N_expect/(A_full*k0/(2*pi)-L_full/(4*pi))
            end
            dk<0.0 ? -dk : dk
            dk>dk_threshold ? dk=dk_threshold : nothing # For small k this limits the size of the interval
            push!(dk_values,dk)
            k0+=dk
        end
        dk_vals_all[i]=dk_values
        println("min/max dk value for interval [$k1, $k2]: ",extrema(dk_values))
        k0=k1
        @time k_res,ten_res=solve_spectrum(solvers[i],basis,billiard,k0,dk_values[1]+tol)
        control=[false for _ in 1:length(k_res)]
        @showprogress for j in eachindex(dk_values)
            dk=dk_values[j]
            k0+=dk
            @time k_new,ten_new=solve_spectrum(solvers[i],basis,billiard,k0,dk+tol)
            overlap_and_merge!(k_res,ten_res,k_new,ten_new,control,k0-dk,k0;tol=tol)
            next!(p)
        end
        idxs_contain=findall(k->(k1<k&&k<k2),k_res) # trim the edges from tol
        k_res=k_res[idxs_contain]
        ten_res=ten_res[idxs_contain]
        control=control[idxs_contain]
        ks[i]=k_res
        tensions[i]=ten_res
        controls[i]=control
    end
    return vcat(ks...),vcat(tensions...),vcat(controls...)
end

# helper for merging vectors of StateData structs
function merge_state_data(A::Vector{StateData{T,T}}) where {T<:Real}
    merged_ks=vcat([s.ks for s in A]...)
    merged_X=vcat([s.X for s in A]...)
    merged_tens=vcat([s.tens for s in A]...)
    return StateData{T, T}(merged_ks,merged_X,merged_tens)
end

"""
    compute_spectrum_with_state_optimized(k1::T, k2::T, basis::Ba, billiard::Bi; N_expect::Integer=3, dk_threshold=T(0.05), tol::T=T(1e-4), d0::T=T(1.0), b0::T=T(2.0), partitions::Integer=10, samplers::Vector{Sam}=[GaussLegendreNodes()], fundamental::Bool=true, display_basis_matrices=false) where {T<:Real,Bi<:AbsBilliard,Ba<:AbsBasis,Sam<:AbsSampler}

Compute the spectrum of a quantum billiard system with state optimization. The function iteratively solves for eigenvalues (`ks`) and tensions (`tens`) over partitioned intervals of wave numbers, merging the resulting state data (`StateData`) objects from each interval. This version explicitly returns merged state data.

## Details
This function partitions the interval `[k1, k2]` into smaller sub-intervals based on the desired number of partitions. For each sub-interval, it computes the eigenvalues and tensions using an optimized solver and handles overlaps between intervals. The results are merged into a single `StateData` object. The function trims unnecessary eigenvalues from the edges to ensure only non-overlaping results are retained.

## Arguments
- `k1::T`: Lower bound of the wave number interval.
- `k2::T`: Upper bound of the wave number interval.
- `basis::Ba`: Basis used for the computations.
- `billiard::Bi`: Billiard object representing the quantum billiard geometry.

## Keyword Arguments
- `N_expect::Integer=3`: Number of expected eigenvalues per interval.
- `dk_threshold::T=T(0.05)`: Maximum allowed step size for wave numbers.
- `tol::T=T(1e-4)`: Tolerance for overlap detection in eigenvalue merging.
- `partitions::Integer=10`: Number of partitions to divide the interval `[k1, k2]` into.
- `samplers::Vector{Sam}=[GaussLegendreNodes()]`: Vector of samplers used for solving. Each sampler must be a subtype of `AbsSampler`.
- `fundamental::Bool=true`: Whether to use fundamental billiard geometry for calculations. If `true`, uses the fundamental area and length.
- `display_basis_matrices::Bool=false`: Whether to display the construced basis with the optimal d.
- `d0::T`: Starting basis dimension scaling factor.
- `b0::T`: Starting basis length scaling factor.

## Returns
A tuple containing:
1. `merged_state::StateData{T,T}`: Merged `StateData` object containing:
   - `ks::Vector{T}`: Wavenumbers (eigenvalues).
   - `X::Vector{Vector{T}}`: Wavefunction expansion coefficients for the used basis.
   - `tens::Vector{T}`: Corresponding tensions for checking the correctness of the eigenvalues.
2. `controls::Vector{T}`: Control flags to see which of the eigenvalues were compared w/ another wrt. lower tension value. 
"""
function compute_spectrum_with_state_optimized(k1::T, k2::T, basis::Ba, billiard::Bi; N_expect::Integer=3, dk_threshold=T(0.05), tol::T=T(1e-4), d0::T=T(1.0), b0::T=T(2.0), partitions::Integer=10, samplers::Vector{Sam}=[GaussLegendreNodes()], fundamental::Bool=true, display_basis_matrices=false) where {T<:Real,Bi<:AbsBilliard,Ba<:AbsBasis,Sam<:AbsSampler}
    solvers,intervals=dynamical_solver_construction(k1,k2,basis,billiard;return_benchmarked_matrices=false,display_benchmarked_matrices=display_basis_matrices,partitions=partitions,samplers=samplers,solver_type=:Accelerated,print_params=false,d0=d0,b0=b0)
    for (i,solver) in enumerate(solvers)
        println("Solver $i d: ", solver.dim_scaling_factor)
        println("Solver $i b: ", solver.pts_scaling_factor)
    end
    dk_vals_all=Vector{Vector{T}}(undef,length(intervals)) # contains all the dk values for each interval in that is part of intervals
    mul_state_data=Vector{StateData{T,T}}(undef,length(intervals))
    controls=Vector{Vector{T}}(undef,length(intervals))
    A_fund=billiard.area_fundamental
    L_fund=billiard.length_fundamental
    A_full=billiard.area
    L_full=billiard.length
    for (i,interval) in enumerate(intervals)
        k1,k2=interval # interval::Tuple{T,T}
        k0=k1
        dk_values=Vector{T}() # length not known
        while k0<k2
            if fundamental
                dk=N_expect/(A_fund*k0/(2*pi)-L_fund/(4*pi)) # weyl estimate
            else
                dk=N_expect/(A_full*k0/(2*pi)-L_full/(4*pi))
            end
            dk<0.0 ? -dk : dk
            dk>dk_threshold ? dk=dk_threshold : nothing # For small k this limits the size of the interval
            push!(dk_values,dk)
            k0+=dk
        end
        dk_vals_all[i]=dk_values
        println("min/max dk value for interval [$k1, $k2]: ",extrema(dk_values))
        k0=k1
        @time state_res::StateData{T,T}=solve_state_data_bundle(solvers[i],basis,billiard,k0,dk_values[1]+tol)
        control=[false for _ in 1:length(state_res.ks)]
        @showprogress for j in eachindex(dk_values)[2:end]
            dk=dk_values[j]
            k0+=dk
            state_new::StateData{T,T}=solve_state_data_bundle(solvers[i],basis,billiard,k0,dk+tol)
            overlap_and_merge_state!(state_res.ks,state_res.tens,state_res.X,state_new.ks,state_new.tens,state_new.X,control,k0-dk,k0;tol=tol)
        end
        state_ks=state_res.ks
        state_tensions=state_res.tens
        state_X=state_res.X
        idxs_contain=findall(k->(k1<k&&k<k2),state_ks) # trim the edges from tol
        trimmed_state=StateData(state_ks[idxs_contain],state_X[idxs_contain],state_tensions[idxs_contain])
        control=control[idxs_contain]
        mul_state_data[i]=trimmed_state
        controls[i]=control
    end
    return merge_state_data(mul_state_data),vcat(controls...)
end
    
"""
    compute_spectrum_with_state(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; tol=1e-4, N_expect::Int=3, dk_threshold=0.05, fundamental::Bool = true)

Wrapper for the k1 to k2 compute_spectrum_with_state function. This one accepts a starting state number and an ending state number.

# Arguments
- `solver`: The solver used to compute the spectrum.
- `basis`: The basis set used in computations.
- `billiard`: The billiard domain for the problem.
- `N1`, `N2`: The starting and ending state numbers of the spectrum range.
- `tol`: Tolerance for overlaps and merging (default: `1e-4`).
- `N_expect`: Expected number of eigenvalues per generalized eigenvalue decomposition, determines width of acceptable results from such a computation (default: `3`).
- `dk_threshold`: Maximum allowed interval size for `dk` (default: `0.05`).
- `fundamental`: Whether to use fundamental domain (default: `true`).

# Returns
- A tuple `(state_res, control)` where state_res contains all the information about the eigenvalues and tensions of the problem together with the wavefunction expansion coefficients in the given basis and a control vector that determines if these values were merged in an overlap such that minimal tensions were compared
"""
function compute_spectrum_with_state(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; tol=1e-4, N_expect::Int=3, dk_threshold=0.05, fundamental::Bool = true)
    # get the k1 and k2 from the N1 and N2
    k1 = k_at_state(N1, billiard; fundamental=fundamental)
    k2 = k_at_state(N2, billiard; fundamental=fundamental)
    println("k1 = $(k1), k2 = $(k2)")
    # Call the k one
    state_res, control = compute_spectrum_with_state(solver, basis, billiard, k1, k2; tol=tol, N_expect=N_expect, dk_threshold=dk_threshold, fundamental=fundamental)
    return state_res, control
end

################################################################
############### EXPANDED BOUNDARY INTEGEAL METHOD###############
################################################################

"""
    compute_spectrum(solver::ExpandedBoundaryIntegralMethod,billiard::Bi,k1::T,k2::T;dk::Function=(k) -> (0.05 * k^(-1/3))) -> Tuple{Vector{T}, Vector{T}}

Computes the spectrum of the expanded BIM and their corresponding tensions for a given billiard problem within a specified wavenumber range.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration for the expanded boundary integral method.
- `billiard::Bi`: The billiard configuration, a subtype of `AbsBilliard`.
- `k1::T`: Starting wavenumber for the spectrum calculation.
- `k2::T`: Ending wavenumber for the spectrum calculation.
- `dk::Function`: Custom function to calculate the wavenumber step size. Defaults to a scaling law inspired by Veble's paper.
- `tol=1e-4`: Tolerance for the overlap_and_merge function that samples a bit outside the merging interval for better results.
- `use_lapack_raw::Bool=false`: Use the ggev LAPACK function directly without Julia's eigen(A,B) wrapper for it. Might provide speed-up for certain situations (small matrices...)
- `kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}`: Custom kernel functions for the boundary integral method. The default implementation is given by (:default,:first,:second) for the default hemlhholtz kernel and it's first and second derivative.

# Returns
- `Tuple{Vector{T}, Vector{T}}`: 
  - First element is a vector of corrected eigenvalues (`λ`).
  - Second element is a vector of corresponding tensions.
"""
function compute_spectrum(solver::ExpandedBoundaryIntegralMethod,billiard::Bi,k1::T,k2::T;dk::Function=(k) -> (0.05*k^(-1/3)),tol=1e-4,use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {T<:Real,Bi<:AbsBilliard}
    basis=AbstractHankelBasis()
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    ks=T[]
    dks=T[]
    k=k1
    while k<k2
        push!(ks,k)
        k+=dk(k)
        push!(dks,dk(k))
    end
    λs_all=T[] 
    tensions_all=T[]
    control=Bool[]
    println("EBIM...")
    @showprogress for i in eachindex(ks)
        dd=dks[i]
        λs,tensions=solve(solver,basis,evaluate_points(bim_solver,billiard,ks[i]),ks[i],dd;use_lapack_raw=use_lapack_raw,kernel_fun=kernel_fun)
        if !isempty(λs)
            overlap_and_merge!(λs_all,tensions_all,λs,tensions,control,ks[i]-dd,ks[i];tol=tol)
            #append!(λs_all,λs)
            #append!(tensions_all,tensions) 
        end
    end
    if isempty(λs_all) # Handle case of no eigenvalues found
        λs_all=[NaN]
        tensions_all=[NaN]
    end
    return λs_all,tensions_all
end

# IN PREPARATION FOR EBIM
function compute_spectrum_new(solver::ExpandedBoundaryIntegralMethod,billiard::Bi,k1::T,k2::T;dk::Function=(k) -> (0.05*k^(-1/3)),tol=1e-4,use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {T<:Real,Bi<:AbsBilliard}
    basis=AbstractHankelBasis()
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    ks=T[]
    ks_tmp=T[] # temps for storing ebim inv checks
    tens_tmp=T[] # temps for storing ebim inv checks
    dks=T[]
    k=k1
    while k<k2
        push!(ks,k)
        k+=dk(k)
        push!(dks,dk(k))
    end
    λs_all=T[] 
    tensions_all=T[]
    control=Bool[]
    @showprogress desc="EBIM with 1/diff(ks) check..." for i in eachindex(ks)
        dd=dks[i]
        λs_in,tensions_in,λs_out,tensions_out=solve_1st_order(solver,basis,evaluate_points(bim_solver,billiard,ks[i]),ks[i],dd;use_lapack_raw=use_lapack_raw,kernel_fun=(kernel_fun[1],kernel_fun[2]))
        if !isempty(λs_in) # overlap and merge 1st order corrections
            overlap_and_merge!(λs_all,tensions_all,λs_in,tensions_in,control,ks[i]-dd,ks[i];tol=tol)
        end
        idx=findmin(tensions_out)[2]
        if log10(tensions_out[idx])<0.0
            push!(ks_tmp,λs_out[idx])
            push!(tens_tmp,tensions_out[idx])     
        end
    end
    println("length of k_tmp: ",length(ks_tmp))
    _,inv_tens=ebim_inv_diff(ks_tmp)
    idxs=findall(x->x>0.0,inv_tens) # only these are sensible
    inv_tens=inv_tens[idxs]
    ks_tmp=ks_tmp[idxs] 
    k_peaks=find_peaks(ks_tmp,log10.(inv_tens);threshold=[1.2*log10(1.0/dk(k)) for k in ks_tmp]) # check in neighboorhod of these ks if we have a solution (could either be or not) and if not do the solve again
    println("length of k peaks: ",length(k_peaks))
    @showprogress desc="EBIM resolving for peaks of 1/diff" for k in k_peaks
        interval=(k-dk(k)/2,k+dk(k)/2)
        existing_solutions=any(λ->interval[1]<=λ<=interval[2],λs_all)
        if !existing_solutions # if none resolve
            println("Re-solving in interval $interval...")
            λs_in,tensions_in,_,_=solve_1st_order(solver,basis,evaluate_points(bim_solver,billiard,k),k,dk(k);use_lapack_raw=use_lapack_raw,kernel_fun=(kernel_fun[1],kernel_fun[2]))
            println("λs_in length: ", length(λs_in))
            if !isempty(λs_in)  # merge, if any
                overlap_and_merge!(λs_all,tensions_all,λs_in,tensions_in,control,interval[1],interval[2];tol = tol)
            end
        end
    end
    if isempty(λs_all) # Handle case of no eigenvalues found
        λs_all=[NaN]
        tensions_all=[NaN]
    end
    return λs_all,tensions_all
end

################################################################
######################## TEST FUNCTIONS ########################
################################################################

"""
    compute_spectrum_test(solver::Sol,basis::Ba,pts::Pts,k1::T,k2::T,dk::T;tol=1e-4) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Pts<:AbsPoints,T<:Real}

Visualizes the spectrum computation with how the overlap and subsequent merging happens. With Makie.wong() we cycle through the colors when plotting the newly computed points, and these show with a stark contrast their position with respect to the final merged (kept) ones which are plotted as black. This serves as a test to see how well the merging algorithm goes.

# Arguments
- `solver::Sol`: The solver used to compute the spectrum.
- `basis::Ba`: The basis set used in computations.
- `pts::Pts`: The points where the spectrum is computed (so we do not input the billiard geometry). Careful about the points, they have to be compatible with the method/solver
- `k1::T`: The starting k.
- `k2::T`: The ending k.
- `tol=1e-4`: Additional padding on the merging algorithm so that we do not miss anything.

# Returns
- A figure showing the spectrum with the newly computed points and the final merged ones.
- `Tuple{Vector{T},Vector{T},Vector{Bool}} = ks, tens, control`: A tuple of final eigenvalues, their tensions and the control vectors which signifies that they were compared aand kept in the algorithm logic.
"""
function compute_spectrum_test(solver::Sol,basis::Ba,pts::Pts,k1::T,k2::T,dk::T;tol=1e-4) where {Sol<:AcceleratedSolver,Ba<:AbsBasis,Pts<:AbsPoints,T<:Real}
    k0=k1
    #initial computation
    k_res,ten_res=solve(solver,basis,pts,k0,dk+tol)
    control=[false for i in 1:length(k_res)]
    cycle=Makie.wong_colors()[1:6]
    f=Figure(resolution=(1000,1000));
    ax=Axis(f[1,1])
    scatter!(ax,k_res,log10.(ten_res),color=(cycle[1],0.5))
    scatter!(ax,k_res,zeros(length(k_res)),color=(cycle[1],0.5))
    i=1
    while k0<k2
        k0+=dk
        k_new,ten_new=solve(solver,basis,pts,k0,dk+tol)
        scatter!(ax,k_new,log10.(ten_new),color=(cycle[mod1(i+1,6)],0.5))
        scatter!(ax,k_new,zeros(length(k_new)),color=(cycle[mod1(i+1,6)],0.5))
        i+=1
        overlap_and_merge!(k_res,ten_res,k_new,ten_new,control,k0-dk,k0;tol=tol)
    end
    scatter!(ax,k_res,log10.(ten_res),color=(:black,1.0),marker=:x,ms=100)
    scatter!(ax,k_res,zeros(length(k_res)),color=(:black,1.0),marker=:x,ms=100)
    display(f)
    return k_res,ten_res,control
end








# NEEDS MORE WORK
#=
"""
    compute_spectrum_adaptive(solver::Sol, basis::Ba, billiard::Bi, k1::T, k2::T; 
                              IntervalK::T = T(10.0), fundamental::Bool = true, 
                              N_expect::Int = 3, log_file::String = "compute_spectrum_log.txt") 
                              -> (ks_final::Vector{T}, tens_final::Vector{T}, control_final::Vector{Bool}) where {Sol <: AcceleratedSolver, Ba <: AbsBasis, Bi <: AbsBilliard,T <: Real}

Compute the eigenvalue spectrum of a quantum billiard system over a specified interval `[k1, k2]`, 
adaptively adjusting computational parameters to ensure accurate level counting and resolution.

# Description

`compute_spectrum_adaptive` calculates the eigenvalues (`ks`) and associated data (`tensions`, `control flags`) 
for a quantum billiard problem within the interval `[k1, k2]`. It divides the interval into smaller subintervals 
and adaptively adjusts the `dk_threshold` parameter in each subinterval to achieve the expected number of energy levels, 
improving the accuracy and reliability of the spectrum computation.

This function is crucial for spectral analysis in quantum billiard systems, where accurate eigenvalue computation is essential.

# Arguments
- `solver::Sol`: An instance of the accelerated solver.
- `basis::Ba`: The basis set.
- `billiard::Bi`: The billiard.
- `k1::T`: The starting wavenumber.
- `k2::T`: The ending wavenumber.

# Keyword Arguments

- `IntervalK::T = T(10.0)`: The length of each subinterval into which `[k1, k2]` is divided. Default is `10.0`.
- `fundamental::Bool = true`: If `true`, computations are performed on the fundamental domain; if `false`, on the full billiard domain.
- `N_expect::Int = 3`: The expected number of energy levels per single generalized eigenvalue problem, guiding the adaptive adjustment of `dk_threshold`.
- `log_file::String = "compute_spectrum_log.txt"`: The file name for logging computation details.

# Returns

A tuple containing:

- `ks_final::Vector{T}`: A vector of computed eigenvalues within the interval `[k1, k2]`.
- `tens_final::Vector{T}`: A vector of associated tensions.
- `control_final::Vector{Bool}`: A vector of boolean control flags for merging analysis.

# Algorithm Details

1. **Interval Division**:
   - The main interval `[k1, k2]` is partitioned into smaller subintervals of length `IntervalK`.

2. **Adaptive `dk_threshold` Adjustment**:
   - For each subinterval `[k_start, k_end]`, the function `spectrum_inner_call` is invoked.
   - `dk_threshold` controls the size of the interval the results of the generalized eigenvalue problem; it is adaptively adjusted based on the diff level count.

3. **Level Counting and Adjustment Logic**:
   - **Theoretical Level Count (`N_smooth`)**: Estimated using the smooth Weyl law for the billiard system.
   - **Actual vs. Theoretical Difference (`th_num_diff`)**: The difference between the actual number of levels found and the smooth theoretical expectation.
   - **Average Difference (`avg_sum_diff`)**: The average of `th_num_diff` over all levels in the subinterval.
   - **Adjustment Criteria**:
     - If `avg_sum_diff > 1.0` (too many levels), decrease `dk_threshold`.
     - If `avg_sum_diff < -1.0` (missing levels), increase `dk_threshold`.
     - Adjustments are constrained within sensible minimum and maximum `dk_threshold` values computed based on `N_expect`.

4. **Iterative Refinement**:
   - The adjustment process repeats until the average difference is within ±1.0 or the `dk_threshold` reaches its limits.
   - A maximum iteration limit (`max_iterations = 20`) prevents infinite loops.

5. **Result Compilation**:
   - Valid eigenvalues and associated data from each subinterval are collected.
   - The final results (`ks_final`, `tens_final`, `control_final`) are aggregates of these subinterval computations.

# Helper Functions

- **`N_smooth(k)`**: Estimates the cumulative number of energy levels up to wavenumber `k` using the smooth part of Weyl's formula.
- **`th_num_diff(k, ks, k0)`**: Calculates the difference between the numerical and theoretical number of levels below `k`, relative to a starting point `k0`.
- **`avg_sum_diff(ks, k0)`**: Computes the average of `th_num_diff` over a set of computed eigenvalues `ks` in a given subinterval.
- **`dk_smallest_func(k)` & `dk_largest_func(k)`**: Determine the smallest and largest sensible values for `dk_threshold` based on `k` and `N_expect`.
- **`spectrum_inner_call(k_start, k_end, dk_threshold_initial)`**: Performs the adaptive computation within a subinterval, adjusting `dk_threshold` iteratively for each subinterval.

"""
function compute_spectrum_adaptive(solver::Sol, basis::Ba, billiard::Bi, k1::T, k2::T; IntervalK::T = T(10.0), fundamental::Bool = true, N_expect::Int = 3, log_file::String = "compute_spectrum_log.txt") where {Sol <: AcceleratedSolver, Ba <: AbsBasis, Bi <: AbsBilliard,T <: Real}

    # Set up the file logger
    #logfile = open(log_file, "w")
    #file_logger = LoggingExtras.SimpleLogger(logfile, Logging.Info)
    # Use the file logger only
    #global_logger(file_logger)

    # Arrays that will contain returned results
    intervals = T[]
    ks_final = T[]
    tens_final = T[]  # Assuming tensors are of type T
    control_final = Bool[]

    # Helpers for determining the average of fluctuations
    # Just calculates the N for the smooth part from a given k
    N_smooth(k) = weyl_law(k, billiard; fundamental = fundamental)

    # Counts the number of levels in an interval [k0, k] for k ∈ ks ⊂ [k0, k1]
    th_num_diff(k, ks::Vector{T}, k0) = count(_k -> _k < k, ks) - (N_smooth(k) - N_smooth(k0))

    # Averages the level count for all k ∈ ks. The main criterion function
    avg_sum_diff(ks::Vector{T}, k0) = sum(th_num_diff(k, ks, k0) for k in ks) / length(ks)

    # Helpers for the limits of the while loop that modifies the threshold dk
    dk_smallest_func(k; fundamental = true) = begin
        denom = fundamental ?
            (billiard.area_fundamental * k) / (2π) - (billiard.length_fundamental) / (4π) :
            (billiard.area * k) / (2π) - (billiard.length) / (4π)
        0.5 * N_expect / denom
    end

    dk_largest_func(k; fundamental = true) = begin
        denom = fundamental ?
            (billiard.area_fundamental * k) / (2π) - (billiard.length_fundamental) / (4π) :
            (billiard.area * k) / (2π) - (billiard.length) / (4π)
        2.0 * N_expect / denom
    end

    # Helper for inner callback -> Iteratively checks whether we are losing or gaining levels based on the previous result
    function spectrum_inner_call(k_start, k_end, dk_threshold_initial)
        dk_threshold = dk_threshold_initial
        dk_smallest = dk_smallest_func(k_end; fundamental = fundamental)
        dk_largest = dk_largest_func(k_start; fundamental = fundamental)
        iteration = 0
        max_iterations = 20  # Prevent infinite loops

        @info "Processing interval [$(k_start), $(k_end)] with initial dk_threshold=$(dk_threshold_initial)"

        # Initialize variables to store the best computed results
        best_k_res = T[]
        best_tens = T[]
        best_control = Bool[]
        best_diff = Inf  # Initialize best_diff to a large value

        # Temporary storage for current subinterval results
        temp_storage = Dict{Int, Tuple{Vector{T}, Vector{T}, Vector{Bool}, T}}()

        while true
            iteration += 1
            if iteration > max_iterations
                @warn "Maximum iterations reached in interval [$(k_start), $(k_end)]. Selecting best available results."
                break
            end

            # Compute the spectrum in the interval [k_start, k_end]
            k_res_current, tens_current, control_current = compute_spectrum(
                solver,
                basis,
                billiard,
                k_start,
                k_end;
                dk_threshold = dk_threshold,
                fundamental = fundamental,
            )

            # Crop the k_res so that we do not have edge outer levels
            valid_indices = findall(x -> x >= k_start && x <= k_end, k_res_current)
            if isempty(valid_indices)
                @warn "No valid levels found in interval [$(k_start), $(k_end)] with dk_threshold=$(dk_threshold)."
                # Continue to the next iteration
                continue
            end

            # Extract valid results
            k_res_current = k_res_current[valid_indices]
            tens_current = tens_current[valid_indices]
            control_current = control_current[valid_indices]

            # Compute average difference
            diff = avg_sum_diff(k_res_current, k_start)

            # Store current results in temporary storage
            temp_storage[iteration] = (k_res_current, tens_current, control_current, diff)

            @info "Iteration $(iteration): dk_threshold=$(dk_threshold), avg_diff=$(diff), levels_found=$(length(k_res_current))"

            # Check if the current diff is closer to zero than the best so far and update the hashmap
            if abs(diff) < abs(best_diff)
                best_diff = diff
                best_k_res = k_res_current
                best_tens = tens_current
                best_control = control_current
            end

            # Adjust dk_threshold based on average difference
            if diff > 1.0 && dk_threshold > dk_smallest
                dk_threshold_old = dk_threshold
                dk_threshold *= 0.9  # We have too many levels, decrease dk
                @debug "Decreasing dk_threshold from $(dk_threshold_old) to $(dk_threshold) (too many levels)"
            elseif diff < -1.0 && dk_threshold < dk_largest
                dk_threshold_old = dk_threshold
                dk_threshold *= 1.1  # We are missing levels, increase dk
                @debug "Increasing dk_threshold from $(dk_threshold_old) to $(dk_threshold) (missing levels)"
            else
                # We are within ±1 or have reached the smallest/largest sensible dk
                if dk_threshold < dk_smallest
                    @warn "dk_threshold ($(dk_threshold)) is smaller than the smallest allowed dk ($(dk_smallest)) for N_expect = $(N_expect)"
                elseif dk_threshold > dk_largest
                    @warn "dk_threshold ($(dk_threshold)) is larger than the largest allowed dk ($(dk_largest)) for N_expect = $(N_expect)"
                end
                @info "Accepting results for interval [$(k_start), $(k_end)] after $(iteration) iterations."
                break  # Exit the loop
            end
        end
        # After iterations, select the results with avg_diff closest to zero
        @info "Selecting results with avg_diff closest to zero (avg_diff=$(best_diff))"
        # Flush temporary storage for this subinterval
        temp_storage = nothing  # Allow garbage collection

        return best_k_res, best_tens, best_control, dk_threshold  # Return best results
    end
    # End of helper functions
    # Fill the intervals with increments by IntervalK
    intervals = [k1]
    k_run = k1
    while k_run < k2
        k_run += IntervalK
        if k_run >= k2
            intervals = [intervals; k2]  # Only the last one goes here
            break
        else
            intervals = [intervals; k_run]
        end
    end
    total_intervals = length(intervals) - 1
    @info "Total intervals to process: $(total_intervals)"
    for i in 1:total_intervals
        k_start = intervals[i]
        k_end = intervals[i + 1]
        # Some sensible starting dk_threshold
        dk_threshold_initial = 0.05
        # Log the start of processing for this interval
        @info "Starting interval $(i)/$(total_intervals): [$(k_start), $(k_end)]"
        k_res, tens, control, final_dk_threshold = spectrum_inner_call(k_start, k_end, dk_threshold_initial)
        append!(ks_final, k_res)
        append!(tens_final, tens)
        append!(control_final, control)
        @info "Finished interval $(i): levels_found=$(length(k_res)), final_dk_threshold=$(final_dk_threshold)"
    end
    @info "Spectrum computation completed. Total levels found: $(length(ks_final))"
    #close(logfile) # close the logger
    return ks_final, tens_final, control_final
end
=#