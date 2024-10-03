#include("../abstracttypes.jl")

using ProgressMeter

function is_equal(x::T, dx::T, y::T, dy::T) :: Bool where {T<:Real}
    # Define the intervals
    x_lower = x - dx
    x_upper = x + dx
    y_lower = y - dy
    y_upper = y + dy
    # Check if the intervals overlap
    return max(x_lower, y_lower) <= min(x_upper, y_upper)
end

function match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    #vectors ks_l and_ks_r must be sorted
    i = j = 1 #counting index
    control = Vector{Bool}()#control bits
    ks = Vector{eltype(ks_l)}()#final wavenumbers
    ts = Vector{eltype(ts_l)}()#final tensions
    while i <= length(ks_l) && j <= length(ks_r)
        x, dx = ks_l[i], ts_l[i]
        y, dy = ks_r[j], ts_r[j]
        if  is_equal(x,dx,y,dy) #check equality with errorbars
            i += 1 
            j += 1
            if dx < dy
                push!(ks, x)
                push!(ts, dx)
                push!(control, true)
            else
                push!(ks, y)
                push!(ts, dy)
                push!(control, true)
            end
        elseif x < y
            i += 1
            push!(ks, x)
            push!(ts, dx)
            push!(control, false)
        else 
            j += 1
            push!(ks, y)
            push!(ts, dy)
            push!(control, false)
        end
    end
    return ks, ts, control 
end

function overlap_and_merge!(k_left, ten_left, k_right, ten_right, control_left, kl, kr; tol=1e-3)
    #check if intervals are empty 
    if isempty(k_left)
        #println("Left interval is empty.")
        append!(k_left, k_right)
        append!(ten_left, ten_right)
        append!(control_left, [false for i in 1:length(k_right)])
        return nothing #return short circuits further evaluation
    end

    #if right is empty just skip the mergeing
    if isempty(k_right)
        #println("Right interval is empty.")
        return nothing
    end
    
    #find overlaps in interval [k1,k2]
    idx_l = k_left .> (kl-tol) .&& k_left .< (kr+tol)
    idx_r = k_right .> (kl-tol) .&& k_right .< (kr+tol)
    
    ks_l,ts_l,ks_r,ts_r = k_left[idx_l], ten_left[idx_l], k_right[idx_r], ten_right[idx_r]
    #check if wavnumbers match in overlap interval
    ks, ts, control = match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    #println("left: $ks_l")
    #println("right: $ks_r")
    #println("overlaping: $ks")
    #i_l = idx_l[1]
    #i_r = idx_r[end]+1
    deleteat!(k_left, idx_l)
    append!(k_left, ks)
    deleteat!(ten_left, idx_l)
    append!(ten_left, ts)
    deleteat!(control_left, idx_l)
    append!(control_left, control)

    fl = findlast(idx_r)
    idx_last = isnothing(fl) ? 1 : fl + 1
    append!(k_left, k_right[idx_last:end])
    append!(ten_left, ten_right[idx_last:end])
    append!(control_left, [false for i in idx_last:length(k_right)])
end

# OLD
function compute_spectrum(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard,k1,k2,dk; tol=1e-4)
    k0 = k1
    num_intervals = ceil(Int, (k2 - k1) / dk)
    println("Scaling Method...")
    p = Progress(num_intervals, 1)
    #initial computation
    k_res, ten_res = solve_spectrum(solver, basis, billiard, k0, dk+tol)
    control = [false for i in 1:length(k_res)]
    while k0 < k2
        println("Doing interval: [$(k0), $(k0+dk)]")
        k0 += dk
        k_new, ten_new = solve_spectrum(solver, basis, billiard, k0, dk+tol)
        overlap_and_merge!(k_res, ten_res, k_new, ten_new, control, k0-dk, k0; tol=tol)
        next!(p)
    end
    return k_res, ten_res, control
end


# NEW WITH ADAPTIVE tol
function compute_spectrum(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard,k1,k2; tol=1e-4, N_expect = 3, dk_threshold=0.05, fundamental=true)
    # Estimate the number of intervals and store the dk values
    k0 = k1
    dk_values = []
    while k0 < k2
        if fundamental
            dk = N_expect / (billiard.area_fundamental * k0 / (2*pi) - billiard.length_fundamental/(4*pi))
        else
            dk = N_expect / (billiard.area * k0 / (2*pi) - billiard.length/(4*pi))
        end
        if dk < 0.0
            dk = -dk
        end
        if dk > dk_threshold # For small k this limits the size of the interval
            dk = dk_threshold
        end
        push!(dk_values, dk)
        k0 += dk
    end
    println("min/max dk value: ", extrema(dk_values))

    # Initialize the progress bar with estimated number of intervals
    println("Scaling Method...")
    p = Progress(length(dk_values), 1)

    # Actual computation using precomputed dk values
    k0 = k1
    k_res, ten_res = solve_spectrum(solver, basis, billiard, k0, dk_values[1] + tol)
    control = [false for i in 1:length(k_res)]

    for i in eachindex(dk_values)
        dk = dk_values[i]
        k0 += dk
        k_new, ten_new = solve_spectrum(solver, basis, billiard, k0, dk + tol)
        overlap_and_merge!(k_res, ten_res, k_new, ten_new, control, k0 - dk, k0; tol=tol)
        next!(p)
    end
    return k_res, ten_res, control
end

# NEW
#=
function compute_spectrum(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard,k1,k2; tol=1e-4, N_expect = 3, dk_threshold=0.1, fundamental=true)
    # Estimate the number of intervals and store the dk values
    k0 = k1
    dk_values = []
    while k0 < k2
        if fundamental
            dk = N_expect / (billiard.area_fundamental * k0 / (2*pi) - billiard.length_fundamental/(4*pi))
        else
            dk = N_expect / (billiard.area * k0 / (2*pi) - billiard.length/(4*pi))
        end
        if dk < 0.0
            dk = -dk
        end
        if dk > dk_threshold # For small k this limits the size of the interval
            dk = dk_threshold
        end
        push!(dk_values, dk)
        k0 += dk
    end
    println("max/min dk value: ", extrema(dk_values))

    # Initialize the progress bar with estimated number of intervals
    println("Scaling Method...")
    p = Progress(length(dk_values), 1)

    # Actual computation using precomputed dk values
    k0 = k1
    k_res, ten_res = solve_spectrum(solver, basis, billiard, k0, dk_values[1] + tol)
    control = [false for i in 1:length(k_res)]

    for i in eachindex(dk_values)
        dk = dk_values[i]
        k0 += dk
        k_new, ten_new = solve_spectrum(solver, basis, billiard, k0, dk + tol)
        overlap_and_merge!(k_res, ten_res, k_new, ten_new, control, k0 - dk, k0; tol=tol)
        next!(p)
    end
    return k_res, ten_res, control
end
=#

# NEW WITH N1 and N2 INSTEAD OF k1 AND k2
function compute_spectrum(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard,N1::Int,N2::Int; tol=1e-4, N_expect = 3, dk_threshold=0.1, fundamental=true)
    # get the k1 and k2 from the N1 and N2
    k1 = k_at_state(N1, billiard; fundamental=fundamental)
    k2 = k_at_state(N2, billiard; fundamental=fundamental)
    # Call the k one
    k_res, ten_res, control = compute_spectrum(solver, basis, billiard, k1, k2; tol=tol, N_expect=N_expect, dk_threshold=dk_threshold, fundamental=fundamental)
    return k_res, ten_res, control
end

#=
function compute_spectrum(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard,N1::Int,N2::Int; tol=1e-4, N_expect = 3, dk_threshold=0.1, fundamental=true)
    # get the k1 and k2 from the N1 and N2
    k1 = k_at_state(N1, billiard; fundamental=fundamental)
    k2 = k_at_state(N2, billiard; fundamental=fundamental)
    # Estimate the number of intervals and store the dk values
    k0 = k1
    dk_values = []
    while k0 < k2
        if fundamental
            dk = N_expect / (billiard.area_fundamental * k0 / (2*pi) - billiard.length_fundamental/(4*pi))
        else
            dk = N_expect / (billiard.area * k0 / (2*pi) - billiard.length/(4*pi))
        end
        if dk < 0.0
            dk = -dk
        end
        if dk > dk_threshold # For small k this limits the size of the interval
            dk = dk_threshold
        end
        push!(dk_values, dk)
        k0 += dk
    end
    println(dk_values)

    # Initialize the progress bar with estimated number of intervals
    println("Scaling Method...")
    p = Progress(num_intervals, 1)

    # Actual computation using precomputed dk values
    k0 = k1
    k_res, ten_res = solve_spectrum(solver, basis, billiard, k0, dk_values[1] + tol)
    control = [false for i in 1:length(k_res)]

    for i in 1:num_intervals
        dk = dk_values[i]
        #println("Doing interval: [$(k0), $(k0 + dk)]")
        k0 += dk
        k_new, ten_new = solve_spectrum(solver, basis, billiard, k0, dk + tol)
        overlap_and_merge!(k_res, ten_res, k_new, ten_new, control, k0 - dk, k0; tol=tol)
        next!(p)
    end
    return k_res, ten_res, control
end
=#



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

struct SpectralData{T} 
    k::Vector{T}
    ten::Vector{T}
    control::Vector{Bool}
    k_min::T
    k_max::T
end

function SpectralData(k,ten,control)
    k_min = minimum(k)
    k_max = maximum(k)
    return SpectralData(k,ten,control,k_min,k_max)
end

function merge_spectra(s1, s2; tol=1e-4)
    # Define the overlap interval manually
    overlap_start = max(s1.k_min - tol / 2, s2.k_min - tol / 2)
    overlap_end = min(s1.k_max + tol / 2, s2.k_max + tol / 2)
    
    # Identify the indices of wavenumbers within the overlap interval
    idx_1 = [(k >= overlap_start) && (k <= overlap_end) for k in s1.k]
    idx_2 = [(k >= overlap_start) && (k <= overlap_end) for k in s2.k]

    ks1 = s1.k[idx_1]
    ts1 = s1.ten[idx_1]
    ks2 = s2.k[idx_2]
    ts2 = s2.ten[idx_2]

    ks_ov, ts_ov, cont_ov = match_wavenumbers(ks1,ts1,ks2,ts2)
    
    ks = append!(s1.k[.~idx_1],ks_ov)
    ts = append!(s1.ten[.~idx_1],ts_ov)
    control = append!(s1.control[.~idx_1],cont_ov)

    append!(ks, s2.k[.~idx_2])
    append!(ts, s2.ten[.~idx_2])
    append!(control, s2.control[.~idx_2])

    p = sortperm(ks) 
    return SpectralData(ks[p], ts[p], control[p])
end

#=
function compute_spectrum(solver::AbsSolver,basis::AbsBasis,billiard::AbsBilliard,N1::Int,N2::Int,dN::Int; N_expect = 2.0, tol=1e-4)
    let solver=solver, basis=basis, billiard=billiard
        N_intervals = range(N1-dN/2,N2+dN/2,step=dN)
        #println(N_intervals)
        if hasproperty(billiard,:angles)
            k_intervals = [k_at_state(n, billiard.area, billiard.length, billiard.angles) for n in N_intervals]
        else
            k_intervals = [k_at_state(n, billiard.area, billiard.length) for n in N_intervals]
        end

        results = Vector{SpectralData}(undef,length(k_intervals)-1)
        for i in 1:(length(k_intervals)-1)
            k1 = k_intervals[i]
            k2 = k_intervals[i+1]
            dk = N_expect * 2.0*pi / (billiard.area * k1) #fix this
            #println(k1)
            #println(k2)
            #println(dk)
            k_res, ten_res, control = compute_spectrum(solver,basis,billiard,k1,k2,dk; tol)
            #println(k_res)
            results[i] = SpectralData(k_res, ten_res, control)
        end

        return reduce(merge_spectra, results)
    end
end
=#

#=
using Makie
function compute_spectrum_test(solver::AbsSolver, basis::AbsBasis, pts::AbsPoints,k1,k2,dk;tol=1e-4, plot_info=false)
    k0 = k1
    #initial computation
    k_res, ten_res = solve(solver, basis, pts, k0, dk+tol)
    control = [false for i in 1:length(k_res)]
    cycle = Makie.wong_colors()[1:6]
    f = Figure(resolution = (1000,1000));
    ax = Axis(f[1,1])
    scatter!(ax, k_res, log10.(ten_res),color=(cycle[1], 0.5))
    scatter!(ax, k_res, zeros(length(k_res)),color=(cycle[1], 0.5))
    #println("iteration 0")
    #println("merged: $k_res")
    
    #println("overlaping: $ks")
    i=1
    while k0 < k2
        #println("iteration $i")
        k0 += dk
        k_new, ten_new = solve(solver, basis, pts, k0, dk+tol)
        scatter!(ax, k_new, log10.(ten_new),color=(cycle[mod1(i+1,6)], 0.5))
        scatter!(ax, k_new, zeros(length(k_new)),color=(cycle[mod1(i+1,6)], 0.5))
        i+=1
        #println("new: $k_new")
        #println("overlap: $(k0-dk), $k0")
        overlap_and_merge!(k_res, ten_res, k_new, ten_new, control, k0-dk, k0; tol=tol)
        #println("merged: $k_res")
        #println("control: $control")
    end
    scatter!(ax, k_res, log10.(ten_res), color=(:black, 1.0), marker=:x,  ms = 100)
    scatter!(ax, k_res, zeros(length(k_res)), color=(:black, 1.0), marker=:x, ms = 100)
    display(f)
    return k_res, ten_res, control
end
=#