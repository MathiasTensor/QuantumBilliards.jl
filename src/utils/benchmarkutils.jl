using Makie, ProgressMeter

# helper for determining the byte size of object "a"
memory_size(a) = Base.format_bytes(Base.summarysize(a)) 

"""
    BenchmarkInfo

Stores benchmark information for a given solver, matrix dimensions, memory usage, construction, decomposition, solution times and eigenvalue results.

- `solver::AbsSolver`: The solver to benchmark.
- `matrix_dimensions::Vector`: The dimensions of the matrices to benchmark.
- `matrix_memory::Vector`: Memory usage of the matrices.
- `construction_time::Float64`: Construction time for the benchmark.
- `decomposition_time::Float64`: Decomposition time for the benchmark.
- `solution_time::Float64`: Solution time for the benchmark.
- `results::Tuple`: Eigenvalue results for the benchmark.
"""
struct BenchmarkInfo
    solver::AbsSolver
    matrix_dimensions::Vector
    matrix_memory::Vector
    construction_time::Float64
    decomposition_time::Float64
    solution_time::Float64
    results::Tuple
end

# Helper function that prints the structs values
function print_benchmark_info(info::BenchmarkInfo)
    println("Solver: $(info.solver)")
    for i in eachindex(info.matrix_dimensions)
        println("Matrix $i, $(info.matrix_dimensions[i]), memory: $(info.matrix_memory[i])")
    end
    println("Construction time: $(info.construction_time) s")
    println("Decomposition time: $(info.decomposition_time) s")
    println("Solution time: $(info.solution_time) s")
    println("Results: $(info.results)")
end

"""
    benchmark_solver(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard, k::T, dk::T; btimes = 1, print_info=true, plot_matrix=false,log=false, return_mat=false, kwargs...) where {T<:Real}

Construct and solve benchmark problems for a given solver, basis, billiard, k, dk with optional kwargs. This is a manual check if the chosen solver params are correct.

# Arguments:
- `solver::AbsSolver`: The solver to benchmark.
- `basis::AbsBasis`: The basis to use for the benchmark.
- `billiard::AbsBilliard`: The billiard to use for the benchmark.
- `k::T`: the wavevector reference point
- `dk::T`: tolerance cutoff for ks in solving the eigenvalue problem.
- `btimes::Int=1`: Number of benchmark runs.
- `print_info::Bool=true`: Whether to print benchmark information.
- `plot_matrix::Bool=false`: Whether to plot the basis matrix.
- `log::Bool=false`: Whether to log the benchmark results.
- `return_mat::Bool=false`: Whether to return the basis matrix.
- `kwargs...`: Additional keyword arguments to pass to the solver and billiard.

# Returns:
- `BenchmarkInfo`: A BenchmarkInfo object containing the benchmark results.
- `Tuple{BenchmarkInfo,Tuple{Matrix{T}}}`: benchmark info the and the matrices for saving.

"""
function benchmark_solver(solver::AbsSolver, basis::AbsBasis, billiard::AbsBilliard, k::T, dk::T; btimes = 1, print_info=true, plot_matrix=false,log=false, return_mat=false, kwargs...) where {T<:Real}
    let L = billiard.length, dim = round(Int, L*k*solver.dim_scaling_factor/(2*pi))
        basis_new = resize_basis(basis,billiard, dim, k)
        pts = evaluate_points(solver, billiard, k)
        function run(fun, args...; kwargs...)
            times = Float64[]
            values = []
            for i in 1:btimes
                res = @timed fun(args...; kwargs...)
                push!(times,res.time)
                push!(values, res.value)
            end
            return values[1], mean(times)
        end
        if print_info  #returns also information about the basis matrix in the last element
            mat_res, mat_time = run(construct_matrices_benchmark, solver, basis_new, pts, k)
        else
            mat_res, mat_time = run(construct_matrices, solver, basis_new, pts, k)
        end
        mat_dim = []
        mat_mem = []
        for A in mat_res
            push!(mat_dim, size(A))
            push!(mat_mem, memory_size(A))
        end
        if typeof(solver) <:AcceleratedSolver
            sol, decomp_time = run(solve, solver, mat_res[1],mat_res[2],k,dk)
            sorted_pairs = sort(collect(zip(sol[1], sol[2])), by = x -> x[2])
            sol = Tuple([[a,b] for (a,b) in sorted_pairs][1:min(5,length(sol[1]))])
            info = BenchmarkInfo(solver,mat_dim,mat_mem,mat_time,decomp_time,mat_time+decomp_time,sol)
        end
        if typeof(solver) <:SweepSolver
            t, decomp_time = run(solve, solver, mat_res[1],mat_res[2])
            res, sol_time = run(solve_wavenumber,solver, basis, billiard,k,dk)
            info = BenchmarkInfo(solver,mat_dim,mat_mem,mat_time,decomp_time,sol_time,res) 
        end  
        if print_info
            print_benchmark_info(info)
        end
        if plot_matrix
            mat_n = length(mat_res)
            f = Figure(resolution = (500*mat_n,500))
            for i in 1:mat_n
                plot_matrix!(f[1,i],mat_res[i],log=log)
            end
            display(f)
        end
        if return_mat
            return info, mat_res
        else
            return info
        end
    end
end

"""
plot_Z!(f::Figure, Z::AbstractMatrix; title::AbstractString="")

Plots the heatmap of a given matrix `Z` with significance levels adjusted based on a 
specified threshold of eps(), and overlays a color bar. Also plots the lines where the first row/col is 0.0.

# Arguments
- `f::Figure`: A `Figure` object where the heatmap will be plotted.
- `Z::Matrix`: A matrix representing the data to be visualized.
- `title::String`: A string specifying the title of the plot (default: "").

# Details
- Entries in `Z` below machine epsilon (`eps()`) are treated as NaN and ignored in the plot.
- The color range is automatically balanced around the maximum absolute value in `Z`.
- A color bar is displayed alongside the plot to indicate the data scale.
"""
function plot_Z!(f::Figure,i::Integer,j::Integer,Z::Matrix;title::String="")
    Z = deepcopy(Z)
    ax=Axis(f[i,j][1,1];title=title)
    m = findmax(abs.(Z))[1]
    Z[abs.(Z).<eps()].=NaN
    nan_row=findfirst(row->all(isnan,Z[row,:]),axes(Z,1))
    nan_col=findfirst(col->all(isnan,Z[:,col]),axes(Z,2))
    Z[isnan.(Z)].=m # to better see
    range_val=(-m,m) 
    hmap=heatmap!(ax,Z,colormap=:balance,colorrange=range_val)
    lines!(ax,[1,size(Z,2)],[nan_row,nan_row],color=:green,linewidth=2,linestyle=:dash)
    lines!(ax,[nan_col,nan_col],[1,size(Z,1)],color=:green,linewidth=2,linestyle=:dash)
    ax.yreversed=false
    ax.aspect=DataAspect()
    Colorbar(f[i,j][1,2],colormap=:balance,limits=Float64.(range_val),tellheight=true)
end

"""
    is_converged_pairwise(current::Matrix{T}, previous::Matrix{T}, tol::T=T(1e-6)) where {T<:Real}

Checks each elemement of the 2 same sized matrices and finds the largest difference between the matrix elemets.

# Arguements
- `current::Matrix{T}`: Current matrix to compare.
- `previous::Matrix{T}`: Previous matrix to compare.
- `tol::T`: Tolerance for comparison (default: 1e-6).

# Returns
- `Bool`: Returns `true` if the largest difference between the matrix elements is less than the tolerance, `false` otherwise.
"""
function is_converged_pairwise(current::Matrix{T}, previous::Matrix{T}, tol::T=T(1e-6)) where {T<:Real}
    @assert size(current)==size(previous) "Matrices must have the same dimensions for comparison."
    valid_mask=.!isnan.(current).&.!isnan.(previous)
    max_diff=maximum(abs.(current[valid_mask].-previous[valid_mask]))
    return max_diff<tol
end

"""
    compute_benchmarks(solver, basis, billiard, k, dk; d_range = [2.0], b_range=[2.0],btimes=1)

Wrapper for benchmark_solver that also iterates over a d and b range to chekc when we have unchanging solutions to the eigenvalues and tensions.

# Arguments
- `solver::AbsSolver`: The solver to benchmark.
- `basis::AbsBasis`: The basis to use for the benchmark.
- `billiard::AbsBilliard`: The billiard to use for the benchmark.
- `k::T`: the wavevector reference point
- `dk::T`: tolerance cutoff for ks in solving the eigenvalue problem.
- `d_range::Vector{T}`: Range of d values to benchmark.
- `b_range::Vector{T}`: Range of b values to benchmark.

# Returns
- `Vector{BenchmarkInfo}`: A vector of BenchmarkInfo objects for each combination of d and b in the given ranges.
"""
function compute_benchmarks(solver::Sol, basis::Ba, billiard::Bi, k::T, dk::T; d_range=[2.0], b_range=[2.0], btimes=1) where {Sol<:AbsSolver,Ba<:AbsBasis,Bi<:AbsBilliard,T<:Real}
    make_solver(solver,d,b) = typeof(solver)(d,b) 
    grid_indices = CartesianIndices((length(d_range), length(b_range)))
    info_matrix = [benchmark_solver(make_solver(solver,d_range[i[1]],b_range[i[2]]),basis,billiard,k,dk;print_info=false,btimes=btimes) for i in grid_indices]
    return info_matrix
end

#### BIM AND EBIM BENCHMARKS ####

#### BENCHMARKS ####

@timeit TO "evaluate_points" function evaluate_points_timed(solver::BoundaryIntegralMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    return evaluate_points(solver,billiard,k)
end

@timeit TO "construct_matrices" function construct_matrices_timed(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    return construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
end

@timeit TO "SVD" function svdvals_timed(A)
    return svdvals(A)
end

function solve_timed(solver::BoundaryIntegralMethod,billiard::Bi,k::Real;multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts=evaluate_points_timed(solver,billiard,k)
    A=construct_matrices_timed(solver,AbstractHankelBasis(),pts,k;multithreaded=multithreaded)
    σs=svdvals_timed(A)
    show(TO)
    reset_timer!(TO)
end

### HELPERS FOR FINDING THE PEAKS ###

"""
    find_peaks(x::Vector{T}, y::Vector{T}; threshold=200.0) where {T<:Real}

Finds the x-coordinates of local maxima in the `y` vector that are greater than the specified `threshold`.

# Arguments
- `x::Vector{T}`: The x-coordinates corresponding to the y-values.
- `y::Vector{T}`: The y-values to search for peaks.
- `threshold::Union{T, Vector{T}}`: Minimum value a peak must exceed to be considered.
  Can be a scalar (applied to all peaks) or a vector (element-wise comparison).
  Default is 200.0.

# Returns
- `Vector{T}`: A vector of x-coordinates where peaks are located.
"""
function find_peaks(x::Vector{T}, y::Vector{T}; threshold::Union{T,Vector{T}}=200.0) where {T<:Real}
    peaks=T[]
    threshold_vec=length(threshold)==1 ? fill(threshold,length(x)) : threshold
    for i in 2:length(y)-1
        if y[i]>y[i-1] && y[i]>y[i+1] && y[i]>threshold_vec[i]
            push!(peaks,x[i])
        end
    end
    return peaks
end

"""
    bim_second_derivative(x::Vector{T}, y::Vector{T}) where {T<:Real}

Computes the second derivative of `y` with respect to `x` using finite differences between the xs in `x`.

# Arguments
- `x::Vector{T}`: The x-coordinates of the data.
- `y::Vector{T}`: The y-values of the data.

# Returns
- `Vector{T}`: Midpoints of the x-values for the second derivative.
- `Vector{T}`: The second derivative of `y` with respect to `x`.
"""
function bim_second_derivative(x::Vector{T}, y::Vector{T}) where {T<:Real}
    first_grad=diff(y)./diff(x)
    first_mid_x=@. (x[1:end-1]+x[2:end])/2
    second_grad=diff(first_grad)./diff(first_mid_x)
    second_mid_x=@. (first_mid_x[1:end-1]+first_mid_x[2:end])/2
    return second_mid_x,second_grad
end

"""
    get_eigenvalues(k_range::Vector{T}, tens::Vector{T}; threshold=200.0) where {T<:Real}

Finds peaks in the second derivative of the logarithm of `tens` with respect to `k_range`. These peaks are as precise as the k step that was chosen in `k_range`.

# Arguments
- `k_range::Vector{T}`: The range of `k` values.
- `tens::Vector{T}`: The tension values.
- `threshold::Real`: Minimum value a peak in the second derivative gradient must exceed. Default is 200.0.

# Returns
- `Vector{T}`: The `k_range` values where peaks in the second derivative gradient are found.
"""
function get_eigenvalues(k_range::Vector{T}, tens::Vector{T}; threshold=200.0) where {T<:Real}
    mid_x,gradient=bim_second_derivative(k_range,log10.(tens))
    return find_peaks(mid_x,gradient;threshold=threshold)
end

### EBIM ###

#### DEBUGGING TOOLS ####

"""
    solve_DEBUG_w_2nd_order_corrections(
        solver::ExpandedBoundaryIntegralMethod,
        basis::Ba,
        pts::BoundaryPoints,
        k;
        kernel_fun=(:default, :first, :second)
    ) -> (Vector{T}, Vector{T}, Vector{T}, Vector{T})

A debug routine that solves the generalized eigenproblem `(A, dA)` at wavenumber `k`, then applies
**both first- and second-order** corrections to refine the approximate roots. Specifically,
it extracts λ from `A*x = λ dA*x`, then does:

  corr₁[i] = -λ[i]
  corr₂[i] = -0.5 * corr₁[i]^2 * real( (v_leftᵀ ddA v_right) / (v_leftᵀ dA v_right) )

Hence two sets of corrected wavenumbers: `k + corr₁` and `k + corr₁ + corr₂`. Tensions are `|corr₁|`
and `|corr₁ + corr₂|`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The EBIM solver config.
- `basis::Ba`: Basis function type.
- `pts::BoundaryPoints`: Boundary geometry.
- `k`: Wavenumber for the eigenproblem.
- `kernel_fun`: A triple `(base, first, second)` or custom functions for kernel & derivatives.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(λ_corrected_1, tens_1, λ_corrected_2, tens_2)`: 
   1. `λ_corrected_1 = k + corr₁` (1st-order),
   2. `tens_1 = abs(corr₁)`,
   3. `λ_corrected_2 = k + corr₁ + corr₂` (2nd-order),
   4. `tens_2 = abs(corr₁ + corr₂)`.
"""
function solve_DEBUG_w_2nd_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    λ,VR,VL=generalized_eigen_all(A,dA)
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    T=eltype(real.(λ))
    λ=real.(λ)
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        numerator=transpose(v_left)*ddA*v_right
        denominator=transpose(v_left)*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    λ_corrected_1=k.+corr_1
    λ_corrected_2=λ_corrected_1.+corr_2
    tens_1=abs.(corr_1)
    tens_2=abs.(corr_1.+corr_2)
    return λ_corrected_1,tens_1,λ_corrected_2,tens_2
end

"""
    ebim_inv_diff(kvals::Vector{T}) where {T<:Real}

Computes the inverse of the differences between consecutive elements in `kvals`. This inverts the small differences between the ks very close to the correct eigenvalues and serves as a visual aid or potential criteria for finding missing levels.

# Arguments
- `kvals::Vector{T}`: A vector of values for which differences are calculated.

# Returns
- `Vector{T}`: The `kvals` vector excluding its last element.
- `Vector{T}`: The inverse of the differences between consecutive elements in `kvals`.
"""
function ebim_inv_diff(kvals::Vector{T}) where {T<:Real}
    kvals_diff=diff(kvals)
    kvals=kvals[1:end-1]
    return kvals,T(1.0)./kvals_diff
end

"""
    visualize_ebim_sweep(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1,k2;dk=(k)->(0.05*k^(-1/3)),multithreaded::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbstractHankelBasis,Bi<:AbsBilliard}

Debugging Function to sweep through a range of `k` values and evaluate the smallest tension for each `k` using the EBIM method. This function identifies corrected `k` values based on the generalized eigenvalue problem and associated tensions, collecting those with the smallest tensions for further analysis.

# Usage
hankel_basis=AbstractHankelBasis()
@time ks_debug,tens_debug,ks_debug_small,tens_debug_small=QuantumBilliards.visualize_ebim_sweep(ebim_solver,hankel_basis,billiard,k1,k2;dk=dk)
scatter!(ax,ks_debug,log10.(tens_debug), color=:blue, marker=:xcross)
-> This gives a sequence of points that fall on a vertical line when close to an actual eigenvalue. 

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration for the EBIM method.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `billiard::Bi`: The billiard geometry, a subtype of `AbsBilliard`.
- `k1`: The initial value of `k` for the sweep.
- `k2`: The final value of `k` for the sweep.
- `dk::Function`: A function defining the step size as a function of `k` (default: `(k) -> (0.05 * k^(-1/3))`).
- `multithreaded::Bool=false`: If the matrix construction should be multithreaded.
- `multithreaded_ks::Bool=true`: If the ks loop should be rather multithreaded.

# Returns
- `Vector{T}`: All corrected `k` values with low tensions throughout the sweep (`ks_all`).
- `Vector{T}`: Inverse tension corresponding to `ks_all` (`tens_all`), which represent the inverse distances between consecutive `ks_all`. Aa large number indicates that we are probably close to an eigenvalue since solution of the ebim sweep tend to accumulate there.
"""
function visualize_ebim_sweep(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1,k2;dk=(k)->(0.05*k^(-1/3)),multithreaded::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    k=k1
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.symmetry)
    T=eltype(k1)
    ks=T[] # these are the evaluation points
    push!(ks,k1)
    k=k1
    while k<k2
        k+=dk(k)
        push!(ks,k)
    end
    ks_all_1=Vector{Union{T,Missing}}(missing,length(ks))
    ks_all_2=Vector{Union{T,Missing}}(missing,length(ks))
    tens_all_1=Vector{Union{T,Missing}}(missing,length(ks))
    tens_all_2=Vector{Union{T,Missing}}(missing,length(ks))
    all_pts=Vector{BoundaryPoints{T}}(undef,length(ks))
    @showprogress desc="Calculating boundary points..." for i in eachindex(ks) 
        all_pts[i]=evaluate_points(bim_solver,billiard,ks[i])
    end
    @info "EBIM smallest tens..."
    p=Progress(length(ks),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)
        ks1,tens1,ks2,tens2=solve_DEBUG_w_2nd_order_corrections(solver,basis,all_pts[i],ks[i],multithreaded=multithreaded)
        idx1=findmin(tens1)[2]
        idx2=findmin(tens2)[2]
        if log10(tens1[idx1])<0.0
            ks_all_1[i]=ks1[idx1]
            tens_all_1[i]=tens1[idx1]   
        end
        if log10(tens2[idx2])<0.0
            ks_all_2[i]=ks2[idx2]
            tens_all_2[i]=tens2[idx2]
        end
        next!(p)
    end
    ks_all_1=skipmissing(ks_all_1)|>collect
    tens_all_1=skipmissing(tens_all_1)|>collect
    ks_all_2=skipmissing(ks_all_2)|>collect
    tens_all_2=skipmissing(tens_all_2)|>collect
    _,logtens_1=ebim_inv_diff(ks_all_1)
    _,logtens_2=ebim_inv_diff(ks_all_2)
    idxs1=findall(x->x>0.0,logtens_1)
    idxs2=findall(x->x>0.0,logtens_2)
    logtens_1=logtens_1[idxs1]
    logtens_2=logtens_2[idxs2]
    ks_all_1=ks_all_1[idxs1]
    ks_all_2=ks_all_2[idxs2]
    return ks_all_1,logtens_1, ks_all_2,logtens_2
end

"""
    visualize_cond_dA_ddA_vs_k(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1::T,k2::T;dk=(k)->(0.05*k^(-1/3)),multithreaded_matrices::Bool=false,multithreaded_ks=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}

Useful function to check the conditions numbers of the relevant Fredholm matrix and it's derivatives in the given k-range. This is to check the numerical stability of the method, especially very close to a true eigenvalue. It is quite useful to plot the ks vs. log of the returned results vectors for A, dA, ddA to see deeper insights.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration for the expanded boundary integral method.
- `billiard::Bi`: The billiard configuration, a subtype of `AbsBilliard`.
- `k1::T`: Starting wavenumber for the spectrum calculation.
- `k2::T`: Ending wavenumber for the spectrum calculation.
- `dk::Function`: Custom function to calculate the wavenumber step size. Defaults to a scaling law inspired by Veble's paper.
- `tol=1e-4`: Tolerance for the overlap_and_merge function that samples a bit outside the merging interval for better results.
- `multithreaded_matrices::Bool=false`: If the Fredholm matrix construction and it's derivatives should be done in parallel.
- `multithreaded_ks::Bool=true`: If the k loop is multithreaded. This is usually the best choice since matrix construction for small k is not as costly.

# Returns
- `(ksA,resultsA)::Tuple{Vector{T},Vector{T}}`: The ks and conditions numbers for the A matrix where LAPACK did not crash.
- `(ksdA,resultsdA)::Tuple{Vector{T},Vector{T}}`: The ks and conditions numbers for the dA matrix where LAPACK did not crash.
- `(ksddA,resultsddA)::Tuple{Vector{T},Vector{T}}`: The ks and conditions numbers for the ddA matrix where LAPACK did not crash.
- `(det_ksA,det_resultsA)::Tuple{Vector{T},Vector{T}}`: The ks and det numbers for the A matrix where LAPACK did not crash.
- `(det_ksdA,det_resultsdA)::Tuple{Vector{T},Vector{T}}`: The ks and det numbers for the dA matrix where LAPACK did not crash.
- `(det_ksddA,det_resultsddA)::Tuple{Vector{T},Vector{T}}`: The ks and det numbers for the ddA matrix where LAPACK did not crash.

"""
function visualize_cond_dA_ddA_vs_k(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1::T,k2::T;dk=(k)->(0.05*k^(-1/3)),multithreaded_matrices::Bool=false,multithreaded_ks=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    basis=AbstractHankelBasis()
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.symmetry)
    ks=T[]
    dks=T[]
    k=k1
    while k<k2
        push!(ks,k)
        kstep=dk(k)
        k+=kstep
        push!(dks,kstep)
    end
    println("EBIM...")
    all_pts=Vector{BoundaryPoints{T}}(undef,length(ks))
    @showprogress desc="Calculating boundary points EBIM..." Threads.@threads for i in eachindex(ks)
        all_pts[i]=evaluate_points(deepcopy(bim_solver),billiard,ks[i])
    end
    resultsA=Vector{Union{T,Missing}}(missing,length(ks))
    resultsdA=Vector{Union{T,Missing}}(missing,length(ks))
    resultsddA=Vector{Union{T,Missing}}(missing,length(ks))
    det_resultsA=Vector{Union{T,Missing}}(missing,length(ks))
    det_resultsdA=Vector{Union{T,Missing}}(missing,length(ks))
    det_resultsddA=Vector{Union{T,Missing}}(missing,length(ks))
    p=Progress(length(ks),1) # first one finished
    println("Constructing dA, ddA and evaluating cond...")
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)
        A,dA,ddA=construct_matrices(solver,basis,all_pts[i],ks[i],multithreaded=multithreaded_matrices)
        try
            cA=cond(A)
            resultsA[i]=cA
        catch e
            @warn "cond(A) failed at k = $(ks[i]) with error $e"
        end
        try
            det_cA=logabsdet(A)[1]
            det_resultsA[i]=det_cA
        catch e
            @warn "logabsdet(A) failed at k = $(ks[i]) with error $e"
        end
        try
            cdA=cond(dA)
            resultsdA[i]=cdA
        catch e 
            @warn "cond(dA) failed at k = $(ks[i]) with error $e"
        end
        try
            det_cdA=logabsdet(dA)[1]
            det_resultsdA[i]=det_cdA
        catch e
            @warn "logabsdet(dA) failed at k = $(ks[i]) with error $e"
        end
        try # since most cases the LAPACK solver will crash when calculating the condition number of ddA. In those cases it is also useless to compute it since we need to divide by ddA in the 2nd order corrections and it will give unstable results.
            cddA=cond(ddA)
            resultsddA[i]=cddA
        catch e 
            @warn "cond(ddA) failed at k = $(ks[i]) with error $e"
        end
        try
            det_cddA=logabsdet(ddA)[1]
            det_resultsddA[i]=det_cddA
        catch e
            @warn "logabsdet(ddA) failed at k = $(ks[i]) with error $e"
        end
        next!(p)
    end
    function filter_valid(xs::Vector{Union{T,Missing}},ks::Vector{T}) where {T}
        idxs=findall(!ismissing,xs)
        return xs[idxs],ks[idxs]
    end
    # Filter condition numbers and their corresponding ks
    resultsA,ksA=filter_valid(resultsA,ks)
    resultsdA,ksdA=filter_valid(resultsdA,ks)
    resultsddA,ksddA=filter_valid(resultsddA,ks)
    # Filter determinants and their corresponding ks
    det_resultsA,det_ksA=filter_valid(det_resultsA,ks)
    det_resultsdA,det_ksdA=filter_valid(det_resultsdA,ks)
    det_resultsddA,det_ksddA=filter_valid(det_resultsddA,ks)
    return (ksA,resultsA),(ksdA,resultsdA),(ksddA,resultsddA),(det_ksA,det_resultsA),(det_ksdA,det_resultsdA),(det_ksddA,det_resultsddA)
end
