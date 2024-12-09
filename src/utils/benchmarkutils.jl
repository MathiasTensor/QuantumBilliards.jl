#include("../abstracttypes.jl")
#include("../solvers/particularsolutionsmethod.jl")
#include("../plotting/matrixplotting.jl")
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
    dynamical_solver_construction(k1::T, k2::T, basis::Ba, billiard::Bi;d0::T = T(1.0), b0::T = T(2.0),solver_type::Symbol = :Accelerated, partitions::Integer = 10,samplers::Vector{Sam}, min_dim::Integer = 100, min_pts::Integer = 500,dd::T = T(0.1), db::T = T(0.3),return_benchmarked_matrices::Bool = true, display_benchmarked_matrices::Bool = true) where {T<:Real, Sam<:AbsSampler, Ba<:AbsBasis, Bi<:AbsBilliard}

Constructs solvers dynamically for a range of wavenumbers, `k1` to `k2`, optimizing for parameters `d` and `b`. This functions allows the user to not have to make manual tests for when the solvers are optimal in a given range of ks, since these parameters change (albeit slowly) throughout the spectrum computations.

# Description of Return


# Arguments
- `k1::T`, `k2::T`: Start and end wavenumbers for the solver.
- `basis::Ba`: Basis object to be resized for the solvers.
- `billiard::Bi`: Billiard object specifying the geometry of the problem.
- `d0::T`: Initial value for parameter `d` (default: 1.0).
- `b0::T`: Initial value for parameter `b` (default: 2.0).
- `solver_type::Symbol`: Type of solver to use. Options are:
  - `:Accelerated`
  - `:ParticularSolutions`
  - `:Decomposition`
  - `:BoundaryIntegralMethod`
- `partitions::Integer`: Number of partitions to divide the `k1` to `k2` range (default: 10).
- `samplers::Vector{Sam}`: Vector of samplers for the solver.
- `min_dim::Integer`: Minimum dimension for the solver's basis (default: 100).
- `min_pts::Integer`: Minimum number of points for evaluation (default: 500).
- `dd::T`: Increment step for parameter `d` during optimization (default: 0.1).
- `db::T`: Increment step for parameter `b` during optimization (default: 0.3).
- `return_benchmarked_matrices::Bool`: If `true`, returns the benchmarked matrices (default: `true`).
- `display_benchmarked_matrices::Bool`: If `true`, displays the matrices as heatmaps (default: `true`).
- `print_params::Bool`: If printing d,b determined params. Default is `true`.

# Returns
- If `return_benchmarked_matrices` is `true`: A tuple `(matrices_k_dict, solvers)`, where:
  - `sorted_entries::Vector{Pair{T,Vector{Matrix{T}}}}`: collected dictionary mapping wavenumbers to matrices.
  - `solvers::Vector`: List of constructed solvers.
- If `return_benchmarked_matrices` is `false`: Only the `solvers` vector.
"""
function dynamical_solver_construction(k1::T, k2::T, basis::Ba, billiard::Bi; d0::T=T(1.0), b0::T=T(2.0), solver_type::Symbol=:Accelerated, partitions::Integer=10, samplers::Vector{Sam}=[GaussLegendreNodes()], min_dim=100, min_pts=500, dd=0.1, db=0.3, return_benchmarked_matrices=true, display_benchmarked_matrices=true, print_params=true) where {T<:Real,Sam<:AbsSampler,Ba<:AbsBasis,Bi<:AbsBilliard}
    ds=Vector{T}(undef,partitions-1) # temp storage for part 1
    bs=Vector{T}(undef,partitions-1) # together with ds construct returned solvers
    matrices_k_dict = Dict{T,Vector{Matrix{T}}}()
    # helper solver constructor
    construct_solver(_d,_b,type_sol) = begin
        if type_sol==:Accelerated
            return ScalingMethodA(_d,_b,samplers;min_dim=min_dim,min_pts=min_pts)
        elseif type_sol==:ParticularSolutions
            return ParticularSolutionsMethod(_d,_b,_b,samplers;min_dim=min_dim,min_pts=min_pts,min_int_pts=min_pts)
        elseif type_sol==:Decomposition
            return DecompositionMethod(_d,_b,samplers;min_dim=min_dim,min_pts=min_pts)
        elseif type_sol==:BoundaryIntegralMethod
            return BoundaryIntegralMethod(_b,samplers;min_pts=min_pts)
        end
    end
    # preallocate solver vectors
    if solver_type==:Accelerated
        solvers=Vector{AcceleratedSolver}(undef,partitions-1)
    elseif solver_type==:ParticularSolutions
        solvers=Vector{ParticularSolutionsMethod}(undef,partitions-1)
    elseif solver_type==:Decomposition
        solvers=Vector{DecompositionMethod}(undef,partitions-1)
    elseif solver_type==:BoundaryIntegralMethod
        solvers=Vector{BoundaryIntegralMethod}(undef,partitions-1)
    end
    # iterate over the ks at the ends and start the next d when the previous smaller k ends. THIS ONE JUST DETERMINES D. WHEN THIS ONE IS OK WE DETERMINE B.
    ks_ends=collect(range(k1,k2,partitions))
    intervals::Vector{Tuple{T,T}} = [(ks_ends[i],ks_ends[i+1]) for i in 1:(length(ks_ends)-1)]
    b=b0 # placeholder for part 1
    @showprogress "Determining optimal d values..." for (i,(_,k_end)) in enumerate(intervals)
        d=d0 # start anew for next k
        converged=false 
        while !converged
            solver=construct_solver(d,b,solver_type)
            L = billiard.length;dim=round(Int,L*k_end*solver.dim_scaling_factor/(2*pi))
            basis_new=resize_basis(basis,billiard,dim,k_end)
            pts=evaluate_points(solver,billiard,k_end) # this is already the correct BoundaryPoints type based on the solver type
            mat=construct_matrices(solver,basis_new,pts,k_end)
            M=deepcopy(mat)
            if length(mat)==1 # BIM 
                mat[abs.(mat).<eps(T)].=NaN
                has_nan_column=any(all(isnan,matrix[:,col]) for col in axes(matrix,2)) # test whether a column is all NaN which means we have reached a satisfactory d. Could use just the end column for NaN test but this is matrix orientation independant and more general. Not crucial step so we can leave it as-is
                if has_nan_column 
                    matrices_k_dict[k_end]=[M] # the b variation will not show in the matrices so we can return them at this step
                    ds[i]=d
                    converged=true # break
                end
            else # Scaling, Decomposition or PSM
                m1,m2=mat
                M1=deepcopy(m1);M2=deepcopy(m2)
                m1[abs.(m1).<eps(T)].=NaN;m2[abs.(m2).<eps(T)].=NaN
                has_nan_column1=any(all(isnan,m1[:,col]) for col in axes(m1,2))
                has_nan_column2=any(all(isnan,m2[:,col]) for col in axes(m2,2))
                if has_nan_column1&&has_nan_column2
                    matrices_k_dict[k_end]=[M1,M2]
                    ds[i]=d
                    converged=true # break
                end
            end
            d+=dd
        end
    end
    previous_ks=fill(NaN,partitions) # for while loop first iteration check since no previous k
    ks_min=Vector{Tuple{T,T,T}}()
    @showprogress "Determining optimal b values..." for (i,(_,k_end)) in enumerate(intervals)
        b=b0
        converged=false 
        while !converged
            solver=construct_solver(ds[i],b,solver_type)
            L = billiard.length;dim=round(Int,L*k_end*solver.dim_scaling_factor/(2*pi))
            basis_new=resize_basis(basis,billiard,dim,k_end)
            dk=2/(billiard.area_fundamental*k_end/(2*pi)-billiard.length_fundamental/(4*pi)) # this ensures a minimum will be found due to weyl's law
            res = solve_wavenumber(solver,basis_new,billiard,k_end,dk)
            k_res,ten=res
            if !isnan(previous_ks[i])&&abs(k_res-previous_ks[i])<sqrt(eps(T))
                converged=true
                bs[i]=b
            end
            push!(ks_min,(k_end,k_res,ten))
            previous_ks[i]=k_res
            b+=db
        end
    end
    if print_params
        printstyled("k evaluation point:",italic=true,color=:cyan,bold=true)
        println()
        println(ks_ends)
        printstyled("Optimal d:",italic=true,color=:cyan,bold=true)
        println()
        println(ds)
        printstyled("Optimal b:",italic=true,color=:cyan,bold=true)
        println()
        println(bs)
        printstyled("Eigenvalues:",italic=true,color=:cyan,bold=true)
        println()
        colors=[:red,:green,:blue,:yellow,:cyan,:magenta] # Define some colors
        for (i,tuple) in enumerate(ks_min)
            color=colors[mod(i-1,length(colors))+1]
            printstyled(tuple,color=color)
            println()
        end
    end
    sorted_entries::Vector{Pair{T,Vector{Matrix{T}}}}=sort(collect(matrices_k_dict);by=x->x[1])
    if display_benchmarked_matrices
        f=Figure(resolution=(500*length(first(values(matrices_k_dict))),500*partitions))
        r=1 
        for (key,vals) in sorted_entries
            c=1
            for val in vals
                plot_Z!(f,r,c,val;title="$key")
                c+=1 
            end
            r+=1 
        end
        display(f)
    end
    solvers=[construct_solver(d,b,solver_type) for (d,b) in zip(ds,bs)]
    if return_benchmarked_matrices
        return sorted_entries, solvers, intervals
    else
        return solvers, intervals
    end
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
