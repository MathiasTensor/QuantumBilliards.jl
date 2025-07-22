using LinearAlgebra, Optim, ProgressMeter

"""
    solve_wavenumber(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,k::Real,dk::Real) -> Tuple{Real, Real}

Solves for the wavenumbers `k0` its corresponding tension `t0` within a given range `dk`. It returns the one with the lowest tension and not all those that are in the interval.

# Arguments
- `solver::SweepSolver`: The solver configuration for performing the sweep.
- `basis::AbsBasis`: The basis to be used. Can use also the AbstractHankelBasis() when solver is `BoundaryIntegralMethod`.
- `billiard::AbsBilliard`: The billiard configuration.
- `k::Real`: Central wavenumber for the optimization.
- `dk::Real`: Range to search for the lowest tension.

# Returns
- `Tuple{Real, Real}`: 
  - `k0`: Wavenumber with lowest tension.
  - `t0`: Corresponding lowest tension.
"""
function solve_wavenumber(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    function f(k)
        return solve(solver,new_basis,pts,k,multithreaded=true) # for a single tensions minima to check always multithread construction.
    end
    res=optimize(f,k-0.5*dk,k+0.5*dk)
    k0,t0=res.minimizer,res.minimum
    return k0,t0
end

"""
    k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::Vector{Real}) -> Vector{Real}

Performs a sweep over a range of wavenumbers `ks` and computes tensions for `res` each. If one wants to use different kernels in the BoundaryIntegralMethod one needs to change the kernel_fun which is the Second Layer potential of the original differential equation. By default the SL potential of the Helmholtz equation is used.

# Arguments
- `solver::SweepSolver`: The solver configuration for performing the sweep.
- `basis::AbsBasis`: The basis to be used. Can use also the AbstractHankelBasis() when solver is `BoundaryIntegralMethod`.
- `billiard::AbsBilliard`: The billiard configuration.
- `ks::Vector{Real}`: Vector of wavenumbers over which to perform the sweep.
- `kernel_fun::Union{Symbol, Function}`: Kernel function to use in the boundary integral method. Defaults to `:default`.
- `multithreaded_matrices::Bool=true`: If the matrix construction should be multithreaded for the basis and gradient matrices. Very dependant on the k grid and the basis choice to determine the optimal choice for what to multithread.
- `multithreaded_ks::Bool=false`: If the k loop is multithreaded.
- `use_combined::Bool=false`: Whether to use Single and Double Layer potential in CFIE methods. Defaults to `false` corresponding to just the Double Layer potential.

# NOTE
When using the Sweep solvers it is advised to parallelize the matrix construction instead of the ks loop since the matrix construction uses the same or more resources as the SVD.

# Returns
- `Vector{Real}`: Tensions of the `solve` function for each wavenumber in `ks`.
"""
function k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks;kernel_fun::Union{Symbol,Function}=:default,multithreaded_matrices::Bool=true,multithreaded_ks=false,use_combined::Bool=false)
    k=maximum(ks)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    res=similar(ks)
    num_intervals=length(ks)
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(num_intervals,1)
    if solver isa BoundaryIntegralMethod
        res[1]=solve_INFO(solver,new_basis,pts,ks[1],kernel_fun=kernel_fun,multithreaded=multithreaded_matrices)
        @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[2:end]
            res[i]=solve(solver,new_basis,pts,ks[i],kernel_fun=kernel_fun,multithreaded=multithreaded_matrices)
            next!(p)
        end
    elseif (solver isa CFIE_polar_nocorners) || (solver isa CFIE_polar_corner_correction)
        res[1]=solve_INFO(solver,new_basis,pts,ks[1],multithreaded=multithreaded_matrices,use_combined=use_combined)
        pts=evaluate_points(solver,billiard,k)
        N=length(pts.xy)
        Rmat=zeros(eltype(res[1]),N,N)
        solver isa CFIE_polar_nocorners ? kress_R_fft!(Rmat) : kress_R_sum!(Rmat,pts.ts) # external R since quite costly
        @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[2:end]
            res[i]=solve_external_R(solver,new_basis,pts,ks[i],Rmat,multithreaded=multithreaded_matrices,use_combined=use_combined)
            next!(p)
        end
    else
        res[1]=solve_INFO(solver,new_basis,pts,ks[1],multithreaded=multithreaded_matrices)
        @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[2:end]
            res[i]=solve(solver,new_basis,pts,ks[i],multithreaded=multithreaded_matrices)
            next!(p)
        end
    end
    return res
end

"""
    refine_minima(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::AbstractVector{T},tens::AbstractVector{T};kernel_fun::Union{Symbol, Function}=:default,multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_combined::Bool= false) where {T<:Real}

Given a coarse sampling of wavenumbers `ks` and their associated spectral indicators `tens` (e.g. singular‐value magnitudes), this function locates and refines the local minima of `log10(abs.(tens))` by a 1D optimization on each interval.

# Arguments
- `solver::SweepSolver`: A boundary‐integral solver implementing `evaluate_points(solver, billiard, k)` → sample geometry at wavenumber `k`, and `solve(solver, basis, pts, k; ...)` → compute the spectrum.
- `basis::AbsBasis`: The basis type passed to `solve`, e.g. `AbstractHankelBasis()`.
- `billiard::AbsBilliard`: The geometric domain description used by `evaluate_points`.
- `ks::AbstractVector{T}`: Coarse wavenumber grid (`T<:Real`).
- `tens::AbstractVector{T}`: Corresponding spectral measures (e.g. smallest singular values) at each `ks[i]`.

# Keyword Arguments
- `kernel_fun::Union{Symbol, Function} = :default`: If supported, which integral‐kernel variant to use.
- `multithreaded_matrices::Bool = true`: Whether to build boundary‐integral matrices in parallel.
- `multithreaded_ks::Bool = false`: Whether to refine each bracketed minimum in parallel.
- `use_combined::Bool = false`: For CFIE solvers, whether to use the combined‐field formulation.
- `threshold::Float64=200`: Minimum 2nd derivative value at k to count as approximate eigenvalue.
- `print_refinement::Bool = true`: Whether to print the refinement progress for each k in ks_approx.

# Returns
Tuple `(sols, tens_refined)` where
- `sols::Vector{T}`: The refined minimizer wavenumbers.
- `tens_refined::Vector{T}`: The corresponding objective values (i.e. minimum of `solve(...)`).
"""
function refine_minima(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::AbstractVector{T},tens::AbstractVector{T};kernel_fun::Union{Symbol,Function}=:default,multithreaded_matrices::Bool=true,multithreaded_ks=false,use_combined::Bool=false,threshold=200.0,print_refinement=true) where {T<:Real}
    N=length(tens)
    @assert N==length(ks)
    ks_approx=get_eigenvalues(ks,abs.(tens);threshold=threshold)
    f_min=nothing
    if solver isa BoundaryIntegralMethod
        f_min=x->begin # x=(k,b)
            solver_new=update_field(solver,:pts_scaling_factor,[x[2]])
            pts=evaluate_points(solver_new,billiard,x[1])
            solve(solver_new,basis,pts,x[1];multithreaded=multithreaded_matrices,kernel_fun=kernel_fun)
        end
    elseif solver isa CFIE_polar_nocorners || solver isa CFIE_polar_corner_correction
        f_min=x->begin # x=(k,b)
            solver_new=update_field(solver,:pts_scaling_factor,[x[2]])
            pts=evaluate_points(solver,billiard,x[1])
            solve(solver_new,basis,pts,x[1];multithreaded=multithreaded_matrices,use_combined=use_combined)
        end
    else
        f_min=x->begin # x=(k,b)
            solver_new=update_field(solver,:pts_scaling_factor,[x[2]])
            solver_new=update_field(solver_new,:dim_scaling_factor,x[2])
            dim=max(solver.min_dim,round(Int,billiard.length*x[1]*solver_new.dim_scaling_factor/(2*pi)))
            new_basis=resize_basis(basis,billiard,dim,x[1])
            pts=evaluate_points(solver_new,billiard,x[1])
            solve(solver_new,new_basis,pts,x[1];multithreaded=multithreaded_matrices)
        end
    end
    sols=similar(ks_approx)
    tens_refined=similar(ks_approx)
    dk=(ks[2]-ks[1])
    p=Progress(N;desc="Refining approximate ks...")
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks_approx) 
        #res=optimize(f_min,ks_approx[i]-dk,ks_approx[i]+dk) # the old 1d optimization with simple refining
        res=optimize(f_min,[ks_approx[i],solver.pts_scaling_factor[1]],NelderMead()) # must be gradient free due to Optim's Dual type usage
        k0,t0=res.minimizer,res.minimum
        sols[i]=k0
        tens_refined[i]=t0
        next!(p)
    end
    if print_refinement
        println("\n===== Refinement summary =====")
        println(rpad(" #",4),rpad("k_approx",12),rpad("k_ref",12),rpad("Δk",12),rpad("t_approx",12),rpad("t_ref",12),"Δt")
        for i in eachindex(sols)
            k_app=ks_approx[i]
            k_ref=sols[i]
            t_app=log10(abs(f_min(ks_approx[i])))       
            t_ref=log10(abs(tens_refined[i]))
            dk=k_ref-k_app
            dt=t_ref-t_app
            println(rpad("$(i)",4),rpad("$(round(k_app,digits=6))",12),rpad("$(round(k_ref,digits=6))",12),rpad("$(round(dk,digits=6))",12),rpad("$(round(t_app,digits=6))",12),rpad("$(round(t_ref,digits=6))",12),"$(round(dt,digits=6))")
        end
        println("================================\n")
    end
    return sols,tens_refined 
end