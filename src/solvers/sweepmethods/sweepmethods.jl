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

function _sweep_dim(solver::SweepSolver,billiard::AbsBilliard,ks)
    kmax=maximum(ks)
    return max(solver.min_dim,round(Int,billiard.length*kmax*solver.dim_scaling_factor/(2*pi)))
end

function _k_sweep(solver::BoundaryIntegralMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_combined::Bool=false,use_krylov::Bool=true,tol=1e-10)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    res=similar(ks)
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices,use_krylov=use_krylov)
    next!(p)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        is_calculating=true
        while is_calculating
            try
                res[i]=solve(solver,new_basis,pts,ks[i];multithreaded=multithreaded_matrices,use_krylov=use_krylov)
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
end

function _k_sweep(solver::CFIE_polar_nocorners,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_combined::Bool=false,use_krylov::Bool=true,tol=1e-10)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    res=similar(ks)
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices)
    next!(p)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        is_calculating=true 
        while is_calculating
            try
                res[i]=solve(solver,new_basis,pts,ks[i];multithreaded=multithreaded_matrices)
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
end

function _k_sweep(solver::ParticularSolutionsMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_combined::Bool=false,use_krylov::Bool=true,tol=1e-10)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    res=similar(ks)
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices,tol=tol)
    next!(p)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        pts_i=evaluate_points(solver,billiard,ks[i])
        basis_i=resize_basis(basis,billiard,dim,ks[i])
        try
            res[i]=solve(solver,basis_i,pts_i,ks[i];multithreaded=multithreaded_matrices,tol=tol)
        catch e
            @warn "Error in k_sweep for k=$(ks[i]): $(e), skipping this k"
            continue
        end
        next!(p)
    end
    return res
end

function _k_sweep(solver::DecompositionMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    res=similar(ks)
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices)
    next!(p)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        is_calculating=true
        pts_i=evaluate_points(solver,billiard,ks[i])
        basis_i=resize_basis(basis,billiard,dim,ks[i])
        while is_calculating
            try
                res[i]=solve(solver,basis_i,pts_i,ks[i];multithreaded=multithreaded_matrices)
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
end

function k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_combined::Bool=false,use_krylov::Bool=true,tol=1e-10)
    return _k_sweep(solver,basis,billiard,ks;multithreaded_matrices=multithreaded_matrices,multithreaded_ks=multithreaded_ks,use_combined=use_combined,use_krylov=use_krylov,tol=tol)
end

############ REFINEMENT ############

# For BIM/CFIE, refinement is controlled by boundary discretization (pts_scaling_factor).
# dim_scaling_factor is only relevant for basis-type solvers and is ignored by some dispatches below.
function _refine_objective(solver::BoundaryIntegralMethod,basis::AbsBasis,billiard::AbsBilliard;multithreaded_matrices::Bool=true)
    return k->begin
        pts=evaluate_points(solver,billiard,k)
        solve(solver,basis,pts,k;multithreaded=multithreaded_matrices)
    end
end

function _refine_objective(solver::CFIE_polar_nocorners,basis::AbsBasis,billiard::AbsBilliard;multithreaded_matrices::Bool=true)
    return k->begin
        pts=evaluate_points(solver,billiard,k)
        solve(solver,basis,pts,k;multithreaded=multithreaded_matrices)
    end
end

function _refine_objective(solver::DecompositionMethod,basis::AbsBasis,billiard::AbsBilliard;multithreaded_matrices::Bool=true)
    return k->begin
        dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
        new_basis=resize_basis(basis,billiard,dim,k)
        pts=evaluate_points(solver,billiard,k)
        solve(solver,new_basis,pts,k;multithreaded=multithreaded_matrices)
    end
end

function _refine_objective(solver::ParticularSolutionsMethod,basis::AbsBasis,billiard::AbsBilliard;multithreaded_matrices::Bool=true)
    return k->begin
        dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
        new_basis=resize_basis(basis,billiard,dim,k)
        pts=evaluate_points(solver,billiard,k)
        solve(solver,new_basis,pts,k;multithreaded=multithreaded_matrices)
    end
end

function refine_minima(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::AbstractVector{T},tens::AbstractVector{T};multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,threshold=200.0,print_refinement::Bool=true) where {T<:Real}
    N=length(tens)
    @assert N==length(ks)
    ks_approx=get_eigenvalues(ks,abs.(tens);threshold=threshold)
    solver_new=update_field(solver,:pts_scaling_factor,2*solver.pts_scaling_factor)
    try
        solver_new=update_field(solver_new,:dim_scaling_factor,1.5*solver.dim_scaling_factor)
    catch _
    end
    f_min=_refine_objective(solver_new,basis,billiard;multithreaded_matrices=multithreaded_matrices)
    sols=similar(ks_approx)
    tens_refined=similar(ks_approx)
    dk=ks[2]-ks[1]
    p=Progress(length(ks_approx);desc="Refining approximate ks...")
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks_approx)
        res=optimize(f_min,ks_approx[i]-dk,ks_approx[i]+dk)
        sols[i]=res.minimizer
        tens_refined[i]=res.minimum
        next!(p)
    end
    if print_refinement
        println("\n===== Refinement summary =====")
        println(rpad(" #",4),rpad("k_approx",12),rpad("k_ref",12),rpad("Δk",12),rpad("t_approx",12),rpad("t_ref",12),"Δt")
        for i in eachindex(sols)
            k_app=ks_approx[i]
            k_ref=sols[i]
            t_app=log10(abs(f_min(k_app)))
            t_ref=log10(abs(tens_refined[i]))
            dk_i=k_ref-k_app
            dt=t_ref-t_app
            println(rpad("$(i)",4),
                    rpad("$(round(k_app,digits=6))",12),
                    rpad("$(round(k_ref,digits=6))",12),
                    rpad("$(round(dk_i,digits=6))",12),
                    rpad("$(round(t_app,digits=6))",12),
                    rpad("$(round(t_ref,digits=6))",12),
                    "$(round(dt,digits=6))")
        end
        println("================================\n")
    end
    return sols,tens_refined
end