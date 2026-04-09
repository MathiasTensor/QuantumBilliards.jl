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

function _k_sweep(solver::BoundaryIntegralMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    if which==:det
        res=zeros(eltype(Complex{eltype(ks)}),length(ks))
    else
        res=similar(ks)
    end
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    next!(p)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        is_calculating=true
        while is_calculating
            try
                res[i]=solve(solver,new_basis,pts,ks[i];multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
end

function _k_sweep(solver::CFIE_kress,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    if which==:det
        res=zeros(eltype(Complex{eltype(ks)}),length(ks))
    else
        res=similar(ks)
    end
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    next!(p)
    Rmat=build_Rmat_kress(solver,pts)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        is_calculating=true 
        while is_calculating
            try
                res[i]=solve(solver,new_basis,pts,ks[i],Rmat;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
end

function _k_sweep(solver::CFIE_kress_corners,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    if which==:det
        res=zeros(eltype(Complex{eltype(ks)}),length(ks))
    else
        res=similar(ks)
    end
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    next!(p)
    Rmat=build_Rmat_kress(solver,pts)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        is_calculating=true 
        while is_calculating
            try
                res[i]=solve(solver,new_basis,pts,ks[i],Rmat;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
end

function _k_sweep(solver::CFIE_alpert,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    new_basis=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    ws=build_cfie_alpert_workspace(solver,pts)
    if which==:det
        res=zeros(eltype(Complex{eltype(ks)}),length(ks))
    else
        res=similar(ks)
    end
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_INFO(solver,new_basis,pts,ks[end];multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    next!(p)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)[1:end-1]
        is_calculating=true 
        while is_calculating
            try
                res[i]=solve(solver,new_basis,pts,ws,ks[i];multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
end

function _k_sweep(solver::ParticularSolutionsMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
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

function _k_sweep(solver::DecompositionMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
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

"""
        k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10)

High-level API for performing a sweep over wavenumbers `ks` to compute tensions. The function dispatches to the appropriate internal `_k_sweep` method based on the type of `solver`. It supports multithreading for both matrix construction and wavenumber sweeps, and allows for the use of Krylov solvers where applicable.

# Inputs
- `solver::SweepSolver`: The solver configuration for performing the sweep.
- `basis::AbsBasis`: The basis to be used. Can use also the AbstractHankelBasis() when solver is `BoundaryIntegralMethod`.
- `billiard::AbsBilliard`: The billiard configuration.
- `ks::AbstractVector{T}`: Vector of wavenumbers to sweep over.
- `multithreaded_matrices::Bool=true`: Whether to use multithreading for matrix construction.
- `multithreaded_ks::Bool=false`: Whether to use multithreading for the wavenumber sweep.
- `use_krylov::Bool=true`: Whether to use Krylov solvers where applicable.
- `tol::Real=1e-10`: Tolerance for convergence in appropriate solvers (`ParticularSolutionsMethod`).
- `which::Symbol=:det_argmin`: Whether to compute the determinant (`:det`) or the smallest singular value (`:svd`) during refinement. Also there is option :det_argmin which can be used for finding minima.
"""
function k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    return _k_sweep(solver,basis,billiard,ks;multithreaded_matrices=multithreaded_matrices,multithreaded_ks=multithreaded_ks,use_krylov=use_krylov,tol=tol,which=which)
end

############ REFINEMENT ############

# Helper function to scale a field of the solver if it exists, otherwise return the solver unchanged. This basically disambigues PSM and DM from BIE type solvers since the latter dont have the associated basis to scale.
function _try_scaling_field(obj,field::Symbol,factor)
    try
        return update_field(obj,field,factor*getfield(obj,field))
    catch _
        return obj
    end
end

function _refined_solver(solver::SweepSolver,pts_factor,dim_factor)
    s=_try_scaling_field(solver,:pts_scaling_factor,pts_factor)
    s=_try_scaling_field(s,:dim_scaling_factor,dim_factor)
    return s
end

function parabolic_refine(f::Function,k0;h,maxiter::Int=6,tol::Float64=1e-13)
    k=clamp(k0,a,b)
    hk=h
    for _ in 1:maxiter
        km=max(a,k-hk)
        kp=min(b,k+hk)
        km==k && return k
        kp==k && return k
        fm=f(km)
        f0=f(k)
        fp=f(kp)
        denom=fm-2*f0+fp
        abs(denom)<1e-16 && return k
        δ=0.5*(kp-km)*(fm-fp)/(2*denom)
        knew=k+δ
        if !(a<=knew<=b)
            return k
        end
        abs(knew-k)<tol && return knew
        k=knew
        hk*=0.5
    end
    return k
end

function refine_minima(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::AbstractVector{T},tens::AbstractVector{T};multithreaded_matrices::Bool=true,threshold=200.0,print_refinement::Bool=true,use_krylov::Bool=true,digits::Int=10,which::Symbol=:svd,pts_refinement_factors=(1.0,1.5,2.0,3.0,4.0),dim_refinement_factors=(1.0,1.1,1.25,1.4,1.5),window_shrink=3.0,final_window_factor=1e-3,optimizer_kwargs=NamedTuple(),stop_k_tol=0.0,stop_t_tol=0.0,initial_refinement_interval=1e-3,final_parabolic_polish::Bool=true,polish_window_fraction=0.1,polish_maxiter::Int=6,polish_tol::Float64=1e-13) where {T<:Real}
    N=length(tens)
    @assert N==length(ks)
    @assert length(pts_refinement_factors)==length(dim_refinement_factors)
    ks_approx= length(ks)==1 ? collect(ks) : get_eigenvalues(collect(ks),abs.(tens);threshold=threshold)
    isempty(ks_approx) && return T[],T[],Vector{Vector{NamedTuple}}()
    nk=length(ks_approx)
    sols=similar(ks_approx)
    tens_refined=similar(ks_approx)
    histories=Vector{Vector{NamedTuple}}(undef,nk)
    dk0= (N>=2) ? abs(ks[2]-ks[1]) : T(initial_refinement_interval)
    p = Progress(nk;desc="Refining minima (optimize)...")
    for i in eachindex(ks_approx)
        kcur=ks_approx[i]
        window=dk0
        hist=NamedTuple[]
        kprev=T(NaN)
        tprev=T(NaN)
        for lev in eachindex(pts_refinement_factors)
            pf=pts_refinement_factors[lev]
            df=dim_refinement_factors[lev]
            solver_cur=_refined_solver(solver,pf,df)
            pts=evaluate_points(solver_cur,billiard,kcur)
            fcur= k->solve(solver_cur,basis,pts,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
            a=kcur-window
            b=kcur+window
            res=isempty(optimizer_kwargs) ? optimize(fcur,a,b) : optimize(fcur,a,b;optimizer_kwargs...)
            knew=res.minimizer
            tnew=res.minimum
            if lev==length(pts_refinement_factors) && final_parabolic_polish
                aw=max(window*T(polish_window_fraction),dk0*final_window_factor)
                a2=knew-aw
                b2=knew+aw
                res2=isempty(optimizer_kwargs) ? optimize(fcur,a2,b2) : optimize(fcur,a2,b2;optimizer_kwargs...)
                knew=res2.minimizer
                tnew=res2.minimum
            end
            push!(hist,(level=lev,pts_factor=pf,dim_factor=df,k=knew,tension=tnew,window=window))
            if lev>1
                kconv= (stop_k_tol>0) && (abs(knew-kprev)<=stop_k_tol)
                tconv= (stop_t_tol>0) && (abs(tnew-tprev)<=stop_t_tol)
                if kconv || tconv
                    kcur=knew
                    tprev=tnew
                    break
                end
            end
            kprev=kcur
            tprev=tnew
            kcur=knew
            window=max(window/window_shrink,dk0*final_window_factor)
        end
        sols[i]=kcur
        tens_refined[i]=tprev
        histories[i]=hist
        next!(p)
    end
    if print_refinement
        println("\n================ refinement summary ================")
        println(rpad("#",4),
                rpad("k_approx",digits+8),
                rpad("k_ref",digits+8),
                rpad("Δk",digits+8),
                rpad("log10|t_app|",digits+10),
                rpad("log10|t_ref|",digits+10),
                "levels")
        for i in eachindex(sols)
            k_app=ks_approx[i]
            k_ref=sols[i]
            t_app=log10(abs(tens[i]))
            t_ref=log10(abs(tens_refined[i]))
            dk_i=k_ref-k_app
            println(rpad("$(i)",4),
                    rpad("$(round(k_app,digits=digits))",digits+8),
                    rpad("$(round(k_ref,digits=digits))",digits+8),
                    rpad("$(round(dk_i,digits=digits))",digits+8),
                    rpad("$(round(t_app,digits=digits))",digits+10),
                    rpad("$(round(t_ref,digits=digits))",digits+10),
                    "$(length(histories[i]))")
        end
        println("===================================================\n")
    end
    sols,tens_refined,histories
end