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

function _k_sweep_prepare(solver::BoundaryIntegralMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    pts=evaluate_points(solver,billiard,kmax)
    Ntot=length(pts.xy)
    T=eltype(pts.ds)
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    solve_first=k->solve_INFO(solver,basis,pts,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    solve_one=k->solve(solver,basis,A,pts,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    return solve_first,solve_one
end
function _k_sweep_prepare(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    pts=evaluate_points(solver,billiard,kmax)
    ws=build_cfie_kress_workspace(solver,pts)
    T=eltype(first(pts).ws)
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    solve_first=k->solve_INFO(solver,basis,pts,ws,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    solve_one=k->solve(solver,basis,A,pts,ws,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    return solve_first,solve_one
end
function _k_sweep_prepare(solver::CFIE_alpert,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    pts=evaluate_points(solver,billiard,kmax)
    ws=build_cfie_alpert_workspace(solver,pts)
    T=eltype(first(pts).ws)
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    solve_first=k->solve_INFO(solver,basis,pts,ws,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    solve_one=k->solve(solver,basis,A,pts,ws,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
    return solve_first,solve_one
end
function _k_sweep_prepare(solver::ParticularSolutionsMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    basis_max=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    solve_first(k)=solve_INFO(solver,basis_max,pts,k;multithreaded=multithreaded_matrices,tol=tol)
    solve_one(k)=solve(solver,basis_max,pts,k;multithreaded=multithreaded_matrices,tol=tol)
    return solve_first,solve_one
end
function _k_sweep_prepare(solver::DecompositionMethod,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    dim=_sweep_dim(solver,billiard,ks)
    basis_max=resize_basis(basis,billiard,dim,kmax)
    pts=evaluate_points(solver,billiard,kmax)
    solve_first(k)=solve_INFO(solver,basis_max,pts,k;multithreaded=multithreaded_matrices)
    solve_one(k)=solve(solver,basis_max,pts,k;multithreaded=multithreaded_matrices)
    return solve_first,solve_one
end
@inline function _k_sweep_result_container(ks;which::Symbol=:det_argmin)
    which==:det ? zeros(eltype(Complex{eltype(ks)}),length(ks)) : similar(ks)
end

"""
        k_sweep(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10)

High-level API for performing a sweep over wavenumbers `ks` to compute tensions. The function dispatches to the appropriate internal `_k_sweep` method based on the type of `solver`. It supports multithreading for both matrix construction and wavenumber sweeps, and allows for the use of Krylov solvers where applicable.

# Inputs
- `solver::SweepSolver`: The solver configuration for performing the sweep.
- `basis::AbsBasis`: The basis to be used. Can use also the AbstractHankelBasis() when solver is `BoundaryIntegralMethod`.
- `billiard::AbsBilliard`: The billiard configuration.
- `ks::AbstractVector{T}`: Vector of wavenumbers to sweep over.
- `multithreaded_matrices::Bool=true`: Whether to use multithreading for matrix construction.
- `use_krylov::Bool=true`: Whether to use Krylov solvers where applicable.
- `tol::Real=1e-10`: Tolerance for convergence in appropriate solvers (`ParticularSolutionsMethod`).
- `which::Symbol=:svd`: Whether to compute the determinant (`:det`) or the smallest singular value (`:svd`) during refinement. Also there is option :det_argmin which can be used for finding minima.
"""
function k_sweep(solver,basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    solve_first,solve_one=_k_sweep_prepare(solver,basis,billiard,ks;multithreaded_matrices=multithreaded_matrices,use_krylov=use_krylov,tol=tol,which=which)
    res=_k_sweep_result_container(ks;which=which)
    println("$(nameof(typeof(solver))) sweep...")
    p=Progress(length(ks),1)
    res[end]=solve_first(ks[end])
    next!(p)
    for i in eachindex(ks)[1:end-1]
        is_calculating=true
        while is_calculating
            try
                res[i]=solve_one(ks[i])
                is_calculating=false
            catch e
                @warn "Error in k_sweep for k=$(ks[i]): $(e), retrying..."
            end
        end
        next!(p)
    end
    return res
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

#=

function newton_refine(f::Function,k0;a=0.0,b=Inf,h=1e-6,maxiter=8,tol=1e-12)
    k=clamp(k0,a,b)
    for _ in 1:maxiter
        hp=min(h,0.5*(b-k))
        hm=min(h,0.5*(k-a))
        if hm<=0 || hp<=0
            return k
        end
        f0=f(k)
        fp=f(k+hp)
        fm=f(k-hm)
        f1=(fp-fm)/(hp+hm)
        f2=2*((fp-f0)/hp-(f0-fm)/hm)/(hp+hm)
        if !isfinite(f1) || !isfinite(f2) || abs(f2)<1e-14
            return k
        end
        k_new=k-f1/f2
        k_new=clamp(k_new,a,b)
        if abs(k_new-k)<tol
            return k_new
        end
        k=k_new
    end
    return k
end

function refine_minima(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::AbstractVector{T},tens::AbstractVector{T};multithreaded_matrices::Bool=true,threshold=200.0,print_refinement::Bool=true,use_krylov::Bool=true,digits::Int=10,which::Symbol=:svd,pts_refinement_factors=(1.0,1.5,2.0,3.0,4.0),dim_refinement_factors=(1.0,1.1,1.25,1.4,1.5),window_shrink=3.0,final_window_factor=1e-3,optimizer_kwargs=NamedTuple(),stop_k_tol=0.0,stop_t_tol=0.0,initial_refinement_interval=1e-3,show_progress::Bool=false) where {T<:Real}
    N=length(tens)
    @assert N==length(ks)
    @assert length(pts_refinement_factors)==length(dim_refinement_factors)
    ks_approx=length(ks)==1 ? collect(ks) : get_eigenvalues(collect(ks),abs.(tens);threshold=threshold)
    if isempty(ks_approx)
        return T[],T[],Vector{Vector{NamedTuple}}()
    end
    nk=length(ks_approx)
    sols=similar(ks_approx)
    tens_refined=similar(ks_approx)
    histories=Vector{Vector{NamedTuple}}(undef,nk)
    show_progress && (p=Progress(nk;desc="Refining minima"))
    for i in eachindex(ks_approx)
        kcur=ks_approx[i]
        dk_grid= length(ks)>=2 ? abs(ks[mod(i+1,length(ks))]-ks[i]) : T(initial_refinement_interval)
        dk0=max(3*dk_grid,T(initial_refinement_interval))
        window=dk0
        hist=NamedTuple[]
        tprev=T(NaN)
        kprev=T(NaN)
        for lev in eachindex(pts_refinement_factors)
            pf=pts_refinement_factors[lev]
            df=dim_refinement_factors[lev]
            solver_cur=_refined_solver(solver,pf,df)
            pts=evaluate_points(solver_cur,billiard,kcur)
            fcur=k->solve(solver_cur,basis,pts,k;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
            a=kcur-window
            b=kcur+window
            res=isempty(optimizer_kwargs) ? optimize(fcur,a,b) : optimize(fcur,a,b;optimizer_kwargs...)
            knew=res.minimizer
            tnew=res.minimum
            push!(hist,(level=lev,pts_factor=pf,dim_factor=df,k=knew,tension=tnew,window=window))
            if lev>1
                kconv=(stop_k_tol>0)&&(abs(knew-kprev)<=stop_k_tol)
                tconv=(stop_t_tol>0)&&(abs(tnew-tprev)<=stop_t_tol)
                if kconv||tconv
                    kcur=knew
                    tprev=tnew
                    break
                end
            end
            kprev=kcur
            tprev=tnew
            kcur=knew
        end
        sols[i]=kcur
        tens_refined[i]=tprev
        histories[i]=hist
        show_progress && next!(p)
    end
    if print_refinement
        println("\n================ Newton refinement summary ================")
        println(rpad("#",4),rpad("k_approx",digits+8),
                rpad("k_ref",digits+8),rpad("Δk",digits+8),
                rpad("log10|t_app|",digits+10),
                rpad("log10|t_ref|",digits+10),"levels")
        for i in eachindex(sols)
            k_app=ks_approx[i]
            k_ref=sols[i]
            idx_app=argmin(abs.(ks.-k_app))
            t_app=log10(abs(tens[idx_app]))
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
        println("==========================================================\n")
    end
    return sols,tens_refined,histories
end

=#

function newton_refine_svd(solver::EBIMSolver,basis::AbsBasis,billiard::AbsBilliard,k0;
    a=0.0,b=Inf,maxiter=8,tol=1e-12,multithreaded_matrices=true)

    k=clamp(k0,a,b)
    λbest=Inf
    tbest=Inf

    @showprogress for _ in 1:maxiter
        pts=evaluate_points(solver,billiard,k)
        N=boundary_matrix_size(pts)
        T=typeof(k)
        A=Matrix{Complex{T}}(undef,N,N)
        dA=Matrix{Complex{T}}(undef,N,N)
        ddA=Matrix{Complex{T}}(undef,N,N)

        construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded_matrices)

        H=Hermitian(dA' * A + A' * dA)
        G=Hermitian(A' * A)
        H2=Hermitian(ddA' * A + 2*(dA' * dA) + A' * ddA)

        eg=eigen(G)
        vals=eg.values
        vecs=eg.vectors
        idx=argmin(vals)
        λ=vals[idx]
        x=@view vecs[:,idx]

        gx=H * x
        λ1=real(dot(x,gx))

        λ2=real(dot(x,H2 * x))
        @inbounds for j in eachindex(vals)
            j==idx && continue
            denom=λ - vals[j]
            abs(denom)<eps(real(T)) && continue
            c=dot(view(vecs,:,j),gx)
            λ2 += 2*abs2(c)/denom
        end

        if isfinite(λ) && λ<λbest
            λbest=λ
            tbest=sqrt(max(zero(T),λ))
        end
        if !isfinite(λ1) || !isfinite(λ2) || abs(λ2)<sqrt(eps(real(T)))
            break
        end

        knew=clamp(k - λ1/λ2,a,b)
        if !isfinite(knew) || abs(knew-k)<tol
            k=knew
            break
        end
        k=knew
    end

    pts=evaluate_points(solver,billiard,k)
    N=boundary_matrix_size(pts)
    T=typeof(k)
    A=Matrix{Complex{T}}(undef,N,N)
    dA=Matrix{Complex{T}}(undef,N,N)
    ddA=Matrix{Complex{T}}(undef,N,N)
    construct_matrices!(solver,basis,A,dA,ddA,pts,k;multithreaded=multithreaded_matrices)
    G=Hermitian(A' * A)
    λ=eigmin(G)
    return k,sqrt(max(zero(T),λ))
end

function refine_minima(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,
    ks::AbstractVector{T},tens::AbstractVector{T};
    multithreaded_matrices::Bool=true,
    threshold=200.0,
    print_refinement::Bool=true,
    use_krylov::Bool=true,
    digits::Int=10,
    which::Symbol=:svd,
    pts_refinement_factors=(1.0,1.5,2.0,3.0,4.0),
    dim_refinement_factors=(1.0,1.1,1.25,1.4,1.5),
    window_shrink=3.0,
    optimizer_kwargs=NamedTuple(),
    stop_k_tol=0.0,
    stop_t_tol=0.0,
    initial_refinement_interval=1e-3,
    show_progress::Bool=false) where {T<:Real}

    N=length(ks)
    @assert N==length(tens)
    @assert length(pts_refinement_factors)==length(dim_refinement_factors)

    ks_approx=length(ks)==1 ? collect(ks) : get_eigenvalues(collect(ks),abs.(tens);threshold=threshold)
    isempty(ks_approx) && return T[],T[],Vector{Vector{NamedTuple}}()

    sols=similar(ks_approx)
    tens_refined=similar(ks_approx)
    histories=Vector{Vector{NamedTuple}}(undef,length(ks_approx))
    show_progress && (p=Progress(length(ks_approx);desc="Refining minima"))

    for i in eachindex(ks_approx)
        kcur=ks_approx[i]
        idx0=argmin(abs.(ks .- kcur))
        dk_grid=length(ks)>1 ? minimum(abs.(ks .- ks[idx0])[abs.(ks .- ks[idx0]).>zero(T)]) : T(initial_refinement_interval)
        window=max(3dk_grid,T(initial_refinement_interval))

        hist=NamedTuple[]
        kprev=T(NaN)
        tprev=T(NaN)

        for lev in eachindex(pts_refinement_factors)
            pf=pts_refinement_factors[lev]
            df=dim_refinement_factors[lev]

            solver_cur=solver
            try
                solver_cur=update_field(solver_cur,:pts_scaling_factor,pf*getfield(solver_cur,:pts_scaling_factor))
            catch
            end
            try
                solver_cur=update_field(solver_cur,:dim_scaling_factor,df*getfield(solver_cur,:dim_scaling_factor))
            catch
            end

            a=kcur-window
            b=kcur+window

            if solver_cur isa EBIMSolver
                knew,tnew=newton_refine_svd(solver_cur,basis,billiard,kcur;
                    a=a,b=b,maxiter=8,tol=1e-12,
                    multithreaded_matrices=multithreaded_matrices)
            else
                dim=max(solver_cur.min_dim,round(Int,billiard.length*kcur*solver_cur.dim_scaling_factor/(2*pi)))
                basis_cur=resize_basis(basis,billiard,dim,kcur)
                pts=evaluate_points(solver_cur,billiard,kcur)
                f=k->solve(solver_cur,basis_cur,pts,k;
                    multithreaded=multithreaded_matrices,
                    use_krylov=use_krylov,
                    which=which)
                res=isempty(optimizer_kwargs) ? optimize(f,a,b) : optimize(f,a,b;optimizer_kwargs...)
                knew=res.minimizer
                tnew=res.minimum
            end

            push!(hist,(level=lev,pts_factor=pf,dim_factor=df,k=knew,tension=tnew,window=window))

            if lev>1
                kconv=(stop_k_tol>0)&&(abs(knew-kprev)<=stop_k_tol)
                tconv=(stop_t_tol>0)&&(abs(tnew-tprev)<=stop_t_tol)
                if kconv || tconv
                    kcur=knew
                    tprev=tnew
                    break
                end
            end

            kprev=kcur
            tprev=tnew
            kcur=knew
            window/=window_shrink
        end

        sols[i]=kcur
        tens_refined[i]=tprev
        histories[i]=hist
        show_progress && next!(p)
    end

    if print_refinement
        println("\n================ Newton refinement summary ================")
        println(rpad("#",4),rpad("k_approx",digits+8),rpad("k_ref",digits+8),
                rpad("Δk",digits+8),rpad("log10|t_app|",digits+10),
                rpad("log10|t_ref|",digits+10),"levels")
        for i in eachindex(sols)
            k_app=ks_approx[i]
            k_ref=sols[i]
            idx_app=argmin(abs.(ks .- k_app))
            t_app=log10(abs(tens[idx_app]))
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
        println("==========================================================\n")
    end

    return sols,tens_refined,histories
end