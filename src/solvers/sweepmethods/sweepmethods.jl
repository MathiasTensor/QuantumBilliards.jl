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
function _k_sweep_prepare(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbsBasis,billiard::AbsBilliard,ks;multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-10,which::Symbol=:det_argmin)
    kmax=maximum(ks)
    pts=evaluate_points(solver,billiard,kmax)
    ws=build_dlp_kress_workspace(solver,pts)
    T=eltype(pts.ws)
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
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
    _second_derivative_sweep_spectrum(x::Vector{T}, y::Vector{T}) where {T<:Real}

Computes the second derivative of `y` with respect to `x` using finite differences between the xs in `x`.

# Arguments
- `x::Vector{T}`: The x-coordinates of the data.
- `y::Vector{T}`: The y-values of the data.

# Returns
- `Vector{T}`: Midpoints of the x-values for the second derivative.
- `Vector{T}`: The second derivative of `y` with respect to `x`.
"""
function _second_derivative_sweep_spectrum(x::Vector{T}, y::Vector{T}) where {T<:Real}
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
    mid_x,gradient=_second_derivative_sweep_spectrum(k_range,log10.(tens))
    return find_peaks(mid_x,gradient;threshold=threshold)
end

############ REFINEMENT ############

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

@inline _matrix_size(ws::CFIEKressWorkspace)=ws.Ntot
@inline _matrix_size(ws::CFIEAlpertWorkspace)=ws.Ntot
@inline _matrix_size(ws::DLPKressWorkspace)=ws.N

function _newton_buffers(::Type{T},N::Int) where {T<:Real}
    A=Matrix{Complex{T}}(undef,N,N)
    dA=Matrix{Complex{T}}(undef,N,N)
    ddA=Matrix{Complex{T}}(undef,N,N)
    G=Matrix{Complex{T}}(undef,N,N)
    H=Matrix{Complex{T}}(undef,N,N)
    H2=Matrix{Complex{T}}(undef,N,N)
    W=Matrix{Complex{T}}(undef,N,N)
    return A,dA,ddA,G,H,H2,W
end

function _prepare_newton_refinement(solver::BoundaryIntegralMethod,basis::AbsBasis,billiard::AbsBilliard,k::T;multithreaded_matrices::Bool=true) where {T<:Real}
    pts=evaluate_points(solver,billiard,k)
    N=boundary_matrix_size(pts)
    A,dA,ddA,G,H,H2,W=_newton_buffers(T,N)
    assemble=kk->construct_matrices!(solver,basis,A,dA,ddA,pts,kk;multithreaded=multithreaded_matrices)
    return assemble,A,dA,ddA,G,H,H2,W
end

function _prepare_newton_refinement(solver::Union{CFIE_alpert,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::AbsBasis,billiard::AbsBilliard,k::T;multithreaded_matrices::Bool=true) where {T<:Real}
    pts=evaluate_points(solver,billiard,k)
    ws=solver isa CFIE_alpert ? build_cfie_alpert_workspace(solver,pts) : build_cfie_kress_workspace(solver,pts)
    N=_matrix_size(ws)
    A,dA,ddA,G,H,H2,W=_newton_buffers(T,N)
    assemble=kk->construct_matrices!(solver,basis,A,dA,ddA,pts,ws,kk;multithreaded=multithreaded_matrices)
    return assemble,A,dA,ddA,G,H,H2,W
end

function _prepare_newton_refinement(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbsBasis,billiard::AbsBilliard,k::T;multithreaded_matrices::Bool=true) where {T<:Real}
    pts=evaluate_points(solver,billiard,k)
    ws=build_dlp_kress_workspace(solver,pts)
    N=_matrix_size(ws)
    A,dA,ddA,G,H,H2,W=_newton_buffers(T,N)
    assemble=kk->construct_matrices!(solver,basis,A,dA,ddA,pts,ws,kk;multithreaded=multithreaded_matrices)
    return assemble,A,dA,ddA,G,H,H2,W
end

function newton_refine_svd!(assemble,A,dA,ddA,G,H,H2,W,k0;a=0.0,b=Inf,maxiter=8,tol=1e-12)
    T=typeof(k0)
    k=clamp(k0,a,b)
    kbest=k
    λbest=T(Inf)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        for _ in 1:maxiter
            assemble(k)
            mul!(G,adjoint(A),A)
            mul!(H,adjoint(dA),A)
            mul!(W,adjoint(A),dA)
            @. H=H+W
            mul!(H2,adjoint(ddA),A)
            mul!(W,adjoint(dA),dA)
            @. H2=H2+2*W
            mul!(W,adjoint(A),ddA)
            @. H2=H2+W
            F=eigen!(Hermitian(G))
            vals=F.values
            vecs=F.vectors
            idx=argmin(vals)
            λ=vals[idx]
            x=@view vecs[:,idx]
            if isfinite(λ) && λ<λbest
                λbest=λ;kbest=k
            end
            gx=H*x
            λ1=real(dot(x,gx))
            λ2=real(dot(x,H2*x))
            @inbounds for j in eachindex(vals)
                j==idx && continue
                denom=λ-vals[j]
                abs(denom)<eps(T) && continue
                c=dot(view(vecs,:,j),gx)
                λ2 += 2*abs2(c)/denom
            end
            if !isfinite(λ1) || !isfinite(λ2) || abs(λ2)<sqrt(eps(T))
                break
            end
            knew=clamp(k-λ1/λ2,a,b)
            if !isfinite(knew)
                break
            end
            if abs(knew-k)<tol
                k=knew
                break
            end
            k=knew
        end
    end
    assemble(k)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(G,adjoint(A),A)
    λ=minimum(eigvals(Hermitian(G)))
    if isfinite(λ) && λ<=λbest
        return k,sqrt(max(zero(T),λ))
    else
        assemble(kbest)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(G,adjoint(A),A)
        λ=minimum(eigvals(Hermitian(G)))
        return kbest,sqrt(max(zero(T),λ))
    end
end

function refine_minima(solver::SweepSolver,basis::AbsBasis,billiard::AbsBilliard,ks::AbstractVector{T},tens::AbstractVector{T};multithreaded_matrices::Bool=true,threshold=200.0,print_refinement::Bool=true,use_krylov::Bool=true,digits::Int=10,which::Symbol=:svd,pts_refinement_factors=(1.0,1.5,2.0),dim_refinement_factors=(1.0,1.1,1.25),window_shrink=3.0,optimizer_kwargs=NamedTuple(),stop_k_tol=0.0,stop_t_tol=0.0,initial_refinement_interval=1e-3,show_progress::Bool=false,newton_max_iter::Int=8,newton_tol::Float64=1e-12,progress_info::Bool=false) where {T<:Real}
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
        idx0=argmin(abs.(ks.-kcur))
        dk_grid=T(initial_refinement_interval)
        if length(ks)>1
            dmin=T(Inf)
            @inbounds for j in eachindex(ks)
                d=abs(ks[j]-ks[idx0])
                if d>zero(T) && d<dmin
                    dmin=d
                end
            end
            isfinite(dmin) && (dk_grid=dmin)
        end
        window=max(3*dk_grid,T(initial_refinement_interval))
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
            if solver_cur isa Union{BoundaryIntegralMethod,CFIE_alpert,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,DLP_kress,DLP_kress_global_corners}
                assemble,A,dA,ddA,G,H,H2,W=_prepare_newton_refinement(solver_cur,basis,billiard,kcur;multithreaded_matrices=multithreaded_matrices)
                knew,tnew=newton_refine_svd!(assemble,A,dA,ddA,G,H,H2,W,kcur;a=a,b=b,maxiter=newton_max_iter,tol=newton_tol)
            else
                dim=max(solver_cur.min_dim,round(Int,billiard.length*kcur*solver_cur.dim_scaling_factor/(2*pi)))
                basis_cur=resize_basis(basis,billiard,dim,kcur)
                pts=evaluate_points(solver_cur,billiard,kcur)
                f=kk->solve(solver_cur,basis_cur,pts,kk;multithreaded=multithreaded_matrices,use_krylov=use_krylov,which=which)
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
            kprev=knew
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
        println(rpad("#",4),rpad("k_approx",digits+8),rpad("k_ref",digits+8),rpad("Δk",digits+8),rpad("log10|t_app|",digits+10),rpad("log10|t_ref|",digits+10),"levels")
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