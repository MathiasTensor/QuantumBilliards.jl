function k_sweep(solver::BIM_hyperbolic,basis::Ba,billiard::AbsBilliard,ks::AbstractVector{T};multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000,M_cdf_base::Int=4000,safety::Real=1e-14) where {T<:Real,Ba<:AbstractHankelBasis}
    symmetry=solver.symmetry
    kmax=maximum(ks)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=M_cdf_base,safety=safety)
    pts_hyp=evaluate_points(solver,billiard,real(kmax),pre;safety=safety,threaded=multithreaded_matrices)
    pts=_BoundaryPointsHypBIM_to_BoundaryPoints(pts_hyp)
    N=length(pts.xy)
    dmin,dmax=d_bounds_hyp(pts_hyp,symmetry;dmin_floor=T(1e-15),pad_max=T(1.1))
    preQ=build_QTaylorPrecomp(;dmin=legendre_d_threshold(),dmax=Float64(dmax)*1.05)
    ws=QTaylorWorkspace(;threaded=false)
    tab=alloc_QTaylorTable(preQ)
    K=Matrix{ComplexF64}(undef,N,N)
    σmins=Vector{T}(undef,length(ks))
    p=Progress(length(ks),1)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        build_QTaylorTable!(tab,preQ,ws,complex(k))
        isnothing(symmetry) ? compute_kernel_matrices_DLP_hyperbolic!(K,pts,tab;multithreaded=multithreaded_matrices) : compute_kernel_matrices_DLP_hyperbolic!(K,pts,symmetry,tab;multithreaded=multithreaded_matrices)
        assemble_DLP_hyperbolic!(K,pts)
        if use_krylov
            σ,_,_,_=smallest_svd_triplet(K;tol=tol,maxiter=maxiter)
            σmins[i]=T(σ)
        else
            @blas_multi_then_1 MAX_BLAS_THREADS S=svdvals(K)
            σmins[i]=T(S[end])
        end
        next!(p)
    end
    return σmins
end

function k_sweep(solver::DLP_hyperbolic_kress,basis::Ba,billiard::AbsBilliard,ks::AbstractVector{T};multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,M_cdf_base::Int=4000,safety::Real=1e-14,mp_dps::Int=80,leg_type::Int=3,which::Symbol=:svd) where {T<:Real,Ba<:AbstractHankelBasis}
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=M_cdf_base,safety=safety)
    pts=evaluate_points(solver,billiard,real(maximum(ks)),pre;safety=safety,threaded=multithreaded_matrices)
    σmins=Vector{T}(undef,length(ks))
    A=Matrix{Complex{T}}(undef,0,0)
    p=Progress(length(ks),1)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        ws=build_dlp_hyperbolic_kress_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
        n=_workspace_dim(ws)
        size(A)!=(n,n)&&(A=Matrix{Complex{T}}(undef,n,n))
        construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded_matrices)
        @blas_multi_then_1 MAX_BLAS_THREADS σmins[i]=svdvals(A)[end]
        next!(p)
    end
    return σmins
end

function k_sweep(solver::DLP_hyperbolic_kress_global_corners,basis::Ba,billiard::AbsBilliard,ks::AbstractVector{T};multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,M_cdf_base::Int=4000,safety::Real=1e-14,mp_dps::Int=80,leg_type::Int=3,which::Symbol=:svd) where {T<:Real,Ba<:AbstractHankelBasis}
    pts=evaluate_points(solver,billiard,real(maximum(ks));M_cdf_base=M_cdf_base,safety=safety,threaded=multithreaded_matrices)
    σmins=Vector{T}(undef,length(ks))
    A=Matrix{Complex{T}}(undef,0,0)
    p=Progress(length(ks),1)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        ws=build_dlp_hyperbolic_kress_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
        n=_workspace_dim(ws)
        size(A)!=(n,n)&&(A=Matrix{Complex{T}}(undef,n,n))
        construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded_matrices)
        @blas_multi_then_1 MAX_BLAS_THREADS σmins[i]=svdvals(A)[end]
        next!(p)
    end
    return σmins
end

function k_sweep(solver::DLP_hyperbolic_log_product,basis::Ba,billiard::AbsBilliard,ks::AbstractVector{T};multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,M_cdf_base::Int=4000,safety::Real=1e-14,mp_dps::Int=80,leg_type::Int=3,which::Symbol=:svd) where {T<:Real,Ba<:AbstractHankelBasis}
    disc=evaluate_points(solver,billiard,real(maximum(ks)))
    G=build_dlp_hyp_log_geom_cache(solver,disc)
    σmins=Vector{T}(undef,length(ks))
    A=Matrix{Complex{T}}(undef,0,0)
    p=Progress(length(ks),1)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        ws=build_dlp_hyp_log_workspace(solver,disc,G,k)
        n=_workspace_dim(ws)
        size(A)!=(n,n)&&(A=Matrix{Complex{T}}(undef,n,n))
        construct_matrices!(solver,A,disc,ws,k;multithreaded=multithreaded_matrices,adjoint_mode=:source)
        @blas_multi_then_1 MAX_BLAS_THREADS σmins[i]=svdvals(A)[end]
        next!(p)
    end
    return σmins
end