_hyp_workspace(solver::DLP_hyperbolic_log_product,pts,k;kwargs...)=build_dlp_hyp_log_workspace(solver,pts,k)
_hyp_workspace(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},pts,k;mp_dps::Int=80,leg_type::Int=3,kwargs...)=build_dlp_hyperbolic_kress_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
function _hyp_points(solver::DLP_hyperbolic_kress,billiard,kmax;
    multithreaded_matrices::Bool=true,M_cdf_base::Int=4000,safety::Real=1e-14,kwargs...)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=M_cdf_base,safety=safety)
    return evaluate_points(solver,billiard,real(kmax),pre;safety=safety,threaded=multithreaded_matrices)
end
function _hyp_points(solver::DLP_hyperbolic_kress_global_corners,billiard,kmax;
    multithreaded_matrices::Bool=true,M_cdf_base::Int=4000,safety::Real=1e-14,kwargs...)
    return evaluate_points(solver,billiard,real(kmax);M_cdf_base=M_cdf_base,safety=safety,threaded=multithreaded_matrices)
end
function _hyp_points(solver::DLP_hyperbolic_log_product,billiard,kmax;kwargs...)
    return evaluate_points(solver,billiard,real(kmax))
end

function k_sweep(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners,DLP_hyperbolic_log_product},basis::Ba,billiard::AbsBilliard,ks::AbstractVector{T};multithreaded_matrices::Bool=true,use_krylov::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40,M_cdf_base::Int=4000,safety::Real=1e-14,mp_dps::Int=80,leg_type::Int=3,which::Symbol=:svd) where {T<:Real,Ba<:AbstractHankelBasis}
    pts=_hyp_points(solver,billiard,maximum(ks);multithreaded_matrices=multithreaded_matrices,M_cdf_base=M_cdf_base,safety=safety)
    σmins=Vector{T}(undef,length(ks))
    A=Matrix{Complex{T}}(undef,0,0)
    p=Progress(length(ks),1)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        ws=_hyp_workspace(solver,pts,k;mp_dps=mp_dps,leg_type=leg_type)
        n=_workspace_dim(ws)
        size(A)!=(n,n) && (A=Matrix{Complex{T}}(undef,n,n))
        construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded_matrices)
        @blas_multi_then_1 MAX_BLAS_THREADS σmins[i]=svdvals(A)[end]
        next!(p)
    end
    return σmins
end