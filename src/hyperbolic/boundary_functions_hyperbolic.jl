################################################################################
################ HYPERBOLIC BOUNDARY DATA SYMMETRIZATION #######################
################################################################################

@inline _hyp_l2_weight(pts::BoundaryPointsHyp,measure::Symbol)=
    measure===:hyperbolic ? pts.dsH :
    measure===:euclidean ? pts.ds :
    error("Unknown measure=$measure. Use :hyperbolic or :euclidean.")

function _hyp_normalize(u::AbstractVector,pts::BoundaryPointsHyp;normalize::Bool=true,measure::Symbol=:hyperbolic)
    normalize || return u
    w=_hyp_l2_weight(pts,measure)
    nrm=sqrt(sum(abs2(u[i])*w[i] for i in eachindex(u)))
    return u./nrm
end

function symmetrize_layer_density(solver::BIM_hyperbolic,ρ::AbstractVector{N},pts::BoundaryPointsHyp{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    isnothing(solver.symmetry) && return pts,ρ
    pts_full=apply_symmetries_to_boundary_points(pts,solver.symmetry,billiard)
    ρ_full=apply_symmetries_to_boundary_function(ρ,solver.symmetry)
    return pts_full,ρ_full
end

function symmetrize_layer_density(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},ρ::AbstractVector{N},pts::BoundaryPointsHyp{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    isnothing(solver.symmetry) && return pts,ρ
    Ifund,full_to_fund,full_to_scale,_,_=symmetry_index_orbits(T,pts,solver.symmetry,billiard)
    length(ρ)==length(Ifund)||error("Density length mismatch: got $(length(ρ)), expected $(length(Ifund)).")
    S=promote_type(N,Complex{T})
    ρ_full=Vector{S}(undef,length(pts.xy))
    @inbounds for q in eachindex(pts.xy)
        ρ_full[q]=full_to_scale[q]*ρ[full_to_fund[q]]
    end
    return pts,ρ_full
end

function symmetrize_layer_density(solver::DLP_hyperbolic_log_product,ρ::AbstractVector{N},disc::DLPHypLogDiscretization{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts=disc.bp
    isnothing(solver.symmetry) && return pts,ρ
    pts_full=apply_symmetries_to_boundary_points(pts,solver.symmetry,billiard)
    ρ_full=apply_symmetries_to_boundary_function(ρ,solver.symmetry)
    return pts_full,ρ_full
end

function symmetrize_layer_density(solver::BIM_hyperbolic,ρs::Vector{<:AbstractVector{N}},pts::Vector{BoundaryPointsHyp{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{BoundaryPointsHyp{T}}(undef,length(pts))
    ρ_all=Vector{Vector{promote_type(N,Complex{T})}}(undef,length(ρs))
    @use_threads multithreading=multithreaded for i in eachindex(ρs)
        pts_all[i],ρ_all[i]=symmetrize_layer_density(solver,ρs[i],pts[i],billiard)
    end
    return pts_all,ρ_all
end

function symmetrize_layer_density(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},ρs::Vector{<:AbstractVector{N}},pts::Vector{BoundaryPointsHyp{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{BoundaryPointsHyp{T}}(undef,length(pts))
    ρ_all=Vector{Vector{promote_type(N,Complex{T})}}(undef,length(ρs))
    @use_threads multithreading=multithreaded for i in eachindex(ρs)
        pts_all[i],ρ_all[i]=symmetrize_layer_density(solver,ρs[i],pts[i],billiard)
    end
    return pts_all,ρ_all
end

function symmetrize_layer_density(solver::DLP_hyperbolic_log_product,ρs::Vector{<:AbstractVector{N}},discs::Vector{DLPHypLogDiscretization{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{BoundaryPointsHyp{T}}(undef,length(discs))
    ρ_all=Vector{Vector{promote_type(N,Complex{T})}}(undef,length(ρs))
    @use_threads multithreading=multithreaded for i in eachindex(ρs)
        pts_all[i],ρ_all[i]=symmetrize_layer_density(solver,ρs[i],discs[i],billiard)
    end
    return pts_all,ρ_all
end

################################################################################
################ HYPERBOLIC BOUNDARY FUNCTION WRAPPERS #########################
################################################################################

function boundary_function(solver::BIM_hyperbolic,ρ::AbstractVector{N},pts::BoundaryPointsHyp{T},billiard::Bi,k::T;normalize::Bool=true,measure::Symbol=:hyperbolic,representation::Symbol=:boundary_function) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_full,ρ_full=symmetrize_layer_density(solver,ρ,pts,billiard)
    return pts_full,_hyp_normalize(ρ_full,pts_full;normalize=normalize,measure=measure)
end

function boundary_function(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},ρ::AbstractVector{N},pts::BoundaryPointsHyp{T},billiard::Bi,k::T;normalize::Bool=true,measure::Symbol=:hyperbolic,representation::Symbol=:boundary_function) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_full,ρ_full=symmetrize_layer_density(solver,ρ,pts,billiard)
    return pts_full,_hyp_normalize(ρ_full,pts_full;normalize=normalize,measure=measure)
end

function boundary_function(solver::DLP_hyperbolic_log_product,ρ::AbstractVector{N},disc::DLPHypLogDiscretization{T},billiard::Bi,k::T;normalize::Bool=true,measure::Symbol=:hyperbolic,representation::Symbol=:boundary_function) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_full,ρ_full=symmetrize_layer_density(solver,ρ,disc,billiard)
    return pts_full,_hyp_normalize(ρ_full,pts_full;normalize=normalize,measure=measure)
end

function boundary_function(solver::BIM_hyperbolic,ρs::Vector{<:AbstractVector{N}},pts::Vector{BoundaryPointsHyp{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true,normalize::Bool=true,measure::Symbol=:hyperbolic,representation::Symbol=:boundary_function) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{BoundaryPointsHyp{T}}(undef,length(ρs))
    u_all=Vector{Vector{promote_type(N,Complex{T})}}(undef,length(ρs))
    @use_threads multithreading=multithreaded for i in eachindex(ρs)
        pts_all[i],u_all[i]=boundary_function(solver,ρs[i],pts[i],billiard,ks[i];normalize=normalize,measure=measure,representation=representation)
    end
    return pts_all,u_all
end

function boundary_function(solver::Union{DLP_hyperbolic_kress,DLP_hyperbolic_kress_global_corners},ρs::Vector{<:AbstractVector{N}},pts::Vector{BoundaryPointsHyp{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true,normalize::Bool=true,measure::Symbol=:hyperbolic,representation::Symbol=:boundary_function) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{BoundaryPointsHyp{T}}(undef,length(ρs))
    u_all=Vector{Vector{promote_type(N,Complex{T})}}(undef,length(ρs))
    @use_threads multithreading=multithreaded for i in eachindex(ρs)
        pts_all[i],u_all[i]=boundary_function(solver,ρs[i],pts[i],billiard,ks[i];normalize=normalize,measure=measure,representation=representation)
    end
    return pts_all,u_all
end

function boundary_function(solver::DLP_hyperbolic_log_product,ρs::Vector{<:AbstractVector{N}},discs::Vector{DLPHypLogDiscretization{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true,normalize::Bool=true,measure::Symbol=:hyperbolic,representation::Symbol=:boundary_function) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{BoundaryPointsHyp{T}}(undef,length(ρs))
    u_all=Vector{Vector{promote_type(N,Complex{T})}}(undef,length(ρs))
    @use_threads multithreading=multithreaded for i in eachindex(ρs)
        pts_all[i],u_all[i]=boundary_function(solver,ρs[i],discs[i],billiard,ks[i];normalize=normalize,measure=measure,representation=representation)
    end
    return pts_all,u_all
end