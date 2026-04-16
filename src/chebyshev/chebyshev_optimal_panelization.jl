# chebyshev_params
# Determine suitable Chebyshev panel count and polynomial degree for the
# scaled Hankel kernel H1x used in the BoundaryIntegralMethod (BIM).
#
# Logic:
# - Estimate a global radial interval [rmin,rmax] from the geometry,
#   including symmetry images if present.
# - Sample many radii rs in [rmin,rmax].
# - Build Chebyshev plans for the scaled Hankel kernel
#       H1x(z) = exp(-i z) H1^(1)(z)
#   for each complex frequency zj.
# - Evaluate the Chebyshev approximation at the sampled radii.
# - Compare against the exact SpecialFunctions.besselhx values.
#
# Notes:
# - Increasing the number of panels is typically more effective than
#   increasing M, especially for highly oscillatory or complex-valued z.
# - The scaling exp(-i z) improves numerical stability for complex k.
#
# Inputs:
#   - solver::BoundaryIntegralMethod:
#       Solver containing symmetry information.
#   - pts::BoundaryPoints{T}:
#       Boundary discretization.
#   - zj::AbstractVector{Complex{T}}:
#       Complex frequencies (e.g. Beyn contour nodes).
#
# Keyword options:
#   - n_panels_init:
#       Initial number of Chebyshev panels.
#   - M_init:
#       Initial polynomial degree per panel.
#   - grading:
#       Panel distribution strategy (:uniform or geometric).
#   - tol:
#       Target max absolute error.
#   - sampling_points:
#       Number of radii sampled in [rmin,rmax].
#   - max_iter:
#       Maximum tuning iterations.
#   - grow_panels:
#       Multiplicative growth factor for panel count.
#   - grow_M:
#       Additive increase of polynomial degree.
#   - geo_ratio:
#       Ratio for geometric grading.
#   - verbose:
#       Print progress information.
#
# Outputs:
#   - n_panels::Int:
#       Tuned number of panels.
#   - M::Int:
#       Tuned polynomial degree.
#   - plans::Vector{ChebHankelPlanH1x}:
#       Chebyshev plans for each zj.
#   - max_errs::Vector{Float64}:
#       Max error per zj.
function chebyshev_params(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}};n_panels_init::Int=15_000,M_init::Int=5,grading::Symbol=:uniform,tol::Real=1e-10,sampling_points::Int=50_000,max_iter::Int=10,grow_panels::Real=1.5,grow_M::Int=2,geo_ratio::Real=1.05,verbose::Bool=false) where {T<:Real}
    rmin,rmax=estimate_rmin_rmax(pts,solver.symmetry)
    rs=collect(range(Float64(rmin),Float64(rmax);length=sampling_points))
    nz=length(zj)
    n_panels=n_panels_init
    M=M_init
    plans=Vector{ChebHankelPlanH1x}(undef,nz)
    approx=Matrix{ComplexF64}(undef,sampling_points,nz)
    exact=Matrix{ComplexF64}(undef,sampling_points,nz)
    max_errs=fill(Inf,nz)
    for it in 1:max_iter
        Threads.@threads for j in eachindex(zj)
            plans[j]=plan_h1x(ComplexF64(zj[j]),Float64(rmin),Float64(rmax);npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
        end
        Threads.@threads for j in eachindex(zj)
            pidx,tloc,invsqrt=panel_and_geom(plans[j],rs)
            eval_h1x!(view(approx,:,j),plans[j],rs,pidx,tloc,invsqrt)
        end
        Threads.@threads for j in eachindex(zj)
            @inbounds for i in eachindex(rs)
                exact[i,j]=SpecialFunctions.besselhx(1,1,ComplexF64(zj[j])*rs[i])
            end
        end
        @inbounds for j in 1:nz
            max_errs[j]=maximum(abs.(view(approx,:,j).-view(exact,:,j)))
        end
        verbose && @info "BIM Chebyshev tuning" iteration=it n_panels=n_panels M=M max_err=maximum(max_errs)
        all(err->err<tol,max_errs) && return n_panels,M,plans,max_errs
        if it%5==0
            M+=grow_M
        else
            n_panels=ceil(Int,grow_panels*n_panels)
        end
    end
    @warn "BIM Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort."
    return n_panels,M,plans,max_errs
end

# chebyshev_params
# Determine suitable Chebyshev panel count and polynomial degree for the kernel evaluation (H0, H1, J0, J1).
#
# Logic:
# - Build a block cache to extract a global radial interval
#   [rmin,rmax] across all boundary components.
# - Sample many radii rs in [rmin,rmax].
# - Build Chebyshev plans for:
#       H0, H1  (Hankel functions)
#       J0, J1  (Bessel functions)
#   for each complex frequency zj.
# - Evaluate the Chebyshev approximations at sampled radii.
# - Compare against exact SpecialFunctions values.
# - Iteratively refine panel count and polynomial degree until the
#   desired accuracy is achieved.
#
# Inputs:
#   - solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,DLP_kress,DLP_kress_global_corners}:
#       Kress-based CFIE solver.
#   - pts::Vector{BoundaryPointsCFIE{T}} or BoundaryPointsCFIE{T}:
#       Boundary components (multi-component geometry).
#   - zj::AbstractVector{Complex{T}}:
#       Complex wavenumbers (e.g. Beyn contour nodes).
#
# Keyword options:
#   - n_panels_init:
#       Initial number of Chebyshev panels.
#   - M_init:
#       Initial polynomial degree.
#   - grading:
#       Panel distribution strategy.
#   - tol:
#       Target max absolute error.
#   - sampling_points:
#       Number of sampled radii.
#   - max_iter:
#       Maximum tuning iterations.
#   - grow_panels:
#       Multiplicative growth factor for panels.
#   - grow_M:
#       Additive growth for polynomial degree.
#   - geo_ratio:
#       Geometric grading ratio.
#   - verbose:
#       Print tuning progress.
#
# Outputs:
#   - n_panels::Int
#   - M::Int
#   - plans0::Vector{ChebHankelPlanH}   (H0)
#   - plans1::Vector{ChebHankelPlanH}   (H1)
#   - plans2::Vector{ChebJPlan}         (J0)
#   - plans3::Vector{ChebJPlan}         (J1)
#   - max_errs0::Vector{Float64}
#   - max_errs1::Vector{Float64}
#   - max_errs2::Vector{Float64}
#   - max_errs3::Vector{Float64}
function chebyshev_params(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,DLP_kress,DLP_kress_global_corners},pts::Union{Vector{BoundaryPointsCFIE{T}},BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};n_panels_init::Int=15_000,M_init::Int=5,grading::Symbol=:uniform,tol::Real=1e-10,sampling_points::Int=50_000,max_iter::Int=10,grow_panels::Real=1.5,grow_M::Int=2,geo_ratio::Real=1.05,verbose::Bool=false) where {T<:Real}
    if solver isa CFIE_kress || solver isa CFIE_kress_corners || solver isa CFIE_kress_global_corners
        block_cache=build_cfie_kress_block_caches(solver,pts;npanels=16,M=4,grading=grading,geo_ratio=geo_ratio) # just need it for rmin and rmax, really hate this hack, but dont care enough, not in hot loop
    else
        block_cache=build_dlp_kress_block_cache(solver,pts;npanels=16,M=4,grading=grading,geo_ratio=geo_ratio) 
    end
    rmin,rmax=block_cache.rmin,block_cache.rmax
    @info "Estimated Chebyshev radial bounds for CFIE kress solvers" rmin=rmin rmax=rmax
    rs=collect(range(rmin,rmax;length=sampling_points))
    nz=length(zj)
    n_panels=n_panels_init
    M=M_init
    plans0=Vector{ChebHankelPlanH}(undef,nz)
    plans1=Vector{ChebHankelPlanH}(undef,nz)
    plans2=Vector{ChebJPlan}(undef,nz)
    plans3=Vector{ChebJPlan}(undef,nz)
    approx0=Matrix{ComplexF64}(undef,sampling_points,nz)
    approx1=Matrix{ComplexF64}(undef,sampling_points,nz)
    approx2=Matrix{ComplexF64}(undef,sampling_points,nz)
    approx3=Matrix{ComplexF64}(undef,sampling_points,nz)
    exact0=Matrix{ComplexF64}(undef,sampling_points,nz)
    exact1=Matrix{ComplexF64}(undef,sampling_points,nz)
    exact2=Matrix{ComplexF64}(undef,sampling_points,nz)
    exact3=Matrix{ComplexF64}(undef,sampling_points,nz)
    max_errs0=fill(Inf,nz)
    max_errs1=fill(Inf,nz)
    max_errs2=fill(Inf,nz)
    max_errs3=fill(Inf,nz)
    for it in 1:max_iter
        plans0,plans1,plans2,plans3=build_CFIE_plans_kress(zj,rmin,rmax;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=Threads.nthreads())
        Threads.@threads for j in eachindex(zj)
            pidx0,tloc0,_=panel_and_geom(plans0[j],rs)
            pidx1,tloc1,_=panel_and_geom(plans1[j],rs)
            pidx2,tloc2,_=panel_and_geom(plans2[j],rs)
            pidx3,tloc3,_=panel_and_geom(plans3[j],rs)
            eval_h!(view(approx0,:,j),plans0[j],rs,pidx0,tloc0)
            eval_h!(view(approx1,:,j),plans1[j],rs,pidx1,tloc1)
            eval_j!(view(approx2,:,j),plans2[j],rs,pidx2,tloc2)
            eval_j!(view(approx3,:,j),plans3[j],rs,pidx3,tloc3)
        end
        Threads.@threads for j in eachindex(zj)
            @inbounds for i in eachindex(rs)
                z=ComplexF64(zj[j])*rs[i]
                exact0[i,j]=SpecialFunctions.besselh(0,1,z)
                exact1[i,j]=SpecialFunctions.besselh(1,1,z)
                exact2[i,j]=SpecialFunctions.besselj(0,z)
                exact3[i,j]=SpecialFunctions.besselj(1,z)
            end
        end
        @inbounds for j in 1:nz
            max_errs0[j]=maximum(abs.(view(approx0,:,j).-view(exact0,:,j)))
            max_errs1[j]=maximum(abs.(view(approx1,:,j).-view(exact1,:,j)))
            max_errs2[j]=maximum(abs.(view(approx2,:,j).-view(exact2,:,j)))
            max_errs3[j]=maximum(abs.(view(approx3,:,j).-view(exact3,:,j)))
        end
        verbose && @info "Chebyshev tuning" iteration=it n_panels=n_panels M=M max_err_H0=maximum(max_errs0) max_err_H1=maximum(max_errs1) max_err_J0=maximum(max_errs2) max_err_J1=maximum(max_errs3)
        (all(err->err<tol,max_errs0) && all(err->err<tol,max_errs1) && all(err->err<tol,max_errs2) && all(err->err<tol,max_errs3)) && return n_panels,M,plans0,plans1,plans2,plans3,max_errs0,max_errs1,max_errs2,max_errs3
        if it%5==0
            M+=grow_M
        else
            n_panels=ceil(Int,grow_panels*n_panels)
        end
    end
    @warn "Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort."
    return n_panels,M,plans0,plans1,plans2,plans3,max_errs0,max_errs1,max_errs2,max_errs3
end

# chebyshev_params
# Determine suitable Chebyshev panel count and polynomial degree for the
# CFIE_alpert H0/H1 kernel evaluation over a global radial interval.
#
# Logic:
# - Build the Alpert geometry workspace once, only to extract the global
#   radial interval [rmin,rmax] from the block cache.
# - Sample many radii rs in [rmin,rmax].
# - Build H0/H1 plans for all contour points zj.
# - Compare Chebyshev approximations against exact SpecialFunctions Hankel
#   values at the sampled radii.
#
# Inputs:
#   - solver::CFIE_alpert{T}:
#       Alpert-based CFIE solver.
#   - pts::Vector{BoundaryPointsCFIE{T}} or BoundaryPointsCFIE{T}:
#       Boundary discretization components.
#   - zj::AbstractVector{Complex{T}}:
#       Complex contour points / target k values to tune against.
#
# Keyword options:
#   - n_panels_init:
#       Initial number of Chebyshev panels.
#   - M_init:
#       Initial panel degree.
#   - grading:
#       Panel grading mode.
#   - tol:
#       Required max absolute error tolerance.
#   - sampling_points:
#       Number of radii sampled in [rmin,rmax].
#   - max_iter:
#       Maximum tuning iterations.
#   - grow_panels:
#       Multiplicative growth factor for panel count.
#   - grow_M:
#       Additive growth for polynomial degree.
#   - geo_ratio:
#       Geometric grading ratio if applicable.
#   - verbose:
#       Whether to print progress info.
#
# Outputs:
#   - n_panels::Int
#   - M::Int
#   - plans0::Vector{ChebHankelPlanH}
#   - plans1::Vector{ChebHankelPlanH}
#   - max_errs0::Vector{Float64}
#   - max_errs1::Vector{Float64}
function chebyshev_params(solver::CFIE_alpert{T},pts::Union{Vector{BoundaryPointsCFIE{T}},BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};n_panels_init::Int=15_000,M_init::Int=5,grading::Symbol=:uniform,tol::Real=1e-10,sampling_points::Int=50_000,max_iter::Int=10,grow_panels::Real=1.5,grow_M::Int=2,geo_ratio::Real=1.05,verbose::Bool=false) where {T<:Real}
    ws=build_cfie_alpert_workspace(solver,pts) 
    geomws=build_cfie_alpert_cheb_workspace(solver,pts,ws,zj;npanels=16,M=4,grading=grading,geo_ratio=geo_ratio)
    rmin,rmax=estimate_cfie_alpert_cheb_rbounds(ws)
    @info "Estimated Chebyshev radial bounds for CFIE_alpert" rmin=rmin rmax=rmax
    rs=collect(range(rmin,rmax;length=sampling_points))
    nz=length(zj)
    n_panels=n_panels_init
    M=M_init
    plans0=Vector{ChebHankelPlanH}(undef,nz)
    plans1=Vector{ChebHankelPlanH}(undef,nz)
    approx0=Matrix{ComplexF64}(undef,sampling_points,nz)
    approx1=Matrix{ComplexF64}(undef,sampling_points,nz)
    exact0=Matrix{ComplexF64}(undef,sampling_points,nz)
    exact1=Matrix{ComplexF64}(undef,sampling_points,nz)
    max_errs0=fill(Inf,nz)
    max_errs1=fill(Inf,nz)
    for it in 1:max_iter
        plans0,plans1=build_CFIE_plans_alpert(zj,rmin,rmax;npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio,nthreads=Threads.nthreads())
        Threads.@threads for j in eachindex(zj)
            pidx0,tloc0,_=panel_and_geom(plans0[j],rs)
            pidx1,tloc1,_=panel_and_geom(plans1[j],rs)
            eval_h!(view(approx0,:,j),plans0[j],rs,pidx0,tloc0)
            eval_h!(view(approx1,:,j),plans1[j],rs,pidx1,tloc1)
        end
        Threads.@threads for j in eachindex(zj)
            @inbounds for i in eachindex(rs)
                z=ComplexF64(zj[j])*rs[i]
                exact0[i,j]=SpecialFunctions.besselh(0,1,z)
                exact1[i,j]=SpecialFunctions.besselh(1,1,z)
            end
        end
        @inbounds for j in 1:nz
            max_errs0[j]=maximum(abs.(view(approx0,:,j).-view(exact0,:,j)))
            max_errs1[j]=maximum(abs.(view(approx1,:,j).-view(exact1,:,j)))
        end
        if verbose
            @info "CFIE_alpert Chebyshev tuning" iteration=it n_panels=n_panels M=M max_err_H0=maximum(max_errs0) max_err_H1=maximum(max_errs1) rmin=rmin rmax=rmax
        end
        if all(err->err<tol,max_errs0) && all(err->err<tol,max_errs1)
            return n_panels,M,plans0,plans1,max_errs0,max_errs1
        end
        if it%5==0
            M+=grow_M
        else
            n_panels=ceil(Int,grow_panels*n_panels)
        end
    end
    @warn "CFIE_alpert Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort."
    return n_panels,M,plans0,plans1,max_errs0,max_errs1
end