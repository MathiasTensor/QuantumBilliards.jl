# checks errors for H0/H1 Chebyshev plans against exact SpecialFunctions.besselh values at sampled radii
# there are 2 similar functions for H0/H1 and J0/J1 since they use different plan types (due to Hankels being much more difficult to handle) 
# and eval functions, but the logic is the same. There is also the legacy H1x function for old BIM code
@inline function _cheb_tloc(plan,r)
    if r<plan.rmin
        return Int32(0),0.0
    end
    p=_find_panel(plan,r)
    P=plan.panels[p]
    return Int32(p),(2*r-(P.b+P.a))/(P.b-P.a)
end

function _check_H0H1_errors!(err0,err1,plans0,plans1,ks,rs)
    nz=length(ks)
    Threads.@threads for j in 1:nz
        e0=0.0
        e1=0.0
        k=ComplexF64(ks[j])
        @inbounds for r in rs
            p0,t0=_cheb_tloc(plans0[j],r)
            p1,t1=_cheb_tloc(plans1[j],r)
            z=k*r
            e0=max(e0,abs(eval_h(plans0[j],p0,t0,r)-SpecialFunctions.besselh(0,1,z)))
            e1=max(e1,abs(eval_h(plans1[j],p1,t1,r)-SpecialFunctions.besselh(1,1,z)))
        end
        err0[j]=e0
        err1[j]=e1
    end
    return err0,err1
end

function _check_J0J1_errors!(err0,err1,plans0,plans1,ks,rs)
    nz=length(ks)
    Threads.@threads for j in 1:nz
        e0=0.0
        e1=0.0
        k=ComplexF64(ks[j])
        @inbounds for r in rs
            p0,t0=_cheb_tloc(plans0[j],r)
            p1,t1=_cheb_tloc(plans1[j],r)
            z=k*r
            e0=max(e0,abs(eval_j(plans0[j],p0,t0,r)-SpecialFunctions.besselj(0,z)))
            e1=max(e1,abs(eval_j(plans1[j],p1,t1,r)-SpecialFunctions.besselj(1,z)))
        end
        err0[j]=e0
        err1[j]=e1
    end
    return err0,err1
end

function _check_H1x_errors!(err,plans,ks,rs)
    nz=length(ks)
    Threads.@threads for j in 1:nz
        e=0.0
        k=ComplexF64(ks[j])
        buf=Vector{ComplexF64}(undef,length(rs))
        pidx=Vector{Int32}(undef,length(rs))
        tloc=Vector{Float64}(undef,length(rs))
        invsqrt=Vector{Float64}(undef,length(rs))
        @inbounds for i in eachindex(rs)
            r=rs[i]
            p,t=_cheb_tloc(plans[j],r)
            pidx[i]=p
            tloc[i]=t
            invsqrt[i]=inv(sqrt(r))
        end
        eval_h1x!(buf,plans[j],rs,pidx,tloc,invsqrt)
        @inbounds for i in eachindex(rs)
            e=max(e,abs(buf[i]-SpecialFunctions.besselhx(1,1,k*rs[i])))
        end
        err[j]=e
    end
    return err
end

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
#   - npanels_h_init:
#       Initial number of Chebyshev panels for H1x interpolation.
#   - M_h_init:
#       Initial polynomial degree for H1x interpolation.
#   - npanels_j_init:
#       Initial number of Chebyshev panels for Bessel J interpolation (if needed).
#   - M_j_init:
#       Initial polynomial degree for Bessel J interpolation (if needed).
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
#   - verbose:
#       Print progress information.
function chebyshev_params(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}};npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,tol::Real=1e-10,sampling_points::Int=50_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,verbose::Bool=false) where {T<:Real}
    rmin_raw,rmax=estimate_rmin_rmax(pts,solver.symmetry)
    rmin_cheb=minimum(hankel_z_chebyshev_cutoff./abs.(zj))
    rmin_interp=max(Float64(rmin_raw),rmin_cheb)
    @info "Estimated Chebyshev radial bounds " rmin_raw rmax rmin_cheb rmin_interp
    rs=collect(range(rmin_interp,Float64(rmax);length=sampling_points))
    nz=length(zj)
    n=npanels_h_init
    M=M_h_init
    plans=Vector{ChebHankelPlanH1x}(undef,nz)
    err=fill(Inf,nz)
    for it in 1:max_iter
        Threads.@threads for j in eachindex(zj)
            plans[j]=plan_h1x(ComplexF64(zj[j]),rmin_interp,Float64(rmax);npanels=n,M=M)
        end
        _check_H1x_errors!(err,plans,zj,rs)
        verbose && @info "Worst H1x | n_panels M" maximum(err) n M
        all(<(tol),err) && return n,M,0,0,plans,err
        it%5==0 ? (M+=grow_M) : (n=ceil(Int,grow_panels*n))
    end
    @warn "BIM Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort."
    return n,M,0,0,plans,err
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
#   - npanels_h_init:
#       Initial panel count for Hankel function interpolation.
#   - M_h_init:
#       Initial polynomial degree for Hankel function interpolation.
#   - npanels_j_init:
#       Initial panel count for Bessel J function interpolation (if needed).
#   - M_j_init:
#       Initial polynomial degree for Bessel J function interpolation (if needed).
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
#   - verbose:
#       Print tuning progress.
function chebyshev_params(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,DLP_kress,DLP_kress_global_corners},pts::Union{Vector{BoundaryPointsCFIE{T}},BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,tol::Real=1e-10,sampling_points::Int=50_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,verbose::Bool=false) where {T<:Real}
    rmin_cheb=minimum(hankel_z_chebyshev_cutoff./abs.(zj))
    ptsv=pts isa Vector ? pts : [pts]
    pts1=pts isa Vector ? (length(pts)==1 ? pts[1] : error("DLP_kress expects one BoundaryPointsCFIE component.")) : pts
    block_cache=solver isa Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners} ?
    build_cfie_kress_block_caches(solver,ptsv;npanels_h=16,M_h=4,rmin_cheb=rmin_cheb) :
    build_dlp_kress_block_cache(solver,pts1;npanels=16,M=4,rmin_cheb=rmin_cheb)
    rmin_h=block_cache.rmin
    rmax=block_cache.rmax
    rsH=collect(range(Float64(rmin_h),Float64(rmax);length=sampling_points))
    rsJ=collect(range(0.0,Float64(rmax);length=sampling_points))
    nz=length(zj)
    nh=npanels_h_init
    nj=npanels_j_init
    Mh=M_h_init
    Mj=M_j_init
    plans0=Vector{ChebHankelPlanH}(undef,nz)
    plans1=Vector{ChebHankelPlanH}(undef,nz)
    plansj0=Vector{ChebJPlan}(undef,nz)
    plansj1=Vector{ChebJPlan}(undef,nz)
    errH0=fill(Inf,nz)
    errH1=fill(Inf,nz)
    errJ0=fill(Inf,nz)
    errJ1=fill(Inf,nz)
    for it in 1:max_iter
        plans0,plans1,plansj0,plansj1=build_CFIE_plans_kress(zj,Float64(rmin_h),Float64(rmax);npanels_h=nh,M_h=Mh,npanels_j=nj,M_j=Mj,nthreads=Threads.nthreads())
        _check_H0H1_errors!(errH0,errH1,plans0,plans1,zj,rsH)
        _check_J0J1_errors!(errJ0,errJ1,plansj0,plansj1,zj,rsJ)
        okH=all(<(tol),errH0)&&all(<(tol),errH1)
        okJ=all(<(tol),errJ0)&&all(<(tol),errJ1)
        verbose && @info "Worst Kress H0 H1 J0 J1 | nh Mh nj Mj" maximum(errH0) maximum(errH1) maximum(errJ0) maximum(errJ1) nh Mh nj Mj
        okH&&okJ && return nh,Mh,nj,Mj,plans0,plans1,plansj0,plansj1,errH0,errH1,errJ0,errJ1
        if !okH
            it%5==0 ? (Mh+=grow_M) : (nh=ceil(Int,grow_panels*nh))
        end
        if !okJ
            it%5==0 ? (Mj+=grow_M) : (nj=ceil(Int,grow_panels*nj))
        end
    end
    @warn "Kress Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort."
    return nh,Mh,nj,Mj,plans0,plans1,plansj0,plansj1,errH0,errH1,errJ0,errJ1
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
#   - tol:
#       Required max absolute error tolerance.
#   - sampling_points:
#       Number of radii sampled in [rmin,rmax].
#   - max_iter:
#       Maximum tuning iterations.
#   - npanels_h_init:
#       Initial panel count for Hankel function interpolation.
#   - M_h_init:
#       Initial polynomial degree for Hankel function interpolation.
#   - npanels_j_init:
#       Initial panel count for Bessel J function interpolation (if needed).
#   - M_j_init:
#       Initial polynomial degree for Bessel J function interpolation (if needed).
#   - grow_panels:
#       Multiplicative growth factor for panel count.
#   - grow_M:
#       Additive growth for polynomial degree.
#   - verbose:
#       Whether to print progress info.
function chebyshev_params(solver::CFIE_alpert{T},pts::Union{Vector{BoundaryPointsCFIE{T}},BoundaryPointsCFIE{T}},zj::AbstractVector{Complex{T}};npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,tol::Real=1e-10,sampling_points::Int=50_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,verbose::Bool=false) where {T<:Real}
    ptsv=pts isa Vector ? pts : [pts]
    ws=build_cfie_alpert_workspace(solver,ptsv)
    rmin_raw,rmax=estimate_cfie_alpert_cheb_rbounds(ws)
    rmin_cheb=minimum(hankel_z_chebyshev_cutoff./abs.(zj))
    rmin=max(Float64(rmin_raw),rmin_cheb)
    @info "Estimated Alpert Chebyshev radial bounds " rmin_raw rmax rmin_cheb rmin
    rs=collect(range(rmin,Float64(rmax);length=sampling_points))
    nz=length(zj)
    nh=npanels_h_init
    Mh=M_h_init
    nj=npanels_j_init
    Mj=M_j_init
    plans0=Vector{ChebHankelPlanH}(undef,nz)
    plans1=Vector{ChebHankelPlanH}(undef,nz)
    err0=fill(Inf,nz)
    err1=fill(Inf,nz)
    for it in 1:max_iter
        plans0,plans1=build_CFIE_plans_alpert(zj,rmin,Float64(rmax);npanels=nh,M=Mh,nthreads=Threads.nthreads())
        _check_H0H1_errors!(err0,err1,plans0,plans1,zj,rs)
        verbose && @info "Worst Alpert H0 H1 | npanels_h M_h" maximum(err0) maximum(err1) nh Mh
        all(<(tol),err0)&&all(<(tol),err1) && return nh,Mh,0,0,plans0,plans1,err0,err1
        it%5==0 ? (Mh+=grow_M) : (nh=ceil(Int,grow_panels*nh))
    end
    @warn "CFIE_alpert Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort."
    return nh,Mh,0,0,plans0,plans1,err0,err1
end