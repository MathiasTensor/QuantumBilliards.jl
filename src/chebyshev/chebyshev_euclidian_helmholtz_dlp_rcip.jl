##################################################################################
# DLP-RCIP CHEBYSHEV ACCELERATION
#
# This section implements the complex-k Chebyshev accelerated DLP-RCIP matrix
# construction used by Beyn contour integration.
#
# The idea mirrors the direct real/complex RCIP implementation, but replaces
# repeated Hankel/Bessel evaluations in the dense physical DLP assembly by
# piecewise-Chebyshev interpolation of:
#
#     H₁^(1)(k r)
#     J₁(k r)
#
# across all contour wavenumbers.
#
# The H₁ and J₁ interpolation plans are intentionally independent:
#
#   - Hankel H₁ uses a radial interval [rmin,rmax]
#   - J₁ uses [0,rmax]
#   - both may use different panel counts and polynomial degrees
#
# because:
#
#   * H₁ is singular at r → 0 and therefore uses the existing low-z fallback
#     machinery from the shared special-function backend.
#
#   * J₁ is regular at r = 0, so it can safely interpolate from zero and often
#     needs far fewer panels.
#
# The RCIP compressed-corner correction itself is still computed using the exact
# direct complex-k assembly, since those local matrices are small and numerical
# robustness matters more than interpolation speed.
#
# Thus:
#
#   Chebyshev acceleration:
#       physical dense DLP matrix assembly
#
#   direct exact evaluation:
#       local RCIP corner recursion / compressed blocks
#
# For mathematical details of the RCIP formulation, compressed inverse
# recursion, Helsing type-b splitting, and the direct complex-k kernel
# construction, see the direct real/complex DLP-RCIP implementation.
# MO 20/5/26 - initial implementation, experimental
##################################################################################

const INV_TWO_PI=1/(2*pi)
const INV_PI=1/pi

###############################################
##### DLP-RCIP CHEBYSHEV WORKSPACE ############
###############################################

# Workspace for multi-k Chebyshev accelerated DLP-RCIP assembly.
# Stores:
#
#   plans1   : H₁^(1)(k r) Chebyshev plans, one per contour wavenumber
#   plansj1  : J₁(k r) Chebyshev plans, one per contour wavenumber
#   h1_tls   : thread-local H₁ temporary buffers
#   j1_tls   : thread-local J₁ temporary buffers
#   ks       : contour wavenumbers as ComplexF64
#   Mk       : number of contour points
#
# The thread-local buffers avoid repeated allocations inside the dense assembly
# loops.
struct DLPRCIPH1J1ChebWorkspace
    plans1::Vector{ChebHankelPlanH}
    plansj1::Vector{ChebJPlan}
    h1_tls::Vector{Vector{ComplexF64}}
    j1_tls::Vector{Vector{ComplexF64}}
    ks::Vector{ComplexF64}
    Mk::Int
end

# Build the multi-k H₁/J₁ Chebyshev workspace for DLP-RCIP assembly.
# Inputs
#   ks           : contour wavenumbers
#   rmin,rmax    : radial interpolation bounds
#
# Optional controls
#   npanels_h    : number of H₁ radial panels
#   npanels_j    : number of J₁ radial panels
#   M_h          : H₁ Chebyshev degree
#   M_j          : J₁ Chebyshev degree
#   plan_nthreads: threads used for plan construction
#   ntls         : number of thread-local temporary buffers
#
# Notes
#   H₁ and J₁ are intentionally constructed with independent discretizations.
#   J₁ starts at r=0 because it is regular there.
function build_dlp_rcip_h1_j1_cheb_workspace(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels_h::Int=10000,npanels_j::Int=3000,M_h::Int=5,M_j::Int=5,plan_nthreads::Int=Threads.nthreads(),ntls::Int=Threads.nthreads())
    Mk=length(ks)
    plans1=Vector{ChebHankelPlanH}(undef,Mk)
    plansj1=Vector{ChebJPlan}(undef,Mk)
    if plan_nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
            plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j)
        end
    else
        Threads.@threads for m in 1:Mk
            k=ComplexF64(ks[m])
            plans1[m]=plan_h(1,1,k,rmin,rmax;npanels=npanels_h,M=M_h)
            plansj1[m]=plan_j(1,k,0.0,rmax;npanels=npanels_j,M=M_j)
        end
    end
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    j1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return DLPRCIPH1J1ChebWorkspace(plans1,plansj1,h1_tls,j1_tls,ComplexF64.(ks),Mk)
end

# Compute the Hankel Chebyshev panel index and local coordinate for radius r.
# If r lies below the interpolation range, returns: pidx = 0
# which activates the low-z fallback path in the shared special-function
# evaluator.
@inline function _h_panel_t_rcip(pl::ChebHankelPlanH,r::Float64)
    if r < pl.rmin
        return Int32(0),0.0
    else
        p=_find_panel(pl,r)
        P=pl.panels[p]
        return Int32(p),(2*r-(P.b+P.a))/(P.b-P.a)
    end
end

# Compute the J₁ Chebyshev panel index and local coordinate for radius r.
# Since J₁ is regular at r=0 and the plan includes zero, no low-z fallback
# sentinel is needed here.
@inline function _j_panel_t_rcip(pl::ChebJPlan,r::Float64)
    p=_find_panel(pl,r)
    P=pl.panels[p]
    return Int32(p),(2*r-(P.b+P.a))/(P.b-P.a)
end

# Evaluate H₁^(1)(k r) and J₁(k r) for all contour wavenumbers at one radius.
# The H₁ and J₁ plans may use different radial discretizations, so separate
# panel lookups are required.
@inline function _h1_j1_at_r_rcip!(h1vals,j1vals,plans1,plansj1,r::Float64)
    pidx_h,t_h=_h_panel_t_rcip(plans1[1],r)
    pidx_j,t_j=_j_panel_t_rcip(plansj1[1],r)
    h1_j1_multi_ks_at_r!(h1vals,j1vals,plans1,plansj1,pidx_h,t_h,pidx_j,t_j,r)
    return nothing
end

# Build the full DLP-RCIP Chebyshev workspace from the RCIP geometry.
# The interpolation bounds are estimated automatically from the boundary
function build_dlp_rcip_cheb_workspace(solver::DLP_rcip,pts::DLPRCIPDiscretization{T,C},ks::AbstractVector{<:Number};npanels_h::Int=10000,npanels_j::Int=3000,M_h::Int=5,M_j::Int=5,plan_nthreads::Int=Threads.nthreads(),ntls::Int=Threads.nthreads(),pad=(T(0.9),T(1.1)),rmax_factor::Real=1.0) where {T<:Real,C}
    rmin,rmax=estimate_rmin_rmax(pts.bp,solver.symmetry;pad=pad,rmax_factor=rmax_factor)
    return build_dlp_rcip_h1_j1_cheb_workspace(ks,Float64(rmin),Float64(rmax);npanels_h=npanels_h,npanels_j=npanels_j,M_h=M_h,M_j=M_j,plan_nthreads=plan_nthreads,ntls=ntls)
end

######################################
##### DLP-RCIP CHEBYSHEV ASSEMBLY ####
######################################

# Assemble the dense physical DLP matrices for all contour wavenumbers using
# Chebyshev-accelerated H₁/J₁ evaluation.
# This constructs the uncompressed physical DLP operator:
# including:
#
#   - diagonal curvature correction
#   - standard off-diagonal DLP kernel
#   - panel-local logarithmic quadrature corrections
#   - optional symmetry images
#
# The resulting matrices are later RCIP-compressed separately.
#
# Notes
#   - physical dense matrix only
#   - no RCIP compression performed here
#   - H₁/J₁ values are reused across all contour ks
function assemble_dlp_rcip_chebyshev!(Ds::Vector{Matrix{ComplexF64}},disc::DLPRCIPDiscretization{T,C},chebws::DLPRCIPH1J1ChebWorkspace,Λ::Matrix{T},ξ::Vector{T},ω::Vector{T};symmetry=nothing,use_panel_logquad::Bool=true,multithreaded::Bool=true) where {T<:Real,C}
    Mk=chebws.Mk
    N=boundary_matrix_size(disc)
    xy=disc.bp.xy
    normal=disc.bp.normal
    ds=disc.bp.ds
    κ=disc.bp.curvature
    panel_id=disc.pdata.panel_id
    local_id=disc.pdata.local_id
    plans1=chebws.plans1
    plansj1=chebws.plansj1
    ks=chebws.ks
    h1_tls=chebws.h1_tls
    j1_tls=chebws.j1_tls
    @inbounds for m in 1:Mk
        fill!(Ds[m],0.0+0.0im)
    end
    add_x=false;add_y=false;add_xy=false
    sxgn=1.0+0im;sygn=1.0+0im;sxy=1.0+0im
    have_rot=false;nrot=1;mrot=0
    cx=zero(T);cy=zero(T)
    if !isnothing(symmetry)
        s=symmetry
        if hasproperty(s,:axis)
            if s.axis==:y_axis
                add_x=true
                sxgn=ComplexF64(s.parity==-1 ? -1.0 : 1.0)
            elseif s.axis==:x_axis
                add_y=true
                sygn=ComplexF64(s.parity==-1 ? -1.0 : 1.0)
            elseif s.axis==:origin
                add_x=true;add_y=true;add_xy=true
                px=s.parity[1]==-1 ? -1.0 : 1.0
                py=s.parity[2]==-1 ? -1.0 : 1.0
                sxgn=ComplexF64(px)
                sygn=ComplexF64(py)
                sxy=ComplexF64(px*py)
            end
        elseif s isa Rotation
            have_rot=true
            nrot=s.n
            mrot=mod(s.m,nrot)
            cx=T(s.center[1])
            cy=T(s.center[2])
        end
    end
    ctab=T[]
    stab=T[]
    χ=ComplexF64[]
    if have_rot
        ctab,stab,χ=_rotation_tables(T,nrot,mrot)
    end
    shift_x=disc.bp.shift_x
    shift_y=disc.bp.shift_y
    @use_threads multithreading=multithreaded for j in 1:N
        tid=Threads.threadid()
        h1vals=h1_tls[tid]
        j1vals=j1_tls[tid]
        xj,yj=xy[j]
        nxj,nyj=normal[j]
        dsj=ds[j]
        pj=panel_id[j]
        jl=local_id[j]
        speed_half_j=dsj/ω[jl]
        @inbounds for i in 1:N
            xi,yi=xy[i]
            if i==j
                diag=Complex{T}(dsj*κ[j]*INV_TWO_PI,zero(T))
                for m in 1:Mk
                    Ds[m][i,j]+=diag
                end
            else
                dx=xi-xj
                dy=yi-yj
                r=hypot(dx,dy)
                invr=inv(r)
                q=(dx*nxj+dy*nyj)*invr
                _h1_j1_at_r_rcip!(h1vals,j1vals,plans1,plansj1,Float64(r))
                if use_panel_logquad && panel_id[i]==pj
                    il=local_id[i]
                    du=abs(ξ[il]-ξ[jl])
                    for m in 1:Mk
                        full=0.5im*ks[m]*h1vals[m]*q*speed_half_j
                        L1=-(ks[m]/pi)*j1vals[m]*q*speed_half_j
                        L2=full-L1*log(du)
                        Ds[m][i,j]+=Λ[il,jl]*L1+ω[jl]*L2
                    end
                else
                    for m in 1:Mk
                        Ds[m][i,j]+=q*(0.5im*ks[m]*h1vals[m])*dsj
                    end
                end
            end
            if add_x
                xr=_x_reflect(xj,shift_x);yr=yj
                nxr,nyr=_x_reflect_normal(nxj,nyj)
                dx=xi-xr;dy=yi-yr;r=hypot(dx,dy)
                if r>eps(T)
                    invr=inv(r);q=(dx*nxr+dy*nyr)*invr
                    _h1_j1_at_r_rcip!(h1vals,j1vals,plans1,plansj1,Float64(r))
                    for m in 1:Mk
                        Ds[m][i,j]+=sxgn*q*(0.5im*ks[m]*h1vals[m])*dsj
                    end
                end
            end
            if add_y
                xr=xj;yr=_y_reflect(yj,shift_y)
                nxr,nyr=_y_reflect_normal(nxj,nyj)
                dx=xi-xr;dy=yi-yr;r=hypot(dx,dy)
                if r>eps(T)
                    invr=inv(r);q=(dx*nxr+dy*nyr)*invr
                    _h1_j1_at_r_rcip!(h1vals,j1vals,plans1,plansj1,Float64(r))
                    for m in 1:Mk
                        Ds[m][i,j]+=sygn*q*(0.5im*ks[m]*h1vals[m])*dsj
                    end
                end
            end
            if add_xy
                xr=_x_reflect(xj,shift_x);yr=_y_reflect(yj,shift_y)
                nxr,nyr=_xy_reflect_normal(nxj,nyj)
                dx=xi-xr;dy=yi-yr;r=hypot(dx,dy)
                if r>eps(T)
                    invr=inv(r);q=(dx*nxr+dy*nyr)*invr
                    _h1_j1_at_r_rcip!(h1vals,j1vals,plans1,plansj1,Float64(r))
                    for m in 1:Mk
                        Ds[m][i,j]+=sxy*q*(0.5im*ks[m]*h1vals[m])*dsj
                    end
                end
            end
            if have_rot
                for l in 1:nrot-1
                    xr,yr=_rot_point(xj,yj,cx,cy,ctab[l+1],stab[l+1])
                    nxr,nyr=_rot_vec(nxj,nyj,ctab[l+1],stab[l+1])
                    dx=xi-xr;dy=yi-yr;r=hypot(dx,dy)
                    if r>eps(T)
                        invr=inv(r);q=(dx*nxr+dy*nyr)*invr
                        _h1_j1_at_r_rcip!(h1vals,j1vals,plans1,plansj1,Float64(r))
                        scale=ComplexF64(χ[l+1])
                        for m in 1:Mk
                            Ds[m][i,j]+=scale*q*(0.5im*ks[m]*h1vals[m])*dsj
                        end
                    end
                end
            end
        end
    end
    return Ds
end

function assemble_dlp_rcip_complex!(D::Matrix{Complex{T}},disc::DLPRCIPDiscretization{T,C},k::ComplexF64,Λ::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true,multithreaded::Bool=true) where {T<:Real,C}
    N=boundary_matrix_size(disc)
    fill!(D,zero(Complex{T}))
    xy=disc.bp.xy
    normal=disc.bp.normal
    ds=disc.bp.ds
    κ=disc.bp.curvature
    panel_id=disc.pdata.panel_id
    local_id=disc.pdata.local_id
    @use_threads multithreading=multithreaded for j in 1:N
        xj,yj=xy[j]
        nxj,nyj=normal[j]
        dsj=ds[j]
        pj=panel_id[j]
        jl=local_id[j]
        speed_half_j=dsj/ω[jl]
        @inbounds for i in 1:N
            xi,yi=xy[i]
            if i==j
                D[i,j]+=Complex{T}(dsj*κ[j]*INV_TWO_PI,zero(T))
            else
                dx=xi-xj
                dy=yi-yj
                r=hypot(dx,dy)
                invr=inv(r)
                q=(dx*nxj+dy*nyj)*invr
                if use_panel_logquad && panel_id[i]==pj
                    il=local_id[i]
                    du=abs(ξ[il]-ξ[jl])
                    full=0.5im*k*H(1,k*r)*q*speed_half_j
                    L1=-(k/pi)*J(1,k*r)*q*speed_half_j
                    L2=full-L1*log(du)
                    D[i,j]+=Λ[il,jl]*L1+ω[jl]*L2
                else
                    D[i,j]+=0.5im*k*H(1,k*r)*q*dsj
                end
            end
        end
    end
    return D
end

function assemble_typeb_K_complex!(lev::DLPRCIPLevelCache{T,C},k::ComplexF64,Λlocal::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true,multithreaded::Bool=true) where {T<:Real,C}
    assemble_dlp_rcip_complex!(lev.K,lev.pts,k,Λlocal,ξ,ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
    lev.K.*=-one(T)
    split_helsing_typeb!(lev.Kstar,lev.Kcirc,lev.K,lev.pts.pdata.ngl)
    return lev.Kstar,lev.Kcirc
end

# Compute the RCIP compressed inverse for one corner at complex k.
# This performs the standard Helsing recursive compression:
#   coarse solve
#   successive refined Schur updates
#   compressed inverse convergence
# using exact complex-k local kernels.
# The Chebyshev acceleration is intentionally not used here since small matrices
function rcip_R_for_corner_complex!(cc::DLPRCIPCornerCache{T,C},Rloc::Matrix{Complex{T}},k::ComplexF64,Λlocal::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true,stop_tol::T=T(1e-13),min_scale::T=T(5e-14),scale_safety::T=T(100),multithreaded::Bool=true) where {T<:Real,C}
    ngl=cc.levels[1].pts.pdata.ngl
    ii=inner_typeb(ngl)
    oo=outer_typeb(ngl)
    scale_floor=scale_safety*min_scale
    istart=findfirst(lev->lev.scale>=scale_floor,cc.levels)
    isnothing(istart) && error("all RCIP levels below safe scale at corner $(cc.meta.cid)")
    lev0=cc.levels[istart]
    assemble_typeb_K_complex!(lev0,k,Λlocal,ξ,ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
    copyto!(cc.Rnew,cc.Icoarse)
    @views cc.Rnew.+=lev0.Kstar[ii,ii]
    finite_matrix(cc.Rnew) || error("non-finite initial RCIP local system at corner $(cc.meta.cid)")
    F=lu!(cc.Rnew)
    copyto!(Rloc,cc.Icoarse)
    ldiv!(F,Rloc)
    for ell in (istart+1):length(cc.levels)
        lev=cc.levels[ell]
        lev.scale<scale_floor && continue
        assemble_typeb_K_complex!(lev,k,Λlocal,ξ,ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
        finite_matrix(lev.Kcirc) || error("non-finite Kcirc at corner $(cc.meta.cid), level $ell")
        @views begin
            Koo=lev.Kcirc[oo,oo]
            Koi=lev.Kcirc[oo,ii]
            Kio=lev.Kcirc[ii,oo]
            Po=cc.P[oo,:]
            Pi=cc.P[ii,:]
            PWo=cc.PW[oo,:]
            PWi=cc.PW[ii,:]
        end
        make_identity_matrix!(cc.S)
        cc.S.+=Koo
        mul!(cc.Tio,Rloc,Kio)
        mul!(cc.Tip,Rloc,Pi)
        mul!(cc.S,Koi,cc.Tio,-one(Complex{T}),one(Complex{T}))
        copyto!(cc.RHS,Po)
        mul!(cc.RHS,Koi,cc.Tip,-one(Complex{T}),one(Complex{T}))
        F=lu!(cc.S)
        ldiv!(cc.Xo,F,cc.RHS)
        copyto!(cc.Xi,cc.Tip)
        mul!(cc.Xi,cc.Tio,cc.Xo,-one(Complex{T}),one(Complex{T}))
        mul!(cc.Rnew,adjoint(PWo),cc.Xo)
        mul!(cc.Rnew,adjoint(PWi),cc.Xi,one(Complex{T}),one(Complex{T}))
        rel=relerr_noalloc(cc.Rnew,Rloc)
        copyto!(Rloc,cc.Rnew)
        rel<stop_tol && break
    end
    return Rloc
end

# Assemble the exact physical corner block corresponding to one RCIP patch.
# This block is subtracted from the global dense matrix before applying the
# compressed inverse correction.
function assemble_physical_K_block_complex!(Kblk::AbstractMatrix{Complex{T}},disc::DLPRCIPDiscretization{T,C},rowids::AbstractVector{Int},colids::AbstractVector{Int},k::ComplexF64,Λ::Matrix{T},ξ::Vector{T},ω::Vector{T};use_panel_logquad::Bool=true) where {T<:Real,C}
    xy=disc.bp.xy;normal=disc.bp.normal;ds=disc.bp.ds;κ=disc.bp.curvature
    panel_id=disc.pdata.panel_id;local_id=disc.pdata.local_id
    fill!(Kblk,zero(Complex{T}))
    @inbounds for β in eachindex(colids)
        j=colids[β]
        xj,yj=xy[j];nxj,nyj=normal[j]
        dsj=ds[j];pj=panel_id[j];jl=local_id[j]
        speed_half_j=dsj/ω[jl]
        for α in eachindex(rowids)
            i=rowids[α]
            xi,yi=xy[i]
            if i==j
                Kblk[α,β]-=Complex{T}(dsj*κ[j]*INV_TWO_PI,zero(T))
            else
                dx=xi-xj;dy=yi-yj
                r=hypot(dx,dy)
                q=(dx*nxj+dy*nyj)/r
                if use_panel_logquad && panel_id[i]==pj
                    il=local_id[i]
                    du=abs(ξ[il]-ξ[jl])
                    full=0.5im*k*H(1,k*r)*q*speed_half_j
                    L1=-(k/pi)*J(1,k*r)*q*speed_half_j
                    L2=full-L1*log(du)
                    Kblk[α,β]-=Λ[il,jl]*L1+ω[jl]*L2
                else
                    Kblk[α,β]-=0.5im*k*H(1,k*r)*q*dsj
                end
            end
        end
    end
    return Kblk
end

# Construct the full RCIP-corrected Fredholm matrices for all contour points.
#   1. Chebyshev dense physical DLP assembly
#   2. sign conversion to Fredholm convention
#   3. exact RCIP compressed inverse per corner
#   4. subtraction of physical corner blocks
#   5. compressed inverse application
#   6. add identity
# Final result: A(k) = I - K_rcip(k)
function construct_rcip_fredholm_chebyshev!(As::Vector{Matrix{ComplexF64}},ws::DLPRCIPWorkspace{T,C,Sym},chebws::DLPRCIPH1J1ChebWorkspace;use_panel_logquad::Bool=true,verbose::Bool=false,multithreaded::Bool=true) where {T<:Real,C,Sym}
    N=boundary_matrix_size(ws.pts)
    Mk=chebws.Mk
    @assert length(As)==Mk
    for m in 1:Mk
        @assert size(As[m])==(N,N)
    end
    assemble_dlp_rcip_chebyshev!(As,ws.pts,chebws,ws.Λ,ws.ξ,ws.ω;symmetry=ws.symmetry,use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
    for m in 1:Mk
        k=chebws.ks[m]
        A=As[m]
        A.*=-1
        for c in eachindex(ws.corners)
            cc=ws.corners[c]
            ids=cc.meta.patch_nodes
            Rloc=ws.Rlocs[c]
            rcip_R_for_corner_complex!(cc,Rloc,k,ws.Λlocal,ws.ξ,ws.ω;use_panel_logquad=use_panel_logquad,multithreaded=multithreaded)
            @views begin
                Kp=view(ws.Kphys,1:length(ids),1:length(ids))
                assemble_physical_K_block_complex!(Kp,ws.pts,ids,ids,k,ws.Λ,ws.ξ,ws.ω;use_panel_logquad=use_panel_logquad)
                A[ids,ids].-=Kp
                Aids=view(A,:,ids)
                tmp=view(ws.Awork,1:N,1:length(ids))
                mul!(tmp,Aids,Rloc)
                copyto!(Aids,tmp)
            end
        end
        @inbounds for i in 1:N
            A[i,i]+=1.0+0.0im
        end
    end
    return As
end

"""
    construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::DLP_rcip,pts::DLPRCIPDiscretization{T,C},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real,C}

Construct RCIP-corrected Fredholm boundary matrices for a collection of
complex contour wavenumbers.

This is the DLP-RCIP backend used by Beyn contour integration for piecewise
smooth geometries with corners. For each contour point `zj[q]`, it
constructs the Fredholm matrix

    A(z_q) = I - K_RCIP(z_q)

where `K_RCIP` is the recursively compressed inverse preconditioned DLP
operator.

The dense physical DLP assembly is accelerated using piecewise-Chebyshev
interpolation of `H₁^(1)(k r)` and `J₁(k r)` across all contour wavenumbers
simultaneously, while the small local RCIP corner recursions are assembled
using exact direct complex-k kernel evaluation.

# Arguments
- `Tbufs::Vector{Matrix{ComplexF64}}`:
  Preallocated output matrices. On return,

      Tbufs[q] = A(zj[q])

  for each contour point.
- `solver::DLP_rcip`:
  DLP-RCIP solver configuration.
- `pts::DLPRCIPDiscretization{T,C}`:
  RCIP boundary discretization.
- `zj::AbstractVector{ComplexF64}`:
  Complex contour wavenumbers.

# Keyword Arguments
- `multithreaded::Bool=true`:
  Enable threaded dense matrix assembly.
- `use_chebyshev::Bool=true`:
  Enable Chebyshev acceleration. Currently this must remain `true`.
- `n_panels_h::Int=15000`:
  Number of radial Chebyshev panels for the Hankel `H₁^(1)` interpolation.
- `M_h::Int=5`:
  Chebyshev polynomial degree for the Hankel interpolation.
- `n_panels_j::Int=3000`:
  Number of radial Chebyshev panels for the Bessel `J₁` interpolation.
- `M_j::Int=5`:
  Chebyshev polynomial degree for the Bessel interpolation.
- `timeit::Bool=false`:
  Enable timing diagnostics.

# Notes
- Intended for complex contour matrix construction (e.g. Beyn contour integration).
- The dense physical operator is Chebyshev accelerated.
- The RCIP compressed-corner correction remains exact (direct kernel assembly).
- Direct non-Chebyshev complex-k RCIP construction is currently not implemented.
"""
function construct_boundary_matrices!(Tbufs::Vector{Matrix{ComplexF64}},solver::DLP_rcip,pts::DLPRCIPDiscretization{T,C},zj::AbstractVector{ComplexF64};multithreaded::Bool=true,use_chebyshev::Bool=true,n_panels_h::Int=15000,M_h::Int=5,n_panels_j::Int=3000,M_j::Int=5,timeit::Bool=false) where {T<:Real,C}
    use_chebyshev || error("Direct complex-k DLP-RCIP construction is not implemented here; use the Chebyshev route.")
    length(Tbufs)==length(zj) || error("length(Tbufs) must match length(zj).")
    @blas_1 begin
        @benchit timeit=timeit "DLP-RCIP workspace" ws=make_dlp_rcip_workspace(solver,pts)
        N=boundary_matrix_size(pts)
        @inbounds for q in eachindex(Tbufs)
            @assert size(Tbufs[q])==(N,N) "Tbufs[$q] has size $(size(Tbufs[q])), expected ($N,$N)."
            fill!(Tbufs[q],0.0+0.0im)
        end
        @benchit timeit=timeit "DLP-RCIP H1/J1 Chebyshev workspace" chebws=build_dlp_rcip_cheb_workspace(solver,pts,zj;npanels_h=n_panels_h,npanels_j=n_panels_j,M_h=M_h,M_j=M_j,plan_nthreads=Threads.nthreads(),ntls=Threads.nthreads())
        @benchit timeit=timeit "DLP-RCIP H1/J1 Chebyshev assembly" construct_rcip_fredholm_chebyshev!(Tbufs,ws,chebws;use_panel_logquad=solver.use_panel_logquad,multithreaded=multithreaded)
    end
    return nothing
end