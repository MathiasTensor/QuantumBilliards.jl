# ===============================================================================
# HYPERBOLIC WAVEFUNCTION RECONSTRUCTION (POINCARÉ DISK, SLP FORM)
# ===============================================================================
#
# PURPOSE
#   Reconstruct interior eigenfunctions ψ(x,y) of the hyperbolic Laplacian
#   (Helmholtz/Laplace–Beltrami) on a billiard embedded in the Poincaré disk,
#   using a single-layer potential (SLP) integral representation.
#
# GEOMETRY + METRIC CONVENTIONS
#   • Coordinates: Euclidean coordinates (x,y) in unit disk: x^2+y^2<1.
#   • Poincaré metric: ds_H = λ(x,y) ds_E with λ(x,y)=2/(1-(x^2+y^2)).
#   • Boundary discretization uses Euclidean arclength weights bd.ds ≈ ds_E.
#   • Hyperbolic boundary element is inserted explicitly: ds_H=λ(q)*bd.ds.
#
# KERNEL CONVENTION
#   • Hyperbolic Green kernel (fundamental solution):
#       G(x,q)=(1/2π)Q_ν(cosh d(x,q)),
#     with ν=-1/2+ik (or your internal convention in QTaylorTable).
#   • We evaluate Q via QuantumBilliards._eval_Q(tab,d) with d=hyperbolic distance.
#
# DENSITY CONVENTION (DIRICHLET EIGENPROBLEM)
#   • For Dirichlet ψ|∂Ω=0, the standard Green identity gives representation
#       ψ(x)=∮_∂Ω G(x,q) u(q) ds_H(q),
#     where u(q)=∂_n ψ(q) (normal derivative on boundary).
#   • Here vec_u[i] holds u(s) samples at boundary quadrature nodes.
#
# SYMMETRY CONVENTION
#   • Only origin-based discrete symmetries are used (safe in Poincaré disk):
#       - Rotations about origin
#       - Reflections across x-axis, y-axis, or through origin
#   • No shifted centers (cx,cy) appear.
#
# OUTPUT CONVENTION
#   • Returns ψ(x,y) on a Cartesian grid (xgrid,ygrid) inside the disk (and optionally inside Ω).
#   • The returned matrices are Complex{T} (wavefunction values), not |ψ|^2.
#
# NUMERICAL SAFETY
#   • δdisk excludes points too close to unit circle to avoid λ blowup and d issues.
#   • λ denominator is clamped at 1e-15.
#
# PERFORMANCE NOTES
#   • Complexity per eigenfunction:
#       O(Ngrid_eval * Nbd) kernel evaluations (dominant).
#   • Uses Threads.@threads on grid points; each grid point loops over boundary nodes.
#   • Pre-extract boundary coordinates qx,qy to avoid SVector indexing cost.
#
# ===============================================================================


#--------------------------------------------------------------------------
# Origin-based symmetry helpers (valid in Poincaré disk)
#--------------------------------------------------------------------------
# rot_point : rotate a point around the origin
# rot_vec   : rotate a vector around the origin
# refl_*    : reflections through coordinate axes / origin
#--------------------------------------------------------------------------
@inline rot_point(x::T,y::T,cθ::T,sθ::T)where{T<:Real}=(cθ*x-sθ*y,sθ*x+cθ*y)
@inline rot_vec(nx::T,ny::T,cθ::T,sθ::T)where{T<:Real}=(cθ*nx-sθ*ny,sθ*nx+cθ*ny)
@inline refl_x_point(x::T,y::T)where{T<:Real}=(-x,y)
@inline refl_y_point(x::T,y::T)where{T<:Real}=(x,-y)
@inline refl_xy_point(x::T,y::T)where{T<:Real}=(-x,-y)
@inline refl_x_vec(nx::T,ny::T)where{T<:Real}=(-nx,ny)
@inline refl_y_vec(nx::T,ny::T)where{T<:Real}=(nx,-ny)
@inline refl_xy_vec(nx::T,ny::T)where{T<:Real}=(-nx,-ny)

#------------------------------------------------------------------------------
# λ_poincare(x,y)->λ
#
# PURPOSE
#   Conformal factor of Poincaré disk metric:
#     ds_H = λ(x,y) ds_E
#
# INPUTS
#   x,y : coordinates in unit disk (Real)
#
# OUTPUTS
#   λ   : Real scalar >=2, grows as r->1.
#
# SAFETY
#   Denominator clamped to >=1e-14 to avoid Inf/NaN at r≈1.
#
# USED FOR
#   Hyperbolic boundary measure:
#     ds_H(q)=λ(q)*bd.ds(q).
#------------------------------------------------------------------------------
@inline function λ_poincare(x::T,y::T) where {T<:Real}
    den=one(T)-muladd(x,x,y*y)
    den=max(den,T(1e-14))
    return T(2)/den
end

#------------------------------------------------------------------------------
# λ2_poincare(x,y)->λ²
#
# PURPOSE
#   Square of the Poincaré conformal factor appearing in the hyperbolic metric.
#
# INPUTS
#   x,y : Euclidean coordinates in the unit disk (Real)
#         Must satisfy x²+y²<1 for physical points.
#
# OUTPUTS
#   λ²  : Real scalar >=4, diverging as r→1⁻.
#------------------------------------------------------------------------------
@inline function λ2_poincare(x::T,y::T)where{T<:Real}
    den=one(T)-muladd(x,x,y*y)
    den=max(den,T(1e-14))
    lam=T(2)/den
    lam*lam
end

#------------------------------------------------------------------------------
# prepare_hyp_bd_xy(bd)->(qx,qy)
#
# PURPOSE
#   Extract boundary node coordinates into two dense vectors to reduce overhead.
#
# INPUTS
#   bd::BoundaryPointsHypBIM{T}
#     Required fields:
#       bd.xy :: Vector{SVector{2,T}} (or similar)
#       bd.ds :: Vector{T}            (Euclidean ds weights)
#
# OUTPUTS
#   qx::Vector{T} : x-coordinates of boundary nodes
#   qy::Vector{T} : y-coordinates of boundary nodes
#
# NOTES
#   • Does NOT compute λ or ds_H.
#   • Hyperbolic ds is always inserted later as ds[j]*λ_poincare(qxj,qyj).
#------------------------------------------------------------------------------
function prepare_hyp_bd_xy(bd::BoundaryPointsHypBIM{T})where{T<:Real}
    N=length(bd.xy)
    qx=Vector{T}(undef,N);qy=Vector{T}(undef,N)
    @inbounds for j in 1:N
        qx[j]=bd.xy[j][1];qy[j]=bd.xy[j][2]
    end
    return qx,qy
end

#------------------------------------------------------------------------------
# slp_kernel_hyp(tab,x,y,xp,yp)->G
#
# PURPOSE
#   Evaluate hyperbolic Green function kernel G(x,q) for SLP.
#
# INPUTS
#   tab::QTaylorTable
#     Precomputed table encoding Q_ν(cosh d) evaluation at this k (ν depends on k).
#   x,y   : target point in domain (Real)
#   xp,yp : source boundary point q (Real)
#
# OUTPUTS
#   G::Complex{T} or Complex{Float64} (depends on _eval_Q)
#     Value of G(x,q)=(1/2π)Q_ν(cosh d(x,q)).
#------------------------------------------------------------------------------
@inline function slp_kernel_hyp(tab::QTaylorTable,x::T,y::T,xp::T,yp::T)where{T<:Real}
    d=Float64(hyperbolic_distance_poincare(x,y,xp,yp))
    return QuantumBilliards._eval_Q(tab,d)/TWO_PI
end

#------------------------------------------------------------------------------
# ψ_hyp_slp(x,y,tab,bd,qx,qy,σ)->ψ
#
# PURPOSE
#   Evaluate interior wavefunction ψ(x) by SLP integral:
#     ψ(x)=∮_∂Ω G(x,q) σ(q) ds_H(q),
#   where ds_H(q)=λ(q)ds_E(q) and bd.ds stores ds_E weights.
#
# INPUTS
#   x,y : target point
#   tab : QTaylorTable for the eigen-wavenumber
#   bd  : BoundaryPointsHypBIM{T} providing bd.ds (Euclidean quadrature weights)
#   qx,qy : boundary node coordinate arrays from prepare_hyp_bd_xy
#   σ  : boundary density samples
#        For your Dirichlet eigenproblem: σ(s)=u(s)=∂_nψ(s).
#
# OUTPUTS
#   ψ(x,y) as Complex{T}
#------------------------------------------------------------------------------
@inline function ψ_hyp_slp(x::T,y::T,tab::QTaylorTable,bd::BoundaryPointsHypBIM{T},qx::AbstractVector{T},qy::AbstractVector{T},σ::AbstractVector{Complex{T}})where{T<:Real}
    ds=bd.ds;sr=zero(T);si=zero(T)
    @inbounds for j in eachindex(σ)
        qxj=qx[j];qyj=qy[j]
        w=slp_kernel_hyp(tab,x,y,qxj,qyj)*(ds[j]*λ_poincare(qxj,qyj))
        σj=σ[j];wr,wi=real(w),imag(w);σr,σi=real(σj),imag(σj)
        sr=muladd(wr,σr,sr)-wi*σi
        si=muladd(wr,σi,si)+wi*σr
    end
    return Complex(sr,si)
end

#------------------------------------------------------------------------------
# ψ_hyp_slp (real density overload)
#
# Same as above but σ is real-valued; output still Complex{T} because kernel is complex.
#------------------------------------------------------------------------------
@inline function ψ_hyp_slp(x::T,y::T,tab::QTaylorTable,bd::BoundaryPointsHypBIM{T},qx::AbstractVector{T},qy::AbstractVector{T},σ::AbstractVector{T})where{T<:Real}
    ds=bd.ds;acc=zero(Complex{T})
    @inbounds for j in eachindex(σ)
        qxj=qx[j];qyj=qy[j]
        w=slp_kernel_hyp(tab,x,y,qxj,qyj)*(ds[j]*λ_poincare(qxj,qyj))
        acc+=w*σ[j]
    end
    return acc
end

#------------------------------------------------------------------------------
# ψ_hyp_slp_sym(x,y,tab,bd,qx,qy,σ,sym)->ψ
#
# PURPOSE
#   Same as ψ_hyp_slp, but includes symmetry images of the boundary.
#
# INPUTS
#   sym : one of:
#     • QuantumBilliards.Rotation(n,m)   (about origin)
#     • Reflection with axis in {:origin,:x_axis,:y_axis}
#   σ   : boundary density in the fundamental domain
#   bd.ds: Euclidean ds weights associated with those fundamental boundary nodes
#
# OUTPUTS
#   ψ(x,y) in the full (symmetrized) domain satisfying required symmetry sector.
#------------------------------------------------------------------------------
function ψ_hyp_slp_sym(x::T,y::T,tab::QTaylorTable,bd::BoundaryPointsHypBIM{T},qx::AbstractVector{T},qy::AbstractVector{T},σ::AbstractVector{Num},sym)where{T<:Real,Num<:Number}
    ds=bd.ds;acc=zero(Complex{T})
    if sym isa QuantumBilliards.Rotation
        n=sym.n;m=mod(sym.m,n)
        @inbounds for j in eachindex(σ)
            qxj=qx[j];qyj=qy[j];σj=σ[j];dsj=ds[j]
            for ℓ in 1:n
                θ=T(2π)*T(ℓ-1)/T(n);cθ=cos(θ);sθ=sin(θ)
                qxr,qyr=rot_point(qxj,qyj,cθ,sθ)
                phase=Complex{T}(cis(T(2π)*T(m)*T(ℓ-1)/T(n)))
                w=slp_kernel_hyp(tab,x,y,qxr,qyr)*(dsj*λ_poincare(qxr,qyr))
                acc+=phase*w*σj
            end
        end
        return acc
    end
    if sym isa Reflection
        if sym.axis==:origin
            px,py=sym.parity;sx=(px<0 ? -one(T) : one(T));sy=(py<0 ? -one(T) : one(T))
            @inbounds for j in eachindex(σ)
                qxj=qx[j];qyj=qy[j];σj=σ[j];dsj=ds[j]
                w=slp_kernel_hyp(tab,x,y,qxj,qyj)*(dsj*λ_poincare(qxj,qyj));acc+=w*σj
                qx1,qy1=refl_x_point(qxj,qyj);w=slp_kernel_hyp(tab,x,y,qx1,qy1)*(dsj*λ_poincare(qx1,qy1));acc+=sx*w*σj
                qx2,qy2=refl_y_point(qxj,qyj);w=slp_kernel_hyp(tab,x,y,qx2,qy2)*(dsj*λ_poincare(qx2,qy2));acc+=sy*w*σj
                qx3,qy3=refl_xy_point(qxj,qyj);w=slp_kernel_hyp(tab,x,y,qx3,qy3)*(dsj*λ_poincare(qx3,qy3));acc+=(sx*sy)*w*σj
            end
            return acc
        elseif sym.axis==:y_axis
            sgn=(sym.parity==-1 ? -one(T) : one(T))
            @inbounds for j in eachindex(σ)
                qxj=qx[j];qyj=qy[j];σj=σ[j];dsj=ds[j]
                w=slp_kernel_hyp(tab,x,y,qxj,qyj)*(dsj*λ_poincare(qxj,qyj));acc+=w*σj
                qx1,qy1=refl_x_point(qxj,qyj);w=slp_kernel_hyp(tab,x,y,qx1,qy1)*(dsj*λ_poincare(qx1,qy1));acc+=sgn*w*σj
            end
            return acc
        elseif sym.axis==:x_axis
            sgn=(sym.parity==-1 ? -one(T) : one(T))
            @inbounds for j in eachindex(σ)
                qxj=qx[j];qyj=qy[j];σj=σ[j];dsj=ds[j]
                w=slp_kernel_hyp(tab,x,y,qxj,qyj)*(dsj*λ_poincare(qxj,qyj));acc+=w*σj
                qx2,qy2=refl_y_point(qxj,qyj);w=slp_kernel_hyp(tab,x,y,qx2,qy2)*(dsj*λ_poincare(qx2,qy2));acc+=sgn*w*σj
            end
            return acc
        end
    end
    error("Unsupported symmetry type:$(typeof(sym))")
end

#------------------------------------------------------------------------------
# _make_grid_and_idxs_for_billiard(ks,billiard;...)->(xgrid,ygrid,idxs,nx,ny)
#
# PURPOSE
#   Build a Cartesian evaluation grid and return the linear indices of points
#   where ψ should be evaluated.
#
# INPUTS
#   ks::Vector{T}
#     Eigen-wavenumbers; only max(ks) is used to set grid resolution.
#
#   billiard
#     Must provide:
#       billiard.length
#       billiard.fundamental_boundary
#       billiard.full_boundary
#
# KEYWORD INPUTS
#   b::Float64
#     Resolution parameter similar to your BIM rule-of-thumb.
#   fundamental::Bool
#     If true: bounding box computed from billiard.fundamental_boundary.
#     Else:    from billiard.full_boundary.
#   inside_only::Bool
#     If true: idxs only include points inside billiard polygon (within chosen domain).
#     If false: idxs include all points inside Poincaré disk (within bounding box).
#   δdisk::T
#     Exclude points with r^2 >= 1-δdisk.
#
# OUTPUTS
#   xgrid::Vector{T},ygrid::Vector{T}
#     Coordinate grids; lengths nx and ny.
#   idxs::Vector{Int}
#     Linear indices into an nx*ny flattened array (column-major with ix+(jy-1)*nx).
#   nx::Int,ny::Int
#     Grid sizes.
#
# NOTES
#   • This function does not allocate pts for all nx*ny; it only constructs pts_disk
#     for disk indices if inside_only=true (to test polygon membership).
#------------------------------------------------------------------------------
function _make_grid_and_idxs_for_billiard(ks::Vector{T},billiard::Bi;b::Float64=5.0,fundamental::Bool=true,inside_only::Bool=true,δdisk::T=T(1e-10))where{T<:Real,Bi<:AbsBilliard}
    kmax=maximum(ks);L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,kmax*L*b/(2π))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,kmax*L*b/(2π))))
    end
    dx=xlim[2]-xlim[1];dy=ylim[2]-ylim[1]
    nx=max(round(Int,kmax*dx*b/(2π)),512)
    ny=max(round(Int,kmax*dy*b/(2π)),512)
    xgrid=collect(T,range(xlim...,nx))
    ygrid=collect(T,range(ylim...,ny))
    r2max=one(T)-δdisk
    disk_idxs=Int[];sizehint!(disk_idxs,nx*ny)
    @inbounds for jy in 1:ny
        y=ygrid[jy]
        for ix in 1:nx
            x=xgrid[ix]
            r2=muladd(x,x,y*y)
            r2<r2max && push!(disk_idxs,ix+(jy-1)*nx)
        end
    end
    if !inside_only
        return xgrid,ygrid,disk_idxs,nx,ny
    end
    Nd=length(disk_idxs)
    pts_disk=Vector{SVector{2,T}}(undef,Nd)
    @inbounds for t in 1:Nd
        idx=disk_idxs[t];ix=(idx-1)%nx+1;jy=(idx-1)÷nx+1
        pts_disk[t]=SVector(xgrid[ix],ygrid[jy])
    end
    mask_in=points_in_billiard_polygon(pts_disk,billiard,round(Int,sqrt(Nd));fundamental_domain=fundamental)
    idxs=Int[];sizehint!(idxs,Nd)
    @inbounds for t in 1:Nd
        mask_in[t] && push!(idxs,disk_idxs[t])
    end
    return xgrid,ygrid,idxs,nx,ny
end

#------------------------------------------------------------------------------
# wavefunction_multi_hyp(ks,vec_u,vec_bd,tabs,billiard;...)->(Psi2ds,xgrid,ygrid)
#
# PURPOSE
#   Compute wavefunctions ψ_i(x,y) on a grid for each eigen-wavenumber k_i.
#
# INPUTS
#   ks::Vector{T}
#     List of eigen-wavenumbers (typically real Float64) used for grid sizing and loop.
#
#   vec_u::Vector{<:AbstractVector}
#     Boundary density for each eigenvalue:
#       vec_u[i][j] ≈ u(s_j)=∂_n ψ_i(s_j).
#     Length of vec_u[i] must match number of boundary nodes in vec_bd[i].
#
#   vec_bd
#     Vector of BoundaryPointsHypBIM objects, one per eigenvalue.
#     Must provide fields:
#       bd.xy :: boundary nodes (Euclidean coordinates)
#       bd.ds :: Euclidean quadrature weights ds_E at nodes
#
#   tabs
#     Vector of QTaylorTable, one per eigenvalue, tabulating Q_ν(cosh d)
#     (and consistent with that k).
#
#   billiard
#     Billiard definition, used only to construct grid + inside mask.
#
# KEYWORD INPUTS
#   b::Float64
#     Controls grid resolution; larger -> finer grid.
#   inside_only::Bool
#     If true: evaluate only for points inside billiard polygon.
#   fundamental::Bool
#     If true: grid bounding box determined from fundamental boundary.
#   symmetry
#     If nothing: evaluate without images.
#     Else: pass a symmetry object handled by ψ_hyp_slp_sym.
#   MIN_CHUNK::Int
#     Controls number of grid points per thread (static schedule).
#   δdisk::T
#     Safety exclusion near r=1.
#
# OUTPUTS
#   Psi2ds::Vector{Matrix{Complex{T}}}
#     Psi2ds[i] is an nx×ny array holding ψ_i evaluated on the grid.
#     Values at grid points not in idxs remain 0+0im (since Psi_flat initialized to zero).
#
#   xgrid::Vector{T},ygrid::Vector{T}
#     The coordinate vectors defining the Cartesian grid.
#------------------------------------------------------------------------------
function wavefunction_multi_hyp(ks::Vector{T},vec_u::Vector{<:AbstractVector},vec_bd::Vector{BoundaryPointsHypBIM},tabs::Vector{QTaylorTable},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=true,symmetry=nothing,MIN_CHUNK::Int=4096,δdisk::T=T(1e-10))where{T<:Real,Bi<:AbsBilliard}
    _psi(x,y,tab,bd,qx,qy,u,symmetry)=symmetry===nothing ? ψ_hyp_slp(x,y,tab,bd,qx,qy,u) : ψ_hyp_slp_sym(x,y,tab,bd,qx,qy,u,symmetry)
    xgrid,ygrid,idxs,nx,ny=_make_grid_and_idxs_for_billiard(ks,billiard;b=b,fundamental=fundamental,inside_only=inside_only,δdisk=δdisk)
    nmask=length(idxs);NT=Threads.nthreads();NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)));q,r=divrem(nmask,NT_eff)
    Psi_flat=zeros(Complex{T},nx*ny)
    Psi2ds=Vector{Matrix{Complex{T}}}(undef,length(ks))
    prog=Progress(length(ks),desc="Constructing hyperbolic SLP wavefunctions...")
    @inbounds for i in eachindex(ks)
        fill!(Psi_flat,zero(Complex{T}))
        tab=tabs[i];bd=vec_bd[i];u=vec_u[i]
        qx,qy=prepare_hyp_bd_xy(bd)
        Threads.@threads :static for t in 1:NT_eff
            lo=(t-1)*q+min(t-1,r)+1
            hi=lo+q-1+(t<=r ? 1 : 0)
            for jj in lo:hi
                idx=idxs[jj]
                ix=(idx-1)%nx+1
                jy=(idx-1)÷nx+1
                x=xgrid[ix];y=ygrid[jy]
                Psi_flat[idx]=_psi(x,y,tab,bd,qx,qy,u,symmetry)
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(prog)
    end
    return Psi2ds,xgrid,ygrid
end

# ------------------------------------------------------------------------------
# wavefunction_multi_with_husimi_hyp(ks,vec_u,vec_bd,tabs,billiard,qs,ps;...)
#
# PURPOSE
#   Compute (i) hyperbolic SLP wavefunctions ψ_i(x,y) on a Cartesian grid and
#   (ii) boundary Poincaré–Husimi densities H_i(q,p) for the same eigenstates.
#
# INPUTS
#   ks::Vector{T}
#     Eigen-wavenumbers (one per state).
#
#   vec_u::Vector{<:AbstractVector}
#     Boundary densities u_i on ∂Ω (Dirichlet case: u_i=∂nψ_i at boundary nodes).
#     Length(vec_u[i]) must match the boundary node count of vec_bd[i].
#
#   vec_bd::Vector{BoundaryPointsHypBIM}
#     Boundary containers (one per state). Required fields:
#       bd.xy  : Euclidean boundary nodes (for ψ reconstruction)
#       bd.ds  : Euclidean ds weights (for ψ reconstruction, combined with λ)
#       bd.ξ   : hyperbolic arclength coordinate (monotone) for Husimi
#       bd.LH  : total hyperbolic boundary length
#       bd.dsH : hyperbolic quadrature weights for Husimi (ds_H)
#
#   tabs::Vector{QTaylorTable}
#     One QTaylorTable per state (must correspond to ks[i]).
#
#   billiard
#     Used for grid construction / inside mask in wavefunction_multi_hyp.
#
#   qs,ps
#     Boundary phase-space grids for Husimi:
#       qs::AbstractVector{<:AbstractVector{T}}  (q grid per state)
#       ps::AbstractVector{<:AbstractVector{T}}  (p grid per state)
#
# KEYWORDS
#   b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=true,symmetry=nothing,
#   MIN_CHUNK::Int=4096,δdisk::T=T(1e-10)
#     Passed to wavefunction_multi_hyp.
#
#   full_p::Bool=false
#     Husimi construction based on time-reversal symmetry:
#       full_p=true  -> H is (q×p) with pgrid=ps[i]
#       full_p=false -> H is (q×p_out) with p_out=[-reverse(ps[i])[1:end-1]; ps[i]]
#
#   show_progress_husimi::Bool=true
#     If true, show progress bar for Husimi computation (thread-safe).
#
# OUTPUTS
#   Psi2ds::Vector{Matrix{Complex{T}}}
#     Wavefunctions ψ_i on the (xgrid,ygrid) grid (nx×ny).
#
#   xgrid::Vector{T},ygrid::Vector{T}
#
#   Hs::Vector{Matrix{T}}
#     Husimi matrices for successful states (failed states dropped).
#
#   qs_out::Vector{<:AbstractVector{T}}
#     The q grids used (returned as-is, aligned with Hs).
#
#   ps_out::Vector{<:AbstractVector{T}}
#     The p grids actually used (ps or the reflected p_out), aligned with Hs.
# ------------------------------------------------------------------------------
function wavefunction_multi_with_husimi_hyp(ks::Vector{T},vec_u::Vector{<:AbstractVector},vec_bd::Vector{BoundaryPointsHypBIM},tabs::Vector{QTaylorTable},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=true,symmetry=nothing,MIN_CHUNK::Int=4096,δdisk::T=T(1e-10),full_p::Bool=false,show_progress_husimi::Bool=true,q_oversample::Float64=2.0,nq_min::Int=1000,np::Int=1000,pmax::T=one(T)) where {T<:Real,Bi<:AbsBilliard}
    Psi2ds,xgrid,ygrid=wavefunction_multi_hyp(ks,vec_u,vec_bd,tabs,billiard;b=b,inside_only=inside_only,fundamental=fundamental,symmetry=symmetry,MIN_CHUNK=MIN_CHUNK,δdisk=δdisk)
    n=length(ks)
    Hs=Vector{Matrix{T}}(undef,n)
    qs_out=Vector{Vector{T}}(undef,n)
    ps_out=Vector{Vector{T}}(undef,n)
    ok=trues(n)
    pbar=show_progress_husimi ? Progress(n;desc="Husimi N=$n") : nothing
    Threads.@threads for i in 1:n
        try
            k=ks[i]
            bd=vec_bd[i]
            LH=bd.LH
            Nq=max(nq_min,ceil(Int,q_oversample*k*LH/(2π)))
            qgrid=collect(range(zero(T),LH;length=Nq+1))[1:end-1]  # periodic: drop endpoint
            pgrid = full_p ? collect(range(-pmax,pmax;length=np)) : collect(range(zero(T),pmax;length=np))
            Np=length(pgrid)
            H=full_p ? Matrix{T}(undef,Nq,Np) : Matrix{T}(undef,Nq,2*Np-1)
            _husimi_on_grid_hyp!(H,k,bd.ξ,vec_u[i],LH,bd.dsH,qgrid,pgrid;full_p=full_p)
            Hs[i]=H
            qs_out[i]=qgrid
            ps_out[i]=full_p ? pgrid : vcat(-reverse(pgrid)[1:end-1],pgrid) 
        catch
            ok[i]=false
        end
        if show_progress_husimi
            next!(pbar)
        end
    end

    return Psi2ds,xgrid,ygrid,Hs[ok],qs_out[ok],ps_out[ok]
end

#------------------------------------------------------------------------------
# normalize_psi_hyperbolic!(ψ,xgrid,ygrid)->ψ
#
# PURPOSE
#   In-place L² normalization of a wavefunction on a Cartesian grid using the
#   HYPERBOLIC area element of the Poincaré disk.
#
#   Hyperbolic area element:
#     dA_H = λ(x,y)² dx dy,
#     λ(x,y)=2/(1-(x²+y²)).
#
#   This rescales ψ so that:
#     ∫_Ω |ψ|² dA_H ≈ 1
#   where the integral is approximated by a Riemann sum on the provided grid:
#     Σ_{i,j} |ψ[j,i]|² λ(x_i,y_j)² dx dy.
#
# INPUTS
#   ψ::AbstractMatrix{Num}
#     Wavefunction values on the grid, size (ny×nx).
#
#   xgrid::AbstractVector{T}
#     Grid x-coordinates (length nx), assumed uniformly spaced.
#
#   ygrid::AbstractVector{T}
#     Grid y-coordinates (length ny), assumed uniformly spaced.
#
# OUTPUTS
#     The same array object, modified in-place:
#       ψ .= ψ / sqrt(normH),
#     where normH is the hyperbolic L² norm computed from the grid.
#------------------------------------------------------------------------------
function normalize_psi_hyperbolic!(ψ::AbstractMatrix{Num},xgrid::AbstractVector{T},ygrid::AbstractVector{T}) where{Num<:Number,T<:Real}
    dx=xgrid[2]-xgrid[1]
    dy=ygrid[2]-ygrid[1]
    s=0.0
    @inbounds for j in eachindex(ygrid), i in eachindex(xgrid)
        x=xgrid[i];y=ygrid[j]
        val=ψ[j,i]
        if !isnan(real(val)) && !isnan(imag(val))
            s+=abs2(val)*λ2_poincare(x,y)
        end
    end
    s*=dx*dy
    ψ./=sqrt(s+eps())
    return ψ
end