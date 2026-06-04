"""

    HypSLPData{T<:Real}

Cache for hyperbolic single-layer-potential reconstruction.
Fields:
- `qx::Vector{T}`: Boundary x-coordinates.
- `qy::Vector{T}`: Boundary y-coordinates.
- `w::Vector{T}`: Quadrature weights paired with the supplied boundary vector.
Weight convention:
- `u_constructed_from=:source_normal`: use `w=dsH`. This is for the 
  Beyn/source-density reconstruction route.
- `u_constructed_from=:target_normal`: use `w=ds`. This is for adjoint
  boundary functions `u=∂ₙᴱψ`.
Runtime formula:
    ψ(x) = Σ_j G_k(x,q_j) u_j w_j
"""
struct HypSLPData{T<:Real}
    qx::Vector{T}
    qy::Vector{T}
    w::Vector{T}
end

"""
    λ_poincare(x::T,y::T) where {T<:Real}

Evaluate the Poincaré conformal factor

    λ(x,y)=2/(1-x²-y²).

Inputs

`x::T`, `y::T`
    Euclidean coordinates inside the unit disk.

Output

`λ::T`
    Metric conformal factor satisfying `ds_H = λ ds_E`.

Numerical safety

The denominator is clamped to avoid overflow near the unit circle.
"""
@inline function λ_poincare(x::T,y::T) where {T<:Real}
    den=max(one(T)-muladd(x,x,y*y),T(1e-14))
    return T(2)/den
end

"""
    λ2_poincare(x::T,y::T) where {T<:Real}

Evaluate the square of the Poincaré conformal factor.

Inputs

`x::T`, `y::T`
    Euclidean coordinates inside the unit disk.

Output

`λ²::T`
    Hyperbolic area density factor satisfying `dA_H = λ² dxdy`.
"""
@inline function λ2_poincare(x::T,y::T) where {T<:Real}
    λ=λ_poincare(x,y)
    return λ*λ
end

@inline function _hyp_slp_weight(bd::HyperbolicBeynPoints,j::Int,u_constructed_from::Symbol)
    if u_constructed_from===:target_normal
        return bd.ds[j]
    elseif u_constructed_from===:source_normal
        return bd.dsH[j]
    else
        error("Unknown u_constructed_from=$u_constructed_from. Use :target_normal or :source_normal.")
    end
end

"""

    prepare_hyp_slp_data(obj::HyperbolicBeynPoints; u_constructed_from=:source_normal) -> HypSLPData{T}

Precompute boundary coordinates and reconstruction weights for hyperbolic SLP
wavefunction evaluation.
Inputs:
- `obj::HyperbolicBeynPoints`: Boundary discretization or wrapper.
- `u_constructed_from::Symbol`: Convention of the supplied boundary vector.
Conventions:
- `:source_normal`: `u` is the direct Beyn/source-normal density from the primal
  matrix. Uses `w=dsH`.
- `:target_normal`: `u` is the adjoint boundary function `∂ₙᴱψ`. Uses `w=ds`.
Output:
- `HypSLPData{T}` with cached `qx`, `qy`, and selected weights `w`.
"""
function prepare_hyp_slp_data(obj::HyperbolicBeynPoints;u_constructed_from::Symbol=:source_normal)
    bd=hyp_bp(obj)
    T=eltype(bd.ds)
    N=length(bd.xy)
    qx=Vector{T}(undef,N)
    qy=Vector{T}(undef,N)
    w=Vector{T}(undef,N)
    @inbounds for j in 1:N
        qx[j]=bd.xy[j][1]
        qy[j]=bd.xy[j][2]
        w[j]=_hyp_slp_weight(bd,j,u_constructed_from)
    end
    return HypSLPData(qx,qy,w)
end

"""
    slp_kernel_hyp(tab::QTaylorTable,x::T,y::T,xp::T,yp::T) where {T<:Real}

Evaluate the hyperbolic Green kernel

    G_k(x,q)=Q_ν(cosh d_H(x,q))/(2π),

where `ν=-1/2+ik`.

Inputs

`tab::QTaylorTable`
    Patched Taylor table for Legendre-Q evaluation.

`x::T`, `y::T`
    Target point.

`xp::T`, `yp::T`
    Source boundary point.

Output

`G::Complex`
    Complex Green kernel value.

Numerical safety

Returns zero if the hyperbolic distance is non-finite or non-positive.
"""
@inline function slp_kernel_hyp(tab::QTaylorTable,x::T,y::T,xp::T,yp::T) where {T<:Real}
    d=hyperbolic_distance_poincare(x,y,xp,yp)
    (!isfinite(d)||d<=zero(T))&&return zero(Complex{T})
    return _eval_Q(tab,Float64(d))/TWO_PI
end

"""
    ψ_hyp_slp_fast(x::T,y::T,tab::QTaylorTable,data::HypSLPData{T},u::AbstractVector{Complex{T}}) where {T<:Real}

Evaluate the hyperbolic SLP reconstruction at one point.

Inputs

`x::T`, `y::T`
    Interior target point.

`tab::QTaylorTable`
    Patched Legendre-Q table for the current eigenvalue.

`data::HypSLPData{T}`
    Cached boundary coordinates and hyperbolic quadrature weights.

`u::AbstractVector{Complex{T}}`
    Boundary density samples, usually `u≈∂ₙψ`.

Output

`ψ::Complex{T}`
    Reconstructed wavefunction value.

Performance

The loop is allocation-free and uses fused real/imaginary accumulation.
"""
@inline function ψ_hyp_slp_fast(x::T,y::T,tab::QTaylorTable,data::HypSLPData{T},u::AbstractVector{Complex{T}}) where {T<:Real}
    sr=zero(T)
    si=zero(T)
    qx=data.qx
    qy=data.qy
    w=data.w
    @inbounds @simd for j in eachindex(u)
        G=slp_kernel_hyp(tab,x,y,qx[j],qy[j])*w[j]
        Gr=real(G);Gi=imag(G)
        ur=real(u[j]);ui=imag(u[j])
        sr=muladd(Gr,ur,sr)-Gi*ui
        si=muladd(Gr,ui,si)+Gi*ur
    end
    return Complex(sr,si)
end

"""
    ψ_hyp_slp_fast(x::T,y::T,tab::QTaylorTable,data::HypSLPData{T},u::AbstractVector{T}) where {T<:Real}

Real-density overload of `ψ_hyp_slp_fast`.

Inputs

`x::T`, `y::T`
    Interior target point.

`tab::QTaylorTable`
    Patched Legendre-Q table.

`data::HypSLPData{T}`
    Cached boundary data.

`u::AbstractVector{T}`
    Real-valued boundary density.

Output

`ψ::Complex{T}`
    Complex reconstructed wavefunction value.
"""
@inline function ψ_hyp_slp_fast(x::T,y::T,tab::QTaylorTable,data::HypSLPData{T},u::AbstractVector{T}) where {T<:Real}
    acc=zero(Complex{T})
    qx=data.qx
    qy=data.qy
    w=data.w
    @inbounds @simd for j in eachindex(u)
        acc+=slp_kernel_hyp(tab,x,y,qx[j],qy[j])*(w[j]*u[j])
    end
    return acc
end

"""
    _make_grid_and_idxs_for_billiard(ks::Vector{T},billiard::Bi; b=5.0,fundamental=true,inside_only=true,δdisk=T(1e-10)) where {T<:Real,Bi<:AbsBilliard}

Construct the Cartesian wavefunction grid and active evaluation indices.

Inputs

`ks::Vector{T}`
    Eigen-wavenumbers. The maximum value controls the grid resolution.

`billiard::Bi`
    Billiard geometry.

Keyword arguments

`b::Float64`
    Grid resolution parameter.

`fundamental::Bool`
    Use the fundamental boundary if true, otherwise the full boundary.

`inside_only::Bool`
    Keep only points inside the billiard if true.

`δdisk::T`
    Exclude points with `x²+y² ≥ 1-δdisk`.

Outputs

`xgrid::Vector{T}`, `ygrid::Vector{T}`
    Cartesian grid vectors.

`idxs::Vector{Int}`
    Linear indices of active evaluation points.

`nx::Int`, `ny::Int`
    Grid dimensions.
"""
function _make_grid_and_idxs_for_billiard(ks::Vector{T},billiard::Bi;b::Float64=5.0,fundamental::Bool=true,inside_only::Bool=true,δdisk::T=T(1e-10)) where {T<:Real,Bi<:AbsBilliard}
    kmax=maximum(ks)
    L=billiard.length
    bdry=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    xlim,ylim=boundary_limits(bdry;grd=max(1000,round(Int,kmax*L*b/TWO_PI)))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,kmax*dx*b/TWO_PI),512)
    ny=max(round(Int,kmax*dy*b/TWO_PI),512)
    xgrid=collect(T,range(xlim...,nx))
    ygrid=collect(T,range(ylim...,ny))
    r2max=one(T)-δdisk
    disk_idxs=Int[]
    sizehint!(disk_idxs,nx*ny)
    @inbounds for jy in 1:ny,ix in 1:nx
        x=xgrid[ix]
        y=ygrid[jy]
        muladd(x,x,y*y)<r2max&&push!(disk_idxs,ix+(jy-1)*nx)
    end
    !inside_only&&return xgrid,ygrid,disk_idxs,nx,ny
    Nd=length(disk_idxs)
    pts_disk=Vector{SVector{2,T}}(undef,Nd)
    @inbounds for t in 1:Nd
        idx=disk_idxs[t]
        ix=(idx-1)%nx+1
        jy=(idx-1)÷nx+1
        pts_disk[t]=SVector(xgrid[ix],ygrid[jy])
    end
    mask_in=points_in_billiard_polygon(pts_disk,billiard,round(Int,sqrt(Nd));fundamental_domain=fundamental)
    idxs=Int[]
    sizehint!(idxs,Nd)
    @inbounds for t in 1:Nd
        mask_in[t]&&push!(idxs,disk_idxs[t])
    end
    return xgrid,ygrid,idxs,nx,ny
end

"""
    d_bounds_hyp_grid(data::HypSLPData{T},xgrid,ygrid,idxs; pad_min=T(0.8),pad_max=T(1.1),dmin_floor=T(1e-8)) where {T<:Real}

Compute distance bounds for Legendre-Q Taylor-table construction.

Inputs

`data::HypSLPData{T}`
    Cached boundary coordinates.

`xgrid`, `ygrid`
    Cartesian grid vectors.

`idxs`
    Active evaluation indices.

Keyword arguments

`pad_min::T`
    Multiplicative padding for the minimum distance.

`pad_max::T`
    Multiplicative padding for the maximum distance.

`dmin_floor::T`
    Lower bound for the returned minimum distance.

Outputs

`(dmin,dmax)::Tuple{T,T}`
    Safe hyperbolic-distance range over all active grid/boundary pairs.

Numerical safety

Non-finite and zero distances are ignored.
"""
function d_bounds_hyp_grid(data::HypSLPData{T},xgrid,ygrid,idxs;pad_min=T(0.8),pad_max=T(1.1),dmin_floor=T(1e-8)) where {T<:Real}
    dmin=typemax(T)
    dmax=zero(T)
    nx=length(xgrid)
    qx=data.qx
    qy=data.qy
    @inbounds for idx in idxs
        ix=(idx-1)%nx+1
        jy=(idx-1)÷nx+1
        x=xgrid[ix]
        y=ygrid[jy]
        for j in eachindex(qx)
            d=hyperbolic_distance_poincare(x,y,qx[j],qy[j])
            if isfinite(d)&&d>zero(T)
                dmin=min(dmin,d)
                dmax=max(dmax,d)
            end
        end
    end
    dmin==typemax(T)&&error("Could not determine finite hyperbolic distance bounds.")
    return max(dmin_floor,pad_min*dmin),pad_max*dmax
end

"""

    wavefunction_multi_hyp(ks,vec_u,vec_bd,billiard; kwargs...) -> Psi2ds,xgrid,ygrid

Reconstruct hyperbolic billiard wavefunctions on one Cartesian grid.
Important convention:
- `u_constructed_from=:source_normal` is the default and is intended for
  `us_all` returned directly by `compute_spectrum_hyp`.
- `u_constructed_from=:target_normal` should be used when `vec_u` comes from the
  adjoint `solve_vect` boundary-function path.
Keyword convention effect:
- `:source_normal` uses reconstruction weights `dsH`.
- `:target_normal` uses reconstruction weights `ds`.
Other keywords control grid resolution, masking, threading, and Legendre-Q table
construction.
"""
function wavefunction_multi_hyp(ks::Vector{T},vec_u::Vector{<:AbstractVector},vec_bd::AbstractVector{<:HyperbolicBeynPoints},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=true,MIN_CHUNK::Int=4096,δdisk::T=T(1e-10),mp_dps::Int=80,leg_type::Int=3,u_constructed_from::Symbol=:source_normal) where {T<:Real,Bi<:AbsBilliard}
    xgrid,ygrid,idxs,nx,ny=_make_grid_and_idxs_for_billiard(ks,billiard;b=b,fundamental=fundamental,inside_only=inside_only,δdisk=δdisk)
    nmask=length(idxs)
    NT=Threads.nthreads()
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    q,r=divrem(nmask,NT_eff)
    Psi_flat=zeros(Complex{T},nx*ny)
    Psi2ds=Vector{Matrix{Complex{T}}}(undef,length(ks))
    prog=Progress(length(ks),desc="Constructing hyperbolic SLP wavefunctions...")
    @inbounds for i in eachindex(ks)
        fill!(Psi_flat,zero(Complex{T}))
        bd=vec_bd[i]
        u=Complex{T}.(vec_u[i])
        data=prepare_hyp_slp_data(bd;u_constructed_from=u_constructed_from)
        dmin,dmax=d_bounds_hyp_grid(data,xgrid,ygrid,idxs;pad_min=T(0.8),pad_max=T(1.1),dmin_floor=T(1e-8))
        dmin=max(dmin,T(1e-4))
        tab=build_QTaylorTable(ComplexF64(ks[i]);dmin=dmin,dmax=dmax,mp_dps=mp_dps,leg_type=leg_type)
        Threads.@threads :static for t in 1:NT_eff
            lo=(t-1)*q+min(t-1,r)+1
            hi=lo+q-1+(t<=r ? 1 : 0)
            @inbounds for jj in lo:hi
                idx=idxs[jj]
                ix=(idx-1)%nx+1
                jy=(idx-1)÷nx+1
                Psi_flat[idx]=ψ_hyp_slp_fast(xgrid[ix],ygrid[jy],tab,data,u)
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(prog)
    end
    return Psi2ds,xgrid,ygrid
end

"""
    normalize_psi_hyperbolic!(ψ::AbstractMatrix{Num},xgrid::AbstractVector{T},ygrid::AbstractVector{T}) where {Num<:Number,T<:Real}

Normalize a wavefunction in the hyperbolic area measure.

Inputs

`ψ::AbstractMatrix{Num}`
    Wavefunction values on the Cartesian grid.

`xgrid::AbstractVector{T}`
    x-grid.

`ygrid::AbstractVector{T}`
    y-grid.

Output

The same matrix `ψ`, normalized in place.

Normalization

The routine rescales `ψ` so that

    ∫ |ψ|² dA_H ≈ 1,

using

    dA_H = λ² dxdy.

Numerical safety

Non-finite wavefunction values are ignored in the norm accumulation.
"""
function normalize_psi_hyperbolic!(ψ::AbstractMatrix{Num},xgrid::AbstractVector{T},ygrid::AbstractVector{T}) where {Num<:Number,T<:Real}
    dx=xgrid[2]-xgrid[1]
    dy=ygrid[2]-ygrid[1]
    s=zero(T)
    @inbounds for j in eachindex(ygrid),i in eachindex(xgrid)
        val=ψ[i,j]
        if isfinite(real(val))&&isfinite(imag(val))
            s+=abs2(val)*λ2_poincare(xgrid[i],ygrid[j])
        end
    end
    ψ./=sqrt(s*dx*dy+eps(T))
    return ψ
end

@inline _hyp_bp(obj::HyperbolicBeynPoints)=hyp_bp(obj)
@inline _hyp_bp(obj)=obj

@inline function _hyp_boundary_ξ(obj)
    bp=_hyp_bp(obj)
    return bp.ξ
end

"""
    wavefunction_multi_hyp_with_husimi(ks,vec_u,vec_bd,billiard; kwargs...)

Construct hyperbolic wavefunctions and boundary Husimi functions.

Assumes `vec_u` contains adjoint-kernel boundary functions

    u = ∂ₙᴱψ,

so wavefunctions use your existing `wavefunction_multi_hyp`, and Husimis use
`ξ=s_H` with Euclidean weights internally through `husimi_on_grid_hyp`.
"""
function wavefunction_multi_hyp_with_husimi(ks::Vector{T},vec_u::Vector{<:AbstractVector},vec_bd::AbstractVector{<:HyperbolicBeynPoints},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=true,MIN_CHUNK::Int=4096,δdisk::T=T(1e-10),mp_dps::Int=80,leg_type::Int=3,q_oversample::Float64=2.0,nq_min::Int=1000,np::Int=1000,pmax::T=one(T),full_p::Bool=false,show_progress::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    Psi2ds,xgrid,ygrid=wavefunction_multi_hyp(ks,vec_u,vec_bd,billiard;b=b,inside_only=inside_only,fundamental=fundamental,MIN_CHUNK=MIN_CHUNK,δdisk=δdisk,mp_dps=mp_dps,leg_type=leg_type)
    bps=[_hyp_bp(bd) for bd in vec_bd]
    qs,ps=make_qp_grids_hyp(ks,bps;q_oversample=q_oversample,nq_min=nq_min,np=np,pmax=pmax,full_p=full_p)
    Hs=husimi_on_grid_hyp(ks,bps,vec_u,qs,ps;full_p=full_p,show_progress=show_progress)
    ps_out=Vector{Vector{T}}(undef,length(ks))
    @inbounds for i in eachindex(ks)
        ps_out[i]=full_p ? ps[i] : vcat(-reverse(ps[i])[1:end-1],ps[i])
    end
    return Psi2ds,xgrid,ygrid,Hs,ps_out,qs
end

"""
    plot_wavefunctions_hyp_BATCH(ks,Psi2ds,xgrid,ygrid,billiard; kwargs...)

Plot a batch of hyperbolic wavefunctions.
"""
function plot_wavefunctions_hyp_BATCH(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},xgrid::AbstractVector{T},ygrid::AbstractVector{T},billiard::Bi;b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental::Bool=true,custom_label::Vector{String}=String[],wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    L=billiard.length
    bdry=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    xlim,ylim=boundary_limits(bdry;grd=max(1000,round(Int,maximum(ks)*L*b/TWO_PI)))
    nrows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*nrows)),size=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*nrows)))
    row=1
    col=1
    @showprogress desc="Plotting hyperbolic wavefunctions..." for j in eachindex(ks)
        title=isempty(custom_label) ? "k=$(round(ks[j],digits=8))" : custom_label[j]
        ax=Axis(f[row,col],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        wm=wave_mode===:auto ? (eltype(Psi2ds[j])<:Real ? :real : :abs) : wave_mode
        ψplot=wavefunction_plot_data(Psi2ds[j];mode=wm)
        cmap=wm in (:real,:imag) ? :balance : Reverse(:gist_heat)
        crange=wm in (:real,:imag) ? (-1,1) : (0,1)
        heatmap!(ax,xgrid,ygrid,ψplot;colormap=cmap,colorrange=crange)
        plt_boundary && plot_boundary!(ax,billiard;fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_hyp(ks,Psi2ds,xgrid,ygrid,billiard; N=100, kwargs...)

Plot hyperbolic wavefunctions, split into batches of size `N`.
"""
function plot_wavefunctions_hyp(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},xgrid::AbstractVector{T},ygrid::AbstractVector{T},billiard::Bi;N::Integer=100,kwargs...) where {T<:Real,Bi<:AbsBilliard}
    return batch_wrapper(plot_wavefunctions_hyp_BATCH,ks,Psi2ds,xgrid,ygrid,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions_with_husimi_hyp_BATCH(ks,Psi2ds,xgrid,ygrid,Hs,ps,qs,billiard; kwargs...)

Plot hyperbolic wavefunctions with their boundary Husimi densities.
"""
function plot_wavefunctions_with_husimi_hyp_BATCH(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},xgrid::AbstractVector{T},ygrid::AbstractVector{T},Hs::Vector{<:AbstractMatrix{T}},ps::Vector{<:AbstractVector{T}},qs::Vector{<:AbstractVector{T}},billiard::Bi;b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=4,fundamental::Bool=true,custom_label::Vector{String}=String[],wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    L=billiard.length
    bdry=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    xlim,ylim=boundary_limits(bdry;grd=max(1000,round(Int,maximum(ks)*L*b/TWO_PI)))
    nrows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(round(Int,3*width_ax*max_cols),round(Int,1.6*height_ax*nrows)),size=(round(Int,3*width_ax*max_cols),round(Int,1.6*height_ax*nrows)))
    row=1
    col=1
    @showprogress desc="Plotting hyperbolic wavefunctions and Husimis..." for j in eachindex(ks)
        title=isempty(custom_label) ? "k=$(round(ks[j],digits=8))" : custom_label[j]
        ax=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        axh=Axis(f[row,col][1,2],xlabel="ξ",ylabel="p",width=width_ax,height=height_ax)
        wm=wave_mode===:auto ? (eltype(Psi2ds[j])<:Real ? :real : :abs) : wave_mode
        ψplot=wavefunction_plot_data(Psi2ds[j];mode=wm)
        cmap=wm in (:real,:imag) ? :balance : Reverse(:gist_heat)
        crange=wm in (:real,:imag) ? (-1,1) : (0,1)
        heatmap!(ax,xgrid,ygrid,ψplot;colormap=cmap,colorrange=crange)
        plt_boundary && plot_boundary!(ax,billiard;fundamental_domain=fundamental,plot_normal=false)
        heatmap!(axh,qs[j],ps[j],Hs[j];colormap=Reverse(:gist_heat))
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_with_husimi_hyp(ks,Psi2ds,xgrid,ygrid,Hs,ps,qs,billiard; N=100, kwargs...)

Plot hyperbolic wavefunctions and Husimi functions, split into batches.
"""
function plot_wavefunctions_with_husimi_hyp(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},xgrid::AbstractVector{T},ygrid::AbstractVector{T},Hs::Vector{<:AbstractMatrix{T}},ps::Vector{<:AbstractVector{T}},qs::Vector{<:AbstractVector{T}},billiard::Bi;N::Integer=100,kwargs...) where {T<:Real,Bi<:AbsBilliard}
    return batch_wrapper(plot_wavefunctions_with_husimi_hyp_BATCH,ks,Psi2ds,xgrid,ygrid,Hs,ps,qs,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions_with_husimi_and_boundary_hyp_BATCH(ks,Psi2ds,xgrid,ygrid,Hs,ps,qs,billiard,us,bps; kwargs...)

Plot hyperbolic wavefunctions, Husimis, and boundary functions `u_E(ξ)`.
"""
function plot_wavefunctions_with_husimi_and_boundary_hyp_BATCH(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},xgrid::AbstractVector{T},ygrid::AbstractVector{T},Hs::Vector{<:AbstractMatrix{T}},ps::Vector{<:AbstractVector{T}},qs::Vector{<:AbstractVector{T}},billiard::Bi,us::Vector{<:AbstractVector{<:Number}},bps::AbstractVector;b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=3,fundamental::Bool=true,custom_label::Vector{String}=String[],wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    L=billiard.length
    bdry=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    xlim,ylim=boundary_limits(bdry;grd=max(1000,round(Int,maximum(ks)*L*b/TWO_PI)))
    nrows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(round(Int,3*width_ax*max_cols),round(Int,2.2*height_ax*nrows)),size=(round(Int,3*width_ax*max_cols),round(Int,2.2*height_ax*nrows)))
    row=1
    col=1
    @showprogress desc="Plotting hyperbolic wavefunctions, Husimis and boundary data..." for j in eachindex(ks)
        title=isempty(custom_label) ? "k=$(round(ks[j],digits=8))" : custom_label[j]
        ax=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        axh=Axis(f[row,col][1,2],xlabel="ξ",ylabel="p",width=width_ax,height=height_ax)
        axu=Axis(f[row,col][2,1:2],xlabel="ξ",ylabel="u_E(ξ)",width=2*width_ax,height=height_ax/2)
        wm=wave_mode===:auto ? (eltype(Psi2ds[j])<:Real ? :real : :abs) : wave_mode
        ψplot=wavefunction_plot_data(Psi2ds[j];mode=wm)
        cmap=wm in (:real,:imag) ? :balance : Reverse(:gist_heat)
        crange=wm in (:real,:imag) ? (-1,1) : (0,1)
        heatmap!(ax,xgrid,ygrid,ψplot;colormap=cmap,colorrange=crange)
        plt_boundary && plot_boundary!(ax,billiard;fundamental_domain=fundamental,plot_normal=false)
        heatmap!(axh,qs[j],ps[j],Hs[j];colormap=Reverse(:gist_heat))
        ξ=_hyp_boundary_ξ(bps[j])
        lines!(axu,ξ,real.(us[j]);linewidth=2,label="Re u_E")
        maximum(abs.(imag.(us[j])))>sqrt(eps(T)) && lines!(axu,ξ,imag.(us[j]);linewidth=2,linestyle=:dash,label="Im u_E")
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_with_husimi_and_boundary_hyp(ks,Psi2ds,xgrid,ygrid,Hs,ps,qs,billiard,us,bps; N=100, kwargs...)

Plot hyperbolic wavefunctions, Husimis, and boundary functions, split into batches.
"""
function plot_wavefunctions_with_husimi_and_boundary_hyp(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},xgrid::AbstractVector{T},ygrid::AbstractVector{T},Hs::Vector{<:AbstractMatrix{T}},ps::Vector{<:AbstractVector{T}},qs::Vector{<:AbstractVector{T}},billiard::Bi,us::Vector{<:AbstractVector{<:Number}},bps::AbstractVector;N::Integer=100,kwargs...) where {T<:Real,Bi<:AbsBilliard}
    return batch_wrapper(plot_wavefunctions_with_husimi_and_boundary_hyp_BATCH,ks,Psi2ds,xgrid,ygrid,Hs,ps,qs,billiard,us,bps;N=N,kwargs...)
end