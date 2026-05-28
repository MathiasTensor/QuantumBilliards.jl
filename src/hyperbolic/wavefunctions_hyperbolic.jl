"""
    HypSLPData{T<:Real}

Cache container for hyperbolic single-layer-potential reconstruction.

Fields

`qx::Vector{T}`
    x-coordinates of boundary quadrature nodes.

`qy::Vector{T}`
    y-coordinates of boundary quadrature nodes.

`wH::Vector{T}`
    Hyperbolic quadrature weights

        wH[j] = λ(q_j) ds_E[j].

Purpose

Avoid repeated coordinate extraction, conformal-factor evaluation, and
hyperbolic boundary-weight construction inside the wavefunction loop.
"""
struct HypSLPData{T<:Real}
    qx::Vector{T}
    qy::Vector{T}
    wH::Vector{T}
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

"""
    prepare_hyp_slp_data(bd::BoundaryPointsHyp{T}) where {T<:Real}

Precompute boundary data for hyperbolic SLP reconstruction.

Inputs

`bd::HyperbolicBeynPoints`
    Boundary discretization with fields:
    `xy::Vector`, `ds::Vector{T}`.

Output

`HypSLPData{T}`
    Cached boundary coordinates and hyperbolic weights.

Details

The stored weights are

    wH[j] = λ(q_j) bd.ds[j],

so runtime reconstruction evaluates

    ψ(x) = Σ_j G_k(x,q_j) u_j wH[j].
"""
function prepare_hyp_slp_data(obj::HyperbolicBeynPoints)
    bd=hyp_bp(obj)
    T=eltype(bd.ds)
    N=length(bd.xy)
    qx=Vector{T}(undef,N)
    qy=Vector{T}(undef,N)
    wH=Vector{T}(undef,N)
    @inbounds for j in 1:N
        x=bd.xy[j][1]
        y=bd.xy[j][2]
        qx[j]=x
        qy[j]=y
        wH[j]=bd.ds[j]*λ_poincare(x,y)
    end
    return HypSLPData(qx,qy,wH)
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
    wH=data.wH
    @inbounds @simd for j in eachindex(u)
        G=slp_kernel_hyp(tab,x,y,qx[j],qy[j])*wH[j]
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
    wH=data.wH
    @inbounds @simd for j in eachindex(u)
        acc+=slp_kernel_hyp(tab,x,y,qx[j],qy[j])*(wH[j]*u[j])
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
    wavefunction_multi_hyp(ks::Vector{T},vec_u::Vector{<:AbstractVector},vec_bd::AbstractVector{<:HyperbolicBeynPoints},billiard::Bi; kwargs...) where {T<:Real,Bi<:AbsBilliard}

Reconstruct several hyperbolic billiard eigenfunctions on one Cartesian grid.

Inputs

`ks::Vector{T}`
    Eigen-wavenumbers.

`vec_u::Vector{<:AbstractVector}`
    Boundary density vectors. Each entry must match the corresponding boundary
    discretization in `vec_bd`.

`vec_bd::AbstractVector{<:HyperbolicBeynPoints}`
    Boundary discretizations for the eigenstates.

`billiard::Bi`
    Billiard geometry used for grid construction and inside-domain masking.

Keyword arguments

`b::Float64=5.0`
    Grid resolution parameter.

`inside_only::Bool=true`
    Evaluate only inside the billiard if true.

`fundamental::Bool=true`
    Build the plotting grid from the fundamental boundary if true.

`MIN_CHUNK::Int=4096`
    Minimum thread chunk size.

`δdisk::T=T(1e-10)`
    Unit-disk safety margin.

`mp_dps::Int=80`
    Multiprecision digits used by the Legendre-Q Taylor builder.

`leg_type::Int=3`
    Legendre-Q convention selector.

Outputs

`Psi2ds::Vector{Matrix{Complex{T}}}`
    Reconstructed wavefunction matrices.

`xgrid::Vector{T}`
    Cartesian x-grid.

`ygrid::Vector{T}`
    Cartesian y-grid.

Algorithm

For each eigenstate:
    1. cache boundary coordinates and hyperbolic weights,
    2. compute grid-to-boundary hyperbolic distance bounds,
    3. build a `QTaylorTable`,
    4. evaluate the SLP on active grid points.

Notes

No symmetry images are generated here. If the eigenproblem was solved in a
symmetry-reduced basis, expand the boundary vector to the full physical boundary
before calling this routine.
"""
function wavefunction_multi_hyp(ks::Vector{T},vec_u::Vector{<:AbstractVector},vec_bd::AbstractVector{<:HyperbolicBeynPoints},billiard::Bi;b::Float64=5.0,inside_only::Bool=true,fundamental::Bool=true,MIN_CHUNK::Int=4096,δdisk::T=T(1e-10),mp_dps::Int=80,leg_type::Int=3) where {T<:Real,Bi<:AbsBilliard}
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
        data=prepare_hyp_slp_data(bd)
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