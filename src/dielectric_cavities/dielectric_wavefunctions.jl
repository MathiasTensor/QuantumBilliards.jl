# For q_a=n_a k and q_out=n_out k,
# D_q(x,y)=-(iq/2)H₁⁽¹⁾(qr) inner/r,
# S_q(x,y)= (i/2)H₀⁽¹⁾(qr) s_y,
# with r=|x-y| and
# inner=γ'_y(t)(x-x_j)-γ'_x(t)(y-y_j)=s_y n_y⋅(x-y).
#
# For trace x=(φ,ψ), χ(n)=1 for TM and χ(n)=n² for TE,
# u_a= +(1/2)∫Γa[D_qa ψ_a+χ_a S_qa φ_a]dt,
# u_0= -(1/2)Σ_a∫Γa[D_qout ψ_a+χ_out S_qout φ_a]dt.
#
# After quadrature,
# interior: D_j=(-iq_a/4)w_jψ_j, S_j=(+iχ_a/4)w_js_jφ_j,
# exterior: D_j=(+iq_out/4)w_jψ_j, S_j=(-iχ_out/4)w_js_jφ_j.
struct WiersigWavefunctionCache{T<:Real}
    x::Vector{T}
    y::Vector{T}
    tx::Vector{T}
    ty::Vector{T}
    s::Vector{T}
    w::Vector{T}
    offsets::Vector{Int}
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    hmin::T
end

# Target-grid representation used only for wavefunction reconstruction.
# With symmetry this stores the fundamental target domain, not the full field.
abstract type AbstractWiersigWavefunctionGrid end

# Cartesian target grid used without symmetry and for reflection symmetries.
# Matrix convention: Psi[ix,iy]=u(x[ix],y[iy]).
struct WiersigCartesianGrid{T<:Real}<:AbstractWiersigWavefunctionGrid
    x::Vector{T}
    y::Vector{T}
end

# Polar target grid used for rotational symmetry. Only one angular wedge is
# reconstructed explicitly around `center`.
# Matrix convention: Psi[ir,iθ]=u(r[ir],θ[iθ]).
struct WiersigPolarGrid{T<:Real}<:AbstractWiersigWavefunctionGrid
    r::Vector{T}
    θ::Vector{T}
    center::NTuple{2,T}
end

# Flatten BoundaryPointsCFIE objects without changing the Wiersig trace ordering.
# `offsets` stores the complete node range of each physical cavity, while
# hmin≈min_j w_j|γ'(t_j)| estimates the smallest physical node spacing.
function build_wiersig_wavefunction_cache(pts::AbstractVector{<:BoundaryPointsCFIE{T}},ws) where {T<:Real}
    N=sum(length(p.xy) for p in pts)
    N>0||throw(ArgumentError("at least one boundary node is required"))
    x=Vector{T}(undef,N);y=similar(x);tx=similar(x);ty=similar(x);s=similar(x);w=similar(x)
    j=1
    @inbounds for p in pts
        length(p.xy)==length(p.tangent)==length(p.ws)||throw(DimensionMismatch("inconsistent BoundaryPointsCFIE lengths"))
        for l in eachindex(p.xy)
            q=p.xy[l];t=p.tangent[l]
            x[j]=q[1];y[j]=q[2];tx[j]=t[1];ty[j]=t[2]
            s[j]=hypot(t[1],t[2]);w[j]=p.ws[l];j+=1
        end
    end
    offsets=ws.geom isa WiersigMultiGeometry ? copy(ws.geom.offs) : [1,N+1]
    offsets[1]==1&&offsets[end]==N+1||throw(DimensionMismatch("physical cavity offsets do not cover the full boundary"))
    hmin=typemax(T)
    @inbounds for j in 1:N
        h=w[j]*s[j]
        h>zero(T)&&isfinite(h)&&(hmin=min(hmin,h))
    end
    hmin<typemax(T)||error("could not determine physical boundary spacing")
    return WiersigWavefunctionCache(x,y,tx,ty,s,w,offsets,minimum(x),maximum(x),minimum(y),maximum(y),hmin)
end

# Complete flattened source range I_a of physical cavity a.
@inline _wiersig_component_range(c::WiersigWavefunctionCache,a::Int)=c.offsets[a]:c.offsets[a+1]-1

# Number of targets explicitly reconstructed before symmetry expansion.
@inline _wiersig_target_count(g::WiersigCartesianGrid)=length(g.x)*length(g.y)
@inline _wiersig_target_count(g::WiersigPolarGrid)=length(g.r)*length(g.θ)

# Flattened Cartesian target index -> physical target coordinates.
@inline function _wiersig_target(g::WiersigCartesianGrid,idx::Int)
    nx=length(g.x);iy=div(idx-1,nx)+1;ix=idx-(iy-1)*nx
    return g.x[ix],g.y[iy]
end

# Flattened polar target index -> Cartesian coordinates used by the Green kernel.
@inline function _wiersig_target(g::WiersigPolarGrid,idx::Int)
    nr=length(g.r);iθ=div(idx-1,nr)+1;ir=idx-(iθ-1)*nr
    r=g.r[ir];θ=g.θ[iθ];cx,cy=g.center
    return cx+r*cos(θ),cy+r*sin(θ)
end

# Complete padded plotting limits around the physical cavity configuration.
@inline function _wiersig_wavefunction_limits(c::WiersigWavefunctionCache{T},exterior_pad::Real) where {T<:Real}
    dx=c.xmax-c.xmin;dy=c.ymax-c.ymin
    xpad=T(exterior_pad)*dx;ypad=T(exterior_pad)*dy
    return (c.xmin-xpad,c.xmax+xpad),(c.ymin-ypad,c.ymax+ypad)
end

# No symmetry: reconstruct the complete Cartesian target domain.
function _wiersig_wavefunction_grid(::Nothing,c::WiersigWavefunctionCache{T},nx::Int,ny::Int,exterior_pad::Real) where {T<:Real}
    xlim,ylim=_wiersig_wavefunction_limits(c,exterior_pad)
    x=collect(T,range(xlim...;length=nx));y=collect(T,range(ylim...;length=ny))
    return WiersigCartesianGrid(x,y)
end

# Reflection fundamental domains:
# :y_axis -> x>=0, :x_axis -> y>=0, :origin -> x>=0,y>=0.
# The omitted half/quarters are restored afterwards using the parity.
function _wiersig_wavefunction_grid(sym::Reflection,c::WiersigWavefunctionCache{T},nx::Int,ny::Int,exterior_pad::Real) where {T<:Real}
    xlim,ylim=_wiersig_wavefunction_limits(c,exterior_pad)
    if sym.axis===:y_axis
        nxr=fld(nx,2)+1;x=collect(T,range(zero(T),max(abs(xlim[1]),abs(xlim[2]));length=nxr));y=collect(T,range(ylim...;length=ny))
    elseif sym.axis===:x_axis
        nyr=fld(ny,2)+1;x=collect(T,range(xlim...;length=nx));y=collect(T,range(zero(T),max(abs(ylim[1]),abs(ylim[2]));length=nyr))
    elseif sym.axis===:origin
        nxr=fld(nx,2)+1;nyr=fld(ny,2)+1
        x=collect(T,range(zero(T),max(abs(xlim[1]),abs(xlim[2]));length=nxr))
        y=collect(T,range(zero(T),max(abs(ylim[1]),abs(ylim[2]));length=nyr))
    else
        throw(ArgumentError("unsupported reflection axis $(sym.axis)"))
    end
    return WiersigCartesianGrid(x,y)
end

# Mixed Z₂×Z₂ reflection: x->-x acts within each cavity and y->-y exchanges
# symmetry-related cavities, so only the x>=0,y>=0 target quadrant is evaluated.
function _wiersig_wavefunction_grid(sym::WiersigMixedReflection,c::WiersigWavefunctionCache{T},nx::Int,ny::Int,exterior_pad::Real) where {T<:Real}
    xlim,ylim=_wiersig_wavefunction_limits(c,exterior_pad)
    nxr=fld(nx,2)+1;nyr=fld(ny,2)+1
    x=collect(T,range(zero(T),max(abs(xlim[1]),abs(xlim[2]));length=nxr))
    y=collect(T,range(zero(T),max(abs(ylim[1]),abs(ylim[2]));length=nyr))
    return WiersigCartesianGrid(x,y)
end

# Rotation(n,m): reconstruct one polar wedge θ∈[0,2π/n) around `sym.center`.
# `nθ` is the requested angular resolution of the complete 2π field.
function _wiersig_wavefunction_grid(sym::Rotation,c::WiersigWavefunctionCache{T},nr::Int,nθ::Int,exterior_pad::Real) where {T<:Real}
    xlim,ylim=_wiersig_wavefunction_limits(c,exterior_pad);cx=T(sym.center[1]);cy=T(sym.center[2])
    R=max(hypot(xlim[1]-cx,ylim[1]-cy),hypot(xlim[1]-cx,ylim[2]-cy),hypot(xlim[2]-cx,ylim[1]-cy),hypot(xlim[2]-cx,ylim[2]-cy))
    nθr=max(2,cld(nθ,sym.n));α=TWO_PI/T(sym.n)
    r=collect(T,range(zero(T),R;length=nr))
    θ=collect(T,range(zero(T),α;length=nθr+1))[1:end-1]
    return WiersigPolarGrid(r,θ,(cx,cy))
end

# Classify each explicitly reconstructed target:
# label=0 exterior Ω₀, label=a physical dielectric cavity Ω_a.
# Polar targets are converted to Cartesian coordinates before classification.
function _wiersig_interior_grid(solver::WiersigKress,g::AbstractWiersigWavefunctionGrid)
    nt=_wiersig_target_count(g);x0,y0=_wiersig_target(g,1);T=typeof(x0)
    pts=Vector{SVector{2,T}}(undef,nt)
    @inbounds for i in 1:nt
        x,y=_wiersig_target(g,i);pts[i]=SVector{2,T}(x,y)
    end
    labels=zeros(Int,nt);grd=max(1000,round(Int,sqrt(nt)))
    @inbounds for a in eachindex(solver.billiards)
        mask=points_in_billiard_polygon(pts,solver.billiards[a],grd;fundamental_domain=false)
        for i in eachindex(mask)
            if mask[i]
                labels[i]==0||throw(ArgumentError("overlapping dielectric cavities at $(pts[i])"))
                labels[i]=a
            end
        end
    end
    return labels
end

# Packed piecewise-Chebyshev approximations of H₀⁽¹⁾(qr),H₁⁽¹⁾(qr).
# hνin[d,m,p,a]: coefficient d, state m, radial panel p, cavity a.
# hνout[d,m,p] : corresponding common-exterior coefficients.
# qin/qout are retained for direct evaluation outside [rmin,rmax].
struct WiersigWavefunctionPlans
    h0in::Array{ComplexF64,4}
    h1in::Array{ComplexF64,4}
    h0out::Array{ComplexF64,3}
    h1out::Array{ComplexF64,3}
    qin::Matrix{ComplexF64}
    qout::Vector{ComplexF64}
    rmin::Float64
    rmax::Float64
    npanels::Int
    h::Float64
    invh::Float64
end

# Characteristic radial scale of the explicitly reconstructed target domain.
# Larger source-target distances use the direct-Hankel fallback.
@inline _wiersig_wavefunction_rmax(g::WiersigCartesianGrid)=hypot(Float64(last(g.x)-first(g.x)),Float64(last(g.y)-first(g.y)))
@inline _wiersig_wavefunction_rmax(g::WiersigPolarGrid)=Float64(last(g.r))

# Build packed Chebyshev plans for all q_{a,m}=n_a k_m and q_{0,m}=n_out k_m.
# One common radial partition is tuned so panel lookup can be reused across states.
function build_wiersig_wavefunction_plans(solver::WiersigKress,ks,c::WiersigWavefunctionCache,g::AbstractWiersigWavefunctionGrid;cheb_tol::Real=1e-10,npanels_init::Int=3000,M_init::Int=5,sampling_points::Int=20_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,rmin_factor::Real=0.85,rmax_pad::Real=1.1,verbose::Bool=false)
    C=length(c.offsets)-1;ns=length(ks);nin=_wiersig_component_indices(solver,C)
    qin=Matrix{ComplexF64}(undef,C,ns);qout=Vector{ComplexF64}(undef,ns)
    qs=Vector{ComplexF64}(undef,ns*(C+1));qmin=Inf;l=1
    @inbounds for m in 1:ns
        k=ComplexF64(ks[m])
        for a in 1:C
            q=ComplexF64(nin[a])*k;qin[a,m]=q;qs[l]=q;l+=1;qmin=min(qmin,abs(q))
        end
        q=ComplexF64(solver.n_out)*k;qout[m]=q;qs[l]=q;l+=1;qmin=min(qmin,abs(q))
    end
    zcut=Float64(hankel_z_chebyshev_cutoff)
    rmin=max(Float64(rmin_factor*c.hmin),zcut/qmin,1e-12)
    rmax=Float64(rmax_pad)*_wiersig_wavefunction_rmax(g)
    rmax>rmin||throw(ArgumentError("invalid radial interval [$rmin,$rmax]"))
    rs=collect(range(rmin,rmax;length=sampling_points))
    npanels=npanels_init;M=M_init
    plans0=Vector{ChebHankelPlanH}(undef,length(qs));plans1=Vector{ChebHankelPlanH}(undef,length(qs))
    err0=fill(Inf,length(qs));err1=fill(Inf,length(qs));converged=false
    for it in 1:max_iter
        Threads.@threads for j in eachindex(qs)
            plans0[j]=plan_h(0,1,qs[j],rmin,rmax;npanels=npanels,M=M)
            plans1[j]=plan_h(1,1,qs[j],rmin,rmax;npanels=npanels,M=M)
        end
        _check_H0H1_errors!(err0,err1,plans0,plans1,qs,rs)
        worst=max(maximum(err0),maximum(err1))
        verbose&&@info "Wiersig wavefunction Chebyshev tuning" worst npanels M rmin rmax
        if worst<cheb_tol
            converged=true;break
        end
        it%5==0 ? (M+=grow_M) : (npanels=ceil(Int,grow_panels*npanels))
    end
    converged||@warn "Wavefunction Chebyshev tuning did not reach tol=$cheb_tol after $max_iter iterations."
    h0in=Array{ComplexF64}(undef,M+1,ns,npanels,C);h1in=similar(h0in)
    h0out=Array{ComplexF64}(undef,M+1,ns,npanels);h1out=similar(h0out)
    l=1
    @inbounds for m in 1:ns
        for a in 1:C
            p0=plans0[l];p1=plans1[l]
            for p in 1:npanels
                copyto!(@view(h0in[:,m,p,a]),p0.panels[p].c);copyto!(@view(h1in[:,m,p,a]),p1.panels[p].c)
            end
            l+=1
        end
        p0=plans0[l];p1=plans1[l]
        for p in 1:npanels
            copyto!(@view(h0out[:,m,p]),p0.panels[p].c);copyto!(@view(h1out[:,m,p]),p1.panels[p].c)
        end
        l+=1
    end
    h=(rmax-rmin)/npanels
    return WiersigWavefunctionPlans(h0in,h1in,h0out,h1out,qin,qout,rmin,rmax,npanels,h,inv(h))
end

# Map a radius to its uniform Chebyshev panel and local coordinate.
# p=0 means that this distance is outside the packed interpolation interval.
@inline function _wiersig_panel(pl::WiersigWavefunctionPlans,r::Float64)
    (r<pl.rmin||r>pl.rmax)&&return 0,0.0
    p=clamp(Int(floor((r-pl.rmin)*pl.invh))+1,1,pl.npanels)
    t=2*(r-(pl.rmin+(p-0.5)*pl.h))*pl.invh
    return p,t
end

# Expand symmetry-reduced traces to the complete physical (φ,ψ) boundary trace.
# Wavefunction reconstruction always uses the complete physical source boundary.
function _wiersig_symmetrize_density(solver::WiersigKress,xs::AbstractVector{<:AbstractVector},pts,ws)
    N=sum(length(p.xy) for p in pts);N2=2N
    out=Vector{Vector{ComplexF64}}(undef,length(xs))
    @inbounds for m in eachindex(xs)
        x=xs[m]
        if length(x)==N2
            out[m]=ComplexF64.(x)
        else
            length(x)==boundary_matrix_size(ws)||throw(DimensionMismatch("state $m has length $(length(x)); expected $N2 or $(boundary_matrix_size(ws))"))
            out[m]=ComplexF64.(expand_wiersig_trace(x,ws))
            length(out[m])==N2||throw(DimensionMismatch("expanded state $m has incorrect length"))
        end
    end
    return out
end

# Precomputed SLP/DLP source amplitudes for every state and physical boundary node.
struct WiersigWavefunctionCoefficients{T<:Real}
    Din::Matrix{Complex{T}}
    Sin::Matrix{Complex{T}}
    Dout::Matrix{Complex{T}}
    Sout::Matrix{Complex{T}}
end

# Din=(-iq_a/4)wψ, Sin=(+iχ_a/4)wsφ,
# Dout=(+iq_out/4)wψ, Sout=(-iχ_out/4)wsφ.
function _wiersig_coefficients(solver::WiersigKress,ks,xs,c::WiersigWavefunctionCache{T}) where {T<:Real}
    ns=length(ks);N=length(c.x);C=length(c.offsets)-1;nin=_wiersig_component_indices(solver,C)
    Din=Matrix{Complex{T}}(undef,ns,N);Sin=similar(Din);Dout=similar(Din);Sout=similar(Din)
    χout=T(_wiersig_slp_factor(solver,solver.n_out))
    @inbounds for m in 1:ns
        φ,ψ=split_wiersig_trace(xs[m]);k=Complex{T}(ks[m]);qout=Complex{T}(solver.n_out)*k
        fDo=Complex{T}(0,one(T)/4)*qout;fSo=-Complex{T}(0,χout/4)
        for j in 1:N
            Dout[m,j]=fDo*c.w[j]*ψ[j];Sout[m,j]=fSo*c.w[j]*c.s[j]*φ[j]
        end
        for a in 1:C
            q=Complex{T}(nin[a])*k;χ=T(_wiersig_slp_factor(solver,nin[a]))
            fDi=-Complex{T}(0,one(T)/4)*q;fSi=Complex{T}(0,χ/4)
            for j in _wiersig_component_range(c,a)
                Din[m,j]=fDi*c.w[j]*ψ[j];Sin[m,j]=fSi*c.w[j]*c.s[j]*φ[j]
            end
        end
    end
    return WiersigWavefunctionCoefficients(Din,Sin,Dout,Sout)
end

# Exterior Chebyshev contribution:
# u += H₁(qr)*(inner/r)*Dout + H₀(qr)*Sout.
@inline function _wiersig_accumulate_out_cheb!(acc,j,p,t,inner,pl,c)
    @inbounds @fastmath for m in eachindex(acc)
        h0=_cheb_clenshaw(@view(pl.h0out[:,m,p]),t);h1=_cheb_clenshaw(@view(pl.h1out[:,m,p]),t)
        acc[m]+=inner*h1*c.Dout[m,j]+h0*c.Sout[m,j]
    end
    return nothing
end

# Interior Chebyshev contribution for physical cavity a.
@inline function _wiersig_accumulate_in_cheb!(acc,j,p,a,t,inner,pl,c)
    @inbounds @fastmath for m in eachindex(acc)
        h0=_cheb_clenshaw(@view(pl.h0in[:,m,p,a]),t);h1=_cheb_clenshaw(@view(pl.h1in[:,m,p,a]),t)
        acc[m]+=inner*h1*c.Din[m,j]+h0*c.Sin[m,j]
    end
    return nothing
end

# Exterior direct-Hankel fallback outside the packed Chebyshev interval.
# The shared small-z series is used near qr=0.
@inline function _wiersig_accumulate_out_direct!(acc,j,r,inner,pl,c)
    @inbounds for m in eachindex(acc)
        z=pl.qout[m]*r;az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0=_small_h0_series(z);h1=_small_h1_series(z)
        else
            h0=H(0,z);h1=H(1,z)
        end
        acc[m]+=inner*h1*c.Dout[m,j]+h0*c.Sout[m,j]
    end
    return nothing
end

# Interior direct-Hankel fallback for physical cavity a.
@inline function _wiersig_accumulate_in_direct!(acc,j,r,a,inner,pl,c)
    @inbounds for m in eachindex(acc)
        z=pl.qin[a,m]*r;az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0=_small_h0_series(z);h1=_small_h1_series(z)
        else
            h0=H(0,z);h1=H(1,z)
        end
        acc[m]+=inner*h1*c.Din[m,j]+h0*c.Sin[m,j]
    end
    return nothing
end

# Reconstruct all states on the selected target grid.
# Exterior targets use all physical boundaries; targets inside Ω_a use Γ_a only.
# Loop order target -> source -> state reuses geometry and panel lookup across states.
function _wiersig_reconstruct!(Psi,g::AbstractWiersigWavefunctionGrid,labels,c::WiersigWavefunctionCache{T},pl::WiersigWavefunctionPlans,coef::WiersigWavefunctionCoefficients{T}) where {T<:Real}
    ns=size(coef.Din,1);N=length(c.x);eps2=eps(T)^2
    accs=[zeros(Complex{T},ns) for _ in 1:Threads.nthreads()]
    @showprogress desc="Wavefunction construction" Threads.@threads :static for idx in eachindex(labels)
        acc=accs[Threads.threadid()];fill!(acc,zero(Complex{T}))
        xp,yp=_wiersig_target(g,idx);a=labels[idx]
        if a==0
            @inbounds for j in 1:N
                dx=xp-c.x[j];dy=yp-c.y[j];r2=muladd(dx,dx,dy*dy)
                r2<=eps2&&continue
                r=sqrt(r2);inner=muladd(c.ty[j],dx,-c.tx[j]*dy)/r;rf=Float64(r);p,t=_wiersig_panel(pl,rf)
                p==0 ? _wiersig_accumulate_out_direct!(acc,j,rf,inner,pl,coef) : _wiersig_accumulate_out_cheb!(acc,j,p,t,inner,pl,coef)
            end
        else
            @inbounds for j in _wiersig_component_range(c,a)
                dx=xp-c.x[j];dy=yp-c.y[j];r2=muladd(dx,dx,dy*dy)
                r2<=eps2&&continue
                r=sqrt(r2);inner=muladd(c.ty[j],dx,-c.tx[j]*dy)/r;rf=Float64(r);p,t=_wiersig_panel(pl,rf)
                p==0 ? _wiersig_accumulate_in_direct!(acc,j,rf,a,inner,pl,coef) : _wiersig_accumulate_in_cheb!(acc,j,p,a,t,inner,pl,coef)
            end
        end
        @inbounds for m in 1:ns;Psi[m][idx]=acc[m];end
    end
    return Psi
end

################################################################################

# No symmetry: the Cartesian field is already complete.
_wiersig_expand_wavefunction(Psi,g::WiersigCartesianGrid,::Nothing)=(Psi,g.x,g.y)

# Reflection symmetry: fill the omitted Cartesian targets using the corresponding
# field parity, without reevaluating the BIE.
function _wiersig_expand_wavefunction(Psi,g::WiersigCartesianGrid,sym::Reflection)
    if sym.axis===:y_axis
        p=ComplexF64(sym.parity);nx=length(g.x);x=vcat(-reverse(g.x[2:end]),g.x);out=Vector{Matrix{ComplexF64}}(undef,length(Psi))
        @inbounds for m in eachindex(Psi)
            F=Psi[m];U=Matrix{ComplexF64}(undef,2nx-1,length(g.y))
            U[nx:end,:].=F;U[1:nx-1,:].=p.*reverse(F[2:end,:];dims=1);out[m]=U
        end
        return out,x,g.y
    elseif sym.axis===:x_axis
        p=ComplexF64(sym.parity);ny=length(g.y);y=vcat(-reverse(g.y[2:end]),g.y);out=Vector{Matrix{ComplexF64}}(undef,length(Psi))
        @inbounds for m in eachindex(Psi)
            F=Psi[m];U=Matrix{ComplexF64}(undef,length(g.x),2ny-1)
            U[:,ny:end].=F;U[:,1:ny-1].=p.*reverse(F[:,2:end];dims=2);out[m]=U
        end
        return out,g.x,y
    elseif sym.axis===:origin
        px=ComplexF64(sym.parity[1]);py=ComplexF64(sym.parity[2])
        return _wiersig_expand_xy_wavefunction(Psi,g,px,py)
    end
    throw(ArgumentError("unsupported reflection axis $(sym.axis)"))
end

# Mixed Z₂×Z₂ symmetry uses the same four-quadrant field expansion.
function _wiersig_expand_wavefunction(Psi,g::WiersigCartesianGrid,sym::WiersigMixedReflection)
    return _wiersig_expand_xy_wavefunction(Psi,g,ComplexF64(sym.intra_parity),ComplexF64(sym.inter_parity))
end

# Expand one Cartesian quadrant to all four using the x/y parity factors.
function _wiersig_expand_xy_wavefunction(Psi,g::WiersigCartesianGrid,px::ComplexF64,py::ComplexF64)
    nx=length(g.x);ny=length(g.y);x=vcat(-reverse(g.x[2:end]),g.x);y=vcat(-reverse(g.y[2:end]),g.y)
    out=Vector{Matrix{ComplexF64}}(undef,length(Psi))
    @inbounds for m in eachindex(Psi)
        F=Psi[m];U=Matrix{ComplexF64}(undef,2nx-1,2ny-1)
        U[nx:end,ny:end].=F
        U[1:nx-1,ny:end].=px.*reverse(F[2:end,:];dims=1)
        U[nx:end,1:ny-1].=py.*reverse(F[:,2:end];dims=2)
        U[1:nx-1,1:ny-1].=(px*py).*reverse(F[2:end,2:end];dims=(1,2))
        out[m]=U
    end
    return out,x,y
end

# C_n rotational symmetry: copy the fundamental polar wedge to the remaining
# sectors using the character exp(i2πml/n).
function _wiersig_expand_wavefunction(Psi,g::WiersigPolarGrid,sym::Rotation)
    nr=length(g.r);nθ=length(g.θ);n=sym.n
    θ=Vector{eltype(g.θ)}(undef,n*nθ)
    @inbounds for l in 0:n-1
        @views θ[l*nθ+1:(l+1)*nθ].=g.θ .+ l*TWO_PI/n
    end
    out=Vector{Matrix{ComplexF64}}(undef,length(Psi))
    @inbounds for j in eachindex(Psi)
        F=Psi[j];U=Matrix{ComplexF64}(undef,nr,n*nθ)
        for l in 0:n-1
            phase=cis(TWO_PI*sym.m*l/n)
            @views U[:,l*nθ+1:(l+1)*nθ].=phase.*F
        end
        out[j]=U
    end
    return out,g.r,θ
end

################################################################################

"""
    wavefunction_multi(solver::WiersigKress,ks,xs,pts;ws=nothing,
        exterior_pad=0.35,npanels_init=3000,M_init=5,rmin_factor=0.9,
        rmax_pad=1.1,nx_min=512,ny_min=512,cheb_tol=1e-10)

Reconstruct a batch of Wiersig dielectric resonant fields.

Without symmetry the complete Cartesian target grid is reconstructed. Reflection
symmetries use a Cartesian half/quarter fundamental domain, while Rotation uses
one polar wedge. The boundary trace is expanded to the complete physical source
boundary; only the target reconstruction is symmetry reduced.

For Cartesian reconstruction returns `Psi,x_grid,y_grid`.
For rotational reconstruction returns `Psi,r_grid,θ_grid`.
"""
function wavefunction_multi(solver::WiersigKress,ks::AbstractVector,xs::AbstractVector{<:AbstractVector},pts::AbstractVector{<:BoundaryPointsCFIE{T}};ws=nothing,b::Union{Real,Symbol}=:auto,exterior_pad::Real=0.35,npanels_init::Int=3000,M_init::Int=5,rmin_factor::Real=0.9,rmax_pad::Real=1.1,nx_min=512,ny_min=512,cheb_tol=1e-10) where {T<:Real}
    ns=length(ks);ns==length(xs)||throw(DimensionMismatch("ks and xs must have equal length"))
    dws=isnothing(ws) ? build_cfie_kress_workspace(solver,pts) : ws
    c=build_wiersig_wavefunction_cache(pts,dws);C=length(c.offsets)-1
    length(solver.billiards)==C||throw(DimensionMismatch("solver has $(length(solver.billiards)) cavities but workspace contains $C"))
    g=_wiersig_wavefunction_grid(solver.symmetry,c,nx_min,ny_min,exterior_pad)
    labels=_wiersig_interior_grid(solver,g)
    full_xs=_wiersig_symmetrize_density(solver,xs,pts,dws)
    coef=_wiersig_coefficients(solver,ks,full_xs,c)
    pl=build_wiersig_wavefunction_plans(solver,ks,c,g;npanels_init=npanels_init,M_init=M_init,rmin_factor=rmin_factor,rmax_pad=rmax_pad,cheb_tol=cheb_tol)
    Psi=g isa WiersigCartesianGrid ? [Matrix{Complex{T}}(undef,length(g.x),length(g.y)) for _ in 1:ns] : [Matrix{Complex{T}}(undef,length(g.r),length(g.θ)) for _ in 1:ns]
    _wiersig_reconstruct!(Psi,g,labels,c,pl,coef)
    return _wiersig_expand_wavefunction(Psi,g,solver.symmetry)
end

"""
    plot_dielectric_wavefunctions(ks,Psi,x_grid,y_grid,pts;...)

Plot Cartesian dielectric wavefunction intensities |ψ|².
"""
function plot_dielectric_wavefunctions(ks::AbstractVector,Psi::AbstractVector{<:AbstractMatrix},x_grid::AbstractVector,y_grid::AbstractVector,pts;maxcols::Int=3,panel_size::Int=600,gap::Int=2,quantile_clip::Real=0.995,colormap=:inferno,boundary_color=:white,boundary_linewidth::Real=2,savepath::Union{Nothing,AbstractString}=nothing)
    n=length(ks);0<quantile_clip<=1||throw(ArgumentError("quantile_clip must lie in (0,1]"))
    nx,ny=length(x_grid),length(y_grid);nc=min(n,maxcols);nr=cld(n,nc)
    width=panel_size*nc+gap*(nc-1);height=panel_size*nr+gap*(nr-1)
    fig=Figure(size=(width,height),resolution=(width,height))
    @inbounds for m in 1:n
        r=cld(m,nc);c=mod1(m,nc)
        ax=Axis(fig[r,c],leftspinevisible=false,rightspinevisible=false,topspinevisible=false,bottomspinevisible=false)
        z=abs2.(Psi[m]);q=quantile(vec(z),quantile_clip);q>0||(q=maximum(z));q>0||(q=one(eltype(z)))
        heatmap!(ax,x_grid,y_grid,z;colormap=colormap,colorrange=(0,q))
        for p in pts
            xy=p.xy;lines!(ax,getindex.(xy,1),getindex.(xy,2);color=boundary_color,linewidth=boundary_linewidth)
        end
        xlims!(ax,first(x_grid),last(x_grid));ylims!(ax,first(y_grid),last(y_grid));hidedecorations!(ax)
    end
    for c in 1:nc;colsize!(fig.layout,c,Fixed(panel_size));end
    for r in 1:nr;rowsize!(fig.layout,r,Fixed(panel_size));end
    colgap!(fig.layout,gap);rowgap!(fig.layout,gap)
    !isnothing(savepath)&&save(savepath,fig)
    return fig
end

"""
    plot_dielectric_wavefunctions_polar(ks,Psi,r_grid,θ_grid,pts;...)

Plot rotational dielectric wavefunction intensities stored on the full polar grid.
"""
function plot_dielectric_wavefunctions_polar(ks::AbstractVector,Psi::AbstractVector{<:AbstractMatrix},r_grid::AbstractVector,θ_grid::AbstractVector,pts;center=(0.0,0.0),maxcols::Int=3,panel_size::Int=600,gap::Int=2,quantile_clip::Real=0.995,colormap=:inferno,boundary_color=:white,boundary_linewidth::Real=2,savepath::Union{Nothing,AbstractString}=nothing)
    n=length(ks);0<quantile_clip<=1||throw(ArgumentError("quantile_clip must lie in (0,1]"))
    nr=length(r_grid);nθ=length(θ_grid);nc=min(n,maxcols);nrows=cld(n,nc)
    width=panel_size*nc+gap*(nc-1);height=panel_size*nrows+gap*(nrows-1);cx,cy=center

    # Convert physical Cartesian boundary points to polar plotting coordinates.
    bpolar=map(pts) do p
        θ=Vector{Float64}(undef,length(p.xy));r=similar(θ)
        @inbounds for j in eachindex(p.xy)
            dx=p.xy[j][1]-cx;dy=p.xy[j][2]-cy;θ[j]=mod(atan(dy,dx),TWO_PI);r[j]=hypot(dx,dy)
        end
        θ,r
    end
    fig=Figure(size=(width,height),resolution=(width,height))
    @inbounds for m in 1:n
        row=cld(m,nc);col=mod1(m,nc)
        ax=PolarAxis(fig[row,col],rticksvisible=false,thetaticksvisible=false,rgridvisible=false,thetagridvisible=false)
        z=abs2.(Psi[m]);q=quantile(vec(z),quantile_clip);q>0||(q=maximum(z));q>0||(q=one(eltype(z)))
        heatmap!(ax,θ_grid,r_grid,permutedims(z);colormap=colormap,colorrange=(0,q))
        for (θb,rb) in bpolar
            lines!(ax,θb,rb;color=boundary_color,linewidth=boundary_linewidth)
        end
        rlims!(ax,first(r_grid),last(r_grid))
    end
    for c in 1:nc;colsize!(fig.layout,c,Fixed(panel_size));end
    for r in 1:nrows;rowsize!(fig.layout,r,Fixed(panel_size));end
    colgap!(fig.layout,gap);rowgap!(fig.layout,gap)
    !isnothing(savepath)&&save(savepath,fig)
    return fig
end