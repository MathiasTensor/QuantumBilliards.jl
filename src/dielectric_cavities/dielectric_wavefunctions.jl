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
#
# Implementation:
# 1. flatten BoundaryPointsCFIE into the exact Wiersig density ordering
#    (`build_wiersig_wavefunction_cache`);
# 2. classify each Cartesian target as exterior or belonging to cavity Ω_a
#    (`_wiersig_interior_grid`);
# 3. expand symmetry-reduced traces to the complete physical boundary
#    (`_wiersig_symmetrize_density`);
# 4. precompute the state/source amplitudes Din,Sin,Dout,Sout
#    (`_wiersig_coefficients`);
# 5. build and pack piecewise-Chebyshev plans for H₀⁽¹⁾(qr),H₁⁽¹⁾(qr)
#    (`build_wiersig_wavefunction_plans`, `_wiersig_panel`);
# 6. use direct Hankel evaluation outside the planned radial interval
#    (`_wiersig_direct_h01`);
# 7. reconstruct with loop order target -> source -> state, reusing r, inner/r
#    and the Chebyshev panel across all resonant states
#    (`_wiersig_reconstruct!` and the `_wiersig_accumulate_*` kernels).
#
# Numerical boundary pieces and physical dielectric cavities are distinct.
# `pts` supplies quadrature nodes, while workspace offsets determine the complete
# source range Γ_a of each physical cavity (`_wiersig_component_range`).
struct WiersigWavefunctionCache{T<:Real}
    # (x,y) coords on the boundary
    x::Vector{T} 
    y::Vector{T}
    # (tx,ty) tangets at the coords above
    tx::Vector{T}
    ty::Vector{T}
    # arclength at the (x,y) above
    s::Vector{T}
    # weights
    w::Vector{T}
    # if multicavity offsets for when geometries start and end as global indexes
    offsets::Vector{Int}
    # bdry limits - plotting params
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    hmin::T
end

# Flatten BoundaryPointsCFIE objects without changing trace ordering.
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
            s[j]=hypot(t[1],t[2]);w[j]=p.ws[l]
            j+=1
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

# Classify Cartesian targets by physical region:
#
#   label=0  exterior Ω₀,
#   label=a  dielectric cavity Ω_a.
#
# Use the original billiard geometry, not the quadrature node representation.
function _wiersig_interior_grid(solver::WiersigKress,x_grid::AbstractVector{T},y_grid::AbstractVector{T}) where {T<:Real}
    pts=[SVector{2,T}(x,y) for y in y_grid for x in x_grid]
    labels=zeros(Int,length(pts))
    grd=max(1000,round(Int,sqrt(length(pts))))
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
# hνin[d,m,p,a] : coefficient d, state m, radial panel p, cavity a,
# hνout[d,m,p]  : common exterior.
# All states share one radial partition [rmin,rmax].
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

# Build packed Chebyshev plans for q_{a,m}=n_a k_m, q_{0,m}=n_out k_m.
# First tune a common (npanels,M) against all material wavenumbers so that
# H₀⁽¹⁾(qr),H₁⁽¹⁾(qr) satisfy cheb_tol on [rmin,rmax], then build the final
# packed plans used during reconstruction.
function build_wiersig_wavefunction_plans(solver::WiersigKress,ks,c::WiersigWavefunctionCache,x_grid,y_grid;cheb_tol::Real=1e-10,npanels_init::Int=3000,M_init::Int=5,sampling_points::Int=20_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,rmin_factor::Real=0.85,rmax_pad::Real=1.1,verbose::Bool=false)
    C=length(c.offsets)-1;ns=length(ks) # number of physical cavities and resonant states
    nin=_wiersig_component_indices(solver,C) # interior refractive index of each physical cavity
    qin=Matrix{ComplexF64}(undef,C,ns);qout=Vector{ComplexF64}(undef,ns) # q_a=n_a*k_m and q_0=n_out*k_m
    qs=Vector{ComplexF64}(undef,ns*(C+1)) # flattened material wavenumbers used by all plans
    qmin=Inf;l=1
    @inbounds for m in 1:ns
        k=ComplexF64(ks[m]) # resonance k_m
        for a in 1:C
            q=ComplexF64(nin[a])*k
            qin[a,m]=q;qs[l]=q;l+=1
            qmin=min(qmin,abs(q))
        end
        q=ComplexF64(solver.n_out)*k
        qout[m]=q;qs[l]=q;l+=1
        qmin=min(qmin,abs(q))
    end
    zcut=Float64(hankel_z_chebyshev_cutoff) # lower |qr| limit for regular Hankel interpolation
    rmin=max(Float64(rmin_factor*c.hmin),zcut/qmin,1e-12) # common lower radial interpolation bound
    rmax=Float64(rmax_pad)*hypot(Float64(last(x_grid)-first(x_grid)),Float64(last(y_grid)-first(y_grid))) # largest required source-target radius
    rmax>rmin||throw(ArgumentError("invalid radial interval [$rmin,$rmax]"))
    rs=collect(range(rmin,rmax;length=sampling_points)) # radii used to validate H₀/H₁ interpolation
    npanels=npanels_init;M=M_init
    plans0=Vector{ChebHankelPlanH}(undef,length(qs))
    plans1=Vector{ChebHankelPlanH}(undef,length(qs))
    err0=fill(Inf,length(qs));err1=fill(Inf,length(qs))
    converged=false
    for it in 1:max_iter
        p=Progress(max_iter,desc="Refining chebyshev plans iter=$it",showspeed=true)
        Threads.@threads for j in eachindex(qs)
            plans0[j]=plan_h(0,1,qs[j],rmin,rmax;npanels=npanels,M=M)
            plans1[j]=plan_h(1,1,qs[j],rmin,rmax;npanels=npanels,M=M)
        end
        _check_H0H1_errors!(err0,err1,plans0,plans1,qs,rs)
        worst=max(maximum(err0),maximum(err1))
        verbose&&@info "Wiersig wavefunction Chebyshev tuning" worst npanels M
        if worst<cheb_tol
            converged=true
            break
        end
        it%5==0 ? (M+=grow_M) : (npanels=ceil(Int,grow_panels*npanels))
    end
    converged||@warn "Wavefunction Chebyshev tuning did not reach tol=$cheb_tol after $max_iter iterations."
    h0in=Array{ComplexF64}(undef,M+1,ns,npanels,C);h1in=similar(h0in) # packed interior H₀/H₁ coefficients
    h0out=Array{ComplexF64}(undef,M+1,ns,npanels);h1out=similar(h0out) # packed exterior H₀/H₁ coefficients
    l=1
    @inbounds for m in 1:ns
        for a in 1:C
            p0=plans0[l];p1=plans1[l] # already validated plans for q_a=n_a*k_m
            for p in 1:npanels
                copyto!(@view(h0in[:,m,p,a]),p0.panels[p].c)
                copyto!(@view(h1in[:,m,p,a]),p1.panels[p].c)
            end
            l+=1
        end
        p0=plans0[l];p1=plans1[l] # already validated plans for q_out=n_out*k_m
        for p in 1:npanels
            copyto!(@view(h0out[:,m,p]),p0.panels[p].c)
            copyto!(@view(h1out[:,m,p]),p1.panels[p].c)
        end
        l+=1
    end
    h=(rmax-rmin)/npanels # uniform radial panel width
    return WiersigWavefunctionPlans(h0in,h1in,h0out,h1out,qin,qout,rmin,rmax,npanels,h,inv(h))
end

# Map r to radial panel p and local Chebyshev coordinate. Just an overload that takes the useful struct here.
# t=2(r-center_p)/h ∈ [-1,1]. p=0 selects direct special-function evaluation.
@inline function _wiersig_panel(pl::WiersigWavefunctionPlans,r::Float64)
    (r<pl.rmin||r>pl.rmax)&&return 0,0.0
    p=clamp(Int(floor((r-pl.rmin)*pl.invh))+1,1,pl.npanels)
    t=2*(r-(pl.rmin+(p-0.5)*pl.h))*pl.invh
    return p,t
end

# Expand reduced Wiersig traces to the complete (φ,ψ) vector of length 2N.
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

# Precomputed source coefficients for all states:
# Din =(-iq_a/4)wψ, Sin =(+iχ_a/4)wsφ,
# Dout=(+iq_0/4)wψ, Sout=(-iχ_0/4)wsφ.
struct WiersigWavefunctionCoefficients{T<:Real}
    Din::Matrix{Complex{T}}
    Sin::Matrix{Complex{T}}
    Dout::Matrix{Complex{T}}
    Sout::Matrix{Complex{T}}
end

function _wiersig_coefficients(solver::WiersigKress,ks,xs,c::WiersigWavefunctionCache{T}) where {T<:Real}
    ns=length(ks);N=length(c.x);C=length(c.offsets)-1
    nin=_wiersig_component_indices(solver,C)
    Din=Matrix{Complex{T}}(undef,ns,N);Sin=similar(Din)
    Dout=similar(Din);Sout=similar(Din)
    χout=T(_wiersig_slp_factor(solver,solver.n_out))
    @inbounds for m in 1:ns
        φ,ψ=split_wiersig_trace(xs[m])
        k=Complex{T}(ks[m]);qout=Complex{T}(solver.n_out)*k
        fDo=Complex{T}(0,one(T)/4)*qout # +iq_out/4
        fSo=-Complex{T}(0,χout/4) # -iχ_out/4
        for j in 1:N
            Dout[m,j]=fDo*c.w[j]*ψ[j]
            Sout[m,j]=fSo*c.w[j]*c.s[j]*φ[j]
        end
        for a in 1:C
            q=Complex{T}(nin[a])*k;χ=T(_wiersig_slp_factor(solver,nin[a]))
            fDi=-Complex{T}(0,one(T)/4)*q # -iq_a/4
            fSi=Complex{T}(0,χ/4) # +iχ_a/4
            for j in _wiersig_component_range(c,a)
                Din[m,j]=fDi*c.w[j]*ψ[j]
                Sin[m,j]=fSi*c.w[j]*c.s[j]*φ[j]
            end
        end
    end
    return WiersigWavefunctionCoefficients(Din,Sin,Dout,Sout)
end

# Direct H₀/H₁ fallback outside the Chebyshev interval; use local series for very small |qr|.
@inline function _wiersig_direct_h01(q::ComplexF64,r::Float64)
    z=q*r # complex Hankel argument
    abs(z)<hankel_z_chebyshev_cutoff_small_z&&return _small_h0_series(z),_small_h1_series(z)
    return SpecialFunctions.besselh(0,1,z),SpecialFunctions.besselh(1,1,z)
end

# Exterior Chebyshev contribution:
# u += H₁(qr)*(inner/r)*Dout + H₀(qr)*Sout.
@inline function _wiersig_accumulate_out_cheb!(acc,j,p,t,inner,pl,c)
    @inbounds @fastmath for m in eachindex(acc)
        h0=_cheb_clenshaw(@view(pl.h0out[:,m,p]),t) # H₀⁽¹⁾(q_out r)
        h1=_cheb_clenshaw(@view(pl.h1out[:,m,p]),t) # H₁⁽¹⁾(q_out r)
        acc[m]+=inner*h1*c.Dout[m,j]+h0*c.Sout[m,j] # DLP + SLP contribution
    end
    return nothing
end

# Interior Chebyshev contribution for physical cavity a.
@inline function _wiersig_accumulate_in_cheb!(acc,j,p,a,t,inner,pl,c)
    @inbounds @fastmath for m in eachindex(acc)
        h0=_cheb_clenshaw(@view(pl.h0in[:,m,p,a]),t) # H₀⁽¹⁾(q_a r)
        h1=_cheb_clenshaw(@view(pl.h1in[:,m,p,a]),t) # H₁⁽¹⁾(q_a r)
        acc[m]+=inner*h1*c.Din[m,j]+h0*c.Sin[m,j] # DLP + SLP contribution
    end
    return nothing
end

# Exterior direct-Hankel contribution.
@inline function _wiersig_accumulate_out_direct!(acc,j,r,inner,pl,c)
    @inbounds for m in eachindex(acc)
        h0,h1=_wiersig_direct_h01(pl.qout[m],r) # H₀⁽¹⁾(q_out r),H₁⁽¹⁾(q_out r)
        acc[m]+=inner*h1*c.Dout[m,j]+h0*c.Sout[m,j] # DLP + SLP contribution
    end
    return nothing
end

# Interior direct-Hankel contribution for physical cavity a.
@inline function _wiersig_accumulate_in_direct!(acc,j,r,a,inner,pl,c)
    @inbounds for m in eachindex(acc)
        h0,h1=_wiersig_direct_h01(pl.qin[a,m],r) # H₀⁽¹⁾(q_a r),H₁⁽¹⁾(q_a r)
        acc[m]+=inner*h1*c.Din[m,j]+h0*c.Sin[m,j] # DLP + SLP contribution
    end
    return nothing
end

# Each source-target geometry quantity is computed once and reused for every
# resonance. Exterior targets sum over all physical boundaries; targets inside
# Ω_a use only Γ_a.
function _wiersig_reconstruct!(Psi,x_grid,y_grid,labels,c::WiersigWavefunctionCache{T},pl::WiersigWavefunctionPlans,coef::WiersigWavefunctionCoefficients{T}) where {T<:Real}
    nx=length(x_grid);ns=size(coef.Din,1);N=length(c.x);eps2=eps(T)^2
    accs=[zeros(Complex{T},ns) for _ in 1:Threads.nthreads()] # thread-local state accumulators
    @showprogress desc="Wavefunction construction" Threads.@threads :static for idx in eachindex(labels)
        acc=accs[Threads.threadid()]
        fill!(acc,zero(Complex{T})) # reset all states for this target
        iy=div(idx-1,nx)+1;ix=idx-(iy-1)*nx # flattened target index -> Cartesian indices
        xp=x_grid[ix];yp=y_grid[iy];a=labels[idx] # target coordinates and physical region
        if a==0 # exterior Ω₀: all cavity boundaries contribute
            @inbounds for j in 1:N
                dx=xp-c.x[j];dy=yp-c.y[j];r2=muladd(dx,dx,dy*dy) # source-target distance²
                r2<=eps2&&continue
                r=sqrt(r2);inner=muladd(c.ty[j],dx,-c.tx[j]*dy)/r # s_j n_j⋅(x-y_j)/r
                rf=Float64(r);p,t=_wiersig_panel(pl,rf) # radial interpolation panel
                p==0 ? _wiersig_accumulate_out_direct!(acc,j,rf,inner,pl,coef) : _wiersig_accumulate_out_cheb!(acc,j,p,t,inner,pl,coef)
            end
        else # interior Ω_a: only Γ_a contributes
            @inbounds for j in _wiersig_component_range(c,a)
                dx=xp-c.x[j];dy=yp-c.y[j];r2=muladd(dx,dx,dy*dy)
                r2<=eps2&&continue
                r=sqrt(r2);inner=muladd(c.ty[j],dx,-c.tx[j]*dy)/r
                rf=Float64(r);p,t=_wiersig_panel(pl,rf)
                p==0 ? _wiersig_accumulate_in_direct!(acc,j,rf,a,inner,pl,coef) : _wiersig_accumulate_in_cheb!(acc,j,p,a,t,inner,pl,coef)
            end
        end
        @inbounds for m in 1:ns
            Psi[m][idx]=acc[m] # store reconstructed fields at this target
        end
    end
    return Psi
end

"""
        wavefunction_multi(solver::WiersigKress,ks::AbstractVector,xs::AbstractVector{<:AbstractVector},pts::AbstractVector{<:BoundaryPointsCFIE{T}};ws=nothing,b::Union{Real,Symbol}=:auto,exterior_pad::Real=0.35,npanels_init::Int=3000,M_init::Int=5,rmin_factor::Real=0.85,rmax_pad::Real=1.1,nx_min=512,ny_min=512,cheb_tol=1e-10) where {T<:Real}

Reconstruct a batch of Wiersig dielectric resonant fields on one Cartesian grid.

For physical cavities Ω_a with refractive indices n_a and common exterior index
n_out, the resonant fields satisfy

    (Δ+n_a²k²)u_a=0,       x∈Ω_a,
    (Δ+n_out²k²)u_0=0,     x∈Ω_0,

with outgoing radiation in the exterior. From the Wiersig boundary trace
`x=(φ,ψ)`, the fields are reconstructed through

    u_a(x)= +(1/2)∫_{Γ_a}[D_{n_ak}(x,y)ψ_a(y)+χ_aS_{n_ak}(x,y)φ_a(y)]dt,
    u_0(x)= -(1/2)Σ_a∫_{Γ_a}[D_{n_outk}(x,y)ψ_a(y)+χ_outS_{n_outk}(x,y)φ_a(y)]dt,
    D_q(x,y)=-(iq/2)H₁⁽¹⁾(q|x-y|) inner/|x-y|,
    S_q(x,y)= (i/2)H₀⁽¹⁾(q|x-y|)|γ'(t_y)|,

and `χ(n)=1` for TM, `χ(n)=n²` for TE.

# Arguments
- `solver::WiersigKress`: dielectric Wiersig-Kress solver.
- `ks`: complex resonant wavenumbers.
- `xs`: corresponding Wiersig boundary traces.
- `pts`: boundary quadrature generated for the current discretization.

# Keywords
- `ws=nothing`: existing Wiersig geometry workspace. Rebuilt only when omitted.
- `b=:auto`: Cartesian points per shortest wavelength. `:auto` reuses the
  quadrature resolution parameter.
- `exterior_pad=0.35`: fractional padding around the physical bounding box.
- `npanels_init=3000`: initial radial Chebyshev panel count; increased automatically until `cheb_tol` is reached.
- `M_init=5`: initial Chebyshev polynomial degree; increased automatically if additional panel refinement is insufficient.
- `cheb_tol=1e-10`: required maximum absolute H₀/H₁ interpolation error over all material wavenumbers used in the reconstruction.
- `rmin_factor=0.85`: lower-radius safety factor relative to boundary spacing.
- `rmax_pad=1.1`: padding of the maximum source-target radial interval.

# Returns
- `Psi`: vector of complex field matrices, one per resonance.
- `x_grid`: Cartesian x coordinates.
- `y_grid`: Cartesian y coordinates.
"""
function wavefunction_multi(solver::WiersigKress,ks::AbstractVector,xs::AbstractVector{<:AbstractVector},pts::AbstractVector{<:BoundaryPointsCFIE{T}};ws=nothing,b::Union{Real,Symbol}=:auto,exterior_pad::Real=0.35,npanels_init::Int=3000,M_init::Int=5,rmin_factor::Real=0.9,rmax_pad::Real=1.1,nx_min=512,ny_min=512,cheb_tol=1e-10) where {T<:Real}
    ns=length(ks)
    dws=isnothing(ws) ? build_cfie_kress_workspace(solver,pts) : ws
    c=build_wiersig_wavefunction_cache(pts,dws)
    C=length(c.offsets)-1
    length(solver.billiards)==C||throw(DimensionMismatch("solver has $(length(solver.billiards)) cavities but workspace contains $C"))
    nin=_wiersig_component_indices(solver,C)
    qmax=0.0
    @inbounds for k in ks
        kc=ComplexF64(k)
        qmax=max(qmax,abs(ComplexF64(solver.n_out)*kc))
        for a in 1:C
            qmax=max(qmax,abs(ComplexF64(nin[a])*kc))
        end
    end
    bval=b===:auto ? maximum(solver.ppw) : Float64(b) # either use a custom b or choose one from solver. Just to sometimes have faster runtime if not so precise wavefunctions are needed.
    dx=c.xmax-c.xmin;dy=c.ymax-c.ymin # physical grid extent in x and y
    xpad=T(exterior_pad)*dx;ypad=T(exterior_pad)*dy # pad the domain to include the exterior radiation pattern
    xlim=(c.xmin-xpad,c.xmax+xpad);ylim=(c.ymin-ypad,c.ymax+ypad) # padded Cartesian limits
    #nx=max(round(Int,qmax*(xlim[2]-xlim[1])*bval/TWO_PI),nx_min) # x-grid resolution from points per shortest wavelength
    #ny=max(round(Int,qmax*(ylim[2]-ylim[1])*bval/TWO_PI),ny_min) # y-grid resolution from points per shortest wavelength
    nx=nx_min;ny=ny_min # fixed grid resolution for now, otherwise this explodes the total cost, as O(Nx * Ny * N) = O(k^3) for N boundary points. This is a lot of work for a single wavefunction, and we have to do this for every resonance.
    x_grid=collect(T,range(xlim...;length=nx));y_grid=collect(T,range(ylim...;length=ny)) # equispaced Cartesian reconstruction grid
    labels=_wiersig_interior_grid(solver,x_grid,y_grid) # classify each target as exterior or belonging to cavity a (for each a)
    full_xs=_wiersig_symmetrize_density(solver,xs,pts,dws) # expand symmetry-reduced density to the full physical boundary
    coef=_wiersig_coefficients(solver,ks,full_xs,c) # precompute D/S Green source coefficients for all states
    pl=build_wiersig_wavefunction_plans(solver,ks,c,x_grid,y_grid;npanels_init=npanels_init,M_init=M_init,rmin_factor=rmin_factor,rmax_pad=rmax_pad,cheb_tol=cheb_tol) # build packed H₀/H₁ Chebyshev plans up to desired tolerance
    Psi=[Matrix{Complex{T}}(undef,nx,ny) for _ in 1:ns] # allocate one complex field matrix per resonance
    _wiersig_reconstruct!(Psi,x_grid,y_grid,labels,c,pl,coef) # reconstruct all states with target -> source -> state loop order
    return Psi,x_grid,y_grid
end

"""

    plot_wiersig_wavefunctions(ks::AbstractVector,Psi::AbstractVector{<:AbstractMatrix},x_grid::AbstractVector,y_grid::AbstractVector,pts;maxcols::Int=3,panel_size::Int=600,gap::Int=2,quantile_clip::Real=0.995,colormap=:inferno,boundary_color=:white,boundary_linewidth::Real=2,savepath::Union{Nothing,AbstractString}=nothing)

Each resonance is displayed as the intensity |ψ(x,y)|² on a fixed-size square panel.

# Arguments
- `ks::AbstractVector`: Complex resonance wavenumbers. Used to determine and validate the number of states.
- `Psi::AbstractVector{<:AbstractMatrix}`: Complex wavefunctions, one matrix per resonance.
- `x_grid::AbstractVector`, `y_grid::AbstractVector`: Cartesian reconstruction grids.
- `pts`: Full physical boundary discretization. Every entry must provide an `xy` field containing the boundary nodes.

# Keywords
- `maxcols=3`: Maximum number of panels per row.
- `panel_size=600`: Width and height in pixels of each wavefunction panel.
- `gap=2`: Gap in pixels between neighboring panels.
- `quantile_clip=0.995`: quantile used as the upper color limit for each state.
- `colormap=:inferno`: Makie colormap used for the intensity.
- `boundary_color=:white`: Boundary overlay color.
- `boundary_linewidth=2`: Boundary overlay linewidth.
- `savepath=nothing`: Optional output path. If supplied, the figure is saved before being returned.

# Returns
- `Figure`:
  The Makie figure containing all resonance wavefunctions.
"""
function plot_dielectric_wavefunctions(ks::AbstractVector,Psi::AbstractVector{<:AbstractMatrix},x_grid::AbstractVector,y_grid::AbstractVector,pts;maxcols::Int=3,panel_size::Int=600,gap::Int=2,quantile_clip::Real=0.995,colormap=:inferno,boundary_color=:white,boundary_linewidth::Real=2,savepath::Union{Nothing,AbstractString}=nothing)
    n=length(ks)
    0<quantile_clip<=1||throw(ArgumentError("quantile_clip must lie in (0,1]"))
    nx,ny=length(x_grid),length(y_grid)
    nc=min(n,maxcols);nr=cld(n,nc)
    width=panel_size*nc+gap*(nc-1);height=panel_size*nr+gap*(nr-1)
    fig=Figure(size=(width,height),resolution=(width,height))
    @inbounds for m in 1:n
        r=cld(m,nc);c=mod1(m,nc)
        ax=Axis(fig[r,c],leftspinevisible=false,rightspinevisible=false,topspinevisible=false,bottomspinevisible=false)
        z=abs2.(Psi[m])
        q=quantile(vec(z),quantile_clip)
        q>0||(q=maximum(z))
        q>0||(q=one(eltype(z)))
        heatmap!(ax,x_grid,y_grid,z;colormap=colormap,colorrange=(0,q))
        for p in pts
            xy=p.xy;lines!(ax,getindex.(xy,1),getindex.(xy,2);color=boundary_color,linewidth=boundary_linewidth)
        end
        xlims!(ax,first(x_grid),last(x_grid));ylims!(ax,first(y_grid),last(y_grid))
        hidedecorations!(ax)
    end
    for c in 1:nc;colsize!(fig.layout,c,Fixed(panel_size));end
    for r in 1:nr;rowsize!(fig.layout,r,Fixed(panel_size));end
    colgap!(fig.layout,gap);rowgap!(fig.layout,gap)
    !isnothing(savepath)&&save(savepath,fig)
    return fig
end