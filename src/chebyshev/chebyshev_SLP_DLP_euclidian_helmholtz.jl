#################################################################
#   CHEBYSHEV-BASED SLP/DLP EVALUATION FOR CFIE ASSEMBLY IN 2D EUCLIDEAN HELMHOLTZ 
# Functions to build Chebyshev-based SLP/DLP evaluation plans for multiple wavenumbers, and to compute the CFIE matrix blocks using these plans. 
# Logic:
# - Build Chebyshev-based Hankel evaluation plans for the SLP and DLP kernels for each wavenumber.
# - Precompute the geometry-related terms (R, invR, inner product, speed, quadrature weights) for each block of the CFIE matrix.
# - For each block, use the precomputed geometry and the Chebyshev plans to evaluate the SLP and DLP contributions for each pair of points, and accumulate the CFIE matrix entries.
#
# API: 
# - `build_CFIE_plans(...)`: Builds Chebyshev-based Hankel evaluation plans for the given wavenumbers and geometry.
# - `build_cfie_block_caches(...)`: Precomputes the geometry-related terms for each block of the CFIE matrix.
# - `h01_multi_ks_at_r!(...)`: Evaluates the SLP and DLP Hankel functions for multiple wavenumbers at given distances.
# - `compute_kernel_matrices_CFIE_chebyshev!(...)`: Main function to compute the CFIE matrix blocks for all wavenumbers, using the appropriate method based on the presence of symmetries and the number of wavenumbers.
# USUALLY NOT CALLED DIRECTLY: 
# - `_accum_cfie_default_sym!(...)`: Default function to accumulate CFIE matrix entries for a given pair of points and their geometric terms.
# - `_one_k_nosymm_CFIE_chebyshev!(...)`: Computes the CFIE matrix blocks for a single wavenumber without using symmetries, using Chebyshev-based SLP/DLP evaluation.
# - `_all_k_nosymm_CFIE_chebyshev!(...)`: Computes the CFIE matrix blocks for all wavenumbers without using symmetries, using Chebyshev-based SLP/DLP evaluation.
#
# Workflow: 
# 1. Call `build_CFIE_plans` to create the Chebyshev evaluation plans for the desired wavenumbers and geometry.
# 2. Call `build_cfie_block_caches` to precompute the geometry-related terms for each block of the CFIE matrix.
# 3. Call `compute_kernel_matrices_CFIE_chebyshev!` to compute the CFIE matrix blocks for all wavenumbers, which will internally call the appropriate function based on the presence of symmetries and the number of wavenumbers.
#
# MO 24/3/26
#################################################################

_TWO_PI=2*pi
_INV_TWO_PI=1/_TWO_PI
_EULER_OVER_PI=MathConstants.eulergamma/pi

#######################
#### MULTI-K H0/H1 ####
#######################

@inline function h01_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH1},plans1::AbstractVector{ChebHankelPlanH1},pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans0)
        h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
        h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
    end
    return nothing
end

################################
#### PLAN BUILDERS FOR CFIE ####
################################

function build_CFIE_plans(ks::AbstractVector{<:Number},rmin::Float64,rmax::Float64;npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,nthreads::Int=1)::Tuple{Vector{ChebHankelPlanH1},Vector{ChebHankelPlanH1}}
    Mk=length(ks)
    plans0=Vector{ChebHankelPlanH1}(undef,Mk)
    plans1=Vector{ChebHankelPlanH1}(undef,Mk)
    if nthreads<=1 || Mk==1
        @inbounds for m in 1:Mk
            k=ComplexF64(ks[m])
            plans0[m]=plan_h1(0,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            plans1[m]=plan_h1(1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
        end
    else
        nt=min(nthreads,Mk)
        chunks=Vector{UnitRange{Int}}(undef,nt)
        base=div(Mk,nt)
        remn=rem(Mk,nt)
        s=1
        for t in 1:nt
            len=base+(t<=remn ? 1 : 0)
            chunks[t]=s:(s+len-1)
            s+=len
        end
        Threads.@threads for tid in 1:nt
            @inbounds for m in chunks[tid]
                k=ComplexF64(ks[m])
                plans0[m]=plan_h1(0,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
                plans1[m]=plan_h1(1,k,rmin,rmax;npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
            end
        end
    end
    return plans0,plans1
end

#################################
#### COMPONENT / BLOCK CACHE ####
#################################

struct CFIEBlockCache{T<:Real}
    same::Bool
    row_offset::Int
    col_offset::Int
    Ni::Int
    Nj::Int
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    speed_j::Vector{T}
    wj::Vector{T}
    pidx::Matrix{Int32}
    tloc::Matrix{Float64}
    logterm::Union{Nothing,Matrix{T}}
    kappa_i::Union{Nothing,Vector{T}}
    Rkress::Union{Nothing,Matrix{T}}
end

struct CFIEBlockSystemCache{T<:Real}
    blocks::Matrix{CFIEBlockCache{T}}
    offsets::Vector{Int}
    rmin::Float64
    rmax::Float64
end

function build_cfie_block_caches(comps::Vector{BoundaryPointsCFIE{T}};npanels::Int=10000,M::Int=5,grading::Symbol=:uniform,geo_ratio::Real=1.05,pad=(T(0.95),T(1.05))) where {T<:Real}
    nc=length(comps)
    offs=component_offsets(comps)
    blocks=Matrix{CFIEBlockCache{T}}(undef,nc,nc)
    global_rmin=typemax(T)
    global_rmax=zero(T)
    for a in 1:nc, b in 1:nc
        pa=comps[a]
        pb=comps[b]
        Ni=length(pa.xy)
        Nj=length(pb.xy)
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Xb=getindex.(pb.xy,1)
        Yb=getindex.(pb.xy,2)
        dXb=getindex.(pb.tangent,1)
        dYb=getindex.(pb.tangent,2)
        ΔX=@. reshape(Xa,Ni,1)-reshape(Xb,1,Nj)
        ΔY=@. reshape(Ya,Ni,1)-reshape(Yb,1,Nj)
        R=hypot.(ΔX,ΔY)
        invR=similar(R)
        @inbounds for j in 1:Nj, i in 1:Ni
            rij=R[i,j]
            invR[i,j]=rij>eps(T) ? inv(rij) : zero(T)
        end
        dXbr=reshape(dXb,1,Nj)
        dYbr=reshape(dYb,1,Nj)
        inner=@. dYbr*ΔX-dXbr*ΔY
        speed_j=@. sqrt(dXb^2+dYb^2)
        wj=copy(pb.ws)
        same=(a==b)
        rmin_blk=typemax(T)
        rmax_blk=zero(T)
        @inbounds for j in 1:Nj, i in 1:Ni
            if same && i==j
                continue
            end
            rij=R[i,j]
            if rij>eps(T)
                rij<rmin_blk && (rmin_blk=rij)
                rij>rmax_blk && (rmax_blk=rij)
            end
        end
        @assert isfinite(rmin_blk) && rmax_blk>zero(T)
        rmin_blk=pad[1]*rmin_blk
        rmax_blk=pad[2]*rmax_blk
        global_rmin=min(global_rmin,rmin_blk)
        global_rmax=max(global_rmax,rmax_blk)
        pidx=Matrix{Int32}(undef,Ni,Nj)
        tloc=Matrix{Float64}(undef,Ni,Nj)
        if same
            ts=pa.ts
            dXa=getindex.(pa.tangent,1)
            dYa=getindex.(pa.tangent,2)
            ddXa=getindex.(pa.tangent_2,1)
            ddYa=getindex.(pa.tangent_2,2)
            ΔT=@. reshape(ts,Ni,1)-reshape(ts,1,Ni)
            logterm=log.(4 .* sin.(ΔT./2).^2)
            logterm[diagind(logterm)].=zero(T)
            κnum=dXa.*ddYa.-dYa.*ddXa
            κden=dXa.^2 .+ dYa.^2
            kappa_i=_INV_TWO_PI.*(κnum./κden)
            Rkress=zeros(T,Ni,Ni)
            kress_R_fft!(Rkress)
            blocks[a,b]=CFIEBlockCache{T}(true,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_j,wj,pidx,tloc,logterm,kappa_i,Rkress)
        else
            blocks[a,b]=CFIEBlockCache{T}(false,offs[a],offs[b],Ni,Nj,R,invR,inner,speed_j,wj,pidx,tloc,nothing,nothing,nothing)
        end
    end
    pref_plan=plan_h1(0,1.0+0im,Float64(global_rmin),Float64(global_rmax);npanels=npanels,M=M,grading=grading,geo_ratio=geo_ratio)
    pans=pref_plan.panels
    for a in 1:nc, b in 1:nc
        blk=blocks[a,b]
        same=blk.same
        @inbounds for j in 1:blk.Nj, i in 1:blk.Ni
            if same && i==j
                blk.pidx[i,j]=Int32(1)
                blk.tloc[i,j]=0.0
            else
                rij=Float64(blk.R[i,j])
                p=_find_panel(pans,rij)
                P=pans[p]
                blk.pidx[i,j]=Int32(p)
                blk.tloc[i,j]=(2*rij-(P.b+P.a))/(P.b-P.a)
            end
        end
    end
    return CFIEBlockSystemCache{T}(blocks,offs,Float64(global_rmin),Float64(global_rmax))
end

############################
#### EXPLICIT WORKSPACE ####
############################

struct CFIEMultiHankelWorkspace
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
end

function CFIEMultiHankelWorkspace(Mk::Int;ntls::Int=(Threads.nthreads()))
    h0_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    h1_tls=[Vector{ComplexF64}(undef,Mk) for _ in 1:ntls]
    return CFIEMultiHankelWorkspace(h0_tls,h1_tls)
end

###################################
#### DEFAULT CFIE ACCUMULATION ####
###################################

@inline function _accum_cfie_default_sym!(A::AbstractMatrix{ComplexF64},i::Int,j::Int,inn::T,invr::T,sj::T,wj::T,h0::ComplexF64,h1::ComplexF64,k::ComplexF64,scale::ComplexF64) where {T<:Real}
    dterm=(0.5im*k)*(inn*invr)*h1
    sterm=(0.5im)*sj*h0
    @inbounds A[i,j]+= -scale*wj*(dterm+1im*k*sterm)
    return nothing
end

#############################################################
#### DIRECT NO-SYMMETRY CFIE ASSEMBLY: ALL k / ONE k ########
#############################################################

function _all_k_nosymm_CFIE_chebyshev!(As::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH1},plans1::Vector{ChebHankelPlanH1},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    Mk=length(plans0)
    ks=Vector{ComplexF64}(undef,Mk)
    αL1=Vector{ComplexF64}(undef,Mk)
    αL2=Vector{ComplexF64}(undef,Mk)
    iks=Vector{ComplexF64}(undef,Mk)
    multithreaded && @assert length(h0_tls)>=Threads.nthreads() && length(h1_tls)>=Threads.nthreads()
    @inbounds for m in 1:Mk
        km=ComplexF64(plans1[m].k)
        ks[m]=km
        αL1[m]=km*_INV_TWO_PI
        αL2[m]=0.5im*km
        iks[m]=1im*km
        fill!(As[m],0)
    end
    αM1=-_INV_TWO_PI
    αM2=0.5im
    sys=block_cache
    blocks=sys.blocks
    nc=size(blocks,1)

    function same_block_col!(blk::CFIEBlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset;co=blk.col_offset
        sj=blk.speed_j[j];wj=blk.wj[j];gj=co+j-1
        gi=ro+j-1;κj=blk.kappa_i[j];rjj=blk.Rkress[j,j]
        @inbounds for m in 1:Mk
            km=ks[m]
            dval=ComplexF64(wj*κj,0.0)
            m1=αM1*sj
            m2=((0.5im-_EULER_OVER_PI)-_INV_TWO_PI*log((km^2/4)*(sj^2)))*sj
            sval=ComplexF64(rjj*m1,0.0)+wj*m2
            As[m][gi,gj]=1.0-(dval+iks[m]*sval)
        end
        @inbounds for i in (j+1):blk.Ni
            gi=ro+i-1
            invr=blk.invR[i,j];p=blk.pidx[i,j];t=blk.tloc[i,j]
            lt=blk.logterm[i,j];rijR=blk.Rkress[i,j]
            inn_ij=blk.inner[i,j];inn_ji=blk.inner[j,i]
            si=blk.speed_j[i];wi=blk.wj[i]
            h01_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,p,t)
            for m in 1:Mk
                h0=h0vals[m];h1=h1vals[m]
                j0=real(h0);j1=real(h1)
                βL1=αL1[m]*j1*invr
                βL2=αL2[m]*h1*invr
                βM1=αM1*j0
                βM2=αM2*h0
                l1ij=βL1*inn_ij;l1ji=βL1*inn_ji
                l2ij=βL2*inn_ij-l1ij*lt
                l2ji=βL2*inn_ji-l1ji*lt
                dvalij=rijR*l1ij+wj*l2ij
                dvalji=rijR*l1ji+wi*l2ji
                m1j=βM1*sj;m1i=βM1*si
                m2j=βM2*sj-m1j*lt
                m2i=βM2*si-m1i*lt
                svalij=rijR*m1j+wj*m2j
                svalji=rijR*m1i+wi*m2i
                As[m][gi,gj]=-(dvalij+iks[m]*svalij)
                As[m][gj,gi]=-(dvalji+iks[m]*svalji)
            end
        end
        return nothing
    end

    function off_block_col!(blk::CFIEBlockCache{T},j::Int,h0vals::Vector{ComplexF64},h1vals::Vector{ComplexF64}) where {T<:Real}
        ro=blk.row_offset;co=blk.col_offset
        sj=blk.speed_j[j];wj=blk.wj[j];gj=co+j-1
        @inbounds for i in 1:blk.Ni
            gi=ro+i-1
            invr=blk.invR[i,j];inn=blk.inner[i,j]
            p=blk.pidx[i,j];t=blk.tloc[i,j]
            h01_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,p,t)
            for m in 1:Mk
                h0=h0vals[m];h1=h1vals[m]
                βL=αL2[m]*h1*invr
                βM=αM2*h0
                dval=wj*(βL*inn)
                sval=wj*(βM*sj)
                As[m][gi,gj]=-(dval+iks[m]*sval)
            end
        end
        return nothing
    end

    for a in 1:nc
        blk=blocks[a,a]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid()
            same_block_col!(blk,j,h0_tls[tid],h1_tls[tid])
        end
    end

    for a in 1:nc, b in 1:nc
        a==b && continue
        blk=blocks[a,b]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid()
            off_block_col!(blk,j,h0_tls[tid],h1_tls[tid])
        end
    end
    return nothing
end

function _one_k_nosymm_CFIE_chebyshev!(A::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH1,plan1::ChebHankelPlanH1,h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_CFIE_chebyshev!([A],pts,[plan0],[plan1],h0_tls,h1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

#################################
#### HIGH LEVEL ENTRY POINTS ####
#################################

#############################################################
# compute_kernel_matrices_CFIE_chebyshev!
#
# Assemble CFIE matrices using Chebyshev-interpolated Hankel kernels.
#
# INPUT
# -----
# As            :: Vector{Matrix{ComplexF64}}
#   Output matrices, one per wavenumber (length = Mk).
#   Each must be preallocated (N×N) and will be overwritten.
#
# A             :: Matrix{ComplexF64}
#   Single-matrix version (Mk = 1).
#
# pts           :: Vector{BoundaryPointsCFIE{T}}
#   Boundary components (outer boundary first, then holes).
#   Each component stores:
#     - xy
#     - tangent
#     - ws
#
# plans0,plans1 :: Vector{ChebHankelPlanH1}
#   Chebyshev interpolation plans for:
#     - H0 (SLP kernel)  -> plans0
#     - H1 (DLP kernel)  -> plans1
#   Must have same length Mk.
#
# plan0,plan1   :: ChebHankelPlanH1
#   Single-k version of the above.
#
# h0_tls,h1_tls :: Vector{Vector{ComplexF64}}
#   Thread-local scratch buffers:
#     - outer length ≥ nthreads()
#     - inner length ≥ Mk
#   Used to store H0/H1 evaluations per thread.
#
# block_cache   :: CFIEBlockSystemCache{T}
#   Precomputed geometry + Kress:
#     - block structure (component offsets)
#     - inner products, invR, log terms
#     - diagonal limits and FFT-based corrections
#
# multithreaded :: Bool (default=true)
#   Enables threading over columns j=1:N.
#
# OUTPUT
# ------
# As / A are filled in-place with CFIE matrices:
#
#   A = I - (D + i*k*S)
#
# where:
#   D = double-layer operator
#   S = single-layer operator
#############################################################

function compute_kernel_matrices_CFIE_chebyshev!(As::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},plans0::Vector{ChebHankelPlanH1},plans1::Vector{ChebHankelPlanH1},h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _all_k_nosymm_CFIE_chebyshev!(As,pts,plans0,plans1,h0_tls,h1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end

function compute_kernel_matrices_CFIE_chebyshev!(A::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},plan0::ChebHankelPlanH1,plan1::ChebHankelPlanH1, h0_tls::Vector{Vector{ComplexF64}},h1_tls::Vector{Vector{ComplexF64}},block_cache::CFIEBlockSystemCache{T};multithreaded::Bool=true) where {T<:Real}
    _one_k_nosymm_CFIE_chebyshev!(A,pts,plan0,plan1,h0_tls,h1_tls,block_cache;multithreaded=multithreaded)
    return nothing
end