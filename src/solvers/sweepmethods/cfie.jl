# Useful reading:
#  - https://github.com/ahbarnett/mpspack - by Alex Barnett & Timo Betcke (MATLAB)
#  - Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
#  - Barnett, A. H., & Betcke, T. (2007). Stability and convergence of the method of fundamental solutions for Helmholtz problems on analytic domains. Journal of Computational Physics, 227(14), 7003-7026.
#  - Zhao, L., & Barnett, A. (2015). Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant. SIAM Journal on Numerical Analysis, Stable URL: https://www.jstor.org/stable/24512689

##########################
#### BESSEL FUNCTIONS ####
##########################

H(n::Int,x::T) where {T<:Real}=Bessels.hankelh1(n,x)
J(n::Int,x::T) where {T<:Real}=Bessels.besselj(n,x)
J(ns::As,x::T) where {T<:Real,As<:AbstractRange{Int}}=Bessels.besselj(ns,x)
two_pi=2*pi
inv_two_pi=1/two_pi
euler_over_pi=MathConstants.eulergamma/pi

###########################
#### CONSTRUCTOR CFIE ####
###########################

struct CFIE_polar_nocorners{T,Bi}<:SweepSolver where {T<:Real,Bi<:AbsBilliard} 
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be rescaled
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Union{Vector{Any},Nothing}
end

function CFIE_polar_nocorners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry=nothing) where {T<:Real,Bi<:AbsBilliard}
    any([!((boundary isa PolarSegment) || (boundary isa CircleSegment)) for boundary in billiard.full_boundary]) && error("CFIE_polar_nocorners only works with polar curves")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return CFIE_polar_nocorners{T,Bi}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

#############################
#### BOUNDARY EVALUATION ####
#############################

# helper function to compute the offsets for each component of the boundary, which are needed to correctly assemble the R matrix for the CFIE method. The offsets indicate the starting index of each component's points in the concatenated list of all boundary points. For example, if we have 3 components with 10, 15, and 20 points respectively, the offsets would be [1, 11, 26, 46].
function component_offsets(comps::Vector)
    nc=length(comps)
    offs=Vector{Int}(undef,nc+1)
    offs[1]=1
    for a in 1:nc
        offs[a+1]=offs[a]+length(comps[a].xy)
    end
    return offs
end

#### use N even for the algorithm - equidistant parameters ####
s(k::Int,N::Int)=two_pi*k/N

struct BoundaryPointsCFIE{T}<:AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}} # the xy coords of the new mesh points
    tangent::Vector{SVector{2,T}} # tangents evaluated at the new mesh points
    tangent_2::Vector{SVector{2,T}} # derivatives of tangents evaluated at new mesh points
    ts::Vector{T} # parametrization that needs to go from [0,2π]
    ws::Vector{T} # the weights for the quadrature at ts
    ws_der::Vector{T} # the derivatives of the weights for the quadrature at ts
    s::Vector{T} # arc lengths at ts
    ds::Vector{T} # diffs between crv lengths at ts
    compid::Int # index of the multi-domain, where the outer boundary is 1, the first inner boundary is 2,... It should be respected since otherwise the tangents/normals will be incorrectly oriented
end

# reverse all BoundaryPointsCFIE except 1st as they correspond to holes in the outer domain.
function _reverse_component_orientation(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    N=length(pts.xy)
    xy=reverse(pts.xy)
    tangent=reverse(-pts.tangent)
    tangent_2=reverse(pts.tangent_2)
    ts=copy(pts.ts) # these can stay the same since they are just the parameters of the curve, and reversing the order of points does not change the parameter values at those points
    ws=copy(pts.ws)
    ws_der=copy(pts.ws_der)
    ds=reverse(pts.ds)
    s=similar(pts.s)
    s[1]=zero(T)
    for i in 2:N
        s[i]=s[i-1]+ds[i-1]
    end
    return BoundaryPointsCFIE(xy,tangent,tangent_2,ts,ws,ws_der,s,ds,pts.compid)
end

function _evaluate_points(solver::CFIE_polar_nocorners{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/(two_pi)))
    isodd(N) ? N+=1 : nothing # make sure Ntot is even, since we need to have an even number of points for the quadrature
    ts=[s(k,N) for k in 1:N]
    ts_rescaled=ts./two_pi # b/c our curves and tangents are defined on [0,1]
    xy=curve(crv,ts_rescaled) 
    tangent_1st=tangent(crv,ts_rescaled)./(two_pi) # ! Rescaled tangents due to chain rule ∂γ/∂θ = ∂γ/∂u * ∂u/∂θ = ∂γ/∂u * 1/(2π)
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi)^2 # ! Rescaled tangents due to chain rule ∂²γ/∂θ² = ∂²γ/∂u² * (∂u/∂θ)² + ∂γ/∂u * ∂²u/∂θ² = ∂²γ/∂u² * 1/(2π)^2 + ∂γ/∂u * 0 = ∂²γ/∂u² * 1/(2π)^2
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N) # we kep these for future with different quadratures ala Kress (1)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ss,ds,idx)
end

function evaluate_points(solver::CFIE_polar_nocorners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(billiard.full_boundary))
    for (idx,crv) in enumerate(billiard.full_boundary)
        pts[idx]= idx==1 ? _evaluate_points(solver,crv,k,idx) : _reverse_component_orientation(_evaluate_points(solver,crv,k,idx))
    end
    return pts
end

# For CFIE with holes, we compute this by looking at the component offsets, which tell us where each component's points start and end in the concatenated array. The last offset gives us the total count of points.
function boundary_matrix_size(pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    return offs[end]-1
end

##################################
#### KRESS CIRCULANT R MATRIX ####
##################################

# Provides kress_R! to compute the circulant R matrix for the Kress method. kress_R! uses the FFT to compute the matrix efficiently, while kress_R! with ts computes it using a direct summation approach. Both functions modify the input matrix R0 in place.
# Ref: Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
# Alex Barnett's idea to use ifft to get the circulant vector kernel and construct the circulant with circshift, .
function kress_R!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    n=N÷2 # integer division
    a=zeros(Complex{T},N) #  build the spectral vector a (first col)
    for m in 1:(n-1)
        a[m+1]=1/m     # positive freq
        a[N-m+1]=1/m     # negative freq
    end # leave a[n+1] == 0  (no 1/n term)
    rjn=real(ifft(a)) # inverse FFT → rjn[j] = (2/N)*∑_{m=1..n-1} (1/m) cos(2π m (j-1)/N)
    ks=0:(N-1) # build the first column, adding the “alternating” correction
    alt=(-1).^ks # alt[j+1] = (-1)^j
    @. R0[:,1]=-two_pi*rjn+(2*two_pi/(N^2))*alt # R0[:,1] = -2π*rjn .+ (4π/N^2)*alt, first col is ref
    for j in 2:N # fill out the rest circulantly:
        @views R0[:,j].=circshift(R0[:,j-1],1) # shift by +1 wrt previous column
    end
    return nothing
end

# legacy from incomplete corner correction implementation where we could not use the FFT due to non-uniform weights. This is a direct O(N^2) summation approach to compute the R matrix, which is less efficient than the FFT-based method but serves as a reference.
function kress_R!(R0::AbstractMatrix{T},ts::Vector{T}) where {T<:Real}
    ds=ts.*T(0.5) # ds[i] = s_i/2
    D=ds.-ds' # D[i,j] = (s_i/2) - (s_j/2) = (s_i - s_j)/2
    R0.=-log.(4 .*sin.(D).^2)
    R0[diagind(R0)].=zero(T)
    return nothing
end

# Build the R matrix for the CFIE method by assembling the circulant R matrices for each component of the boundary. The function takes a vector of BoundaryPointsCFIE objects, computes the appropriate offsets for each component, and fills in the R matrix using the kress_R! function for each component's corresponding block. It is block diagonal since only for boundary interaction within the same component we have the singularity that needs to be corrected by the R matrix, while for interactions between different components the kernel is smooth and does not require correction.
# R = [ R_1  0   0  ...  0
#       0   R_2 0   ...  0
#       0   0   R_3 ...  0
#       ... ... ... ...  ...
#       0   0   0   ...  R_nc ]
function build_Rmat_CFIE(pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    Rmat=zeros(T,Ntot,Ntot)
    for a in 1:length(pts)
        Na=length(pts[a].xy)
        ra=offs[a]:(offs[a+1]-1)
        kress_R!(@view Rmat[ra,ra])
    end
    return Rmat
end

###########################################
#### GEOMETRY CACHE / CHEBYSHEV HELPERS ####
###########################################

struct CFIEGeomCache{T<:Real}
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    logterm::Matrix{T}
    speed::Vector{T}
    kappa::Vector{T}
end

function cfie_geom_cache(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    ts=pts.ts
    N=length(pts.xy)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    ddX=getindex.(pts.tangent_2,1)
    ddY=getindex.(pts.tangent_2,2)
    ΔX=@. X-X'
    ΔY=@. Y-Y'
    R=hypot.(ΔX,ΔY)
    R[diagind(R)].=one(T)
    invR=inv.(R)
    invR[diagind(invR)].=zero(T)
    dX_row=reshape(dX,1,N)
    dY_row=reshape(dY,1,N)
    inner=@. dY_row*ΔX-dX_row*ΔY
    ΔT=ts.-ts'
    logterm=log.(4 .*sin.(ΔT./2).^2)
    logterm[diagind(logterm)].=zero(T)
    speed=@. sqrt(dX^2+dY^2)
    κnum=dX.*ddY.-dY.*ddX
    κden=dX.^2 .+ dY.^2
    kappa=inv_two_pi.*(κnum./κden)
    return CFIEGeomCache(R,invR,inner,logterm,speed,kappa)
end

##############################################
#### PRECOMPUTED SYMMETRY IMAGE GEOMETRY #####
##############################################

struct SymmetryImageCache{T<:Real}
    xy::Vector{SVector{2,T}}
    tangent::Vector{SVector{2,T}}
    speed::Vector{T}
    scale::ComplexF64
end

function build_reflection_image_caches(pts::BoundaryPointsCFIE{T},sym::Reflection) where {T<:Real}
    N=length(pts.xy)
    shift_x=hasproperty(sym,:shift_x) ? T(getproperty(sym,:shift_x)) : zero(T)
    shift_y=hasproperty(sym,:shift_y) ? T(getproperty(sym,:shift_y)) : zero(T)
    ops=_reflect_ops_and_scales(T,sym)
    caches=Vector{SymmetryImageCache{T}}(undef,length(ops))
    for (q,(op,scale_r)) in enumerate(ops)
        xy=Vector{SVector{2,T}}(undef,N)
        tangent=Vector{SVector{2,T}}(undef,N)
        speed=Vector{T}(undef,N)
        pt=zeros(T,2)
        @inbounds for j in 1:N
            xj,yj=pts.xy[j]
            txj,tyj=pts.tangent[j]
            if op==1
                x_reflect_point!(pt,xj,yj,shift_x)
                xy[j]=SVector(pt[1],pt[2])
                tangent[j]=-SVector(-txj,tyj)
            elseif op==2
                y_reflect_point!(pt,xj,yj,shift_y)
                xy[j]=SVector(pt[1],pt[2])
                tangent[j]=-SVector(txj,-tyj)
            else
                xy_reflect_point!(pt,xj,yj,shift_x,shift_y)
                xy[j]=SVector(pt[1],pt[2])
                tangent[j]=-SVector(-txj,-tyj)
            end
            speed[j]=sqrt(tangent[j][1]^2+tangent[j][2]^2)
        end
        caches[q]=SymmetryImageCache{T}(xy,tangent,speed,ComplexF64(scale_r,0.0))
    end
    return caches
end

function build_rotation_image_caches(pts::BoundaryPointsCFIE{T},sym::Rotation) where {T<:Real}
    N=length(pts.xy)
    cx,cy=sym.center
    ctab,stab,χ=_rotation_tables(T,sym.n,mod(sym.m,sym.n))
    caches=Vector{SymmetryImageCache{T}}(undef,sym.n-1)
    for l in 2:sym.n
        xy=Vector{SVector{2,T}}(undef,N)
        tangent=Vector{SVector{2,T}}(undef,N)
        speed=Vector{T}(undef,N)
        pt=zeros(T,2)
        @inbounds for j in 1:N
            xj,yj=pts.xy[j]
            txj,tyj=pts.tangent[j]
            rot_point!(pt,xj,yj,cx,cy,ctab[l],stab[l])
            xy[j]=SVector(pt[1],pt[2])
            tangent[j]=SVector(ctab[l]*txj-stab[l]*tyj,stab[l]*txj+ctab[l]*tyj)
            speed[j]=sqrt(tangent[j][1]^2+tangent[j][2]^2)
        end
        caches[l-1]=SymmetryImageCache{T}(xy,tangent,speed,ComplexF64(χ[l]))
    end
    return caches
end

function build_symmetry_image_caches(pts::BoundaryPointsCFIE{T},symmetry::Union{Nothing,Any,AbstractVector}) where {T<:Real}
    symmetry===nothing && return SymmetryImageCache{T}[]
    syms=symmetry isa AbstractVector ? symmetry : Any[symmetry]
    caches=SymmetryImageCache{T}[]
    for sym in syms
        if sym isa Reflection
            append!(caches,build_reflection_image_caches(pts,sym))
        elseif sym isa Rotation
            append!(caches,build_rotation_image_caches(pts,sym))
        else
            error("Unsupported symmetry type $(typeof(sym))")
        end
    end
    return caches
end

###############################
#### DIRECT A CONSTRUCTION ####
###############################

function _add_symmetry_contributions!(A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},k::T,symmetry;multithreaded::Bool=true) where {T<:Real}
    symmetry===nothing && return A
    αL2=Complex{T}(0,k/2)
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    nc=length(pts)
    for b in 1:nc
        pb=pts[b]
        caches=build_symmetry_image_caches(pb,symmetry)
        isempty(caches) && continue
        Nb=length(pb.xy)
        rb=offs[b]:(offs[b+1]-1)
        for cache in caches
            @use_threads multithreading=multithreaded for j in 1:Nb
                gj=rb[j]
                xj,yj=cache.xy[j]
                txj,tyj=cache.tangent[j]
                sj=cache.speed[j]
                χ=cache.scale
                wj=pb.ws[j]
                @inbounds for a in 1:nc
                    pa=pts[a]
                    Na=length(pa.xy)
                    ra=offs[a]:(offs[a+1]-1)
                    for i in 1:Na
                        gi=ra[i]
                        xi,yi=pa.xy[i]
                        dx=xi-xj
                        dy=yi-yj
                        r2=muladd(dx,dx,dy*dy)
                        r2<=(eps(T))^2 && continue
                        r=sqrt(r2)
                        invr=inv(r)
                        inn=tyj*dx-txj*dy
                        h1=H(1,k*r)
                        h0=H(0,k*r)
                        dterm=αL2*inn*h1*invr
                        sterm=αM2*h0*sj
                        A[gi,gj]+= -(χ*wj)*(dterm+ik*sterm)
                    end
                end
            end
        end
    end
    return A
end

function construct_matrices(solver::CFIE_polar_nocorners,A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    αL1=k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    Gs=[cfie_geom_cache(p) for p in pts]
    nc=length(pts)
    for a in 1:nc
        pa=pts[a]
        Ga=Gs[a]
        Na=length(pa.xy)
        ra=offs[a]:(offs[a+1]-1)
        @inbounds for i in 1:Na
            gi=ra[i]
            si=Ga.speed[i]
            κi=Ga.kappa[i]
            wi=pa.ws[i]
            dval=Complex{T}(wi*κi,zero(T))
            m1=αM1*si
            m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*si^2))*si
            sval=Complex{T}(Rmat[gi,gi]*m1,zero(T))+wi*m2
            A[gi,gi]=one(Complex{T})-(dval+ik*sval)
        end
        @use_threads multithreading=multithreaded for j in 2:Na
            gj=ra[j]
            sj=Ga.speed[j]
            wj=pa.ws[j]
            @inbounds for i in 1:(j-1)
                gi=ra[i]
                si=Ga.speed[i]
                rij=Ga.R[i,j]
                invr=Ga.invR[i,j]
                lt=Ga.logterm[i,j]
                h1=H(1,k*rij)
                h0=H(0,k*rij)
                j1=real(h1)
                j0=real(h0)
                inn_ij=Ga.inner[i,j]
                inn_ji=Ga.inner[j,i]
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                dval_ij=Rmat[gi,gj]*l1_ij+wj*l2_ij
                m1_ij=αM1*j0*sj
                m2_ij=αM2*h0*sj-m1_ij*lt
                sval_ij=Rmat[gi,gj]*m1_ij+wj*m2_ij
                A[gi,gj]=-(dval_ij+ik*sval_ij)
                wi=pa.ws[i]
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                dval_ji=Rmat[gj,gi]*l1_ji+wi*l2_ji
                m1_ji=αM1*j0*si
                m2_ji=αM2*h0*si-m1_ji*lt
                sval_ji=Rmat[gj,gi]*m1_ji+wi*m2_ji
                A[gj,gi]=-(dval_ji+ik*sval_ji)
            end
        end
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        pa=pts[a]
        pb=pts[b]
        Na=length(pa.xy)
        Nb=length(pb.xy)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=getindex.(pa.xy,1)
        Ya=getindex.(pa.xy,2)
        Xb=getindex.(pb.xy,1)
        Yb=getindex.(pb.xy,2)
        dXb=getindex.(pb.tangent,1)
        dYb=getindex.(pb.tangent,2)
        sb=@. sqrt(dXb^2+dYb^2)
        @use_threads multithreading=multithreaded for j in 1:Nb
            gj=rb[j]
            xj=Xb[j]
            yj=Yb[j]
            txj=dXb[j]
            tyj=dYb[j]
            sj=sb[j]
            wj=pb.ws[j]
            @inbounds for i in 1:Na
                gi=ra[i]
                dx=Xa[i]-xj
                dy=Ya[i]-yj
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                invr=inv(r)
                inn=tyj*dx-txj*dy
                h1=H(1,k*r)
                h0=H(0,k*r)
                dval=wj*(αL2*inn*h1*invr)
                sval=wj*(αM2*h0*sj)
                A[gi,gj]=-(dval+ik*sval)
            end
        end
    end
    _add_symmetry_contributions!(A,pts,offs,k,solver.symmetry;multithreaded=multithreaded)
    return A
end

function construct_matrices(solver::CFIE_polar_nocorners,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    return construct_matrices(solver,A,pts,Rmat,k;multithreaded=multithreaded)
end

function solve(solver::CFIE_polar_nocorners,A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        mu=svdvals(A)
        return mu[end]
    end 
end

function solve(solver::CFIE_polar_nocorners,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    A=construct_matrices(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        mu=svdvals(A)
        return mu[end]
    end 
end

function solve(solver::CFIE_polar_nocorners,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    A=construct_matrices(solver,A,pts,Rmat,k;multithreaded=multithreaded)
   if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        mu=svdvals(A)
        return mu[end]
    end 
end

function solve_vect(solver::CFIE_polar_nocorners,A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=Vt[idx,:]
    u_mu=real.(u_mu)
    return mu,u_mu
end

function solve_vect(solver::CFIE_polar_nocorners,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    A=construct_matrices(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=Vt[idx,:]
    u_mu=real.(u_mu)
    return mu,u_mu
end

function solve_eigenvectors_CFIE(solver::CFIE_polar_nocorners,basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{Vector{BoundaryPointsCFIE{eltype(ks)}}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,solver.billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];multithreaded=multithreaded)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

function solve_INFO(solver::CFIE_polar_nocorners,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    t0=time()
    @info "Constructing circulant R matrix..."
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    t1=time()
    @info "Building boundary operator A..."
    A=construct_matrices(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    t2=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    @info "Performing SVD..."
    t3=time()
    if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS s,_,_,_=svdsolve(A,1,:SR)
        reverse!(s)
    else
        s=svdvals(A)
    end
    t4=time()
    build_R=t1-t0
    build_A=t2-t1
    svd_time=t4-t3
    total=build_R+build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("R-matrix build: ",100*build_R/total," %")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s[end]
end

####################
#### CFIE UTILS ####
####################

function plot_boundary_with_weight_INFO(billiard::Bi,solver::CFIE_polar_nocorners;k=20.0,markersize=5) where {Bi<:AbsBilliard}
    f=Figure(resolution=(1200,1200))
    ax=Axis(f[1,1],title="boundary + point‐wise weights",aspect=DataAspect())
    pts_all=evaluate_points(solver,billiard,k)
    N=length(pts_all)
    for i in 1:N
        pts=pts_all[i]
        xs=getindex.(pts.xy,1)
        ys=getindex.(pts.xy,2)
        ws_pts=pts.ws
        scatter!(ax,xs,ys;markersize=markersize,color=ws_pts,colormap=:viridis,strokewidth=0)
        nxs=getindex.(pts.tangent,2)
        nys=-getindex.(pts.tangent,1)
        arrows!(ax,xs,ys,nxs,nys,color=:black,lengthscale=0.1)
        ws_funs=[v->fill(one(eltype(v)),length(v))]
        ws_der_funs=[v->fill(zero(eltype(v)),length(v))]
        panels=length(ws_funs)
        for j in 1:panels
            row=2+div(j-1,2)
            col=1+((j-1) % 2)
            tloc=collect(range(0,1,length=200))
            wline=ws_funs[j](tloc)
            wderline=ws_der_funs[j](tloc)
            a1=Axis(f[row,2*col-1],title="panel $j w(u)",xlabel="u",ylabel="w")
            lines!(a1,tloc,wline,linewidth=2)
            a2=Axis(f[row,2*col],title="panel $j w′(u)",xlabel="u",ylabel="w′")
            lines!(a2,tloc,wderline,linewidth=2)
        end
    end
    return f
end

# LEGACY CODE
#=
function L1_L2_matrix(pts::BoundaryPointsCFIE{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ts=pts.ts
    N=length(pts.xy)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    ΔX=@. X-X'   # ΔX[i,j] = X[i] - X[j] = x(t_i) - x(t_j)
    ΔY=@. Y-Y'   # ΔY[i,j] = Y[i] - Y[j] = y(t_i) - y(t_j)
    R=hypot.(ΔX,ΔY)
    R[diagind(R)].=one(T) # avoid zeros on diagonal, does not influence result since overwritten few lines below
    dX=getindex.(pts.tangent,1) # dx/dt
    dY=getindex.(pts.tangent,2) # dy/dt
    ddX=getindex.(pts.tangent_2,1) # d2x/dt2
    ddY=getindex.(pts.tangent_2,2) # d2y/dt2
    κnum=dX.*ddY.-dY.*ddX  # length-N
    κden=dX.^2 .+dY.^2 # length-N
    κ=(1/(two_pi))*(κnum./κden) # length-N
    ΔT=ts .-ts' # pts.ts ∈ [0,2π]
    dX_mat=reshape(dX,1,N)
    dY_mat=reshape(dY,1,N)
    inner=@. dY_mat*ΔX-dX_mat*ΔY
    H1=zeros(Complex{T},size(R)) 
    @use_threads multithreading=multithreaded for i in 1:N # In this case R is symmetric
        H1[i,i]=one(T) # can be whatever
        for j in i+1:N
            H1ij=H(1,k*R[i,j])
            H1[i,j]=H1ij
            H1[j,i]=H1ij
        end
    end
    J1=real.(H1)
    # assemble L and L1 off the diagonal
    L1=k/(two_pi)*inner.*J1./R
    L=im*k/2*inner.*H1./R
    L2=L.-L1.*log.(4*sin.(ΔT/2).^2)
    # fix diagonal entries by taking the known limits
    L1[diagind(L1)].=zero(Complex{T}) # lim t→s L1 = 0 for SLP
    L2[diagind(L2)].=κ # the diagonal of SLP is 0 so no contribution with log(w'(s))L1(s,s), # the "curvature type" limit for DLP
    return L1,L2
end

function L1_L2_M1_M2_matrix(pts::BoundaryPointsCFIE{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ts=pts.ts
    N=length(pts.xy)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    ΔX=@. X-X'   # ΔX[i,j] = X[i] - X[j] = x(t_i) - x(t_j)
    ΔY=@. Y-Y'   # ΔY[i,j] = Y[i] - Y[j] = y(t_i) - y(t_j)
    R=hypot.(ΔX,ΔY)
    R[diagind(R)].=one(T) # avoid zeros on diagonal, does not influence result since overwritten few lines below
    dX=getindex.(pts.tangent,1) # tangent x
    dY=getindex.(pts.tangent,2) # tangent y
    ddX=getindex.(pts.tangent_2,1) # d2x/dt2
    ddY=getindex.(pts.tangent_2,2) # d2y/dt2
    κnum=dX.*ddY.-dY.*ddX  # length-N
    κden=dX.^2 .+dY.^2 # length-N
    κ=(1/(two_pi))*(κnum./κden) # length-N
    ΔT=ts .-ts'
    dX_mat=reshape(dX,1,N)
    dY_mat=reshape(dY,1,N)
    inner=@. dY_mat*ΔX-dX_mat*ΔY
    H1=zeros(Complex{T},size(R))
    H0=zeros(Complex{T},size(R))  
    @use_threads multithreading=multithreaded for i in 1:N # In this case R is symmetric
        H1[i,i]=1.0 # can be whatever since diagonal limits correction later
        H0[i,i]=1.0 # can be whatever since diagonal limits correction later
        for j in i+1:N
            H1ij=H(1,k*R[i,j])
            H0ij=H(0,k*R[i,j])
            H1[i,j]=H1ij
            H1[j,i]=H1ij
            H0[i,j]=H0ij
            H0[j,i]=H0ij
        end
    end
    # bessels as Re: Hj=Jj-im*Yj
    J1=real.(H1)
    J0=real.(H0)
    speed=@. sqrt(dX^2+dY^2) 
    speed_row=reshape(speed,1,N) # 1×N this should be [x'(τ)^2 + y'(τ) for τ is ts]
    # assemble L and L1 off the diagonal element wise 
    #L1=-k/(2*pi)*inner.*J1./R
    L1=k/(2*pi)*inner.*J1./R
    L=im*k/2*inner.*H1./R
    L2=L.-L1.*log.(4*sin.(ΔT/2).^2)
    # assemble M1 and M2 off the diagonal element wise 
    M1=-1/(two_pi).*J0.*speed_row
    M=im/2. *H0.*speed_row
    M2=M.-M1.*log.(4*sin.(ΔT/2).^2)
    # fix diagonal entries by taking the known limits
    d=diagind(L1)
    L1[d].=zero(Complex{T}) # lim t→s L1 = 0 for SLP
    L2[d].=κ # the "curvature type" limit for DLP
    M1[d].=-1/(two_pi).*speed
    #M2[d].=((im/2-MathConstants.eulergamma/pi).-(1/(two_pi)).*log.((k^2)/4 .*speed.^2)).*speed .+2 .*log.(pts.ws_der).*M1[d] # Kress's modification to DLP limit with 2*log(w'(s))*M1(s,s). Commented out since we are using uniform weights and w'(s)=1, so log(w'(s))=0, so this term does not contribute.
    M2[d].=((im/2-MathConstants.eulergamma/pi).-(1/(two_pi)).*log.((k^2)/4 .*speed.^2)).*speed
    return L1,L2,M1,M2
end
=#
