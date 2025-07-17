##########################
#### BESSEL FUNCTIONS ####
##########################

H(n::Int,x::T) where {T<:Real}=Bessels.hankelh1(n,x)
J(n::Int,x::T) where {T<:Real}=Bessels.besselj(n,x)
J(ns::As,x::T) where {T<:Real,As<:AbstractRange{Int}}=Bessels.besselj(ns,x)

##########################################################################################################
#### WEIGHT FUNCTIONS USED BY KRESS: Boundary Integral Equations in time-harmonic acoustic scattering ####
##########################################################################################################

# The parameter s ∈ [0,2π] in all non-reparametried functions. We rescale it since segment parametrizations go from [0,1]
v(s::T,q::Int) where {T<:Real}=(1/q-1/2)*((pi-s)/pi)^3+1/q*((s-pi)/pi)+0.5
dv(s::T,q::Int) where {T<:Real}=-(3*(1/q-1/2)/π)*((π-s)/π)^2+1/(q*π)
w_kress(s::T,q::Int) where {T<:Real}=2*pi*(v(s,q)^q)/(v(s,q)^q+v(2*pi-s,q)^q)
w_reparametrized(s::T,q::Int) where {T<:Real}=w_kress(2*pi*s,q)/(2*pi)
function dw_reparametrized(t::T,q::Int) where {T<:Real}
    s=2π*t
    As=v(s,q)^q
    Cs=v(2*pi-s,q)^q
    Bs=As+Cs
    dAs=q*v(s,q)^(q-1)*dv(s,q)
    dCs=-q*v(2π-s,q)^(q-1)*dv(2π-s,q)
    dwk=2π*(dAs*Bs-As*(dAs+dCs))/Bs^2
    return dwk
end
v(s::AbstractVector{T},q::Int) where {T<:Real}=v.(s,q)
dv(s::AbstractVector{T},q::Int) where {T<:Real}=dv.(s,q)
w_kress(s::AbstractVector{T},q::Int) where {T<:Real}=w_kress.(s,q)
w_reparametrized(s::AbstractVector{T},q::Int) where {T<:Real}=w_reparametrized.(s,q)
dw_reparametrized(s::AbstractVector{T},q::Int) where {T<:Real}=dw_reparametrized.(s,q)

###########################
#### CONSTRUCTOR CFIE ####
###########################

struct CFIE_polar_nocorners{T,Bi}<:SweepSolver where {T<:Real,Bi<:AbsBilliard} 
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be rescaled
    pts_scaling_factor::Vector{T}
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
end

struct CFIE_polar_corner_correction{T,Bi,F1,F2}<:SweepSolver where {T<:Real,Bi<:AbsBilliard,F1<:Function,F2<:Function} 
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be rescaled
    pts_scaling_factor::Vector{T}
    w::F1
    w_der::F2
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
end

function CFIE_polar_nocorners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    billiard.full_boundary[1] isa PolarSegment ? nothing : error("CFIE_polar_nocorners only works with billiards with 1 PolarSegment full boundary")
    length(billiard.full_boundary)==1 ? nothing : error("CFIE_polar_nocorners only works with billiards with 1 PolarSegment full boundary")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return CFIE_polar_nocorners{T,Bi}(sampler,bs,eps,min_pts,min_pts,billiard)
end

function CFIE_polar_corner_correction(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;q=8,min_pts=20,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    billiard.full_boundary[1] isa PolarSegment ? nothing : error("CFIE_polar_corner_correction only works with billiards with 1 PolarSegment full boundary")
    length(billiard.full_boundary)==1 ? nothing : error("CFIE_polar_corner_correction only works with billiards with 1 PolarSegment full boundary")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    w::Function=v->w_reparametrized(v,q) # quadrature weights 
    w_der::Function=v->dw_reparametrized(v,q) # quadrature weights derivatives 
    sampler=[LinearNodes()]
    return CFIE_polar_corner_correction{T,Bi,typeof(w),typeof(w_der)}(sampler,bs,w,w_der,eps,min_pts,min_pts,billiard)
end

function CFIE_polar_corner_correction(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi,w::F1,w_der::F2;q=8,min_pts=20,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard,F1<:Function,F2<:Function}
    billiard.full_boundary[1] isa PolarSegment ? nothing : error("CFIE_polar_corner_correction only works with billiards with 1 PolarSegment full boundary")
    length(billiard.full_boundary)==1 ? nothing : error("CFIE_polar_corner_correction only works with billiards with 1 PolarSegment full boundary")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return CFIE_polar_corner_correction{T,Bi,F1,F2}(sampler,bs,w,w_der,eps,min_pts,min_pts,billiard)
end

#############################
#### BOUNDARY EVALUATION ####
#############################
two_pi=2*pi

#### use N even for the algorithm - equidistant parameters ####
s(k::Int,N::Int)=two_pi*k/N

struct BoundaryPointsCFIE{T}<:AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}} # the xy coords of the new mesh points
    tangent::Vector{SVector{2,T}} # tangents evaluated at the new mesh points
    tangent_2::Vector{SVector{2,T}} # derivatives of tangents evaluated at new mesh points
    ts::Vector{T} # parametrization that needs to go from [0,2π]
    ws::Vector{T} # the weights for the quadrature at ts
    s::Vector{T} # arc lengths at ts
    ds::Vector{T} # diffs between crv lengths at ts
end

function evaluate_points(solver::CFIE_polar_nocorners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    boundary=billiard.full_boundary[1]
    L=boundary.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/(two_pi)))
    isodd(N) ? N+=1 : nothing # make sure Ntot is even, since we need to have an even number of points for the quadrature
    ts=[s(k,N) for k in 1:N]
    ts_rescaled=ts./two_pi # b/c our curves and tangents are defined on [0,1]
    xy=curve(boundary,ts_rescaled) 
    tangent_1st=tangent(boundary,ts_rescaled)./(two_pi) # ! Rescaled tangents due to chain rule ∂γ/∂θ = ∂γ/∂u * ∂u/∂θ = ∂γ/∂u * 1/(2π)
    tangent_2nd=tangent_2(boundary,ts_rescaled)./(two_pi)^2 # ! Rescaled tangents due to chain rule ∂²γ/∂θ² = ∂²γ/∂u² * (∂u/∂θ)² + ∂γ/∂u * ∂²u/∂θ² = ∂²γ/∂u² * 1/(2π)^2 + ∂γ/∂u * 0 = ∂²γ/∂u² * 1/(2π)^2
    ss=arc_length(boundary,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=[one(T) for _ in 1:N] # weights for the trapezoidal rule, all ones since we use the trapezoidal rule
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ss,ds)
end

function evaluate_points(solver::CFIE_polar_corner_correction{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    crv=billiard.full_boundary[1]
    L=crv.length
    bs=solver.pts_scaling_factor[1]
    N=max(solver.min_pts,round(Int,k*L*bs/two_pi))
    isodd(N) ? N+=1 : nothing
    ts=[s(i,N) for i in 1:N ]
    u0=ts./two_pi
    u=solver.w.(u0) # new local param
    du_du0=solver.w_der.(u0) # derivative w.r.t. u0
    xy_local=curve(crv,u)
    T_loc=tangent(crv,u)
    T2_loc=tangent_2(crv,u)
    J1 = one(T)/two_pi # chain rule: ∂/∂θ = (du/du0)*(du0/dθ) ∂/∂u = du_du0*(1/2π)
    # second derivative requires product + second derivative of w; for simplicity we drop w″ term,
    # which is consistent with Kress’ corner‐correction that only adjusts log term:
    T_global=[SVector(du_dtheta[i]*T_loc[i][1],du_dtheta[i]*T_loc[i][2]) for i in eachindex(T_loc)]
    T2_global=[SVector((du_dtheta[i]^2)*T2_loc[i][1],(du_dtheta[i]^2)*T2_loc[i][2]) for i in eachindex(T2_loc)]
    ss=arc_length(crv,u)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=@. du_du0 * J1 # quadrature weights for kress are du_du0*(1/2π)
    ts_final=two_pi.*u
    return BoundaryPointsCFIE(xy_local,T_global,T2_global,ts_final,ws,ss,ds)
end

function BoundaryPointsCFIE_to_BoundaryPoints(bdPoints::BoundaryPointsCFIE{T}) where {T<:Real}
    normal=[SVector(-getindex(t,2),getindex(t,1)) for t in bdPoints.tangent]
    return BoundaryPoints(bdPoints.xy,normal,bdPoints.s,bdPoints.ds)
end

##################################
#### KRESS CIRCULANT R MATRIX ####
##################################

# Provide 2 functions, kress_R_fft! and kress_R_sum!, to compute the circulant R matrix for the Kress method. kress_R_fft! uses the FFT to compute the matrix efficiently, while kress_R_sum! computes it using a direct summation approach. Both functions modify the input matrix R0 in place. The best performance is achieved with kress_R_fft! for large matrices, while kress_R_sum! is more straightforward and may be easier to understand for smaller matrices.
# Ref: Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
# Alex Barnett's idea to use ifft to get the circulant vector kernel and construct the circulant with circshift, .
function kress_R_fft!(R0::AbstractMatrix{T}) where {T<:Real}
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

function kress_R_sum_LEGACY!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    M=N/2-1
    ks=collect(0:N-1) # build the 1d series s[0:N-1] 
    s=zeros(T,N)
    @inbounds for m in 1:M
        s.+=(1/m).*cos.(two_pi*m.*ks./N)
    end
    alt=(-1).^ks # "alternating" term  cos((N/2)*(i-j)*2π/N) = (-1)^(i-j)
    # build the index‐difference matrix once: idx[i,j] = mod(i-j, N) in 0:(N-1)
    idx=@. mod((1:N)'.-(1:N),N)  # this is N×N of UInts
    @. R0=-2*two_pi/N*(s[idx.+1].-(1/N)*alt[idx.+1]) # fill R0 with one big broadcast
    return nothing
end

function kress_R_sum!(R0::AbstractMatrix{T}, ts::Vector{T}) where {T<:Real}
    ds=ts.*T(0.5) # ds[i] = s_i/2
    D=ds.-ds' # D[i,j] = (s_i/2) - (s_j/2) = (s_i - s_j)/2
    R0.=-log.(4 .*sin.(D).^2)
    R0[diagind(R0)].=zero(T)
    return nothing
  end

################################################################
#### FIRST AND SECOND LAYER BOUNDARY POTENTIAL CONSTRUCTION ####
################################################################

#### ONLY NEED TO USE DOUBLE LAYER POTENTIAL - NO NEUMANN RESONANCES CHECK ####
function L1_L2_matrix(pts::BoundaryPointsCFIE{T},k::T) where {T<:Real}
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
    @inbounds for i in 1:N # In this case R is symmetric
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

function L1_L2_M1_M2_matrix(pts::BoundaryPointsCFIE{T},k::T) where {T<:Real}
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
    @inbounds for i in 1:N # In this case R is symmetric
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
    M2[d].=((im/2-MathConstants.eulergamma/pi).-(1/(two_pi)).*log.((k^2)/4 .*speed.^2)).*speed .+2 .*log.(pts.ws).*M1[d] # Kress's modification to DLP limit with 2*log(w'(s))*M1(s,s)
    #M2[d].=((im/2-MathConstants.eulergamma/pi).-(1/(two_pi)).*log.((k^2)/4 .*speed.^2)).*speed
    return L1,L2,M1,M2
end

#####################################
#### Vectorized Nystrom M Matrix ####
#####################################

function M(solver::CFIE_polar_nocorners,pts::BoundaryPointsCFIE{T},k::T,Rmat::Matrix{T};use_combined::Bool=false) where {T<:Real}
    N=length(pts.xy)
    if use_combined
        L1,L2,M1,M2=L1_L2_M1_M2_matrix(pts,k)
        A_double=Rmat.*L1.+(two_pi/N).*L2 # D
        A_single=Rmat.*M1.+(two_pi/N).*M2 # S
        A=A_double.+(im*k)*A_single # D+i*k*S
    else
        L1,L2=L1_L2_matrix(pts,k)
        A=@. Rmat.*L1.+(two_pi/N).*L2 # pure double layer
    end
    return Diagonal(ones(Complex{T},N))-A
end

function M(solver::CFIE_polar_corner_correction,pts::BoundaryPointsCFIE{T},k::T,Rmat::Matrix{T};use_combined::Bool=false) where {T<:Real}
    N=length(pts.xy)
    ws=pts.ws
    if use_combined
        L1,L2,M1,M2=L1_L2_M1_M2_matrix(pts,k)
        A_double=Rmat.*L1.+(two_pi/N).*L2 # D
        A_single=Rmat.*M1.+(two_pi/N).*M2 # S
        A=(A_double.+(im*k)*A_single).*ws' # D+i*k*S
    else
        L1,L2=L1_L2_matrix(pts,k)
        A=@. Rmat.*L1.+(two_pi/N).*L2 # pure double layer
        A=A.*ws'
    end
    return Diagonal(ones(Complex{T},N))-A
end

##############
#### MAIN ####
##############

function solve(solver::CFIE_polar_nocorners,basis::Ba,pts::BoundaryPointsCFIE{T},k;use_combined::Bool=false) where {T<:Real,Ba<:AbsBasis}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R_fft!(Rmat) # fft work for trapezoidal parametrization, sum needs to be for weights (domains with corners)
    A=M(solver,pts,k,Rmat;use_combined=use_combined)
    mu=svdvals(A)
    return mu[end]
end

function solve(solver::CFIE_polar_corner_correction,basis::Ba,pts::BoundaryPointsCFIE{T},k;use_combined::Bool=false) where {T<:Real,Ba<:AbsBasis}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R_sum!(Rmat,pts.ts)
    A=M(solver,pts,k,Rmat;use_combined=use_combined)
    mu=svdvals(A)
    return mu[end]
end

function solve_vect(solver::CFIE_polar_nocorners,basis::Ba,pts::BoundaryPointsCFIE{T},k;use_combined::Bool=false) where {T<:Real,Ba<:AbsBasis}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R_fft!(Rmat)
    A=M(solver,pts,k,Rmat;use_combined=use_combined)
    _,S,Vt=LAPACK.gesvd!('A','A',A) # do NOT use svd with DivideAndConquer() here b/c singular matrix!!!
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=Vt[idx,:]
    u_mu=real.(u_mu)
    return mu,u_mu
end

function solve_eigenvectors_CFIE(solver::CFIE_polar_nocorners,basis::Ba,ks::Vector{T};use_combined::Bool=false) where {T<:Real,Ba<:AbsBasis}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPointsCFIE{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,solver.billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];use_combined=use_combined)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

####################
#### CFIE UTILS ####
####################

function plot_boundary_with_weight_INFO(billiard::Bi,solver::Union{CFIE_polar_nocorners,CFIE_polar_corner_correction},;k=20.0,markersize=5) where {Bi<:AbsBilliard}
    pts=evaluate_points(solver,billiard,k)
    xs=getindex.(pts.xy,1)
    ys=getindex.(pts.xy,2)
    ws=pts.ws # weights for the quadrature rule
    m=max(1,div(length(solver.ws),2))
    f=Figure(size=(2500+550*2,1200*m),resolution=(2500+550*2,1200*m))
    ax=Axis(f[1,1][1,1],title="Boundary with weights",width=1000,height=1000,aspect=DataAspect())
    scatter!(ax,xs,ys;markersize=markersize,color=ws,colormap=:viridis,strokewidth=0) #  colour by ak so you see where points are denser
    nxs=getindex.(pts.tangent,2)
    nys=-getindex.(pts.tangent,1)
    arrows!(ax,xs,ys,nxs,nys,color=:black,lengthscale=0.1)
    ws_ders=solver.ws_der
    r,c=1,1
    for (i,wder) in enumerate(solver.ws)
        if c>2
            r+=1;c=1
        end
        tloc=collect(range(0.0,1.0,length=200))
        wline=wder(tloc)
        wderline=ws_ders[i](tloc)
        ax=Axis(f[1,2][r,c][1,1],width=500,height=500)
        lines!(ax,tloc,wline;label="panel $i",linewidth=2)
        axislegend(ax;position=:lt)
        ax=Axis(f[1,2][r,c][1,2],width=500,height=500)
        lines!(ax,tloc,wderline;label="panel $i derivative",linewidth=2)
        axislegend(ax;position=:lt)
        c+=1
    end
    return f
end
