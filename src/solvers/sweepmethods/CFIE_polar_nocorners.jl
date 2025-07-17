##########################
#### BESSEL FUNCTIONS ####
##########################

H(n::Int,x::T) where {T<:Real}=Bessels.hankelh1(n,x)
J(n::Int,x::T) where {T<:Real}=Bessels.besselj(n,x)
J(ns::As,x::T) where {T<:Real,As<:AbstractRange{Int}}=Bessels.besselj(ns,x)

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

function CFIE_polar_nocorners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    billiard.full_boundary[1] isa PolarSegment ? nothing : error("CFIE_polar_nocorners only works with billiards with 1 PolarSegment full boundary")
    length(billiard.full_boundary)==1 ? nothing : error("CFIE_polar_nocorners only works with billiards with 1 PolarSegment full boundary")
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return CFIE_polar_nocorners{T,Bi}(sampler,bs,eps,min_pts,min_pts,billiard)
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
end

function evaluate_points(solver::CFIE_polar_nocorners,billiard::Bi,k) where {Bi<:AbsBilliard}
    boundary=billiard.full_boundary[1]
    L=boundary.length
    type=typeof(L)
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/(two_pi)))
    isodd(Ntot) ? Ntot+=1 : nothing # make sure Ntot is even, since we need to have an even number of points for the quadrature
    ts=[s(k,N) for k in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(boundary,ts_rescaled) 
    tangent_1st=tangent(boundary,ts_rescaled)
    tangent_2nd=tangent_2(boundary,ts_rescaled)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts)
end

##################################
#### KRESS CIRCULANT R MATRIX ####
##################################

# Provide 2 functions, kress_R_fft! and kress_R_sum!, to compute the circulant R matrix for the Kress method. kress_R_fft! uses the FFT to compute the matrix efficiently, while kress_R_sum! computes it using a direct summation approach. Both functions modify the input matrix R0 in place. The best performance is achieved with kress_R_fft! for large matrices, while kress_R_sum! is more straightforward and may be easier to understand for smaller matrices.
# Ref: Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.

# Alex Barnett's idea to use ifft to get the circulant vector kernel and construct the circulant with circshift.
function kress_R_fft!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    n=N÷2
    a=zeros(Complex{T},N) #  build the spectral vector a (first col)
    for m in 1:(n-1)
        a[m+1]=1/m     # positive freq
        a[N-m+1]=1/m     # negative freq
    end # leave a[n+1] == 0  (no 1/n term)
    rjn=real(ifft(a)) # inverse FFT → rjn[j] = (2/N)*∑_{m=1..n-1} (1/m) cos(2π m (j-1)/N)
    ks=0:(N-1) # build the first column, adding the “alternating” correction
    alt=(-1).^ks # alt[j+1] = (-1)^j
    @. R0[:,1]=-2*pi*rjn+(4*pi/(N^2))*alt # R0[:,1] = -2π*rjn .+ (4π/N^2)*alt, first col is ref
    for j in 2:N # fill out the rest circulantly:
        @views R0[:,j].=circshift(R0[:,j-1],1) # shift by +1 wrt previous column
    end
    return nothing
end

function kress_R_sum!(R0::AbstractMatrix{T}) where {T<:Real}
    N=size(R0,1)
    M=N/2-1
    ks=collect(0:N-1) # build the 1d series s[0:N-1] 
    s=zeros(T,N)
    @inbounds for m in 1:M
        s.+=(1/m).*cos.(2*pi*m.*ks./N)
    end
    alt=(-1).^ks # "alternating" term  cos((N/2)*(i-j)*2π/N) = (-1)^(i-j)
    # build the index‐difference matrix once: idx[i,j] = mod(i-j, N) in 0:(N-1)
    idx=@. mod((1:N)'.-(1:N),N)  # this is N×N of UInts
    @. R0=-4*pi/N*(s[idx.+1].-(1/N)*alt[idx.+1]) # fill R0 with one big broadcast
    return nothing
end

function kress_R_sum!(R0::AbstractMatrix{T},Δs::AbstractVector{T}) where {T<:Real}
    N=length(Δs)
    ds=Δs.*(T(0.5))  # work with half‐angles to save a division inside the loop
    @inbounds for i in 1:N
        R0[i,i]=zero(T)
        let di=ds[i]
            for j in i+1:N
                d=di-ds[j]
                s2=sin(d)
                v=-log(4*s2*s2)
                R0[i,j]=v
                R0[j,i]=v
            end
        end
    end
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
    L2[diagind(L2)].=κ # the diagona of SLP is 0 so no contribution with log(w'(s))L1(s,s), # the "curvature type" limit for DLP
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
    #M2[d].=((im/2-MathConstants.eulergamma/pi).-(1/(two_pi)).*log.((k^2)/4 .*speed.^2)).*speed .+2 .*log.(pts.ak).*M1[d] # Kress's modification to DLP limit with 2*log(w'(s))*M1(s,s)
    M2[d].=((im/2-MathConstants.eulergamma/pi).-(1/(two_pi)).*log.((k^2)/4 .*speed.^2)).*speed
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

##############
#### MAIN ####
##############

function solve(solver::CFIE_polar_nocorners,basis::Ba,pts::BoundaryPointsCFIE{T},k;use_combined::Bool=false) where {T<:Real,Ba<:AbstractHankelBasis}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R_fft!(Rmat) # fft work for trapezoidal parametrization, sum needs to be for weights (domains with corners)
    A=M(solver,pts,k,Rmat;use_combined=use_combined)
    mu=svdvals(A)
    return mu[end]
end
