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

struct CFIE{T,Bi}<:SweepSolver where {T<:Real,Bi<:AbsBilliard} 
    fundamental::Bool
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be rescaled
    pts_scaling_factor::Vector{T}
    ws::Vector{<:Function} # quadrature weights for each segment, must be same length as the length of "fundamental::Bool" boundary, if true same as fundamental boundary, otherwise full boundary
    ws_der::Vector{<:Function} # quadrature weights derivatives for each segment
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
end

function CFIE(pts_scaling_factor::Union{T,Vector{T}},ws::Vector{<:Function},ws_der::Vector{<:Function},billiard::Bi;min_pts=20,fundamental::Bool=true,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    n_curves=fundamental ? length(billiard.fundamental_boundary) : length(billiard.full_boundary)
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor for _ in 1:n_curves] : pts_scaling_factor
    sampler=[LinearNodes() for _ in 1:n_curves] # placeholder for sampler, since we will rescale the quadrature weights
    return CFIE{T,Bi}(fundamental,sampler,bs,ws,ws_der,eps,min_pts,min_pts,billiard)
end

function CFIE(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,q::Int=8,fundamental::Bool=true,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    n_curves=fundamental ? length(billiard.fundamental_boundary) : length(billiard.full_boundary)
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor for _ in 1:n_curves] : pts_scaling_factor # one needs to be careful there are enough bs for all segments
    sampler=[LinearNodes() for _ in 1:n_curves]
    ws::Vector{Function}=[v->w_reparametrized(v,q) for _ in 1:n_curves] # quadrature weights for each segment, must be same length as the length of "fundamental::Bool" boundary, if true same as fundamental boundary, otherwise full boundary
    ws_der::Vector{Function}=[v->dw_reparametrized(v,q) for _ in 1:n_curves] # quadrature weights derivatives for each segment
    return CFIE{T,Bi}(fundamental,sampler,bs,ws,ws_der,eps,min_pts,min_pts,billiard)
end

#############################
#### BOUNDARY EVALUATION ####
#############################

struct BoundaryPointsCFIE{T}<:AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}} # the xy coords of the new mesh points
    tangent::Vector{SVector{2,T}} # normals evaluated at the new mesh points
    curvature::Vector{T} # curvature evaluated at new mesh points
    sk::Vector{T} # new mesh points by w in solver
    sk_local::Vector{Vector{T}} # the local mesh points in [0,1] parametrizations for each segment
    ak::Vector{T} # the new weights (derivatives) of the new mesh points
end

function evaluate_points(solver::CFIE,billiard::Bi,k) where {Bi<:AbsBilliard}
    two_pi=2*pi
    fundamental=solver.fundamental
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    Ls=[crv.length for crv in boundary];L=sum(Ls);Ls_scaled=cumsum(Ls./L)
    type=typeof(Ls_scaled[1])
    bs=solver.pts_scaling_factor
    Ns=[max(solver.min_pts,round(Int,k*Ls[i]*bs[i]/(two_pi))) for i in eachindex(Ls)];Ntot=sum(Ns)
    iseven(Ntot) ? Ntot+=1 : nothing # make sure Ntot is even, since we need to have an even number of points for the quadrature
    ts_all=midpoints(range(0,two_pi,length=(Ntot)))
    cuts=cumsum([0;Ns])
    ts_per_panel=[ts_all[cuts[i]+1:cuts[i+1]] for i in eachindex(Ns)]
    ws=solver.ws # we need only for adjacent segments the unique qaudrature 
    ws_der=solver.ws_der # derivativs of them
    xy_all=Vector{SVector{2,type}}()
    tangent_all=Vector{SVector{2,type}}()
    kappa_all=Vector{type}()
    sk_all=Vector{type}()
    sk_local_all=Vector{Vector{type}}() # local mesh points in [0,1] parametrization for each segment
    ak_all=Vector{type}()
    for i in eachindex(ts_per_panel) 
        crv=boundary[i]
        t=ts_per_panel[i] 
        if i==1
            t_i=zero(type);t_f=two_pi*Ls_scaled[i] # the previous segments is the end segment and it's end parametrization is 0.0, so t_i=0.0
        else
            t_i=two_pi*Ls_scaled[i-1];t_f=two_pi*Ls_scaled[i] # the start and end of the segment in global parametrization
        end
        t_scaled=(t.-t_i)./(t_f-t_i) # need to rescale to ts_per_panel to local [0,1] parametrization since the ws and ws_der applied locally
        sk_local=ws[i](t_scaled) # we need to evaluate the sk first locally since the ws[i] is a local function (each segment has its own quadrature) and then project it to a global parameter; mapping [0,1] -> [0,1]
        xy=curve(crv,sk_local) # the xy coordinates of the new mesh points, these are global now
        tangent=tangent_vec(crv,sk_local) # the normals evaluated at the new mesh points, these are global now
        kappa=curvature(crv,sk_local) # the curvature evaluated at the new mesh points, these are global now
        ak=ws_der[i](sk_local) # the weights of the new mesh points in the local coordinates
        sk=t_i.+sk_local.*(t_f-t_i) # now we can project it to the global parameter (w : [0,1] -> [0,1])
        append!(xy_all,xy)
        append!(tangent_all,tangent)
        append!(kappa_all,kappa)
        append!(sk_all,sk)
        push!(sk_local_all,sk_local) # need to add as Vector, not splated with append!
        append!(ak_all,ak)
    end
    return BoundaryPointsCFIE(xy_all,tangent_all,kappa_all,sk_all,sk_local_all,ak_all)
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
    N = length(Δs)
    @assert size(R0) == (N, N)
    # work with half‐angles to save a division inside the loop
    ds = Δs .* (T(0.5))
    @inbounds for i in 1:N
        R0[i,i] = zero(T)
        let di = ds[i]
            for j in i+1:N
                d  = di - ds[j]
                s2 = sin(d)
                v  = -log(4 * s2 * s2)
                R0[i,j] = v
                R0[j,i] = v
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
    two_pi=2*pi
    N=length(pts.xy)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    ΔX=@. X-X'   # ΔX[i,j] = X[i] - X[j] = x(t_i) - x(t_j)
    ΔY=@. Y-Y'   # ΔY[i,j] = Y[i] - Y[j] = y(t_i) - y(t_j)
    R=hypot.(ΔX,ΔY)
    R[diagind(R)].=one(T) # avoid zeros on diagonal, does not influence result since overwritten few lines below
    dX=getindex.(pts.tangent,1) # tangent x
    dY=getindex.(pts.tangent,2) # tangent y
    κ=pts.curvature
    Δs=pts.sk .-pts.sk'
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
    L1=-k/(two_pi)*inner.*J1./R
    L=im*k/2*inner.*H1./R
    L2=L.-L1.*log.(4*sin.(Δs/2).^2)
    # fix diagonal entries by taking the known limits
    L1[diagind(L1)].=zero(Complex{T}) # lim t→s L1 = 0 for SLP
    L2[diagind(L2)].=κ./(two_pi) # the diagona of SLP is 0 so no contribution with log(w'(s))L1(s,s)
    return L1,L2
end

function L1_L2_M1_M2_matrix(pts::BoundaryPointsCFIE{T},k::T) where {T<:Real}
    two_pi=2*pi
    N=length(pts.xy)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    ΔX=@. X-X'   # ΔX[i,j] = X[i] - X[j] = x(t_i) - x(t_j)
    ΔY=@. Y-Y'   # ΔY[i,j] = Y[i] - Y[j] = y(t_i) - y(t_j)
    R=hypot.(ΔX,ΔY)
    R[diagind(R)].=one(T) # avoid zeros on diagonal, does not influence result since overwritten few lines below
    dX=getindex.(pts.tangent,1) # tangent x
    dY=getindex.(pts.tangent,2) # tangent y
    κ=pts.curvature
    Δs=pts.sk .-pts.sk'
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
    L1=-k/(2*pi)*inner.*J1./R
    L=im*k/2*inner.*H1./R
    L2=L.-L1.*log.(4*sin.(Δs/2).^2)
    # assemble M1 and M2 off the diagonal element wise 
    M1=-1/(two_pi).*J0.*speed_row
    M=im/2. *H0.*speed_row
    M2=M.-M1.*log.(4*sin.(Δs/2).^2)
    # fix diagonal entries by taking the known limits
    κlim=1/(two_pi)*κ # length-N
    d=diagind(L1)
    L1[d].=zero(Complex{T}) # lim t→s L1 = 0 for SLP
    L2[d].=κlim # the curvature limit for DLP
    M1[d].=-1/(two_pi).*speed
    M2[d].=((im/2-MathConstants.eulergamma/pi).-(1/(two_pi)).*log.((k^2)/4 .*speed.^2)).*speed .+2 .*log.(pts.ak).*M1[d] # Kress's modification to DLP limit with 2*log(w'(s))*M1(s,s)
    return L1,L2,M1,M2
end

#####################################
#### Vectorized Nystrom M Matrix ####
#####################################

function M(pts::BoundaryPointsCFIE{T},k::T,Rmat::Matrix{T};use_combined::Bool=false) where {T<:Real}
    N=length(pts.xy)
    if use_combined
      L1,L2,M1,M2=L1_L2_M1_M2_matrix(pts,k)
      A_double=(Rmat.*L1.+(2π/N).*L2).*pts.ak' # DLP case
      A_single=(Rmat.*M1.+(2π/N).*M2).*pts.ak' # SLP case
      A=(A_double.+(im*k)*A_single) # D+i*k*S
    else
      L1,L2=L1_L2_matrix(pts,k)
      A=(Rmat.*L1.+(2π/N).*L2).*pts.ak' # just DLP case
    end
    return Diagonal(ones(Complex{T},N))-A
end

##############
#### MAIN ####
##############

function solve(solver::CFIE{T},basis::Ba,pts::BoundaryPointsCFIE{T},k;use_combined::Bool=false) where {T<:Real,Ba<:AbstractHankelBasis}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    #kress_R_fft!(Rmat) # or kress_R_sum!(Rmat) for small N
    kress_R_sum!(Rmat,pts.sk)
    A=M(pts,k,Rmat;use_combined=use_combined)
    mu=svdvals(A)
    return mu[end]
end

####################
#### CFIE UTILS ####
####################

function plot_boundary_with_weight_INFO(billiard::Bi,solver::CFIE;k=20.0,markersize=5) where {Bi<:AbsBilliard}
    pts=evaluate_points(solver,billiard,k)
    xs=getindex.(pts.xy,1)
    ys=getindex.(pts.xy,2)
    ak=pts.ak
    m=div(length(solver.ws),2)
    f=Figure(size=(2500+550*2,600*m),resolution=(2500+550*2,600*m))
    ax=Axis(f[1,1][1,1],title="Boundary with weights",width=1000,height=1000,aspect=DataAspect())
    scatter!(ax,xs,ys;markersize=markersize,color=ak,colormap=:viridis,strokewidth=0) #  colour by ak so you see where points are denser
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