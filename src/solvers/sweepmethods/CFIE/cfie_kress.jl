
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

# Build the R matrix for the CFIE_kress method by assembling the circulant R matrices for each component of the boundary. The function takes a vector of BoundaryPointsCFIE objects, computes the appropriate offsets for each component, and fills in the R matrix using the kress_R! function for each component's corresponding block. It is block diagonal since only for boundary interaction within the same component we have the singularity that needs to be corrected by the R matrix, while for interactions between different components the kernel is smooth and does not require correction.
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

###############################
#### DIRECT A CONSTRUCTION ####
###############################

function construct_matrices!(solver::CFIE_kress,A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    αL1=k*inv_two_pi
    αL2=k/2*im
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=k*im
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
    return A
end

function construct_matrices(solver::CFIE_kress,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    return A
end

function solve(solver::CFIE_kress,A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        mu=svdvals(A)
        return mu[end]
    end 
end

function solve(solver::CFIE_kress,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        mu=svdvals(A)
        return mu[end]
    end 
end

function solve(solver::CFIE_kress,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
   if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        mu=svdvals(A)
        return mu[end]
    end 
end

function solve_vect(solver::CFIE_kress,A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::CFIE_kress,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_eigenvectors_CFIE(solver::CFIE_kress,basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
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

function solve_INFO(solver::CFIE_kress,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    t0=time()
    @info "Constructing circulant R matrix..."
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_CFIE(pts)
    t1=time()
    @info "Building boundary operator A..."
    construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
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
#### CFIE_kress UTILS ####
####################

function plot_boundary_with_weight_INFO(billiard::Bi,solver::CFIE_kress;k=20.0,markersize=5) where {Bi<:AbsBilliard}
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