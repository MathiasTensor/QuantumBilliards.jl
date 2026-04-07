
# =============================================================================
#  KRESS-CORRECTED CFIE ASSEMBLY: OVERVIEW AND IMPLEMENTATION NOTES
# =============================================================================
#
#  Reference:
#      R. Kress, "Boundary integral equations in time-harmonic acoustic scattering",
#      Math. Comput. Modelling 15 (1991), 229–243.

#  This implementation is intended for smooth closed 2D boundary components,
#  such as smooth polar curves. The full boundary may consist of one or several
#  disconnected smooth closed components, i.e. the geometry may contain holes.
#
#  Therefore:
#
#      - each connected boundary component is smooth,
#      - each connected boundary component is periodic,
#      - no corner singularities are present,
#      - Kress's smooth-periodic Nyström treatment applies naturally.
#
#  In particular, this code does NOT target the corner case discussed by Kress
#  via graded meshes and endpoint singularity smoothing. That machinery is not
#  needed here because the intended boundaries are smooth.
#
#  The main difficulty is the self-interaction on the same component:
#
#      - the single-layer kernel has a logarithmic singularity,
#      - the double-layer kernel is only weakly singular,
#      - interactions between different components are smooth.
#
#  Kress's idea is to split the self-interaction kernel into:
#
#      singular logarithmic part  +  smooth remainder,
#
#  and to treat the singular logarithmic part by a special periodic quadrature.
#  In the discrete system, that singular part is encoded by the matrix R.
#
# -----------------------------------------------------------------------------
#  HIGH-LEVEL PICTURE
# -----------------------------------------------------------------------------
#
#  For one fixed target point x_i and one source point y_j on the SAME smooth
#  closed component, the self-interaction entry is decomposed into:
#
#      known logarithmic singular structure
#      + smooth correction.
#
#  The logarithmic singular structure is universal on a periodic equispaced
#  mesh and depends on
#
#      -log(4 sin^2((t-s)/2)).
#
#  Kress's Nyström method uses this periodic logarithmic kernel explicitly, and
#  the smooth remainder is then added separately. 
#
# -----------------------------------------------------------------------------
#  R MATRIX
# -----------------------------------------------------------------------------
#
#  The matrix R is the discrete Kress quadrature matrix associated with the
#  logarithmic kernel
#
#      log(4 sin^2((t-s)/2)).
#
#  On an equispaced periodic grid, this kernel depends only on the periodic
#  index difference, so the corresponding discrete matrix is circulant:
#
#      R[i,j] = r(i-j mod N).
#
#  Therefore, R is completely determined by its first column.
#
#  In this implementation:
#
#      kress_R!(R0)
#
#  constructs that first column from the Fourier representation of the
#  logarithmic kernel and then fills the rest of the matrix by circular shifts.
#
# -----------------------------------------------------------------------------
#  MULTIPLE COMPONENTS / HOLES
# -----------------------------------------------------------------------------
#
#  If the boundary consists of several disconnected smooth closed components,
#  each component has its own self-interaction logarithmic singularity.
#
#  Therefore, each component receives its own Kress correction block:
#
#      R_1, R_2, ..., R_nc
#
#  and the global correction matrix is block diagonal:
#
#      R = diag(R_1, R_2, ..., R_nc).
#
#  Interactions between different components are smooth, so they do not use R.
#
#      boundary = Γ_1 ⊔ Γ_2 ⊔ ... ⊔ Γ_nc
#
#      R =
#          [ R_1   0    0   ...   0 ]
#          [  0   R_2   0   ...   0 ]
#          [  0    0   R_3  ...   0 ]
#          [ ...  ...  ...  ...  ...]
#          [  0    0    0   ...  R_nc ]
#
#
#  In the special case of a single smooth closed curve: R = R_1, i.e. the global correction matrix is just one circulant block.
#
# -----------------------------------------------------------------------------
#  ASSEMBLY OF THE CFIE MATRIX
# -----------------------------------------------------------------------------
#
#  For source and target on the same smooth component, the CFIE entry is built as
#
#      A[i,j] = -( DLP contribution + i k SLP contribution ).
#
#  Both DLP and SLP are split into:
#
#      singular logarithmic coefficient × R[i,j]
#      + smooth remainder.
#
#  In the code:
#
#      DLP:
#          l1 = coefficient multiplying the logarithmic singular part
#          l2 = smooth remainder
#
#          dval = R[i,j] * l1 + w_j * l2
#
#      SLP:
#          m1 = coefficient multiplying the logarithmic singular part
#          m2 = smooth remainder
#
#          sval = R[i,j] * m1 + w_j * m2
#
#  and then
#
#      A[i,j] = -( dval + i k * sval ).
#
#  The diagonal is treated separately, using the known limiting formulas for the
#  singular terms and the smooth remainder (see Kress's paper, Eq. 2.2 - Eq. 2.9).
#
#  If source and target lie on different smooth closed components, the kernel is
#  smooth. No split is needed there.
#
# -----------------------------------------------------------------------------
#  LIMITATIONS
# -----------------------------------------------------------------------------
#
#  This implementation is designed for:
#
#      - smooth closed periodic boundary components,
#      - possibly several disconnected components / holes,
#      - no corners.
#
#  It is not a corner-adapted Kress implementation with graded meshes.
#  If true corners were introduced, one would need Kress's corner machinery or
#  some other corner-aware discretization.  [oai_citation:2‡Kress_Nystrom.pdf](sediment://file_000000004384720aaeabe95a080dec45)
# =============================================================================

# Provides kress_R! to compute the circulant R matrix for the Kress method. kress_R! uses the FFT to compute the matrix efficiently, while kress_R! with ts computes it using a direct summation approach. Both functions modify the input matrix R0 in place.
# Ref: Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Mathematics Comput. Modelling Vol 15, pp. 229-243). Pergamon Press, 1991, GB.
# Alex Barnett's code via ifft to get the circulant vector kernel and construct the circulant with circshift.
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
    @. R0[:,1]=-two_pi*rjn-(2*two_pi/(N^2))*alt # R0[:,1] = -2π*rjn .- (4π/N^2)*alt, first col is ref
    for j in 2:N # fill out the rest circulantly:
        @views R0[:,j].=circshift(R0[:,j-1],1) # shift by +1 wrt previous column
    end
    return nothing
end

function kress_R_corner!(R0::AbstractMatrix{T}) where {T<:Real}
    Nint=size(R0,1)
    isodd(Nint) || error("kress_R_corner! expects odd size 2n-1.")
    n=(Nint+1)÷2
    Nfull=2*n
    Rfull=Matrix{T}(undef,Nfull,Nfull)
    kress_R!(Rfull)
    # full ts nodes are k=0,...,2n-1; corner uses interior nodes k=1,...,2n-1
    @views R0.=Rfull[2:end,2:end]
    return nothing
end

# Build the R matrix for the CFIE_kress method by assembling the circulant R matrices for each component of the boundary. The function takes a vector of BoundaryPointsCFIE objects, computes the appropriate offsets for each component, and fills in the R matrix using the kress_R! function for each component's corresponding block. It is block diagonal since only for boundary interaction within the same component we have the singularity that needs to be corrected by the R matrix, while for interactions between different components the kernel is smooth and does not require correction.
# R = [ R_1  0   0  ...  0
#       0   R_2 0   ...  0
#       0   0   R_3 ...  0
#       ... ... ... ...  ...
#       0   0   0   ...  R_nc ]
function build_Rmat_kress(solver::CFIE_kress,pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    Rmat=zeros(T,Ntot,Ntot)
    for a in 1:length(pts)
        ra=offs[a]:(offs[a+1]-1)
        kress_R!(@view Rmat[ra,ra])
    end
    return Rmat
end

#Build the matrix for the Kress corner case by evaluating the logarithmic kernel on the original ts grid (before grading) for each component of the boundary. This function is used when the Kress grading is applied to handle corners, and it computes the R matrix using the kress_log_corner! function for each component's corresponding block. The resulting R matrix is block diagonal, with each block containing the logarithmic kernel evaluated on the original equispaced grid for that component.
function build_Rmat_kress(solver::CFIE_kress_corners,pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    Rmat=zeros(T,Ntot,Ntot)
    for a in 1:length(pts)
        ra=offs[a]:(offs[a+1]-1)
        kress_R_corner!(@view Rmat[ra,ra])
    end
    return Rmat
end

# construct_matrices!
# Low-level function to construct the system matrix A for the CFIE_kress method. This function fills in the entries of A based on the boundary points, their weights, and the geometry of the problem, using the provided Kress R matrix for the singularity regularization. The resulting matrix A is modified in place and can then be used for solving the eigenvalue problem.
#
# Inputs:
# - solver: The CFIE_kress solver instance containing the boundary discretization and weights.
# - A: The system matrix to be filled in by this function.
# - pts: A vector of BoundaryPointsCFIE objects representing the discretized boundary points and their associated weights.
# - Rmat: The precomputed R matrix for the CFIE_kress method, which is used in the construction of the system matrix A.
# - k: The wavenumber for which to construct the system matrix.
# - multithreaded: A boolean flag indicating whether to use multithreading for matrix construction.
#
# Output:
# - The constructed system matrix A for the CFIE_kress method, modified in place.
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

function construct_matrices!(solver::CFIE_kress,A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
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
    return A
end

function construct_matrices!(solver::CFIE_kress_corners,A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    αL1=k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    Gs=[cfie_geom_cache(p,true) for p in pts]
    nc=length(pts)
    for a in 1:nc
        pa=pts[a]
        Ga=Gs[a]
        Na=length(pa.xy)
        ra=offs[a]:(offs[a+1]-1)
        nloc=(Na+1)÷2
        fac=T(nloc/pi)
        @inbounds for i in 1:Na
            gi=ra[i]
            si=Ga.speed[i]
            κi=Ga.kappa[i]
            wi=pa.ws[i]
            jac_i=fac*wi
            dval=Complex{T}(wi*κi,zero(T))
            m1=αM1*si
            m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*si^2)+2*inv_two_pi*log(jac_i))*si
            sval=Complex{T}(jac_i*Rmat[gi,gi]*m1,zero(T))+wi*m2
            A[gi,gi]=one(Complex{T})-(dval+ik*sval)
        end
        @use_threads multithreading=multithreaded for j in 2:Na
            gj=ra[j]
            sj=Ga.speed[j]
            wj=pa.ws[j]
            aj=fac*wj
            @inbounds for i in 1:(j-1)
                gi=ra[i]
                si=Ga.speed[i]
                wi=pa.ws[i]
                ai=fac*wi
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
                dval_ij=aj*Rmat[gi,gj]*l1_ij+wj*l2_ij
                m1_ij=αM1*j0*sj
                m2_ij=αM2*h0*sj-m1_ij*lt
                sval_ij=aj*Rmat[gi,gj]*m1_ij+wj*m2_ij
                A[gi,gj]=-(dval_ij+ik*sval_ij)
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                dval_ji=ai*Rmat[gj,gi]*l1_ji+wi*l2_ji
                m1_ji=αM1*j0*si
                m2_ji=αM2*h0*si-m1_ji*lt
                sval_ji=ai*Rmat[gj,gi]*m1_ji+wi*m2_ji
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

"""
    construct_matrices(solver::Union{CFIE_kress,CFIE_kress_corners},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}

High-level function to construct the system matrix A for the CFIE_kress methods. This function computes the R matrix using build_Rmat_kress and then calls construct_matrices! to fill in the entries of A based on the boundary points, their weights, and the geometry of the problem. The resulting matrix A is returned, which can then be used for solving the eigenvalue problem.
 
# Inputs:
- `solver`: The CFIE_kress / CFIE_kress_corners solver instance containing the boundary discretization and weights.
- `pts`: A vector of BoundaryPointsCFIE objects representing the discretized boundary points and their associated weights.
- `k`: The wavenumber for which to construct the system matrix.
- `multithreaded`: A boolean flag indicating whether to use multithreading for matrix construction.

# Output:
- The constructed system matrix A for the CFIE_kress / CFIE_kress_corners methods.
"""
function construct_matrices(solver::Union{CFIE_kress,CFIE_kress_corners},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    @blas_1 Rmat=build_Rmat_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    return A
end

"""
    solve(solver::CFIE_kress,A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}

High-level function to solve the CFIE eigenvalue problem for a single wavenumber. This function constructs the system matrix using the provided R matrix and then computes the smallest singular value (or determinant) corresponding to the eigenvalue of interest.

# Inputs:
- `solver`: The CFIE_kress solver instance containing the boundary discretization and weights.
- `A`: The system matrix to be filled in by the construct_matrices! function.
- `basis`: The basis used for the solution (not utilized in this function but included for consistency with other solve functions).
- `pts`: A vector of BoundaryPointsCFIE objects representing the discretized boundary points and their associated weights.
- `k`: The wavenumber for which to solve the eigenvalue problem.
- `Rmat`: The precomputed R matrix for the CFIE_kress method, which is used in the construction of the system matrix A.
- `multithreaded`: A boolean flag indicating whether to use multithreading for matrix construction.
- `use_krylov`: A boolean flag indicating whether to use a Krylov method for computing the smallest singular value (if true) or to compute the full SVD (if false).
- `which`: A symbol indicating whether to compute the determinant (:det) or the smallest singular value (:svd) for the eigenvalue. Note that the Krylov method does not support determinant calculation and will fall back to SVD if `:det` is selected. Also there is option :det_argmin which can be used for finding minima.

# Output:
- The smallest singular value (or determinant) corresponding to the eigenvalue of interest.
"""
function solve(solver::Union{CFIE_kress,CFIE_kress_corners},A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

"""
    solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners},A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

High-level function to solve the CFIE eigenvalue problem for a single wavenumber and return both the smallest singular value and the corresponding right singular vector (eigenfunction). This function constructs the system matrix using the provided R matrix and then computes the SVD to extract the smallest singular value and its associated singular vector.

# Inputs:
- `solver`: The CFIE_kress / CFIE_kress_corners solver instance containing the boundary discretization and weights.
- `basis`: The basis used for the solution (not utilized in this function but included for consistency with other solve functions).
- `pts`: A vector of BoundaryPointsCFIE objects representing the discretized boundary points and their associated weights.
- `k`: The wavenumber for which to solve the eigenvalue problem.
- `Rmat`: The precomputed R matrix for the CFIE_kress / CFIE_kress_corners methods, which is used in the construction of the system matrix A.
- `multithreaded`: A boolean flag indicating whether to use multithreading for matrix construction.

# Output:
 - A tuple containing the smallest singular value (eigenvalue) and the corresponding right singular vector for the given wavenumber.
"""
function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners},A::Matrix{Complex{T}},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

"""
    solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

High-level function to solve the CFIE eigenvalue problem for a single wavenumber and return both the smallest singular value and the corresponding right singular vector (eigenfunction).

# Inputs:
- `solver`: The CFIE_kress / CFIE_kress_corners solver instance containing the boundary discretization and weights.
- `basis`: The basis used for the solution (not utilized in this function but included for consistency with other solve functions).
- `pts`: A vector of BoundaryPointsCFIE objects representing the discretized boundary points and their associated weights.
- `k`: The wavenumber for which to solve the eigenvalue problem.

# Output:
- A tuple containing the smallest singular value (eigenvalue) and the corresponding right singular vector for the given wavenumber.
"""
function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    @blas_1 Rmat=build_Rmat_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

"""
    solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}

High-level function to solve the CFIE eigenvalue problem for a vector of wavenumbers. This function iterates over the provided wavenumbers, evaluates the boundary points and weights for each wavenumber, and then calls the solve_vect function to compute the smallest singular value and corresponding singular vector for each wavenumber. The results are collected in vectors and returned as a tuple.

# Inputs:
- `solver`: The CFIE_kress / CFIE_kress_corners solver instance containing the boundary discretization and weights.
- `basis`: The basis used for the solution (not utilized in this function but included for consistency with other solve functions).
- `pts`: A vector of BoundaryPointsCFIE objects representing the discretized boundary points and their associated weights.
- `ks`: A vector of wavenumbers for which to solve the eigenvalue problem.
- `multithreaded`: A boolean flag indicating whether to use multithreading for matrix construction.

# Output:
- A tuple containing two vectors: the first vector contains the smallest singular values (eigenvalues) for each wavenumber, and the second vector contains the corresponding singular vectors for each wavenumber.
"""
function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
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

"""
    solve_INFO(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}

High-level function to solve the CFIE eigenvalue problem while also providing detailed timing and condition number information. This function constructs the system matrix, computes its condition number, performs the SVD / det, and then reports the time taken for each step as well as the condition number of the matrix.
# Inputs:
- `solver`: The CFIE_kress / CFIE_kress_corners solver instance containing the boundary discretization and weights.
- `basis`: The basis used for the solution (not utilized in this function but included for consistency with other solve functions).
- `pts`: A vector of BoundaryPointsCFIE objects representing the discretized boundary points and their associated weights.
- `k`: The wavenumber for which the eigenvalue problem is being solved.
- `multithreaded`: A boolean flag indicating whether to use multithreading for matrix construction.
- `use_krylov`: A boolean flag indicating whether to use a Krylov method for computing the smallest singular value (if true) or to compute the full SVD (if false).
- `which`: A symbol indicating whether to compute the determinant (:det) or the smallest singular value (:svd) for the eigenvalue estimation. Also accepts :det_argmin to compute the determinant at the argument-minimizing wavenumber.

# Output:
# - The smallest singular value (or determinant) corresponding to the eigenvalue of interest, along with detailed timing information for each step of the computation.
"""
function solve_INFO(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    t0=time()
    @info "Constructing circulant R matrix..."
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    Rmat=build_Rmat_kress(solver,pts)
    t1=time()
    @info "Building boundary operator A..."
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    t2=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    t3=time()
    s=@svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
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
    return s
end

##########################
#### CFIE_kress UTILS ####
##########################

# plot_boundary_with_weight_INFO
# Utility function to visualize the boundary points along with their associated weights for the CFIE_kress method. This function creates a figure showing the boundary points colored by their weights, as well as arrows indicating the tangential direction at each point. Additionally, it plots the weight functions and their derivatives for each panel in separate subplots. This visualization can help in understanding the distribution of weights and the geometry of the boundary discretization.
#
# Inputs:
# - billiard: The billiard object representing the geometry of the problem.
# - solver: The CFIE_kress / CFIE_kress_corners solver instance containing the boundary discretization and weights.
# - k: The wavenumber used in the evaluation of the points and weights.
# - markersize: The size of the markers used to plot the boundary points.
#
# Output:
# - A figure object containing the visualizations of the boundary points, weights, and their derivatives
function plot_boundary_with_weight_INFO(billiard::Bi,solver::Union{CFIE_kress,CFIE_kress_corners};k=20.0,markersize=5) where {Bi<:AbsBilliard}
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

################
#### LEGACY ####
################

function solve(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    @blas_1 Rmat=build_Rmat_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{CFIE_kress,CFIE_kress_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end