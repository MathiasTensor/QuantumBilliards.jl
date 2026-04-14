
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
# =============================================================================

# helpers for construct_matrices! to determine the bool kwarg for cfie_geom_cache to decide whether to include the log(w') correction
@inline _is_kress_graded(::CFIE_kress)=false
@inline _is_kress_graded(::CFIE_kress_corners)=true
@inline _is_kress_graded(::CFIE_kress_global_corners)=true

struct CFIEPanelArrays{T<:Real}
    X::Vector{T}
    Y::Vector{T}
    dX::Vector{T}
    dY::Vector{T}
    s::Vector{T}
end

@inline function _panel_arrays_cache(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    s=@. sqrt(dX^2+dY^2)
    return CFIEPanelArrays(X,Y,dX,dY,s)
end

struct CFIEKressWorkspace{T<:Real,M<:AbstractMatrix{T}}
    offs::Vector{Int}
    Rmat::M
    Gs::Vector{CFIEGeomCache{T}}
    parr::Vector{CFIEPanelArrays{T}}
    Ntot::Int
end

function build_cfie_kress_workspace(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    Rmat=build_Rmat_kress(solver,pts)
    Gs=_is_kress_graded(solver) ? [cfie_geom_cache(p,true) for p in pts] : [cfie_geom_cache(p,false) for p in pts]
    parr=[_panel_arrays_cache(p) for p in pts]
    Ntot=offs[end]-1
    return CFIEKressWorkspace(offs,Rmat,Gs,parr,Ntot)
end

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
function build_Rmat_kress(solver::Union{CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
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
function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},Gs::Vector{CFIEGeomCache{T}},parr::Vector{CFIEPanelArrays{T}},offs::Vector{Int},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    nc=length(pts)
    for a in 1:nc
        pa=pts[a]
        Ga=Gs[a]
        Pa=parr[a]
        Na=length(Pa.X)
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
        @use_threads multithreading=(multithreaded && Na>=32) for j in 2:Na
            gj=ra[j]
            sj=Ga.speed[j]
            wj=pa.ws[j]
            @inbounds for i in 1:(j-1)
                gi=ra[i]
                si=Ga.speed[i]
                wi=pa.ws[i]
                r=Ga.R[i,j]
                invr=Ga.invR[i,j]
                lt=Ga.logterm[i,j]
                inn_ij=Ga.inner[i,j]
                inn_ji=Ga.inner[j,i]
                h0,h1=hankel_pair01(k*r)
                j0=real(h0)
                j1=real(h1)
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                dval_ij=Rmat[gi,gj]*l1_ij+wj*l2_ij
                m1_ij=αM1*j0*sj
                m2_ij=αM2*h0*sj-m1_ij*lt
                sval_ij=Rmat[gi,gj]*m1_ij+wj*m2_ij
                A[gi,gj]=-(dval_ij+ik*sval_ij)
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
        Pa=parr[a]
        Pb=parr[b]
        Na=length(Pa.X)
        Nb=length(Pb.X)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=Pa.X;Ya=Pa.Y
        Xb=Pb.X;Yb=Pb.Y
        dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s
        @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            @inbounds for j in 1:Nb
                gj=rb[j]
                dx=xi-Xb[j]
                dy=yi-Yb[j]
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                invr=inv(r)
                inn=dYb[j]*dx-dXb[j]*dy
                h0,h1=hankel_pair01(k*r)
                sj=sb[j]
                wj=pb.ws[j]
                dval=wj*(αL2*inn*h1*invr)
                sval=wj*(αM2*h0*sj)
                A[gi,gj]=-(dval+ik*sval)
            end
        end
    end
    return A
end

function construct_matrices_with_derivatives!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},Gs::Vector{CFIEGeomCache{T}},parr::Vector{CFIEPanelArrays{T}},offs::Vector{Int},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    fill!(A1,zero(Complex{T}))
    fill!(A2,zero(Complex{T}))
    nc=length(pts)
    for a in 1:nc
        pa=pts[a]
        Ga=Gs[a]
        Pa=parr[a]
        Na=length(Pa.X)
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
            m2_1=-(si/(π*k))
            m2_2=(si/(π*k^2))
            sval1=wi*m2_1
            sval2=wi*m2_2
            A[gi,gi]=one(Complex{T})-(dval+ik*sval)
            A1[gi,gi]=-(Complex{T}(0,1)*sval+ik*sval1)
            A2[gi,gi]=-(Complex{T}(0,2)*sval1+ik*sval2)
        end
        @use_threads multithreading=(multithreaded && Na>=32) for j in 2:Na
            gj=ra[j]
            sj=Ga.speed[j]
            wj=pa.ws[j]
            @inbounds for i in 1:(j-1)
                gi=ra[i]
                si=Ga.speed[i]
                wi=pa.ws[i]
                r=Ga.R[i,j]
                invr=Ga.invR[i,j]
                lt=Ga.logterm[i,j]
                inn_ij=Ga.inner[i,j]
                inn_ji=Ga.inner[j,i]
                kr=k*r
                h0,h1=hankel_pair01(kr)
                j0=real(h0)
                j1=real(h1)
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                dval_ij=Rmat[gi,gj]*l1_ij+wj*l2_ij
                l1_ij_1=-(inn_ij*k*j0)*inv_two_pi
                l1_ij_2=(inn_ij*(k*r*j1-j0))*inv_two_pi
                l2_ij_1=(inn_ij*k*(lt*j0+im*π*h0))*inv_two_pi
                l2_ij_2=(inn_ij*(lt*(j0-k*r*j1)+im*π*(h0-k*r*h1)))*inv_two_pi
                dval_ij_1=Rmat[gi,gj]*l1_ij_1+wj*l2_ij_1
                dval_ij_2=Rmat[gi,gj]*l1_ij_2+wj*l2_ij_2
                m1_ij=αM1*j0*sj
                m2_ij=αM2*h0*sj-m1_ij*lt
                sval_ij=Rmat[gi,gj]*m1_ij+wj*m2_ij
                m1_ij_1=(r*sj*j1)*inv_two_pi
                m1_ij_2=(r*sj*(k*r*j0-j1))*inv_two_pi/k
                m2_ij_1=-(r*sj*(lt*j1+im*π*h1))*inv_two_pi
                m2_ij_2=(r*sj*(lt*(j1-k*r*j0)-im*π*k*r*h0+im*π*h1))*inv_two_pi/k
                sval_ij_1=Rmat[gi,gj]*m1_ij_1+wj*m2_ij_1
                sval_ij_2=Rmat[gi,gj]*m1_ij_2+wj*m2_ij_2
                A[gi,gj]=-(dval_ij+ik*sval_ij)
                A1[gi,gj]=-(dval_ij_1+Complex{T}(0,1)*sval_ij+ik*sval_ij_1)
                A2[gi,gj]=-(dval_ij_2+Complex{T}(0,2)*sval_ij_1+ik*sval_ij_2)
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                dval_ji=Rmat[gj,gi]*l1_ji+wi*l2_ji
                l1_ji_1=-(inn_ji*k*j0)*inv_two_pi
                l1_ji_2=(inn_ji*(k*r*j1-j0))*inv_two_pi
                l2_ji_1=(inn_ji*k*(lt*j0+im*π*h0))*inv_two_pi
                l2_ji_2=(inn_ji*(lt*(j0-k*r*j1)+im*π*(h0-k*r*h1)))*inv_two_pi
                dval_ji_1=Rmat[gj,gi]*l1_ji_1+wi*l2_ji_1
                dval_ji_2=Rmat[gj,gi]*l1_ji_2+wi*l2_ji_2
                m1_ji=αM1*j0*si
                m2_ji=αM2*h0*si-m1_ji*lt
                sval_ji=Rmat[gj,gi]*m1_ji+wi*m2_ji
                m1_ji_1=(r*si*j1)*inv_two_pi
                m1_ji_2=(r*si*(k*r*j0-j1))*inv_two_pi/k
                m2_ji_1=-(r*si*(lt*j1+im*π*h1))*inv_two_pi
                m2_ji_2=(r*si*(lt*(j1-k*r*j0)-im*π*k*r*h0+im*π*h1))*inv_two_pi/k
                sval_ji_1=Rmat[gj,gi]*m1_ji_1+wi*m2_ji_1
                sval_ji_2=Rmat[gj,gi]*m1_ji_2+wi*m2_ji_2
                A[gj,gi]=-(dval_ji+ik*sval_ji)
                A1[gj,gi]=-(dval_ji_1+Complex{T}(0,1)*sval_ji+ik*sval_ji_1)
                A2[gj,gi]=-(dval_ji_2+Complex{T}(0,2)*sval_ji_1+ik*sval_ji_2)
            end
        end
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        pa=pts[a]
        pb=pts[b]
        Pa=parr[a]
        Pb=parr[b]
        Na=length(Pa.X)
        Nb=length(Pb.X)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=Pa.X;Ya=Pa.Y
        Xb=Pb.X;Yb=Pb.Y
        dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s
        @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            @inbounds for j in 1:Nb
                gj=rb[j]
                dx=xi-Xb[j]
                dy=yi-Yb[j]
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                invr=inv(r)
                inn=dYb[j]*dx-dXb[j]*dy
                h0,h1=hankel_pair01(k*r)
                sj=sb[j]
                wj=pb.ws[j]
                dval=wj*(Complex{T}(0,k/2)*inn*h1*invr)
                dval1=wj*(-(Complex{T}(0,1)/2)*inn*k*h0)
                dval2=wj*(-(Complex{T}(0,1)/2)*inn*(h0-k*r*h1))
                sval=wj*(Complex{T}(0,1/2)*h0*sj)
                sval1=wj*(-(Complex{T}(0,1)/2)*r*h1*sj)
                sval2=wj*((Complex{T}(0,1)/2)*r*(h1-k*r*h0)*sj/k)
                A[gi,gj]=-(dval+ik*sval)
                A1[gi,gj]=-(dval1+Complex{T}(0,1)*sval+ik*sval1)
                A2[gi,gj]=-(dval2+Complex{T}(0,2)*sval1+ik*sval2)
            end
        end
    end
    return A,A1,A2
end

function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    @blas_1 construct_matrices!(solver,A,pts,ws.Rmat,ws.Gs,ws.parr,ws.offs,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    Gs=_is_kress_graded(solver) ? [cfie_geom_cache(p,true) for p in pts] : [cfie_geom_cache(p,false) for p in pts]
    parr=[_panel_arrays_cache(p) for p in pts]
    return construct_matrices!(solver,A,pts,Rmat,Gs,parr,offs,k;multithreaded=multithreaded)
end

function construct_matrices_with_derivatives!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    @blas_1 construct_matrices_with_derivatives!(solver,A,A1,A2,pts,ws.Rmat,ws.Gs,ws.parr,ws.offs,k;multithreaded=multithreaded)
end

function construct_matrices_with_derivatives!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    Gs=_is_kress_graded(solver) ? [cfie_geom_cache(p,true) for p in pts] : [cfie_geom_cache(p,false) for p in pts]
    parr=[_panel_arrays_cache(p) for p in pts]
    return construct_matrices_with_derivatives!(solver,A,A1,A2,pts,Rmat,Gs,parr,offs,k;multithreaded=multithreaded)
end

function solve(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    @blas_1 Rmat=build_Rmat_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
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

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
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

function solve_INFO(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    t0=time()
    @info "Building boundary operator A from cached Kress workspace..."
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA; sigdigits=4))"
    t2=time()
    s=@svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s
end