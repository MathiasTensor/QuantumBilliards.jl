# Useful reading:
# - Kress, R., Boundary integral equations in time-harmonic acoustic scattering. Math. Comput. Modelling 15 (1991), 229-243.
# - Barnett, A. H. / Betcke, T., mpspack DLP implementation.
# - Zhao, L. / Barnett, A., Robust and efficient solution of the drum problem via Nyström approximation of the Fredholm determinant.

##########################
#### BESSEL FUNCTIONS ####
##########################

H(n::Int,x::T) where {T<:Real}=Bessels.hankelh1(n,x)
H(n::Int,x::Complex{T}) where {T<:Real}=SpecialFunctions.besselh(n,1,x)
const two_pi=2*pi
const inv_two_pi=1/two_pi
@inline function hankel_pair01(x);h0=H(0,x);h1=H(1,x);return h0,h1;end

struct DLPKressWorkspace{T<:Real,M<:AbstractMatrix{T}}
    Rmat::M
    G::CFIEGeomCache{T}
    parr::CFIEPanelArrays{T}
    N::Int
end

struct DLP_kress{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
end

struct DLP_kress_global_corners{T<:Real,Bi<:AbsBilliard,Sym}<:SweepSolver
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Sym
    kressq::Int
end

########################
#### CONSTRUCTORS ######
########################

function DLP_kress(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return DLP_kress{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry)
end

function DLP_kress_global_corners(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,eps=T(1e-15),symmetry::Union{Nothing,AbsSymmetry}=nothing,kressq=4) where {T<:Real,Bi<:AbsBilliard}
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    Sym=typeof(symmetry)
    return DLP_kress_global_corners{T,Bi,Sym}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,kressq)
end

#########################
#### SHARED HELPERS #####
#########################

@inline _is_dlp_kress_graded(::DLP_kress)=false
@inline _is_dlp_kress_graded(::DLP_kress_global_corners)=true

function build_dlp_kress_workspace(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T}) where {T<:Real}
    Rmat=build_Rmat_dlp_kress(solver,pts)
    G=_is_dlp_kress_graded(solver) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    parr=_panel_arrays_cache(pts)
    N=length(pts.xy)
    return DLPKressWorkspace(Rmat,G,parr,N)
end

function build_Rmat_dlp_kress(solver::DLP_kress,pts::BoundaryPointsCFIE{T}) where {T<:Real}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R!(Rmat)
    return Rmat
end

function build_Rmat_dlp_kress(solver::DLP_kress_global_corners,pts::BoundaryPointsCFIE{T}) where {T<:Real}
    N=length(pts.xy)
    Rmat=zeros(T,N,N)
    kress_R_corner!(Rmat)
    return Rmat
end

########################
#### PERIODIC KRESS ####
########################

function _evaluate_points(solver::DLP_kress{T},crv::C,k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    L=crv.length
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*L*bs[1]/two_pi))
    needed=2
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            needed=lcm(needed,sym.n)
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    ts=[s(j,N) for j in 1:N]
    ts_rescaled=ts./two_pi
    xy=curve(crv,ts_rescaled)
    tangent_1st=tangent(crv,ts_rescaled)./(two_pi)
    tangent_2nd=tangent_2(crv,ts_rescaled)./(two_pi)^2
    ss=arc_length(crv,ts_rescaled)
    ds=diff(ss)
    append!(ds,L+ss[1]-ss[end])
    ws=fill(T(two_pi/N),N)
    ws_der=ones(T,N)
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

function _evaluate_points(solver::DLP_kress_global_corners{T},comp::Vector{C},k::T,idx::Int) where {T<:Real,C<:AbsCurve}
    _,_,Ltot=component_lengths(comp)
    bs=solver.pts_scaling_factor
    N=max(solver.min_pts,round(Int,k*Ltot*bs[1]/two_pi))
    needed=1
    if !isnothing(solver.symmetry)
        sym=solver.symmetry
        if sym isa Rotation
            iseven(sym.n) && error("Incompatible. If sym.n is even, please use reflections.")
            needed=lcm(needed,sym.n)
        end
    end
    remN=mod(N,needed)
    remN!=0 && (N+=needed-remN)
    iseven(N) && (N+=needed)
    corners=_component_corner_locations(T,comp)
    σ,tmap,jac,jac2,_=multi_kress_graded_nodes_data(T,N,corners;q=solver.kressq)
    xy=Vector{SVector{2,T}}(undef,N)
    tangent_1st=Vector{SVector{2,T}}(undef,N)
    tangent_2nd=Vector{SVector{2,T}}(undef,N)
    @inbounds for i in 1:N
        q,γt,γtt=_eval_composite_geom_global_t(T,comp,tmap[i])
        xy[i]=q
        tangent_1st[i]=γt*jac[i]
        tangent_2nd[i]=γtt*(jac[i]^2)+γt*jac2[i]
    end
    h=pi/T((N+1)÷2)
    ds=Vector{T}(undef,N)
    @inbounds for i in 1:N
        tx=tangent_1st[i][1]
        ty=tangent_1st[i][2]
        ds[i]=hypot(tx,ty)*h
    end
    ts=σ
    ws=fill(h,N)
    ws_der=jac
    return BoundaryPointsCFIE(xy,tangent_1st,tangent_2nd,ts,ws,ws_der,ds,idx,true,SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)),SVector(zero(T),zero(T)))
end

####################
#### HIGH LEVEL ####
####################

function evaluate_points(solver::DLP_kress{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    comps=_boundary_components(billiard.full_boundary)
    length(comps)==1 || error("DLP_kress supports exactly one outer boundary component.")
    comp=comps[1]
    length(comp)==1 || error("DLP_kress requires a single smooth closed curve.")
    return _evaluate_points(solver,comp[1],k,1)
end

function evaluate_points(solver::DLP_kress_global_corners{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    comps=_boundary_components(billiard.full_boundary)
    length(comps)==1 || error("DLP_kress_global_corners supports exactly one outer boundary component.")
    comp=comps[1]
    isempty(comp) && error("Boundary component cannot be empty.")
    if length(comp)==1
        base=DLP_kress(solver.pts_scaling_factor,solver.billiard;min_pts=solver.min_pts,eps=solver.eps,symmetry=solver.symmetry)
        return _evaluate_points(base,comp[1],k,1)
    else
        return _evaluate_points(solver,comp,k,1)
    end
end

################################
#### LOW-LEVEL DLP ASSEMBLY ####
################################

function construct_dlp_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    N=length(pts.xy)
    @inbounds for i in 1:N
        D[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            _,h1=hankel_pair01(k*r)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            D[i,j]=Rmat[i,j]*l1_ij+pts.ws[j]*l2_ij
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            D[j,i]=Rmat[j,i]*l1_ji+pts.ws[i]*l2_ji
        end
    end
    return D
end

function construct_dlp_split!(solver::Union{DLP_kress,DLP_kress_global_corners},Dlog::AbstractMatrix{Complex{T}},Dsmooth::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(Dlog,zero(Complex{T}))
    fill!(Dsmooth,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        Dsmooth[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            _,h1=hankel_pair01(k*r)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            Dlog[i,j]=Rmat[i,j]*l1_ij
            Dsmooth[i,j]=pts.ws[j]*l2_ij
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            Dlog[j,i]=Rmat[j,i]*l1_ji
            Dsmooth[j,i]=pts.ws[i]*l2_ji
        end
    end
    return Dlog,Dsmooth
end

function construct_fredholm_matrix!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(F,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        F[i,i]=one(Complex{T})-2*Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            _,h1=hankel_pair01(k*r)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            F[i,j]=-2*(Rmat[i,j]*l1_ij+pts.ws[j]*l2_ij)
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            F[j,i]=-2*(Rmat[j,i]*l1_ji+pts.ws[i]*l2_ji)
        end
    end
    return F
end

###########################################
#### OPTIONAL K-DERIVATIVE ASSEMBLIES #####
###########################################

function construct_dlp_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},D::AbstractMatrix{Complex{T}},D1::AbstractMatrix{Complex{T}},D2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    fill!(D,zero(Complex{T}))
    fill!(D1,zero(Complex{T}))
    fill!(D2,zero(Complex{T}))
    N=length(parr.X)
    @inbounds for i in 1:N
        D[i,i]=Complex{T}(pts.ws[i]*G.kappa[i],zero(T))
        D1[i,i]=zero(Complex{T})
        D2[i,i]=zero(Complex{T})
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r=G.R[i,j]
            invr=G.invR[i,j]
            lt=G.logterm[i,j]
            inn_ij=G.inner[i,j]
            inn_ji=G.inner[j,i]
            kr=k*r
            h0,h1=hankel_pair01(kr)
            j0=real(h0)
            j1=real(h1)
            l1_ij=αL1*inn_ij*j1*invr
            l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
            D[i,j]=Rmat[i,j]*l1_ij+pts.ws[j]*l2_ij
            l1_ij_1=-(inn_ij*k*j0)*inv_two_pi
            l1_ij_2=(inn_ij*(k*r*j1-j0))*inv_two_pi
            l2_ij_1=(inn_ij*k*(lt*j0+im*pi*h0))*inv_two_pi
            l2_ij_2=(inn_ij*(lt*(j0-k*r*j1)+im*pi*(h0-k*r*h1)))*inv_two_pi
            D1[i,j]=Rmat[i,j]*l1_ij_1+pts.ws[j]*l2_ij_1
            D2[i,j]=Rmat[i,j]*l1_ij_2+pts.ws[j]*l2_ij_2
            l1_ji=αL1*inn_ji*j1*invr
            l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
            D[j,i]=Rmat[j,i]*l1_ji+pts.ws[i]*l2_ji
            l1_ji_1=-(inn_ji*k*j0)*inv_two_pi
            l1_ji_2=(inn_ji*(k*r*j1-j0))*inv_two_pi
            l2_ji_1=(inn_ji*k*(lt*j0+im*pi*h0))*inv_two_pi
            l2_ji_2=(inn_ji*(lt*(j0-k*r*j1)+im*pi*(h0-k*r*h1)))*inv_two_pi
            D1[j,i]=Rmat[j,i]*l1_ji_1+pts.ws[i]*l2_ji_1
            D2[j,i]=Rmat[j,i]*l1_ji_2+pts.ws[i]*l2_ji_2
        end
    end
    return D,D1,D2
end

function construct_fredholm_matrix_derivatives!(solver::Union{DLP_kress,DLP_kress_global_corners},F::AbstractMatrix{Complex{T}},F1::AbstractMatrix{Complex{T}},F2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},G::CFIEGeomCache{T},parr::CFIEPanelArrays{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_dlp_matrix_derivatives!(solver,F,F1,F2,pts,Rmat,G,parr,k;multithreaded=multithreaded)
    @inbounds for j in axes(F,2),i in axes(F,1)
        F[i,j]*=-2
        F1[i,j]*=-2
        F2[i,j]*=-2
    end
    @inbounds for i in axes(F,1)
        F[i,i]+=one(Complex{T})
    end
    return F,F1,F2
end

########################
#### SOLVER LAYER ######
########################

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_fredholm_matrix!(solver,A,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    G=_is_dlp_kress_graded(solver) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    parr=_panel_arrays_cache(pts)
    construct_fredholm_matrix!(solver,A,pts,Rmat,G,parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_fredholm_matrix_derivatives!(solver,A,A1,A2,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    G=_is_dlp_kress_graded(solver) ? cfie_geom_cache(pts,true) : cfie_geom_cache(pts,false)
    parr=_panel_arrays_cache(pts)
    construct_fredholm_matrix_derivatives!(solver,A,A1,A2,pts,Rmat,G,parr,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_dlp_kress_workspace(solver,pts)
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices!(solver::Union{DLP_kress,DLP_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 Rmat=build_Rmat_dlp_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},k,Rmat::AbstractMatrix{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    N=length(pts.xy)
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 Rmat=build_Rmat_dlp_kress(solver,pts)
    @blas_1 construct_matrices!(solver,A,pts,Rmat,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    mu=S[idx]
    u_mu=conj.(Vt[idx,:])
    return mu,u_mu
end

function solve_vect(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,ks::Vector{T};multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPointsCFIE{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,solver.billiard,ks[i])
        _,u=solve_vect(solver,basis,pts,ks[i];multithreaded=multithreaded)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

function solve_INFO(solver::Union{DLP_kress,DLP_kress_global_corners},basis::Ba,pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    A=Matrix{Complex{T}}(undef,ws.N,ws.N)
    t0=time()
    @info "Building boundary operator A from cached DLP-Kress workspace..."
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
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

#######################
#### SANITY CHECKS ####
#######################

function debug_dlp_split_error(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},ws::DLPKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    D=Matrix{Complex{T}}(undef,ws.N,ws.N)
    Dlog=Matrix{Complex{T}}(undef,ws.N,ws.N)
    Dsmooth=Matrix{Complex{T}}(undef,ws.N,ws.N)
    construct_dlp_matrix!(solver,D,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
    construct_dlp_split!(solver,Dlog,Dsmooth,pts,ws.Rmat,ws.G,ws.parr,k;multithreaded=multithreaded)
    Δ=D-(Dlog+Dsmooth)
    return norm(Δ),maximum(abs.(Δ))
end

function debug_dlp_split_error(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPointsCFIE{T},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_dlp_kress_workspace(solver,pts)
    debug_dlp_split_error(solver,pts,ws,k;multithreaded=multithreaded)
end