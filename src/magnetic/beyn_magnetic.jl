# ==============================================================================
# Magnetic CFIE-Kress Beyn solver
# ==============================================================================

const MagneticBeynPoints=BoundaryPointsCFIE

@inline magnetic_bp(pts::BoundaryPointsCFIE)=pts

struct MagneticContourPrecomp{T,W}
    νj::Vector{ComplexF64}
    wj::Vector{Complex{T}}
    ws::Vector{W}
end

@inline _mag_beyn_dim(solver::MagneticKressSolver,pts::BoundaryPointsCFIE,ν)=
    isnothing(solver.symmetry) ? length(pts.xy) :
    length(symmetry_index_orbits(eltype(pts.ds),pts,solver.symmetry,solver.billiard)[1])

@inline _mag_contour_cache(solver::MagneticKressSolver,pts)=
    build_magnetic_kress_geom_workspace(solver,pts)

function _mag_contour_workspace(solver::MagneticKressSolver,pts,ν,cache;h=1e-5,P=6,Msmall=30,mp_dps::Int=30)
    zmax=cache isa MagneticKressGeomWorkspace ? cache.G.zmax : cache.full.G.zmax
    kws=build_magnetic_kress_taylor_workspace(ComplexF64(ν),zmax;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
    return cache,kws
end

struct MagneticFourierFilter{T}
    U::Matrix{Complex{T}}
    idx::Vector{Int}
    mcut::Int
end

function magnetic_fourier_filter(N::Int,mcut::Int;T=Float64,keep::Symbol=:high)
    modes=collect(-fld(N,2):cld(N,2)-1)
    idx=keep===:high ? findall(abs.(modes).>mcut) :
        keep===:low ? findall(abs.(modes).<=mcut) :
        error("keep must be :high or :low")
    F=fft(Matrix{Complex{T}}(I,N,N),1)/sqrt(T(N))
    F=fftshift(F,1)
    U=Matrix(F[idx,:]')
    return MagneticFourierFilter{T}(U,idx,mcut)
end

@inline project_matrix(A,F::MagneticFourierFilter)=F.U'*A*F.U
@inline lift_vector(v,F::MagneticFourierFilter)=F.U*v

function precompute_magnetic_contour(solver::MagneticKressSolver,pts::BoundaryPointsCFIE,ν0::Complex{T},R::T;nq::Int=64,h=1e-5,P=6,Msmall=30,mp_dps::Int=30) where {T<:Real}
    θ=(TWO_PI/nq).*(collect(0:nq-1).+T(0.5))
    ej=cis.(θ)
    νj=ComplexF64.(ν0.+R.*ej)
    wj=Complex{T}.((R/nq).*ej)
    cache=_mag_contour_cache(solver,pts)
    ws1=_mag_contour_workspace(solver,pts,νj[1],cache;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
    ws=Vector{typeof(ws1)}(undef,nq)
    ws[1]=ws1
    @inbounds for q in 2:nq
        ws[q]=_mag_contour_workspace(solver,pts,νj[q],cache;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
    end
    return MagneticContourPrecomp{T,typeof(ws1)}(νj,wj,ws)
end

function construct_boundary_matrices_precomputed!(solver::MagneticKressSolver,pts::BoundaryPointsCFIE,pc::MagneticContourPrecomp;matrix_kind::Symbol=:cfie_src,multithreaded::Bool=true,timeit::Bool=false,operator_convention::Symbol=:unregularized)
    gws=pc.ws[1][1]
    N=_workspace_dim(gws)
    nq=length(pc.νj)
    Tbufs=[Matrix{ComplexF64}(undef,N,N) for _ in 1:nq]
    @blas_1 begin
        @inbounds for q in 1:nq
            @benchit timeit=timeit "magnetic boundary assembly" construct_magnetic_operator_matrix!(Tbufs[q],pts,pc.ws[q][1],pc.ws[q][2];matrix_kind=matrix_kind,multithreaded=multithreaded,operator_convention=operator_convention)
        end
    end
    return Tbufs
end

function _magnetic_boundary_length(billiard)
    b=billiard.full_boundary
    if length(b)==1 && !(b[1] isa AbstractVector)
        return b[1].length
    elseif _is_single_composite_boundary(b)
        return sum(crv.length for crv in b)
    else
        return sum(crv.length for crv in b if typeof(crv)<:AbsRealCurve)
    end
end

function _magnetic_area_or_error(A)
    A===nothing && error("Please pass Euclidean area A=... to plan_ν_windows_magnetic / compute_spectrum_magnetic.")
    return A
end

function plan_ν_windows_magnetic(solver::MagneticKressSolver,billiard::Bi,ν1::T,ν2::T;A=nothing,L=nothing,use_perimeter::Bool=false,M::Int=50,Rmax::T=T(0.8),Rfloor::T=T(1e-6),iters::Int=8) where {Bi<:AbsBilliard,T<:Real}
    ν2<=ν1 && return T[],T[]
    A=T(_magnetic_area_or_error(A))
    L0=T(L===nothing ? _magnetic_boundary_length(billiard) : L)
    B=solver.bmag
    ρ(ν)=begin
        lead=A/(π*B^2)
        corr=use_perimeter ? L0/(2π*B*sqrt(max(ν,T(1e-14)))) : zero(T)
        max(lead-corr,T(1e-12))
    end
    Rof(ν)=clamp(T(M)/(2ρ(ν)),Rfloor,Rmax)
    ν0s=T[]
    Rs=T[]
    left=ν1
    while left<ν2-T(10)*eps(ν2)
        rem=ν2-left
        rem<=zero(T) && break
        R=clamp(rem/2,Rfloor,Rmax)
        @inbounds for _ in 1:iters
            ν0=left+R
            R=min(Rof(ν0),rem/2)
            R=clamp(R,Rfloor,Rmax)
        end
        push!(ν0s,left+R)
        push!(Rs,R)
        left+=2R
    end
    return ν0s,Rs
end

function construct_B_matrix_magnetic(solver::MagneticKressSolver,pts::BoundaryPointsCFIE,pc::MagneticContourPrecomp;r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0),matrix_kind::Symbol=:cfie_src,multithreaded::Bool=true,timeit::Bool=false,mcut::Union{Nothing,Int}=nothing)
    nq=length(pc.νj)
    νj=pc.νj
    wj=pc.wj
    Tbufs=construct_boundary_matrices_precomputed!(solver,pts,pc;matrix_kind=matrix_kind,multithreaded=multithreaded,timeit=timeit)
    Nfull=size(Tbufs[1],1)
    filt=isnothing(mcut) ? nothing : magnetic_fourier_filter(Nfull,mcut;T=real(eltype(pc.wj)),keep=:high)
    if !isnothing(filt)
        @inbounds for q in 1:nq
            Tbufs[q]=project_matrix(Tbufs[q],filt)
        end
    end
    N=size(Tbufs[1],1)
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq)
    Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq
        Fs[j]=lu!(Tbufs[j];check=false)
    end
    T=real(eltype(wj))
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    xv=reshape(X,:)
    a0v=reshape(A0,:)
    a1v=reshape(A1,:)
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds @showprogress desc="ldiv!+axpy!(magnetic)" for j in 1:nq
        ldiv!(X,Fs[j],V)
        #c=Complex{T}(cospi(νj[j]))
        #BLAS.axpy!(wj[j]*c,xv,a0v)
        #BLAS.axpy!(wj[j]*Complex{T}(νj[j])*c,xv,a1v)
        BLAS.axpy!(wj[j],xv,a0v)
        BLAS.axpy!(wj[j]*Complex{T}(νj[j]),xv,a1v)
    end
    @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    rk=count(>=(svd_tol),Σ)
    rk==0 && return Matrix{Complex{T}}(undef,0,0),Matrix{Complex{T}}(undef,N,0),filt
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    return B,Matrix(Uk),filt
end

function solve_vect_magnetic(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE,pc::MagneticContourPrecomp;r::Int=48,svd_tol::Real=1e-14,rng=MersenneTwister(0),matrix_kind::Symbol=:cfie_src,multithreaded::Bool=true,timeit::Bool=false,mcut::Union{Nothing,Int}=nothing) where {Ba<:AbstractHankelBasis}
    B,Uk,filt=construct_B_matrix_magnetic(solver,pts,pc;r=r,svd_tol=svd_tol,rng=rng,matrix_kind=matrix_kind,multithreaded=multithreaded,timeit=timeit,mcut=mcut)
    isempty(B) && return ComplexF64[],Uk,Matrix{ComplexF64}(undef,0,0),pc.νj[1],zero(real(eltype(pc.wj))),pts,filt
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B)
    return λ,Uk,Y,pc.νj[1],zero(real(eltype(pc.wj))),pts,filt
end

@inline function solve_magnetic(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE,pc::MagneticContourPrecomp;kwargs...) where {Ba<:AbstractHankelBasis}
    λ,_,_,_,_,_,_=solve_vect_magnetic(solver,basis,pts,pc;kwargs...)
    return λ
end

function solve_INFO_magnetic(solver::MagneticKressSolver,basis::Ba,pts::BoundaryPointsCFIE,ν0::Complex{T},R::T;multithreaded::Bool=true,nq::Int=64,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0),use_adaptive_svd_tol::Bool=false,auto_discard_spurious::Bool=false,matrix_kind::Symbol=:cfie_src,h=1e-5,P=6,Msmall=30,mp_dps::Int=30,timeit::Bool=false,mcut::Union{Nothing,Int}=nothing) where {Ba<:AbstractHankelBasis,T<:Real}
    pc=precompute_magnetic_contour(solver,pts,ν0,R;nq=nq,h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
    νj=pc.νj
    wj=pc.wj
    @time "Boundary matrices (magnetic)" Tbufs=construct_boundary_matrices_precomputed!(solver,pts,pc;matrix_kind=matrix_kind,multithreaded=multithreaded,timeit=timeit)
    Nfull=size(Tbufs[1],1)
    filt=isnothing(mcut) ? nothing : magnetic_fourier_filter(Nfull,mcut;T=real(eltype(wj)),keep=:high)
    if !isnothing(filt)
        @inbounds for q in 1:nq
            Tbufs[q]=project_matrix(Tbufs[q],filt)
        end
    end
    N=size(Tbufs[1],1)
    @info "beyn:start(magnetic)" ν0=ν0 R=R nq=nq N=N Nfull=Nfull r=r matrix_kind=matrix_kind mcut=mcut
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq)
    Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds @showprogress desc="lu!(magnetic)" for j in 2:nq
        an=opnorm(Tbufs[j],1)
        Fs[j]=lu!(Tbufs[j];check=false)
        rc=LAPACK.gecon!('1',Fs[j].factors,an)
        println("LAPACK.gecon! = ",rc)
    end
    xv=reshape(X,:)
    a0v=reshape(A0,:)
    a1v=reshape(A1,:)
    @time "ldiv!+axpy!(magnetic)" begin
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds @showprogress desc="ldiv!+axpy!(magnetic)" for j in 1:nq
            ldiv!(X,Fs[j],V)
            #c=Complex{T}(cospi(νj[j]))
            #BLAS.axpy!(wj[j]*c,xv,a0v)
            #BLAS.axpy!(wj[j]*Complex{T}(νj[j])*c,xv,a1v)
            BLAS.axpy!(wj[j],xv,a0v)
            BLAS.axpy!(wj[j]*Complex{T}(νj[j]),xv,a1v)
        end
    end
    @show typeof(A0) size(A0) strides(A0)
    @assert all(isfinite,A0)
    @time "SVD(magnetic)" @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    println("Singular values of A0: ",Σ)
    svd_tol_eff=use_adaptive_svd_tol ? maximum(Σ)*1e-15 : svd_tol
    rk=0
    @inbounds for i in eachindex(Σ)
        if Σ[i]>=svd_tol_eff
            rk+=1
        else
            break
        end
    end
    rk==r && @warn "All singular values are above svd_tol=$(svd_tol_eff), r=$(r) may need to be increased"
    rk==0 && return Complex{T}[],Matrix{Complex{T}}(undef,Nfull,0),T[]
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    @time "eigen(magnetic)" @blas_multi_then_1 MAX_BLAS_THREADS ev=eigen!(B)
    λ=ev.values
    Y=ev.vectors
    Φf=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi MAX_BLAS_THREADS mul!(Φf,Uk,Y)
    Φ=isnothing(filt) ? Φf : filt.U*Φf
    keep=trues(length(λ))
    tens=T[]
    res_keep=T[]
    ybuf=Vector{Complex{T}}(undef,Nfull)
    A=Matrix{Complex{T}}(undef,Nfull,Nfull)
    cache=_mag_contour_cache(solver,pts)
    dropped_out=0
    dropped_res=0
    @inbounds for j in eachindex(λ)
        if abs(λ[j]-ν0)>R
            keep[j]=false
            dropped_out+=1
            continue
        end
        wsj=_mag_contour_workspace(solver,pts,ComplexF64(λ[j]),cache;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
        construct_matrices!(solver,A,pts,wsj[1],wsj[2],λ[j];matrix_kind=matrix_kind,mp_dps=mp_dps,multithreaded=multithreaded,operator_convention=:regularized)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(ybuf,A,@view(Φ[:,j]))
        rj=norm(ybuf)
        @info "ν=$(λ[j]) ||A_full(ν)v_full|| = $(rj) vs. res_tol $res_tol"
        if auto_discard_spurious && rj>=res_tol
            keep[j]=false
            dropped_res+=1
            continue
        end
        push!(tens,T(rj))
        push!(res_keep,T(rj))
    end
    kept=count(keep)
    kept>0 ? @info("STATUS(magnetic): ",kept=kept,dropped_outside=dropped_out,dropped_residual=dropped_res,max_residual=maximum(res_keep)) : @info("STATUS(magnetic): ",kept=0,dropped_outside=dropped_out,dropped_residual=dropped_res)
    return λ[keep],Φ[:,keep],tens
end

function residual_and_norm_select_magnetic(solver::MagneticKressSolver,λ::AbstractVector{Complex{T}},Uk::AbstractMatrix{Complex{T}},Y::AbstractMatrix{Complex{T}},ν0::Complex{T},R::T,pts::BoundaryPointsCFIE;filt=nothing,res_tol::T,matrix_kind::Symbol=:cfie_src,matnorm::Symbol=:one,epss::Real=1e-15,auto_discard_spurious::Bool=true,collect_logs::Bool=false,multithreaded::Bool=true,h=1e-5,P=6,Msmall=30,mp_dps::Int=30) where {T<:Real}
    cache=_mag_contour_cache(solver,pts)
    Nf,rk=size(Uk)
    Nfull=isnothing(filt) ? Nf : size(filt.U,1)
    Φtmp=Matrix{Complex{T}}(undef,Nfull,rk)
    y=Vector{Complex{T}}(undef,Nfull)
    keep=falses(rk)
    tens=Vector{T}(undef,rk)
    tensN=Vector{T}(undef,rk)
    logs=collect_logs ? String[] : nothing
    A=Matrix{Complex{T}}(undef,Nfull,Nfull)
    φf=Vector{Complex{T}}(undef,Nf)
    vecnorm=matnorm===:one ? (v->norm(v,1)) : matnorm===:two ? (v->norm(v)) : (v->norm(v,Inf))
    @inbounds for j in 1:rk
        λj=λ[j]
        if abs(λj-ν0)>R
            tens[j]=T(NaN)
            tensN[j]=T(NaN)
            continue
        end
        mul!(φf,Uk,@view(Y[:,j]))
        if isnothing(filt)
            copyto!(@view(Φtmp[:,j]),φf)
        else
            mul!(@view(Φtmp[:,j]),filt.U,φf)
        end
        wsj=_mag_contour_workspace(solver,pts,ComplexF64(λj),cache;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
        construct_matrices!(solver,A,pts,wsj[1],wsj[2],λj;matrix_kind=matrix_kind,mp_dps=mp_dps,multithreaded=multithreaded,operator_convention=:regularized)
        mul!(y,A,@view(Φtmp[:,j]))
        rj=norm(y)
        tens[j]=rj
        nA=matnorm===:one ? opnorm(A,1) : matnorm===:two ? opnorm(A,2) : opnorm(A,Inf)
        φn=vecnorm(@view(Φtmp[:,j]))
        yn=vecnorm(y)
        tensN[j]=yn/(nA*(φn+epss)+epss)
        keep[j]=!(auto_discard_spurious && rj>=res_tol)
    end
    idx=findall(keep)
    Φ_kept=isempty(idx) ? Matrix{Complex{T}}(undef,Nfull,0) : Φtmp[:,idx]
    return idx,Φ_kept,tens[idx],tensN[idx],(collect_logs ? logs : String[])
end

function imag_ν_check_magnetic_EXPERIMENTAL(solver::MagneticKressSolver,λs::Vector{Vector{Complex{T}}},Uks::Vector{Matrix{Complex{T}}},Ys::Vector{Matrix{Complex{T}}},Is::Vector{Vector{Int}},ν0s::Vector{Complex{T}},Rs::Vector{T},all_pts;res_tol::T,pad::Int=20,group_size::Int=64,matrix_kind::Symbol=:cfie_src,multithreaded::Bool=true,verbose::Bool=true,h=1e-5,P=6,Msmall=30,mp_dps::Int=30) where {T<:Real}
    nw=length(λs)
    idx_inside=Vector{Vector{Int}}(undef,nw)
    idx_keep=Vector{Vector{Int}}(undef,nw)
    residuals=Vector{Vector{T}}(undef,nw)
    local_pos=Dict{Tuple{Int,Int},Int}()
    candidates=Tuple{Int,Int,T,T}[]
    @inbounds for i in 1:nw
        λi=λs[i]
        idx_inside[i]=findall(j->abs(λi[j]-ν0s[i])<=Rs[i],eachindex(λi))
        idx_keep[i]=copy(idx_inside[i])
        residuals[i]=fill(T(NaN),length(idx_inside[i]))
        for (lp,j) in pairs(idx_inside[i])
            local_pos[(i,j)]=lp
            push!(candidates,(i,j,abs(imag(λi[j])),real(λi[j])))
        end
    end
    sort!(candidates;by=c->c[3],rev=true)
    caches=Vector{Any}(fill(nothing,nw))
    drop=Dict{Tuple{Int,Int},Bool}()
    checked=0
    dropped=0
    good_streak=0
    pos=1
    A=Matrix{Complex{T}}(undef,0,0)
    y=Vector{Complex{T}}(undef,0)
    φ=Vector{Complex{T}}(undef,0)
    while pos<=length(candidates)
        stop=min(pos+group_size-1,length(candidates))
        group=@view candidates[pos:stop]
        rdict=Dict{Tuple{Int,Int},T}()
        for iwin in unique(c[1] for c in group)
            sub=[c for c in group if c[1]==iwin]
            isempty(sub) && continue
            pts=all_pts[iwin]
            caches[iwin]===nothing && (caches[iwin]=_mag_contour_cache(solver,pts))
            cache=caches[iwin]
            N=_workspace_dim(cache)
            size(A)!=(N,N) && (A=Matrix{Complex{T}}(undef,N,N))
            length(y)!=N && (y=Vector{Complex{T}}(undef,N))
            length(φ)!=N && (φ=Vector{Complex{T}}(undef,N))
            @inbounds for c in sub
                i,j,_,_=c
                λj=λs[i][j]
                wsj=_mag_contour_workspace(solver,pts,ComplexF64(λj),cache;h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
                construct_matrices!(solver,A,pts,wsj[1],wsj[2],λj;matrix_kind=matrix_kind,mp_dps=mp_dps,multithreaded=multithreaded,operator_convention=:regularized)
                mul!(φ,Uks[i],@view Ys[i][:,j])
                mul!(y,A,φ)
                rdict[(i,j)]=norm(y)
            end
        end
        @inbounds for c in group
            i,j,imj,_=c
            rj=rdict[(i,j)]
            checked+=1
            residuals[i][local_pos[(i,j)]]=rj
            if rj>=res_tol
                drop[(i,j)]=true
                dropped+=1
                good_streak=0
                verbose && @info "DROP magnetic candidate" i=i j=j ν=λs[i][j] abs_imag=imj residual=rj
            else
                good_streak+=1
                good_streak>=pad && break
            end
        end
        good_streak>=pad && break
        pos=stop+1
    end
    @inbounds for i in 1:nw
        old=idx_inside[i]
        mask=[!get(drop,(i,j),false) for j in old]
        idx_keep[i]=old[mask]
        residuals[i]=residuals[i][mask]
    end
    verbose && @info "magnetic imag-tail summary" checked=checked dropped=dropped total_candidates=length(candidates)
    return idx_keep,residuals
end

function compute_spectrum_magnetic(solver::MagneticKressSolver,basis::Ba,billiard::Bi,ν1::T,ν2::T;A=nothing,L=nothing,use_perimeter::Bool=false,m::Int=10,Rmax::T=T(0.8),Rfloor::T=T(1e-6),nq::Int=64,r::Int=m+15,svd_tol::Real=1e-12,res_tol::Real=1e-9,auto_discard_spurious::Bool=true,matrix_kind::Symbol=:cfie_src,multithreaded_matrix::Bool=true,h=1e-5,P=6,Msmall=30,mp_dps::Int=30,return_imag_part::Bool=true,use_imag_residual_check::Bool=false,timeit::Bool=false,do_INFO::Bool=true,mcut::Union{Nothing,Int}=nothing) where {T<:Real,Bi<:AbsBilliard,Ba<:AbstractHankelBasis}
    @time "ν-windows (magnetic)" ν0s,Rs=plan_ν_windows_magnetic(solver,billiard,ν1,ν2;A=A,L=L,use_perimeter=use_perimeter,M=m,Rmax=Rmax,Rfloor=Rfloor)
    idx=findall(>(max(zero(T),Rfloor)),Rs)
    ν0s=isempty(idx) ? T[] : ν0s[idx]
    Rs=isempty(idx) ? T[] : Rs[idx]
    nw=length(ν0s)
    nw==0 && return (return_imag_part ? Complex{T}[] : T[]),T[],Vector{Vector{Complex{T}}}(),Vector{BoundaryPointsCFIE}(),T[]
    println("Number of magnetic windows: ",nw)
    println("Average Rν: ",sum(Rs)/T(nw))
    first_pts=evaluate_points(solver,billiard,T(k_from_ν_magnetic(ν0s[1],solver.bmag)))
    PtsT=typeof(first_pts)
    all_pts=Vector{PtsT}(undef,nw)
    all_pts[1]=first_pts
    @time "Point evaluation (magnetic)" begin
        @showprogress desc="magnetic pts construction" Threads.@threads for i in 1:nw
            i>1 && (all_pts[i]=evaluate_points(solver,billiard,T(k_from_ν_magnetic(ν0s[i],solver.bmag))))
        end
    end
    if do_INFO
        iinfo=cld(nw,2)
        @time "solve_INFO middle disk (magnetic)" begin
            _=solve_INFO_magnetic(solver,basis,all_pts[iinfo],complex(ν0s[iinfo],zero(T)),Rs[iinfo];multithreaded=multithreaded_matrix,nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),use_adaptive_svd_tol=false,auto_discard_spurious=false,matrix_kind=matrix_kind,h=h,P=P,Msmall=Msmall,mp_dps=mp_dps,timeit=timeit,mcut=mcut)
        end
    end
    λs=Vector{Vector{Complex{T}}}(undef,nw)
    Uks=Vector{Matrix{Complex{T}}}(undef,nw)
    Ys=Vector{Matrix{Complex{T}}}(undef,nw)
    filts=Vector{Any}(undef,nw)
    p=Progress(nw,1)
    @time "Beyn pass (all disks) (magnetic)" @inbounds for i in 1:nw
        pc=precompute_magnetic_contour(solver,all_pts[i],complex(ν0s[i],zero(T)),Rs[i];nq=nq,h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
        λ,Uk,Y,_,_,_,filt=solve_vect_magnetic(solver,basis,all_pts[i],pc;r=r,svd_tol=svd_tol,rng=MersenneTwister(0),matrix_kind=matrix_kind,multithreaded=multithreaded_matrix,timeit=timeit,mcut=mcut)
        λs[i]=λ
        Uks[i]=Uk
        Ys[i]=Y
        filts[i]=filt
        pc=nothing
        next!(p)
    end
    ν_list=Vector{Vector{Complex{T}}}(undef,nw)
    tens_list=Vector{Vector{T}}(undef,nw)
    tensN_list=Vector{Vector{T}}(undef,nw)
    phi_list=Vector{Matrix{Complex{T}}}(undef,nw)
    @time "Residuals/tensions pass (magnetic)" @inbounds @showprogress desc="Residuals/tensions (magnetic)" for i in 1:nw
        if isempty(λs[i])
            ν_list[i]=Complex{T}[]
            tens_list[i]=T[]
            tensN_list[i]=T[]
            phi_list[i]=Matrix{Complex{T}}(undef,size(Uks[i],1),0)
            continue
        end
        idx2,Φ_kept,traw,tnorm,_=residual_and_norm_select_magnetic(solver,λs[i],Uks[i],Ys[i],complex(ν0s[i],zero(T)),Rs[i],all_pts[i];filt=filts[i],res_tol=T(res_tol),matrix_kind=matrix_kind,matnorm=:one,auto_discard_spurious=auto_discard_spurious,multithreaded=multithreaded_matrix,h=h,P=P,Msmall=Msmall,mp_dps=mp_dps)
        ν_list[i]=λs[i][idx2]
        tens_list[i]=traw
        tensN_list[i]=tnorm
        phi_list[i]=Matrix(Φ_kept)
    end
    n_by_win=[size(phi_list[i],2) for i in 1:nw]
    offs=zeros(Int,nw)
    @inbounds for i in 2:nw
        offs[i]=offs[i-1]+n_by_win[i-1]
    end
    ntot=offs[end]+n_by_win[end]
    ν_all=Vector{Complex{T}}(undef,ntot)
    tens_all=Vector{T}(undef,ntot)
    tensN_all=Vector{T}(undef,ntot)
    us_all=Vector{Vector{Complex{T}}}(undef,ntot)
    pts_all=Vector{PtsT}(undef,ntot)
    Threads.@threads for i in 1:nw
        n=n_by_win[i]
        n==0 && continue
        off=offs[i]
        Φ=phi_list[i]
        pts=all_pts[i]
        @inbounds for j in 1:n
            ν_all[off+j]=ν_list[i][j]
            tens_all[off+j]=tens_list[i][j]
            tensN_all[off+j]=tensN_list[i][j]
            us_all[off+j]=vec(@view Φ[:,j])
            pts_all[off+j]=pts
        end
    end
    ν_out=return_imag_part ? ν_all : real.(ν_all)
    return ν_out,tens_all,us_all,pts_all,tensN_all
end