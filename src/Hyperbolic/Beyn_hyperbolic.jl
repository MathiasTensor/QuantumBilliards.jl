function plan_k_windows_hyp(::Bi,k1::T,k2::T;M::T=T(10),overlap::T=T(0.5),Rmax::T=T(0.8),Rfloor::T=T(1e-6),kref::T=T(1000)) where {T<:Real,Bi}
    L=k2-k1
    (L<=zero(T) || Rmax<=zero(T)) && return T[],T[]
    ov=clamp(overlap,zero(T),T(0.95))
    if L<=2Rmax
        R=max(Rfloor,min(Rmax,L/2))
        return T[0.5*(k1+k2)],T[R]
    end
    step=max(T(2)*Rfloor,T(2)*Rmax*(one(T)-ov))
    k0s=T[];Rs=T[]
    k0=k1+Rmax
    k0_end=k2-Rmax
    while k0<=k0_end+T(10)*eps(k0_end)
        push!(k0s,k0);push!(Rs,Rmax);k0+=step
    end
    if isempty(k0s) || abs(k0s[end]-k0_end)>T(10)*eps(k0_end)
        push!(k0s,k0_end);push!(Rs,Rmax)
    else
        k0s[end]=k0_end
    end
    #safety: if last disk would start after k1 or end before k2 due to roundoff, nudge endpoints
    k0s[1]=k1+Rmax;k0s[end]=k2-Rmax
    return k0s,Rs
end

function construct_B_matrix_hyp(solver::BIM_hyperbolic,pts::BoundaryPointsHypBIM{T},N::Int,k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol=1e-14,rng=MersenneTwister(0),multithreaded::Bool=true,h=1e-4,P=30,mp_dps::Int=60,leg_type::Int=3)::Tuple{Matrix{Complex{T}},Matrix{Complex{T}}} where {T<:Real}
    @info "Constructing B matrix (hyp) with N=$N, k0=$k0, R=$R, nq=$nq, r=$r"
    θ=(TWO_PI/nq).*(collect(0:nq-1).+0.5)
    ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej
    ks=ComplexF64.(zj)
    Tbufs=[zeros(Complex{T},N,N) for _ in 1:nq]
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry)
    pts_eucl=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts)
    xy=pts_eucl.xy
    if norm(xy[1]-xy[end])<1e-14
        @warn "Duplicate endpoint in boundary points; drop last point!" N=length(xy)
    end
    pre=build_QTaylorPrecomp(dmin=dmin,dmax=dmax,h=h,P=P)
    tabs=alloc_QTaylorTables(pre,nq;k=ks[1])
    ws=QTaylorWorkspace(P;threaded=multithreaded)
    build_QTaylorTable!(tabs,pre,ws,ks;mp_dps=mp_dps,leg_type=leg_type,threaded=multithreaded)
    compute_kernel_matrices_DLP_hyperbolic!(Tbufs,pts_eucl,solver.symmetry,tabs;multithreaded=multithreaded)
    assemble_DLP_hyperbolic!(Tbufs,pts_eucl)
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq);Fs[1]=F1
    A=rand(ComplexF64,N,N)
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq
        an∞=opnorm(Tbufs[j],Inf)
        Fs[j]=lu!(Tbufs[j];check=false)
        rc=LinearAlgebra.LAPACK.gecon!('1',Fs[j].factors,an∞)
        println("rcond=",rc)
    end
    function accum_moments!(A0::Matrix{Complex{T}},A1::Matrix{Complex{T}},X::Matrix{Complex{T}},V::Matrix{Complex{T}})
        xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:)
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 1:nq
            ldiv!(X,Fs[j],V)
            BLAS.axpy!(wj[j],xv,a0v)
            BLAS.axpy!(wj[j]*zj[j],xv,a1v)
        end
        return nothing
    end
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    accum_moments!(A0,A1,X,V)
    @show typeof(A0) size(A0) strides(A0)
    @show A0 isa StridedMatrix
    @show stride(A0,1) stride(A0,2)
    @assert size(A0,1)>0 && size(A0,2)>0
    @assert stride(A0,2)>=max(1,size(A0,1))
    @assert all(isfinite,A0)
    @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    rk=count(>=(svd_tol),Σ)
    rk==0 && return Matrix{Complex{T}}(undef,N,0),Matrix{Complex{T}}(undef,N,0)
    if rk==r
        r_tmp=r+r
        while r_tmp<N
            V,X,A0,A1=beyn_buffer_matrices(T,N,r_tmp,rng)
            accum_moments!(A0,A1,X,V)
            @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
            rk=count(>=(svd_tol),Σ)
            rk<r_tmp && break
            r_tmp+=r
            r_tmp>N && throw(ArgumentError("r > N is impossible: requested r=$(r_tmp), N=$(N)"))
        end
        rk==r_tmp && @warn "All singular values ≥ svd_tol=$(svd_tol); consider increasing r or decreasing R"
    end
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds @simd for j in 1:rk
        @views tmp[:,j]./=Σk[j]
    end
    B=Matrix{Complex{T}}(undef,rk,rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    return B,Uk
end

function solve_vect_hyp(solver::BIM_hyperbolic,basis::Ba,pts::BoundaryPointsHypBIM{T},k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),multithreaded::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    B,Uk=construct_B_matrix_hyp(solver,pts,N,k0,R;nq=nq,r=r,svd_tol=svd_tol,rng=rng,multithreaded=multithreaded,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
    if isempty(B)
        return Complex{T}[],Uk,Matrix{Complex{T}}(undef,0,0),k0,R,pts
    end
    @blas_multi_then_1 MAX_BLAS_THREADS λ,Y=eigen!(B)
    return λ,Uk,Y,k0,R,pts
end

@inline function solve_hyp(solver::BIM_hyperbolic,basis::Ba,pts::BoundaryPointsHypBIM{T},k0::Complex{T},R::T;nq::Int=64,r::Int=48,svd_tol::Real=1e-14,res_tol::Real=1e-8,rng=MersenneTwister(0),multithreaded::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {Ba<:AbstractHankelBasis,T<:Real}
    λ,_,_,_,_,_=solve_vect_hyp(solver,basis,pts,k0,R;nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=rng,multithreaded=multithreaded,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
    return λ
end

function residual_and_norm_select_hyp(solver::BIM_hyperbolic,λ::AbstractVector{Complex{T}},Uk::AbstractMatrix{Complex{T}},Y::AbstractMatrix{Complex{T}},k0::Complex{T},R::T,pts::BoundaryPointsHypBIM{T};res_tol::T,matnorm::Symbol=:one,epss::Real=1e-15,auto_discard_spurious::Bool=true,collect_logs::Bool=false,multithreaded::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {T<:Real}
    N,rk=size(Uk)
    Φtmp=Matrix{Complex{T}}(undef,N,rk)
    y=Vector{Complex{T}}(undef,N)
    keep=falses(rk)
    tens=Vector{T}(undef,rk)
    tensN=Vector{T}(undef,rk)
    logs=collect_logs ? String[] : nothing
    A_buf=fill(zero(Complex{T}),N,N)
    pts_eucl=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts)
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry)
    pre=build_QTaylorPrecomp(dmin=dmin,dmax=dmax,h=T(h),P=P)
    tab=alloc_QTaylorTable(pre;k=ComplexF64(k0))
    ws=QTaylorWorkspace(P;threaded=false)
    vecnorm = matnorm===:one ? (v->norm(v,1)) : matnorm===:two ? (v->norm(v)) : (v->norm(v,Inf))
    @inbounds for j in 1:rk
        λj=λ[j]
        abs(λj-k0)>R && (tens[j]=T(NaN);tensN[j]=T(NaN);continue)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(@view(Φtmp[:,j]),Uk,@view(Y[:,j]))
        build_QTaylorTable!(tab,pre,ws,ComplexF64(λj);mp_dps=mp_dps,leg_type=leg_type)
        compute_kernel_matrices_DLP_hyperbolic!(A_buf,pts_eucl,solver.symmetry,tab;multithreaded=multithreaded)
        assemble_DLP_hyperbolic!(A_buf,pts_eucl)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(y,A_buf,@view(Φtmp[:,j]))
        rj=norm(y)
        tens[j]=rj
        nA=matnorm===:one ? opnorm(A_buf,1) : matnorm===:two ? opnorm(A_buf,2) : opnorm(A_buf,Inf)
        φn=vecnorm(@view(Φtmp[:,j]))
        yn=vecnorm(y)
        tensN[j]=yn/(nA*(φn+epss)+epss)
        if auto_discard_spurious && rj≥res_tol
            collect_logs && push!(logs,"λ=$(λj) ||Aφ||=$(rj) > $res_tol → DROP")
        else
            keep[j]=true
            collect_logs && push!(logs,"λ=$(λj) ||Aφ||=$(rj) < $res_tol ← KEEP")
        end
    end
    idx=findall(keep)
    Φ_kept=isempty(idx) ? Matrix{Complex{T}}(undef,N,0) : Φtmp[:,idx]
    return idx,Φ_kept,tens[idx],tensN[idx],(collect_logs ? logs : String[])
end

function solve_INFO_hyp(solver::BIM_hyperbolic,basis::Ba,pts::BoundaryPointsHypBIM{T},k0::Complex{T},R::T;multithreaded::Bool=true,nq::Int=64,r::Int=48,svd_tol::Real=1e-10,res_tol::Real=1e-10,rng=MersenneTwister(0),use_adaptive_svd_tol::Bool=false,auto_discard_spurious::Bool=false,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3) where {Ba<:AbstractHankelBasis,T<:Real}
    N=length(pts.xy)
    θ=(TWO_PI/nq).*(collect(0:nq-1).+0.5)
    ej=cis.(θ);zj=k0.+R.*ej;wj=(R/nq).*ej
    ks=ComplexF64.(zj)
    V,X,A0,A1=beyn_buffer_matrices(T,N,r,rng)
    @info "beyn:start(hyp)" k0=k0 R=R nq=nq N=N r=r
    Tbufs=[zeros(Complex{T},N,N) for _ in 1:nq]
    dmin,dmax=d_bounds_hyp(pts,solver.symmetry)
    pts_eucl=_BoundaryPointsHypBIM_to_BoundaryPointsBIM(pts)
    pre=build_QTaylorPrecomp(dmin=T(dmin),dmax=T(dmax),h=T(h),P=P)
    tabs=alloc_QTaylorTables(pre,nq;k=ks[1])
    ws=QTaylorWorkspace(P;threaded=multithreaded)
    build_QTaylorTable!(tabs,pre,ws,ks;mp_dps=mp_dps,leg_type=leg_type,threaded=multithreaded)
    @time "DLP(hyp):kernel+assemble" begin
        compute_kernel_matrices_DLP_hyperbolic!(Tbufs,pts_eucl,solver.symmetry,tabs;multithreaded=multithreaded)
        assemble_DLP_hyperbolic!(Tbufs,pts_eucl)
    end
    @blas_multi MAX_BLAS_THREADS F1=lu!(Tbufs[1];check=false)
    Fs=Vector{typeof(F1)}(undef,nq);Fs[1]=F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds @showprogress desc="lu!(hyp)" for j in 2:nq
        Fs[j]=lu!(Tbufs[j];check=false)
    end
    xv=reshape(X,:);a0v=reshape(A0,:);a1v=reshape(A1,:)
    @time "ldiv!+axpy!(hyp)" begin
        @blas_multi_then_1 MAX_BLAS_THREADS @inbounds @showprogress desc="ldiv!+axpy!(hyp)" for j in 1:nq
            ldiv!(X,Fs[j],V)
            BLAS.axpy!(wj[j],xv,a0v)
            BLAS.axpy!(wj[j]*zj[j],xv,a1v)
        end
    end
    @time "SVD(hyp)" @blas_multi_then_1 MAX_BLAS_THREADS U,Σ,W=svd!(A0;full=false)
    println("Singular values (<1e-10 tail inspection): ",Σ)
    svd_tol_eff=use_adaptive_svd_tol ? maximum(Σ)*1e-15 : svd_tol
    rk=0;@inbounds for i in eachindex(Σ)
        if Σ[i]≥svd_tol_eff;rk+=1 else;break end
    end
    rk==r && @warn "All singular values are above svd_tol=$(svd_tol_eff), r=$(r) needs to be increased"
    rk==0 && return Complex{T}[],Matrix{Complex{T}}(undef,N,0),T[]
    Uk=@view U[:,1:rk]
    Wk=@view W[:,1:rk]
    Σk=@view Σ[1:rk]
    tmp=Matrix{Complex{T}}(undef,N,rk);@blas_multi MAX_BLAS_THREADS mul!(tmp,A1,Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j]./=Σk[j] 
    end
    B=Matrix{Complex{T}}(undef,rk,rk);@blas_multi MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
    @time "eigen(hyp)" @blas_multi_then_1 MAX_BLAS_THREADS ev=eigen!(B)
    λ=ev.values;Y=ev.vectors;Phi=Uk*Y
    keep=trues(length(λ));tens=T[];res_keep=T[]
    ybuf=Vector{Complex{T}}(undef,N);A_buf=Matrix{Complex{T}}(undef,N,N)
    tab=alloc_QTaylorTable(pre;k=ComplexF64(k0));ws1=QTaylorWorkspace(P;threaded=false)
    dropped_out=0
    dropped_res=0
    @inbounds for j in eachindex(λ)
        d=abs(λ[j]-k0)
        if d>R;keep[j]=false;dropped_out+=1;continue end
        build_QTaylorTable!(tab,pre,ws1,ComplexF64(λ[j]);mp_dps=mp_dps,leg_type=leg_type)
        fill!(A_buf,zero(eltype(A_buf)))
        compute_kernel_matrices_DLP_hyperbolic!(A_buf,pts_eucl,solver.symmetry,tab;multithreaded=multithreaded)
        assemble_DLP_hyperbolic!(A_buf,pts_eucl)
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(ybuf,A_buf,@view(Phi[:,j]))
        ybn=norm(ybuf)
        @info "k=$(λ[j]) ||A(k)v(k)|| = $(ybn) < $res_tol"
        if auto_discard_spurious && ybn≥res_tol
            keep[j]=false;dropped_res+=1
            if ybn>1e-8
                if ybn>1e-6;@warn "k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , definitely spurious"
                else;@warn "k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , most probably eigenvalue but too low nq" end
            else
                @warn "k=$(λ[j]) ||A(k)v(k)||=$(ybn) > $res_tol , could be spurious or increase nq"
            end
            continue
        end
        push!(tens,T(ybn));push!(res_keep,T(ybn))
    end
    kept=count(keep)
    kept>0 ? @info("STATUS(hyp): ",kept=kept,dropped_outside=dropped_out,dropped_residual=dropped_res,max_residual=maximum(res_keep)) :
             @info("STATUS(hyp): ",kept=0,dropped_outside=dropped_out,dropped_residual=dropped_res)
    return λ[keep],Phi[:,keep],tens
end

function compute_spectrum_hyp(solver::BIM_hyperbolic,basis::Ba,billiard::Bi,k1::T,k2::T;m::Int=10,Rmax::T=T(0.8),overlap::T=T(0.5),nq::Int=64,r::Int=m+15,svd_tol::Real=1e-12,res_tol::Real=1e-9,auto_discard_spurious::Bool=true,multithreaded_matrix::Bool=true,h::T=T(1e-4),P::Int=30,mp_dps::Int=60,leg_type::Int=3,kref::T=T(1000.0),do_INFO::Bool=true,Rfloor::T=T(1e-6)) where {T<:Real,Bi<:AbsBilliard,Ba<:AbstractHankelBasis}
    @time "k-windows (hyp)" k0s,Rs=plan_k_windows_hyp(billiard,k1,k2;M=T(m),overlap=overlap,Rmax=Rmax,Rfloor=Rfloor,kref=kref)
    idx=findall(>(max(zero(T),Rfloor)),Rs)
    k0s=isempty(idx) ? T[] : k0s[idx]
    Rs =isempty(idx) ? T[] : Rs[idx]
    nw=length(k0s);nw==0 && return T[],T[],Vector{Vector{Complex{T}}}(),Vector{BoundaryPointsHypBIM{T}}(),T[]
    println("Number of windows: ",nw);println("Average R: ",sum(Rs)/T(nw))
    all_pts=Vector{BoundaryPointsHypBIM{T}}(undef,nw)
    pre=precompute_hyperbolic_boundary_cdfs(solver,billiard;M_cdf_base=4000,safety=1e-14)
    @time "Point evaluation" @inbounds for i in 1:nw
        all_pts[i]=evaluate_points(solver,billiard,k0s[i],pre;safety=1e-14,threaded=multithreaded_matrix)
        dmin,dmax=d_bounds_hyp(all_pts[i],solver.symmetry)
        @show i k0s[i] Rs[i] length(all_pts[i].xy) dmin dmax
    end
    if do_INFO
        iinfo=cld(nw,2)
        @time "solve_INFO last disk (hyp)" begin
            _=solve_INFO_hyp(solver,basis,all_pts[iinfo],complex(k0s[iinfo],zero(T)),Rs[iinfo];multithreaded=multithreaded_matrix,nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),use_adaptive_svd_tol=false,auto_discard_spurious=false,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        end
    end
    λs=Vector{Vector{Complex{T}}}(undef,nw);Uks=Vector{Matrix{Complex{T}}}(undef,nw);Ys=Vector{Matrix{Complex{T}}}(undef,nw)
    p=Progress(nw,1)
    @time "Beyn pass (all disks) (hyp)" @inbounds for i in 1:nw
        λ,Uk,Y,_,_,_=solve_vect_hyp(solver,basis,all_pts[i],complex(k0s[i],zero(T)),Rs[i];nq=nq,r=r,svd_tol=svd_tol,res_tol=res_tol,rng=MersenneTwister(0),multithreaded=multithreaded_matrix,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        λs[i]=λ;Uks[i]=Uk;Ys[i]=Y
        next!(p)
    end
    ks_list=Vector{Vector{T}}(undef,nw);tens_list=Vector{Vector{T}}(undef,nw);tensN_list=Vector{Vector{T}}(undef,nw);phi_list=Vector{Matrix{Complex{T}}}(undef,nw)
    @time "Residuals/tensions pass (hyp)" @inbounds @showprogress desc="Residuals/tensions (hyp)" for i in 1:nw
        if isempty(λs[i])
            ks_list[i]=T[];tens_list[i]=T[];tensN_list[i]=T[];phi_list[i]=Matrix{Complex{T}}(undef,length(all_pts[i].xy),0);continue
        end
        idx2,Φ_kept,traw,tnorm,_=residual_and_norm_select_hyp(solver,λs[i],Uks[i],Ys[i],complex(k0s[i],zero(T)),Rs[i],all_pts[i];res_tol=T(res_tol),matnorm=:one,epss=1e-15,auto_discard_spurious=auto_discard_spurious,collect_logs=false,multithreaded=multithreaded_matrix,h=h,P=P,mp_dps=mp_dps,leg_type=leg_type)
        ks_list[i]=real.(λs[i][idx2]);tens_list[i]=traw;tensN_list[i]=tnorm;phi_list[i]=Matrix(Φ_kept)
    end
    n_by_win=Vector{Int}(undef,nw);@inbounds for i in 1:nw;n_by_win[i]=size(phi_list[i],2);end
    offs=zeros(Int,nw);@inbounds for i in 2:nw;offs[i]=offs[i-1]+n_by_win[i-1];end
    ntot=offs[end]+n_by_win[end]
    ks_all=Vector{T}(undef,ntot);tens_all=Vector{T}(undef,ntot);tensN_all=Vector{T}(undef,ntot)
    us_all=Vector{Vector{Complex{T}}}(undef,ntot);pts_all=Vector{BoundaryPointsHypBIM{T}}(undef,ntot)
    Threads.@threads for i in 1:nw
        n=n_by_win[i];n==0 && continue
        off=offs[i];ksi=ks_list[i];tr=tens_list[i];tn=tensN_list[i];Φ=phi_list[i];pts=all_pts[i]
        @inbounds for j in 1:n
            ks_all[off+j]=ksi[j];tens_all[off+j]=tr[j];tensN_all[off+j]=tn[j];us_all[off+j]=vec(@view Φ[:,j]);pts_all[off+j]=pts
        end
    end
    return ks_all,tens_all,us_all,pts_all,tensN_all
end