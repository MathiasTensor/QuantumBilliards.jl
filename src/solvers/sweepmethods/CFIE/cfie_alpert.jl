# Ref: HYBRID GAUSS-TRAPEZOIDAL QUADRATURE RULES, Alpert B., 1999, https://math.nist.gov/~BAlpert/ggauss.pdf
# NOTE: could not find table for 32th order...

struct AlpertComponentCache{T<:Real}
    Lp0::Matrix{T} # N × jcorr, base interpolation vectors for +ξ_p
    Lm0::Matrix{T} # N × jcorr, base interpolation vectors for -ξ_p
    xp::Matrix{T}  # jcorr × N
    yp::Matrix{T}
    txp::Matrix{T}
    typ::Matrix{T}
    sp::Matrix{T}
    xm::Matrix{T}
    ym::Matrix{T}
    txm::Matrix{T}
    tym::Matrix{T}
    sm::Matrix{T}
end

@inline function wrap_angle(t::T) where {T<:Real}
    tp=mod(t,two_pi)
    return tp==zero(T) ? T(two_pi) : tp
end

@inline function wrap_diff(t::T) where {T<:Real}
    d=mod(t+T(pi),two_pi)-T(pi)
    return d
end

function trig_cardinal_weights!(l::AbstractVector{T},θ::T,ts::AbstractVector{T}) where {T<:Real}
    N=length(ts)
    fill!(l,zero(T))
    for j in 1:N
        δ=wrap_diff(θ-ts[j])
        if abs(δ)<=64*eps(T)
            l[j]=one(T)
            return l
        end
    end
    invN=inv(T(N))
    @inbounds for j in 1:N
        δ=wrap_diff(θ-ts[j])
        s=sin(δ/2)
        c=cos(δ/2)
        l[j]=invN*sin(T(N)*δ/2)*(c/s) 
    end
    return l
end

@inline function _eval_shifted_source(base::AbstractVector{T},i::Int,X::AbstractVector{T},Y::AbstractVector{T},dX::AbstractVector{T},dY::AbstractVector{T}) where {T<:Real}
    N=length(base)
    x=zero(T);y=zero(T);tx=zero(T);ty=zero(T)
    @inbounds @simd for q in 1:N
        idx=mod1(q-i+1,N)
        w=base[idx]
        x+=w*X[q]
        y+=w*Y[q]
        tx+=w*dX[q]
        ty+=w*dY[q]
    end
    s=sqrt(tx*tx+ty*ty)
    return x,y,tx,ty,s
end

function _build_alpert_component_cache(pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    ts=pts.ts
    N=length(ts)
    h=pts.ws[1]
    jcorr=rule.j
    Lp0=Matrix{T}(undef,N,jcorr)
    Lm0=Matrix{T}(undef,N,jcorr)
    xp=Matrix{T}(undef,jcorr,N)
    yp=Matrix{T}(undef,jcorr,N)
    txp=Matrix{T}(undef,jcorr,N)
    typ=Matrix{T}(undef,jcorr,N)
    sp=Matrix{T}(undef,jcorr,N)
    xm=Matrix{T}(undef,jcorr,N)
    ym=Matrix{T}(undef,jcorr,N)
    txm=Matrix{T}(undef,jcorr,N)
    tym=Matrix{T}(undef,jcorr,N)
    sm=Matrix{T}(undef,jcorr,N)
    tmp=Vector{T}(undef,N)
    @inbounds for p in 1:jcorr
        θp=wrap_angle(ts[1]+h*rule.x[p])
        trig_cardinal_weights!(tmp,θp,ts)
        @views copyto!(Lp0[:,p],tmp)
        θm=wrap_angle(ts[1]-h*rule.x[p])
        trig_cardinal_weights!(tmp,θm,ts)
        @views copyto!(Lm0[:,p],tmp)
        @views bp=Lp0[:,p]
        @views bm=Lm0[:,p]
        for i in 1:N
            xp[p,i],yp[p,i],txp[p,i],typ[p,i],sp[p,i]=_eval_shifted_source(bp,i,X,Y,dX,dY)
            xm[p,i],ym[p,i],txm[p,i],tym[p,i],sm[p,i]=_eval_shifted_source(bm,i,X,Y,dX,dY)
        end
    end
    return AlpertComponentCache(Lp0,Lm0,xp,yp,txp,typ,sp,xm,ym,txm,tym,sm)
end

function _assemble_self_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},C::AlpertComponentCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    ws=pts.ws
    N=length(pts.ts)
    h=ws[1]
    a=rule.a
    jcorr=rule.j
    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        wi=ws[i]
        κi=G.kappa[i]
        A[gi,gi]+=one(Complex{T})-Complex{T}(wi*κi,zero(T))
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]-=ws[j]*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for j in 1:N
            j==i && continue
            m=j-i
            m>N÷2 && (m-=N)
            m<-N÷2 && (m+=N)
            abs(m)<a && continue
            gj=row_range[j]
            A[gi,gj]-=ik*(ws[j]*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        @inbounds for p in 1:jcorr # SLP near Alpert corrections
            fac=h*rule.w[p]
            dx=xi-C.xp[p,i]
            dy=yi-C.yp[p,i]
            r=sqrt(dx*dx+dy*dy)
            coeff=-ik*(fac*(αS*H(0,k*r)*C.sp[p,i]))
            @views basep=C.Lp0[:,p]
            for q in 1:N
                A[gi,row_range[q]]+=coeff*basep[mod1(q-i+1,N)]
            end
            dx=xi-C.xm[p,i]
            dy=yi-C.ym[p,i]
            r=sqrt(dx*dx+dy*dy)
            coeff=-ik*(fac*(αS*H(0,k*r)*C.sm[p,i]))
            @views basem=C.Lm0[:,p]
            for q in 1:N
                A[gi,row_range[q]]+=coeff*basem[mod1(q-i+1,N)]
            end
        end
    end

    return A
end

##############################
#### DESYMMETRIZED KERNEL ####
##############################

# _add_image_block!
# Add one smooth image contribution from source component `pb` into the
# target/source block (a,b) of the desymmetrized Alpert matrix.
#
# Inputs:
#   - A::AbstractMatrix{Complex{T}} :
#       Global system matrix being assembled.
#   - ra::UnitRange{Int} :
#       Global row range of the target component.
#   - rb::UnitRange{Int} :
#       Global column range of the source component.
#   - pa::BoundaryPointsCFIE{T} :
#       Target component points.
#   - pb::BoundaryPointsCFIE{T} :
#       Source component points.
#   - k::T :
#       Real wavenumber.
#   - qfun :
#       Function mapping a source point q -> image point qimg.
#   - tfun :
#       Function mapping a source tangent t -> image tangent timg.
#   - weight :
#       Scalar symmetry/image weight.
#   - multithreaded::Bool=true :
#       Whether to thread over source columns.
#
# Outputs:
#   - Modifies `A` in place by adding the smooth image contribution
#       -(D_img + i k S_img)
function _add_image_block!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,qfun,tfun,weight;multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    Na=length(pa.xy)
    Nb=length(pb.xy)
    Xa=getindex.(pa.xy,1)
    Ya=getindex.(pa.xy,2)
    @use_threads multithreading=multithreaded for j in 1:Nb
        gj=rb[j]
        qimg=qfun(pb.xy[j])
        timg=tfun(pb.tangent[j])
        xj=qimg[1]
        yj=qimg[2]
        txj=timg[1]
        tyj=timg[2]
        sj=sqrt(txj*txj+tyj*tyj)
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
            dval=weight*wj*(αD*inn*H(1,k*r)*invr)
            sval=weight*wj*(αS*H(0,k*r)*sj)
            A[gi,gj]-=(dval+ik*sval)
        end
    end
    return A
end

# _assemble_reflection_images!
# Add all reflection image contributions for one source component block.
#
# Inputs:
#   - A, ra, rb, pa, pb, solver, billiard, k :
#       Standard assembly data.
#   - sym::Reflection :
#       Reflection symmetry object.
#
# Outputs:
#   - Modifies `A` in place by adding the corresponding reflected image terms.
#
# Notes:
#   - For `:y_axis` and `:x_axis`, there is a single reflected image.
#   - For `:origin` (XYReflection), this expands into three image terms:
#       x-image, y-image, and xy-image,
#     with weights σx, σy, and σx*σy respectively.
function _assemble_reflection_images!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},solver::CFIE_alpert{T},billiard::Bi,k::T,sym::Reflection;multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    if sym.axis==:y_axis
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_x(q,billiard),t->image_tangent_x(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:x_axis
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_y(q,billiard),t->image_tangent_y(t),image_weight(sym);multithreaded=multithreaded)
    elseif sym.axis==:origin
        σx=image_weight_x(sym)
        σy=image_weight_y(sym)
        σxy=image_weight_xy(sym)
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_x(q,billiard),t->image_tangent_x(t),σx;multithreaded=multithreaded)
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_y(q,billiard),t->image_tangent_y(t),σy;multithreaded=multithreaded)
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point_xy(q,billiard),t->image_tangent_xy(t),σxy;multithreaded=multithreaded)
    else
        error("Unknown reflection axis $(sym.axis)")
    end
    return A
end


# _assemble_rotation_images!
# Add all nontrivial rotation images for one source component block.
#
# Inputs:
#   - A, ra, rb, pa, pb, k :
#       Standard assembly data.
#   - sym::Rotation :
#       Rotation symmetry descriptor.
#   - costab, sintab, χ :
#       Rotation tables from `_rotation_tables(T, sym.n, sym.m)`.
#
# Outputs:
#   - Modifies `A` in place by adding the smooth rotated image terms.
#
# Notes:
#   - Direct l=0 contribution is not included here; it is assembled separately
#   - Adds l=1,...,n-1 images only.
function _assemble_rotation_images!(A::AbstractMatrix{Complex{T}},ra::UnitRange{Int},rb::UnitRange{Int},pa::BoundaryPointsCFIE{T},pb::BoundaryPointsCFIE{T},k::T,sym::Rotation,costab,sintab,χ;multithreaded::Bool=true) where {T<:Real}
    for l in 1:(sym.n-1)
        phase=χ[l+1]
        _add_image_block!(A,ra,rb,pa,pb,k,q->image_point(sym,q,l,costab,sintab),t->image_tangent(sym,t,l,costab,sintab),phase;multithreaded=multithreaded)
    end
    return A
end


# construct_matrices_symmetry!
# Assemble the CFIE_Alpert matrix on the desymmetrized/fundamental boundary.
#
# Inputs:
#   - solver::CFIE_alpert{T} :
#       Alpert-based CFIE solver with nontrivial symmetry.
#   - A::Matrix{Complex{T}} :
#       Output system matrix.
#   - pts::Vector{BoundaryPointsCFIE{T}} :
#       Boundary points on the desymmetrized boundary only.
#   - k::T :
#       Real wavenumber.
#   - multithreaded::Bool=true :
#       Whether to use threaded assembly.
#
# Outputs:
#   - Modifies `A` in place to contain the desymmetrized operator matrix.
function construct_matrices_symmetry!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    symmetry=solver.symmetry
    isnothing(symmetry) && error("construct_matrices_symmetry! called with symmetry = nothing")
    offs=component_offsets(pts)
    fill!(A,zero(Complex{T}))
    rule=alpert_log_rule(T,solver.alpert_order)
    Gs=[cfie_geom_cache(p) for p in pts]
    Cs=[_build_alpert_component_cache(pts[a],rule) for a in eachindex(pts)]
    nc=length(pts)
    for a in 1:nc
        ra=offs[a]:(offs[a+1]-1)
        _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
    end
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
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
                dval=wj*(αD*inn*H(1,k*r)*invr)
                sval=wj*(αS*H(0,k*r)*sj)
                A[gi,gj]-=(dval+ik*sval)
            end
        end
    end
    # symmetry image contributions
    for sym in symmetry
        if sym isa Reflection
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_reflection_images!(A,ra,rb,pts[a],pts[b],solver,solver.billiard,k,sym;multithreaded=multithreaded)
            end
        elseif sym isa Rotation
            costab,sintab,χ=_rotation_tables(T,sym.n,sym.m)
            for a in 1:nc, b in 1:nc
                ra=offs[a]:(offs[a+1]-1)
                rb=offs[b]:(offs[b+1]-1)
                _assemble_rotation_images!(A,ra,rb,pts[a],pts[b],k,sym,costab,sintab,χ;multithreaded=multithreaded)
            end
        else
            error("Unknown symmetry type $(typeof(sym))")
        end
    end
    return A
end

################################
#### FULL MATRIX CONSTRUCTION ####
################################

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(solver.symmetry)
        offs=component_offsets(pts)
        αL2=Complex{T}(0,k/2)
        αM2=Complex{T}(0,one(T)/2)
        ik=Complex{T}(0,k)
        fill!(A,zero(Complex{T}))
        Gs=[cfie_geom_cache(p) for p in pts]
        rule=alpert_log_rule(T,solver.alpert_order)
        Cs=[_build_alpert_component_cache(pts[a],rule) for a in eachindex(pts)]
        nc=length(pts)
        for a in 1:nc
            ra=offs[a]:(offs[a+1]-1)
            _assemble_self_alpert!(solver,A,pts[a],Gs[a],Cs[a],ra,k,rule;multithreaded=multithreaded)
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
                    inn=tyj*dx-txj*dy
                    invr=inv(r)
                    dval=wj*(αL2*inn*H(1,k*r)*invr)
                    sval=wj*(αM2*H(0,k*r)*sj)
                    A[gi,gj]-=(dval+ik*sval)
                end
            end
        end
        return A
    else
        return construct_matrices_symmetry!(solver,A,pts,k;multithreaded=multithreaded)
    end
end

function construct_matrices(solver::CFIE_alpert,pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    Ntot=boundary_matrix_size(pts)
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    return A
end

############################
#### SOLVE WRAPPERS ########
############################

function solve(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    if use_krylov
        @blas_multi_then_1 MAX_BLAS_THREADS mu,_,_,_=svdsolve(A,1,:SR)
        return mu[1]
    else
        s=svdvals(A)
        return s[end]
    end
end

function solve_vect(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true) where {T<:Real,Ba<:AbsBasis}
    A=construct_matrices(solver,pts,k;multithreaded=multithreaded)
    _,S,Vt=LAPACK.gesvd!('A','A',A)
    idx=findmin(S)[2]
    return S[idx],real.(Vt[idx,:])
end

function solve_INFO(solver::CFIE_alpert,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true) where {T<:Real,Ba<:AbsBasis}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    A=Matrix{Complex{T}}(undef,Ntot,Ntot)
    t0=time()
    @info "Building boundary operator A..."
    construct_matrices!(solver,A,pts,k;multithreaded=multithreaded)
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA;sigdigits=4))"
    @info "Performing SVD..."
    t2=time()
    if use_krylov 
        @blas_multi_then_1 MAX_BLAS_THREADS s,_,_,_=svdsolve(A,1,:SR)
        reverse!(s)
    else
        s=svdvals(A)
    end
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s[end]
end