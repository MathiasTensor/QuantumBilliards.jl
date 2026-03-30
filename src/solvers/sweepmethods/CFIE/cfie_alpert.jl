# Ref: HYBRID GAUSS-TRAPEZOIDAL QUADRATURE RULES, Alpert B., 1999
# This version is endpoint-aware, open-panel, and avoids storing O(jcorr*N^2) dense tensors.

struct AlpertComponentCache{T<:Real}
    us::Vector{T}       # interpolation nodes on [0,1]
    bw::Vector{T}       # barycentric weights for open-panel interpolation
    Le::Matrix{T}       # jcorr × N, left-end interpolation vectors
    Re::Matrix{T}       # jcorr × N, right-end interpolation vectors
    xe::Vector{T}       # left-end corrected source geometry
    ye::Vector{T}
    txe::Vector{T}
    tye::Vector{T}
    se::Vector{T}
    xr::Vector{T}       # right-end corrected source geometry
    yr::Vector{T}
    txr::Vector{T}
    tyr::Vector{T}
    sr::Vector{T}
end

# _panel_us
# Build the open-panel interpolation nodes on [0,1].
#
# Inputs:
#   - ::Type{T} :
#       Scalar type.
#   - N::Int :
#       Number of panel nodes.
#
# Outputs:
#   - us::Vector{T} :
#       Uniform nodes 0, 1/(N-1), ..., 1.
@inline function _panel_us(::Type{T},N::Int) where {T<:Real}
    return collect(range(zero(T),one(T),length=N))
end

# _barycentric_weights_open
# Compute barycentric interpolation weights for a given set of open-panel nodes.
#
# Inputs:
#   - us::AbstractVector{T} :
#       Interpolation nodes on [0,1].
#
# Outputs:
#   - bw::Vector{T} :
#       Barycentric weights.
function _barycentric_weights_open(us::AbstractVector{T}) where {T<:Real}
    N=length(us)
    bw=Vector{T}(undef,N)
    @inbounds for j in 1:N
        v=one(T)
        uj=us[j]
        for m in 1:N
            m==j && continue
            v*=uj-us[m]
        end
        bw[j]=inv(v)
    end
    return bw
end

# _interp_vector_open!
# Build the barycentric interpolation vector at u for the open panel [0,1].
#
# Inputs:
#   - ℓ::AbstractVector{T} :
#       Output interpolation vector.
#   - u::T :
#       Query parameter.
#   - us::AbstractVector{T} :
#       Panel nodes.
#   - bw::AbstractVector{T} :
#       Barycentric weights for `us`.
#
# Outputs:
#   - ℓ :
#       Interpolation coefficients such that f(u) ≈ sum_j ℓ[j] f(us[j]).
function _interp_vector_open!(ℓ::AbstractVector{T},u::T,us::AbstractVector{T},bw::AbstractVector{T}) where {T<:Real}
    fill!(ℓ,zero(T))
    hit=0
    den=zero(T)
    @inbounds for j in eachindex(us)
        δ=u-us[j]
        if abs(δ)<=64*eps(T)
            hit=j
            break
        end
        den+=bw[j]/δ
    end
    if hit!=0
        ℓ[hit]=one(T)
        return ℓ
    end
    @inbounds for j in eachindex(us)
        δ=u-us[j]
        ℓ[j]=(bw[j]/δ)/den
    end
    return ℓ
end

# _interp_geom_from_vec
# Evaluate point/tangent/speed geometry from an interpolation vector.
#
# Inputs:
#   - ℓ :
#       Interpolation coefficients.
#   - X,Y,dX,dY :
#       Panel geometry arrays.
#
# Outputs:
#   - x,y,tx,ty,s :
#       Interpolated point, tangent, and speed.
@inline function _interp_geom_from_vec(ℓ::AbstractVector{T},X::AbstractVector{T},Y::AbstractVector{T},dX::AbstractVector{T},dY::AbstractVector{T}) where {T<:Real}
    x=dot(ℓ,X)
    y=dot(ℓ,Y)
    tx=dot(ℓ,dX)
    ty=dot(ℓ,dY)
    s=sqrt(tx*tx+ty*ty)
    return x,y,tx,ty,s
end

# _build_alpert_component_cache
# Precompute only endpoint-special Alpert data for one smooth panel.
#
# Inputs:
#   - pts::BoundaryPointsCFIE{T} :
#       One smooth panel.
#   - rule::AlpertLogRule{T} :
#       Log-singular Alpert rule.
#
# Outputs:
#   - C::AlpertComponentCache{T} :
#       Lightweight cache storing interpolation metadata and endpoint rules.
#
# Notes:
#   - This intentionally does NOT store interior shifted interpolation vectors.
#   - Interior correction geometry is built on-the-fly inside `_assemble_self_alpert!`.
function _build_alpert_component_cache(pts::BoundaryPointsCFIE{T},rule::AlpertLogRule{T}) where {T<:Real}
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    N=length(X)
    us=_panel_us(T,N)
    bw=_barycentric_weights_open(us)
    h=inv(T(N-1))
    jcorr=rule.j
    ξ=rule.x
    Le=Matrix{T}(undef,jcorr,N)
    Re=Matrix{T}(undef,jcorr,N)
    xe=Vector{T}(undef,jcorr); ye=similar(xe)
    txe=similar(xe); tye=similar(xe); se=similar(xe)
    xr=Vector{T}(undef,jcorr); yr=similar(xr)
    txr=similar(xr); tyr=similar(xr); sr=similar(xr)
    ℓ=Vector{T}(undef,N)
    @inbounds for p in 1:jcorr
        ul=h*ξ[p]
        _interp_vector_open!(ℓ,ul,us,bw)
        @views copyto!(Le[p,:],ℓ)
        xe[p],ye[p],txe[p],tye[p],se[p]=_interp_geom_from_vec(ℓ,X,Y,dX,dY)
        ur=one(T)-h*ξ[p]
        _interp_vector_open!(ℓ,ur,us,bw)
        @views copyto!(Re[p,:],ℓ)
        xr[p],yr[p],txr[p],tyr[p],sr[p]=_interp_geom_from_vec(ℓ,X,Y,dX,dY)
    end

    return AlpertComponentCache(us,bw,Le,Re,xe,ye,txe,tye,se,xr,yr,txr,tyr,sr)
end

# _assemble_self_alpert!
# Assemble the self-panel CFIE block using:
#   - plain trapezoid for DLP off-diagonal
#   - plain trapezoid for far SLP
#   - endpoint-special Alpert corrections near the left/right ends
#   - on-the-fly interior Alpert corrections away from the ends
#
# Inputs:
#   - solver,A,pts,G,C,row_range,k,rule :
#       Standard self-block assembly data.
#   - multithreaded::Bool=true :
#       Whether to thread over target rows.
#
# Outputs:
#   - Modifies `A` in place.
#
# Notes:
#   - This assumes `pts` is ONE smooth panel.
#   - It is not yet a corner-aware or multi-segment panelwise implementation.
function _assemble_self_alpert!(solver::CFIE_alpert{T},
    A::AbstractMatrix{Complex{T}},
    pts::BoundaryPointsCFIE{T},
    G::CFIEGeomCache{T},
    C::AlpertComponentCache{T},
    row_range::UnitRange{Int},
    k::T,
    rule::AlpertLogRule{T};
    multithreaded::Bool=true) where {T<:Real}
    αD=Complex{T}(0,k/2)
    αS=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    h=pts.ws[1]
    wq=pts.ds
    N=length(X)
    a=rule.a
    jcorr=rule.j
    ξ=rule.x
    ω=rule.w
    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        xi=X[i]
        yi=Y[i]
        wi=wq[i]
        κi=G.kappa[i]
        ui=C.us[i]
        A[gi,gi]+=one(Complex{T})-Complex{T}(wi*κi,zero(T))
        @inbounds for j in 1:N
            j==i && continue
            gj=row_range[j]
            rij=G.R[i,j]
            inn=G.inner[i,j]
            invr=G.invR[i,j]
            A[gi,gj]-=wq[j]*(αD*inn*H(1,k*rij)*invr)
        end
        @inbounds for j in 1:N
            j==i && continue
            abs(j-i)<a && continue
            gj=row_range[j]
            A[gi,gj]-=ik*(wq[j]*(αS*H(0,k*G.R[i,j])*G.speed[j]))
        end
        ℓ=Vector{T}(undef,N)
        @inbounds for p in 1:jcorr
            fac=h*ω[p]
            up=ui+h*ξ[p]
            if up<=one(T)
                _interp_vector_open!(ℓ,up,C.us,C.bw)
                xp,yp,txp,typ,sp=_interp_geom_from_vec(ℓ,X,Y,dX,dY)
                dx=xi-xp
                dy=yi-yp
                r=sqrt(dx*dx+dy*dy)
                coeff=-ik*(fac*(αS*H(0,k*r)*sp))
                for q in 1:N
                    A[gi,row_range[q]]+=coeff*ℓ[q]
                end
            else
                dx=xi-C.xr[p]
                dy=yi-C.yr[p]
                r=sqrt(dx*dx+dy*dy)
                coeff=-ik*(fac*(αS*H(0,k*r)*C.sr[p]))
                @views Re_p=C.Re[p,:]
                for q in 1:N
                    A[gi,row_range[q]]+=coeff*Re_p[q]
                end
            end
            um=ui-h*ξ[p]
            if um>=zero(T)
                _interp_vector_open!(ℓ,um,C.us,C.bw)
                xm,ym,txm,tym,sm=_interp_geom_from_vec(ℓ,X,Y,dX,dY)
                dx=xi-xm
                dy=yi-ym
                r=sqrt(dx*dx+dy*dy)
                coeff=-ik*(fac*(αS*H(0,k*r)*sm))
                for q in 1:N
                    A[gi,row_range[q]]+=coeff*ℓ[q]
                end
            else
                dx=xi-C.xe[p]
                dy=yi-C.ye[p]
                r=sqrt(dx*dx+dy*dy)
                coeff=-ik*(fac*(αS*H(0,k*r)*C.se[p]))
                @views Le_p=C.Le[p,:]
                for q in 1:N
                    A[gi,row_range[q]]+=coeff*Le_p[q]
                end
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
        wj=pb.ds[j]
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
            wj=pb.ds[j]
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
                wj=pb.ds[j]
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