# Ref: HYBRID GAUSS-TRAPEZOIDAL QUADRATURE RULES, Alpert B., 1999, https://math.nist.gov/~BAlpert/ggauss.pdf
# NOTE: could not find table for 32th order...

#####################
#### CONSTRUCTOR ####
#####################

struct CFIE_alpert{T,Bi}<:CFIE where {T<:Real,Bi<:AbsBilliard}
    sampler::Vector{LinearNodes}
    pts_scaling_factor::Vector{T}
    dim_scaling_factor::T
    eps::T
    min_dim::Int64
    min_pts::Int64
    billiard::Bi
    symmetry::Union{Nothing,Vector{Any}}
    alpert_order::Int
end

function CFIE_alpert(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts::Int=20,eps::T=T(1e-15),symmetry::Union{Nothing,Vector{Any}}=nothing,alpert_order::Int=16) where {T<:Real,Bi<:AbsBilliard}
    !(alpert_order in (2,3,4,5,6,8,10,12,14,16)) && error("Alpert order not currently supported")
    _=alpert_log_rule(T,alpert_order)
    bs=pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return CFIE_alpert{T,Bi}(sampler,bs,bs[1],eps,min_pts,min_pts,billiard,symmetry,alpert_order)
end

struct AlpertLogRule{T<:Real}
    order::Int
    a::Int
    j::Int
    x::Vector{T}
    w::Vector{T}
end

@inline function alpert_log_rule(::Type{T},order::Int) where {T<:Real}
    if order==2
        return AlpertLogRule{T}(2,1,1,
            T[1.591549430918953e-01],
            T[5.0e-01])
    elseif order==3
        return AlpertLogRule{T}(3,2,2,
            T[1.150395811972836e-01,9.365464527949632e-01],
            T[3.913373788753340e-01,1.108662621124666e+00])
    elseif order==4
        return AlpertLogRule{T}(4,2,3,
            T[2.379647284118974e-02,2.935370741501914e-01,1.023715124251890e+00],
            T[8.795942675593887e-02,4.989017152913699e-01,9.131388579526912e-01])
    elseif order==5
        return AlpertLogRule{T}(5,3,4,
            T[2.339013027203800e-02,2.854764931311984e-01,1.005403327220700e+00,1.994970303994294e+00],
            T[8.609736556158105e-02,4.847019685417959e-01,9.152988869123725e-01,1.013901778984250e+00])
    elseif order==6
        return AlpertLogRule{T}(6,3,5,
            T[4.004884194926570e-03,7.745655373336686e-02,3.972849993523248e-01,1.075673352915104e+00,2.003796927111872e+00],
            T[1.671879691147102e-02,1.636958371447360e-01,4.981856569770637e-01,8.372266245578912e-01,9.841730844088381e-01])
    elseif order==8
        return AlpertLogRule{T}(8,5,7,
            T[6.531815708567918e-03,9.086744584657729e-02,3.967966533375878e-01,1.027856640525646e+00,1.945288592909266e+00,2.980147933889640e+00,3.998861349951123e+00],
            T[2.462194198995203e-02,1.701315866854178e-01,4.609256358650077e-01,7.947291148621894e-01,1.008710414337933e+00,1.036093649726216e+00,1.004787656533285e+00])
    elseif order==10
        return AlpertLogRule{T}(10,6,10,
            T[1.175089381227308e-03,1.877034129831289e-02,9.686468391426860e-02,3.004818668002884e-01,6.901331557173356e-01,1.293695738083659e+00,2.090187729798780e+00,3.016719313149212e+00,4.001369747872486e+00,5.000025661793423e+00],
            T[4.560746882084207e-03,3.810606322384757e-02,1.293864997289512e-01,2.884360381408835e-01,4.958111914344961e-01,7.077154600594529e-01,8.741924365285083e-01,9.661361986515218e-01,9.957887866078700e-01,9.998665787423845e-01])
    elseif order==12
        return AlpertLogRule{T}(12,7,11,
            T[1.674223682668368e-03,2.441110095009738e-02,1.153851297429517e-01,3.345898490480388e-01,7.329740531807683e-01,1.332305048525433e+00,2.114358752325948e+00,3.026084549655318e+00,4.003166301292590e+00,5.000141170055870e+00,6.000001002441859e+00],
            T[6.364190780720557e-03,4.723964143287529e-02,1.450891158385963e-01,3.021659470785897e-01,4.984270739715340e-01,6.971213795176096e-01,8.577295622757315e-01,9.544136554351155e-01,9.919938052776484e-01,9.994621875822987e-01,9.999934408092805e-01])
    elseif order==14
        return AlpertLogRule{T}(14,9,14,
            T[9.305182368545380e-04,1.373832458434617e-02,6.630752760779359e-02,1.979971397622003e-01,4.504313503816532e-01,8.571888631101634e-01,1.434505229617112e+00,2.175177834137754e+00,3.047955068386372e+00,4.004974906813428e+00,4.998525901820967e+00,5.999523015116678e+00,6.999963617883990e+00,7.999999488130134e+00],
            T[3.545060644780164e-03,2.681514031576498e-02,8.504092035093420e-02,1.854526216643691e-01,3.251724374883192e-01,4.911553747260108e-01,6.622933417369036e-01,8.137254578840510e-01,9.235595514944174e-01,9.821609923744658e-01,1.000047394596121e+00,1.000909336693954e+00,1.000119534283784e+00,1.000002835746089e+00])
    elseif order==16
        return AlpertLogRule{T}(16,10,15,
            T[8.371529832014113e-04,1.239382725542637e-02,6.009290785739468e-02,1.805991249601928e-01,4.142832599028031e-01,7.964747731112430e-01,1.348993882467059e+00,2.073471660264395e+00,2.947904939031494e+00,3.928129252248612e+00,4.957203086563112e+00,5.986360113977494e+00,6.997957704791519e+00,7.999888757524622e+00,8.999998754306120e+00],
            T[3.190919086626234e-03,2.423621380426338e-02,7.740135521653088e-02,1.704889420286369e-01,3.029123478511309e-01,4.652220834914617e-01,6.401489637096768e-01,8.051212946181061e-01,9.362411945698647e-01,1.014359775369075e+00,1.035167721053657e+00,1.020308624984610e+00,1.004798397441514e+00,1.000395017352309e+00,1.000007149422537e+00])
    else
        throw(ArgumentError("Unsupported log-Alpert order $order. Available: 2,3,4,5,6,8,10,12,14,16"))
    end
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

@inline function _eval_offgrid_source(l::AbstractVector{T},X::AbstractVector{T},Y::AbstractVector{T},dX::AbstractVector{T},dY::AbstractVector{T}) where {T<:Real}
    x=dot(l,X)
    y=dot(l,Y)
    tx=dot(l,dX)
    ty=dot(l,dY)
    s=sqrt(tx*tx+ty*ty)
    return x,y,tx,ty,s
end

########################################
#### ALPERT SAME-COMPONENT ASSEMBLY ####
########################################

function assemble_self_block_alpert!(solver::CFIE_alpert{T},A::AbstractMatrix{Complex{T}},pts::BoundaryPointsCFIE{T},G::CFIEGeomCache{T},row_range::UnitRange{Int},k::T,rule::AlpertLogRule{T};multithreaded::Bool=true) where {T<:Real}
    αL2=k/2*im
    αM2=Complex{T}(0,one(T)/2)
    αM1=-inv_two_pi
    ik=k*im
    X=getindex.(pts.xy,1)
    Y=getindex.(pts.xy,2)
    dX=getindex.(pts.tangent,1)
    dY=getindex.(pts.tangent,2)
    ts=pts.ts
    h=pts.ws[1]
    N=length(ts)
    a=rule.a
    jcorr=rule.j
    ξ=rule.x
    ω=rule.w
    @use_threads multithreading=multithreaded for i in 1:N
        gi=row_range[i]
        ti=ts[i]
        xi=X[i]
        yi=Y[i]
        si=G.speed[i]
        κi=G.kappa[i]
        ddiag=Complex{T}(h*κi,zero(T))
        sdiag=Complex{T}(zero(T),zero(T))
        sdiag+=Complex{T}(zero(T),zero(T))+h*((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*si^2))*si
        sdiag+=Complex{T}(2*π/(T(N)^2)*(αM1*si),zero(T)) # Kress diagonal log-weight equivalent
        A[gi,gi]+=one(Complex{T})-(ddiag+ik*sdiag)
        @inbounds for δ in a:(N-a)
            j=mod1(i+δ,N)
            gj=row_range[j]
            r=G.R[i,j]
            invr=G.invR[i,j]
            inn=G.inner[i,j]
            sj=G.speed[j]
            wj=pts.ws[j]
            dker=αL2*inn*H(1,k*r)*invr
            sker=αM2*H(0,k*r)*sj
            A[gi,gj]+= -(wj*dker+ik*(wj*sker))
        end
        l=Vector{T}(undef,N)
        @inbounds for p in 1:jcorr
            fac=h*ω[p]
            θp=wrap_angle(ti+h*ξ[p])
            trig_cardinal_weights!(l,θp,ts)
            xp,yp,txp,typ,sp=_eval_offgrid_source(l,X,Y,dX,dY)
            dx=xi-xp
            dy=yi-yp
            r=sqrt(dx*dx+dy*dy)
            inn=typ*dx-txp*dy
            dker=αL2*inn*H(1,k*r)/r
            sker=αM2*H(0,k*r)*sp
            coeff=-(fac*dker+ik*(fac*sker))
            @views A[gi,row_range].+=coeff.*l
            θm=wrap_angle(ti-h*ξ[p])
            trig_cardinal_weights!(l,θm,ts)
            xm,ym,txm,tym,sm=_eval_offgrid_source(l,X,Y,dX,dY)
            dx=xi-xm
            dy=yi-ym
            r=sqrt(dx*dx+dy*dy)
            inn=tym*dx-txm*dy
            dker=αL2*inn*H(1,k*r)/r
            sker=αM2*H(0,k*r)*sm
            coeff=-(fac*dker+ik*(fac*sker))
            @views A[gi,row_range].+=coeff.*l
        end
    end

    return A
end

################################
#### FULL MATRIX CONSTRUCTION ####
################################

function construct_matrices!(solver::CFIE_alpert{T},A::Matrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    αL2=k/2*im
    αM2=Complex{T}(0,one(T)/2)
    ik=k*im
    fill!(A,zero(Complex{T}))
    Gs=[cfie_geom_cache(p) for p in pts]
    rule=alpert_log_rule(T,solver.alpert_order)
    nc=length(pts)
    for a in 1:nc
        ra=offs[a]:(offs[a+1]-1)
        assemble_self_block_alpert!(solver,A,pts[a],Gs[a],ra,k,rule;multithreaded=multithreaded)
    end
    # cross-component blocks unchanged: smooth quadrature
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
                dval=wj*(αL2*inn*H(1,k*r)*invr)
                sval=wj*(αM2*H(0,k*r)*sj)
                A[gi,gj]+= -(dval+ik*sval)
            end
        end
    end
    return A
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