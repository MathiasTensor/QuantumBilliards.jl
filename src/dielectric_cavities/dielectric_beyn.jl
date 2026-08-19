#=
For the dielectric resonance problem A(k)x=0 with C mutually
disjoint interiors and one connected exterior,

A(k)=[diag_a χ_aS_aa(n_ak)   diag_a D_aa(n_ak)-I;
    χ_outS_ext(n_outk)   D_ext(n_outk)+I].

For a positively oriented contour Γ and probe matrix V∈C^{N×r},

A₀=(1/2πi)∮_Γ A(z)⁻¹V dz,
A₁=(1/2πi)∮_Γ zA(z)⁻¹V dz.

If A₀=UΣW* has numerical rank rk, the reduced Beyn matrix is B=U_r*A₁W_rΣ_r⁻¹.
If BY=YΛ, the eigenvalues Λ approximate the enclosed nonlinear resonances and Φ=U_rY
contains the corresponding active Wiersig boundary vectors.

#=
VERIFICATION & SPURIOUS ROOTS

Production Beyn uses a single nq-point contour quadrature. After
B=U_r'*A₁W_rΣ_r⁻¹ is diagonalized as BY=YΛ, each candidate is assigned the
effective retained singular value

    σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]^(-1/2).

Candidates with small σeff depend most strongly on weak retained directions of
the zeroth Beyn moment and are checked first with the original nonlinear
residual. Validation proceeds in increasing σeff and stops once
`validation_padding` consecutive checked candidates pass after the last failed
candidate. Unchecked enclosed candidates are retained.
The dyadic nq÷2 -> nq contour comparison is diagnostic only. It is performed by
the INFO routine to verify that the chosen production nq resolves the contour
integrals adequately; it is not part of the production spectrum calculation.
=#

SPECTRUM

To compute all resonances in
Ω={k∈C: re_min≤Re(k)≤re_max, im_min≤Im(k)≤im_max},
Ω is covered by overlapping superelliptic Beyn contours. For each contour Γ_c,
A₀^(c)=(1/2πi)∮_{Γ_c}A(z)⁻¹V_c dz,
A₁^(c)=(1/2πi)∮_{Γ_c}zA(z)⁻¹V_c dz,
followed by the SVD and reduced-eigenproblem construction.

- TODO: Deep regions with Im(k)<0 may be better divided into horizontal strips because outgoing Hankel functions grow rapidly in the lower half-plane.
  TODO: Structured/HSS factorization for large dense boundary matrices.
Reference:
W.-J. Beyn, Linear Algebra Appl. 436 (2012), 3839–3863.
=#

"""
    WiersigContour

Smooth positively oriented Beyn contour with parametrization `z(θ)` and derivative
`dz(θ)` for `θ∈[0,2π)`. For exponential convergence of the periodic trapezoidal
rule, the parametrization should admit an analytic continuation to a sufficiently
wide strip about the real θ axis.

Built-in circle, ellipse and Fourier rounded-rectangle contours use entire
parametrizations. Custom contours must provide `z`, `dz`, and an `inside`
predicate.
"""
struct WiersigContour{T<:Real,F,G,H}
    center::Complex{T}
    halfwidth::T
    halfheight::T
    z::F
    dz::G
    inside::H
    function WiersigContour(center::Complex{T},halfwidth::T,halfheight::T,z::F,dz::G,inside::H) where {T<:Real,F,G,H}
        halfwidth>zero(T)||throw(ArgumentError("halfwidth must be positive"))
        halfheight>zero(T)||throw(ArgumentError("halfheight must be positive"))
        return new{T,F,G,H}(center,halfwidth,halfheight,z,dz,inside)
    end
end

"""
    WiersigContour(center,radius)

Construct the entire circular contour z(θ)=center+radius*exp(iθ).
"""
function WiersigContour(center::Complex{T},radius::T) where {T<:Real}
    radius>zero(T)||throw(ArgumentError("radius must be positive"))
    return WiersigContour(center,radius,radius)
end

"""
    WiersigContour(center,halfwidth,halfheight)

Construct the entire elliptical contour z(θ)=center+halfwidth*cos(θ)+i*halfheight*sin(θ).
"""
function WiersigContour(center::Complex{T},halfwidth::T,halfheight::T) where {T<:Real}
    halfwidth>zero(T)||throw(ArgumentError("halfwidth must be positive"))
    halfheight>zero(T)||throw(ArgumentError("halfheight must be positive"))
    z=θ->begin
        s,c=sincos(θ)
        center+Complex{T}(halfwidth*c,halfheight*s)
    end
    dz=θ->begin
        s,c=sincos(θ)
        Complex{T}(-halfwidth*s,halfheight*c)
    end
    inside=k->begin
        ξ=real(k-center)/halfwidth
        η=imag(k-center)/halfheight
        ξ^2+η^2<=one(T)
    end
    return WiersigContour{T,typeof(z),typeof(dz),typeof(inside)}(center,halfwidth,halfheight,z,dz,inside)
end

"""
    wiersig_rectangle_contour(center,halfwidth,halfheight;eta3=0.12,eta5=(9eta3-1)/25)

Construct the entire Fourier rounded rectangle z(θ)=center+sx(cosθ-η₃cos3θ+η₅cos5θ)+i sy(sinθ+η₃sin3θ+η₅sin5θ),
with axis extrema `halfwidth` and `halfheight`. Default kwargs were found by chatGPT and they seem quite good so I 
would not really change them much.

The default η₃=0.12, η₅=(9η₃-1)/25=0.0032 satisfies `-1+9η₃-25η₅=0`, flattening the contour near the horizontal and
vertical sides while retaining an entire periodic parametrization. Testing shows same trapezodial convergence with `nq` as the circle contour
"""
function wiersig_rectangle_contour(center::Complex{T},halfwidth::T,halfheight::T;eta3::T=T(0.12),eta5::T=(T(9)*eta3-one(T))/T(25)) where {T<:Real}
    halfwidth>zero(T)||throw(ArgumentError("halfwidth must be positive"))
    halfheight>zero(T)||throw(ArgumentError("halfheight must be positive"))
    scale=one(T)-eta3+eta5
    scale>zero(T)||throw(ArgumentError("invalid Fourier rectangle coefficients"))
    sx=halfwidth/scale
    sy=halfheight/scale
    z=θ->begin
        s1,c1=sincos(θ)
        s3,c3=sincos(T(3)*θ)
        s5,c5=sincos(T(5)*θ)
        center+Complex{T}(sx*(c1-eta3*c3+eta5*c5),sy*(s1+eta3*s3+eta5*s5))
    end
    dz=θ->begin
        s1,c1=sincos(θ)
        s3,c3=sincos(T(3)*θ)
        s5,c5=sincos(T(5)*θ)
        Complex{T}(sx*(-s1+T(3)*eta3*s3-T(5)*eta5*s5),sy*(c1+T(3)*eta3*c3+T(5)*eta5*c5))
    end
    inside=k->begin
        x=abs(real(k-center))
        y=abs(imag(k-center))
        x>halfwidth&&return false
        y>halfheight&&return false
        iszero(y)&&return true
        y==halfheight&&return iszero(x)
        lo=zero(T)
        hi=T(pi)/T(2)
        @inbounds for _ in 1:precision(T) # due to interval halving 2^-53 gives eps in Float64 usually
            θ=(lo+hi)/T(2)
            s1=sin(θ)
            s3=sin(T(3)*θ)
            s5=sin(T(5)*θ)
            yc=sy*(s1+eta3*s3+eta5*s5)
            if yc<y
                lo=θ
            else
                hi=θ
            end
        end
        θ=(lo+hi)/T(2)
        c1=cos(θ)
        c3=cos(T(3)*θ)
        c5=cos(T(5)*θ)
        xb=sx*(c1-eta3*c3+eta5*c5)
        x<=xb
    end
    return WiersigContour(center,halfwidth,halfheight,z,dz,inside)
end

"""
    wiersig_beyn_contour(contour,nq)

Return periodic trapezoidal nodes `z_j=z(θ_j)` and Beyn weights w_j=z'(θ_j)/(i nq), with `θ_j=2π(j-1)/nq`.
"""
function wiersig_beyn_contour(contour::WiersigContour{T},nq::Int) where {T<:Real}
    nq>0||throw(ArgumentError("nq must be positive"))
    Δθ=T(2π)/T(nq)
    z=Vector{Complex{T}}(undef,nq)
    w=similar(z)
    @inbounds for j in 1:nq
        θ=Δθ*T(j-1)
        zj=Complex{T}(contour.z(θ))
        dzj=Complex{T}(contour.dz(θ))
        z[j]=zj
        w[j]=dzj/Complex{T}(0,T(nq))
    end
    return z,w
end

"""
    wiersig_inside_contour(contour,k;tol=nothing)

Test whether `k` lies inside `contour`. The optional tolerance is implemented by
a small radial contraction toward the contour center before applying the stored
membership predicate.
"""
@inline function wiersig_inside_contour(contour::WiersigContour{T},k::Complex{T};tol=nothing) where {T<:Real}
    isnothing(tol)&&return contour.inside(k)
    tolT=T(tol)
    tolT>=zero(T)||throw(ArgumentError("tol must be nonnegative"))
    contour.inside(k)&&return true
    δ=k-contour.center
    iszero(δ)&&return true
    scale=max(contour.halfwidth,contour.halfheight)
    return contour.inside(contour.center+δ*(one(T)-tolT/max(one(T),scale)))
end

"""
    iersig_contour_tessellation(re_min::T,re_max::T,im_min::T,im_max::T,seed::WiersigContour{T};overlap_re::T=zero(T),overlap_im::T=zero(T)) where {T<:Real}

Cover the requested spectral rectangle by translated copies of the seed Beyn
contour. The seed completely determines the contour geometry, including its
parametrization, derivative, bounding half-sizes, and interior predicate.

The requested overlaps determine the maximum nominal center spacing. The actual
numbers of rows and columns are chosen automatically so that the corner of every
lattice cell lies inside the seed contour.
"""
function wiersig_contour_tessellation(re_min::T,re_max::T,im_min::T,im_max::T,seed::WiersigContour{T};overlap_re::T=zero(T),overlap_im::T=zero(T)) where {T<:Real}
    0<=overlap_re<1||throw(ArgumentError("overlap_re must satisfy 0≤overlap_re<1"))
    0<=overlap_im<1||throw(ArgumentError("overlap_im must satisfy 0≤overlap_im<1"))
    W=re_max-re_min
    H=im_max-im_min
    dxmax=T(2)*seed.halfwidth*(one(T)-overlap_re)
    dymax=T(2)*seed.halfheight*(one(T)-overlap_im)
    nx0=max(1,ceil(Int,W/dxmax))
    ny0=max(1,ceil(Int,H/dymax))
    best=typemax(Int)
    bestnx=0
    bestny=0
    nx=nx0
    while bestnx==0||nx*ny0<best
        dx=W/T(nx)
        if seed.inside(seed.center+Complex{T}(dx/T(2),zero(T)))
            ny=ny0
            while true
                dy=H/T(ny)
                if seed.inside(seed.center+Complex{T}(dx/T(2),dy/T(2)))
                    n=nx*ny
                    if n<best
                        best=n
                        bestnx=nx
                        bestny=ny
                    end
                    break
                end
                ny+=1
            end
        end
        nx+=1
    end
    bestnx>0||throw(ArgumentError("could not construct a covering contour tessellation"))
    dx=W/T(bestnx)
    dy=H/T(bestny)
    xs=T[re_min+(T(j)-T(0.5))*dx for j in 1:bestnx]
    ys=T[im_min+(T(j)-T(0.5))*dy for j in 1:bestny]
    return map(Iterators.product(xs,ys)) do (x,y) 
        center=Complex{T}(x,y)
        z=θ->center+(seed.z(θ)-seed.center)
        dz=seed.dz
        inside=k->seed.inside(seed.center+(k-center))
        WiersigContour(center,seed.halfwidth,seed.halfheight,z,dz,inside)
    end |> vec
end

"""
    wiersig_beyn_buffers(::Type{T},N::Int,r::Int,rng::AbstractRNG) where {T<:Real}

Allocate `V,X,A₀,A₁∈C^{N×r}`. At contour node `z_j`, `X=A(z_j)⁻¹V`, `A₀←A₀+w_jX`, and `A₁←A₁+w_jz_jX`.
"""
function wiersig_beyn_buffers(::Type{T},N::Int,r::Int,rng::AbstractRNG) where {T<:Real}
    V=randn(rng,Complex{T},N,r);X=similar(V);A0=zeros(Complex{T},N,r);A1=zeros(Complex{T},N,r)
    return V,X,A0,A1
end

"""
    _wiersig_beyn_effective_sigma(Y,Σ)

Return the effective retained moment singular value associated with each reduced
Beyn eigenvector. For `Y[:,j]`, σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]⁻¹ᐟ².
Small `σeff` means that the candidate depends strongly on weak retained
directions of the zeroth Beyn moment and is therefore checked first.
"""
function _wiersig_beyn_effective_sigma(Y::AbstractMatrix{Complex{T}},Σ::AbstractVector{T}) where {T<:Real}
    rk,n=size(Y)
    length(Σ)>=rk||throw(DimensionMismatch("need at least $rk singular values; received $(length(Σ))"))
    out=Vector{T}(undef,n)
    @inbounds for j in 1:n
        a=zero(T);b=zero(T)
        for l in 1:rk
            y=abs2(Y[l,j])
            a+=y
            b+=y/(Σ[l]^2)
        end
        out[j]=iszero(a)||iszero(b) ? T(Inf) : sqrt(a/b)
    end
    return out
end

"""
    _wiersig_beyn_singular_validation!(validator,inside,σeff,checked,keep;validation_padding=5)

Check enclosed candidates in increasing `σeff`. Stop once
`validation_padding` consecutive checked candidates are good after the most
recent failure. `validator(idx)` must evaluate and update `checked` and `keep`
for the supplied candidate indices.
"""
function _wiersig_beyn_singular_validation!(validator,inside::BitVector,σeff::AbstractVector{T},checked::BitVector,keep::BitVector;validation_padding::Int=5) where {T<:Real}
    validation_padding>0||throw(ArgumentError("validation_padding must be positive"))
    order=findall(inside)
    sort!(order;by=j->σeff[j])
    isempty(order)&&return order
    ncheck=min(length(order),validation_padding);checked_upto=0
    while checked_upto<ncheck
        validator(Vector{Int}(@view order[checked_upto+1:ncheck]))
        checked_upto=ncheck
        lastbad=0
        @inbounds for p in 1:checked_upto
            j=order[p]
            checked[j]&&!keep[j]&&(lastbad=p)
        end
        needed=lastbad==0 ? checked_upto : min(length(order),lastbad+validation_padding)
        needed<=checked_upto&&break
        ncheck=needed
    end
    return order
end

"""
    _wiersig_beyn_rank(Σ::AbstractVector{T},svd_tol::T,relative_svd_tol::Bool) where {T<:Real}

Determine the numerical rank of A₀. Relative mode retains `σ_j≥svd_tol*σ₁`; absolute mode retains `σ_j≥svd_tol`.
"""
@inline function _wiersig_beyn_rank(Σ::AbstractVector{T},svd_tol::T,relative_svd_tol::Bool) where {T<:Real}
    isempty(Σ)&&return 0,svd_tol
    threshold=relative_svd_tol ? svd_tol*Σ[1] : svd_tol
    return count(σ->σ>=threshold,Σ),threshold
end

"""
    _wiersig_beyn_build_reduced_problem(A0::Matrix{Complex{T}},A1::Matrix{Complex{T}};r::Int,r_step::Int,max_r::Int,svd_tol::Union{T,AbstractVector{T}},relative_svd_tol::Bool,verbose::Bool=false) where {T<:Real}

Build the finite-dimensional Beyn problems for one or more SVD tolerances from
already accumulated contour moments. The SVD is computed once. If several
tolerances are supplied they must be nonincreasing; the reduced matrix is formed
once at the largest resulting rank. The problem for any earlier tolerance is
the corresponding leading principal block of this matrix.
"""
function _wiersig_beyn_build_reduced_problem(A0::Matrix{Complex{T}},A1::Matrix{Complex{T}};r::Int,r_step::Int,max_r::Int,svd_tol::Union{T,AbstractVector{T}},relative_svd_tol::Bool,verbose::Bool=false) where {T<:Real}
    tols=svd_tol isa AbstractVector ? svd_tol : T[svd_tol]
    isempty(tols)&&throw(ArgumentError("svd_tol must not be empty"))
    issorted(tols;rev=true)||throw(ArgumentError("svd_tol must be nonincreasing"))
    N,ravailable=size(A0)
    rmax=min(max_r,ravailable);rcur=r
    while true
        A0cur=Matrix(@view A0[:,1:rcur])
        @blas_multi_then_1 MAX_BLAS_THREADS F0=svd!(A0cur;full=false)
        Σ=F0.S
        ranks=Vector{Int}(undef,length(tols));thresholds=Vector{T}(undef,length(tols))
        @inbounds for i in eachindex(tols)
            ranks[i],thresholds[i]=_wiersig_beyn_rank(Σ,tols[i],relative_svd_tol)
        end
        rkmax=maximum(ranks)
        if verbose
            println("Beyn probe dimension         = ",rcur)
            println("Beyn moment singular values = ");println(Σ)
            println("SVD tolerances              = ",tols)
            println("detected moment ranks       = ",ranks)
            println("rank thresholds             = ",thresholds)
        end
        if rkmax<rcur
            rkmax==0&&return (B=Matrix{Complex{T}}(undef,0,0),U=Matrix{Complex{T}}(undef,N,0),singular_values=copy(Σ),rank=0,ranks=ranks,rank_threshold=thresholds[1],rank_thresholds=thresholds,svd_tolerances=collect(tols),probe_dimension=rcur)
            Uk=@view F0.U[:,1:rkmax];Wk=@view F0.V[:,1:rkmax];Σk=@view Σ[1:rkmax];A1cur=@view A1[:,1:rcur]
            tmp=Matrix{Complex{T}}(undef,N,rkmax)
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1cur,Wk)
            @inbounds for j in 1:rkmax
                @views rmul!(tmp[:,j],inv(Σk[j]))
            end
            B=Matrix{Complex{T}}(undef,rkmax,rkmax)
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
            return (B=B,U=Matrix(Uk),singular_values=copy(Σ),rank=ranks[1],ranks=ranks,rank_threshold=thresholds[1],rank_thresholds=thresholds,svd_tolerances=collect(tols),probe_dimension=rcur)
        end
        rcur>=rmax&&throw(ArgumentError("Beyn moment rank remains saturated at max_r=$rmax. Increase max_r and normally nq, or reduce the contour size."))
        rcur=min(rcur+r_step,rmax)
    end
end

"""
    _wiersig_beyn_build_nested_direct(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Real=1e-12,relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}

Accumulate direct Beyn moments using one `V∈C^{N×rmax}` generated before the contour pass. At each `z_j`, assemble `A(z_j)`, factor it once, solve all probe columns simultaneously, then update `A₀+=w_jX_j` and `A₁+=w_jz_jX_j`.
"""
function _wiersig_beyn_build_nested_direct(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::T=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    iseven(length(z))||throw(ArgumentError("nested Beyn refinement requires even nq"))
    N=boundary_matrix_size(ws);rmax=min(max_r,N)
    V,X,A0,A1=wiersig_beyn_buffers(T,N,rmax,rng)
    C0=zeros(Complex{T},N,rmax);C1=zeros(Complex{T},N,rmax)
    A=Matrix{Complex{T}}(undef,N,N)
    xv=vec(X);a0v=vec(A0);a1v=vec(A1);c0v=vec(C0);c1v=vec(C1)
    p=verbose ? Progress(length(z),desc="Beyn contour") : nothing
    @inbounds for j in eachindex(z)
        construct_matrices!(solver,A,pts,ws,z[j];dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        F=lu!(A,ws;check=false)
        ldiv!(X,F,V)
        BLAS.axpy!(w[j],xv,a0v);BLAS.axpy!(w[j]*z[j],xv,a1v)
        if isodd(j)
            wc=T(2)*w[j]
            BLAS.axpy!(wc,xv,c0v);BLAS.axpy!(wc*z[j],xv,c1v)
        end
        verbose && next!(p)
    end
    fine=_wiersig_beyn_build_reduced_problem(A0,A1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
    coarse=_wiersig_beyn_build_reduced_problem(C0,C1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
    return (fine=fine,coarse=coarse)
end

"""
    _wiersig_beyn_build_direct(...)

Accumulate the production Beyn moments using only the requested `nq` contour
rule. Dyadic `nq÷2 -> nq` refinement is reserved for `wiersig_beyn_INFO`.
"""
function _wiersig_beyn_build_direct(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    N=boundary_matrix_size(ws);rmax=min(max_r,N)
    V,X,A0,A1=wiersig_beyn_buffers(T,N,rmax,rng)
    A=Matrix{Complex{T}}(undef,N,N)
    xv=vec(X);a0v=vec(A0);a1v=vec(A1)
    p=verbose ? Progress(length(z),desc="Beyn contour") : nothing
    @inbounds for j in eachindex(z)
        construct_matrices!(solver,A,pts,ws,z[j];dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        F=lu!(A,ws;check=false)
        ldiv!(X,F,V)
        BLAS.axpy!(w[j],xv,a0v)
        BLAS.axpy!(w[j]*z[j],xv,a1v)
        verbose&&next!(p)
    end
    return _wiersig_beyn_build_reduced_problem(A0,A1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
end

"""
    _wiersig_beyn_matrix_residual!(y,A,x;matnorm=:one)

Return `raw=||Ax||₂` and `normalized=||Ax||/(||A|| ||x||)`.
"""
function _wiersig_beyn_matrix_residual!(y::Vector{Complex{T}},A::Matrix{Complex{T}},x::AbstractVector{Complex{T}};matnorm::Symbol=:one) where {T<:Real}
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(y,A,x)
    if matnorm===:one
        nA=opnorm(A,1);nx=norm(x,1);ny=norm(y,1)
    elseif matnorm===:two
        nA=opnorm(A,2);nx=norm(x);ny=norm(y)
    elseif matnorm===:inf
        nA=opnorm(A,Inf);nx=norm(x,Inf);ny=norm(y,Inf)
    else
        throw(ArgumentError("matnorm must be :one, :two, or :inf"))
    end
    return norm(y),ny/(nA*nx)
end

"""
    _wiersig_beyn_residual!(...)

Validate `(k,x)` obtained from Beyn with the original direct Wiersig matrix to check if it is a spurious solution. Returns `raw=||A(k)x||₂` and `normalized=||A(k)x||/(||A(k)|| ||x||)` using the norm family selected by `matnorm`.
"""
function _wiersig_beyn_residual!(A::Matrix{Complex{T}},y::Vector{Complex{T}},solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,k::Complex{T},x::AbstractVector{Complex{T}};dlp_kernel::Symbol=:source,matnorm::Symbol=:one,multithreaded::Bool=true) where {T<:Real}
    construct_matrices!(solver,A,pts,ws,k;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
    return _wiersig_beyn_matrix_residual!(y,A,x;matnorm=matnorm)
end

function _wiersig_beyn_validate_direct!(raw::Vector{T},normalized::Vector{T},checked::BitVector,keep::BitVector,idx::Vector{Int},λ::Vector{Complex{T}},Φ::Matrix{Complex{T}},solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace;res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    isempty(idx)&&return nothing
    N=boundary_matrix_size(ws);A=Matrix{Complex{T}}(undef,N,N);y=Vector{Complex{T}}(undef,N)
    @inbounds for j in idx
        raw[j],normalized[j]=_wiersig_beyn_residual!(A,y,solver,pts,ws,λ[j],@view(Φ[:,j]);dlp_kernel=dlp_kernel,matnorm=matnorm,multithreaded=multithreaded)
        checked[j]=true
        keep[j]=(!filter_raw_residual||raw[j]<res_tol)&&normalized[j]<normalized_res_tol
        verbose&&println("adaptive candidate: k=",λ[j],", raw=",raw[j],", normalized=",normalized[j],", kept=",keep[j])
    end
    return nothing
end

"""
    construct_wiersig_B_matrix(...)

Construct the direct reduced Beyn matrix `B=U_r*A₁W_rΣ_r⁻¹` together with the retained invariant-subspace basis, moment singular values, detected rank, final probe width and contour quadrature data.
"""
function construct_wiersig_B_matrix(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::WiersigContour{T};nq::Int=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    z,w=wiersig_beyn_contour(contour,nq)
    reduced=_wiersig_beyn_build_direct(solver,pts,ws,z,w;r=r,r_step=r_step,max_r=max_r,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose)
    return merge(reduced,(contour=contour,contour_nodes=z,contour_weights=w))
end

"""
    wiersig_beyn_residual(solver,pts,ws,k,x;...)

Allocating version of `_wiersig_beyn_residual!`. Return the direct raw and normalized nonlinear residuals of `(k,x)`.
"""
function wiersig_beyn_residual(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,k,x::AbstractVector;dlp_kernel::Symbol=:source,matnorm::Symbol=:one,multithreaded::Bool=true) where {T<:Real}
    N=boundary_matrix_size(ws)
    A=Matrix{Complex{T}}(undef,N,N)
    y=Vector{Complex{T}}(undef,N)
    return _wiersig_beyn_residual!(A,y,solver,pts,ws,k,x;dlp_kernel=dlp_kernel,matnorm=matnorm,multithreaded=multithreaded)
end

"""
    _wiersig_beyn_candidates(...)

If `BY=YΛ`, lift reduced eigenvectors as `Φ=U_rY`. Candidates outside the
contour are rejected immediately. If `validate_roots=true`, enclosed candidates
are additionally tested with the original direct nonlinear operator `A(k)`.
If `validate_roots=false`, all enclosed finite Beyn roots are accepted without
additional matrix construction.
"""
function _wiersig_beyn_candidates(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,reduced,contour::WiersigContour{T}; validate_roots::Bool=true,res_tol::Real=1e-9,normalized_res_tol::Real=1e-10,filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    N=boundary_matrix_size(ws)
    if reduced.rank==0
        return (values=Complex{T}[],vectors=Matrix{Complex{T}}(undef,N,0),residuals=T[],normalized_residuals=T[],all_values=Complex{T}[],all_vectors=Matrix{Complex{T}}(undef,N,0),all_residuals=T[],all_normalized_residuals=T[],inside=Bool[],kept=Bool[])
    end
    rk=reduced.rank
    @blas_multi_then_1 MAX_BLAS_THREADS EB=eigen(Matrix(@view reduced.B[1:rk,1:rk]))
    λ=Vector{Complex{T}}(EB.values);Y=Matrix{Complex{T}}(EB.vectors)
    Φ=Matrix{Complex{T}}(undef,size(reduced.U,1),size(Y,2))
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φ,@view(reduced.U[:,1:rk]),Y)
    nroots=length(λ);inside=falses(nroots);keep=falses(nroots)
    raw_residuals=fill(T(NaN),nroots);normalized_residuals=fill(T(NaN),nroots)
    @inbounds for j in eachindex(λ)
        inside[j]=isfinite(real(λ[j]))&&isfinite(imag(λ[j]))&&wiersig_inside_contour(contour,λ[j])
    end
    if validate_roots
        Awork=Matrix{Complex{T}}(undef,N,N);ywork=Vector{Complex{T}}(undef,N)
        @inbounds for j in eachindex(λ)
            inside[j]||continue
            raw,nr=_wiersig_beyn_residual!(Awork,ywork,solver,pts,ws,λ[j],@view(Φ[:,j]);dlp_kernel=dlp_kernel,matnorm=matnorm,multithreaded=multithreaded)
            raw_residuals[j]=raw
            normalized_residuals[j]=nr
            keep[j]=(!filter_raw_residual||raw<T(res_tol))&&nr<T(normalized_res_tol)
            verbose&&println("candidate ",j,": k=",λ[j],", inside=true, raw=",raw,", normalized=",nr,", kept=",keep[j])
        end
    else
        keep.=inside # no validation, keep everything inside. If nq is enough this should be ok
        verbose&&println("Direct root validation disabled; accepting ",count(keep)," enclosed Beyn roots.")
    end
    idx=findall(keep);!isempty(idx)&&(idx=idx[sortperm(idx;by=j->(real(λ[j]),imag(λ[j])))])
    return (values=λ[idx],vectors=Φ[:,idx],residuals=raw_residuals[idx],normalized_residuals=normalized_residuals[idx],all_values=λ,all_vectors=Φ,all_residuals=raw_residuals,all_normalized_residuals=normalized_residuals,inside=inside,kept=keep)
end

"""
    wiersig_beyn(...)

Direct Beyn solve using a single `nq`-point production contour quadrature.

After `BY=YΛ`, candidates are assigned effective retained moment singular values

    σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]⁻¹ᐟ².

With `adaptive_validation=true`, enclosed candidates are checked with the direct
Wiersig residual in increasing `σeff`. Checking stops once
`validation_padding` consecutive candidates pass after the most recent failure.
Unchecked enclosed candidates are retained.

`validate_roots=true` validates every enclosed candidate. Setting both
`validate_roots=false` and `adaptive_validation=false` performs no residual
checks. Dyadic contour refinement is diagnostic only and is not part of this
production solve.
"""
function wiersig_beyn(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::WiersigContour{T};nq::Int=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,validate_roots::Bool=false,adaptive_validation::Bool=true,validation_padding::Int=5,res_tol::T=T(1e-8),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    reduced=construct_wiersig_B_matrix(solver,pts,ws,contour;nq=nq,r=r,r_step=r_step,max_r=max_r,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose)
    N=boundary_matrix_size(ws)
    if reduced.rank==0
        empty=(values=Complex{T}[],vectors=Matrix{Complex{T}}(undef,N,0),residuals=T[],normalized_residuals=T[],effective_singular_values=T[],checked=Bool[],all_values=Complex{T}[],all_vectors=Matrix{Complex{T}}(undef,N,0),all_residuals=T[],all_normalized_residuals=T[],all_effective_singular_values=T[],all_checked=Bool[],inside=Bool[],kept=Bool[])
        common=(rank=0,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,rank_threshold=reduced.rank_threshold,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=:none)
        return merge(empty,common)
    end
    rk=reduced.rank
    Ef=nothing
    @blas_multi_then_1 MAX_BLAS_THREADS Ef=eigen(Matrix(@view reduced.B[1:rk,1:rk]))
    λ=Vector{Complex{T}}(Ef.values);Y=Matrix{Complex{T}}(Ef.vectors)
    Φ=Matrix{Complex{T}}(undef,N,length(λ))
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φ,@view(reduced.U[:,1:rk]),Y)
    σeff=_wiersig_beyn_effective_sigma(Y,@view reduced.singular_values[1:reduced.rank])
    nroots=length(λ);inside=falses(nroots);keep=falses(nroots);checked=falses(nroots)
    raw=fill(T(NaN),nroots);normalized=fill(T(NaN),nroots)
    @inbounds for j in eachindex(λ)
        inside[j]=isfinite(real(λ[j]))&&isfinite(imag(λ[j]))&&wiersig_inside_contour(contour,λ[j])
        keep[j]=inside[j]
    end
    inside_idx=findall(inside)
    if validate_roots
        _wiersig_beyn_validate_direct!(raw,normalized,checked,keep,inside_idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
    elseif adaptive_validation&&!isempty(inside_idx)
        validator=idx->_wiersig_beyn_validate_direct!(raw,normalized,checked,keep,idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
        order=_wiersig_beyn_singular_validation!(validator,inside,σeff,checked,keep;validation_padding=validation_padding)
        verbose&&println("singular-support validation: checked=",count(checked),", rejected=",count(inside .& .!keep),", padding=",validation_padding,", σeff first/last=",isempty(order) ? T(NaN) : σeff[first(order)]," / ",isempty(order) ? T(NaN) : σeff[last(order)])
    end
    if length(reduced.ranks)>1
        @inbounds for it in 2:length(reduced.ranks)
            rkq=reduced.ranks[it]
            rkq==reduced.ranks[it-1]&&continue
            Eq=nothing
            @blas_multi_then_1 MAX_BLAS_THREADS Eq=eigen(Matrix(@view reduced.B[1:rkq,1:rkq]))
            λq=Vector{Complex{T}}(Eq.values);Yq=Matrix{Complex{T}}(Eq.vectors)
            Φq=Matrix{Complex{T}}(undef,N,length(λq))
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φq,@view(reduced.U[:,1:rkq]),Yq)
            σq=_wiersig_beyn_effective_sigma(Yq,@view reduced.singular_values[1:rkq])
            iq=findall(j->isfinite(real(λq[j]))&&isfinite(imag(λq[j]))&&wiersig_inside_contour(contour,λq[j]),eachindex(λq))
            known=λ[findall(keep)]
            used=falses(length(known));new=Int[]
            for j in iq
                best=0;bestd=T(Inf)
                for l in eachindex(known)
                    used[l]&&continue
                    d=abs(λq[j]-known[l])
                    tol=T(1e-8)*max(one(T),abs(λq[j]),abs(known[l]))
                    if d<=tol&&d<bestd
                        best=l;bestd=d
                    end
                end
                best==0 ? push!(new,j) : (used[best]=true)
            end
            nvalid=0
            if !isempty(new)
                λnew=λq[new];Φnew=Φq[:,new];σnew=σq[new]
                m=length(new)
                rawnew=fill(T(NaN),m);normalizednew=fill(T(NaN),m)
                checkednew=falses(m);keepnew=trues(m)
                _wiersig_beyn_validate_direct!(rawnew,normalizednew,checkednew,keepnew,collect(1:m),λnew,Φnew,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
                nvalid=count(keepnew)
                λ=vcat(λ,λnew);Φ=hcat(Φ,Φnew);σeff=vcat(σeff,σnew)
                raw=vcat(raw,rawnew);normalized=vcat(normalized,normalizednew)
                inside=vcat(inside,trues(m));checked=vcat(checked,checkednew);keep=vcat(keep,keepnew)
            end
            verbose&&println("SVD tolerance ",reduced.svd_tolerances[it],": rank=",rkq,", new candidates=",length(new),", new valid=",nvalid)
        end
    end
    idx=findall(keep)
    !isempty(idx)&&(idx=idx[sortperm(idx;by=j->(real(λ[j]),imag(λ[j])))])
    method=validate_roots ? :direct_all : adaptive_validation ? :direct_singular_support : :none
    candidates=(values=λ[idx],vectors=Φ[:,idx],residuals=raw[idx],normalized_residuals=normalized[idx],effective_singular_values=σeff[idx],checked=checked[idx],all_values=λ,all_vectors=Φ,all_residuals=raw,all_normalized_residuals=normalized,all_effective_singular_values=σeff,all_checked=checked,inside=inside,kept=keep)
    common=(rank=reduced.rank,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,rank_threshold=reduced.rank_threshold,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=method)
    return merge(candidates,common)
end

"""
    wiersig_beyn_INFO(solver,pts,ws,contour;...)

Diagnostic direct Beyn solve comparing nested `nq÷2` and `nq` trapezoidal
rules. This routine is intended only to verify that a representative production
contour is sufficiently resolved. Both quadratures use the same probe matrix. 
The reported displacement:

    Δ_j=min_l|k_j^(fine)-k_l^(coarse)|

measures convergence of each enclosed fine root under dyadic contour
refinement. Fine roots are additionally checked with the original direct
Wiersig residual. Production `wiersig_beyn` does not perform this dyadic calculation; 
it instead uses effective-singular-value ordered residual validation.
"""
function wiersig_beyn_INFO(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::WiersigContour{T};nq::Int=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::T=T(1e-12),relative_svd_tol::Bool=true,movement_tol::T=T(1e-8),dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),matnorm::Symbol=:one,multithreaded::Bool=true) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    iseven(nq)||throw(ArgumentError("direct Beyn diagnostic requires even nq"))
    N=boundary_matrix_size(ws);rmax=min(max_r,N)
    z,w=wiersig_beyn_contour(contour,nq)
    nested=_wiersig_beyn_build_nested_direct(solver,pts,ws,z,w;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=true)
    fine=nested.fine;coarse=nested.coarse
    coarse.rank==0&&error("coarse Beyn moment has zero numerical rank")
    fine.rank==0&&error("fine Beyn moment has zero numerical rank")
    Ec=nothing;Ef=nothing
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        Ec=eigen(coarse.B)
        Ef=eigen(fine.B)
    end
    λc=Vector{Complex{T}}(Ec.values);λf=Vector{Complex{T}}(Ef.values)
    ic=findall(j->isfinite(real(λc[j]))&&isfinite(imag(λc[j]))&&wiersig_inside_contour(contour,λc[j]),eachindex(λc))
    iff=findall(j->isfinite(real(λf[j]))&&isfinite(imag(λf[j]))&&wiersig_inside_contour(contour,λf[j]),eachindex(λf))
    sort!(ic;by=j->(real(λc[j]),imag(λc[j])))
    sort!(iff;by=j->(real(λf[j]),imag(λf[j])))
    croots=λc[ic];froots=λf[iff]
    Yf=Matrix{Complex{T}}(Ef.vectors)
    Φf=Matrix{Complex{T}}(undef,N,length(iff))
    !isempty(iff)&&@blas_multi_then_1 MAX_BLAS_THREADS mul!(Φf,fine.U,@view(Yf[:,iff]))
    dfc=isempty(croots) ? fill(T(Inf),length(froots)) : T[minimum(abs(k-kc) for kc in croots) for k in froots]
    raw=Vector{T}(undef,length(froots));normalized=similar(raw)
    Awork=Matrix{Complex{T}}(undef,N,N);ywork=Vector{Complex{T}}(undef,N)
    @showprogress "Direct residuals..." for j in eachindex(froots)
        raw[j],normalized[j]=_wiersig_beyn_residual!(Awork,ywork,solver,pts,ws,froots[j],@view(Φf[:,j]);dlp_kernel=dlp_kernel,matnorm=matnorm,multithreaded=multithreaded)
    end
    println()
    println("nq coarse/fine            = ",nq÷2," / ",nq)
    println("rank coarse/fine          = ",coarse.rank," / ",fine.rank)
    println("probe coarse/fine         = ",coarse.probe_dimension," / ",fine.probe_dimension)
    println("roots coarse/fine         = ",length(croots)," / ",length(froots))
    println("max/median displacement   = ",isempty(dfc) ? T(NaN) : maximum(dfc)," / ",isempty(dfc) ? T(NaN) : median(dfc))
    println("max/median direct norm.   = ",isempty(normalized) ? T(NaN) : maximum(normalized)," / ",isempty(normalized) ? T(NaN) : median(normalized))
    println("coarse singular values    = ");println(coarse.singular_values)
    println("fine singular values      = ");println(fine.singular_values)
    @inbounds for j in eachindex(froots)
        if !isfinite(dfc[j])||dfc[j]>movement_tol
            @warn "Beyn root" k=froots[j] Δ=dfc[j] normalized=normalized[j]
        else
            @info "Beyn root" k=froots[j] Δ=dfc[j] normalized=normalized[j]
        end
    end
    return (coarse_nq=nq÷2,fine_nq=nq,coarse=coarse,fine=fine,coarse_roots=croots,fine_roots=froots,fine_vectors=Φf,fine_displacements=dfc,direct_residuals=raw,direct_normalized_residuals=normalized)
end

"""
    _wiersig_beyn_matrix_batch_plan(N,nmat;ram_cap_gib=nothing,ram_fraction=0.75,reserve_gib=8.0)

Choose the largest dense contour-matrix batch allowed by the matrix-storage RAM
budget.

When `ram_cap_gib=nothing`, the budget is

    ram_fraction*Sys.total_memory()-reserve_gib,

so it is based on total physical RAM rather than instantaneous free memory.
`ram_cap_gib` overrides this automatic budget.

Returns the selected `batch_size` together with the single-matrix, total-RAM,
and matrix-budget byte counts.
"""
function _wiersig_beyn_matrix_batch_plan(N::Int,nmat::Int;ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75,reserve_gib::Real=8.0)
    matrix_bytes=N*N*sizeof(ComplexF64)
    total_bytes=Int(Sys.total_memory())
    reserve_bytes=floor(Int,reserve_gib*2.0^30)
    budget_bytes=isnothing(ram_cap_gib) ? floor(Int,ram_fraction*total_bytes)-reserve_bytes : floor(Int,ram_cap_gib*2.0^30)
    budget_bytes>=matrix_bytes||throw(ArgumentError("RAM budget too small for one dense Wiersig matrix"))
    B=clamp(budget_bytes÷matrix_bytes,1,nmat)
    return (batch_size=B,matrix_bytes=matrix_bytes,total_bytes=total_bytes,budget_bytes=budget_bytes)
end

"""
    _wiersig_subset_chebyshev_workspace(cws,js)

Extract a batch of vacuum-wavenumber entries from an existing Chebyshev
workspace without rebuilding radial plans or geometry caches. If the parent
contains M vacuum wavenumbers and C cavities, `js` selects the corresponding
interior plans `(a-1)M+j` and exterior plans `CM+j`, restoring the usual
component-major ordering for the smaller batch.
"""
function _wiersig_subset_chebyshev_workspace(cws::WiersigChebyshevWorkspace{T},js::AbstractVector{<:Integer}) where {T<:Real}
    M=length(cws.ks);C=cws.ncavities;Mb=length(js)
    all(j->1<=j<=M,js)||throw(BoundsError(cws.ks,js))
    ids=Vector{Int}(undef,(C+1)*Mb)
    qin=Matrix{Complex{T}}(undef,C,Mb)
    qout=Vector{Complex{T}}(undef,Mb)
    @inbounds for a in 1:C,l in 1:Mb
        j=js[l]
        ids[(a-1)*Mb+l]=(a-1)*M+j
        qin[a,l]=cws.qin[a,j]
    end
    @inbounds for l in 1:Mb
        j=js[l]
        ids[C*Mb+l]=C*M+j
        qout[l]=cws.qout[j]
    end
    ks=Complex{T}[cws.ks[j] for j in js]
    qall=cws.qall[ids]
    plans0=cws.plans0[ids];plans1=cws.plans1[ids]
    plansj0=cws.plansj0[ids];plansj1=cws.plansj1[ids]
    bfs=CFIE_H0_H1_J0_J1_BesselWorkspace((C+1)*Mb;ntls=Threads.nthreads())
    return WiersigChebyshevWorkspace(cws.direct_ws,cws.block_cache,ks,qin,qout,qall,C,plans0,plans1,plansj0,plansj1,bfs,cws.npanels_h,cws.M_h,cws.npanels_j,cws.M_j,cws.errH0[ids],cws.errH1[ids],cws.errJ0[ids],cws.errJ1[ids])
end

"""
    _wiersig_beyn_build_nested_chebyshev(solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},ws::WiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}},cws::WiersigChebyshevWorkspace{T};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Real=1e-12,relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}

Accumulate Beyn moments after assembling all contour matrices simultaneously
with the multi-k Chebyshev assembler. For contour nodes z₁,...,z_nq, all 
matrices A(z_j) are first constructed in one geometry traversal and stored simultaneously. 
The subsequent contour pass is: A(z_j) -> LU -> A(z_j)⁻¹V -> moment AXPY.
Each matrix is factorized once and all `rmax` probe right-hand sides are solved
simultaneously. The matrices are destroyed by `lu!(A,ws)` after assembly, which
is harmless because they are not needed again.
"""
function _wiersig_beyn_build_nested_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}},cws::WiersigChebyshevWorkspace{T};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::T=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    iseven(length(z))||throw(ArgumentError("nested Beyn refinement requires even nq"))
    N=boundary_matrix_size(ws);rmax=min(max_r,N);nq=length(z)
    V,X,A0,A1=wiersig_beyn_buffers(T,N,rmax,rng)
    C0=zeros(Complex{T},N,rmax);C1=zeros(Complex{T},N,rmax)
    xv=vec(X);a0v=vec(A0);a1v=vec(A1);c0v=vec(C0);c1v=vec(C1)
    mem=_wiersig_beyn_matrix_batch_plan(N,nq;ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
    B=mem.batch_size
    if verbose
        println("total physical RAM           = ",round(mem.total_bytes/2.0^30,digits=2)," GiB")
        println("matrix RAM budget             = ",round(mem.budget_bytes/2.0^30,digits=2)," GiB")
        println("matrix storage mode           = ",B==nq ? "all-k" : B==1 ? "streamed" : "batched")
        println("matrix batch size             = ",B," / ",nq)
    end
    As=[Matrix{ComplexF64}(undef,N,N) for _ in 1:B]
    p=verbose ? Progress(nq,desc="Beyn contour") : nothing
    for first in 1:B:nq
        last=min(first+B-1,nq);js=first:last;nb=length(js)
        work=nb==nq ? cws : _wiersig_subset_chebyshev_workspace(cws,js)
        Asb=nb==B ? As : As[1:nb]
        @benchit timeit=verbose "Chebyshev matrix batch" construct_matrices!(solver,Asb,pts,work;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        @inbounds for (l,j) in enumerate(js)
            F=lu!(Asb[l],ws;check=false)
            ldiv!(X,F,V)
            BLAS.axpy!(w[j],xv,a0v);BLAS.axpy!(w[j]*z[j],xv,a1v)
            if isodd(j)
                wc=T(2)*w[j]
                BLAS.axpy!(wc,xv,c0v);BLAS.axpy!(wc*z[j],xv,c1v)
            end
            verbose&&next!(p)
        end
    end
    As=nothing
    GC.gc()
    fine=_wiersig_beyn_build_reduced_problem(A0,A1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
    coarse=_wiersig_beyn_build_reduced_problem(C0,C1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
    return (fine=fine,coarse=coarse)
end

"""
    _wiersig_beyn_build_chebyshev(...)

Accumulate the production Beyn moments using the requested contour quadrature
and multi-k Chebyshev matrix assembly.

For the common probe matrix `V∈C^{N×rmax}`, contour matrices are assembled in
RAM-limited batches. Each `A(z_j)` is factorized once and all probe right-hand
sides are solved simultaneously,

    X_j=A(z_j)⁻¹V,

after which

    A₀+=w_jX_j,
    A₁+=w_jz_jX_j.

Only the requested `nq` rule is constructed. No coarse contour moments are
formed. The dyadic `nq÷2 -> nq` comparison is reserved for
`wiersig_beyn_chebyshev_INFO`.
"""
function _wiersig_beyn_build_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}},cws::WiersigChebyshevWorkspace{T};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    N=boundary_matrix_size(ws);rmax=min(max_r,N);nq=length(z)
    V,X,A0,A1=wiersig_beyn_buffers(T,N,rmax,rng)
    xv=vec(X);a0v=vec(A0);a1v=vec(A1)
    mem=_wiersig_beyn_matrix_batch_plan(N,nq;ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
    B=mem.batch_size
    if verbose
        println("total physical RAM           = ",round(mem.total_bytes/2.0^30,digits=2)," GiB")
        println("matrix RAM budget            = ",round(mem.budget_bytes/2.0^30,digits=2)," GiB")
        println("matrix storage mode          = ",B==nq ? "all-k" : B==1 ? "streamed" : "batched")
        println("matrix batch size            = ",B," / ",nq)
    end
    As=[Matrix{ComplexF64}(undef,N,N) for _ in 1:B]
    p=verbose ? Progress(nq,desc="Beyn contour") : nothing
    for first in 1:B:nq
        last=min(first+B-1,nq);js=first:last;nb=length(js)
        work=nb==nq ? cws : _wiersig_subset_chebyshev_workspace(cws,js)
        Asb=nb==B ? As : As[1:nb]
        @benchit timeit=verbose "Chebyshev matrix batch" construct_matrices!(solver,Asb,pts,work;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        @inbounds for (l,j) in enumerate(js)
            F=lu!(Asb[l],ws;check=false)
            ldiv!(X,F,V)
            BLAS.axpy!(w[j],xv,a0v)
            BLAS.axpy!(w[j]*z[j],xv,a1v)
            verbose&&next!(p)
        end
    end
    As=nothing;GC.gc()
    return _wiersig_beyn_build_reduced_problem(A0,A1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
end

"""
    construct_wiersig_B_matrix_chebyshev(...)

Construct the production reduced Beyn problem using multi-k Chebyshev matrix
assembly. A Chebyshev workspace is built for all `nq` contour wavenumbers. The contour
matrices are then assembled in RAM-limited batches. For every contour node
`z_j`, `A(z_j)` is independently factorized and applied to the common probe
matrix `V`; the resulting solves are accumulated directly into the zeroth and
first Beyn moments. Only the requested `nq` quadrature is used. Dyadic contour refinement is not
part of this production routine.
"""
function construct_wiersig_B_matrix_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::WiersigContour{T};nq::Int=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),cheb_verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    z,w=wiersig_beyn_contour(contour,nq)
    @benchit timeit=verbose "Chebyshev workspace" cws=build_chebyshev_workspace(solver,pts,z;npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=cheb_verbose)
    reduced=_wiersig_beyn_build_chebyshev(solver,pts,ws,z,w,cws;r=r,r_step=r_step,max_r=max_r,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose,ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
    return merge(reduced,(contour=contour,contour_nodes=z,contour_weights=w,cheb_workspace=cws))
end

"""
    _wiersig_beyn_validate_chebyshev!(...)

Validate the selected Beyn candidates by batched multi-k Chebyshev evaluation
of the nonlinear residual. The caller supplies the candidate indices `idx`;
production adaptive validation normally supplies consecutive candidates ordered
by increasing effective moment singular value `σeff`.
A Chebyshev workspace is built only for the selected candidate wavenumbers and
their Wiersig matrices are assembled simultaneously. `checked[j]` records that
candidate `j` was evaluated and `keep[j]` records whether it satisfies the
requested residual tolerances.
"""
function _wiersig_beyn_validate_chebyshev!(raw::Vector{T},normalized::Vector{T},checked::BitVector,keep::BitVector,idx::Vector{Int},λ::Vector{Complex{T}},Φ::Matrix{Complex{T}},solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace;res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,multithreaded::Bool=true,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),verbose::Bool=false) where {T<:Real}
    isempty(idx)&&return nothing
    ks=Complex{T}[λ[j] for j in idx];N=boundary_matrix_size(ws)
    cws=build_chebyshev_workspace(solver,pts,ks;npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=false)
    As=[Matrix{ComplexF64}(undef,N,N) for _ in idx]
    @benchit timeit=verbose "matrix construction" construct_matrices!(solver,As,pts,cws;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
    y=Vector{Complex{T}}(undef,N)
    @inbounds for (l,j) in enumerate(idx)
        raw[j],normalized[j]=_wiersig_beyn_matrix_residual!(y,As[l],@view(Φ[:,j]);matnorm=matnorm)
        checked[j]=true
        keep[j]=(!filter_raw_residual||raw[j]<res_tol)&&normalized[j]<normalized_res_tol
        verbose&&println("adaptive candidate: k=",λ[j],", raw=",raw[j],", normalized=",normalized[j],", kept=",keep[j])
    end
    return nothing
end

"""
    wiersig_beyn_chebyshev(...)

Solve the dielectric resonance problem with Beyn's contour method using multi-k
Chebyshev matrix assembly.

Production uses only the requested `nq`-point contour quadrature. After the
reduced problem

    BY=YΛ

is solved, each candidate eigenvector `Y[:,j]` is assigned

    σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]⁻¹ᐟ²,

where `σ_l` are the retained singular values of the zeroth Beyn moment.
With `adaptive_validation=true`, enclosed candidates are ordered by increasing
`σeff` and checked by batched Chebyshev residual evaluation. Validation stops
once `validation_padding` consecutive checked candidates pass after the most
recent failure. Unchecked enclosed candidates are retained.

`validate_roots=true` instead validates every enclosed candidate.
Setting both `validate_roots=false` and `adaptive_validation=false` performs no
nonlinear residual checks.

The dyadic `nq÷2 -> nq` convergence comparison is diagnostic only and is
performed by `wiersig_beyn_chebyshev_INFO`.

If `return_workspace=true`, the contour Chebyshev workspace is included in the
returned named tuple.
"""
function wiersig_beyn_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::WiersigContour{T};nq::Int=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,validate_roots::Bool=false,adaptive_validation::Bool=true,validation_padding::Int=5,res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false,return_workspace::Bool=false,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),cheb_verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    reduced=construct_wiersig_B_matrix_chebyshev(solver,pts,ws,contour;nq=nq,r=r,r_step=r_step,max_r=max_r,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,cheb_verbose=cheb_verbose,ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
    N=boundary_matrix_size(ws)
    if reduced.rank==0
        empty=(values=Complex{T}[],vectors=Matrix{Complex{T}}(undef,N,0),residuals=T[],normalized_residuals=T[],effective_singular_values=T[],checked=Bool[],all_values=Complex{T}[],all_vectors=Matrix{Complex{T}}(undef,N,0),all_residuals=T[],all_normalized_residuals=T[],all_effective_singular_values=T[],all_checked=Bool[],inside=Bool[],kept=Bool[])
        common=(rank=0,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,rank_threshold=reduced.rank_threshold,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=:none)
        return return_workspace ? merge(empty,common,(cheb_workspace=reduced.cheb_workspace,)) : merge(empty,common)
    end
    rk=reduced.rank
    Ef=nothing
    @blas_multi_then_1 MAX_BLAS_THREADS Ef=eigen(Matrix(@view reduced.B[1:rk,1:rk]))
    λ=Vector{Complex{T}}(Ef.values);Y=Matrix{Complex{T}}(Ef.vectors)
    Φ=Matrix{Complex{T}}(undef,N,length(λ))
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φ,@view(reduced.U[:,1:rk]),Y)
    σeff=_wiersig_beyn_effective_sigma(Y,@view reduced.singular_values[1:rk])
    nroots=length(λ);inside=falses(nroots);keep=falses(nroots);checked=falses(nroots)
    raw=fill(T(NaN),nroots);normalized=fill(T(NaN),nroots)
    @inbounds for j in eachindex(λ)
        inside[j]=isfinite(real(λ[j]))&&isfinite(imag(λ[j]))&&wiersig_inside_contour(contour,λ[j])
        keep[j]=inside[j]
    end
    inside_idx=findall(inside)
    if validate_roots
        _wiersig_beyn_validate_chebyshev!(raw,normalized,checked,keep,inside_idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=verbose)
    elseif adaptive_validation&&!isempty(inside_idx)
        validator=idx->_wiersig_beyn_validate_chebyshev!(raw,normalized,checked,keep,idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=verbose)
        order=_wiersig_beyn_singular_validation!(validator,inside,σeff,checked,keep;validation_padding=validation_padding)
        verbose&&println("singular-support validation: checked=",count(checked),", rejected=",count(inside .& .!keep),", padding=",validation_padding,", σeff first/last=",isempty(order) ? T(NaN) : σeff[first(order)]," / ",isempty(order) ? T(NaN) : σeff[last(order)])
    end
    if length(reduced.ranks)>1
        @inbounds for it in 2:length(reduced.ranks)
            rkq=reduced.ranks[it]
            rkq==reduced.ranks[it-1]&&continue
            Eq=nothing
            @blas_multi_then_1 MAX_BLAS_THREADS Eq=eigen(Matrix(@view reduced.B[1:rkq,1:rkq]))
            λq=Vector{Complex{T}}(Eq.values);Yq=Matrix{Complex{T}}(Eq.vectors)
            Φq=Matrix{Complex{T}}(undef,N,length(λq))
            @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φq,@view(reduced.U[:,1:rkq]),Yq)
            σq=_wiersig_beyn_effective_sigma(Yq,@view reduced.singular_values[1:rkq])
            iq=findall(j->isfinite(real(λq[j]))&&isfinite(imag(λq[j]))&&wiersig_inside_contour(contour,λq[j]),eachindex(λq))
            known=λ[findall(keep)]
            used=falses(length(known));new=Int[]
            for j in iq
                best=0;bestd=T(Inf)
                for l in eachindex(known)
                    used[l]&&continue
                    d=abs(λq[j]-known[l])
                    tol=T(1e-8)*max(one(T),abs(λq[j]),abs(known[l]))
                    if d<=tol&&d<bestd
                        best=l;bestd=d
                    end
                end
                best==0 ? push!(new,j) : (used[best]=true)
            end
            nvalid=0
            if !isempty(new)
                λnew=λq[new];Φnew=Φq[:,new];σnew=σq[new]
                m=length(new)
                rawnew=fill(T(NaN),m);normalizednew=fill(T(NaN),m)
                checkednew=falses(m);keepnew=trues(m)
                _wiersig_beyn_validate_chebyshev!(rawnew,normalizednew,checkednew,keepnew,collect(1:m),λnew,Φnew,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=verbose)
                nvalid=count(keepnew)
                λ=vcat(λ,λnew);Φ=hcat(Φ,Φnew);σeff=vcat(σeff,σnew)
                raw=vcat(raw,rawnew);normalized=vcat(normalized,normalizednew)
                inside=vcat(inside,trues(m));checked=vcat(checked,checkednew);keep=vcat(keep,keepnew)
            end
            verbose&&println("SVD tolerance ",reduced.svd_tolerances[it],": rank=",rkq,", new candidates=",length(new),", new valid=",nvalid)
        end
    end
    idx=findall(keep)
    !isempty(idx)&&(idx=idx[sortperm(idx;by=j->(real(λ[j]),imag(λ[j])))])
    method=validate_roots ? :chebyshev_all : adaptive_validation ? :chebyshev_singular_support : :none
    candidates=(values=λ[idx],vectors=Φ[:,idx],residuals=raw[idx],normalized_residuals=normalized[idx],effective_singular_values=σeff[idx],checked=checked[idx],all_values=λ,all_vectors=Φ,all_residuals=raw,all_normalized_residuals=normalized,all_effective_singular_values=σeff,all_checked=checked,inside=inside,kept=keep)
    common=(rank=reduced.rank,ranks=reduced.ranks,svd_tolerances=reduced.svd_tolerances,rank_threshold=reduced.rank_threshold,rank_thresholds=reduced.rank_thresholds,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=method)
    return return_workspace ? merge(candidates,common,(cheb_workspace=reduced.cheb_workspace,)) : merge(candidates,common)
end

"""
    wiersig_beyn_chebyshev_INFO(solver,pts,ws,contour;...)

A single-contour diagnostic used to assess whether the chosen Beyn quadrature is
sufficiently resolved for a representative contour. It compares the nested
`nq÷2` and `nq` trapezoidal rules, reports coarse/fine moment ranks and root
movement, and evaluates direct nonlinear residuals of the fine roots.
The principal quadrature-convergence diagnostic is

    Δ_j=min_l|k_j^(fine)-k_l^(coarse)|.

Roots with `Δ_j>movement_tol`, or roots which have no corresponding coarse
solution, are potentially unresolved. This is the same criterion used by the
adaptive production validation.

# Kwargs

- `nq::Int=64`: fine contour quadrature size. Must be even so the nested coarse rule uses `nq÷2` nodes.
- `r::Int=16`: initial probe dimension.
- `r_step::Int=r`: increment used if the detected moment rank saturates the current probe width.
- `max_r::Int=min(boundary_matrix_size(ws),4*r)`: maximum probe dimension allowed during nested rank detection.
- `svd_tol::T=T(1e-12)`: numerical-rank threshold for the zeroth Beyn moment.
- `relative_svd_tol::Bool=true`: if true, use `svd_tol*σ₁`; otherwise use `svd_tol` as an absolute threshold.
- `movement_tol::T=T(1e-8)`: coarse/fine pole-displacement threshold.
- `dlp_kernel::Symbol=:source`: DLP normal convention used in Wiersig matrix assembly.
- `rng::AbstractRNG=MersenneTwister(0)`: random generator used for the Beyn probe matrix.
- `matnorm::Symbol=:one`: matrix/vector norm family used for normalized residuals.
- `multithreaded::Bool=true`: enable threaded matrix assembly.
- `npanels_h_init::Int=15_000`, `M_h_init::Int=5`: initial Hankel Chebyshev parameters.
- `npanels_j_init::Int=3_000`, `M_j_init::Int=5`: initial Bessel-J Chebyshev parameters.
- `cheb_tol::T=T(1e-11)`: target Chebyshev interpolation tolerance.
- `sampling_points::Int=50_000`: number of points used when estimating interpolation errors.
- `max_iter::Int=20`: maximum number of Chebyshev-plan refinement iterations.
- `grow_panels::T=T(1.5)`: multiplicative panel-count growth factor.
- `grow_M::Int=2`: polynomial-degree increment during Chebyshev refinement.
- `cheb_verbose::Bool=false`: print Chebyshev-plan construction diagnostics.
"""
function wiersig_beyn_chebyshev_INFO(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::WiersigContour{T};nq::Int=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::T=T(1e-12),relative_svd_tol::Bool=true,movement_tol::T=T(1e-8),dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),matnorm::Symbol=:one,multithreaded::Bool=true,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),cheb_verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    iseven(nq)||throw(ArgumentError("nested Beyn diagnostic requires even nq"))
    nqcoarse=nq÷2;N=boundary_matrix_size(ws);rmax=min(max_r,N)
    z,w=wiersig_beyn_contour(contour,nq)
    cws=build_chebyshev_workspace(solver,pts,z;npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=cheb_verbose)
    V=randn(rng,Complex{T},N,rmax);X=similar(V)
    f0=zeros(Complex{T},N,rmax);f1=zeros(Complex{T},N,rmax)
    c0=zeros(Complex{T},N,rmax);c1=zeros(Complex{T},N,rmax)
    xv=vec(X);f0v=vec(f0);f1v=vec(f1);c0v=vec(c0);c1v=vec(c1)
    mem=_wiersig_beyn_matrix_batch_plan(N,nq;ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
    B=mem.batch_size
    println("total physical RAM           = ",round(mem.total_bytes/2.0^30,digits=2)," GiB")
    println("matrix RAM budget             = ",round(mem.budget_bytes/2.0^30,digits=2)," GiB")
    println("matrix storage mode           = ",B==nq ? "all-k" : B==1 ? "streamed" : "batched")
    println("matrix batch size             = ",B," / ",nq)
    As=[Matrix{ComplexF64}(undef,N,N) for _ in 1:B]
    p=Progress(nq,desc="contour...")
    for first in 1:B:nq
        last=min(first+B-1,nq);js=first:last;nb=length(js)
        work=nb==nq ? cws : _wiersig_subset_chebyshev_workspace(cws,js)
        Asb=nb==B ? As : As[1:nb]
        @benchit timeit=true "Chebyshev matrix batch" construct_matrices!(solver,Asb,pts,work;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        @inbounds for (l,j) in enumerate(js)
            F=lu!(Asb[l],ws;check=false)
            ldiv!(X,F,V)
            wj=w[j]
            BLAS.axpy!(wj,xv,f0v);BLAS.axpy!(wj*z[j],xv,f1v)
            if isodd(j)
                wc=T(2)*wj
                BLAS.axpy!(wc,xv,c0v);BLAS.axpy!(wc*z[j],xv,c1v)
            end
            next!(p)
        end
    end
    coarse=_wiersig_beyn_build_reduced_problem(c0,c1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=false)
    fine=_wiersig_beyn_build_reduced_problem(f0,f1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=false)
    coarse.rank==0&&error("coarse Beyn moment has zero numerical rank")
    fine.rank==0&&error("fine Beyn moment has zero numerical rank")
    Ec=nothing;Ef=nothing
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        Ec=eigen(coarse.B)
        Ef=eigen(fine.B)
    end
    λc=Vector{Complex{T}}(Ec.values)
    λf=Vector{Complex{T}}(Ef.values)
    ic=findall(j->isfinite(real(λc[j]))&&isfinite(imag(λc[j]))&&wiersig_inside_contour(contour,λc[j]),eachindex(λc))
    iff=findall(j->isfinite(real(λf[j]))&&isfinite(imag(λf[j]))&&wiersig_inside_contour(contour,λf[j]),eachindex(λf))
    sort!(ic;by=j->(real(λc[j]),imag(λc[j])))
    sort!(iff;by=j->(real(λf[j]),imag(λf[j])))
    croots=λc[ic]
    froots=λf[iff]
    Yf=Matrix{Complex{T}}(Ef.vectors)
    Φf=Matrix{Complex{T}}(undef,N,length(iff))
    !isempty(iff)&&@blas_multi_then_1 MAX_BLAS_THREADS mul!(Φf,fine.U,@view(Yf[:,iff]))
    dfc=isempty(croots) ? fill(T(Inf),length(froots)) : T[minimum(abs(k-kc) for kc in croots) for k in froots]
    raw=Vector{T}(undef,length(froots));normalized=similar(raw)
    Awork=Matrix{Complex{T}}(undef,N,N);ywork=Vector{Complex{T}}(undef,N)
    @showprogress "Direct residuals..." for j in eachindex(froots)
        raw[j],normalized[j]=_wiersig_beyn_residual!(Awork,ywork,solver,pts,ws,froots[j],@view(Φf[:,j]);dlp_kernel=dlp_kernel,matnorm=matnorm,multithreaded=multithreaded)
    end
    println()
    println("nq coarse/fine            = ",nqcoarse," / ",nq)
    println("rank coarse/fine          = ",coarse.rank," / ",fine.rank)
    println("probe coarse/fine         = ",coarse.probe_dimension," / ",fine.probe_dimension)
    println("roots coarse/fine         = ",length(croots)," / ",length(froots))
    println("max/median displacement   = ",isempty(dfc) ? T(NaN) : maximum(dfc)," / ",isempty(dfc) ? T(NaN) : median(dfc))
    println("max/median direct norm.   = ",isempty(normalized) ? T(NaN) : maximum(normalized)," / ",isempty(normalized) ? T(NaN) : median(normalized))
    println("coarse singular values    = ");println(coarse.singular_values)
    println("fine singular values      = ");println(fine.singular_values)
    @inbounds for j in eachindex(froots)
        suspicious=!isfinite(dfc[j])||dfc[j]>movement_tol
        if suspicious
            @warn "Beyn root" k=froots[j] Δ=dfc[j] normalized=normalized[j]
        else
            @info "Beyn root" k=froots[j] Δ=dfc[j] normalized=normalized[j]
        end
    end
    return (coarse_nq=nqcoarse,fine_nq=nq,coarse=coarse,fine=fine,coarse_roots=croots,fine_roots=froots,fine_vectors=Φf,fine_displacements=dfc,direct_residuals=raw,direct_normalized_residuals=normalized,cheb_workspace=cws)
end

# helper that will clip the contours into the actual wanted region (if the contours cross say the real axis it can otherwise pick up spurious roots that behave weirdly)
@inline function _wiersig_in_spectrum_region(k::Complex{T},region::Tuple{T,T,T,T}) where {T<:Real}
    re_min,re_max,im_min,im_max=region
    return re_min<=real(k)<=re_max&&im_min<=imag(k)<=im_max
end

"""
    compute_spectrum(solver::AbstractWiersigSolver,contours::AbstractVector{<:WiersigContour{T}};...) where {T<:Real}

# Kwargs

- `chebyshev::Bool=true`: use multi-k Chebyshev matrix assembly.
- `nq::Int=64`: number of production contour quadrature nodes. It need only be even when `do_INFO=true`, because the INFO diagnostic additionally forms the nested `nq÷2` rule.
- `r::Int=16`: initial probe dimension.
- `r_step::Int=r`, `max_r::Int=4*r`: probe-growth controls used when the detected Beyn moment rank saturates the current probe dimension.
- `svd_tol::AbstractVector{T}=T[1e-7,5e-8,1e-8,5e-9,1e-9,5e-10,1e-10,5e-11,1e-11]`: numerical-rank threshold for the zeroth Beyn moment.
- `do_INFO::Bool=true`: run one representative-contour convergence diagnostic before the complete sweep.
- `validate_roots::Bool=false`: directly validate every enclosed root when true.
- `adaptive_validation::Bool=true`: validate candidates in increasing effective moment singular value `σeff`, stopping after `validation_padding` consecutive good candidates follow the last failure.
- `validation_padding::Int=5`: number of consecutive residual-good candidates required before adaptive validation stops.
- `movement_tol::T=T(1e-8)`: coarse/fine pole-displacement threshold used only by the optional dyadic INFO diagnostic.
- `normalized_res_tol::T=T(1e-10)`: normalized nonlinear-residual threshold.
- `merge_atol::T=T(1e-10)`, `merge_rtol::T=T(1e-10)`: tolerances used to merge roots found on overlapping contours.
- `rng_seed::Int=0`: deterministic random-probe seed.
- `multithreaded::Bool=true`: enable threaded matrix assembly.
- `verbose::Bool=true`: print progress and the final spectrum summary.

# Returns

A named tuple with:

- `values::Vector{Complex{T}}`: overlap-merged resonances enclosed by the contour union.
- `vectors::Vector{Vector{Complex{T}}}`: corresponding boundary vectors. Vector lengths may differ because each resonance retains the discretization of its source contour.
- `residuals::Vector{T}`: raw residuals; `NaN` when not evaluated.
- `normalized_residuals::Vector{T}`: normalized residuals; `NaN` when not evaluated.
- `source_contours::Vector{Int}`: source contour retained for each resonance.
- `contours`: supplied contour collection.
- `contour_results`: individual contour results before overlap merging.
- `contour_pts`, `contour_workspaces`: local boundary discretizations and workspaces.
- `contour_dimensions`: boundary-matrix dimension for each contour.
- `contour_k_resolution`, `contour_q_resolution`: local spectral-resolution bounds.
- `INFO`: representative diagnostic result when `do_INFO=true`, otherwise `nothing`.
"""
function compute_spectrum(solver::AbstractWiersigSolver,contours::AbstractVector{<:WiersigContour{T}},region::Tuple{T,T,T,T};chebyshev::Bool=true,nq::Int=64,r::Int=16,r_step::Int=r,max_r::Int=4*r,svd_tol::AbstractVector{T}=T[1e-7,5e-8,1e-8,5e-9,1e-9,5e-10,1e-10,5e-11,1e-11],relative_svd_tol::Bool=true,res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-10),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,rng_seed::Int=0,multithreaded::Bool=true,merge_atol::T=T(1e-10),merge_rtol::T=T(1e-10),npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=10_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),do_INFO::Bool=true,cheb_verbose::Bool=false,verbose::Bool=true,gc_between_contours::Bool=false,validate_roots::Bool=false,adaptive_validation::Bool=true,movement_tol::T=T(1e-8),validation_padding::Int=5,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    do_INFO&&isodd(nq)&&throw(ArgumentError("do_INFO=true requires even nq for the dyadic nq÷2 -> nq diagnostic"))
    ncontours=length(contours)
    C=length(solver.billiards)
    nin=_wiersig_component_indices(solver,C)
    contour_pts=Vector{Any}(undef,ncontours)
    contour_ws=Vector{Any}(undef,ncontours)
    contour_dims=Vector{Int}(undef,ncontours)
    contour_kmax=Vector{T}(undef,ncontours)
    contour_qres=Vector{Vector{T}}(undef,ncontours)
    @showprogress "Boundary workspaces" for ic in eachindex(contours)
        contour=contours[ic]
        kr=abs(real(contour.center))+contour.halfwidth
        ki=abs(imag(contour.center))+contour.halfheight
        kmax=hypot(kr,ki)
        qres=T[max(nin[a],solver.n_out)*kmax for a in 1:C]
        pts=evaluate_points(solver,qres)
        ws=build_cfie_kress_workspace(solver,pts)
        contour_pts[ic]=pts
        contour_ws[ic]=ws
        contour_dims[ic]=boundary_matrix_size(ws)
        contour_kmax[ic]=kmax
        contour_qres[ic]=qres
    end
    if verbose
        println()
        println("contours                = ",ncontours)
        println("nodes/contour           = ",nq)
        println("halfwidth range         = ",minimum(c.halfwidth for c in contours)," : ",maximum(c.halfwidth for c in contours))
        println("halfheight range        = ",minimum(c.halfheight for c in contours)," : ",maximum(c.halfheight for c in contours))
        println("matrix dimension range  = ",minimum(contour_dims)," : ",maximum(contour_dims))
        println("initial/max probe       = ",r," / ",max_r)
        println("relative SVD threshold  = ",relative_svd_tol)
        println("SVD tolerance           = ",svd_tol)
        println("normalized residual tol = ",normalized_res_tol)
        println("Chebyshev               = ",chebyshev ? "local per contour" : "disabled")
        println("discretization          = local per contour")
        println("─────────────────────────────────────────────────")
        println()
    end
    info_result=nothing
    if do_INFO
        zmean=sum(c.center for c in contours)/ncontours
        info_index=argmin(abs(c.center-zmean) for c in contours)
        contour=contours[info_index]
        pts=contour_pts[info_index]
        ws=contour_ws[info_index]
        N=contour_dims[info_index]
        ri=min(r,N);maxri=min(max_r,N);rstepi=min(r_step,maxri)
        verbose&&println("Running Beyn diagnostic on contour at: ",contour.center,", dim=",N)
        info_result=if chebyshev
            wiersig_beyn_chebyshev_INFO(solver,pts,ws,contour;nq=nq,r=ri,r_step=rstepi,max_r=maxri,svd_tol=first(svd_tol),relative_svd_tol=relative_svd_tol,movement_tol=movement_tol,dlp_kernel=dlp_kernel,rng=MersenneTwister(rng_seed+info_index),matnorm=matnorm,multithreaded=multithreaded,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,cheb_verbose=cheb_verbose,ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
        else
            wiersig_beyn_INFO(solver,pts,ws,contour;nq=nq,r=ri,r_step=rstepi,max_r=maxri,svd_tol=first(svd_tol),relative_svd_tol=relative_svd_tol,movement_tol=movement_tol,dlp_kernel=dlp_kernel,rng=MersenneTwister(rng_seed+info_index),matnorm=matnorm,multithreaded=multithreaded)
        end
    end
    results=Vector{Any}(undef,ncontours)
    @showprogress "Beyn spectrum" for ic in eachindex(contours)
        contour=contours[ic]
        pts=contour_pts[ic]
        ws=contour_ws[ic]
        N=contour_dims[ic]
        ri=min(r,N)
        maxri=min(max_r,N)
        rstepi=min(r_step,maxri)
        rng=MersenneTwister(rng_seed+ic)
        result=if chebyshev
            wiersig_beyn_chebyshev(solver,pts,ws,contour;nq=nq,r=ri,r_step=rstepi,max_r=maxri,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose,return_workspace=false,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,cheb_verbose=cheb_verbose,validate_roots=validate_roots,adaptive_validation=adaptive_validation,validation_padding=validation_padding,ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
        else
            wiersig_beyn(solver,pts,ws,contour;nq=nq,r=ri,r_step=rstepi,max_r=maxri,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose,validate_roots=validate_roots,adaptive_validation=adaptive_validation,validation_padding=validation_padding)
        end
        results[ic]=result
        if verbose
            wanted=result.inside .& map(k->_wiersig_in_spectrum_region(k,region),result.all_values)
            checked_wanted=wanted .& result.all_checked
            nwanted=count(wanted .& result.kept)
            nchecked=count(checked_wanted)
            nrejected=count(wanted .& .!result.kept)
            σwanted=result.all_effective_singular_values[wanted]
            σchecked=result.all_effective_singular_values[checked_wanted]
            println("contour ",ic,"/",ncontours,": center=",contour.center,", dim=",N,", rank=",result.rank,", probe=",result.probe_dimension,", accepted=",nwanted,", checked=",nchecked,", rejected=",nrejected,", min σeff=",isempty(σwanted) ? T(NaN) : minimum(σwanted),", checked-through σeff=",isempty(σchecked) ? T(NaN) : maximum(σchecked))
        end
        gc_between_contours&&(GC.gc();GC.gc())
    end
    values=Complex{T}[]
    vectors=Vector{Vector{Complex{T}}}()
    residuals=T[]
    normalized_residuals=T[]
    source_contours=Int[]
    for ic in eachindex(results)
        result=results[ic]
        for j in eachindex(result.values)
            k=result.values[j]
            _wiersig_in_spectrum_region(k,region)||continue # skip value of inside contour but outside the wanted region
            match=0
            best=typemax(T)
            @inbounds for l in eachindex(values)
                tol=merge_atol+merge_rtol*max(one(T),abs(k),abs(values[l]))
                d=abs(k-values[l])
                if d<=tol&&d<best
                    match=l
                    best=d
                end
            end
            raw=result.residuals[j]
            nr=result.normalized_residuals[j]
            if match==0
                push!(values,k)
                push!(vectors,Vector{Complex{T}}(@view result.vectors[:,j]))
                push!(residuals,raw)
                push!(normalized_residuals,nr)
                push!(source_contours,ic)
            elseif isfinite(nr)&&(!isfinite(normalized_residuals[match])||nr<normalized_residuals[match])
                values[match]=k
                vectors[match]=Vector{Complex{T}}(@view result.vectors[:,j])
                residuals[match]=raw
                normalized_residuals[match]=nr
                source_contours[match]=ic
            end
        end
    end
    order=sortperm(eachindex(values);by=i->(real(values[i]),imag(values[i])))
    spectrum_values=values[order]
    spectrum_vectors=vectors[order]
    spectrum_residuals=residuals[order]
    spectrum_normalized_residuals=normalized_residuals[order]
    spectrum_source_contours=source_contours[order]
    if verbose
        println()
        println("──── SPECTRUM SUMMARY ────")
        println("contours solved          = ",ncontours)
        accepted=sum(results) do result
            count(k->_wiersig_in_spectrum_region(k,region),result.values)
        end
        println("accepted                 = ",accepted)
        println("unique overlap merged    = ",length(spectrum_values))
        println("matrix dimension min/max = ",minimum(contour_dims)," / ",maximum(contour_dims))
        for i in eachindex(spectrum_values)
            k=spectrum_values[i];ic=spectrum_source_contours[i];nr=spectrum_normalized_residuals[i]
            if isfinite(nr)
                println(i,": k=",k,", Q=",-real(k)/(2imag(k)),", residual=",nr,", contour=",ic)
            else
                println(i,": k=",k,", Q=",-real(k)/(2imag(k)),", contour=",ic)
            end
        end
        println("─────────────────────────────────────────────────")
        println()
    end
    return (values=spectrum_values,vectors=spectrum_vectors,residuals=spectrum_residuals,normalized_residuals=spectrum_normalized_residuals,source_contours=spectrum_source_contours,contours=contours,contour_results=results,contour_pts=contour_pts,contour_workspaces=contour_ws,contour_dimensions=contour_dims,contour_k_resolution=contour_kmax,contour_q_resolution=contour_qres,INFO=info_result)
end