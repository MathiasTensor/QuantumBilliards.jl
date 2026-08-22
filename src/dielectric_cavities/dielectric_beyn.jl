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

VERIFICATION & SPURIOUS ROOTS

Production Beyn uses a single nq-point contour quadrature. After
B=U_r'*A₁W_rΣ_r⁻¹ is diagonalized as BY=YΛ, each candidate is assigned the
effective retained singular value

    σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]^(-1/2).

Candidates with small σeff depend most strongly on weak retained directions of
the zeroth Beyn moment and are checked first with the original nonlinear
residual. Empirically their residual acceptance is ordered by σeff: weak
candidates fail first and sufficiently strong candidates pass. Production
adaptive validation therefore locates the reject -> accept transition by binary
search, requiring O(log M) residual evaluations for M enclosed candidates, plus
a short `validation_padding` confirmation run.

The original padded linear scan is retained as a debugging/reference method.
A `:check` mode runs the binary search and then validates every enclosed
candidate, asserting that the binary-inferred accepted-root set was exact.
Unchecked candidates on the accepted side are retained.

SPECTRUM TESSELLATION

To compute all resonances in Ω={k∈C: re_min≤Re(k)≤re_max, im_min≤Im(k)≤im_max},
a large horizontal k-range is first divided into resolution intervals. Each
interval is handled by one `_compute_spectrum` call and therefore uses one
common boundary discretization, Wiersig workspace, probe matrix and dense
matrix buffers, sized from the largest |k| in that interval.

    full spectral region
    +-----------+-----------+-----------+
    | interval 1| interval 2| interval 3|
    +-----------+-----------+-----------+
         N₁          N₂          N₃

Typically N₁≤N₂≤N₃ as Re(k) increases. This avoids discretizing the entire
spectral region at the resolution required by its largest k. Adjacent
resolution intervals are independent; reusable data are shared only inside
one `_compute_spectrum` call.

Inside each resolution interval, Ω is partitioned into non-overlapping
ownership cells.

Smooth tessellation
-------------------
Each ownership cell is covered by its own smooth periodic Beyn contour.
The ownership cells are disjoint, while neighboring integration contours
may overlap.

                 Γ₁                 Γ₂

             .---------.       .---------.
          .-'           `-. .-'           `-.
        .'                 X                 `.
       /                 .' `.                 \
      |     +-------+   /     \   +-------+     |
      |     | cell 1|  |       |  | cell 2|     |
      |     +-------+   \     /   +-------+     |
       \                 `. .'                 /
        `.                 X                 .'
          `-.           .-' `-.           .-'
             `---------'       `---------'

                    Γ₁ ∩ Γ₂ ≠ ∅

The two contour boundaries cross at the two X points. Their contour quadratures
are integrated independently, but all contours in the resolution interval share
the same boundary discretization, probe matrix and dense buffers.


Rectangular tessellation
------------------------------
Neighboring cells share geometrically identical straight edges.

        e₁          e₂          e₃
    +-----------+-----------+-----------+
    |           |           |           |
 e₄ |  cell 1   |  cell 2   |  cell 3   | e₇
    |           |           |           |
    +-----------+-----------+-----------+
        e₅          e₆          e₈
                ^
                |
        shared edge

Every unique edge is integrated only once. The same edge moments are reused by its one or two neighboring cells with the
appropriate contour-orientation sign, so the expensive solve

    X(z)=A(z)⁻¹V

is performed only once at every Gauss-Legendre node of that edge.
All rectangular cells within one resolution interval therefore use one common
boundary discretization, probe matrix and dense scratch buffers.

- TODO: Deep regions with Im(k)<0 may be better divided into horizontal strips because outgoing Hankel functions grow rapidly in the lower half-plane.
  TODO: Structured/HSS factorization for large dense boundary matrices.
Reference:
W.-J. Beyn, Linear Algebra Appl. 436 (2012), 3839–3863.
For rect. contour: 
Y. Liu, J. E. Roman, and M. Shao, “Solving nonlinear eigenvalue problems via contour integration and region partitioning,” arXiv:2503.12038 (2025).
=#

abstract type AbstractWiersigContour{T<:Real} end

# Smooth periodic contours integrated with the trapezoidal rule.
struct WiersigSmoothContour{T<:Real,F,G,H}<:AbstractWiersigContour{T}
    center::Complex{T}
    halfwidth::T
    halfheight::T
    z::F
    dz::G
    inside::H
    ownership::Union{Nothing,NTuple{4,T}}
    function WiersigSmoothContour(center::Complex{T},halfwidth::T,halfheight::T,z::F,dz::G,inside::H,
        ownership::Union{Nothing,NTuple{4,T}}) where {T<:Real,F,G,H}
        return new{T,F,G,H}(center,halfwidth,halfheight,z,dz,inside,ownership)
    end
end
function WiersigSmoothContour(center::Complex{T},halfwidth::T,halfheight::T,z::F,dz::G,inside::H;
    ownership::Union{Nothing,NTuple{4,T}}=nothing) where {T<:Real,F,G,H}
    return WiersigSmoothContour(center,halfwidth,halfheight,z,dz,inside,ownership)
end

# Construct the entire Fourier rounded rectangle z(θ)=center+sx(cosθ-η₃cos3θ+η₅cos5θ)+i sy(sinθ+η₃sin3θ+η₅sin5θ),
# with axis extrema `halfwidth` and `halfheight`. The default η₃=0.12,
# η₅=(9η₃-1)/25=0.0032 satisfies `-1+9η₃-25η₅=0`, flattening the contour
# near the horizontal and vertical sides while retaining an entire periodic
# parametrization. Numerical tests show the same exponential trapezoidal
# convergence with `nq` as for the circular contour.
function wiersig_fourier_rectangle_contour(center::Complex{T},halfwidth::T,halfheight::T;eta3::T=T(0.12),eta5::T=(T(9)*eta3-one(T))/T(25)) where {T<:Real}
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
    return WiersigSmoothContour(center,halfwidth,halfheight,z,dz,inside)
end

# Construct the elliptical smooth periodic Beyn contour.
function WiersigSmoothContour(center::Complex{T},halfwidth::T,halfheight::T) where {T<:Real}
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
    return WiersigSmoothContour(center,halfwidth,halfheight,z,dz,inside)
end

# Construct the entire circular contour z(θ)=center+radius*exp(iθ).
function WiersigSmoothContour(center::Complex{T},radius::T) where {T<:Real}
    return WiersigSmoothContour(center,radius,radius)
end
# Exact rectangular contours integrated edge-by-edge with Gauss-Legendre.
# Edge indices and signs are assigned by a rectangular tessellation.
struct WiersigRectangleContour{T<:Real}<:AbstractWiersigContour{T}
    center::Complex{T}
    halfwidth::T
    halfheight::T
    ownership::Union{Nothing,NTuple{4,T}}
    edges::NTuple{4,Int}       # bottom,right,top,left
    signs::NTuple{4,Int8}      # +,+,-,-
end

# general fallback constructor for a rectangle contour
function WiersigRectangleContour(center::Complex{T},halfwidth::T,halfheight::T) where {T<:Real}
    return WiersigRectangleContour(center,halfwidth,halfheight,nothing,(0,0,0,0),(Int8(0),Int8(0),Int8(0),Int8(0)))
end

# Leading dielectric Weyl estimate for the number of states whose real part lies across the horizontal span of one Beyn contour. The probe dimension is chosen as a safety factor times this leading-order count.
function _wiersig_beyn_probe_rank(solver::AbstractWiersigSolver,contour::AbstractWiersigContour{T};factor::Real=3.0,min_probe::Int=50) where {T<:Real}
    fundamental=!isnothing(solver.symmetry)
    kL=real(contour.center)-contour.halfwidth
    kR=real(contour.center)+contour.halfwidth
    C=length(solver.billiards);nin=_wiersig_component_indices(solver,C)
    Nest=zero(T)
    @inbounds for a in 1:C
        b=solver.billiards[a]
        Nest+=delta_area_count_estimate(b,nin[a]*kL,nin[a]*(kR-kL);fundamental=fundamental)
    end
    return max(min_probe,ceil(Int,factor*Nest))
end

# Return periodic trapezoidal nodes `z_j=z(θ_j)` and Beyn weights w_j=z'(θ_j)/(i nq), with `θ_j=2π(j-1)/nq` for smooth periodic contours.
function wiersig_beyn_contour(contour::WiersigSmoothContour{T},nq::I) where {T<:Real,I<:Integer}
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

# Return Gauss-Legendre nodes and Beyn weights for a rectangular contour. 
function wiersig_beyn_contour(contour::WiersigRectangleContour{T},nq::I) where {T<:Real,I<:Integer}
    return wiersig_beyn_contour(contour,(nq,nq))
end
function wiersig_beyn_contour(contour::WiersigRectangleContour{T},nq::NTuple{2,I}) where {T<:Real,I<:Integer}
    nh,nv=nq;nh>0&&nv>0||throw(ArgumentError("Gauss-Legendre orders must be positive"))
    c=contour.center;a=contour.halfwidth;b=contour.halfheight
    z0=(c-Complex{T}(a,b),c+Complex{T}(a,-b),c+Complex{T}(a,b),c+Complex{T}(-a,b))
    z1=(c+Complex{T}(a,-b),c+Complex{T}(a,b),c+Complex{T}(-a,b),c-Complex{T}(a,b))
    ns=(nh,nv,nh,nv);z=Complex{T}[];w=Complex{T}[]
    @inbounds for e in 1:4
        x,wg=gausslegendre(ns[e]);x=T.(x);wg=T.(wg);m=(z0[e]+z1[e])/T(2);d=(z1[e]-z0[e])/T(2)
        for j in eachindex(x);push!(z,m+d*x[j]);push!(w,d*wg[j]/Complex{T}(0,T(TWO_PI)));end
    end
    return z,w
end

# Determine whether k lies inside a smooth periodic contour. For the Fourier
# rectangle this requires a one-dimensional boundary inversion by bisection.
@inline function wiersig_inside_contour(contour::WiersigSmoothContour{T},k::Complex{T};tol=nothing) where {T<:Real}
    isnothing(tol)&&return contour.inside(k)
    tolT=T(tol)
    contour.inside(k)&&return true
    δ=k-contour.center;iszero(δ)&&return true
    scale=max(contour.halfwidth,contour.halfheight)
    return contour.inside(contour.center+δ*(one(T)-tolT/max(one(T),scale)))
end
# for rectangle to determine if k is inside is easy, just check the real and imaginary parts of k against the rectangle bounds.
@inline function wiersig_inside_contour(contour::WiersigRectangleContour{T},k::Complex{T};tol=nothing) where {T<:Real}
    τ=isnothing(tol) ? zero(T) : T(tol)
    δ=k-contour.center
    return abs(real(δ))<=contour.halfwidth+τ&&abs(imag(δ))<=contour.halfheight+τ
end

# One unique straight edge of a rectangular tessellation. Horizontal edges are
# canonically oriented left->right, vertical edges bottom->top. `cells` gives
# the one or two contours using the edge and `signs` their contour orientations.
struct WiersigRectangleEdge{T<:Real}
    z0::Complex{T}
    z1::Complex{T}
    cells::NTuple{2,Int}
    signs::NTuple{2,Int8}
end

# Gauss-Legendre quadrature nodes and weights on a single rectangle edge.
function wiersig_beyn_edge(edge::WiersigRectangleEdge{T},nq::I) where {T<:Real,I<:Integer}
    nq>0||throw(ArgumentError("nq must be positive"))
    x,wg=gausslegendre(nq);x=T.(x);wg=T.(wg)
    m=(edge.z0+edge.z1)/T(2);d=(edge.z1-edge.z0)/T(2)
    z=Vector{Complex{T}}(undef,nq);w=similar(z)
    @inbounds for j in 1:nq
        z[j]=m+d*x[j];w[j]=d*wg[j]/Complex{T}(0,T(TWO_PI))
    end
    return z,w
end

abstract type AbstractWiersigTessellation{T<:Real} end

# Smooth tessellation. Contours are independent and therefore have no shared-edge metadata.
struct WiersigSmoothTessellation{T<:Real,C<:AbstractVector{<:WiersigSmoothContour{T}}}<:AbstractWiersigTessellation{T}
    region::NTuple{4,T}
    contours::C
    nx::Int
    ny::Int
end

# Rectangular tessellation with globally unique edges. Neighboring contours
# reference the same edge index, so every shared-edge quadrature node is solved once.
struct WiersigRectangleTessellation{T<:Real}<:AbstractWiersigTessellation{T}
    region::NTuple{4,T}
    contours::Vector{WiersigRectangleContour{T}}
    edges::Vector{WiersigRectangleEdge{T}}
    nx::Int
    ny::Int
end

"""
    wiersig_contour_tessellation(re_min,re_max,im_min,im_max,seed::WiersigSmoothContour;overlap_re=0,overlap_im=0)

Cover the requested spectral rectangle by independent translated copies of a
smooth periodic contour. These contours are integrated separately and do not
share quadrature nodes.
"""
function wiersig_contour_tessellation(re_min::T,re_max::T,im_min::T,im_max::T,seed::WiersigSmoothContour{T};overlap_re::T=zero(T),overlap_im::T=zero(T)) where {T<:Real}
    0<=overlap_re<1||throw(ArgumentError("overlap_re must satisfy 0≤overlap_re<1"))
    0<=overlap_im<1||throw(ArgumentError("overlap_im must satisfy 0≤overlap_im<1"))
    W=re_max-re_min;H=im_max-im_min
    dxmax=T(2)*seed.halfwidth*(one(T)-overlap_re);dymax=T(2)*seed.halfheight*(one(T)-overlap_im)
    nx0=max(1,ceil(Int,W/dxmax));ny0=max(1,ceil(Int,H/dymax))
    best=typemax(Int);bestnx=0;bestny=0;nx=nx0
    while bestnx==0||nx*ny0<best
        dx=W/T(nx)
        if seed.inside(seed.center+Complex{T}(dx/T(2),zero(T)))
            ny=ny0
            while true
                dy=H/T(ny)
                if seed.inside(seed.center+Complex{T}(dx/T(2),dy/T(2)))
                    n=nx*ny
                    n<best&&(best=n;bestnx=nx;bestny=ny)
                    break
                end
                ny+=1
            end
        end
        nx+=1
    end
    bestnx>0||throw(ArgumentError("could not construct a covering contour tessellation"))
    dx=W/T(bestnx);dy=H/T(bestny)
    xs=T[re_min+(T(j)-T(0.5))*dx for j in 1:bestnx];ys=T[im_min+(T(j)-T(0.5))*dy for j in 1:bestny]
    contours=vec([begin
        center=Complex{T}(xs[ix],ys[iy]);z=θ->center+(seed.z(θ)-seed.center);dz=seed.dz
        inside=k->seed.inside(seed.center+(k-center))
        ownership=(re_min+T(ix-1)*dx,re_min+T(ix)*dx,im_min+T(iy-1)*dy,im_min+T(iy)*dy)
        WiersigSmoothContour(center,seed.halfwidth,seed.halfheight,z,dz,inside;ownership=ownership)
    end for ix in eachindex(xs),iy in eachindex(ys)])
    return WiersigSmoothTessellation((re_min,re_max,im_min,im_max),contours,bestnx,bestny)
end

"""
    wiersig_contour_tessellation(re_min,re_max,im_min,im_max,seed::WiersigRectangleContour)

Partition the requested spectral rectangle into exact rectangular Beyn contours.
All geometrically identical cell sides are represented by one global edge.
Interior edges therefore have two consumers with opposite orientations, while
outer boundary edges have one.
"""
function wiersig_contour_tessellation(re_min::T,re_max::T,im_min::T,im_max::T,seed::WiersigRectangleContour{T}) where {T<:Real}
    W=re_max-re_min;H=im_max-im_min
    nx=max(1,ceil(Int,W/(T(2)*seed.halfwidth)));ny=max(1,ceil(Int,H/(T(2)*seed.halfheight)))
    dx=W/T(nx);dy=H/T(ny)
    xs=T[re_min+T(i)*dx for i in 0:nx];ys=T[im_min+T(j)*dy for j in 0:ny]
    nh=nx*(ny+1)
    hid(ix,iy)=iy*nx+ix
    vid(ix,iy)=nh+(iy-1)*(nx+1)+ix+1
    edges=Vector{WiersigRectangleEdge{T}}(undef,nh+(nx+1)*ny)
    contours=Vector{WiersigRectangleContour{T}}(undef,nx*ny)
    @inbounds for iy in 0:ny,ix in 1:nx
        below=iy==0 ? 0 : (iy-1)*nx+ix
        above=iy==ny ? 0 : iy*nx+ix
        edges[hid(ix,iy)]=WiersigRectangleEdge(Complex{T}(xs[ix],ys[iy+1]),Complex{T}(xs[ix+1],ys[iy+1]),(below,above),(Int8(-1),Int8(1)))
    end
    @inbounds for iy in 1:ny,ix in 0:nx
        left=ix==0 ? 0 : (iy-1)*nx+ix
        right=ix==nx ? 0 : (iy-1)*nx+ix+1
        edges[vid(ix,iy)]=WiersigRectangleEdge(Complex{T}(xs[ix+1],ys[iy]),Complex{T}(xs[ix+1],ys[iy+1]),(left,right),(Int8(1),Int8(-1)))
    end
    @inbounds for iy in 1:ny,ix in 1:nx
        c=(iy-1)*nx+ix
        center=Complex{T}((xs[ix]+xs[ix+1])/T(2),(ys[iy]+ys[iy+1])/T(2))
        ownership=(xs[ix],xs[ix+1],ys[iy],ys[iy+1])
        contours[c]=WiersigRectangleContour(center,dx/T(2),dy/T(2),ownership,(hid(ix,iy-1),vid(ix,iy),hid(ix,iy),vid(ix-1,iy)),(Int8(1),Int8(1),Int8(-1),Int8(-1)))
    end
    return WiersigRectangleTessellation((re_min,re_max,im_min,im_max),contours,edges,nx,ny)
end

"""
    _wiersig_beyn_matrix_batch_plan(::Type{T},N,nmat;ram_cap_gib=nothing,ram_fraction=0.75,reserve_gib=8.0)

Choose the largest dense contour-matrix batch allowed by the matrix-storage RAM
budget.

When `ram_cap_gib=nothing`, the budget is

    ram_fraction*Sys.total_memory()-reserve_gib,

so it is based on total physical RAM rather than instantaneous free memory.
`ram_cap_gib` overrides this automatic budget.

Returns the selected `batch_size` together with the single-matrix, total-RAM,
and matrix-budget byte counts.
"""
function _wiersig_beyn_matrix_batch_plan(::Type{T},N::Int,nmat::Int;ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75,reserve_gib::Real=8.0) where {T<:Real}
    matrix_bytes=N*N*sizeof(Complex{T}) # usually it wont run in higher precision.
    total_bytes=Int(Sys.total_memory())
    reserve_bytes=floor(Int,reserve_gib*2.0^30)
    budget_bytes=isnothing(ram_cap_gib) ? floor(Int,ram_fraction*total_bytes)-reserve_bytes : floor(Int,ram_cap_gib*2.0^30)
    budget_bytes>=matrix_bytes||throw(ArgumentError("RAM budget too small for one dense Wiersig matrix"))
    B=clamp(budget_bytes÷matrix_bytes,1,nmat)
    return (batch_size=B,matrix_bytes=matrix_bytes,total_bytes=total_bytes,budget_bytes=budget_bytes)
end

struct WiersigBeynBuffers{T<:Real}
    V::Matrix{Complex{T}}
    X::Matrix{Complex{T}}
    A0::Matrix{Complex{T}}
    A1::Matrix{Complex{T}}
    A::Matrix{Complex{T}}
    y::Vector{Complex{T}}
    As::Vector{Matrix{Complex{T}}}
end

# allocate reusable dense storage for one _compute_spectrum. All contours/cells
# in the interval have the same boundary discretization and probe dimension.
# `As` stores the reusable batch of dense Wiersig matrices used by Chebyshev assembly.
function WiersigBeynBuffers(::Type{T},N::Int,r::Int,nmat::Int,rng::AbstractRNG;ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    V=randn(rng,Complex{T},N,r);X=similar(V);A0=zeros(Complex{T},N,r);A1=zeros(Complex{T},N,r)
    A=Matrix{Complex{T}}(undef,N,N);y=Vector{Complex{T}}(undef,N)
    if nmat>0 # if chebyshev, otherwise direct requires not As buffers
        mem=_wiersig_beyn_matrix_batch_plan(T,N,nmat;ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
        As=[Matrix{Complex{T}}(undef,N,N) for _ in 1:mem.batch_size]
    else
        As=Matrix{Complex{T}}[]
    end
    return WiersigBeynBuffers(V,X,A0,A1,A,y,As)
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
    _wiersig_beyn_singular_validation_linear!(validator,inside,σeff,checked,keep;validation_padding=5)

Reference adaptive validator. Check enclosed candidates in increasing `σeff`.
After a rejection, continue until `validation_padding` consecutive checked
candidates pass. This is the original production algorithm and is retained as
a debugging/reference implementation.
"""
function _wiersig_beyn_singular_validation_linear!(validator,inside::BitVector,σeff::AbstractVector{T},checked::BitVector,keep::BitVector;validation_padding::Int=5) where {T<:Real}
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

######
# TODO: The linear search is simpler and does not assume monotonic residual
# acceptance. The binary search is much faster for large candidate sets but
# assumes a single reject -> accept transition in increasing σeff. A violation
# detected in the local confirmation window triggers complete validation;
# undetected non-monotonicity remains possible. Use `:check` to verify the
# binary inference against complete validation.

"""
    _wiersig_beyn_singular_validation_binary!(validator,inside,σeff,checked,keep;validation_padding=5)

Production adaptive validator. Locate that transition by binary search, requiring O(log M)
candidate checks for M enclosed candidates. After the first passing candidate is located, up to `validation_padding`
consecutive candidates are checked as a local confirmation. If this confirmation
finds a rejection, monotonicity has already been falsified and all enclosed
candidates are validated explicitly. Unchecked candidates on the rejected side are marked rejected; unchecked
candidates on the accepted side retain their initial `keep=true` state.
"""
function _wiersig_beyn_singular_validation_binary!(validator,inside::BitVector,σeff::AbstractVector{T},checked::BitVector,keep::BitVector;validation_padding::Int=5) where {T<:Real}
    validation_padding>0||throw(ArgumentError("validation_padding must be positive"))
    order=findall(inside)
    sort!(order;by=j->σeff[j])
    M=length(order)
    M==0&&return order
    @inline function checkpos(p::Int)
        j=order[p]
        checked[j]||validator(Int[j])
        return keep[j]
    end
    # If the weakest candidate passes, monotonicity predicts that all pass.
    if checkpos(1)
        last=min(M,validation_padding)
        last>1&&validator(Vector{Int}(@view order[2:last]))
        if any(p->!keep[order[p]],1:last)
            validator(order)
        end
        return order
    end
    # One candidate and it failed.
    if M==1
        return order
    end
    # If the strongest candidate also fails, monotonicity predicts all reject.
    if !checkpos(M)
        @inbounds for j in order
            keep[j]=false
        end
        return order
    end
    # Invariant: order[lo] rejects, order[hi] passes.
    lo=1
    hi=M
    while hi-lo>1
        mid=(lo+hi)>>>1
        if checkpos(mid)
            hi=mid
        else
            lo=mid
        end
    end
    # Under the monotone model every candidate before `hi` rejects.
    @inbounds for p in 1:hi-1
        keep[order[p]]=false
    end
    # Confirm a short accepted run at the transition. A failure here proves
    # that the monotone model is invalid, so fall back to complete validation.
    last=min(M,hi+validation_padding-1)
    confirm=Int[]
    @inbounds for p in hi:last
        j=order[p]
        checked[j]||push!(confirm,j)
    end
    isempty(confirm)||validator(confirm)
    if any(p->!keep[order[p]],hi:last)
        validator(order)
    end
    return order
end

"""
    _wiersig_beyn_singular_validation!(validator,inside,σeff,checked,keep;validation_padding=5,method=:binary)

Adaptive residual validation method.

- `:binary`: O(log M + validation_padding) production search under the observed
  reject -> accept ordering in increasing `σeff`.
- `:linear`: original padded linear search, retained as a reference/debug path.
- `:check`: run the binary algorithm, then validate every enclosed candidate and
  assert that its inferred accepted-root set was exact.
"""
function _wiersig_beyn_singular_validation!(validator,inside::BitVector,σeff::AbstractVector{T},checked::BitVector,keep::BitVector;validation_padding::Int=5,method::Symbol=:binary) where {T<:Real}
    method===:binary&&return _wiersig_beyn_singular_validation_binary!(validator,inside,σeff,checked,keep;validation_padding=validation_padding)
    method===:linear&&return _wiersig_beyn_singular_validation_linear!(validator,inside,σeff,checked,keep;validation_padding=validation_padding)
    method===:check||throw(ArgumentError("adaptive_validation_method must be :binary, :linear, or :check"))
    order=_wiersig_beyn_singular_validation_binary!(validator,inside,σeff,checked,keep;validation_padding=validation_padding)
    inferred=copy(keep)
    unchecked=Int[]
    @inbounds for j in order
        checked[j]||push!(unchecked,j)
    end
    isempty(unchecked)||validator(unchecked)
    bad=Int[]
    @inbounds for j in order
        inferred[j]==keep[j]||push!(bad,j)
    end
    isempty(bad)||error("binary adaptive validation disagrees with complete validation at $(length(bad)) candidate(s): $(bad)")
    return order
end

#########

"""
    _wiersig_beyn_rank(Σ::AbstractVector{T},svd_tol::T,relative_svd_tol::Bool) where {T<:Real}

Determine the numerical rank of A₀. Relative mode retains `σ_j≥svd_tol*σ₁`; absolute mode retains `σ_j≥svd_tol`.
"""
@inline function _wiersig_beyn_rank(Σ::AbstractVector{T},svd_tol::T,relative_svd_tol::Bool) where {T<:Real}
    isempty(Σ)&&return 0,svd_tol
    threshold=relative_svd_tol ? svd_tol*Σ[1] : svd_tol
    return count(σ->σ>=threshold,Σ),threshold
end

function _wiersig_beyn_rank_plateaus(ranks::AbstractVector{<:Integer})
    isempty(ranks)&&return NamedTuple[]
    out=NamedTuple[]
    first=1
    @inbounds for i in 2:length(ranks)+1
        if i>length(ranks)||ranks[i]!=ranks[first]
            push!(out,(rank=ranks[first],first=first,last=i-1,length=i-first))
            first=i
        end
    end
    return out
end

function _wiersig_beyn_stable_rank(ranks::AbstractVector{<:Integer})
    ps=_wiersig_beyn_rank_plateaus(ranks)
    best=nothing
    @inbounds for p in ps
        p.rank>0&&p.length>=2||continue
        if isnothing(best)||p.length>best.length||(p.length==best.length&&p.rank<best.rank)
            best=p
        end
    end
    isnothing(best)&&return nothing
    idx=(best.first+best.last)÷2
    return (rank=best.rank,index=idx,plateau=best)
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
    tols0=svd_tol isa AbstractVector ? collect(svd_tol) : T[svd_tol]
    isempty(tols0)&&throw(ArgumentError("svd_tol must not be empty"));issorted(tols0;rev=true)||throw(ArgumentError("svd_tol must be nonincreasing"))
    N,ravailable=size(A0);rmax=min(max_r,ravailable);rcur=min(r,rmax)
    while true
        A0cur=Matrix(@view A0[:,1:rcur]);@blas_multi_then_1 MAX_BLAS_THREADS F0=svd!(A0cur;full=false);Σ=F0.S
        ranks0=Vector{Int}(undef,length(tols0));thresholds0=Vector{T}(undef,length(tols0))
        @inbounds for i in eachindex(tols0);ranks0[i],thresholds0[i]=_wiersig_beyn_rank(Σ,tols0[i],relative_svd_tol);end
        if verbose
            println("Beyn probe dimension         = ",rcur);println("Beyn moment singular values = ");println(Σ)
            println("SVD tolerances              = ",tols0);println("detected moment ranks       = ",ranks0);println("rank thresholds             = ",thresholds0)
        end
        isat=findfirst(==(rcur),ranks0)
        if isat==1
            rcur>=rmax&&throw(ArgumentError("Beyn moment rank saturates probe=$rcur already at svd_tol=$(tols0[1]). Increase the probe factor or reduce the contour size."))
            rcur=min(rcur+r_step,rmax);continue
        end
        nuse=isnothing(isat) ? length(tols0) : isat-1
        tols=tols0[1:nuse];ranks=ranks0[1:nuse];thresholds=thresholds0[1:nuse];rkmax=maximum(ranks)
        if verbose&&!isnothing(isat)
            println("SVD ladder truncated         = ",tols0[isat]," and below reached probe ceiling ",rcur)
            println("usable SVD tolerances       = ",tols);println("usable moment ranks         = ",ranks)
        end
        rkmax==0&&return (B=Matrix{Complex{T}}(undef,0,0),U=Matrix{Complex{T}}(undef,N,0),singular_values=copy(Σ),rank=0,ranks=ranks,rank_threshold=thresholds[1],rank_thresholds=thresholds,svd_tolerances=tols,probe_dimension=rcur)
        Uk=@view F0.U[:,1:rkmax];Wk=@view F0.V[:,1:rkmax];Σk=@view Σ[1:rkmax];A1cur=@view A1[:,1:rcur]
        tmp=Matrix{Complex{T}}(undef,N,rkmax);@blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp,A1cur,Wk)
        @inbounds for j in 1:rkmax;@views rmul!(tmp[:,j],inv(Σk[j]));end
        B=Matrix{Complex{T}}(undef,rkmax,rkmax);@blas_multi_then_1 MAX_BLAS_THREADS mul!(B,adjoint(Uk),tmp)
        return (B=B,U=Matrix(Uk),singular_values=copy(Σ),rank=ranks[1],ranks=ranks,rank_threshold=thresholds[1],rank_thresholds=thresholds,svd_tolerances=tols,probe_dimension=rcur)
    end
end

# direct matrix constrction Beyn contour accumulator to A0 and A1 using a preallocated V.
function _wiersig_beyn_accumulate_direct!(A0::Matrix{Complex{T}},A1::Matrix{Complex{T}},V::Matrix{Complex{T}},X::Matrix{Complex{T}},A::Matrix{Complex{T}},solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}};dlp_kernel::Symbol=:source,multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    xv=vec(X);a0v=vec(A0);a1v=vec(A1)
    p=verbose ? Progress(length(z),desc="Beyn contour") : nothing
    @inbounds for j in eachindex(z)
        construct_matrices!(solver,A,pts,ws,z[j];dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        F=lu!(A,ws;check=false);ldiv!(X,F,V)
        BLAS.axpy!(w[j],xv,a0v)
        BLAS.axpy!(w[j]*z[j],xv,a1v)
        verbose&&next!(p)
    end
    return nothing
end
# accumulate Beyn moments using a preallocated V and preassembled As. This is used for Chebyshev interpolation of the contour matrices. It is written in this way to to allow for clean batch code bellow. This one does not have verbose printing since no matrices are constructed here.
function _wiersig_beyn_accumulate_chebyshev!(A0::Matrix{Complex{T}},A1::Matrix{Complex{T}},V::Matrix{Complex{T}},X::Matrix{Complex{T}},As::AbstractVector{<:Matrix{Complex{T}}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}}) where {T<:Real}
    xv=vec(X);a0v=vec(A0);a1v=vec(A1)
    @inbounds for j in eachindex(As)
        F=lu!(As[j],ws;check=false);ldiv!(X,F,V)
        BLAS.axpy!(w[j],xv,a0v)
        BLAS.axpy!(w[j]*z[j],xv,a1v)
    end
    return nothing
end

# Accumulate the production Beyn moments using only the requested `nq` contour rule.
function _wiersig_beyn_build_direct(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    N=boundary_matrix_size(ws);rmax=min(max_r,N)
    V=randn(rng,Complex{T},N,rmax);X=similar(V);A0=zeros(Complex{T},N,rmax);A1=zeros(Complex{T},N,rmax);A=Matrix{Complex{T}}(undef,N,N)
    _wiersig_beyn_accumulate_direct!(A0,A1,V,X,A,solver,pts,ws,z,w;dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
    return _wiersig_beyn_build_reduced_problem(A0,A1;r=r,r_step=r_step,max_r=rmax,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
end

struct WiersigBeynEdgeMoments{T<:Real}
    A0::Matrix{Complex{T}}
    A1::Matrix{Complex{T}}
end
# for each edge calculate the A0,A1 moments. These are shared by the neighboring contour.
function _wiersig_beyn_edge_direct(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,edge::WiersigRectangleEdge{T},nq::Int,V::Matrix{Complex{T}},X::Matrix{Complex{T}},A::Matrix{Complex{T}};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    N,r=size(V);z,w=wiersig_beyn_edge(edge,nq)
    A0=zeros(Complex{T},N,r);A1=zeros(Complex{T},N,r)
    xv=vec(X);a0v=vec(A0);a1v=vec(A1)
    @inbounds for j in eachindex(z)
        construct_matrices!(solver,A,pts,ws,z[j];dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        F=lu!(A,ws;check=false);ldiv!(X,F,V)
        BLAS.axpy!(w[j],xv,a0v);BLAS.axpy!(w[j]*z[j],xv,a1v)
    end
    return WiersigBeynEdgeMoments(A0,A1)
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

function _wiersig_beyn_validate_direct!(raw::Vector{T},normalized::Vector{T},checked::BitVector,keep::BitVector,idx::Vector{Int},λ::Vector{Complex{T}},Φ::Matrix{Complex{T}},solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,A::Matrix{Complex{T}},y::Vector{Complex{T}};res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    isempty(idx)&&return nothing
    @inbounds for j in idx
        raw[j],normalized[j]=_wiersig_beyn_residual!(A,y,solver,pts,ws,λ[j],@view(Φ[:,j]);dlp_kernel=dlp_kernel,matnorm=matnorm,multithreaded=multithreaded)
        checked[j]=true;keep[j]=(!filter_raw_residual||raw[j]<res_tol)&&normalized[j]<normalized_res_tol
        verbose&&println("adaptive candidate: k=",λ[j],", raw=",raw[j],", normalized=",normalized[j],", kept=",keep[j])
    end
    return nothing
end

function _wiersig_beyn_validate_direct!(raw::Vector{T},normalized::Vector{T},checked::BitVector,keep::BitVector,idx::Vector{Int},λ::Vector{Complex{T}},Φ::Matrix{Complex{T}},solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace;kwargs...) where {T<:Real}
    N=boundary_matrix_size(ws);A=Matrix{Complex{T}}(undef,N,N);y=Vector{Complex{T}}(undef,N)
    return _wiersig_beyn_validate_direct!(raw,normalized,checked,keep,idx,λ,Φ,solver,pts,ws,A,y;kwargs...)
end

"""
    construct_wiersig_B_matrix(...)

Construct the direct reduced Beyn matrix `B=U_r*A₁W_rΣ_r⁻¹` together with the retained invariant-subspace basis, moment singular values, detected rank, final probe width and contour quadrature data.
"""
function construct_wiersig_B_matrix(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::AbstractWiersigContour{T};nq=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
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
    wiersig_beyn(...)

Direct Beyn solve using a single `nq`-point production contour quadrature.

For a vector of decreasing `svd_tol` values, each distinct numerical rank
defines a complete nested reduced Beyn problem. These spectra are never merged.
Each rank is validated independently. The spectrum with the largest number of
accepted enclosed roots is selected. If several ranks give the same accepted
root count, preference is given to a numerical rank which forms a stable
plateau across consecutive SVD tolerances. The middle tolerance of that plateau
is reported as the representative SVD tolerance.

With `adaptive_validation=true`, enclosed candidates at each rank are ordered
by increasing effective singular value

    σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]⁻¹ᐟ².

`adaptive_validation_method=:binary` assumes that residual acceptance has a
single reject -> accept transition in this ordering and locates it by binary
search. This requires O(log M) residual evaluations for M enclosed candidates,
plus at most `validation_padding` local confirmation checks.

`:linear` retains the original padded sequential algorithm for debugging.
`:check` runs the binary algorithm and subsequently validates every enclosed
candidate, throwing an error if the inferred keep-set differs from the complete
residual test.

`validate_roots=true` validates every enclosed candidate independently of the
adaptive-validation method.
"""
function wiersig_beyn(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::AbstractWiersigContour{T};nq=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,validate_roots::Bool=false,adaptive_validation::Bool=true,adaptive_validation_method::Symbol=:binary,validation_padding::Int=5,res_tol::T=T(1e-8),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    adaptive_validation_method in (:binary,:linear,:check)||throw(ArgumentError("adaptive_validation_method must be :binary, :linear, or :check"))
    reduced=construct_wiersig_B_matrix(solver,pts,ws,contour;nq=nq,r=r,r_step=r_step,max_r=max_r,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose)
    N=boundary_matrix_size(ws)
    if maximum(reduced.ranks;init=0)==0
        empty=(values=Complex{T}[],vectors=Matrix{Complex{T}}(undef,N,0),residuals=T[],normalized_residuals=T[],effective_singular_values=T[],checked=Bool[],all_values=Complex{T}[],all_vectors=Matrix{Complex{T}}(undef,N,0),all_residuals=T[],all_normalized_residuals=T[],all_effective_singular_values=T[],all_checked=Bool[],inside=Bool[],kept=Bool[])
        common=(rank=0,ranks=reduced.ranks,svd_tolerances=reduced.svd_tolerances,selected_svd_tolerance=reduced.svd_tolerances[1],selected_svd_index=1,rank_threshold=reduced.rank_thresholds[1],rank_thresholds=reduced.rank_thresholds,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=:none,adaptive_validation_method=adaptive_validation_method)
        return merge(empty,common)
    end
    stable=_wiersig_beyn_stable_rank(reduced.ranks)
    preferred_rank=isnothing(stable) ? 0 : stable.rank
    selected=nothing;bestcount=-1;selected_it=0
    @inbounds for it in eachindex(reduced.ranks)
        rk=reduced.ranks[it]
        rk==0&&continue
        it>1&&rk==reduced.ranks[it-1]&&continue
        E=nothing
        @blas_multi_then_1 MAX_BLAS_THREADS E=eigen(Matrix(@view reduced.B[1:rk,1:rk]))
        λ=Vector{Complex{T}}(E.values);Y=Matrix{Complex{T}}(E.vectors)
        Φ=Matrix{Complex{T}}(undef,N,length(λ))
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φ,@view(reduced.U[:,1:rk]),Y)
        σeff=_wiersig_beyn_effective_sigma(Y,@view reduced.singular_values[1:rk])
        nroots=length(λ);inside=falses(nroots);keep=falses(nroots);checked=falses(nroots)
        raw=fill(T(NaN),nroots);normalized=fill(T(NaN),nroots)
        for j in eachindex(λ)
            inside[j]=isfinite(real(λ[j]))&&isfinite(imag(λ[j]))&&wiersig_inside_contour(contour,λ[j])
            keep[j]=inside[j]
        end
        inside_idx=findall(inside)
        if validate_roots
            _wiersig_beyn_validate_direct!(raw,normalized,checked,keep,inside_idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
        elseif adaptive_validation&&!isempty(inside_idx)
            validator=idx->_wiersig_beyn_validate_direct!(raw,normalized,checked,keep,idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
            order=_wiersig_beyn_singular_validation!(validator,inside,σeff,checked,keep;validation_padding=validation_padding,method=adaptive_validation_method)
            verbose&&println("SVD tolerance ",reduced.svd_tolerances[it],": rank=",rk,", checked=",count(checked),", rejected=",count(inside .& .!keep),", σeff first/last=",isempty(order) ? T(NaN) : σeff[first(order)]," / ",isempty(order) ? T(NaN) : σeff[last(order)])
        end
        naccepted=count(keep);nrejected=count(inside .& .!keep)
        better=naccepted>bestcount
        preferred=naccepted==bestcount&&rk==preferred_rank&&(selected_it==0||reduced.ranks[selected_it]!=preferred_rank)
        if better||preferred
            bestcount=naccepted;selected_it=it
            selected=(λ=λ,Φ=Φ,σeff=σeff,raw=raw,normalized=normalized,inside=inside,checked=checked,keep=keep)
        end
        verbose&&println("SVD rank spectrum: tolerance=",reduced.svd_tolerances[it],", rank=",rk,", enclosed=",count(inside),", accepted=",naccepted,", rejected=",nrejected,", selected=",selected_it==it)
    end
    if !isnothing(stable)&&reduced.ranks[selected_it]==stable.rank
        selected_it=stable.index
    end
    λ=selected.λ;Φ=selected.Φ;σeff=selected.σeff
    raw=selected.raw;normalized=selected.normalized
    inside=selected.inside;checked=selected.checked;keep=selected.keep
    rk=reduced.ranks[selected_it]
    idx=findall(keep)
    !isempty(idx)&&(idx=idx[sortperm(idx;by=j->(real(λ[j]),imag(λ[j])))])
    method=validate_roots ? :direct_all : adaptive_validation ? Symbol(:direct_singular_,adaptive_validation_method) : :none
    candidates=(values=λ[idx],vectors=Φ[:,idx],residuals=raw[idx],normalized_residuals=normalized[idx],effective_singular_values=σeff[idx],checked=checked[idx],all_values=λ,all_vectors=Φ,all_residuals=raw,all_normalized_residuals=normalized,all_effective_singular_values=σeff,all_checked=checked,inside=inside,kept=keep)
    common=(rank=rk,ranks=reduced.ranks,svd_tolerances=reduced.svd_tolerances,selected_svd_tolerance=reduced.svd_tolerances[selected_it],selected_svd_index=selected_it,rank_threshold=reduced.rank_thresholds[selected_it],rank_thresholds=reduced.rank_thresholds,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,adaptive_validation_method=adaptive_validation_method,validation_method=method)
    verbose&&println("selected SVD tolerance       = ",reduced.svd_tolerances[selected_it]," (rank ",rk,", accepted ",bestcount,")")
    return merge(candidates,common)
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
"""
function _wiersig_beyn_build_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,z::AbstractVector{Complex{T}},w::AbstractVector{Complex{T}},cws::WiersigChebyshevWorkspace{T};r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
    _wiersig_dlp_normal_mode(dlp_kernel)
    N=boundary_matrix_size(ws);rmax=min(max_r,N);nq=length(z)
    V=randn(rng,Complex{T},N,rmax);X=similar(V);A0=zeros(Complex{T},N,rmax);A1=zeros(Complex{T},N,rmax)
    mem=_wiersig_beyn_matrix_batch_plan(T,N,nq;ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction);B=mem.batch_size
    if verbose
        println("total physical RAM           = ",round(mem.total_bytes/2.0^30,digits=2)," GiB")
        println("matrix RAM budget            = ",round(mem.budget_bytes/2.0^30,digits=2)," GiB")
        println("matrix storage mode          = ",B==nq ? "all-k" : B==1 ? "streamed" : "batched")
        println("matrix batch size            = ",B," / ",nq)
    end
    As=[Matrix{Complex{T}}(undef,N,N) for _ in 1:B];p=verbose ? Progress(nq,desc="Beyn contour") : nothing
    for first in 1:B:nq
        last=min(first+B-1,nq);js=first:last;nb=length(js)
        work=nb==nq ? cws : _wiersig_subset_chebyshev_workspace(cws,js);Asb=nb==B ? As : As[1:nb]
        @benchit timeit=verbose "Chebyshev matrix batch" construct_matrices!(solver,Asb,pts,work;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        _wiersig_beyn_accumulate_chebyshev!(A0,A1,V,X,Asb,ws,@view(z[js]),@view(w[js]))
        verbose&&next!(p;step=nb)
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
first Beyn moments.
"""
function construct_wiersig_B_matrix_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::AbstractWiersigContour{T};nq=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),cheb_verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75) where {T<:Real}
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
function _wiersig_beyn_validate_chebyshev!(raw::Vector{T},normalized::Vector{T},checked::BitVector,keep::BitVector,idx::Vector{Int},λ::Vector{Complex{T}},Φ::Matrix{Complex{T}},solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace;res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,multithreaded::Bool=true,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=10_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),verbose::Bool=false) where {T<:Real}
    isempty(idx)&&return nothing
    ks=Complex{T}[λ[j] for j in idx];N=boundary_matrix_size(ws)
    cws=build_chebyshev_workspace(solver,pts,ks;npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=false)
    As=[Matrix{Complex{T}}(undef,N,N) for _ in idx]
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

# Compute all currently missing edge moments of one rectangular cell with one
# Chebyshev workspace. The workspace therefore contains at most one contour's
# quadrature nodes; individual edge moments are returned for caching/reuse.
function _wiersig_beyn_edges_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,edges::Vector{WiersigRectangleEdge{T}},eids::Vector{Int},nh::Int,nv::Int,V::Matrix{Complex{T}},X::Matrix{Complex{T}},As::Vector{Matrix{Complex{T}}};dlp_kernel::Symbol=:source,multithreaded::Bool=true,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=10_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),cheb_verbose::Bool=false,verbose::Bool=false) where {T<:Real}
    isempty(eids)&&return WiersigBeynEdgeMoments{T}[],0
    N,r=size(V);z=Complex{T}[];w=Complex{T}[];owner=Int[]
    moments=[WiersigBeynEdgeMoments(zeros(Complex{T},N,r),zeros(Complex{T},N,r)) for _ in eids]
    @inbounds for (l,eid) in enumerate(eids)
        edge=edges[eid];ne=iszero(imag(edge.z1-edge.z0)) ? nh : nv
        ze,we=wiersig_beyn_edge(edge,ne)
        append!(z,ze);append!(w,we)
        for _ in 1:ne;push!(owner,l);end
    end
    nk=length(z)
    @benchit timeit=verbose "Chebyshev cell workspace" cws=build_chebyshev_workspace(solver,pts,z;npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=cheb_verbose)
    B=min(length(As),nk);xv=vec(X)
    for first in 1:B:nk
        last=min(first+B-1,nk);js=first:last;nb=length(js)
        work=nb==nk ? cws : _wiersig_subset_chebyshev_workspace(cws,js)
        Asb=nb==B ? As : As[1:nb]
        @benchit timeit=verbose "Chebyshev cell batch" construct_matrices!(solver,Asb,pts,work;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        @inbounds for (l,j) in enumerate(js)
            F=lu!(Asb[l],ws;check=false);ldiv!(X,F,V)
            Em=moments[owner[j]]
            BLAS.axpy!(w[j],xv,vec(Em.A0))
            BLAS.axpy!(w[j]*z[j],xv,vec(Em.A1))
        end
    end
    return moments,nk
end

"""
    wiersig_beyn_chebyshev(...)

Solve the dielectric resonance problem with Beyn's contour method using multi-k
Chebyshev matrix assembly.

For a vector of decreasing `svd_tol` values, every resulting numerical rank
defines a complete nested reduced Beyn problem. Spectra from different ranks
are never merged. Each rank is validated independently and the returned
spectrum is the complete spectrum at the lowest SVD tolerance which strictly
increased the number of accepted enclosed roots.

Consequently a lower SVD tolerance which only exposes weak numerical directions
without adding validated resonances cannot pollute the returned spectrum, and
no distance-based matching between successive reduced spectra is required.

At every rank,

    σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]⁻¹ᐟ²

`adaptive_validation_method=:binary` is the production default and locates the
observed reject -> accept transition in increasing `σeff` by binary search.
`:linear` retains the original padded sequential validator and `:check` verifies
the binary inference against complete residual validation.
"""
function wiersig_beyn_chebyshev(solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::AbstractWiersigContour{T};nq=64,r::Int=16,r_step::Int=r,max_r::Int=min(boundary_matrix_size(ws),4*r),svd_tol::Union{T,AbstractVector{T}}=T(1e-12),relative_svd_tol::Bool=true,validate_roots::Bool=false,adaptive_validation::Bool=true,validation_padding::Int=5,res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-8),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,rng::AbstractRNG=MersenneTwister(0),multithreaded::Bool=true,verbose::Bool=false,return_workspace::Bool=false,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=10_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),cheb_verbose::Bool=false,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75,adaptive_validation_method::Symbol=:binary) where {T<:Real}
    adaptive_validation_method in (:binary,:linear,:check)||throw(ArgumentError("adaptive_validation_method must be :binary, :linear, or :check"))
    reduced=construct_wiersig_B_matrix_chebyshev(solver,pts,ws,contour;nq=nq,r=r,r_step=r_step,max_r=max_r,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,dlp_kernel=dlp_kernel,rng=rng,multithreaded=multithreaded,verbose=verbose,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,cheb_verbose=cheb_verbose,ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
    N=boundary_matrix_size(ws)
    if maximum(reduced.ranks;init=0)==0
        empty=(values=Complex{T}[],vectors=Matrix{Complex{T}}(undef,N,0),residuals=T[],normalized_residuals=T[],effective_singular_values=T[],checked=Bool[],all_values=Complex{T}[],all_vectors=Matrix{Complex{T}}(undef,N,0),all_residuals=T[],all_normalized_residuals=T[],all_effective_singular_values=T[],all_checked=Bool[],inside=Bool[],kept=Bool[])
        common=(rank=0,ranks=reduced.ranks,svd_tolerances=reduced.svd_tolerances,selected_svd_tolerance=reduced.svd_tolerances[1],selected_svd_index=1,rank_threshold=reduced.rank_thresholds[1],rank_thresholds=reduced.rank_thresholds,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=:none,adaptive_validation_method=adaptive_validation_method)
        return return_workspace ? merge(empty,common,(cheb_workspace=reduced.cheb_workspace,)) : merge(empty,common)
    end
    stable=_wiersig_beyn_stable_rank(reduced.ranks)
    preferred_rank=isnothing(stable) ? 0 : stable.rank
    selected=nothing;bestcount=-1;selected_it=0
    @inbounds for it in eachindex(reduced.ranks)
        rk=reduced.ranks[it]
        rk==0&&continue
        it>1&&rk==reduced.ranks[it-1]&&continue
        E=nothing
        @blas_multi_then_1 MAX_BLAS_THREADS E=eigen(Matrix(@view reduced.B[1:rk,1:rk]))
        λ=Vector{Complex{T}}(E.values);Y=Matrix{Complex{T}}(E.vectors)
        Φ=Matrix{Complex{T}}(undef,N,length(λ))
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φ,@view(reduced.U[:,1:rk]),Y)
        σeff=_wiersig_beyn_effective_sigma(Y,@view reduced.singular_values[1:rk])
        nroots=length(λ);inside=falses(nroots);keep=falses(nroots);checked=falses(nroots)
        raw=fill(T(NaN),nroots);normalized=fill(T(NaN),nroots)
        for j in eachindex(λ)
            inside[j]=isfinite(real(λ[j]))&&isfinite(imag(λ[j]))&&wiersig_inside_contour(contour,λ[j])
            keep[j]=inside[j]
        end
        inside_idx=findall(inside)
        if validate_roots
            _wiersig_beyn_validate_chebyshev!(raw,normalized,checked,keep,inside_idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=verbose)
        elseif adaptive_validation&&!isempty(inside_idx)
            validator=idx->_wiersig_beyn_validate_chebyshev!(raw,normalized,checked,keep,idx,λ,Φ,solver,pts,ws;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=verbose)
            order=_wiersig_beyn_singular_validation!(validator,inside,σeff,checked,keep;validation_padding=validation_padding,method=adaptive_validation_method)
            verbose&&println("SVD tolerance ",reduced.svd_tolerances[it],": rank=",rk,", checked=",count(checked),", rejected=",count(inside .& .!keep),", σeff first/last=",isempty(order) ? T(NaN) : σeff[first(order)]," / ",isempty(order) ? T(NaN) : σeff[last(order)])
        end
        naccepted=count(keep);nrejected=count(inside .& .!keep)
        better=naccepted>bestcount
        preferred=naccepted==bestcount&&rk==preferred_rank&&(selected_it==0||reduced.ranks[selected_it]!=preferred_rank)
        if better||preferred
            bestcount=naccepted;selected_it=it
            selected=(λ=λ,Φ=Φ,σeff=σeff,raw=raw,normalized=normalized,inside=inside,checked=checked,keep=keep)
        end
        verbose&&println("SVD rank spectrum: tolerance=",reduced.svd_tolerances[it],", rank=",rk,", enclosed=",count(inside),", accepted=",naccepted,", rejected=",nrejected,", selected=",selected_it==it)
    end
    if !isnothing(stable)&&reduced.ranks[selected_it]==stable.rank
        selected_it=stable.index
    end
    λ=selected.λ;Φ=selected.Φ;σeff=selected.σeff
    raw=selected.raw;normalized=selected.normalized
    inside=selected.inside;checked=selected.checked;keep=selected.keep
    rk=reduced.ranks[selected_it]
    idx=findall(keep)
    !isempty(idx)&&(idx=idx[sortperm(idx;by=j->(real(λ[j]),imag(λ[j])))])
    method=validate_roots ? :chebyshev_all : adaptive_validation ? Symbol(:chebyshev_singular_,adaptive_validation_method) : :none
    candidates=(values=λ[idx],vectors=Φ[:,idx],residuals=raw[idx],normalized_residuals=normalized[idx],effective_singular_values=σeff[idx],checked=checked[idx],all_values=λ,all_vectors=Φ,all_residuals=raw,all_normalized_residuals=normalized,all_effective_singular_values=σeff,all_checked=checked,inside=inside,kept=keep)
    common=(rank=rk,ranks=reduced.ranks,svd_tolerances=reduced.svd_tolerances,selected_svd_tolerance=reduced.svd_tolerances[selected_it],selected_svd_index=selected_it,rank_threshold=reduced.rank_thresholds[selected_it],rank_thresholds=reduced.rank_thresholds,probe_dimension=reduced.probe_dimension,moment_singular_values=reduced.singular_values,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=method,adaptive_validation_method=adaptive_validation_method)
    verbose&&println("selected SVD tolerance       = ",reduced.svd_tolerances[selected_it]," (rank ",rk,", accepted ",bestcount,")")
    return return_workspace ? merge(candidates,common,(cheb_workspace=reduced.cheb_workspace,)) : merge(candidates,common)
end

# helper that will clip the contours into the actual wanted region (if the contours cross say the real axis it can otherwise pick up spurious roots that behave weirdly)
@inline function _wiersig_in_spectrum_region(k::Complex{T},region::Tuple{T,T,T,T}) where {T<:Real}
    re_min,re_max,im_min,im_max=region
    return re_min<=real(k)<=re_max&&im_min<=imag(k)<=im_max
end
# Restrict an already enclosed contour root to the requested spectral region and, for tessellated contours, to its unique non-overlapping ownership cell.
@inline function _wiersig_in_spectrum_cell(k::Complex{T},contour::AbstractWiersigContour{T},region::Tuple{T,T,T,T}) where {T<:Real}
    _wiersig_in_spectrum_region(k,region)||return false
    isnothing(contour.ownership)&&return true
    xlo,xhi,ylo,yhi=contour.ownership;x=real(k);y=imag(k)
    xin=xlo<=x&&(x<xhi||xhi==region[2]&&x<=xhi)
    yin=ylo<=y&&(y<yhi||yhi==region[4]&&y<=yhi)
    return xin&&yin
end

function _wiersig_beyn_solve_reduced(R,solver::AbstractWiersigSolver,pts::Vector{BoundaryPointsCFIE{T}},ws::AbstractWiersigGeometryWorkspace,contour::AbstractWiersigContour{T},A::Matrix{Complex{T}},y::Vector{Complex{T}};validate_roots::Bool=false,adaptive_validation::Bool=true,adaptive_validation_method::Symbol=:binary,validation_padding::Int=5,res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-10),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,multithreaded::Bool=true,verbose::Bool=false) where {T<:Real}
    adaptive_validation_method in (:binary,:linear,:check)||throw(ArgumentError("adaptive_validation_method must be :binary, :linear, or :check"))
    N=boundary_matrix_size(ws)
    if maximum(R.ranks;init=0)==0
        empty=(values=Complex{T}[],vectors=Matrix{Complex{T}}(undef,N,0),residuals=T[],normalized_residuals=T[],effective_singular_values=T[],checked=Bool[],all_values=Complex{T}[],all_vectors=Matrix{Complex{T}}(undef,N,0),all_residuals=T[],all_normalized_residuals=T[],all_effective_singular_values=T[],all_checked=Bool[],inside=Bool[],kept=Bool[])
        common=(rank=0,ranks=R.ranks,svd_tolerances=R.svd_tolerances,selected_svd_tolerance=R.svd_tolerances[1],selected_svd_index=1,rank_threshold=R.rank_thresholds[1],rank_thresholds=R.rank_thresholds,probe_dimension=R.probe_dimension,moment_singular_values=R.singular_values,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,validation_method=:none,adaptive_validation_method=adaptive_validation_method)
        return merge(empty,common)
    end
    stable=_wiersig_beyn_stable_rank(R.ranks)
    preferred_rank=isnothing(stable) ? 0 : stable.rank
    selected=nothing;bestcount=-1;selected_it=0
    @inbounds for it in eachindex(R.ranks)
        rk=R.ranks[it];rk==0&&continue;it>1&&rk==R.ranks[it-1]&&continue
        E=nothing;@blas_multi_then_1 MAX_BLAS_THREADS E=eigen(Matrix(@view R.B[1:rk,1:rk]))
        λ=Vector{Complex{T}}(E.values);Y=Matrix{Complex{T}}(E.vectors);Φ=Matrix{Complex{T}}(undef,N,length(λ))
        @blas_multi_then_1 MAX_BLAS_THREADS mul!(Φ,@view(R.U[:,1:rk]),Y)
        σeff=_wiersig_beyn_effective_sigma(Y,@view R.singular_values[1:rk])
        nroots=length(λ);inside=falses(nroots);keep=falses(nroots);checked=falses(nroots);raw=fill(T(NaN),nroots);normalized=fill(T(NaN),nroots)
        for j in eachindex(λ);inside[j]=isfinite(real(λ[j]))&&isfinite(imag(λ[j]))&&wiersig_inside_contour(contour,λ[j]);keep[j]=inside[j];end
        inside_idx=findall(inside)
        if validate_roots
            _wiersig_beyn_validate_direct!(raw,normalized,checked,keep,inside_idx,λ,Φ,solver,pts,ws,A,y;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
        elseif adaptive_validation&&!isempty(inside_idx)
            validator=idx->_wiersig_beyn_validate_direct!(raw,normalized,checked,keep,idx,λ,Φ,solver,pts,ws,A,y;res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
            _wiersig_beyn_singular_validation!(validator,inside,σeff,checked,keep;validation_padding=validation_padding,method=adaptive_validation_method)
        end
        naccepted=count(keep)
        better=naccepted>bestcount
        preferred=naccepted==bestcount&&rk==preferred_rank&&(selected_it==0||R.ranks[selected_it]!=preferred_rank)
        if better||preferred
            bestcount=naccepted;selected_it=it
            selected=(λ=λ,Φ=Φ,σeff=σeff,raw=raw,normalized=normalized,inside=inside,checked=checked,keep=keep)
        end
        verbose&&println("SVD rank spectrum: tolerance=",R.svd_tolerances[it],", rank=",rk,", enclosed=",count(inside),", accepted=",naccepted,", rejected=",count(inside .& .!keep),", selected=",selected_it==it)
    end
    if !isnothing(stable)&&R.ranks[selected_it]==stable.rank
        selected_it=stable.index
    end
    λ=selected.λ;Φ=selected.Φ;σeff=selected.σeff;raw=selected.raw;normalized=selected.normalized;inside=selected.inside;checked=selected.checked;keep=selected.keep;rk=R.ranks[selected_it]
    idx=findall(keep);!isempty(idx)&&(idx=idx[sortperm(idx;by=j->(real(λ[j]),imag(λ[j])))])
    method=validate_roots ? :direct_all : adaptive_validation ? Symbol(:direct_singular_,adaptive_validation_method) : :none
    candidates=(values=λ[idx],vectors=Φ[:,idx],residuals=raw[idx],normalized_residuals=normalized[idx],effective_singular_values=σeff[idx],checked=checked[idx],all_values=λ,all_vectors=Φ,all_residuals=raw,all_normalized_residuals=normalized,all_effective_singular_values=σeff,all_checked=checked,inside=inside,kept=keep)
    common=(rank=rk,ranks=R.ranks,svd_tolerances=R.svd_tolerances,selected_svd_tolerance=R.svd_tolerances[selected_it],selected_svd_index=selected_it,rank_threshold=R.rank_thresholds[selected_it],rank_thresholds=R.rank_thresholds,probe_dimension=R.probe_dimension,moment_singular_values=R.singular_values,contour=contour,dlp_kernel=dlp_kernel,roots_validated=validate_roots,adaptive_validation=adaptive_validation,adaptive_validation_method=adaptive_validation_method,validation_method=method)
    return merge(candidates,common)
end

function _compute_spectrum(solver::AbstractWiersigSolver,tess::AbstractWiersigTessellation{T};chebyshev::Bool=true,nq=64,probe_factor::Real=2.0,min_probe::Int=50,svd_tol::AbstractVector{T}=T[1e-7,5e-8,1e-8,5e-9,1e-9,5e-10,1e-10],relative_svd_tol::Bool=false,res_tol::T=T(1e-9),normalized_res_tol::T=T(1e-10),filter_raw_residual::Bool=false,matnorm::Symbol=:one,dlp_kernel::Symbol=:source,rng_seed::Int=0,multithreaded::Bool=true,npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=10_000,M_j_init::Int=5,cheb_tol::T=T(1e-11),sampling_points::Int=50_000,max_iter::Int=20,grow_panels::T=T(1.5),grow_M::Int=2,plan_threads::Int=Threads.nthreads(),cheb_verbose::Bool=false,verbose::Bool=true,validate_roots::Bool=false,adaptive_validation::Bool=true,validation_padding::Int=5,ram_cap_gib::Union{Nothing,Real}=nothing,ram_fraction::Real=0.75,adaptive_validation_method::Symbol=:binary) where {T<:Real}
    adaptive_validation_method in (:binary,:linear,:check)||throw(ArgumentError("adaptive_validation_method must be :binary, :linear, or :check"))
    _wiersig_dlp_normal_mode(dlp_kernel);contours=tess.contours;region=tess.region
    isempty(contours)&&throw(ArgumentError("tessellation must not be empty"));isempty(svd_tol)&&throw(ArgumentError("svd_tol must not be empty"))
    rectangle=tess isa WiersigRectangleTessellation{T}
    if rectangle
        validnq=nq isa Integer||(nq isa Tuple&&length(nq)==2&&all(x->x isa Integer,nq));validnq||throw(ArgumentError("rectangular tessellation requires scalar nq or (nh,nv)"))
        nh,nv=nq isa Integer ? (Int(nq),Int(nq)) : (Int(nq[1]),Int(nq[2]));nh>0&&nv>0||throw(ArgumentError("Gauss-Legendre orders must be positive"))
    else
        nq isa Integer||throw(ArgumentError("smooth tessellation requires scalar nq"));nq>0||throw(ArgumentError("nq must be positive"))
    end
    ncontours=length(contours);C=length(solver.billiards);nin=_wiersig_component_indices(solver,C)
    re_min,re_max,im_min,im_max=region;kr=max(abs(re_min),abs(re_max));ki=max(abs(im_min),abs(im_max));kmax=hypot(kr,ki)
    qres=T[max(nin[a],solver.n_out)*kmax for a in 1:C];pts=evaluate_points(solver,qres);ws=build_cfie_kress_workspace(solver,pts);N=boundary_matrix_size(ws)
    probe=min(maximum(_wiersig_beyn_probe_rank(solver,c;factor=probe_factor,min_probe=min_probe) for c in contours),N)
    nmat=!chebyshev ? 0 : rectangle ? 2nh+2nv : Int(nq)
    buf=WiersigBeynBuffers(T,N,probe,nmat,MersenneTwister(rng_seed);ram_cap_gib=ram_cap_gib,ram_fraction=ram_fraction)
    results=Vector{Any}(undef,ncontours)
    if !rectangle
        p=verbose ? Progress(ncontours,desc="Beyn smooth contours") : nothing
        @inbounds for ic in eachindex(contours)
            contour=contours[ic];fill!(buf.A0,0);fill!(buf.A1,0);z,w=wiersig_beyn_contour(contour,nq)
            if chebyshev
                @benchit timeit=verbose "Chebyshev contour workspace" cws=build_chebyshev_workspace(solver,pts,z;npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,verbose=cheb_verbose)
                B=min(length(buf.As),length(z))
                for first in 1:B:length(z)
                    last=min(first+B-1,length(z));js=first:last;nb=length(js);work=nb==length(z) ? cws : _wiersig_subset_chebyshev_workspace(cws,js);Asb=nb==B ? buf.As : buf.As[1:nb]
                    @benchit timeit=verbose "Chebyshev matrix batch" construct_matrices!(solver,Asb,pts,work;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
                    _wiersig_beyn_accumulate_chebyshev!(buf.A0,buf.A1,buf.V,buf.X,Asb,ws,@view(z[js]),@view(w[js]))
                end
            else
                _wiersig_beyn_accumulate_direct!(buf.A0,buf.A1,buf.V,buf.X,buf.A,solver,pts,ws,z,w;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
            end
            R=_wiersig_beyn_build_reduced_problem(buf.A0,buf.A1;r=probe,r_step=probe,max_r=probe,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
            results[ic]=_wiersig_beyn_solve_reduced(R,solver,pts,ws,contour,buf.A,buf.y;validate_roots=validate_roots,adaptive_validation=adaptive_validation,adaptive_validation_method=adaptive_validation_method,validation_padding=validation_padding,res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
            verbose&&next!(p)
        end
    else
        cache=Dict{Int,WiersigBeynEdgeMoments{T}}();remaining=[count(!=(0),e.cells) for e in tess.edges]
        nhedges=count(e->iszero(imag(e.z1-e.z0)),tess.edges);nvedges=length(tess.edges)-nhedges;ntotal=nhedges*nh+nvedges*nv
        pedges=verbose ? Progress(ntotal,desc="Beyn unique edge nodes") : nothing;pcells=verbose ? Progress(ncontours,desc="Beyn cells") : nothing
        @inbounds for ic in eachindex(contours)
            contour=contours[ic];fill!(buf.A0,0);fill!(buf.A1,0);a0v=vec(buf.A0);a1v=vec(buf.A1)
            if chebyshev
                missing=Int[];for q in 1:4;eid=contour.edges[q];haskey(cache,eid)||push!(missing,eid);end
                if !isempty(missing)
                    newmom,nnew=_wiersig_beyn_edges_chebyshev(solver,pts,ws,tess.edges,missing,nh,nv,buf.V,buf.X,buf.As;dlp_kernel=dlp_kernel,multithreaded=multithreaded,npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,cheb_tol=cheb_tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,plan_threads=plan_threads,cheb_verbose=cheb_verbose)
                    for l in eachindex(missing);cache[missing[l]]=newmom[l];end
                    verbose&&next!(pedges;step=nnew)
                end
            end
            for q in 1:4
                eid=contour.edges[q];edge=tess.edges[eid]
                Em=if haskey(cache,eid)
                    cache[eid]
                else
                    ne=iszero(imag(edge.z1-edge.z0)) ? nh : nv;E=_wiersig_beyn_edge_direct(solver,pts,ws,edge,ne,buf.V,buf.X,buf.A;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
                    cache[eid]=E;verbose&&next!(pedges;step=ne);E
                end
                α=Complex{T}(contour.signs[q]);BLAS.axpy!(α,vec(Em.A0),a0v);BLAS.axpy!(α,vec(Em.A1),a1v)
                remaining[eid]-=1;iszero(remaining[eid])&&delete!(cache,eid)
            end
            R=_wiersig_beyn_build_reduced_problem(buf.A0,buf.A1;r=probe,r_step=probe,max_r=probe,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,verbose=verbose)
            results[ic]=_wiersig_beyn_solve_reduced(R,solver,pts,ws,contour,buf.A,buf.y;validate_roots=validate_roots,adaptive_validation=adaptive_validation,adaptive_validation_method=adaptive_validation_method,validation_padding=validation_padding,res_tol=res_tol,normalized_res_tol=normalized_res_tol,filter_raw_residual=filter_raw_residual,matnorm=matnorm,dlp_kernel=dlp_kernel,multithreaded=multithreaded,verbose=verbose)
            verbose&&next!(pcells)
        end
        isempty(cache)||@warn "rectangle edge cache was not empty after tessellation" cached_edges=collect(keys(cache))
        any(!iszero,remaining)&&@warn "some rectangle edges were not fully consumed" remaining=remaining
    end
    values=Complex{T}[];vectors=Vector{Vector{Complex{T}}}();residuals=T[];normalized_residuals=T[];source_contours=Int[]
    for ic in eachindex(results)
        result=results[ic];contour=contours[ic]
        @inbounds for j in eachindex(result.values)
            k=result.values[j];_wiersig_in_spectrum_cell(k,contour,region)||continue
            push!(values,k);push!(vectors,Vector{Complex{T}}(@view result.vectors[:,j]));push!(residuals,result.residuals[j]);push!(normalized_residuals,result.normalized_residuals[j]);push!(source_contours,ic)
        end
    end
    order=sortperm(eachindex(values);by=i->(real(values[i]),imag(values[i])))
    return (values=values[order],vectors=vectors[order],residuals=residuals[order],normalized_residuals=normalized_residuals[order],source_contours=source_contours[order],tessellation=tess,contours=contours,contour_results=results,contour_pts=fill(pts,ncontours),contour_workspaces=fill(ws,ncontours),contour_dimensions=fill(N,ncontours),contour_k_resolution=fill(kmax,ncontours),contour_q_resolution=[copy(qres) for _ in 1:ncontours])
end

function _wiersig_merge_spectrum_intervals(results::Vector,region::NTuple{4,T}) where {T<:Real}
    values=Complex{T}[];vectors=Vector{Vector{Complex{T}}}();residuals=T[];normalized_residuals=T[];source_intervals=Int[];source_contours=Int[]
    contours=Any[];contour_results=Any[];contour_pts=Any[];contour_workspaces=Any[];contour_dimensions=Int[];contour_k_resolution=T[];contour_q_resolution=Vector{T}[];offset=0
    @inbounds for ii in eachindex(results)
        R=results[ii];lo,hi=R.tessellation.region[1:2];last=ii==length(results)
        for j in eachindex(R.values)
            k=R.values[j];x=real(k);lo<=x&&(x<hi||last&&x<=hi)||continue
            push!(values,k);push!(vectors,R.vectors[j]);push!(residuals,R.residuals[j]);push!(normalized_residuals,R.normalized_residuals[j])
            push!(source_intervals,ii);push!(source_contours,R.source_contours[j]+offset)
        end
        append!(contours,R.contours);append!(contour_results,R.contour_results);append!(contour_pts,R.contour_pts);append!(contour_workspaces,R.contour_workspaces)
        append!(contour_dimensions,R.contour_dimensions);append!(contour_k_resolution,R.contour_k_resolution);append!(contour_q_resolution,R.contour_q_resolution);offset+=length(R.contours)
    end
    order=sortperm(eachindex(values);by=i->(real(values[i]),imag(values[i])))
    return (values=values[order],vectors=vectors[order],residuals=residuals[order],normalized_residuals=normalized_residuals[order],source_intervals=source_intervals[order],source_contours=source_contours[order],region=region,interval_results=results,tessellations=[R.tessellation for R in results],contours=contours,contour_results=contour_results,contour_pts=contour_pts,contour_workspaces=contour_workspaces,contour_dimensions=contour_dimensions,contour_k_resolution=contour_k_resolution,contour_q_resolution=contour_q_resolution)
end

# Compute the spectrum of dielectric resonances in a rectangular region of the complex plane, using a rectangular tessellation of the region. For this region it uses the same boundary discretization and probe matrix for all cells, caching edge moments for shared edges. The region is subdivided into intervals along the real axis, each of which is solved independently. The results are merged into a single spectrum.
function compute_spectrum(solver::AbstractWiersigSolver,re_min::T,re_max::T,im_min::T,im_max::T,seed::WiersigRectangleContour{T};interval_width::T=T(5),kwargs...) where {T<:Real}
    interval_width>zero(T)||throw(ArgumentError("interval_width must be positive"))
    n=max(1,ceil(Int,(re_max-re_min)/interval_width));dx=(re_max-re_min)/T(n);results=Vector{Any}(undef,n)
    @inbounds for i in 1:n
        lo=re_min+T(i-1)*dx;hi=i==n ? re_max : re_min+T(i)*dx
        results[i]=_compute_spectrum(solver,wiersig_contour_tessellation(lo,hi,im_min,im_max,seed);kwargs...)
    end
    return _wiersig_merge_spectrum_intervals(results,(re_min,re_max,im_min,im_max))
end
# Compute the spectrum of dielectric resonances in a rectangular region of the complex plane, using a smooth tessellation of the region.
function compute_spectrum(solver::AbstractWiersigSolver,re_min::T,re_max::T,im_min::T,im_max::T,seed::WiersigSmoothContour{T};interval_width::T=T(5),overlap_re::T=zero(T),overlap_im::T=zero(T),kwargs...) where {T<:Real}
    interval_width>zero(T)||throw(ArgumentError("interval_width must be positive"))
    n=max(1,ceil(Int,(re_max-re_min)/interval_width));dx=(re_max-re_min)/T(n);results=Vector{Any}(undef,n)
    @inbounds for i in 1:n
        lo=re_min+T(i-1)*dx;hi=i==n ? re_max : re_min+T(i)*dx
        results[i]=_compute_spectrum(solver,wiersig_contour_tessellation(lo,hi,im_min,im_max,seed;overlap_re=overlap_re,overlap_im=overlap_im);kwargs...)
    end
    return _wiersig_merge_spectrum_intervals(results,(re_min,re_max,im_min,im_max))
end

"""
    compute_spectrum(solver,re_min,re_max,im_min,im_max,seed;interval_width=5,...)

Compute all dielectric resonances in the requested spectral rectangle by
splitting the real-k range into independent resolution intervals.

Each interval constructs its own contour tessellation and is solved by one
`_compute_spectrum` call. All contours/cells inside that interval therefore
share one boundary discretization, Wiersig workspace, probe matrix and reusable
dense buffers. The discretization is sized from the largest |k| in the interval,
so a large spectral range does not use the resolution required by its largest k
everywhere.

The real interval `[re_min,re_max]` is divided into

    n=ceil((re_max-re_min)/interval_width)

equal resolution intervals. Adjacent intervals use half-open ownership,

    [k₀,k₁),[k₁,k₂),...,[kₙ₋₁,kₙ],

with the final upper boundary included, so roots on interval boundaries are
returned exactly once.

For `WiersigRectangleContour`, each interval constructs an exact rectangular
tessellation and shared-edge moments are reused only within that interval.

For `WiersigSmoothContour`, each interval constructs a smooth tessellation using
the supplied `overlap_re` and `overlap_im`. Contours have independent contour
quadratures but share the interval boundary discretization and dense buffers.

`interval_width` controls the compromise between boundary over-resolution and
repeated interval setup. Smaller values track the k-dependent boundary
resolution more closely, while larger values reduce the number of workspace
rebuilds.

All remaining keyword arguments are forwarded to `_compute_spectrum`.

# Kwargs

- `chebyshev::Bool=true`: use multi-k Chebyshev matrix assembly. For smooth
  tessellations the Chebyshev workspace is local to each contour. 
  For rectangular tessellations one workspace is built per cell from only the currently uncached edge quadrature nodes. Thus a workspace never contains more than one contour's nodes, while shared-edge moments are cached and reused by neighboring cells.
- `nq=64`: contour quadrature order. For `WiersigSmoothTessellation`, `nq` must
  be a positive integer giving the number of periodic trapezoidal nodes per
  contour. For `WiersigRectangleTessellation`, `nq` may be either a positive
  integer, using the same Gauss-Legendre order on all edges, or `(nh,nv)`,
  giving the orders on horizontal and vertical edges respectively.
- `probe_factor::Real=2.0`: multiplicative safety factor applied to the leading
  dielectric Weyl estimate when selecting the Beyn probe dimension.
- `min_probe::Int=50`: minimum Beyn probe dimension before clipping to the
  boundary-matrix dimension.
- `svd_tol::AbstractVector{T}=T[1e-7,5e-8,1e-8,5e-9,1e-9,5e-10,1e-10]`:
  nonincreasing sequence of numerical-rank thresholds for the zeroth Beyn
  moment. Each distinct resulting rank defines a complete reduced spectrum. Spectra from different thresholds are not merged. Repeated equal ranks across consecutive thresholds define stable rank plateaus, which are preferred when several ranks recover the same number of accepted roots.
- `relative_svd_tol::Bool=false`: if true, interpret each SVD tolerance relative
  to the largest singular value of `A₀`; otherwise use absolute thresholds.
- `res_tol::T=T(1e-9)`: raw nonlinear residual tolerance used when
  `filter_raw_residual=true`.
- `normalized_res_tol::T=T(1e-10)`: normalized nonlinear residual tolerance used
  to accept validated Beyn roots.
- `filter_raw_residual::Bool=false`: additionally require the raw residual
  `||A(k)x||` to satisfy `res_tol`.
- `matnorm::Symbol=:one`: norm family used for the normalized residual. Supported
  values are `:one`, `:two` and `:inf`.
- `dlp_kernel::Symbol=:source`: double-layer normal convention used in Wiersig
  matrix construction.
- `rng_seed::Int=0`: deterministic seed used to construct the random Beyn probe
  matrix or matrices.
- `multithreaded::Bool=true`: enable threaded Wiersig matrix assembly.
- `npanels_h_init::Int=15_000`: initial number of panels used for the Hankel
  Chebyshev interpolation tables.
- `M_h_init::Int=5`: initial polynomial degree used for the Hankel Chebyshev
  interpolation.
- `npanels_j_init::Int=10_000`: initial number of panels used for the Bessel-J
  Chebyshev interpolation tables.
- `M_j_init::Int=5`: initial polynomial degree used for the Bessel-J Chebyshev
  interpolation.
- `cheb_tol::T=T(1e-11)`: target accuracy for the adaptive Chebyshev special-
  function interpolation.
- `sampling_points::Int=50_000`: number of test points used when estimating
  Chebyshev interpolation error.
- `max_iter::Int=20`: maximum number of Chebyshev-plan refinement iterations.
- `grow_panels::T=T(1.5)`: multiplicative panel-count increase during Chebyshev
  refinement.
- `grow_M::Int=2`: polynomial-degree increase during Chebyshev refinement.
- `plan_threads::Int=Threads.nthreads()`: number of threads used while building
  Chebyshev interpolation plans.
- `cheb_verbose::Bool=false`: print Chebyshev-plan construction diagnostics.
- `verbose::Bool=true`: print tessellation information, Beyn progress, per-cell
  diagnostics and the final spectrum summary.
- `validate_roots::Bool=false`: validate every enclosed Beyn candidate with the
  nonlinear Wiersig residual.
- `adaptive_validation::Bool=true`: when `validate_roots=false`, perform
  selective nonlinear residual validation after ordering enclosed candidates by
  increasing effective moment singular value
  `σeff_j=[Σ_l |Y_lj|²/σ_l² / Σ_l |Y_lj|²]⁻¹ᐟ²`.
- `adaptive_validation_method::Symbol=:binary`: adaptive-validation algorithm.
  `:binary` is the production default and assumes that residual acceptance has
  a single reject -> accept transition as `σeff` increases. The transition is
  found in O(log M) residual checks for M enclosed candidates, followed by at
  most `validation_padding` local confirmation checks. `:linear` uses the
  original padded sequential validator. `:check` runs the binary algorithm and
  then validates every enclosed candidate, throwing an error if the inferred
  accepted-root set differs from complete validation.
- `validation_padding::Int=5`: number of consecutive passing candidates used to
  confirm the accepted side of the binary transition. It retains its original
  meaning for `adaptive_validation_method=:linear`.
- `ram_cap_gib::Union{Nothing,Real}=nothing`: optional explicit RAM budget, in
  GiB, for simultaneously stored dense matrices in Chebyshev assembly. If
  `nothing`, the budget is chosen automatically from physical RAM.
- `ram_fraction::Real=0.75`: fraction of total physical RAM available to the
  automatic dense-matrix batch planner when `ram_cap_gib=nothing`.

# Returns

A named tuple with:

- `values::Vector{Complex{T}}`: accepted resonances in `tess.region`, restricted
  to the unique ownership cell of their source contour.
- `vectors::Vector{Vector{Complex{T}}}`: corresponding active Wiersig boundary
  vectors. All vectors returned by one `_compute_spectrum` call have the same
  boundary dimension.
- `residuals::Vector{T}`: raw nonlinear residuals. Entries are `NaN` for roots
  which were not explicitly validated.
- `normalized_residuals::Vector{T}`: normalized nonlinear residuals. Entries are
  `NaN` for roots which were not explicitly validated.
- `source_contours::Vector{Int}`: source contour or rectangular cell index of
  every returned resonance.
- `tessellation`: the supplied tessellation object.
- `contours`: `tess.contours`.
- `contour_results`: individual Beyn results for each ownership cell.
- `contour_pts`: the common boundary quadrature data, repeated once per contour
  for index compatibility.
- `contour_workspaces`: the common Wiersig geometry workspace, repeated once per
  contour for index compatibility.
- `contour_dimensions`: common boundary-matrix dimension repeated once per
  contour.
- `contour_k_resolution`: vacuum-wavenumber resolution bounds used to construct
  the boundary discretization. For a rectangular tessellation the common global
  bound is repeated for every cell.
- `contour_q_resolution`: corresponding interior/exterior physical-wavenumber
  resolution bounds. For a rectangular tessellation the common global bounds
  are repeated for every cell.
"""
compute_spectrum(solver::AbstractWiersigSolver,tess::AbstractWiersigTessellation;kwargs...)=_compute_spectrum(solver,tess;kwargs...)