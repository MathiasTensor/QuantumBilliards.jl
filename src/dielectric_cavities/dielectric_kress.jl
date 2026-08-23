const TWO_PI=2*pi
const INV_TWO_PI=1/TWO_PI
const INV_PI=1/pi
const EULER_OVER_PI=MathConstants.eulergamma/pi

# Let Ω₁,...,Ω_C⊂R² be mutually disjoint dielectric domains with Γ_a=∂Ω_a and
# common exterior Ω₀=R²\⋃ₐΩ̄_a. The refractive indices are n_a>0 and n₀=n_out>0.
#
# For vacuum wavenumber k define q_a=n_a k and q₀=n_out k. Then
# (Δ+q_a²)u_a=0 in Ω_a and (Δ+q₀²)u₀=0 in Ω₀, with u₀ outgoing.
# On Γ_a use x_a=(φ_a,ψ_a), where ψ_a is the Dirichlet trace and φ_a the
# polarization-scaled normal trace. Globally x=(φ,ψ), with
# φ=(φ₁,...,φ_C), ψ=(ψ₁,...,ψ_C). The polarization factor is χ(n)=1 for TM
# and χ(n)=n² for TE.
#
# Wiersig Green function and doubled boundary operators
# With G_q(x,y)=-(i/4)H₀⁽¹⁾(q|x-y|), satisfying (Δ+q²)G_q=δ, define
# S_q=-2G_q and D_q=2∂_{n_y}G_q. For γ'(t)=(t_x,t_y), speed
# s=|γ'|, outward normal sn=(t_y,-t_x), R=x-y and r=|R|,
# inner=t_yR_x-t_xR_y=s n_y⋅R. Hence
# S_q(x,y)=(i/2)H₀⁽¹⁾(qr)s and D_q(x,y)=-(iq/2)H₁⁽¹⁾(qr)inner/r.
#
# Interior and exterior equations (with the n_a)
# The interiors are disconnected, so Ω_a couples only to Γ_a:
# χ(n_a)S_aa(n_a k)φ_a+[D_aa(n_a k)-I]ψ_a=0.
# Hence Sχ,int(k)=diag_a[χ(n_a)S_aa(n_a k)] and
# D_int(k)=diag_a[D_aa(n_a k)], with all a≠b interior blocks exactly zero.
# The exterior Ω₀ is connected, so every Γ_b couples to every Γ_a:
# χ(n₀)Σ_bS_ab(q₀)φ_b+Σ_bD_ab(q₀)ψ_b+ψ_a=0.
# Therefore the resonance matrix is

# A(k)=[Sχ,int(k)  D_int(k)-I;
#       χ(n₀)S_ext(q₀)  D_ext(q₀)+I],

# and resonances satisfy A(k_*)x=0 for x≠0. The library stores normals
# pointing outward from each dielectric cavity. They are therefore outward
# normals for Ω_a but inward normals for the common exterior Ω₀. Rewriting
# the exterior BIE using this same cavity-outward convention reverses the DLP
# jump sign, giving D_ext(q₀)+I.
#
# Kress scheme
# Same-boundary kernels use the periodic logarithmic split
# K(t,s)=K₁(t,s)log[4sin²((t-s)/2)]+K₂(t,s) and Kress product quadrature.
# Distinct physical cavities are disjoint, so Γ_a×Γ_b with a≠b is smooth and
# uses ordinary Nyström quadrature.
#
# For C>1, ppw, min_pts, quadrature_kind, kressq may each be
# supplied as one common value or as a vector of length C. Different cavity
# geometries may therefore use different local Kress discretizations.
#
# Symmetry
# One billiard: symmetry acts internally on Γ and uses native QuantumBilliards
# symmetry reduction.
#
# Multiple billiards: symmetry acts only between complete physical cavities,
# g:Γ_a→Γ_b with b≠a. Local quadratures remain unreduced. Exact symmetry images
# are built from cavity indices and local node indices only.
#
# Rotation(n,m): consecutive groups of n cavities form C_n orbits with
# R(a,j)=(π_R(a),j), equal local node counts and C mod n=0. Character factors
# are χ_l=exp(2πiml/n), l=0,...,n-1.
# Reflection: for :x_axis or :y_axis, consecutive cavity pairs are partners and
# orientation reversal gives j'=N-j+1 with scales (1,p), p=±1. For :origin,
# consecutive groups are [Γ,R_xΓ,R_yΓ,R_xR_yΓ] with index maps
# [j,N-j+1,N-j+1,j] and scales [1,p_x,p_y,p_xp_y].
# Multiple billiards:
# symmetry is represented by exact orbits of global physical boundary nodes.
# A group element may either map Γ_a to another cavity Γ_b or act internally
# on Γ_a. Thus the general action is g:(a,j) -> (a',j').
# Interior reduced assembly retains only orbit images on the target physical
# cavity, whereas the connected exterior retains the complete orbit.
# Material invariance requires n(gx)=n(x), so all cavities in one symmetry orbit
# must have identical refractive indices.

abstract type AbstractWiersigSolver end
abstract type AbstractWiersigGeometryWorkspace end

# SLP gets multiplied by these factors as in J. Wiersig, “Boundary element method for resonances in dielectric microcavities,” J. Opt. A: Pure Appl. Opt. 5, 53–60 (2003). DOI: 10.1088/1464-4258/5/1/308. The arXiv version is physics/0206018.
# The TE factor χ(n)=n² follows from using φ=n⁻² * ∂νψ as the trace, so it is built right into the matrix.
@inline _wiersig_slp_factor(::Val{:TM},n::T) where {T<:Real}=one(T)
@inline _wiersig_slp_factor(::Val{:TE},n::T) where {T<:Real}=n*n

# Product Z₂×Z₂ reflection symmetry for a multi-cavity system:one reflection acts inside each physical cavity, the other exchanges cavity pairs. Previous interior resonance codes did not need this as cavities were not connetcted.
struct WiersigMixedReflection
    intra_parity::Int
    inter_parity::Int
    function WiersigMixedReflection(intra_parity::Int,inter_parity::Int)
        intra_parity in (-1,1)||throw(ArgumentError("intra_parity must be ±1"))
        inter_parity in (-1,1)||throw(ArgumentError("inter_parity must be ±1"))
        return new(intra_parity,inter_parity)
    end
end

# hacks to make it work with CFIE point evaluations since they dont have connected multi cavity symmetries and dont want to
# change the code
@inline _wiersig_node_multiple(::Nothing)=1
@inline _wiersig_node_multiple(::WiersigMixedReflection)=4
@inline _wiersig_node_multiple(sym::Union{Rotation,Reflection})=1 # standard Rotation/Reflection already have the correct node count
# so we adjust ppw sligthly ppw minimally so that we get the correct node count
@inline function _wiersig_adjust_ppw(ppw::T,k::T,L::T,m::Int,min_pts::Int) where {T<:Real}
    m==1 && return ppw
    N=max(min_pts,round(Int,k*L*ppw/two_pi))
    N=cld(N,m)*m
    return T(two_pi*N/(k*L))
end

@inline function _wiersig_with_ppw(q::CFIE_kress{T},ppw::T) where {T<:Real}
    return CFIE_kress(ppw,q.billiard;min_pts=q.min_pts,eps=q.eps,symmetry=q.symmetry)
end
@inline function _wiersig_with_ppw(q::CFIE_kress_corners{T},ppw::T) where {T<:Real}
    return CFIE_kress_corners(ppw,q.billiard;min_pts=q.min_pts,eps=q.eps,symmetry=q.symmetry,kressq=q.kressq,min_t_spacing=q.min_t_spacing)
end
@inline function _wiersig_with_ppw(q::CFIE_kress_global_corners{T},ppw::T) where {T<:Real}
    return CFIE_kress_global_corners(ppw,q.billiard;min_pts=q.min_pts,eps=q.eps,symmetry=q.symmetry,kressq=q.kressq,min_t_spacing=q.min_t_spacing)
end
# hack to change slightly the ppw to get the correct N mod 4 = 0. Hacky as hell but dont want to change interior resonance only source code
@inline function _wiersig_effective_ppw(ppw::T,q::T,L::T,min_pts::Int,sym::S) where {T<:Real,S}
    m=_wiersig_node_multiple(sym)
    m==1&&return ppw
    N=max(min_pts,round(Int,q*L*ppw/TWO_PI))
    N=cld(N,m)*m
    return T(TWO_PI*N/(q*L))

end

# Expand a scalar cavity parameter to C entries, or validate an explicit vector.
function _wiersig_parameter_vector(x,C::Int,name::Symbol)
    if x isa AbstractVector
        length(x)==C||throw(DimensionMismatch("$name has $(length(x)) entries for $C cavities"))
        return collect(x)
    end
    return fill(x,C)
end

# Wiersig dielectric solver using Kress product quadrature.
# For cavities Ω_a with Γ_a=∂Ω_a, the material wavenumbers are q_a=n_a k, q_out=n_out k.
# The stored ppw values control the independent discretization of each Γ_a.
mutable struct WiersigKress{T<:Real,Q,Bi<:AbstractVector{<:AbsBilliard},S<:Union{Nothing,Rotation,Reflection,WiersigMixedReflection}}<:AbstractWiersigSolver
    n_in::AbstractVector{T} # Interior refractive indices n_a.
    n_out::T # Exterior refractive index n_out.
    ppw::AbstractVector{T} # Points per wavelength on Γ_a.
    polarization::Symbol # :TM or :TE.
    quadrature::Q # Kress quadrature rule(s).
    billiards::Bi # Physical dielectric cavities Ω_a.
    symmetry::S # Optional rotation/reflection symmetry. Defaults to Nothing
end

# Construct the validated solver representation.
# The cavity data satisfy n_in=(n₁,...,n_C),     ppw=(b₁,...,b_C),
# with one boundary discretization associated with each physical cavity.
function _wiersig_solver(n_in::AbstractVector{T},n_out::T,ppw::AbstractVector{T},polarization::Symbol,quadrature::Q,billiards::Bi,symmetry::S) where {T<:Real,Q,Bi<:AbstractVector{<:AbsBilliard},S<:Union{Nothing,Rotation,Reflection,WiersigMixedReflection}}
    polarization in (:TM,:TE)||throw(ArgumentError("polarization must be :TM or :TE"))
    return WiersigKress{T,Q,Bi,S}(n_in,n_out,ppw,polarization,quadrature,billiards,symmetry)
end

# Construct the local Kress rule on one physical boundary Γ.
# :smooth uses the periodic logarithmic Kress rule, while :corners and
# :global_corners apply Kress grading near nonsmooth boundary points. The diff
# is just if we have one corner we use :corners because it uses the same grading as Kress 
# used foe the teardrop or :global_corners for piecewise defined billiards with multiple corners.
function _wiersig_make_quadrature(b::T,billiard::AbsBilliard;min_pts::Int,symmetry::S,quadrature_kind::Symbol,kressq::Int,min_t_spacing::T) where {T<:Real,S<:Union{Nothing,Rotation,Reflection,WiersigMixedReflection}}
    quadrature_kind===:smooth&&return CFIE_kress(b,billiard;min_pts=min_pts,symmetry=symmetry)
    quadrature_kind===:corners&&return CFIE_kress_corners(b,billiard;min_pts=min_pts,symmetry=symmetry,kressq=kressq,min_t_spacing=min_t_spacing)
    quadrature_kind===:global_corners&&return CFIE_kress_global_corners(b,billiard;min_pts=min_pts,symmetry=symmetry,kressq=kressq,min_t_spacing=min_t_spacing)
    throw(ArgumentError("quadrature_kind must be :smooth, :corners, or :global_corners; received $quadrature_kind"))
end

# Construct a solver for one dielectric domain Ω with Γ=∂Ω.
# The interior Helmholtz wavenumber is q_in=n_in k and the exterior one is
# q_out=n_out k. Native boundary symmetry may therefore be applied directly
# during discretization of Γ.
function WiersigKress(n_in::L,n_out::L,billiard::AbsBilliard,ppw::T;min_pts::Int=200,polarization::Symbol=:TM,symmetry::Union{Nothing,Rotation,Reflection,WiersigMixedReflection}=nothing,quadrature_kind::Symbol=:smooth,kressq::Int=2,min_t_spacing::T=1e-12) where {L<:Real,T<:Real}
    M=promote_type(typeof(float(n_out)),typeof(float(n_in)),typeof(float(ppw)),typeof(float(min_t_spacing)))
    quadrature=_wiersig_make_quadrature(M(ppw),billiard;min_pts=min_pts,symmetry=symmetry,quadrature_kind=quadrature_kind,kressq=kressq,min_t_spacing=M(min_t_spacing))
    return _wiersig_solver(M[n_in],M(n_out),M[M(ppw)],polarization,quadrature,[billiard],symmetry)
end

# Construct a solver for disjoint dielectric domains Ω₁,...,Ω_C.
# Each cavity may have its own n_a, ppw, and local Kress rule. For C>1 the
# physical boundaries Γ_a are discretized independently and any symmetry g:Γ_a → Γ_b is applied later.
function WiersigKress(n_in::AbstractVector{L},n_out::L,billiards::AbstractVector{<:AbsBilliard},ppw::Union{T,AbstractVector{<:T}};min_pts::Union{Int,AbstractVector{<:Integer}}=200,polarization::Symbol=:TM,symmetry::Union{Nothing,Rotation,Reflection,WiersigMixedReflection}=nothing,quadrature_kind::Union{Symbol,AbstractVector{Symbol}}=:smooth,kressq::Union{Int,AbstractVector{<:Integer}}=2,min_t_spacing::Union{T,AbstractVector{<:T}}=1e-12) where {L<:Real,T<:Real}
    C=length(billiards)
    length(n_in) in (1,C)||throw(DimensionMismatch("n_in has $(length(n_in)) entries for $C dielectric cavities"))
    # For C=1, preserve the common vector-valued solver representation.
    if C==1
        b=ppw isa AbstractVector ? only(ppw) : ppw
        mp=min_pts isa AbstractVector ? only(min_pts) : min_pts
        qk=quadrature_kind isa AbstractVector ? only(quadrature_kind) : quadrature_kind
        kq=kressq isa AbstractVector ? only(kressq) : kressq
        mts=min_t_spacing isa AbstractVector ? only(min_t_spacing) : min_t_spacing
        R=promote_type(L,T,typeof(float(mts)))
        nin=R[n_in[1]]
        bs=R[R(b)]
        quadrature=_wiersig_make_quadrature(bs[1],billiards[1];min_pts=Int(mp),symmetry=symmetry,quadrature_kind=qk,kressq=Int(kq),min_t_spacing=R(mts))
        return _wiersig_solver(nin,R(n_out),bs,polarization,quadrature,collect(billiards),symmetry)
    end
    # Scalar parameters are promoted to cavitywise data x→(x,...,x), a=1,...,C.
    bs0=_wiersig_parameter_vector(ppw,C,:ppw)
    mps=_wiersig_parameter_vector(min_pts,C,:min_pts)
    qks=_wiersig_parameter_vector(quadrature_kind,C,:quadrature_kind)
    kqs=_wiersig_parameter_vector(kressq,C,:kressq)
    mts0=_wiersig_parameter_vector(min_t_spacing,C,:min_t_spacing)
    R=promote_type(L,T,map(x->typeof(float(x)),mts0)...)
    nin=length(n_in)==1 ? fill(R(n_in[1]),C) : R.(n_in)
    bs=R.(bs0)
    quadratures=map(1:C) do a
        _wiersig_make_quadrature(bs[a],billiards[a];min_pts=Int(mps[a]),symmetry=nothing,quadrature_kind=qks[a],kressq=Int(kqs[a]),min_t_spacing=R(mts0[a])) # symmetry=nothing as we have multiple cavities, and therefore we cant desymmetrize the fundamental one as that one is already the desymmetrized one (we can only desymmetrize the whole multi-cavity system)
    end
    return _wiersig_solver(nin,R(n_out),bs,polarization,quadratures,collect(billiards),symmetry)
end

# Test whether Γ consists of several distinct physical cavity boundaries.
@inline _wiersig_is_multibilliard(s::WiersigKress)::Bool=length(s.billiards)>1
# Return the polarization factor multiplying the Wiersig SLP: χ(n)=1 (TM), χ(n)=n² (TE).
@inline _wiersig_slp_factor(s::WiersigKress,n::T) where {T<:Real}=_wiersig_slp_factor(Val(s.polarization),n)

# Evaluate Kress nodes on the boundary components of one billiard.
# If Γ contains several oriented components, the first keeps its native
# orientation while subsequent components are reversed to maintain the
# exterior/interior normal convention used by the boundary operators.
function _wiersig_evaluate_local_points(q::Union{CFIE_kress{T},CFIE_kress_corners{T}},billiard::AbsBilliard,q_resolution::T)::Vector{BoundaryPointsCFIE{T}} where {T<:Real}
    comps=_boundary_components(billiard.full_boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        p=_evaluate_points(q,comp[1],q_resolution,idx)
        pts[idx]=idx==1 ? p : _reverse_component_orientation(q,p)
    end
    return pts
end

# Evaluate a globally graded Kress discretization.
# A composite Γ is treated as one globally parameterized interface whenever
# possible; isolated smooth components fall back to the ordinary smooth Kress parameterization.
function _wiersig_evaluate_local_points(q::CFIE_kress_global_corners{T},billiard::AbsBilliard,q_resolution::T)::Vector{BoundaryPointsCFIE{T}} where {T<:Real}
    boundary=billiard.full_boundary
    # A single smooth segment needs no global corner grading. This possible if full boundary is a a single curve.
    if length(boundary)==1&&!(boundary[1] isa AbstractVector)
        base=CFIE_kress(q.pts_scaling_factor,billiard;min_pts=q.min_pts,eps=q.eps,symmetry=q.symmetry)
        return [_evaluate_points(base,boundary[1],q_resolution,1)]
    end
    # One composite closed interface Γ graded globally across its corners.
    if _is_single_composite_boundary(boundary)
        return [_evaluate_points(q,boundary,q_resolution,1)]
    end
    # For several boundary components in Γ_a, each is discretized independently and orientations are adjusted for holes.
    comps=_boundary_components(boundary)
    pts=Vector{BoundaryPointsCFIE{T}}(undef,length(comps))
    for (idx,comp) in enumerate(comps)
        p=if length(comp)==1 # same logic as above, if hole is a smooth curve use the non graded dispatch
            base=CFIE_kress(q.pts_scaling_factor,billiard;min_pts=q.min_pts,eps=q.eps,symmetry=q.symmetry)
            _evaluate_points(base,comp[1],q_resolution,idx)
        else # grade it
            _evaluate_points(q,comp,q_resolution,idx)
        end
        pts[idx]=idx==1 ? p : _reverse_component_orientation(q,p) # all curves past the first one (first one can be a vector for a composite outer boundary) are holes so reverse their orientations
    end
    return pts
end

# Evaluate collocation points for all physical dielectric boundaries Γ_a.
# Each cavity Ω_a has boundary Γ_a=∂Ω_a and its own requested resolution (ppw)
# q_resolution[a]. For C>1 the local boundaries are discretized independently: Γ = Γ₁ ∪ Γ₂ ∪ ⋯ ∪ Γ_C.
# Any inter-cavity symmetry is applied only later to the assembled physical boundary nodes.
function evaluate_points(s::WiersigKress{T},q_resolution::AbstractVector{T})::Vector{BoundaryPointsCFIE{T}} where {T<:Real}
    C=length(s.billiards)
    length(q_resolution)==C||throw(DimensionMismatch("q_resolution has $(length(q_resolution)) entries for $C cavities"))
    # For one billiard retain the ordinary local boundary-component topology.
    if C==1
        return _wiersig_evaluate_local_points(s.quadrature,s.billiards[1],q_resolution[1])
    end
    # For several cavities Γ_a remains a complete physical cavity. Mixed
    # intra/inter reflection requires each complete periodic node set to have
    # N divisible by four. Only Wiersig's temporary local resolution is changed.
    pts=Vector{BoundaryPointsCFIE{T}}(undef,C)
    @inbounds for a in 1:C
        q=q_resolution[a]
        quadrature=s.quadrature[a]
        if s.symmetry isa WiersigMixedReflection
            L=s.billiards[a].length
            ppw=_wiersig_effective_ppw(s.ppw[a],q,L,quadrature.min_pts,s.symmetry)
            quadrature=_wiersig_with_ppw(quadrature,ppw)
        end
        pa=_wiersig_evaluate_local_points(quadrature,s.billiards[a],q)
        length(pa)==1||throw(ArgumentError("multi-cavity billiard $a must define one complete physical cavity"))
        pts[a]=pa[1]
    end
    return pts
end
# Evaluate every Γ_a at the same requested spectral resolution q_resolution, q_resolution,a = q_resolution, a=1,...,C. Just a way to specificy a single ppw for all the billiards in the system.
function evaluate_points(s::WiersigKress{T},q_resolution::T)::Vector{BoundaryPointsCFIE{T}} where {T<:Real}
    return evaluate_points(s,fill(q_resolution,length(s.billiards)))
end

# One reducedl inter-cavity symmetry orbit. `full` stores global boundary indices and `scales` the
# corresponding representation factors ρ_r used in reduced assembly.
struct WiersigOrbit{T<:Real}
    full::Vector{Int}
    scales::Vector{Complex{T}}
end

# struct holding the geometry metadata for the non-chebyshev pathway for Kress
struct WiersigMultiGeometry{C,G,P,S,O}
    components::C # workspace for each cavity
    Gs::G # conveniece field for geometry caches from components above. Holds information on the geometry and quadrature parameters such as curvatures, speed of the curve, circulant Kress R matrix etc.
    parr::P # for each panel its array cache tangents: X,Y,dX,dY... as matrices
    offs::Vector{Int} # for multibilliard case holds offsets that determine the index at which discretitzazon starts for the next billiard, e.g. Γ_a occupies offs[a]:offs[a+1]-1...
    # for a billiard we can determine the node by either the global index or by knowing the cavity and local displacement from its start from the offsets (offs). if g is a global index we can state it as g -> (a,j) where a is the cavity and j the local displacement. So global_to_block[g] gives the "a" and global_to_local[g] gives "j"
    global_to_block::Vector{Int}
    global_to_local::Vector{Int}
    Ntot::Int # total node count for the system
    Nred::Int # Number of boundary unknowns after symmetry reduction.
    symmetry::S # for easy access to symmetry struct if solver not available
    Ifund::Vector{Int} # indexes of the full non-desymmetrized domain that belong to the fundamental domain
    reduced_orbits::O # precomputed symmetr orbits
end

# Wrapper around the physical Kress geometry used by the Wiersig solver. Here for the subtyping of abstract type
struct WiersigGeometryWorkspace{G}<:AbstractWiersigGeometryWorkspace
    geom::G
end

# Return the interior refractive index associated with every physical cavity.
# A scalar n_in is broadcast to all cavities; otherwise one value is required
# for each physical dielectric component.
function _wiersig_component_indices(s::AbstractWiersigSolver,C::Int)
    length(s.n_in)==1&&return fill(s.n_in[1],C)
    length(s.n_in)==C||throw(DimensionMismatch("solver.n_in has $(length(s.n_in)) entries for $C cavities"))
    return s.n_in
end

# Construct inter-cavity C_n rotation orbits.
# The genuinely new operation here is only the mapping between physical cavity
# blocks. Cavities are ordered by complete symmetry orbits: (1,...,n), (n+1,...,2n), ...
# and a rotation preserves the local boundary-node index inside each rotated
# copy because the full physical cavities were generated with that convention.
function _wiersig_multicavity_orbits(::Type{T},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},sym::Rotation) where {T<:Real}
    C=length(pts) # this determines the number of independant billiards forming the system
    n=sym.n 
    n>=2||throw(ArgumentError("multi-cavity Rotation requires order n≥2")) # if n=2 just use reflection
    mod(C,n)==0||throw(ArgumentError("$C cavities cannot be partitioned into complete C_$n inter-cavity orbits")) # e.g. cant have say 4 billiard and expect they are related by C_3
    # sym.m selects the irrep. χ[l+1] is the character of the group element R^l: χ_l=exp(2πiml/n), l=0,...,n-1.
    _,_,χ=_rotation_tables(T,n,sym.m)
    Ifund=Int[] # full node to desymmetrized node index mapping, can be either 
    orbits=WiersigOrbit{T}[] 
    # Each group g0:g0+n-1 is one complete orbit of physical cavities. Each g0 is the first cavity of one independent C_n symmetry orbit. if n=3 and C=9 we get g0 \in [1,4,7] so we have:
    # g0=1 -> 1:3 -> cavities (1,2,3)
    # g0=4 -> 4:6 -> cavities (4,5,6)
    # g0=7 -> 7:9 -> cavities (7,8,9)
    for g0 in 1:n:C
        Ns=[length(pts[a].xy) for a in g0:g0+n-1] # Node counts of all cavities in the current C_n orbit.
        N=Ns[1] # take the "fundamental domain" slice
        all(==(N),Ns)||throw(DimensionMismatch("rotation-related cavities $(g0):$(g0+n-1) must have identical collocation counts; received $Ns")) # otherwise cant nicely construct the Ifund mapping by just index divisions.
        append!(Ifund,offs[g0]:(offs[g0+1]-1)) # get the cavity g0's indexes in the full system as those are the fundamental indexes
        @inbounds for j in 1:N
            inds=Vector{Int}(undef,n) # for each cavity g0 these will be 
            scales=Vector{Complex{T}}(undef,n)
            for l in 0:n-1 # Enumerate the images R^l of local node j within this cavity orbit.
                # For cavity a offs[a] is the global index of its first boundary node. So inside a rotational orbit that starts at cavity g0, the l-th rotated cavity is a=g0+l. Therefore its local node j has global index g=offs[g0+l]+j-1
                inds[l+1]=offs[g0+l]+j-1
                scales[l+1]=χ[l+1] # Phase of the chosen irrep m on the orbit generated by R^l.
            end
            push!(orbits,WiersigOrbit{T}(inds,scales))
        end
    end
    return Ifund,orbits
end
# Construct symmetry orbits for inter-cavity reflections.
# For :x_axis or :y_axis, cavities are ordered in reflected pairs
# (Γ₁,Γ₂),(Γ₃,Γ₄),... . The first cavity in each pair is the representative.
# Reflection reverses the periodic node ordering, so j→jr with parity p=±1.
# For :origin, cavities are ordered as (Γ,RxΓ,RyΓ,RxRyΓ) with scales
# (1,px,py,px*py); single reflections reverse node order, the double does not.
function _wiersig_multicavity_orbits(::Type{T},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},sym::Reflection) where {T<:Real}
    C=length(pts)
    Ifund=Int[]
    orbits=WiersigOrbit{T}[]
    if sym.axis===:x_axis||sym.axis===:y_axis
        # A single reflection partitions the physical cavities into pairs.
        iseven(C)||throw(ArgumentError("multi-cavity reflection requires an even number of cavities; received C=$C"))
        p=Complex{T}(sym.parity,0) # p=±1 selects the even/odd reflection sector.
        # g0 is the first cavity of each reflected pair: (1,2),(3,4),(5,6),...
        for g0 in 1:2:C
            N1=length(pts[g0].xy)
            N2=length(pts[g0+1].xy)
            # Reflection-related cavities must have identical node counts so each node has a unique reflected partner.
            N1==N2||throw(DimensionMismatch("reflection-related cavities $g0 and $(g0+1) must have identical collocation counts; received $N1 and $N2"))
            append!(Ifund,offs[g0]:(offs[g0+1]-1)) # Keep the first cavity of the pair as the reduced representative.
            @inbounds for j in 1:N1
                jr=_idx_reflect_half_ccw(j,N1) # Reflection reverses the periodic CCW node ordering.
                # inds contains the physical node and its reflected image; scales imposes x(Rg)=p*x(g).
                inds=[offs[g0]+j-1,offs[g0+1]+jr-1]
                scales=Complex{T}[one(Complex{T}),p]
                push!(orbits,WiersigOrbit{T}(inds,scales))
            end
        end
        return Ifund,orbits
    elseif sym.axis===:origin
        # Independent x- and y-reflections produce four physical cavities per complete orbit.
        mod(C,4)==0||throw(ArgumentError("multi-cavity :origin reflection requires C divisible by 4; received C=$C"))
        px=Complex{T}(sym.parity[1],0) # x-reflection parity ±1.
        py=Complex{T}(sym.parity[2],0) # y-reflection parity ±1.
        # Cavities are ordered in groups (Γ,RxΓ,RyΓ,RxRyΓ).
        for g0 in 1:4:C
            Ns=[length(pts[a].xy) for a in g0:g0+3]
            N=Ns[1]
            all(==(N),Ns)||throw(DimensionMismatch("four reflection-related cavities $g0:$(g0+3) must have identical collocation counts; received $Ns"))
            append!(Ifund,offs[g0]:(offs[g0+1]-1)) # Keep the unreflected cavity Γ as representative.
            @inbounds for j in 1:N
                # A single reflection reverses node ordering; applying both reflections restores the original ordering.
                jr=_idx_reflect_half_ccw(j,N)
                # Physical images are Γ:j, RxΓ:jr, RyΓ:jr, RxRyΓ:j.
                inds=[offs[g0]+j-1,offs[g0+1]+jr-1,offs[g0+2]+jr-1,offs[g0+3]+j-1]
                # Representation factors for I,Rx,Ry,RxRy.
                scales=Complex{T}[one(Complex{T}),px,py,px*py]
                push!(orbits,WiersigOrbit{T}(inds,scales))
            end
        end
        return Ifund,orbits
    end
    throw(ArgumentError("unsupported Reflection axis $(sym.axis)"))
end
# Construct symmetry orbits for combined intra- and inter-cavity reflections.
# Physical cavities are ordered in reflected pairs (Γ₁,Γ₂),(Γ₃,Γ₄),... .
# The first cavity in each pair is the representative.
# Each complete physical cavity uses the native XY-compatible periodic boundary
# ordering. For local node q on the representative cavity,
#   qx  = R_x q   : x -> -x,
#   qy  = R_y q   : y -> -y,
#   qxy = R_xR_yq : (x,y) -> (-x,-y).
# R_x acts inside each physical cavity, while the global R_y reflection exchanges
# the two vertically stacked cavities and simultaneously applies the local
# y-reflection to their boundary coordinates. Hence the complete Z₂×Z₂ orbit is
#   I:       Γ_g0:q       with scale 1,
#   R_x:     Γ_g0:qx      with scale p_in,
#   R_y:     Γ_g0+1:qy    with scale p_ex,
#   R_xR_y:  Γ_g0+1:qxy   with scale p_in*p_ex.
function _wiersig_multicavity_orbits(::Type{T},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},sym::WiersigMixedReflection) where {T<:Real}
    C=length(pts)
    iseven(C)||throw(ArgumentError("mixed reflection requires cavity pairs; received C=$C"))
    pin=Complex{T}(sym.intra_parity,0) # x-reflection parity inside each cavity.
    pex=Complex{T}(sym.inter_parity,0) # y-reflection parity exchanging cavity pairs.
    Ifund=Int[]
    orbits=WiersigOrbit{T}[]
    visited=falses(offs[end]-1)
    for g0 in 1:2:C
        N1=length(pts[g0].xy)
        N2=length(pts[g0+1].xy)
        N1==N2||throw(DimensionMismatch("mixed-reflection cavities $g0 and $(g0+1) must have identical node counts; received $N1 and $N2"))
        mod(N1,4)==0||throw(ArgumentError("mixed XY reflection requires boundary node count divisible by 4; received N=$N1"))
        @inbounds for q in 1:N1
            g=offs[g0]+q-1
            visited[g]&&continue
            qx=_idx_reflect_x_quarter_ccw(q,N1) # x -> -x on the representative cavity.
            qy=_idx_reflect_y_quarter_ccw(q,N1) # y -> -y, mapping representative to its exchanged cavity.
            qxy=_idx_rotate_pi_ccw(q,N1) # composition R_x * R_y.
            inds=[
                offs[g0]+q-1,
                offs[g0]+qx-1,
                offs[g0+1]+qy-1,
                offs[g0+1]+qxy-1
            ]
            scales=Complex{T}[one(Complex{T}),pin,pex,pin*pex]
            uinds=Int[]
            uscales=Complex{T}[]
            allowed=true
            for r in eachindex(inds)
                k=findfirst(==(inds[r]),uinds)
                if isnothing(k)
                    push!(uinds,inds[r])
                    push!(uscales,scales[r])
                elseif uscales[k]!=scales[r]
                    allowed=false # Odd parity at a fixed point forces the boundary value to vanish.
                    break
                end
            end
            for gi in unique(inds)
                visited[gi]=true
            end
            allowed||continue
            push!(Ifund,g)
            push!(orbits,WiersigOrbit{T}(uinds,uscales))
        end
    end
    return Ifund,orbits
end

function _wiersig_multicavity_orbits(::Type{T},pts::Vector{BoundaryPointsCFIE{T}},offs::Vector{Int},sym) where {T<:Real}
    throw(ArgumentError("unsupported multi-cavity symmetry type $(typeof(sym)); supported symmetries are Rotation and Reflection or WiersigMixedReflection"))
end

# Native single-billiard symmetry.
function _wiersig_validate_material_symmetry(solver::WiersigKress,geom::CFIEKressWorkspace)
    isnothing(geom.symmetry)&&return true
    nin=_wiersig_component_indices(solver,length(geom.Gs))
    for b in 1:geom.Nred
        orb=geom.reduced_orbits[b]
        isempty(orb.full)&&continue
        a0=geom.global_to_block[orb.full[1]]
        n0=nin[a0]
        @inbounds for r in 2:length(orb.full)
            a=geom.global_to_block[orb.full[r]]
            nin[a]==n0||throw(ArgumentError("symmetry orbit crosses components with different refractive indices"))
        end
    end
    return true
end

# Inter-cavity symmetry g:Γ_a→Γ_b with accounting for refractive indexes. This should error if user
# incorrectly inputs different n_in and expecting symmetry to work
function _wiersig_validate_material_symmetry(solver::WiersigKress,geom::WiersigMultiGeometry)
    isnothing(geom.symmetry)&&return true
    C=length(geom.offs)-1
    nin=_wiersig_component_indices(solver,C)
    for orb in geom.reduced_orbits
        a0=geom.global_to_block[orb.full[1]]
        n0=nin[a0]
        @inbounds for r in 2:length(orb.full)
            a=geom.global_to_block[orb.full[r]]
            nin[a]==n0||throw(ArgumentError("symmetry orbit contains cavities $a0 and $a with different refractive indices $n0 and $(nin[a])"))
        end
    end
    return true
end

# Validate material invariance under the active symmetry.
function _wiersig_validate_material_symmetry(solver::WiersigKress,ws::WiersigGeometryWorkspace)
    return _wiersig_validate_material_symmetry(solver,ws.geom)
end

function build_wiersig_workspace(s::WiersigKress,pts::Vector{<:BoundaryPointsCFIE})
    return build_cfie_kress_workspace(s,pts)
end

"""
    split_wiersig_trace(x)

Split the Wiersig boundary trace `x=[φ;ψ]` into equal-length views `(φ,ψ)`.
"""
function split_wiersig_trace(x::AbstractVector)
    iseven(length(x))||throw(DimensionMismatch("Wiersig trace length must be even"))
    N=length(x)÷2
    return (@view x[1:N]),(@view x[N+1:2N])
end

"""
    expand_wiersig_trace(x,ws)

Expand a symmetry-reduced trace to the complete physical boundary using the same
inter-cavity orbit relation employed by reduced matrix assembly,

    φ(g_{b,r})=ρ_rφ_b,   ψ(g_{b,r})=ρ_rψ_b.

Without symmetry the physical boundary indexing is unchanged.
"""
function expand_wiersig_trace(x::AbstractVector{<:Number},ws::AbstractWiersigGeometryWorkspace)
    geom=ws.geom
    if isnothing(geom.symmetry)
        T=typeof(real(zero(eltype(x))))
        return Complex{T}.(x)
    end
    φred,ψred=split_wiersig_trace(x)
    length(φred)==geom.Nred||throw(DimensionMismatch("reduced trace block has length $(length(φred)); expected $(geom.Nred)"))
    C=promote_type(ComplexF64,eltype(x))
    φfull=zeros(C,geom.Ntot)
    ψfull=zeros(C,geom.Ntot)
    @inbounds for b in 1:geom.Nred
        orb=geom.reduced_orbits[b]
        for r in eachindex(orb.full)
            g=orb.full[r]
            ρ=orb.scales[r]
            φfull[g]=ρ*φred[b]
            ψfull[g]=ρ*ψred[b]
        end
    end
    return vcat(φfull,ψfull)
end

"""
    wiersig_component_range(ws,a)

Return the complete physical boundary-index range belonging to cavity `Γ_a`.
"""
@inline function wiersig_component_range(ws::AbstractWiersigGeometryWorkspace,a::Int)
    return ws.geom.offs[a]:(ws.geom.offs[a+1]-1)
end

"""
    split_wiersig_components(x,ws)

Split a complete physical trace `x=[φ;ψ]` into cavitywise views `(φs,ψs)`.
A symmetry-reduced trace must first be expanded with `expand_wiersig_trace`.
"""
function split_wiersig_components(x::AbstractVector,ws::AbstractWiersigGeometryWorkspace)
    geom=ws.geom
    φ,ψ=split_wiersig_trace(x)
    length(φ)==geom.Ntot||throw(DimensionMismatch("physical trace block has length $(length(φ)); expected $(geom.Ntot)"))
    C=length(geom.offs)-1
    φs=[@view φ[geom.offs[a]:(geom.offs[a+1]-1)] for a in 1:C]
    ψs=[@view ψ[geom.offs[a]:(geom.offs[a+1]-1)] for a in 1:C]
    return φs,ψs
end

# Geometry and Chebyshev lookup data for one ordered interaction Γ_a←Γ_b.
# Self blocks carry the local Kress split from cavity a; cross-cavity blocks
# are smooth and therefore have no logarithmic/Kress correction.
struct WiersigKressBlockCache{T<:Real}
    Ni::Int # Number of target nodes on Γ_a.
    Nj::Int # Number of source nodes on Γ_b.
    row_offset::Int # First global matrix index belonging to target cavity Γ_a.
    col_offset::Int # First global matrix index belonging to source cavity Γ_b.
    same::Bool # True for the singular self-block Γ_a←Γ_a, false for smooth cross-cavity blocks.
    R::Matrix{T} # Pairwise distances r_ij=|x_i-y_j|.
    invR::Matrix{T} # Pairwise inverse distances 1/r_ij; diagonal is only relevant for off-diagonal evaluation.
    inner::Matrix{T} # Source-normal DLP numerator s_j n_j⋅(x_i-y_j).
    speed_i::Vector{T} # Boundary parameter speeds s_i on the target.
    speed_j::Vector{T} # Boundary parameter speeds s_j on the source.
    wi::Vector{T} # Quadrature weights associated with target nodes.
    wj::Vector{T} # Quadrature weights associated with source nodes.
    pidx::Matrix{Int32} # Hankel Chebyshev panel index for each distance r_ij; 0 means direct small-r evaluation.
    tloc::Matrix{Float64} # Hankel local Chebyshev coordinate in [-1,1].
    pidxj::Matrix{Int32} # Bessel-J Chebyshev panel index for each distance r_ij.
    tlocj::Matrix{Float64} # Bessel-J local Chebyshev coordinate in [-1,1].
    logterm::Union{Nothing,Matrix{T}} # Kress periodic logarithm; present only for self-blocks.
    kappa_i::Union{Nothing,Vector{T}} # Target-boundary curvature used in the DLP diagonal limit.
    Rkress::Union{Nothing,Matrix{T}} # Cavity-specific Kress product-integration matrix; present only for self-blocks.
end

# Complete ordered physical-cavity block cache together with the common radial interval used by all H₀/H₁/J₀/J₁ Chebyshev plans.
struct WiersigKressBlockSystemCache{T<:Real}
    blocks::Matrix{WiersigKressBlockCache{T}} # Ordered block cache blocks[a,b] for Γ_a←Γ_b.
    rmin::Float64 # rmin of the radial interval, used only for bounds
    rmax::Float64 # rmax of the radial interval
end

# Determine the common nonzero radial interval needed by all cavity interactions.
# Self-block distances are already available in the direct geometry cache; cross
# blocks are scanned once because R_ab and R_ba contain the same distances.
function _wiersig_radial_bounds(pts::Vector{BoundaryPointsCFIE{T}},direct_ws::WiersigGeometryWorkspace;rmin_cheb::Union{Nothing,Float64}=nothing,pad=(T(0.95),T(1.05))) where {T<:Real}
    geom=direct_ws.geom
    C=length(pts)
    rmin0=typemax(T)
    rmax0=zero(T)
    @inbounds for a in 1:C
        Pa=geom.parr[a]
        Na=length(Pa.X)
        G=geom.Gs[a]
        for j in 2:Na,i in 1:j-1
            r=G.R[i,j]
            if isfinite(r)&&r>eps(T)
                r<rmin0&&(rmin0=r)
                r>rmax0&&(rmax0=r)
            end
        end
        for b in a+1:C
            Pb=geom.parr[b]
            Nb=length(Pb.X)
            for j in 1:Nb,i in 1:Na
                dx=Pa.X[i]-Pb.X[j]
                dy=Pa.Y[i]-Pb.Y[j]
                r=sqrt(muladd(dx,dx,dy*dy))
                r>eps(T)||throw(ArgumentError("distinct dielectric cavities $a and $b touch or overlap"))
                r<rmin0&&(rmin0=r)
                r>rmax0&&(rmax0=r)
            end
        end
    end
    isfinite(rmin0)&&rmax0>zero(T)||throw(ArgumentError("could not determine a nonzero Wiersig radial interval"))
    rrmin=Float64(pad[1]*rmin0)
    rrmax=Float64(pad[2]*rmax0)
    rmin_h=isnothing(rmin_cheb) ? rrmin : max(Float64(rmin_cheb),rrmin)
    return rmin_h,rrmax
end

# Build all ordered Γ_a←Γ_b geometry caches on one common radial Chebyshev grid.
# Self-blocks inherit their cavity-specific Kress data from the direct workspace;
# cross-cavity blocks are smooth and therefore require no Kress correction.
function build_wiersig_kress_block_caches(solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},direct_ws::WiersigGeometryWorkspace;npanels_h::Int=10000,M_h::Int=5,npanels_j::Int=2000,M_j::Int=5,pad=(T(0.95),T(1.05)),rmin_cheb::Union{Nothing,Float64}=nothing) where {T<:Real}
    geom=direct_ws.geom
    C=length(pts)
    length(geom.parr)==C||throw(DimensionMismatch("geometry has $(length(geom.parr)) blocks for $C physical boundaries"))
    rmin_h,rrmax=_wiersig_radial_bounds(pts,direct_ws;rmin_cheb=rmin_cheb,pad=pad)
    # Build reference radial panelizations only. Their q=1 special-function
    # values are irrelevant; we use the panels solely to map each geometric
    # distance r_ij to a cached panel index and local Chebyshev coordinate.
    pref_plan_h=plan_h(0,1,1.0+0im,rmin_h,rrmax;npanels=npanels_h,M=M_h)
    pref_plan_j=plan_j(1,1.0+0im,0.0,rrmax;npanels=npanels_j,M=M_j)
    pansh=pref_plan_h.panels
    pansj=pref_plan_j.panels
    # Build one cache for every ordered interaction Γ_a←Γ_b. The target and
    # source cavities may have different node counts and different quadratures.
    blocks=Matrix{WiersigKressBlockCache{T}}(undef,C,C)
    @inbounds for a in 1:C,b in 1:C
        Pa=geom.parr[a];Pb=geom.parr[b];Na=length(Pa.X);Nb=length(Pb.X)
        R=Matrix{T}(undef,Na,Nb);invR=Matrix{T}(undef,Na,Nb);inner=Matrix{T}(undef,Na,Nb)
        speed_i=collect(Pa.s);speed_j=collect(Pb.s);wi=collect(pts[a].ws);wj=collect(pts[b].ws)
        same=a==b
        # Same-cavity blocks contain the logarithmic singularity. Reuse the
        # geometry and Kress matrix already constructed with cavity a's own
        # local quadrature instead of trying to rebuild or infer that rule here.
        if same
            G=geom.Gs[a]
            copyto!(R,G.R)
            copyto!(invR,G.invR)
            copyto!(inner,G.inner)
            logterm=copy(G.logterm)
            kappa_i=copy(G.kappa)
            if geom isa WiersigMultiGeometry
                Rkress=Matrix{T}(geom.components[a].Rmat)
            else
                ra=geom.offs[a]:(geom.offs[a+1]-1)
                Rkress=Matrix{T}(@view geom.Rmat[ra,ra])
            end
        else
            # Different physical cavities are disjoint, hence their kernels are
            # smooth. Compute only the ordinary distance and DLP geometry needed
            # for direct Nyström evaluation; no Kress split data are defined.
            for j in 1:Nb,i in 1:Na
                dx=Pa.X[i]-Pb.X[j];dy=Pa.Y[i]-Pb.Y[j];r2=muladd(dx,dx,dy*dy)
                r2>eps(T)^2||throw(ArgumentError("distinct dielectric cavities $a and $b touch or overlap"))
                r=sqrt(r2)
                R[i,j]=r
                invR[i,j]=inv(r)
                inner[i,j]=Pb.dY[j]*dx-Pb.dX[j]*dy
            end
            logterm=nothing
            kappa_i=nothing
            Rkress=nothing
        end
        # Precompute the Hankel and Bessel-J radial lookup coordinates for every
        # matrix entry. Diagonal self-interactions are handled analytically, and
        # Hankel distances below rmin_h are marked for direct small-r evaluation.
        pidx=Matrix{Int32}(undef,Na,Nb);tloc=Matrix{Float64}(undef,Na,Nb)
        pidxj=Matrix{Int32}(undef,Na,Nb);tlocj=Matrix{Float64}(undef,Na,Nb)
        for j in 1:Nb,i in 1:Na
            if same&&i==j
                pidx[i,j]=Int32(1);tloc[i,j]=0.0
                pidxj[i,j]=Int32(1);tlocj[i,j]=0.0
                continue
            end
            r=Float64(R[i,j])
            if r<rmin_h
                pidx[i,j]=Int32(0)
                tloc[i,j]=0.0
            else
                p=_find_panel(pref_plan_h,r);P=pansh[p]
                pidx[i,j]=Int32(p)
                tloc[i,j]=(2*r-(P.b+P.a))/(P.b-P.a)
            end
            pj=_find_panel(pref_plan_j,r);Pj=pansj[pj]
            pidxj[i,j]=Int32(pj)
            tlocj[i,j]=(2*r-(Pj.b+Pj.a))/(Pj.b-Pj.a)
        end
        # Store the complete k-independent block data used later by the multi-q
        # Wiersig Chebyshev assembly.
        blocks[a,b]=WiersigKressBlockCache{T}(Na,Nb,geom.offs[a],geom.offs[b],same,R,invR,inner,speed_i,speed_j,wi,wj,pidx,tloc,pidxj,tlocj,logterm,kappa_i,Rkress)
    end
    return WiersigKressBlockSystemCache{T}(blocks,rmin_h,rrmax)
end

# Tune H0/H1/J0/J1 Chebyshev plans over the complete Wiersig radial range.
# The geometry cache uses the local Kress rule associated with every Γ_a.
function chebyshev_params(solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},direct_ws::WiersigGeometryWorkspace,zj::AbstractVector{Complex{T}};npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,tol::Real=1e-10,sampling_points::Int=50_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,verbose::Bool=false) where {T<:Real}
    rmin_cheb=minimum(hankel_z_chebyshev_cutoff./abs.(zj))
    rmin_h,rmax=_wiersig_radial_bounds(pts,direct_ws;rmin_cheb=rmin_cheb)
    rsH=collect(range(Float64(rmin_h),Float64(rmax);length=sampling_points))
    rsJ=collect(range(0.0,Float64(rmax);length=sampling_points))
    nz=length(zj);nh=npanels_h_init;nj=npanels_j_init;Mh=M_h_init;Mj=M_j_init
    plans0=Vector{ChebHankelPlanH}(undef,nz)
    plans1=Vector{ChebHankelPlanH}(undef,nz)
    plansj0=Vector{ChebJPlan}(undef,nz)
    plansj1=Vector{ChebJPlan}(undef,nz)
    errH0=fill(Inf,nz)
    errH1=fill(Inf,nz)
    errJ0=fill(Inf,nz)
    errJ1=fill(Inf,nz)
    for it in 1:max_iter
        plans0,plans1,plansj0,plansj1=build_CFIE_plans_kress(zj,Float64(rmin_h),Float64(rmax);npanels_h=nh,M_h=Mh,npanels_j=nj,M_j=Mj,nthreads=Threads.nthreads())
        _check_H0H1_errors!(errH0,errH1,plans0,plans1,zj,rsH)
        _check_J0J1_errors!(errJ0,errJ1,plansj0,plansj1,zj,rsJ)
        okH=all(<(tol),errH0)&&all(<(tol),errH1)
        okJ=all(<(tol),errJ0)&&all(<(tol),errJ1)
        verbose&&@info "Worst Wiersig H0 H1 J0 J1 | nh Mh nj Mj" maximum(errH0) maximum(errH1) maximum(errJ0) maximum(errJ1) nh Mh nj Mj
        okH&&okJ&&return nh,Mh,nj,Mj,plans0,plans1,plansj0,plansj1,errH0,errH1,errJ0,errJ1
        if !okH
            it%5==0 ? (Mh+=grow_M) : (nh=ceil(Int,grow_panels*nh))
        end
        if !okJ
            it%5==0 ? (Mj+=grow_M) : (nj=ceil(Int,grow_panels*nj))
        end
    end
    @warn "Wiersig Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort."
    return nh,Mh,nj,Mj,plans0,plans1,plansj0,plansj1,errH0,errH1,errJ0,errJ1
end

struct WiersigChebyshevWorkspace{T<:Real,DW,BC,P0,P1,PJ0,PJ1,B}
    direct_ws::DW
    block_cache::BC
    ks::Vector{Complex{T}}
    qin::Matrix{Complex{T}}
    qout::Vector{Complex{T}}
    qall::Vector{Complex{T}}
    ncavities::Int
    plans0::P0
    plans1::P1
    plansj0::PJ0
    plansj1::PJ1
    bessel_ws::B
    npanels_h::Int
    M_h::Int
    npanels_j::Int
    M_j::Int
    errH0::Vector{Float64}
    errH1::Vector{Float64}
    errJ0::Vector{Float64}
    errJ1::Vector{Float64}
end

@inline wiersig_chebyshev_nk(cws::WiersigChebyshevWorkspace)=length(cws.ks)
@inline wiersig_chebyshev_ncavities(cws::WiersigChebyshevWorkspace)=cws.ncavities
@inline _wiersig_qin_index(cws::WiersigChebyshevWorkspace,a::Int,m::Int)=(a-1)*length(cws.ks)+m
@inline _wiersig_qout_index(cws::WiersigChebyshevWorkspace,m::Int)=cws.ncavities*length(cws.ks)+m
@inline boundary_size(cws::WiersigChebyshevWorkspace)=boundary_size(cws.direct_ws)
@inline boundary_matrix_size(cws::WiersigChebyshevWorkspace)=2*boundary_size(cws)
@inline _wiersig_qin_range(M::Int,a::Int)=((a-1)*M+1):(a*M)
@inline _wiersig_qout_range(M::Int,C::Int)=(C*M+1):((C+1)*M)

function build_chebyshev_workspace(s::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},ks::AbstractVector;npanels_h_init::Int=15_000,M_h_init::Int=5,npanels_j_init::Int=3_000,M_j_init::Int=5,tol::Real=1e-11,sampling_points::Int=50_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,plan_threads::Int=Threads.nthreads(),verbose::Bool=true) where {T<:Real}
    C=length(pts)
    nin=_wiersig_component_indices(s,C)
    kvec=Complex{T}.(ks)
    M=length(kvec)
    direct_ws=build_wiersig_workspace(s,pts)
    geom=direct_ws.geom
    qin=Matrix{Complex{T}}(undef,C,M)
    @inbounds for a in 1:C,m in 1:M
        qin[a,m]=Complex{T}(nin[a])*kvec[m]
    end
    qout=Complex{T}(s.n_out).*kvec
    qall=Vector{Complex{T}}(undef,(C+1)*M)
    @inbounds for a in 1:C,m in 1:M
        qall[(a-1)*M+m]=qin[a,m]
    end
    @inbounds for m in 1:M
        qall[C*M+m]=qout[m]
    end
    rmin_cheb=minimum(hankel_z_chebyshev_cutoff./abs.(qall))
    tuned=chebyshev_params(s,pts,direct_ws,qall;npanels_h_init=npanels_h_init,M_h_init=M_h_init,npanels_j_init=npanels_j_init,M_j_init=M_j_init,tol=tol,sampling_points=sampling_points,max_iter=max_iter,grow_panels=grow_panels,grow_M=grow_M,verbose=verbose)
    nh,Mh,nj,Mj,plans0,plans1,plansj0,plansj1,errH0,errH1,errJ0,errJ1=tuned
    block_cache=build_wiersig_kress_block_caches(s,pts,direct_ws;npanels_h=nh,M_h=Mh,npanels_j=nj,M_j=Mj,rmin_cheb=rmin_cheb)
    Mq=(C+1)*M
    bfs=CFIE_H0_H1_J0_J1_BesselWorkspace(Mq;ntls=Threads.nthreads())
    if verbose
        println("Wiersig Kress Chebyshev workspace ready")
        println("physical cavities      = ",C)
        println("vacuum wavenumbers     = ",M)
        println("material q families    = ",C+1)
        println("total q plans          = ",Mq)
        println("full boundary nodes    = ",geom.Ntot)
        println("active boundary nodes  = ",boundary_size(direct_ws))
        println("matrix dimension       = ",boundary_matrix_size(direct_ws))
        println("symmetry               = ",geom.symmetry)
        println("H panels/degree        = ",nh,"/",Mh)
        println("J panels/degree        = ",nj,"/",Mj)
        println("radial interval        = [",block_cache.rmin,",",block_cache.rmax,"]")
        println("max H0/H1/J0/J1 error = ",(maximum(errH0),maximum(errH1),maximum(errJ0),maximum(errJ1)))
    end
    return WiersigChebyshevWorkspace(direct_ws,block_cache,kvec,qin,qout,qall,C,plans0,plans1,plansj0,plansj1,bfs,nh,Mh,nj,Mj,Float64.(errH0),Float64.(errH1),Float64.(errJ0),Float64.(errJ1))
end

"""
For source-normal D, inner_ij=s_j n_j·(x_i-x_j).
For target-normal D', D'_{ij}=D_{ji}(ds_j/ds_i),
so the parameter-space normal numerator is inner'_ij=(s_j/s_i)inner_ji.
This is a transpose relation, not a Hermitian adjoint.
"""
@inline function _wiersig_dlp_normal_mode(dlp_kernel::Symbol)
    dlp_kernel===:source&&return Val(:source)
    dlp_kernel===:target&&return Val(:target)
    throw(ArgumentError("dlp_kernel must be :source or :target; received $dlp_kernel"))
end
@inline _wiersig_direct_dlp_inner(::Val{:source},inn_ij,inn_ji,si,sj)=inn_ij
@inline _wiersig_direct_dlp_inner(::Val{:target},inn_ij,inn_ji,si,sj)=(sj/si)*inn_ji

"""
    build_cfie_kress_workspace(solver::WiersigKress,pts)

Construct the k-independent workspace using the standard CFIE solver API.
For C=1, the native QuantumBilliards geometry and symmetry machinery is used.
For C>1, one unreduced local Kress workspace is constructed per complete
dielectric cavity. These local workspaces are concatenated, after which any
global symmetry is represented by exact inter-cavity index orbits.
"""
function build_cfie_kress_workspace(solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    if !_wiersig_is_multibilliard(solver)
        # One billiard retains its native CFIE component topology and internal
        # symmetry interpretation.
        _wiersig_component_indices(solver,length(pts))
        geom=build_cfie_kress_workspace(solver.quadrature,pts);ws=WiersigGeometryWorkspace{typeof(geom)}(geom)
        _wiersig_validate_material_symmetry(solver,ws)
        return ws
    end
    C=length(solver.billiards);length(pts)==C||throw(DimensionMismatch("pts has $(length(pts)) entries but solver contains $C dielectric cavities"))
    _wiersig_component_indices(solver,C)
    # Each complete Γ_a receives its own singular Kress geometry. Different
    # quadrature kinds are therefore harmless: cross-cavity kernels never use
    # another cavity's Kress correction matrix.
    locals=map(1:C) do a
        build_cfie_kress_workspace(solver.quadrature[a],BoundaryPointsCFIE{T}[pts[a]])
    end
    @inbounds for a in 1:C
        length(locals[a].Gs)==1||throw(ArgumentError("cavity $a generated $(length(locals[a].Gs)) local boundary components; expected one complete dielectric interface"))
        # Multi-billiard symmetry is exclusively inter-cavity, so each local
        # boundary must remain physically complete and unreduced.
        isnothing(locals[a].symmetry)||error("multi-billiard local Kress workspaces must be unreduced")
    end
    # Build global offsets Γ_a = offs[a]:offs[a+1]-1.
    offs=Vector{Int}(undef,C+1);offs[1]=1
    @inbounds for a in 1:C
        offs[a+1]=offs[a]+locals[a].Ntot
    end
    Ntot=offs[end]-1;g2b=Vector{Int}(undef,Ntot);g2l=Vector{Int}(undef,Ntot)
    # Precompute g↦(a,j). This determines in O(1) whether a source image belongs
    # to the same dielectric interior as a target representative.
    @inbounds for a in 1:C
        o=offs[a];Na=locals[a].Ntot
        for j in 1:Na
            g=o+j-1;g2b[g]=a;g2l[g]=j
        end
    end
    Gs=Tuple(locals[a].Gs[1] for a in 1:C);parr=Tuple(locals[a].parr[1] for a in 1:C)
    if isnothing(solver.symmetry)
        # Without reduction, every physical full-boundary node is independent.
        geom=WiersigMultiGeometry(Tuple(locals),Gs,parr,offs,g2b,g2l,Ntot,Ntot,nothing,collect(1:Ntot),WiersigOrbit{T}[])
        return WiersigGeometryWorkspace{typeof(geom)}(geom)
    end
    # Build source orbits exclusively by cavity/index arithmetic.
    Ifund,orbits=_wiersig_multicavity_orbits(T,pts,offs,solver.symmetry)
    geom=WiersigMultiGeometry(Tuple(locals),Gs,parr,offs,g2b,g2l,Ntot,length(Ifund),solver.symmetry,Ifund,orbits)
    _wiersig_validate_material_symmetry(solver,geom)
    return WiersigGeometryWorkspace{typeof(geom)}(geom)
end

# helpers to determine the size of the matrix from the workspace
boundary_size(ws::AbstractWiersigGeometryWorkspace)::Int=isnothing(ws.geom.symmetry) ? ws.geom.Ntot : ws.geom.Nred
boundary_matrix_size(ws::AbstractWiersigGeometryWorkspace)::Int=2*boundary_size(ws)
@inline _wiersig_local_rmat(geom::WiersigMultiGeometry,a::Int,i::Int,j::Int)=geom.components[a].Rmat[i,j]

"""
    _wiersig_assemble_same_component!(S,D,pts,geom,a,q;dlp_kernel=:source,multithreaded=true)

Assemble the singular self-interaction Nyström blocks `S_aa(q)` and `D_aa(q)` for one physical dielectric boundary Γ_a.
This is the only boundary-boundary interaction for which the Helmholtz kernels are singular. If two source and target points belong to distinct physical cavities Γ_a and Γ_b with `a≠b`, their separation is nonzero and ordinary smooth quadrature is used instead.
The Wiersig Green function is `G_q(x,y)=-(i/4)H₀⁽¹⁾(q|x-y|)`. The doubled boundary operators used here are therefore `S_q(x,y)=-2G_q(x,y)=(i/2)H₀⁽¹⁾(q|x-y|)` and `D_q(x,y)=2∂_{n_y}G_q(x,y)=-(iq/2)H₁⁽¹⁾(q|x-y|)((x-y)⋅n_y)/|x-y|`.
For a counterclockwise boundary parameterization `γ(t)=(x(t),y(t))`, write `γ'(t)=(x'(t),y'(t))`, `s(t)=|γ'(t)|` and `s(t)n(t)=(y'(t),-x'(t))`. The geometric DLP numerator stored by the workspace is therefore `inner_ij=y'_j(x_i-x_j)-x'_j(y_i-y_j)=s_j n_j⋅(x_i-x_j)`. The source Jacobian is represented by `s_j`, while `w_j` is the quadrature weight in the periodic computational variable.

Kress logarithmic splitting
---------------------------
For `i≠j`, the doubled DLP is written as `D_ij=R_ijL₁(i,j)+w_jL₂(i,j)`, where `L₁=q/(2π)J₁(qr)inner/r` and `L₂=-(iq/2)H₁⁽¹⁾(qr)inner/r-L₁logterm`.
The doubled SLP is written as `S_ij=R_ijM₁(i,j)+w_jM₂(i,j)`, where `M₁=-(1/(2π))J₀(qr)s_j` and `M₂=(i/2)H₀⁽¹⁾(qr)s_j-M₁logterm`.
Here `r=|x_i-x_j|`. `logterm` is the universal periodic logarithm isolated by the Kress decomposition, while `R_ij` is the dense Kress product-integration matrix representing its analytically integrated trigonometric interpolant.
The terms proportional to `J₀` and `J₁` are the analytic coefficients of the logarithmic singularity. After they are removed, `L₂` and `M₂` possess finite coincidence limits.

Diagonal limits
---------------
The doubled DLP principal-value coincidence limit is inserted analytically as `D_ii=-w_iκ_i`, where `κ_i` is the signed curvature in the convention of the CFIE Kress workspace.
For the SLP, `M₁(ii)=-(1/(2π))s_i` and `M₂(ii)=[i/2-γ_E/π-(1/(2π))log((q²/4)s_i²)]s_i`, giving `S_ii=R_iiM₁(ii)+w_iM₂(ii)`. No Hankel function is therefore evaluated at `r=0`.

DLP normal convention
---------------------
With `dlp_kernel=:source`, the normal is the physical source normal `n_y` and the operator is the usual double-layer operator appearing in the Wiersig formulation.
The alternative target-normal form is generated by `_wiersig_direct_dlp_inner`. Discretely it obeys the weighted-transpose relation `D'_{ij}=D_{ji}(ds_j/ds_i)` and requires no separate singularity analysis.

Pairwise evaluation
-------------------
For every unordered pair `i<j`, the code evaluates `r`, `J₀(qr)`, `J₁(qr)`, `H₀⁽¹⁾(qr)` and `H₁⁽¹⁾(qr)` only once. These values are then reused for the ordered interactions `i←j` and `j←i`. Only the normal numerator, source speed, source weight and Kress-matrix entry change under reversal.

Arguments
---------
- `S,D`: full matrices into which only the block Γ_a×Γ_a is written.
- `pts[a]`: physical boundary quadrature data for Γ_a.
- `geom`: Kress geometry workspace containing local distances, inverse distances, logarithmic terms, speeds, curvature and DLP geometric numerators.
- `a`: physical cavity index.
- `q`: complex material wavenumber of the homogeneous region associated with the block.
- `dlp_kernel`: source-normal or target-normal DLP convention.
- `multithreaded`: enables threaded pair assembly.

Returns
-------
Returns `(S,D)` after modifying the Γ_a×Γ_a block in place.
"""
function _wiersig_assemble_same_component!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},geom::CFIEKressWorkspace{T},a::Int,q::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    iszero(q)&&throw(ArgumentError("Wiersig operator is undefined at q=0"))
    # Resolve Γ_a once. Local geometry indices i,j∈1:Na are embedded in the
    # concatenated physical matrices through the global range ra.
    mode=_wiersig_dlp_normal_mode(dlp_kernel);p=pts[a];G=geom.Gs[a];P=geom.parr[a];Na=length(P.X);ra=geom.offs[a]:(geom.offs[a+1]-1)
    # These are the four constant prefactors occurring in the Kress split and
    # are independent of the source-target indices.
    αL1=q*INV_TWO_PI;αL2=-Complex{T}(0,one(T)/2)*q;αM1=-INV_TWO_PI;αM2=Complex{T}(0,one(T)/2)
    # Insert the analytical coincidence limits directly. This avoids evaluating
    # singular Hankel functions at zero and fixes the principal-value diagonal.
    @inbounds for i in 1:Na
        gi=ra[i];si=G.speed[i];wi=p.ws[i];D[gi,gi]=Complex{T}(-wi*G.kappa[i],zero(T));m1=αM1*si
        m2=(Complex{T}(0,one(T)/2)-EULER_OVER_PI-INV_PI*log(q*si/2))*si
        S[gi,gi]=Complex{T}(geom.Rmat[gi,gi]*m1,zero(T))+wi*m2
    end
    # Evaluate each unordered pair once. The special functions depend only on
    # r and can be reused for the two ordered source-target interactions.
    @use_threads multithreading=(multithreaded&&Na>=32) for j in 2:Na
        gj=ra[j];sj=G.speed[j];wj=p.ws[j]
        @inbounds for i in 1:j-1
            gi=ra[i];si=G.speed[i];wi=p.ws[i];r=G.R[i,j];invr=G.invR[i,j];lt=G.logterm[i,j]
            innSij=G.inner[i,j];innSji=G.inner[j,i];innij=_wiersig_direct_dlp_inner(mode,innSij,innSji,si,sj);innji=_wiersig_direct_dlp_inner(mode,innSji,innSij,sj,si)
            h0,h1=hankel_pair01(q*r);j0,j1=bessel_pair01(q*r)
            # Ordered interaction target i, source j.
            l1=αL1*innij*j1*invr;l2=αL2*innij*h1*invr-l1*lt;D[gi,gj]=geom.Rmat[gi,gj]*l1+wj*l2
            m1=αM1*j0*sj;m2=αM2*h0*sj-m1*lt;S[gi,gj]=geom.Rmat[gi,gj]*m1+wj*m2
            # Ordered interaction target j, source i.
            l1=αL1*innji*j1*invr;l2=αL2*innji*h1*invr-l1*lt;D[gj,gi]=geom.Rmat[gj,gi]*l1+wi*l2
            m1=αM1*j0*si;m2=αM2*h0*si-m1*lt;S[gj,gi]=geom.Rmat[gj,gi]*m1+wi*m2
        end
    end
    return S,D
end

"""
    _wiersig_assemble_same_component!(S,D,pts,geom::WiersigMultiGeometry,a,q;dlp_kernel=:source,multithreaded=true)

Assemble the singular self-interaction blocks `S_aa(q)` and `D_aa(q)` for cavity Γ_a when the complete geometry is stored in `WiersigMultiGeometry`.
The mathematics is identical to the `CFIEKressWorkspace` overload. The doubled kernels are `S_q(x,y)=(i/2)H₀⁽¹⁾(q|x-y|)` and `D_q(x,y)=-(iq/2)H₁⁽¹⁾(q|x-y|)((x-y)⋅n_y)/|x-y|`.
For `i≠j`, the Kress formulas are `D_ij=R_ijL₁+w_jL₂` with `L₁=q/(2π)J₁(qr)inner/r` and `L₂=-(iq/2)H₁⁽¹⁾(qr)inner/r-L₁logterm`, and `S_ij=R_ijM₁+w_jM₂` with `M₁=-(1/(2π))J₀(qr)s_j` and `M₂=(i/2)H₀⁽¹⁾(qr)s_j-M₁logterm`.
The diagonal limits are `D_ii=-w_iκ_i` and `S_ii=R_ii[-s_i/(2π)]+w_i[i/2-γ_E/π-(1/(2π))log((q²/4)s_i²)]s_i`.
The only implementation difference is storage of the Kress matrix. A `WiersigMultiGeometry` contains one local Kress workspace for each physical cavity, so `R_ij` is obtained through `_wiersig_local_rmat(geom,a,i,j)`.
The logarithmic singularity exists only on Γ_a×Γ_a. If `a≠b`, Γ_a×Γ_b is a smooth interaction and no Kress correction is required.
Returns `(S,D)` after modifying only the physical block Γ_a×Γ_a.
"""
function _wiersig_assemble_same_component!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},geom::WiersigMultiGeometry,a::Int,q::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    iszero(q)&&throw(ArgumentError("Wiersig operator is undefined at q=0"))
    mode=_wiersig_dlp_normal_mode(dlp_kernel);p=pts[a];G=geom.Gs[a];P=geom.parr[a];Na=length(P.X);ra=geom.offs[a]:(geom.offs[a+1]-1)
    αL1=q*INV_TWO_PI;αL2=-Complex{T}(0,one(T)/2)*q;αM1=-INV_TWO_PI;αM2=Complex{T}(0,one(T)/2)
    # The diagonal is evaluated from the analytical Kress coincidence limits.
    @inbounds for i in 1:Na
        gi=ra[i];si=G.speed[i];wi=p.ws[i];D[gi,gi]=Complex{T}(-wi*G.kappa[i],zero(T));m1=αM1*si
        m2=(Complex{T}(0,one(T)/2)-EULER_OVER_PI-INV_PI*log(q*si/2))*si
        S[gi,gi]=Complex{T}(_wiersig_local_rmat(geom,a,i,i)*m1,zero(T))+wi*m2
    end
    # The local R matrix belongs exclusively to Γ_a, while the geometrical
    # distance and inner-product caches are likewise local to this cavity.
    @use_threads multithreading=(multithreaded&&Na>=32) for j in 2:Na
        gj=ra[j];sj=G.speed[j];wj=p.ws[j]
        @inbounds for i in 1:j-1
            gi=ra[i];si=G.speed[i];wi=p.ws[i];r=G.R[i,j];invr=G.invR[i,j];lt=G.logterm[i,j]
            innSij=G.inner[i,j];innSji=G.inner[j,i];innij=_wiersig_direct_dlp_inner(mode,innSij,innSji,si,sj);innji=_wiersig_direct_dlp_inner(mode,innSji,innSij,sj,si)
            h0,h1=hankel_pair01(q*r);j0,j1=bessel_pair01(q*r);Rij=_wiersig_local_rmat(geom,a,i,j);Rji=_wiersig_local_rmat(geom,a,j,i)
            # Ordered interaction target i, source j.
            l1=αL1*innij*j1*invr;l2=αL2*innij*h1*invr-l1*lt;D[gi,gj]=Rij*l1+wj*l2
            m1=αM1*j0*sj;m2=αM2*h0*sj-m1*lt;S[gi,gj]=Rij*m1+wj*m2
            # Ordered interaction target j, source i.
            l1=αL1*innji*j1*invr;l2=αL2*innji*h1*invr-l1*lt;D[gj,gi]=Rji*l1+wi*l2
            m1=αM1*j0*si;m2=αM2*h0*si-m1*lt;S[gj,gi]=Rji*m1+wi*m2
        end
    end
    return S,D
end

"""
    _wiersig_sd_entry(pts,geom::CFIEKressWorkspace,gi,gj,q,mode)

Evaluate one full-space Nyström entry `(S_ij(q),D_ij(q))` for arbitrary global target index `gi` and source index `gj`.
This scalar form is required by symmetry-reduced assembly. Instead of constructing the complete dense physical matrices and subsequently projecting them, the reduced operator can request exactly the full-space entries appearing in each symmetry orbit sum. The global indices are first resolved as `gi↔(ib,i)` and `gj↔(jb,j)`, where `ib,jb` are physical boundary-component indices and `i,j` are corresponding local indices.

Same physical component
-----------------------
If `ib==jb`, the kernel may be singular and the Kress formulas are used.
For `i==j`, the analytical diagonal values are `D_ii=-w_iκ_i` and `S_ii=R_ii[-s_i/(2π)]+w_i[i/2-γ_E/π-(1/(2π))log((q²/4)s_i²)]s_i`.
For `i≠j`, `D_ij=R_ijL₁+w_jL₂` with `L₁=q/(2π)J₁(qr)inner/r` and `L₂=-(iq/2)H₁⁽¹⁾(qr)inner/r-L₁logterm`, while `S_ij=R_ijM₁+w_jM₂` with `M₁=-(1/(2π))J₀(qr)s_j` and `M₂=(i/2)H₀⁽¹⁾(qr)s_j-M₁logterm`.

Distinct physical components
----------------------------
If `ib≠jb`, the two boundaries are geometrically separated and the interaction is smooth. The unsplit entries are therefore `S_ij=w_j(i/2)H₀⁽¹⁾(qr)s_j` and `D_ij=w_j[-(iq/2)H₁⁽¹⁾(qr)]inner/r`.
No Kress matrix appears in a cross-component entry.
The function explicitly rejects a vanishing separation between nominally distinct components because such a geometry would invalidate the smooth-interaction assumption.

Returns
-------
Returns `(sval,dval)` without modifying a matrix.
"""
@inline function _wiersig_sd_entry(pts::Vector{BoundaryPointsCFIE{T}},geom::CFIEKressWorkspace{T},gi::Int,gj::Int,q::Complex{T},mode::Val) where {T<:Real}
    # Resolve concatenated full-space indices into physical component and local
    # node indices before choosing singular or smooth quadrature.
    ib=geom.global_to_block[gi];jb=geom.global_to_block[gj];i=geom.global_to_local[gi];j=geom.global_to_local[gj]
    αL1=q*INV_TWO_PI;αL2=-Complex{T}(0,one(T)/2)*q;αM1=-INV_TWO_PI;αM2=Complex{T}(0,one(T)/2)
    if ib==jb
        p=pts[ib];G=geom.Gs[ib];si=G.speed[i];sj=G.speed[j];wi=p.ws[i];wj=p.ws[j]
        if i==j
            dval=Complex{T}(-wi*G.kappa[i],zero(T));m1=αM1*si
            m2=(Complex{T}(0,one(T)/2)-EULER_OVER_PI-INV_PI*log(q*si/2))*si
            return Complex{T}(geom.Rmat[gi,gj]*m1,zero(T))+wi*m2,dval
        end
        r=G.R[i,j];invr=G.invR[i,j];lt=G.logterm[i,j];inn=_wiersig_direct_dlp_inner(mode,G.inner[i,j],G.inner[j,i],si,sj);h0,h1=hankel_pair01(q*r);j0,j1=bessel_pair01(q*r)
        l1=αL1*inn*j1*invr;l2=αL2*inn*h1*invr-l1*lt;dval=geom.Rmat[gi,gj]*l1+wj*l2
        m1=αM1*j0*sj;m2=αM2*h0*sj-m1*lt;sval=geom.Rmat[gi,gj]*m1+wj*m2
        return sval,dval
    end
    # Between distinct components r>0, so the original Hankel kernels are
    # smooth and ordinary source quadrature is sufficient.
    Pi=geom.parr[ib];Pj=geom.parr[jb];pj=pts[jb];dx=Pi.X[i]-Pj.X[j];dy=Pi.Y[i]-Pj.Y[j];r2=muladd(dx,dx,dy*dy)
    r2<=eps(T)^2&&throw(ArgumentError("distinct dielectric boundaries touch or overlap"))
    r=sqrt(r2);invr=inv(r);innSij=Pj.dY[j]*dx-Pj.dX[j]*dy;innSji=Pi.dY[i]*(-dx)-Pi.dX[i]*(-dy)
    inn=_wiersig_direct_dlp_inner(mode,innSij,innSji,Pi.s[i],Pj.s[j]);h0,h1=hankel_pair01(q*r);wj=pj.ws[j]
    return wj*(αM2*h0*Pj.s[j]),wj*(αL2*inn*h1*invr)
end

"""
    _wiersig_sd_entry(pts,geom::WiersigMultiGeometry,gi,gj,q,mode)

Evaluate one full-space Nyström entry `(S_ij(q),D_ij(q))` when the physical discretization is stored in `WiersigMultiGeometry`.
The mathematical cases are identical to the `CFIEKressWorkspace` overload: `ib==jb` uses the singular Kress representation, whereas `ib≠jb` uses the smooth unsplit Hankel kernels.
For a same-cavity off-diagonal entry, `D_ij=R_ij[qJ₁(qr)inner/(2πr)]+w_j[-(iq/2)H₁⁽¹⁾(qr)inner/r-L₁logterm]` and `S_ij=R_ij[-J₀(qr)s_j/(2π)]+w_j[(i/2)H₀⁽¹⁾(qr)s_j-M₁logterm]`.
For a cross-cavity entry, `D_ij=w_j[-(iq/2)H₁⁽¹⁾(qr)]inner/r` and `S_ij=w_j(i/2)H₀⁽¹⁾(qr)s_j`.
The only storage difference is that same-cavity Kress entries are obtained from `_wiersig_local_rmat(geom,ib,i,j)`.
"""
@inline function _wiersig_sd_entry(pts::Vector{BoundaryPointsCFIE{T}},geom::WiersigMultiGeometry,gi::Int,gj::Int,q::Complex{T},mode::Val) where {T<:Real}
    # Resolve global indices into physical cavities and local quadrature nodes.
    ib=geom.global_to_block[gi];jb=geom.global_to_block[gj];i=geom.global_to_local[gi];j=geom.global_to_local[gj]
    αL1=q*INV_TWO_PI;αL2=-Complex{T}(0,one(T)/2)*q;αM1=-INV_TWO_PI;αM2=Complex{T}(0,one(T)/2)
    if ib==jb
        p=pts[ib];G=geom.Gs[ib];si=G.speed[i];sj=G.speed[j];wi=p.ws[i];wj=p.ws[j]
        if i==j
            dval=Complex{T}(-wi*G.kappa[i],zero(T));m1=αM1*si
            m2=(Complex{T}(0,one(T)/2)-EULER_OVER_PI-INV_PI*log(q*si/2))*si
            return Complex{T}(_wiersig_local_rmat(geom,ib,i,j)*m1,zero(T))+wi*m2,dval
        end
        r=G.R[i,j];invr=G.invR[i,j];lt=G.logterm[i,j];inn=_wiersig_direct_dlp_inner(mode,G.inner[i,j],G.inner[j,i],si,sj);h0,h1=hankel_pair01(q*r);j0,j1=bessel_pair01(q*r);Rij=_wiersig_local_rmat(geom,ib,i,j)
        l1=αL1*inn*j1*invr;l2=αL2*inn*h1*invr-l1*lt;dval=Rij*l1+wj*l2
        m1=αM1*j0*sj;m2=αM2*h0*sj-m1*lt;sval=Rij*m1+wj*m2
        return sval,dval
    end
    # Cross-cavity interactions have no singular Kress contribution.
    Pi=geom.parr[ib];Pj=geom.parr[jb];pj=pts[jb];dx=Pi.X[i]-Pj.X[j];dy=Pi.Y[i]-Pj.Y[j];r2=muladd(dx,dx,dy*dy)
    r2<=eps(T)^2&&throw(ArgumentError("distinct dielectric cavities touch or overlap"))
    r=sqrt(r2);invr=inv(r);innSij=Pj.dY[j]*dx-Pj.dX[j]*dy;innSji=Pi.dY[i]*(-dx)-Pi.dX[i]*(-dy)
    inn=_wiersig_direct_dlp_inner(mode,innSij,innSji,Pi.s[i],Pj.s[j]);h0,h1=hankel_pair01(q*r);wj=pj.ws[j]
    return wj*(αM2*h0*Pj.s[j]),wj*(αL2*inn*h1*invr)
end

"""
    _wiersig_sd_entry_interior(pts,geom,gi,gj,q,mode)

Evaluate one SLP/DLP matrix entry for the disconnected dielectric interior.
The physical interior is the disjoint union `Ω_in=Ω₁∪Ω₂∪⋯∪Ω_C`. The Green representation in a particular cavity Ω_a is an integral over its own boundary Γ_a only.
Consequently the interior boundary operators satisfy the exact relation `K_in(Γ_a,Γ_b)=0` for `a≠b`.
This zero is a domain property rather than a numerical approximation. Even though the common exterior couples all cavities, there is no direct interior Green propagation from Γ_b to Ω_a when `a≠b`.
The function therefore returns `(0,0)` whenever target and source belong to different cavities and delegates same-cavity evaluation to `_wiersig_sd_entry`.
"""
@inline function _wiersig_sd_entry_interior(pts::Vector{BoundaryPointsCFIE{T}},geom,gi::Int,gj::Int,q::Complex{T},mode::Val) where {T<:Real}
    # Different physical interiors are disconnected, so their interior operator blocks vanish identically.
    geom.global_to_block[gi]==geom.global_to_block[gj]||return zero(Complex{T}),zero(Complex{T})
    return _wiersig_sd_entry(pts,geom,gi,gj,q,mode)
end

"""
    _wiersig_assemble_sd_full!(S,D,pts,geom,q;dlp_kernel=:source,multithreaded=true)

Assemble the complete unreduced homogeneous layer operators `S(q)` and `D(q)` on all physical cavity boundaries.
This function represents a single connected homogeneous propagation region with one common wavenumber `q`. In the dielectric Wiersig problem its principal role is construction of the exterior operators with `q=q_out=n_out k`.
If there are `C` physical boundaries, the full matrices have block structure `S=[S_ab]_{a,b=1}^C` and `D=[D_ab]_{a,b=1}^C`, where row block `a` is the target boundary Γ_a and column block `b` is the source boundary Γ_b.

Same-cavity blocks
------------------
The blocks Γ_a←Γ_a contain the logarithmic kernel singularity and are assembled by `_wiersig_assemble_same_component!`.
For these blocks, `D_aa=R L₁+wL₂` and `S_aa=R M₁+wM₂`, together with the analytical diagonal limits.

Cross-cavity blocks
-------------------
For `a≠b`, the boundaries are disjoint and every kernel entry is smooth. The entries are `S_ab(i,j)=w_j(i/2)H₀⁽¹⁾(qr)s_j` and `D_ab(i,j)=w_j[-(iq/2)H₁⁽¹⁾(qr)]inner_ij/r_ij`.
No Kress logarithmic correction occurs in an off-diagonal physical block.

Exterior interpretation
-----------------------
For the common exterior Ω₀, every physical source boundary contributes to every target boundary. Thus the exterior matrices are fully coupled even though the dielectric interiors themselves are disconnected.
The resulting matrices have dimension `Ntot×Ntot`, where `Ntot` is the total number of physical boundary nodes before any global symmetry reduction.
"""
function _wiersig_assemble_sd_full!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},geom,q::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    iszero(q)&&throw(ArgumentError("Wiersig operator is undefined at q=0"))
    N=geom.Ntot;size(S)==(N,N)||throw(DimensionMismatch("S has size $(size(S)); expected ($N,$N)"));size(D)==(N,N)||throw(DimensionMismatch("D has size $(size(D)); expected ($N,$N)"))
    fill!(S,zero(Complex{T}));fill!(D,zero(Complex{T}));mode=_wiersig_dlp_normal_mode(dlp_kernel);nc=length(pts)
    # Every diagonal physical block contains the Kress singularity and is built
    # with the same-component product-integration rule.
    for a in 1:nc
        _wiersig_assemble_same_component!(S,D,pts,geom,a,q;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
    end
    # Distinct physical cavities are separated, so their mutual propagators are
    # smooth evaluations of the original doubled Helmholtz kernels.
    αL2=-Complex{T}(0,one(T)/2)*q;αM2=Complex{T}(0,one(T)/2)
    for a in 1:nc,b in 1:nc
        a==b&&continue
        pa=pts[a];pb=pts[b];Pa=geom.parr[a];Pb=geom.parr[b];Na=length(Pa.X);Nb=length(Pb.X);ra=geom.offs[a]:(geom.offs[a+1]-1);rb=geom.offs[b]:(geom.offs[b+1]-1)
        @use_threads multithreading=(multithreaded&&Na>=16) for i in 1:Na
            gi=ra[i];xi=Pa.X[i];yi=Pa.Y[i]
            @inbounds for j in 1:Nb
                gj=rb[j];dx=xi-Pb.X[j];dy=yi-Pb.Y[j];r2=muladd(dx,dx,dy*dy)
                r2<=eps(T)^2&&throw(ArgumentError("distinct dielectric boundaries touch or overlap"))
                r=sqrt(r2);invr=inv(r);innSij=Pb.dY[j]*dx-Pb.dX[j]*dy;innSji=Pa.dY[i]*(-dx)-Pa.dX[i]*(-dy)
                inn=_wiersig_direct_dlp_inner(mode,innSij,innSji,Pa.s[i],Pb.s[j]);h0,h1=hankel_pair01(q*r);wj=pb.ws[j]
                D[gi,gj]=wj*(αL2*inn*h1*invr);S[gi,gj]=wj*(αM2*h0*Pb.s[j])
            end
        end
    end
    return S,D
end

"""
    _wiersig_assemble_sd_reduced!(S,D,pts,geom,q;dlp_kernel=:source,multithreaded=true)

Assemble a symmetry-reduced homogeneous layer operator directly without first constructing the full `Ntot×Ntot` physical matrix. Let reduced source basis vector `b` correspond to the exact discrete symmetry orbit `e_b^red=Σ_rρ_r e_{g_{b,r}}`, where `g_{b,r}` are physical source indices and `ρ_r` are the character factors belonging to the chosen irreducible symmetry sector. For reduced target row `a`, the physical representative is `i_a=Ifund[a]`. The projected matrix entry is therefore `K_red[a,b]=Σ_rρ_rK_full(i_a,g_{b,r})`. This is the exact discrete symmetry projection associated with the orbit representation.

In one connected homogeneous region every source image in the orbit contributes. Thus the sum is over the complete orbit. For the Wiersig problem this is principally the exterior Ω₀, where fields emitted by every physical cavity propagate through the same connected exterior. The scalar function `_wiersig_sd_entry` automatically decides whether each term is a singular same-cavity Kress interaction or a smooth cross-cavity Hankel interaction.
The matrices `S` and `D` have dimension `Nred×Nred`.
"""
function _wiersig_assemble_sd_reduced!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},geom,q::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    mode=_wiersig_dlp_normal_mode(dlp_kernel);N=geom.Nred;fill!(S,zero(Complex{T}));fill!(D,zero(Complex{T}))
    # Each reduced column represents one exact symmetry orbit of full physical
    # source nodes; each reduced row uses one full physical representative.
    @use_threads multithreading=multithreaded for b in 1:N
        orb=geom.reduced_orbits[b];imgs=orb.full;scales=orb.scales
        @inbounds for a in 1:N
            gi=geom.Ifund[a];sacc=zero(Complex{T});dacc=zero(Complex{T})
            # A connected homogeneous region receives contributions from every
            # physical image in the source orbit, weighted by its character.
            for r in eachindex(imgs)
                sval,dval=_wiersig_sd_entry(pts,geom,gi,imgs[r],q,mode);ρ=Complex{T}(scales[r]);sacc+=ρ*sval;dacc+=ρ*dval
            end
            S[a,b]=sacc;D[a,b]=dacc
        end
    end
    return S,D
end

"""
    _wiersig_assemble_interior_full!(S,D,solver,pts,geom,k;dlp_kernel=:source,multithreaded=true)

Assemble the unreduced interior dielectric SLP and DLP operators.
The dielectric interior consists of disconnected physical domains `Ω₁,Ω₂,...,Ω_C`, with boundary `∂Ω_a=Γ_a` and material wavenumber `q_a=n_a k`.
For cavity `a`, the Wiersig interior boundary equation is `χ(n_a)S_aa(n_a k)φ_a+[D_aa(n_a k)-I_a]ψ_a=0`.
Consequently the complete interior operators are exactly block diagonal: `S_in=diag(χ₁S₁₁(n₁k),...,χ_CS_CC(n_Ck))` and `D_in=diag(D₁₁(n₁k),...,D_CC(n_Ck))`.
There are no off-diagonal interior propagators: `S_ab=D_ab=0` for `a≠b`. This is an exact consequence of the disconnected interior Green representations.

Polarization factor
-------------------
The Wiersig SLP multiplier is `χ(n)=1` for TM polarization and `χ(n)=n²` for TE polarization.
The factor `χ(n_a)` multiplies only the SLP block. The DLP block is independent of this polarization scaling.

Assembly strategy
-----------------
`S` and `D` are first filled with zeros, which establishes the exact off-diagonal block structure. Each Γ_a×Γ_a block is then assembled at its own complex wavenumber `q_a=n_a k`, after which its SLP block is multiplied by `χ(n_a)`.
The identity jumps `-I` are not inserted here. They belong to construction of the final Wiersig boundary matrix.
"""
function _wiersig_assemble_interior_full!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},geom,k::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    # Zero initialization realizes the exact block-diagonal structure of the
    # disconnected dielectric interiors before any self-block is written.
    fill!(S,zero(Complex{T}));fill!(D,zero(Complex{T}));nin=_wiersig_component_indices(solver,length(pts))
    for a in eachindex(pts)
        # Interior Ω_a propagates with its own material wavenumber q_a=n_a k.
        q=Complex{T}(nin[a])*k
        _wiersig_assemble_same_component!(S,D,pts,geom,a,q;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        # χ(n_a) belongs only to the SLP trace in the Wiersig transmission
        # convention; the DLP block is left unchanged.
        χ=Complex{T}(_wiersig_slp_factor(solver,nin[a]));ra=geom.offs[a]:(geom.offs[a+1]-1);@views rmul!(S[ra,ra],χ)
    end
    return S,D
end

"""
    _wiersig_assemble_interior_reduced!(S,D,solver,pts,geom,k;dlp_kernel=:source,multithreaded=true)

Assemble the symmetry-reduced interior dielectric operator directly.
This projection differs fundamentally from the homogeneous exterior projection because a reduced symmetry orbit may contain physical source nodes from several different cavities, while an interior Green representation for Ω_c contains sources only on Γ_c. Let reduced target row `a` use representative `gi=Ifund[a]`, and let `c=global_to_block[gi]` be its physical cavity. The target equation therefore uses `q_c=n_c k` and `χ_c=χ(n_c)`.
If reduced source column `b` is `e_b^red=Σ_rρ_r e_{g_{b,r}}`, the interior projected entry is `K_in,red[a,b]=Σ_{r:g_{b,r}∈Γ_c}ρ_rK_cc(gi,g_{b,r})`. All orbit images belonging to other physical cavities are discarded exactly. This implements the identity `∂Ω_c=Γ_c` and prevents symmetry reduction from introducing artificial Green-function coupling between disconnected interiors. For the SLP, the resulting orbit sum is multiplied by `χ_c`; for the DLP no polarization factor is applied. For a rotation that permutes complete identical cavities, a reduced orbit may contain one corresponding node on every cavity. For an interior target on Γ_c, only the image belonging to Γ_c survives this filter. In contrast, the exterior homogeneous operator retains all images.
"""
function _wiersig_assemble_interior_reduced!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},geom,k::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    mode=_wiersig_dlp_normal_mode(dlp_kernel);N=geom.Nred;fill!(S,zero(Complex{T}));fill!(D,zero(Complex{T}));nin=_wiersig_component_indices(solver,length(pts))
    # A reduced column can span multiple physical cavities, but every reduced
    # target equation still belongs to one definite physical interior Ω_c.
    @use_threads multithreading=multithreaded for b in 1:N
        orb=geom.reduced_orbits[b];imgs=orb.full;scales=orb.scales
        @inbounds for a in 1:N
            gi=geom.Ifund[a];ca=geom.global_to_block[gi]
            # The target representative determines both the local material
            # wavenumber and the polarization-dependent SLP multiplier.
            q=Complex{T}(nin[ca])*k;χ=Complex{T}(_wiersig_slp_factor(solver,nin[ca]));sacc=zero(Complex{T});dacc=zero(Complex{T})
            for r in eachindex(imgs)
                gj=imgs[r]
                # Exact domain filter: an interior field in Ω_ca is represented
                # only by boundary sources on ∂Ω_ca=Γ_ca.
                geom.global_to_block[gj]==ca||continue
                sval,dval=_wiersig_sd_entry_interior(pts,geom,gi,gj,q,mode);ρ=Complex{T}(scales[r]);sacc+=ρ*sval;dacc+=ρ*dval
            end
            S[a,b]=χ*sacc;D[a,b]=dacc
        end
    end
    return S,D
end

"""
    _wiersig_assemble_interior!(S,D,solver,pts,geom,k;dlp_kernel=:source,multithreaded=true)

Dispatch to the appropriate interior dielectric assembly according to whether global symmetry is active.
If `geom.symmetry===nothing`, the explicit full-space block-diagonal operators are assembled by `_wiersig_assemble_interior_full!`. If symmetry is active, `_wiersig_assemble_interior_reduced!` constructs the projected operator directly from exact physical symmetry orbits while enforcing the disconnected-interior source filter. In both cases the physical cavity wavenumbers are `q_a=n_a k`, and the SLP blocks contain the appropriate polarization factors `χ(n_a)`.
"""
function _wiersig_assemble_interior!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},geom,k::Number;dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    kc=Complex{T}(k)
    return isnothing(geom.symmetry) ?
        _wiersig_assemble_interior_full!(S,D,solver,pts,geom,kc;dlp_kernel=dlp_kernel,multithreaded=multithreaded) :
        _wiersig_assemble_interior_reduced!(S,D,solver,pts,geom,kc;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
end

"""
    _wiersig_assemble_sd!(S,D,pts,geom,q;dlp_kernel=:source,multithreaded=true)

Dispatch to the full or symmetry-reduced homogeneous layer-operator assembly.
This function represents one connected homogeneous region with a single common wavenumber `q`. In the Wiersig transmission problem it is principally used for the common exterior with `q=q_out=n_out k`.
If `geom.symmetry===nothing`, `_wiersig_assemble_sd_full!` constructs the complete physical `Ntot×Ntot` operators, including singular Kress self-blocks and smooth cross-cavity blocks.
If symmetry is active, `_wiersig_assemble_sd_reduced!` evaluates the projected `Nred×Nred` operators directly using the orbit formula `K_red[a,b]=Σ_rρ_rK_full(Ifund[a],g_{b,r})`.
Unlike the disconnected interior assembly, every source image contributes in a connected homogeneous region.
"""
function _wiersig_assemble_sd!(S::AbstractMatrix{Complex{T}},D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},geom,q::Number;dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    qc=Complex{T}(q)
    return isnothing(geom.symmetry) ?
        _wiersig_assemble_sd_full!(S,D,pts,geom,qc;dlp_kernel=dlp_kernel,multithreaded=multithreaded) :
        _wiersig_assemble_sd_reduced!(S,D,pts,geom,qc;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
end

"""
    construct_matrices!(solver,A,pts,ws,k;dlp_kernel=:source,multithreaded=true)

Assemble the nonlinear Wiersig resonance matrix for the trace vector `x=[φ;ψ]`:
`A(k)=[Sχ,int(k)  D_int(k)-I; χ_outS_ext(n_out k)  D_ext(n_out k)+I]`.
The interior operators are heterogeneous and block diagonal, with cavity wavenumbers `q_a=n_a k` and SLP factors `χ(n_a)`. The exterior is one connected homogeneous region with `q_out=n_out k`, so all physical cavities are mutually coupled. The doubled DLP jump relations produce the diagonal shifts `D_int-I` and `D_ext+I`.
Returns `A`, modified in place.
"""
function construct_matrices!(solver::WiersigKress,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::WiersigGeometryWorkspace,k::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    # Validate material cardinality and symmetry covariance before evaluating the
    # expensive special-function kernels.
    _wiersig_component_indices(solver,length(pts));_wiersig_validate_material_symmetry(solver,ws)
    # The vacuum spectral variable k : q_out=n_out k. Interior q_a=n_a k values are selected separately by the heterogeneous interior assembler.
    kc=Complex{T}(k);qout=Complex{T}(solver.n_out)*kc;N=boundary_size(ws);geom=ws.geom
    χout=Complex{T}(_wiersig_slp_factor(solver,solver.n_out))
    @views begin
        # x=[φ;ψ] gives A11: φ→interior, A12: ψ→interior, A21: φ→exterior and A22: ψ→exterior.
        A11=A[1:N,1:N];A12=A[1:N,N+1:2*N];A21=A[N+1:2*N,1:N];A22=A[N+1:2*N,N+1:2*N]
        # Heterogeneous interior.
        _wiersig_assemble_interior!(A11,A12,solver,pts,geom,kc;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        # Fully coupled homogeneous exterior at q_out.
        _wiersig_assemble_sd!(A21,A22,pts,geom,qout;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
        rmul!(A21,χout)
        # Trace jumps of the doubled DLP: interior D-I, exterior D+I.
        @inbounds for i in 1:N
            A12[i,i]-=one(Complex{T});A22[i,i]+=one(Complex{T})
        end
    end
    return A
end

"""
    solve_vect(solver,pts,ws,k;dlp_kernel=:source,multithreaded=true)

Form `A(k)`, compute its SVD, and return `(σ_min,x_min)`, where `σ_min=min singular_value(A(k))` and `x_min` is the corresponding right singular vector satisfying `A(k)x_min≈0`.
At a resonance, `σ_min→0`, so `x_min=[φ;ψ]` approximates the boundary trace of the resonant state.
"""
function solve_vect(solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},ws::WiersigGeometryWorkspace,k::Complex{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    N=boundary_matrix_size(ws);A=Matrix{Complex{T}}(undef,N,N)
    construct_matrices!(solver,A,pts,ws,k;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
    F=svd(A)
    return F.S[end],F.V[:,end]
end

# k CHEBYSHEV ASSEMBLY
#
# Geometry and operator
# ---------------------
# For disjoint dielectric cavities Ω_a with Γ_a=∂Ω_a, interior indices n_a and
# common exterior index n_out, define q_a=n_a k and q_out=n_out k.
#
# With x=(φ,ψ) and χ(n)=1 for TM, χ(n)=n² for TE,
# A(k)=[Sχ,int(k)  D_int(k)-I; χ_outS_ext(q_out)  D_ext(q_out)+I],
# where Sχ,int=diag_a[χ(n_a)S_aa(n_a k)] and
# D_int=diag_a[D_aa(n_a k)]. Interior cross-cavity blocks vanish exactly,
# whereas the connected exterior contains every ordered pair Γ_b→Γ_a.
#
# q Chebyshev plans
# -----------------------
# For ks=(k₁,...,k_M), the material wavenumbers are stored component-major:
# qall=[n₁k₁,...,n₁k_M,...,n_Ck₁,...,n_Ck_M,n_outk₁,...,n_outk_M].
# Thus ℓ_in(a,m)=(a-1)M+m and ℓ_out(m)=CM+m. Each plan interpolates radial
# Bessel/Hankel functions at one material q; no interpolation in k occurs.
#
# Kress structure
# ---------------
# Same-cavity blocks use the split
# D_ij=R_ij(inner/r)L₁+w_j(inner/r)L₂-w_jlogterm(inner/r)L₁,
# S_ij=R_ijs_jM₁+w_js_jM₂-w_jlogterm s_jM₁,
# with L₁=qJ₁(qr)/(2π), L₂=-(iq/2)H₁⁽¹⁾(qr),
# M₁=-J₀(qr)/(2π), M₂=(i/2)H₀⁽¹⁾(qr).
#
# Distinct cavities are smooth:
# D_ab=w_j(inner/r)[-(iq/2)H₁⁽¹⁾(qr)],
# S_ab=w_js_j[(i/2)H₀⁽¹⁾(qr)].
#
# DLP normal convention
# ---------------------
# :source uses D=2∂_{n_y}G. :target uses D'=2∂_{n_x}G with the discrete
# weighted-transpose relation D'_{ij}=D_{ji}(ds_j/ds_i).
#
# Symmetry
# --------
# The Chebyshev layer never constructs its own symmetry map. `direct_ws`, built
# by `build_cfie_kress_workspace(solver,pts)`, supplies the physical indexing, Ifund, 
# reduced_orbits, global/local maps, Ntot and Nred.
# In a reduced interior row only orbit images on the target cavity contribute;
# in the exterior row every orbit image contributes.

"""
    _wiersig_cheb_dlp_inner(...)

Return the parameter-weighted DLP numerator. Source-normal uses `inner_ij=s_j n_j⋅(x_i-x_j)`; target-normal uses `inner'_{ij}=(s_j/s_i)inner_{ji}`, implementing `D'_{ij}=D_{ji}(ds_j/ds_i)`.
"""
@inline _wiersig_cheb_dlp_inner(::Val{:source},blk,blkrev,i::Int,j::Int,si,sj)=blk.inner[i,j]
@inline _wiersig_cheb_dlp_inner(::Val{:target},blk,blkrev,i::Int,j::Int,si,sj)=(sj/si)*blkrev.inner[j,i]

"""
    _wiersig_all_k_full_chebyshev!(As,solver,pts,cws,normal_mode;multithreaded=true)

Assemble all Wiersig matrices `A(k_m)` in the complete physical boundary basis.
For `C` cavities and `M` stored vacuum wavenumbers `k_m`, define `q_{a,m}=n_a k_m` and `q_{0,m}=n_out k_m`. With `x=(φ,ψ)`, each matrix has the block form `A(k_m)=[Sχ,int(k_m)  D_int(k_m)-I; χ_outS_ext(q_{0,m})  D_ext(q_{0,m})+I]`.
The interior operators are exactly block diagonal: `Sχ,int(k_m)=diag_a[χ(n_a)S_aa(q_{a,m})]` and `D_int(k_m)=diag_a[D_aa(q_{a,m})]`. Hence `S_int,ab=D_int,ab=0` for `a≠b`. The exterior is one connected homogeneous domain, so `S_ext(q_{0,m})=[S_ab(q_{0,m})]` and `D_ext(q_{0,m})=[D_ab(q_{0,m})]` contain every ordered cavity pair.

For same-cavity entries `Γ_a←Γ_a`, Kress splitting is used:
`D_ij=R_ij(inner/r)L₁+w_j(inner/r)L₂-w_jlogterm(inner/r)L₁`,
`S_ij=R_ijs_jM₁+w_js_jM₂-w_jlogterm s_jM₁`,
with `L₁=qJ₁(qr)/(2π)`, `L₂=-(iq/2)H₁⁽¹⁾(qr)`,
`M₁=-J₀(qr)/(2π)` and `M₂=(i/2)H₀⁽¹⁾(qr)`.

For distinct cavities the kernels are smooth:
`D_ab(i,j)=w_j(inner/r)[-(iq/2)H₁⁽¹⁾(qr)]`,
`S_ab(i,j)=w_js_j[(i/2)H₀⁽¹⁾(qr)]`.

The output matrices `As[m]` are modified in place.
"""
function _wiersig_all_k_full_chebyshev!(As::Vector{Matrix{ComplexF64}},solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},cws::WiersigChebyshevWorkspace{T},normal_mode::Val;multithreaded::Bool=true) where {T<:Real}
    M=length(cws.ks);C=cws.ncavities;Mq=(C+1)*M;N=boundary_size(cws);q=cws.qall;nin=_wiersig_component_indices(solver,C)
    @inbounds for m in 1:M
        fill!(As[m],0.0+0.0im)
    end
    # χ_a enters only the interior SLP block on Γ_a, while χ_out multiplies
    # every exterior SLP block because Ω₀ has one common material index.
    χin=ComplexF64.([_wiersig_slp_factor(solver,nin[a]) for a in 1:C]);χout=ComplexF64(_wiersig_slp_factor(solver,solver.n_out))
    # For every material wavenumber q_l, define the radial prefactors
    # L₁=q_lJ₁/(2π), L₂=-(iq_l/2)H₁, M₁=-J₀/(2π), M₂=(i/2)H₀.
    αL1=Vector{ComplexF64}(undef,Mq);αL2=Vector{ComplexF64}(undef,Mq)
    @inbounds for l in 1:Mq
        αL1[l]=q[l]*INV_TWO_PI;αL2[l]=-0.5im*q[l]
    end
    αM1=-INV_TWO_PI;αM2=0.5*im
    plans0=cws.plans0;plans1=cws.plans1;plansj0=cws.plansj0;plansj1=cws.plansj1
    rout=_wiersig_qout_range(M,C)
    p0out=@view plans0[rout];p1out=@view plans1[rout]
    pj0out=@view plansj0[rout];pj1out=@view plansj1[rout]
    h0_tls=cws.bessel_ws.h0_tls;h1_tls=cws.bessel_ws.h1_tls;j0_tls=cws.bessel_ws.j0_tls;j1_tls=cws.bessel_ws.j1_tls
    blocks=cws.block_cache.blocks;nc=size(blocks,1);nc==C||throw(DimensionMismatch("block cache contains $nc physical components but the solver contains $C cavities"))
    # Each diagonal physical block Γ_a×Γ_a appears twice in A(k_m):
    # upper row: interior operator at q_a=n_a k_m,
    # lower row: exterior self-interaction at q_out=n_out k_m.
    # Both share the same geometry/Kress data but use different q plans.
    for a in 1:C
        rin=_wiersig_qin_range(M,a)
        p0in=@view plans0[rin];p1in=@view plans1[rin]
        pj0in=@view plansj0[rin];pj1in=@view plansj1[rin]
        blk=blocks[a,a]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid();h0vals=h0_tls[tid];h1vals=h1_tls[tid];j0vals=j0_tls[tid];j1vals=j1_tls[tid]
            h0in=@view h0vals[rin];h1in=@view h1vals[rin];j0in=@view j0vals[rin];j1in=@view j1vals[rin]
            h0out=@view h0vals[rout];h1out=@view h1vals[rout];j0out=@view j0vals[rout];j1out=@view j1vals[rout]
            ro=blk.row_offset;co=blk.col_offset;gj=co+j-1;sj=blk.speed_j[j];wj=blk.wj[j];χa=χin[a]
            # On i=j, the doubled DLP principal-value limit is D_ii=-w_jκ_j.
            # The SLP diagonal is the regularized Kress finite part
            # R_jj[-s_j/(2π)]+w_j[i/2-γ_E/π-(1/(2π))log((q²/4)s_j²)]s_j.
            gi=ro+j-1;κj=blk.kappa_i[j];rjj=blk.Rkress[j,j];dval=ComplexF64(-wj*κj,0.0);m1diag=αM1*sj
            @inbounds for m in 1:M
                li=(a-1)*M+m;lo=C*M+m
                m2i=((0.5*im-EULER_OVER_PI)-INV_PI*log(q[li]*sj/2))*sj
                m2o=((0.5*im-EULER_OVER_PI)-INV_PI*log(q[lo]*sj/2))*sj
                sini=ComplexF64(rjj*m1diag,0.0)+wj*m2i;sout=ComplexF64(rjj*m1diag,0.0)+wj*m2o;A=As[m]
                # Upper row: χ_aS_aa and D_aa-I.
                A[gi,gi]=χa*sini;A[gi,N+gi]=dval-one(ComplexF64)
                # Lower row: χ_outS_aa^ext and D_aa^ext+I.
                A[N+gi,gi]=χout*sout;A[N+gi,N+gi]=dval+one(ComplexF64)
            end
            # For i≠j, evaluate only the two material families needed by this
            # self-block: q_a for the interior and q_out for the exterior.
            # The same values are reused for both ordered entries i←j and j←i.
            @inbounds for i in j+1:blk.Ni
                gi=ro+i-1;r=blk.R[i,j];invr=blk.invR[i,j];lt=blk.logterm[i,j];Rij=blk.Rkress[i,j];Rji=blk.Rkress[j,i];si=blk.speed_i[i];wi=blk.wi[i]
                # innij is the parameter-weighted normal numerator for target i,
                # source j. innji is the reversed ordered interaction.
                innij=_wiersig_cheb_dlp_inner(normal_mode,blk,blk,i,j,si,sj);innji=_wiersig_cheb_dlp_inner(normal_mode,blk,blk,j,i,sj,si)
                h0_h1_j0_j1_multi_ks_at_r!(h0in,h1in,j0in,j1in,p0in,p1in,pj0in,pj1in,blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(r))
                h0_h1_j0_j1_multi_ks_at_r!(h0out,h1out,j0out,j1out,p0out,p1out,pj0out,pj1out,blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(r))
                # Geometry factors are separated from q-dependent radial factors:
                # D_ij=(Rij*inner/r)L₁+(wj*inner/r)L₂-(wj*lt*inner/r)L₁,
                # S_ij=(Rij*sj)M₁+(wj*sj)M₂-(wj*sj*lt)M₁.
                cD1ij=Rij*innij*invr;cD2ij=wj*innij*invr;cD3ij=wj*lt*innij*invr
                cD1ji=Rji*innji*invr;cD2ji=wi*innji*invr;cD3ji=wi*lt*innji*invr
                cS1j=Rij*sj;cS2j=wj*sj;cS3j=wj*sj*lt;cS1i=Rji*si;cS2i=wi*si;cS3i=wi*si*lt
                for m in 1:M
                    li=(a-1)*M+m;lo=C*M+m
                    # Interior block on Ω_a at q_a=n_a k_m.
                    L1i=αL1[li]*j1vals[li];L2i=αL2[li]*h1vals[li];M1i=αM1*j0vals[li];M2i=αM2*h0vals[li]
                    dij=cD1ij*L1i+cD2ij*L2i-cD3ij*L1i;dji=cD1ji*L1i+cD2ji*L2i-cD3ji*L1i
                    sij=cS1j*M1i+cS2j*M2i-cS3j*M1i;sji=cS1i*M1i+cS2i*M2i-cS3i*M1i
                    # Exterior self-block on the same geometry at q_out=n_out k_m.
                    L1o=αL1[lo]*j1vals[lo];L2o=αL2[lo]*h1vals[lo];M1o=αM1*j0vals[lo];M2o=αM2*h0vals[lo]
                    doij=cD1ij*L1o+cD2ij*L2o-cD3ij*L1o;doji=cD1ji*L1o+cD2ji*L2o-cD3ji*L1o
                    soij=cS1j*M1o+cS2j*M2o-cS3j*M1o;soji=cS1i*M1o+cS2i*M2o-cS3i*M1o;A=As[m]
                    # Populate both ordered same-cavity entries in the interior row.
                    A[gi,gj]=χa*sij;A[gj,gi]=χa*sji;A[gi,N+gj]=dij;A[gj,N+gi]=dji
                    # Populate both ordered self-interactions in the exterior row.
                    A[N+gi,gj]=χout*soij;A[N+gj,gi]=χout*soji;A[N+gi,N+gj]=doij;A[N+gj,N+gi]=doji
                end
            end
        end
    end
    # For a≠b, Γ_a×Γ_b is smooth and contributes only to the common exterior.
    # Visit each unordered cavity pair once. Since r_ab(i,j)=r_ba(j,i), one
    # H₀/H₁ evaluation supplies both ordered interactions Γ_a←Γ_b and Γ_b←Γ_a.
    for a in 1:C-1,b in a+1:C
        blk=blocks[a,b];blkrev=blocks[b,a]
        @use_threads multithreading=multithreaded for j in 1:blk.Nj
            tid=Threads.threadid();h0vals=h0_tls[tid];h1vals=h1_tls[tid]
            h0out=@view h0vals[rout];h1out=@view h1vals[rout]
            roa=blk.row_offset;rob=blk.col_offset;gb=rob+j-1;sj=blk.speed_j[j];wj=blk.wj[j]
            @inbounds for i in 1:blk.Ni
                ga=roa+i-1;r=blk.R[i,j];invr=blk.invR[i,j];si=blk.speed_i[i];wi=blk.wi[i]
                # The radial special functions are identical in the two directions.
                # Only source normal, source speed and source quadrature weight change.
                innab=_wiersig_cheb_dlp_inner(normal_mode,blk,blkrev,i,j,si,sj)
                innba=_wiersig_cheb_dlp_inner(normal_mode,blkrev,blk,j,i,sj,si)
                h0_h1_multi_ks_at_r!(h0out,h1out,p0out,p1out,blk.pidx[i,j],blk.tloc[i,j],Float64(r))
                # Smooth exterior kernels for Γ_a←Γ_b and Γ_b←Γ_a.
                cDab=wj*innab*invr;cSab=wj*sj
                cDba=wi*innba*invr;cSba=wi*si
                for m in 1:M
                    lo=C*M+m;A=As[m]
                    A[N+ga,gb]=χout*cSab*αM2*h0out[m]
                    A[N+ga,N+gb]=cDab*αL2[lo]*h1out[m]
                    A[N+gb,ga]=χout*cSba*αM2*h0out[m]
                    A[N+gb,N+ga]=cDba*αL2[lo]*h1out[m]
                end
            end
        end
    end
    return nothing
end

"""
    _wiersig_all_k_reduced_chebyshev!(As,solver,pts,cws,normal_mode;multithreaded=true)

Assemble all `A(k_m)` directly in the exact symmetry-adapted boundary basis.
Let reduced source basis vector `b` be the orbit combination
`e_b^red=Σ_rρ_r e_{g_{b,r}}`, with physical indices `g_{b,r}` and character factors `ρ_r`. For reduced target representative `g_a=Ifund[a]`, any homogeneous operator satisfies `K_red[a,b]=Σ_rρ_rK_full(g_a,g_{b,r})`.
For the connected exterior this complete orbit sum is used for both `S_ext` and `D_ext`.
For the disconnected dielectric interior, if `g_a∈Γ_c`, only images on the same physical boundary contribute:
`K_int,red[a,b]=Σ_{r:g_{b,r}∈Γ_c}ρ_rK_cc(g_a,g_{b,r})`.
The target cavity `c` fixes `q_c=n_c k_m` and `χ_c`.

Same-cavity terms use the Kress formulas
`D_ij=R_ij(inner/r)L₁+w_j(inner/r)L₂-w_jlogterm(inner/r)L₁` and
`S_ij=R_ijs_jM₁+w_js_jM₂-w_jlogterm s_jM₁`,
while cross-cavity exterior terms use the smooth kernels
`D_ab=w_j(inner/r)L₂` and `S_ab=w_js_jM₂`.
After orbit summation, the doubled-DLP jumps are added once in the reduced basis:
`D_int→D_int-I`, `D_ext→D_ext+I`.
"""
function _wiersig_all_k_reduced_chebyshev!(As::Vector{Matrix{ComplexF64}},solver::WiersigKress,pts::Vector{BoundaryPointsCFIE{T}},cws::WiersigChebyshevWorkspace{T},normal_mode::Val;multithreaded::Bool=true) where {T<:Real}
    M=length(cws.ks);C=cws.ncavities;Mq=(C+1)*M;N=boundary_size(cws);q=cws.qall;nin=_wiersig_component_indices(solver,C)
    @inbounds for m in 1:M
        fill!(As[m],0.0+0.0im)
    end
    # Material coefficients are attached to the target physical cavity in the
    # interior and to the common exterior material in the lower row.
    χin=ComplexF64.([_wiersig_slp_factor(solver,nin[a]) for a in 1:C]);χout=ComplexF64(_wiersig_slp_factor(solver,solver.n_out))
    αL1=Vector{ComplexF64}(undef,Mq);αL2=Vector{ComplexF64}(undef,Mq)
    @inbounds for l in 1:Mq
        αL1[l]=q[l]*INV_TWO_PI;αL2[l]=-0.5im*q[l]
    end
    αM1=-INV_TWO_PI;αM2=0.5*im
    # The complete plan table is component-major. Precompute views for each
    # interior material family and one common view for the exterior family.
    plans0=cws.plans0;plans1=cws.plans1;plansj0=cws.plansj0;plansj1=cws.plansj1
    rin=[_wiersig_qin_range(M,a) for a in 1:C]
    p0in=[@view plans0[rin[a]] for a in 1:C];p1in=[@view plans1[rin[a]] for a in 1:C]
    pj0in=[@view plansj0[rin[a]] for a in 1:C];pj1in=[@view plansj1[rin[a]] for a in 1:C]
    rout=_wiersig_qout_range(M,C)
    p0out=@view plans0[rout];p1out=@view plans1[rout]
    pj0out=@view plansj0[rout];pj1out=@view plansj1[rout]
    h0_tls=cws.bessel_ws.h0_tls;h1_tls=cws.bessel_ws.h1_tls;j0_tls=cws.bessel_ws.j0_tls;j1_tls=cws.bessel_ws.j1_tls
    blocks=cws.block_cache.blocks;geom=cws.direct_ws.geom
    # The direct workspace supplies the exact discrete group action:
    # Ifund[a]          representative full target index,
    # reduced_orbits[b] physical orbit of reduced column b,
    # global_to_block   physical cavity of a full index,
    # global_to_local   local boundary-node index.
    Ifund=geom.Ifund;g2b=geom.global_to_block;g2l=geom.global_to_local;orbits=geom.reduced_orbits
    ntls=length(h0_tls);acc11_tls=[Vector{ComplexF64}(undef,M) for _ in 1:ntls];acc12_tls=[Vector{ComplexF64}(undef,M) for _ in 1:ntls]
    acc21_tls=[Vector{ComplexF64}(undef,M) for _ in 1:ntls];acc22_tls=[Vector{ComplexF64}(undef,M) for _ in 1:ntls]
    # Parallelization is over reduced columns. Each thread accumulates one
    # projected column and therefore never races on matrix entries.
    @use_threads multithreading=multithreaded for b in 1:N
        tid=Threads.threadid();h0vals=h0_tls[tid];h1vals=h1_tls[tid];j0vals=j0_tls[tid];j1vals=j1_tls[tid]
        h0out=@view h0vals[rout];h1out=@view h1vals[rout];j0out=@view j0vals[rout];j1out=@view j1vals[rout]
        acc11=acc11_tls[tid];acc12=acc12_tls[tid];acc21=acc21_tls[tid];acc22=acc22_tls[tid]
        orb=orbits[b];imgs=orb.full;scales=orb.scales
        @inbounds for a in 1:N
            fill!(acc11,0.0+0.0im);fill!(acc12,0.0+0.0im);fill!(acc21,0.0+0.0im);fill!(acc22,0.0+0.0im)
            # Let g_a=Ifund[a] lie on Γ_ca. Then the interior row uses exactly
            # q_ca=n_ca k_m and χ_ca, independently of which orbit image supplies the reduced source column.
            ig=Ifund[a];ib=g2b[ig];i=g2l[ig];ca=ib;χa=χin[ca]
            rc=rin[ca]
            h0in=@view h0vals[rc];h1in=@view h1vals[rc];j0in=@view j0vals[rc];j1in=@view j1vals[rc]
            # The reduced matrix element is accumulated as the exact character sum Σ_rρ_rK_full(g_a,g_{b,r}).
            for rorb in eachindex(imgs)
                jg=imgs[rorb];scale=ComplexF64(scales[rorb]);jb=g2b[jg];j=g2l[jg];blk=blocks[ib,jb];blkrev=blocks[jb,ib]
                si=blk.speed_i[i];sj=blk.speed_j[j];wj=blk.wj[j]
                # Interior Green representation in Ω_ca has support only on Γ_ca.
                # Hence this orbit image contributes upstairs iff jb==ca.
                same_interior=(jb==ca)
                if blk.same
                    if i==j
                        # Limits:
                        # D_ii=-w_jκ_j,
                        # S_ii=R_jj[-s_j/(2π)]+w_j[i/2-γ_E/π-(1/(2π))log((q²/4)s_j²)]s_j.
                        κj=blk.kappa_i[j];rjj=blk.Rkress[j,j];dval=ComplexF64(-wj*κj,0.0);m1=αM1*sj
                        for m in 1:M
                            li=(ca-1)*M+m;lo=C*M+m
                            if same_interior
                                m2i=((0.5*im-EULER_OVER_PI)-INV_PI*log(q[li]*sj/2))*sj
                                acc11[m]+=scale*χa*(ComplexF64(rjj*m1,0.0)+wj*m2i)
                                acc12[m]+=scale*dval
                            end
                            m2o=((0.5*im-EULER_OVER_PI)-INV_PI*log(q[lo]*sj/2))*sj
                            acc21[m]+=scale*χout*(ComplexF64(rjj*m1,0.0)+wj*m2o)
                            acc22[m]+=scale*dval
                        end
                    else
                        rr=blk.R[i,j];invr=blk.invR[i,j];lt=blk.logterm[i,j];Rij=blk.Rkress[i,j]
                        inn=_wiersig_cheb_dlp_inner(normal_mode,blk,blkrev,i,j,si,sj)
                        # A same-cavity interaction needs only the target cavity's
                        # interior q_ca family and the common exterior q_out family.
                        h0_h1_j0_j1_multi_ks_at_r!(h0in,h1in,j0in,j1in,p0in[ca],p1in[ca],pj0in[ca],pj1in[ca],blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(rr))
                        h0_h1_j0_j1_multi_ks_at_r!(h0out,h1out,j0out,j1out,p0out,p1out,pj0out,pj1out,blk.pidx[i,j],blk.tloc[i,j],blk.pidxj[i,j],blk.tlocj[i,j],Float64(rr))
                        # Include the symmetry character scale directly in the
                        # geometry factors so each subsequent radial contribution
                        # is already one term of Σ_rρ_rK_full.
                        cD1=scale*Rij*inn*invr;cD2=scale*wj*inn*invr;cD3=scale*wj*lt*inn*invr
                        cS1=scale*Rij*sj;cS2=scale*wj*sj;cS3=scale*wj*sj*lt
                        for m in 1:M
                            li=(ca-1)*M+m;lo=C*M+m
                            # Interior contribution survives only for source
                            # images belonging to the target cavity Γ_ca.
                            if same_interior
                                L1i=αL1[li]*j1in[m];L2i=αL2[li]*h1in[m]
                                M1i=αM1*j0in[m];M2i=αM2*h0in[m]
                                acc12[m]+=cD1*L1i+cD2*L2i-cD3*L1i
                                acc11[m]+=χa*(cS1*M1i+cS2*M2i-cS3*M1i)
                            end
                            # Exterior contribution always survives because Ω₀
                            # connects every physical boundary component.
                            L1o=αL1[lo]*j1out[m];L2o=αL2[lo]*h1out[m]
                            M1o=αM1*j0out[m];M2o=αM2*h0out[m]
                            acc22[m]+=cD1*L1o+cD2*L2o-cD3*L1o
                            acc21[m]+=χout*(cS1*M1o+cS2*M2o-cS3*M1o)
                        end
                    end
                else
                    # Different physical cavities cannot contribute to the
                    # interior row. Their exterior interaction is smooth:
                    # D_ab=w_j(inner/r)L₂, S_ab=w_js_jM₂.
                    rr=blk.R[i,j];invr=blk.invR[i,j];inn=_wiersig_cheb_dlp_inner(normal_mode,blk,blkrev,i,j,si,sj)
                    # Cross-cavity terms occur only in Ω₀, so only the M
                    # exterior H₀/H₁ plans are evaluated.
                    h0_h1_multi_ks_at_r!(h0out,h1out,p0out,p1out,blk.pidx[i,j],blk.tloc[i,j],Float64(rr))
                    cD=scale*wj*inn*invr;cS=scale*wj*sj
                    for m in 1:M
                        lo=C*M+m
                        acc22[m]+=cD*αL2[lo]*h1out[m]
                        acc21[m]+=χout*cS*αM2*h0out[m]
                    end
                end
            end
            # At this point accXY[m] is the complete projected layer-operator
            # entry. The jump terms are reduced-basis identities, so ±I is added
            for m in 1:M
                A=As[m];A[a,b]=acc11[m];A[a,N+b]=acc12[m];A[N+a,b]=acc21[m];A[N+a,N+b]=acc22[m]
                if a==b
                    A[a,N+b]-=one(ComplexF64)
                    A[N+a,N+b]+=one(ComplexF64)
                end
            end
        end
    end
    return nothing
end

"""
    construct_matrices!(solver,As,pts,cws;dlp_kernel=:source,multithreaded=true)

Assemble all k Wiersig matrices stored in `cws`.
For each `k_m=cws.ks[m]`, the matrix is `A(k_m)=[diag_aχ_aS_aa(n_ak_m)  diag_aD_aa(n_ak_m)-I; χ_outS_ext(n_outk_m)  D_ext(n_outk_m)+I]`. The interior row uses the cavity-dependent material wavenumbers `q_{a,m}=n_a k_m` and is block diagonal in physical cavity index. The exterior row uses the common `q_{out,m}=n_out k_m` and contains all physical cavity couplings. If `cws.direct_ws.geom.symmetry===nothing`, assembly is performed in the complete physical basis. Otherwise the same operator is assembled directly in the exact symmetry-adapted basis using the orbit data inherited from the direct workspace.
"""
function construct_matrices!(solver::WiersigKress,As::Vector{Matrix{ComplexF64}},pts::Vector{BoundaryPointsCFIE{T}},cws::WiersigChebyshevWorkspace{T};dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    _wiersig_component_indices(solver,length(pts))
    normal_mode=_wiersig_dlp_normal_mode(dlp_kernel)
    if isnothing(cws.direct_ws.geom.symmetry)
        _wiersig_all_k_full_chebyshev!(As,solver,pts,cws,normal_mode;multithreaded=multithreaded)
    else
        _wiersig_all_k_reduced_chebyshev!(As,solver,pts,cws,normal_mode;multithreaded=multithreaded)
    end
    return As
end

"""
    _wiersig_single_k_chebyshev_workspace(cws,j)

Extract a one-wavenumber view of a batch Chebyshev workspace without rebuilding any radial plans.
Let the parent workspace contain `M` vacuum wavenumbers and `C` cavities with plan ordering `ℓ_in(a,m)=(a-1)M+m` and `ℓ_out(m)=CM+m`. For the selected `k_j`, the retained plan indices are `(a-1)M+j`, `a=1,...,C`, together with `CM+j`.
The resulting workspace therefore contains exactly the material families `q_a=n_a k_j`, `a=1,...,C`, and `q_out=n_out k_j`, while reusing the same physical geometry, block caches, radial panelization and symmetry workspace.
Its local plan ordering is again component-major with one vacuum wavenumber: `qall=[n₁k_j,...,n_Ck_j,n_outk_j]`.
"""
function _wiersig_single_k_chebyshev_workspace(cws::WiersigChebyshevWorkspace{T},j::Int) where {T<:Real}
    M=length(cws.ks)
    C=cws.ncavities
    # Select the exact plan belonging to k_j from every interior material family,
    # followed by the corresponding exterior plan.
    ids=Vector{Int}(undef,C+1)
    @inbounds for a in 1:C
        ids[a]=(a-1)*M+j
    end
    ids[C+1]=C*M+j
    # Extract node j from the parent C×M material-wavenumber tables, keeping qin as C×1 and preserving cavity order q_a=n_a*k_j.
    qin=reshape(Complex{T}[cws.qin[a,j] for a in 1:C],C,1);qout=Complex{T}[cws.qout[j]];qall=cws.qall[ids]
    # Reuse the already tuned exact-q plans. No Chebyshev fitting or radial
    # cache construction is repeated for single-k extraction.
    plans0=cws.plans0[ids];plans1=cws.plans1[ids];plansj0=cws.plansj0[ids];plansj1=cws.plansj1[ids]
    # The extracted workspace contains C+1 simultaneous radial families and
    # therefore needs a correspondingly sized thread-local evaluation buffer.
    bfs=CFIE_H0_H1_J0_J1_BesselWorkspace(C+1;ntls=Threads.nthreads())
    return WiersigChebyshevWorkspace(cws.direct_ws,cws.block_cache,Complex{T}[cws.ks[j]],qin,qout,qall,C,plans0,plans1,plansj0,plansj1,bfs,cws.npanels_h,cws.M_h,cws.npanels_j,cws.M_j,cws.errH0[ids],cws.errH1[ids],cws.errJ0[ids],cws.errJ1[ids])
end

"""
    construct_matrices!(solver,A,pts,cws,j;dlp_kernel=:source,multithreaded=true)

Assemble the exact Chebyshev Wiersig matrix at `k=cws.ks[j]`. The selected operator is `A(k_j)=[diag_aχ_aS_aa(n_ak_j)  diag_aD_aa(n_ak_j)-I; χ_outS_ext(n_outk_j)  D_ext(n_outk_j)+I]`. The function first extracts the `C+1` material-q plan families associated with `k_j`, then delegates to the ordinary batch assembler with batch size one. Hence the single-k and batch pathways are mathematically identical and differ only in plan selection.
No Chebyshev plans or geometry caches are rebuilt.
"""
function construct_matrices!(solver::WiersigKress,A::Matrix{ComplexF64},pts::Vector{BoundaryPointsCFIE{T}},cws::WiersigChebyshevWorkspace{T},j::Int;dlp_kernel::Symbol=:source,multithreaded::Bool=true) where {T<:Real}
    # Build a one-k view of the parent plan table while sharing all geometry and radial interpolation data.
    work=_wiersig_single_k_chebyshev_workspace(cws,j)
    # Batch size one follows exactly the same full/reduced assembly equations as
    # the multi-k pathway, guaranteeing consistent indexing and jump terms.
    construct_matrices!(solver,Matrix{ComplexF64}[A],pts,work;dlp_kernel=dlp_kernel,multithreaded=multithreaded)
    return A
end


