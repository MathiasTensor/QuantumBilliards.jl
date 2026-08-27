################################################################################
########################## SYMMETRY-REDUCED BOUNDARY MAPS ######################
################################################################################
# Boundary-integral solvers with discrete symmetries are assembled on a reduced
# fundamental domain rather than on the full boundary.
#
# If G is the symmetry group and b denotes one fundamental boundary node, the
# corresponding full-boundary orbit is
#
#     O_b={g·b : g∈G}.
#
# In a chosen irreducible representation, the boundary unknown satisfies
#
#     u(g·b)=χ(g)u(b),
#
# where χ(g) is the parity factor for reflections or the character factor for
# rotations. A full-boundary source sum can therefore be folded into one reduced
# source column:
#
#     K_red(i,b)=Σ_{g∈G} χ(g)K(i,g·b).
#
# The purpose of the orbit maps below is to encode this reduction once in exact
# integer index space. They provide both
#
#     fundamental index -> all full indices in its symmetry orbit
#
# and
#
#     full index -> corresponding fundamental index and irrep factor.
#
# The boundary discretizations are constructed to respect the symmetry exactly:
# the number of nodes is a multiple of |G|, the fundamental domain is the first
# contiguous block, and symmetry-related components have identical node counts
# and known ordering. Consequently no geometric point matching or floating-point
# tolerance is required; all symmetry actions are exact permutations of boundary
# indices.
#
# The same SymmetryOrbitMap representation is used by DLP, DLP-Kress and CFIE-Kress,
# so matrix sizing, reduced kernel assembly, and reconstruction can share one
# common symmetry interface.
################################################################################

@inline symmetry_order(::BilliardGeometry.XAxisReflection)=2
@inline symmetry_order(::BilliardGeometry.YAxisReflection)=2
@inline symmetry_order(::BilliardGeometry.XYAxisReflection)=4
@inline symmetry_order(sym::BilliardGeometry.NFoldRotation)=sym.order
@inline symmetry_order(syms::AbstractVector{<:BilliardGeometry.NFoldRotation})=isempty(syms) ? 1 : syms[1].order

"""
    SymmetryOrbitMap{T}
Store the exact correspondence between a symmetry-reduced boundary and the full
boundary discretization.
For a symmetry group `G` of order `ng`, each reduced boundary degree of freedom
represents one full-boundary orbit
    O_b={g·b : g∈G}.
The matrices `fund_to_full` and `fund_to_scale` describe this expansion directly:
    q = fund_to_full[g,b],
    χ = fund_to_scale[g,b],
meaning that group image `g` of reduced node `b` is full node `q` and
    u[q]=χ*u[Ifund[b]].
The inverse arrays `full_to_fund` and `full_to_scale` provide the corresponding
full-to-reduced lookup.
This representation is used when constructing symmetry-reduced boundary-integral
matrices, where a complete source orbit is folded into one reduced source column.
## Attributes
* `Ifund`: Full-boundary indices chosen as representatives of the fundamental domain.
* `full_to_fund`: Reduced-orbit index associated with every full-boundary node.
* `full_to_scale`: Irrep factor relating every full-boundary node to its representative.
* `fund_to_full`: Full-boundary indices in each orbit; rows are symmetry images and columns are reduced nodes.
* `fund_to_scale`: Irrep factors corresponding to `fund_to_full`.
"""
struct SymmetryOrbitMap{T<:Real}
    Ifund::Vector{Int}
    full_to_fund::Vector{Int}
    full_to_scale::Vector{Complex{T}}
    fund_to_full::Matrix{Int}
    fund_to_scale::Matrix{Complex{T}}
end
Base.length(orbits::SymmetryOrbitMap)=length(orbits.Ifund)
@inline fundamental_size(orbits::SymmetryOrbitMap)=length(orbits.Ifund)
@inline full_size(orbits::SymmetryOrbitMap)=length(orbits.full_to_fund)
@inline orbit_size(orbits::SymmetryOrbitMap)=size(orbits.fund_to_full,1)

"""
    symmetry_orbit(orbits,b)
Return the full-boundary indices and irrep factors represented by reduced node
`b`. If qs,χs=symmetry_orbit(orbits,b),
then the reduced boundary value `u_b` generates the full orbit according to u[qs[l]]=χs[l]*u_b.
The returned arrays are views into `orbits` and therefore allocate no copies.
"""
@inline function symmetry_orbit(orbits::SymmetryOrbitMap,b::Int)
    return @view(orbits.fund_to_full[:,b]),@view(orbits.fund_to_scale[:,b])
end

"""
    _build_periodic_symmetry_orbit_map(::Type{T},perms,scales)
Build a [`SymmetryOrbitMap`](@ref) from exact boundary-index permutations.
Each entry `perms[g]` represents one symmetry-group element: perms[g][q]
is the full-boundary node obtained by applying group element `g` to node `q`.
The corresponding representation factor is `scales[g]`.
The boundary discretization is assumed to have been constructed consistently
with the symmetry:
* `N` is divisible by the group order `ng=length(perms)`;
* the first `N/ng` nodes form the fundamental domain;
* every symmetry image is represented by an exact index permutation.
Under these assumptions no coordinate matching is required. The resulting map is
used directly by reduced boundary-integral assembly.
## Arguments
* `T`: Real scalar type used for the complex irrep factors.
* `perms`: Full-boundary index permutation for each group element.
* `scales`: Parity or character factor associated with each group element.
## Returns
A [`SymmetryOrbitMap{T}`](@ref).
"""
function _build_periodic_symmetry_orbit_map(::Type{T},perms::Vector{Vector{Int}},scales::Vector{Complex{T}}) where {T<:Real}
    ng=length(perms)
    ng>0||throw(ArgumentError("At least one symmetry permutation is required"))
    length(scales)==ng||throw(DimensionMismatch("Received $ng permutations but $(length(scales)) irrep factors"))
    N=length(perms[1])
    all(length(p)==N for p in perms)||throw(DimensionMismatch("All symmetry permutations must have length $N"))
    N%ng==0||throw(ArgumentError("Node count N=$N must be divisible by symmetry-group order $ng"))
    nf=N÷ng
    Ifund=collect(1:nf)
    fund_to_full=Matrix{Int}(undef,ng,nf)
    fund_to_scale=Matrix{Complex{T}}(undef,ng,nf)
    full_to_fund=Vector{Int}(undef,N)
    full_to_scale=Vector{Complex{T}}(undef,N)
    @inbounds for b in 1:nf
        q=Ifund[b]
        for g in 1:ng
            qi=perms[g][q]
            χ=scales[g]
            fund_to_full[g,b]=qi
            fund_to_scale[g,b]=χ
            full_to_fund[qi]=b
            full_to_scale[qi]=χ
        end
    end
    return SymmetryOrbitMap{T}(Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale)
end

################################################################################
######################## PERIODIC SINGLE-BOUNDARY MAPS #########################
################################################################################
# Exact index actions for a periodically ordered single boundary. These formulas
# replace geometric symmetry searches: the boundary-node ordering itself encodes
# the symmetry action.
@inline _idx_reflect_half(q::Int,N::Int)=mod1(N-q+1,N)
@inline _idx_reflect_x_quarter(q::Int,N::Int)=mod1(N÷2-q+1,N)
@inline _idx_reflect_y_quarter(q::Int,N::Int)=mod1(N-q+1,N)
@inline _idx_rotate_pi(q::Int,N::Int)=mod1(q+N÷2,N)
@inline _idx_rotate(q::Int,N::Int,n::Int,l::Int)=mod1(q+l*(N÷n),N)
"""
    symmetry_index_orbits(::Type{T},N,sym::BilliardGeometry.XAxisReflection)
Build the exact two-element boundary orbits generated by reflection across the
x-axis. The periodic discretization must contain an even number of nodes. The first
`N/2` nodes are taken as the fundamental domain and each is paired with its
reflected image. The second member of every orbit carries the parity
`sym.parity_y`.
## Returns
A [`SymmetryOrbitMap`](@ref) with orbit size `2` and reduced size `N/2`.
"""
function symmetry_index_orbits(::Type{T},N::Int,sym::BilliardGeometry.XAxisReflection) where {T<:Real}
    iseven(N)||throw(ArgumentError("XAxisReflection requires an even node count; received N=$N"))
    id=collect(1:N)
    refl=Vector{Int}(undef,N)
    @inbounds for q in 1:N
        refl[q]=_idx_reflect_half(q,N)
    end
    χ=BilliardGeometry.symmetry_irrep_character(T,sym)
    return _build_periodic_symmetry_orbit_map(T,[id,refl],[one(Complex{T}),χ])
end

"""
    symmetry_index_orbits(::Type{T},N,sym::BilliardGeometry.YAxisReflection)
Build the exact two-element boundary orbits generated by reflection across the
y-axis. The periodic discretization must contain an even number of nodes. The first
`N/2` nodes form the fundamental domain and the remaining nodes are their
reflected images. The image contribution carries the parity `sym.parity_x`.
## Returns
A [`SymmetryOrbitMap`](@ref) with orbit size `2` and reduced size `N/2`.
"""
function symmetry_index_orbits(::Type{T},N::Int,sym::BilliardGeometry.YAxisReflection) where {T<:Real}
    iseven(N)||throw(ArgumentError("YAxisReflection requires an even node count; received N=$N"))
    id=collect(1:N)
    refl=Vector{Int}(undef,N)
    @inbounds for q in 1:N
        refl[q]=_idx_reflect_half(q,N)
    end
    χ=BilliardGeometry.symmetry_irrep_character(T,sym)
    return _build_periodic_symmetry_orbit_map(T,[id,refl],[one(Complex{T}),χ])
end

"""
    symmetry_index_orbits(::Type{T},N,sym::BilliardGeometry.XYAxisReflection)
Build the four-element `D₂` boundary orbits generated by reflections across both
coordinate axes. Each fundamental node represents the orbit {I,Rx,Ry,RxRy},
with irrep factors {1,parity_x,parity_y,parity_x*parity_y}.
The periodic node count must therefore be divisible by four and the first `N/4`
nodes form the fundamental domain.
## Returns
A [`SymmetryOrbitMap`](@ref) with orbit size `4` and reduced size `N/4`.
"""
function symmetry_index_orbits(::Type{T},N::Int,sym::BilliardGeometry.XYAxisReflection) where {T<:Real}
    N%4==0||throw(ArgumentError("XYAxisReflection requires N divisible by four; received N=$N"))
    id=collect(1:N)
    rx=Vector{Int}(undef,N)
    ry=Vector{Int}(undef,N)
    rxy=Vector{Int}(undef,N)
    @inbounds for q in 1:N
        rx[q]=_idx_reflect_x_quarter(q,N)
        ry[q]=_idx_reflect_y_quarter(q,N)
        rxy[q]=_idx_rotate_pi(q,N)
    end
    χx=Complex{T}(sym.parity_x)
    χy=Complex{T}(sym.parity_y)
    scales=Complex{T}[one(Complex{T}),χx,χy,χx*χy]
    return _build_periodic_symmetry_orbit_map(T,[id,rx,ry,rxy],scales)
end

"""
    symmetry_index_orbits(::Type{T},N,sym::BilliardGeometry.NFoldRotation)
Build the complete `Cₙ` boundary-orbit decomposition associated with an
`NFoldRotation`.
Although `NFoldRotation` represents one geometric image `R^m`, reduction requires
the complete cyclic group
    I,R,R²,...,Rⁿ⁻¹.
For irrep sector `s=sym.sector`, image `R^l` carries the character
    χ_l=exp(2π i s l/n).
The node count must be divisible by `n=sym.order`; the first `N/n` nodes are the
fundamental domain.
## Returns
A [`SymmetryOrbitMap`](@ref) with orbit size `n` and reduced size `N/n`.
"""
function symmetry_index_orbits(::Type{T},N::Int,sym::BilliardGeometry.NFoldRotation) where {T<:Real}
    n=sym.order
    n>=2||throw(ArgumentError("Rotation order must be at least two; received n=$n"))
    N%n==0||throw(ArgumentError("C$n symmetry requires N divisible by $n; received N=$N"))
    perms=Vector{Vector{Int}}(undef,n)
    scales=Vector{Complex{T}}(undef,n)
    @inbounds for l in 0:n-1
        p=Vector{Int}(undef,N)
        for q in 1:N
            p[q]=_idx_rotate(q,N,n,l)
        end
        perms[l+1]=p
        scales[l+1]=cis(T(2pi)*T(sym.sector*l)/T(n))
    end
    return _build_periodic_symmetry_orbit_map(T,perms,scales)
end

"""
    symmetry_index_orbits(::Type{T},N,syms::AbstractVector{<:BilliardGeometry.NFoldRotation})
Build a complete cyclic orbit map from the nontrivial rotation images returned by
[`Cn_symmetry`](@ref).
All supplied images must belong to the same cyclic group and irrep sector. Since
the complete orbit is determined entirely by the common order and sector, one
representative image is sufficient to construct the map.
"""
function symmetry_index_orbits(::Type{T},N::Int,syms::AbstractVector{<:BilliardGeometry.NFoldRotation}) where {T<:Real}
    isempty(syms)&&throw(ArgumentError("Rotation image collection cannot be empty"))
    order=syms[1].order
    sector=syms[1].sector
    all(s->s.order==order&&s.sector==sector,syms)||throw(ArgumentError("All NFoldRotation images must have identical order and irrep sector"))
    return symmetry_index_orbits(T,N,syms[1])
end

################################################################################
######################## MULTICOMPONENT EXACT ORBITS ############################
################################################################################
# For CFIE-type geometries the same symmetry reduction must also account for
# permutations of complete boundary components. Symmetry-related components are
# constructed with identical discretizations and stored in a known block order,
# so the full node permutation can again be generated exactly from indices.

"""
    _component_block_permutation(pts,component_map;reverse_nodes=false)
Lift an exact permutation of boundary components to a permutation of the
flattened boundary-node indices.
If component `a` maps to component
    b=component_map[a],
then every local node of `a` is mapped to the corresponding local node of `b`.
Symmetry-related components must therefore contain the same number of nodes.
For an orientation-reversing reflection, `reverse_nodes=true` additionally maps
local index
    j -> N-j+1,
so that the transformed component follows the stored boundary orientation.
## Arguments
* `pts`: Boundary components in flattened assembly order.
* `component_map`: Exact component permutation.
* `reverse_nodes`: Reverse the local node ordering after mapping components.
## Returns
A global permutation of the flattened boundary-node indices.
"""
function _component_block_permutation(pts::Vector{BoundaryPoints{T}},component_map::AbstractVector{Int};reverse_nodes::Bool=false) where {T<:Real}
    nc=length(pts)
    length(component_map)==nc||throw(DimensionMismatch("Expected $nc component-map entries, received $(length(component_map))"))
    offs=component_offsets(pts)
    N=offs[end]-1
    perm=Vector{Int}(undef,N)
    @inbounds for a in 1:nc
        b=component_map[a]
        1<=b<=nc||throw(BoundsError(component_map,b))
        Na=length(pts[a])
        Nb=length(pts[b])
        Na==Nb||throw(DimensionMismatch("Symmetry-related components $a and $b have node counts $Na and $Nb"))
        oa=offs[a]
        ob=offs[b]
        for j in 1:Na
            jb=reverse_nodes ? Na-j+1 : j
            perm[oa+j-1]=ob+jb-1
        end
    end
    return perm
end

"""
    _component_identity_map(nc)
Return the identity permutation of `nc` boundary components.
"""
@inline _component_identity_map(nc::Int)=collect(1:nc)

"""
    _component_rotation_map(nc,n,l)
Return the exact component permutation generated by the `l`-th power of a
`Cₙ` rotation.
Components are assumed to be stored in consecutive symmetry-orbit blocks:
    [O₁⁰,O₁¹,...,O₁ⁿ⁻¹,O₂⁰,O₂¹,...,O₂ⁿ⁻¹,...].
Within each block, rotation by `R^l` maps image index
    j -> j+l mod n.
The total component count must therefore be divisible by `n`.
"""
function _component_rotation_map(nc::Int,n::Int,l::Int)
    nc%n==0||throw(ArgumentError("$nc boundary components cannot be partitioned into C$n component orbits"))
    map=Vector{Int}(undef,nc)
    @inbounds for a in 1:nc
        orbit=(a-1)÷n
        img=(a-1)%n
        map[a]=orbit*n+mod(img+l,n)+1
    end
    return map
end

"""
    _component_reflection_pair_map(nc)
Return the component permutation for a single reflection when symmetry-related
components are stored in consecutive pairs: (1,2),(3,4),...
Each pair is exchanged by the reflection. The component count must therefore be
even.
"""
function _component_reflection_pair_map(nc::Int)
    iseven(nc)||throw(ArgumentError("Single-reflection component reduction requires an even number of components; received $nc"))
    map=Vector{Int}(undef,nc)
    @inbounds for a in 1:2:nc
        map[a]=a+1
        map[a+1]=a
    end
    return map
end

"""
    _component_d2_maps(nc)
Return the three nontrivial component permutations of the reflection group `D₂`.
Each physical component orbit must be stored as [I,Rx,Ry,RxRy].
The returned maps implement left multiplication by `Rx`, `Ry`, and `RxRy`
within each four-component block.
## Returns
`rx,ry,rxy`, where each vector is an exact component permutation.
"""
function _component_d2_maps(nc::Int)
    nc%4==0||throw(ArgumentError("D₂ component reduction requires the component count to be divisible by four; received $nc"))
    rx=Vector{Int}(undef,nc)
    ry=Vector{Int}(undef,nc)
    rxy=Vector{Int}(undef,nc)
    @inbounds for a0 in 1:4:nc
        i=a0
        x=a0+1
        y=a0+2
        xy=a0+3
        rx[i]=x
        rx[x]=i
        rx[y]=xy
        rx[xy]=y
        ry[i]=y
        ry[y]=i
        ry[x]=xy
        ry[xy]=x
        rxy[i]=xy
        rxy[xy]=i
        rxy[x]=y
        rxy[y]=x
    end
    return rx,ry,rxy
end

"""
    symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.XAxisReflection)
Build the complete symmetry-orbit map for a multicomponent boundary reflected
across the x-axis.
Symmetry-related components must occur in consecutive pairs and have identical
node counts. Reflection exchanges each component pair and reverses the local
boundary-node ordering.
Each reduced degree of freedom therefore represents two full-boundary nodes with
factors {1,parity_y}.
"""
function symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.XAxisReflection) where {T<:Real}
    nc=length(pts)
    nc==1&&return symmetry_index_orbits(T,pts[1],sym)
    id=_component_block_permutation(pts,_component_identity_map(nc))
    refl=_component_block_permutation(pts,_component_reflection_pair_map(nc);reverse_nodes=true)
    χ=BilliardGeometry.symmetry_irrep_character(T,sym)
    return _build_periodic_symmetry_orbit_map(T,[id,refl],[one(Complex{T}),χ])
end

"""
    symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.YAxisReflection)
Build the complete symmetry-orbit map for a multicomponent boundary reflected
across the y-axis.
Symmetry-related components must occur in consecutive pairs and have identical
node counts. Reflection exchanges each pair and reverses the local node ordering.
Each reduced degree of freedom represents two full-boundary nodes with factors
    {1,parity_x}.
"""
function symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.YAxisReflection) where {T<:Real}
    nc=length(pts)
    nc==1&&return symmetry_index_orbits(T,pts[1],sym)
    id=_component_block_permutation(pts,_component_identity_map(nc))
    refl=_component_block_permutation(pts,_component_reflection_pair_map(nc);reverse_nodes=true)
    χ=BilliardGeometry.symmetry_irrep_character(T,sym)
    return _build_periodic_symmetry_orbit_map(T,[id,refl],[one(Complex{T}),χ])
end

"""
    symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.XYAxisReflection)
Build the complete `D₂` orbit map for a multicomponent boundary invariant under
both coordinate reflections.
Each component orbit must be stored as [I,Rx,Ry,RxRy],
with identical node counts. The single reflections reverse local node ordering,
whereas the double reflection preserves it.
The corresponding irrep factors are {1,parity_x,parity_y,parity_x*parity_y}.
"""
function symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.XYAxisReflection) where {T<:Real}
    nc=length(pts)
    nc==1&&return symmetry_index_orbits(T,pts[1],sym)
    idmap=_component_identity_map(nc)
    rxmap,rymap,rxymap=_component_d2_maps(nc)
    id=_component_block_permutation(pts,idmap)
    rx=_component_block_permutation(pts,rxmap;reverse_nodes=true)
    ry=_component_block_permutation(pts,rymap;reverse_nodes=true)
    rxy=_component_block_permutation(pts,rxymap;reverse_nodes=false)
    χx=Complex{T}(sym.parity_x)
    χy=Complex{T}(sym.parity_y)
    scales=Complex{T}[one(Complex{T}),χx,χy,χx*χy]
    return _build_periodic_symmetry_orbit_map(T,[id,rx,ry,rxy],scales)
end

"""
    symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.NFoldRotation)
Build the complete `Cₙ` orbit map for a multicomponent rotationally symmetric
boundary.
Components are assumed to be stored in consecutive `n`-image blocks, with
identical node counts within each orbit. Rotation preserves the local node order.
For irrep sector `s`, the `l`-th rotational image contributes with χ_l=exp(2π i s l/n).
The resulting reduced boundary contains one representative for every complete
`Cₙ` node orbit.
"""
function symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},sym::BilliardGeometry.NFoldRotation) where {T<:Real}
    nc=length(pts)
    nc==1&&return symmetry_index_orbits(T,pts[1],sym)
    n=sym.order
    nc%n==0||throw(ArgumentError("$nc components cannot be partitioned into C$n symmetry orbits"))
    perms=Vector{Vector{Int}}(undef,n)
    scales=Vector{Complex{T}}(undef,n)
    @inbounds for l in 0:n-1
        cmap=_component_rotation_map(nc,n,l)
        perms[l+1]=_component_block_permutation(pts,cmap)
        scales[l+1]=cis(T(2pi)*T(sym.sector*l)/T(n))
    end
    return _build_periodic_symmetry_orbit_map(T,perms,scales)
end

"""
    symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},syms::AbstractVector{<:BilliardGeometry.NFoldRotation})
Build a multicomponent cyclic orbit map from the nontrivial rotation images
returned by [`Cn_symmetry`](@ref).
All images must belong to the same cyclic group and irrep sector. The complete
group action is reconstructed from their common order and sector.
"""
function symmetry_index_orbits(::Type{T},pts::Vector{BoundaryPoints{T}},syms::AbstractVector{<:BilliardGeometry.NFoldRotation}) where {T<:Real}
    isempty(syms)&&throw(ArgumentError("Rotation image collection cannot be empty"))
    order=syms[1].order
    sector=syms[1].sector
    all(s->s.order==order&&s.sector==sector,syms)||throw(ArgumentError("All NFoldRotation images must have identical order and irrep sector"))
    return symmetry_index_orbits(T,pts,syms[1])
end

@inline symmetry_index_orbits(::Type{T},pts::BoundaryPoints,sym::BilliardGeometry.XAxisReflection) where {T<:Real}=symmetry_index_orbits(T,length(pts),sym)
@inline symmetry_index_orbits(::Type{T},pts::BoundaryPoints,sym::BilliardGeometry.YAxisReflection) where {T<:Real}=symmetry_index_orbits(T,length(pts),sym)
@inline symmetry_index_orbits(::Type{T},pts::BoundaryPoints,sym::BilliardGeometry.XYAxisReflection) where {T<:Real}=symmetry_index_orbits(T,length(pts),sym)
@inline symmetry_index_orbits(::Type{T},pts::BoundaryPoints,sym::BilliardGeometry.NFoldRotation) where {T<:Real}=symmetry_index_orbits(T,length(pts),sym)