using QuantumBilliards
using StaticArrays
using LinearAlgebra
using CairoMakie

# ==============================================================================
# GENERIC HYPERBOLIC TRIANGLE IN THE POINCARE DISK
# ==============================================================================
#
# This file defines a one-parameter family of hyperbolic triangles in the
# Poincare disk model.
#
# The default unperturbed angles are
#
#     A = π/8,
#     B = π/2,
#     C = π/3.
#
# The perturbation is chosen by default as
#
#     A -> A + ϵ,
#     B -> B - ϵ,
#     C -> C,
#
# so that A+B+C is unchanged. Since the hyperbolic triangle area is
#
#     area = π - (A+B+C),
#
# this perturbation preserves the area exactly.
#
# The triangle is placed in the disk as follows:
#
#     O = origin,
#     M = point on positive x-axis,
#     L = third vertex at angle A from the x-axis.
#
# The sides O-M and L-O are radial geodesics, hence straight Euclidean lines.
# The side M-L is a Poincare geodesic arc, i.e. a Euclidean circle arc
# orthogonal to the unit circle.
#
# ==============================================================================

# Hyperbolic triangle billiard represented by Poincare-disk boundary segments.
#
# The fields follow the interface expected by QuantumBilliards.jl.
#
# fundamental_boundary:
#   Boundary of the fundamental domain. Here this equals the full boundary,
#   because no further desymmetrization is used.
#
# full_boundary:
#   Full boundary of the billiard.
#
# desymmetrized_full_boundary:
#   Boundary used by some desymmetrized routines. Here again it equals the
#   full boundary.
#
# length:
#   Hyperbolic perimeter of the full triangle.
#
# length_fundamental:
#   Hyperbolic perimeter of the fundamental domain.
#
# area:
#   Hyperbolic area of the full triangle.
#
# area_fundamental:
#   Hyperbolic area of the fundamental domain.
#
# corners:
#   Euclidean Poincare-disk coordinates of the triangle vertices.
#
# angles:
#   Hyperbolic angles [A,B,C].
#
# angles_fundamental:
#   Angles of the fundamental domain. Here identical to angles.
struct GenericHyperbolicTriangle{T} <: AbsBilliard
    fundamental_boundary::Vector{PolarSegment{T}}
    full_boundary::Vector{PolarSegment{T}}
    desymmetrized_full_boundary::Vector{PolarSegment{T}}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    corners::Vector{SVector{2,T}}
    angles::Vector{T}
    angles_fundamental::Vector{T}
end

# Convert a hyperbolic radial distance l to the corresponding Euclidean radius
# in the Poincare disk.
#
# In the Poincare disk,
#
#     r = tanh(ρ/2),
#
# where ρ is hyperbolic distance from the origin.
#
# Hence a vertex at hyperbolic distance l from the origin is placed at
# Euclidean radius tanh(l/2).
@inline hrad(l)=tanh(l/2)

# Hyperbolic side length from the three angles.
#
# For a hyperbolic triangle with angles A,B,C and side lengths a,b,c opposite
# those angles, the hyperbolic law of cosines for angles gives
#
#     cosh(a) = (cos(A)+cos(B)cos(C))/(sin(B)sin(C)).
#
# Therefore this function returns side a, the side opposite angle A.
@inline function hyp_side_from_angles(A,B,C)
    return acosh((cos(A)+cos(B)*cos(C))/(sin(B)*sin(C)))
end

# Create a straight Euclidean line segment between two Poincare-disk points P,Q.
#
# In the Poincare disk model, geodesics are either:
#
#   1. Euclidean diameters of the disk, or
#   2. Euclidean circular arcs orthogonal to the unit circle.
#
# Therefore this straight segment is a hyperbolic geodesic only when P,Q and
# the origin lie on the same Euclidean line. In the construction below this is
# exactly how it is used: O-M and L-O are radial geodesics.
function radial_geodesic_segment(P::SVector{2,T},Q::SVector{2,T}) where{T<:Real}
    return PolarSegment(
        t->(one(T)-t)*P+t*Q;
        origin=SVector(zero(T),zero(T)),
        rot_angle=zero(T),
    )
end

# Construct the Poincare geodesic arc through two points P,Q.
#
# A non-radial Poincare geodesic is a Euclidean circle orthogonal to the unit
# circle. If c0 is the Euclidean center of that circle, then orthogonality to
# the unit circle implies
#
#     |c0|^2 = R^2 + 1,
#
# where R is the Euclidean radius of the geodesic circle.
#
# Since P and Q lie on the circle,
#
#     |P-c0|^2 = R^2,
#     |Q-c0|^2 = R^2.
#
# Combining this with R^2=|c0|^2-1 gives the linear conditions
#
#     P⋅c0 = (1+|P|^2)/2,
#     Q⋅c0 = (1+|Q|^2)/2.
#
# We solve this 2x2 system for c0, compute R, and then choose the arc between
# P and Q that lies inside the disk.
function poincare_geodesic_arc(P::SVector{2,T},Q::SVector{2,T}) where{T<:Real}
    # Linear system for the circle center c0.
    A=@SMatrix [P[1] P[2];Q[1] Q[2]]
    b=@SVector [(one(T)+dot(P,P))/2,(one(T)+dot(Q,Q))/2]
    c0=A\b

    # Radius of the Euclidean circle orthogonal to the unit disk.
    R=sqrt(dot(c0,c0)-one(T))

    # Angles of P and Q with respect to the geodesic-circle center.
    θP=atan(P[2]-c0[2],P[1]-c0[1])
    θQ=atan(Q[2]-c0[2],Q[1]-c0[1])

    # Counterclockwise angular difference from P to Q.
    Δccw=mod2pi(θQ-θP)

    # Clockwise angular difference from P to Q.
    Δcw=Δccw-2π

    # Test the midpoint of the counterclockwise arc.
    #
    # If that midpoint is inside the unit disk, the counterclockwise arc is the
    # desired Poincare geodesic segment. Otherwise the clockwise arc is the one
    # inside the disk.
    mid=c0+R*SVector(cos(θP+Δccw/2),sin(θP+Δccw/2))
    Δ=norm(mid)<one(T) ? Δccw : Δcw

    # Parametrize the selected circle arc.
    #
    # t=0 gives P and t=1 gives Q.
    return PolarSegment(
        t->c0+R*SVector(cos(θP+Δ*t),sin(θP+Δ*t));
        origin=SVector(zero(T),zero(T)),
        rot_angle=zero(T),
    )
end

# Construct the generic hyperbolic triangle.
#
# Type parameter:
#
#   T:
#     Floating-point type, usually Float64.
#
# Keyword arguments:
#
#   ϵ:
#     Default one-parameter perturbation size.
#
#   dA,dB,dC:
#     Direct angle perturbations. By default dA=ϵ, dB=-ϵ, dC=0, so the area
#     is preserved.
#
# Geometry:
#
#   A,B,C:
#     Triangle angles.
#
#   a,b,c:
#     Hyperbolic side lengths opposite A,B,C.
#
# Placement:
#
#   O:
#     First vertex at the origin.
#
#   M:
#     Second vertex at hyperbolic distance c from O, placed on the positive
#     x-axis. Since the side O-M is opposite C, its length is c.
#
#   L:
#     Third vertex at hyperbolic distance b from O, making angle A with O-M.
#     Since the side O-L is opposite B, its length is b.
function GenericHyperbolicTriangle(::Type{T}=Float64;ϵ::Real=0.0,dA::Real=ϵ,dB::Real=-ϵ,dC::Real=0.0) where{T<:Real}
    # Define the perturbed triangle angles.
    A=T(pi/8)+T(dA)
    B=T(pi/2)+T(dB)
    C=T(pi/3)+T(dC)

    # All angles must be strictly positive.
    A>0&&B>0&&C>0||error("Angles must be positive.")

    # Hyperbolic triangles have angle sum strictly smaller than π.
    A+B+C<T(pi)||error("Hyperbolic triangle needs A+B+C < π.")

    # Side lengths from the hyperbolic law of cosines.
    #
    # a is opposite A, b is opposite B, c is opposite C.
    a=hyp_side_from_angles(A,B,C)
    b=hyp_side_from_angles(B,A,C)
    c=hyp_side_from_angles(C,A,B)

    # First vertex: origin of the Poincare disk.
    O=SVector(zero(T),zero(T))

    # Second vertex: along the positive x-axis at hyperbolic distance c.
    M=SVector(hrad(c),zero(T))

    # Third vertex: at hyperbolic distance b from O and angular direction A.
    L=SVector(hrad(b)*cos(A),hrad(b)*sin(A))

    # Store vertices in the order used to define the boundary.
    corners=[O,M,L]

    # Boundary orientation:
    #
    #   O -> M: radial geodesic, straight Euclidean segment.
    #   M -> L: non-radial Poincare geodesic, circular arc.
    #   L -> O: radial geodesic, straight Euclidean segment.
    full_boundary=PolarSegment{T}[
        radial_geodesic_segment(O,M),
        poincare_geodesic_arc(M,L),
        radial_geodesic_segment(L,O),
    ]

    # Hyperbolic perimeter.
    length=T(a+b+c)

    # Hyperbolic area by Gauss-Bonnet:
    #
    #   area = π - (A+B+C),
    #
    # for curvature -1.
    area=T(pi)-(A+B+C)

    # Build billiard object.
    #
    # No desymmetrization is used here, so all boundary and area/length fields
    # are identical to the full billiard.
    return GenericHyperbolicTriangle{T}(
        full_boundary,
        full_boundary,
        full_boundary,
        length,
        length,
        area,
        area,
        corners,
        [A,B,C],
        [A,B,C],
    )
end

# Plot the billiard in the Poincare disk.
#
# Keyword arguments:
#
#   npts:
#     Number of plotting samples per boundary segment.
#
#   show_vertices:
#     If true, plot the three vertices.
#
#   show_labels:
#     If true, label the vertices O,M,L.
#
#   show_disk:
#     If true, draw the unit circle boundary of the Poincare disk.
#
#   show_samples:
#     If true, draw the sampled boundary points as markers. This is useful for
#     checking segment parametrizations.
function plot_billiard(
    billiard::GenericHyperbolicTriangle{T};
    npts::Int=400,
    show_vertices::Bool=true,
    show_labels::Bool=true,
    show_disk::Bool=true,
    show_samples::Bool=false,
) where{T<:Real}
    # Create square figure.
    fig=Figure(size=(800,800))

    # DataAspect keeps the disk circular rather than elliptical.
    ax=Axis(
        fig[1,1];
        aspect=DataAspect(),
        xlabel="x",
        ylabel="y",
        title="hyperbolic triangle",
    )

    # Draw unit circle, i.e. boundary at infinity of the Poincare disk.
    if show_disk
        θ=range(0,2π;length=600)
        lines!(ax,cos.(θ),sin.(θ),linewidth=2)
    end

    # Draw each boundary segment.
    for seg in billiard.full_boundary
        # Parameter values along this segment.
        ts=range(0.0,1.0;length=npts)

        # Evaluate the QuantumBilliards segment curve.
        pts=curve(seg,ts)

        # Split points into x and y arrays for plotting.
        xs=getindex.(pts,1)
        ys=getindex.(pts,2)

        # Draw boundary curve.
        lines!(ax,xs,ys,linewidth=3)

        # Optionally show the sampling points used for plotting.
        show_samples&&scatter!(ax,xs,ys,markersize=4)
    end

    # Draw vertices.
    if show_vertices
        # Extract vertex coordinates.
        xs=getindex.(billiard.corners,1)
        ys=getindex.(billiard.corners,2)

        # Plot vertices as larger markers.
        scatter!(ax,xs,ys,markersize=14)

        # Add simple vertex labels.
        if show_labels
            for (i,label) in enumerate(["O","M","L"])
                text!(ax,xs[i],ys[i];text=label,offset=(10,10),fontsize=20)
            end
        end
    end

    # Fixed view slightly larger than the disk.
    xlims!(ax,-1.1,1.1)
    ylims!(ax,-1.1,1.1)

    return fig
end

# ==============================================================================
# STANDALONE TEST / PLOT GENERATION
# ==============================================================================
#
# This block runs only when this file is executed directly, not when it is
# included from another script.
#
# It creates two example billiards:
#
#   1. ϵ=0.0, the unperturbed triangle,
#   2. ϵ=0.03, a visibly perturbed area-preserving triangle.
#
# The plots are useful for checking that:
#
#   - the vertices are placed correctly,
#   - the circular side is inside the disk,
#   - the boundary orientation is consistent,
#   - the perturbation changes the shape but preserves the area.

if abspath(PROGRAM_FILE)==@__FILE__
    # --------------------------------------------------------------------------
    # Unperturbed triangle.
    # --------------------------------------------------------------------------

    # Build the unperturbed billiard.
    billiard=GenericHyperbolicTriangle(ϵ=0.0)

    # Print basic geometric data.
    println("angles = ",billiard.angles)
    println("area = ",billiard.area)
    println("length = ",billiard.length)

    # Plot and save the unperturbed triangle.
    fig=plot_billiard(billiard;show_samples=true)
    save("hyperbolic_triangle_eps0.png",fig)

    # --------------------------------------------------------------------------
    # Perturbed triangle.
    # --------------------------------------------------------------------------

    # Build a visibly perturbed billiard.
    billiard2=GenericHyperbolicTriangle(ϵ=0.03)

    # Print basic geometric data.
    println("perturbed angles = ",billiard2.angles)
    println("perturbed area = ",billiard2.area)
    println("perturbed length = ",billiard2.length)

    # Plot and save the perturbed triangle.
    fig2=plot_billiard(billiard2;show_samples=true)
    save("hyperbolic_triangle_eps0.03.png",fig2)
end