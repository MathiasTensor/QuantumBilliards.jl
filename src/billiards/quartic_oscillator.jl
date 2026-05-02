"""
    quartic_radius(θ,E,beta)

Radius of the equipotential curve

    V(r,θ)=r^4[cos^2θ sin^2θ/2+β(cos^4θ+sin^4θ)/4]=E,

so

    r(θ)=(E/(cos^2θ sin^2θ/2+β(cos^4θ+sin^4θ)/4))^(1/4).
"""
@inline function quartic_radius(θ,E,beta) 
    c=cos(θ);s=sin(θ)
    den=0.5*c^2*s^2+beta*0.25*(c^4+s^4)
    return (E/den)^0.25
end

"""
    make_quarter_quartic_oscillator(E::T,beta::T=T(0.01);x0=0,y0=0,rot_angle=0)

Constructs the first-quadrant equipotential boundary of the quartic oscillator

    H=(p_x^2+p_y^2)/2+x^2y^2/2+β(x^4+y^4)/4,

using the polar curve `r(θ)`, `θ∈[0,π/2]`, plus virtual symmetry axes.
"""
function make_quarter_quartic_oscillator(E::T,beta::T=T(0.01);x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    origin=SVector(x0,y0)
    r_func=t->begin
        θ=T(0.5)*T(pi)*t
        r=quartic_radius(θ,E,beta)
        SVector(r*cos(θ),r*sin(θ))
    end
    quarter_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    a=quartic_radius(zero(T),E,beta)
    b=quartic_radius(T(0.5)*T(pi),E,beta)
    y_axis_segment=VirtualLineSegment(SVector(zero(T),b),SVector(zero(T),zero(T));origin=origin,rot_angle=rot_angle)
    x_axis_segment=VirtualLineSegment(SVector(zero(T),zero(T)),SVector(a,zero(T));origin=origin,rot_angle=rot_angle)
    boundary=Union{PolarSegment,VirtualLineSegment}[quarter_segment,y_axis_segment,x_axis_segment]
    corners=[SVector(a,zero(T)),SVector(zero(T),b)]
    return boundary,corners
end

"""
    make_quartic_oscillator_desymmetrized_full_boundary(E::T,beta::T=T(0.01);x0=0,y0=0,rot_angle=0)

Returns only the smooth quarter polar arc `θ∈[0,π/2]`, without virtual axes.
Useful for desymmetrized calculations where symmetry boundaries are handled separately.
"""
function make_quartic_oscillator_desymmetrized_full_boundary(E::T,beta::T=T(0.01);x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    origin=SVector(x0,y0)
    r_func=t->begin
        θ=T(0.5)*T(pi)*t
        r=quartic_radius(θ,E,beta)
        SVector(r*cos(θ),r*sin(θ))
    end
    segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    boundary=Union{PolarSegment}[segment]
    corners=SVector{2,T}[]
    return boundary,corners
end

"""
    make_full_quartic_oscillator(E::T,beta::T=T(0.01);x0=0,y0=0,rot_angle=0)

Constructs the full smooth closed equipotential curve `V(x,y)=E` as one periodic
polar segment with `θ∈[0,2π]`. For `β>0` the denominator of `r(θ)` is positive,
so the boundary has no corners or curvature jumps.
"""
function make_full_quartic_oscillator(E::T,beta::T=T(0.01);x0=zero(T),y0=zero(T),rot_angle=zero(T)) where {T<:Real}
    origin=SVector(x0,y0)
    r_func=t->begin
        θ=T(2)*T(pi)*t
        r=quartic_radius(θ,E,beta)
        SVector(r*cos(θ),r*sin(θ))
    end
    full_segment=PolarSegment(r_func;origin=origin,rot_angle=rot_angle)
    area_full=compute_area(full_segment)
    boundary=[full_segment]
    corners=SVector{2,T}[]
    return boundary,corners,area_full
end

"""
    QuarticOscillatorBilliard{T} <: AbsBilliard

Artificial billiard whose wall is the equipotential curve of the quartic oscillator

    x^2y^2/2+β(x^4+y^4)/4=E.

Fields store the full smooth boundary, a first-quadrant fundamental boundary with
virtual symmetry axes, geometric measures, and the parameters `E` and `β`.
"""
struct QuarticOscillatorBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector
    full_boundary::Vector
    desymmetrized_full_boundary::Vector
    length::T
    length_fundamental::T
    area::T
    energy::T
    beta::T
    corners::Vector{SVector{2,T}}
    area_fundamental::T
    angles::Vector
    angles_fundamental::Vector
end

"""
    QuarticOscillatorBilliard(E::T,beta::T=T(0.01);x0=0,y0=0,rot_angle=0)

Builds the smooth quartic-oscillator equipotential billiard. The default
`beta=0.01` matches the usual chaotic quartic oscillator example.
"""
function QuarticOscillatorBilliard(E::T,beta::T=T(0.01);x0=zero(T),y0=zero(T),rot_angle=zero(T))::QuarticOscillatorBilliard where {T<:Real}
    fundamental_boundary,corners=make_quarter_quartic_oscillator(E,beta;x0=x0,y0=y0,rot_angle=rot_angle)
    full_boundary,_,area_full=make_full_quartic_oscillator(E,beta;x0=x0,y0=y0,rot_angle=rot_angle)
    desymmetrized_full_boundary,_=make_quartic_oscillator_desymmetrized_full_boundary(E,beta;x0=x0,y0=y0,rot_angle=rot_angle)
    area_fundamental=area_full*T(0.25)
    length=sum([crv.length for crv in full_boundary])
    length_fundamental=symmetry_accounted_fundamental_boundary_length(fundamental_boundary)
    angles=[]
    angles_fundamental=[]
    return QuarticOscillatorBilliard(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,area_full,E,beta,corners,area_fundamental,angles,angles_fundamental)
end

"""
    make_quartic_oscillator_and_basis(E::T,beta::T=T(0.01);x0=0,y0=0,rot_angle=0)

Constructs the quartic-oscillator billiard together with an odd-odd
`XYReflection(-1,-1)` real-plane-wave basis on the first quadrant.
"""
function make_quartic_oscillator_and_basis(E::T,beta::T=T(0.01);x0=zero(T),y0=zero(T),rot_angle=zero(T))::Tuple{QuarticOscillatorBilliard{T},RealPlaneWaves} where {T<:Real}
    billiard=QuarticOscillatorBilliard(E,beta;x0=x0,y0=y0,rot_angle=rot_angle)
    symmetry=XYReflection(-1,-1)
    basis=RealPlaneWaves(10,symmetry;angle_arc=Float64(pi/2))
    return billiard,basis
end