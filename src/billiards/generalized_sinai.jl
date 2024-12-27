
"""
    angle_between_points(h, k, x1, y1, x2, y2) -> ϕ

Return the angle (in radians) at the center (h, k) between the two points (x1, y1)
and (x2, y2).
"""
function angle_between_points(h::T, k::T, x1::T, y1::T, x2::T, y2::T) where {T<:Real}
    v1=(x1-h,y1-k)
    v2=(x2-h,y2-k)
    dot_v1_v2=v1[1]*v2[1]+v1[2]*v2[2]
    r1=sqrt(v1[1]^2+v1[2]^2)
    r2=sqrt(v2[1]^2+v2[2]^2)
    return acos(dot_v1_v2/(r1*r2))
end

"""
    circle_top(A, theta) -> (h, k, r)

INTERNAL -> also get circle_bottom from this since (h,k,r) -> (h,-k,r)
Return the center (h, k) and radius r of the circle passing through (0, A) and (1, 1)
with a tangent of slope tan(theta) at (1,1).
"""
function circle_top(A::T, theta::T) where {T<:Real}
    cotθ=cot(theta)
    numerator=(A^2/2)-A+cotθ*(1-A)
    denominator=cotθ*(1-A)-1
    h=numerator/denominator
    k=1+(1-h)*cotθ
    r=sqrt((1-h)^2+(1-k)^2)
    return h,k,r
end

function circle_bottom(A::T, theta::T) where {T<:Real}
    h,k,r=circle_top(A,theta)
    return h,-k,r
end

"""
    circle_right(B, theta) -> (h2, k2, r2)

INTERNAL -> also get circle_left from this since (h,k,r) -> (-h,k,r)
Return the center (h2, k2) and radius r2 of the circle passing through (B,0) and (1,1)
with a tangent of slope tan(theta) at (1,1).
"""
function circle_right(B::T, theta::T) where {T<:Real}
    #=
    cotθ2=cot(theta)
    function k2_of_h2(h2)
        return 1+(1-h2)*cotθ2
    end
    function f(h2)
        k2_=k2_of_h2(h2)
        lhs=(B-h2)^2+k2_^2
        rhs=(1-h2)^2+(1-k2_)^2
        return lhs-rhs
    end
    denom=2*((1-B)-cotθ2)
    h2= -(B^2+2*cotθ2)/denom
    k2=k2_of_h2(h2)
    r2=sqrt((1-h2)^2+(1-k2)^2)
    return h2,k2,r2
    =#
    # Using the same derivation as before:
    #   (B - h2)^2 + (0 - k2)^2 = r2^2
    #   (1 - h2)^2 + (1 - k2)^2 = r2^2
    #   slope from (h2,k2) to (1,1) = -cot(θ2).
    #
    # => k2 = 1 + (1 - h2)*cot(θ2)
    # => (B - h2)^2 + k2^2 = (1 - h2)^2 + (1 - k2)^2
    #
    cotθ2 = cot(theta)

    # 1) Expand the equality of r2^2 from the two points:
    #    (B - h2)^2 + k2^2 = (1 - h2)^2 + (1 - k2)^2
    #
    # 2) Insert k2 = 1 + (1 - h2)*cot(θ2) into that equality
    #    and solve for h2. Then k2 is found easily.

    # We'll do the symbolic manipulations inline or just do them in code:
    # Let me show a minimal approach:

    # We'll define k2 in terms of h2:
    function k2_of_h2(h2)
        return 1 + (1 - h2)*cotθ2
    end

    # Then define f(h2) = LHS - RHS from the equality (B,0) vs (1,1)
    function f(h2)
        k2_ = k2_of_h2(h2)
        lhs = (B - h2)^2 + k2_^2
        rhs = (1 - h2)^2 + (1 - k2_)^2
        return lhs - rhs
    end

    # We want f(h2) = 0. In principle, you can solve analytically as well,
    # but let's do a naive approach (like a small numeric solve) for brevity
    # or we can do the direct algebra.  For completeness, let's do direct
    # algebra here if possible:

    # Expand them:
    #   LHS = B^2 - 2Bh2 + h2^2 + k2^2
    #   RHS = 1 - 2h2 + h2^2 + 1 - 2k2 + k2^2
    #       = 2 - 2h2 - 2k2 + h2^2 + k2^2
    #
    # So LHS - RHS = (B^2 - 2Bh2 + h2^2 + k2^2)
    #               - (2 - 2h2 - 2k2 + h2^2 + k2^2)
    # = B^2 - 2Bh2  - 2 + 2h2 + 2k2
    # = B^2 - 2  + 2k2 + 2h2 - 2Bh2
    #
    # Where k2 = 1 + (1 - h2)*cotθ2
    # => 2k2 = 2 + 2(1 - h2)*cotθ2
    # => 2k2 = 2 + 2cotθ2 - 2h2*cotθ2
    #
    # Then LHS - RHS = B^2 - 2 + [2 + 2cotθ2 - 2h2*cotθ2] + 2h2 - 2Bh2
    # = (B^2 - 2 + 2) + 2cotθ2
    #   + [- 2h2*cotθ2 + 2h2 - 2Bh2]
    # = B^2 + 2cotθ2 + 2h2(1 - B) - 2h2*cotθ2
    # = B^2 + 2cotθ2 + 2h2(1 - B) - 2h2*cotθ2
    #
    # Factor out 2h2 maybe:
    # = B^2 + 2cotθ2 + 2h2[ (1 - B) - h2*cotθ2/ ??? ]  (We need to be careful.)
    # Let's do it systematically:

    # Actually, let's do it directly:
    # set f(h2) = 0 => B^2 + 2cotθ2 + 2h2(1 - B) - 2h2*cotθ2 = 0
    # => 2h2( (1 - B) - cotθ2 ) = - (B^2 + 2cotθ2)
    # => h2 = - (B^2 + 2cotθ2) / (2( (1 - B) - cotθ2 ))
    #
    # That should be the direct formula:

    denom = 2 * ((1 - B) - cotθ2)
    h2 = - (B^2 + 2*cotθ2) / denom
    k2 = k2_of_h2(h2)

    # Then radius:
    r2 = sqrt( (1 - h2)^2 + (1 - k2)^2 )
    return h2, k2, r2
end

function circle_left(B::T, theta::T) where {T<:Real}
    h,k,r=circle_right(B,theta)
    return -h,k,r
end

function circle_helper(ϕ::T, h::T, k::T, r::T) where {T<:Real}
    return h+r*cos(ϕ),k+r*sin(ϕ)
end

function make_quarter_generalized_sinai(half_height::T, half_width::T, theta_right::T, theta_top::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), P1=1.0, P2=1.0) where {T<:Real}
    origin=SVector(x0,y0)
    top=SVector(x0,half_height)
    right=SVector(half_width,y0)
    ht,kt,rt=circle_top(half_height,theta_top)
    hr,kr,rr=circle_right(half_width,theta_right)
    angle_top=angle_between_points(ht,kt,x0,half_height,P1,P2)
    angle_right=angle_between_points(hr,kr,P1,P2,half_width,y0)
    #x_pi_r,y_pi_r=circle_helper(Float64(pi),hr,kr,rr) # since at pi we are not at (half_width,0)
    #angle_right_pi=angle_between_points(hr,kr,x_pi_r,y_pi_r,half_width,y0) # this is the angle between the hr,kr,rr point at pi and the (half_width,0)
    right_arc=CircleSegment(rr,angle_right,T(pi)-angle_right,hr,kr;orientation= -1) # we need to add it so that the geometry is correct. This is also corrected in the left circle segment in the full boundary analogously
    top_arc=CircleSegment(rt,angle_top,T(3*pi/2),ht,kt,orientation= -1)
    line_vertical=VirtualLineSegment(top,origin)
    line_horizontal=VirtualLineSegment(origin,right)
    boundary=Union{CircleSegment,VirtualLineSegment}[right_arc,top_arc,line_vertical,line_horizontal]
    corners=[origin]
    return boundary,corners
end

function make_desymmetrized_full_generalized_sinai(half_height::T, half_width::T, theta_right::T, theta_top::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), P1=1.0, P2=1.0) where {T<:Real}
    ht,kt,rt=circle_top(half_height,theta_top)
    hr,kr,rr=circle_right(half_width,theta_right)
    angle_top=angle_between_points(ht,kt,x0,half_height,P1,P2)
    angle_right=angle_between_points(hr,kr,P1,P2,half_width,y0)
    #x_pi_r,y_pi_r=circle_helper(Float64(pi),hr,kr,rr) # since at pi we are not at (half_width,0)
    #angle_right_pi=angle_between_points(hr,kr,x_pi_r,y_pi_r,half_width,y0)
    right_arc=CircleSegment(rr,angle_right,T(pi)-angle_right,hr,kr,orientation= -1)
    top_arc=CircleSegment(rt,angle_top,T(3*pi/2),x0,y0,orientation= -1)
    boundary=Union{CircleSegment}[right_arc,top_arc]
    corners=[]
    return boundary,corners
end

function make_full_boundary_generalized_sinai(half_height::T, half_width::T, theta_right::T, theta_top::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), P1=1.0, P2=1.0) where {T<:Real}
    origin=SVector(x0,y0)
    top=SVector(x0,half_height)
    right=SVector(half_width,y0)
    left=SVector(-half_width,y0)
    bottom=SVector(x0,-half_height)
    ht,kt,rt=circle_top(half_height,theta_top)
    hr,kr,rr=circle_right(half_width,theta_right)
    hl,kl,rl=circle_left(half_width,theta_right)
    hb,kb,rb=circle_bottom(half_height,theta_top)
    angle_top=angle_between_points(ht,kt,x0,half_height,P1,P2) # can use this for angle_bottom
    angle_right=angle_between_points(hr,kr,P1,P2,half_width,y0) # can use this for angle_left
    #x_pi_r,y_pi_r=circle_helper(Float64(pi),hr,kr,rr) # since at pi we are not at (half_width,0)
    #angle_right_pi=angle_between_points(hr,kr,x_pi_r,y_pi_r,half_width,y0)
    right_arc=CircleSegment(rr,2*angle_right,T(pi)-angle_right,hr,kr,orientation= -1)
    top_arc=CircleSegment(rt,2*angle_top,T(3*pi/2)-angle_top,ht,kt,orientation= -1)
    left_arc=CircleSegment(rl,2*angle_right,-angle_right,hl,kl,orientation= -1)
    bottom_arc=CircleSegment(rb,2*angle_top,T(pi/2)-angle_top,hb,kb,orientation= -1)
    boundary=Union{CircleSegment}[right_arc,top_arc,left_arc,bottom_arc]
    corners=[]
    return boundary,corners
end

struct GeneralizedSinai{T} <: AbsBilliard where {T<:Real}
    fundamental_boundary::Vector{Union{CircleSegment,VirtualLineSegment}}
    full_boundary::Vector{CircleSegment}
    desymmetrized_full_boundary::Vector{CircleSegment}
    length::T
    length_fundamental::T
    area::T
    area_fundamental::T
    half_width::T
    half_height::T
    theta_right::T
    theta_top::T
    corners::Vector{SVector{2,T}}
    angles::Vector
    angles_fundamental::Vector
end

function GeneralizedSinai(half_height::T, half_width::T, theta_right::T, theta_top::T) :: GeneralizedSinai where {T<:Real}
    fundamental_boundary,corners=make_quarter_generalized_sinai(half_height,half_width,theta_right,theta_top)
    desymmetrized_full_boundary,_=make_desymmetrized_full_generalized_sinai(half_height,half_width,theta_right,theta_top)
    full_boundary,_=make_full_boundary_generalized_sinai(half_height,half_width,theta_right,theta_top)
    length=sum([seg.length for seg in full_boundary])
    length_fundamental=sum([seg.length for seg in fundamental_boundary])
    area=4*0.6140 # fix later 
    area_fundamental=0.6140 # fix later
    return GeneralizedSinai(fundamental_boundary,full_boundary,desymmetrized_full_boundary,length,length_fundamental,area,area_fundamental,half_width,half_height,theta_right,theta_top,corners,[],[pi/2])
end

function make_generalized_sinai_and_basis(half_height::T, half_width::T, theta_right::T, theta_top::T; basis_type=:cafb, x0=zero(T), y0=zero(T), rot_angle=zero(T)) where {T<:Real}
    billiard=GeneralizedSinai(half_height,half_width,theta_right,theta_top)
    symmetry=Vector{Any}([XYReflection(-1,-1)])
    if basis_type==:cafb
        basis=CornerAdaptedFourierBessel(10,Float64(pi/2),SVector(x0,y0),rot_angle,symmetry)
    elseif basis_type==:rpw
        basis=RealPlaneWaves(10,symmetry;angle_arc=Float64(pi/2))
    else
        throw(ArgumentError("basis_type must be either :rpw or :cafb"))
    end
    return billiard, basis
end