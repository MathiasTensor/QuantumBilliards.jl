
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

# 8 combinations of circles that form the full generalized 4-pointed sinai billiard

"""
    circle_top_right(A, theta) -> (h, k, r)

INTERNAL -> also get circle_bottom from this since (h,k,r) -> (h,-k,r)
Return the center (h, k) and radius r of the circle passing through (0, A) and (1, 1)
with a tangent of slope tan(theta) at (1,1).
"""
function circle_top_right(A::T, theta::T) where {T<:Real}
    cotθ=cot(theta)
    numerator=(A^2/2)-A+cotθ*(1-A)
    denominator=cotθ*(1-A)-1
    h=numerator/denominator
    k=1+(1-h)*cotθ
    r=sqrt((1-h)^2+(1-k)^2)
    return h,k,r
end

# INTERNAL -> not necessarily the same as the top right since circle is not on y-axis
function circle_top_left(A::T, theta::T) where {T<:Real}
    h,k,r=circle_top(A,theta)
    return -h,k,r
end

# INTERNAL -> same as top right but -k
function circle_bottom_right(A::T, theta::T) where {T<:Real}
    h,k,r=circle_top(A,theta)
    return h,-k,r
end

# INTERNAL -> same as top left but -k
function circle_bottom_left(A::T, theta::T) where {T<:Real}
    h,k,r=circle_top(A,theta)
    return -h,-k,r
end

"""
    circle_right_up(B, theta) -> (h2, k2, r2)

INTERNAL -> also get circle_left from this since (h,k,r) -> (-h,k,r)
Return the center (h2, k2) and radius r2 of the circle passing through (B,0) and (1,1)
with a tangent of slope tan(theta) at (1,1).
"""
function circle_right_up(B::T, theta::T) where {T<:Real}
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

function circle_right_down(B::T, theta::T) where {T<:Real}
    h,k,r=circle_right_up(B,theta)
    return h,-k,r
end

function circle_left_up(B::T, theta::T) where {T<:Real}
    h,k,r=circle_right_up(B,theta)
    return -h,k,r
end

function circle_left_down(B::T, theta::T) where {T<:Real}
    h,k,r=circle_right(B,theta)
    return -h,-k,r
end

function circle_helper(ϕ::T, h::T, k::T, r::T) where {T<:Real}
    return h+r*cos(ϕ),k+r*sin(ϕ)
end

function make_quarter_generalized_sinai(half_height::T, half_width::T, theta_right::T, theta_top::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), P1=1.0, P2=1.0) where {T<:Real}
    origin=SVector(x0,y0)
    top=SVector(x0,half_height)
    right=SVector(half_width,y0)
    ht,kt,rt=circle_top_right(half_height,theta_top)
    hr,kr,rr=circle_right_up(half_width,theta_right)
    angle_top=angle_between_points(ht,kt,x0,half_height,P1,P2)
    angle_right=angle_between_points(hr,kr,P1,P2,half_width,y0)
    x_corr_r,y_corr_r=circle_helper(Float64(pi),hr,kr,rr) # since at pi we are not at (half_width,0)
    angle_right_corr=angle_between_points(hr,kr,x_corr_r,y_corr_r,half_width,y0) # this is the angle between the hr,kr,rr point at pi and the (half_width,0)
    right_arc=CircleSegment(rr,angle_right,T(pi)-angle_right-angle_right_corr,hr,kr;orientation= -1) # we need to subtract it so that the geometry is correct.
    x_corr_t,y_corr_t=circle_helper(Float64(3*pi/2),ht,kt,rt)
    angle_top_corr=angle_between_points(ht,kt,x_corr_t,y_corr_t,x0,half_height) 
    top_arc=CircleSegment(rt,angle_top,T(3*pi/2)-angle_top_corr,ht,kt,orientation= -1)
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
    x_corr_r,y_corr_r=circle_helper(Float64(pi),hr,kr,rr) # since at pi we are not at (half_width,0)
    angle_right_corr=angle_between_points(hr,kr,x_corr_r,y_corr_r,half_width,y0) # this is the angle between the hr,kr,rr point at pi and the (half_width,0)
    right_arc=CircleSegment(rr,angle_right,T(pi)-angle_right-angle_right_corr,hr,kr;orientation= -1) # we need to subtract it so that the geometry is correct.
    x_corr_t,y_corr_t=circle_helper(Float64(3*pi/2),ht,kt,rt)
    angle_top_corr=angle_between_points(ht,kt,x_corr_t,y_corr_t,x0,half_height) 
    top_arc=CircleSegment(rt,angle_top,T(3*pi/2)-angle_top_corr,ht,kt,orientation= -1)
    boundary=Union{CircleSegment}[right_arc,top_arc]
    corners=[]
    return boundary,corners
end

function make_full_boundary_generalized_sinai(half_height::T, half_width::T, theta_right::T, theta_top::T; x0=zero(T), y0=zero(T), rot_angle=zero(T), P1=1.0, P2=1.0) where {T<:Real}
    # Points that define the inner part
    origin=SVector(x0,y0)
    top=SVector(x0,half_height)
    right=SVector(half_width,y0)
    left=SVector(-half_width,y0)
    bottom=SVector(x0,-half_height)
    # Equations for the 8 segments
    htr,ktr,rtr=circle_top_right(half_height,theta_top)
    htl,ktl,rtl=circle_top_left(half_height,theta_top)
    hru,kru,rru=circle_right_up(half_width,theta_right)
    hrd,krd,rrd=circle_right_down(half_width,theta_right)
    hlu,klu,rlu=circle_left_up(half_width,theta_right)
    hll,kll,rll=circle_left_down(half_width,theta_right)
    hbl,kbl,rbl=circle_bottom_left(half_height,theta_top)
    hbr,kbr,rbr=circle_bottom_right(half_height,theta_top)
    ### Corrections ###
    # Right up
    x_corr_ru,y_corr_ru=circle_helper(Float64(pi),hru,kru,rru)
    angle_corr_ru=angle_between_points(hru,kru,x_corr_ru,y_corr_ru,half_width,y0)
    # Top right
    x_corr_tr,y_corr_tr=circle_helper(Float64(3*pi/2),htr,ktr,rtr)
    angle_corr_tr=angle_between_points(htr,ktr,x_corr_tr,y_corr_tr,x0,half_height) 
    # Top left
    x_corr_tl,y_corr_tl=circle_helper(Float64(3*pi/2),htl,ktl,rtl)
    angle_corr_tl=angle_between_points(htl,ktl,x_corr_tl,y_corr_tl,x0,half_height)
    # Left up
    x_corr_lu,y_corr_lu=circle_helper(Float64(0.0),hlu,klu,rlu)
    angle_corr_lu=angle_between_points(hlu,klu,x_corr_lu,y_corr_lu,half_width,y0)
    # Left down
    x_corr_ld,y_corr_ld=circle_helper(Float64(0.0),hll,kll,rll)
    angle_corr_ld=angle_between_points(hll,kll,x_corr_ld,y_corr_ld,half_width,y0)
    # Bottom left
    x_corr_bl,y_corr_bl=circle_helper(Float64(pi),hbl,kbl,rbl)
    angle_corr_bl=angle_between_points(hbl,kbl,x_corr_bl,y_corr_bl,x0,half_height)
    # Bottom right
    x_corr_br,y_corr_br=circle_helper(Float64(pi),hbr,kbr,rbr)
    angle_corr_br=angle_between_points(hbr,kbr,x_corr_br,y_corr_br,x0,half_height)
    # Right down
    x_corr_rd,y_corr_rd=circle_helper(Float64(pi),hrd,krd,rrd)
    angle_corr_rd=angle_between_points(hrd,krd,x_corr_rd,y_corr_rd,half_width,y0)

    ### Angle calculations ###
    angle_ru=angle_between_points(hru,kru,P1,P2,half_width,y0)
    angle_tr=angle_between_points(htr,ktr,x0,half_height,P1,P2)
    
    # Circle Segments
    right_arc_up=CircleSegment(rtr,angle_ru,T(pi)-angle_ru-angle_corr_ru,htr,ktr,orientation= -1)
    top_arc_right=CircleSegment(rtr,angle_tr,T(3*pi/2)-angle_tr-angle_corr_tr,htr,ktr,orientation= -1)
    top_arc_left=CircleSegment(rtl,angle_tr,T(3*pi/2)-angle_tr-angle_corr_tl,htl,ktl,orientation= -1)
    left_arc_up=CircleSegment(rlu,angle_ru,-angle_ru-angle_corr_lu,hlu,klu,orientation= -1)
    left_arc_down=CircleSegment(rll,angle_ru,-angle_ru-angle_corr_ld,hll,kll,orientation= -1)
    bottom_arc_left=CircleSegment(rbl,angle_tr,T(pi/2)-angle_tr-angle_corr_bl,hbl,kbl,orientation= -1)
    bottom_arc_right=CircleSegment(rbr,angle_tr,T(pi/2)-angle_tr-angle_corr_br,hbr,kbr,orientation= -1)
    right_arc_up=CircleSegment(rrd,angle_ru,T(pi)-angle_ru-angle_corr_rd,hrd,krd,orientation= -1)
    boundary=Union{CircleSegment}[right_arc_up,top_arc_right,top_arc_left,left_arc_up,left_arc_down,bottom_arc_left,bottom_arc_right,right_arc_up]
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