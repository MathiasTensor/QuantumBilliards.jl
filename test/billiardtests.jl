using Test
using QuantumBilliards


function geometry_closed_test(segs::Vector{C};atol=1e-6) where {C<:QuantumBilliards.AbsCurve}

    ###############################
    #### CLOSED BOUNDARY TESTS ####
    ###############################

    @testset begin
        N=length(segs)
        N==1 && return 
        # Helpers to extract the endpoints of a segment:
        get_start(seg)=curve(seg,0.0)
        get_end(seg)=curve(seg,1.0)
        # Check each consecutive pair
        for i in 1:(N-1)
            p_end=get_end(segs[i])
            p_next_st=get_start(segs[i+1])
            @test isapprox(p_end,p_next_st;atol=atol)
        end
        # Finally, wrap around: last segment’s end → first segment’s start
        p_endlast=get_end(segs[N])
        p_start0=get_start(segs[1])
        @test isapprox(p_endlast,p_start0;atol=atol)
    end
end

####################
#### GEOMETRIES ####
####################

rect,_=make_rectangle_and_basis(2.0,1.0)
geometry_closed_test(rect.full_boundary)
geometry_closed_test(rect.fundamental_boundary)

circ,_=make_circle_and_basis(1.0)
geometry_closed_test(circ.full_boundary)
geometry_closed_test(circ.fundamental_boundary)

ellip,_=make_ellipse_and_basis(2.0,1.0) 
geometry_closed_test(ellip.full_boundary)
geometry_closed_test(ellip.fundamental_boundary)

tri,_=make_triangle_and_basis(2*pi/3,1.0)
geometry_closed_test(tri.full_boundary)
geometry_closed_test(tri.fundamental_boundary)

mush,_=make_mushroom_and_basis(0.5,1.0,1.0)
geometry_closed_test(mush.full_boundary)
geometry_closed_test(mush.fundamental_boundary)

pros,_=make_prosen_and_basis(0.4) 
geometry_closed_test(pros.full_boundary)
geometry_closed_test(pros.fundamental_boundary)

rob,_=make_robnik_and_basis(0.2) 
geometry_closed_test(rob.full_boundary)
geometry_closed_test(rob.fundamental_boundary)
