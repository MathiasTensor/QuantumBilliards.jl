using Test
using QuantumBilliards

########################
#### GEOMETRY TESTS ####
########################

function geometry_test(segs::Vector{C};atol=1e-6) where {C<:QuantumBilliards.AbsCurve}
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

@testset "Rectangle boundary closedness" begin
    rect,_=make_rectangle_and_basis(2.0,1.0)
    @testset " full_boundary" begin
        geometry_closed_test(rect.full_boundary)
    end
    @testset " fundamental_boundary" begin
        geometry_closed_test(rect.fundamental_boundary)
    end
end

@testset "Circle boundary closedness" begin
    circ,_=make_circle_and_basis(1.0)
    @testset " full_boundary" begin
        geometry_closed_test(circ.full_boundary)
    end
    @testset " fundamental_boundary" begin
        geometry_closed_test(circ.fundamental_boundary)
    end
end

@testset "Ellipse boundary closedness" begin
    ellip,_=make_ellipse_and_basis(2.0,1.0) 
    @testset " full_boundary" begin
        geometry_closed_test(ellip.full_boundary)
    end
    @testset "fundamental_boundary" begin
        geometry_closed_test(ellip.fundamental_boundary)
    end
end

@testset "EquilateralTriangle boundary closedness" begin
    tri,_=make_equilateraltriangle_and_basis(1.0)
    @testset "full_boundary" begin
        geometry_closed_test(tri.full_boundary)
    end
    @testset "fundamental_boundary" begin
        geometry_closed_test(tri.fundamental_boundary)
    end
end

@testset "GeneralizedSinai boundary closedness" begin
    gs,_=make_generalized_sinai_and_basis()
    @testset "full_boundary" begin
        geometry_closed_test(gs.full_boundary)
    end
    @testset "fundamental_boundary" begin
        geometry_closed_test(gs.fundamental_boundary)
    end
end

@testset "Limacon boundary closedness" begin
    lima,_=make_limacon_and_basis(1.0,0.3)  # parameters a=1, b=0.3
    @testset "full_boundary" begin
        geometry_closed_test(lima.full_boundary)
    end
    @testset "fundamental_boundary" begin
        geometry_closed_test(lima.fundamental_boundary)
    end
end

@testset "Mushroom boundary closedness" begin
    mush,_=make_mushroom_and_basis(0.5,1.0,1.0)  # e.g. stem width=0.5, cap radius=1.0, cap height=1.0
    @testset "full_boundary" begin
        geometry_closed_test(mush.full_boundary)
    end
    @testset "fundamental_boundary" begin
        geometry_closed_test(mush.fundamental_boundary)
    end
end

@testset "Prosen boundary closedness" begin
    pros,_=make_prosen_and_basis(0.4)  # e.g. radius=1.0, deformation parameter=0.3
    @testset "full_boundary" begin
        geometry_closed_test(pros.full_boundary)
    end
    @testset "fundamental_boundary" begin
        geometry_closed_test(pros.fundamental_boundary)
    end
end

@testset "Robnik boundary closedness" begin
    rob,_=make_robnik_and_basis(0.2) 
    @testset "full_boundary" begin
        geometry_closed_test(rob.full_boundary)
    end
    @testset "fundamental_boundary" begin
        geometry_closed_test(rob.fundamental_boundary)
    end
end