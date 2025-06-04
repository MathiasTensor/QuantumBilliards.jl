using Test
using StaticArrays
using LinearAlgebra
using SpecialFunctions    # for elliptic integrals
using QuantumBilliards

# =============================================================================
# Helper: Analytical ellipse perimeter using complete elliptic integral of the second kind:
#   P = 4 a E(m), where m = 1 – (b/a)^2
# =============================================================================
ellipse_perimeter(a::T,b::T) where {T<:Real}=4*a*ellipe(1-(b/a)^2)
ellipse_area(a::T,b::T) where {T<:Real}=pi*a*b

@testset "Ellipse geometry" begin
    a=2.0 # semi-major axis
    b=1.0 # semi-minor axis
    ellipse,_=QuantumBilliards.make_ellipse_and_basis(a,b)
    # Extract the single boundary segment (full_boundary should have one curve)
    @test length(ellipse.full_boundary)==1
    seg=ellipse.full_boundary[1]

    # Test that the curve is closed: curve(t=0) ≈ curve(t=1)
    P0=curve(seg,zero(eltype(a)))
    P1=curve(seg,one(eltype(a)))
    @test isapprox(P0,P1;atol=1e-8)

    # Test that arc_length(seg, 1.0) matches the analytical perimeter
    L_numeric=QuantumBilliards.arc_length(seg,one(eltype(a)))
    L_analytic=ellipse_perimeter(a,b)
    @test isapprox(L_numeric,L_analytic;atol=1e-6)

    # Area matches π * a * b
    A_numeric=QuantumBilliards.compute_area(seg)
    A_analytic=ellipse_area(a,b)
    @test isapprox(A_numeric,A_analytic;atol=1e-6)

    # Test that tangent‐vector norm is 1 after normalization
    ts=range(0.0,1.0;length=20)
    tanvecs=QuantumBilliards.tangent_vec(seg,collect(ts))
    @test all([isapprox(norm(v),1.0;atol=1e-6) for v in tanvecs])

    # Test that normal_vec is orthogonal to tangent_vec at each sample
    norvecs=QuantumBilliards.normal_vec(seg,collect(ts))
    @test all([isapprox(abs(dot(tanvecs[i],norvecs[i])),0.0,rtol=1e-8) for i in 1:length(ts)])

    # Test domain/is_inside:
    # - The center (0,0) must be inside.
    # - A point on the major axis but outside, e.g., (a + 0.5, 0), is outside.
    center_point=SVector(0.0,0.0)
    outside_point=SVector(a+0.5,0.0)
    inside_vals=QuantumBilliards.is_inside(seg,[center_point,outside_point])
    @test inside_vals[1]==true
    @test inside_vals[2]==false

    # Test boundary_coords:
    N=10000
    sampler=GaussLegendreNodes()
    bp=QuantumBilliards.boundary_coords(ellipse,sampler,N)
    xy=bp.xy
    normals=bp.normal
    s_vals=bp.s
    ds=bp.ds

    @test length(xy)==length(normals)==length(s_vals)==length(ds)
    @test isapprox(sum(ds),L_numeric;atol=1e-4)  # total ds should sum to L_numeric

    # Check a few random points from boundary_coords lie on the ellipse equation x^2/a^2 + y^2/b^2 ≈ 1
    @test all([isapprox(pt[1]^2/a^2+pt[2]^2/b^2,1.0;atol=1e-3) for pt in  xy[1:10:end]])
end