using Test
using StaticArrays
using QuantumBilliards

#######################################
#### CORNER ADAPTED FOURIER BESSEL ####
#######################################

@testset "CornerAdaptedFourierBessel" begin
    basis=CornerAdaptedFourierBessel(10,2*pi/3,SVector(0.0,0.0),0.0)
    nx=100
    ny=100
    x_grid=range(0.0,1.0,nx)
    y_grid=range(0.0,1.0,ny)

    # type tests
    pts=[SVector(x,y) for y in y_grid for x in x_grid]
    @test typeof(basis_fun(basis,1,10.0,pts)) <: AbstractArray
    @test typeof(basis_fun(basis,1:10,10.0,pts)) <: AbstractArray
    @test typeof(gradient(basis,1,10.0,pts)) <: Tuple
    @test typeof(gradient(basis,1:10,10.0,pts)) <: Tuple
    @test typeof(basis_and_gradient(basis,1,10.0,pts)) <: Tuple
    @test typeof(basis_and_gradient(basis,1:10,10.0,pts)) <: Tuple
    @test typeof(dk_fun(basis,1,10.0,pts)) <: AbstractArray
    @test typeof(dk_fun(basis,1:10,10.0,pts)) <: AbstractArray

    # symmetry on line check
    pts=[SVector(i*cos(2*pi/3),i*sin(2*pi/3)) for i in 0:0.1:1.0]
    vals=basis_fun(basis,1:10,10.0,pts)
    @test all(abs.(sum(vals,dims=2)).≤1e-3)
end

##########################
#### REAL PLANE WAVES ####
##########################

@testset "RealPlaneWaves" begin
    symmetry=Vector{Any}([XYReflection(-1,-1)])
    basis=RealPlaneWaves(10,symmetry;angle_arc=pi/2.0)
    nx=100
    ny=100
    x_grid=range(0.0,1.0,nx)
    y_grid=range(0.0,1.0,ny)

    # type tests
    pts=[SVector(x,y) for y in y_grid for x in x_grid]
    @test typeof(basis_fun(basis,1,10.0,pts)) <: AbstractArray
    @test typeof(basis_fun(basis,1:10,10.0,pts)) <: AbstractArray
    @test typeof(gradient(basis,1,10.0,pts)) <: Tuple
    @test typeof(gradient(basis,1:10,10.0,pts)) <: Tuple
    @test typeof(basis_and_gradient(basis,1,10.0,pts)) <: Tuple
    @test typeof(basis_and_gradient(basis,1:10,10.0,pts)) <: Tuple
    @test typeof(dk_fun(basis,1,10.0,pts)) <: AbstractArray
    @test typeof(dk_fun(basis,1:10,10.0,pts)) <: AbstractArray

    # symmetry on line check
    pts_x=[SVector(i,0.0) for i in 0:0.1:1.0]
    pts_y=[SVector(0.0,i) for i in 0:0.1:1.0]
    vals_x=basis_fun(basis,1:10,10.0,pts_x)
    vals_y=basis_fun(basis,1:10,10.0,pts_y)
    @test all(abs.(sum(vals_x,dims=2)).≤1e-3) && iall(abs.(sum(vals_y,dims=2)).≤1e-3)
end