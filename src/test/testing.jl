using Test

#=
Contains tests for various methods for computing the eigenvalue. It is easiest to test for the accuracy of eigenvalues if integrable models like the rectangle or the circle.
=#

###########################################
#### Vergini Saraceno (ScalingMethodA) ####
###########################################

@testset "ScalingMethodA" begin
    w=2.0
    h=1.0
    d=4.0
    b=7.0
    k1=18.0
    k2=20.0
    k_analytical(_m,_n,_w,_h)=sqrt((_m*pi/_w)^2+(_n*pi/_h)^2)
    ks_analytical=[k_analytical(m,n,w,h) for m=0:10, n=0:10 if (m>0 && n>0)]
    sort!(ks_analytical)
    ks_analytical=filter(k->k1≤k≤k2,ks_analytical) # filter to the range of interest
    
    dk=0.05
    billiard,basis=make_rectangle_and_basis(w,h)
    acc_solver=ScalingMethodA(d,b)
    state_res,_=compute_spectrum_with_state(acc_solver,basis,billiard,k1,k2,dk)
    ks=state_res.ks
    # analytical data - for the odd-odd class we have m,n even
    @test all(k->any(ka->abs(ka-k)≤1e-4,ks_analytical),ks) # check if all ks are in the analytical set up to 1e-4
end

###########################################
#### Expanded Boundary Integral Method ####
###########################################

@testset "EBIM" begin
    w=2.0
    h=1.0
    d=4.0
    b=15.0
    k1=18.0
    k2=20.0
    k_analytical(_m,_n,_w,_h)=sqrt((_m*pi/_w)^2+(_n*pi/_h)^2)
    ks_analytical=[k_analytical(m,n,w,h) for m=0:10, n=0:10 if (m>0 && n>0)]
    sort!(ks_analytical)
    ks_analytical=filter(k->k1≤k≤k2,ks_analytical) # filter to the range of interest

    ebim_dk(k)=0.03*k^(-1/3)
    billiard,basis=make_rectangle_and_basis(w,h)
    symmetries=Vector{Any}([XYReflection(-1,-1)])
    ebim_solver=ExpandedBoundaryIntegralMethod(b,billiard,symmetries=symmetries,x_bc=:D,y_bc=:D)
    ks,_=compute_spectrum(ebim_solver,billiard,k1,k2,dk=ebim_dk,use_lapack_raw=true)
    @test all(k->any(ka->abs(ka-k)≤1e-3,ks_analytical),ks) 
end

##################################
#### Boundary Integral Method ####
##################################

@testset "BoundaryIntegralMethod" begin
    w=2.0
    h=1.0
    d=5.0
    b=15.0
    k1=18.0
    k2=20.0
    k_analytical(_m,_n,_w,_h)=sqrt((_m*pi/_w)^2+(_n*pi/_h)^2)
    ks_analytical=[k_analytical(m,n,w,h) for m=0:10, n=0:10 if (m>0 && n>0)]
    sort!(ks_analytical)
    ks_analytical=filter(k->k1≤k≤k2,ks_analytical) # filter to the range of interest
    k_close_to_true=19.15 # this is close to a true eigenvalue
    billiard,basis=make_rectangle_and_basis(w,h)
    symmetries=Vector{Any}([XYReflection(-1,-1)])
    bim=BoundaryIntegralMethod(b,billiard,symmetries=symmetries,x_bc=:D,y_bc=:D)
    k,_=solve_wavenumber(bim,AbstractHankelBasis(),billiard,k_close_to_true,0.1)
    @test any(ka->abs(ka-k)≤1e-3,ks_analytical)
end

##############################
#### Decomposition Method ####
##############################

@testset "DecompositionMethod" begin
    w=2.0
    h=1.0
    d=5.0
    b=15.0
    k1=18.0
    k2=20.0
    k_analytical(_m,_n,_w,_h)=sqrt((_m*pi/_w)^2+(_n*pi/_h)^2)
    ks_analytical=[k_analytical(m,n,w,h) for m=0:10, n=0:10 if (m>0 && n>0)]
    sort!(ks_analytical)
    ks_analytical=filter(k->k1≤k≤k2,ks_analytical) # filter to the range of interest
    k_close_to_true=19.15 # this is close to a true eigenvalue
    billiard,basis=make_rectangle_and_basis(w,h)
    dm=DecompositionMethod(d,b)
    k,_=solve_wavenumber(dm,basis,billiard,k_close_to_true,0.1)
    @test any(ka->abs(ka-k)≤1e-3,ks_analytical)
end

#####################################
#### Particular Solutions Method ####
#####################################

@testset "ParticularSolutionsMethod" begin
    w=2.0
    h=1.0
    d=5.0
    b=15.0
    k1=18.0
    k2=20.0
    k_analytical(_m,_n,_w,_h)=sqrt((_m*pi/_w)^2+(_n*pi/_h)^2)
    ks_analytical=[k_analytical(m,n,w,h) for m=0:10, n=0:10 if (m>0 && n>0)]
    sort!(ks_analytical)
    ks_analytical=filter(k->k1≤k≤k2,ks_analytical) # filter to the range of interest
    k_close_to_true=19.15 # this is close to a true eigenvalue
    billiard,basis=make_rectangle_and_basis(w,h)
    psm=ParticularSolutionsMethod(d,b,b)
    k,_=solve_wavenumber(psm,basis,billiard,k_close_to_true,0.1)
    @test any(ka->abs(ka-k)≤1e-3,ks_analytical)
end

###################
#### Phi - FDM ####
###################

@testset "phiFD" begin
    function rectangle_phi(x,y,width,height)
        dx=abs(x)-width/2
        dy=abs(y)-height/2
        return max(dx,dy) < 0 ? max(dx,dy) : sqrt(max(dx,0)^2+max(dy,0)^2)
    end
    w=2.0
    h=1.0
    phi(x,y)=rectangle_phi(x,y,w,h)
    k_analytical(_m,_n,_w,_h)=sqrt((_m*pi/_w)^2+(_n*pi/_h)^2)
    ks_analytical=[k_analytical(m,n,w,h) for m=0:10, n=0:10 if (m>0 && n>0)]
    billiard,_=make_rectangle_and_basis(w,h)
    fundamental=false
    fem=QuantumBilliards.FiniteElementMethod(billiard,120,120;k_max=1000.0,offset_x_symmetric=0.1,offset_y_symmetric=0.1,fundamental=fundamental)
    nev=10
    γ=5.0
    σ=1.01
    ks,_=compute_ϕ_fem_eigenmodes(fem,phi,γ,σ,nev=nev,maxiter=50000,tol=1e-8)
    @test all(k->any(ka->abs(ka-k)≤1e-3,ks_analytical),ks) 
end