using Test

#=
Contains tests for various methods for computing the eigenvalue. It is easiest to test for the accuracy of eigenvalues if integrable models like the rectangle or the circle.
=#

@testset "All methods..." begin
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

    ###########################################
    #### Vergini Saraceno (ScalingMethodA) ####
    ###########################################
    #=
    dk=0.05
    billiard,basis=make_rectangle_and_basis(w,h)
    acc_solver=ScalingMethodA(d,b)
    state_res,_=compute_spectrum_with_state(acc_solver,basis,billiard,k1,k2,dk)
    ks=state_res.ks
    # analytical data - for the odd-odd class we have m,n even
    @test all(k->any(ka->abs(ka-k)≤1e-4,ks_analytical),ks) # check if all ks are in the analytical set up to 1e-4
    =#
    #############################
    #### DecompositionMethod ####
    #############################

    kgrid=collect(range(k1,k2,step=1e-4))
    threshold=200.0
    billiard,basis=make_rectangle_and_basis(w,h)
    dm=DecompositionMethod(d,b)
    tens=k_sweep(dm,basis,billiard,kgrid)
    ks=get_eigenvalues(kgrid,tens,threshold=threshold)
    @test all(k->any(ka->abs(ka-k)≤1e-3,ks_analytical),ks) # check if all ks are in the analytical set up to 1e-3 (smaller than the grid spacing)
end
