using Test, QuantumBilliards

analytical_rect_ks(w,h,k1,k2; mmax=40,nmax=40) =
    filter(k->k1<=k<=k2,
        sort([sqrt((m*pi/w)^2+(n*pi/h)^2) for m in 1:mmax for n in 1:nmax]))

all_computed_are_true(ks,ks_true; tol=1e-3) =
    !isempty(ks) && all(k->any(ka->abs(ka-k)<=tol,ks_true), ks)

const W_RECT=2.0
const H_RECT=1.0
const K1_RECT=18.0
const K2_RECT=20.0

@testset "VerginiSaraceno" begin
    w,h=W_RECT,H_RECT
    k1,k2=K1_RECT,K2_RECT
    ks_true=analytical_rect_ks(w,h,k1,k2)
    d=4.0;b=7.0;dk=0.05
    billiard,basis=make_rectangle_and_basis(w,h)
    solver=VerginiSaraceno(d,b)
    state_res,_=compute_spectrum_with_state_scaling_method(solver,basis,billiard,k1,k2,dk;multithreaded_matrices=true,multithreaded_ks=false)
    ks=state_res.ks
    @test all_computed_are_true(ks,ks_true; tol=1e-4)
end

@testset "EBIM BoundaryIntegralMethod" begin
    w,h=W_RECT,H_RECT
    k1,k2=K1_RECT,K2_RECT
    ks_true=analytical_rect_ks(w,h,k1,k2)
    b=15.0
    billiard,_=make_rectangle_and_basis(w,h)
    solver=BoundaryIntegralMethod(b,billiard; symmetry=nothing)
    ebim_dk(k)=0.03*k^(-1/3)
    ks,_=compute_spectrum_ebim(solver,billiard,k1,k2;dk=ebim_dk,use_lapack_raw=false,use_krylov=true,multithreaded_matrices=true,solve_info=false)
    @test all_computed_are_true(ks,ks_true;tol=1e-3)
end

@testset "EBIM DLP_kress_global_corners" begin
    w,h=W_RECT,H_RECT
    k1,k2=K1_RECT,K2_RECT
    ks_true=analytical_rect_ks(w,h,k1,k2)
    b=15.0
    billiard,_=make_rectangle_and_basis(w,h)
    solver=DLP_kress_global_corners(b,billiard;symmetry=nothing,kressq=4)
    ebim_dk(k)=0.03*k^(-1/3)
    ks,_=compute_spectrum_ebim(solver,billiard,k1,k2;dk=ebim_dk,use_lapack_raw=false,use_krylov=true,multithreaded_matrices=true,solve_info=false)
    @test all_computed_are_true(ks,ks_true; tol=1e-3)
end

@testset "BoundaryIntegralMethod local solve" begin
    w,h=W_RECT,H_RECT
    k1,k2=K1_RECT,K2_RECT
    ks_true=analytical_rect_ks(w,h,k1,k2)
    b=15.0;k_close=19.15
    billiard,_=make_rectangle_and_basis(w,h)
    solver=BoundaryIntegralMethod(b,billiard; symmetry=nothing)
    k,_=solve_wavenumber(solver,AbstractHankelBasis(),billiard,k_close,0.1)
    @test any(ka->abs(ka-k)<=1e-3,ks_true)
end

@testset "DecompositionMethod" begin
    w,h=W_RECT,H_RECT
    k1,k2=K1_RECT,K2_RECT
    ks_true=analytical_rect_ks(w,h,k1,k2)
    d=5.0;b=15.0;k_close=19.15
    billiard,basis=make_rectangle_and_basis(w,h)
    solver=DecompositionMethod(d,b)
    k,_=solve_wavenumber(solver,basis,billiard,k_close,0.1)
    @test any(ka->abs(ka-k)<=1e-3,ks_true)
end

@testset "ParticularSolutionsMethod" begin
    w,h=W_RECT,H_RECT
    k1,k2=K1_RECT,K2_RECT
    ks_true=analytical_rect_ks(w,h,k1,k2)
    d=5.0;b=15.0;k_close=19.15
    billiard,basis=make_rectangle_and_basis(w,h)
    solver=ParticularSolutionsMethod(d,b,b)
    k,_=solve_wavenumber(solver,basis,billiard,k_close,0.1)
    @test any(ka->abs(ka-k)<=1e-3,ks_true)
end

@testset "phiFD" begin
    rectangle_phi(x,y,w,h) = begin
        dx=abs(x)-w/2
        dy=abs(y)-h/2
        max(dx,dy)<0 ? max(dx,dy) :
            sqrt(max(dx,0)^2+max(dy,0)^2)
    end
    w,h=W_RECT,H_RECT
    phi(x,y)=rectangle_phi(x,y,w,h)
    ks_true=analytical_rect_ks(w,h,0.0,100.0)
    billiard,_=make_rectangle_and_basis(w,h)
    fem=QuantumBilliards.FiniteElementMethod(billiard,250,250;k_max=1000.0,offset_x_symmetric=0.1,offset_y_symmetric=0.1,fundamental=false)
    Es,_=compute_ϕ_fem_eigenmodes(fem,phi,5.0,1.01;nev=3,maxiter=50000,tol=1e-8)
    ks=sqrt.(abs.(2 .*Es))
    @test all_computed_are_true(ks,ks_true; tol=1e-2)
end

@testset "FDM" begin
    w,h=W_RECT,H_RECT
    ks_true=analytical_rect_ks(w,h,0.0,100.0)
    billiard,_=make_rectangle_and_basis(w,h)
    fem=FiniteElementMethod(billiard,400,400;k_max=1000.0)
    Es,_=compute_fem_eigenmodes(fem;nev=2,maxiter=100000,tol=1e-8)
    ks=sqrt.(abs.(2 .*Es))
    @test all_computed_are_true(ks,ks_true;tol=1e-2)
end