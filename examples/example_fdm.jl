using QuantumBilliards
using CairoMakie
using LinearAlgebra

billiard,_=make_rectangle_and_basis(2.0,1.0)
#billiard,_=make_circle_and_basis(1.0)
#billiard,_=make_prosen_and_basis(0.4)
#billiard,_=make_robnik_and_basis(0.9)
#billiard,_=make_mushroom_and_basis(1.0,1.0,1.0)
#billiard,_=make_generalized_sinai_and_basis()
#billiard,_=make_ellipse_and_basis(2.0,1.0)

println("Doing billiard: ",nameof(typeof(billiard)))

Nx=150 # grid size in x dir
Ny=150 # grid size in y dir
kmax=1000.0 # this is needed to set the grid spacing for the finite difference method, 
# it is just used to set an accuracy grid spacing based
fem=FiniteElementMethod(billiard,Nx,Ny;k_max=kmax)
x_grid,y_grid=fem.x_grid,fem.y_grid
println("Sizes in x,y: ",fem.Nx," , ",fem.Ny)
println("Interior grid pts: ",fem.Q)

nev=100
@time "ARPACK" Es,wavefunctions=compute_fem_eigenmodes(fem,nev=nev,maxiter=100000,tol=1e-8)
ks=sqrt.(abs.(Es))
println("Constructed Wavefunctions")
idxs=findall(x->x>1e-4,ks) # sometimes has some noise at the bottom
ks=ks[idxs]
wavefunctions=wavefunctions[idxs]

wavefunctions=[abs2.(wf) for wf in wavefunctions]
fs=plot_wavefunctions(ks,wavefunctions,x_grid,y_grid,billiard,fundamental=false)
for i in eachindex(fs)
    save("$(nameof(typeof(billiard)))_$(i).png",fs[i])
end
