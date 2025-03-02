using QuantumBilliards, CairoMakie, LinearAlgebra, SparseArrays, Printf

billiard,_=make_rectangle_and_basis(2.0,1.0)
#billiard,_=make_circle_and_basis(1.0)
#billiard,_=make_prosen_and_basis(0.4)
#billiard,_=make_robnik_and_basis(0.9)
#billiard,_=make_mushroom_and_basis(1.0,1.0,1.0)
#billiard,_=make_generalized_sinai_and_basis()
#billiard,_=make_ellipse_and_basis(2.0,1.0)

println("Doing billiard: ",nameof(typeof(billiard)))

fem=QuantumBilliards.FiniteElementMethod(billiard,200,200;k_max=1000.0)
x_grid,y_grid=fem.x_grid,fem.y_grid
println("Sizes in x,y: ",fem.Nx," , ",fem.Ny)
println("Interior grid pts: ",fem.Q)

@time H=QuantumBilliards.FEM_Hamiltonian(fem)

println("Constructed Hamiltonian")

nev=500
Es,wavefunctions=QuantumBilliards.compute_fem_eigenmodes(fem,nev=nev,maxiter=100000,tol=1e-8)
ks=sqrt.(abs.(Es))
println("Constructed Wavefunctions")
idxs=findall(x->x>1e-4,ks) # if ill conditioned
ks=ks[idxs]
wavefunctions=wavefunctions[idxs]

wavefunctions=[abs2.(wf) for wf in wavefunctions]

fs=QuantumBilliards.plot_wavefunctions(ks,wavefunctions,x_grid,y_grid,billiard,fundamental=false)
for i in eachindex(fs)
    save("$(nameof(typeof(billiard)))_$(i)_package.png",fs[i])
end
