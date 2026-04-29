using QuantumBilliards
using LinearAlgebra
using CairoMakie

try_MKL_on_x86_64!()

d=8.0
b=12.0

# geometry is a mushroom billaird with an elliptical hat, with this modification we can tune
# localization of eigenstates in configuration and phase space, mostly by changing the minor/major axes
# of the cap.
billiard,basis=make_ellipse_mushroom_and_basis(1.0,0.5,0.2,1.0)

f=Figure()
ax=Axis(f[1,1])
plot_boundary!(ax,billiard)
save("bdry.png",f)

solver=VerginiSaraceno(d,b)

# make a small DM convergence test, usually the lowest levels are the hardest for VS. This is a check if VS
# will actually work since for really non-convex geoemtries it basically failt
k1=10.0
k2=15.0
dk=0.01
solver_dm=DecompositionMethod(d,b)
tens_dm=k_sweep(solver_dm,basis,billiard,collect(k1:1e-3:k2))

state_res,_=compute_spectrum_with_state_scaling_method(
    solver,
    basis, # the RPW or CAFB basis (from make_ellipse_and_basis), with corner angle pi/2 at the origin
    billiard,
    k1,
    k2,
    dk, # heuristic dk interval, should not really be a constant, actually spectrum should be computed in chunks and checked if levels are missing in any
    multithreaded_matrices=true, # just leave on always, matrices should always be multithreaded for assembly
    multithreaded_ks=false, # for large problems this is best left to false to avoid oversubscription
)

ks=state_res.ks
tens=state_res.tens

f=Figure()
ax=Axis(f[1,1])
lines!(ax,collect(k1:1e-3:k2),log10.(tens_dm),color=:green)
scatter!(ax,ks,log10.(tens),color=:red)
save("$(nameof(typeof(billiard)))_dm_vs_test_sweep.png",f)

# high k calculation
k1=300.0
k2=300.5
dk=0.01

@time "VerginiSaraceno" state_res,_=compute_spectrum_with_state_scaling_method(solver,basis,billiard,k1,k2,dk,multithreaded_matrices=true,multithreaded_ks=false)

ks,us_all,pts_all=boundary_function(state_res,billiard,basis,b=b) # construct the normal derivative of the eigenfunction on the boundary the direct way via gradient matrices under the hood - this dispatch exists only for VerginiSaraceno solver.

Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list=wavefunction_multi_with_husimi(solver,ks,us_all,pts_all,billiard,fundamental=false) # reconstruct the wavefunctions for all states in a batch, also computes the Husimi functions and their projections onto the PSOS. fundamental just means we get the wavefunction on the fundamental domain, which is smaller and faster to compute, but since we want to plot the full wavefunction we set it to false.

# plot the wavefunction with the husimis
fs=plot_wavefunctions_with_husimi(ks,Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,billiard,us_all,pts_all,fundamental=false,max_cols=3)

# save the figure, not many, just the first page.
save("$(nameof(typeof(billiard)))_hus_wav_example.png",fs[1])