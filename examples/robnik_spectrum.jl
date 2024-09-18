using QuantumBilliards
using StaticArrays
using GLMakie
using LinearAlgebra






do_benchmark = false
do_spectrum = false
save_plots = true
billiard_polygon_check = false






# GEOMETRY 
eps = 0.5


# SOLVER PARAMS
d = 5.0
b = 10.0


# TENSION PARAMS
dk = 0.1
L1 = 10.0
L2 = 20.0
k0 = (L1 + L2)/2



# BENCHMARK PARAMS
k0_benchmark = 100.0
dk_benchmark = 0.5





if !isdir("Robnik")
    mkdir("Robnik")
end

if !isdir("Robnik/Robnik_Wavefunctions")
    mkdir("Robnik/Robnik_Wavefunctions")
end

if !isdir("Robnik/Robnik_Husimi")
    mkdir("Robnik/Robnik_Husimi")
end

if !isdir("Robnik/Robnik_Momentum")
    mkdir("Robnik/Robnik_Momentum")
end

if !isdir("Robnik/Robnik_Spectra")
    mkdir("Robnik/Robnik_Spectra")
end

if !isdir("Robnik/Robnik_Eigenvalues")
    mkdir("Robnik/Robnik_Eigenvalues")
end







########## GEOMETRY AND BASIS #################









billiard, basis = make_robnik_and_basis(eps)

fig = Figure()
ax = Axis(fig[1, 1], aspect = DataAspect())
plot_boundary!(ax, billiard)
save("Robnik/robnik_boundary.png", fig)


acc_solverA = ScalingMethodA(d,b)
dm_solver = DecompositionMethod(d,b)
psm_solver = ParticularSolutionsMethod(d,b,b)
acc_solver = acc_solverA
println("Constructed solvers")

# Testing the first 4 Corner Adapted Fourier Bessel functions
f = Figure(resolution = (1000,1000))
plot_basis_test!(f[1,1], basis, billiard; i=1) 
plot_basis_test!(f[1,2], basis, billiard; i=2)
plot_basis_test!(f[1,3], basis, billiard; i=3)
plot_basis_test!(f[2,1], basis, billiard; i=4) 
plot_basis_test!(f[2,2], basis, billiard; i=5)
plot_basis_test!(f[2,3], basis, billiard; i=6)
plot_basis_test!(f[3,1], basis, billiard; i=7) 
plot_basis_test!(f[3,2], basis, billiard; i=8)
plot_basis_test!(f[3,3], basis, billiard; i=9)
save("Robnik/robnik_basis_test.png", f)









############## benchmark_solver - testing purposes ###############








if do_benchmark






println("Doing benchmarks")

open("Robnik/robnik_solver_logs.txt", "w") do io
    redirect_stdout(io)
    

acc_info = benchmark_solver(acc_solver, basis, billiard, k0_benchmark, dk_benchmark; plot_matrix=true);
sleep(10)
dm_info = benchmark_solver(dm_solver, basis, billiard, k0_benchmark, dk_benchmark; plot_matrix=true, log=false);
sleep(10)
psm_info = benchmark_solver(psm_solver, basis, billiard, k0_benchmark, dk_benchmark; plot_matrix=true);
sleep(10)

println(acc_info)
println(dm_info)
println(psm_info)

try
    f = Figure(resolution = (1000,500))
    plot_solver_test!(f,acc_solver,basis,billiard,k0_benchmark,k0+1,0.1)
    save("Robnik/robnik_acc_solver_test.png", f)
catch e
    println(e)
end

try
    f = Figure(resolution = (1000,500))
    plot_solver_test!(f,dm_solver,basis,billiard,k0_benchmark,k0+1,0.1)
    save("Robnik/robnik_dm_solver_test.png", f)
catch e
    println(e)
end

try
    f = Figure(resolution = (1000,500))
    plot_solver_test!(f,psm_solver,basis,billiard,k0_benchmark,k0+1,0.01)
    save("Robnik/robnik_dm_solver_test.png", f)
catch e
    println(e)
end



# Find the eigenvalue and the eigenfunction
f_dm = Figure()
f_psm = Figure()

try
    k_acc, ten_acc = solve_wavenumber(acc_solver, basis, billiard, k0_benchmark, dk_benchmark)
    state_acc = compute_eigenstate(acc_solver, basis, billiard, k_acc)
    f_acc = Figure()
    plot_wavefunction!(f_acc, state_acc, inside_only=false)
    save("Robnik/acc_test_state.png", f_acc)
catch e
    println(e)
end

try
    k_dm, ten_dm = solve_wavenumber(dm_solver, basis, billiard, k0_benchmark, dk_benchmark)
    state_dm = compute_eigenstate(dm_solver, basis, billiard, k_dm)
    plot_wavefunction!(f_dm, state_dm, inside_only=false)
    save("Robnik/dm_test_state.png", f_dm) 
catch e
    println(e)
end



        
end

println("Finished doing benchmarks")

end

    
    
    
    
    
    
################ tensions ##################









if do_spectrum


k_dm, ten_dm = solve_wavenumber(dm_solver, basis, billiard, k0, dk)
k_psm, ten_psm = solve_wavenumber(psm_solver, basis, billiard, k0, dk)

k_range = collect(range(L1,L2,step=0.005))
tens_dm = k_sweep(dm_solver,basis,billiard,k_range)
tens_psm = k_sweep(psm_solver,basis,billiard,k_range)
ks, tens, control = compute_spectrum(acc_solver, basis, billiard, k_range[1], k_range[end], dk)

f = Figure()
ax = Axis(f[1,1], title="Robnik_$(nameof(typeof(basis)))")
lines!(ax,k_range,log10.(tens_dm), color=:red)
lines!(ax,k_range,log10.(tens_psm), color=:green)
scatter!(ax,ks,log10.(tens), color=:black, markersize=8)
scatter!(ax,k_dm,log10.(ten_dm))
scatter!(ax,k_psm,log10.(ten_psm))
display(f)
save("Robnik/Robnik_Spectra/robnik_spectrum_$(L1)_$(L2).png", f)
save_numerical_ks_and_tensions!(ks, tens, "Robnik/Robnik_Eigenvalues/robnik_eigenvalues_$(L1)_$(L2).csv")


end
    
    
    





############## SAVE THE EIGENSTATES ################
    
    
    
    
    
    
    
    
if save_plots 


ks, _ = read_numerical_ks_and_tensions("Robnik/Robnik_Eigenvalues/robnik_eigenvalues_$(L1)_$(L2).csv")


for i in eachindex(ks) 
    println("Saving wavefunction & husimi: ", i)
    f_probability = Figure()
    f_probability_full = Figure()
    f_wavefunction = Figure()
    f_wavefunction_full = Figure()
    f_husimi = Figure()
    f_momentum = Figure()
    try
        state = compute_eigenstate(acc_solver, basis, billiard, ks[i])
        plot_probability!(f_probability, state, inside_only=true)
        plot_probability!(f_probability_full, state, inside_only=true, fundamental_domain=false)
        plot_wavefunction!(f_wavefunction, state, inside_only=true)
        plot_wavefunction!(f_wavefunction_full, state, inside_only=true, fundamental_domain=false)
        plot_husimi_function!(f_husimi, state)
        plot_momentum_function!(f_momentum, state)
        save("Robnik/Robnik_Wavefunctions/$(ks[i])_probability.png", f_probability)
        save("Robnik/Robnik_Wavefunctions/$(ks[i])_probability_full.png", f_probability_full)
        save("Robnik/Robnik_Wavefunctions/$(ks[i])_wavefunction.png", f_wavefunction)
        save("Robnik/Robnik_Wavefunctions/$(ks[i])_wavefunction_full.png", f_wavefunction_full)
        save("Robnik/Robnik_Husimi/$(ks[i])_husimi.png", f_husimi)
        save("Robnik/Robnik_Momentum/$(ks[i])_momentum.png", f_momentum)
    catch e # Can happen with state failing to correctly calculate
        if isa(e, MethodError)
            println("Skipping due to error in computing eigenstate for ks[", i, "]: ", e)
            continue
        else
            rethrow(e)
        end
    end
end


end





if billiard_polygon_check
    billiard_polygon_pts_fundamental = billiard_polygon(billiard, 512*512)
    billiard_polygon_pts_full = billiard_polygon(billiard, 512*512; fundamental_domain=false)
    fig = Figure(resolution = (800, 600))
    ax_fundamental = Axis(fig[1, 1], aspect = DataAspect())
    ax_full = Axis(fig[1, 2], aspect = DataAspect())
    for set in billiard_polygon_pts_fundamental
        xs = [p[1] for p in set]
        ys = [p[2] for p in set]
        scatter!(ax_fundamental, xs, ys, color=:blue, markersize=3)
    end
    for set in billiard_polygon_pts_full
        xs = [p[1] for p in set]
        ys = [p[2] for p in set]
        scatter!(ax_full, xs, ys, color=:blue, markersize=3)
    end
    wait(display(fig))
end
    
    
    
        

