using StaticArrays
using GLMakie
using QuantumBilliards
using LinearAlgebra




do_benchmark = false
do_spectrum = false
save_plots = true






# STADIUM PARAMS
eps = 0.8

# SOLVER PARAMS
d = 3.0
b = 5.0

# TENSION PARAMS
dk = 0.1
L1 = 0.5
L2 = 100.0
k0 = (L1 + L2)/2

# BENCHMARK PARAMS
k0_benchmark = 100.0
dk_benchmark = 0.5



if !isdir("Stadium")
    mkdir("Stadium")
end

if !isdir("Stadium/Stadium_Wavefunctions")
    mkdir("Stadium/Stadium_Wavefunctions")
end

if !isdir("Stadium/Stadium_Husimi")
    mkdir("Stadium/Stadium_Husimi")
end

if !isdir("Stadium/Stadium_Eigenvalues")
    mkdir("Stadium/Stadium_Eigenvalues")
end

if !isdir("Stadium/Stadium_Spectra")
    mkdir("Stadium/Stadium_Spectra")
end

if !isdir("Stadium/Stadium_Momentum")
    mkdir("Stadium/Stadium_Momentum")
end







############# CONSTRUCT GEOMETRY AND BASIS #################










function make_stadium_and_basis(half_width;radius=1.0,x0=zero(half_width),y0=zero(half_width), rot_angle=zero(half_width))
    billiard = Stadium(half_width; radius=radius,x0=x0,y0=y0)
    basis = CornerAdaptedFourierBessel(1, pi/2.0, SVector(x0,y0),rot_angle) 
    return billiard, basis 
end

billiard, basis = make_stadium_and_basis(eps)

fig = Figure()
ax = Axis(fig[1, 1], aspect = DataAspect())
plot_boundary!(ax, billiard)
save("Stadium/stadium_plot.png", fig)



acc_solverA = ScalingMethodA(d,b)
dm_solver = DecompositionMethod(d,b)
psm_solver = ParticularSolutionsMethod(d,b,b)
acc_solver = acc_solverA
println("Constructed solvers")
f = Figure(resolution = (1000,1000))
plot_basis_test!(f[1,1], basis, billiard; i=1) 
save("Stadium/stadium_basis_test.png", f)










############### benchmark_solver ###########








if do_benchmark


println("Doing benchmarks")


open("Stadium/stadium_solver_logs.txt", "w") do io
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

f_acc= Figure(resolution = (1000,500))
f_dm= Figure(resolution = (1000,500))
f_psm= Figure(resolution = (1000,500))

try
    plot_solver_test!(f_acc,acc_solver,basis,billiard,k0_benchmark,k0+1,0.1)
    save("Stadium/stadium_acc_solver_test.png", f)
catch e
    println(e)
end

try
    plot_solver_test!(f_dm,dm_solver,basis,billiard,k0_benchmark,k0+1,0.1)
    save("Stadium/stadium_dm_solver_test.png", f)
catch e
    println(e)
end

try
    plot_solver_test!(f_psm,psm_solver,basis,billiard,k0_benchmark,k0+1,0.01)
    save("Stadium/stadium_dm_solver_test.png", f)
catch e
    println(e)
end



# Find the eigenvalue and the eigenfunction
f_dm = Figure()
f_acc = Figure()

try
    k_acc, ten_acc = solve_wavenumber(acc_solver, basis, billiard, k0_benchmark, dk_benchmark)
    state_acc = compute_eigenstate(acc_solver, basis, billiard, k_acc)
    plot_wavefunction!(f_acc, state_acc, inside_only=false)
    save("Stadium/acc_test_state.png", f_acc)
catch e
    println(e)
end

try
    k_dm, ten_dm = solve_wavenumber(dm_solver, basis, billiard, k0_benchmark, dk_benchmark)
    state_dm = compute_eigenstate(dm_solver, basis, billiard, k_dm)
    plot_wavefunction!(f_dm, state_dm, inside_only=false)
    save("Stadium/dm_test_state.png", f_dm) 
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
ax = Axis(f[1,1], title="Mushroom_$(nameof(typeof(basis)))")
lines!(ax,k_range,log10.(tens_dm), color=:red)
lines!(ax,k_range,log10.(tens_psm), color=:green)
scatter!(ax,ks,log10.(tens), color=:black, markersize=8)
scatter!(ax,k_dm,log10.(ten_dm))
scatter!(ax,k_psm,log10.(ten_psm))
display(f)
save("Stadium/Stadium_Spectra/stadium_spectrum_$(L1)_$(L2).png", f)
save_numerical_ks_and_tensions!(ks, tens, "Stadium/Stadium_Eigenvalues/stadium_eigenvalues.csv")



end




############## SAVE THE EIGENSTATES ################
    
    
    
    
    
    
    
    
if save_plots 

println("Saving wavefunctions and husimi functions...")

ks, _ = read_numerical_ks_and_tensions("Stadium/Stadium_Eigenvalues/stadium_eigenvalues.csv")
filter!(x -> x>3.0, ks)

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
        save("Stadium/Stadium_Wavefunctions/$(ks[i])_probability.png", f_probability)
        save("Stadium/Stadium_Wavefunctions/$(ks[i])_probability_full.png", f_probability_full)
        save("Stadium/Stadium_Wavefunctions/$(ks[i])_wavefunction.png", f_wavefunction)
        save("Stadium/Stadium_Wavefunctions/$(ks[i])_wavefunction_full.png", f_wavefunction_full)
        save("Stadium/Stadium_Husimi/$(ks[i])_husimi.png", f_husimi)
        save("Stadium/Stadium_Momentum/$(ks[i])_momentum.png", f_momentum)
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

        
        
        
    
        
    
    