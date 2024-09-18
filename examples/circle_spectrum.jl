using StaticArrays
using GLMakie
using QuantumBilliards
using FunctionZeros
using LinearAlgebra










do_spectrum = true
save_plots = true
billiard_polygon_check = false


# GEOMETRY
radius = 1.0


# SOLVER PARAMS
d = 3.0
b = 5.0


# TENSION PARAMS
dk = 0.1
L1 = 10.0
L2 = 20.0
k0 = (L1 + L2)/2








if !isdir("Circle")
    mkdir("Circle")
end

if !isdir("Circle/Circle_Wavefunctions")
    mkdir("Circle/Circle_Wavefunctions")
end

if !isdir("Circle/Circle_Husimi")
    mkdir("Circle/Circle_Husimi")
end

if !isdir("Circle/Circle_Momentum")
    mkdir("Circle/Circle_Momentum")
end

if !isdir("Circle/Circle_Eigenvalues")
    mkdir("Circle/Circle_Eigenvalues")
end

if !isdir("Circle/Circle_Spectra")
    mkdir("Circle/Circle_Spectra")
end









##################### HELPERS










function compute_circle_analytical_eigenvalues(k_min::Float64, k_max::Float64, radius::Float64)
    analytical_eigenvalues = []
    nu = 0
    while true
        stop_outer_loop = false
        zero_index = 1
        while true
            # Get the nth zero of the Bessel function of order nu
            zero = besselj_zero(nu, zero_index)
            # Calculate the eigenvalue k_mn
            k_mn = zero / radius
            if k_mn > k_max
                # If we exceed k_max, we break out of this loop
                if zero_index == 1
                    stop_outer_loop = true
                end
                break
            end
            if k_mn >= k_min
                # Append the eigenvalue, nu, and zero_index as a tuple
                push!(analytical_eigenvalues, (k_mn, nu, zero_index))
                # If nu > 0, add the negative nu case for doublets
                if nu > 0
                    push!(analytical_eigenvalues, (k_mn, -nu, zero_index))
                end
            end
            zero_index += 1
        end
        if stop_outer_loop
            break
        end
        nu += 1
    end
    return analytical_eigenvalues
end

function compute_circle_analytical_eigenvaluesNN(k_min::Float64, k_max::Float64, radius::Float64)
    base_eigenvalues = compute_circle_analytical_eigenvalues(k_min, k_max, radius)
    filtered_eigenvalues = filter(e -> isodd(abs(e[2])), base_eigenvalues)
    return filtered_eigenvalues
end

function compute_circle_analytical_eigenvaluesDD(k_min::Float64, k_max::Float64, radius::Float64)
    base_eigenvalues = compute_circle_analytical_eigenvalues(k_min, k_max, radius)
    filtered_eigenvalues = filter(e -> iseven(abs(e[2])), base_eigenvalues)
    return filtered_eigenvalues
end







#######################      BASIS AND SOLVERS      #############################








# Create a CircleBilliard instance and plot it
billiard, basis = make_circle_and_basis(radius)
symmetry = Vector{Any}([XYReflection(-1, -1)])
#basis = CornerAdaptedFourierBessel(3, Float64(pi/2.0), SVector(0.0,0.0), Float64(0.0)) 
basis = RealPlaneWaves(10, symmetry; angle_arc=Float64(pi/2))

acc_solverA = ScalingMethodA(d,b)
dm_solver = DecompositionMethod(d,b)
psm_solver = ParticularSolutionsMethod(d,b,b)
acc_solver = acc_solverA


fig = Figure()
ax = Axis(fig[1,1])
plot_boundary!(ax, billiard)
save("Circle/circle_plot.png", fig)
fig






##################### tensions ######################







if do_spectrum




k_dm, ten_dm = solve_wavenumber(dm_solver, basis, billiard, k0, dk)
k_psm, ten_psm = solve_wavenumber(psm_solver, basis, billiard, k0, dk)
#k_acc, ten_acc = solve_wavenumber(acc_solver, basis, billiard, k0, dk) # This one does not work

k_range = collect(range(L1,L2,step=0.005))
tens_dm = k_sweep(dm_solver,basis,billiard,k_range)
tens_psm = k_sweep(psm_solver,basis,billiard,k_range)
#ks, tens, control = compute_spectrum(acc_solver, basis, billiard, k_range[1], k_range[end], dk) #OLD ONE
ks, tens, control = compute_spectrum(acc_solver, basis, billiard, k_range[1], k_range[end]; N_expect=2) #NEW

analytical_eigenvalues = compute_circle_analytical_eigenvalues(L1,5*L2, radius)
xs = [k for (k, _, _) in analytical_eigenvalues if (k > L1 && k < L2)]
ys = fill(min(minimum(log10.(tens_dm)), minimum(log10.(tens_psm))), length(xs))

analytical_eigenvaluesDD = compute_circle_analytical_eigenvaluesDD(L1,5*L2, radius)
xsDD = [k for (k, _, _) in analytical_eigenvaluesDD if (k > L1 && k < L2)]
ysDD = fill(min(minimum(log10.(tens_dm)), minimum(log10.(tens_psm))), length(xsDD))

analytical_eigenvaluesNN = compute_circle_analytical_eigenvaluesNN(L1,5*L2, radius)
xsNN = [k for (k, _, _) in analytical_eigenvaluesNN if (k > L1 && k < L2)]
ysNN = fill(min(minimum(log10.(tens_dm)), minimum(log10.(tens_psm))), length(xsNN))

f = Figure(resolution = (2000,2000))
ax = Axis(f[1,1], title="Circle_$(nameof(typeof(basis)))")
lines!(ax,k_range,log10.(tens_dm), color=:red)
lines!(ax,k_range,log10.(tens_psm), color=:green)
scatter!(ax,ks,log10.(tens), color=:black, markersize=8)
scatter!(ax,k_dm,log10.(ten_dm))
scatter!(ax,k_psm,log10.(ten_psm))
vlines!(ax, ks, linewidth=0.2)
scatter!(ax, xs, ys, markersize=8, color=:red, marker=:xcross)
scatter!(ax, xsDD, ysDD, markersize=8, color=:blue, marker=:xcross)
scatter!(ax, xsNN, ysNN, markersize=8, color=:green, marker=:diamond)

display(f)
save("Circle/Circle_Spectra/$(nameof(typeof(basis)))_k1_$(L1)_k2_$(L2).png", f)
compute_and_save_closest_pairs!(ks, xs, Float64(pi), "Circle/Circle_Eigenvalues/circle_eigenvalues_comparison_unique_$(L1)_$(L2)"; unique=true)
compute_and_save_closest_pairs!(ks, xs, Float64(pi), "Circle/Circle_Eigenvalues/circle_eigenvalues_comparison_$(L1)_$(L2)"; unique=false)
save_numerical_ks_and_tensions!(ks, tens, "Circle/Circle_Eigenvalues/circle_eigenvalues_$(L1)_$(L2).csv")




end




######## SAVE THE EIGENSTATES ##########


if save_plots



ks, _ = read_numerical_ks_and_tensions("Circle/Circle_Eigenvalues/circle_eigenvalues_$(L1)_$(L2).csv")



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
        save("Circle/Circle_Wavefunctions/$(ks[i])_probability.png", f_probability)
        save("Circle/Circle_Wavefunctions/$(ks[i])_probability_full.png", f_probability_full)
        save("Circle/Circle_Wavefunctions/$(ks[i])_wavefunction.png", f_wavefunction)
        save("Circle/Circle_Wavefunctions/$(ks[i])_wavefunction_full.png", f_wavefunction_full)
        save("Circle/Circle_Husimi/$(ks[i])_husimi.png", f_husimi)
        save("Circle/Circle_Momentum/$(ks[i])_momentum.png", f_momentum)
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


