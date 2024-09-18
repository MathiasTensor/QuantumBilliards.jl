using StaticArrays
using GLMakie
using QuantumBilliards
using LinearAlgebra






do_benchmark = false
do_spectrum = true
save_plots = true






# GEOMETRY
w=Float64(pi/3)
h=1.0


# SOLVERS AND BASIS
d = 3.0
b = 5.0




dk = 0.1
L1 = 100.0
L2 = 110.0
k0 = (L1 + L2)/2








if !isdir("Rectangle")
    mkdir("Rectangle")
end

if !isdir("Rectangle/Rectangle_Wavefunctions")
    mkdir("Rectangle/Rectangle_Wavefunctions")
end

if !isdir("Rectangle/Rectangle_Husimi")
    mkdir("Rectangle/Rectangle_Husimi")
end

if !isdir("Rectangle/Rectangle_Spectra")
    mkdir("Rectangle/Rectangle_Spectra")
end

if !isdir("Rectangle/Rectangle_Eigenvalues")
    mkdir("Rectangle/Rectangle_Eigenvalues")
end

if !isdir("Rectangle/Rectangle_Momentum")
    mkdir("Rectangle/Rectangle_Momentum")
end










########### Analytical helpers ##############










function compute_rectangle_analytical_eigenvalues(k_min::Float64, k_max::Float64, width::Float64, height::Float64)
    analytical_eigenvalues = []
    m = 1
    while true
        stop_outer_loop = false
        n = 1
        while true
            # Compute k_mn^2 for the given m and n
            k_mn_squared = (m * π / width)^2 + (n * π / height)^2
            # Break the inner loop if k_mn^2 exceeds the max value
            if k_mn_squared > k_max^2
                if n == 1
                    stop_outer_loop = true
                end
                break
            end
            # Only store k_mn if it lies within the desired range
            if k_min^2 <= k_mn_squared
                k_mn = sqrt(k_mn_squared)
                push!(analytical_eigenvalues, (k_mn, m, n))  # Store k_mn along with m and n
            end
            n += 1
        end
        # Break the outer loop if the stop condition is met
        if stop_outer_loop
            break
        end
        m += 1
    end
    return analytical_eigenvalues
end

function compute_rectangle_analytical_eigenvaluesNN(k_min::Float64, k_max::Float64, width::Float64, height::Float64)
    base_eigenvalues = compute_rectangle_analytical_eigenvalues(k_min, k_max, width, height)
    filtered_eigenvalues = filter(e -> isodd(e[2]) && isodd(e[3]), base_eigenvalues)
    return filtered_eigenvalues
end

function compute_rectangle_analytical_eigenvaluesDD(k_min::Float64, k_max::Float64, width::Float64, height::Float64)
    base_eigenvalues = compute_rectangle_analytical_eigenvalues(k_min, k_max, width, height)
    filtered_eigenvalues = filter(e -> iseven(e[2]) && iseven(e[3]), base_eigenvalues)
    return filtered_eigenvalues
end










################### Make rectangle and basis










billiard, basis = make_rectangle_and_basis(w, h)


#=
billiard = RectangleBilliard(w, h)
symmetry = Vector{Any}([XYReflection(-1, -1)])
basis = RealPlaneWaves(10, symmetry; angle_arc=pi/2.0)
#basis = CornerAdaptedFourierBessel(3, pi/2, SVector(0.0, 0.0), 0.0)
println("Construted basis")
=#

acc_solverA = ScalingMethodA(d,b)
dm_solver = DecompositionMethod(d,b)
psm_solver = ParticularSolutionsMethod(d,b,b)
acc_solver = acc_solverA
println("Constructed solvers")

f = Figure()
ax = Axis(f[1,1])
plot_boundary!(ax, billiard)
save("Rectangle/rectangle_plot.png", f)

f = Figure(resolution = (1000,500))
plot_basis_test!(f[1,1], basis, billiard; i=1)
plot_basis_test!(f[1,2], basis, billiard; i=2)
plot_basis_test!(f[1,3], basis, billiard; i=3)
save("Rectangle/rectangle_$(nameof(typeof(basis)))_test.png", f)








#### tensions calculation




if do_spectrum
    





k_dm, ten_dm = solve_wavenumber(dm_solver, basis, billiard, k0, dk)
k_psm, ten_psm = solve_wavenumber(psm_solver, basis, billiard, k0, dk)
#k_acc, ten_acc = solve_wavenumber(acc_solver, basis, billiard, k0, dk) # This one does not work for the wavefunction construction

k_range = collect(range(L1,L2,step=0.005))
tens_dm = k_sweep(dm_solver,basis,billiard,k_range)
tens_psm = k_sweep(psm_solver,basis,billiard,k_range)
ks, tens, control = compute_spectrum(acc_solver, basis, billiard, k_range[1], k_range[end], dk)

analytical_eigenvalues = compute_rectangle_analytical_eigenvalues(L1,L2, w, h)
xs = [k for (k,_,_) in analytical_eigenvalues]
ys = fill(min(minimum(log10.(tens_dm)), minimum(log10.(tens_psm))), length(xs))

analytical_eigenvaluesDD = compute_rectangle_analytical_eigenvaluesDD(L1,L2, w, h)
xsDD = [k for (k,_,_) in analytical_eigenvaluesDD]
ysDD = fill(min(minimum(log10.(tens_dm)), minimum(log10.(tens_psm))), length(xsDD))

analytical_eigenvaluesNN = compute_rectangle_analytical_eigenvaluesNN(L1,L2, w, h)
xsNN = [k for (k,_,_) in analytical_eigenvaluesNN]
ysNN = fill(min(minimum(log10.(tens_dm)), minimum(log10.(tens_psm))), length(xsNN))

f = Figure(resolution = (2000,2000))
ax = Axis(f[1,1], title="Rectangle_$(nameof(typeof(basis)))")
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
save("Rectangle/Rectangle_Spectra/$(nameof(typeof(basis)))_k1_$(L1)_k2_$(L2).png", f)
compute_and_save_closest_pairs!(ks, xs, w*h, "Rectangle/Rectangle_Eigenvalues/rectangle_eigenvalue_comparison_unique_$(L1)_$(L2)"; unique=true)
compute_and_save_closest_pairs!(ks, xs, w*h, "Rectangle/Rectangle_Eigenvalues/rectangle_eigenvalue_comparison_$(L1)_$(L2)"; unique=false)
save_numerical_ks_and_tensions!(ks, tens, "Rectangle/Rectangle_Eigenvalues/rectangle_eigenvalues_$(L1)_$(L2).csv")


end



###### SAVE THE EIGENSTATES


if save_plots



ks, _ = read_numerical_ks_and_tensions("Rectangle/Rectangle_Eigenvalues/rectangle_eigenvalues_$(L1)_$(L2).csv")



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
        save("Rectangle/Rectangle_Wavefunctions/$(ks[i])_probability.png", f_probability)
        save("Rectangle/Rectangle_Wavefunctions/$(ks[i])_probability_full.png", f_probability_full)
        save("Rectangle/Rectangle_Wavefunctions/$(ks[i])_wavefunction.png", f_wavefunction)
        save("Rectangle/Rectangle_Wavefunctions/$(ks[i])_wavefunction_full.png", f_wavefunction_full)
        save("Rectangle/Rectangle_Husimi/$(ks[i])_husimi.png", f_husimi)
        save("Rectangle/Rectangle_Momentum/$(ks[i])_momentum.png", f_momentum)
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