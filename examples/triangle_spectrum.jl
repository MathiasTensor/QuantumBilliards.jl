using StaticArrays
using GLMakie
using QuantumBilliards





do_spectrum = true
save_plots = true
billiard_polygon_check = false




# SOLVER PARAMS
d = 3.0
b = 5.0


# TENSION PARAMS
dk = 0.1
L1 = 5.0
L2 = 10.0
k0 = (L1 + L2)/2





if !isdir("Triangle")
    mkdir("Triangle")
end

if !isdir("Triangle/Triangle_Wavefunctions")
    mkdir("Triangle/Triangle_Wavefunctions")
end

if !isdir("Triangle/Triangle_Husimi")
    mkdir("Triangle/Triangle_Husimi")
end

if !isdir("Triangle/Triangle_Momentum")
    mkdir("Triangle/Triangle_Momentum")
end

if !isdir("Triangle/Triangle_Eigenvalues")
    mkdir("Triangle/Triangle_Eigenvalues")
end

if !isdir("Triangle/Triangle_Spectra")
    mkdir("Triangle/Triangle_Spectra")
end







################### GEOMETRY #################







function make_triangle_and_basis(gamma,chi; edge_i=1)
    cor = Triangle(gamma,chi).corners
    x0,y0 = cor[mod1(edge_i+2,3)]
    re = [:Virtual, :Virtual, :Virtual]
    re[edge_i] = :Real 
    tr = Triangle(gamma,chi; curve_types = re, x0 = x0, y0 =y0)
    basis = CornerAdaptedFourierBessel(3, adapt_basis(tr,edge_i+2)...) 
    return tr, basis 
end

gamma = 2/3*pi #sqrt(2)/2 * pi
chi  = 2.0






########################







billiard, basis = make_triangle_and_basis(gamma, chi)

f = Figure(resolution = (1000,500))
plot_basis_test!(f[1,1], basis, billiard; i=1)
plot_basis_test!(f[1,2], basis, billiard; i=2)
plot_basis_test!(f[1,3], basis, billiard; i=3)
save("Triangle/triangle_basis.png", f)


dm_solver = DecompositionMethod(d,b)
psm_solver = ParticularSolutionsMethod(d,b,b)
acc_solver = ScalingMethodA(d, b)







##################### tensions



if do_spectrum



k_dm, ten_dm = solve_wavenumber(dm_solver, basis, billiard, k0, dk)
k_psm, ten_psm = solve_wavenumber(psm_solver, basis, billiard, k0, dk)

k_range = collect(range(L1,L2,step=0.005))
tens_dm = k_sweep(dm_solver,basis,billiard,k_range)
tens_psm = k_sweep(psm_solver,basis,billiard,k_range)
ks, tens, control = compute_spectrum(acc_solver, basis, billiard, k_range[1], k_range[end], dk)

f = Figure(resolution = (2000,2000))
ax = Axis(f[1,1], title="Rectangle_$(nameof(typeof(basis)))")
lines!(ax,k_range,log10.(tens_dm), color=:red)
lines!(ax,k_range,log10.(tens_psm), color=:green)
scatter!(ax,ks,log10.(tens), color=:black, markersize=8)
scatter!(ax,k_dm,log10.(ten_dm))
scatter!(ax,k_psm,log10.(ten_psm))
vlines!(ax, ks, linewidth=0.2)

display(f)
save("Triangle/Triangle_Spectra/$(nameof(typeof(basis)))_k1_$(L1)_k2_$(L2).png", f)
QuantumBilliards.save_numerical_ks_and_tensions!(ks, tens, "Triangle/Triangle_Eigenvalues/triangle_eigenvalues_and_tensions.csv")





end




########## SAVE THE EIGENSTATES #############


if save_plots




ks, _ = QuantumBilliards.read_numerical_ks_and_tensions("Triangle/Triangle_Eigenvalues/triangle_eigenvalues_and_tensions.csv")



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
        save("Triangle/Triangle_Wavefunctions/$(ks[i])_probability.png", f_probability)
        save("Triangle/Triangle_Wavefunctions/$(ks[i])_probability_full.png", f_probability_full)
        save("Triangle/Triangle_Wavefunctions/$(ks[i])_wavefunction.png", f_wavefunction)
        save("Triangle/Triangle_Wavefunctions/$(ks[i])_wavefunction_full.png", f_wavefunction_full)
        save("Triangle/Triangle_Husimi/$(ks[i])_husimi.png", f_husimi)
        save("Triangle/Triangle_Momentum/$(ks[i])_momentum.png", f_momentum)
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



############# BILLIARD POLYGON CHECK #############




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