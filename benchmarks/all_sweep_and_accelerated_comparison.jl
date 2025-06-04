using QuantumBilliards
using DynamicalBilliards
using LinearAlgebra
using CairoMakie
using Statistics
using CSV
using DataFrames

#=
Comparison of different methods for billiards and different geometries.

#### Geometries:
- CircleBilliard
- MushroomBilliard
- StadiumBilliard
- RectangleBilliard
- ProsenBilliard
- RobnikBilliard
- EllipseBilliard
- EquilateralTriangleBilliard
- GeneralizedSinaiBilliard

#### Methods:
- DecompositionMethod
- ParticularSolutionsMethod
- BoundaryIntegralMethod
- ExpandedBoundaryIntegralMethod
- ScalingMethodA

The user can just change the different geometries in the "spectrum_test" function below. Some care needs to be taken when doing the symmetries when applicable, especially for the BoundaryIntegralMethod and ExpandedBoundaryIntegralMethod, as these methods are sensitive to the symmetries of the billiard.

# For example the rectangle has XYReflection symmetry and therefore these options corresponding to Dirichlet (:D) and Neumann (:N) boundary conditions based on the signed integer used in the XYReflection constructor. For other geometries the user can inspect how the geometries are constructed in their corresponding files in the "billiards" folder.

if x==:D && y==:D
    symmetry = Vector{Any}([XYReflection(-1,-1)])
elseif x==:D && y==:N
    symmetry = Vector{Any}([XYReflection(-1,1)])
elseif x==:N && y==:D
    symmetry = Vector{Any}([XYReflection(1,-1)])
elseif x==:N && y==:N
    symmetry = Vector{Any}([XYReflection(1,1)])
end

=#

#=
k1: Minimum wavenumber to start the sweep.
k2: Maximum wavenumber to end the sweep.
make_movie: Boolean flag to indicate whether to create a movie of the Fredholm evolution. Defaults to false since for lartge k1->k2 intervals it leads to weird rescaling of the matrix for the heatmap to correctly plot. If the k1->k2 interval is small, then the created movie makes sense.
b: Billiard size parameter, used in the BoundaryIntegralMethod and ExpandedBoundaryIntegralMethod. Typically it should be chosen from 10.0 to 15.0
x: Boundary condition in x direction, can be :D (Dirichlet) or :N (Neumann).
y: Boundary condition in y direction, can be :D (Dirichlet) or :N (Neumann).
type: Type of symmetry to use, can be :xy (XYReflection), :x (XReflection), :y (YReflection).
sampler_bim: Sampler for the BoundaryIntegralMethod and ExpandedBoundaryIntegralMethod, typically LinearNodes(), but the user can try playing around with PolarSampler.
sampler_other: Sampler for the other methods, typically GaussLegendreNodes() or LinearNodes().
dk: Function that defines the step size for the EBIM sweep, typically dk=(k)->(0.05*k^(-1/3)).
save_wavefunctions: Boolean flag to indicate whether to save the wavefunctions and Husimi plots. Defaults to false.
do_standard: Boolean flag to indicate whether to compute the standard methods (DecompositionMethod and ParticularSolutionsMethod). Defaults to true. For larger k1->k2 intervals it is better to set this to false, as the standard methods are very slow compared to Accelerated Methods and take too long.
do_VS: Boolean flag to indicate whether to compute the Vergini-Saraceno method. Defaults to true. Should be set to false if the geometry is starting to have convex problems.
do_ebim: Boolean flag to indicate whether to compute the EBIM method. Defaults to true.
raw_lapack: Boolean flag to indicate whether to use the raw LAPACK solver for the EBIM method. Defaults to false, as it is not needed for most cases. It just chooses if one needs 2 independant eigen calls for EBIM or just one wrapper as LAPACK.ggev3 which gives us both left and right eigenvectors.
plot_ebim_debug: Boolean flag to indicate whether to plot the EBIM debug information. Defaults to false. If one wants to see how close to the eigenvalue for small dk steps the differences of potential dks between successive solves of the Fredholm matrix converge to the true eigenvalue. Useful also if one suspects that the EBIM method is not getting us an eigenvalue even though we have obvious convergence of succesive solves.
fundamental: Boolean flag to indicate whether to compute the fundamental wavefunctions. Defaults to false.
do_husimi: Boolean flag to indicate whether to compute and plot the Husimi plots. Defaults to true.
=#
function spectrum_test(k1,k2;make_movie=false,b=10.0,x=:D,y=:D,type=:xy,sampler_bim=[LinearNodes()],sampler_other=[GaussLegendreNodes()],dk=(k)->(0.05*k^(-1/3)),save_wavefunctions=false,do_standard=true,do_VS=true,raw_lapack=false,do_ebim=true,fundamental=false,do_husimi=true,plot_ebim_debug=false)
    #billiard,basis=make_circle_and_basis(1.0)
    #billiard,basis=make_stadium_and_basis(1.0)
    #billiard,basis=make_prosen_and_basis(0.6)
    #billiard,basis=make_ellipse_and_basis(2.0,1.0)
    billiard,basis=make_robnik_and_basis(0.15)
    #billiard,basis=make_rectangle_and_basis(2.0, 1.0)
    #billiard,basis=make_equilateral_triangle_and_basis(1.0)
    #billiard,basis=make_generalized_sinai_and_basis()
    #billiard,basis=make_mushroom_and_basis(0.5,1.0,1.0)
    dm_solver=DecompositionMethod(5.0,8.0,sampler_other)
    psm_solver=ParticularSolutionsMethod(5.0,8.0,8.0,sampler_other)
    acc_solver=ScalingMethodA(5.0,8.0,sampler_other)
    if type==:xy
        if x==:D && y==:D
            symmetry=Vector{Any}([XYReflection(-1,-1)])
        elseif x==:D && y==:N
            symmetry=Vector{Any}([XYReflection(-1,1)])
        elseif x==:N && y==:D
            symmetry=Vector{Any}([XYReflection(1,-1)])
        elseif x==:N && y==:N
            symmetry=Vector{Any}([XYReflection(1,1)])
        end
    elseif type==:x
        if x==:D
            symmetry=Vector{Any}([XReflection(-1)])
        elseif x==:N
            symmetry=Vector{Any}([XReflection(1)])
        end
    elseif type==:y
        if y==:D
            symmetry=Vector{Any}([YReflection(-1)])
        elseif y==:N
            symmetry=Vector{Any}([YReflection(1)])
        end
    elseif type==:rot # NOT IMPLEMENTED
        if x==:D
            symmetry=Vector{Any}([QuantumBilliards.Rotation(3,-1)])
        elseif x==:N
            symmetry=Vector{Any}([QuantumBilliards.Rotation(3,1)])
        end
    else
        symmetry=nothing
    end

    bim_solver=QuantumBilliards.BoundaryIntegralMethod(b,sampler_bim,billiard;symmetries=symmetry,x_bc=x,y_bc=y)
    println("bim solver symmetry: ",bim_solver.rule)
    ebim_solver=QuantumBilliards.ExpandedBoundaryIntegralMethod(b,sampler_bim,billiard;symmetries=symmetry,x_bc=x,y_bc=y)
    symmetryBIM=QuantumBilliards.SymmetryRuleBIM(billiard;symmetries=symmetry,x_bc=x,y_bc=y)
    println("Symmetry type: ",symmetryBIM.symmetry_type)
    k_range=collect(range(k1,k2,step=0.0005))
    k=(k1+k2)/2
    pts=QuantumBilliards.evaluate_points(bim_solver,billiard,k)
    hankel_basis=QuantumBilliards.AbstractHankelBasis()
    if do_ebim
        # checks for indexes of singular distance behaviour in bim and ebim matrices (any off diagonal elements)
        #QuantumBilliards.distance_singular_check!(ebim_solver,billiard,k,num_print_idxs=10)
        @time ks_ebim,tens_ebim=compute_spectrum(ebim_solver,billiard,k1,k2,dk=dk,use_lapack_raw=raw_lapack)
        #@time ks_ebim,tens_ebim=QuantumBilliards.compute_spectrum_new(ebim_solver,billiard,k1,k2,dk=dk,use_lapack_raw=raw_lapack)
        @time ks_debug1,tens_debug1,ks_debug2,tens_debug2=QuantumBilliards.visualize_ebim_sweep(ebim_solver,hankel_basis,billiard,k1,k2;dk=dk)
    end
    @time tens_bim=k_sweep(bim_solver,hankel_basis,billiard,k_range)
    if do_standard
        @time tens_dm=k_sweep(dm_solver,basis,billiard,k_range,multithreaded_matrices=true,multithreaded_ks=true)
        @time tens_psm=k_sweep(psm_solver,basis,billiard,k_range,multithreaded_matrices=true,multithreaded_ks=true)
    end
    if do_VS
        @time ks_VS,tens_VS=compute_spectrum(acc_solver,basis,billiard,k1,k2,0.1)
        idxs=findall(x->x<1e-3,tens_VS) # all the correct eigenvalues
        ks_VS=ks_VS[idxs]
        tens_VS=tens_VS[idxs]
    end
    f=Figure(size=(3500,2000),resolution=(3500,2000))
    ax=Axis(f[1,1],title="$(nameof(typeof(billiard)))_$(nameof(typeof(basis)))_$(nameof(typeof(sampler_bim[1])))_$(x)$(y)",ylabel=L"log(f_μ(k))", xlabel=L"k")
    lines!(ax,k_range,log10.(tens_bim),color=:blue)
    if do_standard
        lines!(ax,k_range,log10.(tens_dm),color=:red)
        lines!(ax,k_range,log10.(tens_psm),color=:green)
    end
    if do_VS
        scatter!(ax,ks_VS,log10.(tens_VS),color=:orange)
    end
    if do_ebim
        scatter!(ax,ks_ebim,log10.(tens_ebim),color=:black)
    end
    xlims!(ax,k1,k2)
    # process the gradient
    mid_x,gradient=QuantumBilliards.bim_second_derivative(k_range,log10.(tens_bim))
    # these are the ks
    ks=QuantumBilliards.get_eigenvalues(k_range,tens_bim) # for easier logic
    #vlines!(ax,peaks,linewidth=0.5,color=:red)
    vlines!(ax,ks,linewidth=0.5,color=:red)
    ax=Axis(f[2,1],title="$(nameof(typeof(billiard)))_$(nameof(typeof(basis)))_$(nameof(typeof(sampler_bim[1])))_$(x)$(y)_2nd_GRADIENT",ylabel=L"∂/∂k(log(f_μ(k)))",xlabel=L"k")
    lines!(ax,mid_x,gradient,color=:green)
    vlines!(ax,ks,linewidth=0.5)
    xlims!(ax,k1,k2)

    if do_ebim # checking the groupings of the ks via ther diff(ks) from ebim. As we get closer to the correct solution the diffs become much smaller and serve as a marker for the correct eigenvalues
        ax=Axis(f[3,1],title="EBIM inverse peaks - 1st&2nd order correction",ylabel=L"log(1/diff(λs))",xlabel=L"k")
        scatter!(ax,ks_debug1,log10.(tens_debug1),color=:red,marker=:xcross,label="1st ord. corr.")
        scatter!(ax,ks_debug2,log10.(tens_debug2),color=:green,marker=:utriangle,label="2nd ord. corr.")
        lines!(ax,k_range,log10.(1.0./dk.(k_range)),color=:blue,label="max. dk interval") # to see the bottom line for observation
        peaks_ebim=QuantumBilliards.find_peaks(ks_debug1,log10.(tens_debug1);threshold=2.0*log10(1.0/dk(k2)))
        vlines!(ax,peaks_ebim,linewidth=0.5,color=:red,label="2nd order corr. peaks")
        xlims!(ax,k1,k2)
        axislegend()
    end
    if do_ebim && do_VS && !isempty(ks_VS)
        function find_closest(ks_VS,ks_ebim)
            closest_values=Vector{Float64}(undef,length(ks_ebim))
            for i in eachindex(ks_ebim)
                closest_index=argmin(abs.(ks_VS.-ks_ebim[i]))
                closest_values[i]=ks_VS[closest_index]
            end
            return closest_values
        end
        closest_ks_VS=find_closest(ks_VS,ks_ebim)
        println("Comparison between EBIM and VS...")
        for i in eachindex(ks_ebim)
            println("k EBIM: ",ks_ebim[i], " k VS: ",closest_ks_VS[i])
        end
    end
    save("$(nameof(typeof(billiard)))_spectrum_$(k1)_$(k2)_$(nameof(typeof(sampler_bim[1])))_b_$(b)_x_$(x)_y_$(y).png", f)
    f=Figure(resolution=(4000,3000),size=(4000,3000))  # Define a 6x5 grid with proper resolution
    r_max,c_max=6,5  # Number of rows and columns in the grid
    r=1
    c=1
    for k in ks[1:min(length(ks),r_max*c_max)] # ince bim gives useful info we can check the derivatives of the feedholm matrices as a gauge of the EBIM 
        # Axis for fredholm_matrix
        ax1=Axis(f[r,c],title="k=$k Fredholm",aspect=DataAspect(),width=500,height=500)
        heatmap!(ax1,log10.(abs.(QuantumBilliards.fredholm_matrix(QuantumBilliards.evaluate_points(bim_solver,billiard, k),symmetryBIM,k))))
        # Axis for fredholm_matrix_derivative
        ax2=Axis(f[r,c+1],title="dA/dk k=$k",aspect=DataAspect(),width=500,height=500)
        heatmap!(ax2,log10.(abs.(QuantumBilliards.fredholm_matrix_der(QuantumBilliards.evaluate_points(bim_solver,billiard,k),symmetryBIM,k, kernel_fun=:first))))
        # Axis for fredholm_matrix_second_derivative
        ax3=Axis(f[r,c+2],title="d^2A/dk^2 k=$k",aspect=DataAspect(),width=500,height=500)
        heatmap!(ax3,log10.(abs.(QuantumBilliards.fredholm_matrix_der(QuantumBilliards.evaluate_points(bim_solver,billiard,k),symmetryBIM,k,kernel_fun=:second))))
        if c>6
            c=1
            r+=1
            continue
        end
        c+=3
    end
    save("$(nameof(typeof(billiard)))_A_dA_ddA_$(k1)_$(k2)_$(nameof(typeof(sampler_bim[1])))_b_$(b)_x_$(x)_y_$(y).png", f)
    # save wavefunctions and husimi
    if save_wavefunctions
        if do_ebim
            ks=ks_ebim
        end
        us_all,pts_all=QuantumBilliards.solve_eigenvectors_BIM(bim_solver,billiard,hankel_basis,ks)
        println("Length of us_all: ",[length(us) for us in us_all])
        println("Length of s all: ",[length(pts.ds) for pts in pts_all])
        pts_all,us_all=QuantumBilliards.boundary_function_BIM(bim_solver,us_all,pts_all,billiard) # pts_all is of type BoundaryPoints
        s_vals_all=[pts.s for pts in pts_all]
        if do_husimi
            @time Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list=QuantumBilliards.wavefunction_multi_with_husimi(ks,us_all,pts_all,billiard;fundamental=fundamental,inside_only=false,xgrid_size=1000,ygrid_size=500)
            @time f=QuantumBilliards.plot_wavefunctions_with_husimi_BATCH(ks,[abs.(Psi) for Psi in Psi2ds],x_grid,y_grid,Hs_list,ps_list,qs_list,billiard,us_all,s_vals_all,fundamental=fundamental)
        else
            @time Psi2ds,x_grid,y_grid=QuantumBilliards.wavefunction_multi(ks,us_all,pts_all,billiard;fundamental=fundamental,inside_only=false)
            @time f=QuantumBilliards.plot_wavefunctions_BATCH(ks,[abs.(Psi) for Psi in Psi2ds],x_grid,y_grid,billiard,fundamental=fundamental)
        end
        save("$(nameof(typeof(billiard)))wavefunctions_$(k1)_$(k2)_$(nameof(typeof(sampler_bim[1])))_b_$(b)_x_$(x)_y_$(y).png", f)
    end
    if make_movie
        QuantumBilliards.create_fredholm_movie!(k_range,billiard;symmetries=Vector{Any}([XYReflection(-1,-1)]),sampler=[GaussLegendreNodes()],output_path="$(nameof(typeof(billiard)))_fredholm_evol_$(k1)_$(k2)_$(nameof(typeof(sampler_bim[1]))).mp4")
    end
end

##############
#### MAIN ####
##############

# Here we choose all the relevant parameters for the sweep and the methods we want to test. Left to the user to change.
k1,k2=30.0,32.0
spectrum_test(k1,k2,make_movie=false,b=15.0,x=:D,y=:D,type=:y,sampler_bim=[LinearNodes()],dk=(k)->(0.04*k^(-1/3)),save_wavefunctions=true,do_standard=true,do_VS=true,do_ebim=true,fundamental=false,do_husimi=true,raw_lapack=true)
spectrum_test(k1,k2,make_movie=false,b=15.0,x=:D,y=:N,type=:y,sampler_bim=[LinearNodes()],dk=(k)->(0.04*k^(-1/3)),save_wavefunctions=true,do_standard=false,do_VS=false,do_ebim=true,fundamental=false,do_husimi=true,raw_lapack=true)
spectrum_test(k1,k2,make_movie=false,b=15.0,x=:N,y=:D,type=:y,sampler_bim=[LinearNodes()],dk=(k)->(0.04*k^(-1/3)),save_wavefunctions=true,do_standard=false,do_VS=false,do_ebim=true,fundamental=false,do_husimi=true,raw_lapack=true)
spectrum_test(k1,k2,make_movie=false,b=15.0,x=:N,y=:N,type=:y,sampler_bim=[LinearNodes()],dk=(k)->(0.04*k^(-1/3)),save_wavefunctions=true,do_standard=false,do_VS=false,do_ebim=true,fundamental=false,do_husimi=true,raw_lapack=true)