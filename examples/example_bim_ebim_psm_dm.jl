using QuantumBilliards
using LinearAlgebra
using Printf
using CairoMakie
using ProgressMeter

try_MKL_on_x86_64!()

################## USER PARAMS ###################

#geometry=:star 
geometry=:prosen
#geometry=:stadium 
#geometry=:ellipse

# choose the symmetry (for BoundaryIntegralMethod,CFIE_kress,DLP_kress)
#symmetry=nothing
#symmetry=Rotation(5,0) # needs to have same order as the star geometry below
symmetry=XYReflection(-1,-1) # for stadium this full desymmetrization

#method=:bim_sweep # choose one method family here: 
#method=:psm #- ParticularSolutionsMethod, can do sweeps
#method=:dm #- DecompositionMethod, can do sweeps
method=:ebim_bim # will do plain BIM and EBIM
#method=:ebim_dlp_kress # will do Kress DLP and EBIM with Kress DLP kernel

k1=20.0 # left endpoint of the spectral window to scan
k2=22.0 # right endpoint of the spectral window to scan
step=1e-3
kgrid=collect(k1:step:k2) # explicit k grid to use for the EBIM scan; if not provided, a default adaptive grid will be used based on the local density of states and the dk function provided below
d=10.0 # basis dimension scaling factor for PSM, DM (and VerginiSaraceno method - not used here).
b=15.0 # collocation point scaling factor (points per wavelength) N = ceil(Int, b*k*L/(2*pi)) where L is the boundary length.
plot_wavef=true # whether to reconstruct and plot the wavefunctions

#################################################################

function make_geometry_and_basis(geometry)
    if geometry==:star
        return make_star_and_basis(5) # 5 pointed star
    elseif geometry==:prosen
        return make_prosen_and_basis(0.1,basis_type=:cafb) # For curved non-convex geometries basis type methods
        # quickly fail. Increasing the prosen parameter 0.1 -> 0.15 -> 0.2 completely break the PSM/DM/VerginiSaraceno
    elseif geometry==:stadium
        return make_stadium_and_basis(1.0) 
    elseif geometry==:ellipse
        return make_ellipse_and_basis(1.0,0.5)
    else
        error("Make your own!") 
    end
end

billiard,basis=make_geometry_and_basis(geometry) # geometry object for all methods, plus constructor-provided basis for PSM / DM / naive BIM sweeps

if method==:psm
    solver=ParticularSolutionsMethod(d,b,b) # particular-solutions sweep solver using the constructor-provided basis
elseif method==:dm
    solver=DecompositionMethod(d,b) # decomposition-method sweep solver on the same smooth geometry
elseif method==:bim_sweep
    solver=BoundaryIntegralMethod(b,billiard,symmetry=symmetry) # plain naive BIM sweep solver, not EBIM-corrected
    basis=AbstractHankelBasis() # all BIE formulations have no formal basis, so we make one up for multi-dispacth to be nicer
elseif method==:ebim_bim
    solver=BoundaryIntegralMethod(b,billiard,symmetry=symmetry) # EBIM-capable naive BIM backend using A, dA, and ddA
    basis=AbstractHankelBasis() # all BIE formulations have no formal basis, so we make one up for multi-dispacth to be nicer
elseif method==:ebim_dlp_kress
    solver=DLP_kress(b,billiard,symmetry=symmetry) # smooth periodic DLP Kress solver for EBIM tests on no-corner geometries
    basis=AbstractHankelBasis() # all BIE formulations have no formal basis, so we make one up for multi-dispacth to be nicer
else
    error("Unknown method: $method") # keep method choice explicit and user-controlled from the top of the file
end

################################################################################
################################## RUN #########################################
################################################################################

println("Running $(method) on $(nameof(typeof(billiard))) ...")

f=Figure()
ax=Axis(f[1,1],xlabel="k",ylabel="log10 min. parameter")

if method==:psm || method==:dm || method==:bim_sweep || method==:ebim_bim || method==:ebim_dlp_kress
    # compute the minimization parameters over a grid of k values. The minima represent where the eigenvalues are located approximately to the resolution of the grid.
    tens=k_sweep(solver, 
    basis, # based on the solver type, this is either the constructor-provided basis for PSM/DM or a dummy Hankel basis (AbstractHankelBasis()) for the BIM methods since they dont have a formal basis
    billiard, 
    kgrid, # ks for which the tensions will be computed
    multithreaded_matrices=true, # allow threaded matrix assembly for the sweep for each k
    use_krylov=false, # use the shift-invert Krylov method to find the smallest singular value and vector at each k instead of the full dense SVD; 
    # this is much faster for large problems but not that useful for small and intermediate k-regime
    which=:svd, # alg for minimization parameter -> used for Fredholm based approaches like BIM and CFIE, for PSM/DM it does nothing
    # :svd for smallest singular value, :det_argmin for the min(|det(A)|), :det gives back det(A) which is a complex number to compare if the roots for both coincide (almost impossible for naive BIM)
    ) 
    ks=get_eigenvalues(kgrid,tens;threshold=100.0) # in the logarithmic tension plot, find the local minima that are below the threshold and return their k values as approximate eigenvalues. The threshold is a tuning parameter that should be above the noise floor but below the typical tension values at non-eigenvalues.
    # threshold gives a minimal value for which the local minima is really a minima (in the log scale)
    # one can plo _second_derivative_sweep_spectrum(ks,log10.(abs.(tens))) to get a sense of the noise floor and how to set the threshold.
    lines!(ax,kgrid,log10.(abs.(tens)),color=:red)
    
    if plot_wavef
        Psi2ds=Vector{Matrix{Float64}}(undef,length(ks))
        xgrid=Vector{Float64}(undef,0) # make them exist
        ygrid=Vector{Float64}(undef,0) # make them exist
        if method==:bim_sweep || method==:ebim_bim  || method==:ebim_dlp_kress
            # BIE type solvers have nice wavefunction reconstruction formulas using the layer potentials
            us_all,pts_all=solve_vect(solver,billiard,basis,ks) # computes the singular vector to the smallest singular value. 
            # Nicely enough for DLP this correspons to the normal derivative of the eigenfunction so we dont have to do any more work! 
            # Still need to symmetrize it with boundary_function
            pts_all,us_all=boundary_function(solver,us_all,pts_all,billiard) # symmetrize the boundary function in the correct irrep
            Psi2ds,xgrid,ygrid=wavefunction_multi(ks,us_all,pts_all,billiard;fundamental=false) # reconstruct the wavefunctions for all states in a batch
            # fundamental just means we get the wavefunction on the fundamental domain, which is smaller and faster to compute, but since we want to plot the full wavefunction we set it to false
            Psi2ds=[abs.(Psi2d) for Psi2d in Psi2ds] # makes phase independance easier to handle in the plotting
        else
            @showprogress "constructing eigenstates" for (i,k) in enumerate(ks)
                # Basis type solvers dont have access to solve_vect since we dont get an easy way to get layer potentials
                # in principle what one can do is to call from "state" below the boundary_function method that is 
                # dispatched on the "Eigenstate" struct to get the normal derivative of the wavefunction usually used for 
                # husimis and then insert that into the wavefunction_multi for faster evaluations!
                state=compute_eigenstate(solver,basis,billiard,k) # build the library Eigenstate at this k for basis type methods
                Psi2ds[i],xg,yg=wavefunction(state;fundamental_domain=false) # reconstruct the wavefunction and its plotting grids
                # this is quite slow, since it needs to for each grid point evaluate the entire basis. But this is not a problem
                # as PSM/DM are just for checking basis convergence for VerginiSaraceno
                if i==1
                    global xgrid=xg # keep the common x-grid once
                    global ygrid=yg # keep the common y-grid once
                end
            end
        end
        fs=plot_wavefunctions(ks,Psi2ds,xgrid,ygrid,billiard;fundamental=false)
        save("wavefunction_$(nameof(typeof(billiard)))_$(method)_$(nameof(typeof(symmetry))).png",fs[1])
    end
    println("==============================================================")
end
if method==:ebim_bim || method==:ebim_dlp_kress
    #NOTE The tensions (tens) in EBIM are just the distances of the eigenvalue from the k0 reference eigenvalue where the GEVP was being solved, so they are not directly comparable to the sweep tensions.
    ks,tens=compute_spectrum_ebim(
        solver,                          # EBIM solver backend used to assemble A, dA, and ddA
        billiard,                        # billiard geometry whose boundary points are reused segmentwise during the EBIM scan
        k1,                              # lower endpoint of the EBIM search interval
        k2;                              # upper endpoint of the EBIM search interval
        dk=(k->0.05*k^(-1/3)),           # local sampling step in k; smaller step means denser EBIM centers and more robust overlap merging
        tol=1e-4,                        # tolerance used when merging duplicate corrected eigenvalues from neighboring EBIM centers
        use_lapack_raw=false,            # use Julia/generalized eigen backend instead of raw LAPACK ggev in the dense solve branch (imporant for getting L and R eigenvectors)
        multithreaded_matrices=true,     # allow threaded matrix assembly for A, dA, ddA on each segment / center
        use_krylov=true,                 # use the shift-invert Krylov EBIM correction instead of the full dense generalized eigensolve
        seg_reuse_frac=0.95,             # reuse one boundary discretization for a segment of nearby k values before rebuilding it
        solve_info=true,                 # print one detailed diagnostic INFO solve at the start of the EBIM run
        use_chebyshev=true,              # build derivative matrices with Chebyshev-accelerated Hankel evaluation instead of the direct pathway - since in demo ks are small there is no need
        n_panels=15000,                  # initial number of Chebyshev radial panels used for special-function interpolation
        M=5,                             # Chebyshev polynomial degree on each radial panel
        cheb_param_strategy=:global,     # choose one Chebyshev panelization from the largest k in the interval and reuse it for all segments
        cheb_tol=1e-13,                  # target tolerance used when auto-tuning Chebyshev interpolation parameters
        max_iter=20,                     # maximum number of refinement rounds when tuning Chebyshev parameters
        sampling_points=50_000,          # number of sample radii used to assess Chebyshev interpolation accuracy
        grading=:uniform,                # panel spacing strategy in radius; :uniform is simplest and robust for this example
        grow_panels=1.5,                 # multiplicative growth factor when the tuning loop decides more panels are needed
        grow_M=2,                        # multiplicative growth factor when the tuning loop decides higher Chebyshev degree is needed
        verbose_cheb_panelization=false  # keep the Chebyshev tuning output quiet except for the main solve diagnostics
    )

    scatter!(ax,ks,log10.(tens),color=:blue)

    println()
    println("==============================================================")
    println("$(method) summary")
    println("==============================================================")
    for i in eachindex(ks)
        @printf("state %d: k = %.12f\n",i,ks[i]) # print the EBIM eigenvalues
    end
    println("==============================================================")
    println()
end

save("sweep_$(nameof(typeof(billiard)))_$(method)_$(nameof(typeof(symmetry))).png",f)

println("Done.")