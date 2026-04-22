using QuantumBilliards
using LinearAlgebra
using StaticArrays
using Printf
using CairoMakie

# NOTE: This is a competely detailed example with explanations of each step, 
# meant for users who want to understand the full pipeline of going from geometry 
# to wavefunctions using Beyn. In practice only a few kwargs need to be changed, and 
# Beyn is the most complex solver of them all.

# This example is meant for true corner geometries, where we want a panel-based
# solver rather than the periodic Kress machinery. That is why we use CFIE_alpert.
# Using on on geometries with smooth joins like stadium will make it very ... unuseful
#
# We only plot wavefunctions here.
# We intentionally do NOT try to build Husimis / boundary functions from the
# recovered physical normal derivative, because for corner geometries that step
# is the delicate one and is not the point of this example.
#
# So the pipeline here is:
#
#     geometry -> CFIE_alpert -> Beyn -> layer potential μ -> wavefunctions ψ
#
# and then we visualize ψ only.

# Pick one of the polygon geometries for which Alpert is most useful!
geometry=:rectangle
#geometry=:triangle

# Main CFIE / spectral scale parameter. Roughly controls the boundary resolution
# used by the solver construction.
# N = k * L * b / (2*pi) -> b = points per wavelength
b=20.0 # good for 1e-(7-9) im part at small k (k<100) and 1e-(10-11) im part at k>200 (this is a good estimate to the eigenvalue accuracy). 
# Well below the needed 1% of mean level spacing needed for spectra!

# Beyn search window [k1,k2]. We ask Beyn to find states in this interval.
# Beyn is good for large k windows, for smaller ones EBIM is preffered due to cheybshev construction overheaf (even if only once per compute_spectrum_beyn)
# takes 1h 3min on M3 MAX (64GB 12 cores), but for low RAM systems better to do smaller windows for demoing.
#k1=5.0
#k2=400.0

# demo window - main loop wihtout info solve takes 1min 22sec on M3 MAX (64GB 12 cores)
k1=200.0
k2=205.0

# if still not eonugh RAM use
#k1=60.0
#k2=70.0

# Beyn can handle symmetries  nternally via applying the projection to the subspace iterates, 
# but for simplicity we can use the no-symmetry option as:

#symmetry=nothing (uncomment if want this)

# for rectangle you can try symmetry=XYReflection(-1,-1) to get the odd-odd subspace.
# The symmetry used should actually exist in the geometry due to how the collocation
# logic is constructed.

# Usually desymmetrization helps with accuracy of obtained eigenvalues by 1-2 digits
symmetry=XYReflection(-1,-1) # one of rectangle's symmetries -> check eigenfunctions to verify!
# !!! For square in this example this is not the full symmetry group, so degeneracy is still 
# present, but all states do have the correct odd-odd reflection symmetry (proper subspace)
# !!! Running this on a triangle will produce nonsense since the one in this example has no such symmetry !!!

function make_geometry_and_basis(geometry)
    if geometry==:rectangle
        # Unit square / rectangle example with true corners.
        billiard,_=make_rectangle_and_basis(1.0,1.0)
    elseif geometry==:triangle
        # 2 corners that define a rectangle, and a third corner that creates a 60 degree angle with the first two.
        billiard,_=make_triangle_and_basis(pi/3,2*pi/3) 
    else
        @error "Unknown geometry choice: $geometry, make your own!"
    end
    return billiard
end

billiard=make_geometry_and_basis(geometry)

################################################################################
############################## SOLVER CONSTRUCTION #############################
################################################################################

# Alpert CFIE for corner geometries.
#
# alpert_order=16:
#   use the 16th-order Alpert correction rule near singular interactions.
#
# alpertq=2:
#   number of local singular quadrature panels / correction width knob in the
#   implementation. 2 is a good number to start with at 16th degree
#   Beyn will in the solve information report the minimal distances taking into 
#   account correction nodes, so best keep it below 1e-14 to prevent singular errors.
#
solver=CFIE_alpert(b,billiard;symmetry=symmetry,alpert_order=16,alpertq=2)

################################################################################
############################### BEYN HELPERS ###################################
################################################################################

# Beyn returns for non-desymmetrized geometries all the degeneracies,
# precisely usually up to the imaginary part of the obtained eigenvalue

function run_beyn(
    solver,          # boundary-integral solver (CFIE_kress, CFIE_alpert, BIM, ...)
    billiard,        # billiard geometry on which Weyl windows / collocation are built
    k1,              # left endpoint of the wavenumber interval to scan
    k2;              # right endpoint of the wavenumber interval to scan
    m=50,                    # target number of eigenvalues per Weyl-planned contour window
    Rmax=0.7,                # maximum contour radius; each planned window has half-width ≤ Rmax
    nq=45,                   # number of trapezoidal quadrature nodes on each circular contour
    r=100,                  # Beyn probe rank; should exceed expected number of roots in one window
    svd_tol=1e-12,           # singular-value cutoff used to detect the numerical rank of A0
    res_tol=1e-9,            # residual threshold for discarding spurious roots after contour solve
    auto_discard_spurious=true, # whether to drop roots whose residual ||A(k)φ|| is too large
    multithreaded_matrix=true,  # whether boundary matrix assembly should use multithreading
    use_chebyshev=true,      # whether to use Chebyshev-accelerated Hankel evaluation for complex k
    n_panels_init=15000,     # initial number of Chebyshev radial panels before optional tuning
    M_init=5,                # initial Chebyshev polynomial degree before optional tuning
    do_INFO_init=true,       # whether to run one diagnostic solve_INFO on a representative disk
    do_per_solve_INFO=false, # whether to print timing / diagnostics during every solve
    cheb_tol=1e-13,          # tolerance used when tuning Chebyshev interpolation parameters
    max_iter=20,             # maximum number of refinement iterations in Chebyshev tuning
    sampling_points=50_000,  # number of sample points used when estimating Chebyshev accuracy
    grading=:uniform,        # panel grading strategy for Chebyshev interpolation (:uniform or :geometric) 
    grow_panels=1.5,         # multiplicative growth factor when increasing the number of Chebyshev panels
    grow_M=2,                # multiplicative growth factor when increasing the Chebyshev degree
)

    # compute_spectrum_beyn returns:
    # ks    = candidate eigenvalues kept after residual filtering
    # tens  = raw residual norms ||A(k)φ||
    # us    = boundary densities μ (or φ) for each kept state
    # pts   = boundary discretization object used for each kept state
    # tensN = normalized residuals, scale-free version of the raw residuals
    ks,tens,us,pts,tensN=compute_spectrum_beyn(
        solver,                          # solver defining the boundary operator T(k)
        billiard,                        # billiard used for Weyl planning and point evaluation
        k1,                              # lower scan bound
        k2;                              # upper scan bound
        m=m,                             # target number of levels per planned window
        Rmax=Rmax,                       # cap on contour radius
        nq=nq,                           # number of contour nodes per disk
        r=r,                             # Beyn probe rank / number of random test vectors
        svd_tol=svd_tol,                 # SVD rank-detection threshold
        res_tol=res_tol,                 # residual tolerance for filtering roots
        auto_discard_spurious=auto_discard_spurious, # whether to remove large-residual roots
        multithreaded_matrix=multithreaded_matrix,   # matrix assembly threading flag
        use_chebyshev=use_chebyshev,     # use Chebyshev Hankel interpolation for complex contour evals
        n_panels_init=n_panels_init,     # initial Chebyshev panel count
        M_init=M_init,                   # initial Chebyshev degree
        do_INFO_init=do_INFO_init,       # run one representative diagnostic solve
        do_per_solve_INFO=do_per_solve_INFO, # verbose timings / diagnostics for every window
        cheb_tol=cheb_tol,               # Chebyshev tuning accuracy target
        max_iter=max_iter,               # max tuning iterations for Chebyshev parameters
        sampling_points=sampling_points, # sample count used in Chebyshev parameter selection
        grading=grading,                 # panel grading strategy
        grow_panels=grow_panels,         # panel-count growth factor in Chebyshev tuning
        grow_M=grow_M                    # polynomial-degree growth factor in Chebyshev tuning
    )
    return ks,tens,us,pts,tensN # final kept eigenvalues, residuals, densities, boundary data, normalized residuals
end

################################################################################
############################### RUN BEYN #######################################
################################################################################

println("Running CFIE_alpert Beyn on $(nameof(typeof(billiard))) ...")

ks,tens,us,pts_all,tensN=run_beyn(
    solver,
    billiard,
    k1,
    k2;
    m=50,                  
    Rmax=0.7,             
    nq=45,                
    r=100,                 
    svd_tol=1e-12,         
    res_tol=1e-9,        
    auto_discard_spurious=true,
    multithreaded_matrix=true,
    use_chebyshev=true,    
    n_panels_init=15000,  
    M_init=5,         
    do_INFO_init=true,
    do_per_solve_INFO=false,
)

println()
println("==============================================================")
println("Beyn summary")
println("==============================================================")
for i in eachindex(ks)
    @printf("state %d: k = %.12f, residual = %.6e\n",i,ks[i],tens[i])
end
println("==============================================================")
println()

################################################################################
############################ WAVEFUNCTION MATRICES ##############################
################################################################################

# For CFIE solvers the returned `us` are boundary densities μ.
# The library CFIE overload of wavefunction_multi reconstructs the interior
# wavefunction from these densities directly, so we do not need to build any
# extra boundary-function object here.

# do first 20 of them
ks=ks[1:20]
us=us[1:20]
pts_all=pts_all[1:20]

Psi2ds,x_grid,y_grid=wavefunction_multi(
    ks,                    # eigenvalues k for all states to be reconstructed
    us,                    # layer potentials corresponding to each state
    pts_all,               # boundary discretizations for each state (typically Vector{BoundaryPointsCFIE})
    billiard;              # billiard geometry used to build the common plotting grid and inside-mask
    b=5.0,                 # grid-density scaling: larger b -> finer x/y plotting grid
    inside_only=true,      # evaluate ψ only at points inside the billiard; outside stays zero ->  for checking correctness 
    fundamental=false,     # if true use the fundamental-domain mask/limits, otherwise use the full billiard
    MIN_CHUNK=4096,        # minimum number of interior grid points assigned per thread chunk
    float32_bessel=false   # if true use Float32 Bessel/Hankel evaluation inside ϕ_cfie for speed/memory
)

# Psi2ds = Vector of 2D wavefunction matrices, one per state
# x_grid = common x-grid used for all plotted wavefunctions
# y_grid = common y-grid used for all plotted wavefunctions

# Makie plotting is simplest if we feed it real arrays here.
# For these corner-domain examples we mainly want to inspect nodal structure.
Psi2ds_plot=[real.(Ψ) for Ψ in Psi2ds]

################################################################################
################################## PLOTTING ####################################
################################################################################

# custom labels for each state, showing k and residual
labels=[
    @sprintf("k = %.12f\nres = %.3e",ks[i],tensN[i])
    for i in eachindex(ks)
]

figs=plot_wavefunctions(
    ks,                   # eigenvalues k used for labeling each plotted wavefunction
    Psi2ds_plot,          # vector of 2D wavefunction arrays (typically real/abs/phase processed)
    x_grid,               # x-coordinates of the plotting grid (shared by all states)
    y_grid,               # y-coordinates of the plotting grid (shared by all states)
    billiard;             # billiard geometry used to overlay boundary and mask outside region
    N=50,                 # number of wavefunctions to plot per Makie Figure
    max_cols=4,           # maximum number of subplot columns (layout control)
    width_ax=320,         # width (in pixels) of each subplot axis
    height_ax=320,        # height (in pixels) of each subplot axis
    fundamental=false,    # if true plot only fundamental domain boundary, else full billiard
    custom_label=labels   # optional custom labels for each subplot (overrides default k-labels)
)

# figs = collection (e.g. Vector) of Makie Figure objects containing the plotted wavefunctions

for (i,fig) in enumerate(figs)
    save("cfie_alpert_wavefunctions_$(geometry)_$(i).png",fig)
end

println("Done.")