using QuantumBilliards
using LinearAlgebra
using StaticArrays
using Printf
using CairoMakie

try_MKL_on_x86_64!() # try to use MKL on x86_64 for faster linear algebra, but don't error if not available. 
# On M-series chips this will do nothing and just use OpenBLAS, which is fine.

# NOTE: This is a detailed example for smooth geometries, where the periodic
# Kress machinery is the natural choice. In contrast to the Alpert example,
# here we DO also construct boundary functions and Husimis, because for smooth
# boundaries this is a stable and meaningful thing to do.
#
# So the pipeline here is:
#
#     geometry -> CFIE_kress -> Beyn -> density μ -> boundary function u(s)
#              -> wavefunctions ψ + Husimis H(q,p)
#
# and then we visualize all three together.

# Pick one smooth geometry:
#geometry=:star
geometry=:star_hole
#geometry=:prosen
#geometry=:robnik
#geometry=:ellipse
#geometry=:africa -> has no symmetries

# Main CFIE / spectral scale parameter.
# Roughly:
#
#     N ≈ k * L * b / (2π)
#
# so b behaves like "points per wavelength" on the boundary discretization.
b=10.0 # b can be relatively small to get very good accuracy (1e-14-1e-15 im part - proxy for accuracy) for smooth geometries since Kress
# has spectral convergence on smooth boundaries. 

# Beyn search window [k1,k2]. For demo purposes we keep it small enough to run comfortably.
# this takes 2 min 15 sec on M3 MAX (64GB 12 cores) for the main solve loop without the info solve with the current parameters.
#k1=200.0
#k2=205.0

# if not enough RAM use maybe
k1=80.0
k2=85.0

# Symmetry choices based on billiard chosen
#symmetry=nothing # no symmetry, full billiard, valid for all trivially
#symmetry=XYReflection(-1,-1) # for Prosen/Ellipse billiard this gives the odd-odd subspace, which is a good test case for symmetry reduction.
symmetry=Rotation(5,0) # for the 5 star billiard with or wihtout hole
#symmetry=YReflection(-1) # for the Robnik billiard, this gives the odd subspace which is a good test case for symmetry reduction.

function make_geometry(geometry)
    if geometry==:star
        # Smooth star-shaped billiard without holes with 5-fold symmetry.
        billiard,_=make_star_and_basis(5)
    elseif geometry==:star_hole
        # Smooth outer star with a smooth inner star-shaped hole both with 5-fold symmetry.
        billiard,_=make_star_hole_and_basis(
            Float64; # force Float64 for the geometry parameters
            Rout=1.0, # outer radius of the star
            ϵout=0.22, # outer star deformation parameter (0 is a circle, larger is more deformed)
            mout=5, # number of outer star lobes
            Rin=0.3, # inner radius of the hole star
            ϵin=0.11, # inner star deformation parameter
            min=5, # number of inner star lobes
            xin=0.0, # x center of the inner star hole
            yin=0.0, # y center of the inner star hole
            φin=0.0*pi # rotation angle of the inner star hole in radians
        )
    elseif geometry==:prosen
        # Smooth Prosen billiard with deformation parameter 0.2, which is in the mixed phase space regime.
        billiard,_=make_prosen_and_basis(0.2)
    elseif geometry==:robnik
        # Smooth Robnik billiard with deformation parameter 0.2, which is in the mixed phase space regime.
        billiard,_=make_robnik_and_basis(0.2)
    elseif geometry==:ellipse
        # Smooth ellipse billiard with semiaxes 1 and 0.5, which is integrable.
        billiard,_=make_ellipse_and_basis(1.0,0.5)
    elseif geometry==:africa
        # Smooth "Africa"-shaped billiard with deformation parameter 0.15.
        billiard,_=make_africa_and_basis(0.15)
    else
        @error "Unknown geometry choice: $geometry, make your own!"
    end
    return billiard
end

billiard=make_geometry(geometry)

################################################################################
############################## SOLVER CONSTRUCTION #############################
################################################################################

# CFIE_kress is the smooth-boundary periodic solver.
#
# This is the natural choice for analytic / smooth boundaries where the Kress
# logarithmic split and periodic quadrature are effective.
solver=CFIE_kress(b,billiard;symmetry=symmetry)

################################################################################
############################### BEYN HELPERS ###################################
################################################################################

# The floor for Kress is not as a strong cutoff as

function run_beyn(
    solver,          # boundary-integral solver (here typically CFIE_kress)
    billiard,        # billiard geometry on which Weyl windows / collocation are built
    k1,              # left endpoint of the wavenumber interval to scan
    k2;              # right endpoint of the wavenumber interval to scan
    m=200,                    # target number of eigenvalues per Weyl-planned contour window
    Rmax=0.5,                # maximum contour radius; each planned window has half-width ≤ Rmax
    nq=45,                   # number of trapezoidal quadrature nodes on each circular contour
    r=215,                  # Beyn probe rank; should exceed expected number of roots in one window
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
    return ks,tens,us,pts,tensN
end

################################################################################
############################### RUN BEYN #######################################
################################################################################

println("Running CFIE_kress Beyn on $(nameof(typeof(billiard))) ...")

ks,tens,us,pts_all,tensN=run_beyn(
    solver,
    billiard,
    k1,
    k2;
    m=200,
    Rmax=0.5,
    nq=45,
    r=215,
    svd_tol=1e-12,
    res_tol=1e-9,
    auto_discard_spurious=true,
    multithreaded_matrix=true,
    use_chebyshev=true,
    n_panels_init=15000,
    M_init=5,
    do_INFO_init=true,
    do_per_solve_INFO=false
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
######################## BOUNDARY FUNCTIONS / WAVEFUNCTIONS #####################
################################################################################

# Now construct interior wavefunction matrices ψ(x,y) on one common plotting grid.
Psi2ds,x_grid,y_grid=wavefunction_multi(
    ks,                    # eigenvalues k for all states to be reconstructed
    us,                    # layer potentials / CFIE densities corresponding to each state
    pts_all,               # boundary discretizations for each state
    billiard;              # billiard geometry used to build the common plotting grid and inside-mask
    b=b,                 # grid-density scaling: larger b -> finer x/y plotting grid
    inside_only=true,      # evaluate ψ only at points inside the billiard; outside stays zero
    fundamental=false,     # if true use the fundamental-domain mask/limits, otherwise use the full billiard
    MIN_CHUNK=4096,        # minimum number of interior grid points assigned per thread chunk
    float32_bessel=true   # if true use Float32 Bessel/Hankel evaluation inside ϕ_cfie for speed/memory
)

# For smooth CFIE_kress states we turn the returned layer potentials μ
# into the physical boundary functions u(s) = \partial_n ψ(s).
#
# boundary_function(solver,μ,pts,k) returns the physical boundary function
# evaluated on the same collocation points encoded in pts.
#
# We also collect the corresponding arclength coordinates s from pts so they can
# be used directly for Husimi construction and boundary plotting.
#
# Under the hood the library will evaluate the normal derivative of the wavefunction
# via the normal derivative of the CFIE layer potential, which is stable for smooth boundaries
# via the maue fomrula for the hypersingular kernels. For corners a regularization is needed (not yet done.)
u_bdry=Vector{Vector{ComplexF64}}(undef,length(ks))
for i in eachindex(ks)
    _,u=boundary_function(solver,us[i],pts_all[i],ks[i])
    u_bdry[i]=u
end

# Construct Husimis from the physical boundary functions u(s).
#
# This is the natural smooth-boundary route:
#
#     (k,u(s),s) -> H(s,p)
#
# For multiply connected smooth geometries (for example star_hole), the library
# Husimi helper should handle the boundary data returned by the smooth workflow.
Hs_list,ps_list,qs_list=husimi_functions_from_us_and_boundary_points(
    ks,          # eigenvalues k for labeling and scaling the Husimis
    u_bdry,      # vector of physical boundary functions u(s) for each state (not layer potentials!)
    pts_all,      # corresponding arclength coordinates for each u(s)
    500,         # number of pts on s grid
    500;         # number of pts on p grid
    full_p=false # use p -> -p symmetry when appropriate and reconstruct the full signed grid.
                 # if stumbled if a given irrep has this symmetry, you can set full_p=true to get the full grid and avoid the symmetry check.
                 # it will handle it anyway correctly for the final Husimi, but the intermediate p-grid will be full instead of half, which is less efficient.
)

################################################################################
################################## PLOTTING ####################################
################################################################################

labels=[
    @sprintf("k = %.12f\nres = %.3e",ks[i],tensN[i])
    for i in eachindex(ks)
]

figs=plot_wavefunctions_with_husimi(
    ks,                   # eigenvalues k used for labeling each plotted state
    Psi2ds,               # vector of 2D wavefunction arrays to visualize
    x_grid,               # x-coordinates of the common wavefunction grid
    y_grid,               # y-coordinates of the common wavefunction grid
    Hs_list,              # Husimi matrices, one per state
    ps_list,              # p-grids for the Husimis
    qs_list,              # q-grids for the Husimis
    billiard,             # billiard geometry used to overlay the boundary
    u_bdry,               # physical boundary functions u(s), shown in the lower panel
    pts_all,              # boundary discretizations used for all calcs, especially u_bdry
    N=50,                 # number of states per Figure
    max_cols=3,           # maximum number of subplot columns in one Figure
    width_ax=320,         # width of each axis in pixels
    height_ax=320,        # height of each axis in pixels
    fundamental=false,    # if true plot only the fundamental domain, else the full billiard
    custom_label=labels   # optional custom text label for each state
)

# the blue vertical dashed lines signify where the other boundary (hole) has started. In principle 
# separate plots for these should be made, but it is visually easier to just concatenate the u(s) and
# H(s,p) for all components into a single subplot (the husimis are separately normalized to sum to 1
# for each closed boundary component)

for (i,fig) in enumerate(figs)
    save("cfie_kress_wavefunctions_husimi_$(geometry)_$(i).png",fig)
end

println("Done.")
