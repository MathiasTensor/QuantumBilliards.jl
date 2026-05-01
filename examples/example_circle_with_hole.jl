using QuantumBilliards
using LinearAlgebra
using StaticArrays
using Printf
using CairoMakie

# try MKL if possible, lu! for Beyn is much faster!
try_MKL!()

# Requies moderate ammount of RAM for this, if not try lowering k1,k2
b=6.5
k1=250.0
k2=251.0

# Four-hole geometry with XY reflection symmetry.
# The holes are arranged as symmetric pairs:
#
#   (+a,+c), (-a,+c), (+a,-c), (-a,-c)
#
# along with a central hole ata the origin (0,0).
symmetry=XYReflection(-1,-1)

R=1.0
rh=0.1
rc=0.35
a=0.4
c=0.4
holes=[
    (rc, 0.0, 0.0), # center hole
    (rh, a, c), # top right
    (rh,-a, c), # top left
    (rh, a,-c), # bottom right
    (rh,-a,-c), # bottom left
]
billiard,_=make_multihole_and_basis(R,holes)
solver=CFIE_kress(b,billiard;symmetry=symmetry)

##############
#### BEYN ####
##############

ks,tens,us,pts_all,tensN=compute_spectrum_beyn(
    solver,                         # CFIE_kress solver defining A(k)
    billiard,                       # four-hole billiard geometry
    k1,                             # lower search interval edge
    k2;                             # upper search interval edge
    m=300,                          # target expected states per planned Weyl window
    Rmax=0.5,                       # maximum contour radius
    nq=45,                          # contour quadrature nodes
    r=350,                          # Beyn probe rank
    svd_tol=1e-12,                  # Beyn SVD rank cutoff
    res_tol=1e-9,                   # residual filter cutoff
    auto_discard_spurious=true,     # discard high-residual candidates
    multithreaded_matrix=true,      # threaded matrix assembly
    use_chebyshev=true,             # use Chebyshev Hankel acceleration
    n_panels_init=15000,            # initial Chebyshev radial panels
    M_init=5,                       # initial Chebyshev degree
    do_INFO_init=true,              # one diagnostic representative disk
    do_per_solve_INFO=false,        # avoid verbose per-disk diagnostics
    cheb_tol=1e-13,                 # Chebyshev tuning tolerance
    max_iter=20,                    # maximum Chebyshev tuning iterations
    sampling_points=50_000,         # samples for Chebyshev tuning
    grading=:uniform,               # radial panel grading
    grow_panels=1.5,                # panel growth factor
    grow_M=2,                       # degree growth factor
)

println()
println("==============================================================")
println("Beyn summary")
println("==============================================================")
for i in eachindex(ks)
    @printf("state %d: k = %.12f, residual = %.6e, normalized = %.6e\n",
        i,ks[i],tens[i],tensN[i])
end
println("==============================================================")
println()

##############################################
##### BOUNDARY FUNCTIONS / WAVEFUNCTIONS #####
##############################################

# Formal consistency step. For full-boundary Kress/CFIE discretizations this is
# a no-op, but it keeps the postprocessing pipeline identical to other solvers.
@time "symmetrize layer potential" pts_all,us=symmetrize_layer_potential(
    solver,         # CFIE/DLP-style solver
    us,             # Beyn boundary densities
    pts_all,        # boundary discretizations from Beyn
    billiard,       # geometry
)

Psi2ds,x_grid,y_grid=wavefunction_multi(
    solver,         # CFIE solver
    ks,             # eigenvalues
    us,             # normalized boundary functions / densities
    pts_all,        # boundary discretizations
    billiard;       # geometry
    b=b,            # plotting-grid density factor
    inside_only=true,
    fundamental=false,
    MIN_CHUNK=4096,
    use_chebyshev=true, # use Chebyshev acceleration for the interior wavefunction evals, 
    # which is currently only supported for CFIE used for high k to speed up the evaluation 
    # of many wavefunctions in a batch since the cost of the Chebyshev tuning is amortized over all evals
)

# For CFIE, this should normalize/interpret the CFIE density as the physical
# boundary function used by Husimis and wavefunction reconstruction.
@time "boundary function construction" pts_bdry,u_bdry=boundary_function(
    solver,         # CFIE solver used to interpret boundary data
    us,             # boundary densities from Beyn
    pts_all,        # boundary discretizations
    billiard,       # geometry
    ks,             # eigenvalues
)

Hs_list,ps_list,qs_list=husimi_functions_from_us_and_boundary_points(
    ks,             # eigenvalues
    u_bdry,         # boundary functions
    pts_bdry,       # boundary points
    500,            # q-grid size
    500;            # p-grid size
    full_p=true,
)

##################
#####PLOTTING ####
##################

labels=[
    @sprintf("k = %.12f\nresN = %.3e",ks[i],tensN[i])
    for i in eachindex(ks)
]

figs=plot_wavefunctions_with_husimi(
    ks,
    Psi2ds,
    x_grid,
    y_grid,
    Hs_list,
    ps_list,
    qs_list,
    billiard,
    u_bdry,
    pts_bdry;
    N=100,
    max_cols=3,
    width_ax=320,
    height_ax=320,
    fundamental=false,
    custom_label=labels,
    plt_boundary=true,
)

save("wavefunctions_husimi_$(nameof(typeof(billiard)))_$(nameof(typeof(solver))).png",figs[1])
println("Done.")