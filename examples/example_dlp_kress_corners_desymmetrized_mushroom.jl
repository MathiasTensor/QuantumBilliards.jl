using QuantumBilliards
using LinearAlgebra
using CairoMakie
using ProgressMeter
using StaticArrays

# NOTE: Requires 10 GB of RAM
# If not enough RAM decrease b,k and importantnly also increase min_t_spacing in the kress 
# solver to avoid too close nodes and resulting bad global central node distribution. 
# Checks the corner graded dlp kress for the mushroom billiard againts a standard DLP with just diagonal correction/limit
# it shows that the convergence is a very high order
try_MKL!()

billiard,_=make_mushroom_and_basis(1.0,1.0,1.0)
#billiard,_=make_stadium_and_basis(0.5)
symmetry=XReflection(-1)
#symmetry=XYReflection(-1,-1)

# Basis scaling parameter used internally by BIM/DLP solvers.
# Roughly controls boundary discretization density: N ~ b*k
# Larger b -> more boundary nodes -> higher accuracy but more expensive.
b=20.0

# Spectral window
k1=100.0
k2=101.0

# Corner-adapted Kress version 
# strong kress gradings require a high b or k regime or preferably both.
solver_dlp_kress=DLP_kress_global_corners(b,billiard;symmetry=symmetry,kressq=6,min_t_spacing=1e-15)

# Standard BIM reference solver
solver_bim=BoundaryIntegralMethod(b,billiard;symmetry=symmetry)

#####################################################################
######################## BEYN PARAMETERS ############################
#####################################################################

# Target number of eigenvalues per Beyn contour disk.
# The algorithm automatically tiles [k1,k2] with overlapping disks.
m=100
# Maximum contour radius.
# Larger radius:
#   + fewer disks
#   - harder nonlinear solve
Rmax=0.5
# Number of contour quadrature nodes.
# Higher nq:
#   + more accurate contour integration
#   - more matrix solves
nq=45
# Number of random probing vectors used in Beyn.
# Choose: r >= expected eigenvalue multiplicity
# Slightly overestimating is fine.
r=150
# SVD threshold for determining numerical rank
svd_tol=1e-12
# Residual threshold used to discard bad/spurious roots
res_tol=1e-9
# Automatically discard roots with large residuals
auto_discard_spurious=true
# Use threaded matrix assembly
multithreaded_matrix=true
# During Beyn we evaluate kernels at many COMPLEX k values.
# Direct Hankel evaluations become expensive, so we interpolate
# the radial dependence using piecewise Chebyshev expansions.
use_chebyshev=true
# Run one representative diagnostic solve before the full run
# Useful to inspect interpolation quality and timings.
do_INFO_init=true
# Desired interpolation accuracy
cheb_tol=1e-13
# return complex k for error check
return_imag_part=true

# Returns:
#
# ks      -> eigenvalues
# tens    -> smallest singular values / tensions
# us      -> boundary densities (layer potentials)
# pts     -> boundary discretizations used for each eigenvalue
# tensN   -> normalized tensions (not useful in this context)
ks_dlp_kress,tens_dlp_kress,us_all_dlp_kress,pts_all_dlp_kress,tensN_dlp_kress=compute_spectrum_beyn(
    solver_dlp_kress,
    billiard,
    k1,
    k2;
    # planned eigenvalue count per contour window
    m=m,
    # contour disk radius cap
    Rmax=Rmax,
    # contour quadrature nodes
    nq=nq,
    # Beyn probe rank
    r=r,
    # SVD truncation tolerance
    svd_tol=svd_tol,
    # residual filter threshold
    res_tol=res_tol,
    # remove large-residual roots
    auto_discard_spurious=auto_discard_spurious,
    # threaded matrix assembly
    multithreaded_matrix=multithreaded_matrix,
    # enable Chebyshev Hankel interpolation
    use_chebyshev=use_chebyshev,
    # diagnostics
    do_INFO_init=do_INFO_init,
    # interpolation target accuracy
    cheb_tol=cheb_tol,
    return_imag_part=return_imag_part)

ks_bim,tens_bim,us_all_bim,pts_all_bim,tensN_bim=compute_spectrum_beyn(
    solver_bim,
    billiard,
    k1,
    k2;
    m=m,
    Rmax=Rmax,
    nq=nq,
    r=r,
    svd_tol=svd_tol,
    res_tol=res_tol,
    auto_discard_spurious=auto_discard_spurious,
    multithreaded_matrix=multithreaded_matrix,
    use_chebyshev=use_chebyshev,
    do_INFO_init=do_INFO_init,
    cheb_tol=cheb_tol,
    return_imag_part=return_imag_part)

@info "DLP Kress eigvals: $ks_dlp_kress"
println()
@info "BIM eigvals: $ks_bim"
println()

# all functions below expect real k, so we take real part of computed eigenvalues.
ks_dlp_kress=real.(ks_dlp_kress)
ks_bim=real.(ks_bim)

#####################################################################
####################### SYMMETRIZE DENSITIES ########################
#####################################################################

# The reduced symmetry solve only reconstructs the fundamental-domain
# boundary density.
# We now reconstruct the FULL boundary density by applying symmetry
# operations and phase factors.
# This is done in 3 steps:
# First solve the adjoint problem to get the u(s) on the fundamental domain.
# Evaluate the layer potential at any point in the billiard to get the wavefunction ψ(x,y).
# Project via symmetry in a given irrep onto the full boundary the u(s).
@time "layer density DLP Kress" us_all_dlp_kress,pts_all_dlp_kress=solve_vect(solver_dlp_kress,billiard,AbstractHankelBasis(),ks_dlp_kress)
@time "symmetrize layer density DLP Kress" pts_all_dlp_kress,us_all_dlp_kress=symmetrize_layer_density(solver_dlp_kress,us_all_dlp_kress,pts_all_dlp_kress,billiard)
@time "boundary function construction DLP Kress" pts_bdry_dlp_kress,u_bdry_dlp_kress=boundary_function(solver_dlp_kress,us_all_dlp_kress,pts_all_dlp_kress,billiard,ks_dlp_kress)

@time "layer density BIM" us_all_bim,pts_all_bim=solve_vect(solver_bim,billiard,AbstractHankelBasis(),ks_bim)
@time "symmetrize layer density BIM" pts_all_bim,us_all_bim=symmetrize_layer_density(solver_bim,us_all_bim,pts_all_bim,billiard)
@time "boundary function construction BIM" pts_bdry_bim,u_bdry_bim=boundary_function(solver_bim,us_all_bim,pts_all_bim,billiard,ks_bim)



# Reconstruct real-space wavefunctions from boundary layer densities.
#
# inside_only=false:
#   evaluate also outside the billiard
#   useful for visually checking boundary accuracy
#
# fundamental=false:
#   reconstruct full billiard wavefunctions
#
Psi2ds_dlp_kress,x_grid_dlp_kress,y_grid_dlp_kress=wavefunction_multi(solver_dlp_kress,ks_dlp_kress,us_all_dlp_kress,pts_all_dlp_kress,billiard;inside_only=false,fundamental=false)

Psi2ds_bim,x_grid_bim,y_grid_bim=wavefunction_multi(solver_bim,ks_bim,us_all_bim,pts_all_bim,billiard;inside_only=false,fundamental=false)

# plot the wavefunctions, should be identical if everything worked correctly, and also check visually if they look like eigenfunctions of the billiard
fig1=plot_wavefunctions(ks_dlp_kress,Psi2ds_dlp_kress,x_grid_dlp_kress,y_grid_dlp_kress,billiard;fundamental=false)
save("dlp_kress_wavefunctions_beyn.png",fig1[1])

fig2=plot_wavefunctions(ks_bim,Psi2ds_bim,x_grid_bim,y_grid_bim,billiard;fundamental=false)
save("bim_wavefunctions_beyn.png",fig2[1])