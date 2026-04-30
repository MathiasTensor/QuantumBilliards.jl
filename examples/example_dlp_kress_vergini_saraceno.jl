using QuantumBilliards
using LinearAlgebra
using CairoMakie

try_accelerated_blas!()

billiard,basis=make_ellipse_and_basis(1.0,0.5) # easy corner free geometry to compare Beyn & VS & EBIM

d=10.0 # basis scaling parameter -> determines basis size oversampling 
b=15.0 # pts scaling parameter, basically points per wavelength

#symmetry=nothing
symmetry=XYReflection(-1,-1) # this decreases problem complexity since it reduces matrix size by 4x!

solver_dlp_kress=DLP_kress(b,billiard,symmetry=symmetry) # Kress on DLP kernel, should give very good accuracy om smooth periodic curves
solver_vs=VerginiSaraceno(d,b)# solver for the Vergini Saraceno scaling

k1=5.0 
k2=20.0

# test sweep grid to see if the accelerated solvers's solutions allign on the minima of the sweep
k1_t=15.0
k2_t=20.0

@time "BEYN" ks,tens,us,pts,tensN=compute_spectrum_beyn(
    solver_dlp_kress,                # solver defining the boundary operator T(k)
    billiard,                        # billiard used for Weyl planning and point evaluation
    k1,                              # lower scan bound
    k2;                              # upper scan bound
    m=100,                           # target number of levels per planned window
    Rmax=0.7,                        # cap on contour radius
    nq=45,                           # number of contour nodes per disk
    r=150,                           # Beyn probe rank / number of random test vectors
    svd_tol=1e-12,                   # SVD rank-detection threshold
    res_tol=1e-9,                    # residual tolerance for filtering roots
    auto_discard_spurious=true,      # whether to remove large-residual roots
    multithreaded_matrix=true,       # matrix assembly threading flag
    use_chebyshev=true,              # use Chebyshev Hankel interpolation for complex contour evals (small problem, but still currently only supported path)
)

println()
@info "Found Beyn eigvals: $(length(ks)) (should give 1/4 of all eigvals for ellipse since it is projected onto an odd-odd symmetry subspace in Beyn)"
println()

dk=0.05 # dk interval from which we choose the solutions after each GEVP (regularized EVP), it is a heuristic
@time "VerginiSaraceno" state_res,_=compute_spectrum_with_state_scaling_method(
    solver_vs,
    basis, # the RPW or CAFB basis (from make_ellipse_and_basis), with corner angle pi/2 at the origin
    billiard,
    k1,
    k2,
    dk, # heuristic dk interval, should not really be a constant, actually spectrum should be computed in chunks and checked if levels are missing in any
    multithreaded_matrices=true, # just leave on always, matrices should always be multithreaded for assembly
    multithreaded_ks=false, # fort large problems this is best left to false to avoid overusbscription
    cholesky=false, # EXPERIMENTAL: whether we can circumvent the expensive VS step of filtering out 
    # numerical nullspace of a large symmetric matrix 
)

println()
@info "Found Vergini Saraceno eigvals: $(length(state_res.ks)) (should give 1/4 of all eigvals for ellipse since it is a symmetry reduced quarter billiard)"
println()

# state_res is a StateData struct with fieds:
# ks: the eigenvalues found by the method
# tens: the tensions of the eigenvalues found by the method
# X: basis coefficients stacked as columns of a matrix (accesed by column index of the eigenvalue)

# It is useful to check the convergence of the basis via a simple sweep
# with either DM or PSM. For non-convex geometries at some point the method breaks!
# But ellipse is convex so not problem
# In other example files there are comments on DM method, so please check there
f=Figure()
ax=Axis(f[1,1])
solver_dm=DecompositionMethod(d,b)
kgrid=collect(k1_t:1e-3:k2_t) # just a small grid sample to check minimization param patterns
tens_dm=k_sweep(solver_dm,basis,billiard,kgrid;multithreaded_matrices=true)
lines!(ax,kgrid,log.(tens_dm))
# get all ks_vs ain this kgrid interval
ks_vs=state_res.ks[findall(x->k1_t<x<k2_t,state_res.ks)]
tens_vs=state_res.tens[findall(x->k1_t<x<k2_t,state_res.ks)]
scatter!(ax,ks_vs,log.(tens_vs),color=:red)
save("DM_test_sweep.png",f)

@time "EBIM" λs_ebim,tensions_ebim=compute_spectrum_ebim(
    solver_dlp_kress,
    billiard,
    k1,
    k2;
    dk=(k->0.05*k^(-1/3)), # heuristic dk interval for picking up solutions near the trial center k0 based on the OG one from Veble's monza billiard paper
    # better would be to make this one have a fixed numver of eigenvalues per interval, say 3-4
    use_krylov=true,
    use_chebyshev=false,
    n_panels=15000,
    M=5,
    cheb_tol=1e-13,
    multithreaded_matrices=true,
)

# deduplicate the eigenvalues since we did no desymmetrization for ebim (no projection 
# on desymmetrized subspace like Beyn or dedicated basis in the desymmetrized domain like VerginiSaraceno)
function unique_eigenvalues(ks,tens;tol=1e-4)
    isempty(ks) && return ks
    ks=sort(ks)
    tens=sort(tens)
    out_ks=[ks[1]]
    out_tens=[tens[1]]
    for i in 2:length(ks) 
        if abs(ks[i]-out_ks[end])>tol
            push!(out_ks,ks[i])
            push!(out_tens,tens[i])
        end
    end
    return out_ks,out_tens
end

λs_ebim,tensions_ebim=unique_eigenvalues(λs_ebim,tensions_ebim)

println()
@info "Found EBIM eigvals: $(length(λs_ebim))"
@info "Really hard to desymmetrize, check the DLP_sweep_comparison.png. Best used on geometries without symmetry or just use Beyn like above"
println()

f=Figure()
ax=Axis(f[1,1])
tens_dlp_kress=k_sweep(solver_dlp_kress,AbstractHankelBasis(),billiard,kgrid;multithreaded_matrices=true)
idxs=findall(x->k1_t<x<k2_t,λs_ebim) # get all λs_ebim in this kgrid interval
λs_ebim=λs_ebim[idxs]
tensions_ebim=tensions_ebim[idxs]
lines!(ax,kgrid,log.(tens_dlp_kress))
scatter!(ax,λs_ebim,log.(tensions_ebim),color=:blue)
save("DLP_sweep_comparison.png",f)