using QuantumBilliards
using LinearAlgebra
using CairoMakie

try_MKL!()

billiard,_=make_teardrop_and_basis() # the infamous kress teardrop from this paper: "Boundary Integral Equations in time-harmonic accoustic scattering (1991). A difficult geometry since it has a corner but Kress introduced a grading that effectively achieves a high order convergence

b=15.0
symmetry=nothing # For beyn we can set this to YReflection(-1) to check only eigvals and eigenfunctions of the odd teardrop. But for this example there is no need!
solver=CFIE_kress_corners(b,billiard,symmetry=symmetry,kressq=4) # the Kress CFIE solver for a single corners (NOT multi corners, that one is CFIE_kress_global_corners), with a grading parameter of 4 (the higher the more points are clustered near the corner, but the more ill conditioned the system becomes). 

k1=20.0
k2=22.0
step=1e-3
kgrid=collect(k1:step:k2)
tens=k_sweep(solver, 
    AbstractHankelBasis(), # placeholder basis for all BIE methods
    billiard, # the teardrop billiard geometry
    kgrid, # ks for which the tensions will be computed
    multithreaded_matrices=true, # allow threaded matrix assembly for the sweep for each k
    use_krylov=false, # use the shift-invert Krylov method to find the smallest singular value and vector at each k instead of the full dense SVD; 
    # this is much faster for large problems but not that useful for small and intermediate k-regime
    which=:svd,
    # :svd for smallest singular value, :det_argmin for the min(|det(A)|), :det gives back det(A) which is a complex number to compare if the roots for both coincide (almost impossible for naive BIM)
) 

sols,tens_refined,histories=refine_minima(solver, # like above
    AbstractHankelBasis(), # like above
    billiard, # like above
    kgrid, # the same kgrid to check for minima, but the refinement will be done in a small neighborhood around the minima found in the initial sweep, so the final refined ks will not be on this grid
    tens, # the tensions computed on the above kgrid, used to find the minima to refine around
    multithreaded_matrices=true, # like above
    threshold=200.0, # threshold for what counts as a local minima in log scale
    print_refinement=true, # print the refined minima and their tensions to the console
    use_krylov=false, # like above,
    which=:svd, # like above, 
    pts_refinement_factors=(1.0,1.5,2.0,3.0), # the number of collocation points will be refined by these factors around the minima found in the initial sweep; for example if the initial sweep was done with b ppw scaling, the refinement will be done with 1.5b, 2b, and 3b ppw scaling to check for convergence of the refined minima
    dim_refinement_factors=(1.0,1.1,1.25,1.5) # same logic just with basis size if it is a basis type method
)

# Inf means that the gesvd crashed becuase we are too close to a true eigenvalue where the matrix is singular, but this is actually a good sign that we are close to a true root.

# Check with Beyn the accuracy of the refined minima if they agree
# The im part of Beyn is a good proxy for the accuracy of the eigenvalue (heuristic)
ks,tens,us,pts,tensN=compute_spectrum_beyn(
    solver,                          # solver defining the boundary operator T(k)
    billiard,                        # billiard used for Weyl planning and point evaluation
    k1,                              # lower scan bound
    k2;                              # upper scan bound
    m=50,                            # target number of levels per planned window
    Rmax=0.7,                        # cap on contour radius
    nq=45,                           # number of contour nodes per disk
    r=100,                           # Beyn probe rank / number of random test vectors
    svd_tol=1e-12,                   # SVD rank-detection threshold
    res_tol=1e-9,                    # residual tolerance for filtering roots
    auto_discard_spurious=true,      # whether to remove large-residual roots
    multithreaded_matrix=true,       # matrix assembly threading flag
    use_chebyshev=true,             # use Chebyshev Hankel interpolation for complex contour evals (small problem, but still currently only supported path)
)

# Compare the eigenvalues found by Beyn with the refined minima from the sweep; they should all be close to each other if the refinement worked well and Beyn found the correct roots
for (i,sol) in enumerate(sols)
    beyn_root=ks[argmin(abs.(ks.-sols[i]))]
    println("Difference between refined minimum and Beyn root: $(abs(sols[i]-beyn_root))")
    println("--------------------------------------------------")
end