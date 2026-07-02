using QuantumBilliards
using LinearAlgebra
using CairoMakie
using Printf
using ProgressMeter
using DataFrames
using CSV

# Use MKL if available.
QuantumBilliards.try_MKL!()

# Geometry definitions.
include("generic_triangle.jl")

# ==============================================================================
# USER PARAMETERS
# ==============================================================================

# Perturbation of the generic hyperbolic triangle.
const ϵ=5e-5

# Billiard geometry.
const billiard=GenericHyperbolicTriangle(ϵ=ϵ)

# Basis object used by compute_spectrum_hyp.
# It has a legacy name from Euclidean computations.
const basis=QuantumBilliards.AbstractHankelBasis()

# Symmetry sector.
# Use nothing for the full nonsymmetric spectrum.
# Example:
# const symmetry=XYReflection(-1,-1)
const symmetry=nothing # this one has no simple symmetry

# Boundary points-per-wavelength parameter.
const b=8.0

# Beyn contour parameters.
const nq=40
const m=200
const r=250
const Rmax=0.5

# Residual and geometric tolerances.
const res_tol=1e-9 # residual tolerance for Beyn eigenvalue check (mostly used for INFO and if we check each one)
const origin_vertex_tol=1e-3 # internal geometry checks of billiards, if goes though origin vertex it becomes annoying, so slightly shift it. Does not influence accuracy
# Manual k-intervals to compute. 
const k_intervals=[(5.0,1000.0)]

# Output filename prefix.
const outprefix="hyperbolic_triangle_eps_$(ϵ)"

# ==============================================================================
# SOLVER
# ==============================================================================

# Hyperbolic double-layer Kress solver with global corner grading.
const solver=QuantumBilliards.DLP_hyperbolic_kress_global_corners(b,billiard;symmetry=symmetry,min_pts=200,kressq=2)

# Alternative solvers for playing around
# const solver=QuantumBilliards.DLP_hyperbolic_kress(b,billiard;symmetry=symmetry) # only if smooth boundary without corners
# const solver=QuantumBilliards.BIM_hyperbolic(b;symmetry=symmetry) # slow convergence O(N^3)
# const solver=QuantumBilliards.DLP_hyperbolic_log_product(b,billiard;symmetry=symmetry,ngl=64,near_panels=1) # Gauss-Legendre quadrature with log-product rule, high algebraic convergence

# ==============================================================================
# UTILITIES
# ==============================================================================

# Convert symmetry object to a filename-safe label.
function symmetry_label(symmetry)
    symmetry===nothing&&return "nosym"
    return replace(string(symmetry)," "=>"_","/"=>"_","\\"=>"_",":"=>"_")
end

# Output CSV filename for one interval.
function output_file(k1,k2)
    sym=symmetry_label(symmetry)
    return @sprintf("%s_%s_%.6f_%.6f.csv",outprefix,sym,k1,k2)
end

# Print a visible interval header.
function print_interval_header(k1,k2)
    println()
    println("="^80)
    @printf("Computing interval [%.10f, %.10f]\n",k1,k2)
    println("="^80)
end

# Force garbage collection twice after large arrays are released. 
# This is annoying but mpmath via PyCall has issues with memory fragmentation sometimes, so this is a workaround.
function cleanup!()
    GC.gc()
    GC.gc()
    return nothing
end

# ==============================================================================
# SPECTRUM COMPUTATION
# ==============================================================================

# Compute all eigenvalues/eigenvectors in one k-interval via Beyn
function compute_interval(k1,k2)
    return QuantumBilliards.compute_spectrum_hyp(
        solver,basis,billiard,k1,k2;
        m=m,
        r=r,
        Rmax=Rmax,
        nq=nq,
        res_tol=res_tol,
        origin_vertex_tol=origin_vertex_tol,
    )
end

# Save one interval to CSV.
function save_interval(k1,k2,ks_all,tens_all,tensN_all)
    df=DataFrame(
        k=ks_all,
        tension=tens_all,
        tensionN=tensN_all,
    )

    fname=output_file(k1,k2)
    CSV.write(fname,df)

    println("Saved $fname with $(length(ks_all)) levels")
    return nothing
end

# Process one k-interval.
function process_interval(k1,k2)
    print_interval_header(k1,k2)

    fname=output_file(k1,k2)
    if isfile(fname)
        println("Skipping existing file: $fname")
        return nothing
    end

    ks_all,tens_all,us_all,pts_all,tensN_all=compute_interval(k1,k2)

    save_interval(k1,k2,ks_all,tens_all,tensN_all)

    ks_all=nothing
    tens_all=nothing
    us_all=nothing
    pts_all=nothing
    tensN_all=nothing
    cleanup!()

    return nothing
end

# ==============================================================================
# MAIN LOOP
# ==============================================================================

function main()
    println("Starting selected-interval hyperbolic Beyn run.")
    println("eps = $ϵ")
    println("symmetry = $(symmetry_label(symmetry))")
    println("b = $b")
    println("nq = $nq, m = $m, r = $r, Rmax = $Rmax")
    println("number of intervals = $(length(k_intervals))")

    for (k1,k2) in k_intervals
        process_interval(k1,k2)
    end

    println("Done.")
    return nothing
end

main()