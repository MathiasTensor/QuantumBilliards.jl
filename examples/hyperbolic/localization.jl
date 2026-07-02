using QuantumBilliards
using LinearAlgebra
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

# Perturbation of the Schmit triangle.
const ϵ=1e-5

# Billiard geometry.
const billiard=GenericHyperbolicTriangle(ϵ=ϵ)

# Basis object used by compute_spectrum_hyp.
# It has a legacy name from Euclidean computations.
const basis=QuantumBilliards.AbstractHankelBasis()

# No symmetry reduction.
const symmetry=nothing # this one has no simple symmetry

# Boundary discretization points-per-wavelength parameter.
const b=8.0

# Beyn contour parameters.
const nq=40
const m=500
const r=550
const Rmax=0.5
const res_tol=1e-9

# Husimi grid resolution.
const np_hus=1000

# Number of eigenstates used near each target k.
const Nsample=2000

# Number of states processed at once for Husimi entropy.
const Nchunk=1000

# Target k-centers.
const k_start=10^3.0
const k_end=10^4.0
const Npoints=20
const target_ks=collect(range(k_start,k_end;length=Npoints))

# Output directory.
const outdir="LOCALIZATION/eps_$(ϵ)_k_$(round(Int,k_start))_$(round(Int,k_end))"
mkpath(outdir)

# Geometry constants for Weyl estimates.
const A=billiard.area
const L=billiard.length

# ==============================================================================
# SOLVER
# ==============================================================================

# Hyperbolic double-layer Kress solver with global corner grading.
const solver=QuantumBilliards.DLP_hyperbolic_kress_global_corners(
    b,billiard;
    symmetry=symmetry,
    min_pts=200,
    kressq=2,
)

# ==============================================================================
# BASIC UTILITIES
# ==============================================================================

# Participation-type Husimi entropy: exp(S)/Ngrid, S=-sum P log P.
function husimi_entropy_A(H)
    P=abs.(H) # all should be positive, but just in case
    s=sum(P) # normalization factor, since analytic one is a bit tricky
    s==0&&return NaN # should never be zero, but just in case
    P./=s # normalize to sum=1
    S=0.0 # accumualtion of the entropy sum
    @inbounds for x in P
        x>0&&(S-=x*log(x)) # only add positive contributions, with rule 0*log(0)=0
    end
    return exp(S)/length(P)
end

# Hyperbolic Weyl counting function for (Δ_H+1/4+k^2)ψ=0.
@inline function Nweyl(k)
    λ=k^2+0.25
    return A/(4π)*λ-L/(4π)*sqrt(λ)+1
end

# Weyl density dN/dk, used to estimate interval widths.
@inline function density_weyl(k)
    λ=k^2+0.25
    return A/(2π)*k-L/(4π)*k/sqrt(λ)
end

# Estimate a k-window around target center kc.
function target_interval(kc)
    ρ=density_weyl(kc)
    Δk=1.35*Nsample/ρ # large oversamplig factor for Beyn contour siue 1.35, can decrease to 1.1 if needed
    return kc-Δk/2,kc+Δk/2
end

# Marker file for completed intervals.
done_file(k1,k2)=joinpath(outdir,@sprintf("DONE_%.6f_%.6f.csv",k1,k2))

# Output file for one PH-entropy chunk.
function chunk_file(kc,k1,k2,chunk_id)
    return joinpath(outdir,@sprintf("PH_entropy_kcenter_%.6f_interval_%.6f_%.6f_chunk%03d.csv",kc,k1,k2,chunk_id))
end

# Print interval diagnostic information.
function print_interval_header(kc,k1,k2)
    println()
    println("="^100)
    @printf("Target k center ≈ %.10f\n",kc)
    @printf("Predicted N(k center) ≈ %.2f\n",Nweyl(kc))
    @printf("Interval [%.10f, %.10f], width %.10f\n",k1,k2,k2-k1)
    println("="^100)
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

# Compute all eigenpairs in [k1,k2].
function compute_interval_spectrum(k1,k2)
    return QuantumBilliards.compute_spectrum_hyp(
        solver,basis,billiard,k1,k2;
        m=m,
        r=r,
        Rmax=Rmax,
        nq=nq,
        res_tol=res_tol,
        origin_vertex_tol=1e-3, # filthy hack to avoid singularity at origin vertex for area estimate, but here we dont need it anyway
        return_imag_part=true,
        use_imag_residual_check=true, # trick to avoid cosnturcting Fredholm amtrix for each eigval check
        adjoint_mode=:direct, # direct adjoint mode is better for use since it gets us the normal derivative of the wavefunction
        # directly. If not :direct, we just get the boundary density, which is good for eigenfunction construction
        do_INFO=false,
    )
end

# Keep the Nsample states closest to kc, then order them by real k.
function select_states_near_center(ks_all,tens_all,us_all,pts_all,tensN_all,kc)
    nlev=length(ks_all)
    kvals=real.(ks_all)
    ord=sortperm(abs.(kvals.-kc))
    keep=ord[1:min(Nsample,nlev)]
    keep=sort(keep;by=i->kvals[i])
    return (ks=ks_all[keep],tens=tens_all[keep],us=us_all[keep],pts=pts_all[keep],tensN=tensN_all[keep])
end

# Print selected-state information.
function print_kept_state_info(ks)
    kre=real.(ks)
    @printf("Keeping %d states\n",length(ks))
    @printf("k kept %.10f to %.10f\n",minimum(kre),maximum(kre))
    @printf("Nweyl kept %.2f to %.2f\n",Nweyl(minimum(kre)),Nweyl(maximum(kre)))
end

# ==============================================================================
# HUSIMI ENTROPY COMPUTATION
# ==============================================================================

# Compute PH entropy for one chunk and write the result to disk.
function process_chunk!(kc,k1,k2,data,c0,c1)
    chunk_id=cld(c0,Nchunk)
    outfile=chunk_file(kc,k1,k2,chunk_id)

    # Skip already completed chunks.
    if isfile(outfile)
        println("Skipping existing $outfile")
        return nothing
    end

    # Extract chunk.
    println()
    @printf("Chunk %d: states %d:%d\n",chunk_id,c0,c1)
    inds=c0:c1
    ksc=data.ks[inds]
    tensc=data.tens[inds]
    usc=data.us[inds]
    ptsc=data.pts[inds]
    tensNc=data.tensN[inds]

    # Convert boundary representation to hyperbolic boundary phase-space data.
    bps=[QuantumBilliards.hyp_bp(p) for p in ptsc]

    # Build q,p grids adapted to the current chunk.
    qs,ps=QuantumBilliards.make_qp_grids_hyp(real.(ksc),bps;np=np_hus,full_p=false)

    # Evaluate Husimi densities on the grid.
    Hs=QuantumBilliards.husimi_on_grid_hyp(real.(ksc),bps,usc,qs,ps;full_p=false)

    # Assemble output rows.
    rows=NamedTuple[]
    @showprogress desc="PH entropy" for j in eachindex(ksc)
        push!(rows,(
            k_center=kc,
            interval_start=k1,
            interval_end=k2,
            local_level=c0+j-1,
            k_real=real(ksc[j]),
            k_imag=imag(ksc[j]),
            Nweyl=Nweyl(real(ksc[j])),
            tension=tensc[j],
            tensionN=tensNc[j],
            entropy_A=husimi_entropy_A(Hs[j]),
        ))
    end

    # Write chunk output.
    CSV.write(outfile,DataFrame(rows))
    println("Wrote: ",abspath(outfile))

    # Release large chunk arrays.
    Hs=nothing
    qs=nothing
    ps=nothing
    bps=nothing
    cleanup!()
    return nothing
end

# Process all chunks for one selected group of eigenstates.
function process_all_chunks!(kc,k1,k2,data)
    n=length(data.ks)
    for c0 in 1:Nchunk:n
        c1=min(c0+Nchunk-1,n)
        process_chunk!(kc,k1,k2,data,c0,c1)
    end
    return nothing
end

# Write a marker showing that the interval has been completed.
function write_done_marker(k1,k2)
    marker=done_file(k1,k2)
    CSV.write(marker,DataFrame(done=[true]))
    println("Wrote: ",abspath(marker))
    return nothing
end

# ==============================================================================
# MAIN LOOP
# ==============================================================================

# Process one target k-center.
function process_target_k(kc)
    # Build target interval.
    k1,k2=target_interval(kc)
    print_interval_header(kc,k1,k2)

    # Skip fully completed intervals.
    if isfile(done_file(k1,k2))
        println("Skipping finished interval.")
        return nothing
    end

    # Compute spectrum.
    ks_all,tens_all,us_all,pts_all,tensN_all=compute_interval_spectrum(k1,k2)
    cleanup!()

    # Skip empty intervals.
    nlev=length(ks_all)
    println("Found $nlev levels")
    nlev==0&&return nothing

    # Select states closest to the target center.
    data=select_states_near_center(ks_all,tens_all,us_all,pts_all,tensN_all,kc)
    print_kept_state_info(data.ks)

    # Compute PH entropy in chunks.
    process_all_chunks!(kc,k1,k2,data)

    # Mark interval complete.
    write_done_marker(k1,k2)

    # Release large interval arrays.
    ks_all=nothing
    tens_all=nothing
    us_all=nothing
    pts_all=nothing
    tensN_all=nothing
    data=nothing
    cleanup!()
    return nothing
end

# Run all target k-centers.
function main()
    println("Starting hyperbolic PH entropy computation.")
    println("eps = $ϵ")
    println("output directory = ",abspath(outdir))
    println("Nsample = $Nsample")
    println("Nchunk = $Nchunk")
    println("np_hus = $np_hus")
    println("target k range = [$k_start, $k_end]")
    println("number of target points = $Npoints")

    for kc in target_ks
        process_target_k(kc)
    end

    println("Done.")
    return nothing
end

main()