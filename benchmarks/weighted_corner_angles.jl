#!/usr/bin/env julia
# Independent PPW convergence test for the weighted physical normal density.
# The radius-one circular sector is separable: k_{m,n}=j_{m*pi/alpha,n}.
# Consequently neither discretization supplies its own reference eigenvalues.
# Each sector has one corner alpha at the origin and two pi/2 arc joins.
# The boundary is built entirely from BilliardGeometry's existing primitives;
# no angles, corner powers or singular coefficients are supplied to the solver.
#
# Usage: julia --project -t8 benchmarks/weighted_corner_angles.jl [output_directory]
# ANGLE_BENCH_PPWS=6,8,10,12,15,18,24
# ANGLE_BENCH_ANGLES=1/3,1/2,2/3,3/4,4/3,3/2,7/4 (multiples of pi)
# ANGLE_BENCH_KINDS=weighted,legacy_q4
# ANGLE_BENCH_MODE=geometry prints only the geometry/exact-spectrum audit.
# ANGLE_BENCH_SOURCE_AUDIT=true repeats weighted ppw12 with twice the source nodes.
# ANGLE_BENCH_TIMING_ISOLATED=true may be set only without competing CPU jobs.
# ANGLE_BENCH_BASELINE=/path/to/an/earlier/run also compares every raw root.

using QuantumBilliards, BilliardGeometry, LinearAlgebra, Serialization, Printf, SHA, Random

const ANGLE_QB = QuantumBilliards
const ANGLE_OUTPUT = isempty(ARGS) ? joinpath(tempdir(),"QuantumBilliards-weighted-corner-angles") : abspath(ARGS[1])
const ANGLE_PPWS = parse.(Float64,split(get(ENV,"ANGLE_BENCH_PPWS","6,8,10,12,15,18,24"),','))
const ANGLE_FRACTIONS = [Tuple(parse.(Int,split(value,'/'))) for value in split(get(ENV,
    "ANGLE_BENCH_ANGLES","1/3,1/2,2/3,3/4,4/3,3/2,7/4"),',')]
const ANGLE_KINDS = Symbol.(split(get(ENV,"ANGLE_BENCH_KINDS","weighted,legacy_q4"),','))
const ANGLE_DIAMETER = 0.4
const ANGLE_RADIAL_INDEX = 9
const ANGLE_NQ = 40
const ANGLE_TAYLOR_DEGREE = 18
const ANGLE_SEED = 24301
const ANGLE_SOURCE_AUDIT = get(ENV,"ANGLE_BENCH_SOURCE_AUDIT","true") == "true"
const ANGLE_ISOLATED = get(ENV,"ANGLE_BENCH_TIMING_ISOLATED","false") == "true"
const ANGLE_SIGNATURE = bytes2hex(sha256(read(@__FILE__,String)*join(read(file,String) for
    file in sort([joinpath(root,name) for (root,_,names) in walkdir(dirname(pathof(ANGLE_QB)))
        for name in names if endswith(name,".jl")]))))

"""
    CornerSectorBilliard(alpha::Float64) -> CornerSectorBilliard{Float64}

Benchmark-only Dirichlet sector `0 < r < 1`, `0 < theta < alpha`. The ordered
counterclockwise physical boundary contains the outward radial wall, the circle
arc and the inward radial wall. There are no symmetry or transparent walls.
Its exact separated eigenfunctions are `J_{m*pi/alpha}(k*r)*sin(m*pi*theta/alpha)`.
The origin angle is `alpha`; both remaining physical corners have angle `pi/2`.
"""
struct CornerSectorBilliard{T<:Real} <: BilliardGeometry.AbsBilliard
    fundamental_domain::BilliardGeometry.SimpleDomain{T}
    symmetries::BilliardGeometry.SymmetryRegistry
end

function CornerSectorBilliard(alpha::Float64)
    0 < alpha < 2pi || throw(ArgumentError("the sector opening must lie in (0,2pi)"))
    origin = ANGLE_QB.SVector{2,Float64}(0,0)
    right = ANGLE_QB.SVector{2,Float64}(1,0)
    left = ANGLE_QB.SVector{2,Float64}(cos(alpha),sin(alpha))
    curves = BilliardGeometry.AbsCurve[
        LineSegment(origin,right;segment_id=1),
        CircleSegment(1.0,alpha,0.0,origin;segment_id=2),
        LineSegment(left,origin;segment_id=3)]
    domain = BilliardGeometry.SimpleDomain{Float64}(curves,[origin,right,left],1)
    return CornerSectorBilliard(domain,register_symmetries())
end

"""
    angle_bessel_zeros(order::Float64, upper::Float64) -> Vector{Float64}

Find every positive zero of `J_order(x)` below `upper` by sign bracketing on
a 0.05 grid, followed by bisection to adjacent Float64 numbers. The tested
orders are positive and adjacent zeros are separated by much more than 0.05.
The bracket count is also checked with a half-step scan before an exact
reference is accepted. No root of a boundary matrix enters this oracle.
"""
function angle_bessel_zeros(order::Float64,upper::Float64;step::Float64=0.05)
    roots = Float64[]
    left = min(step/2,1e-4)
    value = ANGLE_QB.SpecialFunctions.besselj(order,left)
    while left < upper
        right = min(left+step,upper)
        other = ANGLE_QB.SpecialFunctions.besselj(order,right)
        if signbit(value) != signbit(other) || other == 0
            lo,hi = left,right
            vlo = value
            for _ in 1:100
                middle = (lo+hi)/2
                (middle == lo || middle == hi) && break
                vmiddle = ANGLE_QB.SpecialFunctions.besselj(order,middle)
                if signbit(vlo) == signbit(vmiddle)
                    lo = middle; vlo = vmiddle
                else
                    hi = middle
                end
            end
            push!(roots,(lo+hi)/2)
        end
        left = right; value = other
    end
    return roots
end

function angle_exact_spectrum(alpha::Float64)
    lambda = pi/alpha
    ground = angle_bessel_zeros(lambda,70.0)
    length(ground) >= ANGLE_RADIAL_INDEX || error("the anchor radial zero was not bracketed")
    center = ground[ANGLE_RADIAL_INDEX]
    lower,upper = center-ANGLE_DIAMETER/2,center+ANGLE_DIAMETER/2
    roots = NamedTuple{(:k,:angular_index,:radial_index,:order),Tuple{Float64,Int,Int,Float64}}[]
    # The first positive zero of J_nu is greater than nu for nu>0. Therefore
    # angular orders above upper cannot contribute any missing eigenvalue.
    for angular in 1:floor(Int,upper/lambda)
        order = angular*lambda
        zeros = angle_bessel_zeros(order,upper)
        check = angle_bessel_zeros(order,upper;step=0.025)
        length(zeros) == length(check) && all(isapprox.(zeros,check;atol=8eps(upper),rtol=0)) ||
            error("Bessel reference scan is inconsistent at order $order")
        for (radial,k) in enumerate(zeros)
            lower < k < upper && push!(roots,(;k,angular_index=angular,radial_index=radial,order))
        end
    end
    sort!(roots;by=root->root.k)
    isempty(roots) && error("the anchor zero is absent from the exact spectrum")
    return (;center,roots)
end

function angle_benchmark_case(numerator::Int,denominator::Int,kind::Symbol,ppw::Float64,
    reference::NamedTuple;source_refinement::Int=1)
    label="angle$(numerator)_$(denominator)_$(kind)_ppw$(ppw)_sources$(source_refinement)"
    saved=joinpath(ANGLE_OUTPUT,label*".jls")
    if isfile(saved)
        old=deserialize(saved)
        get(old,:source_signature,"")==ANGLE_SIGNATURE && get(old,:timing_isolated,false)==ANGLE_ISOLATED && return old
    end
    println("Starting ",label); flush(stdout)
    alpha=pi*numerator/denominator
    seed=ANGLE_SEED+1000numerator+100denominator+round(Int,ppw)
    Random.seed!(seed)
    billiard=CornerSectorBilliard(alpha)
    roots=ComplexF64[]; residuals=Float64[]; nodes=0; actual_ppw=NaN
    taylor_degree=0
    preparation=NaN; solve_seconds=NaN; message=""; status="error"
    started=time_ns()
    try
        defaults=WeightedCornerGrading()
        grading=kind == :weighted ? (source_refinement == 1 ? defaults : WeightedCornerGrading(
            regular_integration_nodes=source_refinement*defaults.regular_integration_nodes,
            corner_integration_nodes=source_refinement*defaults.corner_integration_nodes)) :
            GlobalCornerGrading(kressq=4)
        kernel=CombinedFieldIntegralEquationSolver(ppw,billiard;min_pts=64,grading=grading)
        solver=BeynSolver(kernel;nq=ANGLE_NQ,r=24,Rmax=0.5,
            use_taylor=true,use_chebyshev=false,taylor_degree=ANGLE_TAYLOR_DEGREE)
        points=evaluate_points(kernel,billiard,reference.center+ANGLE_DIAMETER/2)
        nodes=length(points)
        actual_ppw=2pi*nodes/((reference.center+ANGLE_DIAMETER/2)*(alpha+2))
        preparation=(time_ns()-started)*1e-9
        solve_started=time_ns()
        roots,residuals,_=solve_vectors(solver,points,reference.center,ANGLE_DIAMETER)
        solve_seconds=(time_ns()-solve_started)*1e-9
        taylor_degree=solver.taylor_degree
        permutation=sortperm(real.(roots))
        roots=ComplexF64.(roots[permutation]); residuals=Float64.(residuals[permutation])
        status=length(roots) == length(reference.roots) ? "complete" : "count_mismatch"
    catch err
        message=sprint(showerror,err,catch_backtrace())
        println(stderr,message)
    end
    exact=[root.k for root in reference.roots]
    comparable=status == "complete" && !isempty(exact)
    record=(;label,angle_numerator=numerator,angle_denominator=denominator,
        angle_pi=numerator/denominator,corner_lambda=denominator/numerator,kind,requested_ppw=ppw,
        actual_ppw,nodes,center=reference.center,expected_count=length(exact),found_count=length(roots),status,
        max_root_error=comparable ? maximum(abs,roots-exact) : NaN,
        max_real_drift=comparable ? maximum(abs,real.(roots)-exact) : NaN,
        max_imaginary_part=isempty(roots) ? NaN : maximum(abs,imag.(roots)),
        max_residual=isempty(residuals) ? NaN : maximum(residuals),preparation_seconds=preparation,
        solve_seconds,total_seconds=(time_ns()-started)*1e-9,taylor_degree,source_refinement,roots,residuals,
        timing_isolated=ANGLE_ISOLATED,source_signature=ANGLE_SIGNATURE,seed,message)
    serialize(saved,record)
    println("Finished ",label," ",status," N=",nodes," roots=",length(roots),
        " exact_error=",record.max_root_error," solve_seconds=",solve_seconds)
    flush(stdout); GC.gc()
    return record
end

angle_csv(value) = value isa AbstractString ? "\""*replace(value,"\""=>"\"\"")*"\"" : string(value)
function angle_write_tables(records)
    fields=(:angle_numerator,:angle_denominator,:angle_pi,:corner_lambda,:kind,:requested_ppw,
        :actual_ppw,:nodes,:center,:expected_count,:found_count,:status,:max_root_error,:max_real_drift,
        :max_imaginary_part,:max_residual,:preparation_seconds,:solve_seconds,:total_seconds,:taylor_degree,
        :source_refinement,:timing_isolated,:source_signature,:seed,:message)
    open(joinpath(ANGLE_OUTPUT,"convergence.csv"),"w") do io
        println(io,join(fields,','))
        for result in records
            println(io,join((angle_csv(getproperty(result,field)) for field in fields),','))
        end
    end
    open(joinpath(ANGLE_OUTPUT,"roots.csv"),"w") do io
        println(io,"angle_numerator,angle_denominator,kind,requested_ppw,source_refinement,root_index,real_k,imaginary_k,residual")
        for result in records, index in eachindex(result.roots)
            println(io,join((result.angle_numerator,result.angle_denominator,result.kind,result.requested_ppw,
                result.source_refinement,index,real(result.roots[index]),imag(result.roots[index]),result.residuals[index]),','))
        end
    end
    open(joinpath(ANGLE_OUTPUT,"source_quadrature.csv"),"w") do io
        println(io,"angle_numerator,angle_denominator,requested_ppw,nodes,base_count,refined_count,expected_count,status,max_root_drift,base_exact_error,refined_exact_error")
        for fine in records
            fine.kind == :weighted && fine.source_refinement == 2 || continue
            base=findfirst(result->result.kind == :weighted && result.source_refinement == 1 &&
                result.angle_numerator == fine.angle_numerator && result.angle_denominator == fine.angle_denominator &&
                result.requested_ppw == fine.requested_ppw,records)
            base === nothing && continue
            ordinary=records[base]
            complete=ordinary.status == fine.status == "complete"
            drift=complete ? maximum(abs,ordinary.roots-fine.roots) : NaN
            println(io,join((fine.angle_numerator,fine.angle_denominator,fine.requested_ppw,fine.nodes,
                ordinary.found_count,fine.found_count,fine.expected_count,complete ? "complete" : "count_mismatch",
                drift,ordinary.max_root_error,fine.max_root_error),','))
        end
    end
end

function angle_compare_previous(records,directory::String)
    open(joinpath(ANGLE_OUTPUT,"optimization_comparison.csv"),"w") do io
        println(io,"angle_numerator,angle_denominator,kind,requested_ppw,source_refinement,nodes,before_count,after_count,expected_count,status,max_complex_root_change,before_source_signature,after_source_signature")
        for result in records
            file=joinpath(directory,result.label*".jls")
            isfile(file) || error("missing comparison case: $file")
            old=deserialize(file)
            old.center == result.center && old.nodes == result.nodes && old.expected_count == result.expected_count ||
                error("comparison case changed its contour or density mesh: $(result.label)")
            same_count=old.found_count == result.found_count
            drift=same_count && !isempty(result.roots) ? maximum(abs,old.roots-result.roots) : NaN
            println(io,join((result.angle_numerator,result.angle_denominator,result.kind,result.requested_ppw,
                result.source_refinement,result.nodes,old.found_count,result.found_count,result.expected_count,
                same_count ? "same_count" : "count_changed",drift,old.source_signature,result.source_signature),','))
        end
    end
end

mkpath(ANGLE_OUTPUT)
all(kind->kind in (:weighted,:legacy_q4),ANGLE_KINDS) || error("unknown grading kind")
references=Dict{Tuple{Int,Int},NamedTuple}()
open(joinpath(ANGLE_OUTPUT,"exact_roots.csv"),"w") do io
    println(io,"angle_numerator,angle_denominator,corner_lambda,center,root_index,exact_k,angular_index,radial_index,bessel_order")
    for (numerator,denominator) in ANGLE_FRACTIONS
        alpha=pi*numerator/denominator
        reference=angle_exact_spectrum(alpha)
        references[(numerator,denominator)]=reference
        points=evaluate_points(CombinedFieldIntegralEquationSolver(6.0,CornerSectorBilliard(alpha);
            min_pts=64,grading=WeightedCornerGrading()),CornerSectorBilliard(alpha),reference.center+ANGLE_DIAMETER/2)
        lambdas=sort!(unique(round(panel.lambda;digits=12) for panel in points.density_space.panels if panel.lambda != 1))
        expected_lambdas=sort!(unique(round.([pi/alpha,2.0];digits=12)))
        lambdas == expected_lambdas || error("automatic tangent angles are wrong: $lambdas != $expected_lambdas")
        @printf("Sector %d/%d pi: alpha=%.8f, automatically detected lambdas=%s, k0=%.12f, exact count=%d, N6=%d\n",
            numerator,denominator,alpha,string(lambdas),reference.center,length(reference.roots),length(points))
        for (index,root) in enumerate(reference.roots)
            println(io,join((numerator,denominator,denominator/numerator,reference.center,index,
                root.k,root.angular_index,root.radial_index,root.order),','))
        end
    end
end
open(joinpath(ANGLE_OUTPUT,"settings.txt"),"w") do io
    println(io,"Independent circular-sector Dirichlet spectrum; radius R=1.")
    println(io,"Origin angle alpha; the two radial/arc joins have angle pi/2.")
    println(io,"Exact k=j_(m*pi/alpha,n), with every m,n in the contour retained.")
    println(io,"Contour center j_(pi/alpha,9), diameter=",ANGLE_DIAMETER,"; nq=",ANGLE_NQ,
        "; initial Taylor degree=",ANGLE_TAYLOR_DEGREE,"; all automatic validation remains enabled.")
    println(io,"Requested ppw=",ANGLE_PPWS,"; kinds=",ANGLE_KINDS,"; source audit=",ANGLE_SOURCE_AUDIT)
    println(io,"Weighted corner panel floors can exceed requested ppw; actual node counts/ppw are recorded.")
    println(io,"No real-axis projection or deduplication; count mismatches are failures.")
    println(io,"Bessel reference sign scans step 0.05 and 0.025 agree; Float64 bisection.")
    println(io,"Julia=",VERSION,"; threads=",Threads.nthreads(),"; BLAS=",BLAS.get_config(),
        "; initial BLAS threads=",BLAS.get_num_threads(),"; timing isolated=",ANGLE_ISOLATED)
    println(io,"QuantumBilliards=",pathof(ANGLE_QB),"; BilliardGeometry=",pathof(BilliardGeometry))
    println(io,"Source signature=",ANGLE_SIGNATURE)
    println(io,"Random seed per case=",ANGLE_SEED,"+1000*numerator+100*denominator+round(ppw).")
end
get(ENV,"ANGLE_BENCH_MODE","solve") == "geometry" && exit()

# A smallest-mesh warmup compiles the complete public path without entering
# convergence/runtime tables. Timings below include fresh source-cache setup,
# but are labelled isolated only when requested by the benchmark caller.
warm_fraction=first(ANGLE_FRACTIONS)
warm_reference=references[warm_fraction]
for kind in ANGLE_KINDS
    println("Warmup ",kind); flush(stdout)
    billiard=CornerSectorBilliard(pi*warm_fraction[1]/warm_fraction[2])
    kernel=CombinedFieldIntegralEquationSolver(6.0,billiard;min_pts=64,
        grading=kind == :weighted ? WeightedCornerGrading() : GlobalCornerGrading(kressq=4))
    points=evaluate_points(kernel,billiard,warm_reference.center+ANGLE_DIAMETER/2)
    solver=BeynSolver(kernel;nq=ANGLE_NQ,r=24,Rmax=0.5,use_taylor=true,
        use_chebyshev=false,taylor_degree=ANGLE_TAYLOR_DEGREE)
    solve_vectors(solver,points,warm_reference.center,ANGLE_DIAMETER)
end

records=NamedTuple[]
for (numerator,denominator) in ANGLE_FRACTIONS
    reference=references[(numerator,denominator)]
    for ppw in ANGLE_PPWS,kind in ANGLE_KINDS
        push!(records,angle_benchmark_case(numerator,denominator,kind,ppw,reference))
        angle_write_tables(records)
    end
    if ANGLE_SOURCE_AUDIT && :weighted in ANGLE_KINDS
        12.0 in ANGLE_PPWS || push!(records,angle_benchmark_case(numerator,denominator,:weighted,12.0,reference))
        push!(records,angle_benchmark_case(numerator,denominator,:weighted,12.0,reference;source_refinement=2))
        angle_write_tables(records)
    end
end
if haskey(ENV,"ANGLE_BENCH_BASELINE")
    angle_compare_previous(records,abspath(ENV["ANGLE_BENCH_BASELINE"]))
end
println("Saved exact spectra, all returned complex roots and convergence tables in ",ANGLE_OUTPUT)
