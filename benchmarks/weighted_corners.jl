#!/usr/bin/env julia
# Compare actual unknown counts and complex eigenvalue convergence for the
# physical mushroom boundary supplied by BilliardGeometry.  No transparent
# decomposition interfaces enter the boundary integral.  Both reflection
# parities are solved; roots are never projected onto the real axis or deduped.
#
# Usage: julia --project -t8 benchmarks/weighted_corners.jl [output_directory]
# Optional environment: CORNER_BENCH_PPWS=8,10,12; CORNER_BENCH_BANDS=30,100
# CORNER_BENCH_TIMING_ISOLATED=true labels timings only after the caller has
# stopped competing CPU work.  Ordinary runs explicitly label them contended.
# CORNER_BENCH_STATE_AUDIT=true additionally checks one difficult mode in each
# parity at ppw 10/12/15 and both references, using public solve_state recovery.
# Source and field refinement are measured separately from eigenvalue solves.
# Numerical references are refined weighted calculations, not exact solutions.

using QuantumBilliards, BilliardGeometry, LinearAlgebra, Serialization, Printf, SHA

const QB = QuantumBilliards
const OUTPUT = isempty(ARGS) ? joinpath(tempdir(),"QuantumBilliards-weighted-corners") : abspath(ARGS[1])
const PPWS = parse.(Float64,split(get(ENV,"CORNER_BENCH_PPWS","8,10,12,15,18"),','))
const CENTERS = parse.(Float64,split(get(ENV,"CORNER_BENCH_BANDS","30,100"),','))
const REFERENCE_PPWS = (24.0,30.0) # independent mesh increase; both retain the same boundary
const DIAMETER = 0.4             # radius 0.2 is below the production maximum 0.5
const CONTOUR_NODES = 40         # preserve the production contour quadrature
const TAYLOR_DEGREE = 18         # existing automatic validation can increase this
const BILLIARD = MushroomBilliard(0.4,1.0;R=1.0)
const TARGETS = vcat([QB.SVector{2,Float64}(r*cos(angle),r*sin(angle))
    for r in (0.35,0.7,0.9,0.97,0.99)
    for angle in (QB.FastGaussQuadrature.gausslegendre(9)[1].+1).*pi./2],
    [QB.SVector{2,Float64}(p) for p in ((0.2,-0.2),(-0.25,-0.4),(0.1,-0.65),(-0.3,-0.9))])
const ISOLATED = get(ENV,"CORNER_BENCH_TIMING_ISOLATED","false") == "true"
const SOURCE_SIGNATURE = bytes2hex(sha256(join(read(file,String) for
    file in sort([joinpath(root,name) for (root,_,names) in walkdir(joinpath(dirname(pathof(QB))))
        for name in names if endswith(name,".jl")]))))
mkpath(OUTPUT)
open(joinpath(OUTPUT,"field_targets.csv"),"w") do io
    println(io,"x,y")
    for point in TARGETS; println(io,point[1],',',point[2]); end
end
println("Julia ",VERSION,"; BLAS ",BLAS.get_config(),"; geometry ",pathof(BilliardGeometry))
println("Timing isolated: ",ISOLATED,". Counts and errors retain multiplicities.")

"""
    corner_benchmark_case(kind::Symbol, center::Float64, ppw::Float64,
        parity::Int; refined_sources::Bool=false) -> NamedTuple

Solve every accepted root of the contour `|k-center| <= DIAMETER/2` in one
reflection sector of the mushroom. `kind` selects `:weighted`, `:legacy_q2`
or `:legacy_q4`; `ppw` is the requested density resolution and `parity` is
`-1` or `1`. `refined_sources=true` doubles the weighted source orders while
retaining the same density space. The default nonlinear acceptance checks
remain enabled. Weighted fields use the physical normal density `q=∂ₙψ`.

Return an owned numerical record containing complex roots, residuals, actual
dimensions, field samples and timings, or a record of the unresolved error.
An existing record in `OUTPUT` is reused, so interrupted sweeps can resume.
"""
function corner_benchmark_case(kind::Symbol,center::Float64,ppw::Float64,parity::Int;refined_sources::Bool=false)::NamedTuple
    label="$(kind)_k$(center)_ppw$(ppw)_parity$(parity)"*(refined_sources ? "_sources2" : "")
    saved=joinpath(OUTPUT,label*".jls")
    if isfile(saved)
        cached=deserialize(saved)
        get(cached,:source_signature,"")==SOURCE_SIGNATURE &&
            get(cached,:timing_isolated,false)==ISOLATED && return cached
    end
    println("Starting ",label); flush(stdout)
    start=time()
    local result
    try
        grading = if kind == :weighted
            defaults=WeightedCornerGrading()
            refined_sources ? WeightedCornerGrading(
                regular_integration_nodes=2defaults.regular_integration_nodes,
                corner_integration_nodes=2defaults.corner_integration_nodes) : defaults
        else
            GlobalCornerGrading(kressq=kind == :legacy_q2 ? 2 : 4)
        end
        kernel=CombinedFieldIntegralEquationSolver(ppw,BILLIARD;
            min_pts=64,grading=grading,sym_characters=(parity,))
        solver=BeynSolver(kernel;nq=CONTOUR_NODES,r=24,Rmax=0.5,
            use_taylor=true,use_chebyshev=false,taylor_degree=TAYLOR_DEGREE)
        points=evaluate_points(kernel,BILLIARD,center+DIAMETER/2)
        preparation=time()-start; solve_start=time()
        roots,residuals,vectors=solve_vectors(solver,points,center,DIAMETER)
        solve_seconds=time()-solve_start
        order=sortperm(real.(roots)); roots=roots[order]; residuals=residuals[order]; vectors=vectors[:,order]
        field_start=time()
        fields=Matrix{ComplexF64}(undef,length(TARGETS),kind == :weighted ? length(roots) : 0)
        source_field_drift=kind == :weighted ? 0.0 : NaN
        if kind == :weighted
            for j in eachindex(roots)
                q=QB.symmetrize_layer_density(kernel,vectors[:,j],points,BILLIARD)
                fields[:,j]=QB._weighted_corner_potential(points,q,roots[j],TARGETS)
                checked=QB._weighted_corner_potential(points,q,roots[j],TARGETS;refinement=2)
                source_field_drift=max(source_field_drift,norm(fields[:,j]-checked)/norm(checked))
            end
        end
        perimeter=sum(curve.length for curve in full_boundary(BILLIARD))
        result=(;label,kind,center,ppw,parity,refined_sources,status="ok",roots,residuals,
            nodes=length(points),dimension=boundary_matrix_size(kernel,points),
            actual_ppw=2pi*length(points)/((center+DIAMETER/2)*perimeter),
            fields,source_field_drift,preparation_seconds=preparation,solve_seconds,
            field_seconds=time()-field_start,total_seconds=time()-start,
            taylor_degree=solver.taylor_degree,source_signature=SOURCE_SIGNATURE,timing_isolated=ISOLATED)
    catch err
        result=(;label,kind,center,ppw,parity,refined_sources,status="error",
            error=sprint(showerror,err,catch_backtrace()),total_seconds=time()-start,
            source_signature=SOURCE_SIGNATURE,timing_isolated=ISOLATED)
        println(stderr,result.error)
    end
    serialize(saved,result)
    println("Finished ",label," status=",result.status,
        result.status == "ok" ? " roots=$(length(result.roots)) N=$(result.nodes)" : "")
    flush(stdout)
    GC.gc()
    return result
end

results=Any[]
references=Dict{Tuple{Float64,Int},Any}()
reference_drift=Dict{Tuple{Float64,Int},Float64}()
for center in CENTERS, parity in (-1,1)
    coarse=corner_benchmark_case(:weighted,center,REFERENCE_PPWS[1],parity)
    fine=corner_benchmark_case(:weighted,center,REFERENCE_PPWS[2],parity;refined_sources=true)
    append!(results,(coarse,fine)); references[(center,parity)]=fine
    reference_drift[(center,parity)]=coarse.status == fine.status == "ok" &&
        length(coarse.roots)==length(fine.roots) && !isempty(fine.roots) ?
        maximum(abs.(coarse.roots-fine.roots)) : NaN
end
for center in CENTERS, ppw in PPWS, kind in (:weighted,:legacy_q2,:legacy_q4), parity in (-1,1)
    push!(results,corner_benchmark_case(kind,center,ppw,parity))
end
# Resolve the legacy high-band 1e-8 threshold as well; graded-grid convergence
# need not be monotone, so ppw 20 and 24 are both retained in the comparison.
if 100.0 in CENTERS
    for ppw in (20.0,24.0), parity in (-1,1)
        ppw in PPWS && continue
        push!(results,corner_benchmark_case(:legacy_q4,100.0,ppw,parity))
    end
end

open(joinpath(OUTPUT,"convergence.csv"),"w") do io
    println(io,"kind,k_center,requested_ppw,actual_ppw,parity,source_refinement,nodes,matrix_dimension,reference_count,found_count,status,max_root_difference,max_imaginary_part,max_residual,reference_drift,max_field_shape_difference,source_field_drift,preparation_seconds,solve_seconds,field_seconds,timing_isolated")
    for result in results
        result.status == "ok" || (println(io,"$(result.kind),$(result.center),$(result.ppw),NaN,$(result.parity),$(result.refined_sources ? 2 : 1),0,0,0,0,error,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,$(result.timing_isolated)"); continue)
        reference=references[(result.center,result.parity)]
        comparable=reference.status == "ok" && length(result.roots)==length(reference.roots) && !isempty(result.roots)
        error=comparable ? maximum(abs.(result.roots-reference.roots)) : NaN
        field_error=NaN
        if comparable && result.kind == :weighted
            field_error=0.0
            for j in eachindex(result.roots)
                a=result.fields[:,j]; b=reference.fields[:,j]
                # Least-squares complex alignment removes the arbitrary scale
                # and phase of a nonlinear eigenvector; this is a shape check.
                coefficient=dot(a,b)/dot(a,a)
                field_error=max(field_error,norm(coefficient*a-b)/norm(b))
            end
        end
        status=comparable ? "complete" : "count_mismatch"
        max_imag=isempty(result.roots) ? NaN : maximum(abs,imag.(result.roots))
        max_residual=isempty(result.residuals) ? NaN : maximum(result.residuals)
        expected=reference.status == "ok" ? length(reference.roots) : 0
        println(io,join((result.kind,result.center,result.ppw,result.actual_ppw,result.parity,
            result.refined_sources ? 2 : 1,result.nodes,result.dimension,expected,length(result.roots),status,
            error,max_imag,max_residual,reference_drift[(result.center,result.parity)],field_error,
            result.source_field_drift,result.preparation_seconds,result.solve_seconds,result.field_seconds,result.timing_isolated),','))
    end
end
open(joinpath(OUTPUT,"roots.csv"),"w") do io
    println(io,"kind,k_center,requested_ppw,parity,source_refinement,root_index,real_k,imaginary_k,residual")
    for result in results
        result.status == "ok" || continue
        for j in eachindex(result.roots)
            println(io,join((result.kind,result.center,result.ppw,result.parity,
                result.refined_sources ? 2 : 1,j,real(result.roots[j]),imag(result.roots[j]),result.residuals[j]),','))
        end
    end
end
println("Saved numerical references, raw complex roots and convergence tables in ",OUTPUT)

if get(ENV,"CORNER_BENCH_STATE_AUDIT","false") == "true"
    # Select the worst sampled-field mode at ppw 10 in the higher band. The
    # cap is sampled up to r=0.99: cap-localized modes can be exponentially
    # small in the stem, so stem-only relative errors are not meaningful.
    difference=(a,b)->norm((dot(a,b)/dot(a,a)).*a-b)/norm(b)
    state_results=Any[]
    center=maximum(CENTERS)
    for parity in (-1,1)
        reference=references[(center,parity)]
        available=filter(r->r.status=="ok" && r.kind==:weighted &&
            r.center==center && r.parity==parity &&
            (r.ppw in (10.0,12.0,15.0) || r.ppw in REFERENCE_PPWS),results)
        sort!(available;by=r->(r.ppw,r.refined_sources))
        isempty(available) && continue
        anchor=first(available)
        mode=argmax([difference(anchor.fields[:,j],reference.fields[:,j])
            for j in eachindex(reference.roots)])
        for result in available
            saved=joinpath(OUTPUT,"state_"*result.label*"_mode$(mode).jls")
            if isfile(saved)
                cached=deserialize(saved)
                if get(cached,:source_signature,"")==SOURCE_SIGNATURE && get(cached,:targets,nothing)==TARGETS
                    push!(state_results,cached); continue
                end
            end
            println("Polishing physical density: ",result.label," mode=",mode); flush(stdout)
            defaults=WeightedCornerGrading()
            grading=result.refined_sources ? WeightedCornerGrading(
                regular_integration_nodes=2defaults.regular_integration_nodes,
                corner_integration_nodes=2defaults.corner_integration_nodes) : defaults
            kernel=CombinedFieldIntegralEquationSolver(result.ppw,BILLIARD;
                min_pts=64,grading=grading,sym_characters=(parity,))
            points=evaluate_points(kernel,BILLIARD,center+DIAMETER/2)
            k=real(result.roots[mode]); started=time()
            tension,_,q,_=solve_state(kernel,points,k,BILLIARD)
            field=QB._weighted_corner_potential(points,q,k,TARGETS)
            checked=QB._weighted_corner_potential(points,q,k,TARGETS;refinement=2)
            record=(;label=result.label,center,parity,ppw=result.ppw,
                refined_sources=result.refined_sources,mode,k,tension,field,
                raw_field=result.fields[:,mode],source_field_drift=norm(field-checked)/norm(checked),
                seconds=time()-started,source_signature=SOURCE_SIGNATURE,targets=TARGETS)
            serialize(saved,record); push!(state_results,record)
            GC.gc()
        end
    end
    open(joinpath(OUTPUT,"state_convergence.csv"),"w") do io
        println(io,"k_center,parity,requested_ppw,source_refinement,mode_index,real_k,polished_tension,raw_field_difference,polished_field_difference,raw_vs_polished_difference,source_field_drift,audit_seconds")
        for record in state_results
            reference=only(filter(r->r.parity==record.parity && r.refined_sources &&
                r.ppw==REFERENCE_PPWS[2],state_results))
            println(io,join((record.center,record.parity,record.ppw,record.refined_sources ? 2 : 1,
                record.mode,record.k,record.tension,difference(record.raw_field,reference.field),
                difference(record.field,reference.field),difference(record.raw_field,record.field),
                record.source_field_drift,record.seconds),','))
        end
    end
    println("Saved raw-contour and polished physical-density state comparisons.")
end
