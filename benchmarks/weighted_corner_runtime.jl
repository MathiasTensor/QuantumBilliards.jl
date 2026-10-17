#!/usr/bin/env julia
# Warmed full-solve comparison on the existing BilliardGeometry mushroom.
# Run the convergence benchmark first to produce ppw-30/source-refined roots:
#   julia --project -t8 benchmarks/weighted_corners.jl /path/to/convergence
# Then, while other numerical jobs are stopped:
#   CORNER_RUNTIME_REFERENCES=/path/to/convergence \
#   CORNER_RUNTIME_ISOLATED=true julia --project -t8 \
#       benchmarks/weighted_corner_runtime.jl /path/to/runtime
#
# Compilation warmups and resolution selection are recorded separately from
# timed repetitions. Each repetition constructs a fresh mesh and source cache,
# so preparation plus solve includes all work up to accepted root extraction.
# Eigenvectors are retained by both methods; interior field audits are excluded.
# Default residual, rank, imaginary-part and CORK stability filters are preserved.
# CORK keeps its default degree and search budget: failure is recorded honestly.
#
# Optional environment: CORNER_RUNTIME_REPETITIONS=3 (minimum 3),
# CORNER_RUNTIME_METHODS=beyn,cork, CORNER_RUNTIME_KINDS=weighted,legacy,
# CORNER_RUNTIME_PARITIES=-1,1. CORNER_RUNTIME_MODE=smoke runs one correctness
# trial per selected case and never labels those times as isolated benchmarks.

using QuantumBilliards, BilliardGeometry, LinearAlgebra, Serialization, Statistics, SHA

const RT_QB = QuantumBilliards
const RT_OUTPUT = isempty(ARGS) ? joinpath(tempdir(),"QuantumBilliards-weighted-runtime") : abspath(ARGS[1])
const RT_REFERENCE_DIRECTORY = abspath(get(ENV,"CORNER_RUNTIME_REFERENCES",joinpath(tempdir(),"QuantumBilliards-weighted-corners")))
const RT_METHODS = Symbol.(split(get(ENV,"CORNER_RUNTIME_METHODS","beyn,cork"),','))
const RT_KINDS = Symbol.(split(get(ENV,"CORNER_RUNTIME_KINDS","weighted,legacy"),','))
const RT_PARITIES = parse.(Int,split(get(ENV,"CORNER_RUNTIME_PARITIES","-1,1"),','))
const RT_REPETITIONS = parse(Int,get(ENV,"CORNER_RUNTIME_REPETITIONS","3"))
const RT_SMOKE = get(ENV,"CORNER_RUNTIME_MODE","benchmark") == "smoke"
const RT_ISOLATED = !RT_SMOKE && get(ENV,"CORNER_RUNTIME_ISOLATED","false") == "true"
const RT_CENTER = 100.0
const RT_WIDTH = 0.4                  # radius 0.2, below the 0.5 maximum
const RT_ACCURACY = 1.0e-8            # comparison target, not a solver filter
const RT_WEIGHTED_PPW = 10.0
const RT_BILLIARD = MushroomBilliard(0.4,1.0;R=1.0)
const RT_EXPECTED_COUNTS = Dict(-1=>6,1=>9)
const RT_BLAS_CONFIGURATION = string(BLAS.get_config())
const RT_INITIAL_BLAS_THREADS = BLAS.get_num_threads()
const RT_SOURCE_SIGNATURE = bytes2hex(sha256(join(read(file,String) for
    file in sort([joinpath(root,name) for (root,_,names) in walkdir(dirname(pathof(RT_QB)))
        for name in names if endswith(name,".jl")]))))

all(method -> method in (:beyn,:cork),RT_METHODS) || error("methods must be beyn and/or cork")
all(kind -> kind in (:weighted,:legacy),RT_KINDS) || error("kinds must be weighted and/or legacy")
all(parity -> parity in (-1,1),RT_PARITIES) || error("parities must be -1 and/or 1")
RT_REPETITIONS >= 3 || error("at least three timed repetitions are required")
mkpath(RT_OUTPUT)

# Roots come from a separately refined mesh and source rule, not from either
# timed discretization. Keep their original complex parts and multiplicities.
const RT_REFERENCES = Dict{Int,NamedTuple}()
for parity in RT_PARITIES
    file = joinpath(RT_REFERENCE_DIRECTORY,"weighted_k100.0_ppw30.0_parity$(parity)_sources2.jls")
    isfile(file) || error("missing refined numerical reference: $file; run weighted_corners.jl first")
    result = deserialize(file)
    result.status == "ok" || error("refined numerical reference failed for parity $parity")
    roots = sort(ComplexF64.(result.roots);by=real)
    length(roots) == RT_EXPECTED_COUNTS[parity] || error("reference root count is inconsistent for parity $parity")
    RT_REFERENCES[parity] = (;roots,file,source_signature=get(result,:source_signature,"unknown"))
end

"""
    runtime_trial(method::Symbol, kind::Symbol, ppw::Float64, parity::Int,
        phase::Symbol, repetition::Int) -> NamedTuple

Construct one fresh kernel, boundary mesh, source plan and complete accepted
spectrum on `100±0.2`. Separate preparation from the public `solve_vectors`
call. Compare every returned complex root, including multiplicity, with the
independently refined numerical reference. Solver exceptions are recorded;
acceptance filters are never relaxed to turn a failure into a timing sample.
"""
function runtime_trial(method::Symbol,kind::Symbol,ppw::Float64,parity::Int,phase::Symbol,repetition::Int)::NamedTuple
    println("Starting ",phase," ",method," ",kind," ppw=",ppw," parity=",parity," repetition=",repetition)
    flush(stdout)
    GC.gc()
    started=time_ns(); preparation=NaN; solve_seconds=NaN; nodes=0; dimension=0
    taylor_degree=0; returned=ComplexF64[]; residuals=Float64[]
    error_message=""; status="error"; maximum_error=NaN
    try
        grading=kind == :weighted ? WeightedCornerGrading() : GlobalCornerGrading(kressq=4)
        kernel=CombinedFieldIntegralEquationSolver(ppw,RT_BILLIARD;
            min_pts=64,grading=grading,sym_characters=(parity,))
        solver=method == :beyn ? BeynSolver(kernel;nq=40,r=24,Rmax=0.5,
            use_taylor=true,use_chebyshev=false,taylor_degree=18) :
            CORKSolver(kernel;Rmax=0.5)
        points=evaluate_points(kernel,RT_BILLIARD,RT_CENTER+RT_WIDTH/2)
        nodes=length(points); dimension=boundary_matrix_size(kernel,points)
        preparation=(time_ns()-started)*1e-9
        solve_started=time_ns()
        returned,residuals,_=solve_vectors(solver,points,RT_CENTER,RT_WIDTH)
        solve_seconds=(time_ns()-solve_started)*1e-9
        permutation=sortperm(real.(returned))
        returned=ComplexF64.(returned[permutation]); residuals=Float64.(residuals[permutation])
        reference=RT_REFERENCES[parity].roots
        taylor_degree=method == :beyn ? solver.taylor_degree : solver.p
        if length(returned) == length(reference)
            maximum_error=maximum(abs,returned-reference)
            status=maximum_error <= RT_ACCURACY ? "complete" : "accuracy_mismatch"
        else
            status="count_mismatch"
        end
    catch err
        error_message=sprint(showerror,err,catch_backtrace())
        println(stderr,error_message)
    end
    total=(time_ns()-started)*1e-9
    result=(;method,kind,ppw,parity,phase,repetition,status,nodes,dimension,
        expected_count=RT_EXPECTED_COUNTS[parity],found_count=length(returned),
        maximum_error,maximum_imaginary_part=isempty(returned) ? NaN : maximum(abs,imag.(returned)),
        maximum_residual=isempty(residuals) ? NaN : maximum(residuals),
        residual_kind=method==:cork ? "projected_inverse_linearization" : "direct_fredholm_norm",
        preparation_seconds=preparation,solve_seconds,total_seconds=total,
        roots=returned,residuals,taylor_degree,error_message,timing_isolated=RT_ISOLATED,
        julia_threads=Threads.nthreads(),initial_blas_threads=RT_INITIAL_BLAS_THREADS,
        lapack_thread_budget=RT_QB.MAX_BLAS_THREADS,blas_configuration=RT_BLAS_CONFIGURATION,
        source_signature=RT_SOURCE_SIGNATURE)
    serialize(joinpath(RT_OUTPUT,"$(phase)_$(method)_$(kind)_ppw$(ppw)_parity$(parity)_rep$(repetition).jls"),result)
    println("Finished ",phase," ",method," ",kind," parity=",parity," status=",status,
        " roots=",length(returned)," N=",nodes," full_seconds=",round(total;digits=4))
    flush(stdout)
    return result
end

# CSV quoting also preserves complete failure messages containing newlines.
rt_csv(value) = value isa AbstractString ? "\""*replace(value,"\""=>"\"\"")*"\"" : string(value)
function runtime_tables(records)
    open(joinpath(RT_OUTPUT,"trials.csv"),"w") do io
        fields=(:method,:kind,:ppw,:parity,:phase,:repetition,:status,:nodes,:dimension,
            :expected_count,:found_count,:maximum_error,:maximum_imaginary_part,:maximum_residual,:residual_kind,
            :preparation_seconds,:solve_seconds,:total_seconds,:taylor_degree,:timing_isolated,
            :julia_threads,:initial_blas_threads,:lapack_thread_budget,:blas_configuration,:error_message)
        println(io,join(fields,','))
        for result in records
            println(io,join((rt_csv(getproperty(result,field)) for field in fields),','))
        end
    end
    open(joinpath(RT_OUTPUT,"roots.csv"),"w") do io
        println(io,"method,kind,ppw,parity,phase,repetition,root_index,real_k,imaginary_k,residual")
        for result in records, j in eachindex(result.roots)
            println(io,join((result.method,result.kind,result.ppw,result.parity,result.phase,
                result.repetition,j,real(result.roots[j]),imag(result.roots[j]),result.residuals[j]),','))
        end
    end
    open(joinpath(RT_OUTPUT,"medians.csv"),"w") do io
        println(io,"method,kind,ppw,parity,repetitions,successful_repetitions,status,nodes,matrix_dimension,expected_count,max_root_error,max_imaginary_part,median_preparation_seconds,median_solve_seconds,median_total_seconds,timing_isolated")
        for method in RT_METHODS, kind in RT_KINDS, parity in RT_PARITIES
            trials=filter(r->r.phase==:timed && r.method==method && r.kind==kind && r.parity==parity,records)
            isempty(trials) && continue
            successful=filter(r->r.status=="complete",trials)
            first_trial=first(trials)
            values=isempty(successful) ? nothing : (;preparation=median(r.preparation_seconds for r in successful),
                solve=median(r.solve_seconds for r in successful),total=median(r.total_seconds for r in successful))
            complete=length(successful)==RT_REPETITIONS
            println(io,join((method,kind,first_trial.ppw,parity,length(trials),length(successful),
                complete ? "complete" : "unresolved",first_trial.nodes,first_trial.dimension,first_trial.expected_count,
                isempty(successful) ? NaN : maximum(r.maximum_error for r in successful),
                isempty(successful) ? NaN : maximum(r.maximum_imaginary_part for r in successful),
                values===nothing ? NaN : values.preparation,values===nothing ? NaN : values.solve,
                values===nothing ? NaN : values.total,RT_ISOLATED),','))
        end
    end
    open(joinpath(RT_OUTPUT,"combined_medians.csv"),"w") do io
        println(io,"method,kind,ppw,parities,repetitions,status,nodes_per_full_boundary,matrix_dimension_per_sector,reference_states,median_preparation_seconds,median_solve_seconds,median_total_seconds,timing_isolated")
        for method in RT_METHODS, kind in RT_KINDS
            trials=filter(r->r.phase==:timed && r.method==method && r.kind==kind,records)
            isempty(trials) && continue
            preparation=Float64[]; solves=Float64[]; totals=Float64[]
            for repetition in 1:RT_REPETITIONS
                group=filter(r->r.repetition==repetition,trials)
                if length(group)==length(RT_PARITIES) && all(r->r.status=="complete",group)
                    push!(preparation,sum(r.preparation_seconds for r in group))
                    push!(solves,sum(r.solve_seconds for r in group))
                    push!(totals,sum(r.total_seconds for r in group))
                end
            end
            sample=first(trials)
            complete=length(totals)==RT_REPETITIONS
            println(io,join((method,kind,sample.ppw,length(RT_PARITIES),length(totals),
                complete ? "complete" : "unresolved",sample.nodes,sample.dimension,
                sum(RT_EXPECTED_COUNTS[p] for p in RT_PARITIES),
                isempty(totals) ? NaN : median(preparation),isempty(totals) ? NaN : median(solves),
                isempty(totals) ? NaN : median(totals),RT_ISOLATED),','))
        end
    end
end

open(joinpath(RT_OUTPUT,"settings.txt"),"w") do io
    println(io,"Boundary: MushroomBilliard(0.4,1.0;R=1.0), window 99.8-100.2, both requested Y parities.")
    println(io,"Weighted ppw=10. Legacy global Kress q=4, ppw=20 or24 selected against the 1e-8 numerical-reference target.")
    println(io,"Beyn: nq=40, r=24, Rmax=.5, Taylor degree18; all acceptance thresholds retain defaults.")
    println(io,"CORK: default polynomial degree12, guard.05, block10, mstart500, mstep50, stable_checks1; Rmax=.5; all filters unchanged.")
    println(io,"Compilation warmups and resolution selection excluded from medians. Each timed trial builds fresh points/cache and retains vectors.")
    println(io,"Preparation: fresh kernel/solver, density mesh and symmetry size. Solve: source-rule preparation, matrix/polynomial construction, factorization, root extraction and unchanged solver checks.")
    println(io,"One public solve_vectors call per contour, including returned vectors. Beyn residuals are direct Fredholm norms; CORK residuals are projected inverse-linearization indicators, not comparable norms.")
    println(io,"Julia ",VERSION,"; Julia threads ",Threads.nthreads(),"; initial BLAS threads ",RT_INITIAL_BLAS_THREADS,"; LAPACK macro budget ",RT_QB.MAX_BLAS_THREADS)
    println(io,"BLAS ",RT_BLAS_CONFIGURATION,"; timing isolated ",RT_ISOLATED,"; smoke ",RT_SMOKE,"; requested repetitions ",RT_REPETITIONS)
    println(io,"QuantumBilliards source signature: ",RT_SOURCE_SIGNATURE)
    println(io,"Geometry source: ",pathof(BilliardGeometry))
    for parity in RT_PARITIES
        println(io,"Reference parity ",parity,": ",RT_REFERENCES[parity].file,"; reference source signature ",RT_REFERENCES[parity].source_signature)
    end
end

records=NamedTuple[]
legacy_ppw=20.0
if :legacy in RT_KINDS
    selection=[runtime_trial(:beyn,:legacy,legacy_ppw,parity,:selection,0) for parity in RT_PARITIES]
    append!(records,selection); runtime_tables(records)
    if !all(r->r.status=="complete",selection)
        legacy_ppw=24.0
        selection=[runtime_trial(:beyn,:legacy,legacy_ppw,parity,:selection,0) for parity in RT_PARITIES]
        append!(records,selection); runtime_tables(records)
    end
end
cases=[(;method,kind,ppw=kind==:weighted ? RT_WEIGHTED_PPW : legacy_ppw,parity)
    for method in RT_METHODS for kind in RT_KINDS for parity in RT_PARITIES]
if RT_SMOKE
    for case in cases
        push!(records,runtime_trial(case.method,case.kind,case.ppw,case.parity,:smoke,0))
        runtime_tables(records)
    end
else
    for case in cases
        push!(records,runtime_trial(case.method,case.kind,case.ppw,case.parity,:warmup,0))
        runtime_tables(records)
    end
    for repetition in 1:RT_REPETITIONS
        order=iseven(repetition) ? reverse(cases) : cases
        for case in order
            push!(records,runtime_trial(case.method,case.kind,case.ppw,case.parity,:timed,repetition))
            runtime_tables(records)
        end
    end
end
println("Saved correctness records, warmups, complete timed trials and medians in ",RT_OUTPUT)
