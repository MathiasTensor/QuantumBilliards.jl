#!/usr/bin/env julia
# Run in each checkout's environment, without competing numerical jobs:
#   julia --startup-file=no --project -t8 benchmarks/weighted_corner_optimization.jl OUTPUT
# Optional CORNER_OPT_BASELINE=/path/to/previous/OUTPUT compares every retained
# matrix coefficient. Serialized matrices are temporary benchmark fixtures;
# archive only the small CSV tables, not those binary files, in the repository.

using QuantumBilliards, BilliardGeometry, LinearAlgebra, Serialization, Statistics

const OPT_QB = QuantumBilliards
const OPT_OUTPUT = isempty(ARGS) ? joinpath(tempdir(),"weighted-corner-optimization") : abspath(ARGS[1])
const OPT_BASELINE = get(ENV,"CORNER_OPT_BASELINE","")
const OPT_REPETITIONS = parse(Int,get(ENV,"CORNER_OPT_REPETITIONS","5"))
OPT_REPETITIONS >= 5 || error("at least five warmed repetitions are required")
mkpath(OPT_OUTPUT)
BLAS.set_num_threads(1)

billiard = MushroomBilliard(0.4,1.0;R=1.0)
kernel = CombinedFieldIntegralEquationSolver(10.0,billiard;
    min_pts=64,grading=WeightedCornerGrading(),sym_characters=(-1,))
points = evaluate_points(kernel,billiard,100.2)
targets = OPT_QB._fold_boundary(Float64,billiard,points,
    kernel.symmetry,kernel.sym_characters).fundamental_indices
OPT_QB._weighted_corner_source_plan(points,targets)

agreements = NamedTuple[]
for (k,degree) in ((100.0,18),(100.0+0.07im,0))
    matrices = OPT_QB._weighted_corner_matrices(kernel,points,k,degree)
    filename = "coefficients_degree$(degree).jls"
    serialize(joinpath(OPT_OUTPUT,filename),matrices)
    if !isempty(OPT_BASELINE)
        previous = deserialize(joinpath(OPT_BASELINE,filename))
        relative = maximum(norm(a-b)/max(norm(b),eps()) for (a,b) in zip(matrices,previous))
        absolute = maximum(maximum(abs,a-b) for (a,b) in zip(matrices,previous))
        push!(agreements,(;degree,relative,absolute))
        relative < 3e-13 || error("matrix coefficients changed beyond roundoff")
    end
end

trials = NamedTuple[]
for repetition in 1:OPT_REPETITIONS
    fresh = evaluate_points(kernel,billiard,100.2)
    GC.gc()
    trial = @timed OPT_QB._weighted_corner_source_plan(fresh,targets)
    push!(trials,(part="Source setup",repetition,seconds=trial.time,bytes=trial.bytes))
    for (k,degree,part) in ((100.0,18,"Taylor matrices"),(100.0+0.07im,0,"Complex matrix"))
        GC.gc()
        trial = @timed OPT_QB._weighted_corner_matrices(kernel,points,k,degree)
        push!(trials,(;part,repetition,seconds=trial.time,bytes=trial.bytes))
    end
end

open(joinpath(OPT_OUTPUT,"trials.csv"),"w") do io
    println(io,"part,repetition,seconds,bytes")
    for row in trials
        println(io,join((row.part,row.repetition,row.seconds,row.bytes),','))
    end
end
open(joinpath(OPT_OUTPUT,"medians.csv"),"w") do io
    println(io,"part,repetitions,median_seconds,median_bytes,full_nodes,matrix_dimension,julia_threads,blas_threads")
    for part in ("Source setup","Taylor matrices","Complex matrix")
        rows = filter(row->row.part==part,trials)
        seconds = median(row.seconds for row in rows)
        bytes = median(row.bytes for row in rows)
        println(io,join((part,length(rows),seconds,bytes,length(points),length(targets),
            Threads.nthreads(),BLAS.get_num_threads()),','))
        println(part,": ",seconds," s, ",bytes," allocated bytes")
    end
end
open(joinpath(OPT_OUTPUT,"agreement.csv"),"w") do io
    println(io,"degree,max_relative_frobenius_difference,max_absolute_entry_difference")
    for row in agreements
        println(io,join((row.degree,row.relative,row.absolute),','))
        println("Degree ",row.degree," agreement: ",row.relative," relative, ",row.absolute," absolute")
    end
end
