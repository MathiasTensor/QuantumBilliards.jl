#!/usr/bin/env julia
# Compare weighted mesh/source projections in each checkout's environment:
#   julia --startup-file=no --project -t8 benchmarks/weighted_corner_postprocessing.jl OUTPUT
# For the optimized run set CORNER_GEOMETRY_BASELINE to the earlier OUTPUT.
# Eleven warmed timing repetitions require no competing numerical jobs.
# The binary fixture is temporary; archive only CSV results in the repository.

using QuantumBilliards, BilliardGeometry, LinearAlgebra, Serialization, Statistics
const QB = QuantumBilliards
const output = abspath(ARGS[1])
const previous = get(ENV,"CORNER_GEOMETRY_BASELINE","")
BLAS.set_num_threads(1)
mkpath(output)
billiard = MushroomBilliard(0.4,1.0;R=1.0)
kernel = CombinedFieldIntegralEquationSolver(10.0,billiard;
    min_pts=64,grading=WeightedCornerGrading(),sym_characters=(-1,))
points = evaluate_points(kernel,billiard,100.2)
q = ComplexF64[sin(0.031j)+im*cos(0.017j) for j in 1:length(points)]
samples, amplitudes = QB._weighted_corner_boundary_quadrature(points,q)
power, frequencies = momentum_function(q,points)
qs = collect(range(0,sum(curve.length for curve in full_boundary(billiard));length=11))
ps = collect(range(-1,1;length=7))
husimi = husimi_function(100.0,points,q,qs,ps;full_p=true)
fixture = (;xy=points.xy,normal=points.normal,s=points.s,ds=points.ds,
    tangent=points.tangent,tangent_2=points.tangent_2,tphys=points.tphys,
    norm_weights=points.density_space.norm_weights,samples,amplitudes,power,
    frequencies,rellich=QB._rellich(points,q,100.0),husimi=husimi[1])
serialize(joinpath(output,"geometry_state.jls"),fixture)
agreement = NamedTuple[]
if !isempty(previous)
    old = deserialize(joinpath(previous,"geometry_state.jls"))
    for name in propertynames(fixture)
        current = getproperty(fixture,name); before = getproperty(old,name)
        relative = norm(current-before)/max(norm(before),eps())
        absolute = current isa Number ? abs(current-before) : maximum(norm,current-before)
        push!(agreement,(;name,relative,absolute))
        println(name,",",relative,",",absolute)
        relative < 1e-12 || error("geometry/state projection changed beyond roundoff: $name")
    end
end
open(joinpath(output,"geometry_state_agreement.csv"),"w") do io
    println(io,"quantity,relative_difference,maximum_absolute_difference")
    for row in agreement
        println(io,join((row.name,row.relative,row.absolute),','))
    end
end
geometry_times = Float64[]; projection_times = Float64[]
for trial in 1:11
    GC.gc()
    push!(geometry_times,@elapsed evaluate_points(kernel,billiard,100.2))
    GC.gc()
    push!(projection_times,@elapsed QB._weighted_corner_boundary_quadrature(points,q))
end
open(joinpath(output,"geometry_state_medians.csv"),"w") do io
    println(io,"part,median_seconds,full_nodes")
    println(io,"Density mesh,",median(geometry_times),",",length(points))
    println(io,"Boundary source projection,",median(projection_times),",",length(points))
end
println("Density mesh: ",median(geometry_times)," s")
println("Boundary source projection: ",median(projection_times)," s")
