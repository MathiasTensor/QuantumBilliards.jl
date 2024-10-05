
using FastGaussQuadrature
using StaticArrays
using StatsBase
include("../billiards/geometry/polarsegment.jl")

struct LinearNodes <: AbsSampler 
end 

function sample_points(sampler::LinearNodes, N::Int)
    t = midpoints(range(0,1.0,length = (N+1)))
    dt = diff(range(0,1.0,length =(N+1)))
    return t, dt
end

struct GaussLegendreNodes <: AbsSampler 
end 

function sample_points(sampler::GaussLegendreNodes, N::Int)
    x, w = gausslegendre(N)
    t = 0.5 .* x  .+ 0.5
    dt = w .* 0.5 
    return t, dt
end

struct PolarSampler <: AbsSampler
end

function sample_points(sampler::PolarSampler, crv::C, N::Int; α::Float64=4.0, β::Float64=0.7) where {C<:PolarSegments}
    # generate liner sampling
    ts = range(0, 1.0, length=N)
    dts = diff(ts)
    # generate curvatures
    curvatures = curvature(crv, ts)
    # Use a two parameter sigmoid = σ(α, β) = 1/(1+βℯ^-(α*x)) for weights
    new_dts = dts ./ (1 .+ abs(α) * exp.(-abs(β) * abs.(curvatures[1:end-1])))
    # Normalize them so we get 0 -> 1
    new_dts = new_dts / sum(new_dts)
    new_ts = vcat(0.0, cumsum(new_dts)) # generate new ts from new dts
    new_ts[end] = 1.0 # just to make sure
    return new_ts, new_dts
end

#=
#this one is not working yet
function chebyshev_nodes(N::Int)
    x = [cos((2*i-1)/(2*N)*pi) for i in 1:N]
    t = 0.5 .* x  .+ 0.5
    dt = ones(N)  #wrong
    return t, dt
end
=#

struct FourierNodes <: AbsSampler where T<:Real
    primes::Union{Vector{Int64},Nothing}
    lengths::Union{Vector{Float64},Nothing} 
end 

FourierNodes() = FourierNodes(nothing,nothing)
FourierNodes(lengths::Vector{Float64}) = FourierNodes(nothing,lengths)

function sample_points(sampler::FourierNodes, N::Int)
    if isnothing(sampler.primes) 
        M = N
    else
        M = nextprod(sampler.primes,N)
    end

    ts = Vector{Vector{Float64}}(undef,0)
    dts = Vector{Vector{Float64}}(undef,0)
    t::Vector{Float64} = Vector{Float64}(undef,0)
    dt::Vector{Float64} = Vector{Float64}(undef,0)
    if isnothing(sampler.lengths)
        t = collect(i/M for i in 0:(M-1))
        dt = diff(t)
        dt = push!(dt,dt[1])
        push!(ts,t)
        push!(dts,dt)
    else
        crv_lengths::Vector{Float64} = sampler.lengths
        L::Float64 = sum(crv_lengths)

        start::Float64 = 0.0
        dt_end::Float64 = 0.0
        ds::Float64 = 0.0
        for l in crv_lengths
            ds = L/(l*M) 
            println(start*ds)
            t = collect(range(start*ds,1.0,step=ds))
            #println(t)
            dt_end = 1.0 - t[end]
            start = (ds - dt_end)/ds
            push!(ts,t)
            dt = diff(t)
            push!(dt,dt_end)
            push!(dts,dt)
        end
    end
    
    return ts,dts
end


function random_interior_points(billiard::AbsBilliard, N::Int; grd::Int = 1000)
    xlim,ylim = boundary_limits(billiard.fundamental_boundary; grd=grd)
    dx =  xlim[2] - xlim[1]
    dy =  ylim[2] - ylim[1]
    pts = []
 
    #println(length(pts))
    while length(pts)<N
        x = (dx .* rand() .+ xlim[1]) 
        y = (dy .* rand() .+ ylim[1])
        pt = SVector(x,y)
        if is_inside(billiard, [pt])[1] #rework this
            push!(pts,pt)
        end
    end
    return pts
end

#=
#needs some work
function fourier_nodes(N::Int; primes=(2,3,5)) #starts at 0 ends at 
    if primes == false
        M = N
    else
        M = nextprod(primes,N)
    end
    t = collect(i/M for i in 0:(M-1))
    dt = diff(t)
    dt = push!(dt,dt[1])
    return t, dt
end

function fourier_nodes(N::Int, crv_lengths; primes=(2,3,5)) #starts at 0 ends at 
    if primes == false
        M = N
    else
        M = nextprod(primes,N)
    end
    L = sum(crv_lengths)
    ts =Vector{Vector{typeof(L)}}(undef,0)
    dts =Vector{Vector{typeof(L)}}(undef,0)
    start = 0.0
    for l in crv_lengths
        ds = L/(l*M) 
        println(start*ds)
        t = collect(range(start*ds,1.0,step=ds))
        #println(t)
        dt_end = 1.0 - t[end]
        start = (ds - dt_end)/ds
        push!(ts,t)
        dt = diff(t)
        push!(dt,dt_end)
        push!(dts,dt)
    end
    return ts,dts
end
=#