
using FastGaussQuadrature
using StaticArrays
using StatsBase

struct LinearNodes <: AbsSampler 
end 

"""
    sample_points(sampler::LinearNodes, N::Int)

Divides the interval [0,1] into N segments.
Uses the midpoints of these segments as the nodes.
Computes the differences between consecutive nodes as weights.

# Arguments
- `sampler::LinearNodes`: placeholder for structure. Not used internally.
- `N::Int`: Number of lineraly spaced points we want.

# Returns:
- `t::Vector{Float64}`: Array of midpoints from [0,1]
- `dt::Vector{Float64}`: Array of step sizes (differentials).    
"""
function sample_points(sampler::LinearNodes, N::Int)
    t=midpoints(range(0,1.0,length=(N+1)))
    dt=diff(range(0,1.0,length=(N+1)))
    return t,dt
end

struct GaussLegendreNodes <: AbsSampler 
end 

"""
    sample_points(sampler::GaussLegendreNodes, N::Int)

Uses a gausslegendre function from `FastGaussQuadrature` to obtain nodes and weights for Gauss–Legendre quadrature.
Maps the nodes from the interval [-1,1] to [0,1] and then scales the weights accordingly.

# Arguments
- `sampler::GaussLegendreNodes`: placeholder for structure. Not used internally.
- `N::Int`: Number of spaced points we want.

# Returns:
- `t::Vector{Float64}`: Array of parametrizations from [0,1]
- `dt::Vector{Float64}`: Array of step sizes (differentials). 
"""
function sample_points(sampler::GaussLegendreNodes, N::Int)
    x,w=gausslegendre(N)
    t=0.5.*x.+0.5 # rescaled to [0,1]
    dt=w.*0.5 
    return t,dt
end

struct PolarSampler <: AbsSampler
    α::Float64
    β::Float64
    PolarSampler(α::Float64=20.0,β::Float64=1.0)=new(α,β)
end

"""
    sample_points(sampler::PolarSampler,crv::C,N::Int) where {C<:PolarSegments}

Generates a uniform parameterization `ts` over [0,1]. Computes local curvatures along a curve `crv` using a curvature function. Adjusts the spacing `dts` using a two-parameter sigmoid  σ(α,β)=1/(1+βe^{-αx}) based on the curvature.
Normalizes the modified differentials so the new nodes span the interval [0,1].

# Arguments
- `sampler::PolarSampler`: Sampler that holds information on the α and β parameters for thr 2 parameter model.
- `N::Int`: Number of spaced points we want.

# Returns
- `t::Vector{Float64}`: Array of parametrizations from [0,1]
- `dt::Vector{Float64}`: Array of step sizes (differentials). 
"""
function sample_points(sampler::PolarSampler,crv::C,N::Int) where {C<:PolarSegments}
    α=sampler.α
    β=sampler.β
    # generate liner sampling
    ts=range(0,1.0,length=N)
    dts=diff(ts)
    # generate curvatures
    curvatures=curvature(crv,ts)
    # Use a two parameter sigmoid = σ(α, β) = 1/(1+βℯ^-(α*x)) for weights
    new_dts=dts./(1 .+abs(α)*exp.(-abs(β)*abs.(curvatures[1:end-1])))
    # Normalize them so we get 0 -> 1
    new_dts=new_dts/sum(new_dts)
    new_ts=vcat(0.0, cumsum(new_dts)) # generate new ts from new dts
    if isapprox(new_ts[end],1.0) # just to make sure
        new_ts[end]=0.999
    end 
    return new_ts,new_dts
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

"""
    sample_points(sampler::FourierNodes, N::Int)

Generates parametrizatons `ts` and their pairwise differences `dts` using the `FourierNodes` sampler. Used in the construction of the boundary function `u(s)`.

# Arguments
- `sampler::FourierNodes`: Sampler that holds information on the primes and lengths parameters that both define enough geometric information to construct sampling.
- `N::Int`: Number of spaced points we want.

# Returns
- `t::Vector{Float64}`: Array of parametrizations from [0,1]
- `dt::Vector{Float64}`: Array of step sizes (differentials). 
"""
function sample_points(sampler::FourierNodes, N::Int)
    if isnothing(sampler.primes) 
        M=N
    else
        M=nextprod(sampler.primes,N)
    end
    ts=Vector{Vector{Float64}}(undef,0)
    dts=Vector{Vector{Float64}}(undef,0)
    t::Vector{Float64}=Vector{Float64}(undef,0)
    dt::Vector{Float64}=Vector{Float64}(undef,0)
    if isnothing(sampler.lengths)
        t=collect(i/M for i in 0:(M-1))
        dt=diff(t)
        dt=push!(dt,dt[1])
        push!(ts,t)
        push!(dts,dt)
    else
        crv_lengths::Vector{Float64}=sampler.lengths
        L::Float64=sum(crv_lengths)
        start::Float64=0.0
        dt_end::Float64=0.0
        ds::Float64=0.0
        for l in crv_lengths
            ds=L/(l*M) 
            #println(start*ds)
            t=collect(range(start*ds,1.0,step=ds))
            dt_end=1.0-t[end]
            start=(ds-dt_end)/ds
            push!(ts,t)
            dt=diff(t)
            push!(dt,dt_end)
            push!(dts,dt)
        end
    end
    return ts,dts
end

"""
    random_interior_points(billiard::AbsBilliard, N::Int; grd::Int = 1000)

Retrieves the bounding limits of the billiard’s fundamental boundary (using boundary_limits) and generates points within the bounds.
1st Checks if each point is inside the billiard using is_inside then Continues until N valid interior points are collected.

# Arguments
- `billiard::AbsBilliard`: Instance of the geometry so we can check what is the interior.
- `N::Int`: The number of internal points we want.
- `grd::Int=1000`: Parameter that determines the precision of the limits of the billiard boundary. Usually 1000 is enough and there is no need to change.

# Returns:
- `pts::Vector{SVector{2,Float64}}`: A vector of points inside the billiard.
"""
function random_interior_points(billiard::AbsBilliard,N::Int;grd::Int=1000)
    xlim,ylim=boundary_limits(billiard.fundamental_boundary; grd=grd)
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    pts=[]
    while length(pts)<N
        x=(dx.*rand().+xlim[1]) 
        y=(dy.*rand().+ylim[1])
        pt=SVector(x,y)
        if is_inside(billiard,[pt])[1] #TODO Kind of stupid that we have to access 1 element vector b/c there is no single vector implementation of is_inside
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