
# INTERNAL: helper function for the joint distribution
function e1(s::T,t::T) where {T<:Real}
    return 243/(32*pi^2)*t*(2*s+t)*exp((-9*(s^2+s*t+t^2))/(4*pi)) + 81/(128*pi^2)*t*(8*pi-9*t^2)*exp((-27*t^2)/(16*pi))*erfc((3*(2*s+t))/(4*sqrt(pi)))
end

# INTERNAL: helper function for the joint distribution
function e2(s::T, t::T) where {T<:Real}
    return e1(t,s)
end

# INTERNAL: helper function for the joint distribution
function f(s::T, t::T) where {T<:Real}
    return 9/(4*pi)*(s+t)*exp((-9*(s+t)^2)/(4*pi)) + 0.5*erfc((3*(s+t))/(2*sqrt(pi))) + (8*pi-27*(s+t)^2)/(16*pi)*exp((-27*(s+t)^2)/(16*pi))*erfc((3*(s+t))/(4*sqrt(pi)))
end

# INTERNAL: helper function for the joint distribution
function p_hat(s::T, t::T) where {T<:Real} 
    return 243/(32*pi^2)*((s+t)^2)*exp((-9*(s+t)^2)/(4*pi)) + 81/(128*pi^2)*(s+t)*(8*pi-9*(s+t)^2)*exp((-27*(s+t)^2)/(16*pi))*erfc((3*(s+t))/(4*sqrt(pi)))
end

# INTERNAL: helper function for the joint distribution
function g(u::T) where {T<:Real}
    return exp(-9*(u^2)/(4*pi)) - u/2*erfc((3*u)/(2*sqrt(pi))) - u/2*exp((-27*u^2)/(16*pi))*erfc((3*u/(4*sqrt(pi))))
end

# INTERNAL: helper function for the joint distribution
function h(u::T, v::T) where {T<:Real}
    V1(u,v) = exp((-9*((u^2) + u*v+v^2))/(4*pi))
    V2(u,v) = exp(-27*(u^2)/(16*pi))*erfc(3*(u+2*v)/(4*sqrt(pi)))
    V3(u,v) = exp(-27*(v^2)/(16*pi))*erfc(3*(2*u+v)/(4*sqrt(pi)))
    return (9*(u+v)/(4*pi))*V1(u,v) + (8*pi-27*u^2)/(16*pi)*V2(u,v) + (8*pi-27*v^2)/(16*pi)*V3(u,v)
end

# INTERNAL P(s,t) gap probability function for m=2 case GOE and Poisson
function P_gap(s::T, t::T, μ_c::T) where {T<:Real}
    μ_1 = 1.0 - μ_c
    G = g(μ_c*(s+t))
    H = h(μ_c*s, μ_c*t)
    E1 = e1(s,t)
    E2 = e2(s,t)
    P_hat = p_hat(s,t)
    P = 3^7/(32*pi^3)*s*t*(s+t)*exp(-9(s^2+s*t+t^2)/(4*pi))
    F = f(s,t)
    return exp(-μ_1*(s+t))*((μ_1^3)*G+(μ_1^2)*μ_c*(H+2*F)+μ_1*(μ_c^2)*(P_hat+E1+E2)+(μ_c^3)*P)
end

# INTERNAL non-normalized P_r distribution for m=2 case
function P_r(r::T, μ_c::T) where {T<:Real}
    integrand(s) = s*P_gap(s, r*s, μ_c)
    res, _ = quadgk(s -> integrand(s), 0.0, Inf)
    return 2*res
end

# INTERNAL nomrlaization constant for the P_r m=2 case
function normalization_P_r(μ_c::T) where {T<:Real}
    integrand(r) = P_r(r, μ_c)
    res, _ = quadgk(r -> integrand(r), 0.0, 1.0)
    return res
end

"""
    P_r_normalized(r::T, μ_c::T) -> T where {T<:Real}

Returns the normalized probability distribution of the gap ratio `r` for a system with 2 components (regular and chaotic) (parameterized by `μ_c`).

# Arguments
- `r::T`: The gap ratio.
- `μ_c::T`: Chaotic phasespace portion.

# Returns
- `T`: The normalized probability distribution function `P_r_normalized(r, μ_c)` for the gap ratio r.
"""
function P_r_normalized(r::T, μ_c::T) where {T<:Real}
    return 1/normalization_P_r(μ_c)*P_r(r, μ_c)
end

"""
    P_integrable(r::T) -> T where {T<:Real}

Returns the probability distribution of the gap ratio `r` for a fully integrable system (chaotic phasespace portion μ_c = 0.0). It is normalized.

# Arguments
- `r::T`: The gap ratio.

# Returns
- `T`: The probability distribution function `P(r)` for the gap ratio in an integrable system.
"""
function P_integrable(r::T) where {T<:Real}
    return 2/(1+r)^2
end

"""
    P_chaotic(r::T, β::Int) -> T where {T<:Real}

Returns the probability distribution of the gap ratio `r` for a chaotic system (chaotic phasespace portion μ_c = 1.0), parameterized by `β`. It is normalized.

# Arguments
- `r::T`: The gap ratio (must be a real number).
- `β::Int`: The Dyson index:
    - β = 1 for Gaussian Orthogonal Ensemble (GOE),
    - β = 2 for Gaussian Unitary Ensemble (GUE), 
    - β = 4 for Gaussian Symplectic Ensemble (GSE).

# Returns
- `T`: The probability distribution function `P(r)` for the gap ratio in a chaotic system.
"""
function P_chaotic(r::T, β::Int) where {T<:Real}
    integrand(l) = ((l+l^2)^β)/((1+l+l^2)^(1+3*β/2))
    Z_β = quadgk(l -> integrand(l), 0.0, 1.0)[1] # only the value
    return 1/Z_β * integrand(r)
end

"""
    plot_gap_ratios(energies::Vector{T}; nbins::Int=50, μ_c::Union{Nothing,T}=nothing) -> Figure where {T<:Real}

Plots the empirical gap ratio distribution for a given set of energy levels and compares it with theoretical distributions for integrable and chaotic systems.

# Arguments
- `energies::Vector{T}`: A vector of energy levels for which the gap ratios are calculated.
- `nbins::Int=50`: (Optional) The number of bins used to create the histogram of the empirical gap ratio distribution.
- `μ_c::Union{Nothing,T}=nothing`: (Optional) The chaotic phasespace portion `μ_c`.If `nothing` is passed, no mixed distribution is plotted.

# Returns
- `Figure`: A plot figure (`Figure`) that contains:
    - The empirical gap ratio distribution (as a scatter plot).
    - Theoretical gap ratio distributions (as line plots).
"""
function plot_gap_ratios(energies::Vector{T}; nbins::Int=50, μ_c::Union{Nothing,T}=nothing) where {T<:Real}
    energy_differences = diff(energies)
    gap_ratios = Vector{T}(undef, length(energy_differences) - 1)
    for i in eachindex(gap_ratios)
        s_n = energy_differences[i]
        s_n1 = energy_differences[i + 1]
        gap_ratios[i] = min(s_n, s_n1)/max(s_n, s_n1)
    end
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title="Gap Ratios", xlabel="r", ylabel="P(r)")
    hist = Distributions.fit(StatsBase.Histogram, gap_ratios; nbins=nbins)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) / 2
    bin_counts = hist.weights ./ sum(hist.weights) / diff(hist.edges[1])[1]
    scatter!(ax, bin_centers, bin_counts, label="Empirical", color=:black, marker=:cross, markersize=10)
    r_values = range(0, stop=maximum(bin_centers), length=1000)
    integrable = [P_integrable(r) for r in r_values]
    chaotic = [P_chaotic(r,1) for r in r_values] # GOE 
    lines!(ax, r_values, integrable, label="Integrable", color=:blue, linestyle=:dash, linewidth=1)
    lines!(ax, r_values, chaotic, label="Chaotic", color=:green, linestyle=:dot, linewidth=1)
    
    if !isnothing(μ_c)
        mixed = [P_r_normalized(r,μ_c) for r in r_values]
        lines!(ax, r_values, mixed, label="m=2 components, rho=$(round(μ_c; sigdigits=5))", color=:orange, linestyle=:dashdot, linewidth=1)
    end
    xlims!(ax, extrema(r_values))
    axislegend(ax, position=:rt)
    return fig
end
