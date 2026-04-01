#################################################################
#   CHEBYSHEV-BASED SLP/DLP EVALUATION FOR CFIE_alpert ASSEMBLY
#   Faithful multi-k port of the old scalar Alpert assembly.
#
# Strategy:
# - Preserve the old scalar assembly logic exactly.
# - Replace only H(0,k*r), H(1,k*r) by multi-k Chebyshev lookups.
# - Preserve periodic / smooth-panel / composite / symmetry logic.
#
# This batch contains:
#   1) geometry/block caches
#   2) multi-k H0/H1 plan builders
#   3) thread-local workspaces
#   4) shifted-source lookup caches
#   5) reusable context builders
#
# MO 2/4/26
#################################################################

###############################
#### GEOMETRY-ONLY WORKSPACE ###
###############################

struct CFIEAlpertBlockCache{T<:Real}
    same::Bool
    row_offset::Int
    col_offset::Int
    Ni::Int
    Nj::Int
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    speed_j::Vector{T}
    wj::Vector{T}
    pidx::Matrix{Int32}
    tloc::Matrix{Float64}
end

struct CFIEAlpertBlockSystemCache{T<:Real}
    blocks::Matrix{CFIEAlpertBlockCache{T}}
    offsets::Vector{Int}
    rmin::Float64
    rmax::Float64
end

struct CFIEAlpertGeomWorkspace{T<:Real,C}
    rule::AlpertLogRule{T}
    offs::Vector{Int}
    Gs::Vector{CFIEGeomCache{T}}
    Cs::Vector{C}
    topos::Union{Nothing,Vector{AlpertCompositeTopology{T}}}
    gmaps::Union{Nothing,Vector{Vector{Int}}}
    panel_to_comp::Union{Nothing,Vector{Int}}
    block_cache::CFIEAlpertBlockSystemCache{T}
    Ntot::Int
end

#################################
#### BLOCK DISTANCE CACHES   ####
#################################

# build_cfie_alpert_block_caches
# Build the geometry-only block caches used by the Chebyshev Alpert route.
#
# Notes:
#   - This cache is only for ordinary distance lookup and Chebyshev panel lookup.
#   - The actual assembly logic must still follow the old scalar Alpert code.
function build_cfie_alpert_block_caches(
    comps::Vector{BoundaryPointsCFIE{T}};
    npanels::Int=10000,
    M::Int=5,
    grading::Symbol=:uniform,
    geo_ratio::Real=1.05,
    pad=(T(0.95), T(1.05)),
) where {T<:Real}

    nc = length(comps)
    offs = component_offsets(comps)
    blocks = Matrix{CFIEAlpertBlockCache{T}}(undef, nc, nc)

    global_rmin = typemax(T)
    global_rmax = zero(T)

    for a in 1:nc, b in 1:nc
        pa = comps[a]
        pb = comps[b]

        Ni = length(pa.xy)
        Nj = length(pb.xy)

        Xa = getindex.(pa.xy, 1)
        Ya = getindex.(pa.xy, 2)
        Xb = getindex.(pb.xy, 1)
        Yb = getindex.(pb.xy, 2)

        dXb = getindex.(pb.tangent, 1)
        dYb = getindex.(pb.tangent, 2)

        ΔX = reshape(Xa, Ni, 1) .- reshape(Xb, 1, Nj)
        ΔY = reshape(Ya, Ni, 1) .- reshape(Yb, 1, Nj)
        R  = hypot.(ΔX, ΔY)

        invR = similar(R)
        @inbounds for j in 1:Nj, i in 1:Ni
            rij = R[i,j]
            invR[i,j] = rij > eps(T) ? inv(rij) : zero(T)
        end

        dXbr = reshape(dXb, 1, Nj)
        dYbr = reshape(dYb, 1, Nj)
        inner = dYbr .* ΔX .- dXbr .* ΔY

        speed_j = sqrt.(dXb.^2 .+ dYb.^2)
        wj = copy(pb.ws)
        same = (a == b)

        rmin_blk = typemax(T)
        rmax_blk = zero(T)

        @inbounds for j in 1:Nj, i in 1:Ni
            same && i == j && continue
            rij = R[i,j]
            if rij > eps(T)
                rij < rmin_blk && (rmin_blk = rij)
                rij > rmax_blk && (rmax_blk = rij)
            end
        end

        @assert isfinite(rmin_blk) && rmax_blk > zero(T)

        rmin_blk = pad[1] * rmin_blk
        rmax_blk = pad[2] * rmax_blk
        global_rmin = min(global_rmin, rmin_blk)
        global_rmax = max(global_rmax, rmax_blk)

        pidx = Matrix{Int32}(undef, Ni, Nj)
        tloc = Matrix{Float64}(undef, Ni, Nj)

        blocks[a,b] = CFIEAlpertBlockCache{T}(
            same, offs[a], offs[b], Ni, Nj,
            R, invR, inner, speed_j, wj, pidx, tloc
        )
    end

    pref_plan = plan_h(
        0, 1, 1.0 + 0im, Float64(global_rmin), Float64(global_rmax);
        npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio
    )
    pans = pref_plan.panels

    for a in 1:nc, b in 1:nc
        blk = blocks[a,b]
        same = blk.same
        @inbounds for j in 1:blk.Nj, i in 1:blk.Ni
            if same && i == j
                blk.pidx[i,j] = Int32(1)
                blk.tloc[i,j] = 0.0
            else
                rij = Float64(blk.R[i,j])
                p = _find_panel(pref_plan, rij)
                P = pans[p]
                blk.pidx[i,j] = Int32(p)
                blk.tloc[i,j] = (2 * rij - (P.b + P.a)) / (P.b - P.a)
            end
        end
    end

    return CFIEAlpertBlockSystemCache{T}(
        blocks, offs, Float64(global_rmin), Float64(global_rmax)
    )
end

#################################
#### PLAN BUILDERS           ####
#################################

# build_CFIE_plans_alpert
# Build H0/H1 Chebyshev plans for a vector of complex wavenumbers.
function build_CFIE_plans_alpert(
    ks::AbstractVector{<:Number},
    rmin::Float64,
    rmax::Float64;
    npanels::Int=10000,
    M::Int=5,
    grading::Symbol=:uniform,
    geo_ratio::Real=1.05,
    nthreads::Int=1,
)
    Mk = length(ks)
    plans0 = Vector{ChebHankelPlanH}(undef, Mk)
    plans1 = Vector{ChebHankelPlanH}(undef, Mk)

    if nthreads <= 1 || Mk == 1
        @inbounds for m in 1:Mk
            k = ComplexF64(ks[m])
            plans0[m] = plan_h(0,1,k,rmin,rmax; npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio)
            plans1[m] = plan_h(1,1,k,rmin,rmax; npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio)
        end
    else
        nt = min(nthreads, Mk)
        chunks = Vector{UnitRange{Int}}(undef, nt)
        base = div(Mk, nt)
        remn = rem(Mk, nt)
        s = 1
        for t in 1:nt
            len = base + (t <= remn ? 1 : 0)
            chunks[t] = s:(s + len - 1)
            s += len
        end
        Threads.@threads for tid in 1:nt
            @inbounds for m in chunks[tid]
                k = ComplexF64(ks[m])
                plans0[m] = plan_h(0,1,k,rmin,rmax; npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio)
                plans1[m] = plan_h(1,1,k,rmin,rmax; npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio)
            end
        end
    end

    return plans0, plans1
end

# build_cfie_alpert_geom_workspace
# Build the geometry-only workspace reused by all k values.
function build_cfie_alpert_geom_workspace(
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}};
    npanels::Int=10000,
    M::Int=5,
    grading::Symbol=:uniform,
    geo_ratio::Real=1.05,
) where {T<:Real}

    rule = alpert_log_rule(T, solver.alpert_order)
    offs = component_offsets(pts)
    Gs = [cfie_geom_cache(p) for p in pts]
    Cs = [_build_alpert_component_cache(p, rule) for p in pts]

    topo_data = build_join_topology(pts)
    if topo_data === nothing
        topos = nothing
        gmaps = nothing
        panel_to_comp = nothing
    else
        topos, gmaps = topo_data
        panel_to_comp = zeros(Int, length(pts))
        @inbounds for c in eachindex(gmaps), a in gmaps[c]
            panel_to_comp[a] = c
        end
    end

    block_cache = build_cfie_alpert_block_caches(
        pts; npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio
    )
    Ntot = offs[end] - 1

    return CFIEAlpertGeomWorkspace(
        rule, offs, Gs, Cs, topos, gmaps, panel_to_comp, block_cache, Ntot
    )
end

###############################
#### THREAD-LOCAL H0/H1 WS ####
###############################

struct CFIEAlpertMultiBesselWorkspace
    h0_tls::Vector{Vector{ComplexF64}}
    h1_tls::Vector{Vector{ComplexF64}}
end

function CFIEAlpertMultiBesselWorkspace(Mk::Int; ntls::Int=Threads.nthreads())
    h0_tls = [Vector{ComplexF64}(undef, Mk) for _ in 1:ntls]
    h1_tls = [Vector{ComplexF64}(undef, Mk) for _ in 1:ntls]
    return CFIEAlpertMultiBesselWorkspace(h0_tls, h1_tls)
end

###############################
#### MULTI-K H0/H1 EVALS   ####
###############################

@inline function hankels_multi_ks_at_r!(
    h0vals::AbstractVector{ComplexF64},
    h1vals::AbstractVector{ComplexF64},
    plans0::AbstractVector{ChebHankelPlanH},
    plans1::AbstractVector{ChebHankelPlanH},
    pidx::Int32,
    t::Float64,
)
    @inbounds for m in eachindex(plans0)
        h0vals[m] = _cheb_clenshaw(plans0[m].panels[pidx].c, t)
        h1vals[m] = _cheb_clenshaw(plans1[m].panels[pidx].c, t)
    end
    return nothing
end

########################################
#### MULTI-K ALPERT SHIFT LOOKUP WS ####
########################################

struct AlpertPeriodicShiftChebCacheMultiK
    pidxp::Array{Int32,3}   # jcorr × N × Mk
    tlocp::Array{Float64,3}
    pidxm::Array{Int32,3}
    tlocm::Array{Float64,3}
end

struct AlpertSmoothPanelShiftChebCacheMultiK
    pidxp::Array{Int32,3}   # jcorr × N × Mk
    tlocp::Array{Float64,3}
    pidxm::Array{Int32,3}
    tlocm::Array{Float64,3}
end

@inline function _fill_shift_lookup!(
    pidxA::Array{Int32,3},
    tlocA::Array{Float64,3},
    r::Float64,
    m::Int,
    plans0::Vector{ChebHankelPlanH},
    p::Int,
    i::Int,
)
    Pm = plans0[m]
    q = _find_panel(Pm, r)
    P = Pm.panels[q]
    pidxA[p,i,m] = Int32(q)
    tlocA[p,i,m] = (2 * r - (P.b + P.a)) / (P.b - P.a)
    return nothing
end

function _build_alpert_periodic_shift_cheb_cache_multi_k(
    pts::BoundaryPointsCFIE{T},
    C::AlpertPeriodicCache{T},
    plans0::Vector{ChebHankelPlanH},
) where {T<:Real}

    X = getindex.(pts.xy, 1)
    Y = getindex.(pts.xy, 2)
    jcorr, N = size(C.xp)
    Mk = length(plans0)

    pidxp = Array{Int32,3}(undef, jcorr, N, Mk)
    tlocp = Array{Float64,3}(undef, jcorr, N, Mk)
    pidxm = Array{Int32,3}(undef, jcorr, N, Mk)
    tlocm = Array{Float64,3}(undef, jcorr, N, Mk)

    @inbounds for p in 1:jcorr, i in 1:N
        rp = hypot(X[i] - C.xp[p,i], Y[i] - C.yp[p,i])
        rm = hypot(X[i] - C.xm[p,i], Y[i] - C.ym[p,i])
        for m in 1:Mk
            _fill_shift_lookup!(pidxp, tlocp, Float64(rp), m, plans0, p, i)
            _fill_shift_lookup!(pidxm, tlocm, Float64(rm), m, plans0, p, i)
        end
    end

    return AlpertPeriodicShiftChebCacheMultiK(pidxp, tlocp, pidxm, tlocm)
end

function _build_alpert_smooth_panel_shift_cheb_cache_multi_k(
    pts::BoundaryPointsCFIE{T},
    C::AlpertSmoothPanelCache{T},
    plans0::Vector{ChebHankelPlanH},
) where {T<:Real}

    X = getindex.(pts.xy, 1)
    Y = getindex.(pts.xy, 2)
    jcorr, N = size(C.xp)
    Mk = length(plans0)

    pidxp = Array{Int32,3}(undef, jcorr, N, Mk)
    tlocp = Array{Float64,3}(undef, jcorr, N, Mk)
    pidxm = Array{Int32,3}(undef, jcorr, N, Mk)
    tlocm = Array{Float64,3}(undef, jcorr, N, Mk)

    @inbounds for p in 1:jcorr, i in 1:N
        rp = hypot(X[i] - C.xp[p,i], Y[i] - C.yp[p,i])
        rm = hypot(X[i] - C.xm[p,i], Y[i] - C.ym[p,i])
        for m in 1:Mk
            _fill_shift_lookup!(pidxp, tlocp, Float64(rp), m, plans0, p, i)
            _fill_shift_lookup!(pidxm, tlocm, Float64(rm), m, plans0, p, i)
        end
    end

    return AlpertSmoothPanelShiftChebCacheMultiK(pidxp, tlocp, pidxm, tlocm)
end

function _build_alpert_shift_cheb_cache_multi_k(
    pts::BoundaryPointsCFIE{T},
    C,
    plans0::Vector{ChebHankelPlanH},
) where {T<:Real}
    return pts.is_periodic ?
        _build_alpert_periodic_shift_cheb_cache_multi_k(pts, C, plans0) :
        _build_alpert_smooth_panel_shift_cheb_cache_multi_k(pts, C, plans0)
end

function build_alpert_shift_cheb_caches_multi_k(
    pts::Vector{BoundaryPointsCFIE{T}},
    Cs,
    plans0::Vector{ChebHankelPlanH},
) where {T<:Real}
    return [_build_alpert_shift_cheb_cache_multi_k(pts[a], Cs[a], plans0) for a in eachindex(pts)]
end

@inline function _shifted_hankels_multi_ks_at_r!(
    h0vals::AbstractVector{ComplexF64},
    h1vals::AbstractVector{ComplexF64},
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    SC::AlpertPeriodicShiftChebCacheMultiK,
    p::Int,
    i::Int,
    side::Symbol,
)
    if side === :plus
        @inbounds for m in eachindex(plans0)
            h0vals[m] = _cheb_clenshaw(plans0[m].panels[SC.pidxp[p,i,m]].c, SC.tlocp[p,i,m])
            h1vals[m] = _cheb_clenshaw(plans1[m].panels[SC.pidxp[p,i,m]].c, SC.tlocp[p,i,m])
        end
    else
        @inbounds for m in eachindex(plans0)
            h0vals[m] = _cheb_clenshaw(plans0[m].panels[SC.pidxm[p,i,m]].c, SC.tlocm[p,i,m])
            h1vals[m] = _cheb_clenshaw(plans1[m].panels[SC.pidxm[p,i,m]].c, SC.tlocm[p,i,m])
        end
    end
    return nothing
end

@inline function _shifted_hankels_multi_ks_at_r!(
    h0vals::AbstractVector{ComplexF64},
    h1vals::AbstractVector{ComplexF64},
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    SC::AlpertSmoothPanelShiftChebCacheMultiK,
    p::Int,
    i::Int,
    side::Symbol,
)
    if side === :plus
        @inbounds for m in eachindex(plans0)
            h0vals[m] = _cheb_clenshaw(plans0[m].panels[SC.pidxp[p,i,m]].c, SC.tlocp[p,i,m])
            h1vals[m] = _cheb_clenshaw(plans1[m].panels[SC.pidxp[p,i,m]].c, SC.tlocp[p,i,m])
        end
    else
        @inbounds for m in eachindex(plans0)
            h0vals[m] = _cheb_clenshaw(plans0[m].panels[SC.pidxm[p,i,m]].c, SC.tlocm[p,i,m])
            h1vals[m] = _cheb_clenshaw(plans1[m].panels[SC.pidxm[p,i,m]].c, SC.tlocm[p,i,m])
        end
    end
    return nothing
end

###########################################################
#### CONTEXT BUILDERS (REUSABLE FOR BEYN / MULTI-k)    ####
###########################################################

# Keep this symmetry-aware r-range helper only for plan construction.
# It must match the image families that are actually added in assembly.
function estimate_rmin_rmax_cfie_alpert(
    pts::Vector{BoundaryPointsCFIE{T}},
    symmetry::Union{Nothing,Vector{Any}}=nothing;
    pad=(T(0.9), T(1.1)),
    rmax_factor::Real=1.0,
) where {T<:Real}

    nc = length(pts)
    @assert nc > 0
    tol2 = (eps(T))^2

    counts = [length(p.xy) for p in pts]
    offs = Vector{Int}(undef, nc)
    s = 1
    @inbounds for c in 1:nc
        offs[c] = s
        s += counts[c]
    end
    totalN = sum(counts)

    @inline function decode_idx(idx::Int)
        c = searchsortedlast(offs, idx)
        i = idx - offs[c] + 1
        return c, i
    end

    shift_x = hasproperty(pts[1], :shift_x) ? getproperty(pts[1], :shift_x) : zero(T)
    shift_y = hasproperty(pts[1], :shift_y) ? getproperty(pts[1], :shift_y) : zero(T)

    @inline reflect_y_axis(x::T,y::T) = (T(2)*shift_x - x, y)
    @inline reflect_x_axis(x::T,y::T) = (x, T(2)*shift_y - y)
    @inline reflect_origin(x::T,y::T) = (T(2)*shift_x - x, T(2)*shift_y - y)

    nth = Threads.nthreads()
    min2_tls = fill(T(Inf), nth)
    max2_tls = fill(zero(T), nth)

    Threads.@threads for idx in 1:totalN
        ca, ia = decode_idx(idx)
        xa, ya = pts[ca].xy[ia]

        lmin2 = typemax(T)
        lmax2 = zero(T)

        @inbounds for cb in 1:nc
            pb = pts[cb]
            Nb = length(pb.xy)
            for jb in 1:Nb
                xj, yj = pb.xy[jb]

                if !(ca == cb && ia == jb)
                    dx = xa - xj
                    dy = ya - yj
                    d2 = muladd(dx, dx, dy*dy)
                    if d2 > tol2
                        d2 < lmin2 && (lmin2 = d2)
                        d2 > lmax2 && (lmax2 = d2)
                    end
                end

                if symmetry !== nothing
                    for sym in symmetry
                        if sym isa Reflection
                            if sym.axis === :y_axis
                                xr, yr = reflect_y_axis(xj, yj)
                                dx = xa - xr; dy = ya - yr
                                d2 = muladd(dx, dx, dy*dy)
                                if d2 > tol2
                                    d2 < lmin2 && (lmin2 = d2)
                                    d2 > lmax2 && (lmax2 = d2)
                                end
                            elseif sym.axis === :x_axis
                                xr, yr = reflect_x_axis(xj, yj)
                                dx = xa - xr; dy = ya - yr
                                d2 = muladd(dx, dx, dy*dy)
                                if d2 > tol2
                                    d2 < lmin2 && (lmin2 = d2)
                                    d2 > lmax2 && (lmax2 = d2)
                                end
                            elseif sym.axis === :origin
                                xr, yr = reflect_y_axis(xj, yj)
                                dx = xa - xr; dy = ya - yr
                                d2 = muladd(dx, dx, dy*dy)
                                if d2 > tol2
                                    d2 < lmin2 && (lmin2 = d2)
                                    d2 > lmax2 && (lmax2 = d2)
                                end

                                xr, yr = reflect_x_axis(xj, yj)
                                dx = xa - xr; dy = ya - yr
                                d2 = muladd(dx, dx, dy*dy)
                                if d2 > tol2
                                    d2 < lmin2 && (lmin2 = d2)
                                    d2 > lmax2 && (lmax2 = d2)
                                end

                                xr, yr = reflect_origin(xj, yj)
                                dx = xa - xr; dy = ya - yr
                                d2 = muladd(dx, dx, dy*dy)
                                if d2 > tol2
                                    d2 < lmin2 && (lmin2 = d2)
                                    d2 > lmax2 && (lmax2 = d2)
                                end
                            else
                                error("Unknown reflection axis $(sym.axis)")
                            end

                        elseif sym isa Rotation
                            q = SVector{2,T}(xj, yj)
                            costab, sintab, _χ = _rotation_tables(T, sym.n, mod(sym.m, sym.n))
                            for l in 1:(sym.n - 1)
                                qimg = image_point(sym, q, l, costab, sintab)
                                dx = xa - qimg[1]
                                dy = ya - qimg[2]
                                d2 = muladd(dx, dx, dy*dy)
                                if d2 > tol2
                                    d2 < lmin2 && (lmin2 = d2)
                                    d2 > lmax2 && (lmax2 = d2)
                                end
                            end
                        else
                            error("Unknown symmetry type $(typeof(sym))")
                        end
                    end
                end
            end
        end

        tid = Threads.threadid()
        lmin2 < min2_tls[tid] && (min2_tls[tid] = lmin2)
        lmax2 > max2_tls[tid] && (max2_tls[tid] = lmax2)
    end

    min2 = minimum(min2_tls)
    max2 = maximum(max2_tls)

    @assert isfinite(min2) && max2 > zero(T) "estimate_rmin_rmax_cfie_alpert: degenerate geometry"

    rmin = pad[1] * sqrt(min2)
    rmax = pad[2] * rmax_factor * sqrt(max2)
    return Float64(rmin), Float64(rmax)
end

# build_cfie_alpert_cheb_data
# Build all reusable multi-k Alpert Chebyshev objects in one call.
function build_cfie_alpert_cheb_data(
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    ks::AbstractVector{<:Number};
    npanels::Int=15000,
    M::Int=5,
    grading::Symbol=:uniform,
    geo_ratio::Real=1.05,
    nthreads::Int=Threads.nthreads(),
) where {T<:Real}

    geomws = build_cfie_alpert_geom_workspace(
        solver, pts; npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio
    )

    if isnothing(solver.symmetry)
        rmin = geomws.block_cache.rmin
        rmax = geomws.block_cache.rmax
    else
        rmin, rmax = estimate_rmin_rmax_cfie_alpert(
            pts, solver.symmetry; pad=(0.9,1.1), rmax_factor=1.0
        )
    end

    plans0, plans1 = build_CFIE_plans_alpert(
        ks, rmin, rmax;
        npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio, nthreads=nthreads
    )

    SCs = build_alpert_shift_cheb_caches_multi_k(pts, geomws.Cs, plans0)
    wsb = CFIEAlpertMultiBesselWorkspace(length(ks); ntls=Threads.nthreads())

    return geomws, plans0, plans1, SCs, wsb
end

# Single-k convenience version.
function build_cfie_alpert_cheb_data(
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    k::Number;
    npanels::Int=15000,
    M::Int=5,
    grading::Symbol=:uniform,
    geo_ratio::Real=1.05,
) where {T<:Real}

    geomws = build_cfie_alpert_geom_workspace(
        solver, pts; npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio
    )

    if isnothing(solver.symmetry)
        rmin = geomws.block_cache.rmin
        rmax = geomws.block_cache.rmax
    else
        rmin, rmax = estimate_rmin_rmax_cfie_alpert(
            pts, solver.symmetry; pad=(0.9,1.1), rmax_factor=1.0
        )
    end

    plans0, plans1 = build_CFIE_plans_alpert(
        [k], rmin, rmax;
        npanels=npanels, M=M, grading=grading, geo_ratio=geo_ratio, nthreads=1
    )

    SCs = build_alpert_shift_cheb_caches_multi_k(pts, geomws.Cs, plans0)
    wsb = CFIEAlpertMultiBesselWorkspace(1; ntls=Threads.nthreads())

    return geomws, plans0[1], plans1[1], SCs, wsb
end

#################################################################
#   CHEBYSHEV-BASED SLP/DLP EVALUATION FOR CFIE_alpert ASSEMBLY
#   Faithful multi-k port of the old scalar Alpert assembly.
#
# This batch contains:
#   1) helper utilities reused by composite assembly
#   2) direct no-symmetry multi-k assembly
#   3) symmetry image contributions
#   4) high-level entry points
#   5) Beyn-facing construct_boundary_matrices!
#
# MO 2/4/26
#################################################################

#################################
#### COMPOSITE HELPER UTILS  ####
#################################

@inline function _component_id_of_panel(a::Int, gmaps::Vector{Vector{Int}})
    @inbounds for c in eachindex(gmaps)
        a in gmaps[c] && return c
    end
    return 0
end

@inline function _right_neighbor_excluded_count(i::Int, N::Int, a::Int)
    return max(0, i + a - 1 - N)
end

@inline function _left_neighbor_excluded_count(i::Int, a::Int)
    return max(0, a - i)
end

@inline function _panel_xy_tangent_arrays(pts::BoundaryPointsCFIE{T}) where {T<:Real}
    X  = getindex.(pts.xy, 1)
    Y  = getindex.(pts.xy, 2)
    dX = getindex.(pts.tangent, 1)
    dY = getindex.(pts.tangent, 2)
    return X, Y, dX, dY
end

@inline function _eval_on_open_panel_local4(pts::BoundaryPointsCFIE{T}, u::T) where {T<:Real}
    X, Y, dX, dY = _panel_xy_tangent_arrays(pts)
    h = pts.ws[1]
    return _eval_shifted_source_smooth_panel_local4(u, h, X, Y, dX, dY)
end

########################################################
#### MULTI-K DIRECT NO-SYMMETRY CFIE_alpert ASSEMBLY ####
########################################################

# _all_k_nosymm_CFIE_alpert_chebyshev!
# Assemble all CFIE_alpert matrices for all k values at once, without
# symmetry-image terms.
#
# This is a faithful multi-k rewrite of the old scalar Alpert assembly:
#   - same diagonal terms
#   - same DLP trapezoid terms
#   - same SLP far-part exclusions
#   - same Alpert near corrections
#   - same composite smooth-join logic
function _all_k_nosymm_CFIE_alpert_chebyshev!(
    As::Vector{Matrix{ComplexF64}},
    pts::Vector{BoundaryPointsCFIE{T}},
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    geomws::CFIEAlpertGeomWorkspace{T},
    SCs,
    wsb::CFIEAlpertMultiBesselWorkspace;
    multithreaded::Bool=true,
) where {T<:Real}

    Mk = length(plans0)

    αD = Vector{ComplexF64}(undef, Mk)
    iks = Vector{ComplexF64}(undef, Mk)
    @inbounds for m in 1:Mk
        km = ComplexF64(plans0[m].k)
        αD[m] = 0.5im * km
        iks[m] = 1im * km
        fill!(As[m], 0.0 + 0.0im)
    end
    αS = 0.5im

    offs         = geomws.offs
    Gs           = geomws.Gs
    Cs           = geomws.Cs
    blocks       = geomws.block_cache.blocks
    topos        = geomws.topos
    gmaps        = geomws.gmaps
    panel_to_comp = geomws.panel_to_comp
    rule         = geomws.rule
    nc           = length(pts)

    #################################################
    #### PERIODIC SELF BLOCK: faithful old logic ####
    #################################################

    function self_periodic_col!(a::Int, i::Int, h0vals::Vector{ComplexF64}, h1vals::Vector{ComplexF64})
        ptsa = pts[a]
        Ga   = Gs[a]
        Ca   = Cs[a]
        SCa  = SCs[a]
        blk  = blocks[a,a]

        row_range = offs[a]:(offs[a+1]-1)
        N    = length(ptsa.ts)
        jcorr = rule.j
        hh   = ptsa.ws[1]
        aa   = rule.a

        gi = row_range[i]
        si = Ga.speed[i]
        κi = Ga.kappa[i]

        # diagonal
        @inbounds for m in 1:Mk
            As[m][gi,gi] += 1.0 - ComplexF64(hh * si * κi, 0.0)
        end

        # DLP off-diagonal
        @inbounds for j in 1:N
            j == i && continue
            gj = row_range[j]
            hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, blk.pidx[i,j], blk.tloc[i,j])
            invr = blk.invR[i,j]
            inn  = blk.inner[i,j]
            for m in 1:Mk
                As[m][gi,gj] -= hh * (αD[m] * inn * h1vals[m] * invr)
            end
        end

        # SLP far part
        @inbounds for j in 1:N
            j == i && continue
            mm = j - i
            mm >  N ÷ 2 && (mm -= N)
            mm < -N ÷ 2 && (mm += N)
            abs(mm) < aa && continue

            gj = row_range[j]
            hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, blk.pidx[i,j], blk.tloc[i,j])
            sj = Ga.speed[j]
            for m in 1:Mk
                As[m][gi,gj] -= iks[m] * (hh * (αS * h0vals[m] * sj))
            end
        end

        # Alpert near correction
        @inbounds for p in 1:jcorr
            fac = hh * rule.w[p]

            _shifted_hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, SCa, p, i, :plus)
            sp = Ca.sp[p,i]
            for m in 1:Mk
                coeff = -iks[m] * (fac * (αS * h0vals[m] * sp))
                for m4 in 1:4
                    q = Ca.idxp[p,i,m4]
                    As[m][gi, row_range[q]] += coeff * Ca.wtp[p,i,m4]
                end
            end

            _shifted_hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, SCa, p, i, :minus)
            sm = Ca.sm[p,i]
            for m in 1:Mk
                coeff = -iks[m] * (fac * (αS * h0vals[m] * sm))
                for m4 in 1:4
                    q = Ca.idxm[p,i,m4]
                    As[m][gi, row_range[q]] += coeff * Ca.wtm[p,i,m4]
                end
            end
        end

        return nothing
    end

    #####################################################
    #### SMOOTH PANEL SELF BLOCK: faithful old logic ####
    #####################################################

    function self_smooth_col!(a::Int, i::Int, h0vals::Vector{ComplexF64}, h1vals::Vector{ComplexF64})
        ptsa = pts[a]
        Ga   = Gs[a]
        Ca   = Cs[a]
        SCa  = SCs[a]
        blk  = blocks[a,a]

        row_range = offs[a]:(offs[a+1]-1)
        N     = length(ptsa.xy)
        hh    = ptsa.ws[1]
        aa    = rule.a
        jcorr = rule.j

        gi = row_range[i]
        si = Ga.speed[i]
        κi = Ga.kappa[i]
        ui = Ca.us[i]

        # diagonal
        @inbounds for m in 1:Mk
            As[m][gi,gi] += 1.0 - ComplexF64(hh * si * κi, 0.0)
        end

        # DLP off-diagonal
        @inbounds for j in 1:N
            j == i && continue
            gj = row_range[j]
            hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, blk.pidx[i,j], blk.tloc[i,j])
            invr = blk.invR[i,j]
            inn  = blk.inner[i,j]
            for m in 1:Mk
                As[m][gi,gj] -= hh * (αD[m] * inn * h1vals[m] * invr)
            end
        end

        # SLP far part
        @inbounds for j in 1:N
            j == i && continue
            abs(j - i) < aa && continue
            gj = row_range[j]
            hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, blk.pidx[i,j], blk.tloc[i,j])
            sj = Ga.speed[j]
            for m in 1:Mk
                As[m][gi,gj] -= iks[m] * (hh * (αS * h0vals[m] * sj))
            end
        end

        # near correction, respecting panel endpoints
        @inbounds for p in 1:jcorr
            fac = hh * rule.w[p]
            Δu  = hh * rule.x[p]

            if ui + Δu < one(T)
                _shifted_hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, SCa, p, i, :plus)
                sp = Ca.sp[p,i]
                for m in 1:Mk
                    coeff = -iks[m] * (fac * (αS * h0vals[m] * sp))
                    for m4 in 1:4
                        q = Ca.idxp[p,i,m4]
                        As[m][gi, row_range[q]] += coeff * Ca.wtp[p,i,m4]
                    end
                end
            end

            if ui - Δu > zero(T)
                _shifted_hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, SCa, p, i, :minus)
                sm = Ca.sm[p,i]
                for m in 1:Mk
                    coeff = -iks[m] * (fac * (αS * h0vals[m] * sm))
                    for m4 in 1:4
                        q = Ca.idxm[p,i,m4]
                        As[m][gi, row_range[q]] += coeff * Ca.wtm[p,i,m4]
                    end
                end
            end
        end

        return nothing
    end

    ##############################################################
    #### COMPOSITE SELF BLOCK: faithful old smooth-join logic ####
    ##############################################################

    function composite_self_col!(c::Int, l::Int, i::Int, h0vals::Vector{ComplexF64}, h1vals::Vector{ComplexF64})
        gmap = gmaps[c]
        topo = topos[c]

        aidx = gmap[l]
        pa   = pts[aidx]
        Ga   = Gs[aidx]
        Ca   = Cs[aidx]
        SCa  = SCs[aidx]
        ra   = offs[aidx]:(offs[aidx+1]-1)

        Xa = getindex.(pa.xy, 1)
        Ya = getindex.(pa.xy, 2)

        Na = length(pa.xy)
        ha = pa.ws[1]

        gi = ra[i]
        xi = Xa[i]
        yi = Ya[i]
        si = Ga.speed[i]
        κi = Ga.kappa[i]
        ui = Ca.us[i]

        left_smooth  = topo.left_kind[l]  === :smooth
        right_smooth = topo.right_kind[l] === :smooth

        lprev = topo.prev[l]
        lnext = topo.next[l]

        prev_idx = (lprev == 0) ? 0 : gmap[lprev]
        next_idx = (lnext == 0) ? 0 : gmap[lnext]

        prev_pts = (prev_idx == 0) ? nothing : pts[prev_idx]
        next_pts = (next_idx == 0) ? nothing : pts[next_idx]

        prev_ra = (prev_idx == 0) ? (1:0) : (offs[prev_idx]:(offs[prev_idx+1]-1))
        next_ra = (next_idx == 0) ? (1:0) : (offs[next_idx]:(offs[next_idx+1]-1))

        # diagonal
        @inbounds for m in 1:Mk
            As[m][gi,gi] += 1.0 - ComplexF64(ha * si * κi, 0.0)
        end

        # direct component-to-component contributions inside the composite component
        for mm in eachindex(gmap)
            bidx = gmap[mm]
            pb   = pts[bidx]
            rb   = offs[bidx]:(offs[bidx+1]-1)
            blk  = blocks[aidx,bidx]

            Nb  = length(pb.xy)
            Xb  = getindex.(pb.xy, 1)
            Yb  = getindex.(pb.xy, 2)
            dXb = getindex.(pb.tangent, 1)
            dYb = getindex.(pb.tangent, 2)
            sb  = @. sqrt(dXb^2 + dYb^2)

            for j in 1:Nb
                gj = rb[j]

                # DLP off-diagonal
                if !(bidx == aidx && j == i)
                    hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, blk.pidx[i,j], blk.tloc[i,j])
                    invr = blk.invR[i,j]
                    inn  = blk.inner[i,j]
                    for m in 1:Mk
                        As[m][gi,gj] -= pb.ws[j] * (αD[m] * inn * h1vals[m] * invr)
                    end
                end

                # SLP exclusion logic exactly as in old scalar route
                skip_slp = false
                if bidx == aidx
                    skip_slp = abs(j - i) < rule.a
                elseif right_smooth && bidx == next_idx
                    nr = _right_neighbor_excluded_count(i, Na, rule.a)
                    skip_slp = (j <= nr)
                elseif left_smooth && bidx == prev_idx
                    nl = _left_neighbor_excluded_count(i, rule.a)
                    skip_slp = (j > Nb - nl)
                end

                if !skip_slp
                    hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, blk.pidx[i,j], blk.tloc[i,j])
                    sj  = sb[j]
                    fac = pb.ws[j]
                    for m in 1:Mk
                        As[m][gi,gj] -= iks[m] * (fac * (αS * h0vals[m] * sj))
                    end
                end
            end
        end

        # local same-panel Alpert correction
        @inbounds for p in 1:rule.j
            fac = ha * rule.w[p]
            Δu  = ha * rule.x[p]

            if ui + Δu < one(T)
                _shifted_hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, SCa, p, i, :plus)
                sp = Ca.sp[p,i]
                for m in 1:Mk
                    coeff = -iks[m] * (fac * (αS * h0vals[m] * sp))
                    for m4 in 1:4
                        q = Ca.idxp[p,i,m4]
                        As[m][gi, ra[q]] += coeff * Ca.wtp[p,i,m4]
                    end
                end
            end

            if ui - Δu > zero(T)
                _shifted_hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, SCa, p, i, :minus)
                sm = Ca.sm[p,i]
                for m in 1:Mk
                    coeff = -iks[m] * (fac * (αS * h0vals[m] * sm))
                    for m4 in 1:4
                        q = Ca.idxm[p,i,m4]
                        As[m][gi, ra[q]] += coeff * Ca.wtm[p,i,m4]
                    end
                end
            end
        end

        # right smooth neighbor continuation correction
        if right_smooth && next_idx != 0
            for p in 1:rule.j
                Δu = ha * rule.x[p]
                if ui + Δu >= one(T)
                    u2 = ui + Δu - one(T)
                    x, y, tx, ty, s2, idx2, wt2 = _eval_on_open_panel_local4(next_pts, u2)
                    r = hypot(xi - x, yi - y)
                    if isfinite(r) && r > sqrt(eps(T))
                        @inbounds for m in 1:Mk
                            q = _find_panel(plans0[m], Float64(r))
                            P = plans0[m].panels[q]
                            tloc = (2 * Float64(r) - (P.b + P.a)) / (P.b - P.a)
                            h0vals[m] = _cheb_clenshaw(plans0[m].panels[q].c, tloc)
                            h1vals[m] = _cheb_clenshaw(plans1[m].panels[q].c, tloc)
                        end
                        fac = ha * rule.w[p]
                        for m in 1:Mk
                            coeff = -iks[m] * (fac * (αS * h0vals[m] * s2))
                            @inbounds for m4 in 1:4
                                q = idx2[m4]
                                As[m][gi, next_ra[q]] += coeff * wt2[m4]
                            end
                        end
                    end
                end
            end
        end

        # left smooth neighbor continuation correction
        if left_smooth && prev_idx != 0
            for p in 1:rule.j
                Δu = ha * rule.x[p]
                if ui - Δu <= zero(T)
                    u2 = one(T) + ui - Δu
                    x, y, tx, ty, s2, idx2, wt2 = _eval_on_open_panel_local4(prev_pts, u2)
                    r = hypot(xi - x, yi - y)
                    if isfinite(r) && r > sqrt(eps(T))
                        @inbounds for m in 1:Mk
                            q = _find_panel(plans0[m], Float64(r))
                            P = plans0[m].panels[q]
                            tloc = (2 * Float64(r) - (P.b + P.a)) / (P.b - P.a)
                            h0vals[m] = _cheb_clenshaw(plans0[m].panels[q].c, tloc)
                            h1vals[m] = _cheb_clenshaw(plans1[m].panels[q].c, tloc)
                        end
                        fac = ha * rule.w[p]
                        for m in 1:Mk
                            coeff = -iks[m] * (fac * (αS * h0vals[m] * s2))
                            @inbounds for m4 in 1:4
                                q = idx2[m4]
                                As[m][gi, prev_ra[q]] += coeff * wt2[m4]
                            end
                        end
                    end
                end
            end
        end

        return nothing
    end

    #########################################
    #### DIRECT OFF-DIAGONAL BLOCKS      ####
    #########################################

    function offdiag_col!(a::Int, b::Int, j::Int, h0vals::Vector{ComplexF64}, h1vals::Vector{ComplexF64})
        blk = blocks[a,b]
        ro  = blk.row_offset
        co  = blk.col_offset
        gj  = co + j - 1
        sj  = blk.speed_j[j]
        wj  = blk.wj[j]

        @inbounds for i in 1:blk.Ni
            gi = ro + i - 1
            hankels_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, blk.pidx[i,j], blk.tloc[i,j])
            invr = blk.invR[i,j]
            inn  = blk.inner[i,j]
            for m in 1:Mk
                dval = wj * (αD[m] * inn * h1vals[m] * invr)
                sval = wj * (αS * h0vals[m] * sj)
                As[m][gi,gj] -= (dval + iks[m] * sval)
            end
        end

        return nothing
    end

    #########################################
    #### SELF ASSEMBLY DRIVER            ####
    #########################################

    if topos === nothing
        for a in 1:nc
            if pts[a].is_periodic
                @use_threads multithreading=multithreaded for i in 1:length(pts[a].ts)
                    tid = Threads.threadid()
                    self_periodic_col!(a, i, wsb.h0_tls[tid], wsb.h1_tls[tid])
                end
            else
                @use_threads multithreading=multithreaded for i in 1:length(pts[a].xy)
                    tid = Threads.threadid()
                    self_smooth_col!(a, i, wsb.h0_tls[tid], wsb.h1_tls[tid])
                end
            end
        end
    else
        @inbounds for c in eachindex(gmaps)
            gmap = gmaps[c]
            if length(gmap) == 1 && pts[gmap[1]].is_periodic
                a = gmap[1]
                @use_threads multithreading=multithreaded for i in 1:length(pts[a].ts)
                    tid = Threads.threadid()
                    self_periodic_col!(a, i, wsb.h0_tls[tid], wsb.h1_tls[tid])
                end
            else
                @inbounds for l in eachindex(gmap)
                    aidx = gmap[l]
                    Na = length(pts[aidx].xy)
                    @use_threads multithreading=multithreaded for i in 1:Na
                        tid = Threads.threadid()
                        composite_self_col!(c, l, i, wsb.h0_tls[tid], wsb.h1_tls[tid])
                    end
                end
            end
        end
    end

    #########################################
    #### DIRECT NONSELF BLOCKS           ####
    #########################################

    for a in 1:nc, b in 1:nc
        a == b && continue
        if panel_to_comp !== nothing
            ca = panel_to_comp[a]
            cb = panel_to_comp[b]
            ca != 0 && ca == cb && continue
        end
        @use_threads multithreading=multithreaded for j in 1:blocks[a,b].Nj
            tid = Threads.threadid()
            offdiag_col!(a, b, j, wsb.h0_tls[tid], wsb.h1_tls[tid])
        end
    end

    return nothing
end

#######################################################
#### MULTI-K SYMMETRY IMAGE CONTRIBUTION ADDITIONS ####
#######################################################

# _add_image_block_chebyshev_multi_k!
# Add one symmetry image block contribution for all k values.
#
# This is the multi-k analogue of the old scalar `_add_image_block!`.
function _add_image_block_chebyshev_multi_k!(
    As::Vector{Matrix{ComplexF64}},
    ra::UnitRange{Int},
    rb::UnitRange{Int},
    pa::BoundaryPointsCFIE{T},
    pb::BoundaryPointsCFIE{T},
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    h0_tls::Vector{Vector{ComplexF64}},
    h1_tls::Vector{Vector{ComplexF64}},
    qfun,
    tfun,
    weight;
    multithreaded::Bool=true,
) where {T<:Real}

    Mk = length(plans0)
    αD = Vector{ComplexF64}(undef, Mk)
    iks = Vector{ComplexF64}(undef, Mk)
    @inbounds for m in 1:Mk
        km = ComplexF64(plans0[m].k)
        αD[m] = 0.5im * km
        iks[m] = 1im * km
    end
    αS = 0.5im

    Na = length(pa.xy)
    Nb = length(pb.xy)
    Xa = getindex.(pa.xy, 1)
    Ya = getindex.(pa.xy, 2)

    @use_threads multithreading=multithreaded for j in 1:Nb
        tid = Threads.threadid()
        h0vals = h0_tls[tid]
        h1vals = h1_tls[tid]

        gj = rb[j]
        qimg = qfun(pb.xy[j])
        timg = tfun(pb.tangent[j])

        xj = qimg[1]
        yj = qimg[2]
        txj = timg[1]
        tyj = timg[2]

        sj = sqrt(txj * txj + tyj * tyj)
        wd = pb.ws[j]
        ws = pb.ws[j] * sj

        @inbounds for i in 1:Na
            gi = ra[i]
            r = hypot(Xa[i] - xj, Ya[i] - yj)
            r <= eps(T) && continue

            @inbounds for m in 1:Mk
                p = _find_panel(plans0[m], Float64(r))
                P = plans0[m].panels[p]
                tloc = (2 * Float64(r) - (P.b + P.a)) / (P.b - P.a)
                h0vals[m] = _cheb_clenshaw(plans0[m].panels[p].c, tloc)
                h1vals[m] = _cheb_clenshaw(plans1[m].panels[p].c, tloc)
            end

            dx = Xa[i] - xj
            dy = Ya[i] - yj
            invr = inv(r)
            inn = tyj * dx - txj * dy

            @inbounds for m in 1:Mk
                dval = weight * wd * (αD[m] * inn * h1vals[m] * invr)
                sval = weight * ws * (αS * h0vals[m])
                As[m][gi,gj] -= (dval + iks[m] * sval)
            end
        end
    end

    return nothing
end

function _assemble_reflection_images_chebyshev_multi_k!(
    As::Vector{Matrix{ComplexF64}},
    ra::UnitRange{Int},
    rb::UnitRange{Int},
    pa::BoundaryPointsCFIE{T},
    pb::BoundaryPointsCFIE{T},
    solver::CFIE_alpert{T},
    billiard,
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    h0_tls::Vector{Vector{ComplexF64}},
    h1_tls::Vector{Vector{ComplexF64}},
    sym::Reflection;
    multithreaded::Bool=true,
) where {T<:Real}

    if sym.axis == :y_axis
        _add_image_block_chebyshev_multi_k!(As, ra, rb, pa, pb, plans0, plans1, h0_tls, h1_tls,
            q -> image_point_x(q, billiard), t -> image_tangent_x(t), image_weight(sym);
            multithreaded=multithreaded)
    elseif sym.axis == :x_axis
        _add_image_block_chebyshev_multi_k!(As, ra, rb, pa, pb, plans0, plans1, h0_tls, h1_tls,
            q -> image_point_y(q, billiard), t -> image_tangent_y(t), image_weight(sym);
            multithreaded=multithreaded)
    elseif sym.axis == :origin
        σx  = image_weight_x(sym)
        σy  = image_weight_y(sym)
        σxy = image_weight_xy(sym)

        _add_image_block_chebyshev_multi_k!(As, ra, rb, pa, pb, plans0, plans1, h0_tls, h1_tls,
            q -> image_point_x(q, billiard), t -> image_tangent_x(t), σx;
            multithreaded=multithreaded)
        _add_image_block_chebyshev_multi_k!(As, ra, rb, pa, pb, plans0, plans1, h0_tls, h1_tls,
            q -> image_point_y(q, billiard), t -> image_tangent_y(t), σy;
            multithreaded=multithreaded)
        _add_image_block_chebyshev_multi_k!(As, ra, rb, pa, pb, plans0, plans1, h0_tls, h1_tls,
            q -> image_point_xy(q, billiard), t -> image_tangent_xy(t), σxy;
            multithreaded=multithreaded)
    else
        error("Unknown reflection axis $(sym.axis)")
    end

    return nothing
end

function _assemble_rotation_images_chebyshev_multi_k!(
    As::Vector{Matrix{ComplexF64}},
    ra::UnitRange{Int},
    rb::UnitRange{Int},
    pa::BoundaryPointsCFIE{T},
    pb::BoundaryPointsCFIE{T},
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    h0_tls::Vector{Vector{ComplexF64}},
    h1_tls::Vector{Vector{ComplexF64}},
    sym::Rotation,
    costab,
    sintab,
    χ;
    multithreaded::Bool=true,
) where {T<:Real}

    for l in 1:(sym.n - 1)
        phase = χ[l + 1]
        _add_image_block_chebyshev_multi_k!(As, ra, rb, pa, pb, plans0, plans1, h0_tls, h1_tls,
            q -> image_point(sym, q, l, costab, sintab),
            t -> image_tangent(sym, t, l, costab, sintab),
            phase;
            multithreaded=multithreaded)
    end

    return nothing
end

#################################
#### HIGH LEVEL ENTRY POINT  ####
#################################

# compute_kernel_matrices_CFIE_alpert_chebyshev!
# Assemble all CFIE_alpert matrices for all k values.
#
# First assembles the direct no-symmetry operator on the desymmetrized
# boundary, then adds all symmetry-image families if symmetry is present.
function compute_kernel_matrices_CFIE_alpert_chebyshev!(
    As::Vector{Matrix{ComplexF64}},
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    geomws::CFIEAlpertGeomWorkspace{T},
    SCs,
    wsb::CFIEAlpertMultiBesselWorkspace;
    multithreaded::Bool=true,
) where {T<:Real}

    _all_k_nosymm_CFIE_alpert_chebyshev!(As, pts, plans0, plans1, geomws, SCs, wsb;
        multithreaded=multithreaded)

    if isnothing(solver.symmetry)
        return nothing
    end

    offs = geomws.offs
    for sym in solver.symmetry
        if sym isa Reflection
            for a in eachindex(pts), b in eachindex(pts)
                ra = offs[a]:(offs[a+1]-1)
                rb = offs[b]:(offs[b+1]-1)
                _assemble_reflection_images_chebyshev_multi_k!(
                    As, ra, rb, pts[a], pts[b], solver, solver.billiard,
                    plans0, plans1, wsb.h0_tls, wsb.h1_tls, sym;
                    multithreaded=multithreaded
                )
            end
        elseif sym isa Rotation
            costab, sintab, χ = _rotation_tables(T, sym.n, sym.m)
            for a in eachindex(pts), b in eachindex(pts)
                ra = offs[a]:(offs[a+1]-1)
                rb = offs[b]:(offs[b+1]-1)
                _assemble_rotation_images_chebyshev_multi_k!(
                    As, ra, rb, pts[a], pts[b],
                    plans0, plans1, wsb.h0_tls, wsb.h1_tls,
                    sym, costab, sintab, χ;
                    multithreaded=multithreaded
                )
            end
        else
            error("Unknown symmetry type $(typeof(sym))")
        end
    end

    return nothing
end

# Single-k wrapper
function compute_kernel_matrices_CFIE_alpert_chebyshev!(
    A::Matrix{ComplexF64},
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    plan0::ChebHankelPlanH,
    plan1::ChebHankelPlanH,
    geomws::CFIEAlpertGeomWorkspace{T},
    SCs,
    wsb::CFIEAlpertMultiBesselWorkspace;
    multithreaded::Bool=true,
) where {T<:Real}
    compute_kernel_matrices_CFIE_alpert_chebyshev!(
        [A], solver, pts, [plan0], [plan1], geomws, SCs, wsb;
        multithreaded=multithreaded
    )
    return nothing
end

###########################################################
#### BEYN-FACING MATRIX CONSTRUCTION ENTRY POINT       ####
###########################################################

# construct_boundary_matrices!
# Construct all contour matrices T(zj) for Beyn's method using the
# multi-k CFIE_alpert Chebyshev route.
function construct_boundary_matrices!(
    Tbufs::Vector{Matrix{Complex{T}}},
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    zj::AbstractVector{Complex{T}};
    multithreaded::Bool=true,
    use_chebyshev::Bool=true,
    n_panels::Int=15000,
    M::Int=5,
    timeit::Bool=false,
) where {T<:Real}

    Mk = length(zj)
    @assert length(Tbufs) == Mk

    if use_chebyshev
        geomws, plans0, plans1, SCs, wsb = build_cfie_alpert_cheb_data(
            solver, pts, zj;
            npanels=n_panels, M=M, grading=:uniform, geo_ratio=1.05,
            nthreads=Threads.nthreads()
        )

        @inbounds for j in eachindex(Tbufs)
            fill!(Tbufs[j], 0.0 + 0.0im)
        end

        @benchit timeit=timeit "CFIE_alpert Chebyshev" begin
            compute_kernel_matrices_CFIE_alpert_chebyshev!(
                Tbufs, solver, pts, plans0, plans1, geomws, SCs, wsb;
                multithreaded=multithreaded
            )
        end
    else
        @error("Direct matrix construction is only for real k currently")
    end

    return nothing
end

###########################################################
#### OPTIONAL ALLOCATING WRAPPERS                      ####
###########################################################

function compute_kernel_matrices_CFIE_alpert_chebyshev(
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    plans0::Vector{ChebHankelPlanH},
    plans1::Vector{ChebHankelPlanH},
    geomws::CFIEAlpertGeomWorkspace{T},
    SCs,
    wsb::CFIEAlpertMultiBesselWorkspace;
    multithreaded::Bool=true,
) where {T<:Real}
    Mk = length(plans0)
    As = [Matrix{ComplexF64}(undef, geomws.Ntot, geomws.Ntot) for _ in 1:Mk]
    compute_kernel_matrices_CFIE_alpert_chebyshev!(
        As, solver, pts, plans0, plans1, geomws, SCs, wsb;
        multithreaded=multithreaded
    )
    return As
end

function compute_kernel_matrix_CFIE_alpert_chebyshev(
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    plan0::ChebHankelPlanH,
    plan1::ChebHankelPlanH,
    geomws::CFIEAlpertGeomWorkspace{T},
    SCs,
    wsb::CFIEAlpertMultiBesselWorkspace;
    multithreaded::Bool=true,
) where {T<:Real}
    A = Matrix{ComplexF64}(undef, geomws.Ntot, geomws.Ntot)
    compute_kernel_matrices_CFIE_alpert_chebyshev!(
        A, solver, pts, plan0, plan1, geomws, SCs, wsb;
        multithreaded=multithreaded
    )
    return A
end

function construct_boundary_matrix(
    solver::CFIE_alpert{T},
    pts::Vector{BoundaryPointsCFIE{T}},
    k::Complex{T};
    multithreaded::Bool=true,
    use_chebyshev::Bool=true,
    n_panels::Int=15000,
    M::Int=5,
) where {T<:Real}

    geomws, plan0, plan1, SCs, wsb = build_cfie_alpert_cheb_data(
        solver, pts, k;
        npanels=n_panels, M=M, grading=:uniform, geo_ratio=1.05
    )

    A = Matrix{ComplexF64}(undef, geomws.Ntot, geomws.Ntot)

    if use_chebyshev
        compute_kernel_matrices_CFIE_alpert_chebyshev!(
            A, solver, pts, plan0, plan1, geomws, SCs, wsb;
            multithreaded=multithreaded
        )
    else
        error("Direct non-Chebyshev complex-k CFIE_alpert route not implemented here.")
    end

    return A
end