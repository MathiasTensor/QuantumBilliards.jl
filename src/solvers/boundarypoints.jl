struct BoundaryPoints{T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    kappa::Vector{T}
    s::Vector{T}
    ds::Vector{T}
    rdotn::Vector{T}
    w_vs::Vector{T}
    w_dm::Vector{T}
    xy_int::Vector{SVector{2,T}}
    
    # Inner constructor with validation
    function BoundaryPoints{T}(xy, normal, kappa, ds, rdotn, w_vs, w_dm, xy_int) where T<:Real
        n = length(xy)
        # Validate that non-empty vectors have consistent lengths
        for (name, vec) in [(:normal, normal), (:ds, ds), (:rdotn, rdotn), 
                             (:w_vs, w_vs), (:w_dm, w_dm)]
            if !isempty(vec) && length(vec) != n
                error("Length of $name ($(length(vec))) must match xy ($n)")
            end
        end
        new{T}(xy, normal, kappa, ds, rdotn, w_vs, w_dm, xy_int)
    end
end

# 2. Convenience constructor to infer T from xy
function BoundaryPoints(xy::Vector{SVector{2,T}}; 
                        normal=SVector{2,T}[], 
                        kappa=T[],
                        s=T[], 
                        ds=T[], 
                        rdotn=T[], 
                        w_vs=T[], 
                        w_dm=T[], 
                        xy_int=SVector{2,T}[]) where T<:Real
    return BoundaryPoints{T}(xy, normal, kappa, s, ds, rdotn, w_vs, w_dm, xy_int)
end

# 3. Add useful methods
Base.length(bp::BoundaryPoints) = length(bp.xy)
Base.isempty(bp::BoundaryPoints) = isempty(bp.xy)

function _determine_bp_sizes(curves, bs, k)
    Ns = Vector{Int64}(undef,length(curves)) # store the data to indexwise access. This needs to be this way b/c we dont know beforehand which curves are real and which are abstract. Use sizehint! to give an idea as to not need to resize b/c it could the that real and abstract curves and intermingled
    @inbounds for i in eachindex(curves) # make an initial size calculation of the resulting vectors
        crv=curves[i]
        Ns[i] =max(20,round(Int,k*crv.length*bs[i]/2*pi))
    end
    return Ns
end


function boundary_coords(billiard::Bi, samplers, Ns) where {Bi<:AbsBilliard}
    curves = get_boundary_curves_with_ignored(billiard)
    T = typeof(curves[1].length)
    M = length(Ns)
    xy_all = Vector{Vector{SVector{2,T}}}(undef, M)
    normal_all = Vector{Vector{SVector{2,T}}}(undef, M)
    s_all = Vector{Vector{T}}(undef, M)
    ds_all = Vector{Vector{T}}(undef, M)
    w_n_all = Vector{Vector{T}}(undef, M)
    L0 = zero(T)
    for i in eachindex(curves)
        crv = curves[i]
        L = crv.length
        sampler = samplers[i]
        t, dt = sample_points(sampler, Ns[i])
        ds = L*dt #this needs modification!!!
        xy = curve(crv,t)
        normal = domain_gradient_vector(crv, xy)
        normal .= normal./norm(normal)
        rn = dot.(xy, normal)
        xy_all[i] = xy
        normal_all[i] = normal
        s_all[i] = cumsum(ds) + L0 #arc_lengt(crv, xy)
        ds_all[i] = ds  
        w_n_all[i] = (ds.*rn)./(2.0*k.^2)
        L0 += L        
    end

    return BoundaryPoints(vcat(xy_all...);normal = vcat(normal_all...),  w_dm = vcat(w_n_all...), ds = vcat(ds_all...))
end


function get_boundary_curves_with_ignored(domain::D) where D<:AbsSimpleDomain
    is_outer(crv) = (typeof(crv.bc) <: SpecularReflection || typeof(crv.bc) <: QuantumSolverIgnore)
    boundary = filter(is_outer, domain.boundary)
    return connect_curves(boundary)
end

function get_boundary_curves_with_ignored(domain::D) where D<:AbsSimpleDomain
    is_outer(crv) = (typeof(crv.bc) <: SpecularReflection || typeof(crv.bc) <: QuantumSolverIgnore)
    boundary = filter(is_outer, domain.boundary)
    return connect_curves(boundary)
end

function get_boundary_curves_with_ignored(composite_domain::D) where D<:AbsCompositeDomain
    boundary = Vector{AbsCurve}()
    for domain in composite_domain.subdomains
        subboundary = get_boundary_curves(domain)
        append!(boundary,subboundary)
    end
    return connect_curves(boundary)
end

function get_boundary_curves_with_ignored(billiard::B) where B<:AbsBilliard
    return get_boundary_curves_with_ignored(billiard.fundamental_domain)
end