struct BoundaryPoints{T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    kappa::Vector{T}
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
                        ds=T[], 
                        rdotn=T[], 
                        w_vs=T[], 
                        w_dm=T[], 
                        xy_int=SVector{2,T}[]) where T<:Real
    return BoundaryPoints{T}(xy, normal, kappa, ds, rdotn, w_vs, w_dm, xy_int)
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