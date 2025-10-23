struct BoundaryPoints{T}
    xy::Vector{SVector{2,T}}         # boundary nodes
    normal::Vector{SVector{2,T}}     # outward normals
    kappa::Vector{T}                 # curvature (empty if not requested)
    ds::Vector{T}                    # arclength weights
    rdotn::Vector{T}                 # dot(x,n)
    w_vs::Vector{T}                  # ds / (x⋅n)
    w_dm::Vector{T}                  # (ds*(x⋅n)) / (2k^2)
    xy_int::Vector{SVector{2,T}}     # interior points (for PSM), else empty
end

function BoundaryPoints(;xy::Vector{SVector{2,T}}=[],normal::Vector{SVector{2,T}}=[],kappa::Vector{T} =[],ds::Vector{T} =[],rdotn::Vector{T} =[],w_vs::Vector{T} =[],w_dm::Vector{T} =[],xy_int::Vector{SVector{2,T}}=[]) where T <: Real
    return BoundaryPoints{T}(xy,normal,kappa,ds,rdotn,w_vs,w_dm,xy_int)
end

function _determine_bp_sizes(curves, bs, k)
    Ns = Vector{Int64}(undef,length(curves)) # store the data to indexwise access. This needs to be this way b/c we dont know beforehand which curves are real and which are abstract. Use sizehint! to give an idea as to not need to resize b/c it could the that real and abstract curves and intermingled
    @inbounds for i in eachindex(curves) # make an initial size calculation of the resulting vectors
        crv=curves[i]
        Ns[i] =max(20,round(Int,k*crv.length*bs[i]/2*pi))
    end
    return Ns
end