
#this takes care of singular points
function regularize!(u)
    idx = findall(isnan, u)
    for i in idx
        if i != 1
            u[i] = (u[i+1] + u[i-1])/2.0
        else
            u[i] = (u[i+1] + u[end])/2.0
        end
    end
end

function boundary_function(state::S; b=5.0, multithreaded = true) where {S<:AbsState}
    let vec = state.vec, k = state.k, k_basis = state.k_basis, new_basis = state.basis, billiard=state.billiard
        type = eltype(vec)
        boundary = get_boundary_curves_with_ignored(billiard)
        crv_lengths = [crv.length for crv in boundary]
        sampler = FourierNodes([2,3,5],crv_lengths) 
        L = CompositeCurve(boundary).length
        N = max(round(Int, k*L*b/(2*pi)), 512)
        pts = boundary_coords(billiard, sampler, N)
        dX, dY = gradient_matrices(new_basis, k_basis, pts.xy; multithreaded)
        nx = getindex.(pts.normal,1)
        ny = getindex.(pts.normal,2)
        dX = nx .* dX 
        dY = ny .* dY
        U::Array{type,2} = dX .+ dY
        u::Vector{type} = U * vec
        regularize!(u)
        #compute the boundary norm
        w = dot.(pts.normal, pts.xy) .* pts.ds
        integrand = abs2.(u) .* w
        norm = sum(integrand)/(2*k^2)
        #println(norm)
        return u, pts.s::Vector{type}, norm
    end
end
#=
function boundary_function(state::S;b=5.0) where {S<:AbsState}
    vec=state.vec
    k=state.k
    k_basis=state.k_basis
    new_basis=state.basis
    billiard=state.billiard
    T=eltype(vec)
    crv_lengths = [crv.length for crv in boundary]
    sampler = FourierNodes([2,3,5],crv_lengths) 
    L = CompositeCurve(boundary).length
    N = max(round(Int, k*L*b/(2*pi)), 512)
    pts = boundary_coords(billiard, sampler, N)
    @blas_1 dX,dY=gradient_matrices(new_basis,k_basis,pts.xy) # ∂xϕ, ∂yϕ evaluated on pts.xy 
    M=size(dX,1)
    tX=Vector{Complex{T}}(undef,M) # tX = (∂xϕ)(x_i)
    tY=Vector{Complex{T}}(undef,M) # tY = (∂yϕ)(x_i)
    u=Vector{Complex{T}}(undef,M) # u  = ∂nϕ(x_i)
    # 2 GEMVs into empty then fuse normal-combination: ∂_n ϕ = nx ∂_x ϕ + ny ∂_y ϕ
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(tX,dX,vec) # tX = dX*vec
        mul!(tY,dY,vec) # tY = dY*vec
    end
    @fastmath @inbounds @simd for i in 1:M # fuse u = nx.*tX .+ ny.*tY in one loop
        n=pts.normal[i]
        u[i]=muladd(n[2],tY[i],n[1]*tX[i]) # u = n_x tX + n_y tY via muladd
    end
    regularize!(u)
    pts = apply_symmetries_to_boundary_points(pts,new_basis.symmetries,billiard)
    u = apply_symmetries_to_boundary_function(u,new_basis.symmetries)
    acc=zero(T)
    @inbounds @simd for i in eachindex(u) # boundary norm: ∫ |u|^2 (n·x) ds / (2k^2) no temps
        n=pts.normal[i]
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i] # w_i = (n·x) ds
        acc+=w*abs2(u[i]) # accumulate w_i*|u_i|^2
    end
    norm=acc/(2*k^2)
    @blas_1 return u,pts.s::Vector{T},norm
end
=#

function boundary_function(state_bundle::S; b=5.0, multithreaded = true) where {S<:EigenstateBundle}
    let X = state_bundle.X, k_basis = state_bundle.k_basis, ks = state_bundle.ks, new_basis = state_bundle.basis, billiard=state_bundle.billiard 
        type = eltype(X)
        boundary = get_boundary_curves_with_ignored(billiard)
        crv_lengths = [crv.length for crv in boundary]
        sampler = FourierNodes([2,3,5],crv_lengths) 
        L = CompositeCurve(boundary).length
        N = max(round(Int, k_basis*L*b/(2*pi)), 512)
        pts = boundary_coords(billiard, sampler, N)
        dX, dY = gradient_matrices(new_basis, k_basis, pts.xy; multithreaded)
        nx = getindex.(pts.normal,1)
        ny = getindex.(pts.normal,2)
        dX = nx .* dX 
        dY = ny .* dY
        U::Array{type,2} = dX .+ dY
        u_bundle::Matrix{type} = U * X
        for u in eachcol(u_bundle)
            regularize!(u)
        end
        #compute the boundary norm
        w = dot.(pts.normal, pts.xy) .* pts.ds
        norms = [sum(abs2.(u_bundle[:,i]) .* w)/(2*ks[i]^2) for i in eachindex(ks)]
        #println(norm)
        us::Vector{Vector{type}} = [u for u in eachcol(u_bundle)]
        return us, pts.s, norms
    end
end

function momentum_function(u,s)
    fu = rfft(u)
    sr = 1.0/diff(s)[1]
    ks = rfftfreq(length(s),sr).*(2*pi)
    return abs2.(fu)/length(fu), ks
end

function momentum_function(state::S; b=5.0, multithreaded = true) where {S<:AbsState}
    u, s, norm = boundary_function(state; b, multithreaded)
    return momentum_function(u,s)
end

#this can be optimized by usinf FFTW plans
function momentum_function(state_bundle::S; b=5.0, multithreaded = true) where {S<:EigenstateBundle}
    us, s, norms = boundary_function(state_bundle; b, multithreaded)
    mf, ks = momentum_function(us[1],s)
    type = eltype(mf)
    mfs::Vector{Vector{type}} = [mf]
    for i in 2:length(us)
        mf, ks = momentum_function(us[i],s)
        push!(mfs,mf)
    end
    return mfs, ks
end