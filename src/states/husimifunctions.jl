"""
    antisym_vec(x::AbstractVector) → Vector

Construct an antisymmetric vector by reversing `x[2:end]`, negating the
reversed values and prepending them to `x`.

## Arguments
* `x::AbstractVector`: Starting nonnegative half of the final vector.

## Returns
* `v::Vector`: Antisymmetric vector constructed from `x`.
"""
function antisym_vec(x::AbstractVector)
    v=reverse(-x[2:end])
    return append!(v,x)
end

"""
    _husimi_symmetric_window(s::AbstractVector{T},ds::AbstractVector{T},width::Real) where {T<:Real} → Tuple{Vector{T},Vector{T},Int}

Extract the boundary nodes satisfying `s<=width` and construct the corresponding
symmetric relative-coordinate window together with its reflected quadrature
weights.

The first node is not duplicated under reflection, matching [`antisym_vec`](@ref).

## Arguments
* `s::AbstractVector{T}`: Boundary arclength coordinates.
* `ds::AbstractVector{T}`: Corresponding boundary quadrature weights.
* `width::Real`: Positive arclength-window width.

## Returns
* `x::Vector{T}`: Symmetric relative-arclength window.
* `dx::Vector{T}`: Quadrature weights corresponding to `x`.
* `idx::Int`: Number of nodes in the original nonnegative half-window.
"""
function _husimi_symmetric_window(s::AbstractVector{T},ds::AbstractVector{T},width::Real) where {T<:Real}
    mask=s.<=width
    x=collect(s[mask])
    dx=collect(ds[mask])
    idx=length(x)
    x=antisym_vec(x)
    dx=vcat(reverse(dx[2:end]),dx)
    return x,dx,idx
end

"""
    husimi_function(k::T,s::AbstractVector{T},ds::AbstractVector{T},u::AbstractVector{Num},L::T;c::Real=10.0,w::Real=7.0,full_p::Bool=false) where {T<:Real,Num<:Number} → Tuple{Matrix{T},Vector{T},Vector{T}}

Compute the Poincaré-Husimi function on an automatically generated phase-space
grid.

This is the fast sliding-window implementation of the Husimi construction. The
coherent-state width is `1/sqrt(k)`. The parameter `c` determines the grid
density relative to this width, while `w` determines the Gaussian truncation
window.

The physical positions of the boundary nodes are given by `s`, while `ds`
contains the corresponding quadrature weights for integration with respect to
boundary arclength.

## Arguments
* `k::T`: Wavenumber.
* `s::AbstractVector{T}`: Physical boundary arclength coordinates.
* `ds::AbstractVector{T}`: Boundary quadrature weights for integration with respect to arclength.
* `u::AbstractVector{Num}`: Boundary-function values.
* `L::T`: Total boundary length.

## Keyword Arguments
* `c::Real=10.0`: Density of points in the coherent-state peak.
* `w::Real=7.0`: Gaussian truncation width in units of `σ`.
* `full_p::Bool=false`: Whether to explicitly evaluate the full signed momentum interval.

## Returns
* `H::Matrix{T}`: Husimi-function matrix.
* `qs::Vector{T}`: Boundary-position coordinates.
* `ps::Vector{T}`: Signed momentum coordinates.
"""
function husimi_function(k::T,s::AbstractVector{T},ds::AbstractVector{T},u::AbstractVector{Num},L::T;c::Real=10.0,w::Real=7.0,full_p::Bool=false) where {T<:Real,Num<:Number}
    #c density of points in coherent state peak, w width in units of sigma
    #L is the boundary length for periodization
    #compute coherrent state weights
    N=length(s)
    length(ds)==N==length(u)||throw(DimensionMismatch("s, ds and u must have equal length"))
    sig=one(k)/sqrt(k) # width of the Gaussian
    x,dx,idx=_husimi_symmetric_window(s,ds,w*sig)
    a=one(k)/(2*pi*sqrt(pi*k)) # normalization factor (Husimi not normalized to 1)
    uc=CircularVector(u) # allows circular indexing
    gauss=@. exp(-k/2*x^2)*dx
    gauss_l=@. exp(-k/2*(x+L)^2)*dx
    gauss_r=@. exp(-k/2*(x-L)^2)*dx
    if full_p
        ps=collect(range(-one(T),one(T),step=sig/T(c))) # evaluation points in p coordinate if p -> -p cannot be guaranteed
    else
        ps=collect(range(zero(T),one(T),step=sig/T(c))) # evaluation points in p coordinate if p -> -p no >1d irreps
    end
    q_stride=length(s[s.<=sig/T(c)])==0 ? 1 : length(s[s.<=sig/T(c)])
    q_idx=collect(1:q_stride:N)
    if isempty(q_idx) || last(q_idx)!=N
        push!(q_idx,N) # add last point carefully
    end
    qs=s[q_idx] # evaluation points in q coordinate
    H=zeros(T,length(qs),length(ps))
    @fastmath for i in eachindex(ps)
        cs=@. exp(-im*ps[i]*k*x)*gauss + exp(-im*ps[i]*k*(x+L))*gauss_l + exp(-im*ps[i]*k*(x-L))*gauss_r # exp(-im) is the convention since we take the complex conjugate of the wavepacket in the construction of PH functions
        for j in eachindex(q_idx) # innermost loop cant have @simd due to sum
            u_w=uc[q_idx[j]-idx+1:q_idx[j]+idx-1] # window with relevant values of u
            h=sum(cs.*u_w)
            H[j,i]=a*abs2(h)
        end
    end
    if !full_p
        ps=antisym_vec(ps) # make [-1,1] grid
        H_ref=reverse(H[:,2:end];dims=2)   # reflect columns dropping the p=0 duplicate
        H=hcat(H_ref,H)
    end
    return H,qs,ps
end

"""
    husimi_function(k::T,s::AbstractVector{T},ds::AbstractVector{T},u::AbstractVector{Num},L::T,q::T,p::T;w::Real=4.0) where {T<:Real,Num<:Number} → T

Evaluate the Poincaré-Husimi function at a single phase-space point `(q,p)`.

The boundary integral is evaluated using the physical arclength coordinates
`s` and the supplied quadrature weights `ds`. Periodicity is handled by
extending the boundary data by one period in each direction.

## Arguments
* `k::T`: Wavenumber.
* `s::AbstractVector{T}`: Physical boundary arclength coordinates.
* `ds::AbstractVector{T}`: Boundary quadrature weights.
* `u::AbstractVector{Num}`: Boundary-function values.
* `L::T`: Total boundary length.
* `q::T`: Boundary-position coordinate.
* `p::T`: Momentum coordinate.

## Keyword Arguments
* `w::Real=4.0`: Gaussian truncation width in units of `σ`.

## Returns
* `H::T`: Husimi-function value at `(q,p)`.
"""
function husimi_function(k::T,s::AbstractVector{T},ds::AbstractVector{T},u::AbstractVector{Num},L::T,q::T,p::T;w::Real=4.0) where {T<:Real,Num<:Number}
    # original algorithm by Benjamin Batistić in python (https://github.com/clozej/quantum_billiards/blob/crt_public/src/CoreModules/HusimiFunctionsOld.py)
    length(s)==length(ds)==length(u)||throw(DimensionMismatch("s, ds and u must have equal length"))
    width=T(w)/sqrt(k)
    s_ext=vcat(s.-L,s,s.+L)
    u_ext=vcat(u,u,u)
    ds_ext=vcat(ds,ds,ds)
    q_ext=q+L
    lo=searchsortedfirst(s_ext,q_ext-width)
    hi=searchsortedlast(s_ext,q_ext+width)
    nf=sqrt(sqrt(k/π))
    sracc=zero(T)
    siacc=zero(T)
    @inbounds for j in lo:hi
        si=s_ext[j]-q_ext
        wt=nf*exp(-0.5*k*si*si)*ds_ext[j]
        θ=k*p*si
        s_,c_=sincos(θ)
        uj=u_ext[j]
        a=wt*real(uj)
        b=wt*imag(uj)
        sracc+=a*c_+b*s_
        siacc+=b*c_-a*s_
    end
    return (sracc*sracc+siacc*siacc)/(2*π*k) # not the actual normalization
end

"""
    husimi_function(k::T,s::AbstractVector{T},ds::AbstractVector{T},u::AbstractVector{Num},L::T,qs::AbstractVector{T},ps::AbstractVector{T};w::Real=4.0,full_p::Bool=false) where {T<:Real,Num<:Number} → Tuple{Matrix{T},AbstractVector{T},Vector{T}}

Evaluate the normalized Poincaré-Husimi function on prescribed boundary-position
and momentum grids.

The physical locations of the boundary nodes are supplied by `s`, while `ds`
contains the corresponding quadrature weights. If `full_p=false`, only the
nonnegative momentum grid is explicitly evaluated and the negative half is
reconstructed by reflection.

## Arguments
* `k::T`: Wavenumber.
* `s::AbstractVector{T}`: Physical boundary arclength coordinates.
* `ds::AbstractVector{T}`: Boundary quadrature weights.
* `u::AbstractVector{Num}`: Boundary-function values.
* `L::T`: Total boundary length.
* `qs::AbstractVector{T}`: Boundary-position grid.
* `ps::AbstractVector{T}`: Momentum grid.

## Keyword Arguments
* `w::Real=4.0`: Gaussian truncation width in units of `σ`.
* `full_p::Bool=false`: Whether `ps` already spans the full signed momentum interval.

## Returns
* `H::Matrix{T}`: Normalized Husimi-function matrix.
* `qs::AbstractVector{T}`: Boundary-position grid.
* `ps_out::Vector{T}`: Full signed momentum grid.
"""
function husimi_function(k::T,s::AbstractVector{T},ds::AbstractVector{T},u::AbstractVector{Num},L::T,qs::AbstractVector{T},ps::AbstractVector{T};w::Real=4.0,full_p::Bool=false) where {T<:Real,Num<:Number}
    length(s)==length(ds)==length(u)||throw(DimensionMismatch("s, ds and u must have equal length"))
    s_ext=vcat(s.-L,s,s.+L) # concatenate shifted copies so that any window centered at q can be sliced as a contiguous subarray without computing indices modulo L -> no CircularVector needed
    u_ext=vcat(u,u,u) # same for u
    ds_ext=vcat(ds,ds,ds) # same for ds
    nx=length(qs) # number of q grid points
    ny=length(ps) # number of p grid points
    Hp=zeros(T,ny,nx) # preallocate Husimi matrix (for p x q)
    nf=sqrt(sqrt(k/pi)) # normalization factor
    width=T(w)/sqrt(k) # Gaussian width (±wσ)
    # temporary vectors that will grow to the current window size and be reused for each q
    c_re=Vector{T}(undef,0) # buffer for real parts of coefficients
    c_im=Vector{T}(undef,0) # buffer for imaginary parts of coefficients
    si=Vector{T}(undef,0) # buffer for s differences (shifted arclengths (s−q))
    @inbounds for iq in 1:nx
        q=qs[iq]+L # Add +L so that the center q sits in the middle copy of s_ext for slicing
        lo=searchsortedfirst(s_ext,q-width) # find left index of window
        hi=searchsortedlast(s_ext,q+width) # find right index of window
        W=max(0,hi-lo+1) # size of the window indexwise
        if length(c_re)<W # binary-search the indices in s_ext that fall within [q−width, q+width] and resize buffers if needed (since they are reused for each iq)
            resize!(c_re,W)
            resize!(c_im,W)
            resize!(si,W)
        end
        @inbounds for t=0:W-1 # for each point in the window calculate weights and shifted arclengths
            j=lo+t # the index in the window (shifted)
            sdiff=s_ext[j]-q # shifted arclength with the above index
            si[t+1]=sdiff # store shifted arclength
            wt=nf*exp(-0.5*k*sdiff*sdiff)*ds_ext[j] # gaussian weight with quadrature 
            uj=u_ext[j] # corresponding boundary function value in that window
            if uj isa Real # hack to avoid complex multiplications when u is real-valued
                c_re[t+1]=wt*uj # real part of summand
                c_im[t+1]=zero(T) 
            else
                c_re[t+1]=wt*real(uj) # real & imag part of summand separately
                c_im[t+1]=wt*imag(uj)
            end 
        end
        @inbounds for ip in 1:ny # for each p grid point compute Husimi value at (q,p) using the precomputed buffers for this iq index
            kp=k*ps[ip] # k*p
            sracc=zero(T);siacc=zero(T) # real & imag accumulators for the sum
            @inbounds for t=1:W # sum over the window
                θ=kp*si[t] 
                s_,c_=sincos(θ) # s_=sin,c_=cos, the cheap way to compute these
                # (a+ib)(cos-i*sin)=(a*c+b*s)+i(b*c-a*s) to avoid complex multiplications temporaries
                a=c_re[t];b=c_im[t]
                sracc+=a*c_+b*s_
                siacc+=b*c_-a*s_
            end
            # Final Husimi value at (q,p) is the squared modulus of the sum, scaled by 1/(2πk) (removed scaling and just normalize in the end)
            Hp[ip,iq]=(sracc*sracc+siacc*siacc)
        end
    end
    if full_p
        H=permutedims(Hp)
        ps_out=collect(ps)
    else
        H=vcat(reverse(Hp;dims=1),Hp[2:end,:])|>permutedims
        ps_out=vcat(-reverse(ps)[1:end-1],ps)
    end
    H./=sum(H) # normalize it in the end since the 1/(2πk) normalization does not work well in practice with finite grids
    return H,qs,ps_out
end

"""
    husimi_function(k::T,pts::BoundaryPoints{T},u::AbstractVector{Num};c::Real=10.0,w::Real=7.0,full_p::Bool=false) where {T<:Real,Num<:Number} → Tuple{Matrix{T},Vector{T},Vector{T}}

Compute an automatically gridded Poincaré-Husimi function directly from a
[`BoundaryPoints`](@ref) discretization.

## Arguments
* `k::T`: Wavenumber.
* `pts::BoundaryPoints{T}`: Boundary discretization.
* `u::AbstractVector{Num}`: Boundary-function values.

## Keyword Arguments
* `c::Real=10.0`: Density of points in the coherent-state peak.
* `w::Real=7.0`: Gaussian truncation width in units of `σ`.
* `full_p::Bool=false`: Whether to explicitly evaluate the full signed momentum interval.

## Returns
The tuple `(H,qs,ps)` returned by the low-level [`husimi_function`](@ref).
"""
function husimi_function(k::T,pts::BoundaryPoints{T},u::AbstractVector{Num};c::Real=10.0,w::Real=7.0,full_p::Bool=false) where {T<:Real,Num<:Number}
    L=sum(pts.ds)
    return husimi_function(k,pts.s,pts.ds,u,L;c=c,w=w,full_p=full_p)
end

"""
    husimi_function(k::T,pts::BoundaryPoints{T},u::AbstractVector{Num},q::T,p::T;w::Real=4.0) where {T<:Real,Num<:Number} → T

Evaluate the Poincaré-Husimi function at one phase-space point directly from a
[`BoundaryPoints`](@ref) discretization.

## Arguments
* `k::T`: Wavenumber.
* `pts::BoundaryPoints{T}`: Boundary discretization.
* `u::AbstractVector{Num}`: Boundary-function values.
* `q::T`: Boundary-position coordinate.
* `p::T`: Momentum coordinate.

## Keyword Arguments
* `w::Real=4.0`: Gaussian truncation width in units of `σ`.

## Returns
* `H::T`: Husimi-function value at `(q,p)`.
"""
function husimi_function(k::T,pts::BoundaryPoints{T},u::AbstractVector{Num},q::T,p::T;w::Real=4.0) where {T<:Real,Num<:Number}
    L=sum(pts.ds)
    return husimi_function(k,pts.s,pts.ds,u,L,q,p;w=w)
end

"""
    husimi_function(k::T,pts::BoundaryPoints{T},u::AbstractVector{Num},qs::AbstractVector{T},ps::AbstractVector{T};w::Real=4.0,full_p::Bool=false) where {T<:Real,Num<:Number} → Tuple{Matrix{T},AbstractVector{T},Vector{T}}

Evaluate the Poincaré-Husimi function on prescribed grids directly from a
[`BoundaryPoints`](@ref) discretization.

## Arguments
* `k::T`: Wavenumber.
* `pts::BoundaryPoints{T}`: Boundary discretization.
* `u::AbstractVector{Num}`: Boundary-function values.
* `qs::AbstractVector{T}`: Boundary-position grid.
* `ps::AbstractVector{T}`: Momentum grid.

## Keyword Arguments
* `w::Real=4.0`: Gaussian truncation width in units of `σ`.
* `full_p::Bool=false`: Whether `ps` already spans the full signed momentum interval.

## Returns
The tuple `(H,qs,ps_out)` returned by the low-level [`husimi_function`](@ref).
"""
function husimi_function(k::T,pts::BoundaryPoints{T},u::AbstractVector{Num},qs::AbstractVector{T},ps::AbstractVector{T};w::Real=4.0,full_p::Bool=false) where {T<:Real,Num<:Number}
    L=sum(pts.ds)
    return husimi_function(k,pts.s,pts.ds,u,L,qs,ps;w=w,full_p=full_p)
end

"""
    husimi_function(ks::AbstractVector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_pts::AbstractVector{<:BoundaryPoints{T}};c::Real=10.0,w::Real=7.0,full_p::Bool=false) where {T<:Real} → Tuple

Construct automatically gridded Poincaré-Husimi functions for a collection of
states.

Each state uses the physical arclength coordinates and boundary quadrature
weights stored in its corresponding [`BoundaryPoints`](@ref) object.

## Arguments
* `ks::AbstractVector{T}`: Wavenumbers.
* `vec_us::AbstractVector{<:AbstractVector{<:Number}}`: Boundary-function values, one vector per state.
* `vec_pts::AbstractVector{<:BoundaryPoints{T}}`: Boundary discretizations, one per state.

## Keyword Arguments
* `c::Real=10.0`: Density of points in the coherent-state peak.
* `w::Real=7.0`: Gaussian truncation width in units of `σ`.
* `full_p::Bool=false`: Whether to explicitly evaluate the full signed momentum interval.

## Returns
* `Hs_return::Vector{Matrix}`: Husimi-function matrices.
* `ps_return::Vector{Vector}`: Momentum grids.
* `qs_return::Vector{Vector}`: Boundary-position grids.
"""
function husimi_function(ks::AbstractVector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_pts::AbstractVector{<:BoundaryPoints{T}};c::Real=10.0,w::Real=7.0,full_p::Bool=false) where {T<:Real}
    length(ks)==length(vec_us)==length(vec_pts)||throw(DimensionMismatch("ks, vec_us and vec_pts must have equal length"))
    valid_indices=fill(true,length(ks))
    Hs_return=Vector{Matrix}(undef,length(ks))
    ps_return=Vector{Vector}(undef,length(ks))
    qs_return=Vector{Vector}(undef,length(ks))
    p=Progress(length(ks);desc="Constructing husimi matrices, N=$(length(ks))")
    Threads.@threads for i in eachindex(ks)
        try
            pts=vec_pts[i]
            L=sum(pts.ds)
            H,qs,ps=husimi_function(ks[i],pts.s,pts.ds,vec_us[i],L;c=c,w=w,full_p=full_p)
            Hs_return[i]=H
            ps_return[i]=ps
            qs_return[i]=qs
        catch e
            println("Error while constructing Husimi for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(p)
    end
    Hs_return=Hs_return[valid_indices]
    ps_return=ps_return[valid_indices]
    qs_return=qs_return[valid_indices]
    return Hs_return,ps_return,qs_return
end

"""
    husimi_function(ks::AbstractVector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_pts::AbstractVector{<:BoundaryPoints{T}},nx::Integer,ny::Integer;w::Real=4.0,full_p::Bool=false) where {T<:Real} → Tuple

Construct fixed-grid Poincaré-Husimi functions for a collection of states.

A common phase-space grid is used for all states. The physical boundary
positions are obtained from `BoundaryPoints.s` and the integration weights from
`BoundaryPoints.ds`.

## Arguments
* `ks::AbstractVector{T}`: Wavenumbers.
* `vec_us::AbstractVector{<:AbstractVector{<:Number}}`: Boundary-function values, one vector per state.
* `vec_pts::AbstractVector{<:BoundaryPoints{T}}`: Boundary discretizations, one per state.
* `nx::Integer`: Number of boundary-position grid points.
* `ny::Integer`: Number of points on the final signed momentum grid.

## Keyword Arguments
* `w::Real=4.0`: Gaussian truncation width in units of `σ`.
* `full_p::Bool=false`: Whether to explicitly evaluate the full signed momentum interval.

## Returns
* `Hs::Vector{Matrix{T}}`: Husimi-function matrices.
* `ps::Vector{Vector{T}}`: Momentum grids.
* `qs::Vector{Vector{T}}`: Boundary-position grids.
"""
function husimi_function(ks::AbstractVector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_pts::AbstractVector{<:BoundaryPoints{T}},nx::Integer,ny::Integer;w::Real=4.0,full_p::Bool=false) where {T<:Real}
    length(ks)==length(vec_us)==length(vec_pts)||throw(DimensionMismatch("ks, vec_us and vec_pts must have equal length"))
    isempty(ks)&&return Matrix{T}[],Vector{Vector{T}}(),Vector{Vector{T}}()
    imax=argmax(ks)
    L=sum(vec_pts[imax].ds)
    qs=range(zero(T),stop=L,length=nx)
    ps=full_p ? range(-one(T),one(T),length=ny) : range(zero(T),one(T),length=cld(ny,2))
    Hs=Vector{Matrix{T}}(undef,length(ks))
    ok=trues(length(ks))
    pbar=Progress(length(ks);desc="Husimi N=$(length(ks))")
    Threads.@threads for i in eachindex(ks)
        try
            pts=vec_pts[i]
            H,_,_=husimi_function(ks[i],pts.s,pts.ds,vec_us[i],L,qs,ps;w=w,full_p=full_p)
            Hs[i]=H
        catch e
            @debug "Husimi fail at k=$(ks[i])" exception=(e,catch_backtrace())
            ok[i]=false
        end
        next!(pbar)
    end
    ps_out=collect(full_p ? ps : vcat(-reverse(ps)[1:end-1],ps))
    qs_out=collect(qs)
    n=count(ok)
    return Hs[ok],[copy(ps_out) for _ in 1:n],[copy(qs_out) for _ in 1:n]
end