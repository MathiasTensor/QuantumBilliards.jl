"""
    antisym_vec(x::AbstractVector)

Creates an antisymmetric vector by reversing the input vector `x`, negating the reversed values, and appending them to `x`. Used in the original husimi_function construction.

# Arguments
- `x::Vector`: Starting half of the final vector.

# Returns
- `Vector`: An antisymmetric vector constructed from `x`.
"""
function antisym_vec(x)
    v=reverse(-x[2:end])
    return append!(v,x)
end

"""
    husimi_function(k::T,u::Vector{T},s::Vector{T},L::T;c=10.0,w=7.0) where {T<:Real}

Calculates the Husimi function on a grid defined by the boundary `s`. The logic is that from the arclengths `s` we construct the qs with the help of the width of the Gaussian at that k (width = 1/√k) and the parameter c (which determines how many q evaluations we will have per this width at k -> defines the step 1/(√k*c) so we will have tham many more q evaluations in peak). Analogously we do for the ps, where the step size (matrix size in ps direction) is range(0.0,1.0,step=1/(√k*c)) (we do only for half since symmetric with p=0 implies we can use antisym_vec(ps) to recreate the whole -1.0 to 1.0 range while also symmetrizing the Husimi function that is constructed from p=0.0 to 1.0 with the logic perscribed above). The w (how many sigmas we will take) is used in construction of the ds summation weights.
Comment: Due to the Poincare map being an involution, i.e. (ξ,p) →(ξ,−p) we construct only the p=0 to p=1 matrix and then via symmetry automatically obtain the p=-1 to p=0 also.
Comment: Original algorithm by Črt Lozej

# Arguments
- `k<:Real`: Wavenumber of the eigenstate.
- `u::AbstractVector{<:Number}`: Array of boundary function values. Can be complex
- `s::AbstractVector{<:Real}`: Array of boundary points.
- `L<:Real`: Total length of the billiard boundary.
- `c<:Real`: Density of points in the coherent state peak (default: `10.0`).
- `w<:Real`: Width in units of `σ` (default: `7.0`).
- `full_p::Bool=false`: Whether the boundary function is such that p -> -p symmetry is guaranteed. If so it halves the effort.

# Returns
- `H::Matrix{<:Real}`: Husimi function matrix.
- `qs::Vector{<:Real}`: Array of position coordinates on the grid.
- `ps::Vector{<:Real}`: Array of momentum coordinates on the grid.
"""
function husimi_function(k::T,u::AbstractVector{Num},s::AbstractVector{T},L::T;c=10.0,w=7.0,full_p::Bool=false) where {T<:Real,Num<:Number}
    #c density of points in coherent state peak, w width in units of sigma
    #L is the boundary length for periodization
    #compute coherrent state weights
    N=length(s)
    sig=one(k)/sqrt(k) # width of the Gaussian
    x=s[s.<=w*sig]
    idx=length(x) # do not change order here
    x=antisym_vec(x)
    a=one(k)/(2*pi*sqrt(pi*k)) # normalization factor (Husimi not normalized to 1)
    ds=(x[end]-x[1])/length(x) # integration weight
    uc=CircularVector(u) # allows circular indexing
    gauss=@. exp(-k/2*x^2)*ds
    gauss_l=@. exp(-k/2*(x+L)^2)*ds
    gauss_r=@. exp(-k/2*(x-L)^2)*ds
    if full_p
        ps=collect(range(-1.0,1.0,step=sig/c)) # evaluation points in p coordinate if p -> -p cannot be guaranteed
    else
        ps=collect(range(0.0,1.0,step=sig/c)) # evaluation points in p coordinate if p -> -p no >1d irreps
    end
    q_stride=length(s[s.<=sig/c])==0 ? 1 : length(s[s.<=sig/c])
    q_idx=collect(1:q_stride:N)
    if isempty(q_idx) || last(q_idx)!=N
        push!(q_idx,N) # add last point carefully
    end
    qs=s[q_idx] # evaluation points in q coordinate
    H=zeros(typeof(k),length(qs),length(ps))
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
    husimi_function(state::S; b = 5.0, c = 10.0, w = 7.0) where {S<:AbsState}

Calculates the Husimi function for a billiard eigenstate. Wrapper for lower level husimi_function.

# Arguments
- `state<:AbsState`: An eigenstate of the billiard system (contains the basis, billiard and k importantly)
- `b`: Parameter for the boundary function computation (default: `5.0`).
- `c`: Density of points in the coherent state peak (default: `10.0`).
- `w`: Width in units of `σ` (default: `7.0`).

# Returns
- `H::Matrix{<Real}`: Husimi function matrix.
- `qs::Vector{<:Real}`: Array of position coordinates on the grid.
- `ps::Vector{<:Real}`: Array of momentum coordinates on the grid.
"""
function husimi_function(state::S;b=5.0,c=10.0,w=7.0,full_p::Bool=false) where {S<:AbsState}
    L=state.billiard.length
    k=state.k
    u,s,norm=boundary_function(state;b=b)
    return husimi_function(k,u,s,L;c=c,w=w,full_p=full_p)
end

"""
    husimi_function(state_bundle::S; b = 5.0, c = 10.0, w = 7.0) where {S<:EigenstateBundle}

Calculates the Husimi function for a batch of billiard eigenstates that came from a single Scaling Method evaluation, therefore the expansion coefficients in the basis have same length. Wrapper for lower level husimi_function.

# Arguments
- `state<:EigenstateBundle`: An eigenstate of the billiard system (contains the basis, billiard and k importantly)
- `b`: Parameter for the boundary function computation (default: `5.0`).
- `c`: Density of points in the coherent state peak (default: `10.0`).
- `w`: Width in units of `σ` (default: `7.0`).
- `full_p::Bool=false`: Whether the boundary function is such that p -> -p symmetry is guaranteed. If so it halves the effort.

# Returns
- `H::Vector{Matrix{<:Real}}`: Husimi function matrices for that state bundle with same grids.
- `qs::Vector{<:Real}`: Array of position coordinates on the grid.
- `ps::Vector{<:Real}`: Array of momentum coordinates on the grid.
"""
function husimi_function(state_bundle::S;b=5.0,c=10.0,w=7.0,full_p::Bool=false) where {S<:EigenstateBundle}
    L=state_bundle.billiard.length
    ks=state_bundle.ks
    us,s,norm=boundary_function(state_bundle; b=b)
    H,qs,ps=husimi_function(ks[1],us[1],s,L;c=c,w=w,full_p=full_p)
    type=eltype(H)
    valid_indices=fill(true,length(ks))
    Hs=Vector{Matrix{type}}(undef,length(ks))
    Hs[1]=H
    for i in eachindex(ks)[2:end] # no need for multithreading here
        try
            H,qs,ps=husimi_function(ks[i],us[i],s,L;c=c,w=w,full_p=full_p)
            Hs[i]=H
        catch e
            println("Error while constructing Husimi for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
    end
    Hs=Hs[valid_indices]
    return Hs,qs,ps
end

### NEW ###

"""
    husimi_at_point(k::T,s::Vector{T},u::Vector{T},L::T,q::T,p::T) where {T<:Real}

Calculates the Poincaré-Husimi function at point (q, p) in the quantum phase space.

Arguments:
- `k::T`: Wavenumber of the eigenstate.
- `s::AbstractVector{T}`: Array of points on the boundary.
- `u::AbstractVector{<:Number}`: Array of boundary function values.
- `L::T`: Total length of the boundary (maximum(s)).
- `q::T`: Position coordinate in phase space.
- `p::T`: Momentum coordinate in phase space.

Returns:
- `T`: Husimi function value at (q, p).
"""
function husimi_at_point(k::T,s::AbstractVector{T},u::AbstractVector{Num},L::T,q::T,p::T) where {T<:Real,Num<:Number}
    # original algorithm by Benjamin Batistić in python (https://github.com/clozej/quantum_billiards/blob/crt_public/src/CoreModules/HusimiFunctionsOld.py)
    ss=s.-q
    width=4/sqrt(k)
    indx=findall(x->abs(x)<width,ss)
    si=ss[indx]
    ui=u[indx]
    ds=diff(s)  # length N-1
    ds=vcat(ds,L+s[1]-s[end]) # add Nth
    dsi=ds[indx]
    w=sqrt(sqrt(k/π)).*exp.(-0.5*k*si.*si).*dsi
    cr=w.*cos.(k*p*si)  # Coherent state real part
    ci=w.*sin.(k*p*si)  # Coherent state imaginary part
    h=dot(cr-im*ci,ui)  # Husimi integral (minus because of conjugation)
    return abs2(h)/(2*π*k) # not the actual normalization
end

"""
    husimi_on_grid(k::T, s::Vector{T}, u::Vector{T}, L::T, qs::AbstractVector{T}, ps::AbstractVector{T}; full_p::Bool=false) where {T<:Real}

Evaluates the Poincaré-Husimi function on a grid using vectorized operations in a thread-safe manner since only a single thread work on a column of the Husimi matrix.
Due to the Poincare map being an involution, i.e. (ξ,p) →(ξ,−p) we construct only the p=0 to p=1 matrix and then via symmetry automatically obtain the p=-1 to p=0 also.

Arguments:
- `k::T`: Wavenumber of the eigenstate.
- `s::AbstractVector{T}`: Array of points on the boundary.
- `u::AbstractVector{Num}`: Array of boundary function values. Can be complex.
- `L::T`: Total length of the boundary (maximum(s)).
- `qs::AbstractVector{T}`: Array of q values used in the grid.
- `ps::AbstractVector{T}`: Array of p values used in the grid.
- `full_p::Bool=false`: Whether the boundary function is such that p -> -p symmetry is guaranteed. If so it halves the effort.

Returns:
- `H::Matrix{T}`: Husimi function matrix of size (ny, nx).
- `qs::Vector{T}`: Array of q values used in the grid.
- `ps::Vector{T}`: Array of p values used in the grid.
"""
function husimi_on_grid(k::T,s::AbstractVector{T},u::AbstractVector{Num},L::T,qs::AbstractVector{T},ps::AbstractVector{T};full_p::Bool=false) where {T<:Real,Num<:Number}
    n=length(s)
    ds=similar(s)
    @inbounds for j=1:n-1 # build piecewise boundary spacings
        ds[j]=s[j+1]-s[j]
    end
    ds[n]=L+s[1]-s[end] # make sure the last segment wraps around
    s_ext=vcat(s.-L,s,s.+L) # concatenate shifted copies so that any window centered at q can be sliced as a contiguous subarray without computing indices modulo L -> no CircularVector needed
    u_ext=vcat(u,u,u) # same for u
    ds_ext=vcat(ds,ds,ds) # same for ds
    nx=length(qs) # number of q grid points
    ny=length(ps) # number of p grid points
    Hp=zeros(T,ny,nx) # preallocate Husimi matrix (for p x q)
    nf=sqrt(sqrt(k/pi)) # normalization factor
    width=4/sqrt(k) # Gaussian width (±4σ)
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
            w=nf*exp(-0.5*k*sdiff*sdiff)*ds_ext[j] # gaussian weight with quadrature 
            uj=u_ext[j] # corresponding boundary function value in that window
            if uj isa Real # hack to avoid complex multiplications when u is real-valued
                c_re[t+1]=w*uj # real part of summand
                c_im[t+1]=zero(T) 
            else
                c_re[t+1]=w*real(uj) # real & imag part of summand separately
                c_im[t+1]=w*imag(uj)
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
    function husimi_functions_from_StateData(state_data::StateData, billiard::Bi, basis::Ba;  b = 5.0, c = 10.0, w = 7.0) :: Tuple{Vector{Matrix{T}}, Vector{Vector{T}}, Vector{Vector{T}}} where {T<:Real, Bi<:AbsBilliard, Ba<:AbsBasis}

High level wrapper for the construction of the husimi functions from StateData which we compute as we run the compute_spectrum so we do not have to compute the eigenstate for each k in the eigenvalues we get from the spectrum.

# Arguments
- `state_data`: A `StateData` object containing the state data.
- `billiard`: A `Bi` object representing the billiard.
- `basis`: A `Ba` object representing the basis.
- Comment: `c` density of points in coherent state peak, `w` width in units of sigma.
- `full_p::Bool=false`: Whether the boundary function is such that p -> -p symmetry is guaranteed. If so it halves the effort.

# Returns
- `Hs_return::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps_return::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs_return::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function husimi_functions_from_StateData(state_data::StateData,billiard::Bi,basis::Ba;b=5.0,c=10.0,w=7.0,full_p::Bool=false) where {Bi<:AbsBilliard,Ba<:AbsBasis}
    L=billiard.length
    ks=state_data.ks
    valid_indices=fill(true,length(ks))
    Hs_return=Vector{Matrix}(undef,length(ks))
    ps_return=Vector{Vector}(undef,length(ks))
    qs_return=Vector{Vector}(undef,length(ks))
    ks,us,s_vals,_ = boundary_function(state_data,billiard,basis;b=b)
    p=Progress(length(ks);desc="Constructing husimi matrices, N=$(length(ks))")
    Threads.@threads for i in eachindex(ks) 
        try
            H,qs,ps=husimi_function(ks[i],us[i],s_vals[i],L;c=c,w=w,full_p=full_p)
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
    husimi_functions_from_boundary_functions(ks::Vector, us::Vector{Vector}, s_vals::Vector{Vector}, billiard::Bi; c = 10.0, w = 7.0)

An efficient way to ge the husimi functions from the stored `ks`, `us`, `s_vals` that we can save after doing the version of `compute_spectrum` with the `StateData`.

# Arguments
- `ks::Vector`: A vector of eigenvalues.
- `us::Vector{Vector}`: A vector of vectors representing the boundary functions.
- `s_vals::Vector{Vector}`: A vector of vectors representing the evaluation points in s coordinate.
- `full_p::Bool=false`: Whether the boundary function is such that p -> -p symmetry is guaranteed. If so it halves the effort.

# Returns
- `Hs_return::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps_return::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs_return::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function husimi_functions_from_boundary_functions(ks,us,s_vals,billiard::Bi;c=10.0,w=7.0,full_p::Bool=false) where {Bi<:AbsBilliard}
    L=billiard.length
    valid_indices=fill(true,length(ks))
    Hs_return=Vector{Matrix}(undef,length(ks))
    ps_return=Vector{Vector}(undef,length(ks))
    qs_return=Vector{Vector}(undef,length(ks))
    p=Progress(length(ks);desc="Constructing husimi matrices, N=$(length(ks))")
    Threads.@threads for i in eachindex(ks) 
        try
            H,qs,ps=husimi_function(ks[i],us[i],s_vals[i],L;c=c,w=w,full_p=full_p)
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
    husimi_functions_from_us_and_boundary_points(ks::Vector{T},vec_us::Vector{Vector{Num}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi) where {Bi<:AbsBilliard,T<:Real,Num<:Number}

Efficient way to construct the husimi functions (`Vector{Matrix}`) and their grids from the boundary function values along with the vector of `BoundaryPoints` whic containt the .s field which gives the the arclengths.

# Arguments
- `ks::Vector{T}`: A vector of eigenvalues.
- `vec_us::Vector{Vector{<:Number}}`: A vector of vectors representing the boundary function values. Can take complex values
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: A vector of `BoundaryPoints` objects.
- `full_p::Bool=false`: Whether the boundary function is such that p -> -p symmetry is guaranteed. If so it halves the effort.

# Returns
- `Hs::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{Vector}`: A vector of vectors representing the evaluation points in p coordinate.
- `qs::Vector{Vector}`: A vector of vectors representing the evaluation points in q coordinate.
"""
function husimi_functions_from_us_and_boundary_points(ks::Vector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi;full_p::Bool=false) where {Bi<:AbsBilliard,T<:Real}
    vec_of_s_vals=[bdPoints.s for bdPoints in vec_bdPoints]
    Hs_list,ps_list,qs_list=husimi_functions_from_boundary_functions(ks,vec_us,vec_of_s_vals,billiard,full_p=full_p)
    return Hs_list,ps_list,qs_list
end

"""
    function husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks::Vector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_bdPoints::Vector{BoundaryPoints{T}},billiard::Bi, nx::Integer,ny::Integer;full_p::Bool=false) where {Bi<:AbsBilliard,T<:Real}

Efficient way to construct the husimi functions (`Vector{Matrix}`) on a common grid of `size(nx,ny)` from the boundary function values along with the vector of `BoundaryPoints` whic containt the .s field which gives the the arclengths.

# Arguments
- `ks::Vector{T}`: A vector of eigenvalues.
- `vec_us::AbstractVector{<:AbstractVector{<:Number}}`: A vector of vectors representing the boundary function values. Cant take complex values
- `vec_bdPoints::Vector{BoundaryPoints{T}}`: A vector of `BoundaryPoints` objects.
- `billiard::Bi`: The billiard geometry for the total length. Faster than calling `maximum(s)`.
- `nx::Interger`: The size of linearly spaced q grid.
- `ny::Interger`: The size of linearly spaced p grid.
- `full_p::Bool=false`: Whether the boundary function is such that p -> -p symmetry is guaranteed. If so it halves the effort.

# Returns
- `Hs_list::Vector{Matrix}`: A vector of matrices representing the Husimi functions.
- `ps::Vector{T}`: A vector representing the evaluation points in p coordinate (same for all husimi matrices).
- `qs::Vector{T}`: A vector representing the evaluation points in q coordinate (same for all husimi matrices).
"""
function husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks::AbstractVector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}}, vec_bdPoints::AbstractVector{BoundaryPoints{T}},billiard::Bi,nx::Integer,ny::Integer;full_p::Bool=false) where {Bi<:AbsBilliard,T<:Real}
    L=billiard.length
    qs=range(zero(T),stop=L,length=nx)
    ps=full_p ? range(-one(T),one(T),length=ny) : range(zero(T),one(T),length=cld(ny,2))
    vec_s=[bd.s for bd in vec_bdPoints]
    Hs=Vector{Matrix{T}}(undef,length(ks))
    ok=trues(length(ks))
    pbar=Progress(length(ks);desc="Husimi N=$(length(ks))")
    Threads.@threads for i in eachindex(ks)
        try
            H,_,_=husimi_on_grid(ks[i],vec_s[i],vec_us[i],L,qs,ps;full_p=full_p)
            Hs[i]=H
        catch e
            @debug "Husimi fail at k=$(ks[i])" exception=(e,catch_backtrace())
            ok[i]=false
        end
        next!(pbar)
    end
    Hs=Hs[ok]
    return Hs,collect(ps),collect(qs)
end

##########################################################################
#################### MULTI BOUNDARY FUNCTION VERSIONS ####################
##########################################################################

"""

    split_boundary_data_by_component(comps::Vector{BoundaryPointsCFIE{T}},u::AbstractVector{Num}) where {T<:Real,Num<:Number}

Split one concatenated CFIE/Alpert boundary density vector `u` into one density
vector per connected boundary component, using the `compid` labels stored in
`comps`. This is the helper needed for Husimi construction on multiply connected domains.
The boundary discretization used by CFIE-style solvers is often stored as a flat
vector of panels / periodic components, while the Husimi construction should be
performed separately on each connected closed boundary component:

- component 1 = outer boundary,
- components 2:end = holes.

Each entry of `comps` contributes a consecutive block of coefficients in the
concatenated density vector `u`. Entries with the same `compid` belong to the
same connected boundary component and are therefore grouped together. For each
such group, this function constructs:
- `u_comps[a]`: the boundary density restricted to component `a`,
- `s_comps[a]`: a local arclength coordinate for component `a`, starting at 0
  and increasing continuously across all pieces belonging to that component,
- `L_comps[a]`: the total arclength of `a`.

Example: [EllipticFlowerWithOptionalHole] - has an outer boundary consisting of elliptical segments
with `compid=1` and an optional inner hole with `compid=2`. The `comps` vector for this geometry might look like:

[BoundaryPointsCFIE{T}(compid=1, ...), BoundaryPointsCFIE{T}(compid=1, ...), BoundaryPointsCFIE{T}(compid=1, ...), BoundaryPointsCFIE{T}(compid=2, ...)]

where the first entries belong to the outer boundary (compid=1) and the last entry belongs to the inner hole (compid=2).
The function will then split the concatenated density vector `u` into two parts: one for the outer boundary and one for the inner hole, 
and compute the corresponding arclength coordinates and total lengths for each component.

# Arguments
- `comps::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization pieces. These may be whole periodic components or open
  panels, depending on the solver, but pieces sharing the same `compid` are
  interpreted as belonging to one connected closed boundary component.
- `u::AbstractVector{Num}`:
  Concatenated boundary density corresponding to the flattened ordering of
  `comps`.

# Returns
- `u_comps::Vector{Vector{Num}}`:
  Boundary density split into one vector per connected boundary component.
  -> [u for component 1, u for component 2] top example
- `s_comps::Vector{Vector{T}}`:
  Local arclength coordinates for each connected boundary component, each
  starting at 0.
  -> [s for component 1, s for component 2] top example
- `L_comps::Vector{T}`:
  Total perimeter / arclength of each connected boundary component.
  -> [L for component 1, L for component 2] top example

# Notes
The output is designed to be passed directly to component-wise Husimi routines,
where each component is treated with its own period `L_comps[a]`.
"""
function split_boundary_data_by_component(comps::Vector{BoundaryPointsCFIE{T}},u::AbstractVector{Num}) where {T<:Real,Num<:Number}
    isempty(comps) && return Vector{Vector{Num}}(),Vector{Vector{T}}(),T[] # should not happen but just in case
    compids=sort(unique(getfield.(comps,:compid))) # get unique component ids in order of appearance
    # group the indices of comps by their compid so that we can slice u and s accordingly for each component
    # E.g. if we have 3 components with compids [1,1,1,2,2] then groups will be [[1,2,3],[4,5]] so that we know to slice u and s
    # for component 1 using comps[1], comps[2], and comps[3] and for component 2 using comps[4] and comps[5]
    # num of groups is num of unique compids, therefore the number of closed boundary components (outer boundary + holes)
    groups=[findall(c->c.compid==cid,comps) for cid in compids]
    u_comps=Vector{Vector{Num}}(undef,length(groups))
    s_comps=Vector{Vector{T}}(undef,length(groups))
    L_comps=Vector{T}(undef,length(groups))
    p=1 # starting with index 1 in the concatenated u and s, we will slice according to the lengths of the components as we iterate through the groups
    for gi in eachindex(groups)
        idxs=groups[gi] # get the index if the group along with the comps that belong to it
        uparts=Vector{Vector{Num}}(undef,length(idxs))
        sparts=Vector{Vector{T}}(undef,length(idxs))
        soff=zero(T) # arclength offset for this component, since we want the s values for each component to start at 0 
        # and at the total length of that component, we need to keep track of the offset as we concatenate the parts from each comp in the group
        for (m,j) in enumerate(idxs)
            # for each component in the group, we slice u and s according to the length of that component,
            # which is given by length(comps[j].xy) since comps[j].xy gives the points on the boundary for that component
            bd=comps[j] # get the component in the group
            N=length(bd.xy) # number of points in that component, which tells us how many values in u and s correspond to that component
            uj=collect(u[p:p+N-1]) # slice the corresponding part of u for that component [p:p+N-1] since p is the starting index for this component and we need N values for it
            sj=cumsum(bd.ds) # cumulative sum of the arclengths for this component
            sj.-=sj[1] # shift so that the arclength starts at 0
            sj.+=soff # add the offset to account for previous components in the group
            uparts[m]=uj
            sparts[m]=sj
            soff+=sum(bd.ds) # update the offset for the next component in the group
            p+=N # move the starting index for slicing u and s for the next component in the concatenated vector
        end
        u_comps[gi]=vcat(uparts...) # concatenate the parts for this group to get the full u for this component
        s_comps[gi]=vcat(sparts...) # concatenate the parts for this group to get the full s for this component
        L_comps[gi]=soff # the total length of this component is the final offset after concatenating all parts in the group
    end
    return u_comps,s_comps,L_comps
end

"""
    husimi_on_grid_components(k::T,s_comps::Vector{<:AbstractVector{T}},u_comps::Vector{<:AbstractVector{Num}},L_comps::AbstractVector{T},qs_list::Vector{<:AbstractVector{T}},ps::AbstractVector{T};full_p::Bool=false,normalize_components::Bool=true) where {T<:Real,Num<:Number}

Construct one Husimi matrix per connected boundary component on prescribed
component-wise `q` grids and a shared `p` grid.

This is a thin multi-component wrapper around the single-component
`husimi_on_grid` routine. It is intended for multiply connected billiards where
the boundary function has already been split into:
- one arclength vector `s_comps[a]`,
- one density vector `u_comps[a]`,
- one perimeter `L_comps[a]`,
for each connected closed boundary component `a`.

The key point is that the Husimi construction should be carried out separately
on each connected component, because each component has its own natural boundary
periodicity:
- the outer boundary is periodic with period `L_comps[1]`,
- each hole is periodic with its own period `L_comps[a]`.

Accordingly, this function returns one Husimi matrix per component rather than
trying to force all components into a single global boundary period.

If `normalize_components=true`, then each connected component Husimi is
normalized separately:
    sum(Hs[a]) = 1
for every component `a` whose Husimi mass is nonzero.

# Arguments
- `k::T`:
  Wavenumber of the state whose Husimi representation is being constructed.
- `s_comps::Vector{<:AbstractVector{T}}`:
  Component-wise arclength coordinates, each starting at 0 and running over one
  connected closed boundary component.
- `u_comps::Vector{<:AbstractVector{Num}}`:
  Component-wise boundary densities corresponding to `s_comps`.
- `L_comps::AbstractVector{T}`:
  Component-wise total boundary lengths / perimeters.
- `qs_list::Vector{<:AbstractVector{T}}`:
  One `q` grid per connected boundary component. Usually each grid spans
  `[0,L_comps[a]]`.
- `ps::AbstractVector{T}`:
  Shared nonnegative `p` grid if `full_p=false`, or full signed `p` grid if
  `full_p=true`.
- `full_p::Bool=false`:
  Passed through to `husimi_on_grid`. If `false`, the single-component routine
  exploits the `p -> -p` reflection to reconstruct the negative-`p` half.
- `normalize_components::Bool=true`:
  Whether to normalize each component Husimi separately to unit sum.

# Returns
- `Hs::Vector{Matrix{T}}`:
  `Hs[a]` is the Husimi matrix for connected component `a`.
- `qs_out::Vector{Vector{T}}`:
  Stored copies of the `q` grids actually used for each component.
- `ps_out::Vector{Vector{T}}`:
  Stored copies of the `p` grids actually used for each component. These are
  identical across components in the present implementation, but returned
  component-wise for interface symmetry and future flexibility.
"""
function husimi_on_grid_components(k::T,s_comps::Vector{<:AbstractVector{T}},u_comps::Vector{<:AbstractVector{Num}},L_comps::AbstractVector{T},qs_list::Vector{<:AbstractVector{T}},ps::AbstractVector{T};full_p::Bool=false,normalize_components::Bool=true) where {T<:Real,Num<:Number}
    ncomp=length(s_comps)
    length(u_comps)==ncomp || error("u_comps and s_comps must have same length")
    length(L_comps)==ncomp || error("L_comps and s_comps must have same length")
    length(qs_list)==ncomp || error("qs_list and s_comps must have same length")
    Hs=Vector{Matrix{T}}(undef,ncomp)
    qs_out=Vector{Vector{T}}(undef,ncomp)
    ps_out=Vector{Vector{T}}(undef,ncomp)
    for a in 1:ncomp
        length(s_comps[a])==length(u_comps[a]) || error("Component $a has inconsistent data: length(s_comps[$a])=$(length(s_comps[a])) but length(u_comps[$a])=$(length(u_comps[a])).")
        H,qs_a,ps_a=husimi_on_grid(k,s_comps[a],u_comps[a],L_comps[a],qs_list[a],ps;full_p=full_p)
        if normalize_components
            hsum=sum(H)
            hsum>zero(T) && (H./=hsum)
        end
        Hs[a]=H
        qs_out[a]=collect(qs_a)
        ps_out[a]=collect(ps_a)
    end
    return Hs,qs_out,ps_out
end

"""
    husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks::AbstractVector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_bdComps::AbstractVector{<:Vector{BoundaryPointsCFIE{T}}},nx::Integer,ny::Integer;full_p::Bool=false,normalize_components::Bool=true) where {T<:Real}

Construct fixed-grid Husimi functions for a batch of states whose
boundary data may live on multiply connected billiards, including domains with
holes.

This is the high-level multi-state, multi-component wrapper for Husimi
construction from CFIE-style boundary discretizations. For each state `i`:

1. the concatenated boundary density `vec_us[i]` is split into connected
   boundary components using `split_boundary_data_by_component`,
2. one fixed `q` grid is built for each connected component, spanning that
   component's own local arclength interval `[0,L_comps[a]]`,
3. one Husimi matrix is constructed per connected component via
   `husimi_on_grid_components`.

Thus the output for one state is not a single Husimi matrix, but a vector of
Husimi matrices:
- one for the outer boundary,
- one for each hole.

This is the natural representation for multiply connected billiards, since each
connected boundary component defines its own boundary phase-space section.

If `normalize_components=true`, then each connected component Husimi is
normalized separately to unit sum.

# Arguments
- `ks::AbstractVector{T}`:
  Wavenumbers / eigenvalues of the states.
- `vec_us::AbstractVector{<:AbstractVector{<:Number}}`:
  Boundary densities, one concatenated density vector per state.
- `vec_bdComps::AbstractVector{<:Vector{BoundaryPointsCFIE{T}}}`:
  Boundary discretizations, one vector of `BoundaryPointsCFIE` pieces per state.
  Within each state, pieces sharing the same `compid` are interpreted as
  belonging to the same connected boundary component.
- `nx::Integer`:
  Number of `q`-grid points per connected boundary component.
- `ny::Integer`:
  Number of `p`-grid points for the full signed grid if `full_p=true`, or for
  the reconstructed signed grid if `full_p=false`.
- `full_p::Bool=false`:
  Passed through to the underlying Husimi routines. If `false`, only the
  nonnegative `p` half-grid is explicitly computed and the negative half is
  reconstructed by symmetry.
- `normalize_components::Bool=true`:
  Whether to normalize each component Husimi separately to unit mass.

# Returns
- `Hs_all::Vector{Vector{Matrix{T}}}`:
  `Hs_all[i][a]` is the Husimi matrix for state `i` and connected boundary
  component `a`.
- `ps_out::Vector{T}`:
  Common signed `p` grid returned in the user-facing format. If `full_p=false`,
  this is the symmetrized grid reconstructed from the nonnegative half-grid.
- `qs_all::Vector{Vector{Vector{T}}}`:
  `qs_all[i][a]` is the `q` grid used for state `i`, component `a`.
- `L_all::Vector{Vector{T}}`:
  `L_all[i][a]` is the total length of connected component `a` for state `i`.
  This is useful for plotting vertical seam lines after concatenation.
"""
function husimi_functions_from_us_and_boundary_points_FIXED_GRID(ks::AbstractVector{T},vec_us::AbstractVector{<:AbstractVector{<:Number}},vec_bdComps::AbstractVector{<:Vector{BoundaryPointsCFIE{T}}},nx::Integer,ny::Integer;full_p::Bool=false,normalize_components::Bool=true) where {T<:Real}
    length(ks)==length(vec_us)==length(vec_bdComps) || error("Input vectors must have equal length")
    ps=full_p ? collect(range(-one(T),one(T),length=ny)) : collect(range(zero(T),one(T),length=cld(ny,2)))
    Hs_all=Vector{Vector{Matrix{T}}}(undef,length(ks))
    qs_all=Vector{Vector{Vector{T}}}(undef,length(ks))
    L_all=Vector{Vector{T}}(undef,length(ks))
    ok=trues(length(ks))
    pbar=Progress(length(ks);desc="Husimi N=$(length(ks))")
    Threads.@threads for i in eachindex(ks)
        try
            u_comps,s_comps,L_comps=split_boundary_data_by_component(vec_bdComps[i],vec_us[i])
            qs_list=[collect(range(zero(T),stop=Lc,length=nx)) for Lc in L_comps]
            Hs,qs_out,_=husimi_on_grid_components(ks[i],s_comps,u_comps,L_comps,qs_list,ps;full_p=full_p,normalize_components=normalize_components)
            Hs_all[i]=Hs
            qs_all[i]=qs_out
            L_all[i]=collect(L_comps)
        catch e
            @debug "Husimi fail at k=$(ks[i])" exception=(e,catch_backtrace())
            ok[i]=false
        end
        next!(pbar)
    end
    Hs_all=Hs_all[ok]
    qs_all=qs_all[ok]
    L_all=L_all[ok]
    ps_out=full_p ? ps : vcat(-reverse(ps)[1:end-1],ps)
    return Hs_all,ps_out,qs_all,L_all
end