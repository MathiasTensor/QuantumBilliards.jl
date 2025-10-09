using LinearAlgebra, StaticArrays, TimerOutputs, Bessels, ProgressMeter

#### EXPANDED BIM ####

"""
    struct ExpandedBoundaryIntegralMethod{T<:Real}

Represents the configuration for the expanded boundary integral method.

# Fields
- `dim_scaling_factor::T`: Scaling factor for the boundary dimensions (compatibility).
- `pts_scaling_factor::Vector{T}`: Scaling factors for the boundary points.
- `sampler::Vector`: Sampling strategy for the boundary points.
- `eps::T`: Numerical tolerance.
- `min_dim::Int64`: Minimum dimensions (compatibility field).
- `min_pts::Int64`: Minimum points for evaluation.
- `symmetry::Union{Vector{Any},Nothing}`: Symmetry for the configuration.
"""
struct ExpandedBoundaryIntegralMethod{T} <: AcceleratedSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64 
    min_pts::Int64
    symmetry::Union{Vector{Any},Nothing}
end

function ExpandedBoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,symmetry::Union{Vector{Any},Nothing}=nothing) where {T<:Real,Bi<:AbsBilliard}
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[LinearNodes()]
    return ExpandedBoundaryIntegralMethod{T}(1.0,bs,sampler,eps(T),min_pts,min_pts,symmetry)
end

function ExpandedBoundaryIntegralMethod(pts_scaling_factor::Union{T,Vector{T}},samplers::Vector,billiard::Bi;min_pts=20,symmetry::Union{Vector{Any},Nothing}=nothing) where {T<:Real,Bi<:AbsBilliard} 
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return ExpandedBoundaryIntegralMethod{T}(1.0,bs,samplers,eps(T),min_pts,min_pts,symmetry)
end

#### NEW MATRIX APPROACH FOR FASTER CODE #### 

"""
    default_helmholtz_kernel_derivative_matrix(bp::BoundaryPointsBIM{T}, k::T) -> Matrix{Complex{T}}

Constructs the first derivative (with respect to `k`) of the 2D Helmholtz kernel *for all pairs* of points
in the boundary `bp`. Each entry `(i, j)` in the returned matrix corresponds to the derivative of

    cos(φᵢ) * (-im*k/2)*r * H₀^(1)(k*r)

where:
- `r` is the distance between points `i` and `j`,
- `cos(φᵢ)` is `(nᵢ · (pᵢ - pⱼ)) / r`, using the normal at `pᵢ`,
- `H₀^(1)` is the Hankel function of the first kind, order 0.

# Arguments
- `bp::BoundaryPointsBIM{T}`: A set of boundary points, including `(x, y)` coordinates and normals.
- `k::T`: Wavenumber, a real value.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: An `N×N` matrix, where `N` is the number of boundary points. The element `(i,j)`
  is the derivative of the Helmholtz kernel with respect to `k` between points `i` and `j`. Diagonal
  entries where `distance < eps(T)` are set to zero.

**Note**: For `(i ≠ j)`, a mirrored computation is performed for entry `(j,i)` using the normal at `pⱼ`.
Hence, the matrix is typically *not* symmetric, because `cos(φ)` depends on the normal at the source row.
"""
function default_helmholtz_kernel_derivative_matrix(bp::BoundaryPointsBIM{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(xy)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    pref=Complex{T}(zero(T),-k/2) # -im*k/2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:(i-1) # symmetric hankel part
            dx=xi-xs[j];dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            invd=inv(d)
            cos_phi=(nxi*dx+nyi*dy)*invd
            hankel=pref*d*Bessels.hankelh1(0,k*d)
            M[i,j]=cos_phi*hankel
            cos_phi_symmetric=(nx[j]*(-dx)+ny[j]*(-dy))*invd # Hankel is symmetric, but cos_phi is not; compute explicitly for M[j, i]
            M[j,i]=cos_phi_symmetric*hankel
        end
    end
    M[diagind(M)].=Complex(zero(T),zero(T))
    filter_matrix!(M)
    return M
end

"""
    default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPointsBIM{T}, k::T)
        -> Matrix{Complex{T}}

Constructs the second derivative (with respect to `k`) of the 2D Helmholtz kernel *for all pairs* of points
in the boundary `bp`. Each entry `(i, j)` in the returned matrix corresponds to

    cos(φᵢ) * ( im/(2*k) ) * [ ...combination of HankelH1(1) and HankelH1(2)... ]

where `cos(φᵢ) = (nᵢ · (pᵢ - pⱼ)) / r` and `r` is the distance between boundary points `pᵢ` and `pⱼ`.
The exact Hankel expression matches the partial derivative:
    
    (d²/dk²) of [ cos(φᵢ) * H₀^(1)(k*r)* ... ].

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points, containing `(x, y)` and normals.
- `k::T`: Wavenumber, real.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: An `N×N` matrix, where each entry is the second derivative of the Helmholtz kernel
  wrt. `k` between boundary points `i` and `j`. If `distance < eps(T)`, the entry is set to zero.

**Note**: Similar to the first-derivative matrix, the factor `cos(φᵢ)` uses the normal at the source
row `i`, so the matrix is not necessarily symmetric unless the geometry enforces it.
"""
@inline function default_helmholtz_kernel_second_derivative_matrix(bp::BoundaryPointsBIM{T},k::T;multithreaded::Bool=true) where {T<:Real}
    xy=bp.xy
    normals=bp.normal
    N=length(xy)
    M=Matrix{Complex{T}}(undef,N,N)
    xs=getindex.(xy,1)
    ys=getindex.(xy,2)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    pref=Complex{T}(zero(T),inv(2*k)) # im/(2*k)
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i]
        nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:(i-1) # symmetric hankel part
            dx=xi-xs[j];dy=yi-ys[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            invd=inv(d)
            cos_phi=(nxi*dx+nyi*dy)*invd
            hankel=pref*((-2+(k*d)^2)*Bessels.hankelh1(1,k*d)+k*d*Bessels.hankelh1(2,k*d))
            M[i,j]=cos_phi*hankel
            cos_phi_symmetric=(nx[j]*(-dx)+ny[j]*(-dy))*invd # Hankel is symmetric, but cos_phi is not; compute explicitly for M[j, i]
            M[j,i]=cos_phi_symmetric*hankel
        end
    end
    M[diagind(M)].=Complex(zero(T),zero(T))
    filter_matrix!(M)
    return M
end

"""
    _add_pair3_no_symmetry_default!(
        K::AbstractMatrix{C},
        dK::AbstractMatrix{C},
        ddK::AbstractMatrix{C},
        i::Int, j::Int,
        xi::T, yi::T, nxi::T, nyi::T,
        xj::T, yj::T, nxj::T, nyj::T,
        κi::T, k::T, tol2::T;
        scale::Union{T,Complex{T}} = one(Complex{T})
    )::Bool where {T<:Real, C<:Complex}

Add the default 2D Helmholtz double-layer contribution for the pair `(i,j)` **without symmetry images**, together with
its first and second derivatives w.r.t. `k`. On the diagonal (`i==j`) the curvature term `κi/(2π)` is added to `K`
and the function returns `false`. For off-diagonal pairs, the routine fills both directions `(i,j)` and `(j,i)`.

The default kernel and its k-derivatives are:
- `K:   cosφ * (-im*k/2) * H₁^{(1)}(k r)`
- `dK:  cosφ * (-im*k/2) * r * H₀^{(1)}(k r)`
- `ddK: cosφ * (im/(2k)) * [ (-2 + (k r)^2) H₁^{(1)}(k r) + (k r) H₂^{(1)}(k r) ]`

where `r = ‖(xi,yi)-(xj,yj)‖` and `cosφ = (nxi,nyi)⋅((xi,yi)-(xj,yj))/r`.

# Arguments
- `K, dK, ddK`: `AbstractMatrix{C}` – destination matrices for kernel, first and second k-derivatives.
- `i, j`: `Int` – target and source indices.
- `xi, yi`: `T` – target coordinates.
- `nxi, nyi`: `T` – target outward unit normal.
- `xj, yj`: `T` – source coordinates.
- `nxj, nyj`: `T` – source outward unit normal (used only to fill `(j,i)` entries).
- `κi`: `T` – boundary curvature at target point `i`.
- `k`: `T` – (real) wavenumber.
- `tol2`: `T` – squared distance threshold; pairs with `r^2 ≤ tol2` are treated as singular.
- `scale`: `Union{T,Complex{T}}` – multiplicative factor (e.g., parity `±1` or a symmetry character). Defaults to `1+0im`.

# Returns
- `Bool`: `false` for diagonal/self (`i==j`, curvature term added to `K[i,i]`), `true` for regular off-diagonal pairs.
"""
@inline function _add_pair3_no_symmetry_default!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real,C<:Complex}
    dx=xi-xj;dy=yi-yj
    d2=muladd(dx,dx,dy*dy)
    if i==j # when we have no symmetry safely modify the Diagonal elements, otherwise the d^2<tol2 check in the symmetry version 
        @inbounds K[i,j]+=scale*Complex(κi/TWO_PI)
        return false
    end
    d=sqrt(d2);invd=inv(d);kd=k*d
    c_ij=(nxi*dx+nyi*dy)*invd # cosϕ[i,j]
    c_ji=(nxj*(-dx)+nyj*(-dy))*invd # cosϕ[j,i]
    H0,H1,H2=Bessels.besselh(0:2,1,kd) # allocates a 3-vector, but this is the biggest efficiency due to reccurence
    pref=Complex{T}(zero(T),-k/2)  # base: (-im*k/2) * H1(kd)
    pref2=Complex{T}(zero(T),inv(2*k)) # second derivative prefix
    hK=pref*H1 # base val (before cosϕ) # (-im*k/2)*H1(kd)
    hdK=pref*d*H0  # first derivative val (-im*k/2)*d*H0(kd)
    hddK=pref2*((-2+kd*kd)*H1+kd*H2) # second derivative:  im/(2k) * [ (-2 + (kd)^2) H1(kd) + kd H2(kd) ]
    @inbounds begin 
        K[i,j]+=scale*(c_ij*hK)
        dK[i,j]+=scale*(c_ij*hdK)
        ddK[i,j]+=scale*(c_ij*hddK)
        K[j,i]+=scale*(c_ji*hK) # i!=j since otherwise it would terminate in the i==j check
        dK[j,i]+=scale*(c_ji*hdK)
        ddK[j,i]+=scale*(c_ji*hddK)
    end
    return true
end

"""
    _add_pair3_custom!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T,kernel_fun,kernel_der_fun,kernel_der2_fun;scale::Union{T,Complex{T}}=one(Complex{T}))::Bool where {T<:Real,C<:Complex}

Add a custom kernel and its first/second k-derivatives for pair `(i,j)`. The three callbacks must all have
the signature: `(i::Int, j::Int, xi::T, yi::T, nxi::T, nyi::T, xj::T, yj::T, nxj::T, nyj::T, k::T) :: Complex{T}`
Their returned values are multiplied by `scale` and accumulated into `K[i,j], dK[i,j], ddK[i,j]`.

# Arguments
- `K, dK, ddK`: `AbstractMatrix{C}` – destination matrices.
- `i, j`: `Int` – target and source indices.
- `xi, yi, nxi, nyi`: `T` – target point and outward normal.
- `xj, yj, nxj, nyj`: `T` – source point and outward normal. **If you apply symmetry images upstream, pass the transformed normal here.**
- `κi`: `T` – curvature at target `i` (not used in this routine; included for signature uniformity).
- `k`: `T` – wavenumber.
- `tol2`: `T` – squared distance tolerance (unused here; included for signature uniformity).
- `kernel_fun`: `Function` – base kernel evaluator.
- `kernel_der_fun`: `Function` – first derivative in `k`.
- `kernel_der2_fun`: `Function` – second derivative in `k`.
- `scale`: `Union{T,Complex{T}}` – symmetry factor; e.g., `±1` for reflections or `exp(iθ)` for rotations.

# Returns
- `Bool`: always `true` (no singular handling is done here; your callbacks are expected to be well-defined).
"""
@inline function _add_pair3_custom!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T,kernel_fun,kernel_der_fun,kernel_der2_fun;scale::Union{T,Complex{T}}=one(Complex{T})) where {T<:Real,C<:Complex}
    val_ij=kernel_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)*scale
    val_der_ij=kernel_der_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)*scale
    val_der2_ij=kernel_der2_fun(i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,k)*scale
    @inbounds begin
        K[i,j]+=val_ij
        dK[i,j]+=val_der_ij
        ddK[i,j]+=val_der2_ij
    end
    return true
end

"""
    _add_pair3_image_default!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xjr::T,yjr::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::Union{T,Complex{T}}=one(Complex{T}))::Bool where {T<:Real}

Add the **default** Helmholtz double-layer kernel (and its k-derivatives) for a **symmetry image** of the source,
i.e. the source at `(xjr, yjr)` obtained by reflection/rotation of `(xj, yj)`. No curvature is added for coincident
image pairs; if `‖(xi,yi)-(xjr,yjr)‖^2 ≤ tol2`, the contribution is skipped and the function returns `false`.

See `_add_pair3_no_symmetry_default!` for the exact kernel forms.

# Arguments
- `K, dK, ddK`: `AbstractMatrix{Complex{T}}` – destination matrices.
- `i, j`: `Int` – target and (original) source indices.
- `xi, yi, nxi, nyi`: `T` – target coordinates and outward unit normal.
- `xjr, yjr`: `T` – **image** source coordinates (already reflected/rotated).
- `nxj, nyj`: `T` – original source outward normal (unused by default kernel for the `(i,j)` entry).
- `κi`: `T` – curvature at target `i` (not used here since no diagonal/image curvature is added).
- `k`: `T` – wavenumber.
- `tol2`: `T` – squared distance tolerance for skipping coincident image pairs.
- `scale`: `Union{T,Complex{T}}` – symmetry factor (parity `±1` or rotation character `e^{iθ}`).

# Returns
- `Bool`: `true` if the image contribution was added; `false` if it was skipped due to `r^2 ≤ tol2`.
"""
@inline function _add_pair3_image_default!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xjr::T,yjr::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::Union{T,Complex{T}}=one(Complex{T}))::Bool where {T<:Real}
    dx=xi-xjr; dy=yi-yjr
    d2=muladd(dx,dx,dy*dy)
    if d2<=tol2 # For reflections/rotations: if coincident (rare), just skip — no curvature term here. It could happen that one collocation point is very close to the border and the reflected/rotated point is too close to it
        return false # no curvature on reflected self-pairs
    end
    d=sqrt(d2);invd=inv(d);kd=k*d
    cij=(nxi*dx+nyi*dy)*invd
    H0,H1,H2=Bessels.besselh(0:2,1,kd) # allocates a 3-vector, but this is the biggest efficiency due to reccurence
    pref=Complex{T}(zero(T),-k/2)  # base: (-im*k/2) * H1(kd)
    pref2=Complex{T}(zero(T),inv(2*k)) # second derivative prefix
    hK=pref*H1 # base val (before cosϕ) # (-im*k/2)*H1(kd)
    hdK=pref*d*H0  # first derivative val (-im*k/2)*d*H0(kd)
    hddK=pref2*((-2+kd*kd)*H1+kd*H2) # second derivative:  im/(2k) * [ (-2 + (kd)^2) H1(kd) + kd H2(kd) ]
    @inbounds begin
        K[i,j]+=scale*(cij*hK)
        dK[i,j]+=scale*(cij*hdK)
        ddK[i,j]+=scale*(cij*hddK)
    end
    return true
end

"""
    compute_kernel_matrix_with_derivatives(bp::BoundaryPointsBIM{T},k::T;multithreaded::Bool=true,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {T<:Real} -> Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}

Build the kernel matrix `K` and its first/second derivatives w.r.t. `k` **without symmetry images** for a set of
boundary points.

If `kernel_fun === (:default,:first,:second)`, uses the built-in double-layer Helmholtz forms (see
`_add_pair3_no_symmetry_default!`). Otherwise, pass a triple of callbacks with signature: (i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,k::T) :: Complex{T}
# Arguments
- `bp`: `BoundaryPointsBIM{T}` – boundary data (points `xy`, normals, curvature `κ`, and arc-length weights `ds`).
- `k`: `T` – (real) wavenumber about which derivatives are taken.
- `multithreaded`: `Bool` – enable threaded assembly.
- `kernel_fun`: `Union{Tuple{Symbol,Symbol,Symbol}, Tuple{Function,Function,Function}}` – choice of kernel/derivatives.

# Returns
- `K`:   `Matrix{Complex{T}}` – kernel matrix.
- `dK`:  `Matrix{Complex{T}}` – first derivative w.r.t. `k`.
- `ddK`: `Matrix{Complex{T}}` – second derivative w.r.t. `k`.

Notes:
- Diagonal entries of `K` receive the curvature term `κ/(2π)`; `dK` and `ddK` have zero diagonals for the default kernels.
- Off-diagonal entries fill both `(i,j)` and `(j,i)` for the default kernels to account for different normals.
"""
function compute_kernel_matrix_with_derivatives(bp::BoundaryPointsBIM{T},k::T;multithreaded::Bool=true,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {T<:Real}
    N=length(bp.xy)
    K=zeros(Complex{T},N,N)
    dK=zeros(Complex{T},N,N)
    ddK=zeros(Complex{T},N,N)
    xs=getindex.(bp.xy,1);ys=getindex.(bp.xy,2)
    nx=getindex.(bp.normal,1);ny=getindex.(bp.normal,2)
    κ=bp.curvature
    tol2=(eps(T))^2
    isdef=(kernel_fun[1]===:default) # this is enough of a check
    @use_threads multithreading=multithreaded for i in 1:N
        xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:i
            xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
            if isdef
                _add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2)
            else
                _add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3])
            end
        end
    end
    return K,dK,ddK
end

"""
    compute_kernel_matrix_with_derivatives(bp::BoundaryPointsBIM{T},symmetry::Vector{Any},k::T;multithreaded::Bool=true,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {T<:Real} -> Tuple{Matrix{Complex{T}},Matrix{Complex{T}},Matrix{Complex{T}}}

Build the kernel matrix `K` and its first/second derivatives w.r.t. `k` **including symmetry images**. Supported
symmetries are:
- Reflections with fields `axis` (`:x_axis`, `:y_axis`, or `:origin`) and `parity` (`±1` for single-axis or a length-2
  tuple for `:origin`). These contribute with scale factors `sxgn`, `sygn`, or `sxy = sxgn*sygn`.
- `Rotation` symmetries with fields `n::Int` (order), `m::Int` (representation index, taken `mod n`), and
  `center::Tuple{T,T}`. Images are added for `l=1,…,n-1` by rotating the source and multiplying by the character
  `χ_l = exp(im * 2π * m * l / n)`.

For the **default** kernels the source normal of images is not transformed (the DLP at `(i,j)` uses the **target**
normal). For **custom** kernels, the image source normal is reflected/rotated before calling the user callbacks.

# Arguments
- `bp`: `BoundaryPointsBIM{T}` – boundary data (points `xy`, normals, curvature `κ`, arc-length weights `ds`,
  and shifts `shift_x`, `shift_y` for reflection axes).
- `symmetry`: `Vector{Any}` – list of symmetry descriptors (reflections and/or `Rotation`).
- `k`: `T` – wavenumber.
- `multithreaded`: `Bool` – enable threaded assembly.
- `kernel_fun`: `Union{Tuple{Symbol,Symbol,Symbol}, Tuple{Function,Function,Function}}` – built-in or custom kernels
  as in the non-symmetry version.

# Returns
- `K`:   `Matrix{Complex{T}}` – kernel matrix including image contributions.
- `dK`:  `Matrix{Complex{T}}` – first derivative w.r.t. `k`.
- `ddK`: `Matrix{Complex{T}}` – second derivative w.r.t. `k`.

Notes:
- Image self-pairs are **not** given curvature; if an image falls within `tol2` the contribution is skipped.
- Reflection scales are real (`±1`); rotation scales are unit-modulus complex characters `χ_l`.
"""
function compute_kernel_matrix_with_derivatives(bp::BoundaryPointsBIM{T},symmetry::Vector{Any},k::T;multithreaded::Bool=true,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)) where {T<:Real}
    N=length(bp.xy)
    K=zeros(Complex{T},N,N)
    dK=zeros(Complex{T},N,N)
    ddK=zeros(Complex{T},N,N)
    xs=getindex.(bp.xy,1);ys=getindex.(bp.xy,2)
    nx=getindex.(bp.normal,1);ny=getindex.(bp.normal,2)
    κ=bp.curvature
    tol2=(eps(T))^2
    shift_x=bp.shift_x;shift_y=bp.shift_y
    add_x=false;add_y=false;add_xy=false
    sxgn=one(T);sygn=one(T);sxy=one(T)
    have_rot=false
    nrot=1;mrot=0
    cx=zero(T);cy=zero(T)
    @inbounds for s in symmetry # symmetry here is always != nothing
        if hasproperty(s,:axis)
            if s.axis==:y_axis;add_x=true;sxgn=(s.parity==-1 ? -one(T) : one(T)); end
            if s.axis==:x_axis;add_y=true;sygn=(s.parity==-1 ? -one(T) : one(T)); end
            if s.axis==:origin
                add_x=true;add_y=true;add_xy=true
                sxgn=(s.parity[1]==-1 ? -one(T) : one(T))
                sygn=(s.parity[2]==-1 ? -one(T) : one(T))
                sxy=sxgn*sygn
            end
        elseif s isa Rotation
            have_rot=true
            nrot=s.n
            mrot=mod(s.m,nrot)
            cx,cy=s.center
        end
    end
    if have_rot
        ctab,stab,χ=_rotation_tables(T,nrot,mrot)
    end
    isdef=(kernel_fun[1]===:default) # this is enough of a check
    @use_threads multithreading=multithreaded for i in 1:N
            xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i] # i is the target, j is the source
            @inbounds for j in 1:N
                xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
                # base (upper triangle only; mirrors into [j,i]; curvature on diag)
                if j<=i
                    if isdef 
                        _add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2)
                    else
                        _add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3])
                    end
                end
                # reflected legs (always full j=1:N; never add curvature)
                if add_x
                    xjr=_x_reflect(xj,shift_x);yjr=yj
                    if isdef
                        _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sxgn)
                    else
                        nxjr,nyjr=_x_reflect_normal(nxj,nyj)
                        _add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3];scale=sxgn) 
                    end
                end
                if add_y
                    xjr=xj;yjr=_y_reflect(yj,shift_y)
                    if isdef
                        _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sygn)
                    else
                        nxjr,nyjr=_y_reflect_normal(nxj,nyj)
                        _add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3];scale=sygn)
                    end
                end
                if add_xy
                    xjr=_x_reflect(xj,shift_x);yjr=_y_reflect(yj,shift_y)
                    if isdef
                        _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sxy)
                    else
                        nxjr,nyjr=_xy_reflect_normal(nxj,nyj)
                        _add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3];scale=sxy)
                    end
                end
                if have_rot
                @inbounds for l in 1:nrot-1 # l=0 is the direct term we already added; add l=1..nrot-1
                    cl=ctab[l+1];sl=stab[l+1]
                    xjr,yjr=_rot_point(xj,yj,cx,cy,cl,sl)
                    phase=χ[l+1]  # e^{i 2π m l / n}, reflections due to being 1d-irreps have real characters
                    if isdef
                        _add_pair3_image_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=phase)
                    else
                        nxjr,nyjr=_rot_vec(nxj,nyj,cl,sl)
                        _add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxjr,nyjr,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3];scale=phase)
                    end
                end
            end
            end
        end
    return K,dK,ddK
end

"""
    fredholm_matrix_with_derivatives(bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},k::T;kernel_fun::Union{Symbol,Function}=:first,multithreaded::Bool=true) where {T<:Real}

Build the Fredholm matrix `A` and it's derivative matrices `dA/dk & d^2A/dk^2`.

# Arguments
- `bp::BoundaryPointsBIM{T}`: Boundary points with `(x,y)`, normals, and `ds`.
- `symmetry::Vector{Any}`: Symmetry to apply.
- `k::T`: Wavenumber.
- `kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)`: If `:first`, uses the first derivative matrix; if `:second`, uses the second derivative matrix; else a custom function. All these have the same signature: `kernel_fun(i,j, xi,yi,nxi,nyi, xj,yj,nxj,nyj, k) :: Complex`.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Tuple{Matrix{Complex{T}},Matrix{Complex{T}},Matrix{Complex{T}}}`: The 3`N×N` matrices representing the Fredholm matrix and it's first and second derivative, respectively.
"""
function fredholm_matrix_with_derivatives(bp::BoundaryPointsBIM{T},symmetry::Union{Vector{Any},Nothing},k::T;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true) where {T<:Real}
    if isnothing(symmetry)
        K,dK,ddK=compute_kernel_matrix_with_derivatives(bp,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    else
        K,dK,ddK=compute_kernel_matrix_with_derivatives(bp,symmetry,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    end
    ds=bp.ds
    @inbounds for j in 1:length(ds)
        @views K[:,j].*=ds[j]
        @views dK[:,j].*=ds[j]
        @views ddK[:,j].*=ds[j]
    end
    K.*=-one(T);dK.*=-one(T);ddK.*=-one(T)
    @inbounds for i in axes(K,1) # only the regular kernel has +I, for others vanishes due to taking the derivative
        K[i,i]+=one(T)
    end
    return K,dK,ddK
end

"""
    construct_matrices(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::T;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}

High-level routine that builds the Fredholm matrix and its first/second derivatives
for the given boundary `pts` and wavenumber `k`, relying on the matrix-based approach
in `all_fredholm_associated_matrices`.

# Note: Conditioning far away from a real eigenvalue
```math
log(cond(A)) = 3.0982833773882583, det(A) = 0.004361488184059069 + 0.006306845788048928im
log(cond(dA)) = 18.647070547612767, det(dA) = 0.0 + 0.0im
log(cond(ddA)) = LAPACK crashes, det(A) = NaN + NaN*im
```
# Note: Conditioning extremely close (2e-5 away in the k scale) to real eigenvalue
```math
log(cond(A)) = 4.275721005223111, det(A) = -845.8597645859071 + 1.5705301055355108im
log(cond(dA)) = 18.30361267862381, det(dA) = 0.0 + 0.0im
log(cond(ddA)) = 18.852888588688973, det(A) = -0.0 + 0.0im
```
This shows need for `QZ` algorithm for ggev3/ggev and filtering of βs.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: An EBIM solver configuration (its `rule` is used).
- `basis::Ba`: The basis function type (not used directly here, but part of the pipeline).
- `pts::BoundaryPointsBIM{T}`: Boundary points with geometry data.
- `k::T`: The wavenumber.
- `kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)`: A triple specifying
  (base kernel, first derivative kernel, second derivative kernel).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(A, dA, ddA)::Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}`:
  - `A`: The Fredholm matrix.
  - `dA`: The first derivative wrt `k`.
  - `ddA`: The second derivative wrt `k`.
"""
function construct_matrices(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k::T;kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true) where {Ba<:AbstractHankelBasis,T<:Real}
    return fredholm_matrix_with_derivatives(pts,solver.symmetry,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
end

# Fallback to full ggev if krylov, but this has not been found to happen in testing since krylov is quite stable
function solve_full(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    if use_lapack_raw
        λ,VR,VL=generalized_eigen_all_LAPACK_LEGACY(A,dA)
    else
        λ,VR,VL=generalized_eigen_all(A,dA)
    end
    T=eltype(real.(λ))
    valid=(abs.(real.(λ)).<dk) .& (abs.(imag.(λ)).<dk) # use (-dk,dk) × (-dk,dk) instead of disc of radius dk
    if !any(valid)
        return Vector{T}(),Vector{T}() # early termination
    end
    λ=real.(λ[valid])
    VR=VR[:,valid]
    VL=VL[:,valid]
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        t_v_left=transpose(v_left)
        numerator=t_v_left*ddA*v_right
        denominator=t_v_left*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    λ_corrected=k.+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    return λ_corrected,tens
end

"""
    solve(
        solver::ExpandedBoundaryIntegralMethod, 
        basis::Ba, 
        pts::BoundaryPointsBIM{T}, 
        k::T, 
        dk::T;
        use_lapack_raw::Bool=false,
        kernel_fun::Union{Tuple,Function} = (:default, :first, :second)
    ) -> (Vector{T}, Vector{T})

TRADITIONAL FUNTION
Compute approximate "corrected" eigenvalues near the wavenumber `k`, using the expanded boundary integral
method. The routine builds `(A, dA, ddA)` via `construct_matrices`, then solves the generalized eigenvalue
problem `A*x = λ * dA * x`. It filters those eigenvalues whose real part lies in `(-dk, dk)`, then applies
a second-order correction with `ddA`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: EBIM configuration.
- `basis::Ba`: The basis type (unused directly here, but part of the solver pipeline).
- `pts::BoundaryPointsBIM{T}`: Boundary points.
- `k::T`: Central wavenumber.
- `dk::T`: Half-width of the search interval in real and imaginary parts of `λ`.
- `use_lapack_raw::Bool=false`: If true, call a direct LAPACK routine for `A,dA` eigen solves.
- `kernel_functions::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second)`: The base kernel and its derivatives.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `use_krylov::Bool=true`: Large speedups in EPV calculation. If anomalies in result are present set this flag to `False`.

# Returns
- `(λ_corrected::Vector{T}, tension::Vector{T})`: The "corrected" wavenumbers (`k + corrections`)
  for the valid solutions, and a tension measure (`abs(corrections)`).

**Note**: The corrections are computed from the first- and second-order expansions in terms of `λ[i]`,
with final `k_corrected = k + corr₁ + corr₂`.
"""
function solve(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true,use_krylov::Bool=true) where {Ba<:AbstractHankelBasis}
    if use_krylov
        return solve_krylov(solver,basis,pts,k,dk;kernel_fun=kernel_fun,multithreaded=multithreaded)
    else
        return solve_full(solver,basis,pts,k,dk;use_lapack_raw=use_lapack_raw,kernel_fun=kernel_fun,multithreaded=multithreaded)
    end
end

# HELPS PROFILE THE solve FUNCTION AND DETERMINE THE CRITICAL PARAMETERS OF A CALCULATION. THIS USES THE FULL GEPV SOLVE
function solve_full_INFO(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    s=time()
    s_constr=time()
    @info "Constructing A,dA,ddA Fredholm matrix and it's derivatives..."
    @time A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    e_constr=time()
    if use_lapack_raw
        if LAPACK.version()<v"3.6.0"
            s_gev=time()
            @info "Doing ggev!"
            @info "Matrix condition numbers: cond(A) = $(cond(A)), cond(dA) = $(cond(dA))"
            @time α,β,VL,VR=LAPACK.ggev!('V','V',copy(A),copy(dA))
            e_gev=time()
        else
            s_gev=time()
            @info "Doing ggev3!"
            @info "Matrix condition numbers: cond(A) = $(cond(A)), cond(dA) = $(cond(dA))"
            @time α,β,VL,VR=LAPACK.ggev3!('V','V',copy(A),copy(dA))
            e_gev=time()
        end
        λ=α./β
        valid_indices=.!isnan.(λ).&.!isinf.(λ)
        @info "% of valid indices: $(count(valid_indices)/length(λ))"
        λ=λ[valid_indices]
        VR=VR[:,valid_indices]
        VL=VL[:,valid_indices]
        sort_order=sortperm(abs.(λ)) 
        λ=λ[sort_order]
        @info "Smallest eigenvalue: $(minimum(abs.(λ)))"
        VR=VR[:,sort_order]
        VL=VL[:,sort_order]
        normalize!(VR)
        normalize!(VL)
    else
        @info "Solving Julia's ggev for A, dA"
        s_gev=time()
        @time F=eigen(A,dA)
        λ=F.values
        VR=F.vectors 
        @info "Solving Julia's ggev for A' and dA' for the left eigenvectors"
        F_adj=eigen(A',dA') 
        e_gev=time()
        VL=F_adj.vectors 
        valid_indices=.!isnan.(λ).&.!isinf.(λ)
        @info "Number of valid indices: $(count(valid_indices))"
        λ=λ[valid_indices]
        VR=VR[:,valid_indices]
        VL=VL[:,valid_indices]
        sort_order=sortperm(abs.(λ)) 
        λ=λ[sort_order]
        @info "Smallest eigenvalue: $(minimum(abs.(λ)))"
        VR=VR[:,sort_order]
        VL=VL[:,sort_order]
        normalize!(VR)
        normalize!(VL)
    end
    T=eltype(real.(λ))
    valid=(abs.(real.(λ)).<dk) .& (abs.(imag.(λ)).<dk) # use (-dk,dk) × (-dk,dk) instead of disc of radius dk
    #valid=abs.(λ).<dk 
    if !any(valid) # early termination
        total_time=time()-s
        @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
        println("%%%%% SUMMARY %%%%%")
        println("Percentage of total time (most relevant ones): ")
        println("A, dA, ddA construction: $(100*(e_constr-s_constr)/total_time) %")
        println("Generalized eigen: $(100*(e_gev-s_gev)/total_time) %")
        println("%%%%%%%%%%%%%%%%%%%")
        return Vector{T}(),Vector{T}() # early termination. In the INFO function we will get information upon finding a succesful find.
    end
    λ=real.(λ[valid])
    VR=VR[:,valid]
    VL=VL[:,valid]
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    @info "Corrections to the eigenvalues and eigenvectors..."
    s_corr=time()
    @time for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        t_v_left=transpose(v_left)
        numerator=t_v_left*ddA*v_right
        denominator=t_v_left*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    e_corr=time()
    λ_corrected=k.+corr_1.+corr_2
    tens=abs.(corr_1.+corr_2)
    e=time()
    total_time=e-s
    @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("A, dA, ddA construction: $(100*(e_constr-s_constr)/total_time) %")
    println("Generalized eigen: $(100*(e_gev-s_gev)/total_time) %")
    println("2nd order corrections: $(100*(e_corr-s_corr)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return λ_corrected,tens
end

function solve_INFO(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true,use_krylov::Bool=true) where {Ba<:AbstractHankelBasis}
    if use_krylov
        return solve_krylov_INFO(solver,basis,pts,k,dk,kernel_fun=kernel_fun,multithreaded=multithreaded)
    else
        return solve_full_INFO(solver,basis,pts,k,dk;use_lapack_raw=use_lapack_raw,kernel_fun=kernel_fun,multithreaded=multithreaded)
    end
end