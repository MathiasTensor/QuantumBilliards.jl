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
    default_helmholtz_kernel_derivative_matrix(bp_s::BoundaryPointsBIM{T}, xy_t::Vector{SVector{2,T}}, k::T)
        -> Matrix{Complex{T}}

Constructs the first derivative (with respect to `k`) of the 2D Helmholtz kernel between each boundary point
in `bp_s` and a separate list of target points `xy_t`. This is similar to the single-argument version, except
the second set of points is taken from `xy_t` rather than the boundary itself.

# Arguments
- `bp_s::BoundaryPointsBIM{T}`: Source boundary points, which provide `(x, y)` and normal vectors.
- `xy_t::Vector{SVector{2,T}}`: Target points, typically a different set of points in the plane.
- `k::T`: Wavenumber, a real value.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: An `N×M` matrix if `bp_s` has `N` points and `xy_t` has `M` points. The element `(i,j)`
  is the derivative of the Helmholtz kernel w.r.t. `k` between source point `i` and target point `j`.
  If the distance is `< eps(T)`, the entry is set to zero.

**Note**: This routine does not rely on reflection logic. It simply computes the derivative
for each `(source_i, target_j)` pair based on the distance and the dot product with `normalᵢ`.
"""
function default_helmholtz_kernel_derivative_matrix(bp_s::BoundaryPointsBIM{T},xy_t::Vector{SVector{2,T}},k::T;multithreaded::Bool=true) where {T<:Real}
    xy_s=bp_s.xy
    normals=bp_s.normal
    N=length(xy_s)
    M=Matrix{Complex{T}}(undef,N,N)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    x_s=getindex.(xy_s,1)
    y_s=getindex.(xy_s,2)
    x_t=getindex.(xy_t,1)
    y_t=getindex.(xy_t,2)
    tol=eps(T)
    pref=Complex{T}(zero(T),-k/2) # -im*k/2
    @use_threads multithreading=multithreaded for i in 1:N
        xi=x_s[i];yi=y_s[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:N
            dx=xi-x_t[j];dy=yi-y_t[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            invd=inv(d)
            if d<tol # pt on reflection axis!
                M[i,j]=Complex(zero(T),zero(T))
            else
                cos_phi=(nxi*dx+nyi*dy)*invd
                hankel=pref*d*Bessels.hankelh1(0,k*d)
                M[i,j]=cos_phi*hankel
            end
        end
    end
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
    default_helmholtz_kernel_second_derivative_matrix(bp_s::BoundaryPointsBIM{T},
                                                      xy_t::Vector{SVector{2,T}},
                                                      k::T) -> Matrix{Complex{T}}

Constructs the second derivative (with respect to `k`) of the 2D Helmholtz kernel between each source point
in `bp_s` and a separate list of target points `xy_t`. The entry `(i, j)` in the returned `N×M` matrix is
the second derivative of the Helmholtz kernel, computed using the normal at source `i` and the distance
to target `j`.

# Arguments
- `bp_s::BoundaryPointsBIM{T}`: Source boundary points with `(x, y)` and normal vectors.
- `xy_t::Vector{SVector{2,T}}`: A vector of target points in the plane.
- `k::T`: Wavenumber, real.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `Matrix{Complex{T}}`: If there are `N` source points and `M` target points, returns `N×M`. Any pair
  whose distance is `< eps(T)` yields a zero entry.

**Note**: This function does not include reflection or symmetry corrections. It purely evaluates
the second derivative of the kernel formula at each `(source_i, target_j)`.
"""
@inline function default_helmholtz_kernel_second_derivative_matrix(bp_s::BoundaryPointsBIM{T},xy_t::Vector{SVector{2,T}},k::T;multithreaded::Bool=true) where {T<:Real}
    xy_s=bp_s.xy
    normals=bp_s.normal
    N=length(xy_s)
    M=Matrix{Complex{T}}(undef,N,N)
    nx=getindex.(normals,1)
    ny=getindex.(normals,2)
    x_s=getindex.(xy_s,1)
    y_s=getindex.(xy_s,2)
    x_t=getindex.(xy_t,1)
    y_t=getindex.(xy_t,2)
    tol=eps(T)
    pref=Complex{T}(zero(T),inv(2*k)) # im/(2*k)
    @use_threads multithreading=multithreaded for i in 1:N
        xi=x_s[i];yi=y_s[i];nxi=nx[i];nyi=ny[i]
        @inbounds for j in 1:N
            dx=xi-x_t[j];dy=yi-y_t[j]
            d=sqrt(muladd(dx,dx,dy*dy))
            invd=inv(d)
            if d<tol # pt on reflection axis!
                M[i,j]=Complex(zero(T),zero(T))
            else
                cos_phi=(nxi*dx+nyi*dy)*invd
                hankel=pref*((-2+(k*d)^2)*Bessels.hankelh1(1,k*d)+k*d*Bessels.hankelh1(2,k*d))
                M[i,j]=cos_phi*hankel
            end
        end
    end
    filter_matrix!(M)
    return M
end

@inline function add_pair3_no_symmetry_default!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::T=one(T)) where {T<:Real,C<:Complex}
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

@inline function add_pair3_custom!(K::AbstractMatrix{C},dK::AbstractMatrix{C},ddK::AbstractMatrix{C},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xj::T,yj::T,nxj::T,nyj::T,κi::T,k::T,tol2::T,kernel_fun,kernel_der_fun,kernel_der2_fun;scale::T=one(T)) where {T<:Real,C<:Complex}
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

@inline function add_pair3_reflected_default!(K::AbstractMatrix{Complex{T}},dK::AbstractMatrix{Complex{T}},ddK::AbstractMatrix{Complex{T}},i::Int,j::Int,xi::T,yi::T,nxi::T,nyi::T,xjr::T,yjr::T,nxj::T,nyj::T,κi::T,k::T,tol2::T;scale::T=one(T))::Bool where {T<:Real}
    dx=xi-xjr; dy=yi-yjr
    d2=muladd(dx,dx,dy*dy)
    if d2<=tol2
        @inbounds K[i,j]+=scale*Complex(κi/TWO_PI)
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
            if isdef ? 
                add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2) :
                add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3])
            end
        end
    end
    return K,dK,ddK
end

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
    @inbounds for s in symmetry
        if s.axis==:y_axis
            add_x=true
            sxgn=(s.parity==-1 ? -one(T) : one(T))
        elseif s.axis==:x_axis
            add_y=true
            sygn=(s.parity==-1 ? -one(T) : one(T))
        elseif s.axis==:origin
            add_x=true;add_y=true;add_xy=true
            sxgn=(s.parity[1]==-1 ? -one(T) : one(T))
            sygn=(s.parity[2]==-1 ? -one(T) : one(T))
            sxy=sxgn*sygn
        end
    end
    isdef=(kernel_fun[1]===:default) # this is enough of a check
    @use_threads multithreading=multithreaded for i in 1:N
            xi=xs[i];yi=ys[i];nxi=nx[i];nyi=ny[i]
            @inbounds for j in 1:N
                xj=xs[j];yj=ys[j];nxj=nx[j];nyj=ny[j]
                # base (upper triangle only; mirrors into [j,i]; curvature on diag)
                if j<=i
                    isdef ? 
                    add_pair3_no_symmetry_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2) : 
                    add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xj,yj,nxj,nyj,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3])
                end
                # reflected legs (always full j=1:N; never add curvature)
                if add_x
                    xjr=_x_reflect(xj,shift_x);yjr=yj
                    isdef ?
                    add_pair3_reflected_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sxgn) :
                    add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2,kernel_functions[1],kernel_functions[2],kernel_fun[3];scale=sxgn) 
                end
                if add_y
                    xjr=xj;yjr=_y_reflect(yj,shift_y)
                    isdef ?
                    add_pair3_reflected_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sygn) :
                    add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3];scale=sxgn)
                end
                if add_xy
                    xjr=_x_reflect(xj,shift_x);yjr=_y_reflect(yj,shift_y)
                    isdef ?
                    add_pair3_reflected_default!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2;scale=sxy) :
                    add_pair3_custom!(K,dK,ddK,i,j,xi,yi,nxi,nyi,xjr,yjr,nxj,nyj,κ[i],k,tol2,kernel_fun[1],kernel_fun[2],kernel_fun[3];scale=sxgn)
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

# Returns
- `(λ_corrected::Vector{T}, tension::Vector{T})`: The "corrected" wavenumbers (`k + corrections`)
  for the valid solutions, and a tension measure (`abs(corrections)`).

**Note**: The corrections are computed from the first- and second-order expansions in terms of `λ[i]`,
with final `k_corrected = k + corr₁ + corr₂`.
"""
function solve(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
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

# HELPS PROFILE THE solve FUNCTION AND DETERMINE THE CRITICAL PARAMETERS OF A CALCULATION
function solve_INFO(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k,dk;use_lapack_raw::Bool=false,kernel_fun::Union{Tuple{Symbol,Symbol,Symbol},Tuple{Function,Function,Function}}=(:default,:first,:second),multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
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

#### DEBUGGING TOOLS ####

"""
    solve_DEBUG_w_2nd_order_corrections(
        solver::ExpandedBoundaryIntegralMethod,
        basis::Ba,
        pts::BoundaryPointsBIM,
        k;
        kernel_fun=(:default, :first, :second)
    ) -> (Vector{T}, Vector{T}, Vector{T}, Vector{T})

A debug routine that solves the generalized eigenproblem `(A, dA)` at wavenumber `k`, then applies
**both first- and second-order** corrections to refine the approximate roots. Specifically,
it extracts λ from `A*x = λ dA*x`, then does:

  corr₁[i] = -λ[i]
  corr₂[i] = -0.5 * corr₁[i]^2 * real( (v_leftᵀ ddA v_right) / (v_leftᵀ dA v_right) )

Hence two sets of corrected wavenumbers: `k + corr₁` and `k + corr₁ + corr₂`. Tensions are `|corr₁|`
and `|corr₁ + corr₂|`.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The EBIM solver config.
- `basis::Ba`: Basis function type.
- `pts::BoundaryPointsBIM`: Boundary geometry.
- `k`: Wavenumber for the eigenproblem.
- `kernel_fun`: A triple `(base, first, second)` or custom functions for kernel & derivatives.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `(λ_corrected_1, tens_1, λ_corrected_2, tens_2)`: 
   1. `λ_corrected_1 = k + corr₁` (1st-order),
   2. `tens_1 = abs(corr₁)`,
   3. `λ_corrected_2 = k + corr₁ + corr₂` (2nd-order),
   4. `tens_2 = abs(corr₁ + corr₂)`.
"""
function solve_DEBUG_w_2nd_order_corrections(solver::ExpandedBoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun=(:default,:first,:second),multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A,dA,ddA=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    λ,VR,VL=generalized_eigen_all(A,dA)
    valid_indices=.!isnan.(λ).&.!isinf.(λ)
    λ=λ[valid_indices]
    sort_order=sortperm(abs.(λ)) 
    λ=λ[sort_order]
    T=eltype(real.(λ))
    λ=real.(λ)
    corr_1=Vector{T}(undef,length(λ))
    corr_2=Vector{T}(undef,length(λ))
    for i in eachindex(λ)
        v_right=VR[:,i]
        v_left=VL[:,i]
        numerator=transpose(v_left)*ddA*v_right
        denominator=transpose(v_left)*dA*v_right
        corr_1[i]=-λ[i]
        corr_2[i]=-0.5*corr_1[i]^2*real(numerator/denominator)
    end
    λ_corrected_1=k.+corr_1
    λ_corrected_2=λ_corrected_1.+corr_2
    tens_1=abs.(corr_1)
    tens_2=abs.(corr_1.+corr_2)
    return λ_corrected_1,tens_1,λ_corrected_2,tens_2
end

"""
    ebim_inv_diff(kvals::Vector{T}) where {T<:Real}

Computes the inverse of the differences between consecutive elements in `kvals`. This inverts the small differences between the ks very close to the correct eigenvalues and serves as a visual aid or potential criteria for finding missing levels.

# Arguments
- `kvals::Vector{T}`: A vector of values for which differences are calculated.

# Returns
- `Vector{T}`: The `kvals` vector excluding its last element.
- `Vector{T}`: The inverse of the differences between consecutive elements in `kvals`.
"""
function ebim_inv_diff(kvals::Vector{T}) where {T<:Real}
    kvals_diff=diff(kvals)
    kvals=kvals[1:end-1]
    return kvals,T(1.0)./kvals_diff
end

"""
    visualize_ebim_sweep(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1,k2;dk=(k)->(0.05*k^(-1/3)),multithreaded::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbstractHankelBasis,Bi<:AbsBilliard}

Debugging Function to sweep through a range of `k` values and evaluate the smallest tension for each `k` using the EBIM method. This function identifies corrected `k` values based on the generalized eigenvalue problem and associated tensions, collecting those with the smallest tensions for further analysis.

# Usage
hankel_basis=AbstractHankelBasis()
@time ks_debug,tens_debug,ks_debug_small,tens_debug_small=QuantumBilliards.visualize_ebim_sweep(ebim_solver,hankel_basis,billiard,k1,k2;dk=dk)
scatter!(ax,ks_debug,log10.(tens_debug), color=:blue, marker=:xcross)
-> This gives a sequence of points that fall on a vertical line when close to an actual eigenvalue. 

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration for the EBIM method.
- `basis::Ba`: The basis function, a subtype of `AbstractHankelBasis`.
- `billiard::Bi`: The billiard geometry, a subtype of `AbsBilliard`.
- `k1`: The initial value of `k` for the sweep.
- `k2`: The final value of `k` for the sweep.
- `dk::Function`: A function defining the step size as a function of `k` (default: `(k) -> (0.05 * k^(-1/3))`).
- `multithreaded::Bool=false`: If the matrix construction should be multithreaded.
- `multithreaded_ks::Bool=true`: If the ks loop should be rather multithreaded.

# Returns
- `Vector{T}`: All corrected `k` values with low tensions throughout the sweep (`ks_all`).
- `Vector{T}`: Inverse tension corresponding to `ks_all` (`tens_all`), which represent the inverse distances between consecutive `ks_all`. Aa large number indicates that we are probably close to an eigenvalue since solution of the ebim sweep tend to accumulate there.
"""
function visualize_ebim_sweep(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1,k2;dk=(k)->(0.05*k^(-1/3)),multithreaded::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    k=k1
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    T=eltype(k1)
    ks=T[] # these are the evaluation points
    push!(ks,k1)
    k=k1
    while k<k2
        k+=dk(k)
        push!(ks,k)
    end
    ks_all_1=Vector{Union{T,Missing}}(missing,length(ks))
    ks_all_2=Vector{Union{T,Missing}}(missing,length(ks))
    tens_all_1=Vector{Union{T,Missing}}(missing,length(ks))
    tens_all_2=Vector{Union{T,Missing}}(missing,length(ks))
    all_pts=Vector{BoundaryPointsBIM{T}}(undef,length(ks))
    @showprogress desc="Calculating boundary points..." for i in eachindex(ks) 
        all_pts[i]=evaluate_points(bim_solver,billiard,ks[i])
    end
    @info "EBIM smallest tens..."
    p=Progress(length(ks),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)
        ks1,tens1,ks2,tens2=solve_DEBUG_w_2nd_order_corrections(solver,basis,all_pts[i],ks[i],multithreaded=multithreaded)
        idx1=findmin(tens1)[2]
        idx2=findmin(tens2)[2]
        if log10(tens1[idx1])<0.0
            ks_all_1[i]=ks1[idx1]
            tens_all_1[i]=tens1[idx1]   
        end
        if log10(tens2[idx2])<0.0
            ks_all_2[i]=ks2[idx2]
            tens_all_2[i]=tens2[idx2]
        end
        next!(p)
    end
    ks_all_1=skipmissing(ks_all_1)|>collect
    tens_all_1=skipmissing(tens_all_1)|>collect
    ks_all_2=skipmissing(ks_all_2)|>collect
    tens_all_2=skipmissing(tens_all_2)|>collect
    _,logtens_1=ebim_inv_diff(ks_all_1)
    _,logtens_2=ebim_inv_diff(ks_all_2)
    idxs1=findall(x->x>0.0,logtens_1)
    idxs2=findall(x->x>0.0,logtens_2)
    logtens_1=logtens_1[idxs1]
    logtens_2=logtens_2[idxs2]
    ks_all_1=ks_all_1[idxs1]
    ks_all_2=ks_all_2[idxs2]
    return ks_all_1,logtens_1, ks_all_2,logtens_2
end

"""
    visualize_cond_dA_ddA_vs_k(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1::T,k2::T;dk=(k)->(0.05*k^(-1/3)),multithreaded_matrices::Bool=false,multithreaded_ks=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}

Useful function to check the conditions numbers of the relevant Fredholm matrix and it's derivatives in the given k-range. This is to check the numerical stability of the method, especially very close to a true eigenvalue. It is quite useful to plot the ks vs. log of the returned results vectors for A, dA, ddA to see deeper insights.

# Arguments
- `solver::ExpandedBoundaryIntegralMethod`: The solver configuration for the expanded boundary integral method.
- `billiard::Bi`: The billiard configuration, a subtype of `AbsBilliard`.
- `k1::T`: Starting wavenumber for the spectrum calculation.
- `k2::T`: Ending wavenumber for the spectrum calculation.
- `dk::Function`: Custom function to calculate the wavenumber step size. Defaults to a scaling law inspired by Veble's paper.
- `tol=1e-4`: Tolerance for the overlap_and_merge function that samples a bit outside the merging interval for better results.
- `multithreaded_matrices::Bool=false`: If the Fredholm matrix construction and it's derivatives should be done in parallel.
- `multithreaded_ks::Bool=true`: If the k loop is multithreaded. This is usually the best choice since matrix construction for small k is not as costly.

# Returns
- `(ksA,resultsA)::Tuple{Vector{T},Vector{T}}`: The ks and conditions numbers for the A matrix where LAPACK did not crash.
- `(ksdA,resultsdA)::Tuple{Vector{T},Vector{T}}`: The ks and conditions numbers for the dA matrix where LAPACK did not crash.
- `(ksddA,resultsddA)::Tuple{Vector{T},Vector{T}}`: The ks and conditions numbers for the ddA matrix where LAPACK did not crash.
- `(det_ksA,det_resultsA)::Tuple{Vector{T},Vector{T}}`: The ks and det numbers for the A matrix where LAPACK did not crash.
- `(det_ksdA,det_resultsdA)::Tuple{Vector{T},Vector{T}}`: The ks and det numbers for the dA matrix where LAPACK did not crash.
- `(det_ksddA,det_resultsddA)::Tuple{Vector{T},Vector{T}}`: The ks and det numbers for the ddA matrix where LAPACK did not crash.

"""
function visualize_cond_dA_ddA_vs_k(solver::ExpandedBoundaryIntegralMethod,basis::Ba,billiard::Bi,k1::T,k2::T;dk=(k)->(0.05*k^(-1/3)),multithreaded_matrices::Bool=false,multithreaded_ks=true) where {T<:Real,Ba<:AbstractHankelBasis,Bi<:AbsBilliard}
    basis=AbstractHankelBasis()
    bim_solver=BoundaryIntegralMethod(solver.dim_scaling_factor,solver.pts_scaling_factor,solver.sampler,solver.eps,solver.min_dim,solver.min_pts,solver.rule)
    ks=T[]
    dks=T[]
    k=k1
    while k<k2
        push!(ks,k)
        kstep=dk(k)
        k+=kstep
        push!(dks,kstep)
    end
    println("EBIM...")
    all_pts=Vector{BoundaryPointsBIM{T}}(undef,length(ks))
    @showprogress desc="Calculating boundary points EBIM..." Threads.@threads for i in eachindex(ks)
        all_pts[i]=evaluate_points(deepcopy(bim_solver),billiard,ks[i])
    end
    resultsA=Vector{Union{T,Missing}}(missing,length(ks))
    resultsdA=Vector{Union{T,Missing}}(missing,length(ks))
    resultsddA=Vector{Union{T,Missing}}(missing,length(ks))
    det_resultsA=Vector{Union{T,Missing}}(missing,length(ks))
    det_resultsdA=Vector{Union{T,Missing}}(missing,length(ks))
    det_resultsddA=Vector{Union{T,Missing}}(missing,length(ks))
    p=Progress(length(ks),1) # first one finished
    println("Constructing dA, ddA and evaluating cond...")
    @use_threads multithreading=multithreaded_ks for i in eachindex(ks)
        A,dA,ddA=construct_matrices(solver,basis,all_pts[i],ks[i],multithreaded=multithreaded_matrices)
        try
            cA=cond(A)
            resultsA[i]=cA
        catch e
            @warn "cond(A) failed at k = $(ks[i]) with error $e"
        end
        try
            det_cA=logabsdet(A)[1]
            det_resultsA[i]=det_cA
        catch e
            @warn "logabsdet(A) failed at k = $(ks[i]) with error $e"
        end
        try
            cdA=cond(dA)
            resultsdA[i]=cdA
        catch e 
            @warn "cond(dA) failed at k = $(ks[i]) with error $e"
        end
        try
            det_cdA=logabsdet(dA)[1]
            det_resultsdA[i]=det_cdA
        catch e
            @warn "logabsdet(dA) failed at k = $(ks[i]) with error $e"
        end
        try # since most cases the LAPACK solver will crash when calculating the condition number of ddA. In those cases it is also useless to compute it since we need to divide by ddA in the 2nd order corrections and it will give unstable results.
            cddA=cond(ddA)
            resultsddA[i]=cddA
        catch e 
            @warn "cond(ddA) failed at k = $(ks[i]) with error $e"
        end
        try
            det_cddA=logabsdet(ddA)[1]
            det_resultsddA[i]=det_cddA
        catch e
            @warn "logabsdet(ddA) failed at k = $(ks[i]) with error $e"
        end
        next!(p)
    end
    function filter_valid(xs::Vector{Union{T,Missing}},ks::Vector{T}) where {T}
        idxs=findall(!ismissing,xs)
        return xs[idxs],ks[idxs]
    end
    # Filter condition numbers and their corresponding ks
    resultsA,ksA=filter_valid(resultsA,ks)
    resultsdA,ksdA=filter_valid(resultsdA,ks)
    resultsddA,ksddA=filter_valid(resultsddA,ks)
    # Filter determinants and their corresponding ks
    det_resultsA,det_ksA=filter_valid(det_resultsA,ks)
    det_resultsdA,det_ksdA=filter_valid(det_resultsdA,ks)
    det_resultsddA,det_ksddA=filter_valid(det_resultsddA,ks)
    return (ksA,resultsA),(ksdA,resultsdA),(ksddA,resultsddA),(det_ksA,det_resultsA),(det_ksdA,det_resultsdA),(det_ksddA,det_resultsddA)
end
