
using CoordinateTransformations
using StaticArrays

reflect_x=LinearMap(SMatrix{2,2}([-1.0 0.0;0.0 1.0]))
reflect_y=LinearMap(SMatrix{2,2}([1.0 0.0;0.0 -1.0]))

"""
    Reflection

Represents a geometric reflection symmetry.

# Fields
- `sym_map::LinearMap{SMatrix{2,2,Float64,4}}`: The linear map defining the reflection.
- `parity::Union{Int64,Vector{Int64}}`: The parity of the wavefunction under reflection -> is vector for xy - reflection.
- `axis::Symbol`: The axis of reflection (`:x_axis`, `:y_axis`, or `:origin`).
"""
struct Reflection <: AbsSymmetry
    sym_map::LinearMap{SMatrix{2,2,Float64,4}}
    parity::Union{Int64,Vector{Int64}}
    axis::Symbol
end

# X,Y Reflections over potentially shifted axis (depeneding of the shift_x/shift_y of the billiard geometry), where only the coordiante reflection is dependant on the shifts of the reflection axes since the normals are just direction vectors
@inline _x_reflect(x::T,sx::T) where {T<:Real}=(2*sx-x)
@inline _y_reflect(y::T,sy::T) where {T<:Real}=(2*sy-y)
@inline _x_reflect_normal(nx::T,ny::T) where {T<:Real}=(-nx,ny)
@inline _y_reflect_normal(nx::T,ny::T) where {T<:Real}=(nx,-ny)
@inline _xy_reflect_normal(nx::T,ny::T) where {T<:Real}=(-nx,-ny)

"""
    Rotation

Cₙ rotation symmetry specification.

Fields
- n::Int                      # rotation order (e.g. 3,4,…)
- m::Int                      # irrep index in 0:(n-1)
- center::NTuple{2,Float64}   # rotation center (default (0,0))
"""
struct Rotation <: AbsSymmetry
    n::Int
    m::Int
    center::NTuple{2,Float64}
end

# use mod(m,n) so we can wrap m around in case needed
Rotation(n::Int,m::Int;center::Tuple{Real,Real}=(0.0,0.0))=Rotation(n,mod(m,n),(Float64(center[1]),Float64(center[2])))

# rotation of a point (x,y) by an angle α already wrapped in s,c=sincos(α) around a center of rotation (cx,cy). Added a separate K for type of center float.
@inline function _rot_point(x::T,y::T,cx::K,cy::K,c::T,s::T) where {T<:Real,K<:Real}
    xr=cx+c*(x-cx)-s*(y-cy)
    yr=cy+s*(x-cx)+c*(y-cy)
    return xr,yr
end

# rotate (nx,ny) by angle with cos=c, sin=s
@inline _rot_vec(nx::T,ny::T,c::T,s::T) where {T<:Real}=(c*nx-s*ny,s*nx+c*ny)

# tables for cos(lθ), sin(lθ), and characters χ_m(l)=e^{i2π ml/n}. Best place to do it here since it should not be called by user
@inline function _rotation_tables(::Type{T},n::Int,m::Int) where {T<:Real}
    θ=T(TWO_PI)/T(n)
    c1,s1=cos(θ),sin(θ)
    cos_tabulated=Vector{T}(undef,n);sin_tabulated=Vector{T}(undef,n)
    χ=Vector{Complex{T}}(undef,n)
    # for l=0 cos(0)=1 & sin(0)=0, therefore χ(l=0)=1.0
    cos_tabulated[1]=one(T)
    sin_tabulated[1]=zero(T)
    χ[1]=one(Complex{T})
    for l in 2:n
        cl,sl=cos_tabulated[l-1],sin_tabulated[l-1]
        cos_tabulated[l]=cl*c1-sl*s1
        sin_tabulated[l]=sl*c1+cl*s1
    end
    for l in 0:n-1
        χ[l+1]=cis(T(TWO_PI)*T(mod(m,n)*l)/T(n))
    end
    return cos_tabulated,sin_tabulated,χ
end

"""
    XReflection(parity::Int)

Creates a `Reflection` object that reflects over the `y`-axis.

# Arguments
- `parity::Int`: The wavefunction's parity under the reflection.

# Returns
- `Reflection`: A reflection object for the `y`-axis.

"""
function XReflection(parity)
    return Reflection(reflect_x,parity,:y_axis)
end

"""
    YReflection(parity::Int)

Creates a `Reflection` object that reflects over the `x`-axis.

# Arguments
- `parity::Int`: The wavefunction's parity under the reflection.

# Returns
- `Reflection`: A reflection object for the `x`-axis.
"""
function YReflection(parity)
    return Reflection(reflect_y,parity,:x_axis)
end

"""
    XYReflection(parity_x::Int, parity_y::Int)

Creates a `Reflection` object that reflects over both axes (origin symmetry).

# Arguments
- `parity_x::Int`: The wavefunction's parity under reflection over the `y`-axis.
- `parity_y::Int`: The wavefunction's parity under reflection over the `x`-axis.

# Returns
- `Reflection`: A reflection object for the origin symmetry.
"""
function XYReflection(parity_x,parity_y)
    return Reflection(reflect_x∘reflect_y,[parity_x,parity_y],:origin)
end

"""
    reflect_wavefunction(Psi::Matrix, x_grid::Vector, y_grid::Vector, symmetries::Sy; x_axis=0.0, y_axis=0.0) -> (Psi::Matrix, x_grid::Vector, y_grid::Vector)

Applies reflection symmetries to a wavefunction and its grid.

# Arguments
- `Psi::Matrix`: The wavefunction defined on the `x_grid` and `y_grid`.
- `x_grid::Vector`: The x-coordinates of the grid points.
- `y_grid::Vector`: The y-coordinates of the grid points.
- `symmetries::Vector{Any}`: A vector of `Reflection` objects to apply. Should contain only 1 element.
- `x_axis=0.0`: The x-coordinate of the reflection axis. Default is 0.0.
- `y_axis=0.0`: The y-coordinate of the reflection axis. Default is 0.0.

# Returns
- `Psi::Matrix`: The modified wavefunction after applying the reflections.
- `x_grid::Vector`: The updated x-coordinates of the grid points.
- `y_grid::Vector`: The updated y-coordinates of the grid points.
"""
function reflect_wavefunction(Psi,x_grid,y_grid,symmetries;x_axis=0.0,y_axis=0.0)
    x_grid=x_grid.-x_axis  # Shift the grid to move the reflection axis to x=0
    y_grid=y_grid.-y_axis  # Shift the grid to move the reflection axis to y=0
    for sym in symmetries
        if sym.axis==:y_axis
            x= -reverse(x_grid)
            Psi_ref=reverse(sym.parity.*Psi;dims=1)

            Psi=vcat(Psi,Psi_ref)
            x_grid=append!(x,x_grid)
            sorted_indices=sortperm(x_grid)
            x_grid=x_grid[sorted_indices]
        end
        if sym.axis==:x_axis
            y= -reverse(y_grid)
            Psi_ref=reverse(sym.parity.*Psi;dims=2)
            Psi=hcat(Psi_ref,Psi) 
            y_grid=append!(y,y_grid)
            sorted_indices=sortperm(y_grid)
            y_grid=y_grid[sorted_indices]
        end
        if sym.axis==:origin
            # Reflect over both axes (x -> -x, y -> -y)
            # First, reflect over y-axis
            x_reflected= -reverse(x_grid)
            Psi_y_reflected=reverse(sym.parity[1].*Psi;dims=2)
            Psi_y_combined=[Psi_y_reflected Psi]
            x_grid_combined=[x_reflected; x_grid]
            
            # Then, reflect over x-axis
            y_reflected= -reverse(y_grid)
            Psi_x_reflected=reverse(sym.parity[2].*Psi_y_combined;dims=1)
            Psi_x_combined=[Psi_x_reflected; Psi_y_combined]
            y_grid_combined=[y_reflected; y_grid]

            # Permute the indexes
            sorted_indices=sortperm(x_grid_combined)
            x_grid_combined=x_grid_combined[sorted_indices]
            sorted_indices=sortperm(y_grid_combined)
            y_grid_combined=y_grid_combined[sorted_indices]
            
            # Update Psi and grids
            Psi=Psi_x_combined
            x_grid=x_grid_combined
            y_grid=y_grid_combined
        end
    end
    # Shift the grids back to their original positions before returning
    x_grid=x_grid.+x_axis
    y_grid=y_grid.+y_axis
    return Psi,x_grid,y_grid
end

# INTERNAL 
function rotate_wavefunction(Psi_grid::Matrix, x_grid::Vector{T}, y_grid::Vector{T}, rotation::Rotation, billiard::Bi; center::SVector{2, Float64}=SVector{2, Float64}(0.0, 0.0),grid_size::Int = ceil(Int, 0.7*length(x_grid))) where {T<:Real, Bi<:AbsBilliard}
    let n=rotation.n,m=rotation.parity
        new_x_grid,new_y_grid,new_Psi_grid=get_full_area_with_manual_binning(x_grid,y_grid,Psi_grid,billiard,n,m;center=center,grid_size=grid_size)
        return new_Psi_grid,new_x_grid,new_y_grid 
    end
end
