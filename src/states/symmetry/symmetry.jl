# extension of BilliardGeometry.jl to support diagonal symmetries

struct DiagonalReflection<:BilliardGeometry.AbsReflection
    parity::Int
end

struct AntiDiagonalReflection<:BilliardGeometry.AbsReflection
    parity::Int
end

DiagonalReflection()=DiagonalReflection(-1)
AntiDiagonalReflection()=AntiDiagonalReflection(-1)

const reflect_diag=CoordinateTransformations.LinearMap(SMatrix{2,2}([0.0 1.0;1.0 0.0]))
const reflect_antidiag=CoordinateTransformations.LinearMap(SMatrix{2,2}([0.0 -1.0;-1.0 0.0]))

@inline symmetry_irrep_character(::Type{T},sym::DiagonalReflection) where {T<:Real}=Complex{T}(sym.parity)
@inline symmetry_irrep_character(::Type{T},sym::AntiDiagonalReflection) where {T<:Real}=Complex{T}(sym.parity)