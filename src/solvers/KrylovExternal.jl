
function _solve_krylov(solver::BoundaryIntegralMethod,basis::Ba,pts::BoundaryPointsBIM,k;kernel_fun::Union{Symbol,Function}=:default,multithreaded::Bool=true) where {Ba<:AbstractHankelBasis}
    A=construct_matrices(solver,basis,pts,k;kernel_fun=kernel_fun,multithreaded=multithreaded)
    vals,_,_,_=svdsolve(A,1,:SR)
    return vals[1]
end
