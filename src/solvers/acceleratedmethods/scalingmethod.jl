using LinearAlgebra, StaticArrays
using TimerOutputs
abstract type AbsScalingMethod <: AcceleratedSolver 
end

"""
    ScalingMethodA{T} <: AbsScalingMethod where {T<:Real}

Represents the struct containing the parameters of the Scaling Method by Vergini and Saraceno.

Fields:
- `dim_scaling_factor<:Real`: Scaling factor for the basis used. This one should be around 3.0-4.0 as a start to see how well the tension values are.
- `pts_scaling_factor::Vector{<:Real}`: Scaling factor for the boudnary evaluation points used. If it's a single value, it will be used for all dimensions. Should be minimally 4.0, preferably 5.0 as a start. Best to check how changing this changes the minima of the tension values.
- `sampler::Vector`: A vector of samplers that determine the construction of `BoundaryPointsSM` on the boundary. By default `GaussLegendreNodes` are used to sample the boundary.
- `eps::T`: A tolerance for the solver, by default eps(T), so usually machine precision. Best leave it as is.
- `min_dim::Int64`: The minimum dimension for the basis. This is in a way legacy code that is here for compatibility. But since one does not construct this struct outside it's function constructors just leave it.
- `min_pts::Int64`: The minimum number of points for the boundary evaluation. This is in a way legacy code that is here for compatibility. But since one does not construct this struct outside it's function constructors just leave it.
"""
struct ScalingMethodA{T} <: AbsScalingMethod where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end

"""
    ScalingMethodA(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim = 100, min_pts = 500) where T<:Real 

Constructor for ScalingMethodA struct that takes into account default `GaussLegendreNodes` samplers.

# Arguments
- `dim_scaling_factor<:Real`: Scaling factor for the basis used. This one should be around 3.0-4.0 as a start to see how well the tension values are.
- `pts_scaling_factor::Vector{<:Real}`: Scaling factor for the boudnary evaluation points used. If it's a single value, it will be used for all dimensions. Should be minimally 4.0, preferably 5.0 as a start. Best to check how changing this changes the minima of the tension values.
- `min_dim::Int64`: The minimum dimension for the basis. This is in a way legacy code that is here for compatibility. No need to change.
- `min_pts::Int64`: The minimum number of points for the boundary evaluation. This is in a way legacy code that is here for compatibility. No need to change.

# Returns
- `ScalingMethodA{T}`: A struct with the provided parameters.
"""
function ScalingMethodA(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}};min_dim=100,min_pts=500) where T<:Real 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[GaussLegendreNodes()]
return ScalingMethodA(d,bs,sampler,eps(T),min_dim,min_pts)
end

"""
    ScalingMethodA(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{Sam}; min_dim = 100, min_pts = 500) where {T<:Real, Sam<:AbsSampler} 

Constructor for ScalingMethodA struct that takes into account provided samplers.

# Arguments
- `dim_scaling_factor<:Real`: Scaling factor for the basis used. This one should be around 3.0-4.0 as a start to see how well the tension values are.
- `pts_scaling_factor::Vector{<:Real}`: Scaling factor for the boudnary evaluation points used. If it's a single value, it will be used for all dimensions. Should be minimally 4.0, preferably 5.0 as a start. Best to check how changing this changes the minima of the tension values.
- `samplers::Vector{Sam}`: A vector of samplers that determine the construction of `BoundaryPointsSM` on the boundary. Check samplers.jl too see the list to choose.
- `min_dim::Int64`: The minimum dimension for the basis. This is in a way legacy code that is here for compatibility. No need to change.
- `min_pts::Int64`: The minimum number of points for the boundary evaluation. This is in a way legacy code that is here for compatibility. No need to change.

# Returns
- `ScalingMethodA{T}`: A struct with the provided parameters.
"""
function ScalingMethodA(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}},samplers::Vector{Sam};min_dim=100,min_pts=500) where {T<:Real,Sam<:AbsSampler} 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return ScalingMethodA(d,bs,samplers,eps(T),min_dim,min_pts)
end

# UNUSED FOR NOW - HAS NO FUNCTION
struct ScalingMethodB{T} <: AbsScalingMethod where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end

# UNUSED FOR NOW - HAS NO FUNCTION
function ScalingMethodB(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}};min_dim=100,min_pts=500) where T<:Real 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[GaussLegendreNodes()]
return ScalingMethodB(d,bs,sampler,eps(T),min_dim,min_pts)
end

# UNUSED FOR NOW - HAS NO FUNCTION
function ScalingMethodB(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}},samplers::Vector{Sam};min_dim=100,min_pts=500) where {T<:Real,Sam<:AbsSampler} 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return ScalingMethodB(d,bs,samplers,eps(T),min_dim,min_pts)
end

"""
    BoundaryPointsSM{T} <: AbsPoints where {T<:Real}

Represents the boundary points and their information that is neccesery to construct the matrices required in the Scaling Method.

# Fields
- `xy::Vector{SVector{2,T}}`: The coordinates of the boundary points.
- `w::Vector{T}`: The weights of the boundary points. To be used for the weight matrix construction for F and Fk.
"""
struct BoundaryPointsSM{T} <: AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}}
    w::Vector{T}
end

"""
    evaluate_points(solver::AbsScalingMethod, billiard::Bi, k<:Real) where {Bi<:AbsBilliard}

Evaluate points on the boundary of the `billiard` at the given wavenumber `k` which determines the density of points (and the `sampler` hidden in `solver` that distributes them).

# Arguments
- `solver::AbsScalingMethod`: The solver for which the points are evaluated.
- `billiard::Bi`: The billiard on which the points are evaluated.
- `k<:Real`: The wavenumber at which the points are evaluated.

# Returns
- `BoundaryPointsSM{T}`: A struct containing the evaluated points and their weights (all the information needed for the `ScalingMethod`).
"""
function evaluate_points(solver::AbsScalingMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=billiard.fundamental_boundary
    type=eltype(solver.pts_scaling_factor)
    xy_all=Vector{SVector{2,type}}()
    w_all=Vector{type}()
    for i in eachindex(curves)
        crv=curves[i]
        if typeof(crv)<:AbsRealCurve
            L=crv.length
            N=max(solver.min_pts,round(Int,k*L*bs[i]/(2*pi)))
            sampler=samplers[i]
            if crv isa PolarSegment
                if sampler isa PolarSampler
                    t,dt=sample_points(sampler,crv,N)
                else
                    t,dt=sample_points(sampler,N)
                end
                s=arc_length(crv,t)
                ds=diff(s)
                append!(ds,L+s[1]-s[end]) # add the last difference as we have 1 less element. Add L to s[1] so we can logically subtract s[end]
            else
                t,dt=sample_points(sampler,N)
                ds=L.*dt
            end
            xy=curve(crv,t)
            normal=normal_vec(crv,t)
            rn=dot.(xy,normal)
            w=ds./rn
            append!(xy_all,xy)
            append!(w_all,w)
        end
    end
    return BoundaryPointsSM{type}(xy_all, w_all)
end

"""
    construct_matrices_benchmark(solver::ScalingMethodA, basis::Ba, pts::BoundaryPointsSM, k<:Real) where {Ba<:AbsBasis}

Benchmarks the construction of all the matrices needeed for the Scaling Method for a given reference k wavenumber. We need to construct from the solver a matrix that evaluates the basis function on the boundary, apply the weights to it and then multiplies it with the basis matrix transpose. This is detailed in section 6 of Barnett8s thesis: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html. To highlight:

1.) `F=G'*W*G` where `G` is the basis matrix and `W` the weight matrix (that contains the dot products of the radial distance of the point with it's normal derivative directionald derivative)

2.) `Fk=dF/dk=(dG/dk)*W*G + it's transpose`. Both are real Symmetric matrices so under the hood a call to `sygvd` should be made but not neccesery in the end due to numerical nullspace deletion and therefore invertability of the B matrix in Au=λBu -> transpose/inv(B)Au=λu...

With these we now have the neccesery matrices foe the Scaling Method `Fk*u+λF*u=0 <-> eigen(Fk,F) ?-> reduction of numerical nullspace -> ... ` (more in decompositions.jl)

# Arguments
- `solver::ScalingMethodA`: The solver for which the matrices are constructed. Redundant information, used b/c other methods have functions with same signatures and multiple dispatches are useful.
- `basis::Ba`: The basis on which the matrices are constructed.
- `pts::BoundaryPointsSM`: The points on the boundary on which the matrices are constructed.

# Returns
- `F::Matrix{<:Real}`: The matrix that evaluates the basis function on the boundary and applies the weights.
- `Fk::Matrix{<:Real}`: The matrix that evaluates the derivative of the basis function on the boundary and applies the weights.
"""
function construct_matrices_benchmark(solver::ScalingMethodA,basis::Ba,pts::BoundaryPointsSM,k) where {Ba<:AbsBasis}
    to=TimerOutput()
    symmetries=basis.symmetries 
    xy,w=pts.xy,pts.w
    if ~isnothing(symmetries)
        n=(length(symmetries)+1.0)
        w=w.*n
    end
    N=basis.dim
    #basis matrix
    @timeit to "basis_matrix" B=basis_matrix(basis,k,xy)
    type=eltype(B)
    F=zeros(type,(N,N))
    Fk=similar(F)
    @timeit to "F construction" begin 
        @timeit to "weights" T=(w.*B) #reused later
        @timeit to "product" mul!(F,B',T) #boundary norm matrix
    end
    #reuse B
    @timeit to "dk_matrix" B=dk_matrix(basis,k,xy)
    @timeit to "Fk construction" begin 
        @timeit to "product" mul!(Fk,B',T) #B is now derivative matrix
        #symmetrize matrix
        @timeit to "addition" Fk=Fk+Fk'
    end
    print_timer(to)    
    return F,Fk        
end

"""
    construct_matrices(solver::ScalingMethodA, basis::Ba, pts::BoundaryPointsSM, k) where {Ba<:AbsBasis}

Constructs all the matrices needeed for the Scaling Method for a given reference k wavenumber. We need to construct from the solver a matrix that evaluates the basis function on the boundary, apply the weights to it and then multiplies it with the basis matrix transpose. This is detailed in section 6 of Barnett8s thesis: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html. To highlight:

1.) `F=G'*W*G` where `G` is the basis matrix and `W` the weight matrix (that contains the dot products of the radial distance of the point with it's normal derivative directionald derivative)

2.) `Fk=dF/dk=(dG/dk)*W*G + it's transpose`. Both are real Symmetric matrices so under the hood a call to `sygvd` should be made but not neccesery in the end due to numerical nullspace deletion and therefore invertability of the B matrix in Au=λBu -> transpose/inv(B)Au=λu...

With these we now have the neccesery matrices for the Scaling Method `Fk*u+λF*u=0 <-> eigen(Fk,F) ?-> reduction of numerical nullspace -> ... ` (more in decompositions.jl)

# Arguments
- `solver::ScalingMethodA`: The solver for which the matrices are constructed. Redundant information, used b/c other methods have functions with same signatures and multiple dispatches are useful.
- `basis::Ba`: The basis on which the matrices are constructed.
- `pts::BoundaryPointsSM`: The points on the boundary on which the matrices are constructed.

# Returns
- `F::Matrix{<:Real}`: The matrix that evaluates the basis function on the boundary and applies the weights.
- `Fk::Matrix{<:Real}`: The matrix that evaluates the derivative of the basis function on the boundary and applies the weights.
"""
function construct_matrices(solver::ScalingMethodA,basis::Ba,pts::BoundaryPointsSM,k) where {Ba<:AbsBasis}
    xy=pts.xy
    w=pts.w
    symmetries=basis.symmetries
    if ~isnothing(symmetries)
        n=(length(symmetries)+1.0)
        w=w.*n
    end
    N=basis.dim
    #basis matrix
    B=basis_matrix(basis,k,xy)
    type=eltype(B)
    F=zeros(type,(N,N))
    Fk=similar(F)
    T=(w.*B) #reused later
    mul!(F,B',T) #boundary norm matrix
    #reuse B
    B=dk_matrix(basis,k,xy)
    mul!(Fk,B',T) #B is now derivative matrix
    #symmetrize matrix
    Fk=Fk+Fk' 
    return F,Fk    
end

# UNUSED FOR NOW - HAS NO FUNCTION
function construct_matrices_benchmark(solver::ScalingMethodB,basis::Ba,pts::BoundaryPointsSM,k) where {Ba<:AbsBasis}
    to=TimerOutput()
    xy,w=pts.xy,pts.w
    symmetries=basis.symmetries
    if ~isnothing(symmetries)
        n=(length(symmetries)+1.0)
        w=w.*n
    end
    #basis and gradient matrices
    @timeit to "basis_and_gradient_matrices" B,dX,dY=basis_and_gradient_matrices(basis,k,xy)
    N=basis.dim
    type=eltype(B)
    F=zeros(type,(N,N))
    Fk=similar(F)
    @timeit to "F construction" begin 
        @timeit to "weights" T=(w.*B) #reused later
        @timeit to "product" mul!(F,B',T) #boundary norm matrix
    end
    #reuse B
    @timeit to "Fk construction" begin 
        @timeit to "dilation derivative" x=getindex.(xy,1)
        @timeit to "dilation derivative" y=getindex.(xy,2)
        #inplace modifications
        @timeit to "dilation derivative" dX=x.*dX 
        @timeit to "dilation derivative" dY=y.*dY
        #reuse B
        @timeit to "dilation derivative" B=dX.+dY
        @timeit to "product" mul!(Fk,B',T) #B is now derivative matrix
        #symmetrize matrix
        @timeit to "addition" Fk=(Fk+Fk')./k
    end
    print_timer(to)    
    return F,Fk        
end

# UNUSED FOR NOW - HAS NO FUNCTION
function construct_matrices(solver::ScalingMethodB,basis::Ba,pts::BoundaryPointsSM,k) where {Ba<:AbsBasis}
    xy=pts.xy
    w=pts.w
    symmetries=basis.symmetries
    if ~isnothing(symmetries)
        n=(length(symmetries)+1.0)
        w=w.*n
    end
    N=basis.dim
    #basis matrix
    B,dX,dY=basis_and_gradient_matrices(basis,k,pts.xy)
    type=eltype(B)
    F=zeros(type,(N,N))
    Fk=similar(F)
    T=(w.*B) #reused later
    mul!(F,B',T) #boundary norm matrix
    x=getindex.(xy,1)
    y=getindex.(xy,2)
    #inplace modifications
    dX=x.*dX 
    dY=y.*dY
    #reuse B
    B=dX.+dY
    mul!(Fk,B',T) #B is now derivative matrix
    #symmetrize matrix
    Fk=(Fk+Fk')./k
    return F,Fk    
end

"""
    sm_results(mu<:Union{Vector{<:Real},<:Real},k<:Real)

Construct the Scaling method results (the name means that) from the given generalized eigenvalues mu obtained from solving the generalized_eigen in decompositions.jl. Following Barnett's derivation: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html. They need to be processed wrt. the reference wavenumber k that constructed the matrices that were for the generalized_eigen.

# Arguments
- `mu<:Vector{<:Real}`: The generalized eigenvalues obtained from solving the generalized_eigen function.
- `k<:Real`: The reference wavenumber that defined the generalized eigenproblem.
"""
function sm_results(mu,k)
    ks = k .- 2 ./mu .+ 2/k ./(mu.^2) 
    ten = 2.0 .*(2.0 ./ mu).^2
    return ks, ten
end

"""
    solve(solver::AbsScalingMethod, basis::Ba, pts::BoundaryPointsSM, k<:Real, dk<:Real) where {Ba<:AbsBasis}

For a given reference wavenumber k solves the generalized eigenproblem (internally calculates `generalized_eigvals` since we do not require the `X` matrix of coefficients for the wavefunction construction). The `dk` is the interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers). The `dk` is determined empirically for a given k range and the specific geometry (and possibly basis).

# Arguments
- `solver::AbsScalingMethod`: The scaling method to be used.
- `basis::Ba`: The basis function.
- `pts::BoundaryPointsSM`: The boundary points.
- `k<:Real`: The reference wavenumber.
- `dk<:Real`: The interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers).

# Returns
- `ks::Vector{<:Real}`: The computed real wavenumbers.
- `ten::Vector{<:Real}`: The corresponding tensions.
"""
function solve(solver::AbsScalingMethod,basis::Ba,pts::BoundaryPointsSM,k,dk) where {Ba<:AbsBasis}
    F,Fk=construct_matrices(solver,basis,pts,k)
    mu=generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks,ten=sm_results(mu,k)
    idx=abs.(ks.-k).<dk
    ks=ks[idx]
    ten=ten[idx]
    p=sortperm(ks)
    return ks[p],ten[p]
end

"""
    solve(solver::AbsScalingMethod, basis::Ba, pts::BoundaryPointsSM, k<:Real, dk<:Real) where {Ba<:AbsBasis}

Variant of solve where the F and Fk matrices are already constructed.
For a given reference wavenumber k solves the generalized eigenproblem (internally calculates `generalized_eigvals` since we do not require the `X` matrix of coefficients for the wavefunction construction). The `dk` is the interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers). The `dk` is determined empirically for a given k range and the specific geometry (and possibly basis).

# Arguments
- `solver::AbsScalingMethod`: The scaling method to be used.
- `F::Matrix{<:Real}`: The weighted boundary basis matrix (check the `contruct_matrices` for more information)
- `Fk::Matrix{<:Real}`: The weighted derivative basis matrix (check the `construct_matrices` for more information)
- `k<:Real`: The reference wavenumber.
- `dk<:Real`: The interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers).

# Returns
- `ks::Vector{<:Real}`: The computed real wavenumbers.
- `ten::Vector{<:Real}`: The corresponding tensions.
"""
function solve(solver::AbsScalingMethod,F,Fk,k,dk)
    mu=generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks,ten=sm_results(mu,k)
    idx=abs.(ks.-k).<dk
    ks=ks[idx]
    ten=ten[idx]
    p=sortperm(ks)
    return ks[p],ten[p]
end

"""
    solve_vectors(solver::AbsScalingMethod, basis::Ba, pts::BoundaryPointsSM, k<:Real, dk<:Real) where {Ba<:AbsBasis}

Solver in a given `dk` interval with reference wavenumber `k` the corresponding correct wavenumbers, their tensions and the `X` matrix that contains information about the basis expansion coefficients (every column of `X` corresponds to a vector of coefficents that construct the wavefunction for the same index eigenvalue in `ks`). Internally it calls `generalized_eigen` since this also returns/constructs the `X` matrix and not just the generalized eigenvalues.

# Arguments
- `solver::AbsScalingMethod`: The scaling method to be used.
- `basis::Ba`: The basis function.
- `pts::BoundaryPointsSM`: The boundary points.
- `k<:Real`: The reference wavenumber.
- `dk<:Real`: The interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers).

# Returns
- `ks::Vector{<:Real}`: The computed real wavenumbers.
- `ten::Vector{<:Real}`: The corresponding tensions.
- `X::Matrix{<:Real}`: The X matrix that contains information about the basis expansion coefficients (`X[:,i] <-> ks[i]`, check function desc.).
"""
function solve_vectors(solver::AbsScalingMethod,basis::Ba,pts::BoundaryPointsSM,k,dk) where {Ba<:AbsBasis}
    F,Fk=construct_matrices(solver,basis,pts,k)
    mu,Z,C=generalized_eigen(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks,ten=sm_results(mu,k)
    idx=abs.(ks.-k).<dk
    ks=ks[idx]
    ten=ten[idx]
    Z=Z[:,idx]
    X=C*Z #transform into original basis 
    X=(sqrt.(ten))' .* X # Use the automatic normalization via tension values as described in Barnett's thesis
    p=sortperm(ks)
    return  ks[p],ten[p],X[:,p]
end

