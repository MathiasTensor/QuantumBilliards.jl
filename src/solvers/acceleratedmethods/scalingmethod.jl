abstract type AbsScalingMethod <: AcceleratedSolver end

"""
    VerginiSaraceno{T} <: AbsScalingMethod where {T<:Real}

Represents the struct containing the parameters of the Scaling Method by Vergini and Saraceno.

Fields:
- `dim_scaling_factor<:Real`: Scaling factor for the basis used. This one should be around 3.0-4.0 as a start to see how well the tension values are.
- `pts_scaling_factor::Vector{<:Real}`: Scaling factor for the boudnary evaluation points used. If it's a single value, it will be used for all dimensions. Should be minimally 4.0, preferably 5.0 as a start. Best to check how changing this changes the minima of the tension values.
- `sampler::Vector`: A vector of samplers that determine the construction of `BoundaryPoints` on the boundary. By default `GaussLegendreNodes` are used to sample the boundary.
- `eps::T`: A tolerance for the solver, by default eps(T), so usually machine precision. Best leave it as is.
- `min_dim::Int64`: The minimum dimension for the basis. This is in a way legacy code that is here for compatibility. But since one does not construct this struct outside it's function constructors just leave it.
- `min_pts::Int64`: The minimum number of points for the boundary evaluation. This is in a way legacy code that is here for compatibility. But since one does not construct this struct outside it's function constructors just leave it.
"""
struct VerginiSaraceno{T} <: AbsScalingMethod where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end

"""
    VerginiSaraceno(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim = 100, min_pts = 500) where T<:Real 

Constructor for VerginiSaraceno struct that takes into account default `GaussLegendreNodes` samplers.

# Arguments
- `dim_scaling_factor<:Real`: Scaling factor for the basis used. This one should be around 3.0-4.0 as a start to see how well the tension values are.
- `pts_scaling_factor::Vector{<:Real}`: Scaling factor for the boudnary evaluation points used. If it's a single value, it will be used for all dimensions. Should be minimally 4.0, preferably 5.0 as a start. Best to check how changing this changes the minima of the tension values.
- `min_dim::Int64`: The minimum dimension for the basis. This is in a way legacy code that is here for compatibility. No need to change.
- `min_pts::Int64`: The minimum number of points for the boundary evaluation. This is in a way legacy code that is here for compatibility. No need to change.

# Returns
- `VerginiSaraceno{T}`: A struct with the provided parameters.
"""
function VerginiSaraceno(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}};min_dim=100,min_pts=500) where T<:Real 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    sampler=[GaussLegendreNodes()]
return VerginiSaraceno(d,bs,sampler,eps(T),min_dim,min_pts)
end

"""
    VerginiSaraceno(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{Sam}; min_dim = 100, min_pts = 500) where {T<:Real, Sam<:AbsSampler} 

Constructor for VerginiSaraceno struct that takes into account provided samplers.

# Arguments
- `dim_scaling_factor<:Real`: Scaling factor for the basis used. This one should be around 3.0-4.0 as a start to see how well the tension values are.
- `pts_scaling_factor::Vector{<:Real}`: Scaling factor for the boudnary evaluation points used. If it's a single value, it will be used for all dimensions. Should be minimally 4.0, preferably 5.0 as a start. Best to check how changing this changes the minima of the tension values.
- `samplers::Vector{Sam}`: A vector of samplers that determine the construction of `BoundaryPoints` on the boundary. Check samplers.jl too see the list to choose.
- `min_dim::Int64`: The minimum dimension for the basis. This is in a way legacy code that is here for compatibility. No need to change.
- `min_pts::Int64`: The minimum number of points for the boundary evaluation. This is in a way legacy code that is here for compatibility. No need to change.

# Returns
- `VerginiSaraceno{T}`: A struct with the provided parameters.
"""
function VerginiSaraceno(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}},samplers::Vector{Sam};min_dim=100,min_pts=500) where {T<:Real,Sam<:AbsSampler} 
    d=dim_scaling_factor
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor] : pts_scaling_factor
    return VerginiSaraceno(d,bs,samplers,eps(T),min_dim,min_pts)
end

"""
    evaluate_points(solver::AbsScalingMethod, billiard::Bi, k<:Real) where {Bi<:AbsBilliard}

Evaluate points on the boundary of the `billiard` at the given wavenumber `k` which determines the density of points (and the `sampler` hidden in `solver` that distributes them).

# Arguments
- `solver::AbsScalingMethod`: The solver for which the points are evaluated.
- `billiard::Bi`: The billiard on which the points are evaluated.
- `k<:Real`: The wavenumber at which the points are evaluated.

# Returns
- `BoundaryPoints{T}`: A struct containing the evaluated points and their weights (all the information needed for the `ScalingMethod`).
"""
function evaluate_points(solver::AbsScalingMethod,billiard::Bi,k) where {Bi<:AbsBilliard}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=billiard.fundamental_boundary #TODO get curves
    type=eltype(solver.pts_scaling_factor)
    xy_all=Vector{SVector{2,type}}()
    w_all=Vector{type}()
    ds_all=Vector{type}()
    normal_all=Vector{SVector{2,type}}()
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
            normal=normal_vec(crv,t) # TODO domain_gradient_vector
            rn=dot.(xy,normal)
            w=ds./rn
            append!(xy_all,xy)
            append!(w_all,w)
            append!(ds_all,ds)
            append!(normal_all,normal)
        end
    end
    return BoundaryPoints{type}(xy_all,normal_all,Vector{type}(),ds_all,w_all,Vector{type}(),Vector{type}(),Vector{SVector{2,type}}(),zero(type),zero(type))
end

"""
    construct_matrices_benchmark(solver::VerginiSaraceno,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Benchmark the construction of F = G'*(W*G) and Fk = (dG/dk)'*(W*G) + (G'*(W*dG))
using the same algorithm as `construct_matrices` (BLAS syr!/syr2k style; no W*G temp).

Returns (F, Fk) and prints a detailed TimerOutput.
"""
function construct_matrices_benchmark(solver::VerginiSaraceno,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    t0=time()
    xy=pts.xy; w=pts.w; N=basis.dim
    nsym=isnothing(basis.symmetries) ? one(eltype(w)) : 2*one(eltype(w))
    @info "construct_matrices_new" k=k N=N M=length(xy) nsym=nsym blas_threads=BLAS.get_num_threads()
    t=time()
    @blas_1 G=basis_matrix(basis,k,xy;multithreaded)
    @blas_1 dG=dk_matrix(basis,k,xy;multithreaded)
    @info "basis_matrix & dk_matrix" elapsed=(time()-t) sizeG=size(G) sizedG=size(dG)
    t=time()
    _scale_rows_sqrtw!(G,w,nsym)
    F=Matrix{eltype(G)}(undef,N,N)
    @blas_multi MAX_BLAS_THREADS BLAS.syrk!('U','T',one(eltype(G)),G,zero(eltype(G)),F)
    _symmetrize_from_upper!(F)
    @info "F build (√W + SYRK)" elapsed=(time()-t) issym=issymmetric(F)
    t=time()
    _scale_rows_sqrtw!(dG,w,nsym)
    Fk=Matrix{eltype(G)}(undef,N,N)
    @blas_multi_then_1 MAX_BLAS_THREADS BLAS.syr2k!('U','T',one(eltype(G)),G,dG,zero(eltype(G)),Fk)
    _symmetrize_from_upper!(Fk)
    @info "Fk build (√W + SYR2K)" elapsed=(time()-t) issym=issymmetric(Fk)
    @info "total construct_matrices_new" elapsed=(time()-t0)
    return F,Fk
end

"""
    construct_matrices(solver::VerginiSaraceno,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbsBasis}

Constructs all the matrices needeed for the Scaling Method for a given reference k wavenumber. We need to construct from the solver a matrix that evaluates the basis function on the boundary, apply the weights to it and then multiplies it with the basis matrix transpose. This is detailed in section 6 of Barnett8s thesis: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html.

# Arguments
- `solver::VerginiSaraceno`: The solver for which the matrices are constructed. Redundant information, used b/c other methods have functions with same signatures and multiple dispatches are useful.
- `basis::Ba`: The basis on which the matrices are constructed.
- `pts::BoundaryPoints`: The points on the boundary on which the matrices are constructed.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.

# Returns
- `F::Symmetric{<:Real}`: The matrix that evaluates the basis function on the boundary and applies the weights.
- `Fk::Symmetric{<:Real}`: The matrix that evaluates the derivative of the basis function on the boundary and applies the weights.
"""
function construct_matrices(solver::VerginiSaraceno,basis::Ba,pts::BoundaryPoints,k;multithreaded::Bool=true) where {Ba<:AbsBasis}
    xy=pts.xy
    w=pts.w
    N=basis.dim                                 
    nsym=isnothing(basis.symmetries) ? one(eltype(w)) : 2*one(eltype(w))  # symmetry multiplier
    @blas_1 G=basis_matrix(basis,k,xy;multithreaded) # G is the unweighted basis matrix (M×N)
    @blas_1 dG=dk_matrix(basis,k,xy;multithreaded) # dG si the unweighted k derivative ∂G/∂k (M×N)
    _scale_rows_sqrtw!(G,w,nsym) # G <- sqrt(nsym*w) .* G  , inplace row scaling, this is to use BLAS.syrk! trick because W is Diagonal: F = G'*(W*G) = (sqrt(W)*G)'*(sqrt(W)*G)
    F=Matrix{eltype(G)}(undef,N,N) # F: need to allocate N×N real matrix
    @blas_multi MAX_BLAS_THREADS BLAS.syrk!('U','T',one(eltype(G)),G,zero(eltype(G)),F) # F[u ∈ UpperTriangular] = G' * G SYRK, where G is now sqrt(nsym*w).*G
    _symmetrize_from_upper!(F) # fill the bottom part of F by mirroring the upper part
    _scale_rows_sqrtw!(dG,w,nsym) # same trick as with F: dG <. sqrt(nsym*w) .* dG
    Fk=Matrix{eltype(G)}(undef,N,N) # again need to allocate N×N real matrix
    @blas_multi_then_1 MAX_BLAS_THREADS BLAS.syr2k!('U','T',one(eltype(G)),G,dG,zero(eltype(G)),Fk) # Fk[U] = G'*dG + dG'*G  (SYR2K), this is Fk = (dG'*(W*G)) + (G'*(W*dG)) looking at original variable names
    _symmetrize_from_upper!(Fk) # again mirror the upper part to the bottom part
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
    solve(solver::AbsScalingMethod,basis::Ba,pts::BoundaryPoints,k,dk;multithreaded::Bool=true) where {Ba<:AbsBasis}

For a given reference wavenumber k solves the generalized eigenproblem (internally calculates `generalized_eigvals` since we do not require the `X` matrix of coefficients for the wavefunction construction). The `dk` is the interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers). The `dk` is determined empirically for a given k range and the specific geometry (and possibly basis).

# Arguments
- `solver::AbsScalingMethod`: The scaling method to be used.
- `basis::Ba`: The basis function.
- `pts::BoundaryPoints`: The boundary points.
- `k<:Real`: The reference wavenumber.
- `dk<:Real`: The interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
- `ks::Vector{<:Real}`: The computed real wavenumbers.
- `ten::Vector{<:Real}`: The corresponding tensions.
"""
function solve(solver::AbsScalingMethod,basis::Ba,pts::BoundaryPoints,k,dk;multithreaded::Bool=true,cholesky::Bool=false) where {Ba<:AbsBasis}
    F,Fk=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    if cholesky
        @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eig_cholesky(Symmetric(F),Symmetric(Fk);eps_rank=solver.eps)
    else
        @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps) 
    end
    ks,ten=sm_results(mu,k)
    idx=abs.(ks.-k).<dk
    ks=ks[idx]
    ten=ten[idx]
    p=sortperm(ks)
    return ks[p],ten[p]
end

"""
    solve(solver::AbsScalingMethod, basis::Ba, pts::BoundaryPoints, k<:Real, dk<:Real) where {Ba<:AbsBasis}

Variant of solve where the F and Fk matrices are already constructed.
For a given reference wavenumber k solves the generalized eigenproblem (internally calculates `generalized_eigvals` since we do not require the `X` matrix of coefficients for the wavefunction construction). The `dk` is the interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers). The `dk` is determined empirically for a given k range and the specific geometry (and possibly basis).

# Arguments
- `solver::AbsScalingMethod`: The scaling method to be used.
- `F::Matrix{<:Real}`: The weighted boundary basis matrix (check the `contruct_matrices` for more information)
- `Fk::Matrix{<:Real}`: The weighted derivative basis matrix (check the `construct_matrices` for more information)
- `k<:Real`: The reference wavenumber.
- `dk<:Real`: The interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
- `ks::Vector{<:Real}`: The computed real wavenumbers.
- `ten::Vector{<:Real}`: The corresponding tensions.
"""
function solve(solver::AbsScalingMethod,F,Fk,k,dk;cholesky::Bool=false)
    if cholesky
        @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eig_cholesky(Symmetric(F),Symmetric(Fk);eps_rank=solver.eps)
    else
        @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    end
    ks,ten=sm_results(mu,k)
    idx=abs.(ks.-k).<dk
    ks=ks[idx]
    ten=ten[idx]
    p=sortperm(ks)
    return ks[p],ten[p]
end

"""
    solve_vectors(solver::AbsScalingMethod,basis::Ba,pts::BoundaryPoints,k,dk;multithreaded::Bool=true) where {Ba<:AbsBasis}

Solver in a given `dk` interval with reference wavenumber `k` the corresponding correct wavenumbers, their tensions and the `X` matrix that contains information about the basis expansion coefficients (every column of `X` corresponds to a vector of coefficents that construct the wavefunction for the same index eigenvalue in `ks`). Internally it calls `generalized_eigen` since this also returns/constructs the `X` matrix and not just the generalized eigenvalues.

# Arguments
- `solver::AbsScalingMethod`: The scaling method to be used.
- `basis::Ba`: The basis function.
- `pts::BoundaryPoints`: The boundary points.
- `k<:Real`: The reference wavenumber.
- `dk<:Real`: The interval for which we consider the computed `ks` to be valid solutions (correct wavenumbers).
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
- `ks::Vector{<:Real}`: The computed real wavenumbers.
- `ten::Vector{<:Real}`: The corresponding tensions.
- `X::Matrix{<:Real}`: The X matrix that contains information about the basis expansion coefficients (`X[:,i] <-> ks[i]`, check function desc.).
"""
function solve_vectors(solver::AbsScalingMethod,basis::Ba,pts::BoundaryPoints,k,dk;multithreaded::Bool=true,cholesky::Bool=false) where {Ba<:AbsBasis}
    F,Fk=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    if cholesky
        @blas_multi_then_1 MAX_BLAS_THREADS mu,Z,C=generalized_eig_cholesky(Symmetric(F),Symmetric(Fk);eps_rank=solver.eps,want_vecs=true)
    else
        @blas_multi_then_1 MAX_BLAS_THREADS mu,Z,C=generalized_eigen(Symmetric(F),Symmetric(Fk);eps=solver.eps) # pure BLAS
    end
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

function solve_spectrum_with_INFO(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk;multithreaded::Bool=true)
    start_init=time()
    L=billiard.length
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    @info "Basis resizing..."
    @time basis_new=resize_basis(basis,billiard,dim,k)
    @info "Pts on boundary evaluation..."
    s_pts=time()
    @time pts=evaluate_points(solver,billiard, k)
    e_pts=time()
    @info "F & dF/dk matrix construction..."
    s_con=time()
    @time F,Fk=construct_matrices(solver,basis_new,pts,k;multithreaded=multithreaded)
    e_con=time()
    @info "F & dF/dk dims: $(size(F))"
    start1=time()
    @warn "Initial condition num. F before regularization: $(cond(F))"
    @warn "Initial condition num. dF/dk before regularization: $(cond(Fk))"
    end1=time()
    A=Symmetric(F)
    B=Symmetric(Fk)
    @info "Removing numerical nullspace of ill conditioned F and eigenvalue problem..."
    s_reg=time()
    @time d,S=eigen(Symmetric(A))
    e_reg=time()
    @info "Smallest & Largest eigval: $(extrema(d))"
    @info "Nullspace removal with criteria eigval < $(solver.eps*maximum(d))"
    idx=d.>solver.eps*maximum(d)
    @info "Dim of num Nullspace: $(count(!,idx))" # counts the number of falses = dim of nullspace
    q=1.0./sqrt.(d[idx])
    C=@view S[:,idx]
    C_scaled=C.*q'
    n=size(C_scaled,2)
    tmp=Matrix{eltype(B)}(undef,size(B,1),n)
    E=Matrix{eltype(B)}(undef,n,n)
    mul!(tmp,B,C_scaled)
    mul!(E,C_scaled',tmp)
    start2=time()
    @warn "Final eigenvalue problem with new condition number: $(cond(E)) and reduced dimension $(size(E))"
    end2=time()
    s_fin=time()
    @time mu,Z=eigen(Symmetric(E))
    e_fin=time()
    ks,ten=sm_results(mu,k)
    idx=abs.(ks.-k).<dk
    ks=ks[idx]
    ten=ten[idx]
    p=sortperm(ks)
    ks,ten=ks[p],ten[p]
    end_init=time()
    total_time=end_init-start_init-(end2-start2)-(end1-start1)
    @info "Final computation time without extrema of SVD for cond calculation: $(total_time) s"
    println("%%%%% SUMMARY %%%%%")
    println("Percentage of total time (most relevant ones): ")
    println("Boundary Pts evaluation: $(100*(e_pts-s_pts)/total_time) %")
    println("F & dF/dk construction: $(100*(e_con-s_con)/total_time) %")
    println("Nullspace removal: $(100*(e_reg-s_reg)/total_time) %")
    println("Final eigen problem: $(100*(e_fin-s_fin)/total_time) %")
    println("%%%%%%%%%%%%%%%%%%%")
    return ks,ten
end

"""
plot_Z!(f::Figure, Z::AbstractMatrix; title::AbstractString="")

Plots the heatmap of a given matrix `Z` which is either VerginiSaraceno's F or dF/dk matrix.
This way one can check if the basis size is large enough - the red lines will show where 
one makes the cut for numerical nullspace. If the red lines are far away to the edges of the matrix
then the matrix size can be reduced. If they are up to the edges, then the matrix size is not large enough and should be increased. 

# Arguments
- `f::Figure`: A `Figure` object where the heatmap will be plotted.
- `Z::Matrix`: A matrix representing the data to be visualized.
- `title::String`: A string specifying the title of the plot (default: "").
- `epsilon::Float64`: A threshold below which matrix entries are treated as NaN (default: 1e-14).

# Details
- Entries in `Z` below the specified `epsilon` are treated as NaN and ignored in the plot.
- The color range is automatically balanced around the maximum absolute value in `Z`.
- A color bar is displayed alongside the plot to indicate the data scale.
"""
function plot_Z!(f::Figure,i::Integer,j::Integer,Z::Matrix;title::String="",epsilon=1e-14)
    Z=deepcopy(Z)
    ax=Axis(f[i,j][1,1];title=title)
    m=findmax(abs.(Z))[1]
    Z[abs.(Z).<epsilon].=NaN
    nan_row=findfirst(row->all(isnan,Z[row,:]),axes(Z,1))
    nan_col=findfirst(col->all(isnan,Z[:,col]),axes(Z,2))
    Z[isnan.(Z)].=m # to better see
    range_val=(-m,m) 
    hmap=heatmap!(ax,Z,colormap=:balance,colorrange=range_val)
    lines!(ax,[1,size(Z,2)],[nan_row,nan_row],color=:green,linewidth=2,linestyle=:dash)
    lines!(ax,[nan_col,nan_col],[1,size(Z,1)],color=:green,linewidth=2,linestyle=:dash)
    ax.yreversed=false
    ax.aspect=Makie.DataAspect()
    Colorbar(f[i,j][1,2],colormap=:balance,limits=Float64.(range_val),tellheight=true)
end

"""
    is_equal(x::T, dx::T, y::T, dy::T) -> Bool where {T<:Real}

Check if two wavenumbers with their respective tensions overlap. The function constructs intervals around each wavenumber based on the given tensions and checks for overlap.

# Arguments
- `x::T` : The first wavenumber.
- `dx::T` : tension associated with the first wavenumber.
- `y::T` : The second wavenumber.
- `dy::T` : The tension associated with the second wavenumber.

# Returns
`Bool` : `true` if the intervals `[x-dx, x+dx]` and `[y-dy, y+dy]` overlap, `false` otherwise.
"""
function is_equal(x::T,dx::T,y::T,dy::T) :: Bool where {T<:Real}
    x_lower=x-dx
    x_upper=x+dx
    y_lower=y-dy
    y_upper=y+dy
    # Check if the intervals overlap
    return max(x_lower,y_lower)<=min(x_upper,y_upper)
end

"""
    match_wavenumbers(ks_l::Vector{T}, ts_l::Vector{T}, ks_r::Vector{T}, ts_r::Vector{T}) -> Tuple{Vector{T}, Vector{T}, Vector{Bool}} where {T<:Real}

Match wavenumbers and tensions from two sorted lists (`ks_l` and `ks_r`). The function ensures that overlapping wavenumbers (as determined by `is_equal`) are merged, keeping the one with the smaller tension. If no overlap exists, wavenumbers are appended in order of magnitude.

# Arguments
- `ks_l::Vector{T}` : List of wavenumbers from the left list.
- `ts_l::Vector{T}` : List of tensions from the left list.
- `ks_r::Vector{T}` : List of wavenumbers from the right list.
- `ts_r::Vector{T}` : List of tensions from the right list.

# Returns
- `ks::Vector{T}` : List of merged wavenumbers.
- `ts::Vector{T}` : List of merged tensions corresponding to the wavenumbers.
control::Vector{Bool} : A boolean vector indicating whether a merged wavenumber resulted from overlap between `ks_l` and `ks_r`.
"""
function match_wavenumbers(ks_l::Vector{T},ts_l::Vector{T},ks_r::Vector{T},ts_r::Vector{T}) where {T<:Real}
    i=1
    j=1
    ks=T[]
    ts=T[]
    control=Bool[]
    while i<=length(ks_l) && j<=length(ks_r)
        x,dx=ks_l[i],ts_l[i]
        y,dy=ks_r[j],ts_r[j]
        if is_equal(x,dx,y,dy)
            if dx<dy
                push!(ks,x); push!(ts,dx)
            else
                push!(ks,y); push!(ts,dy)
            end
            push!(control,true)
            i+=1
            j+=1
        elseif x<y
            push!(ks,x); push!(ts,dx); push!(control,false)
            i+=1
        else
            push!(ks,y); push!(ts,dy); push!(control,false)
            j+=1
        end
    end
    while i<=length(ks_l)
        push!(ks,ks_l[i]); push!(ts,ts_l[i]); push!(control,false)
        i+=1
    end
    while j<=length(ks_r)
        push!(ks,ks_r[j]); push!(ts,ts_r[j]); push!(control,false)
        j+=1
    end
    return ks,ts,control
end

"""
    match_wavenumbers_with_X(ks_l::Vector, ts_l::Vector, X_l::Vector{Vector}, ks_r::Vector, ts_r::Vector, X_r::Vector{Vector}) -> Tuple{Vector, Vector, Vector{Vector}, Vector{Bool}}

Match wavenumbers and tensions from two input lists, taking into account their respective tensions. If there is any overlap (`is_equal` is called) between the `ks_l` and `ks_r` then we choose the one to push into the `ks` those that have the lowest tension (as more accurate). Otherwise we append those that are smaller of the two. In this way we glue together the `ks_l` and `ks_r` to ks such that it has the smallest tensions and smallest `k` of the two closest to one another.

#Arguments
ks_l::Vector{<:Real} : List of wavenumbers from the left
ts_l::Vector{<:Real} : List of wavenumbers from the right
X_l::Vector{Vector{<:Real}} : List of vectors of vectors of the left sample -> the solve_vector sols for each k in ks_l
ks_r::Vector{<:Real} : List of wavenumbers from the right
ts_r::Vector{<:Real} : List of tensions from the right
X_r::Vector{Vector{<:Real}} : List of vectors of vectors of the right sample -> the solve_vector sols for each k in ks_r

#Returns
ks::Vector{<:Real} : List of merged wavenumbers that match 
ts::Vector{<:Real} : List of merged tensions that 
X_list::Vector{Vector{<:Real}} : List of vectors of vectors of the matched/merged ks
control::Vector{Bool} : List of boolean values indicating whether there was an overlap and we had to choose based on tension value to merge
"""
function match_wavenumbers_with_X(ks_l::Vector{T},ts_l::Vector{T},X_l::Vector{Vector{T}},ks_r::Vector{T},ts_r::Vector{T},X_r::Vector{Vector{T}}) where {T<:Real}
    i=1
    j=1
    ks=T[]
    ts=T[]
    Xs=Vector{Vector{T}}()
    control=Bool[]
    while i<=length(ks_l) && j<=length(ks_r)
        x,dx,Xx=ks_l[i],ts_l[i],X_l[i]
        y,dy,Xy=ks_r[j],ts_r[j],X_r[j]
        if is_equal(x,dx,y,dy)
            if dx<dy
                push!(ks,x); push!(ts,dx); push!(Xs,Xx)
            else
                push!(ks,y); push!(ts,dy); push!(Xs,Xy)
            end
            push!(control,true)
            i+=1
            j+=1
        elseif x<y
            push!(ks,x); push!(ts,dx); push!(Xs,Xx); push!(control,false)
            i+=1
        else
            push!(ks,y); push!(ts,dy); push!(Xs,Xy); push!(control,false)
            j+=1
        end
    end
    while i<=length(ks_l)
        push!(ks,ks_l[i]); push!(ts,ts_l[i]); push!(Xs,X_l[i]); push!(control,false)
        i+=1
    end
    while j<=length(ks_r)
        push!(ks,ks_r[j]); push!(ts,ts_r[j]); push!(Xs,X_r[j]); push!(control,false)
        j+=1
    end
    return ks,ts,Xs,control
end

"""
    overlap_and_merge!(k_left::Vector{T}, ten_left::Vector{T}, k_right::Vector{T}, ten_right::Vector{T}, control_left::Vector{Bool}, kl::T, kr::T; tol::T=1e-3) :: Nothing where {T<:Real}

This function merges two sets of wavenumber data (`k_left`, `ten_left`) and (`k_right`, `ten_right`) that may have overlapping wavenumbers in the interval `[kl - tol, kr + tol]`. 
- It ensures that overlapping wavenumbers are merged, preferring the data with the smaller tension when overlaps occur. 
- Non-overlapping wavenumbers from the right interval (`k_right`) are appended to the left interval (`k_left`) in order.

# Arguments
- `k_left::Vector{T}`: Vector of wavenumbers from the left interval.
- `ten_left::Vector{T}`: Vector of tensions corresponding to `k_left`.
- `k_right::Vector{T}`: Vector of wavenumbers from the right interval.
- `ten_right::Vector{T}`: Vector of tensions corresponding to `k_right`.
- `control_left::Vector{Bool}`: Vector indicating whether each wavenumber in `k_left` was merged (`true`) or not (`false`).
- `kl::T`: Left boundary of the overlapping interval.
- `kr::T`: Right boundary of the overlapping interval.
- `tol::T`: Tolerance for determining overlaps (default: `1e-3`).

# Returns
- `Nothing`: The function modifies `k_left`, `ten_left`, and `control_left` in place.
"""
function overlap_and_merge!(k_left::Vector{T},ten_left::Vector{T},k_right::Vector{T},ten_right::Vector{T},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real}
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(control_left,fill(false,length(k_right)))
        return nothing
    end
    isempty(k_right) && return nothing
    idx_l=(k_left.>(kl-tol)) .&& (k_left.<(kr+tol))
    idx_r=(k_right.>(kl-tol)) .&& (k_right.<(kr+tol))
    ks_l=k_left[idx_l]
    ts_l=ten_left[idx_l]
    ks_r=k_right[idx_r]
    ts_r=ten_right[idx_r]
    ks,ts,control=match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    del_l=findall(idx_l)
    deleteat!(k_left,del_l)
    deleteat!(ten_left,del_l)
    deleteat!(control_left,del_l)
    append!(k_left,ks)
    append!(ten_left,ts)
    append!(control_left,control)
    fl=findlast(idx_r)
    idx_last=isnothing(fl) ? 1 : fl+1
    append!(k_left,k_right[idx_last:end])
    append!(ten_left,ten_right[idx_last:end])
    append!(control_left,fill(false,length(k_right[idx_last:end])))
    return nothing
end

"""
    overlap_and_merge_state!(k_left::Vector, ten_left::Vector, X_left::Vector{Vector}, k_right::Vector, ten_right::Vector, X_right::Vector{Vector}, control_left::Vector{Bool}, kl::T, kr::T; tol::Float64=1e-3) :: Nothing where {T<:Real}

This function merges two sets of wavenumber data (`k_left`, `ten_left`, `X_left`) and (`k_right`, `ten_right`, `X_right`) that may have overlapping wavenumbers in the interval `[kl - tol, kr + tol]`. It ensures that each wavenumber and its associated data are only included once in the merged result, preferring the data with the smaller tension when overlaps occur.

# Arguments
- `k_left::Vector`: Vector of wavenumbers from the left interval.
- `ten_left::Vector`: Vector of tensions corresponding to `k_left`.
- `X_left::Vector{Vector}`: Vector of eigenvectors corresponding to `k_left`.
- `k_right::Vector`: Vector of wavenumbers from the right interval.
- `ten_right::Vector`: Vector of tensions corresponding to `k_right`.
- `X_right::Vector{Vector}`: Vector of eigenvectors corresponding to `k_right`.
- `control_left::Vector{Bool}`: Vector indicating whether each wavenumber in `k_left` was merged (`true`) or not (`false`).
- `kl::T`: Left boundary of the overlapping interval.
- `kr::T`: Right boundary of the overlapping interval.
- `tol::Float64`: Tolerance for determining overlaps (default: `1e-3`).

# Returns
- `Nothing`: The function modifies `k_left`, `ten_left`, `X_left`, and `control_left` in place.
"""
function overlap_and_merge_state!(k_left::AbstractVector{T},ten_left::AbstractVector{T},X_left::Vector{Vector{T}},k_right::AbstractVector{T},ten_right::AbstractVector{T},X_right::Vector{Vector{T}},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real}
    # Check if intervals are empty
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(X_left,X_right)
        append!(control_left,[false for _ in 1:length(k_right)])
        return nothing
    end
    if isempty(k_right)
        return nothing
    end
    # Find overlaps in interval [kl - tol, kr + tol]
    idx_l=k_left.>(kl - tol) .&& (k_left.<(kr+tol))
    idx_r=k_right.>(kl - tol) .&& (k_right.<(kr+tol))
    # Extract overlapping data
    ks_l=k_left[idx_l]
    ts_l=ten_left[idx_l]
    Xs_l=X_left[idx_l]
    ks_r=k_right[idx_r]
    ts_r=ten_right[idx_r]
    Xs_r=X_right[idx_r]
    # Check if wavenumbers match in overlap interval
    ks,ts,Xs,control=match_wavenumbers_with_X(ks_l,ts_l,Xs_l,ks_r,ts_r,Xs_r)
    # For all those that matched we put them in the location where there was ambiguity (overlap) for merging ks_r into ks_l. So we delete all the k_left that were in the overlap interval and merged the matched results into the correct location where we deleted from idx_l
    deleteat!(k_left,findall(idx_l))
    append!(k_left,ks)
    deleteat!(ten_left,findall(idx_l))
    append!(ten_left,ts)
    deleteat!(X_left,findall(idx_l))
    append!(X_left,Xs)
    deleteat!(control_left,findall(idx_l))
    append!(control_left,control)
    # After we are done with in-tolerance ks safely append what is let to the final results. This is the part where
    fl=findlast(idx_r)
    idx_last=isnothing(fl) ? 1 : fl + 1
    append!(k_left,k_right[idx_last:end])
    append!(ten_left,ten_right[idx_last:end])
    append!(X_left,X_right[idx_last:end])
    append!(control_left,[false for _ in idx_last:length(k_right)])
end

"""
    compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T;tol::T=T(1e-4),N_expect=1,dk_threshold::T=T(0.05),fundamental::Bool=true,multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true,cholesky::Bool=true) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}

Computes the spectrum over a range of wavenumbers `[k1, k2]` using the given solver, basis, and billiard, returning the merged `StateData` containing wavenumbers, tensions, and eigenvectors. MAIN ONE -> for both eigenvalues and husimi/wavefunctions since the expansion coefficients of the basis for the k are saved

# Arguments
- `solver`: The solver used to compute the spectrum.
- `basis`: The basis set used in computations.
- `billiard`: The billiard domain for the problem.
- `k1`, `k2`: The starting and ending wavenumbers of the spectrum range.
- `tol`: Tolerance for computations (default: `1e-4`).
- `N_expect`: Expected number of eigenvalues per interval (default: `3`).
- `dk_threshold`: Maximum allowed interval size for `dk` (default: `0.05`).
- `fundamental`: Whether to use fundamental domain properties (default: `true`).
- `multithreaded_matrices::Bool=false`: If the matrix construction should be multithreaded for the basis and gradient matrices. Very dependant on the k grid and the basis choice to determine the optimal choice for what to multithread.
- `multithreaded_ks::Bool=true`: If the k loop is multithreaded. This is usually the best choice since matrix construction for small k is not as costly.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
- `state-res::StateData{T,T}`: A struct containing:
    - `ks::Vector{T}`: Vector of computed wavenumbers.
    - `X::Vector{T}`: Matrix where each column are the coefficients for the basis expansion for the same indexed eigenvalue.
    - `tens::Vector{T}`: Vector of tensions for each eigenvalue
- `control::Vector{Bool}`: Vector signifying if the eigenvalues at that indexed was compared to another and merged.
"""
function compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T;tol::T=T(1e-4),N_expect=1,dk_threshold::T=T(0.05),fundamental::Bool=true,multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,cholesky::Bool=false) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}
    k_vals=T[]
    dk_vals=T[]
    k0=k1
    A_fund=billiard.area_fundamental
    L_fund=billiard.length_fundamental
    A_full=billiard.area
    L_full=billiard.length
    while k0<k2
        dk=fundamental ? N_expect/(A_fund*k0/(2π)-L_fund/(4π)) : N_expect/(A_full*k0/(2π)-L_full/(4π))
        dk=abs(dk)
        dk=min(dk,dk_threshold)
        push!(k_vals,k0)
        push!(dk_vals,dk)
        k0+=dk
    end
    @info "Scaling Method w/ StateData..."
    println("min/max dk: ",extrema(dk_vals))
    println("Total intervals: ",length(k_vals))
    all_states=Vector{StateData{T,T}}(undef,length(k_vals))
    all_states[end]=solve_state_data_bundle_with_INFO(solver,basis,billiard,k_vals[end],dk_vals[end]+tol;multithreaded=multithreaded_matrices)
    @info "Multithreading loop? $(multithreaded_ks), multithreading matrix construction? $(multithreaded_matrices)"
    p=Progress(length(k_vals),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(k_vals)[1:end-1]
        ki=k_vals[i]
        dki=dk_vals[i]
        all_states[i]=solve_state_data_bundle(solver,basis,billiard,ki,dki+tol;multithreaded=multithreaded_matrices,cholesky=cholesky)
        next!(p)
    end
    println("Merging intervals...")
    state_res=all_states[1]
    control=[false for _ in 1:length(state_res.ks)]
    p=Progress(length(all_states)-1,1)
    for i in 2:length(all_states)
        overlap_and_merge_state!(
            state_res.ks,state_res.tens,state_res.X,
            all_states[i].ks,all_states[i].tens,all_states[i].X,
            control,k_vals[i-1],k_vals[i];tol=tol)
        next!(p)
    end
    return state_res,control
end

"""
    compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T,dk::T;tol::T=T(1e-4),multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}

Compute the spectrum of a billiard system over a fixed-resolution wavenumber grid `[k1, k2]` using a specified step size `dk`, and return eigenstates (wavenumbers, tensions, basis coefficients) in a merged `StateData` structure.

This is the fixed-interval version of the Vergini–Saraceno-style solver that **also captures the eigenvectors / basis expansion coefficients** for each wavenumber.
After solving, results are merged using `overlap_and_merge_state!`, keeping lower-tension solutions when overlaps occur.

# Arguments
- `solver::Sol`: Spectral solver (e.g., scaling method), subtype of `AcceleratedSolver`.
- `basis::Ba`: Basis object compatible with the billiard geometry, subtype of `AbsBasis`.
- `billiard::Bi`: The geometry (domain), subtype of `AbsBilliard`.
- `k1::T`, `k2::T`: Start and end values of the wavenumber range.
- `dk::T`: Fixed step size between wavenumber intervals.
- `tol::T=1e-4`: Tolerance for merging results in overlapping intervals.
- `multithreaded_matrices::Bool=false`: Enable multithreading for matrix assembly.
- `multithreaded_ks::Bool=true`: Enable multithreading over `k` intervals.
- `cholesky::Bool=false`: Use Cholesky decomposition for solving linear systems.

# Returns
- `state_res::StateData{T,T}`: Struct holding merged eigenvalues, tensions, and eigenvectors.
  - `state_res.ks`: Final merged wavenumbers.
  - `state_res.tens`: Tension values associated with each wavenumber.
  - `state_res.X`: Basis coefficient vectors for each eigenvalue.
- `control::Vector{Bool}`: Vector indicating if an eigenvalue was involved in overlap resolution (`true`) or added uniquely (`false`).
"""
function compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k1::T,k2::T,dk::T;tol::T=T(1e-4),multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,cholesky::Bool=false) where {Ba<:AbsBasis, Bi<:AbsBilliard, T<:Real}
    k_vals=collect(range(k1,k2,step=dk))
    @info "Scaling Method w/ StateData..."
    println("Total intervals: ",length(k_vals))
    all_states=Vector{StateData{T,T}}(undef,length(k_vals))
    all_states[end]=solve_state_data_bundle_with_INFO(solver,basis,billiard,k_vals[end],dk+tol;multithreaded=multithreaded_matrices)
    @info "Multithreading loop? $(multithreaded_ks), multithreading matrix construction? $(multithreaded_matrices)"
    p=Progress(length(k_vals),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(k_vals)[1:end-1]
        ki=k_vals[i]
        all_states[i]=solve_state_data_bundle(solver,basis,billiard,ki,dk+tol;multithreaded=multithreaded_matrices,cholesky=cholesky)
        next!(p)
    end
    println("Merging intervals...")
    state_res=all_states[1]
    control=[false for _ in 1:length(state_res.ks)]
    p=Progress(length(all_states)-1,1)
    for i in 2:length(all_states)
        overlap_and_merge_state!(
            state_res.ks,state_res.tens,state_res.X,
            all_states[i].ks,all_states[i].tens,all_states[i].X,
            control,k_vals[i-1],k_vals[i];tol=tol)
        next!(p)
    end
    return state_res,control
end

"""
    compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,N1::Int,N2::Int;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental::Bool=true,multithreaded_matrices::Bool=false,multithreaded_ks::Bool=true) where {Ba<:AbsBasis,Bi<:AbsBilliard}

Computes the spectrum over a range of wavenumbers defined by the bracketing interval of their state number `[N1, N2]` using the given solver, basis, and billiard, returning the merged `StateData` containing wavenumbers, tensions, and eigenvectors. MAIN ONE -> for both eigenvalues and husimi/wavefunctions since the expansion coefficients of the basis for the k are saved. This one is just a wrapper function for the k version of this function.

# Arguments
- `solver`: The solver used to compute the spectrum.
- `basis`: The basis set used in computations.
- `billiard`: The billiard domain for the problem.
- `N1::Int`, `N2::Int`: The starting and ending state numbers that will be translated to their corresponding eigenvalues via Weyl's law.
- `tol`: Tolerance for computations (default: `1e-4`).
- `N_expect`: Expected number of eigenvalues per interval (default: `3`).
- `dk_threshold`: Maximum allowed interval size for `dk` (default: `0.05`).
- `fundamental`: Whether to use fundamental domain properties (default: `true`).
- `multithreaded_matrices::Bool=false`: If the matrix construction should be multithreaded for the basis and gradient matrices. Very dependant on the k grid and the basis choice to determine the optimal choice for what to multithread.
- `multithreaded_ks::Bool=true`: If the k loop is multithreaded. This is usually the best choice since matrix construction for small k is not as costly.
- `cholesky::Bool=false`: Use Cholesky decomposition for solving linear systems (default: `false`).

# Returns
- `state-res::StateData{T,T}`: A struct containing:
    - `ks::Vector{T}`: Vector of computed wavenumbers.
    - `X::Vector{T}`: Matrix where each column are the coefficients for the basis expansion for the same indexed eigenvalue.
    - `tens::Vector{T}`: Vector of tensions for each eigenvalue
- `control::Vector{Bool}`: Vector signifying if the eigenvalues at that indexed was compared to another and merged.
"""
function compute_spectrum_with_state_scaling_method(solver::VerginiSaraceno,basis::Ba,billiard::Bi,N1::Int,N2::Int;tol=1e-4,N_expect=1,dk_threshold=0.05,fundamental::Bool=true,multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false,cholesky::Bool=false) where {Ba<:AbsBasis,Bi<:AbsBilliard}
    k1=k_at_state(N1,billiard;fundamental=fundamental)
    k2=k_at_state(N2,billiard;fundamental=fundamental)
    println("k1 = $(k1), k2 = $(k2)")
    return compute_spectrum_with_state_scaling_method(solver,basis,billiard,k1,k2,tol=tol,N_expect=N_expect,dk_threshold=dk_threshold,fundamental=fundamental,multithreaded_matrices=multithreaded_matrices,multithreaded_ks=multithreaded_ks,cholesky=cholesky)
end

"""
    compute_eigenstate(solver::VerginiSaraceno, basis::AbsBasis, billiard::AbsBilliard, k::T; dk::T=0.1) where {T<:Real}

Computes a single eigenstate for a given wavenumber `k` using an accelerated solver. Based on the inputted k it finds the closest one. In principle this will find an even better precide eigenvalue k for which we get it's basis coefficients expansion (`Eigenstate`).

# Arguments
- `solver::VerginiSaraceno`: The solver object to compute the eigenstate.
- `basis::AbsBasis`: The basis in which the eigenstate is represented.
- `billiard::AbsBilliard`: The billiard domain for the eigenstate.
- `k::T`: The wavenumber of the eigenstate.
- `dk::T`: The step size for the solver (default: `0.1`).

# Returns
An `Eigenstate` object representing the computed eigenstate.
"""
function compute_eigenstate(solver::VerginiSaraceno,basis::AbsBasis,billiard::AbsBilliard,k;dk=0.1)
    L=billiard.length
    dim=max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    ks,tens,X=solve_vectors(solver,basis_new,pts,k,dk)
    idx=findmin(abs.(ks.-k))[2]
    k_state=ks[idx]
    ten=tens[idx]
    vec=X[:,idx]
    return Eigenstate(k_state,k,vec,ten,basis_new,billiard)
end

# no need for basis and billiard data due to complex nested hierarchy
"""
    struct StateData{K,T} <: AbsState 

Convenience wrapper for all the relevant results from the computation of a spectrum. It saves the wavenumbers, the tensions and the expansion coefficient for the basis stored as a Vector
"""
struct StateData{K,T} <: AbsState 
    ks::Vector{K}
    X::Vector{Vector{T}}  # Changed from Matrix{T}
    tens::Vector{T}
end

# constructor for the saved data with no billiard or basis information
"""
    StateData(ks::Vector, X::Vector{Matrix}, tens::Vector) :: StateData

Constructor for the convenience wrapper `StateData`. Under the hood it filters the coefficients that are very small (sets them to zero(T) if the val is smaller than eps(T)) so as to get better representation of the wavefunction

# Arguments
- `ks::Vector`: The wavenumbers for which the wavefunction was computed.
- `X::Vector{Matrix}`: The expansion coefficients for the basis stored as a Vector of vectors.
- `tens::Vector`: The tension minima for which the wavefunction was computed.
"""
function StateData(ks::Vector,X::Vector{Matrix},tens::Vector)  
    # Access the first element of the first vector in X
    eps=set_precision(X[1][1])
    type=eltype(X[1])
    if type <: Real
        filtered_array=[[abs(x) > eps ? x : zero(type) for x in vec] for vec in X] # Filter each vector in X individually
    else
        filtered_array=X
    end
    # dim can be gained for each k in ks separately as they all do not have the same dimension as the X vector of vectors has a different dimension for each k
    return StateData(ks,filtered_array,tens)
end

# this is basically the new solve where we incur the smallest penalty for getting the ks and the relevant state information for saving the husimi functions but it is much more efficient than doint it again once we have the eigenvalues
"""
    solve_state_data_bundle(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k,dk;multithreaded::Bool=true) where {Ba<:AbsBasis, Bi<:AbsBilliard}

Solves the generalized eigenvalue problem in a small interval `[k0-dk, k0+dk]` and constructs the `StateData` object in that small interval. This function is iteratively called in the `compute_spectrum` function version that also computes the `StateData` object. The advantage of this version of the function from the regular `solve(solver...)` is that we get the eigenvectors here witjh minimal additional computational cost.

# Arguments
- `solver::VerginiSaraceno`: The solver object to use for the eigenvalue problem.
- `basis<:AbsBasis`: The basis object to use for the eigenvalue problem.
- `billiard<:AbsBilliard`: The billiard object to use for the eigenvalue problem.
- `k<:Real`: The center of the interval for which to solve the eigenvalue problem.
- `dk<:Real`: The width of the interval for which to solve the eigenvalue problem.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
A `StateData` object containing the wavenumbers, the tensions and the expansion coefficients for the basis stored as a Vector of Vectors after a generalized eigenvalue problem computation.
"""
function solve_state_data_bundle(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k,dk;multithreaded::Bool=true,cholesky::Bool=false) where {Ba<:AbsBasis, Bi<:AbsBilliard}
    L=billiard.length
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard, k)
    ks,tens,X_matrix=solve_vectors(solver,basis_new,pts,k,dk;multithreaded=multithreaded,cholesky=cholesky) # this one filters the ks that are outside k+-dk and gives us the filtered out ks, tensions and X matrix of filtered vectors. No need to store dim as we can get it from the length(X[1])
    # Extract columns of X_matrix and store them as a Vector of Vectors b/c it is easier to merge them in the top function -> compute_spectrum_with_state
    X_vectors=[Vector(col) for col in eachcol(X_matrix)]
    return StateData(ks,X_vectors,tens)
end

#### INTERNAL FUNCTION FOR TESTING TIME AND ALLOCATIONS OF MATRIX CONSTRUCTIONS AND EIGENVALUE SOLVING ####
# Primarily used for checking regularizations of ill-conditioned F and dF/dk matrices ala Barnett. Useful for observing allocations, execution time and observing the variation of the condition number as k increases
function solve_state_data_bundle_with_INFO(solver::VerginiSaraceno,basis::Ba,billiard::Bi,k,dk;multithreaded::Bool=true) where {Ba<:AbsBasis, Bi<:AbsBilliard}
    start_init=time()
    L=billiard.length
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    @info "Basis resizing..."
    @time basis_new=resize_basis(basis,billiard,dim,k)
    @info "Pts on boundary evaluation..."
    s_pts=time()
    @time pts=evaluate_points(solver,billiard, k)
    e_pts=time()
    @info "F & dF/dk matrix construction..."
    s_con=time()
    @time F,Fk=construct_matrices(solver,basis_new,pts,k;multithreaded=multithreaded)
    e_con=time()
    @info "F & dF/dk dims: $(size(F))"
    start1=time()
    @blas_multi_then_1 MAX_BLAS_THREADS @warn "Initial condition num. F before regularization: $(cond(F))"
    @blas_multi_then_1 MAX_BLAS_THREADS @warn "Initial condition num. dF/dk before regularization: $(cond(Fk))"
    end1=time()
    A=Symmetric(F)
    B=Symmetric(Fk)
    @info "Removing numerical nullspace of ill conditioned F and eigenvalue problem..."
    s_reg=time()
    @blas_multi_then_1 MAX_BLAS_THREADS @time d,S=eigen(Symmetric(A))
    e_reg=time()
    @info "Smallest & Largest eigval: $(extrema(d))"
    @info "Nullspace removal with criteria eigval > $(solver.eps*maximum(d))"
    idx=d.>solver.eps*maximum(d)
    @info "Dim of num Nullspace: $(count(!,idx))" # counts the number of falses = dim of nullspace
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        q=1.0./sqrt.(d[idx])
        C=@view S[:,idx]
        C_scaled=C.*q'
        n=size(C_scaled,2)
        tmp=Matrix{eltype(B)}(undef,size(B,1),n)
        E=Matrix{eltype(B)}(undef,n,n)
        mul!(tmp,B,C_scaled)
        mul!(E,C_scaled',tmp)
        start2=time()
        @warn "Final eigenvalue problem with new condition number: $(cond(E)) and reduced dimension $(size(E))"
        end2=time()
        s_fin=time()
        @time mu,Z=eigen(Symmetric(E))
        e_fin=time()
        ks,ten=sm_results(mu,k)
        idx=abs.(ks.-k).<dk
        ks=ks[idx]
        ten=ten[idx]
        Z=Z[:,idx]
        X=C_scaled*Z #transform into original basis 
        X=(sqrt.(ten))' .* X # Use the automatic normalization via tension values as described in Barnett's thesis. Maybe also use X = X .* reshape(sqrt.(ten), 1, :) ?
        p=sortperm(ks)
        ks,ten,X= ks[p],ten[p],X[:,p]
        # Extract columns of X_matrix and store them as a Vector of Vectors b/c it is easier to merge them in the top function -> compute_spectrum_with_state
        X_vectors=[Vector(col) for col in eachcol(X)]
        end_init=time()
        total_time=end_init-start_init-(end2-start2)-(end1-start1)
        @info "Final computation time: $(total_time) s"
        println("%%%%% SUMMARY %%%%%")
        println("Percentage of total time (most relevant ones): ")
        println("Boundary Pts evaluation: $(100*(e_pts-s_pts)/total_time) %")
        println("F & dF/dk construction: $(100*(e_con-s_con)/total_time) %")
        println("Nullspace removal: $(100*(e_reg-s_reg)/total_time) %")
        println("Final eigen problem: $(100*(e_fin-s_fin)/total_time) %")
        println("%%%%%%%%%%%%%%%%%%%")
        return StateData(ks,X_vectors,ten)
    end
end

