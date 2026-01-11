"""
    solve_wavenumber(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk;multithreaded::Bool=true)

Solves the wavenumber for an `AcceleratedSolver` by finding the one closest to the given reference wavenumber `k` if found in the `dk` interval.

# Arguments
- `solver<:AcceleratedSolver`: An instance of `AcceleratedSolver`.
- `basis<:AbsBasis`: The basis of the problem, no additional restrictions.
- `billiard<:AbsBilliard`: billiard instance, geometrical information.
- `k::T`: The reference wavenumber.
- `dk::T`: The interval in which to find the closest wavenumber.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
- `T`: The closest wavenumber found in the `dk` interval.
- `T`: The corresponding tension found for the closest wavenumber (by construction smallest tension in the given interval via findmin in the code).
"""
function solve_wavenumber(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk;multithreaded::Bool=true,cholesky::Bool=false)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    ks,ts=solve(solver,new_basis,pts,k,dk;multithreaded=multithreaded,cholesky=cholesky)
    idx=findmin(abs.(ks.-k))[2]
    return ks[idx],ts[idx]
end

"""
    solve_spectrum(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk;multithreaded::Bool=true,cholesky::Bool=true)

Solves for all the wavenumbers and corresponding tensions that `solve(<:AcceleratedSolver...)` gives us in the given interval `dk`.

# Arguments
- `solver<:AcceleratedSolver`: An instance of `AcceleratedSolver`.
- `basis<:AbsBasis`: The basis of the problem, no additional restrictions.
- `billiard<:AbsBilliard`: billiard instance, geometrical information.
- `k::T`: The reference wavenumber.
- `dk::T`: The interval in which to find the wavenumbers and corresponding tensions.
- `multithreaded::Bool=true`: If the matrix construction should be multithreaded.
- `cholesky::Bool=false`: If the generalized eigenproblem should be solved via Cholesky factorization (default false).

# Returns
- `Vector{T}`: The wavenumbers found in the `dk` interval.
- `Vector{T}`: The corresponding tensions found for the wavenumbers in the `dk` interval.
"""
function solve_spectrum(solver::AcceleratedSolver,basis::AbsBasis,billiard::AbsBilliard,k,dk;multithreaded::Bool=true,cholesky::Bool=false)
    dim=max(solver.min_dim,round(Int,billiard.length*k*solver.dim_scaling_factor/(2*pi)))
    new_basis=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver, billiard,k)
    ks,ts=solve(solver,new_basis,pts,k,dk;multithreaded=multithreaded,cholesky=cholesky)
    return ks,ts
end

# INTERNAL FUNCTION THAT GIVES US USEFUL INFORMATION OF THE TIME COMPLEXITY AND STABILITY OF THE ALGORITHM
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