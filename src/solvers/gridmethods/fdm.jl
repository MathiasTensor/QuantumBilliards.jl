using LinearAlgebra, SparseArrays, Arpack

"""
EXAMPLE CODE: 

using QuantumBilliards, CairoMakie

billiard,_=make_rectangle_and_basis(2.0,1.0)
billiard,_=make_circle_and_basis(1.0)
billiard,_=make_prosen_and_basis(0.4)
billiard,_=make_mushroom_and_basis(1.0,1.0,1.0)
billiard,_=make_generalized_sinai_and_basis()
billiard,_=make_ellipse_and_basis(2.0,1.0)

fem=FiniteElementMethod(billiard,300,300,5.0)
x_grid,y_grid=fem.x_grid,fem.y_grid

@time H=FEM_Hamiltonian(fem)
println("Constructed Hamiltonian")

Es,wavefunctions=compute_fem_eigenmodes(fem,nev=99)
ks=sqrt.(abs.(Es))
println("Constructed Wavefunctions")
idxs=findall(x->x>1e-4,ks)
ks=ks[idxs]
wavefunctions=wavefunctions[idxs]
println(ks)

wavefunctions=[abs2.(wf) for wf in wavefunctions]

f=QuantumBilliards.plot_wavefunctions_BATCH(ks,wavefunctions,x_grid,y_grid,billiard,fundamental=false)
# display or save Figure()
"""

"""
Comparisosn of eigenvalues obtained from standard FDM using the 5 point stencil
n  | Numerical  | Analytical | Abs Error | Rel Error | % Relative Error
--------------------------------------------------
1   |  2.48364  |  2.48365  | 1.14245e-05 | 4.59987e-06 |    0.000
2   |  3.14156  |  3.14159  | 3.61271e-05 | 1.14996e-05 |    0.001
3   |  4.00464  |  4.00476  | 1.20446e-04 | 3.00756e-05 |    0.003
4   |  4.57954  |  4.57962  | 8.05448e-05 | 1.75877e-05 |    0.002
5   |  4.96700  |  4.96729  | 2.97028e-04 | 5.97967e-05 |    0.006
6   |  4.96720  |  4.96729  | 9.13952e-05 | 1.83994e-05 |    0.002
7   |  5.55346  |  5.55360  | 1.48164e-04 | 2.66790e-05 |    0.003
8   |  5.98082  |  5.98141  | 5.96739e-04 | 9.97656e-05 |    0.010
9   |  6.28290  |  6.28319  | 2.89011e-04 | 4.59975e-05 |    0.005
10  |  6.75598  |  6.75625  | 2.72978e-04 | 4.04038e-05 |    0.004
11  |  7.02376  |  7.02481  | 1.05012e-03 | 1.49487e-04 |    0.015
12  |  7.02454  |  7.02481  | 2.74659e-04 | 3.90984e-05 |    0.004
13  |  7.11153  |  7.11208  | 5.49741e-04 | 7.72968e-05 |    0.008
14  |  7.45063  |  7.45094  | 3.08457e-04 | 4.13984e-05 |    0.004
15  |  8.00856  |  8.00952  | 9.63517e-04 | 1.20296e-04 |    0.012
16  |  8.00911  |  8.00952  | 4.10931e-04 | 5.13053e-05 |    0.005
17  |  8.08448  |  8.08617  | 1.68770e-03 | 2.08714e-04 |    0.021
18  |  8.67439  |  8.67501  | 6.20778e-04 | 7.15594e-05 |    0.007
19  |  8.95335  |  8.95492  | 1.56197e-03 | 1.74426e-04 |    0.017
20  |  8.95427  |  8.95492  | 6.49544e-04 | 7.25349e-05 |    0.007
21  |  9.15670  |  9.15924  | 2.54003e-03 | 2.77319e-04 |    0.028
22  |  9.15859  |  9.15924  | 6.44347e-04 | 7.03494e-05 |    0.007
23  |  9.42380  |  9.42478  | 9.75379e-04 | 1.03491e-04 |    0.010
24  |  9.48934  |  9.49000  | 6.60758e-04 | 6.96267e-05 |    0.007
25  |  9.93221  |  9.93459  | 2.37602e-03 | 2.39167e-04 |    0.024
26  |  9.93386  |  9.93459  | 7.31150e-04 | 7.35964e-05 |    0.007
27  | 10.23670  | 10.24034  | 3.63768e-03 | 3.55230e-04 |    0.036
28  | 10.23883  | 10.24034  | 1.50997e-03 | 1.47453e-04 |    0.015
29  | 10.47763  | 10.47852  | 8.93026e-04 | 8.52245e-05 |    0.009
30  | 10.93589  | 10.93933  | 3.43630e-03 | 3.14123e-04 |    0.031
31  | 11.10495  | 11.10721  | 2.25799e-03 | 2.03290e-04 |    0.020
32  | 11.10602  | 11.10721  | 1.18527e-03 | 1.06712e-04 |    0.011
33  | 11.16133  | 11.16261  | 1.27142e-03 | 1.13900e-04 |    0.011
34  | 11.32216  | 11.32717  | 5.01123e-03 | 4.42408e-04 |    0.044
35  | 11.32591  | 11.32717  | 1.26046e-03 | 1.11278e-04 |    0.011
36  | 11.59500  | 11.59626  | 1.26302e-03 | 1.08916e-04 |    0.011
37  | 11.80548  | 11.80712  | 1.64604e-03 | 1.39411e-04 |    0.014
38  | 11.95806  | 11.96283  | 4.77332e-03 | 3.99013e-04 |    0.040
39  | 11.96152  | 11.96283  | 1.30733e-03 | 1.09283e-04 |    0.011
40  | 12.01103  | 12.01428  | 3.25160e-03 | 2.70644e-04 |    0.027
41  | 12.41154  | 12.41824  | 6.69128e-03 | 5.38827e-04 |    0.054
42  | 12.41681  | 12.41824  | 1.42801e-03 | 1.14993e-04 |    0.011
43  | 12.56406  | 12.56637  | 2.31190e-03 | 1.83975e-04 |    0.018
44  | 12.94860  | 12.95312  | 4.52220e-03 | 3.49121e-04 |    0.035
45  | 12.95146  | 12.95312  | 1.66299e-03 | 1.28385e-04 |    0.013
46  | 12.99424  | 13.00065  | 6.41757e-03 | 4.93634e-04 |    0.049
47  | 13.37163  | 13.37485  | 3.21780e-03 | 2.40586e-04 |    0.024
48  | 13.37265  | 13.37485  | 2.19985e-03 | 1.64477e-04 |    0.016
49  | 13.50379  | 13.51250  | 8.70841e-03 | 6.44471e-04 |    0.064
50  | 13.51032  | 13.51250  | 2.18374e-03 | 1.61609e-04 |    0.016
51  | 13.55602  | 13.55807  | 2.05124e-03 | 1.51293e-04 |    0.015
52  | 13.73668  | 13.73886  | 2.17461e-03 | 1.58282e-04 |    0.016
53  | 13.91119  | 13.91729  | 6.10076e-03 | 4.38358e-04 |    0.044
54  | 14.04123  | 14.04963  | 8.39953e-03 | 5.97847e-04 |    0.060
55  | 14.04743  | 14.04963  | 2.19719e-03 | 1.56388e-04 |    0.016
56  | 14.21977  | 14.22417  | 4.39732e-03 | 3.09144e-04 |    0.031
57  | 14.22153  | 14.22417  | 2.63131e-03 | 1.84988e-04 |    0.018
58  | 14.43709  | 14.43937  | 2.28290e-03 | 1.58103e-04 |    0.016
59  | 14.59816  | 14.60925  | 1.10932e-02 | 7.59327e-04 |    0.076
60  | 14.89386  | 14.90188  | 8.01795e-03 | 5.38049e-04 |    0.054
61  | 14.89941  | 14.90188  | 2.46756e-03 | 1.65587e-04 |    0.017
62  | 14.93978  | 14.94322  | 3.44059e-03 | 2.30244e-04 |    0.023
63  | 15.09669  | 15.10744  | 1.07497e-02 | 7.11549e-04 |    0.071
64  | 15.10155  | 15.10744  | 5.88298e-03 | 3.89410e-04 |    0.039
65  | 15.42784  | 15.43063  | 2.78935e-03 | 1.80767e-04 |    0.018
66  | 15.58621  | 15.58971  | 3.49611e-03 | 2.24258e-04 |    0.022
67  | 15.69409  | 15.70796  | 1.38762e-02 | 8.83389e-04 |    0.088
68  | 15.70345  | 15.70796  | 4.51515e-03 | 2.87444e-04 |    0.029
69  | 15.70449  | 15.70796  | 3.47521e-03 | 2.21239e-04 |    0.022
70  | 15.89280  | 15.90310  | 1.03043e-02 | 6.47945e-04 |    0.065
71  | 15.89964  | 15.90310  | 3.45576e-03 | 2.17301e-04 |    0.022
72  | 16.01134  | 16.01904  | 7.70659e-03 | 4.81089e-04 |    0.048
73  | 16.01575  | 16.01904  | 3.28726e-03 | 2.05210e-04 |    0.021
74  | 16.15884  | 16.17234  | 1.34985e-02 | 8.34668e-04 |    0.083
75  | 16.16888  | 16.17234  | 3.45963e-03 | 2.13923e-04 |    0.021
76  | 16.50616  | 16.51205  | 5.88976e-03 | 3.56695e-04 |    0.036
77  | 16.50854  | 16.51205  | 3.51527e-03 | 2.12891e-04 |    0.021
78  | 16.65681  | 16.66081  | 4.00007e-03 | 2.40088e-04 |    0.024
79  | 16.79117  | 16.80825  | 1.70881e-02 | 1.01665e-03 |    0.102
80  | 16.90500  | 16.91799  | 1.29904e-02 | 7.67845e-04 |    0.077
81  | 16.91434  | 16.91799  | 3.65599e-03 | 2.16100e-04 |    0.022
82  | 16.94452  | 16.95442  | 9.89942e-03 | 5.83885e-04 |    0.058
83  | 17.22635  | 17.24302  | 1.66766e-02 | 9.67150e-04 |    0.097
84  | 17.34241  | 17.35001  | 7.59806e-03 | 4.37928e-04 |    0.044
85  | 17.34505  | 17.35001  | 4.96569e-03 | 2.86207e-04 |    0.029
86  | 17.38161  | 17.38553  | 3.91833e-03 | 2.25379e-04 |    0.023
87  | 17.80099  | 17.80621  | 5.22146e-03 | 2.93238e-04 |    0.029
88  | 17.88907  | 17.90983  | 2.07593e-02 | 1.15910e-03 |    0.116
89  | 17.89734  | 17.90983  | 1.24924e-02 | 6.97518e-04 |    0.070
90  | 17.90464  | 17.90983  | 5.19600e-03 | 2.90120e-04 |    0.029
91  | 17.90549  | 17.90983  | 4.34062e-03 | 2.42360e-04 |    0.024
92  | 17.92814  | 17.94424  | 1.61066e-02 | 8.97592e-04 |    0.090
93  | 18.07500  | 18.08122  | 6.22097e-03 | 3.44057e-04 |    0.034
94  | 18.07606  | 18.08122  | 5.16714e-03 | 2.85774e-04 |    0.029
95  | 18.20750  | 18.21717  | 9.67284e-03 | 5.30974e-04 |    0.053
96  | 18.29816  | 18.31848  | 2.03143e-02 | 1.10895e-03 |    0.111
97  | 18.31332  | 18.31848  | 5.15442e-03 | 2.81378e-04 |    0.028
98  | 18.48112  | 18.48608  | 4.96188e-03 | 2.68412e-04 |    0.027
"""

# CODE

"""
    struct FiniteElementMethod{T<:Real,Bi<:AbsBilliard}

A Finite Element Method (FEM) solver for quantum billiards.

# Fields:
- `billiard::Bi`: The billiard geometry.
- `ℏ::T`: Reduced Planck's constant.
- `m::T`: Particle mass.
- `Nx::Int, Ny::Int`: Number of grid points in x and y directions.
- `Lx::T, Ly::T`: Domain lengths in x and y directions.
- `dx::T, dy::T`: Grid spacing in x and y directions.
- `x_grid::Vector{T}, y_grid::Vector{T}`: Discretized coordinate grids.
- `mask::Vector{Bool}`: Mask indicating valid points inside the billiard.
- `interior_idx::Vector{Int}`: Index map of interior nodes.
- `Q::Int`: Number of interior nodes.
"""
struct FiniteElementMethod{T<:Real,Bi<:AbsBilliard}
    billiard::Bi
    ℏ::T
    m::T
    Nx::Int
    Ny::Int
    Lx::T
    Ly::T
    dx::T
    dy::T
    x_grid::Vector{T}
    y_grid::Vector{T}
    mask::Vector{Bool}
    interior_idx::Vector{Int}
    Q::Int
end

"""
    compute_interior_index(mask::Vector{Bool}) -> Tuple{Vector{Int}, Int}

Computes an index mapping of interior points inside the billiard. This follows the logic of reference: 
Classical and Quantum Chaos in the Diamond Shaped Billiard, R. Salazar (1), G. Téllez (1), D. Jaramillo (1), D. L. González (2) ((1) Departamento de Física, Universidad de los Andes (2) Department of Physics, University of Maryland)
https://doi.org/10.48550/arXiv.1205.4990

# Arguments:
- `mask::Vector{Bool}`: Boolean vector indicating interior points.

# Returns:
- `interior_idx::Vector{Int}`: Indexed mapping of interior points.
- `Q::Int`: Total number of interior points.
"""
function compute_interior_index(mask::Vector{Bool})
    interior_idx=zeros(Int,length(mask));count=0
    for i in 1:length(mask)
        if mask[i]
            count+=1
            interior_idx[i]=count
        end
    end
    return interior_idx,count
end

"""
    compute_boundary(interior_idx::Vector{Ti},Nx::Ti,Ny::Ti) -> Matrix{Bool} where {Ti<:Integer}

Computes the boundary of a domain given the `interior_idx` matrix, which is derived from `compute_interior_index(mask)`. 

A boundary cell is any interior cell (value > 0) that has at least one neighbor that is an exterior cell (value = 0).

# Arguments
- `interior_idx::Vector{Ti}`: A **flattened** 1D vector of size `Nx * Ny`, where each interior node is assigned a unique index (> 0) and exterior nodes are marked as `0`.
- `Nx::Ti`: Number of grid points in the x-direction.
- `Ny::Ti`: Number of grid points in the y-direction.

# Returns
- `boundary::Matrix{Bool}`: A `Nx × Ny` Boolean matrix where `true` marks boundary points and `false` marks non-boundary points.
"""
function compute_boundary(interior_idx::Vector{Ti},Nx::Ti,Ny::Ti) where {Ti<:Integer}
    boundary=falses(Nx, Ny) 
    interior=reshape(interior_idx,Nx,Ny)
    for j in 1:Ny, i in 1:Nx
        if interior[i,j] > 0  # Only check if it's an interior point
            # Check if any neighbor is outside (interior_idx = 0)
            if (i>1 && interior[i-1,j]==0) ||  # Left
               (i<Nx && interior[i+1,j]==0) ||  # Right
               (j>1 && interior[i,j-1]==0) ||  # Below
               (j<Ny && interior[i,j+1]==0)     # Above
                boundary[i,j]=true
            end
        end
    end
    return boundary::Matrix{Bool}
end

"""
    FiniteElementMethod(billiard::Bi, Nx::Int, Ny::Int, b::T; ℏ::T=1.0, m::T=1.0, fundamental=false, k_max=100.0) where {T<:Real, Bi<:AbsBilliard}

Initializes a FEM solver for a quantum billiard.

# Arguments:
- `billiard::Bi`: The billiard geometry.
- `Nx::Int, Ny::Int`: Number of grid points in x and y directions.
- `ℏ::T=1.0`: Reduced Planck's constant.
- `m::T=1.0`: Particle mass.
- `fundamental::Bool=false`: Use fundamental domain (if applicable).
- `k_max::T=100.0`: Wavevector-based scaling parameter.
- `offset_x_symmetric::T=0.1`: Symmetric +/- offset in x direction. This is crucial as in certain methods like φ-FDM we need cells outside the formal interior. Do not set to 0.0
- `offset_y_symmetric::T=0.1`: Symmetric +/- offset in y direction. This is crucial as in certain methods like φ-FDM we need cells outside the formal interior. Do not set to 0.0

# Returns:
- `FiniteElementMethod` instance with computed grid and index mapping.
"""
function FiniteElementMethod(billiard::Bi,Nx::Int,Ny::Int;ℏ::T=1.0,m::T=1.0,fundamental=false,k_max=100.0,offset_x_symmetric=0.1,offset_y_symmetric=0.1) where {T<:Real,Bi<:AbsBilliard}
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    L=billiard.length
    typ=eltype(L)
    xlim::Tuple{typ,typ},ylim::Tuple{typ,typ}=boundary_limits(boundary;grd=max(1000,round(Int,k_max*L*5.0/(2*pi)))) # b=5.0, not needed here
    xlim=(xlim[1]-offset_x_symmetric,xlim[2]+offset_x_symmetric)
    ylim=(ylim[1]-offset_y_symmetric,ylim[2]+offset_y_symmetric)
    Lx,Ly=xlim[2]-xlim[1],ylim[2]-ylim[1] # domain lengths in x and y
    dx,dy=Lx/(Nx-1),Ly/(Ny-1)
    nx,ny=Nx,Ny
    x_grid,y_grid=collect(typ,range(xlim..., nx)),collect(typ,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental)
    interior_idx,Q=compute_interior_index(mask)
    return FiniteElementMethod(billiard,ℏ,m,Nx,Ny,Lx,Ly,dx,dy,x_grid,y_grid,mask,interior_idx,Q)
end

# NEEDS PRECONDITIONING
"""
    FEM_Hamiltonian(fem::FiniteElementMethod{T}) -> SparseMatrixCSC{T,Int}

Constructs the free FEM Hamiltonian matrix using the 2D Laplacian. If optionally V::Matrix is supplied then it incorporates the potential energy into the Hamiltonian matrix. 
The Hamiltonian is symmetric either way since the potential only comes as digonal terms in the final matrix.

# Arguments:
- `fem::FiniteElementMethod{T}`: The FEM instance.

# Returns:
- `H::SparseMatrixCSC{T,Int}`: The Hamiltonian matrix.
"""
function FEM_Hamiltonian(fem::FiniteElementMethod{T}) where {T<:Real}
    ℏ,m=fem.ℏ,fem.m;A_const=ℏ^2/(2m)
    dx²,dy²=fem.dx^2,fem.dy^2;Nx,Ny=fem.Nx,fem.Ny
    rows,cols,vals=Int[],Int[],T[]

    for i in 2:(Nx-1),j in 2:(Ny-1)
        α=fem.interior_idx[i + (j-1) * Nx];if α==0;continue;end
        for (ni,nj,factor) in [(i+1,j,-A_const/dx²),(i-1,j,-A_const/dx²),
                               (i,j+1,-A_const/dy²),(i,j-1,-A_const/dy²)]
            β=fem.interior_idx[ni + (nj-1) * Nx]
            if β!=0
                push!(rows,α);push!(cols,β);push!(vals,factor)
            end
        end
        push!(rows,α);push!(cols,α);push!(vals,A_const*(2/dx²+2/dy²))
    end
    return sparse(rows,cols,vals,fem.Q,fem.Q)
end

# NEEDS PRECONDITIONING
function FEM_Hamiltonian(fem::FiniteElementMethod{T},V::Matrix{T}) where {T<:Real}
    ℏ,m=fem.ℏ,fem.m;A_const=ℏ^2/(2m)
    dx²,dy²=fem.dx^2,fem.dy^2;Nx,Ny=fem.Nx,fem.Ny
    rows,cols,vals=Int[],Int[],T[]

    for i in 2:(Nx-1),j in 2:(Ny-1)
        α=fem.interior_idx[i + (j-1) * Nx];if α==0;continue;end
        for (ni,nj,factor) in [(i+1,j,-A_const/dx²),(i-1,j,-A_const/dx²),
                               (i,j+1,-A_const/dy²),(i,j-1,-A_const/dy²)]
            β=fem.interior_idx[ni + (nj-1) * Nx]
            if β!=0
                push!(rows,α);push!(cols,β);push!(vals,factor)
            end
        end
        push!(rows,α);push!(cols,α);push!(vals,A_const*(2/dx²+2/dy²)+V[i,j])
    end
    return sparse(rows,cols,vals,fem.Q,fem.Q)
end

"""
    compute_fem_eigenmodes(fem::FiniteElementMethod{T}; nev::Int=100, maxiter=100000, tol=1e-8) -> Tuple{Vector{T}, Vector{Matrix{T}}}

Computes the lowest `nev` eigenvalues and wavefunctions of the FEM Hamiltonian. It uses the standard 5 point stencil

# Arguments:
- `fem::FiniteElementMethod{T}`: The FEM instance.
- `nev::Int=100`: Number of eigenvalues to compute.
- `maxiter::Int=100000`: Maximum iterations for eigensolver. Can be lower since it should not get past the default limit.
- `tol::T=1e-8`: Tolerance for eigensolver convergence.

# Returns:
- `evals::Vector{T}`: Computed eigenvalues (square root for energy).
- `wavefunctions::Vector{Matrix{T}}`: Corresponding wavefunctions.
"""
function compute_fem_eigenmodes(fem::FiniteElementMethod{T};nev::Int=100,maxiter=100000,tol=1e-8) where {T<:Real}
    H=FEM_Hamiltonian(fem)
    nev=min(nev,fem.Q-1)  # Prevent requesting more eigenvalues than available
    evals,evecs=eigs(Symmetric(H),nev=nev,which=:SR,tol=tol,maxiter=maxiter)
    wavefunctions=[zeros(T,fem.Nx,fem.Ny) for _ in 1:nev]
    Threads.@threads for i in 1:fem.Nx
        for j in 1:fem.Ny
            α=fem.interior_idx[i+(j-1)*fem.Nx] 
            if α>0  # Ensure only interior points are accessed
                for n in 1:nev
                    wavefunctions[n][i,j]=evecs[α,n]
                end
            end
        end
    end
    return evals,wavefunctions
end

"""
    compute_boundary_tension(ψ::Matrix{T}, boundary_mask::Matrix{Bool}) -> T

Computes the boundary tension of a wavefunction `ψ` by summing its squared magnitude only at the boundary points. 
It's purpose is to gauge the "badness" of the resulting wavefunction. Ideally should be 0.0.

# Arguments
- `ψ::Matrix{T}`: A Nx × Ny matrix representing the wavefunction over the computational domain -> from `compute_fem_eigenmodes`
- `boundary_mask::Matrix{Bool}`: `Matrix` of the same size as `ψ`, where `true` marks boundary points and `false` marks interior/exterior points.

# Returns
- `T`: boundary tension, computed as `∑ |ψ(i, j)|²` over the boundary.
"""
function compute_boundary_tension(ψ::Matrix{T},boundary_mask::Matrix{Bool})
    ψ.=ψ./norm(ψ)
    return sum(abs2.(ψ)[boundary_mask])  # Sum |ψ|² only at boundary points
end