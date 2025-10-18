module QuantumBilliards
using Bessels
using CoordinateTransformations, Rotations
using LinearAlgebra, StaticArrays, CircularArrays
using Optim
using FFTW
using TimerOutputs
using Random, Distributions
using BilliardGeometry
#using Reexport

#abstract types
include("abstracttypes.jl")
#utils must be included here so modules work
#export AbsBasis
include("utils/coordinatesystems.jl")
include("utils/geometryutils.jl")
include("utils/billiardutils.jl")
include("utils/typeutils.jl")
include("utils/gridutils.jl")
include("utils/symmetry.jl")
export real_length, is_inside


#basis
#include("basis/Basis.jl")
#@reexport using .Basis
include("basis/planewaves/realplanewaves.jl")
export RealPlaneWaves
include("basis/fourierbessel/corneradapted.jl")
export CornerAdaptedFourierBessel
export resize_basis, basis_fun, dk_fun, gradient, basis_and_gradient 


include("solvers/decompositions.jl")
include("solvers/matrixconstructors.jl")
export basis_matrix, basis_and_gradient_matrices, dk_matrix

include("solvers/acceleratedmethods/acceleratedmethods.jl")
include("solvers/sweepmethods/sweepmethods.jl")
export ScalingMethodA, ScalingMethodB 
export DecompositionMethod, ParticularSolutionsMethod
export BoundaryPointsSM, BoundaryPointsDM
export evaluate_points, construct_matrices, construct_matrices_benchmark
export solve, solve_vect
export solve_wavenumber, solve_spectrum
export k_sweep

#spectra
#include("spectra/Spectra.jl")
#@reexport using .Spectra

include("spectra/spectralutils.jl")
include("spectra/unfolding.jl")
export weyl_law
export compute_spectrum
#states
#include("states/States.jl")
#@reexport using .States
include("states/eigenstates.jl")
include("states/basisstates.jl")
include("states/randomstates.jl")

export Eigenstate, EigenstateBundle, BasisState, GaussianRandomState
export compute_eigenstate, compute_eigenstate_bundle

include("states/wavefunctions.jl")
include("states/boundaryfunctions.jl")
include("states/husimifunctions.jl")

export wavefunction #wavefunction_norm 
export boundary_function, momentum_function, husimi_function


end