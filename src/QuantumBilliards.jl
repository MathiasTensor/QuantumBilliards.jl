module QuantumBilliards
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
export Reflection, XReflection, YReflection, XYReflection
export real_length, is_inside

#solvers
#include("solvers/Solvers.jl")
#@reexport using .Solvers
include("solvers/samplers.jl")
export GaussLegendreNodes, LinearNodes, FourierNodes
export sample_points

include("billiards/boundarypoints.jl")
export BoundaryPoints
export boundary_coords, dilated_boundary_points
#basis
#include("basis/Basis.jl")
#@reexport using .Basis
include("basis/planewaves/realplanewaves.jl")
export RealPlaneWaves
include("basis/fourierbessel/corneradapted.jl")
export CornerAdaptedFourierBessel
export resize_basis, basis_fun, dk_fun, gradient, basis_and_gradient 

#billiards
#include("billiards/Billiards.jl")
#@reexport using .Billiards

include("billiards/geometry/geometry.jl")
export LineSegment, VirtualLineSegment
export CircleSegment, VirtualCircleSegment
export DispersingCircleSegment, VirtualDispersingCircleSegment
export PolarSegment, VirtualPolarSegment
export curve, arc_length, tangent, tangent_vec, normal_vec, curvature, symmetry_accounted_fundamental_boundary_length
include("billiards/stadium.jl")
include("billiards/lemon.jl")
include("billiards/sinai.jl")
include("billiards/triangle.jl")
include("billiards/circle.jl")
include("billiards/ellipse.jl")
include("billiards/robnik.jl")
include("billiards/prosen.jl")
include("billiards/mushroom.jl")
include("billiards/rectangle.jl")
export adapt_basis
#include("limacon.jl")
#include("rectangle.jl")
export Stadium, Lemon, Triangle, Sinai
export curve, tangent, normal, arc_length
export tangent_vec, normal_vec
#convenience functions may be moved somewhere else
#export make_stadium_and_basis, make_triangle_and_basis 
export CircleBilliard, make_quarter_circle, make_circle, make_circle_and_basis
export Ellipse, make_quarter_ellipse, make_full_ellipse, make_ellipse_and_basis
export RobnikBilliard, make_half_robnik, make_full_robnik, make_robnik_and_basis
export ProsenBilliard, make_quarter_prosen, make_full_prosen, make_prosen_and_basis
export Mushroom, make_half_mushroom, make_full_mushroom, make_mushroom_and_basis
export RectangleBilliard, make_quarter_rectangle, make_full_rectangle, make_rectangle_and_basis
export Stadium, make_quarter_stadium, make_full_stadium, make_stadium_and_basis

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
export curvature_correction, corner_correction
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
export billiard_polygon

#plotting functions in Makie
#include("plotting/Plotting.jl")
#@reexport using .Plotting
include("plotting/plottingmakie.jl")
export plot_curve!, plot_boundary!
export plot_domain_fun!, plot_domain!
export plot_lattice!
export plot_wavefunction!, plot_wavefunction_gradient!, plot_probability!
export plot_boundary_function!, plot_momentum_function!, plot_husimi_function!
export plot_heatmap!, plot_heatmap_balaced!
export plot_angularly_integrated_density!, plot_radially_integrated_density!
include("plotting/testplotting.jl")
export  plot_geometry_test!, plot_basis_test!, plot_solver_test!, plot_state_test!, plot_matrix!

include("utils/benchmarkutils.jl")
export BenchmarkInfo
export benchmark_solver, compute_benchmarks

include("plotting/benchmarkplotting.jl")
export plot_benchmarks!
include("utils/savingutils.jl")
export save_numerical_ks_and_tensions!, read_numerical_ks_and_tensions, compute_and_save_closest_pairs!
end