module QuantumBilliards

using CairoMakie
using FFTW
using Bessels
using SpecialFunctions
using LinearAlgebra
using SparseArrays
using Arpack
using Random
using ForwardDiff
using QuadGK
using FastGaussQuadrature
using Optim
using StatsBase
using ProgressMeter
using BenchmarkTools
using CSV
using DataFrames
using PrettyTables
using JLD2
using LsqFit
using MKL
using KrylovKit
using LinearMaps
using PyCall
using CoordinateTransformations

#abstract types
include("abstracttypes.jl")
export CoordinateSystem,AbsBilliard,AbsCurve,AbsVirtualCurve,AbsRealCurve,AbsBasis,AbsFundamentalBasis,AbsSolver,AbsPoints,SweepSolver,AcceleratedSolver,AbsSampler,AbsState,StationaryState,AbsGrid,AbsSymmetry,AbsObservable,CFIE
include("utils/macros.jl")
export use_threads,benchit,blas_multi,blas_1,blas_multi_then_1,svd_or_det_solve,try_MKL_on_x86_64!

# utils
include("utils/coordinatesystems.jl")
export CartesianCS,PolarCS,polar_to_cartesian,cartesian_to_polar,_polar_coords!
include("utils/geometryutils.jl")
export angle
include("utils/billiardutils.jl")
export real_length,virtual_length,curve_edge_lengths,is_inside,boundary_limits
include("utils/typeutils.jl") # dont export anything from this file, it's just for internal type utilities
include("utils/special_functions_calls.jl")
export H,hankel_pair01

# geometry
include("billiards/geometry/linesegment.jl")
export LineSegment,VirtualLineSegment,curve,arc_length,tangent,tangent_2,domain
include("billiards/geometry/circlesegment.jl")
export CircleSegment,VirtualCircleSegment
include("billiards/geometry/dispersingcirclesegment.jl") # unused, but maybe useful in the future
include("billiards/geometry/polarsegment.jl")
export PolarSegment,VirtualPolarSegment,compute_area
include("billiards/geometry/geometry.jl")
export tangent_vec,normal_vec,curvature,make_polygon,symmetry_accounted_fundamental_boundary_length
include("solvers/samplers.jl")
export GaussLegendreNodes,LinearNodes,FourierNodes,PolarSampler,sample_points,random_interior_points,random_interior_points_polygon
include("billiards/geometry/kress_grading_single_corner.jl") # dont export anything from this file, it's just for internal use in kress grading
include("billiards/geometry/kress_grading_global_multi_corner.jl") # dont export anything from this file, it's just for internal use in kress grading
include("billiards/geometry/alpert_endpoint_grading.jl") # dont export anything from this file, it's just for internal use in alpert endpoint grading
include("billiards/geometry/boundarypoints.jl")
export BoundaryPoints,boundary_matrix_size,boundary_coords,boundary_coords_desymmetrized_full_boundary,kress_R!,kress_R_corner!
export BoundaryPointsCFIE,CFIEPanelArrays,_panel_arrays_cache,component_offsets,component_lengths,CFIEGeomCache,cfie_geom_cache
include("utils/symmetry.jl")
export Reflection,XReflection,YReflection,XYReflection,Rotation,apply_symmetries_to_boundary_points,apply_symmetries_to_boundary_function
export flatten_points,apply_projection!

# basis
include("basis/planewaves/realplanewaves.jl")
export RealPlaneWaves,resize_basis,basis_fun,dk_fun,gradient,basis_and_gradient
include("basis/fourierbessel/corneradapted.jl")
export CornerAdaptedFourierBessel
include("basis/evanescent/evanescent_pw.jl")
export EvanescentPlaneWaves
include("basis/compositebasis.jl")
export CompositeBasis

# billiards
include("billiards/stadium.jl")
include("billiards/lemon.jl")
include("billiards/triangle.jl")
include("billiards/circle.jl")
include("billiards/ellipse.jl")
include("billiards/robnik.jl")
include("billiards/prosen.jl")
include("billiards/mushroom.jl")
include("billiards/rectangle.jl")
include("billiards/equilateraltriangle.jl")
include("billiards/generalized_sinai.jl")
include("billiards/teardrop.jl")
include("billiards/star.jl")
include("billiards/ellipse_mushroom.jl")
include("billiards/c3.jl")
include("billiards/africa.jl")
include("billiards/circle_with_circle_holes.jl")
include("billiards/star_with_hole.jl")
include("billiards/stadium_within_stadium.jl")
include("billiards/elliptic_flower_with_hole.jl")
# convenience billiard functions
export CircleBilliard,make_quarter_circle,make_circle,make_circle_and_basis
export Ellipse,make_quarter_ellipse,make_full_ellipse,make_ellipse_and_basis
export RobnikBilliard,make_half_robnik,make_full_robnik,make_robnik_and_basis
export ProsenBilliard,make_quarter_prosen,make_full_prosen,make_prosen_and_basis
export Mushroom,make_half_mushroom,make_full_mushroom,make_mushroom_and_basis
export RectangleBilliard,make_quarter_rectangle,make_full_rectangle,make_rectangle_and_basis
export Stadium,make_quarter_stadium,make_full_stadium,make_stadium_and_basis
export EquilateralTriangleBilliard,make_fundamental_equilateral_triangle,make_full_equilateral_triangle,make_equilateral_triangle_and_basis
export adapt_basis,triangle_corners,make_triangle_and_basis
export GeneralizedSinai,make_quarter_generalized_sinai,make_full_boundary_generalized_sinai,make_desymmetrized_full_generalized_sinai,make_generalized_sinai_and_basis
export TeardropBilliard,make_teardrop_and_basis
export Triangle,make_triangle_and_basis
export StadiumWithOptionalHole,make_stadium_with_optional_hole_and_basis
export EllipseMushroom,make_ellipse_mushroom_and_basis
export CircularHoleBilliard,make_circle_with_holes_and_basis,make_multihole_and_basis,make_annulus_and_basis,MultiHoleBilliard,AnnularBilliard
export EllipticFlowerWithOptionalHole,make_elliptic_flower_and_basis
export StarBilliard,make_star_and_basis

# general matrix helpers
include("solvers/fulldecompositions.jl")
export generalized_eigen,generalized_eigvals,generalized_eigen_all,generalized_eigen_all_LAPACK_LEGACY,generalized_eigen_symmetric_LAPACK_LEGACY
export directsum,adjust_scaling_and_samplers
include("solvers/gen_cholesky_rank_red.jl") # dont export this one - experimental
include("solvers/matrixconstructors.jl")
export filter_matrix!,basis_matrix,gradient_matrices,basis_and_gradient_matrices,dk_matrix
include("states/gradients.jl")

# Sweep methods
include("solvers/sweepmethods/basis_sweep/particularsolutionsmethod.jl")
export ParticularSolutionsMethod,solve_full,solve,solve_vect,solve_INFO,evaluate_points,construct_matrices,construct_matrices_benchmark
include("solvers/sweepmethods/basis_sweep/decompositionmethod.jl")
export DecompositionMethod
include("solvers/sweepmethods/dlp/dlp.jl")
export BoundaryIntegralMethod,default_helmholtz_kernel_matrix,default_helmholtz_kernel_derivative_matrix,default_helmholtz_kernel_second_derivative_matrix,compute_kernel_matrix,compute_kernel_matrix!,compute_kernel_matrix_with_derivatives!
export fredholm_matrix!,fredholm_matrix_with_derivatives!,fredholm_matrix,fredholm_matrix_with_derivatives,construct_matrices!
include("solvers/sweepmethods/dlp/dlp_kress.jl")
export DLPKressWorkspace,DLP_kress,DLP_kress_global_corners,build_dlp_kress_workspace,build_Rmat_dlp_kress
export construct_dlp_matrix!,construct_dlp_split!,construct_fredholm_matrix!,construct_dlp_matrix_derivatives!,construct_fredholm_matrix_derivatives!

# Chebyshev machinery - general
include("chebyshev/chebyshev_core.jl")
export _cheb_clenshaw,_breaks_uniform,_breaks_geometric,_chebfit!
include("chebyshev/chebyshev_point_symmetry.jl")
export estimate_rmin_rmax
include("chebyshev/chebyshev_bessels.jl")
export ChebHankelTableH1x,ChebHankelTableH,ChebJTable,_build_table_h1x!,_build_table_h!,_build_table_j!
export ChebHankelPlanH1x,ChebHankelPlanH,ChebJPlan,plan_h1x,plan_h,plan_j,eval_h1x!,eval_h!
export eval_j!,eval_h1x,eval_h,eval_j,eval_h1x_multi_ks!,eval_h_multi_ks!,eval_j_multi_ks!
export h0_h1_j0_j1_multi_ks_at_r!,h0_h1_multi_ks_at_r!

# CFIE
include("solvers/sweepmethods/cfie/alpert_table.jl")
export AlpertLogRule,alpert_log_rule
include("solvers/sweepmethods/cfie/cfie.jl")
export CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners,CFIE_alpert
include("solvers/sweepmethods/cfie/cfie_kress.jl")
export CFIEKressWorkspace,build_cfie_kress_workspace,build_Rmat_kress,plot_boundary_with_weight_INFO
include("solvers/sweepmethods/cfie/cfie_alpert.jl")
export AlpertPeriodicCache,AlpertSmoothPanelCache,CFIEAlpertWorkspace,build_cfie_alpert_workspace,estimate_cfie_alpert_cheb_rbounds

# Chebyshev machinery - applied to specific kernels
include("chebyshev/chebyshev_euclidian_helmholtz_dlp.jl")
export compute_kernel_matrices_DLP_chebyshev!,assemble_fredholm_matrices!,construct_boundary_matrices!
include("chebyshev/chebyshev_euclidian_helmholtz_dlp_kress.jl")
export DLP_kress_BlockCache,DLPKressBlockSystemCache,build_dlp_kress_block_cache,build_DLP_kress_plans,DLP_H0_H1_J0_J1_BesselWorkspace
export build_dlp_kress_cheb_workspace,construct_dlp_kress_matrices_chebyshev!
include("chebyshev/chebyshev_euclidian_helmholtz_cfie_kress.jl")
export CFIE_kress_BlockCache,CFIEBlockSystemCache,CFIE_H0_H1_J0_J1_BesselWorkspace,build_CFIE_plans_kress,build_cfie_kress_block_caches
export compute_kernel_matrices_CFIE_kress_chebyshev!
include("chebyshev/chebyshev_euclidian_helmholtz_cfie_alpert.jl")
export build_CFIE_plans_alpert,CFIE_alpert_BlockCache,CFIE_H0_H1_BesselWorkspace,build_cfie_alpert_block_caches,CFIEAlpertChebWorkspace,build_cfie_alpert_cheb_workspace,compute_kernel_matrices_CFIE_alpert_chebyshev!,compute_kernel_matrices_CFIE_alpert_chebyshev
include("chebyshev/chebyshev_optimal_panelization.jl")
export chebyshev_params

# Expanded BIM
include("solvers/acceleratedmethods/ebim.jl")
export EBIMSolver,solve_full!,solve_krylov!,solve_full_INFO!,solve_krylov_INFO!,solve!,compute_spectrum_ebim,visualize_ebim_sweep

# Beyn's method
include("solvers/acceleratedmethods/beyn.jl")
export construct_B_matrix,residual_and_norm_select,compute_spectrum_beyn

# General sweep methods high level interface
include("solvers/sweepmethods/sweepmethods.jl")
export solve_wavenumber,k_sweep,refine_minima

# Hyperbolic kernels and solvers
include("Hyperbolic/patched_taylor_series_Q.jl")
include("Hyperbolic/DLP_hyperbolic_helmholtz.jl")
include("Hyperbolic/BIM_hyperbolic.jl")
include("Hyperbolic/sampler_hyperbolic.jl")
include("Hyperbolic/weyl_hyperbolic.jl")
include("Hyperbolic/Beyn_hyperbolic.jl")
include("Hyperbolic/husimi_hyperbolic.jl")
include("Hyperbolic/wavefunction_hyperbolic.jl")

#spectra
include("states/eigenstates.jl")
export Eigenstate,compute_eigenstate,EigenstateBundle,compute_eigenstate_bundle,StateData,solve_state_data_bundle,solve_state_data_bundle_with_INFO

# Vergini-Saraceno
include("solvers/acceleratedmethods/scalingmethod.jl")
export VerginiSaraceno,solve_vectors,solve_spectrum_with_INFO,plot_Z!

include("spectra/unfolding.jl")
export weyl_law,k_at_state

#states
include("states/basisstates.jl")
export BasisState
include("states/randomstates.jl")
export GaussianRandomState
include("states/boundaryfunctions.jl")
export boundary_function,boundary_function_with_points,setup_momentum_density
include("states/husimifunctions.jl")
export husimi_function,husimi_on_grid,husimi_functions_from_StateData,husimi_functions_from_boundary_functions
export husimi_functions_from_us_and_boundary_points,husimi_functions_from_us_and_boundary_points_FIXED_GRID
include("states/wavefunctions.jl")
export ϕ,wavefunction_multi,wavefunction_multi_with_husimi,plot_wavefunctions,plot_wavefunctions_with_husimi
include("states/momentumstates.jl")
export momentum_representation_of_wavefunction,computeRadiallyIntegratedDensity,computeAngularIntegratedMomentumDensity

# localization,evolution and some spectral statistics
include("spectra/m_index.jl")
export visualize_overlap,compute_M,shift_s_vals_poincare_birkhoff,classical_phase_space_matrix,compute_overlaps,separate_regular_and_chaotic_states
include("spectra/gap_ratios.jl")
export P_chaotic,P_integrable,P_r_normalized,plot_gap_ratios,average_gap_ratio,plot_average_r_vs_parameter!
include("spectra/localization_entropy.jl")
export localization_entropy,normalized_inverse_participation_ratio_R,plot_P_localization_entropy_pdf!,P_localization_entropy_pdf_data, fit_P_localization_entropy_to_beta,correlation_matrix,correlation_matrix_and_average
include("solvers/gridmethods/fdm.jl")
export FiniteElementMethod,compute_interior_index,FEM_Hamiltonian,compute_fem_eigenmodes,compute_boundary,compute_boundary_tension
include("spectra/evolution.jl")
export Wavepacket,gaussian_wavepacket_2d,gaussian_coefficients,plot_gaussian_from_eigenfunction_expansion,evolution_gaussian_coefficients, animate_wavepacket_evolution!
include("spectra/otoc.jl")
export plot_wavefunctions,X_mn_standard,X_standard,B_standard,microcanocinal_Cn_standard,plot_microcanonical_Cn!,microcanonical_Cn_no_wavepacket, plot_microcanonical_Cn_no_wavepacket!
include("solvers/gridmethods/phi_fdm.jl")
export compute_extended_index,phiFD_Hamiltonian,compute_ϕ_fem_eigenmodes

end