module QuantumBilliards
#using Reexport

#abstract types
include("abstracttypes.jl")
include("utils/macros.jl")

#utils must be included here so modules work
# utils
include("utils/coordinatesystems.jl")
include("utils/geometryutils.jl")
include("utils/billiardutils.jl")
include("utils/typeutils.jl")
include("utils/gridutils.jl")
include("utils/symmetry.jl")
include("utils/boyd.jl")
export companion_matrix, interval_roots_boyd, subdivide_intervals
export Reflection, XReflection, YReflection, XYReflection
export real_length, is_inside

#solvers
include("billiards/geometry/linesegment.jl")
include("billiards/geometry/circlesegment.jl")
include("billiards/geometry/dispersingcirclesegment.jl")
include("billiards/geometry/polarsegment.jl")
include("billiards/geometry/geometry.jl")
include("solvers/samplers.jl")
export GaussLegendreNodes, LinearNodes, FourierNodes, PolarSampler
export sample_points

# boundary
include("billiards/boundarypoints.jl")
export BoundaryPoints
export boundary_coords, dilated_boundary_points

#basis
include("basis/planewaves/realplanewaves.jl")
export RealPlaneWaves
include("basis/fourierbessel/corneradapted.jl")
export CornerAdaptedFourierBessel
export resize_basis, basis_fun, dk_fun, gradient, basis_and_gradient 
include("basis/evanescent/evanescent_pw.jl")
export EvanescentPlaneWaves
include("basis/compositebasis.jl")
export CompositeBasis

#billiards
export LineSegment, VirtualLineSegment
export CircleSegment, VirtualCircleSegment
export DispersingCircleSegment, VirtualDispersingCircleSegment
export PolarSegment, VirtualPolarSegment, PolarSegments
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
include("billiards/equilateraltriangle.jl")
include("billiards/generalized_sinai.jl")
include("billiards/teardrop.jl")
include("billiards/star.jl")
include("billiards/ellipse_mushroom.jl")
export adapt_basis
export Stadium, Lemon, Triangle, Sinai
export curve, tangent, arc_length
export tangent_vec, normal_vec

#convenience functions
export CircleBilliard, make_quarter_circle, make_circle, make_circle_and_basis
export Ellipse, make_quarter_ellipse, make_full_ellipse, make_ellipse_and_basis
export RobnikBilliard, make_half_robnik, make_full_robnik, make_robnik_and_basis
export ProsenBilliard, make_quarter_prosen, make_full_prosen, make_prosen_and_basis
export Mushroom, make_half_mushroom, make_full_mushroom, make_mushroom_and_basis
export RectangleBilliard, make_quarter_rectangle, make_full_rectangle, make_rectangle_and_basis
export Stadium, make_quarter_stadium, make_full_stadium, make_stadium_and_basis
export EquilateralTriangleBilliard, make_fundamental_equilateral_triangle, make_full_equilateral_triangle, make_equilateral_triangle_and_basis
export adapt_basis, triangle_corners, make_triangle_and_basis
export GeneralizedSinai, make_quarter_generalized_sinai, make_full_boundary_generalized_sinai, make_desymmetrized_full_generalized_sinai, make_generalized_sinai_and_basis

include("solvers/decompositions.jl")
include("solvers/matrixconstructors.jl")
export basis_matrix, basis_and_gradient_matrices, dk_matrix

include("solvers/acceleratedmethods/scalingmethod.jl")
include("solvers/acceleratedmethods/acceleratedmethods.jl")


include("solvers/sweepmethods/particularsolutionsmethod.jl")
include("solvers/sweepmethods/decompositionmethod.jl")
include("solvers/sweepmethods/boundaryintegralmethod.jl")
include("solvers/sweepmethods/expanded_boundary_integral_method.jl")
include("solvers/sweepmethods/sweepmethods.jl")
include("solvers/sweepmethods/cfie.jl")
export ScalingMethodA, ScalingMethodB 
export DecompositionMethod, ParticularSolutionsMethod
export BoundaryPointsSM, BoundaryPointsDM
export evaluate_points, construct_matrices, construct_matrices_benchmark
export solve, solve_vect
export solve_wavenumber, solve_spectrum
export k_sweep, refine_minima
export BoundaryIntegralMethod, compute_kernel_matrix, fredholm_matrix, solve_timed, AbstractHankelBasis
export ExpandedBoundaryIntegralMethod, default_helmholtz_kernel_derivative_matrix, default_helmholtz_kernel_second_derivative_matrix, compute_kernel_der_matrix, fredholm_matrix_der

#spectra
include("states/eigenstates.jl")
include("spectra/spectralutils.jl")
include("spectra/unfolding.jl")
export weyl_law
export compute_spectrum, compute_spectrum_optimized, compute_spectrum_with_state_optimized
export curvature_correction, corner_correction, curvature_and_corner_corrections, dos_weyl, k_at_state
include("spectra/spectralStatistics.jl")
export number_variance, plot_subtract_level_counts_from_weyl, probability_berry_robnik, cumulative_berry_robnik, compare_level_count_to_weyl, plot_nnls, plot_cumulative_spacing_distribution, plot_subtract_level_counts_from_weyl, plot_point_distribution!, plot_length_spectrum!, length_spectrum, probability_berry_robnik_brody, cumulative_berry_robnik_brody, plot_U_diff

#states
include("states/basisstates.jl")
include("states/randomstates.jl")
export Eigenstate, EigenstateBundle, BasisState, GaussianRandomState
export compute_eigenstate, compute_eigenstate_bundle
include("states/wavefunctions.jl")
include("states/boundaryfunctions.jl")
include("states/husimifunctions.jl")
include("states/wavefunctions_from_boundary.jl")
include("states/momentumstates.jl")
export wavefunction
export boundary_function, momentum_function, husimi_function, husimi_functions_from_us_and_boundary_points, husimiAtPoint_LEGACY, husimiOnGrid_LEGACY, husimiOnGrid, husimi_functions_from_us_and_boundary_points_FIXED_GRID
export billiard_polygon
export wavefunctions
export read_boundary_function, save_boundary_function!
export save_husimi_functions!, load_husimi_functions, save_vec_from_StateData!, load_vec_from_file, save_BoundaryPoints!, read_BoundaryPoints
export ϕ, wavefunction_multi, wavefunction_multi_with_husimi, plot_wavefunctions, plot_wavefunctions_with_husimi
export momentum_representation_of_wavefunction, computeRadiallyIntegratedDensity, computeAngularIntegratedMomentumDensity

#plotting functions in Makie
include("plotting/plottingmakie.jl")
export plot_curve!, plot_boundary!, plot_boundary_orientation!, plot_symmetry_adapted_boundary
export plot_domain_fun!, plot_domain!
export plot_wavefunction!, plot_wavefunction_gradient!, plot_probability!
export plot_boundary_function!, plot_momentum_function!, plot_husimi_function!
export plot_heatmap!, plot_heatmap_balaced!
export plot_angularly_integrated_density!, plot_radially_integrated_density!, plot_momentum_representation_polar!, plot_momentum_cartesian_representation!
include("plotting/testplotting.jl")
export  plot_geometry_test!, plot_basis_test!, plot_solver_test!, plot_state_test!, plot_matrix!, plot_mean_level_spacing!

include("utils/benchmarkutils.jl")
export BenchmarkInfo
export benchmark_solver, compute_benchmarks

include("plotting/benchmarkplotting.jl")
export plot_benchmarks!
include("utils/savingutils.jl")
export save_numerical_ks_and_tensions!, read_numerical_ks_and_tensions, compute_and_save_closest_pairs!, save_numerical_ks!, read_numerical_ks, filter_and_save_ks_and_tensions!, plot_Z!, dynamical_solver_construction, save_ks_and_husimi_functions!, read_ks_and_husimi_functions
include("utils/rotationutils.jl")
include("spectra/m_index.jl")
export visualize_overlap, compute_M, shift_s_vals_poincare_birkhoff, classical_phase_space_matrix, visualize_quantum_classical_overlap_of_levels!, plot_hist_M_distribution!, compute_overlaps, separate_ks_by_classical_indices, fraction_of_mixed_states, get_mixed_states, coefficient_of_fraction_of_mixed_eigenstates_vs_k, plot_fraction_of_mixed_eigenstates_vs_k, separate_Hs_by_classical_indices, separate_by_classical_indices, separate_ks_and_Hs_by_classical_indices, compute_fractions_of_mixed_eigenstates, plot_fraction_mixed_states, visualize_husimi_and_wavefunction!, save_separation_parameters!, load_separation_parameters, get_mixed_states_boolean_mask
include("spectra/gap_ratios.jl")
export P_chaotic, P_integrable, P_r_normalized, plot_gap_ratios, average_gap_ratio, plot_average_r_vs_parameter!
export compute_spectrum_with_state, StateData, solve_state_data_bundle, husimi_functions_from_boundary_functions, husimi_functions_from_StateData, match_wavenumbers_with_X, overlap_and_merge_state!
include("spectra/localization_entropy.jl")
export localization_entropy, normalized_inverse_participation_ratio_R, plot_P_localization_entropy_pdf!, P_localization_entropy_pdf_data, fit_P_localization_entropy_to_beta, heatmap_M_vs_A_2d, heatmap_R_vs_A_2d, combined_heatmaps_with_husimi, correlation_matrix, correlation_matrix_and_average
include("spectra/reccurence.jl")
export S, plot_S_heatmap!, plot_S_heatmaps!
include("spectra/classical_transport_time.jl")
export generate_p_0_chaotic_init_conditions, calculate_p2_averages, convert_p_0_chaotic_init_conditions_to_cartesian, simulate_trajectories, plot_p2_stats!, generate_intervals_from_limits
include("spectra/antiscar_mushroom.jl")
export calculate_bb_bbox_localization_mushroom, create_husimi_localization_mat, calculate_overlap, get_bb_localization_indexes
include("solvers/gridmethods/fdm.jl")
export FiniteElementMethod, compute_interior_index, FEM_Hamiltonian, compute_fem_eigenmodes, compute_boundary, compute_boundary_tension
include("spectra/evolution.jl")
export Wavepacket, gaussian_wavepacket_2d, gaussian_coefficients, plot_gaussian_from_eigenfunction_expansion, evolution_gaussian_coefficients, animate_wavepacket_evolution!
include("spectra/otoc.jl")
export plot_wavefunctions, wavefunction_multi, X_mn_standard, X_standard, B_standard, microcanocinal_Cn_standard, plot_microcanonical_Cn!, microcanonical_Cn_no_wavepacket, plot_microcanonical_Cn_no_wavepacket!
include("solvers/gridmethods/phi_fdm.jl")
export compute_extended_index, phiFD_Hamiltonian, compute_ϕ_fem_eigenmodes
end