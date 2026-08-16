# API
The following pages document and explain the functionality of all exported types
and functions in the library.

## Index
```@index
```

## Abstract types
```@docs
```
<!-- TODO: list exported symbols from src/abstracttypes.jl: AbsBasis, AbsSolver -->

## Utilities
```@docs
```
<!-- TODO: list exported symbols from src/utils/: make_triangle_and_basis, adapt_basis -->

## Basis
```@docs
```
<!-- TODO: list exported symbols from src/basis/ (fourierbessel/, planewaves/):
     RealPlaneWaves, CornerAdaptedFourierBessel, resize_basis, basis_fun, dk_fun,
     gradient, basis_and_gradient -->

## Solvers
```@docs
```
<!-- TODO: list exported symbols from src/solvers/ (boundarypoints.jl,
     decompositions.jl, matrixconstructors.jl, acceleratedmethods/, sweepmethods/):
     BoundaryPoints, basis_matrix, basis_and_gradient_matrices, dk_matrix,
     SweepSolver, AcceleratedSolver, VerginiSaracenoSolver, print_benchmark_info,
     DecompositionMethodSolver, ParticularSolutionsMethod, BoundaryPointsSM,
     BoundaryPointsDM, evaluate_points, construct_matrices,
     construct_matrices_benchmark, solve, solve_vect, solve_wavenumber,
     solve_spectrum, k_sweep -->

## Spectra
```@docs
```
<!-- TODO: list exported symbols from src/spectra/ (spectralutils.jl, unfolding.jl):
     SpectralData, compute_spectrum, merge_spectra, overlap_and_merge!, weyl_law -->

## States
```@docs
```
<!-- TODO: list exported symbols from src/states/ (eigenstates.jl, basisstates.jl,
     randomstates.jl, symmetry/, wavefunctions.jl, boundaryfunctions.jl,
     husimifunctions.jl):
     Eigenstate, EigenstateBundle, BasisState, GaussianRandomState,
     compute_eigenstate, compute_eigenstate_bundle, wavefunction, compute_psi,
     boundary_limits, get_boundary_curves_with_ignored, boundary_function,
     momentum_function, husimi_function -->
