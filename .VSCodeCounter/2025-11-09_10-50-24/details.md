# Details

Date : 2025-11-09 10:50:24

Directory c:\\Users\\d93490\\.julia\\dev\\QuantumBilliards.jl

Total : 46 files,  3361 codes, 345 comments, 590 blanks, all 4296 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [QuantumBilliards.jl/.github/workflows/CI.yml](/QuantumBilliards.jl/.github/workflows/CI.yml) | YAML | 33 | 2 | 1 | 36 |
| [QuantumBilliards.jl/.github/workflows/CompatHelper.yml](/QuantumBilliards.jl/.github/workflows/CompatHelper.yml) | YAML | 16 | 0 | 1 | 17 |
| [QuantumBilliards.jl/.github/workflows/TagBot.yml](/QuantumBilliards.jl/.github/workflows/TagBot.yml) | YAML | 15 | 0 | 1 | 16 |
| [QuantumBilliards.jl/QuantumBilliards.jl.code-workspace](/QuantumBilliards.jl/QuantumBilliards.jl.code-workspace) | JSON with Comments | 17 | 0 | 0 | 17 |
| [QuantumBilliards.jl/README.md](/QuantumBilliards.jl/README.md) | Markdown | 64 | 0 | 10 | 74 |
| [QuantumBilliards.jl/benchmarks/lemonbm.jl](/QuantumBilliards.jl/benchmarks/lemonbm.jl) | Julia | 120 | 21 | 27 | 168 |
| [QuantumBilliards.jl/benchmarks/righttrianglebm.jl](/QuantumBilliards.jl/benchmarks/righttrianglebm.jl) | Julia | 101 | 2 | 32 | 135 |
| [QuantumBilliards.jl/benchmarks/sinaibm.jl](/QuantumBilliards.jl/benchmarks/sinaibm.jl) | Julia | 84 | 12 | 20 | 116 |
| [QuantumBilliards.jl/benchmarks/spectrumbm.jl](/QuantumBilliards.jl/benchmarks/spectrumbm.jl) | Julia | 33 | 0 | 12 | 45 |
| [QuantumBilliards.jl/benchmarks/stadiumbm.jl](/QuantumBilliards.jl/benchmarks/stadiumbm.jl) | Julia | 98 | 5 | 39 | 142 |
| [QuantumBilliards.jl/benchmarks/sweepbm.jl](/QuantumBilliards.jl/benchmarks/sweepbm.jl) | Julia | 68 | 7 | 17 | 92 |
| [QuantumBilliards.jl/benchmarks/test.jl](/QuantumBilliards.jl/benchmarks/test.jl) | Julia | 9 | 4 | 4 | 17 |
| [QuantumBilliards.jl/benchmarks/trianglebm.jl](/QuantumBilliards.jl/benchmarks/trianglebm.jl) | Julia | 127 | 22 | 33 | 182 |
| [QuantumBilliards.jl/src/QuantumBilliards.jl](/QuantumBilliards.jl/src/QuantumBilliards.jl) | Julia | 53 | 13 | 13 | 79 |
| [QuantumBilliards.jl/src/abstracttypes.jl](/QuantumBilliards.jl/src/abstracttypes.jl) | Julia | 9 | 1 | 3 | 13 |
| [QuantumBilliards.jl/src/basis/fourierbessel/corneradapted.jl](/QuantumBilliards.jl/src/basis/fourierbessel/corneradapted.jl) | Julia | 388 | 43 | 69 | 500 |
| [QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl](/QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl) | Julia | 361 | 29 | 43 | 433 |
| [QuantumBilliards.jl/src/solvers/acceleratedmethods/acceleratedmethods.jl](/QuantumBilliards.jl/src/solvers/acceleratedmethods/acceleratedmethods.jl) | Julia | 18 | 4 | 3 | 25 |
| [QuantumBilliards.jl/src/solvers/acceleratedmethods/verginisaraceno.jl](/QuantumBilliards.jl/src/solvers/acceleratedmethods/verginisaraceno.jl) | Julia | 122 | 8 | 19 | 149 |
| [QuantumBilliards.jl/src/solvers/boundarypoints.jl](/QuantumBilliards.jl/src/solvers/boundarypoints.jl) | Julia | 122 | 4 | 13 | 139 |
| [QuantumBilliards.jl/src/solvers/decompositions.jl](/QuantumBilliards.jl/src/solvers/decompositions.jl) | Julia | 222 | 15 | 39 | 276 |
| [QuantumBilliards.jl/src/solvers/matrixconstructors.jl](/QuantumBilliards.jl/src/solvers/matrixconstructors.jl) | Julia | 112 | 18 | 27 | 157 |
| [QuantumBilliards.jl/src/solvers/sweepmethods/decompositionmethod.jl](/QuantumBilliards.jl/src/solvers/sweepmethods/decompositionmethod.jl) | Julia | 99 | 4 | 15 | 118 |
| [QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl](/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl) | Julia | 25 | 0 | 3 | 28 |
| [QuantumBilliards.jl/src/spectra/spectralutils.jl](/QuantumBilliards.jl/src/spectra/spectralutils.jl) | Julia | 154 | 34 | 24 | 212 |
| [QuantumBilliards.jl/src/spectra/unfolding.jl](/QuantumBilliards.jl/src/spectra/unfolding.jl) | Julia | 17 | 0 | 6 | 23 |
| [QuantumBilliards.jl/src/states/basisstates.jl](/QuantumBilliards.jl/src/states/basisstates.jl) | Julia | 17 | 3 | 3 | 23 |
| [QuantumBilliards.jl/src/states/boundaryfunctions.jl](/QuantumBilliards.jl/src/states/boundaryfunctions.jl) | Julia | 79 | 6 | 6 | 91 |
| [QuantumBilliards.jl/src/states/eigenstates.jl](/QuantumBilliards.jl/src/states/eigenstates.jl) | Julia | 80 | 3 | 8 | 91 |
| [QuantumBilliards.jl/src/states/gradients.jl](/QuantumBilliards.jl/src/states/gradients.jl) | Julia | 70 | 22 | 7 | 99 |
| [QuantumBilliards.jl/src/states/husimifunctions.jl](/QuantumBilliards.jl/src/states/husimifunctions.jl) | Julia | 72 | 8 | 7 | 87 |
| [QuantumBilliards.jl/src/states/randomstates.jl](/QuantumBilliards.jl/src/states/randomstates.jl) | Julia | 13 | 4 | 2 | 19 |
| [QuantumBilliards.jl/src/states/wavefunctions.jl](/QuantumBilliards.jl/src/states/wavefunctions.jl) | Julia | 153 | 19 | 13 | 185 |
| [QuantumBilliards.jl/src/utils/billiardutils.jl](/QuantumBilliards.jl/src/utils/billiardutils.jl) | Julia | 42 | 4 | 7 | 53 |
| [QuantumBilliards.jl/src/utils/coordinatesystems.jl](/QuantumBilliards.jl/src/utils/coordinatesystems.jl) | Julia | 94 | 9 | 18 | 121 |
| [QuantumBilliards.jl/src/utils/geometryutils.jl](/QuantumBilliards.jl/src/utils/geometryutils.jl) | Julia | 1 | 0 | 0 | 1 |
| [QuantumBilliards.jl/src/utils/gridutils.jl](/QuantumBilliards.jl/src/utils/gridutils.jl) | Julia | 83 | 8 | 18 | 109 |
| [QuantumBilliards.jl/src/utils/macros.jl](/QuantumBilliards.jl/src/utils/macros.jl) | Julia | 70 | 5 | 10 | 85 |
| [QuantumBilliards.jl/src/utils/symmetry.jl](/QuantumBilliards.jl/src/utils/symmetry.jl) | Julia | 45 | 1 | 5 | 51 |
| [QuantumBilliards.jl/src/utils/typeutils.jl](/QuantumBilliards.jl/src/utils/typeutils.jl) | Julia | 4 | 1 | 1 | 6 |
| [QuantumBilliards.jl/test/basistests.jl](/QuantumBilliards.jl/test/basistests.jl) | Julia | 19 | 1 | 1 | 21 |
| [QuantumBilliards.jl/test/billiardtests.jl](/QuantumBilliards.jl/test/billiardtests.jl) | Julia | 24 | 1 | 3 | 28 |
| [QuantumBilliards.jl/test/runtests.jl](/QuantumBilliards.jl/test/runtests.jl) | Julia | 4 | 1 | 2 | 7 |
| [QuantumBilliards.jl/test/solvertests.jl](/QuantumBilliards.jl/test/solvertests.jl) | Julia | 4 | 1 | 3 | 8 |
| [QuantumBilliards.jl/test/spectrumtests.jl](/QuantumBilliards.jl/test/spectrumtests.jl) | Julia | 0 | 0 | 1 | 1 |
| [QuantumBilliards.jl/test/statetests.jl](/QuantumBilliards.jl/test/statetests.jl) | Julia | 0 | 0 | 1 | 1 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)