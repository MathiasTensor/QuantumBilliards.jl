using Test, QuantumBilliards, BilliardGeometry, LinearAlgebra, Random

# These tests check the physical normal density q, not the legacy combined-
# potential density μ.  The two CFIE formulations have the same Dirichlet
# spectrum but different unknowns and matrix entries.  Closed-form quadrature
# moments and the circular Bessel eigenfunction supply independent checks;
# mushroom tests additionally retain both reflection sectors and every root.
const WCT_QB = QuantumBilliards

@testset "Weighted complex Bessel sequences preserve values and failures" begin
    # Compare against the dependency's independent single-order public API,
    # including small arguments, both sides of the real axis, large arguments
    # with AMOS accuracy warnings, underflow, overflow and invalid inputs.
    arguments = ComplexF64[complex(realpart,imaginary) for
        realpart in (1e-14,1e-8,0.1,1.0,10.0,100.0,1000.0,1e6)
        for imaginary in (-0.2,0.0,0.2)]
    append!(arguments,ComplexF64[0,1+1000im,1-1000im,1e10+0.2im])
    for kind in (Val(:hankel),Val(:besselj)), argument in arguments
        expected = try
            kind === Val(:hankel) ?
                (WCT_QB.SpecialFunctions.hankelh1(0,argument),WCT_QB.SpecialFunctions.hankelh1(1,argument)) :
                (WCT_QB.SpecialFunctions.besselj(0,argument),WCT_QB.SpecialFunctions.besselj(1,argument))
        catch err
            err
        end
        observed = try
            WCT_QB._weighted_corner_bessel_pair(kind,argument)
        catch err
            err
        end
        if expected isa Exception
            @test typeof(observed) === typeof(expected)
            expected isa WCT_QB.SpecialFunctions.AmosException &&
                @test observed.id == expected.id
        else
            @test observed isa Tuple{ComplexF64,ComplexF64}
            if observed isa Tuple
                @test isapprox(observed[1],expected[1];rtol=2e-14,atol=0)
                @test isapprox(observed[2],expected[2];rtol=2e-14,atol=0)
            end
        end
    end
end

@testset "Weighted integration dispatch and representation guards" begin
    # A solver's grading selects its density representation. A weighted
    # physical q mesh must never enter a periodic combined-layer mu formula.
    @test isempty(Test.detect_ambiguities(WCT_QB;recursive=false))
    billiard = CircleBilliard(1.0)
    periodic = CombinedFieldIntegralEquationSolver(8.0,billiard;min_pts=64)
    weighted = CombinedFieldIntegralEquationSolver(8.0,billiard;
        min_pts=64,grading=WeightedCornerGrading())
    ordinary_points = evaluate_points(periodic,billiard,4.0)
    weighted_points = evaluate_points(weighted,billiard,4.0)
    @test_throws ArgumentError boundary_matrix_size(periodic,weighted_points)
    @test_throws ArgumentError construct_matrices(periodic,weighted_points,4.0;
        multithreaded=false)
    @test_throws ArgumentError construct_matrices(weighted,ordinary_points,4.0;
        multithreaded=false)
    @test_throws ArgumentError WCT_QB._construct_matrices_multi_k_taylor(periodic,
        weighted_points,4.0,ComplexF64[4.01+0.02im],4;multithreaded=false)
    @test_throws ArgumentError WCT_QB.build_cork_polynomial(periodic,
        weighted_points,4.0,0.15,4;multithreaded=false)
    @test_throws ArgumentError WCT_QB._ebim_construct_matrices(periodic,
        weighted_points,4.0;multithreaded=false)

    # These optional accelerators must dispatch to the same physical operator,
    # without edits to the original periodic Chebyshev/Taylor implementations.
    config = ChebyshevConfig(Float64)
    @test WCT_QB.tune_cheb_config(weighted,weighted_points,4.0,config) === config
    z = 4.01+0.02im
    direct = construct_matrices(weighted,weighted_points,z;multithreaded=false)
    from_chebyshev_api = only(WCT_QB._construct_matrices_multi_k_cheb(weighted,
        weighted_points,ComplexF64[z],config;multithreaded=false))
    @test norm(from_chebyshev_api-direct)/norm(direct) < 3e-13
    expanded = ExpandedBIMSolver(weighted;use_taylor=true)
    cache = WCT_QB.EBIMTaylorCache(weighted,weighted_points,4.0,4;
        multithreaded=false)
    cached = construct_matrices(expanded,cache,4.0)
    coefficients = WCT_QB._weighted_corner_matrices(weighted,weighted_points,4.0,2;
        multithreaded=false)
    for (observed,expected) in zip(cached,
        (coefficients[1],coefficients[2],2coefficients[3]))
        @test norm(observed-expected)/norm(expected) < 3e-13
    end
end

@testset "Weighted corner quadrature moments" begin
    for exponent in (-1/3, 0.0, 1.0, 2.5)
        x, w = WCT_QB._weighted_corner_gauss_rule(16, exponent)
        @test all(>(0), w)
        for degree in (0, 1, 5, 12)
            @test isapprox(sum(w .* x.^(exponent+degree)),
                inv(exponent+degree+1); atol=2e-14, rtol=2e-13)
        end
    end
    x, w = WCT_QB._weighted_corner_gauss_rule(24)
    logarithmic = WCT_QB._weighted_corner_log_weights(x, w)
    for degree in 0:12
        @test isapprox(sum(logarithmic .* x.^degree),
            -inv((degree+1)^2); atol=2e-14, rtol=2e-12)
    end
end

@testset "Weighted Gaussian families retain caller ownership" begin
    for exponent in (0.0, 1.0, 2.5)
        nodes, weights = WCT_QB._weighted_corner_gauss_rule(24,exponent)
        expected_nodes, expected_weights = copy(nodes), copy(weights)
        fill!(nodes,NaN); fill!(weights,0)
        fresh_nodes, fresh_weights = WCT_QB._weighted_corner_gauss_rule(24,exponent)
        @test fresh_nodes == expected_nodes
        @test fresh_weights == expected_weights
        @test fresh_nodes !== nodes && fresh_weights !== weights
    end

    # Rule families are shared across physical charts. Reversing a right
    # corner or constructing logarithmic weights must not mutate a cached
    # left-oriented rule, including when task schedules interleave.
    billiard = MushroomBilliard(0.4,1.0)
    kernel = CombinedFieldIntegralEquationSolver(6.0,billiard;
        min_pts=64,grading=WeightedCornerGrading())
    space = evaluate_points(kernel,billiard,6.0).density_space
    left = first(filter(p->p.lambda!=1 && !p.reverse && p.tleft==0,space.panels))
    right = first(filter(p->isapprox(p.lambda,left.lambda;atol=1e-12) &&
        p.reverse && p.tright==1,space.panels))
    reference = WCT_QB._weighted_corner_panel_rule(space,left,left.tleft,left.tright,24)
    basis = copy(reference.basis); parameters = copy(reference.parameters)
    tasks = map(1:8) do index
        Threads.@spawn begin
            panel = isodd(index) ? left : right
            rule = WCT_QB._weighted_corner_panel_rule(space,panel,
                panel.tleft,panel.tright,24)
            a = panel.tleft+(panel.tright-panel.tleft)/3
            b = panel.tleft+2(panel.tright-panel.tleft)/3
            logged = WCT_QB._weighted_corner_panel_rule(space,panel,a,b,24;
                logarithmic=true,target_at_right=iseven(index))
            (sum(rule.basis),sum(logged.logbasis),all(isfinite,logged.logbasis))
        end
    end
    observations = fetch.(tasks)
    @test all(record->record[3],observations)
    @test all(observations[i] == observations[1] for i in 1:2:8)
    @test all(observations[i] == observations[2] for i in 2:2:8)
    repeated = WCT_QB._weighted_corner_panel_rule(space,left,left.tleft,left.tright,24)
    @test repeated.basis == basis
    @test repeated.parameters == parameters
    @test reference.basis == basis
end

@testset "Weighted mushroom geometry and density cardinals" begin
    billiard = MushroomBilliard(0.4, 1.0; R=1.0)
    kernel = CombinedFieldIntegralEquationSolver(8.0, billiard;
        min_pts=64, grading=WeightedCornerGrading())
    points = evaluate_points(kernel, billiard, 10.0)
    space = points.density_space
    @test space !== nothing
    @test length(space.curves) == length(full_boundary(billiard)) == 8
    @test all(curve -> curve.bc isa SpecularReflection, space.curves)
    @test all(>(0), points.ds)
    @test all(>(0), space.norm_weights)
    @test all(x -> all(isfinite,x), points.xy)
    @test length(space.node_panel) == length(points)
    @test any(panel -> isapprox(panel.lambda, 2/3; atol=1e-12), space.panels)
    @test any(panel -> isapprox(panel.lambda, 2; atol=1e-12), space.panels)

    # On a straight incident edge, q(t)=ω(t)(1+2ξ-ξ²) has an elementary
    # integral.  This catches a missing/doubled Jacobi endpoint factor and
    # accidental interpolation of q instead of its smooth amplitude v.
    checked = 0
    for panel in space.panels
        curve = space.curves[panel.curve_index]
        panel.lambda == 1 && continue
        curve isa BilliardGeometry.LineSegment || continue
        ξ = panel.reverse ? 1 .- panel.nodes : panel.nodes
        density = panel.density_factors .* (1 .+ 2ξ .- ξ.^2)
        rule = WCT_QB._weighted_corner_panel_rule(space, panel,
            panel.tleft, panel.tright, panel.source_order)
        lower, upper = panel.reverse ? (1-panel.tright, 1-panel.tleft) :
            (panel.tleft, panel.tright)
        power = panel.power*panel.lambda
        primitive = sum(coefficient*(upper^(power+index-1)-lower^(power+index-1))/(power+index-1)
            for (index,coefficient) in enumerate((1.0,2.0,-1.0)))
        expected = norm(tangent(curve, 0.5))*(panel.right-panel.left)*panel.power*primitive
        @test isapprox(sum(rule.basis*density), expected; atol=2e-12, rtol=2e-11)
        checked += 1
    end
    @test checked > 0

    # A smooth circle has no artificial corner exponents; the original
    # periodic/Kress path still carries no weighted interpolation metadata.
    circle = CircleBilliard(1.0)
    ordinary = CombinedFieldIntegralEquationSolver(8.0, circle;
        min_pts=64, grading=SmoothPeriodicGrading())
    @test evaluate_points(ordinary, circle, 4.0).density_space === nothing
    weighted = CombinedFieldIntegralEquationSolver(8.0, circle;
        min_pts=64, grading=WeightedCornerGrading())
    @test all(panel -> panel.lambda == 1, evaluate_points(weighted, circle, 4.0).density_space.panels)
    @test_throws ArgumentError WeightedCornerGrading(regular_density_nodes=3)
    @test_throws ArgumentError WeightedCornerGrading(corner_integration_nodes=4)
    @test_throws ArgumentError DoubleLayerPotentialSolver(8.0,billiard;
        grading=WeightedCornerGrading())
end

@testset "Weighted close-source geometry avoids coordinate cancellation" begin
    # Translating and rotating a straight edge must not introduce a false
    # target-normal derivative. Its normal chord projection is exactly zero,
    # even when subtracting Cartesian endpoints loses most relative digits.
    line = LineSegment(WCT_QB.SVector(1e5,-1e5),WCT_QB.SVector(1e5+2,-1e5+1))
    parameter, difference = 0.6, 1e-14
    target = curve(line,parameter)
    source = curve(line,parameter-difference)
    normal = only(normal_vec(line,[parameter]))
    distance, projection = WCT_QB._weighted_corner_pair_geometry(line,
        parameter,difference,target,normal,source)
    @test distance == abs(difference)*line.length
    @test projection == 0.0

    billiard = MushroomBilliard(0.4,1.0)
    kernel = CombinedFieldIntegralEquationSolver(8.0,billiard;
        min_pts=64,grading=WeightedCornerGrading())
    points = evaluate_points(kernel,billiard,10.0)
    for panel in points.density_space.panels
        isapprox(panel.lambda,2/3;atol=1e-12) || continue
        target_t, source_t = panel.reverse ? (0.999,0.99900001) : (0.001,0.00100001)
        observed = WCT_QB._weighted_corner_parameter_difference(panel,panel,
            target_t,source_t,0.0)
        expected = setprecision(256) do
            x = panel.reverse ? 1-BigFloat(target_t) : BigFloat(target_t)
            y = panel.reverse ? 1-BigFloat(source_t) : BigFloat(source_t)
            Float64((panel.reverse ? -1 : 1)*BigFloat(panel.right-panel.left)*(x^panel.power-y^panel.power))
        end
        @test isapprox(observed,expected;rtol=5e-15,atol=0)
    end
end

@testset "Weighted CFIE direct, Taylor, CORK and symmetry agreement" begin
    billiard = MushroomBilliard(0.4, 1.0; R=1.0)
    kernel = CombinedFieldIntegralEquationSolver(6.0, billiard;
        min_pts=64, grading=WeightedCornerGrading(), sym_characters=(-1,))
    points = evaluate_points(kernel, billiard, 8.0)
    center, degree, radius = 8.0, 16, 0.15
    coefficients = WCT_QB._weighted_corner_matrices(kernel, points, center, degree;
        multithreaded=false)
    z = center+0.04+0.025im
    direct = construct_matrices(kernel, points, z; multithreaded=false)
    polynomial = zero(direct)
    for coefficient in reverse(coefficients)
        polynomial .= polynomial.*(z-center) .+ coefficient
    end
    @test norm(polynomial-direct)/norm(direct) < 2e-11
    for low_degree in (0,1)
        short = WCT_QB._weighted_corner_matrices(kernel,points,center,low_degree;
            multithreaded=false)
        for index in eachindex(short)
            @test norm(short[index]-coefficients[index])/norm(coefficients[index]) < 2e-12
        end
    end

    cork, _ = WCT_QB.build_cork_polynomial(kernel, points, center, radius, degree;
        multithreaded=false)
    for coordinate in (-0.8, 0.0, 0.7)
        direct_real = construct_matrices(kernel, points, center+radius*coordinate;
            multithreaded=false)
        @test norm(WCT_QB.evaluate_cork_polynomial(cork,coordinate)-direct_real)/norm(direct_real) < 2e-11
    end
    expanded = ExpandedBIMSolver(kernel; use_taylor=false)
    A, derivative, second_derivative = construct_matrices(expanded, points, center;
        multithreaded=false)
    @test norm(A-coefficients[1])/norm(A) < 2e-12
    @test norm(derivative-coefficients[2])/norm(derivative) < 2e-12
    @test norm(second_derivative-2coefficients[3])/norm(second_derivative) < 2e-12

    full = only(WCT_QB._weighted_corner_matrices(kernel, points, center, 0;
        multithreaded=false, reduce_symmetry=false))
    orbits = WCT_QB._fold_boundary(Float64, billiard, points, kernel.symmetry,
        kernel.sym_characters)
    expected = zeros(ComplexF64, length(orbits.fundamental_indices), length(orbits.fundamental_indices))
    for column in axes(full,2), (row,target) in enumerate(orbits.fundamental_indices)
        expected[row,orbits.orbit_of[column]] += full[target,column]*orbits.phase[column]
    end
    @test norm(expected-coefficients[1])/norm(expected) < 2e-13

    # Caches contain geometry and quadrature only. A different Taylor center,
    # degree, complex direct call or threaded schedule must not contaminate
    # another spectral evaluation or the physical density norm quadrature.
    saved_norm = copy(points.density_space.norm_weights)
    saved_ds = copy(points.ds)
    previous_blas_threads = BLAS.get_num_threads()
    threaded = try
        BLAS.set_num_threads(2)
        matrices = WCT_QB._weighted_corner_matrices(kernel,points,center,3;
            multithreaded=true)
        @test BLAS.get_num_threads() == 2
        matrices
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end
    @test maximum(norm(threaded[j]-coefficients[j])/norm(coefficients[j])
        for j in eachindex(threaded)) < 3e-13
    changed = WCT_QB._weighted_corner_matrices(kernel,points,center+0.12,2;
        multithreaded=false)
    fresh_points = evaluate_points(kernel,billiard,center)
    @test fresh_points.xy == points.xy
    fresh = WCT_QB._weighted_corner_matrices(kernel,fresh_points,center+0.12,2;
        multithreaded=true)
    @test maximum(norm(changed[j]-fresh[j])/norm(fresh[j])
        for j in eachindex(fresh)) < 3e-13
    @test norm(construct_matrices(kernel,points,z;multithreaded=true)-direct)/norm(direct) < 3e-13
    @test points.density_space.norm_weights == saved_norm
    @test points.ds == saved_ds
end

@testset "Weighted circular eigenstate has physical normal density" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(12.0, billiard;
        min_pts=72, grading=WeightedCornerGrading())
    solver = BeynSolver(kernel; nq=40, r=8, use_taylor=true,
        use_chebyshev=false, taylor_degree=18)
    root, residual = solve_wavenumber(solver, billiard, 2.4, 0.2;
        multithreaded=false)
    @test abs(root-2.4048255576957728) < 2e-9
    @test residual < solver.res_tol
    state = only(compute_eigenstate(kernel, billiard, real(root);
        multithreaded=false).states)
    @test state.u !== nothing
    @test all(isfinite, state.u)
    @test isapprox(WCT_QB._rellich(state.pts,state.u,real(root)), 1; atol=2e-10)
    partial = BIMEigenstate(state.k,state.vec,state.ten,state.pts)
    recovered, arclength, normalization = boundary_function(partial)
    @test norm(recovered-state.u)/norm(state.u) < 2e-13
    @test arclength == state.pts.s
    @test normalization > 0
    H_state, qs_state, ps_state = husimi_function(partial,billiard,kernel;c=0.5)
    H_points, qs_points, ps_points = husimi_function(state.k,state.pts,recovered;c=0.5)
    @test norm(H_state-H_points) < 2e-13
    @test qs_state == qs_points && ps_state == ps_points
    @test isapprox(sum(H_state),1;atol=2e-13)
    constant_power, _ = momentum_function(ones(length(state.pts)),state.pts)
    @test isapprox(constant_power[1],1;atol=2e-13)
    @test maximum(constant_power[2:end]) < 1e-20
    power_state, frequencies_state = momentum_function(partial,billiard,kernel)
    power_points, frequencies_points = momentum_function(recovered,state.pts)
    @test norm(power_state-power_points) < 2e-13
    @test frequencies_state == frequencies_points
    targets = [WCT_QB.SVector{2,Float64}(r,0.0) for r in (0.0,0.2,0.5,0.8)]
    field = WCT_QB._weighted_corner_potential(state.pts,state.u,real(root),targets;
        multithreaded=false)
    exact = [WCT_QB.SpecialFunctions.besselj(0,real(root)*norm(x)) for x in targets]
    exact ./= sqrt(pi)*abs(WCT_QB.SpecialFunctions.besselj(1,real(root)))
    phase = dot(field,exact)/abs(dot(field,exact))
    @test norm(phase.*field-exact)/norm(exact) < 2e-8
    refined = WCT_QB._weighted_corner_potential(state.pts,state.u,real(root),targets;
        refinement=2, multithreaded=false)
    @test norm(field-refined)/norm(refined) < 2e-10
end

@testset "Weighted mushroom retains both parity spectra under refinement" begin
    billiard = MushroomBilliard(0.4, 1.0; R=1.0)
    for parity in (-1,1)
        spectra = Vector{ComplexF64}[]
        for ppw in (18.0,24.0)
            kernel = CombinedFieldIntegralEquationSolver(ppw,billiard;
                min_pts=64, grading=WeightedCornerGrading(), sym_characters=(parity,))
            solver = BeynSolver(kernel; nq=40, r=12, use_taylor=true,
                use_chebyshev=false, taylor_degree=18)
            roots, residuals = solve_spectrum(solver,billiard,30.0,0.4;
                multithreaded=false)
            @test length(roots) == (parity == -1 ? 3 : 1)
            @test all(<(solver.res_tol), residuals)
            push!(spectra,sort(roots;by=real))
        end
        @test maximum(abs.(spectra[1]-spectra[2])) < 1e-9
        @test maximum(abs,imag.(spectra[2])) < 1e-9
        high_kernel = CombinedFieldIntegralEquationSolver(10.0,billiard;
            min_pts=64,grading=WeightedCornerGrading(),sym_characters=(parity,))
        high_solver = BeynSolver(high_kernel;nq=40,r=24,use_taylor=true,
            use_chebyshev=false,taylor_degree=18)
        high_points = evaluate_points(high_kernel,billiard,100.0)
        high_roots, high_residuals, high_vectors = solve_vectors(high_solver,
            high_points,100.0,0.4;multithreaded=false)
        @test length(high_roots) == (parity == -1 ? 6 : 9)
        @test all(<(high_solver.res_tol),high_residuals)
        @test maximum(abs,imag.(high_roots)) < 1e-8
        if parity == 1
            # This state exposed unconverged iterative singular triplets:
            # a tiny reported singular value alone did not certify its vector.
            index = argmin(abs.(high_roots .- 100.0068323854))
            k = real(high_roots[index])
            tension, physical, _, _ = solve_state(high_kernel,high_points,k,billiard;
                multithreaded=false)
            A = construct_matrices(high_kernel,high_points,k;multithreaded=false)
            orbits = WCT_QB._fold_boundary(Float64,billiard,high_points,
                high_kernel.symmetry,high_kernel.sym_characters)
            reduced = physical[orbits.fundamental_indices]
            @test tension < 1e-8
            @test norm(A*reduced)/norm(reduced) < 1e-8
            vector_tension, vector = solve_vect(high_kernel,high_points,k;
                multithreaded=false)
            @test isapprox(norm(A*vector)/norm(vector),vector_tension;atol=1e-11,rtol=1e-5)
            @test abs(dot(vector,reduced))/(norm(vector)*norm(reduced)) > 1-1e-10
            for requested_krylov in (true,false)
                @test isapprox(solve(high_kernel,high_points,k;
                    use_krylov=requested_krylov,multithreaded=false),vector_tension;
                    atol=1e-11,rtol=1e-5)
            end
            contour_density = WCT_QB.symmetrize_layer_density(high_kernel,
                high_vectors[:,index],high_points,billiard)
            targets = [WCT_QB.SVector{2,Float64}(point) for point in
                ((0.95,0.1),(0.4,0.87),(-0.8,0.4),(-0.22,-0.7))]
            state_field = WCT_QB._weighted_corner_potential(high_points,physical,k,targets;
                multithreaded=false)
            contour_field = WCT_QB._weighted_corner_potential(high_points,contour_density,k,targets;
                multithreaded=false)
            alignment = dot(state_field,contour_field)/dot(state_field,state_field)
            @test norm(alignment.*state_field-contour_field)/norm(contour_field) < 1e-6
        end
    end
end
