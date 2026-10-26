include("reference/variation.jl")

using OffsetArrays: OffsetArray

@testset "GaussianMutation" begin
    op = GaussianMutation(0.5)
    parent = zeros(1000)
    child = QD.vary(Xoshiro(1), op, parent)
    @test child isa Vector{Float64}
    @test parent == zeros(1000)
    @test length(child) == 1000
    @test abs(sum(child) / 1000) < 0.1
    @test 0.4 < sqrt(sum(abs2, child) / 1000) < 0.6
    @test QD.vary(Xoshiro(1), op, parent) == child

    @test QD.vary(Xoshiro(1), GaussianMutation(1), [1, 2, 3]) isa Vector{Float64}
    # the child has the float type of the parent, whatever the type of σ
    @test QD.vary(Xoshiro(1), op, Float32[1, 2]) isa Vector{Float32}
    @test QD.vary(Xoshiro(1), GaussianMutation(0.5f0), [1.0, 2.0]) isa Vector{Float64}
    @test_throws "σ must be positive and finite, got 0" GaussianMutation(0)
    @test_throws "σ must be positive and finite, got -1.0" GaussianMutation(-1.0)
end

@testset "IsoLineVariation" begin
    op = IsoLineVariation(0.1, 0.5)
    @test QD.n_parents(op) == 2
    @test QD.n_parents(GaussianMutation(0.1)) == 1
    @test IsoLineVariation(0.1, 1) isa IsoLineVariation{Float64}
    @test IsoLineVariation() == IsoLineVariation(0.01, 0.2)
    @test IsoLineVariation(σ_line=0.5).σ_iso == 0.01

    p1 = [0.0, 0.0, 0.0]
    p2 = [1.0, 2.0, 3.0]
    child = QD.vary(Xoshiro(1), op, p1, p2)
    @test child isa Vector{Float64} && length(child) == 3
    @test p1 == [0.0, 0.0, 0.0] && p2 == [1.0, 2.0, 3.0]
    @test QD.vary(Xoshiro(1), op, p1, p2) == child
    @test !(child === p1) && !(child === p2)

    # no isotropic noise: the child is on the line through the parents
    line = IsoLineVariation(0.0, 1.0)
    for seed in 1:20
        c = QD.vary(Xoshiro(seed), line, p1, p2)
        @test c ≈ c[1] .* p2          # a multiple of p2 - p1, starting at p1
    end
    # no line term: Gaussian noise around the first parent
    iso = IsoLineVariation(0.3, 0.0)
    noise = reduce(vcat, [QD.vary(Xoshiro(s), iso, fill(2.0, 50), fill(9.0, 50)) .- 2.0 for s in 1:100])
    @test abs(sum(noise) / length(noise)) < 0.03
    @test 0.28 < sqrt(sum(abs2, noise) / length(noise)) < 0.32
    # both zero: a copy of the first parent
    @test QD.vary(Xoshiro(1), IsoLineVariation(0.0, 0.0), p1, p2) == p1
    # the line coefficient has standard deviation σ_line
    coefficients = [QD.vary(Xoshiro(s), IsoLineVariation(0.0, 0.5), [0.0], [1.0])[1] for s in 1:5000]
    @test abs(sum(coefficients) / 5000) < 0.03
    @test 0.47 < sqrt(sum(abs2, coefficients) / 5000) < 0.53

    # the child has the float type the parents promote to, whatever the type of the operator
    @test QD.vary(Xoshiro(1), op, [1, 2, 3], Float32[4, 5, 6]) isa Vector{Float32}
    @test QD.vary(Xoshiro(1), op, [1.0, 2.0, 3.0], Float32[4, 5, 6]) isa Vector{Float64}
    @test QD.vary(Xoshiro(1), IsoLineVariation(0.1f0, 0.2f0), Float32[1, 2], Float32[3, 4]) isa Vector{Float32}
    @test QD.vary(Xoshiro(1), IsoLineVariation(0.1f0, 0.2f0), [1.0, 2.0], [3.0, 4.0]) isa Vector{Float64}
    @test_throws "parents have 2 and 3 entries but must have the same number" QD.vary(Xoshiro(1), op, [1.0, 2.0], [1.0, 2.0, 3.0])
    @test_throws "σ_iso must be non-negative and finite, got -0.1" IsoLineVariation(-0.1, 0.2)
    @test_throws "σ_line must be non-negative and finite, got -1" IsoLineVariation(0.1, -1)
end

@testset "PolynomialMutation" begin
    op = PolynomialMutation(-5.12, 5.12; eta=20)
    @test QD.n_parents(op) == 1
    @test op.eta == 20 && op.proportion == 1
    @test PolynomialMutation(0, 1) isa PolynomialMutation{Int,Int}
    @test PolynomialMutation(0, 1).eta == 20           # the defaults
    @test PolynomialMutation(0, 1; proportion=0.25).proportion == 0.25

    @test_throws "lower must be finite and below upper, got lower = 1 and upper = 1" PolynomialMutation(1, 1)
    @test_throws "lower must be finite and below upper" PolynomialMutation([0, 2], [1, 1])
    @test_throws "lower must be finite and below upper" PolynomialMutation(-Inf, 1)
    @test_throws "eta must be non-negative, got -1.0" PolynomialMutation(0, 1; eta=-1)
    @test_throws "proportion must be between 0 and 1, got 1.5" PolynomialMutation(0, 1; proportion=1.5)
    @test_throws "lower has 2 entries but upper has 3" PolynomialMutation([0, 0], [1, 1, 1])

    # The step never leaves the box, even before the final clipping, for any
    # position, any uniform draw in (0, 1) and any eta: that is what the bounded
    # form is for.
    rng = Xoshiro(1)
    for eta in (0.0, 1.0, 5.0, 20.0, 200.0, 5000.0), (lo, hi) in ((0.0, 1.0), (-5.12, 5.12), (10.0, 10.5))
        span = hi - lo
        inside = true
        edge_exact = true
        for x in (lo, hi, lo + 1e-9, hi - 1e-9, lo + span / 2, [lo + span * rand(rng) for _ in 1:200]...)
            for u in (1e-12, 0.25, 0.5, 0.75, 1 - 1e-12, [rand(rng) for _ in 1:30]...)
                y = x + QD.polynomial_step(x, lo, hi, eta, u)
                inside &= lo - 1e-9 * span <= y <= hi + 1e-9 * span
            end
            # u == 0 gives the lower bound exactly in real arithmetic. For large eta the
            # power underflows, to zero or to a denormal number that has few digits, so
            # the result is the bound only to about 1e-4 of the span; it is inside the box.
            edge_exact &= isapprox(clamp(x + QD.polynomial_step(x, lo, hi, eta, 0.0), lo, hi), lo; atol=1e-4 * span)
        end
        @test inside
        @test edge_exact
    end

    # children stay inside the box, for parents anywhere in it, including on the bounds
    parent = [-5.12, 5.12, 0.0, 5.0, -5.0, 1.0]
    for seed in 1:2000
        child = QD.vary(Xoshiro(seed), op, parent)
        @test all(-5.12 .<= child .<= 5.12)
    end

    # distribution of the step, against the reference implementation of NSGA-II
    # (pymoo's mut_pm): mean and standard deviation of the change of one entry
    function step_moments(x, eta, lo, hi; n=100_000)
        steps = QD.vary(Xoshiro(7), PolynomialMutation(lo, hi; eta), fill(float(x), n)) .- x
        return sum(steps) / n, sqrt(sum(abs2, steps .- sum(steps) / n) / n)
    end
    for (x, eta, lo, hi, mean_step, sd) in ((0.0, 20, 0.0, 1.0, 0.02275, 0.03827),
                                            (0.01, 20, 0.0, 1.0, 0.02030, 0.03978),
                                            (0.5, 20, 0.0, 1.0, 0.0, 0.06298),
                                            (0.9, 5, 0.0, 1.0, -0.04880, 0.13082),
                                            (1.0, 20, 0.0, 1.0, -0.02268, 0.03822),
                                            (0.3, 0, 0.0, 1.0, 0.09926, 0.29429),
                                            (4.0, 10, -5.12, 5.12, -0.19823, 0.88694),
                                            (-5.12, 30, -5.12, 5.12, 0.16015, 0.27243))
        m, s = step_moments(x, eta, lo, hi)
        @test abs(m - mean_step) < 0.02 * (hi - lo) / 10 + 4 * sd / sqrt(100_000)
        @test s ≈ sd rtol = 0.02
    end

    # an entry on a bound only moves inward, and sometimes stays
    from_lower = [QD.vary(Xoshiro(s), PolynomialMutation(0.0, 1.0), [0.0])[1] for s in 1:5000]
    @test all(>=(0.0), from_lower) && any(>(0.0), from_lower) && any(==(0.0), from_lower)
    from_upper = [QD.vary(Xoshiro(s), PolynomialMutation(0.0, 1.0), [1.0])[1] for s in 1:5000]
    @test all(<=(1.0), from_upper) && any(<(1.0), from_upper)

    # eta sets the size of the step
    sd(eta) = step_moments(0.5, eta, 0.0, 1.0)[2]
    @test sd(5) > sd(20) > sd(100)

    # proportion: the share of entries that change
    p = QD.vary(Xoshiro(1), PolynomialMutation(0.0, 1.0; proportion=0.3), fill(0.5, 100_000))
    @test 0.29 < count(!=(0.5), p) / 100_000 < 0.31
    @test QD.vary(Xoshiro(1), PolynomialMutation(0.0, 1.0; proportion=0.0), [0.2, 0.7]) == [0.2, 0.7]
    @test all(!=(0.5), QD.vary(Xoshiro(1), PolynomialMutation(0.0, 1.0; proportion=1.0), fill(0.5, 100)))

    # one bound per entry
    vec_op = PolynomialMutation([0.0, 10.0, -1.0], [1.0, 20.0, 0.0]; eta=2)
    for seed in 1:500
        c = QD.vary(Xoshiro(seed), vec_op, [0.5, 15.0, -0.5])
        @test 0 <= c[1] <= 1 && 10 <= c[2] <= 20 && -1 <= c[3] <= 0
    end
    @test_throws "lower has 3 entries but the parent has 2" QD.vary(Xoshiro(1), vec_op, [0.5, 15.0])

    # the parent must be inside the bounds
    @test_throws "parent entry 2 is 6.0, outside the bounds [-5.12, 5.12]" QD.vary(Xoshiro(1), op, [0.0, 6.0])
    @test_throws "parent entry 1 is NaN" QD.vary(Xoshiro(1), op, [NaN])

    # types and reproducibility
    @test QD.vary(Xoshiro(1), op, Float32[0, 1]) isa Vector{Float32}
    @test QD.vary(Xoshiro(1), op, [0, 1]) isa Vector{Float64}
    parent = [0.5, -1.0, 2.0]
    child = QD.vary(Xoshiro(3), op, parent)
    @test parent == [0.5, -1.0, 2.0] && child != parent && child !== parent
    @test QD.vary(Xoshiro(3), op, parent) == child
end

@testset "UniformCrossover" begin
    op = UniformCrossover()
    @test QD.n_parents(op) == 2
    @test op.proportion == 0.5
    @test UniformCrossover(proportion=0.2).proportion == 0.2
    @test UniformCrossover(0.2) == UniformCrossover(proportion=0.2)
    @test_throws "proportion must be between 0 and 1, got -0.1" UniformCrossover(-0.1)
    @test_throws "proportion must be between 0 and 1, got 2" UniformCrossover(proportion=2)

    p1 = fill(0.0, 100_000)
    p2 = fill(1.0, 100_000)
    child = QD.vary(Xoshiro(1), op, p1, p2)
    @test all(c -> c == 0.0 || c == 1.0, child)
    @test 0.49 < sum(child) / 100_000 < 0.51
    @test 0.19 < sum(QD.vary(Xoshiro(1), UniformCrossover(0.2), p1, p2)) / 100_000 < 0.21
    @test QD.vary(Xoshiro(1), UniformCrossover(0), p1, p2) == p1
    @test QD.vary(Xoshiro(1), UniformCrossover(1), p1, p2) == p2

    # each entry comes from the parent at the same position
    a, b = [1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]
    for seed in 1:100
        c = QD.vary(Xoshiro(seed), op, a, b)
        @test all(i -> c[i] == a[i] || c[i] == b[i], eachindex(c))
    end
    @test a == [1, 2, 3, 4, 5, 6, 7, 8] && b == [10, 20, 30, 40, 50, 60, 70, 80]

    # any entry type: integers stay integers, symbols and strings work
    @test QD.vary(Xoshiro(1), op, a, b) isa Vector{Int}
    @test QD.vary(Xoshiro(1), op, [:a, :b, :c], [:x, :y, :z]) isa Vector{Symbol}
    @test QD.vary(Xoshiro(1), op, ["a", "b"], ["x", "y"]) isa Vector{String}
    @test QD.vary(Xoshiro(1), op, [1, 2], [0.5, 0.25]) isa Vector{Float64}
    @test QD.vary(Xoshiro(1), op, Float32[1, 2], Float32[3, 4]) isa Vector{Float32}
    @test_throws "parents have 2 and 3 entries but must have the same number" QD.vary(Xoshiro(1), op, [1, 2], [1, 2, 3])
    @test QD.vary(Xoshiro(2), op, a, b) == QD.vary(Xoshiro(2), op, a, b)
end

@testset "MAP-Elites with bounded variation" begin
    prob = Rastrigin(6)
    mutation = PolynomialMutation(-5.12, 5.12; eta=20)
    result = solve(prob, MAPElites(prob; dims=(10, 10), batch_size=50, variation=mutation);
                   budget=6_000, rng=Xoshiro(1))
    @test result.n_evals == 6_000 && length(result.archive) > 30
    # every stored solution, descended from the uniform initial population, is inside the box
    @test all(all(-5.12 .<= g .<= 5.12) for (g, _) in QD.elites(result.archive))
    @test solve(prob, MAPElites(prob; dims=(10, 10), batch_size=50, variation=mutation);
                budget=6_000, rng=Xoshiro(1)).history == result.history

    # uniform crossover between archive members, with a mutation in a second emitter
    archive = GridArchive(prob; dims=(10, 10))
    emitters = (VariationEmitter(UniformCrossover(); batch_size=25),
                VariationEmitter(mutation; batch_size=25))
    crossed = solve(prob, QDAlgorithm(archive, emitters); budget=6_000, rng=Xoshiro(2))
    @test crossed.n_evals == 6_000 && length(crossed.archive) > 30
    @test all(all(-5.12 .<= g .<= 5.12) for (g, _) in QD.elites(crossed.archive))
end

@testset "variation keeps the indices of the parents" begin
    # A child has the indices of its parents and the entries of the child of the same
    # parents with ordinary indices.
    p1 = [0.5, -1.0, 2.0]
    p2 = [1.0, 0.0, -2.0]
    o1 = OffsetArray(p1, 0:2)
    o2 = OffsetArray(p2, 0:2)
    for (op, parents, offset) in ((GaussianMutation(0.1), (p1,), (o1,)),
                                  (PolynomialMutation(-5.0, 5.0), (p1,), (o1,)),
                                  (PolynomialMutation([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]), (p1,), (o1,)),
                                  (IsoLineVariation(0.1, 0.2), (p1, p2), (o1, o2)),
                                  (UniformCrossover(0.5), (p1, p2), (o1, o2)))
        child = QD.vary(Xoshiro(7), op, offset...)
        @test axes(child) == axes(o1)
        @test parent(child) == QD.vary(Xoshiro(7), op, parents...)
        @test QD.vary(Xoshiro(7), op, parents...) isa Vector
    end
    @test_throws "parents have the indices 0:2 and 1:3 but must have the same" QD.vary(Xoshiro(1), IsoLineVariation(0.1, 0.2), o1, p2)
    @test_throws "parents have the indices 0:2 and 1:3 but must have the same" QD.vary(Xoshiro(1), UniformCrossover(0.5), o1, p2)
    # views keep working
    @test length(QD.vary(Xoshiro(1), GaussianMutation(0.1), view([1.0, 2.0, 3.0], 2:3))) == 2

    @test_throws "σ must be positive and finite, got Inf" GaussianMutation(Inf)
    @test_throws "σ_iso must be non-negative and finite, got Inf" IsoLineVariation(Inf, 0.1)
    @test_throws "σ_line must be non-negative and finite, got NaN" IsoLineVariation(0.1, NaN)
end

@testset "PolynomialMutation matches pymoo" begin
    # reference/variation.jl: children of pymoo's mut_pm for fixed parents and uniform
    # draws; `polynomial_step` is the step that `vary` takes for one such draw
    child(x, lo, hi, eta, u) = clamp(x + QD.polynomial_step(x, lo, hi, eta, u), lo, hi)
    for c in pymoo_polynomial_cases
        @test child.(c.x, c.lower, c.upper, c.eta, c.u) ≈ c.child rtol = 1e-13
    end
    # QDax's polynomial_mutation takes (x - lo) / span where Deb and pymoo take
    # 1 - (x - lo) / span, and draws its uniforms in Float32
    q = qdax_polynomial_case
    function qdax_child(x, u)
        span = q.upper - q.lower
        d1, d2, p = (x - q.lower) / span, (q.upper - x) / span, 1 / (q.eta + 1)
        step = u < 0.5 ? (2u + d1^(q.eta + 1) * (1 - 2u))^p - 1 : 1 - (2(1 - u) + 2d2^(q.eta + 1) * (u - 0.5))^p
        return clamp(x + step * span, q.lower, q.upper)
    end
    @test qdax_child.(q.x, q.u) ≈ q.child rtol = 1e-6
    @test !(child.(q.x, q.lower, q.upper, q.eta, q.u) ≈ q.child)
end

@testset "Gaussian and iso+line children are distributed as in pyribs and QDax" begin
    # reference/variation.jl: per entry the mean, standard deviation and 10% and 90%
    # quantiles of 200 000 children of fixed parents, and the correlation of the first
    # two entries
    p1, p2 = variation_parents
    n = 200_000
    rng = Xoshiro(9)
    gaussian = reduce(hcat, [QD.vary(rng, GaussianMutation(0.1), p1) for _ in 1:n])
    isoline = reduce(hcat, [QD.vary(rng, IsoLineVariation(0.05, 0.3), p1, p2) for _ in 1:n])
    average(v) = sum(v) / length(v)
    deviation(v) = sqrt(sum(abs2, v .- average(v)) / length(v))
    quantile(v, q) = sort(v)[round(Int, q * length(v))]
    function similar_distribution(children, ref)
        rows = [children[j, :] for j in axes(children, 1)]
        means, sds = average.(rows), deviation.(rows)
        a, b = rows[1] .- means[1], rows[2] .- means[2]
        correlation = average(a .* b) / (sds[1] * sds[2])
        return all(abs.(means .- ref.mean) .<= 0.02 .* ref.std) &&
               all(abs.(sds .- ref.std) .<= 0.01 .* ref.std) &&
               all(abs.(quantile.(rows, 0.1) .- ref.q10) .<= 0.01) &&
               all(abs.(quantile.(rows, 0.9) .- ref.q90) .<= 0.01) &&
               abs(correlation - ref.cor12) <= 0.01
    end
    @test similar_distribution(gaussian, variation_statistics.pyribs_gaussian)
    @test similar_distribution(isoline, variation_statistics.pyribs_isoline)
    @test similar_distribution(isoline, variation_statistics.qdax_isoline)
end

@testset "UniformCrossover and QDax's polynomial_crossover" begin
    # QDax copies floor(proportion * n) positions drawn with replacement, so with
    # proportion 0.5 and 10 entries it changes 1 - 0.9^5 of them; UniformCrossover takes
    # each entry from the second parent with probability 0.5
    rng = Xoshiro(1)
    taken = sum(sum(QD.vary(rng, UniformCrossover(0.5), zeros(10), ones(10))) for _ in 1:200_000) / 2_000_000
    @test taken ≈ 0.5 atol = 0.005
    @test qdax_crossover_fraction ≈ 1 - 0.9^5 atol = 0.005
end
