using QualityDiversity.Testing

include("reference/spread.jl")

samples(points...) = QD.DescriptorSamples([SVector{2,Float64}(p) for p in points])
spread_eval(fitness, points...; phenotype=nothing) = Evaluation(fitness, samples(points...), phenotype)
grid() = SpreadArchive(GridArchive((2, 2), ((0, 1), (0, 1))))

# Noisy descriptors around the first two entries of the solution.
noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x) + 0.1 * randn(rng),
                                         (clamp(x[1] + 0.2 * randn(rng), -1, 1), clamp(x[2] + 0.2 * randn(rng), -1, 1)));
                            init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-1, 1), (-1, 1)))

@testset "DescriptorSamples" begin
    d = samples((0.1, 0.2), (0.3, 0.4))
    @test d isa AbstractVector{SVector{2,Float64}}
    @test length(d) == 2 && d[2] == SVector(0.3, 0.4)
    @test d == samples((0.1, 0.2), (0.3, 0.4))
    @test isequal(spread_eval(1.0, (0.1, 0.2)), spread_eval(1.0, (0.1, 0.2)))
end

@testset "SampledSpread" begin
    prob = SampledSpread(noisy; num_samples=6)
    x = [0.1, 0.2, 0.3]
    ev = QD.evaluate(Xoshiro(1), prob, x)
    rng = Xoshiro(1)
    single = [QD.evaluate(rng, noisy, x) for _ in 1:6]
    @test ev.descriptor isa QD.DescriptorSamples{2}
    @test collect(ev.descriptor) == [s.descriptor for s in single]
    @test ev.fitness ≈ sum(s.fitness for s in single) / 6
    @test QD.evaluate(Xoshiro(1), SampledSpread(noisy; num_samples=6, expectation=QD.Median()), x).fitness ==
          QD.expectation(QD.Median(), [s.fitness for s in single])
    # descriptors of another number type are kept as Float64
    wide = QD.evaluate(Xoshiro(1), prob, big.(x))
    @test wide.descriptor isa QD.DescriptorSamples{2} && collect(wide.descriptor) ≈ collect(ev.descriptor)
    @test QD.evaluations_per_solution(prob) == 6
    @test QD.descriptor_bounds(prob) == QD.descriptor_bounds(noisy)
    test_problem(prob)
    test_problem(SampledSpread(noisy; num_samples=1))

    labelled = StochasticQDProblem((rng, x) -> (0.0, (x[1], x[2]), rand(rng, 1:2)); init=rng -> rand(rng, 2),
                                   bounds=((0, 1), (0, 1)))
    rng = Xoshiro(4)
    draws = [QD.evaluate(rng, labelled, x).phenotype for _ in 1:5]
    @test QD.evaluate(Xoshiro(4), SampledSpread(labelled; num_samples=5), x).phenotype == QD.most_frequent(draws)

    @test_throws "SampledSpread needs a stochastic problem, but Rastrigin is deterministic" SampledSpread(Rastrigin(2); num_samples=3)
    @test_throws "num_samples must be at least 1, got 0" SampledSpread(noisy; num_samples=0)
    @test_throws "SampledSpread needs a real fitness" QD.evaluate(Xoshiro(1), SampledSpread(StochasticQDProblem((rng, x) -> ([1.0, 2.0], (0.0, 0.0)); init=rng -> [0.0], bounds=((0, 1), (0, 1))); num_samples=2), [0.0])
end

@testset "construction" begin
    a = grid()
    @test a isa SpreadArchive{Vector{Float64},Float64,2,Nothing}
    @test a isa QD.AbstractArchive{Vector{Float64},Float64,SVector{2,Float64},Nothing}
    @test length(a) == 0 && QD.capacity(a) == 4
    @test all(isnan, QD.spreads(a)) && length(QD.spreads(a)) == 4
    @test QD.descriptor_bounds(a) == ((0.0, 1.0), (0.0, 1.0))
    @test SpreadArchive(CVTArchive(5, ((0, 1), (0, 1)); samples=100, phenotype=Int, rng=Xoshiro(1))) isa
          SpreadArchive{Vector{Float64},Float64,2,Int}
    full = GridArchive((2, 2), ((0, 1), (0, 1)))
    QD.add!(full, [0.0], Evaluation(1.0, SVector(0.1, 0.1)))
    @test_throws "the archive to wrap must be empty but holds 1 solutions" SpreadArchive(full)
    @test_throws "the categories of a CategoricalArchive do not have" SpreadArchive(CategoricalArchive((1:3,)))
end

@testset "cell, spread and stored descriptor" begin
    a = grid()
    # two of three descriptors in the cell of (0.25, 0.25)
    ev = spread_eval(1.0, (0.9, 0.9), (0.1, 0.2), (0.3, 0.4))
    @test QD.cell_index(a, ev.descriptor) == QD.cell_index(a, SVector(0.25, 0.25)) == 1
    @test QD.add!(a, [1.0], ev) == QD.AddResult(QD.NewCell, 1.0)
    x, stored = only(QD.elites(a))
    @test x == [1.0] && stored.fitness == 1.0
    @test stored.descriptor ≈ SVector(0.2, 0.3)
    d = [SVector(0.9, 0.9), SVector(0.1, 0.2), SVector(0.3, 0.4)]
    expected = (QD.norm(d[1] - d[2]) + QD.norm(d[1] - d[3]) + QD.norm(d[2] - d[3])) / 3
    @test QD.spreads(a)[1] ≈ expected
    @test isnan(QD.spreads(a)[2])

    # a tie goes to the cell visited first
    @test QD.cell_index(a, samples((0.9, 0.9), (0.1, 0.1))) == 4
    @test QD.cell_index(a, samples((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.2, 0.2), (0.8, 0.8))) == 1
    @test QD.cell_index(a, samples((0.9, 0.1), (0.1, 0.9), (0.6, 0.9), (0.4, 0.1))) == 2

    # one descriptor: spread zero, stored as it is
    b = grid()
    QD.add!(b, [0.0], spread_eval(2.0, (0.7, 0.2)))
    @test QD.spreads(b)[2] == 0.0
    @test only(QD.elites(b))[2].descriptor == SVector(0.7, 0.2)
end

@testset "replacement" begin
    a = grid()
    QD.add!(a, [1.0], spread_eval(1.0, (0.1, 0.1), (0.3, 0.3)))
    s = QD.spreads(a)[1]
    # higher fitness and larger spread
    r = QD.add!(a, [2.0], spread_eval(2.0, (0.0, 0.0), (0.4, 0.4)))
    @test r == QD.AddResult(QD.Rejected, 0.0)
    # lower fitness and smaller spread
    r = QD.add!(a, [3.0], spread_eval(0.5, (0.2, 0.2)))
    @test r == QD.AddResult(QD.Rejected, -0.5)
    # equal fitness
    @test QD.add!(a, [4.0], spread_eval(1.0, (0.2, 0.2))).status == QD.Rejected
    # higher fitness, equal spread
    r = QD.add!(a, [5.0], spread_eval(1.5, (0.3, 0.3), (0.1, 0.1)))
    @test r == QD.AddResult(QD.Improved, 0.5)
    @test QD.spreads(a)[1] ≈ s
    # higher fitness, smaller spread
    r = QD.add!(a, [6.0], spread_eval(3.0, (0.25, 0.25)))
    @test r == QD.AddResult(QD.Improved, 1.5)
    @test QD.spreads(a)[1] == 0.0
    @test only(QD.elites(a))[1] == [6.0]
    @test length(a) == 1 && max_fitness(a) == 3.0 && qd_score(a) == 3.0 && mean_fitness(a) == 3.0
    @test coverage(a) == 0.25
    @test QD.sample_elites(Xoshiro(1), a, 3) == [[6.0], [6.0], [6.0]]

    empty!(a)
    @test length(a) == 0 && all(isnan, QD.spreads(a))
    @test QD.add!(a, [1.0], spread_eval(1.0, (0.1, 0.1))).status == QD.NewCell
end

@testset "descriptors outside the bounds" begin
    rejecting = SpreadArchive(GridArchive((2, 2), ((0, 1), (0, 1)); outside=RejectOutside()))
    @test QD.add!(rejecting, [0.0], spread_eval(1.0, (2.0, 2.0), (3.0, 3.0))) == QD.AddResult(QD.Rejected, -Inf)
    # the descriptors inside decide the cell; all of them count for the spread
    @test QD.add!(rejecting, [0.0], spread_eval(1.0, (2.0, 2.0), (0.1, 0.1))).status == QD.NewCell
    @test only(QD.elites(rejecting))[2].descriptor == SVector(0.1, 0.1)
    @test QD.spreads(rejecting)[1] ≈ QD.norm(SVector(1.9, 1.9))
    @test QD.add!(rejecting, [0.0], spread_eval(5.0, (NaN, 0.5), (0.9, 0.1))) == QD.AddResult(QD.Rejected, 0.0)
    rs = QD.add_batch!(rejecting, [[1.0], [2.0], [3.0]],
                       [spread_eval(1.0, (2.0, 2.0), (3.0, 3.0)), spread_eval(5.0, (NaN, 0.5), (0.9, 0.1)),
                        spread_eval(2.0, (0.9, 0.9))])
    @test rs == [QD.AddResult(QD.Rejected, -Inf), QD.AddResult(QD.Rejected, 0.0), QD.AddResult(QD.NewCell, 2.0)]
    @test length(rejecting) == 2
    @test_throws "descriptor component 1 is 2.0, outside the archive bounds" QD.add!(grid(), [0.0], spread_eval(1.0, (2.0, 0.5)))
    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.add!(grid(), [0.0], Evaluation(1.0, QD.DescriptorSamples([SVector(0.1, 0.1, 0.1)])))
    @test_throws "the evaluation has no descriptors" QD.add!(grid(), [0.0], Evaluation(1.0, QD.DescriptorSamples(SVector{2,Float64}[])))
    @test_throws "a SpreadArchive takes the evaluations of a SampledSpread problem" QD.add!(grid(), [0.0], Evaluation(1.0, SVector(0.1, 0.1)))
    @test_throws "a SpreadArchive takes the evaluations of a SampledSpread problem" QD.add_batch!(grid(), [[0.0]], [Evaluation(1.0, SVector(0.1, 0.1))])
end

@testset "add_batch!" begin
    prob = SampledSpread(noisy; num_samples=5)
    for seed in 1:5
        rng = Xoshiro(seed)
        xs = [QD.random_solution(rng, prob) for _ in 1:300]
        evs = QD.evaluate_batch(rng, prob, xs)
        make() = SpreadArchive(GridArchive(prob; dims=(4, 4)))
        before = make()
        QD.add_batch!(before, xs[1:100], evs[1:100])
        batch = deepcopy(before)
        results = QD.add_batch!(batch, xs[101:end], evs[101:end])
        # every result is that of the solution alone against the archive before the batch
        @test results == [QD.add!(deepcopy(before), xs[k], evs[k]) for k in 101:300]
        # and the archive ends up as adding them one at a time leaves it
        sequential = deepcopy(before)
        foreach(k -> QD.add!(sequential, xs[k], evs[k]), 101:300)
        @test collect(QD.elites(batch)) == collect(QD.elites(sequential))
        @test isequal(QD.spreads(batch), QD.spreads(sequential))
        @test qd_score(batch) ≈ qd_score(sequential)
    end
    @test_throws "got 2 solutions but 1 evaluations" QD.add_batch!(grid(), [[0.0], [1.0]], [spread_eval(1.0, (0.1, 0.1))])
end

@testset "the elites of QDax" begin
    # Each solution is added alone; QDax keeps an arbitrary one of several that enter a
    # cell in one batch. The stored descriptor differs by design: QDax stores the centroid.
    for c in spread_cases
        centroids = [SVector(c.centroids[2k-1], c.centroids[2k]) for k in 1:length(c.centroids)÷2]
        a = SpreadArchive(CVTArchive{Int}(centroids, ((0, 1), (0, 1))))
        n = c.num_samples
        count = length(c.fitness) ÷ n
        for k in 1:count
            fitness = c.fitness[(k-1)*n+1:k*n]
            d = c.descriptors[(k-1)*2n+1:k*2n]
            ev = Evaluation(sum(fitness) / n, QD.DescriptorSamples([SVector(d[2j-1], d[2j]) for j in 1:n]))
            QD.add!(a, k, ev)
        end
        elite = zeros(Int, length(centroids))
        for (x, ev) in QD.elites(a)
            elite[QD.cell_index(a, ev.descriptor)] = x
        end
        @test elite == c.elite
        occupied = c.elite .> 0
        @test a.inner.fitness[occupied] ≈ c.elite_fitness[occupied]
        @test QD.spreads(a)[occupied] ≈ c.elite_spread[occupied]
    end
end

@testset "phenotypes" begin
    a = SpreadArchive(GridArchive((2, 2), ((0, 1), (0, 1)); phenotype=Symbol))
    QD.add!(a, [0.0], spread_eval(1.0, (0.1, 0.1); phenotype=:a))
    @test only(QD.elites(a))[2].phenotype === :a
    @test_throws "the evaluation has no phenotype" QD.add!(a, [0.0], spread_eval(2.0, (0.1, 0.1)))
    @test_throws "the evaluation has no phenotype" QD.add_batch!(a, [[0.0], [0.0]], [spread_eval(2.0, (0.1, 0.1); phenotype=:b), spread_eval(2.0, (0.1, 0.1))])
    @test only(QD.elites(a))[2].phenotype === :a
end

@testset "conformance" begin
    prob = SampledSpread(noisy; num_samples=5)
    batchwise = (; sequential_batch=false)
    test_archive(() -> SpreadArchive(GridArchive(prob; dims=(6, 6))), prob; batchwise...)
    test_archive(() -> SpreadArchive(GridArchive(prob; dims=(1, 1))), prob; batchwise...)
    test_archive(() -> SpreadArchive(GridArchive(prob; dims=(6, 6), fitness=Float32, genotype=Vector{Float32})), prob; batchwise...)
    test_archive(() -> SpreadArchive(GridArchive(prob; dims=(6, 6), outside=RejectOutside())), prob; batchwise...)
    test_archive(() -> SpreadArchive(CVTArchive(prob; cells=30, samples=3_000, rng=Xoshiro(1))), prob; batchwise...)

    single = SampledSpread(noisy; num_samples=1)
    test_archive(() -> SpreadArchive(GridArchive(single; dims=(6, 6))), single; batchwise...)

    # an isbits genotype, for the allocation checks
    bits = SampledSpread(StochasticQDProblem((rng, x) -> (-sum(abs2, x), (clamp(x[1] + 0.1 * randn(rng), -1, 1), clamp(x[2], -1, 1)));
                                             init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1), bounds=((-1, 1), (-1, 1)));
                         num_samples=4)
    test_archive(() -> SpreadArchive(GridArchive{SVector{3,Float64}}((6, 6), ((-1, 1), (-1, 1)))), bits; batchwise...)

    labelled = SampledSpread(StochasticQDProblem((rng, x) -> (-sum(abs2, x), (clamp(x[1] + 0.1 * randn(rng), -1, 1), clamp(x[2], -1, 1)), round(Int, 10 * x[3]));
                                                 init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-1, 1), (-1, 1)));
                             num_samples=4)
    test_archive(() -> SpreadArchive(GridArchive(labelled; dims=(6, 6), phenotype=Int)), labelled; batchwise...)

    archive = SpreadArchive(GridArchive(prob; dims=(6, 6)))
    first_solution = QD.random_solution(Xoshiro(1), prob)
    QD.add_batch!(archive, [first_solution], QD.evaluate_batch(Xoshiro(1), prob, [first_solution]))
    test_emitter(VariationEmitter(GaussianMutation(0.1); batch_size=8), archive, prob)
end

@testset "MELS" begin
    prob = SampledSpread(noisy; num_samples=4)
    alg = MELS(prob; dims=(5, 5), batch_size=16)
    @test alg.archive isa SpreadArchive{Vector{Float64},Float64,2,Nothing,<:GridArchive}
    result = solve(prob, alg; budget=4_000, rng=Xoshiro(1))
    @test result.archive isa SpreadArchive
    @test result.n_evals >= 4_000 && result.history[end].n_evals == result.n_evals
    @test length(result.archive) > 5
    @test all(isfinite, QD.spreads(result.archive)[i] for i in result.archive.inner.occupied_idx)
    # the same seed gives the same run, whatever the number of threads
    again = solve(prob, alg; budget=4_000, rng=Xoshiro(1))
    @test collect(QD.elites(again.archive)) == collect(QD.elites(result.archive))

    cvt = MELS(SpreadArchive(CVTArchive(prob; cells=20, samples=2_000, rng=Xoshiro(1))); batch_size=16)
    @test length(solve(prob, cvt; budget=2_000, rng=Xoshiro(2)).archive) > 3

    @test MELS(prob; dims=(5, 5), phenotype=Int).archive isa SpreadArchive{Vector{Float64},Float64,2,Int}

    @test_throws "MELS needs a SampledSpread problem" MELS(noisy; dims=(5, 5))
    @test_throws "MELS needs a SampledSpread problem" MELS(Sampled(noisy; num_samples=3); dims=(5, 5))
    @test_throws "SampledSpread keeps the descriptors of every evaluation, which only a SpreadArchive takes, not a GridArchive" init(prob, MAPElites(prob; dims=(5, 5)); budget=1_000)
    @test_throws "a SpreadArchive needs every evaluation of a solution: wrap the problem as SampledSpread" init(noisy, MELS(prob; dims=(5, 5)); budget=1_000)
    @test_throws "a SpreadArchive needs every evaluation of a solution" init(Sampled(noisy; num_samples=2), MELS(prob; dims=(5, 5)); budget=1_000)
    @test_throws "a SpreadArchive needs a stochastic problem, but Rastrigin is deterministic" init(Rastrigin(2), MELS(SpreadArchive(GridArchive(Rastrigin(2); dims=(5, 5)))); budget=1_000)
end
