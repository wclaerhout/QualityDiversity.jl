noisy_fitness(rng, x) = -sum(abs2, x) + 5 * randn(rng)

function noisy_problem(; sigma=5.0, dim=4)
    return StochasticQDProblem((rng, x) -> (-sum(abs2, x) + sigma * randn(rng),
                                            (clamp(x[1], -3, 3) + 0.1 * randn(rng), clamp(x[2], -3, 3)));
                               init=rng -> 4 .* rand(rng, dim) .- 2,
                               bounds=((-4, 4), (-3, 3)))
end

@testset "StochasticQDProblem" begin
    prob = noisy_problem()
    @test prob isa AbstractStochasticQDProblem && prob isa AbstractQDProblem
    x = [1.0, 0.5, 0.0, 0.0]
    ev = QD.evaluate(Xoshiro(1), prob, x)
    @test ev isa Evaluation && ev.descriptor isa SVector{2,Float64}
    @test QD.evaluate(Xoshiro(1), prob, x) == ev
    @test QD.evaluate(Xoshiro(2), prob, x) != ev
    @test QD.descriptor_bounds(prob) == ((-4.0, 4.0), (-3.0, 3.0))
    @test QD.random_solution(Xoshiro(3), prob) == QD.random_solution(Xoshiro(3), prob)
    @test QD.evaluations_per_solution(prob) == 1

    @test_throws "NoisyThing is a stochastic problem: call evaluate(rng, prob, x)" QD.evaluate(
        (struct_type = (struct NoisyThing <: AbstractStochasticQDProblem end; NoisyThing()); struct_type), x)
    @test_throws "StochasticQDProblem is a stochastic problem: call evaluate(rng, prob, x)" QD.evaluate(prob, x)
    @test_throws "StochasticQDProblem is a stochastic problem: call evaluate_batch(rng, prob, xs)" QD.evaluate_batch(prob, [x])
    @test_throws "descriptor has 3 dimensions but the problem bounds have 2" QD.evaluate(
        Xoshiro(1), StochasticQDProblem((rng, x) -> (0.0, (1, 2, 3)); init=rng -> [0.0], bounds=((0, 1), (0, 1))), x)
    @test_throws "give exactly one of bounds and categories" StochasticQDProblem((rng, x) -> (0.0, (1,)); init=rng -> [0.0])

    labelled = StochasticQDProblem((rng, x) -> (randn(rng), (rand(rng, 1:3), :ring));
                                   init=rng -> randn(rng, 2), categories=(1:3, [:chain, :ring]))
    @test QD.descriptor_categories(labelled) == (1:3, [:chain, :ring])
    @test QD.evaluate(Xoshiro(1), labelled, [0.0, 0.0]).descriptor isa Tuple{Int,Symbol}
end

@testset "evaluate_batch draws one seed per solution" begin
    prob = noisy_problem()
    xs = [QD.random_solution(Xoshiro(i), prob) for i in 1:50]
    rng = Xoshiro(11)
    evs = QD.evaluate_batch(rng, prob, xs)
    # the documented rule: seeds are drawn in the order of the batch, one per solution
    seeds = let r = Xoshiro(11)
        [rand(r, UInt64) for _ in 1:50]
    end
    @test evs == [QD.evaluate(Xoshiro(seeds[i]), prob, xs[i]) for i in 1:50]
    # exactly those 50 draws were taken from the generator
    after = Xoshiro(11)
    foreach(_ -> rand(after, UInt64), 1:50)
    @test rand(rng) == rand(after)
    @test QD.evaluate_batch(Xoshiro(11), prob, xs) == evs
    @test QD.evaluate_batch(Xoshiro(12), prob, xs) != evs
    # the result of one solution does not depend on the rest of the batch
    @test QD.evaluate_batch(Xoshiro(11), prob, xs[1:1])[1] == QD.evaluate(Xoshiro(seeds[1]), prob, xs[1])

    # deterministic problems do not touch the generator
    det = Rastrigin(4)
    r = Xoshiro(5)
    dets = QD.evaluate_batch(r, det, xs)
    @test dets == QD.evaluate_batch(det, xs)
    @test rand(r) == rand(Xoshiro(5))
end

@testset "Sampled" begin
    prob = noisy_problem()
    x = [1.0, 0.5, 0.0, 0.0]
    sampled = Sampled(prob; num_samples=7)
    @test sampled isa AbstractStochasticQDProblem && sampled.expectation isa QD.Average
    @test QD.evaluations_per_solution(sampled) == 7
    @test QD.evaluations_per_solution(Sampled(sampled; num_samples=3)) == 21

    # the average of the evaluations of one generator, in order
    rng = Xoshiro(1)
    singles = [QD.evaluate(rng, prob, x) for _ in 1:7]
    ev = QD.evaluate(Xoshiro(1), sampled, x)
    @test ev.fitness ≈ sum(e.fitness for e in singles) / 7
    @test ev.descriptor ≈ sum(e.descriptor for e in singles) / 7
    @test ev.descriptor isa SVector{2,Float64}
    @test QD.evaluate(Xoshiro(1), sampled, x) == ev

    med = QD.evaluate(Xoshiro(1), Sampled(prob; num_samples=7, expectation=QD.Median()), x)
    @test med.fitness == sort([e.fitness for e in singles])[4]
    @test med.descriptor == SVector(sort([e.descriptor[1] for e in singles])[4], sort([e.descriptor[2] for e in singles])[4])
    even = QD.evaluate(Xoshiro(1), Sampled(prob; num_samples=4, expectation=QD.Median()), x)
    s4 = sort([e.fitness for e in singles[1:4]])
    @test even.fitness ≈ (s4[2] + s4[3]) / 2
    @test QD.expectation(QD.Average(), [1.0, 2.0, 6.0]) == 3.0 && QD.expectation(QD.Median(), [5.0]) == 5.0

    # the samples of one solution can mix integer and floating-point values
    mixed = StochasticQDProblem((rng, x) -> rand(rng, Bool) ? (1, (0,)) : (2.5, (0.5,));
                                init=rng -> [0.0], bounds=((0, 1),))
    averaged = QD.evaluate(Xoshiro(3), Sampled(mixed; num_samples=8), [0.0])
    @test averaged.fitness isa Float64 && averaged.descriptor isa SVector{1,Float64}
    @test 1 <= averaged.fitness <= 2.5

    # the spread shrinks with the square root of the number of samples
    fitness(p, n) = [QD.evaluate(Xoshiro(s), Sampled(p; num_samples=n), x).fitness for s in 1:300]
    spread(v) = sqrt(sum(abs2, v .- sum(v) / length(v)) / length(v))
    @test spread(fitness(prob, 1)) / spread(fitness(prob, 25)) ≈ 5 rtol = 0.25

    # what the wrapper takes from the wrapped problem
    @test QD.descriptor_bounds(sampled) == QD.descriptor_bounds(prob)
    @test QD.fitness_offset(sampled) == 0
    @test QD.descriptor_categories(sampled) === nothing
    @test QD.random_solution(Xoshiro(2), sampled) == QD.random_solution(Xoshiro(2), prob)

    @test_throws "num_samples must be at least 1, got 0" Sampled(prob; num_samples=0)
    @test_throws UndefKeywordError Sampled(prob)
    @test_throws "Sampled needs a stochastic problem, but Rastrigin is deterministic" Sampled(Rastrigin(3); num_samples=2)
    @test_throws "Sampled needs a real fitness, got a Vector{Float64}" QD.evaluate(
        Xoshiro(1), Sampled(StochasticQDProblem((rng, x) -> ([1.0, 2.0], (0.5,)); init=rng -> [0.0], bounds=((0, 1),));
                            num_samples=2), x)
    @test_throws "Sampled needs a descriptor that is an SVector of real numbers, got a Tuple" QD.evaluate(
        Xoshiro(1), Sampled(StochasticQDProblem((rng, x) -> (1.0, (1, :a)); init=rng -> [0.0],
                                                 categories=(1:2, [:a, :b])); num_samples=2), x)
end

@testset "runs with stochastic problems" begin
    prob = noisy_problem()
    run(p; seed=1, budget=2_000) = solve(p, MAPElites(p; dims=(6, 6), batch_size=16); budget, rng=Xoshiro(seed))
    a = run(prob)
    @test a.n_evals == 2_000 && a.history == run(prob).history
    @test a.history != run(prob; seed=2).history
    @test collect(QD.elites(a.archive)) == collect(QD.elites(run(prob).archive))

    # every evaluation of the wrapped problem counts against the budget
    sampled = Sampled(prob; num_samples=5)
    s = run(sampled)
    @test s.n_evals % 5 == 0 && 2_000 <= s.n_evals < 2_000 + 16 * 5
    @test length(s.history) > 1 && s.history[2].n_evals - s.history[1].n_evals == 16 * 5
    @test_throws "budget (50) is smaller than the initial population (80)" init(sampled, MAPElites(sampled; dims=(6, 6), batch_size=16); budget=50, rng=Xoshiro(1), init_size=16)

    # a lucky evaluation inflates the best fitness of a noisy run, and averaging removes most of it
    @test max_fitness(a.archive) > 0 > -1                    # the true best fitness is 0
    @test max_fitness(run(Sampled(prob; num_samples=50); budget=10_000).archive) < max_fitness(a.archive)

    # discrete descriptors and other genotypes
    labelled = StochasticQDProblem((rng, x) -> (-sum(abs2, x) + randn(rng), (rand(rng, 1:3), rand(rng, [:chain, :ring])));
                                   init=rng -> randn(rng, 2), categories=(1:3, [:chain, :ring]))
    r = solve(labelled, MAPElites(labelled); budget=500, rng=Xoshiro(1))
    @test r.archive isa CategoricalArchive && length(r.archive) > 3

    # the CMA emitters use the same loop
    cma = solve(prob, CMAME(prob; dims=(6, 6), sigma0=0.5, num_emitters=2, batch_size=8); budget=1_000, rng=Xoshiro(1))
    @test cma.n_evals >= 1_000 && length(cma.archive) > 1
end
