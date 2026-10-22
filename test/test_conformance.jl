using QualityDiversity.Testing

# A test set that counts failures and errors instead of reporting them, to check
# that the conformance checks reject types that break their contract.
mutable struct CountFailures <: Test.AbstractTestSet
    description::String
    failures::Int
end
CountFailures(description::AbstractString; kwargs...) = CountFailures(description, 0)
Test.record(ts::CountFailures, ::Union{Test.Fail,Test.Error}) = (ts.failures += 1; nothing)
Test.record(ts::CountFailures, ::Test.Pass) = nothing
Test.record(ts::CountFailures, ::Test.Broken) = nothing
Test.record(ts::CountFailures, child::CountFailures) = (ts.failures += child.failures; nothing)
function Test.finish(ts::CountFailures)
    Test.get_testset_depth() == 0 || Test.record(Test.get_testset(), ts)
    return ts
end

failures(f) = (@testset CountFailures "detect" begin f() end).failures

@testset "problems" begin
    test_problem(Rastrigin(2))
    test_problem(Rastrigin(7))
    test_problem(Sphere(2))
    test_problem(Sphere(9))
    test_problem(QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                           init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-3, 3), (-3, 3))))
end

@testset "archives" begin
    prob = Rastrigin(4)
    test_archive(() -> GridArchive(prob; dims=(8, 8)), prob; sequential_batch=false)
    test_archive(() -> GridArchive(prob; dims=(1, 1)), prob; sequential_batch=false)
    test_archive(() -> GridArchive(prob; dims=(5, 3)), prob; sequential_batch=false)
    test_archive(() -> GridArchive(prob; dims=(8, 8), genotype=Vector{Float32}), prob; sequential_batch=false)
    test_archive(() -> GridArchive(prob; dims=(8, 8), fitness=Float32), prob; sequential_batch=false)
    test_archive(() -> GridArchive(prob; dims=(8, 8), outside=ClampOutside()), prob; sequential_batch=false)
    test_archive(() -> GridArchive(prob; dims=(8, 8), outside=RejectOutside()), prob; sequential_batch=false)
end

@testset "CVT archives" begin
    prob = Rastrigin(4)
    centroids = QD.centroids(CVTArchive(Xoshiro(1), prob; cells=40, samples=4_000))
    bounds = QD.descriptor_bounds(prob)
    test_archive(() -> CVTArchive(centroids, bounds), prob; sequential_batch=false)
    test_archive(() -> CVTArchive(centroids[1:1], bounds), prob; sequential_batch=false)
    test_archive(() -> CVTArchive{Vector{Float32},Float32}(centroids, bounds), prob; sequential_batch=false)
    test_archive(() -> CVTArchive(centroids, bounds; outside=ClampOutside()), prob; sequential_batch=false)
    test_archive(() -> CVTArchive(centroids, bounds; outside=RejectOutside()), prob; sequential_batch=false)
end

@testset "categorical archives" begin
    # numeric categories, with a problem that has bounds
    prob = QDProblem(x -> (-sum(abs2, x), (round(x[1]), round(x[2])));
                     init=rng -> 6 .* rand(rng, 3) .- 3, bounds=((-3, 3), (-3, 3)))
    test_archive(() -> CategoricalArchive((-3:3, -3:3)), prob; sequential_batch=false)
    test_archive(() -> CategoricalArchive((-3:3, -3:3); outside=RejectOutside()), prob; sequential_batch=false)
    test_archive(() -> CategoricalArchive(([3, -3, 0, 1, -1, 2, -2], -3:3)), prob; sequential_batch=false)
    test_archive(() -> CategoricalArchive{Vector{Float32},Float32}((-3:3, -3:3)), prob; sequential_batch=false)

    # labels, with a problem that has categories
    labelled = QDProblem(x -> (-sum(abs2, x), (round(Int, clamp(x[1], -3, 3)), x[2] > 0 ? :pos : x[2] < -1 ? :neg : :zero));
                         init=rng -> 6 .* rand(rng, 3) .- 3, categories=(-3:3, [:neg, :zero, :pos]))
    test_problem(labelled)
    test_problem(BiSphere(2))
    test_problem(BiSphere(7))

    noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x) + randn(rng), (clamp(x[1], -1, 1), clamp(x[2], -1, 1)));
                                init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-1, 1), (-1, 1)))
    test_problem(noisy)
    test_problem(Sampled(noisy; num_samples=3))
    test_problem(Sampled(noisy; num_samples=4, expectation=Median()))
    test_problem(StochasticQDProblem((rng, x) -> (randn(rng), (rand(rng, 1:3), :ring));
                                     init=rng -> randn(rng, 2), categories=(1:3, [:chain, :ring])))
    test_archive(() -> CategoricalArchive(labelled), labelled; sequential_batch=false)
    test_archive(() -> CategoricalArchive(labelled; outside=RejectOutside()), labelled; sequential_batch=false)
    test_archive(() -> CategoricalArchive(labelled; genotype=Vector{Float32}, fitness=Float32), labelled; sequential_batch=false)
end

@testset "threshold archives" begin
    prob = Rastrigin(4)
    # results of a batch are against the archive before it, so a batch is not a sequence of adds
    batchwise = (; sequential_batch=false)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8))), prob; batchwise...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(1, 1))), prob; batchwise...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8), fitness=Float32)), prob; batchwise...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8), outside=RejectOutside())), prob; batchwise...)
    # with a finite minimum and a rate below 1, a cell accepts solutions below its elite
    lenient = (; sequential_batch=false, elitist=false)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8)); learning_rate=0.5, threshold_min=-300), prob; lenient...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8)); learning_rate=0.1, threshold_min=-300), prob; lenient...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8), fitness=Float32); learning_rate=0.5, threshold_min=-300), prob; lenient...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8), outside=RejectOutside()); learning_rate=0.5, threshold_min=-300), prob; lenient...)
    test_archive(() -> ThresholdArchive(CVTArchive(Xoshiro(1), prob; cells=40, samples=2_000); learning_rate=0.5, threshold_min=-300), prob; lenient...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(1, 1)); learning_rate=0.5, threshold_min=-300), prob; lenient...)
    test_archive(() -> ThresholdArchive(GridArchive(prob; dims=(8, 8)); learning_rate=1, threshold_min=-300), prob; lenient...)
end

@testset "unstructured archives" begin
    prob = Rastrigin(4)
    test_archive(() -> UnstructuredArchive(prob; capacity=400, threshold=0.5), prob)
    test_archive(() -> UnstructuredArchive(prob; capacity=400, threshold=0), prob)
    test_archive(() -> UnstructuredArchive(prob; capacity=400, threshold=20), prob)       # one region
    test_archive(() -> UnstructuredArchive(prob; capacity=30, threshold=0.5), prob)       # evicts
    test_archive(() -> UnstructuredArchive(prob; capacity=1, threshold=0.5), prob)
    test_archive(() -> UnstructuredArchive(prob; capacity=400, threshold=0.5, genotype=Vector{Float32}, fitness=Float32), prob)
end

@testset "dns archives" begin
    prob = Rastrigin(4)
    # the population is chosen again as a whole for every batch
    population = (; population=true)
    test_archive(() -> DNSArchive(prob; capacity=400, k=3), prob; population...)
    test_archive(() -> DNSArchive(prob; capacity=30, k=3), prob; population...)      # removes
    test_archive(() -> DNSArchive(prob; capacity=30, k=1), prob; population...)
    test_archive(() -> DNSArchive(prob; capacity=30, k=50), prob; population...)     # fewer than k fitter
    test_archive(() -> DNSArchive(prob; capacity=1, k=3), prob; population...)
    test_archive(() -> DNSArchive(prob; capacity=30, k=3, genotype=Vector{Float32}, fitness=Float32), prob; population...)
end

@testset "density archives" begin
    prob = Rastrigin(4)
    # improvement is the negated density, not a gain in fitness
    density = (; sequential_batch=false, gain=false)
    test_archive(() -> DensityArchive(Xoshiro(1), GridArchive(prob; dims=(8, 8)); bandwidth=0.5), prob; density...)
    test_archive(() -> DensityArchive(Xoshiro(1), GridArchive(prob; dims=(8, 8)); bandwidth=0.5, buffer_size=20), prob; density...)
    test_archive(() -> DensityArchive(Xoshiro(1), CVTArchive(Xoshiro(1), prob; cells=40, samples=2_000); bandwidth=2), prob; density...)
    test_archive(() -> DensityArchive(Xoshiro(1), GridArchive(prob; dims=(8, 8), fitness=Float32); bandwidth=0.5), prob; density...)
end

@testset "archive with an isbits genotype" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    test_archive(() -> GridArchive{SVector{3,Float64}}((6, 6), ((-1, 1), (-1, 1))), prob; sequential_batch=false)
    test_archive(() -> DNSArchive{SVector{3,Float64}}(2; capacity=40, k=3), prob; population=true)
    test_archive(() -> DensityArchive(Xoshiro(1), GridArchive{SVector{3,Float64}}((6, 6), ((-1, 1), (-1, 1))); bandwidth=0.2,
                                      buffer_size=30), prob; sequential_batch=false, gain=false)
    test_archive(() -> SlidingBoundariesArchive{SVector{3,Float64}}((6, 6), ((-1, 1), (-1, 1)); remap_frequency=3), prob;
                 sequential_batch=false)
end

@testset "sliding boundaries archives" begin
    prob = Rastrigin(4)
    # results of a batch are against the archive before it, and a remap waits for the end of the batch
    batchwise = (; sequential_batch=false)
    test_archive(() -> SlidingBoundariesArchive(prob; dims=(8, 8)), prob; batchwise...)
    test_archive(() -> SlidingBoundariesArchive(prob; dims=(8, 8), remap_frequency=7, buffer_capacity=50), prob; batchwise...)
    test_archive(() -> SlidingBoundariesArchive(prob; dims=(8, 8), remap_frequency=1, buffer_capacity=10), prob; batchwise...)
    test_archive(() -> SlidingBoundariesArchive(prob; dims=(1, 1), remap_frequency=5), prob; batchwise...)
    test_archive(() -> SlidingBoundariesArchive(prob; dims=(5, 3), remap_frequency=5, fitness=Float32), prob; batchwise...)
end

@testset "front archives" begin
    prob = BiSphere(4)
    reference = QualityDiversity.reference_point(prob)
    test_front_archive(() -> ParetoGridArchive(prob; dims=(5, 5), reference_point=reference), prob)
    test_front_archive(() -> ParetoGridArchive(prob; dims=(3, 3), reference_point=reference, max_front=4), prob)
    test_front_archive(() -> ParetoGridArchive(prob; dims=(4, 4), reference_point=reference, genotype=SVector{4,Float64}), prob)
    test_front_archive(() -> ParetoGridArchive(prob; dims=(4, 4), reference_point=reference, outside=ClampOutside()), prob)
    test_front_archive(() -> ParetoGridArchive(prob; dims=(3, 3), reference_point=reference, max_front=4,
                                               duplicates=KeepDuplicates(), full_front=ReplaceCrowded()), prob)
    test_front_archive(() -> ParetoGridArchive(prob; dims=(3, 3), reference_point=reference, max_front=4,
                                               genotype=SVector{4,Float64}, full_front=ReplaceCrowded()), prob)
    three = QDProblem(x -> (SVector(-sum(abs2, x .- 1), -sum(abs2, x .+ 1), -sum(abs2, x)), (x[1], x[2]));
                      init=rng -> 4 .* rand(rng, 3) .- 2, bounds=((-2, 2), (-2, 2)))
    test_front_archive(() -> ParetoGridArchive(three; dims=(4, 4), reference_point=[-100.0, -100.0, -100.0]), three;
                       check_allocations=false)
end

@testset "variation" begin
    test_variation(GaussianMutation(0.1), [0.5, -1.0, 2.0])
    test_variation(GaussianMutation(2), fill(0.0, 10))
    test_variation(IsoLineVariation(0.1, 0.5), [0.5, -1.0, 2.0]; parent2=[1.0, 0.0, -2.0])
    test_variation(IsoLineVariation(), fill(0.0, 10); parent2=fill(1.0, 10))
    test_variation(PolynomialMutation(-5.12, 5.12), [0.5, -1.0, 5.12, -5.12])
    test_variation(PolynomialMutation([0.0, -1.0], [1.0, 1.0]; eta=5, proportion=0.5), [0.5, 0.0])
    test_variation(UniformCrossover(), [1, 2, 3, 4, 5, 6]; parent2=[7, 8, 9, 10, 11, 12])
    test_variation(UniformCrossover(0.3), [:a, :b, :c]; parent2=[:x, :y, :z])
end

@testset "emitters" begin
    prob = Rastrigin(4)
    xs = [QD.random_solution(Xoshiro(i), prob) for i in 1:20]
    evs = QD.evaluate_batch(prob, xs)
    # a new archive for each emitter: the checks add the children to it, and
    # PolynomialMutation only accepts parents inside its box
    filled() = (a = GridArchive(prob; dims=(8, 8)); QD.add_batch!(a, xs, evs); a)
    for op in (GaussianMutation(0.2), IsoLineVariation(0.05, 0.3), PolynomialMutation(-5.12, 5.12), UniformCrossover())
        test_emitter(VariationEmitter(op; batch_size=16), filled(), prob)
    end
    for (ranker, selection, restart) in ((TwoStageImprovementRanker(), FilterSelection(), NoImprovementRestart()),
                                         (ImprovementRanker(), MuSelection(), ConvergenceRestart()),
                                         (ObjectiveRanker(), MuSelection(), PeriodicRestart(2)))
        test_emitter(CMAEmitter(4; sigma0=1.0, ranker, selection, restart), filled(), prob)
        test_emitter(CMAEmitter(4; sigma0=1.0, ranker, selection, restart, x0=zeros(4), batch_size=9), filled(), prob)
    end
    test_emitter(CMAEmitter(4; sigma0=1.0, lower=-5.12, upper=5.12), filled(), prob)
    for ranker in (TwoStageObjectiveRanker(), RandomDirectionRanker(), TwoStageRandomDirectionRanker(),
                   AcceptedRandomDirectionRanker())
        test_emitter(CMAEmitter(4; sigma0=1.0, ranker), filled(), prob)
    end
    for strategy in (PyribsCMAES, SepCMAES, OpenAIES, LMMAES)
        # LMMAES needs a batch smaller than the dimension
        test_emitter(CMAEmitter(4; sigma0=1.0, strategy, batch_size=3 + 3 * (strategy === OpenAIES)), filled(), prob)
    end
    # genotypes other than Vector{Float64}
    for G in (Vector{Float32}, SVector{4,Float64})
        archive = GridArchive{G}((8, 8), ((-5.12, 5.12), (-5.12, 5.12)))
        QD.add!(archive, convert(G, fill(0.5, 4)), Evaluation(-1.0, SVector(0.5, 0.5)))
        test_emitter(CMAEmitter(4; sigma0=1.0), archive, QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> convert(G, randn(rng, 4)), bounds=((-5.12, 5.12), (-5.12, 5.12))))
    end
    # bandit schedulers, with emitters of different batch sizes in one pool
    pool() = (VariationEmitter(GaussianMutation(0.2); batch_size=16), VariationEmitter(UniformCrossover(); batch_size=5),
              CMAEmitter(4; sigma0=1.0), CMAEmitter(4; sigma0=1.0, batch_size=9))
    for num_active in 1:4, reselect in (ReselectTerminated(), ReselectAll())
        test_emitter(BanditScheduler(pool(); num_active, reselect), filled(), prob)
    end
    test_emitter(BanditScheduler((CMAEmitter(4; sigma0=1.0),); num_active=1, window=1), filled(), prob)
end

struct BadProblem <: AbstractQDProblem end
QD.evaluate(::BadProblem, x) = Evaluation(1.0, SVector(7.0, 0.0))   # outside the bounds
QD.random_solution(rng, ::BadProblem) = [1.0]
QD.descriptor_bounds(::BadProblem) = ((0.0, 1.0), (0.0, 1.0))

struct MutatingVariation <: AbstractVariation end
QD.vary(rng, ::MutatingVariation, parent) = (parent[1] += 1; parent)

@testset "the checks reject types that break their contract" begin
    @test failures(() -> test_problem(BadProblem())) > 0

    # a stochastic problem that ignores the generator it is given, and one that draws from a
    # global generator, are not reproducible from a seed
    struct GlobalNoise <: AbstractStochasticQDProblem end
    QD.evaluate(rng::Xoshiro, ::GlobalNoise, x) = Evaluation(randn(), SVector(0.5))
    QD.random_solution(rng::Xoshiro, ::GlobalNoise) = [0.5]
    QD.descriptor_bounds(::GlobalNoise) = ((0.0, 1.0),)
    @test failures(() -> test_problem(GlobalNoise())) > 0
    @test failures(() -> test_variation(MutatingVariation(), [1.0, 2.0])) > 0
    # a crossover that reports one parent, and a mutation checked as a crossover
    @test failures(() -> test_variation(IsoLineVariation(), [1.0, 2.0])) > 0
    @test failures(() -> test_variation(GaussianMutation(0.1), [1.0, 2.0]; parent2=[3.0, 4.0])) > 0
    # a cell that accepts solutions below its occupant is not elitist
    lenient() = ThresholdArchive(GridArchive(Rastrigin(4); dims=(8, 8)); learning_rate=0.1, threshold_min=-300)
    @test failures(() -> test_archive(lenient, Rastrigin(4); sequential_batch=false)) > 0
    @test failures(() -> test_archive(lenient, Rastrigin(4); elitist=false)) > 0     # its batches are not sequential
    @test failures(() -> test_archive(lenient, Rastrigin(4); elitist=false, sequential_batch=false)) == 0
    @test failures(() -> test_problem(Rastrigin(4))) == 0
end
