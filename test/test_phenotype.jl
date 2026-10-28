using QualityDiversity.Testing

# The phenotype is a label computed from the genotype, so a test can recompute it.
label(x) = string(round(sum(x); digits=3))
labelled = QDProblem(x -> (-sum(abs2, x), (clamp(x[1], -1, 1), clamp(x[2], -1, 1)), label(x));
                     init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-1, 1), (-1, 1)))
# An isbits phenotype, for the allocation checks.
counted = QDProblem(x -> (-sum(abs2, x), (clamp(x[1], -1, 1), clamp(x[2], -1, 1)), round(Int, 100 * x[3]));
                    init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1), bounds=((-1, 1), (-1, 1)))
plain = QDProblem(x -> (-sum(abs2, x), (clamp(x[1], -1, 1), clamp(x[2], -1, 1)));
                  init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-1, 1), (-1, 1)))

@testset "problems return a phenotype" begin
    ev = @inferred QD.evaluate(labelled, [0.1, 0.2, 0.3])
    @test ev isa Evaluation{Float64,SVector{2,Float64},String}
    @test ev.phenotype == "0.6"
    @test QD.evaluate(plain, [0.1, 0.2, 0.3]).phenotype === nothing
    test_problem(labelled)

    categorical = QDProblem(x -> (-sum(abs2, x), (round(Int, clamp(x[1], -1, 1)), :ring), label(x));
                            init=rng -> 2 .* rand(rng, 3) .- 1, categories=(-1:1, [:ring]))
    @test QD.evaluate(categorical, [0.1, 0.2, 0.3]) == Evaluation(-0.14, (0, :ring), "0.6")

    noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x) + randn(rng), (clamp(x[1], -1, 1), clamp(x[2], -1, 1)), label(x));
                                init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-1, 1), (-1, 1)))
    @test QD.evaluate(Xoshiro(1), noisy, [0.1, 0.2, 0.3]).phenotype == "0.6"
    test_problem(noisy)
    @test QD.evaluate(Xoshiro(1), Sampled(noisy; n_samples=4), [0.1, 0.2, 0.3]).phenotype == "0.6"
    test_problem(Sampled(noisy; n_samples=3))
end

@testset "Sampled keeps the most frequent phenotype" begin
    @test QD.most_frequent([:b, :a, :a, :c]) === :a
    @test QD.most_frequent([:b, :a, :a, :b, :c]) === :b      # a tie goes to the first
    @test QD.most_frequent(["x"]) == "x"
    @test QD.most_frequent([2, 1, 1]) === 1

    # the phenotype is a draw, so it varies between the evaluations of one solution
    drawn = StochasticQDProblem((rng, x) -> (-sum(abs2, x), (clamp(x[1], -1, 1), clamp(x[2], -1, 1)), rand(rng, 1:3));
                                init=rng -> 2 .* rand(rng, 3) .- 1, bounds=((-1, 1), (-1, 1)))
    x = [0.1, 0.2, 0.3]
    for seed in 1:20
        rng = Xoshiro(seed)
        draws = [QD.evaluate(rng, drawn, x).phenotype for _ in 1:7]
        ev = QD.evaluate(Xoshiro(seed), Sampled(drawn; n_samples=7), x)
        @test ev.phenotype == QD.most_frequent(draws)
        @test count(==(ev.phenotype), draws) == maximum(p -> count(==(p), draws), draws)
    end
    @test QD.evaluate(Xoshiro(1), Sampled(drawn; n_samples=1), x).phenotype == QD.evaluate(Xoshiro(1), drawn, x).phenotype
    test_problem(Sampled(drawn; n_samples=5))

    # without isequal of its own, a mutable phenotype compares by identity and the first is kept
    mutable struct Box
        value::Int
    end
    boxed = StochasticQDProblem((rng, x) -> (0.0, (0.0, 0.0), Box(rand(rng, 1:2))); init=rng -> [0.0],
                                bounds=((-1, 1), (-1, 1)))
    @test QD.evaluate(Xoshiro(3), Sampled(boxed; n_samples=6), [0.0]).phenotype.value ==
          QD.evaluate(Xoshiro(3), boxed, [0.0]).phenotype.value

    result = solve(drawn, MAPElites(Sampled(drawn; n_samples=3); dims=(4, 4), phenotype=Int); budget=600, rng=Xoshiro(1))
    @test all(((_, ev),) -> ev.phenotype in 1:3, QD.elites(result.archive))

    @test_throws "the function of the problem must return a (fitness, descriptor) or (fitness, descriptor, phenotype) tuple, got a Vector{Float64}" QD.evaluate(QDProblem(x -> [1.0, 0.5]; init=rng -> 0.0, bounds=((0, 1),)), 0.0)
    @test_throws "got a NTuple{4, Float64}" QD.evaluate(QDProblem(x -> (1.0, 0.5, 0.0, 0.0); init=rng -> 0.0, bounds=((0, 1),)), 0.0)
end

@testset "archives store the phenotype" begin
    bounds = ((-1, 1), (-1, 1))
    batchwise = (; sequential_batch=false)
    test_archive(() -> GridArchive(labelled; dims=(6, 6), phenotype=String), labelled; batchwise...)
    test_archive(() -> GridArchive{SVector{3,Float64}}((6, 6), bounds; phenotype=Int), counted; batchwise...)
    centroids = QD.centroids(CVTArchive(labelled; cells=30, samples=3_000, rng=Xoshiro(1)))
    test_archive(() -> CVTArchive(centroids, bounds; phenotype=String), labelled; batchwise...)
    test_archive(() -> CVTArchive{SVector{3,Float64}}(centroids, bounds; phenotype=Int), counted; batchwise...)
    test_archive(() -> CategoricalArchive((-1:1, -1:1); phenotype=String),
                 QDProblem(x -> (-sum(abs2, x), (round(clamp(x[1], -1, 1)), round(clamp(x[2], -1, 1))), label(x));
                           init=rng -> 2 .* rand(rng, 3) .- 1, bounds);
                 batchwise...)
    test_archive(() -> ThresholdArchive(GridArchive(labelled; dims=(6, 6), phenotype=String)), labelled; batchwise...)
    test_archive(() -> ThresholdArchive(GridArchive(labelled; dims=(6, 6), phenotype=String); learning_rate=0.5, threshold_min=-3),
                 labelled; sequential_batch=false, elitist=false)
    test_archive(() -> ThresholdArchive(GridArchive{SVector{3,Float64}}((6, 6), bounds; phenotype=Int)), counted; batchwise...)
    test_archive(() -> UnstructuredArchive(labelled; capacity=100, novelty_threshold=0.2, phenotype=String), labelled)
    test_archive(() -> UnstructuredArchive{SVector{3,Float64}}(2; capacity=100, novelty_threshold=0.2, phenotype=Int), counted)

    bisphere = BiSphere(4)
    objectives = QDProblem(x -> (ev = QD.evaluate(bisphere, x); (ev.fitness, ev.descriptor, label(x)));
                           init=rng -> QD.random_solution(rng, bisphere), bounds=QD.descriptor_bounds(bisphere))
    reference = QD.reference_point(bisphere)
    test_front_archive(() -> ParetoGridArchive(objectives; dims=(5, 5), reference_point=reference, phenotype=String), objectives)
    test_front_archive(() -> ParetoGridArchive(objectives; dims=(3, 3), reference_point=reference, max_front=4, phenotype=String), objectives)
end

@testset "archives without a phenotype store none" begin
    for archive in (GridArchive(plain; dims=(4, 4)), CVTArchive(plain; cells=5, samples=100, rng=Xoshiro(1)),
                    UnstructuredArchive(plain; capacity=10, novelty_threshold=0.1),
                    ParetoGridArchive(BiSphere(2); dims=(2, 2), reference_point=[0.0, 0.0]))
        @test archive isa QD.AbstractArchive{<:Any,<:Any,<:Any,Nothing}
        @test sizeof(archive.phenotypes) == 0
    end
    @test sizeof(CategoricalArchive((1:3,)).phenotypes) == 0
    @test ThresholdArchive(GridArchive(plain; dims=(4, 4))) isa QD.AbstractArchive{<:Any,<:Any,<:Any,Nothing}
end

@testset "a phenotype that does not fit the archive" begin
    x = [0.1, 0.2, 0.3]
    with = QD.evaluate(labelled, x)
    without = QD.evaluate(plain, x)
    for (make_plain, make_labelled) in (
            (() -> GridArchive(plain; dims=(4, 4)), () -> GridArchive(plain; dims=(4, 4), phenotype=String)),
            (() -> CVTArchive(plain; cells=5, samples=100, rng=Xoshiro(1)), () -> CVTArchive(plain; cells=5, samples=100, phenotype=String, rng=Xoshiro(1))),
            (() -> CategoricalArchive((-1:1, -1:1)), () -> CategoricalArchive((-1:1, -1:1); phenotype=String)),
            (() -> ThresholdArchive(GridArchive(plain; dims=(4, 4)); learning_rate=0.5, threshold_min=-3),
             () -> ThresholdArchive(GridArchive(plain; dims=(4, 4), phenotype=String); learning_rate=0.5, threshold_min=-3)),
            (() -> UnstructuredArchive(plain; capacity=10, novelty_threshold=0.1), () -> UnstructuredArchive(plain; capacity=10, novelty_threshold=0.1, phenotype=String)))
        descriptor = make_plain() isa CategoricalArchive ? (0, 0) : with.descriptor
        @test_throws "the evaluation has a phenotype of type String but the archive stores none; build the archive with phenotype=String" QD.add!(make_plain(), x, Evaluation(with.fitness, descriptor, with.phenotype))
        @test_throws "the evaluation has no phenotype but the archive stores phenotypes of type String" QD.add!(make_labelled(), x, Evaluation(with.fitness, descriptor))
        @test_throws "the evaluation has a phenotype of type $Int, which does not convert to the phenotype type String of the archive" QD.add!(make_labelled(), x, Evaluation(with.fitness, descriptor, 3))
        # a batch with one evaluation that does not fit leaves the archive unchanged
        a = make_labelled()
        @test_throws "the evaluation has no phenotype" QD.add_batch!(a, [x, x], [Evaluation(with.fitness, descriptor, "0.6"), Evaluation(with.fitness, descriptor)])
        @test length(a) == 0
        @test isempty(collect(QD.elites(QD.result_archive(a))))
    end
    pareto(phenotype) = ParetoGridArchive(BiSphere(2); dims=(2, 2), reference_point=[-10.0, -10.0], phenotype)
    ev = Evaluation(SVector(-1.0, -2.0), SVector(0.5, 0.5))
    @test_throws "build the archive with phenotype=Symbol" QD.add!(pareto(Nothing), x, Evaluation(ev.fitness, ev.descriptor, :a))
    @test_throws "the evaluation has no phenotype" QD.add!(pareto(Symbol), x, ev)
    a = pareto(Symbol)
    @test_throws "the evaluation has no phenotype" QD.add_batch!(a, [x, x], [Evaluation(ev.fitness, ev.descriptor, :a), ev])
    @test QD.n_solutions(a) == 0

    # a phenotype converts to the type of the archive, and a Union with Nothing takes none
    a = GridArchive(plain; dims=(4, 4), phenotype=Float64)
    QD.add!(a, x, Evaluation(with.fitness, with.descriptor, 1))
    @test only(QD.elites(a))[2].phenotype === 1.0
    a = GridArchive(plain; dims=(4, 4), phenotype=Union{Nothing,String})
    @test QD.add!(a, x, without).status == QD.NewCell
    @test only(QD.elites(a))[2].phenotype === nothing
end

@testset "algorithms keep the phenotype" begin
    for alg in (MAPElites(labelled; dims=(6, 6), phenotype=String),
                CMAME(labelled; dims=(6, 6), sigma0=0.2, phenotype=String),
                CMAMAE(labelled; dims=(6, 6), sigma0=0.2, threshold_min=-3, phenotype=String))
        result = solve(labelled, alg; budget=3_000, rng=Xoshiro(1))
        @test length(result.archive) > 0
        @test all(((x, ev),) -> ev.phenotype == label(x), QD.elites(result.archive))
    end
    archive = UnstructuredArchive(labelled; capacity=50, novelty_threshold=0.2, phenotype=String)
    result = solve(labelled, MAPElites(archive); budget=2_000, rng=Xoshiro(1))
    @test all(((x, ev),) -> ev.phenotype == label(x), QD.elites(result.archive))

    categorical = QDProblem(x -> (-sum(abs2, x), (round(Int, clamp(x[1], -1, 1)), x[2] > 0 ? :pos : :neg), label(x));
                            init=rng -> 2 .* rand(rng, 3) .- 1, categories=(-1:1, [:neg, :pos]))
    result = solve(categorical, MAPElites(categorical; phenotype=String); budget=1_000, rng=Xoshiro(1))
    @test all(((x, ev),) -> ev.phenotype == label(x), QD.elites(result.archive))

    bisphere = BiSphere(4)
    objectives = QDProblem(x -> (ev = QD.evaluate(bisphere, x); (ev.fitness, ev.descriptor, label(x)));
                           init=rng -> QD.random_solution(rng, bisphere), bounds=QD.descriptor_bounds(bisphere))
    result = solve(objectives, MOME(objectives; dims=(4, 4), reference_point=QD.reference_point(bisphere), phenotype=String);
                   budget=3_000, rng=Xoshiro(1))
    @test all(((x, ev),) -> ev.phenotype == label(x), QD.elites(result.archive))

    @test_throws "build the archive with phenotype=String" solve(labelled, MAPElites(labelled; dims=(6, 6)); budget=1_000)
end
