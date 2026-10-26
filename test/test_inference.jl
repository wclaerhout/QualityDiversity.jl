# Calls on the hot path infer a concrete type and, for isbits genotypes, do not
# allocate. Measurements are taken inside functions after a warm-up call.

function allocated(f::F, args::Vararg{Any,N}) where {F,N}
    f(args...)
    return @allocated f(args...)
end

@testset "inference" begin
    prob = Rastrigin(6)
    rng = Xoshiro(1)
    alg = MAPElites(prob; dims=(10, 10), batch_size=32)
    state = @inferred init(prob, alg; budget=640, rng)
    emitter = state.emitters[1]
    archive = state.archive
    d = SVector(0.5, -0.5)

    @inferred QD.evaluate(prob, QD.random_solution(rng, prob))
    @inferred QD.random_solution(rng, prob)
    @inferred QD.cell_index(archive, d)
    @inferred QD.sample_elites(rng, archive, 8)
    @inferred QD.ask(rng, emitter, archive)
    @inferred QD.vary(rng, emitter.op, QD.random_solution(rng, prob))
    @inferred QD.batch_size(emitter)
    @inferred length(archive)
    @inferred QD.capacity(archive)
    @inferred coverage(archive)
    @inferred max_fitness(archive)
    @inferred step!(state)
    @inferred solve!(state)
    @inferred QD.evaluate_batch(prob, [QD.random_solution(rng, prob) for _ in 1:4])

    xs = [QD.random_solution(rng, prob) for _ in 1:4]
    evs = QD.evaluate_batch(prob, xs)
    @inferred QD.add!(archive, xs[1], evs[1])
    @inferred QD.add_batch!(archive, xs, evs)
    @inferred QD.tell!(emitter, xs, evs, QD.add_batch!(archive, xs, evs))
end

@testset "inference with a crossover" begin
    prob = Rastrigin(6)
    alg = MAPElites(prob; dims=(10, 10), batch_size=16, variation=IsoLineVariation(0.05, 0.3))
    state = @inferred init(prob, alg; budget=320, rng=Xoshiro(1))
    emitter = state.emitters[1]
    @inferred QD.ask(Xoshiro(1), emitter, state.archive)
    @inferred QD.vary(Xoshiro(1), emitter.op, [0.0, 1.0], [1.0, 0.0])
    @inferred QD.n_parents(emitter.op)
    @inferred step!(state)
end

@testset "inference with bounded variation" begin
    prob = Rastrigin(6)
    for variation in (PolynomialMutation(-5.12, 5.12), UniformCrossover())
        alg = MAPElites(prob; dims=(10, 10), batch_size=16, variation)
        state = @inferred init(prob, alg; budget=320, rng=Xoshiro(1))
        @inferred QD.ask(Xoshiro(1), state.emitters[1], state.archive)
        @inferred step!(state)
    end
    @inferred QD.vary(Xoshiro(1), PolynomialMutation(-5.12, 5.12), [0.0, 1.0])
    @inferred QD.vary(Xoshiro(1), PolynomialMutation([-1.0, -2.0], [1.0, 2.0]), [0.0, 1.0])
    @inferred QD.vary(Xoshiro(1), UniformCrossover(), [1, 2], [3, 4])
    # the only allocation is the child
    x = [0.5, -1.0, 2.0]
    child = allocated(n -> Vector{Float64}(undef, n), 3)
    @test allocated(QD.vary, Xoshiro(1), PolynomialMutation(-5.12, 5.12), x) == child
    @test allocated(QD.vary, Xoshiro(1), PolynomialMutation([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]), x) == child
    @test allocated(QD.vary, Xoshiro(1), UniformCrossover(), x, x) == child
    @test allocated(QD.vary, Xoshiro(1), GaussianMutation(0.1), x) == child
    @test allocated(QD.vary, Xoshiro(1), IsoLineVariation(0.01, 0.2), x, x) == child
    # a Float32 parent gives a Float32 child, with the Float64 σ converted: still one allocation
    x32 = Float32[0.5, -1.0, 2.0]
    child32 = allocated(n -> Vector{Float32}(undef, n), 3)
    @test (@inferred QD.vary(Xoshiro(1), GaussianMutation(0.1), x32)) isa Vector{Float32}
    @test allocated(QD.vary, Xoshiro(1), GaussianMutation(0.1), x32) == child32
    @test (@inferred QD.vary(Xoshiro(1), IsoLineVariation(0.01, 0.2), x32, x32)) isa Vector{Float32}
    @test allocated(QD.vary, Xoshiro(1), IsoLineVariation(0.01, 0.2), x32, x32) == child32
end

@testset "inference with CMA emitters" begin
    prob = Rastrigin(6)
    for alg in (CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2),
                CMAMAE(prob; dims=(8, 8), sigma0=1.0, threshold_min=-500.0, learning_rate=0.1, num_emitters=2),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=QD.ObjectiveRanker(), selection=QD.MuSelection()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=QD.TwoStageRandomDirectionRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=QD.RandomDirectionRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=QD.AcceptedRandomDirectionRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=QD.PyribsCMAES),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, ranker=QD.TwoStageObjectiveRanker()),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=QD.SepCMAES),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=QD.OpenAIES),
                CMAME(prob; dims=(8, 8), sigma0=1.0, num_emitters=2, strategy=QD.LMMAES, batch_size=3))
        state = @inferred init(prob, alg; budget=600, rng=Xoshiro(1))
        emitter = state.emitters[1]
        xs = @inferred QD.ask(Xoshiro(1), emitter, state.archive)
        evs = QD.evaluate_batch(prob, xs)
        results = QD.add_batch!(state.archive, xs, evs)
        @inferred QD.tell!(emitter, xs, evs, results)
        @inferred QD.batch_size(emitter)
        @inferred step!(state)
        @inferred solve!(state)
    end
end

@testset "evolution-strategy updates do not allocate" begin
    for (strategy, dim) in ((QD.SepCMAES, 20), (QD.LMMAES, 40), (QD.OpenAIES, 20), (QD.CMAES, 100))
        es = strategy(dim; sigma0=0.5)
        order = collect(1:es.batch_size)
        rng = Xoshiro(1)
        QD.sample!(rng, es)
        QD.update!(es, order, 3)
        # one call per strategy: `allocated` would update twice
        update_bytes(es, order, parents) = @allocated QD.update!(es, order, parents)
        update_bytes(deepcopy(es), order, 2)
        free = Bool[]
        for _ in 1:6
            QD.sample!(rng, es)
            parents = es.batch_size ÷ 2
            # The least of three copies: `@allocated` also counts what a collection that
            # happens during the call allocates, such as finalizers of earlier tests.
            copies = [deepcopy(es) for _ in 1:3]
            bytes = minimum(c -> update_bytes(c, order, parents), copies)
            QD.update!(es, order, parents)
            # CMA-ES allocates only in the eigendecomposition, which it skips in some generations
            strategy === QD.CMAES && es.eigen_eval == es.count_eval && continue
            push!(free, bytes == 0)
        end
        @test !isempty(free) && all(free)
    end
end

@testset "inference with several emitters" begin
    prob = Rastrigin(4)
    archive = GridArchive(prob; dims=(5, 5))
    emitters = (VariationEmitter(GaussianMutation(0.1); batch_size=4),
                VariationEmitter(GaussianMutation(1.0); batch_size=3))
    state = init(prob, QDAlgorithm(archive, emitters); budget=100, rng=Xoshiro(1))
    @inferred step!(state)
end

@testset "allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    for outside in (ThrowOutside(), ClampOutside(), RejectOutside())
        archive = GridArchive{SVector{3,Float64}}((10, 10), ((-1, 1), (-1, 1)); outside)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                   # same fitness: rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0  # improvement
        fresh = Evaluation(ev.fitness, SVector(-0.9, 0.9))
        @test allocated(QD.add!, archive, x, fresh) == 0                # new cell
    end

    archive = GridArchive{Vector{Float64}}((10, 10), ((-1, 1), (-1, 1)))
    @test allocated(QD.cell_index, archive, SVector(0.1, 0.2)) == 0
end

@testset "inference with a bandit scheduler" begin
    prob = Rastrigin(4)
    pool = (VariationEmitter(GaussianMutation(0.1); batch_size=8), VariationEmitter(IsoLineVariation(0.01, 0.1); batch_size=4),
            (CMAEmitter(4; sigma0=0.5, batch_size=6) for _ in 1:30)...)
    scheduler = BanditScheduler(pool; num_active=3)
    state = init(prob, QDAlgorithm(GridArchive(prob; dims=(5, 5)), (scheduler,)); budget=10_000, rng=Xoshiro(1))
    @inferred step!(state)
    @inferred QD.ask(Xoshiro(1), state.emitters[1], state.archive)
    @inferred QD.batch_size(state.emitters[1])
end

@testset "DNSArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    xs = [QD.random_solution(Xoshiro(2), prob) for _ in 1:80]
    for phenotype in (Nothing, Int)
        archive = DNSArchive{SVector{3,Float64}}(2; capacity=50, k=3, phenotype)
        evs = [Evaluation(ev.fitness, ev.descriptor, phenotype === Int ? 1 : nothing) for ev in QD.evaluate_batch(prob, xs)]
        QD.add_batch!(archive, xs, evs)
        @inferred QD.cell_index(archive, evs[1].descriptor)
        @inferred QD.add!(archive, xs[1], evs[1])
        @inferred QD.add_batch!(archive, xs[1:5], evs[1:5])
        @test allocated(QD.cell_index, archive, evs[1].descriptor) == 0
        @test allocated(QD.add!, archive, xs[1], evs[1]) == 0
        @test allocated(QD.add!, archive, xs[2], Evaluation(evs[2].fitness + 1, evs[2].descriptor, evs[2].phenotype)) == 0
    end
end

@testset "DensityArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    xs = [QD.random_solution(Xoshiro(2), prob) for _ in 1:80]
    evs = QD.evaluate_batch(prob, xs)
    for phenotype in (Nothing, Int)
        grid = GridArchive{SVector{3,Float64}}((10, 10), ((-1, 1), (-1, 1)); phenotype)
        # a buffer that is full, so that reservoir sampling runs
        archive = DensityArchive(grid; bandwidth=0.2, buffer_capacity=40, rng=Xoshiro(1))
        pevs = [Evaluation(ev.fitness, ev.descriptor, phenotype === Int ? 1 : nothing) for ev in evs]
        QD.add_batch!(archive, xs, pevs)
        @inferred QD.density(archive, pevs[1].descriptor)
        @inferred QD.add!(archive, xs[1], pevs[1])
        @inferred QD.add_batch!(archive, xs[1:5], pevs[1:5])
        @test allocated(QD.cell_index, archive, pevs[1].descriptor) == 0
        @test allocated(QD.density, archive, pevs[1].descriptor) == 0
        @test allocated(QD.add!, archive, xs[1], pevs[1]) == 0
    end
    prob4 = Rastrigin(4)
    state = init(prob4, DDS(prob4; dims=(5, 5), sigma0=0.5, bandwidth=0.5, num_emitters=2, rng=Xoshiro(1)); budget=10_000, rng=Xoshiro(1))
    @inferred step!(state)
end

@testset "SlidingBoundariesArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    xs = [QD.random_solution(Xoshiro(2), prob) for _ in 1:50]
    evs = QD.evaluate_batch(prob, xs)
    for outside in (ThrowOutside(), ClampOutside(), RejectOutside()), phenotype in (Nothing, Int)
        # a remap after every solution, so that each measured call remaps
        archive = SlidingBoundariesArchive{SVector{3,Float64}}((10, 10), ((-1, 1), (-1, 1));
                                                               outside, phenotype, remap_frequency=1, buffer_capacity=20)
        QD.add_batch!(archive, xs, [Evaluation(ev.fitness, ev.descriptor, phenotype === Int ? 1 : nothing) for ev in evs])
        x = xs[1]
        ev = Evaluation(evs[1].fitness, evs[1].descriptor, phenotype === Int ? 1 : nothing)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor, ev.phenotype)) == 0
    end
end

@testset "CVTArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    centroids = QD.centroids(CVTArchive(prob; cells=200, samples=5_000, rng=Xoshiro(1)))
    for outside in (ThrowOutside(), ClampOutside(), RejectOutside())
        archive = CVTArchive{SVector{3,Float64}}(centroids, ((-1, 1), (-1, 1)); outside)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                                       # rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0  # improvement
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness, SVector(-0.9, 0.9))) == 0 # another cell
    end
    archive = CVTArchive(centroids, ((-1, 1), (-1, 1)))
    @inferred QD.sample_elites(Xoshiro(1), (QD.add!(archive, [0.0], Evaluation(1.0, SVector(0.0, 0.0))); archive), 4)
    @inferred max_fitness(archive)
end

@testset "UnstructuredArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    # full and not full, so that both the free-slot and the eviction path run
    for capacity in (200, 5)
        archive = UnstructuredArchive{SVector{3,Float64}}(2; capacity, novelty_threshold=0.1)
        for i in 1:20
            QD.add!(archive, QD.random_solution(Xoshiro(i), prob), Evaluation(-i, rand(Xoshiro(i), SVector{2,Float64})))
        end
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @inferred QD.sample_elites(Xoshiro(1), archive, 4)
        @inferred max_fitness(archive)
        @inferred qd_score(archive)
        far = SVector(50.0, 50.0)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, Evaluation(-100.0, far)) == 0           # rejected, or a new slot
        @test allocated(QD.add!, archive, x, Evaluation(1000.0, far)) == 0           # improvement
        @test allocated(QD.add!, archive, x, Evaluation(2000.0, SVector(60.0, 60.0))) == 0   # new slot, or eviction
    end
end

@testset "CategoricalArchive inference and allocations" begin
    labelled = QDProblem(x -> (-sum(abs2, x), (round(Int, clamp(x[1], -3, 3)), x[2] > 0 ? :pos : :neg));
                         init=rng -> SVector{3,Float64}(6 .* rand(rng, 3) .- 3), categories=(-3:3, [:neg, :pos]))
    numeric = QDProblem(x -> (-sum(abs2, x), (round(x[1]), round(x[2])));
                        init=rng -> SVector{3,Float64}(6 .* rand(rng, 3) .- 3), bounds=((-3, 3), (-3, 3)))
    for (prob, make, other) in ((labelled, outside -> CategoricalArchive{SVector{3,Float64}}(QD.descriptor_categories(labelled); outside), (3, :neg)),
                                (numeric, outside -> CategoricalArchive{SVector{3,Float64}}((-3:3, -3:3); outside), SVector(3.0, -3.0))),
        outside in (ThrowOutside(), RejectOutside())
        archive = make(outside)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                                         # rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0   # improvement
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness, other)) == 0               # another cell
    end
    archive = CategoricalArchive((-3:3, -3:3))
    @inferred QD.sample_elites(Xoshiro(1), (QD.add!(archive, [0.0], Evaluation(1.0, (0, 0))); archive), 4)
end

@testset "ThresholdArchive inference and allocations" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> SVector{3,Float64}(2 .* rand(rng, 3) .- 1),
                     bounds=((-1, 1), (-1, 1)))
    for (rate, floor) in ((1.0, -Inf), (1.0, -10.0), (0.3, -10.0))
        archive = ThresholdArchive(GridArchive{SVector{3,Float64}}((10, 10), ((-1, 1), (-1, 1)));
                                   learning_rate=rate, threshold_min=floor)
        x = QD.random_solution(Xoshiro(1), prob)
        ev = QD.evaluate(prob, x)
        @inferred QD.cell_index(archive, ev.descriptor)
        @inferred QD.add!(archive, x, ev)
        @inferred QD.sample_elites(Xoshiro(1), archive, 4)
        @inferred max_fitness(archive)
        @inferred qd_score(archive)
        @test allocated(QD.cell_index, archive, ev.descriptor) == 0
        @test allocated(QD.add!, archive, x, ev) == 0                                          # rejected
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness + 1, ev.descriptor)) == 0    # improvement
        @test allocated(QD.add!, archive, x, Evaluation(ev.fitness, SVector(-0.9, 0.9))) == 0   # new cell
        @test allocated(max_fitness, archive) == 0
        xs = [QD.random_solution(Xoshiro(k), prob) for k in 1:8]
        @inferred QD.add_batch!(archive, xs, QD.evaluate_batch(prob, xs))
    end
end

@testset "inference with stochastic problems" begin
    noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x) + randn(rng), (x[1], x[2]));
                                init=rng -> randn(rng, 4), bounds=((-3, 3), (-3, 3)))
    x = QD.random_solution(Xoshiro(1), noisy)
    @inferred QD.evaluate(Xoshiro(1), noisy, x)
    @inferred QD.evaluate(Xoshiro(1), Sampled(noisy; num_samples=3), x)
    @inferred QD.evaluate(Xoshiro(1), Sampled(noisy; num_samples=3, expectation=QD.Median()), x)
    @inferred QD.evaluate_batch(Xoshiro(1), noisy, [x, x])
    @inferred QD.evaluations_per_solution(Sampled(noisy; num_samples=3))
    for prob in (noisy, Sampled(noisy; num_samples=3))
        state = @inferred init(prob, MAPElites(prob; dims=(5, 5), batch_size=8); budget=400, rng=Xoshiro(1))
        @inferred step!(state)
        @inferred solve!(state)
    end
end

@testset "inference with MOME" begin
    prob = BiSphere(4)
    alg = MOME(prob; dims=(5, 5), batch_size=8)
    state = @inferred init(prob, alg; budget=400, rng=Xoshiro(1))
    @inferred step!(state)
    @inferred solve!(state)
    archive = state.archive
    d = SVector(0.2, 0.3)
    @inferred QD.cell_index(archive, d)
    @inferred QD.sample_elites(Xoshiro(1), archive, 4)
    @inferred QD.add!(archive, [0.0, 0.0, 0.0, 0.0], QD.evaluate(prob, zeros(4)))
    @inferred qd_score(archive)
    @inferred QD.metrics_record(archive, 1, 0)
    # a rejected solution and a cell lookup do not allocate with two objectives
    x = zeros(4)
    ev = QD.evaluate(prob, x)
    QD.add!(archive, x, ev)
    @test allocated(QD.cell_index, archive, ev.descriptor) == 0
    @test allocated(QD.add!, archive, x, ev) == 0
    @test allocated(QD.add!, archive, x, Evaluation(ev.fitness .- 1, ev.descriptor)) == 0
    # nor does replacing the most crowded solution of a full front, with an isbits genotype
    crowded = ParetoGridArchive{Int}((1,), ((0, 1),); reference_point=[0.0, 0.0], max_front=3, duplicates=QD.KeepDuplicates(), full_front=QD.ReplaceCrowded())
    on_line(t) = Evaluation(SVector(t, 1 - t), SVector(0.5))
    for (k, t) in enumerate((0.1, 0.2, 0.9))
        QD.add!(crowded, k, on_line(t))
    end
    r = @inferred QD.add!(crowded, 4, on_line(0.6))
    @test r.status == QD.Improved && QD.num_solutions(crowded) == 3
    @test allocated(QD.add!, crowded, 5, on_line(0.65)) == 0
    @test allocated(QD.add!, crowded, 6, on_line(0.15)) == 0
    # nor does a front of three objectives
    three = ParetoGridArchive{Int}((1,), ((0, 1),); reference_point=[0.0, 0.0, 0.0], max_front=4, full_front=QD.ReplaceCrowded())
    on_plane(s, t) = Evaluation(SVector(s, t, 1 - s - t), SVector(0.5))
    for (k, (s, t)) in enumerate(((0.1, 0.1), (0.6, 0.2), (0.2, 0.6), (0.3, 0.3)))
        QD.add!(three, k, on_plane(s, t))
    end
    @test allocated(QD.add!, three, 5, on_plane(0.4, 0.4)) == 0
    @test allocated(QD.add!, three, 6, on_plane(0.05, 0.05)) == 0
    @test allocated(QD.max_hypervolume, three) == 0
end

@testset "differentiable QD inference and allocations" begin
    prob = Rastrigin(4)
    xs = [QD.random_solution(Xoshiro(k), prob) for k in 1:20]
    evs, jacobians = QD.evaluate_gradient_batch(prob, xs)
    @inferred QD.evaluate_gradient(prob, xs[1])
    @inferred QD.evaluate_gradient_batch(prob, xs)
    # the closed-form gradients allocate only the Jacobian
    jacobian = allocated((m, n) -> Matrix{Float64}(undef, m, n), 3, 4)
    @test allocated(QD.evaluate_gradient, prob, xs[1]) == jacobian
    @test allocated(QD.evaluate_gradient, Sphere(4), xs[1]) == jacobian
    archive = GradientArchive(GridArchive(prob; dims=(5, 5)))
    @inferred QD.add_batch!(archive, xs, evs, jacobians)
    @inferred QD.add!(archive, xs[1], evs[1], jacobians[1])
    @test allocated(QD.cell_index, archive, evs[1].descriptor) == 0
    @inferred QD.ask(Xoshiro(1), GradientOperatorEmitter(; sigma0=1.0), archive)
    for alg in (CMAMEGA(prob; dims=(5, 5), sigma0=1.0), OMGMEGA(prob; dims=(5, 5), sigma0=1.0))
        state = init(prob, alg; budget=10_000, rng=Xoshiro(1))
        @inferred step!(state)
    end
end
