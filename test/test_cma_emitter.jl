include("reference/cma_emitter.jl")

result(status, improvement) = QD.AddResult(status, Float64(improvement))
fit(f) = Evaluation(f, SVector(0.5, 0.5))

@testset "rankers" begin
    evs = [fit(1.0), fit(5.0), fit(3.0), fit(5.0), fit(2.0)]
    results = [result(QD.Rejected, -2), result(QD.NewCell, 5), result(QD.Improved, 4),
               result(QD.Improved, 1), result(QD.Rejected, -1)]

    # by fitness; the tie between the second and the fourth keeps the order of the batch
    order, values = QD.rank(QD.ObjectiveRanker(), evs, results)
    @test order == [2, 4, 3, 5, 1] && values == [5.0, 5.0, 3.0, 2.0, 1.0]
    # by improvement alone
    order, values = QD.rank(QD.ImprovementRanker(), evs, results)
    @test order == [2, 3, 4, 5, 1] && values == [5.0, 4.0, 1.0, -1.0, -2.0]
    # by status first: a new cell above improvements above rejections
    two = [result(QD.Rejected, 9), result(QD.NewCell, 1), result(QD.Improved, 4),
           result(QD.Improved, 7), result(QD.Rejected, -1)]
    order, values = QD.rank(QD.TwoStageImprovementRanker(), evs, two)
    @test order == [2, 4, 3, 1, 5] && values == [(2, 1.0), (1, 7.0), (1, 4.0), (0, 9.0), (0, -1.0)]
    @test QD.rank_key(QD.TwoStageImprovementRanker(), fit(1.0), result(QD.NewCell, 3)) === (2, 3.0)
    @test QD.rank_key(QD.ObjectiveRanker(), fit(2), result(QD.Rejected, 0)) === 2.0
    # by status first, then by fitness
    order, values = QD.rank(QD.TwoStageObjectiveRanker(), evs, two)
    @test order == [2, 4, 3, 5, 1] && values == [(2, 5.0), (1, 5.0), (1, 3.0), (0, 2.0), (0, 1.0)]
    @test QD.rank_key(QD.TwoStageObjectiveRanker(), fit(2.5), result(QD.Improved, 0)) === (1, 2.5)

    # a candidate the archive has no cell for ranks after every candidate that has one
    outside = QD.outside_result(Float64)
    @test outside == QD.AddResult(QD.Rejected, -Inf)
    @test QD.outside_result(Float32) === QD.AddResult(QD.Rejected, -Inf32)
    @test QD.outside_result(Int) === QD.AddResult(QD.Rejected, typemin(Int))
    @test QD.outside_result(UInt8) === QD.AddResult(QD.Rejected, 0x00)
    mixed = [outside, result(QD.Rejected, -3), result(QD.Improved, 1), outside]
    @test QD.rank(QD.ImprovementRanker(), evs[1:4], mixed)[1] == [3, 2, 1, 4]
    @test QD.rank(QD.TwoStageImprovementRanker(), evs[1:4], mixed)[1] == [3, 2, 1, 4]
    rejecting = GridArchive((2, 2), ((0, 1), (0, 1)); outside=RejectOutside())
    QD.add!(rejecting, [0.0], fit(5.0))
    far = Evaluation(1.0, SVector(5.0, 5.0))
    @test QD.rank_key(QD.ImprovementRanker(), far, QD.add!(rejecting, [0.0], far)) == -Inf
    @test QD.rank_key(QD.ImprovementRanker(), fit(1.0), QD.add!(rejecting, [0.0], fit(1.0))) == -4.0
    # all keys equal, also infinite ones, make a flat batch
    @test QD.flat_ranking([-Inf, -Inf]) && QD.flat_ranking([(0, -Inf), (0, -Inf)]) && !QD.flat_ranking([(0, 1.0), (0, -Inf)])

    # a batch is flat when the whole key of the best and the worst agree
    @test QD.key_distance(4.5, 4.0) == 0.5 && QD.key_distance((2, 3.0), (2, 3.0)) == 0
    @test QD.key_distance((2, 3.0), (0, 3.0)) == 2
    @test QD.flat_ranking([1.0, 1.0 + 1e-13]) && !QD.flat_ranking([1.0, 1.1]) && !QD.flat_ranking([1.0])
    @test QD.flat_ranking([(1, 2.0), (1, 2.0)]) && !QD.flat_ranking([(2, 2.0), (0, 2.0)])

    # a new ranker defines rank_key
    struct NegativeFitnessRanker <: QD.AbstractRanker end
    QD.rank_key(::NegativeFitnessRanker, ev, result) = -Float64(ev.fitness)
    order, _ = QD.rank(NegativeFitnessRanker(), evs, results)
    @test order == [1, 5, 3, 2, 4]
end

@testset "selection and restart rules" begin
    @test QD.parent_count(QD.FilterSelection(), 7, 36) == 7
    @test QD.parent_count(QD.FilterSelection(), 0, 36) == 0
    @test QD.parent_count(QD.MuSelection(), 7, 36) == 18
    @test QD.parent_count(QD.MuSelection(), 0, 7) == 3

    @test QD.restart_due(QD.NoImprovementRestart(), 4, 0)
    @test !QD.restart_due(QD.NoImprovementRestart(), 4, 1)
    @test !QD.restart_due(QD.ConvergenceRestart(), 4, 0)
    @test [QD.restart_due(QD.PeriodicRestart(3), i, 5) for i in 1:7] == [false, false, true, false, false, true, false]
    @test_throws "iterations must be at least 1, got 0" QD.PeriodicRestart(0)
end

@testset "CMAEmitter construction" begin
    e = CMAEmitter(10; sigma0=0.5)
    @test e isa CMAEmitter{QD.TwoStageImprovementRanker,QD.FilterSelection,QD.NoImprovementRestart}
    @test QD.batch_size(e) == 4 + floor(Int, 3 * log(10)) && e.iterations == 0 && e.restarts == 0
    @test QD.batch_size(CMAEmitter(10; sigma0=1, batch_size=20)) == 20
    f = CMAEmitter([1.0, 2.0, 3.0]; sigma0=1, ranker=QD.ImprovementRanker(), selection=QD.MuSelection(),
                   restart=QD.PeriodicRestart(5))
    @test f isa CMAEmitter{QD.ImprovementRanker,QD.MuSelection,QD.PeriodicRestart} && f.es.dim == 3 && f.x0 == [1.0, 2.0, 3.0]
    @test CMAEmitter(Float32[1, 2]; sigma0=1).x0 isa Vector{Float64}

    @test_throws UndefKeywordError CMAEmitter(3)
    @test_throws "sigma0 must be positive and finite, got -1" CMAEmitter(3; sigma0=-1)
    @test_throws "dim must be at least 1, got 0" CMAEmitter(0; sigma0=1)
    @test_throws "batch_size must be at least 2 so that there is a parent, got 1" CMAEmitter(3; sigma0=1, batch_size=1)
    @test_throws "x0 has 2 entries but the dimension is 3" CMAEmitter(3; sigma0=1, x0=[1.0, 2.0])
    @test_throws "x0 has the entry 2.0, outside the bounds [0.0, 1.0]" CMAEmitter([0.5, 2.0]; sigma0=0.1, lower=0, upper=1)
    @test_throws "x0 has the entry -1.0, outside the bounds [0.0, 1.0]" CMAEmitter([-1.0]; sigma0=0.1, lower=0, upper=[1.0])
    @test_throws "x0 must be finite" CMAEmitter(2; sigma0=1, x0=[1.0, Inf])
end

# An archive with a few elites around the origin.
function elite_archive(; G=Vector{Float64}, dims=(8, 8), n=3, kwargs...)
    archive = GridArchive{G}(dims, ((-4, 4), (-4, 4)); kwargs...)
    for i in 1:3
        x = convert(G, fill(Float64(i), n))
        QD.add!(archive, x, Evaluation(-Float64(i), SVector(i - 2.0, 0.0)))
    end
    return archive
end

@testset "CMAEmitter ask and tell!" begin
    archive = elite_archive()
    emitter = CMAEmitter(3; sigma0=0.5)
    xs = QD.ask(Xoshiro(1), emitter, archive)
    @test xs isa Vector{Vector{Float64}} && length(xs) == QD.batch_size(emitter)
    @test QD.ask(Xoshiro(1), CMAEmitter(3; sigma0=0.5), archive) == xs          # reproducible from the rng
    @test QD.ask(Xoshiro(2), CMAEmitter(3; sigma0=0.5), archive) != xs

    @test_throws "ask was called again before tell! reported the results of the last batch" QD.ask(Xoshiro(1), emitter, archive)

    evs = QD.evaluate_batch(QDProblem(x -> (-sum(abs2, x), (x[1], x[2])); init=rng -> zeros(3),
                                      bounds=((-4, 4), (-4, 4))), xs)
    results = QD.add_batch!(archive, xs, evs)
    @test QD.tell!(emitter, xs, evs, results) === nothing
    @test emitter.iterations == 1
    @test_throws "tell! needs the results of the batch that the last ask returned" QD.tell!(emitter, xs, evs, results)
    @test length(QD.ask(Xoshiro(1), emitter, archive)) == QD.batch_size(emitter)    # ask is allowed again after tell!

    fresh = CMAEmitter(3; sigma0=0.5)
    @test_throws "tell! needs the results of the batch that the last ask returned" QD.tell!(fresh, xs, evs, results)
    QD.ask(Xoshiro(1), fresh, archive)
    @test_throws "expected $(QD.batch_size(fresh)) solutions, evaluations and results but got 1, 1 and 1" QD.tell!(fresh, xs[1:1], evs[1:1], results[1:1])

    # genotypes of other types
    f32 = QD.ask(Xoshiro(1), CMAEmitter(3; sigma0=0.5), elite_archive(G=Vector{Float32}))
    @test f32 isa Vector{Vector{Float32}}
    sv = QD.ask(Xoshiro(1), CMAEmitter(3; sigma0=0.5), elite_archive(G=SVector{3,Float64}))
    @test sv isa Vector{SVector{3,Float64}}

    @test_throws "cannot sample elites from an empty archive" QD.ask(Xoshiro(1), CMAEmitter(3; sigma0=1), GridArchive((2,), ((0, 1),)))
    @test_throws "the solutions of the archive have 3 entries but the emitter has dimension 2" QD.ask(Xoshiro(1), CMAEmitter(2; sigma0=1), archive)
end

@testset "CMAEmitter starts from x0 or from an elite" begin
    archive = elite_archive()
    # the first batch is centered on x0, and every later start on an elite
    e = CMAEmitter([10.0, 10.0, 10.0]; sigma0=0.1, batch_size=400)
    xs = QD.ask(Xoshiro(1), e, archive)
    @test all(abs.(sum(xs) ./ 400 .- 10) .< 0.05)
    e = CMAEmitter(3; sigma0=0.1, batch_size=400)
    xs = QD.ask(Xoshiro(1), e, archive)
    center = sum(xs) ./ 400
    @test any(i -> all(abs.(center .- i) .< 0.05), 1:3)       # one of the elites [1,1,1], [2,2,2], [3,3,3]
end

@testset "CMAEmitter restarts" begin
    n = 3
    archive = elite_archive()
    all_rejected(k) = [result(QD.Rejected, -i) for i in 1:k]
    evs(k, f=0.0) = [Evaluation(f - i, SVector(0.0, 0.0)) for i in 1:k]
    ask_tell!(e, results; rng=Xoshiro(1)) = (xs = QD.ask(rng, e, archive); QD.tell!(e, xs, evs(length(xs)), results(length(xs))); xs)

    # no improvement: restart after a batch that the archive rejected
    e = CMAEmitter(n; sigma0=0.5, x0=[9.0, 9.0, 9.0], batch_size=8)
    ask_tell!(e, all_rejected)
    @test e.restarts == 1 && e.restart_pending && e.iterations == 1
    @test e.es.mean == [9.0, 9.0, 9.0]                       # a filter selection with no parent does not move
    xs = QD.ask(Xoshiro(2), e, archive)                      # the restart happens here
    @test !e.restart_pending && e.es.sigma == 0.5
    @test any(i -> e.es.mean == fill(Float64(i), n), 1:3)    # from an elite of the archive

    # an accepted candidate keeps the search going
    one_added(k) = [result(i == 1 ? QD.Improved : QD.Rejected, i == 1 ? 3 : -i) for i in 1:k]
    e = CMAEmitter(n; sigma0=0.5, x0=[9.0, 9.0, 9.0], batch_size=8)
    xs = ask_tell!(e, one_added)
    @test e.restarts == 0 && !e.restart_pending
    @test e.es.mean ≈ xs[1]                                  # one parent: the mean moves to it

    # only convergence restarts with the convergence rule, so a rejected batch does not
    e = CMAEmitter(n; sigma0=0.5, x0=[9.0, 9.0, 9.0], batch_size=8, restart=QD.ConvergenceRestart(),
                   selection=QD.MuSelection())
    ask_tell!(e, all_rejected)
    @test e.restarts == 0 && e.es.mean != [9.0, 9.0, 9.0]    # a mu selection moves anyway

    # periodic restarts
    e = CMAEmitter(n; sigma0=0.5, x0=[9.0, 9.0, 9.0], batch_size=8, restart=QD.PeriodicRestart(2),
                   selection=QD.MuSelection())
    for i in 1:6
        ask_tell!(e, one_added; rng=Xoshiro(i))
    end
    @test e.restarts == 3 && e.iterations == 6

    # CMA-ES convergence restarts: a flat batch (all values equal) stops the search
    e = CMAEmitter(n; sigma0=0.5, x0=[9.0, 9.0, 9.0], batch_size=8, restart=QD.ConvergenceRestart(),
                   ranker=QD.ObjectiveRanker(), selection=QD.MuSelection())
    xs = QD.ask(Xoshiro(1), e, archive)
    QD.tell!(e, xs, [fit(1.0) for _ in xs], all_rejected(8))
    @test e.restarts == 1
end

@testset "CMAEmitter respects bounds" begin
    prob = QDProblem(x -> (-sum(abs2, x .- 0.9), (x[1], x[2])); init=rng -> rand(rng, 4), bounds=((0, 1), (0, 1)))
    archive = GridArchive(prob; dims=(8, 8))
    emitters = Tuple(CMAEmitter(4; sigma0=0.5, lower=0.0, upper=1.0, ranker=QD.ImprovementRanker(),
                                selection=QD.MuSelection(), restart=QD.ConvergenceRestart()) for _ in 1:2)
    result = solve(prob, QDAlgorithm(archive, emitters); budget=2_000, rng=Xoshiro(1))
    stored = first.(collect(QD.elites(result.archive)))
    @test all(x -> all(0 .<= x .<= 1), stored)
    @test max_fitness(result.archive) > -0.01                # the optimum is the corner 0.9

    # A mean on the boundary of the box puts a draw inside with probability 2^-12 in 12
    # dimensions; after the tries the draws move inside instead of failing.
    corner = CMAEmitter(zeros(12); sigma0=0.3, lower=0, upper=1)
    xs = QD.ask(Xoshiro(1), corner, GridArchive((2, 2), ((0, 1), (0, 1))))
    @test length(xs) == QD.batch_size(corner)
    @test all(x -> all(0 .<= x .<= 1), xs)

    # an elite outside the bounds, which another emitter can put in the archive, starts the
    # search from the nearest point inside, not from outside where every draw would be clipped
    outside = GridArchive((2, 2), ((0, 1), (0, 1)))
    QD.add!(outside, [6.0, -3.0, 0.5, 2.0], Evaluation(1.0, SVector(0.5, 0.5)))
    bounded = CMAEmitter(4; sigma0=0.1, lower=0.0, upper=1.0)
    xs = QD.ask(Xoshiro(1), bounded, outside)
    @test bounded.es.mean == [1.0, 0.0, 0.5, 1.0]
    @test all(x -> all(0 .<= x .<= 1), xs)
    @test maximum(x -> maximum(abs, x .- [1.0, 0.0, 0.5, 1.0]), xs) < 0.5
    device = QD.DeviceCMAEmitter(4; n_emitters=2, sigma0=0.1, lower=0.0, upper=1.0)
    @test QD.start_means(Xoshiro(1), device, outside) == [[1.0, 0.0, 0.5, 1.0], [1.0, 0.0, 0.5, 1.0]]
end

@testset "CMAEmitter with an objective ranker maximizes" begin
    prob = Sphere(10)
    archive = GridArchive(prob; dims=(1, 1))
    emitter = CMAEmitter(10; sigma0=1.0, ranker=QD.ObjectiveRanker(), selection=QD.MuSelection(),
                         restart=QD.ConvergenceRestart(), batch_size=12)
    result = solve(prob, QDAlgorithm(archive, (emitter,)); budget=4_000, rng=Xoshiro(1))
    @test max_fitness(result.archive) > 99.99
end

@testset "CMAME and CMAMAE" begin
    prob = Sphere(10)
    alg = CMAME(prob; dims=(8, 8), sigma0=0.5)
    @test alg isa QDAlgorithm && alg.archive isa GridArchive && length(alg.emitters) == 5
    @test all(e -> e isa CMAEmitter{QD.TwoStageImprovementRanker,QD.FilterSelection,QD.NoImprovementRestart}, alg.emitters)
    @test all(e -> e.es.dim == 10 && e.es.sigma0 == 0.5, alg.emitters)
    @test alg.emitters[1] !== alg.emitters[2]
    @test length(CMAME(prob; dims=(8, 8), sigma0=0.5, n_emitters=2, batch_size=11).emitters) == 2
    @test QD.batch_size(CMAME(prob; dims=(8, 8), sigma0=0.5, batch_size=11).emitters[1]) == 11
    @test CMAME(prob; dims=(8, 8), sigma0=0.5, selection=QD.MuSelection()).emitters[1] isa CMAEmitter{<:Any,QD.MuSelection}
    @test CMAME(prob; dims=(8, 8), sigma0=0.5, genotype=Vector{Float32}, fitness=Float32).archive isa
          GridArchive{Vector{Float32},Float32}
    @test_throws UndefKeywordError CMAME(prob; dims=(8, 8))
    @test_throws "n_emitters must be at least 1, got 0" CMAME(prob; dims=(8, 8), sigma0=0.5, n_emitters=0)
    @test_throws "dims" CMAME(Rastrigin(4); sigma0=1)

    mae = CMAMAE(prob; dims=(8, 8), sigma0=0.5, threshold_min=0.0, learning_rate=0.1)
    @test mae.archive isa ThresholdArchive && mae.archive.learning_rate == 0.1 && mae.archive.threshold_min == 0.0
    @test all(e -> e isa CMAEmitter{QD.ImprovementRanker,QD.MuSelection,QD.ConvergenceRestart}, mae.emitters)
    @test CMAMAE(prob; dims=(8, 8), sigma0=0.5, threshold_min=0).archive.learning_rate == 0.01
    @test_throws UndefKeywordError CMAMAE(prob; dims=(8, 8), sigma0=0.5)

    # any archive: the dimension is given
    cvt = CVTArchive(prob; cells=40, samples=2_000, rng=Xoshiro(1))
    over_cvt = CMAME(cvt; sigma0=0.5, dim=10, n_emitters=2)
    @test over_cvt.archive === cvt && length(over_cvt.emitters) == 2
    r = solve(prob, over_cvt; budget=1_500, rng=Xoshiro(1))
    @test r.archive isa CVTArchive && length(r.archive) > 5
    @test_throws UndefKeywordError CMAME(cvt; sigma0=0.5)

    # the archive and the problem must agree on the descriptor dimensions
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, CMAME(GridArchive((4,), ((0, 1),)); sigma0=1, dim=10); budget=100)
end

# Sum of QD scores of several seeds.
function total_qd_score(prob, make; seeds, budget)
    return sum(qd_score(solve(prob, make(); budget, rng=Xoshiro(s)).archive) for s in seeds)
end

@testset "CMA-ME and CMA-MAE fill an archive better than MAP-Elites" begin
    prob = Sphere(10)
    dims = (16, 16)
    seeds = 1:4
    budget = 15_000
    map_elites = total_qd_score(prob, () -> MAPElites(prob; dims, batch_size=30, variation=GaussianMutation(0.5)); seeds, budget)
    cma_me = total_qd_score(prob, () -> CMAME(prob; dims, sigma0=0.5); seeds, budget)
    cma_mae = total_qd_score(prob, () -> CMAMAE(prob; dims, sigma0=0.5, threshold_min=0.0, learning_rate=0.1); seeds, budget)
    @test cma_me > 1.05 * map_elites
    @test cma_mae > 1.15 * map_elites
    @test cma_mae > cma_me

    # a run is reproducible from its seed
    a = solve(prob, CMAME(prob; dims, sigma0=0.5); budget=3_000, rng=Xoshiro(7))
    b = solve(prob, CMAME(prob; dims, sigma0=0.5); budget=3_000, rng=Xoshiro(7))
    @test a.history == b.history && collect(QD.elites(a.archive)) == collect(QD.elites(b.archive))
    # the algorithm is not modified by a run
    alg = CMAME(prob; dims, sigma0=0.5)
    solve(prob, alg; budget=2_000, rng=Xoshiro(1))
    @test length(alg.archive) == 0 && all(e -> e.iterations == 0 && !e.started, alg.emitters)

    # CMA-MAE reports the result archive, whose elites are the best of each cell
    mae = CMAMAE(prob; dims, sigma0=0.5, threshold_min=0.0, learning_rate=0.1)
    r = solve(prob, mae; budget=5_000, rng=Xoshiro(1))
    @test r.archive isa GridArchive && issorted(h.max_fitness for h in r.history)
end

@testset "direction rankers" begin
    archive = GridArchive((4, 4), ((0, 2), (-10, 10)))
    ranker = QD.RandomDirectionRanker()
    @test isempty(ranker.direction)
    @test_throws "the direction of the ranker is not set" QD.rank_key(ranker, fit(1.0), result(QD.NewCell, 1))
    QD.reset_ranker!(Xoshiro(1), ranker, archive)
    @test length(ranker.direction) == 2
    # reproducible from the generator, and a new draw after each reset
    other = QD.RandomDirectionRanker()
    QD.reset_ranker!(Xoshiro(1), other, archive)
    @test other.direction == ranker.direction
    QD.reset_ranker!(Xoshiro(2), other, archive)
    @test other.direction != ranker.direction
    # a standard normal vector scaled by the extent of the archive
    scaled = [QD.reset_ranker!(Xoshiro(i), other, archive) === nothing ? copy(other.direction) : nothing for i in 1:4000]
    @test sqrt(sum(d[1]^2 for d in scaled) / 4000) ≈ 2 rtol = 0.1
    @test sqrt(sum(d[2]^2 for d in scaled) / 4000) ≈ 20 rtol = 0.1

    # candidates are ordered by the projection of the descriptor
    ranker.direction .= [1.0, 2.0]
    evs = [Evaluation(0.0, SVector(1.0, 1.0)), Evaluation(0.0, SVector(0.0, 5.0)), Evaluation(0.0, SVector(2.0, -3.0))]
    results = [result(QD.Rejected, 0), result(QD.NewCell, 0), result(QD.Improved, 0)]
    order, keys = QD.rank(ranker, evs, results)
    @test order == [2, 1, 3] && keys == [10.0, 3.0, -4.0]
    # in two stages the status comes first
    two = QD.TwoStageRandomDirectionRanker([1.0, 2.0])
    order, keys = QD.rank(two, evs, results)
    @test order == [2, 3, 1] && keys == [(2, 10.0), (1, -4.0), (0, 3.0)]
    # the ranking of the CMA-ME paper: accepted above rejected, a new cell is not above an improvement
    accepted = QD.AcceptedRandomDirectionRanker([1.0, 2.0])
    order, keys = QD.rank(accepted, evs, results)
    @test order == [2, 3, 1] && keys == [(1, 10.0), (1, -4.0), (0, 3.0)]
    # an improvement with a larger projection than a new cell ranks first here, and second in two stages
    mixed = [result(QD.NewCell, 0), result(QD.Improved, 0), result(QD.Rejected, 0)]
    @test QD.rank(accepted, evs, mixed)[1] == [2, 1, 3]
    @test QD.rank(two, evs, mixed)[1] == [1, 2, 3]
    @test_throws "the descriptor has 2 dimensions but the direction has 3" QD.rank_key(QD.RandomDirectionRanker([1.0, 2.0, 3.0]), evs[1], results[1])

    # the archive wrappers report their bounds, and others fail with a message
    @test QD.descriptor_bounds(archive) == ((0.0, 2.0), (-10.0, 10.0))
    @test QD.descriptor_bounds(ThresholdArchive(archive; learning_rate=0.1, threshold_min=0.0)) == ((0.0, 2.0), (-10.0, 10.0))
    @test_throws "UnstructuredArchive does not report the bounds of its descriptor space" QD.reset_ranker!(Xoshiro(1), ranker, UnstructuredArchive{Vector{Float64}}(2; capacity=10, novelty_threshold=0.1))
    # rankers without state ignore the reset
    @test QD.reset_ranker!(Xoshiro(1), QD.ObjectiveRanker(), archive) === nothing
end

@testset "CMAEmitter keeps its own ranker and redraws the direction" begin
    archive = elite_archive()
    ranker = QD.TwoStageRandomDirectionRanker()
    a = CMAEmitter(3; sigma0=0.5, ranker)
    b = CMAEmitter(3; sigma0=0.5, ranker)
    @test a.ranker !== ranker && a.ranker !== b.ranker && isempty(ranker.direction)
    @test a isa CMAEmitter{QD.TwoStageRandomDirectionRanker}
    xs = QD.ask(Xoshiro(1), a, archive)
    @test length(a.ranker.direction) == 2 && isempty(b.ranker.direction)
    first_direction = copy(a.ranker.direction)
    evs = [Evaluation(0.0, SVector(i / 10, 0.0)) for i in 1:QD.batch_size(a)]
    results = [result(QD.Rejected, 0) for _ in evs]
    QD.tell!(a, xs, evs, results)
    @test a.restart_pending
    QD.ask(Xoshiro(2), a, archive)
    @test a.ranker.direction != first_direction              # a restart draws a new direction
    # the algorithm constructors give every emitter its own ranker
    alg = CMAME(Sphere(4); dims=(4, 4), sigma0=1.0, n_emitters=3, ranker=QD.RandomDirectionRanker())
    @test length(unique(objectid(e.ranker) for e in alg.emitters)) == 3
end

@testset "CMAEmitter with each evolution strategy" begin
    archive = elite_archive()
    for strategy in (QD.CMAES, QD.SepCMAES, QD.LMMAES, QD.OpenAIES, (dim; kwargs...) -> QD.LMMAES(dim; kwargs..., n_vectors=2))
        # LMMAES needs a batch of at most half the dimension
        dim = 12
        grid = GridArchive{Vector{Float64}}((4, 4), ((-4, 4), (-4, 4)))
        QD.add!(grid, fill(1.0, dim), Evaluation(-1.0, SVector(0.0, 0.0)))
        e = CMAEmitter(dim; sigma0=0.5, strategy, batch_size=6)
        @test e.es isa QD.AbstractEvolutionStrategy && QD.batch_size(e) == 6
        xs = QD.ask(Xoshiro(1), e, grid)
        @test length(xs) == 6 && all(x -> length(x) == dim, xs)
        evs = [Evaluation(-sum(abs2, x), SVector(x[1], x[2])) for x in xs]
        results = QD.add_batch!(grid, xs, evs)
        @test QD.tell!(e, xs, evs, results) === nothing
    end
    @test CMAEmitter(10; sigma0=1, strategy=QD.OpenAIES).es.batch_size == 2 * cld(4 + floor(Int, 3 * log(10)), 2)
    @test_throws "strategy must construct an AbstractEvolutionStrategy, got a Int64" CMAEmitter(3; sigma0=1, strategy=(dim; kwargs...) -> 1)
    @test CMAME(Sphere(10); dims=(4, 4), sigma0=1.0, strategy=QD.SepCMAES).emitters[1] isa CMAEmitter{<:Any,<:Any,<:Any,QD.SepCMAES}
end

# An evolution strategy that returns the scripted batches of reference/cma_emitter.jl and
# records the ranking and the number of parents it is told.
mutable struct ScriptedCandidates <: QD.AbstractEvolutionStrategy
    dim::Int
    batch_size::Int
    x::Matrix{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    told::Vector{Tuple{Vector{Int},Int}}
end
ScriptedCandidates() = ScriptedCandidates(4, 8, zeros(4, 8), fill(-Inf, 4), fill(Inf, 4), Tuple{Vector{Int},Int}[])
QD.reset!(es::ScriptedCandidates, x0) = es
QD.sample!(rng, es::ScriptedCandidates) = (es.x .= cma_emitter_samples[length(es.told)+1]; es)
QD.update!(es::ScriptedCandidates, order, parents) = (push!(es.told, (collect(order), Int(parents))); es)
QD.should_stop(es::ScriptedCandidates, keys) = false

@testset "CMAEmitter matches pyribs: $(c.ranker), $(c.selection), $(c.restart)$(c.integer ? ", integer fitness" : "")" for c in cma_emitter_cases
    # reference/cma_emitter.jl: the statuses, rankings, parent counts and restarts of
    # pyribs's EvolutionStrategyEmitter over scripted rounds. Candidates whose ranking
    # keys are equal are ordered differently: pyribs puts the later one first (the
    # objective ranker in an arbitrary order), CMAEmitter the earlier one, so a ranking
    # is compared by its keys and, in a round without ties, as an order.
    rankers = Dict("2imp" => QD.TwoStageImprovementRanker(), "imp" => QD.ImprovementRanker(),
                   "obj" => QD.ObjectiveRanker(), "2obj" => QD.TwoStageObjectiveRanker(),
                   "rd" => QD.RandomDirectionRanker(), "2rd" => QD.TwoStageRandomDirectionRanker())
    restart = c.restart isa Integer ? QD.PeriodicRestart(c.restart) :
              c.restart == "basic" ? QD.ConvergenceRestart() : QD.NoImprovementRestart()
    selection = c.selection == "mu" ? QD.MuSelection() : QD.FilterSelection()
    function evaluate(x)
        f = -sum(abs2, x .- cma_emitter_center)
        return Evaluation(c.integer ? round(f) : f, SVector(2tanh((x[1] + x[2]) / 2), 2tanh((x[3] + x[4]) / 2)))
    end
    status_code = Dict(QD.Rejected => 0, QD.Improved => 1, QD.NewCell => 2)
    a = GridArchive((6, 6), ((-2, 2), (-2, 2)))
    emitter = CMAEmitter(4; sigma0=1.0, batch_size=8, x0=zeros(4), strategy=(dim; kwargs...) -> ScriptedCandidates(),
                         ranker=rankers[c.ranker], selection, restart)
    for r in eachindex(cma_emitter_samples)
        xs = QD.ask(Xoshiro(r), emitter, a)
        emitter.ranker isa QD.DirectionRanker &&
            (emitter.ranker.direction .= cma_emitter_directions[QD.restarts(emitter)+1])
        evs = evaluate.(xs)
        rs = QD.add_batch!(a, xs, evs)
        keys = [QD.rank_key(emitter.ranker, ev, res) for (ev, res) in zip(evs, rs)]
        QD.tell!(emitter, xs, evs, rs)
        order, parents = emitter.es.told[end]
        @test [status_code[res.status] for res in rs] == c.statuses[r]
        @test parents == c.parents[r]
        @test QD.restarts(emitter) == c.restarts[r]
        @test keys[order] == keys[c.orders[r]]
        allunique(keys) && @test order == c.orders[r]
    end
end
