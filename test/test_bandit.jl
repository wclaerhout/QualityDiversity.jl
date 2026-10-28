include("reference/bandit.jl")

const bandit_prob = Rastrigin(4)

function bandit_archive()
    archive = GridArchive(bandit_prob; dims=(10, 10))
    xs = [QD.random_solution(Xoshiro(1), bandit_prob) for _ in 1:50]
    QD.add_batch!(archive, xs, QD.evaluate_batch(bandit_prob, xs))
    return archive
end

# Runs one round in which the archive gives `statuses` to the batch, in order.
function bandit_round!(scheduler, archive, statuses)
    xs = QD.ask(Xoshiro(1), scheduler, archive)
    @test length(xs) == length(statuses)
    evs = QD.evaluate_batch(bandit_prob, xs)
    QD.tell!(scheduler, xs, evs, [QD.AddResult(s, 0.0) for s in statuses])
    return nothing
end

mutation(n) = VariationEmitter(GaussianMutation(0.1); batch_size=n)

@testset "BanditScheduler construction" begin
    pool = (mutation(2), mutation(3), CMAEmitter(4; sigma0=0.5, batch_size=5))
    s = BanditScheduler(pool; n_active=2)
    @test s isa QD.AbstractEmitter
    @test QD.active(s) == [true, true, false]
    @test QD.active(s) !== s.active
    @test QD.batch_size(s) == 5
    @test BanditScheduler(collect(pool); n_active=1).pool isa Tuple
    @test QD.restarts(mutation(2)) === nothing
    @test QD.restarts(pool[3]) == 0

    c = CMAEmitter(4; sigma0=0.5)
    @test_throws "pool[1] and pool[3] are the same CMAEmitter" BanditScheduler((c, mutation(2), c); n_active=1)
    m = mutation(2)
    @test BanditScheduler((m, m); n_active=1) isa BanditScheduler     # an emitter without state may repeat
    @test_throws "the pool must hold at least one emitter" BanditScheduler((); n_active=1)
    @test_throws "pool[2] is a Int64, not an AbstractEmitter" BanditScheduler((m, 1); n_active=1)
    @test_throws "n_active must be between 1 and the size of the pool (3), got 4" BanditScheduler(pool; n_active=4)
    @test_throws "n_active must be between 1 and the size of the pool (3), got 0" BanditScheduler(pool; n_active=0)
    @test_throws "zeta must be finite and at least 0, got -1" BanditScheduler(pool; n_active=1, zeta=-1)
    @test_throws "zeta must be finite and at least 0, got Inf" BanditScheduler(pool; n_active=1, zeta=Inf)
    @test_throws "window must be at least 1, got 0" BanditScheduler(pool; n_active=1, window=0)
end

@testset "BanditScheduler UCB1 choice" begin
    archive = bandit_archive()
    N, R = QD.NewCell, QD.Rejected
    s = BanditScheduler((mutation(2), mutation(2), mutation(2)); n_active=1, window=10)
    ucb(succ, sel, total) = succ / sel + 0.05 * sqrt(log(total) / sel)

    bandit_round!(s, archive, [N, R])
    @test s.selections == [2, 0, 0] && s.successes == [1, 0, 0]
    @test QD.active(s) == [false, true, false]     # untried emitters score Inf; the first wins the tie
    bandit_round!(s, archive, [R, R])
    @test QD.active(s) == [false, false, true]
    bandit_round!(s, archive, [N, QD.Improved])
    @test s.successes == [1, 0, 2]
    @test ucb(2, 2, 6) > ucb(1, 2, 6) > ucb(0, 2, 6)
    @test QD.active(s) == [false, false, true]     # the emitter that just ran competes too
    bandit_round!(s, archive, [R, R])
    @test ucb(1, 2, 8) > ucb(2, 4, 8)
    @test QD.active(s) == [true, false, false]
    @test s.rounds == 4

    # zeta weighs the bonus of rarely chosen emitters
    greedy = BanditScheduler((mutation(2), mutation(2)); n_active=1, zeta=0)
    bandit_round!(greedy, archive, [R, R])
    bandit_round!(greedy, archive, [N, R])
    bandit_round!(greedy, archive, [R, R])
    @test greedy.selections == [2, 4] && greedy.successes == [0, 1]
    @test QD.active(greedy) == [false, true]

    # The bonus takes the log of the total selections, as Algorithm 1 of the paper does.
    # Here that choice decides: with the log of the total successes, as pyribs takes it,
    # the first emitter would win.
    explorer = BanditScheduler((mutation(8), mutation(2)); n_active=1, zeta=1.5)
    bandit_round!(explorer, archive, [N, N, N, N, N, N, R, R])
    bandit_round!(explorer, archive, [R, R])
    @test explorer.selections == [8, 2] && explorer.successes == [6, 0]
    bonus(succ, sel, total) = succ / sel + 1.5 * sqrt(log(total) / sel)
    @test bonus(0, 2, 10) > bonus(6, 8, 10)
    @test bonus(0, 2, 6) < bonus(6, 8, 6)
    @test QD.active(explorer) == [false, true]
end

@testset "BanditScheduler window" begin
    archive = bandit_archive()
    N, R = QD.NewCell, QD.Rejected
    # with a window of one round, only the emitter that just ran has selections, so the
    # others score Inf, and the first two take turns
    s = BanditScheduler((mutation(2), mutation(2), mutation(2)); n_active=1, window=1)
    order = Int[]
    for _ in 1:6
        push!(order, findfirst(QD.active(s)))
        bandit_round!(s, archive, [N, N])
    end
    @test order == [1, 2, 1, 2, 1, 2]
    @test s.selections == [0, 2, 0]

    # rounds older than the window leave the counts
    w = BanditScheduler((mutation(2), mutation(3)); n_active=2, window=3)
    for statuses in ([N, N, R, R, R], [N, R, R, R, R], [R, R, N, N, N], [R, R, R, R, R])
        bandit_round!(w, archive, statuses)
    end
    @test w.selections == [6, 9]
    @test w.successes == [1, 3]
end

@testset "BanditScheduler reselection" begin
    archive = bandit_archive()
    N, R = QD.NewCell, QD.Rejected
    cma() = CMAEmitter(4; sigma0=0.5, batch_size=4, restart=QD.NoImprovementRestart())
    pool = (cma(), mutation(4), cma())

    # a CMA emitter stays until it restarts; an emitter that cannot restart leaves every round
    s = BanditScheduler(pool; n_active=2)
    @test QD.active(s) == [true, true, false]
    bandit_round!(s, archive, [N, R, R, R, N, N, N, N])
    @test QD.active(s) == [true, false, true]              # the mutation left; the untried one scores Inf
    bandit_round!(s, archive, [N, R, R, R, N, R, R, R])
    @test QD.active(s) == [true, false, true]              # neither CMA emitter restarted
    @test QD.restarts(s.pool[1]) == 0
    bandit_round!(s, archive, [R, R, R, R, N, N, R, R])    # the first one restarts
    @test QD.restarts(s.pool[1]) == 1
    @test s.seen_restarts == [1, 0, 0]
    # it competes with the mutation, which scores higher: 4 of 4 against 2 of 12
    @test QD.active(s) == [false, true, true]

    # ReselectAll chooses every slot again after every round
    a = BanditScheduler((mutation(2), mutation(2), mutation(2)); n_active=2, reselect=QD.ReselectAll())
    bandit_round!(a, archive, [N, N, R, R])
    @test QD.active(a) == [true, false, true]
end

@testset "BanditScheduler batches" begin
    archive = bandit_archive()
    pool = (mutation(3), CMAEmitter(4; sigma0=0.5, batch_size=5), mutation(7))
    s = BanditScheduler(pool; n_active=2, reselect=QD.ReselectAll())
    xs = QD.ask(Xoshiro(3), s, archive)
    @test length(xs) == 8 == QD.batch_size(s)
    @test xs isa Vector{Vector{Float64}}
    # the CMA emitter's part is the batch it sampled
    @test xs[4:8] == [s.pool[2].es.x[:, j] for j in 1:5]
    @test_throws "ask was called again before tell!" QD.ask(Xoshiro(3), s, archive)
    evs = QD.evaluate_batch(bandit_prob, xs)
    rs = QD.add_batch!(archive, xs, evs)
    @test_throws "expected 8 solutions, evaluations and results but got 7, 8 and 8" QD.tell!(s, xs[1:7], evs, rs)
    QD.tell!(s, xs, evs, rs)
    @test s.pool[2].iterations == 1
    @test_throws "tell! needs the results of the batch that the last ask returned" QD.tell!(s, xs, evs, rs)
    @test QD.batch_size(s) == sum(QD.batch_size(e) for (e, a) in zip(pool, QD.active(s)) if a)
end

@testset "MAP-Elites with a bandit scheduler" begin
    prob = Rastrigin(6)
    pool = (VariationEmitter(GaussianMutation(0.1); batch_size=20),
            VariationEmitter(IsoLineVariation(0.01, 0.1); batch_size=20),
            (CMAEmitter(6; sigma0=0.5, batch_size=10) for _ in 1:3)...)
    alg = QDAlgorithm(GridArchive(prob; dims=(10, 10)), (BanditScheduler(pool; n_active=2),))
    result = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test coverage(result.archive) > 0.5
    again = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test collect(QD.elites(again.archive)) == collect(QD.elites(result.archive))
    @test alg.emitters[1].rounds == 0                       # init works on a copy
end

# An emitter whose restarts the test decides: it restarts during the tell! that follows a
# call of `restart_next!`.
mutable struct ScriptedEmitter <: AbstractEmitter
    const n::Int
    const restarting::Bool
    restarts::Int
    restart_next::Bool
end
QD.batch_size(e::ScriptedEmitter) = e.n
QD.ask(rng, e::ScriptedEmitter, archive) = [zeros(1) for _ in 1:e.n]
QD.restarts(e::ScriptedEmitter) = e.restarting ? e.restarts : nothing
function QD.tell!(e::ScriptedEmitter, xs, evs, results)
    e.restart_next && (e.restarts += 1; e.restart_next = false)
    return nothing
end

@testset "BanditScheduler matches pyribs" begin
    # reference/bandit.jl: the emitters pyribs runs in each round, for statuses drawn at
    # random. The counts here cover the whole run, as in pyribs.
    status(k) = (QD.Rejected, QD.Improved, QD.NewCell)[k+1]
    archive = GridArchive((2,), ((0, 1),))
    for case in bandit_cases
        pool = Tuple(ScriptedEmitter(n, r == 1, 0, false) for (n, r) in zip(case.sizes, case.restarting))
        s = BanditScheduler(pool; n_active=case.num_active, reselect=case.reselect, window=10^6)
        for t in eachindex(case.actives)
            ours = findall(QD.active(s))
            expected = case.actives[t]
            if ours != expected
                # only on a tie, where pyribs takes the last of the emitters and this
                # package the first: the emitters chosen differently score the same
                total = sum(s.selections)
                score(i) = QD.ucb1(s.successes[i], s.selections[i], total, s.zeta)
                @test allequal(score.(symdiff(ours, expected)))
                @test maximum(setdiff(ours, expected)) < minimum(setdiff(expected, ours))
            end
            # continue from pyribs's choice, so that the counts stay the same
            fill!(s.active, false)
            s.active[expected] .= true
            foreach(i -> s.pool[i].restart_next = true, case.restarts[t])
            xs = QD.ask(Xoshiro(1), s, archive)
            QD.tell!(s, xs, [Evaluation(0.0, SVector(0.0)) for _ in xs], [QD.AddResult(status(k), 0.0) for k in case.statuses[t]])
        end
    end
end
