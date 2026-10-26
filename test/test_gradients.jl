using ForwardDiff: ForwardDiff
using LinearAlgebra: norm
using Random: Random, AbstractRNG
using QualityDiversity.Testing: test_emitter

include("reference/mega.jl")
include("reference/omg.jl")

# A smooth problem with two linear descriptors; ForwardDiff differentiates it.
smooth_problem(dim) = QDProblem(x -> (-sum(abs2, x .- 0.5) + sin(x[1]), (2tanh(sum(x[1:dim÷2]) / (dim ÷ 2)), 2tanh(x[end])));
                                init=rng -> 2 .* rand(rng, dim) .- 1, bounds=((-2, 2), (-2, 2)))

# The problem of the pyribs reference.
const mega_problem = QDProblem(x -> (-sum(abs2, x .- mega_center) + sin(x[1]), (sum(x[1:3]) / 3, sum(x[4:6]) / 3));
                               init=rng -> randn(rng, 6), bounds=((-2, 2), (-2, 2)))

# An evolution strategy that returns scripted batches and records what it is told.
mutable struct ScriptedStrategy <: QD.AbstractEvolutionStrategy
    dim::Int
    batch_size::Int
    x::Matrix{Float64}
    batches::Vector{Matrix{Float64}}
    told::Vector{Tuple{Vector{Int},Int}}
    resets::Int
end
ScriptedStrategy(batches) = ScriptedStrategy(size(batches[1], 1), size(batches[1], 2), zeros(size(batches[1])),
                                             batches, Tuple{Vector{Int},Int}[], 0)
QD.reset!(es::ScriptedStrategy, x0) = (es.resets += 1; es)
QD.sample!(rng, es::ScriptedStrategy) = (es.x .= es.batches[length(es.told)+1]; es)
QD.update!(es::ScriptedStrategy, order, parents) = (push!(es.told, (collect(order), Int(parents))); es)
QD.should_stop(es::ScriptedStrategy, keys) = false

# One round of the loop, outside `step!`, as pyribs runs it.
function mega_round!(emitter, archive, prob)
    point = QD.ask_gradients(Xoshiro(0), emitter, archive)
    evs, jacobians = QD.evaluate_gradient_batch(prob, point)
    QD.tell_gradients!(emitter, point, evs, jacobians, QD.add_batch!(archive, point, evs))
    xs = QD.ask(Xoshiro(0), emitter, archive)
    evs = QD.evaluate_batch(prob, xs)
    QD.tell!(emitter, xs, evs, QD.add_batch!(archive, xs, evs))
    return xs
end

@testset "evaluate_gradient" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], 2x[2])); init=rng -> randn(rng, 2), bounds=((-3, 3), (-6, 6)))
    ev, jacobian = QD.evaluate_gradient(prob, [1.0, 2.0])
    @test ev == QD.evaluate(prob, [1.0, 2.0])
    @test jacobian == [-2.0 -4.0; 1.0 0.0; 0.0 2.0]

    evs, jacobians = QD.evaluate_gradient_batch(prob, [[1.0, 2.0], [0.0, -1.0]])
    @test evs == [QD.evaluate(prob, [1.0, 2.0]), QD.evaluate(prob, [0.0, -1.0])]
    @test jacobians == [[-2.0 -4.0; 1.0 0.0; 0.0 2.0], [0.0 2.0; 1.0 0.0; 0.0 2.0]]

    # Through `evaluate` for a problem that is not a QDProblem.
    struct Paraboloid <: QD.AbstractQDProblem end
    QD.evaluate(::Paraboloid, x) = Evaluation(-x[1]^2 - 3x[2]^2, SVector(x[1] * x[2], x[2]))
    _, jacobian = QD.evaluate_gradient(Paraboloid(), [1.0, 2.0])
    @test jacobian == [-2.0 -12.0; 2.0 1.0; 0.0 1.0]

    noisy = StochasticQDProblem((rng, x) -> (sum(x), (x[1], x[2])); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 1)))
    @test_throws "StochasticQDProblem is a stochastic problem, which has no gradient" QD.evaluate_gradient(noisy, [0.5, 0.5])
    two = QDProblem(x -> ([x[1], x[2]], (x[1], x[2])); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 1)))
    @test_throws "needs a single real fitness, got a Vector{" QD.evaluate_gradient(two, [0.5, 0.5])
    @test_throws "QDProblem has no gradient: load ForwardDiff (`using ForwardDiff`)" QD.forwarddiff_evaluate_gradient(prob, (1.0, 2.0))
end

@testset "closed-form gradients of the benchmarks" begin
    # ForwardDiff differentiates the benchmarks through `evaluate`, which computes in the
    # number type of the solution; the closed forms agree with it away from the limits.
    for prob in (Sphere(10), Rastrigin(10))
        x = 3 .* randn(Xoshiro(3), 10)
        x[1], x[2] = 7.0, -6.0
        ev, jacobian = QD.evaluate_gradient(prob, x)
        @test ev == QD.evaluate(prob, x)
        automatic = QD.forwarddiff_evaluate_gradient(prob, x)
        @test automatic[1] == ev
        @test jacobian ≈ automatic[2] rtol = 1e-12
        # At a clipping limit the derivative is the one from inside.
        _, at_limit = QD.evaluate_gradient(prob, fill(5.12, 10))
        @test at_limit[2, 1] == (prob isa Sphere ? 1 : 1 / 5)
        @test at_limit[1, 1] != 0
    end
    # Rastrigin(5) puts two entries in the first half and three in the second.
    _, jacobian = QD.evaluate_gradient(Rastrigin(5), zeros(5))
    @test jacobian[2:3, :] == [0.5 0.5 0 0 0; 0 0 1/3 1/3 1/3]
    # BiSphere has two objectives and no gradient for differentiable QD, but its evaluation
    # is generic too.
    bi = ForwardDiff.jacobian(y -> QD.evaluate(BiSphere(4), y).fitness, [0.5, 0.0, -0.5, 1.0])
    @test bi ≈ [1.0 2.0 3.0 0.0; -3.0 -2.0 -1.0 -4.0]
end

@testset "gradient optimizers" begin
    @test_throws "learning_rate must be positive and finite, got 0" QD.GradientAscent(0)
    @test_throws "learning_rate must be positive and finite, got Inf" QD.AdamAscent(; learning_rate=Inf)
    @test_throws "beta1 must be in [0, 1), got 1" QD.AdamAscent(; beta1=1)
    @test_throws "beta2 must be in [0, 1), got -0.1" QD.AdamAscent(; beta2=-0.1)
    @test_throws "epsilon must be positive and finite, got 0" QD.AdamAscent(; epsilon=0)
    @test QD.AdamAscent().learning_rate == 0.002
    @test QD.AdamAscent(; learning_rate=0.1).learning_rate == 0.1     # a keyword, as for OpenAIES
    @test_throws MethodError QD.AdamAscent(0.1)

    state = QD.AscentState(2)
    QD.reset_ascent!(state, [1.0, 2.0])
    QD.ascend!(state, QD.GradientAscent(0.5), [2.0, -4.0])
    @test state.theta == [2.0, 0.0]
    # Adam's first step moves every entry by the learning rate, up to epsilon.
    QD.reset_ascent!(state, [1.0, 2.0])
    QD.ascend!(state, QD.AdamAscent(; learning_rate=0.1), [3.0, -0.5])
    @test state.theta ≈ [1.1, 1.9] atol = 1e-6
    @test state.t == 1
    QD.reset_ascent!(state, [0.0, 0.0])
    @test state.t == 0 && state.m == [0, 0] && state.v == [0, 0]
end

@testset "GradientArchive" begin
    grid() = GridArchive((4, 4), ((0, 1), (0, 1)))
    @test_throws "the archive to wrap must be empty but holds 1 solutions" GradientArchive((g = grid(); QD.add!(g, [0.0], Evaluation(1.0, SVector(0.1, 0.1))); g))
    @test_throws "a SlidingBoundariesArchive moves its elites" GradientArchive(SlidingBoundariesArchive((4, 4), ((0, 1), (0, 1))))
    @test_throws "a GradientArchive wraps a cell archive with SVector descriptors" GradientArchive(CategoricalArchive((1:3,)))
    @test_throws "not a UnstructuredArchive" GradientArchive(UnstructuredArchive(2; novelty_threshold=0.1, capacity=10))

    a = GradientArchive(grid())
    @test QD.capacity(a) == 16 && length(a) == 0
    ev(f, d) = Evaluation(f, SVector(d...))
    J(v) = fill(Float64(v), 3, 2)
    @test_throws "call add!(archive, x, ev, jacobian)" QD.add!(a, [0.0, 0.0], ev(1.0, (0.1, 0.1)))
    @test_throws "call add_batch!(archive, xs, evs, jacobians)" QD.add_batch!(a, [[0.0, 0.0]], [ev(1.0, (0.1, 0.1))])
    @test_throws "must be 3×2, one row for the fitness and one per descriptor dimension, but it is 2×2" QD.add!(a, [0.0, 0.0], ev(1.0, (0.1, 0.1)), zeros(2, 2))
    @test_throws "the Jacobian has entries that are not finite" QD.add!(a, [0.0, 0.0], ev(1.0, (0.1, 0.1)), fill(NaN, 3, 2))
    @test_throws "got 1 solutions, 1 evaluations and 2 Jacobians" QD.add_batch!(a, [[0.0, 0.0]], [ev(1.0, (0.1, 0.1))], [J(1), J(2)])
    @test length(a) == 0

    @test QD.add!(a, [1.0, 1.0], ev(1.0, (0.1, 0.1)), J(1)).status == QD.NewCell
    i = QD.cell_index(a, SVector(0.1, 0.1))
    @test a.gradients[i] == J(1)
    # Of a batch, the cell keeps the gradients of its best solution, the first on a tie, and
    # rejected solutions leave the gradients alone.
    xs = [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]
    evs = [ev(0.5, (0.1, 0.1)), ev(2.0, (0.1, 0.1)), ev(3.0, (0.12, 0.1)), ev(3.0, (0.1, 0.12)), ev(7.0, (0.9, 0.9))]
    results = QD.add_batch!(a, xs, evs, [J(2), J(3), J(4), J(5), J(6)])
    @test [r.status for r in results] == [QD.Rejected, QD.Improved, QD.Improved, QD.Improved, QD.NewCell]
    @test a.gradients[i] == J(4)
    @test a.gradients[QD.cell_index(a, SVector(0.9, 0.9))] == J(6)
    @test first(QD.elites(a.inner)) == ([4.0, 4.0], ev(3.0, (0.12, 0.1)))
    @test length(a) == 2 && QD.max_fitness(a) == 7.0 && QD.qd_score(a) == 10.0 && coverage(a) == 2 / 16
    @test QD.result_archive(a) === a.inner
    @test length(QD.sample_elites(Xoshiro(1), a, 3)) == 3
    @test QD.descriptor_bounds(a) == QD.descriptor_bounds(a.inner)
    empty!(a)
    @test length(a) == 0

    # Around a ThresholdArchive, the result archive is that of the threshold archive.
    t = GradientArchive(ThresholdArchive(grid(); learning_rate=0.5, threshold_min=0))
    QD.add_batch!(t, [[1.0, 1.0], [2.0, 2.0]], [ev(1.0, (0.1, 0.1)), ev(0.8, (0.1, 0.1))], [J(1), J(2)])
    @test t.gradients[i] == J(1)
    @test QD.result_archive(t) === t.inner.result
    @test length(QD.result_archive(t)) == 1
end

@testset "GradientOperatorEmitter" begin
    @test_throws "sigma0 must be positive and finite, got 0" GradientOperatorEmitter(; sigma0=0)
    @test_throws "batch_size must be at least 1, got 0" GradientOperatorEmitter(; sigma0=1, batch_size=0)
    @test_throws "only a GradientArchive keeps, not a GridArchive" QD.ask(Xoshiro(1), GradientOperatorEmitter(; sigma0=1), GridArchive((2,), ((0, 1),)))
    @test_throws "cannot sample elites from an empty archive" QD.ask(Xoshiro(1), GradientOperatorEmitter(; sigma0=1), GradientArchive(GridArchive((2,), ((0, 1),))))

    a = GradientArchive(GridArchive((4, 4), ((0, 1), (0, 1))))
    jacobian = [3.0 4.0; 0.0 2.0; -1.0 0.0]
    QD.add!(a, [1.0, 1.0], Evaluation(1.0, SVector(0.1, 0.1)), jacobian)
    for normalize in (true, false)
        child = only(QD.ask(Xoshiro(7), GradientOperatorEmitter(; sigma0=2.0, batch_size=1, normalize), a))
        rng = Xoshiro(7)
        rand(rng, a.inner.occupied_idx)
        c = 2.0 .* [randn(rng) for _ in 1:3]
        c[1] = abs(c[1])
        rows = normalize ? jacobian ./ ([5.0, 2.0, 1.0] .+ QD.GRADIENT_EPSILON) : jacobian
        @test child ≈ [1.0, 1.0] .+ rows' * c
    end
    @test length(QD.ask(Xoshiro(1), GradientOperatorEmitter(; sigma0=1), a)) == 64

    prob = smooth_problem(4)
    archive = GradientArchive(GridArchive(prob; dims=(5, 5)))
    QD.add_batch!(archive, [fill(0.1, 4), fill(-0.2, 4)], QD.evaluate_gradient_batch(prob, [fill(0.1, 4), fill(-0.2, 4)])...)
    test_emitter(GradientOperatorEmitter(; sigma0=0.5, batch_size=8), archive, prob)
end

@testset "GradientArborescenceEmitter matches pyribs" begin
    for (name, optimizer, thetas, branches, orders, parents) in
        (("adam", QD.AdamAscent(; learning_rate=0.05), mega_adam_thetas, mega_adam_branches, mega_adam_orders, mega_adam_parents),
         ("ascent", QD.GradientAscent(0.5), mega_ascent_thetas, mega_ascent_branches, mega_ascent_orders, mega_ascent_parents))
        @testset "$name" begin
            emitter = GradientArborescenceEmitter(mega_x0; num_descriptors=2, sigma0=1.0, optimizer,
                                                  strategy=(dim; kwargs...) -> ScriptedStrategy(mega_coefficients),
                                                  restart=QD.ConvergenceRestart())
            archive = GridArchive((10, 10), ((-2, 2), (-2, 2)); outside=ClampOutside())
            for r in eachindex(thetas)
                xs = mega_round!(emitter, archive, mega_problem)
                @test reduce(hcat, xs) ≈ branches[r] rtol = 1e-12
                @test emitter.es.told[r] == (orders[r], parents[r])
                @test emitter.ascent.theta ≈ thetas[r] rtol = 1e-12
            end
            @test emitter.iterations == length(thetas) && emitter.restarts == 0
        end
    end
end

@testset "GradientArborescenceEmitter" begin
    @test_throws "num_descriptors must be at least 1, got 0" GradientArborescenceEmitter(3; num_descriptors=0, sigma0=1)
    @test_throws "dim must be at least 1, got 0" GradientArborescenceEmitter(0; num_descriptors=2, sigma0=1)
    @test_throws "x0 has 2 entries but the dimension is 3" GradientArborescenceEmitter(3; num_descriptors=2, sigma0=1, x0=[0, 0])
    @test_throws "strategy must construct an AbstractEvolutionStrategy" GradientArborescenceEmitter(3; num_descriptors=2, sigma0=1, strategy=(d; kw...) -> 1)

    prob = smooth_problem(4)
    emitter = GradientArborescenceEmitter(4; num_descriptors=2, sigma0=0.5)
    @test QD.batch_size(emitter) == 4 + floor(Int, 3 * log(3))
    @test QD.restarts(emitter) == 0
    @test QD.uses_gradients(emitter) && !QD.uses_gradients(CMAEmitter(4; sigma0=1.0))
    archive = GridArchive(prob; dims=(5, 5))
    QD.add!(archive, fill(0.3, 4), QD.evaluate(prob, fill(0.3, 4)))

    @test_throws "asks for its candidates after ask_gradients and tell_gradients!" QD.ask(Xoshiro(1), emitter, archive)
    @test_throws "but none was asked for" QD.tell_gradients!(emitter, [zeros(4)], [], [zeros(3, 4)], [])
    point = QD.ask_gradients(Xoshiro(1), emitter, archive)
    @test point == [fill(0.3, 4)]    # the only elite
    @test point[1] !== emitter.ascent.theta
    @test_throws "expected the gradients of one point but got 2" QD.tell_gradients!(emitter, point, [], [zeros(3, 4), zeros(3, 4)], [])
    @test_throws "the Jacobian is 2×4 but the emitter expects 3×4" QD.tell_gradients!(emitter, point, [], [zeros(2, 4)], [])
    _, jacobians = QD.evaluate_gradient_batch(prob, point)
    QD.tell_gradients!(emitter, point, [], jacobians, [])
    @test [norm(r) for r in eachrow(emitter.jacobian)] ≈ [1, 1, 1] rtol = 1e-7
    @test_throws "tell! needs the results of the batch" QD.tell!(emitter, [], [], [])
    xs = QD.ask(Xoshiro(1), emitter, archive)
    @test_throws "once per round" QD.ask(Xoshiro(1), emitter, archive)
    @test_throws "tell! must follow the ask" QD.ask_gradients(Xoshiro(1), emitter, archive)
    @test_throws "expected 7 solutions, evaluations and results but got 1, 7 and 7" QD.tell!(emitter, xs[1:1], xs, xs)
    # A batch that the archive rejects entirely restarts the search from an elite.
    rejected = [QD.AddResult(QD.Rejected, -1.0) for _ in xs]
    QD.tell!(emitter, xs, QD.evaluate_batch(prob, xs), rejected)
    @test emitter.restarts == 1 && emitter.iterations == 1
    @test emitter.ascent.theta == fill(0.3, 4)    # no parent, no step

    # Without normalization the gradients are kept as they are.
    raw = GradientArborescenceEmitter(4; num_descriptors=2, sigma0=0.5, normalize=false, x0=fill(0.1, 4))
    point = QD.ask_gradients(Xoshiro(1), raw, archive)
    @test point == [fill(0.1, 4)]
    _, jacobians = QD.evaluate_gradient_batch(prob, point)
    QD.tell_gradients!(raw, point, [], jacobians, [])
    @test raw.jacobian == only(jacobians)

    filled() = (a = GridArchive(prob; dims=(5, 5)); QD.add!(a, fill(0.3, 4), QD.evaluate(prob, fill(0.3, 4))); a)
    for optimizer in (QD.AdamAscent(), QD.GradientAscent(1.0)), ranker in (QD.TwoStageImprovementRanker(), QD.ImprovementRanker())
        test_emitter(GradientArborescenceEmitter(4; num_descriptors=2, sigma0=0.5, optimizer, ranker), filled(), prob)
    end
    test_emitter(GradientArborescenceEmitter(4; num_descriptors=2, sigma0=0.5, strategy=QD.SepCMAES, batch_size=6), filled(), prob)
    test_emitter(GradientArborescenceEmitter(4; num_descriptors=2, sigma0=0.5), GradientArchive(ThresholdArchive(GridArchive(prob; dims=(5, 5)))) |>
                 (a -> (QD.add!(a, fill(0.3, 4), QD.evaluate_gradient(prob, fill(0.3, 4))...); a)), prob)

    @test_throws "pool[2] is a GradientArborescenceEmitter, which needs gradients before it asks" BanditScheduler((CMAEmitter(4; sigma0=1.0), emitter); num_active=1)
end

@testset "CMAMEGA, CMAMAEGA and OMGMEGA" begin
    prob = Sphere(20)
    algs = (CMAMEGA(prob; dims=(20, 20), sigma0=10.0),
            CMAMEGA(prob; dims=(20, 20), sigma0=10.0, optimizer=QD.GradientAscent(1.0), num_emitters=2, batch_size=10),
            CMAMAEGA(prob; dims=(20, 20), sigma0=10.0, threshold_min=0),
            OMGMEGA(prob; dims=(20, 20), sigma0=10.0, batch_size=16))
    @test length(algs[2].emitters) == 2 && QD.batch_size(algs[2].emitters[1]) == 10
    @test algs[3].archive isa ThresholdArchive && algs[3].emitters[1].selection isa QD.MuSelection
    @test algs[4].archive isa GradientArchive && algs[4].emitters[1] isa GradientOperatorEmitter
    for alg in algs
        state = init(prob, alg; budget=2_000, rng=Xoshiro(1), init_size=20)
        @test state.n_evals == 20
        step!(state)
        # A round of a gradient arborescence also evaluates its point.
        per_round = sum(e -> QD.batch_size(e) + QD.uses_gradients(e), alg.emitters)
        @test state.n_evals == 20 + per_round
        result = solve!(state)
        again = solve(prob, alg; budget=2_000, rng=Xoshiro(1), init_size=20)
        @test result.history == again.history
        @test coverage(result.archive) > 0.1
    end

    # CVT archive and a QDProblem differentiated by ForwardDiff.
    smooth = smooth_problem(6)
    cvt = CVTArchive(30, QD.descriptor_bounds(smooth); samples=3_000, rng=Xoshiro(1))
    for alg in (CMAMEGA(cvt; sigma0=1.0, dim=6), OMGMEGA(GradientArchive(cvt); sigma0=1.0, batch_size=8))
        result = solve(smooth, alg; budget=500, rng=Xoshiro(2))
        @test length(result.archive) > 5
    end
    @test_throws "needs an archive with SVector descriptors, not a CategoricalArchive" CMAMEGA(CategoricalArchive((1:3, 1:3)); sigma0=1.0, dim=2)

    noisy = StochasticQDProblem((rng, x) -> (sum(x), (x[1], x[2])); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 1)))
    @test_throws "differentiable quality diversity takes deterministic problems, but StochasticQDProblem is stochastic" init(noisy, CMAMEGA(noisy; dims=(4, 4), sigma0=1.0); budget=100)
    @test_throws "differentiable quality diversity takes deterministic problems" init(noisy, OMGMEGA(noisy; dims=(4, 4), sigma0=1.0); budget=100)
    labelled = QDProblem(x -> (sum(x), (1, 2)); init=rng -> rand(rng, 2), categories=(1:3, 1:3))
    alg = QDAlgorithm(GridArchive((3, 3), ((0, 3), (0, 3))), (GradientArborescenceEmitter(2; num_descriptors=2, sigma0=1.0),))
    @test_throws "needs real descriptors, but QDProblem has descriptor_categories" init(labelled, alg; budget=100)

    # A Jacobian from the problem is checked before anything uses it.
    struct BadGradient <: QD.AbstractQDProblem
        rows::Int
        value::Float64
    end
    QD.evaluate(::BadGradient, x) = Evaluation(-sum(abs2, x), SVector(x[1], x[2]))
    QD.evaluate_gradient(p::BadGradient, x) = (QD.evaluate(p, x), fill(p.value, p.rows, length(x)))
    QD.random_solution(rng, ::BadGradient) = rand(rng, 2)
    QD.descriptor_bounds(::BadGradient) = ((0, 1), (0, 1))
    for (bad, message) in ((BadGradient(3, NaN), "the Jacobian has entries that are not finite"),
                           (BadGradient(2, 1.0), "must be 3×2, one row for the fitness and one per descriptor dimension, but it is 2×2"))
        state = init(bad, CMAMEGA(bad; dims=(4, 4), sigma0=1.0); budget=100, rng=Xoshiro(1))
        @test_throws message step!(state)
        @test_throws message init(bad, OMGMEGA(bad; dims=(4, 4), sigma0=1.0); budget=100)
    end
end

# Draws scripted for GradientOperatorEmitter: `rand(rng, cells)` gives the next cell and
# `randn(rng)` the next coefficient.
struct ScriptedDraws <: AbstractRNG
    cells::Vector{Int}
    coefficients::Vector{Float64}
end
Base.rand(r::ScriptedDraws, ::AbstractVector{Int}) = popfirst!(r.cells)
Random.randn(r::ScriptedDraws) = popfirst!(r.coefficients)
Random.randn(r::ScriptedDraws, ::Type{Float64}) = popfirst!(r.coefficients)

@testset "GradientOperatorEmitter matches pyribs and QDax" begin
    # reference/omg.jl: the children that pyribs's GradientOperatorEmitter (with sigma=0,
    # so that it adds no noise to the parents) and QDax's OMGMEGAEmitter make from known
    # parents and coefficients
    function evaluate(x)
        d = tanh.(omg_A * x)
        J = zeros(3, 5)
        J[1, :] = -2 .* (x .- omg_center)
        J[1, 1] += cos(x[1])
        J[2:3, :] = (1 .- d .^ 2) .* omg_A
        return Evaluation(-sum(abs2, x .- omg_center) + sin(x[1]), SVector{2}(d)), J
    end
    archive = GradientArchive(GridArchive((5, 5), ((-1, 1), (-1, 1))))
    for x in eachcol(omg_population)
        ev, J = evaluate(collect(x))
        QD.add!(archive, collect(x), ev, J)
    end
    cells = archive.inner
    for (ref, rtol) in ((omg_pyribs, 1e-12), (omg_qdax, 1e-7))   # QDax adds no 1e-8 to the length
        parents = [findfirst(i -> cells.occupied[i] && cells.genotypes[i] == p, eachindex(cells.genotypes))
                   for p in eachcol(ref.parents)]
        emitter = GradientOperatorEmitter(; sigma0=ref.sigma, batch_size=length(parents))
        children = QD.ask(ScriptedDraws(parents, vec(ref.coefficients)), emitter, archive)
        @test reduce(hcat, children) ≈ ref.children rtol = rtol
    end
end

