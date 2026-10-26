const MetricsRecord = @NamedTuple{n_evals::Int, qd_score::Float64, coverage::Float64,
                                  max_fitness::Float64, mean_fitness::Float64}

# The record of the metrics of an archive that `history` holds. An archive with other
# metrics defines `metrics_record_type` and `metrics_record`.
metrics_record_type(::Type{<:AbstractArchive}) = MetricsRecord

function metrics_record(archive::AbstractArchive, n_evals, offset)
    return (n_evals=n_evals,
            qd_score=Float64(qd_score(archive; offset)),
            coverage=Float64(coverage(archive)),
            max_fitness=Float64(max_fitness(archive)),
            mean_fitness=Float64(mean_fitness(archive)))
end

"""
    QDState

State of a running algorithm, created by `init(prob, alg; ...)`. It holds the
problem, working copies of the algorithm's archive and emitters, the random
number generator, the number of evaluations so far, and `history`, a vector
with the metrics of the result archive after initialization and after every step.

Fields are read with `state.archive`, `state.n_evals`, `state.history`, and so
on.
"""
mutable struct QDState{P<:AbstractQDProblem,A<:AbstractArchive,E<:Tuple,R<:AbstractRNG,C,H<:NamedTuple}
    const problem::P
    const archive::A
    const emitters::E
    const rng::R
    const budget::Int
    const callback::C
    n_evals::Int
    const history::Vector{H}
end

"""
    QDResult

Outcome of [`solve`](@ref): the `archive` of the best solutions found, which is
[`result_archive`](@ref QualityDiversity.result_archive) of the algorithm's
archive, the metrics `history`, and the number of evaluations `n_evals`.
"""
struct QDResult{A<:AbstractArchive,H<:NamedTuple}
    archive::A
    history::Vector{H}
    n_evals::Int
end

Base.show(io::IO, result::QDResult) =
    print(io, "QDResult(", result.n_evals, " evaluations, ", length(result.archive), " cells occupied)")

Base.show(io::IO, state::QDState) =
    print(io, "QDState(", state.n_evals, " of ", state.budget, " evaluations, ", length(state.archive), " cells occupied)")

function Base.show(io::IO, mime::MIME"text/plain", result::QDResult)
    print(io, "QDResult after ", result.n_evals, " evaluations\n  archive: ")
    show(io, mime, result.archive)
    show_metrics(io, result.history)
end

function Base.show(io::IO, mime::MIME"text/plain", state::QDState)
    print(io, "QDState after ", state.n_evals, " of ", state.budget, " evaluations\n  archive: ")
    show(io, mime, state.archive)
    show_metrics(io, state.history)
end

# The latest metrics of a run, one per line.
function show_metrics(io::IO, history)
    isempty(history) && return nothing
    for (name, value) in pairs(last(history))
        name == :n_evals || print(io, "\n  ", name, ": ", value)
    end
    return nothing
end

"""
    init(prob, alg::QDAlgorithm; budget, rng=Random.default_rng(), callback=nothing,
         init_size=<sum of the emitters' batch sizes>) -> QDState

Prepare a run. The archive and emitters of `alg` are copied, so `alg` is not
modified, and the copy is filled with `init_size` solutions from
[`random_solution`](@ref QualityDiversity.random_solution).

`budget` is the number of evaluations, including the initial ones. A problem that
evaluates a solution several times, such as [`Sampled`](@ref), counts every one. A run stops
at the first step that reaches it, so it can overshoot by less than one round
of emitter batches. `callback(state)` is called after initialization and after
every step. Pass a seeded `rng` for a reproducible run.
"""
function CommonSolve.init(prob::AbstractQDProblem, alg::QDAlgorithm;
                          budget::Integer,
                          rng::AbstractRNG=Random.default_rng(),
                          callback=nothing,
                          init_size::Integer=sum(batch_size, alg.emitters))
    init_size >= 1 || throw(ArgumentError("init_size must be at least 1, got $init_size"))
    initial_evals = init_size * evaluations_per_solution(prob)
    budget >= initial_evals ||
        throw(ArgumentError("budget ($budget) is smaller than the initial population ($initial_evals)"))
    check_compatible(prob, alg.archive)
    check_spread_pairing(prob, alg.archive)
    check_device_pairing(prob, alg.archive)
    check_gradient_pairing(prob, alg)
    archive = deepcopy(alg.archive)
    history = metrics_record_type(typeof(result_archive(archive)))[]
    state = QDState(prob, archive, deepcopy(alg.emitters), rng, Int(budget), callback, 0, history)
    xs = initial_batch(state.archive, initial_solutions(rng, prob, state.archive, init_size))
    evaluate_and_add!(state, xs)
    length(state.archive) > 0 ||
        throw(ArgumentError("none of the $init_size initial solutions entered the archive; check that their descriptors lie inside its bounds (an archive with RejectOutside() discards the ones outside)"))
    record_metrics!(state)
    notify!(state.callback, state)
    return state
end

# Fails early, with a clear message, when the archive cannot hold the problem's
# descriptors.
check_compatible(prob, archive) = nothing

function check_compatible(prob::AbstractQDProblem, ::AbstractArchive{G,F,SVector{N,Float64}}) where {G,F,N}
    return check_ndims(prob, N)
end

function check_compatible(prob::AbstractQDProblem, ::AbstractArchive{G,F,D}) where {G,F,D<:Tuple}
    return check_ndims(prob, fieldcount(D))
end

# A `SpreadArchive` takes the evaluations of a `SampledSpread` problem, and only it does.
check_spread_pairing(prob, archive) = nothing
check_spread_pairing(::SampledSpread, ::SpreadArchive) = nothing

function check_spread_pairing(::SampledSpread, archive::AbstractArchive)
    throw(ArgumentError("SampledSpread keeps the descriptors of every evaluation, which only a SpreadArchive takes, not a $(nameof(typeof(archive))); use MELS, or Sampled for the mean descriptor"))
end

function check_spread_pairing(prob::AbstractStochasticQDProblem, ::SpreadArchive)
    throw(ArgumentError("a SpreadArchive needs every evaluation of a solution: wrap the problem as SampledSpread(prob; num_samples)"))
end

function check_spread_pairing(prob::AbstractQDProblem, ::SpreadArchive)
    throw(ArgumentError("a SpreadArchive needs a stochastic problem, but $(nameof(typeof(prob))) is deterministic: every evaluation of a solution is the same and its spread zero"))
end

# A `DeviceCellArchive` evaluates its batches on the device, which takes a deterministic
# problem with descriptor bounds.
check_device_pairing(prob, archive) = nothing

function check_device_pairing(prob::AbstractQDProblem, archive::DeviceCellArchive)
    has_categories(prob) &&
        throw(ArgumentError("a $(nameof(typeof(archive))) needs a problem with descriptor_bounds; one with descriptor_categories runs on the host"))
    return nothing
end

function check_device_pairing(prob::AbstractStochasticQDProblem, archive::DeviceCellArchive)
    throw(ArgumentError("a $(nameof(typeof(archive))) takes deterministic problems; run $(nameof(typeof(prob))) on the host"))
end

# Gradients are computed on the host for deterministic problems with real descriptors.
function check_gradient_pairing(prob, alg::QDAlgorithm)
    alg.archive isa GradientArchive || any(uses_gradients, alg.emitters) || return nothing
    prob isa AbstractStochasticQDProblem &&
        throw(ArgumentError("differentiable quality diversity takes deterministic problems, but $(nameof(typeof(prob))) is stochastic"))
    alg.archive isa DeviceCellArchive &&
        throw(ArgumentError("differentiable quality diversity runs on the host, not on a $(nameof(typeof(alg.archive)))"))
    has_categories(prob) &&
        throw(ArgumentError("differentiable quality diversity needs real descriptors, but $(nameof(typeof(prob))) has descriptor_categories"))
    return nothing
end

function check_ndims(prob, N)
    n = descriptor_ndims(prob)
    n == N || throw(ArgumentError("the problem has $n descriptor dimensions but the archive has $N"))
    return nothing
end

function initial_solutions(rng, prob, ::AbstractArchive{G}, n) where {G}
    xs = Vector{G}(undef, n)
    for i in 1:n
        xs[i] = random_solution(rng, prob)
    end
    return xs
end

# The initial solutions as the archive's emitters hand them over, so that they are
# evaluated where later batches are.
initial_batch(archive, xs) = xs

evaluate_and_add!(state::QDState, xs) = evaluate_and_add!(state, state.archive, xs)

function evaluate_and_add!(state::QDState, ::AbstractArchive, xs)
    evs = evaluate_batch(state.rng, state.problem, xs)
    results = add_batch!(state.archive, xs, evs)
    state.n_evals += length(xs) * evaluations_per_solution(state.problem)
    return evs, results
end

# A `GradientArchive` keeps the gradients of its elites, so every solution of the run is
# evaluated with them.
function evaluate_and_add!(state::QDState, ::GradientArchive, xs)
    evs, _, results = evaluate_gradients_and_add!(state, xs)
    return evs, results
end

function evaluate_gradients_and_add!(state::QDState, xs)
    evs, jacobians = evaluate_gradient_batch(state.problem, xs)
    length(evs) == length(jacobians) == length(xs) ||
        throw(ArgumentError("evaluate_gradient_batch returned $(length(evs)) evaluations and $(length(jacobians)) Jacobians for $(length(xs)) solutions"))
    for (x, ev, jacobian) in zip(xs, evs, jacobians)
        check_jacobian(jacobian, length(ev.descriptor), length(x))
    end
    results = add_evaluated!(state.archive, xs, evs, jacobians)
    state.n_evals += length(xs) * evaluations_per_solution(state.problem)
    return evs, jacobians, results
end

function record_metrics!(state::QDState)
    archive = result_archive(state.archive)
    push!(state.history, metrics_record(archive, state.n_evals, fitness_offset(state.problem)))
    return nothing
end

notify!(::Nothing, state) = nothing
notify!(callback, state) = (callback(state); nothing)

"""
    step!(state::QDState)

One round: for each emitter in turn, ask for candidates, evaluate them, add
them to the archive, and tell the emitter the outcome. Then record the
metrics and call the callback. An emitter that needs gradients first gets the
points it asks for with [`ask_gradients`](@ref QualityDiversity.ask_gradients)
evaluated with their gradients and added to the archive.
"""
function CommonSolve.step!(state::QDState)
    step_emitters!(state, state.emitters)
    record_metrics!(state)
    notify!(state.callback, state)
    return state
end

# Recursion over the tuple unrolls the loop, so each emitter type is dispatched
# statically.
step_emitters!(state::QDState, ::Tuple{}) = nothing
function step_emitters!(state::QDState, emitters::Tuple)
    emitter = first(emitters)
    gradient_round!(state, emitter, ask_gradients(state.rng, emitter, state.archive))
    xs = ask(state.rng, emitter, state.archive)
    evs, results = evaluate_and_add!(state, xs)
    tell!(emitter, xs, evs, results)
    step_emitters!(state, Base.tail(emitters))
end

gradient_round!(state::QDState, emitter, ::Nothing) = nothing

function gradient_round!(state::QDState, emitter, xs)
    evs, jacobians, results = evaluate_gradients_and_add!(state, xs)
    tell_gradients!(emitter, xs, evs, jacobians, results)
    return nothing
end

"""
    solve!(state::QDState) -> QDResult

Step until the evaluation budget is reached. Returns immediately if it already
has been.
"""
function CommonSolve.solve!(state::QDState)
    while state.n_evals < state.budget
        CommonSolve.step!(state)
    end
    return QDResult(result_archive(state.archive), state.history, state.n_evals)
end

"""
    solve(prob, alg::QDAlgorithm; budget, rng=Random.default_rng(), callback=nothing,
          init_size) -> QDResult

Run `alg` on `prob` until `budget` evaluations are spent: `solve!(init(prob, alg; ...))`.
The keyword arguments are those of [`init`](@ref). `alg` is not modified, so it can be
solved again.
"""
function CommonSolve.solve(prob::AbstractQDProblem, alg::QDAlgorithm; kwargs...)
    return solve!(init(prob, alg; kwargs...))
end
