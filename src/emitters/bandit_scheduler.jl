"""
    ReselectRule

Supertype of the rules that tell a [`BanditScheduler`](@ref) which active emitters to
take out of the active set after a round: [`ReselectTerminated`](@ref) and
[`ReselectAll`](@ref). An emitter taken out competes for the free slots with the inactive
ones and may be chosen again.
"""
abstract type ReselectRule end

"""
    ReselectTerminated()

Take an active emitter out of the active set when it restarted during the round, as
[`restarts`](@ref QualityDiversity.restarts) tells, and always if it cannot restart,
such as a [`VariationEmitter`](@ref). The default of [`BanditScheduler`](@ref), as in
pyribs and in Cully (2021).
"""
struct ReselectTerminated <: ReselectRule end

"""
    ReselectAll()

Take every active emitter out of the active set after each round, so that the bandit
chooses all of them again.
"""
struct ReselectAll <: ReselectRule end

"""
    BanditScheduler(pool; n_active, zeta=0.05, window=50, reselect=ReselectTerminated())

An emitter that holds a pool of emitters and runs `n_active` of them each round,
chosen by a multi-armed bandit (Cully, 2021, "Multi-Emitter MAP-Elites"; pyribs
`BanditScheduler`). Emitters that keep adding solutions to the archive run more often.

`pool` is a tuple (or a vector) of emitters for the same genotype type. The first round
runs the first `n_active` of them. `ask` returns the batches of the active emitters one
after the other, in the order of the pool, and [`batch_size`](@ref
QualityDiversity.batch_size) is the sum of their batch sizes, so emitters of different
batch sizes can share a pool. `tell!` gives each active emitter its part of the batch and
counts, for each emitter, the solutions it produced (selections) and those the archive
accepted with `NewCell` or `Improved` (successes), over the last `window` rounds.

Then it chooses the emitters of the next round. `reselect` says which active emitters
leave the active set: [`ReselectTerminated`](@ref) (the default) or
[`ReselectAll`](@ref). The free slots go to the inactive emitters, the ones that just
left included, with the highest UCB1 score

    successes / selections + zeta * sqrt(log(total selections) / selections)

as in Algorithm 1 of the paper. An emitter without selections in the window scores
`Inf`, so every emitter is tried, and a tie goes to the emitter that comes first in the
pool. pyribs takes the log of the total successes instead and counts the whole run.

A mutable emitter, such as a [`CMAEmitter`](@ref), may appear only once in `pool`; build
a pool of several with a comprehension, not by repeating one instance. `tell!` must
follow the `ask` that produced the batch. The emitters of the pool must return their
batches as a `Vector`, so device emitters cannot be scheduled.

# Example

```julia
prob = QualityDiversity.Benchmarks.Rastrigin(10)
archive = GridArchive(prob; dims=(20, 20))
pool = (VariationEmitter(GaussianMutation(0.1); batch_size=36),
        VariationEmitter(IsoLineVariation(0.01, 0.1); batch_size=36),
        (CMAEmitter(10; sigma0=0.5, batch_size=36) for _ in 1:4)...)
alg = QDAlgorithm(archive, (BanditScheduler(pool; n_active=2),))
result = solve(prob, alg; budget=50_000, rng=Xoshiro(1))
```
"""
mutable struct BanditScheduler{E<:Tuple,R<:ReselectRule} <: AbstractEmitter
    const pool::E
    const n_active::Int
    const zeta::Float64
    const reselect::R
    const active::Vector{Bool}
    # Selections and successes of every emitter (columns) in each of the last rounds
    # (rows, a ring: round `r` is in row `mod1(r, window)`).
    const round_selections::Matrix{Int}
    const round_successes::Matrix{Int}
    # Their sums over the rows.
    const selections::Vector{Int}
    const successes::Vector{Int}
    # What `restarts` returned for each emitter at the end of the last round.
    const seen_restarts::Vector{Int}
    # Rounds told so far.
    rounds::Int
    # A batch was asked and is waiting for its results.
    asked::Bool
end

function BanditScheduler(pool::Tuple; n_active::Integer, zeta::Real=0.05, window::Integer=50,
                         reselect::ReselectRule=ReselectTerminated())
    P = length(pool)
    P >= 1 || throw(ArgumentError("the pool must hold at least one emitter"))
    for (i, e) in enumerate(pool)
        e isa AbstractEmitter ||
            throw(ArgumentError("pool[$i] is a $(typeof(e)), not an AbstractEmitter"))
        uses_gradients(e) &&
            throw(ArgumentError("pool[$i] is a $(nameof(typeof(e))), which needs gradients before it asks; a BanditScheduler cannot hold it"))
        for j in 1:i-1
            ismutable(e) && e === pool[j] &&
                throw(ArgumentError("pool[$j] and pool[$i] are the same $(nameof(typeof(e))); every emitter with state must be a separate instance, e.g. built with a comprehension"))
        end
    end
    1 <= n_active <= P ||
        throw(ArgumentError("n_active must be between 1 and the size of the pool ($P), got $n_active"))
    (isfinite(zeta) && zeta >= 0) ||
        throw(ArgumentError("zeta must be finite and at least 0, got $zeta"))
    window >= 1 || throw(ArgumentError("window must be at least 1, got $window"))
    active = [i <= n_active for i in 1:P]
    seen = Int[something(restarts(e), 0) for e in pool]
    return BanditScheduler{typeof(pool),typeof(reselect)}(pool, Int(n_active), Float64(zeta), reselect,
                                                          active, zeros(Int, window, P), zeros(Int, window, P),
                                                          zeros(Int, P), zeros(Int, P), seen, 0, false)
end

BanditScheduler(pool::AbstractVector; kwargs...) = BanditScheduler(Tuple(pool); kwargs...)

"""
    active(scheduler::BanditScheduler) -> Vector{Bool}

Which emitters of the pool run in the next round, in the order of the pool.
"""
active(scheduler::BanditScheduler) = copy(scheduler.active)

"""
    restarts(emitter::AbstractEmitter) -> Union{Int,Nothing}

How many times `emitter` has restarted its search, or `nothing` for an emitter that
never restarts. Optional: the default is `nothing`. A [`BanditScheduler`](@ref) with
[`ReselectTerminated`](@ref) reads it after every round.
"""
restarts(::AbstractEmitter) = nothing

restarts(emitter::CMAEmitter) = emitter.restarts

function batch_size(scheduler::BanditScheduler)
    return active_sum(batch_size, scheduler.pool, scheduler.active, 1)
end

# The sum of `f(e)` over the active emitters `e` of `pool`, the first of which is
# emitter `i` of the scheduler. Recursion over the tuple dispatches each emitter type
# statically.
active_sum(f, ::Tuple{}, active, i) = 0
function active_sum(f, pool::Tuple, active, i)
    n = active[i] ? f(first(pool)) : 0
    return n + active_sum(f, Base.tail(pool), active, i + 1)
end

function ask(rng::AbstractRNG, scheduler::BanditScheduler, archive::AbstractArchive{G}) where {G}
    scheduler.asked &&
        throw(ArgumentError("ask was called again before tell! reported the results of the last batch"))
    xs = Vector{G}(undef, batch_size(scheduler))
    ask_active!(xs, 1, rng, scheduler.pool, scheduler.active, 1, archive)
    scheduler.asked = true
    return xs
end

function ask(::AbstractRNG, ::BanditScheduler, ::DeviceCellArchive)
    throw(ArgumentError("a BanditScheduler collects the batches of its emitters in a Vector and cannot run on a DeviceCellArchive"))
end

ask_active!(xs, pos, rng, ::Tuple{}, active, i, archive) = nothing
function ask_active!(xs, pos, rng, pool::Tuple, active, i, archive)
    if active[i]
        emitter = first(pool)
        batch = ask(rng, emitter, archive)
        n = batch_size(emitter)
        length(batch) == n ||
            throw(ArgumentError("emitter $i of the pool returned $(length(batch)) solutions but its batch_size is $n"))
        copyto!(xs, pos, batch, firstindex(batch), n)
        pos += n
    end
    return ask_active!(xs, pos, rng, Base.tail(pool), active, i + 1, archive)
end

"""
    tell!(scheduler::BanditScheduler, xs, evs, results)

Give each active emitter its part of the batch that the last `ask` returned, count the
selections and successes of each, and choose the emitters of the next round.
"""
function tell!(scheduler::BanditScheduler, xs, evs, results)
    scheduler.asked ||
        throw(ArgumentError("tell! needs the results of the batch that the last ask returned, but nothing was asked since the last tell!"))
    n = batch_size(scheduler)
    (length(xs) == n && length(evs) == n && length(results) == n) ||
        throw(ArgumentError("expected $n solutions, evaluations and results but got $(length(xs)), $(length(evs)) and $(length(results))"))
    Base.require_one_based_indexing(xs, evs, results)
    scheduler.asked = false

    window = size(scheduler.round_selections, 1)
    scheduler.rounds += 1
    row = mod1(scheduler.rounds, window)
    for e in eachindex(scheduler.selections)
        # the row about to be overwritten leaves the window
        scheduler.selections[e] -= scheduler.round_selections[row, e]
        scheduler.successes[e] -= scheduler.round_successes[row, e]
        scheduler.round_selections[row, e] = 0
        scheduler.round_successes[row, e] = 0
    end
    tell_active!(scheduler, 1, scheduler.pool, 1, row, xs, evs, results)
    reselect!(scheduler)
    return nothing
end

tell_active!(scheduler, pos, ::Tuple{}, i, row, xs, evs, results) = nothing
function tell_active!(scheduler, pos, pool::Tuple, i, row, xs, evs, results)
    if scheduler.active[i]
        emitter = first(pool)
        n = batch_size(emitter)
        part = pos:pos+n-1
        tell!(emitter, view(xs, part), view(evs, part), view(results, part))
        accepted = count(r -> r.status != Rejected, view(results, part))
        scheduler.round_selections[row, i] = n
        scheduler.round_successes[row, i] = accepted
        scheduler.selections[i] += n
        scheduler.successes[i] += accepted
        pos += n
    end
    return tell_active!(scheduler, pos, Base.tail(pool), i + 1, row, xs, evs, results)
end

function reselect!(scheduler::BanditScheduler)
    release!(scheduler, scheduler.pool, 1)
    total = sum(scheduler.selections)
    for _ in count(scheduler.active)+1:scheduler.n_active
        best = argmax(i -> ucb1(scheduler.successes[i], scheduler.selections[i], total, scheduler.zeta),
                      findall(!, scheduler.active))
        scheduler.active[best] = true
    end
    return nothing
end

# Takes the active emitters that the reselection rule names out of the active set, and
# notes how often every emitter has restarted.
release!(scheduler, ::Tuple{}, i) = nothing
function release!(scheduler, pool::Tuple, i)
    count_restarts = restarts(first(pool))
    if scheduler.active[i] && leaves(scheduler.reselect, count_restarts, scheduler.seen_restarts[i])
        scheduler.active[i] = false
    end
    scheduler.seen_restarts[i] = something(count_restarts, 0)
    return release!(scheduler, Base.tail(pool), i + 1)
end

leaves(::ReselectAll, count_restarts, seen) = true
leaves(::ReselectTerminated, ::Nothing, seen) = true
leaves(::ReselectTerminated, count_restarts::Integer, seen) = count_restarts > seen

ucb1(successes, selections, total, zeta) =
    selections == 0 ? Inf : successes / selections + zeta * sqrt(log(total) / selections)
