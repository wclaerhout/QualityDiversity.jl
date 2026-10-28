"""
    SelectionRule

Supertype of the rules that say how many of the ranked candidates a
[`CMAEmitter`](@ref) recombines into the next mean: [`FilterSelection`](@ref) and
[`MuSelection`](@ref).
"""
abstract type SelectionRule end

"""
    FilterSelection()

Recombine as many of the first-ranked candidates as the archive accepted, a new cell
or an improvement. With a two-stage ranker, which places the accepted candidates first,
these are the accepted candidates themselves; with another ranker they are the
top-ranked ones, accepted or not. This is the selection of CMA-ME and of pyribs. With
no accepted candidate the distribution does not change.
"""
struct FilterSelection <: SelectionRule end

"""
    MuSelection()

Recombine the best half of the candidates, as CMA-ES does. This is the selection of
CMA-MAE.
"""
struct MuSelection <: SelectionRule end

parent_count(::FilterSelection, added, batch_size) = added
parent_count(::MuSelection, added, batch_size) = batch_size ÷ 2

"""
    RestartRule

Supertype of the rules that tell a [`CMAEmitter`](@ref) when to abandon its search
and start again from a random elite of the archive: [`NoImprovementRestart`](@ref),
[`ConvergenceRestart`](@ref) and [`PeriodicRestart`](@ref). Whatever the rule, the
emitter also restarts when CMA-ES reports that it has converged or broken down.
"""
abstract type RestartRule end

"""
    NoImprovementRestart()

Restart after a batch in which the archive accepted no candidate. This is the rule of
CMA-ME.
"""
struct NoImprovementRestart <: RestartRule end

"""
    ConvergenceRestart()

Restart only when CMA-ES reports that it has converged or broken down.
"""
struct ConvergenceRestart <: RestartRule end

"""
    PeriodicRestart(iterations)

Restart every `iterations` batches.
"""
struct PeriodicRestart <: RestartRule
    iterations::Int
    function PeriodicRestart(iterations::Integer)
        iterations >= 1 || throw(ArgumentError("iterations must be at least 1, got $iterations"))
        return new(iterations)
    end
end

restart_due(::NoImprovementRestart, iterations, added) = added == 0
restart_due(::ConvergenceRestart, iterations, added) = false
restart_due(rule::PeriodicRestart, iterations, added) = iterations % rule.iterations == 0

"""
    CMAEmitter(dim; sigma0, batch_size, strategy, ranker, selection, restart, x0, lower, upper)
    CMAEmitter(x0::AbstractVector; sigma0, ...)

Emitter that searches for solutions with CMA-ES, the covariance matrix adaptation
evolution strategy, in the sense of CMA-ME (Fontaine et al., 2020): it keeps a
Gaussian distribution of solutions, and after every batch it moves the distribution
toward the candidates that its `ranker` places first. The optimizer can be another
evolution strategy. Solutions are real vectors of
length `dim`; the genotype type of the archive can be any `AbstractVector` type that
a vector of `Float64` converts to.

- `sigma0`: the initial step size, the standard deviation of the distribution. It has
  no default, because it depends on the scale of the solutions.
- `batch_size`: the number of candidates per batch, `4 + floor(3 log(dim))` by
  default (rounded up to an even number for [`OpenAIES`](@ref)).
- `strategy`: the optimizer, a subtype of [`AbstractEvolutionStrategy`](@ref) or a
  function with the same arguments as its constructors. It is [`CMAES`](@ref) by
  default; [`SepCMAES`](@ref) and [`LMMAES`](@ref) scale to a high dimension (`LMMAES`
  only works there: it needs at least twice as many parameters as `batch_size`), and
  [`OpenAIES`](@ref) follows a gradient estimate. For an option of a strategy pass
  a function, such as `(dim; kwargs...) -> LMMAES(dim; kwargs..., n_vectors=8)`.
- `ranker`: how the candidates are ordered, [`TwoStageImprovementRanker`](@ref) by
  default. Use [`ImprovementRanker`](@ref) with a [`ThresholdArchive`](@ref),
  [`ObjectiveRanker`](@ref) for an emitter that only maximizes the fitness, and
  [`TwoStageRandomDirectionRanker`](@ref) for one that explores a direction of descriptor
  space. The emitter keeps its own copy of the ranker.
- `selection`: how many of the ranked candidates are recombined,
  [`FilterSelection`](@ref) by default or [`MuSelection`](@ref).
- `restart`: when to start again, [`NoImprovementRestart`](@ref) by default.
- `x0`: the mean of the first distribution. Without it, and after every restart, the
  mean is a random elite of the archive, so the archive must hold a solution when the
  emitter first asks. The second form takes `dim` and `x0` from a vector.
- `lower`, `upper`: bounds of the solutions, scalars or vectors. A sample outside is
  drawn again, and moved inside the bounds after $(RESAMPLE_LIMIT) draws. `x0` must lie
  inside them.

`ask` draws a batch, and the next call of `tell!` must be with the results for that
batch; a second `ask` before that is an error. `tell!` ranks the candidates, adapts the
distribution and decides on a restart; the restart itself happens at the following
`ask`, which has the random number generator and the archive to choose the new start
from. The emitter has the properties `iterations`, the number of batches told, and
`restarts`.

# Example

```jldoctest
julia> using QualityDiversity, QualityDiversity.Benchmarks, Random

julia> prob = Rastrigin(10);

julia> alg = CMAME(prob; dims=(20, 20), sigma0=1.0, n_emitters=3);

julia> result = solve(prob, alg; budget=3_000, rng=Xoshiro(1));

julia> coverage(result.archive) > 0.1
true
```
"""
mutable struct CMAEmitter{R<:AbstractRanker,S<:SelectionRule,T<:RestartRule,E<:AbstractEvolutionStrategy} <: AbstractEmitter
    const es::E
    const ranker::R
    const selection::S
    const restart_rule::T
    const x0::Union{Nothing,Vector{Float64}}
    iterations::Int
    restarts::Int
    # The search has been started: the distribution holds a mean.
    started::Bool
    # The last batch was told, and the next ask starts again from an elite.
    restart_pending::Bool
    # A batch was asked and is waiting for its results.
    asked::Bool
end

function CMAEmitter(dim::Integer; sigma0::Real, batch_size::Union{Nothing,Integer}=nothing,
                    strategy=CMAES, ranker::AbstractRanker=TwoStageImprovementRanker(),
                    selection::SelectionRule=FilterSelection(),
                    restart::RestartRule=NoImprovementRestart(),
                    x0::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    lower=-Inf, upper=Inf)
    sizing = batch_size === nothing ? (;) : (; batch_size)
    es = strategy(dim; sigma0, sizing..., lower, upper)
    es isa AbstractEvolutionStrategy ||
        throw(ArgumentError("strategy must construct an AbstractEvolutionStrategy, got a $(typeof(es))"))
    x0 === nothing || (check_start(x0, dim); check_inside(x0, es.lower, es.upper))
    start = x0 === nothing ? nothing : collect(Float64, x0)
    own_ranker = deepcopy(ranker)
    return CMAEmitter{typeof(own_ranker),typeof(selection),typeof(restart),typeof(es)}(
        es, own_ranker, selection, restart, start, 0, 0, false, false, false)
end

CMAEmitter(x0::AbstractVector{<:Real}; kwargs...) = CMAEmitter(length(x0); x0, kwargs...)

batch_size(emitter::CMAEmitter) = emitter.es.batch_size

# The mean of the next search: `x0` for the first one if the emitter has it, and a
# random elite of the archive otherwise.
function start_mean(rng, emitter::CMAEmitter, archive)
    x0 = emitter.x0
    if !emitter.started && x0 !== nothing
        return x0
    end
    return elite_start(rng, archive, emitter.es)
end

function elite_start(rng, archive, dim)
    elite = sample_elites(rng, archive, 1)[1]
    length(elite) == dim ||
        throw(ArgumentError("the solutions of the archive have $(length(elite)) entries but the emitter has dimension $dim"))
    return collect(Float64, elite)
end

# A random elite, moved inside the bounds of the strategy. An elite can lie outside them:
# `random_solution` and the variation operators of other emitters do not know the bounds,
# and a search that starts outside draws every solution outside.
function elite_start(rng, archive, es::AbstractEvolutionStrategy)
    start = elite_start(rng, archive, es.dim)
    for i in eachindex(start)
        start[i] = clamp(start[i], es.lower[i], es.upper[i])
    end
    return start
end

"""
    ask(rng, emitter::CMAEmitter, archive) -> Vector{G}

A batch of candidates drawn from the search distribution. The first call, and the
first after a restart, starts a new search from a random elite of `archive`, moved
inside the bounds of the strategy if it lies outside them (the first call uses `x0` if
the emitter has one).
"""
function ask(rng::AbstractRNG, emitter::CMAEmitter, archive::AbstractArchive{G}) where {G}
    emitter.asked &&
        throw(ArgumentError("ask was called again before tell! reported the results of the last batch"))
    needs_start(emitter) && start_search!(rng, emitter, archive, start_mean(rng, emitter, archive))
    es = draw_batch!(rng, emitter)
    xs = Vector{G}(undef, es.batch_size)
    for k in 1:es.batch_size
        xs[k] = convert(G, es.x[:, k])
    end
    return xs
end

# Whether the next ask starts a new search: the first one, or one after a restart.
needs_start(emitter::CMAEmitter) = !emitter.started || emitter.restart_pending

# Whether a new search of `emitter` starts from its `x0` rather than from an elite.
starts_from_x0(emitter::CMAEmitter) = !emitter.started && emitter.x0 !== nothing

function start_search!(rng, emitter::CMAEmitter, archive, mean)
    reset!(emitter.es, mean)
    reset_ranker!(rng, emitter.ranker, archive)
    emitter.started = true
    emitter.restart_pending = false
    return nothing
end

# Draws a batch into the strategy and returns it; the batch is in `es.x`.
function draw_batch!(rng, emitter::CMAEmitter)
    sample!(rng, emitter.es)
    emitter.asked = true
    return emitter.es
end

"""
    tell!(emitter::CMAEmitter, xs, evs, results)

Rank the batch that the last `ask` returned, adapt the search distribution to it, and
schedule a restart if the restart rule or CMA-ES asks for one. `xs`, `evs` and
`results` must be those of that batch, in the order of `ask`.
"""
function tell!(emitter::CMAEmitter, xs, evs, results)
    emitter.asked ||
        throw(ArgumentError("tell! needs the results of the batch that the last ask returned, but nothing was asked since the last tell!"))
    es = emitter.es
    (length(xs) == es.batch_size && length(evs) == es.batch_size && length(results) == es.batch_size) ||
        throw(ArgumentError("expected $(es.batch_size) solutions, evaluations and results but got $(length(xs)), $(length(evs)) and $(length(results))"))
    emitter.asked = false
    emitter.iterations += 1
    added = count(r -> r.status != Rejected, results)
    order, keys = rank(emitter.ranker, evs, results)
    update!(es, order, parent_count(emitter.selection, added, es.batch_size))
    if should_stop(es, keys) || restart_due(emitter.restart_rule, emitter.iterations, added)
        emitter.restart_pending = true
        emitter.restarts += 1
    end
    return nothing
end
