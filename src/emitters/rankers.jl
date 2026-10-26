"""
    AbstractRanker

Supertype of the rules that order the candidates of a [`CMAEmitter`](@ref) from best
to worst after they have been evaluated and added to the archive. The optimizer
moves toward the best ones, so the ranker decides what the emitter looks for.

A ranker defines `rank_key(ranker, ev, result)`, which gives a key for one candidate
from its [`Evaluation`](@ref) and its [`AddResult`](@ref). Candidates are ordered by
descending key; ties keep their order in the batch. A key is a `Float64` or a tuple of
numbers that compares lexicographically. A batch is flat, and the search restarts,
when the best and the worst key are the same.

A ranker with state, such as a random direction, defines
[`reset_ranker!`](@ref QualityDiversity.reset_ranker!), which an emitter calls
whenever a search starts. An emitter keeps its own copy of its ranker.

The rankers are [`TwoStageImprovementRanker`](@ref), [`ImprovementRanker`](@ref),
[`ObjectiveRanker`](@ref), [`TwoStageObjectiveRanker`](@ref),
[`RandomDirectionRanker`](@ref), [`TwoStageRandomDirectionRanker`](@ref) and
[`AcceptedRandomDirectionRanker`](@ref).
"""
abstract type AbstractRanker end

"""
    ObjectiveRanker()

Ranks candidates by fitness alone, whatever the archive did with them: the optimizing
emitter of CMA-ME. With [`MuSelection`](@ref) and [`ConvergenceRestart`](@ref) a
[`CMAEmitter`](@ref) with this ranker is a plain CMA-ES that maximizes the fitness; with
the default [`FilterSelection`](@ref) and [`NoImprovementRestart`](@ref) it recombines
as many of the fittest candidates as the archive accepted and restarts when it accepts
none, as in pyribs.
"""
struct ObjectiveRanker <: AbstractRanker end

"""
    ImprovementRanker()

Ranks candidates by the `improvement` of their [`AddResult`](@ref), the gain over the
threshold of the cell. This is the ranking of CMA-MAE, for use with a
[`ThresholdArchive`](@ref).

With a [`DensityArchive`](@ref), whose `improvement` is the negated density, it ranks
the candidates in the sparsest regions of descriptor space first, the ranking of Density
Descent Search ([`DDS`](@ref)).

For an archive where a new cell reports its fitness as `improvement`, as the other
cell archives do, the improvements of new and old cells are on different scales;
[`TwoStageImprovementRanker`](@ref) is the ranking for those.
"""
struct ImprovementRanker <: AbstractRanker end

"""
    TwoStageImprovementRanker()

Ranks candidates first by what the archive did with them, a new cell above an
improved cell above a rejection, and then by `improvement`. This is the ranking of the
improvement emitter of CMA-ME (Fontaine et al., 2020).
"""
struct TwoStageImprovementRanker <: AbstractRanker end

"""
    TwoStageObjectiveRanker()

Ranks candidates first by what the archive did with them, as
[`TwoStageImprovementRanker`](@ref) does, and then by fitness. A new cell ranks above an
improved cell, whatever the fitness.
"""
struct TwoStageObjectiveRanker <: AbstractRanker end

"""
    DirectionRanker

Supertype of the rankers that order candidates by their descriptor projected on a
random direction: [`RandomDirectionRanker`](@ref),
[`TwoStageRandomDirectionRanker`](@ref) and [`AcceptedRandomDirectionRanker`](@ref). The direction is drawn when a search starts, from
a standard normal vector scaled by the extent of the archive along each descriptor
dimension, so a search that follows it moves toward one part of descriptor space. The
archive must report its extent with
[`descriptor_bounds`](@ref QualityDiversity.descriptor_bounds), and its descriptors must
be real vectors.
"""
abstract type DirectionRanker <: AbstractRanker end

"""
    RandomDirectionRanker()

Ranks candidates by the projection of their descriptor on a random direction, the
random direction emitter of CMA-ME (Fontaine et al., 2020). See [`DirectionRanker`](@ref).
The ranker has the property `direction`, empty until a search starts.
"""
struct RandomDirectionRanker <: DirectionRanker
    direction::Vector{Float64}
end
RandomDirectionRanker() = RandomDirectionRanker(Float64[])

"""
    TwoStageRandomDirectionRanker()

Ranks candidates first by what the archive did with them, as
[`TwoStageImprovementRanker`](@ref) does, and then by the projection of their descriptor
on a random direction. A new cell ranks above an improved cell whatever the projection;
this is the `2rd` ranker of pyribs. See [`DirectionRanker`](@ref) and, for the ranking
that the CMA-ME paper describes, [`AcceptedRandomDirectionRanker`](@ref).
"""
struct TwoStageRandomDirectionRanker <: DirectionRanker
    direction::Vector{Float64}
end
TwoStageRandomDirectionRanker() = TwoStageRandomDirectionRanker(Float64[])

"""
    AcceptedRandomDirectionRanker()

Ranks the candidates that the archive accepted, a new cell or an improvement, above
those it rejected, and each group by the projection of the descriptor on a random
direction. A new cell and an improved cell are not told apart. This is the random
direction emitter as Fontaine et al. (2020) describe it, where only the solutions that
improve the archive are ranked by their projection; with [`FilterSelection`](@ref) the
recombined candidates are the accepted ones in that order. See [`DirectionRanker`](@ref).
"""
struct AcceptedRandomDirectionRanker <: DirectionRanker
    direction::Vector{Float64}
end
AcceptedRandomDirectionRanker() = AcceptedRandomDirectionRanker(Float64[])

"""
    rank_key(ranker::AbstractRanker, ev::Evaluation, result::AddResult)

The key that orders one candidate: a `Float64`, or a tuple that compares
lexicographically, where a larger key ranks earlier. `ev` is the evaluation of the
candidate and `result` what the archive made of it. A new ranker defines this method.
"""
function rank_key end

"""
    reset_ranker!(rng, ranker::AbstractRanker, archive)

Prepare `ranker` for a new search. An emitter calls it when its first search starts and
after every restart. The default does nothing; [`DirectionRanker`](@ref)s draw a new
direction.
"""
reset_ranker!(rng, ranker::AbstractRanker, archive) = nothing

function reset_ranker!(rng, ranker::DirectionRanker, archive)
    bounds = descriptor_bounds(archive)
    bounds === nothing &&
        throw(ArgumentError("$(nameof(typeof(archive))) does not report the bounds of its descriptor space, from which a $(nameof(typeof(ranker))) draws its directions"))
    resize!(ranker.direction, length(bounds))
    for (i, (lo, hi)) in enumerate(bounds)
        ranker.direction[i] = randn(rng) * (hi - lo)
    end
    return nothing
end

rank_key(::ObjectiveRanker, ev, result) = Float64(ev.fitness)

rank_key(::ImprovementRanker, ev, result) = Float64(result.improvement)

status_stage(result) = result.status == NewCell ? 2 : result.status == Improved ? 1 : 0

function rank_key(::TwoStageImprovementRanker, ev, result)
    return (status_stage(result), Float64(result.improvement))
end

rank_key(::TwoStageObjectiveRanker, ev, result) = (status_stage(result), Float64(ev.fitness))

function projection(ranker::DirectionRanker, ev)
    direction = ranker.direction
    isempty(direction) &&
        throw(ArgumentError("the direction of the ranker is not set; reset_ranker! draws it when a search starts"))
    length(ev.descriptor) == length(direction) ||
        throw(ArgumentError("the descriptor has $(length(ev.descriptor)) dimensions but the direction has $(length(direction))"))
    return sum(i -> direction[i] * Float64(ev.descriptor[i]), eachindex(direction))
end

rank_key(ranker::RandomDirectionRanker, ev, result) = projection(ranker, ev)

function rank_key(ranker::TwoStageRandomDirectionRanker, ev, result)
    return (status_stage(result), projection(ranker, ev))
end

function rank_key(ranker::AcceptedRandomDirectionRanker, ev, result)
    return (Int(result.status != Rejected), projection(ranker, ev))
end

# Order of the candidates from best to worst, and their keys in that order.
function rank(ranker::AbstractRanker, evs, results)
    keys = [rank_key(ranker, ev, result) for (ev, result) in zip(evs, results)]
    order = sortperm(keys; rev=true, alg=MergeSort)
    return order, keys[order]
end
