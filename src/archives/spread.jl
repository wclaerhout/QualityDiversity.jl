"""
    SpreadArchive(archive::Union{GridArchive,CVTArchive})

Archive for MAP-Elites with low spread (Mace et al., 2023, "The Quality-Diversity
Transformer"), the archive of [`MELS`](@ref): a cell archive that prefers, among
solutions of a noisy problem, those whose evaluations land close together. `archive`
decides which cell a descriptor belongs to; it must be empty, and the new archive uses
it without copying.

The archive takes the evaluations of a [`SampledSpread`](@ref) problem, whose descriptor
holds the descriptors of all the evaluations of a solution. From them it finds

- the cell of the solution: the one its descriptors land in most often, the one visited
  first on a tie;
- its spread: the mean distance between two of its descriptors, zero for a single one;
- its stored descriptor: the mean of its descriptors in that cell, which lies inside the
  cell.

The fitness is the one of the evaluation, the mean over the evaluations by default. An
empty cell takes any solution (`NewCell`, with `improvement` equal to the fitness). An
occupied cell takes a solution only if its fitness is strictly higher and its spread no
larger than those of the elite (`Improved`, with `improvement` the fitness gain); otherwise
the solution is `Rejected`, with the fitness gain as `improvement` if it is negative and
zero if the spread was the reason. A solution whose descriptors all fall outside the bounds
of a `RejectOutside()` archive, or whose spread is not finite, is `Rejected` with zero
`improvement`.

[`add_batch!`](@ref QualityDiversity.add_batch!) computes the status and `improvement` of
every solution against the archive as it was before the batch, and the archive afterwards
is the one that adding the solutions one at a time gives. [`spreads`](@ref
QualityDiversity.spreads) returns the spread of every cell. Cells, sampling, `elites`,
`length`, `capacity`, the stored phenotypes and the metrics are those of the wrapped
archive, which stay exact because an elite is only replaced by a solution with higher
fitness.

A [`CategoricalArchive`](@ref) cannot be wrapped: its descriptors have no distance.

# Example

```jldoctest
julia> using QualityDiversity, StaticArrays

julia> archive = SpreadArchive(GridArchive((2, 2), ((0, 1), (0, 1))));

julia> samples = QualityDiversity.DescriptorSamples([SVector(0.1, 0.1), SVector(0.2, 0.1), SVector(0.9, 0.9)]);

julia> QualityDiversity.add!(archive, [0.0], Evaluation(1.0, samples)).status
NewCell::AddStatus = 2

julia> only(QualityDiversity.elites(archive))[2].descriptor
2-element SVector{2, Float64} with indices SOneTo(2):
 0.15000000000000002
 0.1
```
"""
struct SpreadArchive{G,F<:Real,N,P,A<:CellArchive{G,F,SVector{N,Float64},P}} <: AbstractArchive{G,F,SVector{N,Float64},P}
    inner::A
    # Spread of the elite of every cell; NaN for an empty cell.
    spreads::Vector{Float64}
    # Workspace: the cell of every descriptor of one solution.
    cells::Vector{Int}

    function SpreadArchive{G,F,N,P,A}(inner::A) where {G,F<:Real,N,P,A<:CellArchive{G,F,SVector{N,Float64},P}}
        length(inner) == 0 ||
            throw(ArgumentError("the archive to wrap must be empty but holds $(length(inner)) solutions"))
        return new{G,F,N,P,A}(inner, fill(NaN, capacity(inner)), Int[])
    end
end

function SpreadArchive(inner::CellArchive{G,F,SVector{N,Float64},P}) where {G,F,N,P}
    return SpreadArchive{G,F,N,P,typeof(inner)}(inner)
end

function SpreadArchive(::CategoricalArchive)
    throw(ArgumentError("a SpreadArchive measures distances between descriptors, which the categories of a CategoricalArchive do not have; wrap a GridArchive or a CVTArchive"))
end

function SpreadArchive(::SlidingBoundariesArchive)
    throw(ArgumentError("a SpreadArchive keeps the spread of the elite of every cell, but the cells of a SlidingBoundariesArchive move when it remaps; wrap a GridArchive or a CVTArchive"))
end

"""
    spreads(archive::SpreadArchive) -> Vector{Float64}

The spread of the elite of every cell, in the order of the cell indices; `NaN` for an
empty cell.
"""
spreads(archive::SpreadArchive) = archive.spreads

Base.length(archive::SpreadArchive) = length(archive.inner)

capacity(archive::SpreadArchive) = capacity(archive.inner)

function Base.empty!(archive::SpreadArchive)
    empty!(archive.inner)
    fill!(archive.spreads, NaN)
    return archive
end

descriptor_bounds(archive::SpreadArchive) = descriptor_bounds(archive.inner)

sample_elites(rng::AbstractRNG, archive::SpreadArchive, n::Integer) = sample_elites(rng, archive.inner, n)

elites(archive::SpreadArchive) = elites(archive.inner)

qd_score(archive::SpreadArchive; offset=0) = qd_score(archive.inner; offset)

max_fitness(archive::SpreadArchive) = max_fitness(archive.inner)

mean_fitness(archive::SpreadArchive) = mean_fitness(archive.inner)

"""
    cell_index(archive::SpreadArchive, d)

The cell of descriptor `d` in the wrapped archive. For the [`DescriptorSamples`](@ref) of a
solution, the cell that its descriptors land in most often, the one visited first on a tie;
`0` if the `outside` policy rejects all of them.
"""
cell_index(archive::SpreadArchive, d::AbstractVector{<:Real}) = cell_index(archive.inner, d)

cell_index(archive::SpreadArchive, samples::DescriptorSamples) = most_visited_cell!(archive, samples)

# The cell that most of `samples` fall in, the first on a tie, or `0` if the outside
# policy rejects every one. Leaves the cell of `samples[k]` in `archive.cells[k]`.
function most_visited_cell!(archive::SpreadArchive, samples::DescriptorSamples)
    isempty(samples) && throw(ArgumentError("the evaluation has no descriptors"))
    n = length(samples)
    cells = archive.cells
    length(cells) < n && resize!(cells, n)
    for k in 1:n
        cells[k] = cell_index(archive.inner, samples[k])
    end
    # A loop, not `argmax` and `count` over a view: `cell_index` must not allocate, and
    # that form allocates on Julia 1.10.
    best = 0
    best_count = 0
    for k in 1:n
        cell = cells[k]
        cell == 0 && continue
        count = 0
        for j in 1:n
            count += cells[j] == cell
        end
        if count > best_count
            best = cell
            best_count = count
        end
    end
    return best
end

# Mean of the descriptors that `most_visited_cell!` placed in `cell`.
function mean_in_cell(archive::SpreadArchive{G,F,N}, samples::DescriptorSamples, cell) where {G,F,N}
    total = zero(SVector{N,Float64})
    count = 0
    for k in 1:length(samples)
        if archive.cells[k] == cell
            total += samples[k]
            count += 1
        end
    end
    return total / count
end

# Mean distance between two of the descriptors; zero for fewer than two.
function spread(samples::DescriptorSamples)
    n = length(samples)
    n < 2 && return 0.0
    total = 0.0
    for a in 1:n-1, b in a+1:n
        total += norm(samples[a] - samples[b])
    end
    return total / (n * (n - 1) / 2)
end

# Status and improvement of a solution with `fitness` and `spread` for cell `i`, without
# changing the archive.
function spread_outcome(archive::SpreadArchive, i, fitness, spread)
    inner = archive.inner
    inner.occupied[i] || return AddResult(NewCell, fitness)
    previous = inner.fitness[i]
    improvement = fitness_gain(fitness, previous)
    fitness > previous && spread <= archive.spreads[i] && return AddResult(Improved, improvement)
    return AddResult(Rejected, min(improvement, zero(improvement)))
end

function place_spread!(archive::SpreadArchive, i, x, descriptor, fitness, phenotype, spread)
    place!(archive.inner, i, x, descriptor, fitness, phenotype)
    archive.spreads[i] = spread
    return nothing
end

"""
    add!(archive::SpreadArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the cell its descriptors land in most often if its fitness is higher and
its spread no larger than the elite's; see [`SpreadArchive`](@ref). `ev` comes from a
[`SampledSpread`](@ref) problem.
"""
add!(archive::SpreadArchive, x, ev::Evaluation) = add_samples!(archive, x, spread_evaluation(ev))

function add_samples!(archive::SpreadArchive{G,F,N,P}, x, ev::Evaluation) where {G,F,N,P}
    phenotype = stored_phenotype(P, ev.phenotype)
    fitness = stored_fitness(F, ev.fitness)
    samples = ev.descriptor
    i = most_visited_cell!(archive, samples)
    s = spread(samples)
    i == 0 && return outside_result(F)
    isfinite(s) || return AddResult(Rejected, zero(F))
    result = spread_outcome(archive, i, fitness, s)
    result.status == Rejected ||
        place_spread!(archive, i, x, mean_in_cell(archive, samples, i), fitness, phenotype, s)
    return result
end

# `ev` if it holds the descriptors of all the evaluations of a solution.
spread_evaluation(ev::Evaluation{<:Real,<:DescriptorSamples}) = ev
function spread_evaluation(ev::Evaluation)
    throw(ArgumentError("a SpreadArchive takes the evaluations of a SampledSpread problem, whose descriptor holds those of all the evaluations of a solution; got a descriptor of type $(typeof(ev.descriptor))"))
end

"""
    add_batch!(archive::SpreadArchive, xs, evs) -> Vector{AddResult}

Insert every solution in `xs` with its evaluation from `evs`. The status and `improvement`
of each solution are computed against the archive as it was before the batch, so they do
not depend on the other solutions in the batch. A solution then enters its cell if it
passes against what the cell holds by now, so the archive afterwards is the one that
adding the solutions one at a time gives.
"""
function add_batch!(archive::SpreadArchive{G,F,N,P}, xs::AbstractVector, evs::AbstractVector) where {G,F,N,P}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    Base.require_one_based_indexing(xs, evs)
    # Throws before the archive changes.
    foreach(ev -> (stored_fitness(F, ev.fitness); stored_phenotype(P, ev.phenotype)), evs)
    n = length(xs)
    results = Vector{AddResult{F}}(undef, n)
    cells = Vector{Int}(undef, n)
    sample_spreads = Vector{Float64}(undef, n)
    descriptors = Vector{SVector{N,Float64}}(undef, n)
    for k in 1:n
        ev = spread_evaluation(evs[k])
        i = most_visited_cell!(archive, ev.descriptor)
        s = spread(ev.descriptor)
        cells[k] = i
        sample_spreads[k] = s
        if i == 0
            results[k] = outside_result(F)
        elseif !isfinite(s)
            results[k] = AddResult(Rejected, zero(F))
        else
            descriptors[k] = mean_in_cell(archive, ev.descriptor, i)
            results[k] = spread_outcome(archive, i, stored_fitness(F, ev.fitness), s)
        end
    end
    # A solution that fails against the elite before the batch also fails against any
    # solution of the batch that replaced it, which has higher fitness and no larger spread.
    for k in 1:n
        results[k].status == Rejected && continue
        i = cells[k]
        fitness = stored_fitness(F, evs[k].fitness)
        if spread_outcome(archive, i, fitness, sample_spreads[k]).status != Rejected
            place_spread!(archive, i, xs[k], descriptors[k], fitness,
                          stored_phenotype(P, evs[k].phenotype), sample_spreads[k])
        end
    end
    return results
end
