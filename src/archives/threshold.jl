"""
    ThresholdArchive(archive::CellArchive; learning_rate=1.0, threshold_min=-Inf, result=<copy>)

Archive for CMA-MAE (Fontaine and Nikolaidis, 2023): a cell archive in which each
cell has a fitness threshold that rises slowly as solutions enter the cell,
instead of jumping to the best fitness found. `archive` decides which cell a
descriptor belongs to, so any [`CellArchive`](@ref) works: a [`GridArchive`](@ref),
a [`CVTArchive`](@ref) or a [`CategoricalArchive`](@ref). It must be empty and
have a floating-point fitness type, and the new archive uses it without copying.

A solution with fitness `f` enters its cell if `f` is strictly greater than the
cell's threshold `t`, and then replaces the cell's elite. The threshold becomes
`(1 - learning_rate) * t + learning_rate * f`. The `improvement` of the
[`AddResult`](@ref) is `f - t`, which is what CMA-ME style emitters rank
candidates by; for a rejected solution it is zero or negative. Every threshold
starts at `threshold_min`, and a solution can only enter an empty cell if its
fitness exceeds that value. A good `threshold_min` is the lowest fitness the
problem can produce, and `learning_rate=0.01` is a usual choice.

The elite of a cell is the solution that last passed the threshold, so its
fitness can be lower than the one it replaced when `learning_rate < 1`: a
solution above the threshold but below the elite replaces the elite. The archive
therefore does not hold the best solution found in each cell. A second, plain
archive does: every solution offered to this archive is offered to it as well,
and [`result_archive`](@ref QualityDiversity.result_archive) returns it. Emitters
read this archive, and a run reports the metrics of, and returns, the result
archive. The `result` keyword takes that archive; by default it is an empty copy
of the wrapped one, and it is `nothing` when `learning_rate=1` and
`threshold_min=-Inf`, where this archive already holds the best solutions and is
its own result. It must be empty and of the same type as the wrapped archive.

`learning_rate` must lie in `[0, 1]`. With `learning_rate=0` the thresholds never
change, and with `learning_rate=1` a threshold becomes the fitness of the
solution that entered. To keep the behavior of an archive with another number of
cells, convert the rate with `1 - (1 - learning_rate)^r`, where `r` is the ratio
of the cell counts.

The defaults, `learning_rate=1` and `threshold_min=-Inf`, give the rules of the
wrapped archive: an empty cell accepts any solution and reports
`improvement == f`, and the threshold of a cell is the fitness of its elite. That
setting needs `learning_rate=1`, so `threshold_min=-Inf` with a smaller rate is an
error.

[`add_batch!`](@ref QualityDiversity.add_batch!) computes the status and
`improvement` of every solution against the archive as it was before the batch,
so a solution does not depend on the others in its batch. Of the solutions that
pass the threshold of a cell, the one with the highest fitness (the first, on a
tie) becomes the elite. The threshold of a cell that `m` solutions of the batch
passed becomes `(1 - learning_rate)^m * t + (1 - (1 - learning_rate)^m) * mean(f)`,
where `mean(f)` is over those solutions, or the highest of their fitness values if
`threshold_min` is `-Inf`. For a single solution this is the rule above, which is
what [`add!`](@ref QualityDiversity.add!) applies.
[`thresholds`](@ref QualityDiversity.thresholds) returns the thresholds, one per
cell.

Cells, sampling, `elites`, `length`, `capacity`, `cell_index` and the stored
phenotypes are those of the wrapped archive, and so are the metrics, which describe the elites of this archive
and not the result archive. `qd_score` and `mean_fitness` take constant time. `max_fitness`
does too, except directly after the best elite was replaced by a worse solution,
when the next call makes a pass over the elites.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = ThresholdArchive(GridArchive((10, 10), ((0, 1), (0, 1)));
                                  learning_rate=0.5, threshold_min=0);

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
mutable struct ThresholdArchive{G,F<:AbstractFloat,D,P,A<:CellArchive{G,F,D,P},R<:Union{Nothing,A}} <: AbstractArchive{G,F,D,P}
    const inner::A
    # The archive of the best solutions, or `nothing` if `inner` holds them.
    const result::R
    const thresholds::Vector{F}
    const learning_rate::F
    const threshold_min::F
    # Set when the elite with the highest fitness was replaced by a worse one, so
    # that `inner.fitness_max` is only an upper bound.
    max_stale::Bool

    function ThresholdArchive{G,F,D,P,A,R}(inner::A, result::R, learning_rate, threshold_min) where {G,F<:AbstractFloat,D,P,A<:CellArchive{G,F,D,P},R<:Union{Nothing,A}}
        length(inner) == 0 ||
            throw(ArgumentError("the archive to wrap must be empty but holds $(length(inner)) solutions"))
        0 <= learning_rate <= 1 ||
            throw(ArgumentError("learning_rate must be in [0, 1], got $learning_rate"))
        (isnan(threshold_min) || threshold_min == Inf) &&
            throw(ArgumentError("threshold_min must be a number below Inf, got $threshold_min"))
        threshold_min == -Inf && learning_rate != 1 &&
            throw(ArgumentError("threshold_min can only be -Inf if learning_rate is 1, got $learning_rate; give a finite threshold_min, such as the lowest fitness of the problem"))
        if result !== nothing
            result !== inner || throw(ArgumentError("result must be a different archive from the one it belongs to"))
            length(result) == 0 ||
                throw(ArgumentError("result must be empty but holds $(length(result)) solutions"))
        end
        return new{G,F,D,P,A,R}(inner, result, fill(convert(F, threshold_min), capacity(inner)),
                              convert(F, learning_rate), convert(F, threshold_min), false)
    end
end

function ThresholdArchive(inner::CellArchive{G,F,D,P}; learning_rate::Real=1.0,
                          threshold_min::Real=-Inf,
                          result::Union{Nothing,CellArchive}=default_result(inner, learning_rate, threshold_min)) where {G,F<:AbstractFloat,D,P}
    check_result_type(inner, result)
    return ThresholdArchive{G,F,D,P,typeof(inner),typeof(result)}(inner, result, learning_rate, threshold_min)
end

# The result archive is none, or one of the same type as the archive it belongs to.
check_result_type(inner::CellArchive, ::Nothing) = nothing
check_result_type(inner::A, result::A) where {A<:CellArchive} = nothing
function check_result_type(inner::CellArchive, result)
    throw(ArgumentError("result must be a $(nameof(typeof(inner))) of the same type as the archive it belongs to, got a $(typeof(result))"))
end

function ThresholdArchive(::SlidingBoundariesArchive{G,F}; kwargs...) where {G,F<:AbstractFloat}
    throw(ArgumentError("a ThresholdArchive keeps a threshold for every cell, but the cells of a SlidingBoundariesArchive move when it remaps; wrap a GridArchive or a CVTArchive"))
end

# With the thresholds at the fitness of the elites the archive already keeps the
# best solutions; otherwise an empty copy collects them.
default_result(inner, learning_rate, threshold_min) =
    learning_rate == 1 && threshold_min == -Inf ? nothing : deepcopy(inner)

"""
    thresholds(archive::ThresholdArchive) -> Vector

The fitness threshold of each cell, in the order of the cell indices. A cell
that has never accepted a solution holds `threshold_min`.
"""
thresholds(archive::ThresholdArchive) = archive.thresholds

Base.length(archive::ThresholdArchive) = length(archive.inner)

capacity(archive::ThresholdArchive) = capacity(archive.inner)

function Base.empty!(archive::ThresholdArchive)
    empty!(archive.inner)
    archive.result === nothing || empty!(archive.result)
    fill!(archive.thresholds, archive.threshold_min)
    archive.max_stale = false
    return archive
end

result_archive(archive::ThresholdArchive) = result_or_self(archive, archive.result)

result_or_self(archive, ::Nothing) = archive
result_or_self(archive, result) = result

# Every solution is offered to the result archive, whether or not this archive
# takes it.
offer!(::Nothing, x, ev) = nothing
offer!(result, x, ev) = (add!(result, x, ev); nothing)

offer_batch!(::Nothing, xs, evs) = nothing
offer_batch!(result, xs, evs) = (add_batch!(result, xs, evs); nothing)

cell_index(archive::ThresholdArchive, d) = cell_index(archive.inner, d)

sample_elites(rng::AbstractRNG, archive::ThresholdArchive, n::Integer) =
    sample_elites(rng, archive.inner, n)

elites(archive::ThresholdArchive) = elites(archive.inner)

# Status and improvement of a solution with `fitness` in cell `i`, before it is
# inserted. The improvement of the first solution in a cell is its fitness when
# the thresholds start at `-Inf`.
function classify(archive::ThresholdArchive{G,F}, i, fitness) where {G,F}
    occupied = archive.inner.occupied[i]
    threshold = archive.thresholds[i]
    improvement = threshold == -Inf ? fitness : fitness - threshold
    fitness > threshold || return AddResult(Rejected, improvement)
    return AddResult(occupied ? Improved : NewCell, improvement)
end

# Stores `x` as the elite of cell `i` with the new threshold, keeping the
# running totals of the wrapped archive up to date.
function place_threshold!(archive::ThresholdArchive, i, x, descriptor, fitness, phenotype, threshold)
    inner = archive.inner
    archive.thresholds[i] = threshold
    if !inner.occupied[i]
        inner.occupied[i] = true
        push!(inner.occupied_idx, i)
        _store!(inner, i, x, fitness, descriptor, phenotype)
        _add_to_sum!(inner, Float64(fitness))
        if length(inner.occupied_idx) == 1 || fitness >= inner.fitness_max
            inner.fitness_max = fitness
            archive.max_stale = false
        end
        return nothing
    end
    previous = inner.fitness[i]
    _store!(inner, i, x, fitness, descriptor, phenotype)
    # Two compensated steps, as in `place!` of a `CellArchive`.
    _add_to_sum!(inner, Float64(fitness))
    _add_to_sum!(inner, -Float64(previous))
    if fitness >= inner.fitness_max
        inner.fitness_max = fitness
        archive.max_stale = false
    elseif previous == inner.fitness_max
        archive.max_stale = true
    end
    return nothing
end

"""
    add!(archive::ThresholdArchive, x, ev::Evaluation) -> AddResult

Insert `x` into the cell of `ev.descriptor` if its fitness exceeds the cell's
threshold, and raise the threshold; see [`ThresholdArchive`](@ref). The status
is `NewCell` for an empty cell, `Improved` for an occupied one and `Rejected`
otherwise. A descriptor that the wrapped archive discards gives `Rejected` with
the lowest `improvement`, `-Inf` for a floating-point fitness type.
"""
function add!(archive::ThresholdArchive{G,F,D,P}, x, ev::Evaluation{<:Real}) where {G,F,D,P}
    phenotype = stored_phenotype(P, ev.phenotype)
    fitness = stored_fitness(F, ev.fitness)    # throws before either archive changes
    i = cell_index(archive.inner, ev.descriptor)
    offer!(archive.result, x, ev)
    i == 0 && return outside_result(F)
    result = classify(archive, i, fitness)
    result.status == Rejected && return result
    threshold = next_threshold(archive.learning_rate, archive.thresholds[i], fitness, 1)
    place_threshold!(archive, i, x, ev.descriptor, fitness, phenotype, threshold)
    return result
end

# Threshold of a cell after `count` solutions with fitness summing to `total` passed
# a threshold `t`. One solution gives the update rule itself, so a batch of one
# agrees with `add!` to the last bit.
function next_threshold(rate, t, total, count)
    count == 1 && return rate == 1 ? total : (1 - rate) * t + rate * total
    kept = (1 - rate)^count
    return kept * t + (1 - kept) * (total / count)
end

"""
    add_batch!(archive::ThresholdArchive, xs, evs) -> Vector{AddResult}

Insert every solution in `xs` with its evaluation from `evs`. Each result is
computed against the archive as it was before the batch, and the solutions that
pass the threshold of a cell act on it together; see [`ThresholdArchive`](@ref).
"""
function add_batch!(archive::ThresholdArchive{G,F,D,P}, xs::AbstractVector, evs::AbstractVector) where {G,F,D,P}
    length(xs) == length(evs) ||
        throw(ArgumentError("got $(length(xs)) solutions but $(length(evs)) evaluations"))
    n = length(xs)
    # Every fitness and phenotype is converted once, before either archive changes: a
    # conversion that throws must leave both untouched. The fitness is read up to four
    # times below, so it is kept too.
    fitness = Vector{F}(undef, n)
    phenotypes = batch_phenotypes(P, n)
    for k in 1:n
        fitness[k] = stored_fitness(F, evs[k].fitness)
        store_phenotype!(phenotypes, k, stored_phenotype(P, evs[k].phenotype))
    end
    offer_batch!(archive.result, xs, evs)
    results = Vector{AddResult{F}}(undef, n)
    cells = Vector{Int}(undef, n)
    passing = Int[]
    for k in 1:n
        i = cell_index(archive.inner, evs[k].descriptor)
        cells[k] = i
        results[k] = i == 0 ? outside_result(F) : classify(archive, i, fitness[k])
        results[k].status == Rejected || push!(passing, k)
    end
    # Group the passing solutions by cell, keeping their order within a cell.
    sort!(passing; by=k -> cells[k], alg=MergeSort)
    best = Int[]
    first_in_batch = Int[]
    new_threshold = F[]
    run_start = 1
    while run_start <= length(passing)
        i = cells[passing[run_start]]
        run_end = run_start
        while run_end < length(passing) && cells[passing[run_end+1]] == i
            run_end += 1
        end
        winner = passing[run_start]
        total = zero(F)
        for p in run_start:run_end
            k = passing[p]
            total += fitness[k]
            fitness[k] > fitness[winner] && (winner = k)
        end
        count = run_end - run_start + 1
        push!(best, winner)
        push!(first_in_batch, passing[run_start])
        push!(new_threshold, archive.threshold_min == -Inf ? fitness[winner] :
                             next_threshold(archive.learning_rate, archive.thresholds[i], total, count))
        run_start = run_end + 1
    end
    # Cells are filled in the order of their first solution in the batch, which is
    # the order the elites are sampled in.
    for r in sortperm(first_in_batch)
        k = best[r]
        place_threshold!(archive, cells[k], xs[k], evs[k].descriptor, fitness[k], batch_phenotype(phenotypes, k),
                         new_threshold[r])
    end
    return results
end

qd_score(archive::ThresholdArchive; offset=0) = qd_score(archive.inner; offset)

mean_fitness(archive::ThresholdArchive) = mean_fitness(archive.inner)

function max_fitness(archive::ThresholdArchive)
    inner = archive.inner
    isempty_archive(inner, "max_fitness")
    if archive.max_stale
        inner.fitness_max = maximum(inner.fitness[i] for i in inner.occupied_idx)
        archive.max_stale = false
    end
    return inner.fitness_max
end

descriptor_bounds(archive::ThresholdArchive) = descriptor_bounds(archive.inner)
