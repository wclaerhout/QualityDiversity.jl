"""
    UnstructuredArchive{G,F}(ndims; capacity, novelty_threshold, phenotype=Nothing)
    UnstructuredArchive{G}(ndims; capacity, novelty_threshold, phenotype=Nothing)
    UnstructuredArchive(ndims; capacity, novelty_threshold, phenotype=Nothing)
    UnstructuredArchive(prob::AbstractQDProblem; capacity, novelty_threshold,
                        genotype=default_genotype(prob), fitness=Float64, phenotype=Nothing)

Archive without a fixed partition of descriptor space. It stores up to
`capacity` solutions, each at its own descriptor, and a solution enters if it
is far from the ones already stored (novelty) or better than its nearest
neighbour (local competition).

`ndims` is the number of descriptor dimensions; the last form takes it from
`length(descriptor_bounds(prob))`. `G` is the genotype type, `Vector{Float64}`
by default, and `F<:Real` the fitness type, `Float64` by default. Descriptors
are `SVector{ndims,Float64}`, and distances are Euclidean in descriptor units,
so descriptors on very different scales should be normalized by the problem.
`novelty_threshold` is the distance beyond which a descriptor counts as new, as in
pyribs. `phenotype` is the type of the phenotypes the archive stores, `Nothing` for
none; see [`Evaluation`](@ref).

[`add!`](@ref QualityDiversity.add!) finds the stored solution with the nearest
descriptor, at distance `r`:

- Nothing is stored, or `r > novelty_threshold`: the solution is novel. It fills a free
  slot (`NewCell`, with `improvement` equal to the fitness). If the archive is
  full, it replaces the stored solution with the lowest fitness, but only if
  its fitness is strictly higher (`NewCell`); otherwise it is `Rejected`.
- `r <= novelty_threshold`: the solution competes with its nearest neighbour and
  replaces it if its fitness is strictly higher (`Improved`, with `improvement`
  the difference), and is `Rejected` otherwise. The new descriptor replaces the
  old one, so stored descriptors can end up closer together than
  `novelty_threshold`.

[`cell_index`](@ref QualityDiversity.cell_index) is the index of the stored
solution with the nearest descriptor, or `0` while the archive is empty.
[`capacity`](@ref QualityDiversity.capacity) is the number of slots, so
`coverage` is the fraction of them in use. Descriptors must be finite; a `NaN`
or infinite component throws an `ArgumentError`.

With fewer than `max(64, 4^ndims)` stored solutions the nearest neighbour is found by
a pass over the stored descriptors. From that many on, `add!` keeps a k-d tree of the
stored descriptors, which it rebuilds when the solutions stored since the last build
number more than `64 + 8 sqrt(n)` for `n` stored solutions, and a search checks those
solutions one by one. Both find the same neighbour, and neither allocates. The solution
of lowest fitness, which a full archive replaces, is kept on top of a heap. The archive
has no [`add_batch!`](@ref QualityDiversity.add_batch!) of its own: the default inserts
the solutions of a batch one at a time with `add!`, so the result of a solution depends
on the solutions before it in the batch, unlike in a [`GridArchive`](@ref). A solution
is stored without copying, so the caller must not modify it afterwards. The archive
keeps running totals, as [`GridArchive`](@ref) does, so `qd_score`, `max_fitness` and
`mean_fitness` take constant time. The maximum is exact because a stored solution is
only ever replaced by a better one.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = UnstructuredArchive(2; capacity=100, novelty_threshold=0.1);

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
mutable struct UnstructuredArchive{G,F<:Real,N,P} <: AbstractArchive{G,F,SVector{N,Float64},P}
    const threshold::Float64
    # One entry per slot; the first `n` are in use.
    const fitness::Vector{F}
    const descriptors::Vector{SVector{N,Float64}}
    const genotypes::Vector{G}
    const phenotypes::Vector{P}
    n::Int
    # Running totals over the stored solutions, as in `CellArchive`.
    fitness_sum::Float64
    fitness_sum_error::Float64
    fitness_max::F
    # The slots in use as a binary heap with the lowest fitness on top, the lower slot on
    # a tie, and the position of every slot in it.
    const heap::Vector{Int}
    const heap_position::Vector{Int}
    # k-d tree over a copy of the descriptors of the first `tree_size` slots as they were
    # when it was built; `0` for no tree. A slot stored since is pending: a search skips
    # it in the tree and checks its descriptor directly.
    const tree::CentroidTree{N}
    tree_size::Int
    const pending::Vector{Bool}
    const pending_slots::Vector{Int}
    num_pending::Int

    function UnstructuredArchive{G,F,N,P}(capacity, threshold) where {G,F<:Real,N,P}
        N >= 1 || throw(ArgumentError("ndims must be at least 1, got $N"))
        capacity >= 1 || throw(ArgumentError("capacity must be at least 1, got $capacity"))
        (isfinite(threshold) && threshold >= 0) ||
            throw(ArgumentError("novelty_threshold must be finite and at least 0, got $threshold"))
        return new{G,F,N,P}(Float64(threshold),
                            Vector{F}(undef, capacity),
                            Vector{SVector{N,Float64}}(undef, capacity),
                            Vector{G}(undef, capacity),
                            Vector{P}(undef, capacity),
                            0, 0.0, 0.0, zero(F),
                            Vector{Int}(undef, capacity), Vector{Int}(undef, capacity),
                            CentroidTree{N}(Vector{SVector{N,Float64}}(undef, capacity),
                                            Vector{Int}(undef, capacity), Vector{Int}(undef, capacity)),
                            0, fill(false, capacity), Vector{Int}(undef, capacity), 0)
    end
end

function UnstructuredArchive{G,F}(ndims::Integer; capacity::Integer, novelty_threshold::Real,
                                  phenotype::Type=Nothing) where {G,F<:Real}
    return UnstructuredArchive{G,F,Int(ndims),phenotype}(Int(capacity), novelty_threshold)
end

function UnstructuredArchive{G}(ndims::Integer; kwargs...) where {G}
    return UnstructuredArchive{G,Float64}(ndims; kwargs...)
end

function UnstructuredArchive(ndims::Integer; kwargs...)
    return UnstructuredArchive{Vector{Float64}}(ndims; kwargs...)
end

function UnstructuredArchive(prob::AbstractQDProblem; capacity::Integer, novelty_threshold::Real,
                             genotype::Type=default_genotype(prob), fitness::Type{<:Real}=Float64,
                             phenotype::Type=Nothing)
    return UnstructuredArchive{genotype,fitness}(length(problem_bounds(prob, nothing));
                                                 capacity, novelty_threshold, phenotype)
end

Base.length(archive::UnstructuredArchive) = archive.n

capacity(archive::UnstructuredArchive) = length(archive.fitness)

function Base.empty!(archive::UnstructuredArchive{G,F}) where {G,F}
    archive.n = 0
    archive.fitness_sum = 0.0
    archive.fitness_sum_error = 0.0
    archive.fitness_max = zero(F)
    clear_pending!(archive)
    archive.tree_size = 0
    return archive
end

# `d` as an `SVector`, after checking its dimension and that it is finite.
function checked_descriptor(::UnstructuredArchive{G,F,N}, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    point = SVector{N,Float64}(d)
    for i in 1:N
        isfinite(point[i]) ||
            throw(ArgumentError("descriptor component $i is $(point[i]), which is not finite"))
    end
    return point
end

# Slot of the stored descriptor nearest to `point` (the lowest slot on a tie) and
# the squared distance to it; `(0, Inf)` while the archive is empty.
function nearest_stored(archive::UnstructuredArchive, point)
    if archive.tree_size == 0
        best, best_distance = 0, Inf
        for i in 1:archive.n
            best, best_distance = nearer(archive, point, i, best, best_distance)
        end
        return best, best_distance
    end
    best, best_distance = search(archive.tree, point, 1, archive.tree_size, 0, Inf, Flagged(archive.pending))
    for p in 1:archive.num_pending
        best, best_distance = nearer(archive, point, archive.pending_slots[p], best, best_distance)
    end
    return best, best_distance
end

# Slot `i` and its squared distance to `point` if it is nearer than `best`, or nearer by
# the same distance and lower, and `best` otherwise.
function nearer(archive::UnstructuredArchive, point, i, best, best_distance)
    distance = sum(abs2, point - archive.descriptors[i])
    distance < best_distance || (distance == best_distance && i < best) || return best, best_distance
    return i, distance
end

# Skips the indices whose flag is set.
struct Flagged
    flags::Vector{Bool}
end

(skip::Flagged)(i) = skip.flags[i]

# The number of stored solutions from which a tree finds the nearest one faster than a
# pass over all of them.
tree_threshold(::UnstructuredArchive{G,F,N}) where {G,F,N} = N >= 15 ? typemax(Int) : max(64, 4^N)

# Builds the tree anew when it pays: the archive has reached `tree_threshold` and the
# solutions stored since the last build are too many to check one by one.
function refresh_tree!(archive::UnstructuredArchive)
    n = archive.n
    n >= tree_threshold(archive) || return nothing
    archive.tree_size > 0 && archive.num_pending <= 64 + 8 * isqrt(n) && return nothing
    tree = archive.tree
    for i in 1:n
        tree.points[i] = archive.descriptors[i]
        tree.order[i] = i
    end
    build!(tree.order, tree.axis, tree.points, 1, n)
    archive.tree_size = n
    clear_pending!(archive)
    return nothing
end

function clear_pending!(archive::UnstructuredArchive)
    for p in 1:archive.num_pending
        archive.pending[archive.pending_slots[p]] = false
    end
    archive.num_pending = 0
    return nothing
end

# Slot of the lowest fitness (the lowest slot on a tie). The archive is not empty.
weakest_stored(archive::UnstructuredArchive) = archive.heap[1]

# Slot `a` belongs nearer the top of the heap than slot `b`.
weaker(archive::UnstructuredArchive, a, b) =
    archive.fitness[a] < archive.fitness[b] || (archive.fitness[a] == archive.fitness[b] && a < b)

function heap_place!(archive::UnstructuredArchive, position, slot)
    archive.heap[position] = slot
    archive.heap_position[slot] = position
    return nothing
end

# Adds slot `archive.n`, just filled, to the heap.
function heap_push!(archive::UnstructuredArchive)
    slot = archive.n
    position = slot
    while position > 1
        parent = position >> 1
        weaker(archive, slot, archive.heap[parent]) || break
        heap_place!(archive, position, archive.heap[parent])
        position = parent
    end
    heap_place!(archive, position, slot)
    return nothing
end

# Moves `slot`, whose fitness has risen, down the heap.
function heap_raise!(archive::UnstructuredArchive, slot)
    position = archive.heap_position[slot]
    while true
        child = 2 * position
        child > archive.n && break
        if child < archive.n && weaker(archive, archive.heap[child+1], archive.heap[child])
            child += 1
        end
        weaker(archive, archive.heap[child], slot) || break
        heap_place!(archive, position, archive.heap[child])
        position = child
    end
    heap_place!(archive, position, slot)
    return nothing
end

"""
    cell_index(archive::UnstructuredArchive, d)

Index of the stored solution whose descriptor is nearest to `d`; the lowest
index if two are equally near. `0`, which is not a slot, while the archive is
empty. Throws an `ArgumentError` if `d` has the wrong number of dimensions or a
component that is not finite.
"""
function cell_index(archive::UnstructuredArchive, d::AbstractVector{<:Real})
    point = checked_descriptor(archive, d)
    return first(nearest_stored(archive, point))
end

"""
    add!(archive::UnstructuredArchive, x, ev::Evaluation) -> AddResult

Insert `x` if it is novel or beats its nearest neighbour; see
[`UnstructuredArchive`](@ref) for the rules.
"""
function add!(archive::UnstructuredArchive{G,F,N,P}, x,
              ev::Evaluation{<:Real,<:AbstractVector{<:Real}}) where {G,F,N,P}
    point = checked_descriptor(archive, ev.descriptor)
    fitness = stored_fitness(F, ev.fitness)
    phenotype = stored_phenotype(P, ev.phenotype)
    refresh_tree!(archive)
    nearest, distance = nearest_stored(archive, point)
    if nearest == 0 || distance > archive.threshold^2
        if archive.n < capacity(archive)
            archive.n += 1
            _store!(archive, archive.n, x, fitness, point, phenotype)
            heap_push!(archive)
            _add_to_sum!(archive, Float64(fitness))
            archive.fitness_max = archive.n == 1 ? fitness : max(archive.fitness_max, fitness)
            return AddResult(NewCell, fitness)
        end
        weakest = weakest_stored(archive)
        previous = archive.fitness[weakest]
        fitness > previous || return AddResult(Rejected, fitness_gain(fitness, previous))
        _store!(archive, weakest, x, fitness, point, phenotype)
        heap_raise!(archive, weakest)
        # Two compensated steps, as in `place!` of a `CellArchive`.
        _add_to_sum!(archive, Float64(fitness))
        _add_to_sum!(archive, -Float64(previous))
        archive.fitness_max = max(archive.fitness_max, fitness)
        return AddResult(NewCell, fitness)
    end
    previous = archive.fitness[nearest]
    improvement = fitness_gain(fitness, previous)
    fitness > previous || return AddResult(Rejected, improvement)
    _store!(archive, nearest, x, fitness, point, phenotype)
    heap_raise!(archive, nearest)
    _add_to_sum!(archive, Float64(fitness))
    _add_to_sum!(archive, -Float64(previous))
    archive.fitness_max = max(archive.fitness_max, fitness)
    return AddResult(Improved, improvement)
end

function _store!(archive::UnstructuredArchive, i, x, fitness, descriptor, phenotype)
    archive.genotypes[i] = x
    archive.fitness[i] = fitness
    archive.descriptors[i] = descriptor
    store_phenotype!(archive.phenotypes, i, phenotype)
    if archive.tree_size > 0 && !archive.pending[i]
        archive.pending[i] = true
        archive.num_pending += 1
        archive.pending_slots[archive.num_pending] = i
    end
    return nothing
end

qd_score(archive::UnstructuredArchive; offset=0) = _fitness_sum(archive) + length(archive) * offset

function max_fitness(archive::UnstructuredArchive)
    isempty_archive(archive, "max_fitness")
    return archive.fitness_max
end

function mean_fitness(archive::UnstructuredArchive)
    isempty_archive(archive, "mean_fitness")
    return _fitness_sum(archive) / length(archive)
end

"""
    sample_elites(rng, archive::UnstructuredArchive, n) -> Vector{G}

`n` genotypes drawn with replacement, uniformly over the stored solutions.
Throws an `ArgumentError` if the archive is empty.
"""
function sample_elites(rng::AbstractRNG, archive::UnstructuredArchive{G}, n::Integer) where {G}
    archive.n == 0 && throw(ArgumentError("cannot sample elites from an empty archive"))
    parents = Vector{G}(undef, n)
    for k in 1:n
        parents[k] = archive.genotypes[rand(rng, 1:archive.n)]
    end
    return parents
end

function elites(archive::UnstructuredArchive)
    return ((archive.genotypes[i], Evaluation(archive.fitness[i], archive.descriptors[i], archive.phenotypes[i]))
            for i in 1:archive.n)
end
