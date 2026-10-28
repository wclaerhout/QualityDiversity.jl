"""
    SlidingBoundariesArchive{G,F}(dims, bounds; remap_frequency=100, buffer_capacity=1000,
                                  outside=ClampOutside(), phenotype=Nothing)
    SlidingBoundariesArchive{G}(dims, bounds; kwargs...)
    SlidingBoundariesArchive(dims, bounds; kwargs...)
    SlidingBoundariesArchive(prob::AbstractQDProblem; dims, bounds=nothing,
                             genotype=default_genotype(prob), fitness=Float64, remap_frequency=100,
                             buffer_capacity=1000, outside=ClampOutside(), phenotype=Nothing)

A grid archive whose cell boundaries follow the distribution of the descriptors found
(Fontaine et al., 2019, "Mapping Hearthstone Deck Spaces through MAP-Elites with
Sliding Boundaries"). Where descriptors bunch up, the cells are narrow; where they are
rare, the cells are wide.

`dims`, `bounds`, `G`, `F` and `phenotype` are those of [`GridArchive`](@ref), and the
archive starts as a regular grid over `bounds`. It keeps the genotype, fitness,
descriptor and phenotype of the last `buffer_capacity` solutions offered to it,
accepted or not. Every `remap_frequency` solutions it remaps: along each dimension
with `n` cells, boundary `j` (for `j` in `1:n`) moves to the `⌊(j-1)·m/n⌋+1`-th smallest
of the `m` buffered descriptor values and the upper boundary to the largest, so each
cell holds about as many buffered descriptors as the next. It then empties the cells
and adds back, one at a time, the old elites and the buffered solutions from oldest to
newest, so the best solution found is never lost. A cell can have zero width when
several buffered descriptors share a value; it stays empty.

A descriptor in cell `j` along a dimension lies between boundaries `j` and `j+1`; a
value on a boundary belongs to the cell above it, and a value on the upper boundary to
the last cell. A descriptor outside the boundaries is handled by `outside`, as for a
[`GridArchive`](@ref), but the default is `ClampOutside()`: after a remap the outer
boundaries are the extremes of the buffer, so new solutions often land outside them.
Old elites are always clamped during a remap. Clamping only chooses the cell: the
archive stores the descriptor the problem returned, and a remap places every solution
again from it. An elite whose descriptor has left the buffer can lie beyond the new
boundaries; it then sits in an edge cell, and its descriptor, as
[`elites`](@ref QualityDiversity.elites) returns it, lies outside that cell and outside
`descriptor_bounds(archive)`. Every solution that has a cell enters the
buffer, so with `ClampOutside()` the boundaries grow toward descriptors outside them;
with `RejectOutside()` a solution outside the boundaries is discarded for good and never
enters the buffer. A descriptor with a non-finite component throws an `ArgumentError`
whatever the policy. The solution that triggers a remap gets its result against the
cells before the remap and is then added back from the buffer, so it is not in the
archive afterwards if an old elite with a higher fitness lands in its new cell.

`add_batch!` computes every result against the archive as it was before the batch, as
for the other [`CellArchive`](@ref)s. If the number of offered solutions passes a
multiple of `remap_frequency` during the batch, the archive remaps once, after it.
[`ThresholdArchive`](@ref) and [`SpreadArchive`](@ref) cannot wrap this archive, since
their thresholds and spreads belong to cells that move.

The current boundaries are returned by
[`boundaries(archive)`](@ref QualityDiversity.boundaries) and their extremes by
[`descriptor_bounds(archive)`](@ref QualityDiversity.descriptor_bounds). `empty!`
also clears the buffer and restores the regular grid over `bounds`.

# Example

```jldoctest
julia> using QualityDiversity

julia> archive = SlidingBoundariesArchive((10, 10), ((0, 1), (0, 1)); remap_frequency=50);

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
mutable struct SlidingBoundariesArchive{G,F<:Real,N,O<:OutsidePolicy,P} <: CellArchive{G,F,SVector{N,Float64},P}
    const dims::NTuple{N,Int}
    # The bounds of the regular grid the archive starts with.
    const lower::SVector{N,Float64}
    const upper::SVector{N,Float64}
    # `dims[i] + 1` increasing boundaries along dimension `i`.
    const boundaries::NTuple{N,Vector{Float64}}
    const outside::O
    # The cell storage of a `CellArchive`. A remap swaps it with the spare storage
    # below, so these fields are not `const`.
    occupied::BitVector
    fitness::Vector{F}
    descriptors::Vector{SVector{N,Float64}}
    genotypes::Vector{G}
    phenotypes::Vector{P}
    occupied_idx::Vector{Int}
    fitness_sum::Float64
    fitness_sum_error::Float64
    fitness_max::F
    # Spare cell storage; during a remap it holds the old elites.
    spare_occupied::BitVector
    spare_fitness::Vector{F}
    spare_descriptors::Vector{SVector{N,Float64}}
    spare_genotypes::Vector{G}
    spare_phenotypes::Vector{P}
    spare_occupied_idx::Vector{Int}
    const remap_frequency::Int
    # A ring of the last solutions offered: `buffer_size` of them, the oldest in slot
    # `mod1(buffer_next - buffer_size, buffer_capacity)`. While it is not full, the
    # solutions are in slots `1:buffer_size`.
    const buffer_genotypes::Vector{G}
    const buffer_fitness::Vector{F}
    const buffer_descriptors::Vector{SVector{N,Float64}}
    const buffer_phenotypes::Vector{P}
    buffer_next::Int
    buffer_size::Int
    # Number of solutions offered since the archive was built or emptied.
    offered::Int
    # Workspace for sorting one dimension of the buffered descriptors.
    const scratch::Vector{Float64}

    function SlidingBoundariesArchive{G,F,N,O,P}(dims, lower, upper, outside, remap_frequency,
                                                 buffer_capacity) where {G,F<:Real,N,O<:OutsidePolicy,P}
        for i in 1:N
            dims[i] >= 1 ||
                throw(ArgumentError("dims[$i] is $(dims[i]) but every dimension needs at least one cell"))
            check_bound(i, lower[i], upper[i])
        end
        remap_frequency >= 1 ||
            throw(ArgumentError("remap_frequency must be at least 1, got $remap_frequency"))
        buffer_capacity >= 1 ||
            throw(ArgumentError("buffer_capacity must be at least 1, got $buffer_capacity"))
        ncells = prod(dims)
        cells(T) = Vector{T}(undef, ncells)
        slots(T) = Vector{T}(undef, buffer_capacity)
        boundaries = ntuple(i -> regular_boundaries(lower[i], upper[i], dims[i]), Val(N))
        return new{G,F,N,O,P}(dims, lower, upper, boundaries, outside,
                              falses(ncells), cells(F), cells(SVector{N,Float64}), cells(G), cells(P), Int[],
                              0.0, 0.0, zero(F),
                              falses(ncells), cells(F), cells(SVector{N,Float64}), cells(G), cells(P), Int[],
                              remap_frequency,
                              slots(G), slots(F), slots(SVector{N,Float64}), slots(P), 1, 0, 0,
                              slots(Float64))
    end
end

regular_boundaries(lo, hi, n) = collect(range(lo, hi; length=n + 1))

function SlidingBoundariesArchive{G,F}(dims::Tuple, bounds::Tuple; remap_frequency::Integer=100,
                                       buffer_capacity::Integer=1000,
                                       outside::OutsidePolicy=ClampOutside(),
                                       phenotype::Type=Nothing) where {G,F<:Real}
    lower, upper = grid_box(dims, bounds)
    return SlidingBoundariesArchive{G,F,length(dims),typeof(outside),phenotype}(
        map(Int, dims), lower, upper, outside, Int(remap_frequency), Int(buffer_capacity))
end

function SlidingBoundariesArchive{G}(dims::Tuple, bounds::Tuple; kwargs...) where {G}
    return SlidingBoundariesArchive{G,Float64}(dims, bounds; kwargs...)
end

function SlidingBoundariesArchive(dims::Tuple, bounds::Tuple; kwargs...)
    return SlidingBoundariesArchive{Vector{Float64}}(dims, bounds; kwargs...)
end

function SlidingBoundariesArchive(prob::AbstractQDProblem; dims::Tuple,
                                  bounds::Union{Tuple,Nothing}=nothing,
                                  genotype::Type=default_genotype(prob), fitness::Type{<:Real}=Float64,
                                  remap_frequency::Integer=100, buffer_capacity::Integer=1000,
                                  outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing)
    return SlidingBoundariesArchive{genotype,fitness}(dims, problem_bounds(prob, bounds); remap_frequency,
                                                      buffer_capacity, outside, phenotype)
end

"""
    boundaries(archive::SlidingBoundariesArchive) -> NTuple{N,Vector{Float64}}

A copy of the current cell boundaries: for each descriptor dimension with `n` cells, `n + 1`
nondecreasing values, cell `j` lying between entries `j` and `j + 1`.
"""
boundaries(archive::SlidingBoundariesArchive) = map(copy, archive.boundaries)

descriptor_bounds(archive::SlidingBoundariesArchive) = map(b -> (first(b), last(b)), archive.boundaries)

"""
    cell_index(archive::SlidingBoundariesArchive, d)

Linear index of the cell that contains descriptor `d` under the current boundaries,
numbered as in a [`GridArchive`](@ref). Throws an `ArgumentError` if `d` has the wrong
number of dimensions or a non-finite component. A component outside the boundaries is
handled by the archive's `outside` policy; `RejectOutside()` returns `0`, which is not
a cell.
"""
function cell_index(archive::SlidingBoundariesArchive, d::AbstractVector{<:Real})
    return sliding_cell(archive, archive.outside, d)
end

function sliding_cell(archive::SlidingBoundariesArchive{G,F,N}, outside, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    # Every component is checked before the policy can discard the descriptor: a
    # non-finite component must never reach the buffer, where a remap would sort it.
    for i in 1:N
        isfinite(d[i]) ||
            throw(ArgumentError("descriptor component $i is $(d[i]), but a SlidingBoundariesArchive needs finite descriptors"))
    end
    index = 0
    stride = 1
    for i in 1:N
        x = d[i]
        b = archive.boundaries[i]
        n = archive.dims[i]
        lo = b[1]
        hi = b[n+1]
        if !(lo <= x <= hi)
            x = resolve_outside(outside, i, x, lo, hi)
            isnan(x) && return 0
        end
        # the last cell whose lower boundary is at most `x`; the upper boundary `b[n+1]`
        # belongs to cell `n`
        index += (min(searchsortedlast(b, x), n) - 1) * stride
        stride *= n
    end
    return index + 1
end

function add!(archive::SlidingBoundariesArchive{G,F,N,O,P}, x, ev::Evaluation{<:Real}) where {G,F,N,O,P}
    if (archive.offered + 1) % archive.remap_frequency != 0
        result = add_to_cells!(archive, x, ev)
        discarded(archive, result, ev) || remember!(archive, x, ev)
        return result
    end
    # The solution that triggers a remap is not placed in the old cells, where it could
    # evict an elite that is no longer in the buffer; the remap adds it from the buffer.
    i = cell_index(archive, ev.descriptor)
    fitness = stored_fitness(F, ev.fitness)    # throws before the archive changes
    stored_phenotype(P, ev.phenotype)
    i == 0 && return outside_result(F)
    result = classify(archive, i, fitness)
    remember!(archive, x, ev)
    remap!(archive)
    return result
end

# Whether `result` says that the archive has no cell for the solution, which then does not
# enter the buffer: only `RejectOutside()` discards solutions, and only those outside the
# boundaries.
discarded(archive::SlidingBoundariesArchive, result, ev) = discarded(archive.outside, archive, result, ev)
discarded(::OutsidePolicy, archive, result, ev) = false
discarded(::RejectOutside, archive, result, ev) = result.status == Rejected && cell_index(archive, ev.descriptor) == 0

"""
    add_batch!(archive::SlidingBoundariesArchive, xs, evs) -> Vector{AddResult}

Insert every solution as for any [`CellArchive`](@ref): each result is computed against
the archive as it was before the batch. Every solution that has a cell then enters the
buffer, and the archive remaps once if the number of offered solutions passed a multiple
of `remap_frequency` during the batch.
"""
function add_batch!(archive::SlidingBoundariesArchive, xs::AbstractVector, evs::AbstractVector)
    results = add_batch_to_cells!(archive, xs, evs)
    before = archive.offered
    for k in eachindex(xs, evs)
        discarded(archive, results[k], evs[k]) || remember!(archive, xs[k], evs[k])
    end
    remap_if_due!(archive, before)
    return results
end

# Puts a solution in the buffer, replacing the oldest one when the buffer is full.
function remember!(archive::SlidingBoundariesArchive{G,F,N,O,P}, x, ev) where {G,F,N,O,P}
    slot = archive.buffer_next
    archive.buffer_genotypes[slot] = x
    archive.buffer_fitness[slot] = stored_fitness(F, ev.fitness)
    archive.buffer_descriptors[slot] = ev.descriptor
    store_phenotype!(archive.buffer_phenotypes, slot, stored_phenotype(P, ev.phenotype))
    slots = length(archive.buffer_fitness)
    archive.buffer_next = slot == slots ? 1 : slot + 1
    archive.buffer_size = min(archive.buffer_size + 1, slots)
    archive.offered += 1
    return nothing
end

function remap_if_due!(archive::SlidingBoundariesArchive, before)
    f = archive.remap_frequency
    div(before, f) == div(archive.offered, f) || remap!(archive)
    return nothing
end

# Moves the boundaries to the quantiles of the buffered descriptors and fills the
# cells again from the old elites and the buffer.
function remap!(archive::SlidingBoundariesArchive{G,F,N}) where {G,F,N}
    m = archive.buffer_size
    slots = length(archive.buffer_fitness)
    values = view(archive.scratch, 1:m)
    for i in 1:N
        for k in 1:m
            values[k] = archive.buffer_descriptors[k][i]
        end
        sort!(values; alg=QuickSort)
        b = archive.boundaries[i]
        n = archive.dims[i]
        for j in 1:n
            b[j] = values[div((j - 1) * m, n)+1]
        end
        b[n+1] = values[m]
    end

    swap_storage!(archive)
    clear_cells!(archive)
    for c in archive.spare_occupied_idx
        add_during_remap!(archive, archive.spare_genotypes[c], archive.spare_fitness[c],
                          archive.spare_descriptors[c], archive.spare_phenotypes[c])
    end
    oldest = archive.buffer_next - m
    for k in 0:m-1
        slot = mod1(oldest + k, slots)
        add_during_remap!(archive, archive.buffer_genotypes[slot], archive.buffer_fitness[slot],
                          archive.buffer_descriptors[slot], archive.buffer_phenotypes[slot])
    end
    return nothing
end

function add_during_remap!(archive::SlidingBoundariesArchive, x, fitness, descriptor, phenotype)
    i = sliding_cell(archive, ClampOutside(), descriptor)
    if !archive.occupied[i] || fitness > archive.fitness[i]
        place!(archive, i, x, descriptor, fitness, phenotype)
    end
    return nothing
end

function swap_storage!(archive::SlidingBoundariesArchive)
    archive.occupied, archive.spare_occupied = archive.spare_occupied, archive.occupied
    archive.fitness, archive.spare_fitness = archive.spare_fitness, archive.fitness
    archive.descriptors, archive.spare_descriptors = archive.spare_descriptors, archive.descriptors
    archive.genotypes, archive.spare_genotypes = archive.spare_genotypes, archive.genotypes
    archive.phenotypes, archive.spare_phenotypes = archive.spare_phenotypes, archive.phenotypes
    archive.occupied_idx, archive.spare_occupied_idx = archive.spare_occupied_idx, archive.occupied_idx
    return nothing
end

function Base.empty!(archive::SlidingBoundariesArchive{G,F,N}) where {G,F,N}
    clear_cells!(archive)
    archive.buffer_next = 1
    archive.buffer_size = 0
    archive.offered = 0
    for i in 1:N
        copyto!(archive.boundaries[i], regular_boundaries(archive.lower[i], archive.upper[i], archive.dims[i]))
    end
    return archive
end
