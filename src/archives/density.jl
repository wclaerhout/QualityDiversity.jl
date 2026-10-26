"""
    DensityArchive(inner::AbstractArchive; bandwidth, buffer_capacity=10_000,
                   rng=Random.default_rng())

The archive of Density Descent Search (Lee et al., 2024, "Density Descent for Diversity
Optimization"; pyribs `DensityArchive`). It estimates how densely the descriptors found
so far cover descriptor space, so that emitters can move toward where they are sparse.

Every solution goes to `inner`, an archive with `SVector` descriptors such as a
[`GridArchive`](@ref), which holds the solutions: `solve` returns it, a run records its
metrics, and emitters draw their parents and restarts from it. The descriptors also go
to a buffer of `buffer_capacity` entries that stays a uniform sample of all the
descriptors seen, by reservoir sampling (Li, 1994, Algorithm L) with the generator `rng`.
The archive keeps its own copy of `rng`, since `add!` takes no generator; `init` copies
the archive, so a run is reproducible from the seed of `rng` and the seed of the run.
For a reproducible run, seed the archive as well as `solve`, e.g. `rng=Xoshiro(1)`:
without it, `rng` is `Random.default_rng()`, which the seed of `solve` does not set.

The density at a descriptor `y` is the Gaussian kernel density estimate of the buffer,

    sum(φ(norm(y - b) / h) for b in buffer) / (length(buffer) * h)

with `φ` the standard normal density and `h = bandwidth`, in descriptor units, and zero
while the buffer is empty. `bandwidth` has no default; the paper finds about 5% of the
width of descriptor space a good start. A result has the status that `inner` gives the
solution and `improvement = -density`, so that [`ImprovementRanker`](@ref) ranks the
lowest density first. Every result of a batch is computed against the buffer as it was
before the batch. A density takes time linear in the buffer; [`add_batch!`](@ref
QualityDiversity.add_batch!) computes those of a batch on several threads when the batch
is large enough to pay for them, with the same results whatever the number of threads.

[`result_archive`](@ref QualityDiversity.result_archive) returns `inner`, and `length`,
`capacity`, `cell_index`, `sample_elites`, `elites`, `descriptor_bounds` and the
metrics are those of `inner`. `empty!` empties `inner` and the buffer. The fitness type
must be a floating-point type, for the density. [`DDS`](@ref) builds the algorithm.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> archive = DensityArchive(GridArchive((10, 10), ((0, 1), (0, 1))); bandwidth=0.05, rng=Xoshiro(1));

julia> length(archive), QualityDiversity.capacity(archive)
(0, 100)
```
"""
mutable struct DensityArchive{G,F<:AbstractFloat,N,P,A<:AbstractArchive{G,F,SVector{N,Float64},P},R<:AbstractRNG} <: AbstractArchive{G,F,SVector{N,Float64},P}
    const inner::A
    const rng::R
    const bandwidth::Float64
    # The first `filled` entries are in use.
    const buffer::Vector{SVector{N,Float64}}
    filled::Int
    # State of Algorithm L once the buffer is full: the weight `w` and the number of
    # descriptors to pass over before the next one replaces an entry.
    w::Float64
    skip::Int

    function DensityArchive{G,F,N,P,A,R}(rng::R, inner::A, bandwidth, buffer_capacity) where {G,F<:AbstractFloat,N,P,A<:AbstractArchive{G,F,SVector{N,Float64},P},R<:AbstractRNG}
        length(inner) == 0 ||
            throw(ArgumentError("the archive to wrap must be empty but holds $(length(inner)) solutions"))
        (isfinite(bandwidth) && bandwidth > 0) ||
            throw(ArgumentError("bandwidth must be finite and greater than 0, got $bandwidth"))
        buffer_capacity >= 1 || throw(ArgumentError("buffer_capacity must be at least 1, got $buffer_capacity"))
        archive = new{G,F,N,P,A,R}(inner, rng, Float64(bandwidth), Vector{SVector{N,Float64}}(undef, buffer_capacity), 0, 0.0, 0)
        start_reservoir!(archive)
        return archive
    end
end

function DensityArchive(inner::AbstractArchive{G,F,SVector{N,Float64},P}; bandwidth::Real,
                        buffer_capacity::Integer=10_000,
                        rng::AbstractRNG=Random.default_rng()) where {G,F<:AbstractFloat,N,P}
    # the copy of the default generator is a Xoshiro; dispatch binds its type
    return density_archive(copy(rng), inner, bandwidth, Int(buffer_capacity))
end

function density_archive(own::R, inner::A, bandwidth, buffer_capacity) where {G,F<:AbstractFloat,N,P,A<:AbstractArchive{G,F,SVector{N,Float64},P},R<:AbstractRNG}
    return DensityArchive{G,F,N,P,A,R}(own, inner, bandwidth, buffer_capacity)
end

function DensityArchive(inner::AbstractArchive; kwargs...)
    throw(ArgumentError("a DensityArchive measures distances between descriptors, so it wraps an archive with SVector descriptors and a floating-point fitness, not a $(nameof(typeof(inner)))"))
end

# The first weight of Algorithm L and the number of descriptors to pass over after the
# buffer has filled.
function start_reservoir!(archive::DensityArchive)
    archive.w = exp(log(rand(archive.rng)) / length(archive.buffer))
    archive.skip = reservoir_skip(archive)
    return nothing
end

function reservoir_skip(archive::DensityArchive)
    skip = floor(log(rand(archive.rng)) / log1p(-archive.w))
    # `w` shrinks as more descriptors are seen; cap the count where it no longer fits
    return isfinite(skip) && skip < typemax(Int) ÷ 2 ? Int(skip) : typemax(Int) ÷ 2
end

# Offers one descriptor to the buffer.
function remember!(archive::DensityArchive, point)
    size = length(archive.buffer)
    if archive.filled < size
        archive.filled += 1
        archive.buffer[archive.filled] = point
    elseif archive.skip > 0
        archive.skip -= 1
    else
        archive.buffer[rand(archive.rng, 1:size)] = point
        archive.w *= exp(log(rand(archive.rng)) / size)
        archive.skip = reservoir_skip(archive)
    end
    return nothing
end

"""
    density(archive::DensityArchive, d) -> Float64

The kernel density estimate of the buffer at descriptor `d`; zero while the buffer is
empty. See [`DensityArchive`](@ref). Throws an `ArgumentError` if `d` has the wrong number
of dimensions or a component that is not finite, which would make every later density
`NaN` once in the buffer.
"""
density(archive::DensityArchive, d::AbstractVector{<:Real}) = kernel_density(archive, density_point(archive, d))

# `d` as an `SVector`, after checking its dimension and that it is finite.
function density_point(::DensityArchive{G,F,N}, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    point = SVector{N,Float64}(d)
    for i in 1:N
        isfinite(point[i]) ||
            throw(ArgumentError("descriptor component $i is $(point[i]), which is not finite"))
    end
    return point
end

# The density at `point`, its terms added in the order of the buffer.
function kernel_density(archive::DensityArchive, point)
    archive.filled == 0 && return 0.0
    h = archive.bandwidth
    total = 0.0
    for i in 1:archive.filled
        total += exp(-0.5 * sum(abs2, point - archive.buffer[i]) / h^2)
    end
    return total / (sqrt(2π) * archive.filled * h)
end

function add!(archive::DensityArchive{G,F}, x, ev::Evaluation{<:Real}) where {G,F}
    improvement = -convert(F, density(archive, ev.descriptor))
    result = add!(archive.inner, x, ev)
    remember!(archive, ev.descriptor)
    return AddResult(result.status, improvement)
end

function add_batch!(archive::DensityArchive{G,F}, xs::AbstractVector, evs::AbstractVector) where {G,F}
    points = [density_point(archive, ev.descriptor) for ev in evs]    # throws before any change
    densities = Vector{Float64}(undef, length(points))
    # Each density is summed on one thread in the order of the buffer, so the number of
    # threads changes no result.
    evaluate_each!(j -> kernel_density(archive, points[j]), densities)
    improvements = [-convert(F, d) for d in densities]
    results = add_batch!(archive.inner, xs, evs)
    for ev in evs
        remember!(archive, ev.descriptor)
    end
    return [AddResult(r.status, improvement) for (r, improvement) in zip(results, improvements)]
end

result_archive(archive::DensityArchive) = result_archive(archive.inner)

Base.length(archive::DensityArchive) = length(archive.inner)

capacity(archive::DensityArchive) = capacity(archive.inner)

function Base.empty!(archive::DensityArchive)
    empty!(archive.inner)
    archive.filled = 0
    start_reservoir!(archive)
    return archive
end

descriptor_bounds(archive::DensityArchive) = descriptor_bounds(archive.inner)

cell_index(archive::DensityArchive, d) = cell_index(archive.inner, d)

sample_elites(rng::AbstractRNG, archive::DensityArchive, n::Integer) = sample_elites(rng, archive.inner, n)

elites(archive::DensityArchive) = elites(archive.inner)

qd_score(archive::DensityArchive; offset=0) = qd_score(archive.inner; offset)

max_fitness(archive::DensityArchive) = max_fitness(archive.inner)

mean_fitness(archive::DensityArchive) = mean_fitness(archive.inner)
