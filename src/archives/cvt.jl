"""
    CVTArchive{G,F}(centroids, bounds; outside=ThrowOutside(), phenotype=Nothing)
    CVTArchive{G,F}(cells, bounds; rng=Random.default_rng(), samples=100_000,
                    max_iterations=100, seeding=KMeansPlusPlus(), outside=ThrowOutside(),
                    phenotype=Nothing)
    CVTArchive{G}(...)
    CVTArchive(...)
    CVTArchive(prob::AbstractQDProblem; cells, rng=Random.default_rng(), bounds=nothing,
               samples=100_000, max_iterations=100, seeding=KMeansPlusPlus(),
               genotype=default_genotype(prob), fitness=Float64, outside=ThrowOutside(),
               phenotype=Nothing)

Archive that divides descriptor space into the Voronoi cells of a set of
centroids and keeps the highest-fitness solution found in each cell. A
descriptor belongs to the cell of its nearest centroid. Compared with
[`GridArchive`](@ref), the number of cells does not grow exponentially with the
number of descriptor dimensions, and the cells adapt to a measure: with centroids
from k-means, a region that the descriptors cover densely gets small cells.

The centroids are either given, as a vector of descriptor-sized vectors, or
chosen from `samples` points drawn uniformly from the box `bounds` by k-means
(Lloyd's algorithm), which makes the cells equal in expected population. `seeding`, a
[`CentroidSeeding`](@ref QualityDiversity.CentroidSeeding), picks the starting centroids:
[`KMeansPlusPlus()`](@ref QualityDiversity.KMeansPlusPlus) by default, or
[`GreedyKMeansPlusPlus()`](@ref QualityDiversity.GreedyKMeansPlusPlus), which gives
slightly more even cells for many centroids in few dimensions at several times the cost. The iterations stop after `max_iterations` (100) or when
one lowers the total squared distance from the samples to their centroids by
less than 0.1%. The second form draws its random numbers from `rng`; the same
seed gives the same centroids. For a reproducible run, seed the archive as well as
`solve`, e.g. `rng=Xoshiro(1)`: without it, `rng` is `Random.default_rng()`, which the
seed of `solve` does not set. `centroids(archive)` returns the centroids, and passing them
back rebuilds the same archive. The last form takes the bounds from
[`descriptor_bounds(prob)`](@ref QualityDiversity.descriptor_bounds) unless `bounds`
is given.

`bounds` is a tuple with one `(lo, hi)` pair per descriptor dimension. Given
centroids must lie inside the box and be distinct. `G` is the genotype type,
`Vector{Float64}` by default, and `F<:Real` the fitness type, `Float64` by
default. `outside` says what happens to a descriptor outside the box, as for
[`GridArchive`](@ref): `ThrowOutside()`, `ClampOutside()` or `RejectOutside()`.
`phenotype` is the type of the phenotypes the archive stores, `Nothing` for none.

The nearest centroid is found with a k-d tree, so a lookup takes a logarithmic
number of distance evaluations in few dimensions and does not allocate. In many
dimensions a k-d tree prunes little, and a lookup approaches a pass over all
centroids.

The archive keeps the same running totals as [`GridArchive`](@ref), so the
metrics take constant time.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> archive = CVTArchive(50, ((0, 1), (0, 1)); rng=Xoshiro(1), samples=5_000);

julia> length(archive), QualityDiversity.capacity(archive)
(0, 50)

julia> a = CVTArchive(50, ((0, 1), (0, 1)); samples=5_000);    # no seed

julia> b = CVTArchive(50, ((0, 1), (0, 1)); samples=5_000);

julia> QualityDiversity.centroids(a) == QualityDiversity.centroids(b)    # other cells
false
```
"""
mutable struct CVTArchive{G,F<:Real,N,O<:OutsidePolicy,P} <: CellArchive{G,F,SVector{N,Float64},P}
    const lower::SVector{N,Float64}
    const upper::SVector{N,Float64}
    const outside::O
    const centroids::Vector{SVector{N,Float64}}
    const tree::CentroidTree{N}
    const occupied::BitVector
    const fitness::Vector{F}
    const descriptors::Vector{SVector{N,Float64}}
    const genotypes::Vector{G}
    const phenotypes::Vector{P}
    const occupied_idx::Vector{Int}
    # Running totals, as in GridArchive.
    fitness_sum::Float64
    fitness_sum_error::Float64
    fitness_max::F

    function CVTArchive{G,F,N,O,P}(centroids::Vector{SVector{N,Float64}}, lower, upper,
                                   outside) where {G,F<:Real,N,O<:OutsidePolicy,P}
        check_centroids(centroids, lower, upper)
        ncells = length(centroids)
        return new{G,F,N,O,P}(lower, upper, outside, centroids, CentroidTree(centroids),
                              falses(ncells),
                              Vector{F}(undef, ncells),
                              Vector{SVector{N,Float64}}(undef, ncells),
                              Vector{G}(undef, ncells),
                              Vector{P}(undef, ncells),
                              Int[],
                            0.0, 0.0, zero(F))
    end
end

# Throws unless the box is valid and the centroids are distinct points inside it; the
# CVT archives on the host and on a device share it.
function check_centroids(centroids::Vector{SVector{N,Float64}}, lower, upper) where {N}
    for i in 1:N
        check_bound(i, lower[i], upper[i])
    end
    isempty(centroids) && throw(ArgumentError("need at least one centroid"))
    for (j, c) in enumerate(centroids)
        all(i -> lower[i] <= c[i] <= upper[i], 1:N) ||
            throw(ArgumentError("centroid $j is $(Tuple(c)), outside the bounds"))
    end
    sorted = sort!([Tuple(c) for c in centroids])
    for j in 2:length(sorted)
        sorted[j] == sorted[j-1] &&
            throw(ArgumentError("two centroids are identical, at $(sorted[j]); one of their cells could never be reached"))
    end
    return nothing
end

# The given centroids as points of the box `bounds`.
function centroid_points(centroids, bounds::Tuple)
    N = length(bounds)
    for (j, c) in enumerate(centroids)
        length(c) == N ||
            throw(ArgumentError("centroid $j has $(length(c)) entries but bounds has $N"))
    end
    return SVector{N,Float64}[SVector{N,Float64}(c) for c in centroids]
end

# `cells` centroids chosen by k-means from `samples` uniform points of the box.
function kmeans_centroids(rng::AbstractRNG, cells::Integer, lower::SVector{N,Float64}, upper::SVector{N,Float64};
                          samples::Integer, max_iterations::Integer,
                          seeding::CentroidSeeding=KMeansPlusPlus()) where {N}
    cells >= 1 || throw(ArgumentError("cells must be at least 1, got $cells"))
    samples >= cells ||
        throw(ArgumentError("samples ($samples) must be at least cells ($cells) to place that many centroids"))
    points = [lower .+ (upper .- lower) .* rand(rng, SVector{N,Float64}) for _ in 1:samples]
    return kmeans(rng, points, Int(cells); max_iterations=Int(max_iterations), seeding)
end

function CVTArchive{G,F}(centroids::AbstractVector{<:AbstractVector{<:Real}}, bounds::Tuple;
                         outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing) where {G,F<:Real}
    lower, upper = box(bounds)
    points = centroid_points(centroids, bounds)
    return CVTArchive{G,F,length(bounds),typeof(outside),phenotype}(points, lower, upper, outside)
end

function CVTArchive{G,F}(cells::Integer, bounds::Tuple; rng::AbstractRNG=Random.default_rng(),
                         samples::Integer=100_000, max_iterations::Integer=100,
                         seeding::CentroidSeeding=KMeansPlusPlus(),
                         outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing) where {G,F<:Real}
    lower, upper = box(bounds)
    centers = kmeans_centroids(rng, cells, lower, upper; samples, max_iterations, seeding)
    return CVTArchive{G,F,length(bounds),typeof(outside),phenotype}(centers, lower, upper, outside)
end

CVTArchive{G}(args...; kwargs...) where {G} = CVTArchive{G,Float64}(args...; kwargs...)

CVTArchive(args...; kwargs...) = CVTArchive{Vector{Float64}}(args...; kwargs...)

function CVTArchive(prob::AbstractQDProblem; cells::Integer, rng::AbstractRNG=Random.default_rng(),
                    bounds::Union{Tuple,Nothing}=nothing,
                    genotype::Type=default_genotype(prob), fitness::Type{<:Real}=Float64,
                    samples::Integer=100_000, max_iterations::Integer=100,
                    seeding::CentroidSeeding=KMeansPlusPlus(),
                    outside::OutsidePolicy=ThrowOutside(), phenotype::Type=Nothing)
    return CVTArchive{genotype,fitness}(cells, problem_bounds(prob, bounds); rng, samples, max_iterations,
                                        seeding, outside, phenotype)
end

"""
    centroids(archive::CVTArchive) -> Vector{SVector}

The centroids of the cells, in the order of the cell indices.
"""
centroids(archive::CVTArchive) = archive.centroids

"""
    cell_index(archive::CVTArchive, d)

Index of the cell whose centroid is nearest to descriptor `d`; the lowest index
if two are equally near. Throws an `ArgumentError` if `d` has the wrong number
of dimensions. What happens to a component outside the bounds or `NaN` depends
on the archive's `outside` policy, as for [`GridArchive`](@ref): `ThrowOutside()`
throws, `ClampOutside()` uses the nearest point of the box (and throws for
`NaN`), and `RejectOutside()` returns `0`, which is not a cell.
"""
function cell_index(archive::CVTArchive{G,F,N}, d::AbstractVector{<:Real}) where {G,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    point = SVector(ntuple(i -> inside_component(archive, d, i), Val(N)))
    any(isnan, point) && return 0
    return nearest(archive.tree, point)
end

# The i-th component of `d` after the outside policy of a CVT archive on the host or a
# device; NaN rejects the descriptor.
function inside_component(archive, d, i)
    lo = archive.lower[i]
    hi = archive.upper[i]
    x = Float64(d[i])
    lo <= x <= hi && return x
    return resolve_outside(archive.outside, i, x, lo, hi)
end

descriptor_bounds(archive::CVTArchive) = Tuple(zip(archive.lower, archive.upper))
