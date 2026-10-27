"""
    DeviceCVTArchive{T,F}(device, centroids, bounds, dim; outside=ThrowOutside())
    DeviceCVTArchive{T,F}(device, cells, bounds, dim; rng=Random.default_rng(),
                          samples=100_000, max_iterations=100, seeding=KMeansPlusPlus(),
                          outside=ThrowOutside())
    DeviceCVTArchive(prob::AbstractQDProblem; cells, device, rng=Random.default_rng(),
                     bounds=nothing, samples=100_000, max_iterations=100,
                     seeding=KMeansPlusPlus(), genotype=default_genotype(prob), fitness=Float64,
                     outside=ThrowOutside())

A [`CVTArchive`](@ref) held in the arrays of a device, for MAP-Elites on a GPU: the Voronoi
cells of a set of centroids, with the rules of the host archive, for solutions that are real
vectors of length `dim` with entries of a floating-point type `T`, and fitness values of
type `F`. `device` is the array type to store them in, such as `CuArray`; see
[`DeviceCellArchive`](@ref) for the storage, the kernels and what runs on the host.

The centroids are given, or chosen on the host by k-means from `samples` uniform points of
the box `bounds`, exactly as for a `CVTArchive`, with the same `seeding`: the same
centroids, or the same generator, seed and seeding, give the same cells, and `centroids(archive)` returns them. For a reproducible
run, seed the archive as well as `solve`, e.g. `rng=Xoshiro(1)`: without it, `rng` is
`Random.default_rng()`, which the seed of `solve` does not set. The last form takes
`T` from `genotype`, `F` from `fitness` and `dim` from
[`random_solution(rng, prob)`](@ref QualityDiversity.random_solution).

On the device, the box is divided into a regular grid of about as many buckets as cells,
and each bucket lists the centroids that can be nearest to a point in it, computed once on
the host. A kernel finds the bucket of a descriptor and compares it with those centroids
only, one thread per solution. The distances are computed in `Float64` in the order of the
host archive's k-d tree, so the two assign the same cell, the lowest index if two
centroids are equally near. In many descriptor dimensions the buckets hold more centroids,
and in the limit one bucket holds all of them.

[`MAPElites(archive)`](@ref MAPElites) runs MAP-Elites over it with a
[`DeviceVariationEmitter`](@ref). `CVTArchive(archive)` returns a host copy, for example to
plot it.

# Example

```julia
using CUDA
prob = QualityDiversity.Benchmarks.Rastrigin(20)
archive = DeviceCVTArchive(prob; cells=10_000, device=CuArray, rng=Xoshiro(1),
                           genotype=Vector{Float32}, fitness=Float32)
alg = MAPElites(archive; batch_size=10_000, variation=GaussianMutation(0.5f0))
result = solve(prob, alg; budget=1_000_000, rng=Xoshiro(2))
```
"""
mutable struct DeviceCVTArchive{T,F,N,O<:OutsidePolicy,M<:AbstractMatrix{T},V<:AbstractVector{F},
                                B<:AbstractVector{Bool},I<:AbstractVector{Int32},K<:AbstractVector{UInt64},
                                C<:AbstractMatrix{Float64},D<:AbstractMatrix} <: DeviceCellArchive{T,F,N}
    const lower::SVector{N,Float64}
    const upper::SVector{N,Float64}
    const outside::O
    # The centroids on the host, the k-d tree that `cell_index` searches, and for the
    # kernels the same centroids as the columns of an `N × cells` device matrix, with the
    # buckets of a `CentroidBuckets` in device arrays.
    const centroids::Vector{SVector{N,Float64}}
    const tree::CentroidTree{N}
    const centroid_matrix::C
    const bucket_dims::NTuple{N,Int}
    const bucket_offsets::I
    const bucket_members::I
    const occupied::B
    const fitness::V
    const descriptors::D
    const genotypes::M
    # The occupied cells in increasing order; the first `count` entries are valid.
    const occupied_idx::I
    count::Int
    const best_key::K
    const winner::I
    const flags::I

    function DeviceCVTArchive{T,F,N,O,M,V,B,I,K,C,D}(lower, upper, outside, centroids, tree, centroid_matrix,
                                                   bucket_dims, bucket_offsets, bucket_members, occupied, fitness,
                                                   descriptors, genotypes, occupied_idx, count, best_key, winner,
                                                   flags) where {T,F,N,O,M,V,B,I,K,C,D}
        return new{T,F,N,O,M,V,B,I,K,C,D}(lower, upper, outside, centroids, tree, centroid_matrix, bucket_dims,
                                         bucket_offsets, bucket_members, occupied, fitness, descriptors,
                                         genotypes, occupied_idx, count, best_key, winner, flags)
    end
end

function DeviceCVTArchive{T,F}(device::Type, centroids::AbstractVector{<:AbstractVector{<:Real}}, bounds::Tuple,
                               dim::Integer; outside::OutsidePolicy=ThrowOutside()) where {T<:AbstractFloat,F<:Real}
    lower, upper = box(bounds)
    return device_cvt_archive(T, F, device, centroid_points(centroids, bounds), lower, upper, dim, outside)
end

function DeviceCVTArchive{T,F}(device::Type, cells::Integer, bounds::Tuple, dim::Integer;
                               rng::AbstractRNG=Random.default_rng(), samples::Integer=100_000,
                               max_iterations::Integer=100, seeding::CentroidSeeding=KMeansPlusPlus(),
                               outside::OutsidePolicy=ThrowOutside()) where {T<:AbstractFloat,F<:Real}
    lower, upper = box(bounds)
    points = kmeans_centroids(rng, cells, lower, upper; samples, max_iterations, seeding)
    return device_cvt_archive(T, F, device, points, lower, upper, dim, outside)
end

function DeviceCVTArchive(prob::AbstractQDProblem; cells::Integer, device::Type,
                          rng::AbstractRNG=Random.default_rng(), bounds::Union{Tuple,Nothing}=nothing,
                          samples::Integer=100_000, max_iterations::Integer=100,
                          seeding::CentroidSeeding=KMeansPlusPlus(),
                          genotype::Type=default_genotype(prob), fitness::Type{<:Real}=Float64,
                          outside::OutsidePolicy=ThrowOutside())
    T = device_eltype(genotype)
    return DeviceCVTArchive{T,fitness}(device, cells, problem_bounds(prob, bounds), solution_dim(prob);
                                       rng, samples, max_iterations, seeding, outside)
end

function device_cvt_archive(::Type{T}, ::Type{F}, device, points::Vector{SVector{N,Float64}}, lower, upper,
                            dim, outside) where {T,F,N}
    check_centroids(points, lower, upper)
    s = device_storage(device, T, F, N, dim, length(points))
    tree = CentroidTree(points)
    buckets = CentroidBuckets(tree, lower, upper)
    matrix = copyto!(similar(s.genotypes, Float64, N, length(points)), reduce(hcat, points))
    offsets = copyto!(similar(s.occupied_idx, Int32, length(buckets.offsets)), buckets.offsets)
    members = copyto!(similar(s.occupied_idx, Int32, length(buckets.members)), buckets.members)
    return DeviceCVTArchive{T,F,N,typeof(outside),typeof(s.genotypes),typeof(s.fitness),typeof(s.occupied),
                            typeof(s.occupied_idx),typeof(s.best_key),typeof(matrix),typeof(s.descriptors)}(
        lower, upper, outside, points, tree, matrix, buckets.dims, offsets, members, s.occupied, s.fitness,
        s.descriptors, s.genotypes, s.occupied_idx, 0, s.best_key, s.winner, s.flags)
end

centroids(archive::DeviceCVTArchive) = archive.centroids

"""
    cell_index(archive::DeviceCVTArchive, d)

The cell of descriptor `d`, as [`cell_index`](@ref QualityDiversity.cell_index) of a
[`CVTArchive`](@ref) with the same centroids computes it; the kernels of the archive
assign the same cell.
"""
function cell_index(archive::DeviceCVTArchive{T,F,N}, d::AbstractVector{<:Real}) where {T,F,N}
    length(d) == N ||
        throw(ArgumentError("descriptor has $(length(d)) dimensions but the archive has $N"))
    point = SVector(ntuple(i -> inside_component(archive, d, i), Val(N)))
    any(isnan, point) && return 0
    return nearest(archive.tree, point)
end

"""
    CVTArchive(archive::DeviceCVTArchive)

A [`CVTArchive`](@ref) on the host with the centroids, bounds, `outside` policy and elites
of `archive`, for the functions that need a host archive, such as plotting. Its genotypes
are `Vector{T}`.
"""
function CVTArchive(archive::DeviceCVTArchive{T,F}) where {T<:AbstractFloat,F<:Real}
    host = CVTArchive{Vector{T},F}(archive.centroids, descriptor_bounds(archive); outside=archive.outside)
    return copy_elites!(host, archive)
end
