"""
    DeviceGridArchive{T,F}(device, dims, bounds, dim; outside=ThrowOutside())
    DeviceGridArchive(prob::AbstractQDProblem; dims, device, bounds=nothing,
                      genotype=default_genotype(prob), fitness=Float64, outside=ThrowOutside())

A [`GridArchive`](@ref) held in the arrays of a device, for MAP-Elites on a GPU: the same
cells and rules, for solutions that are real vectors of length `dim` with entries of a
floating-point type `T`, and fitness values of type `F`. `device` is the array type to
store them in, such as `CuArray`; see [`DeviceCellArchive`](@ref) for the storage, the
kernels and what runs on the host. The second form takes `T` from `genotype`, `F` from
`fitness` and `dim` from [`random_solution(rng, prob)`](@ref QualityDiversity.random_solution).

[`MAPElites`](@ref) with `device` builds this archive with a [`DeviceVariationEmitter`](@ref).
`GridArchive(archive)` returns a host copy, for example to plot it.

# Example

```julia
using CUDA
prob = QualityDiversity.Benchmarks.Rastrigin(20)
alg = MAPElites(prob; dims=(100, 100), batch_size=10_000, device=CuArray,
                genotype=Vector{Float32}, fitness=Float32, variation=GaussianMutation(0.5f0))
result = solve(prob, alg; budget=1_000_000, rng=Xoshiro(1))
coverage(result.archive), GridArchive(result.archive)
```
"""
mutable struct DeviceGridArchive{T,F,N,O<:OutsidePolicy,M<:AbstractMatrix{T},V<:AbstractVector{F},
                                 B<:AbstractVector{Bool},I<:AbstractVector{Int32},K<:AbstractVector{UInt64},
                                 D<:AbstractMatrix} <: DeviceCellArchive{T,F,N}
    const dims::NTuple{N,Int}
    const lower::SVector{N,Float64}
    const upper::SVector{N,Float64}
    const outside::O
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

    function DeviceGridArchive{T,F,N,O,M,V,B,I,K,D}(dims, lower, upper, outside, occupied, fitness, descriptors,
                                                  genotypes, occupied_idx, count, best_key,
                                                  winner, flags) where {T,F,N,O,M,V,B,I,K,D}
        return new{T,F,N,O,M,V,B,I,K,D}(dims, lower, upper, outside, occupied, fitness, descriptors, genotypes,
                                      occupied_idx, count, best_key, winner, flags)
    end
end

function DeviceGridArchive{T,F}(device::Type, dims::Tuple, bounds::Tuple, dim::Integer;
                                outside::OutsidePolicy=ThrowOutside()) where {T<:AbstractFloat,F<:Real}
    N = length(dims)
    lower, upper = grid_box(dims, bounds)
    for i in 1:N
        dims[i] >= 1 ||
            throw(ArgumentError("dims[$i] is $(dims[i]) but every dimension needs at least one cell"))
        check_bound(i, lower[i], upper[i])
    end
    s = device_storage(device, T, F, N, dim, prod(dims))
    return DeviceGridArchive{T,F,N,typeof(outside),typeof(s.genotypes),typeof(s.fitness),typeof(s.occupied),
                             typeof(s.occupied_idx),typeof(s.best_key),typeof(s.descriptors)}(
        map(Int, dims), lower, upper, outside, s.occupied, s.fitness, s.descriptors, s.genotypes,
        s.occupied_idx, 0, s.best_key, s.winner, s.flags)
end

function DeviceGridArchive(prob::AbstractQDProblem; dims::Tuple, device::Type,
                           bounds::Union{Tuple,Nothing}=nothing,
                           genotype::Type=default_genotype(prob), fitness::Type{<:Real}=Float64,
                           outside::OutsidePolicy=ThrowOutside())
    T = device_eltype(genotype)
    return DeviceGridArchive{T,fitness}(device, dims, problem_bounds(prob, bounds), solution_dim(prob); outside)
end

function cell_index(archive::DeviceGridArchive, d::AbstractVector{<:Real})
    return grid_cell(archive.dims, archive.lower, archive.upper, archive.outside, d)
end

"""
    GridArchive(archive::DeviceGridArchive)

A [`GridArchive`](@ref) on the host with the cells, bounds, `outside` policy and elites
of `archive`, for the functions that need a host archive, such as plotting. Its
genotypes are `Vector{T}`.
"""
function GridArchive(archive::DeviceGridArchive{T,F}) where {T<:AbstractFloat,F<:Real}
    host = GridArchive{Vector{T},F}(archive.dims, descriptor_bounds(archive); outside=archive.outside)
    return copy_elites!(host, archive)
end
