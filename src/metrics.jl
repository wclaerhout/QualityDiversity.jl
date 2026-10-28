"""
    qd_score(archive; offset=0)

Sum of `fitness + offset` over the occupied cells of `archive`. The offset
keeps negative fitness values from lowering the score; use
[`fitness_offset`](@ref QualityDiversity.fitness_offset) of the problem as a
starting point.

The method for any [`AbstractArchive`](@ref) makes one pass over
[`elites`](@ref QualityDiversity.elites). An archive that keeps running totals
specializes it, as [`GridArchive`](@ref) does.
"""
function qd_score(archive::AbstractArchive; offset=0)
    total = 0.0
    for (_, ev) in elites(archive)
        total += ev.fitness + offset
    end
    return total
end

"""
    coverage(archive)

Fraction of the cells of `archive` that are occupied, in `[0, 1]`.
"""
coverage(archive::AbstractArchive) = length(archive) / capacity(archive)

"""
    max_fitness(archive)

Highest fitness among the occupied cells of `archive`. Throws an
`ArgumentError` if the archive is empty.
"""
function max_fitness(archive::AbstractArchive)
    isempty_archive(archive, "max_fitness")
    return maximum(ev.fitness for (_, ev) in elites(archive))
end

"""
    mean_fitness(archive)

Mean fitness over the occupied cells of `archive`. Throws an `ArgumentError`
if the archive is empty.
"""
function mean_fitness(archive::AbstractArchive)
    isempty_archive(archive, "mean_fitness")
    total = 0.0
    for (_, ev) in elites(archive)
        total += ev.fitness
    end
    return total / length(archive)
end

function isempty_archive(archive, name)
    length(archive) == 0 &&
        throw(ArgumentError("$name is undefined for an archive with no elites"))
    return nothing
end

"""
    CQDScore

The result of [`cqd_score`](@ref): `score`, the continuous QD score; `penalties`, the
penalty weights; `per_penalty`, the mean of `ω` over the targets for each weight, in the
order of `penalties`; and `num_targets`. `score` is `num_targets * sum(per_penalty)`.
With Makie loaded, `lines(result)` plots `per_penalty` against `penalties`.
"""
struct CQDScore
    score::Float64
    penalties::Vector{Float64}
    per_penalty::Vector{Float64}
    num_targets::Int
end

"""
    cqd_score(archive; targets, fitness_range, penalties=10, distance_max) -> CQDScore
    cqd_score(rng::AbstractRNG, archive; num_targets, fitness_range, penalties=10,
              distance_max, bounds=descriptor_bounds(archive)) -> CQDScore

The continuous quality-diversity score of Kent, Branke, Gaier and Mouret (2022), "A
discretization-free metric for assessing quality diversity algorithms"; pyribs
`cqd_score`. It measures an archive without reference to its cells: a user asks for a
solution near a target descriptor `G` and accepts a distance from it at a price `θ`, and
the archive recommends the elite that maximizes

    ω = f / (f_max - f_min) - θ * norm(d - G) / distance_max

where `f` is the elite's fitness and `d` its descriptor. The score is the sum of the
recommended `ω` over every target and every weight. As in the paper and in pyribs, the
fitness is divided by the range, not shifted by `f_min`.

- `targets`: descriptor points, each with the archive's number of dimensions. The second
  form draws `num_targets` of them uniformly within `bounds` with `rng`; the paper draws
  new targets for every run.
- `fitness_range = (f_min, f_max)`: the bounds of the fitness, from the problem or from
  all the runs being compared; it has no default.
- `penalties`: the weights `θ`, or their number, spread evenly over `[0, 1]` with both
  ends included (a single weight is 0).
- `distance_max`: the largest distance in descriptor space, by default the diagonal of
  [`descriptor_bounds(archive)`](@ref QualityDiversity.descriptor_bounds); an archive
  without bounds, such as an [`UnstructuredArchive`](@ref), needs it given.

Throws an `ArgumentError` for an empty archive, a target of the wrong dimension or not
finite, or invalid ranges. The time is proportional to the number of elites times the
number of targets times the number of weights. To follow the score during a run, compute
it in the callback of [`solve`](@ref).

# Example

```jldoctest
julia> using QualityDiversity, StaticArrays

julia> archive = GridArchive((2,), ((0, 1),));

julia> QualityDiversity.add!(archive, [0.0], Evaluation(1.0, SVector(0.25)));

julia> r = cqd_score(archive; targets=[[0.25], [0.75]], fitness_range=(0, 2), penalties=[0, 1]);

julia> r.score, r.per_penalty
(1.5, [0.5, 0.25])
```
"""
function cqd_score(archive::AbstractArchive{G,F,SVector{N,Float64}}; targets::AbstractVector,
                   fitness_range::Tuple{Real,Real}, penalties=10,
                   distance_max::Union{Real,Nothing}=nothing) where {G,F,N}
    isempty_archive(archive, "cqd_score")
    F <: Real || throw(ArgumentError("cqd_score needs a single real fitness, but the archive's fitness type is $F"))
    f_min, f_max = Float64.(fitness_range)
    (isfinite(f_min) && isfinite(f_max) && f_min < f_max) ||
        throw(ArgumentError("fitness_range must be (f_min, f_max) with finite f_min < f_max, got $fitness_range"))
    dmax = distance_max === nothing ? bounds_diagonal(archive) : Float64(distance_max)
    (isfinite(dmax) && dmax > 0) ||
        throw(ArgumentError("distance_max must be finite and greater than 0, got $distance_max"))
    weights = penalty_weights(penalties)
    isempty(targets) && throw(ArgumentError("targets must hold at least one descriptor point"))
    points = [checked_target(t, Val(N)) for t in targets]

    fitness = Float64[]
    descriptors = SVector{N,Float64}[]
    for (_, ev) in elites(archive)
        push!(fitness, Float64(ev.fitness) / (f_max - f_min))
        push!(descriptors, ev.descriptor)
    end
    totals = zeros(length(weights))
    distance = similar(fitness)
    for target in points
        for e in eachindex(descriptors, distance)
            distance[e] = norm(descriptors[e] - target) / dmax
        end
        for (j, θ) in pairs(weights)
            totals[j] += maximum(e -> fitness[e] - θ * distance[e], eachindex(fitness, distance))
        end
    end
    return CQDScore(sum(totals), weights, totals ./ length(points), length(points))
end

function cqd_score(archive::AbstractArchive; kwargs...)
    throw(ArgumentError("cqd_score measures distances between descriptors, which the descriptors of a $(nameof(typeof(archive))) do not have"))
end

function cqd_score(rng::AbstractRNG, archive::AbstractArchive{G,F,SVector{N,Float64}}; num_targets::Integer,
                   bounds::Union{Tuple,Nothing}=nothing, kwargs...) where {G,F,N}
    num_targets >= 1 || throw(ArgumentError("num_targets must be at least 1, got $num_targets"))
    box = bounds === nothing ? archive_bounds(archive, "pass bounds") : bounds
    length(box) == N || throw(ArgumentError("bounds has $(length(box)) entries but the archive has $N descriptor dimensions"))
    lower = SVector{N,Float64}(map(b -> Float64(b[1]), box))
    upper = SVector{N,Float64}(map(b -> Float64(b[2]), box))
    all(i -> isfinite(lower[i]) && isfinite(upper[i]) && lower[i] <= upper[i], 1:N) ||
        throw(ArgumentError("bounds must be finite (lo, hi) pairs with lo <= hi, got $box"))
    targets = [lower .+ rand(rng, SVector{N,Float64}) .* (upper .- lower) for _ in 1:num_targets]
    return cqd_score(archive; targets, kwargs...)
end

# The descriptor bounds of `archive`, or an error that says to pass them.
function archive_bounds(archive, hint)
    bounds = descriptor_bounds(archive)
    bounds === nothing &&
        throw(ArgumentError("$(nameof(typeof(archive))) does not report the bounds of its descriptor space; $hint"))
    return bounds
end

function bounds_diagonal(archive)
    box = archive_bounds(archive, "pass distance_max")
    return norm(map(b -> Float64(b[2]) - Float64(b[1]), box))
end

function penalty_weights(penalties::Integer)
    penalties >= 1 || throw(ArgumentError("penalties must be at least 1, got $penalties"))
    # one weight is 0, as numpy's linspace gives; `range` needs two ends
    return penalties == 1 ? [0.0] : collect(range(0.0, 1.0; length=penalties))
end

function penalty_weights(penalties::AbstractVector{<:Real})
    isempty(penalties) && throw(ArgumentError("penalties must hold at least one weight"))
    all(isfinite, penalties) || throw(ArgumentError("penalties must be finite, got $penalties"))
    return collect(Float64, penalties)
end

function checked_target(t::AbstractVector{<:Real}, ::Val{N}) where {N}
    length(t) == N ||
        throw(ArgumentError("a target has $(length(t)) dimensions but the archive has $N"))
    point = SVector{N,Float64}(t)
    all(isfinite, point) || throw(ArgumentError("targets must be finite, got $t"))
    return point
end
