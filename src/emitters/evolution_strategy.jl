const RESAMPLE_LIMIT = 1_000

"""
    AbstractEvolutionStrategy

Supertype of the optimizers inside a [`CMAEmitter`](@ref): [`CMAES`](@ref),
[`SepCMAES`](@ref), [`LMMAES`](@ref) and [`OpenAIES`](@ref). A strategy keeps a
search distribution over real vectors, draws batches from it, and moves it toward
the solutions that the emitter's ranker places first. It maximizes: the ranking
passed to `update!` lists the best solution first.

A strategy has the fields `dim`, the length of a solution, `batch_size`, `x`, a
`dim × batch_size` matrix that holds the last batch, one solution per column, and
`lower` and `upper`, the bounds of a solution as vectors of length `dim`, which a
[`CMAEmitter`](@ref) reads to move the elite that a search starts from inside them. It
defines

- `StrategyType(dim; sigma0, batch_size, lower, upper)`: `sigma0` is the initial step
  size, `lower` and `upper` are the bounds of a solution, scalars or vectors, and
  `batch_size` has a default;
- `reset!(es, x0)`: restart the search at `x0`;
- `sample!(rng, es)`: draw a batch into `es.x` and return it, drawing a solution
  again if it falls outside the bounds. After $(RESAMPLE_LIMIT) draws outside, the last
  one moves to the nearest point inside the bounds: with the mean on the boundary of the
  box, a draw lands inside with a probability that halves with every dimension. A
  strategy that moves its mean along the stored steps of its solutions computes the step
  of a moved solution from where it lies;
- `update!(es, order, parents)`: adapt to the last batch, where `order` ranks its
  solutions from best to worst and the first `parents` of them are recombined;
- `should_stop(es, keys)`: whether the search has converged or broken down, given the
  ranking keys of the last batch from best to worst; see
  [`rank_key`](@ref QualityDiversity.rank_key).
"""
abstract type AbstractEvolutionStrategy end

function default_batch_size(dim::Integer)
    dim >= 1 || throw(ArgumentError("dim must be at least 1, got $dim"))
    return 4 + floor(Int, 3 * log(dim))
end

# Checks the arguments that every strategy takes and returns the bounds as vectors.
function check_strategy_arguments(dim, sigma0, batch_size, lower, upper)
    dim >= 1 || throw(ArgumentError("dim must be at least 1, got $dim"))
    sigma0 > 0 && isfinite(sigma0) ||
        throw(ArgumentError("sigma0 must be positive and finite, got $sigma0"))
    batch_size >= 2 ||
        throw(ArgumentError("batch_size must be at least 2 so that there is a parent, got $batch_size"))
    lo = bound_vector(lower, dim, "lower")
    hi = bound_vector(upper, dim, "upper")
    for i in 1:dim
        lo[i] < hi[i] ||
            throw(ArgumentError("lower[$i] is $(lo[i]) but must be less than upper[$i], which is $(hi[i])"))
    end
    return lo, hi
end

function bound_vector(bound, dim, name)
    v = bound isa Real ? fill(Float64(bound), dim) : collect(Float64, bound)
    length(v) == dim ||
        throw(ArgumentError("$name has $(length(v)) entries but the dimension is $dim"))
    any(isnan, v) && throw(ArgumentError("$name must not contain NaN"))
    return v
end

function check_start(x0, dim)
    length(x0) == dim ||
        throw(ArgumentError("x0 has $(length(x0)) entries but the dimension is $dim"))
    all(isfinite, x0) || throw(ArgumentError("x0 must be finite"))
    return nothing
end

# Checks that a start of the search lies inside the bounds of the strategy.
function check_inside(x0, lower, upper)
    for (i, v) in zip(eachindex(lower), x0)
        lower[i] <= v <= upper[i] ||
            throw(ArgumentError("x0 has the entry $v, outside the bounds [$(lower[i]), $(upper[i])]"))
    end
    return nothing
end

# Whether column `k` of `x` lies inside the bounds.
function column_inside(x, k, lower, upper)
    for i in axes(x, 1)
        lower[i] <= x[i, k] <= upper[i] || return false
    end
    return true
end

# Moves column `k` of `x` to the nearest point inside the bounds.
function clip_column!(x, k, lower, upper)
    for i in axes(x, 1)
        x[i, k] = clamp(x[i, k], lower[i], upper[i])
    end
    return x
end

# Clips column `k` of `x` and sets column `k` of `y` to its step from `mean` in units of
# `sigma`.
function clip_step!(x, y, k, mean, sigma, lower, upper)
    clip_column!(x, k, lower, upper)
    for i in axes(x, 1)
        y[i, k] = (x[i, k] - mean[i]) / sigma
    end
    return x
end

# Recombination weights `log(parents + 1/2) - log(i)`, normalized to sum to one.
recombination_weights(parents) = recombination_weights!(Vector{Float64}(undef, parents), parents)

# The recombination weights written into the first `parents` entries of `buffer`, returned
# as a view of them.
function recombination_weights!(buffer, parents)
    weights = view(buffer, 1:parents)
    for i in 1:parents
        weights[i] = log(parents + 0.5) - log(i)
    end
    weights ./= sum(weights)
    return weights
end

# Columns `order[1:parents]` of `y` copied into the first `parents` columns of `buffer`,
# returned as a view of them.
function chosen_columns!(buffer, y, order, parents)
    for j in 1:parents, i in axes(y, 1)
        buffer[i, j] = y[i, order[j]]
    end
    return view(buffer, :, 1:parents)
end

# Distance between two ranking keys, a number or a tuple of numbers.
# Equal keys are at distance zero also when they are infinite.
key_distance(a::Real, b::Real) = a == b ? 0.0 : abs(Float64(a) - Float64(b))
key_distance(a::Tuple, b::Tuple) = sqrt(sum(abs2, map(key_distance, a, b)))

# The batch is flat when the best and the worst key are the same.
flat_ranking(keys) = length(keys) >= 2 && key_distance(first(keys), last(keys)) < 1e-12

# The termination criteria that CMA-ES and sep-CMA-ES share: a distribution that
# is ill-conditioned, has shrunk, or has a non-finite state, or a flat ranking.
function covariance_converged(sigma, mean, eigenvalues, keys)
    isfinite(sigma) && all(isfinite, mean) && all(isfinite, eigenvalues) || return true
    lo, hi = extrema(eigenvalues)
    hi / lo > 1e14 && return true
    sigma * sqrt(hi) < 1e-11 && return true
    return flat_ranking(keys)
end
