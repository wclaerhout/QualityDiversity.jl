"""
    LMMAES(dim; sigma0, batch_size=default, n_vectors=batch_size, lower=-Inf, upper=Inf)

Limited-memory matrix adaptation evolution strategy (Loshchilov, Glasmachers and
Beyer, 2019, "Large Scale Black-Box Optimization by Limited-Memory Matrix
Adaptation"), maximizing. For solutions of high dimension it replaces the covariance
matrix by `n_vectors` vectors, which cost `O(n_vectors * dim)` memory and time for a
solution, and it needs no eigendecomposition.

A solution is `x = m + σ d`, where `d` is a standard normal vector `z` that the
vectors `M_j` stretch one after the other, `d ← (1 - c_d,j) d + c_d,j M_j (M_jᵀ d)`.
After a generation the vectors and the path of the step size move toward the
recombined `z` of the parents, and the step size follows the cumulative step size
rule. The learning rates are `c_σ = 2 batch_size / dim`, `c_d,j = 1 / (1.5^(j-1) dim)`
and `c_c,j = batch_size / (4^(j-1) dim)`, as in pyribs; `n_vectors` is at most `dim`.
The solutions of a batch are not mirrored.

# A high dimension only

LM-MA-ES is meant for hundreds of parameters or more, where the covariance matrix of
[`CMAES`](@ref) costs too much, and it requires a batch of at most half the dimension,
`2 batch_size ≤ dim`. Each generation the path of the step size keeps a fraction
`1 - c_σ` of itself; with a larger batch `c_σ` exceeds 1, the fraction is negative, the
path changes sign every generation and the step size follows noise instead of the
search. The constructor therefore throws for such a batch. With the default batch,
`4 + floor(3 log(dim))`, this means `dim ≥ 26`. For fewer parameters use
[`CMAES`](@ref) or [`SepCMAES`](@ref), or pass a smaller `batch_size`. pyribs refuses
only `batch_size > dim`.

The arguments, `sample!`, `update!` and `should_stop` are those of
[`AbstractEvolutionStrategy`](@ref). `should_stop` also reports a step size below
`1e-12`.
"""
mutable struct LMMAES <: AbstractEvolutionStrategy
    const dim::Int
    const batch_size::Int
    const n_vectors::Int
    const sigma0::Float64
    const lower::Vector{Float64}
    const upper::Vector{Float64}
    # Learning rates of the step size path, of the transformation and of the vectors.
    const c_sigma::Float64
    const c_d::Vector{Float64}
    const c_c::Vector{Float64}
    mean::Vector{Float64}
    sigma::Float64
    path_sigma::Vector{Float64}
    # The vectors `M_j`, one in each column, and the number of generations so far.
    vectors::Matrix{Float64}
    generation::Int
    # The last batch: the normal draws `z`, the transformed draws `y` and the solutions `x`.
    z::Matrix{Float64}
    y::Matrix{Float64}
    x::Matrix{Float64}
    # Scratch space for an update: the recombination weights, the draws of the parents, and
    # the weighted means of their `z` and of their `y`.
    const weights::Vector{Float64}
    const chosen::Matrix{Float64}
    const z_mean::Vector{Float64}
    const y_mean::Vector{Float64}
end

function LMMAES(dim::Integer; sigma0::Real, batch_size::Integer=default_batch_size(dim),
                n_vectors::Integer=batch_size, lower=-Inf, upper=Inf)
    lo, hi = check_strategy_arguments(dim, sigma0, batch_size, lower, upper)
    2 * batch_size <= dim || throw(ArgumentError(lmmaes_batch_message(dim, batch_size)))
    1 <= n_vectors <= dim ||
        throw(ArgumentError("n_vectors must be between 1 and the dimension ($dim), got $n_vectors"))
    c_d = [1 / (1.5^(j - 1) * dim) for j in 1:n_vectors]
    c_c = [batch_size / (4.0^(j - 1) * dim) for j in 1:n_vectors]
    return LMMAES(Int(dim), Int(batch_size), Int(n_vectors), Float64(sigma0), lo, hi,
                  2 * batch_size / dim, c_d, c_c, zeros(dim), Float64(sigma0), zeros(dim),
                  zeros(dim, n_vectors), 0,
                  zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size),
                  zeros(batch_size), zeros(dim, batch_size), zeros(dim), zeros(dim))
end

# See #45: pyribs accepts any batch up to the dimension.
function lmmaes_batch_message(dim, batch_size)
    c_sigma = round(2 * batch_size / dim; digits=2)
    smaller = dim >= 4 ? "pass batch_size ≤ $(dim ÷ 2), or " : ""
    return "LMMAES needs a batch of at most half the dimension, but batch_size is $batch_size " *
           "and the dimension $dim: the learning rate of its step size, 2 batch_size / dim = " *
           "$c_sigma, would exceed 1, so the step size would stop adapting. LMMAES is meant for " *
           "a high dimension (at least 26 with the default batch); $(smaller)use CMAES or " *
           "SepCMAES. See the help of QualityDiversity.LMMAES."
end

function reset!(es::LMMAES, x0::AbstractVector{<:Real})
    check_start(x0, es.dim)
    es.mean .= x0
    es.sigma = es.sigma0
    fill!(es.path_sigma, 0)
    fill!(es.vectors, 0)
    es.generation = 0
    return es
end

function sample!(rng::AbstractRNG, es::LMMAES)
    active = min(es.generation, es.n_vectors)
    for k in 1:es.batch_size
        draws = 0
        while true
            for i in 1:es.dim
                es.z[i, k] = randn(rng)
                es.y[i, k] = es.z[i, k]
            end
            for j in 1:active
                # d ← (1 - c_d) d + c_d M_j (M_jᵀ d)
                projection = 0.0
                for i in 1:es.dim
                    projection += es.vectors[i, j] * es.y[i, k]
                end
                for i in 1:es.dim
                    es.y[i, k] = (1 - es.c_d[j]) * es.y[i, k] +
                                 es.c_d[j] * es.vectors[i, j] * projection
                end
            end
            for i in 1:es.dim
                es.x[i, k] = es.mean[i] + es.sigma * es.y[i, k]
            end
            column_inside(es.x, k, es.lower, es.upper) && break
            draws += 1
            if draws > RESAMPLE_LIMIT
                clip_step!(es.x, es.y, k, es.mean, es.sigma, es.lower, es.upper)
                break
            end
        end
    end
    return es.x
end

function update!(es::LMMAES, order::AbstractVector{<:Integer}, parents::Integer)
    length(order) == es.batch_size ||
        throw(ArgumentError("order has $(length(order)) entries but the batch has $(es.batch_size)"))
    0 <= parents <= es.batch_size ||
        throw(ArgumentError("parents must be between 0 and $(es.batch_size), got $parents"))
    es.generation += 1
    parents == 0 && return es
    weights = recombination_weights!(es.weights, parents)
    mueff = 1 / sum(abs2, weights)

    z_mean = mul!(es.z_mean, chosen_columns!(es.chosen, es.z, order, parents), weights)
    es.mean .+= es.sigma .* mul!(es.y_mean, chosen_columns!(es.chosen, es.y, order, parents), weights)

    c_sigma = es.c_sigma
    es.path_sigma .= (1 - c_sigma) .* es.path_sigma .+ sqrt(mueff * c_sigma * (2 - c_sigma)) .* z_mean
    for j in 1:es.n_vectors
        c = es.c_c[j]
        @views es.vectors[:, j] .= (1 - c) .* es.vectors[:, j] .+ sqrt(mueff * c * (2 - c)) .* z_mean
    end
    es.sigma *= exp(c_sigma / 2 * (sum(abs2, es.path_sigma) / es.dim - 1))
    return es
end

function should_stop(es::LMMAES, keys::AbstractVector)
    isfinite(es.sigma) && all(isfinite, es.mean) || return true
    es.sigma < 1e-12 && return true
    return flat_ranking(keys)
end
