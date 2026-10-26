"""
    SepCMAES(dim; sigma0, batch_size=default, lower=-Inf, upper=Inf)

Separable CMA-ES (Ros and Hansen, 2008, "A Simple Modification in CMA-ES Achieving
Linear Time and Space Complexity"), maximizing. It is [`CMAES`](@ref) with a diagonal
covariance matrix: it adapts one variance for each coordinate and no correlations,
which costs `O(dim)` for a generation instead of `O(dim²)` to `O(dim³)`, and it learns
faster in high dimension where there are too few samples to learn a full matrix. The
learning rates of the covariance update are those of the reference implementation
(pycma), which are larger than those of [`CMAES`](@ref) because there are only `dim`
variances to learn. The step size is adapted as in [`CMAES`](@ref).

The arguments, `sample!`, `update!` and `should_stop` are those of
[`AbstractEvolutionStrategy`](@ref).
"""
mutable struct SepCMAES <: AbstractEvolutionStrategy
    const dim::Int
    const batch_size::Int
    const sigma0::Float64
    const lower::Vector{Float64}
    const upper::Vector{Float64}
    mean::Vector{Float64}
    sigma::Float64
    # Diagonal of the covariance matrix.
    variances::Vector{Float64}
    path_c::Vector{Float64}
    path_sigma::Vector{Float64}
    count_eval::Int
    # The last batch: `y = (x - mean) / sigma` and `x`, one solution per column.
    y::Matrix{Float64}
    x::Matrix{Float64}
    # Scratch space for an update: the recombination weights, the steps of the parents and
    # the mean shift.
    const weights::Vector{Float64}
    const chosen::Matrix{Float64}
    const shift::Vector{Float64}
end

function SepCMAES(dim::Integer; sigma0::Real, batch_size::Integer=default_batch_size(dim),
                  lower=-Inf, upper=Inf)
    lo, hi = check_strategy_arguments(dim, sigma0, batch_size, lower, upper)
    return SepCMAES(Int(dim), Int(batch_size), Float64(sigma0), lo, hi,
                    zeros(dim), Float64(sigma0), ones(dim), zeros(dim), zeros(dim), 0,
                    zeros(dim, batch_size), zeros(dim, batch_size),
                    zeros(batch_size), zeros(dim, batch_size), zeros(dim))
end

function reset!(es::SepCMAES, x0::AbstractVector{<:Real})
    check_start(x0, es.dim)
    es.mean .= x0
    es.sigma = es.sigma0
    fill!(es.variances, 1)
    fill!(es.path_c, 0)
    fill!(es.path_sigma, 0)
    es.count_eval = 0
    return es
end

function sample!(rng::AbstractRNG, es::SepCMAES)
    for k in 1:es.batch_size
        draws = 0
        while true
            for i in 1:es.dim
                es.y[i, k] = sqrt(es.variances[i]) * randn(rng)
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

# Constants of the update for `parents` parents, or for the given weights. The learning
# rates of the covariance are those of pycma's `CMA_diagonal` mode, with its rank-mu
# offset of 1/4.
sep_strategy_constants(dim, parents::Integer) = sep_strategy_constants(dim, recombination_weights(parents))

function sep_strategy_constants(dim, weights::AbstractVector)
    mueff = 1 / sum(abs2, weights)
    cc = (1 + 1 / dim + mueff / dim) / (sqrt(dim) + 1 / dim + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 1 / (dim + 2 * sqrt(dim) + mueff / dim)
    cmu = min(1 - c1, (0.25 + mueff + 1 / mueff - 2) / (dim + 4 * sqrt(dim) + mueff / 2))
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    return weights, mueff, cc, cs, c1, cmu, damps
end

function update!(es::SepCMAES, order::AbstractVector{<:Integer}, parents::Integer)
    length(order) == es.batch_size ||
        throw(ArgumentError("order has $(length(order)) entries but the batch has $(es.batch_size)"))
    0 <= parents <= es.batch_size ||
        throw(ArgumentError("parents must be between 0 and $(es.batch_size), got $parents"))
    es.count_eval += es.batch_size
    parents == 0 && return es
    n = es.dim
    weights, mueff, cc, cs, c1, cmu, damps = sep_strategy_constants(n, recombination_weights!(es.weights, parents))
    chi_n = sqrt(n) * (1 - 1 / (4n) + 1 / (21n^2))

    ys = chosen_columns!(es.chosen, es.y, order, parents)
    yw = mul!(es.shift, ys, weights)
    es.mean .+= es.sigma .* yw

    es.path_sigma .= (1 - cs) .* es.path_sigma .+
                     sqrt(cs * (2 - cs) * mueff) .* yw ./ sqrt.(es.variances)
    norm_ps = norm(es.path_sigma)
    decay = 1 - (1 - cs)^(2 * es.count_eval / es.batch_size)
    hsig = norm_ps / sqrt(decay) / chi_n < 1.4 + 2 / (n + 1) ? 1.0 : 0.0
    es.path_c .= (1 - cc) .* es.path_c .+ hsig * sqrt(cc * (2 - cc) * mueff) .* yw

    delta = (1 - hsig) * cc * (2 - cc)
    for i in 1:n
        rank_mu = sum(weights[j] * ys[i, j]^2 for j in 1:parents)
        es.variances[i] = max((1 + c1 * delta - c1 - cmu) * es.variances[i] +
                              c1 * es.path_c[i]^2 + cmu * rank_mu, floatmin(Float64))
    end

    es.sigma *= exp(min(1.0, (cs / damps) * (norm_ps / chi_n - 1)))
    return es
end

function should_stop(es::SepCMAES, keys::AbstractVector)
    return covariance_converged(es.sigma, es.mean, es.variances, keys)
end
