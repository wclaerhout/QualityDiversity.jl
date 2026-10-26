"""
    RankOneRule

Supertype of the rules that set the weight of the rank-one term `c1 pc pcᵀ` in the
covariance update of [`CMAES`](@ref): [`TextbookRankOne`](@ref) and
[`PyribsRankOne`](@ref).
"""
abstract type RankOneRule end

"""
    TextbookRankOne()

The rank-one term of the covariance update with weight `c1`, as in Hansen (2016) and
Loshchilov et al. (2019). This is the default of [`CMAES`](@ref).
"""
struct TextbookRankOne <: RankOneRule end

"""
    PyribsRankOne()

The rank-one term with weight `c1²`, which is what pyribs computes: it multiplies the
term by `c1` twice. The term is then much weaker, and the covariance matrix adapts
almost only through the rank-mu update, so a search stays rounder for longer. Use it
through [`PyribsCMAES`](@ref) to compare with results produced by pyribs.
"""
struct PyribsRankOne <: RankOneRule end

rank_one_rate(::TextbookRankOne, c1) = c1
rank_one_rate(::PyribsRankOne, c1) = c1^2

"""
    CMAES(dim; sigma0, batch_size=default, lower=-Inf, upper=Inf, rank_one=TextbookRankOne())

Covariance matrix adaptation evolution strategy (Hansen and Ostermeier, 2001;
Hansen, 2016, "The CMA Evolution Strategy: A Tutorial"), maximizing. It keeps a
Gaussian search distribution with mean `m`, step size `σ` and covariance `C`, and
is the default optimizer inside a [`CMAEmitter`](@ref). `rank_one`, a
[`RankOneRule`](@ref), sets the weight of the rank-one covariance term;
[`PyribsCMAES`](@ref) is the variant that reproduces pyribs.

A generation has two calls. `sample!(rng, es)` draws `batch_size` solutions,
`x = m + σ B D z` with `z` standard normal and `C = B D² Bᵀ`, and leaves them in
`es.x` as the columns of a matrix. After the caller has ranked them, `update!(es,
order, parents)` moves the distribution toward the `parents` best solutions, the
first entries of `order`. The number of parents may differ from one generation to
the next, and none is allowed; the recombination weights `log(parents + 1/2) -
log(i)` are normalized each time.

`batch_size` defaults to `4 + floor(3 log(dim))`. With `lower` and `upper`, scalars
or vectors, a sample is drawn again until it lies inside; after $(RESAMPLE_LIMIT) draws it
moves to the nearest point inside the bounds, and the update uses the step to that point.

`should_stop(es, keys)` reports the termination criteria of CMA-ES: a covariance
matrix with a condition number above `1e14`, a distribution that has shrunk below
`1e-11`, a non-finite state, or `keys`, the ranking keys from best to worst, being
flat: the best and the worst are within `1e-12` of each other. `reset!(es, x0)`
restarts the search at `x0`.
"""
mutable struct CMAES{R<:RankOneRule} <: AbstractEvolutionStrategy
    const dim::Int
    const batch_size::Int
    const sigma0::Float64
    const lower::Vector{Float64}
    const upper::Vector{Float64}
    mean::Vector{Float64}
    sigma::Float64
    cov::Matrix{Float64}
    # Eigendecomposition of `cov`: `cov = basis * Diagonal(eigenvalues) * basis'`,
    # and `inv_sqrt = basis * Diagonal(eigenvalues .^ -1/2) * basis'`.
    basis::Matrix{Float64}
    eigenvalues::Vector{Float64}
    inv_sqrt::Matrix{Float64}
    path_c::Vector{Float64}
    path_sigma::Vector{Float64}
    # Number of solutions drawn so far, and when the eigendecomposition was updated.
    count_eval::Int
    eigen_eval::Int
    # The last batch: `y = (x - mean) / sigma` and `x`, one solution per column.
    y::Matrix{Float64}
    x::Matrix{Float64}
    # Scratch space for a draw.
    z::Vector{Float64}
    step::Vector{Float64}
    # Scratch space for an update: the recombination weights, the steps of the parents and
    # those steps times their weights, the mean shift and its whitened form, and
    # `basis * Diagonal(eigenvalues .^ -1/2)`.
    const weights::Vector{Float64}
    const chosen::Matrix{Float64}
    const weighted::Matrix{Float64}
    const shift::Vector{Float64}
    const whitened::Vector{Float64}
    const scaled_basis::Matrix{Float64}
    const rank_one::R
end

function CMAES(dim::Integer; sigma0::Real, batch_size::Integer=default_batch_size(dim),
               lower=-Inf, upper=Inf, rank_one::RankOneRule=TextbookRankOne())
    lo, hi = check_strategy_arguments(dim, sigma0, batch_size, lower, upper)
    es = CMAES(Int(dim), Int(batch_size), Float64(sigma0), lo, hi,
               zeros(dim), Float64(sigma0), Matrix{Float64}(I, dim, dim),
               Matrix{Float64}(I, dim, dim), ones(dim), Matrix{Float64}(I, dim, dim),
               zeros(dim), zeros(dim), 0, 0,
               zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim), zeros(dim),
               zeros(batch_size), zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim),
               zeros(dim), zeros(dim, dim), rank_one)
    return es
end

"""
    reset!(es::CMAES, x0)

Restart the search at `x0`: the mean becomes `x0`, the step size `sigma0`, and the
covariance matrix the identity.
"""
function reset!(es::CMAES, x0::AbstractVector{<:Real})
    check_start(x0, es.dim)
    es.mean .= x0
    es.sigma = es.sigma0
    for matrix in (es.cov, es.basis, es.inv_sqrt)
        fill!(matrix, 0)
        for i in 1:es.dim
            matrix[i, i] = 1
        end
    end
    es.eigenvalues .= 1
    es.path_c .= 0
    es.path_sigma .= 0
    es.count_eval = 0
    es.eigen_eval = 0
    return es
end

"""
    sample!(rng, es::CMAES)

Draw a batch from the search distribution into `es.x`, one solution per column,
and return `es.x`.
"""
function sample!(rng::AbstractRNG, es::CMAES)
    for k in 1:es.batch_size
        draws = 0
        while true
            randn!(rng, es.z)
            for i in 1:es.dim
                es.z[i] *= sqrt(es.eigenvalues[i])
            end
            mul!(es.step, es.basis, es.z)
            inside = true
            for i in 1:es.dim
                xi = es.mean[i] + es.sigma * es.step[i]
                es.x[i, k] = xi
                inside &= es.lower[i] <= xi <= es.upper[i]
            end
            if inside
                es.y[:, k] .= es.step
                break
            end
            draws += 1
            if draws > RESAMPLE_LIMIT
                clip_step!(es.x, es.y, k, es.mean, es.sigma, es.lower, es.upper)
                break
            end
        end
    end
    return es.x
end

# Recombination weights for `parents` parents, or the given weights, and the constants
# of the update.
strategy_constants(dim, parents::Integer) = strategy_constants(dim, recombination_weights(parents))

function strategy_constants(dim, weights::AbstractVector)
    mueff = 1 / sum(abs2, weights)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3)^2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)^2 + mueff))
    damps = 1 + 2 * max(0, sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    return weights, mueff, cc, cs, c1, cmu, damps
end

"""
    update!(es::CMAES, order, parents)

Adapt the distribution to the last batch. `order` ranks the solutions of the batch
from best to worst, and the first `parents` of them are recombined. With no
parent only the count of evaluations advances.
"""
function update!(es::CMAES, order::AbstractVector{<:Integer}, parents::Integer)
    length(order) == es.batch_size ||
        throw(ArgumentError("order has $(length(order)) entries but the batch has $(es.batch_size)"))
    0 <= parents <= es.batch_size ||
        throw(ArgumentError("parents must be between 0 and $(es.batch_size), got $parents"))
    es.count_eval += es.batch_size
    parents == 0 && return es
    n = es.dim
    weights, mueff, cc, cs, c1, cmu, damps = strategy_constants(n, recombination_weights!(es.weights, parents))
    chi_n = sqrt(n) * (1 - 1 / (4n) + 1 / (21n^2))

    ys = chosen_columns!(es.chosen, es.y, order, parents)
    yw = mul!(es.shift, ys, weights)                     # mean shift in units of sigma
    es.mean .+= es.sigma .* yw

    # Evolution path of the step size, and the stall indicator for the path of C.
    mul!(es.whitened, es.inv_sqrt, yw)
    es.path_sigma .= (1 - cs) .* es.path_sigma .+ sqrt(cs * (2 - cs) * mueff) .* es.whitened
    norm_ps = norm(es.path_sigma)
    decay = 1 - (1 - cs)^(2 * es.count_eval / es.batch_size)
    hsig = norm_ps / sqrt(decay) / chi_n < 1.4 + 2 / (n + 1) ? 1.0 : 0.0
    es.path_c .= (1 - cc) .* es.path_c .+ hsig * sqrt(cc * (2 - cc) * mueff) .* yw

    # Covariance: rank-one update from the path and rank-mu update from the parents.
    delta = (1 - hsig) * cc * (2 - cc)
    es.cov .*= 1 + c1 * delta - c1 - cmu
    rate = rank_one_rate(es.rank_one, c1)
    for j in 1:n, i in 1:n
        es.cov[i, j] += rate * (es.path_c[i] * es.path_c[j])
    end
    weighted = view(es.weighted, :, 1:parents)
    weighted .= ys .* weights'
    mul!(es.cov, weighted, ys', cmu, 1.0)

    es.sigma *= exp(min(1.0, (cs / damps) * (norm_ps / chi_n - 1)))

    # The decomposition is only recomputed now and then; it costs O(n^3).
    if es.count_eval - es.eigen_eval > es.batch_size / ((c1 + cmu) * n * 10)
        update_eigensystem!(es)
    end
    return es
end

function update_eigensystem!(es::CMAES)
    n = es.dim
    for j in 1:n, i in 1:j-1
        es.cov[i, j] = es.cov[j, i] = (es.cov[i, j] + es.cov[j, i]) / 2
    end
    # `eigen!` overwrites its argument, so it gets a copy in a buffer that is free until the
    # decomposition is read.
    es.scaled_basis .= es.cov
    decomposition = eigen!(Symmetric(es.scaled_basis))
    if all(isfinite, decomposition.values) && all(isfinite, decomposition.vectors)
        es.eigenvalues .= max.(decomposition.values, floatmin(Float64))
        es.basis .= decomposition.vectors
        es.scaled_basis .= es.basis .* (1 ./ sqrt.(es.eigenvalues))'
        mul!(es.inv_sqrt, es.scaled_basis, es.basis')
    else
        es.eigenvalues .= NaN
    end
    es.eigen_eval = es.count_eval
    return es
end

"""
    should_stop(es::CMAES, keys) -> Bool

Whether the search has converged or broken down; see [`CMAES`](@ref). `keys` are the
ranking keys of the last batch from best to worst.
"""
function should_stop(es::CMAES, keys::AbstractVector)
    return covariance_converged(es.sigma, es.mean, es.eigenvalues, keys)
end

"""
    PyribsCMAES(dim; sigma0, batch_size=default, lower=-Inf, upper=Inf)

[`CMAES`](@ref) with the rank-one covariance weight of pyribs, `c1²` instead of `c1`;
see [`PyribsRankOne`](@ref). It exists to reproduce results that were produced with
pyribs, such as those of the CMA-MAE paper, where it makes a search explore with rounder
distributions. Pass it as `strategy=QualityDiversity.PyribsCMAES` to a [`CMAEmitter`](@ref),
[`CMAME`](@ref) or [`CMAMAE`](@ref).
"""
PyribsCMAES(dim::Integer; kwargs...) = CMAES(dim; kwargs..., rank_one=PyribsRankOne())
