"""
    CMAVariant

Supertype of the variants of the update of [`CMAES`](@ref): [`TextbookCMA`](@ref), the
default, and [`PyribsCMA`](@ref).
"""
abstract type CMAVariant end

"""
    TextbookCMA()

The update of Hansen (2016), "The CMA Evolution Strategy: A Tutorial". This is the
default of [`CMAES`](@ref).
"""
struct TextbookCMA <: CMAVariant end

"""
    PyribsCMA()

The update that pyribs 0.12.0's `CMAEvolutionStrategy` computes, which departs from the
tutorial in five places:

1. The rank-one term `pc pcᵀ` has weight `c1²`: pyribs multiplies it by `c1` twice. The
   term is then much weaker, so a search stays rounder for longer.
2. The evolution path `pc` accumulates the mean shift `m′ - m` itself, not divided by
   `σ`, so the rank-one term scales with `σ²`: it vanishes for a small step size and
   dominates the covariance for a large one.
3. The step size changes by `exp(min(1, c_σ/d_σ (‖p_σ‖²/n - 1)/2))`, not
   `exp(min(1, c_σ/d_σ (‖p_σ‖/χₙ - 1)))`.
4. The path `pc` stalls unless `‖p_σ‖²/n / (1 - (1 - c_σ)^{2g}) < 2 + 4/(n + 1)`, with
   `g` the generation, instead of `‖p_σ‖ / √(1 - (1 - c_σ)^{2g}) / χₙ < 1.4 + 2/(n + 1)`.
5. The eigendecomposition is recomputed when more than `λ / (2 n (c1 + c_μ))`
   evaluations have passed, with `c1` and `c_μ` those of `λ ÷ 2` parents, also after a
   generation without parents; the matrix and its inverse square root are made
   symmetric by the larger of each pair of entries, and the eigenvalues are taken in
   absolute value.

Use it through [`PyribsCMAES`](@ref) to reproduce results produced with pyribs.
"""
struct PyribsCMA <: CMAVariant end

rank_one_rate(::TextbookCMA, c1) = c1
rank_one_rate(::PyribsCMA, c1) = c1^2

# Factor of the mean shift `yw`, in units of sigma, that enters the path `pc`.
path_shift_scale(::TextbookCMA, sigma) = 1.0
path_shift_scale(::PyribsCMA, sigma) = sigma

# Whether the path `pc` keeps growing, from the norm of `p_σ` and `decay = 1 - (1 - c_σ)^{2g}`.
path_grows(::TextbookCMA, norm_ps, decay, chi_n, n) = norm_ps / sqrt(decay) / chi_n < 1.4 + 2 / (n + 1)
path_grows(::PyribsCMA, norm_ps, decay, chi_n, n) = norm_ps^2 / n / decay < 2 + 4 / (n + 1)

step_size_factor(::TextbookCMA, cs, damps, norm_ps, chi_n, n) = exp(min(1.0, (cs / damps) * (norm_ps / chi_n - 1)))
step_size_factor(::PyribsCMA, cs, damps, norm_ps, chi_n, n) = exp(min(1.0, (cs / damps) * (norm_ps^2 / n - 1) / 2))

symmetric_entry(::TextbookCMA, a, b) = (a + b) / 2
symmetric_entry(::PyribsCMA, a, b) = max(a, b)

stored_eigenvalue(::TextbookCMA, value) = max(value, floatmin(Float64))
stored_eigenvalue(::PyribsCMA, value) = max(abs(value), floatmin(Float64))

"""
    CMAES(dim; sigma0, batch_size=default, lower=-Inf, upper=Inf, variant=TextbookCMA())

Covariance matrix adaptation evolution strategy (Hansen and Ostermeier, 2001;
Hansen, 2016, "The CMA Evolution Strategy: A Tutorial"), maximizing. It keeps a
Gaussian search distribution with mean `m`, step size `σ` and covariance `C`, and
is the default optimizer inside a [`CMAEmitter`](@ref). `variant`, a
[`CMAVariant`](@ref), chooses the update: the tutorial's, or pyribs's through
[`PyribsCMAES`](@ref).

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
mutable struct CMAES{V<:CMAVariant} <: AbstractEvolutionStrategy
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
    const variant::V
end

function CMAES(dim::Integer; sigma0::Real, batch_size::Integer=default_batch_size(dim),
               lower=-Inf, upper=Inf, variant::CMAVariant=TextbookCMA())
    lo, hi = check_strategy_arguments(dim, sigma0, batch_size, lower, upper)
    es = CMAES(Int(dim), Int(batch_size), Float64(sigma0), lo, hi,
               zeros(dim), Float64(sigma0), Matrix{Float64}(I, dim, dim),
               Matrix{Float64}(I, dim, dim), ones(dim), Matrix{Float64}(I, dim, dim),
               zeros(dim), zeros(dim), 0, 0,
               zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim), zeros(dim),
               zeros(batch_size), zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim),
               zeros(dim), zeros(dim, dim), variant)
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
    parents == 0 && return skip_parents!(es.variant, es)
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
    hsig = path_grows(es.variant, norm_ps, decay, chi_n, n) ? 1.0 : 0.0
    shift_scale = path_shift_scale(es.variant, es.sigma)
    es.path_c .= (1 - cc) .* es.path_c .+ (hsig * sqrt(cc * (2 - cc) * mueff) * shift_scale) .* yw

    # Covariance: rank-one update from the path and rank-mu update from the parents.
    delta = (1 - hsig) * cc * (2 - cc)
    es.cov .*= 1 + c1 * delta - c1 - cmu
    rate = rank_one_rate(es.variant, c1)
    for j in 1:n, i in 1:n
        es.cov[i, j] += rate * (es.path_c[i] * es.path_c[j])
    end
    weighted = view(es.weighted, :, 1:parents)
    weighted .= ys .* weights'
    mul!(es.cov, weighted, ys', cmu, 1.0)

    es.sigma *= step_size_factor(es.variant, cs, damps, norm_ps, chi_n, n)

    eigen_due(es.variant, es, c1, cmu) && update_eigensystem!(es)
    return es
end

# Whether the eigendecomposition is to be recomputed; it costs O(n^3), so it is only
# recomputed now and then.
eigen_due(::TextbookCMA, es, c1, cmu) = es.count_eval - es.eigen_eval > es.batch_size / ((c1 + cmu) * es.dim * 10)

function eigen_due(::PyribsCMA, es, _, _)
    # pyribs's interval uses the constants of `batch_size ÷ 2` parents, whatever the
    # number of parents of the generation.
    _, _, _, _, c1, cmu, _ = strategy_constants(es.dim, recombination_weights!(es.weights, es.batch_size ÷ 2))
    return es.count_eval - es.eigen_eval > es.batch_size / (2 * es.dim * (c1 + cmu))
end

# A generation without parents leaves the distribution as it is; pyribs still recomputes
# a decomposition that is due.
skip_parents!(::TextbookCMA, es) = es

function skip_parents!(variant::PyribsCMA, es)
    eigen_due(variant, es, nothing, nothing) && update_eigensystem!(es)
    return es
end

function update_eigensystem!(es::CMAES)
    n = es.dim
    for j in 1:n, i in 1:j-1
        es.cov[i, j] = es.cov[j, i] = symmetric_entry(es.variant, es.cov[i, j], es.cov[j, i])
    end
    # `eigen!` overwrites its argument, so it gets a copy in a buffer that is free until the
    # decomposition is read.
    es.scaled_basis .= es.cov
    decomposition = eigen!(Symmetric(es.scaled_basis))
    if all(isfinite, decomposition.values) && all(isfinite, decomposition.vectors)
        for i in 1:n
            es.eigenvalues[i] = stored_eigenvalue(es.variant, decomposition.values[i])
        end
        es.basis .= decomposition.vectors
        es.scaled_basis .= es.basis .* (1 ./ sqrt.(es.eigenvalues))'
        mul!(es.inv_sqrt, es.scaled_basis, es.basis')
        for j in 1:n, i in 1:j-1
            es.inv_sqrt[i, j] = es.inv_sqrt[j, i] = symmetric_entry(es.variant, es.inv_sqrt[i, j], es.inv_sqrt[j, i])
        end
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

[`CMAES`](@ref) with the update of pyribs 0.12.0, [`PyribsCMA`](@ref): given the same
batches and rankings, its state equals that of pyribs's `CMAEvolutionStrategy` up to
rounding. It exists to reproduce results that were produced with pyribs, such as those of
the CMA-MAE paper, where its weak rank-one term makes a search explore with rounder
distributions. Pass it as `strategy=QualityDiversity.PyribsCMAES` to a [`CMAEmitter`](@ref),
[`CMAME`](@ref) or [`CMAMAE`](@ref).
"""
PyribsCMAES(dim::Integer; kwargs...) = CMAES(dim; kwargs..., variant=PyribsCMA())
