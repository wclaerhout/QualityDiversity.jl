"""
    OpenAIES(dim; sigma0, batch_size=default, learning_rate=0.001, beta1=0.9,
             beta2=0.999, epsilon=1e-8, lower=-Inf, upper=Inf)

Natural evolution strategy in the form of Salimans et al. (2017, "Evolution
Strategies as a Scalable Alternative to Reinforcement Learning"), maximizing. The
search distribution is an isotropic Gaussian of fixed standard deviation `sigma0`
around a point `θ`. A batch consists of `batch_size / 2` perturbations of `θ` and
their mirror images, `θ ± σ₀ ε`. Each solution gets its centered rank `r` in
`[-1/2, 1/2]`, the gradient estimate is `Σ ε (r₊ - r₋) / (batch_size/2 σ₀)`, and the
Adam optimizer (Kingma and Ba, 2015) with the given hyperparameters moves `θ` along
it.

`batch_size` is even, and by default the default of [`CMAES`](@ref) rounded up to an
even number. With bounds, a pair of mirrored solutions is drawn again if either lies
outside, and moved inside the bounds after $(RESAMPLE_LIMIT) draws, and `θ` is moved
inside the bounds after every step, so that the pairs keep landing inside. Unlike for the covariance matrix strategies, the number of `parents` of
`update!` has no effect: the ranks of the whole batch make up the gradient. A
restart resets `θ` and the state of Adam. `should_stop` also reports a last step of
`θ` below `1e-9` times its norm.

The arguments, `sample!`, `update!` and `should_stop` are those of
[`AbstractEvolutionStrategy`](@ref).
"""
mutable struct OpenAIES <: AbstractEvolutionStrategy
    const dim::Int
    const batch_size::Int
    const sigma0::Float64
    const lower::Vector{Float64}
    const upper::Vector{Float64}
    const learning_rate::Float64
    const beta1::Float64
    const beta2::Float64
    const epsilon::Float64
    theta::Vector{Float64}
    # Adam: first and second moment of the gradient and the number of steps.
    moment1::Vector{Float64}
    moment2::Vector{Float64}
    steps::Int
    # Norm of the last step of `theta` relative to the norm of `theta`.
    update_ratio::Float64
    noise::Matrix{Float64}
    x::Matrix{Float64}
    # Scratch space for an update: the centered ranks, the gradient estimate and the step.
    const ranks::Vector{Float64}
    const gradient::Vector{Float64}
    const step::Vector{Float64}
end

function OpenAIES(dim::Integer; sigma0::Real,
                  batch_size::Integer=2 * cld(default_batch_size(dim), 2),
                  learning_rate::Real=0.001, beta1::Real=0.9, beta2::Real=0.999,
                  epsilon::Real=1e-8, lower=-Inf, upper=Inf)
    lo, hi = check_strategy_arguments(dim, sigma0, batch_size, lower, upper)
    iseven(batch_size) ||
        throw(ArgumentError("batch_size must be even because the solutions are mirrored, got $batch_size"))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    (0 <= beta1 < 1 && 0 <= beta2 < 1) ||
        throw(ArgumentError("beta1 and beta2 must be in [0, 1), got $beta1 and $beta2"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $epsilon"))
    return OpenAIES(Int(dim), Int(batch_size), Float64(sigma0), lo, hi, Float64(learning_rate),
                    Float64(beta1), Float64(beta2), Float64(epsilon),
                    zeros(dim), zeros(dim), zeros(dim), 0, Inf,
                    zeros(dim, batch_size), zeros(dim, batch_size),
                    zeros(batch_size), zeros(dim), zeros(dim))
end

function reset!(es::OpenAIES, x0::AbstractVector{<:Real})
    check_start(x0, es.dim)
    es.theta .= x0
    fill!(es.moment1, 0)
    fill!(es.moment2, 0)
    es.steps = 0
    es.update_ratio = Inf
    return es
end

function sample!(rng::AbstractRNG, es::OpenAIES)
    half = es.batch_size ÷ 2
    for k in 1:half
        draws = 0
        while true
            for i in 1:es.dim
                e = randn(rng)
                es.noise[i, k] = e
                es.noise[i, k+half] = -e
                es.x[i, k] = es.theta[i] + es.sigma0 * e
                es.x[i, k+half] = es.theta[i] - es.sigma0 * e
            end
            column_inside(es.x, k, es.lower, es.upper) &&
                column_inside(es.x, k + half, es.lower, es.upper) && break
            draws += 1
            if draws > RESAMPLE_LIMIT
                # The noise stays as drawn: the gradient estimate follows the mirrored pair.
                clip_column!(es.x, k, es.lower, es.upper)
                clip_column!(es.x, k + half, es.lower, es.upper)
                break
            end
        end
    end
    return es.x
end

function update!(es::OpenAIES, order::AbstractVector{<:Integer}, parents::Integer)
    length(order) == es.batch_size ||
        throw(ArgumentError("order has $(length(order)) entries but the batch has $(es.batch_size)"))
    0 <= parents <= es.batch_size ||
        throw(ArgumentError("parents must be between 0 and $(es.batch_size), got $parents"))
    n, half = es.batch_size, es.batch_size ÷ 2
    # Centered rank of every solution: 1/2 for the best, -1/2 for the worst.
    ranks = es.ranks
    for (position, k) in enumerate(order)
        ranks[k] = (n - position) / (n - 1) - 0.5
    end
    gradient = fill!(es.gradient, 0.0)
    for k in 1:half
        @views gradient .+= es.noise[:, k] .* (ranks[k] - ranks[k+half])
    end
    gradient ./= half * es.sigma0

    es.steps += 1
    es.moment1 .= es.beta1 .* es.moment1 .+ (1 - es.beta1) .* gradient
    es.moment2 .= es.beta2 .* es.moment2 .+ (1 - es.beta2) .* gradient .^ 2
    scale = es.learning_rate * sqrt(1 - es.beta2^es.steps) / (1 - es.beta1^es.steps)
    step = es.step
    step .= scale .* es.moment1 ./ (sqrt.(es.moment2) .+ es.epsilon)
    for i in 1:es.dim
        es.theta[i] = clamp(es.theta[i] + step[i], es.lower[i], es.upper[i])
    end
    es.update_ratio = norm(step) / max(norm(es.theta), floatmin(Float64))
    return es
end

function should_stop(es::OpenAIES, keys::AbstractVector)
    all(isfinite, es.theta) || return true
    es.update_ratio < 1e-9 && return true
    return flat_ranking(keys)
end
