"""
    GradientArborescenceEmitter(dim; n_descriptors, sigma0, optimizer=AdamAscent(),
                                batch_size, strategy, ranker, selection, restart, x0,
                                normalize=true)
    GradientArborescenceEmitter(x0::AbstractVector; n_descriptors, sigma0, ...)

Emitter of CMA-MEGA (Fontaine and Nikolaidis, 2021, "Differentiable Quality Diversity"),
for solutions that are real vectors of length `dim` and problems with `n_descriptors`
descriptor dimensions and gradients; see
[`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient).

The emitter keeps a solution point `θ`. Every round starts with the evaluation of `θ`
together with the gradients of its fitness, `∇f`, and of its descriptor components, `∇mⱼ`,
which are normalized to unit length (unless `normalize=false`), and `θ` is offered to the
archive. The candidates of the round are the branches of a *gradient arborescence*:
`θ + c₀∇f + Σⱼ cⱼ∇mⱼ`, with the coefficients `c` drawn from a search distribution of an
evolution strategy, CMA-ES by default. The coefficients may take either sign, as in the
paper and pyribs. After the round, the candidates are ranked as in a [`CMAEmitter`](@ref),
the distribution of the coefficients moves toward the best ones, and `θ` moves by the
`optimizer` along the step `Σᵢ wᵢ (c⁽ⁱ⁾ ⋅ ∇)` that recombines the selected candidates with
the weights of CMA-ES. A round without selected candidates leaves `θ` where it is.

- `sigma0`: the initial step size of the distribution of the coefficients, which starts
  at zero. It has no default; the paper uses values from `0.002` to `10`.
- `optimizer`: how `θ` moves, [`AdamAscent`](@ref) (the default) or
  [`GradientAscent`](@ref).
- `batch_size`, `strategy`, `ranker`, `selection`, `restart`: as for [`CMAEmitter`](@ref),
  with the strategy searching the `n_descriptors + 1` coefficients. The defaults are those
  of CMA-ME, as in the paper.
- `x0`: the first `θ`. Without it, and after every restart, `θ` is a random elite of the
  archive, and the distribution of the coefficients starts again from zero.

A round evaluates `batch_size + 1` solutions. The loop of [`step!`](@ref) gets the point
from [`ask_gradients`](@ref QualityDiversity.ask_gradients), evaluates it with
[`evaluate_gradient_batch`](@ref QualityDiversity.evaluate_gradient_batch), adds it to the
archive and hands the gradients to
[`tell_gradients!`](@ref QualityDiversity.tell_gradients!) before it calls `ask`; `tell!`
must follow that `ask`. The emitter has the properties `iterations` and `restarts`, as a
[`CMAEmitter`](@ref). Solutions have no bounds: a step along a gradient has no natural
way to respect them.
"""
mutable struct GradientArborescenceEmitter{R<:AbstractRanker,S<:SelectionRule,T<:RestartRule,
                                           E<:AbstractEvolutionStrategy,O<:GradientOptimizer} <: AbstractEmitter
    const es::E
    const optimizer::O
    const ranker::R
    const selection::S
    const restart_rule::T
    const x0::Union{Nothing,Vector{Float64}}
    const normalize::Bool
    # The solution point `θ` and the state of the optimizer.
    const ascent::AscentState
    # The gradients at `θ`, one row per function, normalized if `normalize` is set.
    const jacobian::Matrix{Float64}
    iterations::Int
    restarts::Int
    started::Bool
    restart_pending::Bool
    # Where the emitter is in a round: the point was asked for, its gradients told, and
    # the candidates asked for.
    gradients_asked::Bool
    gradients_told::Bool
    asked::Bool
end

function GradientArborescenceEmitter(dim::Integer; n_descriptors::Integer, sigma0::Real,
                                     optimizer::GradientOptimizer=AdamAscent(),
                                     batch_size::Union{Nothing,Integer}=nothing, strategy=CMAES,
                                     ranker::AbstractRanker=TwoStageImprovementRanker(),
                                     selection::SelectionRule=FilterSelection(),
                                     restart::RestartRule=NoImprovementRestart(),
                                     x0::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                     normalize::Bool=true)
    dim >= 1 || throw(ArgumentError("dim must be at least 1, got $dim"))
    n_descriptors >= 1 || throw(ArgumentError("n_descriptors must be at least 1, got $n_descriptors"))
    sizing = batch_size === nothing ? (;) : (; batch_size)
    es = strategy(n_descriptors + 1; sigma0, sizing..., lower=-Inf, upper=Inf)
    es isa AbstractEvolutionStrategy ||
        throw(ArgumentError("strategy must construct an AbstractEvolutionStrategy, got a $(typeof(es))"))
    x0 === nothing || check_start(x0, dim)
    start = x0 === nothing ? nothing : collect(Float64, x0)
    own_ranker = deepcopy(ranker)
    return GradientArborescenceEmitter{typeof(own_ranker),typeof(selection),typeof(restart),typeof(es),typeof(optimizer)}(
        es, optimizer, own_ranker, selection, restart, start, normalize, AscentState(dim),
        zeros(n_descriptors + 1, dim), 0, 0, false, false, false, false, false)
end

GradientArborescenceEmitter(x0::AbstractVector{<:Real}; kwargs...) =
    GradientArborescenceEmitter(length(x0); x0, kwargs...)

batch_size(emitter::GradientArborescenceEmitter) = emitter.es.batch_size

restarts(emitter::GradientArborescenceEmitter) = emitter.restarts

uses_gradients(::GradientArborescenceEmitter) = true

needs_start(emitter::GradientArborescenceEmitter) = !emitter.started || emitter.restart_pending

function start_search!(rng, emitter::GradientArborescenceEmitter, archive)
    dim = length(emitter.ascent.theta)
    x0 = emitter.x0
    theta = !emitter.started && x0 !== nothing ? x0 : elite_start(rng, archive, dim)
    reset_ascent!(emitter.ascent, theta)
    reset!(emitter.es, zeros(size(emitter.jacobian, 1)))
    reset_ranker!(rng, emitter.ranker, archive)
    emitter.started = true
    emitter.restart_pending = false
    return nothing
end

"""
    ask_gradients(rng, emitter::GradientArborescenceEmitter, archive) -> Vector{G}

The solution point `θ`, as a batch of one. The first call, and the first after a restart,
starts a new search from `x0` or a random elite of `archive`.
"""
function ask_gradients(rng::AbstractRNG, emitter::GradientArborescenceEmitter, archive::AbstractArchive{G}) where {G}
    emitter.asked &&
        throw(ArgumentError("tell! must follow the ask that returned the last batch before the next round starts"))
    needs_start(emitter) && start_search!(rng, emitter, archive)
    emitter.gradients_asked = true
    emitter.gradients_told = false
    point = Vector{G}(undef, 1)
    point[1] = convert(G, copy(emitter.ascent.theta))
    return point
end

"""
    tell_gradients!(emitter::GradientArborescenceEmitter, xs, evs, jacobians, results)

Keep the gradients at the solution point that the last
[`ask_gradients`](@ref QualityDiversity.ask_gradients) returned, normalized unless the
emitter was built with `normalize=false`.
"""
function tell_gradients!(emitter::GradientArborescenceEmitter, xs, evs, jacobians, results)
    emitter.gradients_asked ||
        throw(ArgumentError("tell_gradients! needs the gradients at the point that the last ask_gradients returned, but none was asked for"))
    length(jacobians) == 1 ||
        throw(ArgumentError("expected the gradients of one point but got $(length(jacobians))"))
    jacobian = only(jacobians)
    size(jacobian) == size(emitter.jacobian) ||
        throw(ArgumentError("the Jacobian is $(join(size(jacobian), "×")) but the emitter expects $(join(size(emitter.jacobian), "×")): one row for the fitness and one per descriptor dimension, one column per entry of the solution"))
    copyto!(emitter.jacobian, jacobian)
    emitter.normalize && normalize_rows!(emitter.jacobian)
    emitter.gradients_asked = false
    emitter.gradients_told = true
    return nothing
end

"""
    ask(rng, emitter::GradientArborescenceEmitter, archive) -> Vector{G}

The branches of the gradient arborescence at the solution point, one per coefficient
vector drawn from the search distribution.
"""
function ask(rng::AbstractRNG, emitter::GradientArborescenceEmitter, archive::AbstractArchive{G}) where {G}
    emitter.gradients_told && !emitter.asked ||
        throw(ArgumentError("a GradientArborescenceEmitter asks for its candidates after ask_gradients and tell_gradients!, once per round"))
    es = emitter.es
    sample!(rng, es)
    emitter.asked = true
    theta = emitter.ascent.theta
    gradients = transpose(emitter.jacobian)
    xs = Vector{G}(undef, es.batch_size)
    for k in 1:es.batch_size
        branch = copy(theta)
        mul!(branch, gradients, view(es.x, :, k), true, true)
        xs[k] = convert(G, branch)
    end
    return xs
end

"""
    tell!(emitter::GradientArborescenceEmitter, xs, evs, results)

Rank the batch that the last `ask` returned, move the distribution of the coefficients and
the solution point, and schedule a restart if the restart rule or the evolution strategy
asks for one.
"""
function tell!(emitter::GradientArborescenceEmitter, xs, evs, results)
    emitter.asked ||
        throw(ArgumentError("tell! needs the results of the batch that the last ask returned, but nothing was asked since the last tell!"))
    es = emitter.es
    (length(xs) == es.batch_size && length(evs) == es.batch_size && length(results) == es.batch_size) ||
        throw(ArgumentError("expected $(es.batch_size) solutions, evaluations and results but got $(length(xs)), $(length(evs)) and $(length(results))"))
    emitter.asked = false
    emitter.gradients_told = false
    emitter.iterations += 1
    added = count(r -> r.status != Rejected, results)
    order, keys = rank(emitter.ranker, evs, results)
    parents = parent_count(emitter.selection, added, es.batch_size)
    # The step is taken from the coefficients of this batch, before the strategy moves on.
    if parents > 0
        weights = recombination_weights(parents)
        coefficients = zeros(size(es.x, 1))
        for (w, k) in zip(weights, view(order, 1:parents))
            coefficients .+= w .* view(es.x, :, k)
        end
        ascend!(emitter.ascent, emitter.optimizer, transpose(emitter.jacobian) * coefficients)
    end
    update!(es, order, parents)
    if should_stop(es, keys) || restart_due(emitter.restart_rule, emitter.iterations, added)
        emitter.restart_pending = true
        emitter.restarts += 1
    end
    return nothing
end
