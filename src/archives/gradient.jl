"""
    GradientArchive(archive)

Archive for OMG-MEGA (Fontaine and Nikolaidis, 2021): a cell archive that also keeps, next
to every elite, the gradients of its fitness and descriptor, which the
[`GradientOperatorEmitter`](@ref) steps along. `archive` decides everything else: which
cell a descriptor belongs to and which solution a cell keeps. It is a
[`GridArchive`](@ref), a [`CVTArchive`](@ref) or a [`ThresholdArchive`](@ref) around one,
with `SVector` descriptors; it must be empty, and the new archive uses it without copying.

Every solution offered to this archive comes with its Jacobian, an `(N + 1) × n` matrix for
a solution of `n` entries and `N` descriptor dimensions, as
[`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient) returns it. A run with this
archive therefore evaluates every solution with its gradients, the initial ones included,
and counts each as one evaluation. The paper does the same; pyribs instead evaluates the
gradients of a parent again when it is drawn, and QDax keeps them as this archive does.

[`add!`](@ref QualityDiversity.add!) and [`add_batch!`](@ref QualityDiversity.add_batch!)
take the Jacobians as a fourth argument, `add!(archive, x, ev, jacobian)` and
`add_batch!(archive, xs, evs, jacobians)`, and throw without them; the results are those of
the wrapped archive. A cell keeps the Jacobian of the solution that the wrapped archive
keeps: of the solutions of a batch that enter a cell, the one with the highest fitness (the
first on a tie). The memory is that of `N + 1` vectors as long as a solution, for every
occupied cell.

Cells, sampling, `elites`, `length`, `capacity`, `cell_index`, the metrics and
[`result_archive`](@ref QualityDiversity.result_archive) are those of the wrapped archive.
"""
struct GradientArchive{G,F,D,P,A<:AbstractArchive{G,F,D,P}} <: AbstractArchive{G,F,D,P}
    inner::A
    # The Jacobian of the elite of each cell; unassigned for a cell that never held one.
    gradients::Vector{Matrix{Float64}}

    function GradientArchive{G,F,D,P,A}(inner::A) where {G,F,D,P,A<:AbstractArchive{G,F,D,P}}
        length(inner) == 0 ||
            throw(ArgumentError("the archive to wrap must be empty but holds $(length(inner)) solutions"))
        return new{G,F,D,P,A}(inner, Vector{Matrix{Float64}}(undef, capacity(inner)))
    end
end

GradientArchive(inner::CellArchive{G,F,SVector{N,Float64},P}) where {G,F<:Real,N,P} =
    GradientArchive{G,F,SVector{N,Float64},P,typeof(inner)}(inner)

function GradientArchive(::SlidingBoundariesArchive{G,F,N}) where {G,F<:Real,N}
    throw(ArgumentError("a GradientArchive keeps the gradients of the elite of every cell, but a SlidingBoundariesArchive moves its elites when it remaps; wrap a GridArchive or a CVTArchive"))
end

GradientArchive(inner::ThresholdArchive{G,F,SVector{N,Float64},P}) where {G,F,N,P} =
    GradientArchive{G,F,SVector{N,Float64},P,typeof(inner)}(inner)

function GradientArchive(archive::AbstractArchive)
    throw(ArgumentError("a GradientArchive wraps a cell archive with SVector descriptors, or a ThresholdArchive around one, not a $(nameof(typeof(archive)))"))
end

# The cell archive that holds the elites of the archive that a `GradientArchive` wraps.
gradient_storage(archive::CellArchive) = archive
gradient_storage(archive::ThresholdArchive) = archive.inner

Base.length(archive::GradientArchive) = length(archive.inner)

capacity(archive::GradientArchive) = capacity(archive.inner)

Base.empty!(archive::GradientArchive) = (empty!(archive.inner); archive)

result_archive(archive::GradientArchive) = result_archive(archive.inner)

cell_index(archive::GradientArchive, d) = cell_index(archive.inner, d)

sample_elites(rng::AbstractRNG, archive::GradientArchive, n::Integer) = sample_elites(rng, archive.inner, n)

elites(archive::GradientArchive) = elites(archive.inner)

descriptor_bounds(archive::GradientArchive) = descriptor_bounds(archive.inner)

qd_score(archive::GradientArchive; offset=0) = qd_score(archive.inner; offset)

max_fitness(archive::GradientArchive) = max_fitness(archive.inner)

mean_fitness(archive::GradientArchive) = mean_fitness(archive.inner)

function add!(archive::GradientArchive, x, ev::Evaluation)
    throw(ArgumentError("a GradientArchive keeps the gradients of its elites: call add!(archive, x, ev, jacobian)"))
end

function add_batch!(archive::GradientArchive, xs::AbstractVector, evs::AbstractVector)
    throw(ArgumentError("a GradientArchive keeps the gradients of its elites: call add_batch!(archive, xs, evs, jacobians)"))
end

"""
    add!(archive::GradientArchive, x, ev::Evaluation, jacobian) -> AddResult

Offer `x` to the wrapped archive and keep `jacobian` if `x` becomes the elite of its cell.
"""
function add!(archive::GradientArchive, x, ev::Evaluation, jacobian::AbstractMatrix)
    return only(add_batch!(archive, [x], [ev], [jacobian]))
end

"""
    add_batch!(archive::GradientArchive, xs, evs, jacobians) -> Vector{AddResult}

Offer the solutions `xs` to the wrapped archive, and keep the Jacobian of each that becomes
the elite of its cell. A Jacobian of the wrong size throws before the archive changes.
"""
function add_batch!(archive::GradientArchive{G,F,SVector{N,Float64}}, xs::AbstractVector, evs::AbstractVector,
                    jacobians::AbstractVector) where {G,F,N}
    length(xs) == length(evs) == length(jacobians) ||
        throw(ArgumentError("got $(length(xs)) solutions, $(length(evs)) evaluations and $(length(jacobians)) Jacobians"))
    Base.require_one_based_indexing(xs, evs, jacobians)
    for k in eachindex(xs)
        check_jacobian(jacobians[k], N, length(xs[k]))
    end
    cells = [cell_index(archive.inner, ev.descriptor) for ev in evs]
    results = add_batch!(archive.inner, xs, evs)
    # The wrapped archive keeps, in every cell, the first of the accepted solutions with
    # the highest fitness.
    kept = Dict{Int,Int}()
    for k in eachindex(results)
        results[k].status == Rejected && continue
        current = get(kept, cells[k], 0)
        if current == 0 || convert(F, evs[k].fitness) > convert(F, evs[current].fitness)
            kept[cells[k]] = k
        end
    end
    for (i, k) in kept
        store_gradient!(archive.gradients, i, jacobians[k])
    end
    return results
end

function store_gradient!(gradients, i, jacobian)
    if isassigned(gradients, i) && size(gradients[i]) == size(jacobian)
        copyto!(gradients[i], jacobian)
    else
        gradients[i] = Matrix{Float64}(jacobian)
    end
    return nothing
end

# The loop offers the solutions of a run with their Jacobians to every archive; only a
# `GradientArchive` keeps them.
add_evaluated!(archive::AbstractArchive, xs, evs, jacobians) = add_batch!(archive, xs, evs)
add_evaluated!(archive::GradientArchive, xs, evs, jacobians) = add_batch!(archive, xs, evs, jacobians)
