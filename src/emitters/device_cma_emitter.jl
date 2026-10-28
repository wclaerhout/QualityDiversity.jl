"""
    DeviceCMAEmitter(dim; n_emitters, sigma0, kwargs...)

The emitter of CMA-ME over a [`DeviceCellArchive`](@ref): `n_emitters` independent CMA-ES
searches, each a [`CMAEmitter`](@ref) with the keyword arguments of `CMAEmitter`, that ask
together. Their candidates form one batch, which is evaluated and added to the archive on
the device at once, so every search of a round sees the archive as it was before the
round, as in pyribs. A host [`CMAME`](@ref) instead steps its emitters one after another,
as in the CMA-ME paper. The searches that restart draw their new starts from the archive
together, at the start of the round. A device only pays off with many searches, or many
candidates per search: the batch has `n_emitters` times the batch size of one search.

The searches stay on the host: each draws its candidates from its own distribution, the
searches in parallel on the threads of the host, each with a generator seeded by a number
drawn in order from the run's generator, so that a run is reproducible from its seed
whatever the number of threads. The batch is then copied to the device. After the batch is
added, its fitness values, descriptors and results are copied back once, and each search
ranks its part and adapts its distribution as a `CMAEmitter` does, the searches in
parallel on the threads of the host.

The property `searches` holds the searches, for example to read their `iterations` and
`restarts`.
"""
mutable struct DeviceCMAEmitter{E<:CMAEmitter} <: AbstractEmitter
    const searches::Vector{E}
    # A batch was asked and is waiting for its results.
    asked::Bool
end

function DeviceCMAEmitter(dim::Integer; n_emitters::Integer, kwargs...)
    n_emitters >= 1 || throw(ArgumentError("n_emitters must be at least 1, got $n_emitters"))
    first_search = CMAEmitter(dim; kwargs...)
    searches = [first_search; [CMAEmitter(dim; kwargs...) for _ in 2:n_emitters]]
    return DeviceCMAEmitter{typeof(first_search)}(searches, false)
end

batch_size(emitter::DeviceCMAEmitter) = sum(batch_size, emitter.searches)

"""
    ask(rng, emitter::DeviceCMAEmitter, archive::DeviceCellArchive) -> DeviceBatch

The candidates of every search, in the order of the searches, as one batch on the device.
"""
function ask(rng::AbstractRNG, emitter::DeviceCMAEmitter, archive::DeviceCellArchive{T}) where {T}
    dim = solution_length(archive)
    first(emitter.searches).es.dim == dim ||
        throw(ArgumentError("the solutions of the archive have $dim entries but the emitter has dimension $(first(emitter.searches).es.dim)"))
    searches = emitter.searches
    starts = start_means(rng, emitter, archive)
    # One seed per search, drawn in order, so the batch does not depend on the threads.
    seeds = [rand(rng, UInt64) for _ in searches]
    X = Matrix{T}(undef, dim, batch_size(emitter))
    λ = batch_size(first(searches))
    Threads.@threads for i in eachindex(searches)
        search_rng = Random.Xoshiro(seeds[i])
        starts[i] === nothing || start_search!(search_rng, searches[i], archive, starts[i])
        es = draw_batch!(search_rng, searches[i])
        X[:, (i-1)*λ+1:i*λ] = es.x
    end
    emitter.asked = true
    return DeviceBatch(copyto!(similar(archive.genotypes, T, size(X)), X))
end

# The mean of the new search of every search that starts one, `nothing` for the others. The
# elites they start from are drawn with one call, in the order of the searches, and copied
# from the device together.
function start_means(rng, emitter::DeviceCMAEmitter, archive)
    searches = emitter.searches
    means = Vector{Union{Nothing,Vector{Float64}}}(nothing, length(searches))
    from_elite = [needs_start(s) && !starts_from_x0(s) for s in searches]
    elites = sample_elites(rng, archive, count(from_elite))
    e = 0
    for (i, search) in pairs(searches)
        if from_elite[i]
            # Moved inside the bounds of the search, as `elite_start` does on the host.
            es = search.es
            means[i] = clamp.(collect(Float64, elites[e += 1]), es.lower, es.upper)
        elseif needs_start(search)
            means[i] = search.x0
        end
    end
    return means
end

function ask(rng::AbstractRNG, emitter::DeviceCMAEmitter, archive::AbstractArchive)
    throw(ArgumentError("a DeviceCMAEmitter adds its candidates to a device archive such as a DeviceGridArchive, not a $(nameof(typeof(archive)))"))
end

"""
    tell!(emitter::DeviceCMAEmitter, xs, evs, results)

Rank the part of the batch of every search and adapt its distribution, the searches in
parallel. `xs`, `evs` and `results` must be those of the batch that the last `ask`
returned.
"""
function tell!(emitter::DeviceCMAEmitter, xs, evs, results)
    emitter.asked ||
        throw(ArgumentError("tell! needs the results of the batch that the last ask returned, but nothing was asked since the last tell!"))
    n = batch_size(emitter)
    (length(xs) == n && length(evs) == n && length(results) == n) ||
        throw(ArgumentError("expected $n solutions, evaluations and results but got $(length(xs)), $(length(evs)) and $(length(results))"))
    emitter.asked = false
    host_evs = host_evaluations(evs)
    host_results = host_add_results(results)
    searches = emitter.searches
    λ = batch_size(first(searches))
    Threads.@threads for i in eachindex(searches)
        part = (i-1)*λ+1:i*λ
        tell!(searches[i], view(xs, part), view(host_evs, part), view(host_results, part))
    end
    return nothing
end
