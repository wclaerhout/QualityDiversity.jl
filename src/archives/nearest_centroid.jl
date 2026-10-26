# Nearest-point search over a fixed set of centroids, and k-means to choose them.
#
# `CentroidTree` is a k-d tree stored as the permutation `order` of the points:
# the point in the middle of a range is the root of that range, and the ranges
# on either side are its subtrees. Splitting along the axis of largest spread
# keeps the tree useful in the handful of dimensions descriptors have. A search
# allocates nothing, and neither does `build!`, so the archives whose points
# change rebuild a tree in place over the first points of its vectors.

struct CentroidTree{N}
    points::Vector{SVector{N,Float64}}   # in the order of the cells
    order::Vector{Int}                   # indices of `points`, arranged as a tree
    axis::Vector{Int}                    # split axis of the node at each position of `order`
end

function CentroidTree(points::Vector{SVector{N,Float64}}) where {N}
    order = collect(1:length(points))
    axis = ones(Int, length(points))
    build!(order, axis, points, 1, length(points))
    return CentroidTree{N}(points, order, axis)
end

# Arranges `order[lo:hi]` as a tree. With `boxes`, a vector of `(lows, highs)` pairs, the
# position of the root of every range gets the bounding box of the points of the range.
function build!(order, axis, points::Vector{SVector{N,Float64}}, lo, hi, boxes=nothing) where {N}
    lo > hi && return nothing
    lows = points[order[lo]]
    highs = lows
    for k in lo+1:hi
        p = points[order[k]]
        lows = min.(lows, p)
        highs = max.(highs, p)
    end
    mid = (lo + hi) >> 1
    record_box!(boxes, mid, lows, highs)
    lo == hi && return nothing
    split = argmax(highs - lows)
    select_middle!(order, points, split, lo, mid, hi)
    axis[mid] = split
    build!(order, axis, points, lo, mid - 1, boxes)
    build!(order, axis, points, mid + 1, hi, boxes)
    return nothing
end

record_box!(::Nothing, mid, lows, highs) = nothing
record_box!(boxes, mid, lows, highs) = (boxes[mid] = (lows, highs); nothing)

# Rearranges `order[lo:hi]` so that no point before position `mid` has a larger coordinate
# `axis` than the point at `mid`, and none after it a smaller one. Quickselect with a
# median-of-three pivot and a three-way partition, so that many equal coordinates, as
# repeated descriptors give, take linear time.
function select_middle!(order, points, axis, lo, mid, hi)
    while lo < hi
        a = points[order[lo]][axis]
        b = points[order[(lo+hi)>>1]][axis]
        c = points[order[hi]][axis]
        pivot = max(min(a, b), min(max(a, b), c))
        # [lo, below) less than the pivot, [below, i) equal, (above, hi] greater
        below, i, above = lo, lo, hi
        while i <= above
            v = points[order[i]][axis]
            if v < pivot
                order[below], order[i] = order[i], order[below]
                below += 1
                i += 1
            elseif v > pivot
                order[above], order[i] = order[i], order[above]
                above -= 1
            else
                i += 1
            end
        end
        if mid < below
            hi = below - 1
        elseif mid > above
            lo = above + 1
        else
            return nothing
        end
    end
    return nothing
end

"""
    nearest(tree::CentroidTree, q) -> Int

Index of the point of `tree` closest to `q` in Euclidean distance; the lowest
index among equally close points.
"""
function nearest(tree::CentroidTree{N}, q::SVector{N,Float64}) where {N}
    index, _ = search(tree, q, 1, length(tree.order), 0, Inf, Returns(false))
    return index
end

# The nearest point to `q` among positions `lo:hi` of the tree and the candidate `best` at
# squared distance `best_distance`, with its squared distance; the lower index on a tie.
# Points `i` with `skip(i)` are passed over.
function search(tree::CentroidTree{N}, q::SVector{N,Float64}, lo::Int, hi::Int,
                best::Int, best_distance::Float64, skip::S)::Tuple{Int,Float64} where {N,S}
    lo > hi && return (best, best_distance)
    mid = (lo + hi) >> 1
    i = tree.order[mid]
    c = tree.points[i]
    if !skip(i)
        distance = sum(abs2, c - q)
        if distance < best_distance || (distance == best_distance && i < best)
            best, best_distance = i, distance
        end
    end
    lo == hi && return (best, best_distance)
    axis = tree.axis[mid]
    offset = q[axis] - c[axis]
    if offset < 0
        best, best_distance = search(tree, q, lo, mid - 1, best, best_distance, skip)
        if offset * offset <= best_distance
            best, best_distance = search(tree, q, mid + 1, hi, best, best_distance, skip)
        end
    else
        best, best_distance = search(tree, q, mid + 1, hi, best, best_distance, skip)
        if offset * offset <= best_distance
            best, best_distance = search(tree, q, lo, mid - 1, best, best_distance, skip)
        end
    end
    return (best, best_distance)
end

# k-means with k-means++ seeding. Lloyd's iterations stop when no sample changes
# cluster, or when an iteration lowers the total squared distance from the
# samples to their centroids by less than the fraction `tolerance` of it.
function kmeans(rng::AbstractRNG, samples::Vector{SVector{N,Float64}}, k::Int;
                max_iterations::Int=100, tolerance::Float64=1e-3) where {N}
    centers = kmeans_seeds(rng, samples, k)
    labels = zeros(Int, length(samples))
    sums = zeros(SVector{N,Float64}, k)
    counts = zeros(Int, k)
    previous = Inf
    for _ in 1:max_iterations
        tree = CentroidTree(centers)
        fill!(sums, zero(SVector{N,Float64}))
        fill!(counts, 0)
        changed = false
        total = 0.0
        for (s, x) in enumerate(samples)
            c = nearest(tree, x)
            changed |= labels[s] != c
            labels[s] = c
            total += sum(abs2, x - centers[c])
            sums[c] += x
            counts[c] += 1
        end
        for j in 1:k
            # a cluster that lost all its samples restarts at a random sample
            centers[j] = counts[j] > 0 ? sums[j] / counts[j] : samples[rand(rng, 1:length(samples))]
        end
        (changed && previous - total > tolerance * total) || break
        previous = total
    end
    return centers
end

# k-means++: each seed is a sample drawn with probability proportional to its
# squared distance from the seeds chosen so far.
function kmeans_seeds(rng::AbstractRNG, samples::Vector{SVector{N,Float64}}, k::Int) where {N}
    n = length(samples)
    seeds = Vector{SVector{N,Float64}}(undef, k)
    seeds[1] = samples[rand(rng, 1:n)]
    distance = [sum(abs2, x - seeds[1]) for x in samples]
    for j in 2:k
        total = sum(distance)
        pick = n
        if total > 0
            threshold = rand(rng) * total
            running = 0.0
            for i in 1:n
                running += distance[i]
                if running >= threshold
                    pick = i
                    break
                end
            end
        else
            pick = rand(rng, 1:n)
        end
        seeds[j] = samples[pick]
        for i in 1:n
            distance[i] = min(distance[i], sum(abs2, samples[i] - seeds[j]))
        end
    end
    return seeds
end

# Appends to `out` the indices of the points of `tree` within squared distance `radius2` of
# `q`, in the order of the tree.
function within!(out::Vector{Int}, tree::CentroidTree{N}, q::SVector{N,Float64}, radius2::Float64,
                 lo::Int=1, hi::Int=length(tree.order)) where {N}
    lo > hi && return out
    mid = (lo + hi) >> 1
    i = tree.order[mid]
    c = tree.points[i]
    sum(abs2, c - q) <= radius2 && push!(out, i)
    lo == hi && return out
    axis = tree.axis[mid]
    offset = q[axis] - c[axis]
    # the points left of `mid` are no further along `axis` than `c`, those right of it no less
    (offset <= 0 || offset * offset <= radius2) && within!(out, tree, q, radius2, lo, mid - 1)
    (offset >= 0 || offset * offset <= radius2) && within!(out, tree, q, radius2, mid + 1, hi)
    return out
end

"""
    CentroidBuckets(tree, lower, upper)

A regular grid of buckets over the box from `lower` to `upper`, with the centroids of `tree`
that can be nearest to a point of each bucket, so that a nearest-centroid search compares a
point with the few centroids of its bucket instead of all of them. A device kernel uses it
to find the cell of a descriptor of a `DeviceCVTArchive`.

`members[offsets[b]:offsets[b+1]-1]` are the centroids of bucket `b`, in increasing order;
the buckets are numbered as the cells of a [`GridArchive`](@ref) with `dims` cells along
each dimension. A centroid is kept for a bucket when its distance to the bucket is at most
the smallest distance within which some centroid covers the whole bucket: every centroid
that is nearest to a point of the bucket, and every one tied with it, passes that test.
The buckets are widened by a margin of `1e-9` of the box on every side before the test, so
that a point that rounding places in the neighboring bucket is still covered.

There are about as many buckets as centroids, laid out in proportion to the sides of the
box. When the centroids of the buckets would number more than 32 per centroid, as in many
descriptor dimensions, the grid is halved along every dimension until they do not; one
bucket that holds every centroid is the last resort.
"""
struct CentroidBuckets{N}
    dims::NTuple{N,Int}
    offsets::Vector{Int32}
    members::Vector{Int32}
end

function CentroidBuckets(tree::CentroidTree{N}, lower::SVector{N,Float64}, upper::SVector{N,Float64}) where {N}
    ncells = length(tree.points)
    widths = upper - lower
    scale = prod(widths)^(1 / N)
    dims = ntuple(r -> max(1, round(Int, ncells^(1 / N) * widths[r] / scale)), Val(N))
    while true
        buckets = bucket_members(tree, lower, upper, dims)
        (all(==(1), dims) || length(buckets.members) <= 32 * ncells) && return buckets
        dims = map(m -> max(1, m ÷ 2), dims)
    end
end

function bucket_members(tree::CentroidTree{N}, lower::SVector{N,Float64}, upper::SVector{N,Float64},
                        dims::NTuple{N,Int}) where {N}
    widths = upper - lower
    margin = 1e-9 * widths
    offsets = Int32[1]
    members = Int32[]
    candidates = Int[]
    for index in CartesianIndices(dims)
        k = SVector{N,Int}(Tuple(index))
        lo = lower + (k .- 1) .* widths ./ SVector(dims) - margin
        hi = lower + k .* widths ./ SVector(dims) + margin
        center = (lo + hi) / 2
        half_diagonal = sqrt(sum(abs2, hi - lo)) / 2
        nearest_distance = sqrt(sum(abs2, tree.points[nearest(tree, center)] - center))
        # every centroid that can be nearest to a point of the box is within this of its center
        empty!(candidates)
        within!(candidates, tree, center, (nearest_distance + 2half_diagonal)^2 * (1 + 1e-12))
        cover = minimum(i -> farthest_squared(tree.points[i], lo, hi), candidates)
        for i in sort!(candidates)
            closest_squared(tree.points[i], lo, hi) <= cover * (1 + 1e-12) && push!(members, i)
        end
        push!(offsets, length(members) + 1)
    end
    return CentroidBuckets{N}(dims, offsets, members)
end

# The squared distance from `c` to the nearest and to the farthest point of the box.
closest_squared(c, lo, hi) = sum(abs2, c - clamp.(c, lo, hi))
farthest_squared(c, lo, hi) = sum(abs2, max.(abs.(c - lo), abs.(c - hi)))
