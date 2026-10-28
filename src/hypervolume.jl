"""
    hypervolume(points, reference_point) -> Float64

Hypervolume of a set of `points`, vectors of objective values that are maximized: the
volume of the region that the points dominate and that lies above `reference_point`.
A point contributes only if it is larger than the reference point in every objective,
and a dominated point adds nothing. The empty set has volume `0`.

For two objectives it sorts the points and adds the areas of the steps under them. For
more it slices along the last objective and takes the hypervolume of each slice, which
takes time of the order of `n^(m-1) log n` for `n` points and `m` objectives and suits the
small sets that a Pareto front in a cell has.

# Example

```jldoctest
julia> using QualityDiversity

julia> hypervolume([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]], [0.0, 0.0])
6.0

julia> hypervolume([[1.0, 1.0, 1.0], [2.0, 2.0, 0.5]], [0.0, 0.0, 0.0])
2.5
```
"""
function hypervolume(points, reference_point::AbstractVector{<:Real})
    m = length(reference_point)
    m >= 1 || throw(ArgumentError("the reference point needs at least one objective"))
    ref = copyto!(Vector{Float64}(undef, m), reference_point)
    pts = Vector{Float64}[]
    for p in points
        length(p) == m ||
            throw(ArgumentError("a point has $(length(p)) objectives but the reference point has $m"))
        any(isnan, p) && throw(ArgumentError("a point has a NaN objective"))
        all(j -> p[j] > ref[j], 1:m) && push!(pts, copyto!(Vector{Float64}(undef, m), p))
    end
    return slice_volume(pts, ref, m)
end

# Volume above `ref` of the points in `pts` (all above `ref`) in the first `d` objectives.
function slice_volume(pts, ref, d)
    isempty(pts) && return 0.0
    if d == 1
        return maximum(p[1] for p in pts) - ref[1]
    elseif d == 2
        return staircase_area(sort(pts; by=p -> -p[1]), ref)
    end
    # Slabs between the values of the last objective: the points at least as high as the
    # lower end of a slab all reach into it.
    sorted = sort(pts; by=p -> -p[d])
    volume = 0.0
    for i in eachindex(sorted)
        top = sorted[i][d]
        bottom = i < length(sorted) ? sorted[i+1][d] : ref[d]
        top > bottom || continue
        volume += (top - bottom) * slice_volume(sorted[1:i], ref, d - 1)
    end
    return volume
end

# Area above `ref` of two-objective points sorted by decreasing first objective.
function staircase_area(sorted, ref)
    area = 0.0
    height = ref[2]
    for i in eachindex(sorted)
        height = max(height, sorted[i][2])
        next = i < length(sorted) ? sorted[i+1][1] : ref[1]
        area += (sorted[i][1] - next) * (height - ref[2])
    end
    return area
end
