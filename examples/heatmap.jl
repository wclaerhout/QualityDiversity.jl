# # Plotting archives
#
# Plotting lives in a package extension: loading a Makie backend, such as
# CairoMakie or GLMakie, adds plot recipes for archives, and the package gains no
# plotting functions of its own. The recipes are
#
# - `heatmap(archive)` for an archive with two descriptor dimensions, and
#   `scatter(archive)` for an `UnstructuredArchive`,
# - `ecdfplot(archive)` for the distribution of the elite fitness, and
# - `lines(cqd_score(...))` for the continuous QD score against its penalty weight.
#
# Run it from the repository root, after the one-time setup
#
#     julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# with
#
#     julia --project=examples examples/heatmap.jl
#
# It saves the figures next to this file.

using QualityDiversity, QualityDiversity.Benchmarks
using CairoMakie
using Random: Xoshiro

# ## Two archives
#
# The same problem and budget, once with a grid and once with a centroidal Voronoi
# tessellation (CVT) of 1000 cells.

prob = Rastrigin(10)
bounds = QualityDiversity.descriptor_bounds(prob)
variation = GaussianMutation(0.5)
grid = solve(prob, MAPElites(prob; dims=(40, 40), batch_size=100, variation); budget=100_000, rng=Xoshiro(1))
cvt = solve(prob, MAPElites(CVTArchive(1_000, bounds; rng=Xoshiro(1)); batch_size=100, variation);
            budget=100_000, rng=Xoshiro(1))

# ## Heatmaps
#
# `heatmap!(ax, archive)` draws the fitness of every cell; empty cells are
# transparent. A grid shows one rectangle per cell, a CVT archive its Voronoi
# cells, and a `CategoricalArchive` one square per pair of categories, with the
# category values as the axis labels. Archives that wrap another, such as a
# `ThresholdArchive`, show the archive of their best solutions.

fig = Figure(size=(1000, 420))
for (column, (title, result)) in enumerate(("GridArchive" => grid, "CVTArchive" => cvt))
    ax = Axis(fig[1, 2*column-1]; title, xlabel="descriptor 1", ylabel="descriptor 2", aspect=1)
    hm = heatmap!(ax, result.archive)
    Colorbar(fig[1, 2*column], hm; label="fitness")
end
save(joinpath(@__DIR__, "heatmaps.png"), fig) #src
fig #md

# ## Distribution of the elite fitness
#
# `ecdfplot` shows, for each fitness value, the fraction of elites at or below it.
# Unlike the QD score it does not sum over the cells, so it compares archives of
# different sizes.

fig = Figure(size=(500, 350))
ax = Axis(fig[1, 1]; xlabel="fitness", ylabel="fraction of elites")
ecdfplot!(ax, grid.archive; label="GridArchive")
ecdfplot!(ax, cvt.archive; label="CVTArchive")
axislegend(ax; position=:lt)
save(joinpath(@__DIR__, "ecdf.png"), fig) #src
fig #md

# ## Metrics over a run
#
# `result.history` holds the metrics after every round, so the progress of a run
# is a line plot.

fig = Figure(size=(500, 350))
ax = Axis(fig[1, 1]; xlabel="evaluations", ylabel="QD score")
for (label, result) in ("GridArchive" => grid, "CVTArchive" => cvt)
    lines!(ax, [r.n_evals for r in result.history], [r.qd_score for r in result.history]; label)
end
axislegend(ax; position=:rb)
save(joinpath(@__DIR__, "qd_score.png"), fig) #src
fig #md

# ## Continuous QD score
#
# The QD score depends on the cells of an archive. The continuous QD score (Kent et
# al., 2022) does not: it asks, for random target descriptors, how good the elite
# that the archive recommends is, with a penalty for its distance from the target
# that grows with a weight `θ`. `lines(result)` plots the mean score against `θ`.

fitness_range = (-QualityDiversity.fitness_offset(prob), 0.0)    # the range of Rastrigin(10)
fig = Figure(size=(500, 350))
ax = Axis(fig[1, 1]; xlabel="penalty weight θ", ylabel="mean ω over the targets")
for (label, result) in ("GridArchive" => grid, "CVTArchive" => cvt)
    score = cqd_score(Xoshiro(2), result.archive; n_targets=200, fitness_range)
    lines!(ax, score; label)
end
axislegend(ax)
save(joinpath(@__DIR__, "cqd.png"), fig) #src
fig #md
