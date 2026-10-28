# # Noisy problems
#
# Many problems do not give the same answer twice: a simulation with random
# initial conditions, a robot on an uneven floor, a measurement with noise. An
# archive that stores one evaluation per solution then fills with lucky solutions,
# whose single evaluation was better than their average, and with solutions in the
# wrong cell, whose descriptor happened to land there.
#
# This tutorial shows the effect and the two remedies the package offers:
# averaging several evaluations with `Sampled`, and MAP-Elites with low spread
# (`MELS`), which also prefers solutions whose descriptor varies little.
#
# Set up once, from the repository root:
#
#     julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Run:
#
#     julia --project=examples examples/noisy.jl

using QualityDiversity
using Random: AbstractRNG, Xoshiro

# ## A noisy problem
#
# The solutions are vectors of 6 numbers. The expected fitness is the negated
# squared norm, and every evaluation adds noise of standard deviation 1. The
# descriptor is the first two entries, clipped to the box, with noise of standard
# deviation `0.02 + |x[3]| / 2`: the third entry decides how reliable the descriptor of
# a solution is, without changing its expected fitness.
#
# A stochastic problem is a function of a random number generator and a solution.
# It must draw every random number from that generator, which the run seeds, so
# that the run is reproducible from its seed.

expected_fitness(x) = -sum(abs2, x)
expected_descriptor(x) = [clamp(x[1], -2, 2), clamp(x[2], -2, 2)]
mean(v) = sum(v) / length(v)

function noisy_score(rng::AbstractRNG, x)
    fitness = expected_fitness(x) + randn(rng)
    spread = 0.02 + 0.5 * abs(x[3])
    descriptor = expected_descriptor(x) .+ spread .* randn(rng, 2)
    return fitness, (clamp(descriptor[1], -2, 2), clamp(descriptor[2], -2, 2))
end

noisy = StochasticQDProblem(noisy_score; init=rng -> randn(rng, 6), bounds=((-2, 2), (-2, 2)))
nothing #hide

# ## How good are the elites really?
#
# To judge an archive, every elite is evaluated without noise: its expected
# fitness is compared with the fitness the archive stored, and its expected
# descriptor with the cell it is stored in. `judge` returns the number of elites,
# their mean stored and mean expected fitness, and the fraction of elites in the
# wrong cell.

function judge(archive)
    elites = collect(QualityDiversity.elites(archive))
    stored = [ev.fitness for (x, ev) in elites]
    expected = [expected_fitness(x) for (x, ev) in elites]
    misplaced = count(((x, ev),) -> QualityDiversity.cell_index(archive, expected_descriptor(x)) !=
                                    QualityDiversity.cell_index(archive, ev.descriptor), elites)
    return (elites=length(elites), stored_fitness=mean(stored), expected_fitness=mean(expected),
            wrong_cell=misplaced / length(elites))
end

# ## One evaluation per solution
#
# A stochastic problem works with every algorithm. Plain MAP-Elites evaluates each
# solution once.

budget = 200_000
dims = (10, 10)

plain = solve(noisy, MAPElites(noisy; dims); budget, rng=Xoshiro(1))
judge(plain.archive)

# The stored fitness is far above the expected one: of many solutions with similar
# expected fitness, the archive keeps the one with the luckiest evaluation. And
# nearly every elite sits in a cell its expected descriptor does not belong to.

# ## Averaging several evaluations
#
# `Sampled` evaluates every solution `n_samples` times and reports the mean
# fitness and the mean descriptor (`expectation=QualityDiversity.Median()` takes the median). Every
# evaluation counts against the budget, so the run scores ten times fewer
# solutions.

averaged = Sampled(noisy; n_samples=10)
mean_run = solve(averaged, MAPElites(averaged; dims); budget, rng=Xoshiro(1))
judge(mean_run.archive)

# The stored fitness is now much closer to the expected fitness, and fewer elites
# are in the wrong cell. The expected fitness is lower than with one evaluation,
# because the run tried ten times fewer solutions: averaging trades search for
# reliability.

# ## Low spread
#
# Averaging does not stop a solution with an unreliable descriptor from taking a
# cell. MAP-Elites with low spread (Mace et al., 2023) keeps every descriptor of
# the `n_samples` evaluations: the spread of a solution is the mean distance
# between them, and a solution replaces an elite only if it is fitter and its
# spread is no larger. `SampledSpread` evaluates the solutions, and `MELS` builds
# the algorithm around a `SpreadArchive`.

spread_prob = SampledSpread(noisy; n_samples=10)
mels = solve(spread_prob, MELS(spread_prob; dims); budget, rng=Xoshiro(1))
judge(mels.archive)

# Still fewer elites of MELS are in the wrong cell, at some cost in fitness. Its
# elites have reliable descriptors: their third entry, which sets the descriptor
# noise, is smaller.

mean_abs_x3(archive) = mean([abs(x[3]) for (x, _) in QualityDiversity.elites(archive)])
(MELS=mean_abs_x3(mels.archive), mean_of_10=mean_abs_x3(mean_run.archive))
