"""
    MAPElites(prob; dims, bounds=nothing, batch_size=64, variation=GaussianMutation(0.1),
              genotype=default_genotype(prob), fitness=Float64, outside=ClampOutside(),
              phenotype=Nothing, device=nothing) -> QDAlgorithm

MAP-Elites: an archive laid out by the problem's descriptor space, and one
[`VariationEmitter`](@ref) that applies the `variation` operator to
`batch_size` children per round, with parents chosen uniformly from the
archive. `variation` can be a mutation such as [`GaussianMutation`](@ref) or a
crossover such as [`IsoLineVariation`](@ref).

For a problem with `descriptor_bounds`, the archive is a [`GridArchive`](@ref)
with `dims` cells along each descriptor dimension, laid out over `bounds`, by default
the bounds of the problem; `dims` is required. For a problem with
[`descriptor_categories`](@ref QualityDiversity.descriptor_categories), it is a
[`CategoricalArchive`](@ref) with one cell for each combination of categories;
`dims` and `bounds` do not apply and are an error. `outside` and `phenotype` are
passed to the archive; see [`GridArchive`](@ref) and [`CategoricalArchive`](@ref).
A descriptor outside the bounds goes to the nearest cell by default, as in pyribs and
QDax, so that a random initial population never stops a run; pass
`outside=ThrowOutside()` to catch a descriptor function that does not match its
bounds. A category has no nearest cell, so the default for a problem with categories
is `ThrowOutside()`.
A problem that returns a phenotype needs `phenotype` set to its type. The archive
stores genotypes of the type that `random_solution` returns, see
[`default_genotype`](@ref QualityDiversity.default_genotype); `genotype` overrides it.

Everything runs on the host unless `device` names an array type, such as `CuArray`:
then the archive is a [`DeviceGridArchive`](@ref) in arrays of that type and the
emitter a [`DeviceVariationEmitter`](@ref), so the variation, the evaluation and the
insertion of a batch stay on the device. That needs a problem with `descriptor_bounds`,
real-vector genotypes, no phenotype, and a `variation` that the device emitter supports.

# Example

```julia
prob = QualityDiversity.Benchmarks.Rastrigin(10)
result = solve(prob, MAPElites(prob; dims=(20, 20)); budget=100_000, rng=Xoshiro(1))
coverage(result.archive)
```
"""
function MAPElites(prob::AbstractQDProblem; dims::Union{Tuple,Nothing}=nothing,
                   bounds::Union{Tuple,Nothing}=nothing,
                   batch_size::Integer=64,
                   variation::AbstractVariation=GaussianMutation(0.1),
                   genotype::Type=default_genotype(prob),
                   fitness::Type{<:Real}=Float64,
                   outside::OutsidePolicy=default_outside(prob),
                   phenotype::Type=Nothing,
                   device::Union{Type,Nothing}=nothing)
    archive = map_elites_archive(device, descriptor_space(prob), prob, dims, bounds, genotype, fitness,
                                 outside, phenotype)
    return variation_algorithm(archive; batch_size, variation)
end

# The `outside` policy of the algorithm constructors: the nearest cell, as pyribs and
# QDax do, except for a problem with categories, which cannot be clamped.
default_outside(prob::AbstractQDProblem) = default_outside(descriptor_space(prob))
default_outside(::Tuple) = ClampOutside()
default_outside(::DescriptorCategories) = ThrowOutside()

# The archive of `MAPElites(prob; ...)`: a grid over the descriptor bounds, on the host or
# on `device`, or a CategoricalArchive for a problem with descriptor categories.
function map_elites_archive(device, ::Tuple, prob, dims, bounds, genotype, fitness, outside, phenotype)
    dims === nothing &&
        throw(ArgumentError("dims is required for a problem with descriptor_bounds: pass the number of cells along each dimension, e.g. dims=(20, 20)"))
    return grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype, device)
end

function map_elites_archive(::Nothing, ::DescriptorCategories, prob, dims, bounds, genotype, fitness, outside, phenotype)
    (dims === nothing && bounds === nothing) ||
        throw(ArgumentError("dims and bounds do not apply to a problem with descriptor_categories; its cells are the combinations of categories"))
    return CategoricalArchive(prob; genotype, fitness, outside, phenotype)
end

function map_elites_archive(::Type, ::DescriptorCategories, prob, dims, bounds, genotype, fitness, outside, phenotype)
    throw(ArgumentError("a DeviceGridArchive needs a problem with descriptor_bounds; one with descriptor_categories runs on the host"))
end

# The grid of the `(prob; ...)` constructors over `bounds`, by default those of the
# problem: a GridArchive, or with `device` an array type a DeviceGridArchive in arrays
# of that type, which stores no phenotypes.
function grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype, device=nothing)
    return grid_archive(device, prob, dims, problem_bounds(prob, bounds), genotype, fitness, outside, phenotype)
end

grid_archive(::Nothing, prob, dims, bounds, genotype, fitness, outside, phenotype) =
    GridArchive(prob; dims, bounds, genotype, fitness, outside, phenotype)

function grid_archive(device::Type, prob, dims, bounds, genotype, fitness, outside, phenotype)
    phenotype === Nothing ||
        throw(ArgumentError("a DeviceGridArchive stores no phenotypes; build it without phenotype"))
    return DeviceGridArchive(prob; dims, device, bounds, genotype, fitness, outside)
end

# `archive` with one VariationEmitter, or a DeviceVariationEmitter for an archive on a device.
variation_algorithm(archive::AbstractArchive; batch_size, variation) =
    QDAlgorithm(archive, (VariationEmitter(variation; batch_size),))

variation_algorithm(archive::DeviceCellArchive; batch_size, variation) =
    QDAlgorithm(archive, (DeviceVariationEmitter(variation; batch_size),))

# The threshold and density archives take a floating-point fitness; the constructors
# that wrap one say so before they build the grid.
function check_float_fitness(fitness::Type, what)
    fitness <: AbstractFloat ||
        throw(ArgumentError("a $what needs a floating-point fitness type such as Float64, got $fitness"))
    return nothing
end

# `num_emitters` emitters made by `make`, as the tuple that QDAlgorithm takes.
function emitter_tuple(make, num_emitters::Integer)
    num_emitters >= 1 || throw(ArgumentError("num_emitters must be at least 1, got $num_emitters"))
    return Tuple(make() for _ in 1:num_emitters)
end

"""
    MAPElites(archive::AbstractArchive; batch_size=64, variation=GaussianMutation(0.1))
        -> QDAlgorithm

MAP-Elites over any archive: `archive` with one [`VariationEmitter`](@ref) that
applies `variation` to `batch_size` children per round, or a
[`DeviceVariationEmitter`](@ref) for a [`DeviceCellArchive`](@ref). Use it to run MAP-Elites
with an archive other than the grid, such as a [`CVTArchive`](@ref) or a
[`DeviceCVTArchive`](@ref).

# Example

```julia
prob = QualityDiversity.Benchmarks.Rastrigin(10)
archive = CVTArchive(prob; cells=500, rng=Xoshiro(1))
result = solve(prob, MAPElites(archive); budget=100_000, rng=Xoshiro(2))
```
"""
function MAPElites(archive::AbstractArchive; batch_size::Integer=64,
                   variation::AbstractVariation=GaussianMutation(0.1))
    return variation_algorithm(archive; batch_size, variation)
end

"""
    CMAME(prob; dims, sigma0, num_emitters=5, batch_size=<CMA-ES default>,
          ranker=TwoStageImprovementRanker(), selection=FilterSelection(),
          restart=NoImprovementRestart(), x0=nothing, lower=-Inf, upper=Inf,
          bounds=nothing, genotype=default_genotype(prob), fitness=Float64,
          outside=ClampOutside(), phenotype=Nothing, device=nothing) -> QDAlgorithm

CMA-ME (Fontaine et al., 2020): a [`GridArchive`](@ref) with `dims` cells along each
descriptor dimension, as for [`MAPElites`](@ref), and `num_emitters` independent
[`CMAEmitter`](@ref)s, each with its own search distribution. The emitters are the
improvement emitter of the paper: candidates ranked by what the archive did with them,
the ones the archive accepted recombined, and a restart from a random elite when a
batch is rejected entirely. The paper runs 15 emitters of 36 candidates; the default is
5 emitters with the batch size of CMA-ES, `4 + floor(3 log(dim))`, so pass
`num_emitters=15, batch_size=36` for its settings.

`sigma0` is the initial step size of the search distributions and has no default,
because it depends on the scale of the solutions. The solutions must be real vectors;
their length is that of `random_solution(rng, prob)`. `lower` and `upper` bound the
solutions, and `x0` gives the mean of the first distribution of every emitter, by
default a random elite. The other arguments are those of [`MAPElites`](@ref) and
[`CMAEmitter`](@ref).

With `device`, an array type such as `CuArray`, the archive is a [`DeviceGridArchive`](@ref)
and the emitters are the searches of one [`DeviceCMAEmitter`](@ref): all searches ask
together, and their candidates are evaluated and added on the device as one batch, so
each search sees the archive as it was before the round. That pays off with many
emitters, hundreds rather than five.

# Example

```julia
prob = QualityDiversity.Benchmarks.Rastrigin(10)
result = solve(prob, CMAME(prob; dims=(20, 20), sigma0=1.0); budget=50_000, rng=Xoshiro(1))
qd_score(result.archive; offset=fitness_offset(prob))
```
"""
function CMAME(prob::AbstractQDProblem; dims::Tuple, sigma0::Real,
               bounds::Union{Tuple,Nothing}=nothing,
               genotype::Type=default_genotype(prob),
               fitness::Type{<:Real}=Float64,
               outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing,
               device::Union{Type,Nothing}=nothing, kwargs...)
    archive = grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype, device)
    return CMAME(archive; sigma0, dim=solution_dim(prob), kwargs...)
end

"""
    CMAME(archive::AbstractArchive; sigma0, dim, num_emitters=5, kwargs...) -> QDAlgorithm

CMA-ME over any archive, such as a [`CVTArchive`](@ref): `archive` with `num_emitters`
[`CMAEmitter`](@ref)s for solutions of length `dim`. The keyword arguments are those of
[`CMAEmitter`](@ref). For a [`DeviceCellArchive`](@ref) the emitters are the `num_emitters`
searches of one [`DeviceCMAEmitter`](@ref).
"""
function CMAME(archive::AbstractArchive; sigma0::Real, dim::Integer, num_emitters::Integer=5, kwargs...)
    return QDAlgorithm(archive, emitter_tuple(() -> CMAEmitter(dim; sigma0, kwargs...), num_emitters))
end

function CMAME(archive::DeviceCellArchive; sigma0::Real, dim::Integer, num_emitters::Integer=5, kwargs...)
    return QDAlgorithm(archive, (DeviceCMAEmitter(dim; num_emitters, sigma0, kwargs...),))
end

"""
    CMAMAE(prob; dims, sigma0, threshold_min, learning_rate=0.01, num_emitters=5,
           batch_size=<CMA-ES default>, ranker=ImprovementRanker(),
           selection=MuSelection(), restart=ConvergenceRestart(), x0=nothing,
           lower=-Inf, upper=Inf, bounds=nothing, genotype=default_genotype(prob),
           fitness=Float64, outside=ClampOutside(), phenotype=Nothing) -> QDAlgorithm

CMA-MAE (Fontaine and Nikolaidis, 2023): [`CMAEmitter`](@ref)s as for [`CMAME`](@ref)
but ranking candidates by their improvement over the threshold of their cell,
recombining the best half, and restarting only on convergence, over a
[`ThresholdArchive`](@ref) around a [`GridArchive`](@ref). `threshold_min`, the
lowest fitness the problem can produce, and `learning_rate` are those of the
[`ThresholdArchive`](@ref). Because the elites of that archive can get worse, `solve`
returns its result archive, which holds the best solution of every cell. The paper
runs 15 emitters of 36 candidates; see [`CMAME`](@ref) for the default. The other
arguments are those of [`CMAME`](@ref), without `device`: the threshold archive runs
on the host.
"""
function CMAMAE(prob::AbstractQDProblem; dims::Tuple, sigma0::Real, threshold_min::Real,
                learning_rate::Real=0.01, num_emitters::Integer=5,
                ranker::AbstractRanker=ImprovementRanker(),
                selection::SelectionRule=MuSelection(),
                restart::RestartRule=ConvergenceRestart(),
                bounds::Union{Tuple,Nothing}=nothing,
                genotype::Type=default_genotype(prob),
                fitness::Type{<:Real}=Float64,
                outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing, kwargs...)
    check_float_fitness(fitness, "ThresholdArchive")
    grid = grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    archive = ThresholdArchive(grid; learning_rate, threshold_min)
    return CMAME(archive; sigma0, dim=solution_dim(prob), num_emitters, ranker, selection, restart, kwargs...)
end

"""
    DDS(prob; dims, sigma0, bandwidth, buffer_capacity=10_000, rng=Random.default_rng(),
        num_emitters=15, batch_size=36, ranker=ImprovementRanker(), selection=MuSelection(),
        restart=ConvergenceRestart(), x0=nothing, lower=-Inf, upper=Inf, bounds=nothing,
        genotype=default_genotype(prob), fitness=Float64, outside=ClampOutside(),
        phenotype=Nothing) -> QDAlgorithm

Density Descent Search with a kernel density estimate (DDS-KDE; Lee et al., 2024): a
[`DensityArchive`](@ref) around a [`GridArchive`](@ref) with `dims` cells, and
`num_emitters` [`CMAEmitter`](@ref)s that rank their candidates by the density of the
descriptors found so far, lowest first, recombine the best half and restart on
convergence, the settings of the paper. It searches for diversity alone: the fitness only
decides which solution the grid keeps in a cell. `bandwidth`, `buffer_capacity` and
`rng`, which drives the buffer, are the arguments of the [`DensityArchive`](@ref). For a
reproducible run, seed `DDS` as well as `solve`, e.g. `rng=Xoshiro(1)`: without it, `rng`
is `Random.default_rng()`, which the seed of `solve` does not set. The other arguments are
those of
[`CMAME`](@ref), without `device`: the density archive runs on the host.

Density descent drives the searches toward empty regions, also out of the region where
the problem's descriptors change. On a problem that clips its solutions to a box, such as
`Rastrigin`, a search outside the box sees the descriptors of the edge of the box, and with
the default `strategy` it seldom converges, so it seldom restarts. Pass the box as `lower`
and `upper` to keep the searches inside. With `strategy=QualityDiversity.PyribsCMAES` such
a search restarts, as in pyribs: its rank-one covariance term grows with the square of the
step size, and the covariance matrix becomes ill-conditioned.

# Example

```julia
prob = QualityDiversity.Benchmarks.Rastrigin(10)
alg = DDS(prob; dims=(20, 20), sigma0=0.5, bandwidth=0.5, rng=Xoshiro(1))
result = solve(prob, alg; budget=50_000, rng=Xoshiro(2))
coverage(result.archive)
```
"""
function DDS(prob::AbstractQDProblem; dims::Tuple, sigma0::Real, bandwidth::Real,
             buffer_capacity::Integer=10_000, rng::AbstractRNG=Random.default_rng(),
             num_emitters::Integer=15, batch_size::Integer=36,
             ranker::AbstractRanker=ImprovementRanker(),
             selection::SelectionRule=MuSelection(),
             restart::RestartRule=ConvergenceRestart(),
             bounds::Union{Tuple,Nothing}=nothing,
             genotype::Type=default_genotype(prob),
             fitness::Type{<:Real}=Float64,
             outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing, kwargs...)
    check_float_fitness(fitness, "DensityArchive")
    grid = grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    archive = DensityArchive(grid; bandwidth, buffer_capacity, rng)
    return CMAME(archive; sigma0, dim=solution_dim(prob), num_emitters, batch_size, ranker, selection,
                 restart, kwargs...)
end

"""
    CMAMEGA(prob; dims, sigma0, optimizer=AdamAscent(), num_emitters=1,
            batch_size=<CMA-ES default>, ranker=TwoStageImprovementRanker(),
            selection=FilterSelection(), restart=NoImprovementRestart(), x0=nothing,
            normalize=true, bounds=nothing, genotype=default_genotype(prob), fitness=Float64,
            outside=ClampOutside(), phenotype=Nothing) -> QDAlgorithm

CMA-MEGA (Fontaine and Nikolaidis, 2021, "Differentiable Quality Diversity"): a
[`GridArchive`](@ref) with `dims` cells along each descriptor dimension, as for
[`MAPElites`](@ref), and `num_emitters` [`GradientArborescenceEmitter`](@ref)s, which step
along the gradients of the fitness and the descriptor. The problem must provide them; see
[`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient). The ranking, selection
and restart rules and the optimizer are those of CMA-MEGA (Adam) in the paper, which runs
one emitter of 36 candidates. The default batch size is that of CMA-ES over the `N + 1`
gradient coefficients, `4 + floor(3 log(N + 1))` for `N` descriptor dimensions, as in
pyribs; pass `batch_size=36` for the setting of the paper.

`sigma0` is the initial step size of the distribution of the gradient coefficients and has
no default; the paper uses `10` on the sphere and Rastrigin benchmarks. The other arguments
are those of [`MAPElites`](@ref) and [`GradientArborescenceEmitter`](@ref).

# Example

```julia
prob = QualityDiversity.Benchmarks.Sphere(100)    # gradients in closed form
result = solve(prob, CMAMEGA(prob; dims=(100, 100), sigma0=10.0); budget=20_000, rng=Xoshiro(1))
coverage(result.archive)
```
"""
function CMAMEGA(prob::AbstractQDProblem; dims::Tuple, sigma0::Real,
                 bounds::Union{Tuple,Nothing}=nothing,
                 genotype::Type=default_genotype(prob),
                 fitness::Type{<:Real}=Float64,
                 outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing, kwargs...)
    archive = grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    return CMAMEGA(archive; sigma0, dim=solution_dim(prob), kwargs...)
end

"""
    CMAMEGA(archive::AbstractArchive; sigma0, dim, num_emitters=1, kwargs...) -> QDAlgorithm

CMA-MEGA over any archive with `SVector` descriptors, such as a [`CVTArchive`](@ref):
`archive` with `num_emitters` [`GradientArborescenceEmitter`](@ref)s for solutions of length
`dim`. The keyword arguments are those of [`GradientArborescenceEmitter`](@ref).
"""
function CMAMEGA(archive::AbstractArchive{G,F,SVector{N,Float64}}; sigma0::Real, dim::Integer,
                 num_emitters::Integer=1, kwargs...) where {G,F,N}
    make() = GradientArborescenceEmitter(dim; num_descriptors=N, sigma0, kwargs...)
    return QDAlgorithm(archive, emitter_tuple(make, num_emitters))
end

function CMAMEGA(archive::AbstractArchive; kwargs...)
    throw(ArgumentError("differentiable quality diversity needs an archive with SVector descriptors, not a $(nameof(typeof(archive)))"))
end

"""
    CMAMAEGA(prob; dims, sigma0, threshold_min, learning_rate=0.01, optimizer=AdamAscent(),
             num_emitters=1, batch_size=<CMA-ES default>, ranker=ImprovementRanker(),
             selection=MuSelection(), restart=ConvergenceRestart(), x0=nothing,
             normalize=true, bounds=nothing, genotype=default_genotype(prob), fitness=Float64,
             outside=ClampOutside(), phenotype=Nothing) -> QDAlgorithm

CMA-MAEGA (Fontaine and Nikolaidis, 2023): [`CMAMEGA`](@ref) with the rules of
[`CMAMAE`](@ref): candidates ranked by their improvement over the threshold of their cell,
the best half recombined, restarts on convergence only, over a [`ThresholdArchive`](@ref)
around a [`GridArchive`](@ref). `threshold_min` and `learning_rate` are those of the
[`ThresholdArchive`](@ref); `optimizer`, the rule that moves the solution point, is that of
the [`GradientArborescenceEmitter`](@ref)s. `solve` returns the result archive, which holds
the best solution of every cell. The default batch size is that of [`CMAMEGA`](@ref).
"""
function CMAMAEGA(prob::AbstractQDProblem; dims::Tuple, sigma0::Real, threshold_min::Real,
                  learning_rate::Real=0.01,
                  ranker::AbstractRanker=ImprovementRanker(),
                  selection::SelectionRule=MuSelection(),
                  restart::RestartRule=ConvergenceRestart(),
                  bounds::Union{Tuple,Nothing}=nothing,
                  genotype::Type=default_genotype(prob),
                  fitness::Type{<:Real}=Float64,
                  outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing, kwargs...)
    check_float_fitness(fitness, "ThresholdArchive")
    grid = grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype)
    archive = ThresholdArchive(grid; learning_rate, threshold_min)
    return CMAMEGA(archive; sigma0, dim=solution_dim(prob), ranker, selection, restart, kwargs...)
end

"""
    OMGMEGA(prob; dims, sigma0, batch_size=64, normalize=true, bounds=nothing,
            genotype=default_genotype(prob), fitness=Float64, outside=ClampOutside(),
            phenotype=Nothing) -> QDAlgorithm

OMG-MEGA (Fontaine and Nikolaidis, 2021, "Differentiable Quality Diversity"): MAP-Elites
whose children are gradient steps from their parents, a [`GradientArchive`](@ref) around a
[`GridArchive`](@ref) with `dims` cells along each descriptor dimension and a
[`GradientOperatorEmitter`](@ref). Every solution of the run is evaluated with its
gradients; see [`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient). `sigma0`, the
standard deviation of the gradient coefficients, has no default; the paper uses `10` on the
sphere and Rastrigin benchmarks. The paper draws 36 children per round; the default
`batch_size=64` is that of [`MAPElites`](@ref). The other arguments are those of
[`MAPElites`](@ref) and [`GradientOperatorEmitter`](@ref).
"""
function OMGMEGA(prob::AbstractQDProblem; dims::Tuple, sigma0::Real,
                 bounds::Union{Tuple,Nothing}=nothing,
                 genotype::Type=default_genotype(prob),
                 fitness::Type{<:Real}=Float64,
                 outside::OutsidePolicy=ClampOutside(), phenotype::Type=Nothing, kwargs...)
    archive = GradientArchive(grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype))
    return OMGMEGA(archive; sigma0, kwargs...)
end

"""
    OMGMEGA(archive::GradientArchive; sigma0, batch_size=64, normalize=true) -> QDAlgorithm

OMG-MEGA over any [`GradientArchive`](@ref), such as one around a [`CVTArchive`](@ref).
"""
OMGMEGA(archive::GradientArchive; sigma0::Real, batch_size::Integer=64, normalize::Bool=true) =
    QDAlgorithm(archive, (GradientOperatorEmitter(; sigma0, batch_size, normalize),))

# Length of the solutions of `prob`, from one random solution, drawn as `default_genotype`
# draws it.
solution_dim(prob::AbstractQDProblem) = length(random_solution(Random.Xoshiro(0), prob))

# The reference point of the problem, as a default for the keyword of the same name.
problem_reference_point(prob) = reference_point(prob)

"""
    MOME(prob; dims, reference_point=reference_point(prob), max_front=50, batch_size=64,
         variation=GaussianMutation(0.1), bounds=nothing, genotype=default_genotype(prob),
         outside=ClampOutside(), duplicates=RejectDuplicates(), full_front=RejectNewcomer(),
         phenotype=Nothing) -> QDAlgorithm

Multi-objective MAP-Elites (Pierrot et al., 2022): a [`ParetoGridArchive`](@ref), which keeps a
Pareto front of at most `max_front` solutions in each cell of a grid with `dims` cells along
each descriptor dimension, and one [`VariationEmitter`](@ref) that applies `variation` to
`batch_size` children per round. A parent is a solution drawn uniformly from the front of a
cell drawn uniformly among the occupied ones. The problem returns a vector of objectives as
its fitness. `reference_point` is the corner from which hypervolume is measured; a problem
can supply it with [`reference_point`](@ref QualityDiversity.reference_point). `duplicates` and
`full_front` decide about a solution whose objectives equal those of a stored one and about
one for a full front; see [`ParetoGridArchive`](@ref). With `full_front=QualityDiversity.ReplaceCrowded()` the
archive keeps fronts as MOME-PGX does (Janmohamed et al., 2023), whose crowding-based
selection of parents and policy-gradient variation are not part of this constructor.
The default `variation` is that of [`MAPElites`](@ref); the MOME paper uses polynomial
mutation with `eta=1` together with a crossover on its Rastrigin benchmarks, and
`variation=PolynomialMutation(lower, upper; eta=1)` is the mutation alone.

`solve` returns the archive, and a run records its MOQD score, coverage, the largest
hypervolume of a cell and the number of solutions.

# Example

```julia
prob = QualityDiversity.Benchmarks.BiSphere(10)
result = solve(prob, MOME(prob; dims=(20, 20)); budget=50_000, rng=Xoshiro(1))
qd_score(result.archive), global_hypervolume(result.archive)
```
"""
function MOME(prob::AbstractQDProblem; dims::Tuple,
              reference_point::AbstractVector{<:Real}=problem_reference_point(prob),
              max_front::Integer=50, batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1),
              bounds::Union{Tuple,Nothing}=nothing, genotype::Type=default_genotype(prob),
              outside::OutsidePolicy=ClampOutside(), duplicates::DuplicatePolicy=RejectDuplicates(),
              full_front::FullFrontPolicy=RejectNewcomer(), phenotype::Type=Nothing)
    archive = ParetoGridArchive(prob; dims, reference_point, max_front, bounds, genotype, outside,
                                duplicates, full_front, phenotype)
    return variation_algorithm(archive; batch_size, variation)
end

"""
    MOME(archive::ParetoGridArchive; batch_size=64, variation=GaussianMutation(0.1)) -> QDAlgorithm

MOME over a given archive: `archive` with one [`VariationEmitter`](@ref).
"""
function MOME(archive::ParetoGridArchive; batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1))
    return variation_algorithm(archive; batch_size, variation)
end

"""
    MELS(prob::SampledSpread; dims, batch_size=64, variation=GaussianMutation(0.1),
         bounds=nothing, genotype=default_genotype(prob), fitness=Float64,
         outside=ClampOutside(), phenotype=Nothing) -> QDAlgorithm

MAP-Elites with low spread (Mace et al., 2023): MAP-Elites for a noisy problem in which an
elite is replaced only by a solution with higher fitness and no larger spread, the mean
distance between the descriptors of its evaluations. The archive is a
[`SpreadArchive`](@ref) around a [`GridArchive`](@ref) with `dims` cells along each
descriptor dimension, and one [`VariationEmitter`](@ref) applies `variation` to
`batch_size` children per round.

`prob` is a [`SampledSpread`](@ref) problem, which evaluates every solution `num_samples`
times; pass the same problem to `solve`. The other arguments are those of
[`MAPElites`](@ref).

# Example

```julia
noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x), (x[1], x[2]) .+ 0.05 .* randn(rng, 2));
                            init=rng -> randn(rng, 5), bounds=((-3, 3), (-3, 3)))
prob = SampledSpread(noisy; num_samples=10)
result = solve(prob, MELS(prob; dims=(20, 20)); budget=200_000, rng=Xoshiro(1))
```
"""
function MELS(prob::SampledSpread; dims::Tuple, batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1),
              bounds::Union{Tuple,Nothing}=nothing, genotype::Type=default_genotype(prob),
              fitness::Type{<:Real}=Float64, outside::OutsidePolicy=ClampOutside(),
              phenotype::Type=Nothing)
    archive = SpreadArchive(grid_archive(prob; dims, bounds, genotype, fitness, outside, phenotype))
    return variation_algorithm(archive; batch_size, variation)
end

function MELS(prob::AbstractQDProblem; kwargs...)
    throw(ArgumentError("MELS needs a SampledSpread problem, which evaluates every solution several times: wrap the stochastic problem as SampledSpread(prob; num_samples)"))
end

"""
    MELS(archive::SpreadArchive; batch_size=64, variation=GaussianMutation(0.1)) -> QDAlgorithm

MELS over a given archive, such as a [`SpreadArchive`](@ref) around a [`CVTArchive`](@ref):
`archive` with one [`VariationEmitter`](@ref).
"""
function MELS(archive::SpreadArchive; batch_size::Integer=64,
              variation::AbstractVariation=GaussianMutation(0.1))
    return variation_algorithm(archive; batch_size, variation)
end
