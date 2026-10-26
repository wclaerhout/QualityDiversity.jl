"""
    QualityDiversity

Quality-diversity optimization for any genotype that can be written as a Julia
type.

Five roles make up an algorithm: a problem ([`AbstractQDProblem`](@ref)), an
archive ([`AbstractArchive`](@ref)), variation operators
([`AbstractVariation`](@ref)), emitters ([`AbstractEmitter`](@ref)), and the
algorithm that combines them ([`QDAlgorithm`](@ref)). Each role is an abstract
type with a documented set of required and optional functions.

Interface functions such as `evaluate` and `add!` are `public` but not
exported. Extend them with qualified names:

```julia
QualityDiversity.evaluate(p::MyProblem, x) = ...
```

The values of keyword arguments, such as the rankers and restart rules of a
[`CMAEmitter`](@ref), are `public` too: `ranker=QualityDiversity.ObjectiveRanker()`.
The benchmark problems are in [`QualityDiversity.Benchmarks`](@ref).
"""
module QualityDiversity

using CommonSolve: CommonSolve, init, step!, solve!, solve
using Compat: @compat
using LinearAlgebra: I, Symmetric, eigen!, mul!, norm
using Random: Random, AbstractRNG, randn!
using StaticArrays: SVector, pushfirst

include("evaluation.jl")
include("interfaces.jl")
include("metrics.jl")
include("hypervolume.jl")
include("problems/qdproblem.jl")
include("problems/stochastic.jl")
include("gradients.jl")
include("problems/benchmarks.jl")
include("archives/cell_archive.jl")
include("archives/grid.jl")
include("archives/sliding.jl")
include("archives/nearest_centroid.jl")
include("archives/cvt.jl")
include("archives/categorical.jl")
include("archives/unstructured.jl")
include("archives/dns.jl")
include("archives/density.jl")
include("archives/threshold.jl")
include("archives/spread.jl")
include("archives/pareto.jl")
include("archives/gradient.jl")
include("archives/device_cell_archive.jl")
include("archives/device_grid.jl")
include("archives/device_cvt.jl")
include("variation/gaussian.jl")
include("variation/isoline.jl")
include("variation/polynomial.jl")
include("variation/crossover.jl")
include("emitters/variation_emitter.jl")
include("emitters/evolution_strategy.jl")
include("emitters/cma_es.jl")
include("emitters/sep_cma_es.jl")
include("emitters/lm_ma_es.jl")
include("emitters/openai_es.jl")
include("emitters/rankers.jl")
include("emitters/cma_emitter.jl")
include("emitters/bandit_scheduler.jl")
include("emitters/gradient_optimizers.jl")
include("emitters/gradient_arborescence.jl")
include("emitters/gradient_operator.jl")
include("emitters/device_emitter.jl")
include("emitters/device_cma_emitter.jl")
include("solve.jl")
include("algorithms.jl")
include("testing.jl")

# The exported names are the ones a script uses: the five roles, the problem wrappers,
# the archives, variations, emitters and algorithms, the CommonSolve verbs and the
# metrics. Everything else that is part of the interface is `public`: the interface
# functions, the keyword values (rankers, selection and restart rules, policies, the
# optimizers and evolution strategies), the result and state types, and the Benchmarks
# submodule. Use them with a qualified name, `QualityDiversity.ObjectiveRanker()`.
export AbstractQDProblem, AbstractStochasticQDProblem, AbstractArchive, AbstractVariation, AbstractEmitter
export Evaluation, QDAlgorithm, QDProblem, StochasticQDProblem, Sampled, SampledSpread
export GridArchive, SlidingBoundariesArchive, CVTArchive, CategoricalArchive, UnstructuredArchive, DNSArchive, DensityArchive, ThresholdArchive,
    SpreadArchive, ParetoGridArchive, GradientArchive, DeviceGridArchive, DeviceCVTArchive
export ThrowOutside, ClampOutside, RejectOutside
export GaussianMutation, IsoLineVariation, PolynomialMutation, UniformCrossover
export VariationEmitter, CMAEmitter, BanditScheduler, GradientArborescenceEmitter, GradientOperatorEmitter
export MAPElites, CMAME, CMAMAE, CMAMEGA, CMAMAEGA, OMGMEGA, MOME, MELS, DDS
export init, step!, solve!, solve
export qd_score, coverage, max_fitness, mean_fitness, cqd_score, hypervolume, global_hypervolume

@compat public Benchmarks
@compat public AddStatus, AddResult, QDState, QDResult, CQDScore, num_solutions, max_hypervolume, max_scores
@compat public Expectation, Average, Median
@compat public OutsidePolicy, DuplicatePolicy, RejectDuplicates, KeepDuplicates, FullFrontPolicy, RejectNewcomer, ReplaceCrowded
@compat public DeviceVariationEmitter, DeviceCMAEmitter
@compat public GradientOptimizer, AdamAscent, GradientAscent
@compat public AbstractEvolutionStrategy, CMAES, PyribsCMAES, SepCMAES, LMMAES, OpenAIES
@compat public AbstractRanker, ObjectiveRanker, ImprovementRanker, TwoStageImprovementRanker, TwoStageObjectiveRanker,
    DirectionRanker, RandomDirectionRanker, TwoStageRandomDirectionRanker, AcceptedRandomDirectionRanker
@compat public SelectionRule, FilterSelection, MuSelection
@compat public RestartRule, NoImprovementRestart, ConvergenceRestart, PeriodicRestart
@compat public ReselectRule, ReselectTerminated, ReselectAll
@compat public Rejected, Improved, NewCell
@compat public evaluate, evaluate_batch, evaluate_gradient, evaluate_gradient_batch, evaluations_per_solution, expectation, random_solution, default_genotype,
    descriptor_bounds, descriptor_categories, fitness_offset, reference_point
@compat public add!, add_batch!, cell_index, sample_elites, elites, capacity, result_archive
@compat public vary, n_parents
@compat public CellArchive, boundaries, density, centroids, thresholds, spreads, DescriptorSamples
@compat public DeviceCellArchive, DeviceBatch, EvaluationBatch, AddResultBatch
@compat public Testing
@compat public reset!, sample!, update!, should_stop
@compat public RankOneRule, TextbookRankOne, PyribsRankOne
@compat public ask, tell!, ask_gradients, tell_gradients!, batch_size, restarts, active, rank_key, reset_ranker!

end
