```@meta
CurrentModule = QualityDiversity
```

# Choosing an algorithm

Every algorithm is an archive and a tuple of emitters. The constructors below choose
both with the rules of the paper that introduced the algorithm, with one exception:
[`CMAME`](@ref) and [`CMAMAE`](@ref) run five emitters with the batch size of CMA-ES
rather than the 15 emitters of 36 candidates of the papers, which cost about ten times
more evaluations per round; pass `n_emitters=15, batch_size=36` for the papers'
settings. The API reference gives every keyword.

## Algorithms

| Constructor | Algorithm | Archive | Emitters | Use it when |
|---|---|---|---|---|
| [`MAPElites`](@ref) | MAP-Elites (Mouret and Clune, 2015) | [`GridArchive`](@ref), or [`CategoricalArchive`](@ref) for discrete descriptors | one [`VariationEmitter`](@ref) | any genotype; the first thing to try |
| [`CMAME`](@ref) | CMA-ME (Fontaine et al., 2020) | [`GridArchive`](@ref) | five [`CMAEmitter`](@ref)s (15 × 36 in the paper) | real-vector genotypes, especially many parameters |
| [`CMAMAE`](@ref) | CMA-MAE (Fontaine and Nikolaidis, 2023) | [`ThresholdArchive`](@ref) around a grid | five [`CMAEmitter`](@ref)s (15 × 36 in the paper) | as CMA-ME, when fitness should count for more |
| [`CMAMEGA`](@ref) | CMA-MEGA (Fontaine and Nikolaidis, 2021) | [`GridArchive`](@ref) | one [`GradientArborescenceEmitter`](@ref) | the fitness and descriptor have gradients |
| [`CMAMAEGA`](@ref) | CMA-MAEGA (Fontaine and Nikolaidis, 2023) | [`ThresholdArchive`](@ref) around a grid | one [`GradientArborescenceEmitter`](@ref) | as CMA-MEGA, when fitness should count for more |
| [`OMGMEGA`](@ref) | OMG-MEGA (Fontaine and Nikolaidis, 2021) | [`GradientArchive`](@ref) around a grid | one [`GradientOperatorEmitter`](@ref) | gradients, and a simpler search than CMA-MEGA |
| [`DDS`](@ref) | Density Descent Search (Lee et al., 2024) | [`DensityArchive`](@ref) around a grid | fifteen [`CMAEmitter`](@ref)s | exploring descriptor space with little regard for fitness |
| [`MOME`](@ref) | Multi-objective MAP-Elites (Pierrot et al., 2022) | [`ParetoGridArchive`](@ref) | one [`VariationEmitter`](@ref) | several objectives |
| [`MELS`](@ref) | MAP-Elites with low spread (Mace et al., 2023) | [`SpreadArchive`](@ref) around a grid | one [`VariationEmitter`](@ref) | noisy descriptors |

A few rules of thumb:

- **Start with MAP-Elites.** It works with any genotype for which you can write a
  variation operator, and its one parameter that matters is the size of the mutation.
- **Real vectors with many parameters:** the CMA emitters adapt the shape of their
  search to the problem and usually reach better fitness than a fixed Gaussian
  mutation. `sigma0`, the initial step size, has no default because it depends on the
  scale of the solutions; a tenth to a half of the range of a parameter is a usual
  start.
- **The optimizer of the CMA emitters** is the `strategy` keyword.
  [`CMAES`](@ref QualityDiversity.CMAES), the default, suits up to a few hundred
  parameters. Beyond that, [`SepCMAES`](@ref QualityDiversity.SepCMAES) and
  [`LMMAES`](@ref QualityDiversity.LMMAES) cost less per solution. `LMMAES` works only in
  a high dimension: it needs at least twice as many parameters as the batch size, so at
  least 26 with the default batch, and throws otherwise.
- **CMA-ME or CMA-MAE:** CMA-MAE keeps searching where it has already found good
  solutions, which CMA-ME abandons once a region is covered. It needs `threshold_min`,
  the lowest fitness the problem can produce.
- **Gradients:** when the fitness and the descriptor are differentiable functions of a
  real vector, the algorithms of differentiable quality diversity step along their
  gradients and fill an archive in far fewer evaluations than the others. Load
  ForwardDiff.jl (`using ForwardDiff`) to compute the gradients of a [`QDProblem`](@ref),
  or define [`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient) for the problem.
  CMA-MEGA computes the gradients once per round, at one point; OMG-MEGA computes them for
  every solution, which costs more when a gradient is expensive.
- **Noisy problems:** wrap the problem in [`Sampled`](@ref) to average several
  evaluations; use MELS when the descriptor is noisy too. See the
  [noisy problems tutorial](../tutorials/noisy.md).

## Archives

The constructors pick a grid, but any archive can be given to [`MAPElites`](@ref),
[`CMAME`](@ref) and the others in place of the problem:

```julia
archive = CVTArchive(prob; cells=1_000, rng=Xoshiro(1))
alg = MAPElites(archive; batch_size=100)
```

| Archive | Cells | Use it when |
|---|---|---|
| [`GridArchive`](@ref) | a regular grid over the descriptor bounds | few descriptor dimensions (up to about 4) |
| [`CVTArchive`](@ref) | the Voronoi cells of centroids from k-means | more descriptor dimensions, or a fixed number of cells |
| [`SlidingBoundariesArchive`](@ref) | a grid whose boundaries follow the quantiles of recent descriptors | descriptors crowd into a small part of their range |
| [`CategoricalArchive`](@ref) | one per combination of discrete values | descriptors are labels or small integers |
| [`UnstructuredArchive`](@ref) | none: a solution enters if it is far from those stored | descriptor bounds are unknown (novelty search) |
| [`DNSArchive`](@ref) | none: a population chosen by distance to fitter neighbours | as above, with a fixed population size |
| [`ThresholdArchive`](@ref) | those of the archive it wraps | CMA-MAE |
| [`SpreadArchive`](@ref) | those of the grid or CVT archive it wraps | MELS |
| [`DensityArchive`](@ref) | those of the archive it wraps | DDS |
| [`GradientArchive`](@ref) | those of the archive it wraps | OMG-MEGA |
| [`ParetoGridArchive`](@ref) | a grid, with a Pareto front in each cell | several objectives |

The algorithm constructors put a descriptor outside the bounds of the grid in the
nearest cell, as pyribs and QDax do, so that a random initial population never stops a
run. An archive built directly, such as `GridArchive(prob; dims)`, throws instead,
which catches a descriptor function that does not match its bounds. Both take
`outside=ThrowOutside()`, `outside=ClampOutside()` or `outside=RejectOutside()`, which
discards the solution.

## Variation

| Operator | Parents | For |
|---|---|---|
| [`GaussianMutation`](@ref) | 1 | real vectors |
| [`IsoLineVariation`](@ref) | 2 | real vectors; moves along the line between two elites (Vassiliades and Mouret, 2018) |
| [`PolynomialMutation`](@ref) | 1 | real vectors within bounds |
| [`UniformCrossover`](@ref) | 2 | vectors of any element type |

Any other genotype needs an operator of its own; see the
[graph tutorial](../tutorials/graphs.md).

## Combining emitters

[`QDAlgorithm`](@ref) takes any archive and any tuple of emitters, for example a
Gaussian and an iso-line emitter side by side:

```julia
archive = GridArchive(prob; dims=(50, 50))
emitters = (VariationEmitter(GaussianMutation(0.1); batch_size=50),
            VariationEmitter(IsoLineVariation(0.01, 0.2); batch_size=50))
alg = QDAlgorithm(archive, emitters)
```

All emitters run every round. A [`BanditScheduler`](@ref) instead holds a pool of
emitters and runs the most successful few each round.
