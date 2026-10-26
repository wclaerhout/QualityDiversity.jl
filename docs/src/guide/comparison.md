```@meta
CurrentModule = QualityDiversity
```

# Coming from pyribs or QDax

[pyribs](https://pyribs.org) and [QDax](https://github.com/adaptive-intelligent-robotics/QDax)
are the two main quality-diversity libraries in Python. This package offers most of
their algorithms, under the names below, and checks several of them against those
libraries on the same inputs (`test/reference`).

## Names

| QualityDiversity.jl | pyribs | QDax |
|---|---|---|
| [`GridArchive`](@ref) | `GridArchive` | `MapElitesRepertoire` with grid centroids |
| [`CVTArchive`](@ref) | `CVTArchive` | `MapElitesRepertoire` with CVT centroids |
| [`SlidingBoundariesArchive`](@ref) | `SlidingBoundariesArchive` | — |
| [`CategoricalArchive`](@ref) | `CategoricalArchive` | — |
| [`UnstructuredArchive`](@ref) | `ProximityArchive` | `UnstructuredRepertoire` |
| [`DNSArchive`](@ref) | `DNSArchive` | `DominatedNoveltyRepertoire` |
| [`ThresholdArchive`](@ref) | `GridArchive(learning_rate=...)` | — |
| [`DensityArchive`](@ref) | `DensityArchive` | — |
| [`SpreadArchive`](@ref) | — | `MELSRepertoire` |
| [`ParetoGridArchive`](@ref) | — | `MOMERepertoire` |
| [`VariationEmitter`](@ref) | `GaussianEmitter`, `IsoLineEmitter` | `MixingEmitter` |
| [`CMAEmitter`](@ref) | `EvolutionStrategyEmitter` | `CMAEmitter` and its subclasses |
| [`BanditScheduler`](@ref) | `BanditScheduler` | — |
| [`CMAES`](@ref QualityDiversity.CMAES), [`SepCMAES`](@ref QualityDiversity.SepCMAES), [`LMMAES`](@ref QualityDiversity.LMMAES), [`OpenAIES`](@ref QualityDiversity.OpenAIES) | `CMAEvolutionStrategy`, `SeparableCMAEvolutionStrategy`, `LMMAEvolutionStrategy`, `OpenAIEvolutionStrategy` | `CMAES` |
| [`ImprovementRanker`](@ref QualityDiversity.ImprovementRanker), [`TwoStageImprovementRanker`](@ref QualityDiversity.TwoStageImprovementRanker), … | `ImprovementRanker`, `TwoStageImprovementRanker`, … | emitter subclasses |
| [`solve`](@ref) | a loop over `scheduler.ask()` and `scheduler.tell()` | a loop over `map_elites.update`, or `scan_update` |
| [`cqd_score`](@ref) | `cqd_score` | — |
| `heatmap(archive)` (Makie) | `grid_archive_heatmap`, `cvt_archive_heatmap` | `plot_2d_map_elites_repertoire` |
| `ecdfplot(archive)` (Makie) | `archive_ecdf` | — |

Where the libraries differ in a rule, such as how a batch is inserted or how ties
are broken, the docstring of the type says which rule this package follows and why.

## Differences in approach

- **Genotypes.** pyribs and QDax store solutions as arrays of numbers. Here a solution
  can be any Julia type, such as a graph or a molecule, with a `vary` method for it; see
  the [graph tutorial](../tutorials/graphs.md).
- **Fitness is maximized,** as in both libraries.
- **One loop.** An algorithm is an archive and a tuple of emitters, run by
  [`solve`](@ref) or one round at a time with [`step!`](@ref).
- **Randomness.** Every function that draws random numbers takes the generator, and a
  run is reproducible from its seed, also with several threads.

## Speed

`benchmark/compare` runs the same problems in all three libraries: MAP-Elites, CMA-ME
and CMA-MAE on Rastrigin with 20 parameters and a 100 × 100 grid, MOME, and MAP-Elites
and CMA-ME on a GPU. Its
[README](https://github.com/wclaerhout/QualityDiversity.jl/blob/main/benchmark/compare/README.md)
gives the settings, the data of every run and how to reproduce them. In short, on one CPU
core:

- MAP-Elites is 2 to 14 times faster than pyribs, most at small batches. It is faster
  than QDax at batches of 100 and below; at 1000 and above QDax, compiled with JAX and
  with a grid lookup in place of its centroid search, is 1.2 to 1.35 times slower in
  `float64` and 1.3 to 1.7 times faster in `float32`;
- CMA-ME is 5 to 7 times faster than pyribs and CMA-MAE 6 times;
- on a GPU at a batch of 10 000, MAP-Elites with a grid is level with QDax in `float32` and
  1.35 times slower in `float64`; with a CVT archive QDax is 2.7 times faster in `float32`
  and 2.5 times slower in `float64`.

![MAP-Elites: time against batch size on one CPU core and on a GPU, in float64 and float32](../assets/speed-map-elites.svg)

![Median time of MAP-Elites, CMA-ME, CMA-MAE and MOME on one CPU core and of MAP-Elites on a GPU, for QualityDiversity.jl, pyribs and QDax](../assets/speed-algorithms.svg)

The objective in these benchmarks takes nanoseconds, so the numbers measure the library.
With an expensive objective the evaluation dominates and the libraries differ little.
