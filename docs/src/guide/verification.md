```@meta
CurrentModule = QualityDiversity
```

# Verification

This page lists, for every component, how it is checked against code written by others:
[pyribs](https://pyribs.org) 0.12.0, [QDax](https://github.com/adaptive-intelligent-robotics/QDax)
0.5.0, [pymoo](https://pymoo.org), [MOME-PGX](https://github.com/adaptive-intelligent-robotics/MOME_PGX),
or the equations of the paper. Where the result differs, the table says why. Items that
need more work link to an issue.

## Methods

The table names the strongest check that passes.

| Method | What is compared |
|---|---|
| **Reference** | The same scripted inputs go through the other library and through this package, and the outputs agree exactly or to rounding: statuses, improvements, kept solutions, cells, metrics, children. A Python script in [`test/reference`](https://github.com/wclaerhout/QualityDiversity.jl/tree/main/test/reference) writes the other library's outputs, and a test compares them on every run of the test suite. |
| **Paper** | The update equations of the paper, written independently in NumPy (`test/reference/generate_strategies.py`), run on fixed inputs. Used where a library departs from the paper. |
| **Distribution** | For an operator that draws random numbers, which no two libraries draw alike: the mean, standard deviation, quantiles and correlation of 200 000 children of fixed parents. |
| **Whole run** | A full run in two libraries with the same settings, in [`benchmark/compare`](https://github.com/wclaerhout/QualityDiversity.jl/tree/main/benchmark/compare); coverage and QD score agree within the spread between seeds. This catches gross errors, not small ones. |
| **Internal** | Two paths of this package agree: device and host, ForwardDiff and closed-form gradients. |

## Archives

| Component | Compared with | Method | Result |
|---|---|---|---|
| [`GridArchive`](@ref) | pyribs `GridArchive`, QDax `MapElitesRepertoire` | Reference | Statuses, improvements, cells and kept solutions after every batch equal pyribs; kept solutions equal QDax. |
| [`CVTArchive`](@ref) | pyribs `CVTArchive`, QDax `get_cells_indices` | Reference | The same cell for every descriptor (pyribs clips descriptors to the bounds first, QDax does not). The k-means centroids are not compared yet: [#84](https://github.com/wclaerhout/QualityDiversity.jl/issues/84). |
| [`CategoricalArchive`](@ref) | pyribs `CategoricalArchive` (QDax has none) | Reference | Equal. |
| [`SlidingBoundariesArchive`](@ref) | pyribs `SlidingBoundariesArchive` (QDax has none) | Reference | Equal up to fitness ties and pyribs's tolerance of 1e-6 below a boundary. pyribs 0.12.0 clips descriptors to the old bounds while it re-adds solutions during a remap; the reference corrects this. |
| [`ThresholdArchive`](@ref) | pyribs `GridArchive` with `learning_rate` | Reference | Equal. |
| [`UnstructuredArchive`](@ref) | pyribs `ProximityArchive` (one neighbor, local competition); QDax `UnstructuredRepertoire` | Reference (pyribs) | Equal to pyribs, one solution at a time. QDax keeps far more solutions, closer together than its threshold: [#81](https://github.com/wclaerhout/QualityDiversity.jl/issues/81). |
| [`DNSArchive`](@ref) | pyribs `DNSArchive`, QDax `DominatedNoveltyRepertoire` | Reference | Equal to both. |
| [`DensityArchive`](@ref) | pyribs `DensityArchive` (QDax has none) | Reference | Equal densities. A whole DDS run differs: see Algorithms. |
| [`ParetoGridArchive`](@ref) | QDax `MOMERepertoire`, MOME-PGX, pymoo | Reference | Equal fronts and hypervolumes. By default, solutions with equal fitness vectors are not stored twice ([`RejectDuplicates`](@ref QualityDiversity.RejectDuplicates)); [`KeepDuplicates`](@ref QualityDiversity.KeepDuplicates) gives QDax's behavior. |
| [`SpreadArchive`](@ref) | QDax `MELSRepertoire` | Reference | Equal. |
| [`GradientArchive`](@ref) | through OMG-MEGA, below | Reference | Equal. |
| [`DeviceGridArchive`](@ref), [`DeviceCVTArchive`](@ref) | the host archives | Internal | The same cells as on the host, ties included. |

## Variation

| Component | Compared with | Method | Result |
|---|---|---|---|
| [`GaussianMutation`](@ref) | pyribs `GaussianOperator` | Distribution | Same formula; distributions agree. |
| [`IsoLineVariation`](@ref) | pyribs `IsoLineOperator`, QDax `isoline_variation` | Distribution | Same formula; distributions agree with both. |
| [`PolynomialMutation`](@ref) | pymoo `mut_pm` (Deb's form); QDax `polynomial_mutation` | Reference (pymoo) | Equal to pymoo to 1e-13. QDax differs: it uses `(x - lo)/span` where Deb uses `1 - (x - lo)/span`, draws its uniforms in `Float32`, and mutates exactly `⌊proportion·n⌋` entries instead of each with probability `proportion`. |
| [`UniformCrossover`](@ref) | QDax `polynomial_crossover` | Distribution | QDax draws `⌊proportion·n⌋` positions with replacement, so with proportion 0.5 it changes 41% of 10 entries; this package takes each entry from the second parent with probability `proportion`. |

## Emitters and evolution strategies

| Component | Compared with | Method | Result |
|---|---|---|---|
| [`VariationEmitter`](@ref) | pyribs, QDax | Whole run | MAP-Elites on Rastrigin agrees with both at every batch size. |
| [`CMAEmitter`](@ref): rankers, selection, restarts | pyribs `EvolutionStrategyEmitter` | Reference | Statuses, ranking keys, parent counts and restarts equal for six rankers, both selection rules and three restart rules. Candidates with equal keys are ordered differently: this package puts the earlier one first, pyribs the later one (and, for the objective ranker, an arbitrary one). |
| [`CMAES`](@ref QualityDiversity.CMAES) | the CMA-ES tutorial (Hansen, 2016) | Paper | Equal. pyribs's CMA-ES departs from the tutorial in five places: the rank-one weight `c1²`, an evolution path `p_c` not divided by `σ`, a step-size update `(‖p_σ‖²/n − 1)/2`, another `h_σ` test, and a lazily updated eigendecomposition. |
| [`PyribsCMAES`](@ref QualityDiversity.PyribsCMAES) | pyribs `CMAEvolutionStrategy` | Whole run | Reaches pyribs's coverage in CMA-MAE runs, but copies only the `c1²` of the five differences, so its state differs from pyribs's after a few generations: [#79](https://github.com/wclaerhout/QualityDiversity.jl/issues/79). |
| [`SepCMAES`](@ref QualityDiversity.SepCMAES) | Ros and Hansen (2008) with pycma's constants | Paper | Equal. pyribs's sep-CMA-ES uses other learning rates than pycma, besides the differences of its CMA-ES: [#83](https://github.com/wclaerhout/QualityDiversity.jl/issues/83). |
| [`LMMAES`](@ref QualityDiversity.LMMAES) | the paper; pyribs `LMMAEvolutionStrategy` | Reference | Equal to both. |
| [`OpenAIES`](@ref QualityDiversity.OpenAIES) | the paper; pyribs `OpenAIEvolutionStrategy` | Reference | Equal to both. |
| [`BanditScheduler`](@ref) | pyribs `BanditScheduler` | Reference | The same choice in all 230 rounds without a tie. |
| [`GradientArborescenceEmitter`](@ref) | pyribs `GradientArborescenceEmitter` | Reference | Equal branches, rankings, parent counts and points, with Adam and with gradient ascent. |
| [`GradientOperatorEmitter`](@ref) | pyribs `GradientOperatorEmitter`, QDax `OMGMEGAEmitter` | Reference | Equal children for the same parents and coefficients. pyribs also adds Gaussian noise to every parent before the gradient step (its `sigma`), which the paper and QDax do not; QDax's `sigma_g` is a variance. |
| [`DeviceVariationEmitter`](@ref QualityDiversity.DeviceVariationEmitter), [`DeviceCMAEmitter`](@ref QualityDiversity.DeviceCMAEmitter) | QDax on a GPU; the host emitters | Whole run, Internal | Agree; the GPU runs are to be repeated with the current version: [#85](https://github.com/wclaerhout/QualityDiversity.jl/issues/85). |

## Problems and metrics

| Component | Compared with | Method | Result |
|---|---|---|---|
| [`Rastrigin`](@ref QualityDiversity.Benchmarks.Rastrigin), [`BiSphere`](@ref QualityDiversity.Benchmarks.BiSphere), [`Sphere`](@ref QualityDiversity.Benchmarks.Sphere) | the definitions in the comparison scripts | Reference | Equal. |
| [`Sampled`](@ref) | QDax `average`, `median` | Reference | Equal. |
| [`qd_score`](@ref), [`coverage`](@ref), [`max_fitness`](@ref), [`mean_fitness`](@ref) | pyribs archive statistics, QDax `default_qd_metrics` | Reference | Equal. pyribs subtracts its `qd_score_offset` from every fitness and QDax adds its `qd_offset`. |
| [`cqd_score`](@ref) | pyribs `cqd_score` | Reference | Equal. |
| [`hypervolume`](@ref) | pymoo | Reference | Equal to 1e-15. |
| Gradients from ForwardDiff | closed forms of `Sphere` and `Rastrigin` | Internal | Equal. |

## Algorithms

A whole run per algorithm against the library that has it, with the settings of both set
equal.

| Algorithm | Compared with | Result |
|---|---|---|
| [`MAPElites`](@ref) | pyribs, QDax | Agree. |
| [`CMAME`](@ref) | pyribs, QDax | Agree with pyribs for improvement, optimizing and random direction emitters. QDax recombines and restarts differently and reaches other coverage. |
| [`CMAMAE`](@ref) | pyribs | Lower coverage than pyribs, from pyribs's covariance update; with `strategy=PyribsCMAES` it agrees. |
| [`CMAMEGA`](@ref) | pyribs | Agree. |
| [`MOME`](@ref) | QDax | Agree, with equal global hypervolume. |
| [`MELS`](@ref) | QDax | QDax reaches slightly higher coverage and lower best fitness: [#82](https://github.com/wclaerhout/QualityDiversity.jl/issues/82). |
| [`DDS`](@ref) | pyribs | Lower coverage than pyribs: [#80](https://github.com/wclaerhout/QualityDiversity.jl/issues/80). |

The default settings of the algorithm constructors follow pyribs's emitter defaults, not
always those of the papers: [#86](https://github.com/wclaerhout/QualityDiversity.jl/issues/86).
