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
| [`CVTArchive`](@ref) | pyribs `CVTArchive`, QDax `get_cells_indices` | Reference | The same cell for every descriptor (pyribs clips descriptors to the bounds first, QDax does not). The k-means centroids are compared by their mean squared distance to 1 000 000 uniform points (100 000 samples, 3 seeds): 1.709e-4 for 1 000 cells in two dimensions, against 1.694e-4 for QDax and 1.725e-4 for pyribs; 1.902e-5, 1.856e-5 and 1.975e-5 for 10 000 cells; 5.08e-2, 5.06e-2 and 5.06e-2 for 1 000 cells in six dimensions. QDax's scikit-learn `KMeans` seeds with greedy k-means++, which explains its lead: scikit-learn with plain k-means++, this package's default, gives 1.900e-5 for 10 000 cells. Its longer iterations explain the six-dimensional gap. pyribs seeds at random samples. [`GreedyKMeansPlusPlus`](@ref QualityDiversity.GreedyKMeansPlusPlus) gives the greedy seeding. |
| [`CategoricalArchive`](@ref) | pyribs `CategoricalArchive` (QDax has none) | Reference | Equal. |
| [`SlidingBoundariesArchive`](@ref) | pyribs `SlidingBoundariesArchive` (QDax has none) | Reference | Equal up to fitness ties and pyribs's tolerance of 1e-6 below a boundary. pyribs 0.12.0 clips descriptors to the old bounds while it re-adds solutions during a remap; the reference corrects this. |
| [`ThresholdArchive`](@ref) | pyribs `GridArchive` with `learning_rate` | Reference | Equal. |
| [`UnstructuredArchive`](@ref) | pyribs `ProximityArchive` (one neighbor, local competition); QDax `UnstructuredRepertoire` | Reference (pyribs) | Equal to pyribs, one solution at a time. QDax 0.5.0 computes wrong nearest-neighbour distances (an extra axis on the mask of empty slots), so of 150 random solutions it keeps 108–124, hundreds of pairs closer than its threshold; reported as [QDax#247](https://github.com/adaptive-intelligent-robotics/QDax/issues/247), with a fix proposed in [QDax#248](https://github.com/adaptive-intelligent-robotics/QDax/pull/248). With that line corrected it keeps 25–27, no two closer than the threshold, and the rest of the difference is by design: QDax rejects a solution whose second-nearest neighbour is also within the threshold, and decides a batch jointly, dropping a solution if a fitter one of the same batch is within the threshold. A plain implementation of these two rules gives the same solutions as corrected QDax. |
| [`DNSArchive`](@ref) | pyribs `DNSArchive`, QDax `DominatedNoveltyRepertoire` | Reference | Equal to both. |
| [`DensityArchive`](@ref) | pyribs `DensityArchive` (QDax has none) | Reference | Equal densities. pyribs's reservoir sampling replaces the wrong elements when it replaces several in one batch; this package follows Li (1994). See DDS under Algorithms. |
| [`ParetoGridArchive`](@ref) | QDax `MOMERepertoire`, MOME-PGX, pymoo | Reference | Equal fronts and hypervolumes. By default, solutions with equal fitness vectors are not stored twice ([`RejectDuplicates`](@ref QualityDiversity.RejectDuplicates)); [`KeepDuplicates`](@ref QualityDiversity.KeepDuplicates) gives QDax's behavior. |
| [`SpreadArchive`](@ref) | QDax `MELSRepertoire` | Reference | Equal. When several solutions of one batch pass against the elite of a cell, QDax writes them all to the cell in one scatter, and an arbitrary one of them remains; this package keeps the one that adding them one at a time keeps. |
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
| [`PyribsCMAES`](@ref QualityDiversity.PyribsCMAES) | pyribs `CMAEvolutionStrategy` | Reference | Equal mean, step size, covariance and paths to 1e-15 over a short run and a 20-generation run, with the eigendecomposition recomputed in the same generations. It copies all five differences; see [`PyribsCMA`](@ref QualityDiversity.PyribsCMA). |
| [`SepCMAES`](@ref QualityDiversity.SepCMAES) | Ros and Hansen (2008) with pycma's constants | Paper | Equal. pyribs's sep-CMA-ES departs from pycma in six places, which account for the whole difference: with all six undone, a copy of pyribs equals this package to 2e-16. Its rank-one rate is `c1 · conedf` where pycma has `conedf`, applied twice, which makes it 10⁶–10¹⁴ times smaller for 20 to 1 000 dimensions; its rank-μ offset is 0 where pycma has 0.25; and it shares the step-size rule, the rank-one weight and the unscaled path `p_c` of its CMA-ES. The `h_σ` test changes nothing in these runs. Asked upstream: [pyribs#731](https://github.com/icaros-usc/pyribs/issues/731). |
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
| [`MELS`](@ref) | QDax | Agree. On noisy 10-dimensional Rastrigin over 20 seeds, coverage 0.283 ± 0.023 against 0.291 ± 0.021 and best fitness −68.1 ± 10.4 against −69.5 ± 8.6, differences of 1.1 and 0.5 standard errors. QDax's rule for several solutions entering one cell makes no measurable difference: with it, this package reaches 0.281 and −69.5. |
| [`DDS`](@ref) | pyribs | With `strategy=PyribsCMAES`, agrees with pyribs once pyribs's reservoir sampling is corrected (coverage 0.89–0.90 against 0.87–0.92 on 20-dimensional Rastrigin); pyribs as released reaches 0.76–0.80. With the default `CMAES`, searches that leave the box of `Rastrigin` seldom restart (coverage 0.66–0.70); with the box as `lower` and `upper`, 0.90–0.92. pyribs also restarts every search after its first round, in which the buffer is empty and every batch has density zero. |

The default settings of the algorithm constructors follow pyribs's emitter defaults, not
always those of the papers; the docstring of each constructor gives the settings of its
paper.
