<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-wordmark-dark.png">
    <img alt="QualityDiversity.jl" src="docs/src/assets/logo-wordmark.png" width="400">
  </picture>
</p>

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://wclaerhout.github.io/QualityDiversity.jl/dev/)
[![CI](https://github.com/wclaerhout/QualityDiversity.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/wclaerhout/QualityDiversity.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/wclaerhout/QualityDiversity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/wclaerhout/QualityDiversity.jl)

Quality-diversity (QD) optimization in Julia: MAP-Elites and the algorithms built from
the same parts, such as CMA-ME, CMA-MAE, MOME and Dominated Novelty Search.

A QD algorithm does not look for one best solution. It looks for the best solution of
every kind, where the kind of a solution is given by a few numbers called its
descriptor, and returns an archive of them.

QualityDiversity.jl works with **any genotype you can write as a Julia type**: vectors
of numbers, and also graphs, molecules, programs or recipes. You define a function
that scores and describes a solution and, for a new genotype, a way to vary it; the
library compiles specialized code for it. Python's QD libraries,
[pyribs](https://pyribs.org) and [QDax](https://github.com/adaptive-intelligent-robotics/QDax),
store solutions as arrays of numbers, which makes such genotypes slow or impossible.

The package is under development and not yet registered.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/wclaerhout/QualityDiversity.jl")
```

Julia 1.10 or later.

## Quick start

```julia
using QualityDiversity, QualityDiversity.Benchmarks
using Random: Xoshiro

prob = Rastrigin(10)                                  # a benchmark with 10 parameters
alg = MAPElites(prob; dims=(20, 20), variation=GaussianMutation(0.3))
result = solve(prob, alg; budget=100_000, rng=Xoshiro(1))

coverage(result.archive), max_fitness(result.archive)
```

For a problem of your own, wrap a function that returns the fitness (maximized) and
the descriptor:

```julia
f(x) = (-sum(abs2, x), clamp.((x[1], x[2]), -2, 2))
prob = QDProblem(f; init=rng -> randn(rng, 5), bounds=((-2, 2), (-2, 2)))
result = solve(prob, MAPElites(prob; dims=(20, 20)); budget=50_000, rng=Xoshiro(1))
```

Switching algorithm is one argument:

```julia
result = solve(prob, CMAME(prob; dims=(20, 20), sigma0=0.5); budget=50_000, rng=Xoshiro(1))
```

With a Makie backend loaded, `heatmap(result.archive)` draws the archive (`scatter` for an
`UnstructuredArchive`); with Tables.jl loaded, `DataFrame(result.archive)` lists its elites.

### A custom genotype

Any type works as a genotype: the archive stores whatever type `init` returns. Define a
variation operator for it. Here `graph_prob` is a `QDProblem` whose function scores a
graph and whose `init` draws a random one:

```julia
using Graphs

struct EdgeFlip <: AbstractVariation end

function QualityDiversity.vary(rng, ::EdgeFlip, parent::SimpleGraph{Int})
    child = copy(parent)
    a, b = rand(rng, vertices(child), 2)
    a == b || (has_edge(child, a, b) ? rem_edge!(child, a, b) : add_edge!(child, a, b))
    return child
end

alg = MAPElites(graph_prob; dims=(20, 20), variation=EdgeFlip())
```

The [graph tutorial](https://wclaerhout.github.io/QualityDiversity.jl/dev/tutorials/graphs/)
works through a complete example.

## What is included

| | |
|---|---|
| Algorithms | MAP-Elites, CMA-ME, CMA-MAE, CMA-MEGA, CMA-MAEGA, OMG-MEGA, Density Descent Search, MOME, MAP-Elites with low spread |
| Archives | grid, CVT, sliding boundaries, categorical, threshold (CMA-MAE), unstructured (novelty), dominated novelty, density, spread, gradient (OMG-MEGA), Pareto grid |
| Emitters | variation emitter, CMA emitter (CMA-ES, sep-CMA-ES, LM-MA-ES, OpenAI-ES, with the rankers and restart rules of pyribs), gradient arborescence and gradient operator emitters, bandit scheduler |
| Variation | Gaussian, iso-line, polynomial (bounded), uniform crossover; your own for any type |
| Problems | function wrappers, noisy problems with averaged or spread-aware evaluation, benchmarks |
| Metrics | QD score, coverage, continuous QD score, hypervolume and MOQD score |
| Extensions | Makie plots (heatmaps, scatter, ECDF, CQD curve); archives as Tables.jl tables; MAP-Elites and CMA-ME with the archive on a GPU; gradients by ForwardDiff |

Every run is reproducible from its seed, also with threaded evaluation. Several archives
and emitters are checked against pyribs and QDax on the same inputs.

## Speed

![Median time of MAP-Elites, CMA-ME, CMA-MAE and MOME on one CPU core and of MAP-Elites on a GPU, for QualityDiversity.jl, pyribs and QDax](docs/src/assets/speed-algorithms.svg)

`benchmark/compare` runs the same problems in QualityDiversity.jl, pyribs and QDax; its
[README](benchmark/compare/README.md) has the settings and results. On one CPU core,
MAP-Elites on Rastrigin is 2 to 14 times faster than pyribs and CMA-ME 5 to 7 times
faster. QDax, with a grid index in place of its centroid search, is faster in `float32`
from a batch of 1000 up; in `float64` QualityDiversity.jl is faster at every batch size.
On a GPU at a batch of 10 000, MAP-Elites with a grid is level with QDax in `float32` and
1.35 times slower in `float64`; with a CVT archive QDax is 2.7 times faster in `float32` and
2.5 times slower in `float64`.

## Documentation

The [documentation](https://wclaerhout.github.io/QualityDiversity.jl/dev/) has
tutorials, a guide to choosing an algorithm, how to write your own problem, variation,
emitter or archive, and the API reference.

## License

MIT; see [LICENSE](LICENSE).
