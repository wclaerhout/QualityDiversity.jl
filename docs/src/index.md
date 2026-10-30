# QualityDiversity.jl

Quality-diversity (QD) optimization in Julia: MAP-Elites and the algorithms built from
the same parts, such as CMA-ME, CMA-MAE, MOME and Dominated Novelty Search.

A QD algorithm does not look for one best solution. It looks for the best solution of
every kind, where the kind of a solution is given by a few numbers called its
descriptor, and returns an archive of them: the most efficient network for each
density and clustering, the fastest gait for each way of using the legs.

QualityDiversity.jl works with **any genotype a user can write as a Julia type**:
vectors of numbers, and also graphs, molecules, programs or recipes. You define the
type, a function that scores and describes it, and a way to vary it; the library
compiles specialized code for it. Real vectors are supported throughout and get
good defaults.

## Installation

The package is not yet registered. Install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/wclaerhout/QualityDiversity.jl")
```

It needs Julia 1.10 or later.

## Quick start

```@example index
using QualityDiversity, QualityDiversity.Benchmarks
using Random: Xoshiro

prob = Rastrigin(10)                                  # a benchmark with 10 parameters
alg = MAPElites(prob; dims=(20, 20), variation=GaussianMutation(0.3))
result = solve(prob, alg; budget=100_000, rng=Xoshiro(1))

coverage(result.archive), max_fitness(result.archive)
```

For a problem of your own, wrap a function that returns the fitness (maximized) and
the descriptor:

```@example index
f(x) = (-sum(abs2, x), clamp.((x[1], x[2]), -2, 2))
prob = QDProblem(f; init=rng -> randn(rng, 5), bounds=((-2, 2), (-2, 2)))
result = solve(prob, MAPElites(prob; dims=(20, 20)); budget=50_000, rng=Xoshiro(1))
result.history[end]
```

Switching algorithm is one argument:

```julia
solve(prob, CMAME(prob; dims=(20, 20), sigma0=0.5); budget=50_000)
```

With a Makie backend loaded, `heatmap(result.archive)` draws the archive (`scatter` for an
`UnstructuredArchive`); with Tables.jl loaded, `DataFrame(result.archive)` lists its elites.

## Exported and public names

`using QualityDiversity` brings in the names a script needs: the problem wrappers,
the archives, variations, emitters and algorithms, `solve` and the metrics. The rest of
the interface is public but not exported, so it never clashes with another package and
you write it with a qualified name: the functions you extend for a type of your own, such
as `QualityDiversity.evaluate`, and the values of keyword arguments, such as
`CMAME(prob; ranker=QualityDiversity.ObjectiveRanker(), ...)`. The benchmark problems
live in the submodule `QualityDiversity.Benchmarks`, which exports them:

```julia
using QualityDiversity, QualityDiversity.Benchmarks
prob = Sphere(20)
```

## What is included

- **Algorithms:** MAP-Elites, CMA-ME, CMA-MAE, CMA-MEGA, CMA-MAEGA and OMG-MEGA
  (differentiable quality diversity), Density Descent Search, MOME (several
  objectives), MAP-Elites with low spread (noisy problems), novelty-style archives and
  Dominated Novelty Search, and a bandit scheduler to choose among emitters.
- **Archives:** grid, CVT, sliding-boundaries, categorical, threshold, unstructured,
  dominated-novelty, density, spread, gradient and Pareto archives.
- **Evolution strategies** for the CMA emitters: CMA-ES, sep-CMA-ES, LM-MA-ES and
  OpenAI-ES, with the rankers, selection and restart rules of pyribs.
- **Noisy problems,** evaluated with a generator of their own so that runs stay
  reproducible on any number of threads.
- **Metrics:** QD score, coverage, continuous QD score, and hypervolume for several
  objectives.
- **Plots** through a Makie extension, **tables** of elites through a Tables.jl
  extension, **GPU** archives through a KernelAbstractions extension, and
  **gradients** through a ForwardDiff extension.
- **Conformance checks** in `QualityDiversity.Testing` for the types you define.

## Where to go next

- The [MAP-Elites tutorial](tutorials/rastrigin_tutorial.md) walks through a first run.
- [A custom genotype](tutorials/graphs.md) evolves graphs.
- [Colour pairs and emotions](tutorials/colours.md), an extra example, evolves pairs of colours
  from calm to arousing and from sad to happy.
- [Strange attractors](tutorials/attractors.md), an extra example, evolves quadratic maps of
  the plane into strange attractors, from thin curves to dense folds.
- [Concepts](guide/concepts.md) explains the parts of the package and how a run proceeds.
- [Choosing an algorithm](guide/algorithms.md) compares the algorithms and archives.
- [Coming from pyribs or QDax](guide/comparison.md) maps their names to this package.
- [Verification](guide/verification.md) lists how each component is checked against pyribs, QDax and the papers.
