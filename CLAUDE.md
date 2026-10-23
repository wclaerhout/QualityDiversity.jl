# QualityDiversity.jl — project guide

This file is the design reference and working agreement for this repository.
Read it fully before writing code. When a decision below conflicts with an
instinct, follow the decision; when something is undecided (see
[Open decisions](#open-decisions)), ask the author instead of choosing.

The package name is `QualityDiversity`. It is not registered in the General registry:
as of 2026-10-19 the author is using and testing it first, and decides when to register.

---

## 1. What this package is

A Julia library for quality-diversity (QD) optimization: MAP-Elites and the
algorithms built from the same parts (CMA-ME, MOME, novelty-based archives,
differentiable QD).

**What distinguishes it from pyribs and QDax.** Both Python libraries already
offer interchangeable archives and emitters, and QDax is very fast for
problems that fit JAX: fixed-shape arrays and jittable scoring functions.
Neither handles *structured, variable-size, discrete genotypes* well —
molecules, reaction networks, polyomino assemblies, graphs, syntax trees,
recipes. In Python those become slow object loops; in JAX they need padding
or do not work at all.

This package targets that gap: **fast QD for any genotype a user can write as
a Julia type.** The user defines a type and a few methods; the library
compiles specialized code for it. Continuous vectors are fully supported and
get good defaults, but they are not the only first-class case.

Design goal: simple from the outside, fast on the inside.

- Outside: a user with a fitness function and a descriptor function gets a
  filled archive in a few lines. Switching algorithm is one argument.
- Inside: concretely typed containers, no allocation in the archive hot path,
  no runtime dispatch in the main loop.

Speed matters most when evaluation is cheap (benchmarks, analytic problems,
fast simulators). When evaluation is expensive (calling an ML model, a
chemistry toolkit), the selling points are ergonomics and composability.
Neither should be sacrificed for the other.

### Out of scope (for now)

- Reinforcement-learning environments and neural-policy training loops.
- A plotting backend in the core package (plotting lives in an extension).
- Distributed (multi-node) evaluation. Threaded and batched evaluation are in
  scope.

---

## 2. Design principles

1. **Roles are abstract types; behavior is a small set of functions.** Every
   component plays one of five roles (Problem, Archive, Variation, Emitter,
   Algorithm). Each role has an abstract type and a documented contract of
   required and optional methods. Generic code is written only against those
   contracts.
2. **One loop, many algorithms.** Most QD algorithms are the same
   ask → evaluate → add → tell loop with a different archive and emitters.
   There is one generic loop. `MAPElites(...)`, `CMAME(...)` and friends are
   convenience constructors that assemble components, not separate loops.
3. **Dispatch, not flags.** Algorithm variants are chosen by passing a
   different type, never by a `Symbol` or `Bool` switch.
4. **Type parameters carry the hot-path types.** Archives are parametrized by
   genotype `G`, fitness `F`, descriptor `D` and phenotype `P` (`Nothing` when
   none is stored), so everything downstream is concretely typed and batch
   outputs can be preallocated.
5. **Batch-first.** Emitters produce batches; evaluation is batched. This is
   the hook for threading, GPU, and calls out to Python.
6. **Explicit randomness.** Every function that draws random numbers takes an
   `AbstractRNG` as its first argument. A constructor that keeps or uses a generator
   (`CVTArchive`, `DeviceCVTArchive`, `DensityArchive`, `DDS`) takes it as the keyword
   `rng`, by default `Random.default_rng()`, as Clustering.jl's `kmeans` and Agents.jl do;
   its docstring says to seed it, since the seed of `solve` does not. A run is
   reproducible from its seeds.
7. **Fail fast.** Invalid input throws an error with a message that tells the
   user what is wrong. Never silently clamp, skip, or continue. A descriptor outside
   the bounds is not invalid input: the algorithm constructors clamp it by default, as
   pyribs and QDax do, and `outside=ThrowOutside()` restores the check (see Decisions).
8. **Interface functions are `public`, not exported.** Users extend them with
   qualified names (`QualityDiversity.evaluate(p::MyProblem, x) = ...`), as
   with `Base.show`. This avoids clashes such as `evaluate` (Distances.jl) and
   `sample` (StatsBase.jl). Exported are what a script uses: the five role types,
   `Evaluation`, `QDAlgorithm`, the problem wrappers, archives, outside policies,
   variations, emitters, algorithm constructors, the CommonSolve verbs (`init`, `step!`,
   `solve!`, `solve`) and the main metrics. Keyword values (strategies, rankers,
   selection, restart and reselect rules, Pareto policies, `Average`/`Median`, the
   gradient optimizers), result and state types, the device emitters and the secondary
   metrics are `public` only, as are abstract types that are only subtyped. The
   benchmark problems live in the public submodule `QualityDiversity.Benchmarks`, which
   exports them. `test/test_contracts.jl` pins the export list.

---

## 3. Core abstractions

This is the target shape of `src/interfaces.jl` and `src/evaluation.jl`.
Signatures are normative. `GridArchive`, `VariationEmitter` and the loop
implement them; the other concrete types in the layout follow the roadmap.

```julia
# ── Results of one evaluation ─────────────────────────────
struct Evaluation{F,D,P}
    fitness::F        # Real (single objective) or AbstractVector (multi-objective)
    descriptor::D     # usually SVector{N,Float64}
    phenotype::P      # `nothing` unless the problem returns one
end
Evaluation(fitness, descriptor) = Evaluation(fitness, descriptor, nothing)

@enum AddStatus Rejected Improved NewCell

struct AddResult{F}
    status::AddStatus
    improvement::F    # fitness gain over the previous occupant (or over the
                      # archive's threshold for a new cell); used by CMA-ME ranking
end

# ── Problem ───────────────────────────────────────────────
abstract type AbstractQDProblem end

# ── Archive ───────────────────────────────────────────────
abstract type AbstractArchive{G,F,D,P} end

# ── Variation ─────────────────────────────────────────────
abstract type AbstractVariation end

# ── Emitter ───────────────────────────────────────────────
abstract type AbstractEmitter end

# ── Algorithm, state, result ──────────────────────────────
struct QDAlgorithm{A<:AbstractArchive,E<:Tuple}
    archive::A
    emitters::E
end
```

### Contracts

Fitness is **maximized**. Users who minimize return the negated value.

**`AbstractQDProblem`**

| Function | Status | Meaning |
|---|---|---|
| `evaluate(prob, x) -> Evaluation` | required | fitness and descriptor of one solution |
| `random_solution(rng, prob)` | required | one random genotype, for initialization |
| `descriptor_bounds(prob)` | required, unless categories | per-dimension `(lo, hi)` tuples |
| `descriptor_categories(prob)` | alternative to bounds | per-dimension collections of discrete descriptor values; descriptors are then tuples such as `(3, :ring)` |
| `evaluate_batch(prob, xs) -> Vector{Evaluation}` | optional | default: threaded loop over `evaluate` |
| `evaluate_gradient(prob, x) -> (Evaluation, jacobian)` | optional | for differentiable QD: the `(N+1) × n` Jacobian of fitness and descriptor; default by ForwardDiff (extension) |
| `evaluate_gradient_batch(prob, xs)` | optional | default: threaded loop over `evaluate_gradient` |

`QDProblem(f; init, bounds)` (or `categories` instead of `bounds`) wraps a plain function `f(x) -> (fitness,
descriptor)`, or `(fitness, descriptor, phenotype)`, so most users never subtype.

`AbstractStochasticQDProblem <: AbstractQDProblem` is for noisy evaluation. It defines
`evaluate(rng, prob, x)` instead of `evaluate(prob, x)`, and `evaluate_batch(rng, prob, xs)`
draws one seed per solution from the run's generator, in the order of the batch, and gives
each solution its own `Xoshiro` before the threaded section, so a run is reproducible from
its seed whatever the number of threads. Deterministic problems do not consume the run's
generator. `StochasticQDProblem(f; init, bounds)` wraps `f(rng, x)`, and
`Sampled(prob; num_samples, expectation=Average())` scores a solution by the average (or
`Median()`) of `num_samples` evaluations. `evaluations_per_solution(prob)` is how many
evaluations one solution costs; the budget of a run counts them.

**`AbstractArchive{G,F,D}`**

| Function | Status | Meaning |
|---|---|---|
| `add!(archive, x, ev) -> AddResult` | required | try to insert one solution |
| `cell_index(archive, d)` | required | cell for descriptor `d`; must not allocate |
| `sample_elites(rng, archive, n) -> Vector{G}` | required | `n` parents, uniform over occupied cells by default |
| `elites(archive)` | required | iterator over `(genotype, evaluation)` of occupied cells |
| `Base.length(archive)` | required | number of occupied cells |
| `capacity(archive)` | required | number of cells |
| `add_batch!(archive, xs, evs)` | optional | default: loop over `add!`; the cell archives and `ThresholdArchive` compute every result against the archive as it was before the batch |
| `result_archive(archive)` | optional | the archive of the best solutions found, which `solve` returns and whose metrics a run records; default: the archive itself |
| `Base.empty!(archive)` | optional | |

Concrete archives: `GridArchive`, `SlidingBoundariesArchive` (a grid whose boundaries
follow the distribution of recent descriptors), `CVTArchive`, `CategoricalArchive` (discrete
descriptors, numbers or labels), `ThresholdArchive` (CMA-MAE thresholds around
a cell archive, with a result archive of the best solutions), `UnstructuredArchive` (novelty-style,
distance threshold), `DNSArchive` (Dominated Novelty Search: a population chosen by
distance to fitter neighbours), `DensityArchive` (Density Descent Search: the density of
recent descriptors around a result archive), `GradientArchive` (OMG-MEGA: the gradients of
every elite next to a cell archive), `ParetoGridArchive` (MOME: a Pareto front of at most `max_front` solutions in every cell of a grid, for fitness vectors).

Storage layout for cell-based archives (`GridArchive`, `SlidingBoundariesArchive`,
`CVTArchive`, `CategoricalArchive`, subtypes of `CellArchive`, which holds the shared code) is struct-of-arrays:
`occupied::BitVector`, `fitness::Vector{F}`, `descriptors::Vector{D}`,
`genotypes::Vector{G}`, `phenotypes::Vector{P}` (unassigned slots left `#undef`
and guarded by `occupied`), plus `occupied_idx::Vector{Int}` so a uniform parent
draw is O(1).

**`AbstractVariation`**

| Function | Status | Meaning |
|---|---|---|
| `vary(rng, op, parent)` | required for mutations | one child |
| `vary(rng, op, p1, p2)` | required for crossovers | one child |
| `n_parents(op)` | required for crossovers | `1` (default) or `2`; `VariationEmitter` draws that many parents per child |

Defaults for `AbstractVector{<:Real}` genotypes: `GaussianMutation(σ)`,
`IsoLineVariation(σ_iso, σ_line)`, `PolynomialMutation(lower, upper)` for
bounded genotypes, and `UniformCrossover(proportion)`. A user with a custom genotype defines
`vary` for their own operator type.

**`AbstractEmitter`**

| Function | Status | Meaning |
|---|---|---|
| `ask(rng, emitter, archive) -> Vector{G}` | required | a batch of candidates |
| `tell!(emitter, xs, evs, results)` | optional | update internal state; default no-op |
| `batch_size(emitter)` | required | |
| `restarts(emitter)` | optional | number of times the emitter has restarted; default `nothing`, for an emitter that never restarts |
| `ask_gradients(rng, emitter, archive)` | optional | points whose gradients the emitter needs before `ask`; default `nothing` |
| `tell_gradients!(emitter, xs, evs, jacobians, results)` | required if `ask_gradients` returns points | |

Concrete emitters: `VariationEmitter(op; batch_size)` (select parents
uniformly, apply `op`) and `CMAEmitter` (a search distribution whose update
is driven by a ranker, a selection rule and a restart rule, all types chosen by
dispatch; the optimizer is an `AbstractEvolutionStrategy`: `CMAES`, `SepCMAES`,
`LMMAES` or `OpenAIES`), `GradientArborescenceEmitter` (CMA-MEGA: a CMA-ES search over
the coefficients of the gradients at a point that moves by `AdamAscent` or `GradientAscent`),
`GradientOperatorEmitter` (OMG-MEGA: a random gradient step from an elite of a
`GradientArchive`), and `BanditScheduler` (a pool of emitters of which a bandit
runs a few each round). An emitter with state may require `tell!` to follow the `ask` that
produced the batch.

**Algorithm and state** — CommonSolve.jl interface:

```julia
state = init(prob, alg; rng, budget, callback)   # -> QDState
step!(state)                                     # one ask/evaluate/add/tell round
solve!(state)                                    # run to budget
result = solve(prob, alg; kwargs...)             # init + solve!, returns QDResult
```

`QDState` holds the problem, a working copy of the algorithm's archive and
emitters, the RNG, the evaluation count, and the metrics history. Stepping
one iteration at a time is a supported use, not a debugging hack.

### The loop

```julia
function step!(state)
    for emitter in state.emitters          # tuple: unrolled, type-stable
        # an emitter that needs gradients gets its points evaluated with them first
        xs  = ask(state.rng, emitter, state.archive)
        evs = evaluate_batch(state.problem, xs)
        rs  = add_batch!(state.archive, xs, evs)
        tell!(emitter, xs, evs, rs)
        state.n_evals += length(xs)
    end
    record_metrics!(state)
    return state
end
```

Initialization fills the archive with `random_solution` draws before the
first step.

### Metrics

`qd_score(archive; offset)`, `coverage(archive)`, `max_fitness(archive)`,
`mean_fitness(archive)`. QD-score needs an offset so negative fitness values
do not reduce the score; the problem may supply a default via
`fitness_offset(prob)` (optional, default `0`). `cqd_score` measures an archive without
reference to cells (see Decisions).

---

## 4. Performance rules

These hold for code in `src/`. Tests in `test/` check them.

- No runtime dispatch in `step!`, `add!`, `cell_index`, `sample_elites`,
  `ask`, or `vary` for the built-in types. Check with `@inferred` and JET.
- `cell_index` and `add!` do not allocate for `isbits` genotypes. Check with
  `@allocated` after warm-up.
- Emitters are stored in a `Tuple`, not a `Vector{AbstractEmitter}`.
- Real-valued descriptors are `SVector`s from StaticArrays.jl. Discrete
  descriptors are tuples whose element types are those of the categories, so
  they are concretely typed too.
- Batch outputs are preallocated with types computed from the archive's type
  parameters, not grown with `push!` into an untyped container.
- No `@inbounds` without profiling evidence; see
  `.claude/rules/julia-inbounds.md`.
- Default `evaluate_batch` uses threads. Leave cores free for the user when
  running benchmarks.

---

## 5. Repository layout

```
QualityDiversity.jl/
├── Project.toml
├── CLAUDE.md                     # this file
├── src/
│   ├── QualityDiversity.jl       # module, exports, `public` declarations, includes
│   ├── interfaces.jl             # abstract types + documented function stubs
│   ├── evaluation.jl             # Evaluation, AddStatus, AddResult
│   ├── hypervolume.jl            # hypervolume of a set of points, any number of objectives
│   ├── gradients.jl              # evaluate_gradient and evaluate_gradient_batch
│   ├── problems/
│   │   ├── qdproblem.jl          # QDProblem wrapper
│   │   ├── stochastic.jl         # StochasticQDProblem, Sampled, SampledSpread
│   │   └── benchmarks.jl         # module Benchmarks: Rastrigin, Sphere, BiSphere
│   ├── archives/
│   │   ├── cell_archive.jl       # CellArchive: shared storage, add!, metrics
│   │   ├── grid.jl
│   │   ├── sliding.jl            # SlidingBoundariesArchive
│   │   ├── nearest_centroid.jl   # k-d tree, k-means and centroid buckets for the CVT archives
│   │   ├── cvt.jl
│   │   ├── categorical.jl
│   │   ├── threshold.jl
│   │   ├── spread.jl             # SpreadArchive for MELS
│   │   ├── pareto.jl             # ParetoGridArchive
│   │   ├── device_cell_archive.jl # DeviceCellArchive: storage and insertion in device arrays
│   │   ├── device_grid.jl        # DeviceGridArchive
│   │   ├── device_cvt.jl         # DeviceCVTArchive
│   │   ├── dns.jl                # DNSArchive
│   │   ├── density.jl            # DensityArchive for DDS
│   │   ├── gradient.jl           # GradientArchive for OMG-MEGA
│   │   └── unstructured.jl
│   ├── variation/
│   │   ├── gaussian.jl
│   │   ├── isoline.jl
│   │   ├── polynomial.jl
│   │   └── crossover.jl         # UniformCrossover
│   ├── emitters/
│   │   ├── variation_emitter.jl
│   │   ├── evolution_strategy.jl # AbstractEvolutionStrategy and what the strategies share
│   │   ├── cma_es.jl             # the CMA-ES optimizer inside CMAEmitter
│   │   ├── sep_cma_es.jl
│   │   ├── lm_ma_es.jl
│   │   ├── openai_es.jl
│   │   ├── rankers.jl
│   │   ├── cma_emitter.jl
│   │   ├── bandit_scheduler.jl   # BanditScheduler
│   │   ├── gradient_optimizers.jl # AdamAscent, GradientAscent: how CMA-MEGA moves its point
│   │   ├── gradient_arborescence.jl # GradientArborescenceEmitter (CMA-MEGA)
│   │   ├── gradient_operator.jl  # GradientOperatorEmitter (OMG-MEGA)
│   │   ├── device_emitter.jl     # DeviceVariationEmitter
│   │   └── device_cma_emitter.jl # DeviceCMAEmitter
│   ├── solve.jl                  # QDAlgorithm, QDState, QDResult, init/step!/solve!
│   ├── algorithms.jl             # MAPElites(), CMAME(), ... constructors
│   ├── metrics.jl
│   └── testing.jl                # QualityDiversity.Testing: interface conformance checks
├── ext/
│   ├── QualityDiversityMakieExt.jl        # archive heatmaps, metric curves
│   ├── QualityDiversityKernelAbstractionsExt.jl  # kernels of the device archive and emitter
│   ├── QualityDiversityForwardDiffExt.jl  # gradients of any problem for differentiable QD
│   ├── QualityDiversityTablesExt.jl      # archives as Tables.jl tables
│   └── QualityDiversityPythonCallExt.jl   # (planned) batched evaluation through Python
├── test/
│   ├── runtests.jl               # includes the files below in a fixed order; some use
│   │                             # helpers that earlier files define
│   ├── test_<component>.jl       # one file per archive, emitter, strategy or problem family:
│   │                             # archives, sliding, cvt, categorical, threshold, spread,
│   │                             # unstructured, dns, density, pareto, variation, emitters,
│   │                             # cma_es, evolution_strategies, cma_emitter, bandit,
│   │                             # problems, stochastic, phenotype, solve, cqd, contracts
│   ├── test_conformance.jl       # runs Testing checks on every built-in type
│   ├── test_inference.jl         # @inferred and allocation checks
│   ├── test_device.jl            # device archive and emitter on the CPU backend and JLArrays
│   ├── test_gradients.jl         # gradients, GradientArchive, the MEGA emitters and algorithms
│   ├── test_graphs.jl, test_gpu_evaluation.jl, test_makie.jl  # examples and extensions
│   ├── test_quality.jl           # Aqua, ExplicitImports; includes test_jet.jl when QD_JET=1
│   ├── test_jet.jl               # JET (CI runs it in one job, since JET breaks on new Julia)
│   └── reference/                # values from pyribs, QDax and MOME-PGX, and the Python
│                                 # scripts that generate them
├── benchmark/
│   ├── Project.toml              # BenchmarkTools lives here, not in the package
│   ├── benchmarks.jl             # BenchmarkTools SUITE
│   ├── run.jl                    # runs the suite and prints a table
│   └── compare/                  # same problem in pyribs and QDax, for the comparison table
│       └── gpu/                  # MAP-Elites on a CUDA GPU; CUDA lives in its own Project.toml
├── docs/                         # Documenter.jl
│   ├── Project.toml              # Documenter, Literate, CairoMakie, Graphs
│   ├── make.jl                   # turns the tutorial scripts into pages, builds and deploys
│   └── src/
│       ├── index.md
│       ├── assets/               # logo (grid, light and dark), wordmark for the README, favicon
│       ├── guide/                # concepts, choosing an algorithm, extending, GPU, comparison
│       └── api/                  # the docstrings, one page per role
└── examples/
    ├── Project.toml              # Graphs.jl and JLArrays for the examples
    ├── rastrigin_tutorial.jl     # step-by-step walkthrough (tutorial)
    ├── gpu_evaluation.jl         # optional: evaluate_batch on a GPU array type
    ├── graphs.jl                 # structured genotype, proves the pitch (tutorial)
    ├── noisy.jl                  # Sampled and MELS on a noisy problem (tutorial)
    ├── mome.jl                   # several objectives (tutorial)
    ├── heatmap.jl                # plots with CairoMakie (tutorial)
    └── colours.jl                # pairs of colours by emotion (extra example)
```

The tutorials of the documentation are the scripts marked "(tutorial)", which
`docs/make.jl` turns into pages with Literate.jl, so the code on a page is the code
that runs. Lines ending in `#src` are left out of the page and lines starting with
`#md` appear only there; comments inside a function start with `##`, since Literate
turns a comment that starts with `# ` into text. The documentation build runs every
tutorial and doctest and fails on a broken cross-reference or on a public name whose
docstring no API page includes.

`QualityDiversity.Testing` is a public submodule. A user who defines a new
problem, archive or emitter calls e.g.
`QualityDiversity.Testing.test_problem(MyProblem())` to check that it meets
the contract, is inferrable, and returns the right types. It lives in `src/`
(not `test/`) so users can call it; it depends only on the `Test` stdlib.

---

## 6. Roadmap

Work through phases in order. Each phase ends with a green test suite on
Julia LTS and the current release.

- [x] **Phase 1 — Contracts.** `interfaces.jl`, `evaluation.jl`, module file
      with exports and `public`. Docstrings for every abstract type and
      contract function. No implementations.
- [x] **Phase 2 — First end-to-end run.** `GridArchive`, `GaussianMutation`,
      `VariationEmitter`, `QDProblem`, `Rastrigin`, `init`/`step!`/`solve`,
      `MAPElites()`. Example `examples/rastrigin_tutorial.jl` fills an archive.
- [x] **Phase 3 — Guarantees.** Metrics, `Testing` submodule, conformance
      tests, inference and allocation tests, Aqua/ExplicitImports/JET.
- [x] **Phase 4 — Benchmarks.** BenchmarkTools suite; comparison with pyribs
      on Rastrigin (same archive shape, batch size, evaluation budget).
- [x] **Phase 5 — Structured genotype.** `examples/graphs.jl` with a
      Graphs.jl genotype, custom `vary`, descriptor from graph statistics.
- [x] **Phase 6 — More algorithms.** One at a time, easiest first. Each comes
      with tests, a conformance check from `Testing`, and a benchmark entry.
      The list follows what pyribs and QDax offer, minus what is out of scope.
  - [x] `IsoLineVariation` and `VariationEmitter` with two parents
  - [x] Bounded variation: `PolynomialMutation` (Deb's bounded form, as in
        NSGA-II and pymoo) with the bounds given to the operator, and
        `UniformCrossover` for vectors of any entry type (QDax calls it
        `polynomial_crossover`)
  - [x] `CVTArchive` (pyribs, QDax): centroids from k-means, nearest-centroid
        lookup with a k-d tree
  - [x] `CategoricalArchive` (pyribs): cells from discrete descriptor values
  - [x] `UnstructuredArchive` (QDax) / `ProximityArchive` (pyribs):
        novelty-style, a solution enters if it is far from its neighbours
  - [x] Threshold archive for CMA-MAE (pyribs `learning_rate`): `ThresholdArchive`,
        a separate type from `GridArchive` that wraps any cell archive
  - [x] `CMAEmitter` with a CMA-ES core, the two-stage improvement, improvement
        and objective rankers, filter and mu selection, and restart rules;
        `CMAME()` and `CMAMAE()`
  - [x] Random-direction rankers (CMA-ME-RD) and the two-stage objective ranker;
        further evolution strategies (sep-CMA-ES, LM-MA-ES, OpenAI-ES)
  - [x] `ParetoGridArchive` and `MOME()` (QDax), with hypervolume metrics
  - [x] `SlidingBoundariesArchive` (pyribs): a grid whose boundaries follow the
        descriptors found
  - [x] `BanditScheduler` for choosing among emitters (pyribs)
  - [x] `DNSArchive` (pyribs, QDax): Dominated Novelty Search
  - [x] `DensityArchive` (pyribs) and `DDS()`: Density Descent Search
  - [x] Stochastic problems: `AbstractStochasticQDProblem` and `Sampled`
  - [x] MAP-Elites with low spread (QDax `MELS`): `SpreadArchive` stores a spread
        for every cell, with `SampledSpread` problems
- [ ] **Phase 7 — Extensions and docs.**
  - [x] Makie: archive heatmaps, archive ECDF, and the plot of the CQD score (the score
        itself is a core metric)
  - [x] Device MAP-Elites: `DeviceGridArchive` and `DeviceVariationEmitter` with
        KernelAbstractions kernels, and a GPU row in the comparison with QDax
  - [x] Device CVT archive: `DeviceCVTArchive`, the nearest centroid found in a kernel
        among the centroids of a bucket, reusing the insertion and variation kernels
  - [x] Device CMA-ME: `DeviceCMAEmitter`, many CMA-ES searches that ask together over a
        device archive, with their updates on the host
  - [ ] Device CMA-MAE: a device `ThresholdArchive` with the batch threshold update of
        the CMA-MAE paper, for `DeviceCMAEmitter`
  - [ ] Device MOME: `ParetoGridArchive` fronts in device arrays, a batch inserted in
        parallel over the cells, with the dominance checks of a cell in parallel too
  - [x] Documenter site with tutorials, deployed to GitHub Pages by
        `.github/workflows/Documentation.yml`
  - [x] Differentiable QD: `evaluate_gradient` with a ForwardDiff extension, CMA-MEGA,
        CMA-MAEGA and OMG-MEGA
  - [ ] PythonCall; Bayesian-optimization emitter

GPU support is optional and never a dependency of the package. Scoring a batch
on a device works through `evaluate_batch` (`examples/gpu_evaluation.jl`,
tested with JLArrays on the CPU). MAP-Elites with the archive, the variation and
the insertion on a device is `DeviceGridArchive` with `DeviceVariationEmitter`,
whose kernels are in the KernelAbstractions extension; see Decisions.

Out of scope, as in the introduction: reinforcement-learning and
neural-policy methods in QDax (PGA-ME, QD-PG, DCRL, ME-ES on policies, DIAYN,
DADS, SMERL, PBT, TD3/SAC), AURORA (descriptors from a trained autoencoder),
and distributed MAP-Elites.

### Next priorities

The open Phase 7 items differ in importance. Pursue these, in this order:

1. **First release, when the author says so.** Set the version to 0.1.0, install with
   `Pkg.add("QualityDiversity")` in the README and the docs, and let the author trigger
   JuliaRegistrator. Create no `v0.1.0` tag before that: TagBot makes it once the
   registry takes the version.
2. **PythonCall extension.** Batched evaluation through Python, for expensive
   scoring with Python tools (ML models, chemistry toolkits).

Device CMA-MAE, device MOME and the Bayesian-optimization emitter are add-ons;
take them up after these.

---

## Decisions

Settled; change them only with the author's agreement.

- **Fitness thresholds.** `GridArchive` is plain MAP-Elites: an empty cell
  accepts any solution, and a `NewCell` reports `improvement == fitness`.
  CMA-MAE thresholds (per-cell threshold with a learning rate) live in
  `ThresholdArchive`, which wraps a cell archive.
- **Batch results.** `add_batch!` of the cell archives and of
  `ThresholdArchive` computes the status and `improvement` of every solution
  against the archive as it was before the batch, as pyribs and QDax do, so that
  emitters rank candidates independently of the order in the batch. Of the
  solutions that enter a cell the best is kept (the first on a tie), so a cell
  archive ends up as if the solutions had been added one at a time.
  `ThresholdArchive` moves a threshold by the batch rule of the CMA-MAE paper.
  Archives without their own `add_batch!` (`UnstructuredArchive`) loop over `add!`.
- **Result archive.** A `ThresholdArchive` replaces the elite of a cell whenever
  a solution passes the threshold, so its elites can get worse (as in CMA-MAE
  and pyribs). It therefore also offers every solution to a plain archive, and
  `result_archive(archive)` returns it; `solve` returns it and a run records its
  metrics, while emitters read the `ThresholdArchive`. With `learning_rate=1` and
  `threshold_min=-Inf` the archive is its own result. `threshold_min=-Inf`
  requires `learning_rate=1`.
- **CMA emitters.** `CMAEmitter` works on real vectors. `tell!` has neither the
  random number generator nor the archive, so it only marks a restart; the next
  `ask` starts the new search from a random elite, drawn with the run's generator.
  Without `x0` the first search starts from a random elite as well, so the archive
  is filled before the first `ask`. `sigma0` has no default. `CMAME` defaults to the
  rules of CMA-ME (two-stage improvement ranking, filter selection, restart after a
  rejected batch) and `CMAMAE` to those of CMA-MAE (improvement ranking, mu
  selection, restart on convergence only) over a `ThresholdArchive`. The optimizer is
  chosen by the `strategy` keyword (a type, `CMAES` by default). A ranker may hold
  state, such as a random direction: an emitter keeps its own copy of its ranker, and
  calls `reset_ranker!(rng, ranker, archive)` whenever a search starts. A batch is flat
  when the whole ranking key of its best and its worst candidate agree, status
  included, as in pyribs and QDax. `PyribsCMAES` is CMA-ES with the rank-one weight `c1²`
  that pyribs computes, for reproducing results produced with pyribs; `CMAES` follows
  the textbook `c1`. `AcceptedRandomDirectionRanker` is the random direction ranking of
  the CMA-ME paper; the two-stage one is that of pyribs.
- **Bandit scheduler.** `BanditScheduler(pool; num_active)` (pyribs; Cully, 2021,
  "Multi-Emitter MAP-Elites") is an emitter that holds a tuple of emitters and runs
  `num_active` of them each round, so the loop does not change. `ask` returns the batches
  of the active emitters one after the other, in the order of the pool; `tell!` hands each
  its part and counts, per emitter, the solutions it produced (selections) and those the
  archive accepted (successes, `NewCell` or `Improved`), over the last `window` rounds (50
  by default, as in the paper; pyribs counts the whole run). Then it chooses the emitters
  of the next round, so `batch_size(scheduler)`, the sum of the batch sizes of the active
  emitters, is known before `ask` and emitters of different batch sizes can share a pool.
  The first round runs the first `num_active` emitters. An emitter leaves the active set
  by the reselection rule, a type: `ReselectTerminated()` (default) when it restarted
  during the round, which `restarts(emitter)` tells, or always if it cannot restart (its
  `restarts` is `nothing`), as in pyribs and the paper; `ReselectAll()` every round. The
  free slots go to the inactive emitters (including the ones that just left) with the
  highest UCB1 score `successes/selections + zeta*sqrt(log(total selections)/selections)`,
  as in Algorithm 1 of the paper, with `zeta = 0.05`; an emitter without selections in the
  window scores `Inf`, and ties go to the earlier emitter of the pool. pyribs takes the
  log of the total successes instead, and takes the last emitter on a tie.
  `test/reference/bandit.jl` holds the emitters that pyribs 0.12.0 runs in each of 230
  rounds of scripted statuses; with the whole run as window, ours makes the same choice in
  every round without a tie, and the log term never changes a choice there. A mutable
  emitter may appear only once in the pool.
- **Dominated novelty.** `DNSArchive(prob; capacity, k)` (Bahlous-Boldi, Faldor et al.,
  2025; pyribs `DNSArchive`, QDax `DominatedNoveltyRepertoire`) keeps a population of at
  most `capacity` solutions without cells. `add_batch!` merges the batch into the
  population and gives every solution its dominated novelty: the mean Euclidean distance
  in descriptor space to its `k` nearest other solutions with fitness at least as high
  (all of them if there are fewer), `Inf` if there is none. A neighbour of equal fitness
  counts, as in pyribs and QDax; the paper counts strictly fitter ones only, which gives
  every solution of a fitness plateau `Inf` and stops pushing them apart. The `capacity`
  solutions with the highest scores survive, a newer one before an older one on a tie
  (QDax; pyribs breaks ties arbitrarily). The selection is joint, so a result depends on
  the rest of the batch, unlike the batch rule of the cell archives: a solution of the
  batch that survives is `NewCell`, one that does not is `Rejected`, and its
  `improvement` is its score (pyribs returns it as `dns`). `add!` is a batch of one. The
  fitness type is a floating-point type, for the `Inf` score. Parents are drawn
  uniformly, as in pyribs and QDax; the paper draws them from the better half.
  `cell_index` is the slot of the nearest stored descriptor. `max_fitness` and
  `mean_fitness` are recomputed after every batch, since a remove can lower them.
  `test/reference/dns.jl` holds the populations that QDax (commit 46b8fee; DNS is not in
  QDax 0.5.0) and pyribs 0.12.0 keep after each batch of four sequences; ours match.
- **Density descent.** `DensityArchive(inner; bandwidth, buffer_capacity=10_000, rng)`
  (Lee et al., 2024, "Density Descent for Diversity Optimization"; pyribs `DensityArchive`;
  QDax has none) wraps an archive with `SVector` descriptors, as `ThresholdArchive` and
  `SpreadArchive` do: every solution goes to `inner`, which `result_archive` returns and
  whose elites emitters draw, and the descriptors go to a buffer of `buffer_capacity`, kept a
  uniform sample of all descriptors seen by reservoir sampling (Li, 1994, Algorithm L;
  pyribs's version picks the wrong elements when it replaces several in one batch). The
  generator is given at construction (the keyword `rng`), as for `CVTArchive`, since `add!` has none. The
  density at a descriptor is the Gaussian kernel density estimate of the buffer,
  `sum(φ(|y - b| / h)) / (|B| h)` with `h = bandwidth`, as in the paper and pyribs, zero
  for an empty buffer; `bandwidth` has no default (the paper finds about 5% of the width of
  descriptor space good). A result keeps the status of `inner` and has `improvement =
  -density` computed before the batch enters the buffer, so `ImprovementRanker` ranks the
  lowest density first, as pyribs's `DensityRanker`; every result of a batch is computed
  against the buffer before the batch. `DDS(prob; dims, sigma0, bandwidth, rng)` is a
  `DensityArchive` around a `GridArchive` with 15 `CMAEmitter`s of 36 solutions, ranked by
  improvement, with mu selection and restarts on convergence, as in the paper; the grid
  clamps descriptors outside its bounds by default (pyribs clips), since density descent
  leaves the known region on purpose. Only the kernel estimate is offered; the normalizing
  flow of DDS-CNF would need a neural-network package. `test/reference/density.jl` holds
  the densities that pyribs 0.12.0 reports for three sequences; ours match.
- **Stochastic evaluation.** Noisy problems receive an explicit generator,
  `evaluate(rng, prob, x)`, and opt in through `AbstractStochasticQDProblem`;
  deterministic problems keep `evaluate(prob, x)`. The loop calls
  `evaluate_batch(state.rng, prob, xs)`: for a stochastic problem it draws one `UInt64`
  seed per solution from the run's generator, one draw at a time in the order of the
  batch, before the threaded section, and evaluates solution `i` with `Xoshiro(seed_i)`.
  Results depend on the seed of the run and not on the number of threads. `Sampled`
  evaluates the wrapped problem `num_samples` times with the generator of the solution, and
  every evaluation counts against the budget.
- **Spread of noisy solutions.** MAP-Elites with low spread (QDax `MELS`) uses
  `SpreadArchive`, which wraps a `GridArchive` or `CVTArchive`, as `ThresholdArchive` wraps
  a cell archive; the other archives store no spread and their `add!` does not change. A
  `CategoricalArchive` cannot be wrapped: its descriptors have no distance. The problem is
  `SampledSpread(prob; num_samples)`, a sibling of `Sampled` whose evaluation has the mean
  fitness, every descriptor of the `num_samples` evaluations as a `DescriptorSamples`, and
  the most frequent phenotype. The archive computes the rest, since only it knows its
  cells: the spread is the mean pairwise distance between the descriptors (zero for one
  sample), the cell is the one the descriptors land in most often (the one visited first
  on a tie; QDax takes the lowest cell index), and the stored descriptor is the mean of the
  descriptors in that cell, which lies inside it (QDax stores the centroid). A solution
  replaces the elite of a cell if its fitness is strictly higher and its spread is no
  larger, as in QDax; one rejected for its spread reports zero `improvement`.
  `add_batch!` follows the batch rule of the cell archives: every result is computed
  against the archive before the batch, and the archive ends up as if the solutions had
  been added one at a time (QDax keeps an arbitrary one of several that enter a cell).
  `MELS(prob; dims)` takes a `SampledSpread` problem. `init` throws unless a
  `SpreadArchive` meets a `SampledSpread` problem and the other way round, so a
  deterministic problem never reaches it. The MAP-Elites benchmarks must not slow down.
- **Multi-objective archives.** `ParetoGridArchive` has the cells of a `GridArchive` and a
  front of at most `max_front` solutions (50 by default) in each. The fitness is a vector, all
  objectives maximized, and the archive stores it as `SVector{M,Float64}`; it needs a
  `reference_point`, below the values of interest, for hypervolume (a problem can supply it with
  `reference_point(prob)`). `add!` removes the stored solutions that the new one beats in every
  objective. Two rules are types chosen by keyword, as `outside` is. `duplicates`:
  `RejectDuplicates()` (default) rejects a solution that a stored one matches in every
  objective, so equal fitness vectors are not kept twice; `KeepDuplicates()` keeps both, as the
  MOME paper (Pierrot et al., 2022, §4: only a strictly dominating solution drops the offspring)
  and QDax do. `full_front`, for a solution that beats none of a full front:
  `RejectNewcomer()` (default) rejects it, as QDax's code does; `ReplaceCrowded()` removes the
  one of the front and the new solution with the smallest NSGA-II crowding distance (per
  objective the gap between neighbours over the range of the objective, summed; the ends of
  every objective infinite, so never removed; an objective with zero range adds nothing; on a
  tie the new solution, else the one stored first), as MOME-PGX does (Janmohamed, Pierrot and
  Cully, 2023, §4.2). The text of the MOME paper replaces a random solution instead, which
  neither QDax nor MOME-PGX does. The result is `NewCell` for an empty cell and `Improved`
  otherwise, with the hypervolume that the cell gains as its `improvement` (`AddResult{Float64}`,
  zero when rejected; negative when `ReplaceCrowded` removes a solution that covered more).
  `add_batch!` reports every result against the fronts before the batch, and then offers every
  solution in order against the front of its cell as it is by then, so the archive ends as if
  the solutions were added one at a time, as the cell archives do; a solution reported
  `Rejected` can still enter a front in which an earlier solution of the batch made room
  (QDax inserts a batch jointly). `MOME()` takes both keywords; the crowding-based parent selection and policy gradients of
  MOME-PGX are not offered. `test/reference/crowding.jl` holds the fronts that the MOME-PGX code
  (commit 58c34b7) keeps for eight insertion sequences; ours, with `KeepDuplicates` and
  `ReplaceCrowded`, match. `length` is the number of occupied cells and `num_solutions` the
  number of solutions. `sample_elites` draws an occupied cell, then a solution of its front,
  both uniformly. `qd_score` is the MOQD score, the sum of the hypervolumes of the fronts, and
  takes no offset; `max_fitness` and `mean_fitness` are not defined. The record of a run of such
  an archive has the fields of `ParetoMetricsRecord`, chosen by `metrics_record_type`. `hypervolume`
  is exact for any number of objectives.
- **CQD score.** `cqd_score(archive; targets, fitness_range, penalties=10, distance_max)`
  is the continuous QD score of Kent, Branke, Gaier and Mouret (2022), as pyribs
  `cqd_score`; QDax has none. For each target descriptor `G` and penalty weight `θ` the
  archive recommends the elite that maximizes `ω = f / (f_max - f_min) - θ d(G) /
  distance_max`, with `d` the Euclidean distance between its descriptor and `G`, and the
  score is the sum of those `ω` over targets and weights. The fitness is divided by the
  range without subtracting `f_min`, as in equation 1 of the paper and in pyribs.
  `fitness_range = (f_min, f_max)` has no default (the paper takes the problem's known
  values or the range over all compared runs); `penalties` is a number of weights spread
  evenly over `[0, 1]` (ends included, as pyribs's `linspace`) or the weights themselves;
  `distance_max` defaults to the diagonal of `descriptor_bounds(archive)`. Targets are
  given, as in pyribs; `cqd_score(rng, archive; num_targets, ...)` draws them uniformly
  within the descriptor bounds, so that a run draws new targets, as the paper does. The
  result, a `CQDScore`, also holds the mean `ω` over targets for each weight, which the
  Makie extension plots against the weight (`lines(result)`), the figure of the paper.
  The score is defined for archives with a real fitness and `SVector` descriptors; an
  empty archive throws. `test/reference/cqd.jl` holds scores of pyribs 0.12.0's
  `cqd_score` for three archives; ours match.
- **Archive ECDF.** The Makie extension adds `Makie.convert_arguments` for `ECDFPlot`, so
  `ecdfplot(archive)` draws the empirical distribution of the elite fitness, as pyribs's
  `archive_ecdf`; Makie's own attributes pick the complementary curve or counts. A
  `ThresholdArchive` shows its result archive, as for the heatmap.
- **Archive heatmaps.** The Makie extension adds `Makie.convert_arguments` methods for
  `Heatmap`, so `heatmap(archive)`, `heatmap!(ax, archive)` and `plot(archive)` work and the
  package gains no plotting function of its own. It draws archives with two descriptor
  dimensions and throws for any other number. A `GridArchive` shows one rectangle per cell and
  a `CVTArchive` its Voronoi cells sampled on a `pixels` by `pixels` grid; a
  `ParetoGridArchive` shows the hypervolume of each front, and a `ThresholdArchive` or a
  `DensityArchive` its result archive. Empty cells are `NaN`, which Makie leaves transparent. `CategoricalArchive` and
  archives with more dimensions are not drawn yet.
- **Device MAP-Elites.** The CPU is the default and nothing moves to a device unless the
  user asks: `MAPElites(prob; dims, device=CuArray)` (any array type: `ROCArray`,
  `oneArray`, or `Array` for the same kernels on the CPU) builds a `DeviceGridArchive` and
  a `DeviceVariationEmitter`, and `MAPElites(archive)` takes a `DeviceCVTArchive`; without
  them nothing changes, and the MAP-Elites benchmarks must not slow down. The kernels are
  written with KernelAbstractions.jl, a weak dependency (with Atomix.jl, which it loads)
  that CUDA.jl, AMDGPU.jl and oneAPI.jl load, in `QualityDiversityKernelAbstractionsExt`;
  the types and the host-side methods are in the package. `DeviceGridArchive` and
  `DeviceCVTArchive` are subtypes of `DeviceCellArchive`, which holds the shared storage,
  insertion, metrics and emitter, as `CellArchive` does on the host; they differ only in
  how a descriptor finds its cell. The storage is struct-of-arrays in device arrays:
  genotypes of a fixed length `dim` as a `dim × cells` matrix of a floating-point type
  `T`, fitness of a real type `F`, descriptors as an `N × cells` matrix of the element type
  of the archive's descriptor type (`Float64`, whatever `T` is; `descriptor_eltype`), and no
  phenotypes. Descriptors are never rounded to `T`, in storage or between kernels, so a
  descriptor just below a cell boundary lands in the same cell on host and device (#47). The archives meet the archive contract (descriptors `SVector{N,Float64}`,
  genotypes `Vector{T}` on the host); `GridArchive(archive)` and `CVTArchive(archive)` copy
  them to the host. A batch of candidates is a `DeviceBatch`, the columns of a device
  matrix, so the generic loop runs unchanged; evaluations come back as an
  `EvaluationBatch` and add results as an `AddResultBatch`, both device arrays. The default
  `evaluate_batch` on a device runs `evaluate(prob, x)` (for a `QDProblem`, its function)
  in a kernel, one solution per thread, so it must compile for the device: no allocation
  and no string building. A problem may instead define `evaluate_batch` for the batch and
  return an `EvaluationBatch` or a vector of `Evaluation`s. `add_batch!` follows the batch
  rule of the cell archives; the cell of each solution is chosen with atomic maximum and
  minimum operations on the fitness and the position in the batch, and a descriptor that
  `ThrowOutside()` refuses throws on the host before the archive changes. The cell of a
  descriptor is computed in `Float64`, exactly as on the host, so device and host archives
  assign the same cells, ties included; a device without `Float64`, such as an Apple GPU,
  cannot run it. `DeviceCVTArchive` takes its centroids, or the k-means of the host, as
  `CVTArchive` does; on the host it builds a regular grid of about as many buckets as cells,
  each listing the centroids that can be nearest to a point of it (a superset computed with
  the k-d tree), and a kernel compares a descriptor only with the centroids of its bucket.
  In many descriptor dimensions the buckets are coarsened until they hold at most 32
  centroids per centroid in total, down to one bucket with all of them. The initial
  population is drawn on the host with `random_solution`, copied to the device once, and
  scored there. Every `ask` draws one `UInt64` from the run's generator and the kernels
  derive their random numbers from it with a counter-based generator (Philox4x32-10) keyed
  on the child and the entry, so a run is reproducible from its seed on a given backend
  whatever the scheduling of the threads, and its random numbers differ from those of the
  CPU path. Parents are uniform over the occupied cells, listed in cell order. The device
  emitter supports `GaussianMutation` and `IsoLineVariation`; stochastic problems and the
  other archives stay on the CPU. Defaults are those of the CPU path (`Float64`); the
  documentation recommends `Float32` for consumer GPUs, which compute `Float64` slowly.
  `genotype` and `fitness` set the precision of the solutions and the stored fitness, but
  the arithmetic of the evaluation is that of the problem's code: one `Float64` literal
  makes it `Float64`, silently. The section "Float32 on a GPU" of `docs/src/guide/gpu.md`
  explains this, shows a problem written in `eltype(x)` and the CPU check
  `typeof(evaluate(prob, rand(Float32, n)).fitness)`, and warns that the built-in
  `Rastrigin` and `Sphere` compute in `Float64`.
- **Device CMA-ME.** `CMAME(prob; dims, sigma0, device=CuArray)` builds a
  `DeviceGridArchive` and one `DeviceCMAEmitter` holding `num_emitters` CMA-ES searches,
  each an ordinary `CMAEmitter`; `CMAME(archive)` does the same for any `DeviceCellArchive`.
  The searches ask together and their candidates are evaluated and added on the device as
  one batch, so within a round every search sees the archive as it was before the round,
  as in pyribs; the host `CMAME` keeps stepping its emitters one after another, as in the
  CMA-ME paper. The searches stay on the host, which reuses the code, rankers, selection
  and restart rules of `CMAEmitter` for every strategy: each round draws, in order from the
  run's generator, the elites that restarting searches start from (copied from the device
  in one gather) and one seed per search, and the searches then sample with their own
  `Xoshiro` in parallel on the host threads; the batch is copied to the device. After it is
  added, its fitness values, descriptors and results are copied back once and the
  searches update in parallel. A run is reproducible from its seed whatever the number of
  threads, and its random numbers differ from those of the host `CMAME`. Samples are drawn
  on the host and not on the device because a batch of a search has fewer candidates than
  parameters, so the candidates are a smaller transfer than a sampling transform. A device
  pays off with many searches (hundreds); with the five of the paper it is slower than the
  host.
- **Differentiable QD.** CMA-MEGA, CMA-MAEGA and OMG-MEGA (Fontaine and Nikolaidis, 2021,
  "Differentiable Quality Diversity"; CMA-MAEGA from the CMA-MAE paper) take problems whose
  fitness (a single real) and `SVector` descriptor have gradients.
  `evaluate_gradient(prob, x)` returns the evaluation and the `(N+1) × n` Jacobian, fitness
  first. The default differentiates `evaluate` with ForwardDiff, a weak dependency
  (`QualityDiversityForwardDiffExt`); without ForwardDiff it throws, naming both ways out. This
  works because problem code computes in the number type of the solution (see Julia style). A
  problem may define the method itself (closed form, or another AD package); `Sphere` and
  `Rastrigin` do, which is several times faster and needs no ForwardDiff. At a clipping limit
  their derivative is the one from inside (ForwardDiff 1.x takes the one from outside). Stochastic problems, categorical descriptors and
  device archives are refused at `init`. An evaluation with gradients counts as one evaluation,
  as in the paper and pyribs. The loop gets gradients through an optional emitter phase,
  `ask_gradients`/`tell_gradients!` (pyribs's `ask_dqd`/`tell_dqd`): before `ask`, the points an
  emitter asks for are evaluated with gradients, added to the archive and counted; the default
  `ask_gradients` returns `nothing`, which compiles away, so the other algorithms do not slow
  down. A `BanditScheduler` cannot hold such an emitter.
  `GradientArborescenceEmitter` (CMA-MEGA) follows the paper and pyribs, where QDax differs:
  `θ` moves by `AdamAscent` (default, learning rate 0.002 as in every experiment of the paper)
  or `GradientAscent` (QDax's plain step), a type passed as `optimizer`; the fitness coefficient
  may take either sign (QDax forces it non-negative); `θ` starts at `x0` or a random elite
  (QDax starts at zero); the gradients are normalized by default, with `1e-8` added to the
  length, as pyribs. The coefficients come from any `AbstractEvolutionStrategy` over `N+1`
  dimensions starting at zero, ranked, selected and restarted by the rules of `CMAEmitter`.
  The step of `θ` is `J' Σᵢ wᵢ cᵢ` over the selected candidates with the weights of CMA-ES,
  computed before the strategy updates; a round without parents does not move `θ` (pyribs then
  steps toward zero). The point `θ` is offered to the archive but not ranked, as in pyribs.
  `test/reference/mega.jl` holds the branches, rankings, parent counts and points of pyribs
  0.12.0 over five scripted rounds, for Adam and for gradient ascent; ours match.
  OMG-MEGA keeps the gradients of every elite, as QDax and the paper do (pyribs instead
  evaluates the gradients of each drawn parent again, two evaluations per child):
  `GradientArchive` wraps a cell archive or a `ThresholdArchive` around one (not a
  `SlidingBoundariesArchive`, whose elites move) and keeps the Jacobian of the solution that the
  wrapped archive keeps; its `add!` and `add_batch!` take the Jacobians as a fourth argument and
  throw without them, and a run with it evaluates every solution with gradients.
  `GradientOperatorEmitter` draws an elite uniformly and coefficients with standard deviation
  `sigma0` (pyribs `sigma_g`; QDax takes the variance), the first in absolute value as in the paper, and
  steps along the normalized gradients. `CMAMEGA` and `CMAMAEGA` run one emitter by default, as
  Algorithm 1 of the paper; `sigma0` has no default.
- **Sliding boundaries.** `SlidingBoundariesArchive` (pyribs; Fontaine et al., 2019) is a
  grid whose boundaries along each dimension move to the quantiles of the descriptors of
  the last `buffer_capacity` solutions offered to it (1000 by default), every
  `remap_frequency` solutions (100 by default). It starts as a regular grid over `bounds`.
  A remap empties the cells and adds back, one at a time, the old elites and then the
  buffered solutions from oldest to newest, as pyribs does, so the best solution is
  never lost. `add_batch!` follows the batch rule of the cell archives; a remap happens
  after the batch in which the count of offered solutions passes a multiple of
  `remap_frequency`, never in the middle of a batch (pyribs remaps mid-batch). The
  solution that `add!` offers when a remap is due is not placed in the old cells, where it
  could evict an elite that is no longer in the buffer; the remap adds it, as pyribs does.
  Its result is computed against the archive before the remap, where pyribs reports the
  one after. One solution at a time, the archive matches pyribs (`test/reference/sliding.jl`)
  up to ties in fitness and pyribs's tolerance of `1e-6` below a boundary, with one pyribs
  bug corrected in the reference: pyribs 0.12.0 clips descriptors to the old bounds while
  it adds the solutions back during a remap, so when the boundaries widen, solutions land
  in the wrong cells. A descriptor outside the boundaries is handled by `outside`, as for a
  `GridArchive`, but the default is `ClampOutside()`, as pyribs clips: after a remap the
  outer boundaries are the extremes of the buffer, so new solutions often land outside
  them. Clamping only chooses the cell; the stored and buffered descriptor is the one the
  problem returned. Every solution that has a cell enters the buffer, also one that is
  rejected, so with `ClampOutside()` the next remap widens the boundaries to it; with
  `RejectOutside()` a solution outside the boundaries is discarded for good and never enters
  the buffer (#40). Old elites are always clamped during a remap, so an elite whose
  descriptor has left the buffer and lies beyond the new boundaries sits in an edge cell
  with a descriptor outside the reported bounds. A descriptor with a non-finite component
  throws whatever the policy, before it can reach the buffer, where a remap would sort it.
  `ThresholdArchive` and `SpreadArchive` cannot wrap it: their thresholds and spreads
  belong to cells that move.
- **Fitness values.** Every archive throws an `ArgumentError` for a fitness that is `NaN`
  or infinite, also after conversion to its fitness type (`Float32` overflow), before it
  changes, also inside a batch, as pyribs does (it rejects non-finite objectives and
  measures); a problem marks an infeasible solution with a low finite fitness. One check,
  `stored_fitness`, serves the cell, threshold, sliding, spread, unstructured and DNS
  archives and the default `add_batch!`; `ParetoGridArchive` checks every objective, and the
  device archives check on the host before they change. `Testing.test_archive` and
  `test_front_archive` require it of every archive. Any `Real` fitness type works,
  unsigned ones included: the status compares before it subtracts, and `fitness_gain` makes
  the `improvement` of a worse unsigned fitness zero instead of wrapping around. The device
  archives order `-0.0` and `0.0` as equal, as the host does, so a tie keeps the first.
- **Bounds of an evolution strategy.** A sample outside `lower`/`upper` is drawn again; after
  `RESAMPLE_LIMIT` (1000) draws the last one is clipped into the box (pyribs resamples
  without limit and can hang: with the mean on the boundary a draw lands inside with
  probability `2^-dim`). CMA-ES, sep-CMA-ES and LM-MA-ES then compute the stored step `y` of
  the clipped solution from where it lies, so the mean moves to it; LM-MA-ES's `z` and
  OpenAI-ES's noise stay as drawn. A `CMAEmitter` refuses an `x0` outside its bounds.
- **LM-MA-ES batch.** `LMMAES` requires `2 batch_size ≤ dim`, so that `c_σ = 2 batch_size / dim`
  is at most 1 (#45); above 1 the step-size path changes sign every generation. With the
  default batch this means `dim ≥ 26`. The error says why and names `CMAES`/`SepCMAES`;
  pyribs refuses only `batch_size > dim`.
- **Tables.** With Tables.jl loaded (`QualityDiversityTablesExt`) every archive is a table of
  the elites of its `result_archive`, as the plots show: columns `genotype`, `fitness`,
  `descriptor_1`... and `phenotype` when stored. Genotypes and phenotypes are `deepcopy`'d,
  so changing a row never changes the archive.
- **Outside-bounds descriptors.** The raw archives (`GridArchive`, `CVTArchive`, ...)
  throw by default. The algorithm constructors (`MAPElites`, `CMAME`, `CMAMAE`, the MEGA
  algorithms, `MOME`, `MELS`, `DDS`) clamp by default, as pyribs and QDax do, so a random
  initial population never stops a run; for a problem with `descriptor_categories`,
  `MAPElites` throws by default, since a category has no nearest cell. The `outside`
  keyword takes `ThrowOutside()`, `ClampOutside()` or `RejectOutside()`, chosen by
  dispatch.
- **Grid limits.** Every `(prob; ...)` constructor takes `bounds=nothing`, which
  `problem_bounds` resolves to `descriptor_bounds(prob)`; a given `bounds` overrides it.
- **Algorithm and problem.** `MAPElites(prob; dims)` takes the problem for its
  descriptor bounds; for a problem with `descriptor_categories`,
  `MAPElites(prob)` builds a `CategoricalArchive` and `dims` is an error. `init` checks that archive and problem agree on the
  number of descriptor dimensions.
- **Discrete descriptors.** `CategoricalArchive(categories)` takes one
  collection of allowed values per dimension, numbers or labels such as
  `Symbol`s. Its descriptor type is the tuple of their element types, e.g.
  `Tuple{Int,Symbol}`, so labels need no encoding. A problem declares
  `descriptor_categories` (or `QDProblem(...; categories)`) instead of
  `descriptor_bounds`. Archives that need geometry keep `SVector` descriptors.
- **Benchmark clipping.** `Rastrigin` clips the solution to `[-5.12, 5.12]`
  inside `evaluate`. Solution-space bounds for mutations are not part of the
  problem contract.
- **Phenotype storage.** The mutated object (the genotype, e.g. a reaction
  sequence) can differ from the described object (the phenotype, e.g. the
  product). The mapping between them belongs to the problem; archives store the
  phenotype next to the genotype. `Evaluation{F,D,P}` has an optional
  `phenotype::P`, and the two-argument `Evaluation(fitness, descriptor)` gives
  `P = Nothing`. A `QDProblem` function may return `(fitness, descriptor)` or
  `(fitness, descriptor, phenotype)`. Archives carry `P` as a fourth type
  parameter, `AbstractArchive{G,F,D,P}`, store `phenotypes::Vector{P}`, and
  return the phenotype through `elites(archive)`; variation sees genotypes only.
  Archive and algorithm constructors take `phenotype=Nothing`, the type to store.
  An evaluation whose phenotype does not fit the archive throws before the
  archive changes, also inside a batch. `Sampled` keeps the phenotype that occurs
  most often among its evaluations (`isequal`, the first on a tie); its
  `expectation` applies to the fitness and the descriptor only.
  The phenotype is opt-in and free when unused: a problem that returns no
  phenotype needs no change, `nothing` occupies no memory, and the code for
  `P = Nothing` must compile to what it would be without the field. The
  allocation tests cover `P = Nothing`, and the MAP-Elites benchmarks must not
  slow down.

## Open decisions

Ask the author before resolving any of these.

None at present.

## Current work (as of 2026-10-23)

Not settled design; the state of the work and the plan agreed with the author.

- **Order to 0.1.0.** PR #69 (LM-MA-ES, fixes #45) is merged → #47 (PR #70) is merged →
  benchmarks in `Float32` on a device (PR #71) is merged → docstring of the clamped elites
  of `SlidingBoundariesArchive` (PR #72) is merged → release PR (0.1.0,
  `Pkg.add("QualityDiversity")` in README and docs, a `[compat]` entry for Tables) → figures.
- **`fix-sliding-outside` is dropped for now.** Commit f5bb1fc ("Widen sliding boundaries
  to outside descriptors"), kept only on that local branch, removes the `outside` keyword
  of `SlidingBoundariesArchive` and widens the outer boundaries at once to every offered
  descriptor and, at a remap, to the old elites, so no stored descriptor lies outside the
  reported bounds. #40 itself is fixed on main (#61). Rebasing it would conflict with the
  keyword, deduplication and export changes in `sliding.jl`.
- **#47: PR #70, merged.** Device archives stored
  descriptors in the genotype float type `T`, so with `Float32` genotypes host and device
  could bin a descriptor near a boundary differently. The branch stores them in
  `descriptor_eltype` of the archive (`Float64`; the author chose this over a literal
  `Float64` and over a new type parameter, which would need the host archives to take it
  too), adds the test "Float64 descriptors with Float32 genotypes" in
  `test/test_device.jl` (grid and CVT on `JLArray`, 10 of 19 checks fail without the fix),
  and the docs section "Float32 on a GPU". Full suite passes on release and LTS. The GPU
  comparison (`benchmark/compare/gpu`) is not rerun for it: the author defers that until
  the package is more stable.
- **Benchmarks in `Float32` on a device: PR #71, merged.** The
  scoring of `Rastrigin` and `Sphere` is in `rastrigin(prob, x, T)` and `sphere(prob, x,
  T)`, with every constant converted to `T`; `evaluate` passes `promote_type(Float64,
  eltype(x))` and `device_evaluate`, which only the kernel calls, `float(eltype(x))`. The
  CPU results are bit-identical to before (checked for `Float64`, `Float32` and `BigFloat`
  inputs); on a device `Float32` solutions score in `Float32`, which can differ from the CPU
  fitness in the last digits. The docstrings and "Float32 on a GPU" say so, and
  `test/test_device.jl` checks it. `benchmark/compare/gpu` defines its own Rastrigin in
  `eltype(x)`, so the published GPU figures are not affected.
- **Speed figures.** A few plots, drawn by a script from CSV files that the comparison
  scripts write, images in `docs/src/assets`:
  - CPU MAP-Elites, time against batch size: QualityDiversity.jl, pyribs, QDax (grid index);
    CPU bars for CMA-ME, CMA-MAE and MOME.
  - GPU against GPU only. Never put QDax on a GPU next to QualityDiversity.jl on a CPU; the
    table in `benchmark/compare/README.md` that mixes them is to be split.
  - Same precision side by side (`float32` with `float32`), the same hardware and core
    pinning as the tables there, and the versions recorded.
  - A capability table (✓/~/✗) checked against the pyribs 0.12 and QDax source. pyribs can
    store object solutions (`solution_dim=()`, `solution_dtype=object`) but its emitters are
    numeric only; QDax cannot. The README sentence "slow or impossible" overstates it for
    pyribs.
- **City-network example** (new and separate; `examples/graphs.jl` stays): connect 20 fixed
  cities; fitness is robustness to the failure of one city, descriptors are total link
  length (cost) and mean hops (latency). Show the heatmap and three networks drawn on the
  map. Compare with pyribs + networkx (total time and time without evaluation); no QDax bar,
  since QDax cannot run it. Adding networkx to `benchmark/compare/requirements.txt` is not
  yet agreed.
- **Small:** the CI.yml comment "Codecov accepts tokenless uploads only from forks" is
  wrong; uploads need the `CODECOV_TOKEN` secret.

### How the author wants to work

- Before a design question on a component, explain in plain language what it is and how it
  behaves; then ask, with options phrased by their effect, not by internal names.
- For every component, read pyribs **and** QDax (and the paper), and state the differences
  in the explanation and in Decisions; if one library lacks it, say so.
- Run `git fetch` and look at `origin/main` before reporting the state of the repository or
  branching.
- Run test suites in a separate `git worktree` (or a copy without `Manifest.toml` for the
  LTS), never in the main working tree.
- One issue, one branch, one PR; full suite on release and LTS, PR text approved by the
  author; then wait before starting the next.

## Minor things still to do

Small follow-ups that do not block anything. Take them up when convenient.

- To think about: a number type `T` as a type parameter of the optimizer states, instead
  of `Float64`, for CMA-ES, the other evolution strategies and the MEGA emitters (the
  point, Adam's moments, the kept Jacobians, and the Jacobians a `GradientArchive`
  stores). `Float64` fields are concrete, so this is about genericity, not speed: it
  would allow `Float32` searches, for example to keep everything on a device, and keep a
  `BigFloat` Jacobian unrounded. Do it across all optimizer states at once or not at all,
  so the package does not mix `T` and `Float64`; ask the author before starting.

---

## Conventions

### Julia style

- Julia compat lower bound: **1.10 (LTS)**. Every dependency gets a
  `[compat]` entry whose lower bound works on 1.10. Run `Pkg.resolve()` after
  editing `Project.toml`.
- Annotate method arguments only as tightly as the implementation needs.
  `AbstractVector{<:Real}`, not `Vector{Float64}`. Annotate to control
  dispatch, not to document intent. An `AbstractArray` annotation promises
  that the code works for offset axes, views and GPU arrays; see
  `.claude/rules/julia-generic-indexing.md`.
- Problem code computes in the number type of the solution, never converting entries to
  `Float64`: the benchmarks use `promote_type(Float64, eltype(x))`, and `QDProblem` gives a
  real descriptor that type too, so ForwardDiff's dual numbers and `BigFloat` pass through
  `evaluate` while ordinary input keeps `Float64`. Archives and optimizer states keep their
  concrete storage types (`SVector{N,Float64}` descriptors, `Float64` CMA-ES state), which is
  what keeps them fast; nothing is differentiated through them.
- Use American spelling. Avoid jargon that experts in the field do not widely
  accept, and do not let technical prose read like a pitch deck.
- Parametric constructors: inner constructor with unconstrained value
  arguments; outer constructors compute type parameters and delegate inward,
  so `T(args...)`, `T{A}(args...)` and `T{A,B}(args...)` coerce identically.
- Use the short keyword form `f(; batch_size)` when the variable name matches.
- Prefer immutable `struct`s. A `mutable struct` needs a reason (for example
  `QDState`); mark fields that never change as `const`.
- Test error messages, not just exception types:
  `@test_throws "descriptor has 3 dimensions but the archive has 2" add!(...)`.
- Write docstrings for every exported or `public` name, with a runnable
  example where it helps. Test examples before committing them.

### Comments and commit messages

Comments, docstrings and commit messages describe what is true of the code
now, for a reader who has only the repository. No references to plans,
phases, roadmap items, chat sessions, or history ("previously", "for now").
Full rule: `.claude/rules/code-comments.md`. A hook (`.claude/hooks/`) flags
violations after each edit.

### Working with Julia

- Use Revise in a persistent session to avoid paying compilation on every
  run. If a Julia MCP server is configured, use it; otherwise use
  `julia --project` from the shell and keep runs few and purposeful.
- Activate the test environment with `TestEnv.activate()` (TestEnv, Revise,
  BenchmarkTools, JET live in the global environment, not in this project).
- `julia +lts` runs the LTS, `julia +release` the current release (juliaup).
  Run the full suite on both before declaring a phase done.
- When adding a dependency, choose it on its merits, not because it happens
  to be installed.
- Run `Pkg.test()` as the final check before a pull request; iterate with
  Revise and targeted includes. A change to a docstring, an example or `docs/`
  also needs a clean `julia --project=docs docs/make.jl`.
- Do not add developer tools (BenchmarkTools, JET, Cthulhu, Revise) to the
  package's `[deps]`. Test-only dependencies go in `test/Project.toml`.

### Git and GitHub

- Subject lines ≤ 50 characters where possible, ≤ 72 at most.
- Do not post comments, issues or PRs on GitHub without the author's approval
  of the exact text, and do not represent the author on GitHub without
  consent.
- Reference an issue or PR that motivated a change in a code comment, not in
  the subject line. Close issues with `Fixes #xyz` in the body; for several,
  write `Fixes #abc; fixes #def` (ranges and comma lists do not work).
- Attribute agent assistance with an `Assisted-by:` trailer naming the model,
  not `Co-authored-by`.
- Do not commit planning documents, agent conversations or benchmark
  narratives. Add them to `.git/info/exclude`.
- This file and `.claude/` are kept out of the repository: they are listed in
  `.git/info/exclude` and never committed. Git treats excluded files as disposable, so
  checking out a commit from before they left the repository (before `e39899b`)
  overwrites them, and moving back deletes them; keep a copy before doing so.
- A copy of this file and `.claude/` is pushed to the orphan branch `agent-notes`, which
  holds only them, so that a session on another machine can fetch them
  (`git checkout origin/agent-notes -- CLAUDE.md .claude`, then unstage with
  `git reset -q CLAUDE.md .claude`). It is never merged into `main`. After changing this
  file, push the new copy there too.

### Skills available in this repository

The skills live in the user-level `~/.claude/skills` (see
[timholy/claude_config](https://github.com/timholy/claude_config)); none is
stored in this repository.

| Skill | Use |
|---|---|
| `/qd-new-component` | add a problem, archive, variation, emitter or algorithm that meets its contract, with tests |
| `/profile-performance` | find hot spots, runtime dispatch and allocations; propose fixes |
| `/qd-benchmark` | maintain the BenchmarkTools suite and the pyribs comparison |
| `/review-design`, `/review-api`, `/review-integration`, `/review-implement` | review the type hierarchy, public API and integration against this file, then implement the plan |
| `/freshen-aqua`, `/freshen-explicit-imports`, `/freshen-runic`, `/freshen-coverage`, `/freshen-docstrings`, `/freshen-docs`, `/freshen-generic-axes` | bring one quality check up to date |
| `/freshen-package` | run the `/freshen-*` skills in a sensible order |
| `/limit-struct-mutability` | make `mutable struct`s immutable or mark fields `const` |
