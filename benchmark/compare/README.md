# QualityDiversity.jl, pyribs and QDax on Rastrigin

MAP-Elites on the 20-dimensional Rastrigin function, run with the same
settings in three libraries. Later sections do the same for CMA-ME and CMA-MAE,
for MOME against QDax, and for MAP-Elites on a GPU.

| setting | value |
|---|---|
| problem | Rastrigin, 20 parameters, clipped to [-5.12, 5.12]; descriptor = mean of the first and of the last 10 clipped parameters |
| archive | 100 × 100 grid over [-5.12, 5.12]², plain MAP-Elites (no thresholds) |
| mutation | Gaussian, σ = 0.5, no clipping |
| initial population | one batch of uniform random solutions |
| budget | 360 000 evaluations, a multiple of every batch size |
| batch sizes | 36, 100, 1000, 10 000 |
| repeats | 21 timed runs per batch size, one seed each, after a warm-up run (3 for QDax with its own cell lookup, see below) |
| hardware | every library pinned to one core of a 4-vCPU cloud container (Intel Xeon @ 2.1 GHz) with `taskset -c 1` |
| software | Julia 1.13.1 (1 thread), Python 3.13, pyribs 0.12.0, numpy 2.5.4, QDax 0.5.0, jax 0.11.2 (CPU) |

`compare.jl`, `compare.py` and `compare_qdax.py` define the problem
separately. Each prints the same columns. `compare.jl` and `compare.py` print
the fitness and descriptor of one fixed solution first, so the two can be
checked against each other:

```
Julia   reference fitness -407.77649216868974 descriptor (-3.0450526315789466, 3.0450526315789475)
pyribs  reference fitness -407.7764921686897  descriptor (-3.0450526315789475, 3.045052631578947)
```

Time covers building the archive, the initial batch and the rest of the
budget; compilation is excluded. Julia is timed with `@elapsed` and Python
with `time.perf_counter`, after one untimed run that compiles. "Middle half" is
the range from the first to the third quartile of the run times.

Every script prints its table and writes one row per timed run to a CSV file in
`results/` (`results.jl` and `results.py` write the same columns, with the
software versions, the commit and the hardware). `plot.jl` draws the figures of
the documentation from these files:

```
julia --project=docs benchmark/compare/plot.jl
```

`run_all.sh` runs every script on one machine with the core pinning described
below; the CSV files in `results/` come from one such run.

## How each library is used

- **QualityDiversity.jl:** `MAPElites` with a `GaussianMutation` and the
  default threaded `evaluate_batch` (one thread here). The archive records
  its metrics after every step. It runs in `Float64` with the package's
  `Rastrigin`, and in `Float32` with the same function written as a
  `QDProblem` with `Float32` arithmetic, `genotype=Vector{Float32}`,
  `fitness=Float32` and `GaussianMutation(0.5f0)`. The archive then holds
  `Float32` genotypes and fitness values and the mutation produces `Float32`
  children, with no change to the package.
- **pyribs:** `GridArchive`, `GaussianEmitter` and `Scheduler`. The objective
  is vectorized numpy, the fastest way to write it in Python.
- **QDax:** the whole run, including the initial batch, is one `jax.jit`
  function that scans `MAPElites.scan_update`, as the QDax documentation
  recommends. The objective and the mutation are jitted array code, and QDax's
  own QD score, maximum fitness and coverage are computed after every step.
  QDax is written for accelerators and these runs use the CPU. It works in
  `float32` by default; `JAX_ENABLE_X64=1` gives `float64`, which is what the
  other two libraries use.

QDax finds the cell of a descriptor by comparing it with every centroid
(`get_cells_indices`). That works for any tessellation and is cheap on a GPU,
but on a CPU it costs batch size × number of cells and takes almost all of the
run time: at batch size 1000 it is 148 ms per step, against under 1 ms for
the objective, the mutation, the selection and the metrics together. So QDax
is reported twice:

- **QDax, centroid search** is QDax exactly as documented. It takes about 45 s
  whatever the batch size, so each batch size has only 3 timed runs; the run
  times differ by under 5%.
- **QDax, grid index** replaces that one function with the arithmetic index of a
  regular grid (`QDAX_GRID_LOOKUP=1`). A check at start-up confirms it assigns
  the same cell as QDax's own lookup to 20 000 random descriptors (0 differ).
  Nothing else changes. This is the fair CPU comparison: it is what someone
  who needs QDax to be fast on a CPU would write.

## Results

Median time in seconds for the whole budget, one core:

| batch size | QualityDiversity.jl, float64 | QualityDiversity.jl, float32 | pyribs | QDax, grid index, float32 | QDax, grid index, float64 | QDax, centroid search, float32 |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.348 | 0.335 | 3.659 | 0.853 | 1.371 | 44.7 |
| 100 | 0.317 | 0.280 | 1.658 | 0.451 | 0.805 | 43.3 |
| 1000 | 0.313 | 0.260 | 0.590 | 0.168 | 0.332 | 42.8 |
| 10 000 | 0.317 | 0.278 | 0.568 | 0.160 | 0.321 | 46.8 |

Time of QDax divided by time of QualityDiversity.jl at the same precision, so
values above 1 mean QualityDiversity.jl is faster:

| batch size | float32 (QDax, grid index) | float64 (QDax, grid index) | QDax, centroid search, against float32 |
|---:|---:|---:|---:|
| 36 | 2.5 | 3.9 | 133 |
| 100 | 1.6 | 2.5 | 155 |
| 1000 | 0.65 | 1.06 | 164 |
| 10 000 | 0.58 | 1.01 | 168 |

Time of pyribs divided by time of QualityDiversity.jl in `float64`:
10.5, 5.2, 1.9 and 1.8 for batch sizes 36, 100, 1000 and 10 000.

In short: QualityDiversity.jl is faster than QDax on a CPU up to a batch size
of about 100, and QDax with the grid index is faster from 1000 up. In
`float32`, the precision QDax uses by default, QDax is about 1.5 to 1.7 times
faster at batch sizes 1000 and 10 000. In `float64` the two are level from
batch size 1000. `Float32` makes QualityDiversity.jl 4% to 17% faster, less
than the factor of two QDax gains, because Julia's scalar `cos` is not much
faster in `Float32` while XLA vectorizes it. pyribs is slower than both at
every batch size.

QualityDiversity.jl takes about the same time at every batch size (0.26 to
0.35 s, over a million evaluations per second), because each solution costs
the same work: one mutation, one evaluation and one insertion. The libraries
that vectorize a whole batch gain from larger batches.

### Full tables

**QualityDiversity.jl, float64, 1 thread**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.314 | 0.348 | 0.333–0.375 | 1 034 400 | 0.986 | -130.5 | 4 779 398 |
| 100 | 0.268 | 0.317 | 0.305–0.338 | 1 134 354 | 0.986 | -133.7 | 4 786 431 |
| 1000 | 0.282 | 0.313 | 0.291–0.348 | 1 149 679 | 0.986 | -133.2 | 4 796 417 |
| 10 000 | 0.262 | 0.317 | 0.288–0.345 | 1 134 320 | 0.942 | -130.6 | 4 614 685 |

**QualityDiversity.jl, float32, 1 thread**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.287 | 0.335 | 0.311–0.353 | 1 073 332 | 0.989 | -132.2 | 4 777 084 |
| 100 | 0.252 | 0.280 | 0.270–0.288 | 1 287 608 | 0.986 | -133.5 | 4 779 398 |
| 1000 | 0.238 | 0.260 | 0.251–0.273 | 1 382 371 | 0.986 | -133.8 | 4 793 316 |
| 10 000 | 0.242 | 0.278 | 0.265–0.292 | 1 293 366 | 0.945 | -131.1 | 4 617 796 |

**pyribs** (`OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`)

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 3.370 | 3.659 | 3.565–3.828 | 98 386 | 0.988 | -136.5 | 4 790 761 |
| 100 | 1.443 | 1.658 | 1.607–1.743 | 217 188 | 0.988 | -134.0 | 4 796 704 |
| 1000 | 0.544 | 0.590 | 0.574–0.599 | 609 720 | 0.986 | -132.5 | 4 803 731 |
| 10 000 | 0.522 | 0.568 | 0.550–0.584 | 634 046 | 0.945 | -135.2 | 4 630 502 |

**QDax, grid index, float32**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.732 | 0.853 | 0.815–0.881 | 422 009 | 0.987 | -131.1 | 4 780 165 |
| 100 | 0.404 | 0.451 | 0.425–0.474 | 798 945 | 0.988 | -136.9 | 4 789 608 |
| 1000 | 0.146 | 0.168 | 0.161–0.184 | 2 140 452 | 0.985 | -135.3 | 4 788 907 |
| 10 000 | 0.143 | 0.160 | 0.148–0.166 | 2 245 315 | 0.944 | -127.8 | 4 623 051 |

**QDax, grid index, float64**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.272 | 1.371 | 1.343–1.414 | 262 609 | 0.986 | -133.5 | 4 780 745 |
| 100 | 0.723 | 0.805 | 0.769–0.818 | 447 417 | 0.985 | -133.3 | 4 771 358 |
| 1000 | 0.300 | 0.332 | 0.318–0.343 | 1 083 718 | 0.986 | -131.9 | 4 795 016 |
| 10 000 | 0.287 | 0.321 | 0.310–0.352 | 1 119 774 | 0.942 | -131.1 | 4 613 920 |

**QDax, centroid search, float32** (3 runs per row)

| batch size | min s | median s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 44.259 | 44.722 | 8 050 | 0.985 | -139.7 | 4 748 242 |
| 100 | 43.191 | 43.289 | 8 316 | 0.986 | -141.1 | 4 789 824 |
| 1000 | 41.725 | 42.759 | 8 419 | 0.987 | -134.7 | 4 804 787 |
| 10 000 | 45.567 | 46.773 | 7 697 | 0.948 | -124.1 | 4 629 452 |

Coverage and QD score agree across the libraries at each batch size to within
the spread between seeds. At batch size 10 000 every library fills fewer cells
(about 0.94), because the budget allows only 36 rounds.

## What the numbers do not say

- **One CPU core.** QDax is built for accelerators, where the work per step is
  parallel across thousands of solutions. These runs use one CPU core; the
  [GPU section](#map-elites-on-a-gpu) runs QDax and QualityDiversity.jl on a GPU.
- **Cheap objective.** Rastrigin takes about 200 ns per solution. A slower
  objective shrinks the ratios for every library, and an objective that cannot
  be written as jittable array code cannot be used in QDax at all.
- **The QDax grid index is my change.** It is a small, checked replacement of
  one function and not part of QDax. Both QDax variants are reported so the
  difference is visible.
- **`float32` against `float64`.** Compare like with like: QualityDiversity.jl
  in `float32` with QDax in `float32`, and `float64` with `float64`. pyribs
  runs in `float64`.
- **Spread.** The run-to-run spread on this shared machine is about ±10%
  within a batch size, so close values, such as QualityDiversity.jl against
  QDax in `float64` at 1000 and 10 000, are level.
- **Threads.** Everything is pinned to one core. Neither the threaded
  `evaluate_batch` of QualityDiversity.jl nor multi-core JAX is measured.

## Reproducing

```
# Julia
taskset -c 1 julia --threads=1 benchmark/compare/compare.jl

# pyribs and QDax
python -m venv venv && venv/bin/pip install -r benchmark/compare/requirements.txt
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
taskset -c 1 venv/bin/python benchmark/compare/compare.py
QDAX_GRID_LOOKUP=1 taskset -c 1 venv/bin/python benchmark/compare/compare_qdax.py
JAX_ENABLE_X64=1 QDAX_GRID_LOOKUP=1 taskset -c 1 venv/bin/python benchmark/compare/compare_qdax.py
REPEATS=3 taskset -c 1 venv/bin/python benchmark/compare/compare_qdax.py   # about 15 minutes
```

## CMA-ME and CMA-MAE

`compare_cma.jl`, `compare_cma.py` and `compare_cma_qdax.py` run the emitters of
CMA-ME (Fontaine et al., 2020) and CMA-MAE (Fontaine and Nikolaidis, 2023) on the
same Rastrigin problem and the same 100 × 100 grid.

| setting | value |
|---|---|
| emitters | 5 independent CMA-ES searches, each with batch size 12 (`4 + floor(3 log 20)`), initial step size 1, first search from the origin, no bounds on the solutions |
| methods | CMA-ME with improvement emitters, with optimizing emitters, and with random direction emitters; CMA-MAE (learning rate 0.01, minimum threshold -820) |
| budget | 100 020 evaluations |
| repeats | 11 runs per method, one seed each, after a warm-up run; QDax compiles once per method |
| hardware, software | as above; QDax at `float32` and `float64`, the others in `float64` |

Each library's own defaults are used for everything not listed. QualityDiversity.jl
steps the emitters one after the other, as the CMA-ME paper does, pyribs adds the
solutions of all emitters at once, and QDax runs one emitter of its pool per step.

### Results

Median over 11 runs. Time is for the whole budget on one core. Coverage and QD score
are those of the archive of the best solutions (QD-score offset 820).

| method | library | median s | coverage | max fitness | QD score |
|---|---|---:|---:|---:|---:|
| CMA-ME, improvement | QualityDiversity.jl | 0.591 | 0.999 | -126.7 | 4 654 785 |
| | pyribs | 3.285 | 0.999 | -119.9 | 4 638 187 |
| | QDax, float32 | 12.82 | 0.998 | -129.0 | 4 704 976 |
| | QDax, float64 | 13.21 | 0.998 | -127.6 | 4 702 235 |
| CMA-ME, optimizing | QualityDiversity.jl | 0.732 | 0.079 | -15.4 | 490 787 |
| | pyribs | 2.962 | 0.081 | -17.9 | 504 191 |
| | QDax, float32 | 13.13 | 0.621 | -114.2 | 3 248 358 |
| | QDax, float64 | 13.34 | 0.625 | -116.8 | 3 263 107 |
| CMA-ME, random direction | QualityDiversity.jl | 0.626 | 0.999 | -131.5 | 4 291 023 |
| | pyribs | 3.396 | 0.999 | -132.1 | 4 301 840 |
| | QDax, float32 | 13.05 | 0.815 | -129.5 | 3 648 340 |
| | QDax, float64 | 13.41 | 0.817 | -127.0 | 3 634 406 |
| CMA-MAE | QualityDiversity.jl | 0.687 | 0.444 | -90.1 | 2 543 936 |
| | pyribs | 3.700 | 0.582 | -111.1 | 3 060 205 |
| CMA-MAE, covariance update of pyribs | QualityDiversity.jl | 0.753 | 0.570 | -118.8 | 3 037 135 |

QDax has no CMA-MAE.

QualityDiversity.jl is four to six times faster than pyribs and about 20 times
faster than QDax on this CPU run. The QDax time says little about QDax on an accelerator, which
this does not measure: each QDax step here is a batch of 12, far too small to use one.

### Where the libraries agree and differ

- **CMA-ME with improvement, optimizing and random direction emitters:**
  QualityDiversity.jl and pyribs agree within the spread between seeds. The
  optimizing emitter fills few cells in both, as in the CMA-ME paper.
- **QDax's emitters behave differently.** Its optimizing emitters fill far more cells
  (coverage 0.62 against 0.08), and its random direction emitters fill fewer (0.82
  against 0.999). From reading the QDax code: it recombines all
  candidates of a batch with log-decreasing weights, not the best half or the
  accepted ones (a comment in the code says this departs from the paper), it
  restarts on its own rules, and an emitter measures improvement against the archive as
  it stood after that emitter's previous batch, not before the current one. It is a
  looser variant of CMA-ME, so its columns are not a check on the other two.
- **CMA-MAE: pyribs reaches a higher coverage than QualityDiversity.jl.** The cause
  is in pyribs's CMA-ES. Its rank-one covariance term is `c1² pc pcᵀ`, where
  Hansen's tutorial and the LM-MA-ES paper (Algorithm 1) have `c1 pc pcᵀ`, so the
  covariance matrix adapts almost only through the rank-mu term and the search stays
  rounder for longer, which helps CMA-MAE fill the grid. QualityDiversity.jl follows
  the textbook update. The CMA-MAE paper ran all its experiments with pyribs. With
  `strategy=QualityDiversity.PyribsCMAES`, which only changes that coefficient, QualityDiversity.jl
  reaches 0.570 coverage and a QD score of 3.04 million against 0.582 and 3.06
  million for pyribs. Swapping the textbook term into pyribs lowers its CMA-MAE
  coverage from 0.58 to 0.46, and the step size rule and the scaling of the path make
  no difference. CMA-ME with improvement emitters is the same with either update
  (QD scores 4.65 million with the textbook term and 4.64 million with the pyribs term).
- **Random direction ranking.** The CMA-ME paper ranks the accepted solutions by
  their projection on the direction, without putting new cells first. pyribs and
  `TwoStageRandomDirectionRanker` rank new cells first; `AcceptedRandomDirectionRanker`
  follows the paper. Here both give the same QD score (4.29 million).

### Not measured

The same caveats as for MAP-Elites hold: no GPU, one core, a cheap objective, and a
spread of about ±10% in time between runs. In addition, 5 emitters of batch 12 and
100 000 evaluations are small next to the CMA-MAE paper (15 emitters of batch 36 and
10 000 iterations), so the QD scores here do not compare with its tables.

### Reproducing

```
# Julia
taskset -c 1 julia --threads=1 benchmark/compare/compare_cma.jl

# pyribs and QDax
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
taskset -c 1 venv/bin/python benchmark/compare/compare_cma.py
taskset -c 1 venv/bin/python benchmark/compare/compare_cma_qdax.py
JAX_ENABLE_X64=1 taskset -c 1 venv/bin/python benchmark/compare/compare_cma_qdax.py   # about 8 minutes each
```

## MOME

`compare_mome.jl` and `compare_mome_qdax.py` run multi-objective MAP-Elites (Pierrot et al.,
2022) on the two-objective BiSphere benchmark of this package. pyribs has no multi-objective
archive, so only QualityDiversity.jl and QDax are compared.

| setting | value |
|---|---|
| problem | BiSphere, 10 parameters, clipped to [-5.12, 5.12]; objectives `-‖x - 1‖²` and `-‖x + 1‖²`, both maximized; descriptor = mean of the first and of the last 5 clipped parameters |
| archive | 20 × 20 grid over [-5.12, 5.12]², a Pareto front of at most 50 solutions in each cell |
| hypervolume | reference point `(-374.5, -374.5)`, the lowest value of both objectives |
| mutation | Gaussian, σ = 0.2, one emitter, batch size 64; parents drawn from a uniformly chosen occupied cell and then uniformly from its front |
| initial population | one batch of uniform random solutions |
| budget | 100 032 evaluations |
| repeats | 11 runs for QualityDiversity.jl, 3 for QDax (about 11 minutes each), one seed each, after a warm-up run |
| hardware, software | as above; QualityDiversity.jl in `float64`, QDax at its default `float32` with its own centroid search |

Both print the same columns, and the first lines of the two outputs give the fitness and the
descriptor of one fixed solution, as a check that the problems agree.

### Results

Median over the runs. The MOQD score is the sum over the cells of the hypervolume of their fronts, and the global
hypervolume is that of the Pareto front of all solutions together.

| library | median s | evals/s | coverage | MOQD score | global hypervolume | largest cell hypervolume | solutions |
|---|---:|---:|---:|---:|---:|---:|---:|
| QualityDiversity.jl | 0.038 | 2 647 300 | 0.332 | 15 507 946 | 139 935 | 135 285 | 1 408 |
| QDax | 686 | 146 | 0.295 | 14 098 354 | 139 930 | 135 329 | 1 398 |

The results agree. The global hypervolume and the hypervolume of the best cell are equal to within
0.03% and 0.04%. Coverage and MOQD score are noisy from run to run: over 30 seeds the coverage of
QualityDiversity.jl lies between 0.25 and 0.43 and the MOQD score between 11.9 and 19.0 million,
and the QDax values are inside those ranges.

QualityDiversity.jl is about 18 000 times faster here. That ratio says little about QDax
in the setting it is made for. QDax inserts the solutions of a batch one at a time in a scan
over arrays of fixed shape, which is slow on one CPU core and says nothing about its speed on
an accelerator, and the problem is cheap. It does show what a single core does with the two
approaches when the objectives are analytic.

### How the fronts were checked

The per-cell update of QualityDiversity.jl was run on the same insertion sequences as QDax's
`MOMERepertoire` in one cell, with two and three objectives and fronts of 4, 10 and 50 solutions
(`test/reference`). Both keep the same solutions in every case, including which solutions a full
front keeps. The hypervolume agrees with the indicator of pymoo to 1e-15 for two, three and
four objectives. One difference is by design: QualityDiversity.jl rejects a solution whose
objectives equal those of a stored one, where QDax keeps both.

### Reproducing

```
# Julia
taskset -c 1 julia --threads=1 benchmark/compare/compare_mome.jl

# QDax
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
REPEATS=3 taskset -c 1 venv/bin/python benchmark/compare/compare_mome_qdax.py   # about 45 minutes
```

## MAP-Elites on a GPU

The MAP-Elites comparison of the first section, with the archive, the variation and the
evaluation on a GPU. `gpu/compare_gpu.jl` runs `MAPElites(prob; device=CuArray)`, which
stores a `DeviceGridArchive` in CUDA arrays and draws children with a
`DeviceVariationEmitter`; the Rastrigin function is a `QDProblem` that a kernel evaluates
one solution per thread. QDax runs `compare_qdax.py` unchanged, with the CUDA build of jax.
pyribs 0.12 keeps its grid archive and Gaussian emitter in numpy, so it cannot run this
on a GPU; it is run on one core of the same machine for reference.

| setting | value |
|---|---|
| problem, archive, mutation, budget, batch sizes | as in the first section |
| repeats | 21 timed runs per batch size, one seed each, after a warm-up run |
| hardware | NVIDIA GeForce RTX 5090 (driver 580.178), AMD Ryzen Threadripper PRO 5955WX; the host side of every run pinned to one core with `taskset` |
| software | Julia 1.13.1, CUDA.jl 6.4.2, KernelAbstractions 0.9.45; Python 3.13, jax 0.11.2 with `jax-cuda12-plugin`, QDax 0.5.0, pyribs 0.12.0 |

### Results

Median time in seconds for the whole budget. QualityDiversity.jl on the CPU is the
`compare.jl` run on this machine, with one thread and with eight.

| batch size | QualityDiversity.jl, GPU, float32 | QDax, GPU, float32 | QDax, GPU, grid index, float32 | QualityDiversity.jl, GPU, float64 | QDax, GPU, float64 | QDax, GPU, grid index, float64 | QualityDiversity.jl, CPU, float32, 1 thread | QualityDiversity.jl, CPU, float32, 8 threads | pyribs, CPU, 1 core |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.178 | 1.340 | 1.374 | 1.397 | 1.556 | 1.453 | 0.173 | 0.109 | 2.335 |
| 100 | 0.495 | 0.478 | 0.481 | 0.562 | 0.590 | 0.523 | 0.166 | 0.096 | 1.024 |
| 1000 | 0.049 | 0.048 | 0.048 | 0.080 | 0.096 | 0.052 | 0.158 | 0.074 | 0.330 |
| 10 000 | 0.0059 | 0.006 | 0.005 | 0.0097 | 0.048 | 0.006 | 0.156 | 0.073 | 0.385 |

On the GPU the two libraries are level in `float32` at every batch size. Each step costs
both about 0.1 ms whatever the batch size, so up to a batch of 100 the GPU is slower than
one CPU core, and at 1000 and above it is faster: at 10 000 the whole budget of 360 000
evaluations takes 6 ms, 26 times less than one core and 12 times less than eight. In
`float64` QDax with its own cell lookup slows down at large batches, because it compares
every descriptor with every centroid and this card computes `float64` slowly; QDax with the
grid index is faster than QualityDiversity.jl at 1000 and 10 000 in `float64`. Coverage and
QD score agree with the CPU runs within the spread between seeds.

**QualityDiversity.jl, GPU, float32**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.0341 | 1.1781 | 1.1348–1.1942 | 305 580 | 0.986 | -132.6 | 4 771 348 |
| 100 | 0.4513 | 0.4951 | 0.4838–0.5073 | 727 104 | 0.987 | -131.4 | 4 786 044 |
| 1000 | 0.0476 | 0.0493 | 0.0488–0.0575 | 7 296 366 | 0.985 | -130.8 | 4 794 318 |
| 10 000 | 0.0057 | 0.0059 | 0.0059–0.0060 | 61 009 918 | 0.938 | -131.9 | 4 606 737 |

**QualityDiversity.jl, GPU, float64**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.1398 | 1.3971 | 1.3401–1.4902 | 257 676 | 0.986 | -137.9 | 4 784 610 |
| 100 | 0.5305 | 0.5622 | 0.5468–0.5839 | 640 315 | 0.987 | -131.6 | 4 783 858 |
| 1000 | 0.0775 | 0.0802 | 0.0787–0.0864 | 4 488 786 | 0.986 | -137.5 | 4 793 983 |
| 10 000 | 0.0094 | 0.0097 | 0.0096–0.0114 | 36 961 210 | 0.944 | -129.6 | 4 622 994 |

**QDax, GPU, float32, centroid search**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.332 | 1.340 | 1.335–1.341 | 268 606 | 0.987 | -133.9 | 4 780 142 |
| 100 | 0.474 | 0.478 | 0.478–0.479 | 752 430 | 0.987 | -133.7 | 4 789 104 |
| 1000 | 0.048 | 0.048 | 0.048–0.048 | 7 531 673 | 0.985 | -131.8 | 4 788 820 |
| 10 000 | 0.006 | 0.006 | 0.006–0.006 | 64 507 412 | 0.944 | -127.8 | 4 627 217 |

**QDax, GPU, float64, centroid search**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.556 | 1.556 | 1.556–1.556 | 231 376 | 0.986 | -133.5 | 4 780 745 |
| 100 | 0.590 | 0.590 | 0.590–0.590 | 610 558 | 0.985 | -133.4 | 4 771 358 |
| 1000 | 0.096 | 0.096 | 0.096–0.096 | 3 756 051 | 0.986 | -131.9 | 4 795 016 |
| 10 000 | 0.048 | 0.048 | 0.048–0.048 | 7 541 751 | 0.942 | -131.1 | 4 613 920 |

### CVT archive

`compare_gpu.jl` also runs a `DeviceCVTArchive` whose centroids are the centers of the
100 × 100 grid cells, the centroids that `compare_qdax.py` gives QDax. This compares the
nearest-centroid search of the two libraries directly: QDax compares every descriptor with
every centroid, and QualityDiversity.jl with the centroids of its bucket in a grid of buckets
built once on the host (about 9 here). Both find the same cells as the grid, and the
`float32` runs of the two QualityDiversity.jl archives give identical QD scores.

| batch size | QualityDiversity.jl, CVT, float32 | QDax, centroid search, float32 | QualityDiversity.jl, CVT, float64 | QDax, centroid search, float64 |
|---:|---:|---:|---:|---:|
| 36 | 1.572 | 1.340 | 1.637 | 1.556 |
| 100 | 0.562 | 0.478 | 0.588 | 0.590 |
| 1000 | 0.057 | 0.048 | 0.086 | 0.096 |
| 10 000 | 0.0068 | 0.006 | 0.0102 | 0.048 |

In `float32` QDax is up to 15% faster; the CVT archive pays for a second kernel per step.
In `float64` the bucket search is five times faster than QDax's at a batch of 10 000,
because it computes about 9 distances per descriptor instead of 10 000.

### CMA-ME

`compare_cma_gpu.jl` runs CMA-ME on the same problem and grid, with 1 200 000
evaluations, on the host and with a `DeviceCMAEmitter`, for 5, 50 and 500 CMA-ES searches
of 12 candidates each (8 host threads, 5 runs per row). The device emitter samples and
updates the searches on the host threads and evaluates and adds all their candidates on
the GPU as one batch, so each search sees the archive as it was before the round; the host
`CMAME` steps its searches one after another.

| searches | host, median s | device, median s | host, QD score | device, QD score |
|---:|---:|---:|---:|---:|
| 5 | 1.410 | 5.029 | 5 035 124 | 5 035 042 |
| 50 | 1.511 | 0.842 | 5 037 730 | 5 037 605 |
| 500 | 2.516 | 0.332 | 5 033 974 | 5 033 762 |

With the 5 searches of the CMA-ME paper a round is a batch of 60, too small for a GPU, and
the device emitter is 3.6 times slower. With 500 searches it is 7.6 times faster than the
host with as many searches, and 4.2 times faster than the host with 5. The QD scores agree
within the spread between seeds. QDax runs one emitter of its pool per step, so it has no
counterpart; pyribs adds the batches of all its emitters at once, on the CPU.

### Not measured

- **Other GPUs.** The kernels are written with KernelAbstractions and should run on AMD and
  Intel GPUs as well, but only CUDA has been timed, and the tests run them on the CPU. Apple
  GPUs have no `Float64`, which the cell lookup needs.
- **CMA-MAE and MOME.** They stay on the host. For reference, QDax on this GPU takes 0.95 s
  for the CMA-ME of the CPU comparison (5 emitters of 12), where QualityDiversity.jl on one
  core takes 0.31 s, and 3.5 s for MOME, where QualityDiversity.jl takes 0.017 s (QDax
  inserts a batch one solution at a time).

### Reproducing

```
# On a machine with several GPUs, choose one by its nvidia-smi index; without
# CUDA_DEVICE_ORDER the CUDA runtime numbers the devices fastest first.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1

# Julia
taskset -c 2 julia benchmark/compare/gpu/compare_gpu.jl
taskset -c 0-7 julia --threads=8 benchmark/compare/gpu/compare_cma_gpu.jl

# QDax and pyribs, with the CUDA build of jax
python -m venv venv && venv/bin/pip install -r benchmark/compare/requirements.txt "jax[cuda12]"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
taskset -c 4 venv/bin/python benchmark/compare/compare_qdax.py
JAX_ENABLE_X64=1 taskset -c 4 venv/bin/python benchmark/compare/compare_qdax.py
QDAX_GRID_LOOKUP=1 taskset -c 4 venv/bin/python benchmark/compare/compare_qdax.py
JAX_ENABLE_X64=1 QDAX_GRID_LOOKUP=1 taskset -c 4 venv/bin/python benchmark/compare/compare_qdax.py
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 taskset -c 2 venv/bin/python benchmark/compare/compare.py
```
