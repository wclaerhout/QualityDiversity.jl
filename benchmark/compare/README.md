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
| hardware | AMD Ryzen Threadripper PRO 5955WX (16 cores); every library pinned to one core with `taskset -c 2`, and QualityDiversity.jl also run with 8 threads on cores 0–7 |
| software | Julia 1.13.1, Python 3.12.3, pyribs 0.12.0, numpy 2.5.4, QDax 0.5.0, jax 0.11.2 (CPU) |

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
below; the CSV files in `results/` and every table on this page come from one such run.

## How each library is used

- **QualityDiversity.jl:** `MAPElites` with a `GaussianMutation` and the
  default threaded `evaluate_batch`. The archive records
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
run time: at batch size 1000 a step takes 79 ms with it and 0.33 ms with the
grid index below. So QDax is reported twice:

- **QDax, centroid search** is QDax exactly as documented. It takes about 29 s
  whatever the batch size, so each batch size has only 3 timed runs; the run
  times differ by under 1%.
- **QDax, grid index** replaces that one function with the arithmetic index of a
  regular grid (`QDAX_GRID_LOOKUP=1`). A check at start-up confirms it assigns
  the same cell as QDax's own lookup to 20 000 random descriptors (0 differ).
  Nothing else changes. This is the fair CPU comparison: it is what someone
  who needs QDax to be fast on a CPU would write.

## Results

![MAP-Elites on one CPU core: time against batch size for QualityDiversity.jl, pyribs and QDax](../../docs/src/assets/speed-map-elites-cpu.svg)

Median time in seconds for the whole budget, one core:

| batch size | QualityDiversity.jl, float64 | QualityDiversity.jl, float32 | pyribs | QDax, grid index, float32 | QDax, grid index, float64 | QDax, centroid search, float32 |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.172 | 0.159 | 2.444 | 0.923 | 0.926 | 28.9 |
| 100 | 0.172 | 0.160 | 1.062 | 0.434 | 0.526 | 28.3 |
| 1000 | 0.169 | 0.158 | 0.332 | 0.119 | 0.228 | 28.6 |
| 10 000 | 0.168 | 0.156 | 0.355 | 0.094 | 0.203 | 28.8 |

Time of QDax divided by time of QualityDiversity.jl at the same precision, so
values above 1 mean QualityDiversity.jl is faster:

| batch size | float32 (QDax, grid index) | float64 (QDax, grid index) | QDax, centroid search, against float32 |
|---:|---:|---:|---:|
| 36 | 5.8 | 5.4 | 181 |
| 100 | 2.7 | 3.1 | 177 |
| 1000 | 0.75 | 1.35 | 181 |
| 10 000 | 0.60 | 1.21 | 185 |

Time of pyribs divided by time of QualityDiversity.jl in `float64`:
14.2, 6.2, 2.0 and 2.1 for batch sizes 36, 100, 1000 and 10 000.

In short: QualityDiversity.jl is faster than QDax on a CPU up to a batch size
of 100 in both precisions. In `float32`, the precision QDax uses by default,
QDax with the grid index is 1.3 and 1.7 times faster at batch sizes 1000 and
10 000. In `float64` QualityDiversity.jl stays 1.2 to 1.35 times faster there.
`Float32` makes QualityDiversity.jl 7% to 8% faster, less than the factor of
two QDax gains, because Julia's scalar `cos` is not much faster in `Float32`
while XLA vectorizes it. pyribs is slower than both at every batch size.

QualityDiversity.jl takes about the same time at every batch size (0.16 to
0.17 s, over two million evaluations per second), because each solution costs
the same work: one mutation, one evaluation and one insertion. The libraries
that vectorize a whole batch gain from larger batches.

### Eight threads

`compare.jl` with `--threads=8`, pinned to cores 0–7. Median time in seconds:

| batch size | float64, 1 thread | float64, 8 threads | float32, 1 thread | float32, 8 threads |
|---:|---:|---:|---:|---:|
| 36 | 0.172 | 0.166 | 0.159 | 0.157 |
| 100 | 0.172 | 0.166 | 0.160 | 0.154 |
| 1000 | 0.169 | 0.076 | 0.158 | 0.072 |
| 10 000 | 0.168 | 0.080 | 0.156 | 0.070 |

Only the evaluation is threaded, and only when a batch would take longer than about
100 µs on one thread; a Rastrigin evaluation takes about 200 ns, so batches of 36 and 100
stay on one thread. From 1000 up, eight threads halve the time; the mutation and the
insertion, which stay on one thread, take the rest. The archives are the same as with one
thread: the results do not depend on the number of threads.

### Full tables

**QualityDiversity.jl, float64, 1 thread**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.163 | 0.172 | 0.170–0.178 | 2 089 464 | 0.986 | -129.4 | 4 777 337 |
| 100 | 0.161 | 0.172 | 0.168–0.175 | 2 091 113 | 0.986 | -130.6 | 4 787 824 |
| 1000 | 0.162 | 0.169 | 0.166–0.170 | 2 133 556 | 0.986 | -133.2 | 4 795 986 |
| 10 000 | 0.162 | 0.168 | 0.165–0.174 | 2 142 268 | 0.941 | -130.3 | 4 612 801 |

**QualityDiversity.jl, float32, 1 thread**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.155 | 0.159 | 0.157–0.163 | 2 268 421 | 0.989 | -132.4 | 4 779 777 |
| 100 | 0.154 | 0.160 | 0.157–0.163 | 2 248 346 | 0.987 | -133.8 | 4 780 504 |
| 1000 | 0.150 | 0.158 | 0.156–0.159 | 2 283 371 | 0.986 | -133.3 | 4 795 119 |
| 10 000 | 0.149 | 0.156 | 0.152–0.160 | 2 301 294 | 0.945 | -132.0 | 4 618 760 |

**QualityDiversity.jl, float64, 8 threads**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.162 | 0.166 | 0.163–0.168 | 2 168 805 | 0.986 | -129.4 | 4 777 337 |
| 100 | 0.161 | 0.166 | 0.165–0.167 | 2 173 190 | 0.986 | -130.6 | 4 787 824 |
| 1000 | 0.074 | 0.076 | 0.075–0.077 | 4 751 489 | 0.986 | -133.2 | 4 795 986 |
| 10 000 | 0.072 | 0.080 | 0.076–0.086 | 4 511 013 | 0.941 | -130.3 | 4 612 801 |

**QualityDiversity.jl, float32, 8 threads**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.151 | 0.157 | 0.155–0.160 | 2 299 594 | 0.989 | -132.4 | 4 779 777 |
| 100 | 0.152 | 0.154 | 0.153–0.155 | 2 335 461 | 0.987 | -133.8 | 4 780 504 |
| 1000 | 0.071 | 0.072 | 0.072–0.073 | 4 975 708 | 0.986 | -133.3 | 4 795 119 |
| 10 000 | 0.062 | 0.070 | 0.066–0.072 | 5 163 632 | 0.945 | -132.0 | 4 618 760 |

**pyribs** (`OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`)

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 2.417 | 2.444 | 2.434–2.451 | 147 275 | 0.988 | -136.5 | 4 790 761 |
| 100 | 1.054 | 1.062 | 1.061–1.064 | 338 873 | 0.988 | -134.0 | 4 796 704 |
| 1000 | 0.331 | 0.332 | 0.332–0.333 | 1 084 082 | 0.986 | -132.5 | 4 803 731 |
| 10 000 | 0.354 | 0.355 | 0.354–0.356 | 1 012 882 | 0.945 | -135.2 | 4 630 502 |

**QDax, grid index, float32**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.922 | 0.923 | 0.922–0.924 | 390 237 | 0.987 | -131.1 | 4 780 165 |
| 100 | 0.433 | 0.434 | 0.434–0.435 | 828 998 | 0.988 | -136.9 | 4 789 608 |
| 1000 | 0.119 | 0.119 | 0.119–0.119 | 3 023 285 | 0.985 | -135.3 | 4 788 907 |
| 10 000 | 0.094 | 0.094 | 0.094–0.094 | 3 831 936 | 0.944 | -127.8 | 4 623 051 |

**QDax, grid index, float64**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 0.925 | 0.926 | 0.926–0.927 | 388 687 | 0.986 | -133.5 | 4 780 745 |
| 100 | 0.525 | 0.526 | 0.526–0.526 | 684 820 | 0.985 | -133.3 | 4 771 358 |
| 1000 | 0.227 | 0.228 | 0.227–0.228 | 1 581 990 | 0.986 | -131.9 | 4 795 016 |
| 10 000 | 0.203 | 0.203 | 0.203–0.204 | 1 770 102 | 0.942 | -131.1 | 4 613 920 |

**QDax, centroid search, float32** (3 runs per row)

| batch size | min s | median s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 28.821 | 28.851 | 12 478 | 0.985 | -139.7 | 4 748 242 |
| 100 | 28.317 | 28.344 | 12 701 | 0.986 | -141.1 | 4 789 824 |
| 1000 | 28.467 | 28.568 | 12 602 | 0.987 | -134.7 | 4 804 787 |
| 10 000 | 28.782 | 28.818 | 12 492 | 0.948 | -124.1 | 4 629 452 |

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
- **Spread.** The middle half of the run times lies within 8% of the median for
  every row, and within 1% for the Python libraries, so differences of a few
  percent, such as between the 1- and 8-thread runs at a batch of 36, are level.
- **Threads.** Only QualityDiversity.jl is also run with several threads;
  multi-core JAX and numpy are not measured.

## Reproducing

`run_all.sh` runs all the commands of this page in order. The CPU runs alone:

```
# Julia
taskset -c 2 julia --threads=1 benchmark/compare/compare.jl
taskset -c 0-7 julia --threads=8 benchmark/compare/compare.jl

# pyribs and QDax
python -m venv venv && venv/bin/pip install -r benchmark/compare/requirements.txt
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
taskset -c 2 venv/bin/python benchmark/compare/compare.py
QDAX_GRID_LOOKUP=1 taskset -c 2 venv/bin/python benchmark/compare/compare_qdax.py
JAX_ENABLE_X64=1 QDAX_GRID_LOOKUP=1 taskset -c 2 venv/bin/python benchmark/compare/compare_qdax.py
REPEATS=3 taskset -c 2 venv/bin/python benchmark/compare/compare_qdax.py   # about 10 minutes
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
| hardware, software | as above, one core; QDax at `float32` and `float64`, the others in `float64` |

Each library's own defaults are used for everything not listed. QualityDiversity.jl
steps the emitters one after the other, as the CMA-ME paper does, pyribs adds the
solutions of all emitters at once, and QDax runs one emitter of its pool per step.

### Results

![CMA-ME, CMA-MAE and MOME on one CPU core: median time of each library](../../docs/src/assets/speed-algorithms-cpu.svg)

Median over 11 runs. Time is for the whole budget on one core. Coverage and QD score
are those of the archive of the best solutions (QD-score offset 820).

| method | library | median s | coverage | max fitness | QD score |
|---|---|---:|---:|---:|---:|
| CMA-ME, improvement | QualityDiversity.jl | 0.293 | 0.999 | -126.8 | 4 651 868 |
| | pyribs | 2.042 | 0.999 | -119.8 | 4 642 313 |
| | QDax, float32 | 8.339 | 0.997 | -129.4 | 4 675 812 |
| | QDax, float64 | 8.079 | 0.998 | -129.0 | 4 692 007 |
| CMA-ME, optimizing | QualityDiversity.jl | 0.362 | 0.078 | -18.9 | 483 419 |
| | pyribs | 1.941 | 0.084 | -19.9 | 525 502 |
| | QDax, float32 | 8.372 | 0.622 | -114.7 | 3 248 358 |
| | QDax, float64 | 8.086 | 0.625 | -116.8 | 3 263 107 |
| CMA-ME, random direction | QualityDiversity.jl | 0.328 | 0.999 | -129.4 | 4 311 734 |
| | pyribs | 2.082 | 0.999 | -136.3 | 4 305 520 |
| | QDax, float32 | 8.379 | 0.825 | -129.1 | 3 706 244 |
| | QDax, float64 | 8.094 | 0.822 | -124.7 | 3 683 559 |
| CMA-MAE | QualityDiversity.jl | 0.391 | 0.458 | -96.0 | 2 632 049 |
| | pyribs | 2.301 | 0.561 | -120.6 | 2 992 796 |
| CMA-MAE, covariance update of pyribs | QualityDiversity.jl | 0.409 | 0.560 | -111.2 | 2 992 945 |

QDax has no CMA-MAE.

QualityDiversity.jl is five to seven times faster than pyribs and 22 to 28 times
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
  reaches 0.560 coverage and a QD score of 2.99 million, as pyribs does (0.561 and 2.99
  million). Swapping the textbook term into pyribs lowers its CMA-MAE
  coverage to about 0.46, and the step size rule and the scaling of the path make
  no difference. CMA-ME with improvement emitters is the same with either update
  (QD scores 4.65 million with the textbook term and 4.66 million with the pyribs term).
- **Random direction ranking.** The CMA-ME paper ranks the accepted solutions by
  their projection on the direction, without putting new cells first. pyribs and
  `TwoStageRandomDirectionRanker` rank new cells first; `AcceptedRandomDirectionRanker`
  follows the paper. Here both give the same QD score (4.31 million).

### Not measured

The same caveats as for MAP-Elites hold: no GPU, one core and a cheap objective. The
run times are steady: the fastest run of every row is within 2% of the median. In
addition, 5 emitters of batch 12 and 100 000 evaluations are small next to the CMA-MAE
paper (15 emitters of batch 36 and 10 000 iterations), so the QD scores here do not
compare with its tables.

### Reproducing

```
# Julia
taskset -c 2 julia --threads=1 benchmark/compare/compare_cma.jl

# pyribs and QDax
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
taskset -c 2 venv/bin/python benchmark/compare/compare_cma.py
taskset -c 2 venv/bin/python benchmark/compare/compare_cma_qdax.py
JAX_ENABLE_X64=1 taskset -c 2 venv/bin/python benchmark/compare/compare_cma_qdax.py   # about 6 minutes each
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
| repeats | 11 runs for QualityDiversity.jl, 3 for QDax (about 7 minutes each), one seed each, after a warm-up run |
| hardware, software | as above, one core; both in `float64` (QDax with `JAX_ENABLE_X64=1`), QDax with its own centroid search |

Both print the same columns, and the first lines of the two outputs give the fitness and the
descriptor of one fixed solution, as a check that the problems agree.

### Results

Median over the runs. The MOQD score is the sum over the cells of the hypervolume of their fronts, and the global
hypervolume is that of the Pareto front of all solutions together.

| library | median s | evals/s | coverage | MOQD score | global hypervolume | largest cell hypervolume | solutions |
|---|---:|---:|---:|---:|---:|---:|---:|
| QualityDiversity.jl | 0.019 | 5 208 489 | 0.328 | 15 288 619 | 139 936 | 135 294 | 1 411 |
| QDax | 413 | 242 | 0.323 | 15 208 862 | 139 921 | 135 348 | 1 352 |

The results agree. The global hypervolume and the hypervolume of the best cell are equal to within
0.01% and 0.04%. Coverage and MOQD score are noisy from run to run: over 30 seeds the coverage of
QualityDiversity.jl lies between 0.25 and 0.43 and the MOQD score between 11.9 and 19.0 million,
and the QDax values are inside those ranges.

QualityDiversity.jl is about 20 000 times faster here. That ratio says little about QDax
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
taskset -c 2 julia --threads=1 benchmark/compare/compare_mome.jl

# QDax
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
JAX_ENABLE_X64=1 REPEATS=3 taskset -c 2 venv/bin/python benchmark/compare/compare_mome_qdax.py   # about 30 minutes
```

## MAP-Elites on a GPU

The MAP-Elites comparison of the first section, with the archive, the variation and the
evaluation on a GPU. `gpu/compare_gpu.jl` runs `MAPElites(prob; device=CuArray)`, which
stores a `DeviceGridArchive` in CUDA arrays and draws children with a
`DeviceVariationEmitter`; the Rastrigin function is a `QDProblem` that a kernel evaluates
one solution per thread. QDax runs `compare_qdax.py` unchanged, with the CUDA build of jax.
pyribs 0.12 keeps its grid archive and Gaussian emitter in numpy, so it cannot run this
on a GPU. The CPU runs of the first section were made on the same machine.

| setting | value |
|---|---|
| problem, archive, mutation, budget, batch sizes | as in the first section |
| repeats | 21 timed runs per batch size, one seed each, after a warm-up run |
| hardware | NVIDIA GeForce RTX 5090 (driver 580.178.04), AMD Ryzen Threadripper PRO 5955WX; the host side of every run pinned to one core with `taskset -c 2` |
| software | Julia 1.13.1, CUDA.jl 6.4.2, KernelAbstractions 0.9.45; Python 3.12.3, jax 0.11.2 with `jax-cuda12-plugin`, QDax 0.5.0 |

### Results

![MAP-Elites on a GPU: time against batch size for QualityDiversity.jl and QDax](../../docs/src/assets/speed-map-elites-gpu.svg)

Median time in seconds for the whole budget:

| batch size | QualityDiversity.jl, float32 | QDax, float32 | QDax, grid index, float32 | QualityDiversity.jl, float64 | QDax, float64 | QDax, grid index, float64 |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.210 | 1.334 | 1.346 | 1.224 | 1.576 | 1.453 |
| 100 | 0.421 | 0.478 | 0.483 | 0.466 | 0.596 | 0.523 |
| 1000 | 0.047 | 0.048 | 0.048 | 0.075 | 0.097 | 0.052 |
| 10 000 | 0.0051 | 0.0056 | 0.0049 | 0.0088 | 0.048 | 0.0065 |

On the GPU the two libraries are level in `float32` at every batch size. Each step costs
both about 0.1 ms whatever the batch size, so up to a batch of 100 the GPU is slower than
one CPU core, and at 1000 and above it is faster: at 10 000 the whole budget of 360 000
evaluations takes 5 ms, 31 times less than one core and 14 times less than eight (the
`float32` rows of the first section). In `float64` QDax with its own cell lookup slows
down at large batches, because it compares every descriptor with every centroid and this
card computes `float64` slowly; QDax with the grid index is faster than
QualityDiversity.jl at 1000 and 10 000 in `float64`. Coverage and QD score agree with the
CPU runs within the spread between seeds.

**QualityDiversity.jl, GPU, float32**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.0706 | 1.2101 | 1.1393–1.2680 | 297 501 | 0.986 | -132.6 | 4 772 875 |
| 100 | 0.4069 | 0.4207 | 0.4109–0.4744 | 855 692 | 0.986 | -131.4 | 4 787 345 |
| 1000 | 0.0410 | 0.0467 | 0.0418–0.0538 | 7 705 882 | 0.985 | -130.9 | 4 794 021 |
| 10 000 | 0.0051 | 0.0051 | 0.0051–0.0052 | 70 112 256 | 0.938 | -131.9 | 4 608 536 |

**QualityDiversity.jl, GPU, float64**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.1563 | 1.2243 | 1.2161–1.2675 | 294 048 | 0.986 | -137.5 | 4 787 136 |
| 100 | 0.4387 | 0.4660 | 0.4472–0.5109 | 772 551 | 0.987 | -131.7 | 4 783 865 |
| 1000 | 0.0700 | 0.0748 | 0.0735–0.0771 | 4 814 409 | 0.986 | -137.4 | 4 794 253 |
| 10 000 | 0.0086 | 0.0088 | 0.0087–0.0090 | 41 115 918 | 0.944 | -128.9 | 4 622 224 |

**QDax, GPU, float32, centroid search**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.326 | 1.334 | 1.331–1.337 | 269 864 | 0.987 | -133.9 | 4 780 142 |
| 100 | 0.474 | 0.478 | 0.477–0.478 | 753 641 | 0.987 | -133.7 | 4 789 104 |
| 1000 | 0.047 | 0.048 | 0.048–0.048 | 7 555 599 | 0.985 | -131.8 | 4 788 820 |
| 10 000 | 0.006 | 0.006 | 0.006–0.006 | 64 501 960 | 0.944 | -127.8 | 4 627 217 |

**QDax, GPU, float32, grid index**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.341 | 1.346 | 1.343–1.347 | 267 480 | 0.987 | -131.1 | 4 780 142 |
| 100 | 0.482 | 0.483 | 0.483–0.483 | 745 377 | 0.988 | -136.9 | 4 789 584 |
| 1000 | 0.048 | 0.048 | 0.048–0.048 | 7 450 632 | 0.985 | -133.2 | 4 788 820 |
| 10 000 | 0.005 | 0.005 | 0.005–0.005 | 73 649 994 | 0.944 | -127.8 | 4 627 217 |

**QDax, GPU, float64, centroid search**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.576 | 1.576 | 1.576–1.576 | 228 428 | 0.986 | -133.5 | 4 780 745 |
| 100 | 0.596 | 0.596 | 0.596–0.596 | 604 013 | 0.985 | -133.4 | 4 771 358 |
| 1000 | 0.097 | 0.097 | 0.097–0.097 | 3 728 486 | 0.986 | -131.9 | 4 795 016 |
| 10 000 | 0.048 | 0.048 | 0.048–0.048 | 7 525 006 | 0.942 | -131.1 | 4 613 920 |

**QDax, GPU, float64, grid index**

| batch size | min s | median s | middle half s | evals/s | coverage | max fitness | QD score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.453 | 1.453 | 1.453–1.453 | 247 770 | 0.986 | -133.5 | 4 780 745 |
| 100 | 0.523 | 0.523 | 0.523–0.523 | 687 923 | 0.985 | -133.3 | 4 771 358 |
| 1000 | 0.052 | 0.052 | 0.052–0.052 | 6 871 663 | 0.986 | -131.9 | 4 795 016 |
| 10 000 | 0.006 | 0.006 | 0.006–0.006 | 55 583 452 | 0.942 | -131.1 | 4 613 920 |

### CVT archive

`compare_gpu.jl` also runs a `DeviceCVTArchive` whose centroids are the centers of the
100 × 100 grid cells, the centroids that `compare_qdax.py` gives QDax. This compares the
nearest-centroid search of the two libraries directly: QDax compares every descriptor with
every centroid, and QualityDiversity.jl with the centroids of its bucket in a grid of buckets
built once on the host (about 9 here). Both find the same cells as the grid, and the
`float32` runs of the two QualityDiversity.jl archives give identical QD scores.

| batch size | QualityDiversity.jl, CVT, float32 | QDax, centroid search, float32 | QualityDiversity.jl, CVT, float64 | QDax, centroid search, float64 |
|---:|---:|---:|---:|---:|
| 36 | 1.347 | 1.334 | 1.545 | 1.576 |
| 100 | 0.542 | 0.478 | 0.592 | 0.596 |
| 1000 | 0.065 | 0.048 | 0.096 | 0.097 |
| 10 000 | 0.0151 | 0.0056 | 0.0189 | 0.048 |

In `float32` QDax is faster from a batch of 100, 2.7 times at 10 000. The device archives
keep descriptors in `Float64` and find the nearest centroid in `Float64`, so that host and
device assign the same cells, and this card computes `Float64` slowly; the CVT archive also
pays for a second kernel per step. In `float64` the two are level up to a batch of 1000,
and the bucket search is 2.5 times faster than QDax's at 10 000, because it computes about
9 distances per descriptor instead of 10 000.

### CMA-ME

`compare_cma_gpu.jl` runs CMA-ME on the same problem and grid, with 1 200 000
evaluations, on the host and with a `DeviceCMAEmitter`, for 5, 50 and 500 CMA-ES searches
of 12 candidates each (8 host threads on cores 0–7, 5 runs per row). The device emitter
samples and updates the searches on the host threads and evaluates and adds all their
candidates on the GPU as one batch, so each search sees the archive as it was before the
round; the host `CMAME` steps its searches one after another.

| searches | host, median s | device, median s | host, QD score | device, QD score |
|---:|---:|---:|---:|---:|
| 5 | 1.281 | 4.922 | 5 035 124 | 5 035 042 |
| 50 | 1.322 | 0.831 | 5 037 730 | 5 037 605 |
| 500 | 2.248 | 0.329 | 5 033 974 | 5 033 762 |

With the 5 searches of the CMA-ME paper a round is a batch of 60, too small for a GPU, and
the device emitter is 3.8 times slower. With 500 searches it is 6.8 times faster than the
host with as many searches, and 3.9 times faster than the host with 5. The QD scores agree
within the spread between seeds. QDax runs one emitter of its pool per step, so it has no
counterpart; pyribs adds the batches of all its emitters at once, on the CPU.

### Not measured

- **Other GPUs.** The kernels are written with KernelAbstractions and should run on AMD and
  Intel GPUs as well, but only CUDA has been timed, and the tests run them on the CPU. Apple
  GPUs have no `Float64`, which the cell lookup needs.
- **CMA-MAE and MOME.** They stay on the host. For reference, QDax on this GPU takes 0.95 s
  for the CMA-ME of the CPU comparison (5 emitters of 12), where QualityDiversity.jl on one
  core takes 0.29 s, and 3.5 s for MOME, where QualityDiversity.jl takes 0.019 s (QDax
  inserts a batch one solution at a time). These two QDax runs are not part of `run_all.sh`.

### Reproducing

```
# On a machine with several GPUs, choose one by its nvidia-smi index; without
# CUDA_DEVICE_ORDER the CUDA runtime numbers the devices fastest first.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1

# Julia
taskset -c 2 julia benchmark/compare/gpu/compare_gpu.jl
taskset -c 0-7 julia --threads=8 benchmark/compare/gpu/compare_cma_gpu.jl

# QDax, with the CUDA build of jax
python -m venv venv && venv/bin/pip install -r benchmark/compare/requirements.txt "jax[cuda12]"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
taskset -c 2 venv/bin/python benchmark/compare/compare_qdax.py
JAX_ENABLE_X64=1 taskset -c 2 venv/bin/python benchmark/compare/compare_qdax.py
QDAX_GRID_LOOKUP=1 taskset -c 2 venv/bin/python benchmark/compare/compare_qdax.py
JAX_ENABLE_X64=1 QDAX_GRID_LOOKUP=1 taskset -c 2 venv/bin/python benchmark/compare/compare_qdax.py
```
