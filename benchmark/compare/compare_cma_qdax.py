"""CMA-ME on 20-dimensional Rastrigin with QDax.

Run (one core, float32 as QDax does by default):
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        taskset -c 0 python compare_cma_qdax.py

Environment variables:
    JAX_ENABLE_X64=1    float64, as in compare_cma.jl and compare_cma.py
    REPEATS=5           timed runs per method (default 11)

The timed runs go to results/ (see results.py), in a file named after the
precision and the device.

The problem, grid, number of emitters, batch size, step size and evaluation
budget are those of compare_cma.jl. QDax has the improvement, optimizing and
random direction emitters of CMA-ME, which it runs one at a time from a pool,
and no CMA-MAE. QDax is written for accelerators; this runs it on the CPU. The
whole run, including the initial batch, is one jitted function that scans the
update step, and the QDax metrics (QD score, maximum fitness and coverage) are
computed after every step. The cells are found with QDax's own centroid search.

Differences from the other two libraries, from reading the QDax code:
the first search of every emitter starts at the origin (`mean_init=None`),
as in the other scripts, but the initial batch of random solutions is added to
the archive first; the distribution is updated from all candidates of a batch
with log-decreasing weights, not from the accepted ones (the QDax code says so);
and an emitter measures improvement against the archive as it stood after its
own previous batch, not before the current one.
"""
import os
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import qdax
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import default_qd_metrics

import results

DIM = 20                  # solution dimension
DIMS = (100, 100)         # archive cells
SIGMA0 = 1.0              # initial step size of every search
EMITTERS = 5
BATCH = 12                # 4 + floor(3 log(DIM)), the CMA-ES default
BUDGET = 100_020          # evaluations, a multiple of EMITTERS * BATCH and of BATCH
REPEATS = int(os.environ.get("REPEATS", 11))   # timed runs per method, one seed each
LIMIT = 5.12
OFFSET = 41.0 * DIM       # the same QD-score offset as in the other scripts

METHODS = {
    "CMA-ME, improvement emitters": CMAImprovementEmitter,
    "CMA-ME, optimizing emitters": CMAOptimizingEmitter,
    "CMA-ME, random direction emitters": CMARndEmitter,
}


def scoring_function(x, key):
    """Rastrigin objective and descriptor, as in QualityDiversity.Rastrigin."""
    x = jnp.clip(x, -LIMIT, LIMIT)
    objective = -(10 * DIM + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x), axis=1))
    half = DIM // 2
    measures = jnp.stack([x[:, :half].mean(axis=1), x[:, half:].mean(axis=1)], axis=1)
    return objective, jnp.clip(measures, -LIMIT, LIMIT), {}


def make_run(emitter_class):
    centroids = compute_euclidean_centroids(DIMS, [-LIMIT, -LIMIT], [LIMIT, LIMIT])
    emitter = CMAPoolEmitter(
        num_states=EMITTERS,
        emitter=emitter_class(batch_size=BATCH, genotype_dim=DIM, centroids=centroids, sigma_g=SIGMA0),
    )
    map_elites = MAPElites(
        scoring_function=scoring_function,
        emitter=emitter,
        metrics_function=partial(default_qd_metrics, qd_offset=OFFSET),
    )
    steps = BUDGET // BATCH - 1   # the initial batch is the first of the rounds

    @jax.jit
    def run(key):
        key, init_key, loop_key = jax.random.split(key, 3)
        initial = jax.random.uniform(key, (BATCH, DIM), minval=-LIMIT, maxval=LIMIT)
        repertoire, emitter_state, _ = map_elites.init(initial, centroids, init_key)
        (repertoire, _, _), metrics = jax.lax.scan(
            map_elites.scan_update, (repertoire, emitter_state, loop_key), (), length=steps
        )
        return metrics

    return run


def timed(run, seed):
    key = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    metrics = jax.block_until_ready(run(key))
    seconds = time.perf_counter() - start
    return (
        seconds,
        float(metrics["coverage"][-1]) / 100,
        float(metrics["max_fitness"][-1]),
        float(metrics["qd_score"][-1]),
    )


if __name__ == "__main__":
    dtype = jnp.zeros(()).dtype
    print(f"QDax {qdax.__version__}, jax {jax.__version__}, {jax.devices()[0]}, {dtype}, {REPEATS} runs per row, budget {BUDGET}")
    print("| method | min s | median s | evals/s (median) | coverage | max fitness | QD score |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    device, gpu = results.jax_device()
    software = results.software(("QDax", qdax.__version__), ("jax", jax.__version__))
    hardware = results.hardware() + (f"; {gpu}" if gpu else "")
    precision, name = str(dtype), f"cma-qdax-{dtype}-{device}"
    rows = []
    for method, emitter_class in METHODS.items():
        run = make_run(emitter_class)
        timed(run, 0)   # compile
        runs = [(seed, *timed(run, seed)) for seed in range(1, REPEATS + 1)]
        fastest, median_time, _, _ = results.summary([r[1] for r in runs])
        print(
            f"| {method} | {fastest:.3f} | {median_time:.3f} "
            f"| {round(BUDGET / median_time)} "
            f"| {statistics.median(r[2] for r in runs):.3f} "
            f"| {statistics.median(r[3] for r in runs):.1f} "
            f"| {statistics.median(r[4] for r in runs):.0f} |"
        )
        # "CMA-ME, improvement emitters" is method "CMA-ME" and variant "improvement
        # emitters" in the CSV file, whose fields hold no commas.
        short, _, variant = method.partition(", ")
        rows += [
            dict(benchmark="cma", library='QDax', method=short, variant=variant.replace(", ", "; "),
                 device=device, precision=precision, batch_size=BATCH, searches=EMITTERS, seed=seed,
                 seconds=seconds, coverage=cov, max_fitness=best, qd_score=score,
                 software=software, hardware=hardware)
            for seed, seconds, cov, best, score in runs
        ]
    results.write_runs(name, rows)
