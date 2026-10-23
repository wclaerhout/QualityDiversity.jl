"""MAP-Elites on 20-dimensional Rastrigin with QDax.

Run (one core, float32 as QDax does by default):
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        taskset -c 0 python compare_qdax.py

Environment variables:
    JAX_ENABLE_X64=1    float64, as in compare.jl and compare.py
    QDAX_GRID_LOOKUP=1  replace QDax's cell lookup by direct arithmetic (below)
    REPEATS=5           timed runs per batch size (default 21)

The timed runs go to results/ (see results.py), in a file named after the cell
lookup, the precision and the device.

QDax finds the cell of a descriptor by comparing it with every centroid, which
suits any tessellation and an accelerator, but costs batch size × number of
cells on a CPU and dominates the run time here. With QDAX_GRID_LOOKUP=1 that one
function is replaced by the index of a regular grid; a check at start-up
confirms it returns the cells QDax's own lookup returns.

The problem, grid, mutation, batch sizes and evaluation budget are those of
compare.jl. QDax is written for accelerators; this runs it on the CPU. The
whole run, including the initial batch, is one jitted function that scans the
update step, the way the QDax documentation recommends, and the QDax metrics
(QD score, maximum fitness and coverage) are computed after every step.
"""
import os
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import qdax
from qdax.core.containers import mapelites_repertoire
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import default_qd_metrics

import results

DIM = 20                  # solution dimension
DIMS = (100, 100)         # archive cells
SIGMA = 0.5               # Gaussian mutation standard deviation
BUDGET = 360_000          # evaluations, a multiple of every batch size
BATCH_SIZES = (36, 100, 1000, 10_000)
REPEATS = int(os.environ.get("REPEATS", 21))   # timed runs per batch size, one seed each
LIMIT = 5.12
OFFSET = 41.0 * DIM       # the same QD-score offset as Rastrigin in Julia


def scoring_function(x, key):
    """Rastrigin objective and descriptor, as in QualityDiversity.Rastrigin."""
    x = jnp.clip(x, -LIMIT, LIMIT)
    objective = -(10 * DIM + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x), axis=1))
    half = DIM // 2
    measures = jnp.stack([x[:, :half].mean(axis=1), x[:, half:].mean(axis=1)], axis=1)
    return objective, jnp.clip(measures, -LIMIT, LIMIT), {}


def grid_cells_indices(descriptors, centroids):
    """Cell of each descriptor on the regular grid, in the order of the centroids."""
    cells = jnp.array(DIMS)
    scaled = (descriptors + LIMIT) / (2 * LIMIT) * cells
    index = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, cells - 1)
    return index[:, 1] * DIMS[0] + index[:, 0]   # meshgrid order: first descriptor varies fastest


def use_grid_lookup():
    """Replace QDax's lookup after checking that it agrees with it."""
    centroids = compute_euclidean_centroids(DIMS, [-LIMIT, -LIMIT], [LIMIT, LIMIT])
    descriptors = jax.random.uniform(jax.random.PRNGKey(7), (20_000, 2), minval=-LIMIT, maxval=LIMIT)
    expected = mapelites_repertoire.get_cells_indices(descriptors, centroids)
    actual = grid_cells_indices(descriptors, centroids)
    mismatches = int(jnp.sum(expected != actual))
    assert mismatches <= 2, f"{mismatches} of 20000 descriptors land in a different cell"
    mapelites_repertoire.get_cells_indices = grid_cells_indices
    return mismatches


def gaussian_mutation(x, key):
    return x + SIGMA * jax.random.normal(key, x.shape, x.dtype)


def make_run(batch_size):
    emitter = MixingEmitter(
        mutation_fn=gaussian_mutation,
        variation_fn=lambda x1, x2, key: x1,   # unused: variation_percentage is 0
        variation_percentage=0.0,
        batch_size=batch_size,
    )
    map_elites = MAPElites(
        scoring_function=scoring_function,
        emitter=emitter,
        metrics_function=partial(default_qd_metrics, qd_offset=OFFSET),
    )
    centroids = compute_euclidean_centroids(DIMS, [-LIMIT, -LIMIT], [LIMIT, LIMIT])
    steps = BUDGET // batch_size - 1   # the initial batch is the first of the rounds

    @jax.jit
    def run(key):
        key, init_key, loop_key = jax.random.split(key, 3)
        initial = jax.random.uniform(key, (batch_size, DIM), minval=-LIMIT, maxval=LIMIT)
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
    grid = os.environ.get("QDAX_GRID_LOOKUP") == "1"
    lookup = "direct grid index" if grid else "QDax centroid search"
    if grid:
        print(f"grid lookup agrees with the centroid search on all but {use_grid_lookup()} of 20000 descriptors")
    print(f"QDax {qdax.__version__}, jax {jax.__version__}, {jax.devices()[0]}, {dtype}, cell lookup: {lookup}, {REPEATS} runs per row")
    print("| batch size | min s | median s | middle half s | evals/s (median) | coverage | max fitness | QD score |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    device, gpu = results.jax_device()
    software = results.software(("QDax", qdax.__version__), ("jax", jax.__version__))
    hardware = results.hardware() + (f"; {gpu}" if gpu else "")
    variant = "grid index" if grid else "centroid search"
    rows = []
    for batch_size in BATCH_SIZES:
        run = make_run(batch_size)
        timed(run, 0)   # compile
        runs = [(seed, *timed(run, seed)) for seed in range(1, REPEATS + 1)]
        fastest, median_time, lo, hi = results.summary([r[1] for r in runs])
        print(
            f"| {batch_size} | {fastest:.3f} | {median_time:.3f} | {lo:.3f}–{hi:.3f} "
            f"| {round(BUDGET / median_time)} "
            f"| {statistics.median(r[2] for r in runs):.3f} "
            f"| {statistics.median(r[3] for r in runs):.1f} "
            f"| {statistics.median(r[4] for r in runs):.0f} |"
        )
        rows += [
            dict(benchmark="map_elites", library="QDax", method="MAP-Elites", variant=variant,
                 device=device, precision=str(dtype), batch_size=batch_size, seed=seed,
                 seconds=seconds, coverage=cov, max_fitness=best, qd_score=score,
                 software=software, hardware=hardware)
            for seed, seconds, cov, best, score in runs
        ]
    results.write_runs(f"map_elites-qdax-{'grid' if grid else 'centroids'}-{dtype}-{device}", rows)
