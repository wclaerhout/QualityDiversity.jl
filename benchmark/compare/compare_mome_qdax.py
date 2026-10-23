"""MOME on the two-objective BiSphere benchmark with QDax.

Run (one core, float32 as QDax does by default):
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        taskset -c 0 python compare_mome_qdax.py

Environment variables:
    JAX_ENABLE_X64=1    float64, as in compare_mome.jl
    REPEATS=5           timed runs (default 11)

The timed runs go to results/ (see results.py), in a file named after the precision
and the device; the qd_score column holds the MOQD score.

The problem, grid, mutation, batch size, front length and evaluation budget are those of
compare_mome.jl. The whole run, including the initial batch, is one jitted function that scans
the update step, and the QDax metrics are computed after every step. The cells are found with
QDax's own centroid search. pyribs has no multi-objective archive.
"""
import os
import statistics
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import qdax
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.mome import MOME
from qdax.utils.metrics import default_moqd_metrics

import results

warnings.filterwarnings("ignore")

DIM = 10                  # solution dimension
DIMS = (20, 20)           # archive cells
SIGMA = 0.2               # Gaussian mutation standard deviation
BATCH = 64
MAX_FRONT = 50            # solutions kept in the Pareto front of a cell
BUDGET = 100_032          # evaluations, a multiple of the batch size
REPEATS = int(os.environ.get("REPEATS", 11))   # timed runs, one seed each
LIMIT = 5.12
CENTER = 1.0
LOWEST = -DIM * (LIMIT + CENTER) ** 2
REFERENCE_POINT = jnp.array([LOWEST, LOWEST])


def scoring_function(x, key):
    """Objectives and descriptor, as in QualityDiversity.BiSphere."""
    x = jnp.clip(x, -LIMIT, LIMIT)
    near = jnp.sum((x - CENTER) ** 2, axis=1)
    far = jnp.sum((x + CENTER) ** 2, axis=1)
    half = DIM // 2
    measures = jnp.stack([x[:, :half].mean(axis=1), x[:, half:].mean(axis=1)], axis=1)
    return jnp.stack([-near, -far], axis=1), jnp.clip(measures, -LIMIT, LIMIT), {}


def gaussian_mutation(x, key):
    return x + SIGMA * jax.random.normal(key, x.shape, x.dtype)


def make_run():
    emitter = MixingEmitter(
        mutation_fn=gaussian_mutation,
        variation_fn=lambda x1, x2, key: x1,   # unused: variation_percentage is 0
        variation_percentage=0.0,
        batch_size=BATCH,
    )
    mome = MOME(
        scoring_function=scoring_function,
        emitter=emitter,
        metrics_function=partial(default_moqd_metrics, reference_point=REFERENCE_POINT),
        pareto_front_max_length=MAX_FRONT,
    )
    centroids = compute_euclidean_centroids(DIMS, [-LIMIT, -LIMIT], [LIMIT, LIMIT])
    steps = BUDGET // BATCH - 1   # the initial batch is the first of the rounds

    @jax.jit
    def run(key):
        key, init_key, loop_key = jax.random.split(key, 3)
        initial = jax.random.uniform(key, (BATCH, DIM), minval=-LIMIT, maxval=LIMIT)
        repertoire, emitter_state, _ = mome.init(initial, centroids, init_key)
        (repertoire, _, _), metrics = jax.lax.scan(
            mome.scan_update, (repertoire, emitter_state, loop_key), (), length=steps
        )
        return metrics

    return run


def timed(run, seed):
    key = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    metrics = jax.block_until_ready(run(key))
    seconds = time.perf_counter() - start
    moqd = metrics["moqd_score"][-1]
    return (
        seconds,
        float(metrics["coverage"][-1]) / 100,
        float(jnp.sum(jnp.where(jnp.isfinite(moqd), moqd, 0.0))),
        float(metrics["global_hypervolume"][-1]),
        float(metrics["max_hypervolume"][-1]),
        float(metrics["number_solutions"][-1]),
    )


if __name__ == "__main__":
    dtype = jnp.zeros(()).dtype
    x = jnp.linspace(-6, 6, DIM)[None, :]
    fitness, descriptor, _ = scoring_function(x, None)
    print(f"reference fitness {tuple(float(v) for v in fitness[0])} descriptor {tuple(float(v) for v in descriptor[0])} "
          f"reference point {tuple(float(v) for v in REFERENCE_POINT)}")
    print(f"\nQDax {qdax.__version__}, jax {jax.__version__}, {jax.devices()[0]}, {dtype}, {REPEATS} runs, budget {BUDGET}")
    print("| min s | median s | evals/s (median) | coverage | MOQD score | global hypervolume | max hypervolume | solutions |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    device, gpu = results.jax_device()
    software = results.software(("QDax", qdax.__version__), ("jax", jax.__version__))
    hardware = results.hardware() + (f"; {gpu}" if gpu else "")
    run = make_run()
    timed(run, 0)   # compile
    runs = [(seed, *timed(run, seed)) for seed in range(1, REPEATS + 1)]
    fastest, median_time, _, _ = results.summary([r[1] for r in runs])
    print(
        f"| {fastest:.3f} | {median_time:.3f} | {round(BUDGET / median_time)} "
        f"| {statistics.median(r[2] for r in runs):.3f} "
        f"| {statistics.median(r[3] for r in runs):.0f} "
        f"| {statistics.median(r[4] for r in runs):.0f} "
        f"| {statistics.median(r[5] for r in runs):.0f} "
        f"| {statistics.median(r[6] for r in runs):.0f} |"
    )
    results.write_runs(f"mome-qdax-{dtype}-{device}", [
        dict(benchmark="mome", library="QDax", method="MOME", device=device, precision=str(dtype),
             batch_size=BATCH, seed=seed, seconds=seconds, coverage=cov, qd_score=moqd,
             global_hypervolume=global_hv, max_hypervolume=max_hv, solutions=round(solutions),
             software=software, hardware=hardware)
        for seed, seconds, cov, moqd, global_hv, max_hv, solutions in runs
    ])
