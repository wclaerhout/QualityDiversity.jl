"""MAP-Elites on 20-dimensional Rastrigin with pyribs.

Run:  python compare.py
The matching Julia script is compare.jl; both print the same columns and write
the timed runs to results/ (see results.py). The problem, archive, mutation,
batch sizes and evaluation budget are the same.
"""
import os
import statistics
import time

import numpy as np
import ribs
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler

import results

DIM = 20                  # solution dimension
DIMS = (100, 100)         # archive cells
SIGMA = 0.5               # Gaussian mutation standard deviation
BUDGET = 360_000          # evaluations, a multiple of every batch size
BATCH_SIZES = (36, 100, 1000, 10_000)
REPEATS = int(os.environ.get("REPEATS", 21))   # timed runs per batch size, one seed each
LIMIT = 5.12
OFFSET = 41.0 * DIM       # the same QD-score offset as Rastrigin in Julia


def evaluate(x):
    """Rastrigin objective and descriptor, as in QualityDiversity.Rastrigin."""
    x = np.clip(x, -LIMIT, LIMIT)
    objective = -(10 * DIM + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(axis=1))
    half = DIM // 2
    measures = np.stack([x[:, :half].mean(axis=1), x[:, half:].mean(axis=1)], axis=1)
    return objective, np.clip(measures, -LIMIT, LIMIT)


def run_once(batch_size, seed):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    archive = GridArchive(
        solution_dim=DIM,
        dims=DIMS,
        ranges=[(-LIMIT, LIMIT)] * 2,
        qd_score_offset=-OFFSET,   # pyribs subtracts the offset
        seed=seed,
        dtype=np.float64,
    )
    emitter = GaussianEmitter(
        archive,
        sigma=SIGMA,
        initial_solutions=rng.uniform(-LIMIT, LIMIT, size=(batch_size, DIM)),
        batch_size=batch_size,
        seed=seed,
    )
    scheduler = Scheduler(archive, [emitter])
    for _ in range(BUDGET // batch_size):
        solutions = scheduler.ask()
        objective, measures = evaluate(solutions)
        scheduler.tell(objective, measures)
    seconds = time.perf_counter() - start
    stats = archive.stats
    return seconds, stats.coverage, stats.obj_max, stats.qd_score


if __name__ == "__main__":
    reference = evaluate(np.linspace(-6, 6, DIM)[None, :])
    print(f"reference fitness {float(reference[0][0])} descriptor {tuple(float(m) for m in reference[1][0])}")
    print(f"\nPython, pyribs {ribs.__version__}, numpy {np.__version__}")
    print(f"{REPEATS} runs per row")
    print("| batch size | min s | median s | middle half s | evals/s (median) | coverage | max fitness | QD score |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    software = results.software(("pyribs", ribs.__version__), ("numpy", np.__version__))
    hardware = results.hardware()
    rows = []
    for batch_size in BATCH_SIZES:
        run_once(batch_size, 0)   # warm up (numba compiles on first use)
        runs = [(seed, *run_once(batch_size, seed)) for seed in range(1, REPEATS + 1)]
        fastest, median_time, lo, hi = results.summary([r[1] for r in runs])
        print(
            f"| {batch_size} | {fastest:.3f} | {median_time:.3f} | {lo:.3f}–{hi:.3f} "
            f"| {round(BUDGET / median_time)} "
            f"| {statistics.median(r[2] for r in runs):.3f} "
            f"| {statistics.median(r[3] for r in runs):.1f} "
            f"| {statistics.median(r[4] for r in runs):.0f} |"
        )
        rows += [
            dict(benchmark="map_elites", library="pyribs", method="MAP-Elites", variant="grid",
                 device="cpu", precision="float64", threads=1, batch_size=batch_size, seed=seed,
                 seconds=seconds, coverage=cov, max_fitness=best, qd_score=score,
                 software=software, hardware=hardware)
            for seed, seconds, cov, best, score in runs
        ]
    results.write_runs("map_elites-pyribs-cpu", rows)
