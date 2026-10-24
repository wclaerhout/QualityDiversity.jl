"""CMA-ME and CMA-MAE on 20-dimensional Rastrigin with pyribs.

Run:  python compare_cma.py
The matching scripts are compare_cma.jl (QualityDiversity.jl) and
compare_cma_qdax.py (QDax); all print the same columns and write the timed
runs to results/ (see results.py). The problem, archive,
number of emitters, batch size, step size and evaluation budget are the same.
"""
import os
import statistics
import time

import numpy as np
import ribs
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

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
LEARNING_RATE = 0.01

# ranker, selection rule, restart rule and whether the archive has thresholds
METHODS = {
    "CMA-ME, improvement emitters": ("2imp", "filter", "no_improvement", False),
    "CMA-ME, optimizing emitters": ("obj", "mu", "basic", False),
    "CMA-ME, random direction emitters": ("2rd", "filter", "no_improvement", False),
    "CMA-MAE": ("imp", "mu", "basic", True),
}


def evaluate(x):
    """Rastrigin objective and descriptor, as in QualityDiversity.Rastrigin."""
    x = np.clip(x, -LIMIT, LIMIT)
    objective = -(10 * DIM + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(axis=1))
    half = DIM // 2
    measures = np.stack([x[:, :half].mean(axis=1), x[:, half:].mean(axis=1)], axis=1)
    return objective, np.clip(measures, -LIMIT, LIMIT)


def make_archive(seed, thresholds):
    return GridArchive(
        solution_dim=DIM,
        dims=DIMS,
        ranges=[(-LIMIT, LIMIT)] * 2,
        qd_score_offset=-OFFSET,   # pyribs subtracts the offset
        learning_rate=LEARNING_RATE if thresholds else 1.0,
        threshold_min=-OFFSET if thresholds else -np.inf,
        seed=seed,
        dtype=np.float64,
    )


def run_once(method, seed):
    ranker, selection, restart, thresholds = METHODS[method]
    start = time.perf_counter()
    archive = make_archive(seed, thresholds)
    # CMA-MAE keeps the best solution of every cell in a second archive.
    result_archive = make_archive(seed, False) if thresholds else None
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=np.zeros(DIM),
            sigma0=SIGMA0,
            ranker=ranker,
            selection_rule=selection,
            restart_rule=restart,
            batch_size=BATCH,
            seed=seed * 100 + i,
        )
        for i in range(EMITTERS)
    ]
    scheduler = Scheduler(archive, emitters, result_archive=result_archive)
    for _ in range(BUDGET // (EMITTERS * BATCH)):
        solutions = scheduler.ask()
        objective, measures = evaluate(solutions)
        scheduler.tell(objective, measures)
    seconds = time.perf_counter() - start
    stats = (result_archive or archive).stats
    return seconds, stats.coverage, stats.obj_max, stats.qd_score


if __name__ == "__main__":
    print(f"Python, pyribs {ribs.__version__}, numpy {np.__version__}, {REPEATS} runs per row, budget {BUDGET}")
    print("| method | min s | median s | evals/s (median) | coverage | max fitness | QD score |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    software = results.software(("pyribs", ribs.__version__), ("numpy", np.__version__))
    hardware = results.hardware()
    device, precision, name = "cpu", "float64", "cma-pyribs-cpu"
    rows = []
    for method in METHODS:
        run_once(method, 0)   # warm up (numba compiles on first use)
        runs = [(seed, *run_once(method, seed)) for seed in range(1, REPEATS + 1)]
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
            dict(benchmark="cma", library='pyribs', method=short, variant=variant.replace(", ", "; "),
                 device=device, precision=precision, batch_size=BATCH, searches=EMITTERS, seed=seed,
                 seconds=seconds, coverage=cov, max_fitness=best, qd_score=score,
                 software=software, hardware=hardware)
            for seed, seconds, cov, best, score in runs
        ]
    results.write_runs(name, rows)
