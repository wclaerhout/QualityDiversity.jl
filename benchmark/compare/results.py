"""Timing and CSV output shared by the Python comparison scripts.

Every script writes one row per timed run to results/<name>.csv, in the columns of
results.jl, which the Julia scripts write and plot.jl reads; a column that does not
apply is empty.
"""
import os
import platform
import statistics
import subprocess

COLUMNS = (
    "benchmark", "library", "method", "variant", "device", "precision", "threads",
    "batch_size", "searches", "seed", "seconds", "coverage", "max_fitness", "qd_score",
    "global_hypervolume", "max_hypervolume", "solutions", "software", "hardware",
)

HERE = os.path.dirname(os.path.abspath(__file__))


def commit():
    """The commit of the repository, marked when tracked files outside results/ changed."""
    try:
        run = lambda *args: subprocess.run(["git", "-C", HERE, *args], capture_output=True,
                                           text=True, check=True).stdout.strip()
        hash = run("rev-parse", "--short", "HEAD")
        # The CSV files of results/ do not count as changes: a run rewrites them.
        changes = run("status", "--porcelain", "--untracked-files=no", "--", "../..",
                      ":!../../benchmark/compare/results")
        return hash if not changes else hash + " with changes"
    except (OSError, subprocess.CalledProcessError):
        return "unknown commit"


def software(*packages):
    """Python and the given `(name, version)` pairs, with the commit of the scripts."""
    names = "; ".join(f"{name} {version}" for name, version in packages)
    return f"Python {platform.python_version()}; {names} ({commit()})"


def hardware():
    """The model name of the CPU."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def jax_device():
    """`("cpu", "")` or `("gpu", name of the device)` for the default jax device."""
    import jax

    device = jax.devices()[0]
    return ("cpu", "") if device.platform == "cpu" else ("gpu", device.device_kind)


def csv_field(x):
    s = "" if x is None else str(x)
    if any(c in s for c in ',"\n'):
        raise ValueError(f"a CSV field may not contain a comma, quote or newline: {s}")
    return s


def write_runs(name, rows):
    """Write `rows`, each a dict with some of COLUMNS, to results/<name>.csv."""
    directory = os.path.join(HERE, "results")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name + ".csv")
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(csv_field(row.get(c)) for c in COLUMNS) + "\n")
    print(f"wrote {path}")
    return path


def summary(times):
    """Minimum, median and quartiles of the run times."""
    lo, _, hi = statistics.quantiles(times, n=4, method="inclusive")
    return min(times), statistics.median(times), lo, hi
