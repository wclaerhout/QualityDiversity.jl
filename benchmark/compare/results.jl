# Timing and CSV output shared by the Julia comparison scripts. Every script writes one
# row per timed run to results/<name>.csv, in the columns below, which results.py writes
# for the Python scripts and plot.jl reads; a column that does not apply is empty.
using QualityDiversity
using Statistics: median, quantile

const COLUMNS = (:benchmark, :library, :method, :variant, :device, :precision, :threads,
                 :batch_size, :searches, :seed, :seconds, :coverage, :max_fitness, :qd_score,
                 :global_hypervolume, :max_hypervolume, :solutions, :software, :hardware)

# Runs `run(seed)` once with seed 0 to compile, then once per seed in `1:repeats`, and
# returns one named tuple per timed run: the seed, the time of `run` and `stats(result)`,
# computed after the clock stops.
function timed_runs(run, stats, repeats)
    run(0)
    return map(1:repeats) do seed
        result = nothing
        seconds = @elapsed result = run(seed)
        (; seed, seconds, stats(result)...)
    end
end

function software()
    dir = pkgdir(QualityDiversity)
    commit = try
        hash = readchomp(`git -C $dir rev-parse --short HEAD`)
        # The CSV files of results/ do not count as changes: a run rewrites them.
        changes = readchomp(`git -C $dir status --porcelain --untracked-files=no -- . ':!benchmark/compare/results'`)
        isempty(changes) ? hash : hash * " with changes"
    catch
        "unknown commit"
    end
    return "Julia $VERSION; QualityDiversity $(pkgversion(QualityDiversity)) ($commit)"
end

hardware() = strip(Sys.cpu_info()[1].model)

# Writes `rows`, each a named tuple with some of `COLUMNS`, to results/<name>.csv.
function write_runs(name, rows)
    dir = joinpath(@__DIR__, "results")
    mkpath(dir)
    path = joinpath(dir, name * ".csv")
    open(path, "w") do io
        println(io, join(COLUMNS, ","))
        for row in rows
            println(io, join((csv_field(get(row, c, "")) for c in COLUMNS), ","))
        end
    end
    println("wrote ", path)
    return path
end

function csv_field(x)
    s = string(x)
    occursin(r"[,\"\n]", s) && throw(ArgumentError("a CSV field may not contain a comma, quote or newline: $s"))
    return s
end

# The columns that the scripts print for a group of timed runs.
function summary(runs)
    times = [r.seconds for r in runs]
    lo, hi = quantile(times, (0.25, 0.75))
    return (min=minimum(times), median=median(times), lo, hi,
            stat=f -> median(f(r) for r in runs))
end
