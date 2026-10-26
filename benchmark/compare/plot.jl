# Draws the speed figures of the documentation from the CSV files in results/, which the
# comparison scripts write (see results.jl), into docs/src/assets.
# Run:  julia --project=docs benchmark/compare/plot.jl
using CairoMakie
using Statistics: median, quantile

const RESULTS = joinpath(@__DIR__, "results")
const ASSETS = joinpath(@__DIR__, "..", "..", "docs", "src", "assets")

# Every row of every CSV file, as a Dict from column name to string.
function read_runs(dir)
    rows = Dict{String,String}[]
    for file in sort(readdir(dir; join=true))
        endswith(file, ".csv") || continue
        lines = readlines(file)
        header = split(lines[1], ',')
        for line in lines[2:end]
            isempty(line) && continue
            fields = split(line, ',')
            length(fields) == length(header) ||
                error("$file: a row has $(length(fields)) fields, the header $(length(header))")
            push!(rows, Dict(String(h) => String(f) for (h, f) in zip(header, fields)))
        end
    end
    return rows
end

matches(row; kwargs...) = all(row[String(k)] == v for (k, v) in kwargs)
seconds(rows) = [parse(Float64, r["seconds"]) for r in rows]

# The color and marker of each library are fixed, whatever a figure shows.
const INK = (primary=colorant"#0b0b0b", secondary=colorant"#52514e", grid=colorant"#e6e5e0")
const SURFACE = colorant"#fcfcfb"
const LIBRARIES = (("QualityDiversity.jl", colorant"#2a78d6", :circle),
                   ("pyribs", colorant"#eb6834", :rect),
                   ("QDax", colorant"#1baf7a", :diamond))
style(library) = only(s for s in LIBRARIES if s[1] == library)

function apply_theme!()
    set_theme!(Theme(fontsize=13, backgroundcolor=SURFACE, textcolor=INK.primary,
                     Axis=(backgroundcolor=SURFACE, xgridcolor=INK.grid, ygridcolor=INK.grid,
                           leftspinecolor=INK.secondary, bottomspinecolor=INK.secondary,
                           xtickcolor=INK.secondary, ytickcolor=INK.secondary,
                           topspinevisible=false, rightspinevisible=false,
                           xlabelcolor=INK.secondary, ylabelcolor=INK.secondary,
                           titlealign=:left, titlefont=:bold),
                     Legend=(framevisible=false, backgroundcolor=SURFACE),
                     Lines=(linewidth=2,), Scatter=(markersize=10, strokecolor=SURFACE, strokewidth=2)))
end

# The hardware of the runs in `rows` and their number, for the caption of a figure.
function provenance(rows)
    hardware = join(sort(unique(h for r in rows for h in split(r["hardware"], "; "))), ", ")
    counts = join(string.(sort(unique(run_counts(rows)))), ", ", " or ")
    return "$hardware. Median of $counts runs; bars span the middle half."
end
run_counts(rows) = values(Dict(key => count(==(key), keys_of(rows)) for key in unique(keys_of(rows))))
keys_of(rows) = [(r["library"], r["device"], r["variant"], r["precision"], r["threads"], r["batch_size"], r["method"]) for r in rows]

# Ticks at 1, 2 and 5 times a power of ten from `lo` to `hi`, written as plain numbers.
# Over more than three powers of ten only the powers themselves are ticked.
function log_ticks(lo, hi)
    multiples = hi / lo > 1000 ? (1,) : (1, 2, 5)
    values = [m * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in multiples]
    values = filter(v -> lo <= v <= hi, values)
    return (values, [v >= 1 ? string(round(Int, v)) : string(round(v; sigdigits=1)) for v in values])
end

# One series of a time-against-batch-size panel: the runs of one library and variant.
struct Series
    label::String
    library::String
    linestyle::Symbol
    rows::Vector{Dict{String,String}}
end

# Draws the series on `ax` and returns the smallest and largest value drawn.
function time_panel!(ax, series)
    lo_all, hi_all = Inf, -Inf
    for s in series
        isempty(s.rows) && continue
        _, color, marker = style(s.library)
        sizes = sort(unique(parse(Int, r["batch_size"]) for r in s.rows))
        times = [seconds(filter(r -> r["batch_size"] == string(b), s.rows)) for b in sizes]
        lo = [quantile(t, 0.25) for t in times]
        hi = [quantile(t, 0.75) for t in times]
        rangebars!(ax, sizes, lo, hi; color, whiskerwidth=6, linewidth=1.5)
        lines!(ax, sizes, median.(times); color, linestyle=s.linestyle, label=s.label)
        scatter!(ax, sizes, median.(times); color, marker, label=s.label)
        lo_all, hi_all = min(lo_all, minimum(lo)), max(hi_all, maximum(hi))
    end
    ax.xscale = log10
    ax.yscale = log10
    ax.xticks = ([36, 100, 1000, 10_000], ["36", "100", "1000", "10 000"])
    return lo_all, hi_all
end

function save_figure(fig, name)
    mkpath(ASSETS)
    path = joinpath(ASSETS, name)
    save(path, fig)
    println("wrote ", path)
end

# Time against batch size for MAP-Elites: the CPU in the top row and the GPU in the bottom
# row, one column per precision. Solid lines use a grid lookup, dashed ones a search over
# the centroids of the cells.
function map_elites_figure(runs; name)
    rows = filter(r -> r["benchmark"] == "map_elites", runs)
    isempty(rows) && return @warn "no MAP-Elites runs; $name not drawn"
    devices = (("cpu", "CPU, one core", cpu_series), ("gpu", "GPU", gpu_series))
    fig = Figure(size=(880, 820))
    Label(fig[1, 1:2], "MAP-Elites: time against batch size"; halign=:left, font=:bold, fontsize=15)
    drawn = Series[]
    for (i, (device, device_title, series_of)) in enumerate(devices)
        device_rows = filter(r -> r["device"] == device, rows)
        axes = Axis[]
        lo, hi = Inf, -Inf
        for (k, precision) in enumerate(("float64", "float32"))
            ax = Axis(fig[2 + i, k]; title="$device_title, $precision",
                      xlabel=i == length(devices) ? "batch size" : "",
                      ylabel=k == 1 ? "seconds for 360 000 evaluations" : "")
            series = series_of(filter(r -> r["precision"] == precision, device_rows))
            l, h = time_panel!(ax, series)
            lo, hi = min(lo, l), max(hi, h)
            append!(drawn, filter(s -> !isempty(s.rows), series))
            push!(axes, ax)
        end
        isfinite(lo) || continue
        ymin = maximum(v for v in log_ticks(lo / 10, lo)[1] if v <= lo / 1.1)   # a tick below the data
        for ax in axes
            xlims!(ax, 28, 13_000)
            ylims!(ax, ymin, hi * 1.3)
            ax.yticks = log_ticks(ymin, hi * 1.3)
        end
    end
    # One entry per label drawn in any panel.
    labels = unique(s.label for s in drawn)
    entry(s) = [LineElement(; color=style(s.library)[2], linestyle=s.linestyle, linewidth=2),
                MarkerElement(; color=style(s.library)[2], marker=style(s.library)[3], markersize=10,
                              strokecolor=SURFACE, strokewidth=2)]
    Legend(fig[2, 1:2], [entry(first(s for s in drawn if s.label == l)) for l in labels], labels;
           orientation=:horizontal, halign=:left, nbanks=2, padding=(0, 0, 0, 0), patchsize=(42, 14))
    Label(fig[5, 1:2], provenance(reduce(vcat, (s.rows for s in drawn))) * " Lower is faster.";
          halign=:left, color=INK.secondary, fontsize=11, word_wrap=true, tellwidth=false)
    save_figure(fig, name)
end

cpu_series(rows) = [
    Series("QualityDiversity.jl, grid", "QualityDiversity.jl", :solid,
           filter(r -> matches(r; library="QualityDiversity.jl", threads="1"), rows)),
    Series("pyribs, grid", "pyribs", :solid, filter(r -> matches(r; library="pyribs"), rows)),
    Series("QDax, grid index", "QDax", :solid, filter(r -> matches(r; library="QDax", variant="grid index"), rows)),
]

gpu_series(rows) = [
    Series("QualityDiversity.jl, grid", "QualityDiversity.jl", :solid,
           filter(r -> matches(r; library="QualityDiversity.jl", variant="grid"), rows)),
    Series("QDax, grid index", "QDax", :solid, filter(r -> matches(r; library="QDax", variant="grid index"), rows)),
    Series("QualityDiversity.jl, CVT", "QualityDiversity.jl", :dash,
           filter(r -> matches(r; library="QualityDiversity.jl", variant="cvt"), rows)),
    Series("QDax, centroid search", "QDax", :dash,
           filter(r -> matches(r; library="QDax", variant="centroid search"), rows)),
]

# One row of the algorithms figure: a label and, per library, the runs it draws.
struct AlgorithmRow
    label::String
    selects::Vector{Pair{String,Function}}
end

# The runs of one setting with a batch size of 10 000 on the GPU, in float32.
gpu_map_elites(variant) = r -> matches(r; benchmark="map_elites", device="gpu", precision="float32",
                                      batch_size="10000", variant)

const CPU_ROWS = [
    AlgorithmRow("MAP-Elites\nbatch 10 000",
                 let me(r) = matches(r; benchmark="map_elites", device="cpu", precision="float64",
                                     batch_size="10000")
                     ["QualityDiversity.jl" => r -> me(r) && r["threads"] == "1",
                      "pyribs" => me,
                      "QDax" => r -> me(r) && r["variant"] == "grid index"]
                 end),
    AlgorithmRow("CMA-ME",
                 [l => r -> matches(r; benchmark="cma", method="CMA-ME", variant="improvement emitters",
                                    precision="float64") for l in first.(LIBRARIES)]),
    AlgorithmRow("CMA-MAE",
                 [l => r -> matches(r; benchmark="cma", method="CMA-MAE", variant="", precision="float64")
                  for l in first.(LIBRARIES)]),
    AlgorithmRow("MOME", [l => r -> matches(r; benchmark="mome", precision="float64") for l in first.(LIBRARIES)]),
]

const GPU_ROWS = [
    AlgorithmRow("MAP-Elites, grid\nbatch 10 000",
                 ["QualityDiversity.jl" => gpu_map_elites("grid"), "QDax" => gpu_map_elites("grid index")]),
    AlgorithmRow("MAP-Elites, CVT\nbatch 10 000",
                 ["QualityDiversity.jl" => gpu_map_elites("cvt"), "QDax" => gpu_map_elites("centroid search")]),
]

# Draws one dot per library and row on `ax`, at the median with a bar over the middle half,
# and returns the runs drawn.
function algorithms_panel!(ax, runs, algorithm_rows)
    offsets = Dict("QualityDiversity.jl" => -0.25, "pyribs" => 0.0, "QDax" => 0.25)
    drawn = Dict{String,String}[]
    lo, hi = Inf, -Inf
    for (k, row) in enumerate(algorithm_rows), (library, select) in row.selects
        rows = filter(r -> r["library"] == library && select(r), runs)
        isempty(rows) && continue
        _, color, marker = style(library)
        t = seconds(rows)
        y = k + offsets[library]
        q1, m, q3 = quantile(t, 0.25), median(t), quantile(t, 0.75)
        rangebars!(ax, [y], [q1], [q3]; direction=:x, color, whiskerwidth=6)
        scatter!(ax, [m], [y]; color, marker, label=library)
        text!(ax, q3, y; text=format_seconds(m), align=(:left, :center), offset=(8, 0),
              color=INK.secondary, fontsize=12)
        lo, hi = min(lo, q1), max(hi, q3)
        append!(drawn, rows)
    end
    xlims!(ax, lo / 1.5, hi * 4)    # room on the right for the labels
    ylims!(ax, length(algorithm_rows) + 0.5, 0.5)
    ax.xticks = log_ticks(lo / 1.5, hi * 4)
    return drawn
end

# Median time of every algorithm measured: on one CPU core in float64 at the top, on the GPU
# in float32 below, one row per algorithm and one dot per library, on a logarithmic axis.
function algorithms_figure(runs; name)
    panels = (("CPU, one core, float64", CPU_ROWS, 4), ("GPU, float32", GPU_ROWS, 2))
    fig = Figure(size=(880, 700))
    Label(fig[1, 1], "Time of each algorithm"; halign=:left, font=:bold, fontsize=15, tellwidth=false)
    drawn = Dict{String,String}[]
    local ax
    for (i, (title, algorithm_rows, n)) in enumerate(panels)
        ax = Axis(fig[2 + i, 1]; title, xscale=log10, ygridvisible=false, yreversed=true,
                  yticks=(1:n, [r.label for r in algorithm_rows]),
                  xlabel=i == length(panels) ? "seconds for the whole budget (lower is faster)" : "")
        append!(drawn, algorithms_panel!(ax, runs, algorithm_rows))
        rowsize!(fig.layout, 2 + i, Auto(n))
    end
    isempty(drawn) && return @warn "no runs for $name"
    Legend(fig[2, 1], [MarkerElement(; color, marker, markersize=10) for (_, color, marker) in LIBRARIES],
           collect(first.(LIBRARIES)); orientation=:horizontal, halign=:left, padding=(0, 0, 0, 0),
           tellwidth=false)
    Label(fig[5, 1], provenance(drawn) * " pyribs has no MOME and QDax no CMA-MAE; pyribs does not run on a GPU.";
          halign=:left, color=INK.secondary, fontsize=11, word_wrap=true, tellwidth=false)
    save_figure(fig, name)
end

format_seconds(t) = t >= 10 ? "$(round(Int, t)) s" : t >= 1 ? "$(round(t; digits=1)) s" :
                    t >= 0.01 ? "$(round(t; sigdigits=2)) s" : "$(round(1000t; sigdigits=2)) ms"

function main()
    apply_theme!()
    runs = read_runs(RESULTS)
    algorithms_figure(runs; name="speed-algorithms.svg")
    map_elites_figure(runs; name="speed-map-elites.svg")
end

main()
