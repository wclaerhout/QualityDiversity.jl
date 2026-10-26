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
function provenance(rows; spread=" Bars span the middle half.")
    hardware = join(sort(unique(h for r in rows for h in split(r["hardware"], "; "))), ", ")
    counts = join(string.(sort(unique(run_counts(rows)))), ", ", " or ")
    return "$hardware. Median of $counts runs.$spread"
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
    Label(fig[5, 1:2], provenance(reduce(vcat, (s.rows for s in drawn))) *
          " Lower is faster. QDax with a grid index is QDax with its centroid search replaced by the index of a regular grid.";
          halign=:left, color=INK.secondary, fontsize=11, word_wrap=true, tellwidth=false)
    save_figure(fig, name)
end

cpu_series(rows) = [
    Series("QualityDiversity.jl, grid index", "QualityDiversity.jl", :solid,
           filter(r -> matches(r; library="QualityDiversity.jl", threads="1"), rows)),
    Series("pyribs, grid index", "pyribs", :solid, filter(r -> matches(r; library="pyribs"), rows)),
    Series("QDax, grid index", "QDax", :solid, filter(r -> matches(r; library="QDax", variant="grid index"), rows)),
]

gpu_series(rows) = [
    Series("QualityDiversity.jl, grid index", "QualityDiversity.jl", :solid,
           filter(r -> matches(r; library="QualityDiversity.jl", variant="grid"), rows)),
    Series("QDax, grid index", "QDax", :solid, filter(r -> matches(r; library="QDax", variant="grid index"), rows)),
    Series("QualityDiversity.jl, centroid search", "QualityDiversity.jl", :dash,
           filter(r -> matches(r; library="QualityDiversity.jl", variant="cvt"), rows)),
    Series("QDax, centroid search", "QDax", :dash,
           filter(r -> matches(r; library="QDax", variant="centroid search"), rows)),
]

# One panel of the algorithms figure: a title and, per library, the runs it draws.
struct AlgorithmPanel
    title::String
    selects::Vector{Pair{String,Function}}
end

# The runs of one setting with a batch size of 10 000 on the GPU.
gpu_map_elites(variant, precision) = r -> matches(r; benchmark="map_elites", device="gpu", precision,
                                                 batch_size="10000", variant)

const CPU_PANELS = [
    AlgorithmPanel("MAP-Elites, batch 10 000",
                   let me(r) = matches(r; benchmark="map_elites", device="cpu", precision="float64",
                                       batch_size="10000")
                       ["QualityDiversity.jl" => r -> me(r) && r["threads"] == "1",
                        "pyribs" => me,
                        "QDax" => r -> me(r) && r["variant"] == "grid index"]
                   end),
    AlgorithmPanel("CMA-ME",
                   [l => r -> matches(r; benchmark="cma", method="CMA-ME", variant="improvement emitters",
                                      precision="float64") for l in first.(LIBRARIES)]),
    AlgorithmPanel("CMA-MAE",
                   [l => r -> matches(r; benchmark="cma", method="CMA-MAE", variant="", precision="float64")
                    for l in first.(LIBRARIES)]),
    AlgorithmPanel("MOME", [l => r -> matches(r; benchmark="mome", precision="float64") for l in first.(LIBRARIES)]),
]

const GPU_PANELS = [
    AlgorithmPanel("$lookup, $precision",
                   ["QualityDiversity.jl" => gpu_map_elites(ours, precision), "QDax" => gpu_map_elites(qdax, precision)])
    for precision in ("float32", "float64")
    for (lookup, ours, qdax) in (("grid index", "grid", "grid index"), ("centroid search", "cvt", "centroid search"))
]

# Draws one bar per library on `ax`, from zero to the median, with a line over the middle
# half and the median written above, and returns the runs drawn.
function algorithm_panel!(ax, runs, panel)
    drawn = Dict{String,String}[]
    hi = 0.0
    x = 0
    for (library, select) in panel.selects
        rows = filter(r -> r["library"] == library && select(r), runs)
        isempty(rows) && continue
        x += 1
        _, color, _ = style(library)
        t = seconds(rows)
        q1, m, q3 = quantile(t, 0.25), median(t), quantile(t, 0.75)
        barplot!(ax, [x], [m]; color, width=0.7)
        cap = 0.08    # half the width of the ends of the line, in bar positions
        linesegments!(ax, [(x, q1), (x, q3), (x - cap, q1), (x + cap, q1), (x - cap, q3), (x + cap, q3)];
                      color=INK.primary, linewidth=1.5)
        text!(ax, x, q3; text=format_seconds(m), align=(:center, :bottom), offset=(0, 4),
              color=INK.secondary, fontsize=12)
        hi = max(hi, q3)
        append!(drawn, rows)
    end
    xlims!(ax, 0.4, max(x, 2) + 0.6)
    ylims!(ax, 0, 1.2hi)    # room above the tallest bar for its label
    return drawn
end

# Median time of every algorithm measured, one panel per algorithm with its own linear axis
# from zero and one bar per library: on one CPU core in float64 in the top row, MAP-Elites on
# the GPU in both precisions below.
function algorithms_figure(runs; name)
    rows_of_panels = (("CPU, one core, float64", CPU_PANELS), ("MAP-Elites on a GPU, batch 10 000", GPU_PANELS))
    ncols = maximum(length(panels) for (_, panels) in rows_of_panels)
    fig = Figure(size=(880, 640))
    Label(fig[1, 1:ncols], "Time of each algorithm"; halign=:left, font=:bold, fontsize=15, tellwidth=false)
    drawn = Dict{String,String}[]
    for (i, (device_title, panels)) in enumerate(rows_of_panels)
        Label(fig[1 + 2i, 1:ncols], device_title; halign=:left, font=:bold, tellwidth=false)
        for (k, panel) in enumerate(panels)
            ax = Axis(fig[2 + 2i, k]; title=panel.title, titlefont=:regular, titlesize=12,
                      ylabel=k == 1 ? "seconds" : "", xticksvisible=false, xticklabelsvisible=false,
                      xgridvisible=false)
            append!(drawn, algorithm_panel!(ax, runs, panel))
        end
    end
    isempty(drawn) && return @warn "no runs for $name"
    Legend(fig[2, 1:ncols], [PolyElement(; color) for (_, color, _) in LIBRARIES],
           collect(first.(LIBRARIES)); orientation=:horizontal, halign=:left, padding=(0, 0, 0, 0),
           tellwidth=false)
    caption = provenance(drawn; spread=" Black lines span the middle half.") *
              " Lower is faster. pyribs has no MOME and QDax no CMA-MAE; pyribs does not run on a GPU." *
              " QDax with a grid index is QDax with its centroid search replaced by the index of a regular grid."
    Label(fig[7, 1:ncols], caption;
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
