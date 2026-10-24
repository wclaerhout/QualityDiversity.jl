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

# The hardware and software of the runs in `rows`, for the caption of a figure.
function provenance(rows)
    hardware = join(sort(unique(r["hardware"] for r in rows)), " | ")
    return "$hardware. Median of $(join(sort(unique(string(n) for n in run_counts(rows))), " or ")) runs; bars span the middle half."
end
run_counts(rows) = values(Dict(key => count(==(key), keys_of(rows)) for key in unique(keys_of(rows))))
keys_of(rows) = [(r["library"], r["variant"], r["precision"], r["threads"], r["batch_size"], r["method"]) for r in rows]

# Ticks at 1, 2 and 5 times a power of ten from `lo` to `hi`, written as plain numbers.
function log_ticks(lo, hi)
    values = [m * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in (1, 2, 5)]
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

# Time against batch size for MAP-Elites on `device`, one panel per precision.
function map_elites_figure(runs, device; title, series_of, name)
    rows = filter(r -> matches(r; benchmark="map_elites", device), runs)
    isempty(rows) && return @warn "no MAP-Elites runs on the $device; $name not drawn"
    fig = Figure(size=(880, 460))
    Label(fig[1, 1:2], title; halign=:left, font=:bold, fontsize=15)
    axes = Axis[]
    lo, hi = Inf, -Inf
    for (k, precision) in enumerate(("float64", "float32"))
        ax = Axis(fig[3, k]; title=precision, xlabel="batch size",
                  ylabel=k == 1 ? "seconds for 360 000 evaluations (lower is faster)" : "")
        l, h = time_panel!(ax, series_of(filter(r -> r["precision"] == precision, rows)))
        lo, hi = min(lo, l), max(hi, h)
        push!(axes, ax)
    end
    ymin = maximum(v for v in log_ticks(lo / 10, lo)[1] if v <= lo / 1.1)   # a tick below the data
    for ax in axes
        xlims!(ax, 28, 13_000)
        ylims!(ax, ymin, hi * 1.3)
        ax.yticks = log_ticks(ymin, hi * 1.3)
    end
    # One entry per series drawn in either panel.
    drawn = filter(s -> !isempty(s.rows), series_of(rows))
    elements = [[LineElement(; color=style(s.library)[2], linestyle=s.linestyle, linewidth=2),
                 MarkerElement(; color=style(s.library)[2], marker=style(s.library)[3], markersize=10,
                               strokecolor=SURFACE, strokewidth=2)] for s in drawn]
    Legend(fig[2, 1:2], elements, [s.label for s in drawn]; orientation=:horizontal, halign=:left,
           padding=(0, 0, 0, 0), patchsize=(42, 14))
    Label(fig[4, 1:2], provenance(reduce(vcat, (s.rows for s in drawn))); halign=:left, color=INK.secondary, fontsize=11)
    save_figure(fig, name)
end

cpu_series(rows) = [
    Series("QualityDiversity.jl", "QualityDiversity.jl", :solid,
           filter(r -> matches(r; library="QualityDiversity.jl", threads="1"), rows)),
    Series("pyribs", "pyribs", :solid, filter(r -> matches(r; library="pyribs"), rows)),
    Series("QDax, grid index", "QDax", :solid, filter(r -> matches(r; library="QDax", variant="grid index"), rows)),
]

gpu_series(rows) = [
    Series("QualityDiversity.jl", "QualityDiversity.jl", :solid,
           filter(r -> matches(r; library="QualityDiversity.jl", variant="grid"), rows)),
    Series("QDax", "QDax", :solid, filter(r -> matches(r; library="QDax", variant="centroid search"), rows)),
    Series("QDax, grid index", "QDax", :dash, filter(r -> matches(r; library="QDax", variant="grid index"), rows)),
]

# Median time of CMA-ME, CMA-MAE and MOME on one CPU core, one row per algorithm and one
# dot per library, on a logarithmic axis.
function algorithms_figure(runs; name)
    algorithms = (("CMA-ME", r -> matches(r; benchmark="cma", method="CMA-ME", variant="improvement emitters")),
                  ("CMA-MAE", r -> matches(r; benchmark="cma", method="CMA-MAE", variant="")),
                  ("MOME", r -> matches(r; benchmark="mome")))
    rows = filter(r -> r["device"] == "cpu" && r["precision"] == "float64" && r["threads"] in ("1", "") &&
                       any(f(r) for (_, f) in algorithms), runs)
    isempty(rows) && return @warn "no CMA-ME, CMA-MAE or MOME runs; $name not drawn"
    fig = Figure(size=(880, 360))
    Label(fig[1, 1], "CMA-ME, CMA-MAE and MOME on one CPU core, float64"; halign=:left, font=:bold, fontsize=15)
    ax = Axis(fig[3, 1]; xlabel="seconds for the whole budget (lower is faster)", xscale=log10,
              yticks=(1:length(algorithms), [a[1] for a in algorithms]), yreversed=true, ygridvisible=false)
    offsets = Dict("QualityDiversity.jl" => -0.22, "pyribs" => 0.0, "QDax" => 0.22)
    lo, hi = Inf, -Inf
    for (k, (_, f)) in enumerate(algorithms), (library, color, marker) in LIBRARIES
        t = seconds(filter(r -> f(r) && r["library"] == library, rows))
        isempty(t) && continue
        y = k + offsets[library]
        q1, m, q3 = quantile(t, 0.25), median(t), quantile(t, 0.75)
        rangebars!(ax, [y], [q1], [q3]; direction=:x, color, whiskerwidth=6)
        scatter!(ax, [m], [y]; color, marker, label=library)
        text!(ax, q3, y; text=format_seconds(m), align=(:left, :center), offset=(8, 0),
              color=INK.secondary, fontsize=12)
        lo, hi = min(lo, q1), max(hi, q3)
    end
    xlims!(ax, lo / 1.5, hi * 4)    # room on the right for the labels
    ylims!(ax, length(algorithms) + 0.5, 0.5)
    ax.xticks = log_ticks(lo / 1.5, hi * 4)
    Legend(fig[2, 1], ax; orientation=:horizontal, halign=:left, merge=true, padding=(0, 0, 0, 0))
    Label(fig[4, 1], provenance(rows) * " pyribs has no MOME and QDax no CMA-MAE.";
          halign=:left, color=INK.secondary, fontsize=11)
    save_figure(fig, name)
end

format_seconds(t) = t >= 10 ? "$(round(Int, t)) s" : t >= 1 ? "$(round(t; digits=1)) s" : "$(round(t; sigdigits=2)) s"

function main()
    apply_theme!()
    runs = read_runs(RESULTS)
    map_elites_figure(runs, "cpu"; title="MAP-Elites on one CPU core", series_of=cpu_series,
                      name="speed-map-elites-cpu.svg")
    algorithms_figure(runs; name="speed-algorithms-cpu.svg")
    map_elites_figure(runs, "gpu"; title="MAP-Elites on a GPU", series_of=gpu_series,
                      name="speed-map-elites-gpu.svg")
end

main()
