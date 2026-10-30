# Build the documentation: julia --project=docs docs/make.jl
using Documenter
using Literate
using QualityDiversity
using CairoMakie
CairoMakie.activate!(type="png")
using KernelAbstractions
using Tables

# The Makie and Tables extensions have docstrings of their own, which the API reference shows.
const QualityDiversityMakieExt = Base.get_extension(QualityDiversity, :QualityDiversityMakieExt)
const QualityDiversityTablesExt = Base.get_extension(QualityDiversity, :QualityDiversityTablesExt)

const EXAMPLES = joinpath(@__DIR__, "..", "examples")
const GENERATED = joinpath(@__DIR__, "src", "tutorials")

# The tutorials are the scripts in examples/, turned into pages by Literate, so the code
# in the documentation is the code that runs.
const TUTORIALS = ["rastrigin_tutorial.jl", "graphs.jl", "noisy.jl", "mome.jl", "heatmap.jl"]
const EXTRA_EXAMPLES = ["colours.jl", "attractors.jl", "arm.jl"]

rm(GENERATED; force=true, recursive=true)
for file in [TUTORIALS; EXTRA_EXAMPLES]
    Literate.markdown(joinpath(EXAMPLES, file), GENERATED; documenter=true, credit=false)
end

makedocs(;
    sitename="QualityDiversity.jl",
    modules=[QualityDiversity, QualityDiversityMakieExt, QualityDiversityTablesExt],
    authors="Ward Claerhout",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://wclaerhout.github.io/QualityDiversity.jl",
        edit_link="main",
        assets=["assets/favicon.ico"],
        size_threshold_warn=200 * 1024,
        size_threshold=400 * 1024,
    ),
    checkdocs=:public,
    pages=[
        "Home" => "index.md",
        "Tutorials" => ["tutorials/$(splitext(file)[1]).md" for file in TUTORIALS],
        "Extra examples" => ["tutorials/$(splitext(file)[1]).md" for file in EXTRA_EXAMPLES],
        "Guide" => [
            "guide/concepts.md",
            "guide/algorithms.md",
            "guide/extending.md",
            "guide/gpu.md",
            "guide/comparison.md",
            "guide/verification.md",
        ],
        "API reference" => [
            "api/problems.md",
            "api/archives.md",
            "api/variation.md",
            "api/emitters.md",
            "api/algorithms.md",
            "api/metrics.md",
            "api/device.md",
            "api/testing.md",
        ],
    ],
)

deploydocs(; repo="github.com/wclaerhout/QualityDiversity.jl.git", devbranch="main", push_preview=false)
