# MAP-Elites on 20-dimensional Rastrigin with QualityDiversity.jl on a CUDA GPU: the
# archive, the variation and the evaluation of every batch run on the device, with a
# DeviceGridArchive and with a DeviceCVTArchive whose centroids are the centers of the
# grid cells, the centroids that compare_qdax.py gives QDax's centroid search.
# Run:  julia benchmark/compare/gpu/compare_gpu.jl
# On a machine with several GPUs, choose one with CUDA_DEVICE_ORDER=PCI_BUS_ID and
# CUDA_VISIBLE_DEVICES=<index in nvidia-smi>; without CUDA_DEVICE_ORDER the CUDA runtime
# numbers the devices fastest first.
# The problem, grid, mutation, batch sizes, budget and columns are those of
# ../compare.jl; compare_qdax.py runs QDax on the same device. The timed runs go to
# ../results/ (see ../results.jl).
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CUDA
using QualityDiversity
using Random: Xoshiro
include(joinpath(@__DIR__, "..", "results.jl"))

const DIM = 20
const DIMS = (100, 100)
const SIGMA = 0.5
const BUDGET = 360_000
const BATCH_SIZES = (36, 100, 1000, 10_000)
const REPEATS = parse(Int, get(ENV, "REPEATS", "21"))
const LIMIT = 5.12
const OFFSET = 41.0 * DIM

# The Rastrigin benchmark written for any element type, so that the kernel computes in
# the precision of the archive. A device kernel runs it on one solution per thread.
function rastrigin(x::AbstractVector{T}) where {T}
    limit = T(LIMIT)
    half = length(x) ÷ 2
    total = zero(T)
    first_half = zero(T)
    second_half = zero(T)
    for i in eachindex(x)
        xi = clamp(x[i], -limit, limit)
        total += xi^2 - 10 * cospi(2 * xi)
        i - firstindex(x) < half ? (first_half += xi) : (second_half += xi)
    end
    return -(10 * T(length(x)) + total),
           (clamp(first_half / half, -limit, limit), clamp(second_half / (length(x) - half), -limit, limit))
end

problem(::Type{T}) where {T} = QDProblem(rastrigin; init=rng -> (2 .* rand(rng, T, DIM) .- 1) .* T(LIMIT),
                                         bounds=((-LIMIT, LIMIT), (-LIMIT, LIMIT)))

# The centers of the cells of the grid, in the order of QDax's compute_euclidean_centroids.
const CENTERS = vec([[-LIMIT + (i - 0.5) * 2LIMIT / DIMS[1], -LIMIT + (j - 0.5) * 2LIMIT / DIMS[2]]
                     for i in 1:DIMS[1], j in 1:DIMS[2]])

make_algorithm(::Val{:grid}, T, batch_size) =
    MAPElites(problem(T); dims=DIMS, batch_size, variation=GaussianMutation(T(SIGMA)), device=CuArray,
              genotype=Vector{T}, fitness=T)
make_algorithm(::Val{:cvt}, T, batch_size) =
    MAPElites(DeviceCVTArchive{T,T}(CuArray, CENTERS, ((-LIMIT, LIMIT), (-LIMIT, LIMIT)), DIM);
              batch_size, variation=GaussianMutation(T(SIGMA)))

# `solve` returns once the last batch is on the device; the metrics recorded after every
# step wait for it, so the time covers the whole run.
timed_batch(archive, T, batch_size) =
    timed_runs(seed -> solve(problem(T), make_algorithm(archive, T, batch_size); budget=BUDGET, rng=Xoshiro(seed)),
               result -> (coverage=coverage(result.archive),
                          max_fitness=Float64(max_fitness(result.archive)),
                          qd_score=Float64(qd_score(result.archive; offset=OFFSET))),
               REPEATS)

rows = []
const SOFTWARE = software() * "; CUDA.jl $(pkgversion(CUDA))"
const HARDWARE = hardware() * "; " * CUDA.name(CUDA.device())
for (name, variant, archive) in (("DeviceGridArchive", "grid", Val(:grid)), ("DeviceCVTArchive", "cvt", Val(:cvt))),
    T in (Float32, Float64)

    println("\nJulia $VERSION, CUDA ", CUDA.name(CUDA.device()), ", $name, $T, $REPEATS runs per row")
    println("| batch size | min s | median s | middle half s | evals/s (median) | coverage | max fitness | QD score |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for batch_size in BATCH_SIZES
        runs = timed_batch(archive, T, batch_size)
        s = summary(runs)
        println("| ", batch_size,
                " | ", round(s.min; digits=4),
                " | ", round(s.median; digits=4),
                " | ", round(s.lo; digits=4), "–", round(s.hi; digits=4),
                " | ", round(Int, BUDGET / s.median),
                " | ", round(s.stat(r -> r.coverage); digits=3),
                " | ", round(s.stat(r -> r.max_fitness); digits=1),
                " | ", round(Int, s.stat(r -> r.qd_score)), " |")
        for r in runs
            push!(rows, (; benchmark="map_elites", library="QualityDiversity.jl", method="MAP-Elites", variant,
                         device="gpu", precision=lowercase(string(T)), batch_size, r...,
                         software=SOFTWARE, hardware=HARDWARE))
        end
    end
end
write_runs("map_elites-qualitydiversity-gpu", rows)
