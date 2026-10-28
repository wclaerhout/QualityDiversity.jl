# CMA-ME on 20-dimensional Rastrigin with QualityDiversity.jl, on the host and with
# DeviceCMAEmitter on a CUDA GPU, for a growing number of CMA-ES searches.
# Run:  julia --threads=8 benchmark/compare/gpu/compare_cma_gpu.jl
# Choose the GPU as for compare_gpu.jl. The searches of the device emitter sample and
# update on the host threads, so the number of threads matters. The timed runs go to
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
const BUDGET = 1_200_000     # a multiple of every round: 12 candidates per search
const SEARCHES = (5, 50, 500)
const REPEATS = parse(Int, get(ENV, "REPEATS", "5"))
const LIMIT = 5.12f0
const OFFSET = 41.0 * DIM

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

const prob = QDProblem(rastrigin; init=rng -> (2 .* rand(rng, Float32, DIM) .- 1) .* LIMIT,
                       bounds=((-5.12, 5.12), (-5.12, 5.12)))

make_algorithm(searches, device) =
    CMAME(prob; dims=DIMS, sigma0=1.0, n_emitters=searches, genotype=Vector{Float32}, fitness=Float32,
          (device === nothing ? (;) : (; device))...)

println("Julia $VERSION, $(Threads.nthreads()) threads, CUDA ", CUDA.name(CUDA.device()), ", $REPEATS runs per row")
println("| searches | where | median s | evals/s | coverage | QD score |")
println("|---:|---|---:|---:|---:|---:|")
rows = []
const SOFTWARE = software() * "; CUDA.jl $(pkgversion(CUDA))"
const HARDWARE = hardware() * "; " * CUDA.name(CUDA.device())
for searches in SEARCHES, (where, device) in (("host", nothing), ("device", CuArray))
    solve(prob, make_algorithm(searches, device); budget=36 * searches, rng=Xoshiro(0))   # compile
    runs = map(1:REPEATS) do seed
        alg = make_algorithm(searches, device)
        seconds = @elapsed result = solve(prob, alg; budget=BUDGET, rng=Xoshiro(seed))
        (; seed, seconds, coverage=coverage(result.archive),
         qd_score=Float64(qd_score(result.archive; offset=OFFSET)))
    end
    t = median(r.seconds for r in runs)
    println("| ", searches, " | ", where, " | ", round(t; digits=3), " | ", round(Int, BUDGET / t),
            " | ", round(median(r.coverage for r in runs); digits=3), " | ", round(Int, median(r.qd_score for r in runs)), " |")
    for r in runs
        push!(rows, (; benchmark="cma_searches", library="QualityDiversity.jl", method="CMA-ME", variant=where,
                     device=where == "host" ? "cpu" : "gpu", precision="float32", threads=Threads.nthreads(),
                     batch_size=12 * searches, searches, r..., software=SOFTWARE, hardware=HARDWARE))
    end
end
write_runs("cma_searches-qualitydiversity", rows)
