# MAP-Elites on 20-dimensional Rastrigin with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/compare/compare.jl
# The matching pyribs script is compare.py; both print the same columns and write the
# timed runs to results/ (see results.jl).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
VERSION < v"1.11" && Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using QualityDiversity, QualityDiversity.Benchmarks
using Random: Xoshiro
include("results.jl")

const DIM = 20                  # solution dimension
const DIMS = (100, 100)         # archive cells
const SIGMA = 0.5               # Gaussian mutation standard deviation
const BUDGET = 360_000          # evaluations, a multiple of every batch size
const BATCH_SIZES = (36, 100, 1000, 10_000)
const REPEATS = parse(Int, get(ENV, "REPEATS", "21"))   # timed runs per batch size, one seed each

const OFFSET = 41.0 * DIM       # the same QD-score offset as Rastrigin in Python

# The Float64 problem is the Rastrigin benchmark of the package.
const prob = Rastrigin(DIM)

# The Float32 problem is the same function written with Float32 arithmetic, the
# way a user would define it with QDProblem. The archive then holds
# Vector{Float32} genotypes and Float32 fitness values, and the mutation
# produces Float32 children.
function rastrigin32(x::Vector{Float32})
    limit = 5.12f0
    half = DIM ÷ 2
    total = 0f0
    first_half = 0f0
    second_half = 0f0
    for i in 1:DIM
        xi = clamp(x[i], -limit, limit)
        total += xi^2 - 10f0 * cos(2f0 * Float32(π) * xi)
        if i <= half
            first_half += xi
        else
            second_half += xi
        end
    end
    return -(10f0 * DIM + total),
           (clamp(first_half / half, -limit, limit), clamp(second_half / (DIM - half), -limit, limit))
end

const prob32 = QDProblem(rastrigin32;
                         init=rng -> (2f0 .* rand(rng, Float32, DIM) .- 1f0) .* 5.12f0,
                         bounds=((-5.12, 5.12), (-5.12, 5.12)))

make_algorithm(::Type{Float64}, batch_size) =
    MAPElites(prob; dims=DIMS, batch_size, variation=GaussianMutation(SIGMA))
make_algorithm(::Type{Float32}, batch_size) =
    MAPElites(prob32; dims=DIMS, batch_size, variation=GaussianMutation(Float32(SIGMA)),
              genotype=Vector{Float32}, fitness=Float32)
problem(::Type{Float64}) = prob
problem(::Type{Float32}) = prob32

# Evaluating a fixed solution lets compare.py confirm that both sides score
# the same problem.
reference = QualityDiversity.evaluate(prob, collect(range(-6, 6; length=DIM)))
println("reference fitness ", reference.fitness, " descriptor ", Tuple(reference.descriptor))

function timed_batch(T, batch_size)
    p = problem(T)
    return timed_runs(seed -> solve(p, make_algorithm(T, batch_size); budget=BUDGET, rng=Xoshiro(seed)),
                      result -> (coverage=coverage(result.archive),
                                 max_fitness=Float64(max_fitness(result.archive)),
                                 qd_score=Float64(qd_score(result.archive; offset=OFFSET))),
                      REPEATS)
end

rows = []
const SOFTWARE, HARDWARE = software(), hardware()
for T in (Float64, Float32)
    println("\nJulia $VERSION, $(Threads.nthreads()) thread(s), $T genotypes, $REPEATS runs per row")
    println("| batch size | min s | median s | middle half s | evals/s (median) | coverage | max fitness | QD score |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for batch_size in BATCH_SIZES
        runs = timed_batch(T, batch_size)
        s = summary(runs)
        println("| ", batch_size,
                " | ", round(s.min; digits=3),
                " | ", round(s.median; digits=3),
                " | ", round(s.lo; digits=3), "–", round(s.hi; digits=3),
                " | ", round(Int, BUDGET / s.median),
                " | ", round(s.stat(r -> r.coverage); digits=3),
                " | ", round(s.stat(r -> r.max_fitness); digits=1),
                " | ", round(Int, s.stat(r -> r.qd_score)), " |")
        for r in runs
            push!(rows, (; benchmark="map_elites", library="QualityDiversity.jl", method="MAP-Elites",
                         variant="grid", device="cpu", precision=lowercase(string(T)),
                         threads=Threads.nthreads(), batch_size, r..., software=SOFTWARE, hardware=HARDWARE))
        end
    end
end
write_runs("map_elites-qualitydiversity-cpu-t$(Threads.nthreads())", rows)
