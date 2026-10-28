# CMA-ME and CMA-MAE on 20-dimensional Rastrigin with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/compare/compare_cma.jl
# The matching scripts are compare_cma.py (pyribs) and compare_cma_qdax.py (QDax);
# all print the same columns and write the timed runs to results/ (see results.jl).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
VERSION < v"1.11" && Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using QualityDiversity, QualityDiversity.Benchmarks
using Random: Xoshiro
include("results.jl")

const DIM = 20                  # solution dimension
const DIMS = (100, 100)         # archive cells
const SIGMA0 = 1.0              # initial step size of every search
const EMITTERS = 5
const BATCH = 12                # 4 + floor(3 log(DIM)), the CMA-ES default
const BUDGET = 100_020          # evaluations, a multiple of EMITTERS * BATCH and of BATCH
const REPEATS = parse(Int, get(ENV, "REPEATS", "11"))   # timed runs per method, one seed each
const OFFSET = 41.0 * DIM       # the same QD-score offset as in the other scripts
const THRESHOLD_MIN = -OFFSET   # lowest fitness of Rastrigin, for CMA-MAE
const LEARNING_RATE = 0.01

const prob = Rastrigin(DIM)
const x0 = zeros(DIM)

# Each method is a constructor; all start every search at the origin, as the
# other two libraries do for their first search.
const METHODS = (
    "CMA-ME, improvement emitters" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, n_emitters=EMITTERS, batch_size=BATCH, x0),
    "CMA-ME, optimizing emitters" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, n_emitters=EMITTERS, batch_size=BATCH, x0,
                    ranker=QualityDiversity.ObjectiveRanker(), selection=QualityDiversity.MuSelection(), restart=QualityDiversity.ConvergenceRestart()),
    "CMA-ME, random direction emitters" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, n_emitters=EMITTERS, batch_size=BATCH, x0,
                    ranker=QualityDiversity.TwoStageRandomDirectionRanker()),
    "CMA-MAE" =>
        () -> CMAMAE(prob; dims=DIMS, sigma0=SIGMA0, n_emitters=EMITTERS, batch_size=BATCH, x0,
                     threshold_min=THRESHOLD_MIN, learning_rate=LEARNING_RATE),
    # Rows that only this package has: the random direction ranking that the CMA-ME paper
    # describes, and the covariance update of pyribs (see QualityDiversity.PyribsCMAES).
    "CMA-ME, random direction emitters, ranking of the paper" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, n_emitters=EMITTERS, batch_size=BATCH, x0,
                    ranker=QualityDiversity.AcceptedRandomDirectionRanker()),
    "CMA-MAE, covariance update of pyribs" =>
        () -> CMAMAE(prob; dims=DIMS, sigma0=SIGMA0, n_emitters=EMITTERS, batch_size=BATCH, x0,
                     threshold_min=THRESHOLD_MIN, learning_rate=LEARNING_RATE, strategy=QualityDiversity.PyribsCMAES),
    "CMA-ME, improvement emitters, covariance update of pyribs" =>
        () -> CMAME(prob; dims=DIMS, sigma0=SIGMA0, n_emitters=EMITTERS, batch_size=BATCH, x0,
                    strategy=QualityDiversity.PyribsCMAES),
)

# "CMA-ME, improvement emitters" is method "CMA-ME" and variant "improvement emitters" in
# the CSV file, whose fields hold no commas.
function method_and_variant(name)
    parts = split(name, ", "; limit=2)
    return parts[1], length(parts) == 1 ? "" : replace(parts[2], ", " => "; ")
end

println("Julia $VERSION, $(Threads.nthreads()) thread(s), $REPEATS runs per row, budget $BUDGET")
println("| method | min s | median s | evals/s (median) | coverage | max fitness | QD score |")
println("|---|---:|---:|---:|---:|---:|---:|")
rows = []
const SOFTWARE, HARDWARE = software(), hardware()
for (name, make) in METHODS
    runs = timed_runs(seed -> solve(prob, make(); budget=BUDGET, rng=Xoshiro(seed)),
                      result -> (coverage=coverage(result.archive),
                                 max_fitness=Float64(max_fitness(result.archive)),
                                 qd_score=Float64(qd_score(result.archive; offset=OFFSET))),
                      REPEATS)
    s = summary(runs)
    println("| ", name,
            " | ", round(s.min; digits=3),
            " | ", round(s.median; digits=3),
            " | ", round(Int, BUDGET / s.median),
            " | ", round(s.stat(r -> r.coverage); digits=3),
            " | ", round(s.stat(r -> r.max_fitness); digits=1),
            " | ", round(Int, s.stat(r -> r.qd_score)), " |")
    method, variant = method_and_variant(name)
    for r in runs
        push!(rows, (; benchmark="cma", library="QualityDiversity.jl", method, variant, device="cpu",
                     precision="float64", threads=Threads.nthreads(), batch_size=BATCH, searches=EMITTERS,
                     r..., software=SOFTWARE, hardware=HARDWARE))
    end
end
write_runs("cma-qualitydiversity-cpu-t$(Threads.nthreads())", rows)
