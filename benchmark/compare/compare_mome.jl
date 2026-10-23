# MOME on the two-objective BiSphere benchmark with QualityDiversity.jl.
# Run:  julia --threads=1 benchmark/compare/compare_mome.jl
# The matching script for QDax is compare_mome_qdax.py; both print the same columns and
# write the timed runs to results/ (see results.jl).
# pyribs has no multi-objective archive.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
VERSION < v"1.11" && Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using QualityDiversity, QualityDiversity.Benchmarks
using Random: Xoshiro
include("results.jl")

const DIM = 10                  # solution dimension
const DIMS = (20, 20)           # archive cells
const SIGMA = 0.2               # Gaussian mutation standard deviation
const BATCH = 64
const MAX_FRONT = 50            # solutions kept in the Pareto front of a cell
const BUDGET = 100_032          # evaluations, a multiple of the batch size
const REPEATS = parse(Int, get(ENV, "REPEATS", "11"))   # timed runs, one seed each

const prob = BiSphere(DIM)
const REFERENCE = QualityDiversity.reference_point(prob)

# Fitness and descriptor of one fixed solution, to compare with compare_mome_qdax.py.
reference = QualityDiversity.evaluate(prob, collect(range(-6, 6; length=DIM)))
println("reference fitness ", Tuple(reference.fitness), " descriptor ", Tuple(reference.descriptor),
        " reference point ", Tuple(REFERENCE))

make_algorithm() = MOME(prob; dims=DIMS, max_front=MAX_FRONT, batch_size=BATCH,
                        variation=GaussianMutation(SIGMA))

println("\nJulia $VERSION, $(Threads.nthreads()) thread(s), $REPEATS runs, budget $BUDGET")
println("| min s | median s | evals/s (median) | coverage | MOQD score | global hypervolume | max hypervolume | solutions |")
println("|---:|---:|---:|---:|---:|---:|---:|---:|")
runs = timed_runs(seed -> solve(prob, make_algorithm(); budget=BUDGET, rng=Xoshiro(seed)),
                  result -> (a = result.archive;
                             (coverage=coverage(a), qd_score=qd_score(a), global_hypervolume=global_hypervolume(a),
                              max_hypervolume=QualityDiversity.max_hypervolume(a),
                              solutions=QualityDiversity.num_solutions(a))),
                  REPEATS)
s = summary(runs)
println("| ", round(s.min; digits=3),
        " | ", round(s.median; digits=3),
        " | ", round(Int, BUDGET / s.median),
        " | ", round(s.stat(r -> r.coverage); digits=3),
        " | ", round(Int, s.stat(r -> r.qd_score)),
        " | ", round(Int, s.stat(r -> r.global_hypervolume)),
        " | ", round(Int, s.stat(r -> r.max_hypervolume)),
        " | ", round(Int, s.stat(r -> r.solutions)), " |")
# The qd_score column holds the MOQD score.
const SOFTWARE, HARDWARE = software(), hardware()
write_runs("mome-qualitydiversity-cpu-t$(Threads.nthreads())",
           [(; benchmark="mome", library="QualityDiversity.jl", method="MOME", device="cpu", precision="float64",
             threads=Threads.nthreads(), batch_size=BATCH, r..., software=SOFTWARE, hardware=HARDWARE)
            for r in runs])
