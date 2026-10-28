# BenchmarkTools suite for the hot path. Run it with `julia benchmark/run.jl`.
# Leave cores free for the machine while it runs; the threaded evaluate_batch
# uses the threads Julia was started with.
using BenchmarkTools
using QualityDiversity, QualityDiversity.Benchmarks
using Random: Xoshiro
using StaticArrays: SVector

const QD = QualityDiversity
const SUITE = BenchmarkGroup()

const DIM = 20
const prob = Rastrigin(DIM)
const rng = Xoshiro(1)
const x = QD.random_solution(rng, prob)
const ev = QD.evaluate(prob, x)
const BATCH = 100

new_archive() = GridArchive(prob; dims=(100, 100))

function filled_archive(n)
    archive = new_archive()
    xs = [QD.random_solution(rng, prob) for _ in 1:n]
    QD.add_batch!(archive, xs, QD.evaluate_batch(prob, xs))
    return archive
end

# ── problem ───────────────────────────────────────────────
SUITE["problem"]["evaluate"] = @benchmarkable QD.evaluate($prob, $x)
const noisy = StochasticQDProblem((rng, x) -> (-sum(abs2, x) + randn(rng), (x[1], x[2]));
                                  init=rng -> randn(rng, 20), bounds=((-3, 3), (-3, 3)))
const sampled = Sampled(noisy; n_samples=10)
SUITE["problem"]["StochasticQDProblem/evaluate"] = @benchmarkable QD.evaluate($rng, $noisy, $x)
SUITE["problem"]["Sampled/evaluate"] = @benchmarkable QD.evaluate($rng, $sampled, $x)
const spread_problem = SampledSpread(noisy; n_samples=10)
SUITE["problem"]["SampledSpread/evaluate"] = @benchmarkable QD.evaluate($rng, $spread_problem, $x)
for n in (36, 100, 1000)
    xs = [QD.random_solution(rng, prob) for _ in 1:n]
    SUITE["problem"]["evaluate_batch"]["batch=$n"] = @benchmarkable QD.evaluate_batch($prob, $xs)
    SUITE["problem"]["StochasticQDProblem/evaluate_batch"]["batch=$n"] = @benchmarkable QD.evaluate_batch($rng, $noisy, $xs)
end

# ── variation ─────────────────────────────────────────────
const mutation = GaussianMutation(0.5)
SUITE["variation"]["GaussianMutation"] = @benchmarkable QD.vary($rng, $mutation, $x)
const isoline = IsoLineVariation(0.01, 0.2)
const x2 = QD.random_solution(rng, prob)
SUITE["variation"]["IsoLineVariation"] = @benchmarkable QD.vary($rng, $isoline, $x, $x2)
const polynomial = PolynomialMutation(-5.12, 5.12; eta=20)
SUITE["variation"]["PolynomialMutation"] = @benchmarkable QD.vary($rng, $polynomial, $x)
const uniform = UniformCrossover()
SUITE["variation"]["UniformCrossover"] = @benchmarkable QD.vary($rng, $uniform, $x, $x2)

# ── archive ───────────────────────────────────────────────
const archive = filled_archive(20_000)
const d = ev.descriptor
SUITE["archive"]["cell_index"] = @benchmarkable QD.cell_index($archive, $d)
SUITE["archive"]["add!/rejected"] = @benchmarkable QD.add!($archive, $x, $ev)
SUITE["archive"]["add!/new cell"] = @benchmarkable QD.add!(a, $x, $ev) setup = (a = new_archive()) evals = 1
SUITE["archive"]["add!/improved"] =
    @benchmarkable QD.add!(a, $x, $(QD.Evaluation(ev.fitness + 1, ev.descriptor))) setup =
        (a = new_archive(); QD.add!(a, $x, $ev)) evals = 1
SUITE["archive"]["sample_elites/n=100"] = @benchmarkable QD.sample_elites($rng, $archive, 100)
SUITE["archive"]["add_batch!/n=100"] = let xs = [QD.random_solution(rng, prob) for _ in 1:BATCH]
    evs = QD.evaluate_batch(prob, xs)
    @benchmarkable QD.add_batch!($archive, $xs, $evs)
end
SUITE["archive"]["metrics"]["coverage"] = @benchmarkable coverage($archive)
SUITE["archive"]["metrics"]["qd_score"] = @benchmarkable qd_score($archive; offset=800.0)
SUITE["archive"]["metrics"]["max_fitness"] = @benchmarkable max_fitness($archive)
SUITE["archive"]["metrics"]["mean_fitness"] = @benchmarkable mean_fitness($archive)

const cvt = let a = CVTArchive(prob; cells=1000, samples=20_000, rng=Xoshiro(3))
    xs = [QD.random_solution(rng, prob) for _ in 1:20_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    a
end
SUITE["archive"]["CVTArchive/cell_index (1000 cells)"] = @benchmarkable QD.cell_index($cvt, $d)
SUITE["archive"]["CVTArchive/add!/rejected"] = @benchmarkable QD.add!($cvt, $x, $ev)
SUITE["archive"]["CVTArchive/sample_elites/n=100"] = @benchmarkable QD.sample_elites($rng, $cvt, 100)

function filled_sliding(remap_frequency)
    a = SlidingBoundariesArchive(prob; dims=(100, 100), remap_frequency)
    for _ in 1:20
        xs = [QD.random_solution(rng, prob) for _ in 1:BATCH]
        QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    end
    return a
end
const dns = let a = DNSArchive(prob; capacity=1024, k=3)
    xs = [QD.random_solution(rng, prob) for _ in 1:2_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    a
end
SUITE["archive"]["DNSArchive/add! (1024 solutions)"] = @benchmarkable QD.add!($dns, $x, $ev)
SUITE["archive"]["DNSArchive/add_batch!/n=100"] = let xs = [QD.random_solution(rng, prob) for _ in 1:BATCH]
    evs = QD.evaluate_batch(prob, xs)
    @benchmarkable QD.add_batch!($dns, $xs, $evs)
end
const density_archive = let a = DensityArchive(GridArchive(prob; dims=(100, 100), outside=ClampOutside()); bandwidth=0.5, rng=Xoshiro(4))
    xs = [QD.random_solution(rng, prob) for _ in 1:20_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    a
end
SUITE["archive"]["DensityArchive/density (10_000 buffered)"] = @benchmarkable QD.density($density_archive, $d)
SUITE["archive"]["DensityArchive/add!"] = @benchmarkable QD.add!($density_archive, $x, $ev)
const sliding = filled_sliding(typemax(Int))
SUITE["archive"]["SlidingBoundariesArchive/cell_index"] = @benchmarkable QD.cell_index($sliding, $d)
SUITE["archive"]["SlidingBoundariesArchive/add!/rejected"] = @benchmarkable QD.add!($sliding, $x, $(QD.Evaluation(-1e6, ev.descriptor)))
# with a full buffer of 1000 solutions, every call remaps
const remapping = filled_sliding(1)
SUITE["archive"]["SlidingBoundariesArchive/add! with remap"] = @benchmarkable QD.add!($remapping, $x, $ev)

# ── emitter ───────────────────────────────────────────────
const emitter = VariationEmitter(mutation; batch_size=BATCH)
SUITE["emitter"]["ask/batch=100"] = @benchmarkable QD.ask($rng, $emitter, $archive)

# ── algorithm ─────────────────────────────────────────────
const alg = MAPElites(prob; dims=(100, 100), batch_size=BATCH, variation=mutation)
const state = init(prob, alg; budget=typemax(Int), rng=Xoshiro(2))
for _ in 1:200
    step!(state)    # reach a steady state before measuring
end
SUITE["algorithm"]["step!/batch=100"] = @benchmarkable step!($state)
const sliding_state = init(prob, MAPElites(SlidingBoundariesArchive(prob; dims=(100, 100)); batch_size=BATCH,
                                          variation=mutation); budget=typemax(Int), rng=Xoshiro(2))
for _ in 1:200
    step!(sliding_state)
end
SUITE["algorithm"]["SlidingBoundariesArchive/step!/batch=100"] = @benchmarkable step!($sliding_state)
SUITE["algorithm"]["solve/360k evaluations"] =
    @benchmarkable solve($prob, $alg; budget=360_000, rng=Xoshiro(3)) samples = 5 evals = 1 seconds = 120

const unstructured = let a = UnstructuredArchive(prob; capacity=1000, novelty_threshold=0.2)
    xs = [QD.random_solution(rng, prob) for _ in 1:20_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    a
end
SUITE["archive"]["UnstructuredArchive/cell_index (1000 slots)"] = @benchmarkable QD.cell_index($unstructured, $d)
SUITE["archive"]["UnstructuredArchive/add!/rejected"] = @benchmarkable QD.add!($unstructured, $x, $ev)
SUITE["archive"]["UnstructuredArchive/sample_elites/n=100"] = @benchmarkable QD.sample_elites($rng, $unstructured, 100)

const threshold = let a = ThresholdArchive(new_archive(); learning_rate=0.1, threshold_min=-800.0)
    xs = [QD.random_solution(rng, prob) for _ in 1:20_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(prob, xs))
    a
end
SUITE["archive"]["ThresholdArchive/cell_index"] = @benchmarkable QD.cell_index($threshold, $d)
SUITE["archive"]["ThresholdArchive/add!/rejected"] = @benchmarkable QD.add!($threshold, $x, $ev)
SUITE["archive"]["ThresholdArchive/add!/improved"] =
    @benchmarkable QD.add!($threshold, $x, $(QD.Evaluation(ev.fitness + 1000, ev.descriptor)))
SUITE["archive"]["ThresholdArchive/metrics/max_fitness"] = @benchmarkable max_fitness($threshold)
SUITE["archive"]["ThresholdArchive/add_batch!/n=100"] = let xs = [QD.random_solution(rng, prob) for _ in 1:BATCH]
    evs = QD.evaluate_batch(prob, xs)
    @benchmarkable QD.add_batch!($threshold, $xs, $evs)
end

const spread_archive = let a = SpreadArchive(GridArchive(spread_problem; dims=(100, 100), outside=ClampOutside()))
    xs = [QD.random_solution(rng, spread_problem) for _ in 1:20_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(rng, spread_problem, xs))
    a
end
const spread_ev = QD.evaluate(rng, spread_problem, x)
SUITE["archive"]["SpreadArchive/cell_index (10 samples)"] = @benchmarkable QD.cell_index($spread_archive, $(spread_ev.descriptor))
SUITE["archive"]["SpreadArchive/add!/rejected"] =
    @benchmarkable QD.add!($spread_archive, $x, $(QD.Evaluation(-1e9, spread_ev.descriptor)))
SUITE["archive"]["SpreadArchive/add_batch!/n=100"] = let xs = [QD.random_solution(rng, spread_problem) for _ in 1:BATCH]
    evs = QD.evaluate_batch(rng, spread_problem, xs)
    @benchmarkable QD.add_batch!($spread_archive, $xs, $evs)
end
SUITE["algorithm"]["MELS/step!"] = let alg = MELS(spread_problem; dims=(100, 100), batch_size=BATCH, outside=ClampOutside())
    state = init(spread_problem, alg; budget=10^9, rng=Xoshiro(1))
    @benchmarkable step!($state)
end

# ── CMA-ES emitters ───────────────────────────────────────
const sphere = Sphere(20)
const cma_archive = let a = GridArchive(sphere; dims=(100, 100))
    xs = [QD.random_solution(rng, sphere) for _ in 1:2_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(sphere, xs))
    a
end
const cma_emitter = CMAEmitter(20; sigma0=0.5)
SUITE["emitter"]["Sphere/evaluate"] = @benchmarkable QD.evaluate($sphere, $(QD.random_solution(rng, sphere)))
SUITE["emitter"]["CMAEmitter/ask"] = @benchmarkable QD.ask($rng, $cma_emitter, $cma_archive)
SUITE["emitter"]["CMAEmitter/tell!"] = let xs = QD.ask(rng, CMAEmitter(20; sigma0=0.5), cma_archive)
    evs = QD.evaluate_batch(sphere, xs)
    results = QD.add_batch!(deepcopy(cma_archive), xs, evs)
    # tell! needs a fresh ask before each call; only tell! is timed
    @benchmarkable QD.tell!(e, $xs, $evs, $results) setup = (e = CMAEmitter(20; sigma0=0.5); QD.ask($rng, e, $cma_archive)) evals = 1
end
for (name, kwargs) in (("SepCMAES", (; strategy=QD.SepCMAES)), ("LMMAES", (; strategy=QD.LMMAES, batch_size=10)),
                       ("OpenAIES", (; strategy=QD.OpenAIES)),
                       ("RandomDirection", (; ranker=QD.TwoStageRandomDirectionRanker())))
    SUITE["emitter"]["CMAEmitter/$name/ask"] = let e = CMAEmitter(20; sigma0=0.5, kwargs...)
        @benchmarkable QD.ask($rng, $e, $cma_archive)
    end
    SUITE["emitter"]["CMAEmitter/$name/tell!"] = let xs = QD.ask(rng, CMAEmitter(20; sigma0=0.5, kwargs...), cma_archive)
        evs = QD.evaluate_batch(sphere, xs)
        results = QD.add_batch!(deepcopy(cma_archive), xs, evs)
        @benchmarkable QD.tell!(e, $xs, $evs, $results) setup = (e = CMAEmitter(20; sigma0=0.5, $kwargs...); QD.ask($rng, e, $cma_archive)) evals = 1
    end
end
SUITE["algorithm"]["CMAME/step!"] = let alg = CMAME(sphere; dims=(100, 100), sigma0=0.5)
    state = init(sphere, alg; budget=10^9, rng=Xoshiro(1))
    @benchmarkable step!($state)
end
SUITE["algorithm"]["BanditScheduler/step!"] = let
    # the pool of Cully (2021) in miniature: mutation, iso-line and CMA emitters, 4 of 12 active
    pool = ((VariationEmitter(GaussianMutation(0.5); batch_size=36) for _ in 1:4)...,
            (VariationEmitter(IsoLineVariation(0.01, 0.1); batch_size=36) for _ in 1:4)...,
            (CMAEmitter(20; sigma0=0.5, batch_size=36) for _ in 1:4)...)
    alg = QDAlgorithm(GridArchive(sphere; dims=(100, 100)), (BanditScheduler(pool; n_active=4),))
    state = init(sphere, alg; budget=10^9, rng=Xoshiro(1))
    for _ in 1:50
        step!(state)
    end
    @benchmarkable step!($state)
end
SUITE["algorithm"]["DDS/step!"] = let alg = DDS(sphere; dims=(100, 100), sigma0=0.5, bandwidth=0.5, rng=Xoshiro(1))
    state = init(sphere, alg; budget=10^9, rng=Xoshiro(1))
    for _ in 1:30
        step!(state)    # past the first 10_000 descriptors, so that the buffer is full
    end
    @benchmarkable step!($state)
end
SUITE["algorithm"]["CMAMAE/step!"] = let alg = CMAMAE(sphere; dims=(100, 100), sigma0=0.5, threshold_min=0.0, learning_rate=0.01)
    state = init(sphere, alg; budget=10^9, rng=Xoshiro(1))
    @benchmarkable step!($state)
end


# ── multi-objective archive ───────────────────────────────
const bisphere = BiSphere(20)
const pareto_archive = let a = ParetoGridArchive(bisphere; dims=(100, 100), reference_point=QD.reference_point(bisphere))
    xs = [QD.random_solution(rng, bisphere) for _ in 1:5_000]
    QD.add_batch!(a, xs, QD.evaluate_batch(bisphere, xs))
    a
end
const pareto_x = QD.random_solution(rng, bisphere)
const pareto_ev = QD.evaluate(bisphere, pareto_x)
SUITE["problem"]["BiSphere/evaluate"] = @benchmarkable QD.evaluate($bisphere, $pareto_x)
SUITE["archive"]["ParetoGridArchive/cell_index"] = @benchmarkable QD.cell_index($pareto_archive, $(pareto_ev.descriptor))
SUITE["archive"]["ParetoGridArchive/add!/rejected"] = @benchmarkable QD.add!($pareto_archive, $pareto_x, $(QD.Evaluation(pareto_ev.fitness .- 1e6, pareto_ev.descriptor)))
SUITE["archive"]["ParetoGridArchive/add!/new solution"] = @benchmarkable QD.add!(a, $pareto_x, $pareto_ev) setup = (a = deepcopy($pareto_archive)) evals = 1
SUITE["archive"]["ParetoGridArchive/sample_elites/n=100"] = @benchmarkable QD.sample_elites($rng, $pareto_archive, 100)
SUITE["archive"]["ParetoGridArchive/qd_score"] = @benchmarkable qd_score($pareto_archive)
SUITE["archive"]["ParetoGridArchive/global_hypervolume"] = @benchmarkable global_hypervolume($pareto_archive)
SUITE["algorithm"]["MOME/step!"] = let alg = MOME(bisphere; dims=(100, 100))
    state = init(bisphere, alg; budget=10^9, rng=Xoshiro(1))
    @benchmarkable step!($state)
end


# ── differentiable quality diversity ──────────────────────
SUITE["problem"]["Sphere/evaluate_gradient"] = @benchmarkable QD.evaluate_gradient($sphere, $(QD.random_solution(rng, sphere)))
const gradient_archive = let a = GradientArchive(GridArchive(sphere; dims=(100, 100)))
    xs = [QD.random_solution(rng, sphere) for _ in 1:5_000]
    QD.add_batch!(a, xs, QD.evaluate_gradient_batch(sphere, xs)...)
    a
end
SUITE["emitter"]["GradientOperatorEmitter/ask/batch=100"] =
    @benchmarkable QD.ask($rng, $(GradientOperatorEmitter(; sigma0=10.0, batch_size=BATCH)), $gradient_archive)
SUITE["archive"]["GradientArchive/add_batch!/n=100"] = let xs = [QD.random_solution(rng, sphere) for _ in 1:BATCH]
    evs, jacobians = QD.evaluate_gradient_batch(sphere, xs)
    @benchmarkable QD.add_batch!(a, $xs, $evs, $jacobians) setup = (a = deepcopy($gradient_archive)) evals = 1
end
for (name, alg) in (("CMAMEGA", CMAMEGA(sphere; dims=(100, 100), sigma0=10.0)),
                    ("CMAMAEGA", CMAMAEGA(sphere; dims=(100, 100), sigma0=10.0, threshold_min=0.0)),
                    ("OMGMEGA", OMGMEGA(sphere; dims=(100, 100), sigma0=10.0, batch_size=BATCH)))
    SUITE["algorithm"]["$name/step!"] = let state = init(sphere, alg; budget=10^9, rng=Xoshiro(1))
        @benchmarkable step!($state)
    end
end
