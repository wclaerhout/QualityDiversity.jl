# The device archive and emitter. `Array` runs the kernels on the CPU backend of
# KernelAbstractions, and JLArrays on a GPU backend that executes on the CPU and refuses
# scalar indexing, so both run without a GPU.
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using QualityDiversity.Testing

const KAExt = Base.get_extension(QualityDiversity, :QualityDiversityKernelAbstractionsExt)

isdefined(@__MODULE__, :GPUExample) || include(joinpath(@__DIR__, "..", "examples", "gpu_evaluation.jl"))

# The elites of an archive by cell, to compare a device archive with a host one.
elites_by_cell(host, archive) = Dict(QD.cell_index(host, ev.descriptor) => (x, ev) for (x, ev) in QD.elites(archive))

@testset "Philox4x32-10" begin
    # known-answer vectors of Random123 (kat_vectors, philox4x32 with 10 rounds)
    @test KAExt.philox(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000) ==
          (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
    @test KAExt.philox(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff) ==
          (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
    @test KAExt.philox(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0) ==
          (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)

    key = KAExt.key_of(0x0123456789abcdef)
    for T in (Float32, Float64)
        z = [KAExt.normal(T, KAExt.random_words(key, KAExt.STREAM_ENTRY, j, i)) for j in 1:200, i in 1:100]
        @test eltype(z) == T
        @test abs(sum(z) / length(z)) < 0.02
        @test abs(sum(abs2, z) / length(z) - 1) < 0.03
        @test all(isfinite, z)
    end
    @test 0 < KAExt.uniform32(0x00000000) < KAExt.uniform32(0xffffffff) < 1
    @test 0 < KAExt.uniform64(0x00000000, 0x00000000) < KAExt.uniform64(0xffffffff, 0xffffffff) < 1

    # parent slots are uniform over 1:count
    for count in Int32[1, 7, 1000]
        slots = [KAExt.uniform_slot(KAExt.random_words(key, KAExt.STREAM_PARENT, j, 0)[1:2]..., count)
                 for j in 1:20_000]
        @test extrema(slots) == (1, count)
        if count == 7
            tally = [Base.count(==(s), slots) for s in 1:7]
            @test all(t -> abs(t - 20_000 / 7) < 200, tally)
        end
    end

    # fitness keys order as the values do
    values32 = Float32[-Inf, -1f30, -2, -1, -0.0, 0, 1f-30, 1, 2, 1f30, Inf]
    @test issorted(KAExt.fitness_key.(values32))
    @test issorted(KAExt.fitness_key.(Float64.(values32)))
    @test issorted(KAExt.fitness_key.([typemin(Int), -3, 0, 5, typemax(Int)]))
end

@testset "DeviceGridArchive construction" begin
    prob = Rastrigin(4)
    a = DeviceGridArchive(prob; dims=(8, 5), device=Array)
    @test a isa QD.AbstractArchive{Vector{Float64},Float64,SVector{2,Float64},Nothing}
    @test length(a) == 0
    @test QD.capacity(a) == 40
    @test QD.solution_length(a) == 4
    @test QD.descriptor_bounds(a) == QD.descriptor_bounds(prob)
    @test DeviceGridArchive(prob; dims=(8, 5), device=JLArray, genotype=Vector{Float32},
                            fitness=Float32).genotypes isa JLArray{Float32}
    @test_throws "dims has 2 entries but bounds has 1" DeviceGridArchive{Float64,Float64}(Array, (2, 2), ((0, 1),), 3)
    @test_throws "dims[2] is 0 but every dimension needs at least one cell" DeviceGridArchive{Float64,Float64}(Array, (2, 0), ((0, 1), (0, 1)), 3)
    @test_throws "bounds[1] is (1.0, 0.0) but lo must be finite and less than hi" DeviceGridArchive{Float64,Float64}(Array, (2, 2), ((1, 0), (0, 1)), 3)
    @test_throws "solutions must have at least one entry, got dim = 0" DeviceGridArchive{Float64,Float64}(Array, (2, 2), ((0, 1), (0, 1)), 0)
    @test_throws "genotype must be a type such as Vector{Float32}, got Vector{Int64}" DeviceGridArchive(prob; dims=(2, 2), device=Array, genotype=Vector{Int})
end

@testset "add_batch! agrees with GridArchive" begin
    rng = Xoshiro(1)
    for outside in (ThrowOutside(), ClampOutside(), RejectOutside()), F in (Float64, Float32, Float16, Int, UInt32)
        host = GridArchive{Vector{Float64},F}((4, 3), ((0, 1), (0, 1)); outside)
        dev = DeviceGridArchive{Float64,F}(Array, (4, 3), ((0, 1), (0, 1)), 3; outside)
        reach = outside isa ThrowOutside ? (0.0, 1.0) : (-0.3, 1.3)
        for round in 1:6
            n = rand(rng, 1:60)
            xs = [randn(rng, 3) for _ in 1:n]
            # few distinct fitness values, so that solutions tie for a cell
            evs = [Evaluation(F(rand(rng, F <: Unsigned ? (0:6) : (-3:3))), SVector(reach[1] + (reach[2] - reach[1]) * rand(rng),
                                                          reach[1] + (reach[2] - reach[1]) * rand(rng)))
                   for _ in 1:n]
            @test QD.add_batch!(dev, xs, evs) == QD.add_batch!(host, xs, evs)
            @test length(dev) == length(host)
            @test elites_by_cell(host, dev) == elites_by_cell(host, host)
        end
        @test qd_score(dev; offset=4) ≈ qd_score(host; offset=4)
        @test max_fitness(dev) == max_fitness(host)
        @test mean_fitness(dev) ≈ mean_fitness(host)
        converted = GridArchive(dev)
        @test converted isa GridArchive{Vector{Float64},F}
        @test elites_by_cell(host, converted) == elites_by_cell(host, host)
        @test qd_score(converted) ≈ qd_score(host)
    end

    # the batch on the device, with an EvaluationBatch or with host evaluations
    host = GridArchive{Vector{Float32},Float32}((6, 6), ((0, 1), (0, 1)))
    dev = DeviceGridArchive{Float32,Float32}(Array, (6, 6), ((0, 1), (0, 1)), 2)
    X = rand(rng, Float32, 2, 50)
    evs = [Evaluation(Float32(-sum(abs2, x)), SVector(Float64(x[1]), Float64(x[2]))) for x in eachcol(X)]
    batch = QD.EvaluationBatch{2}(Float32[ev.fitness for ev in evs], X)
    results = QD.add_batch!(dev, QD.DeviceBatch(copy(X)), batch)
    @test results isa QD.AddResultBatch{Float32}
    @test collect(results) == QD.add_batch!(host, collect.(eachcol(X)), evs)
    @test results[3] == collect(results)[3]
    @test elites_by_cell(host, dev) == elites_by_cell(host, host)
    @test QD.add_batch!(deepcopy(dev), QD.DeviceBatch(copy(X)), evs) == QD.add_batch!(deepcopy(dev), QD.DeviceBatch(copy(X)), batch)
    @test isempty(QD.add_batch!(dev, QD.DeviceBatch(zeros(Float32, 2, 0)), QD.EvaluationBatch{2}(Float32[], zeros(Float32, 2, 0))))
end

@testset "Float64 descriptors with Float32 genotypes" begin
    # just below a cell boundary at 0.5, which rounding to Float32 would reach (#47)
    edge = 0.499999999
    @test Float32(edge) == 0.5f0
    prob = QDProblem(x -> (-sum(abs2, x), (edge, 0.5)); init=rng -> rand(rng, Float32, 3), bounds=((0, 1), (0, 1)))
    X = JLArray(rand(Xoshiro(3), Float32, 3, 4))
    evs = QD.evaluate_batch(prob, QD.DeviceBatch(X))
    @test evs.descriptors isa JLArray{Float64}
    @test all(==(edge), Array(evs.descriptors)[1, :])
    xs = collect.(eachcol(Array(X)))
    host_evs = collect(evs)
    # the second centroid is the nearer, the first wins the tie that rounding would make
    centroids = [[0.75, 0.5], [0.25, 0.5]]
    for (host, make) in ((GridArchive{Vector{Float32},Float32}((10, 4), ((0, 1), (0, 1))),
                          () -> DeviceGridArchive{Float32,Float32}(JLArray, (10, 4), ((0, 1), (0, 1)), 3)),
                         (CVTArchive{Vector{Float32},Float32}(centroids, ((0, 1), (0, 1))),
                          () -> DeviceCVTArchive{Float32,Float32}(JLArray, centroids, ((0, 1), (0, 1)), 3)))
        expected = QD.add_batch!(host, xs, host_evs)
        for given in (evs, host_evs)
            dev = make()
            @test dev.descriptors isa JLArray{Float64}
            @test collect(QD.add_batch!(dev, QD.DeviceBatch(copy(X)), given)) == expected
            @test elites_by_cell(host, dev) == elites_by_cell(host, host)
            @test only(QD.elites(dev))[2].descriptor == SVector(edge, 0.5)
        end
    end
end

@testset "add_batch! errors" begin
    dev = DeviceGridArchive{Float64,Float64}(Array, (4, 4), ((0, 1), (0, 1)), 2)
    QD.add!(dev, [0.1, 0.1], Evaluation(1.0, SVector(0.1, 0.1)))
    before = deepcopy(dev)
    unchanged() = elites_by_cell(GridArchive(before), dev) == elites_by_cell(GridArchive(before), before) &&
                  Array(dev.best_key) == Array(before.best_key) && Array(dev.winner) == Array(before.winner) &&
                  Array(dev.flags) == Array(before.flags)

    xs = [[0.5, 0.5], [0.2, 0.3], [0.7, 0.8]]
    good = [Evaluation(2.0, SVector(0.5, 0.5)), Evaluation(3.0, SVector(0.2, 0.3)), Evaluation(4.0, SVector(0.7, 0.8))]
    outside = [good[1], Evaluation(3.0, SVector(0.2, 1.5)), good[3]]
    @test_throws "descriptor component 2 is 1.5, outside the archive bounds [0.0, 1.0]" QD.add_batch!(dev, xs, outside)
    @test unchanged()
    for bad in (NaN, Inf, -Inf)
        @test_throws "the fitness of solution 3 of the batch is not finite" QD.add_batch!(dev, xs, [good[1], good[2], Evaluation(bad, SVector(0.7, 0.8))])
        @test unchanged()
    end
    # -0.0 and 0.0 tie, so the first stays, as on the host
    for (first_fitness, second_fitness) in ((-0.0, 0.0), (0.0, -0.0), (-0.0f0, 0.0f0))
        T = typeof(first_fitness)
        tie = DeviceGridArchive{T,T}(Array, (4, 4), ((0, 1), (0, 1)), 2)
        host = GridArchive{Vector{T},T}((4, 4), ((0, 1), (0, 1)))
        pair = [Evaluation(first_fitness, SVector(0.1, 0.1)), Evaluation(second_fitness, SVector(0.1, 0.1))]
        QD.add_batch!(tie, [T[1, 1], T[2, 2]], pair)
        QD.add_batch!(host, [T[1, 1], T[2, 2]], pair)
        @test first.(collect(QD.elites(tie))) == first.(collect(QD.elites(host))) == [T[1, 1]]
    end
    clamped = DeviceGridArchive{Float64,Float64}(Array, (4, 4), ((0, 1), (0, 1)), 2; outside=ClampOutside())
    @test_throws "descriptor component 1 is NaN and cannot be clamped" QD.add_batch!(clamped, xs[1:1], [Evaluation(1.0, SVector(NaN, 0.5))])
    @test length(clamped) == 0
    # the archive still works after an error
    @test [r.status for r in QD.add_batch!(dev, xs, good)] == [QD.NewCell, QD.NewCell, QD.NewCell]
    @test length(dev) == 4

    @test_throws "got 3 solutions but 2 evaluations" QD.add_batch!(dev, xs, good[1:2])
    @test_throws "solution has 3 entries but the archive stores solutions of length 2" QD.add_batch!(dev, [[0.1, 0.2, 0.3]], good[1:1])
    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.add_batch!(dev, xs[1:1], [Evaluation(1.0, SVector(0.1, 0.2, 0.3))])
    @test_throws "the evaluation has a phenotype of type String but the archive stores none" QD.add_batch!(dev, xs[1:1], [Evaluation(1.0, SVector(0.1, 0.2), "p")])
    @test_throws "descriptors must be 2 × 3, one column per solution, got 2 × 2" QD.EvaluationBatch{2}(zeros(3), zeros(2, 2))
    @test_throws "the kernels of the device archives need KernelAbstractions.jl" QD.device_add_batch!(nothing, nothing, nothing, nothing)
    @test_throws "the kernels of the device archives need KernelAbstractions.jl" QD.device_evaluate!(nothing, nothing, nothing, nothing)
    @test_throws "the kernels of the device archives need KernelAbstractions.jl" QD.device_vary!(nothing, nothing, nothing, nothing)
    @test_throws "a BanditScheduler collects the batches of its emitters in a Vector and cannot run on a DeviceCellArchive" QD.ask(Xoshiro(1), BanditScheduler((VariationEmitter(GaussianMutation(0.1); batch_size=2),); num_active=1), dev)
end

@testset "conformance" begin
    prob = Rastrigin(4)
    test_archive(() -> DeviceGridArchive(prob; dims=(8, 8), device=Array), prob; sequential_batch=false)
    test_archive(() -> DeviceGridArchive(prob; dims=(1, 1), device=Array), prob; sequential_batch=false)
    test_archive(() -> DeviceGridArchive(prob; dims=(5, 3), device=Array, genotype=Vector{Float32}, fitness=Float32),
                 prob; sequential_batch=false)
    test_archive(() -> DeviceGridArchive(prob; dims=(8, 8), device=Array, outside=RejectOutside()), prob;
                 sequential_batch=false)
    test_archive(() -> DeviceGridArchive(prob; dims=(8, 8), device=JLArray), prob; sequential_batch=false)

    for op in (GaussianMutation(0.1), IsoLineVariation(0.01, 0.2))
        archive = DeviceGridArchive(prob; dims=(8, 8), device=Array)
        QD.add_batch!(archive, [QD.random_solution(Xoshiro(k), prob) for k in 1:20],
                      [QD.evaluate(prob, QD.random_solution(Xoshiro(k), prob)) for k in 1:20])
        test_emitter(QD.DeviceVariationEmitter(op; batch_size=16), archive, prob)
    end
end

@testset "DeviceVariationEmitter" begin
    prob = Rastrigin(6)
    archive = DeviceGridArchive(prob; dims=(10, 10), device=Array)
    xs = [QD.random_solution(Xoshiro(k), prob) for k in 1:50]
    QD.add_batch!(archive, xs, [QD.evaluate(prob, x) for x in xs])
    elite_set = [x for (x, _) in QD.elites(archive)]

    emitter = QD.DeviceVariationEmitter(GaussianMutation(1e-9); batch_size=300)
    batch = QD.ask(Xoshiro(1), emitter, archive)
    @test batch isa QD.DeviceBatch{Float64}
    @test size(parent(batch)) == (6, 300)
    # a tiny step leaves each child next to an elite, and every elite is drawn
    nearest = [argmin(e -> maximum(abs, e - x), elite_set) for x in batch]
    @test all(maximum(abs, nearest[j] - batch[j]) < 1e-7 for j in eachindex(batch))
    @test length(unique(nearest)) == length(elite_set)
    @test collect(QD.ask(Xoshiro(1), emitter, archive)) == collect(batch)
    @test collect(QD.ask(Xoshiro(2), emitter, archive)) != collect(batch)

    # the step has the standard deviation of the operator, in the element type of the archive
    wide = QD.DeviceVariationEmitter(GaussianMutation(0.5); batch_size=2000)
    steps = [minimum(e -> sum(abs2, e - x), elite_set) for x in QD.ask(Xoshiro(3), wide, archive)]
    @test abs(sum(steps) / length(steps) / (6 * 0.25) - 1) < 0.1

    # iso-line children with no isotropic noise lie on the line through two elites
    line = QD.DeviceVariationEmitter(IsoLineVariation(0.0, 0.3); batch_size=200)
    children = QD.ask(Xoshiro(4), line, archive)
    on_line(x) = any(((p, q),) -> (d = q - p; d == zero(d) ? x ≈ p :
                                         (t = sum((x - p) .* d) / sum(abs2, d); isapprox(x, p + t * d; atol=1e-9))),
                     ((p, q) for p in elite_set, q in elite_set))
    @test all(on_line, children)

    @test_throws "DeviceVariationEmitter supports GaussianMutation and IsoLineVariation, not PolynomialMutation" QD.DeviceVariationEmitter(PolynomialMutation(-1.0, 1.0); batch_size=4)
    @test_throws "batch_size must be at least 1, got 0" QD.DeviceVariationEmitter(GaussianMutation(0.1); batch_size=0)
    @test_throws "cannot sample elites from an empty archive" QD.ask(Xoshiro(1), emitter, DeviceGridArchive(prob; dims=(2, 2), device=Array))
    @test_throws "a DeviceVariationEmitter draws its parents from a device archive such as a DeviceGridArchive, not a GridArchive" QD.ask(Xoshiro(1), emitter, GridArchive(prob; dims=(2, 2)))
end

@testset "evaluate_batch of a device batch" begin
    rng = Xoshiro(2)
    for prob in (Rastrigin(6), Sphere(6))
        X = 4 .* randn(rng, 6, 40)
        evs = QD.evaluate_batch(prob, QD.DeviceBatch(X))
        @test evs isa QD.EvaluationBatch{2,Float64}
        @test collect(evs) == [QD.evaluate(prob, x) for x in eachcol(X)]
        @test QD.evaluate_batch(Xoshiro(1), prob, QD.DeviceBatch(X)) == evs
    end
    # the benchmarks compute in Float32 on a device, and in Float64 in `evaluate`
    for prob in (Rastrigin(6), Sphere(6))
        X = 4 .* randn(rng, Float32, 6, 40)
        evs = QD.evaluate_batch(prob, QD.DeviceBatch(JLArray(X)))
        @test evs isa QD.EvaluationBatch{2,Float32}
        @test QD.device_evaluate(prob, X[:, 1]).fitness isa Float32
        @test QD.evaluate(prob, X[:, 1]).fitness isa Float64
        host = [QD.evaluate(prob, x) for x in eachcol(X)]
        @test Array(evs.fitness) ≈ [ev.fitness for ev in host] rtol = 1e-5
        @test Array(evs.descriptors) ≈ reduce(hcat, [ev.descriptor for ev in host]) rtol = 1e-5
    end
    f32 = QDProblem(x -> (-sum(abs2, x), (x[1], x[2])); init=rng -> rand(rng, Float32, 3), bounds=((0, 1), (0, 1)))
    X = rand(rng, Float32, 3, 10)
    evs = QD.evaluate_batch(f32, QD.DeviceBatch(JLArray(X)))
    @test evs.fitness isa JLArray{Float32}
    @test Array(evs.fitness) == [-sum(abs2, x) for x in eachcol(X)]
    @test Array(evs.descriptors) == X[1:2, :]
    @test isempty(QD.evaluate_batch(f32, QD.DeviceBatch(zeros(Float32, 3, 0))))
    # a phenotype goes through the kernel and is left out of the batch, which stores none
    labelled = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]), :ring); init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    X = rand(rng, 3, 4)
    evs = QD.evaluate_batch(labelled, QD.DeviceBatch(X))
    @test evs isa QD.EvaluationBatch{2,Float64}
    @test collect(evs) == [Evaluation(-sum(abs2, x), SVector(x[1], x[2])) for x in eachcol(X)]

    wrong = QDProblem(x -> (-sum(abs2, x), (x[1], x[2], x[3])); init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    @test_throws "the descriptor of QDProblem has 3 dimensions but its bounds have 2" QD.evaluate_batch(wrong, QD.DeviceBatch(rand(3, 2)))
    vector = QDProblem(x -> (-sum(abs2, x), x[1:2]); init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    @test_throws "it must be a tuple or an SVector" QD.evaluate_batch(vector, QD.DeviceBatch(rand(3, 2)))
    multi = QDProblem(x -> ([x[1], x[2]], (x[1], x[2])); init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    @test_throws "not an Evaluation with a real fitness; define QualityDiversity.evaluate_batch(prob::QDProblem, xs::DeviceBatch)" QD.evaluate_batch(multi, QD.DeviceBatch(rand(3, 2)))
    @test_throws "is a stochastic problem" QD.evaluate_batch(StochasticQDProblem((rng, x) -> (0.0, (0.0, 0.0)); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 1))), QD.DeviceBatch(rand(2, 2)))
end

@testset "MAP-Elites on a device" begin
    prob = Rastrigin(6)
    alg = MAPElites(prob; dims=(10, 10), batch_size=200, variation=GaussianMutation(0.5), device=Array)
    @test alg.archive isa DeviceGridArchive{Float64,Float64}
    @test alg.emitters[1] isa QD.DeviceVariationEmitter
    result = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test result.n_evals == 6_000
    @test result.archive isa DeviceGridArchive
    @test length(alg.archive) == 0                           # solve works on a copy
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(1)).history == result.history
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(2)).history != result.history
    last = result.history[end]
    @test last.coverage == coverage(result.archive)
    @test last.qd_score ≈ qd_score(result.archive; offset=QD.fitness_offset(prob))
    @test last.max_fitness == max_fitness(result.archive)
    host = GridArchive(result.archive)
    @test length(host) == length(result.archive)
    @test qd_score(host) ≈ qd_score(result.archive)

    # the quality of the host run
    cpu = solve(prob, MAPElites(prob; dims=(10, 10), batch_size=200, variation=GaussianMutation(0.5));
                budget=6_000, rng=Xoshiro(1))
    @test abs(coverage(result.archive) - coverage(cpu.archive)) < 0.1

    # JLArrays refuse scalar indexing, so the whole loop runs as kernels and copies
    f32 = QDProblem(x -> (-sum(abs2, x), (clamp(x[1], -1f0, 1f0), clamp(x[2], -1f0, 1f0)));
                    init=rng -> 2 .* rand(rng, Float32, 4) .- 1, bounds=((-1, 1), (-1, 1)))
    for op in (GaussianMutation(0.1f0), IsoLineVariation(0.01f0, 0.2f0))
        jl = MAPElites(f32; dims=(8, 8), batch_size=64, variation=op, device=JLArray,
                       genotype=Vector{Float32}, fitness=Float32)
        jl_result = solve(f32, jl; budget=2_000, rng=Xoshiro(3))
        @test jl_result.archive.genotypes isa JLArray{Float32}
        @test coverage(jl_result.archive) > 0.5
        array = MAPElites(f32; dims=(8, 8), batch_size=64, variation=op, device=Array,
                          genotype=Vector{Float32}, fitness=Float32)
        # the same archive; the backends sum the QD score in different orders
        array_result = solve(f32, array; budget=2_000, rng=Xoshiro(3))
        @test [h.coverage for h in array_result.history] == [h.coverage for h in jl_result.history]
        @test elites_by_cell(GridArchive(array_result.archive), array_result.archive) ==
              elites_by_cell(GridArchive(array_result.archive), jl_result.archive)
    end

    # an archive given directly, and a problem that scores its batch on the host
    gpu_prob = GPUExample.GPURastrigin(JLArray, 6)
    archive = DeviceGridArchive(gpu_prob; dims=(10, 10), device=Array, genotype=Vector{Float32}, fitness=Float32)
    gpu_alg = MAPElites(archive; batch_size=100, variation=GaussianMutation(0.5f0))
    @test gpu_alg.emitters[1] isa QD.DeviceVariationEmitter
    @test coverage(solve(gpu_prob, gpu_alg; budget=3_000, rng=Xoshiro(1)).archive) > 0.3

    # the host emitter also works with a device archive, through the host contract
    host_alg = QDAlgorithm(DeviceGridArchive(prob; dims=(10, 10), device=Array),
                           (VariationEmitter(GaussianMutation(0.5); batch_size=100),))
    @test coverage(solve(prob, host_alg; budget=3_000, rng=Xoshiro(1)).archive) > 0.3

    categories = QDProblem(x -> (0.0, (1, 1)); init=rng -> rand(rng, 2), categories=(1:2, 1:2))
    @test_throws "a DeviceGridArchive needs a problem with descriptor_bounds" MAPElites(categories; device=Array)
    @test_throws "dims is required" MAPElites(prob; device=Array)
    @test_throws "a DeviceGridArchive stores no phenotypes" MAPElites(prob; dims=(2, 2), device=Array, phenotype=String)
    noisy = StochasticQDProblem((rng, x) -> (0.0, (0.0, 0.0)); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 1)))
    @test_throws "a DeviceGridArchive takes deterministic problems" init(noisy, MAPElites(noisy; dims=(2, 2), device=Array); budget=100)
end

@testset "centroid buckets" begin
    rng = Xoshiro(4)
    points = [rand(rng, SVector{2,Float64}) for _ in 1:300]
    tree = QD.CentroidTree(points)
    for _ in 1:50
        q = rand(rng, SVector{2,Float64})
        radius2 = 0.05 * rand(rng)
        @test sort!(QD.within!(Int[], tree, q, radius2)) == [i for i in eachindex(points) if sum(abs2, points[i] - q) <= radius2]
    end
    buckets = QD.CentroidBuckets(tree, SVector(0.0, 0.0), SVector(1.0, 1.0))
    @test length(buckets.offsets) == prod(buckets.dims) + 1
    @test all(b -> issorted(buckets.members[buckets.offsets[b]:buckets.offsets[b+1]-1]), 1:prod(buckets.dims))
    # the nearest centroid of every point is among the centroids of its bucket
    for _ in 1:2000
        q = rand(rng, SVector{2,Float64})
        k = map((x, n) -> min(floor(Int, x * n) + 1, n), Tuple(q), buckets.dims)
        b = LinearIndices(buckets.dims)[k...]
        @test QD.nearest(tree, q) in buckets.members[buckets.offsets[b]:buckets.offsets[b+1]-1]
    end
    # in many dimensions one bucket holds every centroid
    many = [rand(rng, SVector{8,Float64}) for _ in 1:50]
    wide = QD.CentroidBuckets(QD.CentroidTree(many), zero(SVector{8,Float64}), ones(SVector{8,Float64}))
    @test length(wide.members) <= 32 * length(many)
end

@testset "DeviceCVTArchive" begin
    prob = Rastrigin(4)
    host = CVTArchive(prob; cells=40, samples=4000, rng=Xoshiro(5))
    dev = DeviceCVTArchive(prob; cells=40, samples=4000, device=Array, rng=Xoshiro(5))
    # the same seeding gives the same centroids as on the host
    greedy = DeviceCVTArchive(prob; cells=40, samples=4000, device=Array, rng=Xoshiro(5),
                              seeding=QD.GreedyKMeansPlusPlus())
    @test QD.centroids(greedy) == QD.centroids(CVTArchive(prob; cells=40, samples=4000, rng=Xoshiro(5),
                                                          seeding=QD.GreedyKMeansPlusPlus()))
    @test dev isa DeviceCVTArchive{Float64,Float64,2}
    @test dev isa QD.DeviceCellArchive
    @test QD.centroids(dev) == QD.centroids(host)
    @test QD.capacity(dev) == 40
    @test QD.descriptor_bounds(dev) == QD.descriptor_bounds(host)
    given = DeviceCVTArchive{Float32,Float32}(JLArray, QD.centroids(host), QD.descriptor_bounds(prob), 4)
    @test given.genotypes isa JLArray{Float32}
    @test QD.centroids(given) == QD.centroids(host)
    @test_throws "centroid 2 is (5.0, 0.0), outside the bounds" DeviceCVTArchive{Float64,Float64}(Array, [[0.0, 0.0], [5.0, 0.0]], ((-1, 1), (-1, 1)), 2)
    @test_throws "two centroids are identical" DeviceCVTArchive{Float64,Float64}(Array, [[0.0, 0.0], [0.0, 0.0]], ((-1, 1), (-1, 1)), 2)
    @test_throws "centroid 1 has 3 entries but bounds has 2" DeviceCVTArchive{Float64,Float64}(Array, [[0.0, 0.0, 0.0]], ((-1, 1), (-1, 1)), 2)
    @test_throws "cells must be at least 1, got 0" DeviceCVTArchive{Float64,Float64}(Array, 0, ((0, 1), (0, 1)), 2; rng=Xoshiro(1))

    # add_batch! agrees with CVTArchive, for every outside policy and fitness type
    rng = Xoshiro(6)
    for outside in (ThrowOutside(), ClampOutside(), RejectOutside()), F in (Float64, Float32), device in (Array, JLArray)
        h = CVTArchive{Vector{Float64},F}(30, ((0, 1), (0, 2)); samples=3000, outside, rng=Xoshiro(7))
        d = DeviceCVTArchive{Float64,F}(device, 30, ((0, 1), (0, 2)), 3; samples=3000, outside, rng=Xoshiro(7))
        lo, hi = outside isa ThrowOutside ? ((0.0, 0.0), (1.0, 2.0)) : ((-0.3, -0.3), (1.3, 2.3))
        for round in 1:5
            n = rand(rng, 1:60)
            xs = [randn(rng, 3) for _ in 1:n]
            evs = [Evaluation(F(rand(rng, -3:3)), SVector(lo[1] + (hi[1] - lo[1]) * rand(rng), lo[2] + (hi[2] - lo[2]) * rand(rng)))
                   for _ in 1:n]
            @test QD.add_batch!(d, xs, evs) == QD.add_batch!(h, xs, evs)
            @test elites_by_cell(h, d) == elites_by_cell(h, h)
        end
        @test qd_score(d) ≈ qd_score(h)
        converted = CVTArchive(d)
        @test converted isa CVTArchive{Vector{Float64},F}
        @test elites_by_cell(h, converted) == elites_by_cell(h, h)
    end
    strict = DeviceCVTArchive{Float64,Float64}(Array, 30, ((0, 1), (0, 2)), 3; samples=3000, rng=Xoshiro(7))
    @test_throws "descriptor component 2 is 2.5, outside the archive bounds" QD.add_batch!(strict, [zeros(3)], [Evaluation(1.0, SVector(0.5, 2.5))])
    @test length(strict) == 0

    # the kernels assign the cell of the host k-d tree, also at exact ties between centroids
    step = 1 / 10
    grid = vec([[(i - 0.5) * step, (j - 0.5) * step] for i in 1:10, j in 1:10])
    tied = DeviceCVTArchive{Float64,Float64}(Array, grid, ((0, 1), (0, 1)), 1)
    points = [SVector(rand(rng), rand(rng)) for _ in 1:500]
    append!(points, [(SVector{2}(grid[i]) + SVector{2}(grid[j])) / 2 for i in eachindex(grid) for j in eachindex(grid)
                     if 0 < sum(abs2, grid[i] - grid[j]) <= 2.01 * step^2])
    xs = [[0.0] for _ in points]
    evs = [Evaluation(Float64(k), p) for (k, p) in enumerate(points)]
    for (x, ev) in zip(xs, evs)
        empty!(tied)
        QD.add_batch!(tied, [x], [ev])
        @test only(QD.host_occupied(tied)) == QD.cell_index(tied, ev.descriptor)
    end

    test_archive(() -> DeviceCVTArchive{Float64,Float64}(Array, QD.centroids(host), QD.descriptor_bounds(prob), 4), prob;
                 sequential_batch=false)
    test_archive(() -> DeviceCVTArchive{Float32,Float32}(JLArray, QD.centroids(host), QD.descriptor_bounds(prob), 4;
                                                         outside=RejectOutside()), prob; sequential_batch=false)

    # MAP-Elites over it, reproducible and close to the host archive
    alg = MAPElites(dev; batch_size=200, variation=GaussianMutation(0.5))
    @test alg.emitters[1] isa QD.DeviceVariationEmitter
    result = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test result.archive isa DeviceCVTArchive
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(1)).history == result.history
    cpu = solve(prob, MAPElites(host; batch_size=200, variation=GaussianMutation(0.5)); budget=6_000, rng=Xoshiro(1))
    @test abs(coverage(result.archive) - coverage(cpu.archive)) < 0.15
    noisy = StochasticQDProblem((rng, x) -> (0.0, (0.0, 0.0)); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 1)))
    @test_throws "a DeviceCVTArchive takes deterministic problems" init(noisy, MAPElites(DeviceCVTArchive{Float64,Float64}(Array, [[0.5, 0.5]], ((0, 1), (0, 1)), 2)); budget=100)
end

@testset "DeviceCMAEmitter" begin
    prob = Rastrigin(6)
    emitter = QD.DeviceCMAEmitter(6; num_emitters=4, sigma0=0.5)
    @test QD.batch_size(emitter) == 4 * QD.batch_size(emitter.searches[1])
    @test length(emitter.searches) == 4
    # every search keeps its own copy of a ranker with state
    directed = QD.DeviceCMAEmitter(6; num_emitters=2, sigma0=0.5, ranker=QD.TwoStageRandomDirectionRanker())
    @test directed.searches[1].ranker !== directed.searches[2].ranker
    @test_throws "num_emitters must be at least 1, got 0" QD.DeviceCMAEmitter(6; num_emitters=0, sigma0=0.5)

    archive = DeviceGridArchive(prob; dims=(10, 10), device=Array)
    xs = [QD.random_solution(Xoshiro(k), prob) for k in 1:30]
    QD.add_batch!(archive, xs, [QD.evaluate(prob, x) for x in xs])
    test_emitter(QD.DeviceCMAEmitter(6; num_emitters=3, sigma0=0.5), deepcopy(archive), prob)

    # the searches start from x0, or from elites of the archive
    near = QD.DeviceCMAEmitter(6; num_emitters=3, sigma0=1e-6, x0=fill(0.25, 6))
    batch = QD.ask(Xoshiro(1), near, archive)
    @test batch isa QD.DeviceBatch{Float64}
    @test all(x -> maximum(abs, x .- 0.25) < 1e-4, batch)
    elite_set = [x for (x, _) in QD.elites(archive)]
    from_elites = QD.DeviceCMAEmitter(6; num_emitters=5, sigma0=1e-6)
    @test all(x -> any(e -> maximum(abs, x - e) < 1e-4, elite_set), QD.ask(Xoshiro(2), from_elites, archive))
    @test collect(QD.ask(Xoshiro(3), deepcopy(near), archive)) == collect(QD.ask(Xoshiro(3), deepcopy(near), archive))

    @test_throws "tell! needs the results of the batch that the last ask returned" QD.tell!(QD.DeviceCMAEmitter(6; num_emitters=2, sigma0=0.5), [], [], [])
    asked = deepcopy(near)
    b = QD.ask(Xoshiro(1), asked, archive)
    @test_throws "expected 27 solutions, evaluations and results but got 2, 2 and 2" QD.tell!(asked, b[1:2], b[1:2], b[1:2])
    @test_throws "a DeviceCMAEmitter adds its candidates to a device archive such as a DeviceGridArchive, not a GridArchive" QD.ask(Xoshiro(1), near, GridArchive(prob; dims=(2, 2)))
    @test_throws "the solutions of the archive have 6 entries but the emitter has dimension 4" QD.ask(Xoshiro(1), QD.DeviceCMAEmitter(4; num_emitters=2, sigma0=0.5), archive)

    # CMA-ME on a device: reproducible, with restarts, close to the host in quality
    alg = CMAME(prob; dims=(10, 10), sigma0=0.5, num_emitters=8, device=Array)
    @test alg.archive isa DeviceGridArchive
    @test only(alg.emitters) isa QD.DeviceCMAEmitter
    result = solve(prob, alg; budget=10_000, rng=Xoshiro(1))
    @test result.n_evals >= 10_000
    @test solve(prob, alg; budget=10_000, rng=Xoshiro(1)).history == result.history
    # the mean over seeds: a single run varies by about 0.1 in coverage
    mean_coverage(alg) = sum(s -> coverage(solve(prob, alg; budget=20_000, rng=Xoshiro(s)).archive), 1:6) / 6
    @test abs(mean_coverage(alg) - mean_coverage(CMAME(prob; dims=(10, 10), sigma0=0.5, num_emitters=8))) < 0.1
    state = init(prob, alg; budget=10_000, rng=Xoshiro(1))
    solve!(state)
    @test sum(s.restarts for s in only(state.emitters).searches) > 0
    @test all(s -> s.iterations == only(state.emitters).searches[1].iterations, only(state.emitters).searches)

    # every ranker and strategy runs, on JLArrays as well, over a grid or a CVT archive
    for kwargs in ((; ranker=QD.TwoStageRandomDirectionRanker()), (; ranker=QD.ObjectiveRanker(), selection=QD.MuSelection()),
                   (; strategy=QD.SepCMAES), (; strategy=QD.OpenAIES, restart=QD.ConvergenceRestart()))
        jl = CMAME(prob; dims=(10, 10), sigma0=0.5, num_emitters=4, device=JLArray, genotype=Vector{Float32},
                   fitness=Float32, kwargs...)
        @test coverage(solve(prob, jl; budget=3_000, rng=Xoshiro(2)).archive) > 0.2
    end
    cvt = CMAME(DeviceCVTArchive(prob; cells=50, samples=5000, device=Array, rng=Xoshiro(1)); sigma0=0.5, dim=6,
                num_emitters=4)
    @test coverage(solve(prob, cvt; budget=3_000, rng=Xoshiro(2)).archive) > 0.3
    @test_throws "a DeviceGridArchive stores no phenotypes" CMAME(prob; dims=(2, 2), sigma0=0.5, device=Array, phenotype=String)
end
