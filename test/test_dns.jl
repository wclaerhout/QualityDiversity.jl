include("reference/dns.jl")

dns_batch(fs, ds) = ([[Float64(f)] for f in fs], [Evaluation(Float64(f), SVector{length(d),Float64}(d)) for (f, d) in zip(fs, ds)])
stored_descriptors(a) = [Tuple(ev.descriptor) for (_, ev) in QD.elites(a)]

@testset "DNSArchive construction" begin
    a = DNSArchive(2; capacity=10, k=3)
    @test a isa DNSArchive{Vector{Float64},Float64,2,Nothing}
    @test a isa AbstractArchive{Vector{Float64},Float64,SVector{2,Float64},Nothing}
    @test length(a) == 0
    @test QD.capacity(a) == 10
    @test QD.cell_index(a, SVector(0.0, 0.0)) == 0
    @test DNSArchive{Int,Float32}(1; capacity=2, k=1) isa DNSArchive{Int,Float32,1}
    @test DNSArchive(Rastrigin(4); capacity=5, k=2, phenotype=Int) isa DNSArchive{Vector{Float64},Float64,2,Int}

    @test_throws "capacity must be at least 1, got 0" DNSArchive(2; capacity=0, k=3)
    @test_throws "k must be at least 1, got 0" DNSArchive(2; capacity=5, k=0)
    @test_throws "ndims must be at least 1, got 0" DNSArchive(0; capacity=5, k=3)
    @test_throws "expected F<:AbstractFloat" DNSArchive{Vector{Float64},Int}(2; capacity=5, k=3)   # Inf needs a float
    @test_throws "expected F<:AbstractFloat" DNSArchive(Rastrigin(4); capacity=5, k=3, fitness=Int)
    labels = QDProblem(x -> (0.0, (1, :a)); init=rng -> rand(rng, 2), categories=(1:2, [:a]))
    @test_throws "has descriptor_categories, not descriptor_bounds" DNSArchive(labels; capacity=5, k=3)
end

@testset "DNSArchive scores" begin
    # k = 2: descriptor 0 has fitter neighbours at 1 and 3, descriptor 1 at 3 and 10,
    # descriptor 3 only one at 10, and descriptor 10 none
    a = DNSArchive(1; capacity=3, k=2)
    rs = QD.add_batch!(a, dns_batch(1:4, [(0,), (1,), (3,), (10,)])...)
    @test rs == [QD.AddResult(QD.Rejected, 2.0), QD.AddResult(QD.NewCell, 5.5), QD.AddResult(QD.NewCell, 7.0),
                 QD.AddResult(QD.NewCell, Inf)]
    @test stored_descriptors(a) == [(1.0,), (3.0,), (10.0,)]
    @test max_fitness(a) == 4
    @test mean_fitness(a) == 3
    @test qd_score(a; offset=1) == 12
    @test coverage(a) == 1

    # the next batch is scored against the population, and the stored solutions with it
    rs = QD.add_batch!(a, dns_batch([5], [(2,)])...)
    @test only(rs) == QD.AddResult(QD.NewCell, Inf)
    @test stored_descriptors(a) == [(3.0,), (10.0,), (2.0,)]   # 1 is now 1 from a fitter one
end

@testset "DNSArchive equal fitness" begin
    # equal fitness counts as fitter, so solutions on a plateau compete on distance
    a = DNSArchive(1; capacity=2, k=1)
    rs = QD.add_batch!(a, dns_batch([1, 1, 1], [(0,), (1,), (5,)])...)
    @test [r.improvement for r in rs] == [1.0, 1.0, 4.0]
    # descriptors 0 and 1 tie; the later one survives
    @test [r.status for r in rs] == [QD.Rejected, QD.NewCell, QD.NewCell]
    @test stored_descriptors(a) == [(1.0,), (5.0,)]

    # a solution of the batch goes before a stored one with the same score
    b = DNSArchive(1; capacity=2, k=1)
    QD.add_batch!(b, dns_batch([1, 1], [(0,), (4,)])...)
    @test only(QD.add_batch!(b, dns_batch([1], [(8,)])...)) == QD.AddResult(QD.NewCell, 4.0)
    @test stored_descriptors(b) == [(4.0,), (8.0,)]

    # the best solutions can all leave: two equal ones at the same place score 0
    c = DNSArchive(1; capacity=2, k=1)
    QD.add_batch!(c, dns_batch([5, 5, 1, 1], [(0,), (0,), (10,), (-10,)])...)
    @test stored_descriptors(c) == [(10.0,), (-10.0,)]
    @test max_fitness(c) == 1
end

@testset "DNSArchive add!" begin
    a = DNSArchive(1; capacity=3, k=2)
    b = DNSArchive(1; capacity=3, k=2)
    xs, evs = dns_batch([4, 1, 3, 2, 6, 5], [(0,), (2,), (7,), (1,), (-3,), (4,)])
    for (x, ev) in zip(xs, evs)
        @test QD.add!(a, x, ev) == only(QD.add_batch!(b, [x], [ev]))
    end
    @test collect(QD.elites(a)) == collect(QD.elites(b))
    @test QD.cell_index(a, SVector(6.0)) == findfirst(==((7.0,)), stored_descriptors(a))

    @test_throws "descriptor component 1 is NaN, which is not finite" QD.add!(a, [0.0], Evaluation(9.0, SVector(NaN)))
    @test_throws "descriptor has 2 dimensions but the archive has 1" QD.add!(a, [0.0], Evaluation(9.0, SVector(1.0, 2.0)))
    @test collect(QD.elites(a)) == collect(QD.elites(b))
    @test_throws "descriptor component 1 is Inf" QD.add_batch!(a, dns_batch([9, 9], [(1,), (Inf,)])...)
    @test collect(QD.elites(a)) == collect(QD.elites(b))
    @test_throws "got 2 solutions but 1 evaluations" QD.add_batch!(a, xs[1:2], evs[1:1])

    empty!(a)
    @test length(a) == 0
    @test QD.add!(a, xs[1], evs[1]) == QD.AddResult(QD.NewCell, Inf)
end

@testset "DNSArchive phenotypes" begin
    a = DNSArchive{Vector{Float64}}(1; capacity=2, k=1, phenotype=String)
    QD.add_batch!(a, [[1.0], [2.0], [3.0]],
                  [Evaluation(1.0, SVector(0.0), "a"), Evaluation(2.0, SVector(5.0), "b"), Evaluation(3.0, SVector(6.0), "c")])
    # moving survivors to earlier slots keeps their phenotypes with them
    @test Set((only(x), ev.phenotype) for (x, ev) in QD.elites(a)) == Set([(1.0, "a"), (3.0, "c")])
    @test_throws "the evaluation has no phenotype" QD.add!(a, [4.0], Evaluation(4.0, SVector(9.0)))
    @test length(a) == 2
end

@testset "MAP-Elites with a DNS archive" begin
    prob = Rastrigin(6)
    alg = MAPElites(DNSArchive(prob; capacity=200, k=3); batch_size=32)
    result = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test length(result.archive) == 200
    again = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test collect(QD.elites(again.archive)) == collect(QD.elites(result.archive))
    first_round = init(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test max_fitness(result.archive) > max_fitness(first_round.archive)
end

@testset "DNSArchive matches QDax and pyribs" begin
    # reference/dns.jl: the solutions that QDax and pyribs keep after each batch
    for case in dns_cases
        total = length(case.fitness)
        descriptors = [SVector(case.descriptors[2i-1], case.descriptors[2i]) for i in 1:total]
        a = DNSArchive{Int}(2; capacity=case.capacity, k=case.k)
        for (b, expected) in enumerate(case.kept)
            ids = (b-1)*case.batch+1:b*case.batch
            QD.add_batch!(a, collect(ids), [Evaluation(case.fitness[i], descriptors[i]) for i in ids])
            @test sort!([x for (x, _) in QD.elites(a)]) == expected
        end
    end
end

# Dominated novelty of solution `i` among `fitness` and `points`, from its definition.
function definition_score(fitness, points, i, k)
    distances = sort([sqrt(sum(abs2, points[i] - points[j])) for j in eachindex(points)
                      if j != i && fitness[j] >= fitness[i]])
    isempty(distances) && return Inf
    return sum(distances[1:min(k, end)]) / min(k, length(distances))
end

@testset "DNSArchive scores from a tree, $N dimensions" for N in (1, 2)
    # more solutions than `16 * 4^N`, so that the archive scores them with a tree; coarse
    # values make equal fitness and equal descriptors common
    rng = Xoshiro(N)
    a = DNSArchive(N; capacity=300, k=3)
    @test 300 + 40 >= QD.tree_threshold(a)
    for _ in 1:12
        fs = round.(randn(rng, 40); digits=1)
        ds = [Tuple(round.(rand(rng, N); digits=2)) for _ in 1:40]
        fitness = [[ev.fitness for (_, ev) in QD.elites(a)]; fs]
        points = [[ev.descriptor for (_, ev) in QD.elites(a)]; [SVector{N,Float64}(d) for d in ds]]
        stored = length(a)
        rs = QD.add_batch!(a, dns_batch(fs, ds)...)
        @test [r.improvement for r in rs] == [definition_score(fitness, points, stored + j, 3) for j in 1:40]
    end
end

@testset "a tree and a pass give the same scores" begin
    for seed in 1:4, N in (1, 3, 6), k in (1, 4)
        rng = Xoshiro(seed)
        a = DNSArchive(N; capacity=200, k)
        merged = rand(rng, 1:400)
        for v in (a.merged_fitness, a.merged_descriptors, a.tree.order, a.tree.axis, a.tree_boxes, a.tree_max)
            resize!(v, merged)
        end
        digits = rand(rng, 0:2)
        for m in 1:merged
            a.merged_fitness[m] = round(randn(rng); digits)
            a.merged_descriptors[m] = SVector{N}(round.(rand(rng, N); digits=digits + 1))
        end
        QD.build_fitness_tree!(a, merged)
        @test all(m -> QD.dominated_novelty(a, m, merged) === QD.dominated_novelty_scan(a, m, merged), 1:merged)
    end
end
