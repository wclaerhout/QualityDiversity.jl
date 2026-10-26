brute_nearest_slot(archive, d) =
    argmin([sum(abs2, SVector{2,Float64}(d) - e.descriptor) for (_, e) in QD.elites(archive)])

@testset "UnstructuredArchive construction" begin
    a = UnstructuredArchive(2; capacity=10, novelty_threshold=0.5)
    @test a isa UnstructuredArchive{Vector{Float64},Float64,2}
    @test a isa QD.AbstractArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test length(a) == 0 && QD.capacity(a) == 10 && coverage(a) == 0
    @test UnstructuredArchive{Vector{Float32}}(3; capacity=4, novelty_threshold=1) isa
          UnstructuredArchive{Vector{Float32},Float64,3}
    @test UnstructuredArchive{Vector{Float32},Float32}(3; capacity=4, novelty_threshold=1) isa
          UnstructuredArchive{Vector{Float32},Float32,3}
    b = UnstructuredArchive(Rastrigin(4); capacity=7, novelty_threshold=0.25, fitness=Float32)
    @test b isa UnstructuredArchive{Vector{Float64},Float32,2} && b.threshold == 0.25
    @test UnstructuredArchive(2; capacity=3, novelty_threshold=0).threshold == 0.0

    @test_throws "capacity must be at least 1, got 0" UnstructuredArchive(2; capacity=0, novelty_threshold=1)
    @test_throws "ndims must be at least 1, got 0" UnstructuredArchive(0; capacity=3, novelty_threshold=1)
    @test_throws "novelty_threshold must be finite and at least 0, got -1" UnstructuredArchive(2; capacity=3, novelty_threshold=-1)
    @test_throws "novelty_threshold must be finite and at least 0, got NaN" UnstructuredArchive(2; capacity=3, novelty_threshold=NaN)
    @test_throws "novelty_threshold must be finite and at least 0, got Inf" UnstructuredArchive(2; capacity=3, novelty_threshold=Inf)
    @test_throws UndefKeywordError UnstructuredArchive(2; capacity=3, threshold=1)
    labels = QDProblem(x -> (0.0, (1, :a)); init=rng -> rand(rng, 2), categories=(1:2, [:a]))
    @test_throws "has descriptor_categories, not descriptor_bounds" UnstructuredArchive(labels; capacity=3, novelty_threshold=1)
end

@testset "UnstructuredArchive novelty and local competition" begin
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))
    a = UnstructuredArchive(2; capacity=10, novelty_threshold=1.0)

    @test QD.cell_index(a, [0.0, 0.0]) == 0
    r = QD.add!(a, [1.0], ev(-5.0, (0, 0)))
    @test r.status == QD.NewCell && r.improvement == -5.0 && length(a) == 1

    # farther than the threshold: novel, whatever its fitness
    r = QD.add!(a, [2.0], ev(-50.0, (1.5, 0)))
    @test r.status == QD.NewCell && r.improvement == -50.0 && length(a) == 2

    # at exactly the threshold the solution is not novel; it competes with its neighbour
    r = QD.add!(a, [3.0], ev(-9.0, (0, 1.0)))
    @test r.status == QD.Rejected && r.improvement == -4.0 && length(a) == 2
    r = QD.add!(a, [3.0], ev(-5.0, (0, 1.0)))
    @test r.status == QD.Rejected && r.improvement == 0.0
    r = QD.add!(a, [4.0], ev(-2.0, (0, 1.0)))
    @test r.status == QD.Improved && r.improvement == 3.0 && length(a) == 2
    # the winner brings its own descriptor
    @test QD.cell_index(a, [0.0, 1.0]) == 1
    @test collect(QD.elites(a))[1][1] == [4.0]
    @test collect(QD.elites(a))[1][2].descriptor == SVector(0.0, 1.0)

    # just beyond the threshold of both stored descriptors: novel again
    r = QD.add!(a, [5.0], ev(-100.0, (0.0, 2.5)))
    @test r.status == QD.NewCell && length(a) == 3

    # with threshold 0 only an identical descriptor competes
    z = UnstructuredArchive(2; capacity=10, novelty_threshold=0)
    @test QD.add!(z, [1.0], ev(1.0, (0.5, 0.5))).status == QD.NewCell
    @test QD.add!(z, [2.0], ev(0.0, (0.5, 0.5 + 1e-12))).status == QD.NewCell
    @test QD.add!(z, [3.0], ev(0.5, (0.5, 0.5))).status == QD.Rejected
    @test QD.add!(z, [4.0], ev(2.0, (0.5, 0.5))).status == QD.Improved
    @test length(z) == 2
end

@testset "UnstructuredArchive when full" begin
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))
    a = UnstructuredArchive(2; capacity=3, novelty_threshold=1.0)
    for (i, f) in enumerate((-3.0, -1.0, -2.0))
        QD.add!(a, [Float64(i)], ev(f, (10i, 0)))
    end
    @test length(a) == QD.capacity(a) == 3 && coverage(a) == 1
    @test qd_score(a; offset=10) == 24 && max_fitness(a) == -1.0

    # a novel solution no better than the weakest is rejected
    r = QD.add!(a, [9.0], ev(-3.0, (100, 0)))
    @test r.status == QD.Rejected && r.improvement == 0.0 && length(a) == 3
    r = QD.add!(a, [9.0], ev(-4.0, (100, 0)))
    @test r.status == QD.Rejected && r.improvement == -1.0

    # a better one replaces the weakest (fitness -3, genotype [1.0])
    r = QD.add!(a, [9.0], ev(-1.5, (100, 0)))
    @test r.status == QD.NewCell && r.improvement == -1.5 && length(a) == 3
    @test sort(first.(collect(QD.elites(a)))) == [[2.0], [3.0], [9.0]]
    @test qd_score(a) ≈ -4.5 && max_fitness(a) == -1.0 && mean_fitness(a) ≈ -1.5
    @test QD.cell_index(a, [100.0, 0.0]) == 1 && collect(QD.elites(a))[1][1] == [9.0]

    # a better one close to a stored solution still improves it
    r = QD.add!(a, [7.0], ev(5.0, (20.0, 0.5)))
    @test r.status == QD.Improved && r.improvement == 6.0
    @test max_fitness(a) == 5.0 && qd_score(a) ≈ 1.5

    # Both replacements add the new fitness and subtract the old one as two compensated
    # steps, so the total with its error term tracks the exact sum far below one rounding
    # even when their difference is not representable (magnitudes below 1 against 2^53).
    b = UnstructuredArchive(2; capacity=500, novelty_threshold=0.5)
    rng = Xoshiro(3)
    exact = BigFloat(0)
    for i in 1:500
        QD.add!(b, [0.0], ev(-rand(rng), (i, 0)))                       # fills the slots
    end
    for i in 1:500
        new = 2.0^53 * (1 + rand(rng))
        QD.add!(b, [0.0], ev(new, (isodd(i) ? i : 1000 + i, 0)))      # the nearest, or the weakest once full
        exact += new
    end
    @test length(b) == 500
    two_terms = BigFloat(b.fitness_sum) + BigFloat(b.fitness_sum_error)
    @test abs(two_terms - exact) <= 1e-24 * exact
    @test qd_score(b) == Float64(exact)
end

@testset "UnstructuredArchive cell_index finds the nearest stored descriptor" begin
    rng = Xoshiro(5)
    a = UnstructuredArchive(2; capacity=50, novelty_threshold=0.05)
    for _ in 1:300
        QD.add!(a, [0.0], Evaluation(-rand(rng), rand(rng, SVector{2,Float64})))
    end
    @test length(a) > 20
    for _ in 1:200
        d = 1.5 .* rand(rng, SVector{2,Float64}) .- 0.25
        @test QD.cell_index(a, d) == brute_nearest_slot(a, d)
        @test QD.cell_index(a, collect(d)) == QD.cell_index(a, d)
    end
    # equally near: the lowest slot
    t = UnstructuredArchive(1; capacity=5, novelty_threshold=0.1)
    QD.add!(t, [1.0], Evaluation(0.0, SVector(0.0)))
    QD.add!(t, [2.0], Evaluation(0.0, SVector(2.0)))
    @test QD.cell_index(t, [1.0]) == 1
end

@testset "UnstructuredArchive rejects bad descriptors" begin
    a = UnstructuredArchive(2; capacity=5, novelty_threshold=1.0)
    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.cell_index(a, [0.0, 0.0, 0.0])
    @test_throws "descriptor has 1 dimensions but the archive has 2" QD.add!(a, [1.0], Evaluation(1.0, SVector(0.0)))
    @test_throws "descriptor component 2 is NaN, which is not finite" QD.cell_index(a, [0.0, NaN])
    @test_throws "descriptor component 1 is Inf, which is not finite" QD.add!(a, [1.0], Evaluation(1.0, SVector(Inf, 0.0)))
    @test length(a) == 0
    @test_throws "cannot sample elites from an empty archive" QD.sample_elites(Xoshiro(1), a, 3)
    @test_throws "max_fitness is undefined" max_fitness(a)
    @test_throws "mean_fitness is undefined" mean_fitness(a)
    @test qd_score(a) == 0 && qd_score(a; offset=5) == 0
end

@testset "UnstructuredArchive running metrics" begin
    rng = Xoshiro(11)
    a = UnstructuredArchive(2; capacity=200, novelty_threshold=0.04)
    function check(a)
        @test coverage(a) == length(a) / QD.capacity(a)
        @test qd_score(a; offset=0) ≈ generic_qd_score(a; offset=0) rtol = 1e-14
        @test qd_score(a; offset=250.5) ≈ generic_qd_score(a; offset=250.5) rtol = 1e-14
        @test max_fitness(a) == generic_max_fitness(a)
        @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
    end
    # new slots, improvements, evictions and rejections, checking along the way
    for round in 1:20, _ in 1:500
        QD.add!(a, [0.0], Evaluation(-100 * rand(rng), rand(rng, SVector{2,Float64})))
        round % 4 == 0 && QD.add!(a, [0.0], Evaluation(-100 * rand(rng) + 50, rand(rng, SVector{2,Float64})))
    end
    check(a)
    @test length(a) == 200
    empty!(a)
    @test length(a) == 0 && qd_score(a) == 0 && isempty(collect(QD.elites(a)))
    @test QD.add!(a, [1.0], Evaluation(-2.0, SVector(0.5, 0.5))).status == QD.NewCell
    @test max_fitness(a) == -2.0 && qd_score(a; offset=3) == 1.0
end

@testset "UnstructuredArchive sampling" begin
    a = UnstructuredArchive(2; capacity=20, novelty_threshold=0.1)
    for i in 1:5
        QD.add!(a, [Float64(i)], Evaluation(0.0, SVector(i / 5, 0.0)))
    end
    parents = QD.sample_elites(Xoshiro(1), a, 2_000)
    @test parents isa Vector{Vector{Float64}}
    @test sort(unique(parents)) == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    @test all(i -> abs(count(==([Float64(i)]), parents) - 400) < 80, 1:5)    # roughly uniform
    @test QD.sample_elites(Xoshiro(1), a, 7) == QD.sample_elites(Xoshiro(1), a, 7)
end

@testset "MAPElites over an UnstructuredArchive" begin
    prob = Rastrigin(6)
    archive = UnstructuredArchive(prob; capacity=300, novelty_threshold=0.8)
    alg = MAPElites(archive; batch_size=50, variation=IsoLineVariation(0.05, 0.3))
    result = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test result.n_evals == 6_000
    @test result.archive isa UnstructuredArchive
    @test length(archive) == 0                     # the algorithm is not modified
    @test length(result.archive) > 100
    @test issorted(r.max_fitness for r in result.history)
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(1)).history == result.history

    wrong = UnstructuredArchive(1; capacity=10, novelty_threshold=0.1)
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, MAPElites(wrong); budget=100)
end

@testset "UnstructuredArchive replaces the first of equally weak solutions" begin
    a = UnstructuredArchive(1; capacity=3, novelty_threshold=0.1)
    for (x, f, d) in (([1.0], 1.0, 0.0), ([2.0], 0.0, 0.5), ([3.0], 0.0, 1.0))
        QD.add!(a, x, Evaluation(f, SVector(d)))
    end
    # a novel solution no fitter than the weakest is rejected
    @test QD.add!(a, [4.0], Evaluation(0.0, SVector(2.0))).status == QD.Rejected
    # a fitter one takes the slot of the first of the two weakest
    @test QD.add!(a, [5.0], Evaluation(0.5, SVector(3.0))).status == QD.NewCell
    @test sort(first.(collect(QD.elites(a)))) == [[1.0], [3.0], [5.0]]
    @test a.genotypes[2] == [5.0]
end

# The rules of `UnstructuredArchive` written out over plain vectors, one solution at a time.
function unstructured_rules!(slots, capacity, threshold, f, d)
    distances = [sum(abs2, d - s[2]) for s in slots]
    nearest = isempty(slots) ? 0 : argmin(distances)    # the first on a tie
    if nearest == 0 || distances[nearest] > threshold^2
        length(slots) < capacity && (push!(slots, (f, d)); return QD.NewCell)
        # `<`, not `argmin`: -0.0 and 0.0 are equally weak, and the first of them goes
        weakest = 1
        for i in eachindex(slots)
            slots[i][1] < slots[weakest][1] && (weakest = i)
        end
        f > slots[weakest][1] || return QD.Rejected
        slots[weakest] = (f, d)
        return QD.NewCell
    end
    f > slots[nearest][1] || return QD.Rejected
    slots[nearest] = (f, d)
    return QD.Improved
end

@testset "UnstructuredArchive with a tree, $N dimensions" for N in (1, 2)
    # more stored solutions than `16 * 4^N`, so that the archive searches a tree, and
    # enough of them replaced that it rebuilds it; coarse values make ties common
    rng = Xoshiro(N)
    capacity = 400
    threshold = N == 1 ? 0.002 : 0.02
    digits = N == 1 ? 3 : 2
    a = UnstructuredArchive(N; capacity, novelty_threshold=threshold)
    @test capacity > QD.tree_threshold(a)
    slots = Tuple{Float64,SVector{N,Float64}}[]
    for _ in 1:3000
        f = round(randn(rng); digits=1)
        d = SVector{N,Float64}(round.(rand(rng, N); digits))
        @test QD.add!(a, [f], Evaluation(f, d)).status == unstructured_rules!(slots, capacity, threshold, f, d)
        q = SVector{N,Float64}(round.(rand(rng, N); digits))
        @test QD.cell_index(a, q) == argmin([sum(abs2, q - s[2]) for s in slots])
    end
    @test a.tree_size > 0
    @test [(ev.fitness, ev.descriptor) for (_, ev) in QD.elites(a)] == slots
    empty!(a)
    @test length(a) == 0 && a.tree_size == 0 && a.num_pending == 0
    @test QD.add!(a, [1.0], Evaluation(1.0, zero(SVector{N,Float64}))).status == QD.NewCell
end
