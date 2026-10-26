include("reference/categorical.jl")

@testset "CategoricalArchive construction" begin
    a = CategoricalArchive((0:3, [20, 10]))
    @test a isa CategoricalArchive{Vector{Float64},Float64,Tuple{Int,Int}}
    @test a isa QD.CellArchive{Vector{Float64},Float64,Tuple{Int,Int}}
    @test QD.capacity(a) == 8 && length(a) == 0 && coverage(a) == 0
    @test a.categories == ([0, 1, 2, 3], [20, 10])                   # in the order given

    labels = CategoricalArchive((0:5, [:alkane, :aromatic, :ring], ["a", "b"], (true, false)))
    @test labels isa QD.AbstractArchive{Vector{Float64},Float64,Tuple{Int,Symbol,String,Bool}}
    @test QD.capacity(labels) == 6 * 3 * 2 * 2
    @test CategoricalArchive{Vector{Float32},Float32}(([1, 2],)) isa
          CategoricalArchive{Vector{Float32},Float32,Tuple{Int}}
    @test CategoricalArchive(([0.5, 1.5],); outside=RejectOutside()).outside isa RejectOutside

    @test_throws "categories must have at least one entry" CategoricalArchive(())
    @test_throws "categories[2] is empty" CategoricalArchive((1:3, Int[]))
    @test_throws "categories[1] contains 2 more than once" CategoricalArchive(([2, 1, 2],))
    @test_throws "categories[1] contains a more than once" CategoricalArchive(([:a, :b, :a],))
    # `0.0` and `-0.0` are one category, which would leave a cell unreachable
    @test_throws "categories[1] contains -0.0 more than once" CategoricalArchive(([0.0, -0.0, 1.0],))
    @test_throws "categories[1] is [1.0, NaN] but its values must be finite" CategoricalArchive(([1.0, NaN],))
    @test_throws "categories[1] is [1.0, Inf] but its values must be finite" CategoricalArchive(([1.0, Inf],))
    @test_throws "ClampOutside() has no meaning for a CategoricalArchive" CategoricalArchive((1:3,); outside=ClampOutside())
    # values of mixed types would give a Tuple{Int,Any} descriptor and a Dict{Any,Int}
    @test_throws "categories[2] mixes values of the types Symbol and Int64; the values of a dimension must have one type" CategoricalArchive((1:3, [:a, 1]))
    @test_throws "categories[1] mixes values of the types Int64 and Float64" CategoricalArchive((Any[1, 2.5],))
    # a collection typed wider than its values is narrowed
    @test CategoricalArchive((Any[1, 2],)) isa CategoricalArchive{Vector{Float64},Float64,Tuple{Int}}
    @test CategoricalArchive((Any[1, 2],)).lookups[1] isa Dict{Int,Int}
end

@testset "CategoricalArchive cell_index" begin
    a = CategoricalArchive((0:3, [10, 20], [-1.5, 0.5, 2.5]))
    @test QD.capacity(a) == 24
    # cells are numbered like a grid: first dimension fastest, values in the order given
    expected = 0
    for k in 1:3, j in 1:2, i in 1:4
        expected += 1
        @test QD.cell_index(a, (i - 1, 10 * j, [-1.5, 0.5, 2.5][k])) == expected
    end
    @test QD.cell_index(a, SVector(2.0, 20.0, 0.5)) == QD.cell_index(a, (2, 20, 0.5))   # equal values match
    @test QD.cell_index(a, [2, 20, 0.5]) == QD.cell_index(a, (2, 20, 0.5))

    # a rounded negative number is -0.0, which is the category 0
    @test QD.cell_index(a, (-0.0, 10, 0.5)) == QD.cell_index(a, (0, 10, 0.5))
    @test QD.cell_index(CategoricalArchive(([-0.0, 1.0],)), (0,)) == 1

    shuffled = CategoricalArchive(([30, 10, 20],))
    @test [QD.cell_index(shuffled, (v,)) for v in (30, 10, 20)] == [1, 2, 3]

    s = CategoricalArchive((0:5, [:alkane, :aromatic, :ring]))
    @test QD.cell_index(s, (3, :ring)) == 16
    @test QD.cell_index(s, (0, :alkane)) == 1 && QD.cell_index(s, (5, :ring)) == 18

    @test_throws "descriptor has 2 dimensions but the archive has 3" QD.cell_index(a, (0, 10))
    @test_throws "descriptor has 2 dimensions but the archive has 3" QD.cell_index(a, [0, 10])
    @test_throws "descriptor component 1 is 1.5, which is not one of the categories [0, 1, 2, 3]" QD.cell_index(a, (1.5, 10, 0.5))
    @test_throws "descriptor component 2 is 15, which is not one of the categories [10, 20]" QD.cell_index(a, (1, 15, 0.5))
    @test_throws "descriptor component 3 is NaN, which is not one of the categories" QD.cell_index(a, (1, 10, NaN))
    @test_throws "descriptor component 2 is chain, which is not one of the categories [:alkane, :aromatic, :ring]" QD.cell_index(s, (1, :chain))
    @test_throws "descriptor component 2 is 3, which is not one of the categories" QD.cell_index(s, (1, 3))
    @test_throws "descriptor component 1 is 4.0" QD.add!(a, [1.0], Evaluation(1.0, (4.0, 10, 0.5)))
    @test_throws "[1, 2, 3, 4] … [99, 100] (100 values)" QD.cell_index(CategoricalArchive((1:100,)), (0,))

    rejecting = CategoricalArchive((0:3, [10, 20]); outside=RejectOutside())
    @test QD.cell_index(rejecting, (1.5, 10)) == 0
    @test QD.cell_index(rejecting, (1, 15)) == 0
    @test QD.cell_index(rejecting, [1, NaN]) == 0
    @test QD.cell_index(rejecting, (1, 10)) == 2
    r = QD.add!(rejecting, [1.0], Evaluation(1.0f0, (1.5, 10)))
    @test r.status == QD.Rejected && r.improvement === -Inf && length(rejecting) == 0
end

@testset "CategoricalArchive stores and reports like any cell archive" begin
    rng = Xoshiro(2)
    a = CategoricalArchive((0:9, [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]))
    syms = [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]
    @test QD.add!(a, [1.0], Evaluation(-3.0, (2, :d))).status == QD.NewCell
    @test QD.add!(a, [2.0], Evaluation(-1.0, (2, :d))).status == QD.Improved
    @test QD.add!(a, [3.0], Evaluation(-2.0, (2, :d))).status == QD.Rejected
    @test QD.add!(a, [4.0], Evaluation(-9.0, (3, :c))).status == QD.NewCell
    @test sort(first.(collect(QD.elites(a)))) == [[2.0], [4.0]]
    @test Set(e.descriptor for (_, e) in QD.elites(a)) == Set([(2, :d), (3, :c)])

    # a vector or a float with an equal value is stored as the category itself
    b = CategoricalArchive((0:9, 0:9))
    QD.add!(b, [1.0], Evaluation(1.0, SVector(2.0, 3.0)))
    @test only(collect(QD.elites(b)))[2].descriptor === (2, 3)

    for _ in 1:3000
        QD.add!(a, [0.0], Evaluation(-100 * rand(rng), (rand(rng, 0:9), rand(rng, syms))))
    end
    @test length(a) == 100 && coverage(a) == 1.0
    @test qd_score(a; offset=100) ≈ generic_qd_score(a; offset=100) rtol = 1e-14
    @test max_fitness(a) == generic_max_fitness(a)
    @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
    empty!(a)
    @test length(a) == 0 && qd_score(a) == 0
    @test_throws "max_fitness is undefined" max_fitness(a)
end

@testset "QDProblem with categories" begin
    prob = QDProblem(x -> (-sum(abs2, x), (length(x), x[1] > 0 ? :up : :down));
                     init=rng -> randn(rng, 2), categories=(1:3, [:up, :down]))
    @test QD.descriptor_categories(prob) == (1:3, [:up, :down])
    @test QD.descriptor_bounds(prob) === nothing
    ev = QD.evaluate(prob, [1.0, 2.0])
    @test ev.fitness == -5.0 && ev.descriptor === (2, :up)
    @test QD.evaluate(prob, [-1.0]).descriptor === (1, :down)
    @test QD.descriptor_ndims(prob) == 2

    archive = CategoricalArchive(prob)
    @test archive isa CategoricalArchive{Vector{Float64},Float64,Tuple{Int,Symbol}}
    @test QD.capacity(archive) == 6
    @test CategoricalArchive(prob; genotype=Vector{Float32}, fitness=Float32, outside=RejectOutside()) isa
          CategoricalArchive{Vector{Float32},Float32,Tuple{Int,Symbol},<:Any,<:Any,RejectOutside}

    bounded = QDProblem(identity; init=identity, bounds=((0, 1),))
    @test QD.descriptor_categories(bounded) === nothing
    @test QD.descriptor_ndims(bounded) == 1
    @test_throws "defines no descriptor_categories" CategoricalArchive(bounded)
    @test_throws "descriptor has 3 dimensions but the problem categories have 2" QD.evaluate(
        QDProblem(x -> (0.0, (1, 2, 3)); init=identity, categories=(1:3, 1:3)), 0)
    @test_throws "give exactly one of bounds and categories" QDProblem(identity; init=identity)
    @test_throws "give exactly one of bounds and categories" QDProblem(identity; init=identity, bounds=((0, 1),), categories=(1:2,))
end

@testset "MAPElites over a CategoricalArchive" begin
    # a solution's descriptor is its rounded first coordinate and the sign of the second
    prob = QDProblem(x -> (-sum(abs2, x), (round(Int, clamp(x[1], -3, 3)), x[2] >= 0 ? :pos : :neg));
                     init=rng -> 6 .* rand(rng, 2) .- 3, categories=(-3:3, [:neg, :pos]))
    archive = CategoricalArchive(prob)
    alg = MAPElites(archive; batch_size=20, variation=GaussianMutation(1.0))
    result = solve(prob, alg; budget=2_000, rng=Xoshiro(1))
    @test result.n_evals == 2_000
    @test result.archive isa CategoricalArchive
    @test length(result.archive) == 14 && coverage(result.archive) == 1.0
    @test length(archive) == 0
    @test solve(prob, alg; budget=2_000, rng=Xoshiro(1)).history == result.history
    @test max_fitness(result.archive) > -0.1

    # MAPElites(prob) builds the archive from the categories
    shortcut = MAPElites(prob; batch_size=20, outside=RejectOutside())
    @test shortcut.archive isa CategoricalArchive{Vector{Float64},Float64,Tuple{Int,Symbol}}
    @test shortcut.archive.outside isa RejectOutside && QD.capacity(shortcut.archive) == 14
    @test MAPElites(prob; genotype=Vector{Float32}, fitness=Float32).archive isa
          CategoricalArchive{Vector{Float32},Float32}
    @test solve(prob, MAPElites(prob; batch_size=20, variation=GaussianMutation(1.0)); budget=2_000,
                rng=Xoshiro(1)).history == result.history
    @test_throws "dims and bounds do not apply to a problem with descriptor_categories" MAPElites(prob; dims=(7, 2))
    @test_throws "dims and bounds do not apply to a problem with descriptor_categories" MAPElites(prob; bounds=((0, 1), (0, 1)))
    @test_throws "dims is required for a problem with descriptor_bounds" MAPElites(Rastrigin(4))
    @test MAPElites(Rastrigin(4); dims=(5, 5), bounds=((-1, 1), (-1, 1))).archive.upper == SVector(1.0, 1.0)

    # the archive and the problem must agree on the number of descriptor dimensions
    wrong = CategoricalArchive((1:3,))
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, MAPElites(wrong); budget=100)

    # numeric categories with a problem that has bounds work as well
    numeric = QDProblem(x -> (-sum(abs2, x), (round(clamp(x[1], -3, 3)), round(clamp(x[2], -3, 3))));
                        init=rng -> 6 .* rand(rng, 2) .- 3, bounds=((-3, 3), (-3, 3)))
    r = solve(numeric, MAPElites(CategoricalArchive((-3:3, -3:3)); batch_size=20, variation=GaussianMutation(1.0));
              budget=2_000, rng=Xoshiro(1))
    @test length(r.archive) == 49
    @test eltype(first.(QD.elites(r.archive))) == Vector{Float64}
end

@testset "CategoricalArchive matches pyribs" begin
    # reference/categorical.jl: the statuses, improvements, elites and metrics of pyribs's
    # CategoricalArchive after each batch of a sequence
    pyribs_status = Dict(0 => QD.Rejected, 1 => QD.Improved, 2 => QD.NewCell)
    c = categorical_case
    a = CategoricalArchive(c.categories)
    descriptor(i) = (c.categories[1][c.first[i]+1], c.categories[2][c.second[i]+1])
    numbers(a) = sort([Int(only(x)) for (x, _) in QD.elites(a)])
    statuses, values = QD.AddStatus[], Float64[]
    for b in 1:length(c.fitness)÷c.batch
        ids = (b-1)*c.batch+1:b*c.batch
        rs = QD.add_batch!(a, [[Float64(i)] for i in ids], [Evaluation(c.fitness[i], descriptor(i)) for i in ids])
        append!(statuses, [r.status for r in rs])
        append!(values, [r.improvement for r in rs])
        @test numbers(a) == c.elites[b]
        elites, cov, score, best, average = c.stats[:, b]
        @test length(a) == elites && coverage(a) ≈ cov
        @test qd_score(a; offset=-c.offset) ≈ score rtol = 1e-12
        @test max_fitness(a) == best
        @test mean_fitness(a) ≈ average rtol = 1e-12
    end
    @test statuses == [pyribs_status[s] for s in c.statuses]
    @test values ≈ c.values rtol = 1e-12
end
