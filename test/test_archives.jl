include("reference/grid.jl")

@testset "GridArchive construction" begin
    a = GridArchive((4, 5), ((0, 1), (-1, 1)))
    @test a isa GridArchive{Vector{Float64},Float64,2}
    @test a isa AbstractArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test QD.capacity(a) == 20
    @test length(a) == 0
    @test GridArchive{Int}((2,), ((0, 1),)) isa GridArchive{Int,Float64,1}
    @test GridArchive{Int,Float32}((2,), ((0, 1),)) isa GridArchive{Int,Float32,1}
    @test GridArchive(Rastrigin(4); dims=(3, 3)) isa GridArchive{Vector{Float64},Float64,2}

    @test_throws "dims has 2 entries but bounds has 1" GridArchive((2, 2), ((0, 1),))
    @test_throws "dims[2] is 0" GridArchive((2, 0), ((0, 1), (0, 1)))
    @test_throws "bounds[1] is (1.0, 1.0)" GridArchive((2,), ((1, 1),))
    # a width that overflows would put every descriptor in the first cell
    @test_throws "bounds[1] is (-1.0e308, 1.0e308), whose width Inf is not finite" GridArchive((10,), ((-1e308, 1e308),))
    @test_throws "dims must have at least one entry" GridArchive((), ())
end

@testset "GridArchive.cell_index" begin
    a = GridArchive((4, 5), ((0, 1), (-1, 1)))
    @test QD.cell_index(a, [0.0, -1.0]) == 1
    @test QD.cell_index(a, [0.3, -1.0]) == 2
    @test QD.cell_index(a, [0.0, -0.5]) == 5
    @test QD.cell_index(a, [1.0, 1.0]) == 20     # upper bound is in the last cell
    @test QD.cell_index(a, SVector(0.99, 0.99)) == 20
    all_cells = [QD.cell_index(a, [x, y]) for x in 0.125:0.25:1, y in -0.8:0.4:1]
    @test sort(vec(all_cells)) == 1:20

    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.cell_index(a, [0.0, 0.0, 0.0])
    @test_throws "descriptor component 1 is 1.5, outside the archive bounds [0.0, 1.0]" QD.cell_index(a, [1.5, 0.0])
    @test_throws "descriptor component 2 is NaN" QD.cell_index(a, [0.5, NaN])
end

@testset "GridArchive.add!" begin
    a = GridArchive((2, 2), ((0, 1), (0, 1)))
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))

    r = QD.add!(a, [1.0], ev(-3.0, (0.1, 0.1)))
    @test r.status == QD.NewCell
    @test r.improvement == -3.0
    @test length(a) == 1

    r = QD.add!(a, [2.0], ev(-1.0, (0.2, 0.2)))
    @test r.status == QD.Improved
    @test r.improvement == 2.0
    @test length(a) == 1

    r = QD.add!(a, [3.0], ev(-2.0, (0.2, 0.2)))
    @test r.status == QD.Rejected
    @test r.improvement == -1.0
    r = QD.add!(a, [3.0], ev(-1.0, (0.2, 0.2)))
    @test r.status == QD.Rejected
    @test r.improvement == 0.0

    @test only(first.(collect(QD.elites(a)))) == [2.0]

    QD.add!(a, [4.0], ev(5.0, (0.9, 0.9)))
    @test length(a) == 2
    @test QD.coverage(a) == 0.5
    @test_throws "outside the archive bounds" QD.add!(a, [0.0], ev(0.0, (2.0, 0.0)))
    @test_throws "descriptor has 1 dimensions but the archive has 2" QD.add!(a, [0.0], Evaluation(0.0, SVector(0.5)))

    # fitness is converted to the archive's fitness type
    b = GridArchive{Vector{Float64},Float32}((1,), ((0, 1),))
    @test QD.add!(b, [0.0], Evaluation(1, SVector(0.5))).improvement === 1.0f0

    empty!(a)
    @test length(a) == 0
    @test QD.add!(a, [1.0], ev(-3.0, (0.1, 0.1))).status == QD.NewCell
end

@testset "GridArchive.add_batch!" begin
    a = GridArchive((2,), ((0, 1),))
    evs = [Evaluation(1.0, SVector(0.1)), Evaluation(2.0, SVector(0.2)), Evaluation(1.0, SVector(0.9))]
    rs = QD.add_batch!(a, [[1.0], [2.0], [3.0]], evs)
    # the results are against the empty archive, so both solutions in the first cell are new
    @test [r.status for r in rs] == [QD.NewCell, QD.NewCell, QD.NewCell]
    @test [r.improvement for r in rs] == [1.0, 2.0, 1.0]
    @test rs isa Vector{QD.AddResult{Float64}}
    @test sort(first.(collect(QD.elites(a)))) == [[2.0], [3.0]]          # the better one stays
    @test_throws "got 1 solutions but 3 evaluations" QD.add_batch!(a, [[1.0]], evs)
    @test isempty(QD.add_batch!(a, Vector{Float64}[], Evaluation{Float64,SVector{1,Float64}}[]))
    # of equally fit solutions for one cell, the first stays
    b = GridArchive((2,), ((0, 1),))
    QD.add_batch!(b, [[1.0], [2.0]], [Evaluation(1.0, SVector(0.1)), Evaluation(1.0, SVector(0.2))])
    @test first.(collect(QD.elites(b))) == [[1.0]]
end

@testset "fitness values an archive takes" begin
    ev(f) = Evaluation(f, SVector(0.1))
    for bad in (NaN, Inf, -Inf, NaN32)
        a = GridArchive((2,), ((0, 1),))
        QD.add!(a, [1.0], ev(1.0))
        @test_throws "fitness must be finite, got $bad" QD.add!(a, [2.0], ev(bad))
        @test_throws "fitness must be finite, got $bad" QD.add_batch!(a, [[2.0], [3.0]], [ev(2.0), ev(bad)])
        @test collect(QD.elites(a)) == [([1.0], ev(1.0))] && qd_score(a) == 1.0
    end
    # a fitness that overflows the fitness type of the archive
    small = GridArchive{Vector{Float64},Float32}((2,), ((0, 1),))
    @test_throws "fitness 1.0e300 is too large for the fitness type Float32 of the archive" QD.add!(small, [1.0], ev(1e300))
    # a NaN fitness throws although the descriptor is discarded
    rejecting = GridArchive((2,), ((0, 1),); outside=RejectOutside())
    @test_throws "fitness must be finite" QD.add!(rejecting, [1.0], Evaluation(NaN, SVector(5.0)))

    # unsigned fitness: a worse solution is rejected with zero improvement, not wrapped around
    for archive in (GridArchive{Vector{Float64},UInt8}((2,), ((0, 1),)),
                    UnstructuredArchive{Vector{Float64},UInt8}(1; capacity=4, novelty_threshold=0.5))
        @test QD.add!(archive, [1.0], Evaluation(UInt8(5), SVector(0.1))) == QD.AddResult(QD.NewCell, UInt8(5))
        @test QD.add!(archive, [2.0], Evaluation(UInt8(3), SVector(0.1))) == QD.AddResult(QD.Rejected, UInt8(0))
        @test QD.add!(archive, [3.0], Evaluation(UInt8(9), SVector(0.1))) == QD.AddResult(QD.Improved, UInt8(4))
        @test max_fitness(archive) == 9
    end
    unsigned = GridArchive{Vector{Float64},UInt8}((2,), ((0, 1),))
    QD.add!(unsigned, [1.0], Evaluation(UInt8(5), SVector(0.1)))
    @test QD.add_batch!(unsigned, [[2.0]], [Evaluation(UInt8(3), SVector(0.1))]) == [QD.AddResult(QD.Rejected, UInt8(0))]
end

# The results of a batch are against the archive before it; the archive after it
# is the one that adding the solutions one at a time gives.
@testset "add_batch! of the cell archives" begin
    ev(f, d) = Evaluation(f, d)
    makers = (() -> GridArchive((6, 6), ((0, 1), (0, 1))),
              () -> CVTArchive(20, ((0, 1), (0, 1)); samples=1_000, rng=Xoshiro(1)),
              () -> GridArchive{Vector{Float32},Float32}((6, 6), ((0, 1), (0, 1)); outside=RejectOutside()))
    for make in makers
        a = make()
        d = SVector(0.3, 0.3)
        QD.add!(a, [0.0], ev(5.0, d))
        # one occupied cell: below the occupant, above it, further above it, equal to it
        rs = QD.add_batch!(a, [[1.0], [2.0], [3.0], [4.0]], [ev(3.0, d), ev(7.0, d), ev(9.0, d), ev(5.0, d)])
        @test [r.status for r in rs] == [QD.Rejected, QD.Improved, QD.Improved, QD.Rejected]
        @test [r.improvement for r in rs] == [-2, 2, 4, 0]
        @test only(first.(collect(QD.elites(a)))) == [3.0]
        @test max_fitness(a) == 9 && qd_score(a) == 9
        # a tie among the solutions that enter goes to the first
        b = make()
        QD.add_batch!(b, [[1.0], [2.0]], [ev(4.0, d), ev(4.0, d)])
        @test only(first.(collect(QD.elites(b)))) == [1.0]
    end

    rng = Xoshiro(8)
    for make in makers, _ in 1:20
        one_by_one = make(); batched = make()
        for _ in 1:3
            xs = [[rand(rng)] for _ in 1:40]
            evs = [ev(-10 * rand(rng), rand(rng, SVector{2,Float64})) for _ in 1:40]
            before = deepcopy(batched)
            rs = QD.add_batch!(batched, xs, evs)
            foreach((x, e) -> QD.add!(one_by_one, x, e), xs, evs)
            # each result is that of the solution added alone to the archive before the batch
            @test all(k -> rs[k] == QD.add!(deepcopy(before), xs[k], evs[k]), 1:40)
            @test collect(QD.elites(batched)) == collect(QD.elites(one_by_one))
            @test qd_score(batched; offset=10) ≈ qd_score(one_by_one; offset=10) rtol = 1e-12
            @test max_fitness(batched) == max_fitness(one_by_one)
        end
    end

    # a descriptor the archive discards does not stop the others
    r = GridArchive((2, 2), ((0, 1), (0, 1)); outside=RejectOutside())
    rs = QD.add_batch!(r, [[1.0], [2.0]], [ev(1.0, SVector(5.0, 0.5)), ev(2.0, SVector(0.1, 0.1))])
    @test [x.status for x in rs] == [QD.Rejected, QD.NewCell] && rs[1].improvement == -Inf
    @test_throws "descriptor component 1 is 5.0" QD.add_batch!(GridArchive((2, 2), ((0, 1), (0, 1))),
                                                        [[1.0]], [ev(1.0, SVector(5.0, 0.5))])
    c = CategoricalArchive((1:3, [:a, :b]))
    rs = QD.add_batch!(c, [[1.0], [2.0], [3.0]], [ev(1.0, (1, :a)), ev(2.0, (1, :a)), ev(0.5, (3, :b))])
    @test [x.status for x in rs] == fill(QD.NewCell, 3) && length(c) == 2
    @test Set(first.(collect(QD.elites(c)))) == Set([[2.0], [3.0]])
end

@testset "GridArchive.sample_elites" begin
    a = GridArchive((10,), ((0, 1),))
    @test_throws "cannot sample elites from an empty archive" QD.sample_elites(Xoshiro(1), a, 3)
    QD.add!(a, [1.0], Evaluation(0.0, SVector(0.05)))
    QD.add!(a, [2.0], Evaluation(0.0, SVector(0.55)))
    parents = QD.sample_elites(Xoshiro(1), a, 200)
    @test parents isa Vector{Vector{Float64}}
    @test Set(parents) == Set([[1.0], [2.0]])
    @test QD.sample_elites(Xoshiro(2), a, 20) == QD.sample_elites(Xoshiro(2), a, 20)
end

@testset "metrics" begin
    a = GridArchive((4,), ((0, 1),))
    @test QD.coverage(a) == 0.0
    @test_throws "max_fitness is undefined for an archive with no elites" max_fitness(a)
    @test_throws "mean_fitness is undefined" mean_fitness(a)
    QD.add!(a, [0.0], Evaluation(-2.0, SVector(0.1)))
    QD.add!(a, [0.0], Evaluation(-4.0, SVector(0.6)))
    @test coverage(a) == 0.5
    @test max_fitness(a) == -2.0
    @test mean_fitness(a) == -3.0
    @test qd_score(a) == -6.0
    @test qd_score(a; offset=10) == 14.0
end

@testset "GridArchive outside policies" begin
    bounds = ((0, 1), (0, 1))
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))

    @test GridArchive((2, 2), bounds) isa GridArchive{Vector{Float64},Float64,2,ThrowOutside}
    @test_throws "outside the archive bounds" QD.cell_index(GridArchive((2, 2), bounds), [1.5, 0.5])

    clamped = GridArchive((2, 2), bounds; outside=ClampOutside())
    @test QD.cell_index(clamped, [-3.0, 0.2]) == QD.cell_index(clamped, [0.0, 0.2])
    @test QD.cell_index(clamped, [9.0, 9.0]) == 4
    @test QD.add!(clamped, [1.0], ev(1.0, (2.0, 2.0))).status == QD.NewCell
    @test_throws "descriptor component 2 is NaN and cannot be clamped" QD.cell_index(clamped, [0.5, NaN])

    rejecting = GridArchive{Vector{Float64},Float32}((2, 2), bounds; outside=RejectOutside())
    @test QD.cell_index(rejecting, [1.5, 0.5]) == 0
    @test QD.cell_index(rejecting, [0.5, NaN]) == 0
    r = QD.add!(rejecting, [1.0], ev(1.0, (1.5, 0.5)))
    @test r.status == QD.Rejected
    @test r.improvement === -Inf32    # below the improvement of any solution with a cell
    @test length(rejecting) == 0
    @test QD.add!(rejecting, [1.0], ev(1.0, (0.5, 0.5))).status == QD.NewCell

    # every constructor form takes the policy
    @test GridArchive{Vector{Float64}}((2, 2), bounds; outside=ClampOutside()) isa GridArchive{Vector{Float64},Float64,2,ClampOutside}
    @test GridArchive(Rastrigin(4); dims=(2, 2), outside=RejectOutside()) isa GridArchive{Vector{Float64},Float64,2,RejectOutside}
end

@testset "GridArchive bounds override" begin
    prob = Rastrigin(4)
    a = GridArchive(prob; dims=(2, 2), bounds=((-1, 1), (-2, 2)))
    @test a.lower == SVector(-1.0, -2.0)
    @test a.upper == SVector(1.0, 2.0)
    @test GridArchive(prob; dims=(2, 2)).upper == SVector(5.12, 5.12)
end

# The reference for the running metrics: the generic methods, which make a
# fresh pass over the elites.
generic_qd_score(a; offset) = invoke(qd_score, Tuple{QD.AbstractArchive}, a; offset)
generic_max_fitness(a) = invoke(max_fitness, Tuple{QD.AbstractArchive}, a)
generic_mean_fitness(a) = invoke(mean_fitness, Tuple{QD.AbstractArchive}, a)

@testset "GridArchive running metrics" begin
    rng = Xoshiro(11)
    a = GridArchive((20, 20), ((0, 1), (0, 1)))
    function check(a)
        @test coverage(a) == length(a) / QD.capacity(a)
        @test qd_score(a; offset=0) ≈ generic_qd_score(a; offset=0) rtol = 1e-14
        @test qd_score(a; offset=250.5) ≈ generic_qd_score(a; offset=250.5) rtol = 1e-14
        @test max_fitness(a) == generic_max_fitness(a)
        @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
    end

    # new cells, improvements and rejections, checking along the way
    for round in 1:20, _ in 1:500
        d = SVector(rand(rng), rand(rng))
        QD.add!(a, [0.0], Evaluation(-100 * rand(rng), d))
        round % 4 == 0 && QD.add!(a, [0.0], Evaluation(-100 * rand(rng) + 50, d))
    end
    check(a)
    @test length(a) > 100

    # a rejected solution changes nothing
    before = (qd_score(a), max_fitness(a), mean_fitness(a), length(a))
    elite_descriptor = first(QD.elites(a))[2].descriptor
    @test QD.add!(a, [0.0], Evaluation(-1e6, elite_descriptor)).status == QD.Rejected
    @test (qd_score(a), max_fitness(a), mean_fitness(a), length(a)) == before

    # a solution that beats the maximum raises it, and one that does not leaves it
    best = max_fitness(a)
    QD.add!(a, [0.0], Evaluation(best + 5, SVector(0.5, 0.5)))
    @test max_fitness(a) == best + 5
    check(a)

    # empty! resets the totals
    empty!(a)
    @test qd_score(a) == 0
    @test coverage(a) == 0
    @test_throws "max_fitness is undefined for an archive with no elites" max_fitness(a)
    @test_throws "mean_fitness is undefined for an archive with no elites" mean_fitness(a)
    QD.add!(a, [0.0], Evaluation(-7.0, SVector(0.1, 0.1)))
    @test max_fitness(a) == -7.0     # not the maximum from before empty!
    @test mean_fitness(a) == -7.0
    @test qd_score(a; offset=10) == 3.0

    # negative fitness: the maximum is the largest, not the one of smallest magnitude
    b = GridArchive((4,), ((0, 1),))
    QD.add!(b, [0.0], Evaluation(-5.0, SVector(0.1)))
    QD.add!(b, [0.0], Evaluation(-2.0, SVector(0.6)))
    QD.add!(b, [0.0], Evaluation(-9.0, SVector(0.9)))
    @test max_fitness(b) == -2.0
end

@testset "GridArchive running sum rounding" begin
    # One million improvements spread over 10 000 cells. The exact sum is taken
    # in BigFloat from the stored fitness values.
    function sum_error(F; updates=1_000_000)
        rng = Xoshiro(5)
        a = GridArchive{Vector{Float64},F}((100, 100), ((0, 1), (0, 1)))
        centers = [SVector((i - 0.5) / 100, (j - 0.5) / 100) for i in 1:100 for j in 1:100]
        naive = 0.0
        stored = Dict{Int,F}()
        for _ in 1:updates
            c = rand(rng, 1:10_000)
            old = get(stored, c, nothing)
            new = old === nothing ? F(-130 - 100 * rand(rng)) : F(old + 0.1 * rand(rng))
            r = QD.add!(a, [0.0], Evaluation(new, centers[c]))
            if r.status == QD.NewCell
                naive += Float64(new)
            elseif r.status == QD.Improved
                naive += Float64(new) - Float64(old)
            end
            r.status == QD.Rejected || (stored[c] = new)
        end
        exact = sum(BigFloat(f) for f in values(stored))
        relative(x) = Float64(abs(BigFloat(x) - exact) / abs(exact))
        return relative(mean_fitness(a) * length(a)), relative(naive)
    end

    running, naive = sum_error(Float64)
    @test running <= 4 * eps()
    @test naive > 100 * running    # a plain running sum is much less accurate
    running32, _ = sum_error(Float32)
    @test running32 <= 4 * eps()

    # An improvement adds the new fitness and subtracts the old one as two compensated
    # steps, so the total with its error term tracks the exact sum far below one rounding
    # even when the difference of the two is not representable: here an elite of magnitude
    # below 1 is replaced by one near 2^53, whose unit in the last place is 2.
    a = GridArchive((1000,), ((0, 1),))
    rng = Xoshiro(3)
    exact = BigFloat(0)
    for c in 1:1000
        d = SVector((c - 0.5) / 1000)
        QD.add!(a, [0.0], Evaluation(-rand(rng), d))
        new = 2.0^53 * (1 + rand(rng))
        QD.add!(a, [0.0], Evaluation(new, d))
        exact += new
    end
    two_terms = BigFloat(a.fitness_sum) + BigFloat(a.fitness_sum_error)
    @test abs(two_terms - exact) <= 1e-24 * exact    # the rounded differences would give about 1e-17
    @test qd_score(a) == Float64(exact)
end

@testset "GridArchive matches pyribs and QDax" begin
    # reference/grid.jl: the cells, statuses and improvements that pyribs's GridArchive
    # reports for a sequence of batches, the elites that it and QDax's MapElitesRepertoire
    # keep after each batch, and the metrics of both
    pyribs_status = Dict(0 => QD.Rejected, 1 => QD.Improved, 2 => QD.NewCell)
    c = grid_case
    a = GridArchive(c.dims, c.bounds)
    descriptor(i) = SVector{2}(c.descriptors[:, i])
    cells = CartesianIndices(c.dims)
    @test all(Tuple(cells[QD.cell_index(a, descriptor(i))]) == Tuple(c.grid_index[:, i] .+ 1)
              for i in eachindex(c.fitness))
    numbers(a) = sort([Int(only(x)) for (x, _) in QD.elites(a)])
    statuses, values = QD.AddStatus[], Float64[]
    for b in 1:length(c.fitness)÷c.batch
        ids = (b-1)*c.batch+1:b*c.batch
        rs = QD.add_batch!(a, [[Float64(i)] for i in ids], [Evaluation(c.fitness[i], descriptor(i)) for i in ids])
        append!(statuses, [r.status for r in rs])
        append!(values, [r.improvement for r in rs])
        @test numbers(a) == c.elites[b] == c.qdax_elites[b]
        # pyribs subtracts its offset from every fitness and QDax adds its offset
        elites, cov, score, normalized, best, average = c.stats[:, b]
        @test length(a) == elites && coverage(a) ≈ cov
        @test qd_score(a; offset=-c.offset) ≈ score rtol = 1e-12
        @test qd_score(a; offset=-c.offset) / QD.capacity(a) ≈ normalized rtol = 1e-12
        @test max_fitness(a) == best
        @test mean_fitness(a) ≈ average rtol = 1e-12
        qdax_score, qdax_coverage, qdax_best = c.qdax_metrics[:, b]
        @test qd_score(a; offset=-c.offset) ≈ qdax_score rtol = 1e-12
        @test 100 * coverage(a) ≈ qdax_coverage && max_fitness(a) == qdax_best
    end
    @test statuses == [pyribs_status[s] for s in c.statuses]
    @test values ≈ c.values rtol = 1e-12
end
