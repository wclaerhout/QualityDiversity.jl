include("reference/sliding.jl")

offer!(a, f, d) = QD.add!(a, [Float64(f)], Evaluation(Float64(f), SVector{length(d),Float64}(d)))

@testset "SlidingBoundariesArchive construction" begin
    a = SlidingBoundariesArchive((4, 5), ((0, 1), (-1, 1)))
    @test a isa SlidingBoundariesArchive{Vector{Float64},Float64,2,ClampOutside}
    @test a isa QD.CellArchive{Vector{Float64},Float64,SVector{2,Float64},Nothing}
    @test QD.capacity(a) == 20
    @test length(a) == 0
    @test QD.boundaries(a) == ([0.0, 0.25, 0.5, 0.75, 1.0], [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    @test QD.boundaries(a) !== a.boundaries
    @test QD.descriptor_bounds(a) == ((0.0, 1.0), (-1.0, 1.0))
    @test SlidingBoundariesArchive{Int,Float32}((2,), ((0, 1),)) isa SlidingBoundariesArchive{Int,Float32,1}
    @test SlidingBoundariesArchive(Rastrigin(4); dims=(3, 3), outside=ThrowOutside(), phenotype=Int) isa
          SlidingBoundariesArchive{Vector{Float64},Float64,2,ThrowOutside,Int}

    @test_throws "dims has 2 entries but bounds has 1" SlidingBoundariesArchive((2, 2), ((0, 1),))
    @test_throws "dims[2] is 0" SlidingBoundariesArchive((2, 0), ((0, 1), (0, 1)))
    @test_throws "bounds[1] is (1.0, 1.0)" SlidingBoundariesArchive((2,), ((1, 1),))
    @test_throws "remap_frequency must be at least 1, got 0" SlidingBoundariesArchive((2,), ((0, 1),); remap_frequency=0)
    @test_throws "buffer_capacity must be at least 1, got 0" SlidingBoundariesArchive((2,), ((0, 1),); buffer_capacity=0)
end

@testset "SlidingBoundariesArchive cells before a remap" begin
    a = SlidingBoundariesArchive((4, 5), ((0, 1), (-1, 1)))
    g = GridArchive((4, 5), ((0, 1), (-1, 1)))
    for x in 0.05:0.1:1, y in -0.95:0.1:1
        @test QD.cell_index(a, SVector(x, y)) == QD.cell_index(g, SVector(x, y))
    end
    @test QD.cell_index(a, [1.0, 1.0]) == 20
    @test QD.cell_index(a, [0.25, -1.0]) == 2      # a value on a boundary is in the cell above it
    @test QD.cell_index(a, [-3.0, 7.0]) == QD.cell_index(a, [0.0, 1.0])

    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.cell_index(a, [0.0, 0.0, 0.0])
    @test_throws "descriptor component 2 is NaN, but a SlidingBoundariesArchive needs finite descriptors" QD.cell_index(a, [0.5, NaN])
    @test_throws "descriptor component 1 is Inf" QD.cell_index(a, [Inf, 0.0])

    t = SlidingBoundariesArchive((4,), ((0, 1),); outside=ThrowOutside())
    @test_throws "descriptor component 1 is 1.5, outside the archive bounds [0.0, 1.0]" QD.cell_index(t, [1.5])
    @test_throws "a SlidingBoundariesArchive cannot use RejectOutside()" SlidingBoundariesArchive((4,), ((0, 1),); outside=RejectOutside())
end

@testset "SlidingBoundariesArchive remap" begin
    a = SlidingBoundariesArchive((4,), ((0, 1),); remap_frequency=5, buffer_capacity=5)
    for (f, d) in zip(1:4, [0.01, 0.02, 0.03, 0.04])
        offer!(a, f, (d,))
    end
    @test length(a) == 1                           # all four share the first cell
    @test QD.boundaries(a) == ([0.0, 0.25, 0.5, 0.75, 1.0],)
    @test offer!(a, 5, (0.9,)).status == QD.NewCell   # the result is that of the archive before the remap
    # boundary j is value ⌊(j-1)·5/4⌋+1 of the five sorted values, the last the largest
    @test QD.boundaries(a) == ([0.01, 0.02, 0.03, 0.04, 0.9],)
    @test QD.descriptor_bounds(a) == ((0.01, 0.9),)
    @test length(a) == 4
    stored = Dict(only(ev.descriptor) => only(x) for (x, ev) in QD.elites(a))
    @test stored == Dict(0.01 => 1.0, 0.02 => 2.0, 0.03 => 3.0, 0.9 => 5.0)   # 0.04 and 0.9 share the last cell
    @test max_fitness(a) == 5
    @test qd_score(a) == 11
    @test sort!([QD.cell_index(a, ev.descriptor) for (_, ev) in QD.elites(a)]) == 1:4

    # four more solutions do not remap; the fifth does, with the last five offered
    for (f, d) in zip(6:9, [0.5, 0.6, 0.7, 0.8])
        offer!(a, f, (d,))
    end
    @test QD.boundaries(a) == ([0.01, 0.02, 0.03, 0.04, 0.9],)
    offer!(a, 0, (0.95,))
    @test QD.boundaries(a) == ([0.5, 0.6, 0.7, 0.8, 0.95],)
    # old elites below the new boundaries are clamped into the first cell, and 0.8, 0.9
    # and 0.95 share the last; the best of each stays
    @test Dict(only(ev.descriptor) => only(x) for (x, ev) in QD.elites(a)) ==
          Dict(0.5 => 6.0, 0.6 => 7.0, 0.7 => 8.0, 0.8 => 9.0)
    @test max_fitness(a) == 9
end

@testset "SlidingBoundariesArchive repeated values" begin
    a = SlidingBoundariesArchive((4,), ((0, 1),); remap_frequency=4, buffer_capacity=4)
    for (f, d) in zip(1:4, [0.2, 0.2, 0.2, 0.7])
        offer!(a, f, (d,))
    end
    @test QD.boundaries(a) == ([0.2, 0.2, 0.2, 0.7, 0.7],)
    @test QD.cell_index(a, [0.2]) == 3             # the first two cells have width zero
    @test QD.cell_index(a, [0.1]) == 3
    @test QD.cell_index(a, [0.7]) == 4
    @test Dict(only(ev.descriptor) => only(x) for (x, ev) in QD.elites(a)) == Dict(0.2 => 3.0, 0.7 => 4.0)
end

@testset "SlidingBoundariesArchive outside the boundaries" begin
    # clamped solutions enter the buffer, so the boundaries grow toward them
    a = SlidingBoundariesArchive((2,), ((0, 1),); remap_frequency=4)
    @test offer!(a, 1, (0.5,)).status == QD.NewCell
    @test offer!(a, 2, (3.0,)).status == QD.Improved   # clamped into the cell of 0.5
    @test offer!(a, 3, (-2.0,)).status == QD.NewCell
    offer!(a, 4, (0.6,))
    @test QD.descriptor_bounds(a) == ((-2.0, 3.0),)
    @test QD.boundaries(a) == ([-2.0, 0.6, 3.0],)
    @test Dict(only(ev.descriptor) => only(x) for (x, ev) in QD.elites(a)) == Dict(-2.0 => 3.0, 0.6 => 4.0)

    # a non-finite component after one outside the boundaries throws before the archive changes
    for outside in (ClampOutside(), ThrowOutside())
        c = SlidingBoundariesArchive((2, 2), ((0, 1), (0, 1)); outside, remap_frequency=4)
        for (f, d) in zip(1:3, [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9)])
            offer!(c, f, d)
        end
        @test_throws ArgumentError offer!(c, 4, (2.0, NaN))
        @test length(c) == 3
        @test c.offered == 3
    end

    # the archive does not change when a descriptor throws
    t = SlidingBoundariesArchive((2,), ((0, 1),); outside=ThrowOutside(), remap_frequency=1)
    @test_throws "outside the archive bounds" offer!(t, 1, (2.0,))
    @test t.offered == 0
    @test QD.boundaries(t) == ([0.0, 0.5, 1.0],)
    @test_throws "needs finite descriptors" QD.add_batch!(t, [[1.0], [2.0]],
                                                         [Evaluation(1.0, SVector(0.5)), Evaluation(2.0, SVector(NaN))])
    @test length(t) == 0
    @test t.offered == 0
end

@testset "SlidingBoundariesArchive batches" begin
    xs = [[Float64(f)] for f in 1:7]
    evs = [Evaluation(Float64(f), SVector(d)) for (f, d) in zip(1:7, [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.9])]

    # no remap in the middle of a batch: every result is against the archive before it
    a = SlidingBoundariesArchive((4,), ((0, 1),); remap_frequency=3)
    rs = QD.add_batch!(a, xs, evs)
    @test [r.status for r in rs] == [QD.NewCell, QD.NewCell, QD.NewCell, QD.NewCell, QD.NewCell, QD.NewCell, QD.NewCell]
    @test a.offered == 7
    # the count passed 3 and 6, but the archive remapped once, after the batch, on all seven
    @test QD.boundaries(a) == ([0.1, 0.1, 0.3, 0.5, 0.9],)
    @test Dict(only(ev.descriptor) => only(x) for (x, ev) in QD.elites(a)) ==
          Dict(0.1 => 2.0, 0.3 => 4.0, 0.9 => 7.0)

    # a batch that passes no multiple does not remap
    b = SlidingBoundariesArchive((4,), ((0, 1),); remap_frequency=10)
    QD.add_batch!(b, xs, evs)
    @test QD.boundaries(b) == ([0.0, 0.25, 0.5, 0.75, 1.0],)
    QD.add_batch!(b, xs[1:3], evs[1:3])
    @test QD.boundaries(b) != ([0.0, 0.25, 0.5, 0.75, 1.0],)

    # a full buffer keeps the newest solutions
    c = SlidingBoundariesArchive((2,), ((0, 1),); remap_frequency=7, buffer_capacity=3)
    QD.add_batch!(c, xs, evs)
    @test QD.boundaries(c) == ([0.5, 0.5, 0.9],)
end

@testset "SlidingBoundariesArchive remap does not evict first" begin
    # 1 and 3 share the first of three regular cells, and 3 triggers the remap: it must
    # not evict 1 beforehand, since 1 is no longer in the buffer and the remap puts the two
    # in different cells
    a = SlidingBoundariesArchive{Int}((3,), ((0, 1),); remap_frequency=3, buffer_capacity=2)
    rs = [QD.add!(a, id, Evaluation(f, SVector(d))) for (id, f, d) in ((1, 1.0, 0.30), (2, 0.0, 0.20), (3, 2.0, 0.05))]
    @test [r.status for r in rs] == [QD.NewCell, QD.Rejected, QD.Improved]
    @test QD.boundaries(a) == ([0.05, 0.05, 0.2, 0.2],)
    @test sort!([x for (x, _) in QD.elites(a)]) == [1, 3]              # pyribs keeps the same
end

@testset "SlidingBoundariesArchive matches pyribs" begin
    # reference/sliding.jl: the boundaries and elites of pyribs after each solution, added
    # one at a time; the initial regular grid differs from numpy's by rounding
    for case in sliding_cases
        N = length(case.dims)
        a = SlidingBoundariesArchive{Int}(case.dims, ntuple(_ -> (-1, 1), N); remap_frequency=case.remap_frequency,
                                          buffer_capacity=case.buffer_capacity)
        for i in eachindex(case.fitness)
            QD.add!(a, i, Evaluation(case.fitness[i], SVector{N}(case.descriptors[(i-1)*N+1:i*N])))
            @test reduce(vcat, QD.boundaries(a)) ≈ case.boundaries[i] atol = 1e-15
            @test sort!([x for (x, _) in QD.elites(a)]) == case.kept[i]
        end
    end
end

@testset "SlidingBoundariesArchive phenotypes and empty!" begin
    a = SlidingBoundariesArchive{Vector{Float64}}((3,), ((0, 1),); remap_frequency=2, phenotype=String)
    QD.add!(a, [1.0], Evaluation(1.0, SVector(0.2), "a"))
    QD.add!(a, [2.0], Evaluation(2.0, SVector(0.8), "b"))
    @test Set((only(x), ev.phenotype) for (x, ev) in QD.elites(a)) == Set([(1.0, "a"), (2.0, "b")])
    @test_throws "the evaluation has no phenotype" QD.add!(a, [3.0], Evaluation(3.0, SVector(0.5)))
    @test a.offered == 2

    empty!(a)
    @test length(a) == 0
    @test a.offered == 0
    @test QD.boundaries(a) == ([0.0, 1 / 3, 2 / 3, 1.0],)
    QD.add!(a, [1.0], Evaluation(1.0, SVector(0.2), "a"))
    @test a.offered == 1
    QD.add!(a, [2.0], Evaluation(2.0, SVector(0.2), "c"))
    @test QD.boundaries(a) == ([0.2, 0.2, 0.2, 0.2],)       # the buffer was emptied too
end

@testset "SlidingBoundariesArchive cannot be wrapped" begin
    a = SlidingBoundariesArchive((4, 4), ((0, 1), (0, 1)))
    @test_throws "the cells of a SlidingBoundariesArchive move when it remaps" ThresholdArchive(a)
    @test_throws "the cells of a SlidingBoundariesArchive move when it remaps" ThresholdArchive(a; learning_rate=0.5, threshold_min=0)
    @test_throws "the cells of a SlidingBoundariesArchive move when it remaps" SpreadArchive(a)
end

@testset "MAP-Elites with sliding boundaries" begin
    # descriptors bunched near the origin of a wide range
    prob = QDProblem(x -> (-sum(abs2, x), (x[1]^3, x[2]^3)); init=rng -> 2 .* rand(rng, 4) .- 1,
                     bounds=((-8, 8), (-8, 8)))
    alg = MAPElites(SlidingBoundariesArchive(prob; dims=(10, 10)); batch_size=32)
    result = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test result.archive isa SlidingBoundariesArchive
    grid = solve(prob, MAPElites(prob; dims=(10, 10), batch_size=32, outside=ClampOutside());
                 budget=5_000, rng=Xoshiro(1))
    @test coverage(result.archive) > coverage(grid.archive)
    again = solve(prob, alg; budget=5_000, rng=Xoshiro(1))
    @test collect(QD.elites(again.archive)) == collect(QD.elites(result.archive))
    @test QD.boundaries(again.archive) == QD.boundaries(result.archive)
    @test length(alg.archive) == 0                                 # init works on a copy
    @test alg.archive.offered == 0
end
