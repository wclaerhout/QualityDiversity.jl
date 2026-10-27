include("reference/cvt.jl")

using StaticArrays: SVector, setindex

brute_nearest(points, q) = argmin([sum(abs2, p - q) for p in points])

@testset "CentroidTree finds the nearest point" begin
    for n in 1:6, k in (1, 2, 3, 10, 100, 700)
        rng = Xoshiro(100n + k)
        points = [rand(rng, SVector{n,Float64}) for _ in 1:k]
        tree = QD.CentroidTree(points)
        @test all(1:300) do _
            q = 1.4 .* rand(rng, SVector{n,Float64}) .- 0.2     # also outside the points' box
            QD.nearest(tree, q) == brute_nearest(points, q)
        end
    end

    # points on a lattice, queried at the midpoints between them: many exact
    # ties, which go to the lowest index
    lattice = [SVector(x, y) for x in 0.0:0.25:1.0 for y in 0.0:0.25:1.0]
    shuffled = lattice[randperm(Xoshiro(1), length(lattice))]
    tree = QD.CentroidTree(shuffled)
    for q in (SVector(0.125, 0.125), SVector(0.375, 0.625), SVector(0.5, 0.125), SVector(0.625, 0.875))
        @test QD.nearest(tree, q) == brute_nearest(shuffled, q)
    end
    @test QD.nearest(QD.CentroidTree([SVector(0.3, 0.3)]), SVector(5.0, -2.0)) == 1
    @test @inferred(QD.nearest(tree, SVector(0.1, 0.1))) isa Int

    # more than 127 dimensions: the split axis is an Int, not an Int8. The points are
    # spread most along the last dimension, so the root splits along it.
    rng = Xoshiro(130)
    wide = [setindex(0.01 * rand(rng, SVector{130,Float64}), Float64(k), 130) for k in 1:5]
    tree = QD.CentroidTree(wide)
    @test tree.axis[3] == 130
    @test all(1:20) do _
        q = setindex(0.01 * rand(rng, SVector{130,Float64}), 5 * rand(rng), 130)
        QD.nearest(tree, q) == brute_nearest(wide, q)
    end
end

@testset "k-means" begin
    # on a line, the best 4 centroids for a uniform measure are the quarter midpoints
    rng = Xoshiro(1)
    samples = [SVector(rand(rng)) for _ in 1:40_000]
    # The centers come out near those: in one dimension k-means magnifies the
    # sampling noise of 40 000 points several times, so they are not exact.
    @test sort(first.(QD.kmeans(Xoshiro(2), samples, 4))) ≈ [0.125, 0.375, 0.625, 0.875] atol = 0.02
    tight = QD.kmeans(Xoshiro(2), samples, 4; tolerance=1e-10, max_iterations=500)
    @test sort(first.(tight)) ≈ [0.125, 0.375, 0.625, 0.875] atol = 0.01
    # a converged result is a fixed point: every center is the mean of the samples nearest to it
    tree = QD.CentroidTree(tight)
    labels = [QD.nearest(tree, x) for x in samples]
    means = [sum(samples[labels .== j]) / count(==(j), labels) for j in 1:4]
    @test maximum(abs.(first.(means) .- first.(tight))) < 1e-9

    # in the plane: all cells are used, and the quantization error is far below
    # that of the k-means++ seeds
    samples = [rand(rng, SVector{2,Float64}) for _ in 1:20_000]
    seeds = QD.kmeans_seeds(Xoshiro(3), QD.KMeansPlusPlus(), samples, 100)
    centers = QD.kmeans(Xoshiro(3), samples, 100)
    error(c) = let tree = QD.CentroidTree(c)
        sum(x -> sum(abs2, x - c[QD.nearest(tree, x)]), samples) / length(samples)
    end
    @test error(centers) < 0.85 * error(seeds)
    @test error(centers) < 1.2 * (1 / 100) / 6      # about one cell side^2 / 6 for a unit square
    tree = QD.CentroidTree(centers)
    @test length(unique(QD.nearest(tree, x) for x in samples)) == 100
    @test all(c -> all(0 .<= c .<= 1), centers)

    @test QD.kmeans(Xoshiro(3), samples, 100) == centers
    @test QD.kmeans(Xoshiro(4), samples, 100) != centers
    @test QD.kmeans(Xoshiro(3), samples, 1) ≈ [sum(samples) / length(samples)]

    # greedy k-means++ seeds are distinct samples, closer to the samples than k-means++ seeds;
    # after Lloyd's iterations the two end close together for so few cells
    greedy = QD.kmeans_seeds(Xoshiro(3), QD.GreedyKMeansPlusPlus(), samples, 100)
    @test length(greedy) == 100 && allunique(greedy) && all(in(Set(samples)), greedy)
    plain_errors = [error(QD.kmeans_seeds(Xoshiro(s), QD.KMeansPlusPlus(), samples, 100)) for s in 1:5]
    greedy_errors = [error(QD.kmeans_seeds(Xoshiro(s), QD.GreedyKMeansPlusPlus(), samples, 100)) for s in 1:5]
    @test sum(greedy_errors) < 0.9 * sum(plain_errors)
    plain = [error(QD.kmeans(Xoshiro(s), samples, 100)) for s in 1:5]
    greedy = [error(QD.kmeans(Xoshiro(s), samples, 100; seeding=QD.GreedyKMeansPlusPlus())) for s in 1:5]
    @test sum(greedy) ≈ sum(plain) rtol = 0.02
    @test QD.kmeans(Xoshiro(3), samples, 100; seeding=QD.KMeansPlusPlus()) == centers
    @test QD.kmeans(Xoshiro(3), samples, 100; seeding=QD.GreedyKMeansPlusPlus()) ==
          QD.kmeans(Xoshiro(3), samples, 100; seeding=QD.GreedyKMeansPlusPlus())
    @test QD.kmeans(Xoshiro(3), samples, 1; seeding=QD.GreedyKMeansPlusPlus()) ≈ [sum(samples) / length(samples)]
end

@testset "CVTArchive seeding" begin
    bounds = ((0, 1), (-1, 1))
    plain = CVTArchive(40, bounds; samples=4_000, rng=Xoshiro(1))
    @test QD.centroids(CVTArchive(40, bounds; samples=4_000, rng=Xoshiro(1), seeding=QD.KMeansPlusPlus())) ==
          QD.centroids(plain)
    greedy = CVTArchive(40, bounds; samples=4_000, rng=Xoshiro(1), seeding=QD.GreedyKMeansPlusPlus())
    @test QD.capacity(greedy) == 40 && QD.centroids(greedy) != QD.centroids(plain)
    @test all(c -> 0 <= c[1] <= 1 && -1 <= c[2] <= 1, QD.centroids(greedy))
    prob = Rastrigin(4)
    @test QD.centroids(CVTArchive(prob; cells=40, samples=4_000, rng=Xoshiro(1), seeding=QD.GreedyKMeansPlusPlus())) ==
          QD.centroids(CVTArchive(40, QD.descriptor_bounds(prob); samples=4_000, rng=Xoshiro(1),
                                  seeding=QD.GreedyKMeansPlusPlus()))
    @test_throws TypeError CVTArchive(40, bounds; samples=4_000, seeding=:greedy)
end

@testset "CVTArchive construction" begin
    bounds = ((0, 1), (-1, 1))
    a = CVTArchive(40, bounds; samples=4_000, rng=Xoshiro(1))
    @test a isa CVTArchive{Vector{Float64},Float64,2,ThrowOutside}
    @test a isa QD.CellArchive && a isa AbstractArchive{Vector{Float64},Float64,SVector{2,Float64}}
    @test QD.capacity(a) == 40 && length(a) == 0
    @test length(QD.centroids(a)) == 40
    @test all(c -> 0 <= c[1] <= 1 && -1 <= c[2] <= 1, QD.centroids(a))
    @test CVTArchive(40, bounds; samples=4_000, rng=Xoshiro(1)).centroids == a.centroids      # same seed
    @test CVTArchive(40, bounds; samples=4_000, rng=Xoshiro(2)).centroids != a.centroids

    # the parametric forms agree
    @test CVTArchive{Int}(5, bounds; samples=500, rng=Xoshiro(1)) isa CVTArchive{Int,Float64,2}
    @test CVTArchive{Int,Float32}(5, bounds; samples=500, rng=Xoshiro(1)) isa CVTArchive{Int,Float32,2}
    @test CVTArchive(Rastrigin(4); cells=7, samples=700, rng=Xoshiro(1)) isa CVTArchive{Vector{Float64},Float64,2}
    z = CVTArchive(Rastrigin(4); cells=7, samples=700, bounds=((-1, 1), (-1, 1)), rng=Xoshiro(1))
    @test all(c -> all(abs.(c) .<= 1), QD.centroids(z))
    # a descriptor with more than 127 dimensions, widest along the last one
    wide = CVTArchive(4, ntuple(i -> i == 130 ? (0, 1000) : (0, 1), 130); rng=Xoshiro(1), samples=40)
    @test wide isa CVTArchive{Vector{Float64},Float64,130} && QD.capacity(wide) == 4
    @test wide.tree.axis[2] == 130
    @test QD.add!(wide, [1.0], Evaluation(1.0, rand(Xoshiro(2), SVector{130,Float64}))).status == QD.NewCell

    # rng is a keyword with the default generator as its default, and max_iterations
    # reaches k-means from every form
    @test CVTArchive(7, bounds; samples=700) isa CVTArchive{Vector{Float64},Float64,2}
    @test QD.capacity(CVTArchive(Rastrigin(4); cells=7, samples=700)) == 7
    one_step = CVTArchive(Rastrigin(4); cells=7, samples=700, max_iterations=1, rng=Xoshiro(1))
    @test one_step.centroids != CVTArchive(Rastrigin(4); cells=7, samples=700, rng=Xoshiro(1)).centroids
    @test one_step.centroids == CVTArchive(7, QD.descriptor_bounds(Rastrigin(4)); samples=700, max_iterations=1, rng=Xoshiro(1)).centroids
    @test_throws MethodError CVTArchive(Xoshiro(1), 7, bounds)
    @test_throws MethodError CVTArchive(Xoshiro(1), Rastrigin(4); cells=7)

    # given centroids, as vectors or tuples of numbers, and rebuilt from another archive
    given = CVTArchive{Vector{Float64},Float64}([[0.1, 0.0], [0.9, 0.5], [0.5, -0.5]], bounds)
    @test QD.capacity(given) == 3
    @test QD.cell_index(given, [0.8, 0.4]) == 2
    rebuilt = CVTArchive(QD.centroids(a), bounds)
    @test all(QD.cell_index(rebuilt, d) == QD.cell_index(a, d) for d in eachcol(rand(Xoshiro(1), 2, 500)))

    @test_throws "cells must be at least 1, got 0" CVTArchive(0, bounds; rng=Xoshiro(1))
    @test_throws "samples (10) must be at least cells (20)" CVTArchive(20, bounds; samples=10, rng=Xoshiro(1))
    @test_throws "centroid 2 is (2.0, 0.0), outside the bounds" CVTArchive([[0.5, 0.0], [2.0, 0.0]], bounds)
    @test_throws "two centroids are identical, at (0.5, 0.0)" CVTArchive([[0.5, 0.0], [0.1, 0.1], [0.5, 0.0]], bounds)
    @test_throws "need at least one centroid" CVTArchive(Vector{Float64}[], bounds)
    @test_throws "centroid 1 has 3 entries but bounds has 2" CVTArchive([[0.5, 0.0, 0.0]], bounds)
    @test_throws "bounds[1] is (1.0, 1.0)" CVTArchive([[1.0, 0.0]], ((1, 1), (0, 1)))
    @test_throws "bounds must have at least one entry" CVTArchive(3, (); rng=Xoshiro(1))
end

@testset "CVTArchive.cell_index" begin
    rng = Xoshiro(5)
    bounds = ((0, 1), (-1, 1), (10, 20))
    a = CVTArchive(60, bounds; samples=6_000, rng)
    for _ in 1:500
        d = SVector(rand(rng), 2rand(rng) - 1, 10 + 10rand(rng))
        @test QD.cell_index(a, d) == brute_nearest(QD.centroids(a), d)
        @test QD.cell_index(a, collect(d)) == QD.cell_index(a, d)      # any vector type
    end
    @test_throws "descriptor has 2 dimensions but the archive has 3" QD.cell_index(a, [0.5, 0.0])
    @test_throws "descriptor component 3 is 25.0, outside the archive bounds [10.0, 20.0]" QD.cell_index(a, [0.5, 0.0, 25.0])
    @test_throws "descriptor component 1 is NaN" QD.cell_index(a, [NaN, 0.0, 15.0])

    clamped = CVTArchive(QD.centroids(a), bounds; outside=ClampOutside())
    @test QD.cell_index(clamped, [-5.0, 0.0, 15.0]) == QD.cell_index(clamped, [0.0, 0.0, 15.0])
    @test QD.cell_index(clamped, [3.0, 7.0, 99.0]) == QD.cell_index(clamped, [1.0, 1.0, 20.0])
    @test_throws "descriptor component 1 is NaN and cannot be clamped" QD.cell_index(clamped, [NaN, 0.0, 15.0])

    rejecting = CVTArchive{Vector{Float64},Float32}(QD.centroids(a), bounds; outside=RejectOutside())
    @test QD.cell_index(rejecting, [-5.0, 0.0, 15.0]) == 0
    @test QD.cell_index(rejecting, [0.5, NaN, 15.0]) == 0
    r = QD.add!(rejecting, [1.0], Evaluation(1.0, SVector(5.0, 0.0, 15.0)))
    @test r.status == QD.Rejected && r.improvement === -Inf32 && length(rejecting) == 0
end

@testset "CVTArchive stores and reports like any cell archive" begin
    rng = Xoshiro(2)
    a = CVTArchive(30, ((0, 1), (0, 1)); samples=3_000, rng)
    ev(f, d) = Evaluation(f, SVector{2,Float64}(d))
    @test QD.add!(a, [1.0], ev(-3.0, (0.1, 0.1))).status == QD.NewCell
    @test QD.add!(a, [2.0], ev(-1.0, (0.1, 0.1))).status == QD.Improved
    @test QD.add!(a, [3.0], ev(-2.0, (0.1, 0.1))).status == QD.Rejected
    @test only(first.(collect(QD.elites(a)))) == [2.0]

    for _ in 1:3000
        QD.add!(a, [0.0], Evaluation(-100 * rand(rng), SVector(rand(rng), rand(rng))))
    end
    @test length(a) == 30 && coverage(a) == 1.0
    @test qd_score(a; offset=100) ≈ generic_qd_score(a; offset=100) rtol = 1e-14
    @test max_fitness(a) == generic_max_fitness(a)
    @test mean_fitness(a) ≈ generic_mean_fitness(a) rtol = 1e-14
    empty!(a)
    @test length(a) == 0 && qd_score(a) == 0
    @test_throws "max_fitness is undefined" max_fitness(a)
end

@testset "MAPElites over a CVTArchive" begin
    prob = Rastrigin(6)
    archive = CVTArchive(prob; cells=100, samples=5_000, rng=Xoshiro(1))
    alg = MAPElites(archive; batch_size=50, variation=IsoLineVariation(0.05, 0.3))
    @test alg isa QDAlgorithm && alg.archive === archive
    result = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test result.n_evals == 6_000
    @test result.archive isa CVTArchive
    @test length(archive) == 0                     # the algorithm is not modified
    @test length(result.archive) > 60
    @test issorted(r.qd_score for r in result.history)
    @test solve(prob, alg; budget=6_000, rng=Xoshiro(1)).history == result.history
    @test MAPElites(archive).emitters[1].batch_size == 64

    # the archive and the problem must agree on the number of descriptor dimensions
    wrong = CVTArchive(10, ((0, 1),); samples=100, rng=Xoshiro(1))
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, MAPElites(wrong); budget=100)
end

@testset "nearest centroid ties go to the lowest index" begin
    # Centroids on an integer lattice, numbered in a shuffled order, and queries halfway
    # between them: distances are exact, so a query is often as near to centroids on both
    # sides of a split plane, and the k-d tree must find the one a brute-force search finds.
    lattice = [SVector(Float64(i), Float64(j)) for i in 0:5 for j in 0:5]
    for seed in 1:5
        points = lattice[randperm(Xoshiro(seed), length(lattice))]
        tree = QD.CentroidTree(points)
        for qx in 0:0.5:5, qy in 0:0.5:5
            q = SVector(qx, qy)
            distances = [sum(abs2, p - q) for p in points]
            @test QD.nearest(tree, q) == argmin(distances)
        end
    end
end

@testset "CVTArchive finds the cells of pyribs and QDax" begin
    # reference/cvt.jl: the cells of pyribs's CVTArchive, which clips a descriptor to the
    # bounds first, and of QDax's get_cells_indices, which does not, for fixed centroids
    c = cvt_case
    centroids = [SVector{3}(c.centroids[:, i]) for i in axes(c.centroids, 2)]
    descriptors = [SVector{3}(c.descriptors[:, i]) for i in axes(c.descriptors, 2)]
    clamped = CVTArchive(centroids, ((0, 1), (0, 1), (0, 1)); outside=ClampOutside())
    @test [QD.cell_index(clamped, d) for d in descriptors] == c.pyribs .+ 1
    wide = CVTArchive(centroids, ((-1, 2), (-1, 2), (-1, 2)))
    @test [QD.cell_index(wide, d) for d in descriptors] == c.qdax .+ 1
end
