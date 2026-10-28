include("reference/cqd.jl")

# An archive that stores every solution it is given, at its own descriptor.
function all_elites(fitness, descriptors)
    archive = UnstructuredArchive(length(first(descriptors)); capacity=length(fitness), novelty_threshold=0.0)
    for (f, d) in zip(fitness, descriptors)
        QD.add!(archive, [0.0], Evaluation(f, SVector{length(d)}(d)))
    end
    @assert length(archive) == length(fitness)
    return archive
end

@testset "cqd_score by hand" begin
    a = GridArchive((4,), ((0, 1),))
    for (f, d) in ((3.0, 0.1), (1.0, 0.4), (2.0, 0.9))
        QD.add!(a, [0.0], Evaluation(f, SVector(d)))
    end
    # target 0.5, fitness over 3 and distances 0.4, 0.1, 0.4 over the diagonal 1:
    # θ = 0 recommends the best fitness, 1.0; θ = 0.5 still the first, 1 - 0.2; θ = 1, 1 - 0.4
    r = cqd_score(a; targets=[[0.5]], fitness_range=(0, 3), penalties=3)
    @test r isa QD.CQDScore
    @test r.penalties == [0.0, 0.5, 1.0]
    @test r.per_penalty ≈ [1.0, 0.8, 0.6]
    @test r.score ≈ 2.4
    @test r.n_targets == 1

    # a target at an elite: with a high penalty the elite there wins
    r = cqd_score(a; targets=[[0.4]], fitness_range=(0, 3), penalties=[5.0])
    @test r.per_penalty ≈ [1 / 3]

    # the fitness is divided by the range, not shifted by its minimum (paper and pyribs)
    r1 = cqd_score(a; targets=[[0.5]], fitness_range=(0, 3), penalties=[0])
    r2 = cqd_score(a; targets=[[0.5]], fitness_range=(-3, 0), penalties=[0])
    @test r1.score == r2.score == 1.0

    # one weight is 0, as numpy's linspace(0, 1, 1)
    @test cqd_score(a; targets=[[0.5]], fitness_range=(0, 3), penalties=1).penalties == [0.0]
    # distance_max scales the penalty
    @test cqd_score(a; targets=[[0.5]], fitness_range=(0, 3), penalties=[1], distance_max=2).per_penalty ≈ [0.8]
end

@testset "cqd_score properties" begin
    rng = Xoshiro(4)
    fitness = randn(rng, 60)
    descriptors = [2 .* rand(rng, 2) .- 1 for _ in 1:60]
    a = all_elites(fitness, descriptors)
    targets = [2 .* rand(rng, 2) .- 1 for _ in 1:20]
    r = cqd_score(a; targets, fitness_range=(-3, 3), penalties=7, distance_max=2sqrt(2))
    # a higher penalty can only lower each ω
    @test issorted(r.per_penalty; rev=true)
    # with no penalty every target gets the best elite
    @test r.per_penalty[1] ≈ maximum(fitness) / 6
    @test r.score ≈ r.n_targets * sum(r.per_penalty)
    # the order of the targets and of the elites does not matter
    @test cqd_score(a; targets=reverse(targets), fitness_range=(-3, 3), penalties=7, distance_max=2sqrt(2)).score ≈ r.score
    b = all_elites(reverse(fitness), reverse(descriptors))
    @test cqd_score(b; targets, fitness_range=(-3, 3), penalties=7, distance_max=2sqrt(2)).score ≈ r.score
    # nor the kind of archive holding the same elites
    grid = GridArchive((100, 100), ((-1, 1), (-1, 1)))
    QD.add_batch!(grid, [[0.0] for _ in fitness], [Evaluation(f, SVector{2}(d)) for (f, d) in zip(fitness, descriptors)])
    kept = [(ev.fitness, Vector(ev.descriptor)) for (_, ev) in QD.elites(grid)]
    @test cqd_score(grid; targets, fitness_range=(-3, 3), penalties=7).score ≈
          cqd_score(all_elites(first.(kept), last.(kept)); targets, fitness_range=(-3, 3), penalties=7, distance_max=2sqrt(2)).score
    # a wrapper is measured by its own elites, as qd_score
    t = ThresholdArchive(GridArchive((100, 100), ((-1, 1), (-1, 1))))
    QD.add_batch!(t, [[0.0] for _ in fitness], [Evaluation(f, SVector{2}(d)) for (f, d) in zip(fitness, descriptors)])
    @test cqd_score(t; targets, fitness_range=(-3, 3), penalties=7).score ≈ cqd_score(grid; targets, fitness_range=(-3, 3), penalties=7).score
    # Float32 fitness
    g32 = GridArchive{Vector{Float64},Float32}((4,), ((0, 1),))
    QD.add!(g32, [0.0], Evaluation(3.0f0, SVector(0.1)))
    @test cqd_score(g32; targets=[[0.1]], fitness_range=(0, 3), penalties=[0]).score ≈ 1
    @inferred cqd_score(a; targets, fitness_range=(-3, 3), penalties=7, distance_max=2sqrt(2))
end

@testset "cqd_score with random targets" begin
    a = GridArchive((10, 10), ((0, 1), (-2, 2)))
    QD.add!(a, [0.0], Evaluation(1.0, SVector(0.5, 0.0)))
    QD.add!(a, [0.0], Evaluation(0.2, SVector(0.9, 1.5)))
    r = cqd_score(Xoshiro(3), a; n_targets=50, fitness_range=(0, 1))
    @test r.n_targets == 50
    # the targets are drawn uniformly within the bounds, with the generator given
    rng = Xoshiro(3)
    targets = [SVector(0.0, -2.0) .+ rand(rng, SVector{2,Float64}) .* SVector(1.0, 4.0) for _ in 1:50]
    @test cqd_score(a; targets, fitness_range=(0, 1)).score == r.score
    @test cqd_score(Xoshiro(3), a; n_targets=50, fitness_range=(0, 1)).score == r.score
    @test cqd_score(Xoshiro(4), a; n_targets=50, fitness_range=(0, 1)).score != r.score
    # other bounds
    s = cqd_score(Xoshiro(3), a; n_targets=50, fitness_range=(0, 1), bounds=((0.5, 0.5), (0, 0)))
    @test s.per_penalty[end] ≈ 1.0                       # every target is at the best elite
    u = UnstructuredArchive(1; capacity=4, novelty_threshold=0.0)
    QD.add!(u, [0.0], Evaluation(1.0, SVector(0.3)))
    @test cqd_score(Xoshiro(1), u; n_targets=3, fitness_range=(0, 1), distance_max=1, bounds=((0, 1),)).n_targets == 3
end

@testset "cqd_score errors" begin
    a = GridArchive((4,), ((0, 1),))
    @test_throws "cqd_score is undefined for an archive with no elites" cqd_score(a; targets=[[0.5]], fitness_range=(0, 1))
    QD.add!(a, [0.0], Evaluation(1.0, SVector(0.3)))
    @test_throws "a target has 2 dimensions but the archive has 1" cqd_score(a; targets=[[0.5, 0.5]], fitness_range=(0, 1))
    @test_throws "targets must be finite" cqd_score(a; targets=[[NaN]], fitness_range=(0, 1))
    @test_throws "targets must hold at least one descriptor point" cqd_score(a; targets=Vector{Float64}[], fitness_range=(0, 1))
    @test_throws "fitness_range must be (f_min, f_max) with finite f_min < f_max, got (1, 1)" cqd_score(a; targets=[[0.5]], fitness_range=(1, 1))
    @test_throws "fitness_range must be" cqd_score(a; targets=[[0.5]], fitness_range=(0, Inf))
    @test_throws "penalties must be at least 1, got 0" cqd_score(a; targets=[[0.5]], fitness_range=(0, 1), penalties=0)
    @test_throws "penalties must hold at least one weight" cqd_score(a; targets=[[0.5]], fitness_range=(0, 1), penalties=Float64[])
    @test_throws "penalties must be finite" cqd_score(a; targets=[[0.5]], fitness_range=(0, 1), penalties=[Inf])
    @test_throws "distance_max must be finite and greater than 0, got 0" cqd_score(a; targets=[[0.5]], fitness_range=(0, 1), distance_max=0)
    @test_throws "n_targets must be at least 1, got 0" cqd_score(Xoshiro(1), a; n_targets=0, fitness_range=(0, 1))
    @test_throws "bounds has 2 entries but the archive has 1" cqd_score(Xoshiro(1), a; n_targets=2, fitness_range=(0, 1), bounds=((0, 1), (0, 1)))
    @test_throws "bounds must be finite (lo, hi) pairs with lo <= hi" cqd_score(Xoshiro(1), a; n_targets=2, fitness_range=(0, 1), bounds=((1, 0),))

    u = UnstructuredArchive(1; capacity=4, novelty_threshold=0.0)
    QD.add!(u, [0.0], Evaluation(1.0, SVector(0.3)))
    @test_throws "UnstructuredArchive does not report the bounds of its descriptor space; pass distance_max" cqd_score(u; targets=[[0.0]], fitness_range=(0, 1))
    @test_throws "pass bounds" cqd_score(Xoshiro(1), u; n_targets=3, fitness_range=(0, 1), distance_max=1)
    c = CategoricalArchive((1:3,))
    @test_throws "cqd_score measures distances between descriptors, which the descriptors of a CategoricalArchive do not have" cqd_score(c; targets=[[1]], fitness_range=(0, 1))
    bi = BiSphere(2)
    p = ParetoGridArchive(bi; dims=(3, 3), reference_point=QD.reference_point(bi))
    QD.add!(p, [0.0, 0.0], QD.evaluate(bi, [0.0, 0.0]))
    @test_throws "cqd_score needs a single real fitness" cqd_score(p; targets=[[0.0, 0.0]], fitness_range=(0, 1), distance_max=1)
end

@testset "cqd_score matches pyribs" begin
    # reference/cqd.jl: the scores of pyribs's cqd_score for the elites of an archive
    for case in cqd_cases
        N = case.dims
        E = length(case.fitness)
        a = all_elites(case.fitness, [case.descriptors[(e-1)*N+1:e*N] for e in 1:E])
        T = case.num_targets
        for (it, expected) in enumerate(case.scores)
            targets = [case.targets[((it-1)*T+t-1)*N+1:((it-1)*T+t)*N] for t in 1:T]
            r = cqd_score(a; targets, fitness_range=case.fitness_range, penalties=case.penalties,
                          distance_max=case.distance_max)
            @test r.score ≈ expected rtol = 1e-12
        end
    end
end
