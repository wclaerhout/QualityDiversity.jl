@testset "QDProblem" begin
    prob = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]));
                     init=rng -> randn(rng, 2), bounds=((-3, 3), (-3, 3)))
    ev = QD.evaluate(prob, [1.0, 2.0])
    @test ev.fitness == -5.0
    @test ev.descriptor == SVector(1.0, 2.0)
    @test QD.descriptor_bounds(prob) == ((-3.0, 3.0), (-3.0, 3.0))
    @test length(QD.random_solution(Xoshiro(1), prob)) == 2
    @test QD.fitness_offset(prob) == 0

    bad = QDProblem(x -> (0.0, (1.0, 2.0, 3.0)); init=rng -> [0.0], bounds=((0, 1), (0, 1)))
    @test_throws "descriptor has 3 dimensions but the problem bounds have 2" QD.evaluate(bad, [0.0])
    @test_throws "each bound must be a (lo, hi) pair" QDProblem(identity; init=identity, bounds=((1, 2, 3),))
    named = QDProblem(x -> (0.0, (a=x[1], b=x[2])); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 1)))
    @test_throws "a descriptor must be a tuple or a vector of reals, got a @NamedTuple{a::Float64, b::Float64}" QD.evaluate(named, [0.5, 0.5])

    evs = QD.evaluate_batch(prob, [[1.0, 2.0], [0.0, 0.0]])
    @test [e.fitness for e in evs] == [-5.0, 0.0]

    # a descriptor returned as a vector infers like one returned as a tuple
    vectors = QDProblem(x -> (-sum(abs2, x), [x[1], x[2]]); init=rng -> randn(rng, 2), bounds=((-3, 3), (-3, 3)))
    @test (@inferred QD.evaluate(vectors, [1.0, 2.0])) == ev
    @test eltype(@inferred QD.evaluate_batch(vectors, [[1.0, 2.0]])) == Evaluation{Float64,SVector{2,Float64},Nothing}
    static = QDProblem(x -> (-sum(abs2, x), SVector(x[1], x[2])); init=rng -> randn(rng, 2), bounds=((-3, 3), (-3, 3)))
    @test (@inferred QD.evaluate(static, [1.0, 2.0])) == ev

    # device_evaluate keeps the phenotype and leaves the descriptor as the function returned
    # it; the length of the descriptor is checked from its type before a kernel runs
    labelled = QDProblem(x -> (-sum(abs2, x), (x[1], x[2]), :ring); init=rng -> randn(rng, 2), bounds=((-3, 3), (-3, 3)))
    @test QD.evaluate(labelled, [1.0, 2.0]) == Evaluation(-5.0, SVector(1.0, 2.0), :ring)
    @test (@inferred QD.device_evaluate(labelled, [1.0, 2.0])) === Evaluation(-5.0, (1.0, 2.0), :ring)
    @test (@inferred QD.device_evaluate(prob, [1.0, 2.0])) === Evaluation(-5.0, (1.0, 2.0))
    @test QD.device_evaluate(bad, [0.0]) === Evaluation(0.0, (1.0, 2.0, 3.0))
    scalar = QDProblem(x -> 1.0; init=rng -> [0.0], bounds=((0, 1),))
    @test_throws "must return a (fitness, descriptor) or (fitness, descriptor, phenotype) tuple" QD.evaluate(scalar, [0.0])
    @test_throws "must return a (fitness, descriptor) or (fitness, descriptor, phenotype) tuple" QD.device_evaluate(scalar, [0.0])
end

@testset "evaluate_batch on threads" begin
    # the results do not depend on the size of the batch or on the number of threads
    prob = QDProblem(x -> (sum(k -> sin(k * x[1]), 1:500), (x[1], x[2])); init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    xs = [QD.random_solution(Xoshiro(k), prob) for k in 1:300]
    @test QD.evaluate_batch(prob, xs) == [QD.evaluate(prob, x) for x in xs]
    @test QD.evaluate_batch(prob, xs[1:2]) == [QD.evaluate(prob, x) for x in xs[1:2]]
    @test isempty(QD.evaluate_batch(prob, empty(xs)))
    cheap = QDProblem(x -> (-sum(abs2, x), (x[1], x[2])); init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    @test QD.evaluate_batch(cheap, xs) == [QD.evaluate(cheap, x) for x in xs]
    # an error in the function comes out as it is, whichever thread raised it
    # (the first two solutions pass, so the rest of the batch runs on threads)
    failing = QDProblem(x -> (x[1] < 0.5 ? error("no good: " * string(x[1] > 0)) : sum(k -> sin(k * x[1]), 1:500),
                              (x[1], x[2]));
                        init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    passing = filter(x -> x[1] >= 0.5, xs)
    @test_throws ErrorException("no good: true") QD.evaluate_batch(failing, [passing[1:2]; xs])
    noisy = StochasticQDProblem((rng, x) -> (sum(k -> sin(k * x[1]), 1:500) + randn(rng), (x[1], x[2]));
                                init=rng -> rand(rng, 3), bounds=((0, 1), (0, 1)))
    @test QD.evaluate_batch(Xoshiro(1), noisy, xs) == QD.evaluate_batch(Xoshiro(1), noisy, xs)
    @test QD.evaluate_batch(Xoshiro(1), noisy, xs)[1:2] == QD.evaluate_batch(Xoshiro(1), noisy, xs[1:2])
end

@testset "Rastrigin" begin
    prob = Rastrigin(4)
    ev = QD.evaluate(prob, zeros(4))
    @test ev.fitness == 0.0
    @test ev.descriptor == SVector(0.0, 0.0)

    ev = QD.evaluate(prob, [1.0, 3.0, -2.0, 2.0])
    @test ev.descriptor == SVector(2.0, 0.0)
    @test ev.fitness ≈ -(40 + sum(x^2 - 10cos(2π * x) for x in [1.0, 3.0, -2.0, 2.0]))

    # solutions outside the box are clipped before scoring
    @test QD.evaluate(prob, fill(100.0, 4)) == QD.evaluate(prob, fill(5.12, 4))
    @test QD.evaluate(prob, fill(5.12, 4)).descriptor == SVector(5.12, 5.12)

    @test_throws "solution has 3 entries but Rastrigin has dimension 4" QD.evaluate(prob, zeros(3))
    @test_throws "at least 2 dimensions" Rastrigin(1)

    # the offset makes every fitness non-negative
    xs = range(-5.12, 5.12; length=100_001)
    @test all(10 + x^2 - 10cos(2π * x) <= 41 for x in xs)

    x = QD.random_solution(Xoshiro(1), prob)
    @test x isa Vector{Float64} && length(x) == 4 && all(abs.(x) .<= 5.12)
end

@testset "Sphere" begin
    prob = Sphere(4)
    shift = 0.4 * 5.12
    # the optimum has fitness 100 and a descriptor of the sums of the parameters
    ev = QD.evaluate(prob, fill(shift, 4))
    @test ev.fitness ≈ 100 && ev.descriptor ≈ SVector(2shift, 2shift)
    # the corner of the box farthest from the optimum has fitness 0
    @test QD.evaluate(prob, fill(-5.12, 4)).fitness ≈ 0 atol = 1e-12
    @test 0 < QD.evaluate(prob, zeros(4)).fitness < 100
    # outside the box the fitness is negative, and a parameter is mapped back by 5.12 / θ
    ev = QD.evaluate(prob, [10.24, 0.0, -10.24, 0.0])
    @test ev.fitness < 0 && ev.descriptor ≈ SVector(0.5, -0.5)
    # the descriptor of the largest sums reaches the bounds
    @test QD.evaluate(prob, fill(5.12, 4)).descriptor ≈ SVector(10.24, 10.24)
    @test QD.descriptor_bounds(prob) == ((-10.24, 10.24), (-10.24, 10.24))
    @test QD.descriptor_bounds(Sphere(20))[1] == (-51.2, 51.2)
    @test QD.evaluate(Sphere(5), fill(1.0, 5)).descriptor ≈ SVector(2.0, 3.0)      # halves of 2 and 3

    x = QD.random_solution(Xoshiro(1), prob)
    @test x isa Vector{Float64} && length(x) == 4 && all(abs.(x) .<= 5.12)
    @test QD.random_solution(Xoshiro(1), prob) == x
    @test QD.fitness_offset(prob) == 0
    @test_throws "solution has 3 entries but Sphere has dimension 4" QD.evaluate(prob, zeros(3))
    @test_throws "Sphere needs at least 2 dimensions" Sphere(1)
end

@testset "evaluations in the number type of the solution" begin
    x = [0.3, -1.2, 2.0, 0.7]
    for prob in (Sphere(4), Rastrigin(4), BiSphere(4))
        ev = QD.evaluate(prob, x)
        @test ev.descriptor isa SVector{2,Float64}
        @test QD.evaluate(prob, Float32.(x)).descriptor isa SVector{2,Float64}
        precise = QD.evaluate(prob, big.(x))
        @test precise.descriptor isa SVector{2,BigFloat}
        @test Float64.(precise.descriptor) ≈ ev.descriptor
        @inferred QD.evaluate(prob, x)
    end
    prob = QDProblem(x -> (sum(x), (x[1], 2)); init=rng -> rand(rng, 2), bounds=((0, 1), (0, 3)))
    @test QD.evaluate(prob, [0.5, 0.2]).descriptor === SVector(0.5, 2.0)
    @test QD.evaluate(prob, Float32[0.5, 0.2]).descriptor isa SVector{2,Float64}
    @test QD.evaluate(prob, big.([0.5, 0.2])).descriptor isa SVector{2,BigFloat}
    @inferred QD.evaluate(prob, [0.5, 0.2])
end
