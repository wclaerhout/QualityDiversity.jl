using LinearAlgebra: diag, eigvals, Symmetric

# Minimizes `f` with the first half of the ranked batch as parents; returns the best value.
function minimize!(f, es, x0, rng; generations)
    QD.reset!(es, x0)
    best = Inf
    for _ in 1:generations
        x = QD.sample!(rng, es)
        values = [f(view(x, :, k)) for k in 1:es.batch_size]
        order = sortperm(values)
        best = min(best, values[order[1]])
        QD.update!(es, order, es.batch_size ÷ 2)
        QD.should_stop(es, -values[order]) && break
    end
    return best
end

@testset "CMAES construction" begin
    es = QD.CMAES(10; sigma0=0.5)
    @test es.dim == 10 && es.batch_size == 4 + floor(Int, 3 * log(10)) && es.sigma0 == 0.5
    @test QD.CMAES(3; sigma0=1, batch_size=8).batch_size == 8
    @test QD.CMAES(2; sigma0=1, lower=0, upper=[1, 2]).upper == [1.0, 2.0]
    @test_throws "dim must be at least 1, got 0" QD.CMAES(0; sigma0=1)
    @test_throws "sigma0 must be positive and finite, got 0" QD.CMAES(3; sigma0=0)
    @test_throws "sigma0 must be positive and finite, got Inf" QD.CMAES(3; sigma0=Inf)
    @test_throws "batch_size must be at least 2 so that there is a parent, got 1" QD.CMAES(3; sigma0=1, batch_size=1)
    @test_throws "lower has 2 entries but the dimension is 3" QD.CMAES(3; sigma0=1, lower=[0, 0])
    @test_throws "lower[2] is 5.0 but must be less than upper[2], which is 1.0" QD.CMAES(2; sigma0=1, lower=[0, 5], upper=[1, 1])
    @test_throws "upper must not contain NaN" QD.CMAES(2; sigma0=1, upper=[1, NaN])
    @test_throws "x0 has 2 entries but the dimension is 3" QD.reset!(QD.CMAES(3; sigma0=1), [1.0, 2.0])
    @test_throws "x0 must be finite" QD.reset!(QD.CMAES(2; sigma0=1), [1.0, NaN])
end

@testset "CMAES recombination weights" begin
    # for 10 solutions and 5 parents the effective selection mass is about 3.17
    weights, mueff, cc, cs, c1, cmu, damps = QD.strategy_constants(10, 5)
    @test sum(weights) ≈ 1 && issorted(weights; rev=true) && all(>(0), weights)
    @test weights[1] ≈ (log(5.5) - log(1)) / sum(log(5.5) - log(i) for i in 1:5)
    @test mueff ≈ 3.17 atol = 0.01
    @test 0 < c1 < 1 && 0 < cmu < 1 && c1 + cmu < 1 && 0 < cc < 1 && 0 < cs < 1 && damps >= 1
    # one parent: all the weight on it
    w1, m1, = QD.strategy_constants(10, 1)
    @test w1 == [1.0] && m1 ≈ 1
end

@testset "CMAES sampling" begin
    es = QD.CMAES(4; sigma0=0.3, batch_size=6)
    QD.reset!(es, [1.0, 2.0, 3.0, 4.0])
    x = copy(QD.sample!(Xoshiro(1), es))
    @test QD.sample!(Xoshiro(1), es) === es.x && size(x) == (4, 6)
    @test es.y ≈ (x .- es.mean) ./ es.sigma
    @test x != QD.sample!(Xoshiro(2), es)
    @test copy(QD.sample!(Xoshiro(3), es)) == QD.sample!(Xoshiro(3), es)

    # mean and spread of a large batch around the start
    big = QD.CMAES(3; sigma0=0.5, batch_size=20_000)
    QD.reset!(big, [10.0, -5.0, 0.0])
    s = QD.sample!(Xoshiro(4), big)
    @test vec(sum(s; dims=2)) ./ 20_000 ≈ [10.0, -5.0, 0.0] atol = 0.02
    @test [sqrt(sum(abs2, s[i, :] .- big.mean[i]) / 20_000) for i in 1:3] ≈ fill(0.5, 3) atol = 0.02

    # samples stay inside the bounds
    bounded = QD.CMAES(3; sigma0=2.0, batch_size=200, lower=0.0, upper=[1.0, 2.0, 3.0])
    QD.reset!(bounded, [0.5, 1.0, 1.5])
    xb = QD.sample!(Xoshiro(5), bounded)
    @test all(0 .<= xb .<= [1.0, 2.0, 3.0])
    # a distribution that does not reach the bounds puts its draws on the nearest point
    # inside, and the update sees the steps to it
    stuck = QD.CMAES(2; sigma0=0.01, lower=10.0, upper=11.0)
    QD.reset!(stuck, [0.0, 0.0])
    xs = QD.sample!(Xoshiro(1), stuck)
    @test all(==(10.0), xs)
    @test stuck.y ≈ (xs .- stuck.mean) ./ stuck.sigma
end

@testset "CMAES converges on the sphere" begin
    n = 10
    evaluations = Int[]
    for seed in 1:5
        es = QD.CMAES(n; sigma0=1.0)
        QD.reset!(es, fill(3.0, n))
        rng = Xoshiro(seed)
        while true
            x = QD.sample!(rng, es)
            values = [sum(abs2, view(x, :, k)) for k in 1:es.batch_size]
            order = sortperm(values)
            QD.update!(es, order, es.batch_size ÷ 2)
            minimum(values) < 1e-8 && break
            es.count_eval > 20_000 && break
        end
        push!(evaluations, es.count_eval)
    end
    # about 1500 evaluations with the default batch of 10
    @test all(<(2_500), evaluations) && all(>(800), evaluations)
end

@testset "CMAES learns the shape of an ellipsoid" begin
    n = 10
    a = [10.0^(6 * (i - 1) / (n - 1)) for i in 1:n]            # condition number 1e6
    es = QD.CMAES(n; sigma0=1.0)
    best = minimize!(x -> sum(a .* abs2.(x)), es, fill(1.0, n), Xoshiro(1); generations=20_000)
    @test best < 1e-10
    # the covariance matrix approaches the inverse of the Hessian, up to a factor
    QD.update_eigensystem!(es)
    ratio = diag(es.cov) .* a
    @test maximum(ratio) / minimum(ratio) < 4
    @test 1e5 < maximum(es.eigenvalues) / minimum(es.eigenvalues) < 1e7
    # the learned directions are close to the axes: off-diagonal terms are small
    scale = sqrt.(diag(es.cov))
    correlation = es.cov ./ (scale * scale')
    @test maximum(abs, correlation - I(n)) < 0.5
end

@testset "CMAES converges on Rosenbrock" begin
    rosenbrock(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:length(x)-1)
    es = QD.CMAES(5; sigma0=0.5)
    best = minimize!(rosenbrock, es, zeros(5), Xoshiro(1); generations=5_000)
    @test best < 1e-9
    @test es.mean ≈ ones(5) atol = 1e-3
end

@testset "CMAES update" begin
    es = QD.CMAES(3; sigma0=1.0, batch_size=6)
    QD.reset!(es, zeros(3))
    QD.sample!(Xoshiro(1), es)
    # no parents: only the count of evaluations advances
    QD.update!(es, 1:6, 0)
    @test es.count_eval == 6 && es.mean == zeros(3) && es.sigma == 1.0 && es.cov == [i == j for i in 1:3, j in 1:3]
    # one parent: the mean moves to that parent
    best = es.x[:, 4]
    QD.update!(es, [4, 1, 2, 3, 5, 6], 1)
    @test es.mean ≈ best && es.count_eval == 12
    # the covariance matrix stays symmetric and positive definite
    for g in 1:50
        QD.sample!(Xoshiro(g), es)
        QD.update!(es, randperm(Xoshiro(g), 6), 1 + g % 3)
    end
    @test es.cov ≈ es.cov' && all(>(0), eigvals(Symmetric(es.cov)))
    @test_throws "order has 5 entries but the batch has 6" QD.update!(es, 1:5, 2)
    @test_throws "parents must be between 0 and 6, got 7" QD.update!(es, 1:6, 7)
end

@testset "CMAES stopping criteria" begin
    es = QD.CMAES(3; sigma0=1.0)
    QD.reset!(es, zeros(3))
    @test !QD.should_stop(es, [3.0, 2.0, 1.0])
    @test QD.should_stop(es, [1.0, 1.0 + 1e-13])                 # flat
    @test !QD.should_stop(es, [1.0])
    es.sigma = 1e-12
    @test QD.should_stop(es, [3.0, 2.0])                          # the distribution has shrunk
    es.sigma = 1.0
    es.eigenvalues .= [1.0, 1.0, 1e-15]
    @test QD.should_stop(es, [3.0, 2.0])                          # ill-conditioned
    es.eigenvalues .= 1
    es.mean[1] = NaN
    @test QD.should_stop(es, [3.0, 2.0])                          # broken down
    # a covariance that is no longer finite has no eigendecomposition, and stops the search
    QD.reset!(es, zeros(3))
    es.cov[2, 2] = Inf
    QD.update_eigensystem!(es)
    @test all(isnan, es.eigenvalues)
    @test QD.should_stop(es, [3.0, 2.0])
end
