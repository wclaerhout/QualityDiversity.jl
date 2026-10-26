using LinearAlgebra: norm, dot

include("reference/strategies.jl")
include("reference/pyribs_strategies.jl")

# Batch `X` (one solution per column) given as the flat vector of the reference file.
columns(flat, dim) = reshape(flat, dim, :)

# Sets the last batch of `es` to `X` as `sample!` would have left it.
function load_batch!(es, X)
    es.x .= X
    es.y .= (X .- es.mean) ./ es.sigma
    return es
end

strategies = (QD.CMAES, QD.SepCMAES, QD.LMMAES, QD.OpenAIES)

@testset "construction of $(nameof(S))" for S in strategies
    dim = 26
    es = S(dim; sigma0=0.5)
    @test es isa QD.AbstractEvolutionStrategy && es.dim == dim && size(es.x) == (dim, es.batch_size)
    @test es.batch_size >= 2 && es.sigma0 == 0.5
    @test S(dim; sigma0=1, batch_size=8).batch_size == 8
    @test_throws "sigma0 must be positive and finite, got 0" S(dim; sigma0=0)
    @test_throws "dim must be at least 1, got 0" S(0; sigma0=1)
    @test_throws "batch_size must be at least 2 so that there is a parent, got 1" S(dim; sigma0=1, batch_size=1)
    @test_throws "lower has 2 entries but the dimension is 26" S(dim; sigma0=1, lower=[0, 0])
    @test_throws "lower[3] is 5.0 but must be less than upper[3], which is 1.0" S(3; sigma0=1, lower=[0, 0, 5], upper=1)
    @test_throws "x0 has 2 entries but the dimension is 26" QD.reset!(es, [1.0, 2.0])
    @test_throws "x0 must be finite" QD.reset!(es, fill(NaN, dim))
    @test_throws UndefKeywordError S(dim)
end

@testset "sampling with $(nameof(S))" for S in strategies
    dim = 16
    es = S(dim; sigma0=0.3, batch_size=8)
    QD.reset!(es, collect(1.0:dim))
    x = copy(QD.sample!(Xoshiro(1), es))
    @test size(x) == (dim, 8) && QD.sample!(Xoshiro(1), es) === es.x && es.x == x
    @test x != QD.sample!(Xoshiro(2), es)
    @test all(isfinite, x)
    # centered on the start, with spread of the order of sigma0
    # (LMMAES needs a batch of at most half the dimension)
    bdim, bsize = S === QD.LMMAES ? (1_000, 400) : (12, 4_000)
    big = S(bdim; sigma0=0.3, batch_size=bsize)
    QD.reset!(big, collect(1.0:bdim))
    s = QD.sample!(Xoshiro(3), big)
    @test maximum(abs, vec(sum(s; dims=2)) ./ bsize .- collect(1.0:bdim)) < 0.08
    @test sqrt(sum(abs2, s .- collect(1.0:bdim)) / (bsize * bdim)) ≈ 0.3 rtol = 0.05

    # bounds: a solution outside is drawn again
    bounded = S(dim; sigma0=0.3, batch_size=S === QD.LMMAES ? 8 : 200, lower=0.0, upper=2.0)
    QD.reset!(bounded, fill(1.0, dim))
    @test all(0 .<= QD.sample!(Xoshiro(4), bounded) .<= 2)
    # out of reach: after the tries a draw moves to the nearest point inside
    narrow = S(dim; sigma0=1.0, batch_size=4, lower=0.0, upper=1e-9)
    QD.reset!(narrow, fill(5.0, dim))
    xs = QD.sample!(Xoshiro(1), narrow)
    @test all(==(1e-9), xs)
    hasproperty(narrow, :y) && @test narrow.y ≈ (xs .- narrow.mean) ./ narrow.sigma

    # OpenAIES keeps its point inside the bounds after a step, so that mirrored pairs keep
    # landing inside
    if S === QD.OpenAIES
        edge = S(3; sigma0=0.2, batch_size=6, lower=0.0, upper=1.0, learning_rate=0.5)
        QD.reset!(edge, [0.99, 0.5, 0.01])
        for g in 1:20
            x = QD.sample!(Xoshiro(g), edge)
            order = sortperm(vec(x[1, :] .- x[3, :]); rev=true)    # pushes toward (1, ·, 0)
            QD.update!(edge, order, 3)
        end
        @test all(0 .<= edge.theta .<= 1)
        @test edge.theta[1] == 1.0 && edge.theta[3] == 0.0
    end
end

@testset "update! with $(nameof(S))" for S in strategies
    dim = 16
    es = S(dim; sigma0=0.3, batch_size=8)
    QD.reset!(es, zeros(dim))
    QD.sample!(Xoshiro(1), es)
    @test_throws "order has 5 entries but the batch has 8" QD.update!(es, 1:5, 2)
    @test_throws "parents must be between 0 and 8, got 9" QD.update!(es, 1:8, 9)
    # the batch moves the distribution; the best solution of a one-parent update becomes the mean
    # (not for OpenAIES, which follows a gradient estimate)
    if S !== QD.OpenAIES
        before = copy(es.mean)
        QD.update!(es, randperm(Xoshiro(1), 8), 0)
        @test es.mean == before                       # no parent: no move
        QD.update!(es, [3, 1, 2, 4, 5, 6, 7, 8], 1)
        @test es.mean ≈ es.x[:, 3]
    end
    # a restart forgets the adaptation
    QD.reset!(es, ones(dim))
    S === QD.OpenAIES ? (@test es.theta == ones(dim) && es.steps == 0) :
        (@test es.mean == ones(dim) && es.sigma == es.sigma0)
end

@testset "$(nameof(S)) maximizes" for (S, f, gens) in ((QD.CMAES, x -> -sum(abs2, x), 120),
                                                        (QD.SepCMAES, x -> -sum(abs2, x), 150),
                                                        (QD.LMMAES, x -> -sum(abs2, x), 300),
                                                        (QD.OpenAIES, x -> -sum(abs2, x), 600))
    dim = 14
    es = S === QD.OpenAIES ? S(dim; sigma0=0.1, batch_size=30, learning_rate=0.02) :
         S(dim; sigma0=0.5, batch_size=S === QD.LMMAES ? 7 : 10)
    rng = Xoshiro(1)
    QD.reset!(es, fill(2.0, dim))
    start = f(fill(2.0, dim))
    best = -Inf
    for _ in 1:gens
        x = QD.sample!(rng, es)
        values = [f(view(x, :, k)) for k in 1:es.batch_size]
        order = sortperm(values; rev=true)
        best = max(best, values[order[1]])
        QD.update!(es, order, es.batch_size ÷ 2)
        QD.should_stop(es, values[order]) && break
    end
    @test best > start / 1000
end

@testset "SepCMAES follows the reference update" begin
    n = length(sep_x0)
    es = QD.SepCMAES(n; sigma0=sep_s0, batch_size=8)
    QD.reset!(es, sep_x0)
    for (g, flat) in enumerate(sep_X)
        load_batch!(es, columns(flat, n))
        QD.update!(es, sep_orders[g], sep_parents[g])
    end
    @test es.mean ≈ sep_mean rtol = 1e-10
    @test es.sigma ≈ sep_sigma rtol = 1e-10
    @test es.variances ≈ sep_var rtol = 1e-10

    # constants of pycma for 5 parents out of 10 in dimension 10
    weights, mueff, cc, cs, c1, cmu, damps = QD.sep_strategy_constants(10, 5)
    @test mueff ≈ 3.1672992814107035 && sum(weights) ≈ 1
    @test cc ≈ (1 + 1 / 10 + mueff / 10) / (sqrt(10) + 1 / 10 + 2 * mueff / 10)
    @test c1 ≈ 1 / (10 + 2 * sqrt(10) + mueff / 10) && 0 < cmu < 1 - c1
    # it learns faster than the full covariance matrix has to, and keeps variances positive
    @test c1 > QD.strategy_constants(10, 5)[5] && cmu > QD.strategy_constants(10, 5)[6]
    for g in 1:60
        QD.sample!(Xoshiro(g), es)
        QD.update!(es, randperm(Xoshiro(g), 8), 1 + g % 4)
    end
    @test all(>(0), es.variances) && all(isfinite, es.variances)
end

@testset "LMMAES follows the reference update" begin
    n = 12
    es = QD.LMMAES(n; sigma0=lm_s0, batch_size=6, n_vectors=3)
    @test es.c_sigma ≈ 2 * 6 / n && es.c_d ≈ [1 / n, 1 / (1.5n), 1 / (2.25n)] && es.c_c ≈ [6 / n, 6 / (4n), 6 / (16n)]
    QD.reset!(es, lm_x0)
    for g in eachindex(lm_Z)
        es.z .= columns(lm_Z[g], n)
        es.y .= columns(lm_Y[g], n)
        es.x .= es.mean .+ es.sigma .* es.y
        QD.update!(es, lm_orders[g], lm_parents[g])
    end
    @test es.mean ≈ lm_mean rtol = 1e-10
    @test es.sigma ≈ lm_sigma rtol = 1e-10
    @test es.path_sigma ≈ lm_ps rtol = 1e-10
    @test vec(es.vectors) ≈ lm_M rtol = 1e-10
    # pyribs's LMMAEvolutionStrategy, given the same batches, ends in the same state
    # (reference/pyribs_strategies.jl)
    @test es.mean ≈ pyribs_lm.mean rtol = 1e-10
    @test es.sigma ≈ pyribs_lm.sigma rtol = 1e-10
    @test es.path_sigma ≈ pyribs_lm.path_sigma rtol = 1e-10
    @test vec(es.vectors) ≈ pyribs_lm.vectors rtol = 1e-10
    @test es.generation == 3

    # the first batch is the plain normal draw, and later ones are stretched along the vectors
    fresh = QD.LMMAES(n; sigma0=1.0, batch_size=6, n_vectors=2)
    QD.reset!(fresh, zeros(n))
    QD.sample!(Xoshiro(1), fresh)
    @test fresh.y == fresh.z && fresh.x == fresh.y
    fresh.vectors[:, 1] .= 1 / sqrt(n)
    fresh.generation = 1
    QD.sample!(Xoshiro(1), fresh)
    z = fresh.z[:, 1]
    @test fresh.y[:, 1] ≈ (1 - fresh.c_d[1]) .* z .+ fresh.c_d[1] .* fill(1 / sqrt(n), n) .* (sum(z) / sqrt(n))

    # a batch of more than half the dimension makes c_sigma exceed 1
    @test_throws ("LMMAES needs a batch of at most half the dimension, but batch_size is 6 and the dimension 10: " *
                  "the learning rate of its step size, 2 batch_size / dim = 1.2, would exceed 1, so the step size " *
                  "would stop adapting. LMMAES is meant for a high dimension (at least 26 with the default batch); " *
                  "pass batch_size ≤ 5, or use CMAES or SepCMAES. See the help of QualityDiversity.LMMAES.") QD.LMMAES(10; sigma0=1, batch_size=6)
    @test QD.LMMAES(10; sigma0=1, batch_size=5).c_sigma == 1
    @test_throws "pass batch_size ≤ 2, or use CMAES" QD.LMMAES(4; sigma0=1, batch_size=3)
    @test_throws "(at least 26 with the default batch); use CMAES or SepCMAES" QD.LMMAES(3; sigma0=1, batch_size=2)
    # the default batch fits from 26 on
    @test_throws "batch_size is 13 and the dimension 25" QD.LMMAES(25; sigma0=1)
    @test all(dim -> QD.LMMAES(dim; sigma0=1).c_sigma <= 1, 26:2_000)
    @test_throws "n_vectors must be between 1 and the dimension (26), got 27" QD.LMMAES(26; sigma0=1, n_vectors=27)
    @test QD.LMMAES(26; sigma0=1).n_vectors == QD.LMMAES(26; sigma0=1).batch_size
end

@testset "OpenAIES follows the reference update" begin
    n = length(oa_x0)
    es = QD.OpenAIES(n; sigma0=0.3, batch_size=8, learning_rate=0.05)
    QD.reset!(es, oa_x0)
    for g in eachindex(oa_E)
        E = columns(oa_E[g], n)
        es.noise .= hcat(E, -E)
        QD.update!(es, oa_orders[g], 4)
    end
    @test es.theta ≈ oa_theta rtol = 1e-10
    @test es.theta ≈ pyribs_openai.theta rtol = 1e-10   # pyribs's OpenAIEvolutionStrategy
    @test es.steps == 3

    # the batch consists of mirrored pairs around the center
    QD.sample!(Xoshiro(1), es)
    @test es.x[:, 1:4] .+ es.x[:, 5:8] ≈ 2 .* es.theta .* ones(1, 4)
    @test es.x[:, 1:4] .- es.theta ≈ 0.3 .* es.noise[:, 1:4]
    # the number of parents has no effect
    other = QD.OpenAIES(n; sigma0=0.3, batch_size=8, learning_rate=0.05)
    QD.reset!(other, oa_x0)
    other.noise .= es.noise
    es2 = deepcopy(other)
    QD.update!(other, 1:8, 1)
    QD.update!(es2, 1:8, 8)
    @test other.theta == es2.theta

    @test QD.OpenAIES(10; sigma0=1).batch_size == 2 * cld(4 + floor(Int, 3 * log(10)), 2)
    @test_throws "batch_size must be even because the solutions are mirrored, got 7" QD.OpenAIES(5; sigma0=1, batch_size=7)
    @test_throws "learning_rate must be positive, got 0" QD.OpenAIES(5; sigma0=1, learning_rate=0)
    @test_throws "beta1 and beta2 must be in [0, 1)" QD.OpenAIES(5; sigma0=1, beta1=1)
    @test_throws "epsilon must be positive, got 0" QD.OpenAIES(5; sigma0=1, epsilon=0)
    # mirrored solutions are drawn again together
    bounded = QD.OpenAIES(6; sigma0=0.3, batch_size=40, lower=-1.0, upper=1.0)
    QD.reset!(bounded, zeros(6))
    @test all(-1 .<= QD.sample!(Xoshiro(2), bounded) .<= 1)
end

@testset "stopping criteria of the other strategies" begin
    flat = [1.0, 1.0 + 1e-13]
    spread = [3.0, 2.0, 1.0]
    for S in (QD.SepCMAES, QD.LMMAES, QD.OpenAIES)
        es = S(10; sigma0=1.0, batch_size=4)
        QD.reset!(es, zeros(10))
        @test !QD.should_stop(es, spread) && QD.should_stop(es, flat) && !QD.should_stop(es, [1.0])
    end
    es = QD.SepCMAES(3; sigma0=1.0)
    QD.reset!(es, zeros(3))
    es.sigma = 1e-12
    @test QD.should_stop(es, spread)
    es.sigma = 1.0
    es.variances .= [1.0, 1.0, 1e-15]
    @test QD.should_stop(es, spread)
    es.variances .= 1
    es.mean[2] = NaN
    @test QD.should_stop(es, spread)

    lm = QD.LMMAES(10; sigma0=1.0, batch_size=5)
    QD.reset!(lm, zeros(10))
    lm.sigma = 1e-13
    @test QD.should_stop(lm, spread)
    lm.sigma = NaN
    @test QD.should_stop(lm, spread)

    oa = QD.OpenAIES(4; sigma0=1.0, batch_size=6)
    QD.reset!(oa, ones(4))
    oa.update_ratio = 1e-10
    @test QD.should_stop(oa, spread)
    oa.update_ratio = 1.0
    oa.theta[1] = Inf
    @test QD.should_stop(oa, spread)
end

@testset "CMAES follows the reference update" begin
    n = length(cma_x0)
    es = QD.CMAES(n; sigma0=cma_s0, batch_size=8)
    QD.reset!(es, cma_x0)
    for (g, flat) in enumerate(cma_X)
        load_batch!(es, columns(flat, n))
        QD.update!(es, cma_orders[g], cma_parents[g])
    end
    @test es.mean ≈ cma_mean rtol = 1e-10
    @test es.sigma ≈ cma_sigma rtol = 1e-10
    @test vec(es.cov) ≈ cma_cov rtol = 1e-10
end

@testset "$(nameof(S)) follows the reference over a long run" for (S, ref) in
        ((QD.CMAES, (x0=cmalong_x0, s0=cmalong_s0, X=cmalong_X, orders=cmalong_orders, parents=cmalong_parents,
                     mean=cmalong_mean, sigma=cmalong_sigma, shape=cmalong_shape)),
         (QD.SepCMAES, (x0=seplong_x0, s0=seplong_s0, X=seplong_X, orders=seplong_orders, parents=seplong_parents,
                        mean=seplong_mean, sigma=seplong_sigma, shape=seplong_shape)))
    # reference/strategies.jl: generations that switch hsig off and reach the cap of the
    # step-size change
    n = length(ref.x0)
    es = S(n; sigma0=ref.s0, batch_size=length(ref.orders[1]))
    QD.reset!(es, ref.x0)
    for (g, flat) in enumerate(ref.X)
        load_batch!(es, columns(flat, n))
        QD.update!(es, ref.orders[g], ref.parents[g])
    end
    @test es.mean ≈ ref.mean rtol = 1e-10
    @test es.sigma ≈ ref.sigma rtol = 1e-10
    @test vec(S === QD.CMAES ? es.cov : es.variances) ≈ ref.shape rtol = 1e-10
end

@testset "PyribsCMAES weakens the rank-one term" begin
    n = length(cma_x0)
    textbook = QD.CMAES(n; sigma0=cma_s0, batch_size=8)
    pyribs = QD.PyribsCMAES(n; sigma0=cma_s0, batch_size=8)
    @test textbook.rank_one isa QD.TextbookRankOne && pyribs.rank_one isa QD.PyribsRankOne
    @test pyribs isa QD.CMAES && pyribs.batch_size == 8 && pyribs.sigma0 == cma_s0
    for es in (textbook, pyribs)
        QD.reset!(es, cma_x0)
        load_batch!(es, columns(cma_X[1], n))
        QD.update!(es, cma_orders[1], cma_parents[1])
    end
    # one update from the same batch: same mean, step size and path, and a covariance
    # that differs by (c1 - c1²) pc pcᵀ
    c1 = QD.strategy_constants(n, cma_parents[1])[5]
    @test pyribs.mean == textbook.mean && pyribs.sigma == textbook.sigma && pyribs.path_c == textbook.path_c
    @test textbook.cov - pyribs.cov ≈ (c1 - c1^2) .* (textbook.path_c * textbook.path_c') atol = 1e-14
    @test norm(textbook.path_c) > 0.1 && textbook.cov != pyribs.cov
    @test QD.rank_one_rate(QD.TextbookRankOne(), 0.1) == 0.1 && QD.rank_one_rate(QD.PyribsRankOne(), 0.1) ≈ 0.01
    # the variant fills an archive through an emitter
    prob = Sphere(6)
    alg = CMAME(prob; dims=(6, 6), sigma0=0.5, num_emitters=2, strategy=QD.PyribsCMAES)
    @test alg.emitters[1].es isa QD.CMAES{QD.PyribsRankOne}
    @test coverage(solve(prob, alg; budget=1_000, rng=Xoshiro(1)).archive) > 0.1
end
