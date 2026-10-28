using JET: JET

@testset "JET" begin
    JET.test_package(QualityDiversity; target_modules=(QualityDiversity,))

    prob = Rastrigin(4)
    alg = MAPElites(prob; dims=(5, 5), batch_size=8)
    state = init(prob, alg; budget=64, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(state)
    JET.@test_opt target_modules = (QualityDiversity,) QD.add!(state.archive, [0.0, 0.0, 0.0, 0.0],
                                                            Evaluation(1.0, SVector(0.0, 0.0)))
    JET.@test_opt target_modules = (QualityDiversity,) QD.cell_index(state.archive, SVector(0.0, 0.0))
    JET.@test_opt target_modules = (QualityDiversity,) QD.sample_elites(Xoshiro(1), state.archive, 4)
    JET.@test_opt target_modules = (QualityDiversity,) QD.ask(Xoshiro(1), state.emitters[1], state.archive)
    JET.@test_opt target_modules = (QualityDiversity,) QD.vary(Xoshiro(1), GaussianMutation(0.1), [0.0, 0.0])

    pool = (VariationEmitter(GaussianMutation(0.1); batch_size=8), CMAEmitter(4; sigma0=0.5, batch_size=6),
            CMAEmitter(4; sigma0=0.5, batch_size=6))
    bandit = init(prob, QDAlgorithm(GridArchive(prob; dims=(5, 5)), (BanditScheduler(pool; n_active=2),));
                  budget=10_000, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(bandit)

    dns = init(prob, MAPElites(DNSArchive(prob; capacity=50, k=3); batch_size=8); budget=10_000, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(dns)
    JET.@test_opt target_modules = (QualityDiversity,) QD.add!(dns.archive, [0.0, 0.0, 0.0, 0.0],
                                                            Evaluation(1.0, SVector(0.0, 0.0)))

    dds = init(prob, DDS(prob; dims=(5, 5), sigma0=0.5, bandwidth=0.5, n_emitters=2, rng=Xoshiro(1)); budget=10_000,
               rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(dds)

    sliding = init(prob, MAPElites(SlidingBoundariesArchive(prob; dims=(5, 5), remap_frequency=16); batch_size=8);
                   budget=64, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(sliding)
    JET.@test_opt target_modules = (QualityDiversity,) QD.add!(sliding.archive, [0.0, 0.0, 0.0, 0.0],
                                                            Evaluation(1.0, SVector(0.0, 0.0)))

    noisy = Sampled(StochasticQDProblem((rng, x) -> (-sum(abs2, x) + randn(rng), (x[1], x[2]));
                                        init=rng -> randn(rng, 4), bounds=((-3, 3), (-3, 3))); n_samples=2)
    noisy_state = init(noisy, MAPElites(noisy; dims=(5, 5), batch_size=8); budget=200, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(noisy_state)
    JET.@test_opt target_modules = (QualityDiversity,) QD.evaluate_batch(Xoshiro(1), noisy, [randn(4), randn(4)])

    bi = BiSphere(4)
    mome_state = init(bi, MOME(bi; dims=(5, 5), batch_size=8); budget=200, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(mome_state)
    JET.@test_opt target_modules = (QualityDiversity,) QD.sample_elites(Xoshiro(1), mome_state.archive, 4)
    JET.@test_opt target_modules = (QualityDiversity,) QD.add!(mome_state.archive, zeros(4), QD.evaluate(bi, zeros(4)))

    cma = CMAME(prob; dims=(5, 5), sigma0=1.0, n_emitters=2)
    cma_state = init(prob, cma; budget=200, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(cma_state)
    JET.@test_opt target_modules = (QualityDiversity,) QD.ask(Xoshiro(1), cma_state.emitters[1], cma_state.archive)
    for kwargs in ((; ranker=QD.TwoStageRandomDirectionRanker()), (; strategy=QD.SepCMAES), (; strategy=QD.OpenAIES),
                   (; strategy=QD.LMMAES, batch_size=2))
        other = CMAME(prob; dims=(5, 5), sigma0=1.0, n_emitters=2, kwargs...)
        other_state = init(prob, other; budget=200, rng=Xoshiro(1))
        JET.@test_opt target_modules = (QualityDiversity,) step!(other_state)
    end
    mae = CMAMAE(prob; dims=(5, 5), sigma0=1.0, threshold_min=-500.0, learning_rate=0.1, n_emitters=2)
    mae_state = init(prob, mae; budget=200, rng=Xoshiro(1))
    JET.@test_opt target_modules = (QualityDiversity,) step!(mae_state)
    for alg in (CMAMEGA(prob; dims=(5, 5), sigma0=1.0, n_emitters=2),
                CMAMEGA(prob; dims=(5, 5), sigma0=1.0, optimizer=QD.GradientAscent(1.0)),
                CMAMAEGA(prob; dims=(5, 5), sigma0=1.0, threshold_min=-500.0),
                OMGMEGA(prob; dims=(5, 5), sigma0=1.0, batch_size=8))
        mega_state = init(prob, alg; budget=200, rng=Xoshiro(1))
        JET.@test_opt target_modules = (QualityDiversity,) step!(mega_state)
        JET.@test_opt target_modules = (QualityDiversity,) QD.ask(Xoshiro(1), mega_state.emitters[1], mega_state.archive)
    end
end
