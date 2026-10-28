include("reference/pareto.jl")
include("reference/crowding.jl")

pareto_archive(; dims=(3, 3), ref=[0.0, 0.0], kwargs...) =
    ParetoGridArchive{Vector{Float64}}(dims, ((0, 1), (0, 1)); reference_point=ref, kwargs...)

ev2(a, b, d=(0.2, 0.2)) = Evaluation(SVector(Float64(a), Float64(b)), SVector(Float64(d[1]), Float64(d[2])))

@testset "hypervolume" begin
    @test hypervolume([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]], [0.0, 0.0]) == 6.0
    @test hypervolume([[1.0, 1.0, 1.0], [2.0, 2.0, 0.5]], [0.0, 0.0, 0.0]) == 2.5
    @test hypervolume(Vector{Float64}[], [0.0, 0.0]) == 0.0
    @test hypervolume([[2.0]], [0.5]) == 1.5 && hypervolume([[2.0], [3.0]], [0.5]) == 2.5
    # a dominated point adds nothing, and so does a point that is not above the reference point
    front = [[3.0, 1.0], [1.0, 3.0]]
    @test hypervolume(vcat(front, [[0.5, 0.5], [1.0, 1.0]]), [0.0, 0.0]) == hypervolume(front, [0.0, 0.0])
    @test hypervolume(vcat(front, [[5.0, -1.0], [0.0, 9.0]]), [0.0, 0.0]) == hypervolume(front, [0.0, 0.0])
    # points that are not above the reference point in every objective add nothing
    @test hypervolume(front, [1.0, 1.0]) == 0.0
    # a point below the reference point in one objective adds nothing to the others
    @test hypervolume([[3.0, 1.0], [0.5, 5.0]], [1.0, 0.0]) == 2.0
    # the order of the points does not matter, and vectors of any kind work
    shuffled = [[1.0, 1.0, 3.0], [3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [2.0, 2.0, 2.0]]
    @test hypervolume(shuffled, [0.0, 0.0, 0.0]) == hypervolume(reverse(shuffled), [0.0, 0.0, 0.0])
    @test hypervolume([SVector(3.0, 1.0), SVector(1.0, 3.0)], SVector(0.0, 0.0)) == 5.0
    # against the hypervolume indicator of pymoo
    for c in hv_cases
        points = [c.points[(i-1)*c.m+1:i*c.m] for i in 1:length(c.points)÷c.m]
        @test hypervolume(points, zeros(c.m)) ≈ c.volume rtol = 1e-12
    end
    @test_throws "a point has 3 objectives but the reference point has 2" hypervolume([[1.0, 1.0, 1.0]], [0.0, 0.0])
    @test_throws "a point has a NaN objective" hypervolume([[1.0, NaN]], [0.0, 0.0])
    @test_throws "the reference point needs at least one objective" hypervolume([Float64[]], Float64[])
end

@testset "ParetoGridArchive construction" begin
    a = ParetoGridArchive((4, 5), ((0, 1), (-1, 1)); reference_point=[0.0, 0.0, 0.0], max_front=7)
    @test a isa ParetoGridArchive{Vector{Float64},3,2} && a isa QD.AbstractArchive{Vector{Float64},SVector{3,Float64},SVector{2,Float64}}
    @test QD.capacity(a) == 20 && length(a) == 0 && QD.num_solutions(a) == 0 && a.max_front == 7
    @test QD.descriptor_bounds(a) == ((0.0, 1.0), (-1.0, 1.0)) && QD.reference_point(a) == SVector(0.0, 0.0, 0.0)
    prob = BiSphere(4)
    b = ParetoGridArchive(prob; dims=(5, 5), reference_point=QD.reference_point(prob), genotype=SVector{4,Float64})
    @test b isa ParetoGridArchive{SVector{4,Float64},2} && b.max_front == 50 && QD.capacity(b) == 25
    @test_throws "dims[2] is 0 but every dimension needs at least one cell" pareto_archive(dims=(3, 0))
    @test_throws "bounds[1] is (1.0, 0.0) but lo must be finite and less than hi" ParetoGridArchive{Vector{Float64}}((2,), ((1, 0),); reference_point=[0.0, 0.0])
    @test_throws "dims has 2 entries but bounds has 1" ParetoGridArchive{Vector{Float64}}((2, 2), ((0, 1),); reference_point=[0.0, 0.0])
    @test_throws "the reference point has 1 entries but several objectives need at least 2" ParetoGridArchive{Vector{Float64}}((2,), ((0, 1),); reference_point=[0.0])
    # the reference point is a required keyword, as for MOME; it is never positional
    @test_throws UndefKeywordError ParetoGridArchive((2, 2), ((0, 1), (0, 1)))
    @test_throws MethodError ParetoGridArchive((2, 2), ((0, 1), (0, 1)), [0.0, 0.0])
    @test_throws UndefKeywordError ParetoGridArchive(prob; dims=(2, 2))
    @test_throws "the reference point must be finite" pareto_archive(ref=[0.0, -Inf])
    @test_throws "max_front must be at least 1, got 0" pareto_archive(max_front=0)
end

@testset "ParetoGridArchive add!" begin
    a = pareto_archive()
    r = QD.add!(a, [1.0], ev2(1, 3))
    @test r.status == QD.NewCell && r.improvement == 3.0 && r isa QD.AddResult{Float64}
    r = QD.add!(a, [2.0], ev2(3, 1))
    @test r.status == QD.Improved && r.improvement == 2.0
    @test length(a) == 1 && QD.num_solutions(a) == 2 && QD.qd_score(a) == 5.0
    # equal or beaten in every objective: rejected, and the archive is unchanged
    for (f1, f2) in ((1, 3), (3, 1), (0.5, 0.5), (1, 1), (2.9, 1))
        r = QD.add!(a, [9.0], ev2(f1, f2))
        @test r.status == QD.Rejected && r.improvement == 0.0
    end
    @test QD.num_solutions(a) == 2 && QD.qd_score(a) == 5.0
    # a solution on the front that adds area
    r = QD.add!(a, [3.0], ev2(2, 2))
    @test r.status == QD.Improved && r.improvement == 1.0 && QD.qd_score(a) == 6.0 && QD.num_solutions(a) == 3
    # a solution that beats some of the front removes those and keeps the others
    r = QD.add!(a, [4.0], ev2(2.5, 2.5))
    @test r.status == QD.Improved && QD.num_solutions(a) == 3
    @test sort([g[1] for (g, _) in QD.elites(a)]) == [1.0, 2.0, 4.0]
    @test QD.qd_score(a) ≈ hypervolume([[1, 3], [3, 1], [2.5, 2.5]], [0, 0])
    # a solution that beats the whole front
    r = QD.add!(a, [5.0], ev2(4, 4))
    @test QD.num_solutions(a) == 1 && r.improvement ≈ 16 - 7.25
    @test QD.qd_score(a) == 16.0

    # other cells are independent
    QD.add!(a, [6.0], ev2(1, 1, (0.9, 0.9)))
    @test length(a) == 2 && QD.num_solutions(a) == 2 && QD.qd_score(a) == 17.0
    @test QD.coverage(a) == 2 / 9
    @test QD.max_hypervolume(a) == 16.0 && QD.global_hypervolume(a) == 16.0

    # objectives below the reference point count as solutions with no volume
    b = pareto_archive()
    r = QD.add!(b, [1.0], ev2(-1, 5))
    @test r.status == QD.NewCell && r.improvement == 0.0 && QD.num_solutions(b) == 1 && QD.qd_score(b) == 0.0

    @test_throws "the fitness of a solution must be a vector of 2 objectives for a ParetoGridArchive, got a Float64" QD.add!(a, [1.0], Evaluation(1.0, SVector(0.2, 0.2)))
    @test_throws "fitness has 3 objectives but the archive has 2" QD.add!(a, [1.0], Evaluation(SVector(1.0, 2.0, 3.0), SVector(0.2, 0.2)))
    @test_throws "every objective must be finite, got [NaN, 1.0]" QD.add!(a, [1.0], ev2(NaN, 1))
    @test_throws "every objective must be finite, got [Inf, 1.0]" QD.add!(a, [1.0], ev2(Inf, 1))
    @test_throws "every objective must be finite" QD.add_batch!(a, [[1.0], [2.0]], [ev2(1, 1), ev2(1, -Inf)])
    @test_throws "descriptor has 3 dimensions but the archive has 2" QD.add!(a, [1.0], Evaluation(SVector(1.0, 2.0), SVector(0.1, 0.1, 0.1)))
    @test_throws "outside the archive bounds" QD.add!(a, [1.0], ev2(1, 1, (2.0, 0.5)))
    rejecting = pareto_archive(outside=RejectOutside())
    @test QD.add!(rejecting, [1.0], ev2(1, 1, (2.0, 0.5))) == QD.AddResult(QD.Rejected, -Inf)
    rs = QD.add_batch!(rejecting, [[1.0], [2.0]], [ev2(1, 1, (2.0, 0.5)), ev2(1, 1, (0.5, 0.5))])
    @test rs[1] == QD.AddResult(QD.Rejected, -Inf) && rs[2].status == QD.NewCell
    @test length(rejecting) == 1
    empty!(rejecting)
    # the objectives are checked also for a solution the archive discards
    @test_throws "every objective must be finite" QD.add!(rejecting, [1.0], ev2(NaN, 1, (2.0, 0.5)))
    @test_throws "every objective must be finite" QD.add_batch!(rejecting, [[1.0]], [ev2(NaN, 1, (2.0, 0.5))])
    @test length(rejecting) == 0
end

@testset "a full front rejects what beats none of it" begin
    a = pareto_archive(max_front=3)
    for (k, t) in enumerate((0.1, 0.5, 0.9))
        @test QD.add!(a, [Float64(k)], ev2(t, 1 - t)).status in (QD.NewCell, QD.Improved)
    end
    before = collect(QD.elites(a))
    r = QD.add!(a, [9.0], ev2(0.3, 0.7))                  # on the front, but there is no room
    @test r.status == QD.Rejected && r.improvement == 0.0 && collect(QD.elites(a)) == before
    # a solution that beats one of the three still enters
    r = QD.add!(a, [8.0], ev2(0.6, 0.6))
    @test r.status == QD.Improved && QD.num_solutions(a) == 3 && 8.0 in [g[1] for (g, _) in QD.elites(a)]
end

@testset "duplicates" begin
    a = pareto_archive()
    @test a.duplicates === QD.RejectDuplicates()
    QD.add!(a, [1.0], ev2(1, 3))
    @test QD.add!(a, [2.0], ev2(1, 3)) == QD.AddResult(QD.Rejected, 0.0) && QD.num_solutions(a) == 1
    b = pareto_archive(duplicates=QD.KeepDuplicates())
    QD.add!(b, [1.0], ev2(1, 3))
    @test QD.add!(b, [2.0], ev2(1, 3)) == QD.AddResult(QD.Improved, 0.0)
    @test QD.num_solutions(b) == 2 && QD.qd_score(b) == 3.0
    # a solution that beats both copies removes both; one that one copy beats stays out
    @test QD.add!(b, [3.0], ev2(0.5, 3)).status == QD.Rejected
    @test QD.add!(b, [4.0], ev2(1, 4)).status == QD.Improved
    @test [g for (g, _) in QD.elites(b)] == [[4.0]] && QD.qd_score(b) == 4.0
    # a full front of copies rejects one more copy, whatever the full-front rule
    for full_front in (QD.RejectNewcomer(), QD.ReplaceCrowded())
        c = pareto_archive(duplicates=QD.KeepDuplicates(), full_front=full_front, max_front=2)
        QD.add!(c, [1.0], ev2(1, 3)); QD.add!(c, [2.0], ev2(1, 3))
        @test QD.add!(c, [3.0], ev2(1, 3)).status == QD.Rejected
        @test sort([g[1] for (g, _) in QD.elites(c)]) == [1.0, 2.0]
    end
end

@testset "a full front replaces its most crowded solution" begin
    function full_front(ts; max_front=3)
        a = pareto_archive(; max_front, full_front=QD.ReplaceCrowded())
        for (k, t) in enumerate(ts)
            QD.add!(a, [Float64(k)], ev2(t, 1 - t))
        end
        return a
    end
    ids(a) = sort([g[1] for (g, _) in QD.elites(a)])
    # of 0.1, 0.2, 0.6 and 0.9 on the line, 0.2 has the closest neighbours
    a = full_front((0.1, 0.2, 0.9))
    before = QD.qd_score(a)
    r = QD.add!(a, [4.0], ev2(0.6, 0.4))
    @test r.status == QD.Improved && ids(a) == [1.0, 3.0, 4.0] && QD.num_solutions(a) == 3
    @test r.improvement ≈ hypervolume([[0.1, 0.9], [0.6, 0.4], [0.9, 0.1]], [0, 0]) - before
    @test QD.qd_score(a) ≈ before + r.improvement
    # the new solution is the most crowded: rejected, and the archive is unchanged
    a = full_front((0.1, 0.5, 0.9))
    before = collect(QD.elites(a))
    @test QD.add!(a, [4.0], ev2(0.3, 0.7)) == QD.AddResult(QD.Rejected, 0.0)
    @test collect(QD.elites(a)) == before
    # a tie between the new solution and a stored one removes the new one (the values are
    # binary fractions, so the distances tie exactly)
    a = full_front((0.125, 0.375, 0.875))
    @test QD.add!(a, [4.0], ev2(0.625, 0.375)).status == QD.Rejected && ids(a) == [1.0, 2.0, 3.0]
    # a tie between stored solutions removes the one stored first: 0.375 before 0.25
    a = full_front((0.125, 0.375, 0.25, 0.5, 0.875); max_front=5)
    @test QD.add!(a, [6.0], ev2(0.75, 0.25)).status == QD.Improved && ids(a) == [1.0, 3.0, 4.0, 5.0, 6.0]
    # the ends of the front stay; a new end takes the place of the old one
    a = full_front((0.1, 0.9); max_front=2)
    @test QD.add!(a, [3.0], ev2(0.5, 0.5)).status == QD.Rejected
    r = QD.add!(a, [3.0], ev2(0.95, 0.04))
    @test r.status == QD.Improved && ids(a) == [1.0, 3.0]
    @test r.improvement ≈ hypervolume([[0.1, 0.9], [0.95, 0.04]], [0, 0]) - hypervolume([[0.1, 0.9], [0.9, 0.1]], [0, 0])
    @test r.improvement < 0
    # with one slot the stored solution and the new one are both ends
    a = full_front((0.2,); max_front=1)
    @test QD.add!(a, [2.0], ev2(0.8, 0.2)).status == QD.Rejected && ids(a) == [1.0]
    # an objective in which every solution is equal adds nothing to the distance
    ev3(f...) = Evaluation(SVector(Float64.(f)...), SVector(0.2, 0.2))
    a = ParetoGridArchive{Float64}((2, 2), ((0, 1), (0, 1)); reference_point=zeros(3), max_front=3, full_front=QD.ReplaceCrowded())
    for (k, t) in enumerate((0.1, 0.2, 0.9))
        QD.add!(a, Float64(k), ev3(t, 1 - t, 0.5))
    end
    @test QD.add!(a, 4.0, ev3(0.6, 0.4, 0.5)).status == QD.Improved
    @test sort([g for (g, _) in QD.elites(a)]) == [1.0, 3.0, 4.0]
    # a solution that beats one of a full front takes its place without a crowding test
    a = full_front((0.1, 0.5, 0.9))
    @test QD.add!(a, [4.0], ev2(0.5, 0.55)).status == QD.Improved && ids(a) == [1.0, 3.0, 4.0]
end

@testset "fronts kept as by MOME-PGX" begin
    for c in crowding_cases
        m = c.objectives
        a = ParetoGridArchive{Int}((1,), ((0.0, 1.0),); reference_point=fill(-10.0, m), max_front=c.max_front, duplicates=QD.KeepDuplicates(), full_front=QD.ReplaceCrowded())
        for i in 1:length(c.fitness)÷m
            QD.add!(a, i - 1, Evaluation(SVector{m,Float64}(c.fitness[(i-1)*m+1:i*m]), SVector(0.5)))
        end
        @test sort([g for (g, _) in QD.elites(a)]) == c.kept
    end
end

@testset "fronts kept as by QDax" begin
    for c in front_cases
        m = c.objectives
        a = ParetoGridArchive{Int}((1,), ((0.0, 1.0),); reference_point=fill(-10.0, m), max_front=c.max_front)
        for i in 1:length(c.fitness)÷m
            QD.add!(a, i - 1, Evaluation(SVector{m,Float64}(c.fitness[(i-1)*m+1:i*m]), SVector(0.5)))
        end
        @test sort([g for (g, _) in QD.elites(a)]) == c.kept
    end
end

@testset "front invariants under random insertions, $duplicates, $full_front" for duplicates in (QD.RejectDuplicates(), QD.KeepDuplicates()),
                                                                                         full_front in (QD.RejectNewcomer(), QD.ReplaceCrowded())
    rng = Xoshiro(4)
    a = pareto_archive(; dims=(2, 2), max_front=6, duplicates, full_front)
    for _ in 1:600
        # objectives on a coarse lattice, so that equal objective vectors occur
        f = round.(rand(rng, 2); digits=1)
        QD.add!(a, [f[1]], Evaluation(SVector(f...), SVector(rand(rng, 2)...)))
    end
    fronts = Dict{Int,Vector{SVector{2,Float64}}}()
    for (_, ev) in QD.elites(a)
        push!(get!(fronts, QD.cell_index(a, ev.descriptor), SVector{2,Float64}[]), ev.fitness)
    end
    @test length(fronts) == length(a) == 4
    for front in values(fronts)
        @test 1 <= length(front) <= 6
        @test all(f -> !any(g -> g != f && all(g .>= f), front), front)
        duplicates isa QD.RejectDuplicates && @test allunique(front)
    end
    @test QD.num_solutions(a) == sum(length, values(fronts))
    @test QD.qd_score(a) ≈ sum(hypervolume(front, [0.0, 0.0]) for front in values(fronts))
    @test QD.qd_score(a) ≈ sum(a.volumes) && a.volume_sum + a.volume_sum_error ≈ sum(a.volumes)
    @test QD.global_hypervolume(a) ≈ hypervolume(reduce(vcat, values(fronts)), [0.0, 0.0])
    @test QD.global_hypervolume(a) <= QD.qd_score(a) + 1e-12
    @test QD.max_hypervolume(a) == maximum(a.volumes)
end

@testset "fronts of three objectives under random batches, $duplicates, $full_front" for duplicates in (QD.RejectDuplicates(), QD.KeepDuplicates()),
                                                                                             full_front in (QD.RejectNewcomer(), QD.ReplaceCrowded())
    rng = Xoshiro(6)
    a = pareto_archive(; dims=(2, 2), ref=[0.0, 0.0, 0.0], max_front=5, duplicates, full_front)
    one_by_one = deepcopy(a)
    largest = Float64[]
    for _ in 1:80
        n = rand(rng, 1:12)
        xs = [rand(rng, 1) for _ in 1:n]
        # objectives on a coarse lattice, so that ties and equal objective vectors occur
        evs = [Evaluation(SVector{3}(round.(rand(rng, 3) .* 4; digits=1)), SVector(rand(rng, 2)...)) for _ in 1:n]
        before = deepcopy(a)
        results = QD.add_batch!(a, xs, evs)
        @test results == [QD.add!(deepcopy(before), x, ev) for (x, ev) in zip(xs, evs)]
        foreach((x, ev) -> QD.add!(one_by_one, x, ev), xs, evs)
        push!(largest, QD.max_hypervolume(a))
        @test QD.max_hypervolume(a) == maximum(a.volumes)
    end
    @test sort(collect(QD.elites(a)); by=e -> (e[2].descriptor, e[2].fitness)) ==
          sort(collect(QD.elites(one_by_one)); by=e -> (e[2].descriptor, e[2].fitness))
    fronts = Dict{Int,Vector{SVector{3,Float64}}}()
    for (_, ev) in QD.elites(a)
        push!(get!(fronts, QD.cell_index(a, ev.descriptor), SVector{3,Float64}[]), ev.fitness)
    end
    for (i, front) in fronts
        @test a.volumes[i] ≈ hypervolume(front, [0.0, 0.0, 0.0])
    end
    # only replacing a crowded solution can lower the largest hypervolume of a cell
    full_front isa QD.RejectNewcomer && @test issorted(largest)
end

@testset "ParetoGridArchive add_batch!" begin
    rng = Xoshiro(2)
    prob = BiSphere(4)
    xs = [QD.random_solution(rng, prob) for _ in 1:120]
    evs = QD.evaluate_batch(prob, xs)
    make() = ParetoGridArchive(prob; dims=(4, 4), reference_point=QD.reference_point(prob), max_front=5)
    a, b = make(), make()
    QD.add_batch!(a, xs[1:60], evs[1:60])
    QD.add_batch!(b, xs[1:60], evs[1:60])
    batch = QD.add_batch!(a, xs[61:120], evs[61:120])
    # every result is that of the solution alone, against the archive before the batch
    @test all(k -> batch[k] == QD.add!(deepcopy(b), xs[60+k], evs[60+k]), 1:60)
    # and the archive afterwards has what adding one at a time leaves, up to the order within a front
    for (x, ev) in zip(xs[61:120], evs[61:120])
        QD.add!(b, x, ev)
    end
    fronts(arch) = sort([(QD.cell_index(arch, ev.descriptor), Tuple(ev.fitness)) for (_, ev) in QD.elites(arch)])
    @test length(a) == length(b) && QD.num_solutions(a) == QD.num_solutions(b)
    @test QD.qd_score(a) ≈ QD.qd_score(b)
    # two solutions of one batch that beat each other: only the better one stays
    c = pareto_archive()
    rs = QD.add_batch!(c, [[1.0], [2.0]], [ev2(1, 1), ev2(2, 2)])
    @test [r.status for r in rs] == [QD.NewCell, QD.NewCell] && QD.num_solutions(c) == 1
    @test only(QD.elites(c))[1] == [2.0]
    @test_throws "got 2 solutions but 1 evaluations" QD.add_batch!(c, [[1.0], [2.0]], [ev2(1, 1)])

    # with crowding replacement and duplicates as well
    make_crowded() = ParetoGridArchive(prob; dims=(2, 2), reference_point=QD.reference_point(prob), max_front=3,
                                       duplicates=QD.KeepDuplicates(), full_front=QD.ReplaceCrowded())
    a, b = make_crowded(), make_crowded()
    QD.add_batch!(a, xs[1:60], evs[1:60])
    QD.add_batch!(b, xs[1:60], evs[1:60])
    copies = vcat(xs[61:120], xs[61:70])
    copy_evs = vcat(evs[61:120], evs[61:70])
    batch = QD.add_batch!(a, copies, copy_evs)
    @test all(k -> batch[k] == QD.add!(deepcopy(b), copies[k], copy_evs[k]), eachindex(copies))
    for (x, ev) in zip(copies, copy_evs)
        QD.add!(b, x, ev)
    end
    @test collect(QD.elites(a)) == collect(QD.elites(b))
end

@testset "ParetoGridArchive sampling" begin
    a = pareto_archive()
    QD.add!(a, [0.0], ev2(1, 1, (0.1, 0.1)))                      # a cell with one solution
    for (k, t) in enumerate((1.0, 2.0, 3.0))                      # a cell with three
        QD.add!(a, [10.0k], ev2(t, 4 - t, (0.9, 0.9)))
    end
    draws = QD.sample_elites(Xoshiro(1), a, 60_000)
    share(v) = count(==([v]), draws) / length(draws)
    # a cell is drawn uniformly and then a solution of its front
    @test share(0.0) ≈ 1 / 2 atol = 0.01
    @test all(v -> abs(share(v) - 1 / 6) < 0.01, (10.0, 20.0, 30.0))
    @test QD.sample_elites(Xoshiro(2), a, 5) == QD.sample_elites(Xoshiro(2), a, 5)
    @test_throws "cannot sample elites from an empty archive" QD.sample_elites(Xoshiro(1), pareto_archive(), 1)
end

@testset "ParetoGridArchive metrics and empty!" begin
    a = pareto_archive()
    @test QD.qd_score(a) == 0 && QD.coverage(a) == 0 && QD.global_hypervolume(a) == 0
    @test_throws "max_hypervolume is undefined for an archive with no elites" QD.max_hypervolume(a)
    @test_throws "max_scores is undefined for an archive with no elites" QD.max_scores(a)
    QD.add!(a, [1.0], ev2(1, 3)); QD.add!(a, [2.0], ev2(3, 1)); QD.add!(a, [3.0], ev2(2, 2, (0.9, 0.1)))
    @test QD.max_scores(a) == SVector(3.0, 3.0)
    @test_throws "max_fitness is not defined for a ParetoGridArchive" QD.max_fitness(a)
    @test_throws "mean_fitness is not defined for a ParetoGridArchive" QD.mean_fitness(a)
    @test_throws "the MOQD score of a ParetoGridArchive takes no offset" QD.qd_score(a; offset=1.0)
    @test QD.qd_score(a; offset=0) == 9.0
    record = QD.metrics_record(a, 17, 0)
    @test record == (n_evals=17, qd_score=9.0, coverage=2 / 9, max_hypervolume=5.0, num_solutions=3)
    @test QD.metrics_record_type(typeof(a)) == typeof(record)
    empty!(a)
    @test length(a) == 0 && QD.num_solutions(a) == 0 && QD.qd_score(a) == 0 && isempty(collect(QD.elites(a)))
    @test QD.metrics_record(a, 0, 0).max_hypervolume == 0.0
    @test QD.add!(a, [1.0], ev2(1, 3)).status == QD.NewCell

    # A changed cell adds its new hypervolume and subtracts the old one as two compensated
    # steps, so the total with its error term tracks the exact sum far below one rounding
    # even when their difference is not representable (volumes below 2 against 2^53).
    b = ParetoGridArchive{Vector{Float64}}((1000,), ((0, 1),); reference_point=[0.0, 0.0])
    rng = Xoshiro(3)
    exact = BigFloat(0)
    for c in 1:1000
        d = SVector((c - 0.5) / 1000)
        QD.add!(b, [0.0], Evaluation(SVector(1 + rand(rng), 1.0), d))
        volume = 2.0^53 * (1 + rand(rng))
        QD.add!(b, [0.0], Evaluation(SVector(volume, 1.0), d))        # dominates the first point
        exact += volume
    end
    @test QD.num_solutions(b) == 1000
    two_terms = BigFloat(b.volume_sum) + BigFloat(b.volume_sum_error)
    @test abs(two_terms - exact) <= 1e-24 * exact
    @test QD.qd_score(b) == Float64(exact)
end

@testset "BiSphere" begin
    prob = BiSphere(4)
    ev = QD.evaluate(prob, [1.0, 1.0, 1.0, 1.0])
    @test ev.fitness == SVector(0.0, -16.0) && ev.descriptor == SVector(1.0, 1.0)
    @test QD.evaluate(prob, [-1.0, -1.0, -1.0, -1.0]).fitness == SVector(-16.0, 0.0)
    @test QD.evaluate(prob, [0.0, 0.0, 0.0, 0.0]).fitness == SVector(-4.0, -4.0)
    @test QD.evaluate(prob, fill(9.0, 4)).fitness == SVector(-4 * 4.12^2, -4 * 6.12^2)        # clipped to 5.12
    @test all(all(QD.evaluate(prob, QD.random_solution(Xoshiro(s), prob)).fitness .>= QD.reference_point(prob)) for s in 1:200)
    @test_throws "BiSphere needs at least 2 dimensions for its 2-D descriptor, got 1" BiSphere(1)
    @test_throws "solution has 3 entries but BiSphere has dimension 4" QD.evaluate(prob, [1.0, 2.0, 3.0])
    @test_throws "Rastrigin does not define reference_point" QD.reference_point(Rastrigin(3))
end

@testset "MOME" begin
    prob = BiSphere(6)
    alg = MOME(prob; dims=(8, 8), batch_size=32)
    @test alg isa QDAlgorithm && alg.archive isa ParetoGridArchive{Vector{Float64},2} && alg.archive.max_front == 50
    @test length(alg.emitters) == 1 && QD.batch_size(alg.emitters[1]) == 32
    @test alg.archive.reference_point == QD.reference_point(prob)
    @test MOME(prob; dims=(8, 8), max_front=7, reference_point=[-1.0, -2.0]).archive.max_front == 7
    @test MOME(prob; dims=(8, 8), reference_point=[-1.0, -2.0]).archive.reference_point == SVector(-1.0, -2.0)
    own = pareto_archive()
    @test MOME(own; batch_size=5).archive === own
    @test_throws "dims" MOME(prob)
    @test_throws "Rastrigin does not define reference_point" MOME(Rastrigin(4); dims=(4, 4))
    @test_throws "the problem has 2 descriptor dimensions but the archive has 1" init(prob, MOME(ParetoGridArchive{Vector{Float64}}((3,), ((0, 1),); reference_point=[-1e3, -1e3])); budget=100)

    result = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test result.archive isa ParetoGridArchive && result.n_evals >= 6_000
    history = result.history
    @test eltype(history) == QD.ParetoMetricsRecord && history[1].n_evals == 32 && history[end].n_evals == result.n_evals
    @test issorted([h.qd_score for h in history])                       # fronts only gain volume
    @test history[end].qd_score == QD.qd_score(result.archive) && history[end].num_solutions == QD.num_solutions(result.archive)
    @test history[end].coverage == QD.coverage(result.archive) && history[end].max_hypervolume == QD.max_hypervolume(result.archive)
    @test QD.num_solutions(result.archive) > length(result.archive)            # fronts hold several solutions
    @test QD.max_scores(result.archive)[1] > -2 && QD.max_scores(result.archive)[2] > -2
    # a run is reproducible from its seed
    again = solve(prob, alg; budget=6_000, rng=Xoshiro(1))
    @test again.history == history && collect(QD.elites(again.archive)) == collect(QD.elites(result.archive))
    @test length(alg.archive) == 0                                               # the algorithm is not modified
    # for the same solutions, a front that keeps every non-dominated one holds at least the
    # hypervolume of a front cut off at two
    rng = Xoshiro(3)
    xs = [QD.random_solution(rng, prob) for _ in 1:1_500]
    evs = QD.evaluate_batch(prob, xs)
    wide = MOME(prob; dims=(8, 8), max_front=100).archive
    narrow = MOME(prob; dims=(8, 8), max_front=2).archive
    QD.add_batch!(wide, xs, evs)
    QD.add_batch!(narrow, xs, evs)
    @test length(wide) == length(narrow)
    @test all(wide.volumes .>= narrow.volumes) && QD.qd_score(wide) > QD.qd_score(narrow)
    @test QD.num_solutions(narrow) <= 2 * length(narrow)
    @test QD.num_solutions(narrow) < QD.num_solutions(wide)
end

@testset "a batch leaves the fronts as adding one at a time does" begin
    # C beats both stored solutions; D beats neither of them, so against the full front
    # before the batch it is rejected, but after C took their place it enters.
    for full_front in (QD.RejectNewcomer(), QD.ReplaceCrowded())
        make() = (a = ParetoGridArchive((1,), ((0, 1),); reference_point=[0.0, 0.0], max_front=2, full_front);
                  QD.add_batch!(a, [[1.0], [2.0]], [Evaluation([1.0, 3.0], SVector(0.5)), Evaluation([3.0, 1.0], SVector(0.5))]);
                  a)
        batch = [Evaluation([4.0, 4.0], SVector(0.5)), Evaluation([5.0, 0.5], SVector(0.5))]
        together = make()
        results = QD.add_batch!(together, [[3.0], [4.0]], batch)
        one_by_one = make()
        foreach((x, ev) -> QD.add!(one_by_one, x, ev), [[3.0], [4.0]], batch)
        @test sort(first.(collect(QD.elites(together)))) == sort(first.(collect(QD.elites(one_by_one)))) == [[3.0], [4.0]]
        @test results[1].status == QD.Improved
        # the results are those against the front before the batch
        @test results[2] == QD.add!(make(), [4.0], batch[2])
    end
end
