"""
    QualityDiversity.Testing

Conformance checks for the types a user defines. Each function runs inside a
`@testset`, so failures are reported by the test runner of the caller:

```julia
using Test, QualityDiversity
@testset "MyProblem" begin
    QualityDiversity.Testing.test_problem(MyProblem())
end
```

The checks cover the contract of each role, whether the calls are inferrable,
that the results have the right types, and, for archives, that `cell_index`
(and `add!` for `isbits` genotypes) does not allocate.

- [`test_problem`](@ref QualityDiversity.Testing.test_problem)
- [`test_archive`](@ref QualityDiversity.Testing.test_archive)
- [`test_front_archive`](@ref QualityDiversity.Testing.test_front_archive)
- [`test_variation`](@ref QualityDiversity.Testing.test_variation)
- [`test_emitter`](@ref QualityDiversity.Testing.test_emitter)
"""
module Testing

using Random: AbstractRNG, Xoshiro
using Test: @inferred, @test, @test_throws, @testset

import ..QualityDiversity as QD

export test_problem, test_archive, test_front_archive, test_variation, test_emitter

# Allocation of one call, measured after a warm-up call. The call goes through
# a function barrier so that the arguments have concrete types.
function allocations(f::F, args::Vararg{Any,N}) where {F,N}
    f(args...)
    return @allocated f(args...)
end

# Tests the descriptor space of `prob` and returns its categories (or `nothing`) and
# its bounds (or `nothing`).
function test_descriptor_space(prob)
    categories = QD.descriptor_categories(prob)
    bounds = categories === nothing ? QD.descriptor_bounds(prob) : nothing
    if categories === nothing
        @testset "descriptor_bounds" begin
            @test bounds isa Tuple
            @test !isempty(bounds)
            @test all(b -> length(b) == 2 && b[1] < b[2], bounds)
        end
    else
        @testset "descriptor_categories" begin
            @test categories isa Tuple
            @test !isempty(categories)
            @test all(c -> !isempty(c), categories)
        end
    end
    return categories, bounds
end

# Tests that a real descriptor lies inside the bounds, or each of the descriptors of a
# `SampledSpread` problem.
function test_descriptor_values(d, bounds)
    @test d isa AbstractVector{<:Real}
    @test length(d) == length(bounds)
    @test all(i -> bounds[i][1] <= d[i] <= bounds[i][2], eachindex(bounds))
    return nothing
end

function test_descriptor_values(samples::QD.DescriptorSamples, bounds)
    @test !isempty(samples)
    foreach(d -> test_descriptor_values(d, bounds), samples)
    return nothing
end

# Tests that one evaluation has the types and the descriptor space of the problem.
function test_evaluation_values(ev, categories, bounds)
    @test ev isa QD.Evaluation
    @test ev.fitness isa Union{Real,AbstractVector{<:Real}}
    if categories === nothing
        test_descriptor_values(ev.descriptor, bounds)
    else
        @test ev.descriptor isa Tuple
        @test length(ev.descriptor) == length(categories)
        @test all(i -> any(isequal(ev.descriptor[i]), categories[i]), eachindex(categories))
    end
    return nothing
end

"""
    test_problem(prob; rng=Xoshiro(1), n=5, rtol=0)

Check that `prob` meets the contract of [`AbstractQDProblem`](@ref QualityDiversity.AbstractQDProblem), or of
[`AbstractStochasticQDProblem`](@ref QualityDiversity.AbstractStochasticQDProblem).

`random_solution` is called `n` times. Every solution must evaluate to an
[`Evaluation`](@ref QualityDiversity.Evaluation) with a real or vector fitness and a descriptor that lies
inside `descriptor_bounds(prob)` and has one entry per bound. For a problem
that defines `descriptor_categories(prob)` instead, the descriptor must be a
tuple with one entry equal to a category of each dimension. `evaluate` must be
inferrable, deterministic, and agree with `evaluate_batch`.

`evaluate_batch` must give exactly the results of `evaluate` unless `rtol` is
positive. A problem that evaluates batches on a GPU, in lower precision, agrees
only up to rounding: with `rtol` the fitness and descriptor values then have to
match within the relative tolerance `rtol`, and within `rtol` in absolute terms
for values near zero.

For a stochastic problem, `evaluate(rng, prob, x)` must give the same result for the same
seed, `evaluate(prob, x)` without a generator must throw, and `evaluate_batch(rng, prob,
xs)` must be reproducible from the generator. `rtol` does not apply.
"""
function test_problem(prob::QD.AbstractQDProblem; rng::AbstractRNG=Xoshiro(1), n::Integer=5,
                      rtol::Real=0)
    @testset "test_problem($(nameof(typeof(prob))))" begin
        categories, bounds = test_descriptor_space(prob)

        xs = [QD.random_solution(rng, prob) for _ in 1:n]
        @testset "random_solution" begin
            @test isconcretetype(eltype(xs))
            @test QD.random_solution(Xoshiro(7), prob) == QD.random_solution(Xoshiro(7), prob)
        end

        @testset "evaluate" begin
            for x in xs
                ev = @inferred QD.evaluate(prob, x)
                test_evaluation_values(ev, categories, bounds)
                @test QD.evaluate(prob, x) == ev
            end
        end

        @testset "evaluate_batch" begin
            evs = QD.evaluate_batch(prob, xs)
            @test evs isa AbstractVector{<:QD.Evaluation}
            @test length(evs) == n
            reference = [QD.evaluate(prob, x) for x in xs]
            if rtol == 0
                @test evs == reference
            else
                @test all(zip(evs, reference)) do (a, b)
                    isapprox(a.fitness, b.fitness; rtol, atol=rtol) &&
                        (a.descriptor isa Tuple ? a.descriptor == b.descriptor :
                         isapprox(a.descriptor, b.descriptor; rtol, atol=rtol))
                end
            end
        end

        @testset "fitness_offset" begin
            @test QD.fitness_offset(prob) isa Real
        end

        @testset "evaluations_per_solution" begin
            @test QD.evaluations_per_solution(prob) isa Integer
            @test QD.evaluations_per_solution(prob) >= 1
        end
    end
    return nothing
end

function test_problem(prob::QD.AbstractStochasticQDProblem; rng::AbstractRNG=Xoshiro(1),
                      n::Integer=5)
    @testset "test_problem($(nameof(typeof(prob))))" begin
        categories, bounds = test_descriptor_space(prob)

        xs = [QD.random_solution(rng, prob) for _ in 1:n]
        @testset "random_solution" begin
            @test isconcretetype(eltype(xs))
            @test QD.random_solution(Xoshiro(7), prob) == QD.random_solution(Xoshiro(7), prob)
        end

        @testset "evaluate" begin
            for x in xs
                ev = @inferred QD.evaluate(Xoshiro(3), prob, x)
                test_evaluation_values(ev, categories, bounds)
                # every random number comes from the generator, so a seed fixes the result
                @test QD.evaluate(Xoshiro(3), prob, x) == ev
            end
            @test_throws "is a stochastic problem" QD.evaluate(prob, xs[1])
        end

        @testset "evaluate_batch" begin
            evs = QD.evaluate_batch(Xoshiro(5), prob, xs)
            @test evs isa AbstractVector{<:QD.Evaluation}
            @test length(evs) == n
            @test QD.evaluate_batch(Xoshiro(5), prob, xs) == evs
            @test_throws "is a stochastic problem" QD.evaluate_batch(prob, xs)
        end

        @testset "fitness_offset" begin
            @test QD.fitness_offset(prob) isa Real
        end

        @testset "evaluations_per_solution" begin
            @test QD.evaluations_per_solution(prob) isa Integer
            @test QD.evaluations_per_solution(prob) >= 1
        end
    end
    return nothing
end

"""
    test_archive(make_archive, prob; rng=Xoshiro(1), n=200, check_allocations=true, elitist=true,
                 sequential_batch=true, population=false, gain=true)

Check that the archives built by `make_archive()` meet the contract of
[`AbstractArchive`](@ref QualityDiversity.AbstractArchive). `make_archive` is called without arguments and must
return a new, empty archive each time. `prob` supplies solutions and
evaluations; its descriptors must lie inside the archive's bounds. A stochastic
problem is evaluated with a fixed seed.

The checks cover an empty archive, the status and `improvement` of each kind
of insertion, `length`, `capacity`, `cell_index` (on an archive holding all `n`
solutions), `sample_elites`, `elites`,
`add_batch!`, and `empty!` when the archive defines it. `elites` must return the
phenotype that came with each stored genotype, of the archive's phenotype type. A
fitness that is `NaN` or infinite must throw an `ArgumentError` from `add!` and from
`add_batch!` before the archive changes. With
`check_allocations`, `cell_index` must not allocate, and neither must `add!`
when the genotype type is `isbits`.

An archive is `elitist` if a cell only accepts a solution with higher fitness
than its occupant, so that adding the same solution again, or a worse one, is
`Rejected`. An archive whose cells accept solutions below the occupant, such as a
[`ThresholdArchive`](@ref QualityDiversity.ThresholdArchive) with a finite `threshold_min` and a `learning_rate`
below 1, passes `elitist=false`. The checks then require only that a status
agrees with the sign of `improvement`: `Rejected` for zero or less, `Improved` for
more than zero.

By default `add_batch!` must give the results of `add!` on each solution in order.
An archive that computes the results of a batch against its state before the
batch, such as a [`GridArchive`](@ref QualityDiversity.GridArchive) or a [`ThresholdArchive`](@ref QualityDiversity.ThresholdArchive), passes
`sequential_batch=false`. The checks then require that the results are valid and
that the result of each solution equals that of adding the solution alone to the
archive as it was before the batch.

An archive that chooses its whole population again from the stored solutions and the
batch, such as a [`DNSArchive`](@ref QualityDiversity.DNSArchive), passes `population=true`. A solution then has no
cell to improve, so the checks require only that each result is `NewCell` or
`Rejected`, that the solutions of a batch added to an empty archive that are `NewCell`
are the ones it stores, and that the archive never holds more than its capacity.

By default `improvement` is a gain in fitness: positive for an `Improved` solution and
zero or less for a `Rejected` one. An archive that reports another quantity there, such
as a [`DensityArchive`](@ref QualityDiversity.DensityArchive), whose `improvement` is the negated density, passes
`gain=false`, and the checks ignore the value of `improvement`.
"""
function test_archive(make_archive, prob::QD.AbstractQDProblem;
                      rng::AbstractRNG=Xoshiro(1), n::Integer=200,
                      check_allocations::Bool=true, elitist::Bool=true,
                      sequential_batch::Bool=true, population::Bool=false, gain::Bool=true)
    archive = make_archive()
    A = typeof(archive)
    G, F, D, P = _archive_types(archive)
    @testset "test_archive($(nameof(A)))" begin
        @testset "types" begin
            @test archive isa QD.AbstractArchive{G,F,D,P}
            @test isconcretetype(A)
            @test QD.capacity(archive) isa Integer
            @test QD.capacity(archive) >= 1
        end

        @testset "empty archive" begin
            @test length(archive) == 0
            @test isempty(collect(QD.elites(archive)))
        end

        xs = [QD.random_solution(rng, prob) for _ in 1:n]
        evs = QD.evaluate_batch(Xoshiro(2), prob, xs)

        @testset "cell_index" begin
            # on a filled archive: one without a fixed partition has no cell to
            # point to while it is empty
            a = make_archive()
            QD.add_batch!(a, xs, evs)
            for ev in evs
                i = @inferred QD.cell_index(a, ev.descriptor)
                @test i isa Integer
                @test 1 <= i <= QD.capacity(a)
                @test QD.cell_index(a, ev.descriptor) == i
            end
            if check_allocations
                @test allocations(QD.cell_index, a, evs[1].descriptor) == 0
            end
        end

        @testset "add!" begin
            a = make_archive()
            ev = evs[1]
            r = @inferred QD.add!(a, xs[1], ev)
            @test r isa QD.AddResult{F}
            @test r.status == QD.NewCell
            @test length(a) == 1
            if population
                for j in 2:min(n, QD.capacity(a) + 2)
                    r = QD.add!(a, xs[j], evs[j])
                    @test r.status in (QD.NewCell, QD.Rejected)
                    @test length(a) == min(j, QD.capacity(a))
                end
            else
                r = QD.add!(a, xs[1], ev)
                elitist ? (@test r.status == QD.Rejected) : (@test r.status in (QD.Rejected, QD.Improved))
                gain && @test (r.status == QD.Rejected) == (r.improvement <= 0)
                @test length(a) == 1

                worse = QD.Evaluation(_shift_fitness(ev.fitness, -1), ev.descriptor, ev.phenotype)
                r = QD.add!(a, xs[1], worse)
                elitist ? (@test r.status == QD.Rejected) : (@test r.status in (QD.Rejected, QD.Improved))
                gain && @test (r.status == QD.Rejected) == (r.improvement <= 0)
                gain && elitist && @test r.improvement < 0

                better = QD.Evaluation(_shift_fitness(ev.fitness, 1), ev.descriptor, ev.phenotype)
                r = QD.add!(a, xs[2], better)
                @test r.status == QD.Improved
                gain && @test r.improvement > 0
                @test length(a) == 1
                @test only(collect(QD.elites(a)))[1] == convert(G, xs[2])
                @test isequal(only(collect(QD.elites(a)))[2].phenotype, ev.phenotype)
            end

            if check_allocations && isbitstype(G)
                @test allocations(QD.add!, a, xs[1], ev) == 0
            end
        end

        @testset "add_batch!" begin
            a = make_archive()
            b = make_archive()
            rs = QD.add_batch!(a, xs, evs)
            @test rs isa AbstractVector{<:QD.AddResult}
            @test length(rs) == n
            @test 1 <= length(a) <= QD.capacity(a)
            new_cells = count(r -> r.status == QD.NewCell, rs)
            if population
                @test all(r -> r.status in (QD.Rejected, QD.NewCell), rs)
                @test length(a) == new_cells == min(n, QD.capacity(a))
                kept = [xs[j] for j in eachindex(xs) if rs[j].status == QD.NewCell]
                @test all(p -> any(x -> convert(G, x) == p[1], kept), collect(QD.elites(a)))
            elseif sequential_batch
                @test rs == [QD.add!(b, x, ev) for (x, ev) in zip(xs, evs)]
                @test length(a) == length(b)
                # every solution that fills a cell stays, unless a full archive evicts one
                @test length(a) < QD.capacity(a) ? new_cells == length(a) : new_cells >= length(a)
            else
                @test all(r -> r.status in (QD.Rejected, QD.Improved, QD.NewCell), rs)
                @test new_cells >= length(a)       # solutions that share a new cell all report it
                # each result is that of the solution added alone to the archive before the batch
                half = n ÷ 2
                before = make_archive()
                QD.add_batch!(before, xs[1:half], evs[1:half])
                batch = QD.add_batch!(deepcopy(before), xs[half+1:n], evs[half+1:n])
                @test all(k -> batch[k] == QD.add!(deepcopy(before), xs[half+k], evs[half+k]),
                          1:min(25, n - half))
            end
        end

        @testset "fitness that is not finite" begin
            a = make_archive()
            QD.add!(a, xs[1], evs[1])
            stored = collect(QD.elites(a))
            for bad in (NaN, Inf, -Inf)
                ev = QD.Evaluation(bad, evs[2].descriptor, evs[2].phenotype)
                @test_throws ArgumentError QD.add!(a, xs[2], ev)
                @test_throws ArgumentError QD.add_batch!(a, xs[2:3], [evs[3], ev])
                @test isequal(collect(QD.elites(a)), stored)
            end
        end

        @testset "result_archive" begin
            a = make_archive()
            QD.add_batch!(a, xs, evs)
            r = QD.result_archive(a)
            @test r isa QD.AbstractArchive{G,F,D,P}
            @test length(r) >= length(a)
            @test QD.max_fitness(r) >= QD.max_fitness(a)
            @test QD.result_archive(r) === r
        end

        @testset "elites and sample_elites" begin
            a = make_archive()
            QD.add_batch!(a, xs, evs)
            stored = collect(QD.elites(a))
            @test length(stored) == length(a)
            @test all(p -> p[1] isa G && p[2] isa QD.Evaluation && p[2].phenotype isa P, stored)
            @test all(p -> _stored_with(G, p, xs, evs), stored)
            @test all(p -> QD.cell_index(a, p[2].descriptor) in 1:QD.capacity(a), stored)
            @test length(unique(QD.cell_index(a, p[2].descriptor) for p in stored)) == length(stored)

            parents = @inferred QD.sample_elites(Xoshiro(3), a, 25)
            @test parents isa Vector{G}
            @test length(parents) == 25
            @test all(p -> any(q -> q[1] == p, stored), parents)
            @test QD.sample_elites(Xoshiro(3), a, 25) == parents
        end

        @testset "empty!" begin
            if hasmethod(empty!, Tuple{A})
                a = make_archive()
                QD.add_batch!(a, xs, evs)
                empty!(a)
                @test length(a) == 0
                @test isempty(collect(QD.elites(a)))
                @test QD.add!(a, xs[1], evs[1]).status == QD.NewCell
            end
        end
    end
    return nothing
end

"""
    test_front_archive(make_archive, prob; rng=Xoshiro(1), n=200, check_allocations=true)

Check that the archives built by `make_archive()` meet the contract of an archive that keeps
a Pareto front in every cell, such as a [`ParetoGridArchive`](@ref QualityDiversity.ParetoGridArchive). `prob` is a problem with
a vector fitness whose solutions are evaluated for the checks, and `make_archive()` returns an
empty archive for it whose reference point lies below the fitness values of `prob`. The
fitness type of the archive must be a vector of reals.

The result of `add!` must be `NewCell` for an empty cell, `Rejected` with zero `improvement`
for a solution that a stored one matches in every objective, and `Improved` with a positive
`improvement` for a solution that beats the front. Every cell must hold a front of solutions
none of which beats another, `length` counts the occupied cells and `num_solutions` the
solutions that `elites` yields. `add_batch!` must give each solution the result that it has
alone against the archive as it was before the batch, and leave the archive as adding the
solutions one at a time does. An objective that is `NaN` or infinite must throw an
`ArgumentError` from `add!` and from `add_batch!` before the archive changes. `sample_elites` must draw stored
genotypes, inferrably, and reproducibly from the generator. `elites` must return the
phenotype that came with each stored genotype. The metrics must agree with the
hypervolume of the fronts. `add!` must not allocate for a rejected solution with an `isbits`
genotype and two objectives when `check_allocations` is true.
"""
function test_front_archive(make_archive, prob::QD.AbstractQDProblem; rng::AbstractRNG=Xoshiro(1),
                            n::Integer=200, check_allocations::Bool=true)
    archive = make_archive()
    A = typeof(archive)
    G, F, D, P = _archive_types(archive)
    @testset "test_front_archive($(nameof(A)))" begin
        @testset "types" begin
            @test archive isa QD.AbstractArchive{G,F,D,P}
            @test isconcretetype(A)
            @test F <: AbstractVector{<:Real}
            @test QD.capacity(archive) >= 1
        end

        @testset "empty archive" begin
            @test length(archive) == 0 && QD.num_solutions(archive) == 0
            @test isempty(collect(QD.elites(archive)))
            @test QD.qd_score(archive) == 0
            @test QD.coverage(archive) == 0
            @test_throws "cannot sample elites from an empty archive" QD.sample_elites(Xoshiro(1), archive, 1)
        end

        xs = [QD.random_solution(rng, prob) for _ in 1:n]
        evs = QD.evaluate_batch(Xoshiro(2), prob, xs)

        @testset "cell_index" begin
            for ev in evs
                i = @inferred QD.cell_index(archive, ev.descriptor)
                @test 1 <= i <= QD.capacity(archive)
            end
            if check_allocations
                @test allocations(QD.cell_index, archive, evs[1].descriptor) == 0
            end
        end

        @testset "add!" begin
            a = make_archive()
            ev = evs[1]
            r = @inferred QD.add!(a, xs[1], ev)
            @test r.status == QD.NewCell && r.improvement > 0
            @test length(a) == 1 && QD.num_solutions(a) == 1

            # a copy adds no hypervolume, whether or not the archive keeps it
            copy = QD.add!(a, xs[1], ev)
            @test copy.improvement == 0
            worse = QD.Evaluation(_shift_fitness(ev.fitness, -1), ev.descriptor, ev.phenotype)
            r = QD.add!(a, xs[2], worse)
            @test r.status == QD.Rejected && r.improvement == 0
            @test QD.num_solutions(a) == (copy.status == QD.Rejected ? 1 : 2)

            better = QD.Evaluation(_shift_fitness(ev.fitness, 1), ev.descriptor, ev.phenotype)
            r = QD.add!(a, xs[3], better)
            @test r.status == QD.Improved && r.improvement > 0
            # the new solution beats the old one in every objective and replaces it
            @test QD.num_solutions(a) == 1
            @test only(collect(QD.elites(a)))[1] == convert(G, xs[3])

            if check_allocations && isbitstype(G)
                @test allocations(QD.add!, a, xs[3], better) == 0
            end
        end

        @testset "fronts" begin
            a = make_archive()
            for (x, ev) in zip(xs, evs)
                QD.add!(a, x, ev)
            end
            stored = collect(QD.elites(a))
            @test length(stored) == QD.num_solutions(a)
            cells = Dict{Int,Vector{F}}()
            for (_, ev) in stored
                push!(get!(cells, QD.cell_index(a, ev.descriptor), F[]), ev.fitness)
            end
            @test length(cells) == length(a)
            for front in values(cells)
                @test all(f -> !any(g -> g !== f && all(g .>= f), front), front)
            end
            @test QD.qd_score(a) ≈ sum(QD.hypervolume(front, QD.reference_point(a)) for front in values(cells))
            @test QD.coverage(a) == length(a) / QD.capacity(a)
            @test QD.max_hypervolume(a) ≈ maximum(QD.hypervolume(front, QD.reference_point(a)) for front in values(cells))
            @test QD.global_hypervolume(a) >= QD.max_hypervolume(a) - 1e-9
        end

        @testset "add_batch!" begin
            a = make_archive()
            half = n ÷ 2
            QD.add_batch!(a, xs[1:half], evs[1:half])
            batch = QD.add_batch!(deepcopy(a), xs[half+1:n], evs[half+1:n])
            @test batch isa AbstractVector{<:QD.AddResult}
            @test all(k -> batch[k] == QD.add!(deepcopy(a), xs[half+k], evs[half+k]), 1:min(25, n - half))
            together = deepcopy(a)
            QD.add_batch!(together, xs[half+1:n], evs[half+1:n])
            one_by_one = deepcopy(a)
            foreach((x, ev) -> QD.add!(one_by_one, x, ev), xs[half+1:n], evs[half+1:n])
            @test isequal(collect(QD.elites(together)), collect(QD.elites(one_by_one)))
        end

        @testset "objectives that are not finite" begin
            a = make_archive()
            QD.add!(a, xs[1], evs[1])
            stored = collect(QD.elites(a))
            for bad in (NaN, Inf, -Inf)
                ev = QD.Evaluation([bad; evs[2].fitness[2:end]], evs[2].descriptor, evs[2].phenotype)
                @test_throws ArgumentError QD.add!(a, xs[2], ev)
                @test_throws ArgumentError QD.add_batch!(a, xs[2:3], [evs[3], ev])
                @test isequal(collect(QD.elites(a)), stored)
            end
        end

        @testset "result_archive" begin
            @test QD.result_archive(archive) === archive
        end

        @testset "elites and sample_elites" begin
            a = make_archive()
            QD.add_batch!(a, xs, evs)
            stored = collect(QD.elites(a))
            @test all(p -> p[1] isa G && p[2] isa QD.Evaluation && p[2].phenotype isa P, stored)
            @test all(p -> _stored_with(G, p, xs, evs), stored)
            parents = @inferred QD.sample_elites(Xoshiro(3), a, 25)
            @test parents isa Vector{G}
            @test all(p -> any(q -> q[1] == p, stored), parents)
            @test QD.sample_elites(Xoshiro(3), a, 25) == parents
        end

        @testset "empty!" begin
            a = make_archive()
            QD.add_batch!(a, xs, evs)
            empty!(a)
            @test length(a) == 0 && QD.num_solutions(a) == 0 && QD.qd_score(a) == 0
            @test isempty(collect(QD.elites(a)))
            @test QD.add!(a, xs[1], evs[1]).status == QD.NewCell
        end
    end
    return nothing
end

_archive_types(::QD.AbstractArchive{G,F,D,P}) where {G,F,D,P} = (G, F, D, P)

# The stored pair `(x, ev)` has the phenotype of an evaluation of `x` in `evs`; the
# archive holds `x` converted to its genotype type `G`.
_stored_with(::Type{G}, (x, ev), xs, evs) where {G} =
    any(k -> convert(G, xs[k]) == x && isequal(evs[k].phenotype, ev.phenotype), eachindex(xs, evs))

_shift_fitness(f::Real, by) = f + by
_shift_fitness(f::AbstractVector, by) = f .+ by

"""
    test_variation(op, parent; parent2=nothing, rng=Xoshiro(1))

Check that the variation operator `op` meets the contract of
[`AbstractVariation`](@ref QualityDiversity.AbstractVariation) for the genotype `parent`. Without `parent2` the
operator is a mutation, `vary(rng, op, parent)`, and `n_parents(op)` must be
`1`; with it, a crossover, `vary(rng, op, parent, parent2)`, and `n_parents(op)`
must be `2`.

The call must be inferrable, leave its parents unchanged, and give the same
child for the same RNG state. A new child must not be the parent object.
"""
function test_variation(op::QD.AbstractVariation, parent;
                        parent2=nothing, rng::AbstractRNG=Xoshiro(1))
    @testset "test_variation($(nameof(typeof(op))))" begin
        parents = parent2 === nothing ? (parent,) : (parent, parent2)
        @test QD.n_parents(op) == length(parents)
        saved = map(deepcopy, parents)
        child = @inferred QD.vary(Xoshiro(5), op, parents...)
        @test parents == saved
        @test QD.vary(Xoshiro(5), op, parents...) == child
        @test !any(p -> child === p, parents)
        @test typeof(QD.vary(Xoshiro(6), op, parents...)) == typeof(child)
    end
    return nothing
end

# Evaluates a batch and adds it to `archive`, with the gradients that a `GradientArchive`
# keeps.
function evaluate_and_add(prob, archive, xs)
    evs = QD.evaluate_batch(Xoshiro(2), prob, xs)
    return evs, QD.add_batch!(archive, xs, evs)
end

function evaluate_and_add(prob, archive::QD.GradientArchive, xs)
    evs, jacobians = QD.evaluate_gradient_batch(prob, xs)
    return evs, QD.add_batch!(archive, xs, evs, jacobians)
end

# The type of the batches that the emitters of `archive` return.
batch_type(::QD.AbstractArchive{G}) where {G} = Vector{G}
batch_type(::QD.DeviceCellArchive) = QD.DeviceBatch

"""
    test_emitter(emitter, archive, prob; rng=Xoshiro(1))

Check that `emitter` meets the contract of [`AbstractEmitter`](@ref QualityDiversity.AbstractEmitter). `archive`
must be an archive for `prob` that already holds at least one solution. The
check modifies both `emitter` and `archive`.

`batch_size` must be a positive integer, and `ask` must return that many
genotypes of the archive's genotype type, inferrably, as a `Vector`, or as a
[`DeviceBatch`](@ref QualityDiversity.DeviceBatch) for a
[`DeviceCellArchive`](@ref QualityDiversity.DeviceCellArchive), none of them an object
the archive stores. The batch must then pass through `evaluate_batch`, `add_batch!` and
`tell!`.

An emitter whose [`ask_gradients`](@ref QualityDiversity.ask_gradients) returns solutions
first gets them back through `evaluate_gradient_batch`, `add_batch!` and
`tell_gradients!`, as in `step!`. For a
[`GradientArchive`](@ref QualityDiversity.GradientArchive) every batch is evaluated with
its gradients.
"""
function test_emitter(emitter::QD.AbstractEmitter, archive::QD.AbstractArchive{G},
                      prob::QD.AbstractQDProblem; rng::AbstractRNG=Xoshiro(1)) where {G}
    @testset "test_emitter($(nameof(typeof(emitter))))" begin
        @test length(archive) >= 1
        @test QD.batch_size(emitter) isa Integer
        @test QD.batch_size(emitter) >= 1

        points = QD.ask_gradients(Xoshiro(4), emitter, archive)
        if points !== nothing
            @testset "tell_gradients!" begin
                @test points isa batch_type(archive)
                evs, jacobians = QD.evaluate_gradient_batch(prob, points)
                results = QD.add_evaluated!(archive, points, evs, jacobians)
                @test QD.tell_gradients!(emitter, points, evs, jacobians, results) === nothing
            end
        end

        xs = @inferred QD.ask(Xoshiro(5), emitter, archive)
        @test xs isa batch_type(archive)
        @test length(xs) == QD.batch_size(emitter)
        if ismutabletype(G) && xs isa Vector
            # The archive stores by reference, so a candidate must be a new object.
            stored = Set{UInt}(objectid(g) for (g, _) in QD.elites(archive))
            @test !any(x -> objectid(x) in stored, xs)
        end

        @testset "tell!" begin
            evs, results = evaluate_and_add(prob, archive, xs)
            @test QD.tell!(emitter, xs, evs, results) === nothing
        end
    end
    return nothing
end

end
