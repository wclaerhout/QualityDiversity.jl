"""
    StochasticQDProblem(f; init, bounds)
    StochasticQDProblem(f; init, categories)

Wrap a function of a random number generator and a solution as a stochastic problem.
`f(rng, x)` returns a `(fitness, descriptor)` or `(fitness, descriptor, phenotype)` tuple
for the solution `x` and draws every random number from `rng`. The other arguments are those of [`QDProblem`](@ref). See
[`AbstractStochasticQDProblem`](@ref) for how the generators are made, and note that
`f` is called from several threads at once.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> prob = StochasticQDProblem((rng, x) -> (-sum(abs2, x) + 0.1 * randn(rng), (x[1], x[2]));
                                  init = rng -> randn(rng, 2), bounds = ((-3, 3), (-3, 3)));

julia> a = QualityDiversity.evaluate(Xoshiro(1), prob, [1.0, 2.0]);

julia> a == QualityDiversity.evaluate(Xoshiro(1), prob, [1.0, 2.0])
true

julia> a.fitness == QualityDiversity.evaluate(Xoshiro(2), prob, [1.0, 2.0]).fitness
false
```
"""
struct StochasticQDProblem{F,I,S} <: AbstractStochasticQDProblem
    f::F
    init::I
    space::S
end

StochasticQDProblem(f; init, bounds=nothing, categories=nothing) =
    StochasticQDProblem(f, init, problem_space(bounds, categories))

evaluate(rng::AbstractRNG, prob::StochasticQDProblem, x) = evaluation_of(prob.space, prob.f(rng, x))

random_solution(rng::AbstractRNG, prob::StochasticQDProblem) = prob.init(rng)

descriptor_bounds(prob::StochasticQDProblem{F,I,<:Tuple}) where {F,I} = prob.space

descriptor_categories(prob::StochasticQDProblem{F,I,<:DescriptorCategories}) where {F,I} =
    prob.space.values

"""
    Expectation

Supertype of the rules that turn the results of several evaluations of one solution into
one: [`Average`](@ref) and [`Median`](@ref). A rule defines `expectation(rule, values)`
for a vector of real numbers.
"""
abstract type Expectation end

"""
    Average()

The mean of the evaluations. This is the default of [`Sampled`](@ref).
"""
struct Average <: Expectation end

"""
    Median()

The median of the evaluations, which an outlier moves less than it moves the mean. It is
taken for every descriptor dimension separately.
"""
struct Median <: Expectation end

"""
    expectation(rule::Expectation, values::AbstractVector{<:Real})

The number that `rule` makes of the evaluations `values` of one solution. A new rule
defines this method.
"""
function expectation end

expectation(::Average, values) = sum(values) / length(values)

function expectation(::Median, values)
    sorted = sort(values)
    n = length(sorted)
    middle = (n + 1) ÷ 2
    return isodd(n) ? sorted[middle] : (sorted[middle] + sorted[middle+1]) / 2
end

"""
    Sampled(prob; num_samples, expectation=Average())

Score every solution of the stochastic problem `prob` with `num_samples` evaluations and
return the [`Expectation`](@ref) of their fitness and, for every dimension, of their
descriptor. An archive then holds the averages and not the single lucky evaluation that a
noisy problem can give a poor solution.

`Sampled` is itself a stochastic problem; the evaluations of one solution run one after the
other with the generator of that solution, so the results are reproducible from the seed of
the run. Every evaluation counts against the budget of a run, so `num_samples` times fewer
solutions are scored for the same budget; see
[`evaluations_per_solution`](@ref QualityDiversity.evaluations_per_solution). The fitness
must be a real number and the descriptor a vector of real numbers. The descriptor bounds
and the fitness offset are those of `prob`.

If `prob` returns a phenotype, the result has the phenotype that occurs most often among
the evaluations, compared with `isequal`; on a tie, the one that occurs first. The
`expectation` applies to the fitness and the descriptor only. A phenotype type without its
own `isequal` and `hash` compares by identity, so every evaluation differs and the result has
the phenotype of the first.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> noisy = StochasticQDProblem((rng, x) -> (-x[1]^2 + randn(rng), (x[1], x[2]));
                                   init = rng -> randn(rng, 2), bounds = ((-3, 3), (-3, 3)));

julia> prob = Sampled(noisy; num_samples=100);

julia> ev = QualityDiversity.evaluate(Xoshiro(1), prob, [1.0, 0.5]);

julia> abs(ev.fitness + 1) < 0.5
true

julia> QualityDiversity.evaluations_per_solution(prob)
100
```
"""
struct Sampled{P<:AbstractStochasticQDProblem,E<:Expectation} <: AbstractStochasticQDProblem
    problem::P
    num_samples::Int
    expectation::E

    function Sampled{P,E}(problem::P, num_samples::Integer, expectation::E) where {P,E}
        num_samples >= 1 ||
            throw(ArgumentError("num_samples must be at least 1, got $num_samples"))
        return new{P,E}(problem, num_samples, expectation)
    end
end

function Sampled(problem::AbstractStochasticQDProblem; num_samples::Integer,
                 expectation::Expectation=Average())
    return Sampled{typeof(problem),typeof(expectation)}(problem, num_samples, expectation)
end

function Sampled(problem::AbstractQDProblem; kwargs...)
    throw(ArgumentError("Sampled needs a stochastic problem, but $(nameof(typeof(problem))) is deterministic: its evaluations would all be the same"))
end

function evaluate(rng::AbstractRNG, prob::Sampled, x)
    fitness, descriptors, phenotypes = sample_evaluations(rng, prob, x)
    return Evaluation(expectation(prob.expectation, fitness),
                      expected_descriptor(prob.expectation, descriptors),
                      most_frequent(phenotypes))
end

# The fitness values, descriptors and phenotypes of `prob.num_samples` evaluations of `x`
# by `prob.problem`, one after the other with `rng`.
function sample_evaluations(rng::AbstractRNG, prob, x)
    first = evaluate(rng, prob.problem, x)
    name = nameof(typeof(prob))
    first.fitness isa Real ||
        throw(ArgumentError("$name needs a real fitness, got a $(typeof(first.fitness))"))
    first.descriptor isa SVector{<:Any,<:Real} ||
        throw(ArgumentError("$name needs a descriptor that is an SVector of real numbers, got a $(typeof(first.descriptor))"))
    n = prob.num_samples
    # Floating-point containers: the samples of one solution can mix integers and floats.
    fitness = Vector{float(typeof(first.fitness))}(undef, n)
    descriptors = Vector{float_svector(first.descriptor)}(undef, n)
    phenotypes = phenotype_samples(first.phenotype, n)
    fitness[1] = first.fitness
    descriptors[1] = first.descriptor
    for i in 2:n
        ev = evaluate(rng, prob.problem, x)
        fitness[i] = ev.fitness
        descriptors[i] = ev.descriptor
        record_phenotype!(phenotypes, i, ev.phenotype)
    end
    return fitness, descriptors, phenotypes
end

float_svector(::SVector{N,T}) where {N,T} = SVector{N,float(T)}

# The phenotypes of the evaluations of one solution; `nothing` when the problem returns
# none, so that such a problem keeps no vector of them.
phenotype_samples(::Nothing, n) = nothing
function phenotype_samples(first, n)
    phenotypes = Vector{typeof(first)}(undef, n)
    phenotypes[1] = first
    return phenotypes
end

record_phenotype!(::Nothing, i, ::Nothing) = nothing
record_phenotype!(phenotypes, i, phenotype) = (phenotypes[i] = phenotype; nothing)

# The value that occurs most often in `values`, compared with `isequal`; the first one on
# a tie.
most_frequent(::Nothing) = nothing
function most_frequent(values::AbstractVector{T}) where {T}
    counts = Dict{T,Int}()
    for v in values
        counts[v] = get(counts, v, 0) + 1
    end
    best = first(values)
    for v in values
        counts[v] > counts[best] && (best = v)
    end
    return best
end

function expected_descriptor(rule, descriptors::Vector{SVector{N,T}}) where {N,T}
    column = Vector{T}(undef, length(descriptors))
    mean = ntuple(Val(N)) do j
        for i in eachindex(descriptors)
            column[i] = descriptors[i][j]
        end
        expectation(rule, column)
    end
    return SVector{N}(mean)
end

random_solution(rng::AbstractRNG, prob::Sampled) = random_solution(rng, prob.problem)

descriptor_bounds(prob::Sampled) = descriptor_bounds(prob.problem)

fitness_offset(prob::Sampled) = fitness_offset(prob.problem)

evaluations_per_solution(prob::Sampled) =
    prob.num_samples * evaluations_per_solution(prob.problem)

"""
    DescriptorSamples{N}(samples::Vector{SVector{N,Float64}})

The descriptors of the evaluations of one solution, in the order of the evaluations: the
descriptor of an [`Evaluation`](@ref) of a [`SampledSpread`](@ref) problem. It is an
`AbstractVector` whose entries are the descriptors. A [`SpreadArchive`](@ref) finds the cell
and the spread of the solution from them.
"""
struct DescriptorSamples{N} <: AbstractVector{SVector{N,Float64}}
    samples::Vector{SVector{N,Float64}}
end

Base.size(d::DescriptorSamples) = size(d.samples)
Base.getindex(d::DescriptorSamples, i::Int) = d.samples[i]
Base.IndexStyle(::Type{<:DescriptorSamples}) = IndexLinear()

"""
    SampledSpread(prob; num_samples, expectation=Average())

Score every solution of the stochastic problem `prob` with `num_samples` evaluations, as
[`Sampled`](@ref) does, but keep every descriptor: the evaluation has the
[`Expectation`](@ref) of the fitness, the descriptors of all the evaluations as a
[`DescriptorSamples`](@ref), and the phenotype that occurs most often. A
[`SpreadArchive`](@ref), the archive of [`MELS`](@ref), takes these evaluations, and only
it: the cell of a solution and its spread depend on the cells of the archive.

The evaluations of one solution run one after the other with the generator of that
solution, and each counts against the budget of a run. The fitness must be a real number
and the descriptor a vector of real numbers. The descriptor bounds and the fitness offset
are those of `prob`.

# Example

```jldoctest
julia> using QualityDiversity, Random

julia> noisy = StochasticQDProblem((rng, x) -> (-x[1]^2 + randn(rng), (x[1], x[2] + 0.1 * randn(rng)));
                                   init = rng -> randn(rng, 2), bounds = ((-3, 3), (-3, 3)));

julia> ev = QualityDiversity.evaluate(Xoshiro(1), SampledSpread(noisy; num_samples=5), [1.0, 0.5]);

julia> length(ev.descriptor), ev.descriptor[1][1]
(5, 1.0)
```
"""
struct SampledSpread{P<:AbstractStochasticQDProblem,E<:Expectation} <: AbstractStochasticQDProblem
    problem::P
    num_samples::Int
    expectation::E

    function SampledSpread{P,E}(problem::P, num_samples::Integer, expectation::E) where {P,E}
        num_samples >= 1 ||
            throw(ArgumentError("num_samples must be at least 1, got $num_samples"))
        return new{P,E}(problem, num_samples, expectation)
    end
end

function SampledSpread(problem::AbstractStochasticQDProblem; num_samples::Integer,
                       expectation::Expectation=Average())
    return SampledSpread{typeof(problem),typeof(expectation)}(problem, num_samples, expectation)
end

function SampledSpread(problem::AbstractQDProblem; kwargs...)
    throw(ArgumentError("SampledSpread needs a stochastic problem, but $(nameof(typeof(problem))) is deterministic: its evaluations would all be the same and its spread zero"))
end

function evaluate(rng::AbstractRNG, prob::SampledSpread, x)
    fitness, descriptors, phenotypes = sample_evaluations(rng, prob, x)
    return Evaluation(expectation(prob.expectation, fitness), descriptor_samples(descriptors),
                      most_frequent(phenotypes))
end

descriptor_samples(descriptors::Vector{SVector{N,Float64}}) where {N} = DescriptorSamples{N}(descriptors)
descriptor_samples(descriptors::Vector{<:SVector{N}}) where {N} =
    DescriptorSamples{N}(map(SVector{N,Float64}, descriptors))

random_solution(rng::AbstractRNG, prob::SampledSpread) = random_solution(rng, prob.problem)

descriptor_bounds(prob::SampledSpread) = descriptor_bounds(prob.problem)

fitness_offset(prob::SampledSpread) = fitness_offset(prob.problem)

evaluations_per_solution(prob::SampledSpread) =
    prob.num_samples * evaluations_per_solution(prob.problem)
