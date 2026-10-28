"""
    Evaluation(fitness, descriptor)
    Evaluation(fitness, descriptor, phenotype)
    Evaluation{F,D,P}(fitness, descriptor, phenotype)

The result of evaluating one solution.

`fitness` is a `Real` for a single objective or an `AbstractVector` for
several. Fitness is maximized; a problem that minimizes returns the negated
value. Archives take only finite fitness values: a problem that wants to mark a
solution as infeasible returns a low finite value. `descriptor` locates the
solution in descriptor space and is usually an `SVector{N,Float64}`.

`phenotype` is what the problem computed from the genotype and described, when
the two differ: the product of a reaction sequence, the shape that a set of
assembly rules builds. An archive whose phenotype type is `P` stores it next to
the genotype, and [`elites`](@ref QualityDiversity.elites) returns it, so it
need not be computed again. Variation sees only genotypes. Without a phenotype
the field is `nothing` and `P` is `Nothing`, which takes no memory in an
archive.

# Example

```jldoctest
julia> using QualityDiversity, StaticArrays

julia> Evaluation(1.0, SVector(0.5, 0.5)).phenotype === nothing
true

julia> Evaluation(1.0, SVector(0.5, 0.5), "CCO").phenotype
"CCO"
```
"""
struct Evaluation{F,D,P}
    fitness::F
    descriptor::D
    phenotype::P
end

Evaluation(fitness, descriptor) = Evaluation(fitness, descriptor, nothing)

Evaluation{F,D}(fitness, descriptor) where {F,D} = Evaluation{F,D,Nothing}(fitness, descriptor, nothing)

# Field by field, so that evaluations holding equal vectors are equal.
Base.:(==)(a::Evaluation, b::Evaluation) =
    a.fitness == b.fitness && a.descriptor == b.descriptor && a.phenotype == b.phenotype
Base.isequal(a::Evaluation, b::Evaluation) =
    isequal(a.fitness, b.fitness) && isequal(a.descriptor, b.descriptor) && isequal(a.phenotype, b.phenotype)
Base.hash(ev::Evaluation, h::UInt) = hash(ev.phenotype, hash(ev.descriptor, hash(ev.fitness, hash(:Evaluation, h))))

"""
    AddStatus

Outcome of inserting a solution into an archive.

- `Rejected`: the archive left unchanged.
- `Improved`: the solution replaced a worse occupant of its cell.
- `NewCell`: the solution filled a cell that was empty.
"""
@enum AddStatus Rejected Improved NewCell

"""
    AddResult{F}(status, improvement)

Result of [`add!`](@ref QualityDiversity.add!).

`improvement` is the fitness gain over the previous occupant of the cell, or
over the archive's threshold for a cell that was empty. It is zero or
negative for a rejected solution, and zero for an unsigned fitness type, which
has no negative values. A solution the archive has no cell for, one that
`RejectOutside()` discards, gets the lowest improvement there is, `-Inf` for a
floating-point fitness type, so that a ranker places it after every solution that
has a cell. Emitters that rank candidates by improvement, such as CMA-ME, read this
field. The archives whose `improvement` means something else, such as
[`DensityArchive`](@ref) and [`DNSArchive`](@ref), say so.
"""
struct AddResult{F}
    status::AddStatus
    improvement::F
end

# The fitness of an evaluation as an archive with fitness type `F` stores it. A fitness that
# is NaN or infinite, or becomes infinite in `F`, throws before the archive changes: it
# would make a cell impossible to improve and the sums behind the metrics NaN.
function stored_fitness(::Type{F}, fitness::Real) where {F}
    isfinite(fitness) || fitness_error(F, fitness)
    f = convert(F, fitness)
    isfinite(f) || fitness_error(F, fitness)
    return f
end

# Out of line, so that the checks of `stored_fitness` stay cheap enough to inline.
@noinline function fitness_error(F, fitness)
    isfinite(fitness) &&
        throw(ArgumentError("fitness $fitness is too large for the fitness type $F of the archive"))
    throw(ArgumentError("fitness must be finite, got $fitness"))
end

# Throws for a real fitness that `stored_fitness` refuses. Another kind of fitness, such as
# a vector of objectives, is for its archive to check.
check_fitness(::Type{F}, fitness::Real) where {F} = (stored_fitness(F, fitness); nothing)
check_fitness(::Type{F}, fitness) where {F} = nothing

# How much `new` improves on `old`, negative if it is worse. An unsigned type has no
# negative values, so a worse fitness gains zero instead of wrapping around.
fitness_gain(new, old) = new - old
fitness_gain(new::Unsigned, old::Unsigned) = new >= old ? new - old : zero(new)

# The improvement of a solution the archive has no cell for, below that of any solution
# in a cell: `-Inf` for a floating-point fitness type and the smallest value otherwise,
# which is zero for an unsigned type.
outside_improvement(::Type{F}) where {F<:AbstractFloat} = -F(Inf)
outside_improvement(::Type{F}) where {F<:Integer} = typemin(F)
outside_improvement(::Type{F}) where {F} = zero(F)

# The result of a solution the archive has no cell for.
outside_result(::Type{F}) where {F} = AddResult(Rejected, outside_improvement(F))

# The phenotype of an evaluation as an archive with phenotype type `P` stores it. A
# mismatch throws before the archive changes.
function stored_phenotype(::Type{P}, phenotype) where {P}
    phenotype isa P && return phenotype
    # Julia cannot ask whether a conversion exists (`hasmethod` sees the generic fallbacks
    # of `convert`), so a missing one is caught here; any other error passes through.
    try
        return convert(P, phenotype)
    catch err
        err isa MethodError && err.f === convert || rethrow()
        throw(ArgumentError("the evaluation has a phenotype of type $(typeof(phenotype)), which does not convert to the phenotype type $P of the archive"))
    end
end

function stored_phenotype(::Type{P}, ::Nothing) where {P}
    Nothing <: P && return nothing
    throw(ArgumentError("the evaluation has no phenotype but the archive stores phenotypes of type $P; return one from the problem, or build the archive without phenotype"))
end

function stored_phenotype(::Type{Nothing}, phenotype)
    throw(ArgumentError("the evaluation has a phenotype of type $(typeof(phenotype)) but the archive stores none; build the archive with phenotype=$(typeof(phenotype))"))
end

stored_phenotype(::Type{Nothing}, ::Nothing) = nothing

# Writes a phenotype into the storage of an archive. A `Vector{Nothing}` holds only
# `nothing`, so the write is skipped: an archive without phenotypes then compiles to the
# code it would have without the field, bounds check included.
store_phenotype!(phenotypes::AbstractVector, i, phenotype) = (phenotypes[i] = phenotype; nothing)
store_phenotype!(::Vector{Nothing}, i, ::Nothing) = nothing
store_phenotype!(::Nothing, i, ::Nothing) = nothing

# The phenotypes of a batch as the archive stores them, converted once with
# `stored_phenotype` before the archive changes. An archive without phenotypes keeps no
# buffer, so that its batch methods allocate nothing for one.
batch_phenotypes(::Type{Nothing}, n) = nothing
batch_phenotypes(::Type{P}, n) where {P} = Vector{P}(undef, n)
batch_phenotype(::Nothing, k) = nothing
batch_phenotype(phenotypes::Vector, k) = phenotypes[k]

# Moves the phenotype in slot `from` to slot `to`.
move_phenotype!(phenotypes::AbstractVector, to, from) = (phenotypes[to] = phenotypes[from]; nothing)
move_phenotype!(::Vector{Nothing}, to, from) = nothing
