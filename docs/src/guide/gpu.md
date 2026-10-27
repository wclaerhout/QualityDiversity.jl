```@meta
CurrentModule = QualityDiversity
```

# Running on a GPU

Everything runs on the CPU unless you ask for a device, and the package does not
depend on any GPU package. There are two ways to use a GPU, depending on what is
expensive.

## Evaluating a batch on a GPU

When the evaluation is expensive and the rest of the run is cheap, keep the archive on
the CPU and evaluate each batch on the GPU. The loop calls
[`evaluate_batch`](@ref QualityDiversity.evaluate_batch) once per round, and a problem
type can define it: copy the batch to the device, compute with array operations or a
kernel, and copy the fitness and descriptors back.

```julia
function QualityDiversity.evaluate_batch(prob::MyProblem, xs::AbstractVector)
    X = CuArray(reduce(hcat, xs))                     # one column per solution
    fitness, descriptors = my_gpu_scoring(X)          # array code on the device
    f, d = Array(fitness), Array(descriptors)
    return [Evaluation(f[j], SVector(d[1, j], d[2, j])) for j in eachindex(f)]
end
```

`examples/gpu_evaluation.jl` does this for the Rastrigin function with any array type
(`CuArray`, `ROCArray`, `MtlArray`, `oneArray`), and is tested with JLArrays, an array
type that runs on the CPU.

## MAP-Elites and CMA-ME on a device

When evaluation is cheap, the time goes into the archive and the variation, and moving
a batch to the GPU and back every round costs more than it saves. Then the whole run
can stay on the device:

```julia
using CUDA, QualityDiversity, QualityDiversity.Benchmarks
using Random: Xoshiro

prob = Rastrigin(100)
alg = MAPElites(prob; dims=(100, 100), batch_size=10_000, device=CuArray,
                genotype=Vector{Float32}, fitness=Float32)
result = solve(prob, alg; budget=10_000_000, rng=Xoshiro(1))
```

With `device`, [`MAPElites`](@ref) builds a [`DeviceGridArchive`](@ref), whose cells are
in arrays of that type, and a [`DeviceVariationEmitter`](@ref QualityDiversity.DeviceVariationEmitter), which draws parents and
mutates them in a kernel. [`CMAME`](@ref) accepts `device` too and builds a
[`DeviceCMAEmitter`](@ref QualityDiversity.DeviceCMAEmitter), many CMA-ES searches whose candidates are evaluated and
added together. For a CVT archive, build a [`DeviceCVTArchive`](@ref) and pass it to
`MAPElites(archive)`.

- **Kernels.** The kernels are written with KernelAbstractions.jl, which CUDA.jl,
  AMDGPU.jl and oneAPI.jl load. `device=Array` runs the same kernels on the CPU after
  `using KernelAbstractions`, which is how the tests run them.
- **Evaluation.** By default the problem's `evaluate` runs in a kernel, one solution
  per thread, so it must compile for the device: no allocation and no error messages
  built from values. [`Rastrigin`](@ref QualityDiversity.Benchmarks.Rastrigin),
  [`Sphere`](@ref QualityDiversity.Benchmarks.Sphere) and a [`QDProblem`](@ref)
  with a simple function work. Otherwise define `evaluate_batch(prob,
  xs::DeviceBatch)` and work on `parent(xs)`, a matrix with one solution per column.
- **Precision.** On consumer GPUs use `Float32` solutions and fitness, and a problem
  that computes in `Float32`; see [Float32 on a GPU](@ref).
- **Reproducibility.** A run is reproducible from its seed on a given device, whatever
  the scheduling of the threads, but its random numbers differ from those of the CPU
  path.
- **Results.** `GridArchive(archive)` and `CVTArchive(archive)` copy a device archive
  to the CPU, for plotting and analysis.

The device archives support MAP-Elites with [`GaussianMutation`](@ref) and
[`IsoLineVariation`](@ref), and CMA-ME. Stochastic problems, phenotypes and the other
archives run on the CPU.

## Float32 on a GPU

Consumer GPUs compute `Float64` many times more slowly than `Float32`, so on them a run
should use `Float32` wherever the precision is enough. Four things have a precision,
and they are set in different places:

| What | Type on a consumer GPU | Set by |
|---|---|---|
| solutions | `Float32` | `genotype=Vector{Float32}` |
| fitness stored in the archive | `Float32` | `fitness=Float32` |
| the arithmetic of the evaluation | whatever the problem's code computes in | the problem |
| descriptors, bounds and the cell lookup | always `Float64` | the archive |

**Descriptors stay `Float64`.** A device archive and a CPU archive put every solution in
the same cell. That needs the descriptor at full precision: a descriptor just below a
cell boundary, such as `0.499999999` with a boundary at `0.5`, rounds to `0.5f0` in
`Float32` and would land in the next cell. Descriptors are a few numbers per solution,
so keeping them in `Float64` costs little. A GPU without `Float64`, such as an Apple GPU,
cannot run the device archives.

**The arithmetic is set by the problem, not by `genotype`.** With
`genotype=Vector{Float32}` the problem receives `Float32` solutions, but nothing makes
its code compute in `Float32`. A single `Float64` value, such as the literal `10.0` or
`5.12`, promotes the whole computation to `Float64`. The run is then correct but slow,
and nothing reports it, since the archive converts the fitness to `Float32` when it
stores it.

Write the scoring function in the element type of the solution:

```julia
using QualityDiversity

function rastrigin(x::AbstractVector{T}) where {T<:AbstractFloat}
    limit = T(5.12)                          # 5.12 alone is a Float64
    total = zero(T)
    for xi in x
        xi = clamp(xi, -limit, limit)
        total += xi^2 - 10 * cospi(2 * xi)   # integers keep the type T
    end
    h = length(x) ÷ 2
    first_half = sum(view(x, 1:h)) / h
    second_half = sum(view(x, h+1:length(x))) / (length(x) - h)
    return -(10 * length(x) + total), (first_half, second_half)
end

prob = QDProblem(rastrigin; init=rng -> 10 .* rand(rng, Float32, 20) .- 5,
                 bounds=((-5.12, 5.12), (-5.12, 5.12)))
```

Integers, `zero(T)`, `one(T)`, `T(...)` and `Float32` literals such as `5.12f0` keep the
type; `Float64` literals and `Float64(...)` do not.

Check the precision on the CPU before running on a GPU: one `Float32` solution must give
a `Float32` fitness.

```julia
x = rand(Float32, 20)
typeof(QualityDiversity.evaluate(prob, x).fitness)   # Float32; Float64 means the code promotes
```

The descriptor of that evaluation is an `SVector{2,Float64}` whatever the code computes
in, since descriptors are always `Float64`.

The built-in [`Rastrigin`](@ref QualityDiversity.Benchmarks.Rastrigin) and
[`Sphere`](@ref QualityDiversity.Benchmarks.Sphere) compute on a device in the element
type of the solution, so the example at the top of this page evaluates in `Float32`.
Their `evaluate` on the CPU computes in `Float64`, also with `Float32` solutions, so the
check above does not apply to them, and the same `Float32` solution can get a fitness on
the device that differs from the CPU one in the last digits.

## When it pays off

A GPU pays off for large batches. On the benchmark of the
[comparison](comparison.md), MAP-Elites with batches of 100 or fewer is faster on one
CPU core than on a GPU, and with batches of 10 000 the GPU is about 28 times faster in `float32`.
A device CMA-ME is slower than the CPU with the five searches of the default and
faster with hundreds.
