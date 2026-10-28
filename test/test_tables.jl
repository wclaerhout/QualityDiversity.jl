using DataFrames: DataFrame, nrow, names
using Tables: Tables

const TablesExt = Base.get_extension(QualityDiversity, :QualityDiversityTablesExt)

# An archive whose descriptor type does not say how many dimensions it has.
struct VectorDescriptorArchive <: QD.AbstractArchive{Vector{Float64},Float64,Vector{Float64},Nothing} end
QD.elites(::VectorDescriptorArchive) = ()

@testset "extension loads" begin
    @test TablesExt isa Module
    @test Tables.istable(GridArchive)
    @test Tables.rowaccess(GridArchive)
    @test Tables.istable(UnstructuredArchive{Vector{Float64},Float64,2,Nothing})
end

@testset "GridArchive" begin
    archive = GridArchive((3, 2), ((0, 3), (10, 12)))
    QD.add!(archive, [1.0, 2.0], Evaluation(1.5, SVector(0.5, 10.5)))
    QD.add!(archive, [3.0, 4.0], Evaluation(-2.0, SVector(2.5, 11.5)))
    df = DataFrame(archive)
    @test names(df) == ["genotype", "fitness", "descriptor_1", "descriptor_2"]
    @test eltype.(eachcol(df)) == [Vector{Float64}, Float64, Float64, Float64]
    @test nrow(df) == length(archive) == 2
    # one row per elite, in the order of `elites`
    @test df.genotype == [x for (x, _) in QD.elites(archive)]
    @test df.fitness == [ev.fitness for (_, ev) in QD.elites(archive)]
    @test df.descriptor_1 == [ev.descriptor[1] for (_, ev) in QD.elites(archive)]
    @test df.descriptor_2 == [ev.descriptor[2] for (_, ev) in QD.elites(archive)]
    # copies: changing a genotype of the table leaves the archive as it is
    i = QD.cell_index(archive, SVector(0.5, 10.5))
    @test df.genotype[1] == archive.genotypes[i] && df.genotype[1] !== archive.genotypes[i]
    df.genotype[1][1] = 99.0
    @test archive.genotypes[i] == [1.0, 2.0]

    names_ = (:genotype, :fitness, :descriptor_1, :descriptor_2)
    types = Tuple{Vector{Float64},Float64,Float64,Float64}
    @test Tables.schema(archive) === Tables.Schema{names_,types}()
    @test Tables.columntable(archive) == (genotype=[[1.0, 2.0], [3.0, 4.0]], fitness=[1.5, -2.0],
                                          descriptor_1=[0.5, 2.5], descriptor_2=[10.5, 11.5])
    @test Tables.rowtable(archive)[1] == (genotype=[1.0, 2.0], fitness=1.5, descriptor_1=0.5, descriptor_2=10.5)
    rows = Tables.rows(archive)
    @test Tables.schema(rows) === Tables.schema(archive)
    @test eltype(rows) === NamedTuple{names_,types}
    @test length(rows) == 2
    row, state = @inferred Nothing iterate(rows)
    @test row == first(Tables.rowtable(archive))
end

@testset "fitness type" begin
    archive = GridArchive{Vector{Float64},Float32}((2, 2), ((0, 1), (0, 1)))
    QD.add!(archive, [0.0], Evaluation(0.25f0, SVector(0.1, 0.1)))
    df = DataFrame(archive)
    @test eltype(df.fitness) === Float32 && df.fitness == [0.25f0]
end

@testset "phenotype column" begin
    archive = GridArchive((2, 2), ((0, 1), (0, 1)); phenotype=String)
    QD.add!(archive, [0.0], Evaluation(1.0, SVector(0.1, 0.1), "CCO"))
    QD.add!(archive, [1.0], Evaluation(2.0, SVector(0.9, 0.9), "CCC"))
    df = DataFrame(archive)
    @test names(df) == ["genotype", "fitness", "descriptor_1", "descriptor_2", "phenotype"]
    @test eltype(df.phenotype) === String
    @test df.phenotype == ["CCO", "CCC"]
    @test Tables.rowtable(archive)[2].phenotype == "CCC"
end

@testset "copies of phenotypes" begin
    archive = GridArchive((2, 2), ((0, 1), (0, 1)); phenotype=Vector{Int})
    QD.add!(archive, [0.0], Evaluation(1.0, SVector(0.1, 0.1), [1, 2]))
    row = first(Tables.rowtable(archive))
    push!(row.phenotype, 3)
    @test only(QD.elites(archive))[2].phenotype == [1, 2]
end

@testset "ThresholdArchive shows its result archive" begin
    archive = ThresholdArchive(GridArchive((2, 2), ((0, 1), (0, 1))); learning_rate=0.5, threshold_min=0.0)
    QD.add!(archive, [1.0], Evaluation(10.0, SVector(0.1, 0.1)))
    QD.add!(archive, [2.0], Evaluation(6.0, SVector(0.1, 0.1)))    # passes the threshold 5
    @test [ev.fitness for (_, ev) in QD.elites(archive)] == [6.0]
    df = DataFrame(archive)
    @test df.fitness == [10.0] && df.genotype == [[1.0]]
    @test Tables.columntable(archive) == Tables.columntable(QD.result_archive(archive))
end

@testset "CategoricalArchive" begin
    archive = CategoricalArchive((0:5, [:alkane, :aromatic, :ring]); phenotype=Int)
    QD.add!(archive, [0.0], Evaluation(1.5, (3, :ring), 7))
    QD.add!(archive, [1.0], Evaluation(0.5, (1, :alkane), 8))
    df = DataFrame(archive)
    @test names(df) == ["genotype", "fitness", "descriptor_1", "descriptor_2", "phenotype"]
    @test eltype.(eachcol(df)) == [Vector{Float64}, Float64, Int, Symbol, Int]
    @test nrow(df) == length(archive) == 2
    @test df.descriptor_1 == [3, 1] && df.descriptor_2 == [:ring, :alkane] && df.phenotype == [7, 8]
end

@testset "ParetoGridArchive" begin
    archive = ParetoGridArchive{Vector{Float64}}((2, 2), ((0, 1), (0, 1)); reference_point=[0.0, 0.0])
    QD.add!(archive, [0.0], Evaluation(SVector(1.0, 3.0), SVector(0.2, 0.7)))
    QD.add!(archive, [1.0], Evaluation(SVector(3.0, 1.0), SVector(0.2, 0.7)))
    QD.add!(archive, [2.0], Evaluation(SVector(2.0, 2.0), SVector(0.8, 0.2)))
    df = DataFrame(archive)
    @test names(df) == ["genotype", "fitness", "descriptor_1", "descriptor_2"]
    @test nrow(df) == QD.n_solutions(archive) == 3 && length(archive) == 2
    @test eltype(df.fitness) === SVector{2,Float64}
    @test df.fitness == [ev.fitness for (_, ev) in QD.elites(archive)]
    @test df.genotype == [x for (x, _) in QD.elites(archive)]
    @test Base.IteratorSize(typeof(Tables.rows(archive))) === Base.SizeUnknown()
end

@testset "empty archives" begin
    for archive in (GridArchive((2, 2), ((0, 1), (0, 1))),
                    UnstructuredArchive(3; capacity=4, novelty_threshold=0.1),
                    CategoricalArchive((1:2, [:a, :b])),
                    ParetoGridArchive{Vector{Float64}}((2, 2), ((0, 1), (0, 1)); reference_point=[0.0, 0.0]))
        df = DataFrame(archive)
        @test nrow(df) == 0
        @test names(df) == ["genotype"; "fitness"; ["descriptor_$i" for i in 1:length(Tables.schema(archive).names)-2]]
        @test eltype(df.genotype) === Vector{Float64}
        @test eltype(df.descriptor_1) === (archive isa CategoricalArchive ? Int : Float64)
        @test isempty(Tables.rowtable(archive))
    end
    @test eltype(DataFrame(CategoricalArchive((1:2, [:a, :b]))).descriptor_2) === Symbol
end

@testset "other archives" begin
    prob = Rastrigin(4)
    for alg in (MAPElites(prob; dims=(4, 4), batch_size=16),
                MAPElites(SlidingBoundariesArchive(prob; dims=(4, 4), remap_frequency=50); batch_size=16),
                MAPElites(UnstructuredArchive(prob; capacity=30, novelty_threshold=0.2); batch_size=16),
                MAPElites(DNSArchive(prob; capacity=30, k=3); batch_size=16),
                CMAMAE(prob; dims=(4, 4), sigma0=0.5, threshold_min=-500.0, learning_rate=0.1))
        archive = solve(prob, alg; budget=200, rng=Xoshiro(1)).archive
        df = DataFrame(archive)
        @test nrow(df) == length(archive) > 0
        @test all(x -> length(x) == 4, df.genotype)
        @test df.fitness == [ev.fitness for (_, ev) in QD.elites(archive)]
    end
end

@testset "descriptor without a static length" begin
    @test_throws "does not say how many there are" Tables.rows(VectorDescriptorArchive())
    @test_throws "does not say how many there are" DataFrame(VectorDescriptorArchive())
end
