```@meta
CurrentModule = QualityDiversity
```

# Archives

An archive stores the solutions found so far, and decides which of them to keep.

## Interface

```@docs
AbstractArchive
QualityDiversity.AddStatus
QualityDiversity.AddResult
QualityDiversity.add!
QualityDiversity.add_batch!
QualityDiversity.cell_index
QualityDiversity.sample_elites
QualityDiversity.elites
QualityDiversity.capacity
QualityDiversity.result_archive
```

## Cell archives

```@docs
QualityDiversity.CellArchive
GridArchive
SlidingBoundariesArchive
QualityDiversity.boundaries
CVTArchive
QualityDiversity.centroids
CategoricalArchive
```

## Seeding the k-means of a CVT archive

```@docs
QualityDiversity.CentroidSeeding
QualityDiversity.KMeansPlusPlus
QualityDiversity.GreedyKMeansPlusPlus
```

## Descriptors outside the bounds

```@docs
QualityDiversity.OutsidePolicy
ThrowOutside
ClampOutside
RejectOutside
```

## Archives around a cell archive

```@docs
ThresholdArchive
QualityDiversity.thresholds
SpreadArchive
QualityDiversity.spreads
DensityArchive
QualityDiversity.density
GradientArchive
```

## Archives without cells

```@docs
UnstructuredArchive
DNSArchive
```

## Tables

Loading Tables.jl (for example through DataFrames or CSV) makes every archive a table
with one row per elite: `DataFrame(result.archive)` has the columns `genotype`,
`fitness`, `descriptor_1` to `descriptor_N` and, for an archive that stores one,
`phenotype`.

```@docs
Main.QualityDiversityTablesExt
```

## Several objectives

```@docs
ParetoGridArchive
QualityDiversity.n_solutions
QualityDiversity.DuplicatePolicy
QualityDiversity.RejectDuplicates
QualityDiversity.KeepDuplicates
QualityDiversity.FullFrontPolicy
QualityDiversity.RejectNewcomer
QualityDiversity.ReplaceCrowded
```
