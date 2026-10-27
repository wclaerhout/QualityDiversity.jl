```@meta
CurrentModule = QualityDiversity
```

# Emitters

An emitter proposes a batch of candidates each round and may learn from what the
archive did with them.

## Interface

```@docs
AbstractEmitter
QualityDiversity.ask
QualityDiversity.tell!
QualityDiversity.batch_size
QualityDiversity.restarts
```

## Variation emitter

```@docs
VariationEmitter
```

## CMA emitter

```@docs
CMAEmitter
```

### Evolution strategies

```@docs
QualityDiversity.AbstractEvolutionStrategy
QualityDiversity.CMAES
QualityDiversity.PyribsCMAES
QualityDiversity.SepCMAES
QualityDiversity.LMMAES
QualityDiversity.OpenAIES
QualityDiversity.reset!
QualityDiversity.sample!
QualityDiversity.update!
QualityDiversity.should_stop
QualityDiversity.CMAVariant
QualityDiversity.TextbookCMA
QualityDiversity.PyribsCMA
```

### Rankers

```@docs
QualityDiversity.AbstractRanker
QualityDiversity.rank_key
QualityDiversity.reset_ranker!
QualityDiversity.ObjectiveRanker
QualityDiversity.ImprovementRanker
QualityDiversity.TwoStageImprovementRanker
QualityDiversity.TwoStageObjectiveRanker
QualityDiversity.DirectionRanker
QualityDiversity.RandomDirectionRanker
QualityDiversity.TwoStageRandomDirectionRanker
QualityDiversity.AcceptedRandomDirectionRanker
```

### Selection rules

```@docs
QualityDiversity.SelectionRule
QualityDiversity.FilterSelection
QualityDiversity.MuSelection
```

### Restart rules

```@docs
QualityDiversity.RestartRule
QualityDiversity.NoImprovementRestart
QualityDiversity.ConvergenceRestart
QualityDiversity.PeriodicRestart
```

## Gradient emitters

Emitters for differentiable quality diversity, which step along the gradients of the
fitness and the descriptor; see [`evaluate_gradient`](@ref QualityDiversity.evaluate_gradient).
An emitter that needs the gradients at points of its own asks for them before every
round.

```@docs
GradientArborescenceEmitter
GradientOperatorEmitter
QualityDiversity.ask_gradients
QualityDiversity.tell_gradients!
QualityDiversity.GradientOptimizer
QualityDiversity.AdamAscent
QualityDiversity.GradientAscent
```

## Bandit scheduler

```@docs
BanditScheduler
QualityDiversity.active
QualityDiversity.ReselectRule
QualityDiversity.ReselectTerminated
QualityDiversity.ReselectAll
```
