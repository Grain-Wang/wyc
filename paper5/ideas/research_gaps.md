# Research Gaps After the Core-Library Audit

## Scope

The scope is pure-software, low-cost post-training NAS for a pretrained LLM on one NVIDIA A800 80GB. Hardware measurement, device-specific search, and large continued-pretraining recipes are excluded. This file records formal gaps for Turn 2; it does not authorize any experiment.

## Problems that are substantially solved or saturated

- **The existence of post-training LLM NAS.** Search for Efficient LLMs, Puzzle, Jet-Nemotron, Týr, and TraceNAS make the broad framing non-novel.
- **Uniform-to-non-uniform pruning.** Minitron, MultiPruner, Týr, TraceNAS, and Sheared LLaMA already allocate depth/width/head/FFN capacity non-uniformly.
- **Greedy context-dependent block removal.** SLEB and BlockPruner recompute whole-model scores after every deletion.
- **Local block distillation followed by assembly.** LANA, NAS-BERT, Puzzle, and Distill-then-Replace cover local teacher supervision and composition.
- **Generic global evolutionary compression.** EvoPress, Týr, Search for Efficient LLMs, TraceNAS, and DarwinLM make “use evolution instead of greedy” inadequate.
- **Generic zero-cost proxy invention.** NASWOT, TE-NAS, ZiCo, LPZero, and TraceNAS form a strong ladder; changing the scalar proxy is not enough.
- **Compatibility glue/adapters for block reassembly.** Building LLMs Like LEGO directly covers lightweight glue layers, while Puzzle uses global KD.
- **Multi-domain calibration as a slogan.** The calibration-data study and GPrune-LLM already establish sensitivity and cross-distribution neuron treatment.

## Highly crowded problems

### Joint depth/width/head/FFN search

The search space is useful but cannot itself carry novelty. A contribution must change how non-additive quality, recovery, uncertainty, or multiple budgets are optimized.

### Training-free quality ranking

Whole-model perplexity, logits, Taylor/Fisher/Hessian criteria, activation metrics, gradient traces, and symbolic proxy search are all represented. A new score must predict a clearly different target and win under equal calibration/evaluation cost.

### Recovery-aware candidate selection

Minitron retrains candidates, TraceNAS targets recovery alignment, DarwinLM inserts multistep training into evolution, and NIRVANA links pruning saliency to training dynamics. Remaining claims must specify the recovery horizon, adapter family, token budget, and terminal quantity.

### Interaction-aware pruning

Static local scores are weak, but SLEB/BlockPruner recompute, Týr builds a globally evaluated supernet, TraceNAS uses a global gradient signal, and EvoPress uses global evolution. “We model interactions” is not a sufficient claim.

## Remaining gaps with defensible algorithmic space

### Gap G1 — Predicting compositional functional damage before assembly

Puzzle scores one replacement at a time and sums the scores even though child-generated states drift. Existing global proxies score full candidates but do not explain or cheaply reuse how a local replacement residual propagates through the parent computation. A transported-residual estimator could be both reusable and mechanistic, provided it predicts multi-replacement rank beyond additive scores, SLEB-style recomputation, TraceNAS, and equal-budget candidate evaluation.

Covered by: **Direction 1**.

### Gap G2 — Predicting terminal quality after a fixed, cheap recovery process

Immediate loss is not the final objective, but actually adapting every candidate defeats low-cost NAS. Existing work either uses a first-order alignment proxy or spends training tokens in candidate selection. A finite-horizon, candidate-specific predictor of recovery under a named adapter/optimizer remains plausible.

Covered by: **Direction 2**.

### Gap G3 — One coherent post-training architecture path across compression budgets

Many methods rerun search independently at each budget; supernet methods amortize queries only after expensive special training. A post-training method that optimizes integrated regret across a budget interval and produces stable nested edits could reduce repeated search and architecture churn. The distinction from OFA/AmoebaLLM must remain explicit: no elastic-supernet training.

Covered by: **Direction 3**.

### Gap G4 — Architecture-level selection robustness under tiny, shifting calibration sets

Calibration choice can change pruning outcomes. GPrune-LLM addresses neuron-level cross-distribution behavior, but the statistical reliability of choosing one whole architecture from many adaptively scored candidates remains underdeveloped. The gap is selection-aware uncertainty, paired comparisons, and out-of-calibration regret—not simply using more domains.

Covered by: **Direction 4**.

### Gap G5 — Non-myopic decisions under a strict candidate-evaluation budget

Greedy methods are myopic and evolution can require many whole-model evaluations. A conditional marginal response model with explicit value-of-information could choose which interventions to measure and which multi-step path to explore. The novelty bar is high because EvoPress/Týr/TraceNAS already handle global quality in different ways.

Covered by: **Direction 5**.

## Potential Gap Signals not promoted to independent directions

- **Cross-model transfer of architecture rankings:** potentially useful, but evidence is too thin and domain shift may dominate.
- **Task-vector preservation rather than average perplexity:** important, but risks becoming benchmark engineering unless tied to a new constrained-search algorithm.
- **Joint architecture and calibration-data selection:** partly absorbed into Direction 4; a separate direction would be too close.
- **Recovery-token allocation across blocks:** could strengthen Direction 2, but alone resembles resource scheduling.
- **Higher-than-pairwise architecture interactions:** likely exist, but a raw higher-order tensor model would be sample-inefficient and less defensible than Direction 1's functional structure.
- **Search-space expansion to SSM/linear attention:** operators alone do not define an algorithmic contribution and Jet-Nemotron/Distill-then-Replace already occupy the obvious space.

## Gap-to-direction separation check

| Direction | Primary target | Explicitly not its core contribution |
|---|---|---|
| 1. Causal Error Transport | Functional composition of replacement errors | Generic pruning, glue layers, or a new operator library |
| 2. Terminal-Recovery NAS | Quality after a fixed recovery horizon | Immediate zero-cost score or training every candidate |
| 3. Budget-Path NAS | Joint decisions across many compression budgets | A single-budget Pareto search or trained elastic supernet |
| 4. Risk-Calibrated NAS | Statistical robustness of architecture selection | Neuron-level robust importance or “more calibration data” |
| 5. Non-Myopic Intervention Search | Sample-efficient decision policy for candidate evaluation | Generic evolution or static pairwise scoring |
