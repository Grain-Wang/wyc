# Global Novelty Audit

## Audit standard

The question is not whether an idea differs in implementation, but whether its central algorithmic claim remains after comparison with the closest 2023–2026 work and a skeptical reviewer. “Survives” means it is worth a canary **only after user selection**; it does not mean Paper Candidate status.

| Direction | Main Idea | Fallback Idea | Closest Prior Work | Main Overlap | What Is Actually New | Novelty Threat | Top-Tier Novelty Strength | Reviewer Rejection | Survives Audit? |
|---|---|---|---|---|---|---|---|---|---|
| **1. Causal Error Transport** | Transport vector-valued local replacement residuals through downstream Jacobian sketches; optimize sparse cross terms/corrections | Receding-horizon search on actual child states with nonlinear truncated rollouts | Puzzle; LANA; NAS-BERT; Týr; TraceNAS; EvoPress; Building LLMs Like LEGO | Local block libraries, global interactions, derivative signals, compatibility repair | A functional composition law that reuses transported residuals to predict multi-replacement quality, plus an explicit approximation hierarchy | **MEDIUM** | **STRONG** | “Only Taylor/pairwise scoring; global interaction methods already exist and nonlinear drift invalidates it.” | **YES, conditional.** Must beat equal-cost whole-candidate, TraceNAS, Týr, and EvoPress and expose when the approximation fails. |
| **2. Terminal-Recovery NAS** | Predict loss after (K) fixed recovery steps using sketched curvature/adapter dynamics | Geometric certificate of irreducible error outside the feasible adapter repair subspace | Minitron; TraceNAS; DarwinLM; LLM-Pruner; Sheared LLaMA | Recovery-aware ranking, gradients, short candidate training, LoRA repair | A specified finite-horizon terminal objective predicted without recovering every candidate; fallback targets representability rather than dynamics | **HIGH** | **STRONG** | “DarwinLM and TraceNAS already cover training awareness; successive halving is simpler and more accurate.” | **YES, fragile.** It survives only if there is an equal-token regime where terminal prediction clearly dominates progressive candidate training. |
| **3. Budget-Path NAS** | Optimize a nested edit path minimizing integrated regret over a budget interval without supernet training | Budget-conditioned whole-architecture policy with active cross-budget comparisons | OFA; NAS-BERT; AutoDistil; AmoebaLLM; Týr | Multi-budget subnets, nested elasticity, accuracy predictors, Pareto search | A post-training cross-budget path objective with conditional edit reuse and no elastic-supernet training | **HIGH** | **MODERATE** | “Nestedness is operational convenience; elastic/multi-budget NAS is old and independent search is better.” | **YES, but needs reframing.** Search/recovery savings—not model-zoo convenience—must be the main result. |
| **4. Risk-Calibrated NAS** | Paired distributionally robust whole-architecture scoring with selection confidence and adaptive racing | Architecture-disagreement calibration coreset | Search Efficient LLM; SLEB; Týr; calibration-data study; GPrune-LLM | Calibration sensitivity, multi-domain scoring, robust pruning/data choice | Treating adaptive whole-architecture selection as the statistical object, with paired regret and token allocation | **HIGH** | **MODERATE** | “GPrune-LLM plus standard DRO/bootstrap or simply more diverse calibration already solves this.” | **YES, narrow.** Must show a winner's-curse failure that neuron-level robustness and larger diverse samples do not fix. |
| **5. Non-Myopic Intervention Search** | Conditional marginal response model plus information-aware lookahead over reversible structural interventions | Active listwise whole-architecture tournaments | SLEB; BlockPruner; Týr; TraceNAS; EvoPress; predictor-based NAS | Dynamic importance, evolution, global candidate scoring, active surrogate search | Architecture evaluations are treated as interventions that teach a state-dependent edit-response policy | **CRITICAL** | **WEAK** | “Generic BO/bandits applied to pruning; the response model needs more samples than it saves.” | **NO at current framing for a top-tier claim.** Retain only as a high-risk candidate; kill if generic BO matches it. |

## Threat-specific conclusions

### Ideas actively rejected

- **Plain pairwise interaction scoring:** too close to the obvious response to Puzzle and weaker than global-evaluation methods.
- **Compatibility adapters/glue layers:** directly threatened by Building LLMs Like LEGO and Puzzle's global KD.
- **Training-aware evolution:** directly covered by DarwinLM.
- **Multi-domain neuron importance:** directly covered by GPrune-LLM.
- **A new scalar zero-cost proxy discovered by search:** LPZero and TraceNAS make this too incremental.
- **Joint depth/width/head/FFN search:** already pervasive; it can be an evaluation space, never the main contribution.

### Why Direction 1 still survives

Its claim is mechanistic and falsifiable: local perturbations have vector direction, downstream amplification, and cross terms. Neither “evaluate the entire candidate globally” nor “train a glue layer” provides the same reusable composition estimator. The risk remains substantial because a first-order model may fail at useful compression.

### Why Direction 2 remains distinct but dangerous

TraceNAS predicts recovery alignment at one global derivative snapshot, while DarwinLM actually trains offspring. Direction 2 predicts a predeclared finite-horizon terminal quantity and must win on the cost–ranking frontier. If it cannot, it is unnecessary complexity.

### Why Directions 3–5 are not co-equal

Direction 3 has a clear new objective but old neighboring machinery. Direction 4 has a real phenomenon but a narrowing novelty window after GPrune-LLM. Direction 5 has neither a sufficiently unique LLM mechanism nor protection from generic optimizer baselines; even positive experiments could remain a weak paper.

## Global kill rule

No direction may advance merely because it beats a static importance baseline. A surviving direction must beat its listed killer baseline under matched candidate evaluations, calibration tokens, recovery tokens, and wall-clock accounting. This audit does not authorize selection or experimentation.
