# Candidate Ranking — Awaiting Human Selection

## Scoring rule

Scores reflect the evidence available in Turn 2, not experimental results. For **Novelty risk**, a higher score means lower risk. Feasibility is conditioned on one NVIDIA A800 80GB and deliberately small 0.5B–3B development models. No rank is an automatic selection.

| Rank | Direction | Problem importance /15 | Algorithmic novelty /20 | Mechanism depth /10 | Novelty risk /10 | Reviewer defensibility /10 | Experimental feasibility /10 | Single-A800 feasibility /10 | Expected signal /5 | Baseline availability /5 | Story clarity /5 | Total /100 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | Causal Error Transport for Compositional Block NAS | 14 | 19 | 10 | 8 | 8 | 9 | 9 | 4 | 4 | 5 | **90** |
| **2** | Terminal-Recovery NAS under a Fixed Adaptation Budget | 15 | 18 | 10 | 6 | 7 | 8 | 8 | 4 | 5 | 5 | **86** |
| **3** | Budget-Path NAS without Elastic-Supernet Training | 13 | 15 | 8 | 6 | 7 | 9 | 10 | 3 | 4 | 5 | **80** |
| **4** | Risk-Calibrated Architecture Selection from Tiny Data | 14 | 13 | 7 | 5 | 6 | 10 | 10 | 4 | 4 | 4 | **77** |
| **5** | Non-Myopic Intervention Search | 13 | 12 | 7 | 3 | 4 | 8 | 9 | 3 | 5 | 4 | **68** |

## Rank 1 — Causal Error Transport for Compositional Block NAS

**Why first:** it targets the explicit approximation enabling Puzzle/LANA-style search, offers a mathematical mechanism rather than an optimizer swap, admits clean rank-prediction and selected-architecture tests, and can be studied at 1.5B without training a large model. Its main and fallback use genuinely different assumptions: perturbative error transport versus nonlinear state rollout.

**Main risk:** reviewers may reduce it to “Taylor plus pairwise terms,” and nonlinear drift may destroy usefulness at high compression. The canary must demonstrate rank/cost gains over actual-candidate global baselines, not just additive scoring.

## Rank 2 — Terminal-Recovery NAS under a Fixed Adaptation Budget

**Why second:** the target is arguably more important than Rank 1—deployment quality after recovery—and the finite-horizon formulation is deep. It ranks second because TraceNAS and DarwinLM make novelty defense harder, curvature estimation is costly, and simple successive halving may dominate.

**Main risk:** even if terminal prediction correlates with recovery, reviewers can argue that short candidate training is simpler and equally cheap.

## Rank 3 — Budget-Path NAS without Elastic-Supernet Training

**Why third:** it is highly feasible and defines a clean integrated-regret objective, with useful multi-budget deliverables from one search. It trails the top two because OFA, NAS-BERT, AutoDistil, and AmoebaLLM make multi-budget novelty crowded, and nestedness may be viewed as an engineering constraint.

**Main risk:** independent optima may be genuinely non-nested, turning the proposed path into an avoidable quality compromise.

## Rank 4 — Risk-Calibrated Architecture Selection from Tiny Data

**Why fourth:** the failure phenomenon is likely and the single-A800 study is easy to execute rigorously. GPrune-LLM, however, sharply narrows the novelty window; standard bootstrap/DRO/diverse calibration are dangerous killer baselines.

**Main risk:** positive results may amount to generic robust statistics rather than a new NAS contribution.

## Rank 5 — Non-Myopic Intervention Search

**Why fifth:** it addresses real search inefficiency, but EvoPress/Týr/TraceNAS and generic BO/bandit methods make novelty and sample-efficiency difficult to defend. Its fallback is especially crowded.

**Main risk:** the response model may require more full-model interventions than it saves, while generic BO matches the result.

## Required comparison questions

### 1. Why is Rank 1 first?

It combines a direct unsolved failure in the strongest nearest neighbor (non-compositional block scores), a derived algorithmic object (transported functional residuals), a clear low-cost test, and a paper story that joins mechanism, approximation, search, and architecture quality. It is not first because it is easiest; it is first because success would establish an independent method and explanatory result.

### 2. Why is Rank 2 not first?

Rank 2 has equal or greater problem importance, but it faces three simultaneous hazards: TraceNAS already sells a recoverability proxy, DarwinLM directly trains candidates during evolution, and finite-horizon curvature may cost more than progressive recovery. Rank 1's boundary against prior work is cleaner.

### 3. Which direction has the strongest novelty?

**Direction 1**, narrowly. Direction 2's finite-horizon target is also strong, but its neighboring 2026 work is closer.

### 4. Which direction has the lowest experimental risk?

**Direction 4.** Calibration sensitivity is already documented, and the required tests are forward-only. Low experimental risk does not imply strong publication novelty.

### 5. Which direction is most likely to collide with nearest prior work?

**Direction 5**, due to EvoPress, Týr, TraceNAS, predictor-based NAS, Bayesian optimization, and active ranking. Direction 4 is next because of GPrune-LLM.

### 6. Which direction could succeed experimentally yet still be rejected for weak novelty?

**Direction 5** most clearly; a sample-efficient search curve can still look like generic BO. **Direction 4** also faces this risk if gains reduce to standard DRO or more diverse data.

### 7. Which direction has the strongest Main Idea?

**Direction 1's Causal Error Transport**: it gives the clearest new mathematical object, causal mechanism, approximation ladder, and falsifiable advantage over the exact assumption used by Puzzle.

### 8. Which direction has the most valuable Fallback Idea?

**Direction 2's Repair-Subspace Certificate**: even if finite-horizon dynamics are inaccurate, irreducible damage relative to a fixed adapter family is a meaningful, optimizer-independent target and could guide both architecture and recovery-capacity allocation.

### 9. Which direction best fits one A800?

**Direction 4** has the lowest compute burden. Among directions with strong top-tier novelty, **Direction 1** has the best novelty/feasibility balance.

### 10. Which direction has the best chance of a top-tier CCF-C paper story?

**Direction 1**, contingent on beating equal-cost global baselines and validating the error-transport mechanism. Direction 2 could surpass it if terminal-recovery prediction works, but its prior-work and computational risks are higher.

## Rank 1 versus Rank 2

| Question | Rank 1: Causal Error Transport | Rank 2: Terminal-Recovery NAS |
|---|---|---|
| Core target | Quality of composing multiple structural replacements before recovery | Quality after a fixed recovery process |
| Main advantage | Cleaner mechanistic gap in Puzzle/LANA; reusable block-library estimator; lower validation cost | Targets the true deployed objective; potentially broader across search spaces |
| Main risk | Perturbative approximation may fail as edits accumulate | TraceNAS/DarwinLM overlap; curvature cost; optimizer nonlinearity |
| Killer comparison | Whole-candidate scoring, Týr, TraceNAS, EvoPress | DarwinLM/successive halving, TraceNAS, actual recovery oracle |
| A800 burden | Moderate; mainly inference and sketched JVPs | Higher; derivatives plus real recovery curves |
| Story if successful | Explains and fixes non-compositional local NAS | Makes recovery a predictable search objective |
| Why Rank 1 wins now | More defensible novelty boundary and stronger cost-controlled canary | Greater upside, but a materially higher chance that strong simple baselines erase the claim |

## Most dangerous reviewer objection per direction

1. **Direction 1:** “A fragile Taylor/pairwise approximation cannot beat existing global candidate scoring at useful compression.”
2. **Direction 2:** “DarwinLM's progressive training is simpler and TraceNAS already captures recoverability; your predictor adds cost without value.”
3. **Direction 3:** “OFA/AmoebaLLM already solve multi-budget architectures, and nestedness merely sacrifices per-budget optima.”
4. **Direction 4:** “GPrune-LLM plus standard DRO/bootstrap or a larger diverse calibration set covers the contribution.”
5. **Direction 5:** “This is generic Bayesian optimization/bandits for pruning and cannot learn reliably in the claimed low-sample regime.”

## Directions not worth continuing even if a weak positive signal appears

- **Direction 5** is not worth continuing if generic BO/evolution matches it, even if it beats greedy pruning.
- **Direction 4** is not worth continuing if the gain is reproduced by stratified sampling, standard bootstrap/DRO, or simply 2–4× more random calibration data at equal cost.
- **Direction 3** is not worth continuing if its only benefit is stable filenames/storage and it loses materially to shared-cache independent search.
- **Direction 2** is not worth continuing if it predicts rankings but consumes as many tokens or wall-clock hours as successive halving.
- **Direction 1** is not worth continuing if it only beats additive Puzzle scoring but not equal-budget whole-candidate/global baselines.

## Human Selection Gate

This ranking is a recommendation, not authorization. No direction has been selected, no canary has been created or run, and none is a Paper Candidate. Work must stop here until the user explicitly chooses a direction and either its Main Idea or Fallback Idea for canary validation.
