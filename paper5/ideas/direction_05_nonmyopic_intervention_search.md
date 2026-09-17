# Direction 5 — Non-Myopic Intervention Search for Structured LLM Compression

## 1. Core Research Problem

Can post-training NAS choose a globally good sequence of structural edits under a strict whole-model evaluation budget when each edit changes the marginal value of all remaining edits?

## 2. Why This Problem Matters

SLEB and BlockPruner observe context dependence but commit greedily; evolutionary methods handle global quality but spend evaluations broadly and provide weak credit assignment. At LLM scale, the scarce resource is not CPU optimizer iterations but informative whole-model forward/backward interventions. A search algorithm that actively decides which edit interactions to measure and plans beyond the next deletion could improve the quality–evaluation frontier across many structured search spaces.

## 3. Closest Prior Work

1. [SLEB](../reference_papers_processed/2024_SLEB.md): iterative whole-model block-removal verification.
2. [BlockPruner](../reference_papers_processed/2025_BlockPruner.md): greedy attention/MLP residual-block pruning.
3. [Týr-the-Pruner](../reference_papers_processed/2025_Tyr_The_Pruner.md): coarse-to-fine evolutionary global allocation.
4. [TraceNAS](../reference_papers_processed/2026_TraceNAS.md): evolutionary joint depth/width search with a global proxy.
5. [EvoPress](https://proceedings.mlr.press/v267/sieberling25a.html): global evolutionary compression designed because additive/independent errors fail.

## 4. Exact Gap

Greedy methods have no value-to-go, while evolutionary methods do not use structural interventions to learn how marginal edit damage changes with the surviving architecture. The missing mechanism is a decision-dependent response model plus value-of-information acquisition: spend an evaluation where it most improves the final compression policy, not simply where a candidate currently looks promising.

==================================================
MAIN IDEA
==================================================

## 5. Main Idea

### 5.1 Core Algorithmic Idea

Formulate structured compression as a finite-horizon partially observed decision process and learn a **Conditional Marginal Response Model (CMRM)** online. The model predicts the damage distribution of edit (u) given a compact state of the surviving architecture and previously observed interventions. At each step, a two-step rollout balances predicted final quality and information gain: it may evaluate an edit that is not immediately best if that intervention resolves uncertainty shared by many future decisions. Accepted changes are reversible within a trust region until confidence is sufficient.

### 5.2 Mathematical Form

Let (s_t) encode surviving groups, budget remaining, layerwise activation summaries, and observed edit responses. An expensive intervention returns

\[
y_t=\mathcal L(A_t\oplus u_t)-\mathcal L(A_t)+\epsilon_t.
\]

Fit (p_\theta(y\mid s,u)) with a structured low-rank state–edit embedding. Define an approximate value function

\[
V_h(s)=\min_{u\in\mathcal U(s)}
\mathbb E_\theta[y(s,u)+V_{h-1}(T(s,u))]
-\beta I(y;\theta\mid s,u),
\]

where the mutual-information term rewards measurements that reduce policy uncertainty. Execute the edit only when its risk-adjusted (Q)-value is better than retaining the current structure; otherwise record the intervention and revert. The final path must meet (C(A_T)\le B).

### 5.3 Why It Should Work

Related edits share response structure: removing one FFN group changes the sensitivity of nearby groups and the competing value of depth versus width. Conditional modeling reuses this evidence; information-aware lookahead avoids spending all evaluations around the current greedy path. Reversible probes separate measurement from commitment.

### 5.4 Falsifiable Hypothesis

At equal whole-model evaluation count, CMRM will find lower held-out loss than greedy recomputation and EvoPress/Týr-style evolution, and its learned conditional marginals will reduce prediction error by at least 20% over static edit scores after the first 20 interventions.

### 5.5 Smallest Canary

**Design only; do not run in Turn 2.**

- Parent model: Qwen2.5-1.5B.
- Model size: 1.5B.
- Dataset: WikiText-2 calibration and disjoint C4/FineWeb validation.
- Calibration size: 64 sequences × 512 tokens.
- Metric: best held-out loss versus number of complete interventions; conditional marginal MAE; simple-regret at 20/30% reduction.
- Variable: greedy/evolution/CMRM, lookahead depth, information coefficient, and state features.
- Expected signal: static marginal estimates become biased after several edits, while conditional updates predict direction and magnitude.
- Success criterion: ≥20% marginal-error reduction and a consistent best-found quality gain at 25/50/100 evaluations over all strong baselines.
- Failure criterion: static rankings remain sufficient, evolution wins at equal evaluations, or learned state requires too many observations.
- Estimated GPU memory: 20–35 GB.
- Estimated runtime: 3–7 A800-hours; planning estimate.

### 5.6 Strong Baselines

- Simple baseline: static importance ordering.
- Closest-paper baseline: SLEB/BlockPruner greedy recomputation.
- Reviewer baseline: random search, Bayesian optimization, and standard contextual bandit with the same features.
- Killer baselines: EvoPress, Týr coarse-to-fine evolution, and TraceNAS evolution at matched complete-candidate evaluations.

### 5.7 Main Novelty Claim

“Our method differs from prior work because it treats architecture evaluations as information-bearing interventions and learns decision-dependent marginal responses for non-myopic planning, rather than greedily committing edits or globally evolving candidates without structural credit assignment.”

### 5.8 Strongest Reviewer Rejection

“This is generic Bayesian optimization/bandits applied to pruning; EvoPress already offers low-evaluation global search, and a learned response model cannot be trained reliably from the tiny sample budget it is supposed to save.”

### 5.9 Response to Reviewer

The only convincing response is an equal-evaluation curve plus transfer/ablation evidence showing that the architecture-state factorization and intervention information gain—not generic BO—drive sample efficiency. A generic BO baseline matching the result kills the claim.

### 5.10 Main Idea Kill Criteria

- No reproducible conditional-marginal shift after edits.
- Generic BO/evolution matches or beats CMRM at equal evaluations.
- More than 100–200 interventions are needed before the model helps.
- Value-of-information probes consume savings without improving the selected architecture.
- The method's features are model-specific heuristics rather than a reusable formulation.

==================================================
FALLBACK IDEA
==================================================

## 6. Fallback Idea

### 6.1 When to Switch

Switch if edit-level marginal responses are too noisy or nonstationary to learn, but whole-architecture comparisons remain reliable enough for active ranking.

### 6.2 Alternative Algorithmic Idea

Use **Active Listwise Architecture Tournaments (ALAT)**. Maintain a diverse population but learn only ordinal whole-architecture preferences. Construct small tournaments whose outcomes maximally reduce uncertainty over the feasible Pareto champion, and generate challengers by disagreement between an ensemble of rankers. This abandons edit-level credit assignment entirely.

### 6.3 Why It Is Non-equivalent

CMRM models state-conditioned effects of individual edits and plans an edit sequence. ALAT treats architectures as atomic structured objects and learns listwise order; it has no marginal-effect or transition model. It can work when local credit is unidentifiable but relative candidate quality is stable.

### 6.4 Mathematical Form

Let (s_\theta(A)) be an ensemble rank score over architecture encodings. For an evaluated list (mathcal T=(A_1,\ldots,A_m)), minimize Plackett–Luce loss

\[
\mathcal L_{\mathrm{rank}}=-\sum_{k=1}^{m}
\log\frac{\exp s_\theta(A_{\pi_k})}
{\sum_{j=k}^{m}\exp s_\theta(A_{\pi_j})}.
\]

Select the next tournament by

\[
\mathcal T^*=\arg\max_{\mathcal T}
\frac{H[p(A^*\mid\mathcal D)]-
\mathbb E_{y_\mathcal T}H[p(A^*\mid\mathcal D,y_\mathcal T)]}
{\operatorname{cost}(\mathcal T)}
\]

subject to structural diversity and budget feasibility.

### 6.5 Why It Could Still Work

Ordinal comparisons can be easier to learn than calibrated loss values and naturally focus capacity near the top of the ranking. Whole-architecture encodings can capture higher-order interactions without attributing them to individual edits.

### 6.6 Falsifiable Hypothesis

ALAT will reach the same best-candidate quality as standard evolution with at most half the complete-candidate evaluations and outperform random/listwise search without champion-entropy acquisition.

### 6.7 Smallest Canary

**Design only.** Fix a 500-architecture offline pool for a 1.5B model, reveal scores sequentially, and compare ALAT, random, evolution, generic BO, and pairwise active ranking at 25/50/100 reveals. Only if the offline study succeeds should an online GPU test be considered. Estimated online memory: 20–30 GB; runtime: 2–5 A800-hours.

### 6.8 Strong Baselines

- Random and regularized evolution.
- Generic Bayesian optimization with architecture encoding.
- Pairwise/listwise ranker with random acquisition.
- TraceNAS/Týr candidate scores as fixed features.

### 6.9 Novelty Risk

**CRITICAL.** Predictor-based NAS, listwise ranking, and active learning are mature; without a distinctly post-training LLM theorem/mechanism, this is likely ordinary surrogate NAS.

### 6.10 Fallback Kill Criteria

- Generic BO/evolution needs the same or fewer evaluations.
- Ranker uncertainty is miscalibrated and acquisition collapses.
- Offline-pool gains fail in online candidate generation.
- The method's only novelty is an architecture encoder or acquisition-function swap.

==================================================
DIRECTION-LEVEL ASSESSMENT
==================================================

## 7. A800 Feasibility

- Model: 1.5B primary, 3B confirmation.
- Memory: 20–35 GB for candidate interventions.
- Compute: 50–200 calibration evaluations per setting; no recovery training.
- Dataset: small unlabeled calibration/validation shards.
- Training requirement: tiny response/ranking models, primarily CPU-side.
- Calibration requirement: 64–128 sequences.
- Expected time: canary 3–7 GPU-hours; full paper 250–500 A800-hours.
- Feasibility verdict: **HIGH**, though scientific risk rather than compute is limiting.

## 8. Top-Tier CCF-C Evidence Requirement

Required: direct measurement of decision-dependent marginal shifts; best-quality-versus-evaluation curves on several models/budgets/search spaces; matched comparisons with SLEB, BlockPruner, Týr, TraceNAS, EvoPress, generic BO, and evolution; ablations of state, information gain, reversibility, and lookahead; transfer of the response representation; wall-clock accounting; and failure cases where interventions do not generalize. The paper must offer an LLM-specific structural principle, not merely a search-controller benchmark.

## 9. Direction-Level Risks

- Novelty risk: **CRITICAL** for the fallback and **HIGH** for the main idea.
- Experiment risk: **HIGH**—small-data response learning may not work.
- Compute risk: **LOW–MEDIUM**.
- Reviewer risk: **CRITICAL**—“generic BO/bandit” is difficult to rebut.
- Saturation risk: **CRITICAL**—EvoPress/Týr/TraceNAS and predictor-based NAS crowd the space.

## 10. Final Direction Verdict

**HIGH RISK**
