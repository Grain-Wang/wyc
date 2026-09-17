# Direction 1 — Causal Error Transport for Compositional Block NAS

## 1. Core Research Problem

How can a post-training NAS method predict the whole-model damage of composing many locally distilled or inherited block alternatives without evaluating every assembled LLM? Local replace-one scores are not compositional: an upstream replacement changes the state distribution seen by every downstream replacement.

## 2. Why This Problem Matters

This is the approximation that makes Puzzle/LANA-style block libraries cheap enough to search, yet it is also their clearest algorithmic weakness. If local scores cannot rank composed models, a huge nominal search space is illusory: the optimizer confidently solves the wrong objective and later global distillation pays to repair its mistakes. The problem applies to layer deletion, width changes, attention replacement, and other residual-stream surgery. A reliable, reusable composition law would support a top-tier story because it changes both the NAS objective and the scientific understanding of when local model surgery composes.

## 3. Closest Prior Work

1. [Puzzle](../reference_papers_processed/2025_Puzzle.md): replace-one-block scores plus additive MIP assembly; explicitly needs global KD for compatibility.
2. [LANA](../reference_papers_processed/2022_LANA.md): locally distilled replacement library and additive constrained selection.
3. [NAS-BERT](../reference_papers_processed/2021_NAS_BERT.md): approximates model quality from blockwise losses.
4. [Týr-the-Pruner](../reference_papers_processed/2025_Tyr_The_Pruner.md): iterative expected-error supernet plus global validation.
5. [Building LLMs Like LEGO](https://aclanthology.org/2026.acl-long.2081/): reassembles heterogeneous blocks and learns glue layers for compatibility.

## 4. Exact Gap

The missing mechanism is a cheap mapping from each local functional perturbation to its downstream effect under composition. Puzzle assumes additive scalar damage, while glue layers/global KD repair incompatibility only after a candidate is chosen. Týr and TraceNAS use global signals but do not expose a reusable error-transport model for a block library. The objective therefore omits both downstream amplification and signed cancellation/reinforcement among replacement residuals.

==================================================
MAIN IDEA
==================================================

## 5. Main Idea

### 5.1 Core Algorithmic Idea

Build a **Causal Error Transport (CET)** model around the frozen parent LLM. For every candidate replacement, measure its vector-valued residual on parent states rather than compressing it immediately to one scalar. Transport that residual through a randomized low-rank sketch of the parent's downstream Jacobian. The squared norm of the sum of transported residuals provides first-order main effects and cross terms. A sparse set of two-replacement probes estimates only the conditional corrections for pairs whose transported residuals strongly align or whose positions are close. Search then minimizes the resulting structured quadratic functional-damage estimate under a parameter/FLOP budget.

The novelty is not “pairwise interactions.” It is the derivation of architecture interactions from residual-network error transport, with a testable approximation hierarchy: additive local score → transported main effects → transported cross terms → sparse state-drift corrections.

### 5.2 Mathematical Form

Let the parent be (h_{i+1}=F_i(h_i)), output (z=R(h_L)), and let option (a_i) at position (i) induce

\[
\delta_{i,a_i}(h_i)=\widetilde F_{i,a_i}(h_i)-F_i(h_i).
\]

On parent trajectories, use Jacobian-vector products and a fixed random output sketch (S) to obtain

\[
r_{i,a_i}=S J_{z\leftarrow h_{i+1}}\delta_{i,a_i}(h_i).
\]

For architecture (A=(a_1,\ldots,a_L)), the first compositional estimate is

\[
\widehat D_1(A)=\left\|\sum_i r_{i,a_i}\right\|_2^2
=\sum_i\|r_{i,a_i}\|_2^2+2\sum_{i<j}\langle r_{i,a_i},r_{j,a_j}\rangle.
\]

For a small selected edge set (E), estimate state-drift corrections

\[
c_{ij}(a_i,a_j)=S J_{z\leftarrow h_{j+1}}
\left[\delta_{j,a_j}(h_j+J_{h_j\leftarrow h_{i+1}}\delta_{i,a_i})-\delta_{j,a_j}(h_j)\right],
\]

and optimize

\[
\min_A\; \widehat D(A)=\widehat D_1(A)+
\sum_{(i,j)\in E}\omega_{ij}\langle r_{j,a_j},c_{ij}(a_i,a_j)\rangle
\quad\text{s.t.}\quad C(A)\le B.
\]

The discrete problem is a sparse quadratic multiple-choice knapsack, solvable by MISOCP for small libraries or beam/dual search for larger ones.

Pseudocode sketch:

1. Cache parent states on a calibration set.
2. For each position/option, compute its local residual and downstream JVP sketch.
3. Select suspicious pairs using alignment, distance, and uncertainty; measure conditional corrections only for them.
4. Solve the constrained structured quadratic search.
5. Validate only the final shortlist with actual whole-model loss.

### 5.3 Why It Should Work

Residual networks make a local modification an explicit perturbation to a shared state stream. Transport preserves direction: two locally harmful changes may cancel, while two small aligned changes may amplify. Random projections retain inner products approximately, so the search can reuse compact vectors rather than run every architecture. Sparse second-order corrections target exactly where linearization breaks.

### 5.4 Falsifiable Hypothesis

At equal whole-model evaluation count, CET will improve Kendall $\tau$ between predicted and true multi-replacement rankings by at least 0.15 over additive replace-one scoring and select an architecture with lower held-out teacher KL/perplexity at 20–40% structural reduction. The claim fails if transport adds no stable rank information across seeds, budgets, and calibration shards.

### 5.5 Smallest Canary

**Design only; do not run in Turn 2.**

- Parent model: Qwen2.5-1.5B or another permissively available 1–1.5B decoder already supported by the eventual codebase.
- Model size: 1.5B maximum for the first test.
- Dataset: WikiText-2 plus a disjoint C4/FineWeb validation shard.
- Calibration size: 64 sequences × 512 tokens; repeat with three 32-sequence shards.
- Metric: Kendall $\tau$/top-5 recall for 40 sampled 4–8-edit architectures; held-out perplexity and teacher KL.
- Variable: additive score, transported-main score, transported-cross score, and sparse correction count.
- Expected signal: multi-edit ranking error grows with aligned transported residuals; CET recovers the order.
- Success criterion: $\Delta\tau\ge0.15$ over additive scoring on both validation corpora and better best-of-equal-evaluations quality.
- Failure criterion: $\Delta\tau<0.05$, sign instability across shards, or actual full-candidate evaluation is equally cheap at the tested space.
- Estimated GPU memory: 35–55 GB with sequential JVPs, activation recomputation, and low-rank sketches.
- Estimated runtime: 4–8 A800-hours for the canary; planning estimate, not measured.

### 5.6 Strong Baselines

- Simple baseline: sum of replace-one-block KL/loss scores.
- Closest-paper baseline: Puzzle-style additive MIP.
- Reviewer baseline: SLEB/BlockPruner-style sequential recomputation with the same full-model evaluation budget.
- Killer baselines: Týr global-logit search, TraceNAS global gradient score, [EvoPress](https://proceedings.mlr.press/v267/sieberling25a.html), and direct whole-candidate scoring of an equal-sized shortlist.

### 5.7 Main Novelty Claim

“Our method differs from prior work because it derives reusable non-additive architecture scores by transporting vector-valued local replacement errors through the pretrained model's computation, rather than summing local scalar losses or repeatedly evaluating complete candidates.”

### 5.8 Strongest Reviewer Rejection

“This is a Taylor approximation with pairwise terms; Týr/TraceNAS already use global derivatives, EvoPress already rejects additive pruning, and nonlinear state drift will make the estimator unreliable exactly at useful compression ratios.”

### 5.9 Response to Reviewer

This objection is answerable only with three pieces of evidence: (1) a derivation and ablation separating transport from generic pairwise regression; (2) rank/cost curves against equal-budget global baselines; and (3) an error-bound or empirical remainder diagnostic that states when sparse corrections cease to be valid. Without all three, the contribution should be reframed or killed.

### 5.10 Main Idea Kill Criteria

- Transported residuals do not outperform additive scores under equal cost.
- Gains vanish against actual-candidate shortlist evaluation, TraceNAS, or EvoPress.
- The sparse correction graph becomes dense at 20–40% reduction, destroying low cost.
- The estimator works only on one model/calibration corpus.
- A closer paper is found that already derives the same residual-transport objective for post-training architecture composition.

==================================================
FALLBACK IDEA
==================================================

## 6. Fallback Idea

### 6.1 When to Switch

Switch if first-order transport is systematically inaccurate at practical compression, if second-order corrections become dense, or if novelty is covered by a direct transported-error prior.

### 6.2 Alternative Algorithmic Idea

Use **Receding-Horizon State Rollout (RHSR)**. Maintain a small beam of partial architectures and their actual cached child states. At each decision, roll out (H>1) future edits using cheap truncated suffix evaluation and a terminal parent-state discrepancy bound. Choose the next edit from the best horizon plan, execute it, refresh states, and repeat. This treats architecture construction as closed-loop control rather than estimating a fixed global quadratic score.

### 6.3 Why It Is Non-equivalent

CET assumes a perturbative functional model around parent trajectories and reuses transported residuals globally. RHSR makes no linearization or pairwise-additivity assumption: it observes the nonlinear state after accepted edits and plans a short sequence from that state. It can succeed when the Taylor model fails, at the cost of more inference.

### 6.4 Mathematical Form

For partial architecture state (s_t=(A_t,H_t)), action (u_t) applies one structural edit and produces cached hidden states (H_{t+1}=T(H_t,u_t)). Select

\[
u_t=\arg\min_{u}\min_{u_{t+1:t+H-1}}
\sum_{k=t}^{t+H-1} d(s_k,u_k)+V_{\text{bound}}(s_{t+H})
\]

subject to remaining resource feasibility. The terminal bound uses parent/child hidden-state disagreement on a truncated suffix; only the first action is committed before replanning.

### 6.5 Why It Could Still Work

Even if perturbations are strongly nonlinear, short actual rollouts expose the changed input distribution. Lookahead can avoid the irreversible mistakes of greedy Distill-then-Replace/SLEB while using far fewer complete candidates than unconstrained evolution.

### 6.6 Falsifiable Hypothesis

With the same number of block forward evaluations, horizon-2/3 rollout will find lower held-out loss than greedy recomputation and match or beat global evolution using at most one quarter of its complete-candidate evaluations.

### 6.7 Smallest Canary

**Design only.** Use the same 1.5B parent, 64×512-token calibration set, and 20–30% block-edit budget. Compare greedy, horizon 2, horizon 3, and EvoPress-style evolution. Success requires consistent best-found loss gains over greedy at equal block-forward count; failure is no gain or rollout cost approaching exhaustive whole-candidate scoring. Estimated memory: 25–40 GB. Estimated runtime: 3–6 A800-hours.

### 6.8 Strong Baselines

- SLEB/BlockPruner greedy recomputation.
- Distill-then-Replace greedy validation.
- EvoPress/global evolutionary search with matched evaluations.
- Random/beam search using the same horizon but no terminal bound.

### 6.9 Novelty Risk

**MEDIUM.** Model-predictive/receding-horizon search is established broadly, and the paper must show an LLM-specific state/cost construction rather than merely importing beam search.

### 6.10 Fallback Kill Criteria

- Horizon (H>1) does not beat greedy under matched cost.
- Terminal bounds are not correlated with completed architecture quality.
- Full-candidate evolution dominates at the same wall-clock budget.
- The method reduces to ordinary beam search without a defensible state or planning mechanism.

==================================================
DIRECTION-LEVEL ASSESSMENT
==================================================

## 7. A800 Feasibility

- Model: primary development at 1.5B, confirmation at 3B; at most one 7B sanity experiment.
- Memory: 35–55 GB for sequential JVP sketches; fallback needs 25–40 GB.
- Compute: inference, JVPs, and local replacement evaluation; no full pretraining.
- Dataset: small unlabeled calibration corpora plus held-out language/downstream sets.
- Training requirement: none for inherited/pruned options; optional pre-existing locally distilled blocks are inputs, not produced at scale in the first study.
- Calibration requirement: 32–128 sequences, with shard repeats.
- Expected time: canary 4–8 GPU-hours; a complete multi-model paper approximately 300–600 A800-hours including baselines and repetitions.
- Feasibility verdict: **HIGH**, if JVPs are sequential/sketched and the block library is deliberately small.

## 8. Top-Tier CCF-C Evidence Requirement

A top-tier result needs: a verified breakdown of additive local scores; CET ranking gains across models, edit types, budgets, and domains; equal-cost wins over direct candidate evaluation, greedy recomputation, TraceNAS, Týr, and EvoPress; transport/cross-term/correction ablations; approximation-error diagnostics; search wall-clock and memory; cases of cancellation and amplification; robustness to calibration size; and negative cases where nonlinear drift invalidates the approximation. The closed story is “local block surgery fails for a measurable causal reason; transported functional errors predict that reason cheaply; better predictions yield better architectures.”

## 9. Direction-Level Risks

- Novelty risk: **MEDIUM**—global interaction is crowded, though the functional derivation is distinct.
- Experiment risk: **MEDIUM**—rank improvement may not translate to selected-model gains.
- Compute risk: **LOW–MEDIUM**—JVP implementation can be costly but remains single-GPU.
- Reviewer risk: **HIGH**—likely dismissed as a Taylor approximation unless equal-cost evidence is rigorous.
- Saturation risk: **MEDIUM**—Puzzle's explicit weakness makes this an obvious target.

## 10. Final Direction Verdict

**STRONG RESEARCH OPPORTUNITY**
