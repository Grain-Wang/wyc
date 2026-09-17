# Direction 3 — Budget-Path NAS without Elastic-Supernet Training

## 1. Core Research Problem

How can one post-training search produce a coherent family of compact architectures across a continuous range of compression budgets, instead of rerunning independent searches that yield unstable, mutually incompatible designs?

## 2. Why This Problem Matters

Compression is rarely a single point: model owners need several sizes, and the acceptable budget often changes after evaluation. Independent search repeats cost and can cause architecture churn—an edit selected at 20% compression may be restored at 30% while an unrelated component is removed. OFA, AutoDistil, and AmoebaLLM amortize budgets through trained elastic networks, but that up-front training conflicts with the low-cost post-training setting. A principled post-training architecture path would contribute a new multi-budget objective and a reusable family from one calibration pass.

## 3. Closest Prior Work

1. [Once-for-All](../reference_papers_processed/2020_Once_For_All.md): nested elastic subnets after expensive progressive supernet training.
2. [AutoDistil](../reference_papers_processed/2022_AutoDistil.md): budget-partitioned SuperLMs and few-shot subnet ranking.
3. [NAS-BERT](../reference_papers_processed/2021_NAS_BERT.md): adaptive-size task-agnostic Transformer compression.
4. [AmoebaLLM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8f11e548311c7fd3f33596a4d1dd41f0-Abstract-Conference.html): arbitrary-depth/width subnets from one adapted model.
5. [Týr-the-Pruner](../reference_papers_processed/2025_Tyr_The_Pruner.md): global non-uniform allocation at chosen sparsity levels.

## 4. Exact Gap

Existing low-cost methods generally optimize one budget at a time; elastic methods require purpose-built supernet training. The missing objective jointly minimizes quality regret over a **budget interval** while constraining edit consistency. The key mechanism is not another Pareto sweep, but coupling architecture decisions across budgets so that evidence and structure are reused.

==================================================
MAIN IDEA
==================================================

## 5. Main Idea

### 5.1 Core Algorithmic Idea

Construct a **Nested Architecture Path (NAP)** on an edit DAG. Nodes are feasible post-training architectures and edges are irreversible structural edits (remove a layer/group, reduce FFN width, or replace an operator with a cheaper option). Rather than minimize loss separately at each budget, optimize one monotone path whose prefixes serve every budget. Use conditional marginal-damage estimates that update along the path, plus a cross-budget regret term that prevents one budget from sacrificing all others. Solve by dynamic programming on a restricted DAG or a Lagrangian shortest-path method with column generation.

### 5.2 Mathematical Form

Let (mathcal B=\{B_1>\cdots>B_K\}) be decreasing resource budgets, (A_k) the architecture served at (B_k), and (A_{k+1}\preceq A_k) mean that (A_{k+1}) is obtained only by allowed shrink edits. Optimize

\[
\min_{A_1\succeq\cdots\succeq A_K}
\sum_{k=1}^{K}w_k\left[\widehat L(A_k)-\widehat L_k^*\right]
+\gamma\sum_{k=1}^{K-1}\Omega(A_k,A_{k+1})
\]

subject to (C(A_k)\le B_k). Here (widehat L_k^*) is the best known independent-search lower bound and (Omega) penalizes large discontinuous changes, not shrinkage itself.

Represent each edit edge (e:u\to v) by cost reduction (Delta C_e), conditional damage (widehat d_e(u)), and uncertainty (q_e). A path objective is

\[
\min_{p}\sum_{e\in p}\alpha_e(\mathcal B)\widehat d_e(u_e)
+\lambda\sum_{e\in p}q_e,
\]

where (alpha_e) equals the total weight of budgets whose served prefix includes the edit. Column generation evaluates a new edit only when its reduced cost can improve the current path.

### 5.3 Why It Should Work

The best single-budget decisions share substantial information. A path objective rewards edits that remain useful over a wide budget range and delays edits whose harm is acceptable only at extreme compression. Nestedness also enables parameter inheritance and incremental recovery. Conditional edge updates prevent the path from assuming that all edit damages are static.

### 5.4 Falsifiable Hypothesis

Using the same total candidate-evaluation budget as (K) independent searches, NAP will reduce area under the quality-regret-versus-compression curve by at least 15%, while achieving at least 80% edit retention between adjacent budgets and no more than a small (predefined 2% relative-loss) penalty at any individual budget.

### 5.5 Smallest Canary

**Design only; do not run in Turn 2.**

- Parent model: Qwen2.5-1.5B.
- Model size: 1.5B.
- Dataset: WikiText-2 calibration, disjoint C4/FineWeb validation.
- Calibration size: 64 sequences × 512 tokens.
- Metric: area under held-out perplexity-regret curve over 10/20/30/40% parameter reduction; edit-retention/Jaccard between adjacent budgets; evaluations used.
- Variable: independent search versus nested path; static versus conditional edge costs; path regularization.
- Expected signal: early low-damage edits remain useful across budgets, reducing repeated exploration.
- Success criterion: ≥15% lower integrated regret at equal evaluations, ≥80% adjacent edit retention, and no budget with >2% relative loss penalty versus independent search.
- Failure criterion: independent optima are fundamentally non-nested and the path pays >2% at multiple budgets, or savings vanish after conditional re-evaluation.
- Estimated GPU memory: 20–35 GB; most path optimization is CPU-side.
- Estimated runtime: 3–6 A800-hours for all budget points; planning estimate.

### 5.6 Strong Baselines

- Simple baseline: one global importance ordering truncated at each budget.
- Closest-paper baseline: independent Týr/BlockPruner searches per budget.
- Reviewer baseline: Once-for-All/AmoebaLLM-style nested family where a comparable public checkpoint/procedure is feasible.
- Killer baseline: pool all independently found Pareto architectures and select per budget; independent search with shared cached evaluations.

### 5.7 Main Novelty Claim

“Our method differs from prior work because it optimizes an entire monotone post-training architecture path over a budget interval without training an elastic supernet, explicitly minimizing cross-budget regret and reusing conditional edit evidence.”

### 5.8 Strongest Reviewer Rejection

“OFA, NAS-BERT, AutoDistil, and AmoebaLLM already produce many subnets; nestedness is an operational convenience that sacrifices the best architecture at each budget and is not an algorithmic advance.”

### 5.9 Response to Reviewer

The response requires a formal cross-budget objective, an evaluation-cost advantage, and evidence that path consistency enables reuse/recovery beyond convenience. If a shared-cache independent Pareto search matches quality and cost, the direction loses its reason to exist.

### 5.10 Main Idea Kill Criteria

- The optimal architecture sets across budgets have low nesting and force material regret.
- A single static importance ordering matches the proposed path.
- Shared-cache independent searches have equal cost and better quality.
- The only benefit is model-storage convenience rather than search/recovery efficiency.
- A post-training paper is found with the same integrated cross-budget path objective.

==================================================
FALLBACK IDEA
==================================================

## 6. Fallback Idea

### 6.1 When to Switch

Switch if strict nestedness causes unacceptable quality loss but rankings still exhibit systematic budget dependence that can be learned from sparse evaluations.

### 6.2 Alternative Algorithmic Idea

Learn a **Budget-Conditioned Architecture Policy (BCAP)** that directly emits a whole architecture for budget (B). Train it from sparse, actively chosen architecture comparisons using a listwise ordinal objective and a feasibility projection. Cross-budget regularization encourages smooth decision boundaries but does not require one architecture to be a subnetwork of another.

### 6.3 Why It Is Non-equivalent

NAP is a discrete monotone path optimizer with irreversible shared edits. BCAP is a conditional global surrogate/policy; architectures at neighboring budgets may be non-nested and can exchange components. BCAP can succeed precisely when nestedness—the Main Idea's core assumption—is false.

### 6.4 Mathematical Form

Let (pi_\theta(B)) output logits over per-layer choices, projected by (Pi_B) to a feasible discrete architecture. Given observed pairwise preferences (A\succ_B A'), train

\[
\min_\theta\sum_{(A,A',B)}
-\log\sigma\left(s_\theta(A,B)-s_\theta(A',B)\right)
+\lambda\int\left\|\frac{\partial s_\theta(A,B)}{\partial B}\right\|^2dB.
\]

Active acquisition selects ((A,B)) pairs with maximal expected Pareto-front change, and the deployed choice is (A_B=\Pi_B(\pi_\theta(B))).

### 6.5 Why It Could Still Work

Budget changes can reverse the value of depth versus width without supporting a nested path. A conditional policy shares statistical strength across budgets while preserving those reversals, amortizing search without an elastic supernet.

### 6.6 Falsifiable Hypothesis

With ≤25% of the evaluations used by independent searches, BCAP will recover at least 90% of their hypervolume/quality frontier across held-out budgets and interpolate to two unseen budgets better than nearest-budget reuse.

### 6.7 Smallest Canary

**Design only.** Use the 1.5B model, six observed budgets and two held-out budgets, 100–200 cached candidate scores, and compare a budget-agnostic surrogate, independent surrogates, and BCAP. Success is ≥90% frontier recovery and lower held-out-budget regret; failure is no transfer or the acquisition cost approaching independent search. Estimated memory: 20–30 GB; runtime: 2–5 A800-hours plus CPU surrogate fitting.

### 6.8 Strong Baselines

- Static importance ordering.
- Independent per-budget predictors/search.
- Once-for-All accuracy predictor or equivalent budget-conditioned predictor.
- Gaussian-process/random-forest surrogate with budget as an ordinary feature.

### 6.9 Novelty Risk

**HIGH.** Conditional predictors and multi-objective NAS are mature. The architecture representation, active cross-budget acquisition, and post-training evidence must distinguish this from standard surrogate NAS.

### 6.10 Fallback Kill Criteria

- Cross-budget transfer does not reduce evaluation count.
- A generic surrogate with budget as a feature matches BCAP.
- Held-out-budget interpolation is unstable across models.
- The contribution reduces to ordinary multi-objective NAS.

==================================================
DIRECTION-LEVEL ASSESSMENT
==================================================

## 7. A800 Feasibility

- Model: 1.5B primary, 3B replication, optional 7B sparse confirmation.
- Memory: 20–35 GB for candidate scoring; graph/policy optimization is CPU-light.
- Compute: cached inference across several budgets; no full-model training.
- Dataset: small unlabeled calibration corpus plus held-out language/tasks.
- Training requirement: none for NAP; small CPU/GPU surrogate for BCAP.
- Calibration requirement: 64–128 sequences, reused across budgets with held-out shards.
- Expected time: canary 3–6 GPU-hours; full paper 200–450 A800-hours.
- Feasibility verdict: **HIGH**.

## 8. Top-Tier CCF-C Evidence Requirement

Required: evidence of budget-specific ranking changes; integrated regret and evaluation-cost curves across at least two model families; comparison with independent search, static orderings, shared-cache search, OFA/AmoebaLLM-like elastic baselines, and generic Pareto surrogates; nestedness/conditional-cost/column-generation ablations; unseen-budget interpolation; incremental recovery/storage analysis; and negative cases where strict nesting hurts. The story must establish a scientific multi-budget objective, not a model-zoo convenience.

## 9. Direction-Level Risks

- Novelty risk: **HIGH**—elastic and multi-budget NAS have extensive prior art.
- Experiment risk: **MEDIUM**—useful edit paths may exist but offer modest quality gains.
- Compute risk: **LOW**.
- Reviewer risk: **HIGH**—reviewers may view nestedness as an application constraint.
- Saturation risk: **HIGH**—OFA, NAS-BERT, AutoDistil, and AmoebaLLM are strong conceptual threats.

## 10. Final Direction Verdict

**PROMISING BUT NEEDS REFRAMING**
