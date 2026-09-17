# Direction 4 — Risk-Calibrated Architecture Selection from Tiny Data

## 1. Core Research Problem

When hundreds of architectures are adaptively scored on a tiny calibration set, how can post-training NAS avoid selecting a calibration-specific winner whose ranking collapses on new domains or samples?

## 2. Why This Problem Matters

Low-cost NAS derives its efficiency from small calibration sets, which creates both distribution shift and a winner's-curse effect: the best observed candidate is the one most likely to exploit noise. Prior pruning studies show that calibration choice materially changes outcomes, and GPrune-LLM already targets distribution-sensitive neuron scores. The unresolved problem is architecture-level selection risk after many adaptive comparisons. A solution would make low-cost NAS claims statistically credible and improve transfer without expensive labels.

## 3. Closest Prior Work

1. [SLEB](../reference_papers_processed/2024_SLEB.md): small-sample iterative perplexity scoring with calibration-corpus ablations.
2. [Search for Efficient LLMs](../reference_papers_processed/2024_Search_Efficient_LLM.md): candidate perplexity from few calibration samples.
3. [Týr-the-Pruner](../reference_papers_processed/2025_Tyr_The_Pruner.md): global logits validation and data-choice sensitivity.
4. [Is C4 Dataset Optimal for Pruning?](https://arxiv.org/abs/2410.07461): direct evidence that pruning depends unexpectedly on calibration data.
5. [GPrune-LLM](https://arxiv.org/abs/2603.13418) (**PREPRINT**): cross-distribution neuron behavior and global sparsity allocation.

## 4. Exact Gap

Existing methods typically optimize a sample mean, choose calibration data heuristically, or robustify individual neuron rankings. They do not account for the uncertainty introduced by selecting one **whole architecture** after adaptive, repeated comparisons. The missing objective is a paired, distributionally robust architecture score with valid selection confidence and targeted allocation of the limited calibration budget.

==================================================
MAIN IDEA
==================================================

## 5. Main Idea

### 5.1 Core Algorithmic Idea

Develop **Paired Distributionally Robust NAS (PDR-NAS)**. Partition unlabeled calibration text into lightweight domains/shards, and evaluate each candidate as a paired loss difference from the parent or incumbent on the same tokens. Maintain a hierarchical uncertainty model that separates example noise, domain shift, and architecture-by-domain interaction. Search minimizes a worst-plausible (or CVaR) quality loss plus a confidence penalty. A sequential racing rule spends new tokens only on candidates/domains capable of changing the selected architecture.

### 5.2 Mathematical Form

For candidate (A), domain (d), and sample (n), define paired damage

\[
\Delta_{A,d,n}=\ell(A;x_{d,n})-\ell(A_0;x_{d,n}).
\]

Estimate domain means (widehat\mu_{A,d}), covariance (widehat\Sigma_A), and standard errors (s_{A,d}). For an ambiguity set around empirical domain weights (widehat p), score

\[
R(A)=\max_{q:D_f(q\|\widehat p)\le\rho}
\sum_d q_d\widehat\mu_{A,d}
+\kappa\sqrt{q^\top\widehat\Sigma_Aq}.
\]

The selected architecture is

\[
A^*=\arg\min_{A:C(A)\le B}R(A).
\]

At each round, allocate the next calibration batch to

\[
(A,d)^*=\arg\max_{A,d}
\Pr\bigl(A\text{ can displace current incumbent after observing domain }d\bigr)
/\operatorname{cost}(A,d).
\]

Candidates whose lower confidence bound is worse than the incumbent's upper bound are eliminated.

### 5.3 Why It Should Work

Paired differences cancel much token-level variance shared by parent and candidate. A domain ambiguity set prevents one convenient corpus from silently defining “quality,” while sequential racing avoids uniformly multiplying calibration cost. The score directly targets the selection decision rather than attempting to make every local importance statistic invariant.

### 5.4 Falsifiable Hypothesis

Under the same total calibration-token budget, PDR-NAS will reduce worst-domain selected-architecture regret by at least 20% and increase top-5 selection stability across calibration resamples by at least 15 percentage points versus mean-loss ranking, without degrading average held-out quality by more than 1% relative.

### 5.5 Smallest Canary

**Design only; do not run in Turn 2.**

- Parent model: Qwen2.5-1.5B.
- Model size: 1.5B.
- Dataset: four unlabeled strata (Wikipedia, web, code, mathematical text) with disjoint held-out shards.
- Calibration size: total 32/64/128 sequences × 512 tokens.
- Metric: worst-domain and mean selection regret against a larger held-out oracle; ranking Jaccard/Kendall across resamples; calibration tokens.
- Variable: mean score, worst-domain score, DRO radius, confidence penalty, and adaptive allocation.
- Expected signal: aggressive candidates have heterogeneous domain damage and mean-score winners are unstable.
- Success criterion: ≥20% lower worst-domain regret and ≥15-point stability gain at equal tokens, with ≤1% mean-quality penalty.
- Failure criterion: candidate rankings are stable already, GPrune-style robust local scores match the result, or robustness only trades away average quality.
- Estimated GPU memory: 18–28 GB.
- Estimated runtime: 2–5 A800-hours; planning estimate.

### 5.6 Strong Baselines

- Simple baseline: mean calibration perplexity/teacher KL.
- Closest-paper baseline: GPrune-LLM's cross-distribution scoring adapted to the same structural space.
- Reviewer baseline: stratified mean, worst-domain mean, bootstrap lower-confidence ranking, and larger random calibration set.
- Killer baseline: spend the same tokens on a single diverse coreset or on direct held-out candidate validation.

### 5.7 Main Novelty Claim

“Our method differs from prior work because it treats low-cost NAS as a statistically uncertain whole-architecture selection problem, combining paired distributional robustness with adaptive confidence racing rather than merely changing local importance scores or pooling multiple calibration domains.”

### 5.8 Strongest Reviewer Rejection

“GPrune-LLM already addresses cross-distribution calibration bias, and standard DRO/bootstrapping plus more diverse data is enough; this is statistics wrapped around pruning rather than a new NAS algorithm.”

### 5.9 Response to Reviewer

The response must show a selection-specific failure that local robust scores cannot fix, derive the paired racing procedure, and demonstrate a better quality–token frontier than larger/diverse calibration. If the method's gain comes only from seeing more domains, the reviewer is correct.

### 5.10 Main Idea Kill Criteria

- Whole-architecture rankings do not vary materially across calibration shards.
- Simple stratification/bootstrap or GPrune-LLM matches PDR-NAS.
- Robust selection consistently harms average and downstream quality.
- Adaptive racing spends as many tokens as direct validation.
- The contribution cannot be separated from generic off-the-shelf DRO.

==================================================
FALLBACK IDEA
==================================================

## 6. Fallback Idea

### 6.1 When to Switch

Switch if robust objectives are too conservative or reviewers correctly regard them as generic, but ranking instability is demonstrably caused by an unrepresentative tiny calibration set.

### 6.2 Alternative Algorithmic Idea

Use **Architecture-Disagreement Coreset Selection (ADCS)**. Rather than robustifying scores on fixed data, jointly choose calibration examples that maximally distinguish plausible architecture pairs. Begin with a diverse seed, compute per-example pairwise loss-difference signatures for a small candidate committee, and select examples that cover unresolved ordering boundaries. Rebuild the committee as search progresses.

### 6.3 Why It Is Non-equivalent

PDR-NAS changes the risk objective and quantifies uncertainty for fixed observed samples. ADCS changes which samples are observed, aiming to identify rankings efficiently; it can retain an ordinary mean score. It remains useful if worst-case robustness is unnecessarily conservative.

### 6.4 Mathematical Form

For candidate committee (mathcal C), define an example signature

\[
v(x)=\left[\ell(A;x)-\ell(A';x)\right]_{(A,A')\in\mathcal P(\mathcal C)}.
\]

Choose coreset (S), (|S|\le m), to maximize weighted pairwise coverage

\[
\max_S\sum_{(A,A')}w_{AA'}
\log\left(\epsilon+\sum_{x\in S}
\frac{|v_{AA'}(x)|}{\widehat\sigma_{AA'}(x)+\epsilon}\right)
-\lambda\sum_{x\ne x'\in S}\operatorname{sim}(x,x').
\]

Weights emphasize pairs near the current decision boundary; greedy submodular selection gives an efficient approximation.

### 6.5 Why It Could Still Work

Architecture selection needs examples that expose differences among candidate structures, not merely a representative language sample. Informative pairwise examples can improve rank identification even when no single distributionally robust objective is desirable.

### 6.6 Falsifiable Hypothesis

At 32–64 sequences, ADCS will match the ranking accuracy of 4× random calibration data and transfer its selected architecture to at least two held-out domains better than diversity-only or high-loss coresets.

### 6.7 Smallest Canary

**Design only.** Use 20–40 fixed candidates, a 1.5B model, a 2,000-example unlabeled pool, and select 32/64 examples. Compare random, embedding diversity, high parent loss, gradient diversity, and ADCS. Success is equivalent top-5 recall/Kendall to 4× random and stable held-out selection; failure is selection overfits the committee or requires scoring the whole pool with every candidate. Estimated memory: 18–28 GB; runtime: 3–6 A800-hours.

### 6.8 Strong Baselines

- Random and stratified random calibration.
- Embedding-diversity/k-center coreset.
- Parent-loss/KL-based data selection.
- GPrune-LLM's calibration treatment or the calibration-data study's best heuristic.

### 6.9 Novelty Risk

**HIGH.** Active data/coreset selection is mature, and scoring a large pool can erase savings. Architecture-pair disagreement and end-to-end selection regret must be essential.

### 6.10 Fallback Kill Criteria

- Pool-scoring cost exceeds saved candidate evaluation.
- Selected examples do not transfer beyond the committee.
- Standard diversity or high-loss sampling matches ADCS.
- Gains disappear on a second model family.

==================================================
DIRECTION-LEVEL ASSESSMENT
==================================================

## 7. A800 Feasibility

- Model: 1.5B primary, 3B replication.
- Memory: 18–30 GB; forward-only candidate scoring.
- Compute: many small paired evaluations; no recovery training.
- Dataset: public unlabeled text divided into transparent domains; evaluation remains disjoint.
- Training requirement: small statistical/surrogate fits only.
- Calibration requirement: explicitly 32–128 sequences.
- Expected time: canary 2–6 GPU-hours; full paper 180–350 A800-hours.
- Feasibility verdict: **VERY HIGH**.

## 8. Top-Tier CCF-C Evidence Requirement

Required: a reproducible selection-instability phenomenon; an architecture-level uncertainty derivation; equal-token comparisons against large/diverse calibration, GPrune-LLM, bootstrap/DRO, and data-selection baselines; transfer across models, budgets, and unseen domains; ablations for pairing, hierarchy, ambiguity radius, and racing; selection-valid confidence diagnostics; compute accounting; and negative cases where task-specific calibration is preferable. The paper must show that the NAS decision—not just a neuron score—needs statistical correction.

## 9. Direction-Level Risks

- Novelty risk: **HIGH** after GPrune-LLM.
- Experiment risk: **LOW–MEDIUM** because calibration sensitivity is well supported.
- Compute risk: **LOW**.
- Reviewer risk: **HIGH**—could be viewed as generic robust statistics.
- Saturation risk: **HIGH**—2026 work directly targets generalization-aware pruning.

## 10. Final Direction Verdict

**PROMISING BUT NEEDS REFRAMING**
