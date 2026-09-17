# Direction 2 — Terminal-Recovery NAS under a Fixed Adaptation Budget

## 1. Core Research Problem

Post-training NAS usually ranks an architecture before recovery, although the deployed model is evaluated after LoRA, distillation, or continued adaptation. How can search predict the terminal quality of each architecture after a fixed, small recovery budget without actually training every candidate?

## 2. Why This Problem Matters

Immediate compression damage and recoverable damage are different. Minitron reports that lightweight retraining can change architecture rankings; DarwinLM spends multistep training inside search; TraceNAS designs a proxy around recovery alignment. If the search target is wrong, low-cost search either rejects a highly repairable architecture or selects damage that the allowed adapter cannot fix. A finite-horizon recovery model could turn recovery from an afterthought into the search objective and substantially reduce candidate-training cost.

## 3. Closest Prior Work

1. [Minitron](../reference_papers_processed/2024_Minitron.md): candidate retraining stabilizes/reveals rankings but is costly.
2. [TraceNAS](../reference_papers_processed/2026_TraceNAS.md): whole-model gradient-trace correlation as a recovery-alignment proxy.
3. [LLM-Pruner](../reference_papers_processed/2023_LLM_Pruner.md): structured pruning followed by LoRA recovery.
4. [DarwinLM](https://arxiv.org/abs/2502.07780): progressive multistep training within evolutionary candidate selection.
5. [Sheared LLaMA](../reference_papers_processed/2024_Sheared_LLaMA.md): architecture pruning plus expensive continued pretraining.

## 4. Exact Gap

Existing methods use immediate quality, first-order similarity to the parent, or actual short training. None in the audited set predicts a named optimizer/adapter's **finite-horizon terminal loss** from low-rank local dynamics while avoiding per-candidate recovery. The missing objective jointly represents initial damage, alignment with feasible update directions, curvature/conditioning, and a fixed token/step budget.

==================================================
MAIN IDEA
==================================================

## 5. Main Idea

### 5.1 Core Algorithmic Idea

Define **Finite-Horizon Recovery Dynamics (FHRD)** for each candidate. Around the inherited candidate, approximate the loss dynamics of a fixed LoRA (or selected-weight) recovery procedure using low-rank Hessian/NTK sketches. Instead of asking whether gradients resemble the parent once, predict how much candidate damage remains after (K) specified optimizer steps. Search minimizes predicted terminal held-out teacher KL plus architecture cost. A multi-fidelity evaluator computes cheap spectral moments for all candidates and exact short recovery only for an uncertainty-selected calibration subset.

### 5.2 Mathematical Form

For architecture (A), let (phi) be its allowed recovery parameters, (e_A) the teacher–student logit residual at (phi_0), and (J_A=\partial z_A/\partial\phi). Under squared/logit-linearized distillation,

\[
H_A\approx J_A^\top WJ_A+\lambda I,\qquad
g_A=J_A^\top We_A.
\]

For (K) gradient steps with step size (eta),

\[
\phi_K-\phi_0\approx-\eta\sum_{t=0}^{K-1}(I-\eta H_A)^t g_A.
\]

The predicted terminal residual is

\[
\widehat e_{A,K}=e_A-J_A\eta\sum_{t=0}^{K-1}(I-\eta H_A)^t g_A,
\]

and the search objective is

\[
\min_{A:C(A)\le B}\;\widehat{\mathcal L}_{K}(A)
=\|\widehat e_{A,K}\|_W^2+\beta\,U_K(A),
\]

where (U_K) is uncertainty from sketching and linearization. Lanczos/Hutchinson sketches estimate the required spectral action without forming (H_A).

Pseudocode sketch:

1. Fix recovery family, optimizer, (K), and token budget before search.
2. For each candidate, compute initial residual, adapter JVP/VJP sketches, and low-order Hessian spectral moments.
3. Predict terminal loss and uncertainty.
4. Actually recover only high-value/high-uncertainty candidates to update a residual calibrator.
5. Search/shortlist by predicted terminal loss under the structural constraint.

### 5.3 Why It Should Work

Two architectures with equal immediate loss can expose very different residuals to the feasible adapter subspace and have different conditioning. The linearized dynamics explicitly captures both facts and the finite step budget. The method targets the same quantity used to choose the deployed model while amortizing expensive recovery across only a small calibration subset.

### 5.4 Falsifiable Hypothesis

FHRD will predict post-$K$-step candidate rankings with Kendall $\tau\ge0.6$ and improve it by at least 0.15 over immediate loss, TraceNAS, gradient norm, and one-step loss decrease, while using at most 10% of the recovery tokens required to train every candidate.

### 5.5 Smallest Canary

**Design only; do not run in Turn 2.**

- Parent model: Qwen2.5-0.5B for implementation validation, then Qwen2.5-1.5B.
- Model size: 0.5B/1.5B.
- Dataset: 90% FineWeb/C4 calibration for teacher KL, 10% disjoint holdout; downstream sanity set not used for selection.
- Calibration size: 64 sequences × 512 tokens.
- Metric: Kendall $\tau$ between predicted and actual 20/100/500-step LoRA recovery rankings; token-normalized selection regret.
- Variable: architecture (depth/FFN/head reductions), $K$, LoRA rank, and curvature approximation rank.
- Expected signal: initial-loss rankings flip after recovery, and predicted terminal dynamics anticipates flips.
- Success criterion: $\Delta\tau\ge0.15$, $\tau\ge0.6$, and best selected candidate within 2% relative held-out KL of the oracle retrain-all choice at ≤10% search recovery tokens.
- Failure criterion: rankings do not flip, linear prediction fails after 20 steps, or direct successive halving is cheaper and better.
- Estimated GPU memory: 40–65 GB with sequential curvature-vector products.
- Estimated runtime: 6–12 A800-hours for a 12–20-candidate canary; planning estimate.

### 5.6 Strong Baselines

- Simple baseline: immediate calibration perplexity/teacher KL.
- Closest-paper baseline: TraceNAS gradient-trace score.
- Reviewer baseline: one-step gradient improvement / gradient norm and Minitron-style short retraining.
- Killer baseline: DarwinLM progressive multistep racing with exactly matched recovery tokens; oracle retrain-all is the upper bound.

### 5.7 Main Novelty Claim

“Our method differs from prior work because it predicts architecture-specific quality after a specified finite recovery process using sketched optimization dynamics, rather than scoring unrecovered candidates, correlating one global gradient snapshot, or training every candidate during search.”

### 5.8 Strongest Reviewer Rejection

“DarwinLM is already training-aware and TraceNAS already predicts recoverability; a linearized Hessian model will be less accurate and more complicated than spending the same budget on successive-halving recovery.”

### 5.9 Response to Reviewer

This is the decisive objection. The response must be an equal-token/equal-wall-clock Pareto curve showing that FHRD wins in ranking regret before DarwinLM can meaningfully train enough candidates, plus ablations showing finite-horizon curvature—not generic learned calibration—causes the gain. If no such regime exists, the main idea is not publishable.

### 5.10 Main Idea Kill Criteria

- Actual post-recovery rankings are already predicted by immediate loss or TraceNAS.
- Linearized terminal predictions are unstable across recovery horizons or LoRA ranks.
- DarwinLM/successive halving dominates at equal tokens and wall clock.
- Curvature sketches cost more than directly recovering the candidate pool.
- The method only works at 0.5B and fails at 1.5B/3B.

==================================================
FALLBACK IDEA
==================================================

## 6. Fallback Idea

### 6.1 When to Switch

Switch if multi-step dynamics are too nonlinear or optimizer-specific, but candidate damage still differs in whether a fixed low-cost adapter can represent its repair.

### 6.2 Alternative Algorithmic Idea

Use a **Repair-Subspace Certificate (RSC)**. Measure the component of architecture-induced functional error that lies outside the tangent subspace spanned by allowed recovery parameters. Rank candidates by irreducible residual under a regularized projection, without simulating optimization steps. Search can also choose where to place a fixed total adapter-rank budget so that structural damage is most coverable.

### 6.3 Why It Is Non-equivalent

FHRD models an optimizer trajectory and finite time; RSC is an optimizer-free geometric certificate of representability. FHRD can distinguish fast from slow recovery within the same subspace, while RSC only separates repairable from irreparable damage and remains meaningful if the linearized time trajectory is inaccurate.

### 6.4 Mathematical Form

With residual (e_A) and adapter Jacobian (J_A), define

\[
\rho(A)=\min_{\Delta\phi}\|e_A-J_A\Delta\phi\|_W^2+\lambda\|\Delta\phi\|_2^2
=\|e_A\|_W^2-g_A^\top(H_A)^{-1}g_A.
\]

Randomized range finding approximates the projection. With per-layer ranks (r_i), jointly optimize

\[
\min_{A,\mathbf r}\;\widehat\rho(A,\mathbf r)
\quad\text{s.t.}\quad C(A)\le B,\ \sum_i r_i\le R.
\]

### 6.5 Why It Could Still Work

Architecture damage outside the feasible update span cannot be removed by any number of small-adapter steps. That geometry is independent of precise learning rates and can remain predictive when early learning curves cross or nonlinear optimization defeats a finite-horizon approximation.

### 6.6 Falsifiable Hypothesis

The irreducible-residual estimate will predict final recovered ranking better than initial loss and gradient norm across at least two optimizers/horizons, and allocated adapter rank will reduce recovery regret under the same parameter budget.

### 6.7 Smallest Canary

**Design only.** Use 0.5B/1.5B models, 12–20 structured candidates, LoRA ranks 4/8/16, and 64×512 calibration tokens. Compare projection residual with actual 500-step recovery across two learning rates. Success requires (Delta\tau\ge0.15) over initial loss on both rates; failure is rank dependence on optimizer that geometry cannot explain. Estimated memory: 35–60 GB; runtime: 5–10 A800-hours.

### 6.8 Strong Baselines

- Immediate teacher KL/perplexity.
- Gradient norm and one-step improvement.
- TraceNAS gradient-trace correlation.
- Uniform LoRA-rank allocation and oracle short recovery.

### 6.9 Novelty Risk

**HIGH.** NTK/linearized fine-tuning and influence-function ideas are mature, and NIRVANA-like pruning work already invokes training dynamics. The architecture-search formulation and irreducible-damage evidence must be clearly new.

### 6.10 Fallback Kill Criteria

- Projection residual does not generalize across calibration shards/optimizers.
- A simple gradient norm is equally predictive.
- Low-rank range finding is not cheaper than short recovery.
- The joint rank-allocation component provides no gain over uniform LoRA.

==================================================
DIRECTION-LEVEL ASSESSMENT
==================================================

## 7. A800 Feasibility

- Model: 0.5B for debugging, 1.5B/3B for claims, optional 7B one-setting confirmation.
- Memory: 40–65 GB with sequential JVP/VJP and curvature sketches.
- Compute: calibration derivatives plus a limited set of short LoRA recoveries.
- Dataset: unlabeled web-text calibration and held-out corpora; public downstream benchmarks only for final validation.
- Training requirement: 20–500-step LoRA/KD only for sparse calibration/oracle points.
- Calibration requirement: 32–128 sequences and at least three shards.
- Expected time: canary 6–12 GPU-hours; full paper 500–900 A800-hours because actual recovery curves and repetitions are necessary.
- Feasibility verdict: **MEDIUM–HIGH** on one A800, with strict candidate-pool and horizon limits.

## 8. Top-Tier CCF-C Evidence Requirement

Required evidence: real rank flips after recovery; terminal-rank prediction across models, structures, horizons, adapter ranks, and data; equal-token comparisons with DarwinLM/successive halving; TraceNAS and immediate-loss baselines; Hessian-rank, uncertainty, and active-calibration ablations; downstream as well as perplexity outcomes; wall-clock/memory accounting; failure cases where linearization breaks; and a mechanism plot tying eigenspectrum/repair subspace to recoverability. The story closes only if the predictor saves enough candidate training to change the feasible search regime.

## 9. Direction-Level Risks

- Novelty risk: **HIGH**—TraceNAS, DarwinLM, and training-dynamics pruning are close.
- Experiment risk: **HIGH**—terminal dynamics may be too nonlinear or rankings may not flip enough.
- Compute risk: **MEDIUM**—curvature and oracle recovery curves are expensive but bounded.
- Reviewer risk: **HIGH**—complexity must beat successive halving, not only immediate scores.
- Saturation risk: **HIGH**—recovery-aware pruning is rapidly developing in 2025–2026.

## 10. Final Direction Verdict

**STRONG RESEARCH OPPORTUNITY**
