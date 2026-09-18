# H1 protocol: Additive Breakdown

**Status:** design only; no formal H1 run or scientific verdict. The companion
[`h1_config.yaml`](../../../configs/canary/direction_01/h1_config.yaml) is a
frozen design record, not an input accepted by an existing runner. The current
`smoke/run_smoke.py` accepts only its smoke JSON and explicitly labels its output
preliminary. The older `paper5/experiments/interaction_canary.py` is a separate
generic interaction probe. A dedicated H1 entry point and per-sequence evidence
writer would be needed after review; neither is implemented here.

## 1. Research question

For two structural edits at decoder blocks $i$ and $j$, do their single-edit
changes in next-token negative log-likelihood (NLL) predict the joint-edit
change well enough to rank candidate architectures? This tests the additive
assumption that motivates Direction 1, before testing any error-transport
algorithm.

## 2. Hypothesis

For a fixed evaluation split $S$, let $L_S(A)$ be token-weighted mean NLL of
the parent with edit set $A$. Define

$$
\Delta_{i,S}=L_S(\{i\})-L_S(\varnothing),\quad
\Delta_{ij,S}=L_S(\{i,j\})-L_S(\varnothing),\quad
I_{ij,S}=\Delta_{ij,S}-(\Delta_{i,S}+\Delta_{j,S}).
$$

H1 predicts material, stable $I_{ij}$ values and decision-relevant pair
misranking. The practical additive null predicts small residuals and near-perfect
ranking. The **primary** interaction calculation uses singles and pairs on the
same held-out validation windows, so calibration-to-validation shift is not
mistaken for structural interaction. A secondary deployment diagnostic uses
calibration singles to predict validation pairs and reports the additional
cross-split error separately.

## 3. Model choice

Use the base, non-instruction-tuned
[`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B), pinned to
revision `8faed761d45a263340a0528343f099c05c9a4323`, in `bfloat16`
evaluation mode on one A800. Its pinned
[configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/8faed761d45a263340a0528343f099c05c9a4323/config.json)
has 28 decoder blocks. This follows the 1–1.5B first-canary choice in the
[Direction 1 research opportunity](../../../ideas/direction_01_causal_error_transport.md).
The GPU index is deliberately absent from the YAML and would be selected at
launch through `paper5/scripts/select_gpu.sh`. No model weights are downloaded
in this design phase.

## 4. Dataset choice

Use the raw WikiText-2 `train.txt` and `valid.txt` files in
[PyTorch examples](https://github.com/pytorch/examples/tree/main/word_language_model/data/wikitext-2),
the same source family as the smoke test. The calibration set is 96 non-overlapping
512-token windows sampled from `train`; the held-out validation set is 64 such
windows from `valid`. Freeze the seed, tokenizer, source-file SHA-256 digests,
selected window indices, and package versions in any future run artifact. Reject
a source hash mismatch or insufficient windows before evaluation. Do not use
WikiText-2 test data. All candidate architectures see the same windows within
each split; the three disjoint 32-window calibration shards assess stability.
The hashes in the YAML come from the local smoke artifact and must be checked
against the actual files before an H1 run; they are not H1 evidence.

## 5. Edit definition

Use the existing `whole_transformer_block_skip` operation: replace one decoder
block's output with its residual-stream input through a temporary forward hook,
then restore the parent model. Use zero-based block indices
`[1, 3, 6, 8, 10, 12, 15, 17, 19, 21, 24, 26]`, the evenly spaced 12-block
selection with one excluded block at each edge of a 28-block model. A pair
applies both hooks simultaneously. Before scoring, require finite parent NLL,
no-edit equality, hook restoration, and order-independent pair effects.

## 6. Number of edits

Score one parent, 12 unique single edits, and all $\binom{12}{2}=66$ unique
two-edit architectures on each split. The edit family and locations are fixed
before seeing H1 validation results. This is a diagnostic of block removal; a
positive result alone does not establish generality to locally distilled block
replacements or prove that the proposed transport method works.

## 7. Pair sampling strategy

Use exhaustive unordered pairs from the fixed 12 locations, sorted by
`(first_layer, second_layer)`. There is no adaptive pair selection and no
validation-driven resampling. Report effects by layer distance and by the depth
of the upstream edit as descriptive slices; keep all 66 pairs in primary metrics.

## 8. Metrics and variance analysis

- **Additive error:** $\widehat\Delta_{ij,S}=\Delta_{i,S}+\Delta_{j,S}$;
  report mean and median $|\Delta_{ij,S}-\widehat\Delta_{ij,S}|$, RMSE, and
  scatter against measured $\Delta_{ij,S}$ in NLL units.
- **Interaction residual:** report signed $I_{ij,S}$, its sign counts, and the
  median of $|I_{ij,S}|/\max(|\Delta_{i,S}|+|\Delta_{j,S}|,10^{-12})$.
  Flag pairs whose denominator is below `0.01` NLL separately so small singles
  do not create a misleading ratio-only claim.
- **Rank correlation:** report Spearman $\rho$ and Kendall $\tau_b$ between
  predicted and measured pair damage, plus overlap of the 17 lowest-damage
  pairs (the top quartile of 66, rounded up). Lower NLL damage is better.
- **Variance:** future H1 code must retain per-window NLL for the parent, each
  single, and each pair. Compute paired window-level residuals
  $I_{ij,s}=L_s(\{i,j\})-L_s(\{i\})-L_s(\{j\})+L_s(\varnothing)$.
  Resample the 64 validation windows **jointly across all architectures** for
  2,000 paired bootstrap replicates; report 95% intervals for median relative
  residual, additive MAE, Spearman $\rho$, Kendall $\tau_b$, and top-quartile
  overlap. Do not bootstrap the 66 overlapping pairs as independent units.
  Compare residual signs and rankings across the three disjoint calibration
  shards. These intervals describe sensitivity to sampled windows from one
  corpus, not cross-domain generalization.

## 9. Expected A800 runtime

The planned workload is 79 architectures per split and 6,471,680 scored input
tokens in total (79 × (96 + 64) × 512), before repeated safety checks. Plan for
**1–4 hours on one A800**, excluding first-time downloads and implementation.
This is an unmeasured scheduling estimate; the RTX 3050 smoke timing and the
Direction 1 full-canary estimate cannot establish A800 throughput for this
1.5B H1 design. A later approved run must record actual wall time and peak
memory and stop for review if the workload exceeds the planned budget.

## 10. Failure criteria and interpretation

Use the existing generic interaction probe's point-estimate thresholds as an
explicit H1 diagnostic, rather than tuning them after viewing validation:

- **Problem confirmed:** median relative residual $\ge0.10$, Spearman
  $\rho\le0.90$, and top-quartile overlap $\le0.75$, all on validation.
- **Weak signal:** exactly two of those three conditions hold.
- **Problem not confirmed:** median relative residual $<0.03$, Spearman
  $\rho\ge0.95$, and overlap $\ge0.90$.
- **Inconclusive:** all remaining point-estimate cases. Report the intervals
  alongside every point decision. A point-estimate confirmation is not treated
  as stable H1 support unless the bootstrap lower bound for median relative
  residual is above `0.03`, and the upper bounds for Spearman and overlap are
  below `0.95` and `0.90`, respectively. Otherwise report it as an unstable
  point signal.

Stop without an H1 verdict if source hashes, model revision, edit semantics,
finite metrics, or restoration checks fail. If residuals are nonzero but ranking
remains reliable, H1 has not shown a decision-relevant additive breakdown.
Even a confirmed H1 supports only this one model, edit family, and corpus;
Direction 1 remains a Research Opportunity, not a Paper Candidate. No H2
transport or formal H1 execution is authorized by this document.
