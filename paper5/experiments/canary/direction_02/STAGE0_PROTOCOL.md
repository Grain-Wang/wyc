# D2-Stage0 V1: fixed-budget recovery and architecture selection

Status: authorized implementation; GPU feasibility and quality freeze pending.
The user selected Direction 2 on 2026-09-23. D1's current whole-block-skip route
ends at `6ee61e753d8e67b21c56adfb2edeec082e076f10`; CET is paused. Preserve
H1-small INCONCLUSIVE and multi-edit STOP_COMPLEX_METHOD_INVESTMENT verbatim.
The original Direction 2 idea is unchanged. This is a LoRA + next-token CE / NLL
problem probe, not a reproduction of its teacher-KL FHRD proposal. No Hessian,
NTK, JVP/VJP predictor, NAS searcher, RSC, H2, or other direction is authorized.

## Question and prior-work check

Does choosing before recovery lose a materially relevant amount of terminal
quality, compared with choosing after the same fixed recovery recipe?
Rank reversal alone is neither a novelty claim nor a Paper Candidate gate.

Primary-source check on 2026-09-23, bounded to less than 30 minutes:

- [DarwinLM, v3, sections 3.4–3.5 / Algorithm 1](https://arxiv.org/html/2502.07780v3):
  training-aware selection progressively trains and eliminates offspring. Its
  KL-based fitness, fine-grained pruning and evolutionary search differ from this
  small CE/LoRA depth probe. Short training is a necessary strong comparator.
- [TraceNAS, v1, sections 3.4 and 4](https://arxiv.org/html/2602.02891v1):
  a sparsity-weighted Pearson correlation of low-rank gradient traces measures
  alignment with the parent. It explicitly targets recovery potential. We do
  not implement it or treat the general idea of recoverability as new.
- [Minitron, Table 1](https://arxiv.org/html/2407.14679v2): lightweight retraining
  changes relative outcomes of pruning choices; its scale and KD recipe do not
  establish a quantitative quality threshold for this experiment.
- [LLM-Pruner](https://arxiv.org/abs/2305.11627): structural pruning followed by
  LoRA recovery provides precedent, not validation of this particular recipe.

Positive Stage0 evidence would only justify considering a subsequent method
feasibility study. Negative evidence is limited to this model, pool and recipe.

## Data and information boundaries

No ready small C4/FineWeb subset was found in the available project cache. Use
the existing PyTorch-examples WikiText-2 train text, SHA256
`9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f`, as expressly
authorized fallback. Preserve its distributed preprocessing, including `<unk>`
and tokenized punctuation; do not call it a new raw-web corpus.
Source: [PyTorch examples train file](https://github.com/pytorch/examples/blob/main/word_language_model/data/wikitext-2/train.txt).
The SHA pins the actual bytes; the mutable upstream URL is only provenance.

Split on top-level article headings (`= title =`), not sections (`= = ... = =`).
Assign documents by SHA256 of `20260923:split:` + document-content SHA modulo
100: T below 70, S 70–84, E 85–99. Identical document content cannot cross splits.
Within each document tokenize independently, add_special_tokens=False using the
pinned 1.5B tokenizer; discard incomplete tails. Select 400/64/64 nonoverlapping
512-token windows, in deterministic document-round-robin order (hash-sorted
documents and then hash-sorted window offsets). Stop if any split is too small.
Record source byte spans, document IDs/hashes, document token counts, window
offsets and token hashes. No document contributes to more than one split.

D1 used this train source for calibration, so these are **internal exploratory
evaluations**, not blind tests or cross-corpus evidence. Document-disjointness
within D2 is mandatory even though historical non-exposure is not claimed.
Preparation/tokenization happens locally; send verified token files to the A800.
The GPU pilot and formal training open only T/S. E's separate token file is read
only after all 16 trajectories and S-based strategies have been persisted.
The E manifest can be audited beforehand; no E loss may inform a run decision.

## Candidate construction and physical structure

Read only calibration rows of the frozen D1 H1-small per-window CSV. Rank its
66 two-block architectures by mean calibration NLL (lexicographic tie break).
Take the top 16 as a competitive pool; keep the best, then take seven others by
SHA256(`20260923:candidate:` + compact JSON of the pair). Freeze all eight before
recovery. Do not consult D1 validation labels, recovery losses or E. Report the
sunk cost of 66 × 96 candidate calibration windows plus 96 parent windows; no
claim that historical information is free. The D1 hook scores only preselect;
real D2 candidate losses are remeasured on S.

Every candidate inherits the same parent and physically removes exactly two of
28 decoder blocks. Replace the layer ModuleList with the 26 retained objects,
update num_hidden_layers and attention layer_idx, disable KV caching. Verify
parameter mapping, retained forward order and the absence of removed-block
execution; no output-replacement hook implements compression. No speedup claim.
All 26 retained layers have q_proj/v_proj LoRA of identical capacity.

## Pilot and quality gate

One preselected candidate (best historical calibration score), seed 17, at most
10 optimizer updates. No E evaluation. Measure parent and all eight initial
candidates on S, before recovery; these are needed to assess pool feasibility.
Verify finite loss/gradients, real optimizer updates, strictly frozen base,
nonzero adapter change, physical mapping and save/reload equivalence. Zero B
initialization can give zero A gradients initially; not every matrix must have
a nonzero first-step gradient. The pilot adapter and optimizer are discarded.

Proposed quality definition is NLL excess ≤ log(1.15), i.e. perplexity at most
15% worse than the unmodified parent on the same split. This is an explicit
Stage0 working tolerance, not a literature standard or D1's 0.30 rule. The
pilot report must show the parent, all initial S losses and the 10-step S loss,
and explain whether the tolerance leaves at least two meaningful competitors.
Formal entry requires at least two initial S candidates within this tolerance,
a finite pilot endpoint within it, and engineering/budget checks passing. This
conservative gate may reject recoverable candidates; it does not disprove
recovery if 10 steps are insufficient. Do not relax the tolerance to pass.

After pilot review, persist a quality/budget freeze referring to its hashes and
this protocol, commit that freeze and synchronize the exact commit before formal
training. Final quality is assessed separately on E: in each seed at least two
100-step candidates AND Ref100 must satisfy the same frozen excess-NLL bound.
No E-driven threshold or recipe revision. Engineering failure, inadequate
pilot quality, or infeasible budget stops the run and preserves evidence.

## Fixed recipe and checkpoints

Executable configuration: `paper5/configs/canary/direction_02/recovery_stage0.json`.
Parent bf16, no quantization. LoRA r=8, alpha=16, dropout=0 on q_proj/v_proj only;
FP32 adapter master parameters/AdamW states, bf16 autocast forward. AdamW
lr=1e-4, weight_decay=0, constant LR, clip norm 1.0. Next-token cross-entropy;
no teacher, label smoothing or auxiliary objective. Sequence 512, microbatch 1,
accumulation 4. Only LoRA parameters train. Gradient checkpointing is an
implementation memory option; preserve the fixed optimization/data semantics.

Eight candidates × seeds 17/29, 100 updates each, checkpoints 0/20/50/100 on
one uninterrupted trajectory. Same seed gives the same shuffled T order for
all candidates, exactly 400 windows, no replacement. Each trajectory starts
from pristine inherited parent weights and freshly initialized adapters and
optimizer; no cross-candidate/pilot updates. Per trajectory: 204800 input tokens,
204400 effective next-token labels, 400 unique windows; report unique documents.
S evaluation uses the same 64 windows at each checkpoint with dropout disabled.
Save CPU adapter/optimizer/RNG state in D2 cache; do not commit weights.

## Frozen selection and evaluation

For each seed, persist all choices before E is opened. Stable ties use candidate
ID. Every strategy is evaluated at its selected candidate's **100-step** endpoint:

| Strategy | Information used | Training updates |
| --- | --- | ---: |
| B0 | all 0-step S, train winner | 100 |
| B20 | all 20-step S, continue winner | 240 |
| BSH | 8 at 20 → 4 at 50 → 2 at 100; final S winner | 380 |
| Ref100 | all 100-step S; expensive reference | 800 |

Full trajectories permit offline replay, but BSH must only query surviving
candidates at each stage. All E comparisons use 100-step models plus the parent.
Retain per-window and per-document losses. Primary G0 = E(B0@100) − E(Ref100@100),
possibly negative. Report B20/BSH differences, finite-pool E hindsight simple
regret (selection-biased reference), 0→100 S Top-1/Top-3, Spearman/Kendall, curves,
and damage recovery. Initial damage reduction is measured on S; initial candidate
E losses are not evaluated, so do not claim an E initial-to-final recovery curve.

Token-weighted NLL is the point estimand. Resample E **documents**, retaining all
their windows and using the same 2000 bootstrap draws across candidates/seeds.
Recompute token-weighted ratios per draw. Report per-seed paired percentile 95%
intervals, not seed-pooled significance. Hindsight minimum is reselected per
draw and its regret intervals are descriptive. Only 100 steps is the endpoint.

## Costs, limits and decisions

Report separately measured total audit cost, policy information costs and
estimated standalone execution time. Count S/E forwards, candidate construction,
adapter I/O and historical preselection. Policy replay is not measured speedup.
For engineering feasibility, multiply pilot-measured step/forward/loading/I/O
cost projection by 1.5, add 300 seconds reserve, and include spent pilot time.
Stop if the complete 16 trajectories plus evaluation cannot fit 14400 seconds.
Pilot ≤1800 seconds; CPU analysis ≤1800 seconds. Count all GPU-process elapsed
time across phases/retries; reserve time for checkpointing before deadline.

Use one A800; inspect current process occupancy and free memory, leaving ≥4096
MiB beyond observed peak (pilot admission initially needs ≥24576 MiB free).
Never terminate another process. Detached screen/tmux, exclusive task lock,
PID/heartbeat/private log/exit code and cumulative budget ledger are mandatory.
NaN, base mutation, split contamination, mapping errors, missing trajectory or
budget failure stop the task. Never drop a failed candidate or silently reduce
the pool/seeds. A saved partial run is not a completed experiment.

Report separate engineering, quality, problem and simple-baseline fields:

- RECOVERY_PIPELINE_FAILED for engineering/data/optimizer failure; no inference.
- QUALITY_INFEASIBLE for too few quality-meaningful candidates under this recipe.
- NO_MATERIAL_SELECTION_GAP if both seeds' G0 upper95 ≤0.02.
- PROBLEM_SIGNAL only if quality passes and both G0 >0.02 with lower95 >0.
- Otherwise INCONCLUSIVE; non-significance does not prove equivalence.
- Separately SIMPLE_BASELINE_SUFFICIENT if the same B20 or BSH has both
  difference upper95 ≤0.02, selected models satisfy quality, and its update and
  estimated total cost are lower than Ref100. The working cost requirement is
  ≤50% of Ref100 updates and strictly lower estimated standalone seconds.

The 0.02 nats/token discussion margin is a working choice, not a quality cutoff.
No outcome authorizes a complex method or another direction.

## Provenance and publication

Only D2 paths and necessary CURRENT.md changes; preserve original AutoResearch
worktree and D1 bytes. Record pilot code SHA, formal executed SHA and subsequent
archive SHA separately. Commit code before GPU execution and commit post-pilot
freeze before formal training. Ordinary fast-forward publication only to
`Grain-Wang/wyc`, `refs/heads/paper5`; no forced updates/history rewrites.
Archive small reviewed files under
`paper5/results/canary/direction_02/recovery_stage0/20260923_stage0_01/`.
Exclude raw corpus, weights/adapters, caches, raw logs, credentials and machine
connection information. Re-read GitHub ref, result list and hashes at completion.
