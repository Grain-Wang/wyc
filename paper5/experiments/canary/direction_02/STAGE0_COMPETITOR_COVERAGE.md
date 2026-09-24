# D2-Stage0 strongest-competitor coverage supplement V1

This is the single authorized post-V2 exploratory supplement, based on archive
`123cd0ca2c7011a8bd1560aee47984954fe09a20` (V2 execution
`53182b532ad8980eb0c4f09f122be941e2671b79`). It does not replace the original
eight-candidate experiment, its `NO_MATERIAL_SELECTION_GAP` and
`SIMPLE_BASELINE_SUFFICIENT` decisions, the V1 pilot stop, or capacity results.
V2 S and E results have already been viewed. E is a reused internal exploratory
evaluation set; this is not blind preregistered confirmation.

## Verified coverage gap and fixed addition

The historical calibration-only ranking over 66 pairs × 96 windows has
`(12,15)`, `(10,15)`, `(10,12)` in positions 1–3. The original competitive pool
was its first 16 pairs. The old rule retained the first pair and selected seven
others by ascending SHA256 of UTF8(`20260923:candidate:`) concatenated with
compact JSON of the pair. Recomputing this rule must reproduce the archived
manifest exactly (SHA256
`8ef24b80b4a2a58bdb384ec4cc4c21e86704bed9b50f3e6245a591e8437bc60e`).
The two runners-up were not among a00–a07. If this check fails, stop.

Add only a08=(10,15) and a09=(10,12), using original zero-based block indices.
Each runs seeds 17 and 29 for 100 updates, from pristine inherited weights,
fresh LoRA and AdamW. The original 16 trajectories, S/E measurements and parent
reference are reused read-only, without training or model reevaluation.

All scientific settings and T/S/E hashes from QUALITY V2 are retained:
Qwen2.5-1.5B revision 8faed761d45a263340a0528343f099c05c9a4323; physical removal
of two decoder blocks; bf16 base; FP32 LoRA r=8, alpha=16, dropout=0 on retained
q_proj/v_proj; AdamW lr=1e-4, weight_decay=0, fixed LR, clip=1; sequence512,
microbatch1, accumulation4; checkpoints 0/20/50/100 on one continuous trajectory.
Each seed has the same 400-window order for every candidate. Each new trajectory
uses 204800 input tokens / 204400 effective labels, not additional unique data.
No new pilot, parent recovery, candidate, seed, tuning, or adaptive stopping.

## Information barrier and analysis

All four new trajectories and their S scores must be complete before persisting
the expanded ten-candidate S scores and choices. New E tokens cannot be opened
until that record is verified against all 20 trajectory records and endpoint
hashes. E is never used for policy replay, training decisions or filtering.
The new per-window file contains only new measurements; an explicitly merged
analysis file contains the original rows unchanged plus the new rows.

Per seed: B0 selects by S@0; B20 selects by S@20; BSH uses 10@20 → 5@50 →
2@100 and selects by S@100; Ref100 selects from all ten by S@100. Ties use ID.
Every strategy evaluates its selected 100-step endpoint. Training-update
information budgets are respectively 100, 280, 450, 1000. Record allowed S
queries, S/E evaluation, historical preselection, construction and I/O costs.
Offline replay is not measured acceleration; reused computation is not free.

Reuse the original label-weighted NLL and document-cluster paired bootstrap:
2000 draws, seed20260924, joint resampling across models, separately reported
training seeds. G0=E(B0@100)-E(Ref100@100) can be negative. Also report finite-set
E-hindsight simple regret and its selection-bias limitation, S ranking changes,
Top1/Top3, Spearman/Kendall, intermediate quality and recovery curves.
These post-hoc intervals are exploratory descriptions only.

The PPL-ratio quality threshold remains 1.15 against the same unmodified parent:
at least two candidates and Ref100 must pass at 100-step E in each seed.
Initial and intermediate quality cannot remove candidates. The material gap
remains 0.02 nats/token. Original classification rules are unchanged, with
quality, problem signal and simple-baseline sufficiency shown separately.
Compare original eight and expanded ten results in separate report sections.
Do not attribute gains over an unadapted parent to compression.

## Resource accounting and failure boundary

Use only explicitly shared physical GPU2, UUID
GPU-5c9c1a6d-52ff-000a-11e9-be3f1b36d43d, after permission/duplicate checks and
three short resource samples. Never affect existing processes or switch cards.
Reuse the D2 environment/model/token cache and the original shared task lock.
Additional GPU time is at most 3600 seconds and also counts toward the original
14400-second limit. Inherit the archived 2375.137250719592 seconds and verify
the live V2 ledger matches; do not reset elapsed time for this run ID.

Before launch freeze a conservative projection using observed update, window,
construction and checkpoint costs, original 1.5 safety factor and 300-second
reserve. Count 400 updates, 1024 S windows, 256 E windows, eight constructions,
16 checkpoint writes. Reject if either budget is infeasible. Detached launcher
has an outer deadline; the runner checkpoints and stops before its deadline.
Record heartbeat, PID, exit status and raw logs privately; cumulative ledger
includes startup, loading, training, evaluation, persistence and failures.
CPU analysis retains the original 1800-second ceiling.

NaN/OOM, mapping/data/state errors, incomplete trajectories or budget/resource
failure stop the attempt; preserve evidence and make no survivor-only conclusion.
No automatic scientific changes or repeat submission.

If B0 still shows no material selection loss, recommend ending this candidate
family/recipe probe and pausing complex D2 predictors. A usable signal first
requires checking B20/BSH; unstable/one-seed evidence remains inconclusive.
No automatic extra coverage or FHRD/RSC/Hessian/NTK implementation follows.

## Outputs and provenance

Independent run: `20260924_competitor_coverage_01`, under
`paper5/results/canary/direction_02/recovery_competitor_coverage/`.
Freeze config, source hashes, expanded candidate manifest, data manifest,
budget and protocol before code commit. Record that exact execution SHA in
runtime/checkpoints; the later result archive SHA is a distinct commit.
Preserve all historical protocols/configs/results byte-for-byte.

Archive small numerical results, manifests, curves, strategy/cost/runtime
records and sanitized audit/handoff only. Raw logs, tokens, source corpus,
credentials, connection details and adapters stay outside Git. Ordinary push
only to Grain-Wang/wyc refs/heads/paper5, followed by independent GitHub readback.
Stop after this supplement.
