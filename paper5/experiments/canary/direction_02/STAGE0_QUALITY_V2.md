# D2-Stage0 V2: quality admission timing supplement

This supplement changes only the timing of quality admission. STAGE0_PROTOCOL.md,
the V1 configuration, pilot QUALITY_INFEASIBLE stop, and exploratory capacity
results remain immutable. All other recipe, selection, quality and statistical
rules in the original protocol apply. The user explicitly authorized this V2
and the original eight candidates × seeds 17/29 after reviewing archive
`d44f48f49f80bb91d489eb80e2ab6ea0d08bc8b9`.

## Evidence and scope of the amendment

Engineering pilot execution: `75e3253b8f8b78c5998730c039ad346231d350bc`.
Capacity continuation execution: `c63ec1a028d8f8fdc901a208b67a82e8b15c51c5`.
The a00/seed17 S PPL ratios at 20/50/100 were observed before this revision:
1.109312888 / 1.000838081 / 0.906947284. This establishes capacity feasibility
for one candidate/recipe only. It does not establish architecture-selection
loss, superiority to short-training policies, or a need for complex methods.
This is internal exploratory problem validation, not a fully blind preregistered
confirmation. The same-seed a00 rerun is not an additional independent replicate.

V2 no longer requires two candidates to meet the final quality bound before
recovery, nor requires the ten-step pilot to meet that bound. Entry instead
requires the completed engineering pilot, completed single-candidate 100-step
capacity evidence, unchanged data/candidates, and resource/complete-budget checks.
Record failed initial and intermediate quality faithfully; do not delete,
replace, or stop individual candidates because of their quality.

The numerical bound is unchanged: PPL(candidate)/PPL(unmodified parent) ≤ 1.15.
Only after all sixteen trajectories and S strategy choices are persisted may E
be opened. For each seed, at least two 100-step candidates AND its S-selected
Ref100 must pass on E. If either seed fails, QUALITY_INFEASIBLE takes precedence
and PROBLEM_SIGNAL is forbidden. The original G0, 0.02 margin, document-cluster
2000-bootstrap intervals, B0/B20/BSH/Ref100 definitions and verdict rules are
unchanged. Initial and 20/50-step S quality are descriptive, not admission tests.

## Execution and immutable inputs

Executable configuration: `paper5/configs/canary/direction_02/recovery_stage0_v2.json`.
Only design/run identity and explicit amendment metadata differ from V1; tests
reject any recipe change. The original eight-candidate manifest and T/S/E data
manifest are copied byte-for-byte. T/S/E token arrays remain in the original D2
cache and are read-only; no retokenization, resampling or split changes occur.
E tokens are hashed/opened only after the persisted selection gate passes.

Reuse run_stage0 training, checkpointing, endpoint evaluation, policy replay and
statistical analysis. Each of the sixteen trajectories starts with pristine
inherited parent weights, fresh LoRA and fresh AdamW. No pilot/capacity checkpoint
is loaded into a formal trajectory. Keep 0/20/50/100 checkpoints and S losses;
evaluate all 100-step models and the unmodified parent on E. No parent recovery
training is added. Do not interpret adapted-candidate gains over that parent as
evidence that compression improves performance.

Before execution, commit the V2 supplement, configuration and formal freeze.
Record the base archive SHA, actual V2 execution SHA and later result archive
SHA separately. Results go only to
`paper5/results/canary/direction_02/recovery_stage0_v2/20260923_stage0_quality_v2_01/`;
private adapters/logs use corresponding independent D2 cache/scratch paths.

## Shared resource and cumulative budget

Only physical GPU2, UUID `GPU-5c9c1a6d-52ff-000a-11e9-be3f1b36d43d`, is authorized
under existing shared-use rules. Require the existing permission/admission checks,
three short consecutive resource samples, duplicate checks, the original D2 task
lock plus the V2 lock, detached session, PID/heartbeat/log/exit code. Never affect
other processes. Retain the measured peak plus memory reserve and stop on resource
failure, OOM, NaN, inconsistent state/data, or incomplete trajectories.

Copy the completed cumulative ledger (pilot + capacity = 418.11011994164437 s)
into the V2 cache; keep the original ledgers unchanged. A new run_id never clears
spent time. Project all 1600 updates, 4096 S + 1088 E windows, 33 constructions and
64 checkpoint writes. Use the more conservative pilot/capacity measured unit
costs, include capacity wall-time overhead in the per-update bound, multiply
remaining work by the original 1.5 safety factor and add 300 s. Stop if this plus
spent time exceeds 14400 s. CPU analysis remains ≤1800 s. No extra pilot, fewer
candidates, changed seeds, extra training or automatic recovery retries.

Report actual audit cost, policy information costs and estimated standalone
costs separately. Publish complete small results and sanitized audit/handoff only;
exclude raw logs, credentials, connection details, corpus and adapter weights.
After GitHub readback verification, stop. No FHRD/RSC or new direction follows.
