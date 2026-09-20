# H1 multi-edit execution record

**Current final status: COMPLETE / STOP_COMPLEX_METHOD_INVESTMENT (V2).**
The single authorized run completed and was verified. Executed code:
`eba72bc9e9f59634e0f2f2639d05c1de1c38f308`. The earlier block below is retained
as the historical pre-amendment stop record. Final evidence and interpretation
are in the last section; H1-small remains INCONCLUSIVE.

**BLOCKED BEFORE CODE COMMIT / SYNCHRONIZATION / MODEL EXECUTION.**
The preflight reconstructed historical 0.5B smoke validation windows from its
archived seed and found source-token overlap with the fixed V_new selection:
smoke 256-token window 324 overlaps new 512-token window 162, and smoke window
784 overlaps new window 392, each by 256 tokens. Tokenizer JSON, vocabulary and
merges are byte-identical across the cached revisions; configuration differences
(chat template and EOS token) are unused by raw-text, no-special-token encoding.
This is partial exposure in a different-model experiment, not an existing full
512-token multi-edit measurement. The conservative unobserved-window requirement
cannot be certified. No candidate, window, threshold or protocol was changed to
work around this finding. The implementation below is an **uncommitted draft**,
not an executed or launch-ready release; do not use it until the isolation issue
has been reviewed under a separately approved decision. No H2 was started.

Approved design: `abf29124a7acfa0c2b16575b7183468d0ef138df`.
Authorization: the user approved one fixed-budget diagnostic after CPU review.
The historical proposal text and H1-small INCONCLUSIVE verdict remain unchanged.
If subsequently authorized and executed, actual code SHA must be recorded
separately in new-run config/state/runtime. No executed-code SHA exists this turn.

This implementation reuses the archived calibration reader, candidate generator,
paired bootstrap, local data SHA checking, model evaluation and block-hook restoration.
No H2 implementation or new k=2 evaluation is added. Safety checks use one approved
k=4 architecture. The normal run consumes 14,464 measurement and 8 safety windows.

Order (run only from the independent, committed `paper5` repository on A800):

1. Verify input hashes, cache completeness, clean tracked tree and no existing job.
2. Activate `autoresearch_paper5`; set `PAPER5_DATA_ROOT=~/whr/paper5/data`.
3. CPU only: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m paper5.experiments.canary.direction_01.run_h1_multi_edit --stage prepare`.
   This freezes candidate/prediction and token manifests; cache weights are hashed
   against the pinned LFS blob. Missing cache/data or failed isolation stops.
4. Launch exactly once using detached screen and `bash paper5/scripts/launch_h1_multi_edit.sh`.
   The launcher dynamically recommends a single A800 and requires 8,192 MiB free
   (4,096 MiB working allowance plus 4,096 MiB reserve). It records occupancy,
   never terminates other processes, and keeps stdout/stderr and exit code under
   the project scratch directory named by the executed code commit.
5. The guarded model phase is capped at 7,200 seconds including imports/loading;
   the separately guarded CPU analysis is capped at 1,800 seconds. Neither retries.
6. Verify all result identities, original file hashes, and completion before archiving.

The manifest excludes old validation windows and their ±1 neighbors; the historical
same-model sanity's two 128-token windows are also checked for token overlap (a
conflict stops rather than changes the approved 64-window selection). The earlier
different-model 0.5B smoke used the same corpus; V_new is internal confirmation,
not an untouched external test. No WikiText test data is read.

CPU tests cover exact candidates, configuration immutability, second-order algebra,
query isolation, window exclusion, completeness, paired bootstrap, selection loss,
budget ceilings, strict decision gates and a full synthetic CPU analysis. Synthetic
results exist only in pytest temporary directories and are never archived as evidence.

Implementation QA: task-specific Black, repository Ruff, applicable CPU pytest and
diff checks are required. Existing unrelated formatting debt must not be changed
to make this task appear responsible for repository-wide formatting.

Implementation self-check before synchronization: 37 CPU tests passed, including
the full synthetic analysis; repository Ruff, new-file Black, shell syntax and
diff checks passed. A tracked-file Black sweep found only the pre-existing
`tools/researchclaw/gpu_queue/state.py` formatting issue, left unchanged.

## V2 isolation amendment (supersedes the historical blocking status above)

The user explicitly authorized this single isolation revision before any new
multi-edit forward. The opening block and draft QA above are retained as the
historical stop record, not the current isolation rule. The amendment is
`H1_multi_edit_isolation_v2.md`; machine-readable freeze is its adjacent JSON.

Initial independent HEAD: `abf29124a7acfa0c2b16575b7183468d0ef138df`, branch `paper5`.
All five initial draft files were preserved before incremental changes; initial
hashes are in the JSON. Original AutoResearch HEAD, worktree diff and index hashes
match the preceding stop record. Its unrelated modifications were not changed.
The six H1-small files match their archived hashes locally and on A800.
A800 HEAD at inspection: `02bdcd523ed2c2715a396fd624b647ef356a1c87`, tracked tree
clean, only the historical H1-small log untracked. No multi-edit output/cache,
runner, live process or screen session existed. No new multi-edit model forward
or measurement had occurred, and no new confirmation losses were used.
Synthetic unit-test labels are not model measurements.

### Historical use audit and evidence limits

Audit inspected tracked paper5 results/configs/code and Git history, local smoke
cache, and A800 paper5 results/runs/scratch. Three recorded validation uses exist:

| Experiment | Reported starting commit | Archived source/config/results | Validation sampling |
| --- | --- | --- | --- |
| 0.5B smoke | `a336a08cf08c371dab1764e1e33db8e0a38aebda` | `8cc8e6c`, `smoke/run_smoke.py`, `smoke/core.py`, `configs/canary/direction_01/smoke/rtx3050_qwen_0_5b.json`, `smoke_metrics.json` | 256 tokens, 8 windows, CPU randperm seed 20260918 (run seed+1) |
| 1.5B sanity | `a970d3a4acb7a55a9e8b7f4af78b93d850a887c7` | `5a2dc51`, `run_h1_sanity.py`, `h1_sanity.yaml`, sanity config/results | 128 tokens, 2 windows, seed 20260919 |
| H1-small | `5f9b6d76121e562d372862be372341c9c239f582` | results `02bdcd523ed2c2715a396fd624b647ef356a1c87`, `run_h1_small.py`, `h1_config.yaml` | 512 tokens, 64 V_old, source indices archived per window |

The earlier failed H1-small attempt at `f235f62` stopped on train download before
model loading/validation preparation. Its error log remains remote. The sanity
scratch backup has the same config and actual-run record as the archived sanity;
it does not introduce different windows. The earlier generic interaction canary
has no recorded run/results and its config targets a different split; no test
text was accessed in this task. No other recorded paper5 valid experiment was
found in the inspected locations. This is a records audit, not proof that no
unrecorded use ever happened.

Smoke/sanity reported starting commits precede creation of their runner files;
the sources are archived in the subsequent result commits, not in those starting
commits. Executed dirty-tree hashes and historical per-window manifests for these
two small runs were not recorded. Their indices are deterministic reconstructions
from archived code/config, not recovered contemporaneous manifests. This residual
provenance limit is disclosed rather than claiming historical bytes were captured.

For smoke the current replay used the existing original environment:
PyTorch **2.5.1+cu121**, transformers **4.56.1**. For sanity the A800 replay used its
original **2.4.1+cu121**, transformers **4.56.1**. Both reproduce the same sampling
indices. The previous cross-PyTorch replay uncertainty is thus resolved for this
sampling operation. No model was loaded during this replay.

### Source coordinates and overlap verification

Train SHA256: `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f`.
Valid SHA256: `f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8`.
Smoke tokenizer revision: `23c3f65dbb728a248a1dbe9ac90216e6ef3de5ac` (0.5B).
H1/V2 tokenizer revision: `8faed761d45a263340a0528343f099c05c9a4323` (1.5B).
Both are Qwen2TokenizerFast: entire UTF-8 decoded text, no added special tokens,
no truncation/padding, no chat template, contiguous non-overlapping chunks.
Tokenizer JSON/vocab/merges hashes are identical; config SHA differs. Full token
stream replay independently verifies the actual coordinates despite that difference:
270,674 tokens; `<i8` SHA256
`b7146fd14705e9786e34069f22825bd01d95b94922da397223283d45f246b073` for both revisions.
Every tokenizer artifact hash is retained in the isolation JSON. There are 528
complete 512-token chunks. The original V_old ±1 exclusion left 354; V2 leaves 325.

| Historical window | Actual source-token range (half-open) | Original V_new | New range | Overlap |
| --- | --- | --- | --- | ---: |
| smoke 324 (256) | [82944, 83200) | 162 (512) | [82944, 83456) | 256 |
| smoke 784 (256) | [200704, 200960) | 392 (512) | [200704, 201216) | 256 |

The blocker is confirmed from source hashes, full token streams and sampling
reconstruction, independently of screenshots or integer index coincidence.
Only portions of these confirmation texts had historical different-model smoke
measurements; no complete new 512-token multi-edit measurement had been made.

All historical ranges and per-window exclusions are explicit in the JSON.
Smoke indices: `[362,784,625,697,852,324,616,909]` (256 tokens).
Sanity indices: `[153,73]` (128 tokens), ranges `[19584,19712)` and `[9344,9472)`.
H1-small's 64 indices and ranges are retained unchanged in the JSON.

Removed from original V_new:
- 162 and 392: direct smoke overlap above;
- 163 and 393: ±1 neighbors of those directly exposed 512-token chunks;
- 313: neighbor of chunk 312, which intersects smoke 625 `[160000,160256)`;
- 37: neighbor of chunk 38, which intersects sanity 153 `[19584,19712)`.

Next six eligible windows in the unchanged hash ranking, in order:
`[436,73,364,527,476,218]`. Full original and V2 ordered lists and tensor hashes
are frozen in JSON. No candidate or model loss informed this replacement.
The candidate manifest remains
`7c40b1a1963d76e99144d7b27608fbfc6e68f1831e244f332e217a9c704021b9`.

Confirmed: source identity, actual token-coordinate equivalence, deterministic
replayed indices, intersections/exclusions, absence of new multi-edit measurements,
unchanged historical evidence and candidate plan. Uncertain: unrecorded historical
uses, missing contemporaneous smoke/sanity index manifests/dirty-tree hashes,
document dependence and base-model pretraining exposure. No independence or
pristine-benchmark claim follows. The confirmation set is only
**项目内隔离后的内部确认集**.

### V2 pre-commit CPU checks

- Repository `ruff check .`: PASS.
- All task Python files: Black PASS; shell syntax and diff whitespace: PASS.
- Whole-tree `black --check .` was attempted but its process pool stalled in this
  sandbox; only the identified new task processes were terminated. Equivalent
  in-process Black sweep of 63 tracked/task Python files found just the known
  unrelated `tools/researchclaw/gpu_queue/state.py` debt, unchanged from HEAD.
- CPU analysis/tools suite: **99 passed**; torch-dependent CPU suites in a separate
  process: **8 passed** (total 107). The combined run initially had 106 pass / 1
  failure: the pre-existing review test deliberately forbids torch imports while
  SciPy's array helper detects torch already imported during other test collection.
  The same tests pass in their intended separated processes; no test was deleted,
  weakened or skipped, and no unrelated implementation was modified.
- V2 tests explicitly cover real 256/512 overlap and half-open boundaries,
  historical ±1 protection including sanity, exact deterministic 64-window
  selection, incompatible token coordinate rejection, insufficient eligible count
  rejection, unchanged calibration/candidate manifest and unchanged original files.
- Current stage: amendment/code ready for commit; remote prepare and model run
  still pending. This checkpoint is not experiment completion. The exact code
  SHA will be written by prepare after synchronization and before model loading.

## Verified V2 execution outcome — 2026-09-20

The user explicitly authorized upload of the nine task files and their project
research/audit summaries to `Grain-Wang/wyc`, `refs/heads/paper5`. The earlier
approval rejection was retried through the same GitHub channel after that explicit
authorization and passed. At the code-publication checkpoint, only the nine listed task paths were
published; no raw run logs, authentication material or numerical-result paths
were uploaded at that stage. Subsequent result-archive authorization is recorded below.

Code/plan commit: `eba72bc9e9f59634e0f2f2639d05c1de1c38f308`
(`fix(paper5): isolate multi-edit confirmation windows under V2`), parent
`abf29124a7acfa0c2b16575b7183468d0ef138df`. GitHub branch update was non-forced.
Local and A800 independent branches advanced to that exact commit. No reset,
rebase, history overwrite, original-worktree change or H1-small rerun occurred.
The launcher receives the authorized Conda initialization location through
`PAPER5_CONDA_INIT`; the machine-specific path is not stored in Git.

Remote QA: Ruff and all task-file Black checks passed; 99 analysis/tools tests
plus 8 torch-dependent CPU tests passed in separate processes. CPU prepare
finished with exit 0 at `2026-09-20T13:28:51.523855+00:00`, before the new model
phase. Its manifest, predictions and used configuration match the locally saved
pre-forward snapshot byte-for-byte after completion.

A single detached-screen run dynamically selected A800 GPU 1, initially with
81,029 MiB free. Model phase: `2026-09-20T13:30:16.794293+00:00` through
`2026-09-20T13:45:27.374930+00:00`; **910.579366 seconds**. Launcher wall time
including setup/imports and analysis: **930 seconds** (13:30:04–13:45:34 UTC).
CPU analysis: **2.110655 seconds**. Both hard budgets passed. Peak allocated GPU
memory: **3,570.78125 MiB**; peak reserved: **4,416 MiB**. Torch 2.4.1+cu121,
transformers 4.56.1. No other process was terminated. The screen/model processes
exited normally; launcher exit code **0**, final state **COMPLETE**.

All 90 candidates completed C=96 and V2=64; V2 parent=64. There are exactly
**14,464 measurement rows + 8 safety windows** (14,472 total, below 14,480).
Every architecture/window identity was checked, with no duplicate, missing or
nonfinite measurement. C-based strategy choices were frozen before V2 forwards
and independently recomputed from the archived C rows. G/R point metrics,
quality counts, candidate manifest, and the original decision function replay
agree. The bootstrap remains 20,000 joint window resamples, seed 20260923;
thresholds, strata, method costs and the original protocol are unchanged.

### Original-protocol decision

**STOP_COMPLEX_METHOD_INVESTMENT**, no qualifying k, no H2. All six global/low
panels contain **0/15** candidates with V2 ΔNLL ≤ 0.30; every low panel fails the
practicality gate. Selection losses exist within this damaged candidate family,
but do not establish a practical NAS opportunity. This is not evidence that all
multi-edit architectures are additive. H1-small remains **INCONCLUSIVE**.

| Low k | Additive G | Adjusted 98.333333% interval | Additive R | Minimum candidate ΔNLL | Quality count |
| --- | ---: | --- | ---: | ---: | ---: |
| 4 | 0.109984772 | [0.087178027, 0.133714395] | 0.116490208 | 0.508735126 | 0/15 |
| 6 | 0.120566752 | [0.075688659, 0.164785570] | 0.120566752 | 1.049195126 | 0/15 |
| 8 | 0.268270638 | [0.212149351, 0.324058066] | 0.372885477 | 1.912761007 | 0/15 |

Descriptively, additive Top-5 C reranking resolves the k=4 low selection within
the ε=0.02 interval rule; quadratic Top-1 does so for k=6 low. These candidates
still fail the quality gate. For k=8 low, even the full-C reference has hindsight
regret 0.104615 (95% interval [0.062903, 0.147666]), indicating C-to-V2 selection
drift in addition to severe damage. No simple-strategy success or failure here
overrides the failed practical gate or authorizes H2.

### Archive and integrity

Complete numerical outputs (13 files) remain at the A800 project-relative path
`repo/paper5/results/canary/direction_01/H1_multi_edit/`. The unchanged original
H1-small six files were hash-verified again on both machines after completion.
This V2 set is only **项目内隔离后的内部确认集**.

Private raw logs, relative to the A800 paper5 project root:

- `scratch/h1-v2-preflight-eba72bc/preflight.log` and `exit_code`;
- `scratch/h1-multi-edit-eba72bc9e9f59634e0f2f2639d05c1de1c38f308/run.log` and `exit_code`.

The independent local copy contains the complete mirrored result/log archive at
`.local-deps/H1_multi_edit_V2_archive/`, plus `verification.json` with the
completeness checks and hashes. It is ignored by Git. Raw logs remain in that private archive. The subsequently
authorized numerical-result publication is recorded below.
No more model work is authorized or needed to complete this diagnostic.

| Artifact | SHA256 |
| --- | --- |
| `calibration_choices.json` | `1145ad6753c22397a41dac2bfbeb3ab11df0817794768e6ec5f604147c8d4d39` |
| `config_used.yaml` | `157ed9ed67dd4b8bb2656a12e8fbeeadf84ffd5d3e951216c2ec48683a510c8e` |
| `costs.csv` | `53519e5dbb3dd33d68673a3570838b3ccad32801de23c484ab9ab68cdb37a8f4` |
| `manifest.json` | `a4874fdc9b954bd45fcf4df4977227c44eca76cd32f2b3fed831db78eb0d0b39` |
| `metrics.json` | `e6911225cecd699bfe93a264e577131c5e476f13dad176132d7252a345d96b04` |
| `per_window_nll.csv` | `2c2ca84e7cfb8ebae9869a5b887d9791d5f8d833d7ae4a13d7b082756b213951` |
| `predictions.json` | `d469df438023330c1cd473ca45a60d31ce646f5741fe764479772f86034ded35` |
| `resource_preflight.json` | `3c4cf8967241b22c1509cd93ba089d471da0b4f4fa17520a7d99b5894d77d91f` |
| `results.csv` | `b694a860001cde60b13371c15e7059de507ab99a1396cf88af1562e9aeb03052` |
| `runtime.json` | `8ecf123a39a681fec8b472739aeb2535c5545081b72f2f13919db18b204f008f` |
| `state.json` | `e3ecbf22833fc289a10bf67008d9db424647b20a8d7de4d2ef06d4d6d6cf47e7` |
| `strategy_results.csv` | `592920cd9258bd653aba533a08f39a90c9c9e728456cc08601573347451d81fc` |
| `summary.md` | `c0d357663b7d31441181e21f2387df9393fdfb263d4a4066aac3721396684895` |

## Authorized numerical-result archive publication

After completion, the user explicitly authorized the 13 existing small result
artifacts plus this execution record and `.codex/handoff/CURRENT.md` for
`https://github.com/Grain-Wang/wyc.git`, branch `paper5`. This expands the earlier
pre-execution-file authorization and the rejected result-summary upload scope.
Repository visibility is unchanged. Only the same GitHub approval channel is used;
a renewed rejection requires stopping, not changing tools or bypassing review.

The 13 artifacts are copied byte-for-byte from the completed independent archive
to `paper5/results/canary/direction_01/H1_multi_edit/`. Their SHA256 values remain
exactly those in the table above. The original tar, extracted archive and H1-small
six files are unchanged. The archived configuration/state/runtime continue to name
**actual executed code `eba72bc9e9f59634e0f2f2639d05c1de1c38f308`**; the later
commit containing these result files is an **archive commit**, never an executed
model-code revision. No model, protocol, candidate, threshold, statistic or verdict
was changed or rerun for publication.

Read-only integrity checks confirm the complete 14,464 measurement rows, fixed
90 candidates, V2 64-window manifest, pre-forward freeze, recorded choices,
original-protocol decision and all 13 artifact hashes. JSON/YAML/CSV and necessary
runtime metadata were inspected. Raw logs, credentials, connection details,
original corpora, model weights and caches are excluded. The reviewable upload
inventory contains exact relative paths, byte sizes and SHA256 for the 15 files.
Publication success and GitHub branch/content verification are reported separately
in the delivery message; an approval, commit or push attempt alone is not proof
that the results are readable from GitHub. Stop after this archive task; no new
experiment, H2, CET, Fallback or direction change is authorized.
