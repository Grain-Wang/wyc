# H1 multi-edit execution record

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
