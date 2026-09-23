# D2-Stage0 execution record

Current status: implementation and local CPU/data preflight; A800 connectivity
blocked. No D2 GPU pilot, optimizer update on the parent model, formal recovery
trajectory or E measurement has occurred. This is not an experiment result.

## Authorized scope and preserved baseline

D1 baseline is `6ee61e753d8e67b21c56adfb2edeec082e076f10`. Local independent
`paper5` began clean at that commit and origin/paper5 was rechecked at it.
The original AutoResearch HEAD remains `5f9b6d76121e562d372862be372341c9c239f582`;
its existing diff and index hashes were verified unchanged. No original-worktree
edit, history rewrite or D1 rerun is authorized. D1's current block-skip route
is finished, CET paused, and D2-Stage0 is explicitly selected by the user.

The idea/ranking and D1 final summary were read. Primary-source verification of
DarwinLM, TraceNAS, Minitron and LLM-Pruner was completed within the 30-minute
cap; links and conservative interpretation are in STAGE0_PROTOCOL.md. No broad
new literature search or novelty claim was needed.

## Environment and data

The local CPU test environment is isolated under the ignored D2 dependency
directory, Python 3.12 with torch 2.4.1+cpu / transformers 4.56.1. The original
D1 and shared environments were not upgraded. The A800 `autoresearch_paper5`
environment has **not yet been inspected**, because the existing local SSH alias
timed out before authentication on three bounded connection attempts. No new
address, tunnel or remote connection channel was substituted.

No ready C4/FineWeb small subset was found in the accessible project cache.
The cached WikiText-2 train text was copied read-only into the D2 source cache;
its SHA matches D1. Six small tokenizer/config files were downloaded locally
from the exact Qwen2.5-1.5B revision and match D1's recorded hashes. No parent
weights or full corpus collection were downloaded locally. Training must remain
on the authorized A800.

Preparation writes document/window manifests and sealed token files; only the
small manifests are publishable. Each trajectory gets 400 unique T windows.
S/E are 64 windows each, from mutually exclusive complete document sets. D1
calibration reuse is disclosed; no historical non-exposure or blind-test claim.

Local preparation completed: 624 unique documents; available T/S/E windows are
3269/590/808. The selected 400/64/64 windows come from 400/64/64 distinct documents,
respectively, with no cross-split document IDs. Each window has 512 input tokens
and 511 effective labels. The E file was hashed for transport integrity but no
E model loss was computed or inspected. Selection/token manifests remain fixed.

The frozen candidate pairs (original zero-based layer indices) are:
`[12,15]`, `[3,15]`, `[8,15]`, `[12,17]`, `[6,10]`, `[6,12]`, `[8,17]`, `[8,10]`.
They were selected exclusively from historical calibration rows using the
prespecified top-16 / best-plus-hash rule, before any recovery measurement.

Local CPU verification: **22 tests passed** with CUDA disabled, covering physical
layer execution/mapping, equal adapter capacity, frozen base/actual adapter
updates, valid zero first-step A gradients, exact save/reload, document isolation,
deterministic candidate/data order, insufficient data, permitted policy queries,
paired document bootstrap, signed G0, quality/simple-baseline gates, sealed E,
duplicate/budget stops and resource admission. Ruff, per-file Black and shell
syntax checks passed. Multi-file Black's sandbox worker pool hung; only this
task's formatter processes were stopped, then per-file checks passed. This was
a tooling issue, not a model-run failure. No broad unrelated tests or edits.

These are CPU engineering checks on tiny random test fixtures and prepared
manifests; they do **not** establish parent-model recovery, candidate quality,
throughput, selection loss, baseline sufficiency, or A800 feasibility. Those
remain unmeasured until connectivity and the GPU pilot gates are satisfied.

## Continuing after connectivity is restored

1. Read live Git state and the current record; do not regenerate candidates or
   splits. Inspect `autoresearch_paper5`, its installed versions and CUDA support.
   If dependencies are missing/incompatible, create a D2-specific project
   environment using the recorded direct versions, without upgrading D1/base.
2. Verify remote `paper5` history without reset/rebase/forced merge. Synchronize
   the exact ordinary local commit using the authorized Git bundle/SSH path.
   Check D1 output bytes before and after sync, including any previously
   untracked copies that became tracked by the D1 archive commit; never overwrite
   a differing file. Transfer only prepared D2 token files and validate hashes.
3. Run the D2 CPU tests on A800 with CUDA disabled. Inspect module names and
   pinned cache file hashes. Use one detached screen session to call
   `paper5/scripts/direction_02/launch_stage0.sh pilot` from the remote repo.
   The project-local interpreter, model snapshot, D2 cache and exact SHA are
   supplied through D2_PYTHON, D2_MODEL_SNAPSHOT, D2_CACHE and D2_EXPECTED_SHA;
   machine-specific values and raw logs stay private.
4. Review pilot.json and resource metadata. Check real step/forward/I/O times,
   base immutability, adapter change, save/reload identity and initial/10-step
   quality. If engineering, quality or the full 16-trajectory projection fails,
   retain results and stop. Ten steps without loss improvement is not a general
   falsification of recovery.
5. Only after all pilot gates pass, run the CPU `--stage freeze`, commit the
   formal_freeze.json and related pilot evidence, synchronize that exact SHA,
   then launch once with `formal`. Formal execution rejects an uncommitted
   freeze, changed recipe, duplicate phase, missing trajectories or premature E
   access. The E gate is enforced before loading its separate token file.
6. Inspect completeness, budgets and checkpoints; archive reviewed small files
   to the authorized GitHub `paper5` branch and read back its ref, file list and
   hashes. Keep the actual execution SHA distinct from the later archive commit.
   No force push or approval-channel substitution. Stop after this Stage0.

## Provenance fields

- D1 result baseline: `6ee61e753d8e67b21c56adfb2edeec082e076f10`.
- D2 implementation commit: recorded by the delivery Git commit.
- D2 pilot executed code SHA: **not yet applicable — not started**.
- D2 formal executed code SHA: **not yet applicable — not started**.
- D2 numerical result archive SHA: **not yet applicable — no GPU results**.

Raw logs/connection diagnostics, tokens, parent weights, LoRA/optimizer
checkpoints and caches remain excluded from Git. No credentials, connection
addresses or original sessions belong in this record.
