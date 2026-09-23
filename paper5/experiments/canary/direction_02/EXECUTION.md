# D2-Stage0 execution record

Current status: implementation and CPU/data preparation; A800 SSH is restored,
but GPU admission is blocked by occupancy and missing explicit sharing permission.
No D2 GPU pilot, optimizer update on the parent model, formal recovery
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
D1 and shared environments were not upgraded. Earlier connection attempts timed
out; following the user's connectivity update, the original SSH alias succeeded.
The A800 `autoresearch_paper5` environment was inspected with CUDA disabled:
torch 2.4.1+cu121, transformers 4.56.1 and numpy 2.2.6 match the recipe, while
SciPy 1.18.1 and tokenizers 0.22.2 differ from the D2 pins. A project D2 environment
was prepared with read-only system-site inheritance and private overrides
to SciPy 1.15.3 and tokenizers 0.22.0, using locally downloaded wheels offline.
All nine direct dependencies now match D2 pins; CUDA remained uninitialized.
The original SciPy/tokenizers versions were rechecked unchanged. Twelve
transferred files (two wheels, six tokenizer/config files, source text and three
token arrays) passed exact byte-count/SHA256 checks, including E for transport
integrity only. No new connection address, tunnel or alternate upload channel
was substituted.

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

Local CPU verification: **25 tests passed** with CUDA disabled, covering physical
layer execution/mapping, equal adapter capacity, frozen base/actual adapter
updates, valid zero first-step A gradients, exact save/reload, document isolation,
deterministic candidate/data order, insufficient data, permitted policy queries,
paired document bootstrap, signed G0, quality/simple-baseline gates, sealed E,
duplicate/budget stops and resource admission. Additional admission tests cover
missing/stage-limited/expired permission, busy GPUs despite free memory, and
resource changes or a duplicate D2 process during recheck. Ruff, per-file Black and shell
syntax checks passed. Multi-file Black's sandbox worker pool hung; only this
task's formatter processes were stopped, then per-file checks passed. This was
a tooling issue, not a model-run failure. No broad unrelated tests or edits.

These are CPU engineering checks on tiny random test fixtures and prepared
manifests; they do **not** establish parent-model recovery, candidate quality,
throughput, selection loss, baseline sufficiency, or A800 feasibility. Those
remain unmeasured until resource permission and the GPU pilot gates are satisfied.

## GPU admission hold

The live query found all three A800s busy: GPU 0/1/2 utilization was 100/81/95%,
with 35178/13137/11720 MiB free. These are observations, not allocations. No
existing D2 GPU process was found. No task-specific co-tenancy permission was
established; historical permissions for unrelated projects are not reused.
The user's latest instruction explicitly requires a pause in this situation.

`GPU_ACCESS.md` records the operational gate. Both the detached launcher and
direct Python GPU entry require separately verified task/stage/device permission,
then recheck live occupancy and duplicate tasks. Exclusive use also requires
clean repeated samples; sharing requires explicit permission. No real permission
file or delayed automatic job has been created. The original protocol/config and
data/candidate manifests remain byte-identical; no scientific settings changed.

## Continuing after resource permission is established

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
   pinned cache file hashes. Verify server permission under GPU_ACCESS.md and
   re-query occupancy and existing task processes. Only if admitted, use one detached screen session to call
   `paper5/scripts/direction_02/launch_stage0.sh pilot` from the remote repo.
   The project-local interpreter, model snapshot, D2 cache and exact SHA are
   supplied through D2_PYTHON, D2_MODEL_SNAPSHOT, D2_CACHE, D2_EXPECTED_SHA and
   D2_GPU_PERMISSION_FILE;
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
- Initial D2 implementation commit: `66926eeac7df801783a228ea2907738904646359`.
  Its first ordinary push failed authentication. Later delivery and admission
  changes must be distinguished from an actual GPU execution commit.
- D2 pilot executed code SHA: **not yet applicable — not started**.
- D2 formal executed code SHA: **not yet applicable — not started**.
- D2 numerical result archive SHA: **not yet applicable — no GPU results**.

Raw logs/connection diagnostics, tokens, parent weights, LoRA/optimizer
checkpoints and caches remain excluded from Git. No credentials, connection
addresses or original sessions belong in this record.
