# D2-Stage0 execution record

Current status: **pilot completed; formal admission stopped by the fixed quality
gate**. The single ten-update trajectory finished with exit code 0 on explicitly
authorized shared physical GPU 2. Formal recovery and E evaluation never started.
This is a completed pilot, not a completed 16-trajectory Stage0 comparison.

## Completed pilot and decision

Actual pilot execution code: `75e3253b8f8b78c5998730c039ad346231d350bc`.
That exact commit was ordinarily pushed to GitHub after authentication was
restored, then synchronized to A800 using bundle and an ordinary fast-forward.
Thirteen existing D1 archive files matched the incoming Git objects byte-for-byte;
all 28 existing D1 files including the private log retained their bytes and mtimes.
No original-worktree edit, force push, reset, rebase or duplicate commit occurred.

Remote CPU tests: 25 passed in 16.17 seconds. Source/tokenizer/model and all T/S/E
integrity checks passed without E evaluation. The inherited Ruff Python module
could not find a venv-local binary; the same pinned Ruff 0.12.11 native executable
completed static checks read-only. The failed invocation and successful checks
are both retained privately; D1 and D2 package installations were not changed.

The user explicitly authorized sharing GPU 2 under existing laboratory rules.
Three short prelaunch samples showed no D2 job and sufficient memory; a private
permission record restricted admission to index 2 and recorded its UUID. The
launcher and Python admission guards were retained. PCI order and the actual
model PID's GPU UUID were checked; no other GPU or process was modified.
The detached session retained launcher PID, heartbeat, private log and exit code.

Engineering passed: finite optimization, actual LoRA change, frozen base bytes,
physical block mapping and exact save/reload NLL agreement. Candidate a00 (remove
original blocks 12/15), seed 17, completed exactly ten updates. Parent S NLL was
2.461201183; a00 changed from 2.624146957 to 2.608408062 (improvement 0.015738895).
The prespecified PPL ratio limit remains 1.15, excess NLL 0.139761942. Initial
qualifiers were 0/8, below the required two. The pilot endpoint's PPL ratio was
1.158593626, also outside the gate. No tolerance was relaxed after observing data.

Quality label: **QUALITY_INFEASIBLE**, scoped to the frozen pilot admission rule.
Problem signal and simple-baseline sufficiency: **NOT_EVALUATED**. Ten updates
show a short recovery signal; these observations do not establish that 100 updates
would fail or invalidate Direction 2 generally. No formal freeze, full trajectories,
E model losses, G0, bootstrap intervals or formal strategy results were fabricated.

Measured GPU-process wall time: 189.121 seconds. Peak allocated/reserved memory:
3736.218/4202 MiB. The conservative full-task projection was 10534.927 seconds
(2.926 hours), within 14400 seconds; shared timings are not a measured speedup.
Quality, rather than projected cost or engineering, stops this run.

All ten original preparation/pilot files were copied byte-for-byte for local
archival; five were already committed preparation evidence. The 704 S rows and
ten training updates were independently checked against the frozen manifests and
pilot aggregates. Derived pilot-only summary, decision, costs and runtime records
are under `paper5/results/canary/direction_02/recovery_stage0/20260923_stage0_01/`.
Raw logs and adapters remain private. Actual execution SHA and later archive
commit must be reported separately; the latter is the result files' Git commit.

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

Those CPU checks used tiny random fixtures and prepared manifests, and did not
establish parent-model recovery or scientific outcomes. The subsequent pilot
evidence above establishes the engineering chain and S admission result only;
100-step quality, selection loss and baseline sufficiency remain unmeasured.

## Earlier GPU admission hold (resolved before this pilot)

The earlier CPU-preparation query found all three A800s busy: GPU 0/1/2 utilization was 100/81/95%,
with 35178/13137/11720 MiB free. These are observations, not allocations. No
existing D2 GPU process was found. No task-specific co-tenancy permission was
established; historical permissions for unrelated projects are not reused.
The user's CPU-preparation instruction required a pause in that situation;
explicit shared permission for GPU 2 was subsequently provided before the pilot.

`GPU_ACCESS.md` records the operational gate. Both the detached launcher and
direct Python GPU entry require separately verified task/stage/device permission,
then recheck live occupancy and duplicate tasks. Exclusive use also requires
clean repeated samples; sharing requires explicit permission. At that stage no real permission
file or delayed automatic job had been created. The later explicit GPU 2 shared
authorization was recorded before the pilot, as described above. The original protocol/config and
data/candidate manifests remain byte-identical; no scientific settings changed.

## Stop boundary

The original post-pilot quality gate failed. Do not invoke `freeze`, start the
formal launcher, add recovery steps, choose another seed/candidate, weaken the
threshold or open E. Keep the original protocol and idea unchanged. Finish only
the authorized small-result archival and GitHub read-back, then stop. A changed
scientific setting requires a separate user decision. No complex method or new
direction is authorized by this pilot outcome.

## Provenance fields

- D1 result baseline: `6ee61e753d8e67b21c56adfb2edeec082e076f10`.
- Initial D2 implementation commit: `66926eeac7df801783a228ea2907738904646359`.
  Its first ordinary push failed authentication. Later delivery and admission
  changes must be distinguished from an actual GPU execution commit.
- D2 pilot executed code SHA: `75e3253b8f8b78c5998730c039ad346231d350bc`.
- D2 formal executed code SHA: **not yet applicable — not started**.
- D2 pilot result archive SHA: the later Git commit adding the verified pilot
  files and this completed execution record; it is not the executed model SHA.

Raw logs/connection diagnostics, tokens, parent weights, LoRA/optimizer
checkpoints and caches remain excluded from Git. No credentials, connection
addresses or original sessions belong in this record.
