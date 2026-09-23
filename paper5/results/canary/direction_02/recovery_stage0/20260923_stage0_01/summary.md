# D2-Stage0: stopped at the pilot quality gate

Status: **STOPPED_AT_PILOT_QUALITY_GATE**. Pilot completed normally (exit 0);
formal recovery and E evaluation did not start. This is not a completed 16-trajectory
Stage0 comparison. Quality label: **QUALITY_INFEASIBLE**, scoped to the frozen
pilot admission rule, not to unmeasured 100-step recovery.

Executed pilot code: `75e3253b8f8b78c5998730c039ad346231d350bc`. No formal execution SHA exists.
The result archive commit is a later Git commit. Design: D2_STAGE0_V1.

## Observations and fixed gate

| Quantity | Observed S NLL |
| --- | ---: |
| Unmodified parent | 2.461201183 |
| Fixed pilot candidate a00, step 0 | 2.624146957 |
| Same candidate, step 10 | 2.608408062 |

Candidate a00 removes original blocks 12 and 15; seed 17, fixed LoRA/CE recipe.
The S improvement is 0.015738895 nats/token. Base bytes were unchanged,
adapters changed, and save/reload NLL difference was exactly zero.
All eight candidates physically remove two blocks and have equal adapter capacity.

The prespecified working tolerance is parent-relative PPL <=1.15, equivalent to
excess NLL <=0.139761942. It is not a literature standard. Initial qualifying
candidates: **0/8**, versus the required >=2. The ten-step endpoint has excess
NLL 0.147206878 and relative PPL 1.158593626; it also misses
the unchanged threshold. No formal freeze, threshold relaxation, candidate
replacement, additional steps, seed search or E-driven decision was performed.

## Cost and resource evidence

Physical A800 GPU 2, explicit shared-use permission; timings reflect shared use.
Pilot GPU-process elapsed: 189.121 seconds, including startup,
loading, measurements and I/O. Peak allocated/reserved: 3736.218/
4202.000 MiB. Ten optimizer updates used 20,480 input tokens and
20,440 labels from 40 unique T windows. There are 704 recorded S evaluation rows,
plus eight structural T forwards and two save/reload S forwards.

The original conservative full-task projection was 10534.927
seconds (2.926 hours), within the four-hour cap.
This is an estimate, not measured formal cost or strategy speedup. Quality, not
engineering, memory or projected runtime, stopped formal admission.

## What remains unmeasured

G0, both seeds' 100-step endpoints, S ranking changes through 100 updates,
B0/B20/BSH/Ref100 endpoint comparisons, E simple regret, and document bootstrap
intervals are **not evaluated**. Formal metrics, strategy results, formal curves,
E losses and 100-step adapters do not exist and were not fabricated.

The observed short recovery signal does not establish that 100 steps would fail.
The stop does not reject all of Direction 2, justify a complex predictor, or meet
Paper Candidate criteria. Any changed quality gate or scientific setting requires
a separate decision; this run does not proceed automatically.

The original five preparation files (including historical preflight status) are
preserved. Newly copied raw numerical outputs are pilot.json, pilot_nll.csv,
pilot_training_curve.csv, pilot_resource_preflight.json and gpu_budget.json.
Derived pilot-only decision/cost/runtime records and the SHA256 manifest explain
this stop. No raw logs, credentials, source corpus, model or adapter weights are
published. D1 result bytes and its original judgments remain unchanged.
