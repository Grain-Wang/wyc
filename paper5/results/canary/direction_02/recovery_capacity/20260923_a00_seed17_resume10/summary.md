# Exploratory recovery-capacity diagnostic — completed

Decision: **EXPLORATORY_CAPACITY_FEASIBLE**. a00, original blocks 12/15 removed, seed 17.

Resumed the original step-10 adapter, AdamW moments/update counters, RNG and training order; performed only updates 11–100. The original Stage0 QUALITY_INFEASIBLE admission stop remains unchanged. No formal 8×2 trajectories or E evaluation were run.

Execution code: `c63ec1a028d8f8fdc901a208b67a82e8b15c51c5`.
Original pilot code: `75e3253b8f8b78c5998730c039ad346231d350bc`.
Original pilot archive: `66d1b974c4379cc0866fc10f29a739a228e819d8`.
The subsequent result archive commit is separate from execution; use the Git commit containing this directory.

Unmodified parent S NLL: 2.461201183498. Fixed quality rule: PPL ratio ≤ 1.15.

| Step | S NLL | NLL / parent | PPL / parent | Source |
| --- | ---: | ---: | ---: | --- |
| 0 | 2.624146957 | 1.066205792 | 1.176972865 | original pilot; not rerun |
| 10 | 2.608408062 | 1.059810990 | 1.158593626 | original pilot; not rerun |
| 20 | 2.564941987 | 1.042150477 | 1.109312888 | this continuation |
| 50 | 2.462038914 | 1.000340375 | 1.000838081 | this continuation |
| 100 | 2.363530232 | 0.960315738 | 0.906947284 | this continuation |

All three scheduled evaluations pass the unchanged threshold. This supports **recovery capacity for this single candidate and fixed recipe only**. It does not establish two usable candidates, a material selection gap, superiority to B20/BSH, or a need for FHRD/RSC. The diagnostic was authorized after observing the pilot stop and is exploratory internal evaluation. S improvement below the unmodified parent is not evidence about E or cross-corpus generalization.

Additional diagnostic GPU wall time: 228.989281 s; including the original pilot: 418.110120 / 14400 s. Sum of 90 measured training-update times: 65.905140 s. Three S64 evaluations: 6.729431 s. Checkpoint writes: 0.498742 s. Other loading, resource checks and I/O are included in wall time, not treated as free.

Physical GPU 2, UUID `GPU-5c9c1a6d-52ff-000a-11e9-be3f1b36d43d`; shared permission, existing processes untouched. Peak allocated/reserved: 3737.093/4182.000 MiB. Detached task exited 0 and its PID is gone.

Checks: local and remote CPU suites each 29 passed; Ruff, Black and shell syntax passed. Restored step-10 first-S-window loss difference = 0; frozen-base SHA unchanged; adapters updated. All 90 continuation rows and 192 S-window measurements are complete. Forty-six original result/protocol/code files match their pre-run hashes; 38 preexisting result mtimes are unchanged. The original step-10 checkpoint and GPU ledger are unchanged. D1 and the original Stage0 stop records are preserved.

Archive contents: 12 unmodified runner artifacts plus this summary, costs.json, audit.json and original_artifact_manifest.json. Config includes the unchanged recipe; data manifest contains identities/hashes only. No E token access, raw logs, raw corpus or adapter weights are published.

Private project artifacts: `cache/direction_02/recovery_capacity/20260923_a00_seed17_resume10/step_20.pt`, `step_50.pt`, `step_100.pt`; logs and exit code: `scratch/direction_02/recovery_capacity/20260923_a00_seed17_resume10/`. Paths are relative to the remote paper5 project root.

**STOP.** Recovery capacity is feasible for a00/seed17. Any formal protocol or 8×2 launch requires renewed user approval. No complex-method work or new direction follows automatically.
