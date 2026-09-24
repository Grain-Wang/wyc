# Strongest-competitor coverage: completed exploratory supplement

Base archive: `123cd0ca2c7011a8bd1560aee47984954fe09a20`.
Original V2 execution: `53182b532ad8980eb0c4f09f122be941e2671b79`.
Supplement execution: `28a3db86d1fd85e2505040b4aed8a79f58803978`. The commit containing this report is the later results archive, not execution code.

Exactly four fresh trajectories completed: a08=(10,15), a09=(10,12), each seeds17/29 ×100 updates. All sixteen original trajectories were reused read-only. No extra pilot, parent recovery, candidates or seeds were run.
This supplement was authorized after V2 S/E results had been viewed. E was reused; all intervals are exploratory descriptions, not new confirmatory evidence. Original eight-candidate results remain unchanged.

## Original eight (separate archived result)

{'complex_method_authorized': False, 'engineering': 'PASSED', 'problem': 'NO_MATERIAL_SELECTION_GAP', 'quality': 'USABLE', 'simple_baseline': 'SIMPLE_BASELINE_SUFFICIENT', 'sufficient_strategies': ['B20', 'BSH'], 'verdict': 'NO_MATERIAL_SELECTION_GAP'}

Both seeds selected a00 under all four policies, G0=0 [0,0], and 8/8 candidates passed quality. These results were not rerun or reclassified.

## Expanded ten

{'complex_method_authorized': False, 'engineering': 'PASSED', 'problem': 'NO_MATERIAL_SELECTION_GAP', 'quality': 'USABLE', 'simple_baseline': 'SIMPLE_BASELINE_SUFFICIENT', 'sufficient_strategies': ['B20', 'BSH'], 'verdict': 'NO_MATERIAL_SELECTION_GAP'}

| Seed | Policy | Selected | E NLL at100 | Difference vs Ref100, 95% interval | Hindsight simple regret, 95% interval |
| --- | --- | --- | ---: | --- | --- |
| 17 | B0 | a00 | 2.471813731 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |
| 17 | B20 | a00 | 2.471813731 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |
| 17 | BSH | a00 | 2.471813731 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |
| 17 | Ref100 | a00 | 2.471813731 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |
| 29 | B0 | a00 | 2.477565538 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |
| 29 | B20 | a00 | 2.477565538 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |
| 29 | BSH | a00 | 2.477565538 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |
| 29 | Ref100 | a00 | 2.477565538 | 0.000000000 [0.000000000, 0.000000000] | 0.000000000 [0.000000000, 0.000000000] |

PPL-ratio quality threshold=1.15; material selection margin=0.02 nats/token. Parent E NLL=2.564784789. At least two candidates and the S-selected Ref100 must satisfy quality in each seed.

| Seed | Passing candidates | Ref100 quality | S rank at0 | S rank at100 | Spearman | Kendall | Top1 retained | Top3 overlap |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |
| 17 | 10/10 | True | a00, a08, a09, a03, a02, a04, a05, a07, a06, a01 | a00, a09, a03, a02, a08, a05, a07, a06, a04, a01 | 0.854545 | 0.733333 | True | 0.666667 |
| 29 | 10/10 | True | a00, a08, a09, a03, a02, a04, a05, a07, a06, a01 | a00, a09, a08, a03, a02, a05, a07, a06, a04, a01 | 0.915152 | 0.822222 | True | 1.000000 |

## Added-candidate recovery curves

| Seed | Candidate | S NLL@0 | S NLL@20 | S NLL@50 | S NLL@100 | E NLL@100 | E PPL ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | a08 | 2.637774780 | 2.584698686 | 2.489147615 | 2.396183252 | 2.495571941 | 0.933128044 |
| 17 | a09 | 2.640146773 | 2.581220027 | 2.477147613 | 2.371706130 | 2.480541199 | 0.919207319 |
| 29 | a08 | 2.637774780 | 2.579990979 | 2.489111405 | 2.381739749 | 2.489895321 | 0.927846037 |
| 29 | a09 | 2.640146773 | 2.594858410 | 2.481420355 | 2.380947454 | 2.490184331 | 0.928114233 |

All-candidate intermediate NLL/PPL, recovery amounts and final quality are in quality_by_checkpoint.json and metrics.json. Initial/intermediate quality did not filter candidates. The parent was not adapted; performance relative to that parent does not establish a benefit caused by compression.

## Information and statistical audit

Four new trajectories, 400 updates, 16 checkpoints, 1024 new S rows and 256 new E rows completed. The expanded data contain 6464 measurements including unchanged historical rows. Expanded S scores and selections were persisted before opening new E. Every policy uses its chosen 100-step endpoint.
Document-cluster paired bootstrap uses 2000 draws, shared resamples across candidates, seed20260924, and separate reporting for training seeds17/29. Finite-pool E-hindsight regret has post-selection bias; original and expanded sets are not independent experiments.
CPU tests: 44 local and 44 remote passed; Ruff, per-file Black, shell syntax and diff checks passed. Fresh optimizer and zero LoRA-B at all four starts, optimizer counters at20/50/100, base preservation and all16 checkpoint hashes were audited.

## Costs

Measured additional GPU time: 676.885004 seconds. Prior ledger: 2375.137251; cumulative: 3052.022255. Limits: additional3600 / cumulative14400 seconds. Peak allocated/reserved: 3737.093/4210.000 MiB. Physical GPU2, shared, UUID `GPU-5c9c1a6d-52ff-000a-11e9-be3f1b36d43d`.
CPU analysis including result I/O: 11.035850 seconds; separate completion audit: 2.347865 seconds. These do not include Python/import and transfer overhead.
New training input tokens=819200, effective labels=817600. Unique T content remains 400 windows/204800 input tokens, reused across candidates and seeds. Original1600 training updates remain part of the historical audit cost.

| Seed | Policy | Update information budget | S query windows | E windows | Standalone seconds estimate |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | B0 | 100 | 640 | 64 | 1298.211884 |
| 17 | B20 | 280 | 640 | 64 | 1940.251994 |
| 17 | BSH | 450 | 1088 | 64 | 2607.263989 |
| 17 | Ref100 | 1000 | 640 | 64 | 4538.992635 |
| 29 | B0 | 100 | 640 | 64 | 1298.211884 |
| 29 | B20 | 280 | 640 | 64 | 1940.251994 |
| 29 | BSH | 450 | 1088 | 64 | 2607.263989 |
| 29 | Ref100 | 1000 | 640 | 64 | 4538.992635 |

Standalone policy costs reuse the original observed unit-cost basis, include common historical preselection (753.011602 estimated seconds), construction and policy-boundary checkpoint I/O. They are estimates, not measured speedups. Full trajectories already computed do not make replay information free. Measured component timings appear in audit_costs.json.

## Interpretation and stop

The two fixed strongest omitted competitors did not establish a material B0 selection loss under the original threshold. Recommend ending this candidate family and recovery recipe probe, and pausing complex D2 predictors. No further candidates or method development is authorized.
Original V2 verdicts are preserved. No FHRD/RSC, Hessian/NTK, other research direction or extra experiment follows.

## Archive and retained private files

Preserved 84 pre-existing result files and 65 V2 checkpoints/ledger files by hash and mtime. Frozen source hashes also cover D1, V1/capacity, and old protocol/config files. The original local AutoResearch workspace was untouched.
Private project-relative logs: `scratch/direction_02/recovery_competitor_coverage/20260924_competitor_coverage_01/` (run.log, launcher_pid, exit_code). Adapter checkpoints and heartbeat: `cache/direction_02/recovery_competitor_coverage/20260924_competitor_coverage_01/`. These are retained on A800 and not uploaded.
Complete small-artifact file list (no missing required numerical outputs):

- `REPORT.md`
- `artifact_manifest.json`
- `audit.json`
- `audit_costs.json`
- `budget_projection.json`
- `candidate_manifest.json`
- `config_used.json`
- `costs.csv`
- `data_manifest.json`
- `e_losses.json`
- `e_unsealed.json`
- `expanded_per_window_nll.csv`
- `freeze.json`
- `gpu_budget.json`
- `metrics.json`
- `new_trajectories.json`
- `original_vs_expanded.json`
- `per_document_nll.csv`
- `per_window_nll.csv`
- `pilot.json`
- `prior_gpu_budget.json`
- `quality_by_checkpoint.json`
- `resource_preflight.json`
- `runtime.json`
- `s_scores.json`
- `selection.json`
- `state.json`
- `strategy_results.csv`
- `summary.md`
- `training_curve.csv`
- `trajectories.json`
