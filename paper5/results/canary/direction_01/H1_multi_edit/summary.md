# H1 multi-edit fixed-budget diagnostic

Isolation version: H1_multi_edit_isolation_v2. 确认集：项目内隔离后的内部确认集。

Decision: **STOP_COMPLEX_METHOD_INVESTMENT**. Original H1-small: **INCONCLUSIVE**, unchanged. H2 was not run.

All 90 candidates completed C=96 and V_new=64; k=2 reused without new forwards.

## Low panels (primary)

| k | Practical / quality count | Additive G (adjusted interval) | Additive R | Best simple result |
| --- | --- | --- | --- | --- |
| 4 | False / 0 | 0.109985 [0.08717802691137673, 0.13371439522173553] | 0.116490 | s1_top5: G=-0.006505, R=0.000000 (post-hoc descriptive, not a new strategy) |
| 6 | False / 0 | 0.120567 [0.07568865884708677, 0.16478556981881276] | 0.120567 | s2: G=0.000000, R=0.000000 (post-hoc descriptive, not a new strategy) |
| 8 | False / 0 | 0.268271 [0.2121493512865508, 0.3240580660944089] | 0.372885 | direct_5: G=0.014837, R=0.119452 (post-hoc descriptive, not a new strategy) |

## Scope and uncertainty

All global/low panels, candidate losses, numerical errors, ranks, Top-k, calibration choices, confirmation halves and calibration-shard sensitivities are retained in metrics.json and CSVs. No damaged candidate was dropped. Only practical low panels enter the continuation decision. G uses the fixed calibration reference, can be negative, and differs from numerical error or rank disagreement. R uses the finite 15-candidate V_new hindsight minimum, not the true search-space optimum or an independent test. Bootstrap is paired across every architecture and k; minimum reselection makes R intervals descriptive. Document dependence can weaken nominal coverage. Low pools are additive-defined and not representative of all NAS candidates.

s1/s2 scores were frozen before new calibration labels; policy choices before V_new. Source hashes, pinned tokenizer, token tensor hashes and excluded old windows are in manifest.json. H1 sanity short windows were checked for token overlap. V_new is internal confirmation, not a fresh corpus; the plan was authored after V_old and a historical different-model 0.5B smoke experiment on the same corpus.

## Cost and decision

costs.csv counts candidate construction (parent + 12 singles) for every cold-start policy. Quadratic scores cost 66 additional historical pair evaluations (6,336 windows), not free information. Additive Top-5 and direct-5 both cost 43 cold units; quadratic Top-1 and direct-11 cost 79; quadratic Top-5 costs 109 versus full direct's 103. With history sunk, both Top-5 rerankers and direct-5 cost 30 new units across six panels. Full-C audit labels are not free policy queries. The full direct reference leaves only 24 units above the 79-unit continuation gate. Hook intervention still computes original blocks, so no structural speedup is established.

Exact approved design: `abf29124a7acfa0c2b16575b7183468d0ef138df`; executed code: `eba72bc9e9f59634e0f2f2639d05c1de1c38f308`.
GPU 1 (NVIDIA A800 80GB PCIe), model phase 910.579 s, peak allocated 3570.781 MiB; CPU analysis 2.111 s.

The approved gate yields STOP_COMPLEX_METHOD_INVESTMENT; qualifying k: []. This supports no transport-method claim and authorizes no H2 execution. Stop after this single diagnostic.
