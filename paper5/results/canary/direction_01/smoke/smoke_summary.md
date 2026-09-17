# Direction 1 Local Smoke Test

> **THIS IS NOT A FORMAL SCIENTIFIC CANARY RESULT.**
>
> All measurements below are **SMOKE TEST / PRELIMINARY** and must not be used to claim H1 or H2.

## Pipeline status: PIPELINE READY

- Model: `Qwen/Qwen2.5-0.5B` at `23c3f65dbb728a248a1dbe9ac90216e6ef3de5ac`
- Device: `NVIDIA GeForce RTX 3050 Laptop GPU`
- OOM: `False`
- Edit family: `whole_transformer_block_skip`
- Unique single edits: 6
- Unique pairs: 15
- Calibration: 8 × 256 tokens
- Validation: 8 × 256 tokens
- Peak allocated GPU memory: 1278.92 MiB
- Runtime: 58.47 seconds

## Pipeline checks

- `cuda_available`: `True`
- `no_edit_context_matches_parent`: `True`
- `no_edit_max_nll_difference`: `0.0`
- `repeated_no_edit_max_logit_abs_difference`: `0.0`
- `single_edit_restored_parent`: `True`
- `pair_order_independent`: `True`
- `same_seed_single_reproducible`: `True`
- `candidate_memory_released`: `True`
- `baseline_allocated_memory_mib`: `982.16650390625`
- `final_allocated_memory_mib`: `982.16650390625`
- `all_metrics_finite`: `True`
- `h2_real_model_runtime`: `DEFERRED_TO_A800`

## Descriptive pipeline outputs

These values only demonstrate that additive-analysis code ran end to end. The sample is too small and the hardware/protocol are not the formal A800 design.

### calibration

- `pair_count`: `15`
- `pearson_additive_vs_actual`: `0.9921678165162457`
- `spearman_additive_vs_actual`: `0.9892857142857143`
- `kendall_tau_b_additive_vs_actual`: `0.9428571428571428`
- `additive_mae`: `0.1077496218213847`
- `mean_abs_interaction`: `0.1077496218213847`
- `fraction_relative_abs_interaction_gt_0_10`: `0.4`
- `amplification_count`: `12`
- `cancellation_count`: `3`
- `zero_count`: `0`

### validation

- `pair_count`: `15`
- `pearson_additive_vs_actual`: `0.9900144206034224`
- `spearman_additive_vs_actual`: `0.9785714285714285`
- `kendall_tau_b_additive_vs_actual`: `0.9047619047619048`
- `additive_mae`: `0.1256395447176267`
- `mean_abs_interaction`: `0.1256395447176267`
- `fraction_relative_abs_interaction_gt_0_10`: `0.4`
- `amplification_count`: `12`
- `cancellation_count`: `3`
- `zero_count`: `0`

## H2 status

H2 runtime validation deferred to A800. Only a toy-tensor JVP shape/value unit test was run locally.

## Scientific verdict

No H1 or H2 verdict is permitted from this smoke test.
