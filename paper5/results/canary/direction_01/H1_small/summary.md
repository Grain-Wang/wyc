# Direction 1 H1-small empirical observation

This is a 12-edit, 66-pair hypothesis-validation run using whole-block structural interventions. It does not evaluate a transport predictor or establish a method claim.

## Run

- GPU: `0` (`NVIDIA A800 80GB PCIe`)
- Runtime: 1179.54 seconds
- Model: `Qwen/Qwen2.5-1.5B` at `8faed761d45a263340a0528343f099c05c9a4323`
- Dataset: raw WikiText-2 train calibration and valid evaluation
- Scope: 12 single interventions and all 66 unordered pairs

## Validation interaction

- Mean absolute interaction: 0.0691665 NLL
- Median relative absolute interaction: 0.068372
- Signed residual range: [-0.384349, 0.363445] NLL
- Signed residual Q1 / median / Q3: 0.00653748 / 0.0167544 / 0.054078 NLL
- Positive / negative / zero pairs: 54 / 12 / 0

## Validation prediction baselines

| Baseline | MAE (NLL) | RMSE (NLL) | Spearman | Kendall tau-b | Lowest-17 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| additive | 0.0691665 | 0.116035 | 0.959127 | 0.849883 | 0.882353 |
| mean_interaction | 0.0707987 | 0.106362 | 0.959127 | 0.849883 | 0.882353 |
| linear_regression | 0.0632844 | 0.103708 | 0.959962 | 0.852681 | 0.823529 |

## Protocol interpretation

- H1 verdict: `INCONCLUSIVE`
- Point verdict: `INCONCLUSIVE`
- Stable confirmation under the prespecified bootstrap rule: `False`
- Paired-window 95% interval for median relative residual: [0.05872702191368138, 0.07884889817474999]
- Paired-window 95% interval for additive Spearman: [0.9509435340778625, 0.9633023692725186]
- Paired-window 95% interval for lowest-17 overlap: [0.8235294117647058, 0.9411764705882353]
- All uncertainty intervals and calibration-shard diagnostics are in `metrics.json`.
- This observation covers one model, one corpus, and one intervention family. It cannot establish interaction transport or a final algorithm.
