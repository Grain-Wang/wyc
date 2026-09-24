# Exploratory strongest-competitor coverage

Base archive: `123cd0ca2c7011a8bd1560aee47984954fe09a20`.
New execution code: `28a3db86d1fd85e2505040b4aed8a79f58803978`; later result archive commit is distinct.

Four new trajectories complete; sixteen original trajectories reused read-only.
E was already viewed. Intervals describe this post-hoc internal analysis, not confirmation.

## Original eight (unchanged)

{'complex_method_authorized': False, 'engineering': 'PASSED', 'problem': 'NO_MATERIAL_SELECTION_GAP', 'quality': 'USABLE', 'simple_baseline': 'SIMPLE_BASELINE_SUFFICIENT', 'sufficient_strategies': ['B20', 'BSH'], 'verdict': 'NO_MATERIAL_SELECTION_GAP'}

## Expanded ten

{'engineering': 'PASSED', 'quality': 'USABLE', 'problem': 'NO_MATERIAL_SELECTION_GAP', 'verdict': 'NO_MATERIAL_SELECTION_GAP', 'simple_baseline': 'SIMPLE_BASELINE_SUFFICIENT', 'sufficient_strategies': ['B20', 'BSH'], 'complex_method_authorized': False}

| Seed | B0 | Ref100 | G0 [95% interval] | Quality |
| --- | --- | --- | --- | --- |
| 17 | a00 | a00 | 0.000000000 [0.000000000, 0.000000000] | 10/10 |
| 29 | a00 | a00 | 0.000000000 [0.000000000, 0.000000000] | 10/10 |

See metrics.json for ranking, recovery, simple regret and B20/BSH results; costs.csv separates policy information budgets and standalone estimates.
The parent was not recovery-trained; no compression-benefit attribution is supported.
Stop after this single supplement; no automatic candidates, complex predictors or new direction.
