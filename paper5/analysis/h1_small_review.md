# H1-small 决策级结果复核

证据固定于 `02bdcd523ed2c2715a396fd624b647ef356a1c87`。本报告为已有结果的 CPU 复核与事后诊断，
不是新的模型实验。历史 H1 verdict 保留为 **INCONCLUSIVE**。

核心发现：12 singles / 66 pairs 的原始统计与配对窗口 bootstrap 均可复现；
A、B 两种情形下，三个 baseline 都选中本次候选集合实测最佳的 **(12,15)**，
Top-1 simple regret 为 **0**，前三名顺序也一致。
这支持“存在非零交互与部分错排”，尚不支持“加法已损害重要架构选择”。

## 1. 证据与完整性

读取 [方向说明](../ideas/direction_01_causal_error_transport.md)、
[冻结协议](../experiments/canary/direction_01/H1_protocol.md)、
[配置](../configs/canary/direction_01/h1_config.yaml)、
[runner](../experiments/canary/direction_01/run_h1_small.py) 和
[六个归档结果](../results/canary/direction_01/H1_small/)。
复核脚本对结果、配置使用固定 SHA 的 `git show SHA:path`；复用的两个 CPU
分析模块在导入前逐字节核对归档版本。未使用原工作区的未提交文件。

- 归档提交：`02bdcd523ed2c2715a396fd624b647ef356a1c87`。
- 实验起始代码提交：`5f9b6d76121e562d372862be372341c9c239f582`，
  正好是归档提交的父提交；两者用途不同，不是 provenance 冲突。
- 模型：`Qwen/Qwen2.5-1.5B`；requested/resolved revision 均为
  `8faed761d45a263340a0528343f099c05c9a4323`；bf16。
- 随机种子：`20260918`；validation 选窗用 seed+1；bootstrap 用原 seed。
- Train SHA256：
  `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f`。
- Valid SHA256：
  `f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8`。
- config_used 与冻结 YAML（扣除 run_metadata）完全一致；
  run_metadata 与 metrics metadata 一致；runner/core 代码 SHA 一致。
  这里确认的是归档内 provenance 一致性；本轮没有重新读取原始文本、权重
  或重新 tokenize，不能把它称作对原始模型执行的独立重放。

六个原文件保持逐字节不变：

| 文件 | SHA256 |
| --- | --- |
| config_used.yaml | `e164d26699c8f8e8180acec8440d6d74465f6cf7c41cff268807d7a2cdd1cdd6` |
| interaction_heatmap.svg | `8b77afaf93ed9843fa3f821bff6aa884dd8a50c1be96084d87bf5508a6545699` |
| metrics.json | `9744a0827579042b3c36026b06fc162dee3b793f49846d51b8d5b4983edd39a2` |
| per_window_nll.csv | `c0b2f17b4320af7664d9e58ba974f8dbeb3662fbf10a9b74826e6bb9185e5c86` |
| results.csv | `e0cafb3d4859005e7273aaa1abbecb098fa591c8f6a7c0e992353357411e7e17` |
| summary.md | `e42697f804b1424ece6195a149a2291bca8a5cef9cfe63bfa3f8a789efda0284` |

### 窗口与架构合同

逐行解析 12,640 条 NLL：calibration 为 79×96=7,584 条，validation 为
79×64=5,056 条。每个 split 恰有 parent、12 singles、66 个无序 pairs；
每个 architecture 的 sample_index 恰覆盖同一 0..95 或 0..63，
source_window_index 与元数据逐项一致且 split 内唯一。没有重复行、缺行、
非有限值或窗口错位。train 与 valid 是不同源文件，不能拿两个文件的
整数窗口索引是否相同来判断 leakage。

窗口长为 512 个输入 token，runner 实际对每窗 511 个 next-token labels 求均值。
所有窗等长，因此窗口均值的平均与 token 加权均值等价。
CSV 未保存 token 本身；共享 token 的证据是源索引合同及冻结 runner 的统一输入逻辑。

Parent mean NLL：calibration **2.527988**；
validation **2.520895**。逐窗口重建的 12 个单编辑增量如下；
全部 66 个 pair 均参与复算，而不是只检查摘要或部分行。

| Block（0-based） | Calibration Δ NLL | Validation Δ NLL |
| --- | ---: | ---: |
| 1 | 3.004447 | 3.108133 |
| 3 | 0.183047 | 0.154821 |
| 6 | 0.109025 | 0.105764 |
| 8 | 0.131361 | 0.139192 |
| 10 | 0.070682 | 0.074808 |
| 12 | 0.066587 | 0.062930 |
| 15 | 0.067188 | 0.073549 |
| 17 | 0.102604 | 0.108246 |
| 19 | 0.172133 | 0.163140 |
| 21 | 0.249799 | 0.207957 |
| 24 | 0.248236 | 0.214763 |
| 26 | 0.399105 | 0.385745 |

### 一致性结果与文档限制

`results.csv` 的全部 66 行字段、`metrics.json` 的全部 analysis 字段（含
三 calibration shards）、summary baseline 表及 SVG 的 132 个对称非对角数值均匹配。
完整数值比较容差为 atol=1e-10、rtol=1e-9；独立 SciPy 相关性计算只有浮点舍入差异。
**未发现会改变结果、排名或 verdict 的数值不一致。**

历史 protocol 开头仍写着“design only / runner 未实现”；它与后续归档状态不一致，
属于过时叙述，不是指标冲突，本轮保留原文。协议提及的 prediction scatter
未出现在这六个原文件中；现有 SVG 是 interaction heatmap。本轮不补写或替换原产物。

## 2. 两种情形与原指标复算

A：以 validation 单编辑分数为特征，预测 validation 双编辑效果。
mean-interaction/OLS 参数仍只由 calibration 拟合。A 只表示同数据交互诊断，
不是仅凭 calibration 选模型的独立搜索评测。

B：参数、特征、候选评分和最终选定全部只依赖 calibration；
validation 只评估已选候选及事后候选集合参照。
这是原 `deployment_metrics_using_calibration_singles` 的决策层展开。
两 split 共享同一批 block pairs，故也不是对未见 pair 的泛化证明。

Additive 为 Δ_i+Δ_j；mean-interaction 加上 calibration 均值
**0.0457849710 NLL**；OLS 按 i<j 排序拟合
**b0=0.0513056249、b1=1.0201892032、b2=0.9103432164**。
OLS 的 upstream/downstream 系数不同是冻结特征定义，不是本轮调参。
常数 offset 不改变排序。

| 情形 | Baseline | MAE | Median AE | RMSE | Pearson | Spearman | Kendall τ-b |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | additive | 0.069167 | 0.026426 | 0.116035 | 0.995805 | 0.959127 | 0.849883 |
| A | mean_interaction | 0.070799 | 0.038874 | 0.106362 | 0.995805 | 0.959127 | 0.849883 |
| A | linear_regression | 0.063284 | 0.031082 | 0.103708 | 0.995815 | 0.959962 | 0.852681 |
| B | additive | 0.080135 | 0.031223 | 0.129178 | 0.995900 | 0.949692 | 0.839627 |
| B | mean_interaction | 0.088688 | 0.064430 | 0.120891 | 0.995900 | 0.949692 | 0.839627 |
| B | linear_regression | 0.076625 | 0.051290 | 0.109111 | 0.996059 | 0.956622 | 0.848019 |

完整 66-pair validation interaction 主结果保持为：

- mean |I| = **0.0691665093**，median |I| = **0.0264258403** NLL；
- signed min / Q1 / median / Q3 / max =
  **−0.384349 / 0.006537 / 0.016754 / 0.054078 / 0.363445** NLL；
- relative |I| min / Q1 / median / Q3 / max =
  **0.001082 / 0.034765 / 0.068372 / 0.131322 / 0.689117**；
- relative |I| >10% 为 **22/66（33.33%）**；amplification/cancellation 为 **54/12**；
  小分母（|Δ_i|+|Δ_j|<0.01）计数为 0。

### 配对窗口 bootstrap 与原判定

原算法和独立实现均使用 seed=20260918、2,000 replicates，每次抽取 64 个
validation 窗口索引，并将**同一个索引向量用于全部 79 个 architecture**；
每次重新计算 parent、singles、pairs、interaction 和 ranking。
没有把 66 个共享层的 pairs 当成独立样本抽取。区间如下：

| 指标 | 原 95% interval（独立复现一致） |
| --- | --- |
| additive_mae | [0.062005, 0.077021] |
| kendall_tau_b | [0.834033, 0.862005] |
| median_relative_residual | [0.058727, 0.078849] |
| spearman | [0.950944, 0.963302] |
| top_quartile_overlap | [0.823529, 0.941176] |

原 CONFIRMED 的三条点条件分别为 median relative≥0.10、ρ≤0.90、
Top-17 overlap≤0.75。本次 0.068372、0.959127、0.882353 **一条也不满足**。
NOT CONFIRMED 的三条联合条件也不满足（median 不小于0.03，overlap 不到0.90）。
因此原点判定与最终判定均为 **INCONCLUSIVE**，stable_confirmation=False。
没有借 exploratory 子集重判，也没有事后放宽阈值。

这些 interval 仅表征本次已采样文本窗的敏感性；相邻文本仍可能相关，
不构成独立语料、随机架构总体或跨模型置信保证。

## 3. 架构选择到底有没有受损？

定义预测胜者 a_hat=argmin predicted，集合参照 a_ref=argmin validation Δ。
simple regret = Δ_val(a_hat)−min_a Δ_val(a)，单位为 nats/token。
并列时用 lexicographic pair 顺序，不随机打破。

**validation“实测最优者”仅是本次 66 个候选的事后参照；
不是整个搜索空间的真实最优，也不是独立最终测试集上的最优。**

| 情形 | Baseline | 预测选中组合 | 预测 Δ | 选中者实测 validation Δ | Simple regret |
| --- | --- | --- | ---: | ---: | ---: |
| A | additive | (12,15) | 0.136478 | 0.160914 | 0.000000 |
| A | mean_interaction | (12,15) | 0.182263 | 0.160914 | 0.000000 |
| A | linear_regression | (12,15) | 0.182460 | 0.160914 | 0.000000 |
| B | additive | (12,15) | 0.133774 | 0.160914 | 0.000000 |
| B | mean_interaction | (12,15) | 0.179559 | 0.160914 | 0.000000 |
| B | linear_regression | (12,15) | 0.180400 | 0.160914 | 0.000000 |

六种情形的参照最佳者均为 **(12,15)**，其 validation NLL 为
**2.6818090342**，增量 **0.1609142926**；calibration 增量为 **0.1600976164**。
六组预测前三名与 validation 实测前三名，连同顺序，均为：

1. (12,15)：Δ_val=0.1609142926；
2. (10,12)：Δ_val=0.1692674030；
3. (10,15)：Δ_val=0.1768193226。

以上是从逐窗口 NLL 重新计算得到，未把用户提示作为证据。
calibration 实测第二、第三名的顺序相反；这不改变本次 calibration-only 胜者。
前两名 validation 差距仅 0.008353 NLL，因此 0 regret 是这个有限样本的事实，
不是证明总体最优或未来重复必然同选。

### Top-k 集合重合

| 情形 | Baseline | Top-1 | Top-3 | Top-5 | Top-10 | Top-17 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | additive | 1/1 | 3/3 | 4/5 | 9/10 | 15/17 |
| A | mean_interaction | 1/1 | 3/3 | 4/5 | 9/10 | 15/17 |
| A | linear_regression | 1/1 | 3/3 | 4/5 | 9/10 | 14/17 |
| B | additive | 1/1 | 3/3 | 4/5 | 9/10 | 13/17 |
| B | mean_interaction | 1/1 | 3/3 | 4/5 | 9/10 | 13/17 |
| B | linear_regression | 1/1 | 3/3 | 4/5 | 9/10 | 14/17 |

Top-5/10/17 并非完全一致，说明有局部错排；但不能把这些错排直接换算为 Top-1 选择损失。
例如 A-additive 的前五是 (12,15),(10,12),(10,15),(6,12),(12,17)；
B-additive 的前五是 (12,15),(10,12),(10,15),(12,17),(15,17)。
实测第五名 (6,15) 两者均遗漏，但实测第一名没有丢失。

### 先筛选前 k 个，再真实评估的结果

下表适用于**每一种** A/B × additive/mean-interaction/OLS（共六种）。
逐组合、逐 k 的完整记录由脚本输出，不能仅由 overlap 推断最佳值。

| 保留 k | Shortlist 中事后最佳 validation Δ | 对集合参照 regret | 仅 calibration 复评选定者的 validation Δ | Regret |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.160914293 | 0 | 0.160914293 | 0 |
| 3 | 0.160914293 | 0 | 0.160914293 | 0 |
| 5 | 0.160914293 | 0 | 0.160914293 | 0 |
| 10 | 0.160914293 | 0 | 0.160914293 | 0 |
| 17 | 0.160914293 | 0 | 0.160914293 | 0 |

“Shortlist 事后最佳”以 validation 标签取 min，仅是候选覆盖诊断；
对 B，实际可执行策略是在 shortlist 上用 calibration 实测 NLL 选定，
然后查 validation。两种量分别计算，本次碰巧全部选择 (12,15)。
A 的 shortlist 本来就看过 validation singles，不能因追加 calibration 复评就变成干净搜索流程。
mean-interaction 与 OLS 的 66 个 calibration pair 标签成本也不能视为免费。
如果这些66个calibration pair已经测全，直接选择calibration实测最小者同样是
(12,15)；本次没有证据证明把这些标签再拟合成OLS能改善最终选择。

## 4. 误差来源（全部为 exploratory）

以下切片在看到 H1-small 后设计，只解释现象，不替换全体 66 pairs 主结果或原 verdict。

### 高损伤层的尺度效应

按 calibration single damage 排序，最重的层依次为 1、26、21、24。
第1层单编辑 Δ_cal=3.004447、Δ_val=3.108133，远大于其余层。
包含第1层的 11 pairs 与其余55 pairs 的组间差异解释了 validation pair Δ
总平方离差的 **98.02%**。这是方差分解，不是 Pearson 因果贡献或独立样本显著性检验。
因此整体 Pearson≈0.996 主要显示模型能区分“极坏”和“较轻损伤”，
不能单独证明低损伤区域精确排序。

| 排除 calibration 最重层 | Pairs | A MAE | A Pearson | A Spearman | A Kendall | B Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无（全体主结果） | 66 | 0.069167 | 0.995805 | 0.959127 | 0.849883 | 0.949692 |
| 1 | 55 | 0.045834 | 0.876035 | 0.933911 | 0.810101 | 0.917821 |
| 1,26 | 45 | 0.038006 | 0.853957 | 0.898287 | 0.769697 | 0.868511 |
| 1,26,21 | 36 | 0.032720 | 0.843992 | 0.897555 | 0.749206 | 0.867954 |

排除第1层后 Pearson 降至0.876035，Spearman 仍为0.933911。
排除更多层后局部排序继续变弱，但绝对误差反而下降；
缩小损伤范围也会机械地降低相关性。这些子集没有独立候选总体解释。
其中第1层相关 pairs 占全部 |I| 总和 **44.78%**、I² 总和 **57.42%**。

### 大交互是否集中在不会被选择的架构？

| Pair | Signed I | 实测 Δ_val | 实测排名 | A-additive 排名 | B-additive 排名 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (1,3) | -0.384349 | 2.878605 | 56 | 62 | 63 |
| (24,26) | 0.363445 | 0.963952 | 55 | 55 | 54 |
| (1,21) | 0.331141 | 3.647231 | 66 | 64 | 65 |
| (1,19) | 0.279607 | 3.550881 | 65 | 63 | 62 |
| (21,24) | 0.264796 | 0.687516 | 53 | 45 | 48 |
| (1,12) | 0.226309 | 3.397372 | 62 | 56 | 56 |
| (21,26) | 0.215151 | 0.808853 | 54 | 54 | 55 |
| (1,6) | 0.210114 | 3.424012 | 63 | 59 | 60 |
| (1,24) | 0.188112 | 3.511008 | 64 | 65 | 64 |
| (6,8) | 0.168803 | 0.413759 | 41 | 20 | 17 |

A-additive 预测前17之外占全部 |I| 的 **90.96%**、I² 的 **98.22%**；
B-additive 预测前17之外分别为 **87.93% / 95.09%**。
大部分总体误差确实发生在这些策略不会优先评估的差候选上，
但不能说全部如此：**(6,8)** 被 B 排到第17，实测第41，I=0.168803，
是进入较宽 shortlist 的明显错误；它未改变 Top-1/Top-3。

A 的预测前17内，Pearson/Spearman/Kendall 为
0.844068/0.865196/0.720588，same-split MAE=0.024287。
B 的预测前17内，calibration→validation MAE=0.036512，
Spearman=0.926471；该集合的 same-split interaction MAE=0.032398。
B 的预测误差同时含交互与跨 split 分布变化，不能把二者混为一谈。

### 层距离分组

| 距离组 | Pairs | A MAE | A Spearman |
| --- | ---: | ---: | ---: |
| distance_1_to_5 | 21 | 0.117546 | 0.970130 |
| distance_6_to_12 | 24 | 0.033302 | 0.970435 |
| distance_13_plus | 21 | 0.061775 | 0.977922 |

距离≤5的组误差较高，但分组候选不同、关键层分布不同；这不是距离的因果效应，
也没有支持据此调节 H1 阈值或挑新 validation 候选。

**数值误差、排序错误、选择损失分别回答不同问题：**
MAE 反映评分校准，Top-k/相关性反映序关系，regret 才直接量化本次选择代价。
本次三者的结论是“数值不完全可加；部分中段错排；最佳候选未选错”。
OLS 的更低全局 MAE 没有带来更好的最佳架构。

## 5. 多编辑方案的必要性与边界

[H1_multi_edit_protocol.md](../experiments/canary/direction_01/H1_multi_edit_protocol.md)
给出一个待人工批准、不可自动执行的固定90候选诊断：
k=2复用本结果；k=4/6/8各30个，分全空间样本与calibration低损伤样本；
候选清单、数据隔离、预算、实际选择损失门槛与停止条件都预先锁定。

值得做的理由仅是**两编辑结论不能外推到同时四到八编辑**，而且利用现有
calibration singles/pairs 能用很小的面板比较加法与完整二阶评分。
这不是原协议“已稳定击穿后扩大位置/配对”的自动扩展：其触发条件未满足，
所以这是另一份待批准的诊断提案。低损伤候选上的 regret 是核心，
不能靠删除第1层后损失爆炸来证明 NAS 价值。

主要反方理由：本次加法已选对最优，二阶方法又预付了66个pair测量；
在仅90个候选的规模上，直接calibration评估非常有竞争力。
若低损伤区仍无实际 regret，或简单二阶/等预算直接评估已解决，
应停止为 CET 搭建复杂方法，而不是继续加 k 寻找正结果。

## 6. CPU 复现与自检

环境：Python 3.12.3、NumPy 2.2.6、
SciPy 1.15.3、PyYAML 6.0.3。
在从归档提交建立的独立副本内，以 Python 3.12 的隔离环境安装这些 CPU 依赖及
pytest 8.4.2、ruff 0.12.11、black 25.1.0；模型库不需要安装。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  .local-deps/review-env/bin/python -m paper5.analysis.review_h1_small \
  --output .local-deps/h1_review.json
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  .local-deps/review-env/bin/python -m pytest \
  paper5/tests/test_h1_small_review.py paper5/tests/test_h1_small_analysis.py -q
```

脚本输出全部79 architecture均值、三baseline、A/B各shortlist、exploratory切片、
原2000次bootstrap复现及设计候选清单；输出仅允许写入被忽略的 `.local-deps/`，
不会写回 H1_small。测试覆盖窗口破坏、validation label泄漏、shortlist事后
最优与calibration选择的区别、联合重采样以及冻结候选清单，并禁止复核导入
torch/transformers。额外探测的两项旧 `test_interaction_canary.py` 测试因其
模块顶层依赖未安装的torch失败，不纳入本次CPU-only适用测试，历史文件保持不变。

本轮适用测试 **20 passed**；仓库级 `ruff check .` 通过；两个新增Python文件的
Black检查通过；`git diff --check` 通过。仓库级Black检查发现一个已有格式问题：
`tools/researchclaw/gpu_queue/state.py` 的多行 `executescript` 调用换行。
该文件与归档提交完全一致，本轮不修改无关工具；不能据此宣称全仓库Black通过。
最初Black的多进程检查在沙箱中挂起，已终止，逐文件只读复查确认是上述单个遗留项。
原仓库没有根 `tests/` 目录，本次pytest显式指向新增复核与原H1分析测试，
未运行包含模型前向的smoke测试或任何GPU代码。

## 7. 明确回答

1. **已证实什么？** 固定归档内部一致；原逐窗统计/配对bootstrap可复现；
   双编辑存在非零交互与部分错排；在本次66候选上，六种策略的Top-1 regret均为0。
2. **尚未证实什么？** 稳定、实际有意义的架构选择损失；多编辑加法失效；
   其他模型/语料/edit family的普适性；CET效果或成本优势；完整NAS价值。
3. **简单加法有没有选错重要候选？** 没选错本次最佳者，前三名也正确。
   较宽shortlist有遗漏及误入，不能宣称完全不影响所有下游搜索用途；
   但没有观察到本次Top-1选择损失。
4. **多编辑实验是否值得做？** 仅值得作为一次固定预算、允许得到负结论的
   诊断；不是H1成功后的扩张，更不是方法实现授权。直接评估成本可能使其提前失去研究价值。
5. **现在有理由启动H2吗？** **没有。** 保留INCONCLUSIVE与Research Opportunity状态，
   不升级Paper Candidate；等待用户确认是否批准所提多编辑范围。
