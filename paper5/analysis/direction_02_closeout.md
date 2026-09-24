# Direction 2 当前问题探针收尾

日期：2026-09-24。状态：**当前候选家族与恢复配方的验证结束；FHRD/RSC 暂停；不是 Paper Candidate。**

## 固定证据

本次只阅读下列既有报告、协议与当前 Git/接续状态；未发现需要重新全量审计的矛盾，没有重跑 CPU 审计或模型实验。

| 证据 | 结果归档提交 | 实际执行代码提交 |
| --- | --- | --- |
| [原 8 候选 V2 报告](../results/canary/direction_02/recovery_stage0_v2/20260923_stage0_quality_v2_01/REPORT.md) | `123cd0ca2c7011a8bd1560aee47984954fe09a20` | `53182b532ad8980eb0c4f09f122be941e2671b79` |
| [唯一一次强竞争者覆盖报告](../results/canary/direction_02/recovery_competitor_coverage/20260924_competitor_coverage_01/REPORT.md) | `319cb64e8a95705c29fd5f893354ca0632adf894` | `28a3db86d1fd85e2505040b4aed8a79f58803978` |

原 8 个候选完成 16 条轨迹；唯一补充新增历史竞争池第二、第三名 `(10,15)`、`(10,12)`，完成 4 条轨迹。扩展集合共 **10 个候选、seed 17/29**，不是把原 16 条再次训练得到的另一组重复。每条正式轨迹从父模型继承权重、新 LoRA 和 optimizer 开始，完成 100 个 optimizer updates；此前 a00 容量续训只作为准入依据。

## 结论及其边界

- 原 V2 两个 seed 均为 8/8 质量达标；扩展集合均为 10/10，按既定 100 步 E 上相对父模型 PPL ≤1.15 判断。恢复链路有效，当前候选恢复后质量可用。
- 原 8 候选和扩展 10 候选中，**B0、B20、BSH、Ref100 在两个 seed 均选择 a00，即删除原模型零起点层 `(12,15)`**。G0、B20/BSH 相对 Ref100 的终点差值，以及有限候选 E 事后 simple regret，均为 0，报告中的文档配对区间均为 `[0,0]`。
- 这些零区间源于策略最终引用同一个 checkpoint，不是“不同架构普遍等价”的统计证明。中段排名确有变化；扩展集合 seed 17 的 S Top-3 也变化，但没有改变终点最优选择。
- 原判定 **NO_MATERIAL_SELECTION_GAP** 和 **SIMPLE_BASELINE_SUFFICIENT** 保持。恢复成功并未产生已测得的实际选择损失，不足以支持复杂恢复预测器的必要性。
- 这是 Qwen2.5-1.5B、真实删除两个 decoder blocks、100 步 next-token CE/LoRA、固定候选与 WikiText-2 内部数据上的结论。V2 质量准入修订发生在查看 a00 的 S 曲线之后；覆盖补充时 E 已曝光。结果属于内部探索，不是新的盲确认。
- 未测试完整 FHRD/KL、RSC 或其动力学预测器，不能将此次收尾写成完整方法被否定，也不外推至其他恢复配方、模型或所有恢复感知 NAS。父模型没有做对应恢复训练，不能把恢复后候选优于未适配父模型归因于压缩。

**执行决定：结束当前“双层删除＋100 步 CE/LoRA”问题验证；不追加候选、步数、seed 或容量试验；暂停 FHRD/RSC。** D1 当前整层跳过路线仍结束，CET 仍暂停。D1 原始结果和判定、D2 V1 先导停止、capacity、V2 与覆盖补充的各自结论全部保留，不覆盖或重判。

## 可复用基础与保留状态

| 基础 | 入口与用途 | 复用边界 |
| --- | --- | --- |
| 数据划分和追溯 | [stage0_data.py](../experiments/canary/direction_02/stage0_data.py)：文档 ID、源/tokenizer/token 哈希、窗口清单、固定训练顺序 | T/S/E 为项目内文档互斥的 400/64/64 个 512-token 窗；与 D1 calibration 的关系已披露。旧 E 已曝光，不能重新命名为新盲评价集 |
| 真实结构变更和恢复 | [stage0_model.py](../experiments/canary/direction_02/stage0_model.py)、[run_stage0.py](../experiments/canary/direction_02/run_stage0.py)：层映射、冻结 base、LoRA、保存重载和 0/20/50/100 轨迹 | 复用工程能力不等于复用已曝光的评价证据，也不自动授权新模型运行 |
| 评价与策略分析 | [stage0_analysis.py](../experiments/canary/direction_02/stage0_analysis.py)、[stage0_coverage_analysis.py](../experiments/canary/direction_02/stage0_coverage_analysis.py)：逐窗/逐文档 NLL、S 选择先于 E、预算受限回放、配对 bootstrap | 原 8/扩展 10 的策略与成本分别记录；有限集合事后 oracle 仍有选择偏差 |
| 运行保护 | [stage0_resources.py](../experiments/canary/direction_02/stage0_resources.py)、[D2 CPU tests](../tests/direction_02/) | 资源许可、重复进程、显存、预算与异常停止检查可复用；新方向须重新取得授权 |

依据既有覆盖报告，新增 GPU 运行 676.885 秒，含先导、capacity 与 V2 的累计运行 3052.022 秒；不是本轮新增成本。本轮没有连接 A800、移动/删除/改写结果、缓存或 checkpoint，也没有重新复制归档。收尾文档提交是新的文档提交，不能当作实验执行或数值结果归档 SHA。

下一步仅为 [Direction 3–5 定向复评与选择建议](next_direction_decision_after_d2.md)。**尚无新方向获得 GPU 实验授权，等待用户选择。**
