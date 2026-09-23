# D2 exploratory recovery-capacity diagnostic V1

用户在原 Stage0 工程先导通过、质量准入失败后，另行授权一次探索性续训。
本文件不修改 STAGE0_PROTOCOL.md，不替换原 QUALITY_INFEASIBLE 停止记录。
原执行代码为 `75e3253b8f8b78c5998730c039ad346231d350bc`，先导归档为
`66d1b974c4379cc0866fc10f29a739a228e819d8`。

唯一对象为 a00（删除父模型原层 12/15），seed 17。从已保存 step_10.pt
恢复 adapter、AdamW 一二阶矩及 step、CPU/CUDA RNG、完整训练窗口顺序。
checkpoint SHA256 固定在 recovery_capacity.json。禁止重新训练前 10 步。
复用原 Qwen2.5-1.5B revision、tokenizer、T400/S64、bf16 base、FP32 LoRA
q_proj/v_proj、r8/alpha16/dropout0、AdamW lr1e-4/weight_decay0、clip1、
512 token/microbatch1/accumulation4。仅 adapter 可训练，不搜索超参数。

续训执行 updates 11..100，即原顺序 [40:400] 的 360 个不同训练窗口，新增
184320 输入 token、183960 有效 labels；含原 10 步共 204800/204400。
只在 20/50/100 保存并评价同一 S64；0/10 与父模型值直接引用先导原记录。
恢复后另用 S 的首窗核对 step10 重载一致性，不产生新的选择依据。
所有步数走同一 optimizer 轨迹，即使 20 步达标也按授权完成到 100 步。
E token 文件始终封存，诊断没有 E 解封或正式训练入口。

质量固定为 exp(mean_S_NLL - parent_S_NLL) <= 1.15；同时报告 NLL、
NLL 比值、PPL 及 PPL 比值。这是原工作性质量容忍度，非行业标准。
100 步未通过记 EXPLORATORY_CAPACITY_QUALITY_NOT_MET，并建议该候选/配方
不进入正式 Stage0。通过只记 EXPLORATORY_CAPACITY_FEASIBLE，说明该候选
恢复容量可行；不能说明两个候选达标、存在选择损失或恢复预测器有必要。
两种结果均停止，必须由用户重新批准才能启动正式协议。
本诊断由观察先导结果后提出，属于内部探索评价，不是预注册确认试验。

固定物理 GPU2 及配置所列 UUID，继承明确共享许可；现有 permission/admission
检查保留，另记录 capacity 阶段授权。启动前连续三次采样，原 Stage0 与本诊断
共用互斥锁；独立运行目录拒绝重复启动。无权干预其他进程或切卡。
先导实耗 189.12083893828094 秒仍计入原 14400 秒硬上限。按原先导最慢
update/window 乘 1.5 加 300 秒预估，检查剩余预算，超时保存中断 checkpoint。
资源不足、NaN、base 改变、checkpoint/数据/参数映射不符即停止，不丢弃异常
后继续报告科学成功。CPU 分析不超过原 30 分钟。

新增结果仅写 recovery_capacity/20260923_a00_seed17_resume10，adapter/运行日志
仅写独立 direction_02 cache/scratch 子路径。原 Stage0 协议、配置、runner、
results、cache ledger/checkpoint 和 D1 结果保持原字节。公开归档只含小型数值、
配置、数据/候选 manifest、摘要、成本与脱敏 runtime；不提交权重/语料/原始日志。
执行前完成代码提交，同步确切 SHA；结果归档 SHA 另行记录。
