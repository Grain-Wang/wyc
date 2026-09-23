# AutoResearch 当前接续状态

> **2026-09-23 paper5 / 单候选 exploratory capacity 续训已授权，尚未运行**
>
> 用户授权同一 a00（移除原层 12/15）、seed 17 从既有 step10 checkpoint 继续至 100；20/50/100 仅评价 S，1.15 PPL 比值门槛不变。新增独立 recovery_capacity 协议、入口及输出目录；不改下方原 Stage0 停止结论，不打开 E，不自动启动 8×2。先通过 CPU 续训一致性测试，提交并同步代码后在物理 GPU2 共享许可下运行，计入原 4 小时预算。无论容量是否通过，完成后停止等待用户重新批准正式协议。

> **2026-09-23 paper5 Direction 2 / D2-Stage0（先导完成，质量门禁停止）**
>
> - 当前用户选择仍是 Direction 2 的一次 LoRA + next-token CE / NLL 问题探针；不是 FHRD/KL 复现，不实现 Hessian、NTK、JVP/VJP、NAS 或完整方法。下方旧 paper1 等任务不继承。
> - D1 当前整层跳过路线结束，CET 暂停。基准 `6ee61e753d8e67b21c56adfb2edeec082e076f10` 的 H1-small INCONCLUSIVE 与多编辑 STOP_COMPLEX_METHOD_INVESTMENT 保持不变。
> - GitHub 认证已恢复；代码 `75e3253b8f8b78c5998730c039ad346231d350bc` 普通推送并精确快进同步 A800。远端 25 项 CPU tests、Ruff/Black/shell 检查和源/tokenizer/T/S/E/模型哈希通过。Ruff 的 `python -m` 继承环境入口缺失，静态检查改用同版本现有原生程序完成，未改 D1 环境。
> - 用户随后明确授权只共用物理 A800 GPU 2。已有许可机制、连续采样、重复任务检查和 detached screen 均使用；未切到其他卡，未干预现有进程。实际先导执行 SHA 为 `75e3253b8f8b78c5998730c039ad346231d350bc`。
> - 单候选 a00（原层 12/15）、seed 17 的 10 步先导真正完成，退出码 0；base 不变、LoRA 更新有效、保存重载 NLL 完全一致。GPU 累计 189.121 秒，峰值 allocated/reserved 为 3736.218/4202 MiB；完整任务保守估算 10534.927 秒，预算门槛通过。
> - **质量门禁未通过，禁止自动正式开跑**：父模型 S NLL 2.461201183，8 个初始候选中 0 个满足既定 PPL 比 <=1.15（至少需 2 个）。a00 从 2.624146957 降至 2.608408062，但终点相对 PPL 1.158593626 仍超标。判定 QUALITY_INFEASIBLE 仅指本轮先导准入；不证明 100 步恢复无效。
> - 16 条正式轨迹、100 步模型、E 评价、G0、B20/BSH 和 bootstrap 均未执行；问题信号与简单 baseline 均为 NOT_EVALUATED。未创建 formal_freeze，不改阈值、候选或 seed，不进入复杂方法、H2 或其他方向。
> - 结果入口：`paper5/results/canary/direction_02/recovery_stage0/20260923_stage0_01/summary.md`，含先导逐窗 NLL、10 步曲线、工程/质量/成本/运行记录及原始产物 SHA 清单。结果归档提交是后续提交，必须与实际执行 SHA 区分。
> - 400/64/64 个 T/S/E 窗及 8 个候选保持冻结，WikiText-2 train 与 D1 calibration 的历史关系已披露；仍仅内部探索评价。原始 A800 产物、私有日志和先导适配器保留，原工作区未动。此轮完成归档后停止；任何科学设定修订需另行决定。

> **2026-09-20 paper5 H1 多编辑隔离修订 V2（当前任务状态）**
>
> - 用户已授权仅修正内部确认窗口隔离，并在测试、提交和 A800 预检通过后继续一次原固定 90 候选诊断；未授权 H2 或扩量。此前阻塞已按源码/数据/原 PyTorch 版本复核成立。
> - 原 H1-small 六个文件与 INCONCLUSIVE 保留；原多编辑协议不改。V2 补充、冻结 JSON 和执行记录位于 `paper5/experiments/canary/direction_01/`。
> - V2 保留旧窗及 ±1 排除，再排除所有已核验 smoke/sanity source-token 重叠窗及 ±1；原 hash 规则从 325 合法窗取 64。移除 37/162/163/313/392/393；按 hash 次序补入 436/73/364/527/476/218。候选 manifest 不变。
> - 单次实验现已真正完成：执行 SHA `eba72bc9e9f59634e0f2f2639d05c1de1c38f308` 已发布并通过 SSH/bundle 精确同步 A800；远端 99+8 CPU tests 和预检通过。GPU 1 模型阶段 910.579 秒、分析 2.111 秒；14,464 测量行+8 安全窗，完整性核验通过，退出码 0、state COMPLETE。
> - 原协议判定为 **STOP_COMPLEX_METHOD_INVESTMENT**：low k=4/6/8 的加法 G 分别 0.109985/0.120567/0.268271，但各 low 面板质量合格候选均 0/15，实用性门槛全部失败。保持 H1-small INCONCLUSIVE，不进入 H2、不扩量；不是 Paper Candidate，不将损伤面板的选择误差包装成实用 NAS 价值。
> - 用户随后明确授权将 13 个既有小型结果文件及执行/接续摘要归档到 `Grain-Wang/wyc` 的 `paper5`。结果目录为 `paper5/results/canary/direction_01/H1_multi_edit/`；从完成归档逐字节复制，不重跑或改数值。实际执行 SHA 始终为 `eba72bc9e9f59634e0f2f2639d05c1de1c38f308`，后续结果归档 commit 不得当作执行代码。GitHub 分支和内容复核以交付记录为准。
> - 原始数值结果仍在 A800 对应结果目录；本地原始镜像及私有日志保留于忽略目录 `.local-deps/H1_multi_edit_V2_archive/`。原始日志、认证/连接信息、模型和语料不上传。结果 SHA 与证据入口见 H1_multi_edit_execution.md；本轮归档完成后停止，不开展任何新实验。
> - 本地原 AutoResearch 的既有修改未触碰；使用已授权独立副本。称谓固定为“项目内隔离后的内部确认集”，无跨文档独立或预训练无污染主张。已披露旧 smoke/sanity 缺 contemporaneous index manifest 的追溯限制。

> **2026-09-13 `paper5` NAS/LLM 研究目录与方向选择门禁（优先于下方旧项目内容）**
>
> - 当前活动分支为 `paper5`，NAS/LLM 研究内容统一归档于 `paper5/`；不得再建立根目录 `research/nas_llm/` 作为并行研究目录。
> - 根 `AGENTS.md` 已加入 **Research Direction Selection Gate**：首次形成至多五个候选方向后必须向用户完整汇报并停止；用户明确选择方向前，不得认定 Rank 1、创建或执行 canary、启动训练、进入算法 prototype、大规模 baseline、Paper Candidate 或完整论文阶段。方向被证伪后的切换也必须重新确认。
> - 现有未运行的 interaction canary 脚本、配置、测试和依赖声明已保留并迁移到 `paper5/` 对应目录；未产生数值结果，亦未启动模型下载、推理或训练。该实现不是选题确认或科学证据，只有用户明确授权相应 Research Opportunity 后才可继续。

> **2026-08-27 Round7/Round8 P0 最终状态（优先于下方全部内容）**
>
> - 当前分支为 `paper1`，Round7/Round8 P0 代码与协议提交 `f88ff65`、Depth Anything 本地锁定修复 `04b28d2`、运行时 provenance `65a4b3f`、正式 004-A 结果与停止决定 `9cd7dea`。Round6/Round7 回应文件名已修正为 `response_round6.md`、`response_round7.md`；历史目录名因既有链接保留。本轮新增 `response_round8.md`。
> - post-Step003 历史选择为 `RECOVER_TWO_REAL_DATASETS`；Cityscapes、ScanNet v2、Matterport3D 恰好三个候选均为 `PENDING_SOURCE_ACCESS/BLOCKED_SOURCE_ACCESS`，不是 coverage FAIL。但 004-A 已触发更上游科学停止，故不再为 CoVoL 恢复第二数据集。
> - NYUv2 diagnostic-only corpus 为 100 图、59 clusters、1200 local rows（四族各 300），machine-check 1200/1200；独立规则 parser 100/100，held-out-template text-only macro-F1 0.488，自动 surface-form 1200/1200；人类 naturalness 未评估。
> - 004-A 已在单张 A800 上完成 100 图/1200 rows、10,000 次 cluster bootstrap。region AbsRel degradation：semantic-preserving `0.001156 [0.000579, 0.001777]`、target deletion `0.000055 [-0.001198, 0.001109]`、local entity conflict `0.001620 [0.000195, 0.002903]`、depth relation conflict `0.000806 [0.000347, 0.001298]`。semantic-preserving CI 不含 0，违反预注册控制条件，正式状态为 `STOP_H_SENSITIVITY`。
> - 逐行 CSV SHA256 为 `a2d45fe96581d3234aa41d62c2a63f3e793f705e56c6054e9c8c3818111db721`，summary SHA256 为 `e4a304b1e6c2d8db6b1b95a666fe7f9fb88e73c7200addc400aecb10b2ce4659`。summary 锁定 TR2M、Depth Anything、DINOv2、CLIP、NYUv2、代码、协议与运行时；环境为 Python 3.12.13、torch 2.5.0+cu121、CUDA 12.1、A800 80GB。
> - 科学状态为 **CoVoL STOPPED_BY_H_SENSITIVITY_CONTROL / Claim-F STOPPED / Claim-M STOPPED / Paper Candidate 否**。不能选择性忽略 semantic-preserving 对照，也不能继续 004-B、005、006、007、008、D0/D1 或 router。Q-GeoRoute 仍 PARKED；下一步是重新形成至多五个通过 Research Opportunity Gate 的候选并更新唯一范围锁。
> - 最终本地 QA：仓库 Ruff 通过；tools/paper1 全部 Python 文件 Black 通过；`tools/tests + paper1/tests` 共 196 tests 通过；JSON/CSV 完整性、敏感信息扫描与 `git diff --check` 通过。

> **2026-08-25 Round6 回应与 GPU 排队暂停状态（优先于下方全部内容）**
>
> - Round6 的正式回应已更新为 `paper1/responce_from_reviewer/response_round6.md`，`paper1/README.md` 已同步入口。回复逐项区分 `DONE-CODE`、`FROZEN-DESIGN`、`OPEN` 与 `BLOCKED-BY-STEP003`，没有把协议修改写成实现或科学结果。
> - 统计与实验设计已冻结为：full-official-crop region weights；cluster-balanced 主 estimand；per-seed dev retention one-sided 95% LCB；internal-test retention/CI 与 `STOP_TEST_RETENTION_VIOLATION`；`CVaR/WorstOf3@Dev-Ret>=0.80`；三 seeds `17/29/43` 的 paired seed×cluster hierarchical bootstrap；实体级 OOF/cache/training-manifest 与 operating-point lineage；Main-PR/Risk-L2D-C 仅 target construction 不同的 executable contract。
> - 上述 Round6 修订尚未进入 scalar objective、constrained evaluator、bootstrap、feature callables、OOF validator、PyTorch D0/D1/router 或 intervention builder。`steps/004/005/006/008` 已改为显式 blocked 状态，指标层改为 `ROUND6-CODE-REVISION-PENDING`。
> - shared CUDA canary 已在 GPU 2 与 1 个既有 compute process 共存时 `PASSED/attempt=1`；约 47,577 MiB free，PyTorch `2.4.1+cu121` / CUDA 12.1，tensor sum 1024.0，约 2.85 s，peak allocated/reserved 约 0.0044/2.0 MiB。它只验证调度和环境，不是 D0/D1/router 或科学证据。
> - 按用户指令，dual queue 已 graceful drain；后台 `screen` 已停止。exclusive job 保持 `PENDING/attempt=0`，shared job 与结果保持 `PASSED/attempt=1`，SQLite state、配置和结果均保留。后续可从同一 state 恢复，不会重复 shared job；恢复前仍须满足科学门禁与用户明确指令。
> - 本轮交付前仓库级 Ruff 通过，所有 tracked/untracked Python 文件逐文件 Black check 通过，`tools/tests + paper1/tests` 合计 156 tests 通过，`git diff --check` 通过。首次 pytest 收集因本地 `PYTHONPATH` 未包含 `tools/` 失败，补齐仓库源路径后完整复跑通过；该环境入口问题没有被写成代码故障。
> - 科学状态没有改变：**Research Opportunity / Claim-F UNVERIFIED / Claim-M STOPPED_FOR_CURRENT_TWO_DATASET_BRANCH**，不是 Paper Candidate。下一项能改变论文判断的动作仍是选择并通过合规真实 outdoor dataset 的 Step003，或正式缩窄为 NYUv2 单数据集 controlled stress-testing 后重新做 novelty/power gate；当前不启动更多 GPU 实验。

> **2026-08-25 A800 安全快捷连接与状态快照（优先于下方远程连接旧内容）**
>
> - 当前分支为 `paper1`，基线提交为 `54feea1`。ResearchClaw 已升级为 0.7.0，并新增 `remote check|connect|snapshot|show`；机器私密 profile、项目专用 ED25519 私钥和 pinned known_hosts 全部位于已忽略的 `.local-deps/ssh/`，不得提交。
> - 仓库根目录的私密快捷入口为 `.local-deps/ssh/a800 check|connect|snapshot|show`。`check` 和 `show` 只读；只有显式 `snapshot` 会原子更新 `~/whr/A800_STATUS.md`。通用配置格式与安全边界见 `tools/docs/REMOTE_EXECUTION.md`。
> - 项目公钥已幂等加入授权 A800 的 `authorized_keys`，禁用 agent/端口/X11 转发与 user rc；客户端强制 key-only、严格主机指纹校验、唯一 identity 和清空转发。连续两次 `BatchMode=yes` 连接成功。旧私密 Markdown 中的密码按用户决定保留作人工应急，但新工具不会读取或回退使用它。
> - 当前 `~/whr/A800_STATUS.md` 权限为 `0600`，记录服务器/GPU、Conda 与项目 Python、`whr` 摘要、QA、queue、result 和 checkpoint 状态；已核验不含地址、账号、密码、私钥、token 或本机绝对路径。`check` 前后文件 mtime 不变。
> - 当前快照再次确认远端 97 项 paper1 tests、Ruff、Black 通过，Step003 为 `STOP_TWO_DATASET_CLAIM`，conditional detectability BLOCKED，checkpoint 为 0。此次工作只改进安全连接与状态可观测性，不新增科学证据；研究仍为 **Research Opportunity / Claim-F UNVERIFIED / Claim-M STOPPED_FOR_CURRENT_TWO_DATASET_BRANCH**，不是 Paper Candidate。
> - 2026-08-25 GPU 队列升级为显式双模式对照：`shared` 在 NVIDIA 报告剩余显存不少于任务声明峰值加 4096 MiB reserve 时允许与他人进程共存；`exclusive` 仍要求 4 次×30 秒连续满足无 compute PID、显存 `<1024 MiB`、利用率 `<5%`。同一 comparison group 必须恰有一对命令、输入、输出、seed 和声明显存一致的 shared/exclusive 任务；不终止或抢占其他用户进程。
> - 双模式 CUDA canary 的 shared 分支已在有 1 个共存计算进程、剩余 47577 MiB 的卡上 `PASSED`：tensor sum `1024.0`，PyTorch `2.4.1+cu121`/CUDA `12.1`，任务峰值 allocated/reserved 分别约 `0.0044/2.0 MiB`。exclusive 分支曾在独立 `screen` 等待完整空闲窗口，现已按上方新状态 graceful drain。私密 `queue/GPU_EXPERIMENT_STATUS.md` 以 `0600` 原子更新状态、日志路径和产物 SHA256；每次运行目录以 `0700` 保存 allocation、completion、stdout/stderr 与结构化 result，文件默认 `0600`。该 canary 只证明调度和 CUDA 可执行，不是科学结果。
> - 已显式刷新脱敏 `~/whr/A800_STATUS.md`，SHA256 为 `77f94bbafb2f841c8b90ebee37a50b0195ea99db38895e8c19ab51b660d86ec6`；其中可直接发现双模式报告和 SQLite 的 shared `PASSED` / exclusive `PENDING`，未包含凭据或机器绝对路径。
> - A800 管理规则：需要 Conda 命令时显式加载管理员提供的初始化脚本，不修改 shell 启动文件；Conda 激活时禁止启动 VNC，未来如确需 VNC 必须先 deactivate；不执行系统更新；任何大下载、环境或实验前后检查家目录容量，优先保持低于 200G，绝不接近 400G。当前 `whr/paper1` 约 7.4G；完整家目录扫描超时，因此总用量尚未验证，不得误写为低于配额。

> **2026-08-24 Step003 真实可行性门禁最终状态（优先于下方旧内容）**
>
> - 当前分支为 `grain_paper1`；数据合同修复提交为 `435240e`。Step003 inferential 组合固定为 NYUv2+KITTI；VKITTI2 只能是 `synthetic_structured_auxiliary_only`，adapter 必须从完整官方解压目录生成 canonical source，不能接受外部挑帧列表。
> - 已在授权的远程 `whr` 使用 Python 3.12.13 和 CPU-only 队列准备 official-training 数据；未访问 official benchmark test，未使用 GPU，旧 GPU canary 已停止。远程 Ruff、Black、97 个 paper1 tests 与 training pilot builder 均通过。
> - 固定 pilot 为每数据集 500 图、300/100/100 split。NYUv2 有 500 eligible images、105779 eligible pairs、156 independent eligible clusters，coverage PASS；当前冻结 KITTI RGB source 没有可信 local instance-mask/depth oracle，三项均为 0，coverage FAIL。
> - 正式决策为 `STOP_TWO_DATASET_CLAIM`。KITTI 只保留 `image_level_sensitivity_only`；调度器将 conditional detectability 标为 BLOCKED，因此没有 power artifact、Step005 或 intervention corpus 任务。独立 replay 的 manifest、split audit、coverage CSV/JSON 全部逐字节同哈希。
> - 可移植结果位于 `paper1/results/covol/annotation_coverage.{csv,json}` 与 `step003_feasibility_gate.json`，不含机器路径或认证信息；原始数据、cache、active 配置和队列状态只留在远程未入 Git。
> - 科学状态为 **Research Opportunity / Claim-F UNVERIFIED / Claim-M STOPPED_FOR_CURRENT_TWO_DATASET_BRANCH**，不是 Paper Candidate。NYUv2 单数据集 PASS 不能外推跨域 Claim-M；若要恢复该主张，必须先明确选择另一个满足相同 provenance/coverage 且至少 20 个独立 clusters 的真实 outdoor dataset。当前停止，不自动进入 Step005。

> **2026-08-24 `grain_paper1` 工具精简覆盖状态（优先于下方全部旧内容）**
>
> - 已从 `chore/integrate-tools-and-paper1` 的 `e90be8b` 创建并切换到 `grain_paper1`。`tools/` 现在只保留根 `AGENTS.md` 约束的通用科研核心：16 阶段产物契约与门禁、真实公共文献检索、本地/密钥 SSH 实验执行、协作式 GPU 队列和 PDF 文献转换。
> - 已硬删除旧 LLM 产品层、外部 agents/skills、网站与前端、HITL、MCP、语音、Overleaf、ARC benchmark、Docker/Colab/模拟执行、营销资源和对应测试；旧 CLI 不保留兼容占位。工作流仅在阶段 4 调用真实学术 API、阶段 13 执行真实实验，其余阶段只验证科研负责人提供的证据产物，不生成假模板、假引用或假结果。
> - 每次工作区/阶段执行会发现并记录最近的权威 `AGENTS.md` 哈希。阶段 5/8/10 强制验证 `PASS|STOP` 门禁及工作区内的真实 evidence 路径；阶段 16 只接受 `STOP|REFINE|PIVOT|PAPER_CANDIDATE`。SSH 强制非交互密钥认证、保留主机密钥检查、最多 4 GPU、唯一远程目录和结束清理。
> - 当前本地验证：`ruff check .` 通过；`black --check .` 在 `tools/` 检查 40 文件、在 `paper1/` 检查 36 文件通过；精简后工具测试 `44 passed`，`paper1` 回归 `92 passed`；CLI tools list/init/status、GPU queue validate/dry-run 与 `git diff --check` 通过。未在本轮连接或改变远程服务器。
> - 下方 GPU 队列远端状态是前一轮最后一次核对结果，不代表本轮重新查询。队列实现和 paper1 示例已并入本次精简，但正式任务仍只能在真实输入和对应执行入口就绪后启动。
> - 这些变化只改进科研执行与治理，不新增科学事实。科学状态仍为 **Research Opportunity / Claim-F UNVERIFIED / Claim-M UNVERIFIED**，未达到 Paper Candidate；下一研究动作仍是取得合法真实输入并通过 Step003 coverage/power gate。

> **2026-08-24 GPU 队列与远端部署旧覆盖状态**
>
> - 当时分支为 `chore/integrate-tools-and-paper1`，修改基线为 `e90be8b`。已新增通用的 `researchclaw gpu-queue`：SQLite 持久状态、依赖/gate 阻断、失败重试、进程组超时、输出哈希、同账户文件锁、重启恢复和 drain 停止。
> - 冻结的协作式 GPU 策略为：无 compute PID、显存使用 `<1024 MiB`、利用率 `<5%`，每 30 秒采样且连续 20 次才算空闲；随机退避后二次检查；只运行独立单卡任务且最多并发 2 个；不终止或抢占其他用户进程。
> - `paper1/configs/covol/remote_queue.example.yaml` 只排队 QA 与 Step003 manifest/coverage/power gate。Steps005--008 尚无训练执行入口，不能提前加入正式队列；真实输入未就绪时也不能启动主队列。
> - 远端已在 `~/whr/paper1` 下建立隔离的数据、缓存、产物、运行、队列和 Python 3.12 环境，代码与私有绝对路径配置已部署；配置 `validate`/`dry-run` 成功，dry-run 未创建状态库。远端 Ruff、Black 和 `paper1` 的 92 个测试通过。
> - 节点没有 `tmux`，已使用 `screen` 启动独立的单任务 CUDA canary 队列。检查时三张 GPU 均有其他计算进程，所以 canary 正确保持 `PENDING`，没有占用 GPU；达到连续 10 分钟严格空闲后才会执行极小张量计算并自动退出。
> - 本地新增范围 Ruff/Black 通过；队列与 CLI 目标测试 `23 passed`，`paper1` 回归 `92 passed`，`git diff --check` 通过。仓库级 `tools/` 仍有既存 lint/format/缺失模块债务，未在本任务中扩大修改。
> - 这些结果只证明调度器和现有研究协议可执行，不是科学证据。科学状态仍为 **Research Opportunity / Claim-F UNVERIFIED / Claim-M UNVERIFIED**，未达到 Paper Candidate；下一研究动作仍是取得合法真实输入并通过 Step003 coverage/power gate。

> **2026-08-22 Round-5 最新覆盖状态（优先于本文全部旧内容）**
>
> - 已从远端 fast-forward 到审稿基线 `ba4bf48`，新增意见为 `paper1/responce_from_reviewer/review_round5.md`；本轮回复为 `response_round5.md`。
> - 用户已改为授权本机运行小规模代码测试，但校内 Linux 仍不可达：本机不得下载完整研究数据、训练模型或生成科学结论；在用户明确通知恢复前不得再连接 Linux。
> - 上一轮 47 文件的 paper1 回归已在本机隔离 Python 3.12 环境复核：`ruff check paper1`、`black --check paper1` 通过，`pytest paper1/tests -q` 为 `92 passed`。仓库级检查仍被 `tools/` 中既存 lint/format debt 阻断，本轮未批量改写该无关范围。
> - Round-5 已修正：Main-PR 的 complete-caption WorstOf3 风险、标准 signed Lagrangian 与冻结优化常量；`cluster_id` bootstrap；dev-only frozen threshold + internal-test CVaR/WorstOf3 paired CI；VKITTI2 2.0.3 adapter/provenance 和 hash-linked fallback decision；feature extractor allowlist/runtime sanitizer；D0 learned-null/active-gradient 公平合同；conditional detectability 的克制命名。
> - VKITTI2 的所有变化按基础 `SceneXX` 聚类，官方只有 5 个独立基础场景；不得把天气/相机 clone 冒充 20 个独立场景。真实门禁可能因此 STOP，这是协议要求而非待绕过的工程问题。
> - 科学状态仍为 **Research Opportunity / Claim-F UNVERIFIED / Claim-M UNVERIFIED**。仍缺真实 manifest、coverage/detectability、checkpoint、实体级 OOF cache、缺陷复现、killer baselines 和主结果。
> - 旧恢复点 `stash@{0}`（`codex-pre-review-pull-20260822`）仍保留，不得无授权删除。

> **2026-08-22 最新覆盖状态（优先于下方旧快照）**
>
> - 当前分支 `chore/integrate-tools-and-paper1` 已包含远端审稿基线 `cae15e4`；审稿改进提交 `6e197dc` 已于 2026-08-22 推送到同名远端分支。新增意见与回复分别为 `review_20260821_153624.md`、`response_20260821_153624.md`。
> - 拉取前工作保存在 `stash@{0}`（`codex-pre-review-pull-20260822`），且已成功 apply 到当前工作树；远程验证、提交和推送完成前不得删除该恢复点。
> - 用户明确要求：本机禁止运行测试或下载实验数据；所有测试与数据操作只能在 `sshconfig.md` 指定的远程 Linux 上执行，数据必须放在远程 `whr` 下。禁止提交或披露 `sshconfig.md` 中的私密连接信息。
> - 审稿后主线改名为 **Main-PR**（cross-fitted partial residual，而非 orthogonal moment）。唯一待证伪边界是在两个冻结同任务 metric-depth candidates 之间，在 clean retention `>=0.80` 约束下最小化局部 caption-error upper-tail regret；dev 固定最低 CVaR threshold，internal-test 主报 CVaR/WorstOf3，hypervolume 降为 secondary。
> - 已实现但尚未远程验证：TIGER 最近邻/机制矩阵、唯一目标与 RU-CVaR/dual/伪代码、training-only NYUv2/KITTI adapters、scene-sequence connected-component split、trusted provenance、coverage/power gates、formal failure lineage、bootstrap invalid-denominator stop、扩展 feature firewall、B/C 等宽双 permutation、faithful/matched baseline 合同。回复草案为 `paper1/responce_from_reviewer/response_20260821_153624.md`。
> - 科学状态仍为 **Research Opportunity / UNVERIFIED**；没有真实数据门禁、checkpoint 或主结果，不能升级 Paper Candidate。
> - 2026-08-22 用户说明当前在校外，暂时无法访问校内 Linux 节点，并明确要求停止连接 Linux、直接提交本轮工作。故本轮 Ruff/Black/Pytest、真实数据下载及 coverage/power 门禁均为 `DEFERRED_BY_USER`，不得写成已通过。
> - 用户后续明确通知 Linux 可用后，下一原子动作是：在远程 `whr` 下保留已有文件并建立隔离工作副本 → 同步修改 → 在 `vlm` 环境确认 Python 3.12 → 远程 Ruff/Black/Pytest → 仅用 official-training 数据运行 NYUv2/KITTI coverage 与 20-grid × 5,000 formal power gate → 同步必要修复并追加提交。
>
> 下方为 2026-08-21 旧快照，只保留作历史导航；其中基线提交、环境命令和“下一原子动作”如与本覆盖状态冲突，均已失效。

- 更新时间：2026-08-21 14:13（Asia/Shanghai）
- 接续类型：脱敏、可移植状态快照
- 原始 Codex 会话：未包含

## 启动检查

1. 完整读取根目录 `AGENTS.md`；
2. 运行 `git status --short --branch`、`git log -1 --oneline`，以实际工作树为准；
3. 阅读 `paper1/README.md`、`paper1/steps/README.md` 和最新 review/response；
4. 确认没有新的真实结果后，再执行本文的“下一原子动作”。

本文只是接续导航，优先级低于 `AGENTS.md`、源码、配置、测试和实验产物。

## 仓库状态

- 规范主目录名：`AutoResearch`。历史上曾出现 `AutoReasearch` 拼写，不能据此创建第二个仓库。
- 唯一 Git 主仓库：最外层 `AutoResearch`；`tools/` 只是遵循根 `AGENTS.md` 的通用工具箱，没有独立 Git 属性或研究目标。
- 接续快照所在分支：`chore/integrate-tools-and-paper1`。
- 接续前基线提交：`9d48060162bce151798e7361220afe02fd78905e`（`research: harden OOF stacking and review controls`）。
- 远程：`git@github.com:Grain-Wang/AutoResearch.git`。
- 远程默认 `main` 仍较旧；在本分支合并前，另一台机器必须显式 checkout 本分支。
- `paper1/responce_from_reviewer/` 的目录名虽有历史拼写问题，但已被现有链接使用，本任务不顺带改名。

## 用户的持续目标

在有限时间和算力内，依据根 `AGENTS.md` 自主推进一篇以算法创新为核心、具有弱 CCF-B 或强 CCF-C 竞争力的完整论文。用户主要负责最终方向确认与论文责任，不应被要求长期参与标注、调参或手工评测。

## paper1 当前结论

- 原主线 **CoVoL-Depth** 已由预注册 004-A 返回 `STOP_H_SENSITIVITY`，不再是活跃 Research Opportunity，也不是 Paper Candidate。
- `Claim-F` 与 `Claim-M` 均为 `STOPPED_BY_H_SENSITIVITY_CONTROL`；不得继续 D0/D1、OOF、router、killer baseline 或第二数据集恢复。
- 负结果支持 released TR2M 对表面文本改写的一般敏感性，但不支持局部语义冲突特异效应或 fallback/router 价值。
- 备用想法 Q-GeoRoute 仍为 `PARKED`；CoVoL Gate-0 虽已否定，但尚未完成新范围、近期近邻和 Research Opportunity Gate，不能自动切换。

主阅读入口：

- `paper1/README.md`
- `paper1/ideas/01_counterfactual_value_of_language_depth.md`
- `paper1/steps/README.md`
- `paper1/responce_from_reviewer/review_round8.md`
- `paper1/responce_from_reviewer/response_round8.md`

## 已完成的可执行基础

- image/scene/sequence/RGB 内容哈希级 split 与 official-test 泄漏审计；
- scene-group OOF expert stacking plan/cache 审计，禁止 router 使用 expert 训练内预测；
- router feature denylist 与逐列 provenance，禁止干预元数据泄漏；
- paired scene/drive cluster bootstrap，replicate 内重算 clean denominator、retention、CVaR、Pareto 与 hypervolume；
- 候选无关的固定 hypervolume reference；
- 同构 D0/D1、image-only twins、shuffled-caption negative control、outer-5/inner-4 cross-fit 的协议；
- Claim-F 的 `B-direct`、`C-direct`、`C-permuted` 正交对照；
- 四类 direct deferral baseline、四类 robust expert killer 和 artifact/grounding control 的合同。

截至基线提交，`paper1` 共 22 个单元测试通过，Ruff、Black 和 Git diff 检查通过。这些检查只验证协议实现，不是科学证据。

## 因停止门禁而有意未执行

- PyTorch 同构 `D0/D1`、OOF/final checkpoints 和真实 expert cache；
- image-only twins 与 shuffled-caption control 结果；
- outer/inner cross-fit router、AUROC 和 Claim-F controls；
- direct baselines、robust expert baselines、artifact/grounding controls；
- official crop/valid-depth adapters、真实 cluster CI、latency 和结果表。

## 下一原子动作

重新检索最新近邻并形成至多五个通过 Research Opportunity Gate 的候选：每个候选必须有可复现的算法缺陷、未被最近邻覆盖的非等价算法路径和低成本可证伪 probe。选择新主线前更新 `001_primary_scope_lock.md`；不得以事后修改 CoVoL 对照或阈值恢复旧链路。

## 环境与验证

首选环境为 Python 3.12 的 Conda 环境 `auto_research`；依赖必须写入声明文件，不能只存在于某台机器。基础检查从仓库根目录执行：

```powershell
conda run -n auto_research python -m ruff check paper1
conda run -n auto_research python -m black --check paper1
conda run -n auto_research python -m pytest paper1/tests -q --basetemp .local-deps/pytest-paper1
```

原始 PDF、环境目录和缓存不提交。`sshconfig.md` 只保存在本地并已忽略；A800 的真实主机、账号与认证材料需要私密分发，仓库只保留 `tools/docs/REMOTE_EXECUTION.md` 中的通用流程。

## 建议接续提示

```text
遵循 AGENTS.md 和 .codex/handoff/CURRENT.md，先核对 Git、paper1 状态以及最新 review/response。当前唯一允许推进的研究动作是 NYUv2/KITTI source adapters 与 annotation coverage/power gate；不得把现有单元测试写成科学结果，也不得提前启动大规模 GPU 训练。
```
