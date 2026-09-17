# Direct Nearest Neighbors

## Selection rule

These 22 papers directly overlap at least two of the following: LLM/Transformer architecture search, pretrained-model reuse, post-training modification, heterogeneous layer/operator selection, and real-hardware constraints. The ordering is thematic and chronological, **not** a candidate ranking. Structured-pruning papers outside this set remain novelty-relevant and are catalogued in `literature_table.md`.

## At-a-glance map

| Cluster | Direct neighbors | Primary overlap |
|---|---|---|
| Transformer NAS precursors | HAT, AdaBERT, NAS-BERT, AutoTinyBERT, AutoDistil, FlexiBERT | Supernets, heterogeneous layers, distillation, latency/size constraints |
| Layerwise assembly precursor | LANA | Local distillation plus constrained global operator selection |
| Pretrained LLM search/compression | Search for Efficient LLMs, Minitron, MultiPruner, EfficientLLM, ELM, ZeroLM, TraceNAS | Inherited weights, structured search, low-cost proxy/recovery |
| Large-scale post-training/hybrid design | Puzzle, Nemotron-H, Minitron-SSM, Jet-Nemotron | Blockwise search, pruning/distillation, attention–SSM conversion/placement |
| 2026 deployment-conditioned work | Distill-then-Replace, MobileLLM-Flash, LLMForge, joint architecture–quantization NAS | Replacement placement, hardware-in-loop search, surrogate/DSE, quantized search |

## Detailed neighbors

### D01. HAT: Hardware-Aware Transformers for Efficient Natural Language Processing

- **Citation/source:** Hanrui Wang et al., ACL 2020. [ACL Anthology](https://aclanthology.org/2020.acl-main.686/) · [arXiv:2005.14187](https://arxiv.org/abs/2005.14187) · [code](https://github.com/mit-han-lab/hardware-aware-transformers)
- **Why direct:** searches heterogeneous Transformer layer dimensions and arbitrary encoder–decoder attention under measured latency on CPU, GPU, and IoT hardware.
- **Method:** train a weight-sharing SuperTransformer, then use evolutionary search to select device-specialized SubTransformers.
- **Difference from later LLM work:** encoder–decoder machine translation rather than pretrained causal-decoder surgery; no post-training attention/SSM conversion.
- **Novelty threat:** establishes that heterogeneous Transformer NAS and real-device latency constraints are old, strong precedents.
- **Limitation / Potential Gap Signal:** the supernet must be trained for the search space and does not model LLM prefill/decode or KV-cache behavior.

### D02. AdaBERT: Task-Adaptive BERT Compression with Differentiable NAS

- **Citation/source:** Daoyuan Chen et al., IJCAI 2020. [Proceedings](https://www.ijcai.org/Proceedings/2020/341) · [arXiv:2001.04246](https://arxiv.org/abs/2001.04246)
- **Why direct:** casts pretrained-language-model compression as differentiable architecture search with task-oriented distillation and an efficiency loss.
- **Method:** differentiably selects a task-specific compact student architecture.
- **Difference:** small BERT-style encoders and task-specific students; not a large decoder-only post-training search.
- **Novelty threat:** prior art for jointly using distillation and efficiency-aware differentiable search in pretrained NLP models.
- **Limitation / Potential Gap Signal:** task-specific search and retraining do not address reusable LLM architecture adaptation or measured serving costs.

### D03. NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with NAS

- **Citation/source:** Jin Xu et al., KDD 2021. [arXiv:2105.14444](https://arxiv.org/abs/2105.14444)
- **Why direct:** searches multiple size/latency-adaptive pretrained Transformer subnets while remaining downstream-task agnostic.
- **Method:** blockwise search-space pruning and performance approximation make a self-supervised supernet tractable.
- **Difference:** BERT encoder compression with a trained supernet; operators and deployment metrics are narrower than current heterogeneous LLM search.
- **Novelty threat:** directly anticipates blockwise task-agnostic Transformer NAS and multi-budget subnet extraction.
- **Limitation / Potential Gap Signal:** its proxy and latency evidence do not cover autoregressive decode, KV cache, or SSM/linear-attention blocks.

### D04. LANA: Latency Aware Network Acceleration

- **Citation/source:** Pavlo Molchanov et al., 2021; formal venue `UNVERIFIED`. [arXiv:2107.10624](https://arxiv.org/abs/2107.10624)
- **Why direct:** introduces layerwise teacher-feature distillation of alternative operations followed by constrained integer-linear selection under latency budgets.
- **Method:** independently trains replacements at each layer, then solves a large combinatorial architecture with ILP.
- **Difference:** demonstrated on vision networks, not Transformer/LLM residual streams.
- **Novelty threat:** foundational precedent for local distillation scores plus globally constrained layer/operator assembly—the central mechanism later used by Puzzle.
- **Limitation / Potential Gap Signal:** additive local accuracy/cost assumptions can miss cross-layer compatibility. The paper says code would be shared; a code URL was not verified.

### D05. AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient Pre-trained Language Models

- **Citation/source:** Yichun Yin et al., ACL-IJCNLP 2021. [ACL Anthology](https://aclanthology.org/2021.acl-long.400/) · [arXiv:2107.13686](https://arxiv.org/abs/2107.13686) · [code](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/AutoTinyBERT)
- **Why direct:** one-shot searches depth, dimensions, and heads for small pretrained language models under latency constraints.
- **Method:** weight-sharing training plus architecture-hyperparameter search and a reuse-oriented development procedure.
- **Difference:** tiny BERT encoders rather than post-training decoder adaptation.
- **Novelty threat:** covers latency-constrained one-shot Transformer hyperparameter search.
- **Limitation / Potential Gap Signal:** search cost still includes purpose-built one-shot training; operator heterogeneity and real serving regimes are limited.

### D06. AutoDistil: Few-shot Task-agnostic NAS for Distilling Large Language Models

- **Citation/source:** Dongkuan Xu et al., 2022; formal venue `UNVERIFIED` from the arXiv record. [arXiv:2201.12507](https://arxiv.org/abs/2201.12507) · [project/code link reported by authors](https://aka.ms/autodistil)
- **Why direct:** searches distilled Transformer students without retraining every candidate and explicitly targets variable budgets.
- **Method:** partitions a huge space into compact subspaces, trains one SuperLM per subspace with self-attention distillation, then performs lightweight search.
- **Difference:** the term “large language models” refers to the teacher context of the period; experiments are BERT-scale encoders, not modern generative LLM serving.
- **Novelty threat:** strong precedent for task-agnostic distillation-aware supernet search and subspace partitioning.
- **Limitation / Potential Gap Signal:** no heterogeneous efficient sequence operators or phase-specific hardware model.

### D07. FlexiBERT: Are Current Transformer Architectures too Homogeneous and Rigid?

- **Citation/source:** Shikhar Tuli et al., 2022 preprint; formal publication `UNVERIFIED`. [arXiv:2205.11656](https://arxiv.org/abs/2205.11656)
- **Why direct:** explicitly searches heterogeneous Transformer layers with diverse operations and dimensions.
- **Method:** graph-similarity architecture embedding plus the BOSHNAS Bayesian/second-order neural surrogate.
- **Difference:** candidate models are trained/evaluated in a BERT benchmark rather than inherited from one large pretrained causal model.
- **Novelty threat:** covers heterogeneous Transformer spaces, graph-aware predictors, and interaction-sensitive surrogate modeling.
- **Limitation / Potential Gap Signal:** transfer of its surrogate to large pretrained decoder surgery and real hardware is unverified.

### D08. Search for Efficient Large Language Models

- **Citation/source:** Xuan Shen et al., NeurIPS 2024. [arXiv:2409.17372](https://arxiv.org/abs/2409.17372) · [code](https://github.com/shawnricecake/search-llm)
- **Why direct:** explicitly performs training-free architecture search over pretrained LLM subnets, then repairs inherited weights with calibration data.
- **Method:** structured subnet selection plus weight reformation using omitted weights.
- **Difference:** primarily deletion/subnet compression; it does not search a rich set of replacement sequence operators or directly condition on hardware workloads.
- **Novelty threat:** any claim of “training-free LLM architecture search from a pretrained checkpoint” is already covered.
- **Limitation / Potential Gap Signal:** inherited-subnet quality and standard resource measures do not fully expose operator compatibility or phase-specific runtime.

### D09. Compact Language Models via Pruning and Knowledge Distillation (Minitron)

- **Citation/source:** Saurav Muralidharan et al., 2024; formal venue `UNVERIFIED`. [arXiv:2407.14679](https://arxiv.org/abs/2407.14679)
- **Why direct:** derives a family of compact LLM architectures from one pretrained model through depth, width, head, embedding, and MLP pruning plus distillation recovery.
- **Method:** empirical architecture enumeration/search across pruning axes followed by lightweight retraining/KD.
- **Difference:** the space deletes or shrinks Transformer components; it does not introduce alternative sequence operators.
- **Novelty threat:** architecture derivation by structured pruning is mathematically close to subnet NAS even when marketed as compression.
- **Limitation / Potential Gap Signal:** pruning-axis combinations are explored with practical heuristics rather than a general interaction-aware predictor.

### D10. Puzzle: Distillation-Based NAS for Inference-Optimized LLMs

- **Citation/source:** Akhiad Bercovich et al., 2024; formal venue `UNVERIFIED`. [arXiv:2411.19146](https://arxiv.org/abs/2411.19146)
- **Why direct:** hardware-aware post-training NAS at tens-of-billions scale, with block replacements selected to meet inference constraints.
- **Method:** parallel blockwise local distillation and mixed-integer programming; derived models are recovered/aligned after assembly.
- **Difference:** focuses on teacher-block alternatives and throughput/memory-constrained assembly; search-space details are tied to the parent family.
- **Novelty threat:** strongest precedent for blockwise-distilled, inference-constrained architecture assembly of a pretrained LLM.
- **Limitation / Potential Gap Signal:** the assembly objective inherits the separability assumptions of local block scores and additive constraints.

### D11. MultiPruner: Balanced Structure Removal in Foundation Models

- **Citation/source:** J. Pablo Muñoz, Jinjie Yuan, and Nilesh Jain, 2025; formal venue `UNVERIFIED`. [arXiv:2501.09949](https://arxiv.org/abs/2501.09949) · [code](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning)
- **Why direct:** training-free iterative search removes residual blocks, MLP channels, and attention heads from pretrained foundation models.
- **Method:** sequential multidimensional pruning restores balance after block removal.
- **Difference:** deletion-only and quality-focused rather than replacement-operator and hardware-workload conditioned.
- **Novelty threat:** a strong low-cost baseline for any claimed non-uniform LLM depth/width/head search.
- **Limitation / Potential Gap Signal:** sequential dimension ordering may miss joint interactions and cannot add an efficient operator absent from the parent.

### D12. EfficientLLM: Scalable Pruning-Aware Pretraining for Architecture-Agnostic Edge Language Models

- **Citation/source:** Xingrun Xing et al., 2025; formal venue `UNVERIFIED`. [arXiv:2502.06663](https://arxiv.org/abs/2502.06663) · [code](https://github.com/Xingrun-Xing2/EfficientLLM)
- **Why direct:** treats saliency-driven structured pruning during pretraining as automated architecture design for edge LMs.
- **Method:** continuously optimizes minimal parameter groups while scaling pruning-aware pretraining.
- **Difference:** changes pretraining itself rather than adapting a fixed finished model with a low-cost post-training procedure.
- **Novelty threat:** covers “architecture-agnostic” saliency design and makes pure pruning-based architecture discovery highly competitive.
- **Limitation / Potential Gap Signal:** substantial pretraining remains required and real device/workload metrics are not the central search signal.

### D13. ZeroLM: Data-Free Transformer Architecture Search for Language Models

- **Citation/source:** Zhen-Song Chen et al., 2025; formal venue and code `UNVERIFIED`. [arXiv:2503.18646](https://arxiv.org/abs/2503.18646)
- **Why direct:** develops a Transformer-specific zero-cost proxy and validates ranking on the heterogeneous FlexiBERT space.
- **Method:** decomposes Transformer submodules and combines efficient weight-statistics capacity scores.
- **Difference:** data-free proxy evaluation on benchmark-scale spaces, not post-training LLM operator replacement or real-hardware optimization.
- **Novelty threat:** directly covers training-free architecture quality scoring specialized to Transformers.
- **Limitation / Potential Gap Signal:** proxy validity for pretrained causal LLMs and cross-layer heterogeneous operator interactions is unverified.

### D14. Nemotron-H: Accurate and Efficient Hybrid Mamba–Transformer Models

- **Citation/source:** NVIDIA et al., 2025; formal venue `UNVERIFIED`. [arXiv:2504.03624](https://arxiv.org/abs/2504.03624)
- **Why direct:** replaces most attention layers with Mamba for constant per-token compute/memory and introduces MiniPuzzle compression for the resulting hybrid.
- **Method:** full hybrid pretraining plus pruning/distillation compression of the trained 56B model.
- **Difference:** primary hybrid topology is designed/trained as a family rather than discovered solely by post-training search.
- **Novelty threat:** demonstrates large-scale hybrid placement plus post-training architecture compression and real inference gains.
- **Limitation / Potential Gap Signal:** the public paper does not isolate a general rule explaining placement transfer across task, context, and hardware.

### D15. Minitron-SSM: Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning

- **Citation/source:** Ali Taghibakhshi et al., 2025; formal venue `UNVERIFIED`. [arXiv:2504.11409](https://arxiv.org/abs/2504.11409)
- **Why direct:** adapts post-training structured architecture reduction to hybrid attention–SSM models.
- **Method:** group-aware SSM pruning combined with FFN, embedding, and layer pruning, followed by distillation recovery.
- **Difference:** removal/compression of an existing hybrid rather than search for a new per-layer operator topology.
- **Novelty threat:** any SSM-aware structured pruning claim must distinguish itself from group-aware state integrity preservation.
- **Limitation / Potential Gap Signal:** the architecture space is inherited from Nemotron-H and emphasizes compression, not workload-conditioned replacement.

### D16. Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

- **Citation/source:** Yuxian Gu et al., NeurIPS 2025. [arXiv:2508.15884](https://arxiv.org/abs/2508.15884)
- **Why direct:** the closest explicit precedent for post-training search of hybrid LLM architecture and real inference performance.
- **Method:** freezes pretrained MLPs, learns full-attention placement/elimination, selects/designs linear-attention blocks, and performs hardware-aware hyperparameter search.
- **Difference from Puzzle:** changes sequence operators and attention placement rather than mainly assembling distilled size/shape alternatives.
- **Novelty threat:** covers the broad claim of converting a pretrained full-attention LLM into a hardware-aware heterogeneous hybrid through PostNAS.
- **Limitation / Potential Gap Signal:** the abstract does not establish whether the learned placement generalizes across tasks, workloads, or devices; detailed code availability is `UNVERIFIED`.

### D17. Elastic Architecture Search for Efficient Language Models

- **Citation/source:** Shang Wang, ICME 2025 per arXiv metadata. [arXiv:2510.27037](https://arxiv.org/abs/2510.27037)
- **Why direct:** searches flexible compact language models using efficient Transformer blocks and dynamic head/dimension modules.
- **Method:** elastic architecture space with block-characteristic-preserving distillation losses.
- **Difference:** reports masked and causal LM search but is not clearly a post-training conversion of a modern large backbone.
- **Novelty threat:** covers elastic operator/dimension/head search with distillation-aware discrimination.
- **Limitation / Potential Gap Signal:** real-hardware cost, pretrained inheritance details, and code are `UNVERIFIED`.

### D18. Distill-then-Replace: Efficient Task-Specific Hybrid Attention Model Construction

- **Citation/source:** Xiaojie Xia et al., 2026 preprint; formal venue/code `UNVERIFIED`. [arXiv:2601.11667](https://arxiv.org/abs/2601.11667)
- **Why direct:** converts a pretrained full-attention model into a task-specific hybrid by local distillation and layer replacement.
- **Method:** distill linear-attention counterparts, then greedily replace attention blocks while monitoring validation performance.
- **Difference:** authors explicitly avoid costly NAS; search is a single greedy task-specific path without a hardware objective.
- **Novelty threat:** establishes local distill-then-place conversion even when it is not labeled NAS.
- **Limitation / Potential Gap Signal:** greedy validation may miss interacting replacements and deployment-specific optima.

### D19. TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation

- **Citation/source:** Prajna G. Malettira et al., 2026 preprint; formal venue/code `UNVERIFIED`. [arXiv:2602.02891](https://arxiv.org/abs/2602.02891)
- **Why direct:** searches non-uniform LLM depth and width with a global zero-shot proxy that explicitly targets structural dependencies.
- **Method:** scale-invariant gradient-trace/loss-landscape alignment; the paper reports single-GPU discovery in 8.5 hours.
- **Difference:** pruning-only space and training-aware recovery comparison rather than heterogeneous operator replacement or hardware latency.
- **Novelty threat:** weakens claims that all existing low-cost scores are purely local or interaction-blind.
- **Limitation / Potential Gap Signal:** its global proxy is not yet demonstrated across mixed attention/SSM/linear-attention primitives or workload metrics.

### D20. MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale Deployment

- **Citation/source:** Hanxian Huang et al., ACL Industry Track 2026 per arXiv metadata. [arXiv:2603.15954](https://arxiv.org/abs/2603.15954)
- **Why direct:** hardware-in-the-loop architecture search jointly selects layers, dimensions, and attention skipping for pretrained-backbone-derived mobile LMs.
- **Method:** learn a mobile latency model, inherit candidate weights from a backbone, then search quality–latency Pareto fronts with limited continued pretraining.
- **Difference:** deliberately avoids specialized attention kernels/operators and targets standard mobile runtimes.
- **Novelty threat:** covers context-aware attention skipping, inherited-weight candidate evaluation, and separate mobile prefill/decode evidence.
- **Limitation / Potential Gap Signal:** transfer beyond mobile CPUs and beyond deletion/skipping spaces is not established; code URL is `UNVERIFIED`.

### D21. LLMForge: Multi-Backend Hardware-Aware NAS with Infinite-Head Attention

- **Citation/source:** Xinting Jiang et al., 2026 preprint; formal venue/code `UNVERIFIED`. [arXiv:2605.17653](https://arxiv.org/abs/2605.17653)
- **Why direct:** jointly searches language-model architecture and multi-backend hardware costs using a learned architecture surrogate.
- **Method:** Infinite-Head Attention expands per-layer Q/K/V/head choices; Forge-Former predicts quality; NSGA-II searches GPU, systolic, and ring-accelerator objectives including TTFT, TPOT, and energy.
- **Difference:** searched sub-billion models are retrained under matched recipes rather than converted from one large finished LLM.
- **Novelty threat:** directly covers hardware-conditioned heterogeneous attention search, learned quality ranking, and multi-objective Pareto design.
- **Limitation / Potential Gap Signal:** reliance on a trained surrogate and specified hardware backends leaves out pretrained compatibility and out-of-distribution workload transfer.

### D22. LLM Compression with Jointly Optimizing Architectural and Quantization Choices

- **Citation/source:** Hoang-Loc La et al., 2026 preprint; formal venue/code `UNVERIFIED`. [arXiv:2606.04063](https://arxiv.org/abs/2606.04063)
- **Why direct:** differentiable search jointly chooses structured architecture and mixed-precision quantization for pretrained LLM deployment.
- **Method:** continuous architecture/precision optimization with latency-aware comparison to sequential NAS-then-quantization.
- **Difference:** focuses on pruning/linear-layer precision rather than heterogeneous sequence-operator topology.
- **Novelty threat:** establishes that architecture and quantization should not be claimed as a new joint-search combination.
- **Limitation / Potential Gap Signal:** full search-space details, formal publication, exact hardware setup generality, and code remain `UNVERIFIED` from the metadata checked here.

## Nearest-neighbor boundary conclusions

- “Post-training NAS for an efficient LLM” alone is not novel after Search for Efficient LLMs, Puzzle, and Jet-Nemotron.
- “Hardware-aware heterogeneous Transformer search” alone is not novel after HAT and LLMForge.
- “Blockwise distillation plus constrained assembly” alone is not novel after LANA and Puzzle.
- “Training-free non-uniform LLM pruning/search” alone is not novel after Search for Efficient LLMs, MultiPruner, ZeroLM, and TraceNAS.
- “Convert attention layers to efficient operators and choose their placement” is directly threatened by Jet-Nemotron and Distill-then-Replace.
- These are literature boundaries only; no Research Opportunity or candidate ranking is made in this turn.
