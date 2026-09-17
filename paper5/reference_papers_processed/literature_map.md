# Literature Map: Neural Architecture Search for Efficient Large Language Models

## Scope and evidence policy

- **Turn:** TURN 1 — Literature Expansion & Literature Map only.
- **Coverage:** 79 papers: 22 direct nearest neighbors, 32 strongly related methods, and 25 foundations/adjacent methods.
- **Time emphasis:** 2019–2026, with 2025–2026 work separated because it most directly changes the novelty landscape. Six 2018 first submissions are retained only as field-defining foundations: ENAS was published at ICML 2018, and the other five became 2019 conference papers.
- **Primary evidence:** paper pages/abstracts from arXiv, OpenReview, ACL Anthology, PMLR, or official proceedings; official repositories are recorded only when directly linked by the paper metadata or paper page.
- **Interpretation rule:** a limitation below is a literature-derived observation, not a proposed research direction. Any unresolved item is explicitly labeled **Potential Gap Signal**.

The complete per-paper evidence is in `literature_table.md`; the closest 22 works are expanded in `direct_neighbors.md`; query provenance and unresolved metadata are in `search_log.md`.

## Development timeline

| Period | Main development | Representative papers |
|---|---|---|
| 2018–2020 | Weight sharing, differentiable search, direct hardware constraints, once-for-all specialization | ENAS, DARTS, ProxylessNAS, MnasNet, FBNet, ChamNet, Once-for-All, BigNAS |
| 2020–2022 | NAS moves into Transformer/BERT compression; blockwise search and task-agnostic distillation appear | HAT, AdaBERT, DynaBERT, NAS-BERT, AutoTinyBERT, LANA, AutoDistil, FlexiBERT, DARTFormer |
| 2023–2024 | LLM compression and efficient sequence operators mature; architecture search becomes feasible through inherited weights, local scores, and blockwise distillation | LLM-Pruner, Sheared LLaMA, SliceGPT, Search for Efficient LLMs, Puzzle, Minitron; Mamba, GLA, Based, Jamba, Samba, Hymba |
| 2025 | Post-training LLM NAS and hybrid compression become explicit research objects | Jet-Nemotron/PostNAS, Nemotron-H/MiniPuzzle, Minitron-SSM, ZeroLM, MultiPruner, EfficientLLM, ELM |
| 2026 | Search becomes more hardware-conditioned and interaction-aware, but current evidence is mostly recent preprints | MobileLLM-Flash, LLMForge, TraceNAS, joint architecture–quantization NAS, Distill-then-Replace |

## Taxonomy

### 1. NAS foundations

- **Core problem:** search a combinatorial architecture space without independently training every candidate.
- **Main methods:** controller/RL search, evolutionary search, continuous relaxation, weight-sharing supernets, Bayesian optimization.
- **Representative papers:** ENAS, DARTS, BANANAS, BigNAS.
- **Common assumptions:** candidate rankings from shared weights or a learned surrogate are sufficiently faithful; a mostly fixed operator macro-space is meaningful.
- **Resolved:** established reusable search abstractions and large reductions in per-candidate training cost.
- **Potential Gap Signal:** the rank fidelity of generic NAS machinery is not established for inherited, post-training LLM subnets whose layer choices interact through residual streams and autoregressive decoding.

### 2. Efficient NAS

- **Core problem:** reduce search/training expense while retaining useful ranking quality.
- **Main methods:** once-for-all progressive shrinking, single-stage supernets, zero-shot scores, compact search subspaces, performance predictors.
- **Representative papers:** Once-for-All, BigNAS, NASWOT, TE-NAS, Zen-NAS, ZiCo, AutoDistil.
- **Common assumptions:** a local or initialization-time proxy transfers to trained accuracy; subnet interference can be controlled by training schedules or subspace partitioning.
- **Resolved:** image-model NAS can run without full candidate training; Transformer search can reuse a small number of supernets.
- **Potential Gap Signal:** most zero-cost proxies were validated on vision spaces or small encoders, not on pretrained autoregressive LLMs with heterogeneous attention/SSM primitives.

### 3. Hardware-aware NAS

- **Core problem:** choose architectures under device-specific latency, energy, or memory constraints rather than FLOPs alone.
- **Main methods:** measured latency lookup tables, learned latency predictors, hardware-in-the-loop sampling, constrained evolutionary/ILP search, multi-objective Pareto search.
- **Representative papers:** MnasNet, ProxylessNAS, FBNet, ChamNet, HAT, HW-NAS-Bench, HELP, LANA, LLMForge.
- **Common assumptions:** operator costs are additive or accurately predicted; measurement conditions represent deployment; quality and cost predictors generalize across candidates.
- **Resolved:** direct latency targets produce different designs across CPUs, GPUs, and edge devices; latency predictors can amortize repeated measurements.
- **Potential Gap Signal:** additive latency models may miss fusion, occupancy, memory hierarchy, kernel launch, and cross-layer state/KV effects in heterogeneous LLMs.

### 4. Transformer NAS

- **Core problem:** automate depth, width, head, FFN, and operator choices in Transformer encoders or encoder–decoders.
- **Main methods:** heterogeneous supernets, evolutionary search, differentiable operator selection, graph-based surrogates, task-aware distillation.
- **Representative papers:** HAT, AdaBERT, NAS-BERT, AutoTinyBERT, AutoDistil, FlexiBERT, DARTFormer.
- **Common assumptions:** masked-language-model or task losses predict downstream utility; encoder findings transfer to larger causal decoders.
- **Resolved:** heterogeneous layer dimensions and attention types can outperform uniformly scaled small Transformers.
- **Potential Gap Signal:** much of this evidence predates decoder-only LLM deployment and does not jointly model prefill, token-by-token decode, and KV/state memory.

### 5. LLM NAS

- **Core problem:** search useful LLM architectures without the prohibitive cost of training many large candidates.
- **Main methods:** inherited-weight subnet search, calibration-based weight reformation, saliency-guided pruning/search, structured zero-shot proxies, elastic modules.
- **Representative papers:** Search for Efficient Large Language Models, ZeroLM, ELM, TraceNAS, EfficientLLM.
- **Common assumptions:** inherited weights or lightweight recovery preserve relative quality; pruning dimensions constitute a sufficiently expressive search space.
- **Resolved:** depth/width/head/FFN choices can be searched from pretrained checkpoints and can yield deployable smaller subnets.
- **Potential Gap Signal:** direct LLM NAS remains split between compression-only spaces and from-scratch elastic spaces; evidence for mixed sequence-operator spaces is sparse.

### 6. Post-training NAS

- **Core problem:** derive new architectures from a pretrained teacher/backbone without repeating full pretraining.
- **Main methods:** local block distillation followed by global constrained selection; frozen-module attention conversion; inherited-weight pruning plus limited continued training.
- **Representative papers:** LANA, Puzzle, Search for Efficient Large Language Models, Jet-Nemotron, MobileLLM-Flash, joint architecture–quantization NAS.
- **Common assumptions:** block-local fidelity composes globally; inherited weights make candidate quality comparable; a small recovery corpus repairs architectural surgery.
- **Resolved:** post-training architecture derivation scales to multi-billion-parameter models and can optimize measured deployment performance.
- **Potential Gap Signal:** local block scores and additive cost models do not explicitly capture cross-layer compatibility or error propagation in heterogeneous hybrids.

### 7. Training-free NAS

- **Core problem:** rank architectures with no candidate training or only calibration passes.
- **Main methods:** activation-pattern scores, Jacobian/gradient/Hessian statistics, parameter statistics, perplexity/saliency evaluation, loss-landscape alignment.
- **Representative papers:** NASWOT, TE-NAS, Zen-NAS, ZiCo, ZeroLM, Search for Efficient Large Language Models, TraceNAS.
- **Common assumptions:** a cheap scalar proxy is monotonic with downstream quality; a small calibration set is representative.
- **Resolved:** useful rank correlation is possible without full training in several bounded search spaces.
- **Potential Gap Signal:** scalar proxies often score components independently and may not preserve rankings when attention/SSM placement creates non-additive interactions.

### 8. Blockwise / layerwise NAS

- **Core problem:** make large combinatorial search tractable by evaluating interchangeable blocks locally.
- **Main methods:** blockwise local distillation, feature-map matching, per-block loss/perplexity, ILP/MIP assembly, greedy replacement.
- **Representative papers:** LANA, NAS-BERT, Puzzle, BlockPruner, MultiPruner, Distill-then-Replace.
- **Common assumptions:** candidate block utility is approximately context-independent; local teacher matching predicts end-to-end behavior.
- **Resolved:** many alternatives can be trained/evaluated in parallel and assembled under exact resource constraints.
- **Potential Gap Signal:** local equivalence to a teacher may not preserve global retrieval, long-context routing, or autoregressive stability after multiple replacements.

### 9. Hybrid Transformer–SSM architectures

- **Core problem:** retain attention's recall/expressivity while exploiting recurrent or state-space decoding efficiency.
- **Main methods:** serial interleaving, parallel attention–SSM heads, sparse attention layers among SSM layers, learned/selective attention routing.
- **Representative papers:** H3, Jamba, Zamba, Samba, Hymba, Taipan, Nemotron-H, Jet-Nemotron.
- **Common assumptions:** a small amount of attention repairs the associative-recall weaknesses of efficient recurrent operators; placement heuristics transfer across tasks and context lengths.
- **Resolved:** hybrid models can improve the quality–throughput frontier over pure attention or pure SSM designs.
- **Potential Gap Signal:** most placements are manually designed or searched under a fixed workload; task-, context-, and hardware-dependent placement is incompletely characterized.

### 10. Efficient attention / sequence operators

- **Core problem:** replace quadratic full attention or reduce its decoding state and memory traffic.
- **Main methods:** MQA/GQA/MLA, sparse/sliding attention, kernelized linear attention, retention, gated linear recurrence, SSMs, long convolutions.
- **Representative papers:** MQA, GQA, DeepSeek-V2/MLA, Longformer, BigBird, Performer, Linear Transformers, RetNet, GLA, DeltaNet, Gated DeltaNet, Mamba/Mamba-2, RWKV, Hyena, H3, Based.
- **Common assumptions:** benchmark perplexity/accuracy captures retrieval behavior; custom kernels and idealized sequence regimes are available.
- **Resolved:** decode KV/state memory can be dramatically reduced; IO-aware kernels make exact or linear attention substantially faster.
- **Potential Gap Signal:** operator rankings can reverse with batch size, context length, prompt/decode ratio, kernel maturity, and required recall pattern.

### 11. Hardware-aware LLM design

- **Core problem:** connect architecture decisions to real prefill latency, decode latency, throughput, memory capacity, and energy.
- **Main methods:** simulator/roofline models, empirical latency models, serving-system measurements, phase-specific optimization.
- **Representative papers:** LLMCompass, PagedAttention, DistServe, Splitwise, Puzzle, Jet-Nemotron, MobileLLM-Flash, LLMForge.
- **Common assumptions:** a fixed serving stack and workload distribution are representative; per-token or per-request metrics can be optimized separately.
- **Resolved:** prefill is commonly compute-intensive while decode is often bandwidth/state dominated; KV paging and phase disaggregation materially change observed system performance.
- **Why FLOPs/parameters are insufficient:** they omit KV/state footprint, memory traffic, kernel fusion and launch overhead, tensor-core shape/utilization, occupancy, paging, batching, and the different critical paths of prefill versus decode.
- **Potential Gap Signal:** architecture search rarely treats workload distribution—batch size, context length, generated length, and serving phase—as an explicit conditioning variable.

### 12. Architecture predictor / proxy

- **Core problem:** estimate accuracy or resource cost cheaply enough to search a large space.
- **Main methods:** Gaussian-process/Bayesian surrogate, graph similarity embeddings, learned Transformer surrogate, zero-cost gradients/activations, latency meta-learning.
- **Representative papers:** ChamNet, BANANAS, FlexiBERT/BOSHNAS, HELP, ZeroLM, TraceNAS, LLMForge/Forge-Former.
- **Common assumptions:** training/search samples cover the deployment space; pairwise rankings remain stable out of distribution.
- **Resolved:** learned predictors can reduce both quality evaluation and device profiling cost.
- **Potential Gap Signal:** few predictors jointly represent architecture interactions, pretrained-weight compatibility, and device/workload-specific costs.

### 13. Structured pruning / model transformation

- **Core problem:** remove layers, heads, channels, or dimensions from a pretrained model while preserving quality.
- **Main methods:** gradient/Hessian saliency, activation-weight scores, block influence/perplexity, layer collapse, evolutionary structured compression, knowledge-distillation recovery.
- **Representative papers:** SparseGPT, Wanda, LLM-Pruner, Sheared LLaMA, SliceGPT, FLAP, SLEB, ShortGPT, LaCo, BlockPruner, EvoPress, MultiPruner, Minitron, Minitron-SSM.
- **Common assumptions:** removal scores are separable; the retained subnetwork plus recovery training is the relevant architecture optimum.
- **Resolved:** post-training depth/width/head/FFN reduction is a strong baseline and is mathematically an architecture search when it selects a structured subgraph.
- **Novelty relevance:** any future claim about LLM architecture selection must compare against this literature even if a paper does not use the term NAS.
- **Potential Gap Signal:** most pruning spaces only delete structure; they rarely replace layers with qualitatively different sequence operators under real hardware constraints.

### 14. 2025–2026 latest works

- **Core problem:** turn pretrained LLMs into deployment-specific architectures while searching richer operator and quantization choices.
- **Representative papers:** MultiPruner, EfficientLLM, ZeroLM, Nemotron-H, Minitron-SSM, Jet-Nemotron, ELM, Distill-then-Replace, TraceNAS, MobileLLM-Flash, LLMForge, joint architecture–quantization NAS.
- **Emerging methods:** PostNAS with frozen MLPs; hybrid-model group-aware pruning; global trace-alignment proxy; hardware-in-the-loop mobile search; hardware-conditioned attention dimensions; joint mixed-precision and architecture search.
- **What appears covered:** basic depth/width/head pruning, attention-layer skipping/placement, hardware-aware latency search, block-local distillation, and joint pruning/quantization all now have direct precedents.
- **Potential Gap Signal:** interaction-aware global scoring, workload-conditioned search, and principled transfer of local block scores across heterogeneous operators remain less densely evidenced than deletion-based compression.
- **Verification caveat:** several 2026 papers are arXiv records or future-dated conference comments as of the search date; their formal publication and code availability remain `UNVERIFIED` where noted.

## Cross-cutting comparison

| Family | Candidate quality signal | Architecture freedom | Hardware signal | Typical adaptation cost | Principal limitation line |
|---|---|---|---|---|---|
| Supernet NAS | Shared-weight validation loss | Depth/width/operators | Optional LUT/predictor | High supernet training, cheap search | Weight-sharing rank disorder |
| Local-distillation assembly | Per-block teacher feature loss | Replacement block per layer | Exact additive constraint or measured table | Parallel block training plus recovery | Cross-layer interactions approximated as additive |
| Training-free subnet search | Calibration loss/saliency/gradient score | Mostly deletion and width/depth | Usually FLOPs/params; sometimes latency | Very low search, optional recovery | Proxy fidelity and calibration dependence |
| PostNAS operator conversion | Inherited weights plus local/global adaptation | Attention placement/type and hardware parameters | Prefill/decode throughput or latency | Moderate post-training, no full restart | Limited evidence on transfer across workloads/tasks |
| Hybrid manual design | Full pretraining evaluation | Attention/SSM topology | Runtime measured after design | Full training | Placement is not systematically optimized |
| Structured pruning | Importance score plus recovery | Delete blocks/heads/channels/dimensions | Often params/FLOPs; sometimes runtime | Calibration to modest retraining | Cannot introduce a better operator than the parent |
| Learned surrogate search | Predicted quality/rank | Potentially broad | Separate or joint cost predictor | Dataset/profiling upfront | Out-of-distribution generalization |

## Hardware metric map

| Metric | What architecture choices affect it | Why FLOPs/parameters can mislead | Representative evidence |
|---|---|---|---|
| Prefill latency / TTFT | attention complexity, tensor shapes, kernel fusion, prompt length | two equal-FLOP designs can have different IO and tensor-core utilization | FlashAttention, LLMCompass, Jet-Nemotron, MobileLLM-Flash, LLMForge |
| Decode latency / TPOT | KV/state size, batch size, memory traffic, recurrent state update | decoding can be memory-bound and dominated by cache reads rather than arithmetic | MQA, GQA, PagedAttention, DistServe, Splitwise, Nemotron-H |
| Throughput | batching, capacity, memory footprint, parallelism, serving policy | a smaller model may underutilize hardware or enable a much larger batch | Puzzle, PagedAttention, Jet-Nemotron |
| KV/state memory | number of KV heads, retained attention layers, context length, state dimension | parameter count excludes runtime cache that grows with sequence/batch | MQA, GQA, MLA, Jamba, Nemotron-H |
| Energy | memory movement, kernel efficiency, device substrate | FLOPs omit data movement and substrate-specific costs | MnasNet, LLMForge |
| Occupancy / tensor-core use | head and hidden dimensions, block shapes, quantization | irregular dimensions can lower utilization despite fewer operations | LLMCompass, MobileLLM-Flash, LLMForge |
| Kernel-launch overhead | number and composition of small operators | additive FLOP models do not price launches or fusion barriers | FlashAttention/2, GLA, Mamba-2 |

## Congestion assessment (descriptive, not ranking)

The densest prior-art regions are:

1. structured deletion of pretrained LLM depth/width/heads/FFN channels;
2. weight-sharing NAS and zero-cost proxies in conventional vision spaces;
3. handcrafted or family-level hybrid attention–SSM architectures;
4. device-specific latency-aware NAS using LUTs or learned predictors;
5. blockwise local distillation followed by constrained assembly.

Less densely evidenced observations are retained only as **Potential Gap Signals**:

- **Potential Gap Signal:** global, interaction-aware scoring for multiple heterogeneous layer replacements in a pretrained LLM;
- **Potential Gap Signal:** search conditioned jointly on hardware and workload regime (prefill/decode mix, batch size, context and generation length);
- **Potential Gap Signal:** proxy transfer across full attention, sparse attention, linear attention, and SSM primitives under inherited weights;
- **Potential Gap Signal:** systematic evidence explaining when full-attention placement should change with task, context length, or hardware;
- **Potential Gap Signal:** resource predictors that jointly model latency, KV/state memory, and cross-layer kernel effects instead of optimizing FLOPs or an additive per-block table alone.

These are map annotations only. They have not been converted into Research Opportunities, ranked, or selected.
