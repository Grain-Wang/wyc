# Core Literature Library: Low-Cost Post-Training NAS for Pretrained LLMs

This directory contains a deliberately small, source-verified seed corpus for pure-software, low-cost post-training architecture search on pretrained language models. The core set has 20 papers: 17 formally published papers and 3 clearly labeled preprints. Nine papers are direct nearest neighbors. Hardware-aware work is included only when its reusable algorithmic mechanism directly informs post-training model transformation; device measurement and deployment specialization are out of scope.

The source PDFs are retained locally in `../reference_papers_origin/` and ignored by Git. The corresponding full-text Markdown conversions in this directory are the tracked research artifacts. `GOOD` means the required sections are readable after conversion. `FAIR` means the paper is still usable, but the local PDF remains authoritative for some equations, figures, or tables.

| ID | Paper | Year | Venue | Category | PDF | Markdown | Code | Verified |
|---:|---|---:|---|---|---|---|---|---|
| 01 | Once-for-All: Train One Network and Specialize it for Efficient Deployment | 2020 | ICLR | Foundation / Training-Free NAS | [PDF](../reference_papers_origin/2020_Once_For_All.pdf) | [Markdown](2020_Once_For_All.md) | [Code](https://github.com/mit-han-lab/once-for-all) | Formal; GOOD |
| 02 | Neural Architecture Search without Training | 2021 | ICML | Training-Free NAS / Architecture Proxy | [PDF](../reference_papers_origin/2021_NASWOT.pdf) | [Markdown](2021_NASWOT.md) | [Code](https://github.com/BayesWatch/nas-without-training) | Formal; GOOD |
| 03 | NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search | 2021 | KDD | Direct Neighbor / Post-Training NAS | [PDF](../reference_papers_origin/2021_NAS_BERT.pdf) | [Markdown](2021_NAS_BERT.md) | UNVERIFIED | Formal; GOOD |
| 04 | Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective | 2021 | ICLR | Training-Free NAS / Architecture Proxy | [PDF](../reference_papers_origin/2021_TE_NAS.pdf) | [Markdown](2021_TE_NAS.md) | [Code](https://github.com/VITA-Group/TENAS) | Formal; GOOD |
| 05 | Few-shot Task-agnostic Neural Architecture Search for Distilling Large Language Models | 2022 | NeurIPS | Direct Neighbor / Blockwise Distillation | [PDF](../reference_papers_origin/2022_AutoDistil.pdf) | [Markdown](2022_AutoDistil.md) | [Code](https://github.com/microsoft/autodistil) | Formal; GOOD |
| 06 | LANA: Latency Aware Network Acceleration | 2022 | ECCV | Post-Training NAS / Blockwise Distillation | [PDF](../reference_papers_origin/2022_LANA.pdf) | [Markdown](2022_LANA.md) | UNVERIFIED | Formal; FAIR |
| 07 | LLM-Pruner: On the Structural Pruning of Large Language Models | 2023 | NeurIPS | Structured Compression / Architecture Proxy | [PDF](../reference_papers_origin/2023_LLM_Pruner.pdf) | [Markdown](2023_LLM_Pruner.md) | [Code](https://github.com/horseee/LLM-Pruner) | Formal; FAIR |
| 08 | ZiCo: Zero-shot NAS via Inverse Coefficient of Variation on Gradients | 2023 | ICLR | Training-Free NAS / Architecture Proxy | [PDF](../reference_papers_origin/2023_ZiCo.pdf) | [Markdown](2023_ZiCo.md) | UNVERIFIED | Formal; FAIR |
| 09 | Compact Language Models via Pruning and Knowledge Distillation | 2024 | NeurIPS | Direct Neighbor / Structured Compression | [PDF](../reference_papers_origin/2024_Minitron.pdf) | [Markdown](2024_Minitron.md) | [Code](https://github.com/NVlabs/Minitron) | Formal; GOOD |
| 10 | SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks | 2024 | ICML | Structured Compression / Architecture Interaction | [PDF](../reference_papers_origin/2024_SLEB.pdf) | [Markdown](2024_SLEB.md) | [Code](https://github.com/jiwonsong-dev/SLEB) | Formal; FAIR |
| 11 | Search for Efficient Large Language Models | 2024 | NeurIPS | Direct Neighbor / Training-Free NAS | [PDF](../reference_papers_origin/2024_Search_Efficient_LLM.pdf) | [Markdown](2024_Search_Efficient_LLM.md) | [Code](https://github.com/shawnricecake/search-llm) | Formal; FAIR; alternate-parser recovery included |
| 12 | Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning | 2024 | ICLR | Structured Compression / Post-Training Adaptation | [PDF](../reference_papers_origin/2024_Sheared_LLaMA.pdf) | [Markdown](2024_Sheared_LLaMA.md) | [Code](https://github.com/princeton-nlp/LLM-Shearing) | Formal; GOOD |
| 13 | SliceGPT: Compress Large Language Models by Deleting Rows and Columns | 2024 | ICLR | Structured Compression / Architecture Adaptation | [PDF](../reference_papers_origin/2024_SliceGPT.pdf) | [Markdown](2024_SliceGPT.md) | [Code](https://github.com/microsoft/TransformerCompression) | Formal; GOOD |
| 14 | BlockPruner: Fine-grained Pruning for Large Language Models | 2025 | Findings of ACL | Structured Compression / Training-Free NAS | [PDF](../reference_papers_origin/2025_BlockPruner.pdf) | [Markdown](2025_BlockPruner.md) | [Code](https://github.com/MrGGLS/BlockPruner) | Formal; GOOD |
| 15 | Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search | 2025 | NeurIPS | Direct Neighbor / Post-Training NAS | [PDF](../reference_papers_origin/2025_Jet_Nemotron.pdf) | [Markdown](2025_Jet_Nemotron.md) | UNVERIFIED | Formal; GOOD |
| 16 | MultiPruner: Balanced Structure Removal in Foundation Models | 2025 | arXiv | Structured Compression / Training-Free NAS | [PDF](../reference_papers_origin/2025_MultiPruner.pdf) | [Markdown](2025_MultiPruner.md) | [Code](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning) | PREPRINT; GOOD |
| 17 | Puzzle: Distillation-Based NAS for Inference-Optimized LLMs | 2025 | ICML | Direct Neighbor / Blockwise Distillation | [PDF](../reference_papers_origin/2025_Puzzle.pdf) | [Markdown](2025_Puzzle.md) | [Code](https://github.com/NVlabs/puzzle) | Formal; FAIR |
| 18 | Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization | 2025 | NeurIPS | Direct Neighbor / Architecture Interaction | [PDF](../reference_papers_origin/2025_Tyr_The_Pruner.pdf) | [Markdown](2025_Tyr_The_Pruner.md) | UNVERIFIED | Formal; FAIR |
| 19 | Distill-then-Replace: Efficient Task-Specific Hybrid Attention Model Construction | 2026 | arXiv | Direct Neighbor / Blockwise Distillation | [PDF](../reference_papers_origin/2026_Distill_Then_Replace.pdf) | [Markdown](2026_Distill_Then_Replace.md) | UNVERIFIED | PREPRINT; GOOD |
| 20 | TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation | 2026 | arXiv | Direct Neighbor / Architecture Interaction | [PDF](../reference_papers_origin/2026_TraceNAS.pdf) | [Markdown](2026_TraceNAS.md) | UNVERIFIED | PREPRINT; FAIR |

## Why each paper matters

1. **Once-for-All** establishes elastic supernet training and cheap post hoc subnet specialization, a central cost-amortization baseline for low-cost search.
2. **NASWOT** is a canonical zero-cost architecture-ranking method and clarifies what a genuinely training-free proxy must provide.
3. **NAS-BERT** provides a blockwise, task-agnostic Transformer supernet and performance approximation for adaptive-size pretrained-model compression.
4. **TE-NAS** combines trainability and expressivity signals into a low-cost search rule and provides a theoretically motivated proxy baseline.
5. **AutoDistil** partitions the Transformer search space into compact SuperLMs and uses task-agnostic self-attention distillation for lightweight student search.
6. **LANA** contributes a teacher-guided local replacement library and fast constrained global assembly; only these software mechanisms, not its latency objective, are relevant here.
7. **LLM-Pruner** defines coupled structural groups, gradient-based importance, and low-cost LoRA recovery for pretrained LLM architecture reduction.
8. **ZiCo** supplies a strong gradient-statistics zero-shot proxy suitable as a baseline for calibration-based architecture scoring.
9. **Minitron** systematically varies depth, width, attention, and MLP structure and recovers compact LLMs with a small fraction of pretraining data.
10. **SLEB** iteratively removes and re-evaluates blocks, exposing how layer importance changes with the surviving inter-layer context.
11. **Search for Efficient Large Language Models** is the central precedent for training-free pretrained-LLM subnet search plus calibration-based weight reformation.
12. **Sheared LLaMA** makes target depth, head, hidden, and FFN dimensions explicit architecture variables and studies low-compute recovery from a pretrained LLM.
13. **SliceGPT** supplies a mathematically grounded dense-width transformation and a strong post-training baseline without irregular sparsity.
14. **BlockPruner** turns MHA and MLP residual blocks into a fine-grained training-free search space scored by perplexity and pruned iteratively.
15. **Jet-Nemotron** contributes PostNAS from a pretrained full-attention model with learned layer/operator placement; its hardware tuning is outside this project's scope.
16. **MultiPruner** combines residual-block, MLP-channel, and attention-head removal in a low-cost iterative pipeline and is a strong multi-axis subnet baseline.
17. **Puzzle** is the strongest direct precedent for constructing a locally distilled block library and assembling a pretrained LLM by constrained optimization.
18. **Týr-the-Pruner** directly studies non-uniform global LLM sparsity allocation, inter-structure dependence, efficient supernet construction, and coarse-to-fine search.
19. **Distill-then-Replace** is a direct non-NAS mathematical neighbor that locally distills replacement blocks and greedily composes a pretrained hybrid architecture.
20. **TraceNAS** proposes a global zero-shot loss-landscape alignment score for joint depth-width LLM search and directly challenges purely local importance proxies.

## Conversion quality notes

Every PDF has a one-to-one full-text Markdown conversion containing the extracted abstract, method material, experimental sections, and references. Automated checks also found formula and search-space/proxy signals in the corpus. The following conversions are usable but rated `FAIR`:

- `2022_LANA.md`: some diagram and mathematical glyph layout is imperfect.
- `2023_LLM_Pruner.md`: several compact equations are fragmented by the source font encoding.
- `2023_ZiCo.md`: some displayed equations lose ideal LaTeX layout.
- `2024_SLEB.md`: several result tables are flattened into readable text rather than Markdown tables.
- `2024_Search_Efficient_LLM.md`: the primary parser loses custom-font math glyphs on core method pages; a page-aligned second-parser extraction is appended for those pages.
- `2025_Puzzle.md`: some distillation equations are fragmented, while surrounding definitions and method prose remain readable.
- `2025_Tyr_The_Pruner.md`: some optimization notation is fragmented, while the search procedure and results remain readable.
- `2026_TraceNAS.md`: some dense mathematical notation has imperfect spacing and line structure.

For these entries, consult the ignored local PDF whenever exact notation, table geometry, or figure layout matters.

## Scope and verification policy

- Publication status, title, author list, year, and venue were checked against primary paper pages such as conference proceedings, OpenReview, ACL Anthology, or arXiv.
- `PREPRINT` is used only for MultiPruner, Distill-then-Replace, and TraceNAS; no conference venue is claimed for them.
- A code URL is included only when it was identified from a primary paper record or the paper itself. Otherwise it is explicitly `UNVERIFIED`.
- This index records no Research Opportunity, candidate ranking, or final innovation claim.
- The previously generated broad literature-map files in this directory are legacy survey artifacts and are not part of this 20-paper core corpus.
