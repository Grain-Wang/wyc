# Literature Search Log

## Run metadata

- **Search date:** 2026-09-17 (CST, Asia/Shanghai)
- **Final metadata audit:** 2026-09-17 19:34 CST
- **Research topic:** Neural Architecture Search for Efficient Large Language Models
- **Permitted stage:** literature retrieval, verification, classification, and map construction only
- **Local seed audit:** `paper5/reference_papers_origin/`, `paper5/reference_papers_processed/`, and `paper5/ideas/` were empty at the start of this turn. No local PDF or prior candidate was treated as coverage.
- **Output boundary:** no paper PDF, model, dataset, canary, prototype, experiment, or research-direction file was created.

## Source hierarchy used

1. arXiv abstract pages and the official arXiv export API for canonical title, full author sequence, first-submission year, abstract, identifier, comments, and journal reference;
2. ACL Anthology, PMLR, OpenReview, CVF Open Access, IJCAI proceedings, and official conference pages for formal-publication checks;
3. repository URLs explicitly linked from a paper's primary metadata/page;
4. web search only to locate primary sources.

Blogs, secondary summaries, search-result snippets, and unsourced benchmark tables were excluded as final evidence.

## Query log

The table records query families rather than every URL-encoded variant. Boolean variants, exact-title variants, and year filters were used within each family.

| Time (CST) | Query / query family | Primary source(s) | Important results retained | Notes |
|---|---|---|---|---|
| 2026-09-17, initial pass | `"neural architecture search" AND (LLM OR "large language model")` | arXiv, OpenReview, proceedings | Search for Efficient LLMs; Puzzle; Jet-Nemotron; ZeroLM; TraceNAS; LLMForge | Broad web results were noisy; only primary records retained. |
| 2026-09-17 | `("post-training" OR pretrained) AND "architecture search" AND language model` | arXiv | Search for Efficient LLMs; Jet-Nemotron; MobileLLM-Flash; joint architecture–quantization NAS | “Post-training NAS” is recent terminology; older layerwise transformation papers required separate queries. |
| 2026-09-17 | `LANA latency aware network acceleration`, `Puzzle distillation NAS`, `Jet-Nemotron PostNAS`, `Nemotron-H MiniPuzzle` | arXiv, official proceedings/repository links | LANA; Puzzle; Jet-Nemotron; Nemotron-H | Exact-title seed verification. LANA code link remains `UNVERIFIED`. |
| 2026-09-17 | `HAT hardware-aware transformers`, `NAS-BERT`, `AutoTinyBERT`, `AutoDistil`, `AdaBERT`, `FlexiBERT`, `DARTFormer` | ACL Anthology, IJCAI, arXiv, OpenReview | Seven Transformer/NLP NAS precursors plus DARTFormer | Established the pre-LLM lineage and avoided treating decoder NAS as field-first. |
| 2026-09-17 | `hardware-aware NAS latency predictor energy memory`, `platform-aware NAS`, `device-adaptive latency NAS` | arXiv, ICLR/OpenReview, CVF | MnasNet; ProxylessNAS; FBNet; ChamNet; HW-NAS-Bench; HELP; HAT | Search separated measured latency, lookup-table latency, and learned predictors. |
| 2026-09-17 | `zero-shot NAS`, `zero-cost proxy`, `training-free architecture score`, `gradient NAS proxy`, `architecture performance predictor` | PMLR, OpenReview, CVF, arXiv | NASWOT; TE-NAS; Zen-NAS; ZiCo; BANANAS; ZeroLM; TraceNAS | Generic vision proxies retained as foundations, not direct evidence for LLM rank fidelity. |
| 2026-09-17 | `one-shot NAS`, `weight-sharing NAS`, `once-for-all`, `supernet`, `efficient NAS` | PMLR, OpenReview, arXiv | ENAS; DARTS; Once-for-All; BigNAS | 2018 preprints kept only where formal 2019-era publications are foundational. |
| 2026-09-17 | `structured LLM pruning layer block head FFN width depth` | arXiv, PMLR, OpenReview | LLM-Pruner; Sheared LLaMA; SliceGPT; ShortGPT; SLEB; LaCo; BlockPruner; MultiPruner; FLAP | Treated structured pruning as architecture search when it selects a structured subgraph. |
| 2026-09-17 | `evolutionary structured model compression`, `post-training architecture optimization LLM` | arXiv | EvoPress; Minitron; Minitron-SSM; EfficientLLM | Titles without “NAS” were explicitly included for novelty coverage. |
| 2026-09-17 | `Transformer Mamba hybrid`, `attention SSM hybrid`, `Mamba attention placement`, `hybrid LLM architecture` | arXiv | Jamba; Zamba; Samba; Hymba; Taipan; Nemotron-H; Jet-Nemotron | Most architectures use fixed/manual patterns; Jet-Nemotron is the direct searched-placement exception. |
| 2026-09-17 | `linear attention hybrid full attention`, `selective attention SSM`, `sparse full attention placement` | arXiv | H3; Based; Zoology; GLA; DeltaNet; Gated DeltaNet; Distill-then-Replace | Retrieved both operator papers and placement/conversion methods. |
| 2026-09-17 | `MHA MQA GQA MLA KV cache`, `efficient Transformer decoding attention` | arXiv | MQA; GQA; DeepSeek-V2/MLA | These are search-space primitives and conversion precedents, not NAS papers. |
| 2026-09-17 | `RetNet RWKV Hyena H3 linear attention Performer Longformer BigBird` | arXiv, PMLR, OpenReview | RetNet; RWKV; Hyena; H3; Linear Transformers; Performer; Longformer; BigBird | Primitive coverage prioritized mechanisms with distinct state/KV and retrieval behavior. |
| 2026-09-17 | `LLM inference prefill decode latency memory bandwidth throughput hardware` | arXiv, USENIX, ACM/ISCA | LLMCompass; PagedAttention; DistServe; Splitwise | Used to audit why FLOPs/parameters are insufficient hardware objectives. |
| 2026-09-17 | `FlashAttention IO-aware`, `FlashAttention-2 occupancy work partitioning` | arXiv, NeurIPS | FlashAttention; FlashAttention-2 | Kernel evidence retained because operator rankings depend on IO and utilization. |
| 2026-09-17 | `2025 2026 LLM architecture search`, `hardware-aware LLM NAS 2026`, `post neural architecture search LLM` | arXiv sorted by submission date | MultiPruner; EfficientLLM; ZeroLM; Minitron-SSM; Jet-Nemotron; ELM; TraceNAS; MobileLLM-Flash; LLMForge; joint architecture–quantization NAS | Latest-work pass; several records remain preprints. |
| 2026-09-17, final pass | arXiv API `id_list` for all 79 selected identifiers | Official arXiv export API | Canonical metadata for the final 79-paper corpus | API returned all 79 entries; no selected identifier was missing. |

## ArXiv corpus-expansion passes

Before selecting the final corpus, broad arXiv result sets were inspected in descending submission order:

- architecture search + Transformer/LLM: 100 records;
- LLM architecture/compression: 150 records;
- Transformer NAS: 120 records;
- hybrid Mamba/SSM/linear attention: 120 records;
- generic NAS/proxy/hardware foundations: 250 records;
- efficient sequence operators: 120 records;
- structured/layer/block LLM pruning: 180 records.

These result pools overlap. Counts above are retrieval limits, not unique-paper counts and not claims that every result was relevant. Final inclusion required direct methodological or metric relevance.

## Inclusion and exclusion decisions

### Included

- papers defining a NAS strategy, supernet, predictor, or zero-cost proxy relevant to the requested taxonomy;
- Transformer/BERT NAS predecessors needed to establish chronology;
- post-training LLM architecture derivation even when described as pruning, surgery, or distillation;
- efficient sequence operators plausible as search-space primitives;
- hardware/system papers that establish P/D, KV memory, bandwidth, occupancy, or kernel effects;
- recent 2025–2026 preprints with direct overlap, clearly marked by publication status.

### Excluded

- generic LLM quantization papers with no architecture variable;
- unstructured pruning papers beyond two proxy-relevant anchors (SparseGPT and Wanda);
- operator papers that duplicated a mechanism already represented without adding a distinct architecture/hardware tradeoff;
- blog summaries, vendor marketing pages, and papers for which no primary record could be located;
- weakly related edge-AI NAS papers lacking Transformer/LLM, proxy, or hardware-method relevance.

## Source limitations and failed/low-yield routes

- Broad keyword search for `LLM NAS` produced unrelated uses of “NAS” and generic compression; exact-title and arXiv-category filtering were needed.
- Direct OpenReview forum requests encountered a browser-verification challenge during the final link audit. OpenReview URLs not independently corroborated were removed; arXiv metadata or another official proceedings page remains the source for those rows.
- Venue metadata is often absent from arXiv even when a paper may later have appeared formally. No venue was inferred solely from memory.
- Many paper pages say that code/models exist without exposing a single canonical repository in the checked metadata. Those code fields remain `UNVERIFIED`.
- No publisher/API failure caused a selected paper to lack canonical arXiv metadata. Some formal proceedings and code repositories were not exhaustively checked because the arXiv record was sufficient for bibliographic inclusion and secondary-source inference was disallowed.
- Exact compute is inconsistently reported and difficult to normalize across pretraining, supernet training, local distillation, calibration search, and recovery. The table records qualitative regimes and only a few explicit figures from abstracts.

## `UNVERIFIED` register

### Formal venue/status

The checked primary metadata did not verify a formal venue for at least the following high-priority papers: LANA, FlexiBERT, Minitron, Puzzle, MultiPruner, EfficientLLM, ZeroLM, Nemotron-H, Minitron-SSM, Distill-then-Replace, TraceNAS, LLMForge, and joint architecture–quantization NAS. Several operator papers have the same caveat in `literature_table.md`.

### Code

Canonical code URLs remain `UNVERIFIED` for most papers. Verified or paper-linked exceptions recorded in the corpus include HAT, AutoTinyBERT, AutoDistil's author project link, Search for Efficient LLMs, MultiPruner, EfficientLLM, Sheared LLaMA, DARTS, Once-for-All, LLMCompass, vLLM/PagedAttention, and Performer.

### Costs and hardware detail

- Exact GPU-hours, accelerator counts, energy, training tokens, and search/recovery splits are `UNVERIFIED` unless explicitly stated in the table.
- TraceNAS's 8.5 single-GPU search time, Puzzle's up-to-45B recovery-token statement, and Minitron/Minitron-SSM relative token statements come from the paper abstracts; hardware-normalized comparisons remain unavailable.
- Where a paper reports “latency” without a fully audited device/runtime/workload configuration in this pass, those details are `UNVERIFIED`.

### 2026 records

MobileLLM-Flash is labeled ACL Industry Track 2026 by its arXiv comment. Distill-then-Replace, TraceNAS, LLMForge, and joint architecture–quantization NAS remain preprints in the checked metadata; later venue or code changes were not assumed.

## Completion audit

- [x] 79 high-relevance papers, within the requested 60–100 range
- [x] 22 direct nearest neighbors, within the requested 15–25 range
- [x] 32 strongly related methods
- [x] 25 foundations/adjacent methods
- [x] dedicated 2025–2026 coverage
- [x] all 14 requested taxonomy categories represented
- [x] primary URLs attached to every paper
- [x] uncertain venues, code, costs, and device details marked `UNVERIFIED`
- [x] structured pruning treated as a novelty-relevant architecture-search neighbor
- [x] no Research Opportunity, candidate ranking, canary, GPU run, prototype, baseline experiment, Paper Candidate, or paper prose produced
