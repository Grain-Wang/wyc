---
id: "2026_Distill_Then_Replace"
title: "Distill-then-Replace: Efficient Task-Specific Hybrid Attention Model Construction"
authors: ["Xiaojie Xia", "Huigang Zhang", "Chaoliang Zhong", "Jun Sun", "Yusuke Oishi"]
year: 2026
venue: "arXiv"
publication_status: "PREPRINT"
category: "Direct Neighbor / Blockwise Distillation"
source_pdf: "../reference_papers_origin/2026_Distill_Then_Replace.pdf"
paper_url: "https://arxiv.org/abs/2601.11667"
pdf_url: "https://arxiv.org/pdf/2601.11667"
code_url: "UNVERIFIED"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# Distill-then-Replace: Efficient Task-Specific Hybrid Attention Model Construction

**Authors:** Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, Yusuke Oishi

**Venue / Year:** arXiv (PREPRINT)

**Category:** Direct Neighbor / Blockwise Distillation

**Why this paper matters for low-cost post-training LLM NAS:** It is a direct non-NAS mathematical neighbor that locally distills replacement blocks and greedily composes a pretrained hybrid architecture.

**Primary record:** [https://arxiv.org/abs/2601.11667](https://arxiv.org/abs/2601.11667)

**Local source:** [2026_Distill_Then_Replace.pdf](../reference_papers_origin/2026_Distill_Then_Replace.pdf)

## Full converted text

## **Distill-then-Replace: Efficient Task-Specific** **Hybrid Attention Model Construction**

Xiaojie Xia [1[0000] _[−]_ [0002] _[−]_ [6486] _[−]_ [7557]], Huigang Zhang [1[0000] _[−]_ [0002] _[−]_ [7567] _[−]_ [7177]],
Chaoliang Zhong [1[0000] _[−]_ [0003] _[−]_ [1697] _[−]_ [6932]], Jun Sun [1[0000] _[−]_ [0002] _[−]_ [0967] _[−]_ [4859]], and
Yusuke Oishi [2[0000] _[−]_ [0003] _[−]_ [4264] _[−]_ [8932]]


1 Fujitsu Research & Development Center CO., LTD, China
```
      {xiaxiaojie,zhanghuigang,clzhong,sunjun}@fujitsu.com
```

2 Fujitsu Research, FUJITSU LTD, Japan
`{oishi.y@fujitsu.com` }


**Abstract.** Transformer architectures deliver state-of-the-art accuracy
via dense full-attention, but their quadratic time and memory complexity with respect to sequence length limits practical deployment. Linear
attention mechanisms offer linear or near-linear scaling yet often incur
performance degradation. Hybrid models that integrate full and linear
attention layers promise a balance between efficiency and expressiveness,
but face two major challenges: training such hybrid models from scratch
is computationally expensive, and manually designing the optimal placement of attention types is highly nontrivial. We propose **DtR** ( **D** istill**t** hen- **R** eplace), which first transfers weights from the pretrained fullattention modules to its linear attention counterparts through blockwise
local distillation, and then applies a greedy layer replacement strategy
that iteratively substitutes full attention blocks with linear ones while
monitoring validation performance on the target task. DtR yields a taskspecific hybrid model in a single efficient pass, without costly re-training

    - r neural architecture search, and can be applied to any pretrained fullattention backbone for diverse downstream tasks.


**Keywords:** Hybrid Attention Models · Task-Specific Models · Blockwise Local Distillation.


**1** **Introduction**


Transformer architectures[40] achieve state-of-the-art performance through dense,
softmax-based self-attention, yet suffer from quadratic time and memory complexity with respect to sequence length. This poses significant constraints for
long-context tasks such as document understanding and extended dialogue[31].
Recent works have developed efficient alternatives with linear or near-linear
complexity [23,29,33]. Models such as RetNet [37], Mamba [22], and Gated
DeltaNet [43] compress the KV Cache into a static hidden state, substantially
reducing memory and computation. However, these methods often incur performance penalties on tasks requiring precise recall or nuanced contextual understanding [41], due to historical information compression.




--- end of page=0 ---

2 Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi


Hybrid architectures that interleave full-attention and linear-attention blocks
balance efficiency and expressivity [34,19,30], preserving transformer accuracy on
critical tokens while achieving linear scaling elsewhere. Yet two barriers impede
adoption: (i) training from scratch requires resources comparable to full-scale
pretraining, making it prohibitively costly; and (ii) optimal placement of attention types is a complex engineering challenge with no systematic solution.
We address both issues by proposing efficient, task-specific hybrid model creation from existing pretrained transformers. Our approach, **DtR** ( **D** istill- **t** hen**R** eplace), transfers weights to linear counterparts via blockwise local distillation,
avoiding expensive re-pretraining, then applies a greedy layer-replacement algorithm that incrementally substitutes full-attention blocks with linear ones while
monitoring task-specific validation performance. DtR produces the resulting hybrid model in a single low-cost pass, requires no costly architectural search,
and can be instantiated from any pretrained full-attention backbone for diverse
downstream tasks.


**2** **Related works**


**2.1** **Linear Complexity Models**


While Transformers [40] have become the de facto standard for LLMs, their
global attention suffers from quadratic complexity _O_ ( _N_ [2] ). Early works proposed
kernel-based linear attention mechanisms, such as Linear Transformer [29] and
Performer [11], which approximate the softmax attention map. More recently,
architectures unify Transformers’ parallel training with RNNs’ efficient inference. RWKV [33] reformulates linear attention into a recurrent form for linear
inference scaling. RetNet [37] introduces multi-scale exponential decay retention,
achieving training parallelism, low-cost inference, and strong performance. Structured state-space models such as Mamba [22,15] incorporate data-dependent
state transitions for high expressivity and subquadratic scaling. Gated Linear
Attention [44]and Gated DeltaNet [43] integrate gating mechanisms into linear
recurrent units, enhancing stability and capacity. These models demonstrate that
efficient attention alternatives can achieve competitive performance.


**2.2** **Hybrid Architectures**


Pure linear architectures sometimes struggle with high-fidelity associative recall and in-context retrieval. Hybrid models bridge this gap by combining both
paradigms. H3 [17] first effectively connected SSMs with transformers on language tasks. More recently, Jamba [30], Samba [34], and Zamba [19] wove Mamba
layers with transformer blocks in complementary ways, attaining state-of-the-art
results. Griffin [16] mixes gated linear recurrences with local attention, showing
that limited global attention suffices to recover full-attention capabilities. JetNemotron [25] employs post-training neural architecture search to discover optimal attention placement automatically. These works establish hybridization as a
promising direction. However, these hybrid models typically require substantial
training resources and extended training time.




--- end of page=1 ---

Efficient Task-Specific Hybrid Attention Model Construction 3


**2.3** **LLM Distillation**


Knowledge Distillation (KD) [27] compresses large teachers into smaller students. For generative LLMs, MiniLLM [24]identifies that forward KL divergence
causes students to overestimate low-probability regions, proposing reverse KL
instead. [1] explores generalized distillation through data selection strategies. Recent works [42,20] investigate cross-architecture distillation, where transformer
teachers distill into linear or hybrid students (e.g., Mamba-2), ensuring efficient
students inherit strong reasoning capabilities. Yet cross-architecture distillation
remains data-intensive and computationally demanding.


**3** **Methodology**


Our proposed DtR (Distill-then-Replace) introduces a streamlined, cost-effective
pipeline for converting any pretrained, standard transformer into a task-specific
hybrid model with an optimized mixture of full- and linear-attention layers. It
consists of two core components: weight transferring via blockwise local distillation and an automated greedy layer-replacement algorithm.


**3.1** **Blockwise Local Distillation to Linear Attention Counterparts**


In this stage, our objective is to derive a linear attention mechanism that can
effectively approximate the behavior of each softmax-based full-attention block
in the original transformer architecture. Inspired by the methodology introduced
in [7], we train each linear counterpart independently, such that it accurately
reproduces the output of its corresponding full-attention module when linear
modules are provided with the same hidden state input as the parent block.
As illustrated in Fig. 1, the block-wise local distillation objective _L_ is concisely expressed as


_L_ = MSE( _O_ full _, O_ linear)


where _O_ full and _O_ linear represent the outputs of the original full-attention block
and the linear counterpart block, respectively.
This decoupled distillation ensures that each linear attention module depends
solely on the behavior of its associated full-attention block, without requiring
back-propagation through the entire network. Consequently, all linear attention
modules can be trained in parallel, significantly reducing computational overhead
and accelerating convergence compared to end-to-end joint pretraining. This
design not only enhances training efficiency but also preserves the functional
fidelity of the original attention mechanism, facilitating seamless integration into
existing architectures.


**3.2** **Task-Specific Hybrid Architecture Construction**


Upon obtaining both the original full-attention transformer model and its distilled linear counterparts, we now seek an optimal hybrid architecture that balances performance and efficiency for a specific downstream task. Rather than




--- end of page=2 ---

4 Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi


### Figure 1

Caption: ** Linear attention weight by blockwise local distillation. (a) Overall distillation

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
framework from full attention to linear attention. (b) BLD (blockwise local distillation),
which are trained in parallel and independently.


resorting to costly neural architecture search or end-to-end fine-tuning of hybrid
configurations, we propose a greedy, validation-driven layer replacement strategy
that constructs a task-specific hybrid model in a single, low-overhead pass.
Given a pretrained full-attention model _M_ full with _L_ blocks and a downstream task _T_, we construct an efficient hybrid model through greedy layer
replacement.


_Setup._ We evaluate the baseline performance _P_ ( _M_ full; _T_ ) on a task validation
set and establish a minimum tolerated acceptable threshold _P_ min. The goal is to
replace as many full-attention layers with their distilled linear counterparts as
possible, while maintaining _P_ ( _M_ hybrid; _T_ ) _≥P_ min.


_Greedy Layer Replacement._ Starting from _M_ full, we iteratively substitute one
full-attention block at a time with its linear counterpart and evaluate validation
performance. In each iteration, we select the layer whose replacement yields the
highest validation score _P_ _[∗]_ . If _P_ _[∗]_ _≥P_ min, we commit the replacement and
continue; otherwise, we stop. The procedure requires only forward passes—no
back-propagation or retraining.


_Output models._ We record two models: (i) _M_ best, the hybrid achieving the highest validation performance throughout the search, and (ii) _M_ - pt, the hybrid
with the maximum number of linear layers while satisfying the performance
constraint.


The full procedure is presented in Algorithm 1. Our greedy layer replacement
algorithm is motivated by the observation that full and linear attention mech



--- end of page=3 ---

Efficient Task-Specific Hybrid Attention Model Construction 5


**Algorithm 1** DtR (Distill-then-Replace): Efficient Task-Specific Hybrid Attention Model Construction
**Require:** Full-attention model _Mfull_ with _L_ blocks; Training data _D_ ; Downstream
task training dataset _Ttest_ ; Downstream task validation set _Tval_ ; Metric _P_ ( _·_ ; _T_ );
Minimum performance threshold _Pmin_
**Ensure:** _Mbest_ : highest-performance hybrid model; _Pbest_ : its metric; _Mopt_ : maximumreplacement hybrid with _P ≥Pmin_ ; _Popt_ : its metric
1: **Phase 1:** BLD(Blockwise Local Distillation) to Linear Attention Counterparts
2: **for** _ℓ_ = 1 to _L_ **in parallel do**
3: _M_ [(] _linear_ _[ℓ]_ [)] _[←]_ [BLD][(] _[M]_ [(] _full_ _[ℓ]_ [)] [;] _[ D]_ [)]
4: **end for**

5: **Phase 2:** Greedy Layer Replacement to Hybrid Attention Model
6: _M ←Mfull_, _Pbest ←P_ ( _M_ ; _Tval_ ), _Mbest ←M_, _Mopt ←M_, _Popt ←P_ ( _M_ ; _Tval_ )
7: **repeat**
8: _M_ _[∗]_ _, P_ _[∗]_ _, ℓ_ _[∗]_ _←∅, −∞, −_ 1
9: **for** each full-attention layer index _ℓ_ in _Mfull_ **do**
10: _Mℓ_ _←_ Replace-Layer-to-Linear( _M, ℓ, M_ [(] _linear_ _[ℓ]_ [)] [)]
11: _Pℓ_ _←P_ ( _Mℓ_ ; _Tval_ )
12: ( _M_ _[∗]_ _, P_ _[∗]_ _, ℓ_ _[∗]_ ) _←_ ( _Mℓ, Pℓ, ℓ_ ) **if** _Pℓ_ _> P_ _[∗]_

13: **end for**

14: _Mbest ←_ _M_ _[∗]_, _Pbest ←P_ _[∗]_ **if** _P_ _[∗]_ _≥Pbest_
15: _M ←M_ _[∗]_, _Mopt ←M_, _Popt ←P_ _[∗]_ **if** _P_ _[∗]_ _≥Pmin_ **else break**
16: **until** _P_ _[∗]_ _< Pmin_
17: _Pbest_ _[test]_ _[←P]_ [(] _[M][best]_ [;] _[ T][test]_ [)][,] _[ P]_ _opt_ _[test]_ _[←P]_ [(] _[M]_ _opt_ [;] _[ T]_ _test_ [)]
18: **return** _Mbest_, _Pbest_ _[test]_ [,] _[ M][opt]_ [,] _[ P]_ _opt_ _[test]_


anisms exhibit task-dependent representational strengths. Consequently, where
linear modules are placed significantly affects performance: some layers can be
replaced without loss—and sometimes with gain—while others are critical to
preserve. By iteratively evaluating validation performance after each candidate
replacement, our method adapts to each task’s unique knowledge distribution,
identifying an effective hybrid configuration without exhaustive search. The procedure requires only a forward pass per candidate layer, making it highly efficient.
Crucially, it automatically retains full-attention layers essential for expressivity
and replaces less sensitive ones with linear approximations, achieving substantial
speedups while maintaining or even improving task performance.
Importantly, our framework is backbone-agnostic and task-general: it can be
applied to any pretrained full-attention transformer (e.g., GPT, LLaMA, Qwen)
and adapted to various downstream tasks without modifications to the core
algorithm. This enables rapid deployment of efficient, high-performing models at
a fraction of the cost of training hybrid architectures from scratch or conducting
exhaustive architecture search.


**4** **Experiments**


Having introduced our proposed DtR (Distill-then-Replace), we now present a
series of experiments designed to build a comprehensive case for its effectiveness.




--- end of page=4 ---

6 Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi


**4.1** **Experiment Setup**


**Datasets, Models and Evaluation Details.** We evaluate our method across
a diverse suite of benchmarks spanning commonsense knowledge, reasoning, code
generation, information extraction, long-context summarization and specific domain tasks. For commonsense and reasoning tasks, we consider ArcC [13], ArcE

[13], BoolQ(BQ) [12], CommonsenseQA(CQ) [38], HellaSwag(HS) [45], OpenbookQA (OBA) [32], PIQA [8] and Winogrande(WG) [35]. For mathematical
reasoning, we include GSM8K [14] and MathQA(MQ) [3]. For code generation,
we evaluate on HumanEval(HE) [10] and MBPP [5]. For information extraction, we include SWDE [4]. For long-context evaluation, we include SAMSum, a
dialogue summarization subtask, and TREC, a question classification subtask,
both from LongBench [6]. For specific domain tasks, we include two MMLU [26]
subtasks: econometrics(EC) and marketing(MK), and PubMedQA(PM) [28]. We
adopt the official validation split as the evaluation dataset in our algorithm if
available; otherwise, we randomly sample a small subset from the training set.
We conduct experiments on three decoder-only base models: Qwen2.5-1.5B

[39], Llama-3.2-3B-Instruct [21], and Llama-3.1-8B-Instruct [21], with 28, 28,
and 32 layers, respectively. We evaluate using LM-Evaluation-Harness [18] following a zero-shot protocol (except for GSM8K, which adopt 4-shot evaluation),
reporting each dataset’s official metrics, such as accuracy (acc), normalized accuracy (acc_norm), pass1, F1-score, or Rouge-L, where higher values indicate
better performance.


**Method Details.** Our method consists of two stages. In the blockwise local
distillation stage, we randomly sample about 100M mini-corpus from a combination of Nemotron-CC [36] and Redstone-QA [9]. We only train the linear
modules using a MSE loss while keeping the embeddings, MLPs and LM head
frozen. During layer replacement stage, we first evaluate the full-attention model
and record its performance as a baseline, then iteratively replace layers to linear
attention on each task’s validation set to obtain the best architecture, and then
evaluate on their test split. The alternative linear modules evaluated are Gated
DeltaNet(GDN) [43], Gated Linear Attention(GLA) [44] and JetBlock(JET)

[25], all configured with the same hidden size as corresponding full-attention
layers. The entire procedure is executed on a single NVIDIA A800 GPU.


**4.2** **Results**


**DtR-searched Best Hybrid Model Performance.** We begin by evaluating
the performance of the best hybrid model across various linear attention variants, as shown in Table 1, using different pretrained base models(Qwen2.5-1.5B,
Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct). _P_ base and _P_ best denote the
performance of the base and searched best hybrid models on each test dataset,
respectively; # _R_ is the number of replaced layers to linear attention. The performance in **bold** indicates that the searched hybrid model matches or exceeds
the pretrained base model on the corresponding task.




--- end of page=5 ---

Efficient Task-Specific Hybrid Attention Model Construction 7


**Table 1.** Performance of base models ( **Qwen2.5-1.5B**, **Llama-3.2-3B-Instruct** and
**Llama-3.1-8B-Instruct** ) and best DtR-searched hybrid models with replaced linear
modules ( **GLA**, **GDN** and **JET** ) across tasks.


**Qwen2.5-1.5B** **Llama-3.2-3B-Instruct** **Llama-3.1-8B-Instruct**
**Datasets** **GLA** **GDN** **JET** **GLA** **GDN** **JET** **GLA** **GDN** **JET**

_P_ base _P_ best #R _P_ best #R _P_ best #R _P_ base _P_ best #R _P_ best #R _P_ best #R _P_ base _P_ best #R _P_ best #R _P_ best #R
ArcC 45.1 **45.3** 1 **45.7** 3 **45.2** 2 46.4 44.7 6 43.1 2 **46.5** 4 55.0 **55.4** 2 **57.1** 7 **55.5** 2

ArcE 72.0 **74.0** 2 **74.4** 4 **74.7** 2 67.9 **69.8** 5 **70.2** 8 **70.8** 4 79.6 **80.7** 7 **81.5** 9 **80.7** 2

BQ 73.0 **75.4** 3 **76.0** 3 **75.2** 3 78.5 **80.2** 9 **80.3** 11 **80.5** 10 84.2 **85.1** 13 **84.7** 12 **84.6** 4
CQ 75.2 75.1 1 74.8 1 74.1 4 67.7 **68.4** 6 64.8 7 **68.2** 3 77.2 75.1 2 75.7 2 **77.3** 10
HS 50.1 **50.2** 2 **50.4** 2 **52.9** 4 52.4 **52.4** 3 **52.4** 1 **54.8** 3 59.2 **59.3** 2 **59.3** 1 **59.3** 2

OBA 40.6 **40.8** 2 40.0 2 **40.8** 8 36.0 **36.2** 10 **37.0** 3 **37.6** 2 43.4 **43.8** 4 **43.4** 5 **44.4** 8

PIQA 75.7 **75.7** 3 75.3 2 75.0 7 75.5 73.9 6 75.0 2 **75.8** 8 80.0 **80.1** 1 79.7 2 **80.1** 1
WG 63.3 **63.7** 3 **63.3** 5 **64.2** 6 68.0 67.2 4 **68.0** 5 **68.1** 4 74.0 **74.2** 6 **75.1** 10 **74.5** 2

GSM8K 61.4 61.3 3 60.9 1 60.1 1 63.9 63.8 1 63.6 1 63.2 1 76.0 **76.3** 2 **77.0** 2 75.1 1

MQ 35.0 **36.4** 3 **36.3** 2 **35.8** 3 34.5 **35.7** 4 **35.8** 4 **35.3** 3 39.4 **41.0** 4 **41.1** 7 **40.8** 3
HE 47.6 **48.2** 1 **48.2** 1 **47.6** 1 51.2 **54.9** 1 **53.1** 1 **54.3** 1 58.5 **62.8** 1 **65.2** 4 **64.6** 1

MBPP 50.0 **52.9** 2 **52.7** 2 **52.7** 2 54.8 **58.2** 2 **57.1** 2 **56.6** 1 59.8 **64.3** 2 **63.8** 2 **62.9** 2

SWDE 86.4 **89.3** 5 **89.4** 3 **89.1** 3 85.8 **89.0** 4 **87.7** 5 **88.4** 4 91.1 **92.3** 5 **92.4** 8 **91.6** 3

SAMSum 42.5 **44.4** 4 **43.9** 4 **43.8** 3 42.4 **42.5** 2 42.3 1 **42.9** 2 43.9 **45.1** 1 **45.1** 3 **44.5** 1

TREC 71.0 **75.0** 4 **76.0** 5 **76.0** 5 72.0 **76.0** 4 **75.0** 5 **76.0** 5 73.5 **77.0** 5 **77.5** 4 **78.0** 6

EC 46.5 45.6 1 43.9 1 **46.5** 2 39.5 **39.5** 1 36.0 5 39.3 2 51.8 **53.5** 1 50.9 1 **53.5** 1

MK 83.3 82.5 2 **83.8** 4 **83.8** 1 86.8 82.5 2 **86.8** 2 **87.2** 1 88.9 **90.2** 1 **89.3** 7 **89.7** 4

PM 63.8 **72.2** 7 **72.2** 9 **68.4** 6 69.8 **72.2** 7 **72.0** 6 **73.0** 6 75.0 **78.0** 13 **77.6** 12 **78.4** 8


The results demonstrate strong empirical evidence for the effectiveness of
task-specific hybrid architectures. Across all configurations (3 base models _×_ 3
linear variants _×_ 18 tasks = 162 settings), **81.5% (132/162)** achieve performance comparable to or exceeding the full-attention baseline. More importantly,
_every task admits at least one hybrid configuration that matches or surpasses the_
_base model_, confirming that optimal placement is task-dependent rather than
universal. Even in the remaining configurations where slight degradation occurs,
the average drop is only 1.8% absolute while the efficiency gains. Section 4.3
provides empirical grounding: probing analysis confirms that low-impact-score
layers (high substitutability) are replaced early and high-impact-score layers preserved, validating the algorithm’s implicit prioritization.


**Performance-Efficiency Trade-offs.** As previously noted, incorporating more
linear attention modules can lead to higher inference speed; however, this often
comes at the cost of reduced accuracy. Using Llama3.2-3B-Instruct as the base
model, we compare decoding throughput across configurations with different
numbers of layers replaced by Jet-Block, under context lengths of 512, 2,048,
16,384 and 65,536, following same evaluation protocol in [25]. As illustrated in
### Figure 2

Caption: , decoding throughput increases with the number of linear layers, and

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
greater speedups are observed at longer context lengths.
We allow a maximum performance drop of 5% relative to the base model on
the validation set and select the hybrid model with the most linear attention
layers as the optimal configuration. Then we examine the resulting trade-off between performance and efficiency on each test dataset. As shown in Table 2, _P_ base
and _P_ - pt denote the performance of the base and optimal hybrid models, respectively; _Drop_ represents the degradation percent of _P_ - pt compared with _P_ base;




--- end of page=6 ---

8 Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi


### Figure 2

Caption: ** Throughput comparison under context length of 512, 2,048, 16,384 and 65,536.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
The numbers above points indicate the speedup relative to base full-attention model.


**Table 2.** Performance of base model ( _P_ base) and optimized model ( _P_ - pt), showing performance drop percent ( _Drop_ (%)), number of replaced layers (# _R_ ) and corresponding
throughput speedup( _×_ ) under context lengths of 2,048.


ArcC ArcE BQ CQ HS OBA PIQA WG GSM8K MQ HE MBPP SWDE SAMSum TREC EC MK PM
_P_ base 46.4 67.9 78.6 67.7 52.4 36.0 75.5 68.0 63.9 34.5 51.2 54.8 85.8 42.4 72.0 39.5 86.8 69.8
_P_ - pt 43.3 66.5 76.5 64.7 49.7 34.8 71.9 64.1 59.6 33.5 48.8 52.7 81.8 40.0 68.5 35.7 79.4 68.8
_Drop_ 6.68 2.06 2.67 4.43 5.15 3.33 4.77 5.74 6.73 2.90 4.69 3.83 4.66 5.66 4.86 9.62 8.53 1.43
# _R_ 10 11 16 11 9 13 14 12 6 10 6 5 11 10 10 10 10 15
_Speedup_ ( _×_ ) 1.47 1.54 2.13 1.54 1.38 1.42 1.86 1.61 1.24 1.47 1.24 1.17 1.54 1.47 1.47 1.47 1.47 1.98


and # _R_ indicates the number of replaced layers to linear attention in the optimal
model. The hybrid model achieves substantial speedups with controlled performance degradation (average drop 4.8%). With more layers replaced to linear
attention,the hybrid model achieves substantially higher inference throughput,
a trend that is consistently visualized in Fig. 2. By prescribing an acceptable
degradation margin, the performance and efficiency can be flexibly balanced to
meet diverse real-world deployment constraints.


**Replacement Order Analysis.** We investigate the impact of layer replacement trajectories using different base models, linear attention variants, and
downstream tasks.

As depicted in Fig. 3, we progressively replace all layers with linear attention
by our DtR method. The result suggests that the parametric storage of knowledge is distributed heterogeneously across different base models, while downstream tasks exhibit distinct knowledge demands. Consequently, it is essential
to identify a task-specific model architecture.
Interestingly, we observe that the layer replacement trajectories across different linear attention variants exhibit a remarkable degree of consistency. This
suggests that the optimal placement order of linear attention modules is primar



--- end of page=7 ---

Efficient Task-Specific Hybrid Attention Model Construction 9


### Figure 3

Caption: ** Layer replacement trajectories on PubmedQA (PM) and CommonsenseQA

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
(CQ) using Qwen2.5-1.5B and Llama3.2-3B-Instruct (28 layers each) with linear attention variants: Gated Linear Attention (GLA) and Jet-Block (JET).


ily governed by the base model architecture and task characteristics, rather than
the specific design of the linear attention mechanism itself.


### Figure 4

Caption: ** Comparison of different replacement strategies.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


**Comparison with Alternative Replacement Strategies.** We compare our
proposed DtR against several alternative layer replacement strategies: (1) **Local**
**layer importance** : layers are replaced one by one according to their importance
scores, which are computed by ranking the performance drop caused by replacing




--- end of page=8 ---

10 Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi


each layer to linear attention locally in the base model; (2) **Uniform interleav-**
**ing** : full-attention and linear-attention layers are evenly interleaved across the
model depth; (3) **Random layer replacement** : in each iteration, layers to be
replaced with linear attention are selected at random.
The results are shown in Fig. 4. Our replacement strategy consistently and
significantly outperforms other methods across varying numbers of replaced layers, two distinct base models and tasks, highlighting the effectiveness of greedy,
validation-guided replacement in identifying task-specific hybrid architectures.


**Experimental Cost Analysis.** We report the total experimental cost of our
method on a single NVIDIA A800 GPU on the PubMedQA dataset with different base models. Note that the greedy search ends when all full-attention layers
are replaced, providing an upper-bound greedy search runtime. Table 3 shows
that the entire pipeline completes in just a few hours on a single A800 GPU,
enabling efficient conversion from a pretrained model into a task-specific hybrid architecture. The data requirement is also minimal—approximately 100M
general-domain tokens for blockwise local distillation and only a small taskspecific validation set.


**Table 3.** GPU hours across base models during BLD and greedy replacement on
PubMedQA, adopting 100M general-domain tokens for blockwise local distillation and
500 samples for task-specific validation.


BLD Greedy replacement Total
Base Model
GPU Hours GPU Hours GPU Hours


Qwen2.5-1.5B 2.5 2.5 5.0
Llama3.2-3B-Instruct 4.0 3.0 7.0

Llama3.1-8B-Instruct 8.0 7.0 15.0


### Figure 5

Caption: ** Supervised fine-tuning (SFT) on DtR-searched hybrid model for further gains.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


**Fine-Tuning for Further Gains on DtR-searched Hybrid Model.** Building upon the best hybrid models obtained through our DtR procedure, we in



--- end of page=9 ---

Efficient Task-Specific Hybrid Attention Model Construction 11


vestigate the impact of supervised fine-tuning (SFT) on downstream task performance, using Llama3.2-3B-Instruct (28 layers) as the base model and JetBlock as the linear attention module. We apply SFT to the official training data

- f CommonsenseQA and PubMedQA for each best hybrid model, varying the
number of replaced attention layers across {4, 8, 12, 16, 20, 24, 28}.
As shown in Fig. 5, our DtR-searched hybrid models either match or surpass
the performance of the base model with fewer than 12 to 16 layers replaced by
linear attention. Moreover, when supervised fine-tuning is applied to these hybrid
configurations, their performance improves further—sometimes even surpassing
that of the fine-tuned base model. This demonstrates that our DtR-searched
hybrid architectures can effectively benefit from additional improvement mechanisms such as fine-tuning.


**4.3** **Mutual Information based Empirical Analysis**


Following the Information Bottleneck principle in [2], we operationalize mutual information estimation without explicit density modeling by treating linear
probe accuracy as a practical lower bound on _I_ ( **H** ; _Y_ ) —the mutual information between hidden states **H** and task labels _Y_ . Building on this, we design a
probing-based criterion to identify which full-attention layers can be replaced by
their linear counterparts without incurring performance loss.


**Impact Score.** For each layer _ℓ_, let _A_ [(] full _[ℓ]_ [)] [be the accuracy of a linear probe]
trained on the full module’s hidden states **H** [(] full _[ℓ]_ [)] [to predict task label] _[ Y]_ [ . Similarly,]
_A_ [(] linear _[ℓ]_ [)] [is the probe accuracy on the linear module’s states] **[ H]** [(] linear _[ℓ]_ [)] [, and] _[ A]_ [(] cross _[ℓ]_ [)] [is]
the accuracy of the full-module probe evaluated on **H** [(] linear _[ℓ]_ [)] [. Here,] _[ A]_ [linear][ measures]
task-relevant information retained by the linear module, while _A_ cross measures
alignment with the full module’s discriminative directions.
A good replacement candidate should have both _A_ linear and _A_ cross close to
_A_ full. Thus we define a layer-wise **impact score** that is high when replacement
is harmful (i.e., the layer should be kept as full attention):


Impact Score [(] _[ℓ]_ [)] = 1 _−_ _[A]_ linear [(] _[ℓ]_ [)] [+] _[ A]_ [(] cross _[ℓ]_ [)] _._

2 _A_ [(] _[ℓ]_ [)]
full


Intuitively, if linear module perfectly preserves both task information and
alignment ( _A_ linear _≈_ _A_ cross _≈_ _A_ full), then impact score _≈_ 0 — the layer should be
replaced early. If the linear module performs poorly on either metric, importance
becomes positive, indicating the layer should be retained. The score is monotonic
with respect to the greedy replacement order: layers with lower importance are
replaced earlier.
We validate this by computing the Spearman rank correlation between the
normalized Impact Score [(] _[ℓ]_ [)] and the actual replacement order (1 for earliest replaced, _L_ for last). Fig. 6 shows consistently high positive correlations across
tasks, confirming that our probing score reliably predicts replacement. This provides theoretical grounding for our method: the algorithm implicitly prioritizes
layers with low impact score (high linear substitutability) for replacement.




--- end of page=10 ---

12 Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi


### Figure 6

Caption: ** Visualization between the impact score (normalized from 0 to 1) and actual

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
replacement order (taller = replaced later, should be kept as full attention) across three
tasks(PubMedQA, CommonsenseQA and BoolQ), using Qwen2.5-1.5B (28 layers) as
the base model and Jet-Block as the linear attention module.


**Fixed-ratio layer replacement analysis: why validation feedback mat-**
**ters.** While Impact Score explain replacement order post-hoc, whether they
can prescribe optimal configurations remains open. As Table 4 shown, the two
configurations diverge significantly: Impact-Score-Guided’s 10-layer setup underperforms Greedy-Layer-Replacement with the same budget. This reveals **layer**
**interactions** : replacing one layer alters the functional role of others. Validation
feedback captures these dependencies, while static impact score cannot. The performance gap quantifies the cost of skipping task-specific validation, justifying
greedy search’s modest overhead.


**Table 4.** Greedy layer replacement outperforms static Impact-Score guidance under
fixed replacement budget (10/28 layers, Qwen2.5-1.5B, Jet-Block).


Task Method Replaced Layers Acc (%) _∆_


Greedy Layer Replacement {4,8,12,19,21,22,23,24,25,26} **70.2**          PubMedQA
Impact-Score-Guided {0,6,10,13,17,20,22,23,26,27} 66.4          - 3.8


**64.8**                                  CommonsenseQA [Greedy Layer Replacement {2,3,6,10,14,19,23,24,25,26}]
Impact-Score-Guided {4,5,7,10,13,14,17,24,25,27} 61.3          - 3.5


Greedy Layer Replacement {2,5,10,18,19,22,23,24,25,26} **73.0**          BoolQ
Impact-Score-Guided {1,5,10,11,14,18,19,20,26,27} 70.4          - 2.6


**5** **Conclusion**


We have presented a practical and efficient framework for automatically constructing task-specific hybrid models that judiciously integrates full attention
and linear attention mechanisms. By first distilling linear attention counterpart
for each full attention block via blockwise local distillation, and then applying
a greedy, validation-guided replacement strategy, our proposed DtR identifies
a high-performing hybrid architecture without costly pretraining or neural ar



--- end of page=11 ---

Efficient Task-Specific Hybrid Attention Model Construction 13


chitecture search. Extensive experiments show that our approach yields taskspecific hybrid models, which strategically retain full attention in task-critical
layers to preserve expressive power, while adopting linear attention elsewhere to
gain computational efficiency.
Crucially, our approach is both backbone-agnostic and task-general: it can
be seamlessly applied to any pretrained full-attention models and adapted to
diverse downstream tasks with minimal overhead. We believe this paradigm offers a scalable and deployment-friendly pathway toward efficient adaptation of
foundation models, particularly in resource-constrained or latency-sensitive scenarios.


## References


1. Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Garea, S.R., Geist, M., Bachem,
O.: On-policy distillation of language models: Learning from self-generated mistakes. In: The Twelfth International Conference on Learning Representations
(2024)
2. Alain, G., Bengio, Y.: Understanding intermediate layers using linear classifier
probes. arXiv preprint arXiv:1610.01644 (2016)
3. Amini, A., Gabriel, S., Lin, S., Koncel-Kedziorski, R., Choi, Y., Hajishirzi, H.:
Mathqa: Towards interpretable math word problem solving with operation-based
formalisms. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: Human language technologies,
volume 1 (long and short papers). pp. 2357–2367 (2019)
4. Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Zinsley, D., Zou,
J., Rudra, A., Ré, C.: Simple linear attention language models balance the recallthroughput tradeoff. arXiv preprint arXiv:2402.18668 (2024)
5. Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E.,
Cai, C., Terry, M., Le, Q., et al.: Program synthesis with large language models.
arXiv preprint arXiv:2108.07732 (2021)
6. Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng,
A., Hou, L., et al.: Longbench: A bilingual, multitask benchmark for long context
understanding. In: Proceedings of the 62nd annual meeting of the association for
computational linguistics (volume 1: Long papers). pp. 3119–3137 (2024)
7. Bercovich, A., Ronen, T., Abramovich, T., Ailon, N., Assaf, N., Dabbah, M., Galil,
I., Geifman, A., Geifman, Y., Golan, I., et al.: Puzzle: Distillation-based nas for
inference-optimized llms. arXiv preprint arXiv:2411.19146 (2024)
8. Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al.: Piqa: Reasoning about physical
commonsense in natural language. In: Proceedings of the AAAI conference on
artificial intelligence. vol. 34, pp. 7432–7439 (2020)
9. Chang, Y., Cui, L., Dong, L., Huang, S., Huang, Y., Huang, Y., Li, S., Lv, T., Ma,
S., Sun, Q., et al.: Redstone: Curating general, code, math, and qa data for large
language models. arXiv preprint arXiv:2412.03398 (2024)
10. Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H.P.D.O., Kaplan, J., Edwards,
H., Burda, Y., Joseph, N., Brockman, G., et al.: Evaluating large language models
trained on code. arXiv preprint arXiv:2107.03374 (2021)
11. Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T.,
Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., et al.: Rethinking attention with
performers. arXiv preprint arXiv:2009.14794 (2020)




--- end of page=12 ---

14 Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi


12. Clark, C., Lee, K., Chang, M.W., Kwiatkowski, T., Collins, M., Toutanova,
K.: Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv
preprint arXiv:1905.10044 (2019)
13. Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., Tafjord,
O.: Think you have solved question answering? try arc, the ai2 reasoning challenge.
arXiv preprint arXiv:1803.05457 (2018)
14. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert,
M., Tworek, J., Hilton, J., Nakano, R., et al.: Training verifiers to solve math word
problems. arXiv preprint arXiv:2110.14168 (2021)
15. Dao, T., Gu, A.: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060
(2024)
16. De, S., Smith, S.L., Fernando, A., Botev, A., Cristian-Muraru, G., Gu, A., Haroun,
R., Berrada, L., Chen, Y., Srinivasan, S., et al.: Griffin: Mixing gated linear
recurrences with local attention for efficient language models. arXiv preprint
arXiv:2402.19427 (2024)
17. Fu, D.Y., Dao, T., Saab, K.K., Thomas, A.W., Rudra, A., Ré, C.: Hungry hungry hippos: Towards language modeling with state space models. arXiv preprint
arXiv:2212.14052 (2022)
18. Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C.,
Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N.,
Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L.,
Tang, E., Thite, A., Wang, B., Wang, K., Zou, A.: The language model evaluation harness (07 2024). `[https://doi.org/10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602)`, `[https:](https://zenodo.org/records/12608602)`
```
  //zenodo.org/records/12608602
```

19. Glorioso, P., Anthony, Q., Tokpanov, Y., Whittington, J., Pilault, J., Ibrahim,
A., Millidge, B.: Zamba: A compact 7b ssm hybrid model. arXiv preprint
arXiv:2405.16712 (2024)
20. Goldstein, D., Alcaide, E., Lu, J., Cheah, E.: Radlads: Rapid attention distillation
to linear attention decoders at scale. arXiv preprint arXiv:2505.03005 (2025)
21. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models.
arXiv preprint arXiv:2407.21783 (2024)
22. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state
spaces. In: First conference on language modeling (2024)
23. Gu, A., Goel, K., Ré, C.: Efficiently modeling long sequences with structured state
spaces. arXiv preprint arXiv:2111.00396 (2021)
24. Gu, Y., Dong, L., Wei, F., Huang, M.: Minillm: Knowledge distillation of large
language models. arXiv preprint arXiv:2306.08543 (2023)
25. Gu, Y., Hu, Q., Yang, S., Xi, H., Chen, J., Han, S., Cai, H.: Jet-nemotron:
Efficient language model with post neural architecture search. arXiv preprint
arXiv:2508.15884 (2025)
26. Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., Steinhardt, J.: Measuring massive multitask language understanding. arXiv preprint
arXiv:2009.03300 (2020)
27. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network.
arXiv preprint arXiv:1503.02531 (2015)
28. Jin, Q., Dhingra, B., Liu, Z., Cohen, W., Lu, X.: Pubmedqa: A dataset for biomedical research question answering. In: Proceedings of the 2019 conference on empirical
methods in natural language processing and the 9th international joint conference

  - n natural language processing (EMNLP-IJCNLP). pp. 2567–2577 (2019)




--- end of page=13 ---

Efficient Task-Specific Hybrid Attention Model Construction 15


29. Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.: Transformers are rnns: Fast
autoregressive transformers with linear attention. In: International conference on
machine learning. pp. 5156–5165. PMLR (2020)
30. Lenz, B., Lieber, O., Arazi, A., Bergman, A., Manevich, A., Peleg, B., Aviram,
B., Almagor, C., Fridman, C., Padnos, D., et al.: Jamba: Hybrid transformermamba language models. In: The Thirteenth International Conference on Learning
Representations (2025)
31. Liu, Y., Yu, J., Xu, Y., Li, Z., Zhu, Q.: A survey on transformer context extension:
Approaches and evaluation. arXiv preprint arXiv:2503.13299 (2025)
32. Mihaylov, T., Clark, P., Khot, T., Sabharwal, A.: Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint
arXiv:1809.02789 (2018)
33. Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S.,
Cao, H., Cheng, X., Chung, M., Grella, M., et al.: Rwkv: Reinventing rnns for the
transformer era. arXiv preprint arXiv:2305.13048 (2023)
34. Ren, L., Liu, Y., Lu, Y., Shen, Y., Liang, C., Chen, W.: Samba: Simple hybrid state
space models for efficient unlimited context language modeling. arXiv preprint
arXiv:2406.07522 (2024)
35. Sakaguchi, K., Bras, R.L., Bhagavatula, C., Choi, Y.: Winogrande: An adversarial
winograd schema challenge at scale. Communications of the ACM **64** (9), 99–106
(2021)
36. Su, D., Kong, K., Lin, Y., Jennings, J., Norick, B., Kliegl, M., Patwary, M., Shoeybi,
M., Catanzaro, B.: Nemotron-cc: Transforming common crawl into a refined longhorizon pretraining dataset. In: Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). pp. 2459–
2475 (2025)
37. Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., Wei, F.: Retentive
network: A successor to transformer for large language models. arXiv preprint
arXiv:2307.08621 (2023)
38. Talmor, A., Herzig, J., Lourie, N., Berant, J.: Commonsenseqa: A question answering challenge targeting commonsense knowledge. In: Proceedings of the 2019
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).
pp. 4149–4158 (2019)
39. Team, Q., et al.: Qwen2 technical report. arXiv preprint arXiv:2407.10671 **2** (3)
(2024)
40. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,
Ł., Polosukhin, I., et al.: Attention is all you need. Advances in neural information
processing systems **30** (1), 261–272 (2017)
41. Waleffe, R., Byeon, W., Riach, D., Norick, B., Korthikanti, V., Dao, T., Gu, A.,
Hatamizadeh, A., Singh, S., Narayanan, D., et al.: An empirical study of mambabased language models. arXiv preprint arXiv:2406.07887 (2024)
42. Yang, M., Rezagholizadeh, M., Li, G., Appia, V., Barsoum, E.: Zebra-llama: Towards extremely efficient hybrid models. arXiv preprint arXiv:2505.17272 (2025)
43. Yang, S., Kautz, J., Hatamizadeh, A.: Gated delta networks: Improving mamba2
with delta rule. arXiv preprint arXiv:2412.06464 (2024)
44. Yang, S., Wang, B., Shen, Y., Panda, R., Kim, Y.: Gated linear attention transformers with hardware-efficient training. In: International Conference on Machine
Learning. pp. 56501–56523. PMLR (2024)
45. Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., Choi, Y.: Hellaswag: Can a machine
really finish your sentence? arXiv preprint arXiv:1905.07830 (2019)




--- end of page=14 ---
