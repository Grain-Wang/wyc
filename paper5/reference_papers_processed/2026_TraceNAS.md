---
id: "2026_TraceNAS"
title: "TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation"
authors: ["Prajna G. Malettira", "Manish Nagaraj", "Arjun Roy", "Shubham Negi", "Kaushik Roy"]
year: 2026
venue: "arXiv"
publication_status: "PREPRINT"
category: "Direct Neighbor / Architecture Interaction"
source_pdf: "../reference_papers_origin/2026_TraceNAS.pdf"
paper_url: "https://arxiv.org/abs/2602.02891"
pdf_url: "https://arxiv.org/pdf/2602.02891"
code_url: "UNVERIFIED"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation

**Authors:** Prajna G. Malettira, Manish Nagaraj, Arjun Roy, Shubham Negi, Kaushik Roy

**Venue / Year:** arXiv (PREPRINT)

**Category:** Direct Neighbor / Architecture Interaction

**Why this paper matters for low-cost post-training LLM NAS:** It proposes a global zero-shot loss-landscape alignment score for joint depth-width LLM search and directly challenges purely local importance proxies.

**Primary record:** [https://arxiv.org/abs/2602.02891](https://arxiv.org/abs/2602.02891)

**Local source:** [2026_TraceNAS.pdf](../reference_papers_origin/2026_TraceNAS.pdf)

## Full converted text

## **TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**

**Prajna G. Malettira** [1] **Manish Nagaraj** [1] **Arjun Roy** [1] **Shubham Negi** [1 2] _[ ‡]_ **Kaushik Roy** [1]



## Abstract


Structured pruning is essential for efficient deployment of Large Language Models (LLMs). The
varying sensitivity of LLM sub-blocks to pruning necessitates the identification of optimal nonuniformly pruned models. Existing methods evaluate the importance of layers, attention heads, or
weight channels in isolation. Such localized focus
ignores the complex global structural dependencies that exist across the model. Training-aware
structured pruning addresses global dependencies,
but its computational cost can be just as expensive as post-pruning training. To alleviate the
computational burden of training-aware pruning
and capture global structural dependencies, we
propose TraceNAS, a training-free Neural Architecture Search (NAS) framework that jointly explores structured pruning of LLM depth and width.
TraceNAS identifies pruned models that maintain
a high degree of loss landscape alignment with
the pretrained model using a scale-invariant zeroshot proxy, effectively selecting models that exhibit maximal performance potential during postpruning training. TraceNAS is highly efficient,
enabling high-fidelity discovery of pruned models

   - n a single GPU in 8.5 hours, yielding a 10 _×_ reduction in GPU-hours compared to training-aware
methods. Evaluations on the Llama and Qwen
families demonstrate that TraceNAS is competitive with training-aware baselines across commonsense and reasoning benchmarks.


**1** **Introduction**


Recent breakthroughs in Large Language Models
(LLMs) (Anthropic, 2023; OpenAI, 2023) have delivered
unprecedented capabilities but require massive scale
in parameters and computation. Foundational scaling
laws (Kaplan et al., 2020) have established that smaller


1Purdue University 2Google _‡_ Work done while at Purdue University. Correspondence to: Prajna G. Malettira
_<_ pmaletti@purdue.edu _>_ .


_Preprint. February 4, 2026._



|64<br>62<br>60<br>58<br>0 10|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|0<br>10<br>58<br>60<br>62<br>64||||||||
|0<br>10<br>58<br>60<br>62<br>64|||||CPT|Tokens(Bub<br>1B Tok|ble Size)<br> ens|
|0<br>10<br>58<br>60<br>62<br>64||||||20B To<br>~~100B T~~|kens<br>~~ okens~~|
|0<br>10<br>58<br>60<br>62<br>64||20<br>30<br>40<br>5|20<br>30<br>40<br>5|20<br>30<br>40<br>5|20<br>30<br>40<br>5|0<br>60|70|


### Figure 1

Caption: _ **Search Efficiency.** TraceNAS identifies optimal nonuniform architectures in 8.5 GPU hours [3], achieving competitive

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
accuracy with 10 _×_ less search overhead than training-aware baselines. The area of the bubble is proportional to the total tokens
used for recovery training, highlighting that TraceNAS identified
architectures have high recovery potential.


LMs are less sample-efficient, and thus training them
from scratch is a resource-intensive bottleneck. Model

compression enables the realization of efficient models
that preserve the extensive knowledge of the pretrained
base. Techniques such as quantization (Frantar et al., 2022;
Liu et al., 2024), distillation (Sarah et al., 2024; Li & Jin,
2022) and pruning (Xia et al., 2023; Sun et al., 2023) have
become essential, allowing for the accessible deployment

- f high-performance architectures. Among these, structured
pruning (Ma et al., 2023) aims to remove architectural units
like attention heads, weight columns, or transformer blocks
to achieve immediate hardware speedup. This process can
be naturally formulated as a Neural Architecture Search
(NAS) (Sieberling et al.; Tao et al., 2023; Tang et al., 2025)
problem, where the pretrained LLM is conceptualized as a
super-network (Yu et al., 2020; Cai et al., 2019) that serves
as the search space.


Structured pruning is inherently more disruptive than the
localized masking used in unstructured pruning (Sun et al.,
2023; An et al., 2024; Frantar & Alistarh, 2023). While
unstructured methods leave the model’s architecture intact,
structured pruning physically alters it and risks severing
critical pathways in the representational flow. The primary
bottleneck in adapting unstructured methods to structured
pruning is the lack of a comparable global metric across
the model’s layers (Ma et al., 2023). Local importance
heuristics such as activation-weighted magnitudes (Sun
et al., 2023; An et al., 2024) or Hessian-based reconstruc

3TraceNAS, ShearedLlama and Flextron report search time on
a NVIDIA A100, DarwinLM on L40S and E [3] reports NPU hours.







1




--- end of page=0 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**



tions (Frantar & Alistarh, 2023) identify redundancy within
isolated layers and are often incomparable across model
depth due to varying magnitude scales (Ma et al., 2023).
We address this limitation by utilizing gradients (Das et al.,
2023; Yang et al., 2025) as global signals that naturally
capture the complex inter-dependencies across the model.


Previous search methods for structured pruning employ
mask learning (Xia et al., 2023; Yuan et al., 2025a) or
training-aware search (Tang et al., 2025; Bercovich et al.,
2025). While these training-aware approaches define the
state-of-the-art, they demand substantial computational resources. This overhead stems from the iterative gradient
updates required for _training-aware_ model discovery and
also necessitates a continuous stream of unseen tokens to

prevent pruned models from overfitting to calibration data.
This cost can make the search for an efficient model just as
demanding as post-pruning recovery, motivating the need
for _training-free NAS_ (Wu et al., 2021; Tran et al., 2021;
Ingolfsson et al., 2022) using zero-shot performance proxies.
However, existing proxies are primarily designed to identify trainable, randomly initialized architectures by focusing

- n heuristics such as gradient stability (Abdelfattah et al.,
2021) or loss landscape smoothness (Li et al., 2023). These
metrics are insufficient for LLM pruning as they prioritize
trainability, the ability of a network to learn from scratch.
In contrast, the performance of pruned LLMs depends on
inheritance, the ability of the pruned model to recover performance by preserving the pretrained loss landscape (Frankle
& Carbin, 2018; Frankle et al., 2020; Chen et al., 2020).


In this paper, we introduce TraceNAS, a training-free NAS
framework for joint depth and width pruning of LLMs. Our
approach leverages the observation that a pretrained model
resides in broad, stable regions of the loss landscape (Frankle et al., 2020; Li et al., 2018). We propose a zero-shot
proxy that utilizes gradient trace to capture the impact of
structural pruning on model sensitivity. Specifically, the
proxy measures the sparsity-weighted aggregate over the
Pearson Correlation coefficients between the gradient traces

- f the pruned and pretrained base models. Since structured
pruning alters activation and gradient magnitudes, the use

- f Pearson Correlation ensures scale invariance, thus capturing the alignment of gradient traces on the pretrained loss
landscape independently of scale shifts.


To ensure the search remains tractable across varying model
scales, we compute these gradients within a low-rank (Hu
et al., 2022; Zhao et al., 2024) subspace. By leveraging
the intrinsic low-dimensionality (Aghajanyan et al., 2021)

- f LLMs, TraceNAS captures the principal gradient components of the model while reducing memory overhead by

- rders of magnitude. Unlike training-aware methods which
require significant memory overhead to maintain optimizer
states and activations for backpropagation across a large



set of calibration tokens, our training-free approach only
requires the gradients of each candidate computed on a minimal calibration set. This reduction in memory consumption
enables high-fidelity model discovery (Fig. 1), bypassing the
compute intensive requirements typical of training-aware
NAS. The primary contributions of this work are:


1. **Scalable Training-free NAS Framework:** We introduce TraceNAS, a unified pruning framework (Fig. 2)
for the joint optimization of LLM depth and width.
Our approach enables the zero-shot discovery of nonuniform pruned models, facilitating efficient compression across model scales without search-time training.
2. **Zero-shot Proxy for Functional Inheritance:** We
introduce a scale-invariant, gradient-based zero-shot
proxy that evaluates gradient alignment between the
pruned and pretrained models. Our proxy achieves
superior Spearman Rho (Spearman, 1961) ( _ρ_ = 0 _._ 94)
and Kendall Tau (Kendall, 1938) ( _τ_ = 0 _._ 82) correlations with downstream accuracy, enabling accurate
zero-shot discovery.
3. **Efficient Architecture Discovery:** We show that TraceNAS achieves a 10 _×_ reduction in both GPU-hours

and total calibration data compared to training-aware
methods (Fig. 1). This drastically lowers the overhead
for large-scale model compression by ensuring search
costs are significantly lower than recovery training.


**2** **Related Work**


**2.1** **Language Model (LM) Pruning**


LLM pruning methods vary by granularity: unstructured
weight sparsification (Frantar & Alistarh, 2023) or structured removal (Ma et al., 2023; Kim et al., 2024) of transformer blocks, attention heads or weight channels. These
methods are further categorized by their computational
complexity relative to the model’s hidden dimension _d_ .
Early _O_ ( _d_ ) magnitude pruning (Han et al., 2015) meth
- ds evaluate weights in isolation, ignoring the outlier features characteristic of LLMs (Kovaleva et al., 2021; Luo
et al., 2021; Yin et al., 2023). To preserve these outliers,
_O_ ( _d_ [2] ) metrics like Wanda (Sun et al., 2023) and FLAP (An
et al., 2024) utilize weight-activation products. Furthermore,
_O_ ( _d_ [3] ) optimization-based methods like SparseGPT (Frantar
& Alistarh, 2023) and LLM-Pruner (Ma et al., 2023) utilize
second-order inverse Hessians (Hassibi et al., 1993) to minimize layer-wise reconstruction error. However, these local
heuristics prioritize layer properties and fail to account for
how pruning-induced errors propagate through the model.


**2.2** **Influence-based Importance**


Beyond these heuristics, recent research has explored influence functions (IF) (Koh & Liang, 2017; Kwon et al.,
2023) to better capture how model perturbations affect sen


2




--- end of page=1 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**



sitivity. For instance, LayerIF (Askari et al., 2025) uses
IF to estimate layer importance; however, it remains locally heuristic as it perturbs layers individually to measure
validation loss sensitivity. This inherently fails to capture
the joint impact of multi-layer and width pruning on the
model’s representation. Furthermore, the _O_ ( _d_ [3] ) complexity

- f standard IF is prohibitive for LLMs, requiring inverse
Hessian computations for all model parameters. In contrast,
TraceNAS adopts an efficient first-order gradient-tracing
approach inspired by TracIn (Pruthi et al., 2020) to evaluate
the global impact of pruning. By operating with _O_ ( _d_ [2] ) complexity, TraceNAS identifies non-uniform pruned architectures that maintain functional alignment with the pretrained
base model, bypassing the computational bottlenecks and local limitations of LayerIF. We provide a detailed comparison
between IF, IF-based pruning and TraceNAS in B.1.


**2.3** **Training-Aware LM Pruning**


Recent structured pruning methods, such as ShearedLlama (Xia et al., 2023) and E [3] (Yuan et al., 2025b), use
mask learning via _L_ 0 regularization and differential mask

- ptimization. While effective, these methods target uniform
sparsity across layers, failing to account for the heterogeneous importance distribution across a model’s width (Tao
et al., 2023). This has prompted a shift toward discovering
non-uniform architectures via evolution search. Methods

like SIMPLE (Tao et al., 2023) and EvoPress (Sieberling
et al.) employ evolutionary optimization for heterogeneous
discovery. On the other hand, DarwinLM (Tang et al., 2025)
uses a curriculum-style training-aware search, performing
lightweight fine-tuning for every searched candidate. Similarly, PUZZLE (Bercovich et al., 2025) utilizes distillationbased NAS to identify optimal configurations by minimizing block-wise KL divergence. However, the training-aware
search inherent to these methods mean their model discovery
can be just as computationally intensive as the post-pruning
recovery training.


**2.4** **Zero-shot NAS for LM Pruning**


As models scale, these training-intensive searches become
intractable, necessitating training-free solutions for NAS.
Existing proxies are formulated to identify trainable, randomly initialized architectures by focusing on heuristics
such as gradient stability (Abdelfattah et al., 2021), model
expressivity (Mellor et al., 2021; Jiang et al., 2023), or loss
landscape smoothness (Li et al., 2023). These proxies are
insufficient for evaluating pruned models as they do not
account for how well a model inherits pretrained knowledge. To address this, recent works LPZero (Dong et al.,
2024a) and Pruner-Zero (Dong et al., 2024b) formulate zeroshot pruned model discovery as a two-fold problem: first
searching for unique symbolic pruning metrics for the target
pretrained model, and subsequently applying these metrics
to discover the optimal structured (Dong et al., 2024a) or un


structured (Dong et al., 2024b) sub-network. While modular,
the dual-stage process adds overhead to the search pipeline.
In contrast, TraceNAS introduces a unified training-free
fitness proxy that directly evaluates a compressed model’s
functional alignment with its pretrained base, streamlining
the search for pruned models that have high performance
potential and fast convergence during post-pruning recov
ery.


**3** **Methodology**


**3.1** **Problem Definition**


We formulate pruned model discovery as a constrained discrete search over a pretrained super-network, _Mbase_ (Cai
et al., 2019; Yu et al., 2020). We seek an optimal _M_ [ˆ] _sub ∈G_
that maximizes a training-free proxy Φ( _·, ·_ ), that measures
functional inheritance; the ability of _Msub_ to recover performance by preserving the gradient direction of _Mbase_ - n
it’s loss landscape:

ˆ
_Msub_ = arg max Φ( _Msub, Mbase_ ) s.t. _P_ ( _Msub_ ) _≤_ _C,_
_M∈G_


where _G_ is the search space, _P_ ( _·_ ) denotes the parameter
count, and _C_ is the target constraint. By maximizing Φ,
we identify models with maximal potential for performance
recovery during post-pruning _continued pretraining_ (CPT)
without the overhead of search-time training.


**3.2** **Search Space Encoding**


The search space _G_ is defined as a joint distribution over the
model’s depth and sub-block-wise width, enabling the discovery of non-uniformly pruned architectures from _Mbase_ .
By operating at sub-block granularity, the TraceNAS search
framework treats attention and MLP modules as independent units, thereby capturing diverse structural sparsity.
**Architecture Encoding** A candidate _Msub_ is parameterized by a tuple ( **d** _,_ _**κ**_ ). The depth configuration **d** _∈{_ 0 _,_ 1 _}_ _[L]_

is a discrete mask over the _L_ blocks of _Mbase_ . Setting
_dl_ = 0 deactivates the _l_ - th block while preserving the
residual stream, allowing the search to explore varying
model depths. We enforce a minimum depth constraint,

- _dl ≥_ _Lmin_, to ensure sufficient computational capacity
for sequential reasoning. Simultaneously, the width configuration _**κ**_ = _{κ_ 1 _, . . ., κL}_ where _κl_ = ( _rattn_ [(] _[l]_ [)] _[, r]_ _mlp_ [(] _[l]_ [)] [)] _[ ∈]_ [(0] _[,]_ [ 1]]
defines the parameter retention ratio for each attention and
MLP sub-block. By decoupling the sparsity ratios, our
framework identifies heterogeneous architectures that distinctly prioritize retention in sensitive sub-modules while
respecting the global budget _C_ .


**3.3** **In-Place Architectural Realization**


We realize candidate sub-networks _Msub_ by mapping sparsity ratios _**κ**_ to binary masks and implementing them via an
efficient in-place masking strategy.



3




--- end of page=2 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


### Figure 2

Caption: _ **Visualization of the TraceNAS search framework.** TraceNAS uses a gradient-based, training-free proxy to guide structural

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
pruning. Following a one-time initialization of base gradient traces ( _g_ base) and importance scores ( _Il_ ), a population of depth ( **d** ) and
width ( _**κ**_ ) candidates undergoes iterative evolution via crossover and mutation. Each width configuration is realized using an _O_ ( _d_ [2] )
activation-weighted heuristic. Subsequently, candidates _Msub_ are ranked by the zero-shot proxy Φ, which measures the gradient trace
alignment between the active layers of _Msub_ relative to _Mbase_ .



**Width Mask Generation** To realize a layer’s width configuration _κl_ into a weight mask, we use the activationweighted product: _I_ ( _κl_ ) _l,j_ = [�] _i_ _[|][W][l,]_ [(] _[ij]_ [)] _[| ⊙∥][X][l,j][∥]_ [2][ (][Sun]

et al., 2023) as a saliency metric. This approach explicitly
accounts for the outlier features characteristic of LLMs (Kovaleva et al., 2021; Luo et al., 2021; Yin et al., 2023) by
scaling weight magnitudes by the _L_ 2-norm of their corresponding input activations. By identifying channels that
are functionally critical to the model’s representational flow,
_I_ ( _κl_ ) _l,j_ ensures that mask realizations are tailored to preserve the underlying representation of _Mbase_ .


Adopting this _O_ ( _d_ [2] ) heuristic avoids the _O_ ( _d_ [3] ) bottleneck

- f second-order reconstruction methods (Frantar & Alistarh,
2023; Ma et al., 2023; Tang et al., 2025), enabling the highfrequency scoring and realization of hundreds of distinct
candidates during the evolutionary search. Crucially, while
_I_ ( _κl_ ) _l,j_ provides a local channel-selection heuristic, the
global distribution of the sparsity ratios _κl_ is governed by
the TraceNAS proxy Φ (Sec. 3.4), which accounts for the
inter-layer dependencies and impact of pruning that local
heuristics ignore.
**In-Place Masking** To score pruned candidates, we use
an in-place masking strategy that operates directly on the
weight matrices of _Mbase_ . This eliminates the need for instantiating the population of candidate models during each
search iteration. For depth exploration, _Mbase_ is modified
according to _dl ∈_ **d** ; if _dl_ = 0, the entire block is bypassed
and activations are routed through the residual connection,
ensuring functional continuity (Veit et al., 2016; Gromov
et al., 2024). For active blocks ( _dl_ = 1) the binary masks
are applied via a temporary modify-then-restore pointer:
_Wbase_ _[′]_ [=] _[ W][base][ ⊙]_ [Mask][(] _[κ][l]_ [)][. This allows the search frame-]
work to execute forward and backward passes over a calibration set to capture the cumulative effects of pruning without



the memory overhead of instantiating _Msub_ . Once the gradient trace is recorded for scoring, the original weights are
restored, preserving _Mbase_ for subsequent search iterations.


**3.4** **TraceNAS: Evaluating Functional Alignment**


To evaluate the functional integrity of realized candidates
_Msub_, we propose the proxy Φ. Structured pruning induces a significant representational shift that manifests as
an immediate drop in performance (Tran et al., 2022). This
initial shock obscures the distinction between fundamen
tal structural collapse and architectures that preserve the
essential representational pathways required for effective
post-pruning recovery.


Furthermore, as discussed in Sec. 2.1, importance-based
metrics (Sun et al., 2023; An et al., 2024) prioritize local
weight properties while overlooking the structural interdependencies captured by gradients (Das et al., 2023). To
address this, we define Φ as a zero-shot, gradient-based
proxy that quantifies functional inheritance. Φ measures the
directional alignment between the gradient trace of _Msub_
and _Mbase_ . This directional alignment allows Φ to identify
models that are favorably positioned within the pretrained
convergence basin, thereby enhancing their potential for
functional recovery.
**Establishing the Functional Anchor** We define the optimal functional state as the gradient trace _gbase_ - f the pretrained model _Mbase_ . Using a calibration set _B_, _gbase_ provides the directional anchor within the optimization landscape that sub-networks _Msub_ must align with to ensure
functional inheritance. The gradient traces for the base and
candidate models are calculated as


_g_ = E _b∈B_ [ _∇θL_ ( _M_ ( _b_ ; _θ_ ))]


where _θ_ denotes the trainable model parameters.



4




--- end of page=3 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**



**Low-Rank Gradient Manifold** To ensure computationally tractability across model sizes, we compute gradient
traces _g_ within a low-rank subspace. Since storing full-rank
LLM gradients is memory-prohibitive, we attach Low-Rank
Adapters (LoRA) (Hu et al., 2022) to all linear projections.
This approach leverages the observation that the functional
knowledge of LLMs resides within a low-dimensional manifold (Aghajanyan et al., 2021; Zhao et al., 2024), allowing us
to capture global dependencies while significantly reducing
the memory footprint of gradient storage.
**Measuring Functional Alignment** To quantify how effectively _Msub_ inherits the functional state of _Mbase_, we
calculate the Pearson Correlation Coefficient _ρ_ [(] _[l]_ [)] per layer
_l ∈_ _L_ . For each transformer-block _l_, _ρ_ [(] _[l]_ [)] measures the alignment between the low-rank gradient traces of the pruned
attention and MLP sub-blocks:





 _,_



�



 _gbase_ [(] _[l]_ [)] _[−]_ _[µ]_ _gbase_ [(] _[l]_ [)]


_σ_ ( _l_ )
 _gbase_



_ρ_ [(] _[l]_ [)] = [1]

_Nl_



_gsub_ [(] _[l]_ [)] _[−]_ _[µ]_ _gsub_ [(] _[l]_ [)]

_σ_ ( _l_ )

- []  _gsub_



(1)



crossover and multiplicative Gaussian jitter-based mutation.
This joint evolution allows TraceNAS to explore the architectural trade-offs between depth and width while adhering
to the constraint _C_ .

**Search Granularity** The activations used for mask generation are obtained at the attention output ( _Wo_ ) and MLP
down ( _Wdown_ ) projection layers. These projections consolidate sub-block representations, and their input activations serve as proxies for the importance of preceding upstream weights. Once generated, these masks are applied
to the corresponding projections: _Wq, Wk, Wv_ for attention
and _Wup, Wgate_ for MLP. For Grouped-Query Attention
(GQA) (Ainslie et al., 2023) models, we do not prune _Wk_
and _Wv_ to ensure KV-cache compatibility. Furthermore,
pruned MLP hidden dimensions are rounded to multiples of
_m_ = 32 to maintain efficient tensor operations.


**3.6** **Interpreting Functional Inheritance through**
**Gradient Trace Alignment**


The effectiveness of Φ as a proxy for functional inheritance
is driven by three key observations:


  - **Manifold Anchoring:** High-performing pretrained
models reside within flat, stable convergence
basins (Frankle et al., 2020; Li et al., 2018). A high
alignment score Φ implies that the first-order Taylor
expansion of the sub-network’s loss surface remains
congruent with that of the base model. By preserving
the gradient’s directional signature, _Msub_ remains anchored within the original functional manifold. This
ensures that post-pruning recovery initiates from a region of high directional certainty, enabling a stable loss
trajectory rather than necessitating a costly search for
a new local minimum.

  - **Global Sensitivity via Chain-Rule:** Unlike metrics
that evaluate layers in isolation (Sun et al., 2023; Askari
et al., 2025), _gsub_ is computed via backpropagation
through the masked computational graph. This captures inter-layer disruptions: if an upstream block bottlenecks the representational flow, subsequent layer
gradient traces will de-correlate from the base model

   - ptimization manifold. This causes a sharp drop in
Φ, allowing the proxy to detect structural incoherence
and broken residual streams that local heuristics are

inherently blind to.

  - **Intrinsic Dimensionality Invariance:** Empirical evidence shows that gradient alignment is largely rankinvariant (Fig. 3c, Appendix C.3). This indicates that
Φ can capture a pruned model’s functional capability
within a low-dimensional (Aghajanyan et al., 2021)
manifold. This robustness ensures that the relative performance ranking of candidates remains stable across
gradient ranks (Zhao et al., 2024), justifying the use of
low-rank subspaces as a reliable and scalable surrogate



where _Nl_ is the dimensionality of the low-rank subspace,
and _µ, σ_ denote the mean and standard deviation of the
gradient elements. By standardizing the traces, we decouple
the directional alignment from magnitude shifts inherently
induced by structural pruning.
**Sparsity-Weighted Aggregation** We formulate the final proxy Φ by aggregating sub-block correlations using
a sparsity-weighted summation. While gradient standardization ensures local scale-invariance, a uniform average

- f _ρ_ [(] _[l]_ [)] across layers fails to account for the heterogeneous
representation capacity of _Msub_ . We therefore weight each
correlation coefficient by its corresponding retention ratio
_r_ [(] _[l]_ [)] _∈_ _κl_ :




- _rattn_ [(] _[l]_ [)] _[·][ρ]_ [(] _[l]_ [)][+]
_l∈_ Attn _l∈_



Φ( _Msub, Mbase_ ) =


_r_ [(] _[l]_ [)]

- _mlp_ _[·][ρ]_ [(] _[l]_ [)][ (2)]

_l∈_ MLP



This formulation anchors the global score in the highcapacity regions of _Msub_, prioritizing the sub-blocks that
serve as the primary repositories of functional inheritance.
By weighting _ρ_ [(] _[l]_ [)] by parameter density, we prevent Φ from
being skewed by the high-variance noise typical of aggressively pruned sub-blocks.


**3.5** **Selection & Evolutionary Mechanics**


To navigate the discrete search space _G_, we employ an evolutionary strategy driven by the proxy Φ. The initial search
population _P_ 0 is generated using a global importance prior;
detailed initialization procedures and a search convergence
analysis are provided in Appendix A.
**Search Space Evolution** We employ a hybrid evolution
strategy for the joint ( **d** _,_ _**κ**_ ) search space. Depth evolution
explores discrete block configurations via contiguous-range
crossover and bit-flip mutations. Simultaneously, width evolution explores continuous sparsity ratios using interpolation



5




--- end of page=4 ---

for full-rank analysis.


**4** **Experiments**


**4.1** **Experimental Setup**



**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


**4.2** **TraceNAS Proxy Validation: Performance**
**Correlation and Stability**



**Models and Datasets:** We evaluate TraceNAS across mul
tiple scales using Llama-2-7B (Touvron et al., 2023), Llama3.1-8B (Dubey et al., 2024) and Qwen-2.5-14B, covering
both Multi-Head Attention (MHA) (Vaswani et al., 2017)
and Grouped-Query Attention (GQA) (Ainslie et al., 2023)
architectures. We use the FineWeb-Edu (Penedo et al., 2024)
100BT dataset for both evolutionary search and post-pruning
CPT. For the search, we use a fixed calibration set of 65,536
tokens (16 sequences) per pruned candidate to evaluate a
total population of 1500 candidates. This is followed by
20B tokens of post-pruning CPT. Results on additional models and CPT token budgets is provided in Appendix D.2
and D.3.

**Implementation Details** The evolutionary search is performed with a population of 30 candidates over 50 search
iterations. On a single NVIDIA H200 GPU, the search concludes in _∼_ 2 hours, a significant acceleration over the 8.5hour A100 baseline reported in Fig. 1. The highest scored
sub-networks then undergo CPT on a cluster of six H200

_∼_
GPUs ( 16 hours) using a global batch size of 1024 and
peak learning rate of 1 _e_ _[−]_ [4] . We use a Warmup-Stable-Decay
(WSD) (Hu et al., 2024) scheduler, with context lengths set
to 4096 for Llama-2 and Qwen-2.5, and 8192 for Llama-3.1.
Detailed hyperparameters are available in Appendix A.3.
**Baselines:** We benchmark TraceNAS against leading
structured and layer-pruning methods. This includes, masklearning methods, ShearedLLaMA (Xia et al., 2023) and
_E_ [3] - Pruner (Yuan et al., 2025a), which employ Lagrangian

- ptimization. For training-aware pruning, we compare
against DarwinLM (Tang et al., 2025). We further evaluate against Minitron (Sreenivas et al., 2024), Flextron (Cai
et al., 2024), LoRAP (Li et al., 2024) and uniformly pruned
models found via TraceNAS to isolate the benefits of nonuniform pruning. Comparisons with unstructured pruning
methods like Wanda (Sun et al., 2023), FLAP (An et al.,
2024) and PrunerZero (Dong et al., 2024b) are included in
Appendix D.4.
**Evaluation Benchmarks:** Models are evaluated using
the lm-evaluation-harness (Gao et al., 2024)
across 0-shot commonsense reasoning benchmarks: ARCeasy (Clark et al., 2018), LogiQA (Liu et al., 2020),
PIQA (Bisk et al., 2020), SciQ (Welbl et al., 2017)
and BoolQ (Clark et al., 2019). We also report 5-shot
performance on MMLU (Hendrycks et al., 2020) and
WinoGrande (Sakaguchi et al., 2020), 10-shot on HellaSwag (Zellers et al., 2019), and 25-shot on ARC Challenge (Clark et al., 2018).



To evaluate TraceNAS as a proxy for recovery potential and
verify its consistency across hyperparameters, we analyze
70 candidate models sampled from the AmoebaLLM (Fu
et al., 2024) search space. This evaluates the proxy in an
environment unbiased by our specific evolutionary search

- r large-scale CPT. We compute Spearman Rho (SP) and
Kendall Tau (KT) correlations against Wikitext-2 perplexity
(PPL), MMLU accuracy and average downstream accuracy
after Once-for-All (OFA) (Cai et al., 2019; Sukthanker et al.)
finetuning on Alpaca (Taori et al., 2023).


_Table 1._ Correlation of zero-shot proxies with model performance
calculated over 70 pruned models and averaged over 3 seeds. We
report Spearman Rho (SP) and Kendall Tau (KT) for perplexity
(PPL), MMLU and average commonsense reasoning accuracies.
Best correlation values per column are **bolded** and underlined
values denote second best correlation.


Method PPL MMLU Avg. Acc.

SP KT SP KT SP KT


#Params 0.87 0.72 0.11 0.07 0.66 0.50
NASWOT 0.69 0.50 0.35 0.26 0.39 0.29
ZiCo 0.87 0.72 0.11 0.07 0.66 0.50
GradNorm **0.93 0.79** 0.45 0.32 0.93 0.78
Synaptic Saliency **0.93** 0.78 0.37 0.25 0.89 0.73
MeCo 0.92 **0.79** 0.03 0.02 0.75 0.58
PrunerZero 0.44 0.31 **0.88 0.69** 0.68 0.50


TraceNAS – Dot 0.57 0.46 0.35 0.25 0.58 0.47
TraceNAS – Cosine 0.91 0.77 0.48 0.35 0.91 0.79
TraceNAS – Unweighted 0.85 0.71 0.23 0.16 0.77 0.62
TraceNAS (Ours) **0.93 0.79** 0.54 0.39 **0.94 0.82**


**Correlation with Downstream Performance** TraceNAS

achieves superior ranking correlation with downstream performance, as shown in Table 1. Unlike trainability proxies
like NASWOT (Mellor et al., 2021) and ZiCo (Li et al.,
2023), which rank models based on their expressivity and
convergence capability from random initializations, our
proxy explicitly accounts for pretrained knowledge inheritance. By measuring the alignment between the pruned
and base model gradient traces, TraceNAS captures how
structural changes impact the functional sensitivity of the
model. This allows the proxy to account for complex reasoning dependencies that feature-based or loss-landscape
smoothness-based metrics fail to capture.


While GradNorm (Abdelfattah et al., 2021) and Synaptic
Saliency (Tanaka et al., 2020) effectively rank PPL, they pri
- ritize the health of the gradient signal and whether pruning
disproportionately impacts individual layers. This provides
an incomplete picture of the candidate’s representational integrity. Similarly, while MeCo (Jiang et al., 2023) performs
well on PPL, it fails to translate to task-specific accuracy.
PrunerZero (Dong et al., 2024b) achieves the highest correlation on MMLU by incorporating gradient information, but



6




--- end of page=5 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**







|1.0|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|||||||||
|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|||||||||
|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~||||||Samples (N)<br>Context (CL)<br>~~Rk ()~~|Samples (N)<br>Context (CL)<br>~~Rk ()~~|Samples (N)<br>Context (CL)<br>~~Rk ()~~|
|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|~~anr~~|~~anr~~|~~anr~~|~~anr~~|~~anr~~|~~anr~~|~~anr~~|~~anr~~|
|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|Hyperparameter<br>0.4<br>0.6<br>0.8<br>1.0<br>Kendall's<br>Samples (N)<br>Context (CL)<br>~~Rank (r)~~|~~anr~~|~~anr~~|~~anr~~|rpa|ramet|ramet|er|er|
|N Samp|es|4|8<br>|16<br>|16<br>|64<br>|64<br>|128<br>|256|
|CL Tokens|CL Tokens|128|256|512|512|1024|1024|2048|4096|
|Rankr|Rankr|2|8|64|64|256|256|1024|4096|


### Figure 3

Caption: _ **TraceNAS proxy stability analysis.** We report Kendall

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
_τ_ correlation between ranking scores across search hyperparameters (a) number of samples (N), (b) context length (CL), and (c)
LoRA rank ( _r_ ) which define the x-axis. High _τ_ values demonstrate
that TraceNAS consistently ranks models across search settings.


this success does not generalize to PPL or average downstream accuracy. Additionally, #Params has high ranking
potential for both PPL and average accuracy, however this
does not translate to a good measure of a model’s knowledge retention capability, which is needed for MMLU. This
tells us that while increasing model size is a good indicator of generalizability potential, #Params cannot capture
the nuances required to analyze a pruned model’s factual
knowledge and reasoning capabilities.


In contrast, TraceNAS maintains superior ranking stability
across all three metrics by leveraging Sparsity-Weighted
Pearson Correlation (Φ). Φ is specifically designed to isolate functional signals from directional noise through gradient centering. By using layer-wise sparsity as a weighting
factor, Φ anchors the global fitness score in high-capacity
regions. This prevents the ranking from being skewed by
the high variance noise typical of aggressively pruned subblocks. We further validate the choice of our proxy by
comparing against alternative alignment metrics, including
dot product, cosine similarity and unweighted aggregation

- f Pearson Correlation Coefficients. A detailed analysis of
the correlation results is presented in Table 1 is provided in
Appendix C.
**Hyperparameter Stability Analysis** We evaluate the robustness of TraceNAS by computing the Kendall _τ_ correlation between various search hyperparameters and the highest
setting in each category ( _N_ = 256, _CL_ = 4096, _r_ = 4096).
As shown in Fig. 3, the proxy maintains high ranking consistency across all variables, confirming that gradient trace
alignment effectively captures model inheritance even under
constrained search settings. Our analysis of these internal
ranking correlations follows:


  - **Number of Calibration Samples (** _N_ **):** We observe
that correlation monotonically align with the _N_ = 256
configuration as sample density increases. While ranking sensitivity is present at _N_ = 4 with _τ ≈_ 0 _._ 57, stability improves significantly to _τ ≈_ 0 _._ 71 at _N_ = 128.
This trend suggests that representative gradient align


ment is captured with relatively few samples, justifying

   - ur sample-efficient search settings.

  - **Context Length (** _CL_ **):** We observe that correlation
with _CL_ = 4096 remains high across context lengths,
rising from _τ ≈_ 0 _._ 70 at shorter contexts to _τ ≈_ 0 _._ 80
at _CL_ = 2048. The high agreement for all _CL ≥_
1024 rankings suggests that the proxy’s saliency is
largely preserved once the model captures sufficient
long-range dependencies, allowing for significantly
reduced compute during pruned model discovery.

  - **Low-rank Gradient Subspace (** _r_ **):** We observe that
proxy stability is most significant across varying gradient subspace dimensions. Correlation with the fullrank ( _r_ = 4096) configuration remains consistently
high, exceeding _τ ≈_ 0 _._ 77 even at _r_ = 2, and peaking
at _τ ≈_ 0 _._ 82 for _r_ = 8 before stabilizing. This validates

   - ur use of _r_ = 64 during the search, confirming that
low-rank gradient traces are sufficient to capture functional nuances while minimizing memory overhead
during the search.


Detailed inter-hyperparameter correlations, comparing each
configuration directly to downstream task performance, are
provided in Appendix C.3.


**4.3** **Main Results**


We evaluate the capability of TraceNAS to discover high
performing pruned models by evaluating on Llama-2-7B,
Llama-3.1-8B, and Qwen-2.5-14B Instruct to 2.7B, 4.6B,
and 8.4B parameters, respectively. As shown in Tables 2
and 3, TraceNAS identifies sub-networks with high sample
efficiency, requiring only 98M total search tokens, a 10 _×_
and 4 _×_ reduction compared to DarwinLM and ShearedLLaMA, while achieving superior accuracy. This efficiency
validates the high ranking capability and stability of our
proxy observed in Sec. 4.2.

**Performance on Llama-2-7B** TraceNAS achieves the

highest average accuracy (62.81%) on Llama-2 7B across
eight benchmarks, outperforming DarwinLM (62.57%) and
ShearedLLaMA (62.63%). To ensure a fair comparison,
we retrained DarwinLM and ShearedLLaMA on our 20B
token distribution ( _†_ ). Notably, ShearedLLaMA’s performance drops to 60.86% when restricted to our 20B token
budget, highlighting its reliance on high-token-count recovery to reach competitive accuracy. While DarwinLM
remains competitive, its performance slightly dips when
using our unfiltered FineWeb-Edu samples rather than the
highly curated top 0.1 percentile used in its original report.


Furthermore, TraceNAS significantly outperforms LoRAP (Li et al., 2024), E [3] - Pruner (Yuan et al., 2025a), and

- ur uniform baseline. Our 2.7B model has comparable the
performance with Flextron (Cai et al., 2024) despite the latter having a larger parameter footprint and requiring 4 _._ 5 _×_



7




--- end of page=6 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


_Table 2._ Pruning results for LLaMA-2-7B. Averages are calculated across eight reasoning benchmarks. TraceNAS achieves
highest average accuracy while requiring significantly fewer search tokens.


Method #Params Srch Recv PIQA WG ArcE ArcC HS SciQ LQA BQ Avg.


Llama-2-7B 6.7B    -    - 77.69 74.11 76.34 52.81 78.96 93.80 29.80 77.73 70.16


LoRAP 2.7B   -   - 57.20 47.90 31.30 26.30 30.00 51.20 27.50 61.90 41.66
Flextron 3.4B 1K 90B 74.10 62.00 66.50    - 68.50    -    -    -    E [3] 2.7B 0.5B   - 71.20 57.30 65.40 38.10 54.60 88.40 28.10 63.60 58.34
ShearedLLaMA 2.7B 0.4B 50B **75.80** **64.20** 67.00 41.20 **70.80** 90.80 28.20 63.00 62.63
DarwinLM 2.7B 1.0B 10B 73.12 63.77 69.94 **43.94** 66.95 90.20 28.57 **64.10** 62.57


Uniform 3.4B 98M 20B 65.10 58.30 69.50 38.20 54.30 87.10 26.03 59.90 57.30
ShearedLLaMA 2.7B 0.4B 20B _[†]_ 72.92 60.80 67.46 38.06 64.73 90.30 28.72 63.91 60.86
DarwinLM 2.7B 1.0B 20B _[†]_ 73.17 63.59 69.90 42.91 66.57 89.00 29.18 63.63 62.24
TraceNAS (Ours) 2.7B 98M 20B 73.40 63.97 **70.20** 43.80 68.54 **91.10** **29.71** 61.76 **62.81**


_Table 3._ Pruning results for Llama-3.1-8B and Qwen-2.5-14B-Instruct. TraceNAS has the best average accuracy and surpasses baselines

- n several complex reasoning benchmarks. Averages are calculated across the eight reasoning benchmarks, not including MMLU.


Method #Params Srch Recv MMLU PIQA WG ArcE ArcC HS SciQ LQA BQ Avg.


Llama-3.1-8B 8B  -  - 64.78 81.22 77.42 81.64 57.59 81.81 96.20 31.02 82.14 73.63


Minitron 4.4B  - 94B 60.50 76.82 73.50 76.89 56.14 76.03 96.20 31.02 83.14 71.22
DarwinLM 4.6B 1B 10B 43.21 **74.59** 65.03 73.27 51.27 71.13 93.40 30.72 71.00 66.30
Uniform (Ours) 4.6B 98M 20B 28.76 68.32 58.54 68.23 47.89 66.42 88.48 27.54 66.59 61.50
DarwinLM 4.6B 1B 20B _[†]_ **43.32** 74.53 65.26 73.13 51.62 **71.17** 93.24 30.76 70.08 66.22
TraceNAS (Ours) 4.6B 98M 20B 30.12 74.23 **66.08** **73.50** **52.26** 69.01 **93.90** **30.81** **73.00** **66.60**


Qwen-2.5 14B 14B  -  - 82.53 82.10 79.71 85.81 72.44 85.10 96.70 41.16 87.95 78.87


Minitron 8.4B  - 94B 36.10 80.41 80.03 83.38 64.24 83.10 97.10 33.48 84.43 75.77
DarwinLM 8.4B 1B 10B 55.97 78.12 70.63 79.41 **57.42** 74.93 89.30 33.10 73.57 69.56
E [3] 8.2B 0.5B - 36.50 76.90 63.00 76.00 47.90 67.00 93.70 30.00 66.50 65.13
Uniform (Ours) 8.4B 98M 20B 51.23 71.32 62.64 77.12 50.02 69.89 88.91 28.53 69.45 64.73
DarwinLM 8.4B 1B 20B _[†]_ 56.97 **78.17** 72.09 79.86 57.03 **74.96** 90.32 **32.80** 73.17 69.80
TraceNAS (Ours) 8.4B 98M 20B **58.79** 76.63 **73.24** **81.07** 56.05 73.77 **95.70** 32.56 **73.70** **70.34**



more CPT tokens. The substantial performance gap between
TraceNAS and the uniform baseline (5.51% average) underscores the necessity of non-uniform architecture search
for maintaining representational integrity. We provide performance scalability results of our Llama-2-2.7B model by
evaluating recovery performance at 10B and 50B tokens in
Appendix D.3.

**Generalization to GQA Architectures** Evaluation on
Llama-3.1 and Qwen-2.5 demonstrates the generalizability of TraceNAS to GQA architectures. For Llama-3.18B, TraceNAS significantly improves upon the uniform
baseline (66.60% vs 61.50%) and outperforms DarwinLM
across critical reasoning tasks such as ARC-C (52.26%) and
BoolQ (73.00%). On Qwen-2.5-14B, TraceNAS maintains
its lead with a 70.34% average, surpassing both DarwinLM
(69.80%) and E [3] - Pruner (65.13%).


While Minitron (Sreenivas et al., 2024) achieves higher accuracies across reasoning baseline and TraceNAS models,
it requires an intensive 94B tokens for recovery, approximately 4 _._ 7 _×_ the computational cost of TraceNAS. Across
all architectures, TraceNAS produces sub-networks whose
gradient traces are highly aligned with the pretrained gradi

  - Models trained on our CPT data setup and the original codebase provided in the respective papers.



ent trace, allowing for high-fidelity recovery and discovered
at a fraction of the computational search cost required by
existing training-aware methods. Detailed perplexity evaluations across different sparsity levels and a speedup analysis
against all publicly available baselines are provided in Appendix C.4 and Appendix D.1, respectively.


**5** **Conclusion**


In this work, we introduce TraceNAS, a zero-shot NAS
framework for non-uniform structured pruning of LLMs.
By leveraging a novel scale-invariant gradient proxy, TraceNAS identifies sub-networks that maintain high gradient trace alignment with the pretrained base model. This
alignment ensures the preservation of functional inheritance,
allowing for efficient recovery during continued pretraining. Our framework demonstrates consistent performance
gains across Llama-2, Llama-3.1, and Qwen-2.5 architectures, achieving an order-of-magnitude reduction in search

- verhead compared to training-aware baselines. By eliminating the prohibitive cost of search-time training, TraceNAS
provides a scalable and high-fidelity foundation for model
compression that effectively retains the complex reasoning
capabilities of dense pretrained models.



8




--- end of page=7 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**



**Acknowledgments**


This work was supported in part by the Center for the
Co-Design of Cognitive Systems (COCOSYS), a DARPAsponsored JUMP center, the Semiconductor Research Corporation (SRC), the National Science Foundation (NSF) and
the Department of Energy (DOE).


**Impact Statement**


The research presented in this paper advances the accessibility of high-performing LLMs by significantly lowering the
computational barriers to large-scale model compression.
By providing a training-free framework for architectural discovery, we enable the development of efficient models even
under resource constraints. Furthermore, the substantial
reduction in GPU-hours required for model search directly
mitigates the carbon footprint and energy consumption associated with large-scale NAS. These advancements enable
the deployment of LLMs in edge environments, supporting
sustainable AI development. To promote reproducibility,
an anonymous GitHub repository will be made available to
reviewers and area chairs during the discussion period, as
per the ICML Author Guidelines.


## References


Abdelfattah, M. S., Mehrotra, A., Dudziak, Ł., and Lane,
N. D. Zero-cost proxies for lightweight nas. _arXiv_
_preprint arXiv:2101.08134_, 2021.


Aghajanyan, A., Gupta, S., and Zettlemoyer, L. Intrinsic dimensionality explains the effectiveness of language
model fine-tuning. In Zong, C., Xia, F., Li, W., and
Navigli, R. (eds.), _Proceedings of the 59th Annual Meet-_
_ing of the Association for Computational Linguistics_
_and the 11th International Joint Conference on Natu-_
_ral Language Processing (Volume 1: Long Papers)_, pp.
7319–7328, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.
[568. URL https://aclanthology.org/2021.](https://aclanthology.org/2021.acl-long.568/)
[acl-long.568/.](https://aclanthology.org/2021.acl-long.568/)


Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y.,
Lebron, F., and Sanghai, S. Gqa: Training generalized´
multi-query transformer models from multi-head checkpoints. _arXiv preprint arXiv:2305.13245_, 2023.


An, Y., Zhao, X., Yu, T., Tang, M., and Wang, J. Fluctuationbased adaptive structured pruning for large language models. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, volume 38, pp. 10865–10873, 2024.


Anthropic, P. Introducing claude. _March_, 14:2023, 2023.


Askari, H., Gupta, S., Wang, F., Chhabra, A., and Chen,
M. Layerif: Estimating layer quality for large lan


guage models using influence functions. _arXiv preprint_
_arXiv:2505.23811_, 2025.


Bercovich, A., Ronen, T., Abramovich, T., Ailon, N., Assaf, N., Dabbah, M., Galil, I., Geifman, A., Geifman, Y.,
Golan, I., Haber, N., Karpas, E. D., Koren, R., Levy,
I., Molchanov, P., Mor, S., Moshe, Z., Nabwani, N.,
Puny, O., Rubin, R., Schen, I., Shahaf, I., Tropp, O.,
Argov, O. U., Zilberstein, R., and El-Yaniv, R. Puzzle:
Distillation-based NAS for inference-optimized LLMs.
In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S.,
Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu,
J. (eds.), _Proceedings of the 42nd International Confer-_
_ence on Machine Learning_, volume 267 of _Proceedings_

_of Machine Learning Research_, pp. 3806–3830. PMLR,
[13–19 Jul 2025. URL https://proceedings.mlr.](https://proceedings.mlr.press/v267/bercovich25a.html)
[press/v267/bercovich25a.html.](https://proceedings.mlr.press/v267/bercovich25a.html)


Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning
about physical commonsense in natural language. In _Pro-_
_ceedings of the AAAI conference on artificial intelligence_,
volume 34, pp. 7432–7439, 2020.


Cai, H., Gan, C., Wang, T., Zhang, Z., and Han, S. Oncefor-all: Train one network and specialize it for efficient
deployment. _arXiv preprint arXiv:1908.09791_, 2019.


Cai, R., Muralidharan, S., Heinrich, G., Yin, H., Wang,
Z., Kautz, J., and Molchanov, P. Flextron: Manyin-one flexible large language model. _arXiv preprint_
_arXiv:2406.10260_, 2024.


Chen, T., Frankle, J., Chang, S., Liu, S., Zhang, Y., Wang,
Z., and Carbin, M. The lottery ticket hypothesis for pretrained bert networks. _Advances in neural information_
_processing systems_, 33:15834–15846, 2020.


Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins,
M., and Toutanova, K. Boolq: Exploring the surprising
difficulty of natural yes/no questions. _arXiv preprint_
_arXiv:1905.10044_, 2019.


Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try arc, the ai2 reasoning challenge.
_arXiv preprint arXiv:1803.05457_, 2018.


Das, R. J., Sun, M., Ma, L., and Shen, Z. Beyond size:
How gradients shape pruning decisions in large language
models. _arXiv preprint arXiv:2311.04902_, 2023.


Dong, P., Li, L., Liu, X., Tang, Z., Liu, X., Wang, Q., and
Chu, X. Lpzero: Language model zero-cost proxy search
from zero. _arXiv preprint arXiv:2410.04808_, 2024a.


Dong, P., Li, L., Tang, Z., Liu, X., Pan, X., Wang, Q., and
Chu, X. Pruner-zero: Evolving symbolic pruning metric
from scratch for large language models. _arXiv preprint_
_arXiv:2406.02924_, 2024b.



9




--- end of page=8 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**



Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A.,
Letman, A., Mathur, A., Yang, A., Fan, A., et al. The
llama 3 herd of models. _arXiv preprint arXiv:2407.21783_,
2024.


Frankle, J. and Carbin, M. The lottery ticket hypothesis:
Finding sparse, trainable neural networks. _arXiv preprint_
_arXiv:1803.03635_, 2018.


Frankle, J., Dziugaite, G. K., Roy, D., and Carbin, M. Linear
mode connectivity and the lottery ticket hypothesis. In
_International Conference on Machine Learning_, pp. 3259–
3269. PMLR, 2020.


Frantar, E. and Alistarh, D. Sparsegpt: Massive language
models can be accurately pruned in one-shot. In _Interna-_
_tional conference on machine learning_, pp. 10323–10337.
PMLR, 2023.


Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq:
Accurate post-training quantization for generative pretrained transformers. _arXiv preprint arXiv:2210.17323_,
2022.


Fu, Y., Yu, Z., Li, J., Qian, J., Zhang, Y., Yuan, X., Shi, D.,
Yakunin, R., and Lin, Y. C. Amoeballm: Constructing
any-shape large language models for efficient and instant
deployment. _Advances in Neural Information Processing_
_Systems_, 37:78299–78319, 2024.


Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi,
A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li,
H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,
J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.
The language model evaluation harness, 07 2024. URL
[https://zenodo.org/records/12608602.](https://zenodo.org/records/12608602)


Gromov, A., Tirumala, K., Shapourian, H., Glorioso,
P., and Roberts, D. A. The unreasonable ineffectiveness of the deeper layers, 2024. _URL https://arxiv._

_org/abs/2403.17887_, 2024.


Han, S., Mao, H., and Dally, W. J. Deep compression: Compressing deep neural networks with pruning,
trained quantization and huffman coding. _arXiv preprint_
_arXiv:1510.00149_, 2015.


Hassibi, B., Stork, D. G., and Wolff, G. J. Optimal brain
surgeon and general network pruning. In _IEEE interna-_
_tional conference on neural networks_, pp. 293–299. IEEE,
1993.


Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,
M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. _arXiv preprint_
_arXiv:2009.03300_, 2020.



Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang,
S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation

 - f large language models. _ICLR_, 1(2):3, 2022.


Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z.,
Fang, Y., Huang, Y., Zhao, W., et al. Minicpm: Unveiling
the potential of small language models with scalable training strategies. _arXiv preprint arXiv:2404.06395_, 2024.


Ingolfsson, T. M., Vero, M., Wang, X., Lamberti, L., Benini,
L., and Spallanzani, M. Reducing neural architecture
search spaces with training-free statistics and computational graph clustering. In _Proceedings of the 19th ACM_
_International Conference on Computing Frontiers_, pp.
213–214, 2022.


Jiang, T., Wang, H., and Bie, R. Meco: zero-shot nas with

 - ne data and single forward pass via minimum eigenvalue of correlation. _Advances in Neural Information_
_Processing Systems_, 36:61020–61047, 2023.


Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and
Amodei, D. Scaling laws for neural language models.
_arXiv preprint arXiv:2001.08361_, 2020.


Kendall, M. G. A new measure of rank correlation.
_Biometrika_, 30(1-2):81–93, 1938.


Kim, B.-K., Kim, G., Kim, T.-H., Castells, T., Choi, S.,
Shin, J., and Song, H.-K. Shortened llama: A simple
depth pruning for large language models. _arXiv preprint_
_arXiv:2402.02834_, 11:1, 2024.


Koh, P. W. and Liang, P. Understanding black-box predictions via influence functions. In _International conference_

_on machine learning_, pp. 1885–1894. PMLR, 2017.


Kovaleva, O., Kulshreshtha, S., Rogers, A., and Rumshisky,
A. Bert busters: Outlier dimensions that disrupt transformers. _arXiv preprint arXiv:2105.06990_, 2021.


Kwon, Y., Wu, E., Wu, K., and Zou, J. Datainf: Efficiently
estimating data influence in lora-tuned llms and diffusion
models. _arXiv preprint arXiv:2310.00902_, 2023.


Li, G., Yang, Y., Bhardwaj, K., and Marculescu, R. Zico:
Zero-shot nas via inverse coefficient of variation on gradients. _arXiv preprint arXiv:2301.11300_, 2023.


Li, G., Tang, Y., and Zhang, W. Lorap: Transformer sublayers deserve differentiated structured compression for
large language models. _arXiv preprint arXiv:2404.09695_,
2024.


Li, H., Xu, Z., Taylor, G., Studer, C., and Goldstein, T.
Visualizing the loss landscape of neural nets. _Advances_
_in neural information processing systems_, 31, 2018.



10




--- end of page=9 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**



Li, L. and Jin, Z. Shadow knowledge distillation: Bridging

  - ffline and online knowledge transfer. _Advances in Neural_
_Information Processing Systems_, 35:635–649, 2022.


Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang,
Y. Logiqa: A challenge dataset for machine reading
comprehension with logical reasoning. _arXiv preprint_
_arXiv:2007.08124_, 2020.


Liu, Z., Oguz, B., Zhao, C., Chang, E., Stock, P., Mehdad,
Y., Shi, Y., Krishnamoorthi, R., and Chandra, V. Llm-qat:
Data-free quantization aware training for large language
models. In _Findings of the Association for Computational_
_Linguistics: ACL 2024_, pp. 467–484, 2024.


Luo, Z., Kulmizev, A., and Mao, X. Positional artefacts
propagate through masked language model embeddings.
In _Proceedings of the 59th annual meeting of the Associ-_
_ation for Computational Linguistics and the 11th interna-_
_tional joint conference on natural language processing_
_(volume 1: long papers)_, pp. 5312–5327, 2021.


Ma, X., Fang, G., and Wang, X. Llm-pruner: On the structural pruning of large language models. _Advances in_
_neural information processing systems_, 36:21702–21720,
2023.


Mellor, J., Turner, J., Storkey, A., and Crowley, E. J. Neural
architecture search without training. In _International_
_conference on machine learning_, pp. 7588–7598. PMLR,
2021.


Mukherjee, S., Mitra, A., Jawahar, G., Agarwal, S., Palangi,
H., and Awadallah, A. Orca: Progressive learning from
complex explanation traces of gpt-4. _arXiv preprint_
_arXiv:2306.02707_, 2023.


OpenAI, R. Gpt-4 technical report. arxiv 2303.08774. _View_
_in Article_, 2(5):1, 2023.


Penedo, G., Kydl´ıcek, H., Lozhkov, A., Mitchell, M., Raffel,ˇ
C. A., Von Werra, L., Wolf, T., et al. The fineweb datasets:
Decanting the web for the finest text data at scale. _Ad-_
_vances in Neural Information Processing Systems_, 37:
30811–30849, 2024.


Pruthi, G., Liu, F., Kale, S., and Sundararajan, M. Estimating training data influence by tracing gradient descent.
_Advances in Neural Information Processing Systems_, 33:
19920–19930, 2020.


Sakaguchi, K., Le Bras, R., Bhagavatula, C., and Choi, Y.
Winogrande: An adversarial winograd schema challenge
at scale. In _Proceedings of the AAAI Conference on_
_Artificial Intelligence_, volume 34, pp. 8732–8740, 2020.


Sarah, A., Nittur Sridhar, S., Szankin, M., and Sundaresan, S. Llama-nas: Efficient neural architecture search



for large language models. In _European Conference on_
_Computer Vision_, pp. 67–74. Springer, 2024.


Sieberling, O., Kuznedelev, D., Kurtic, E., and Alistarh,
D. Evopress: Accurate dynamic model compression
via evolutionary search. In _Forty-second International_
_Conference on Machine Learning_ .


Spearman, C. The proof and measurement of association
between two things. 1961.


Sreenivas, S. T., Muralidharan, S., Joshi, R., Chochowski,
M., Mahabaleshwarkar, A. S., Shen, G., Zeng, J., Chen,
Z., Suhara, Y., Diao, S., et al. Llm pruning and distillation in practice: The minitron approach. _arXiv preprint_
_arXiv:2408.11796_, 2024.


Sukthanker, R. S., Staffler, B., Hutter, F., and Klein, A.
Large language model compression with neural architecture search. In _Workshop on Machine Learning and_
_Compression, NeurIPS 2024_ .


Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and
effective pruning approach for large language models.
_arXiv preprint arXiv:2306.11695_, 2023.


Tanaka, H., Kunin, D., Yamins, D. L., and Ganguli, S. Pruning neural networks without any data by iteratively conserving synaptic flow. _Advances in neural information_
_processing systems_, 33:6377–6389, 2020.


Tang, S., Sieberling, O., Kurtic, E., Shen, Z., and Alistarh, D. Darwinlm: Evolutionary structured pruning of
large language models. _arXiv preprint arXiv:2502.07780_,
2025.


Tao, C., Hou, L., Bai, H., Wei, J., Jiang, X., Liu, Q., Luo, P.,
and Wong, N. Structured pruning for efficient generative
pre-trained language models. In Rogers, A., Boyd-Graber,
J., and Okazaki, N. (eds.), _Findings of the Association for_
_Computational Linguistics: ACL 2023_, pp. 10880–10895,
Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.
[692. URL https://aclanthology.org/2023.](https://aclanthology.org/2023.findings-acl.692/)
[findings-acl.692/.](https://aclanthology.org/2023.findings-acl.692/)


Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X.,
Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford
alpaca: An instruction-following llama model, 2023.


Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,`
Azhar, F., et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023.


Tran, C., Fioretto, F., Kim, J.-E., and Naidu, R. Pruning
has a disparate impact on model accuracy. _Advances in_
_neural information processing systems_, 35:17652–17664,
2022.



11




--- end of page=10 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**



Tran, L.-T., Ali, M. S., and Bae, S.-H. A feature fusion based indicator for training-free neural architecture
search. _IEEE Access_, 9:133914–133923, 2021.


Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information_
_processing systems_, 30, 2017.


Veit, A., Wilber, M. J., and Belongie, S. Residual networks
behave like ensembles of relatively shallow networks.
_Advances in neural information processing systems_, 29,
2016.


Wang, Z., Wohlwend, J., and Lei, T. Structured pruning

 - f large language models. In _Proceedings of the 2020_
_conference on empirical methods in natural language_
_processing (emnlp)_, pp. 6151–6162, 2020.


Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. _arXiv preprint_
_arXiv:1707.06209_, 2017.


Wu, M.-T., Lin, H.-I., and Tsai, C.-W. A training-free
genetic neural architecture search. In _Proceedings of_
_the 2021 ACM International Conference on Intelligent_
_Computing and Its Emerging Applications_, pp. 65–70,
2021.


Xia, M., Gao, T., Zeng, Z., and Chen, D. Sheared llama:
Accelerating language model pre-training via structured
pruning. _arXiv preprint arXiv:2310.06694_, 2023.


Yang, Y., Zhen, K., Ganesh, B., Galstyan, A., Huybrechts,
G., Muller, M., K¨ ubler, J. M., Swaminathan, R. V.,¨
Mouchtaris, A., Bodapati, S. B., et al. Wanda++: Pruning large language models via regional gradients. _arXiv_
_preprint arXiv:2503.04992_, 2025.


Yin, L., Wu, Y., Zhang, Z., Hsieh, C.-Y., Wang, Y., Jia, Y.,
Li, G., Jaiswal, A., Pechenizkiy, M., Liang, Y., et al. Outlier weighed layerwise sparsity (owl): A missing secret
sauce for pruning llms to high sparsity. _arXiv preprint_
_arXiv:2310.05175_, 2023.


Yu, J., Jin, P., Liu, H., Bender, G., Kindermans, P.-J., Tan,
M., Huang, T., Song, X., Pang, R., and Le, Q. Bignas:
Scaling up neural architecture search with big singlestage models. In _European Conference on Computer_
_Vision_, pp. 702–717. Springer, 2020.


Yuan, T., Bai, H., Pan, Y., Cao, X., Zhang, T., Hou, L., Hu,
T., and Yu, X. E [3]  - pruner: Towards efficient, economical,
and effective layer pruning for large language models.
_arXiv preprint arXiv:2511.17205_, 2025a.



Yuan, T., Bai, H., Pan, Y., Cao, X., Zhang, T., Hou, L.,
Hu, T., and Yu, X. E [3]  - pruner: Towards efficient, economical, and effective layer pruning for large language
[models, 2025b. URL https://arxiv.org/abs/](https://arxiv.org/abs/2511.17205)

[2511.17205.](https://arxiv.org/abs/2511.17205)


Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,
Y. Hellaswag: Can a machine really finish your sentence?
_arXiv preprint arXiv:1905.07830_, 2019.


Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar,
A., and Tian, Y. Galore: Memory-efficient llm training by gradient low-rank projection. _arXiv preprint_
_arXiv:2403.03507_, 2024.



12




--- end of page=11 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


## Appendix


TABLE OF CONTENTS


**A Implementation & Search Analysis** **13**
A.1 Evolutionary Search Initialization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
A.2 Search Space Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
A.3 Evolution Search and CPT Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
A.4 Pseudocode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15


**B** **Motivation & Conceptual Framework** **16**
B.1 Comparison with Influence Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
B.2 Influence Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
B.3 Structural Sensitivity via Data-Centric Influence . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
B.4 TracIn & the Link to TraceNAS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

B.5 Why this works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17


**C Validation & Proxy Correlation** **17**
C.1 Correlation with Downstream Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

C.2 Validating Sparsity-Weighted Pearson Correlation . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
C.3 Robustness of Proxy Stability: Correlation with Downstream Accuracy . . . . . . . . . . . . . . . . 18
C.4 Performance Across Different Sparsities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19


**D Scalability & Speedup Analysis** **20**
D.1 Speedup Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
D.2 Extending TraceNAS to Different Model Scales . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
D.3 Training with more tokens . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
D.4 Evaluating TraceNAS for Unstructured Pruning . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21


**E** **Generation Quality Example** **22**


**F** **Limitations** **22**


**A** **Implementation & Search Analysis**


**A.1** **Evolutionary Search Initialization**


_Table 4._ Comparison of LLaMA-2-7B pruning using evolutionary search and importance based initialization, highlighted as TraceNAS
(Ours).


Method #Params PIQA WG ArcE ArcC HS SciQ LQA BQ Avg.

Llama-2-7B 6.7B 77.69 74.11 76.34 52.81 78.96 93.80 29.80 77.73 70.16

TraceNAS - Random Search 2.7B 68.42 58.86 66.72 37.13 58.45 88.79 27.85 59.98 58.28
TraceNAS - Uniform Init 2.7B 72.24 61.21 69.72 41.79 65.24 89.96 28.84 59.79 61.10
TraceNAS - Random Init 2.7B 71.96 60.87 68.98 40.75 63.11 90.75 28.76 59.85 60.63
TraceNAS (Ours) 2.7B 73.40 63.97 **70.20** 43.80 68.54 **91.10** **29.71** 61.76 **62.81**


To improve search efficiency and avoid sub-optimal architectural configurations, we warm-start the evolutionary search
using a layer importance global prior. Instead of initializing the population uniformly or randomly within the search space,
we bias candidates toward structurally stable regions of the pretrained model. Specifically, we compute per-layer importance
scores based on the expectation of the weight-activation product (Sun et al., 2023), defined as _Il_ = E� _|Wl| ⊙∥Xl∥_ 2�, which
characterizes the sensitivity of each layer.


These importance scores define a non-uniform sampling prior over the sub-module width sparsity configurations _**κ**_ =
_{κ_ 1 _, . . ., κL}_, where _κl_ = ( _rattn_ [(] _[l]_ [)] _[, r]_ _mlp_ [(] _[l]_ [)] [)][. Biasing the initial population] _[ P]_ [0][ toward high-importance layers anchors the]


13




--- end of page=12 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**




















|)<br>02 TraceNASScore(<br>.<br>01<br>.<br>00<br>.<br>01<br>.<br>26 28 30<br>. . .|Col2|Col3|Col4|
|---|---|---|---|
|2.6<br>2.8<br>3.0<br>0.1<br>0.0<br>0.1<br>0.2<br>TraceNAS Score ( )||||



### Figure 4

Caption: _ **TraceNAS Evolutionary Dynamics and Search Convergence.** Search trajectory for pruning Llama-2-7B to 2.7B across 50

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
iterations. (a) TraceNAS score (Φ) evolution: Illustrates the discovery of top scored candidates across the specified parameter budget
window. The red star indicates the model with maximal functional inheritance for the given constraint. (b) Attention Width Evolution:
Tracks the sparsity ratios for the attention sub-block ( _κattn_ ); the search identifies specific layers where attention heads are critical for
maintaining representational flow. (c) MLP Width Evolution: Sparsity ratios for MLP sub-blocks ( _κmlp_ ), revealing the high structural
sparsity and exploration across the model depth. (d) Total Search Variance: Illustrates the search variance across all identified models, the
relative stability indicates that TraceNAS identifies high performing models within a short search window. In figures (b), (c) and (d),
_κ_ = 0 _._ 0 indicates no pruning in that layer and _κ_ = 1 indicates the layer has been dropped.


search in regions that preserve the representational capacity of the base model while adhering to the parameter budget _C_ .
This initialization strategy reduces search variance typically caused by stochastic sparsity assignments and prevents early
convergence to architectures that fail to inherit pretrained capabilities.


To quantify the benefits of importance-based initialization, we conduct an ablation study on LLaMA-2-7B comparing our
approach to uniform and random initialization strategies. Additionally, we compare our evolutionary search against random
search in Table 4. Importance-based initialization provides a significantly stronger starting point than standard schemes;
when initialized using _Il_, TraceNAS achieves an average accuracy of 62.81%, representing a 1.71% improvement over
uniform and a 2.18% improvement over random initialization. These results indicate that leveraging layer-wise importance
effectively warm-starts the search, accelerating the discovery of high-performing sub-networks under a fixed parameter
budget.


Furthermore, evolutionary search yields a substantial 4.54% absolute gain in average accuracy over random search. This
improvement confirms that selecting elite candidates via Φ, coupled with importance-based initialization, provides a robust
search scheme for iterative refinement. By effectively navigating the discrete architectural space, this approach enables the
consistent discovery of high performing models, without the high variance of random search.


**A.2** **Search Space Analysis**


To analyze the dynamics of the evolutionary search, Fig. 4 illustrates the behavior of TraceNAS across 50 iterations.
As shown in Fig. 4(a), the population moves from initial stochastic exploration toward candidates with progressively
higher gradient alignment scores. The presence of negative alignment values in early iterations suggests that importancebased initialization alone cannot guarantee functional inheritance, as initial sparsity patterns introduce representational
instability. This behavior also confirms that the prior does not overly constrain the search, leaving sufficient flexibility for
the evolutionary process to explore the broad search space.


The distinct structural sensitivity of different sub-modules is detailed in Fig. 4(b) and (c). While the global importance prior
is relatively uniform, the search discovers unique sparsity profiles for attention ( _κattn_ ) and MLP ( _κmlp_ ) widths. Specifically,
the first iteration of the search finds that the final MLP layers are very important and thus they are not pruned, whereas
attention modules less importance in later layers and thus initial layers have high sparsity. This indicates that representational
integrity relies on different depth-wise configurations for different sub-module types.


Finally, Fig. 4(d) reports the population-level variance. The zero variance in the first iteration confirms that the search begins
from a single anchored model, with diversity expanding only as mutations and crossovers are introduced in subsequent
steps. The subsequent stabilization and variance plateau signal that the search successfully converged to a set of high-quality
architectures. This trend validates the efficiency of the evolutionary mechanism in navigating the discrete search space and
identifying optimal non-uniform configurations within a limited iteration budget.


14




--- end of page=13 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


**A.3** **Evolution Search and CPT Hyperparameters**


Table 5 provides comprehensive overview of the evolution search and CPT hyperparameters used to realize the models
presented in TraceNAS.


_Table 5._ Hyperparameter Configuration for TraceNAS Search (Left) and CPT (Right).


Search Parameters CPT Parameters

Parameter Parameter
2.7B 4.6B 8.4B 2.7B 4.6B 8.4B


Population Size ( _P_ ) 30 Learning Rate 1 _×_ 10 _[−]_ [4]

Elites ( _K_ ) 10 Batch Size 1024
Crossover Rate 0.7 LR Scheduler WSD

Mutation Rate 0.2 / 0.2 WSD Ratios 0.05 / 0.65 / 0.30
Search Iterations ( _N_ ) 50 Context Length 4,096 8,192 4,096

         -          - Overall Tokens 20B


**A.4** **Pseudocode**


Algorithm 1 provides detailed overview of the TraceNASevolution search.


**Algorithm 1** TraceNAS
**Input** **:** Pretrained _Mbase_ ; Budget _C_ ; Calibration set _B_ ; Sparsity _**κ**_ ; Elite size _k_ ; Search iterations _T_
**Output :** Optimal pruned sub-network _M_ [ˆ] _sub_

Initialize _Mbase_ with LoRA modules;
_gbase ←_ E _b∈B_ [ _∇θL_ ( _Mbase_ ( _b_ ; _θ_ ))] ; // Functional Anchor
_Il ←_ E� _|Wl| ⊙∥Xl∥_ 2� ; // Compute block importance for all _L_


Initialize _P_ 0 with random depth **d** _∈{_ 0 _,_ 1 _}_ _[L]_ and width _**κ**_ weighted by _Il_

**for** _t_ = 1 **to** _T_ **do**

**for** _each candidate_ ( _Msub_ ( **d** _,_ _**κ**_ )) _∈_ _Pt_ **do**

// Structural Realization
**if** # _Params_ ( _Msub_ ) _> C_ **then**

_θ_ ( _Msub_ ) _←−∞_ ; // Budget Penalty
**continue**
**else**

Realize masks _Wl_ _[′]_ [=] _[ W][l][ ⊙]_ [Mask][(] _[κ][l]_ [)][ based on active layers] **[ d]**
Route activations via residual connections for deactivated layers ( _dl_ = 0)


// Gradient-based Evaluation
_gsub ←_ E _b∈B_ [ _∇θL_ ( _Msub_ ( _b_ ; _θ_ ))]
**for** _each active layer l_ **do**

_ρ_ [(] _[l]_ [)] _←_ Pearson Correlation Coefficient( _gsub_ [(] _[l]_ [)] _[, g]_ _base_ [(] _[l]_ [)] [)][ ;] // Gradient Trace Correlation



// Fitness Aggregation
Φ( _Msub_ ) _←_ [�] _l_ Attn _[r]_ _attn_ [(] _[l]_ [)] _[ρ]_ [(] _[l]_ [)][ +]



_l∈_ MLP _[r]_ _mlp_ [(] _[l]_ [)] _[ρ]_ [(] _[l]_ [)]



_l∈_ Attn _[r]_ _attn_ [(] _[l]_ [)] _[ρ]_ [(] _[l]_ [)][ +][ �]



// Elite Selection and Reproduction
_Et ←_ Select top _k_ candidates from _Pt_ based on Φ _Pt_ +1 _←_ _Et_ ; // Carry over elite set directly
**while** _|Pt_ +1 _| < |Pt|_ **do**

Parent _A, B ←_ Sample from _Et_ Child _C ←_ Crossover( _A, B_ ) + Mutation( _C_ ) Add Child _C_ to _Pt_ +1


**return** _M_ [ˆ] _sub ←_ arg max _{PT }_
Φ


15




--- end of page=14 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


**B** **Motivation & Conceptual Framework**


**B.1** **Comparison with Influence Functions**


To further clarify the positioning of TraceNAS, we distinguish our framework from data-centric influence functions (Koh &
Liang, 2017; Kwon et al., 2023), specifically LayerIF (Askari et al., 2025) and TracIn (Pruthi et al., 2020).


**B.2** **Influence Functions**


Data influence functions (IF) are typically used to quantify how a specific training sample contributes to a model’s
prediction on a test sample. Specifically, IF quantifies how model parameters _θ_ _[∗]_ change when a specific training point _zk_ is
infinitesimally up-weighted. Formally, assuming the loss _L_ is twice differentiable, the influence of the _k_ - th training point on
the parameters is defined as:



_Iθ∗_ ( _zk_ ) := _[dθ]_ [(] _[k]_ [)]

_dϵ_



= _−H_ ( _θ_ _[∗]_ ) _[−]_ [1] _∇θL_ ( _zk, θ_ )

- ��� _ϵ_ =0



where _H_ ( _θ_ ) is the Hessian matrix. The influence of a training sample _zi_ - n the validation loss across _m_ validation points is
then:



_I_ ( _zi_ ) = _−_



_m_

- _∇θL_ ( _zj_ _[V]_ _[, θ]_ [)] _[⊤][H]_ [(] _[θ]_ [)] _[−]_ [1] _[∇][θ][L]_ [(] _[z][i][, θ]_ [)]

_j_ =1



This measures whether a specific sample has a beneficial or detrimental impact on predictive performance.


**B.3** **Structural Sensitivity via Data-Centric Influence**


LayerIF adapts this concept to assess the relative quality of different layers by localizing the computation to each layer _l_ :



_I_ [(] _[l]_ [)] ( _zi_ ) = _−_



_m_ _−_ 1

- _∇_ [(] _θ_ _[l]_ [)] _[L]_ [(] _[z]_ _j_ _[V]_ _[, θ]_ [)] _[⊤]_ [�] _H_ [(] _[l]_ [)] ( _θ_ )� _∇_ [(] _θ_ _[l]_ [)] _[L]_ [(] _[z][i][, θ]_ [)]

_j_ =1



By aggregating positive influence scores, LayerIF derives a vector _S_ where each element _S_ [(] _[l]_ [)] = [�] _[n]_ _i_ =1 [I][[] _[I]_ [(] _[l]_ [)][(] _[z][i]_ [)] _[ >]_
0] _·I_ [(] _[l]_ [)] ( _zi_ ), where I[ _·_ ] is the indicator function, captures the cumulative contribution of training data to validation performance
through that specific layer. Crucially, LayerIF maintains a static model architecture; its analysis is data-centric, keeping the
architecture fixed to evaluate layer specialization across different data samples.


**B.4** **TracIn & the Link to TraceNAS**


In contrast, TraceNAS keeps the data static and instead evaluates model sensitivity. We analyze the functional performance

- f the model as its architecture is structurally perturbed. Our formulation is inspired by TracIn, which simplifies standard
influence functions by removing the _O_ ( _d_ [3] ) complexity of the inverse Hessian. TracIn computes a first-order approximation
via the dot-product alignment between gradients of training and test samples:



TracIn( _zi, zj_ ) =



_T_

- _ηt∇θL_ ( _zi, θt_ ) _· ∇θL_ ( _zj, θt_ )


_t_ =1



We extend this simplification to the domain of structural pruning by comparing the base model’s gradient ( _gbase_ ) and the
pruned candidate’s gradient ( _gsub_ ) on the same fixed data.


As detailed in Sec. 3.4, to quantify how effectively _Msub_ inherits the functional state of _Mbase_, we calculate the sub-blockwise Pearson Correlation Coefficient _ρ_ [(] _[l]_ [)] :





 _,_



�



 _gbase_ [(] _[l]_ [)] _[−]_ _[µ]_ _gbase_ [(] _[l]_ [)]


_σ_ ( _l_ )
 _gbase_



_ρ_ [(] _[l]_ [)] = [1]

_Nl_



_gsub_ [(] _[l]_ [)] _[−]_ _[µ]_ _gsub_ [(] _[l]_ [)]

_σ_ ( _l_ )

- []  _gsub_



where _Nl_ is the dimensionality of the low-rank subspace, _gbase_ and _gsub_ are defined as _g_ = E _b∈B_ [ _∇θL_ ( _M_ ( _b_ ; _θ_ ))], _µ_ and
_σ_ denote mean and standard deviation of gradient elements. By standardizing the traces _g_, _ρ_ [(] _[l]_ [)] captures the directional
alignment decoupled from the magnitude shifts induced by pruning.


16




--- end of page=15 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


We aggregate these correlations using Sparsity-Weighted Aggregation to account for heterogeneous representation capacity:




- _rattn_ [(] _[l]_ [)] _[·][ ρ]_ [(] _[l]_ [)][ +]
_l∈_ Attn _l∈_



Φ( _Msub_ ) =



- _rmlp_ [(] _[l]_ [)] _[·][ ρ]_ [(] _[l]_ [)]

_l∈_ MLP



where _r_ [(] _[l]_ [)] is the retention ratio. This formulation anchors the global score in high-capacity regions, preventing Φ from being
skewed by high-variance noise in aggressively pruned sub-blocks. This ensures the candidate resides within the original
pretrained convergence basin, facilitating high-fidelity recovery.


**B.5** **Why this works**


The primary reason this formulation works is that it treats the gradient as a local “topographic map” of the optimization
landscape. While magnitude-based proxies like GradNorm measure if the landscape is stable, Φ measures if the gradient
direction of the pruned model still points in the same direction as the original.


  - **Gradient Trace as a Proxy for the Optimization Path:** The gradient trace represents the direction of steepest descent.
By calculating the alignment between _gbase_ and _gsub_, we are measuring whether the sub-network wants to move toward
the same local minima as the pretrained model. This is why we refer to it as Functional Inheritance, whether the pruned
model inherits the optimization intent of its base model.


  - **Decoupling Magnitude from Direction:** We used the Pearson Correlation Coefficient ( _ρ_ [(] _[l]_ [)] ) specifically to decouple
the directional signal from magnitude shifts. Structural pruning inherently reduces the total weight volume, which
naturally suppresses gradient magnitudes. If we used a simple dot product, the score would drop simply because the
model is smaller. Standardizing the traces via _µ_ and _σ_ allows us to evaluate if the logic of the layer remains, regardless

   - f its reduced power.


**C** **Validation & Proxy Correlation**


This section provides a detailed analysis of the TraceNAS proxy Φ and its ranking correlation with model perplexity (PPL),
MMLU, and average downstream accuracies compared against the established baselines in Table 1.


**C.1** **Correlation with Downstream Performance**


As shown in Table 1, TraceNAS achieves superior ranking correlation with downstream performance by effectively modeling
functional inheritance. Proxies like NASWOT (Mellor et al., 2021) and ZiCo (Li et al., 2023) are formulated to rank
models based on expressivity and convergence capability from random initializations. NASWOT measures the linear
separability of data representations by quantifying the dissimilarity in feature patterns using the Hamming distance. This
metric characterizes the richness of data representations in a network, this is reflected in ranking performance on PPL
( _τ_ = 0 _._ 72). However, this does not account for whether a pruned architecture retains the specific linguistic knowledge,
MMLU correlation of ( _τ_ = 0 _._ 07) that is already encoded in the weights.


Similarly, ZiCo serves as a measure of loss landscape smoothness using the inverse coefficient of variation. This is defined
as the ratio of the mean of gradients to their standard deviation across samples. This assumes that networks with high
convergence speed and generalization capacity exhibit high absolute mean gradients and low variance, and will generalize
well. This measure of generalizability is reflected in PPL ( _τ_ = 0 _._ 5) ranking. These properties of ZiCo are crucial for

- ptimization, yet they are inherently blind to the pretrained knowledge of extensively pretrained LLMs and the functional
disruption caused by pruning, MMLU _τ_ = 0 _._ 26,. These metrics optimize for a model that could learn well, whereas pruning
requires a model that has already learned and preserved its pretrained distributions. This lack of accounting for pretrained
knowledge inheritance explains their limited ranking correlation on specialized tasks like MMLU, where representational
fidelity, not just trainability, is the primary driver of performance.


In contrast, GradNorm (Abdelfattah et al., 2021) functions as a robust proxy for pruned variants of pretrained models. This
stems from the principle it operates on: that the stability of a model, expressed through its gradient _L_ 2 norm is a good
measure of potential performance. Similarly, Synaptic Saliency (Tanaka et al., 2020) approximates parameter importance
by measuring the impact of parameter removal on the model’s loss function. Specifically, the proxy measures how much
structural perturbations, in the form of parameter removal, impacts the model’s total loss. However, these metrics primarily
prioritize structural health and do not account for the global impact of pruning on representational depth or the specific
disruption caused by multi-parameter pruning to knowledge retention tasks like MMLU. This limitation is reflected in their


17




--- end of page=16 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


Kendall _τ_ correlation values of only _τ_ = 0 _._ 45 and _τ_ = 0 _._ 37 on MMLU, respectively.


To further highlight the need for proxies to account for the global impact of pruning, we analyze MeCo (Jiang et al., 2023).
MeCo measures generalization capacity of a model using the minium eigen values of the Pearson Correlation matrix across
feature representations. This effectively captures generalizability, as reflected in PPL ranking ( _τ_ = 0 _._ 79). However, it does
not translate to the complex reasoning and knowledge retention abilities required in pruned models, MMLU _τ_ = 0 _._ 02, that
inherit a distorted version of the pretrained weight state. Conversely, PrunerZero (Dong et al., 2024b), achieves the highest
ranking correlation on MMLU, _τ_ = 0 _._ 69. However, this high performance does not translate to model generalizability in
the form of PPL or average accuracy on downstream tasks. We attribute this to the proxy formulation, defined as the product

- f the weight norms and the min-max scaled gradient vector. The high ranking correlation on MMLU is driven by the high
magnitude weights acting as pointers to the high knowledge retention regions within the pretrained model. These regions are
isolated by the min-max scaled gradients, making the proxy susceptible to outliers, thus resulting in poor performance on
PPL ( _τ_ = 0 _._ 31). Lastly, #Params is a strong indicator of model generalizability through PPL ranking ( _τ_ = 0 _._ 72). However,
it does not account for representational collapse in pruned models, MMLU correlation _τ_ = 0 _._ 07. Furthermore, it would not
be able to distinguish between models under a contrained parameter budget.


In contrast, TraceNAS provides the best end-to-end ranking correlation. While GradNorm is sensitive to mean directional
shifts in gradients, TraceNAS employs centering via Pearson Correlation to isolate the functional gradient trace alignment
from directional noise. This de-noises the signal to reveal the underlying structural inheritance that simpler magnitude-based
metrics miss and functions as a reliable proxy for model performance.


**C.2** **Validating Sparsity-Weighted Pearson Correlation**


To justify the formulation of Φ, we evaluate dot product (TraceNAS - Dot), cosine similarity (TraceNAS - Cosine), and
unweighted aggregation of Pearson correlation coefficients (TraceNAS - Unweighted) within the our low-rank gradient
setup. We see that cosine similarity is more robust than dot product, which fails due to the magnitude shifts caused by
structural pruning. However, cosine similarity remains sensitive to mean-gradient bias. Pearson correlation coefficient
addresses this by centering the gradient traces, however, uniformly aggregating these coefficients across layers introduces
instability in highly compressed blocks. By using layer sparsity as a weighting factor, Φ acts as a dynamic noise-filtering
mechanism, thus de-emphasizing highly pruned layers. This weighting anchors the global score in the high-capacity regions
identified as the primary repositories of functional inheritance. By prioritizing sub-blocks with higher parameter density, we
prevent the global fitness signal from being skewed by the high-variance typical of aggressively pruned, low-capacity blocks.
Furthermore, dot product fails as it lacks the scale-invariance necessary to handle the significant magnitude shifts inherently
caused by structural pruning.


**C.3** **Robustness of Proxy Stability: Correlation with Downstream Accuracy**


As shown in Sec. 4.2, the TraceNAS proxy (Φ) demonstrates strong robustness and inter hyperparameter stability in ranking
pruned model performance across accuracy and perplexity. To further validate these results we show that Φ, consistently
shows high correlation with downstream performance metrics under various search constraints. We measure Spearman _ρ_
and Kendall _τ_ correlations between the proxies (generated across all hyperparameters) and WikiText-2 Perplexity (PPL),
MMLU accuracy and average accuracy in Table 6 and 7.


_Table 6._ Spearman _ρ_ correlation between various search hyperparameters and model PPL and average downstream accuracy. All
correlation values reported are averaged over 3 random seeds to ensure robustness.


(a) _N_ (b) _CL_ (c) _r_


_N_ PPL Avg Acc _CL_ PPL Avg Acc _r_ PPL Avg Acc


4 0.893 0.832 128 0.932 0.864 2 0.923 0.900

8 0.925 0.883 256 0.915 0.834 8 0.933 0.902

16 0.930 0.902 512 0.896 0.820 64 0.930 0.902

64 0.923 0.896 1024 0.850 0.780 256 0.928 0.902

128 0.928 0.903 2048 0.915 0.864 1024 0.928 0.902

256 0.841 0.748 4096 0.930 0.902 4096 0.955 0.912


The results confirm that TraceNAS maintains a stable and predictive ranking signal across a wide range of hyperparameter
settings, validating that the inherited optimization landscape of the base model is reliably captured without requiring dense
calibration. We provide a detailed analysis of the results in Table 6:


18




--- end of page=17 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


_Table 7._ Kendall _τ_ correlation between various search hyperparameters and model PPL and average downstream accuracy. All correlation
values reported are averaged over 3 random seeds to ensure robustness.


(a) _N_ (b) _CL_ (c) _r_


_N_ PPL Avg Acc _CL_ PPL Avg Acc _r_ PPL Avg Acc


4 0.742 0.680 128 0.803 0.708 2 0.791 0.745

8 0.790 0.732 256 0.779 0.689 8 0.803 0.748

16 0.797 0.748 512 0.750 0.671 64 0.797 0.748

64 0.778 0.736 1024 0.712 0.636 256 0.795 0.746

128 0.790 0.739 2048 0.767 0.706 1024 0.794 0.748

256 0.707 0.613 4096 0.797 0.748 4096 0.837 0.760


  - Calibration Sample Density ( _N_ ): With as few as 4 calibration samples, the proxy achieves a strong correlation of
0.832 with average accuracy. This correlation peaks and stabilizes between _N_ = 16 and _N_ = 128 at approximately
0.90, supporting the use of relatively low sample counts during evolutionary search to maximize efficiency without
sacrificing ranking fidelity. Interestingly, the correlation dips noticeably at N=256, which may result from additional
samples introducing less optimal data points that could dilute the alignment proxy. This suggests a potential benefit
from filtering or weighting calibration samples to maintain high-quality proxy scores.


  - Context Length ( _CL_ ): Proxy correlations remain high at shorter context windows; for example, at _CL_ = 128, the PPL
correlation is 0.932 and the accuracy correlation is 0.864. Increasing context length to _CL_ = 4096 further improves
alignment, reaching an accuracy correlation of 0.902. This indicates that while local dependencies are captured early,
longer contexts enhance the proxy’s ability to predict complex reasoning performance.


  - Low-Rank Gradient Subspace ( _r_ ): The proxy shows exceptional stability across gradient subspace ranks. Even at a
very low rank of _r_ = 2, correlation with average accuracy remains at 0.900, validating the Intrinsic Dimensionality
Invariance principle: the core manifold dynamics necessary for effective pruning exist within a compact subspace.
Although correlation improves slightly to 0.912 at full rank ( _r_ = 4096), the minimal gain does not justify the additional

   - verhead, confirming the effectiveness of using lower-rank subspaces during search.


By maintaining approximately 0.90 correlation with downstream accuracy across these hyperparameters, TraceNAS provides
a scalable and reliable foundation for zero-shot model compression. This stability ensures that the identified sub-networks
consistently reside within the original pretrained convergence basin regardless of search-time resource limitations.


**C.4** **Performance Across Different Sparsities**



We evaluate TraceNAS across a range of sparsity levels and report WikiText-2
perplexity for LLaMA-2-7B in Fig. 5 after lightweight CPT with 2.5B tokens. We
generate compressed models ranging between 7B (0% sparsity) and 3.3B (50%
sparsity) parameters using TraceNAS over 200 evolutionary search iterations on
16 sequences of data from FineWeb-Edu. Across all sparsity regimes, TraceNAS
consistently achieves lower perplexity than competing width and layer pruning
baselines.


As expected, perplexity increases with sparsity for all methods; however, TraceNAS exhibits a flatter degradation curve. At 50% sparsity, TraceNAS attains
a perplexity of approximately 11, compared to roughly 15 for DarwinLM and

- ver 100 for ShortGPT. This resilience under aggressive compression suggests
that the gradient trace proxy effectively identifies architectures that remain within
the pretrained model’s convergence basin, mitigating representational collapse as
parameters are removed.



|50 ShortGPT|ShortGP|PT|Col4|Col5|Col6|
|---|---|---|---|---|---|
|0<br>10<br>20<br>30<br>40<br>50<br>Sparsity (%)<br>5<br>10<br>20<br>40<br>80<br>ShortGPT<br>~~Shortened Llama~~<br>~~EvoPress~~<br>~~DarwinLM~~<br>TraceNAS|ShortG<br>~~Shorte~~|PT<br>~~ed Llama~~||||
|0<br>10<br>20<br>30<br>40<br>50<br>Sparsity (%)<br>5<br>10<br>20<br>40<br>80<br>ShortGPT<br>~~Shortened Llama~~<br>~~EvoPress~~<br>~~DarwinLM~~<br>TraceNAS|<br>~~EvoPres~~<br>~~Darwin~~|<br>~~s~~<br>~~LM~~||||
|0<br>10<br>20<br>30<br>40<br>50<br>Sparsity (%)<br>5<br>10<br>20<br>40<br>80<br>ShortGPT<br>~~Shortened Llama~~<br>~~EvoPress~~<br>~~DarwinLM~~<br>TraceNAS|TraceN|AS||||
|0<br>10<br>20<br>30<br>40<br>50<br>Sparsity (%)<br>5<br>10<br>20<br>40<br>80<br>ShortGPT<br>~~Shortened Llama~~<br>~~EvoPress~~<br>~~DarwinLM~~<br>TraceNAS||||||
|0<br>10<br>20<br>30<br>40<br>50<br>Sparsity (%)<br>5<br>10<br>20<br>40<br>80<br>ShortGPT<br>~~Shortened Llama~~<br>~~EvoPress~~<br>~~DarwinLM~~<br>TraceNAS||||||
|0<br>10<br>20<br>30<br>40<br>50<br>Sparsity (%)<br>5<br>10<br>20<br>40<br>80<br>ShortGPT<br>~~Shortened Llama~~<br>~~EvoPress~~<br>~~DarwinLM~~<br>TraceNAS||||||
|0<br>10<br>20<br>30<br>40<br>50<br>Sparsity (%)<br>5<br>10<br>20<br>40<br>80<br>ShortGPT<br>~~Shortened Llama~~<br>~~EvoPress~~<br>~~DarwinLM~~<br>TraceNAS||||||


### Figure 5

Caption: _ **TraceNAS PPL across**

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
**sparsity levels.** WikiText-2 perplexity
reported for pruned models identified
via TraceNAS evolutionary search and
trained using 2.5B tokens of CPT.







Notably, the robustness of TraceNAS at high sparsity arises from architectural
selection rather than additional recovery training. All methods are evaluated under comparable lightweight CPT budgets, yet
TraceNAS consistently identifies sub-networks with substantially lower perplexity by prioritizing architectures that preserve

- ptimization landscape. TraceNAS leverages gradient alignment to capture global sensitivity across layers, enabling the
retention of long-range dependencies even under severe compression. The key advantage of TraceNAS lies in its ability to


19




--- end of page=18 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


discover non-uniform architecture configurations tailored to a given parameter budget. Whereas layer-dropping approaches
primarily remove entire blocks in a uniform manner, TraceNAS jointly optimizes depth and per-layer width, retaining
parameters where they are most necessary. By guiding the search using gradient alignment the resulting sub-networks
maintain geometric alignment with the base model’s optimization landscape, yielding significantly more stable performance
as sparsity increases.


**D** **Scalability & Speedup Analysis**


**D.1** **Speedup Analysis**


_Table 8._ Inference phase specific speedup analysis on NVIDIA A100. _Sp_ and _Sd_ denote the prefill and decode speedup, respectively.


Model TTFT TPOT Throughput Peak Memory _Sp_ _Sd_
(ms) (ms) (T/s) (GB) (Prefill) (Decode)


Llama-2 7B 1462.05 35.80 27.93 15.01 1.00 _×_ 1.00 _×_

ShearedLlama 2.7B 3071.05 33.24 47.92 7.88 0.48 _×_ 1.08 _×_

DarwinLM 2.7B 649.83 **12.96** **83.50** **6.07** 2.25 _×_ **2.76** _×_
TraceNAS 2.7B (Ours) 346.56 17.61 56.80 6.33 **4.22** _×_ 2.03 _×_


Llama-3.1 8B 960.25 42.55 23.50 16.52 1.00 _×_ 1.00 _×_

Minitron 4.4B 1340.86 34.61 31.70 10.49 0.72 _×_ 1.23 _×_

DarwinLM 4.6B 918.68 **18.95** **56.54** 11.53 1.05 _×_ **2.25** _×_
TraceNAS 4.6B (Ours) 556.52 26.61 37.58 **9.98** **1.73** _×_ 1.60 _×_


Qwen 2.5 14B 1927.08 79.43 12.59 30.29 1.00 _×_ 1.00 _×_
Minitron 8.4B 5361.37 72.19 20.05 18.03 0.36 _×_ 1.10 _×_

DarwinLM 8.4B 4717.75 **46.18** **29.99** 19.88 0.41 _×_ **1.72** _×_
TraceNAS 8.4B (Ours) 1322.30 54.52 18.34 **17.71** **1.46** _×_ 1.46 _×_


The practical advantages of the TraceNAS architectures are detailed in Table 8. We evaluate inference performance on
an NVIDIA A100 GPU using a 4096 token prompt and a 128 token generation window. To isolate architectural effects
under a uniform execution path, all models are evaluated using native PyTorch with a standardized inference configuration.
Results are recorded over 20 independent trials to reduce runtime variance and ensure stable, comparable measurements. As
shown in the results, our architectures demonstrate significant improvements in memory efficiency and interactive latency,
primarily driven by reductions in TTFT.


While training-aware baselines like DarwinLM demonstrate higher raw decode throughput ( _Sd_ ), our models achieve a
substantial reduction in Time To First Token (TTFT), which is a critical metric for interactive responsiveness. Specifically,
at the 2.7B scale, TraceNAS reaches a 4.22 _×_ prefill speedup ( _Sp_ ), representing a 1.97 _×_ improvement over DarwinLM’s
prefill speedup. This trend of superior prefill efficiency persists across larger scales, suggesting that our gradient alignment

- bjective effectively identifies layer configurations that minimize the computational overhead of the initial prompt processing
phase.


Furthermore, TraceNAS architectures exhibit superior memory efficiency, maintaining the lowest peak memory utilization
at the 8.4B and 14B parameter scales. For instance, our 8.4B configuration requires only 17.71 GB of peak memory,

- utperforming both Minitron and DarwinLM. While TraceNAS models do not maximize raw decode throughput, they
prioritize TTFT and memory efficiency, which are more critical for real-time, user-facing deployment.


**D.2** **Extending TraceNAS to Different Model Scales**


_Table 9._ Pruning results for Llama-3.1-70B Instruct.


Method #Params MMLU PIQA WG ArcE ArcC HS SciQ LQA BQ Avg.


Llama-3.1-70B Instruct 70B 81.22 83.89 85.9 87.2 70.39 87.44 96.12 37.21 85.2 79.17


PUZZLE 51B **80.20**     - 84.53     - **69.20 85.58**     -     -     -
DarwinLM 35B      - 81.2 83.5 82.5 60.5 80.3 95.4 33.0 84.2 75.07
TraceNAS (Ours) 40B 77.21 **82.1** **85.21 83.19** 60.32 78.21 **97.32 34.12 85.12 75.70**


We demonstrate that TraceNAS consistently identifies high-performing sub-networks across a wide range of model scales
in Tables 9 and 10. At the 70B scale, pruning Llama-3.1-70B to 40B parameters, TraceNAS maintains strong reasoning


20




--- end of page=19 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


_Table 10._ Pruning results for Pythia-2.8B and Gemma-2-2B .


Method #Params MMLU PIQA WG ArcE ArcC HS SciQ LQA BQ Avg.


Pythia-2.8B 2.8B 24.30 74.04 60.37 64.43 35.65 60.73 88.4 28.41 63.48 59.48


DarwinLM 1.4B 25.1 71.3 57.3 61.2 34.7 54.5 82.9 27.9 65.0 56.85
TraceNAS (Ours) 1.4B **25.5** **71.8** **58.69 62.83** 34.4 54.11 **84.9 28.18 64.92 57.47**


Gemma2-2B 2.5B 51.35 79.27 71.66 80.21 53.32 74.65 95.7 29.18 72.72 69.58


DarwinLM 1.2B 25.3 61.3 52.1 48.5 23.2 30.5 80.0 26.4 55.5 47.18
TraceNAS (Ours) 1.2B **26.44** **70.34 57.93 66.70 36.77 51.84 91.2 27.49 65.62 58.23**


performance, with MMLU accuracy of 77.21 and a SciQ accuracy of 97.32. These results show that TraceNAS gradient
alignment proxy effectively navigates the large-scale architecture space without search-time training. These are results are
achieved after performing post-pruning supervised finetuning (SFT) for 5000 steps using LoRA (Hu et al., 2022) and the
Orca (Mukherjee et al., 2023) finetuning dataset, further showcasing that TraceNAS can be used as a proxy for finetuning
performance.


The benefits of TraceNAS are equally evident in smaller models. For Pythia-2.8B pruned to 1.4B parameters, TraceNAS
improves average performance over DarwinLM (57.47 vs. 56.85) and achieves the highest accuracy on nearly all reasoning
tasks. For Gemma2-2B pruned to 1.2B parameters, TraceNAS dramatically reduces the performance drop seen in DarwinLM,
achieving an average of 58.23 compared to 47.18 while leading across nearly all individual tasks. The results reported for
the 2B models are after for CPT on FineWeb-Edu 100BT for 10B tokens.


Overall, these results highlight that TraceNAS is robust across scales. By leveraging gradient alignment to guide architectural
selection, it consistently produces non-uniform sub-networks that outperform training-aware pruning approaches, for both
massive 70B models or highly compressed 1.2B models.


**D.3** **Training with more tokens**


We provide results on training our Llama-2-2.7B pruned model on 10B tokens, 20B tokens and 50B tokens and compare
with baselines ShearedLlama and DarwinLM.


_Table 11._ TraceNAS LLaMA-2-2.7B model trained across 10B, 20B and 50B tokens on FineWeb-Edu 100BT subset.


Method #Params Srch Recv PIQA WG ArcE ArcC HS SciQ LQA BQ Avg.


Llama-7B 6.7B    -    - 77.69 74.11 76.34 52.81 78.96 93.80 29.80 77.73 70.16


ShearedLLaMA 2.7B 0.4B 50B **75.80** 64.20 67.00 41.20 **70.80** 90.80 28.20 63.00 62.63

DarwinLM 2.7B 1.0B 10B 73.12 63.77 69.94 43.94 66.95 90.20 28.57 **64.10** 62.57


TraceNAS (Ours) 2.7B 98M 10B 71.54 59.43 68.25 40.24 66.40 89.40 28.57 62.01 60.73
TraceNAS (Ours) 2.7B 98M 20B 73.40 63.97 70.20 43.80 68.54 91.10 **29.71** 61.76 62.81
TraceNAS (Ours) 2.7B 98M 50B 73.95 **64.35** **70.93** **44.67** 69.12 **91.28** 28.18 62.42 **63.11**


**D.4** **Evaluating TraceNAS for Unstructured Pruning**


We evaluate the capability of TraceNAS as a proxy for unstructured pruning of the dense Llama-2-7B model and evaluate
its performance after SFT. We compare 50% pruned TraceNAS model against magnitude pruning (Han et al., 2015),
FLAP (An et al., 2024), Wanda (Wang et al., 2020), ShortenedLlama (Kim et al., 2024), AmoebaLLM (Fu et al., 2024) and
PrunerZero (Dong et al., 2024b) in Table 12.


We finetune our 50% pruned model on the Orca (Mukherjee et al., 2023) dataset for 10,000 steps with LoRA (Hu et al.,
2022), AmoebaLLM SFT on Alpaca for 10,000 steps, 50% pruned and finetuned Wanda model, 50% pruned FLAP model
and the reported ShortenedLlama results. Evaluating TraceNAS under this setup showcases its robustness as a proxy for
performance potential under unstructured pruning and SFT post-pruning recovery.


The results in Table 12 showcase that TraceNAS has the highest average accuracy at 63.15% outperforming Wanda (61.81%)
and PrunerZero (62.76%). The high performance of Wanda showcases why we chose it as a mask generation metric and our
higher performance further validates that TraceNAS significantly boosts performance over Wanda.


21




--- end of page=20 ---

**TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation**


_Table 12._ Evaluating Llama-2-7B unstructured pruning using TraceNAS against SOTA unstructured pruning baselines. Average accuracy
excludes MMLU.


Method #Params MMLU WG ArcE ArcC HS BQ Avg.


Llama-2-7B 6.7B 77.69 76.34 52.81 78.96 93.80 70.16 74.41


ShortenedLlama 50% 24.76 51.54 50.63 26.45 43.60 62.23 46.89
Magnitude 50% 30.69 59.19 58.84 33.53 45.49 54.59 50.33
FLAP 50% 27.67 56.75 55.43 31.57 51.33 59.45 50.91

AmoebaLLM 50% 30.60 60.10 56.60 31.50 51.30 65.70 53.04

Wanda 50% **39.27** 68.19 72.77 39.59 52.49 **75.99** 61.81

PrunerZero 50%             - 67.80 69.19 38.82 **69.12** 68.87 62.76
TraceNAS (Ours) 50% 38.76 **68.83 73.86 42.51** 56.42 74.13 **63.15**


**E** **Generation Quality Example**


_Table 13._ **Qualitative Analysis:** TraceNAS (2.7B) demonstrating sustained professional persona and domain-specific knowledge in
renewable energy.


**Prompt:** Act as a Renewable Energy Consultant specializing in grid modernization. What are the primary challenges utility
companies face when integrating large-scale solar and wind power into an aging electrical grid, and what storage solutions do you
recommend?

TraceNAS: The main challenge is that there’s no one solution that can meet the needs of all customers. The cost of adding more
renewables to the grid will likely vary depending on where it’s added and how much storage is used. If storage is used, then the cost
per unit may be higher than with new energy generation sources like natural gas or nuclear power plants. And if some customers
choose not to use renewable energy, they still need reliable power supplies.
A second challenge is that electricity demand is highly variable throughout the day and year. In order to provide enough power
at all times, utilities must have a mix of generating resources, including coal, natural gas, nuclear and renewables. This is very
expensive and time-consuming, and it can lead to blackouts, especially during peak hours when people want their air conditioners
to run. What is your favorite renewable energy project to date?
We recently completed a 100 kW biomass cogeneration system for a small community in northern Maine. Biomass cogeneration,

  - r producing both heat and electricity from biomass, has long been underutilized because it’s difficult to get a contractor to work

  - n a site that doesn’t have access to water. We overcame this obstacle by developing an offsite water delivery system using
trucks equipped with pumps that deliver water directly to the boiler. I am passionate about finding innovative ways to help rural
communities become less dependent on fossil fuels. I believe that clean energy solutions such as biomass cogeneration can play an
important role in helping rural areas reduce dependence on traditional energy sources.


**F** **Limitations**


TraceNAS provides an efficient zero-shot metric for model pruning, yet several constraints remain. The proxy Φ captures
immediate functional alignment but does not explicitly model loss landscape curvature or smoothness, factors which may
influence long-term convergence. Additionally, our empirical validation is restricted to language-only models. Although
the framework is extensible, its performance on multi-modal architectures has not yet been verified. From a deployment
perspective, our search process does not incorporate hardware-specific latency bottlenecks or inference-engine optimizations
such as vLLM. Furthermore, we do not perform a full multi-objective search to map the entire Pareto frontier between size,
speed, and accuracy, focusing instead on validating the Φ proxy under fixed architectural constraints. Future work will
explore hardware-aware metrics and cross-modal generalizability to broaden the utility of zero-shot structured pruning.


22




--- end of page=21 ---
