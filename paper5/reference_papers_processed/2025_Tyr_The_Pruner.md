---
id: "2025_Tyr_The_Pruner"
title: "Tyr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization"
authors: ["Guanchen Li", "Yixing Xu", "Zeping Li", "Ji Liu", "Xuanwu Yin", "Dong Li", "Emad Barsoum"]
year: 2025
venue: "NeurIPS 2025"
publication_status: "FORMALLY PUBLISHED"
category: "Direct Neighbor / Architecture Interaction"
source_pdf: "../reference_papers_origin/2025_Tyr_The_Pruner.pdf"
paper_url: "https://proceedings.neurips.cc/paper_files/paper/2025/hash/db2cbf43a349bc866111e791b58c7bf4-Abstract-Conference.html"
pdf_url: "https://proceedings.neurips.cc/paper_files/paper/2025/file/db2cbf43a349bc866111e791b58c7bf4-Paper-Conference.pdf"
code_url: "UNVERIFIED"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# Tyr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization

**Authors:** Guanchen Li, Yixing Xu, Zeping Li, Ji Liu, Xuanwu Yin, Dong Li, Emad Barsoum

**Venue / Year:** NeurIPS 2025 (FORMALLY PUBLISHED)

**Category:** Direct Neighbor / Architecture Interaction

**Why this paper matters for low-cost post-training LLM NAS:** It directly addresses non-uniform global LLM sparsity allocation, inter-structure dependence, efficient supernet construction, and coarse-to-fine search.

**Primary record:** [https://proceedings.neurips.cc/paper_files/paper/2025/hash/db2cbf43a349bc866111e791b58c7bf4-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2025/hash/db2cbf43a349bc866111e791b58c7bf4-Abstract-Conference.html)

**Local source:** [2025_Tyr_The_Pruner.pdf](../reference_papers_origin/2025_Tyr_The_Pruner.pdf)

## Full converted text

# **Týr-the-Pruner: Structural Pruning LLMs via** **Global Sparsity Distribution Optimization**

**Guanchen Li, Yixing Xu, Zeping Li, Ji Liu, Xuanwu Yin, Dong Li, Emad Barsoum**
Advanced Micro Devices, Inc. (AMD)
```
 {guanchen,yixing.xu,zeping.li,ji.liu,xuanwu.yin,d.li,emad.barsoum}@amd.com

```

## Abstract


Structural pruning enhances hardware-agnostic inference efficiency for large language models (LLMs) yet often fails to maintain comparable performance. Local
pruning performs efficient layer-by-layer compression but ignores global topology.
Although global pruning aims to identify an optimal sparse model, intuitive meth
     - ds typically adopt a two-stage paradigm that first evaluates substructure saliency
and then applies global pruning, which ignores inter-structure dependencies and
fails to achieve end-to-end optimization. To address these limitations, we propose
**Týr-the-Pruner**, an efficient end-to-end search-based global structural pruning
framework. This framework constructs a supernet by repeatedly applying local
pruning across a range of sparsity ratios to each layer in an LLM, with the core
goal of determining the optimal sparsity distribution under a target overall sparsity
ratio. Concretely, we introduce an effective local pruning and an expectation error
accumulation approach to improve supernet construction. Furthermore, we employ
an iterative prune-and-search strategy with coarse-to-fine sparsity granularity to
ensure efficient search convergence. Experimental results show that Týr-the-Pruner
achieves state-of-the-art structural pruning, retaining **97%**     - f the dense model’s
performance while removing a challenging **50%**     - f Llama-3.1-70B’s parameters.


**1** **Introduction**


Large language models (LLMs) have significantly advanced natural language processing, achieving
exceptional performance in tasks such as text understanding, generation, and reasoning [53, 7, 3].
However, the computational and storage resources required for model deployment incur high costs
and environmental impacts, limiting their accessibility in resource-constrained scenarios. Model
compression techniques, such as quantization [21, 10], pruning [9, 23, 39], and low-rank decomposition [45], are essential for reducing LLM size and computational demands. This paper focuses on
structural pruning, which enhances inference efficiency in a hardware-agnostic manner.


Existing structural pruning methods for LLMs are typically classified into local and global techniques.
Local pruning methods [17, 26], which prune layers individually, enable efficient compression of
hundred-billion-scale LLMs on a single GPU via offload approaches. However, they overlook global
dependencies in model topology and restrict the sparsity to be uniform across layers. Global pruning
methods [23, 18, 1] alleviate local constraints, facilitating sparsity allocation and the potential for

- ptimal pruning. However, many existing methods estimate the saliency of local substructures and
prune them accordingly via global ranking, ignoring inter-structure dependencies and hindering endto-end optimization. Such methods may also suffer from the inefficiency of backpropagation-based
saliency estimation and overfitting when calibration data is limited. Therefore, a question arises:


_How to achieve_ _**efficient global**_ _structural pruning with_ _**end-to-end**_ _optimization?_


39th Conference on Neural Information Processing Systems (NeurIPS 2025).




--- end of page=0 ---

To address this challenge, we propose **Týr-the-Pruner**, an efficient search-based global pruning
framework with end-to-end optimization. Our framework constructs a supernet by applying local
pruning to each layer, producing pruned copies with different sparsity ratios. The objective is
to identify an optimal subnet that satisfies the target overall sparsity ratio within the supernet by
determining the optimal sparsity distribution across layers. We use evolutionary search [22] to solve
this optimization problem. To construct reliable supernets and perform effective and efficient search,
we make the following contributions:


    - **To improve supernet construction**, we propose an effective local pruning approach for
attention heads and feed-forward networks (FFN), using Taylor expansion-based first- and
second-order optimization information to identify redundant structures and adjust remaining
weights. Pruning and weight adjustments are applied progressively and finely to preserve
accuracy. Additionally, we introduce an expectation error accumulation approach to address
the challenge of unclear error propagation caused by the multiple pruned copies within the
supernet. This approach ensures balanced mutual awareness across sparse structures during
supernet construction.


    - **To enhance the efficacy and efficiency of subnet search**, we employ a tailored distillationinspired metric as the optimization objective to guide the search process, aiming to preserve
the subnet’s generative capability. In general, Týr-the-Pruner is formed as an iterative pruneand-search framework that refines sparsity allocation for each layer with reduced search
space and fast convergence. Each iteration prunes and constructs a supernet across a specific
range of sparsity ratios, coupled with a sparsity-shift-driven evolutionary search, where
random sparsity shifts between layers generate parent candidates, and the best-performing

     - nes are filtered as offspring. The sparsity interval is refined after each iteration.


By making these contributions, Týr-the-Pruner achieves end-to-end global pruning with strong
efficacy and efficiency. Notably, the proposed framework only requires 4M tokens for calibration
and search. Experimental results demonstrate that Týr-the-Pruner surpasses state-of-the-art pruning
methods. For example, Týr-the-Pruner outperforms the SOTA method FLAP, achieving 3.45 lower
perplexity in language comprehension and 10.26% higher average downstream accuracy when
pruning 37.5% of the parameters of Llama-3.1-8B. Moreover, it maintains 97% performance with
50% pruning on Llama-3.1-70B, a sparsity ratio that is considered aggressive for existing methods.


**2** **Method**


This section presents Týr-the-Pruner, a novel structural pruning framework for large language models
(cf. Section 2.1 for preliminaries), as illustrated in Figure 1. This framework **(1) constructs a**
**supernet** by applying local pruning across various sparsity ratios to each model layer, aiming to
**(2) search the optimal sparsity distribution** under a target overall sparsity ratio. Specifically, we
propose an effective local pruning approach (cf. Section 2.2) and an expectation error accumulation
approach (cf. Section 2.3) to enhance supernet construction. An iterative prune-and-search strategy
with coarse-to-fine sparsity granularity (cf. Section 2.4) ensures efficient search convergence.


**2.1** **Preliminaries**


Large language models typically use the Transformer decoder architecture [42], as shown in Figure 1(a). Each Transformer layer consists of two key components: the multi-head self-attention
(MHA) and the feed-forward network (FFN), followed by a residual connection and layer normalization. Given the input **X** _ℓ−_ 1 to the _ℓ_ - th layer, the output **X** _ℓ_ can be expressed as:


**X** = LayerNorm ( **X** _ℓ−_ 1 + MHA( **X** _ℓ−_ 1)) _,_
(1)
**X** _ℓ_ = LayerNorm ( **X** + FFN( **X** )) _._


The MHA mechanism captures dependencies across different positions in the input sequence with
multiple attention heads, each with its query ( **W** _q_ ), key ( **W** _k_ ), value ( **W** _v_ ), and out ( **W** _o_ ) linear
transformations. Modern LLMs typically employ a SwiGLU-based FFN [34], consisting of gate
( **W** _gate_ ), up ( **W** _up_ ), and down ( **W** _down_ ) linear transformations, with activation after the gate. This
structure aids in extracting non-linear representations.


2




--- end of page=1 ---

_̸_



_̸_



_̸_



_̸_



### Figure 1

Caption: **An overview for Týr-the-Pruner** . Large language models (a) will be effectively locally

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
pruned across multiple sparsity ratios and constructed into a supernet (b). An iterative prune-andsearch strategy will be used to select the optimal sparse structure for each layer while maintaining a
target overall sparsity ratio: pruning and sparsity-shift-driven evolutionary search are implemented
iteratively with a coarse-to-fine sparsity interval granularity (c). Ultimately, the post-pruned LLM
with the optimal sparsity distribution (d) is obtained.


Structural pruning for LLMs can be applied across four key dimensions: (1) attention heads, (2) FFN
intermediate neurons, (3) embedding dimension size, and (4) model depth. It can be isotropic (uniform
sparsity across layers) or non-isotropic (layer-specific sparsity). This paper focuses on pruning
attention heads and FFN intermediate neurons with non-uniform sparsity: pruning functionally
independent heads and neurons allows for controllable accuracy loss, while layer-specific sparsity
further enhances pruning by tailoring compression to each layer’s characteristics.


**2.2** **Effective Local Pruning**


**Redundant structure identification and weight adjustment** . When pruning is scoped to the local
level, one can determine the pruning outcome by eliminating the redundant input channels of each

`o_proj` and `down_proj` modules, with a consistent sparsity across layers. Assuming the weight of
a layer is **W** _∈_ R _[d]_ [in] _[×][d]_ [out] and its input activation is **X** _∈_ R _[d]_ [N] _[×][d]_ [in], the pruned weight **W** [�] satisfies the
sparsity constraint _C_ . The corresponding optimization objective is expressed as:


argmin **W**           - _[||]_ **[XW]** _[ −]_ **[X]** **W** [�] _||_ 2 [2] s.t. _C_ ( **W** [�] ) = _C._ (2)


The pruning process can be viewed as a perturbation applied to the weights: **W** [�] = **W** _−_ _δ_ **W** . Therefore, the error function is given by _E_ = _∥_ **XW** _−_ **XW** [�] _∥_ 2 [2] [=] _[ ∥]_ **[X]** _[δ]_ **[W]** _[∥]_ [2] 2 [, which can be approximated]
by a Taylor series expansion around **W** and whose local fluctuations can be defined as:


_̸_



_∂E_
_δE_ =

  - _∂_ **W**


_̸_




- _⊤_


_̸_




[1]

2 _[δ]_ **[W]** _[⊤]_ _∂_ _[∂]_ **W** [2] _[E]_


_̸_



_δ_ **W** + [1]


_̸_



_∂_ **W** [2]

~~�~~ - ��
**H** =0 _̸_



_δ_ **W** + _O_ - _∥δ_ **W** _∥_ [3][�]

~~�~~    - ~~�~~ ~~�~~
_≈_ 0

_̸_



_._ (3)


_̸_




~~�~~ **G** _[⊤]_ ~~��~~ _̸≈_ 0 - _̸_



_̸_


_δE_ reflects the effect of _δ_ **W** - n the pruning error, which we aim to minimize. The first-order gradient
**G** cannot be neglected, as the calibration samples are inevitably misaligned with the proprietary
closed-source pre-training data. The Hessian matrix **H** helps to identify pruning-sensitive weights
from a curvature perspective. Considering the sparsity constraint ( _δ_ **W** _p,_ : = **W** _p,_ :: the _p_ - th input
channel of **W** is to be pruned), we design the redundant channels and weight adjustment as follows:


3




--- end of page=2 ---

**W** _p,_ : = argmin **W** _p,_ :




- �� **G** _p,_ : **W** _p,⊤_ :�� + 2 [ _∥_ **WH** _[−]_ _p,_ [1] :] _∥p,p_ 2 [2]



_, δ_ **W** _∼p,_ : = _−_ **H** _[−]_ _∼_ [1] _p,∼p_ **[G]** _[∼][p,]_ [:] _[.]_ (4)



**H** = **X** _[⊤]_ **X** and **G** = **HW** (analytic solutions computed without backpropagation, efficient) are
used as estimates of the local optimization information. The channel _p_ with the least error impact
is identified and pruned, while _δ_ **W** adjusts the remaining weights to compensate for pruning errors

_∼_
( _p_ represents other channels that have not been pruned).


**Pruning heads and neurons** . In our framework, feed-forward network neurons are pruned based

- n individual channel saliency computed from the `down_proj` layer, where each channel acts as the
atomic unit for ranking and removal. For multi-head self-attention, saliency is first computed per

- utput channel of the `o_proj` layer, then aggregated (averaged) across channels belonging to the
same head, which is treated as the atomic unit for pruning.


**Progressively pruning and weight adjustment** . We adopt progressive pruning with an appropriately
fine granularity: finer granularity enables unpruned weights to gradually and uniformly compensate
for pruning losses in small increments while enabling precise and dynamic redundant channel
identification. Reducing granularity does not significantly complicate pruning, as the key intermediate
variable **H** _[−]_ [1] can be rapidly adjusted to account for partial channel pruning in _O_ ( _d_ [2] in [)][ complexity [][8][].]


Detailed analysis can be found in Section A.1.


**2.3** **Prune-to-supernet across Multiple Sparsity Ratios**



As illustrated in Figure 1(b), a super- Perplexity@Wikitext2 Perplexity@Wikitext2
net will be constructed by repeatedly Llama-3.1-8B: **5.84** Llama-3.1-8B: **5.84**
applying local pruning across a range Local pruned 50%: **538.23** Local pruned 50%: **58.09**

- f sparsity ratios to each LLM layer, Dense Dense Dense Dense
producing pruned copies with varying 𝑋 𝑋
sparsity ratios. However, this introduces Pruned To-prune Pruned To-prune
challenges in error accumulation across
layers. Error accumulation introduces Layer 𝑙 Layer 𝑙+ 1 Layer 𝑙 Layer 𝑙+ 1
an additional forward pass of the post- (a) w/o Error Accum. (b) w/ Error Accum.
pruned layer, using its output activation

|Col1|Dense|Col3|
|---|---|---|
|𝑋|𝑋|𝑋|
|𝑋|P|runed|
||||
||Layer𝑙<br>|Layer𝑙<br>|


|P|runed|Col3|Col4|To-prune|Col6|
|---|---|---|---|---|---|
|P|runed|||||
|Layer𝑙<br>|Layer𝑙<br>|Layer𝑙<br>|Layer𝑙+ 1<br>|Layer𝑙+ 1<br>|Layer𝑙+ 1<br>|


### Figure 2

Caption: Implementing layerwise error accumulation gives

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.

as input for the next layer. The change

a more accurate pruning result than not. Solid lines indicate

in the input directly affects the optimiza
forward propagation, and dashed lines indicate pruning.

tion of the subsequent layer. In the example shown in Figure 2, pruning half of Llama-3.1-8B’s parameters using the local pruning approach
with error accumulation results in significantly lower language comprehension perplexity than pruning
without it. This performance gap highlights the critical role of error accumulation: it enables deeper
layer pruning to be aware of shallower layer pruning.


The existence of multiple sparse structures complicates error accumulation, making it unclear which
pathway to prioritize. To address this issue, we propose an _expectation error accumulation_ approach
to enable balanced mutual awareness among the sparse structures in the supernet. Let the output
activation of the _e_ - th sparse structure with sparsity _Se_ in layer _ℓ_ be **X** _ℓ_ +1 _,e_ . We define the expectation

- utput activation **X** _ℓ_ +1 for this layer as:



Perplexity@Wikitext2
Llama-3.1-8B: **5.84**

Local pruned 50%: **538.23**



Perplexity@Wikitext2
Llama-3.1-8B: **5.84**

Local pruned 50%: **58.09**















(a) w/o Error Accum.



(b) w/ Error Accum.



### Figure 2

Caption: Implementing layerwise error accumulation gives

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
a more accurate pruning result than not. Solid lines indicate
forward propagation, and dashed lines indicate pruning.




~~�~~ _Ee_ =11 _−_ [(1] _S_ _[ −]_ _e_ _[S][e]_ [)] **X** _ℓ_ +1 _,e._ (5)



**X** _ℓ_ +1 =



_E_



_e_ =1



A higher scaling factor is assigned to low sparsity weights because their output activations are more
stable and reliable. By enabling expectation error accumulation and applying the local pruning
approach, we can prune Llama-3.1-8B to create nine sparse structures in each layer, with 12.5% as the
sparsity interval (covers complete pruning and abandoned pruning). The post-pruned model achieves
a language comprehension perplexity of 66.38 on the WikiText-2 task, with manually picking the
50% sparse structure as an example. This result is close to the ideal perplexity of 58.09 achieved
under full error accumulation and significantly better than the 208.92 perplexity from random error
accumulation and 538.23 perplexity with abandoned error accumulation.


4




--- end of page=3 ---

**2.4** **Týr-the-Pruner**


By introducing effective local pruning and expectation error accumulation approaches, we can
construct a supernet to tackle the global sparsity allocation problem. Specifically, we address the
following issues to achieve efficient and effective sparsity allocation: (1) defining generalizable
criteria for selecting a better sparse structure, (2) achieving an efficient search-based sparse structure
selection while maintaining overall sparsity, and (3) handling the contradiction between fine-grained
sparsity intervals and the large search space.


**Align to dense model behaviors to win** . Towards the definition of better sparse structures, we
consider that large language models are designed for multi-task generalization. Thus, guiding sparse
structure selection on a single task risks overfitting. To mitigate this, we adopt a distillation-inspired
metric to measure the similarity between sparse and dense models. A salient similarity indicates
that the current sparse structure is better aligned with the dense model, making it more suitable
for selection. Specifically, let **h** [dense] _ℓ_ and **h** [sparse] _ℓ,e_ denote the activations of the dense and _e_ - th sparse
(structure) models at layer _ℓ_, and **z** [dense] and **z** [sparse] _{e}_ represent the logits of the dense model and selected
( _{e}_ = _{eℓ}_ _[L]_ _ℓ_ =1 [) sparse subnet. The optimization objective is formulated as follows:]



2
_αℓ_  - �� **h** dense _ℓ_ _−_ **h** [sparse] _ℓ,e_  - ��2
_ℓ_



_{e_ ˆ _}_ = argmin _{e}_


2 [+] _[ β]_ [ KL][(] **[z]** [dense] _[||]_ **[z]** _{_ [sparse] _e}_ [)] _[.]_ (6)



**Sparse structure selection via evolutionary search** . Evolutionary search can achieve convergence
in model architecture optimization [36, 22]. Compared to intuitive router training, evolutionary
search requires no additional parameters. It maintains constant overall sparsity by shifting sparsity
between sparse structures from different layers, whereas router training relies on penalty terms for
suboptimal soft sparsity control. Evolutionary search is efficient, as it allows the just-in-time loading
(cf. Section 3.5) of sparse structures and leverages the backpropagation-free feature.


Mutation (stochastic perturbation) in our evolutionary search arises from sparsity shifts across layers
(cf. _Select Sparse Structure_ in Figure 1). For instance, the sparsity of the _ℓ_ - th layer may decrease
by _s_ %, while the _ℓ_ _[′]_ - th layer increases by _s_ % (achieved by selecting different sparse structures). In
each generation, we randomly generate such a group of sparsity distributions as candidates. Starting
from the root generation, the performance of candidates is evaluated, and the best-performing ones
are selected to generate new candidates for the next generation. Generations continue to be explored
until the optimal sparsity distribution is found.


**Iterative prune-to-supernet and evolutionary search** . The search space for selecting sparse
structures with fine-grained sparsity is enormous. For instance, constructing a supernet with a sparsity
interval of 1.5625% would result in 65 sparse structures per MHA/FFN layer. For a 40-layer LLM,
this would lead to over 5K sparse structures, creating a 10 [145] - scaled search space. Identifying
solutions in this large search space is difficult and costly. To address this challenge, Týr-the-Pruner
adopts an iterative prune-and-search strategy that progressively refines sparsity granularity. In each
iteration, we (1) prune the model, (2) construct a supernet with a narrower sparsity interval, and (3)
perform evolutionary search to locate the optimal sparsity distribution. This coarse-to-fine paradigm
improves both search efficiency and accuracy in sparse structure discovery, as illustrated below.





The algorithmic procedures for local pruning, supernet construction, evolutionary search, and the

- verall Týr-the-Pruner framework are detailed in Algorithms 1 to 4 of Section A.2.


5




--- end of page=4 ---

Table 1: **Post pruning performance comparison of different methods** . Language comprehension
perplexity is validated on the Wikitext2 test set with a sequence length of 4096, where a lower value
reflects better performance. Downstream accuracy (%, higher is better) is averaged across ARC-Easy,
ARC-Challenge, BoolQ, HellaSwag, OpenbookQA, RTE, WinoGrande, and MMLU, with MMLU
using a 5-shot benchmark and others a 0-shot benchmark. The best results are shown in **bold** .












|Sparsity|Method|PerplexityonWikitext2↓|Col4|Col5|AverageDownstreamAccuracy(%)↑|Col7|Col8|
|---|---|---|---|---|---|---|---|
|**Sparsity**|**Method**|Llama-2|Llama-3.x|Mistral|Llama-2|Llama-3.x|Mistral|
|**Sparsity**|**Method**|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3<br>Nemo|7B<br>13B|2-3B 0-8B 1-8B|7B-v0.3 Nemo|
|0%|N/A|5.12<br>4.57|7.29<br>5.76<br>5.84|4.95<br>5.35|57.96 62.05|57.01 64.08 64.77|63.72<br>66.24|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|8.86<br>5.67<br>7.52<br>5.69<br>8.25<br>7.19<br>6.24<br>6.09<br>6.11<br>5.17<br>5.86<br>5.21<br>5.94<br>5.21<br>6.11<br>5.75<br>**5.84**<br>**5.03**|12.42<br>13.90<br>13.14<br>12.25<br>10.12<br>9.98<br>18.71<br>20.46<br>22.10<br>182.24<br>86.91<br>18.46<br>11.14<br>8.24<br>8.26<br>11.32<br>10.37<br>9.30<br>11.11<br>10.15<br>9.87<br>10.25<br>8.34<br>8.07<br>**9.16**<br>**7.39**<br>**7.41**|7.58<br>7.72<br>7.46<br>7.95<br>7.00<br>9.74<br>6.86<br>7.27<br>6.17<br>6.79<br>5.84<br>7.62<br>5.75<br>7.04<br>6.18<br>7.68<br>**5.61**<br>**6.31**|53.27 59.16<br>53.23 57.26<br>55.89 59.70<br>55.40 57.41<br>53.38 59.78<br>55.85 61.91<br>55.29 61.94<br>54.63 57.55<br>**56.98 62.66**|53.13 57.75 58.50<br>  52.46 59.41 60.36<br>  51.64 57.55 56.82<br>  38.02 33.95 47.89<br>  46.98 53.96 54.04<br>  51.37 57.55 57.54<br>  52.23 57.19 58.53<br>  47.74 55.72 56.66<br>**  54.78 62.01 63.02**|59.49<br>59.46<br>58.67<br>59.96<br>59.67<br>53.27<br>59.44<br>56.82<br>55.26<br>58.23<br>62.46<br>60.24<br>62.06<br>53.89<br>59.51<br>57.67<br>**63.05**<br>**64.15**|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|23.41<br>17.94<br>18.84<br>9.00<br>16.84<br>12.50<br>9.21<br>19.92<br>11.56<br>7.11<br>7.49<br>6.65<br>**7.46**<br>9.19<br>8.31<br>7.50<br>7.51<br>**5.79**|1464.20 4836.41 3418.83<br>128.77<br>124.86<br>137.17<br>45.44<br>47.73<br>55.43<br>94.12<br>48.95<br>962.72<br>25.14<br>18.65<br>19.35<br>43.50<br>28.74<br>52.69<br>122.63<br>17.40<br>17.03<br>15.64<br>**12.65**<br>12.30<br>**12.53**<br>13.14<br>**10.38**|35.20<br>124.20<br>22.91<br>20.79<br>12.08<br>19.37<br>17.83<br>15.34<br>10.24<br>11.81<br>7.39<br>9.91<br>7.16<br>9.57<br>8.01<br>13.59<br>**7.08**<br>**7.87**|46.68 51.86<br>45.47 52.77<br>51.40 58.04<br>49.92 38.17<br>44.09 49.56<br>52.59 60.50<br>51.99 59.55<br>49.36 54.37<br>**54.64 61.16**|41.25 38.12 38.62<br>  46.26 48.58 49.80<br>  45.87 50.01 48.49<br>  33.93 34.53 32.40<br>  39.55 42.36 40.88<br>  41.61 38.72 39.20<br>  33.29 44.27 42.19<br>  44.01 47.41 49.20<br>**  51.72 58.50 58.66**|51.07<br>51.68<br>51.84<br>53.65<br>52.26<br>46.27<br>49.13<br>41.30<br>46.32<br>45.26<br>58.05<br>45.59<br>55.94<br>45.95<br>52.64<br>48.83<br>**60.22**<br>**60.61**|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|70.96<br>52.24<br>87.77<br>96.00<br>35.10<br>26.22<br>19.97<br>34.70<br>37.75<br>14.96<br>12.13<br>13.01<br>11.28<br>12.74<br>12.41<br>11.33<br>**10.29**<br>**7.17**|554.88 5.1E+04 9.3E+04<br>494.07<br>1645.83 1377.02<br>98.41<br>176.81<br>237.50<br>344.17<br>2422.78 3627.00<br>161.10<br>87.93<br>70.93<br>283.53<br>50.36<br>125.98<br>182.00<br>27.69<br>28.87<br>**26.05**<br>22.61<br>21.54<br>27.88<br>**21.64**<br>**18.09**|2347.69<br>864.38<br>429.78<br>462.92<br>27.68<br>38.46<br>31.85<br>74.87<br>24.90<br>32.10<br>14.01<br>15.53<br>10.43<br>16.00<br>11.81<br>27.01<br>**10.25**<br>**11.47**|43.66 43.13<br>41.55 47.60<br>43.80 51.83<br>40.45 35.69<br>35.96 40.36<br>47.53 51.89<br>47.42 51.74<br>43.51 48.54<br>**52.21 58.67**|41.28 39.16 38.97<br>  40.24 38.89 38.85<br>  37.40 39.96 38.97<br>  33.08 30.59 32.56<br>  33.26 32.40 32.53<br>  33.35 34.77 36.55<br>  32.76 40.81 39.87<br>  39.28 41.51 43.07<br>**  46.11 53.66 53.46**|35.80<br>42.52<br>40.44<br>42.88<br>43.30<br>39.55<br>38.13<br>33.59<br>37.94<br>37.42<br>48.90<br>44.86<br>48.91<br>45.81<br>44.90<br>45.57<br>**52.34**<br>**54.63**|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|226.40 187.23<br>256.71 1129.00<br>65.34<br>50.66<br>122.28<br>47.89<br>117.40<br>53.96<br>32.91<br>24.70<br>28.41<br>44.17<br>25.49<br>16.89<br>**16.17**<br>**9.59**|2313.30 1473.71 1678.15<br>  6019.01 2.1E+04 5.4E+04<br>205.09<br>384.04<br>353.21<br>262.92<br>187.41<br>188.47<br>473.50<br>302.15<br>288.32<br>356.02<br>102.76<br>366.34<br>320.14<br>80.90<br>198.87<br>272.98<br>82.12<br>134.28<br>**29.84**<br>**38.59**<br>**30.89**|5532.76 6804.52<br>     6019.01 5.9E+04<br>54.66<br>69.15<br>91.34<br>293.59<br>74.04<br>469.93<br>24.18<br>24.96<br>29.58<br>23.14<br>34.81<br>79.46<br>**15.53**<br>**16.85**|36.99 39.47<br>       34.89 41.79<br>39.43 43.84<br>32.26 35.82<br>31.70 35.17<br>32.60 42.66<br>39.46 40.40<br>39.84 44.04<br>**47.41 54.58**|34.09 37.51 36.52<br>         33.96 35.21 33.28<br>  33.52 34.55 34.32<br>  32.29 33.86 32.39<br>  30.97 31.63 31.58<br>  32.51 33.14 34.45<br>  33.85 32.58 34.16<br>  33.29 38.68 36.59<br>**  41.41 47.41 47.79**|35.05<br>38.00<br>33.93<br>33.25<br>36.17<br>34.95<br>33.59<br>32.27<br>32.64<br>32.89<br>39.93<br>38.42<br>40.95<br>37.99<br>40.57<br>39.34<br>**46.21**<br>**47.92**|



**3** **Experiments**


**3.1** **Experimental Settings**


**Models** . We conduct experiments using the widely adopted large language models Llama2, Llama3.x,
and Mistral [41, 7, 15], focusing on models with over three billion parameters. The pruning targets
include attention heads and FFN neurons, which are applied to the Transformer backbone. The
`embed_tokens` and `lm_head` layers remain unchanged.


**Calibration** . For calibration, we consider FineWeb [31], a high-quality dataset curated from Common
Crawl snapshots with rigorous deduplication and filtering. Specifically, we extract about 4M tokens
(about 1k samples for a maximum input length of 4k) from its FineWeb-Edu subset to construct
calibration samples, ensuring high data quality and efficiency.


**Evaluation** . We use perplexity as one evaluation metric for language comprehension performance

[9], validated on the WikiText2 [27] test set. To evaluate the impact of compression across various
downstream tasks, we report 0-shot accuracy on ARC [6], BoolQ [5], HellaSwag [51], OpenBookQA

[28], RTE [43], and WinoGrande [33] tasks, as well as 5-shot accuracy on the MMLU [13] benchmark.


**Implementation details** . We implement Týr-the-Pruner with PyTorch [30] and leverage the HuggingFace Transformers and Datasets libraries [47] to manage models and datasets. For local pruning,


6




--- end of page=5 ---

Table 2: **Post pruning performance on massive language models** . Accuracy (%, higher is better)
serves as the comparison metric. MMLU employed a 5-shot benchmark, while other tasks used
0-shot benchmarks. The percentage of average accuracy maintenance after pruning was recorded,
with values _≥_ 95% highlighted in green and values _<_ 95% in red. The best results are shown in **bold** .














|Model|Sparsity|Method|Arc-C Arc-E BoolQ HellaSwag OBQA RTE WinoGrande MMLU AVG|
|---|---|---|---|
|Llama-2-70B<br>0%<br>N/A<br>54.44<br>82.74<br>83.73<br>64.77<br>37.40<br>67.87<br>77.98<br>68.79<br>67.22 (100%)<br>50%<br>SliceGPT<br>38.65<br>68.39<br>69.63<br>38.40<br>25.00<br>63.54<br>67.40<br>50.20<br>52.65 (78%)<br>LLM-Pruner<br>21.93<br>29.08<br>43.18<br>26.26<br>14.00<br>51.62<br>49.25<br>23.77<br>32.39 (48%)<br>ZipLM<br>46.67<br>77.61<br>82.26<br>56.94<br>34.00<br>68.95<br>75.61<br>54.33<br>62.05 (92%)<br>OSSCAR<br>**48.21**<br>78.37<br>81.99<br>57.00<br>32.60<br>67.15<br>76.64<br>56.05<br>62.25 (93%)<br>FLAP<br>40.02<br>70.79<br>74.74<br>51.83<br>32.00<br>60.29<br>67.88<br>39.65<br>54.65 (81%)<br>Týr-the-Pruner<br>**48.21**<br>**79.12**<br>**83.18**<br>**60.04**<br>**35.20**<br>**70.76**<br>**78.14**<br>**60.58**<br>**64.40 (96%)**|0%|N/A|54.44<br>82.74<br>83.73<br>64.77<br>37.40<br>67.87<br>77.98<br>68.79<br>67.22 (100%)|
|Llama-2-70B<br>0%<br>N/A<br>54.44<br>82.74<br>83.73<br>64.77<br>37.40<br>67.87<br>77.98<br>68.79<br>67.22 (100%)<br>50%<br>SliceGPT<br>38.65<br>68.39<br>69.63<br>38.40<br>25.00<br>63.54<br>67.40<br>50.20<br>52.65 (78%)<br>LLM-Pruner<br>21.93<br>29.08<br>43.18<br>26.26<br>14.00<br>51.62<br>49.25<br>23.77<br>32.39 (48%)<br>ZipLM<br>46.67<br>77.61<br>82.26<br>56.94<br>34.00<br>68.95<br>75.61<br>54.33<br>62.05 (92%)<br>OSSCAR<br>**48.21**<br>78.37<br>81.99<br>57.00<br>32.60<br>67.15<br>76.64<br>56.05<br>62.25 (93%)<br>FLAP<br>40.02<br>70.79<br>74.74<br>51.83<br>32.00<br>60.29<br>67.88<br>39.65<br>54.65 (81%)<br>Týr-the-Pruner<br>**48.21**<br>**79.12**<br>**83.18**<br>**60.04**<br>**35.20**<br>**70.76**<br>**78.14**<br>**60.58**<br>**64.40 (96%)**|50%|SliceGPT<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|38.65<br>68.39<br>69.63<br>38.40<br>25.00<br>63.54<br>67.40<br>50.20<br>52.65 (78%)<br>21.93<br>29.08<br>43.18<br>26.26<br>14.00<br>51.62<br>49.25<br>23.77<br>32.39 (48%)<br>46.67<br>77.61<br>82.26<br>56.94<br>34.00<br>68.95<br>75.61<br>54.33<br>62.05 (92%)<br>**48.21**<br>78.37<br>81.99<br>57.00<br>32.60<br>67.15<br>76.64<br>56.05<br>62.25 (93%)<br>40.02<br>70.79<br>74.74<br>51.83<br>32.00<br>60.29<br>67.88<br>39.65<br>54.65 (81%)<br>**48.21**<br>**79.12**<br>**83.18**<br>**60.04**<br>**35.20**<br>**70.76**<br>**78.14**<br>**60.58**<br>**64.40 (96%)**|
|Llama-3.1-70B<br>0%<br>N/A<br>60.58<br>87.29<br>85.29<br>66.50<br>37.00<br>70.04<br>79.64<br>78.72<br>70.63 (100%)<br>50%<br>SliceGPT<br>32.08<br>58.00<br>63.85<br>34.02<br>20.60<br>53.43<br>56.99<br>32.60<br>43.95 (62%)<br>LLM-Pruner<br>21.42<br>25.38<br>38.81<br>26.22<br>13.80<br>54.87<br>50.83<br>24.95<br>32.04 (45%)<br>ZipLM<br>48.55<br>78.54<br>80.55<br>55.98<br>31.60<br>66.79<br>78.37<br>62.73<br>62.89 (89%)<br>OSSCAR<br>48.29<br>78.62<br>81.44<br>54.69<br>32.80<br>68.23<br>77.58<br>60.38<br>62.75 (89%)<br>FLAP<br>37.54<br>68.90<br>67.34<br>43.98<br>26.40<br>60.65<br>72.30<br>54.40<br>53.94 (76%)<br>Týr-the-Pruner<br>**56.74**<br>**85.40**<br>**85.20**<br>**64.07**<br>**36.40**<br>**71.48**<br>**78.91**<br>**70.29**<br>**68.56 (97%)**|0%|N/A|60.58<br>87.29<br>85.29<br>66.50<br>37.00<br>70.04<br>79.64<br>78.72<br>70.63 (100%)|
|Llama-3.1-70B<br>0%<br>N/A<br>60.58<br>87.29<br>85.29<br>66.50<br>37.00<br>70.04<br>79.64<br>78.72<br>70.63 (100%)<br>50%<br>SliceGPT<br>32.08<br>58.00<br>63.85<br>34.02<br>20.60<br>53.43<br>56.99<br>32.60<br>43.95 (62%)<br>LLM-Pruner<br>21.42<br>25.38<br>38.81<br>26.22<br>13.80<br>54.87<br>50.83<br>24.95<br>32.04 (45%)<br>ZipLM<br>48.55<br>78.54<br>80.55<br>55.98<br>31.60<br>66.79<br>78.37<br>62.73<br>62.89 (89%)<br>OSSCAR<br>48.29<br>78.62<br>81.44<br>54.69<br>32.80<br>68.23<br>77.58<br>60.38<br>62.75 (89%)<br>FLAP<br>37.54<br>68.90<br>67.34<br>43.98<br>26.40<br>60.65<br>72.30<br>54.40<br>53.94 (76%)<br>Týr-the-Pruner<br>**56.74**<br>**85.40**<br>**85.20**<br>**64.07**<br>**36.40**<br>**71.48**<br>**78.91**<br>**70.29**<br>**68.56 (97%)**|50%|SliceGPT<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|32.08<br>58.00<br>63.85<br>34.02<br>20.60<br>53.43<br>56.99<br>32.60<br>43.95 (62%)<br>21.42<br>25.38<br>38.81<br>26.22<br>13.80<br>54.87<br>50.83<br>24.95<br>32.04 (45%)<br>48.55<br>78.54<br>80.55<br>55.98<br>31.60<br>66.79<br>78.37<br>62.73<br>62.89 (89%)<br>48.29<br>78.62<br>81.44<br>54.69<br>32.80<br>68.23<br>77.58<br>60.38<br>62.75 (89%)<br>37.54<br>68.90<br>67.34<br>43.98<br>26.40<br>60.65<br>72.30<br>54.40<br>53.94 (76%)<br>**56.74**<br>**85.40**<br>**85.20**<br>**64.07**<br>**36.40**<br>**71.48**<br>**78.91**<br>**70.29**<br>**68.56 (97%)**|



we iteratively prune and adjust weights by removing one attention head or 16 FFN neurons at a time.
The prune-and-search process consists of 4 iterations, where the sparsity interval at the _i_ - th iteration
is set to 12 _._ 5% _/_ 2 _[i][−]_ [1] . In each iteration, we explore 50 generations with 128 offspring candidates per
generation. The sparsity shifts of the attention or FFN layers are independent to ensure the consistency

- f the sparsity interval granularity. Candidate validation is performed using the distillation-inspired
metric with vocabulary logits. We follow [36] to enhance validation efficiency: the 128 offspring are
first validated on 2K tokens, and the top 16 are selected. These 16 survivors are then validated on
16K tokens, from which the top 4 are selected, and finally, the best one is validated and selected on
128K tokens. **To ensure a fair comparison**, we use the same FineWeb-Edu samples for calibration to
reproduce the baselines. The benchmark results of the baselines may outperform their reported results
due to the improved calibration sample size and data quality. All experiments for Týr-the-Pruner were
conducted on 4 AMD Instinct™MI250 (64GB) Accelerators, with models less than 13B parameters
running on a single accelerator.


**3.2** **Performance**


**Language comprehension and downstream task performance of post-pruned LLMs** . We applied
structural pruning to various large language models using Týr-the-Pruner at overall sparsity levels

- f 12.5%, 25%, 37.5%, and 50%. The performance was benchmarked against state-of-the-art
methods, including ShortGPT (layer pruning) [24], LaCO+ (ShortGPT with LaCO layer merging)

[50], SliceGPT (embedding dimension pruning) [2], Wanda-SP [37, 1], LLM-Pruner [23], ZipLM

[17], OSSCAR [26], and FLAP [1]. Table 1 summarizes the comparative results, highlighting postpruning performance in language comprehension and downstream tasks (cf. Section A.8 for detailed
results within each task).


Týr-the-Pruner demonstrates competitive performance across various sparsity ratios and LLMs. It
consistently achieves state-of-the-art results at low sparsity ratios ( _≤_ 25%). For instance, pruning
12.5% of Llama-3-8B’s parameters yields the lowest perplexity (7.39) and the highest average
downstream accuracy (62.37%), surpassing the previous advanced methods, LLM-Pruner and LaCO+,
by 8.0% and 2.6%. At higher sparsities ( _≥_ 37.5%), maintaining performance poses a significant
challenge for existing methods, with advanced techniques like OSSCAR often exhibiting perplexities
exceeding 100 and accuracies dropping below 40%. Týr-the-Pruner, by contrast, excels under these
conditions. For example, at 37.5% sparsity, the pruned Mistral-Nemo model achieves a perplexity of
11.47 and an accuracy of 55.63%, substantially outperforming ZipLM and FLAP.


**Scale up to massive language models** . Structural pruning of massive language models challenges
post-pruned performance and resource budgets. We incorporated a CPU offload policy into typical
baseline methods to ensure a fair comparison on 70B-scale models. Table 2 compares the post-pruning
performance of Llama-2-70B and Llama-3.1-70B at 50% sparsity.


Experimental results demonstrate Týr-the-Pruner’s strong scalability under high sparsity for massive
models. LLM-Pruner shows clear scaling limitations, maintaining only 48% accuracy when pruning


7




--- end of page=6 ---

Table 3: **Inference efficiency of post-pruned LLMs**
**with Týr-the-Pruner** . Benchmarks were conducted

- n a single AMD Instinct™MI250 accelerator using
PyTorch (HipBlas) for LLM inference, with input
and output sequence lengths set to 2048.


**Model** **Sparsity #Params** **TTFT** **Decode Throughput**



Table 4: **Ablation study on local pruning** .
Wikitext2 perplexity and 0-shot accuracy on
ARC-C, ARC-E, and BoolQ are reported.





Llama-3.1-8B


Mistral-Nemo



0% 8.0B 2.49 (1.00x) 12.27 (1.00x)
25% 6.1B 1.94 (1.28x) 14.13 (1.15x)
50% 4.3B 1.42 (1.75x) 16.97 (1.38x)


0% 14.3B 4.16 (1.00x) 6.68 (1.00x)
25% 11.0B 3.34 (1.25x) 7.55 (1.13x)
50% 7.8B 2.49 (1.67x) 8.93 (1.34x)




|Method Configuration|Wikitext2 ARC-C ARC-E BoolQ|
|---|---|
|FLAP<br>-|134.28<br>20.99<br>43.18<br>52.29|
|Local Pruning<br>Default<br>Wikitext2 Calibrate<br>C4 Calibrated<br>w/o progressive pru<br>w/o Hessian<br>w/o Gradient|**58.09**<br>24.06<br>**58.67**<br>**63.46**<br> d<br>49.00<br>20.05<br>54.84<br>61.71<br>73.07<br>21.42<br>57.58<br>62.17<br>  ning<br>63.48<br>23.38<br>56.65<br>62.17<br>109.88<br>22.53<br>51.68<br>46.48<br>67.31<br>**25.60**<br>57.83<br>62.17|
|Local Pruning<br>&<br>Build Supernet<br>Default<br>w/o Error Accum.<br>w/ Random Error A<br>w/ Uniform Error A|**66.38**<br>23.05<br>**58.46**<br>**62.35**<br>538.23<br>21.93<br>33.54<br>40.31<br>   ccum.<br>208.92<br>22.70<br>39.14<br>45.05<br>   ccum.<br>75.10<br>**23.72**<br>53.03<br>60.06|



Llama-2-70B. In contrast, Týr-the-Pruner achieves 97% accuracy maintenance when pruning Llama3.1-70B, outperforming alternative methods.


**Inference efficiency of post-pruned LLMs** . To evaluate the efficiency gains of post-pruned LLMs,
we constructed inference benchmarks summarized in Table 3. For Llama-3.1-8B, 50% sparsity
reduces time to first token (TTFT, in seconds) by 43% and boosts decode throughput (tokens/s) by
38%. These results highlight pruning as a key technique for inference optimization in large language
models. More detailed efficiency analysis can be found in Section A.4.


**3.3** **Ablation Study**



**Prune-to-supernet** . The effectiveness of local pruning and supernet construction depends on factors
such as calibration samples, the implementation of local pruning, and error accumulation. Table 4
presents ablation study evaluating these factors for pruning Llama-3.1-8B at 50% sparsity. Experimental results show that FineWeb-Edu is consistently preferred as a calibration source, emphasizing the
importance of selecting high-quality calibration samples. The presence of both first- and second-order

- ptimization information and progressive pruning significantly impacts accuracy, demonstrating
their necessity. Furthermore, the proposed expectation error accumulation approach outperforms
alternatives, showcasing its ability to make sparse structures mutually aware appropriately.


**Evolutionary search direction** . To assess the Table 5: **Ablation study on search direction** .
impact of search direction on final performance,

Wikitext2 perplexity and 0-shot accuracy on

we compare the effects of minimizing single
ARC-C, ARC-E, BoolQ are reported.

task losses versus our similarity-based metric
when pruning 50% of Llama-3.1-8B’s parame- **Search Direction** **Wikitext2** **ARC-C** **ARC-E** **BoolQ**
ters, as shown in Table 5. Experiments show that Wikitext2 Perplexity **17.22** 29.69 64.06 62.23
single-task search underperforms our metric, which Fineweb-Edu PerplexitySimilarity-based 31.6528.56 31.06 **32.51** 64.18 **65.87** 62.1763.12
achieves optimal accuracy by calculating the simi- Similarity-based Logits-only 30.89 31.83 65.36 **64.62**
larity across activations from the first, median, last,

|Search Direction|Wikitext2 ARC-C ARC-E BoolQ|
|---|---|
|Wikitext2 Perplexity<br>Fineweb-Edu Perplexity<br>Similarity-based<br>Similarity-based Logits-only|**17.22**<br>29.69<br>64.06<br>62.23<br>31.65<br>31.06<br>64.18<br>62.17<br>28.56<br>**32.51**<br>**65.87**<br>63.12<br>30.89<br>31.83<br>65.36<br>**64.62**|

and logits layers, requiring 96 GB for hidden activation checkpointing. Due to this overhead, the
logits-only metric was favored, maintaining strong performance with reduced resource demands.


**Effect comparison: Týr-the-Pruner vs.** **fine-**
**grained search-only strategy** . Figure 3 demonstrates the advantages of Týr-the-Pruner over the
search-only strategy in efficacy and efficiency in
identifying the optimal 50% sparsity distribution **Accuracy = 43.58**

- n Llama-3.1-8B. In which the search-only strategy
uses a fine-grained 3.125% sparsity interval. Exper- **Accuracy = 47.79**
imental results show that Týr-the-Pruner achieves a
similar convergence trend as the search-only strat
|Col1|Col2|
|---|---|
|**Accuracy = 47.79**<br>**Accuracy = 43.58**|**Accuracy = 47.79**<br>**Accuracy = 43.58**|


### Figure 3

Caption: Týr-the-Pruner has faster conver

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
egy but with faster convergence, fewer generations,

gence, fewer exploration generations, shorter

and reduced search time. Additionally, the final

search time, and better search outcomes com
post-pruned model discovered by Týr-the-Pruner

pared to the fine-grained search-only approach.

- utperforms the search-only strategy, with an average accuracy of 47.79 compared to 43.58. Our
evolutionary search maintains time efficiency, with a single generation requiring only 190 seconds.



Table 5: **Ablation study on search direction** .
Wikitext2 perplexity and 0-shot accuracy on
ARC-C, ARC-E, BoolQ are reported.









### Figure 3

Caption: Týr-the-Pruner has faster convergence, fewer exploration generations, shorter

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
search time, and better search outcomes compared to the fine-grained search-only approach.



8




--- end of page=7 ---

Table 6: Týr-the-Pruner progressively refines and optimizes the sparsity distribution in iterations,
steadily enhancing performance.

|Method|Wikitext2|ARC-C ARC-E BoolQ HellaSwag OBQA RTE WinoGrande MMLU|AVG|
|---|---|---|---|
|w/o search|66.38|23.55<br>58.46<br>62.35<br>32.51<br>16.60<br>51.26<br>52.88<br>28.34|40.74|
|search-only|**27.96**|25.34<br>59.30<br>64.71<br>36.52<br>22.20<br>55.23<br>56.20<br>29.17|43.58|
|Týr-the-Pruner I1<br>Týr-the-Pruner I2<br>Týr-the-Pruner I3<br>Týr-the-Pruner I4<br>Týr-the-Pruner I5|28.92<br>31.80<br>29.75<br>30.89<br>29.56|26.45<br>56.19<br>62.17<br>37.05<br>22.20<br>50.54<br>56.75<br>29.29<br>29.27<br>62.54<br>63.51<br>38.18<br>23.80<br>50.54<br>56.85<br>30.23<br>29.86<br>63.09<br>64.62<br>39.28<br>25.00<br>51.62<br>59.51<br>31.62<br>31.83<br>65.36<br>66.64<br>**39.99**<br>24.80<br>58.12<br>**61.80**<br>**33.76**<br>**31.87**<br>**65.57**<br>**66.69**<br>39.08<br>**26.40**<br>**58.84**<br>60.89<br>33.67|42.58<br>44.37<br>45.58<br>47.79<br>**47.88**|



To further examine how Týr-the-Pruner refines model performance across successive iterations and to
highlight its advantage over the search-only strategy, we report the 50% post-pruned results of Llama3.1-8B on multiple tasks: Wikitext2 perplexity ( _↓_ ), 0-shot accuracy ( _↑_ ) on Arc, BoolQ, HellaSwag,
OBQA, RTE, and WinoGrande, and 5-shot accuracy ( _↑_ ) on MMLU, as summarized in Table 6.


Experimental results underscore the superiority of Týr-the-Pruner compared to the search-only strategy. Under isotropic pruning, the w/o search baseline consistently delivers suboptimal performance
across all tasks. By contrast, Týr-the-Pruner begins to surpass the search-only approach as early as
the second iteration (I2), highlighting the advantage of progressively refining the sparsity distribution.
The search-only method, constrained by the vast search space, suffers from extended search times and
limited effectiveness. Týr-the-Pruner achieves an ideal result at the fourth iteration (I4). Although
performance continues to improve at the fifth iteration (I5), the gains diminish, making four iterations
the default configuration.


**3.4** **Compatibility with Quantization and Unstructured Sparsity**


Týr-the-Pruner is compatible with further compression techniques. Table 7 reports results of applying
quantization and unstructured sparsity to the 50% pruned Llama-3.1-8B model produced by Týrthe-Pruner. The results show that quantization preserves over 99% of the pruned model’s accuracy,
while unstructured sparsity at 50% and 2:4 granularity achieves performance consistent with prior
work [9, 25]. These findings indicate that Týr-the-Pruner provides a strong foundation for multi-stage
compression pipelines, retaining both high accuracy and robustness to further compression.


Table 7: Quantization and unstructured sparsity applied to the 50% pruned Llama-3.1-8B with
Týr-the-Pruner. Reported Wikitext2 perplexity and accuracy (%) across benchmarks.

|Method|Wikitext2|Arc-C Arc-E BoolQ HellaSwag OBQA RTE WinoGrande MMLU|AVG|
|---|---|---|---|
|Týr-the-Pruner @ FP16<br>AWQ [21] @ W4A16<br>SmoothQuant [49] @ W8A8<br>RTN @ FP8E4M3<br>SparseGPT [9] @ 50%<br>ALPS [25] @ M4N2|30.89<br>34.79<br>31.31<br>31.05<br>47.05<br>70.22|31.83<br>65.36<br>66.64<br>39.99<br>24.80<br>58.12<br>61.80<br>33.76<br>31.06<br>64.65<br>66.09<br>39.34<br>24.60<br>60.77<br>60.77<br>31.43<br>31.23<br>64.94<br>65.50<br>40.18<br>25.80<br>58.84<br>61.80<br>32.45<br>31.31<br>65.28<br>66.45<br>40.17<br>24.60<br>58.84<br>61.48<br>32.28<br>27.99<br>59.39<br>65.32<br>37.38<br>23.60<br>53.07<br>61.48<br>28.32<br>23.04<br>53.62<br>63.43<br>34.22<br>21.40<br>52.71<br>58.17<br>27.34|47.79 (100%)<br>47.34 (%99.1)<br>47.59 (%99.6)<br>47.55 (%99.5)<br>44.57 (%93.3)<br>41.74 (%87.3)|



**3.5** **Memory/Storage Efficiency Analysis of Týr-the-Pruner**



Týr-the-Pruner employs a supernet search tech
Table 8: Resource requirements of Týr-the-Pruner.

nique, where storing a large-scale supernet in
memory is costly. To address memory concerns, **Model Size** **Submodules** **HBM Usage** **Disk Storage Usage**
we optimize our approach by storing pruned sub- 7-8B 576 14-16GB 39.6GB
structures on disk instead of in high-bandwidth 13B70B 1440720 140GB26GB 414.7GB66.6GB
memory (HBM). An integer Python list is used
to track the currently selected substructures, en
|ModelSize|Submodules|HBMUsage|DiskStorageUsage|
|---|---|---|---|
|7-8B<br>13B<br>70B|576<br>720<br>1440|14-16GB<br>26GB<br>140GB|39.6GB<br>66.6GB<br>414.7GB|

suring that only one entire LLM is loaded into HBM at any given time (e.g., the 7B model uses
approximately 14GB, and the 13B model uses around 26GB). Table 8 provides detailed data on HBM
and disk storage occupancy. Furthermore, since there is no dependency between iterations (iterative
prune-and-search phase), the storage from previous iterations can be cleaned, further minimizing disk
usage. Due to the low cost of disk storage, these memory and storage demands are highly acceptable.



Table 8: Resource requirements of Týr-the-Pruner.







9




--- end of page=8 ---

**4** **Related Work**


**Pruning techniques for compressing large language models** . The growing complexity of
Transformer-based language models, now reaching hundreds of billions of parameters, has intensified
the necessity for effective pruning strategies. Pruning methods are generally divided into unstructural
and structural approaches. Unstructural pruning [9, 37] achieves high accuracy by selectively zeroing
individual elements in the weight. However, it often requires specialized hardware, such as 2:4
sparse tensor cores [54], for end-to-end acceleration. Structural pruning enables hardware-agnostic
acceleration by removing entire weight groups, but it may result in a pronounced loss of accuracy.


Structural pruning of LLMs can be approached as local optimization, alleviating memory constraints
from loading the full model. ZipLM [17] accelerates inference by leveraging the Optimal Brain
Surgeon (OBS) [12] theory, pruning weights to minimize the impact on the Hessian matrix and
adjusting the remaining weights to reduce layerwise loss. Building on ZipLM, OSSCAR [26]
introduces a permutation search between pruned and remaining weights within each layer, further
reducing pruning-induced loss. Some approaches apply global optimization strategies to prune LLMs,

- vercoming local constraints, enabling customized sparsity distributions, and potentially finding

- ptimal solutions. Fisher information was introduced as a saliency metric to guide structure pruning
via global dynamic programming [18]. LLM-Pruner [23] defines broad substructure dependency
groups and then evaluates their saliency to guide pruning. FLAP [1] uses a global metric that
considers both weights and activations for sparsity allocation, followed by layerwise pruning and
bias adjustments to mitigate pruning losses.


Additionally, there is growing interest in embedding dimension [2] and depth [24, 50] pruning
techniques for LLMs. Some training-aware structural pruning methods [52, 20, 29] are also gaining
attention, as they further enhance pruning effectiveness by considering training dynamics.


**Neural architecture search (NAS) for LLM compression** . Several studies have applied NAS to
compress LLMs, seeking architectures that reduce inference costs while maintaining accuracy. multi
- bjective NAS has been employed to explore various search space definitions, identifying compressed
LLM architectures that enhance efficiency and accuracy when fine-tuned on specific downstream
tasks [16]. LLaMAFlex [4] fine-tunes LLMs into supernets with a Gumbel softmax-based trainable
subnet router, realized a “rain once, deploy many" model compression. EvoPress [36] proves that
evolutionary search can determine suitable layerwise compression configurations and extends this
method to support mixed-precision quantization and non-isotropic unstructural sparsity.


This paper presents a novel structural pruning framework, Týr-the-Pruner, for large language models.
Unlike conventional methods, this framework searches for the optimal sparsity distribution within a
supernet. Through enhanced supernet construction and an iterative prune-and-search technique, it
achieves end-to-end global pruning optimization with strong efficiency and efficacy, setting a new
benchmark for post-pruning accuracy maintenance.


**5** **Limitations**


Týr-the-Pruner achieves state-of-the-art structural pruning outcomes by constructing reliable supernets
and employing an iterative prune-and-search process. We have significantly reduced the search space
and the number of generations explored. However, the search time cost remains non-negligible.
Fair time costs in model compression are often considered acceptable, as the goal is to achieve a
sufficiently optimized pruned model. However, we will continue to optimize it in future work.


**6** **Conclusion**


This paper introduces Týr-the-Pruner, an end-to-end global structural pruning framework for large
language models. By constructing a supernet through local pruning across various sparsity ratios
and using evolutionary search to identify the optimal subnet, our framework achieves the optimal
sparsity distribution under a target overall sparsity ratio. We propose an effective local pruning
and an expectation error accumulation approach to enhance supernet construction. Additionally, an
iterative prune-and-search strategy with coarse-to-fine sparsity granularity ensures rapid convergence.
Extensive experiments show that Týr-the-Pruner outperforms state-of-the-art methods, achieving 50%
parameter pruning while retaining 97% accuracy on Llama-3.1-70B.


10




--- end of page=9 ---

## References


[1] Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Fluctuation-based adaptive structured
pruning for large language models. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan,
editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference

   - n Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational
Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 10865–
10873. AAAI Press, 2024.


[2] Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari Do Nascimento, Torsten Hoefler, and James
Hensman. Slicegpt: Compress large language models by deleting rows and columns. In The Twelfth
International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024.
OpenReview.net, 2024.


[3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, MariaFlorina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33:

                                                                Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6 12,
2020, virtual, 2020.


[4] Ruisi Cai, Saurav Muralidharan, Hongxu Yin, Zhangyang Wang, Jan Kautz, and Pavlo Molchanov.
Llamaflex: Many-in-one llms via generalized pruning and weight sharing. In The Thirteenth International
Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net,
2025.


[5] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Jill Burstein,
Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT
2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 2924–2936.
Association for Computational Linguistics, 2019.


[6] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
arXiv:1803.05457v1, 2018.


[7] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang,
Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien
Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern,
Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe
Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel
Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego GarciaOlano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan,
Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis
Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar,
Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov,
Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der
Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie
Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe,
Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin
Stone, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024.


[8] Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate post-training
quantization and pruning. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and
A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural
Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December
9, 2022, 2022.


[9] Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot.
In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan

                                                    Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23 29 July 2023, Honolulu,
Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 10323–10337. PMLR,
2023.


11




--- end of page=10 ---

[10] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: accurate post-training quantization for generative pre-trained transformers. CoRR, abs/2210.17323, 2022.


[11] Shangqian Gao, Chi-Heng Lin, Ting Hua, Zheng Tang, Yilin Shen, Hongxia Jin, and Yen-Chang Hsu.
DISP-LLM: dimension-independent structural pruning for large language models. In Amir Globersons,
Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information
Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.


[12] Babak Hassibi and David G. Stork. Second order derivatives for network pruning: Optimal brain surgeon.
In Stephen Jose Hanson, Jack D. Cowan, and C. Lee Giles, editors, Advances in Neural Information
Processing Systems 5, [NIPS Conference, Denver, Colorado, USA, November 30 - December 3, 1992],
pages 164–171. Morgan Kaufmann, 1992.


[13] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob
Steinhardt. Measuring massive multitask language understanding. Proceedings of the International
Conference on Learning Representations (ICLR), 2021.


[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference


   - n Learning Representations, ICLR 2022, Virtual Event, April 25 29, 2022. OpenReview.net, 2022.


[15] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego
de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and
William El Sayed. Mistral 7b. CoRR, abs/2310.06825, 2023.


[16] Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, and Cedric Archambeau. Structural
pruning of large language models via neural architecture search. In AutoML Conference 2023 (Workshop).


[17] Eldar Kurtic, Elias Frantar, and Dan Alistarh. Ziplm: Inference-aware structured pruning of language
models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey
Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural

                                                            Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 16, 2023,
2023.


[18] Woosuk Kwon, Sehoon Kim, Michael W. Mahoney, Joseph Hassoun, Kurt Keutzer, and Amir Gholami.
A fast post-training pruning framework for transformers. In Sanmi Koyejo, S. Mohamed, A. Agarwal,
Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35:
Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA,
USA, November 28 - December 9, 2022, 2022.


[19] Qi Le, Enmao Diao, Ziyan Wang, Xinran Wang, Jie Ding, Li Yang, and Ali Anwar. Probe pruning:
Accelerating llms through dynamic pruning via model-probing. arXiv preprint arXiv:2502.15618, 2025.


[20] Shengrui Li, Xueting Han, and Jing Bai. Nuteprune: Efficient progressive pruning with numerous teachers
for large language models. CoRR, abs/2402.09773, 2024.


[21] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao,
Xingyu Dang, Chuang Gan, and Song Han. AWQ: activation-aware weight quantization for on-device
LLM compression and acceleration. In Phillip B. Gibbons, Gennady Pekhimenko, and Christopher De Sa,
editors, Proceedings of the Seventh Annual Conference on Machine Learning and Systems, MLSys 2024,

                       Santa Clara, CA, USA, May 13 16, 2024. mlsys.org, 2024.


[22] Yuqiao Liu, Yanan Sun, Bing Xue, Mengjie Zhang, Gary G Yen, and Kay Chen Tan. A survey on
evolutionary neural architecture search. IEEE transactions on neural networks and learning systems,
34(2):550–570, 2021.


[23] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language
models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey
Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural

                                                            Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 16, 2023,
2023.


[24] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and
Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect. CoRR,
abs/2403.03853, 2024.


12




--- end of page=11 ---

[25] Xiang Meng, Kayhan Behdin, Haoyue Wang, and Rahul Mazumder. ALPS: improved optimization for
highly sparse one-shot pruning for large language models. In Amir Globersons, Lester Mackey, Danielle
Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural
Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024,
NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.


[26] Xiang Meng, Shibal Ibrahim, Kayhan Behdin, Hussein Hazimeh, Natalia Ponomareva, and Rahul
Mazumder. OSSCAR: one-shot structured pruning in vision and language models with combinatorial

   - ptimization. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria,

        July 21 27, 2024. OpenReview.net, 2024.


[27] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models.
In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26,
2017, Conference Track Proceedings. OpenReview.net, 2017.


[28] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity?
a new dataset for open book question answering. In EMNLP, 2018.


[29] Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary,
Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact language models via
pruning and knowledge distillation. CoRR, abs/2407.14679, 2024.


[30] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z.
Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang,
Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library.
In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and
Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference


   - n Neural Information Processing Systems 2019, NeurIPS 2019, December 8 14, 2019, Vancouver, BC,
Canada, pages 8024–8035, 2019.


[31] Guilherme Penedo, Hynek Kydlícek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel,
Leandro von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at
scale. CoRR, abs/2406.17557, 2024.


[32] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4.

arXiv preprint arXiv:2304.03277, 2023.


[33] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial
winograd schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019.


[34] Noam Shazeer. GLU variants improve transformer. CoRR, abs/2002.05202, 2020.


[35] Xuan Shen, Pu Zhao, Yifan Gong, Zhenglun Kong, Zheng Zhan, Yushu Wu, Ming Lin, Chao Wu, Xue Lin,
and Yanzhi Wang. Search for efficient large language models. In Amir Globersons, Lester Mackey, Danielle
Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural
Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024,
NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.


[36] Oliver Sieberling, Denis Kuznedelev, Eldar Kurtic, and Dan Alistarh. EvoPress: Accurate dynamic
model compression via evolutionary search. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon LacosteJulien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the
42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning
Research, pages 55556–55590. PMLR, 13–19 Jul 2025.


[37] Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. A simple and effective pruning approach for large
language models. In The Twelfth International Conference on Learning Representations, ICLR 2024,

                  Vienna, Austria, May 7 11, 2024. OpenReview.net, 2024.


[38] Shengkun Tang, Oliver Sieberling, Eldar Kurtic, Zhiqiang Shen, and Dan Alistarh. Darwinlm: Evolutionary
structured pruning of large language models. arXiv preprint arXiv:2502.07780, 2025.


[39] Yehui Tang, Yunhe Wang, Yixing Xu, Yiping Deng, Chao Xu, Dacheng Tao, and Chang Xu. Manifold
regularized dynamic network pruning. In IEEE Conference on Computer Vision and Pattern Recognition,

                     CVPR 2021, virtual, June 19 25, 2021, pages 5018–5028. Computer Vision Foundation / IEEE, 2021.


13




--- end of page=12 ---

[40] Yehui Tang, Yunhe Wang, Yixing Xu, Dacheng Tao, Chunjing Xu, Chao Xu, and Chang Xu. SCOP: scientific control for reliable neural network pruning. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell,
Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems

                                                                33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6 12,
2020, virtual, 2020.


[41] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller,
Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan
Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh
Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao,
Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy
Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan
Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin
Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien
Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned
chat models. CoRR, abs/2307.09288, 2023.


[42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy
Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances
in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing

                    Systems 2017, December 4 9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017.


[43] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE:
A multi-task benchmark and analysis platform for natural language understanding. In 7th International
Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.


[44] Boyao Wang, Rui Pan, Shizhe Diao, Xingyuan Pan, Jipeng Zhang, Renjie Pi, and Tong Zhang. Adaptpruner: Adaptive structural pruning for efficient small language model training. CoRR, abs/2502.03460,
2025.


[45] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. SVD-LLM: truncation-aware singular value
decomposition for large language model compression. CoRR, abs/2403.07378, 2024.


[46] Yuxin Wang, Minghua Ma, Zekun Wang, Jingchang Chen, Liping Shan, Qing Yang, Dongliang Xu,
Ming Liu, and Bing Qin. CFSP: an efficient structured pruning framework for llms with coarse-tofine activation information. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa,
Barbara Di Eugenio, and Steven Schockaert, editors, Proceedings of the 31st International Conference


   - n Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19 24, 2025, pages 9311–9328.
Association for Computational Linguistics, 2025.


[47] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric
Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin
Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Qun
Liu and David Schlangen, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural

                                         -                                          Language Processing: System Demonstrations, EMNLP 2020 Demos, Online, November 16 20, 2020,
pages 38–45. Association for Computational Linguistics, 2020.


[48] Shangyu Wu, Hongchao Du, Ying Xiong, Shuai Chen, Tei-Wei Kuo, Nan Guan, and Chun Jason Xue.
Evop: Robust LLM inference via evolutionary pruning. CoRR, abs/2502.14910, 2025.


[49] Guangxuan Xiao, Ji Lin, Mickaël Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant:
Accurate and efficient post-training quantization for large language models. In Andreas Krause, Emma
Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International
Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of
Proceedings of Machine Learning Research, pages 38087–38099. PMLR, 2023.


[50] Yifei Yang, Zouying Cao, and Hai Zhao. Laco: Large language model pruning via layer collapse. In Yaser
Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational

                                          Linguistics: EMNLP 2024, Miami, Florida, USA, November 12 16, 2024, pages 6401–6417. Association
for Computational Linguistics, 2024.


14




--- end of page=13 ---

[51] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really
finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational
Linguistics, 2019.


[52] Mingyang Zhang, Hao Chen, Chunhua Shen, Zhen Yang, Linlin Ou, Xinyi Yu, and Bohan Zhuang.
Loraprune: Structured pruning meets low-rank parameter-efficient fine-tuning. In Lun-Wei Ku, Andre
Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL

                                       2024, Bangkok, Thailand and virtual meeting, August 11 16, 2024, pages 3013–3026. Association for
Computational Linguistics, 2024.


[53] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,
Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of
large language models. CoRR, abs/2303.18223, 2023.


[54] Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, and Hongsheng Li.
Learning N: M fine-grained structured sparse neural networks from scratch. In 9th International Conference


   - n Learning Representations, ICLR 2021, Virtual Event, Austria, May 3 7, 2021. OpenReview.net, 2021.


15




--- end of page=14 ---

**NeurIPS Paper Checklist**


1. **Claims**


Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction sections offer a comprehensive discussion of the
manuscript’s context, intuition, and ambitions, as well as its contributions.

Guidelines:


      - The answer NA means that the abstract and introduction do not include the claims
made in the paper.

      - The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.

      - The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.

       - It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.

2. **Limitations**


Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of this work are discussed at the end of the manuscript.

Guidelines:


      - The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.

      - The authors are encouraged to create a separate "Limitations" section in their paper.

      - The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.

      - The authors should reflect on the scope of the claims made, e.g., if the approach was

       - nly tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.

      - The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.

      - The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.

      - If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.

      - While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. **Theory assumptions and proofs**


Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]


16




--- end of page=15 ---

Justification: This manuscript uses existing theories as a basis, while providing complete
and correct derivations.


Guidelines:


   - The answer NA means that the paper does not include theoretical results.

   - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

   - All assumptions should be clearly stated or referenced in the statement of any theorems.

   - The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.

   - Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.

   - Theorems and Lemmas that the proof relies upon should be properly referenced.


4. **Experimental result reproducibility**


Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions

  - f the paper (regardless of whether the code and data are provided or not)?


Answer: [Yes]


Justification: The pipeline of the methods and the details of experiments are presented with
corresponding reproducible credentials.


Guidelines:


   - The answer NA means that the paper does not include experiments.

   - If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.

   - If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.

   - Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often

    - ne good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case

    - f a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.

   - While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.


5. **Open access to data and code**


17




--- end of page=16 ---

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental
material?


Answer: [No]


Justification: At this stage, the experimental setup and details are sufficient to guarantee
reproducibility. Further materials will be made available in future updates.


Guidelines:


   - The answer NA means that paper does not include experiments requiring code.

   - Please see the NeurIPS code and data submission guidelines ( `[https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy)`
`[public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)` ) for more details.

   - While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).

   - The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines ( `[https:](https://nips.cc/public/guides/CodeSubmissionPolicy)`
`[//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)` ) for more details.

   - The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.

   - The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.

   - At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).

   - Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.


6. **Experimental setting/details**


Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?


Answer: [Yes]


Justification: The pipeline of the methods and the details of experiments are presented with
corresponding reproducible credentials.


Guidelines:


   - The answer NA means that the paper does not include experiments.

   - The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.

   - The full details can be provided either with the code, in appendix, or as supplemental
material.


7. **Experiment statistical significance**


Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?


Answer: [Yes]


Justification: The results contain the standard deviation of the results over several random

runs.


Guidelines:


   - The answer NA means that the paper does not include experiments.

   - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.


18




--- end of page=17 ---

    - The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

    - The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)

    - The assumptions made should be given (e.g., Normally distributed errors).

    - It should be clear whether the error bar is the standard deviation or the standard error

     - f the mean.

    - It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis

     - f Normality of errors is not verified.

    - For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).

    - If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.


8. **Experiments compute resources**


Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?


Answer: [Yes]


Justification: The paper provided sufficient information on the computer resources needed
to reproduce the experiments.


Guidelines:


    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster,

     - r cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).


9. **Code of ethics**


Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics `[https://neurips.cc/public/EthicsGuidelines](https://neurips.cc/public/EthicsGuidelines)` ?


Answer: [Yes]


Justification: The research conducted in the paper conforms with the NeurIPS Code of
Ethics.


Guidelines:


    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).


10. **Broader impacts**


Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?


Answer: [NA]


Justification: There is no societal impact of the work performed: This paper discusses the
lightweighting of LLMs that can reduce the consumption of resources without obvious
broader impacts.


19




--- end of page=18 ---

Guidelines:


    - The answer NA means that there is no societal impact of the work performed.

    - If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

    - Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

    - The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.

    - The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.

    - If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).

11. **Safeguards**


Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:


    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.

12. **Licenses for existing assets**


Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?

Answer: [Yes]


Justification: The original owners of assets, including data and models used in the paper, are
properly credited, and the licenses and terms of use are explicitly mentioned and properly
respected.

Guidelines:


    - The answer NA means that the paper does not use existing assets.

    - The authors should cite the original paper that produced the code package or dataset.

    - The authors should state which version of the asset is used and, if possible, include a
URL.


20




--- end of page=19 ---

    - The name of the license (e.g., CC-BY 4.0) should be included for each asset.

    - For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

    - If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, `paperswithcode.com/datasets`
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.

    - For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.

    - If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.


13. **New assets**


Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?


Answer: [NA]


Justification: The paper does not release new assets.


Guidelines:


    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose
asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.


14. **Crowdsourcing and research with human subjects**


Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?


Answer: [NA]


Justification: The paper does not involve crowdsourcing nor research with human subjects.


Guidelines:


    - The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be
included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation,

     - r other labor should be paid at least the minimum wage in the country of the data
collector.


15. **Institutional review board (IRB) approvals or equivalent for research with human**
**subjects**

Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?


Answer: [NA]


Justification: The paper does not involve crowdsourcing nor research with human subjects.


Guidelines:


    - The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.


21




--- end of page=20 ---

    - Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.

    - For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

16. **Declaration of LLM usage**


Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used

  - nly for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any
important, original, or non-standard components.

Guidelines:


    - The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.

    - Please refer to our LLM policy ( `[https://neurips.cc/Conferences/2025/LLM](https://neurips.cc/Conferences/2025/LLM)` )
for what should or should not be described.


22




--- end of page=21 ---

**A** **Appendix**


**A.1** **Theoretical Foundations of Local Pruning**


**Redundant channel identification** . We consider first- and second-order terms to minimize Equation (3). For the first-order term, we identify the to-prune channel _p_ by argmin **W** _p,_ : - �� **G** _p,_ : **W** _p,⊤_ :���,
which identifies the weights with the minimal contribution in the gradient direction [23]. For the
second-order term, we employ the Optimal Brain Surgeon (OBS) method [12], which optimizes
argmin **W** _p,_ : - 2[ _∥_ **HW** _[−]_ _p,_ [1] :] _∥p,p_ [2] 2 - by considering the inverse of the diagonal elements of the Hessian matrix.

This method measures each channel’s contribution to the curvature of the loss function.


The identification metric for redundant channels is derived from a manual design that takes into
account both first- and second-order optimization information, distinguishing it from previous work.
Table 4 demonstrates the validity of our metric by ablation.


**Weight adjustment** . We minimize Equation (3) by applying the Lagrange multiplier method to
impose constraints on the _p_ - th channel should be pruned ( _δ_ **W** _p,_ : = **W** _p,_ :):


_L_ ( _δ_ **W** _,_ _**λ**_ ) = **G** _[⊤]_ _δ_ **W** + [1] (7)

2 _[δ]_ **[W]** _[⊤]_ **[H]** _[δ]_ **[W]** [ +] _**[ λ]**_ _[⊤]_ [(] _[δ]_ **[W]** _[p,]_ [:] _[ −]_ **[W]** _[p,]_ [:][)] _[ .]_


Under the constraints, the resulting loss function _L_ ( _δ_ **W** _,_ _**λ**_ ) will be differentiated with respect to _δ_ **W**
and _**λ**_ to find the minimum value:







_∂L_ ( _∂δδ_ **WW** _,_ _**λ**_ ) = **G** + **H** _δ_ **W** + **E** _p_ _**λ**_ _[⊤]_ = **0** _,_

_∂L_ ( _δ∂_ **W** _**λ**_ _,_ _**λ**_ ) = _δ_ **W** _p,_ : _−_ **W** _p,_ : = **0** _._



(8)



For the **G** + **H** _δ_ **W** + **E** _p_ _**λ**_ _[⊤]_ term, we use _p_ and _∼_ _p_ to denote channels to prune and channels to
remain. Corresponding variables can be expanded in this way:



**G** _p,_ :

- **G** _∼p,_ :



+ **H** _p,p_ **H** _p,∼p_ = 0

- - **H** _∼p,p_ = 0 **H** _∼p,∼p_



_δ_ **W** _p,_ :

- � _δ_ **W** _∼p,_ :



_**λ**_ _[⊤]_
+

**0**

-


= **0** _,_ (9)




where the elements of the Hessian matrix corresponding to the pruned positions _p_ can be set to zero
(when a channel of the weights is pruned, the same position of the Hessian/invHessian matrix are
pruned correspondingly [8]). Overall, the solution is _δ_ **W** _∼p,_ : = _−_ **H** _[−]_ _∼_ [1] _p,∼p_ **[G]** _[∼][p,]_ [:][.]


**Fast update of inverse Hessian matrix** . When the _p_ - th channel is pruned, the inverse Hessian
matrix **H** _[−]_ [1] must be updated to account for the removal of the corresponding channel _p_ in **W** . This
update can be efficiently derived by leveraging the properties of partitioned matrices and applying the
Sherman-Morrison-Woodbury formula. The main idea is that the pruning of the _p_ - th channel results
in a rank-1 update to **H** _[−]_ [1], which is mathematically represented as:


1
**H** _[−]_ [1] _←_ **H** _[−]_ [1] _−_ [ **H** _[−]_ [1] ] _pp_ **H** _[−]_ : _,p_ [1] **[H]** _[−]_ _p,_ [1] : _[.]_ (10)


By updating the inverse Hessian with a rank-1 adjustment, the influence of the _p_ - th channel is properly
removed through the outer product of the corresponding column and row vectors, using the reciprocal

- f the _p_ - th diagonal element. The updated **H** _[−]_ [1] ensures consistency for the remaining channels,
enabling efficient and scalable pruning operations. This method has a time complexity of _O_ ( _d_ [2] in [)][,]
avoiding full recomputation of the inverse and ensuring computational efficiency.


23




--- end of page=22 ---

**A.2** **Algorithms for Týr-the-Pruner**


**Algorithm 1** Function local_pruning


1: **Inputs:** to-prune weight **W**,
input activations **X**,
sparsity _S_,
pruning granularity (pruning times) _K_
2: **Mask** _←_ - nes_like( **W** )
3: **H** _←_ **X** _[⊤]_ **X**

4: **G** _←_ **HW**

5: **for** _k ←_ 1 **to** _K_ **do**



**Algorithm 2** Function prune_to_supernet


1: **Inputs:** LLM weights _{_ **W** 1 _,_ **W** 2 _, ...,_ **W** _L}_,
sparsity ratios
_{S_ 1 _,_ 1 _, ..., S_ 1 _,E, ..., SL,E}_,
input activations for first weight **X**,
pruning granularity (pruning times) _K_
2: **for** _ℓ_ _←_ 1 **to** _L_ **do**
3: **X** _list _←_ []
4: **for** - _e ←_ 1 **to** _E_ **do**
5: **W** _ℓ,e ←_ local_pruning( **W** _ℓ,_ **X** _, Sℓ,e, K_ )
6: store( **W** [�] _ℓ,e_ )
7: **X** _list _._ append( **X** _·_ **W** [�] _ℓ,e_ )
8: **end for**
9: **X** _←_ [�] _[E]_ _e_ =1 ~~�~~ _Ee_ =11 _−_ [(1] _S_ _[−]_ _ℓ,e_ _[S][ℓ,e]_ [)] **[X]** [_list][[] _[e]_ []]
10: **end for**
11: **Return** _{_ **W** [�] _ℓ,e}_ _[L,E]_ _ℓ_ =1 _,e_ =1







6: _p ←_ argmin _p_




- �� **G** _p,_ : **W** _p,⊤_ :�� + 2 _∥_ [ **HW** _[−]_ _p,_ [1] :] _∥p,p_ [2] 2



7: **Mask** _p ←_ 0
8: **W** _∼p,_ : _←_ **W** _∼p,_ : + **H** _[−]_ _∼_ [1] _p,∼p_ **[G]** _∼p,_ :
9: **H** _[−]_ [1] _←_ **H** _[−]_ [1] _−_ [ **[H]** _[−]_ 1 [1] ] _p,p_ **H** _[−]_ : _,p_ [1] **[H]** _[−]_ _p,_ [1] :

10: **end for**

11: **Return Mask** _⊙_ **W**


**Algorithm 3** Function evolutionary_search



1: **Inputs:** sparse structures W = _{_ **W** 1 _,_ 1 _, ...,_ **W** 1 _,E, ...,_ **W** _L,E}_,
sparsity ratios _{Sℓ}ℓ_ _[L]_ =1 [, sparsity interval] _[ S][g]_

2: **procedure** makeCandidates( _numCanidates_,W, _{Sℓ}ℓ_ _[L]_ =1 [,] _[ S][g]_ [)]
3: _Candidates ←_ []
4: **for** _i ←_ 1 **to** _numCanidates_ **do**
5: _Candidates._ append(randSparsityShift(W _, {Sℓ}ℓ_ _[L]_ =1 _[, S][g][,]_ [ randChoice(] _[L]_ [)] _[,]_ [ randChoice(] _[L]_ [)))]
6: **end for**
7: **end procedure: return** _Candidates_
8: _{S_ [�] _ℓ}ℓ_ _[L]_ =1 _[←{][S][ℓ][}][L]_ _ℓ_ =1
9: **for** _g ←_ 1 **to** _numGenerations_ **do**
10: _Offsprings ←_ makeCandidates( _numCanidates,_ W _, {S_ [�] _ℓ}ℓ_ _[L]_ =1 _[, S][g]_ [)]
11: _{S_ [�] _ℓ}ℓ_ _[L]_ =1 _[←]_ [checkSparsity(argminSearchMetric(] _[Offsprings]_ [))]
12: **Return** _{S_ [�] _ℓ}ℓ_ _[L]_ =1


**Algorithm 4** Function Týr-the-Pruner


1: **Inputs:** LLM weights _{_ **W** 1 _,_ **W** 2 _, ...,_ **W** _L}_, input activations for first weight **X**,
pruning granularity (pruning times) _K_, overall sparsity _S_, sparsity interval _S_ _[g]_,
num sparse structures _E_, iterations _T_
2: **procedure** generateSparsities( _L_, _E_, _{Sℓ}ℓ_ _[L]_ =1 [,] _[ S][g]_ [)]
3: _Sparsities_ = _{}_
4: **for** _ℓ_ _←_ 0 **to** range( _L_ ) **do**
5: **for** _e ←_ 0 **to** range( _E_ ) **do**
6: _Sparsities._ append( _Sℓ_ _−_ (( _e −_ 1) _×_ 0 _._ 5) _× S_ _[g]_ + _i × S_ _[g]_ )
7: **end for**

8: **end for**
9: **end procedure: return** _Sparsities_
10: _{S_ [�] _ℓ}ℓ_ _[L]_ =1 _[←{][S][}][L]_
11: **for** _t ←_ 1 **to** _T_ **do**
12: _Sparsities ←_ generateSparsities( _L, E, {S_ [�] _ℓ}ℓ_ _[L]_ =1 _[, S][g]_ [)]
13: _{_ **W** [�] _ℓ,e}_ _[L,E]_ _ℓ_ =1 _,e_ =1 _[←]_ [prune][_][to][_][supernet(] _[{]_ **[W]** _[ℓ][}]_ _ℓ_ _[L]_ =1 _[,][ Sparsities][,]_ **[ X]** _[, K]_ [)]

14: _{S_ [�] _ℓ}ℓ_ _[L]_ =1 _[←]_ [evolutionary][_][search(] _[{]_ **W** [�] _ℓ,e}_ _[L,E]_ _ℓ_ =1 _,e_ =1 _[,][ {][S]_ [�] _[ℓ][}]_ _ℓ_ _[L]_ =1 _[, S][g]_ [)]
15: _S_ _[g]_ _←_ _S_ _[g]_ _×_ 0 _._ 5

16: **end for**
17: **Return** compress( _{_ **W** [�] _ℓ,e}_ _[L,E]_ _ℓ_ =1 _,e_ =1 _[,][ {][S]_ [�] _[ℓ][}]_ _ℓ_ _[L]_ =1 [)]


24




--- end of page=23 ---

**A.3** **Further Comparisons**


To further demonstrate the effectiveness of our proposed
method, Týr-the-Pruner, we
conducted a more comprehensive comparison. The competitors include the pure subnet
search framework SearchLLM

[35], the probe-based dynamic
pruning approach ProbePruning [19], the sparsity distribution optimizer Adapt-Pruner

[44], the coarse-and-fine combined approach CFSP [46],
the calibration-free approach
PruneNet [19], the structureindependent approach DISPLLM [11], the cluster-based
evolutionary pruning approach
EvoP [48], and the search
- nly approach DarwinLLM

[38]. The experimental results,
with competitor performance
taken from their respective papers, are presented in Table 9.



Table 9: **Further comparisons** . Perplexity on Wikitext2 (lower
is better) and 0-shot accuracy (%, higher is better, DarwinLLM
reported the 25-shot Arc-C benchmark) serve as the comparison
metrics. Optimal results are **bolded** .























significantly outperforms other
structured pruning methods,

|Model|Sparsity|Method|Wikitext2 ↓|BoolQ ↑ WinoGrande ↑ ARC-E ↑ ARC-C ↑|
|---|---|---|---|---|
|Llama-7B|0%|N/A|5.68|71.38<br>67.01<br>67.45<br>41.38|
|Llama-7B|20%<br>25%|SearchLLM<br>Týr-the-Pruner|**6.89**<br>7.36|70.98<br>74.92<br>64.23<br>36.52<br>**75.81**<br>**75.68**<br>**66.36**<br>**42.06**|
|Llama-2-7B|0%|N/A|5.12|77.68<br>69.06<br>76.30<br>43.43|
|Llama-2-7B|30%|PruneNet<br>DISP-LLM|-<br>6.85|-<br>61.09<br>53.20<br>33.53<br>-<br>62.27<br>59.81<br>33.19|
|Llama-2-7B|37.5%<br>40%|Týr-the-Pruner<br>ProbePruning|10.29<br>**8.01**|**68.87**<br>**66.93**<br>**71.13**<br>**38.31**<br>64.70<br>58.10<br>62.50<br>37.70|
|Llama-2-7B|50%|DISP-LLM<br>DarwinLM<br>Týr-the-Pruner|**9.84**<br>-<br>16.17|-<br>58.41<br>43.06<br>25.85<br>62.70<br>55.80<br>63.30<br>**38.10**<br>**65.54**<br>**62.12**<br>**66.12**<br>**33.62**|
|Llama-2-13B|0%|N/A|4.57|80.61<br>72.22<br>79.46<br>48.46|
|Llama-2-13B|20%<br>25%|EvoP<br>Týr-the-Pruner|6.33<br>**5.79**|-<br>68.00<br>73.00<br>40.00<br>**81.35**<br>**72.06**<br>**77.74**<br>**44.97**|
|Llama-2-13B|30%<br>37.5%|DISP-LLM<br>Týr-the-Pruner|**5.77**<br>7.17|-<br>66.85<br>63.80<br>39.42<br>80.76<br>**72.06**<br>**76.35**<br>**43.26**|
|Llama-2-13B|50%|CFSP<br>DISP-LLM<br>Týr-the-Pruner|-<br>**7.11**<br>9.59|-<br>64.17<br>62.33<br>38.05<br>-<br>59.27<br>52.57<br>33.28<br>74.46<br>**70.09**<br>**72.18**<br>**39.85**|
|Llama-3.1-8B|0%|N/A|5.84|82.17<br>73.56<br>81.31<br>51.54|
|Llama-3.1-8B|40%<br>50%|Adapt-Pruner<br>DarwinLM<br>Týr-the-Pruner|33.75<br>-<br>**30.89**|-<br>56.75<br>45.16<br>25.97<br>62.20<br>57.30<br>59.60<br>**34.20**<br>**66.64**<br>**61.80**<br>**65.86**<br>**31.83**|
|Llama-3-8B|0%|N/A|5.76|81.10<br>73.01<br>80.05<br>50.43|
|Llama-3-8B|40% MLP-only<br>25%|ProbePruning<br>Týr-the-Pruner|14.90<br>**13.14**|70.30<br>67.20<br>57.40<br>39.00<br>**76.02**<br>**71.11**<br>**75.63**<br>**42.15**|

achieving better performance even at higher sparsities compared to other methods at lower sparsities.
In particular, Týr-the-Pruner surpasses the search-based methods SearchLLM, EvoP, and DarwinLLM,
demonstrating the effectiveness of our effective local pruning approach, expected error accumulation,
and iterative prune-and-search strategy.



**A.4** **Efficiency Analysis on Non-isotropic Structural Pruning**


Large language models (LLMs) with non-isotropic pruning may be considered to exhibit inferior
inference efficiency compared to those with isotropic sparsity across layers. To explore, we provide
a comparative analysis of inference efficiency for Llama-3.1-8B and Mistral-Nemo, both pre- and
post-50% structural pruning. The evaluation was conducted on an AMD Instinct™MI250 Accelerator
using Pytorch (HipBlas), covering both prefilling and decoding tasks across a range of sentence
lengths, as illustrated in Figure 4.


The variance (Var) quantifies the degree of variation in sparsity under non-isotropic pruning conditions;
a larger variance indicates more fluctuation in sparsity across layers. As shown in Figure 4, the 50%
structural pruned LLMs achieve up to 1.3x or greater speedup in both prefilling and decoding tasks
compared to their dense counterparts across most sentence lengths. Variations in layer sparsity do
not have a significant impact on efficiency. A slight efficiency decrease is only observed when the
variance reaches 1. In this case, the reduction in efficiency is likely due to the frequent high sparsity,
which leads to more memory-bottlenecked “thin” matrix multiplications in the computational graph.


### Figure 5

Caption: (a) and Figure 5(b) compare the sparsity distributions of the MHA and FFN layers in Llama3.1-8B after 50% pruning with Týr-the-Pruner and the search-only methods, respectively. The sparsity

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
distribution obtained by Týr-the-Pruner resembles that of the search-only strategy, yet Týr-the-Pruner
performs better. Its search process is more refined, incorporating multiple rounds of expectation error
accumulation, ultimately leading to a superior sparsity distribution and higher performance in the
pruned model.


### Figure 5

Caption: (c) and Figure 5(d) compare the sparsity distributions of the MHA and FFN layers in the 50%

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
post-pruned Llama-3.1-8B across different iterations of Týr-the-Pruner. Týr-the-Pruner identifies a


25




--- end of page=24 ---

(a) Llama-3.1-8B prefilling benchmarks (b) Llama-3.1-8B decoding benchmarks


(c) Mistral-Nemo prefilling benchmarks (d) Mistral-Nemo decoding benchmarks


### Figure 4

Caption: Pre- and post-pruning large language model inference benchmarks.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


(a) MHA: Týr-the-Pruner vs. Search-only (b) FFN: Týr-the-Pruner vs. Search-only


(c) MHA: variations during iterations (d) FFN: variations during iterations


### Figure 5

Caption: Sparsity distribution of Týr-the-Pruner and the search-only strategy on Llama-3.1-8B.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


relatively ideal and coarse-grained sparsity distribution in the first search (with a sparsity interval of
12.5%). In the subsequent iterations (2nd, 3rd, and 4th), with sparsity intervals of 6.25%, 3.125%,
and 1.5625%, respectively, the sparsity distribution is progressively refined and optimized, ultimately
converging to an optimal solution.


26




--- end of page=25 ---

**A.5** **Sparsity Distribution of Different Pruning Methods**


Different pruning methods vary in the distribution of sparsity. Figure 6(a) and Figure 6(b) show
the sparsity distributions of MHA and FFN of Llama-3.1-8B after 50% pruning by a series of LLM
structural pruning methods, respectively.


(a) MHA (b) FFN


### Figure 6

Caption: Sparsity distributions with different structural pruning methods.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


ZipLM and OSSCAR maintain isotropic sparsity distribution. LLM-Pruner incorporates prior
knowledge, recognizing that the shallow and deep layers of LLMs are more pruning-sensitive and
thus preserve them while only isotropically pruning the intermediate layers. These three methods
fail to account for the unique characteristics of different LLMs, leading to clear suboptimal sparsity
assignments. Conversely, FLAP combines local activations and weights to assess the global sparsity
distribution, resulting in non-isotropic pruning. While this method seeks a balance between local and
global sparsity, it does not fully address the gap between them, making it challenging to achieve an

- ptimal sparsity distribution.


Týr-the-Pruner’s sparsity distribution clearly differs from that of other methods. It directly searches
for the optimal sparsity distribution at the global level without the local and global gaps. The resulting
sparsity distribution does not adhere to prior assumptions: for instance, the 2-nd FFN layer is largely
retained, while the 12-th FFN layer is entirely pruned, and there is no discernible pattern in the
sparsity ratio as layers become deeper or shallower. This demonstrates that model optimization
should fully account for the model’s unique characteristics.


**A.6** **Prune and Tune**


Table 10: **Results of pruning and finetuning** . Perplexity on Wikitext2 (lower is better) and accuracy
(%, higher is better) serve as the comparison metrics. MMLU employed a 5-shot benchmark, while

- ther tasks used 0-shot benchmarks. An asterisk (*) indicates a fine-tuned result. Optimal and
suboptimal results are **bolded** and underlined, respectively.






|Sparsity Method Wikit|ext2 ↓ Avg. Acc. ↑ Arc-C ↑ Arc-E ↑ BoolQ ↑ HellaSwag ↑ OBQA ↑ RTE ↑ WinoGrande ↑ MMLU ↑|
|---|---|
|0%<br>N/A<br>5.|84<br>64.77<br>51.54<br>81.31<br>82.17<br>60.04<br>33.20<br>71.12<br>73.56<br>65.20|
|37.5%<br>LLM-Pruner<br>70.<br>LLM-Pruner*<br>17.<br>NutePrune*<br>14.<br>FLAP<br>21.<br>FLAP*<br>18.<br>Týr-the-Pruner<br>18.<br>Týr-the-Pruner*<br>**12.**|93<br>32.87<br>19.68<br>32.07<br>40.03<br>27.55<br>13.20<br>52.71<br>50.83<br>26.88<br>97<br>47.11<br>32.51<br>64.18<br>63.09<br>47.79<br>26.00<br>55.23<br>58.56<br>29.53<br>31<br>51.33<br>36.77<br>62.92<br>68.78<br>49.45<br>29.20<br>58.12<br>66.93<br>38.49<br>54<br>43.07<br>23.98<br>52.15<br>64.62<br>36.50<br>23.40<br>55.60<br>58.17<br>30.17<br>47<br>48.43<br>34.39<br>62.33<br>65.87<br>44.77<br>28.80<br>56.68<br>58.88<br>35.74<br>09<br>53.46<br>39.68<br>73.53<br>70.55<br>47.12<br>30.00<br>58.84<br>66.54<br>41.43<br>**65**<br>**58.22**<br>**46.93**<br>**77.02**<br>**75.75**<br>**54.99**<br>**32.60**<br>**59.57**<br>**68.82**<br>**50.06**|
|50%<br>LLM-Pruner<br>288<br>LLM-Pruner*<br>27.<br>NutePrune*<br>23.<br>FLAP<br>134<br>FLAP*<br>51.<br>Týr-the-Pruner<br>30.<br>Týr-the-Pruner*<br>**19.**|.32<br>31.58<br>19.62<br>28.70<br>37.83<br>26.36<br>13.40<br>52.35<br>49.64<br>24.70<br>34<br>40.39<br>25.68<br>53.49<br>45.84<br>39.88<br>22.00<br>53.79<br>55.25<br>27.22<br>55<br>42.71<br>30.01<br>53.84<br>55.61<br>38.02<br>24.20<br>55.23<br>58.25<br>26.50<br>.28<br>36.59<br>20.99<br>43.18<br>52.29<br>29.43<br>16.80<br>52.71<br>54.14<br>23.18<br>29<br>43.01<br>29.18<br>53.32<br>60.83<br>37.15<br>22.00<br>56.68<br>57.14<br>27.77<br>89<br>47.79<br>31.83<br>65.36<br>66.64<br>39.99<br>24.80<br>58.12<br>61.80<br>33.76<br>**68**<br>**51.83**<br>**38.65**<br>**70.92**<br>**67.25**<br>**48.62**<br>**31.60**<br>**60.29**<br>**62.12**<br>**35.22**|



Structured pruning is often followed by parameter-efficient fine-tuning to restore model performance

[23, 40, 20]. To evaluate the benefits of our proposed method, Týr-the-Pruner, in the context of


27




--- end of page=26 ---

post-pruning fine-tuning, we conducted fine-tuning experiments on the Llama-3.1-8B model. The
fine-tuning employed the parameter-efficient method LoRA (rank=16) [14], and the post-pruned
LLM was fine-tuned for three epochs on the Alpaca-GPT4 dataset [32].


Table 10 presents the experimental results. Týr-the-Pruner demonstrates superior performance at both
37.5% and 50% sparsity, with many of the results without fine-tuning already outperforming those
fine-tuned by other methods. Furthermore, Týr-the-Pruner surpasses the state-of-the-art trainingaware pruning method, NutePrune [20].


**A.7** **Statistical Significance Analysis**



To verify the robustness of the proposed method, Týr-the-Pruner, we adjust the random seeds (the change of
random seeds triggers the change of
calibration samples and the change
in random sparsity shift) for multiple
(number of tests: n=5) experiments
and observe the error bar ( _±_ standard
deviation), as shown in Table 11.



Table 11: Statistical significance analysis for Týr-the-Pruner.


|Model|Sparsity|Wikitext2 ↓ BoolQ ↑ ARC-E ↑ ARC-C ↑|
|---|---|---|
|Llama-2-7B|25%|7.51_ ±_ 0.07<br>69.45_ ±_ 0.04 75.13_ ±_ 0.10 42.58_ ±_ 0.09|
|Llama-2-13B|25%|5.79_ ±_ 0.00<br>81.35_ ±_ 0.06 77.74_ ±_ 0.03 44.97_ ±_ 0.05|
|Llama-3.1-8B|25%<br>50%|10.38_ ±_ 0.11 76.36_ ±_ 0.12 77.23_ ±_ 0.09 45.48_ ±_ 0.06<br>30.89_ ±_ 0.21 66.64_ ±_ 0.26 65.86_ ±_ 0.33 31.83_ ±_ 0.16|



From the global observation of experimental results, the proposed method performs relatively
consistently in multiple randomized trials, with standard deviations within acceptable limits ( _<_ 0.21
for Wikitext2 perplexity and _<_ 0.33 for downstream performance). From the local observation of
experimental results, it can be seen that pruning yields a more stable performance for larger models

- r under lower sparsity ratios.


28




--- end of page=27 ---

**A.8** **Detailed Downstream Task Results**


Table 12: 0-shot acc (%) on ARC-Challenge.



Table 13: 0-shot acc (%) on ARC-Easy.


|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|43.43 48.46|42.32 50.43 51.54|48.81<br>55.72|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|36.18 43.86<br>37.97 43.60<br>41.81 46.25<br>**43.34** 45.05<br>38.40 44.45<br>41.55 49.15<br>42.41** 49.23** <br>40.02 42.15<br> 42.06 48.05|37.80 43.94 44.20<br>  37.63 43.69 44.20<br>  35.15 41.64 42.15<br>  24.06 19.28 34.04<br>  31.83 38.57 37.97<br>  38.23 40.19 42.49<br> 38.14 40.70 40.78<br>  33.45 41.30 41.47<br>** 38.82 47.44 49.15**|43.43<br>46.16<br>42.15<br>45.14<br>42.49<br>31.66<br>46.16<br>48.89<br>40.02<br>43.52<br>47.27<br>52.30<br>46.59<br>30.80<br>43.86<br>45.48<br>**48.55**<br>**54.35**|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|32.34 37.88<br>30.89 38.65<br>37.88 41.47<br>38.14 20.82<br>28.24 37.63<br>39.51** 46.93** <br>40.53 45.65<br>31.91 40.78<br>** 42.58** 44.97|30.97 26.88 27.39<br>  32.34 36.26 36.77<br>  28.33 35.32 37.46<br>  18.09 16.89 19.20<br>  22.35 26.11 24.57<br> 29.52 18.43 20.31<br>  18.17 27.47 23.46<br>  26.54 31.91 33.45<br>** 35.41 42.15 45.48**|33.96<br>38.99<br>33.87<br>40.10<br>38.99<br>24.49<br>37.37<br>23.21<br>31.66<br>31.40<br>43.69<br>27.39<br>42.66<br>27.22<br>36.52<br>40.70<br>**44.88**<br>**48.38**|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|28.58 31.14<br>28.24 32.08<br>32.00 36.60<br>27.99 21.76<br>17.58 24.40<br>33.53 32.08<br>35.49 33.96<br>29.18 35.75<br>** 38.31 43.26**|25.85 26.88 27.56<br>  25.43 27.30 27.05<br>  23.29 27.65 27.39<br>  20.39 20.90 19.97<br>  17.49 16.89 16.98<br>  20.56 19.80 21.16<br>  18.77 26.54 23.98<br>  24.40 25.17 23.98<br>**   30.97 38.99 39.68**|29.10<br>27.90<br>28.24<br>31.57<br>28.67<br>19.45<br>20.65<br>20.31<br>20.90<br>19.28<br>38.31<br>37.03<br>36.77<br>37.63<br>29.69<br>32.51<br>**38.31**<br>**42.41**|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|23.46 28.16<br>23.55 27.05<br>24.91 30.63<br>17.58 19.54<br>18.60 19.54<br>20.14 27.99<br>23.81 25.34<br>29.10 27.47<br>** 33.62 39.85**|21.84 23.29 23.56<br>  21.25 24.74 22.87<br>  18.86 20.99 21.50<br>  20.73 18.60 19.54<br>  19.11 17.32 19.62<br>  19.20 17.15 20.48<br>  20.56 17.58 19.97<br>  22.78 21.76 20.99<br>**   25.51 32.34 31.83**|26.19<br>32.22<br>24.49<br>21.67<br>19.45<br>18.52<br>18.00<br>18.00<br>18.52<br>21.59<br>23.72<br>21.16<br>27.13<br>20.82<br>25.34<br>28.24<br>**32.94**<br>**32.59**|


|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|76.30 79.46|74.49 80.05 81.31|79.67<br>83.00|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|65.87 75.55<br>68.39 75.04<br>73.40 77.78<br>74.62 76.43<br>72.05 77.10<br>75.72** 79.80** <br>**76.01** 79.59<br>71.38 72.69<br> 75.84 79.62|68.10 71.17 72.18<br>  64.69 73.44 75.67<br>  67.68 74.71 75.51<br>  48.95 28.70 64.44<br>  62.92 70.83 72.47<br> 71.51 73.74 75.34<br>  71.55 74.37 76.05<br>  64.44 73.36 74.16<br>** 72.94 79.08 79.80**|71.51<br>75.84<br>71.25<br>75.67<br>76.85<br>53.70<br>77.61<br>79.29<br>73.32<br>75.46<br>78.62<br>79.63<br>78.28<br>52.90<br>75.76<br>75.88<br>**79.84**<br>**81.61**|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|52.74 61.24<br>53.03 64.73<br>71.80 74.92<br>70.41 33.59<br>59.97 70.20<br>74.66** 78.45** <br>74.45 77.57<br>64.23 69.23<br>** 75.13** 77.74|49.58 38.85 43.18<br>  49.41 53.41 55.47<br>  58.67 67.80 68.60<br>  37.46 42.26 28.41<br>  50.72 59.43 57.79<br> 61.32 27.86 26.47<br>  27.95 53.70 40.03<br>  53.96 60.31 65.95<br>** 69.40 75.63 77.23**|52.57<br>63.30<br>52.23<br>62.50<br>71.46<br>47.56<br>70.83<br>53.66<br>65.87<br>64.02<br>75.88<br>50.04<br>75.59<br>51.60<br>67.22<br>68.69<br>**77.23**<br>**80.13**|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|41.58 48.95<br>36.11 48.57<br>62.75 67.47<br>57.03 32.37<br>38.93 54.76<br>68.48 61.95<br>68.90 62.16<br>53.45 58.16<br>** 71.13 76.35**|40.07 37.50 39.94<br>  40.49 39.27 40.45<br>  46.89 55.72 57.49<br>  26.94 25.72 25.00<br>  31.69 32.53 32.07<br>  27.99 27.10 27.02<br>  28.03 54.21 47.10<br>  46.55 46.63 52.15<br>**   64.52 72.56 73.53**|33.88<br>42.72<br>35.40<br>46.09<br>58.46<br>40.45<br>47.90<br>35.90<br>47.10<br>47.98<br>70.54<br>70.79<br>71.04<br>70.08<br>56.65<br>61.70<br>**71.38**<br>**75.51**|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|32.45 37.75<br>30.01 37.71<br>48.40 54.84<br>27.95 35.86<br>28.11 33.54<br>29.38 54.00<br>50.72 45.92<br>47.01 43.18<br>** 66.12 72.18**|30.39 31.23 32.83<br>  29.00 28.58 28.54<br>  36.20 41.33 41.62<br>  26.89 30.98 30.47<br>  24.49 28.24 28.70<br>  27.57 25.57 28.28<br>  27.15 28.07 26.05<br>  27.23 42.30 43.18<br>**   56.23 65.36 65.36**|32.87<br>36.28<br>30.18<br>28.96<br>43.56<br>35.23<br>32.79<br>35.94<br>28.70<br>28.16<br>50.84<br>49.96<br>59.22<br>41.92<br>52.61<br>52.57<br>**66.37**<br>**66.04**|



Table 15: 0-shot acc (%) on HellaSwag.





Table 14: 0-shot acc (%) on BoolQ.
















|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|77.68 80.61|73.00 81.10 82.17|82.17<br>85.14|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|74.77 75.84<br>61.13 68.90<br>73.12 80.67<br>71.68 77.28<br>**76.48** 80.43<br>69.36 82.84<br>69.02** 83.00** <br>70.98 76.21<br> 70.67 82.78|63.30 73.70 70.70<br>  62.72 72.78 70.06<br>  68.99 75.75 75.57<br>  51.90 53.64 63.09<br>  65.72 74.34 71.90<br>  65.60 75.63 77.00<br> 68.59 74.80 79.91<br>  60.06 73.49 71.87<br>** 72.32 80.12 80.24**|77.31<br>66.21<br>77.19<br>68.62<br>81.19<br>77.71<br>77.31<br>68.04<br>72.72<br>77.58<br>**82.26**<br>71.83<br>81.53<br>73.03<br>77.49<br>80.24<br>82.11<br>**82.94**|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|62.17 62.54<br>50.83 58.23 <br>68.93 79.27<br>68.96 62.17<br>62.97 68.35<br>67.19 81.31<br>66.42 79.48<br>65.47 68.81<br>** 69.45 81.35**|44.83 37.80 37.65<br>** 70.64** 63.85 59.14<br>  65.81 72.02 67.68<br>  46.02 48.90 42.17<br>  61.59 60.89 57.89<br>  59.20 56.02 65.08<br>  54.28 60.06 65.66<br>  64.89 68.29 67.28<br> 67.89** 76.02 76.36**|67.25<br>67.22<br>75.14<br>66.70<br>75.78<br>68.41<br>62.45<br>61.93<br>68.78<br>64.25<br>77.16<br>65.14<br>77.13<br>64.28<br>65.14<br>63.82<br>**79.39**<br>**82.26**|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|62.17 37.25 <br>62.11 62.78<br>63.00 71.44<br>62.26 62.17<br>61.74 62.11<br>64.89 76.79<br>64.65 74.25<br>63.46 65.60<br>** 68.87 80.76**|** 68.87** 56.57 55.66<br>  63.30 48.78 45.38<br>  42.08 50.49 46.85<br>  52.66 38.13 51.68<br>  50.70 41.31 40.03<br>  49.76 51.56 61.47<br>  49.54 58.01 62.26<br>  61.93 62.66 64.62<br> 66.33** 70.09 70.55**|45.60<br>58.99<br>63.12<br>64.62<br>65.41<br>60.06<br>62.05<br>49.97<br>62.35<br>61.87<br>69.91<br>62.72<br>67.37<br>62.26<br>62.54<br>65.50<br>**70.85**<br>**74.65**|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|62.17 62.20<br>54.83 59.51<br>57.16 62.26<br>46.91 62.14<br>38.23 61.31<br>43.79 64.80<br>61.62 62.94<br>58.50 65.14<br>** 65.54 74.46**|46.61 62.57 62.17<br>  44.40 55.32 51.41<br>  40.76 41.74 38.56<br>  41.59 54.77 40.37<br>  38.10 39.54 37.83<br>  44.95 54.19 57.43<br>  56.48 53.36 61.04<br>  51.25 61.65 52.29<br>**   62.26 65.63 66.64**|51.90<br>55.29<br>42.66<br>46.57<br>51.13<br>51.53<br>48.99<br>43.06<br>43.24<br>43.94<br>62.72<br>62.23<br>60.95<br>62.17<br>61.47<br>48.87<br>**62.17**<br>**65.26**|


|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|57.14 60.04|55.20 60.11 60.04|60.92<br>62.90|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|49.88 55.70<br>51.11 56.22<br>52.32 56.16<br>**56.53** 53.96<br>51.60 57.06<br>55.41 59.42<br>55.39** 59.53** <br>53.98 57.21<br> 55.88 59.39|49.76 55.12 55.09<br>  48.94 54.51 54.68<br>  47.73 52.03 50.97<br>  34.60 27.29 40.76<br>  43.62 49.63 50.02<br>  48.85 51.87 52.37<br> 48.60 51.35 55.21<br>  43.88 50.97 51.66<br>** 51.55 56.52 56.32**|54.93<br>56.26<br>54.79<br>55.95<br>54.57<br>50.66<br>55.97<br>52.57<br>51.28<br>52.34<br>57.92<br>54.77<br>57.83<br>54.05<br>54.20<br>51.15<br>**58.27**<br>**59.31**|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|41.94 47.70<br>42.16 49.34<br>46.16 49.84<br>51.21 34.47<br>38.80 46.81<br>51.57 55.93<br>51.61 55.16<br>47.73 51.42<br>** 52.86 57.49**|37.31 28.89 28.37<br>  39.36 43.97 43.92<br>  39.29 43.29 42.10<br>  28.87 28.04 27.25<br>  32.91 33.94 33.05<br>  33.39 32.32 30.47<br>  26.55 36.45 36.44<br>  37.10 42.54 43.16<br>**   46.62 53.10 52.87**|42.51<br>43.61<br>42.73<br>45.33<br>44.24<br>40.37<br>44.23<br>34.65<br>38.66<br>38.11<br>51.34<br>43.72<br>50.69<br>43.09<br>45.80<br>44.20<br>**58.27**<br>**55.04**|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|33.53 39.31<br>33.56 41.75<br>37.65 41.27<br>35.07 29.49<br>28.17 33.06<br>38.29 45.79<br>42.86 48.20<br>41.53 45.52<br>** 48.47 54.11**|31.44 32.11 30.71<br>  33.12 34.16 34.07<br>  32.08 34.14 33.37<br>  26.48 25.66 26.38<br>  26.68 27.52 27.55<br>  26.89 29.11 27.43<br>  26.66 30.70 31.49<br>  32.69 36.48 36.50<br>**   39.97 47.22 47.12**|27.72<br>34.81<br>31.45<br>32.18<br>34.50<br>32.48<br>30.63<br>26.08<br>29.38<br>28.36<br>40.65<br>35.69<br>41.37<br>35.09<br>37.49<br>39.28<br>**46.01**<br>**48.22**|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|28.61 32.44<br>27.72 31.64<br>30.91 32.35<br>26.65 28.52<br>26.76 27.78<br>26.53 35.84<br>32.21 32.16<br>37.02 41.13<br>** 42.62 49.45**|28.01 27.82 27.87<br>  28.31 27.71 26.01<br>  28.22 28.96 29.07<br>  26.32 26.72 26.73<br>  26.60 26.43 26.36<br>  26.46 27.52 26.42<br>  26.58 27.81 26.92<br>  26.29 32.96 29.43<br>**   33.68 39.71 39.99**|26.30<br>30.83<br>28.28<br>27.51<br>29.60<br>29.02<br>27.61<br>26.11<br>27.09<br>25.96<br>32.17<br>30.51<br>32.15<br>30.26<br>33.09<br>32.51<br>**38.68**<br>**40.24**|



29




--- end of page=28 ---

Table 16: 0-shot acc (%) on OpenBookQA.



Table 17: 0-shot acc (%) on RTE.










|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|31.40 35.20|31.00 34.80 33.20|33.40<br>36.40|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|28.20 33.20<br>30.00 30.20<br>**32.00** 34.00<br>31.60 32.00<br>28.40 34.40<br>31.60 34.60<br>31.20** 35.80** <br>29.20 32.40<br> 31.20** 35.80**|26.60 33.00 30.80<br>  25.60 30.80 31.20<br>  27.20 28.60 26.60<br>  15.40 13.20 22.80<br>  24.60 27.20 26.40<br>  27.50 25.80 26.60<br> 27.00 25.40 26.40<br>  27.60 30.60 30.40<br>**  29.20 33.40 34.60**|27.00<br>31.60<br>25.60<br>28.80<br>27.20<br>26.40<br>28.00<br>28.60<br>26.80<br>31.00<br>**34.20**<br>32.80<br>32.60<br>31.80<br>33.40<br>31.60<br>**34.20**<br>**34.80**|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|23.40 27.00<br>25.20 25.20<br>25.00 30.40<br>29.20 17.80<br>21.00 28.80<br>31.40** 34.80** <br>31.40 34.20<br>27.40 29.80<br>** 31.60** 34.20|23.20 19.60 18.40<br>  22.40 20.60 20.00<br>  23.00 24.40 22.60<br>  12.40 14.20 13.40<br>  15.40 19.80 18.00<br> 17.60 24.40 18.80<br>  13.00 20.20 21.60<br>  24.60 26.60 28.40<br>** 28.20 34.00 31.80**|20.40<br>23.00<br>23.40<br>22.20<br>23.00<br>23.00<br>26.60<br>21.20<br>20.40<br>21.00<br>29.20<br>19.20<br>24.40<br>21.00<br>29.80<br>28.00<br>**33.40**<br>**31.80**|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|21.60 22.00<br>17.00 21.40<br>19.80 27.20<br>17.60 13.40<br>12.80 17.80<br>25.60 27.00<br>25.20 26.40<br>24.20 27.20<br>** 31.00 32.40**|18.80 18.80 18.40<br>  17.00 17.80 16.60<br>  17.60 17.00 15.00<br>  11.80 12.60 11.80<br>  12.20 12.80 13.20<br>  13.80 14.40 13.20<br>  14.20 15.20 14.80<br>  21.60 22.80 23.40<br>**   26.00 29.80 30.00**|17.20<br>17.20<br>17.00<br>19.80<br>17.00<br>17.40<br>14.80<br>12.60<br>14.80<br>13.00<br>21.80<br>14.60<br>21.60<br>17.80<br>24.00<br>25.80<br>**26.20**<br>**29.20**|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|16.00 17.80<br>16.20 18.00<br>16.60 22.00<br>12.20 11.80<br>12.60 12.00<br>14.00 19.60<br>17.00 20.00<br>21.20 25.80<br>** 27.20 30.40**|18.20 16.80 17.00<br>  14.60 14.00 14.00<br>  14.20 14.00 12.80<br>  13.00 13.60 13.40<br>  12.40 13.40 13.40<br>  12.40 13.80 11.60<br>  11.80 11.60 10.60<br>  13.20 21.40 16.80<br>**   20.40 26.60 24.80**|15.40<br>14.00<br>16.60<br>16.00<br>14.80<br>15.00<br>13.60<br>13.80<br>14.80<br>15.40<br>17.40<br>13.20<br>16.60<br>14.40<br>21.40<br>21.00<br>**22.80**<br>**26.20**|


|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|62.82 65.34|54.87 67.87 71.12|68.95<br>64.26|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|55.96 63.90<br>63.54 58.48<br>64.26 58.84<br>58.48 64.98<br>62.82 61.73<br>61.37 63.18<br>58.84 61.73<br>57.76 59.21<br>** 66.06 67.15**|55.96 57.04 62.82<br>  57.40** 68.23** 70.40<br>  58.48 64.62 63.90<br>  46.93 57.04 57.76<br>  48.38 56.68 61.01<br>  50.18 66.06 60.65<br>  56.32 64.62 63.18<br>  50.54 52.35 55.96<br>**   56.68** 66.79** 71.84**|**70.04**<br>57.40<br>65.34<br>**64.26**<br>66.06<br>57.40<br>57.76<br>62.82<br>65.34<br>55.60<br>68.23<br>61.37<br>69.31<br>59.57<br>67.51<br>56.68<br>68.95<br>62.82|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|57.76 59.57<br>53.79 62.45<br>55.96 66.79<br>48.38 52.71<br>56.68 50.54<br>55.60 68.23<br>52.35 67.51<br>**62.82** 64.26<br> 62.09** 69.31**|48.38 62.82** 63.90**<br>  57.04 63.90 58.12<br>  59.21 58.12 57.40<br>  52.71 52.71 53.43<br>  52.35 52.35 53.07<br>  55.23 50.18 51.99<br>  48.74 54.51 50.54<br>  53.43 50.90 52.71<br>**  59.57 63.90** 63.18|64.98<br>**63.54**<br>64.26<br>58.12<br>57.40<br>52.71<br>53.79<br>53.43<br>55.23<br>53.07<br>**68.95**<br>53.43<br>63.18<br>53.79<br>63.90<br>49.46<br>65.34<br>59.57|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|**62.09** 52.35 <br>57.40 55.60<br>53.43 58.48<br>48.38 52.71<br>52.71 52.71<br>58.12 64.26<br>51.62 63.18<br>48.38 54.51<br> 61.37** 65.70**|** 58.48** 50.54 53.79<br>  57.04 54.87 58.84<br>  53.07 52.71 53.43<br>  54.51 46.57 50.54<br>  52.71 52.71 52.71<br>  52.71 50.54 53.79<br>  49.82 52.71 50.90<br>  46.57 51.26 55.60<br> 55.96** 60.29 58.84**|49.82<br>53.79<br>**61.37**<br>54.15<br>55.23<br>52.71<br>53.07<br>49.46<br>51.26<br>52.71<br>54.15<br>52.35<br>60.29<br>52.71<br>53.79<br>**55.96**<br>58.84<br>54.15|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|51.26 51.62<br>51.62** 61.01**<br>53.43 52.71<br>53.07 52.71<br>53.07 52.71<br>52.71 52.71<br>53.43 52.71<br>45.49 58.48<br>** 55.96** 59.93|51.99** 60.65** 51.62<br>**  57.40** 51.62 45.13<br>  53.07 53.43 55.96<br>  54.51 53.07 51.62<br>  52.71 52.71 52.35<br>  53.07 52.35 52.71<br>  50.90 47.65 51.26<br>  51.62 53.07 52.71<br>  53.43 58.84** 58.12**|51.26<br>50.54<br>53.43<br>49.82<br>53.07<br>52.71<br>52.71<br>49.46<br>52.71<br>53.07<br>52.71<br>51.26<br>53.79<br>55.96<br>52.71<br>55.23<br>**53.79**<br>**60.65**|



Table 18: 0-shot acc (%) on WinoGrande.



Table 19: 5-shot acc (%) on MMLU.










|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|69.06 72.22|69.06 73.01 73.56|73.64<br>73.64|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|68.98 71.03<br>68.35 70.96 <br>67.32 70.48<br>67.64 70.72<br>64.25 69.30<br>**70.48** 72.69<br>69.46 73.40<br>68.03 70.32<br> 70.09** 73.85**|67.64 71.67 70.09<br>** 69.61** 73.16** 73.16**<br>  60.54 67.17 66.61<br>  52.01 51.38 58.80<br>  60.93 65.35 65.51<br>  60.77 68.75 66.93<br>  60.30 68.19 68.27<br>  62.19 69.53 70.40<br> 67.40** 73.24** 72.53|70.24<br>**74.11**<br>71.35<br>74.03<br>70.64<br>64.01<br>66.54<br>62.27<br>66.06<br>68.51<br>72.53<br>65.98<br>71.59<br>65.11<br>70.24<br>68.43<br>**73.40**<br>72.69|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|65.67 70.96<br>64.25 69.77<br>65.27 70.40<br>63.69 54.30<br>57.70 61.56<br>67.96** 72.38** <br>68.11 70.56<br>64.72 68.03<br>** 68.51** 72.06|61.40 53.99 55.17<br>  63.14 67.32 65.27<br>  57.62 62.67 59.75<br>  51.78 48.78 49.57<br>  52.57 55.41 55.17<br> 51.85 58.88 58.25<br>  52.72 56.91 56.75<br>  57.06 62.75 63.46<br>** 64.01 71.11 71.11**|67.25<br>63.14<br>67.80<br>**71.59**<br>61.48<br>58.25<br>60.62<br>57.06<br>56.20<br>57.62<br>66.93<br>54.70<br>64.33<br>55.41<br>64.72<br>64.64<br>**71.11**<br>70.01|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|60.30 65.98 <br>59.04 65.82<br>46.73 65.19<br>49.49 49.57<br>51.07 52.57<br>60.46 63.46<br>61.64 63.14<br>61.72 64.96<br>** 66.93 72.06**|** 61.56** 54.54 55.09<br>  58.01 60.38 61.25<br>  55.09 54.78 53.04<br>  48.38 51.30 49.49<br>  49.96 51.07 50.83<br>  50.36 53.83 54.38<br>  50.04 55.09 54.46<br>  52.57 57.85 58.17<br> 60.22** 66.54 66.54**|59.19<br>53.99<br>59.43<br>55.49<br>55.80<br>54.85<br>49.33<br>51.38<br>51.78<br>50.36<br>57.38<br>51.14<br>57.38<br>53.83<br>57.38<br>56.20<br>**64.17**<br>**65.27**|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|56.20 61.80<br>51.38 59.19<br>54.62 56.99<br>49.17 50.04<br>50.43 49.88<br>48.54 56.91<br>51.54 52.80<br>56.51 61.72<br>** 62.12 70.09**|51.46 54.70 54.14<br>  51.22 53.75 52.17<br>  51.30 50.99 49.88<br>  51.07 47.43 51.22<br>  51.07 50.12 49.64<br>  51.54 49.17 52.49<br>  51.70 48.86 51.14<br>  50.51 52.80 54.14<br>**   53.28 60.30 61.80**|53.12<br>52.64<br>51.30<br>50.12<br>51.38<br>49.33<br>49.88<br>48.07<br>50.20<br>49.80<br>51.93<br>50.59<br>52.49<br>50.43<br>52.88<br>52.09<br>**59.43**<br>**59.04**|


|Sparsity|Method|LLaMA-2|LLaMA-3 .x|Mistral|
|---|---|---|---|---|
|Sparsity|Method|7B<br>13B|2-3B<br>0-8B<br>1-8B|7B-v0.3 Nemo|
|0%|N/A|45.84 55.06|65.27 56.17 65.20|62.18<br>68.83|
|12.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|**46.28** 54.16<br>45.34** 54.71** <br>42.88 53.43<br>39.99 44.64<br>37.40 50.51<br>41.33 53.57<br>40.02 53.25<br>40.60 47.66<br> 44.07 54.61|56.33** 55.90** 62.14<br> 58.70 53.10** 63.54**<br>  55.85 47.34 53.22<br>  25.78 26.69 43.26<br>  48.71 39.30 50.26<br>  58.38 48.31 58.95<br>  58.08 47.31 58.45<br>  49.86 41.90 52.29<br>** 59.50** 49.32 59.66|61.44<br>**68.10**<br>**61.69**<br>67.23<br>58.38<br>64.58<br>56.42<br>58.22<br>54.09<br>54.59<br>58.62<br>63.20<br>58.71<br>63.83<br>55.82<br>52.98<br>59.11<br>64.66|
|25%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|37.38 48.00<br>**43.63 53.82** <br>40.20 51.21<br>29.40 29.53<br>27.37 32.58<br>32.79 45.99<br>31.04 46.28<br>30.57 42.66<br> 34.90 52.18|36.10 34.30 34.88<br> 39.36 35.78** 59.68**<br>  36.46 35.02 32.30<br>  24.47 24.07 25.78<br>  30.95 28.49 27.47<br>  41.67 24.79 42.25<br>  44.89 24.89 43.06<br>  35.96 34.47 39.18<br>** 52.12 42.66** 51.22|**59.66**<br>50.66<br>55.28<br>**62.63**<br>45.76<br>55.35<br>37.15<br>25.25<br>33.72<br>32.60<br>51.22<br>51.12<br>49.51<br>51.18<br>47.99<br>31.09<br>52.17<br>57.68|
|37.5%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|**39.40** 48.03<br>38.94** 52.76** <br>35.04 46.98<br>25.79 24.05<br>24.65 25.50<br>30.85 43.82<br>28.98 42.65<br>26.17 36.58<br> 31.56 44.72|36.35 25.16 30.64<br> 28.57 27.54 27.13<br>  27.21** 29.13** 25.17<br>  23.87 23.44 25.62<br>  24.38 24.63 26.88<br>  31.82 24.76 33.97<br>  34.01 25.05 33.98<br>  29.24 27.89 30.17<br>** 43.78** 24.92** 41.43**|23.90<br>**50.72**<br>27.50<br>39.10<br>31.33<br>39.03<br>26.61<br>22.99<br>25.97<br>25.79<br>38.48<br>34.54<br>35.42<br>37.06<br>37.64<br>27.59<br>**42.96**<br>47.63|
|50%|ShortGPT<br>LaCO+<br>SliceGPT<br>Wanda-sp<br>LLM-Pruner<br>ZipLM<br>OSSCAR<br>FLAP<br>Týr-the-Pruner|25.77 23.97<br>23.81 40.19<br>**29.37** 38.94<br>24.55 25.97<br>25.81 24.63<br>25.70 29.40<br>25.35 31.36<br>23.87 29.40<br> 26.06** 40.29**|23.00 24.24 22.97<br>  25.95 25.48 26.08<br>  24.93 25.54 25.17<br>  25.72 24.21 25.76<br>  25.24 23.31 24.70<br>  25.35 24.87 26.16<br>  25.71 25.60 26.33<br>  23.49 23.40 23.18<br>**  30.46 26.46 33.76**|23.37<br>32.22<br>24.47<br>25.35<br>26.34<br>28.24<br>25.13<br>23.72<br>25.89<br>25.17<br>27.92<br>28.46<br>25.27<br>27.97<br>25.07<br>24.23<br>**33.51**<br>**33.34**|



30




--- end of page=29 ---
