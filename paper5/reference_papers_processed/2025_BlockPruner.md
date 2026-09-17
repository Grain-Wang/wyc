---
id: "2025_BlockPruner"
title: "BlockPruner: Fine-grained Pruning for Large Language Models"
authors: ["Longguang Zhong", "Fanqi Wan", "Ruijun Chen", "Xiaojun Quan", "Liangzhi Li"]
year: 2025
venue: "Findings of ACL 2025"
publication_status: "FORMALLY PUBLISHED"
category: "Structured Compression / Training-Free NAS"
source_pdf: "../reference_papers_origin/2025_BlockPruner.pdf"
paper_url: "https://aclanthology.org/2025.findings-acl.262/"
pdf_url: "https://aclanthology.org/2025.findings-acl.262.pdf"
code_url: "https://github.com/MrGGLS/BlockPruner"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# BlockPruner: Fine-grained Pruning for Large Language Models

**Authors:** Longguang Zhong, Fanqi Wan, Ruijun Chen, Xiaojun Quan, Liangzhi Li

**Venue / Year:** Findings of ACL 2025 (FORMALLY PUBLISHED)

**Category:** Structured Compression / Training-Free NAS

**Why this paper matters for low-cost post-training LLM NAS:** It turns MHA and MLP residual blocks into a fine-grained training-free search space scored by perplexity and pruned iteratively.

**Primary record:** [https://aclanthology.org/2025.findings-acl.262/](https://aclanthology.org/2025.findings-acl.262/)

**Local source:** [2025_BlockPruner.pdf](../reference_papers_origin/2025_BlockPruner.pdf)

## Full converted text

# **BlockPruner: Fine-grained Pruning for Large Language Models**

**Longguang Zhong** **[1]** **, Fanqi Wan** **[1]** **, Ruijun Chen** **[1]** **, Xiaojun Quan** [*] **[1]** **, Liangzhi Li** **[2]**

1School of Computer Science and Engineering, Sun Yat-sen University
2Meetyou AI Lab
{zhonglg5,wanfq,chenrj8}@mail2.sysu.edu.cn
quanxj3@mail.sysu.edu.cn, liliangzhi@xiaoyouzi.com



## Abstract


With the rapid growth in the size and complexity of large language models (LLMs), the
costs associated with their training and inference have escalated significantly. Research
indicates that certain layers in LLMs harbor
substantial redundancy, and pruning these layers has minimal impact on the overall performance. While various layer pruning meth
  - ds have been developed based on this insight,
they generally overlook the finer-grained redundancies within the layers themselves. In this
paper, we delve deeper into the architecture

  - f LLMs and demonstrate that finer-grained
pruning can be achieved by targeting redundancies in multi-head attention (MHA) and
multi-layer perceptron (MLP) blocks. We propose a novel, training-free structured pruning
approach called BlockPruner. Unlike existing layer pruning methods, BlockPruner segments each Transformer layer into MHA and
MLP blocks. It then assesses the importance

  - f these blocks using perplexity measures and
applies a heuristic search for iterative pruning.
We applied BlockPruner to LLMs of various
sizes and architectures and validated its performance across a wide range of downstream
tasks. Experimental results show that BlockPruner achieves more granular and effective
pruning compared to state-of-the-art baselines.


**1** **Introduction**


Large language models (LLMs) (Zhao et al., 2023;
Minaee et al., 2024) have demonstrated outstanding performance across a diverse array of natural
language processing tasks. However, their growing size and complexity have led to substantial
computational demands and increased memory usage, creating obstacles for deployment in resourceconstrained environments. Model compression
techniques (Gao et al., 2020; Li et al., 2023; Wang
et al., 2024) have emerged as a promising solution


*Corresponding author



|Col1|Col2|Col3|L<br>M<br>M|ayer<br>HA<br>LP|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


### Figure 1

Caption: Block Influence (BI) scores (Men et al., 2024)

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
for the Llama2-7B model (Touvron et al., 2023b) computed at both layer and block levels, where blocks/layers
with lower BI scores indicate less importance. The
model has 32 Transformer layers, each containing one
MHA and one MLP block, totaling 64 blocks. Blocklevel BI scores are generally lower than layer-level
scores, indicating finer-grained redundancies.


to address the challenges of deploying large, computationally intensive models. These techniques
aim to transform large models into more compact versions that require less storage and execute
with lower latency, while minimizing performance
degradation. Model compression methods typically
involve knowledge distillation (Huang et al., 2022;
Gu et al., 2024), quantization (Yao et al., 2022;
Dettmers et al., 2023), and pruning (van der Ouderaa et al., 2024; Ashkboos et al., 2024). In this
study, we primarily focus on pruning, a technique
that can be combined with these other methods to
achieve more effective and efficient compression.
Recent research on layer redundancy has shown
that LLMs contain a substantial number of redun
dant layers (Yang et al., 2024; Men et al., 2024;
Chen et al., 2024). Removing these layers does not
severely impact the model’s performance. To quantify this redundancy, researchers have investigated
various similarity-based measurement methods and
developed corresponding pruning strategies, includ






5065


_Findings of the Association for Computational Linguistics: ACL 2025_, pages 5065–5080
July 27 - August 1, 2025 ©2025 Association for Computational Linguistics




--- end of page=0 ---

ing layer merging (Yang et al., 2024) and layer
removal (Men et al., 2024). These methods not

- nly maintain the original width of the model architecture and avoid introducing additional structures,
but also demonstrate superior performance. Furthermore, Gromov et al. (2024) posited that this

- bserved redundancy may be intrinsically linked
to the residual structure (He et al., 2016) inherent in the Transformer architecture. Building on
this intuition and recognizing that Transformer layers can be further subdivided into smaller resid
ual blocks, namely multi-head attention (MHA)
and multi-layer perceptron (MLP) [1], we hypothesize that fine-grained block redundancies could exist within LLMs. Consequently, we conducted a
preliminary experiment to assess the significance

- f blocks at varying granularities. Specifically,
we sampled 32 instances from the Alpaca dataset
(Taori et al., 2023) and employed the Block Influence (BI) metric (Men et al., 2024) to evaluate
blocks at layer and block levels, as depicted in Figure 1. The results reveal that block-level BI scores

are generally lower than layer-level BI scores, indicating that fine-grained redundancies at the block
level are more significant within the model.
Building on these findings, we argue that finergrained pruning can be effectively implemented in
LLMs. Therefore, we introduce BlockPruner, a
novel, training-free structured pruning approach.
Unlike existing methods that focus on entire layers,
BlockPruner segments each Transformer layer into
MHA and MLP blocks. It then evaluates the im
portance of these blocks using perplexity measures
and applies a heuristic search for iterative pruning.

To validate the effectiveness of our method, we
applied BlockPruner to six LLMs of varying sizes
and architectures, and evaluated their performance
using five representative benchmarks. Our experimental results demonstrate that BlockPruner provides more granular and effective pruning compared to state-of-the-art baselines. Additionally,
we performed a series of analytical experiments to
investigate the impact of block type, block importance metrics, and data on pruning effectiveness.
Our findings confirm that LLMs contain substantial redundancies at the block level compared to the
layer level, demonstrating that fine-grained pruning
is more effective and appropriate than layer-based
approaches for compressing these models.


1In this work, unless otherwise specified, we refer to a
block as one of the two sublayers: MHA or MLP.



**2** **Related Work**


Pruning is a well-established technique to compress and accelerate neural networks by removing
superfluous weights or structures within models.
Pruning methods can be broadly categorized into
unstructured pruning and structured pruning.


**Unstructured pruning.** Unstructured pruning
targets individual weights, eliminating redundant
connections in neural networks by setting the corresponding weights to zero. For instance, SparseGPT
(Frantar and Alistarh, 2023) formulates pruning
as a layer-wise sparse regression problem, approximately solving it via a sequence of efficient Hessian
updates and weight reconstructions. Wanda (Sun
et al., 2024) computes the importance score of each
weight based on the product of the magnitude of
each weight and the norm of the corresponding input activation, identifying and removing weights
with lower importance scores. OWL (Yin et al.,
2024) identifies the correlation between pruning efficacy and the retention ratio of outliers, assigning
different sparsity ratios to each layer based on the

- bserved outlier ratio. RIA (Zhang et al., 2024b)
introduces a metric that considers both weight and
activation information, utilizing a permutation strategy for the input channels of weight matrices to
enhance pruning performance. BESA (Xu et al.,
2024) adopts a layer-wise pruning strategy, independently pruning each Transformer layer to minimize the reconstruction error between the outputs

- f pruned and dense Transformer layers, which
avoids accumulating errors across layers.


**Structured pruning.** Structured pruning focuses

- n broader network structures, such as neurons,

attention heads, or even entire modules. LLMPruner (Ma et al., 2023) utilizes gradient information to identify interdependent structures within
LLMs, pruning the least important groups and subsequently using Low-Rank Adaptation (LoRA) (Hu
et al., 2022) to restore the performance of pruned
models. LoRAPrune (Zhang et al., 2023) estimates
the importance of pre-trained weights using LoRA
gradients, iteratively removing redundant channels
in the weight matrices and recovering the pruned
models’ performance through fine-tuning. ShearedLLaMA (Xia et al., 2024) learns a set of pruning
masks to extract a sub-network with the specified
target structure from the source model, employing a dynamic batch loading algorithm to adjust
the data proportion of each domain based on the



5066




--- end of page=1 ---

loss reduction rate in different domains. SliceGPT

(Ashkboos et al., 2024) introduces the concept of
computational invariance, achieving compression
by removing rows or columns corresponding to
smaller principal components in the weight matrix.
LaCo (Yang et al., 2024) proposes a concise layer
pruning approach, reducing model size by merging
layers while maintaining the overall model structure. ShortGPT (Men et al., 2024) introduces a
metric for measuring layer importance, achieving
model compression by removing redundant layers.

Although unstructured pruning can maintain performance at higher pruning ratios, it often requires
additional hardware or library support, making
model acceleration impractical. Current structured
pruning methods typically require retraining the
model after pruning to avoid performance collapse.
While layer pruning techniques like LaCo eliminate
the need for additional retraining, their disregard
for fine-grained block redundancy makes it challenging to avoid significant performance loss.
Concurrent and independent of our research,
FINERCUT (Zhang et al., 2024a) also presents
a fine-grained block pruning algorithm. However,
their study does not delve into the rationale behind
treating Transformer layers as two distinct sublayers for pruning purposes. In contrast, we began by
conducting preliminary experiments that unveiled
the fine-grained block redundancy within Transformer models. This discovery led us to propose
the concept of minimal residual blocks. Additionally, we explored how pruning different types of
blocks impacts model performance. While FINERCUT assesses block importance by comparing the
similarity between the output logits of the original
and pruned models, this metric may fall short in
ensuring that the pruned model produces coherent
and semantically meaningful text, as it disregards
semantic nuances. In our approach, we evaluate
block importance using the perplexity of the pruned
model, a metric that more effectively captures the
fluency and quality of its outputs. To further support our perspective, we present a detailed comparison of these two metrics in Appendix D.


**3** **Methodology**


The proposed fine-grained block pruning method
(BlockPruner) is depicted in Figure 3. It begins
by decomposing each Transformer layer into two
minimal residual blocks (§3.1). We then evaluate
the importance of each block by leveraging per


### Figure 2

Caption: Illustration depicting that a Transformer layer

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
can be subdivided into two residual blocks.


plexity for our iterative block pruning framework
(§3.2). Finally, we iteratively prune the block with
the lowest importance (§3.3).


**3.1** **Minimal Residual Block**


Most contemporary LLMs (Brown et al., 2020;
Touvron et al., 2023a,b) are built upon the GPT architecture (Radford et al., 2019), which constitutes
a decoder-only model comprising multiple Transformer layers, an embedding layer, and a language
model head. As depicted in Figure 2, each Transformer layer can be decomposed into two primary
residual blocks: the multi-head attention (MHA)
block and the multi-layer perceptron (MLP) block.
Formally, consider the input hidden states of the
_i_ th Transformer layer, denoted as _Xi−_ 1 _∈_ R _[n][×][d]_,
where _n_ represents the length of the input sequence,
and _d_ represents the hidden layer dimension of the
model. The computational process within the _i_ th
Transformer layer can be represented as follows:


_Xi_ _[′]_ [= MHA(LN(] _[X][i][−]_ [1][)) +] _[ X][i][−]_ [1] _[,]_ (1)


_Xi_ = MLP(LN( _Xi_ _[′]_ [)) +] _[ X]_ _i_ _[′][.]_ (2)


_·_
Here, LN( ) denotes the layer normalization module and _Xi_ _[′]_ _[∈]_ [R] _[n][×][d]_ [ represents the intermediate]
hidden states after the MHA block.

Equations (1) and (2) indicate that both types

- f residual blocks can be abstracted into the same

computational formula. Hence, we argue that treating MLP and MHA blocks as the minimal units for
pruning is a reasonable choice, which is substantiated by our subsequent experimental results.


**3.2** **Block Importance**


While previous layer pruning methods (Men et al.,

2024; Chen et al., 2024) rely solely on the simi


















5067




--- end of page=2 ---

Block Importance



Distribution

















Layer 1









Layer 2


Layer n-1


Layer n





















Model Step 1



Step 2



Step 3



**...**



### Figure 3

Caption: Overview of our BlockPruner. We iteratively calculate the importance score for each block (MHA or

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
MLP) to obtain the block importance distribution, and subsequently remove the block with the lowest importance.



larity between layer inputs and outputs to measure
layer importance, we argue that this approach overlooks the layer’s contribution to the overall model
performance, while our metric considers its broader
impact on the final output. To address the drawback, we introduce _perplexity_ as a measure of block
importance. Specifically, we determine the importance score of each block by masking it and then
computing the perplexity of the new model on a
given dataset. Intuitively, a block with the lowest
importance score indicates that its removal results
in minimal performance degradation. This method
more effectively captures each block’s overall impact on the model’s performance, thereby more
accurately reflecting its significance.
Mathematically, perplexity is defined as the exponential of the average negative log-likelihood of
a sequence of words. Given a sequence of words
_w_ 1 _, . . ., wn_ and a language model that predicts the
probability _pθ_ ( _wi|w<i_ ) for each word _wi_, the perplexity PPL is calculated as:



ploying this finer-grained pruning approach, we
aim to better preserve the critical components and
capabilities of the model while aggressively removing the less significant blocks.

For an LLM _M_ with _L_ layers, we first divide
them into 2 _L_ blocks, consisting of MHA and MLP
blocks. Then, we perform iterative pruning search

- n a calibration dataset _C_ to sequentially prune _K_
blocks. The steps are outlined as follows:


**Step 1: Mask Block** . For each block _Bi_ (MHA

- r MLP) in _M_, we generate a modified model _M_ [ˆ]
by masking out this block.


**Step 2: Calculate Importance** . We compute
the perplexity _Pi_ for the modified model _M_ [ˆ] - n the
calibration dataset _C_ as the importance score for
the masked block _Bi_ .


**Step 3: Sort and Prune** . After computing the
importance scores for all blocks, we sort these
scores and remove the block with the lowest importance score from _M_ to create a new model.


**Step 4: Iterate** . The aforementioned steps are
iteratively repeated until _K_ blocks are removed.


By iteratively removing the blocks with the lowest importance scores, we aim to prune the LLM
while minimizing performance degradation on the
calibration dataset _C_ . This fine-grained block pruning approach provides a more targeted method for
pruning LLMs compared to traditional layer-level
pruning techniques, thereby facilitating more efficient model compression while better preserving
the model’s performance. The detailed procedure
for this pruning process is outlined in Algorithm 1.




_−_ [1]
PPL = exp(

_n_



_n_

- log _pθ_ ( _wi|w<i_ )) _,_ (3)


_i_ =1



where _pθ_ ( _wi|w<i_ ) denotes the probability of word
_wi_ given the preceding words in the sequence.


**3.3** **Iterative Search for Block Pruning**


Unlike existing layer pruning techniques, which
indiscriminately remove entire Transformer layers, we propose a novel fine-grained pruning strategy. This strategy selectively prunes MHA or MLP
blocks based on their defined importance. By em


5068




--- end of page=3 ---

**Algorithm 1** Iterative Block Pruning


**Input:** Model _M_ with _L_ layers, calibration
dataset _C_, number of blocks to remove _K_
**Output:** Pruned model _M_ _[∗]_

1: _M_ 0 _←M_

2: Split the model _M_ 0 into 2 _L_ blocks

3: **for** _j_ = 1 to _K_ **do**

4: **for** _i_ = 1 to 2 _L −_ _j_ + 1 **do**

5: Create model _M_ [ˆ] by masking block _Bi_ ;

6: Compute the perplexity _Pi_ for _M_ [ˆ]  - n the
calibration dataset _C_ ;


7: **end for**


8: Sort the blocks based on their perplexities;

9: Remove the block with the lowest perplexity
from _Mj−_ 1 and obtain _Mj_ ;

10: **end for**


11: _M_ _[∗]_ _←MK_

12: **return** Pruned model _M_ _[∗]_


**4** **Experiments**


In this section, we first introduce the experimental
setups and then present the main results.


**4.1** **Experimental Setups**


**Models.** To validate the widespread effectiveness

- f our pruning method, we experiment with three
series of models: Llama2 (Touvron et al., 2023b),
Baichuan2 (Yang et al., 2023), and Qwen1.5 (Bai
et al., 2023). These models share analogous architectures as described in equations (1) and (2).
Due to computational constraints, we employ 7B
and 13B models for Llama2 and Baichuan2, respectively, and 7B and 14B models for Qwen1.5.


**Baselines.** We compare our method with several
state-of-the-art structured pruning methods. The
specific baseline methods include **SliceGPT** (Ashkboos et al., 2024), **LaCo** (Yang et al., 2024), **Short-**
**GPT** (Men et al., 2024), and **Relative Magnitude**
(Samragh et al., 2023; Men et al., 2024). SliceGPT
achieves pruning by removing rows or columns
corresponding to smaller principal components in
the weight matrix. LaCo merges model layers from
deep to shallow, using model output representations to calculate thresholds to avoid over-merging.
ShortGPT eliminates redundant layers by calculating Block Influence. Relative Magnitude (RM)
_f_ ( _x_ )
uses _||_ _x_ + _f_ ( _x_ ) _[||]_ [ as an importance metric for layers,]
where _f_ ( _._ ) represents the non-residual part of the
Transformer layer, and employs the same pruning
method as ShortGPT. For SliceGPT, we used the of


ficial implementation [2] . For LaCo, we implemented
it based on their code and controlled the number of

pruned layers by adjusting the merging threshold.
For ShortGPT and RM, we reproduced the results
based on their manuscripts. More detailed implementation information is provided in Appendix A.


**Data and GPUs.** In our main experiment, we
utilize the Alpaca dataset (Taori et al., 2023) to
calculate importance scores. For our method, we
employ only 256 samples to compute perplexity,
and we discuss the influence of varying sample
sizes in Section 5.4. To ensure consistency, we use
the same number of samples for ShortGPT and Relative Magnitude methods as shown in Appendix A.
Moreover, the effect of sample size on ShortGPT
and Relative Magnitude is detailed in Appendix I.
All experiments are conducted on two RTX 4090
GPUs, and the execution times for different meth
- ds are reported in Appendix G.


**Evaluations.** Following SliceGPT, we use LM
Evaluation Harness (Gao et al., 2023) for evaluation and validation on five well-known benchmarks:
PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi
et al., 2021), HellaSwag (Zellers et al., 2019), ARCe and ARC-c (Clark et al., 2018). We also utilize
Wikitext2 dataset (Merity et al., 2016) for evaluating the perplexity after pruning. More comprehensive details of can be found in Appendix C.


**4.2** **Main Results**


Prior studies (Yang et al., 2024; Ashkboos et al.,
2024) have generally constrained the pruning ratio
to approximately 25%. In line with these studies,
we also restricted the pruning ratio to this range
in our main experiments. Since it is challenging
to achieve identical pruning ratios across different
methods and models, we select the closest available
pruning ratios for comparison.
As shown in Table 1, our BlockPruner method
significantly outperforms previous structured pruning baselines in terms of average performance and
achieves the best results across most benchmarks,
even though the pruning ratios in our method are
slightly higher than that of baselines. We also

- bserve that Llama2-13B maintains better performance at higher pruning ratios compared to
Llama2-7B, with Baichuan2 and Qwen1.5 exhibiting similar behavior. This suggests that as the


2As SliceGPT’s official code does not support Baichuan2
and Qwen1.5, we only employ it on the Llama2 series models.



5069




--- end of page=4 ---

**Model** **Method** **Ratio (%)** **PPL (** _↓_ **)** **PIQA** **WinoGrande** **HellaSwag** **ARC-e** **ARC-c** **Avg. Score**



**Llama2-7B**


**Llama2-13B**


**Baichuan2-7B**


**Baichuan2-13B**


**Qwen1.5-7B**


**Qwen1.5-14B**



Dense 0 5.47 79.05 69.06 75.99 74.54 46.16 68.96

SliceGPT 21.45 30.74 72.42 59.91 56.04 **63.64** 37.12 57.83

LaCo 21.02 50.39 68.34 60.46 54.08 55.39 35.84 54.82

RM 21.02 676.80 54.46 49.25 29.22 34.43 22.53 37.98

ShortGPT 21.02 18.45 70.24 **65.90** 62.63 56.06 36.09 58.18

BlockPruner 21.99 **11.51** **74.21** 62.43 **65.87** 61.07 **37.29** **60.17**


Dense 0 4.89 80.52 72.14 79.36 77.36 49.23 71.72

SliceGPT 21.52 23.95 74.32 65.59 60.71 **68.52** **42.41** 62.31

LaCo 24.37 13.97 72.42 59.27 60.44 54.34 34.56 56.21

RM 24.37 10.08 73.72 66.61 66.80 66.12 41.98 63.05

ShortGPT 24.37 20.06 72.74 **70.80** 67.80 60.35 41.30 62.60

BlockPruner 25.12 **8.16** **76.93** 66.30 **72.20** 65.82 41.38 **64.53**


Dense 0 6.04 77.48 68.27 72.18 72.98 42.75 66.73

LaCo 21.57 26.46 68.28 58.56 51.50 52.90 28.50 51.95

RM 21.57 189.78 59.96 52.33 30.87 38.17 23.63 40.99

ShortGPT 21.57 31.05 63.71 **62.67** 50.01 47.31 30.72 50.88

BlockPruner 22.45 **15.38** **69.75** 61.48 **58.09** **58.08** **33.02** **56.08**


Dense 0 6.66 78.84 70.40 75.23 74.07 47.70 69.25

LaCo 22.68 27.07 70.89 58.01 54.00 57.11 32.94 54.59

RM 22.68 17.70 68.99 67.88 63.78 57.49 37.54 59.14

ShortGPT 22.68 20.69 69.31 **68.27** 61.71 56.52 36.69 58.50

BlockPruner 24.19 **15.36** **71.44** 64.01 **64.20** **59.81** **37.88** **59.47**


Dense 0 7.95 79.22 66.46 76.92 62.16 42.66 65.48

LaCo 20.97 39.23 70.40 58.64 56.35 46.89 32.85 53.03

RM 20.97 2026.31 67.36 49.88 42.00 **54.17** 28.58 48.40

ShortGPT 20.97 49.88 69.53 **62.12** 58.87 43.60 32.17 53.26

BlockPruner 21.83 **20.58** **71.71** 55.56 **59.31** 53.70 **33.28** **54.71**


Dense 0 7.44 79.87 70.56 79.41 68.48 47.01 69.07

LaCo 22.25 16.32 71.55 58.33 60.16 53.70 34.04 55.56

RM 22.25 55.99 67.08 53.28 42.08 50.72 29.01 48.43

ShortGPT 22.25 1237.21 58.60 55.96 36.16 38.09 34.81 44.72

BlockPruner 23.72 **15.67** **75.24** **61.48** **66.92** **59.51** **39.08** **60.45**



Table 1: Zero-shot downstream task performance of various models using different pruning methods. “Dense”
represents the original, unpruned models. “PPL” means the perplexity on Wikitext2.



model scale grows, so does the number of redundant blocks, allowing for more pruning space.
Furthermore, it’s noteworthy that models with
lower perplexity on the Wikitext2 dataset tend to
perform better, which highlights the correlation between perplexity and model effectiveness. This
further supports the validity of perplexity as a reliable metric for evaluating model performance. Remarkably, although our method performs pruning
searches on the Alpaca dataset, it achieves lower
perplexity on the Wikitext2 dataset.
Finally, we observe that while approaches such
as ShortGPT and Relative Magnitude result in a
significant decline in model performance across
different tasks, BlockPruner stands out by avoiding such drastic reductions. This suggests that our
proposed block pruning method effectively mitigates performance degradation during the pruning
process. Due to space constraints, we have moved
the details of pruning baselines and comparisons
across various pruning ratios to Appendix L.



**5** **Analyses**


**5.1** **Ablation Study**

To assess the influence of various key operations
within the proposed pruning algorithm on its performance, we undertake a thorough ablation study
across six models. In particular, we first remove all
blocks with the lowest importance scores at once,
without the iterative search procedure. Then, we
substitute the fine-grained block pruning with a
coarser-grained layer pruning approach. The results of these experiments are shown in Table 2.
The experimental findings highlight that solely
relying on the perplexity metric without incorporating a search component can result in subpar
pruning results and even performance deterioration.
This phenomenon may stem from the intrinsic nature of perplexity, which, unlike other importance
metrics focusing solely on local block influence, is
inherently influenced by the interaction among multiple blocks due to its derivation from the model’s



5070




--- end of page=5 ---

**Model** **Method** **Ratio (%)** **Avg. Score**














|Col1|Col2|
|---|---|
|||
|MHA<br>||
|MLP<br>MHA&MLP||


|Col1|Col2|Col3|
|---|---|---|
||||
|MHA<br>|||
|~~MLP~~<br>MHA&M|LP||


|Llama2-7B BlockPruner 21.99 60.17 - search 20.95 55.89 (-7.11%)|Col2|Col3|Col4|
|---|---|---|---|
|**Llama2-7B**<br>- search<br>20.95<br>55.89 (-7.11%)<br>- block<br>21.02<br>58.63 (-2.56%)<br>**Llama2-13B**<br>**BlockPruner**<br>25.12<br>**64.53**<br>- search<br>25.08<br>58.58 (-9.21%)<br>- block<br>24.37<br>62.91 (-2.51%)<br><br><br>|**Llama2-7B**<br>- search<br>20.95<br>55.89 (-7.11%)<br>- block<br>21.02<br>58.63 (-2.56%)<br>**Llama2-13B**<br>**BlockPruner**<br>25.12<br>**64.53**<br>- search<br>25.08<br>58.58 (-9.21%)<br>- block<br>24.37<br>62.91 (-2.51%)<br><br><br>|**Llama2-7B**<br>- search<br>20.95<br>55.89 (-7.11%)<br>- block<br>21.02<br>58.63 (-2.56%)<br>**Llama2-13B**<br>**BlockPruner**<br>25.12<br>**64.53**<br>- search<br>25.08<br>58.58 (-9.21%)<br>- block<br>24.37<br>62.91 (-2.51%)<br><br><br>|**Llama2-7B**<br>- search<br>20.95<br>55.89 (-7.11%)<br>- block<br>21.02<br>58.63 (-2.56%)<br>**Llama2-13B**<br>**BlockPruner**<br>25.12<br>**64.53**<br>- search<br>25.08<br>58.58 (-9.21%)<br>- block<br>24.37<br>62.91 (-2.51%)<br><br><br>|
|**Baichuan2-7B**|**BlockPruner**<br>- search<br>- block|22.45<br>22.39<br>21.57|**56.08**<br>38.81 (-30.80%)<br>54.76 (-2.36%)|
|**Baichuan2-13B**<br>**BlockPruner**<br>24.19<br>**59.47**<br>- search<br>24.19<br>55.95 (-5.92%)<br>- block<br>24.95<br>58.22 (-2.10%)|**Baichuan2-13B**<br>**BlockPruner**<br>24.19<br>**59.47**<br>- search<br>24.19<br>55.95 (-5.92%)<br>- block<br>24.95<br>58.22 (-2.10%)|**Baichuan2-13B**<br>**BlockPruner**<br>24.19<br>**59.47**<br>- search<br>24.19<br>55.95 (-5.92%)<br>- block<br>24.95<br>58.22 (-2.10%)|**Baichuan2-13B**<br>**BlockPruner**<br>24.19<br>**59.47**<br>- search<br>24.19<br>55.95 (-5.92%)<br>- block<br>24.95<br>58.22 (-2.10%)|
|**Qwen1.5-7B**|**BlockPruner**<br>- search<br>- block|21.83<br>20.90<br>20.97|**54.71**<br>37.72 (-31.06%)<br>52.66 (-3.75%)|
|**Qwen1.5-14B**|**BlockPruner**<br>- search<br>- block|23.72<br>22.98<br>22.25|**60.45**<br>40.80 (-32.51%)<br>60.10 (-0.58%)|



Table 2: Average score of ablation study of BlockPruner

- n downstream tasks. “- search” indicates dropping the
iterative search procedure and directly removing blocks
with the lowest importance score. “- block” means we
substitute the fine-grained block pruning with a coarsergrained layer pruning approach.


- utput calculation. While perplexity aids in identifying redundant blocks within the model, it doesn’t
directly yield an optimal pruning sequence.
Furthermore, pruning at the layer level rather
than the block level yields less robust performance.
This observation indicates that the model contains
fine-grained redundancies, and segmenting layers
into smaller blocks for pruning allows for more
efficient removal of this redundancy, thereby better
preserving the model’s capabilities. Additionally,
we provide ablation experiments at higher sparsity
levels, with results presented in Appendix E.


**5.2** **Redundancies Between MHA and MLP**


To investigate the significance and roles of the
MHA and MLP modules in modern LLMs, we
conduct pruning experiments focusing exclusively

- n MHA or MLP blocks. We apply this pruning
strategy to two models of varying sizes, Llama27B and Llama2-13B, while keeping the pruning
ratios below 33%. The results illustrated in Figure
4 reveal several notable observations.

Before reaching a pruning ratio of 17%, pruning

- nly the MHA blocks results in less performance
loss compared to pruning MLP blocks and even
matches the performance of mixed pruning. This
indicates that MHA modules in LLMs may possess greater redundancy than initially anticipated,
whereas MLP modules are relatively less redundant.
However, when the pruning ratio surpasses 17%,
further pruning of MHA blocks leads to a sharp



### Figure 4

Caption: The impact of pruning MHA and MLP individually with different pruning ratios on model performance. “MHA&MLP” represents the original BlockPruner algorithm. Results show that MHA modules in

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
LLMs are more redundant than MLP modules.


decline in performance. This trend suggests that
as pruning advances, the redundant MHA blocks
are progressively removed, leaving only the crucial
MHA blocks. Moreover, in the larger model, the
sharp decline in performance occurs at higher pruning ratios, which is consistent with the finding that
larger models contain more redundant blocks. Such
redundancy may stem from factors like insufficient
training, resulting in higher initial redundancy.


We also examine the proportion of MHA blocks
removed during pruning. Specifically, we present
the number of MHA and MLP blocks removed at

different pruning stages. In Figure 5 (left), we set
the number of removed blocks to 60. In Figure 5
(right), the models have 22 and 28 blocks removed,
respectively, maintaining a pruning ratio of 30%.


The results in Figure 5 (left) for both models reveal a consistent tendency to initially remove only
MHA blocks. As the pruning process progresses
and more blocks are removed, the proportion of
MHA blocks being pruned follows a zigzag downward trend. Notably, the curve for Llama2-13B
shifts to the right compared to Llama2-7B, suggesting that the larger model contains more redundant
MHA blocks. This is further emphasized in Figure
5 (right), where, at the same pruning ratio, Llama213B prunes more MHA blocks than Llama2-7B.
Additionally, given that our pruning method tends
to remove more MHA blocks at equivalent pruning
ratios, it can significantly reduce the usage of the
key-value (KV) cache (Pope et al., 2023) in MHA,
which potentially accelerate the inference process.
To validate this, we also conducted a comparison

- f the inference speed among various models obtained through different pruning methods, with the
results detailed in Appendix F.



5071




--- end of page=6 ---

### Figure 5

Caption: **Left** : The proportion of MHA blocks removed

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
during the pruning process, relative to the total number

- f removed blocks. **Right** : The number of different
blocks removed from models at a pruning ratio of 30%.


**5.3** **Perplexity for Block Redundancy**


In this section, we explore the impact of different
block importance metrics. Generally, Block Influence (BI) and Relative Magnitude (RM) measure
the importance of a block based solely on its input and output hidden states, thereby reflecting the
block’s local influence. In contrast, perplexity is derived from the model’s output representations and
thus can better measure a block’s overall influence.
However, as indicated in the ablation study, using perplexity without the iterative search procedure leads to a significant decline in performance.
This suggests that while perplexity alone may not
be a strong block importance metric, our iterative
search method allows for a more effective use of it.

As illustrated in Figure 6, when BI and RM are
applied in dynamic pruning algorithms, they sometimes achieve performance comparable to perplexity at lower pruning ratios. However, as the pruning
ratio increases, their limitations become evident,
resulting in a sharp decline in model performance.
This suggests that these local metrics do not adequately capture the impact of different blocks on
the model’s overall performance.
In summary, perplexity leverages global information to effectively measure block redundancy,
especially when used with a dynamic pruning strategy. This combination captures the complex interactions among blocks. In contrast, local metrics
like BI and RM are useful in specific scenarios but
don’t reflect the overall contribution of blocks to
the model, particularly at higher pruning ratios.


**5.4** **Impact of Data on Pruning**


In the work on SliceGPT (Ashkboos et al., 2024),
the authors also used the Wikitext2 (Merity et al.,
2016) and Alpaca (Taori et al., 2023) datasets for
pruning experiments. They observed that the Al


### Figure 6

Caption: The impact of different block importance

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
metrics on the pruning performance of BlockPruner



### Figure 7

Caption: **Left** : The performance of BlockPruner on

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
the Alpaca and Wikitext2 datasets using a calibration
dataset of 256 samples. **Right** : Impact of sample sizes

- n BlockPruner’s performance on Alpaca, with the numbers indicating the sample sizes used.


paca dataset often yielded better pruning results. In

- ur study, we obtain similar findings. As shown in
### Figure 7

Caption: (left), when pruning Llama2-7B, the performance across different pruning ratios is significantly higher when using the Alpaca dataset compared to Wikitext2. We hypothesize that this may

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
be due to the Alpaca dataset being an instructionfollowing dataset, which is more closely aligned
with downstream tasks. This suggests that the
choice of dataset has a significant impact on the
final pruning performance of the model.
To determine the appropriate sample size and
analyze its impact on the pruning performance of
BlockPruner, we extract varying numbers of instances from the Alpaca dataset and conduct pruning experiments using Llama2-7B. The results presented in Figure 7 (right) indicate that increasing
the sample size beyond 256 yields no significant
improvement in the pruning effect of BlockPruner.
Therefore, we set the number of samples to 256.


**6** **Conclusion**


In this work, we introduce BlockPruner, a novel
structured pruning approach for efficiently pruning
LLMs. BlockPruner decomposes Transformer layers into two minimal residual blocks and leverages















5072




--- end of page=7 ---

a block importance metric based on perplexity in
conjunction with an iterative pruning search algorithm, where the two components work together
to progressively eliminate redundant blocks. Extensive experiments across various models show
that our method outperforms other baselines in
post-pruning performance. Our findings uncover
fine-grained block redundancy in LLMs, highlighting significant differences in redundancy levels
across different block types. We hope our work
contributes to a deeper understanding of the importance of different blocks within LLMs.


## Limitations


Our current work has three potential limitations.
First, while perplexity serves as a useful indicator

- f block importance, it may not be the optimal metric. Second, while our proposed pruning search algorithm is effective, other combinatorial optimization algorithms might identify superior pruning
sequences. Lastly, due to constraints in computational resources, we did not apply our method to
prune larger models. Nevertheless, our approach is
highly scalable and readily adaptable for pruning
larger models in future research.


**Ethics Statement**


The aim of this study is to provide a generalizable
pruning method for large language models. All
models and datasets used in our experiments are
publicly accessible and do not contain any private
information. We strictly adhere to the usage policies of these resources and utilize them solely for
research purposes.


**Acknowledgments**


This work was supported by the National Natural
Science Foundation of China (No. 62176270) and
the Guangdong Basic and Applied Basic Research
Foundation (No. 2023A1515012832).


## References


Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari
do Nascimento, Torsten Hoefler, and James Hensman.
[2024. SliceGPT: Compress large language models](https://openreview.net/forum?id=vXxardq6db)
[by deleting rows and columns. In](https://openreview.net/forum?id=vXxardq6db) _The Twelfth Inter-_
_national Conference on Learning Representations_ .


Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,
Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei
Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin,
Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu,



Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren,
Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong
Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang,
Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu,
Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang
Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang
[Zhu. 2023. Qwen technical report.](https://arxiv.org/abs/2309.16609) _arXiv preprint_
_arXiv:2309.16609_ .


Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi,
[et al. 2020. Piqa: Reasoning about physical com-](https://ojs.aaai.org/index.php/AAAI/article/view/6239)
[monsense in natural language. In](https://ojs.aaai.org/index.php/AAAI/article/view/6239) _Proceedings of the_
_AAAI conference on artificial intelligence_, volume 34,
pages 7432–7439.


Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
[Askell, et al. 2020. Language models are few-shot](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
[learners.](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) _Advances in neural information processing_
_systems_, 33:1877–1901.


Xiaodong Chen, Yuxuan Hu, and Jing Zhang. 2024.

[Compressing large language models by stream-](https://arxiv.org/abs/2403.19135v1)
[lining the unimportant layer.](https://arxiv.org/abs/2403.19135v1) _arXiv preprint_
_arXiv:2403.19135_ .


Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,
Ashish Sabharwal, Carissa Schoenick, and Oyvind
[Tafjord. 2018. Think you have solved question an-](https://arxiv.org/abs/1803.05457)
[swering? try arc, the ai2 reasoning challenge.](https://arxiv.org/abs/1803.05457) _arXiv_
_preprint arXiv:1803.05457_ .


Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
[Luke Zettlemoyer. 2023. QLoRA: Efficient finetun-](https://openreview.net/forum?id=OUIFPHEgJU)
[ing of quantized LLMs. In](https://openreview.net/forum?id=OUIFPHEgJU) _Thirty-seventh Confer-_
_ence on Neural Information Processing Systems_ .


[Elias Frantar and Dan Alistarh. 2023. Sparsegpt: Mas-](https://proceedings.mlr.press/v202/frantar23a)
[sive language models can be accurately pruned in](https://proceedings.mlr.press/v202/frantar23a)

[one-shot. In](https://proceedings.mlr.press/v202/frantar23a) _International Conference on Machine_
_Learning_, pages 10323–10337. PMLR.


Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman,
Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li,
Kyle McDonell, Niklas Muennighoff, Chris Ociepa,
Jason Phang, Laria Reynolds, Hailey Schoelkopf,
Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou.
[2023. A framework for few-shot language model](https://doi.org/10.5281/zenodo.10256836)
[evaluation.](https://doi.org/10.5281/zenodo.10256836)


Shangqian Gao, Feihu Huang, Jian Pei, and Heng
[Huang. 2020. Discrete model compression with re-](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.html)
[source constraint for deep neural networks. In](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.html) _Pro-_
_ceedings of the IEEE/CVF conference on computer_
_vision and pattern recognition_, pages 1899–1908.


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
[Alex Vaughan, et al. 2024. The llama 3 herd of mod-](https://arxiv.org/abs/2407.21783)
[els.](https://arxiv.org/abs/2407.21783) _arXiv preprint arXiv:2407.21783_ .



5073




--- end of page=8 ---

Andrey Gromov, Kushal Tirumala, Hassan Shapourian,
[Paolo Glorioso, and Daniel A Roberts. 2024. The un-](https://arxiv.org/abs/2403.17887)
[reasonable ineffectiveness of the deeper layers.](https://arxiv.org/abs/2403.17887) _arXiv_
_preprint arXiv:2403.17887_ .


Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024.

[MiniLLM: Knowledge distillation of large language](https://openreview.net/forum?id=5h0qf7IBZZ)
[models. In](https://openreview.net/forum?id=5h0qf7IBZZ) _The Twelfth International Conference on_
_Learning Representations_ .


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
[Sun. 2016. Deep residual learning for image recog-](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
[nition. In](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) _Proceedings of the IEEE conference on_
_computer vision and pattern recognition_, pages 770–
778.


Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
[Chen. 2022. LoRA: Low-rank adaptation of large](https://openreview.net/forum?id=nZeVKeeFYf9)
[language models. In](https://openreview.net/forum?id=nZeVKeeFYf9) _International Conference on_
_Learning Representations_ .


Yukun Huang, Yanda Chen, Zhou Yu, and Kathleen
McKeown. 2022. [In-context learning distillation:](https://arxiv.org/abs/2212.10670)
[Transferring few-shot learning ability of pre-trained](https://arxiv.org/abs/2212.10670)
[language models.](https://arxiv.org/abs/2212.10670) _arXiv preprint arXiv:2212.10670_ .


Albert Q Jiang, A Sablayrolles, A Mensch, C Bamford,
D Singh Chaplot, Ddl Casas, F Bressand, G Lengyel,
[G Lample, L Saulnier, et al. 2023. Mistral 7b.](https://arxiv.org/abs/2310.06825) _arXiv_
_preprint arXiv:2310.06825_ .


Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang,
[and Eduard Hovy. 2017. RACE: Large-scale ReAd-](https://aclanthology.org/D17-1082)
[ing comprehension dataset from examinations. In](https://aclanthology.org/D17-1082)
_Proceedings of the 2017 Conference on Empirical_
_Methods in Natural Language Processing_, pages 785–
794, Copenhagen, Denmark. Association for Computational Linguistics.


Yixiao Li, Yifan Yu, Qingru Zhang, Chen Liang,
Pengcheng He, Weizhu Chen, and Tuo Zhao. 2023.
[Losparse: Structured compression of large language](https://proceedings.mlr.press/v202/li23ap.html)
[models based on low-rank and sparse approximation.](https://proceedings.mlr.press/v202/li23ap.html)
In _International Conference on Machine Learning_,
pages 20336–20350. PMLR.


Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.

[TruthfulQA: Measuring how models mimic human](https://aclanthology.org/2022.acl-long.229)
[falsehoods. In](https://aclanthology.org/2022.acl-long.229) _Proceedings of the 60th Annual Meet-_
_ing of the Association for Computational Linguistics_
_(Volume 1: Long Papers)_, pages 3214–3252, Dublin,
Ireland. Association for Computational Linguistics.


Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blun[som. 2017. Program induction by rationale genera-](https://doi.org/10.18653/v1/P17-1015)
[tion: Learning to solve and explain algebraic word](https://doi.org/10.18653/v1/P17-1015)
[problems. In](https://doi.org/10.18653/v1/P17-1015) _Proceedings of the 55th Annual Meet-_
_ing of the Association for Computational Linguistics_
_(Volume 1: Long Papers)_, pages 158–167, Vancouver,
Canada. Association for Computational Linguistics.


Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023.

[LLM-pruner: On the structural pruning of large lan-](https://openreview.net/forum?id=J8Ajf9WfXP)
[guage models. In](https://openreview.net/forum?id=J8Ajf9WfXP) _Thirty-seventh Conference on Neu-_
_ral Information Processing Systems_ .



Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang,
Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng
Chen. 2024. [Shortgpt: Layers in large language](https://arxiv.org/abs/2403.03853)
[models are more redundant than you expect.](https://arxiv.org/abs/2403.03853) _arXiv_
_preprint arXiv:2403.03853_ .


Stephen Merity, Caiming Xiong, James Bradbury, and
[Richard Socher. 2016. Pointer sentinel mixture mod-](https://arxiv.org/abs/1609.07843)
[els.](https://arxiv.org/abs/1609.07843) _arXiv preprint arXiv:1609.07843_ .


Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish
[Sabharwal. 2018. Can a suit of armor conduct elec-](https://aclanthology.org/D18-1260.pdf)
[tricity? a new dataset for open book question answer-](https://aclanthology.org/D18-1260.pdf)
[ing. In](https://aclanthology.org/D18-1260.pdf) _Conference on Empirical Methods in Natural_
_Language Processing_ .


Shervin Minaee, Tomas Mikolov, Narjes Nikzad,
Meysam Chenaghlu, Richard Socher, Xavier Am[atriain, and Jianfeng Gao. 2024. Large language](https://arxiv.org/abs/2402.06196)
[models: A survey.](https://arxiv.org/abs/2402.06196) _arXiv preprint arXiv:2402.06196_ .


Reiner Pope, Sholto Douglas, Aakanksha Chowdhery,
Jacob Devlin, James Bradbury, Jonathan Heek, Kefan
[Xiao, Shivani Agrawal, and Jeff Dean. 2023. Effi-](https://arxiv.org/abs/2211.05102)
[ciently scaling transformer inference.](https://arxiv.org/abs/2211.05102) _Proceedings_

_of Machine Learning and Systems_, 5.


Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
[Dario Amodei, Ilya Sutskever, et al. 2019. Language](https://dcmpx.remotevs.com/net/cloudfront/d4mucfpksywv/SL/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
[models are unsupervised multitask learners.](https://dcmpx.remotevs.com/net/cloudfront/d4mucfpksywv/SL/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) _OpenAI_
_blog_, 1(8):9.


Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat[ula, and Yejin Choi. 2021. Winogrande: An adver-](https://dl.acm.org/doi/abs/10.1145/3474381)
[sarial winograd schema challenge at scale.](https://dl.acm.org/doi/abs/10.1145/3474381) _Commu-_
_nications of the ACM_, 64(9):99–106.


Mohammad Samragh, Mehrdad Farajtabar, Sachin
Mehta, Raviteja Vemulapalli, Fartash Faghri, Devang Naik, Oncel Tuzel, and Mohammad Rastegari.
[2023. Weight subcloning: direct initialization of](https://arxiv.org/abs/2312.09299)
[transformers using larger pretrained ones.](https://arxiv.org/abs/2312.09299) _arXiv_
_preprint arXiv:2312.09299_ .


Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter.
[2024. A simple and effective pruning approach for](https://openreview.net/forum?id=PxoFut3dWW)
[large language models. In](https://openreview.net/forum?id=PxoFut3dWW) _The Twelfth International_
_Conference on Learning Representations_ .


Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann
Dubois, Xuechen Li, Carlos Guestrin, Percy Liang,
and Tatsunori B. Hashimoto. 2023. Stanford alpaca:
An instruction-following llama model. [https://](https://github.com/tatsu-lab/stanford_alpaca)
[github.com/tatsu-lab/stanford_alpaca.](https://github.com/tatsu-lab/stanford_alpaca)


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al. 2023a. Llama: [Open and effi-](https://arxiv.org/abs/2302.13971)
[cient foundation language models.](https://arxiv.org/abs/2302.13971) _arXiv preprint_
_arXiv:2302.13971_ .


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti



5074




--- end of page=9 ---

Bhosale, et al. 2023b. [Llama 2: Open founda-](https://arxiv.org/abs/2307.09288)
[tion and fine-tuned chat models.](https://arxiv.org/abs/2307.09288) _arXiv preprint_
_arXiv:2307.09288_ .


Tycho F. A. van der Ouderaa, Markus Nagel, Mart Van
[Baalen, and Tijmen Blankevoort. 2024. The LLM](https://openreview.net/forum?id=DYIIRgwg2i)
[surgeon. In](https://openreview.net/forum?id=DYIIRgwg2i) _The Twelfth International Conference on_
_Learning Representations_ .


Wenxiao Wang, Wei Chen, Yicong Luo, Yongliu Long,
Zhengkai Lin, Liye Zhang, Binbin Lin, Deng Cai,
[and Xiaofei He. 2024. Model compression and effi-](https://arxiv.org/abs/2402.09748)
[cient inference for large language models: A survey.](https://arxiv.org/abs/2402.09748)
_arXiv preprint arXiv:2402.09748_ .


Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi
Chen. 2024. [Sheared LLaMA: Accelerating lan-](https://openreview.net/forum?id=09iOdaeOzp)
[guage model pre-training via structured pruning. In](https://openreview.net/forum?id=09iOdaeOzp)
_The Twelfth International Conference on Learning_
_Representations_ .


Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang,
Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao,
[and Ping Luo. 2024. BESA: Pruning large language](https://openreview.net/forum?id=gC6JTEU3jl)
[models with blockwise parameter-efficient sparsity](https://openreview.net/forum?id=gC6JTEU3jl)
[allocation. In](https://openreview.net/forum?id=gC6JTEU3jl) _The Twelfth International Conference_

_on Learning Representations_ .


Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang,
Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang,
Dong Yan, et al. 2023. Baichuan 2: Open large-scale
language models. _arXiv preprint arXiv:2309.10305_ .


[Yifei Yang, Zouying Cao, and Hai Zhao. 2024. Laco:](https://arxiv.org/abs/2402.11187)
[Large language model pruning via layer collapse.](https://arxiv.org/abs/2402.11187)
_arXiv preprint arXiv:2402.11187_ .


Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang,
Xiaoxia Wu, Conglong Li, and Yuxiong He. 2022.
[Zeroquant: Efficient and affordable post-training](https://proceedings.neurips.cc/paper_files/paper/2022/file/adf7fa39d65e2983d724ff7da57f00ac-Paper-Conference.pdf)
[quantization for large-scale transformers.](https://proceedings.neurips.cc/paper_files/paper/2022/file/adf7fa39d65e2983d724ff7da57f00ac-Paper-Conference.pdf) In _Ad-_
_vances in Neural Information Processing Systems_,
volume 35, pages 27168–27183. Curran Associates,
Inc.


Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh,
Yaqing Wang, Yiling Jia, Mykola Pechenizkiy,
Yi Liang, Zhangyang Wang, and Shiwei Liu. 2024.

[Outlier weighed layerwise sparsity (OWL): A miss-](https://openreview.net/forum?id=pOBvr1PxFd)
[ing secret sauce for pruning LLMs to high sparsity.](https://openreview.net/forum?id=pOBvr1PxFd)


Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin
[Choi. 2018. SWAG: A large-scale adversarial dataset](https://aclanthology.org/D18-1009)
[for grounded commonsense inference. In](https://aclanthology.org/D18-1009) _Proceed-_
_ings of the 2018 Conference on Empirical Methods in_
_Natural Language Processing_, pages 93–104, Brussels, Belgium. Association for Computational Linguistics.


Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali
[Farhadi, and Yejin Choi. 2019. Hellaswag: Can a](https://arxiv.org/abs/1905.07830)
[machine really finish your sentence?](https://arxiv.org/abs/1905.07830) _arXiv preprint_
_arXiv:1905.07830_ .



Mingyang Zhang, Chunhua Shen, Zhen Yang, Linlin
[Ou, Xinyi Yu, Bohan Zhuang, et al. 2023. Prun-](https://arxiv.org/abs/2305.18403)
[ing meets low-rank parameter-efficient fine-tuning.](https://arxiv.org/abs/2305.18403)
_arXiv preprint arXiv:2305.18403_ .


Yang Zhang, Yawei Li, Xinpeng Wang, Qianli Shen,
Barbara Plank, Bernd Bischl, Mina Rezaei, and Kenji
[Kawaguchi. 2024a. Finercut: Finer-grained inter-](https://arxiv.org/abs/2405.18218)
[pretable layer pruning for large language models.](https://arxiv.org/abs/2405.18218)
_arXiv preprint arXiv:2405.18218_ .


Yingtao Zhang, Haoli Bai, Haokun Lin, Jialin Zhao,
[Lu Hou, and Carlo Vittorio Cannistraci. 2024b. Plug-](https://openreview.net/forum?id=Tr0lPx9woF)
[and-play: An efficient post-training pruning method](https://openreview.net/forum?id=Tr0lPx9woF)
[for large language models. In](https://openreview.net/forum?id=Tr0lPx9woF) _The Twelfth Interna-_
_tional Conference on Learning Representations_ .


Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
[Zhang, Junjie Zhang, Zican Dong, et al. 2023. A](https://arxiv.org/abs/2303.18223)
[survey of large language models.](https://arxiv.org/abs/2303.18223) _arXiv preprint_
_arXiv:2303.18223_ .



5075




--- end of page=10 ---

**A** **Details of Implementations**


In this section, we detail our experimental setup.
We sampled from the Alpaca dataset with a fixed
random seed of 42. For SliceGPT, we followed
the original paper’s configuration, using 1024 samples, a sparsity ratio set at 30%, and a maximum
sequence length of 2048. For ShortGPT, RM, and
BlockPruner, we sampled 256 samples from the
dataset, with the same maximum sequence length

- f 2048. For LaCo, we adjusted the merging thresh
- ld using the provided code and data to achieve the
corresponding pruning ratio.


**B** **Details of Datasets**


**B.1** **Pruning Datasets**


**Alpaca** (Taori et al., 2023) is a general instructionfollowing dataset containing 52,000 questions.
Each sample comprises three fields: instruction, input, and response. We selected 10% of the dataset
and utilized 256 samples for the main experiments.
Perplexity calculation was performed uniformly
across all text in the samples without differentiation between fields.


**B.2** **Evaluation Datasets**


All downstream task datasets were partitioned and
evaluated using the default configuration of LM
Evaluation Harness.

**Wikitext-2** (Merity et al., 2016) is a collection

- f over 100 million tokens extracted from verified
Good and Featured articles on Wikipedia. This
dataset is commonly used to measure the quality of
a model’s text generation. We employed samples
from the pre-split test set for calculating perplexity.
**PIQA** (Bisk et al., 2020) is a dataset designed to
evaluate natural language models’ understanding

- f physical commonsense. It employs a multiplechoice format where the model selects the most

appropriate solution from two options given a goal.
**WinoGrande** (Sakaguchi et al., 2021) is an extensive dataset to evaluate models’ commonsense

reasoning capabilities. It comprises 44,000 questions. The dataset features fill-in-the-blank tasks
with binary options, aiming to select the correct

- ption for a given sentence that requires commonsense reasoning.
**HellaSwag** (Zellers et al., 2019) is also a dataset
designed to assess models’ commonsense reasoning abilities, specifically to highlight the limitations

- f current models in handling commonsense natural language reasoning tasks. Despite being triv


**Model** **Method** **Ratio(%)** **Avg.Score**


SliceGPT 21.45 57.93
**Llama2-7B**
SliceGPT _[∗]_ 21.45 57.83


SliceGPT 21.52 62.34
**Llama2-13B**
SliceGPT _[∗]_ 21.52 62.31


Table 3: Comparison of average performance on downstream tasks between the official SliceGPT results and

- ur reproduced results (indicated by “ _∗_ ” for our results).


ial for humans (with >95% accuracy), the dataset
presents significant difficulties for models. The
evaluation is conducted using four-way multiplechoice questions.
**ARC** (Clark et al., 2018) dataset comprises 7,787
multiple-choice science exam questions sourced
from various origins. Each question typically offers four answer options. These questions are categorized into two distinct difficulty sets: 2,590
questions for Challenge Set and 5,197 for Easy Set.


**C** **Details of Evaluations**


Ensuring a fair and comprehensive comparison,
we employed the same version of the LM Evaluation Harness as used in the SliceGPT experiments,

- btaining evaluation scores under identical experimental configurations. These scores closely match
those reported in the SliceGPT paper, as detailed
in Table 3. For consistency, we present our reproduced results in the main experiments.
For evaluating the performance of pruned models on downstream tasks, we utilized five multiplechoice QA datasets: PIQA, WinoGrande, HellaSwag, ARC-e, and ARC-c. Additionally, to assess text generation quality, we calculated perplexity using the test set of the Wikitext2 dataset. For
the downstream task evaluations, we adhered to
the default evaluation parameters and zero-shot settings, with a batch size set to 1. For perplexity
calculations, the maximum text length was set to
2048, maintaining a batch size of 1 as well.


**D** **Perplexity and JS Divergence in Block**
**Evaluation**


Recent work such as FINERCUT (Zhang et al.,
2024a) proposes a fine-grained pruning algorithm
that evaluates block importance using the **JS di-**
**vergence** between the output distributions of the

- riginal and pruned models. While this metric captures distributional shifts, it overlooks the semantic
fluency and coherence of generated text—key as


5076




--- end of page=11 ---

pects for maintaining the practical utility of LLMs.
In contrast, we adopt **perplexity (PPL)** as a global
importance metric derived from sequence-level loglikelihood, which more directly reflects how pruning impacts output quality and fluency in models.
To validate this perspective, we conducted experiments using both metrics across various model
scales and pruning ratios. The results, summarized
in Table 4, indicate that PPL consistently outperforms JS divergence under different configurations.
These findings demonstrate that PPL better reflects
the fluency and quality of the pruned model’s outputs, reinforcing its suitability as a block importance metric for LLM pruning.


**Model** **Metric** **Ratio(%)** **Avg.Score**


PPL 16.98/24.99/30.00/37.02 **64.33** / **57.65** / **55.01** / **49.36**
**Llama2-7B**
JS 14.99/26.00/29.01/37.02 63.76/57.08/53.71/49.04


PPL 14.54/25.12/31.62/34.88 **69.38** / **64.52** / **61.52** / **60.13**
**Llama2-13B**
JS 14.54/26.75/32.45/34.88 69.28/64.36/61.00/59.61


Table 4: Comparison of PPL and JS divergence across
different pruning ratios and model scales.


**E** **Ablation at Higher Sparsity**


BlockPruner is motivated by the goal of preserving model performance more effectively through
fine-grained block pruning. Evaluating how block
pruning performs at different levels of granularity, particularly under higher sparsity, is crucial for
supporting our motivations and claims. In light

- f this, we conducted ablation experiments with
higher sparsity ratios on Llama2-7B and Llama213B models. The results, shown in Table 5, confirm
that our approach remains effective, further validating the motivations behind BlockPruner.


**Model** **Unit** **Ratio(%)** **PPL(** _↓_ **)** **Avg.Score**


Block 30.00/37.02/43.03 **16.28** /27.47/ **49.65** **55.02** / **49.36** / **46.95**
**Llama2-7B**
Layer 30.03/36.04/42.05 16.58/ **27.05** /60.85 53.04/47.32/43.91


Block 31.62/36.52/41.39 **9.64** / **12.54** / **17.02** **61.52** / **59.34** / **54.15**
**Llama2-13B**
Layer 31.68/36.56/41.43 10.48/13.29/18.04 59.31/56.17/51.61


Table 5: Average score of BlockPruner at different pruning granularities under higher sparsity.


**F** **Inference Speed after Pruning**


In this section, we evaluate the inference speed by
measuring the time required to generate 128 tokens using models obtained from different pruning
methods, all employing KV caches for efficient decoding. Each configuration is repeated 20 times to
ensure statistically robust results, and we report the
average inference time across runs.



As shown in Table 6, our method consistently
achieves the greatest speedup at comparable pruning ratios. This improvement stems from the fact
that our approach prunes a greater proportion of
MHA blocks at the same overall pruning ratio compared to other methods, leading to a substantial
reduction in KV cache usage, which directly accelerates the inference process of LLMs.


**Model** **Method** **Ratio(%)** **Inference Time (ms)** **Speedup**



Table 6: The inference speed differences among models

- btained using different pruning methods, where “Original” denotes the unpruned model.


**G** **Time Costs of Pruning Methods**


Our approach relies on PPL to determine block importance, which requires calculating PPL before
pruning, making it challenging to design a more
efficient pruning strategy. We have compared other
block importance metrics (in Section 5.3) but found
that PPL still preserves the model’s performance
best. Moreover, since our method better maintains
model performance and pruning is one-time with
- ut increasing subsequent inference overhead, so
we believe the trade-off is worthwhile. The comparison results of pruning times between BlockPruner
and other methods are presented in Table 7.


**Model** **BlockPruner** **SliceGPT** **ShortGPT** **RM** **LaCo**


**Llama2-7B** 45 mins 2 hours 9 mins 2 mins < 1 mins 2 mins


**Llama2-13B** 2 hours 27 mins 3 hours 30 mins 2 mins < 1 mins 24 mins


Table 7: Execution time of BlockPruner and other pruning methods in the main experiment.


**H** **Post-training after Pruning**


We sampled 8,000 instances from the Alpaca
dataset and conducted post-training on the pruned
Llama2-7B and Llama2-13B models obtained via

BlockPruner using LoRA. All linear layers, excluding the embedding layer and the language model
head, were trained. The LoRA rank and LoRA
_α_ parameters were set to 32 and 10, respectively,
with a learning rate of 2e-4 and a batch size of 1.
Additionally, we configured the gradient accumulation steps to 4 and employed a linear learning rate



**Llama2-7B**


**Llama2-13B**



Original 0.00 4044.30 1.00
BlockPruner 24.00 **2747.88** **1.47**

SliceGPT 24.47 3226.68 1.25

ShortGPT 24.03 3094.36 1.31


Original 0.00 7285.73 1.00
BlockPruner 21.05 **3873.20** **1.88**

SliceGPT 21.52 4099.08 1.78

ShortGPT 21.93 4111.65 1.77



5077




--- end of page=12 ---

|Col1|Col2|Col3|
|---|---|---|
||||
||||
|BlockPrune<br>BlockPrune<br>Dense|r<br>r+Post train||


|Llama2-13B|Col2|
|---|---|
|||
|||
|||
|uner<br>uner+Post train||















### Figure 8

Caption: Average score of BlockPruner with varying

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
pruning ratios before and after post-training.


scheduler. We controlled the pruning ratios within
the range of 24% to 33%. The results are shown
in Figure 8. It can be seen that after training, our
models showed further improvement at different
pruning ratios. The Llama2-7B and Llama2-13B
models recovered to 89% and 92% of the performance of the unpruned models, respectively, when
pruned by approximately 1/4.


**I** **Sensitivity to Sample Size**


ShortGPT uses Block Influence as the importance
metric for layers, while RM uses Relative Magnitude. The former calculates the similarity between
the input and output hidden states of a layer, while
the latter utilizes the input and the non-residual
part of the output. In our preliminary experiments,
we found that these two metrics are not sensitive

to sample size. We sampled different numbers of
instances from the test set of the Alpaca dataset
to observe their impact on these metrics, and the
results are shown in Figure 9. We can see that all
the lines almost overlap, indicating that Block Influence and Relative Magnitude are not sensitive to
the sample size. We speculate that this may be due
to the limited information provided by the changes
in the input and output of a single layer.


**J** **Retention of Reasoning Ability After**
**Pruning**


To further assess the effectiveness of Block
Pruner on reasoning-intensive scenarios, we evaluate its zero-shot performance on the AQuA-RAT
dataset (Ling et al., 2017), a benchmark designed
for algebraic word problem solving with rationales. We compare BlockPruner against several
strong pruning baselines using the LLaMA2-7B
and LLaMA2-13B models.

As summarized in Table 8, BlockPruner consistently maintains competitive performance, achiev


ing accuracy close to the unpruned models and

- utperforming alternative pruning strategies such
as ShortGPT, Relative Magnitude (RM), and LaCo.
These results demonstrate that our method preserves the reasoning capabilities of LLMs even
under significant structural compression.


**Method** **LLaMA2-7B** **LLaMA2-13B**


Original 31.89 30.31
BlockPruner **29.92** **29.53**

ShortGPT 29.13 27.95

RM **29.92** 26.77

LaCo 24.80 27.95


Table 8: Zero-shot accuracy (%) on AQuA-RAT for
models pruned with different methods. The pruning
ratios for all methods are consistent with those used in

the main experiments.


**K** **Generalization to New Model Series**


To evaluate the generalization capability of BlockPruner to newly released model series, we conduct additional experiments on two recent architectures: Mistral-7B-v0.3 (Jiang et al., 2023) and
LLaMA3-8B (Grattafiori et al., 2024). We compare

- ur method against ShortGPT and Relative Magnitude (RM) across five pruning ratios, reporting
average downstream task scores on each.
As shown in Table 9, BlockPruner consistently
achieves superior performance on both model families across all sparsity levels. The performance
gap becomes more pronounced as the pruning ratio
increases, suggesting that our fine-grained pruning approach is particularly robust under highcompression regimes. These results confirm the
effectiveness and adaptability of BlockPruner beyond the models evaluated in the main paper.


**Model** **Metric** **Ratio(%)** **Avg.Score**


BlockPruner 25.12/27.55/29.98/36.00 **58.02** / **57.02** / **56.10** / **49.18**

**Mistral-7B-v0.3** ShortGPT 24.07/27.08/30.09/36.11 57.40/53.26/42.79/41.41

RM 24.07/27.08/30.09/36.11 37.75/35.89/36.49/35.52


BlockPruner 24.86/27.58/30.30/35.73 **52.36** / **50.08** / **46.38** / **45.41**

**LLaMA3-8B** ShortGPT 24.45/27.16/29.88/35.31 41.97/44.51/43.71/42.85

RM 24.45/27.16/29.88/35.31 39.34/38.90/37.48/36.80


Table 9: Downstream task performance on recently released model families. BlockPruner consistently maintains stronger performance across pruning ratios.


**L** **Varying Pruning Ratios**


To broadly demonstrate the superiority of our
method, we present the pruning effects of BlockPruner, ShortGPT, and Relative Magnitude on six
representative large models at different pruning



5078




--- end of page=13 ---

### Figure 9

Caption: The changes in Block Influence and Relative Magnitude of the model under different sample sizes.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.






















|ShortGPT|Col2|Col3|
|---|---|---|
|RM<br>BlockPruner|||















### Figure 10

Caption: Average score of BlockPruner with varying pruning ratios compared with ShortGPT and RM.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.



ratios. As depicted in Figure 10, our method effectively minimizes performance loss throughout
the pruning process, avoiding any sudden drops in
performance. In contrast, RM exhibits significant
instability and is prone to performance collapse.
ShortGPT performs relatively well, but in the pruning experiments on Qwen1.5-14B, it also leads to
severe performance degradation at higher pruning
ratios. Overall, the advantages of our method become more pronounced as both model size and
pruning ratio increase.



**M** **Evaluation on Additional Datasets**


We extended the primary experiment by incorporating four additional well-established evaluation
datasets: SWAG (Zellers et al., 2018), TruthfulQA
(Lin et al., 2022), OpenBookQA (Mihaylov et al.,
2018), and RACE (Lai et al., 2017). As illustrated in Table 10, our proposed method consistently surpasses previous pruning baselines across
this broader range of datasets, further demonstrating its effectiveness and generalization capability.



5079




--- end of page=14 ---

**Model** **Method** **Ratio (%)** **PIQA** **WinoGrande** **HellaSwag** **ARC-e** **ARC-c** **TruthfulQA** **RACE** **SWAG** **ObenBookQA** **Avg. Score**



**Llama2-7B**


**Llama2-13B**


**Baichuan2-7B**


**Baichuan2-13B**


**Qwen1.5-7B**


**Qwen1.5-14B**



Dense 0 79.05 69.06 75.99 74.54 46.16 25.34 39.43 76.65 44.00 58.91

SliceGPT 21.45 72.42 59.91 56.04 **63.64** 37.12 25.34 **37.22** 61.57 33.20 49.61

LaCo 21.02 68.34 60.46 54.08 55.39 35.84 **28.40** 29.57 61.75 **39.80** 48.18

RM 21.02 54.46 49.25 29.22 34.43 22.53 26.07 22.58 38.60 27.60 33.86

ShortGPT 21.02 70.24 **65.90** 62.63 56.06 36.09 26.81 34.07 64.84 37.20 50.43

BlockPruner 21.99 **74.21** 62.43 **65.87** 61.07 **37.29** 22.03 34.83 **69.81** 37.20 **51.64**


Dense 0 80.52 72.14 79.36 77.36 49.23 26.07 40.77 78.04 45.40 60.99

SliceGPT 21.52 74.32 65.59 60.71 **68.52** **42.41** 24.72 37.42 65.61 39.80 53.23

LaCo 24.37 72.42 59.27 60.44 54.34 34.56 23.62 31.87 67.93 **41.00** 49.49

RM 24.37 73.72 66.61 66.80 66.12 41.98 20.81 38.28 68.08 38.40 53.42

ShortGPT 24.37 72.74 **70.80** 67.80 60.35 41.30 24.60 37.80 68.67 **41.00** 53.90

BlockPruner 25.12 **76.93** 66.30 **72.20** 65.82 41.38 **24.97** **38.85** **72.94** 40.60 **55.55**


Dense 0 77.48 68.27 72.18 72.98 42.75 23.01 38.76 75.26 40.00 56.74

LaCo 21.57 68.28 58.56 51.50 52.90 28.50 21.42 31.10 62.37 **33.60** 45.36

RM 21.57 59.96 52.33 30.87 38.17 23.63 **25.09** 22.01 47.38 27.40 36.32

ShortGPT 21.57 63.71 **62.67** 50.01 47.31 30.72 24.60 30.62 55.81 31.20 44.07

BlockPruner 22.45 **69.75** 61.48 **58.09** **58.08** **33.02** 20.81 **33.21** **64.95** 32.20 **47.95**


Dense 0 78.84 70.40 75.23 74.07 47.70 26.93 41.15 76.87 43.60 59.42

LaCo 22.68 70.89 58.01 54.00 57.11 32.94 20.69 29.38 **67.79** 33.80 47.18

RM 22.68 68.99 67.88 63.78 57.49 37.54 25.46 **36.84** 64.50 33.80 50.70

ShortGPT 22.68 69.31 **68.27** 61.71 56.52 36.69 **26.93** 36.27 64.14 34.40 50.47

BlockPruner 24.19 **71.44** 64.01 **64.20** **59.81** **37.88** 23.50 36.75 67.43 **35.40** **51.16**


Dense 0 79.22 66.46 76.92 62.16 42.66 34.76 42.11 76.22 41.60 58.01

LaCo 20.97 70.40 58.64 56.35 46.89 32.85 25.34 32.92 **63.43** **37.20** 47.11

RM 20.97 67.36 49.88 42.00 **54.17** 28.58 21.66 23.54 58.32 35.40 42.32

ShortGPT 20.97 69.53 **62.12** 58.87 43.60 32.17 **32.07** 31.00 57.72 33.40 46.72

BlockPruner 21.83 **71.71** 55.56 **59.31** 53.70 **33.28** 25.70 **34.74** 61.32 33.80 **47.68**


Dense 0 79.87 70.56 79.41 68.48 47.01 35.86 41.05 76.72 43.60 60.28

LaCo 22.25 71.55 58.33 60.16 53.70 34.04 22.28 **33.78** 65.79 35.00 48.29

RM 22.25 67.08 53.28 42.08 50.72 29.01 26.44 27.08 58.64 32.40 42.97

ShortGPT 22.25 58.60 55.96 36.16 38.09 34.81 27.05 26.99 39.89 31.40 38.77

BlockPruner 23.72 **75.24** **61.48** **66.92** **59.51** **39.08** **30.60** **33.78** **67.39** **38.20** **52.47**



Table 10: Zero-shot downstream task performance of various models using varied pruning methods. “Dense” denotes
the original, unpruned models. All evaluations are conducted with the same configuration to ensure comparability.


5080




--- end of page=15 ---
