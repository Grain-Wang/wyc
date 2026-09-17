---
id: "2025_MultiPruner"
title: "MultiPruner: Balanced Structure Removal in Foundation Models"
authors: ["J. Pablo Munoz", "Jinjie Yuan", "Nilesh Jain"]
year: 2025
venue: "arXiv"
publication_status: "PREPRINT"
category: "Structured Compression / Training-Free NAS"
source_pdf: "../reference_papers_origin/2025_MultiPruner.pdf"
paper_url: "https://arxiv.org/abs/2501.09949"
pdf_url: "https://arxiv.org/pdf/2501.09949"
code_url: "https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# MultiPruner: Balanced Structure Removal in Foundation Models

**Authors:** J. Pablo Munoz, Jinjie Yuan, Nilesh Jain

**Venue / Year:** arXiv (PREPRINT)

**Category:** Structured Compression / Training-Free NAS

**Why this paper matters for low-cost post-training LLM NAS:** It combines residual-block, MLP-channel, and attention-head removal in a low-cost iterative pipeline and is a strong multi-axis subnet baseline.

**Primary record:** [https://arxiv.org/abs/2501.09949](https://arxiv.org/abs/2501.09949)

**Local source:** [2025_MultiPruner.pdf](../reference_papers_origin/2025_MultiPruner.pdf)

## Full converted text

## **MultiPruner: Balanced Structure Removal in Foundation Models**

**J. Pablo Muñoz** **[1]** [*] **, Jinjie Yuan** **[2]** [*] **, Nilesh Jain** **[1]**

1Intel Labs, 2Intel Corporation


{pablo.munoz, jinjie.yuan, nilesh.jain}@intel.com



## Abstract


Recently, state-of-the-art approaches for pruning large pre-trained models (LPMs) have
demonstrated that the training-free removal of
non-critical residual blocks in Transformers

is viable for reducing model size, achieving
results that outperform previous training-free
pruning approaches. Motivated by these findings, we extend BlockPruner (Zhong et al.,
2024) and propose **MultiPruner**, a pruning
approach that surpasses recent training-free
pruning methods by adopting a multidimensional, iterative, fine-grained pruning strategy. In MultiPruner, multidimensional pruning reinstates the structural balance in blockpruned models by sequentially compressing
along three dimensions: i) residual blocks,
ii) channels of multilayer perceptrons (MLP),
and iii) attention heads. This solution enhances zero-shot accuracy on downstream tasks
compared to other techniques while improving model compression ratios, producing compressed models with fewer computing and
memory requirements. Extensive experiments
demonstrate the advantages of the proposed
method across various large pre-trained models.
The code and pruning configurations are avail[able at https://github.com/IntelLabs/Hardware-](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning)
[Aware-Automated-Machine-Learning.](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning)


**1** **Introduction**


Large pre-trained models (LPMs), also known as
foundation models (FMs) (Bommasani et al., 2021),
such as GPT-4 (Achiam et al., 2024), are producing

- utstanding results across a variety of domains, and
have motivated an increase in investment in arti
ficial intelligence (AI)-related ventures. State-ofthe-art AI models often have billions of parameters
and require large clusters of graphics processing
units (GPUs) to train. Pre-trained models are usually subject to a less resource-intensive subsequent


*Co-first authors.



stage in which the model is adapted or fine-tuned
for a downstream task. Beyond the challenges of
training and fine-tuning, deploying these models
requires complex systems with significant computing and memory capacity. Two delimited stages
are in play during inference: prefill and decode. In
the prefill stage, all the required caches are created,
which tends to be compute-bound. In the decode
stage, the model uses the existing caches to generate new tokens, which tends to be memory-bound.
Given the substantial resource requirements for
training, fine-tuning, and deploying these models, model compression techniques have become
increasingly important. Pruning and quantizing
LPMs have been proposed to reduce resource requirements, improve model performance, and enable deployment to more limited environments.
However, choosing a particular compression algorithm requires considering the large parameter
space of these models and the cost in time and resources that this process will take. For instance,
when utilizing pruning, the algorithm should be capable of efficiently analyzing which model components might be removed, guaranteeing a sustained
performance, e.g., with a minor tolerable drop in

accuracy.
Recently, ShortGPT (Men et al., 2024) demonstrated that removing Transformer blocks based

- n their relative importance is a viable approach
for pruning large pre-trained models. BlockPruner
(Zhong et al., 2024) took this idea further to demonstrate that Transformer blocks can be partitioned
into their two sub-components based on the residual connections, i.e., minimal residual blocks and
showed that this more fine-grained pruning approach results in models with higher a pruning
ratio and accuracy. However, these works assume
that structured pruning should be applied to the
Transformer’s depth dimension, which can easily
lead to over-pruning in a single dimension, removing necessary layers or blocks. This paper con



--- end of page=0 ---

**Model**






























|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||**0.6**||||














|Col1|03<br>.|Col3|Col4|
|---|---|---|---|
||**0.3**|||







**Weight**

















### Figure 1

Caption: MultiPruner adopts a multidimensional fine-grained pruning method to make pruning more balanced,

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
resulting in a higher-performance pruned model.



siders these challenges and extends BlockPruner
(Zhong et al., 2024), resulting in a method to advance the state-of-the-art in structured pruning of
large pre-trained models. MultiPruner removes this
assumption and demonstrate that block pruning
can be complemented with additional pruning in

- ther dimensions, maintaining the target efficiency.
Our approach, MultiPruner, is a training-free approach that extends the benefits of state-of-the-art
training-free block pruning approaches and produces smaller models with higher accuracy. Specifically, MultiPruner operates in three pruning stages.
First, it prunes the least important residual blocks,
leveraging the insights from BlockPruner (Zhong
et al., 2024) to identify minimal residual blocks that
can be removed without significantly impacting
performance. Next, it applies a fine-grained pruning strategy to the MLP channels, followed by the
attention heads, aiming to prune the model more
accurately. This sequential pruning process ensures
that each dimension is optimally compressed, resulting in a smaller and more efficient model.

In summary, MultiPruner advances training-free
model compression and includes the following contributions:


1. A pruning algorithm, MultiPruner, that extends BlockPruner (Zhong et al., 2024) and
provides an iterative fine-grained pruning



strategy across multiple dimensions, leading to enhanced zero-shot accuracy on downstream tasks and improved model compression rates.


2. Studies to explore the challenges in multidimensional pruning, such as the pruning order
and ratios for different types of structures.


3. Extensive experiments demonstrating the effectiveness of MultiPruner across various

pruning ratios and models.


The following sections provide a comprehensive

- verview of the benefits and applications of MultiPruner, organized as follows: Section 2 discusses
the proposed method followed by Section 3 with
experimental results. Related work is discussed in
Section 4. We conclude with our thoughts on the
impact of our research.


**2** **Methodology**


MultiPruner is motivated by recent block pruning
algorithms, e.g., Zhong et al. (2024); Men et al.
(2024), that remove elements on a single dimension. Based on their importance, these methods
prune Transformer blocks or their sub-components,
like self-attention or multilayer perceptron (MLP)
blocks. However, focusing solely on these coarse
residual blocks might leave pruning opportunities




--- end of page=1 ---

unrealized in other dimensions. MultiPruner adopts
a fine-grained approach to structural pruning while
exploring the structural balance of the given architecture, since by removing complete residual
blocks in Transformers, we are altering the original
network design and the choices that might have
been determined after expensive exploration and
experimentation by the model’s creators. For instance, a pruned network becomes shallower after
applying block pruning, but its overall width remains unchanged. By pruning in other orthogonal
dimensions, MultiPruner attempts to reinstate this
balance, resulting in compressed high-performing
models that closely follow the original design considerations. The benefits of MultiPruner are demonstrated by experiments results in Section 3.


**Algorithm 1** Multidimensional Pruning with Fixed
Thresholds per Pruning Target


**Input:** Set of minimal residual blocks _M_ from a model
_m_, s.t. _M_ = _{Mi_ _|_ _Mi_ _∈_ _M,_ type( _Mi_ ) _∈_
_{_ MLP _,_ ATTN _}}_, Calibration dataset _C_, Metric _ϕ_, Target
pruning ratios _τ_ 1, _τ_ 2, and _τ_ 3 for each pruning modality,
MLP channel group size _g_ MLP, ATTN channel group size

_g_ ATTN.
**Output:** Pruned model _m_ _[∗]_

1: _τ ←_ 0
2: **while** _τ < τ_ 1 **do**
3: **for all** _Mi ∈M_ **do**
4: _Si ←_ BlockImportance( _Mi, m, C, ϕ_ )
5: **end for**
6: _M_ min _←_ arg min _Mi∈M Si_
7: _M ←M \ {Mmin}_
8: _τ ←_ PruningRatio( _m_ )
9: **end while**
10: WeightReordering( _m_ )
11: **for all** _t_ _∈_ _{_ MLP _,_ ATTN _}, τt_ _∈_ _{τ_ 2 _, τ_ 3 _}, gt_ _∈_
_{g_ MLP _, g_ ATTN _}_ **do**
12: _Mt_ = _{Mi | Mi ∈M,_ type( _Mi_ ) == _t}_
13: **while** _τ < τt_ **do**
14: **for all** _Mi ∈Mt_ **do**
15: _Si ←_ WidthImportance( _Mi_ [: _,_ :- _gt_ ] _, m, C, ϕ_ )
16: **end for**
17: _M_ min = arg min _Mi∈Mt Si_
18: _M_ min = _M_ min[: _,_ :- _gt_ ]
19: _τ ←_ PruningRatio( _m_ )
20: **end while**

21: **end for**
22: **return** _m_ _[∗]_ with the remaining and altered blocks in _M_


**Problem Definition** Given a dense model _m_,
e.g., Llama-2-7B (Touvron et al., 2023), associated with a set of minimum residual blocks _M_

from each Transformer block (self-attention or
MLP), and a target pruning ratio _τ_, MultiPruner
takes a finer-grained pruning approach compared
to other pruning solutions to obtain a model _m_ _[∗]_

with an associated subset of altered blocks from

_M_ . **MultiPruner’s objective is to find a high-**



**performing pruning configuration that results**
**in a similar pruning ratio to the competing state-**

**of-the-art block pruning while maintaining a**
**structural balance and improving its zero-shot**
**performance on downstream tasks.**


**2.1** **Multidimensional Fine-Grained Pruning**


MultiPruner targets three main elements for structure removal:


  - Residual Transformer Blocks (Depth)


  - MLP Channels (Width)


  - Attention Heads (Width)


In the depth dimension, MultiPruner removes
_iteratively_ the least important minimal residual
blocks as in BlockPruner (Zhong et al., 2024). In
the width dimension, MultiPruner removes groups

- f channels from the MLP and attention heads.

When exploring various architectural dimensions

- f the model for pruning and considering that
a full search of architecture configurations is
not practical for large pre-trained models, even
when using training-free approaches and zero-shot
evaluation, a natural research question arises:


_(1)_ _Should_ _MultiPruner_ _prune_ _the_ _model’s_
_depth and width in parallel or sequentially to_

_obtain a high-performing pruned model?_


The parallel strategy adds complexity and
does not provide insights regarding the contributions of pruning in each dimension. As detailed
in Section 3, experimentally, we have observed
that following sequential steps to prune different
model’s dimensions yields the best pruned models.
However, if the decision is to prune each dimension
sequentially, we are confronted with a second
research question:


_(2) What is the recommended order to se-_
_quentially explore the removal of structures in_

_each dimension?_


Intuitively, a coarse-to-fine-grained order, i.e.,
from blocks to MLP channels to attention heads,
will result in a more precise pruned model. This
intuition is supported by our experimental results
in Section 3, which demonstrate that MultiPruner
achieves better performance when it first removes
structures along the depth of the model. Once this




--- end of page=2 ---

stage is completed, it focuses on structures along
the width of selected components. As shown in
### Figure 1

Caption: , MultiPruner begins by pruning the least

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
important residual Transformer blocks, reducing
the model’s depth. Following this, it targets the
MLP Channels, and finally, it prunes the attention
heads, which raises an additional research question:


_(3) When must MultiPruner stop pruning in_
_each dimension or type of component?_


To answer this last question, we assign each

- f the three pruning targets with a pruning ratio
threshold, _τ_ 1 _, τ_ 2 _,_ and _τ_ 3 (i.e., the target ratio _τ_ ).
To discover values for these thresholds that yield
high-performing models, we utilize two search
strategies:


  - Fixed thresholds per pruning target.


  - Fixed threshold for the depth dimension
and evolutionary search to discover Pareto
   - ptimal configurations on the width dimension.


Next, we discuss these search variants in more

detail.


**Sequential Pruning with Fixed Targets** Experimentally, we discover the value for hyperparameters, _τ_ 1 _, τ_ 2 _,_ and _τ_ 3 that determine the pruning ratio of each type of component: complete
residual blocks, MLP channels, and attention heads,
respectively. The entire process is detailed in Algorithm 1 and Figure 1.
The algorithm begins by pruning the depth dimension, where MultiPruner removes the least important residual Transformer blocks iteratively until
the pruning ratio _τ_ 1 is reached (lines 2-9 in Algorithm 1). The importance metric used for all
pruning steps is the perplexity (PPL) on the calibration dataset. This step ensures that the model’s
depth is reduced in a controlled manner, preserving
the most critical blocks based on their importance

scores.

After completing the depth pruning, MultiPruner
performs a weight reordering step (line 10 in Algorithm 1). This step aims to prioritize the less
important channels and heads for pruning. By re
- rdering the weights based on an importance metric, the channels and heads are reordered such that
the least important ones are positioned last, making
them the primary candidates for pruning. In our



main experiment, we employed the L1 Norm as the
reorder metric, and more details can be found in
Appendix A.
Once the weight reordering is complete, MultiPruner shifts focus to the width dimension. It

sequentially prunes non-essential groups of channels in the MLP and attention heads, adhering to
the pruning ratios _τ_ 2 _,_ and _τ_ 3, respectively (lines
11-21 in Algorithm 1). The algorithm calculates
the importance scores for each component type and
removes the least important channel group or head
iteratively until the target pruning ratio is achieved.
This sequential approach ensures that each dimension is pruned effectively, balancing depth and
width reduction. By following this method, MultiPruner achieves a high-performing pruned model
that closely aligns with the original design considerations while significantly reducing the model’s
size.


**Fixed Threshold for Depth and Evolution-**
**ary Search for Width** After removing residual
blocks in the depth pruning stage, we have experimentally observed that stopping at half of the pruning ratio, i.e., _τ/_ 2, provides an effective heuristic
to initiate the pruning using evolutionary search in
the width dimension. This stopping point opens a
significant opportunity to remove elements in the
width dimension, resulting in a better balance in
the model.


**Algorithm 2** Multidimensional Pruning with Fixed
Delimitation and Evolutionary Search.


**Input:** Set of minimal residual blocks _M_ from a model
_m_, s.t. _M_ = _{Mi_ _|_ _Mi_ _∈_ _M,_ type( _Mi_ ) _∈_
_{_ MLP _,_ ATTN _}}_, Calibration dataset _C_, Metric _ϕ_, Target
pruning ratio _τ_, Number of evaluations in evolutionary
search _N_, Search Space _S_ .
**Output:** Pruned model _m_ _[∗]_

1: _t ←_ 0
2: **while** _t <_ _[τ]_ 2 **[do]**

3: **for all** _Mi ∈M_ **do**
4: _Si ←_ BlockImportance( _Mi, m, C, ϕ_ )
5: **end for**
6: _M_ min _←_ arg min _Mi∈M Si_
7: _M ←M \ {Mmin}_
8: _t ←_ PruningRatio( _m_ )
9: **end while**
10: WeightReordering( _m_ )
11: _{_ ( _Si, M_ _[∗]_ _i_ [)] _[}][N]_ _i_ =0 _←_
EvolutionarySearch( _Mi, m, C, ϕ, S_ )
12: _M ←M_ _[∗]_ arg max _i{Si|_ PruningRatio( _M_ _[∗]_ _i_ [)==] _[τ]_ _[}][N]_ _i_ =0
13: **return** _m_ _[∗]_


Algorithm 2 describes the steps followed to obtain the pruned model _m_ _[∗]_ .
The algorithm begins by pruning residual blocks




--- end of page=3 ---

**Model** **Method** **Ratio (%) PPL (** _↓_ **) PIQA WinoG HellaS ARC-e ARC-c Avg. Score**



**Llama2-7B**


**Llama2-13B**


**Baichuan2-7B**


**Baichuan2-13B**


**Qwen1.5-7B**


**Qwen1.5-14B**



Dense 0 5.47 79.05 69.06 75.99 74.54 46.16 68.96

SliceGPT 21.45 30.74 72.42 59.91 56.04 63.64 37.12 57.83

LaCo 21.02 50.39 68.34 60.46 54.08 55.39 35.84 54.82

RM 21.02 676.80 54.46 49.25 29.22 34.43 22.53 37.98

ShortGPT 21.02 18.45 70.24 **65.90** 62.63 56.06 36.09 58.18

BlockPruner 21.99 11.51 74.21 62.43 65.87 61.07 37.29 60.17

**MultiPruner** 21.96 **9.33** **74.65** 64.64 **68.94** **64.77** **41.13** **62.83**


Dense 0 4.89 80.52 72.14 79.36 77.36 49.23 71.72

SliceGPT 21.52 23.95 74.32 65.59 60.71 68.52 42.41 62.31

LaCo 24.37 13.97 72.42 59.27 60.44 54.34 34.56 56.21

RM 24.37 10.08 73.72 66.61 66.80 66.12 41.98 63.05

ShortGPT 24.37 20.06 72.74 70.80 67.80 60.35 41.30 62.60

BlockPruner 25.12 8.16 76.93 66.30 72.20 65.82 41.38 64.53

**MultiPruner** 25.01 **7.19** **77.80** **71.90** **75.62** **71.38** **46.50** **68.64**


Dense 0 6.04 77.48 68.27 72.18 72.98 42.75 66.73

LaCo 21.57 26.46 68.28 58.56 51.50 52.90 28.50 51.95

RM 21.57 189.78 59.96 52.33 30.87 38.17 23.63 40.99

ShortGPT 21.57 31.05 63.71 62.67 50.01 47.31 30.72 50.88

BlockPruner 22.45 15.38 69.75 61.48 58.09 **58.08** 33.02 56.08

**MultiPruner** 22.41 **12.37** **70.46** **64.72** **60.92** 56.99 **34.90** **57.60**


Dense 0 6.66 78.84 70.40 75.23 74.07 47.70 69.25

LaCo 22.68 27.07 70.89 58.01 54.00 57.11 32.94 54.59

RM 22.68 17.70 68.99 67.88 63.78 57.49 37.54 59.14

ShortGPT 22.68 20.69 69.31 **68.27** 61.71 56.52 36.69 58.50

BlockPruner 24.19 15.36 **71.44** 64.01 **64.20** **59.81** **37.88** **59.47**

**MultiPruner** 24.01 **10.99** 69.97 66.46 64.03 57.95 37.20 59.12


Dense 0 7.95 79.22 66.46 76.92 62.16 42.66 65.48

LaCo 20.97 39.23 70.40 58.64 56.35 46.89 32.85 53.03

RM 20.97 2026.31 67.36 49.88 42.00 54.17 28.58 48.40

ShortGPT 20.97 49.88 69.53 **62.12** 58.87 43.60 32.17 53.26

BlockPruner 21.83 20.58 71.71 55.56 59.31 53.70 33.28 54.71

**MultiPruner** 21.81 **18.22** **71.76** 59.59 **60.60** **59.01** **35.92** **57.38**


Dense 0 7.44 79.87 70.56 79.41 68.48 47.01 69.07

LaCo 22.25 16.32 71.55 58.33 60.16 53.70 34.04 55.56

RM 22.25 55.99 67.08 53.28 42.08 50.72 29.01 48.43

ShortGPT 22.25 1237.21 58.60 55.96 36.16 38.09 34.81 44.72

BlockPruner 23.72 15.67 75.24 61.48 **66.92** 59.51 39.08 60.45

**MultiPruner** 23.51 **12.94** **75.41** **62.98** 63.26 **69.19** **41.21** **62.41**



Table 1: Zero-shot downstream task performance of various models using different pruning methods. “Dense”
denotes the original (unpruned) models. “PPL” refers to the perplexity in Wikitext2. PIQA, WinoG, HellaS, ARC-e,
and ARC-c represent their respective accuracies.



in the depth dimension until half of the target pruning ratio ( _τ/_ 2) is reached. This heuristic provides
a balanced starting point for further pruning in the
width dimension. Following the depth pruning,
MultiPruner employs the Non-dominated Sorting
Genetic Algorithm (NSGA-II) to discover Pareto
- ptimal pruning configurations for the width dimension. This strategy combines the pruning of
MLP channels and attention heads simultaneously,
leveraging evolutionary search to achieve a more
precise and high-performing pruned model. Figure
2 illustrates the search progression for the evolutionary search on the width dimension, starting
from a block-pruned model with an 11% pruning
ratio. The plot shows the calibration perplexity



(PPL) against the pruning ratio, highlighting the
effectiveness of the evolutionary search in identifying optimal configurations. Specifically, when the
target pruning ratio is 22%, on the dense ratios, as
indicated in line 12 of Algorithm 2, the subnetwork
with this ratio (within the red line in the figure) and
the lowest PPL will be selected as the final pruned
model. As demonstrated in Section 3, this heuristic
results in compressed models with similar pruning
ratios but better average accuracy than sequential
pruning with fixed targets. However, it is essential
to note that the evolutionary search is computationally expensive and time-consuming to obtain better
results.




--- end of page=4 ---

**Model** **Method** **Ratio (%) PPL (** _↓_ **) Avg. Score**


Dense 0 7.81 67.67

**Llama3.2-3B** BlockPruner 9.39 13.07 62.31







**Llama3.1-8B**


**Llama3-8B**


**Qwen2.5-7B**



Dense 0 6.24 73.75

BlockPruner 10.65 10.58 66.75

BlockPruner 19.95 15.37 59.08


Dense 0 6.14 72.73

BlockPruner 10.13 10.88 66.46

BlockPruner 20.47 22.36 57.59


Dense 0 6.85 72.04

BlockPruner 10.34 9.88 67.44

BlockPruner 20.29 17.17 57.44


|SearchProgression(|NSGA2)|
|---|---|
|Block Pruned Model<br>Target Pruned Model||



Table 2: Zero-shot downstream task performance

- f more advanced LLMs compared to BlockPruner.
“Dense” denotes the original (unpruned) models. “PPL”
refers to the perplexity in Wikitext2. Average score
means the average accuracy across five downstream
tasks.


**2.2** **Performance-Recovery Stage**


To demonstrate the benefits of MultiPruner in production environments, it is essential to consider an
additional performance-recovery stage. This stage
involves fine-tuning the pruned model to recover
accuracy on downstream tasks before deployment.


**3** **Experiments**


**3.1** **Setup**


**Models** To demonstrate the broad applicability

- f MultiPruner, we conducted experiments using
the following models: Llama3.2-3B, Llama3.18B, Llama3-8B (Dubey et al., 2024), Llama2-7B,
Llama2-13B (Touvron et al., 2023), Qwen2.5-7B
(Yang et al., 2024a), Qwen1.5-7B, Qwen1.5-14B
(Bai et al., 2023), Baichuan2-7B and Baichuan213B (Yang et al., 2023). These models share similar architectures, and their Transformer blocks are
composed of self-attention (MHA, GQA (Ainslie
et al., 2023), etc.) and multilayer perceptrons
(MLP).


**Baselines** Building on the analysis by Zhong et al.
(2024), we include a comparison of MultiPruner
with other recent approaches, i.e. SliceGPT (Ashkboos et al., 2024), LaCo (Yang et al., 2024b), Short


### Figure 2

Caption: Search progression for evolutionary search

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
(NSGA2) on width dimension (Llama2-7B), starting
from a block-pruned model with a pruning ratio of 11%.
The red line is positioned at 22% (the target pruning
ratio). We select the subnetwork with the lowest PPL
value on this line as the final pruned model. This search
progression figure omits subnetworks with PPL _>_ 100.


GPT (Men et al., 2024) and Relative Magnitude
(Samragh et al., 2023), demonstrating the benefits

- f the proposed approach. In our environment, we
reproduced the results of BlockPruner, ensuring the
fairness of the competition.


**Data and Evaluations** For fairness, we follow
BlockPruner (Zhong et al., 2024) in using the Alpaca dataset [1] as the calibration dataset and employ
perplexity as the metric for Block or Width importance. The calibration set consists of either 128 or

256 samples. Regarding evaluation, MultiPruner
computes the perplexity (PPL) of Wikitext2 (Merity et al., 2016), and then utilizes _lm-eval-harness_
(Gao et al., 2023) to obtain the zero-shot accuracy

- n the following downstream tasks: Physical Interaction Question Answering (PIQA) (Bisk et al.,
2020), Large-scale Winograd Schema Challenge
(WinoGrande) (Sakaguchi et al., 2021), HellaSwag
(Zellers et al., 2019), and AI2 Reasoning Challenges (ARC-e, ARC-c) (Clark et al., 2018). The
reader can find all hyperparameters for our experiments in Appendix D and in the code repository.


**3.2** **Main Results**


The results presented in Tables 1 and 2 provide
a comprehensive comparison of various pruning
methods applied to different large language models, focusing on zero-shot downstream task performance. The metrics considered include perplexity


1https://github.com/tatsu-lab/stanford_alpaca




--- end of page=5 ---

**Ratio (%)** **Method** **PPL (** _↓_ **)** **Avg. Score**


BlockPruner 11.51 60.17

22 MultiPruner **9.33** 62.83

MultiPruner-Evol 9.74 **63.66**


BlockPruner 12.16 59.01

24 MultiPruner **10.01** 61.99

MultiPruner-Evol 10.61 **63.15**


BlockPruner 14.07 55.81

27 MultiPruner **11.59** 60.20

MultiPruner-Evol 12.00 **60.79**


BlockPruner 16.32 54.41

31 MultiPruner **13.77** **57.71**

MultiPruner-Evol 14.78 56.51


Table 3: Zero-shot downstream task performance
for Llama2-7B with BlockPruner, MultiPruner, and
MultiPruner-Evol at different pruning ratios. “PPL”
refers to the perplexity in Wikitext2. Average score
means the average accuracy across five downstream
tasks.


- n Wikitext2 and accuracy scores on five benchmark tasks. The _Dense_ row represents the performance of the original, unpruned models. Our
proposed method, MultiPruner, achieves the lowest perplexity and highest average scores across
most of the evaluated LLMs, outperforming other
training-free pruning approaches. For example,
regarding the Llama2-7B, MultiPruner achieves
the lowest perplexity (9.33) and the highest average score (62.83) among all pruning techniques,
indicating superior retention of the model’s language understanding capabilities and robustness
across diverse tasks. Overall, the results demon
strate that MultiPruner offers an effective trade-off

between model compression and performance retention, making it a promising solution for deploying large language models in resource-constrained
environments.


**3.2.1** **Block Pruning in Multiple Dimensions**


Notably, MultiPruner outperforms the block-level
pruning method, BlockPruner, by employing a
more sophisticated approach, block pruning in multiple dimensions. Specifically, MultiPruner subdivides the pruning units into MLP channels and
attention heads, allowing for more precise pruning than the coarse-grained block pruning used by
BlockPruner or other layer pruning methods. The
experimental results in the tables demonstrate that
this finer granularity leads to more effective model
pruning.



### Figure 3

Caption: Comparison of BlockPruner, MultiPruner and

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
MultiPruner-Evol at different pruning ratios for Llama27B. Average score means the average accuracy across
five downstream tasks.


**3.3** **Exploration of More Pruning Ratios**


As shown in Table 3 and Figure 3, we explore
the performance of the Llama2-7B model using
BlockPruner and our proposed method MultiPruner
across various pruning ratios (22%, 24%, 27%, and
31%). Across all pruning ratios, MultiPruner consistently achieves lower perplexity and higher average scores than BlockPruner. For instance, as
the pruning ratio reaches 27%, MultiPruner demonstrates superior performance with a perplexity of
11.59 and an average accuracy score of 60.20. At
this pruning ratio, MultiPruner reduces the perplexity of BlockPruner by 2.48 and improves the
average score by 4.39%. The consistent outperformance of MultiPruner can be attributed to its

finer-grained pruning strategy, which allows for
more precise and effective pruning, as evidenced
by the superior results across various metrics and
pruning ratios. The reader can find more results
with diverse pruning ratios in Appendix C.


**3.4** **Comparison with Multidimensional**
**Pruning via Evolutionary Search**


In Table 3 and Figure 3, in addition to comparing BlockPruner and MultiPruner, we also compare the zero-shot downstream task performance

- f MultiPruner and MultiPruner-Evol. Both Mul
tiPruner and MultiPruner-Evol outperform BlockPruner, demonstrating the effectiveness of our multidimensional pruning approach. When comparing
MultiPruner to MultiPruner-Evol, we observe that
MultiPruner-Evol often achieves slightly higher
average scores but slightly higher perplexity than
MultiPruner. For instance, at a 24% pruning ratio, MultiPruner-Evol achieves a higher average
score (63.15) compared to MultiPruner (61.99) but
a slightly higher perplexity (10.61 vs. 10.01). It
is essential to note that MultiPruner-Evol incurs

higher computational costs and requires more prun

















--- end of page=6 ---

**Pruning Dimension** **Wikitext2 Avg. Score**
**Block MLP Channel Attn Head** **PPL (** _↓_ **)**


✓(1) ✓(2) ✓(3) 9.33 **62.83**

_Order variations:_

✓(1) ✓(3) ✓(2) 9.19-0.14 61.87-0.96
✓(2) ✓(1) ✓(3) 9.71+0.38 62.18-0.65
✓(3) ✓(1) ✓(2) 10.26+0.93 61.63-1.20
✓(2) ✓(3) ✓(1) 10.08+0.75 60.14-2.69
✓(3) ✓(2) ✓(1) 9.89+0.56 62.19-0.64
✓(1) ✓(2) ✓(2) 11.25+1.92 62.21-0.62

_Existence variations:_

✓(1) ✓(2) ✗ 9.19-0.14 61.87-0.96
✓(1) ✗ ✓(2) 11.04+1.71 61.69-1.14
✗ ✓(1) ✓(2) **9.02**  - 0.31 57.79-5.04
✓(1) ✗ ✗ 11.51+2.18 60.17-2.66
✗ ✓(1) ✗ 9.41+0.08 57.83-5.00
✗ ✗ ✓(1) 142.20+132.87 40.60-22.23


Table 4: Ablation studies on Llama2-7B + MultiPruner

with the pruning ratio of 22%. Average score means
the average accuracy across five downstream tasks. The
numbers in the “Pruning Dimension” columns indicate
the order in which the pruning dimensions are applied.
“✗” indicates that the corresponding pruning dimension
is not applied.


ing time. In contrast, MultiPruner offers a more
balanced trade-off between pruning cost and task
performance with lower computational overhead,
making it more efficient.


**3.5** **Ablation Studies**


In Table 4, we present ablation studies on the
Llama2-7B model using MultiPruner with a pruning ratio of 22%. The table examines the impact

- f pruning dimensions and their pruning order on
model performance, considering both perplexity
and accuracy scores.
**Order Variations:** As shown in the table,

the results demonstrate that the default order of

Block, MLP Channel, and Attention Head pruning achieves optimal performance. When the order

- f the pruning stages is altered, we observe variations in performance. For example, starting with
MLP Channel or Attention Head generally leads
to higher perplexity and lower average scores, confirming that the default order, which moves from
coarse to fine granularity progressively, is more
effective for precise pruning.
**Existence Variations:** The existence variations

further validate the necessity of each pruning stage.
Removing any single stage results in degraded performance. For instance, omitting the MLP channel
pruning stage increases perplexity to 11.04 and decreases the average score to 61.69. Removing two



**Method** **Ratio** **PPL (** _↓_ **)** **Avg. Score**


Dense 0 5.47 68.96

MultiPruner 22% 9.33 62.83

MultiPruner **w/ tune** 22% **8.08** **64.18**


Table 5: Zero-shot downstream task performance of
the compressed Llama2-7B model with recovery tuning
(post-training). “PPL” refers to perplexity on Wikitext2. The average score is the mean accuracy across
five downstream tasks.


stages, such as MLP channel and attention head
pruning (i.e., BlockPruner), leads to even more
significant performance drops, with perplexity rising to 11.51 and the average score falling to 60.17.
Relying solely on attention head pruning results
in a drastic increase in perplexity to 142.20 and a
significant drop in the average score to 40.60, highlighting that most attention heads are essential and
cannot be excessively pruned.


**3.6** **Sensitivity Exploration of Self-Attention**

**and MLP**


_What happens when increasing/decreasing the_
_pruning of MLP channels or attention heads (the_
_weight of the overall target pruning ratio)?_ This
section explores the sensitivity of MLP channel
and attention head pruning. As shown in Figure
4, the four plots illustrate the impact of varying
the weight of the target pruning ratio allocated to
MLP Channels and Attention Heads on model performance.

The first two plots show the effect of changing
the MLP channel ratio weight while keeping the

- verall target pruning ratio constant. As the MLP
ratio weight increases from 20% to 80%, the Wikitext2 PPL decreases, but the average score initially
rises, reaching a maximum of around 50%, and
then declines. This indicates that excessive pruning

- f MLP channels can negatively impact the model’s

- verall performance. The latter two plots depict
varying weights’ influence on the attention head ratio. As this ratio weight increases from 0% to 25%,
the Wikitext2 PPL consistently rises, particularly
when the ratio weight exceeds 5%. Similarly, the
average score decreases as the head ratio weight
increases after 5%, with a notable drop beyond the
15% mark. This suggests that the attention heads
are more sensitive to pruning, and even a slight
increase in the pruning ratio can lead to substantial
performance loss.




--- end of page=7 ---

|10.50<br>10.25<br>10.00<br>9.75<br>9.50<br>9.25<br>9.00<br>8.75<br>20 30 40|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|20<br>30<br>40<br>8.75<br>9.00<br>9.25<br>9.50<br>9.75<br>10.00<br>10.25<br>10.50<br>|||50<br>60<br>7|50<br>60<br>7|80|80|


|63.0|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|60.5<br>61.0<br>61.5<br>62.0<br>62.5<br>Avg. Score||||||||
|60.5<br>61.0<br>61.5<br>62.0<br>62.5<br>Avg. Score||||||||


|1<br>1<br>1<br>1|6<br>5<br>4<br>3|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|1<br>1<br>1<br><br>ex|9<br>0<br>1<br>2<br>||||||


|62.75<br>62.50<br>Score<br>62.25<br>62.00<br>Avg.<br>61.75<br>61.50<br>61.25|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|61.25<br>61.50<br>61.75<br>62.00<br>62.25<br>62.50<br>62.75<br>Avg. Score|||||||



### Figure 4

Caption: The results of increasing/decreasing the weight of the target pruning ratio allocated to pruning MLP

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
Channels or Attention Heads (Llama2-7B with a target ratio of 22%). For example, an MLP ratio weight of 50%
means that the pruning ratio for the MLP channel pruning stage is 22% _×_ 50% = 11%. The yellow point represents
the ratio weight we adopted in the most experimental results, which is Block : MLP Channel : Attention Head =
44% : 52% : 4%. Note that the average score means the average accuracy across five downstream tasks.


**Attn** **MLP** **Attn** **MLP** **Attn** **MLP** **Attn** **MLP** **Attn** **MLP** **Attn** **MLP** **Attn** **MLP** **Attn** **MLP**



**Layer 0-7**


**Layer 8-15**


**Layer 16-23**


**Layer 24-31**




|Col1|3328 11008|3328 5888|Col4|4096 11008|4096 11008|3968 11008|4096 11008|4096 6912|
|---|---|---|---|---|---|---|---|---|
||**4096**<br>**11008**|**4096**<br>**11008**|**4096**<br>**11008**|**4096**<br>**11008**|**4096**<br>**5888**|**4096**<br>**11008**|**4096**<br>**11008**|**4096**<br>**11008**|
||**3584**<br>**3840**||**11008**|**3840**<br>**11008**|**3968**<br>**11008**|**11008**|**3968**<br>**11008**|**1792**|
||**1792**|**4096**<br>**3840**|**4096**<br>**3840**|**1792**|**11008**|**11008**|**3968**<br>**11008**|**4096**<br>**11008**|



### Figure 5

Caption: Details of the pruned Llama2-7B model obtained by MultiPruner, including the width of the self-attention

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
and MLP modules across different layers. The numbers within the colored boxes represent the channel sizes, while
the white boxes indicate blocks that have been completely removed.



**Inference Speedup**
**Method** **Ratio**
**Prefill Stage** **Decode Stage**


Dense  - 1.00 _×_ 1.00 _×_

MultiPruner 22% **1.32** _×_ **1.28** _×_


Table 6: Inference benchmark results for Llama2-7B

(dense vs. pruned). The batch size is 1 and the number

- f batches is 10. The prompt length is 512. Number of
new tokens is 16. The evaluation was conducted on an

Intel® Xeon® Platinum 8480+ processor with 56 cores,
leveraging Intel® Advanced Matrix Extensions (AMX)
to accelerate inference.


**3.7** **Recovery Tuning of the Pruned Model**


Following most of the work (Ma et al., 2023; Zhong
et al., 2024), we also conducted post-training on the
pruned model using the cleaned version of Alpaca.
The results, shown in Table 5, indicate significant
performance improvements after just two epochs

- f recovery tuning. Specifically, MultiPruner **w/**
**tune** achieves a reduced perplexity of 8.08 and an
increased average score of 64.18. The recovery
tuning phase effectively enhances the performance

- f the pruned model, making it closer to the original
dense model while requiring less computational

resources.


**3.8** **Inference Speedup**


The inference speedup results for Llama2-7B
model, as shown in Table 6, demonstrate the ef


ficacy of the MultiPruner method in enhancing
inference performance on an Intel® Xeon® Platinum 8480+ processor. The pruned model, with
a 22% reduction in parameters, consistently outperforms the dense model. Specifically, the prefill
phase exhibits a speedup of 1.32 _×_, while the decode phase achieves a speedup of approximately
1.28 _×_ . These results underscore the potential of
model pruning techniques to significantly reduce
inference latency, facilitating more efficient deployment of large language models in real-world applications. The reader can find additional inference
benchmark results across various batch sizes in

Appendix B.


**3.9** **Example of the Pruned Model**


### Figure 5

Caption: illustrates the pruning results for Llama27B obtained using MultiPruner. We observe that

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
the latter part of the model (layers 16 to 31) is
pruned more extensively. Notably, the intermediate
size of three MLP modules is reduced to 1792, and
block pruning is concentrated in the blocks of layers 16 to 31, suggesting that the latter part of the
model has more parameter redundancy, while the
earlier layers might be more critical. Overall, by
selectively reducing channel sizes and removing
entire blocks where necessary via MultiPruner, the
pruned model achieves a more compact architecture without compromising its ability to perform
downstream tasks effectively.




--- end of page=8 ---

**4** **Related Work**


The increasing size of large pre-trained models
(LPMs) has motivated the development of costeffective compression techniques to reduce these
models’ footprint and enable deployment in a
broader range of devices. Many methods have
been proposed, e.g., pruning (LeCun et al., 1989),
quantization (Gholami et al., 2021), and knowledge
distillation (Hinton et al., 2015), that overcome the
high cost of previous generations of compression
algorithms that had fewer resources and time constraints for their execution. Since the paper focuses

- n pruning, we discuss the evolution of pruning approaches and the latest developments emphasizing
efficiency in the compression of LPMs.


**4.1** **Pruning Large Pre-trained Models**


Pruning is a popular model compression technique
that targets removing or masking redundant elements in a neural network. Pruning methods utilize a _pruning criteria_ to accomplish this removal,
combined with several strategies to detect the least
critical model components efficiently. However,
special considerations are required when attempting to prune large pre-trained models. Previous
approaches that were successful in pruning small
Transformer-based models are not practical for
large models, e.g., Movement (Sanh et al., 2020)

- r Block pruning (Lagunas et al., 2021), because
they require expensive weights updates.


**4.1.1** **Unstructured Pruning**

Unstructured pruning approaches remove or mask
individual weights without any pre-determined pattern. Wanda (Sun et al., 2023) prunes weights utilizing an unstructured or semistructured strategy by
applying a pruning criterion based on the weight’s
magnitude and the norm of the input activation.
BESA (Xu et al., 2024) employs a reconstruction
loss per block to sparsify the model. Once sparsity has been induced in the model using any of
the mentioned techniques, parameter-efficient finetuning techniques (PEFT), e.g., Hu et al. (2022);
Muñoz et al. (2024), can be applied to recover the
accuracy for a downstream task.
Although unstructured pruning can achieve high
levels of sparsity, it faces limitations due to the
requirement of complex decompression algorithms.


**4.1.2** **Structured Pruning**

Structured pruning focuses on removing elements
at a higher granularity than unstructured pruning.



For instance, in the case of Transformer blocks,
these algorithms might remove attention heads or
groups of channels in linear layers of the multilayer perceptron (MLP) component. The benefits

- f _structured_ pruning are more straightforward to
realize than its unstructured counterpart since one
can extract the pruned model as a smaller version

- f the original pre-trained model and utilize the
same runtime used by the dense model to realize
the benefits and acceleration.

LoRAPrune (Zhang et al., 2024) proposes a
structured pruning approach for LPMs that is
guided by analyzing the weights and gradients of
low-rank adapters (LoRA) (Hu et al., 2022) to determine the importance of components of the Transformer block. Recently, several algorithms have
been proposed to perform structural removal in a
neural network efficiently but without incurring
the cost of updating the weights of LPMs. Hence,
most state-of-the-art approaches take the _training-_
_free_ path. LLMPruner (Ma et al., 2023) removes
network structures using gradient information and
recovers any accuracy drops utilizing parameterefficient fine-tuning (PEFT) techniques. ShortGPT
(Men et al., 2024) exploits block redundancy in
Transformer-based models and proposes a Block
Influence (BI) metric to decide which blocks to
prune. BI is a local metric based on the evolution of hidden stages in each block. BlockPruner
(Zhong et al., 2024) improves over ShortGPT by
proposing a global metric, e.g., the model’s perplexity, that is computed by masking the candidate
block and assessing its impact if removed. The
candidate that results in the most minor drop in
performance is removed from the model, which is
iteratively conducted until reaching the target pruning ratio. BlockPruner also increases the pruning
granularity by focusing on the multi-head attention
(MHA) and multilayer perceptrons (MLP), i.e., the
minimal residual blocks.


**5** **Conclusion**


Pruning large pre-trained models requires efficient
algorithms that consider their immense resource
requirements. This paper presents MultiPruner, an
efficient training-free structured pruning approach
that outperforms other pruning methods. Thanks to
their smaller size, the pruned models from MultiPruner accelerate inference and extend the range of
devices where machine learning practitioners can
deploy these models.




--- end of page=9 ---

## Limitations


Due to the complexity of foundation models, the
search spaces utilized in our experiments attempt
to obtain a good balance between efficiency and
efficacy. With a larger computing budget, a finergrained search might lead to even better results
and optimal balancing of the pruning dimensions.
Although accuracy is a good indicator of the performance of the pruned models compared to the
dense and baseline models, it does not capture the
intricacies of these large models. For instance, compressed models are more efficient and have accuracy similar to the base foundation model but might
behave differently under certain conditions. Our research focuses on improving the efficiency of large
models. Still, additional investigations are required
from the larger research community to understand
better the impact of the different methods for model
compression on the quality of the output from these
models.


## References


Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
[Shyamal Anadkat, et al. 2024. Gpt-4 technical report.](https://arxiv.org/abs/2303.08774)
_Preprint_, arXiv:2303.08774.


Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury
Zemlyanskiy, Federico Lebrón, and Sumit Sanghai.
2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. _arXiv_
_preprint arXiv:2305.13245_ .


Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari
do Nascimento, Torsten Hoefler, and James Hensman.
[2024. SliceGPT: Compress large language models](https://openreview.net/forum?id=vXxardq6db)
[by deleting rows and columns. In](https://openreview.net/forum?id=vXxardq6db) _The Twelfth Inter-_
_national Conference on Learning Representations_ .


Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,
Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei
Huang, et al. 2023. Qwen technical report. _arXiv_
_preprint arXiv:2309.16609_ .


Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng
Gao, and Yejin Choi. 2020. Piqa: Reasoning about
physical commonsense in natural language. In _Thirty-_
_Fourth AAAI Conference on Artificial Intelligence_ .


Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ
Altman, Simran Arora, Sydney von Arx, Michael S
Bernstein, Jeannette Bohg, Antoine Bosselut, Emma
[Brunskill, et al. 2021. On the opportunities and risks](https://crfm.stanford.edu/assets/report.pdf)

[of foundation models.](https://crfm.stanford.edu/assets/report.pdf) _ArXiv_ .


Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,
Ashish Sabharwal, Carissa Schoenick, and Oyvind



[Tafjord. 2018. Think you have solved question an-](https://api.semanticscholar.org/CorpusID:3922816)
[swering? try arc, the ai2 reasoning challenge.](https://api.semanticscholar.org/CorpusID:3922816) _ArXiv_,
abs/1803.05457.


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, et al. 2024. The llama 3 herd of models. _arXiv_
_preprint arXiv:2407.21783_ .


Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman,
Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li,
Kyle McDonell, Niklas Muennighoff, Chris Ociepa,
Jason Phang, Laria Reynolds, Hailey Schoelkopf,
Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou.
[2023. A framework for few-shot language model](https://doi.org/10.5281/zenodo.10256836)
[evaluation.](https://doi.org/10.5281/zenodo.10256836)


Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao,
[Michael W. Mahoney, and Kurt Keutzer. 2021. A](https://arxiv.org/abs/2103.13630)
[survey of quantization methods for efficient neural](https://arxiv.org/abs/2103.13630)
[network inference.](https://arxiv.org/abs/2103.13630) _CoRR_, abs/2103.13630.


Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.

[Distilling the knowledge in a neural network.](https://arxiv.org/abs/1503.02531)
_Preprint_, arXiv:1503.02531.


Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022. LoRA: Low-rank adaptation of
large language models. In _International Conference_

_on Learning Representations (ICLR)_ .


François Lagunas, Ella Charlaix, Victor Sanh, and
[Alexander Rush. 2021. Block pruning for faster trans-](https://doi.org/10.18653/v1/2021.emnlp-main.829)
[formers. In](https://doi.org/10.18653/v1/2021.emnlp-main.829) _Proceedings of the 2021 Conference on_
_Empirical Methods in Natural Language Process-_
_ing_, pages 10619–10629, Online and Punta Cana,
Dominican Republic. Association for Computational
Linguistics.


[Yann LeCun, John Denker, and Sara Solla. 1989. Op-](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)
[timal brain damage.](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) In _Advances in Neural In-_
_formation Processing Systems_, volume 2. MorganKaufmann.


Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023.

[LLM-pruner: On the structural pruning of large lan-](https://openreview.net/forum?id=J8Ajf9WfXP)
[guage models. In](https://openreview.net/forum?id=J8Ajf9WfXP) _Thirty-seventh Conference on Neu-_
_ral Information Processing Systems_ .


Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang,
Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng
[Chen. 2024. Shortgpt: Layers in large language mod-](https://arxiv.org/abs/2403.03853)
[els are more redundant than you expect.](https://arxiv.org/abs/2403.03853) _Preprint_,
arXiv:2403.03853.


Stephen Merity, Caiming Xiong, James Bradbury, and
Richard Socher. 2016. Pointer sentinel mixture models. _arXiv preprint arXiv:1609.07843_ .


J. Pablo Muñoz, Jinjie Yuan, and Nilesh Jain. 2024.

[SQFT: Low-cost model adaptation in low-precision](https://aclanthology.org/2024.findings-emnlp.749)




--- end of page=10 ---

[sparse foundation models. In](https://aclanthology.org/2024.findings-emnlp.749) _Findings of the Associ-_
_ation for Computational Linguistics: EMNLP 2024_,
pages 12817–12832, Miami, Florida, USA. Association for Computational Linguistics.


Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat[ula, and Yejin Choi. 2021. Winogrande: An adver-](https://doi.org/10.1145/3474381)
[sarial winograd schema challenge at scale.](https://doi.org/10.1145/3474381) _Commun._
_ACM_, 64(9):99–106.


Mohammad Samragh, Mehrdad Farajtabar, Sachin
Mehta, Raviteja Vemulapalli, Fartash Faghri, Devang Naik, Oncel Tuzel, and Mohammad Rastegari.
[2023. Weight subcloning: direct initialization of](https://arxiv.org/abs/2312.09299)
[transformers using larger pretrained ones.](https://arxiv.org/abs/2312.09299) _Preprint_,
arXiv:2312.09299.


Victor Sanh, Thomas Wolf, and Alexander M. Rush.
2020. Movement pruning: adaptive sparsity by finetuning. In _Proceedings of the 34th International Con-_
_ference on Neural Information Processing Systems_,
NIPS ’20, Red Hook, NY, USA. Curran Associates
Inc.


Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico
Kolter. 2023. A simple and effective pruning approach for large language models. _arXiv preprint_
_arXiv:2306.11695_ .


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint_
_arXiv:2307.09288_ .


Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang,
Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao,
[and Ping Luo. 2024. Besa: Pruning large language](https://arxiv.org/abs/2402.16880)
[models with blockwise parameter-efficient sparsity](https://arxiv.org/abs/2402.16880)
[allocation.](https://arxiv.org/abs/2402.16880) _Preprint_, arXiv:2402.16880.


Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang,
Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang,
Dong Yan, et al. 2023. Baichuan 2: Open large-scale
language models. _arXiv preprint arXiv:2309.10305_ .


An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, et al. 2024a. Qwen2
technical report. _arXiv preprint arXiv:2407.10671_ .


[Yifei Yang, Zouying Cao, and Hai Zhao. 2024b. Laco:](https://arxiv.org/abs/2402.11187)
[Large language model pruning via layer collapse.](https://arxiv.org/abs/2402.11187)
_Preprint_, arXiv:2402.11187.


Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali
Farhadi, and Yejin Choi. 2019. Hellaswag: Can a
machine really finish your sentence? In _Proceedings_

_of the 57th Annual Meeting of the Association for_
_Computational Linguistics_ .


Mingyang Zhang, Hao Chen, Chunhua Shen, Zhen
Yang, Linlin Ou, Xinyi Yu, and Bohan Zhuang. 2024.

[LoRAPrune: Pruning meets low-rank parameter-](https://openreview.net/forum?id=9KVT1e1qf7)
[efficient fine-tuning.](https://openreview.net/forum?id=9KVT1e1qf7)



Longguang Zhong, Fanqi Wan, Ruijun Chen, Xiaojun
[Quan, and Liangzhi Li. 2024. Blockpruner: Fine-](https://arxiv.org/abs/2406.10594)
[grained pruning for large language models.](https://arxiv.org/abs/2406.10594) _Preprint_,
arXiv:2406.10594.




--- end of page=11 ---

**Weight Importance Metric** **PPL (** _↓_ **)** **Avg. Score**


w/o Reordering 10.78 61.50
_L_ [1]  - Norm **9.33** **62.83**
Wanda (Sun et al., 2023) 9.43 61.84


Table 7: Ablation studies for weight reordering on
Llama2-7B + MultiPruner with pruning ratio of 22%.


**Inference Speedup**
**Batch Size** **Model** **Ratio**
**Prefill** **Decode**


Dense       - 1.00 _×_ 1.00 _×_
1
MultiPruner 22% **1.32** _×_ **1.28** _×_


Dense       - 1.00 _×_ 1.00 _×_
2
MultiPruner 22% **1.46** _×_ **1.29** _×_


Dense       - 1.00 _×_ 1.00 _×_
4
MultiPruner 22% **1.06** _×_ **1.29** _×_


Dense       - 1.00 _×_ 1.00 _×_
8
MultiPruner 22% **1.15** _×_ **1.28** _×_


Dense       - 1.00 _×_ 1.00 _×_
16
MultiPruner 22% **1.19** _×_ **1.28** _×_


Table 8: Inference benchmark results for Llama2-7B

(dense vs. pruned). Number of batches is 10. The
prompt length is 512. Number of new tokens is 16. The
evaluation was conducted on an Intel® Xeon® Plat
inum 8480+ processor with 56 cores, leveraging Intel®
Advanced Matrix Extensions (AMX) to accelerate infer
ence.


**A** **Impact of Weight Reordering**


Table 7 presents the results of ablation studies on
the impact of weight reordering before width pruning. The results indicate that applying weight re
- rdering (whether _L_ [1] - norm or Wanda (Sun et al.,
2023)) significantly improves performance, with
the _L_ [1] - norm performing better than the Wanda
strategy. These findings demonstrate that weight
reordering enhances the width pruning stage.


**B** **Inference Benchmark Results**


Table 8 shows the inference benchmark results for

different batch sizes.


**C** **Exploration of More Pruning Ratios**


Table 9 and Figure 6 show the results with pruning
ratios ranging from 1% to 21%.


**D** **Hyper-parameters**


Table 10 provides a comprehensive overview of the
hyper-parameters used in our experiments, ensuring reproducibility and clarity for the readers.
For most LLMs in our experiments, the ratio
weights we adopted are Block : MLP Channel :









|10|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|2<br>4<br>6<br>8<br>0||||||||
|2<br>4<br>6<br>8<br>0||||||||
|2<br>4<br>6<br>8<br>0||||||||
|2<br>4<br>6<br>8<br>0||||||||
|0<br>5<br>10<br>15<br>20<br>Pruning Ratio (%)<br>0<br>PPL<br>Avg. Accuracy||PPL<br>Avg. Accura|cy|||||
|0<br>5<br>10<br>15<br>20<br>Pruning Ratio (%)<br>0<br>PPL<br>Avg. Accuracy||||||||


### Figure 6

Caption: Visualization of Table 9 (performance on zeroshot downstream tasks with pruning ratios ranging from

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
1% to 21%). “PPL” refers to the perplexity in Wikitext2.
Average accuracy means the average accuracy across
five downstream tasks.


Attention Head = 44% : 52% : 4%. For a target
pruning ratio of 22%, this translates to:


  - Block Pruning Ratio Threshold ( _τ_ 1): 22% _×_
44% = 9.68%.


  - MLP Channel Pruning Ratio Threshold ( _τ_ 2):
9.68% + 22% _×_ 52% = 21.12%.


  - Attention Head Pruning Ratio Threshold ( _τ_ 3):

22.00%.




--- end of page=12 ---

**Model** **Method** **Ratio** **PPL (** _↓_ **)** **PIQA** **WinoG** **HellaS** **ARC-e** **ARC-c** **Avg. Score**


Dense 0% 5.47 79.05 69.06 75.99 74.54 46.16 68.96



**Llama2-7B**



MultiPruner



1% 5.78+0.31 78.94 68.82 75.92 74.07 45.99 68.75-0.21

2% 5.95+0.48 78.84 68.75 75.45 74.12 45.56 68.54-0.42

3% 5.96+0.49 78.51 69.22 75.33 74.20 45.31 68.51-0.45

4% 6.03+0.56 78.29 68.98 75.10 73.61 45.39 68.28-0.68

5% 6.14+0.67 78.18 69.06 74.95 73.65 45.48 68.26-0.70

6% 6.24+0.77 78.02 69.30 74.57 73.19 45.31 68.08-0.88

7% 6.33+0.86 77.86 69.46 74.32 73.44 44.62 67.94-1.02

8% 6.42+0.95 77.86 68.90 74.34 72.69 45.05 67.77-1.19

9% 6.50+1.03 77.64 68.59 74.27 69.36 43.34 66.64-2.32

10% 6.55+1.08 77.37 68.19 74.07 71.00 44.45 67.02-1.94

11% 6.63+1.16 77.64 67.72 73.60 70.75 42.58 66.46-2.50

12% 7.10+1.63 76.33 68.43 73.77 69.82 44.03 66.48-2.48

13% 7.22+1.75 76.93 68.90 73.35 67.17 42.06 65.68-3.28

14% 7.56+2.09 77.26 67.96 72.27 68.64 43.52 65.93-3.03

15% 7.66+2.19 76.77 67.40 71.82 68.10 42.24 65.26-3.70

16% 7.93+2.46 76.44 66.85 71.49 66.12 40.78 64.34-4.62

17% 8.10+2.63 75.84 66.61 71.36 65.40 40.27 63.90-5.06

18% 8.62+3.15 75.95 66.61 71.45 65.11 41.89 64.20-4.76

19% 8.77+3.30 75.14 64.56 69.49 63.51 40.61 62.66-6.30

20% 9.05+3.58 75.08 64.80 69.04 62.71 40.27 62.38-6.58

21% 9.16+3.69 74.81 64.80 69.35 65.53 41.47 63.19-5.77



Table 9: Performance on zero-shot downstream tasks with pruning ratios ranging from 1% to 21%. “Dense” denotes
the original (unpruned) models. “PPL” refers to the perplexity in Wikitext2.


**Hyper-parameter** **Value**


_Pruning Stage:_
Weight Reorder Importance Metric _L_ [1]   - norm
MLP Channel Group Size Hidden size // 4 (e.g., 1024 for Llama2-7B)
Attention Channel Group Size Head size (one head)
Calibration Dataset tatsu-lab/alpaca
Block / Channel / Head Importance Metric Perplexity (PPL)
Number of Calibration Samples for Block Pruning 256
Number of Calibration Samples for Width Pruning 128


_Recovery Stage:_
Dataset yahma/alpaca-cleaned
Epoch 2

Batch Size 32

Learning Rate 1e-4

LoRA Rank 16

LoRA Alpha 32
LoRA Dropout 0.05
LoRA Target Modules q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj.


Table 10: Hyper-parameters used in the experiments. The pruning stage’s hyperparameters apply to all LLMs used
in our experiment, while the hyperparameters for the recovery stage are specific to Llama2-7B. It is important to
note that the pruning ratio weights (distribution) may vary across different LLMs. For instance, in models employing
the GQA strategy, MultiPruner generally avoids pruning attention heads due to the smaller number of K and V
heads, which are highly sensitive. For more detailed information, please refer to the code repository.




--- end of page=13 ---
