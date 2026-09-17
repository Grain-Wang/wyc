---
id: "2023_LLM_Pruner"
title: "LLM-Pruner: On the Structural Pruning of Large Language Models"
authors: ["Xinyin Ma", "Gongfan Fang", "Xinchao Wang"]
year: 2023
venue: "NeurIPS 2023"
publication_status: "FORMALLY PUBLISHED"
category: "Structured Compression / Architecture Proxy"
source_pdf: "../reference_papers_origin/2023_LLM_Pruner.pdf"
paper_url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html"
pdf_url: "https://proceedings.neurips.cc/paper_files/paper/2023/file/44956951349095f74492a5471128a7e0-Paper-Conference.pdf"
code_url: "https://github.com/horseee/LLM-Pruner"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# LLM-Pruner: On the Structural Pruning of Large Language Models

**Authors:** Xinyin Ma, Gongfan Fang, Xinchao Wang

**Venue / Year:** NeurIPS 2023 (FORMALLY PUBLISHED)

**Category:** Structured Compression / Architecture Proxy

**Why this paper matters for low-cost post-training LLM NAS:** It defines coupled structural groups, gradient-based importance, and low-cost LoRA recovery for pretrained LLM architecture reduction.

**Primary record:** [https://proceedings.neurips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html)

**Local source:** [2023_LLM_Pruner.pdf](../reference_papers_origin/2023_LLM_Pruner.pdf)

## Full converted text

# **LLM-Pruner: On the Structural Pruning** **of Large Language Models**

**Xinyin Ma** **Gongfan Fang** **Xinchao Wang** _[∗]_
National University of Singapore
```
       maxinyin@u.nus.edu, gongfan@u.nus.edu, xinchao@nus.edu.sg

```

## Abstract


Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes
with a substantial model size, which presents significant challenges in both the
deployment, inference, and training stages. With LLM being a general-purpose
task solver, we explore its compression in a task-agnostic manner, which aims to
preserve the multi-task solving and language generation ability of the original LLM.
One challenge to achieving this is the enormous size of the training corpus of LLM,
which makes both data transfer and model post-training over-burdensome. Thus,
we tackle the compression of LLMs within the bound of two constraints: being taskagnostic and minimizing the reliance on the original training dataset. Our method,
named LLM-Pruner, adopts structural pruning that selectively removes non-critical
coupled structures based on gradient information, maximally preserving the majority of the LLM’s functionality. To this end, the performance of pruned models
can be efficiently recovered through tuning techniques, LoRA, in merely _3 hours_,
requiring only _50K_ data. We validate the LLM-Pruner on three LLMs, including
LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still
exhibit satisfactory capabilities in zero-shot classification and generation. The code
is available at: `[https://github.com/horseee/LLM-Pruner](https://github.com/horseee/LLM-Pruner)`


**1** **Introduction**


Recently, Large Language Models (LLMs) [39, 51, 50, 44, 64, 4, 74] have demonstrated remarkable
proficiency in language understanding and generation. With the increase in model size, they are
better equipped to handle complex tasks [3, 5, 58, 60] and even exhibit emergent abilities [57].
However, notwithstanding their impressive performance, LLMs pose challenges in deployment and
inference. Their extensive scale engenders substantial computational demands, and the multitude

- f parameters involved can induce long latencies and other related issues. Several techniques are
proposed to solve these problems, like model pruning [56, 61, 72, 21], knowledge distillation [46, 41,
47],quantization [1, 13] within the context of pre-trained language model (PLM).


While previous methods have effectively maintained model performance amidst parameter reduction,
they primarily target compression within specialized domains or for designated tasks in the context

- f task-specific compression. For instance, a PLM is fine-tuned on a particular dataset, such as one

- f the classification tasks in the GLUE benchmark [53], after which these models are distilled into
a smaller classification model [46, 18]. Although this paradigm could potentially be employed for
LLM compression, it compromises the LLM’s capacity as a versatile task solver, rendering it suited
to a single task exclusively.


_∗_ Corresponding author


37th Conference on Neural Information Processing Systems (NeurIPS 2023).




--- end of page=0 ---

**Evaluation**
QA NLI MRC





**Compression**











LLaMA-5.4B by LLM-Pruner



LLaMA-7B







### Figure 1

Caption: Illustration of LLM-Pruner. (i) Task-specific compression: the model is fine-tuned then

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
compressed on a specific task. (ii) TinyBERT: First distill the model on unlabeled corpus and then
fine-tune it on the specific task. (iii) LLM-Pruner: Task-agnostic compression within 3 hours.


Thus, we strive to compress the LLM in a new setting: to reduce the LLM’s size while preserving
its diverse capabilities as general-purpose task solvers, as depicted in Figure 1. This introduces the
task-agnostic compression of LLMs, which presents two key challenges:


- **The size of the training corpus of the LLM is enormous.** Previous compression methods heavily
depend on the training corpus. The LLM has escalated the corpus scale to 1 trillion tokens or more

[17, 51]. The extensive storage needs and protracted transmission times make the dataset difficult
to acquire. Furthermore, if the dataset is proprietary, acquisition of the training corpus verges on
impossibility, a situation encountered in [74, 39].

- **The unacceptably long duration for the post-training of the pruned LLM.** Existing methods
require a substantial amount of time for post-training the smaller model [55, 28]. For instance, the
general distillation in TinyBERT takes around 14 GPU days [20]. Even post-training a task-specific
compressed model of BERT demands around 33 hours [61, 22]. As the size of both the model and
corpus for LLMs increases rapidly, this step will invariably consume an even more extensive time.


To tackle the aforementioned challenges associated with the task-agnostic compression of LLMs, we
introduce a novel approach called LLM-Pruner. Since our goal is to compress LLMs with reduced
data dependency and expedited post-training, how to prune model with the minimal disruption to the

- rigin is crucial. To accomplish this, we propose a dependency detection algorithm that identifies all
the dependent structures within the model. Once the coupled structure is identified, we employ an
efficient importance estimation strategy to select the optimal group for pruning under the task-agnostic
setting, where the first-order information and an approximated hessian information is taken into
account. Finally, a rapid recovery stage is executed to post-train the pruned model with limited data.


**Contribution.** In this paper, we propose a novel framework, LLM-Pruner, for the task-agnostic
compression of the large language model. To the best of our knowledge, LLM-Pruner is the first
framework designed for structured pruning of LLMs. We conclude the advantages of the LLMPruner as (i) Task-agnostic compression, where the compressed language model retains its ability to
serve as a multi-task solver. (ii) Reduced demand for the original training corpus, where only 50k
publicly available samples are needed for compression, significantly reducing the budget for acquiring
the training data (iii) Quick compression, where the compression process ends up in three hours.
(iv) An automatic structural pruning framework, where all the dependent structures are grouped
without the need for any manual design. To evaluate the effectiveness of LLM-Pruner, we conduct
extensive experiments on three large language models: LLaMA-7B, Vicuna-7B, and ChatGLM-6B.
The compressed models are evaluated using nine datasets to assess both the generation quality and
the zero-shot classification performance of the pruned models. The experimental results demonstrate
that even with the removal of 20% of the parameters, the pruned model maintains 94.97% of the
performance of the original model.


2




--- end of page=1 ---

**2** **Related Work**


**Compression of Language Model.** Language models [9, 31, 25] have gained much attention and
increase the need to reduce the size of parameters and reduce the latency [23, 48]. To compress the
language model, previous works can be divided into several categories: network pruning [21, 63, 32,
15], knowledge distillation [46, 47, 40], quantization [68, 1, 71] and other techniques, like early exit

[62] or dynamic token reduction [69]. We focus on the pruning of the language models, especially
structural pruning [26]. Structural pruning removes the entire filter from the neural network, which is
more hardware friendly. There are several ways to remove the structure, such as l1-dependent pruning

[16, 72], first-order importance estimation [18], hessian-based estimation [21, 54] or the optimal
brain surgeon [24, 21]. As for the pruning unit in structural pruning, some works adopt the entire layer

[10] as the minimal unit, and others take the multi-head attention [52] or the feed-forward layers

[18, 36] as the basic structure to prune. CoFi [61] studies the pruning unit in different granularity.


**Efficient and Low Resource Compression.** With the growing size of neural network models, there
is an increasing demand for efficient and low-resource compression [67, 66, 30, 29, 65]. As for the
efficient compression, [22] accelerate the post-training by defining the reconstruction error as a linear
least squares problem. [13, 12] propose the layer-wise optimal brain surgeon. As for the constraint

- f availability of the training corpus, data-free pruning [45, 70] come up with several strategies to
prune the model by measuring neurons’ similarity. Besides, [34, 33, 42] proposes methods that
distill the model without reliance on the training corpus of the model. However, those methods are
too time-consuming, involving synthesizing samples by backpropagating the pre-trained models.


**3** **Methods**


In this section, we provide a detailed explanation of LLM-Pruner. Following the conventional model
compression pipeline[22], LLM-Pruner consists of three steps: **(1) Discovery Stage** (Section 3.1).
This step focuses on identifying groups of interdependent structures within LLMs. **(2) Estimation**
**Stage** (Section 3.2). Once the coupled structures are grouped, the second step entails estimating the
contribution of each group to the overall performance of the model and deciding which group to be
pruned. **(3) Recover Stage** (Section 3.3). This step involves fast post-training that alleviates potential
performance degradation caused by the removal of structures.


**3.1** **Discover All Coupled Structure in LLMs**


In light of the limited availability of data for post-training, it becomes imperative to prioritize the
removal of structures with minimal damage when compressing the model. This underscores the
dependency-based structural pruning, which ensures coupled structures are pruned in unison. We
provide an experiment in Section 4.3 to show the importance of dependency-based structural pruning
when compressing the large language model.


**Structure Dependency in LLMs.** Similar to [11], the pruning begins by building the dependency
for LLMs. Assume _Ni_ and _Nj_ are two neurons in the model, In( _Ni_ ) and Out( _Ni_ ) represents all the
neurons that point towards or point from _Ni_ . The dependency between structures can be defined as:


_Nj ∈_ Out( _Ni_ ) _∧_ Deg _[−]_ ( _Nj_ ) = 1 _⇒_ _Nj_ is dependent on _Ni_ (1)


where Deg _[−]_ ( _Nj_ ) represents the in-degree of neuron _Nj_ . Noting that this dependency is directional,
we can therefore correspondingly obtain another dependency:


_Ni ∈_ In( _Nj_ ) _∧_ Deg [+] ( _Ni_ ) = 1 _⇒_ _Ni_ is dependent on _Nj_ (2)


where Deg [+] ( _Ni_ ) represents the out-degree of neuron _Ni_ . The principle of dependency here is, if a
current neuron (e.g., _Ni_ ) depends solely on another neuron (e.g., _Nj_ ), and the neuron _Nj_ is subjected
to pruning, it follows that the neuron _Ni_ must also undergo pruning. We provide a detailed case in
Appendix A.


**Trigger the Dependency Graph.** By having the definition of dependency, the coupled structures
in the LLM can be analyzed automatically. Considering any neuron within the LLM as the initial


3




--- end of page=2 ---

Group Type A: MLP


_̸_



Group Type B: Multi-head Attention


_̸_



Group Type C: Channel-wise Grouping


_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



_̸_



L x


_̸_



_̸_



_̸_



Query Key Value


_̸_



_̸_



_̸_


|Col1|Col2|
|---|---|
|Norm:|Norm:|
|QKV<br>Head n<br>…<br>Head 1<br>MHA:<br>Embed-<br>ding:|QKV<br>Head n<br>…<br>Head 1<br>MHA:<br>Embed-<br>ding:|



### Figure 2

Caption: Illustration of the coupled structures in LLaMA. We simplify the neurons in each layer to

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
make the dependent group clear. The trigger neuron, marked as a circle with a bell, cause weights
with dependency pruned (dashed lines), which may propagate (red dashed lines) to coupled neurons
(dashed circles). A group can be triggered by a variety of trigger neurons. Taking Group Type B as
an example, the trigger for this group involves (i) the attention head, (ii) the output neuron in Query,
Key or Value, and (iii) the input neuron in the final output projection.


trigger, it possesses the capability to activate neurons that depend on it. Subsequently, these newly
triggered neurons can serve as the subsequent triggers to identify the dependency and activate their
respective dependent neurons. This iterative process continues until no new neurons are detected.
Those neurons then form a group for further pruning. Taking LLaMA as an example, by searching

- ver all the neurons as the initial trigger, we can locate all the coupled structures, as shown in Figure2.


Given the diversity in the structure of different LLMs, manual analysis and removal of coupled
structures in each LLM could be extremely time-consuming. However, by employing LLM-Pruner,
all coupled structures can be automatically identified and extracted.


**3.2** **Grouped Importance Estimation of Coupled Structure**


Till now, all coupled structures within the model are grouped. Weights within the same group should
be pruned simultaneously, as partial pruning not only increases parameter size but also introduces
misaligned intermediate representations. Therefore, we estimate the importance of the group as a
whole, as opposed to evaluating the importance of modules. Given the limited access to the training
dataset, we explore the use of public datasets or manually created samples as alternative resources.
Although the domains of these datasets may not perfectly align with the training set, they still provide
valuable information for assessing the importance.


**Vector-wise Importance.** Suppose that given a dataset _D_ = _{xi, yi}_ _[N]_ _i_ =1 [, where N is the number of]
samples. In our experiments, we set N equal to 10 and we use some public datasets as the source of
_D_ . A group (as previously defined as a set of coupled structures) can be defined as _G_ = _{Wi}_ _[M]_ _i_ =1 [,]
where M is the number of coupled structures in one group and _Wi_ is the weight for each structure.
While pruning, our goal is to remove the group that has the least impact on the model’s prediction,
which can be indicated by the deviation in the loss. Specially, to estimate the importance of _Wi_, the
change in loss can be formulated as [24]:


_̸_



_IWi_ = _|_ ∆ _L_ ( _D_ ) _|_ = _|LWi_ ( _D_ ) _−LWi_ =0( _D_ ) _|_ = _|_ _[∂][L]_ _∂W_ _[⊤]_ [(] _[D]_ _i_ [)] _Wi_

                      - ~~�~~                      -                      =0 _̸_




_−_ [1] _⊤HWi_ + _O_ - _∥Wi∥_ [3][�] _|_ (3)

2 _[W][i]_


_̸_



_̸_


where _H_ is the hessian matrix. Here, _L_ represents the next-token prediction loss. The first term is
typically neglected in prior work [24, 54, 12], as the model has already converged on the training
dataset, where _∂L_ _[⊤]_ _/∂Wi ≈_ 0. However, since _D_ here is not extracted from the original training data,
which means that _∂L_ _[⊤]_ _/∂Wi ̸≈_ 0. This presents a desirable property for determining the importance

- f _Wi_ by the gradient term under LLMs, since computation of the second term, the Hessian matrix,

- n the LLM is impractical with _O_ - _N_ [2][�] complexity.


4




--- end of page=3 ---

**Element-wise Importance.** The above can be considered as an estimate for the weight _Wi_ . We
can derive another measure of importance at a finer granularity, where each parameter within _Wi_ is
assessed for its significance:



_IW ki_ [=] _[ |]_ [∆] _[L]_ [(] _[D]_ [)] _[|]_ [ =] _[ |L][W][ k]_ _i_ [(] _[D]_ [)] _[ −L][W][ k]_ _i_ [=0][(] _[D]_ [)] _[|]_ [ =] _[ |]_ _[∂][L]_ [(] _[D][k]_ [)]



_i_ _[H][kk][W][ k]_ _i_ [+] _[ O]_    - _∥Wi_ _[k][∥]_ [3][�] _|_ (4)
2 _[W][ k]_




_[L]_ [(] _[D]_ [)]

_∂Wi_ _[k]_ _Wi_ _[k]_ _[−]_ [1] 2



Here, _k_ represents the k-th parameter in _Wi_ . The diagonal of the hessian _Hkk_ can be approximated
by the Fisher information matrix, and the importance can be defined as:



2
+ _O_  - _∥Wi_ _[k][∥]_ [3][�] _|_ (5)




_IW ki_ [=] _[ |L][W][ k]_ _i_ [(] _[D]_ [)] _[ −L][W][ k]_ _i_ [=0][(] _[D]_ [)] _[| ≈|]_ _[∂][L]_ [(] _[D][k]_ [)]



2




_[L]_ [(] _[D]_ [)]

_∂Wi_ _[k]_ _Wi_ _[k]_ _[−]_ 2 [1]



_N_


_j_ =1




- _∂∂WL_ ( _Di_ _[k]_ _j_ ) _Wi_ _[k]_



**Group Importance.** By utilizing either _IW ki_ [or] _[ I][W][i]_ [, we estimate the importance at the granu-]
larity of either a parameter or a vector of weight. Remembering that our goal is to estimate the
importance of _G_, we aggregate the importance scores in four ways: (i) Summation: _IG_ = [�] _[M]_ _i_ =1 _[I][W][i]_

- r _IG_ = [�] _[M]_ _i_ =1 - _k_ _[I]_ _W_ _[ k]_ _i_ [, (ii) Production:] _[ I][G]_ [ =][ �] _i_ _[M]_ =1 _[I][W]_ _i_ [or] _[ I][G]_ [ =][ �] _[M]_ _i_ =1 - _k_ _[I]_ _W_ _[ k]_ _i_ [, (iii) Max:]

_IG_ = max _[M]_ _i_ =1 _[I][W]_ _i_ [or] _[ I][G]_ [ = max] _[M]_ _i_ =1 - _k_ _[I]_ _W_ _[ k]_ _i_ [; (iv) Last-Only: Since deleting the last executing]

structure in a dependency group is equivalent to erasing all the computed results within that group,
we assign the importance of the last executing structure as the importance of the group: _IG_ = _IWl_ - r
_IG_ = [�] _k_ _[I]_ _W_ _[ k]_ _l_ [, where] _[ l]_ [ is the last structure. After assessing the importance of each group, we rank]

the importance of each group and prune the groups with lower importance based on a predefined
pruning ratio.


**3.3** **Fast Recovery with Low-rank Approximation**


In order to expedite the model recovery process and improve its efficiency under limited data, it
is crucial to minimize the number of parameters that need optimization during the recovery phase.
To facilitate this, we employ the low-rank approximation, LoRA[19], to post-train the pruned
model. Each learnable weight matrix in the model, denoted as _W_, encompassing both pruned and
unpruned linear projection in the LLM, can be represented as _W_ . The update value ∆ _W_ for _W_
can be decomposed as ∆ _W_ = _PQ ∈_ R _[d][−][×][d]_ [+], where _P ∈_ R _[d][−][×][d]_ and _Q ∈_ R _[d][×][d]_ [+] . The forward
computation can now be expressed as:


_f_ ( _x_ ) = ( _W_ + ∆ _W_ ) _X_ + _b_ = ( _WX_ + _b_ ) + ( _PQ_ ) _X_ (6)


where _b_ is the bias in the dense layer. Only training _P_ and _Q_ reduces the overall training complexity,
reducing the need for large-scale training data. Besides, the extra parameters _P_ and _Q_ can be
reparameterized into ∆ _W_, which would not cause extra parameters in the final compressed model.


**4** **Experiments**


**4.1** **Experimental Settings**


**Foundation Large Language Model.** To showcase the effectiveness and versatility of LLM-Pruner,
we test it over three open-source large language models with two kinds of structure: LLaMA-7B [51],
Vicuna-7B [4] [2] and ChatGLM-6B [74].


**Evaluation and Datasets.** To assess the performance of the model in the task-agnostic setting,
we follow LLaMA’s evaluation to perform zero-shot task classification on common sense reasoning
datasets: BoolQ [6], PIQA [2], HellaSwag [73], WinoGrande [43], ARC-easy [7], ARC-challenge [7]
and OpenbookQA [38]. Follow [14], the model ranks the choices in the multiple choice tasks or
generates the answer in the open-ended generation [3] . Additionally, we complement our evaluation
with a zero-shot perplexity (PPL) analysis on WikiText2 [37] and PTB [35].


2https://huggingface.co/lmsys/vicuna-7b-delta-v0
3https://github.com/EleutherAI/lm-evaluation-harness


5




--- end of page=4 ---

Table 1: Zero-shot performance of the compressed LLaMA-7B. The average is calculated among
seven classification datasets. ‘Underline’ indicates the best pruning-only performance, while ‘bold’
represents the overall best performance with the same pruning ratio, considering both pruning and
post-training. The ‘Channel’ strategy only prunes the dependent group of Type C, while all other
methods employ the ‘Block’ strategy to prune dependent groups in both Type A and Type B. Since

[51] did not provide its prompt, the evaluation of the result with _[⋆]_ is performed under different






|prompts, which is lower than the official results.|Col2|Col3|Col4|
|---|---|---|---|
|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|
|Ratio = 0%<br>LLaMA-7B[51]<br>LLaMA-7B_⋆_|-<br>-<br>12.62<br>22.14|76.5<br>79.8<br>76.1<br>70.1<br>72.8<br>47.6<br>57.2<br>73.18<br>78.35<br>72.99<br>67.01<br>67.45<br>41.38<br>42.40|68.59<br>63.25|
|Ratio = 20%<br>w/o tune<br>L2<br>582.41<br>1022.17<br>59.66<br>58.00<br>37.04<br>52.41<br>33.12<br>28.58<br>29.80<br>42.65<br>Random<br>27.51<br>43.19<br>61.83<br>71.33<br>56.26<br>54.46<br>57.07<br>32.85<br>35.00<br>52.69<br>Channel<br>74.63<br>153.75<br>62.75<br>62.73<br>41.40<br>51.07<br>41.38<br>27.90<br>30.40<br>45.38<br>Vector<br>22.28<br>41.78<br>61.44<br>71.71<br>57.27<br>54.22<br>55.77<br>33.96<br>38.40<br>53.25<br>Element2<br>19.77<br>36.66<br>59.39<br>75.57<br>65.34<br>61.33<br>59.18<br>37.12<br>39.80<br>56.82<br>Element1<br>19.09<br>34.21<br>57.06<br>75.68<br>66.80<br>59.83<br>60.94<br>36.52<br>**40.00**<br>56.69|582.41<br>1022.17<br>27.51<br>43.19|59.66<br>58.00<br>37.04<br>52.41<br>33.12<br>28.58<br>29.80<br>61.83<br>71.33<br>56.26<br>54.46<br>57.07<br>32.85<br>35.00|42.65<br>52.69|
|Ratio = 20%<br>w/o tune<br>L2<br>582.41<br>1022.17<br>59.66<br>58.00<br>37.04<br>52.41<br>33.12<br>28.58<br>29.80<br>42.65<br>Random<br>27.51<br>43.19<br>61.83<br>71.33<br>56.26<br>54.46<br>57.07<br>32.85<br>35.00<br>52.69<br>Channel<br>74.63<br>153.75<br>62.75<br>62.73<br>41.40<br>51.07<br>41.38<br>27.90<br>30.40<br>45.38<br>Vector<br>22.28<br>41.78<br>61.44<br>71.71<br>57.27<br>54.22<br>55.77<br>33.96<br>38.40<br>53.25<br>Element2<br>19.77<br>36.66<br>59.39<br>75.57<br>65.34<br>61.33<br>59.18<br>37.12<br>39.80<br>56.82<br>Element1<br>19.09<br>34.21<br>57.06<br>75.68<br>66.80<br>59.83<br>60.94<br>36.52<br>**40.00**<br>56.69|74.63<br>153.75|62.75<br>62.73<br>41.40<br>51.07<br>41.38<br>27.90<br>30.40|45.38|
|Ratio = 20%<br>w/o tune<br>L2<br>582.41<br>1022.17<br>59.66<br>58.00<br>37.04<br>52.41<br>33.12<br>28.58<br>29.80<br>42.65<br>Random<br>27.51<br>43.19<br>61.83<br>71.33<br>56.26<br>54.46<br>57.07<br>32.85<br>35.00<br>52.69<br>Channel<br>74.63<br>153.75<br>62.75<br>62.73<br>41.40<br>51.07<br>41.38<br>27.90<br>30.40<br>45.38<br>Vector<br>22.28<br>41.78<br>61.44<br>71.71<br>57.27<br>54.22<br>55.77<br>33.96<br>38.40<br>53.25<br>Element2<br>19.77<br>36.66<br>59.39<br>75.57<br>65.34<br>61.33<br>59.18<br>37.12<br>39.80<br>56.82<br>Element1<br>19.09<br>34.21<br>57.06<br>75.68<br>66.80<br>59.83<br>60.94<br>36.52<br>**40.00**<br>56.69|22.28<br>41.78<br>19.77<br>36.66<br>19.09<br>34.21|61.44<br>71.71<br>57.27<br>54.22<br>55.77<br>33.96<br>38.40<br>59.39<br>75.57<br>65.34<br>61.33<br>59.18<br>37.12<br>39.80<br>57.06<br>75.68<br>66.80<br>59.83<br>60.94<br>36.52<br>**40.00**|53.25<br>56.82<br>56.69|
|Ratio = 20%<br>w/ tune<br>Channel<br>22.02<br>38.67<br>59.08<br>73.39<br>64.02<br>60.54<br>57.95<br>35.58<br>38.40<br>55.57<br>Vector<br>18.84<br>33.05<br>65.75<br>74.70<br>64.52<br>59.35<br>60.65<br>36.26<br>39.40<br>57.23<br>Element2<br>**17.37**<br>30.39<br>**69.54**<br>76.44<br>68.11<br>**65.11**<br>63.43<br>**37.88**<br>**40.00**<br>**60.07**<br>Element1<br>17.58<br>**30.11**<br>64.62<br>**77.20**<br>**68.80**<br>63.14<br>**64.31**<br>36.77<br>39.80<br>59.23|Ratio = 20%<br>w/ tune<br>Channel<br>22.02<br>38.67<br>59.08<br>73.39<br>64.02<br>60.54<br>57.95<br>35.58<br>38.40<br>55.57<br>Vector<br>18.84<br>33.05<br>65.75<br>74.70<br>64.52<br>59.35<br>60.65<br>36.26<br>39.40<br>57.23<br>Element2<br>**17.37**<br>30.39<br>**69.54**<br>76.44<br>68.11<br>**65.11**<br>63.43<br>**37.88**<br>**40.00**<br>**60.07**<br>Element1<br>17.58<br>**30.11**<br>64.62<br>**77.20**<br>**68.80**<br>63.14<br>**64.31**<br>36.77<br>39.80<br>59.23|Ratio = 20%<br>w/ tune<br>Channel<br>22.02<br>38.67<br>59.08<br>73.39<br>64.02<br>60.54<br>57.95<br>35.58<br>38.40<br>55.57<br>Vector<br>18.84<br>33.05<br>65.75<br>74.70<br>64.52<br>59.35<br>60.65<br>36.26<br>39.40<br>57.23<br>Element2<br>**17.37**<br>30.39<br>**69.54**<br>76.44<br>68.11<br>**65.11**<br>63.43<br>**37.88**<br>**40.00**<br>**60.07**<br>Element1<br>17.58<br>**30.11**<br>64.62<br>**77.20**<br>**68.80**<br>63.14<br>**64.31**<br>36.77<br>39.80<br>59.23|Ratio = 20%<br>w/ tune<br>Channel<br>22.02<br>38.67<br>59.08<br>73.39<br>64.02<br>60.54<br>57.95<br>35.58<br>38.40<br>55.57<br>Vector<br>18.84<br>33.05<br>65.75<br>74.70<br>64.52<br>59.35<br>60.65<br>36.26<br>39.40<br>57.23<br>Element2<br>**17.37**<br>30.39<br>**69.54**<br>76.44<br>68.11<br>**65.11**<br>63.43<br>**37.88**<br>**40.00**<br>**60.07**<br>Element1<br>17.58<br>**30.11**<br>64.62<br>**77.20**<br>**68.80**<br>63.14<br>**64.31**<br>36.77<br>39.80<br>59.23|
|Ratio = 20%<br>w/ tune<br>Channel<br>22.02<br>38.67<br>59.08<br>73.39<br>64.02<br>60.54<br>57.95<br>35.58<br>38.40<br>55.57<br>Vector<br>18.84<br>33.05<br>65.75<br>74.70<br>64.52<br>59.35<br>60.65<br>36.26<br>39.40<br>57.23<br>Element2<br>**17.37**<br>30.39<br>**69.54**<br>76.44<br>68.11<br>**65.11**<br>63.43<br>**37.88**<br>**40.00**<br>**60.07**<br>Element1<br>17.58<br>**30.11**<br>64.62<br>**77.20**<br>**68.80**<br>63.14<br>**64.31**<br>36.77<br>39.80<br>59.23|18.84<br>33.05<br>**17.37**<br>30.39<br>17.58<br>**30.11**|65.75<br>74.70<br>64.52<br>59.35<br>60.65<br>36.26<br>39.40<br>**69.54**<br>76.44<br>68.11<br>**65.11**<br>63.43<br>**37.88**<br>**40.00**<br>64.62<br>**77.20**<br>**68.80**<br>63.14<br>**64.31**<br>36.77<br>39.80|57.23<br>**60.07**<br>59.23|



Table 2: Zero-shot performance of the compressed LLaMA-13B. Here we adopt Element [1] as the






|importance estimation for ‘Channel‘ and ‘Block’.|Col2|Col3|Col4|
|---|---|---|---|
|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>Method<br>WikiText2_↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|
|Ratio = 0%<br>LLaMA-13B_⋆_|11.58<br>20.24|68.47<br>78.89<br>76.24<br>70.09<br>74.58<br>44.54<br>42.00|64.97|
|Ratio = 20%<br>w/o tune<br>L2<br>Random<br>Channel<br>Block|61.15<br>91.43<br>19.24<br>31.84<br>49.03<br>106.48<br>16.01<br>29.28|61.50<br>67.57<br>52.90<br>57.54<br>50.13<br>31.14<br>36.80<br>63.33<br>73.18<br>63.54<br>60.85<br>64.44<br>36.26<br>38.00<br>62.39<br>66.87<br>49.17<br>58.96<br>49.62<br>31.83<br>33.20<br>67.68<br>77.15<br>73.41<br>65.11<br>68.35<br>38.40<br>42.40|51.08<br>57.09<br>50.29<br>61.79|
|Ratio = 20%<br>w/ tune<br>L2<br>20.97<br>38.05<br>**73.24**<br>76.77<br>71.86<br>64.64<br>67.59<br>39.93<br>40.80<br>62.12<br>Random<br>16.84<br>31.98<br>64.19<br>76.06<br>68.89<br>63.30<br>66.88<br>38.31<br>40.80<br>59.78<br>Channel<br>17.58<br>29.76<br>69.20<br>76.55<br>68.89<br>66.38<br>62.08<br>38.99<br>39.60<br>60.24<br>Block<br>**15.18**<br>**28.08**<br>70.31<br>**77.91**<br>**75.16**<br>**67.88**<br>**71.09**<br>**42.41**<br>**43.40**<br>**64.02**|Ratio = 20%<br>w/ tune<br>L2<br>20.97<br>38.05<br>**73.24**<br>76.77<br>71.86<br>64.64<br>67.59<br>39.93<br>40.80<br>62.12<br>Random<br>16.84<br>31.98<br>64.19<br>76.06<br>68.89<br>63.30<br>66.88<br>38.31<br>40.80<br>59.78<br>Channel<br>17.58<br>29.76<br>69.20<br>76.55<br>68.89<br>66.38<br>62.08<br>38.99<br>39.60<br>60.24<br>Block<br>**15.18**<br>**28.08**<br>70.31<br>**77.91**<br>**75.16**<br>**67.88**<br>**71.09**<br>**42.41**<br>**43.40**<br>**64.02**|Ratio = 20%<br>w/ tune<br>L2<br>20.97<br>38.05<br>**73.24**<br>76.77<br>71.86<br>64.64<br>67.59<br>39.93<br>40.80<br>62.12<br>Random<br>16.84<br>31.98<br>64.19<br>76.06<br>68.89<br>63.30<br>66.88<br>38.31<br>40.80<br>59.78<br>Channel<br>17.58<br>29.76<br>69.20<br>76.55<br>68.89<br>66.38<br>62.08<br>38.99<br>39.60<br>60.24<br>Block<br>**15.18**<br>**28.08**<br>70.31<br>**77.91**<br>**75.16**<br>**67.88**<br>**71.09**<br>**42.41**<br>**43.40**<br>**64.02**|Ratio = 20%<br>w/ tune<br>L2<br>20.97<br>38.05<br>**73.24**<br>76.77<br>71.86<br>64.64<br>67.59<br>39.93<br>40.80<br>62.12<br>Random<br>16.84<br>31.98<br>64.19<br>76.06<br>68.89<br>63.30<br>66.88<br>38.31<br>40.80<br>59.78<br>Channel<br>17.58<br>29.76<br>69.20<br>76.55<br>68.89<br>66.38<br>62.08<br>38.99<br>39.60<br>60.24<br>Block<br>**15.18**<br>**28.08**<br>70.31<br>**77.91**<br>**75.16**<br>**67.88**<br>**71.09**<br>**42.41**<br>**43.40**<br>**64.02**|



**Implementation Details.** In the model pruning process, we use 10 randomly selected samples
from Bookcorpus [75], each truncated to a sequence length of 128, as the calibration samples for
establishing dependency and calculating the gradient for both LLaMA and Vicuna. For ChatGLM,
we select 10 random samples from DailyDialog [27]. During the recovery phase, we utilize the
cleaned version of Alpaca [49], which comprises approximately 50k samples. Remarkably, tuning
these samples requires merely 3 hours on a single GPU with only 2 epochs. More hyper-parameters

- f pruning and training can be found in Appendix B.



**Statistics of the Compressed**

Table 3: Statistics of the base model and the compressed model.

**Model.** Table 3 presents the

ation is conducted using the infer
|Model|Strategy Ratio|#Params #MACs Memory Latency|
|---|---|---|
|LLaMA-7B<br>Vicuna-7B|-<br>-<br>Channel<br>20%<br>Block<br>20%<br>Channel<br>50%<br>Block<br>50%|6.74B<br>424.02G<br>12884.5MiB<br>69.32s<br>5.39B<br>339.36G<br>10363.6MiB<br>61.50s<br>5.42B<br>339.60G<br>10375.5MiB<br>58.55s<br>3.37B<br>212.58G<br>6556.3MiB<br>40.11s<br>3.35B<br>206.59G<br>6533.9MiB<br>37.54s|

ence mode, where the model is fed a sentence consisting of 64 tokens. The latency is tested under
the test set of WikiText2 on a single A5000. Here, the ‘Block’ strategy implies that the pruned
unit in the model consists of Group Type A and Group Type B as illustrated in Figure 2, whereas
‘Channel’ indicates that the unit to be pruned is Group Type C. We delve into an analysis of these two
choices in Section 4.2(Channel Strategy vs. Block Strategy). The pruning ratio stated here denotes
the approximate ratio of parameters to be pruned since the number of parameters within each pruned
structure does not perfectly match the total number of pruned parameters.



Table 3: Statistics of the base model and the compressed model.









**4.2** **Zero-shot Performance**


Table 1,2,4 and 5 shows the zero-shot performance of the pruned model. Based on the evaluation
conducted on LLaMA, employing a 20% parameter reduction without post-training, the pruned


6




--- end of page=5 ---

|Table 4: Zero-shot performance of the compressed Vicuna-7B|Col2|Col3|Col4|
|---|---|---|---|
|Pruned Model<br>Method<br>WikiText2_ ↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruned Model<br>Method<br>WikiText2_ ↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruned Model<br>Method<br>WikiText2_ ↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruned Model<br>Method<br>WikiText2_ ↓_<br>PTB_↓_<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|
|Ratio = 0%<br>Vicuna-7B|16.11<br>61.37|76.57<br>77.75<br>70.64<br>67.40<br>65.11<br>41.21<br>40.80|62.78|
|Ratio = 20%<br>w/o tune<br>l2<br>3539.98<br>5882.21<br>55.90<br>56.15<br>32.37<br>51.85<br>30.01<br>28.41<br>28.20<br>40.41<br>random<br>34.63<br>112.44<br>61.47<br>70.89<br>54.67<br>56.27<br>55.60<br>31.74<br>34.60<br>52.18<br>Channel<br>71.75<br>198.88<br>51.77<br>63.93<br>42.58<br>55.17<br>43.94<br>29.27<br>33.40<br>45.72<br>Vector<br>27.03<br>92.51<br>62.17<br>71.44<br>55.80<br>53.43<br>55.77<br>33.28<br>37.80<br>52.81<br>Element2<br>24.70<br>94.34<br>62.87<br>75.41<br>64.00<br>58.41<br>60.98<br>37.12<br>39.00<br>56.83<br>Element1<br>25.74<br>92.88<br>61.70<br>75.30<br>63.75<br>56.20<br>63.22<br>36.60<br>37.00<br>56.25|3539.98<br>5882.21<br>34.63<br>112.44|55.90<br>56.15<br>32.37<br>51.85<br>30.01<br>28.41<br>28.20<br>61.47<br>70.89<br>54.67<br>56.27<br>55.60<br>31.74<br>34.60|40.41<br>52.18|
|Ratio = 20%<br>w/o tune<br>l2<br>3539.98<br>5882.21<br>55.90<br>56.15<br>32.37<br>51.85<br>30.01<br>28.41<br>28.20<br>40.41<br>random<br>34.63<br>112.44<br>61.47<br>70.89<br>54.67<br>56.27<br>55.60<br>31.74<br>34.60<br>52.18<br>Channel<br>71.75<br>198.88<br>51.77<br>63.93<br>42.58<br>55.17<br>43.94<br>29.27<br>33.40<br>45.72<br>Vector<br>27.03<br>92.51<br>62.17<br>71.44<br>55.80<br>53.43<br>55.77<br>33.28<br>37.80<br>52.81<br>Element2<br>24.70<br>94.34<br>62.87<br>75.41<br>64.00<br>58.41<br>60.98<br>37.12<br>39.00<br>56.83<br>Element1<br>25.74<br>92.88<br>61.70<br>75.30<br>63.75<br>56.20<br>63.22<br>36.60<br>37.00<br>56.25|71.75<br>198.88|51.77<br>63.93<br>42.58<br>55.17<br>43.94<br>29.27<br>33.40|45.72|
|Ratio = 20%<br>w/o tune<br>l2<br>3539.98<br>5882.21<br>55.90<br>56.15<br>32.37<br>51.85<br>30.01<br>28.41<br>28.20<br>40.41<br>random<br>34.63<br>112.44<br>61.47<br>70.89<br>54.67<br>56.27<br>55.60<br>31.74<br>34.60<br>52.18<br>Channel<br>71.75<br>198.88<br>51.77<br>63.93<br>42.58<br>55.17<br>43.94<br>29.27<br>33.40<br>45.72<br>Vector<br>27.03<br>92.51<br>62.17<br>71.44<br>55.80<br>53.43<br>55.77<br>33.28<br>37.80<br>52.81<br>Element2<br>24.70<br>94.34<br>62.87<br>75.41<br>64.00<br>58.41<br>60.98<br>37.12<br>39.00<br>56.83<br>Element1<br>25.74<br>92.88<br>61.70<br>75.30<br>63.75<br>56.20<br>63.22<br>36.60<br>37.00<br>56.25|27.03<br>92.51<br>24.70<br>94.34<br>25.74<br>92.88|62.17<br>71.44<br>55.80<br>53.43<br>55.77<br>33.28<br>37.80<br>62.87<br>75.41<br>64.00<br>58.41<br>60.98<br>37.12<br>39.00<br>61.70<br>75.30<br>63.75<br>56.20<br>63.22<br>36.60<br>37.00|52.81<br>56.83<br>56.25|
|Ratio = 20%<br>w/ tune<br>Vector<br>19.94<br>**74.66**<br>63.15<br>74.59<br>61.95<br>60.30<br>60.48<br>36.60<br>**39.40**<br>56.64<br>Element2<br>**18.97**<br>76.78<br>60.40<br>75.63<br>**65.45**<br>**63.22**<br>**63.05**<br>**37.71**<br>39.00<br>**57.78**<br>Element1<br>19.69<br>78.25<br>**63.33**<br>**76.17**<br>65.13<br>60.22<br>62.84<br>37.12<br>39.20<br>57.71|Ratio = 20%<br>w/ tune<br>Vector<br>19.94<br>**74.66**<br>63.15<br>74.59<br>61.95<br>60.30<br>60.48<br>36.60<br>**39.40**<br>56.64<br>Element2<br>**18.97**<br>76.78<br>60.40<br>75.63<br>**65.45**<br>**63.22**<br>**63.05**<br>**37.71**<br>39.00<br>**57.78**<br>Element1<br>19.69<br>78.25<br>**63.33**<br>**76.17**<br>65.13<br>60.22<br>62.84<br>37.12<br>39.20<br>57.71|Ratio = 20%<br>w/ tune<br>Vector<br>19.94<br>**74.66**<br>63.15<br>74.59<br>61.95<br>60.30<br>60.48<br>36.60<br>**39.40**<br>56.64<br>Element2<br>**18.97**<br>76.78<br>60.40<br>75.63<br>**65.45**<br>**63.22**<br>**63.05**<br>**37.71**<br>39.00<br>**57.78**<br>Element1<br>19.69<br>78.25<br>**63.33**<br>**76.17**<br>65.13<br>60.22<br>62.84<br>37.12<br>39.20<br>57.71|Ratio = 20%<br>w/ tune<br>Vector<br>19.94<br>**74.66**<br>63.15<br>74.59<br>61.95<br>60.30<br>60.48<br>36.60<br>**39.40**<br>56.64<br>Element2<br>**18.97**<br>76.78<br>60.40<br>75.63<br>**65.45**<br>**63.22**<br>**63.05**<br>**37.71**<br>39.00<br>**57.78**<br>Element1<br>19.69<br>78.25<br>**63.33**<br>**76.17**<br>65.13<br>60.22<br>62.84<br>37.12<br>39.20<br>57.71|



model manages to retain 89.8% of the performance exhibited by the unpruned model. Furthermore,
through the efficient post-training, the classification accuracy further improves to 60.07%, achieving
94.97% of the accuracy attained by the original model. This demonstration proves the feasibility

- f using LLM-Pruner to effectively compress the model, even without relying on training data,
and within a remarkably short period of time. Surprisingly, we discover that on most datasets, the
pruned model with 5.4B LLaMA even outperformed chatGLM-6B. This highlights the superiority

- f the LLM-Pruner: if a smaller model with a customized size is required, LLM-Pruner is more
cost-effective compared to retraining another model with a satisfying performance. However, with
50% parameters pruned, a large accuracy degradation is observed (see Appendix C.4). Compressing
LLMs under high compression rates still remains a large challenge.


The compression results of Vicuna-7B align with those of LLaMA, as pruning 20% of parameters on
Vicuna-7B maintains performance at 92.03% of the original model. We test a smaller pruning rate of
10% on chatGLM-7B, where the pruned model only experiences a marginal performance decrease

- f 0.89%, which can be recovered through post-training. Despite the pruned model outperforming
the uncompressed model, we don’t assert it is better than the original model. This is largely because
chatGLM-6B, a bilingual model, has limited English pre-training exposure. Post-training, however,
introduces it to more English corpus, albeit limited, improving its English comprehension.


**Ablation: Impact of Importance Estimation.** We conduct tests on all proposed importance
estimation techniques mentioned in Section 3.2. The results can be found in Table 1 and 4. Here,
_Element_ [n] represents the importance evaluation utilizing the n-th order term in Eq.5. _Vector_ represents
the result corresponding to Eq.3. Based on the results obtained from LLaMA-7B and Vicuna-7B,
pruning algorithms achieved the best average performance mostly by leveraging the second-order
derivatives for each parameter. Nonetheless, given that first-order derivatives are considerably more
efficient than second-order derivatives, though yielding slightly inferior results, we still vote for the
first-order term as a competitive method. Besides, the results on chatGLM-7B differed significantly
from these findings. The importance estimation on each parameter fails, performing even worse than
l2, while the importance estimation on the weight matrix reaches the best performance.



**Ablation: Channel Strategy vs. Block Strategy.**
From the results presented in Table 2, it is evident
that pruning ‘Channel’ significantly deteriorates
performance compared to pruning ‘Block’. This
discrepancy arises because the layers within the
stacked transformer do not evenly distribute their
importance. As shown in Figure 3, the first and
last layers have a profound impact on the model’s
performance, and pruning them results in more substantial performance degradation compared to other
layers. However, due to the uniform treatment of
the ‘Channel’ group across all layers, it becomes
inevitable to prune the first and last layers, leading
to a significant decline in performance.


7









|20<br>19<br>18 exty|Col2|Wikitext2 PTB<br>50<br>45<br>40<br>it|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|18<br>19<br>20<br> exty|18<br>19<br>20<br> exty||||||||
|18<br>19<br>20<br> exty|18<br>19<br>20<br> exty||||||||
|1<br>1<br>text erp|6<br>7||||||||
|1<br>1<br>text erp|6<br>7||||||||
|0<br>5<br>10<br><br>13<br>14<br>15<br>|4<br>5|4<br>5|||||||
|0<br>5<br>10<br><br>13<br>14<br>15<br>|4<br>5|4<br>5|||||||
|0<br>5<br>10<br><br>13<br>14<br>15<br>|4<br>5|4<br>5|||||||


### Figure 3

Caption: Layer sensitivity for Pruning: Removing Groups in only one layer.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.




--- end of page=6 ---

Table 5: Zero-shot Performance of the compressed ChatGLM-6B


Pruned Model Method PIQA HellaSwag WinoGrande ARC-e ARC-c OBQA Average



Ratio = 10%

w/o tune




|6795 4637 5233 4836 2995 3740<br>. . . . . .|4705<br>.|
|---|---|
|61.97<br>37.22<br>49.72<br>42.05<br>28.24<br>35.40<br>65.29<br>43.18<br>51.30<br>47.52<br>29.52<br>34.60<br>66.32<br>43.51<br>53.04<br>47.56<br>**30.72**<br>**35.80**<br>54.35<br>28.07<br>50.59<br>27.82<br>24.66<br>33.20|42.43<br>45.24<br>46.16<br>36.45|



w/ tune Vector **67.74** **46.35** **53.99** **51.01** 29.95 35.00 **47.34**








|Col1|Col2|Col3|
|---|---|---|
||||
||||



### Figure 4

Caption: The pruning results on LLaMA-7B (left) and Vicuna7B (right) with different pruning rates.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


**4.3** **More Analysis**



### Figure 5

Caption: Perplexity on zero-shot

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
datasets across varyhing steps.



**Impact of Different Pruning Rates.** We investigate the impact of pruning the LLM at various
pruning ratios in Figure 4. We compare our pruning results with the L2 strategy because L2 is also
a data-free pruning algorithm. It is observed in the experiment of LLaMA that when the pruning
ratio reaches approximately 20%, the magnitude-dependent algorithm experiences a rapid collapse,
leading to the loss of information. Conversely, by employing LLM-Pruner, we are able to increase
the pruning ratio to around 60% while achieving an equivalent perplexity level. Furthermore, in the
case of Vicuna-7B, removing 10% parameters results in a performance decline equivalent to that of
LLM-Pruner with 60%. The utilization of LLM-Pruner enables a significant increase in the number

- f model parameters that can be pruned, thereby substantially reducing computational overhead.


**Tuning on the External Dataset.** To tune the pruned model, we utilize the external dataset
Alpaca [49]. The evaluation curves of the pruned model on two zero-shot datasets during the posttraining process are depicted in Figure 5. The results demonstrate a rapid decrease in the perplexity of
the pruned model within 300 steps, followed by a gradual increase. We provide a more comprehensive
evaluation in Appendix C.3. It is important to note that if the model is trained for an excessive number

- f steps, it runs the risk of overfitting the external dataset, potentially compromising its performance
in other general-purpose tasks.


**Impact of Dependency-based Structured Pruning.** To study the importance of dependency-based
structural pruning, we conduct an experiment to disrupt dependencies within groups, where each
weight matrix _Wi_ is pruned solely based on the importance score estimated on itself. Table 6
presents the results demonstrating the impact of dependencies in structural pruning. In the absence

- f dependencies, the model nearly fails in the zero-shot generation and classification tasks. Even
with tuning, the model fails to recover, showing a substantial difference compared to the results in
dependency-based pruning.


**Impact of Different Aggregation Strategies.** We conduct tests on the aggregation algorithms
proposed in Section 3.2. Our experimental results unveil notable discrepancies in model performance
across different aggregation strategies, with particular emphasis on the ‘Last-only’ strategy. Among
the evaluated approaches, the ‘Max’ strategy attains the most favorable outcomes in terms of perplexity, signifying enhanced coherence and fluency in sentence generation. However, it is important to
note that the ‘Max’ strategy exhibits the poorest zero-shot classification results compared to all four
strategies. Conversely, the ‘Last-only’ strategy showcases superior classification performance but


8




--- end of page=7 ---

Table 6: Effect of the dependency-based structural pruning. Average represents the average
performance on 7 classification datasets.



Table 7: Impact of different aggregation strategies on group importance estimation. Experiments are performed on LLaMA-7B.


|Col1|Method|WikiText2↓ PTB↓ Average↑|
|---|---|---|
|w/o Tuning|w/o dependency<br>w/ dependency|68378.42<br>79942.47<br>38.32<br>19.09<br>34.21<br>56.69|
|w/ Tuning|w/o dependency<br>w/ dependency|13307.46<br>13548.08<br>38.10<br>17.58<br>30.11<br>59.23|


|Method|WikiText2↓ PTB↓|ARC-e↑ PIQA↑ OBQA↑|
|---|---|---|
|Summation<br>Max<br>Production<br>Last-only|66.13<br>164.25<br>62.59<br>144.38<br>77.63<br>192.88<br>130.00<br>170.88|40.70<br>63.49<br>34.80<br>39.60<br>63.71<br>34.60<br>37.84<br>62.08<br>35.00<br>41.92<br>64.75<br>35.20|



suffers from the poorest generation quality. In our experiments, we make a trade-off by selecting the
‘Summation’ strategy since it shows both good generalization quality and classification performance.



**Comparison with DistilBERT** We show the comparison results of DistilBERT and LLM-Pruner on LLaMA-7B in Table
8. LLM-Pruner outperforms DistilBERT by 4.24% on average
with even a smaller size. The reason lies in that LLM-Pruner
minimizes model disruption during pruning, whereas DistilBERT merely selects one layer out of two. As a result, the
model pruned by LLM-Pruner demands less data to recover its
performance compared with DistilBERT, consequently achieving superior performance.



Table 8: DistilBert vs. LLM-Pruner.
The average here means the average


|score on the above seven|datasets.|
|---|---|
|Pruning Ratio<br>#Param|Average|
|DistilBert<br>3.50B<br>LLM-Pruner<br>3.35B|44.64<br>48.88|



**Scratch Training vs. Pruning.** We compare LLM-Pruner with StableLM-3B [4] with a similar
parameter size. To ensure fairness, both models are fine-tuned on the Alpaca dataset. The experimental
results of these two models are shown in the Table 9. LLM-Pruner crafts lightweight LLMs with
low resources, and even can sometimes achieve better performance than LLMs from scratch training.
However, we also acknowledge that the LLaMA-3B obtained by LLM-Pruner will not always

- utperform other 3B models from scratch training, due to the huge gap in the size of training corpus.


Table 9: Scratch Training (StableLM-3B) vs. Pruning (LLaMA-3B by LLM-Pruner)


Pruning Ratio #Param Latency BoolQ PIQA HellaSwag WinoGrande ARC-e ARC-c OBQA Average


StableLM-3B 3.6B 31.69s 48.78 69.48 44.52 54.62 50.93 25.17 27.40 45.84

LLaMA-3B 3.6B 37.96s 61.41 70.08 51.01 55.01 46.80 30.38 37.40 50.30


**More Data for Recovery** Despite our primary experiments being conducted using 50k samples,
we remain convinced that the inclusion of additional data could substantially enhance the recovery
process, albeit at a considerably higher computational cost. Consequently, we conduct an experiment
aimed at model recovery with more data, employing a dataset comprising 2.59 million samples [59].
The results are detailed in Table 10. From the results, it is evident that the performance of the compressed model closely approximates that of the base model, exhibiting only a marginal performance
decrease of 0.89%.


Table 10: Model Recovery: 50k samples vs. 2.59M samples


Model #Samples BoolQ PIQA HellaSwag WinoGrande ARC-e ARC-c OBQA Average


LLaMA-7B  - 73.18 78.35 72.99 67.01 67.45 41.38 42.40 63.25
LLaMA-5.4B 50k [49] 64.62 77.20 68.80 63.14 64.31 36.77 39.80 59.23
LLaMA-5.4B 2.59M [59] 76.57 77.37 66.60 65.82 70.62 40.70 38.80 62.36


**Case Study.** We provide some examples of sentences generated by the model compressed using
LLM-Pruner in Table 11. We made efforts to ensure a minimal overlap between these generated
sentences and the information contained in the tuning corpus, which demonstrates that the information

- riginates from the original model rather than the tuning corpus. We provide additional examples
in the Appendix, including the generated sentences of the model without post-training. From the


4https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b


9




--- end of page=8 ---

Table 11: Some examples of generations obtained with the original model and the pruned model.


Model Original Model Pruned Model



LLaMa The universe is the entirety of space, time, matter, and energy
that exists. It includes all matter and energy, whether visible

        - r not, and in fact the whole universe is invisible. There are
many theories regarding its origin and ultimate fate...


Vicuna 10 steps to build an ios app: 1. Research and develop the idea.
2. Decide on the platform you want to use (iOS, Android,

       - r both). 3. Create a plan and budget. 4. Hire a team of
developers. 5. Design the user interface (UI). 6. Develop the
user interface (UI). 7. Test and iterate on the app. 8. Submit
the app to the app store. 9. Optimize the app for performance
and user experience (UX). 10. Market and promote the app.


ChatGLM Write a poetry about love and peace. In the midst of life’s
hustle and bustle, I find your love a soothing peace. It is a
feeling that cannot be conveyed, A connection that cannot
be distanced. With your smile, your eyes, your embrace, I
feel a sense of belonging that only you can give. It is a love
that lasts forever, A love that cannot be defeat...



The universe is the entirety of space, time, matter, and
energy that exists. The laws that make up physics — electrons orbiting atoms, magnets attracting other objects,
planets orbiting the Sun—have been consistent over ...


10 steps to build an ios app: 1. Plan your app: decide what
it should do, what it will look like, and what it will cost
to build it. 2. Hire a developer: A good iOS developer
will help you design the user interface, create the app, and
make sure it’s bug-free. 3. Create a user interface: This is
where the magic happens. Your app’s user interface will
be designed and mocked up at this point...


Write a poetry about love and peace. Love and peace, two
such wonderful things\ A pure and endless desire in my
heart\ And both of them, I must seek for\ A long, long
time, I know..\ Love, I know, is a feeling of being\ A
perfect partner, in every sense\ And peace, I need it, so
much, one day\ A long, long way, my heart will go..



cases in Table 11, it is evident that the sentences generated by the compressed model are comparable
to those produced by the original model. They exhibit fluency, relevance, and informativeness
regarding the given topic. Nevertheless, during our experiments, we observed that the pruned model’s
performance deviates from that of the original model, particularly when generating lengthy sentences.
Occasionally, it may generate sentences that are meaningless or contain repetitive tokens.


**5** **Conclusion**


In this paper, we propose LLM-Pruner, a structured pruning approach for large language models.
LLM-Pruner aims to compress sizable language models in a task-agnostic manner while minimizing
the dependency on the original training corpus and preserving the linguistic capabilities of LLMs.
LLM-Pruner accomplishes this by iteratively examining each neuron within the model as a trigger
for identifying dependency groups, thereby constructing the LLM’s dependency graph. Subsequently,
LLM-Pruner assesses the importance of these groups using both parameter-wise and weight-wise
estimation. Finally, we utilize LoRA for fast recovery and adjustment of the pruned model. We evaluate the efficacy of LLM-Pruner on three distinct models—LLaMA, Vicuna, and ChatGLM—utilizing
various zero-shot datasets. Our experimental results indicate that LLM-Pruner successfully prunes
the model, reducing computational burden while retaining its zero-shot capabilities. Nevertheless,
considerable performance degradation occurs when employing high pruning rates, such as the removal

- f 50% of LLaMA’s parameters, resulting in a substantial decline in model performance. Additionally,
we observe instances in which the model generates incoherent sentences. Addressing the challenges
associated with compressing LLMs at higher pruning rates remains a challenging task.


**Acknowledgment**


This project is supported by the Ministry of Education, Singapore, under its Academic Research Fund
Tier 2 (Award Number: MOE-T2EP20122-0006), and the National Research Foundation, Singapore,
under its Medium Sized Center for Advanced Robotics Technology Innovation.


10




--- end of page=9 ---

## References


[1] Haoli Bai, Wei Zhang, Lu Hou, Lifeng Shang, Jing Jin, Xin Jiang, Qun Liu, Michael Lyu, and Irwin King.
Binarybert: Pushing the limit of bert quantization. _arXiv preprint arXiv:2012.15701_, 2020.


[2] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about
physical commonsense in natural language. In _Thirty-Fourth AAAI Conference on Artificial Intelligence_,
2020.


[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
_Advances in neural information processing systems_, 33:1877–1901, 2020.


[4] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan
Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source
chatbot impressing gpt-4 with 90% chatgpt quality, March 2023. URL `[https://lmsys.org/blog/](https://lmsys.org/blog/2023-03-30-vicuna/)`
`[2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/)` .


[5] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts,
Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language
modeling with pathways. _arXiv preprint arXiv:2204.02311_, 2022.


[6] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In _Proceedings of_
_the 2019 Conference of the North American Chapter of the Association for Computational Linguistics:_
_Human Language Technologies, Volume 1 (Long and Short Papers)_, pages 2924–2936, Minneapolis,
Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL
`[https://aclanthology.org/N19-1300](https://aclanthology.org/N19-1300)` .


[7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
_arXiv:1803.05457v1_, 2018.


[8] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm. int8 (): 8-bit matrix multiplication
for transformers at scale. _arXiv preprint arXiv:2208.07339_, 2022.


[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_, 2018.


[10] Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured
dropout. _arXiv preprint arXiv:1909.11556_, 2019.


[11] Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and Xinchao Wang. Depgraph: Towards any
structural pruning, 2023.


[12] Elias Frantar and Dan Alistarh. Massive language models can be accurately pruned in one-shot. _arXiv_
_preprint arXiv:2301.00774_, 2023.


[13] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization
for generative pre-trained transformers. _arXiv preprint arXiv:2210.17323_, 2022.


[14] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding,
Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, et al. A framework for few-shot language model
evaluation. _Version v0. 0.1. Sept_, 2021.


[15] Fu-Ming Guo, Sijia Liu, Finlay S. Mungall, Xue Lin, and Yanzhi Wang. Reweighted proximal pruning
for large-scale language representation. _CoRR_, abs/1909.12486, 2019. URL `[http://arxiv.org/abs/](http://arxiv.org/abs/1909.12486)`
`[1909.12486](http://arxiv.org/abs/1909.12486)` .


[16] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections
for efficient neural network. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, _Advances in Neural Information Processing Systems_, volume 28. Curran Associates, Inc., 2015. URL `[https://proceedings.neurips.cc/paper_files/paper/2015/file/](https://proceedings.neurips.cc/paper_files/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf)`
`[ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf)` .


[17] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford,
Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal
large language models. _arXiv preprint arXiv:2203.15556_, 2022.


11




--- end of page=10 ---

[18] Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu. Dynabert: Dynamic bert with
adaptive width and depth. _Advances in Neural Information Processing Systems_, 33:9782–9793, 2020.


[19] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.


[20] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert:
Distilling bert for natural language understanding. In _Findings of the Association for Computational_
_Linguistics: EMNLP 2020_, pages 4163–4174, 2020.


[21] Eldar Kurtic, Daniel Campos, Tuan Nguyen, Elias Frantar, Mark Kurtz, Benjamin Fineran, Michael Goin,
and Dan Alistarh. The optimal bert surgeon: Scalable and accurate second-order pruning for large language
models. _arXiv preprint arXiv:2203.07259_, 2022.


[22] Woosuk Kwon, Sehoon Kim, Michael W Mahoney, Joseph Hassoun, Kurt Keutzer, and Amir Gholami. A
fast post-training pruning framework for transformers. _arXiv preprint arXiv:2204.09656_, 2022.


[23] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. _arXiv preprint arXiv:1909.11942_,
2019.


[24] Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. _Advances in neural information_
_processing systems_, 2, 1989.


[25] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves
Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language
generation, translation, and comprehension. _arXiv preprint arXiv:1910.13461_, 2019.


[26] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient
convnets. _arXiv preprint arXiv:1608.08710_, 2016.


[27] Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. Dailydialog: A manually
labelled multi-turn dialogue dataset. In _Proceedings of The 8th International Joint Conference on Natural_
_Language Processing (IJCNLP 2017)_, 2017.


[28] Chen Liang, Haoming Jiang, Zheng Li, Xianfeng Tang, Bin Yin, and Tuo Zhao. Homodistil: Homotopic
task-agnostic distillation of pre-trained transformers. _arXiv preprint arXiv:2302.09632_, 2023.


[29] Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, and Xinchao Wang. Dataset distillation via factorization.
In _Advances in Neural Information Processing Systems_, 2022.


[30] Songhua Liu, Jingwen Ye, Runpeng Yu, and Xinchao Wang. Slimmable dataset condensation. In _IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, 2023.


[31] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv_
_preprint arXiv:1907.11692_, 2019.


[32] Zejian Liu, Fanrong Li, Gang Li, and Jian Cheng. Ebert: Efficient bert inference with dynamic structured
pruning. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_, pages 4814–
4823, 2021.


[33] Xinyin Ma, Yongliang Shen, Gongfan Fang, Chen Chen, Chenghao Jia, and Weiming Lu. Adversarial
self-supervised data-free distillation for text classification. In _Proceedings of the 2020 Conference on_
_Empirical Methods in Natural Language Processing (EMNLP)_, pages 6182–6192, 2020.


[34] Xinyin Ma, Xinchao Wang, Gongfan Fang, Yongliang Shen, and Weiming Lu. Prompting to distill:
Boosting data-free knowledge distillation via reinforced prompt. _arXiv preprint arXiv:2205.07523_, 2022.


[35] Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated
corpus of English: The Penn Treebank. _Computational Linguistics_, 19(2):313–330, 1993. URL `[https:](https://www.aclweb.org/anthology/J93-2004)`
`[//www.aclweb.org/anthology/J93-2004](https://www.aclweb.org/anthology/J93-2004)` .


[36] JS McCarley, Rishav Chakravarti, and Avirup Sil. Structured pruning of a bert-based question answering
model. _arXiv preprint arXiv:1910.06360_, 2019.


[37] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models,
2016.


12




--- end of page=11 ---

[38] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity?
a new dataset for open book question answering. In _EMNLP_, 2018.


[39] OpenAI. Gpt-4 technical report, 2023.


[40] Haojie Pan, Chengyu Wang, Minghui Qiu, Yichang Zhang, Yaliang Li, and Jun Huang. Meta-kd: A meta
knowledge distillation framework for language model compression across domains. _CoRR_, abs/2012.01266,
2020. URL `[https://arxiv.org/abs/2012.01266](https://arxiv.org/abs/2012.01266)` .


[41] Haojie Pan, Chengyu Wang, Minghui Qiu, Yichang Zhang, Yaliang Li, and Jun Huang. Meta-kd: A
meta knowledge distillation framework for language model compression across domains. _arXiv preprint_
_arXiv:2012.01266_, 2020.


[42] Ahmad Rashid, Vasileios Lioutas, Abbas Ghaddar, and Mehdi Rezagholizadeh. Towards zero-shot
knowledge distillation for natural language processing, 2020.


[43] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial
winograd schema challenge at scale, 2019.


[44] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman
Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter

   - pen-access multilingual language model. _arXiv preprint arXiv:2211.05100_, 2022.


[45] Suraj Srinivas and R Venkatesh Babu. Data-free parameter pruning for deep neural networks. _arXiv_
_preprint arXiv:1507.06149_, 2015.


[46] Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for bert model compression.
_arXiv preprint arXiv:1908.09355_, 2019.


[47] Siqi Sun, Zhe Gan, Yuwei Fang, Yu Cheng, Shuohang Wang, and Jingjing Liu. Contrastive distillation on
intermediate representations for language model compression. In _Proceedings of the 2020 Conference_

_on Empirical Methods in Natural Language Processing (EMNLP)_, pages 498–508, Online, November
2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.36. URL `[https:](https://aclanthology.org/2020.emnlp-main.36)`
`[//aclanthology.org/2020.emnlp-main.36](https://aclanthology.org/2020.emnlp-main.36)` .


[48] Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. Mobilebert: a
compact task-agnostic bert for resource-limited devices. _arXiv preprint arXiv:2004.02984_, 2020.


[49] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang,
and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. `[https://github.](https://github.com/tatsu-lab/stanford_alpaca)`
`[com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)`, 2023.


[50] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng,
Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. _arXiv_
_preprint arXiv:2201.08239_, 2022.


[51] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation
language models. _arXiv preprint arXiv:2302.13971_, 2023.


[52] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head selfattention: Specialized heads do the heavy lifting, the rest can be pruned. In _Proceedings of the 57th Annual_
_Meeting of the Association for Computational Linguistics_, pages 5797–5808, 2019.


[53] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue:
A multi-task benchmark and analysis platform for natural language understanding. _arXiv preprint_
_arXiv:1804.07461_, 2018.


[54] Chaoqi Wang, Roger Grosse, Sanja Fidler, and Guodong Zhang. Eigendamage: Structured pruning in
the kronecker-factored eigenbasis. In _International conference on machine learning_, pages 6566–6575.
PMLR, 2019.


[55] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep self-attention
distillation for task-agnostic compression of pre-trained transformers. _Advances in Neural Information_
_Processing Systems_, 33:5776–5788, 2020.


[56] Ziheng Wang, Jeremy Wohlwend, and Tao Lei. Structured pruning of large language models. _arXiv_
_preprint arXiv:1910.04732_, 2019.


13




--- end of page=12 ---

[57] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama,
Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. _arXiv_
_preprint arXiv:2206.07682_, 2022.


[58] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain

   - f thought prompting elicits reasoning in large language models. _arXiv preprint arXiv:2201.11903_, 2022.


[59] Minghao Wu, Abdul Waheed, Chiyu Zhang, Muhammad Abdul-Mageed, and Alham Fikri Aji. Lamini-lm:
A diverse herd of distilled models from large-scale instructions, 2023.


[60] Yiquan Wu, Kun Kuang, Yating Zhang, Xiaozhong Liu, Changlong Sun, Jun Xiao, Yueting Zhuang, Luo
Si, and Fei Wu. De-biased court’s view generation with causality. In _Proceedings of the 2020 Conference_

_on Empirical Methods in Natural Language Processing (EMNLP)_, pages 763–780, 2020.


[61] Mengzhou Xia, Zexuan Zhong, and Danqi Chen. Structured pruning learns compact and accurate models.
_arXiv preprint arXiv:2204.00408_, 2022.


[62] Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. DeeBERT: Dynamic early exiting for
accelerating BERT inference. In _Proceedings of the 58th Annual Meeting of the Association for Computa-_
_tional Linguistics_, pages 2246–2251, Online, July 2020. Association for Computational Linguistics. doi:
10.18653/v1/2020.acl-main.204. URL `[https://aclanthology.org/2020.acl-main.204](https://aclanthology.org/2020.acl-main.204)` .


[63] Dongkuan Xu, Ian En-Hsu Yen, Jinxi Zhao, and Zhibin Xiao. Rethinking network pruning–under the
pre-train and fine-tune paradigm. In _Proceedings of the 2021 Conference of the North American Chapter of_
_the Association for Computational Linguistics: Human Language Technologies_, pages 2376–2382, 2021.


[64] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua,
and Colin Raffel. mt5: A massively multilingual pre-trained text-to-text transformer. _arXiv preprint_
_arXiv:2010.11934_, 2020.


[65] Xingyi Yang, Jingwen Ye, and Xinchao Wang. Factorizing knowledge in neural networks. In _European_
_Conference on Computer Vision_, 2022.


[66] Xingyi Yang, Daquan Zhou, Songhua Liu, Jingwen Ye, and Xinchao Wang. Deep model reassembly. In
_Advances in Neural Information Processing Systems_, 2022.


[67] Yiding Yang, Jiayan Qiu, Mingli Song, Dacheng Tao, and Xinchao Wang. Distilling knowledge from graph
convolutional networks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, 2020.


[68] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He.
Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. _Advances in_
_Neural Information Processing Systems_, 35:27168–27183, 2022.


[69] Deming Ye, Yankai Lin, Yufei Huang, and Maosong Sun. TR-BERT: Dynamic token reduction for
accelerating BERT inference. In _Proceedings of the 2021 Conference of the North American Chapter of_
_the Association for Computational Linguistics: Human Language Technologies_, pages 5798–5809, Online,
June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.463. URL
`[https://aclanthology.org/2021.naacl-main.463](https://aclanthology.org/2021.naacl-main.463)` .


[70] Edouard Yvinec, Arnaud Dapogny, Matthieu Cord, and Kevin Bailly. Red++: Data-free pruning of deep
neural networks via input splitting and output merging. _IEEE Transactions on Pattern Analysis and_
_Machine Intelligence_, 45(3):3664–3676, 2022.


[71] Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. Q8bert: Quantized 8bit bert. In _2019 Fifth_
_Workshop on Energy Efficient Machine Learning and Cognitive Computing-NeurIPS Edition (EMC2-NIPS)_,
pages 36–39. IEEE, 2019.


[72] Ofir Zafrir, Ariel Larey, Guy Boudoukh, Haihao Shen, and Moshe Wasserblat. Prune once for all: Sparse
pre-trained language models. _arXiv preprint arXiv:2111.05754_, 2021.


[73] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really
finish your sentence? In _Proceedings of the 57th Annual Meeting of the Association for Computational_
_Linguistics_, 2019.


[74] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi
Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. _arXiv preprint arXiv:2210.02414_,
2022.


[75] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja
Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading
books. In _The IEEE International Conference on Computer Vision (ICCV)_, December 2015.


14




--- end of page=13 ---

**A** **Detailed Explanations for the Dependency Rules**


**Group B: Multi-head Attention**







Node J







Node J





Node I





Node I









Node I







Case 1: Node J is dependent on Node I



Query Key Value


Case 2: Node I is dependent on Node J



Case 3: Node K is not dependent on Node J



### Figure 6

Caption: Illustrations of the two dependency rules. All the cases are extracted from the multi-head

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
attention module.


We provide a detailed explanation of the two dependency rules. It is important to note that these
dependency rules do not pertain solely to the forward computation. Instead, they represent directional
relationships that exist in both directions. For instance, removing a node in a subsequent layer may
also result in the pruning of a node in the preceding layer. Recall the two dependency rules as follows:


_Nj ∈_ Out( _Ni_ ) _∧_ Deg _[−]_ ( _Nj_ ) = 1 _⇒_ _Nj_ is dependent on _Ni_ (7)


_Ni ∈_ In( _Nj_ ) _∧_ Deg [+] ( _Ni_ ) = 1 _⇒_ _Ni_ is dependent on _Nj_ (8)


where _Ni_ and _Nj_ are two neurons. In( _Ni_ ) and Out( _Ni_ ) represents all the neurons that point towards

- r point from _Ni_ . Deg _[−]_ ( _Ni_ ) and Deg [+] ( _Ni_ ) represents the in-degree and out-degree of neuron _Ni_ .
### Figure 6

Caption: serves as an illustration of the two dependency rules:

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


- In case 1, Node I and Node J satisfy the rule stated in Eq.7. Consequently, Node J depends on
Node I. When Node I is pruned, it is necessary to prune Node J as well.


- In case 2, Node I and Node J satisfy the rule Eq.8. Thus, Node I is dependent on Node J. If Node J
is pruned, it becomes imperative to prune Node I as well.


- In case 3, Node J and Node K do not meet the requirement of Eq.7 due to the mismatch in
Deg _[−]_ ( _Nk_ ) _̸_ = 1. Thus, with Node J pruned, Node K would not be affected.


**B** **Implementation Details**


**B.1** **For Pruning**


**For the baseline** Given the lack of previous work on the structural pruning of Large Language
Models in a task-agnostic and low-resource setting, there is currently no existing baseline for our
model. To provide a comprehensive demonstration of the effectiveness of LLM-Pruner, we employ
two additional methods for evaluation, alongside the data-free pruning method. All of these methods
are built upon the dependent groups identified in Section 3.1:


- L2: We assess the importance of each group based on the magnitude of its weight matrix.


- Random: This method involves randomly selecting certain groups for pruning.


**For the ‘Block’ Group.** Based on the findings presented in Table 3, it is preferable to leave the
first three layers and the final layer unchanged, as modifying parameters in those layers significantly
impacts the model. Within each module, such as the MLP or the Multi-head Attention, the discovered
groups are pruned based on a pre-set ratio. For instance, in the MLP layer of LLaMA-7B, we
identified 11,008 groups, and with a 25% pruning ratio, the module would prune 2,752 groups. It
is worth noting that the pruning rate for the selected groups is higher than the pruning ratio for the
parameters, as certain layers (e.g., the embedding layer and excluded layers mentioned) retain their
parameters. When aiming for a parameter pruning ratio of 20%, we prune 25% from Layer 5 to Layer
30. Similarly, for a 50% parameter removal, we prune 60% of the groups from Layer 4 to Layer 30.


15




--- end of page=14 ---

**For the ‘Channel’ Group.** The Group ’Channel’ exhibits a resemblance to dimension pruning
in the model, targeting the pruning of certain dimensions. In the case of the Query, Key, and Value
projection in MHA, only the input dimension is pruned, while for the Output projection in MHA,

- nly the output dimension is pruned. It is important to note that the entire dependency is established
automatically, without any manual design involved. The ‘Channel’ group operates in a complementary
manner to the ‘Block Group’. In the ‘Channel’ Group, the pruning ratio of the group equals to the
pruning ratio of the parameters, as all weight matrices, including the embedding matrix, undergo
pruning. Therefore, a 20% pruning ratio of parameters implies pruning 20% of the groups, while a
50% pruning ratio implies pruning 50% of the groups.


**B.2** **For Recovery Stage**



We follow [19] in our recovery stage. We set the rank _d_ to 8 in our
experiment. The learning rate is set to 1e-4 with 100 warming steps.
The batch size of training is selected from {64, 128} and the AdamW

- ptimizer is employed in our experiment. The best training epoch we
found is 2 epochs, as training with more epochs even has a negative
impact on the model performance. We run our experiment on a single
GPU with 24GB memory, using approximately 2.5 hours if RTX4090
is utilized. All the linear module is taken into account for efficient
tuning. An ablation experiment for this is shown in Table 12.


**C** **More Analysis**


**C.1** **Pruning vs. Quantization**



Table 12: Ablation: Tuning
different modules in the re
covery stage


|Module|WikiText↓ PTB↓|
|---|---|
|ALL<br>- MLP<br>- MHA|17.36<br>29.99<br>17.64<br>30.63<br>17.62<br>30.23|



Here, we conduct a comparative analysis of different compression techniques and illustrate that
these techniques can be effectively combined with little performance degradation. We have chosen
LLM.int8() [8] as a representative example of quantization methods. Our results show that LLM.int8()

- utperforms LLM-Pruner while LLM-Pruner enhances latency, reduces parameter size. When these
two techniques are applied in tandem, they collectively reduce memory consumption and expedite
inference, offering a balanced approach that combines the benefits of both methods.

|Table 13: Pruning and Quantization on LLaMA-7B|Col2|Col3|Col4|
|---|---|---|---|
|Pruning Ratio<br>#Param<br>Memory<br>Latency<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>#Param<br>Memory<br>Latency<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>#Param<br>Memory<br>Latency<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|Pruning Ratio<br>#Param<br>Memory<br>Latency<br>BoolQ<br>PIQA<br>HellaSwag<br>WinoGrande<br>ARC-e<br>ARC-c<br>OBQA<br>Average|
|LLaMA-7B<br>LLM.int8()<br>LLaMA-5.4B<br>LLaMA-5.4B + LLM.int8()|6.74B<br>12884.5MiB<br>69.32s<br>6.74B<br>6777.7MiB<br>76.20s<br>5.47B<br>10488.4MiB<br>58.55s<br>5.47B<br>5444.37MiB<br>63.10s|73.18<br>78.35<br>72.99<br>67.01<br>67.45<br>41.38<br>42.40<br>73.36<br>78.18<br>73.01<br>66.93<br>67.47<br>40.87<br>41.80<br>76.57<br>77.37<br>66.60<br>65.82<br>70.62<br>40.70<br>38.80<br>76.39<br>76.71<br>66.62<br>66.46<br>70.54<br>40.19<br>39.20|63.25<br>63.09<br>62.36<br>62.30|



**C.2** **Global Pruning vs. Local Pruning**


we present a comparative analysis between global pruning and local pruning, where the pruning ratio
is 20% and the base model is LLaMA-7B. Global pruning refers to ranking all groups in the model
together, whereas local pruning involves only ranking groups within the same module for pruning.
The outcome of global pruning leads to varying widths across different layers and modules, whereas
local pruning ensures uniformity across all layers.


Based on our experimental findings, we observed a slight advantage of local pruning over global
pruning. We think this is because of the varying magnitudes in different layers or modules, which
makes the importance scores incomparable between groups across different layers.


Table 14: Results of global pruning and local pruning

|Method|WikiText2↓ PTB↓|BoolQ PIQA HellaSwag WinoGrande ARC -e ARC -c OBQA|Average|
|---|---|---|---|
|Element1 - local<br>Element1 - global|19.09<br>34.21<br>20.84<br>32.86|57.06<br>75.68<br>66.80<br>59.83<br>60.94<br>36.52<br>40.00<br>63.15<br>73.23<br>63.31<br>66.38<br>55.85<br>35.49<br>38.00|56.69<br>56.49|



16




--- end of page=15 ---

**C.3** **Overfitting Phenomena in Post-Training**


We present a comprehensive analysis of the overfitting issue in the recovery stage, as previously
mentioned in Figure 5. Here the results cover all 9 datasets across various training steps. Based on
the findings presented in Table 15, a noticeable trend emerges: the accuracy or generation quality
initially shows improvement but subsequently experiences a slight decline. This pattern suggests
that the recovery process is completed within a short period. And given that the training corpus
is domain-constrained, more training epochs can result in overfitting to the specific dataset while
potentially compromising the original capabilities of the language model.


Table 15: The PPL and Accuracy on different training steps

|Step|WikiText2↓ PTB↓|BoolQ PIQA HellaSwag WinoGrande ARC -e ARC -c OBQA|Average|
|---|---|---|---|
|0<br>200<br>400<br>600<br>800<br>1000<br>1200<br>1400|19.09<br>34.21<br>18.10<br>30.66<br>17.69<br>30.26<br>17.69<br>30.57<br>17.64<br>30.57<br>17.67<br>30.60<br>17.74<br>30.75<br>17.88<br>30.85|57.06<br>75.68<br>66.80<br>59.83<br>60.94<br>36.52<br>40.00<br>64.62<br>77.20<br>68.80<br>63.14<br>64.31<br>36.77<br>39.80<br>63.00<br>76.66<br>68.75<br>63.54<br>64.39<br>37.20<br>40.60<br>66.24<br>76.28<br>68.52<br>63.85<br>64.48<br>37.37<br>41.00<br>65.05<br>76.22<br>68.38<br>63.77<br>63.64<br>37.29<br>40.80<br>66.39<br>76.17<br>68.24<br>64.17<br>63.05<br>37.37<br>41.60<br>65.75<br>76.28<br>68.28<br>63.77<br>63.30<br>37.63<br>41.20<br>64.34<br>76.28<br>68.31<br>63.85<br>63.47<br>37.80<br>41.20|56.69<br>59.24<br>59.16<br>59.68<br>59.31<br>59.57<br>59.46<br>59.32|



**C.4** **Pruning with Large Rates**


Additionally, we conducted tests on LLaMA-7B and Vicuna-7B with 50% parameters pruned. We

- bserve a significant decrease in performance compared to the base model. However, the recovery
stage proved to be beneficial, resulting in an improvement of approximately 7.39%. Pruning a
Language Model with such a high pruning rate remains a challenging task.


Table 16: The PPL and Accuracy on LLaMA-7B with 50% parameters pruned





|Pruned Model Method|WikiText2 ↓ PTB↓|BoolQ PIQA HellaSwag WinoGrande ARC-e ARC-c OBQA|Average|
|---|---|---|---|
|Ratio = 50%<br>w/o tune<br>l2<br>39266.42<br>48867.85<br>55.11<br>53.59<br>27.03<br>49.49<br>26.43<br>29.01<br>34.40<br>39.29<br>random<br>3887.90<br>4337.27<br>46.79<br>53.37<br>27.50<br>50.59<br>28.07<br>27.90<br>30.00<br>37.75<br>Channel<br>13891.92<br>16114.91<br>40.37<br>52.18<br>25.72<br>48.86<br>25.72<br>28.24<br>30.40<br>35.93<br>Vector<br>141.06<br>236.24<br>62.17<br>55.11<br>27.25<br>49.88<br>29.00<br>25.77<br>34.00<br>40.45<br>Element2<br>106.07<br>266.65<br>52.57<br>60.45<br>35.86<br>49.01<br>32.83<br>25.51<br>34.80<br>41.58<br>Element1<br>112.44<br>255.38<br>52.32<br>59.63<br>35.64<br>53.20<br>33.50<br>27.22<br>33.40<br>42.13|39266.42<br>48867.85<br>3887.90<br>4337.27|55.11<br>53.59<br>27.03<br>49.49<br>26.43<br>29.01<br>34.40<br>46.79<br>53.37<br>27.50<br>50.59<br>28.07<br>27.90<br>30.00|39.29<br>37.75|
|Ratio = 50%<br>w/o tune<br>l2<br>39266.42<br>48867.85<br>55.11<br>53.59<br>27.03<br>49.49<br>26.43<br>29.01<br>34.40<br>39.29<br>random<br>3887.90<br>4337.27<br>46.79<br>53.37<br>27.50<br>50.59<br>28.07<br>27.90<br>30.00<br>37.75<br>Channel<br>13891.92<br>16114.91<br>40.37<br>52.18<br>25.72<br>48.86<br>25.72<br>28.24<br>30.40<br>35.93<br>Vector<br>141.06<br>236.24<br>62.17<br>55.11<br>27.25<br>49.88<br>29.00<br>25.77<br>34.00<br>40.45<br>Element2<br>106.07<br>266.65<br>52.57<br>60.45<br>35.86<br>49.01<br>32.83<br>25.51<br>34.80<br>41.58<br>Element1<br>112.44<br>255.38<br>52.32<br>59.63<br>35.64<br>53.20<br>33.50<br>27.22<br>33.40<br>42.13|13891.92<br>16114.91|40.37<br>52.18<br>25.72<br>48.86<br>25.72<br>28.24<br>30.40|35.93|
|Ratio = 50%<br>w/o tune<br>l2<br>39266.42<br>48867.85<br>55.11<br>53.59<br>27.03<br>49.49<br>26.43<br>29.01<br>34.40<br>39.29<br>random<br>3887.90<br>4337.27<br>46.79<br>53.37<br>27.50<br>50.59<br>28.07<br>27.90<br>30.00<br>37.75<br>Channel<br>13891.92<br>16114.91<br>40.37<br>52.18<br>25.72<br>48.86<br>25.72<br>28.24<br>30.40<br>35.93<br>Vector<br>141.06<br>236.24<br>62.17<br>55.11<br>27.25<br>49.88<br>29.00<br>25.77<br>34.00<br>40.45<br>Element2<br>106.07<br>266.65<br>52.57<br>60.45<br>35.86<br>49.01<br>32.83<br>25.51<br>34.80<br>41.58<br>Element1<br>112.44<br>255.38<br>52.32<br>59.63<br>35.64<br>53.20<br>33.50<br>27.22<br>33.40<br>42.13|141.06<br>236.24<br>106.07<br>266.65<br>112.44<br>255.38|62.17<br>55.11<br>27.25<br>49.88<br>29.00<br>25.77<br>34.00<br>52.57<br>60.45<br>35.86<br>49.01<br>32.83<br>25.51<br>34.80<br>52.32<br>59.63<br>35.64<br>53.20<br>33.50<br>27.22<br>33.40|40.45<br>41.58<br>42.13|
|Ratio = 50%<br>w/ tune<br>Channel<br>1122.15<br>1092.26<br>40.76<br>54.84<br>26.94<br>49.41<br>27.86<br>25.43<br>32.20<br>36.77<br>Vector<br>43.47<br>68.51<br>**62.11**<br>64.96<br>40.52<br>51.54<br>46.38<br>28.33<br>32.40<br>46.61<br>Element2<br>45.70<br>69.33<br>61.47<br>68.82<br>**47.56**<br>**55.09**<br>**46.46**<br>28.24<br>35.20<br>**48.98**<br>Element1<br>**38.12**<br>**66.35**<br>60.28<br>**69.31**<br>47.06<br>53.43<br>45.96<br>**29.18**<br>**35.60**<br>48.69|Ratio = 50%<br>w/ tune<br>Channel<br>1122.15<br>1092.26<br>40.76<br>54.84<br>26.94<br>49.41<br>27.86<br>25.43<br>32.20<br>36.77<br>Vector<br>43.47<br>68.51<br>**62.11**<br>64.96<br>40.52<br>51.54<br>46.38<br>28.33<br>32.40<br>46.61<br>Element2<br>45.70<br>69.33<br>61.47<br>68.82<br>**47.56**<br>**55.09**<br>**46.46**<br>28.24<br>35.20<br>**48.98**<br>Element1<br>**38.12**<br>**66.35**<br>60.28<br>**69.31**<br>47.06<br>53.43<br>45.96<br>**29.18**<br>**35.60**<br>48.69|Ratio = 50%<br>w/ tune<br>Channel<br>1122.15<br>1092.26<br>40.76<br>54.84<br>26.94<br>49.41<br>27.86<br>25.43<br>32.20<br>36.77<br>Vector<br>43.47<br>68.51<br>**62.11**<br>64.96<br>40.52<br>51.54<br>46.38<br>28.33<br>32.40<br>46.61<br>Element2<br>45.70<br>69.33<br>61.47<br>68.82<br>**47.56**<br>**55.09**<br>**46.46**<br>28.24<br>35.20<br>**48.98**<br>Element1<br>**38.12**<br>**66.35**<br>60.28<br>**69.31**<br>47.06<br>53.43<br>45.96<br>**29.18**<br>**35.60**<br>48.69|Ratio = 50%<br>w/ tune<br>Channel<br>1122.15<br>1092.26<br>40.76<br>54.84<br>26.94<br>49.41<br>27.86<br>25.43<br>32.20<br>36.77<br>Vector<br>43.47<br>68.51<br>**62.11**<br>64.96<br>40.52<br>51.54<br>46.38<br>28.33<br>32.40<br>46.61<br>Element2<br>45.70<br>69.33<br>61.47<br>68.82<br>**47.56**<br>**55.09**<br>**46.46**<br>28.24<br>35.20<br>**48.98**<br>Element1<br>**38.12**<br>**66.35**<br>60.28<br>**69.31**<br>47.06<br>53.43<br>45.96<br>**29.18**<br>**35.60**<br>48.69|
|Ratio = 50%<br>w/ tune<br>Channel<br>1122.15<br>1092.26<br>40.76<br>54.84<br>26.94<br>49.41<br>27.86<br>25.43<br>32.20<br>36.77<br>Vector<br>43.47<br>68.51<br>**62.11**<br>64.96<br>40.52<br>51.54<br>46.38<br>28.33<br>32.40<br>46.61<br>Element2<br>45.70<br>69.33<br>61.47<br>68.82<br>**47.56**<br>**55.09**<br>**46.46**<br>28.24<br>35.20<br>**48.98**<br>Element1<br>**38.12**<br>**66.35**<br>60.28<br>**69.31**<br>47.06<br>53.43<br>45.96<br>**29.18**<br>**35.60**<br>48.69|43.47<br>68.51<br>45.70<br>69.33<br>**38.12**<br>**66.35**|**62.11**<br>64.96<br>40.52<br>51.54<br>46.38<br>28.33<br>32.40<br>61.47<br>68.82<br>**47.56**<br>**55.09**<br>**46.46**<br>28.24<br>35.20<br>60.28<br>**69.31**<br>47.06<br>53.43<br>45.96<br>**29.18**<br>**35.60**|46.61<br>**48.98**<br>48.69|


Table 17: The PPL and Accuracy on Vicuna-7B with 50% parameters pruned








|Pruned Model Method|WikiText2 ↓ PTB↓|BoolQ PIQA HellaSwag WinoGrande ARC-e ARC-c OBQA|Average|
|---|---|---|---|
|Ratio = 0%<br>Vicuna-7B|16.11<br>61.37|76.57<br>77.75<br>70.64<br>67.40<br>65.11<br>41.21<br>40.80|62.78|
|Ratio = 50%<br>w/o tune<br>l2<br>54516.03<br>66274.63<br>45.99<br>53.48<br>26.55<br>47.83<br>27.53<br>28.58<br>30.40<br>37.19<br>random<br>17020.73<br>13676.54<br>48.17<br>53.43<br>27.31<br>50.43<br>26.30<br>29.78<br>30.20<br>37.95<br>Channel<br>8360.30<br>10651.30<br>38.69<br>53.10<br>26.42<br>50.20<br>25.97<br>29.52<br>29.60<br>36.22<br>Vector<br>189.87<br>409.75<br>62.14<br>55.33<br>26.99<br>51.93<br>27.86<br>26.02<br>32.60<br>40.41<br>Element2<br>143.85<br>427.77<br>53.76<br>59.79<br>34.86<br>50.28<br>33.29<br>27.30<br>34.60<br>41.98<br>Element1<br>162.38<br>469.74<br>52.45<br>58.65<br>34.27<br>49.72<br>32.41<br>28.75<br>35.80<br>41.72|54516.03<br>66274.63<br>17020.73<br>13676.54|45.99<br>53.48<br>26.55<br>47.83<br>27.53<br>28.58<br>30.40<br>48.17<br>53.43<br>27.31<br>50.43<br>26.30<br>29.78<br>30.20|37.19<br>37.95|
|Ratio = 50%<br>w/o tune<br>l2<br>54516.03<br>66274.63<br>45.99<br>53.48<br>26.55<br>47.83<br>27.53<br>28.58<br>30.40<br>37.19<br>random<br>17020.73<br>13676.54<br>48.17<br>53.43<br>27.31<br>50.43<br>26.30<br>29.78<br>30.20<br>37.95<br>Channel<br>8360.30<br>10651.30<br>38.69<br>53.10<br>26.42<br>50.20<br>25.97<br>29.52<br>29.60<br>36.22<br>Vector<br>189.87<br>409.75<br>62.14<br>55.33<br>26.99<br>51.93<br>27.86<br>26.02<br>32.60<br>40.41<br>Element2<br>143.85<br>427.77<br>53.76<br>59.79<br>34.86<br>50.28<br>33.29<br>27.30<br>34.60<br>41.98<br>Element1<br>162.38<br>469.74<br>52.45<br>58.65<br>34.27<br>49.72<br>32.41<br>28.75<br>35.80<br>41.72|8360.30<br>10651.30|38.69<br>53.10<br>26.42<br>50.20<br>25.97<br>29.52<br>29.60|36.22|
|Ratio = 50%<br>w/o tune<br>l2<br>54516.03<br>66274.63<br>45.99<br>53.48<br>26.55<br>47.83<br>27.53<br>28.58<br>30.40<br>37.19<br>random<br>17020.73<br>13676.54<br>48.17<br>53.43<br>27.31<br>50.43<br>26.30<br>29.78<br>30.20<br>37.95<br>Channel<br>8360.30<br>10651.30<br>38.69<br>53.10<br>26.42<br>50.20<br>25.97<br>29.52<br>29.60<br>36.22<br>Vector<br>189.87<br>409.75<br>62.14<br>55.33<br>26.99<br>51.93<br>27.86<br>26.02<br>32.60<br>40.41<br>Element2<br>143.85<br>427.77<br>53.76<br>59.79<br>34.86<br>50.28<br>33.29<br>27.30<br>34.60<br>41.98<br>Element1<br>162.38<br>469.74<br>52.45<br>58.65<br>34.27<br>49.72<br>32.41<br>28.75<br>35.80<br>41.72|189.87<br>409.75<br>143.85<br>427.77<br>162.38<br>469.74|62.14<br>55.33<br>26.99<br>51.93<br>27.86<br>26.02<br>32.60<br>53.76<br>59.79<br>34.86<br>50.28<br>33.29<br>27.30<br>34.60<br>52.45<br>58.65<br>34.27<br>49.72<br>32.41<br>28.75<br>35.80|40.41<br>41.98<br>41.72|
|Channel<br>206.23<br>526.89<br>62.11<br>58.38<br>32.29<br>51.22<br>33.08<br>24.06<br>30.20<br>41.62<br>Ratio = 50%<br>w/ tune<br>Vector<br>46.11<br>147.52<br>**62.14**<br>64.91<br>39.80<br>50.91<br>47.77<br>27.30<br>32.60<br>46.49<br>Element2<br>42.99<br>**139.91**<br>58.87<br>**69.48**<br>46.38<br>**54.78**<br>46.89<br>29.01<br>34.80<br>48.60<br>Element1<br>**41.64**<br>143.74<br>62.08<br>**69.48**<br>**46.79**<br>54.54<br>**48.15**<br>**29.78**<br>**36.20**<br>**49.57**|Channel<br>206.23<br>526.89<br>62.11<br>58.38<br>32.29<br>51.22<br>33.08<br>24.06<br>30.20<br>41.62<br>Ratio = 50%<br>w/ tune<br>Vector<br>46.11<br>147.52<br>**62.14**<br>64.91<br>39.80<br>50.91<br>47.77<br>27.30<br>32.60<br>46.49<br>Element2<br>42.99<br>**139.91**<br>58.87<br>**69.48**<br>46.38<br>**54.78**<br>46.89<br>29.01<br>34.80<br>48.60<br>Element1<br>**41.64**<br>143.74<br>62.08<br>**69.48**<br>**46.79**<br>54.54<br>**48.15**<br>**29.78**<br>**36.20**<br>**49.57**|Channel<br>206.23<br>526.89<br>62.11<br>58.38<br>32.29<br>51.22<br>33.08<br>24.06<br>30.20<br>41.62<br>Ratio = 50%<br>w/ tune<br>Vector<br>46.11<br>147.52<br>**62.14**<br>64.91<br>39.80<br>50.91<br>47.77<br>27.30<br>32.60<br>46.49<br>Element2<br>42.99<br>**139.91**<br>58.87<br>**69.48**<br>46.38<br>**54.78**<br>46.89<br>29.01<br>34.80<br>48.60<br>Element1<br>**41.64**<br>143.74<br>62.08<br>**69.48**<br>**46.79**<br>54.54<br>**48.15**<br>**29.78**<br>**36.20**<br>**49.57**|Channel<br>206.23<br>526.89<br>62.11<br>58.38<br>32.29<br>51.22<br>33.08<br>24.06<br>30.20<br>41.62<br>Ratio = 50%<br>w/ tune<br>Vector<br>46.11<br>147.52<br>**62.14**<br>64.91<br>39.80<br>50.91<br>47.77<br>27.30<br>32.60<br>46.49<br>Element2<br>42.99<br>**139.91**<br>58.87<br>**69.48**<br>46.38<br>**54.78**<br>46.89<br>29.01<br>34.80<br>48.60<br>Element1<br>**41.64**<br>143.74<br>62.08<br>**69.48**<br>**46.79**<br>54.54<br>**48.15**<br>**29.78**<br>**36.20**<br>**49.57**|
|Channel<br>206.23<br>526.89<br>62.11<br>58.38<br>32.29<br>51.22<br>33.08<br>24.06<br>30.20<br>41.62<br>Ratio = 50%<br>w/ tune<br>Vector<br>46.11<br>147.52<br>**62.14**<br>64.91<br>39.80<br>50.91<br>47.77<br>27.30<br>32.60<br>46.49<br>Element2<br>42.99<br>**139.91**<br>58.87<br>**69.48**<br>46.38<br>**54.78**<br>46.89<br>29.01<br>34.80<br>48.60<br>Element1<br>**41.64**<br>143.74<br>62.08<br>**69.48**<br>**46.79**<br>54.54<br>**48.15**<br>**29.78**<br>**36.20**<br>**49.57**|46.11<br>147.52<br>42.99<br>**139.91**<br>**41.64**<br>143.74|**62.14**<br>64.91<br>39.80<br>50.91<br>47.77<br>27.30<br>32.60<br>58.87<br>**69.48**<br>46.38<br>**54.78**<br>46.89<br>29.01<br>34.80<br>62.08<br>**69.48**<br>**46.79**<br>54.54<br>**48.15**<br>**29.78**<br>**36.20**|46.49<br>48.60<br>**49.57**|



**D** **Generations From Compressed Model**


Table 18, 19, 20 and 21 show more examples of the models pruned by LLM-Pruner. We present
the generation results of both the pruned model with post-training and without post-training. The
absence of post-training allows us to better understand the information retained in the model. We
include the results of ChatGLM-6B in two languages as it is a bilingual model.


17




--- end of page=16 ---

Table 18: Generated Examples from the Compressed LLaMA-5.4B


Model Generated Sentences


Neural Sequential Model, especially transformers, have become very popular due to their impressive performance in
different areas. The neural network model can recognize the human voice more efficiently than speech recognition,
and speech can be better distinguished. However, its high energy consumption has been the biggest problem. Many
efforts have been made, such as minimizing the number of layers and the length of the neural network, but it has had
no effect. The work is to develop a low-dimensional sound model which is suitable for the high frequency band of the
speech. After reducing the ...



LLaMA

w/ Tune


LLaMA

w/o Tune



AI can create a logo in seconds or design a book cover, and they need to do it much faster than us humans can. So
what are the implications that this has for branding design? Giving Creative Control to Algorithms. The first wave of
‘creative’ branding has come from designers and agencies submitting their creations to algorithms. They are algorithms
that have been programmed to create, and are being asked to replace designers in the creative process. For example, in
1998 Google used a random process to select words and colours for their logo. Now, in


What’s great about the holiday season, is it provides us with the excuse we need to reconnect with the ones we have
grown apart from. There are many people who I only talk to a couple of times per year during the year and never
interact with them anymore, and while I’m completely okay with that – I feel they too should use Christmas as a great

- pportunity to get back in touch with people in their lives that maybe they were too busy to talk to before. If you are

- n good terms with a family member or friend, and you haven’t been talking to them lately due to how


Neural Sequential Model, especially transformers, we designed them as a neural network that transforms input words
to corresponding output. In our model, the 73 neural nodes are arranged in the form of L-shaped and are used to
construct the two-dimensional lattice and form a sentence. This sentence is then transferred to the decoder to learn the

words. Through 5.4.1 million repetitions, we can form a language model through the neural network we designed.
With the help of a word-level data correction method, we made a strong progress in the recognition of human language.


AI can create a logo in seconds. You’ll see a design in an instant and you’ll be ?? that you’re getting one the best
logo designs around. The process is quick and easy and if you need it now you’ll pay just a fraction of the cost for a
professional logo designer. Why would I need a logo? It helps a brand stand out from the competition. It’s the visual
cue that people respond to - the logo. It’s what makes you stop and look at the brand when you’re walking by a street.
Why invest in a logo?


What’s great about the holiday season is being with the people that you love, and the holidays are the perfect occasion
to celebrate with the people that you love most. Being together with family and friends is one of those special occasions
that can have a remarkable effect on how we perceive our lives, which is why they become some of the most cherished
memories of our lifetimes. But holiday planning is more complex than a simple gathering. In fact, many families need
professional advice to find the right way to handle the emotions involved. And that’s why planning, preparation,


Table 19: Generated Examples from the Compressed Vicuna-5.4B



Model Generated Sentences


Last night we all danced together in the rain, all our cares and worries forgotten. And as the sun began to rise, I looked
into your eyes and knew that everything was going to be okay. That’s the way it is now, forever and forever. I love you,
my baby, and I always will.



Vicuna

w/ Tune


Vicuna

w/o Tune



It fills me with such pride and joy to be working along side her, learning from her and sharing our experiences, joys,
and challenges as partners. Our bond is growing stronger each day and I can’t help but feel grateful for the beautiful
life she has brought into my own. To my loving wife, I love you more and more each day. I can’t wait to spend the rest

- f my life with you and raising our future children together proyecting the light of Christ into the world. And to all of
you reading this, I encourage you to take this seriously and consider your relationship with the one


Meet me at midnight tonight at the Bell Tower in town. Bring a torch, and we’ll take the treasure off the church’s
hands, no matter the cost.’ With a deep sense of trepidation, Alaric finds himself agreeing to the plan. He spends the
day getting everything he can from the village, preparing for the inevitable attack he’ll have to launch. When the day
arrives, he and his companions mount their steeds and set off towards the towering keep of the temple. As they draw
closer, the wind whispers


Last night we all danced together in the rain. The water felt good on my face. The song played over and over, but I
didn’t care. I just kept on dancing with you. I don’t know what to say. I’m not good with words. But I know that I’ve
never felt this way before. With anyone. Not even myself. And I don’t want to let that feeling go. I don’t know what
you feel, or what you want from me. But I know that this is real. And I don’t want to let it go


It fills me with such pride and joy, it’s like seeing my baby for the first time, and I just want to hold it close and never
let it go. The thought that it might have to be put down one day breaks my heart. I want to do everything in my power
to protect it, to keep it safe and give it the best life possible. But it’s not mine. It’s never been mine to keep. And I
know that one day, sooner or later, it will have to be given away. I can’t help but feel a sense of responsibility that I
don’t ...


Meet me at midnight, I promise you’ll be the only one there. It’s hard for me to believe, but I’m scared. I don’t want
to be alone tonight, but I know I have to be brave. Hurry down the hill, I’m waiting for you, I’ll take you to a place that
I know is safe. We won’t be seen, nobody’ll know. It’s almost time for you to go. I don’t want you to be late tonight.


18




--- end of page=17 ---

Table 20: Generated Examples from the Compressed ChatGLM-5.6B (English)


Model Generated Sentences


How to buy a cat? It’s important to understand that while it’s understandable that you might be looking todnle a cat for
yourself, adoption is a great way to bring home a furry friend without having to shop. Here are some things to consider
before adoption: 1. First, make sure you know you’re interested in Jr. or Vioness. This determines whether you’re
looking for a young adult cat or an older cat. 2. research your options: Look up reputable cat owners or organizations
that foster cats in your area. This will give you information on cat owners



ChatGLM

w/ Tune


ChatGLM

w/o Tune



What is McDonald’s? McDonald’s is a popular American fast-food restaurant chain that offers a variety of food and
beverage items. The company was founded in 1940 by Richard and Maurice Mc Eu Eu in San Francisco, and the
chain was originally called "McNugguggets." Today, McDonald’s is known for its popular fast-food items such as
McDonald’s Drive-Thru, McChickens, and McFlurryrry. The company also offers a wide range of other food and
beverage items such as Happy Endings, Big Macs, and ...


What is Mean Square Error? The Mean Square Error (MSE) is a measure of the difference between a target value and
the best possible estimate of its true value, based on the data. It is calculated by taking the average difference between
the actual value and the estimated value over the※ number of times the target value is repeated.


How to buy a cat? It’s important to understand that adopting a cat from a shelter or organization is a valuable and
responsible adoption option, as well as one that may be safer and more comfortable for you and your family.
Additionally, it’s important to read carefully before adopting to ensure that you choose a Cat who is suitable for you
and your family.


What is McDonald‘s? McDonald’s is a popular fast-food restaurant chain from United States known for its restaurant
concepts such as McNugguggets, McFlurryrry, and McRibibs. The chain started in 1955 as a small operation operation
called "Speedy McDonald’s" in San Antonio, Texas and it quickly expanded and became a widely recognized and
popular phenomenon in the country. Today, McDonald’s is global icon known around the world.


What is Mean Square Error? Mean Square Error (MSE) is a measure of the accuracy of a function or a measurement
when it is repeated and the result is expected to be within a certain range. It is calculated by taking the difference
between two sets of measurements and measuring the difference squared. In simple terms, it means the mean sum of
the errors over a given period of time or the difference between the actual and expected values of ...



Table 21: Generated Examples from the Compressed ChatGLM-5.6B (Chinese)


Model Generated Sentences


`请给我推荐两个北京著名的景点` `北京著名的景点有很多`, `以下是两个` :1. `颐和园` : `颐和园是北京的一座皇家园`
`林`, `也是世界文化遗产。它是一座是一座结合了中国传统建筑和现代艺术的杰作`, `被誉为` “ `万园之园` ” `。颐和园`
`内有很多建筑和景点`, `包括长廊、排排坐、佛香阁、万寿山等`, `非常适合漫步和参观。` 2. `故宫` : `故宫是中国古`
`代皇宫`, `也是世界文化遗产`, `是北京的必去之地。它是世界上是世界上现存最大的、最完整的、最完整的、`
`最著名的古代宫殿宫殿建筑之一。故宫内有很多建筑和景点`, `包括紫禁城`



ChatGLM

w/ Tune


ChatGLM

w/o Tune



`简单来说`, `相对论指出` `相对论是由爱因斯坦提出的一个关于时间和空间如何解释和变化的的理论`, `被称为狭`
`义相对论。相对论认为`, `时间和空间并不是绝对的`, `而是相对的`, `也就是说`, `它们的测量值取决于观察者的惯性`
`参照系。根据相对论`, `当物体以高速运动或以或以某种方式加速时`, `时间和空间的测量值会发生变化。最著`
`名的是`, `时间和空间的测量值会` “ `膨胀` ”, `即随着物体运动`, `它们看起来会变得越来越长或越来越短。`


`如何健康的生活` `要保持健康的生活方式`, `可以参考以下建议` :1. `定期锻炼` : `适量运动可以帮助增强身体健康`, `减`
`轻压力`, `改善心情。建议每周至少进行` 3-5 `次有氧运动`, `如慢跑、快走、游泳、瑜伽等。` 2. `饮食均衡` : `控制饮食`
`摄入量`, `摄入摄入适量、健康的蛋白质、碳水化合物、脂肪以及维生素和矿物质。` 3. `睡眠充足` : `充足的睡眠`
`有助于恢复身体和大脑功能`, `提高身体免疫力。建议每晚睡眠` 7-8 `小时。` 4. `勤洗手` : `避免传染病毒。`


`请给我推荐两个北京著名的景点` `北京著名的景点太多`, `以下是两个必去的景点` :1. `故宫博物院` : `故宫是世界上`
`是世界上是世界上保存最完整的、最著名的宫殿建筑之一`, `也是中国历史文化名城和世界文化遗产。游客可`
`以参观紫禁城紫禁城`, `欣赏到欣赏到大量的历史文物和艺术品。` 2. `长城` : `长城是中国古代著名的防御工程之`
`一`, `也是中国历史文化名城和世界文化遗产。游客可以参观长城的各种不同高度和不同用途的客房现代物流`
`设施`, `还可以在长城上徒步。`


`简单来说`, `相对论指出` `相对论是由爱因斯坦提出的一组理论和概念`, `包括两个部分` : `狭义相对论和广义相对`
`论。狭义相对论是描述在高速运动或缩短物体长度时`, `时间和空间如何弯曲的理论。它指出`, `当我们测量一`
`个物体的长度或加速运动时`, `我们的测量结果观测到的是相对一个观察者观察者静止静止的物体长度或速度`
```
的尺子。这个理论解释了为什么光速是恒定不变的不变的。广义无味相对论是描述引力如何加速和改变物
```

`体的质量的。它指出`, `引力是由质量` m `和引力场引起的`, `两个物体引力场的中心的引力是`


`如何健康的生活` `想要过健康的生活可以参考以下建议` : 1. `保持规律的规律的充足的睡眠` : `尽可能每晚每晚`
`睡` 7-8 `个小时`, `保持规律的规律的睡眠时间有助于维持身体健康和精神状态。` 2. `多运动` : `每天进行适量的运动`
`有助于减轻压力和增强身体免疫力`, `建议每天进行适量的有氧运动和力量训练。` 3. `饮食均衡` : `保持均衡的饮`
`食`, `包括摄入足够的各种营养素和禁止过量摄入垃圾食品和饮料。` 4. `减少不良习惯` : `尽可能减少不良习惯`, `如`
```
吸烟、酗酒和过度使用电子设备等。

```

19




--- end of page=18 ---
