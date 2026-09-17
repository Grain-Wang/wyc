---
id: "2022_AutoDistil"
title: "Few-shot Task-agnostic Neural Architecture Search for Distilling Large Language Models"
authors: ["Dongkuan Xu", "Subhabrata Mukherjee", "Xiaodong Liu", "Debadeepta Dey", "Wenhui Wang", "Xiang Zhang", "Ahmed Hassan Awadallah", "Jianfeng Gao"]
year: 2022
venue: "NeurIPS 2022"
publication_status: "FORMALLY PUBLISHED"
category: "Direct Neighbor / Blockwise Distillation"
source_pdf: "../reference_papers_origin/2022_AutoDistil.pdf"
paper_url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/b7c12689a89e98a61bcaa65285a41b7c-Abstract-Conference.html"
pdf_url: "https://proceedings.neurips.cc/paper_files/paper/2022/file/b7c12689a89e98a61bcaa65285a41b7c-Paper-Conference.pdf"
code_url: "https://github.com/microsoft/autodistil"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# Few-shot Task-agnostic Neural Architecture Search for Distilling Large Language Models

**Authors:** Dongkuan Xu, Subhabrata Mukherjee, Xiaodong Liu, Debadeepta Dey, Wenhui Wang, Xiang Zhang, Ahmed Hassan Awadallah, Jianfeng Gao

**Venue / Year:** NeurIPS 2022 (FORMALLY PUBLISHED)

**Category:** Direct Neighbor / Blockwise Distillation

**Why this paper matters for low-cost post-training LLM NAS:** It partitions the Transformer search space into compact SuperLMs and uses task-agnostic self-attention distillation for lightweight student search.

**Primary record:** [https://proceedings.neurips.cc/paper_files/paper/2022/hash/b7c12689a89e98a61bcaa65285a41b7c-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b7c12689a89e98a61bcaa65285a41b7c-Abstract-Conference.html)

**Local source:** [2022_AutoDistil.pdf](../reference_papers_origin/2022_AutoDistil.pdf)

## Full converted text

# **Few-shot Task-agnostic Neural Architecture Search for** **Distilling Large Language Models**



**Dongkuan Xu** _[⇤]_
NC State University
dxu27@ncsu.edu



**Subhabrata Mukherjee** _[†]_

Microsoft Research

submukhe@microsoft.com



**Xiaodong Liu**
Microsoft Research



**Debadeepta Dey**
Microsoft Research



**Xiang Zhang**
Penn State University



**Wenhui Wang**
Microsoft Research



**Ahmed Hassan Awadallah**


Microsoft Research


## Abstract



**Jianfeng Gao**
Microsoft Research



Traditional knowledge distillation (KD) methods manually design student architectures to compress large models given pre-specified computational cost. This requires several trials to find viable students, and repeating the process with change in
computational budget. We use Neural Architecture Search (NAS) to automatically
distill several compressed students with variable cost from a large model. Existing
NAS methods train a single SuperLM consisting of millions of subnetworks with
weight-sharing, resulting in interference between subnetworks of different sizes.
Additionally, many of these works are task-specific requiring task labels for SuperLM training. Our framework AutoDistil addresses above challenges with
the following steps: (a) Incorporates inductive bias and heuristics to partition Transformer search space into _K_ compact sub-spaces (e.g., _K_ =3 can generate typical
student sizes of base, small and tiny); (b) Trains one SuperLM for each sub-space
using task-agnostic objective (e.g., self-attention distillation) with weight-sharing

     - f students; (c) Lightweight search for the optimal student without re-training.
Task-agnostic training and search allow students to be reused for fine-tuning on any
downstream task. Experiments on GLUE benchmark demonstrate AutoDistil
to outperform state-of-the-art KD and NAS methods with upto 41x reduction in
[computational cost. Code and models are available at aka.ms/autodistil.](https://aka.ms/autodistil)


**1** **Introduction**


While large pre-trained language models (e.g., BERT [1], GPT-3 [2]) are effective, their huge size
poses significant challenges for downstream applications in terms of energy consumption and cost

- f inference [3] limiting their usage in on the edge scenarios and under constrained computational
inference budgets. Knowledge distillation [4, 5, 6, 7] has shown strong results in compressing pretrained language models into small student models. However, these works require pre-specification of
the student architecture and computational cost (e.g., number of parameters, FLOPs) for distillation.
This poses two significant challenges: (i) it requires several trials to come up with viable architectures
as they are hand-engineered and to define several hyper-parameters (e.g., number of layers and
attention heads, hidden dimension, etc.); (ii) one has to re-run distillation with any change in
specification for the student architecture or computational cost for using it in a target environment.


_⇤_ Work done while interning at Microsoft Research.

_†_ Corresponding author and project lead.


36th Conference on Neural Information Processing Systems (NeurIPS 2022).




--- end of page=0 ---

### Figure 1

Caption: AutoDistil uses few-shot task-agnostic NAS to distill several compressed students with

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
variable #FLOPs (x-axis) from _K_ =3 SuperLMs (corresponding to each point cloud) trained on _K_
sub-spaces of Transformer search space. Each student (blue dot) extracted from the SuperLM is
fine-tuned on MNLI with accuracy on y-axis. The best student from each SuperLM is marked in
red. Given any state-of-the-art distilled model, AutoDistil generates a better candidate with less
#FLOPs and improved task performance from corresponding search space. More discussions on
trade-off and selection strategies can be found in the Appendix (Sections G, H).


Neural Architecture Search (NAS) [8, 9, 10, 11] provides a natural solution to automatically search
through a large space of candidate models.The dominant NAS paradigm consists of two main steps:
(a) Training a Super model combining all possible architectures into a single graph and jointly training
them via weight-sharing; (b) Searching for optimal architecture from Super model with best accuracy

- n a downstream task, satisfying user-specified latency constraint for target device. Parallel to above
computer vision (CV) works, NAS has shown strong results in recent works like DynaBERT [12],
AutoTinyBERT [13] and NAS-BERT [14] for natural language understanding (NLU).


**Drawbacks of existing NAS methods.**

**[D1: Co-adaptation in weight-sharing]** Above works train one single large Super Language Model
(SuperLM) consisting of millions of diverse student architectures. This results in some undesirable
effects of co-adaptation [15] like conflicts in weight-sharing where bigger student models converge
faster in contrast to smaller ones converging slower [16, 11].

**[D2: Multi-stage training]** A single SuperLM may not have sufficient capacity to encode a large
search space. As a result, these works use multi-stage training process, where they first conduct
NAS to identify candidate students and then perform further pre-training [13] and knowledge distillation [14] of the candidates.

**[D3: Task-specific training]** NAS works in CV domain (e.g., AutoFormer [17], Once-for-all [10],
One-Shot NAS [11, 18]) leverage hard class labels from a task (e.g., image classification) or soft
labels from ImageNet pre-trained models (e.g., MobileNet [7], RegNet [19]) for task-specific optimization with accuracy as evaluation metric. Different from CV domain, NLU tasks have different

- bjectives and evaluation metrics for classification (e.g., MNLI), regression (e.g., STS-B) and correlation (e.g., CoLA). This makes it challenging to adapt existing NAS works to the NLU domain in a
task-agnostic setting. Recent NAS works in the NLU domain are not fully task-agnostic. For instance,
DynaBERT [12] accesses both task labels for knowledge distillation and task development set for
network rewiring. NAS-BERT [14] performs two-stage knowledge distillation with pre-training and
fine-tuning of the candidates. While AutoTinyBERT [13] also explores task-agnostic training, we
demonstrate better performance from few-shot NAS and much cheaper cost from single stage training
without additional pre-training and distillation. We present a detailed discussion on the differences
between AutoDistil and existing KD and NAS methods based on fine-grained search space,
different training strategies and amortized training cost in Table 1. More comparison details are
summarized in Section A of Appendix.


**Contributions.** We address above challenges with fully task-agnostic few-shot NAS consisting of
three steps. **(S1) Search space design.** We partition the Transformer search space into _K_ sub-spaces
considering important architectural hyper-parameters like the network depth, width and attention
heads. We further leverage inductive bias and heuristics to limit the number of student architectures
in each sub-space. **(S2) Fully task-agnostic SuperLM training.** We train _K_ SuperLM overall, one


2




--- end of page=1 ---

Table 1: Comparing AutoDistil with existing KD and NAS methods on aspects as task-agnostic
training; generating students with variable compression cost; single-stage training without additional
adaptation; SuperLM training with compact search space to mitigate interference ( _P_ denotes partial).


Method Task- Variable NAS
agnostic Compression Single SuperLM Compact
Stage Training Search


BERT-PKD 7 7



SparseBERT 7 7
DistilBERT X 7
TinyBERT X 7
MOBILEBERT X 7
MINILM X 7



N/A



DynaBERT 7 X X One-shot 7
NAS-BERT _P_ X 7 One-shot 7
AutoTinyBERT _P_ X 7 One-shot 7
AutoDistil X X X Few-shot X


for every sub-space. This allows each SuperLM more capacity to encode a sub-space as opposed to a
single large one. We train each SuperLM with a fully task-agnostic objective (without accessing any
task labels) like deep self-attention distillation, where we transfer knowledge from the self-attention
module (including keys, queries and values) of a pre-trained teacher (e.g., BERT) to the student and
use weight-sharing to train the SuperLM. **(S3) Lightweight optimal student search.** We obtain

- ptimal student(s) directly from well-trained SuperLM(s) without any re-training that can be _simply_
_fine-tuned_ - n downstream tasks. Our contributions over existing NAS works can be summarized as:


- In contrast to prior works (e.g., DynaBERT, AutoTinyBERT, NAS-BERT), we do a single-stage

training combining NAS and distillation with no further pre-training or augmentation and demonstrate superior performance of the NAS process itself with significantly reduced training cost.
Obtained subnetworks are simply fine-tuned on downstream tasks.

- Fully task-agnostic training with subnetwork attention state alignment for self-attention relation

distillation and search in contrast to prior works in NLU (e.g., DynaBERT, NAS-BERT) and CV
(e.g., AutoFormer, BigNAS, Once-For-All).

Few-shot NAS to mitigate gradient conflicts in SuperNet training compared to prior One-shot NAS

works in NLU (e.g., DynaBERT, AutoTinyBERT, NAS-BERT). AutoFormer in the CV domain is
an exception to this point which also uses few-shot NAS but accesses task labels during training.

- Strong results over all the above NAS and distillation works in NLU with 3 _x_ additional compression

 - ver best performing distillation technique with negligible performance drop.


**2** **Background**


We present an overview of Transformers [20], especially its two main sub-layers, multi-head selfattention (MHA) and feed-forward network (FFN). Transformer layers are stacked to encode contextual information for input tokens as: **X** _[l]_ = Transformer _l_ ( **X** _[l][−]_ [1] ) _, l 2_ [1 _, L_ ] where _L_ is the number

- f Transformer layers, **X** _[l]_ _2_ R _[s][⇤][d][hid]_, _s_ is the sentence length, and _dhid_ is the hidden dimension. In
the following, we omit the layer indices for simplicity.
**Multi-Head Self-Attention (MHA).** Given previous Transformer layer’s output **X**, MHA computes:



Attention( **Q** _h,_ **K** _h,_ **V** _h_ ) = softmax( **[Q]** _[h]_ **[K]** _h_ _[>]_

~~_p_~~ _dhead_



) **V** _h_ ; **Q** _h,_ **K** _h,_ **V** _h_ = **X** _**W**_ _h_ _[Q]_



_h_ _[ K][,]_ **[ X]** _**[W]**_ _[ V]_ _h_



_h_ _[,]_ [ (1)]



_h_ _[Q][,]_ **[ X]** _**[W]**_ _h_ _[ K]_



MHA( **X** ) = Concat(head1 _, · · ·,_ head _H_ ) _**W**_ _[O]_ _,_ (2)



where _**W**_ _[Q]_



_h_ _[ K]_ [,] _**[ W]**_ _h_ _[ V]_



where _**W**_ _h_ _[Q]_ [,] _**[ W]**_ _h_ _[ K]_ [,] _**[ W]**_ _h_ _[ V]_ _[2]_ [ R] _[d][hid][⇤][d][head]_ [,] _**[ W]**_ _[ O][ 2]_ [ R] _[d][hid][⇤][d][hid]_ [ are linear transformations.] **[ Q]** _[h]_ [,] **[ K]** _[h]_ [,] **[ V]** _[h]_

_2_ R _[s][⇤][d][head]_ are called queries, keys, and values, respectively. _H_ is the number of heads. head _h_ =
Attention( **Q** _h,_ **K** _h,_ **V** _h_ ) denotes the _h_ - th attention head. Concat is the concatenating operation.
_dhead_ = _dhid/H_ is the dimension of each head.
**Feed-Forward Network (FFN).** Each Transformer layer contains an FNN sub-layer, which is stacked

- n the MHA. FFN consists of two linear transformations with a ReLU activation as:



_h_ [,] _**[ W]**_ _h_ _[ K]_



FFN( _x_ ) = max(0 _, x_ _**W**_ [1] + _b_ 1) _**W**_ [2] + _b_ 2 _,_ (3)


3




--- end of page=2 ---

_**K**_ **Search Partitions**



_**K**_ **-shot Learning for SuperLM**



### Figure 2

Caption: Overview of AutoDistil . It considers _K_ partitions of the Transformer architecture

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
subspace to train one SuperLM for each partition with weight-sharing of the constituent subnetworks
trained via task-agnostic deep self-attention distillation. Optimal compressed subnetworks can be
easily extracted from the SuperLMs without additional training or distillation.


where _**W**_ [1] _2_ R _[d][hid][⇤][d][f]_, _**W**_ [2] _2_ R _[d][f][ ⇤][d][hid]_, _b_ 1 _2_ R _[d][f]_, and _b_ 2 _2_ R _[d][hid]_ . In addition, there are residual
connection and layer normalization on top of MHA and FFN (denoted by _⊕_ in Figure 2), which are
formulated as LayerNorm(x + MHA(x)) and LayerNorm(x + FFN(x)), respectively.


**3** **Few-shot Task-agnostic NAS**


Given a large pre-trained language model (e.g., BERT) as teacher, AutoDistil distills several
compressed models with variable computational cost with the following major components.


**3.1** **Search Space Design**


**Searchable transformer components.** From Transformers overview (Section 2) and our framework
(Figure 2), we observe four important hyper-parameters for the Transformer blocks to include:
(1) Feed-forward network (FFN) dimension - we encode this by the MLP (multi-layer perceptron)
_df_
ratio defined as _r_ = _dhid_ [, with] _[ d][f]_ [ and] _[ d][hid]_ [ representing the intermediate dimension of the FFN and]
hidden dimension respectively; (2) Number of layers ( _L_ ) to capture the network depth; (3) Hidden
dimension ( _dhid_ ) to encode input; (4) Attention heads ( _H_ ) for multi-head self-attention.

All of the above factors are important for model capacity and have a significant impact on the model
size and computational cost. For instance, different layers have different feature representation
capabilities. Recent works show that Transformer models are overparameterized [21, 22], such as
the feed-forward layer (FFN), which is one of the most computation intensive components [23].
Therefore, we search for optimal MLP ratio and hidden dimension that reduce computational cost
resulting from FFN layers. Furthermore, studies [24, 25] show that attention heads can be redundant
when they learn to encode similar relationships for each word. Thus, we make the number of attention
heads searchable as well.


**Inductive bias.** Prior works on CNNs [26] and Transformers [27] demonstrate that thinner and deeper
neural networks with improved representation capacity perform better than wider and shallower ones.
We incorporate this as an inductive bias to decide the number of layers to consider for the students in
each of our _K_ sub-spaces (base, small, tiny), where we prefer deeper students in terms of the number

- f layers. Furthermore, we constrain all the Transformer layers in a given student model to share
identical and homogeneous structures, i.e., the same number of attention heads, hidden dimension,
etc. This not only reduces the size of the search space, it is also more friendly to hardware and
software frameworks [13].


**Search space partition.** Existing works [13, 14] train a single large SuperLM containing millions of
student architectures by weight-sharing. This leads to performance degradation due to optimization
interference and convergence of subnetworks with very different sizes [11]. To mitigate such


4




--- end of page=3 ---

interference, we employ a few-shot learning strategy [17, 16] as follows. We partition the whole
Transformer search space into _K_ sub-spaces such that each sub-space covers different sizes of student
models given by the number of parameters. For instance, _K_ = 3 can cover typical student sizes,
namely base, small and tiny versions. Table 2 shows the parameter ranges for the _K_ sub-spaces,
along with the student configurations contained in each. Extensive details on the search space design
can be found in the Appendix (Section D).


We now encode each sub-space into a SuperLM, where each student model in the space is a subnetwork of the SuperLM. Furthermore, all the student subnetworks share the weights of their common
dimensions, with the SuperLM being the largest one in the search space. Considering _K_ independent
SuperLMs, each one now has more capacity to encode a sub-space, in contrast to a limited capacity
single SuperLM as in prior works. Furthermore, our choices for the heuristic partition and inductive
bias result in less number of student models of comparable size in each sub-space which alleviates
conflicts in weight-sharing.

We extract student subnet- SuperLMTiny SuperLMSmall SuperLMBase BERT
works from the SuperLM #Subnets 256 256 256 N/A
by a simple truncation strat- #Layers (4, 7, 1) (9, 12, 1) (9, 12, 1) 12
egy like bottom-left extrac- #Hid_dim (128, 224, 32) (256, 352, 32) (544, 640, 32) 768
tion. In particular, given MLP Ratio (2.0, 3.5, 0.5) (2.5, 4.0, 0.5) (2.5, 4.0, 0.5) 4.0
a specific architecture _↵_ = #Heads (7, 10, 1) (7, 10, 1) (9, 12, 1) 12
_{l, dhid, r, h}_, (i) we first #FLOPs 40-367 _M_ 0.5-2.1 _G_ 2.1-7.9 _G_ 11.2 _G_
extract alternate _l_ Trans- #Params 4-10 _M_ 12-28 _M_ 39-79 _M_ 109 _M_
former layers from the SuperLM; (ii) then extract Table 2: The search space of AutoDistil with _K_ =3 partitions, each
bottom-left sub-matrices in consisting of 256 subnets with variable computational cost. We train
terms of _dhid_ and _r_ from the - ne SuperLM with weight-sharing for _each partition_ with child models

- riginal matrices that repre- sharing transformer blocks. Each tuple represents the lowest value,
sent the hidden dimension highest value, and steps for each factor.
and the MLP ratio respectively; (iii) finally, for the attention heads, we extract the leftmost _h_ heads and retain the dimension

- f each head as the SuperLM. This strategy is used during sampling sub-networks for SuperLM
training via weight sharing after which the sub-network weights are updated via self-attention relation
distillation. There can be better strategies to extract sub-networks (e.g., ordering heads by importance)
that we defer to future work. We did explore some strategies for selecting layers (alternate vs. top vs.
bottom) with results in Section B of Appendix.


**3.2** **Task-agnostic SuperLM Training**


We illustrate SuperLM training process in Algorithm 1. Given a large pre-trained language model (e.g.,
BERT) as teacher, we initialize the SuperLM with the weights of teacher. In each step of SuperLM
training, we randomly sample several student subnetworks from the search space; apply knowledge
distillation between sampled subnetworks and the teacher to accumulate gradients; and then update
the SuperLM. During sampling, we employ Sandwich rule [28], also used in BigNAS [11], that
samples the smallest subnetwork, the largest subnetwork and _M_ random ones for updating SuperLM.
The motivation is to improve the performance of all subnetworks by increasing the performance
lower bound (smallest subnetwork) and upper bound (largest one) across all subnetworks.


We leverage deep self-attention distillation [4] for task-agnostic training. To this end, we employ
multi-head self-attention relation distillation to align the attention distributions as well as scaled
dot-product of keys, queries and values of the teacher and sampled student subnetworks. Consider
**A** 1, **A** 2, **A** 3 to denote the queries, keys and values of multiple relation heads of teacher model,
and **B** 1, **B** 2, **B** 3 respectively for a sampled subnetwork. Mean squared error (MSE( _·_ )) between
multi-head self-attention relation of teacher and sampled subnetwork is used as distillation objective:



SuperLMTiny SuperLMSmall SuperLMBase BERT



#Subnets 256 256 256 N/A
#Layers (4, 7, 1) (9, 12, 1) (9, 12, 1) 12
#Hid_dim (128, 224, 32) (256, 352, 32) (544, 640, 32) 768
MLP Ratio (2.0, 3.5, 0.5) (2.5, 4.0, 0.5) (2.5, 4.0, 0.5) 4.0
#Heads (7, 10, 1) (7, 10, 1) (9, 12, 1) 12



#FLOPs 40-367 _M_ 0.5-2.1 _G_ 2.1-7.9 _G_ 11.2 _G_

#Params 4-10 _M_ 12-28 _M_ 39-79 _M_ 109 _M_



Table 2: The search space of AutoDistil with _K_ =3 partitions, each
consisting of 256 subnets with variable computational cost. We train

- ne SuperLM with weight-sharing for _each partition_ with child models
sharing transformer blocks. Each tuple represents the lowest value,
highest value, and steps for each factor.



_H_
X


_k_ =1



_ik_ [)] _[,]_ (4)



_L_ =



3
X


_i_ =1



_βiLi, Li_ = [1]

_H_



MSE( **R** _[T]_




_[T]_ _ik_ _[,]_ **[ R]** _[S]_




_[T]_ _i_ [= softmax(] **[A]** _[i]_ **[A]** _[>]_ _i_




_[S]_ _i_ [= softmax(] **[B]** _[i]_ **[B]** _[>]_ _i_



where **R** _[T]_




_[>]_ _i_ _[/][p][d][k]_ [)][,] **[ R]** _[S]_ _i_



where **R** _[T]_ _i_ [= softmax(] **[A]** _[i]_ **[A]** _[>]_ _i_ _[/][p][d][k]_ [)][,] **[ R]** _[S]_ _i_ [= softmax(] **[B]** _[i]_ **[B]** _[>]_ _i_ _[/][p][d][k]_ [)][,] _[ H]_ [ is the number of attention]

heads; **R** _[T]_ _i_ [represents the teacher’s] _[ Q][ −]_ _[Q]_ [,] _[ K][ −]_ _[K]_ [, or] _[ V][ −]_ _[V]_ [ relation;] **[ R]** _[S]_ _i_ [represents the same for]

student. **R** _[T]_ [is the relation information based on one attention head, and] _[ d]_ [ is the attention head size.]




_[T]_ _i_ [represents the teacher’s] _[ Q][ −]_ _[Q]_ [,] _[ K][ −]_ _[K]_ [, or] _[ V][ −]_ _[V]_ [ relation;] **[ R]** _[S]_ _i_



_ik_ [is the relation information based on one attention head, and] _[ d][k]_ [ is the attention head size.]



5




--- end of page=4 ---

Relation knowledge distillation avoids the introduction of additional parameters to transform the
student’s representations with different dimensions to align to that of the teacher. For the teacher
model and subnetworks with different number of attention heads, we first concatenate the selfattention vectors of different attention heads of the subnetwork and then split them according to the
number of relation heads of the teacher model. Then, we align their queries with the same number of
relation heads for distillation. In addition, we only transfer the self-attention knowledge from the last
layer of the teacher model to the last layer of the student model.



The SuperLM for sub-space _Ak_ is trained as:


_**W**_ _A_ _[⇤]_ _k_ [=] _[ argmin]_ _**[W]**_ [E] _[↵][2A]_ [[] _[L]_ [(] _**[W]**_ _[↵]_ [;] _**[ U]**_ [;] _[ D][train]_ [)]] _[,]_

(5)


where, _K_ is the number of sub-space partitions; _**W**_ are the weights of the SuperLM;
_**W**_ _↵_ are the weights in _**W**_ specified by the architecture _↵_ ; _**U**_ are the weights of the teacher
model including the self-attention module
used for distillation; _Dtrain_ is the training data
set, and _L_ ( _·_ ) is the self-attention loss function
from Eqn. (4).


**3.3** **Lightweight Optimal Student Search**


We outline two search strategies for selecting
the optimal student subnetwork.



Randomly sample a subnetwork _s_ from _Sk_
Calculate self-attention distil. loss between
subnetwork _s_ and teacher _T_ with Eqn. (4)
Accumulate gradients
**end for**
Update _Sk_ with the accumulated gradients
**end for**

**end for**

**end for**



**Algorithm 1** Few-shot Task-agnostic Knowledge
Distillation with AutoDistil .


**Input:** Partitioned _K_ sub-spaces _Ak_ ; initialized _K_ SuperLMs _Sk_  - n _Ak_ ; pre-trained teacher _T_ ; unlabeled
data _D_ ; training epochs _E_ ; sampling steps _M_
**Output:** Trained SuperLMs _{Sk}_
**for** _k_ = 1 **to** _K_ **do**


**for** _i_ = 1 **to** _E_ **do**


Get a batch of data from _D_

**for** _batch_ in _D_ **do**

Clear gradients in SuperLM _Sk_
**for** _m_ = 1 **to** _M_ **do**



**Task-agnostic search.** We adopt this to be our

**end for**

primary strategy to compare against all base
**end for**

lines since it does not access any task label
information. We compute the task-agnostic self-attention distillation loss for all student subnetworks
using Eqn. (4) on a heldout validation set from the unlabeled training corpus. The student subnetworks are directly obtained by bottom-left extraction from the well-trained SuperLM (outlined in
Section 3.1). This process is lightweight since it does not require any training or adaptation of the
student and number of subnetworks is limited. The optimal student is given by the subnetwork with
least validation loss subject to following constraint.



_A_ _[⇤]_ [=] _[ argmin][↵][2A]_ 1 _,_ 2 _,···K_ _[L]_ [(] _**[W]**_ _↵_ _[ ⇤]_



_↵_ _[⇤]_



_↵_ _[ ⇤]_ [;] _[ D][val]_ [)] _[,]_ _s.t._ _g_ ( _↵_ ) _< c,_ (6)



where _**W**_ _↵_ _[⇤]_ [is the weights of architecture] _[ ↵]_ [obtained from] _**[ W]**_ _[ ⇤]_ _Ak_ [,] _[ D][val]_ [ is the validation data set,] _[ L]_


_·_
is the self-attention distillation loss, and _g_ ( ) is a function to calculate the computational cost (e.g.,
#FLOPs, #parameters) of the subnetwork subject to a given user-specified resource constraint _c_ .



where _**W**_ _[⇤]_



_↵_ _[⇤]_ [is the weights of architecture] _[ ↵]_ [obtained from] _**[ W]**_ _[ ⇤]_



**Task-proxy search.** We compare our task-agnostic search against another strategy that considers a
proxy task (e.g., MNLI [29]) with label information to fine-tune the 256 candidate subnetworks in
each sub-space. The optimal student in each sub-space is given by the one with the best downstream
task performance (e.g., accuracy). Note that, for this strategy, the proxy task is used only during
search while the NAS training is still fully task-agnostic.


**4** **Experiments**


**Datasets.** We conduct experiments on General Language Understanding Evaluation (GLUE) benchmark [30]. We compare our method with the baseline methods on two single-sentence classification
tasks (CoLA [31], SST-2 [32]), two similarity and paraphrase tasks (MRPC [33], QQP [34]), and
three inference tasks (MNLI [29], QNLI [35], RTE [36, 37, 38, 39]) [3] . We report accuracy for MNLI,
QNLI, QQP, SST-2, RTE, report f1 for MRPC, and report Matthew’s correlation for CoLA.
**Baselines.** We compare against several _task-agnostic methods_ [4] generating compressed models from
BERTbase teacher, using (i) knowledge distillation like BERTSMALL [40], Truncated BERT [29],
DistilBERT [5], TinyBERT [6], MINILM [4]; as well as those based on Neural Architecture Search,
like AutoTinyBERT [13], DynaBERT [12], and NAS-BERT [14].


3We ignore STS-B for a fair comparison with our strongest KD baseline MINILM [4] that do not report it.
4For a fair comparison, we do not include MobileBERT [7] that uses BERTlarge as teacher.


6




--- end of page=5 ---

**AutoDistil configuration.** We use uncased BERTBASE as the teacher consisting of 12 Transformer layers, 12 attention heads; with the hidden dimension and MLP ratio being 768 and 4,
respectively. It consists of 109 _M_ parameters with 11 _._ 2 _G_ FLOPs. We use English Wikipedia and
BookCorpus data for SuperLM training with WordPiece tokenization. We use a batch size of 128
and 4 _e_ - 5 as the peak learning rate for 10 epochs. The maximum sequence length is set to 128.
The coefficients in distillation objective (Eqn. (4)), _β_ 1, _β_ 2, and _β_ 3, are all set to 1. We distill the
self-attention knowledge of the last layer to train the SuperLM. Both the teacher and SuperLM are
initialized with pre-trained BERTBASE. Other hyper-parameter settings are shown in Appendix. We
use 16 _V_ 100 GPUs to train the SuperLM with 336 GPU-hours as the training cost.


**4.1** **Finding the Optimal Compressed Models**


**AutoDistil** Agnostic is obtained by fully task-agnostic training and task-agnostic search without
using any task label information. We set a constraint in Eqn. (6) such that the #FLOPs of the optimal
compressed model is atleast 50% less than the teacher model. We rank all the subnetworks contained
in all the partitions of the trained SuperLM by their self-attention distillation loss on the heldout
validation set, and select the one that meets the constraint with the minimum loss.
**AutoDistil** Proxy uses MNLI [29] as a proxy to estimate downstream task performance of different
subnetworks. Prior work [41] has demonstrated performance improvements in MNLI to be correlated
to other GLUE tasks. To this end, we fine-tune all the 256 subnetworks in each partition of the
trained superLMs, and select corresponding subnetworks with the best trade-off (more discussions

- n the optimal trade-off can be found in the Appendix (Section G)) between task performance
(accuracy) and computational cost (#FLOPs). This results in _K_ =3 optimal students, corresponding to
AutoDistilProxyB, AutoDistilProxyS and AutoDistilProxyT obtained from the corresponding
sub-spaces of SuperLMBase, SuperLMSmall and SuperLMTiny, respectively. Notably all students are

- btained from the AutoDistil SuperLM still trained in a fully task-agnostic fashion.


**4.1.1** **Comparison with Traditional Knowledge Distillation Baselines**


We compare AutoDistil against state-of-the-art KD models distilled from the same teacher
BERTBASE in Table 3 with respect to the following measures: computational cost in the form of (i)
FLOPs and (ii) parameters, along with (iii) improvement in the average task performance aggregated

- ver all the GLUE tasks. We observe that the compressed model AutoDistilAgnostic generated
via our task-agnostic SuperLM training leads to upto 3 _x_ reduction in FLOPs over state-of-the-art
distilled models (e.g., MINILM [4], TinyBERT [6], DistilBERT [5]) that are hand-engineered while
matching the overall task performance. The most aggressive compressed version corresponding to
AutoDistilProxyT obtains a massive 41 _x_ reduction in FLOPs over BERTBASE while incurring
5 point accuracy drop in GLUE (excluding CoLA) and 10 point drop (including CoLA). Notably
CoLA is a syntactic task in contrast to other semantic tasks in the benchmark like natural language
inference, paraphrase detection and sentiment classification. This depicts an interesting impact of
massive model compression on varying task types.


**4.1.2** **Comparison with Neural Architecture Search Baselines**



We report the performance of several NASgenerated student models of comparable FLOPs
and parameters from corresponding papers in
Table 3. AutoDistil outperforms all competing methods on aggregate for all sizes; except
for small-sized model; where it has marginally
lower performance (0.1 points on avg) compared
to AutoTinyBERT. It is worthwhile to note that
computational cost of training process is another
important dimension for comparing methods.
This is especially important when comparing
to NAS methods that use multi-stage training;
where additional pre-training and distillation is
applied to NAS-generated candidates.


To better understand the impact of single-stage
vs. multi-stage methods on the training cost, we



Table 4: Training cost (V100 GPU hours) comparison for generating students of similar FLOPs.
_NR_ AutoTiny- BERT does not report the cost of SuperNet training - typically the most expensive step.
Further Training refers to additional pre-training
applied to NAS-generated candidates.


7



Cost (GPU AutoTiny AutoTiny Auto
hours) BERT BERT-Fast Distil



SuperNet
Training



_NR_ _NR_ 336



Search 150 12 <1



Further
Training



870 290 0




--- end of page=6 ---

Table 3: Performance comparison between students from traditional task-agnostic distillation; multistage one-shot NAS with additional pre-training, distillation; and single-stage few-shot AutoDistil
Our results are averaged over 5 runs with baselines reported from corresponding papers.


Model #FLOPs #Para MNLI-m QNLI QQP SST-2 CoLA MRPC RTE Average
(Metric) (G) (M) (Acc) (Acc) (Acc) (Acc) (Mcc) (Acc) (Acc)


BERTBASE [1] (teacher) 11.2 109 84.5 91.7 91.3 93.2 58.9 87.3 68.6 82.2


Base _-sized Models from Task-agnostic KD Methods and AutoDistil_
BERTSMALL [40] 5.66 66.5 81.8 89.8 90.6 91.2 53.5 84.9 67.9 80.0
Truncated BERT [29] 5.66 66.5 81.2 87.9 90.4 90.8 41.4 82.7 65.5 77.1
DistilBERT[5] 5.66 66.5 82.2 89.2 88.5 91.3 51.3 87.5 59.9 78.6
TinyBERT [6] 5.66 66.5 83.5 90.5 90.6 91.6 42.8 88.4 72.2 79.9
MINILM [4] 5.66 66.5 84.0 91.0 91.0 92.0 49.2 88.4 71.5 81.0
AutoDistilProxyB 4.40 50.1 83.8 90.8 91.1 91.1 55.0 88.8 71.9 81.7


Small _-sized Models from Multi-stage One-shot NAS Methods and AutoDistil_
AutoTinyBERT-KD-S1 [13] 1.69 30.0 82.3 89.7 89.9 91.4 47.3 88.5 71.1 80.0
DynaBERT [12] 1.81 37.7 82.3 88.5 90.4 92.0 43.7 81.4 63.2 77.4
NAS-BERT10 [14] 2.30 10.0 76.4 86.3 88.5 88.6 34.0 79.1 66.6 74.2
AutoDistilProxyS 2.02 26.1 83.2 90.0 90.6 90.1 48.3 88.3 69.4 79.9
AutoDistilAgnostic 2.13 26.8 82.8 89.9 90.8 90.6 47.1 87.3 69.0 79.6


Tiny _-sized Models from Multi-stage One-shot NAS Methods and AutoDistil_
AutoTinyBERT-KD-S4 [13] 0.30 10.1 76.0 85.5 86.9 86.8 20.4 81.4 64.9 71.7
NAS-BERT5 [14] 0.87 5.00 74.4 84.9 85.8 87.3 19.8 79.6 66.6 71.2
AutoDistilProxyT 0.27 6.88 79.0 86.4 89.1 85.9 24.8 78.5 64.3 72.6


compare the overall cost of NAS for AutoDistil and that reported in AutoTinyBERT [5] for the
small model segment in Table 4. AutoDistil is much cheaper due to its single-stage training protocol;
where no additional pre-training or distillation is needed. It is worth noting that the overall SuperNet
training cost of AutoDistil (the most expensive component of NAS) is less or comparable to the
additional training cost of re-training candidate models for AutoTinyBERT. Note that AutoTinyBERT
does not report their SuperNet training cost. Additionally, AutoDistil has a much faster search
mechanism due to (1) inductive biases built into the search space definition to limit the number

- f student architectures and (2) task-agnostic search that only requires computing self-attention
validation loss without the need for any training.


Finally, we show the pareto frontier of student subnetworks generated by several KD and NAS
methods in Figure 3 for the MNLI task. The blue points represent all the subnetworks extracted from
AutoDistil and red points denote the optimal ones, all fine-tuned on the MNLI task. We observe
the optimal AutoDistil models to outperform several competing methods.


**4.1.3** **Task-agnostic Training Strategies**



We study different task-agnostic strategies for SuperLM train- Table 5: Comparing task-agnostic
ing in AutoDistil . Specifically, we compare three strate- SuperLM training strategies.
gies in Table 5. (i) We replacing the KD loss in Eqn. (4) with
masked language modeling (MLM) loss [1] to calculate gradi- Strategy MRPC RTE MNLI
ents which is the most widely used task-agnostic pre-training MLM 89.4 68.2 82.2
and distillation strategy. (ii) KD _att_ +Cont further continues KD _att_ +Cont. 91.0 71.8 83.5
training the searched compressed models on the large language KD _att_ 91.2 71.5 83.2
corpus. (iii) KD _att_ is the strategy adopted in AutoDistil
for self-attention distillation. We evaluate subnetworks with the same architecture (6 layers, 768
hidden, 12 heads, MLP ratio 4) from the trained SuperLM. We fine-tune subnetworks on MRPC,
RTE, and MNLI tasks, and report f1, accuracy, and accuracy, respectively. MRPC and RTE are
paraphrase detection and natural language inference tasks, respectively. They are low-resource
tasks with limited training examples ( 3 _K_ labels each) for fine-tuning. This allows us to evaluate
transferability of the compressed models trained on unlabeled general domain (e.g., Wikipedia)
and fine-tuned on downstream tasks with limited labels. MNLI has a larger number of examples



Table 5: Comparing task-agnostic
SuperLM training strategies.



Strategy MRPC RTE MNLI



MLM 89.4 68.2 82.2
KD _att_ +Cont. 91.0 71.8 83.5
KD _att_ 91.2 71.5 83.2



5Other NAS methods either use a different hardware for training or do not report the cost.


8




--- end of page=7 ---

(a) Acc vs #FLOPs (SuperLMBase).(b) Acc vs #FLOPs
(SuperLMSmall).



(c) Acc vs #FLOPs (SuperLMTiny).



(d) Acc vs #Para (SuperLMBase). (e) Acc vs #Para (SuperLMSmall). (f) Acc vs #Para (SuperLMTiny).


### Figure 3

Caption: Computational cost vs. task (MNLI) performance trade-off for all 256 subnetworks

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
contained in each of _K_ SuperLMs (base, small and tiny). 3(a)-3(c) show the trade-off between
accuracy (Y-axis) and #FLOPs (X-axis), and 3(d)-3(f) show the trade-off between accuracy (Y-axis)
and #Para (X-axis) with optimal compressed AutoDistil student for each SuperLM in red.


(400 _K_ ). Prior work [41] has shown MNLI to transfer well to other tasks in GLUE. First, we observe
self-attention distillation to perform better than MLM, for SuperLM training. Second, we observe
limited performance gains with continued training of the optimal subnetworks from NAS as done in
existing works **demonstrating the effectiveness of our single-stage training protocol** .


**4.1.4** **One-shot vs. Few-shot NAS with Varying** _K_ Table 6: Search space design strategies.


For few-shot NAS, we choose _K_ =3 (i.e. 3 sub
Search Space Size (#subnetworks)

spaces) for following reasons: (1) 3 sub-spaces corre- Task One-shot ( _K_ = 1) Few-shot ( _K_ = 3)
spond to base, small and tiny model sizes; as used in

27 864 11232 256*3

prior work in CV; e.g. AutoFormer [17], (2) searching over different values of _K_ is resource-extensive MRPC 88.2 87.5 85.1 91.2
since it requires training _K_ SuperLMs for each choice RTE 67.2 64.5 62.8 71.8

- f _K_, (3) As _K_ increases, search process becomes
similar to the undesirable brute-force discrete search that trains all models in search space individually.


To understand the effect of few-shot vs. one-shot NAS, we compare the performance of a single space
( _K_ = 1) to multiple sub-spaces ( _K_ = 3). We extract subnetworks with same architecture (6 layers,
768 hidden, 12 heads, MLP ratio 4) from trained SuperLMs for each strategy for evaluation with
results in Table 6. For one-shot NAS, we consider a single search space containing varying number of
subnetworks (e.g., 27, 864, 11232). Few-shot NAS contains 256 subnetworks in each partition. We
fine-tune subnetworks on RTE and MRPC tasks, and report accuracy and f1 respectively. We observe
fewer subnetworks in a single search space for one-shot NAS result in better performance. This
results from optimization interference and gradient conflicts as the number and size of subnetworks
increase in the space. Finally, our design strategy performs the best while containing lesser number

- f subnetworks **demonstrating the benefit of few-shot NAS for language model distillation** .


**4.1.5** **Comparing Search Strategies and Optimal Architectures**


From Table 3, we observe that the student models AutoDistilAgnostic and AutoDistilProxyS

- btained from SuperLMsmall by task-agnostic and task-proxy search strategies respectively obtain a
similar trade-off between performance and cost. The task-proxy search results in a minor performance
gain 0 _._ 3 over the fully task-agnostic search mechanism. Table 7 shows the configuration of searched

- ptimal architectures from AutoDistil with corresponding computational cost. For reference, we
also show the architecture of the teacher BERTBASE and a state-of-the-art distilled model MINILM [4]
that are hand-engineered. We observe that the obtained architectural hyper-parameters are quite


9



Search Space Size (#subnetworks)
Task
One-shot ( _K_ = 1) Few-shot ( _K_ = 3)



27 864 11232 256*3



MRPC 88.2 87.5 85.1 91.2

RTE 67.2 64.5 62.8 71.8




--- end of page=8 ---

non-standard and difficult to obtain by Table 7: Architecture comparison between the optimal comtrial and error considering the large pressed students searched by AutoDistil with state-ofspace of Transformer architectures. the-art hand-engineered students distilled from BERTBASE.
We also observe that optimal compressed models have thin-and-deep Model #Layers #Hid Ratio #Heads #FLOPs #Para
structure consistent with findings that BERTBASE 12 768 4 12 11.2G 109M
thinner and deeper models perform MINILM 6 768 4 6 5.66G 66.5M
better [26] than wider and shallower

- nes. While we use this as an induc- AutoDis.Agnostic 11 352 4 10 2.13G 26.8M
tive bias for sub-space partitioning,our search space (Table 2) also con-tains diverse subnetworks with differ- AutoDis.AutoDis.AutoDis.ProxyProxyProxyBST 12117 160544352 3.534 1098 0.27G 6.88M4.40G 50.1M2.02G 26.1M
ent depth and width. Non-maximal MLP ratio and attention heads for optimal compression indicate
that self-attention and feed-forward layers of Transformers are overparameterized [21, 22].



Table 7: Architecture comparison between the optimal compressed students searched by AutoDistil with state-ofthe-art hand-engineered students distilled from BERTBASE.



Model #Layers #Hid Ratio #Heads #FLOPs #Para


BERTBASE 12 768 4 12 11.2G 109M
MINILM 6 768 4 6 5.66G 66.5M


AutoDis.Agnostic 11 352 4 10 2.13G 26.8M
AutoDis.ProxyB 12 544 3 9 4.40G 50.1M
AutoDis.ProxyS 11 352 4 8 2.02G 26.1M
AutoDis.ProxyT 7 160 3.5 10 0.27G 6.88M



**5** **Related Work**


**Task-specific knowledge distillation.** Knowledge distillation (KD) [42] is a widely used technique
for model compression, which transfers knowledge from a large teacher to a smaller student. Taskspecific KD aims to generate smaller students by using downstream task label information. Typical
task-specific KD works include BERT-PKD [43], BERTSMALL [40], TinyBERT [6], DynaBERT [12],
and SparseBERT [44]. While task-specific KD often achieves good task performance, a typical
drawback is that it is resource-consuming to run KD for each and every task, and also not scalable.
**Task-agnostic knowledge distillation.** In contrast to task-specific KD, we explore task-agnostic KD
that does not use any task label information. The distilled task-agnostic models can be re-used by
simply fine-tuning on downstream tasks. Task-agnostic KD leverages knowledge from soft target
probabilities, hidden states, layer mappings and self-attention distributions of teacher to train student
models. Typical task-agnostic KD works include DistilBERT [5], MobileBERT [7], and MiniLM [4].
MobileBERT assumes that students have the same number of layers as the teacher for layer-by-layer
distillation. MiniLM transfers self-attention knowledge from the last layer of the teacher to that of the
student. These works rely on hand-designed architecture for the student models for KD that requires
several trials, and needs to be repeated for a new student with a different cost. In contrast, we develop
techniques to automatically design and distill several student models with variable cost using NAS.
**Neural Architecture Search.** While NAS has been extensively studied in computer vision [8,
9, 10, 11], there has been relatively less exploration in natural language processing. Evolved
Transformer [45] and HAT [46] search for efficient sub-networks from the Transformer architecture
for machine translation tasks. Some recent approaches closest to our method include, DynaBERT [12],
AutoTinyBERT [13] and NAS-BERT [14]. DynaBERT performs task-specific distillation. NASBERT performs two-stage knowledge distillation with pre-training and fine-tuning of candidates.
Similar to above approaches, AutoTinyBERT also employs one-shot NAS with a single large search
space containing millions of subnetworks that result in co-adaption and weight-sharing challenges
for SuperLM training. Further it also uses a multi-stage training protocol for further pre-training
and distillation of the NAS-generated candidates. In contrast, AutoDistil employs few-shot NAS
with a compact search space design with a single-stage task-agnostic training protocol. This further
allows us to do a lightweight search for the optimal student without re-training. [16] studies few-shot
task-specific NAS in the CV domain with CNN architectures, and different design and search spaces
from AutoDistil . More comparisons between them can be found in the Appendix (Section A.2).


**6** **Conclusion**


We develop a few-shot task-agnostic NAS method, namely AutoDistil to distil large language
models into compressed students with variable computational cost. To address the co-adaption
and weight-sharing challenges for SuperLM training, we partition the Transformer search space
into _K_ compact sub-spaces covering important architectural components like its depth, width, and
attention heads. We leverage self-attention distillation for fully task-agnostic SuperLM training and
lightweight optimal search without any re-training. Obtained students can be re-used by simply
fine-tuning on tasks. AutoDistil generates students with 3 _x_ less computational cost (FLOPs)
than state-of-the-art task-agnostic KD methods while obtaining a similar task performance in GLUE.


10




--- end of page=9 ---

## References


[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirec
tional transformers for language understanding. In _NAACL_, pages 4171–4186, Minneapolis, Minnesota,
June 2019. Association for Computational Linguistics.


[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind

Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens
Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark,
Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models
are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors,
_Advances in Neural Information Processing Systems_, volume 33, pages 1877–1901. Curran Associates,

Inc., 2020.


[3] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep

learning in NLP. In _ACL_, pages 3645–3650, Florence, Italy, July 2019. Association for Computational
Linguistics.


[4] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep self-attention

distillation for task-agnostic compression of pre-trained transformers. In H. Larochelle, M. Ranzato,
R. Hadsell, M. F. Balcan, and H. Lin, editors, _Advances in Neural Information Processing Systems_,
volume 33, pages 5776–5788. Curran Associates, Inc., 2020.


[5] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert:

smaller, faster, cheaper and lighter. _arXiv preprint arXiv:1910.01108_, 2019.


[6] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu.

Tinybert: Distilling bert for natural language understanding. In _Proceedings of the 2020 Conference on_
_Empirical Methods in Natural Language Processing: Findings_, pages 4163–4174, 2020.


[7] Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. Mobilebert: a

compact task-agnostic bert for resource-limited devices. In _Proceedings of the 58th Annual Meeting of the_
_Association for Computational Linguistics_, pages 2158–2170, 2020.


[8] Hieu Pham, Melody Guan, Barret Zoph, Quoc Le, and Jeff Dean. Efficient neural architecture search via

parameters sharing. In _International Conference on Machine Learning_, pages 4095–4104. PMLR, 2018.


[9] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V

Le. Mnasnet: Platform-aware neural architecture search for mobile. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, pages 2820–2828, 2019.


[10] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once for all: Train one network and

specialize it for efficient deployment. In _International Conference on Learning Representations_, 2020.


[11] Jiahui Yu, Pengchong Jin, Hanxiao Liu, Gabriel Bender, Pieter-Jan Kindermans, Mingxing Tan, Thomas

Huang, Xiaodan Song, Ruoming Pang, and Quoc Le. Bignas: Scaling up neural architecture search with
big single-stage models. In _European Conference on Computer Vision_, pages 702–717. Springer, 2020.


[12] Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu. Dynabert: Dynamic bert with

adaptive width and depth. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors,
_Advances in Neural Information Processing Systems_, volume 33, pages 9782–9793. Curran Associates,

Inc., 2020.


[13] Yichun Yin, Cheng Chen, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu. AutoTinyBERT: Automatic

hyper-parameter optimization for efficient pre-trained language models. In _Proceedings of the 59th Annual_
_Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on_
_Natural Language Processing (Volume 1: Long Papers)_, pages 5146–5157. Association for Computational
Linguistics, August 2021.


[14] Jin Xu, Xu Tan, Renqian Luo, Kaitao Song, Jian Li, Tao Qin, and Tie-Yan Liu. NAS-BERT: task-agnostic

and adaptive-size BERT compression with neural architecture search. In Feida Zhu, Beng Chin Ooi, and
Chunyan Miao, editors, _KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data_
_Mining, Virtual Event, Singapore, August 14-18, 2021_, pages 1933–1943. ACM, 2021.


[15] Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding

and simplifying one-shot architecture search. In _International Conference on Machine Learning_, pages
550–559. PMLR, 2018.


11




--- end of page=10 ---

[16] Yiyang Zhao, Linnan Wang, Yuandong Tian, Rodrigo Fonseca, and Tian Guo. Few-shot neural architecture

search. In _International Conference on Machine Learning_, pages 12707–12718. PMLR, 2021.


[17] Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for

visual recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages
12270–12280, 2021.


[18] Tianyi Chen, Bo Ji, Tianyu Ding, Biyi Fang, Guanyi Wang, Zhihui Zhu, Luming Liang, Yixin Shi, Sheng

Yi, and Xiao Tu. Only train once: A one-shot neural network training and pruning framework. _Advances_

_in Neural Information Processing Systems_, 34, 2021.


[19] Jing Xu, Yu Pan, Xinglin Pan, Steven Hoi, Zhang Yi, and Zenglin Xu. Regnet: Self-regulated network for

image classification. _IEEE Transactions on Neural Networks and Learning Systems_, 2022.


[20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz

Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in neural information processing_
_systems_, pages 5998–6008, 2017.


[21] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In _NeurIPS_,

pages 14014–14024, 2019.


[22] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self
attention: Specialized heads do the heavy lifting, the rest can be pruned. _arXiv preprint arXiv:1905.09418_,
2019.


[23] Prakhar Ganesh, Yao Chen, Xin Lou, Mohammad Ali Khan, Yin Yang, Deming Chen, Marianne Winslett,

Hassan Sajjad, and Preslav Nakov. Compressing large-scale transformer-based models: A case study on
bert. _arXiv preprint arXiv:2002.11985_, 2020.


[24] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In H. Wallach,

H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, _Advances in Neural_
_Information Processing Systems_, volume 32. Curran Associates, Inc., 2019.


[25] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self
attention: Specialized heads do the heavy lifting, the rest can be pruned. In _Proceedings of the 57th Annual_
_Meeting of the Association for Computational Linguistics_, pages 5797–5808, Florence, Italy, July 2019.
Association for Computational Linguistics.


[26] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua

Bengio. Fitnets: Hints for thin deep nets. In Yoshua Bengio and Yann LeCun, editors, _3rd International_
_Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference_
_Track Proceedings_, 2015.


[27] Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and Joey Gonzalez. Train big,

then compress: Rethinking model size for efficient training and inference of transformers. In _ICML_, pages
5958–5968. PMLR, 2020.


[28] Jiahui Yu and Thomas S Huang. Universally slimmable networks and improved training techniques. In

_Proceedings of the IEEE/CVF international conference on computer vision_, pages 1803–1811, 2019.


[29] Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence

understanding through inference. In _Proceedings of the 2018 Conference of the North American Chapter of_
_the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_,
pages 1112–1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.


[30] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE:

A multi-task benchmark and analysis platform for natural language understanding. In _Proceedings of_
_the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP_, pages
353–355, Brussels, Belgium, November 2018. Association for Computational Linguistics.


[31] Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. Neural network acceptability judgments, 2018.


[32] Richard Socher et al. Recursive deep models for semantic compositionality over a sentiment treebank.

In _Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing_, pages
1631–1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics.


[33] William B Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In

_Proceedings of the Third International Workshop on Paraphrasing (IWP2005)_, 2005.


12




--- end of page=11 ---

[34] Zihan Chen, Hongbo Zhang, Xiaoji Zhang, and Leqi Zhao. Quora question pairs. _URL https://www. kaggle._

_com/c/quora-question-pairs_, 2018.


[35] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for

machine comprehension of text. _arXiv preprint arXiv:1606.05250_, 2016.


[36] Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge.

In _Machine Learning Challenges Workshop_, pages 177–190. Springer, 2005.


[37] R Bar Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor.

The second pascal recognising textual entailment challenge. In _Proceedings of the Second PASCAL_
_Challenges Workshop on Recognising Textual Entailment_, 2006.


[38] Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and William B Dolan. The third pascal recognizing

textual entailment challenge. In _Proceedings of the ACL-PASCAL workshop on textual entailment and_
_paraphrasing_, pages 1–9, 2007.


[39] Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth pascal recognizing textual

entailment challenge. In _TAC_, 2009.


[40] Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: On the

importance of pre-training compact models. _arXiv preprint arXiv:1908.08962_, 2019.


[41] Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, and Michael

Carbin. The lottery ticket hypothesis for pre-trained bert networks. In H. Larochelle, M. Ranzato,
R. Hadsell, M. F. Balcan, and H. Lin, editors, _Advances in Neural Information Processing Systems_,
volume 33, pages 15834–15846. Curran Associates, Inc., 2020.


[42] Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. In _NIPS_

_Deep Learning and Representation Learning Workshop_, 2015.


[43] Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for bert model compression.

In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th_
_International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_, pages 4323–4332,
2019.


[44] Dongkuan Xu, Ian EH Yen, Jinxi Zhao, and Zhibin Xiao. Rethinking network pruning–under the pre-train

and fine-tune paradigm. In _Proceedings of the Human Language Technology Conference of the NAACL_,
2021.


[45] David So, Quoc Le, and Chen Liang. The evolved transformer. In Kamalika Chaudhuri and Ruslan

Salakhutdinov, editors, _Proceedings of the 36th International Conference on Machine Learning_, volume 97

   - f _Proceedings of Machine Learning Research_, pages 5877–5886. PMLR, 09–15 Jun 2019.


[46] Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, and Song Han. Hat:

Hardware-aware transformers for efficient natural language processing. In _Annual Conference of the_
_Association for Computational Linguistics_, 2020.


[47] Hassan Sajjad, Fahim Dalvi, Nadir Durrani, and Preslav Nakov. On the effect of dropping layers of

pre-trained transformer models. _arXiv preprint arXiv:2004.03844_, 2020.


[48] John Wieting and Kevin Gimpel. ParaNMT-50M: Pushing the limits of paraphrastic sentence embeddings

with millions of machine translations. In _Proceedings of the 56th Annual Meeting of the Association for_
_Computational Linguistics (Volume 1: Long Papers)_, pages 451–462, Melbourne, Australia, July 2018.
Association for Computational Linguistics.


[49] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja

Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading
books. In _Proceedings of the IEEE international conference on computer vision_, pages 19–27, 2015.


[50] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey.

_International Journal of Computer Vision_, 129(6):1789–1819, 2021.


[51] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for

contrastive learning of visual representations. In _ICML_, pages 1597–1607. PMLR, 2020.


13




--- end of page=12 ---
