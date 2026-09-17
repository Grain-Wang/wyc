---
id: "2021_NAS_BERT"
title: "NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search"
authors: ["Jin Xu", "Xu Tan", "Renqian Luo", "Kaitao Song", "Jian Li", "Tao Qin", "Tie-Yan Liu"]
year: 2021
venue: "KDD 2021"
publication_status: "FORMALLY PUBLISHED"
category: "Direct Neighbor / Post-Training NAS"
source_pdf: "../reference_papers_origin/2021_NAS_BERT.pdf"
paper_url: "https://doi.org/10.1145/3447548.3467262"
pdf_url: "https://arxiv.org/pdf/2105.14444"
code_url: "UNVERIFIED"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search

**Authors:** Jin Xu, Xu Tan, Renqian Luo, Kaitao Song, Jian Li, Tao Qin, Tie-Yan Liu

**Venue / Year:** KDD 2021 (FORMALLY PUBLISHED)

**Category:** Direct Neighbor / Post-Training NAS

**Why this paper matters for low-cost post-training LLM NAS:** It provides a blockwise, task-agnostic Transformer supernet and performance approximation for adaptive-size pretrained-model compression.

**Primary record:** [https://doi.org/10.1145/3447548.3467262](https://doi.org/10.1145/3447548.3467262)

**Local source:** [2021_NAS_BERT.pdf](../reference_papers_origin/2021_NAS_BERT.pdf)

## Full converted text

## **NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression** **with Neural Architecture Search**

Jin Xu [1], Xu Tan [2], Renqian Luo [3], Kaitao Song [4], Jian Li [1], Tao Qin [2], Tie-Yan Liu [2]

1Institute for Interdisciplinary Information Sciences, Tsinghua University, China
3University of Science and Technology of China, 4Nanjing University of science and technology, China
2Microsoft Research Asia, China
j-xu18@mails.tsinghua.edu.cn,xuta@microsoft.com,lrq@mail.ustc.edu.cn,kt.song@njust.edu.cn
lijian83@mail.tsinghua.edu.cn,{taoqin,tyliu}@microsoft.com



## ABSTRACT


While pre-trained language models (e.g., BERT) have achieved impressive results on different natural language processing tasks, they
have large numbers of parameters and suffer from big computational and memory costs, which make them difficult for real-world
deployment. Therefore, model compression is necessary to reduce
the computation and memory cost of pre-trained models. In this
work, we aim to compress BERT and address the following two challenging practical issues: (1) The compression algorithm should be
able to output multiple compressed models with different sizes and
latencies, in order to support devices with different memory and
latency limitations; (2) The algorithm should be downstream task
agnostic, so that the compressed models are generally applicable for
different downstream tasks. We leverage techniques in neural architecture search (NAS) and propose _NAS-BERT_, an efficient method
for BERT compression. NAS-BERT trains a big supernet on a carefully designed search space containing a variety of architectures
and outputs multiple compressed models with adaptive sizes and
latency. Furthermore, the training of NAS-BERT is conducted on
standard self-supervised pre-training tasks (e.g., masked language
model) and does not depend on specific downstream tasks. Thus,
the compressed models can be used across various downstream
tasks. The technical challenge of NAS-BERT is that training a big
supernet on the pre-training task is extremely costly. We employ
several techniques including block-wise search, search space pruning, and performance approximation to improve search efficiency
and accuracy. Extensive experiments on GLUE and SQuAD benchmark datasets demonstrate that NAS-BERT can find lightweight
models with better accuracy than previous approaches, and can be
directly applied to different downstream tasks with adaptive model
sizes for different requirements of memory or latency.


1Correspondence to: Xu Tan <xuta@microsoft.com> and Jian Li <lijian83@mail.tsinghua.edu.cn>.


Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation

- n the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
_KDD ’21, August 14–18, 2021, Virtual Event, Singapore_
© 2021 Association for Computing Machinery.
ACM ISBN 978-1-4503-8332-5/21/08...$15.00
[https://doi.org/10.1145/3447548.3467262](https://doi.org/10.1145/3447548.3467262)



**CCS CONCEPTS**


- **Computing methodologies** → **Natural language processing** ;
**Neural networks** .


**KEYWORDS**


BERT compression; task-agnostic; adaptive; neural architecture
search; pre-training


**ACM Reference Format:**
Jin Xu [1], Xu Tan [2], Renqian Luo [3], Kaitao Song [4], Jian Li [1], Tao Qin [2], Tie-Yan
Liu [2] . 2021. NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression
with Neural Architecture Search. In _Proceedings of the 27th ACM SIGKDD_
_Conference on Knowledge Discovery and Data Mining (KDD ’21), August_
_14–18, 2021, Virtual Event, Singapore._ ACM, New York, NY, USA, 11 pages.
[https://doi.org/10.1145/3447548.3467262](https://doi.org/10.1145/3447548.3467262)


**1** **INTRODUCTION**


Pre-trained Transformer [45]-based language models like BERT [10],
XLNet [54] and RoBERTa [30] have achieved impressive performance on a variety of downstream natural language processing
tasks. These models are pre-trained on massive language corpus
via self-supervised tasks to learn language representation and finetuned on specific downstream tasks. Despite their effectiveness,
these models are quite expensive in terms of computation and mem
- ry cost, which makes them difficult for the deployment on different
downstream tasks and various resource-restricted scenarios such as

- nline servers, mobile phones, and embedded devices. Therefore, it
is crucial to compress pre-trained models for practical deployment.
Recently, a variety of compression techniques have been adopted
to compress pre-trained models, such as pruning [14, 33], weight
factorization [24], quantization [38, 56], and knowledge distillation [5, 16, 18, 37, 40, 41]. Several existing works [5, 14, 24, 33, 37,
41, 43, 56] compress a large pre-trained model into a small or fast
model with fixed size on the pre-training or fine-tuning stage and
have achieved good compression efficiency and accuracy. However,
from the perspective of practical deployment, a fixed-size model
cannot be deployed in devices with different memory and latency
constraints. For example, smaller models are preferred in embedded devices than in online servers, and faster inference speed is
more critical in online services than in offline services. On the

- ther hand, some previous methods [5, 16] compress the models

- n the fine-tuning stage for each specific downstream task. This
can achieve good accuracy due to the dedicated design in each task.
However, compressing the model for each task can be laborious
and a compressed model for one task may not generalize well on
another downstream task.




--- end of page=0 ---

In this paper, we study the BERT compression in a different
setting: the compressed models need to cover different sizes and
latencies, in order to support devices with different kinds of mem
- ry and latency constraints; the compression is conducted on the
pre-training stage so as to be downstream task-agnostic [1] . To this
end, we propose NAS-BERT, which leverages neural architecture
search (NAS) to automatically compress BERT models. We carefully design a search space that contains multi-head attention [45],
separable convolution [20], feed-forward network and identity operations with different hidden sizes to find efficient models. In order
to search models with adaptive sizes that satisfy diverse requirements of memory and latency constraints in different devices, we
train a big supernet that contains all the candidate operations and
architectures with weight sharing [1–3, 55]. In order to reduce the
laborious compressing on each downstream task, we directly train
the big supernet and get the compressed model on the pre-training
task to make it applicable across different downstream tasks.
However, it is extremely expensive to directly perform architecture search in a big supernet on the heavy pre-training task.
To improve the search efficiency and accuracy, we employ several
techniques including block-wise search, search space pruning and
performance approximation during the search process: (1) We adopt
block-wise search [25] to divide the supernet into blocks so that
the size of the search space can be reduced exponentially. To train
each block, we leverage a pre-trained teacher model, divide the
teacher model into blocks similarly, and use the input and output
hidden states of the corresponding teacher block as paired data for
training. (2) To further reduce the search cost of each block (even
if block-wise search has greatly reduced the search space) due to
the heavy burden of the pre-training task, we propose progressive
shrinking to dynamically prune the search space according to the
validation loss during training. To ensure that architectures with
different sizes and latencies can be retained during the pruning
process, we further divide all the architectures in each block into
several bins according to their model sizes and perform progressive shrinking in each bin. (3) We obtain the compressed models
under specific constraints of memory and latency by assembling
the architectures in every block using performance approximation,
which can reduce the cost in model selection.

We evaluate the models compressed by NAS-BERT on the GLUE [46]
and SQuAD [35, 36] benchmark datasets. The results show that
NAS-BERT can find lightweight models with various sizes from 5M
to 60M with better accuracy than that achieved by previous work.
Our contributions of NAS-BERT can be summarized as follows:


  - We carefully design a search space that contains various
architectures and different sizes, and apply NAS on the
pre-training task to search for efficient lightweight models,
which is able to deliver adaptive model sizes given different
requirements of memory or latency and apply for different
downstream tasks.


1Here “task-agnostic" only refers to that we do not need to compress the model
specifically for each downstream tasks, but we can still use a pre-trained model to help
training the downstream tasks, such as knowledge distillation. Actually, we study the
full “task-agnostic" setting that we do not use the pre-trained model for knowledge
distillation in downstream tasks (see Section 4.4.3). The results still demonstrate the
effectiveness of our proposed method.




  - We further apply block-wise search, progressive shrinking
and performance approximation to reduce the huge search
cost and improve the search accuracy.

  - Experiments on the GLUE and SQuAD benchmark datasets
demonstrate the effectiveness of NAS-BERT for model compression.


**2** **RELATED WORK**

**2.1** **BERT Model Compression**


Recently, compressing pre-trained models such as BERT [10] has
been studied extensively and several techniques have been proposed such as knowledge distillation, pruning, weight factorization,
quantization and so on. Existing works [5, 18, 24, 24, 37, 38, 40–
43, 56, 56] aim to compress the pre-trained model into a fixed size

- f the model and have achieved a trade-off between the small parameter size (usually no more than 66M) and the good performance.
However, these compressed models cannot be deployed in devices
with different memory and latency constraints. Recent works [16]
can deliver adaptive models for each specific downstream task and
demonstrate the effectiveness of the task-oriented compression. For
practical deployment, it can be laborious to compress models from
each task. Other works [13] can produce compressed models on the
pre-training stage that can directly generalize to downstream tasks,
and allow for efficient pruning at inference time. However, they do
not explore the potential of different architectures as in our work.
Different from existing works, NAS-BERT aims for task-agnostic
compression on the pre-training stage which eliminates the labori
- us compression for each specific downstream task, and carefully
designs the search space which is capable to explore the potential

- f different architectures and deliver various models given diverse
memory and latency requirements.


**2.2** **Neural Architecture Search for Efficient**
**Models**


Many works have leveraged NAS to search efficient models [2, 3,
17, 19, 26, 28, 29, 43, 47, 55]. Most of them focus on computer vision
tasks and rely on specific designs on the convolutional layers (e.g.,
inverted bottleneck convolution [17] or elastic kernel size [2, 55]).
Among them, once-for-all [2] and BigNAS [55] train a big supernet
that contains all the candidate architectures and can get a specialized sub-network by selecting from the supernet to support
various requirements (e.g., model size and latency). HAT [47] also
trains a supernet with the adaptive widths and depths for machine
translation tasks. Our proposed NAS-BERT also trains a big supernet. However, different from these methods, we target model
compression for BERT at the pre-training stage, which is a more
challenging task due to the large model size, search space and huge
pre-training cost. In our preliminary experiments, directly applying
the commonly used single path optimization [1, 2, 7, 15] in such a
big supernet cannot even converge. Therefore, we introduce several techniques including block-wise search, progressive shrinking,
and performance approximation to reduce the training cost, search
space and improve search efficiency. Tsai et al. [43] apply one-shot
NAS to search a faster Transformer but they cannot deliver multiple
architectures to meet various constraints for deployment. Different
from Tsai et al. [43], NAS-BERT 1) progressively shrinks the search




--- end of page=1 ---

space to allocate more resources to promising architectures and
thus can deliver various architectures without adding much computation; 2) designs bins in the shrinking algorithm to guarantee that
we can search architectures to meet diverse memory and latency
constraints. 3) explores novel architectures with convolution layer,
multi-head attention, and feed-forward layer, and achieves better
performance than previous works for BERT compression.


**3** **METHOD**


In this section, we describe NAS-BERT, which conducts neural architecture search to find small, novel and accurate BERT models. To
meet the requirements of deployment for different memory and latency constraints and across different downstream tasks, we 1) train
a supernet with a novel search space that contains different sizes

- f models for various resource-restricted devices, and 2) directly
search the models on the pre-training task to make them generalizable on different downstream tasks. The method can be divided into
three steps: 1) search space design (Section 3.1); 2) supernet training
(Section 3.2); 3) model selection (Section 3.3). Due to the huge cost
to train the big supernet on the heavy pre-training task and select
compressed models under specific constraints, we introduce several
techniques including block-wise search, search space pruning and
performance approximation in Section 3.2 and 3.3 to reduce the
search space and improve the search efficiency.


**3.1** **Search Space Design**

A novel search space allows the potential of combinations of different operations, instead of simply stacking basic Transformer block
(multi-head attention and feed-forward network) as in the original
BERT model. We adopt the chain-structured search space [12], and
construct an over-parameterized supernet A with _𝐿_ layers and each
layer contains all candidate operations in O = { _𝑜_ 1 _,_ - · · _,𝑜𝐶_ }, where
_𝐶_ is the number of predefined candidate operations. Residual connection is applied to each layer by adding the input to the output.
There are _𝐶_ _[𝐿]_ possible paths (architectures) in the supernet, and
a specific architecture _𝑎_ = ( _𝑎_ [1] _,_ - · · _,𝑎_ _[𝐿]_ ) is a sub-net (path) in the
supernet, where _𝑎_ _[𝑙]_ ∈O is the operation in the _𝑙_ - th layer, as shown
in Fig. 2 (a). We adopt weight sharing mechanism that is widely
used in NAS [1, 2] for efficient training, where each architecture
(path) shares the same set of operations in each layer.
We further describe each operation in O as follows: 1) Multihead attention (MHA) and feed-forward network (FFN), which
are the two basic operations in Transformer and are popular in
pre-training models (in this way we can cover BERT model as a
subnet in our supernet). 2) Separable convolution (SepConv), whose
effectiveness and efficiency in natural language processing tasks
have been demonstrated by previous work [20, 21]. 3) Identity

- peration, which can support architectures with the number of
layers less than _𝐿_ . Identity operation is regarded as a placeholder in
a candidate architecture and can be removed to obtain a shallower
network. Apart from different types of operations, to allow adaptive
model sizes, each operation can have different hidden sizes: {128,
192, 256, 384, 512}. In this way, architectures in the search space can
be of different depths and widths. The complete candidate operation
set O contains (1 + 1 + 3) ∗ 5 + 1 = 26 operations, where the first
product term represents the number of types of operations and



**Table 1: Candidate operation set. For each type of opera-**
**tion including multi-head attention (MHA), feed-forward**
**network (FFN) and separable convolution (SepConv) in each**
**row, we list the number of heads in MHA and the size of the**
**intermediate layer in FFN, and kernel size in SepConv under**
**different hidden sizes.**


|HiddenSize|128|192|256|384|512|
|---|---|---|---|---|---|
|MHA|2 Heads|3 Heads|4 Heads|6 Heads|8 Heads|
|FFN|512|768|1024|1536|2048|
|SepConv|Kernel {3, 5, 7}|Kernel {3, 5, 7}|Kernel {3, 5, 7}|Kernel {3, 5, 7}|Kernel {3, 5, 7}|
|Identity|Identity|Identity|Identity|Identity|Identity|



### Figure 1

Caption: Structure of separable convolution.**

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


3 represents the SepConv with different kernel size {3, 5, 7}, the
second product term represents that there are 5 different hidden
sizes. We present 26 operations in Table 1 and structure of separable
convolution in Fig. 1.


**3.2** **Supernet Training**


_3.2.1_ _Block-Wise Training with Knowledge Distillation._ Directly
training the whole supernet causes huge cost due to its large model
size and huge search space. With limited computational resources
(total training time, steps, etc.), the amortized training time of each
architecture from the huge search space is insufficient for accurate
evaluation [7, 27, 31]. Inspired by Li et al. [25], we adopt blockwise search to uniformly divide the supernet A into _𝑁_ blocks
(A1 _,_ A2 _,_ - · · _,_ A _𝑁_ ) to reduce the search space and improve the efficiency. To train each block independently and effectively, knowledge distillation is applied with a pre-trained BERT model. The
pre-trained BERT model (teacher) is divided into corresponding
_𝑁_ blocks as in Fig. 2. The input and output hidden states of the
corresponding block in the teacher model are used as the paired
data to train the block in the supernet (student). Specifically, the
_𝑛_ - th student block receives the output of the ( _𝑛_ - 1)-th teacher
block as the input and is optimized to predict the output of the _𝑛_ - th
teacher block with mean square loss


L _𝑛_ = || _𝑓_ (Y _𝑛_      - 1; A _𝑛_ ) − Y _𝑛_ || [2] 2 _[,]_ (1)


where _𝑓_ (·; A _𝑛_ ) is the mapping function of _𝑛_ - th block A _𝑛_, Y _𝑛_ is
the output of the _𝑛_ - th block of the teacher model (Y0 is the output

- f the embedding layer of the teacher model). At each training
step, we randomly sample an architecture from the search space
following Bender et al. [1], Cai et al. [2], Guo et al. [15], which
is memory-efficient due to the single path optimization. Different
from Li et al. [25], we allow different hidden sizes and incorporate
identity layer in each block to support elastic width and depth to
derive models that meet various requirements. Besides, the search
space within each block in our work is larger compared to Li et al.






--- end of page=2 ---

Teacher

(BERTbase)


Student



Block 1 Block 2 Block 3



Layer 6


Layer 5


Layer 4


Layer 3


Layer 2


Layer 1









Block 4





Operation


(a)



Transformer Layer Hidden Size Transformation


(b)



### Figure 2

Caption: (a) an architecture (path) in the supernet. (b) an illustration of block-wise distillation (** _𝑁_ = 4 **blocks). The supernet**

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
**(student) and the pre-trained teacher model are divided into blocks respectively and each student block is trained to mimic**
**the corresponding teacher block.**




[25] (100x larger) which is much more sample in-efficient and requires more techniques (described in Section 3.2.2) to improve the
training efficiency. Since the hidden sizes of the student block may
be different from that in the teacher block, we cannot directly leverage the input and output hidden of the teacher block as the training
data of the student block. To solve this problem, we use a learnable
linear transformation layer at the input and output of the student
block to transform each hidden size to match that of the teacher

block, as shown in Fig. 2.



_3.2.2_ _Progressive Shrinking._ Although block-wise training can largely
reduce the search space, the supernet still requires huge time for
convergence due to the heavy pre-training task. To further improve
the training effectiveness and efficiency, we propose to progressively shrink the search space in each block during training to
allocate more training resources to more promising candidates [27,
32, 48]. However, simply pruning the architectures cannot ensure
to obtain different sizes of models, since larger models in A _𝑛_ are
likely to be pruned on the early stage of training due to its difficulty of optimization [7, 27, 31] and smaller models are likely to
be pruned at the late stage due to limited capacity. Therefore, we
assign the architectures in A _𝑛_ into different _bins_ where each bin
represents a short range of model sizes. Besides, we also apply latency constraints in each bin to avoid models accepted parameter
size but large latency. Denote _𝑝𝑏_ = _[𝑏]_ [·] _[ 𝑝]_ [(] _[𝑎][𝑡]_ [)][ and] _[ 𝑙][𝑏]_ [=] _[𝑏]_ [·] _[ 𝑙]_ [(] _[𝑎][𝑡]_ [)][ as]



being deleted. The design of bins ensures the diversity of models
when shrinking the search space, which makes it possible to select
a model given diverse constraints at the model selection stage.


**3.3** **Model Selection**


After the supernet training with progressive shrinking, each block
contains _𝑚_ ∗ _𝐵_ possible architectures and the whole supernet ( _𝑁_
blocks) contains ( _𝑚_ ∗ _𝐵_ ) _[𝑁]_ complete architectures. The model selection procedure is as follows: 1) We build a large lookup table
_𝐿𝑇𝑎𝑟𝑐ℎ_ with ( _𝑚_ ∗ _𝐵_ ) _[𝑁]_ items, where each item contains the metainformation of a complete architecture: _(architecture, parameter,_
_latency, loss)_ . Since it is extremely time-consuming to measure the
exact latency and loss for ( _𝑚_ ∗ _𝐵_ ) _[𝑁]_ (e.g., 10 [8] in our experiments)
architectures, we use performance approximation to obtain the
two values as described in the next paragraph. 2) For a given constraint of model size and inference latency, we select the top _𝑇_
architectures with low loss from _𝐿𝑇𝑎𝑟𝑐ℎ_ that meet the parameter
and latency constraint. 3) We evaluate the validation loss of the top
_𝑇_ complete architectures on the dev set and select the best one as
the final compressed model.
Next we introduce the performance approximation of the latency
and loss when building the lookup table _𝐿𝑇𝑎𝑟𝑐ℎ_ . We measure the
latency of each candidate operation (just 26 in our design) on the
target device and store in a lookup table _𝐿𝑇𝑙𝑎𝑡_ in advance, and then
approximate the latency of an architecture _𝑙_ ( _𝑎_ ) by _𝑙_ ( _𝑎_ ) = [�] _𝑙_ _[𝐿]_ =1 _[𝑙]_ [(] _[𝑎][𝑙]_ [)]
following Cai et al. [3], where _𝑙_ ( _𝑎_ _[𝑙]_ ) is from _𝐿𝑇𝑙𝑎𝑡_ . To approximate
the loss of an architectures in _𝐿𝑇𝑎𝑟𝑐ℎ_, we add up the block-wise
distillation loss of the sub-architecture in each block on the dev set.

Obtaining the dev loss of all sub-architectures in all blocks only
involves _𝑚_ ∗ _𝐵_ ∗ _𝑁_ evaluations.


**4** **EXPERIMENT**

**4.1** **Experimental Setup**


_4.1.1_ _Supernet Training Setup._ The supernet consists of _𝐿_ = 24 layers, which is consistent with BERTbase [10] (BERTbase has 12 Transformer layers with 24 sub-layers in total, since each Transformer




_[𝑏]_ _[𝑏]_

_𝐵_ [·] _[ 𝑝]_ [(] _[𝑎][𝑡]_ [)][ and] _[ 𝑙][𝑏]_ [=] _𝐵_



size but large latency. Denote _𝑝𝑏_ = _𝐵_ [·] _[ 𝑝]_ [(] _[𝑎][𝑡]_ [)][ and] _[ 𝑙][𝑏]_ [=] _𝐵_ [·] _[ 𝑙]_ [(] _[𝑎][𝑡]_ [)][ as]

the maximum parameter size and latency for the _𝑏_ - th bin, where
_𝑝_ (·) and _𝑙_ (·) calculate the parameter size and latency, _𝑎_ _[𝑡]_ is the
largest model in the search space and _𝐵_ is the number of bins. The
architecture _𝑎_ in _𝑏_ - th bin should meet (1) _𝑝𝑏_ _> 𝑝_ ( _𝑎_ ) _> 𝑝𝑏_ - 1 and (2)
_𝑙𝑏_ _> 𝑙_ ( _𝑎_ ) _> 𝑙𝑏_ - 1. Architectures that cannot satisfy the constraint

- f latency are removed.
Then we conduct the progressive shrinking algorithm in each
bin at the end of each training epoch as follows: 1) **Sample** _𝐸_
architectures in each bin and get the validation losses on the dev
set; 2) **Rank** the _𝐸_ architectures according to their validation losses
in descending order; 3) **Remove** _𝑅_ architectures with the largest
losses. The shrinking algorithm terminates when there are only _𝑚_
architectures left in the search space to avoid all the architectures




--- end of page=3 ---

**Table 2: Comparison of NAS-BERT models and hand-designed BERT models under different sizes (60M, 30M, 10M, 5M) on**
**GLUE dev set. “PF” means pre-training and fine-tuning. “KD” means two-stage knowledge distillation. MNLI is reported in**
**the matched set. Spearman correlation is reported for STS-B. Matthews correlation is reported for CoLA. Accuracy is reported**
**for other tasks.**


|Setting|FLOPs|Speedup|MNLI QQP QNLI CoLA SST-2 STS -B RTE MRPC|AVG|
|---|---|---|---|---|
|BERT60 + PF<br>NAS-BERT60 + PF|1.3e10<br>1.3e10|2.2×<br>2.2×|82.6<br>90.3<br>89.4<br>52.6<br>92.1<br>88.3<br>75.6<br>89.2<br>83.0<br>90.9<br>90.8<br>53.8<br>92.3<br>88.7<br>76.7<br>88.9|82.5<br>**83.2**|
|BERT60 + KD<br>NAS-BERT60 + KD|1.3e10<br>1.3e10|2.2×<br>2.2×|83.2<br>90.5<br>90.2<br>56.3<br>91.8<br>88.8<br>78.5<br>88.5<br>84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5|83.5<br>**84.2**|
|BERT30 + PF<br>NAS-BERT30 + PF|7.1e9<br>7.0e9|3.6 ×<br>3.6 ×|80.0<br>89.6<br>87.6<br>40.6<br>90.8<br>87.7<br>73.2<br>84.1<br>80.4<br>90.0<br>87.8<br>48.1<br>90.3<br>87.3<br>71.4<br>84.9|79.2<br>**80.0**|
|BERT30 + KD<br>NAS-BERT30 + KD|7.1e9<br>7.0e9|3.6 ×<br>3.6 ×|80.8<br>89.8<br>88.7<br>44.7<br>90.5<br>87.6<br>70.3<br>85.2<br>81.0<br>90.2<br>88.4<br>48.7<br>90.5<br>87.6<br>71.8<br>84.6|79.7<br>**80.3**|
|BERT10 + PF<br>NAS-BERT10 + PF|2.3e9<br>2.3e9|9.1 ×<br>8.7 ×|74.0<br>87.6<br>84.9<br>25.7<br>88.3<br>85.3<br>64.1<br>81.9<br>76.0<br>88.4<br>86.3<br>27.8<br>88.6<br>84.3<br>68.7<br>81.5|74.0<br>**75.2**|
|BERT10 + KD<br>NAS-BERT10 + KD|2.3e9<br>2.3e9|9.1 ×<br>8.7 ×|74.4<br>87.8<br>85.7<br>32.5<br>86.6<br>85.2<br>66.9<br>77.9<br>76.4<br>88.5<br>86.3<br>34.0<br>88.6<br>84.8<br>66.6<br>79.1|74.6<br>**75.5**|
|BERT5 + PF<br>NAS-BERT5 + PF|8.6e8<br>8.7e8|20.7 ×<br>23.6 ×|67.7<br>84.1<br>80.9<br>10.4<br>81.6<br>81.1<br>62.8<br>78.6<br>74.2<br>85.7<br>83.9<br>19.6<br>84.9<br>82.8<br>67.0<br>80.0|68.4<br>**72.3**|
|BERT5 + KD<br>NAS-BERT5 + KD|8.6e8<br>8.7e8|20.7 ×<br>23.6 ×|67.9<br>83.2<br>80.6<br>12.6<br>82.8<br>81.0<br>61.9<br>78.1<br>74.4<br>85.8<br>84.9<br>19.8<br>87.3<br>83.0<br>66.6<br>79.6|68.5<br>**72.7**|



layer has a MHA and FFN). We use a pre-trained BERTbase [10] as
the teacher model. The detailed configurations of the search space,
teacher model training can be found in Appendix A.1 and A.2. The
supernet is divided into _𝑁_ = 4 blocks and the search space in each
block is divided into _𝐵_ = 10 bins. We first train the supernet for 3
epochs without progressive shrinking as a warm start, and then
begin to shrink at the end of each later epoch. We randomly sample
_𝐸_ = 2000 architectures for validation (evaluate all architectures
when the number of architectures in search space is less than 2000)
and perform the progressive shrinking to remove _𝑅_ = _𝐸_ /2 architectures for each bin as in Section 3.2.2. The shrinking process
terminates when only _𝑚_ = 10 architectures are left in each bin
in each block, and the training also ends. The supernet is trained

- n English Wikipedia plus BookCorpus (16GB size) using masked
language modeling (MLM) following BERT [2] [10], with a batch size

- f 1024 sentences and each sentence consists of 512 tokens. The

training costs 3 days on 32 NVIDIA P40 GPUs while training the
BERTbase teacher model costs 5 days. Other training configurations
remain the same as the teacher model. We select _𝑇_ = 100 models

from the table _𝐿𝑇𝑎𝑟𝑐ℎ_ - n the model selection stage. In progressive
shrinking, to reduce the time of evaluating all the candidates, we

- nly evaluate on 5 batches rather than the whole dev set, which
is accurate enough for the pruning according to our preliminary
study.


_4.1.2_ _Evaluation Setup on Downstream Tasks._ We evaluate the effectiveness of NAS-BERT by pre-training the compressed models _from_
_scratch_ - n the pre-training task and fine-tuning on the GLUE [46]
and SQuAD benchmarks [35, 36]. The GLUE benchmark includes
CoLA [50], SST-2 [39], MRPC [11], STS-B [4], QQP [6], MNLI [51],
QNLI [36] and RTE [9]. Similar to previous methods [16, 37, 44, 49],


2The next sentence tasks are not considered since previous work [30] have achieved
good results without it.



we also apply knowledge distillation and conduct it on two stages
(i.e., pre-training and fine-tuning) as the default setting for evaluation. Considering the focus of our work is to compress pre-trained
models with novel architectures instead of knowledge distillation,
we only use prediction layer distillation and leave the various distillation techniques like layer-by-layer distillation, embedding layer
distillation and attention matrix distillation [16, 18, 37, 41, 42, 49]
that can further improve the performance as to future work. During
fine-tuning on the GLUE benchmark, RTE, MRPC and STS-B are
started from the model fine-tuned on MNLI following Liu et al. [30].


_4.1.3_ _FLOPs and Latency._ Following Clark et al. [8], we infer FLOPs
with batch size of 1 and the sequence length of 128. The latency
used in progressive shrinking and model selection is measured

- n Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60 GHz with 12 cores,
but our method can be easily applied to other devices (e.g., mobile
platforms, embedded devices) by using the lookup table _𝐿𝑇𝑙𝑎𝑡_ (described in Section 3.3) measured and built for the corresponding
device. Following Sun et al. [41], Wang et al. [49], the inference
time is evaluated on QNLI training set with batch size of 128 and
the maximum sequence length of 128. The numbers reported are
the average of 100 batches.


**4.2** **Results on the GLUE Datasets**


While our NAS-BERT can compress models with adaptive sizes, we

- nly show the results of compressed models with 60M, 30M, 10M
and 5M parameter sizes (denoted as NAS-BERT60, NAS-BERT30,
NAS-BERT10 and NAS-BERT5 respectively) on the GLUE benchmark due to the large evaluation cost, and list the model structures
with different sizes in Table 10. We compare our NAS-BERT models with hand-designed BERT models under the same parameter
size and latency (denoted as BERT60, BERT30, BERT10 and BERT5




--- end of page=4 ---

**Table 3: Results on the dev and test set of the GLUE benchmark. “*” means using data augmentation. The test set results are**

**obtained from the official GLUE leaderboard.**


Model Params MNLI QQP QNLI CoLA SST-2 STS-B RTE MRPC AVG


|dev set|Col2|Col3|Col4|
|---|---|---|---|
|Teacher|110M|85.2<br>91.0<br>91.3<br>61.0<br>92.9<br>90.3<br>76.0<br>87.7|84.4|
|DistilBERT<br>MiniLM<br>BERT-of-Theseus<br>PD-BERT<br>DynaBERT*|66M<br>66M<br>66M<br>66M<br>60M|82.2<br>88.5<br>89.2<br>51.3<br>91.3<br>86.9<br>59.9<br>87.5<br>84.0<br>91.0<br>91.0<br>49.2<br>92.0<br>-<br>71.5<br>88.4<br>82.3<br>89.6<br>89.5<br>51.1<br>91.5<br>88.7<br>68.2<br>-<br>82.5<br>90.7<br>89.4<br>-<br>91.1<br>-<br>66.7<br>84.9<br>84.2<br>**91.2**<br>91.5<br>56.8<br>92.7<br>89.2<br>72.2<br>84.1|79.6<br>-<br>-<br>-<br>82.7|
|NAS-BERT<br>60M<br>84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5<br>84.2<br>NAS-BERT*<br>60M<br>**84.8**<br>**91.2**<br>**91.9**<br>**58.7**<br>**93.1**<br>**89.9**<br>**79.8**<br>**88.9**<br>**84.8**|NAS-BERT<br>60M<br>84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5<br>84.2<br>NAS-BERT*<br>60M<br>**84.8**<br>**91.2**<br>**91.9**<br>**58.7**<br>**93.1**<br>**89.9**<br>**79.8**<br>**88.9**<br>**84.8**|NAS-BERT<br>60M<br>84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5<br>84.2<br>NAS-BERT*<br>60M<br>**84.8**<br>**91.2**<br>**91.9**<br>**58.7**<br>**93.1**<br>**89.9**<br>**79.8**<br>**88.9**<br>**84.8**|NAS-BERT<br>60M<br>84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5<br>84.2<br>NAS-BERT*<br>60M<br>**84.8**<br>**91.2**<br>**91.9**<br>**58.7**<br>**93.1**<br>**89.9**<br>**79.8**<br>**88.9**<br>**84.8**|


|test set|Col2|Col3|Col4|
|---|---|---|---|
|Teacher|110M|84.8<br>89.0<br>91.7<br>57.1<br>94.1<br>88.0<br>71.0<br>84.1|82.5|
|BERT-of-Theseus<br>PD-BERT<br>BERT-PKD<br>TinyBERT*|66M<br>66M<br>66M<br>66M|82.4<br>**89.3**<br>89.6<br>47.8<br>92.2<br>84.1<br>66.2<br>83.2<br>82.8<br>88.5<br>88.9<br>-<br>91.8<br>-<br>65.3<br>81.7<br>81.5<br>88.9<br>89.0<br>-<br>92.0<br>-<br>65.5<br>79.9<br>**84.6**<br>89.1<br>90.4<br>**51.1**<br>**93.1**<br>83.7<br>70.0<br>82.6|79.4<br>-<br>-<br>80.6|
|NAS-BERT<br>NAS-BERT*|60M<br>60M|83.5<br>88.9<br>90.9<br>48.4<br>92.9<br>86.1<br>**73.7**<br>84.5<br>84.1<br>88.8<br>**91.2**<br>50.5<br>92.6<br>**86.9**<br>72.7<br>**86.4**|81.1<br>**81.7**|



respectively). We follow several principles when manually designing the BERT models: 1) The size of the embedding layer is the
same as that of the corresponding NAS-BERT model; 2) We use the

- riginal BERT structure (MHA plus FFN) and keep the parameter,
latency, depth and width as close as possible to the corresponding NAS-BERT model. The baseline BERT models in Table 2 are:
BERT60 (L=10, H=512, A=8), BERT30 (L=6, H=512, A=8), BERT10
(L=6, H=256, A=4) and BERT5 (L=6, H=128, A=2) where L is the number of layers, H is the hidden size, and A is the number of attention
heads. To demonstrate the advantages of architectures searched
by NAS-BERT, the comparisons are evaluated in two settings: 1)

- nly with pre-training and fine-tuning (PF), and 2) pre-training and
fine-tuning with two-stage knowledge distillation (KD). The results
are shown in Table 2, from which we can see that NAS-BERT outperforms hand-designed BERT baseline across almost all the tasks
under various model sizes. Especially, the smaller the model size is,
the larger gap can be observed (e.g., NAS-BERT5 vs. BERT5). The
results show that NAS-BERT can search for efficient lightweight
models that are better than Transformer based models.


**4.3** **Comparison with Previous Work**


_4.3.1_ _Comparison in commonly used model size ._ Next, we compare
the effectiveness of our NAS-BERT to previous methods on BERT
compression. Since they usually compress BERT into a model size

- f about 66M or 60M, we use our NAS-BERT60 for comparison. We
mainly compare our NAS-BERT with methods whose teacher model
is BERTbase for a fair comparison including 1) DistilBERT [37],
which uses knowledge distillation on the pre-training stage; 2)
BERT-PKD [41], which distills the knowledge from the intermediate layers and the final output logits on the pre-training stage; 3)
BERT-of-Theseus [53], which uses module replacement for compression; 4) MiniLM [49], which transfers the knowledge from the



self-attention module; 5) PD-BERT [44], which distills the knowledge from the target domain in BERT training; 6) DynaBERT [16],
which uses network rewiring to adjust width and depth of BERT for
each downstream task. 7) TinyBERT [18], which leverage embedding layer, hidden layer, and attention matrix distillation to mimic
the teacher model at both the pre-training and fine-tuning stages.
The comparison of their teacher models is presented in Table 9. We
do not compare with MobileBERT [42] since it uses a teacher model
with much higher accuracy than the commonly used BERTbase
(nearly matches to the accuracy of BERTlarge), and the pre-training
computations of its student model (4096 batch size * 740,000 steps)
is significantly larger than that of other method [10, 37, 41] and
NAS-BERT (2048 batch size * 125,000 steps). Nevertheless, we will
compare with MobileBERT in future work. To make comparison
with DynaBERT and TinyBERT, we also use their data augmentation [16, 18] on downstream tasks. Table 3 reports the results

- n the dev and test set of the GLUE benchmark. Without data

augmentation, NAS-BERT achieves better results on nearly all the
tasks compared to previous work. Further, with data augmentation,
NAS-BERT outperforms DynaBERT and TinyBERT. Different from
these methods that leverage advanced knowledge distillation techniques in pre-training and/or fine-tuning, NAS-BERT mainly takes
advantage of architectures and achieves better accuracy, which
demonstrates the advantages of NAS-BERT in model compression.


_4.3.2_ _Comparison in extremely small model size._ Compressing the
pre-training model into an extremely small model with good accuracy is challenging. Previous works usually compress the model
at the fine-tuning stage for each specific task with sophisticated
techniques. For example, AdaBERT [5] searches a task-specific architecture (6M - 10M) for each task, and introduces the special
distillation techniques (e.g., using probe classifiers to hierarchically
decompose the task-useful knowledge from the teacher model) and




--- end of page=5 ---

data augmentations to boost the performance. To show the effectiveness of our searched architecture in the extremely small size
itself, we compare NAS-BERT with AdaBERT without sophisticated
techniques. Specifically, we do not use two-stage distillation and
data augmentation for NAS-BERT5, and do not use data augmentation and probe classifiers for AdaBERT. As shown in Table 4, we
can find that NAS-BERT5 (with an only 5M model size) can greatly
improve upon AdaBERT (with slightly larger model size) by 1.9%,
2.8% and 10.3% (absolute differences) on QNLI, MRPC, and RTE.


**Table 4: Comparison in extremely small model size.**

|Setting|QNLI/Params|MRPC/Params|RTE/Params|
|---|---|---|---|
|AdaBERT<br>NAS-BERT|82.0/7.9M<br>83.9/5.0M|77.2/7.5M<br>80.0/5.0M|56.7/8.6M<br>67.0/5.0M|



**4.4** **Ablation Study**


_4.4.1_ _Ablation Study on Progressive Shrinking._ To verify the effectiveness of progressive shrinking (PS), we train the supernet
with the same number of training epochs but without progressive
shrinking. We follow the same procedure in NAS-BERT for model
selection and final evaluation on downstream tasks. Due to the huge
evaluation cost during the model selection on the whole search
space without progressive shrinking, it costs 50 hours (evaluation
cost on the shrinked search space only takes 5 minutes since there
are only 10 architectures remaining in each bin in each block). The
results are shown in Table 5. It can be seen that NAS-BERT with

progressive shrinking searches better architectures, with less total
search time.


### Figure 3

Caption: Loss curve of supernet training. Progressive shrink-**

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
**ing starts at the epoch 4.**


We further show the training loss curve in Fig. 3. It can be seen
that the superset without progressive shrinking suffers from slow
convergence. The huge number of architectures in the supernet
need long time for sufficient training. Given a fixed budget of training time, progressive shrinking can ensure the promising architectures to be trained with more resources and result in more accurate

evaluation, and thus better architectures can be selected. On the
contrary, without progressive shrinking, the amortized training
time of each architecture is insufficient, resulting in inaccurate
evaluation and model selection.


_4.4.2_ _Different Progressive Shrinking Approaches._ Instead of pruning architectures (paths) from the search space, we can also prune

- perations (nodes) from the supernet [32, 34] in progressive shrinking. From the perspective of supernet, the former is to remove paths



and the latter is to remove nodes from the supernet. To evaluate the
performance of operations (nodes) in supernet, at the end of each
training epoch, we evaluate the architectures on the dev set and
prune the search space according to the performance (validation
loss) of operations. The validation loss of the operation _𝑜𝑖_ in _𝑙_ - layer
is estimated by the mean validation losses of all the architectures
whose operation in the _𝑙_ - th layer _𝑎_ _[𝑙]_ = _𝑜𝑖_ . The shrinking algorithm
proceeds as follows:


- Sample _𝐸_ architectures and get the validation losses on the dev

set.

- Rank operations according to their mean validation losses in
descending order.

- Prune operations with the largest losses from the supernet repeatedly until removing _𝑅_ ( _𝑅_ is a hyper-parameter to control the
speed of pruning) of the architectures in the search space.


The shrinking algorithm performs at the end of each training
epoch, and terminates when only _𝑚_ = 10 architectures are left in
each bin in each block, and the training also ends. For the fair comparison, we set _𝑚_ = 10 and _𝐸_ = 1000, which are same as settings in
Section 4.1. At the end of each epoch, we perform this algorithm
to remove _𝑅_ = 30% architectures for each bin. In this way, the
algorithm terminates at the same epoch as that in Section 3.2.2.
At the end of each training epoch, we evaluate the architectures

- n the dev set and prune the search space according to the performance of _operations_ . As shown in Table 6, pruning architectures in
progressive shrinking achieves better results.


_4.4.3_ _Ablation study of two-stage distillation._ We further study the
effectiveness of NAS-BERT by removing the distillation on pretraining or fine-tuning stage: 1) only using the distillation on the
pre-training stage; 2) only using the distillation on the fine-tuning
stage. The results are presented in Table 7. We can find that 1)
NAS-BERT consistently outperforms the BERT baseline in different
settings; 2) distillation on either pre-training or fine-tuning stage
can improve the accuracy, and two-stage distillation can further get
better accuracy; 3) when removing the distillation at the fine-tuning
stage, this setting becomes fully task-agnostic (the model is compressed in pre-training stage, and only needs fine-tuning on each
downstream task). We can find that NAS-BERT can still achieve a
score of 83.5 on average, which outperforms the previous works in
Table 3. These results demonstrate that our searched architectures
are effective regardless of the specific distillation settings. NASBERT can still achieve good results when removing distillation on
the fine-tuning stage.


**4.5** **Results on the SQuAD Datasets**


We further evaluate our method on the SQuAD v1.1 [36] and SQuAD
v2.0 [35], which require to predict the answer span based on the
given question and passage. For SQuAD v1.1, each question has
the corresponding answer based on the given passage. And for
SQuAD v2.0, some questions do not have the corresponding answer.
Following previous practices [30], we add an additional binary
classification layer to predict whether the answer exists for SQuAD
v2.0. The results are shown in Table 8. We can find that our NASBERT outperforms previous works on both SQuAD v1.1 and v2.0.




--- end of page=6 ---

**Table 5: The results of NAS-BERT with and without progressive shrinking (PS) on NAS-BERT** 60 **.**

|Setting|MNLI QQP QNLI CoLA SST -2 STS -B RTE MRPC|AVG|
|---|---|---|
|w/o PS<br>w PS|83.5<br>90.6<br>90.8<br>55.5<br>92.0<br>88.1<br>77.4<br>86.5<br>84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5|83.1<br>**84.2**|



**Table 6: The results of different progressive shrinking approaches on NAS-BERT** 60 **. PS-arch and PS-op denote pruning archi-**
**tectures and operations in progressive shrinking.**

|Setting|MNLI QQP QNLI CoLA SST -2 STS -B RTE MRPC|AVG|
|---|---|---|
|PS-arch<br>PS-op|84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5<br>83.8<br>91.0<br>90.2<br>54.5<br>92.0<br>88.8<br>75.6<br>86.1|**84.2**<br>82.8|



**Table 7: Ablation study of two-stage knowledge distillation. “PD” or “FD” means to add knowledge distillation on the pre-**
**training or the fine-tuning stage.**


|Setting|PD FD|MNLI QQP QNLI CoLA SST -2 STS -B RTE MRPC|AVG|
|---|---|---|---|
|BERT60<br>NAS-BERT60|✓<br>✓<br>✓<br>✓|83.2<br>90.5<br>90.2<br>56.3<br>91.8<br>88.8<br>78.5<br>88.5<br>84.1<br>91.0<br>91.3<br>58.1<br>92.1<br>89.4<br>79.2<br>88.5|83.5<br>**84.2**|
|BERT60<br>NAS-BERT60|✓<br>✓|83.2<br>90.3<br>89.5<br>55.0<br>91.6<br>88.6<br>77.8<br>87.3<br>83.3<br>90.9<br>91.3<br>55.6<br>92.0<br>88.6<br>78.5<br>87.5|82.9<br>**83.5**|
|BERT60<br>NAS-BERT60|✓<br>✓|83.1<br>90.4<br>90.4<br>54.3<br>91.2<br>88.7<br>75.6<br>87.0<br>83.7<br>90.8<br>91.0<br>54.2<br>92.1<br>89.4<br>76.0<br>87.5|82.6<br>**83.1**|



**Table 8: Results on the dev set of the SQuAD v1.1 and SQuAD**
**v2.0 dataset. MiniLM** - **means that MiniLM is trained with**
**batch size 1024 and 400,000 steps. Thus we train the NAS-**
**BERT** - **with batch size 2048 and 200,000 steps for a fair com-**
**parison.**

|son.|Col2|Col3|Col4|
|---|---|---|---|
|Model|Params|SQuAD v1.1<br>EM<br>F1|SQuAD v2.0<br>EM<br>F1|
|Teacher|110M|81.8<br>88.9<br>74.5<br>77.9|81.8<br>88.9<br>74.5<br>77.9|
|DistilBERT<br>BERT-PKD<br>MiniLM†<br>TinyBERT|66M<br>66M<br>66M<br>66M|79.1<br>86.9<br>-<br>-<br>77.1<br>85.3<br>66.3<br>69.8<br>-<br>-<br>-<br>76.4<br>79.7<br>87.5<br>69.9<br>73.4|79.1<br>86.9<br>-<br>-<br>77.1<br>85.3<br>66.3<br>69.8<br>-<br>-<br>-<br>76.4<br>79.7<br>87.5<br>69.9<br>73.4|
|NAS-BERT<br>NAS-BERT†|60M<br>60M|80.5<br>88.0<br>73.2<br>76.3<br>**81.2**<br>**88.4**<br>**73.9**<br>**77.1**|80.5<br>88.0<br>73.2<br>76.3<br>**81.2**<br>**88.4**<br>**73.9**<br>**77.1**|



**5** **CONCLUSION**


In this paper, we propose NAS-BERT, which leverages neural architecture search (NAS) to compress BERT models. We carefully
design a search space with different operations associated with
different hidden sizes, to explore the potential of diverse architectures and derive models with adaptive sizes according to the
memory and latency requirements of different devices. The compression is conducted on the pre-training stage and is downstream
task agnostic, where the compressed models can be applicable for
different downstream tasks. Experiments on the GLUE and SQuAD
benchmark datasets demonstrate the effectiveness of our proposed
NAS-BERT compared with both hand-designed BERT baselines and
previous works on BERT compression. For future work, we will
explore more advanced search space and NAS methods to achieve
better performance.



**6** **ACKNOWLEDGEMENTS**


Jin Xu and Jian Li are supported in part by the National Natural
Science Foundation of China Grant 61822203, 61772297, 61632016
and the Zhongguancun Haihua Institute for Frontier Information
Technology, Turing AI Institute of Nanjing and Xi’an Institute for
Interdisciplinary Information Core Technology.


## REFERENCES


[1] Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc
Le. 2018. Understanding and simplifying one-shot architecture search. In _Inter-_
_national Conference on Machine Learning_ . 550–559.

[2] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. 2019. Oncefor-All: Train One Network and Specialize it for Efficient Deployment. In _Inter-_
_national Conference on Learning Representations_ .

[3] Han Cai, Ligeng Zhu, and Song Han. 2018. ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. In _International Conference on_
_Learning Representations_ .

[4] Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. 2017.
SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual
Focused Evaluation. In _Proceedings of the 11th International Workshop on Semantic_
_Evaluation (SemEval-2017)_ . 1–14.

[5] Daoyuan Chen, Yaliang Li, Minghui Qiu, Zhen Wang, Bofang Li, Bolin Ding,
Hongbo Deng, Jun Huang, Wei Lin, and Jingren Zhou. 2020. AdaBERT: TaskAdaptive BERT Compression with Differentiable Neural Architecture Search.
In _Proceedings of the Twenty-Ninth International Joint Conference on Artificial_
_Intelligence, IJCAI-20_, Christian Bessiere (Ed.). International Joint Conferences

[on Artificial Intelligence Organization, 2463–2469. https://doi.org/10.24963/ijcai.](https://doi.org/10.24963/ijcai.2020/341)
[2020/341 Main track.](https://doi.org/10.24963/ijcai.2020/341)

[6] Zihan Chen, Hongbo Zhang, Xiaoji Zhang, and Leqi Zhao. 2018. Quora question
pairs.

[7] Xiangxiang Chu, Bo Zhang, Ruijun Xu, and Jixiang Li. 2019. Fairnas: Rethinking
evaluation fairness of weight sharing neural architecture search. _arXiv preprint_
_arXiv:1907.01845_ (2019).

[8] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. 2019.
ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.
In _International Conference on Learning Representations_ .

[9] Ido Dagan, Oren Glickman, and Bernardo Magnini. 2006. The PASCAL Recognising Textual Entailment Challenge. In _Machine Learning Challenges. Evaluating_
_Predictive Uncertainty, Visual Object Classification, and Recognising Tectual Entail-_
_ment_ . 177–190.




--- end of page=7 ---

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding. In
_Proceedings of the 2019 Conference of the North American Chapter of the Association_
_for Computational Linguistics: Human Language Technologies, Volume 1 (Long and_
_Short Papers)_ . Association for Computational Linguistics, Minneapolis, Minnesota,
[4171–4186. https://doi.org/10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)

[11] William B. Dolan and Chris Brockett. 2005. Automatically Constructing a Corpus

   - f Sentential Paraphrases. In _Proceedings of the Third International Workshop on_
_Paraphrasing (IWP2005)_ .

[12] Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. 2018. Neural architecture
search: A survey. _arXiv preprint arXiv:1808.05377_ (2018).

[13] Angela Fan, Edouard Grave, and Armand Joulin. 2019. Reducing Transformer
Depth on Demand with Structured Dropout. In _International Conference on Learn-_
_ing Representations_ .

[14] Mitchell Gordon, Kevin Duh, and Nicholas Andrews. 2020. Compressing BERT:
Studying the Effects of Weight Pruning on Transfer Learning. In _Proceedings of_
_the 5th Workshop on Representation Learning for NLP_ . 143–155.

[15] Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei,
and Jian Sun. 2020. Single path one-shot neural architecture search with uniform
sampling. In _European Conference on Computer Vision_ . Springer, 544–560.

[16] Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu. 2020.
DynaBERT: Dynamic BERT with Adaptive Width and Depth. _Advances in Neural_
_Information Processing Systems_ 33 (2020).

[17] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. 2019.
Searching for mobilenetv3. In _Proceedings of the IEEE International Conference on_
_Computer Vision_ . 1314–1324.

[18] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang
Wang, and Qun Liu. 2020. TinyBERT: Distilling BERT for Natural Language
Understanding. In _Proceedings of the 2020 Conference on Empirical Methods in_
_Natural Language Processing: Findings_ . 4163–4174.

[19] Haifeng Jin, Qingquan Song, and Xia Hu. 2019. Auto-keras: An efficient neural
architecture search system. In _Proceedings of the 25th ACM SIGKDD International_
_Conference on Knowledge Discovery & Data Mining_ . 1946–1956.

[20] Lukasz Kaiser, Aidan N Gomez, and Francois Chollet. 2018. Depthwise Separable
Convolutions for Neural Machine Translation. In _International Conference on_
_Learning Representations_ .

[21] Antonios Karatzoglou, Nikolai Schnell, and Michael Beigl. 2020. Applying depthwise separable and multi-channel convolutional neural networks of varied kernel
size on semantic trajectories. _Neural Computing and Applications_ 32, 11 (2020),
6685–6698.

[22] Diederik P Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In _ICLR (Poster)_ .

[23] Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2019. Large memory layers with product keys. In
_Advances in Neural Information Processing Systems_ . 8548–8559.

[24] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush
Sharma, and Radu Soricut. 2019. ALBERT: A Lite BERT for Self-supervised
Learning of Language Representations. In _International Conference on Learning_
_Representations_ .

[25] Changlin Li, Jiefeng Peng, Liuchun Yuan, Guangrun Wang, Xiaodan Liang, Liang
Lin, and Xiaojun Chang. 2020. Block-wisely Supervised Neural Architecture
Search with Knowledge Distillation. In _Proceedings of the IEEE/CVF Conference_

_on Computer Vision and Pattern Recognition_ . 1989–1998.

[26] Ting Li, Junbo Zhang, Kainan Bao, Yuxuan Liang, Yexin Li, and Yu Zheng. 2020.
Autost: Efficient neural architecture search for spatio-temporal prediction. In
_Proceedings of the 26th ACM SIGKDD International Conference on Knowledge_
_Discovery & Data Mining_ . 794–802.

[27] Xiang Li, Chen Lin, Chuming Li, Ming Sun, Wei Wu, Junjie Yan, and Wanli
Ouyang. 2020. Improving one-shot nas by suppressing the posterior fading. In
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ .
13836–13845.

[28] Bill Yuchen Lin, Ying Sheng, Nguyen Vo, and Sandeep Tata. 2020. FreeDOM: A
Transferable Neural Architecture for Structured Information Extraction on Web
Documents. In _Proceedings of the 26th ACM SIGKDD International Conference on_
_Knowledge Discovery & Data Mining_ . 1092–1102.

[29] Hanxiao Liu, Karen Simonyan, and Yiming Yang. 2018. DARTS: Differentiable
Architecture Search. In _International Conference on Learning Representations_ .

[30] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A
robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_
(2019).

[31] Renqian Luo, Tao Qin, and Enhong Chen. 2019. Balanced One-shot Neural
[Architecture Optimization. arXiv:1909.10815 [cs.LG]](https://arxiv.org/abs/1909.10815)

[32] Renqian Luo, Xu Tan, Rui Wang, Tao Qin, Enhong Chen, and Tie-Yan Liu. 2020.
Neural architecture search with gbdt. _arXiv preprint arXiv:2007.04785_ (2020).

[33] J Scott McCarley. 2019. Pruning a bert-based question answering model. _arXiv_
_preprint arXiv:1910.06360_ (2019).




[34] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr
Dollár. 2020. Designing network design spaces. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_ . 10428–10436.

[35] Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know What You Don’t Know:
Unanswerable Questions for SQuAD. In _Proceedings of the 56th Annual Meeting of_
_the Association for Computational Linguistics (Volume 2: Short Papers)_ . 784–789.

[36] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.
SQuAD: 100,000+ Questions for Machine Comprehension of Text. In _EMNLP_ .
2383–2392.

[37] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. _arXiv_
_preprint arXiv:1910.01108_ (2019).

[38] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami,
Michael W Mahoney, and Kurt Keutzer. 2020. Q-BERT: Hessian Based Ultra
Low Precision Quantization of BERT.. In _AAAI_ . 8815–8821.

[39] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning,
Andrew Ng, and Christopher Potts. 2013. Recursive Deep Models for Semantic
Compositionality Over a Sentiment Treebank. In _EMNLP_ . 1631–1642.

[40] Kaitao Song, Hao Sun, Xu Tan, Tao Qin, Jianfeng Lu, Hongzhi Liu, and Tie-Yan
Liu. 2020. LightPAFF: A Two-Stage Distillation Framework for Pre-training and
Fine-tuning. _arXiv preprint arXiv:2004.12817_ (2020).

[41] Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. 2019. Patient Knowledge Distillation for BERT Model Compression. In _Proceedings of the 2019 Conference on_
_Empirical Methods in Natural Language Processing and the 9th International Joint_
_Conference on Natural Language Processing (EMNLP-IJCNLP)_ . 4314–4323.

[42] Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny
Zhou. 2020. MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
Devices. In _Proceedings of the 58th Annual Meeting of the Association for Compu-_
_tational Linguistics_ . 2158–2170.

[43] Henry Tsai, Jayden Ooi, Chun-Sung Ferng, Hyung Won Chung, and Jason Riesa.
2020. Finding Fast Transformers: One-Shot Neural Architecture Search by Component Composition. _arXiv preprint arXiv:2008.06808_ (2020).

[44] Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Well-read
students learn better: On the importance of pre-training compact models. _arXiv_
_preprint arXiv:1908.08962_ (2019).

[45] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In _Advances in neural information processing systems_ . 5998–6008.

[46] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R
Bowman. 2018. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In _International Conference on Learning Representa-_
_tions_ .

[47] Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan,
and Song Han. 2020. HAT: Hardware-Aware Transformers for Efficient Natural
Language Processing. In _Proceedings of the 58th Annual Meeting of the Association_
_for Computational Linguistics_ . 7675–7688.

[48] Linnan Wang, Saining Xie, Teng Li, Rodrigo Fonseca, and Yuandong Tian. 2019.
Sample-efficient neural architecture search by learning action space. _arXiv_
_preprint arXiv:1906.06832_ (2019).

[49] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020.
MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of
Pre-Trained Transformers. In _Advances in Neural Information Processing Systems_
_33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS_
_2020, December 6-12, 2020, virtual_, Hugo Larochelle, Marc’Aurelio Ranzato, Raia
[Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). https://proceedings.](https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
[neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

[50] Alex Warstadt, Amanpreet Singh, and Samuel Bowman. 2019. Neural Network
Acceptability Judgments. _Transactions of the Association for Computational Lin-_
_guistics_ 7 (2019), 625–641.

[51] Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A Broad-Coverage
Challenge Corpus for Sentence Understanding through Inference. In _NAACL_ .
1112–1122.

[52] Felix Wu, Angela Fan, Alexei Baevski, Yann Dauphin, and Michael Auli. 2018.
Pay Less Attention with Lightweight and Dynamic Convolutions. In _International_
_Conference on Learning Representations_ .

[53] Canwen Xu, Wangchunshu Zhou, Tao Ge, Furu Wei, and Ming Zhou. 2020. BERT
   - f-Theseus: Compressing BERT by Progressive Module Replacing. In _Proceedings_

_of the 2020 Conference on Empirical Methods in Natural Language Processing_
_(EMNLP)_ . 7859–7869.

[54] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov,
and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language
understanding. In _Advances in neural information processing systems_ . 5753–5763.

[55] Jiahui Yu, Pengchong Jin, Hanxiao Liu, Gabriel Bender, Pieter-Jan Kindermans,
Mingxing Tan, Thomas Huang, Xiaodan Song, Ruoming Pang, and Quoc Le. 2020.
Bignas: Scaling up neural architecture search with big single-stage models. In
_European Conference on Computer Vision_ . Springer, 702–717.

[56] Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. 2019. Q8bert:
Quantized 8bit bert. _arXiv preprint arXiv:1910.06188_ (2019).




--- end of page=8 ---

**A** **REPRODUCIBILITY**

**A.1** **Operation Set and Search Space**


_The Design Choice of the Operation Set._ In addition to MHA
and FFN, LSTM, convolution and variants of MHA and FFN have
achieved good performance in many NLP tasks [5, 20, 21]. We describe the considerations to choose the operations as follows: 1)
LSTM is not considered due to slow training and inference speed.
2) In our preliminary experiments, we try some variants of MHA
and FFN [23, 52], but fail to observe the advantages of small model
size and/or better performance. 3) Considering the parameter size

- f convolution is _𝐾_ ∗ _𝐻_ [2] and separable convolution (SepConv) is
_𝐻_ [2] + _𝐾_ ∗ _𝐻_ where _𝐾_ and _𝐻_ are the kernel size and hidden size,
instead of convolution, we use SepConv with a larger kernel (with
a larger receptive field) without significantly increasing the model
size and the latency. Based on these considerations, we add SepConv
into the candidate operation set.
To determine the possible hidden sizes for operations, we mainly
consider the range of the compressed model sizes. Previous works [37,
41] usually compress the pre-trained BERT model into a small model
(usually no more than 66M) for efficiency and effectiveness. Similarly, in this work, we aim to obtain compressed models of less
than 66M. Therefore, we choose the hidden sizes between 128 and
512 for the candidate operations, which enables a good trade-off
between efficiency and effectiveness [3] .


_The Complexity of the Search Space._ The supernet consists of
_𝐿_ = 24 layers. If we do not use block-wise search, there would be
26 [24] ≈ 10 [34] paths (possible candidate architectures) in the supernet.
We divide the supernet into _𝑁_ = 4 blocks and each block contains 6
layers. Within each block, the hidden size of the 6 layers are required
to be consistent, and the hidden sizes across different blocks can
be different. So there are 5 ∗ 6 [6] = 233280 paths (possible candidate
sub-architectures) in each block, where the first 5 is the number of
candidate hidden sizes and 6 [6] represents that there are 6 operations
in 6 layers. Due to the identity operation, there is an unnecessary
increase in the possible sequence of operations (architecture) as
pointed in Li et al. [25]. For example, the architecture {FFN, identity,
FFN, identity} is equivalent to {FFN, FFN, identity, identity}. Thus
we only keep the architectures that all of the identity operations
are at the end (e.g., {FFN, FFN, identity, identity}) and delete other
redundant architectures. After cleaning the redundancy, the search
space in each block is reduced from the original 233280 to 97650,
which largely improves the sample efficiency. We can select subarchitectures from each block and ensemble them to get a complete
model. Considering _𝑁_ = 4 blocks, there are 97650 [4] (about 10 [20]

possible combinations). Therefore the number of possible models
is reduced from 10 [34] to 10 [20] .


**A.2** **Training Configurations**


_Teacher Model._ We train the BERTbase (L=12, H=768, A=12) [10]
as the teacher model, where L is the number of layers, H is the
hidden size, and A is the number of attention heads. Following Devlin et al. [10], we use BookCorpus plus English Wikipedia as pretraining data (16GB in total). We use Adam [22] with a learning rate


3We do not use a hidden size smaller than 128 since it cannot yield models with enough

accuracy.




- f 1e-4, _𝛽_ 1 = 0 _._ 9 and _𝛽_ 2 = 0 _._ 999. The learning rate is warmed up to
a peak value of 5e-4 for the first 10,000 steps, and then linearly decays. The weight decay is 0.01 and the dropout rate is 0.1. We apply
the best practices proposed in Liu et al. [30] to train the BERTbase

- n 16 NVIDIA V100 GPUs with large batches leveraging gradient
accumulation (2048 samples per batch) and 125000 training steps.


**A.3** **Comparison of Teacher Models**


We present the performance of the teacher model for reproducibility
and compare it with teacher models used in other works in Table 9.
The teacher model (IB-BERT) of MobileBERT [42] achieves the
performance close to the BERTlarge, which is much better than

- ur teacher model and those used in most related works such as

TinyBERT, DynaBERT, DistilBERT. Thus we do not compare NASBERT with MobileBERT in Table 3. Except for MobileBERT, other
methods use BERTbase as the teacher model. Our teacher model
is better than others on average, which is mainly caused by the
volatility of RTE and CoLA (small dataset). After removing these
two datasets, the performance of the teacher model (average score:
89.74) is close to the teacher model of other methods (DistilBERT:
89.73, BERT-of-Theseus: 88.76 and DynaBERT: 89.68). In this way,
NAS-BERT can still show its effectiveness compared with other
approaches, without considering RTE and CoLA.









### Figure 4

Caption: The pipeline of two-stage distillation.**

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


**A.4** **Two-stage distillation**


Two-stage distillation means applying knowledge distillation in
both the pre-training and the fine-tuning stage. Previous meth
- ds [14, 18, 37, 40] have proved that using two-stage distillation is
superior to the single-stage distillation. The pipeline of two-stage
distillation is shown in Fig. 4. The procedure of two-stage distillation can be summarized as follows:


(1) Pre-train the teacher model on the pre-training corpus.
(2) Pre-train the light-weight student model with knowledge
distillation from the pre-trained teacher model in step 1.
(3) Fine-tune the pre-trained teacher model in step 1 on the
downstream task.

(4) Fine-tune the pre-trained student model in step 2 with knowledge distillation from the fine-tuned teacher model in step 3

    - n the target downstream task.

To simplify our expression, we denote the parameter of the
student and the teacher model as _𝜃𝑆_ and _𝜃𝑇_ respectively. We adopt
a general formulation for knowledge distillation in both stages:



L( _𝑥,𝑦_ ; _𝜃𝑆,𝜃𝑇_ ) =



{X _,_ Y}
∑︁ (1 − _𝜆_ ) · L _𝑀𝐿𝐸_ ( _𝑥,𝑦_ ; _𝜃𝑆_ )

( _𝑥,𝑦_ )


+ _𝜆_ - L _𝐾𝐿_ ( _𝑓_ ( _𝑥_ ; _𝜃𝑇_ ) _, 𝑓_ ( _𝑥_ ; _𝜃𝑆_ )) _,_



(2)




--- end of page=9 ---

**Table 9: The accuracy of the teacher models on dev set of the GLUE benchmark. IB-BERT* achieves the performance close to**
**the BERT** large **.**







|Teacher|Method|MNLI -m QQP QNLI CoLA SST -2 STS -B RTE MRPC|AVG|
|---|---|---|---|
|BERTbase|DistilBERT<br>MiNiLM<br>BERT-of-Theseus<br>DynaBERT<br>Ours|86.7<br>89.6<br>91.8<br>56.3<br>92.7<br>89.0<br>69.3<br>88.6<br>84.5<br>91.3<br>91.7<br>58.9<br>93.2<br>-<br>68.6<br>87.3<br>83.5<br>89.8<br>91.2<br>54.3<br>91.5<br>88.9<br>71.1<br>89.5<br>84.8<br>90.9<br>92.0<br>58.1<br>92.9<br>89.8<br>71.1<br>87.7<br>85.2<br>91.0<br>91.3<br>61.0<br>92.9<br>90.3<br>76.0<br>87.7|83.0<br>-<br>82.3<br>83.4<br>84.4|
|IB-BERT*|MobileBERT|87.0<br>-<br>93.2<br>-<br>94.1<br>-<br>-<br>87.3|-|


**Table 10: Architectures of NAS-BERT. “E”, “M”, “F” and “S” refer to Embedding, MHA, FFN and SepConv. “S3” refers to SepConv**
**with the kernel size 3. The number in the subscript refers to the hidden size.**


|Params|Architectures|
|---|---|
|60M|E512 →S3512 →M512 →M512 →S7512 →F512 →F512 →F512 →M512 →M512 →S7512 →F512 →F512<br>→M512 →S5512 →M512 →F512 →S3512 →F512 →M512 →M512 →S5512 →F512 →F512 →F512|
|55M|E512 →F512 →S5512 →M512 →F512 →M512 →F512 →S7512 →M512 →S7512 →M512 →F512 →S5512<br>→S3512 →M512 →S5512 →S5512 →F512 →F512 →F512 →M512 →M512 →S5512 →F512|
|50M|E384 →F512 →S5512 →M512 →F512 →S5512 →F512 →S7512 →M512 →S7512 →F512 →F512 →S3512<br>→M512 →S5512 →S7512 →F512 →M512 →F512 →M512 →S5512 →M512 →F512|
|45M|E384 →S3512 →M512 →S7512 →F512 →S5512 →F512 →S7512 →M512 →S7512 →M512 →F512 →S7512<br>→S3512 →M512 →S7512 →S3512 →F512 →M512 →F512 →M512 →S3512 →S7512 →F512|
|40M|E384 →S3512 →M512 →S5512 →M512 →S5512 →F512 →S5512 →M512 →S7512 →F512 →F512 →S3512<br>→M512 →S5512 →S5512 →S3512 →F512 →S7512 →M512 →M512 →F512|
|35M|E256 →S3512 →S7512 →M512 →S7512 →S3512 →F512 →M512 →S5512 →S7512 →S5512 →S3512 →M512<br>→S5512 →S5512 →S3512 →F512 →S7512 →M512 →M512 →F512|
|30M|E256 →S3512 →M512 →S5512 →S7512 →F512 →S5512 →M512 →S7512 →S5512 →F512 →S3512 →S5512<br>→M512 →F512 →S3512 →S7512 →M512 →S3512 →F512 →S5512|
|25M|E256 →S3384 →M384 →S5384 →S3384 →S3384 →S5384 →S7512 →S5512 →M512 →F512 →F512 →M512<br>→S512 →M512 →M512 →F512|
|20M|E128 →S3384 →M384 →S5384 →S3384 →S3384 →S5384 →M512 →F512 →F512 →M512 →S3384 →S5384<br>→M384 →S5384 →S3384 →S5384|
|15M|E128 →S3384 →M384 →S5384 →S3384 →S3384 →S5384 →S5384 →S5384 →M384 →S5384 →S5384 →S7384<br>→M384 →S5384 →S5384 →S5384 →S5384 →S5384 →S3384 →S5384 →M384 →S5384 →S3384 →S5384|
|10M|E128 →S3384 →S5384 →M384 →S3384 →S5384 →M384 →S7384 →M384 →S3384 →S7384 →M384 →S3384|
|5M|E64 →S3192 →S7192 →M192 →S7192 →S7192 →M192 →S3192 →M192 →S7192 →S5192 →M192 →S3192|



where L _𝑀𝐿𝐸_ is the maximal likelihood loss of the student model
_𝜃𝑆_ - ver the training corpus (X _,_ Y) (i.e., masked language model
loss on the pre-training stage or classification/regression loss on
the fine-tuning stage), and L _𝐾𝐿_ is the KL divergence between the
predicted probability distribution _𝑓_ ( _𝑥_ ; _𝜃𝑇_ ) of the teacher model
_𝜃𝑇_ and the distribution _𝑓_ ( _𝑥_ ; _𝜃𝑆_ ) of the student model _𝜃𝑆_ . _𝜆_ is a
hyper-parameter to trade off L _𝑀𝐿𝐸_ and L _𝐾𝐿_, and is set as 0 _._ 5 in

- ur experiments. There are other advanced distillation techniques
(e.g., embedding distillation or attention distillation) but here we

- nly consider prediction layer distillation.



**A.5** **Architectures of NAS-BERT**


Our NAS-BERT can generate different compressed models given
specific constraints, as described in Section 3.3. Besides the several NAS-BERT models we evaluate in Table 2, we further select
architectures with various sizes, from 5M to 60M with 5M intervals,
yielding a total of 12 different architectures with different sizes. We
present the architectures in Table 10.




--- end of page=10 ---
