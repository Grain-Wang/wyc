---
id: "2020_Once_For_All"
title: "Once-for-All: Train One Network and Specialize it for Efficient Deployment"
authors: ["Han Cai", "Chuang Gan", "Tianzhe Wang", "Zhekai Zhang", "Song Han"]
year: 2020
venue: "ICLR 2020"
publication_status: "FORMALLY PUBLISHED"
category: "Foundation / Training-Free NAS"
source_pdf: "../reference_papers_origin/2020_Once_For_All.pdf"
paper_url: "https://openreview.net/forum?id=HylxE1HKwS"
pdf_url: "https://arxiv.org/pdf/1908.09791"
code_url: "https://github.com/mit-han-lab/once-for-all"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# Once-for-All: Train One Network and Specialize it for Efficient Deployment

**Authors:** Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han

**Venue / Year:** ICLR 2020 (FORMALLY PUBLISHED)

**Category:** Foundation / Training-Free NAS

**Why this paper matters for low-cost post-training LLM NAS:** It establishes elastic supernet training and cheap post hoc subnet specialization, a key cost-amortization baseline for low-cost architecture search.

**Primary record:** [https://openreview.net/forum?id=HylxE1HKwS](https://openreview.net/forum?id=HylxE1HKwS)

**Local source:** [2020_Once_For_All.pdf](../reference_papers_origin/2020_Once_For_All.pdf)

## Full converted text

Published as a conference paper at ICLR 2020

#### - - ONCE FOR-ALL: TRAIN ONE NETWORK AND SPE

##### CIALIZE IT FOR EFFICIENT DEPLOYMENT


**Han Cai** [1] **, Chuang Gan** [2] **, Tianzhe Wang** [1] **, Zhekai Zhang** [1] **, Song Han** [1]

1Massachusetts Institute of Technology, 2MIT-IBM Watson AI Lab
_{_ hancai, chuangg, songhan _}_ @mit.edu


## ABSTRACT


We address the challenging problem of efficient inference across many devices
and resource constraints, especially on edge devices. Conventional approaches
either manually design or use neural architecture search (NAS) to find a specialized
neural network and train it from scratch for _each_ case, which is computationally
prohibitive (causing _CO_ 2 emission as much as 5 cars’ lifetime Strubell et al. (2019))
thus unscalable. In this work, we propose to train a once-for-all (OFA) network that
supports diverse architectural settings by decoupling training and search, to reduce
the cost. We can quickly get a specialized sub-network by selecting from the OFA
network without additional training. To efficiently train OFA networks, we also
propose a novel progressive shrinking algorithm, a generalized pruning method
that reduces the model size across many more dimensions than pruning (depth,
width, kernel size, and resolution). It can obtain a surprisingly large number of subnetworks ( _>_ 10 [19] ) that can fit different hardware platforms and latency constraints
while maintaining the same level of accuracy as training independently. On diverse
edge devices, OFA consistently outperforms state-of-the-art (SOTA) NAS methods
(up to 4.0% ImageNet top1 accuracy improvement over MobileNetV3, or same
accuracy but 1.5 _×_ faster than MobileNetV3, 2.6 _×_ faster than EfficientNet w.r.t
measured latency) while reducing many orders of magnitude GPU hours and _CO_ 2
emission. In particular, OFA achieves a new SOTA 80.0% ImageNet top-1 accuracy
under the mobile setting ( _<_ 600M MACs). OFA is the winning solution for the
3rd Low Power Computer Vision Challenge (LPCVC), DSP classification track
and the 4th LPCVC, both classification track and detection track. Code and 50
pre-trained models (for many devices & many latency constraints) are released at
[https://github.com/mit-han-lab/once-for-all.](https://github.com/mit-han-lab/once-for-all)


## 1 INTRODUCTION


Deep Neural Networks (DNNs) deliver state-of-the-art accuracy in many machine learning applications. However, the explosive growth in model size and computation cost gives rise to new challenges

- n how to efficiently deploy these deep learning models on _diverse_ hardware platforms, since they
have to meet _different_ hardware efficiency constraints (e.g., latency, energy). For instance, one mobile
application on App Store has to support a diverse range of hardware devices, from a high-end Samsung Note10 with a dedicated neural network accelerator to a 5-year-old Samsung S6 with a much
slower processor. With different hardware resources (e.g., on-chip memory size, #arithmetic units),
the optimal neural network architecture varies significantly. Even running on the same hardware,
under different battery conditions or workloads, the best model architecture also differs a lot.


Given different hardware platforms and efficiency constraints (defined as deployment scenarios),
researchers either design compact models specialized for mobile (Howard et al., 2017; Sandler et al.,
2018; Zhang et al., 2018) or accelerate the existing models by compression (Han et al., 2016; He
et al., 2018) for efficient deployment. However, designing specialized DNNs for every scenario
is engineer-expensive and computationally expensive, either with human-based methods or NAS.
Since such methods need to _repeat_ the network design process and _retrain_ the designed network
from scratch for _each_ case. Their total cost grows linearly as the number of deployment scenarios
increases, which will result in excessive energy consumption and _CO_ 2 emission (Strubell et al., 2019).
It makes them unable to handle the vast amount of hardware devices (23.14 billion IoT devices till


1




--- end of page=0 ---

Published as a conference paper at ICLR 2020



train a **once-for-all** network

specialized sub-nets



Previous: O(N) design cost



|Col1|Col2|O e, nync|Col4|76|1<br>.|
|---|---|---|---|---|---|
||**Train**<br> **Get**|<br>** Ma**|~~**73.3**~~||**75.2**|
|||||** mes,**||
|**70.0**|**7**|**0.4**<br>|**in Fou**<br>**e**|** r Ti**<br>**t Four**||
||**67.4**|**Tr**|**a**<br>**G**|<br>||


6 9 12 15 18 21 24

Samsung Note10 Latency (ms)







OFA MobileNetV3



77


75


73


71


69


67











### Figure 1

Caption: Left: a single once-for-all network is trained to support versatile architectural configurations

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
including depth, width, kernel size, and resolution. Given a deployment scenario, a specialized subnetwork is directly selected from the once-for-all network without training. Middle: this approach
reduces the cost of specialized deep learning deployment from O(N) to O(1). Right: once-for-all
network followed by model selection can derive many accuracy-latency trade-offs by training only

- nce, compared to conventional methods tha ~~t r~~ equire repeated training.

2018 [1] ) and highly dynamic deployment environments (different battery conditions, different latency
requirements, etc.).

This paper introduces a new solution to tackle this challenge – designing a _once-for-all network_ that
can be directly deployed under diverse architectural configurations, amortizing the training cost. The
inference is performed by selecting only part of the once-for-all network. It flexibly supports different
depths, widths, kernel sizes, and resolutions without retraining. A simple example of _Once-for-All_
(OFA) is illustrated in Figure 1 (left). Specifically, we decouple the model training stage and the
neural architecture search stage. In the model training stage, we focus on improving the accuracy

- f all sub-networks that are derived by selecting different parts of the once-for-all network. In the

|ecializat<br>edictors|ionstag<br>Given<br>.|e wesam<br>,<br>thetarget<br>uctedtoge<br>f i li|pleasubse<br>hardwarea<br>taspecializ<br>d l|t of sub -ne<br>ndconstra<br>ed sub -ne<br>t kd|
|---|---|---|---|---|
|<br> ecializat<br> edictors<br>|s cond<br>l t|s cond<br>l t|s cond<br>l t|s cond<br>l t|



- f the weights to maintain the accuracy of a large number of sub-networks (more than 10 [19] in our
experiments). It is computationally prohibitive to enumerate all sub-networks to get the exact gradient
in each update step, while randomly sampling a few sub-networks in each step will lead to significant
accuracy drops. The challenge is that different sub-networks are interfering with each other, making
the training process of the whole once-for-all network inefficient. To address this challenge, we
propose a _progressive shrinking_ algorithm for training the once-for-all network. Instead of directly

- ptimizing the once-for-all network from scratch, we propose to first train the largest neural network
with _maximum_ depth, width, and kernel size, then progressively fine-tune the once-for-all network to
support _smaller_ sub-networks that share weights with the larger ones. As such, it provides better
initialization by selecting the most important weights of larger sub-networks, and the opportunity to
distill smaller sub-networks, which greatly improves the training efficiency. From this perspective,
progressive shrinking can be viewed as a generalized network pruning method that shrinks multiple
dimensions (depth, width, kernel size, and resolution) of the full network rather than only the width
dimension. Besides, it targets on maintaining the accuracy of all sub-networks rather than a single
pruned network.


We extensively evaluated the effectiveness of OFA on ImageNet with many hardware platforms
(CPU, GPU, mCPU, mGPU, FPGA accelerator) and efficiency constraints. Under all deployment
scenarios, OFA consistently improves the ImageNet accuracy by a significant margin compared to
SOTA hardware-aware NAS methods while saving the GPU hours, dollars, and _CO_ 2 emission by

- rders of magnitude. On the ImageNet mobile setting (less than 600M MACs), OFA achieves a new
SOTA 80.0% top1 accuracy with 595M MACs (Figure 2). To the best of our knowledge, this is the
first time that the SOTA ImageNet top1 accuracy reaches 80% under the mobile setting.


1https://www.statista.com/statistics/471264/iot-number-of-connected-devices-worldwide/


2




--- end of page=1 ---

Published as a conference paper at ICLR 2020



595M MACs

80.0% Top-1



81







79


77


75


73


71





















69

|ACs<br>O<br>op-1|Col2|Col3|14|xreductio|n|Col7|Col8|Xception|
|---|---|---|---|---|---|---|---|---|
|O<br><br> ACs<br> op-1|nce-for-All (o<br>ﬃ|urs)|Res|NetXt-50|InceptionV|3|||
|MBNetV3|cientNet|NASN<br>De|et-A<br>nseNet-169|||DPN-92||ResNetXt-10|
|Pr<br>Am|oxylessNAS<br>oebaNet<br>|DenseNet-12|1||DenseNet|-264|||
|MB<br>PN<br>Shu<br>DA|NetV2<br>ASNet<br>ﬄeNet<br>RTS<br>Inc|eptionV2|ResNet|-50|||ResNet-10||
|IGCV3|-D|2|M<br>4<br>Model Size|M<br>8|M<br>16|M<br>32<br>|M<br>64<br>**The higher**|M<br><br>** the better**|
|Mo|bileNetV1 (M|BNetV1)||Hand|crafted<br>Aut|oML<br>**The lowe**|** r the better**||


0 1 2 3 4 5 6 7 8 9

MACs (Billion)


### Figure 2

Caption: Comparison between OFA and state-of-the-art CNN models on ImageNet. OFA provides

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
80.0% ImageNet top1 accuracy under the mobile setting ( _<_ 600M MACs).


## 2 RELATED WORK


**Efficient Deep Learning.** Many efficient neural network architectures are proposed to improve the
hardware efficiency, such as SqueezeNet (Iandola et al., 2016), MobileNets (Howard et al., 2017;
Sandler et al., 2018), ShuffleNets (Ma et al., 2018; Zhang et al., 2018), etc. Orthogonal to architecting
efficient neural networks, model compression (Han et al., 2016) is another very effective technique
for efficient deep learning, including network pruning that removes redundant units (Han et al., 2015)

- r redundant channels (He et al., 2018; Liu et al., 2017), and quantization that reduces the bit width
for the weights and activations (Han et al., 2016; Courbariaux et al., 2015; Zhu et al., 2017).


**Neural Architecture Search.** Neural architecture search (NAS) focuses on automating the architecture design process (Zoph & Le, 2017; Zoph et al., 2018; Real et al., 2019; Cai et al., 2018a; Liu et al.,
2019). Early NAS methods (Zoph et al., 2018; Real et al., 2019; Cai et al., 2018b) search for highaccuracy architectures without taking hardware efficiency into consideration. Therefore, the produced
architectures (e.g., NASNet, AmoebaNet) are not efficient for inference. Recent hardware-aware
NAS methods (Cai et al., 2019; Tan et al., 2019; Wu et al., 2019) directly incorporate the hardware
feedback into architecture search. Hardware-DNN co-design techniques (Jiang et al., 2019b;a; Hao
et al., 2019) jointly optimize neural network architectures and hardware architectures. As a result,
they can improve inference efficiency. However, given new inference hardware platforms, these
methods need to repeat the architecture search process and retrain the model, leading to prohibitive
GPU hours, dollars, and _CO_ 2 emission. They are not scalable to a large number of deployment
scenarios. The individually trained models do not share any weight, leading to large total model size
and high downloading bandwidth.


**Dynamic Neural Networks.** To improve the efficiency of a given neural network, some work
explored skipping part of the model based on the input image. For example, Wu et al. (2018); Liu &
Deng (2018); Wang et al. (2018) learn a controller or gating modules to adaptively drop layers; Huang
et al. (2018) introduce early-exit branches in the computation graph; Lin et al. (2017) adaptively
prune channels based on the input feature map; Kuen et al. (2018) introduce stochastic downsampling
point to reduce the feature map size adaptively. Recently, Slimmable Nets (Yu et al., 2019; Yu &
Huang, 2019b) propose to train a model to support multiple width multipliers (e.g., 4 different global
width multipliers), building upon existing human-designed neural networks (e.g., MobileNetV2 0.35,
0.5, 0.75, 1.0). Such methods can adaptively fit different efficiency constraints at runtime, however,
still inherit a pre-designed neural network (e.g., MobileNet-v2), which limits the degree of flexibility
(e.g., only width multiplier can adapt) and the ability in handling new deployment scenarios where
the pre-designed neural network is not optimal. In this work, in contrast, we enable a much more
diverse architecture space (depth, width, kernel size, and resolution) and a significantly larger number

- f architectural settings (10 [19] v.s. 4 (Yu et al., 2019)). Thanks to the diversity and the large design


3




--- end of page=2 ---

Published as a conference paper at ICLR 2020



Full Full Full Full



Elastic

**Resolution**



Elastic Elastic
**Kernel Size** **Depth**

Partial Partial



Elastic Elastic

**Width**



Elastic



Partial Partial Partial



**Depth**



### Figure 3

Caption: Illustration of the progressive shrinking process to support different depth _D_, width _W_,

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
kernel size _K_ and resolution _R_ . It leads to a large space comprising diverse sub-networks ( _>_ 10 [19] ).


space, we can derive new specialized neural networks for many different deployment scenarios rather

progressive shrinking algorithm to tackle this challenge.


## 3 METHOD

|ingneuralnetw<br>enetworktoac<br>mtotacklethis|orkthatlimitsth<br>hievethisflexibi<br>challenge<br>.|eoptimizationh<br>lity whichmot<br>,|eadroom Howev<br>.<br>ivatesustodesig|er it<br>,<br>nthe|
|---|---|---|---|---|
|<br>   ing neural netw<br>   e network to ac<br> m to tackle this|<br>    ork that limits th<br>    hieve this ﬂexibi<br>   challenge.|<br>      e optimization h<br>     lity, which mot|<br>       eadroom. Howev<br>      ivates us to desig||
|<br>   ing neural netw<br>   e network to ac<br> m to tackle this|<br>    ork that limits th<br>    hieve this ﬂexibi<br>   challenge.|<br>      e optimization h<br>     lity, which mot|<br>       eadroom. Howev<br>      ivates us to desig||



## 3.1 PROBLEM FORMALIZATION


Assuming the weights of the once-for-all network as _Wo_ and the architectural configurations as
_{archi}_, we then can formalize the problem as



min
_Wo_




- _Lval_ - _C_ ( _Wo, archi_ )� _,_ (1)

_archi_



where _C_ ( _Wo, archi_ ) denotes a selection scheme that selects part of the model from the once-for-all
network _Wo_ to form a sub-network with architectural configuration _archi_ . The overall training

- bjective is to optimize _Wo_ to make each supported sub-network maintain the _same_ level of accuracy
as _independently_ training a network with the same architectural configuration.


## 3.2 ARCHITECTURE SPACE


Our once-for-all network provides one model but supports many sub-networks of different sizes,
covering four important dimensions of the convolutional neural networks (CNNs) architectures, i.e.,
depth, width, kernel size, and resolution. Following the common practice of many CNN models (He
et al., 2016; Sandler et al., 2018; Huang et al., 2017), we divide a CNN model into a sequence of
units with gradually reduced feature map size and increased channel numbers. Each unit consists of a
sequence of layers where only the first layer has stride 2 if the feature map size decreases (Sandler
et al., 2018). All the other layers in the units have stride 1.


We allow each unit to use arbitrary numbers of layers (denoted as _elastic depth_ ); For each layer,
we allow to use arbitrary numbers of channels (denoted as _elastic width_ ) and arbitrary kernel sizes
(denoted as _elastic kernel size_ ). In addition, we also allow the CNN model to take arbitrary input
image sizes (denoted as _elastic resolution_ ). For example, in our experiments, the input image size
ranges from 128 to 224 with a stride 4; the depth of each unit is chosen from _{_ 2, 3, 4 _}_ ; the width
expansion ratio in each layer is chosen from _{_ 3, 4, 6 _}_ ; the kernel size is chosen from _{_ 3, 5, 7 _}_ .
Therefore, with 5 units, we have roughly ((3 _×_ 3) [2] + (3 _×_ 3) [3] + (3 _×_ 3) [4] ) [5] _≈_ 2 _×_ 10 [19] different
neural network architectures and each of them can be used under 25 different input resolutions. Since
all of these sub-networks share the same weights (i.e., _Wo_ ) (Cheung et al., 2019), we only require
7.7M parameters to store all of them. Without sharing, the total model size will be prohibitive.


## 3.3 TRAINING THE ONCE-FOR-ALL NETWORK


**Na¨ıve Approach.** Training the once-for-all network can be cast as a multi-objective problem, where
each objective corresponds to one sub-network. From this perspective, a na¨ıve training approach
is to directly optimize the once-for-all network from scratch using the exact gradient of the overall

- bjective, which is derived by enumerating all sub-networks in each update step, as shown in Eq. (1).
The cost of this approach is linear to the number of sub-networks. Therefore, it is only applicable to
scenarios where a limited number of sub-networks are supported (Yu et al., 2019), while in our case,
it is computationally prohibitive to adopt this approach.


Another na¨ıve training approach is to sample a few sub-networks in each update step rather than
enumerate all of them, which does not have the issue of prohibitive cost. However, with such a large
number of sub-networks that share weights, thus interfere with each other, we find it suffers from


4




--- end of page=3 ---

Published as a conference paper at ICLR 2020


**Network Pruning**



Train the

full model


Train the

full model



Shrink the model

(4 dimensions)







**Progressive Shrinking**



**single pruned**

**network**


**once-for-all**


**network**



Fine-tune
both large and
small sub-nets



### Figure 4

Caption: Progressive shrinking can be viewed as a generalized network pruning technique with

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
much higher flexibility. Compared to network pruning, it shrinks more dimensions (not only width)
and provides a much more powerful once-for-all network that can fit different deployment scenarios






|uniti|Col2|Col3|
|---|---|---|
|unit i<br>|||



shrink the depth





O1


O2







9x9



3x3 unit i


train with full depth



shrink the depth


|7x|Col2|Col3|7|Col5|Col6|5x5<br>Transform Transform<br>Matrix Matrix<br>25 25|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||



### Figure 5

Caption: Left: Kernel transformation matrix for elastic kernel size. Right: Progressive shrinking for

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
elastic depth. Instead of skipping each layer independently, we keep the first _D_ layers and skip the
last (4 _−_ _D_ ) layers. The weights of the early layers are shared.

significant accuracy drop. In the following section, we introduce a solution to address this challenge,

|s<br>i|i ifi|
|---|---|
|s<br>i|gn<br>.e.,_ pr_|



**Progressive Shrinking.** The once-for-all network comprises many sub-networks of different sizes

|channel mportance early layers are shared.|Col2|Col3|
|---|---|---|
|ng section, we introduce a soluti<br> all network comprises many su|ng section, we introduce a soluti<br> all network comprises many su|ng section, we introduce a soluti<br> all network comprises many su|
|ng section, we introduce a soluti<br> all network comprises many su|we introduce a|we introduce a|
|ng section, we introduce a soluti<br> all network comprises many su|we introduce a|sou<br>  ny su|
|ng section, we introduce a soluti<br> all network comprises many su|||
|ng section, we introduce a soluti<br> all network comprises many su|k comprises ma|k comprises ma|

where small sub-networks are nested in large sub-networks. To prevent interference between the
sub-networks, we propose to enforce a training order from large sub-networks to small sub-networks
in a progressive manner. We name this training scheme as _progressive shrinking_ (PS). An example of
the training process with PS is provided in Figure 3 and Figure 4, where we start with training the
largest neural network with the maximum kernel size (e.g., 7), depth (e.g., 4), and width (e.g., 6).
Next, we progressively fine-tune the network to support smaller sub-networks by gradually adding
them into the sampling space (larger sub-networks may also be sampled). Specifically, after training
the largest network, we first support elastic kernel size which can choose from _{_ 3, 5, 7 _}_ at each layer,
while the depth and width remain the maximum values. Then, we support elastic depth and elastic
width sequentially, as is shown in Figure 3. The resolution is elastic throughout the whole training
process, which is implemented by sampling different image sizes for each batch of training data. We
also use the knowledge distillation technique after training the largest neural network (Hinton et al.,
2015; Ashok et al., 2018; Yu & Huang, 2019b). It combines two loss terms using both the soft labels
given by the largest neural network and the real labels.


Compared to the na¨ıve approach, PS prevents small sub-networks from interfering large sub-networks,
since large sub-networks are already well-trained when the once-for-all network is fine-tuned to
support small sub-networks. Regarding the small sub-networks, they share the weights with the
large ones. Therefore, PS allows initializing small sub-networks with the most important weights of
well-trained large sub-networks, which expedites the training process. Compared to network pruning
(Figure 4), PS also starts with training the full model, but it shrinks not only the width dimension but
also the depth, kernel size, and resolution dimensions of the full model. Additionally, PS fine-tunes
both large and small sub-networks rather than a single pruned network. As a result, PS provides
a much more powerful once-for-all network that can fit diverse hardware platforms and efficiency
constraints compared to network pruning. We describe the details of the PS training flow as follows:


5




--- end of page=4 ---

Published a ~~s a conference pap~~ er at ICLR 2020


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|eratICLR20|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|~~ a~~|~~  co~~|~~  nfe~~|~~  re~~|~~  nc~~|~~  e~~|~~ap~~|~~ap~~|


















|Col1|im|
|---|---|
||i<br>channel<br>|
||sorting<br>O|


|Col1|im|
|---|---|
||i<br>channel<br>sorting<br>O<br>O2|
|||


|channel mportance|Col2|Col3|
|---|---|---|
|portance<br>0.82<br>0.11<br>0.46<br>reorg.||channel|
|portance<br>0.82<br>0.11<br>0.46<br>reorg.|||
|portance<br>0.82<br>0.11<br>0.46<br>reorg.||sorting<br>O1<br>O2<br>O3|
|portance<br>0.82<br>0.11<br>0.46<br>reorg.|||
|portance<br>0.82<br>0.11<br>0.46<br>reorg.|||







### Figure 6

Caption: Progressive shrinking for elastic width. In this example, we progressively support 4, 3, and

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
2 channel settings. We perform channel sorting and pick the most important channels (with large L1
norm) to initialize the smaller channel settings. The important channels’ weights are shared.


_•_ **Elastic Kernel Size** (Figure 5 left). We let the center of a 7x7 convolution kernel also serve as
a 5x5 kernel, the center of which can also be a 3x3 kernel. Therefore, the kernel size becomes
elastic. The challenge is that the centering sub-kernels (e.g., 3x3 and 5x5) are shared and need

|Ela<br>5x<br>las|st|
|---|---|
|** Ela**<br> 5x<br>las|5<br>ti|


|cK<br>ker<br>T<br>.|e|
|---|---|
|** c K**<br> ker<br>. T|n<br> h|


|Col1|Col2|ze(F<br>cent<br>enge|Col4|
|---|---|---|---|
|** nel**<br> l, th<br> ch|**  Si**|**  ze** (<br>  cen<br> eng|**  ze** (<br>  cen<br> eng|
|** nel**<br> l, th<br> ch|e<br> all|e<br> all|t<br> e|


|ft)<br>.<br>hc<br>ece|Col2|
|---|---|
|ft).<br>   h c<br>   e ce|an<br>   n|


|Col1|Col2|thecenter<br>bea3x3ke<br>sub -kernel|nte|Col5|
|---|---|---|---|---|
|e le<br>    also<br>   erin|t|the<br>     be a<br>    sub|the<br>     be a<br>    sub|the<br>     be a<br>    sub|
|e le<br>    also<br>   erin|<br>   g|<br>   g|<br>   g|<br>   g|


|7|convo<br>erefor<br>x3and|Col3|Col4|
|---|---|---|---|
|7|convo<br>       erefor<br>     x3 and|onv<br>       refo<br>     3 a|onv<br>       refo<br>     3 a|
|h<br>     3|h<br>     3|h<br>     3|r<br>     nd|

to play multiple roles (independent kernel and part of a large kernel). The weights of centered
sub-kernels may need to have different distribution or magnitude as different roles. Forcing them
to be the same degrades the performance of some sub-networks. Therefore, we introduce kernel
transformation matrices when sharing the kernel weights. We use separate kernel transformation
matrices for different layers. Within each layer, the kernel transformation matrices are shared
among different channels. As such, we only need 25 _×_ 25 + 9 _×_ 9 = 706 extra parameters to store
the kernel transformation matrices in each layer, which is negligible.


_•_ **Elastic Depth** (Figure 5 right). To derive a sub-network that has _D_ layers in a unit that originally
has _N_ layers, we keep the _first_ D layers and skip the last _N −_ _D_ layers, rather than keeping _any_
_D_ layers as done in current NAS methods (Cai et al., 2019; Wu et al., 2019). As such, one depth
setting only corresponds to one combination of layers. In the end, the weights of the first D layers
are shared between large and small models.


_•_ **Elastic Width** (Figure 6). Width means the number of channels. We give each layer the flexibility
to choose different channel expansion ratios. Following the progressive shrinking scheme, we first
train a full-width model. Then we introduce a channel sorting operation to support partial widths.
It reorganizes the channels according to their importance, which is calculated based on the L1
norm of a channel’s weight. Larger L1 norm means more important. For example, when shrinking
from a 4-channel-layer to a 3-channel-layer, we select the largest 3 channels, whose weights are
shared with the 4-channel-layer (Figure 6 left and middle). Thereby, smaller sub-networks are
initialized with the most important channels on the once-for-all network which is already well
trained. This channel sorting operation preserves the accuracy of larger sub-networks.


## 3.4 SPECIALIZED MODEL DEPLOYMENT WITH ONCE-FOR-ALL NETWORK


Having trained a once-for-all network, the next stage is to derive the specialized sub-network for a
given deployment scenario. The goal is to search for a neural network that satisfies the efficiency
(e.g., latency, energy) constraints on the target hardware while optimizing the accuracy. Since OFA
decouples model training from neural architecture search, we do not need any training cost in this
stage. Furthermore, we build _neural-network-twins_ to predict the latency and accuracy given a neural
network architecture, providing a quick feedback for model quality. It eliminates the repeated search
cost by substituting the measured accuracy/latency with predicted accuracy/latency (twins).


Specifically, we randomly sample 16K sub-networks with different architectures and input image
sizes, then measure their accuracy on 10K validation images sampled from the original training set.
These [architecture, accuracy] pairs are used to train an accuracy predictor to predict the accuracy of
a model given its architecture and input image size [2] . We also build a latency lookup table (Cai et al.,
2019) on each target hardware platform to predict the latency. Given the target hardware and latency
constraint, we conduct an evolutionary search (Real et al., 2019) based on the neural-network-twins
to get a specialized sub-network. Since the cost of searching with neural-network-twins is negligible,


2Details of the accuracy predictor is provided in Appendix A.


6




--- end of page=5 ---

Published as a conference paper at ICLR 2020


w/o PS w/ PS
78


75





73


70


67



|35%<br>.|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|Col30|Col31|Col32|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~||||||||
|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~|~~**3.5%**~~||||||||||||
|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|||**3.3%**|**3.3%**|||**3.4%**|**3.4%**|||**3.7%**|**3.7%**||||||||
|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**||||||||||||||||||||||||
|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**|**3.4%**||||||||||||||||||||||||
|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~||||||||||||||||||||||||
|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~|**2.8%**<br>~~**3.5%**~~||||||||||||||||||||||||||||
|**2.8%**<br>~~**3.5%**~~||||||||||||||||||||||||||||||||
|**2.5%**||||||||||||||||||||||||||||||||
|**2.5%**||||||||||||||||||||||||||||||||
|**2.5%**||||||||||||||||||||||||||||||||


D=2

W=3


K=3



D=2

W=6


K=3



D=2

W=6


K=7



D=2

W=3


K=7



D=4

W=3


K=3



D=4

W=3


K=7



D=4

W=6


K=3



D=4

W=6


K=7



Sub-networks under various architecture configurations

D: depth, W: width, K: kernel size


### Figure 7

Caption: ImageNet top1 accuracy (%) performances of sub-networks under resolution 224 _×_ 224.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
“(D = _d_, W = _w_, K = _k_ )” denotes a sub-network with _d_ layers in each unit, and each layer has an
w ~~idt~~ h expan ~~s~~ ion ra ~~ti~~ - _w_ and kernel size _k_ .


we only need 40 GPU hours to collect the data pairs, and the cost stays constant regardless of
#deployment scenarios.


## 4 EXPERIMENTS


In this section, we first apply the progressive shrinking algorithm to train the once-for-all network on
ImageNet (Deng et al., 2009). Then we demonstrate the effectiveness of our trained once-for-all
network on various hardware platforms (Samsung S7 Edge, Note8, Note10, Google Pixel1, Pixel2,
LG G8, NVIDIA 1080Ti, V100 GPUs, Jetson TX2, Intel Xeon CPU, Xilinx ZU9EG, and ZU3EG
FPGAs) with different latency constraints.


## 4.1 TRAINING THE ONCE-FOR-ALL NETWORK ON IMAGENET


**Training Details.** We use the same architecture space as MobileNetV3 (Howard et al., 2019). For
training the full network, we use the standard SGD optimizer with Nesterov momentum 0.9 and
weight decay 3 _e_ _[−]_ [5] . The initial learning rate is 2.6, and we use the cosine schedule (Loshchilov &
Hutter, 2016) for learning rate decay. The full network is trained for 180 epochs with batch size 2048

- n 32 GPUs. Then we follow the schedule described in Figure 3 to further fine-tune the full network [3] .
The whole training process takes around 1,200 GPU hours on V100 GPUs. This is a one-time training
cost that can be amortized by many deployment scenarios.


**Results.** Figure 7 reports the top1 accuracy of sub-networks derived from the once-for-all networks
that are trained with our progressive shrinking (PS) algorithm and without PS respectively. Due to
space limits, we take 8 sub-networks for comparison, and each of them is denoted as “(D = _d_, W =
_w_, K = _k_ )”. It represents a sub-network that has _d_ layers for all units, while the expansion ratio and
kernel size are set to _w_ and _k_ for all layers. PS can improve the ImageNet accuracy of sub-networks
by a significant margin under all architectural settings. Specifically, without architecture optimization,
PS can achieve 74.8% top1 accuracy using 226M MACs under the architecture setting (D=4, W=3,
K=3), which is on par with MobileNetV3-Large. In contrast, without PS, it only achieves 71.5%,
which is 3.3% lower.


## 4.2 SPECIALIZED SUB-NETWORKS FOR DIFFERENT HARDWARE AND CONSTRAINTS


We apply our trained once-for-all network to get different specialized sub-networks for diverse
hardware platforms: from the cloud to the edge. **On cloud devices**, the latency for GPU is measured
with batch size 64 on NVIDIA 1080Ti and V100 with Pytorch 1.0+cuDNN. The CPU latency is
measured with batch size 1 on Intel Xeon E5-2690 v4+MKL-DNN. **On edge devices**, including
mobile phones, we use Samsung, Google and LG phones with TF-Lite, batch size 1; for mobile GPU,


3Implementation details can be found in Appendix B.


7




--- end of page=6 ---

Published as a conference paper at ICLR 2020

|Model|ImageNet<br>Top1(%)|MACs|Mobile<br>latency|Searchcost<br>(GPUhours)|Trainingcost<br>(GPUhours)|Col7|Total cost (N = 40)|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Model|ImageNet<br>Top1 (%)|MACs|Mobile<br>latency|Search cost<br>(GPU hours)|Training cost<br>(GPU hours)|Training cost<br>(GPU hours)|GPU hours|GPU hours|_CO_2e (lbs)|_CO_2e (lbs)|AWS cost|
|MobileNetV2 [31]<br>MobileNetV2 #1200|72.0<br>73.5|300M<br>300M|66ms<br>66ms|0<br>0|150_N_<br>1200_N_|150_N_<br>1200_N_|6k<br>48k|6k<br>48k|1.7k<br>13.6k|1.7k<br>13.6k|$18.4k<br>$146.9k|
|NASNet-A [44]<br>DARTS[25]|74.0<br>73.1|564M<br>595M|-<br>-|48,000_N_<br>96_N_|-<br>250_N_|-<br>250_N_|1,920k<br>14k|1,920k<br>14k|544.5k<br>4.0k|544.5k<br>4.0k|$5875.2k<br>$42.8k|
|MnasNet [33]<br>FBNet-C [36]<br>ProxylessNAS [4]<br>SinglePathNAS [8]<br>AutoSlim [38]<br>MobileNetV3-Large [15]|74.0<br>74.9<br>74.6<br>74.7<br>74.2<br>75.2|317M<br>375M<br>320M<br>328M<br>305M<br>219M|70ms<br>-<br>71ms<br>-<br>63ms<br>58ms|40,000_N_<br>216_N_<br>200_N_<br>288 + 24_N_<br>180<br>-||-<br>360_N_<br>300_N_<br>384_N_<br>300_N_<br>180_N_|1,6<br>2<br>2<br>1<br>1<br>7.|00k<br>3k<br>0k<br>7k<br>2k<br>2k|453.8k<br>6.5k<br>5.7k<br>4.8k<br>3.4k<br>1.8k||$4896.0k<br>$70.4k<br>$61.2k<br>$52.0k<br>$36.7k<br>$22.2k|
|OFA w/o PS<br>OFA w/ PS<br>OFA w/ PS #25<br>OFA w/ PS #75|72.4<br>**76.0**<br>**76.4**<br>**76.9**|235M<br>230M<br>230M<br>230M|59ms<br>58ms<br>58ms<br>58ms|40<br>40<br>40<br>40|12<br>12|1200<br>1200<br>00 + 25_N_<br>00 + 75_N_|1.<br>1.<br>2.<br>4.|2k<br>2k<br>2k<br>2k|0.34k<br>0.34k<br>0.62k<br>1.2k||$3.7k<br>$3.7k<br>$6.7k<br>$13.0k|
|OFALarge w/ PS #75|**80.0**|595M|-|40|12|00 + 75_N_|4.|2k|1.2k||$13.0k|
|||||||||||||



Table 1: Comparison with SOTA hardware-aware NAS methods on Pixel1 phone. OFA decouples
model training from neural architecture search. The search cost and training cost both stay constant
as the number of deployment scenarios grows. “#25” denotes the specialized sub-networks are
fine-tuned for 25 epochs after grabbing weights from the once-for-all network. “ _CO_ 2 _e_ ” denotes _CO_ 2
emission which is calculated based on Strubell et al. (2019). AWS cost is calculated based on the
price of on-demand P3.16xlarge instances.



ProxylessNAS


FBNet


MnasNet


OFA





**454k**



0 12500 25000 37500 50000


### Figure 8

Caption: OFA saves orders of magnitude design cost compared to NAS methods.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.

we use Jetson TX2 with Pytorch 1.0+cuDNN, batch size of 16; for embedded FPGA, we use Xilinx
ZU9EG and ZU3EG FPGAs with Vitis AI [4], batch size 1.


**Comparison with NAS on Mobile Devices.** Table 1 reports the comparison between OFA and
state-of-the-art hardware-aware NAS methods on the mobile phone (Pixel1). OFA is much more
efficient than NAS when handling multiple deployment scenarios since the cost of OFA is _constant_
while others are _linear_ to the number of deployment scenarios ( _N_ ). **With** _N_ **= 40, the total** _CO_ 2
**emissions of OFA is 16** _×_ **fewer than ProxylessNAS, 19** _×_ **fewer than FBNet, and 1,300** _×_ **fewer**
**than MnasNet (Figure 8).** Without retraining, OFA achieves 76.0% top1 accuracy on ImageNet,
which is 0.8% higher than MobileNetV3-Large while maintaining similar mobile latency. We can
further improve the top1 accuracy to 76.4% by fine-tuning the specialized sub-network for 25 epochs
and to 76.9% by fine-tuning for 75 epochs. Besides, we also observe that OFA with PS can achieve
3.6% better accuracy than without PS.


**OFA under Different Computational Resource Constraints.** Figure 9 summarizes the results

- f OFA under different MACs and Pixel1 latency constraints. OFA achieves 79.1% ImageNet top1
accuracy with 389M MACs, being 2.8% more accurate than EfficientNet-B0 that has similar MACs.
With 595M MACs, OFA reaches a new SOTA 80.0% ImageNet top1 accuracy under the mobile
setting ( _<_ 600M MACs), which is 0.2% higher than EfficientNet-B2 while using 1.68 _×_ fewer MACs.
More importantly, OFA runs much faster than EfficientNets on hardware. Specifically, with 143ms
Pixel1 latency, OFA achieves 80.1% ImageNet top1 accuracy, being 0.3% more accurate and 2.6 _×_
faster than EfficientNet-B2. We also find that training the searched neural architectures from scratch
cannot reach the same level of accuracy as OFA, suggesting that not only neural architectures but
also pre-trained weights contribute to the superior performances of OFA.


### Figure 10

Caption: reports detailed comparisons between OFA and MobileNetV3 on six mobile devices.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
Remarkably, **OFA can produce the entire trade-off curves with many points over a wide range**

**of latency constraints by training only once** (green curve). It is impossible for previous NAS
methods (Tan et al., 2019; Cai et al., 2019) due to the prohibitive training cost.


4https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html


8




--- end of page=7 ---

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||Published~~ a~~|Published~~ a~~||||||||||||||||||
|||Published~~ a~~|Published~~ a~~||||||||||||||||||
|||Published~~ a~~|Published~~ a~~||||||||||||||||||
|||Published~~ a~~|Published~~ a~~|~~ s a con~~|~~ s a con~~|~~  erence~~|~~paper at~~|~~ CLR 2020~~|||||||||||||


OFA OFA - Train from scratch EfficientNet



81


80


79


78


77


76





















|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|81|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||**1.68x**|** MAC**|** s**||)|||||~~**8**~~|~~**01**~~|||**.6**<br>|**x latency**<br>|
||~~**80.**~~|~~**0**~~|<br>**redu**|<br>**ction**|||%||||||~~**.**~~|||<br>**re**|<br>**duction**|
||**79.6**|||||**79.8**|c (|~~80~~|||~~**7**~~|~~**9.8**~~||||||
||||||||c|||||||||||
|~~**79.**~~|||||||t A|||||||||||
||||~~**78**~~||||e|~~79~~||~~**7**~~|~~**8.7**~~|||||||
|~~**2.8%**~~|~~** higher**~~||~~**.**~~||||eN|||||~~**7**~~|~~**2**~~||||~~**788**~~|
|<br>~~**acc**~~|<br>~~**uracy**~~||||||g||||||~~**.**~~||||~~**.**~~|
|||~~**77.**~~|~~**9**~~||||a|~~78~~|||~~**78**~~|~~**.0**~~||||||
||||||||Im|||||||**38%**|** hi**|** gh**|** er**|
||~~**77.6**~~||||||1|||||||**.**<br>**ac**|<br>**cur**|<br>**acy**|<br>|
||||||||p-|77||||||||||
||~~**77.0**~~|||||||||||||||||
||||||||To|||~~**76**~~|~~**4**~~|||||||
||~~**763**~~||||||||||~~**.**~~|||~~**76**~~|~~**3**~~|||
||~~**.**~~|||||||~~76~~||||||~~**.**~~||||
|~~00~~<br>~~4~~|~~0~~<br>~~600~~|~~0~~<br>~~600~~|~~800~~|~~800~~|~~1000~~<br>~~12~~|~~1000~~<br>~~12~~|~~0~~|~~0~~|~~0~~|~~50 100~~|~~50 100~~|~~50 100~~|~~ 150 200~~|~~ 150 200~~|~~ 150 200~~|~~  250 300~~|~~  250 300~~|
||||||~~,~~<br>~~,~~|~~,~~<br>~~,~~||||||||||||
||MAC|MAC|s (M)|s (M)||||||Googl|Googl|Googl|e Pixel1 L|e Pixel1 L|e Pixel1 L|atency (|atency (|
|||||||||||||||||||
|||||||||||||||||||
|~~OFA ac~~|~~ hieves 80~~|~~ hieves 80~~|~~ .0% top~~|~~ .0% top~~|~~  1 accurac~~|~~  1 accurac~~|~~   with 5~~|~~   95M~~|~~   95M~~|~~    MACs an~~|~~    MACs an~~|~~    MACs an~~|~~    d 80.1%~~|~~    d 80.1%~~|~~    d 80.1%~~|~~     top1 a~~|~~     top1 a~~|


143ms Pixel1 latency, setting a new SOTA ImageNet top1 accuracy on the mobile setting.


OFA #25 OFA MobileNetV3



77


75


73


71


69


67


77


75


73


71


69


67



|Col1|Col2|76.|
|---|---|---|
||**74.7**|**76**|
|**73.**|**1**||
|**70.5**||~~**73.3**~~|
||**70.4**||
|**67.4**|||


25 40 55 70 85 100


Samsung S7 Edge Latency (ms)

|Col1|Col2|7|4.9|76|.4|
|---|---|---|---|---|---|
||**73.3**||~~**733**~~||**75.2**|
|**71.4**|||~~**.**~~|||
||**7**|**0.4**||||
|**67**|**.4**|||||



18 24 30 36 42 48 54 60


Google Pixel1 Latency (ms)



|Col1|Col2|Col3|Col4|74|9<br>.|7|6 .1|(%)<br>2|77<br>75|Col11|Col12|Col13|Col14|Col15|75.|76 .6<br>5|Col18|Col19|Col20|Col21|Col22|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||~~**75**~~|~~**75**~~|~~**75**~~||**7**|**7**|**3.6**|**3.6**||||~~**52**~~|~~**52**~~|~~**52**~~|~~**52**~~|
||**728**||||||~~**.**~~|Net Ac<br>|<br>73<br>|<br>73<br>|<br>73<br>|<br>73<br>|<br>73<br>|<br>73<br>|<br>73<br>|||~~**.**~~||||
|**0.4**|**.**||||~~**73.3**~~|||||**71.4**||||||~~**73.3**~~|~~**73.3**~~|||||
|**0.4**||||||||-1 Ima|g<br><br>71|g<br><br>71|g<br><br>71|g<br><br>71|g<br><br>71|g<br><br>71|g<br><br>71|||||||
||**7**|**0.4**|**0.4**||||||||||**70.4**|**70.4**||||||||
|**6**|**4**|||||||To|<br><br>69|**6**|**7.4**|**7.4**||||||||||
|Samsu<br><br>26<br>32<br>**.**|Samsu<br><br>26<br>32<br>**.**|ng Note8 L<br>38<br>44|ng Note8 L<br>38<br>44|ng Note8 L<br>38<br>44|atency (m<br>50<br>56<br>|atency (m<br>50<br>56<br>|s)<br>62<br>68|s)<br>62<br>68|~~67~~<br>7<br>|Samsung Note10<br>9<br>11<br>13<br>15|Samsung Note10<br>9<br>11<br>13<br>15|Samsung Note10<br>9<br>11<br>13<br>15|Samsung Note10<br>9<br>11<br>13<br>15|Samsung Note10<br>9<br>11<br>13<br>15|Samsung Note10<br>9<br>11<br>13<br>15|Latency (m<br>17<br>19<br>2|Latency (m<br>17<br>19<br>2|s)<br>1<br>2|3|||
||||**7**|**7**|**4.7**|**75**|**.8**|Top-1 ImageNet Acc (%)<br><br>69<br>71<br>73<br>75<br>~~77~~<br><br>**2**|Top-1 ImageNet Acc (%)<br><br>69<br>71<br>73<br>75<br>~~77~~<br><br>**2**|||||~~**747**~~|~~**747**~~||**7**|**6.4**|**6.4**|**6.4**|**6.4**|
||**73.4**||||||**75.**|**75.**|**75.**|||**73.0**|**73.0**|~~**.**~~|~~**.**~~|||**75.2**|**75.2**|**75.2**|**75.2**|
|**71.5**|||||~~**73.3**~~|||||**71.1**|**71.1**|||||~~**73.**~~||||||
|||**70.4**||||||||||**70.**|**70.**|**4**|**4**|||||||
|**6**|**74**|||||||||**67**|**67**|**4**|**4**|||||||||


23 28 33 38 43 48 53 58 63 68


Google Pixel2 Latency (ms)



7 10 13 16 19 22 25


LG G8 Latency (ms)



77


75


73


71


69


67


77


75


73


71


69


67













67



### Figure 10

Caption: OFA consistently outperforms MobileNetV3 on mobile platforms.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


**OFA for Diverse Hardware Platforms.** Besides the mobile platforms, we extensively studied the
effectiveness of OFA on six additional hardware platforms (Figure 11) using the ProxylessNAS
architecture space (Cai et al., 2019). OFA consistently improves the trade-off between accuracy and
latency by a significant margin, especially on GPUs which have more parallelism. With similar latency
as MobileNetV2 0.35, “OFA #25” improves the ImageNet top1 accuracy from MobileNetV2’s 60.3%
to 72.6% (+12.3% improvement) on the 1080Ti GPU. Detailed architectures of our specialized models
are shown in Figure 14. It reveals the insight that using the _same_ model for different deployment
scenarios with _only_ the width multiplier modified has a limited impact on efficiency improvement:
the accuracy drops quickly as the latency constraint gets tighter.


**OFA for Specialized Hardware Accelerators.** There has been plenty of work on NAS for generalpurpose hardware, but little work has been focused on specialized hardware accelerators. We
quantitatively analyzed the performance of OFA on two FPGAs accelerators (ZU3EG and ZU9EG)
using Xilinx Vitis AI with 8-bit quantization, and discuss two design principles.


**Principle 1** : memory access is expensive, computation is cheap. An efficient CNN should do _as_
_much as_ computation with _a small amount_ - f memory footprint. The ratio is defined as the arithmetic
intensity (OPs/Byte). The higher OPs/Byte, the less memory bounded, the easier to parallelize.
Thanks to OFA’s diverse choices of sub-network architectures (10 [19] ) (Section 3.3), and the OFA


9




--- end of page=8 ---

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|~~ 2020~~||||


OFA #25 OFA MnasNet MobileNetV2 Slimmable Nets


|77<br>c(%)<br>73 7|Col2|Col3|73.8<br>2.6|75.3|Col6|Col7|76.4|
|---|---|---|---|---|---|---|---|
|c (%)<br>73<br>~~77~~<br>~~**7**~~|c (%)<br>73<br>~~77~~<br>~~**7**~~|c (%)<br>73<br>~~77~~<br>~~**7**~~|~~**2.6**~~<br>**73.8**|**75.3**||||
|et Ac<br>69|et Ac<br>69|et Ac<br>69|||~~**6**~~|~~**9.8**~~|**72.0**|
|1 ImageN<br>66|1 ImageN<br>66|1 ImageN<br>66|~~**6**~~|~~**54**~~||||
|1 ImageN<br>66|1 Im|1 Im||~~**.**~~||||
||Top-|62||||||
||Top-|62|**60.3**|||||


|73.0<br>6|Col2|75.3|Col4|76.1<br>720|Col6|Col7|77<br>73|Col9|Col10|72.0|74.6|75|Col14|.7<br>720|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**6**<br>**73.0**|**6**<br>**73.0**|**75.3**|**75.3**|**720**<br>**76.1**|**720**<br>**76.1**|**720**<br>**76.1**|73<br>~~77~~|73<br>~~77~~||~~**72.0**~~|**74.6**|**75**|**75**|**720**<br>**.7**|**720**<br>**.7**||||||||||||
|**6**<br>**73.0**|**6**<br>**73.0**|**75.3**|**75.3**|**720**<br>**76.1**|**720**<br>**76.1**|**720**<br>**76.1**|||||||||||||||||||||
||||**69**|**.8**|**.**|||69|**71.1**|||~~**69.8**~~|~~**69.8**~~||**.**||||||||||||
|~~**654**~~||||||||66||~~**654**~~|||||||||||||||||
|~~**.**~~||||||||||~~**.**~~|||||||||||||||||
|~~**.**~~||||||||62|||||||||||||||||||
|**60.3**||||||||||**60.3**|||||||||||||||||



|77<br>73<br>7<br>69<br>66<br>62|Col2|75.|4 75.8|Col5|77<br>73|Col7|Col8|73.6|Col10|
|---|---|---|---|---|---|---|---|---|---|
|62<br>66<br>69<br>73<br>77<br>**7**|**72.9**||||||**72.8**|**72.8**|**72.8**|
|62<br>66<br>69<br>73<br>77<br>**7**|~~**0**~~**.3**|||**72.0**|**72.0**|||||
|62<br>66<br>69<br>73<br>77<br>**7**||~~**65.4**~~|~~**69.8**~~||~~66~~<br>69<br>**67.0**|**69.4**|**69.0**|**71.5**||
|62<br>66<br>69<br>73<br>77<br>**7**|||||62<br>**59**|**63.3**||||
|62<br>66<br>69<br>73<br>77<br>**7**|**60.3**|||||**.1**||||


1.5 2.0 2.5 3.0 3.5 4.0


Xilinx ZU9EG FPGA Latency (ms)

Batch Size = 1 (Quantized)



~~58~~


58



30 45 60 75 90 105


Jetson TX2 Latency (ms)

Batch Size = 16



|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**67.0**|**69.6**||**69.0**|**71.5**|
||**63.3**||||
|**5**|**9.1**||||


3.0 4.0 5.0 6.0 7.0 8.0


Xilinx ZU3EG FPGA Latency (ms)

Batch Size = 1 (Quantized)



10 14 18 22 26 30


NVIDIA 1080Ti Latency (ms)

Batch Size = 64



4 6 8 10 12


NVIDIA V100 Latency (ms)

Batch Size = 64











9 11 13 15 17 19

Intel Xeon CPU Latency (ms)

Batch Size = 1


**73.7**

**72.8**















77


73


69


66


62


58


58









~~58~~


77


73


69


66


62


58



### Figure 11

Caption: Specialized OFA models consistently achieve significantly higher ImageNet accuracy

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
with s ~~imilar latency than non-spe~~ ciali ~~zed neural networks on CPU~~, GP ~~U, mGPU, and FPGA. More~~
remarkably, specializing for a new hardware platform does not add training cost using OFA.





0.0



0.0


|11: Sp<br>milarla|ecialized<br>tenc tha|OFA<br>nnon-|Col4|mo<br>s e|
|---|---|---|---|---|
|<br>ably, sp|~~ y~~<br>ecializin|<br>g for a|<br>g for a|~~   p~~<br>   ne|
|||||~~M~~|
|0.0|||||
|0.0|||||
|7.5|||||
|7.5|||||
|00<br>2.5<br>5.0|||||
|00<br>2.5<br>5.0|||||
|00<br>2.5<br>5.0|||||


|sistentl<br>neural|yachie<br>network|Col3|vesi<br>son|gnifi<br>CPU|
|---|---|---|---|---|
|<br>    areplat|<br>form do|<br>form do|<br> es n|<br>  ot ad|
|~~2~~|~~MnasN~~|~~MnasN~~|~~et~~||
|0.0<br>|||||
|0.0<br>|||||
|0.0|||||
|0.0|||||
|00<br>0.0<br>0.0|||||
|00<br>0.0<br>0.0|||||
|00<br>0.0<br>0.0|||||


|yhi<br>U|gherIm<br>mGPU|ageN<br>andF|Col4|etac<br>PGA|curac<br>Mor|Col7|
|---|---|---|---|---|---|---|
|~~,~~<br>    nin|~~,~~<br>g cost u|<br>  sing O|<br>  sing O|<br> FA.|~~   .~~|~~   .~~|
|~~Ours~~|~~)~~||||||
||80.0||||||
||80.0||||||
|PGA<br>/s)|60.0||||||
|PGA<br>/s)|60.0||||||
|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|||||
|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|||||
|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|ZU3EG<br>(GOPS<br>00<br>20.0<br>40.0|||||



### Figure 12

Caption: OFA models improve the arithmetic intensity (OPS/Byte) and utilization (GOPS/s)

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
compared with the MobileNetV2 and MnasNet (measured results on Xilinx ZU9EG and ZU3EG
FPGA).


model twin that can quickly give the accuracy/latency feedback (Section 3.4), the evolutionary search
can automatically find a CNN architecture that has higher arithmetic intensity. As shown in Figure 12,
OFA’s arithmetic intensity is 48%/43% higher than MobileNetV2 and MnasNet (MobileNetV3 is
not supported by Xilinx Vitis AI). Removing the memory bottleneck results in higher utilization and
GOPS/s by 70%-90%, pushing the operation point to the upper-right in the roofline model (Williams
et al., 2009), as shown in Figure 13. (70%-90% looks small in the log scale but that is significant).


**Principle 2** : the CNN architecture should be co-designed with the hardware accelerator’s cost model.
The FPGA accelerator has a specialized depth-wise engine that is pipelined with the point-wise
engine. The pipeline throughput is perfectly matched for 3x3 kernels. As a result, OFA’s searched
model only has 3x3 kernel (Figure 14, a) on FPGA, despite 5x5 and 7x7 kernels are also in the search
space. Additionally, large kernels sometimes cause “out of BRAM” error on FPGA, giving high cost.
On Intel Xeon CPU, however, more than 50% operations are large kernels. Both FPGA and GPU
models are wider than CPU, due to the large parallelism of the computation array.


## 5 CONCLUSION


We proposed _Once-for-All_ (OFA), a new methodology that decouples model training from architecture
search for efficient deep learning deployment under a large number of hardware platforms. Unlike


10




--- end of page=9 ---

Published as a conference paper at ICLR 2020


(a) on Xilinx ZU9EG FPGA (b) on Xilinx ZU3EG FPGA


### Figure 13

Caption: Quantative study of OFA’s roofline model on Xilinx ZU9EG and ZU3EG FPGAs (log

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
scale). OFA model increased the arithmetic intensity by 33%/43% and GOPS/s by 72%/92% on these
two FPGAs compared with MnasNet.


(a) 4.1ms latency on Xilinx ZU3EG (batch size = 1).


(c) 14.9ms latency on NVIDIA 1080Ti (batch size = 64).


### Figure 14

Caption: OFA can design specialized models for different hardware and different latency constraint.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
“MB4 3x3” means “mobile block with expansion ratio 4, kernel size 3x3”. FPGA and GPU models are
wider than CPU model due to larger parallelism. Different hardware has different cost model, leading
to different optimal CNN architectures. OFA provides a unified and efficient design methodology.


previous approaches that design and train a neural network for _each_ deployment scenario, we designed
a _once-for-all network_ that supports different architectural configurations, including elastic depth,
width, kernel size, and resolution. It reduces the training cost (GPU hours, energy consumption, and
_CO_ 2 emission) by orders of magnitude compared to conventional methods. To prevent sub-networks

- f different sizes from interference, we proposed a progressive shrinking algorithm that enables
a large number of sub-network to achieve the same level of accuracy compared to training them
independently. Experiments on a diverse range of hardware platforms and efficiency constraints
demonstrated the effectiveness of our approach. OFA provides an automated ecosystem to efficiently
design efficient neural networks with the hardware cost model in the loop.


ACKNOWLEDGMENTS


We thank NSF Career Award #1943349, MIT-IBM Watson AI Lab, Google-Daydream Research
Award, Samsung, Intel, Xilinx, SONY, AWS Machine Learning Research Award for supporting this


11




--- end of page=10 ---

Published as a conference paper at ICLR 2020


research. We thank Samsung, Google and LG for donating mobile phones. We thank Shuang Wu and
Lei Deng for drawing the Figure 2.


## REFERENCES


Anubhav Ashok, Nicholas Rhinehart, Fares Beainy, and Kris M Kitani. N2n learning: Network to
network compression via policy gradient reinforcement learning. In _ICLR_, 2018.


Han Cai, Tianyao Chen, Weinan Zhang, Yong Yu, and Jun Wang. Efficient architecture search by
network transformation. In _AAAI_, 2018a.


Han Cai, Jiacheng Yang, Weinan Zhang, Song Han, and Yong Yu. Path-level network transformation
for efficient architecture search. In _ICML_, 2018b.


Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task
and hardware. In _ICLR_ [, 2019. URL https://arxiv.org/pdf/1812.00332.pdf.](https://arxiv.org/pdf/1812.00332.pdf)


Brian Cheung, Alex Terekhov, Yubei Chen, Pulkit Agrawal, and Bruno Olshausen. Superposition of
many models into one. In _NeurIPS_, 2019.


Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Binaryconnect: Training deep neural
networks with binary weights during propagations. In _NeurIPS_, 2015.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _CVPR_, 2009.


Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single
path one-shot neural architecture search with uniform sampling. _arXiv preprint arXiv:1904.00420_,
2019.


Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for
efficient neural network. In _NeurIPS_, 2015.


Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks
with pruning, trained quantization and huffman coding. In _ICLR_, 2016.


Cong Hao, Xiaofan Zhang, Yuhong Li, Sitao Huang, Jinjun Xiong, Kyle Rupnow, Wen-mei Hwu,
and Deming Chen. Fpga/dnn co-design: An efficient design methodology for 1ot intelligence on
the edge. In _2019 56th ACM/IEEE Design Automation Conference (DAC)_, pp. 1–6. IEEE, 2019.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In _CVPR_, 2016.


Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. Amc: Automl for model
compression and acceleration on mobile devices. In _ECCV_, 2018.


Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv_
_preprint arXiv:1503.02531_, 2015.


Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun
Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenetv3. In _ICCV_
_2019_, 2019.


Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand,
Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for
mobile vision applications. _arXiv preprint arXiv:1704.04861_, 2017.


Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected
convolutional networks. In _CVPR_, 2017.


Gao Huang, Danlu Chen, Tianhong Li, Felix Wu, Laurens van der Maaten, and Kilian Q Weinberger.
Multi-scale dense networks for resource efficient image classification. In _ICLR_, 2018.


12




--- end of page=11 ---

Published as a conference paper at ICLR 2020


Forrest N Iandola, Song Han, Matthew W Moskewicz, Khalid Ashraf, William J Dally, and Kurt
Keutzer. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and¡ 0.5 mb model size.
_arXiv preprint arXiv:1602.07360_, 2016.


Weiwen Jiang, Lei Yang, Edwin Sha, Qingfeng Zhuge, Shouzhen Gu, Yiyu Shi, and Jingtong
Hu. Hardware/software co-exploration of neural architectures. _arXiv preprint arXiv:1907.04650_,
2019a.


Weiwen Jiang, Xinyi Zhang, Edwin H-M Sha, Lei Yang, Qingfeng Zhuge, Yiyu Shi, and Jingtong Hu.
Accuracy vs. efficiency: Achieving both through fpga-implementation aware neural architecture
search. In _Proceedings of the 56th Annual Design Automation Conference 2019_, pp. 1–6, 2019b.


Jason Kuen, Xiangfei Kong, Zhe Lin, Gang Wang, Jianxiong Yin, Simon See, and Yap-Peng
Tan. Stochastic downsampling for cost-adjustable inference and improved regularization in
convolutional networks. In _CVPR_, 2018.


Ji Lin, Yongming Rao, Jiwen Lu, and Jie Zhou. Runtime neural pruning. In _NeurIPS_, 2017.


Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan
Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. In _ECCV_,
2018.


Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. In _ICLR_,
2019.


Lanlan Liu and Jia Deng. Dynamic deep neural networks: Optimizing accuracy-efficiency trade-offs
by selective execution. In _AAAI_, 2018.


Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning
efficient convolutional networks through network slimming. In _ICCV_, 2017.


Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. _arXiv_
_preprint arXiv:1608.03983_, 2016.


Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for
efficient cnn architecture design. In _ECCV_, 2018.


Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. Regularized evolution for image
classifier architecture search. In _AAAI_, 2019.


Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In _CVPR_, 2018.


Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep
learning in nlp. In _ACL_, 2019.


Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and
Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. In _Proceedings of the_
_IEEE Conference on Computer Vision and Pattern Recognition_, pp. 2820–2828, 2019.


Xin Wang, Fisher Yu, Zi-Yi Dou, Trevor Darrell, and Joseph E Gonzalez. Skipnet: Learning dynamic
routing in convolutional networks. In _ECCV_, 2018.


Samuel Williams, Andrew Waterman, and David Patterson. Roofline: An insightful visual performance model for floating-point programs and multicore architectures. Technical report, Lawrence
Berkeley National Lab.(LBNL), Berkeley, CA (United States), 2009.


Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian,
Peter Vajda, Yangqing Jia, and Kurt Keutzer. Fbnet: Hardware-aware efficient convnet design via
differentiable neural architecture search. In _CVPR_, 2019.


Zuxuan Wu, Tushar Nagarajan, Abhishek Kumar, Steven Rennie, Larry S Davis, Kristen Grauman,
and Rogerio Feris. Blockdrop: Dynamic inference paths in residual networks. In _CVPR_, 2018.


13




--- end of page=12 ---

Published as a conference paper at ICLR 2020


Jiahui Yu and Thomas Huang. Autoslim: Towards one-shot architecture search for channel numbers.
_arXiv preprint arXiv:1903.11728_, 2019a.


Jiahui Yu and Thomas Huang. Universally slimmable networks and improved training techniques. In
_ICCV_, 2019b.


Jiahui Yu, Linjie Yang, Ning Xu, Jianchao Yang, and Thomas Huang. Slimmable neural networks. In
_ICLR_, 2019.


Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient
convolutional neural network for mobile devices. In _CVPR_, 2018.


Chenzhuo Zhu, Song Han, Huizi Mao, and William J Dally. Trained ternary quantization. In _ICLR_,
2017.


Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. In _ICLR_, 2017.


Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V Le. Learning transferable architectures
for scalable image recognition. In _CVPR_, 2018.


## A DETAILS OF THE ACCURACY PREDICTOR



We use a three-layer feedforward neural ~~network that has 400 hidden units in each layer as the~~

convergence, the root-mean-square error (R ~~MSE) between predicted accuracy and estimated accuracy~~

|etwork tha|t has 400 h|idden units|in each la|yer as the|
|---|---|---|---|---|
|~~de each laye~~|~~r in the neu~~|~~  ral network~~|~~   into a one-~~|~~    hot vector~~|
|and we ass|ign zero ve|ctors to la|ers that ar|e skipped.|
|<br>~~or that repr~~|<br>~~ esents the i~~|<br>~~  nput image~~|<br>~~   size. We co~~|<br>~~    ncatenate~~|
|<br>~~ts the whol~~|<br>~~  neural net~~|<br>~~  ork archite~~|<br>~~   cture and in~~|<br>~~    ut imae~~|
|<br>~~f~~|<br>~~ l~~|<br>~~   h~~|<br>~~   i~~|~~    p g~~<br>|
|~~orward ne~~<br>|~~ ura networ~~<br>|~~  to get te~~<br>|~~   redcted ac~~<br>|~~    curacy. In~~<br>|
|~~iction mod~~<br>|~~ el canprovi~~<br>|~~de very acc~~<br>|~~ uratepredi~~<br>|~~ctions. At~~<br>|


- n the test set is only 0.21%. Figure 15 shows the relationship between the RMSE of the accuracy
prediction model and the final results (i.e., the accuracy of selected sub-networks). We can find that
lower RMSE typically leads to better final results.


75.5


74.6


73.8


72.9


72.0

0 5 10 15 20

RMSE of Acc Prediction Model (%)


### Figure 15

Caption: Performances of selected sub-networks using different accuracy prediction model.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


## B IMPLEMENTATION DETAILS OF PROGRESSIVE SHRINKING


After training the full network, we first have one stage of fine-tuning to incorporate elastic kernel size.
In this stage (i.e., _K ∈_ [7 _,_ 5 _,_ 3]), we sample one sub-network in each update step. The network is
fine-tuned for 125 epochs with an initial learning rate of 0.96. All other training settings are the same
as training the full network.


14




--- end of page=13 ---

Published as a conference paper at ICLR 2020


Next, we have two stages of fine-tuning to incorporate elastic depth. We sample two sub-networks
and aggregate their gradients in each update step. The first stage (i.e., _D ∈_ [4 _,_ 3]) takes 25 epochs
with an initial learning rate of 0.08 while the second stage (i.e., _D ∈_ [4 _,_ 3 _,_ 2]) takes 125 epochs with
an initial learning rate of 0.24.


Finally, we have two stages of fine-tuning to incorporate elastic width. We sample four sub-networks
and aggregate their gradients in each update step. The first stage (i.e., _W ∈_ [6 _,_ 4]) takes 25 epochs
with an initial learning rate of 0.08 while the second stage (i.e., _W ∈_ [6 _,_ 4 _,_ 3]) takes 125 epochs with
an initial learning rate of 0.24.


15




--- end of page=14 ---
