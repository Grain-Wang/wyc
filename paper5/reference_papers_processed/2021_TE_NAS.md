---
id: "2021_TE_NAS"
title: "Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective"
authors: ["Wuyang Chen", "Xinyu Gong", "Zhangyang Wang"]
year: 2021
venue: "ICLR 2021"
publication_status: "FORMALLY PUBLISHED"
category: "Training-Free NAS / Architecture Proxy"
source_pdf: "../reference_papers_origin/2021_TE_NAS.pdf"
paper_url: "https://openreview.net/forum?id=Cnon5ezMHtu"
pdf_url: "https://arxiv.org/pdf/2102.11535"
code_url: "https://github.com/VITA-Group/TENAS"
converter: "PyMuPDF4LLM 0.2.3 + PyMuPDF Layout 1.26.6"
---

# Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective

**Authors:** Wuyang Chen, Xinyu Gong, Zhangyang Wang

**Venue / Year:** ICLR 2021 (FORMALLY PUBLISHED)

**Category:** Training-Free NAS / Architecture Proxy

**Why this paper matters for low-cost post-training LLM NAS:** It combines trainability and expressivity signals into a low-cost search rule and is a useful theoretical proxy baseline.

**Primary record:** [https://openreview.net/forum?id=Cnon5ezMHtu](https://openreview.net/forum?id=Cnon5ezMHtu)

**Local source:** [2021_TE_NAS.pdf](../reference_papers_origin/2021_TE_NAS.pdf)

## Full converted text

Published as a conference paper at ICLR 2021

## NEURAL ARCHITECTURE SEARCH ON IMAGENET IN FOUR GPU HOURS: A THEORETICALLY INSPIRED PERSPECTIVE


**Wuyang Chen, Xinyu Gong, Zhangyang Wang**
Department of Electrical and Computer Engineering
The University of Texas at Austin, Austin, TX, USA
_{_ wuyang.chen,xinyu.gong,atlaswang _}_ @utexas.edu


## ABSTRACT


Neural Architecture Search (NAS) has been explosively studied to automate the
discovery of top-performer neural networks. Current works require heavy training of supernet or intensive architecture evaluations, thus suffering from heavy
resource consumption and often incurring search bias due to truncated training

     - r approximations. Can we select the best neural architectures without involving
any training and eliminate a drastic portion of the search cost? We provide an
affirmative answer, by proposing a novel framework called _training-free neural_
_architecture search_ ( **TE-NAS** ). TE-NAS ranks architectures by analyzing the spectrum of the neural tangent kernel (NTK) and the number of linear regions in the
input space. Both are motivated by recent theory advances in deep networks and
can be computed without any training and any label. We show that: (1) these
two measurements imply the _trainability_ and _expressivity_     - f a neural network; (2)
they strongly correlate with the network’s test accuracy. Further on, we design a
pruning-based NAS mechanism to achieve a more flexible and superior trade-off
between the trainability and expressivity during the search. In NAS-Bench-201
and DARTS search spaces, TE-NAS completes high-quality search but only costs
**0.5** and **4** GPU hours with one 1080Ti on CIFAR-10 and ImageNet, respectively.
We hope our work inspires more attempts in bridging the theoretical findings of
deep networks and practical impacts in real NAS applications. Code is available at:
[https://github.com/VITA-Group/TENAS.](https://github.com/VITA-Group/TENAS)


## 1 INTRODUCTION

The recent development of deep networks significantly contributes to the success of computer vision.
Thanks to many efforts by human designers, the performance of deep networks have been significantly
boosted (Krizhevsky et al., 2012; Simonyan & Zisserman, 2014; Szegedy et al., 2015; He et al., 2016;
Xie et al., 2017). However, the manual creation of new network architectures not only costs enormous
time and resources due to trial-and-error, but also depends on the design experience that does not
always scale up. To reduce the human efforts and costs, neural architecture search ( **NAS** ) has recently
amassed explosive interests, leading to principled and automated discovery for good architectures in
a given search space of candidates (Zoph & Le, 2016; Brock et al., 2017; Pham et al., 2018; Liu et al.,
2018a; Chen et al., 2018; Bender et al., 2018; Gong et al., 2019; Chen et al., 2020a; Fu et al., 2020).


As an optimization problem, NAS faces two core questions: 1) “ **how to evaluate** ”, i.e. the objective
function that defines what are good architectures we want; 2) “ **how to optimize** ”, i.e. by what means
we could effectively optimize the objective function. These two questions are entangled and highly
non-trivial, since the search spaces are of extremely high dimension, and the generalization ability

- f architectures cannot be easily inferred (Dong & Yang, 2020; Dong et al., 2020). Existing NAS
methods mainly leverage the validation set and conduct accuracy-driven architecture optimization.
They either formulate the search space as a super-network (“supernet”) and make the training loss
differentiable through the architecture parameters (Liu et al., 2018b), or treat the architecture selection
as a sequential decision making process (Zoph & Le, 2016) or evolution of genetics (Real et al.,
2019). However, these NAS algorithms suffer from heavy consumption of both time and GPU
resources. Training a supernet till convergence is extremely slow, even with many effective heuristics
for sampling or channel approximations (Dong & Yang, 2019; Xu et al., 2019). Approximated proxy


1




--- end of page=0 ---

Published as a conference paper at ICLR 2021


inference such as truncated training/early stopping can accelerate the search, but is well known to
introduce search bias to the inaccurate results obtained (Pham et al., 2018; Liang et al., 2019; Tan
et al., 2020). The heavy search cost not only slows down the discovery of novel architectures, but
also blocks us from more meaningfully understanding the NAS behaviors.


On the other hand, the analysis of neural network’s trainability (how effective a network can be

- ptimized via gradient descent) and expressivity (how complex the function a network can represent)
has witnessed exciting development recently in the deep learning theory fields. By formulating neural
networks as a Gaussian Process (no training involved), the gradient descent training dynamics can be
characterized by the Neural Tangent Kernel (NTK) of infinite (Lee et al., 2019) or finite (Yang, 2019)
width networks, from which several useful measures can be derived to depict the network trainability
at the initialization. Hanin & Rolnick (2019a;b); Xiong et al. (2020) describe another measure of
network expressivity, also without any training, by counting the number of unique linear regions that
a neural network can divide in its input space. We are therefore inspired to ask:


- _**How to optimize**_ _NAS at network’s initialization without involving any training, thus significantly_
_eliminating a heavy portion of the search cost?_

- _Can we define_ _**how to evaluate**_ _in NAS by analyzing the trainability and expressivity of architectures,_
_and further benefit our understanding of the search process?_


Our answers are **yes** to both questions. In this work, we propose TE-NAS, a framework for trainingfree neural architecture search. We leverage _two indicators_, the condition number of NTK and
the number of linear regions, that can decouple and effectively characterize the trainability and
expressivity of architectures respectively in complex NAS search spaces. Most importantly, these
two indicators can be measured in a training-free and label-free manner, thus largely accelerates
the NAS search process and benefits the understanding of discovered architectures. To our best
knowledge, TE-NAS makes the first attempt to bridge the theoretical findings of deep neural networks
and real-world NAS applications. While we intend not to claim that the two indicators we use are the

- nly nor the best options, we hope our work opens a door to theoretically-inspired NAS and inspires
the discovery of more deep network indicators. Our contributions are summarized as below:


- We identify and investigate two training-free and label-free indicators to rank the quality of deep
architectures: the spectrum of their NTKs, and the number of linear regions in their input space.
Our study finds that they reliably indicate the trainability and expressivity of a deep network
respectively, and are strongly correlated with the network’s test accuracy.

- We leverage the above two theoretically-inspired indicators to establish a training-free NAS
framework, **TE-NAS**, therefore eliminating a drastic portion of the search cost. We further
introduce a pruning-based mechanism, to boost search efficiency and to more flexibly trade-off
between trainability and expressivity.

- In NAS-Bench-201/DARTS search spaces, **TE-NAS** discovers architectures with a strong performance at remarkably lower search costs, compared to previous efforts. With just one 1080Ti, it

 - nly costs 0.5 GPU hours to search on CIFAR10, and 4 GPU hours on ImageNet, respectively,
setting the new record for ultra-efficient yet high-quality NAS.


## 2 RELATED WORKS


**Neural architecture search (NAS)** is recently proposed to accelerate the principled and automated
discovery of high-performance networks. However, most works suffer from heavy search cost, for
both weight-sharing based methods (Liu et al., 2018b; Dong & Yang, 2019; Liu et al., 2019; Yu
et al., 2020a; Li et al., 2020a; Yang et al., 2020a) and single-path sampling-based methods (Pham
et al., 2018; Guo et al., 2019; Real et al., 2019; Tan et al., 2020; Li et al., 2020c; Yang et al., 2020b).
A one-shot super network can share its parameters to sampled sub-networks and accelerate the
architecture evaluations, but it is very heavy and hard to optimize and suffers from a poor correlation
between its accuracy and those of the sub-networks (Yu et al., 2020c). Sampling-based methods
achieve more accurate architecture evaluations, but their truncated training still imposes bias to the
performance ranking since this is based on the results of early training stages.


Instead of estimating architecture performance by direct training, people also try to predict network’s
accuracy (or ranking), called **predictor based NAS** methods (Liu et al., 2018a; Luo et al., 2018; Dai
et al., 2019; Luo et al., 2020). Graph neural network (GNN) is a popular choice as the predictor
model (Wen et al., 2019; Chen et al., 2020b). Siems et al. (2020) even propose the first large-scale


2




--- end of page=1 ---

Published as a conference paper at ICLR 2021


surrogate benchmark, where most of the architectures’ accuracies are predicted by a pretrained GNN
predictor. The learned predictor can achieve highly accurate performance evaluation. However,
the data collection step - sampling representative architectures and train them till converge - still
requires extremely high cost. People have to sample and train 2,000 to 50,000 architectures to serve
as the training data for the predictor. Moreover, none of these works can demonstrate the cross-space
transferability of their predictors. This means one has to repeat the data collection and predictor
training whenever facing an unseen search space, which is highly nonscalable.


The heavy cost of architecture evaluation hinders the **understanding** - f the NAS search process. Recent pioneer works like Shu et al. (2019) observed that DARTS and ENAS tend to favor architectures
with wide and shallow cell structures due to their smooth loss landscape. Siems et al. (2020) studied
the distribution of test error for different cell depths and numbers of parameter-free operators. Chen
& Hsieh (2020) for the first time regularizes the Hessian norm of the validation loss and visualizes
the smoother loss landscape of the supernet. Li et al. (2020b) proposed to approximate the validation
loss landscape by learning a mapping from neural architectures to their corresponding validate losses.
Still, these analyses cannot be directly leveraged to guide the design of network architectures.


Mellor et al. (2020) recently proposed a NAS framework that does not involve training, which
shares the same motivation with us towards training-free architecture search at initialization. They
empirically find that the correlation between sample-wise input-output Jacobian can indicate the
architecture’s test performance. However, why does the Jacobian work is not well explained and
demonstrated. Their search performance on NAS-Bench-201 is still left behind by the state-of-the-art
NAS works, and they did not extend to DARTs space.


Meanwhile, we see the evolving development of **deep learning theory** - n neural networks. NTK
(neural tangent kernel) is proposed to characterize the gradient descent training dynamics of infinite
wide (Jacot et al., 2018) or finite wide deep networks (Hanin & Nica, 2019). Wide networks are also
proved to evolve as linear models under gradient descent (Lee et al., 2019). This is further leveraged
to decouple the trainability and generalization of networks (Xiao et al., 2019). Besides, a natural
measure of ReLU network’s expressivity is the number of linear regions it can separate in its input
space (Raghu et al., 2017; Montufar, 2017; Serra et al., 2018; Hanin & Rolnick, 2019a;b; Xiong´
et al., 2020). In our work, we for the first time discover two important indicators that can effectively
rank architectures, thus bridging the theoretic findings and real-world NAS applications. Instead of
claiming the two indicators we discover are the best, we believe there are more meaningful properties

- f deep networks that can benefit the architecture search process. We leave them as open questions
and encourage the community to study.


## 3 METHODS


The core motivation of our TE-NAS framework is to achieve architecture evaluation without involving
any training, to significantly accelerate the NAS search process and reduce the search cost. In section
3.1 we present our study on two important indicators that reflect the trainability and expressivity

- f a neural network, and in section 3.2 we design a novel pruning-based method that can achieve a
superior trade-off between the two indicators.


## 3.1 ANALYZING TRAINABILITY AND EXPRESSIVITY OF DEEP NETWORKS


Trainability and expressivity are distinct notions regarding a neural network (Xiao et al., 2019). A
network can achieve high performance only if the function it can represent is complex enough and at
the same time, it can be effectively trained by gradient descent.


## 3.1.1 TRAINABILITY BY CONDITION NUMBER OF NTK


The trainability of a neural network indicates how effective it can be optimized using gradient descent
(Burkholz & Dubatovka, 2019; Hayou et al., 2019; Shin & Karniadakis, 2020). Although some heavy
networks can theoretically represent complex functions, they not necessarily can be effectively trained
by gradient descent. One typical example is that, even with a much more number of parameters, Vgg
networks (Simonyan & Zisserman, 2014) usually perform worse and require more special engineering
tricks compared with ResNet family (He et al., 2016), whose superior trainability property is studied
by Yang & Schoenholz (2017).


3




--- end of page=2 ---

Published as a conference paper at ICLR 2021


Recent work (Jacot et al., 2018; Lee et al., 2019; Chizat et al., 2019) studied the gradient descent
training of neural networks using a quantity called the neural tangent kernel (NTK). The finite width
NTK is defined by Θ( [ˆ] _**x**_ _,_ _**x**_ _[′]_ ) = _J_ ( _**x**_ ) _J_ ( _**x**_ _[′]_ ) _[T]_, where _Jiα_ ( _**x**_ ) = _∂θα_ _zi_ _[L]_ [(] _**[x]**_ [)][ is the Jacobian evaluated at]
a point _**x**_ for parameter _θα_, and _zi_ _[L]_ [is the output of the] _[ i]_ [-th neuron in the last output layer] _[ L]_ [.]


Lee et al. (2019) further proves that wide neural networks evolve as linear models using gradient
descent, and their training dynamics is controlled by ODEs that can be solved as

_µt_ ( _**X**_ train) = ( **I** _−_ _e_ _[−][η]_ [ ˆΘ][train] _[t]_ ) _**Y**_ train (1)

for training data. Here _µt_ ( _**x**_ ) = E[ _zi_ _[L]_ [(] _**[x]**_ [)]][ is the expected outputs of an infinitely wide network.][ ˆΘ][train]
denotes the NTK between the training inputs, and _**X**_ train and _**Y**_ train are drawn from the training set
_D_ train. As the training step _t_ tends to infinity we can see that Eq. 1 reduce to _µ_ ( _**X**_ train) = _**Y**_ train.


The relationship between the conditioning of Θ [ˆ] and the trainability of networks is studied by Xiao
et al. (2019), and we brief the conclusion as below. We can write Eq. 1 in terms of the spectrum of Θ:

_µt_ ( _**X**_ train) _i_ = ( **I** _−_ _e_ _[−][ηλ][i][t]_ ) _**Y**_ train _,i,_ (2)



where _λi_ are the eigenvalues of Θ [ˆ] train and we order
the eigenvalues _λ_ 0 _≥· · · ≥_ _λm_ . As it has been
hypothesized by Lee et al. (2019) that the maximum
feasible learning rate scales as _η ∼_ 2 _/λ_ 0, plugging
this scaling for _η_ into Eq. 2 we see that the _λm_
will converge exponentially at a rate given by 1 _/κ_,
where _κ_ = _λ_ 0 _/λm_ is the condition number. Then
we can conclude that if the _κ_ - f the NTK associated
with a neural network diverges then it will become
untrainable, so we use _κ_ as a metric for trainability:



70


60


50


40


30


20


10



|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
|||||K|endall-tau = -0.42|
| | | | | | |


0 200000 NTK400000 _∑_ = _[∏]_ _∏_ _[max]_ _min_ 600000 800000



72 _._ 5


70 _._ 0


67 _._ 5


65 _._ 0


62 _._ 5


60 _._ 0


57 _._ 5



40 60 NTK80 _∑_ = _[∏]_ _∏_ _[max]_ _min_ 100 120



### Figure 1

Caption: ** Condition number of NTK _κN_ exhibits

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
negative correlation with the test accuracy of architectures in NAS-Bench201 (Dong & Yang, 2020).



_κN_ = _[λ]_ [0] _._ (3)

_λm_



_κN_ is calculated without any gradient descent or label. Figure 1 demonstrates that the _κN_ is
negatively correlated with the architecture’s test accuracy, with the Kendall-tau correlation as _−_ 0 _._ 42.
Therefore, minimizing the _κN_ during the search will encourage the discovery of architectures with
high performance.


## 3.1.2 EXPRESSIVITY BY NUMBER OF LINEAR REGIONS



The expressivity of a neural network indicates how complex the function
it can represent (Hornik et al., 1989; Giryes et al., 2016). For ReLU
networks, each ReLU function defines a linear boundary and divides
its input space into two regions. Since the composition of piecewise
linear functions is still piecewise linear, every ReLU network can be
seen as a piecewise linear function. The input space of a ReLU network
can be partitioned into distinct pieces (i.e. linear regions) (Figure 2),
each of which is associated with a set of affine parameters, and the
function represented by the network is affine when restricted to each
piece. Therefore, it is natural to measure the expressivity of a ReLU
network with the number of linear regions it can separate.



### Figure 2

Caption: ** Example of linear

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
regions divided by a ReLU
network [1]



Following Raghu et al. (2017); Serra et al. (2018); Hanin & Rolnick (2019a;b); Xiong et al. (2020),
we introduce the definition of activation patterns and linear regions for ReLU networks.


**Definition 1. Activation Patterns and Linear Regions (Xiong et al. (2020))** _Let N be a ReLU_
_CNN. An activation pattern of N is a function_ _**P**_ _from the set of neurons to {_ 1 _, −_ 1 _}, i.e., for each_
_neuron z in N_ _, we have_ _**P**_ ( _z_ ) _∈{_ 1 _, −_ 1 _}. Let θ be a fixed set of parameters (weights and biases) in_
_N_ _, and_ _**P**_ _be an activation pattern. The region corresponding to_ _**P**_ _and θ is_

_**R**_ ( _**P**_ ; _θ_ ) := _{_ _**x**_ [0] _∈_ R _[C][×][H][×][W]_ : _z_ ( _**x**_ [0] ; _θ_ ) _·_ _**P**_ ( _z_ ) _>_ 0 _,_ _∀z ∈N},_ (4)

_where z_ ( _**x**_ [0] ; _θ_ ) _is the pre-activation of a neuron z. Let RN_ _,θ denote the number of linear re-_
_gions of N at θ, i.e., RN_ _,θ_ := # _{_ _**R**_ ( _**P**_ ; _θ_ ) : _**R**_ ( _**P**_ ; _θ_ ) _̸_ = _∅_ _for some activation pattern_ _**P**_ _}._

1Plot is generated by us with the same method described by Hanin & Rolnick (2019a).


4




--- end of page=3 ---

Published as a conference paper at ICLR 2021


Eq. 4 tells us that a linear region in the input space is a
set of input data _**x**_ [0] that satisfies a certain fixed activation
pattern _**P**_ ( _z_ ), and therefore the number of linear regions
_RN_ _,θ_ measures how many unique activation patterns that
can be divided by the network.


In our work, we repeat the measurement of the number of
linear regions by sampling network parameters from the
Kaiming Norm Initialization (He et al., 2015), and calculate the average as the approximation to its expectation:


_R_ ˆ _N ≃_ E _θRN_ _,θ_ (5)


We iterate through all architectures in NAS-Bench-201
(Dong & Yang, 2020), and calculate their numbers of
linear regions (without any gradient descent or label).
### Figure 3

Caption: demonstrates that the number of linear regions

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
is positively correlated with the architecture’s test accuracy, with the Kendall-tau correlation as 0 _._ 5. Therefore,
maximizing the number of linear regions during the
search will also encourage the discovery of architectures with high performance.



70


60


50


40


30


20


10


0




|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
||||||K|endall-ta|u = 0.50|



0 500 1000 _R_ ˆ _N_ : #Linear Regions1500 2000 2500 3000 3500

### Figure 3

Caption: ** Number of linear regions _R_ [ˆ] _N_ - f

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
architectures in NAS-Bench201 exhibits positive correlation with test accuracies.



0 _._ 35


0 _._ 30


0 _._ 25


0 _._ 20


0 _._ 15


0 _._ 10


0 _._ 05

















0 _._ 00

|Col1|NTKκN|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||<br>Linear Reg|ions_ RN_||0.306<br>0.312|0.311||
||0.279||||||
|~~0192~~|||||||
|~~.~~|||||||
||0.103|0.133<br>~~0.135~~|0.133<br>~~0.135~~||~~0091~~|~~0.139~~|
||||||~~.~~||
||||||||

zero skip-connect conv 1 _×_ 1 conv 3 _×_ 3 avg ~~p~~      - ol 3 _×_ 3



Finally, in Figure 4 we analyze the operator composi- zero _×_ _×_ ~~p~~ _×_
tion of top 10% architecture by maximizing _R_ [ˆ] _N_ and **Figure 4:** _κN_ and _R_ [ˆ] _N_ prefer different operators

in NAS-Bench201.

minimizingˆ _κN_, respectively. We can clearly see that
_RN_ and _κN_ have different preferences for choosing operators. They both choose a large ratio of
conv3 _×_ 3 for high generalization performance. But meanwhile, _R_ [ˆ] _N_ heavily selects conv1 _×_ 1, and
_κN_ leads to skip-connect, favoring the gradient flow.



### Figure 4

Caption: ** _κN_ and _R_ [ˆ] _N_ prefer different operators

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
in NAS-Bench201.



## 3.2 PRUNING-BY-IMPORTANCE ARCHITECTURE SEARCH


Given the strong correlation between the architecture’s test accuracy and its _κN_ and _R_ [ˆ] _N_, how to build
an efficient NAS framework on top of them? We motivate this section by addressing two questions:
_1. How to combine κN and_ _R_ [ˆ] _N together, with a good explicit trade-off?_


We first need to turn the two measurements _κN_ and _R_ [ˆ] _N_ into one combined function, based on which
we can rank architectures. As seen in Figure 1 and 3, the magnitudes of _κN_ and _R_ [ˆ] _N_ differ much.
To avoid one overwhelming the other numerically, one possible remedy is normalization; but we
cannot pre-know the ranges nor the value distributions of _κN_ and _R_ [ˆ] _N_, before computing them over a
search space. In order to make our combined function _well defined before search_ and _agnostic to the_
_search space_, instead of using the numerical values of _κN_ and _R_ [ˆ] _N_, we could refer to their relative
rankings. Specifically, each time by comparing the sampled set of architectures peer-to-peer, we can
directly sum up the two relative rankings of _κN_ and _R_ [ˆ] _N_ as the selection criterion. The equal-weight
summation treats trainability and expressivity with the same importance conceptually [1] and delivers
the best empirical result: we thus choose it as our default combined function. We also tried some

- ther means to combine the two, and the ablation studies can be found in Appendix D.2.
_2. How to search more efficiently?_


Sampling-based methods like reinforcement learning or evolution can use rankings as the reward

- r filtering metric. However, they are inefficient, especially for complex cell-based search space.
Consider a network stacked by repeated cells (directed acyclic graphs) (Zoph et al., 2018; Liu et al.,
2018b). Each cell has _E_ edges, and on each edge we only select one operator out of _|O|_ ( _O_ is the
set of operator candidates). There are _|O|_ _[E]_ unique cells, and for sampling-based methods, _γ · |O|_ _[E]_
networks have to be sampled during the search. The ratio _γ_ can be interpreted as the sampling
efficiency: a method with small _γ_ can find good architectures faster. However, the search time cost of
sampling-based methods still scales up with the size of the search space, i.e., _|O|_ _[E]_ .


Inspired by recent works on pruning-from-scratch (Lee et al., 2018; Wang et al., 2020), we propose
a pruning-by-importance NAS mechanism to quickly shrink the search possibilities and boost the


1We tried some weighted summations of the two, and find their equal-weight summation to perform the best.


5




--- end of page=4 ---

Published as a conference paper at ICLR 2021


efficiency further, reducing the cost from _|O|_ _[E]_ to _|O| · E_ . Specifically, we start the search with a
super-network _N_ 0 composed of all possible operators and edges. In the outer loop, for every round we
prune one operator on each edge. The outer-loop stops when the current supernet _Nt_ is a single-path
network [2], i.e., the algorithm will return us the final searched architecture. For the inner-loop, we
measure the change of _κN_ and _R_ [ˆ] _N_ before and after pruning each individual operator, and assess its
importance using the sum of two ranks. We order all currently available operators in terms of their
importance, and prune the lowest-importance operator on each edge.


The whole pruning process is extremely fast. As we will demonstrate later, our approach is principled
and can be applied to different spaces without making any modifications. This pruning-by-importance
mechanism may also be extended to indicators beyond _κN_ and _R_ [ˆ] _N_ . We summarize our training-free
and pruning-based NAS framework, TE-NAS, in Algorithm 1.


**Algorithm 1:** TE-NAS: Training-free Pruning-based NAS via Ranking of _κN_ and _R_ [ˆ] _N_ .


**1 Input:** supernet _N_ 0 stacked by cells, each cell has _E_ edges, each edge has _|O|_ - perators, step _t_ = 0.

**2 while** _Nt is not a single-path network_ **do**

**3** **for** _each operator oj in Nt_ **do**

**4** ∆ _κt,oj_ = _κNt −_ _κNt\oj_ _▷_ the higher ∆ _κt,oj_ the more likely we will prune _oj_

**5** ∆ _Rt,oj_ = _RNt −_ _RNt\oj_ _▷_ the lower ∆ _Rt,oj_ the more likely we will prune _oj_

**6** Get importance by _κN_ : _sκ_ ( _oj_ ) = index of _oj_ in descendingly sorted list [∆ _κt,o_ 1 _, ...,_ ∆ _κt,o|Nt|_ ]

**7** Get importance by _RN_ : _sR_ ( _oj_ ) = index of _oj_ in ascendingly sorted list [∆ _Rt,o_ 1 _, ...,_ ∆ _Rt,o|Nt|_ ]

**8** Get importance _s_ ( _oj_ ) = _sκ_ ( _oj_ ) + _sR_ ( _oj_ )

**9** _Nt_ +1 = _Nt_

**10** **for** _each edge ei, i_ = 1 _, ..., E_ **do**

**11** _j_ _[∗]_ = arg min _j{s_ ( _oj_ ) : _oj ∈_ _ei}_ _▷_ find the operator with greatest importance on each edge.

**12** _Nt_ +1 = _Nt_ +1 _\oj_ _[∗]_


**13** _t_ = _t_ + 1


**14 return** _Pruned single-path network Nt._



## 3.2.1 VISUALIZATION OF SEARCH PROCESS


TE-NAS benefits us towards a better understanding

- f the search process. We can analyze the trajectory of _κN_ and _R_ [ˆ] _N_ during the search. It is worth
noting that our starting point _N_ 0, the un-pruned
supernet, is assumed to be of the highest expressivity (as it is composed of all operators in the search
space and has the largest number of parameters and
ReLU functions). However, it has poor trainability,
as people find many engineering techniques are required to effectively training the supernet (Yu et al.,
2020a;b). Therefore, during pruning we are expecting to strengthen the trainability of the supernet,
while retaining its expressivity as much as possible.


As we observe in Figure 5, the supernet _N_ is first
pruned by quickly reducing _κN_, i.e., increasing the
network’s trainability. After that, as the improvement of _κN_ is almost plateaued, the method carefully fine-tunes the architecture without sacrificing
too much expressivity _R_ [ˆ] _N_ .



200


175


150


125


100


75


50


1600


1400


1200


1000


800


600


400


200


0



|Col1|Col2|Col3|Col4|0|Col6|
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
|3||2|||1|
| | | | | | |


2250.0 2500.0 _R_ ˆ _N_ : #Linear Regions2750.0 3000.0 3250.0


|Col1|Col2|Col3|Col4|Col5|0|Col7|
|---|---|---|---|---|---|---|
||||||||
|||||||~~1~~|
||||||||
||||||||
||||||||
||||||||
|6|||~~5~~|4|2<br>3|2<br>3|



3250.0 3262.5 _R_ ˆ _N_ : #Linear Regions3275.0 3287.5 3300.0


### Figure 5

Caption: ** Pruning trajectory on NAS-Bench-201

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
(top) and DARTs search space (bottom). Number “0”
indicates the supernet _N_ 0 before any pruning, which
is of high expressivity but poor trainability.



2Different search spaces may have different criteria for the single-path network. In NAS-Bench201 (Dong
& Yang, 2020) each edge only keeps one operator at the end of the search, while in DARTS space (Liu et al.,
2018b) there are two operators on each edge in the searched network.


6




--- end of page=5 ---

Published as a conference paper at ICLR 2021


## 4 EXPERIMENTS


In this section, we evaluate our TE-NAS on two search spaces: NAS-Bench-201 (Dong & Yang,
2020) and DARTS (Liu et al., 2018b). Search and training protocols are summarized in Appendix A.
[Our code is available at: https://github.com/VITA-Group/TENAS.](https://github.com/VITA-Group/TENAS)


## 4.1 RESULTS ON NAS-BENCH-201


NAS-Bench-201 (Dong & Yang, 2020) provides a standard cell-based search space (containing
15,625 architectures) and a database of architecture’s performance evaluated under a unified protocol.
The network’s test accuracy can be directly obtained by querying the database, which facilitates
people to focus on studying NAS algorithms without network evaluation. NAS-Bench-201 supports
three datasets (CIFAR-10, CIFAR-100, ImageNet-16-120 (Chrabaszcz et al., 2017)). The operation
space contains _none_ ( _zero_ ), _skip connection_, _conv_ 1 _×_ 1, _conv_ 3 _×_ 3 _convolution_, and _average pooling_
3 _×_ 3. We refer to their paper for details of the space. Our search is dataset-specific, i.e. the search
and evaluation are conducted on the same dataset.


**Table 1:** Comparison with state-of-the-art NAS methods on NAS-Bench-201. Test accuracy with mean and
deviation are reported. “optimal” indicates the best test accuracy achievable in NAS-Bench-201 search space.


**Search Cost** **Search**
**Architecture** **CIFAR-10** **CIFAR-100** **ImageNet-16-120** **(GPU sec.)** **Method**


ResNet (He et al., 2016) 93.97 70.86 43.63  -

RSPS (Li & Talwalkar, 2020) 87 _._ 66(1 _._ 69) 58 _._ 33(4 _._ 34) 31 _._ 14(3 _._ 88) 8007.13 random
ENAS (Pham et al., 2018) 54 _._ 30(0 _._ 00) 15 _._ 61(0 _._ 00) 16 _._ 32(0 _._ 00) 13314.51 RL
DARTS (1st) (Liu et al., 2018b) 54 _._ 30(0 _._ 00) 15 _._ 61(0 _._ 00) 16 _._ 32(0 _._ 00) 10889.87 gradient
DARTS (2nd) (Liu et al., 2018b) 54 _._ 30(0 _._ 00) 15 _._ 61(0 _._ 00) 16 _._ 32(0 _._ 00) 29901.67 gradient
GDAS (Dong & Yang, 2019) 93 _._ 61(0 _._ 09) 70 _._ 70(0 _._ 30) 41 _._ 84(0 _._ 90) 28925.91 gradient


NAS w.o. Training (Mellor et al., 2020) 91 _._ 78(1 _._ 45) 67 _._ 05(2 _._ 89) 37 _._ 07(6 _._ 39) 4.8 training-free
TE-NAS (ours) **93** _._ **9** ( **0** _._ **47** ) **71** _._ **24** ( **0** _._ **56** ) **42** _._ **38** ( **0** _._ **46** ) 1558 training-free


**Optimal** 94.37 73.51 47.31  -

We run TE-NAS for four independent times with different random seeds, and report the mean and
standard deviation in Table 1. We can see that TE-NAS achieves the best accuracy on all three
datasets, and largely reduces the search cost (5 _× ∼_ 19 _×_ reduction). Although Mellor et al. (2020)
requires even less search time (by only sampling 25 architectures), they suffer from much inferior
accuracy performance, with notably larger deviations across different search rounds.


## 4.2 RESULTS ON CIFAR-10 WITH DARTS SEARCH SPACE


**Architecture Space** The DARTs operation space _O_ contains eight choices: _none_ ( _zero_ ), _skip_
_connection_, _separable convolution_ 3 _×_ 3 and 5 _×_ 5, _dilated separable convolution_ 3 _×_ 3 and 5 _×_ 5,
_max pooling_ 3 _×_ 3, _average pooling_ 3 _×_ 3. Following previous works (Liu et al., 2018b; Chen et al.,
2019; Xu et al., 2019), for evaluation phases, we stack 20 cells to compose the network and set the


**Table 2:** Comparison with state-of-the-art NAS methods on CIFAR-10.


**Test Error** **Params** **Search Cost** **Search**
**Architecture**
**(%)** **(M)** **(GPU days)** **Method**


AmoebaNet-A (Real et al., 2019) 3 _._ 34(0 _._ 06) 3.2 3150 evolution
PNAS (Liu et al., 2018a) _[⋆]_ 3 _._ 41(0 _._ 09) 3.2 225 SMBO
ENAS (Pham et al., 2018) 2.89 4.6 0.5 RL
NASNet-A (Zoph et al., 2018) 2.65 3.3 2000 RL


DARTS (1st) (Liu et al., 2018b) 3 _._ 00(0 _._ 14) 3.3 0.4 gradient
DARTS (2nd) (Liu et al., 2018b) 2 _._ 76(0 _._ 09) 3.3 1.0 gradient
SNAS (Xie et al., 2018) 2 _._ 85(0 _._ 02) 2.8 1.5 gradient
GDAS (Dong & Yang, 2019) 2.82 2.5 0.17 gradient
BayesNAS (Zhou et al., 2019) 2 _._ 81(0 _._ 04) 3.4 0.2 gradient
ProxylessNAS (Cai et al., 2018) _[†]_ 2.08 5.7 4.0 gradient
P-DARTS (Chen et al., 2019) 2.50 3.4 0.3 gradient
PC-DARTS (Xu et al., 2019) 2 _._ 57(0 _._ 07) 3.6 0.1 gradient
SDARTS-ADV (Chen & Hsieh, 2020) 2 _._ 61(0 _._ 02) 3.3 1.3 gradient


TE-NAS (ours) 2 _._ 63(0 _._ 064) 3.8 0.05 _[‡]_ training-free


_⋆_ No cutout augmentation.

_†_ Different space: PyramidNet (Han et al., 2017) as the backbone.

_‡_ Recorded on a single GTX 1080Ti GPU.


7




--- end of page=6 ---

Published as a conference paper at ICLR 2021


initial channel number as 36. We place the reduction cells at the 1/3 and 2/3 of the network and each
cell consists of six nodes.


**Results** We run TE-NAS for four independent times with different random seeds, and report the
mean and standard deviation. Table 2 summarizes the performance of TE-NAS compared with other
popular NAS methods. TE-NAS achieves a test error of 2.63%, ranking among the top of recent NAS
results, but meanwhile largely reduces the search cost to only 0.05 GPU-day. ProxylessNAS achieves
the lowest test error, but it searches on a different space with a much longer search time and has a
larger model size. Besides, Mellor et al. (2020) did not extend to their Jacobian-based framework to
DARTs search space for CIFAR-10 or ImageNet classification.


## 4.3 RESULTS ON IMAGENET WITH DARTS SEARCH SPACE


**Architecture Space** Following previous works (Xu et al., 2019; Chen et al., 2019), the architecture
for ImageNet is slightly different from that for CIFAR-10. During retraining evaluation, the network
is stacked with 14 cells with the initial channel number set to 48, and we follow the mobile setting to
control the FLOPs not exceed 600 MB by adjusting the channel number. The spatial resolution is
downscaled from 224 _×_ 224 to 28 _×_ 28 with the first three convolution layers of stride 2.


**Results** As shown in Table 3, we achieve a top-1/5 test error of 24.5%/7.5%, achieving competitive
performance with recent state-of-the-art works in the ImageNet mobile setting. However, TE-NAS

- nly cost four GPU hours with only one 1080Ti. Searching on ImageNet takes a longer time than on
CIFAR-10 due to the larger input size and more network parameters.


**Table 3:** Comparison with state-of-the-art NAS methods on ImageNet under the mobile setting.


**Architecture** **Test Error(%)** **Params** **Search Cost** **Search**
top-1 top-5 **(M)** **(GPU days)** **Method**


NASNet-A (Zoph et al., 2018) 26.0 8.4 5.3 2000 RL
AmoebaNet-C (Real et al., 2019) 24.3 7.6 6.4 3150 evolution
PNAS (Liu et al., 2018a) 25.8 8.1 5.1 225 SMBO
MnasNet-92 (Tan et al., 2019) 25.2 8.0 4.4        - RL


DARTS (2nd) (Liu et al., 2018b) 26.7 8.7 4.7 4.0 gradient
SNAS (mild) (Xie et al., 2018) 27.3 9.2 4.3 1.5 gradient
GDAS (Dong & Yang, 2019) 26.0 8.5 5.3 0.21 gradient
BayesNAS (Zhou et al., 2019) 26.5 8.9 3.9 0.2 gradient
P-DARTS (CIFAR-10) (Chen et al., 2019) 24.4 7.4 4.9 0.3 gradient
P-DARTS (CIFAR-100) (Chen et al., 2019) 24.7 7.5 5.1 0.3 gradient
PC-DARTS (CIFAR-10) (Xu et al., 2019) 25.1 7.8 5.3 0.1 gradient
TE-NAS (ours) 26.2 8.3 6.3 0.05 training-free


PC-DARTS (ImageNet) (Xu et al., 2019) _[†]_ 24.2 7.3 5.3 3.8 gradient
ProxylessNAS (GPU) (Cai et al., 2018) _[†]_ 24.9 7.5 7.1 8.3 gradient
TE-NAS (ours) _[†]_ 24.5 7.5 5.4 0.17 training-free


_†_ The architecture is searched on ImageNet, otherwise it is searched on CIFAR-10 or CIFAR-100.


## 5 CONCLUSION


The key questions in Neural Architecture Search (NAS) are “what are good architectures” and “how
to find them”. Validation loss or accuracy are possible answers but not enough, due to their search
bias and heavy evaluation cost. Our work demonstrates that two theoretically inspired indicators, the
spectrum of NTK and the number of linear regions, not only strongly correlate with the network’s
performance, but also benefit the reduced search cost and decoupled analysis of the network’s
trainability and expressivity. Without involving any training, our TE-NAS achieve competitive NAS
performance with minimum search time. We for the first time bridge the gap between the theoretic
findings of deep neural networks and real-world NAS applications, and we encourage the community
to further explore more meaningful network properties so that we will have a better understanding of
good architectures and how to search them.


ACKNOWLEDGEMENT


This work is supported in part by the NSF Real-Time Machine Learning program (Award Number:
2053279), and the US Army Research Office Young Investigator Award (W911NF2010240).


8




--- end of page=7 ---

Published as a conference paper at ICLR 2021


## REFERENCES


Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding
and simplifying one-shot architecture search. In _International Conference on Machine Learning_,
pp. 549–558, 2018.


Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Smash: one-shot model
architecture search through hypernetworks. _arXiv preprint arXiv:1708.05344_, 2017.


Rebekka Burkholz and Alina Dubatovka. Initialization of relus for dynamical isometry. In _Advances_
_in Neural Information Processing Systems_, pp. 2385–2395, 2019.


Han Cai, Ligeng Zhu, and Song Han. Proxylessnas: Direct neural architecture search on target task
and hardware. _arXiv preprint arXiv:1812.00332_, 2018.


Liang-Chieh Chen, Maxwell Collins, Yukun Zhu, George Papandreou, Barret Zoph, Florian Schroff,
Hartwig Adam, and Jon Shlens. Searching for efficient multi-scale architectures for dense image
prediction. In _Advances in Neural Information Processing Systems_, pp. 8699–8710, 2018.


Wuyang Chen, Xinyu Gong, Xianming Liu, Qian Zhang, Yuan Li, and Zhangyang Wang. Fasterseg:
Searching for faster real-time semantic segmentation. In _International Conference on Learning_
_Representations_, 2020a.


Xiangning Chen and Cho-Jui Hsieh. Stabilizing differentiable architecture search via perturbationbased regularization. _arXiv preprint arXiv:2002.05283_, 2020.


Xin Chen, Lingxi Xie, Jun Wu, and Qi Tian. Progressive differentiable architecture search: Bridging
the depth gap between search and evaluation. In _Proceedings of the IEEE International Conference_

_on Computer Vision_, pp. 1294–1303, 2019.


Xin Chen, Lingxi Xie, Jun Wu, Longhui Wei, Yuhui Xu, and Qi Tian. Fitting the search space of
weight-sharing nas with graph convolutional networks. _arXiv preprint arXiv:2004.08423_, 2020b.


Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming.
2019.


Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an
alternative to the cifar datasets, 2017.


Xiaoliang Dai, Peizhao Zhang, Bichen Wu, Hongxu Yin, Fei Sun, Yanghan Wang, Marat Dukhan,
Yunqing Hu, Yiming Wu, Yangqing Jia, et al. Chamnet: Towards efficient network design through
platform-aware model adaptation. In _Proceedings of the IEEE Conference on computer vision and_
_pattern recognition_, pp. 11398–11407, 2019.


Xuanyi Dong and Yi Yang. Searching for a robust neural architecture in four gpu hours. In
_Proceedings of the IEEE Conference on computer vision and pattern recognition_, pp. 1761–1770,
2019.


Xuanyi Dong and Yi Yang. Nas-bench-102: Extending the scope of reproducible neural architecture
search. _arXiv preprint arXiv:2001.00326_, 2020.


Xuanyi Dong, Lu Liu, Katarzyna Musial, and Bogdan Gabrys. Nats-bench: Benchmarking nas
algorithms for architecture topology and size. _arXiv preprint arXiv:2009.00437_, 2020.


Yonggan Fu, Wuyang Chen, Haotao Wang, Haoran Li, Yingyan Lin, and Zhangyang Wang. Autogandistiller: Searching to compress generative adversarial networks. In _International Conference on_
_Machine Learning_, pp. 3292–3303. PMLR, 2020.


Raja Giryes, Guillermo Sapiro, and Alex M Bronstein. Deep neural networks with random gaussian
weights: A universal classification strategy? _IEEE Transactions on Signal Processing_, 64(13):
3444–3457, 2016.


Xinyu Gong, Shiyu Chang, Yifan Jiang, and Zhangyang Wang. Autogan: Neural architecture search
for generative adversarial networks. In _Proceedings of the IEEE/CVF International Conference on_
_Computer Vision_, pp. 3224–3234, 2019.


9




--- end of page=8 ---

Published as a conference paper at ICLR 2021


Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei, and Jian Sun. Single
path one-shot neural architecture search with uniform sampling. _arXiv preprint arXiv:1904.00420_,
2019.


Dongyoon Han, Jiwhan Kim, and Junmo Kim. Deep pyramidal residual networks. In _Proceedings of_
_the IEEE conference on computer vision and pattern recognition_, pp. 5927–5935, 2017.


Boris Hanin and Mihai Nica. Finite depth and width corrections to the neural tangent kernel. _arXiv_
_preprint arXiv:1909.05989_, 2019.


Boris Hanin and David Rolnick. Complexity of linear regions in deep networks. _arXiv preprint_
_arXiv:1901.09021_, 2019a.


Boris Hanin and David Rolnick. Deep relu networks have surprisingly few activation patterns. In
_Advances in Neural Information Processing Systems_, pp. 361–370, 2019b.


Soufiane Hayou, Arnaud Doucet, and Judith Rousseau. On the impact of the activation function on
deep neural networks training. _arXiv preprint arXiv:1902.06853_, 2019.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing
human-level performance on imagenet classification. In _Proceedings of the IEEE international_
_conference on computer vision_, pp. 1026–1034, 2015.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_,
pp. 770–778, 2016.


Kurt Hornik, Maxwell Stinchcombe, Halbert White, et al. Multilayer feedforward networks are
universal approximators. _Neural networks_, 2(5):359–366, 1989.


Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and´
generalization in neural networks. In _Advances in neural information processing systems_, pp.
8571–8580, 2018.


Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In _Advances in neural information processing systems_, pp. 1097–1105,
2012.


Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha SohlDickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models
under gradient descent. In _Advances in neural information processing systems_, pp. 8572–8583,
2019.


Namhoon Lee, Thalaiyasingam Ajanthan, and Philip HS Torr. Snip: Single-shot network pruning
based on connection sensitivity. _arXiv preprint arXiv:1810.02340_, 2018.


Guohao Li, Guocheng Qian, Itzel C Delgadillo, Matthias Muller, Ali Thabet, and Bernard Ghanem.
Sgas: Sequential greedy architecture search. In _Proceedings of the IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition_, pp. 1620–1630, 2020a.


Liam Li and Ameet Talwalkar. Random search and reproducibility for neural architecture search. In
_Uncertainty in Artificial Intelligence_, pp. 367–377. PMLR, 2020.


Yanxi Li, Minjing Dong, Yunhe Wang, and Chang Xu. Neural architecture search in a proxy validation
loss landscape. In _International Conference on Machine Learning_, pp. 5853–5862. PMLR, 2020b.


Zhihang Li, Teng Xi, Jiankang Deng, Gang Zhang, Shengzhao Wen, and Ran He. Gp-nas: Gaussian
process based neural architecture search. In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, pp. 11933–11942, 2020c.


Hanwen Liang, Shifeng Zhang, Jiacheng Sun, Xingqiu He, Weiran Huang, Kechen Zhuang, and
Zhenguo Li. Darts+: Improved differentiable architecture search with early stopping. _arXiv_
_preprint arXiv:1909.06035_, 2019.


10




--- end of page=9 ---

Published as a conference paper at ICLR 2021


Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan
Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. In _Proceedings_

_of the European Conference on Computer Vision (ECCV)_, pp. 19–34, 2018a.


Chenxi Liu, Liang-Chieh Chen, Florian Schroff, Hartwig Adam, Wei Hua, Alan L Yuille, and Li FeiFei. Auto-deeplab: Hierarchical neural architecture search for semantic image segmentation. In
_Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 82–92, 2019.


Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. _arXiv_
_preprint arXiv:1806.09055_, 2018b.


Renqian Luo, Fei Tian, Tao Qin, Enhong Chen, and Tie-Yan Liu. Neural architecture optimization.
In _Advances in neural information processing systems_, pp. 7816–7827, 2018.


Renqian Luo, Xu Tan, Rui Wang, Tao Qin, Enhong Chen, and Tie-Yan Liu. Semi-supervised neural
architecture search. _arXiv preprint arXiv:2002.10389_, 2020.


Joseph Mellor, Jack Turner, Amos Storkey, and Elliot J Crowley. Neural architecture search without
training. _arXiv preprint arXiv:2006.04647_, 2020.


Guido Montufar. Notes on the number of linear regions of deep neural networks.´ _Sampling Theory_
_Appl., Tallinn, Estonia, Tech. Rep_, 2017.


Hieu Pham, Melody Y Guan, Barret Zoph, Quoc V Le, and Jeff Dean. Efficient neural architecture
search via parameter sharing. _arXiv preprint arXiv:1802.03268_, 2018.


Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the
expressive power of deep neural networks. In _international conference on machine learning_, pp.
2847–2854. PMLR, 2017.


Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. Regularized evolution for image
classifier architecture search. In _Proceedings of the aaai conference on artificial intelligence_,
volume 33, pp. 4780–4789, 2019.


Thiago Serra, Christian Tjandraatmadja, and Srikumar Ramalingam. Bounding and counting linear
regions of deep neural networks. In _International Conference on Machine Learning_, pp. 4558–4566.
PMLR, 2018.


Yeonjong Shin and George Em Karniadakis. Trainability of relu networks and data-dependent
initialization. _Journal of Machine Learning for Modeling and Computing_, 1(1), 2020.


Yao Shu, Wei Wang, and Shaofeng Cai. Understanding architectures learnt by cell-based neural
architecture search. In _International Conference on Learning Representations_, 2019.


Julien Siems, Lucas Zimmer, Arber Zela, Jovita Lukasik, Margret Keuper, and Frank Hutter. Nasbench-301 and the case for surrogate benchmarks for neural architecture search. _arXiv preprint_
_arXiv:2008.09777_, 2020.


Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image
recognition. _arXiv preprint arXiv:1409.1556_, 2014.


Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In
_Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 1–9, 2015.


Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and
Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. In _Proceedings of the_
_IEEE Conference on Computer Vision and Pattern Recognition_, pp. 2820–2828, 2019.


Mingxing Tan, Ruoming Pang, and Quoc V Le. Efficientdet: Scalable and efficient object detection.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp.
10781–10790, 2020.


Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by
preserving gradient flow. _arXiv preprint arXiv:2002.07376_, 2020.


11




--- end of page=10 ---

Published as a conference paper at ICLR 2021


Wei Wen, Hanxiao Liu, Hai Li, Yiran Chen, Gabriel Bender, and Pieter-Jan Kindermans. Neural
predictor for neural architecture search. _arXiv preprint arXiv:1912.00848_, 2019.


Lechao Xiao, Jeffrey Pennington, and Samuel S Schoenholz. Disentangling trainability and generalization in deep learning. _arXiv preprint arXiv:1912.13053_, 2019.


Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual´
transformations for deep neural networks. In _Proceedings of the IEEE conference on computer_
_vision and pattern recognition_, pp. 1492–1500, 2017.


Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. Snas: stochastic neural architecture search.
_arXiv preprint arXiv:1812.09926_, 2018.


Huan Xiong, Lei Huang, Mengyang Yu, Li Liu, Fan Zhu, and Ling Shao. On the number of linear
regions of convolutional neural networks. _arXiv preprint arXiv:2006.00978_, 2020.


Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian, and Hongkai Xiong.
Pc-darts: Partial channel connections for memory-efficient architecture search. In _International_
_Conference on Learning Representations_, 2019.


Ge Yang and Samuel Schoenholz. Mean field residual networks: On the edge of chaos. In _Advances_
_in neural information processing systems_, pp. 7103–7114, 2017.


Greg Yang. Scaling limits of wide neural networks with weight sharing: Gaussian process behavior,
gradient independence, and neural tangent kernel derivation. _arXiv preprint arXiv:1902.04760_,
2019.


Zhaohui Yang, Yunhe Wang, Xinghao Chen, Boxin Shi, Chao Xu, Chunjing Xu, Qi Tian, and Chang
Xu. Cars: Continuous evolution for efficient neural architecture search. In _Proceedings of the_
_IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, June 2020a.


Zhaohui Yang, Yunhe Wang, Dacheng Tao, Xinghao Chen, Jianyuan Guo, Chunjing Xu, Chao Xu,
and Chang Xu. Hournas: Extremely fast neural architecture search through an hourglass lens.
_arXiv preprint arXiv:2005.14446_, 2020b.


Jiahui Yu, Pengchong Jin, Hanxiao Liu, Gabriel Bender, Pieter-Jan Kindermans, Mingxing Tan,
Thomas Huang, Xiaodan Song, Ruoming Pang, and Quoc Le. Bignas: Scaling up neural architecture search with big single-stage models. _arXiv preprint arXiv:2003.11142_, 2020a.


Kaicheng Yu, Rene Ranftl, and Mathieu Salzmann. How to train your super-net: An analysis of
training heuristics in weight-sharing nas. _arXiv preprint arXiv:2003.04276_, 2020b.


Kaicheng Yu, Christian Sciuto, Martin Jaggi, Claudiu Musat, and Mathieu Salzmann. Evaluating the
search phase of neural architecture search. In _ICLR_, 2020c.


Hongpeng Zhou, Minghao Yang, Jun Wang, and Wei Pan. Bayesnas: A bayesian approach for neural
architecture search. _arXiv preprint arXiv:1905.04919_, 2019.


Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. _arXiv preprint_
_arXiv:1611.01578_, 2016.


Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V Le. Learning transferable architectures
for scalable image recognition. In _Proceedings of the IEEE conference on computer vision and_
_pattern recognition_, pp. 8697–8710, 2018.


12




--- end of page=11 ---

Published as a conference paper at ICLR 2021


## A IMPLEMENTATION DETAILS


For _κN_ we sample one mini-batch of size 32 from the training set, and calculate Θ( [ˆ] _**x**_ _,_ _**x**_ _[′]_ ) =
_J_ ( _**x**_ ) _J_ ( _**x**_ _[′]_ ) _[T]_ . For _R_ [ˆ] _N_ we sample 5000 images, forward them through the network, and collect the
activation patterns from all ReLU layers. The calculation of both _κN_ and _R_ [ˆ] _N_ are repeated three
times in all experiments, where each time the network weights are randomly drawn from Kaiming
Norm Initialization (He et al., 2015) without involving any training (network weights are fixed).


Our retraining settings (after search) follow previous works (Xu et al., 2019; Chen et al., 2019; Chen
& Hsieh, 2020). On CIFAR-10, we train the searched network with cutout regularization of length
16, drop-path (Zoph et al., 2018) with probability as 0.3, and an auxiliary tower of weight 0.4. On
ImageNet, we also use label smoothing during training. On both CIFAR-10 and ImageNet, the
network is optimized by an SGD optimizer with cosine annealing, with learning rate initialized as
0.025 and 0.5, respectively.


## B SEARCHED ARCHITECTURE


We visualize the searched normal and reduction cells in figure 6 and 7, which is directly searched on
CIFAR-10 and ImageNet respectively.

dil_conv_5x5













sep_conv_3x3























(a) Normal Cell



(b) Reduction Cell



### Figure 6

Caption: ** Normal and Reduction cells discovered by TE-NAS on CIFAR-10.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.







































(a) Normal Cell





(b) Reduction Cell



### Figure 7

Caption: ** Normal and Reduction cells discovered by TE-NAS on imageNet.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


C DEPTH AND WIDTH PREFERENCE OF _κN_ AND _R_ [ˆ] _N_ IN DARTS SPACE


To analyze the impact of different architectures on trainability and expressivity in DARTs search
space, we visualize _κN_ and _R_ [ˆ] _N_ with different depths and width. Following Shu et al. (2019), the
depth of a cell is defined as the number of connections on the longest path from input nodes to the

- utput node, and the width of a cell is the summation of the edges of the intermediate nodes that
are connected to the input nodes. We randomly sample 20,000 architectures in DARTs space, and
plot the visualizations in Figure 8. Good architectures should exhibit low _κN_ (good trainability, blue
dots in Figure 8(a)) and high _R_ ˆ _N_ tell us that in DARTs space shallow but wide cells are preferred to favor both trainability and _R_ [ˆ] _N_ (powerful expressivity, red dots in Figure 8(b)). Therefore, _κN_ and
expressivity. This conclusion matches the findings by Shu et al. (2019): existing NAS algorithms
tend to favor architectures with wide and shallow cell structures, which enjoy fast convergence with
smooth loss landscape and accurate gradient information.


13




--- end of page=12 ---

Published as a conference paper at ICLR 2021


8



8


7


6


5


4


3



7


6


5


4


3



700


680


660


640


620


600


580


560



3246 _._ 5


3246 _._ 0


3245 _._ 5


3245 _._ 0


3244 _._ 5



1 _._ 0 1 _._ 5 2 _._ 0 2 _._ 5 3 _._ 0 3 _._ 5 4 _._ 0
Depth


(a) _κN_



1 _._ 0 1 _._ 5 2 _._ 0 2 _._ 5 3 _._ 0 3 _._ 5 4 _._ 0
Depth


(b) _R_ [ˆ] _N_



### Figure 8

Caption: ** Depth and width preference of (a) _κN_ and (b) _R_ [ˆ] _N_      - n DARTs Search Space.

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.


## D MORE ABLATION STUDIES


D.1 SEARCH WITH ONLY _κN_ OR _R_ [ˆ] _N_


As we observed in Table 4, searching with only _κN_ - r _R_ [ˆ] _N_ leads to inferior performance, which
indicates the importance of maintaining both trainability and expressivity during the search.


**Table 4:** Search with only _κN_        - r _R_ [ˆ] _N_        - n CIFAR-100 in NAS-Bench-201 space.


Methods CIFAR-100 Test Accuracy


Prune with only _κN_ 69.25 (1.29)
Prune with only _R_ [ˆ] _N_ 70.48 (0.29)
TE-NAS **71.24** (0.56)


D.2 DIFFERENT COMBINATION OPTIONS FOR _κN_ AND _R_ [ˆ] _N_


Pruning by _s_ ( _oj_ ) = _sκ_ ( _oj_ ) + _sR_ ( _oj_ ) is not the only option (see Algorithm 1). Here in this study we
consider more:


1) pruning by _s_ ( _oj_ ) = min( _sκ_ ( _oj_ ) _, sR_ ( _oj_ )), i.e., pruning by the worst case.

2) Pruninng by _s_ ( _oj_ ) = max( _sκ_ ( _oj_ ) _, sR_ ( _oj_ )), i.e., pruning by the best case.

3) pruning by summation of changes ∆ _κt,oj_ + ∆ _Rt,oj_, i.e., directly use the numerical values

     - f the changes.


As we observed in Table 5, our TE-NAS stands out of all options. This means a good trade-off
between _κN_ and _R_ [ˆ] _N_ are important, and also the ranking strategy is better than directly using
numerical values.


**Table 5:** Search with only _κN_        - r _R_ [ˆ] _N_        - n CIFAR-100 in NAS-Bench-201 space.


Methods CIFAR-100 Test Accuracy


Prune by _s_ ( _oj_ ) = min( _sκ_ ( _oj_ ) _, sR_ ( _oj_ )) 70.75 (0.73)
Prune by _s_ ( _oj_ ) = max( _sκ_ ( _oj_ ) _, sR_ ( _oj_ )) 70.33 (1.09)
Prune by ∆ _κt,oj_ + ∆ _Rt,oj_ 70.47 (0.68)
Prune by _s_ ( _oj_ ) = _sκ_ ( _oj_ ) + _sR_ ( _oj_ ) (TE-NAS) **71.24** (0.56)


14




--- end of page=13 ---

Published as a conference paper at ICLR 2021


D.3 CORRELATION BETWEEN TEST ACCURACY AND COMBINATION OF _κN_ AND _R_ [ˆ] _N_



### Figure 9

Caption: indicates that by using the summation of the ranking of both _R_ ˆ _N_, the combined metric achieves a _κN_ and

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
much higher correlation with the test accuracy. The reason can be explained by
### Figure 4

Caption: , as _κN_ and _R_ [ˆ] _N_ prefers different

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.

- perators in terms of trainability and Expressivity. Their combination can filter

- ut bad architectures in both aspects and
strongly correlate with networks’ final
performance.


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
|||-0.64|||||
|Ken|dall-tau =|-0.64|-0.64|-0.64|-0.64|-0.64|



0 500 1000 1500 2000 2500 3000
rank( _R_ [ˆ] _N_ ) + rank( _κN_ )

### Figure 9

Caption: ** Summation of ranking of _κN_ and _R_ [ˆ] _N_ exhibits

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
stronger (negative) correlation with the test accuracy of architectures in NAS-Bench201 (Dong & Yang, 2020).



70


60


50


40


30


20


10


0





E GENERALIZATION V.S. TEST ACCURACY



Conceptually, the generalization gap is the
difference between a model’s performance

- n training data and its performance on
unseen data drawn from the same distribution (e.g., testing set). In comparison, the
two indicators _κN_ (trainability) and _RN_
(expressiveness) of a network determine
how well the training set could be fit (i.e.,
training set accuracy), and do not directly
indicate its generalization gap (or test set
accuracy). Indeed, probing generalization

- f an untrained network at its initialization
is a daunting, open challenge that seems
to go beyond the current theory scope.



70





60


50


40


30


20


10

|K|endall-tau =|0.79|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||



20 40 60 80 100
Train Accuracy (CIFAR-100)

### Figure 10

Caption: ** The correlation between test accuracy and the

Meaning: The figure supports the method or empirical discussion described by this caption; consult the local source PDF for the visual encoding.
training accuracy in NAS-Bench201 (Dong & Yang, 2020).



In NAS, we are searching for the architecture with the best test accuracy. As shown in Figure 10,
in NAS-Bench201 the training accuracy strongly correlates with test accuracy. This also seems to
be a result of the current standard search space design that could have implicitly excluded severe

- verfitting. This explains why _κN_ and _RN_, which only focuses on trainability and expressiveness
during training, can still achieve good search results of test accuracy.


15




--- end of page=14 ---
