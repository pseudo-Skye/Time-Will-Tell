# MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives

*   **Summary:** The article highlights **time series discords** as a popular method for practitioners due to its simplicity. However, the effectiveness of discords is limited by the sensitivity to the **subsequence length parameter** set by the user. The article introduces MERLIN, an algorithm that can **efficiently and exactly find discords of all lengths** in massive time series archives. The authors demonstrate the utility of MERLIN in **detecting subtle anomalies** that cannot be detected by existing algorithms or human inspection, and also highlight how computational redundancies can make MERLIN significantly **faster** than comparable algorithms. ([paper](https://ieeexplore.ieee.org/document/9338376), [code](https://github.com/salesforce/Merlion))

### Content

1.  [Introduction](#1-introduction)
2.  [Preliminary](#2-preliminary)
3.  [Methodology](#3-methodology)
4.  [Experiments](#4-experiments)

## 1\. Introduction

### **Time series discords**

Time series discords are subsequences of a time series that are maximally far away from their nearest neighbors. In other words, they are the parts of a time series that are the most different from the rest of the data. 

### Limitations of existing works

#### Machine learning

(1) **The need for manual tuning of parameters:** Many existing methods require the user to manually tune one or more parameters, such as the window size or threshold value, in order to achieve good performance. This can be time-consuming and subjective, and may result in suboptimal performance.

(2) **Overfitting:** With so many parameters to fit on a small dataset, the risk of overfitting increases, which can lead to suboptimal performance and decreased reliability in anomaly detection. This means that the model may be **detecting patterns or features that are specific to the training data but not present in new data**. As a result, the model may mistakenly identify these patterns or features as anomalies, even if they are not truly anomalous (**increase false positives**).

#### Time series discords

The following example shows that there is a “sweet spot” (or rather, sweet range) for subsequence length when performing anomaly discovery. In some cases, the analyst may have prior knowledge or experience that can guide them in choosing a good value for the subsequence length parameter. However, in anomaly and novelty discovery tasks, it is often **necessary to try different subsequence lengths to see which works best for the particular dataset and anomaly detection task**. 

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/230253509-74faa17e-86e8-46e6-8dd2-3bef0ddd3a21.png">
</p>

#### SOTA discords discovery algorithm - DRAG

[DRAG](https://ieeexplore.ieee.org/document/4470262) requires a single input parameter $r$. This value should ideally be set such that it is **a little less than** the discord distance, that is, the distance between the discord and its nearest neighbor. If $r$ is much too small, the algorithm will give the correct result, but have a time and space complexity of $O(n^2)$. In contrast, if $r$ is set too large, the algorithm will return null, a situation we denote as a failure. Of course, the situation can be remedied, but requires the user to reduce the r value and try again. Hence, choosing a good value of $r$ is critical for the DRAG to be efficient, but it is very difficult parameter to set. 

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/231066164-12acab68-e79a-4362-8b38-b06c548ec1fd.png">
</p>

For example, the time taken for DRAG given values for r that range from 1.0 to 40.0. **For any value greater than 10.27 (discord value) the algorithm reports failure and must be restarted with a lower value.** Suppose that we had guessed r = 10.25, then DRAG would have taken 1.48 seconds to find the discord. However, had we guessed a value that was just 2.5% less, DRAG would have taken 9.7 times longer. Has we guessed r = 1.0 (a perfectly reasonable value on visually similar data), DRAG would have taken 98.9 times longer. In the other direction, had we guessed any greater than 1% more, DRAG would have failed. The time it takes to complete a failed run is about 1/6 the time of our successful run when r =  
was set to the 10.25 guess. So, while failure is cheaper, it is not free.

### Why we need MERLIN

(1) MERLINE can **remove the need to set even that sole parameter (sequence length)**. It can efficiently and exactly discover discords of every possible length, then either report all of them, or just the top-K discords under an arbitrary user-defined scoring metric. 

(2) MERLINE relieves the issue of estimation of $r$ in the DRAG algorithm. 

## 2\. Preliminary

### Definition of time series discord

Given a time series $T$, the subsequence $D$ of length $L$ is said to be the discord of $T$ if $D$ has the largest distance to its nearest non-self match:

$\\forall C \\in T$, non-self match $M\_D$ of $D$, and non-self match $M\_C$ of $C$, $\\min (\text{Dist}(D, M\_D))>\\min (\text{Dist}(C, M\_C))$

<p align="center" width="100%">
<img width="40%" src = "https://user-images.githubusercontent.com/117964124/230281410-9a223807-ad97-4136-a6ab-d83dd7f74d7d.png">
</p>

### DRAG algorithm

#### Phase 1

Start with empty set $C$ = {} (candidate set of discords)  
_**is\_candidate**_ = T  
For each subsequence $s$ in time series  
       For each candidate $c$ in $C$  
              if $c$ is non-trivial (far away from s) and $\text{dist}(s,c) \< r$  
                     prune $c$ from $C$  
                     _**is\_candidate**_ = F  
       if _**is\_candidate**_  
       # if no one is pruned (all $\text{dist}(s,c) \\geq r$)       
              add $s$ to $C$

$r$ is expected to be smaller than $\text{Dist}(D, M\_D)$ (discord distance). Thus, if the subsequence currently under consideration is greater than $r$ from any item in the set, then it may be the discord, so it is added to the set. However, if any items in the set $C$ are less than $r$ from the subsequence under consideration, we know that they could not be discords. At the end of Phase 1, the set $C$ is guaranteed to contain the true discord, possibly with some additional false positives.

\*Note that at this phase, the distance is only calculated **from the current subsequence to every subsequence** **before** **it**. So, the distance from the current subsequence to the subsequence after it is NOT calculated. 

#### Phase 2

for each $c$ in $C$ do  
       initialize $c\_d=\\infty$   
end for

Start with empty set $D$ = {} (set of discords)  
_**is\_discord**_ = T  
For each subsequence $s$ in time series  
       For each candidate $c$ in $C$  
              if $c$ is non-trivial (far away from $s$)  
                     get $d = dist\\left(s, c\\right)$      
                     if $d \< r$  
                            prune $c$ from $C$  
                            _**is\_discord**_ = F  
                     else  
                            $c\_d = min(c\_d, d)$  

We simply consider each subsequence’s distance to every member of our set, doing a best-so-far search for each candidate’s nearest neighbor. There are there situations for the subsequences in time series:

(1) The distance $d$ between the discord candidate $c$ and the subsequence $s$ is **greater than** the current value of $c\_d$. If this is true we do nothing.

(2) The distance $d$ between the discord candidate $c$ and the subsequence $s$ is **less than** $r$. If this happens it means that the discord candidate can not be a discord, it is a false positive. We can permanently remove it from the set $C$.

(3) The distance $d$ between the discord candidate $c$ and the subsequence $s$ is **less than** the current value of $c\_d$ (**greater than** $r$). If this is true we simply update the current distance to the nearest neighbor.

For more details about DRAG, it can be found [here](https://ieeexplore.ieee.org/document/4470262).

### Z-normalized Euclidean Distance

The z-normalized Euclidean distance $D\_{ze}$ is defined as the Euclidean distance $D\_e$ between the z-normalized or normal form of two sequences, where the z-normalized form $\\hat X$ is obtained by transforming a sequence $X$ of length $L$ so it has mean $\\mu = 0$ and standard deviation $\\sigma = 1$.

$$  
\\begin{gathered}  
\\hat{X}=\\frac{X-\\mu\_X}{\\sigma\_X} \\\\  
D\_{z e}(X, Y)=D\_e(\\hat{X}, \\hat{Y})=\\sqrt{\\left(\\hat{x}\_1-\\hat{y}\_1\\right)^2+\\ldots+\\left(\\hat{x}\_L-\\hat{y}\_L\\right)^2}  
\\end{gathered}  
$$

Given the variance of the sequence $X$ $$\\sigma\_X^2  =\\frac{\\sum\\limits\_{i}^{L} (x\_i-\\mu\_X)^2}{L} $$

we can derive the length of the sequence $L$ as $$L =\\sum\_i^L\\left(\\frac{x\_i-\\mu\_X}{\\sigma\_X}\\right)^2$$

Then, we can derive the z-normalized Euclidean distance between $X$ and $Y$ as 

$$  
\\begin{aligned}  
D\_{z e}(X, Y)^2 & =\\sum\_i^L\\left(\\frac{x\_i-\\mu\_X}{\\sigma\_X}-\\frac{y\_i-\\mu\_Y}{\\sigma\_Y}\\right)^2 \\\\  
& =\\sum\_i^L\\left(\\frac{x-\\mu\_X}{\\sigma\_X}\\right)^2+\\sum\_i^L\\left(\\frac{y-\\mu\_Y}{\\sigma\_Y}\\right)^2-2 \\sum\_i^L\\left(\\frac{x-\\mu\_X}{\\sigma\_X}\\right)\\left(\\frac{y-\\mu\_Y}{\\sigma\_Y}\\right) \\\\  
& =2 L\\left(1-\\frac{1}{L} \\sum\_i^L\\left(\\frac{x-\\mu\_X}{\\sigma\_X}\\right)\\left(\\frac{y-\\mu\_Y}{\\sigma\_Y}\\right)\\right) \\\\  
& =2 L(1-\text{corr}(X, Y))  
\\end{aligned}  
$$

As the Pearson correlation is limited to range $\[-1,1\]$, the distance boundary will fall in the range $\[0, 2\\sqrt{L}\]$.

## 3\. Methodology

### Purpose of MERLIN

Find the optimal range of $r$ that reduce the failure trials to accelerate the DRAG algorithm across all discord length. 

### Exploration of discord value w.r.t. discord length

<p align="center" width="100%">
<img width="60%" src = "https://user-images.githubusercontent.com/117964124/231355612-324545c9-c749-442d-bb1f-284d1233d91d.png">
</p>

As we can observe, when we make the discord length longer, the discord score can **increase, decrease or stay the same**. Thus, it is a bad idea that we use the discord value $d\_i$ at length $i$ as $r$ to discover discord at length $i+1$. The most ideal situation is that the choice of $r$ is slightly below the optimal $d\_i$. As the **green line** is slightly below the blue line, the MERLIN is proposed to set $r = \\mu -2\\sigma$ by looking at the **variance** of the last few (say five) discord values. If DRAG reports failure, we repeatedly **subtract another** $\\sigma$ from the current value of r until it reports success.

For the first five discord lengths, MERLIN sets the **first** $r$ by the upper bound $2\\sqrt{L}$ of discord distances which range $\[0, 2\\sqrt{L}\]$ as we introduced above, and **keeps halving** it until we get a success. However, $2\\sqrt{L}$ is a very weak bound, and likely to produce many failures. So, **for the next four items**, we can use the previous discord distance, **minus an epsilon, say 1%**. In the very unlikely event that this was too conservative and resulted in a failure, we can **keep subtracting an additional 1%** until we get a success.

### The MERLIN Algorithm

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/231361331-2f1f51a2-cdda-4cae-8d0b-94d1dd47aa49.png">
</p>

**Reason for discovering discords from short sequence length to long sequence length:** It is only for the first invocation of DRAG that we are completely uncertain about a good value for r, and we may have multiple failure runs and/or invoke DRAG with too small of a value for r, making it run slow. It is much faster to do this on the shorter subsequence lengths.

### Failure cases of MERLIN

(1) **A subsequence of the constant region:** the z-normalize will fail as the $\\sigma = 0$. However, such a situation usually indicates i.e., a device disconnection or heart failure that warrants an alarm. 

(2) **'twin freak' problem:** more than one anomaly, and they look the same. This can be solved by changing the first nearest neighbor to the $k^{th}$ nearest neighbor. However, this issue is quite rare in practice. 

## 4\. Experiments

### Unsuitability of Benchmarks

(1) **Anomalies without label:**  Any algorithm that does find this anomaly will be penalized as having produced a **false positive**. 

(2) **Obvious and trivial anomalies:** obvious anomalies that any algorithm can easily detect. 

### Advantages of MERLIN

(1) Can discover **ultra-subtle** anomalies

<p align="center" width="100%">
<img width="60%" src = "https://user-images.githubusercontent.com/117964124/231377844-cf51c5e3-7d6e-4b0c-b638-225bae357b77.png">
</p>

<p align="center" width="100%">
<img width="60%" src = "https://user-images.githubusercontent.com/117964124/231378399-ac336260-3408-4b59-9009-6928542c52b0.png">
</p>

(2) Free of **assumptions about the anomaly duration**

<p align="center" width="100%">
<img width="60%" src = "https://user-images.githubusercontent.com/117964124/231381364-3804b032-9599-476f-8d9a-f73409c957d6.png">
</p>

(3) MERLIN achieves better running **efficiency**

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/231382206-2519942b-d304-4534-88ab-32834d38271a.png">
</p>

     They will use the worst-case dataset from MERLIN, **random walk**. For such data, the top-1 discord is only **slightly** further away from its nearest neighbor than any randomly chosen subsequence, meaning that the candidate set in DRAG phase 1 grows **relatively large** even if given a good value for $r$.

### The threshold for discord distances

Thresholds can often be learned with simple human-in-the-loop algorithms. In brief, the user can simply examine a sorted list of all candidate anomalies. The discord distance of the first one she rejects as “not an anomaly”, can be used as the threshold for future datasets from the same domain. 

### Evaluation metrics

Each algorithm is tasked with locating the **one** location it thinks is most likely to be anomalous (We removed the handful of examples that have no claimed anomaly). If that location is within $\\pm 1\\%$ of $T$ from a ground truth anomaly, we count that prediction as a success. 

### MERLIN++ (2023)

By utilizing the indexing technique called **Orchard's algorithm**, MERLIN++ is faster than MERLIN but produces identical results. For more details, it can be found [here](https://link.springer.com/article/10.1007/s10618-022-00876-7).
