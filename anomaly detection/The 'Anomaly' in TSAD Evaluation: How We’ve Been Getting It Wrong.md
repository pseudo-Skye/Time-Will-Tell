# The 'Anomaly' in TSAD Evaluation: How We’ve Been Getting It Wrong

_**Author:** pseudo\_Skye   **Updated:** 2023-04-05_

Time series anomaly detection is an essential task in various domains such as finance, healthcare, and security. However, the existing evaluation metrics for time series anomaly detection algorithms have limitations that can lead to misleading results. In this article, we will discuss the limitations of existing evaluation metrics and introduce the new evaluation metrics that address these limitations from the most recent publications. 

### Content

1.  [Towards a Rigorous Evaluation of Time-series Anomaly Detection (AAAI 2022)](#towards-a-rigorous-evaluation-of-time-series-anomaly-detection-aaai-2022)
2.  [Local Evaluation of Time Series Anomaly Detection Algorithms (KDD 2022)](#local-evaluation-of-time-series-anomaly-detection-algorithms-kdd-2022)

## Towards a Rigorous Evaluation of Time-series Anomaly Detection (AAAI 2022)

**Summary:** The paper highlights that most studies apply a peculiar evaluation protocol called point adjustment (PA) before scoring. The authors reveal that the PA protocol has a great possibility of inflating F1 scores and thus can be misleading. They propose a new evaluation protocol called PA%K that can provide more reliable evaluation results. ([paper](https://arxiv.org/abs/2109.05257))

### **PA definition**

If **at least one moment** in a contiguous anomaly segment is detected as an anomaly, the **entire segment** is then considered to be correctly predicted as an anomaly.

### **Problems of PA**

(1) PA has a high possibility of overestimating the model performance.   

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/227907712-1374b666-77e4-4b37-bfbd-989a284f8e98.png">
</p>

(2) No baseline and relative comparison against the proposed model. 

### **Types of anomalies**

have been discussed [here](https://github.com/pseudo-Skye/Time-Matters/blob/main/TFAD%20(CIKM%2022).md#different-anomaly-types). 

### **Methodology of TAD**

Unsupervised model - trained by only normal signals, then assign anomaly scores to inputs. 

(1) **Reconstruction-based AD method:** Train a model to minimize the distance between a normal input and its reconstruction, then anomalies are those hard to reconstruct by the model and yield a large distance. 

(2) **Forecast-based AD method:** Train a model to predict the signal that comes after the normal input, and take the distance between the ground truth and the predicted signal as an anomaly score.

### How to label test set with PA

PA adjusts $\\hat{y}\_t$ to 1 for all $t \\in S\_m$ if anomaly score is higher than $\\delta$ at least once in $S\_m$:

$$\\hat{y}\_t= \\begin{cases}1, & \\text { if } \\mathcal{A}\\left(\\boldsymbol{w}\_t\\right)>\\delta \\\\ & \\text { or } t \\in S\_m \\text { and } \\exists\_{t^\\prime \\in S\_m} A(\\omega\_{t^\\prime} > \\delta) \\\\ 0, & \\text { otherwise. }\\end{cases}$$

Where $\\hat{y}\_t$ is the predicted anomaly label, and $\\boldsymbol{w}\_t=\\{x\_t, x\_{t+1}, …, x\_{t+\\tau -1}\\}$ is the segment window with window size $\\tau$. The F1 score derived by PA is denoted as $F1\_{PA}$, otherwise the original F1 is measured without PA. 

### Pitfalls of the TAD with F1

F1 can unexpectedly **underestimate** the detection capability. In fact, due to the incomplete test set labeling, some signals labeled as anomalies share more statistics with normal signals. 

### Pitfalls of the TAD with $`F1_{PA}`$

#### 1\. PA increases P, R and F1

   Given the definition of precision (P), recall (R) and F1, after PA, we have $P (\\uparrow)= \\frac{TP (\\uparrow)}{TP (\\uparrow) +FP (-)}$, $R (\\uparrow) = \\frac{TP (\\uparrow)}{TP (\\uparrow)+FN (\\downarrow)}$, and thus increases $F1 = \\frac{2PR}{P+R}$.

#### 2\. Random anomaly score can achieve high $`F1_{PA}`$

    $F1\_{PA}$ for the case of uniform random anomaly scores varying with the threshold $\\delta$ for different $t\_e-t\_s$ (anomaly segment length). When the anomaly segment is long enough, $F1\_{PA}$ approaches 1 as $\\delta$ increases. 

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/228407084-84ab52c8-084e-4bcb-a66f-1bda23fcfdc6.png">
</p>


#### 3\. Untrained model with comparably high F1

A deep neural network is generally initialized with random weights drawn from a Gaussian distribution $\\mathcal{N}\\left(0, \\sigma^2\\right)$, where $\\sigma$ is much smaller than 1. Without training, the outputs of the model are close to zero because they also follow a zero-mean Gaussian distribution. So, when we measure the anomaly scores by the Euclidean distance between the input and output, no matter if it is a reconstruction-based or forecasting-based method, we have an output approximately to zero. Thus, we have 

$$  
\\mathcal{A}\\left(\\boldsymbol{w}\_t\\right)=\\left\\|\\boldsymbol{w}\_t-\\eta\\right\\|\_2 \\simeq\\left\\|\\boldsymbol{w}\_t\\right\\|\_2  
$$

where $\\eta=f\_\\theta\\left(\\boldsymbol{w}\_t\\right)$ and $\\theta \\sim \\mathcal{N}\\left(0, \\sigma^2\\right)$. In this case, t**he anomaly score is just given by the value of the data point itself**. 

### Proposed solutions to the evaluation problems

#### New baseline proposed

F1 is measured from the prediction of a randomly initialized reconstruction model with simple architecture, such as an untrained autoencoder comprising a single-layer LSTM.

Alternatively, the anomaly score can be defined as the input itself. 

#### New evaluation protocol PA%K

The idea of PA%K is to apply PA to $S\_m$ only if the ratio of the number of correctly detected anomalies in $S\_m$ to its length exceeds the PA%K threshold K.

$$\\hat{y}\_t= \\begin{cases}1, & \\text { if } \\mathcal{A}\\left(\\boldsymbol{w}\_t\\right)>\\delta \\text { or } \\\\ & t \\in S\_m \\text { and } \\frac{\\left|\\left\\{t^{\\prime} \\mid t^{\\prime} \\in S\_m, \\mathcal{A}\\left(\\boldsymbol{w}\_{t^{\\prime}}\\right)>\\delta\\right\\}\\right|}{\\left|S\_m\\right|}>\\mathrm{K} \\\\ 0, & \\text { otherwise }\\end{cases}$$

K can be selected manually between 0 and 100 based on prior knowledge. For example, if the test set labels are reliable, a larger K is allowable. If a user wants to remove the dependency on K, it is recommended to **measure the area under the curve of** $F1\_{PA\\%K}$ **obtained by increasing K from 0 to 100**.

### Comparison of different metrics

#### Correlation between $`F1_{PA}`$ and F1

**No evidence** is given to assure the existence of the correlation between $F1\_{PA}$ and F1. Therefore, comparing the SOTA methods using only $F1\_{PA}$ may have a risk of improper evaluation of the detection performance. 

#### Test cases

1\. Random anomaly score 

2\. Input itself as an anomaly score

3\. Anomaly score from the randomized model

#### Experiment setting

(1) Case 2&3: window size $\\tau = 120$.

(2) Case 1&3: randomness with 5 different seeds and take the mean value. 

(3) No preprocessing such as early time steps removal or downsampling.

(4) Thresholds were obtained that yielded the best score.

#### comparison result

The **up arrow** is displayed with the result for the following cases:  
(1) $F1\_{PA}$ is higher than Case 1, (2) F1 is higher than Case 2 or 3, whichever is greater.

<p align="center" width="100%">
<img width="80%" src = "https://user-images.githubusercontent.com/117964124/228460070-ff4757c0-8134-4e8e-93c4-8531983b54d0.png">
</p>

*   Case 1: $F1\_{PA}$ appears to yield the SOTA methods; thus distinguishing whether the SOTA method successfully detects anomalies or whether it merely outputs a random anomaly score irrelevant to the input is impossible. (extreme low F1 but high $F1\_{PA}$ - illusion of the robust TAD methods)
*   Case 2&3: F1 score depends on the length of the input window. The longer the input window, the larger the F1. From the F1 score derived under setting Case 2&3, the proposed SOTA methods may have obtained marginal or even no advancement against the baselines.
*   The $F1\_{PA\\%K}$ of a well-trained model is expected to show constant results regardless of the value of K.

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/228469280-7bc977d5-f902-4226-8ef2-a693fa91c2f5.png">
</p>


#### \* Problems left for the future: **How to address the dependency of the threshold?**

   Existing TAD methods set the threshold after investigating the test dataset or simply use the optimal threshold that yields the best F1. Thus, t**he detection result depends significantly on threshold selection**. Additional metrics with reduced dependency such as receiver operating characteristic (AUROC) or area under precision-recall (AUPR) curve will help in rigorous evaluation. 

## Local Evaluation of Time Series Anomaly Detection Algorithms (KDD 2022)

**summary:** This paper talks about the limitations of classical precision and recall in TAD evaluation. It proposes a new metric named 'affiliation metrics' that is theoretically principled, parameter-free, robust against adversary predictions, retains a physical meaning (as they are connected to quantities expressed in time units), and is locally interpretable (allowing troubleshooting detection at individual event level). ([paper](https://arxiv.org/abs/2206.13167), [code](https://github.com/ahstat/affiliation-metrics-py))

### Limitations of the classical metrics for TAD

1\. Unawareness of temporal adjacency (A)

    Penalize without tolerance for the wrong predictions located closely to the ground truth event. 

2\. Unawareness of the event durations (B)

    Point-wise evaluation reward more if the anomalous segment is longer. 

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/228541600-de384eda-c110-42fc-ba4b-2cce751ce8a0.png">
</p>

### Contribution of the affiliation metrics

1\. handle the limitations (A) and (B) introduced by the presence of time.

2\. Parameter-free definition.

3\. Expressiveness of the scores, which means a slight improvement of the predictions should result in a slight improvement of the scores.

4\. Loacal interpretability of the scores.

5\. Existence of statistical bounds of the scores. 

### Methodology of affiliation metrics

#### STEP 1: Calculate the average directed distance between sets to measure how far the events are one from each other

*   The definition of the average directed distance: $\\mathrm{dist}(X,Y) = \\frac{1}{|X|} \\int\_{x \\in X} \\mathrm{dist}(x,Y) dx$

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/228718707-35210de1-4131-4d86-af0d-c945e88dd94a.png">
</p>

*   **Precision:** From prediction ($Y^\\prime$) to ground truth ($Y$) (left)

       Here, $|Y^\\prime| = |TP+FPs| = 5 (\\mathrm{min})$. The distance of the TPs to the ground truth is 0 as the vertical distance indicates. The distance from prediction to ground truth is given by the distance from FPs to the ground truth $\\int\_{1}^{2} y^\\prime dy^\\prime = 1.5 (\\mathrm{min})$. Thus, we have $$\\mathrm{dist}(Y^\\prime, Y) = \\frac{1}{|Y^\\prime|}\\int\_{1}^{2} y^\\prime dy^\\prime = \\frac{1.5}{5} = 0.3 (\\mathrm{min}) = 18 \\mathrm{s} $$

       In this case, **removing the FPs can largely increase the precision**. 

*   **Recall:** From ground truth to prediction (right)

       Here, we calculate the distance within the range of ground truth given the definition of recall that how many TPs are corrected classified. $|Y| = 10 (\\mathrm{min})$, and the distance from ground truth to TPs are given by $\\int\_{0}^{5} ydy+ 2\\int\_{0}^{0.5} ydy = 12.75 (\\mathrm{min})$. Thus, we have $$\\mathrm{dist}(Y, Y^\\prime) = \\frac{1}{|Y|}\\int\_{0}^{5} ydy+ 2\\int\_{0}^{0.5} ydy = \\frac{12.75}{10} = 1.275 (\\mathrm{min}) = 76.5 \\mathrm{s}$$

       In this case, **removing the FPs wouldn't change the results of recall**. 

#### STEP 2: Affiliate each prediction to the closest ground truth event

*   **Zone of affiliation:** Assign each time stamp along the time axis to the nearest ground truth event ($gt\_j$), then the time zone will be partitioned into several intervals. 
*   **Individual precision:** $D\_{p\_j} = \\mathrm{dist} \\left( Y^\\prime \\cap I\_j, gt\_j \\right)$
*   **Individual recall:** $D\_{r\_j} = \\mathrm{dist} \\left( gt\_j, Y^\\prime \\cap I\_j \\right)$

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/228727491-73b5bd82-1a8d-46a8-afc7-2be76f6881b7.png">
</p>

*   **Example:** the time zone has been partitioned into 3 zones of affiliation named $I\_1$ (discussed in STEP 1), $I\_2$ and $I\_3$. Thus, we have $D\_{p\_1} = 18 \\mathrm{s}$, and $D\_{r\_1} = 76.5 \\mathrm{s}$. Similarly, we can have the individual precision and recall of the other zones as $D\_{p\_2} = 11 \\mathrm{min} 30 \\mathrm{s}$, $D\_{r\_2} = 2 \\mathrm{min} 30 \\mathrm{s}$, $D\_{p\_3} = 31 \\mathrm{min} 15\\mathrm{s}$, and $D\_{r\_3} = 2 \\mathrm{min} 30\\mathrm{s}$. Individual distances are providing a summarized view of each ground truth event expressed in a meaningful unit.

It would be desirable to **normalize** each individual value to the \[0, 1\] range, by assuming that **each ground truth event is equally important**. 

#### STEP 3: Convert the observed temporal distances into probabilities

*   **Individual precision probability:** For the precision, the distance from $X$ to the ground truth $gt\_j$ is a random variable with a cumulative distribution function $F\_{p\_j}$. The survival function is given by $\\overline F\_{p\_j}(d) = 1- F\_{p\_j}(d-)$. Thus, the individual precision probability is given by 

$$P\_{\\text {precision }\_j}=\\frac{1}{\\left|\\mathrm{pred} \\cap I\_j\\right|} \\int\_{x \\in \\mathrm{pred} \\cap I\_j} \\bar{F}\_{\\text {precision }\_j}\\left(\\mathrm{dist}\\left(x, \\mathrm{gt}\_j\\right)\\right) d x $$

       **Explanation:**

       The cumulative distribution function means that if we randomly sample a point **within the affiliation zone**, what is the possibility that the distance $d$ **from the point to the** $gt\_j$ is not more than $D$. We know that the points fall within $gt\_j$ are of distance zero based on the definition in STEP 1. Suppose the distance of $gt\_j$ to the nearest zone boundary is $m$ and the distance to the other boundary is $M$, we have **(a)** $F(d=0) = \\frac{|gt\_j|}{|I\_j|}$, **(b)** if $D \\leq m$, $F(d \\leq D) = \\frac{|gt\_j|+2D}{|I\_j|}$, and **(c)** if $m \< D \\leq M$, $F(d \\leq D) = \\frac{|gt\_j|+m+D}{|I\_j|}$. Also, we know that $F(d \\leq M) = 1$. By the definition given by the paper, let $F(d=0) = 0$ yield a high precision score if we take $\\bar F(d=0) = 1$. Thus, for any $0 \< d \\leq M$, we have $\\bar F(d) = 1- \\frac{|gt\_j|+d+min(d,m)}{|I\_j|}$. For example, we have point A whose distance to $gt$ is within the range of $m$, its precision score is given by $\\bar F(d) = 1-\\frac{2+1+1}{9} = \\frac{5}{9} = 0.556$. As the precision score of predictions in zone $I\_j$ is given by the integral of the precision score, we have $$P\_{\\text {precision }\_j}=\\frac{1}{\\left|\\mathrm{pred} \\cap I\_j\\right|} \\int\_{x \\in \\mathrm{pred} \\cap I\_j} \\bar{F}\_{\\text {precision }\_j}\\left(\\mathrm{dist}\\left(x, \\mathrm{gt}\_j\\right)\\right) d x = \\frac{1}{2 + 0.5} \\left\[\\int\_1^2\\left( \\frac{2}{9}x + \\frac{1}{3}\\right) dx + \\int\_2^3 1 dx +  \\int\_{8.5}^9 \\left(-\\frac{1}{9}x + 1\\right) dx\\right\]= 0.672$$

<p align="center" width="100%">
<img width="30%" src = "https://user-images.githubusercontent.com/117964124/228994670-fe6d6be2-0ffb-45d4-9be8-25ffc3f0a959.png">
</p>


*   **Individual recall probability:** Similarly, we can have $$P\_{\\text {recall }\_j}=\\frac{1}{\\left|\\mathrm{gt}\_j\\right|} \\int\_{y \\in \\mathrm{gt}\_j} \\bar{F}\_{y, \\text { recall }\_j}\\left(\\mathrm{dist}\\left(y, \\text { pred } \\cap I\_j\\right)\\right) d y$$

       **Explanation:**

       The cumulative distribution function of recall means that if we sample a point **within the affiliation zone**, the score of distance $d$ from the sample point to the predictions. The distance score function is thus given by $$\\bar{F}\_{y, \\text { recall }\_j}(d)=1-\\frac{\\min \\left(d, m\_y\\right)+d}{\\left|I\_j\\right|}$$ 

       where $m\_y$ is the distance of the point $y$ to the nearest border of $I\_j$. For example, in distances of the point to the predictions (two segments) are $d\_1 = 1$ and $d\_2 = 4$ respectively. Based on the definition of $d = dist(y, \\mathrm{pred} \\cap I\_j)$ which is determined by the shortest distance to the set, we have the distance $d = 1$. Also, as $m\_y = 4$, we have the score of distance $d=1$ is given by $1-\\frac{1+1}{9} = 0.778$. Hence, the recall score is given by the integral of each point within $gt\_j$, and we have $$P\_{\\text {recall }\_j}:=\\frac{1}{\\left|\\mathrm{gt}\_j\\right|} \\int\_{y \\in \\mathrm{gt}\_j} \\bar{F}\_{y, \\text { recall }\_j}\\left(\\mathrm{dist}\\left(y, \\text { pred } \\cap I\_j\\right)\\right) d y = \\frac{1}{2} \\left\[\\int\_2^3 dy + \\int\_3^4 \\left(-\\frac{2}{9}y+\\frac{5}{3} \\right)dy \\right\] =  0.944$$

**From my perspective**, the probability function here **is not** given by the possibility of sampling a random point of $y$ whose distance $d \\leq D$. However, it is more intuitive if we let the score of points that falls within the range of predictions be 1 (perfect recall), and let the score drops by the **slope of** $\\frac{2}{|I\_j|} =\\frac{2}{9}$ as the points get far away from the predictions. Also, given the use of $m\_y$, the score remains the same if the points are halfway from the predictions to the borders. This can be seen from the example on the right. 

<p align="center" width="100%">
<img width="50%" src = "https://user-images.githubusercontent.com/117964124/229961711-9b499b48-8b5f-4e8c-8b9d-86ab073f2f24.png">
</p>

#### STEP 4: Averaging of the individual precision/recall probabilities

The precision/recall are defined as the mean over the defined individual probabilities. We let $S:=\\left\\{j \\in \[ 1, n \] ; \\mathrm{pred} \\cap I\_j \\neq \\varnothing\\right\\}$, and obtain:  
$$P\_{\\text {precision }}:=\\frac{1}{|S|} \\sum\_{j \\in S} P\_{\\text {precision }\_j}, \\quad P\_{\\text {recall }}:=\\frac{1}{n} \\sum\_{j=1}^n P\_{\\text {recall }}$$

### Advantages

*   As for algorithmic comparisons, they are expressive enough to perform a fair comparison without the need for introducing additional parameters.
*   The assessment of the proximity between the predicted and the ground truth labels, which is the primary aspect to compare anomaly detection algorithms from a research point of view.

<p align="center" width="100%">
<img width="60%" src = "https://user-images.githubusercontent.com/117964124/229991857-50a5e81f-7bab-4c82-939f-4ccaac1bb4c6.png">
</p>
