# TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Frequency Analysis

*   Summary: Anomaly detection in time series faces two main challenges: modeling complex temporal dependencies and limited labels. Although some methods have been developed, most only look at the time domain and ignore the frequency domain. The authors propose a new method called Time-Frequency Analysis based Time Series Anomaly Detection (TFAD) that uses both time and frequency domains to improve performance. They also use techniques like data augmentation and time series decomposition to make their model work better. ([paper](https://arxiv.org/abs/2210.09693), [code](https://github.com/DAMO-DI-ML/CIKM22-TFAD))
 
### Content

1.  [Introduction](#1-introduction)
2.  [Preliminary](#2-preliminary)
3.  [Methodology](#3-methodology)
4.  [Experiments](#4-experiments)

## 1\. Introduction

### Challenges of anomaly detection in time series

(1) How to model the relationship between a point/subsequence and its temporal context for different types of anomalies. (global point anomaly, seasonality anomaly,  
shapelet anomaly, etc.)

(2) How to deal with limited labels.

     Data augmentation is a possible solution. Although some data augmentation methods have been proposed for time series data ([survey paper IJCAI 21](https://arxiv.org/abs/2002.12478)), how to design and apply data augmentation in time series anomaly detection remains unaddressed.

### Reasons we need frequency domain

It is much easier to detect in the frequency domain than in the time domain for some complex group anomalies and seasonality anomalies. Some work ([KDD workshop 20](https://arxiv.org/abs/2002.09545)) uses data augmentation in the frequency domain to model the time series. 

### Challenges of modeling cross-domain information

How to systematically and directly utilize the frequency domain and time domain information **simultaneously** in modeling time series anomaly detection is unaddressed. 

### Techniques of TFAD

(1) Address the challenge of **capturing context dependency: time series decomposition**

     "Time series decomposition" is a technique used to break down a time series data into its individual components, such as trend, seasonality, and noise. By doing this, it becomes easier to understand and analyze each component separately.

     TFAD is a window-based model structure that uses this decomposition technique to identify anomalies in each of the different components. The goal is to reduce interference between the components so that anomalies can be more accurately identified. In other words, the decomposition module helps to isolate the different components and analyze them independently, which makes it easier to detect unusual patterns or anomalies in each component.

(2) Address the challenge of **label scarcity: data augmentation**

     Data augmentation of TFAD is conducted in different views: normal data augmentation, abnormal data augmentation (not fully considered in existing works), time-domain data augmentation, and frequency-domain data augmentation.

### Main contributions of TFAD

(1) Integrate the frequency domain and time domain

(2) Combine time series decomposition with representation network. Simple TCN performs well with high efficiency. 

(3) Various data augmentation can overcome the label scarcity problem.

## 2\. Preliminary

### Different anomaly types

#### (1) Non-sequential anomaly

 **Definition**: set $\\mathcal{D} \\in R$ as the data space and the normality follows distribution $\\mathcal{N}^+$, which has a probability density function $p^+$, then the anomaly is defined as $$A=\\left\\{d \\in \\mathcal{D} \\mid p^{+}(d) \\leq \\tau\\right\\}, \\tau>0$$ where $\\tau$ is the threshold of anomaly. 

 **Non-sequential anomaly types:**

 (a) Point anomaly (global anomaly): individual data point deviates from normality (most common).

  (b) Context anomaly (conditional anomaly): anomalous in a specific context (e.g., the temperature of 20 degrees in Antarctica).

  (c) Group anomaly (collective anomaly): a group of abnormal points. 

#### (2) Sequential anomaly

 **Definition**: Given a time series as $X = \\left(x\_0, x\_1, x\_2, ..., x\_n \\right)$, where $x\_i$ is the data point at timestamp $i$. Time series can be formally defined by structural modeling to include trend, seasonality, and shapelets, as $$X=\\sum\_n\\left\\{A \\sin \\left(2 \\pi \\omega\_n T\\right)+B \\cos \\left(2 \\pi \\omega\_n T\\right)\\right\\}+\\tau(T)$$ where $T = \\left(1,2,3, ..., n\\right)$ is the timestamp, $w\_n$ is the frequency of wave $n$, and $\\tau(T)$ is the trend component. 

 _**\* Structural modeling:**_ Structural modeling is a technique used to understand and analyze the underlying structure of complex systems. In the context of time series data, it refers to a mathematical framework used to model the various components that contribute to the overall pattern observed in the data. The equation above represents the [Fourier series](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/The%20Ultimate%20Guide%20to%20Fourier%20Transform.md#2-fourier-series) approximation to represent complex periodic patterns in data. It decomposes the time series data into three components: trend, seasonality, and shapelets. **Trend** refers to the overall direction of the data over time, while **seasonality** refers to patterns that repeat at regular intervals (such as daily, weekly, or yearly). **Shapelets** refer to shorter patterns or features within the data that are unique to specific time periods.

 **Sequential anomaly types:**

 (a) Point anomaly (global and context point anomaly): $|x\_t- \\hat{x\_t}| > \\sigma$.

 (b) Pattern anomaly:

       b.1 Shapelet anomaly:  $s(\\rho(\\cdot), \\hat{\\rho}(\\cdot))>\\sigma$ where $s(\\cdot)$ is the similarity function, and $\\hat{\\rho}$ is the expected shapelet.

       b.2 Seasonal anomaly: $s(\\omega(\\cdot), \\hat{\\omega}(\\cdot))>\\sigma$.

       b.3 Trend anomaly: $s(\\tau(\\cdot), \\hat{\\tau}(\\cdot))>\\sigma$.

<p align="center" width="100%">
<img width="60%" src = "https://user-images.githubusercontent.com/117964124/225518930-1af4b796-748d-4dc4-bb54-682946147f06.png">
</p>

## 3\. Methodology

### Motivation: the uncertainty principle of time-frequency analysis

Given the signal $s(t)$, we can have the spectrum $S(\\omega)$ by [Fourier transform](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/The%20Ultimate%20Guide%20to%20Fourier%20Transform.md#6-the-continuous-time-fourier-transform) as $$S(\\omega)=\\frac{1}{\\sqrt{2 \\pi}} \\int\_{-\\infty}^{\\infty} s(t) e^{-j \\omega t} d t$$ where the $\\frac{1}{\\sqrt{2 \\pi}}$ is the normalization factor of the basis function $e^{-j \\omega t}$. The computation of normalizing the complex function can be found [here](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/The%20Ultimate%20Guide%20to%20Fourier%20Transform.md#33-complex-basis-function). 

The uncertainty principle expresses the fundamental relationship between the standard deviation (shows the **broadness** of the signal) of the time domain $\\sigma\_t$ and frequency domain $\\sigma\_\\omega$ as $$\\sigma\_t^2=\\int(t-\<t>)^2|s(t)|^2 d t, \\sigma\_\\omega^2=\\int(\\omega-\<\\omega>)^2|S(\\omega)|^2 d \\omega$$ 
$$\sigma_t \sigma_\omega \geq \frac{1}{2}$$
\*The Fourier uncertainty principle, also known as the **Heisenberg uncertainty principle** in Fourier analysis, states that there is a fundamental trade-off between the accuracy of our knowledge of a signal's time-domain and frequency-domain information. Specifically, the principle states that if we know a signal's time-domain information very precisely, then we cannot know its frequency-domain information very well, and vice versa. In other words, the more concentrated a signal is in time, the more spread out it will be in frequency, and vice versa. A detailed explanation of the uncertainty principle including the variables mentioned in the equation can be found [here](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/The%20Ultimate%20Guide%20to%20Fourier%20Transform.md#76-uncertainty-principle).

### Motivation: data augmentation and decomposition

Unlike most existing works where augmented data should follow the original data distribution, we consider two kinds of data augmentation methods for the anomaly detection task: data augmentation for normal data and anomaly data. Meanwhile, this work also decomposes the time series into different main components, and shows that simple temporal convolution neural networks (TCN) can bring desirable performance. 

### Diagram of TFAD

<p align="center" width="100%">
<img width="80%" src = "https://user-images.githubusercontent.com/117964124/227147476-f4d8f259-c973-4a6a-b889-1caac2884366.png">
</p>

### Network design of TFAD

#### 1\. Data augmentation module

   (1) **Normal data augmentation:** generate data with low noise, which is **more normal**. Use Robust STL to get trend and seasonal information of time series data, and the residual (noises) are ignored; use Fourier transform to get the frequency domain information, and slightly modify both the imaginary and the real parts to gain new data. 

   (2) **Anomaly data augmentation:** generate anomaly data, which is **more anomalous**. 

        (a) **Point anomaly:** changing the scale of a data point means changing the value of the data point. For example, if you have a data point with a value of 10, you could change the scale of that data point by multiplying it by 2, which would give you a new data point with a value of 20. 

        (b) **context anomaly:** One way is to exchange a point or a short sequence of data with another point or sequence of data. Another way is to mix up two different time series, which means taking data from two different time series and combining them in a way that creates an anomaly. 

#### 2\. Data decomposition module

**Hodrick–Prescott (HP) filter** is adopted for time series decomposition. The HP filter is the best-known and most widely used method to separate the trend from the cycle. Denote time series $y\_t, t=\\{1,2, \\ldots, T\\}$ contains a trend component $\\tau\_t$, a cyclical component $c\_t$. That is, $y\_t=\\tau\_t+c\_t$. Then, in HP filter, the **trend component** can be obtained by solving the following minimization problem  

$$  
\\min \_\\tau\\left(\\sum\_{t=1}^T\\left(y\_t-\\tau\_t\\right)^2+\\lambda \\sum\_{t=2}^{T-1}\\left\[\\left(\\tau\_{t+1}-\\tau\_t\\right)-\\left(\\tau\_t-\\tau\_{t-1}\\right)\\right\]^2\\right)  
$$

The first term of the equation penalizes the cyclic component, and the second term of the equation penalizes the growth rate of the trend component. The trade-off between the two goals is governed by the smoothing parameter $\\lambda$. The higher the value of $\\lambda$, the smoother is the estimated trend.

#### 3\. Window splitting module

The assumption is that suppose the context window is normal. If the pattern of the full window is consistent with the pattern of the context window, then there is no anomaly in the suspect window. If there is an anomaly in the suspect window, the pattern of the full window will not be consistent with the context window. The model is trained to give a high score for instances with an anomaly in the suspect window.

<p align="center" width="100%">
<img width="40%" src = "https://user-images.githubusercontent.com/117964124/227164198-2dfdecd6-4438-43a8-ae93-268798079321.png">
</p>

#### 4\. Time and frequency branches

The original time series will be first decomposed into trend and residual components. For each component, we set the full window sequence and context window sequence with the aforementioned window splitting. After that, time-domain representation learning and frequency-domain representation learning for each window sequence will be done to gain rich information on sequences. After that, the distance between the context window and the full window would be measured to calculate the anomaly score.

<p align="center" width="100%">
<img width="80%" src = "https://user-images.githubusercontent.com/117964124/227168183-f187b4b6-cbb4-412e-881a-81d485228f67.png">
</p>


[Discrete Fourier transform (DFT)](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/The%20Ultimate%20Guide%20to%20Fourier%20Transform.md#4-discrete-fourier-transform) is used to obtain the frequency domain features. Although wavelet transform is considered to contain time and frequency information simultaneously, but evaluation shows that it does not gain a good performance as DFT. 

#### 5\. Anomaly score module

$$  
A S=\\mathcal{F}\\left\\{\\text{dis}\\left(R V\_{\\text {treT }}, R V\_{\\text {res } T}, R V\_{\\text {treF }}, R V\_{\\text {res }}\\right)\\right\\}  
$$  

where $R V\_{\\text {tre }}, R V\_{\\text {res } T}$ are representation results of the trend component and residual component in the time domain respectively, $R V\_{\\text {treF }}, R V\_{\\text {res }} F$ are representation results of trend component and residual component in frequency domain respectively, and the distance function is the **cosine similarity**. The suspect window is labeled as an anomaly by a **threshold** for anomaly score. For each data point, the **vote strategy** is adopted to label the anomaly. As it belongs to many suspect windows, if more than half are labeled as anomalies, then the data point will be labeled as an anomaly. 

## 4\. Experiments

### Dataset

(1) **Univariate:** Yahoo, KPI

(2) **Multivariate:** SMAP, MSL

### Metrics

F1 score

<p align="center" width="100%">
<img width="40%" src = "https://user-images.githubusercontent.com/117964124/227186565-de650bc5-0344-481e-a9f9-75e8664f1f23.png">
</p>

Where the supervised setting (sup.) means all labeled data are utilized and unsupervised setting (un.) means the label information is not utilized. Every setting has been run ten times, and the mean and variance are reported. 

### Ablation studies

*   **Hightlight:** With the **decomposition module** added in the time branch, the F1 score gains nearly 30% improvement compared with the same base TCN model, which demonstrates the benefits of decomposition in TFAD model.

### Model analysis and discussion

*   **Window size:** Considering the anomaly sequences are near to each other, when the suspect window is tested, the anomalies will change the representation of the full window and context window significantly, which will conceal the anomalies in the suspect window in the upcoming series. 
*   **Representation learning:** TCN (F1=0.75) works much better than Transformer (F1=0.59). This because the window length can not be set too long to distinguish between the full window and the context window. In this case, TCN can better model the local time series information in the window than Transformer networks.


