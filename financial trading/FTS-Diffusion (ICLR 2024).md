# Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns
This paper addresses the challenge of **limited data availability** for training deep learning models in financial applications. Traditional methods struggle to model financial time series due to their irregular and scale-invariant patterns, which differ in duration and magnitude. To tackle this problem, the authors introduce **FTS-Diffusion**, a generative framework designed to handle irregularity and scale-invariance. One interesting aspect of this work is how it looks at financial time series data from **three** angles: **pattern, duration, and magnitude**. By using these statistics, the proposed framework can better understand and model financial patterns. ([paper](https://openreview.net/pdf?id=CdjnzWsQax), code is not released)

## Content
1. [Fundamentals](#fundamentals)
2. [Challenges and motivation](#challenges-and-motivation)
3. [Architecture of FTS-Diffusion](#architecture-of-fts-diffusion)
4. [Methodology](#methodology)
5. [Experiments](#experiments)
6. [Insights](#insights)

## Fundamentals
### Diffusion in time series
This work employs a diffusion-based network to generate synthetic financial data. I have created a comprehensive tutorial on the diffusion model in time series, which can be found [here](https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/Dva%20(CIKM%2023).md#forward-diffusion-process). However, their approach to **denoising** differs from that of other diffusion-based time series forecasting methods. We will delve into this topic further in the upcoming methodology section.

### Dynamic time warping
This work also uses dynamic time warping (DTW) as a key method to measure distances between financial patterns. DTW can find similarities between patterns **regardless of their size or duration**. We also have a detailed [tutorial](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/DTW.md#dynamic-time-warping) on DTW available.


### Irregularity and scale-invariance in financial time series
- **Irregularity**: Irregularity in financial time series refers to the lack of predictable, uniform patterns or frequencies in the data. Unlike data that show consistent, repeating patterns at fixed intervals (like regular heartbeats on an ECG chart), financial data often behaves in unpredictable ways with no clear rhythm.

    **Example**: Imagine tracking the daily closing prices of a stock over a year. Instead of seeing a steady, repeating pattern each week or month, you might notice the prices fluctuate wildly with no apparent regularity. 

- **Scale-invariance**: Scale-invariance in financial time series means that certain patterns within the data maintain their shape and characteristics even when **observed over different timeframes or scales**. This means the patterns look similar regardless of whether you zoom in or out on the timeline.

    **Example**: Consider a price movement pattern in a stock chart that resembles a sharp increase followed by a gradual decrease. This pattern might occur over a few days, weeks, or months, but the overall shape remains consistent regardless of the timeframe.

<p align="center" width="100%">
<img width="70%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/bc69f4e0-1f9d-4bfe-a8f5-c7d1a0d36bbe">
</p>


## Challenges and motivation
### Challenges of financial data
1. There's just **not enough data** to train deep-learning models well. Unlike in other sciences where you can run experiments to get more data, in finance, we're stuck with what we've got from the past.
2. Financial data like prices and returns, is really **noisy**, which means it's hard to pick out the important stuff from all the extra noise.
Put all together, Using lots of **noisy data** and **limited datasets** can cause a deep learning model to **overfit**.

### Problems of existing methods
1. Data augmentation by generative models (e.g., GAN and Diffusion) to create more time series data. But when it comes to financial data, these models struggle because financial data suffers from the problem of **scale-invariance**.
2. Most existing models expect patterns to be regular and consistent and struggle to capture complex patterns. When time series data is divided into **fixed intervals**, these models often end up capturing only part of a pattern or a blend of multiple different patterns, rather than accurately representing the complete pattern.

## Architecture of FTS-Diffusion
### Overview
1. **Pattern recognition module**: This module identifies irregular and scale-invariant patterns in financial time series data. It uses a new algorithm called **Scale-Invariant Subsequence Clustering (SISC)** with DTW to accurately detect and separate these complex patterns.
2. **Generation module**: Once patterns are identified, this module synthesizes segments of scale-invariant patterns. It employs a **diffusion-based network** that creates synthetic data segments based on the identified patterns.
3. **Evolution module**: The evolution module connects and sequences the synthesized segments to form a complete time series. It uses a **pattern transition network** to capture the dynamic relationship between consecutive patterns.

<p align="center" width="100%">
<img width="70%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/0552eb01-ad08-418e-ab7e-533f2a5b165b">
</p>

### Problem definition
Given a times series of length $T = \sum_m {t_m}$, it can be decomposed into $M$ segments, where each segment is of length $t_m$. The segment $\mathbf{x}_m$ is sampled from a conditional distribution $f(\cdot | \mathbf{p}, \alpha, \beta)$ dependent on the pattern $\mathbf{p} \in \mathcal{P}$, duration scaled by $\alpha$ and magnitude scaled by $\beta$. Then, the transition of these three aspects of the time series can be described by **Markov model** as $Q(\mathbf{p}_j,\alpha_j,\beta_j| \mathbf{p}_i,\alpha_i,\beta_i)$. 

<p align="center" width="100%">
<img width="60%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/4dec80cd-e9fd-4926-a4d8-f9be7f3844d4">
</p>

Thus the previous three modules can be described as:
1. **Pattern recognition module**: identify patterns $\mathbf{p} \in \mathcal{P}$, and group the segments into clusters according to their patterns.
2. **Generation module**: Learn the distribution $f(\cdot | \mathbf{p}, \alpha, \beta)$.
3. **Evolution module**: Learn the pattern transition probabilities $Q(\mathbf{p}_j,\alpha_j,\beta_j| \mathbf{p}_i,\alpha_i,\beta_i)$.

The algorithm starts by sampling a segment of a time series from the real observed dataset. Then, it uses that segment as a starting point to generate more data. It figures out what **the next segment** should be like in terms of its pattern, how long it should be, and how big it should be. Using these details, the model just keeps generating new segments to the collection, making it grow larger and larger.

### Methodology
#### 1. Pattern recognition 
This work proposes the **Scale-Invariant Subsequence Clustering (SISC)** algorithm to divide the time series into segments of **variable lengths** and group them into $K$ clusters. The optimal segment length $l^*$ is found by the one that **minimizes the distance to the nearest clustering centroid**:

$$
l^*=\underset{l \in \[l_{\min }, l_{\max }\]}{\arg \min } d\left(X_{t: t+l}, \mathbf{p}\right), ~\forall \mathbf{p} \in \mathcal{P} .
$$

Here, the distance $d(\cdot)$ is determined by **DTW**. Based on the experimental setting, this work set minimum and maximum segment lengths to be 10 and 21 respectively. 

<p align="center" width="100%">
<img width="30%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/923c36e5-4a5b-486b-aba8-f089b4fedea7">
</p>

<p align="center" width="100%">
<img width="60%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/8011025c-26ea-41d9-a050-4b938fdd8877">
</p>

#### 2. Pattern generation

<p align="center" width="100%">
<img width="40%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/85b2d7cb-d11d-4da5-b1db-ce4512dbe097">
</p>

**(a) Scaling autoencoder**

This encoder takes the variable-length segments $\mathbf{x}_m$ and stretches them into fixed-length representations $\mathbf{x}_m^0$ that match the dimension of the reference patterns $\mathbf{p}$, and the decoder reconstructs the variable-length segments from the fixed-length representations. 


**(b) Pattern-conditioned diffusion network**

The **forward diffusion** is slightly different from the classical diffusion process, here the noised time series sample at diffusion step $N$ is given by (I simplify $\mathbf{x}_m^0$ as $\mathbf{x}^0$):
   
$$
\mathbf{x}^N=\mathbf{x}^0+\sum_{i=0}^{N-1} \mathcal{N}\left(\mathbf{x}^{i+1} ; \sqrt{1-\beta}\left(\mathbf{x}^i-\mathbf{p}\right), \beta I\right)
$$

In this equation, $\beta$ represents the **magnitude** of the segments. 

*This work does not discuss if this equation can be simplified further like the classical one, and if not, the forward diffusion process would involve sampling from a Gaussian distribution at each step, which could be very time-consuming.

The denoising process is given by: 

$$
\mathbf{x}^0=x^N-\sum_{i=0}^{N-1} \epsilon_\theta\left(\mathbf{x}^{i+1}, i, \mathbf{p}\right)
$$

In this equation, $\epsilon_\theta$ is the neural network that aims to recover the time series from the added Gaussian noise, and $\epsilon^i$ is the noise added during the diffusion process data step $i$. The loss function of the diffusion network is given by: 

$$
\mathcal{L}(\theta)=\mathbb{E}\_{\mathbf{x}_m}\[||\mathbf{x}\_m-\hat{\mathbf{x}}\_m ||\_2^2\] + \mathbb{E}\_{\mathbf{x}\_m^0, i, \epsilon}\[||\epsilon^i-\epsilon\_\theta(\mathbf{x}\_m^{i+1}, i, \mathbf{p})||\_2^2\]
$$

#### 3. Pattern evolution
**Markov chain** is used to model how different patterns $p$, lengths $\alpha$, and magnitudes $\beta$ transition between consecutive segments of generated data, and the network is written as:

$$
\left(\hat{p}\_{m+1}, \hat{\alpha}\_{m+1}, \hat{\beta}_{m+1}\right)=\phi\left(p_m, \alpha_m, \beta_m\right)
$$

where $m$ means the current step, and $m+1$ denotes the next step. The network is trained via the loss function:

$$
\mathcal{L}(\phi)=\mathbb{E}\_{\mathbf{x}\_m}\[\ell\_{C E}\left(p_{m+1}, \hat{p}\_{m+1}\right)+||\alpha\_{m+1}-\hat{\alpha}\_{m+1}||\_2^2+||\beta\_{m+1}-\hat{\beta}\_{m+1}||_2^2\]
$$

where $\ell\_{C E}$ is the [cross entropy](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Information%20entropy.md#4-cross-entropy). 

## Experiments 
### Experimental settings
1. **Dataset**: S&P 500, Google stock, corn futures
2. Compare the **returns**: In finance, the actual prices of assets tend to jump around a lot and don't really follow a predictable pattern. This makes it hard to use mathematical models to understand them. However, when we look at the changes in prices over time, called **returns**, they tend to behave more consistently (consistent mean and variance). So, to see how well the proposed method works, this work compares the return series generated by FTS-Diffusion to those generated by other baselines.

<p align="center" width="100%">
<img width="60%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/04f03c71-aa70-4d1d-b335-2972606d23b4">
</p>

&nbsp;&nbsp;&nbsp;&nbsp; Two facts that can be observed from the above figure:

&nbsp;&nbsp;&nbsp;&nbsp; (1) The return follows a **heavy-tailed** distribution (tails won't approach zero as the x value gets larger) based on both the distribution plot and [QQ-plot](https://www.youtube.com/watch?v=okjYjClSjOg). 

&nbsp;&nbsp;&nbsp;&nbsp; (2) Autocorrelations of absolute returns decay slowly over time.

&nbsp;&nbsp;&nbsp;&nbsp; It seems the proposed method can generate synthetic stock data that follows these two characteristics. However, the decay in autocorrelation in the generated data is smoother than in the real ones.

3. **Evaluation**: KS and AD statistics.
   - **KS test**: It is more sensitive to differences in the **middle** (or center) of the distribution. If two distributions have noticeable differences in their **central values** (e.g., means or medians), the KS test is more likely to detect these differences.
   - **AD Test**: It pays more attention to the **tails** of the distribution. This means it is more likely to detect differences in the **extreme values** or the shape of the distribution's tails.

      These two statistics can compare the generated time series with the observed time series from the perspective of **distribution**.

4. Downstream task test
   - **Train on mixture, test on real**: this paper uses **LSTM** to predict the **next day's** stock prices and trains the model with 70% generated data and 30% real data. The results show that for other generative models, prediction accuracy drops when more generated data is used. However, **the proposed method's performance stays almost the same**.
   - **Training on augmentation, test on real**: this test gradually adds more generated data to the training set to see if it improves prediction accuracy. The results show that when **100 years** of generated time series data is added, the prediction accuracy significantly improves compared to other generative models.

## Insights
Some takeaways:

1. Financial patterns are diverse but can repeat at different levels of detail.
2. Training the model with a large amount of stock data can improve prediction accuracy, making **foundation models** a potential solution.
