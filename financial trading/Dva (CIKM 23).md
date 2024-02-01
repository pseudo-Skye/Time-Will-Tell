# D-VA: Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction
In this research, the focus is on addressing the challenging task of **multi-step** stock price prediction, which is essential for various financial applications, including risk management and derivatives pricing.  The work is a **multi-variate input to single-variate output prediction (regression) task**, and weighs the amount of each stock for an investment portfolio based on the prediction results. 

Given the highly stochastic nature of stock data, the authors propose a novel approach **D-VA** that combines a deep **hierarchical variational autoencoder (VAE)** with **diffusion probabilistic** techniques to perform sequence-to-sequence stock prediction. The diffusion module gradually adds random noise to the data to account for the stock noises, and learns an **energy function** to denoise the corrupted data. The experiments on **portfolio investment** are measured by **Sharpe ratio**. ([paper](https://arxiv.org/abs/2309.00073), [code](https://github.com/koa-fin/dva))

**This work has very similar ideas to **D<sup>3</sup>VAE**: ["Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement"](https://arxiv.org/pdf/2301.03028.pdf) from NeuraIPS 2022.* So, I will combine the technique details from both works to provide a more comprehensive discussion. **Note that the notations used in this tutorial are slightly different from those used in the original paper.**

### Content
1. [Fundamentals](#fundamentals)
2. [Challenges](#challenges)
3. [Diffusion in time series](#diffusion-in-time-series)
4. [Architecture of D-VA model](#architecture-of-d-va-model)
5. [Experiment](#experiment)
6. [The portfolio management](#the-portfolio-management)

## Fundamentals 
This section provides an explanation covering fundamental financial concepts. Also, you can find related articles introducing the basics of [VAE](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/VAE.md#decoding-variational-autoencoders-exploring-the-mathematical-foundations) and [diffusion model](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models). If you are already familiar with these topics, you can just go ahead and skip this section.

### Liquidity Horizon
This refers to the period of time within which an investor expects to be able to sell an asset or investment and convert it into cash without significantly impacting the market price. The explanation of liquidity can be found [here](https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/Python%20library%20FinRL.md#market-liquidity).

### Relation between liquidity horizon and multi-step stock price prediction
Multi-step price predictions can help investors determine optimal entry and exit points over this multi-day period to avoid adversely impacting market prices.

### Low-level features in deep learning
The lower-level features represent basic characteristics of the data, and the model can then build higher-level features by combining these lower-level ones. For example, in an image, low-level factors might represent details like edges, textures, or small patterns. Higher-level factors might represent objects, shapes, or more complex structures in the image. VAEs can learn these hierarchical representations, which makes them capable of capturing low-level features that are essential for understanding and generating data.

## Challenges
1.	**Complexity of Stock Prices**: Stock prices are highly unpredictable and change frequently. The data used for training these models are ‘downsampled’ at specific times from the continuous series, which might not fully capture how stock prices behave. 
2.	**Stochasticity of target sequence**: The data representing the target series, which is what the model is trying to predict, includes random, unpredictable variations (stochastic noise). If the model is trained to predict this noisy data directly, it might not perform well when faced with new, unseen data during testing.
   
### Problems with existing works
1. Most works are **single-step classification** tasks and are limited to low representation expressiveness.
2. These models couldn't fully understand or describe the complex patterns and characteristics present in the stock price data, especially the inherent noises. 

### How D-VA solves the challenges
1.	**Handle complexity of stock prices**

     - The **hierarchical VAE** enhances understanding of stock price behavior by capturing complex and [low-level factors](#low-level-features-in-deep-learning). 
   
2. **Handle stochasticity of target sequence**
  
     - **X-diffusion**: Gradually introduce random noise to the input data.
  	
     - **Y-diffusion**: Augment the target series with noise via a coupled diffusion process.
  	
     - **Denoise**: Predicted noisy targets are "cleaned" to obtain the generalized, "true" target sequence.


## Diffusion in time series
In this section, we provide a very detailed explanation of the diffusion model used in time series. The diffusion model uses a step-by-step process to add random noise to data (**forward diffusion process**) and then figure out how to undo that noise (**denoising score-matching**) to create the original data again.

### Forward diffusion process
<p align="center" width="100%">
<img width="60%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/c9091479-9cd8-4c31-a9d8-a921090eb102">
</p>

Given the process, we have $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$, where $\alpha_t = 1-\beta_t$ and $\bar{\alpha}_t = \Pi\_{i=1}^t \alpha_i$. The detailed calculation can be found in this [post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Based on this equation, instead of sampling $n$ times for each diffusion step, we can obtain $q(x_t|x_0)$ **at any given time step** $t$ based on the original $x_0$. This method is called **reparameterization trick**. **Note that in the general diffusion model, the training process doesn’t use examples in line with the forward process but rather it uses samples from arbitrary timestep t.**

Given time series $X$, we can obtain the noisy samples $X_1, X_2, X_3, ... X_N$, where 

$$
q\left(X_n \mid X\right) =\mathcal{N}\left(X_n ; \sqrt{\bar{\alpha}_n} X,\left(1-\bar{\alpha}_n\right) I\right), 
X_n =\sqrt{\bar{\alpha}_n} X +\sqrt{\left(1-\bar{\alpha}_n\right)} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

We can also decompose $X$ as the ideal part (which contains an informative pattern) and the noisy part as $X:= &lt X_r, \epsilon_X>$, then the diffused noisy $X^{(t)}$ can be rewritten as:

$$
X_n =\sqrt{\bar{\alpha}_n} (X_r + \epsilon_X) +\sqrt{\left(1-\bar{\alpha}_n\right)} \epsilon := \<\sqrt{\bar{\alpha}_n} X_r, \sqrt{\bar{\alpha}_n} \epsilon_X +\sqrt{\left(1-\bar{\alpha}_n\right)} \epsilon\>
$$

We can define the ideal part of $X_n$ as ${X}\_{n,r} =\sqrt{\bar{\alpha}\_n}X_r$ and the noisy part as $\epsilon_{X_n} = \sqrt{\bar{\alpha}_n} \epsilon_X +\sqrt{\left(1-\bar{\alpha}_n\right)} \epsilon$. Based on the [VAE](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/VAE.md) process as we discussed before, we can obtain the latent variable distribution by encoder $q\_\phi(z|x)$ and decoder $p\_\theta(x|z)$. Combined with the diffusion technique, we can have:

$$
q_\phi(Z_n|X_n) = q_\phi(Z_n|{X}\_{n,r}, \epsilon_{X_n}), \ p_\theta(\hat{Y}\_n|Z_n) = p\_\theta(\hat{Y}\_{n,r}|Z\_n) p\_\theta(\epsilon_{\hat {Y}_n}|Z_n) 
$$

Suppose we have a model that takes input $X$ and tries to approximate $Y$. Both $X$ and $Y$ have noise, but ideally, $Y$ has an ideal part, $Y_r$, that the model should perfectly capture if there were no noise. So, for our prediction, denoted as $\hat Y$, we can break it down into an ideal part and a noisy part, like this: $\hat Y := &lt \hat Y_r, \epsilon_{\hat Y}>$. 

When our model is perfect and makes $||Y_r - \hat Y_r|| \rightarrow 0$ (meaning the ideal part is well predicted), the prediction error can be expressed as $||Y - \hat Y|| = ||\epsilon_{Y} - \epsilon_{\hat Y}|| > 0$. This tells us that **the noise cannot be fully captured by the model**. Thus, $\epsilon_{Y}$ can be thought of as a mix of two types of uncertainty:  

* Aleatoric Uncertainty (residual noise, $\delta_{\hat Y}$)
* Epistemic Uncertainty (estimated noise, $\epsilon_{\hat Y}$)

Aleatoric uncertainty is challenging to model accurately, so it's intuitive that: 

$$
D_\mathrm{KL} \left(p(\epsilon_{Y})||p_\theta(\epsilon_{\hat Y})\right) > D_\mathrm{KL} \left(p(\epsilon_{\hat Y})||p\_\theta(\epsilon\_{\hat Y})\right)
$$

However, if we look at the diffused version of $Y$, we notice that the noisy part of $Y_n$ can be represented as $\epsilon_{Y_n} = \sqrt{\bar{\alpha^\prime}\_n} \epsilon\_Y +\sqrt{\left(1-\bar{\alpha^\prime}\_n\right)} \epsilon$. This noisy part starts to resemble $\epsilon$ when $n \rightarrow \infty$. Since $\epsilon$ follows a standard Gaussian distribution, it becomes easier to model the noisy part in the diffused data using $p_\theta(\epsilon_{\hat{Y}_n}|Z_n)$. So in the paper, the author claims that "**by coupling the generative and diffusion process, the overall prediction uncertainty of the model can be reduced**" as described in their equation (4):

$$
\lim_{n \rightarrow \infty} D_\mathrm{KL}\left(p\left(\epsilon_{Y_n}\right) || p_\theta\left(\epsilon_{\hat{Y}\_n} | Z_n\right)\right) < D_\mathrm{KL}\left(p\left(\epsilon_{Y}\right)||p_\theta\left(\epsilon_{\hat{Y}}\right)\right)
$$

The forward diffusion process can be visualized in the following figure: 

<p align="center" width="100%">
<img width="80%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/1261e7bc-505f-436a-a61e-31e47fdd6af0">
</p>

As described in the figure, for the coupled diffusion process, this work aims to minimize the divergence between the VAE predicted $p_\theta(\hat Y_n)$ and diffused ground truth $p(Y_n)$. 

### Denoising score matching
Although the time series data can be augmented with the coupled diffusion probabilistic model, the generative distribution $p_\theta(\hat{Y}_n)$ tends to move toward the diffused target series $p(Y_n)$ which has been corrupted. However, it is possible to obtain samples closer to the **ideal part** of data, by adding an extra denoising step on the final sample $\hat{Y}_n$. 
This [tutorial](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Score%20matching.md) provides a comprehensive introduction to denoising score-matching, which is essential to review before diving into the specifics of the techniques employed in this section. 

According to how we defined forward diffusion, we kept including Gaussian errors in the target sequence $Y$ over time as $Y_n =\sqrt{\bar{\alpha}_n} Y +\sqrt{\left(1-\bar{\alpha}_n\right)} \epsilon$. Then, to reconstruct $Y$ from the corrupted $Y_n$, we can first obtain the probability of finding the system in the state given $E(Y_n;\zeta)$ as:

$$
p(Y_n ;  {\zeta})=\frac{1}{Z(\zeta)} \exp (-E(Y_n ;  {\zeta}))
$$

The denoising score matching aims to minimize the distance between the score of $p(Y_n ;  {\zeta})$ and $q_\epsilon(Y_n|Y)$ as:

$$
L_{DSM_{q_{\epsilon}}}(\zeta)=\mathbb{E}\_{q_{\epsilon}(Y, Y_n)}\left\[\frac{1}{2}\left\Vert\frac{\partial \log p(Y_n ;  {\zeta})}{\partial Y_n}-\frac{\partial \log q_{\epsilon}(Y_n \mid Y)}{\partial Y_n}\right\Vert^2\right\]
$$

To calculate the score of $p(Y_n ;  {\zeta})$, we have:

$$
\frac{\partial \log p(Y_n ;  {\zeta})}{\partial Y_n} = \frac{\partial \log \frac{1}{Z(\zeta)} \exp (-E(Y_n ;  {\zeta}))}{\partial Y_n} = \frac {\partial (-\log Z(\zeta) - E(Y_n ;  {\zeta}))} {\partial Y_n} = -\frac {\partial E(Y_n ;  {\zeta})} {\partial Y_n} = -\nabla_{Y_n} E(Y_n ;  {\zeta})
$$

To calculate the score of $q_\epsilon(Y_n|Y)$, we have:

$$
\frac{\partial \log q_{\epsilon}(Y_n \mid Y)}{\partial Y_n} = \frac{1}{\left(1-\bar{\alpha}_n\right) \epsilon^2} (Y-Y_n)
$$

Thus, the denoising score matching objective function can be written as:

$$
L_{DSM_{q_{\epsilon}}}(\zeta, n)=\mathbb{E}\_{q_{\epsilon}(Y, Y_n)}\left\[\frac{1}{2}\left\Vert\frac{\partial \log p(Y_n ;  {\zeta})}{\partial Y_n}-\frac{\partial \log q_{\epsilon}(Y_n \mid Y)}{\partial Y_n}\right\Vert^2\right\] = \mathbb{E}\_{q_{\epsilon}(Y, Y_n)} l(\epsilon) \left\Vert -\epsilon^2 \nabla_{Y_n} E(Y_n ;  {\zeta}) - (Y-Y_n) \right\Vert^2 = \mathbb{E}\_{q_{\epsilon}(Y, Y_n)} l(\epsilon) \left\Vert (Y-Y_n)+ \epsilon^2 \nabla_{Y_n} E(Y_n ;  {\zeta}) \right\Vert^2
$$

Thus, to denoise the corrupted $Y_n$, we can apply a **single-step gradient denoising jump** as:

$$
Y_\text{clean} = Y_n - \epsilon^2 \nabla_{Y_n} E(Y_n ;  {\zeta})
$$

**In both the paper of DVA and D<sup>3</sup>VAE**, as the diffusion step $n$ gets larger, the diffused $Y_n$ is getting close to $\hat{Y_n}$ which is generated from the VAE model based on the diffused $X_n$. Thus, we can have the above objective function as:

$$
L_{DSM_{q_{\epsilon}}}(\zeta, n)= \mathbb{E}\_{q_{\epsilon}(Y, \hat{Y_n})} l(\epsilon) \left\Vert (Y-\hat{Y_n})+ \epsilon^2 \nabla_{\hat{Y_n}} E(\hat{Y_n}; {\zeta})\right\Vert^2
$$

and 

$$
Y_\text{clean} = \hat{Y_n} - \epsilon^2 \nabla_{\hat{Y_n}}E(\hat{Y_n}; {\zeta})
$$

The paper claims that here $\nabla_{\hat{Y_n}} E(\hat{Y_n};{\zeta})$ is an estimation of the sum of the noise produced by the generative VAE and the inherent random noise in the data. 

---------------------------------------------------------------
**HOWEVER, HERE IS MY CONCERN...**

Based on the math in those equations, even though the paper didn't explain this part in detail, the idea behind simplifying the score-matching objective function is assuming that the corrupted data $\hat{Y_n}$ should follow a Gaussian distribution given the input $Y$. However, only when $n$ reaches a considerably large value we have $\hat{Y_n} \rightarrow Y_n$. Otherwise, $q_\epsilon(Y_n\mid Y) \neq q_\epsilon(\hat{Y_n}\mid Y)$, indicating an inability to calculate derivatives for $\log q_{\epsilon}(\hat{Y_n} \mid Y)$ in the same manner as $\log q_{\epsilon}(Y_n \mid Y)$. Thus, based on the underlying mathematics, the denoising gradient can only be applied at **large steps** of the diffused data, and the assumption may be weak if $n$ is not sufficiently large.

----------------------------------------------------------------

## Architecture of D-VA model
<p align="center" width="100%">
<img width="50%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/df929126-8e2f-4817-9c3f-473a87d31844">
</p>

Putting together the loss equations, the objective function of the proposed method is as follows: 

$$
\mathcal{L}=L_\text{MSE}+\gamma \cdot L_\text{KL}+\eta \cdot L_\text{DSM}
$$

where $\gamma$ and $\eta$ refers to the tradeoff parameters. Additionally, $L_\text{MSE}$ calculates the overall mean squared error (MSE) between the predicted sequence $\hat{Y_n}$ and diffused sequence $Y_n$ for all diffusion steps. 

### Training process
1. Applying a coupled diffusion process to both the input $X$ and target $Y$ sequences, resulting in diffused sequences $X_n$ and $Y_n$.
2. Training a VAE to generate predictions $\hat{Y_n}$ from the diffused input sequence $X_n$. These predictions are then matched to the diffused target sequence $Y_n$.
3. Simultaneously, train a denoising energy function $E(\hat{Y_n}; {\zeta})$ to obtain "clean" predictions $Y$ from the generated predictions $\hat{Y_n}$.

### Inference process
1. The trained hierarchical VAE model is employed to generate predictions $\hat{Y}$ from the input sequences $X$.
2. The predicted sequence $\hat{Y}$ undergoes a one-step denoising jump by the energy function, obtaining the cleaned results as $\hat{Y}_\text{final}$.

## Experiment
### Data preprocessing
- **Input**: $\mathbf{X} = \{\mathbf{x}\_{t-T+1}, \mathbf{x}\_{t-T+2}, \cdots, \mathbf{x}\_t\}$

  During $T$ trading days, $\mathbf{x}\_t=\[o_t, h_t, l_t, v_t, \Delta_t, r_t\]$ for each day includes the **normalized** open($o_t = o_t/c_{t-1}$), high($h_t = h_t/c_{t-1}$), low prices($l_t = l_t/c_{t-1}$), volume traded($v_t$), absolute returns($\Delta_t = c_t-c_{t-1}$), and percentage returns($r_t = c_t/c_{t-1}$).

- **Output**: $\mathbf{y}=\{r_{t+1}, r_{t+2}, \cdots, r_{t+T^{\prime}}\}$ over the next $T^{\prime}$ trading days.

### Dataset
<p align="center" width="100%">
<img width="50%" src = "https://github.com/pseudo-Skye/StudyNotes/assets/117964124/b3058d0e-37d9-4987-b98c-cea7df513f4d">
</p>

- The dataset comprises three years of **daily** stock prices.
- **ACL18** (2014/01/01-2017/01/01): historical prices of 88 high trade volume stocks from the U.S. market, which represents the top 8-10 stocks in capital size **across 9 major industries**. 
- Test year: 2016, 2019, 2022

### Baselines and comparisons
Most baseline models are derived from the **general seq2seq task**. The findings indicate that D-VA works better (lower MSE), particularly in predicting the stock prices for the **next 10 days**.

## The portfolio management
This work can be used for portfolio allocation based on its multistep predictions. Say that if the model predicts the future $t$ days returns and there are $S$ stocks in the pool. 

### Mean-Variance optimization (learn the optimal weight of each stock)
Based on the ideas of [portfolio mean and variance](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Math%20ideas%20look-up%20dictionary.md#2-portfolio-mean-and-variance), the average return of each stock is denoted as $u_t \in \mathbb{R}^{S}$ and covariance matrix across the stock pool as $\Sigma_t \in \mathbb{R}^{S \times S}$. If we invest by the portfolio weight $w \in \mathbb{R}^{S}$, the portfolio mean is $w^{\prime}u_t$ and the portfolio variance is $w^{\prime} \Sigma_t w$. The purpose of portfolio management is to **maximize the expected returns while minimizing the risk (variance)**, and here the paper uses **Markowitz’ mean-variance optimization** to learn an optimal weight $w$, such that:

$$
\max_w (w^{\prime}u_t - \frac{\tau}{2} w^{\prime} \Sigma_t w), ~\sum w^{\prime}\mathbf{1} = 1
$$

where $\tau$ is the **risk-aversion parameter**, which can be tuned as a hyper-parameter on the validation set.

**This work also sets the **no short-sales constraint**, which forces all elements in weights $`w`$ to be **non-negative**.* 

### Graphical lasso regularization (learn the precision matrix between stocks)
The Graphical Lasso aims to create **a sparse graph**, meaning it wants to keep only the important connections and set the rest to zero. This sparsity can help obtain a more interpretable model that can be used for tasks such as feature selection and network inference. Here, we can estimate a sparse [precision matrix](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Math%20ideas%20look-up%20dictionary.md#4-precision-matrix) by solving the following optimization problem:

$$
\max_{\Theta} \log \text{det}(\Theta)-\text{tr}(\Sigma \Theta)-\lambda\|\Theta\|_1
$$

where $\Theta$ is an inverted covariance matrix to be learned, and $\lambda$ is a regularization parameter that controls the level of sparsity. 

Objective Explanation:
- $\text{tr}(\Sigma \Theta)$ : The trace of the product of $\Sigma$ and $\Theta$, which encourages many off-diagonal entries to become zero.
- $\log \text{det}(\Theta)$ : The log determinant of $\Theta$. Maximizing the determinant of $\Theta$ encourages the variables to be less correlated with each other, which results in a more compact precision matrix (a sparse precision matrix that accurately models the data).
- $\|\Theta\|_1$ : The L1 norm (sum of absolute values) of the elements in $\Theta$.
- $\left(\lambda\|\Theta\|_1\right)$: This term enforces sparsity in $\Theta$. The larger the $\lambda$, the sparser the estimated precision matrix.

### Evaluation metric and baselines
**Evaluation metric**: [Sharpe Ratio (SR)](https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/Python%20library%20FinRL.md#evaluation-metrics): $\text{SR} = \frac{\hat \mu}{\hat \sigma}$
**Baseline**: The equal-weight portfolio ($w_s = \frac{1}{S}$)

*The last part of the paper doesn't explain **how they use the learned precision matrix for assigning weights** and **what is the buy and sell strategy**. I think the approach involves instantly adjusting investment weights for the predicted days $t$. However, **transaction fees** are not considered in this task.*


