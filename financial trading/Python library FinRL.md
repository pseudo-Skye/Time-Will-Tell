# FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
FinRL is a powerful Deep Reinforcement Learning (DRL) library designed to make quantitative finance and stock trading accessible to beginners while providing advanced capabilities for experienced practitioners. It offers a wide range of features and tools to facilitate the development of stock trading strategies and provides a platform for learning and experimentation in the field of finance. The trading environments are based on the OpenAI Gym framework. ([paper](https://arxiv.org/abs/2011.09607), [code](https://github.com/AI4Finance-Foundation/FinRL)) At the end of this article, there is a very detailed explanation of [Mahalanobis distance](#mahalanobis-distance), which will be used as an important financial turbulence index in FinRL. 

## Benefits of DRL in financial trading
1. **Portfolio Scalability**: This means that DRL allows you to manage and trade a portfolio of assets (like stocks) effectively, even if the portfolio contains many different assets. DRL can adapt to handle many assets in the portfolio, making it scalable. 
2. **Market Model Independence**: This means that DRL doesn't rely on a specific model of how the market works. In finance, there are different theories and models about how markets behave. DRL doesn't need to follow any particular model but learns how to make decisions by interacting with real market data.

## Fundamentals in stock trading
### Market liquidity
It refers to the ease with which an asset or security can be bought or sold in the market without significantly affecting its price. It reflects the ability of traders to execute large orders quickly without causing significant price fluctuations. In a highly liquid market, there are many buyers and sellers, and trading volumes are high, making it easy to enter (buy) or exit (sell) a position.

### Bid-Ask Spread
The bid-ask spread is a crucial concept in stock trading. It represents the difference between the highest price a buyer is willing to pay (bid) for a stock and the lowest price a seller is willing to accept (ask) for that same stock at a given moment. The spread is the difference between these two prices. It's essentially a transaction cost. When you buy a stock, you typically pay a price slightly higher than the bid price, and when you sell, you receive a price slightly lower than the ask price.

#### Example
Suppose you're interested in buying shares of XYZ Company, and you're looking at the market quotes:

Bid Price: $10.00

Ask Price: $10.05

**In this scenario**:

The **bid price** ($10.00) is the highest price that a buyer in the market is willing to pay for a share of XYZ Company at that moment. In other words, if you want to sell your shares immediately, the most you can get for each share is $10.00.

The **ask price** ($10.05) is the lowest price at which a seller in the market is willing to sell a share of XYZ Company. If you want to buy shares immediately, you'll need to pay $10.05 per share.

The difference between the two prices is the **bid-ask spread**: $10.05 (Ask Price) - $10.00 (Bid Price) = $0.05

In this example, the bid-ask spread is $0.05, meaning that if you want to buy shares immediately, you'll pay a slightly higher price ($10.05) than the current highest bid price ($10.00). Conversely, if you want to sell shares immediately, you'll receive a slightly lower price ($10.00) than the current lowest ask price ($10.05). Bid prices are typically lower than ask prices, creating the spread. This spread represents transaction costs and the profit margin for market makers and provides liquidity to the market. However, it's important to note that bid and ask prices can change rapidly due to market dynamics and trading activity.
 	  
## Features of FinRL
1. **Comparison with existing strategies**: It allows users to compare their trading strategies with existing schemes effortlessly. It provides standard evaluation baselines and fine-tuned DRL algorithms, such as DQN, DDPG, PPO, SAC, A2C, TD3.
2. **Modular architecture**: The library is organized in a layered architecture with a modular structure, making it highly adaptable and customizable. Users can easily extend and modify components to suit their specific needs.
3. **Support for various stock markets**: FinRL supports multiple stock markets, including NASDAQ-100, DJIA, S&P 500, HSI, SSE 50, and CSI 300, allowing users to simulate trading environments across different markets.
4. **Incorporation of trading constraints**: It takes into account crucial trading constraints such as transaction costs, market liquidity, and investor risk aversion, ensuring that trading strategies are realistic and practical.
5. **Reproducibility**: FinRL emphasizes reproducibility, making it easier for users to replicate experiments and verify results. 
6. **Multiple-level of granularity**: they allow data frequency of the state features to be daily, hourly or on a minute basis.
7. **Application demonstrations**: The library includes three application demonstrations: single stock trading, multiple stock trading, and portfolio allocation.

## The guide of FinRL
### The architecture of the FinRL library
**Three layers**: (1) stock market environment, (2) DRL trading agent, and (3) stock trading applications. 
![image](https://github.com/pseudo-Skye/StudyNotes/assets/117964124/2474e606-3fe3-4b18-90bd-fd6272ac5087)

### Baselines in the FinRL library
These traditional strategies serve as benchmarks for evaluating the performance of more advanced Deep Reinforcement Learning (DRL) trading strategies in the FinRL library. They represent different approaches to balancing returns and risks in investment portfolios.
1. **Passive Buy-and-Hold Trading Strategy**: This is one of the simplest investment strategies. With this approach, an investor buys a set of assets, such as stocks, and holds onto them for an extended period, regardless of market fluctuations. The idea is to benefit from the long-term growth of these assets.
2. **Mean-Variance Strategy**: This strategy aims to find a balance between maximizing returns and minimizing risks. It involves calculating the expected returns and volatility (risk) of different assets in a portfolio. Then, the investor constructs a portfolio that optimally balances these two factors to achieve the desired risk-return trade-off.
3. **Min-Variance Strategy**: Similar to the mean-variance strategy, the min-variance strategy also focuses on risk minimization. However, instead of seeking a specific level of return, it aims to construct a portfolio with the lowest possible risk (volatility) while still achieving some return.
4. **Momentum Trading Strategy**: This strategy is based on the idea that assets that have performed well recently will continue to do so, and those that have performed poorly will continue to underperform. Investors following this strategy buy assets that have shown positive momentum and sell assets with negative momentum.
5. **Equal-Weighted Strategy**: In this approach, an investor allocates an equal amount of capital to each asset in their portfolio. This strategy assumes that all assets have an equal chance of performing well.

### Overview of different benchmark DRL algorithms in FinRL
![image](https://github.com/pseudo-Skye/StudyNotes/assets/117964124/93b344bf-1690-4206-b8fb-b87b79df0004)

### Evaluation metrics
1. **Final Portfolio Value**: This metric tells you the total value of your investment portfolio at the end of your trading period. 
2. **Annualized Return**: Annualized return measures how much, on average, your investment grows each year. 
3. **Annualized Standard Deviation**: This metric indicates the average variability or fluctuations in your investment returns over a year. Lower values suggest lower risk.
4. **Maximum Drawdown Ratio**: This measures the maximum loss you experienced from the peak value your investment portfolio reached to the lowest value your portfolio reaches after the peak, before it starts to recover. Smaller drawdowns are better because they indicate lower losses during tough times.
5. **Sharpe Ratio**: It evaluates the risk-adjusted return of your investment. A higher Sharpe ratio suggests you're getting better returns for the level of risk you're taking.
Many investors and fund managers use the S&P 500 as a benchmark to evaluate the performance of their investment portfolios. It provides a standard against which to measure returns and assess risk-adjusted performance.

## Process of training and testing the RL model
1. **Data split**: split the dataset into train, validation, and test sets.
2. Use **rolling window** to rebalance the portfolio and retrain the model periodically, FinRL provides rolling window selection among daily, monthly, quarterly, yearly or customized. 
3. Add **trading constraints**: set transaction cost by a flat fee (fixed amount per trade) or per share percentage (1/1000 – 2/1000 per trade).
4. Consider **market liquidity**: add **bid-ask spread** as a parameter to the stock closing price to simulate real-world trading experience.
5. Risk control: employ a **financial turbulence index** that measures extreme price fluctuations.

### Mahalanobis distance
The measure of the financial turbulence index is based on the **Mahalanobis distance**. To grasp the concept of Mahalanobis distance, it's crucial to first delve into the geometric significance of linear transformations, eigenvalues, and eigenvectors. You can find a comprehensive [tutorial](https://www.3blue1brown.com/topics/linear-algebra) that investigates these fundamental topics. Next, we must familiarize ourselves with the notion of a [covariance matrix](https://www.cuemath.com/algebra/covariance-matrix/) which describes the correlation between variables. 

#### Linear transformation and covariance: the Cholesky factor
Imagine we have a dataset that represents a circle within a 2D space. Within this context, we have the capacity to manipulate this circle by applying scaling and rotation transformations. These transformations can be elegantly understood through the lens of eigenvalues and eigenvectors.

![image](https://github.com/pseudo-Skye/StudyNotes/assets/117964124/81a0a69b-1979-4828-b700-56627a323848)

The combined scaling and rotation operation can be mathematically represented as $T = RS$, where $S$ represents the scaling function, and $R$ embodies the rotation function. If we introduce parameters like the rotation angle $\theta$, as well as the scaling factors $s_x$ and $s_y$, we can express $S$ and $R$ as:

$$
S = \begin{bmatrix}s_x & 0 \\\ 0 & s_y \end{bmatrix}, R=\begin{bmatrix} \cos (\theta) & -\sin (\theta) \\\ \sin (\theta) & \cos (\theta)\end{bmatrix}
$$

Given the dataset $D$, after the transformation $T = RS$, the dataset becomes $D^\prime = TD$, where the original dataset has covariance $\Sigma$, and the dataset $D^\prime$ has covariance $\Sigma^\prime$. We can then write the representation of the covariance $\Sigma^\prime$ by its eigenvectors and eigenvalues as:

$$
\begin{split}
\Sigma^\prime V &= VL \\
&= \begin{bmatrix}\vec{v_1} & \vec{v_2} \end{bmatrix} \begin{bmatrix}l_1 & 0 \\\ 0 & l_2 \end{bmatrix}
\end{split}
$$

where $\[\vec{v_1} \\ \vec{v_2}\]$ describes the direction of rotation $R$, and matrix $L$ describes the scaling matrix $S$ as it scales the eigenvectors along its direction. It is important to note that $R$ is an orthogonal matrix, and $S$ is a diagonal matrix. Thus, we have $R^{-1} = R^T$ and $S = S^T$. The property of $S$ is self-evident, and the **property of the orthogonal $R$ can be proved by**:

$$
R^T R = \begin{bmatrix}\vec{v_1} & \vec{v_2} \end{bmatrix}^T \begin{bmatrix}\vec{v_1} & \vec{v_2} \end{bmatrix}  = \begin{bmatrix}\vec{v_1}^T \\\ \vec{v_2}^T \end{bmatrix} \begin{bmatrix}\vec{v_1} & \vec{v_2} \end{bmatrix} = \begin{bmatrix}\vec{v_1}^T\vec{v_1} & \vec{v_1}^T \vec{v_2} \\\ \vec{v_2}^T \vec{v_1} & \vec{v_2}^T \vec{v_2}\end{bmatrix} = \begin{bmatrix}1 & 0 \\\ 0 & 1\end{bmatrix} = I
$$

$$
(R^T R) R^{-1} = R^T (RR^{-1}) = I R^{-1} = R^T I = R^T = R^{-1}
$$

The covariance matrix $\Sigma^\prime$ can be represented as:

$$
\Sigma^\prime = VLV^{-1} = V \sqrt{L} \sqrt{L} V^{-1} = RSSR^{-1} = RSS^TR^T = RS(SR)^T = TT^T
$$

The key insight is that the transformation $RS$ applied to the original dataset $D$ and the representation $RS(SR)^T$ of the covariance matrix are equivalent. This is because **the covariance matrix measures how data points vary with respect to each other, and this variation is captured by the combined transformation $RS$**. 

It is important to know that the transformation matrix $T$ here is named as the **Cholesky factor**. It is usually written as $\Sigma^\prime = TT^T$. Geometrically, the Cholesky matrix transforms uncorrelated variables into variables whose variances and covariances are given by $\Sigma^\prime$. You can go the other way by taking the inversed matrix $T^{-1}$ to transform the correlated variables into uncorrelated ones. Thus, to decorate the variables and standardize the distribution, we can apply $D = T^{-1}(D^\prime- \mu^\prime)$ to the correlated dataset. 

#### Euclidean distance and linear transformation
Consider a circle centered at the origin with a radius of 1. The Euclidean distance (ED) of every data point on the circle to the origin is 1, which can be expressed as 

$$
X^TX = (IX)^TIX = X^TI^TIX = 1
$$

where $X^T = \[x_1, x_2\]$. Now, for a circle with its center at $\mu^T = \[\mu_1, \mu_2\]$, the ED is represented as

$$
(X-\mu)^T(X-\mu) = (X-\mu)^TI^TI(X-\mu)
$$

If we apply a specific transformation matrix $L$ to the original space of the circle, which involves scaling its radius and rotation, the ED of a data point in the new space is given by: 

$$
(X-\mu)^TL^TL(X-\mu)
$$

This transformation causes the original circle to become an ellipse, the same as shown in the above figure. 

Thus, consider a tilted ellipse after transformation which represents the correlated dataset $Y$, we can decorrelate and standardize it by applying the Cholesky factor as $X = L^{-1}(Y-\mu)$, the ED of $X$ then can be calculated as:

$$
X^TX = [L^{-1}(Y-\mu)]^T L^{-1}(Y-\mu) = (Y-\mu)^T \left(L^{-1}\right)^T L^{-1}(Y-\mu) \tag{1}
$$

Here, we will use several matrix identities such as $(AB)^T = B^TA^T$, $(AB)^{-1} = B^{-1}A^{-1}$, and $\left(A^T\right)^{-1} = \left(A^{-1}\right)^T$. The relationship between inverse and transpose can be proved by:

$$
\left(L^{-1}\right)^T = \left(L^{-1}\right)^T \left(L^T \left(L^T\right)^{-1}\right) = \left(\left(L^{-1}\right)^T L^T\right) \left(L^T\right)^{-1} = \left(LL^{-1}\right)^T \left(L^T\right)^{-1} = I\left(L^T\right)^{-1} = \left(L^T\right)^{-1}
$$

Thus, we have $\left(L^{-1}\right)^T L^{-1} = \left(L^T\right)^{-1} L^{-1} = \left(LL^T\right)^{-1}$ continue the above calculation of Eq.(1) by:

$$
X^TX = (Y-\mu)^T\left(LL^T\right)^{-1}(Y-\mu) = (Y-\mu)^T \Sigma_Y^{-1}(Y-\mu)
$$

This is equation $X^TX = (Y-\mu)^T \Sigma_Y^{-1}(Y-\mu)$ is called the **squared Mahalanobis distance**. The Mahalanobis distance is a way to measure distances between data points, but it's unique because it considers both the variance (how data spreads out) and the relationships (correlations) between different features.

To simplify it, imagine you want to measure distances between data points. Mahalanobis distance first transforms the data to make it easier to compare by removing any correlations and making all the variables have the same scale. Then, it calculates distances between these transformed data points just like you would with regular distances (like the Euclidean distance).

#### Get back to the financial turbulence index

$$
\text{turbulence}_t=\left(y_t-\mu\right)^{\prime} \Sigma^{-1}\left(y_t-\mu\right) \in \mathbb{R}
$$

where $y_t \in \mathbb{R}^n$ denotes the stock returns for the current time period $t$, $\mu \in \mathbb{R}^n$ denotes the average  of history returns, and $\Sigma \in \mathbb{R}^{n \times n}$ denotes the covariance of historical returns. 

**Please note that the formula used in the paper is corrected, the formula can be varied depending on the dimension of $y_t$ by definition.**
