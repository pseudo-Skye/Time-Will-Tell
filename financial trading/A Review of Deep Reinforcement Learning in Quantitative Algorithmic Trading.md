# Review: Deep Reinforcement Learning in Quantitative Algorithmic Trading
This article is a summarized version of the paper: [Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review](https://arxiv.org/abs/2106.00123). This paper reviews progress made so far of AI in automated **low-frequency** quantitative stock trading. It narrows its focus to trading agents created with DRL and does not discuss attempts to predict stock prices because it believes it's more effective to manage risks rather than trying to predict the unpredictable nature of stock markets.

## Two questions the paper can answer
**Q1: Can we create a program to make trading decisions like a human trader does, but with the help of advanced machine learning techniques like DRL?**

The paper concluded that DRL has huge potential in being applied in algorithmic trading from **minute to daily** timeframes, with several prototype systems being developed showing great success in particular markets. 

**Q2: can DRL lead to a super-human agent that can beat the market and outperform professional human traders?**

The paper inferred that DRL systems can compete with professional traders with respect to the risk-adjusted return rates on short (15-minute) or long (daily) timeframes in particular hand-picked markets, but more research needs to be done. 

## Key challenges that DRL faces when applied to financial trading
1. **Data Quality and Availability**: In the financial world, high-quality data is often not free, and sometimes, it may be limited, especially for certain timeframes (like very short-term or long-term data). 
2. **Partial Observability of the Environment**: Financial markets are complex, and not everything happening in the market is visible or easy to observe. There's always some level of uncertainty or hidden information in the financial environment. For example, a sudden surge in positive tweets about a particular stock can impact its price, but this information might not be readily available or quantifiable.
3. **Exploration/Exploitation Dilemma**: This dilemma is about finding the right balance between exploring new strategies and exploiting known ones. In reinforcement learning, exploration is like trying out new things to see if they work better, while exploitation is sticking with what's known to be effective. However, in financial trading, excessive exploration can be costly because it involves making many transactions, which can result in high transaction costs and the loss of capital.

## Fundamentals of the financial trading 
Before we delve into the algorithms used in financial trading, it's recommended to grasp several key concepts.
- **Low-frequency trading**: trade from 1 minute to a few days time frame. 
- **Volatility**: A statistical measure of the dispersion of returns for a given security or market index. In most cases, the higher the volatility, the riskier the security. Volatility is often measured from either the standard deviation or variance between returns from that same security or market index. Negative volatility, or when your investments are losing money, is something investors and traders want to avoid. The SR helps by emphasizing the risk associated with these negative returns.
- **Sortino ratio (SR)**: It is calculated by dividing the difference between the portfolio’s expected return and the risk-free rate by the standard deviation of negative returns, also known as downside deviation. The SR is useful for investors who are more concerned about potential losses than overall volatility.
- **Mean-reverting**: Some assets tend to follow a pattern where, after moving away from their average value, they tend to revert or return to that average value over time.

## Literature review
The paper reviews the existing work from three different types of DRL, the critic, actor, and critic-actor based model in quantitative trading. 

### Critic-only approach (DQN)
Think of the "critic" as the evaluator. It's like a judge that tells you how good or bad your actions are in a certain situation. In reinforcement learning, the critic uses an action-value function (Q-function) to estimate the expected reward you can get by taking a specific action in a given situation. The Q function can be approximated successfully by DQN. (Most published method)

#### Important related works of critic-only DRL
1. **Adaptive stock trading strategies with deep reinforcement learning methods** (2020)
    - **Contributions**:
      - Use GRU to extract temporal features.
      - Use a new reward function to handle the risk: the Sortino ratio (SR)
    - **Limitations**: 
        - Only 1 share per transaction
        - Transaction costs were not added to the equation

2. **Deep Reinforcement Learning for Trading** (2020, future contracts)

    The paper uses LSTM to build a Q-network. This network helps make decisions about when to buy and sell these futures contracts. They incorporate volatility into their strategy, recommending larger trades in stable markets and smaller trades in volatile markets to manage risk. This work will be further discussed in [critic-actor](#critic-actor-approach) based model.

#### Limitations of the critic-only approach
1. **Discrete vs. Continuous Actions**: The critic method, like DQN, is designed for solving problems with discrete actions. However, in financial trading, actions like buying and selling are continuous (a variable quantity of an asset, i.e., buy 10.5 shares) and require special techniques to adapt. 
2. **Continuous State Representation**: Stock prices are continuous values, and handling multiple stocks and assets together in a reinforcement learning model can lead to complex and challenging state and action spaces.
3. **Sensitive to Reward Function**: Small changes in how rewards are defined can significantly affect the learning process and outcomes.

### Actor-only approach (PG)
The "actor" is like the decision-maker. It's the part that figures out what action to take based on the current situation. Instead of just guessing, the actor tries to make the best decision by considering the probability of each action leading to a high reward.

#### Important related works of actor-only DRL:
**Deep Direct Reinforcement Learning for Financial Signal Representation and Trading** (2016)
- **Contributions**:
  - The first use of DL for real-time financial trading.
  - Remodel the RNN for simultaneous environment sensing.
  - Use DL for feature extraction combined with fuzzy learning.
  - Use task-aware BPTT to handle the gradient vanishing.
- **Limitations**:
  - Lack of experiments comparison between DL feature extraction on raw data and technical indicators. (Another study proves that using DL (i.e., LSTM) is better compared to dense one, and the right technical indicators matter in high return)
  - No comparison to human trading experts’ performance.
  - Obscure evaluation metrics.

#### Limitations of the actor-only approach
Takes longer to train in case bad action is considered good as total rewards are good.
#### Advantages of the actor-only approach
The action space can be continuous and learn a direct mapping of what to do in the particular state.

### Critic-actor approach 
In actor-critic reinforcement learning, you have both the decision-maker (actor) and the evaluator (critic) working together. The actor suggests actions based on probabilities, and the critic evaluates those actions by estimating the expected rewards. 
#### Important related works of actor-critic DRL
1. **Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy**. (2020)
**This research team has many related works in DRL-based trading.** It employs the financial turbulence index to measure the impact of market crashes and extreme instability. However, a good trading agent is believed to make huge profits through high volatility. 
2. **Stock Trading Bot Using Deep Reinforcement Learning** (2019)
This work combines the DRL with **sentiment analysis**. They use RCNN to classify news sentiment via news headlines. Also, it compares 3 different rewards and finds the agent managed to learn via if the action was profitable or not. 
3. **Deep reinforcement learning for trading**. (2020)
This study directly compares **A2C** algorithm with PG and DQN.
    - **Advantages of A2C**:
      - Because A2C encourages more trading, it might recognize the price increase early and buy and sell the stock multiple times during the upward move. Each time it buys low and sells high, making a profit. So, while the profit from each trade may be smaller, the cumulative profit from all these smaller trades can be more significant.
      - The study also found that A2C is effective in dealing with markets that tend to revert to their average values over time. It may recognize when the price has moved too far from the average and make trades to profit from the expected reversion to the mean.
#### Advantages of the actor-critic approach
They are considered better because they have two helpers: one that learns how to make decisions, and another that checks how good those decisions are. This approach is particularly good at tackling the challenges that come with complex tasks like trading in financial markets, where the data changes a lot, and small changes in strategy can have big consequences.

## Problems of research in AI-based trading
1. Most works had experiments conducted in unrealistic settings, lacking **(1)** testing in real-time, online trading platforms, and **(2)** comparisons between algorithms built on different types of DRL or traders. The research is still in the very early stages of development.
2. Researchers often use various benchmarks to measure the performance of their trading strategies. Comparing the performance of different strategies can be tricky because researchers may use different environments, timelines, and metrics to evaluate them.
3. Research in this area might not be publicly available because it is a profitable field, and private companies might keep their methods secret. As a result, it's hard to know what the state-of-the-art methods are. 

## Direction for future work
To address the challenges in AI-based financial trading, this section will outline the factors that should be taken into account when starting research in this domain. We must consider aspects like selecting an appropriate baseline for comparison, determining the timeframes for training and testing, and deciding whether to incorporate technical indicators.
### Existing problems in training and testing and their solution
1. **Timeframes used in trading**: Many of the research papers being reviewed in the context of trading focus on daily timeframes. Over longer timeframes, such as days and weeks, market movements can be influenced by complex factors like political news, legal developments, public sentiment, social trends, and big trades by institutional investors. These factors are difficult to model or predict using quantitative algorithms.

    **Proposed solution**: When you analyse market activity on a much shorter scale, such as an hour, minute, or even a second, you often see patterns. These patterns are believed to result from algorithmic trading strategies, and they can be exploited by intelligent systems or trading algorithms. (i.e., HFT)

2. Many of the reviewed papers use years as testing intervals to assess the performance of their trading algorithms. However, in reality, deploying a trading algorithm for years without retraining is not advisable. Market conditions change, and the relationships between different assets (covariance) can shift over time.
  
    **Proposed solution**: trading and testing sessions should be of a fixed, relatively short length, such as 2-3 months. This approach allows the algorithm to stay more up-to-date with the market and adapt to changing conditions.

### Recommended baseline model/benchmark for DRL
Turtle trading strategy. The introduction can be found [here](https://www.investopedia.com/articles/trading/08/turtle-trading.asp#:~:text=Turtles%20were%20taught%20very%20specifically,highs%20as%20an%20entry%20signal.).

### Technical indicators used in stock trading
This paper mentions - Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Average Directional Index (ADX), Commodity Channel Index (CCI), On-Balance Volume (OBV) and moving average and exponential moving average. 

The challenges fall in choosing the effective indicators and considering the dependencies between them. The solution is DNN-based for **dimensionality reduction (AE)** and **features extraction (CNN, LSTM)**. 

### Recommended evaluation metrics
Yearly return rate on a certain timeframe, cumulative wealth, Sharpe ratio.

### Recommended further research direction
1. Experiments on live-trading platforms (but the real SOTA ones may be secrets)
2. Direct comparison between DRL with human traders.
3. More comparisons among SOTA DRL under similar conditions and data sources.
4. More research about DRL under critical market conditions (stock market crashes).
5. In case the transaction fees might overcome the actual small gains of relatively high-frequency trading, picking the perfect timescale should be addressed more.
6. Construct a simulated environment to do back testing, and consider connection latencies in the real world.  
