# FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
FinRL is a powerful Deep Reinforcement Learning (DRL) library designed to make quantitative finance and stock trading accessible to beginners while providing advanced capabilities for experienced practitioners. It offers a wide range of features and tools to facilitate the development of stock trading strategies and provides a platform for learning and experimentation in the field of finance. The trading environments are based on the OpenAI Gym framework. ([paper](https://arxiv.org/abs/2011.09607), [code](https://github.com/AI4Finance-Foundation/FinRL)) 

You may also need to check this useful [blog](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Mahalanobis%20distance.md) with a very detailed explanation of **Mahalanobis distance**, which will be used as an important **financial turbulence index** in FinRL. 

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
5. Risk control: employ a **[financial turbulence index](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Mahalanobis%20distance.md)** that measures extreme price fluctuations. (**Please note that the formula of financial turbulence index used in the paper is corrected in my blog, the formula can be varied depending on the dimension of $y_t$ by definition.**)
