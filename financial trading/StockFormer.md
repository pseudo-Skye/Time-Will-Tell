# StockFormer: Learning Hybrid Trading Machines with Predictive Coding
In this paper, StockFormer is introduced, a novel hybrid trading machine that combines **predictive coding** and RL techniques. StockFormer is designed to model **multiple time series data**, such as the trading records of different stocks, which are evolving simultaneously over time. It employs **three Transformer branches** to capture latent states of **temporal dynamics** (short- and long-term) and **asset relations**. These states are fused by an RL agent using an actor-critic algorithm in a unified state space. Notably, the model is jointly trained by propagating the critic's gradients back to the predictive coding module, leading to superior performance in terms of **portfolio returns** and **Sharpe ratios**. ([paper](https://www.ijcai.org/proceedings/2023/0530.pdf), [code](https://github.com/gsyyysg/StockFormer))

## Fundamentals
Feel free to bypass this section if you're already well-versed in the following topics. As the proposed method is transformer-based, first you can look up this [online resource](http://jalammar.github.io/illustrated-transformer/) to understand the self-attention and transformers. 
 
### Understanding the discount factor in RL
Imagine you have a decision maker (like a robot) that can be in different situations (states) $s_j \in \mathcal{S}$ and can take different actions $a_i \in \mathcal{A}$. When this decision maker is in a particular situation, it has a choice of actions to take.

Now, there's something called a "policy" $\pi$, $\pi(\cdot): \mathcal{S} \rightarrow \mathcal{A}$, which is like a rulebook. It tells the decision maker which action to pick in each situation. When the decision maker follows this policy and takes an action in a certain situation, it gets a reward $R_{a_i}\left(s_j, s_k\right)$, which means the environment changes from $s_j$ to state $s_k$ after the decision maker takes action $a_i$. The reward tells us how good or bad that action was.

The objective is to find a policy $\pi$ such that

$$
\max_{\pi: S(n) \rightarrow a_i} \lim_{T \rightarrow \infty} E\left\[\sum_{n=0}^T \beta^n R_{a_i}\left(S(n), S(n+1)\right)\right\](1)
$$

Where $\beta$ is a discount factor. The discount factor determines how much an agent cares about rewards in the distant future relative to those in the immediate future. Discounting future rewards makes sense because it reflects the fact that future rewards are less certain than immediate rewards. The equation above is usually called an MDP problem with an **infinite horizon discounted reward criteria**.

If we set $\beta \rightarrow 0$, then only immediate rewards will be considered. If we set $\beta = 1$, then all future rewards will be considered equally important as immediate rewards, and the equation is called **infinite horizon sum reward criteria**, which is not a good optimization as the sum would not converge. In practice, we use a value between 0 and 1 to balance immediate and future rewards. This good [example]( https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning) can be found to explain the necessity of setting $\beta$ between 0 and 1.

### Partially observable Markov decision process (POMDP) in trading 
In trading, you often have to make decisions without knowing the true state of the market. POMDPs allow you to work with partial information and make informed decisions based on what you can observe.
* **Challenge**: One big challenge in using reinforcement learning for finance is dealing with the messy, real-world data from financial markets. We want to extract the important stuff from this noisy data to make good decisions.
* **Goal**: Imagine we have a certain number of steps (let's say 1,000 days) to make money in the stock market. Our goal is to learn the best way to invest during these steps and make as much money as possible.
* **The Components in POMDP**:
    - **Observation Space (O)**: This is like the information we can see. It includes things like historical stock prices, technical indicators (like moving averages), and how different stocks move together (their covariance).
    - **State Space (S)**: It's made up of different hidden states that describe what's going on in the market. Plus, there's a state that tells us how much money we have and how many shares of each stock we own.
    - **Action Space (A)**: This is what we can do. We can choose to buy more shares, sell the ones we have, or hold onto them. (**continuous** action space)
    - **Reward Function (R)**: At the end of each day, we get a reward based on how our investments are doing. 

### Discussion about states and observations in trading
* **State**: State in trading refers to the internal representation of the underlying conditions or factors that affect the financial market at a given moment. The "state" is not directly observable, and it often represents latent or hidden variables that influence market dynamics.
* **Observation**: Observation in trading refers to the data or information that traders have access to or can directly observe. These are the measurable data points that traders use to gain insights into the market, like historical stock prices, trading volumes, and technical indicators.
  
    In this paper, the total account balance, as well as the number of shares you own for each stock, can be treated as part of the state space rather than an observation. While they are directly observable, they are often considered part of the state space because they provide essential information about your financial position and portfolio composition, and they can have a significant impact on your future trading actions.


## Problems of the existing methods
### RL-based methods
The paper mentions that a common approach is to treat this problem as a Markov decision process (MDP) and apply RL algorithms to the observed data (e.g., stock prices, and trading volumes). The concern raised here is that these approaches assume that the observed data alone can accurately represent **(1)** the complex correlation between stocks and **(2)** the ever-changing dynamics in financial markets.

* **Temporal Dynamics**: RL models, especially simple model-free RL (the algorithms adjust their strategies based on the data they observe), often focus on immediate rewards and do not inherently incorporate long-term temporal dynamics. They may not effectively capture trends or patterns that unfold over extended time periods.

* **Stock Correlations**: RL models typically consider individual states in the environment without explicitly modeling or capturing the complex relationships and correlations between different stocks in a portfolio. They might not effectively account for how events affecting one stock can influence others.

### Stock price prediction-based methods
They use fixed trading rules (e.g., buy and hold), and the focus is on long-term investment rather than frequent trading. 

## How StockFormer addresses these problems
1. **Three Transformer-Like Networks**: they are designed to learn different types of information from the market data: long-term trends, short-term trends, and how different assets (like stocks) relate to each other.
2. **Multi-Head Feed-Forward Networks**: To handle multiple pieces of information from various assets happening at the same time, StockFormer employs multi-head feed-forward networks in the attention block. This helps the model understand and learn diverse patterns across different assets. For example:
   - **Temporal Patterns**: Stocks may exhibit different temporal patterns, such as daily, weekly, or seasonal trends. Each head could focus on learning a different aspect of these temporal patterns.
   - **Dependencies**: The relationships between stocks can be nonlinear. Some heads may focus on capturing linear correlations, while others may capture more complex, nonlinear dependencies.
   - **Sector Relationships**: Stocks within the same sector (e.g., technology, healthcare) may exhibit sector-specific relationships, and different heads can specialize in understanding these sector-related dependencies.
3. **Policy Optimization**: The actor-critic method is used, which helps make decisions based on evaluations of actions. Gradients related to the critic's assessment of actions are used. These gradients are propogated to the relational state encoder, which is where information about how different assets relate to each other is stored.

## The new version of the transformer in StockFormer
![image](https://github.com/pseudo-Skye/StudyNotes/assets/117964124/a6b16fda-bd9c-4996-ac45-41a71417e8c4)
**Diversified multi-head attention (DMH-Attn) block**: Instead of using the single FFN as the original transformer, this paper uses **individual FFNs for each head** to better capture the diverse patterns and relationships within financial data.

**The reason for adding Query in the residual block**: The query in a transformer-based model typically represents the **task-specific information**. It contains information about what the model is looking for or what it's trying to predict. In this way, the model ensures that task-specific information is preserved and not diluted by the contextual information provided by the attention mechanism.

## The architecture of StockFormer
![image](https://github.com/pseudo-Skye/StudyNotes/assets/117964124/4f4707c6-d08a-49cd-8538-9afafc1491e2)

### Relation inference module
This module is DMH-Attn based. The input is the technical indicators of $N$ stocks (**partially masked**), and the covariance matrix of the stocks. The purpose of this module is to **reconstruct the masked statistics** based on the dynamic relation.  The loss is given by the distance between the predicted statistics and ground truth statistics, which indicates the performance of reconstruction. 
### Future prediction module
This module is also DMH-Attn-based. The input is the price records of the stocks and the decoder tries to predict the possible return ratio for both short-term and long-term. The loss is given by **(1)** the distance between the predicted ratio and true ratio, and **(2)** whether the predicted ranking maintains the same order as the ground truth. 

For example, if stock A has a better return than stock B, we want **(1)** the predicted return ratio to be close to the true return of both A and B respectively, and **(2)** our predicted return of A should be larger than B as the ground truth suggests. 
### Latent state integration
This process is also multi-head attention-based. From the previous steps, we will have the latent state that indicates the dynamic correlation, and the short-term and long-term temporal dynamics. The focused states will be concatenated with **the shares of all trading targets that we hold**, and feed to the following RL system. 
### Joint learning with RL
Soft actor-critic (SAC) is used in this work as the RL module for trading. The joint training is used to propagate the gradients back to the relation inference module. 

## Experiments
### Baseline used for performance comparison
1. **Market benchmarks**: CSI-300 and NASDAQ Index
2. **Basic trading strategy**: Min-variance portfolio allocation strategy
3. **Stock prediction-based model**: Use “buy and hold” – buy one stock each day that has the highest estimated return in the next $H$ days, and then sell it $H$ days later.
4. **RL-based model**: the RL baseline models from the [FinRL library]( https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/Python%20library%20FinRL.md). 

### Evaluation metrics
Portfolio return (PR), annual return (AR), Sharpe ratio (SR), and maximum drawdown (MDD). The introduction of different metrics used in financial trading can be found [here]( https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/Python%20library%20FinRL.md#evaluation-metrics).

_*Transaction cost is included in the experiments._

### Interesting experimental results
* Among these three datasets, almost all RL-based methods outperform the rest including the prediction-based models.
* The StockFormer outperforms all the baseline models by a large margin as it benefits from policy optimization over the extracted relational and predictive states.
* With the tough Cryptocurrency dataset, most prediction-based models run into trouble. They try too hard to predict how things change over time, but this dataset is just too wild and unpredictable. However, even when the prediction part of StockFormer struggles in this situation, it manages to do well when it teams up with predictive coding states in the RL system.

