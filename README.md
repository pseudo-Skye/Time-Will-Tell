# Time-Matters: A collection of tutorials covering the latest research in time series.
Before we start, let's read the following paragraph: 

*"Within the context of domesticated pets, I harbor a distinct preference for the amiable and occasionally mysterious feline companions. The charming attributes inherent in these four-legged creatures evoke a deep sense of admiration and companionship within the depths of my personal preferences."*

Then answer: 

*How much brainy RAM do you need to process this paragraph?*🧠💾

Well, this paragraph is just a cool way of saying, *"I'm a cat person!"* 🐱 

So...

-----------------
****Skip the repo intro and jump to the freshest updates by the [GATEWAY](#whats-new)***

Welcome to the ultimate time series research paper survival guide! Ever get a headache from reading research papers? Well, here's the free Panadol without side effects. I'm here to make the journey smoother for new readers. Originally, it was just my way of saving time for a quick literature review (then I could spend an hour more in the Zelda kingdom)

You see, many papers (including mine) love to flex their academic muscles, tossing around professional language to impress the scholarly gods. It's the nature of research – all about rigor and seriousness. But the silver lining? they all follow a familiar routine: **problems, challenges, methodology, a sprinkle of baseline comparisons, and voila!** Here in my repo, it's all about simplicity and clarity. Forget the fancy jargon and convoluted sentences! For each paper, I cut through the noise, and break down complex ideas into bite-sized, understandable pieces. With this straightforward roadmap, you might only need **1-2 hours** to grasp both the important ideas and the technical details in each paper. 

Whoops, almost skipped the math part! Math is like running a marathon in a maze, no quick fix here unless you're deep into the nerd zone. But I've got your back with **super detailed explanations** of all those equations, and sometimes I might toss in some figures to spice up the explanation. 

Finally, lay down some NOTES for this repo:

**(1) Update Frequency:** Whenever the mood strikes, aim for at least one research tutorial fortnightly. But if I'm drowning in submission deadlines, updates might take a rain check. 

**(2) Tutorial Content:** My research is mainly about time series anomaly detection, but I also dip into the time series pool for stocks, crimes, and more. Papers get the nod based on importance (including the old-school ones), state-of-art ideas, and their relevance to my current research.

**(3) Reader Interaction:** I'll drop an update note and refresh the content table every time I jazz up this repo with fresh research tutorials. Feel free to ask questions, have discussions, and recommend new papers by creating an issue. 

## Tutorial Overview

<table>
  <tr>
    <th>Category</th>
    <th>Article</th>
    <th>Year</th>
    <th>Source</th>
    <th>Description</th>
  </tr>
  <tr>
    <td rowspan="3">Financial Trading</td>
    <td><a href="https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/A%20Review%20of%20Deep%20Reinforcement%20Learning%20in%20Quantitative%20Algorithmic%20Trading.md">Review: Deep Reinforcement Learning in Quantitative Algorithmic Trading </a> </td>
    <td>2021</td>
    <td>ArXiv</td>
    <td>This article reviews progress made so far of AI in automated low-frequency quantitative stock trading. Focus on trading agents created with deep reinforcement learning. </td>
  </tr>
  <tr>
    <td><a href="https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/Python%20library%20FinRL.md"> FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance  </a></td>
    <td>2021</td>
    <td>ICAIF</td>
    <td>FinRL is a powerful deep RL library designed to make quantitative finance and stock trading accessible to beginners while providing advanced capabilities for experienced practitioners.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/pseudo-Skye/Time-Matters/blob/main/financial%20trading/StockFormer.md"> StockFormer: Learning Hybrid Trading Machines with Predictive Coding  </a></td>
    <td>2023</td>
    <td>IJCAI</td>
    <td>StockFormer is a hybrid trading machine that combines predictive coding and RL techniques to model multiple time series data in stock trading. </td>
  </tr>
  <tr>
    <td rowspan="3">Anomaly Detection</td>
    <td><a href="https://github.com/pseudo-Skye/Time-Matters/blob/main/anomaly%20detection/Anomaly%20in%20TSAD%20Evaluation.md">The 'Anomaly' in TSAD Evaluation: How We’ve Been Getting It Wrong </a></td>
    <td>2022</td>
    <td>AAAI, KDD</td>
    <td>The existing evaluation metrics for time series anomaly detection algorithms have limitations that can lead to misleading results. In this article, we will discuss the limitations of existing evaluation metrics and introduce the new evaluation metrics that address these limitations from the most recent publications. </td>
  </tr>
  <tr>
    <td><a href="https://github.com/pseudo-Skye/Time-Matters/blob/main/anomaly%20detection/TFAD%20(CIKM%2022).md"> TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Frequency Analysis  </a></td>
    <td>2022</td>
    <td>CIKM</td>
    <td>This work uses both time and frequency domains to improve performance, and it also use techniques like data augmentation and time series decomposition to make the model work better.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/pseudo-Skye/Time-Matters/blob/main/anomaly%20detection/MERLIN%20(ICDM%2020).md"> MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives  </a></td>
    <td>2020</td>
    <td>ICDM</td>
    <td>The article highlights time series discords as a popular method for practitioners due to its simplicity. It introduces MERLIN, an algorithm that can efficiently and exactly find discords of all lengths in massive time series archives. </td>
  </tr>
  <tr>
    <td rowspan="4">Series-ly Mathematical</td>
    <td><a href="https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/The%20Ultimate%20Guide%20to%20Fourier%20Transform.md">The Ultimate Guide to Fourier Transform: Learn from Scratch to Pro  </a></td>
    <td>2023</td>
    <td>Homemade</td>
    <td>In this tutorial, I will be covering all the fundamental content of Fourier Series, including all the mathematical proofs you need to understand how Fourier Transform works. </td>
  </tr>
  <tr>
    <td><a href="https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Mahalanobis%20distance.md"> Mahalanobis distance in finance</td>
    <td>2023  </a></td>
    <td>Homemade</td>
    <td>The measure of the financial turbulence index is based on the Mahalanobis distance. Check this out. </td>
  </tr>
  <tr>
    <td><a href="https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/VAE.md"> Decoding Variational Autoencoders: Exploring the Mathematical Foundations </a></td>
    <td>2023</td>
    <td>Homemade</td>
    <td>This article introduces the fundamental concepts of variational autoencoder (VAE) from a probability model perspective. </td>
  </tr>
  <tr>
    <td><a href="https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Score%20matching.md"> Decoding the Logic of Score Matching: An Energy-Based Solution </a></td>
    <td>2024</td>
    <td>Homemade</td>
    <td>This article introduces score matching and its interaction with energy-based functions from a purely mathematical perspective. Additionally, we will also explore an important technique called denoising score matching, which can be used for time series denoising.</td>
  </tr>
</table>

## What's New
[12/01/2024] "[Decoding the Logic of Score Matching: An Energy-Based Solution](https://github.com/pseudo-Skye/Series-ly-Mathematical/blob/main/Score%20matching.md)": A new tutorial about score matching, energy function, and **denoising score matching** (denoising autoencoder based). 

[05/01/2024] In 2024, may your research flourish, ideas blossom, and every paper penned be a step toward brilliance and success.








