---
layout: post
title: "The Present and Future of Deep Learning-Dominated Algorithmic Trading"
date: 2025-10-14
categories: [Research, Review]
tags: [algorithmic trading, deep learning, quantitative finance, LSTM, transformer, GNN, high-frequency trading]
math: true
mermaid: true
---

## Algorithmic Trading: When Speed Was Everything

In 2014, Michael Lewis revealed the world of ultra-short-term trading, or high-frequency trading (HFT), through his book 'Flash Boys'. The core was simple: **"Speed is everything."**

The traders in the book poured enormous time and money into problems like fiber optic cable routes, server locations, and priority placement of network switches. All this effort was to shorten trade execution speed by mere milliseconds (ms).

This wasn't just a technology competition—it translated into real economic value. Algorithmic trading systems used by institutional investors and hedge funds were designed to respond to price signals faster than a human eye-blink. For a long time, speed was the only rule of the game for making markets, capturing arbitrage opportunities, and front-running individual investors. The faster, the more money you made.

But it's not 2014 anymore. The market landscape has completely changed over the past few years. Speed is no longer everything. Of course it's still important, but now it's just one piece of a huge puzzle. Another important piece we'll examine today is **'prediction'**.

## The Collapse of Traditional Quant Models

For decades, the traditional quant approach has been largely rule-based. Combining technical indicators like Moving Average (MA), Bollinger Bands (BB), RSI thresholds to design strategies and repeating endless backtesting. These strategies were deterministic, explainable, and above all, fast. But simultaneously **extremely fragile.**

And indeed, over recent years, those limitations have started showing.

As market structures changed over time, past rules no longer worked. Market microstructure became increasingly complex, individual investor order flow became a major factor moving markets, and now sentiment on social media shakes stock prices more strongly than earnings announcements. The price formation process is becoming increasingly nonlinear and high-dimensional.

While someone still wrestles with logistic regression models, someone else is using 12-layer transformer models. And that 'someone else' might be your trading counterparty.

This is precisely the topic we'll cover today: 'How is deep learning changing financial markets?' In this article, based on the review paper *"Deep learning for algorithmic trading: A systematic review of predictive models and optimization strategies"*, we'll examine what models are being used, what problems they're applied to, and where the real bottlenecks lie.

## Why Deep Learning? Complexity, Noise, and Non-Stationarity

Deep learning isn't new to software engineers, but its introduction to finance is relatively recent. Its adoption in finance wasn't smooth. Decades-old infrastructure was too outdated to run latest AI models, regulators questioned the opacity of 'black box' models. Most importantly, it was difficult for financial teams managing real capital to trust systems whose internal workings they couldn't understand. However, in today's trading environment dominated by noise, speed, and complexity, deep learning architectures provide strong competitive advantage beyond just being viable alternatives. Indeed, deep learning's influence in finance is spreading very rapidly.

This isn't because deep learning models are smarter. It's because they have excellent **scalability**. Deep learning can extract meaningful signals directly from noisy raw market data without manually defined features. It doesn't care whether data follows normal distribution or variables are independent. It handles sequence data, non-stationarity, and complex joint distributions between variables in ways dimensionally different from existing models.

Of course, deep learning doesn't magically solve everything. Used incorrectly, it produces terrible results. But when the right architecture combines with sufficient data, deep learning identifies patterns that rule-based models could never discover.

Today's paper digs precisely into this point. The authors pose three core research questions:

1. **RQ1**: How are deep learning algorithms currently applied in trading?
2. **RQ2**: What are the limitations of deep learning models?
3. **RQ3**: What are the most promising development directions ahead?

## Deep Learning Models for Algorithmic Trading

The paper classifies currently used deep learning models into 7 major architectures. Each model has its own strengths and weaknesses and is utilized differently for specific prediction and trading tasks.

<div align="center">
    <img src="/assets/images/paper_review/overview_different_AI_models_for_algorithmic_trading.jpg" width="800" alt="Overview of different AI models for Algorithmic Trading">
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Overview of different AI models for Algorithmic Trading
    </a>
</div>

### 1. RNN (Recurrent Neural Networks)

<div align="center">
    <img src="/assets/images/paper_review/architecture_RNN.jpg" width="600" alt="Architecture of RNN">
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of RNN
    </a>
</div>

RNN is the most traditional deep learning approach for time series forecasting. It implements 'memory' through a recurrent structure that uses the output of previous timesteps as input to the model again. In principle, it can capture temporal dependencies in financial time series data, but in practice it's vulnerable to **long-term dependency problems**. Due to vanishing gradient problems, performance sharply degrades in volatile markets, and it easily overfits on noisy data.

The RNN hidden state $h_t$ is updated through previous hidden state $h_{t-1}$ and current input $x_t$ as follows:

$$
h_{t}=\sigma(W_{h}h_{t-1}+W_{x}x_{t}+b_{h})
$$

- $h_t$: Hidden state at time $t$
- $x_t$: Input at time $t$
- $W_h$, $W_x$: Weight matrices for hidden state and input respectively
- $b_h$: Bias term
- $\sigma$: Nonlinear activation function such as tanh or ReLU

### 2. LSTM (Long Short-Term Memory)

<div align="center">
    <img src="/assets/images/paper_review/architecture_LSTM.jpg" width="800" alt="Architecture of LSTM">
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of LSTM
    </a>
</div>

LSTM emerged to solve RNN's long-term dependency problem. By introducing three gates—**input, forget, output**—it learns what information to remember and what to discard.

Expressed in formulas, each gate and cell state are updated as follows:

$$
f_{t}=\sigma(W_{f}[h_{t-1},x_{t}]+b_{f}) \quad (\text{Forget Gate})
$$

$$
i_{t}=\sigma(W_{i}[h_{t-1},x_{t}]+b_{t}) \quad (\text{Input Gate})
$$

$$
C_{t}=f_{t}\cdot C_{t-1}+i_{t}\cdot \tanh(W_{C}[h_{t-1},x_{t}]+b_{C}) \quad (\text{Cell State})
$$

$$
o_{t}=\sigma(W_{o}[h_{t-1},x_{t}]+b_{o}) \quad (\text{Output Gate})
$$

$$
h_{t}=o_{t}\cdot \tanh(C_{t}) \quad (\text{Hidden State})
$$

- $f_t$, $i_t$, $o_t$: Forget, input, output gates respectively
- $C_t$: Cell state, responsible for long-term memory
- $h_t$: Hidden state, used for short-term memory and output
- $W$, $b$: Weight matrices and biases for each gate

Research shows LSTM overwhelmed existing statistical models and simple ML models (SMA, EMA, etc.) in short-term prediction, but performance tended to degrade as prediction horizon lengthened. To compensate, hybrid architectures combining traditional econometric models like CNN-LSTM, BiLSTM-Attention, even LSTM-GARCH showed good results.

### 3. CNN (Convolutional Neural Networks)

<div align="center">
    <img src="/assets/images/paper_review/architecture_CNN.jpg" width="800" alt="Architecture of CNN">
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of CNN
    </a>
</div>

CNN, mainly used for image processing, primarily plays a supportive role in trading. It treats 1D time series data like images, extracts features from price signals or various market indicators, then passes them to sequence models like LSTM.

The convolution operation of filter $w$ on 1D input $x$ is defined as:

$$
(x * w)(t) = \sum_{i=0}^{k-1} x_{t+i} w_i
$$

- $t$: Time step
- $k$: Kernel (filter) size
- $w_i$: Filter weights

While some research attempted to automate technical analysis by inputting price charts directly to CNN, more influential studies mostly used CNN as part of hybrid models.

### 4. Autoencoders (AEs & VAEs)

<div align="center">
    <img src="/assets/images/paper_review/architecture_autoencoders.jpg" width="600" alt="Architecture of Autoencoders">
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of Autoencoders
    </a>
</div>

Autoencoders (standard and variational) are mainly used for **anomaly detection and dimensionality reduction**. For example, detecting abnormal trading activity, removing noise from price signals, and compressing multi-asset data into more manageable latent spaces.

Autoencoders consist of an encoder that transforms input data $x$ into latent representation $z$, and a decoder that reconstructs $x$ from $z$:

$$
z=\sigma(W_{e}x+b_{e}) \quad (\text{Encoder})
$$

$$
\hat{x}=\sigma(W_{d}z+b_{d}) \quad (\text{Decoder})
$$

- $x$: Original input data
- $z$: Low-dimensional compressed latent representation
- $\hat{x}$: Reconstructed output data
- $W_e$, $W_d$: Weight matrices for encoder and decoder respectively
- $b_e$, $b_d$: Biases for encoder and decoder respectively

Particularly, Variational Autoencoders (VAE) are used to generate synthetic data for stress testing. VAE is a generative model that assumes probability distribution in latent space, with the encoder outputting parameters of posterior probability distribution as follows:

<div align="center">
    <img src="/assets/images/paper_review/architecture_variational_autoencoders_VAE.jpg" width="600" alt="Architecture of Variational Autoencoders (VAE)">
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of Variational Autoencoders (VAE)
    </a>
</div>

$$
q_{\phi}(z|x) = \mathcal{N}(z|\mu_{\phi}(x), \sigma_{\phi}(x))
$$

Where each term means:

- $q_{\phi}(z\|x)$: Variational approximation to posterior probability distribution
- $\mu_{\phi}(x)$: Mean of distribution
- $\sigma_{\phi}(x)$: Standard deviation of distribution

### 5. GNN (Graph Neural Networks)

GNN is a cutting-edge approach that models markets as huge graphs. Individual stocks or traders are represented as nodes, and correlations or capital flows between them as edges. Through this, GNN was effective in detecting market manipulation from high-frequency trading data or predicting price movements based on interdependencies among multiple assets.

GNN's general message-passing operation is as follows:

$$
h_{i}^{(k+1)}=\sigma\left[W^{(k)}\cdot h_{i}^{(k)}+\sum_{j\in N(i)}\frac{1}{c_{ij}}\cdot W^{(k)}\cdot h_{j}^{(k)}\right]
$$

- $h_{i}^{(k)}$: Hidden state of node $i$ at layer $k$
- $\mathcal{N}(i)$: Set of neighbor nodes of node $i$
- $W^{(k)}$: Learnable weight matrix at layer $k$
- $c_{ij}$: Normalization constant

However, performance greatly depends on how the graph is constructed, and computational cost is expensive.

### 6. Transformers

<div align="center">
    <img src="/assets/images/paper_review/architecture_transformer.jpg" width="600" alt="Architecture of Transformer">
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of Transformer
    </a>
</div>

Transformers are the most prominent latest models in this field. Thanks to the core **Attention Mechanism**, they're highly suitable for extracting long-term patterns from noisy financial data. Instead of processing data sequentially like RNN or LSTM, they process in parallel and dynamically recalculate the importance of input values.

Self-Attention score is calculated as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q$, $K$, $V$: Queries, Keys, Values matrices respectively, derived from input sequence
- $d_k$: Dimension of key vectors, scaling factor for gradient stabilization

Recent papers showed excellent performance in high-frequency limit order book data or multi-asset volatility prediction, but clear limitations exist: requiring enormous amounts of data and long training times, with high risk of overfitting to noise.

### 7. Reinforcement Learning (RL)

Reinforcement learning learns optimal trading strategies that maximize cumulative reward (usually return or Sharpe ratio) through interaction with simulated environments. Recently, Deep RL combining LSTM or CNN is mainly used.

RL's learning process is determined by the Bellman equation representing the value of current state ($s_t$) and action ($a_t$):

$$
Q(s_{t},a_{t})=r_{t}+\gamma\max_{a'} Q(s_{t+1},a^{\prime})
$$

- $Q(s_t, a_t)$: Expected sum of future rewards when taking action $a_t$ in state $s_t$ (Q-value)
- $r_t$: Immediate reward for action $a_t$
- $\gamma$: Discount factor for future rewards

Models like DQN (Deep Q-Networks) showed performance exceeding existing strategies by simultaneously considering technical indicators and sentiment data, but have problems of low learning efficiency and vulnerability to unpredictable market conditions.

## Key Pattern: Hybrids Stronger Than Pure Models

The pattern throughout the paper is clear: **Hybrid models almost always outperform pure models.** Especially architectures combining financial **domain knowledge** like market microstructure, volatility clustering, behavioral psychology with deep learning's flexibility achieved the best results. Pure black box models still struggle in extreme situations, and models mixing statistical foundations with deep learning are showing better outcomes.

## Remaining Real-World Challenges

Of course, there aren't only advantages. Deep learning models also carry numerous problems:

- **Data Quality**: Deep learning can directly handle noisy raw data, but financial data noise exceeds those limits. Market structure keeps changing so past distributions don't hold, and there's far more noise than signal, making models easily overfit to meaningless patterns. Ultimately, the learning process becomes unstable and predictive power easily loses consistency.
- **Overfitting**: Training complex models on market data with low signal-to-noise ratio risks excessive optimization to historical data.
- **Interpretability**: The "black box" nature makes it difficult to understand why models made certain decisions. This is a major obstacle in the strictly regulated financial industry.
- **Computational Complexity and Latency**: Deep learning models are huge and slow. In trading, fast inference speed is a basic requirement, not a luxury. In a world where everyone is a 'flash boy', if your model is too slow, you'll be left in the dust. Without GPU acceleration, batch processing, and careful pipeline orchestration, meeting latency budgets is difficult.

This is **an operational war** beyond simple modeling problems. Not just getting predictions right, but knowing when and why models fail, and being able to replace them quickly without causing cascading problems.

## Conclusion: From Trader to System Engineer

These tools definitely work. But they force us to think like **system engineers** rather than traders. Not just choosing a model, but choosing a failure surface area, calibration strategy, and infrastructure commitment.

As deep learning spreads, the competition paradigm will shift from 'who has the biggest model' to 'who adapts fastest when data changes'. The center of complexity has moved beyond hand-crafted rules to tradeoffs between architecture, operations, and governance. And making those tradeoffs carefully has now become the responsibility of software engineers and quants.

---

### References

Bhuiyan, M. S. M., Rafi, M. A., Rodrigues, G. N., Mir, M. N. H., Ishraq, A., Mridha, M. F., & Shin, J. (2025). Deep learning for algorithmic trading: A systematic review of predictive models and optimization strategies. *Array, 26*, 100390. [https://doi.org/10.1016/j.array.2025.100390](https://doi.org/10.1016/j.array.2025.100390)
