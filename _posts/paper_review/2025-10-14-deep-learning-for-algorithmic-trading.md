---
layout: post 
title: "딥러닝이 지배하는 알고리즘 트레이딩의 현재와 미래"
date: 2025-10-14
categories: [Research, Review]
tags: [algorithmic trading, deep learning, quantitative finance, LSTM, transformer, GNN, high-frequency trading]
math: true
mermaid: true
---

## 알고리즘 트레이딩, 속도가 전부였던 시절

2014년 마이클 루이스는 '플래시 보이즈(Flash Boys)'라는 책을 통해 초단타매매, 즉 고빈도 알고리즘 트레이딩(HFT)의 세계를 세상에 알렸습니다. 핵심은 간단했습니다: **"속도가 전부다."**  

책 속의 트레이더들은 광섬유 케이블의 경로, 서버의 위치, 네트워크 스위치의 우선 배치 같은 문제에 엄청난 시간과 돈을 쏟아붓습니다. 이 모든 노력은 거래 체결 속도를 고작 몇 밀리초(ms) 단축하기 위함이었습니다.  

이것은 단순한 기술 경쟁이 아니라, 실질적인 경제적 가치로 이어졌습니다. 기관 투자자와 헤지펀드가 사용하는 알고리즘 트레이딩 시스템은 인간이 눈을 깜빡이는 것보다 더 빠르게 가격 신호에 반응하도록 설계되었기 때문입니다. 오랫동안 속도는 시장을 만들고, 차익 거래 기회를 포착하며, 개인 투자자들을 앞지르는 유일한 게임의 법칙이었습니다. 더 빠를수록 더 많은 돈을 벌 수 있었죠.  

그러나 지금은 2014년이 아닙니다. 지난 몇 년 사이 시장의 판도가 완전히 달라졌습니다. 속도는 더 이상 전부가 아닙니다. 물론 여전히 중요하지만, 이제는 거대한 퍼즐의 한 조각에 불과합니다. 오늘 살펴볼 또 다른 중요한 조각은 바로 **'예측'**입니다.  

## 전통적 퀀트 모델의 붕괴

수십 년간 전통적인 퀀트의 접근 방식은 대체로 규칙 기반(rule-based) 이었습니다. 이동평균(MA), 볼린저 밴드(BB), RSI 임계값 같은 기술적 지표를 조합해 전략을 설계하고, 끝없는 백테스팅을 반복했습니다. 이러한 전략들은 결정론적이고 설명 가능하며, 무엇보다 빨랐습니다. 하지만 동시에 **무척 취약했습니다.**

그리고 실제로 최근 몇 년 사이, 그 한계가 드러나기 시작했습니다.

시간이 지남에 따라 시장 구조가 변하며 과거의 규칙이 더는 통하지 않게 된 것입니다. 시장 미시구조(microstructure)는 점점 더 복잡해지고, 개인 투자자의 주문 흐름이 시장을 움직이는 주요 요인이 되었으며, 이제는 실적 발표보다 소셜 미디어의 정서(sentiment) 가 주가를 더 강하게 흔듭니다. 가격 형성 과정은 점점 더 비선형적이고 고차원적인 체계로 바뀌고 있습니다.  

누군가가 아직도 로지스틱 회귀 모델을 붙들고 씨름하는 동안, 다른 누군가는 12개 층을 쌓은 트랜스포머 모델을 이용하고 있습니다. 그리고 그 '다른 누군가'는 어쩌면 당신의 거래 상대방일지도 모릅니다.  

이것이 바로 오늘 이 글에서서 다룰 주제, '딥러닝이 어떻게 금융 시장을 바꾸고 있는가'입니다.이번 글에서는 그 변화를 체계적으로 정리한 리뷰 논문 *“Deep learning for algorithmic trading: A systematic review of predictive models and optimization strategies”* 을 바탕으로 현재 어떤 모델들이 사용되고 있는지, 어떤 문제에 적용되고 있는지, 그리고 진짜 병목은 어디에 있는지를 살펴보겠습니다.

## 왜 딥러닝인가? 복잡성, 노이즈, 그리고 비정상성

딥러닝은 소프트웨어 엔지니어들에게 새로운 것이 아니지만, 금융 분야에 도입된 것은 비교적 최근입니다. 딥러닝의 금융권 도입은 순탄치 않았습니다. 수십 년 된 인프라는 최신 AI 모델을 구동하기엔 너무 낡았고, 규제 당국은 '블랙박스' 모델의 불투명성을 문제 삼았습니다. 무엇보다 실제 자본을 운용하는 금융팀이 내부 작동을 알 수 없는 시스템을 신뢰하기란 어려웠기 때문입니다. 하지만 노이즈, 속도, 복잡성이 지배하는 오늘날의 트레이딩 환경에서 딥러닝 아키텍처는 단순히 실행 가능한 대안을 넘어, 강력한 경쟁 우위를 제공합니다. 실제로 금융 분야에서 딥러닝의 영향력은 매우 빠르게 확산되고 있습니다.  

이는 딥러닝 모델이 더 똑똑해서가 아닙니다. **확장성(scale)** 이 뛰어나기 때문입니다. 딥러닝은 사람이 일일이 정의한 특징(feature) 없이도, 노이즈가 섞인 원시 시장 데이터에서 직접 유의미한 신호를 추출할 수 있습니다. 데이터가 정규분포를 따르든, 변수들이 서로 독립적이든 신경 쓰지 않습니다. 시퀀스 데이터, 비정상성(non-stationarity), 그리고 변수 간의 복잡한 결합 분포를 기존 모델과는 차원이 다른 방식으로 처리합니다.

물론 딥러닝이 마법같이 모든 것을 해결하지는 않습니다. 잘못 사용하면 형편없는 결과를 낳을 뿐이죠. 하지만 올바른 아키텍처와 충분한 데이터가 결합될 때, 딥러닝은 규칙 기반 모델이 영원히 알아내지 못할 패턴을 식별해냅니다.

오늘 살펴볼 논문은 바로 이 지점을 파고듭니다. 저자들은 세 가지 핵심 연구 질문을 던집니다:

1. **RQ1**: 딥러닝 알고리즘은 현재 트레이딩에서 어떻게 적용되고 있는가?  
2. **RQ2**: 딥러닝 모델의 한계점은 무엇인가?  
3. **RQ3**: 앞으로 가장 유망한 발전 방향은 무엇인가?  

## 알고리즘 트레이딩을 위한 딥러닝 모델

논문은 현재 사용되는 딥러닝 모델을 크게 7가지 아키텍처로 분류합니다. 각 모델은 저마다의 장단점을 가지며, 특정 예측 및 트레이딩 작업에 맞춰 다르게 활용되고 있습니다.

<div align="center">
    <img src="/assets/images/paper_review/overview_different_AI_models_for_algorithmic_trading.jpg" width="800" alt="Overview of different AI models for Algorithmic Trading">
    <br>
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Overview of different AI models for Algorithmic Trading
    </a>
</div>

### 1. RNN (Recurrent Neural Networks)

<div align="center">
    <img src="/assets/images/paper_review/architecture_RNN.jpg" width="600" alt="Architecture of RNN">
    <br>
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of RNN
    </a>
</div>

RNN은 시계열 예측을 위한 가장 전통적인 딥러닝 접근법입니다. 이전 타임스텝의 출력을 다시 모델의 입력으로 사용하는 순환 구조를 통해 '기억'을 구현합니다. 원칙적으로는 금융 시계열 데이터의 시간적 종속성을 포착할 수 있지만, 실제로는 **장기 의존성(long-term dependencies) 문제**에 취약합니다. 그래디언트 소실(vanishing gradient) 문제 때문에 변동성이 큰 시장에서는 성능이 급격히 저하되고, 노이즈가 많은 데이터에서는 과적합되기 쉽습니다.  

RNN의 은닉 상태(hidden state) $h_t$는 이전 은닉 상태 $h_{t-1}$와 현재 입력 $x_t$를 통해 다음과 같이 업데이트됩니다.

$$
h_{t}=\sigma(W_{h}h_{t-1}+W_{x}x_{t}+b_{h})
$$

- $h_t$: 시간 $t$에서의 은닉 상태  
- $x_t$: 시간 $t$에서의 입력  
- $W_h$, $W_x$: 각각 은닉 상태와 입력에 대한 가중치 행렬  
- $b_h$: 편향(bias) 항  
- $\sigma$: tanh 또는 ReLU와 같은 비선형 활성화 함수  

### 2. LSTM (Long Short-Term Memory)

<div align="center">
    <img src="/assets/images/paper_review/architecture_LSTM.jpg" width="800" alt="Architecture of LSTM">
    <br>
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of LSTM
    </a>
</div>

LSTM은 RNN의 장기 의존성 문제를 해결하기 위해 등장한 모델입니다. **입력, 망각, 출력** 이라는 세 개의 게이트(gate)를 도입하여, 어떤 정보를 기억하고 어떤 정보를 버릴지 학습합니다.

수식으로 표현하면 각 게이트와 셀 상태는 다음과 같이 업데이트됩니다:

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

- $f_t$, $i_t$, $o_t$: 각각 망각, 입력, 출력 게이트  
- $C_t$: 셀 상태(cell state), 장기 기억을 담당  
- $h_t$: 은닉 상태(hidden state), 단기 기억 및 출력에 사용  
- $W$, $b$: 각 게이트에 대한 가중치 행렬과 편향  

연구에 따르면 LSTM은 단기 예측에서는 기존 통계 모델이나 단순 ML 모델(SMA, EMA 등)을 압도했지만, 예측 기간이 길어질수록 성능이 저하되는 경향을 보였습니다. 이를 보완하기 위해 CNN-LSTM, BiLSTM-Attention, 심지어 전통 계량 모델을 결합한 LSTM-GARCH 와 같은 하이브리드 아키텍처들이 좋은 성과를 보였습니다.

### 3. CNN (Convolutional Neural Networks)

<div align="center">
    <img src="/assets/images/paper_review/architecture_CNN.jpg" width="800" alt="Architecture of CNN">
    <br>
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of CNN
    </a>
</div>

주로 이미지 처리에 사용되는 CNN은 트레이딩 분야에서 주로 보조적인 역할을 합니다. 1차원 시계열 데이터를 이미지처럼 간주하여 가격 신호나 여러 시장 지표에서 특징을 추출한 뒤, LSTM 같은 시퀀스 모델로 넘겨주는 식입니다.

1차원 입력 $x$에 대한 필터 $w$의 합성곱 연산은 다음과 같이 정의됩니다:

$$
(x * w)(t) = \sum_{i=0}^{k-1} x_{t+i} w_i
$$

- $t$: 시간 스텝  
- $k$: 커널(필터)의 크기  
- $w_i$: 필터의 가중치  

일부 연구에서는 가격 차트 자체를 CNN에 입력해 기술적 분석을 자동화하려는 시도도 있었지만, 더 영향력 있는 연구들은 대부분 하이브리드 모델의 일부로 CNN을 활용했습니다.

### 4. Autoencoders (AEs & VAEs)

<div align="center">
    <img src="/assets/images/paper_review/architecture_autoencoders.jpg" width="600" alt="Architecture of Autoencoders">
    <br>
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of Autoencoders
    </a>
</div>

오토인코더(표준 및 변형)는 주로 **이상 탐지 및 차원 축소**에 사용됩니다. 예를 들어, 비정상적인 거래 활동을 감지하거나, 가격 신호의 노이즈를 제거하고, 다중 자산 데이터를 더 다루기 쉬운 잠재 공간으로 압축하는 데 쓰입니다.

오토인코더는 입력 데이터 $x$를 잠재 표현 $z$로 변환하는 인코더와, $z$로부터 $x$를 복원하는 디코더로 구성됩니다:

$$
z=\sigma(W_{e}x+b_{e}) \quad (\text{Encoder})
$$

$$
\hat{x}=\sigma(W_{d}z+b_{d}) \quad (\text{Decoder})
$$

- $x$: 원본 입력 데이터  
- $z$: 저차원으로 압축된 잠재 표현  
- $\hat{x}$: 복원된 출력 데이터  
- $W_e$, $W_d$: 각각 인코더와 디코더의 가중치 행렬  
- $b_e$, $b_d$: 각각 인코더와 디코더의 편향  

특히 변형 오토인코더(VAE)는 스트레스 테스트를 위한 가상 데이터를 생성하는 데 활용되기도 합니다. VAE는 잠재 공간에 확률 분포를 가정하는 생성 모델로, 인코더는 다음과 같이 사후 확률 분포의 파라미터를 출력합니다:  

<div align="center">
    <img src="/assets/images/paper_review/architecture_variational_autoencoders_VAE.jpg" width="600" alt="Architecture of Variational Autoencoders (VAE)">
    <br>
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of Variational Autoencoders (VAE)
    </a>
</div>

$$
q_{\phi}(z|x) = \mathcal{N}(z|\mu_{\phi}(x), \sigma_{\phi}(x))
$$

여기서 각 항은 다음을 의미합니다:

- $q_{\phi}(z\|x)$: 사후 확률 분포에 대한 변분 근사(variational approximation)  
- $\mu_{\phi}(x)$: 분포의 평균  
- $\sigma_{\phi}(x)$: 분포의 표준편차  

### 5. GNN (Graph Neural Networks)

GNN은 시장을 하나의 거대한 그래프로 모델링하는 최신 접근법입니다. 개별 주식이나 트레이더를 노드(node)로, 이들 간의 상관관계나 자금 흐름을 엣지(edge)로 표현하는 것이죠. 이를 통해 GNN은 고빈도 거래 데이터에서 시장 조작 행위를 탐지하거나, 여러 자산 간의 상호 의존성을 기반으로 가격 움직임을 예측하는 데 효과적이었습니다.

GNN의 일반적인 메시지 전달(message-passing) 연산은 다음과 같습니다:

$$
h_{i}^{(k+1)}=\sigma\left[W^{(k)}\cdot h_{i}^{(k)}+\sum_{j\in N(i)}\frac{1}{c_{ij}}\cdot W^{(k)}\cdot h_{j}^{(k)}\right]
$$

- $h_{i}^{(k)}$: $k$번째 레이어에서 노드 $i$의 은닉 상태  
- $\mathcal{N}(i)$: 노드 $i$의 이웃 노드 집합  
- $W^{(k)}$: $k$번째 레이어의 학습 가능한 가중치 행렬
- $c_{ij}$: 정규화 상수

다만, 그래프를 어떻게 구성하느냐에 따라 성능이 크게 좌우되고 연산 비용이 비싸다는 단점이 있습니다.

### 6. Transformers

<div align="center">
    <img src="/assets/images/paper_review/architecture_transformer.jpg" width="600" alt="Architecture of Transformer">
    <br>
    <a href="https://doi.org/10.7717/peerj-cs.2555">
        Architecture of Transformer
    </a>
</div>

트랜스포머는 이 분야에서 가장 주목받는 최신 모델입니다. 핵심인 **어텐션 메커니즘(Attention Mechanism)** 덕분에 시끄러운 금융 데이터에서 장기적인 패턴을 추출하는 데 매우 적합합니다. RNN이나 LSTM처럼 데이터를 순차적으로 처리하는 대신, 병렬로 처리하며 입력값들의 중요도를 동적으로 재계산합니다.

셀프 어텐션(Self-Attention) 스코어는 다음과 같이 계산됩니다:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q$, $K$, $V$: 각각 쿼리(Queries), 키(Keys), 값(Values) 행렬로, 입력 시퀀스로부터 파생됨  
- $d_k$: 키 벡터의 차원으로, 그래디언트 안정화를 위한 스케일링 인자

최근 논문들은 고빈도 호가창(limit order book) 데이터나 다중 자산 변동성 예측에서 뛰어난 성능을 보였지만, 막대한 양의 데이터와 긴 학습 시간을 요구하며 노이즈에 과적합될 위험이 크다는 한계도 명확합니다.

### 7. Reinforcement Learning (RL)

강화학습은 시뮬레이션된 환경과의 상호작용을 통해 누적 보상(주로 수익률이나 샤프 지수)을 극대화하는 최적의 트레이딩 전략을 스스로 학습합니다. 최근에는 LSTM이나 CNN을 결합한 심층 강화학습(Deep RL)이 주로 사용됩니다.

강화학습의 학습 과정은 현재 상태($s_t$)와 행동($a_t$)의 가치를 나타내는 벨만 방정식에 의해 결정됩니다:

$$
Q(s_{t},a_{t})=r_{t}+\gamma\max_{a'} Q(s_{t+1},a^{\prime})
$$

- $Q(s_t, a_t)$: 상태 $s_t$에서 행동 $a_t$를 했을 때 기대되는 미래 보상의 합(Q-value)
- $r_t$: 행동 $a_t$에 대한 즉각적인 보상
- $\gamma$: 미래 보상에 대한 할인 계수(discount factor)

DQN(Deep Q-Networks)과 같은 모델들은 기술적 지표와 감성 데이터를 동시에 고려하여 기존 전략을 능가하는 성과를 보였지만, 학습 효율이 낮고 예측 불가능한 시장 상황에 취약하다는 문제가 있습니다.

## 핵심 패턴: 순수 모델보다 강력한 하이브리드

논문 전체를 관통하는 패턴은 명확합니다: **하이브리드 모델은 거의 항상 순수 모델보다 뛰어난 성능을 보입니다.** 특히 시장 미시구조, 변동성 군집, 행동 심리학 등 금융 **도메인 지식** 을 딥러닝의 유연성과 결합한 아키텍처들이 최고의 결과를 냈습니다. 순수한 블랙박스 모델은 여전히 극단적인 상황에서 어려움을 겪으며, 통계적 기반과 딥러닝을 혼합한 모델들이 더 나은 성과를 보이고 있습니다.

## 여전히 남은 현실의 과제들

물론 장점만 있는 것은 아닙니다. 딥러닝 모델들도 수많은 문제점을 안고 있습니다.

- **데이터 품질**: 딥러닝은 노이즈가 섞인 원시 데이터를 직접 처리할 수 있지만, 금융 데이터의 노이즈는 그 한계를 넘어섭니다. 시장 구조가 계속 변해 과거의 분포가 유지되지 않고, 신호보다 잡음이 훨씬 많아 모델이 의미 없는 패턴에 쉽게 과적합될 수 있습니다. 결국 학습 과정이 불안정해지고, 예측력은 일관성을 잃기 쉽습니다.
- **과적합(Overfitting)**: 신호 대 잡음비가 낮은 시장 데이터에 복잡한 모델을 학습시키면 과거 데이터에만 과도하게 최적화될 위험이 큽니다.
- **해석 가능성**: "블랙박스"라는 특성 때문에 모델이 왜 그런 결정을 내렸는지 이해하기 어렵습니다. 이는 규제가 엄격한 금융 산업에서 큰 걸림돌입니다.
- **연산 복잡성 및 지연 시간**: 딥러닝 모델은 거대하고 느립니다. 트레이딩에서 빠른 추론 속도는 사치가 아니라 기본입니다. 모두가 '플래시 보이스'인 지금, 당신의 모델이 너무 느리다면 먼지 속에 남겨질 뿐입니다. GPU 가속, 배치 처리, 신중한 파이프라인 오케스트레이션 없이는 레이턴시 예산을 맞추기조차 어렵습니다.

이것은 단순한 모델링 문제를 넘어선 **운영의 전쟁**입니다. 예측을 맞히는 것뿐만 아니라, 모델이 언제, 왜 실패하는지 알고, 연쇄적인 문제를 일으키지 않으면서 신속하게 교체할 수 있어야 합니다.

## 결론: 트레이더에서 시스템 엔지니어로

이러한 도구들은 분명 효과가 있습니다. 하지만 우리에게 트레이더가 아닌 **시스템 엔지니어처럼 생각할 것**을 강요합니다. 단순히 모델을 선택하는 것이 아니라, 실패의 표면적, 보정 전략, 그리고 인프라에 대한 약속을 선택하는 것입니다.

딥러닝이 확산되면서 경쟁의 패러다임은 '누가 가장 큰 모델을 가졌는가'에서 '데이터가 변할 때 누가 가장 빨리 적응하는가'로 이동할 것입니다. 복잡성의 중심축은 더 이상 손으로 만든 규칙이 아니라, 아키텍처·운영·거버넌스 간의 트레이드오프 위로 이동했습니다. 그리고 이제 그 트레이드오프를 신중하게 결정하는 것은 소프트웨어 엔지니어와 퀀트의 몫이 되었습니다.

---

### References

Bhuiyan, M. S. M., Rafi, M. A., Rodrigues, G. N., Mir, M. N. H., Ishraq, A., Mridha, M. F., & Shin, J. (2025). Deep learning for algorithmic trading: A systematic review of predictive models and optimization strategies. *Array, 26*, 100390. [https://doi.org/10.1016/j.array.2025.100390](https://doi.org/10.1016/j.array.2025.100390)
