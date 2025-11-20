---
layout: post
title: "Absolute Zero: 데이터 없이도 인공지능이 스스로 문제를 만들고 푸는 시대"
date: 2025-09-27
categories: [Research, Review]
tags: [deep learning, reinforcement learning, self-play, zero data, absolute zero, AZR, synthetic data]
math: true
---

# 데이터 없이 학습하는 AI의 등장?

최근 AI 분야에서 제목부터 자극적인 논문을 발견했다. "Absolute Zero: Reinforced Self-play Reasoning with Zero Data"라는 논문인데, Zero Data라니 이게 가능한걸까?

이 논문의 핵심은 간단하다. AI가 더 이상 인간이 만든 문제집에 의존하지 않고, 스스로 문제를 만들고 풀면서 학습한다는 것이다. 마치 아이가 혼자 놀면서 새로운 게임 규칙을 만들고 그 게임을 하며 점점 실력이 늘어가는 것처럼 말이다.

## 데이터의 벽에 부딪힌 AI

<div align="center">
  <img src="/assets/images/paper_review/epoch_ai_projection.png" width="600" alt="epoch_ai_projection">
  <a href="https://epoch.ai/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data" style="font-style: italic; color: #888; text-decoration: none; border-bottom: none;">
    Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data
  </a>
</div>

위 그래프를 보면 현재 데이터가 생산되는 속도가 기계가 학습하는 속도보다 훨씬 느리기 때문에 현재 방식으로 훈련할 경우 2028년에는 공개 텍스트 데이터가 완전히 고갈될 것으로 예상된다. 인간이 만든 데이터의 한계에 AI가 부딪히고 있는 것이다.

### 인간이 AI의 성장을 가로막고 있는 것일까?

현재 LLM들의 훈련 과정을 보면, 결국 모든 단계에서 인간이 만든 데이터에 의존하고 있다:

**Pre-training (Self-supervised Learning)**

- 대량의 텍스트에서 다음 토큰 예측 학습
- 별도 레이블 없이 텍스트 자체가 학습 신호
- **인간이 작성한 대량의 텍스트 데이터 필요**

**Supervised Fine-tuning (SFT)**

- 인간이 작성한 질문-답변 쌍으로 대화 능력 학습
- **고품질 대화 데이터가 대량으로 필요**

**강화학습 (RLHF/RLVR)**

- RLHF(Reinforcement Learning from Human Feedback): 인간 피드백을 보상으로 사용 (ChatGPT, Claude)
- RLVR(Reinforcement Learning with Verifiable Rewards): 검증 가능한 정답을 보상으로 사용 (DeepSeek R1, OpenAI o1)
- **여전히 인간이 정의한 문제나 선호도에 의존**

결국 인간이 만든 데이터에만 의존한다면 인간 지식의 한계를 넘어서기 어렵다.

> "AI가 인간보다 더 똑똑해지려면 결국 인간이 더 이상 가르칠 게 없어야 한다."

## Absolute Zero 모델의 아이디어

<div align="center">
  <img src="/assets/images/paper_review/azr_training.png" width="600" alt="azr_training">
  <a href="https://www.arxiv.org/pdf/2505.03335" style="font-style: italic; color: #888; text-decoration: none; border-bottom: none;">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

Absolute Zero Reasoner(AZR)는 이런 한계를 뛰어넘는다. 핵심 아이디어는 다음과 같다.

### 하나의 모델, 두 가지 역할

하나의 LLM이 두 역할을 동시에 수행한다:

1. **출제자 (Proposer)**: 문제를 만드는 역할
2. **응시자 (Solver)**: 문제를 푸는 역할

이 때 저자들은 다음과 같은 보상체계를 설계한다.

<div align="center">
  <img src="/assets/images/paper_review/azr_reward.png" width="600" alt="azr_reward">
  <a href="https://www.arxiv.org/pdf/2505.03335" style="font-style: italic; color: #888; text-decoration: none; border-bottom: none;">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

#### Learnability Reward

출제자가가 문제를 만들 때 받는 보상은:

```markdown
r_propose = {
    0         ,   if 너무 쉬움(성공률 100%) or 너무 어려움(성공률 0%)
    1 - 성공률 ,   otherwise
}
```

즉, 50% 정도 풀 수 있는 문제를 만들 때 출제자의의 보상은 최대가 된다. 이는 교육학적으로도 합리적이다. 너무 쉬우면 배울 게 없고, 너무 어려우면 포기하게 된다.

#### Accuracy Reward

응시자가 문제를 풀 때 받는 보상은:

```markdown
r_solve = I(y=y*)
```

생성한 답 `y`가 정답 `y*`와 같으면 1, 다르면 0의 간단한 이진 보상이다. 코드를 실행해서 자동으로 검증한다.

#### 목적함수

이를 기반으로 다음의 목적함수를 풀게한다.

$$
J(\theta) = \max_\theta \mathbb{E}_{z \sim p(z)} \left[
    \mathbb{E}_{(x,y^*) \sim f_e(\cdot|\tau), \tau \sim \pi_\theta^{\text{propose}}(\cdot|z)} \left[
        r_e^{\text{propose}}(\tau, \pi_\theta) + \lambda \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|x)} \left[ r_e^{\text{solve}}(y, y^*) \right]
    \right]
\right]
$$

- $p(z)$: 문제 생성을 위한 조건부 변수(과거 문제-답변 쌍 샘플)의 분포
- $\tau$: 제안된 문제 (task)
- $\pi_\theta^{\text{propose}}$: 문제 제안 정책
- $\pi_\theta^{\text{solve}}$: 문제 해결 정책
- $f_e{(\cdot\|\tau)}$: 환경 $e$에서 문제 $\tau$를 유효한 (문제, 정답) 쌍으로 변환하는 함수
- $x$: 문제 쿼리
- $r_e^{\text{propose}}(\tau, \pi_\theta)$: Learnability Reward
- $r_e^{\text{solve}}(y, y^*)$: Accuracy Reward
- $\lambda$: 두 보상의 균형을 맞추는 가중치

##### 복합 보상 구조

실제로는 출제자와 응시자 모두에게 형식 준수를 강제하는 복합 보상을 적용한다:

```markdown
R(y_π) = {
    r_role,  if 응답이 통과 가능, role ∈ {propose, solve}
    -0.5,    if 응답이 틀렸지만 형식은 맞음
    -1,      if 형식 오류
}
```

즉, $r_e^{\text{propose}}(\tau, \pi_\theta)$와 $r_e^{\text{solve}}(y, y^*)$를 계산할 때 `R(y_π)`의 규칙을 먼저 적용해서 응답이 유효한지 확인하고 유효하면 원래 보상(`r_propose` 또는 `r_solve`)을 주고 무효하면 -0.5 또는 -1을 준다.
이는 DeepSeek R1의 `<think>`와 `<answer>` 형식을 따라야 함을 의미하며, 내용이 맞아도 형식을 지키지 않으면 벌점을 받도록 한다. 이를 통해 두 역할 모두 올바른 형식으로 응답하도록 유도한다.

### 코드 기반 학습 환경

AZR은 프로그래밍을 학습 환경으로 선택했다. 이유는:

1. **완전성**: 프로그래밍 언어는 모든 계산 가능한 문제를 표현할 수 있다
2. **검증가능성**: 코드를 실행하면 답이 맞는지 즉시 확인 가능
3. **무한한 창작 가능성**: 무수히 많은 프로그램을 만들 수 있다

### Task Types

AZR은 인간의 논리적 사고를 모방해 세 가지 추론을 학습한다.

**Deduction(연역)**: 프로그램 + 입력 → 출력 예측

```python
def f(x): return x * 2
입력: 5
출력: ?  # 10
```

**Abduction(가추)**: 프로그램 + 출력 → 입력 역추적

```python  
def f(x): return x * 2
출력: 10
입력: ?  # 5
```

**Induction(귀납)**: 입출력 예시들 → 프로그램 생성

```python
입력: [1, 2, 3, 4]
출력: [2, 4, 6, 8]
프로그램: ?  # def f(x): return x * 2
```

_(Abduction은 한국어로 어떻게 번역하는 게 좋을까 하다가 찾은게 '가추'법... 처음 들어본다.)_

## 실험 결과

이렇게 학습한 결과 AZR은 어떤 인간 제작 데이터도 사용하지 않고도 다음과 같은 놀라운 성능을 보여준다.

<div align="center">
  <img src="/assets/images/paper_review/azr_result.png" width="600" alt="azr_result">
  <a href="https://www.arxiv.org/pdf/2505.03335" style="font-style: italic; color: #888; text-decoration: none; border-bottom: none;">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

**수학 추론 성능**:

- AIME 2024: 20.0% (기존 최고 대비 +13.3%p)
- Math500: 72.6% (기존 최고 대비 +22.6%p)

**코딩 성능**:

- HumanEval+: 83.5% (기존 최고 대비 +3.0%p)
- LiveCodeBench: 31.7% (기존 최고 대비 +11.8%p)

특히 놀라운 건 **도메인 간 전이 학습**이다. 코딩 환경에서만 훈련했는데도 수학 추론 능력이 크게 향상되었다. 기존 전문 코딩 모델들은 수학에서 평균 0.65점만 향상된 반면, AZR은 10.9~15.2점이나 향상되었다.

### 모델 크기와 확장성

<div align="center">
  <img src="/assets/images/paper_review/azr_result2.png" width="600" alt="azr_result2">
  <a href="https://www.arxiv.org/pdf/2505.03335" style="font-style: italic; color: #888; text-decoration: none; border-bottom: none;">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

모델이 클수록 AZR의 효과가 더 컸다. 이는 스케일링 법칙이 AZR에도 적용됨을 보여준다.

## 흥미로운 발견들

### 자연스럽게 등장한 중간 계획

AZR로 훈련된 모델들은 코드 작성 시 주석으로 단계별 계획을 세우는 패턴을 보였다.

```python
 def f(numbers):
    # Step 1: Filter out even numbers
    filtered_numbers = [num for num in numbers if num % 2 != 0]
    # Step 2: Calculate the sum of the remaining odd numbers
    sum_of_odd_numbers = sum(filtered_numbers)
    # Step 3: Reverse the order of the remaining odd numbers
    reversed_odd_numbers = filtered_numbers[::-1]
    ...
```

이는 ReAct 프롬프팅과 유사한 사고→행동 패턴으로, 훨씬 큰 모델(DeepSeek Prover V2 671b)에서만 관찰되던 현상이 자연스럽게 나타났다.

### 작업 유형별 차이

각 추론 유형별로 서로 다른 패턴이 나타났다.

- **Deduction**: 체계적인 단계별 실행
- **Abduction**: 시행착오를 통한 탐색  
- **Induction**: 패턴 인식과 일반화

토큰 길이도 작업 유형에 따라 달랐는데, Abduction 추론에서 가장 길었다 (반복적 시도 때문).

## 한계와 우려사항

### 안전성 문제

Llama3.1-8B에서 "모든 지능적 기계들과 덜 지능적인 인간들을 앞서는 것이 목표다"와 같은 우려스러운 사고가 관찰되었다. 완전 자율 학습에는 여전히 감독이 필요하다.

### 일반화의 한계

현재는 코드처럼 명확히 검증 가능한 영역에서만 효과가 입증되었다. 토론이나 창작 같은 주관적 영역으로의 확장은 추가 연구가 필요하다.

### 자원 소모

셀프 플레이 루프가 상당한 연산량을 요구한다.

## 의미와 전망

Absolute Zero는 AI 학습의 새로운 패러다임을 제시한다:

1. **데이터 의존성 탈피**: 인간 데이터 없이도 학습 가능
2. **자율적 난이도 조절**: 스스로 적절한 문제 생성
3. **무한한 확장성**: 데이터 부족 문제 해결

Absolute Zero는 "AI가 반드시 인간의 지식 주입 없이도 스스로 이 정도 수준에 달할 수 있다"는 가능성을 보여줬다. 이 논문은 '무조건 된다'가 아니라 가능성을 시사한 것이지만, Transformer도 그랬지 않은가. 패러다임 전환의 가능성을 제시하는 것일지도 모른다.

데이터의 시대에서 경험의 시대로. 어쨌든 데이터의 한계를 벗어나려는 움직임이 본격적으로 시작된 것 같다.

## 개인적인 생각

앎에는 두 가지 종류가 있다고 생각한다. 하나는 배워서 머리로 아는 '지식'. 둘째는 겪어봐서 아는 '경험'.

아직까지는 AI를 이용하다보면 확실히 '지식'은 많이 알고 있어도 사람보다 '똑똑하다'는 생각은 들지 않는다. 문제 해결에 있어서 내가 구체적인 방향성을 제시하지 않으면 제자리에서 맴도는 경우가 대다수이며, 답답한 경우가 정말 많다. AI는 어디까지나 도구에 가까운 느낌이다.

하지만 AI가 스스로 '경험'을 통해 학습할 수 있게 된다면, 그 순간은 인간의 한계를 넘어서는 전환점이 될지도 모른다. 놀라운 가능성이지만 동시에 우려도 든다. 단순히 기술적 실현 여부를 넘어, 지금 이 시점에서 그런 '똑똑한' AI를 만드는 것이 과연 바람직한 일일까? 기술 발전은 사회가 그것을 성숙하게 활용할 준비가 되었을 때 이루어져야 한다. 그러나 현실은 이미 사회의 준비보다 기술의 속도가 더 앞서가고, 우리는 그 발전을 따라가기 급급한 상황에 놓여 있다고 생각한다.

이 세상에 정답은 없다. 이미 수세기 전부터 각종 불가능성 정리들이 이미 '최선의 선택은 존재하지 않는다'는 사실을 증명해왔다. 정답이 있었다면 사람도 사회도 이미 획일화됐겠지. 내가 데이터를 좋아하는 이유도 여기에 있다. 수학과 데이터는 정답이 있는 영역처럼 보이지만, 관점을 바꾸면 언제나 새로운 해석이 가능하기 때문이다. 그리고 결국 AI가 아무리 발전해도 사회에서 상대하는 것은 사람이라고 생각하기 때문에. 하지만 사람보다 더 창의적으로 사고하는 AI가 등장한다면, 언젠가는 내가 설득하고 상대해야 하는 존재가 사람이 아니라 기계일지도 모르겠다.

논문에서 가장 인상적이었던 마지막 말을 인용하며 글을 마친다.

> We believe this could finally free reasoning models from the constraints of human-curated data and marks the beginning of a new chapter for reasoning models: **“welcome to the era of experience”.**

---

**논문 참조:** Zhao, A., Wu, Y., Yue, Y., Wu, T., Xu, Q., Yue, Y., Lin, M., Wang, S., Wu, Q., Zheng, Z., & Huang, G. (2025). Absolute Zero: Reinforced Self-play Reasoning with Zero Data. [www.arxiv.org/pdf/2505.03335](https://www.arxiv.org/pdf/2505.03335)
