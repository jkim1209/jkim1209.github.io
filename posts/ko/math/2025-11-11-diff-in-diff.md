---
layout: post
title: "이중차분법 (Difference-in-Differences)"
date: 2025-11-11
categories: [Econometrics, Causal Inference]
tags: [DiD, Difference-in-Differences, Diff-in-Diff, causal inference, econometrics, microeconometrics, staggered DiD, synthetic control]
math: true
---

## 서론: OLS의 한계와 인과추론의 딜레마

"새로운 지하철 노선 개통이 A 지역의 집값을 올렸을까?"  
"최저임금 인상이 B 주의 고용률을 낮췄을까?"

정책 효과를 분석할 때 가장 먼저 떠오르는 단순 비교 방법들은 자주 오류를 낳습니다.

- **단순 시점 비교 (Before vs. After)**: A 지역의 정책 전후 집값만 비교하면, 도시 전반의 공통된 추세(common trend)를 정책 효과로 착각할 수 있습니다.  
- **단순 그룹 비교 (Treated vs. Control)**: 지하철이 있는 A 지역과 없는 C 지역을 비교하면, 원래 존재하던 고유한 특성의 차이(time-invariant difference)를 정책 효과로 오인할 수 있습니다.

이 두 가지 교란요인 — 시간의 흐름에 따른 변화와 그룹 간 고유한 차 — 를 동시에 제거하기 위해 고안된 방법이 바로 **이중차분법(Difference-in-Differences, DiD)** 입니다.

---

## 1. 이중차분법(DiD)의 핵심 논리: 두 번의 차감

DiD는 이름 그대로 두 번의 차분(차감)을 통해 정책의 순수 효과를 추정합니다.

### 1차 차분: 각 그룹 내에서 '시간의 흐름' 효과 제거

- **처리 집단의 변화량 ($\Delta Y_T$)**  
  = [처리 집단의 시행 후 평균] - [시행 전 평균]  
  = 정책 효과 + 시간의 변화 효과  
- **통제 집단의 변화량 ($\Delta Y_C$)**  
  = [통제 집단의 시행 후 평균] - [시행 전 평균]  
  = 시간의 변화 효과만 포함

### 2차 차분: 그룹 간 차이를 통해 '순수 효과' 분리

> **DiD 추정량 = (처리 집단의 변화량) - (통제 집단의 변화량)**  
> $$
> \hat{\delta}_{DiD} = (Y_{T,After} - Y_{T,Before}) - (Y_{C,After} - Y_{C,Before})
> $$

통제 집단의 변화량(시간 흐름 효과)을 빼줌으로써, 처리 집단의 변화량에 섞여 있던 시간 효과가 제거되고, 결과적으로 순수한 정책 효과(ATT: Average Treatment Effect on the Treated)만 남게 됩니다.

<div align="center">
  <img src="/assets/images/math/did_parallel_trends.png" width="800" alt="Difference-in-Differences Parallel Trends">
  <br>
  <a href="https://patrickthiel.com/the-great-regression-with-python-difference-in-differences-regressions/" target="_blank" rel="noopener noreferrer">
    Graphical representation of DiD idea
  </a>
</div>

---

## 2. 회귀분석을 통한 DiD 구현

이중차분의 논리는 단순한 선형회귀식으로도 표현할 수 있습니다.

$$
Y_{it} = \beta_0 + \beta_1 \text{Treated}_i + \beta_2 \text{After}_t + \delta (\text{Treated}_i \times \text{After}_t) + \mathbf{X}'_{it}\gamma + \epsilon_{it}
$$

| 계수      | 의미                                           | 해석                      |
| :-------- | :--------------------------------------------- | :------------------------ |
| $\beta_0$ | 기준 시점의 통제집단 평균                      | Intercept                 |
| $\beta_1$ | 그룹 간 고유한 차이 (집단 간 불변 수준 차)     | time-invariant difference |
| $\beta_2$ | 시간의 흐름에 따른 공통 변화                   | Common time trend         |
| $\delta$  | 정책 시행 후 처리집단에서만 나타나는 추가 효과 | DiD 추정량 (ATT)          |

상호작용항 $(\text{Treated}_i \times \text{After}_t)$의 추정계수 $\hat{\delta}$가 바로 DiD 추정량이며, 이 계수의 p-value로 통계적 유의성을 검정할 수 있습니다.

---

## 3. DiD의 핵심 가정: 평행 추세 가정 (Parallel Trends Assumption)

DiD 분석이 타당하기 위해서는 다음 가정이 반드시 성립해야 합니다.

> "정책이 시행되지 않았다면, 처리 집단과 통제 집단은 시간이 지나도 평행한 추세를 보였을 것이다."

만약 이 가정이 깨진다면, 두 그룹의 추세 차이까지 정책 효과로 오인될 수 있습니다.

- **검증의 어려움**: 반사실적(counterfactual) 상황을 직접 관찰할 수 없기 때문에 완벽한 검증은 불가능합니다.  
- **대리 검증 방법**: 정책 시행 이전의 여러 시점 데이터를 확보하여, 그 기간 동안 두 그룹이 실제로 평행한 추세를 보였는지 시각적으로(그래프) 혹은 통계적으로 확인합니다.

---

## 4. 고전적 사례: Card & Krueger (1994)

DiD의 대표적 응용은 최저임금과 고용을 분석한 Card & Krueger(1994) 연구입니다.

- **정책**: 1992년 뉴저지(NJ) 주는 최저임금을 \$4.25에서 \$5.05로 인상.  
- **처리집단**: 뉴저지 주의 패스트푸드 레스토랑.  
- **통제집단**: 최저임금을 인상하지 않은 인접 주 펜실베이니아(PA).  
- **결과변수**: 고용 인원.

분석 결과, 뉴저지 주의 고용은 감소하지 않았으며, 이는 “최저임금 인상 → 고용감소”라는 통념을 뒤집은 중요한 결과로 평가받습니다.

---

## 5. 심화 논의: DiD의 한계와 현대적 대안

기본적인 2x2 DiD 모형은 직관적이고 강력하지만, 현실의 복잡한 데이터에 적용할 때는 한계가 있습니다.  
최근 계량경제학은 이러한 한계를 극복하기 위해 다양한 확장 모형을 제시했습니다.

### 5-1. 시차적 정책 도입: Staggered DiD

#### (1) 무엇이 문제인가?

현실에서는 정책이 모든 지역에 동시에 도입되기보다, 시간차를 두고 순차적으로 도입되는 경우가 많습니다(Staggered Adoption). 과거에는 이러한 데이터를 다음과 같은 2-way fixed effect (TWFE) 모델로 분석하는 것이 일반적이었습니다.

$$Y_{it} = \alpha_i + \gamma_t + \delta^{TWFE} D_{it} + \epsilon_{it}$$

- $\alpha_i$는 개체 고정효과, $\gamma_t$는 시간 고정효과입니다.
- $D_{it}$는 개체 $i$가 시간 $t$에 처리를 받으면 1이 되는 더미 변수입니다.

연구자들은 $\delta^{TWFE}$가 우리가 원하는 평균적인 정책 효과(ATT)일 것이라고 오랫동안 믿어왔습니다. 하지만 최근 연구들은 정책의 효과가 이질적(heterogeneous)일 때, 이 $\delta^{TWFE}$가 심각한 편의(bias)를 가질 수 있음을 증명했습니다.

#### (2) "나쁜 비교" 문제의 핵심: Goodman-Bacon 분해

Goodman-Bacon (2021)은 $\delta^{TWFE}$ 추정량이 실제로는 어떤 2x2 DiD 추정치들의 가중 평균으로 구성되는지를 명확하게 분해하여 보여줍니다. 이 분해에 따르면, $\delta^{TWFE}$는 다음 두 종류의 비교를 모두 포함합니다.

1. **좋은 비교 (Good Comparisons)**: 아직 처리되지 않은 집단이나 영원히 처리되지 않는 집단을 통제 집단으로 사용하는 깨끗한 비교.
2. **나쁜 비교 (Bad Comparisons)**: 이미 처리를 받은 집단을 통제 집단으로 활용하는 비교. 예를 들어, 2012년에 처리를 받은 집단의 효과를 추정하면서, 2010년에 이미 처리를 받은 집단을 통제 집단으로 사용하는 경우입니다.

만약 정책의 효과가 시간이 지남에 따라 변하거나(dynamic treatment effects), 처리 시점에 따라 다르다면(heterogeneous treatment effects), 이러한 "나쁜 비교"는 $\delta^{TWFE}$에 예측할 수 없는 편의를 유발하며, 심지어 실제 효과와 부호가 반대인 결과를 낳을 수도 있습니다.

#### (3) 어떻게 해결하는가?: 분해하고 종합하라 (Decompose and Aggregate)

Callaway & Sant'Anna (2021)와 Sun & Abraham (2021)등 최신 연구들은 "나쁜 비교"를 원천적으로 제거하는 새로운 접근법을 제안합니다. 이들의 핵심 아이디어는 TWFE처럼 모든 데이터를 한 번에 분석하지 않고, 데이터를 의미 있는 단위로 분해하여 각각의 효과를 추정한 뒤, 이를 다시 목적에 맞게 종합하는 것입니다.

##### ① 1단계: 그룹-시간 평균 처리 효과(GATT) 추정

가장 중요한 개념은 그룹-시간 평균 처리 효과(Group-Time Average Treatment Effect, GATT) 입니다.

> $$GATT(g, t) = E[Y_t(1) - Y_t(0) | G_g=1]$$

- $G_g=1$: 특정 시점 $g$에 처음으로 처리를 받은 집단(코호트).
- $Y_t(1), Y_t(0)$: 시간 $t$에서의 처리/미처리 시 잠재적 결과.
- **의미**: $g$ 시점에 처리를 받은 집단이, $t$ 시점에서 경험한 평균적인 정책 효과.

예를 들어, $GATT(2012, 2014)$는 2012년에 정책이 도입된 도시들이 2014년에 경험한 평균적인 정책 효과를 의미합니다. Callaway & Sant'Anna의 방법론은 각 $GATT(g, t)$를 추정할 때, 깨끗한 통제 집단(아직 처리되지 않았거나 영원히 처리되지 않는 집단)만을 사용하여 "나쁜 비교"를 원천적으로 차단합니다.

##### ② 2단계: 의미 있는 방식으로 효과 종합하기 (Aggregation)

수많은 $GATT(g, t)$ 추정치들을 얻고 나면, 연구자는 이를 다양한 방식으로 종합하여 질문에 답할 수 있습니다.

- **그룹(코호트)별 평균 효과**: 특정 코호트가 처리 이후 경험한 효과의 평균.
    > $$ATT(g) = E_{t \ge g} [GATT(g,t)]$$
- **이벤트 스터디 (Event Study)**: 처리를 받은 후 시간이 얼마나 경과했는지(relative time)에 따라 정책 효과가 어떻게 변하는지 동적으로 분석.
    > $$ATT(e) = E_{g} [GATT(g, g+e)] \quad (e \ge 0)$$
- **전체 평균 효과 (Overall ATT)**: 모든 처리 집단과 처리 후 기간에 대한 가중 평균.

이러한 접근법은 하나의 왜곡될 수 있는 $\delta^{TWFE}$ 계수 대신, 정책 효과의 동학(dynamics)과 이질성(heterogeneity)을 투명하고 강건하게 보여주는 풍부한 정보를 제공합니다. 시차적 적용 데이터가 있다면, TWFE 대신 이러한 새로운 방법론을 사용하는 것이 현재의 표준입니다.

### 5-2. 적절한 통제집단이 없을 때: 합성 통제법 (Synthetic Control Method)

#### 무엇이 문제인가?

DiD의 핵심인 평행 추세 가정을 만족하는 단일 통제 집단을 찾기 어려운 경우가 많습니다. 특히 처리 집단이 주(state)나 국가처럼 매우 크고 고유한 특성을 가질 때 그렇습니다. 예를 들어, "캘리포니아 주가 특정한 담배 규제 법안을 통과시켰을 때 그 효과는?" 이라는 질문에, 과연 텍사스나 플로리다 중 어느 주가 캘리포니아의 적절한 '쌍둥이'가 되어줄 수 있을까요? 아마 없을 것입니다.

#### 해결책: "가상의 쌍둥이"를 만들다

합성 통제법(Synthetic Control Method, SCM)은 단 하나의 통제 집단을 찾는 대신, 여러 통제 집단(donor pool)들을 가중 평균하여 처리 집단과 매우 유사한 움직임을 보이는 "가상의 최적 통제 집단"을 만들어내는 방법론입니다.

<div align="center">
  <img src="/assets/images/math/scm-graph.jpg" width="800" alt="Synthetic Control Method">
  <br>
  <a href="https://lost-stats.github.io/Model_Estimation/Research_Design/synthetic_control_method.html" target="_blank" rel="noopener noreferrer">
    Synthetic Control Method
  </a>
</div>

- **작동 방식**: SCM은 정책 시행 이전(pre-treatment) 기간 동안, 처리 집단의 실제 값과 합성 통제 집단의 값이 거의 완벽하게 일치하도록 만드는 최적의 가중치(weights)를 데이터 기반으로 찾아냅니다.
- **효과 추정**: 이 가중치를 그대로 유지한 채, 정책 시행 이후(post-treatment) 기간의 처리 집단 실제 값과 합성 통제 집단의 값을 비교합니다. 두 경로 사이의 차이가 바로 정책의 인과적 효과가 됩니다.
- **장점**: 연구자가 임의로 통제 집단을 고르는 것을 방지하고, 어떤 통제 집단들이 어떤 가중치로 조합되었는지 투명하게 보여주므로 결과의 신뢰도를 높입니다. SCM은 특히 처리 집단의 수가 하나(N=1)인 사례 연구(case study)에서 매우 강력한 힘을 발휘합니다.

#### 합성 통제법(SCM)의 구체적인 방법론

SCM의 핵심 목표는 정책 시행 전 기간 동안 처리 집단(treated unit)의 특성과 거의 완벽하게 일치하는 가상의 통제 집단(synthetic control)을 데이터 기반으로 구축하는 것입니다. 이 과정은 다음과 같은 명확한 단계로 이루어집니다.

##### ① 데이터 구성 (The Setup)

먼저 분석에 필요한 데이터를 구성합니다.

- **처리 집단 (Treated Unit)**: 정책의 영향을 받은 단 하나의 개체 (i=1). (예: 캘리포니아 주)
- **도너 풀 (Donor Pool)**: 정책의 영향을 받지 않은 여러 개의 통제 집단 후보군 (i=2, ..., J+1). (예: 캘리포니아를 제외한 다른 주들)
- **기간**: 정책 시행 전 기간($t = 1, \dots, T_0$)과 시행 후 기간($t = T_0+1, \dots, T$)으로 나뉩니다.

##### ② 2단계: 최적의 가중치(W) 찾기 (The Optimization Problem)

SCM의 핵심은 도너 풀에 속한 각 통제 집단에 부여할 최적의 가중치 벡터 $W = (w_2, \dots, w_{J+1})'$를 찾는 것입니다. 이 가중치는 두 가지 중요한 제약 조건을 가집니다.

1. **음수 가중치 없음 (Non-negativity)**: $w_j \ge 0$ for all $j \in \{2, \dots, J+1\}$
2. **가중치의 합은 1 (Sum to one)**: $\sum_{j=2}^{J+1} w_j = 1$

이 제약 조건은 합성 통제 집단이 도너 풀의 볼록 조합(convex combination)으로만 만들어지도록 하여, 임의의 외삽(extrapolation)을 방지하는 중요한 역할을 합니다.

가중치는 정책 시행 전($t \le T_0$) 기간 동안, 처리 집단의 특성과 합성 통제 집단의 특성 간의 거리가 최소화되도록 선택됩니다. 여기서 '특성'이란, 결과 변수($Y$)의 과거 값들과 결과에 영향을 미치는 다른 예측 변수($X$)들을 모두 포함합니다.

수학적으로는 다음 최적화 문제를 풉니다.

> $$\min_{W} (X_1 - X_0 W)'V(X_1 - X_0 W)$$

- $X_1$: 처리 집단의 정책 시행 전 특성 벡터.
- $X_0$: 도너 풀에 속한 모든 통제 집단들의 정책 시행 전 특성 행렬.
- $W$: 우리가 찾고자 하는 가중치 벡터.
- $V$: 각 특성 변수의 상대적 중요도를 나타내는 가중 행렬. 보통 V는 정책 시행 전 기간의 평균 제곱 예측 오차(MSPE)를 최소화하도록 데이터 기반으로 결정됩니다.

##### ③ 3단계: 합성 통제 집단 구축 (Constructing the Synthetic Control)

최적의 가중치 벡터 $$W^*=(w_2^*, \dots, w_{J+1}^*)'$$ 를 찾았다면, 모든 기간 $(t=1, \dots, T)$ 에 대해 합성 통제 집단의 결과 변수 값을 계산할 수 있습니다.

$$Y_{synthetic, t} = \sum_{j=2}^{J+1} w_j^* Y_{j,t}$$

최적화 과정 덕분에, 정책 시행 전 기간($t \le T_0$) 동안에는 실제 처리 집단의 값($Y_{1,t}$)과 합성 통제 집단의 값($Y_{synthetic, t}$)이 거의 동일한 경로를 그리게 됩니다.

##### ④ 4단계. 정책 효과 추정 (Estimating the Treatment Effect)

정책 효과는 정책 시행 후($t > T_0$) 기간 동안, 실제 처리 집단의 값과 가상 통제 집단의 값 사이의 차이로 간단하게 계산됩니다.

> $$\hat{\alpha}_t = Y_{1,t} - Y_{synthetic, t} = Y_{1,t} - \sum_{j=2}^{J+1} w_j^* Y_{j,t}$$

이 차이($\hat{\alpha}_t$)가 바로 정책이 없었을 경우의 반사실적(counterfactual) 경로 대비 실제 경로의 차이, 즉 정책의 효과를 나타냅니다.

##### ⑤ 5단계. 통계적 추론 (Inference)

SCM에서 추정된 효과가 통계적으로 유의한지, 즉 우연히 발생한 것인지 판단하는 것은 조금 더 복잡합니다. 표준오차를 직접 계산하기 어렵기 때문에, 보통 플라시보 검정(Placebo Test) 또는 순열 검정(Permutation Test)이라 불리는 방법을 사용합니다.

1. **가짜 처리 집단 설정**: 도너 풀에 있는 각각의 통제 집단을 마치 처리 집단인 것처럼 가정합니다.
2. **SCM 반복 적용**: 이 '가짜' 처리 집단들 각각에 대해 1~4단계의 SCM 분석을 모두 반복하여, '가짜' 정책 효과를 계산합니다.
3. **효과 비교**: 실제 처리 집단에서 계산된 정책 효과($\hat{\alpha}_t$)를 수많은 가짜 정책 효과들의 분포와 비교합니다.
4. **유의성 판단**: 만약 실제 처리 집단의 효과가 대부분의 가짜 효과들보다 월등히 크다면, 이 효과는 우연히 발생한 것이 아니라고 결론 내릴 수 있습니다.

이처럼 SCM은 직관적인 아이디어를 바탕으로 하면서도, 데이터 기반 최적화와 정교한 통계적 추론 과정을 통해 단일 사례 연구에 대한 높은 신뢰도를 제공하는 강력한 방법론입니다.

---

## 정리: DiD의 핵심 요약

| 구분        | 내용                                                    |
| ----------- | ------------------------------------------------------- |
| 목적        | 시간 변화와 그룹 차이를 동시에 통제                     |
| 핵심 가정   | 평행 추세 가정 (Parallel Trends)                        |
| 추정식      | $(Y_{T,After}-Y_{T,Before})-(Y_{C,After}-Y_{C,Before})$ |
| 대표 사례   | Card & Krueger (1994)                                   |
| 확장 방법론 | Staggered DiD, Synthetic Control Method                 |

---

## References

- Card, D., & Krueger, A. (1994). *Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania*. **American Economic Review**, 84(4), 772–793.  
- Goodman-Bacon, A. (2021). *Difference-in-Differences with Variation in Treatment Timing*. **Journal of Econometrics**, 225(2), 254–277.  
- Callaway, B., & Sant’Anna, P. H. (2021). *Difference-in-Differences with multiple time periods*. **Journal of Econometrics**, 225(2), 200–230.  
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic control methods for comparative case studies*. **Journal of the American Statistical Association**, 105(490), 493–505.
- [The Great Regression — with Python: Difference-in-Differences Regressions](https://patrickthiel.com/the-great-regression-with-python-difference-in-differences-regressions/)
- [Understanding Synthetic Control and Causal Inference in A/B Testing](https://medium.com/@suraj_bansal/understanding-synthetic-control-and-causal-inference-in-a-b-testing-e10e67d570a0)
- [LOST-stats Synthetic Control Method (SCM)](https://lost-stats.github.io/Model_Estimation/Research_Design/synthetic_control_method.html)
