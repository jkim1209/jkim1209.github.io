---
layout: post 
title: "'최고'의 커피는 무엇일까? AI가 인간의 모호한 취향을 이해하는 법" 
date: 2025-10-02
categories: [Research, Review]
tags: [expert system, fuzzy logic, AHP, MCDM, q-rung orthopair fuzzy sets, decision making] 
math: true
---

## "ChatGPT야, 이 도시에서 제일 맛있는 커피는 뭐야?"

우리가 AI에게 던지는 "최고의 OO"라는 질문은 사실 AI에게 가장 어려운 질문 중 하나입니다. AI가 단순히 인터넷의 별점 평균을 계산해서 알려준다면, 우리는 만족할 수 있을까요? '평점이 가장 높은 곳'과 우리가 느끼는 '최고의 장소'는 분명 다릅니다. 후자는 가격, 분위기, 서비스, 개인적인 추억 등 수많은 요소가 복잡하게 얽힌, 지극히 주관적인 영역이기 때문입니다.

이것은 데이터를 넘어 '전문가의 안목'을 요구하는 질문입니다. 최근 발표된 한 논문은 바로 이 문제, 즉 인간의 모호하고 다층적인 선호도를 AI가 어떻게 이해하고 최적의 결정을 내릴 수 있는지에 대한 새로운 청사진을 제시합니다.

## 무엇이 의사결정을 복잡하게 만드는가?

'최고의 커피'를 고르는 과정은 단순히 맛 하나로 결정되지 않습니다. 어떤 사람은 가격을, 다른 사람은 원두의 공정무역 인증을, 또 다른 누군가는 사회적 지위를 나타내는 브랜드를 중요하게 생각합니다. 이처럼 서로 상충하고 때로는 비교조차 어려운 여러 기준 속에서 최적의 대안을 찾아내는 것을 다기준 의사결정(MCDM, Multi-Criteria Decision-Making)이라고 합니다.

<div align="center">
    <img src="/assets/images/paper_review/ahp-hierarchy.png" width="600" alt="hierarchy">
    <br>
    <a href="https://www.1000minds.com/decision-making/analytic-hierarchy-process-ahp">
        Analytic Hierarchy Process example
    </a>
</div>

이 복잡한 문제를 풀기 위해 연구자들은 계층 분석법(AHP, Analytic Hierarchy Process)라는 강력한 도구를 사용합니다. AHP는 위 그림처럼 '목표-기준-대안'의 계층을 만들고, 각 기준의 중요도를 전문가가 1:1로 비교하여 가중치를 매기는 방식입니다.

하지만 여기에는 치명적인 한계가 있습니다. 전문가에게 "가격이 품질보다 정확히 1.5배 더 중요합니까?"라고 물으면 자신 있게 답할 수 있는 사람이 몇이나 될까요? 인간의 판단은 본질적으로 "음... 가격이 좀 더 중요한 것 같긴 한데..."와 같이 모호합니다. 바로 이 "불확실성" 때문에 기존의 AHP는 한계에 부딪힙니다.

## AI, '모호함'을 배우다: q-ROFS

이 문제를 해결하기 위해 등장한 것이 바로 퍼지 논리(Fuzzy Logic)입니다. '좋다/나쁘다'의 흑백논리를 넘어 '조금 좋다', '매우 좋다'와 같은 중간 지대를 수학적으로 표현하는 방식이죠.

그리고 이 논문은 여기서 한 걸음 더 나아간 최신 이론, q-rung 직교쌍 퍼지 집합(q-ROFS, q-rung Orthopair Fuzzy Sets)을 AHP에 접목합니다.

### 전문가의 '망설임'까지 담아내는 q-ROFS

기존의 퍼지 이론이 '어느 정도 속하는가'(소속도)에 집중했다면, q-ROFS는 '어느 정도 속하는가'(소속도, $\tilde{E}_S$)와 '어느 정도 속하지 않는가'(비소속도, $\dot{G}_S$)를 함께 고려합니다.

가장 큰 혁신은 이 두 값의 관계에 있습니다. q-ROFS는 다음 조건을 만족하며 소속도와 비소속도에 훨씬 큰 자유를 부여합니다.

$$
0 \ge \tilde{E}_{S}(r)^{q} + \dot{G}_{S}(r)^{q} \le 1, \quad (q \ge 1)
$$

여기서 파라미터 $q$ 값이 커질수록, 소속도와 비소속도가 가질 수 있는 값의 범위가 넓어집니다. 이는 전문가가 특정 항목에 대해 "좋다고 말하기도 애매하고, 나쁘다고 할 수도 없는" 망설임이나 불확실성까지도 수학적으로 표현할 수 있게 해줍니다.  

<div align="center">
    <img src="/assets/images/paper_review/qrofs-comparison.png" width="500" alt="Comparison of fuzzy set spaces">
    <br>
    Membership degree(μ)는 어떤 대상이 집합에 "속하는 정도", Non-membership degree(ν)는 "속하지 않는 정도"를 의미한다.  
    IFS(q=1)에서는 두 값의 합이 1을 넘지 않도록 제한되지만, q가 커질수록 (μ, ν)을 동시에 더 크게 허용해 전문가의 망설임이나 불확실성까지 표현할 수 있다.
</div>


### TR-q-ROFNS: 실제 계산을 위한 확장

q-ROFS가 이론적 틀이라면, 논문에서 실제 의사결정 문제에 사용된 것은 TR-q-ROFNS (Triangular q-rung orthopair fuzzy number set)입니다.  
이는 점 단위 값 대신 삼각 퍼지 수 $(a_1, a_2, a_3)$로 소속도와 비소속도를 표현해, 전문가가 “대략 이 정도”라고 애매하게 평가한 부분을 범위(하한–중앙–상한)로 다룰 수 있도록 확장한 형태입니다.  
덕분에 보다 현실적인 불확실성을 반영할 수 있고, MCDM 같은 실제 의사결정 문제에 적용하기 용이합니다.

### 모호함을 점수로 바꾸는 방법: Score & Accuracy Functions

이렇게 표현된 모호한 퍼지 값들의 순위를 매기기 위해 논문은 점수 함수(Score Function)와 정확도 함수(Accuracy Function)를 사용합니다.

$$
S(\tilde{a}) = \frac{(a_1+a_2+a_3)(1+E^q-G^q)}{6}, \quad
H(\tilde{a}) = \frac{(a_1+a_2+a_3)(1+E^q+G^q)}{6}
$$

이 함수들은 전문가가 내린 판단(값의 범위 $a_{1,2,3}$), 그 판단에 대한 확신($E$), 비확신($G$)을 종합하여 하나의 명확한 점수로 환산합니다.

### 최종 결과: 남아공 최고의 커피 브랜드

연구진은 다섯 가지 주요 기준(C1: Availability, C2: Effectiveness, C3: Price, C4: Quality, C5: Quantity)과 세 개의 대안 브랜드(A1, A2, A3)를 대상으로 TR-q-ROFNS 기반 FAHP를 적용했습니다.

<div align="center">
    <img src="/assets/images/paper_review/qrofs-result.png" width="500" alt="qrofs-result">
    <br>
    <a href="https://peerj.com/articles/cs-2555/" target="_blank">
        Final ranking
    </a>
</div>


모든 기준을 종합한 최종 우선순위 계산 결과, A2 가 가장 높은 점수를 받아 남아공 최고의 커피 브랜드로 선정되었습니다.

이는 단순히 평점 평균을 내는 방식이 아니라, "불확실성과 모호함을 반영해 전문가 판단을 수학적으로 체계화한 결과"라는 점에서 의미가 있습니다.

## 설명 가능한 '전문가 AI'의 시대를 열 수 있을까?

이 논문은 AI가 인간의 주관적인 영역에 어떻게 발을 들여놓을 수 있는지 보여주는 청사진과 같습니다. 복잡한 문제를 체계적으로 구조화하고, 인간의 모호한 판단을 정교하게 수학적 언어로 번역하여, 최종적으로는 투명하고 설명 가능한 결정을 내리는 전 과정을 담고 있기 때문입니다.

데이터를 기반으로 패턴을 학습하는 '블랙박스' 모델들과 달리, 이와 같은 논리 기반 전문가 시스템은 '왜' 그런 결정을 내렸는지 그 과정을 명확히 추적할 수 있다는 강력한 장점을 가집니다. AI가 우리의 취향을 이해하고 더 나은 추천을 해주는 미래는, 어쩌면 더 많은 데이터를 학습하는 것이 아니라, 이처럼 우리의 모호함과 불확실성을 더 깊이 이해하는 것에서부터 시작될지도 모르겠습니다.

## 한계와 미래 연구 방향

- **계산 복잡성**: q-rung 퍼지 AHP는 많은 연산을 요구하므로 대규모 문제에는 계산 비용이 큽니다. 연구진도 실제 서비스화를 위해 소프트웨어 최적화가 필요하다고 언급합니다.
- **비소속도 값 정당화 문제**: 전문가가 "이 브랜드는 나쁘다"고 확신하지 않는 경우, 비소속도(non-membership grade)를 어떻게 설정할지가 실증적으로 어렵습니다.  

---

## References

Huang Y, Gulistan M, Rafique A, Chammam W, Aurangzeb K, Rehman AU. 2025. *The technique of fuzzy analytic hierarchy process (FAHP) based on the triangular q-rung fuzzy numbers (TR-q-ROFNS) with applications in best African coffee brand selection.* PeerJ Computer Science 11:e2555 [https://doi.org/10.7717/peerj-cs.2555](https://doi.org/10.7717/peerj-cs.2555)
