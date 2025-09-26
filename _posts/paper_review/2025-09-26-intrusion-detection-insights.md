---
layout: post
title: "침입 탐지 논문에서 배우는 정형 데이터 모델링: CTR 예측에의 적용 가능성"
date: 2025-09-26
categories: [Research, Review, Paper, ML]
tags: [deep-learning, machine-learning, intrusion-detection, tabular-data, SMOTE, CTR, FT-transformer]
---

# 정형 데이터: 머신러닝 vs 딥러닝

Ali et al. "Deep Learning vs. Machine Learning for Intrusion Detection in Computer Networks: A Comparative Study" (2025)를 읽으면서 현재 내가 참가 중인 CTR 예측 경진대회의 데이터와 공통점이 많은 것 같아 한번 정리해 보게 되었다. 

## 연구 결과와 의외의 발견

연구진들은 CIC-IDS2017 데이터셋을 사용해서 네트워크 침입 탐지에서 전통적인 머신러닝과 딥러닝 접근법을 체계적으로 비교했다. 

**전통적 ML 결과:**

- **Random Forest: 99.88% 정확도, 97.46% F1-점수** (최고 성능)
- Decision Tree: 99.83% 정확도, 97.60% F1-점수
- K-Nearest Neighbors: 99.88% 정확도, 97.46% F1-점수  
- Logistic Regression: 96.91% 정확도, 74.00% F1-점수
- SVM: 97.32% 정확도, 73.33% F1-점수
- Naive Bayes: 64.59% 정확도, 48.88% F1-점수

**딥러닝 결과:**

- LSTM: 98% 정확도
- CNN: 98% 정확도  
- MLP: 97% 정확도

결과를 보면 **Random Forest가 가장 높은 성능**을 보였다. 딥러닝이 만능은 아니라는 점을 확실히 보여준다.

## 공통된 도전과제들

이 연구에서 흥미로웠던 점은 내가 CTR 경진대회에서 본 데이터와 비슷하다는 점이었다.

**불균형 데이터 문제:** 내 CTR 데이터에서 클릭률이 2% 미만인 것처럼, 이들도 "Web Attacks" 같은 정상 트래픽 대비 약 1%의 소수 클래스를 잘 판별해내야 하는 과제를 풀고 있다. 그들은 **SMOTE(Synthetic Minority Oversampling Technique)** 기법을 사용하였고, 이를 통해 모든 모델의 성능이 향상되었다고 한다.

**피처 엔지니어링:** 연구진들은 0.85 임계값으로 상관관계 기반 피처 선택을 했다. 현재 나는 0.95 (그룹 간은 0.90)을 임계값으로 상관관계 기반 피처 선택만을 했는데, 상관관계만으로 피처 선택을 한다는 점에서 흥미로웠다.

## Random Forest의 우수한 성능

논문에서 가장 인상적인 부분은 Random Forest의 성능이었다.

Random Forest가 딥러닝 모델들보다 우수한 이유로 제시된 것들:

- 고차원 데이터에서 비선형 관계를 잘 포착
- 클래스 불균형을 다른 모델들보다 잘 처리
- 앙상블 특성으로 인한 강건성
- 해석가능성과 계산 효율성

## 딥러닝의 한계점들

딥러닝 모델들은 여러 한계점을 보였다:

**소수 클래스 탐지 어려움:** 특히 클래스 3, 4에서 여전히 misclassification이 발생했다. LSTM이 가장 좋은 균형을 보였지만 완벽하지는 않았다.

**계산 비용:** 저자들은 딥러닝 모델들이 상당한 계산 자원을 요구한다고 지적했다. 특히 실시간 탐지가 필요한 환경에서는 지연 시간이 문제가 될 수 있다.

**오버피팅 경향:** MLP 모델에서 validation loss의 변동이 관찰되어 과적합 가능성이 제기되었다.

## 어떻게 적용해볼까

이 논문을 읽고 어떻게 적용해볼 수 있을까 생각해봤다.

**SMOTE 적용:** 불균형 데이터에서 SMOTE의 효과가 검증되었으니, 내 전처리 파이프라인에서도 활용해볼 수 있을지 살펴봐야겠다.

**상관관계 분석 지속:** 상관관계만으로 피처를 선택한다는 것이 한편으론 불안했는데, 이 부분은 다른 방식으로 피처를 선택하더라도 크게 모델의 성능에 영향을 줄 것 같진 않아 후순위로 미루어도 될 것 같다.

**FT-Transformer 사용에 대한 재검토:** Random Forest 같은 전통적 방법도 여전히 강력하다는 것을 인지하고 적용해봐야겠다.

## 테이블 데이터에 대한 균형 잡힌 시각

이 논문은 테이블 데이터에서 딥러닝이 항상 최고의 선택은 아니라는 현실적 교훈을 준다. 특히:

- **데이터 특성에 따른 모델 선택의 중요성**
- **해석가능성과 계산 효율성의 가치**
- **전통적 방법들의 지속적인 경쟁력**

네트워크 침입 탐지와 클릭 예측 모두 테이블 데이터를 다루는 분류 문제이지만, 최적의 접근법은 구체적인 데이터 특성과 요구사항에 따라 달라질 수 있다는 점을 다시 한번 확인했다.

## 결론

이 연구는 머신러닝 모델 선택에 있어서 성급한 일반화를 피해야 한다는 점을 보여준다. Random Forest 같은 전통적 방법이 여전히 강력한 성능을 보일 수 있고, 딥러닝은 상황에 따라 과도한 복잡성을 가져올 수 있다.

내 CTR 경진대회에서도 FT-Transformer 외에 Random Forest나 다른 앙상블 방법들을 더 체계적으로 비교해보는 것이 좋을 것 같다. 때로는 단순한 해결책이 더 효과적일 수 있다는 교훈을 얻었다.

---

**논문 참조:** Ali, M.L.; Thakur, K.; Schmeelk, S.; Debello, J.; Dragos, D. Deep Learning vs. Machine Learning for Intrusion Detection in Computer Networks: A Comparative Study. *Applied Sciences* 2025, 15, 1903. [https://doi.org/10.3390/app15041903](https://doi.org/10.3390/app15041903)
