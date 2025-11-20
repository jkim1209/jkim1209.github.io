---
tags: Python, PyTorch, FT-Transformer, LSTM, CTR Prediction
date: 2025
icon: 📊
---

# CTR Prediction Project: Toss 광고 클릭률 예측

## 프로젝트 개요

복잡한 Tabular 데이터셋에서 사용자의 행동 패턴을 분석하여 광고 클릭 확률 예측 모델을 개발하는 프로젝트입니다.

GitHub Repository: [https://github.com/jkim1209/toss-ctr-prediction](https://github.com/jkim1209/toss-ctr-prediction)

## 나의 역할과 기여도

데이터 분석 및 모델링 파트를 담당하여, 예측 성능을 끌어올리는 핵심 역할을 수행했습니다.

- 데이터 분석 및 전처리: 초기 데이터 분석을 통해 수많은 피처 간의 복잡한 상관관계를 파악하고, 이를 바탕으로 프로젝트의 핵심인 전처리 및 피처 엔지니어링 전략을 수립하고 리드했습니다.
- 모델 구현 및 실험: FT-Transformer를 메인 모델로 선정하고, LSTM을 결합하여 시퀀스 정보의 잠재적 특징을 추출하는 하이브리드 모델 구조를 구현했습니다.

## 주요 기술 및 구현 내용

### 사용 기술

- Framework: Python, PyTorch, Polars, Pandas, NumPy, Scikit-learn
- Models: FT-Transformer, LSTM

### 핵심 구현

- Parquet 형식의 대용량 데이터 처리 및 피처 엔지니어링 파이프라인
- FT-Transformer 기반의 분류 모델 학습 및 5-Fold 교차검증
- LSTM을 이용한 사용자 행동 시퀀스 임베딩 생성

![모델 아키텍처](/projects/assets/images/04/01.png)

![피처 엔지니어링](/projects/assets/images/04/02.png)

![학습 과정](/projects/assets/images/04/03.png)

## Troubleshooting

### 문제: 성능 정체 구간 돌파

**문제 상황**

모델 학습 초기, 평가지표 점수가 약 0.3450에서 더 이상 향상되지 않는 정체 구간에 부딪혔습니다. 당시 사용자 행동 시퀀스 정보는 시퀀스의 시작, 끝, 길이 등 단순한 통계량으로만 활용되고 있었습니다.

**해결 과정**

1. 시퀀스 데이터가 담고 있는 순차적, 동적 정보를 모델이 충분히 활용하지 못하고 있다고 판단했습니다.
2. LSTM 모델을 도입하여, 각 사용자의 행동 시퀀스 전체를 압축된 하나의 임베딩 벡터로 변환했습니다. 이 임베딩 벡터는 시퀀스의 순서와 패턴 정보를 함축하고 있으며, 다른 정적 피처들과 함께 모델에 입력됩니다.
3. 그 결과 평가지표 점수가 0.3480 이상으로 상승하며 성능 정체 구간을 돌파했습니다.

## 성과 및 결과

### 정량적 성과

토스 CTR 예측 경진대회 최종 상위 10% 달성 (70위)

- Private Score: 0.34814
- Baseline Score: 0.33185

*Score = Average Precision과 Weighted Log Loss의 평균

### 경험 및 교훈

- 대용량 Tabular 데이터에 최신 딥러닝 아키텍처인 FT-Transformer를 적용하고, 대용량 데이터 처리를 위한 재현 가능한 실험 파이프라인을 구축하는 전체 과정을 경험했습니다.
- 시계열이나 순서 정보가 중요한 데이터에서 LSTM과 같은 모델을 피처 추출기로 활용하는 피처 엔지니어링이 모델 성능에 큰 영향을 미칠 수 있음을 직접 확인했습니다.

![최종 결과](/projects/assets/images/04/04.png)
