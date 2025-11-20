---
tags: Python, PyTorch, NLP, LLM, QLoRA, Dialogue Summarization
date: 2025
icon: 💬
---

# NLP/LLM Project: 일상 대화 요약

## 프로젝트 개요

일상 대화의 핵심을 정확하고 간결하게 요약하는 한국어 대화 요약 모델 개발을 목표로 한 프로젝트입니다.

GitHub Repository: [https://github.com/jkim1209/NLI-Dialogue-Summarization](https://github.com/jkim1209/NLI-Dialogue-Summarization)

Presentation Slides: [Google Slides](https://docs.google.com/presentation/d/1n3ZpdBC2U84vmS9k6mGyuqrokve7uZQvnuWrouwvFv4)

## 나의 역할과 기여도

팀장으로서 프로젝트를 총괄하며, LLM(거대 언어 모델) 모델링 파트를 주도적으로 담당했습니다.

- 데이터 전략 수립: Paraphrase, Speaker Swap 등 데이터 증강 기법을 기획하고 적용하여 모델의 일반화 성능을 높이는 데 기여했습니다.
- LLM 모델링 및 파인튜닝: Decoder-only 아키텍처 (SOLAR, Qwen3 등) 을 선정하고, QLoRA 기법을 활용해 제한된 리소스 환경에서 효율적인 미세조정을 진행했습니다.
- 프롬프트 엔지니어링: 모델이 최적의 요약문을 생성할 수 있도록 다양한 프롬프트 구조를 설계하고 테스트했습니다.
- 성능 분석 및 개선: T5 모델과 LLM의 ROUGE 점수 및 생성 결과물의 질적 차이를 비교 분석하여 최종 모델을 선정하였습니다.

## 주요 기술 및 구현 내용

### 사용 기술

- Framework: Python, PyTorch, Transformers, PEFT, TRL, BitsAndBytes
- Models: T5, KoBART, SOLAR 10.7B, Qwen3 0.6B

### 핵심 구현

- 데이터 증강 파이프라인 구축 (Paraphrase, Speaker Swap, Synthetic Generation)
- Encoder-Decoder (T5, KoBART) 및 Decoder-only (LLM) 모델 비교 실험
- QLoRA 기법을 이용한 4-bit 양자화 및 미세조정

![데이터 증강 파이프라인](/projects/assets/images/05/01.png)

## Troubleshooting

### 문제: LLM 파인튜닝 시 VRAM 부족 문제

**문제 상황**

기존의 Encoder-Decoder 모델과 달리 LLM 모델은 한글 학습에 더 많은 토큰 수를 필요로 하는데, 여기에 요약을 지시하는 프롬프트까지 함께 입력으로 받아 토큰 수가 급증하게 됩니다. 이로 인해 QLoRA로 파인튜닝하는 과정에서 VRAM(24GB) 부족으로 인한 OOM 에러 및 커널 중단 현상이 반복적으로 발생했습니다.

**해결 과정**

1. **배치 크기 최소화**:. 배치 크기 최소화: Batch Size를 1로 설정하여 단일 학습 데이터가 차지하는 메모리 점유율을 최소화했습니다. 이 때 Gradient Accumulation 기법을 적용하여 전과 같은 배치 사이즈로 학습하는 것과 같은 효과를 얻었습니다.
2. **입력 길이 최적화**:. 입력 길이 최적화: max_token 길이를 대부분의 대화 길이를 커버하는 4096으로 줄이고, 지나치게 긴 대화를 학습할 때는 zero-shot이 적용되게 하였습니다.
3. 결과적으로 제한된 환경에서도 성공적으로 파인튜닝을 수행할 수 있었습니다.

## 성과 및 결과

### 정량적 성과

일상 대화 요약 대회 **최종 1위 달성**

- Private ROUGE: 47.9550
- Baseline ROUGE: 15.8301

### 경험 및 교훈

- QLoRA를 활용하여 개인용 GPU 같은 제한된 리소스 환경에서도 거대 언어 모델을 성공적으로 미세조정하는 경험을 쌓았습니다.
- LLM 파인튜닝 과정에서 ROUGE 점수라는 정량적 지표와 실제 사람이 느끼는 요약문의 자연스러움이라는 정성적 품질 사이의 균형을 맞추는 것의 중요성을 배웠습니다.
- 모델링 과정 전반에서 팀원들과 다양한 접근법을 논의하며, 객관적 근거를 바탕으로 최적의 방향을 결정하는 협업 의사결정 능력을 키웠습니다.

![대회 결과](/projects/assets/images/05/02.png)
