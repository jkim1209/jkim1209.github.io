---
tags: Python, PyTorch, Computer Vision, ViT, Image Classification
date: 2025
icon: 📄
title: 문서 이미지 분류
description: 다양한 산업 도메인의 문서 이미지를 포함한 여러 이미지를 정확히 분류하는 모델을 개발합니다. ViT-SigLIP 백본과 클래스별 증강, Focal Loss, 고해상도 사전학습 모델을 결합하여 Macro F1 0.9692를 기록하며 리더보드 1위 성과를 달성했습니다.
---

## 프로젝트 개요

다양한 산업 도메인의 문서 이미지를 포함한 여러 이미지를 17개 클래스로 분류하는 모델을 개발하는 프로젝트입니다.

GitHub Repository: [https://github.com/jkim1209/doc_img_classification](https://github.com/jkim1209/doc_img_classification)

Presentation Slides: [Google Slides](https://docs.google.com/presentation/d/10o12igXX3xXg1zpI-M4KdHMrxI4OToEL/)

## 나의 역할과 기여도

팀장으로서 프로젝트 전반을 이끌며, 데이터 분석부터 모델링, 성능 최적화까지 핵심적인 역할을 수행했습니다.

- 데이터 분석 및 전처리: 클래스 불균형 문제를 파악하고, 문서/신분증/차량 등 종류에 따라 최적화된 데이터 증강 파이프라인을 설계 및 적용했습니다.
- 모델 구현 및 실험: ViT를 포함한 다양한 사전학습 모델을 실험하며 데이터 특성에 가장 적합한 아키텍처를 탐색했습니다.

## 주요 기술 및 구현 내용

### 사용 기술

- Framework: Python, PyTorch, Torchvision, timm, Albumentations, OpenCV
- Models: ViT, ConvNeXt, EfficientNet 등

### 핵심 구현

- 클래스 특성을 고려한 맞춤형 데이터 증강 파이프라인
- Focal Loss를 이용한 클래스 불균형 문제 해결
- 고해상도 이미지(512+)에 사전학습된 모델 도입
- 2단계 추론을 도입하여 1단계에서 불확실하게 분류된 샘플 재추론

<div style="display: flex; gap: 1rem; align-items: flex-start;">
  <img src="/projects/assets/images/04/01.png" alt="데이터 증강" style="flex: 1; max-width: 50%; height: auto;" />
  <img src="/projects/assets/images/04/02.png" alt="모델 구조" style="flex: 1; max-width: 50%; height: auto;" />
</div>

![실험 결과](/projects/assets/images/04/03.png)

## Troubleshooting

### 문제: 고해상도 데이터에서의 성능 한계 돌파

**문제 상황**

클래스별 맞춤 데이터 증강 기법을 적용했음에도 불구하고, Test Macro F1 Score가 약 0.85에서 더 이상 오르지 않는 한계에 직면했습니다.

**해결 과정**

1. 기존에 사용하던 모델들이 저해상도(224~384px) 이미지로 사전학습되어, 글자 등 세밀한 특징이 중요한 문서 이미지의 정보를 충분히 담지 못한다고 판단했습니다.
2. 동일 모델의 입력 해상도만 512px 이상으로 높여 학습했지만, 오히려 성능이 하락했습니다. 저해상도에 익숙한 모델이 고해상도 이미지의 노이즈에 더 민감하게 반응했기 때문입니다.
3. 가설을 수정하여, 처음부터 512px 이상의 고해상도 이미지로 사전학습된 모델을 도입했습니다. 그 결과 Test Macro F1 Score가 0.93까지 급상승했으며, 추가적인 데이터 정제 및 Focal Loss 적용으로 최종 0.95 이상의 점수를 달성했습니다.

## 성과 및 결과

### 정량적 성과

문서 이미지 분류 대회 최종 1위 달성

- Private Macro F1: 0.9692
- Baseline Macro F1: 0.1695

### 경험 및 교훈

- 컴퓨터 비전 프로젝트에서 가설을 세우고 검증하며 문제의 근본 원인을 찾아가는 능력을 길렀습니다.
- 사전학습(Pre-training) 환경과 실제 과제(Task)의 데이터 특성을 일치시키는 것이 모델 성능에 얼마나 중요한지, 그리고 모델 개선만큼이나 고품질의 데이터를 확보하는 것(Data-centric AI)이 중요하다는 것을 다시 한번 체감했습니다.
- 단순한 결과 개선보다 문제의 근본 원인을 논리적으로 추적하며, 가설 설정과 검증을 반복하는 분석적 사고력과 집요함을 기를 수 있었습니다.

![최종 결과](/projects/assets/images/04/04.png)
