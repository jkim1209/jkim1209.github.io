---
tags: Python, OCR, PaddleOCR, Vision-LLM, RAG, FastAPI, React
date: 2025
icon: 🧾
---

# [진행중] ReceiptVerify: AI 기반 영수증 검증 및 이상탐지 시스템

## 프로젝트 개요

개인사업자 및 기업의 영수증 처리 업무를 자동화하고 위변조 또는 적격증빙 요건 미충족 영수증을 탐지하기 위해 개발된 시스템입니다. Fine-tuned PaddleOCR과 Qwen2.5-VL를 이용하여 빠른 필드 추출을 수행하고, 법령 기반 검증 규칙과 RAG 시스템을 통해 적격증빙 요건 위반 여부를 자동으로 판단하며 법적 근거를 제시합니다. 세금계산서 등 법적 증빙 서류뿐만 아니라 편의점·음식점 등 일반 영수증도 레이아웃에 관계없이 처리할 수 있습니다.

**프로젝트 기간:** 2025년 11월 10일 ~ 현재 (진행중)

## 시스템 아키텍처

### Hybrid OCR System

영수증 필드 추출의 정확도와 속도를 동시에 달성하기 위한 2단계 시스템입니다.

**Stage 1: Fine-tuned PaddleOCR**

- PP-OCRv3 MobileNetV3 모델을 CORD, SROIE, Custom 데이터셋(총 1,989장)으로 파인튜닝
- Detection Hmean 69.83%, Recognition Accuracy 91.06% 달성
- 처리 속도: ~500ms/image

**Stage 2: Vision-LLM Fallback**

- Qwen2.5-VL-7B-Instruct 8bit 모델 활용 (로컬 GPU 추론)
- 신뢰도 평가 시스템(7가지 요인)으로 0.6 미만일 때 자동 전환
- Few-shot prompting으로 95%+ 정확도 목표, ~3초 처리

**Smart Fallback 로직**

```
PaddleOCR 추출 → 신뢰도 평가 (7가지 요인)
├─ 신뢰도 > 0.6: PaddleOCR 결과 사용 (~500ms)
└─ 신뢰도 ≤ 0.6: Vision-LLM 재추출 (~3s)
```

### 2-Stage Validation System

**Stage 1: Anomaly Score Calculation** (전체 문서 처리)

- 템플릿 기반 검증 (50%): 필수 필드, 형식 검증
- 룰 기반 검증 (30-50%): 금액 범위, 날짜/시간, 비즈니스 로직
- LLM 검증 (40%, 조건부): 컨텍스트 기반 추론 (GPT-4o-mini)
- 결과: 이상 점수 0-100 및 위험도 분류 (Low/Medium/High)

**Stage 2: Legal Explanation Generation** (이상 케이스만)

- 트리거: 이상 점수 ≥ 30 또는 사용자 요청 시
- RAG 검색: 법령(부가가치세법, 법인세법, 소득세법), 국세청 FAQ, Casebook
- LLM 설명 생성: 법적 근거 제시, 위반 사항 설명, 개선 방안 제시

### RAG/IR System

**Knowledge Base:**

- 법령 문서: 50+ 조문 (부가가치세법, 법인세법, 소득세법)
- 국세청 FAQ: 20+ 개 (지출증빙, 적격증빙 관련)
- Rulebook: 영수증 타입별 필수/권장 필드
- Casebook: 10+ 케이스 스터디

**구현:**

- OpenAI text-embedding-3-small (1536 dim) + FAISS IndexFlatL2
- Metadata filtering (source_type, topics, receipt_types)
- Hybrid search (Vector + Metadata)

## 주요 기술 및 구현 내용

### 사용 기술

**Backend**

- Python 3.10+, FastAPI
- PaddleOCR PP-OCRv3 MobileNetV3 (Fine-tuned)
- Qwen2.5-VL-7B-Instruct 8bit (~7-9GB VRAM)
- GPT-4o-mini (LLM 검증, 선택적)
- OpenAI embeddings + FAISS (RAG)

**Frontend**

- React 18, TypeScript, Vite
- Tailwind CSS
- react-i18next (한국어/영어)

**MLOps**

- Docker + Docker Compose
- PaddlePaddle GPU 2.6.1, CUDA 11.7/11.8
- WANDB (실험 추적)

### 핵심 구현

**1. PaddleOCR Fine-tuning**

- 통합 데이터셋 구축: CORD 1,000장 + SROIE 626장 + Custom 17장
- Detection: PP-OCRv3 MobileNetV3, 150 epochs, Early Stopping @ 122
- Recognition: PP-OCRv5 Korean, 100 epochs, Early Stopping @ 20
- No Overfitting 달성 (Val-Test diff < 3%)

**2. Hybrid OCR System**

- 신뢰도 평가 시스템 (7가지 요인): 필수 필드 누락, OCR 블록 수, 검증 실패, 비현실적 금액 등
- Smart Fallback 로직: confidence ≤ 0.6 시 자동 전환
- Vision-LLM Few-shot Prompting: 6개 실제 한국 영수증 예시 + 한국 세법 준수

**3. 이상탐지 시스템**

- 데이터 제약사항: 정상 영수증 1,643개만 존재, 위조 영수증 0개
- ML 기반 불가능 → Rule-based + Zero-shot LLM 접근
- 2-Stage Validation: Score 계산(Stage 1) + 법적 설명(Stage 2) 분리

**4. RAG 기반 Legal Explanation**

- 법령 크롤링: 법인세법, 소득세법, 부가가치세법 조문
- 국세청 FAQ 크롤링: 지출증빙, 적격증빙 관련
- 문서 전처리: JSON 변환, 청킹, 메타데이터 정리
- FAISS 인덱스 생성 및 Retriever 구현

## 주요 개선 사항 및 문제 해결

**1. OCR Fine-tuning**

- Pretrained 모델 대비 Detection Hmean 20% 향상 (50% → 70%)
- Recognition Accuracy 11% 향상 (80% → 91%)
- Overfitting 방지 (Validation-Test 차이 3% 미만)

**2. Hybrid OCR System**

- 속도와 정확도의 균형: 평균 처리 시간 <1초, 정확도 92-95% 목표
- 비용 효율: PaddleOCR + Qwen2.5-VL 로컬 추론으로 $0 달성
- Smart Fallback으로 Vision-LLM 사용률 <50% 유지

**3. 데이터 부족 문제 해결**

- ML 기반 이상탐지 불가능 (위조 영수증 데이터 0개)
- Rule-based + Zero-shot LLM으로 대안 제시
- 도메인 지식 및 법령 기반 검증으로 신뢰성 확보
