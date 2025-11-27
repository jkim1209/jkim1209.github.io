---
tags: Python, Vision-LLM, Qwen2.5-VL, EasyOCR, RAG, FastAPI, React
date: 2025
icon: 🧾
title: [진행중] ReceiptVerify: AI 기반 영수증 검증 및 이상탐지 시스템
description: 다양한 종류의 영수증을 자동으로 검증하고 이상/허위를 탐지하는 AI 기반 웹 애플리케이션입니다. 특히 대한민국의 세금계산서, 계산서, 현금영수증 등 법적 증빙 서류에 대한 검증에 특화되어 있으며, 법인세법·부가가치세법 기반 적격증빙 요건 자동 검증과 RAG 시스템을 통한 법적 근거 제시 기능을 제공합니다. Qwen2.5-VL (4bit) + EasyOCR 기반 Vision-LLM Only 아키텍처로 95%+ 정확도의 필드 추출을 제공하며, 기타 한국·미국·일본·중국 등 다국어 영수증도 언어 및 화폐 단위를 자동 감지하여 처리합니다.
---

## 프로젝트 개요

개인사업자 및 기업의 영수증 처리 업무를 자동화하고 위변조 또는 적격증빙 요건 미충족 영수증을 탐지하기 위해 개발된 시스템입니다. Qwen2.5-VL (4bit quantization) + EasyOCR 기반 Vision-LLM Only 파이프라인으로 빠르고 정확한 필드 추출을 수행하고, 법령 기반 검증 규칙과 RAG 시스템을 통해 적격증빙 요건 위반 여부를 자동으로 판단하며 법적 근거를 제시합니다.

**프로젝트 기간:** 2025년 11월 10일 ~ 현재 (진행중)

**v2.0.0 주요 변경:** PaddleOCR Fine-tuning 기반 Hybrid 시스템에서 Vision-LLM Only 아키텍처로 전환. 학습 불필요, VRAM 요구사항 감소 (7GB → 4GB), 다국어 지원 추가.

<details>
<summary><b>데모 스크린샷 (클릭하여 펼치기)</b></summary>

![영수증 업로드](/projects/assets/images/07/ko01.png)

![추출 결과](/projects/assets/images/07/ko02.png)

![이상탐지 결과](/projects/assets/images/07/ko03.png)

<div style="display: flex; gap: 1rem; align-items: flex-start;">
  <img src="/projects/assets/images/07/ko04.png" alt="다국어 지원" style="flex: 1; max-width: 50%; height: auto;" />
  <img src="/projects/assets/images/07/ko05.png" alt="히스토리" style="flex: 1; max-width: 50%; height: auto;" />
</div>

</details>

## 시스템 아키텍처

v2.0.0: 기존 Hybrid OCR 시스템(PaddleOCR → VLM Fallback)에서 단일 Vision-LLM 파이프라인으로 단순화했습니다.

```markdown
                        User Upload Image
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Vision Service (backend-vision:8002)                  │
│              Qwen2.5-VL-7B (4bit) + EasyOCR                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 1: EasyOCR (bbox extraction)                        │  │
│  │    • Korean + English, GPU accelerated                    │  │
│  │    • OCR text → Country hints (₩, Seoul, etc.)            │  │
│  │    • Output: [{text, bbox, confidence}, ...]              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 2: Qwen2.5-VL (4bit, ~4GB VRAM)                     │  │
│  │    • Auto document type detection (4 types)               │  │
│  │    • Country/Currency auto detection                      │  │
│  │    • Few-shot prompting (10+ examples)                    │  │
│  │    • JSON field extraction (2-3s)                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Step 3: Bbox Matcher                                     │  │
│  │    • Fuzzy matching (threshold: 0.7)                      │  │
│  │    • Field normalization (amount, date, business no.)     │  │
│  │    • Frontend bbox visualization coordinates              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    HTTP (httpx async client)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Main API (backend:8000)                            │
│              FastAPI + PostgreSQL                               │
│                                                                 │
│    • Template Validation (4 base templates)                     │
│    • Rule-based Anomaly Detection (score 0-1)                   │
│    • LLM Verification (GPT-4o-mini, optional)                   │
│    • RAG Legal Explanation (FAISS + OpenAI)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Base Document Types (4가지)

| Type | 설명 | 예시 |
|------|------|------|
| `korean_tax_invoice_traditional` | 전통적 종이 세금계산서 | 노란색/회색 헤더 |
| `korean_tax_invoice_electronic` | 전자/위수탁 세금계산서 | 빨간색/파란색 테두리 |
| `simple_receipt` | 간이영수증, 카드전표, 현금영수증 | 편의점, 카페, 식당 |
| `multi_language` | 해외 영수증 | US, JP, CN 등 |

### Country Detection System

OCR 텍스트 분석을 통해 국가/통화를 자동 감지합니다.

| 감지 요소 | 패턴 예시 |
|-----------|-----------|
| 통화 기호 | ₩→KR, $→US, ¥→JP/CN, €→EU, £→GB |
| 주소 패턴 | 서울/부산→KR, State/Street→US, 都/県/市→JP |
| 사업자번호 | XXX-XX-XXXXX→KR, XX-XXXXXXX→US |
| 언어 감지 | 한글→KR, 日本語→JP, 简体中文→CN |

위 4단계 기준 scoring 으로 정확한 국가 판별.

### 2-Stage Validation System

**Stage 1: Anomaly Score Calculation** (전체 문서 처리)

- 템플릿 기반 검증 (50%): 필수 필드, 형식 검증
- 룰 기반 검증 (30-50%): 금액 범위, 날짜/시간, 비즈니스 로직
- LLM 검증 (40%, 조건부): 컨텍스트 기반 추론 (GPT-4o-mini)
- 결과: 이상 점수 0-100 및 위험도 분류 (Low/Medium/High)

**Stage 2: Legal Explanation Generation** (이상 케이스만)

- 트리거: 이상 점수 ≥ 30 이면서 법적 증빙 서류이거나 또는 사용자 요청 시
- RAG 검색: 법령(부가가치세법, 법인세법, 소득세법), 국세청 FAQ
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
- **Vision-LLM**: Qwen2.5-VL-7B-Instruct (4bit, ~4GB VRAM)
- **OCR**: EasyOCR (Korean + English, GPU)
- **LLM Verification**: GPT-4o-mini (선택적)
- **RAG**: OpenAI embeddings + FAISS
- PostgreSQL

**Frontend**

- React 18, TypeScript, Vite
- Tailwind CSS
- react-i18next (한국어/영어)

**MLOps**

- Docker + Docker Compose (NVIDIA GPU support)
- Microservices: backend-main (8000), backend-vision (8002)

### Docker 서비스 구조

```markdown
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                       │
├─────────────┬─────────────┬────────────┬────────────────┤
│   frontend  │   backend   │  backend   │      db        │
│   :3000     │   :8000     │  -vision   │   PostgreSQL   │
│   (React)   │   (FastAPI) │   :8002    │   :5432        │
│             │             │  (GPU)     │                │
└─────────────┴──────┬──────┴─────┬──────┴────────────────┘
                     │            │
                     └────────────┘
                   HTTP (receipt_network)
```

### 핵심 구현

**1. Vision-LLM Field Extraction**

- Qwen2.5-VL-7B-Instruct (4bit quantization)
- Few-shot prompting: 10+ 실제 한국 영수증 예시 + 법령 기반 필드 정의
- 문서 타입별 프롬프트 분리 (tax_invoice_kr, simple_receipt, multilang, pharmacy_receipt_kr)
- 처리 시간: 2-3초/이미지 (RTX 3090 기준)

**2. Bbox Matching System**

- EasyOCR bbox 결과와 Vision-LLM 추출값 매칭
- Fuzzy matching (threshold: 0.7)
- 필드별 정규화: 금액 (쉼표 제거), 날짜 (YYYY-MM-DD), 사업자번호 (하이픈 제거)
- Frontend에서 추출 필드 위치 시각화 제공

**3. 이상탐지 시스템**

- 데이터 제약사항: 정상 영수증만 존재, 위조 영수증 0개
- ML 기반 불가능 → Rule-based + Zero-shot LLM 접근
- 2-Stage Validation: Score 계산(Stage 1) + 법적 설명(Stage 2) 분리

**4. RAG 기반 Legal Explanation**

- 법령 크롤링: 법인세법, 소득세법, 부가가치세법 조문
- 국세청 FAQ 크롤링: 지출증빙, 적격증빙 관련
- FAISS 인덱스 생성 및 Retriever 구현

## 성능

| 구분 | Qwen2.5-VL (4bit) + EasyOCR |
|------|----------------------------|
| **Field Extraction Accuracy** | 95%+ |
| **처리 시간** | 2-3초/이미지 |
| **비용** | $0 (로컬 GPU) |
| **VRAM** | ~4-5GB |
| **Document Types** | 4가지 자동 감지 |
| **Multi-language** | KR, US, JP, CN |
| **Bbox Extraction** | EasyOCR |

## 아키텍처 변경 히스토리

### v1.x → v2.0.0 전환 이유

**v1.x Hybrid OCR의 한계:**

- PaddleOCR Fine-tuning에 필요한 데이터 부족 (CORD/SROIE 1,989장 학습)
- 2-Stage 파이프라인 복잡성 (OCR → Confidence → Fallback)
- 높은 VRAM 요구 (~7-9GB for 8bit)
- 한국어 중심, 다국어 지원 제한적

**v2.0.0 Vision-LLM Only의 장점:**

- Fine-tuning 불필요 (Few-shot prompting으로 대체)
- 단일 파이프라인 단순화
- VRAM 요구사항 감소 (4bit: ~4GB)
- 다국어 자동 감지 (KR, US, JP, CN)
- 문서 타입 자동 분류

### Deprecated

v2.0.0에서 제거된 컴포넌트:

- PaddleOCR fine-tuned models
- OCR training scripts/configs
- Detection/Recognition evaluation scripts
- backend-ocr microservice
