---
tags: Python, Vision-LLM, Qwen2.5-VL, EasyOCR, RAG, FastAPI, React
date: 2025
icon: 🧾
title: "[WIP] ReceiptVerify: AI-based Receipt Verification & Anomaly Detection System"
description: An AI-based web application that automatically verifies various types of receipts and detects anomalies/fraud. Specialized in verifying legal proof documents in South Korea, including tax invoices, invoices, and cash receipts, with automated validation of qualified evidence requirements based on the Corporate Tax Act and Value Added Tax Act, and provides legal justification through a RAG system. Provides 95%+ accuracy field extraction using Vision-LLM Only architecture with Qwen2.5-VL (4bit) + EasyOCR, and also automatically detects language and currency to process multi-language receipts from Korea, US, Japan, China, etc.
---

## Project Overview

A system developed to automate receipt processing for sole proprietors and corporations in South Korea, and to detect forged/falsified receipts or receipts that do not meet qualified evidence requirements. Performs fast and accurate field extraction using Qwen2.5-VL (4bit quantization) + EasyOCR-based Vision-LLM Only pipeline, automatically determines whether qualified evidence requirements are violated through law-based validation rules and a RAG system, and presents legal justification.

**Project Period:** November 10, 2025 ~ Present (In Progress)

**v2.0.0 Major Changes:** Transitioned from PaddleOCR Fine-tuning based Hybrid system to Vision-LLM Only architecture. No training required, reduced VRAM requirements (7GB → 4GB), added multi-language support.

<details>
<summary><b>Demo Screenshots (Click to expand)</b></summary>

![Receipt Upload](/projects/assets/images/07/en01.png)

![Extraction Results](/projects/assets/images/07/en02.png)

![Anomaly Detection Results](/projects/assets/images/07/en03.png)

<div style="display: flex; gap: 1rem; align-items: flex-start;">
  <img src="/projects/assets/images/07/en04.png" alt="Multi-language Support" style="flex: 1; max-width: 50%; height: auto;" />
  <img src="/projects/assets/images/07/en05.png" alt="History" style="flex: 1; max-width: 50%; height: auto;" />
</div>

</details>

## System Architecture

Simplified from the existing Hybrid OCR system (PaddleOCR → VLM Fallback) to a single Vision-LLM pipeline.

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

### Document Types (4 types)

| Type | Description | Examples |
|------|-------------|----------|
| `korean_tax_invoice_traditional` | Traditional paper tax invoice | Yellow/gray header |
| `korean_tax_invoice_electronic` | Electronic/delegated tax invoice | Red/blue border |
| `simple_receipt` | Simple receipts, card slips, cash receipts | Convenience stores, cafes, restaurants |
| `multi_language` | Foreign receipts | US, JP, CN, etc. |

### Country Detection System

Automatically detects country/currency through OCR text analysis.

| Detection Factor | Pattern Examples |
|------------------|------------------|
| Currency symbols | ₩→KR, $→US, ¥→JP/CN, €→EU, £→GB |
| Address patterns | Seoul/Busan→KR, State/Street→US, 都/県/市→JP |
| Business numbers | XXX-XX-XXXXX→KR, XX-XXXXXXX→US |
| Language detection | 한글→KR, 日本語→JP, 简体中文→CN |

Accurate country identification based on the above 4 criteria scoring.

### 2-Stage Validation System

**Stage 1: Anomaly Score Calculation** (All documents)

- Template-based validation (50%): Required fields, format validation
- Rule-based validation (30-50%): Amount range, date/time, business logic
- LLM validation (40%, conditional): Context-based reasoning (GPT-4o-mini)
- Result: Anomaly score 0-100 and risk classification (Low/Medium/High)

**Stage 2: Legal Explanation Generation** (Anomaly cases only)

- Trigger: Anomaly score ≥ 30 AND legal proof document, OR user request
- RAG search: Laws (Value Added Tax Act, Corporate Tax Act, Income Tax Act), National Tax Service FAQ
- LLM explanation generation: Legal justification, violation explanation, improvement suggestions

### RAG/IR System

**Knowledge Base:**

- Legal documents: 50+ articles (Value Added Tax Act, Corporate Tax Act, Income Tax Act)
- National Tax Service FAQ: 20+ items (expenditure evidence, qualified evidence)
- Rulebook: Required/recommended fields by receipt type
- Casebook: 10+ case studies

**Implementation:**

- OpenAI text-embedding-3-small (1536 dim) + FAISS IndexFlatL2
- Metadata filtering (source_type, topics, receipt_types)
- Hybrid search (Vector + Metadata)

## Key Technologies and Implementation

### Technologies Used

**Backend**

- Python 3.10+, FastAPI
- **Vision-LLM**: Qwen2.5-VL-7B-Instruct (4bit, ~4GB VRAM)
- **OCR**: EasyOCR (Korean + English, GPU)
- **LLM Verification**: GPT-4o-mini (optional)
- **RAG**: OpenAI embeddings + FAISS
- PostgreSQL

**Frontend**

- React 18, TypeScript, Vite
- Tailwind CSS
- react-i18next (Korean/English)

**MLOps**

- Docker + Docker Compose (NVIDIA GPU support)
- Microservices: backend-main (8000), backend-vision (8002)

### Docker Service Structure

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

### Core Implementations

**1. Vision-LLM Field Extraction**

- Qwen2.5-VL-7B-Instruct (4bit quantization)
- Few-shot prompting: 10+ real Korean receipt examples + law-based field definitions
- Separate prompts by document type (tax_invoice_kr, simple_receipt, multilang, pharmacy_receipt_kr)
- Processing time: 2-3s/image (RTX 3090)

**2. Bbox Matching System**

- Matching EasyOCR bbox results with Vision-LLM extracted values
- Fuzzy matching (threshold: 0.7)
- Field normalization: Amount (remove commas), Date (YYYY-MM-DD), Business number (remove hyphens)
- Frontend visualization of extracted field positions

**3. Anomaly Detection System**

- Data constraints: Only normal receipts exist, 0 fraudulent receipts
- ML-based approach not feasible → Rule-based + Zero-shot LLM approach
- 2-Stage Validation: Separate Score calculation (Stage 1) and Legal explanation (Stage 2)

**4. RAG-based Legal Explanation**

- Law crawling: Corporate Tax Act, Income Tax Act, Value Added Tax Act articles
- National Tax Service FAQ crawling: Expenditure evidence, qualified evidence
- FAISS index creation and Retriever implementation

## Performance

| Metric | Qwen2.5-VL (4bit) + EasyOCR |
|--------|----------------------------|
| **Field Extraction Accuracy** | 95%+ |
| **Processing Time** | 2-3s/image |
| **Cost** | $0 (local GPU) |
| **VRAM** | ~4-5GB |
| **Document Types** | 4 types auto-detection |
| **Multi-language** | KR, US, JP, CN |
| **Bbox Extraction** | EasyOCR |

## Architecture Change History

### v1.x → v2.0.0 Transition Reasons

**v1.x Hybrid OCR Limitations:**

- Insufficient data for PaddleOCR Fine-tuning (trained on 1,989 CORD/SROIE images)
- 2-Stage pipeline complexity (OCR → Confidence → Fallback)
- High VRAM requirements (~7-9GB for 8bit)
- Korean-centric, limited multi-language support

**v2.0.0 Vision-LLM Only Advantages:**

- No Fine-tuning required (replaced with Few-shot prompting)
- Single pipeline simplification
- Reduced VRAM requirements (4bit: ~4GB)
- Automatic multi-language detection (KR, US, JP, CN)
- Automatic document type classification

### Deprecated

Components removed in v2.0.0:

- PaddleOCR fine-tuned models
- OCR training scripts/configs
- Detection/Recognition evaluation scripts
- backend-ocr microservice
