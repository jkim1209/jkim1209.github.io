---
tags: Python, OCR, PaddleOCR, Vision-LLM, RAG, FastAPI, React
date: 2025
icon: 🧾
title: [WIP] AI-based Receipt Verification System
description: An AI-based web application that automatically verifies various types of receipts and detects anomalies/fraud. Specialized in verifying legal proof documents in South Korea, including tax invoices, invoices, and cash receipts, with automated validation of qualified evidence requirements based on the Corporate Tax Act and Value Added Tax Act, and provides legal justification through a RAG system. Provides 95%+ accuracy field extraction using a Hybrid OCR system combining Fine-tuned PaddleOCR and Qwen2.5-VL, and processes various receipt formats regardless of layout. Automatically verifies law-based validation items including amount consistency checks, business registration number format and checksum validation, and qualified evidence requirements for simple receipts, presenting relevant legal provisions and improvement suggestions when violations are detected.
---

## Project Overview

A system developed to automate receipt processing for sole proprietors and corporations, and to detect forged/falsified receipts or receipts that do not meet qualified evidence requirements. Performs rapid field extraction using Fine-tuned PaddleOCR and Qwen2.5-VL, automatically determines whether qualified evidence requirements are violated through law-based validation rules and a RAG system, and presents legal justification. Can process not only legal proof documents such as tax invoices, but also general receipts from convenience stores, restaurants, etc., regardless of layout.

**Project Period:** November 10, 2025 ~ Present (In Progress)

## System Architecture

### Hybrid OCR System

A two-stage system designed to achieve both accuracy and speed in receipt field extraction.

**Stage 1: Fine-tuned PaddleOCR**

- Fine-tuned PP-OCRv3 MobileNetV3 model with CORD, SROIE, Custom datasets (total 1,989 images)
- Achieved Detection Hmean 69.83%, Recognition Accuracy 91.06%
- Processing speed: ~500ms/image

**Stage 2: Vision-LLM Fallback**

- Utilizing Qwen2.5-VL-7B-Instruct 8bit model (local GPU inference)
- Automatic fallback when confidence < 0.6 via confidence evaluation system (7 factors)
- Target 95%+ accuracy with Few-shot prompting, ~3 seconds processing

**Smart Fallback Logic**

```bash
PaddleOCR Extraction → Confidence Evaluation (7 factors)
├─ Confidence > 0.6: Use PaddleOCR result (~500ms)
└─ Confidence ≤ 0.6: Re-extract with Vision-LLM (~3s)
```

### 2-Stage Validation System

**Stage 1: Anomaly Score Calculation** (All documents)

- Template-based validation (50%): Required fields, format validation
- Rule-based validation (30-50%): Amount range, date/time, business logic
- LLM validation (40%, conditional): Context-based reasoning (GPT-4o-mini)
- Result: Anomaly score 0-100 and risk classification (Low/Medium/High)

**Stage 2: Legal Explanation Generation** (Anomaly cases only)

- Trigger: Anomaly score ≥ 30 or user request
- RAG search: Laws (Value Added Tax Act, Corporate Tax Act, Income Tax Act), National Tax Service FAQ, Casebook
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
- PaddleOCR PP-OCRv3 MobileNetV3 (Fine-tuned)
- Qwen2.5-VL-7B-Instruct 8bit (~7-9GB VRAM)
- GPT-4o-mini (LLM validation, optional)
- OpenAI embeddings + FAISS (RAG)

**Frontend**

- React 18, TypeScript, Vite
- Tailwind CSS
- react-i18next (Korean/English)

**MLOps**

- Docker + Docker Compose
- PaddlePaddle GPU 2.6.1, CUDA 11.7/11.8
- WANDB (experiment tracking)

### Core Implementations

**1. PaddleOCR Fine-tuning**

- Integrated dataset: CORD 1,000 images + SROIE 626 images + Custom 17 images
- Detection: PP-OCRv3 MobileNetV3, 150 epochs, Early Stopping @ 122
- Recognition: PP-OCRv5 Korean, 100 epochs, Early Stopping @ 20
- Achieved No Overfitting (Val-Test diff < 3%)

**2. Hybrid OCR System**

- Confidence evaluation system (7 factors): Missing required fields, OCR block count, validation failures, unrealistic amounts, etc.
- Smart Fallback logic: Automatic fallback when confidence ≤ 0.6
- Vision-LLM Few-shot Prompting: 6 real Korean receipt examples + Korean tax law compliance

**3. Anomaly Detection System**

- Data constraints: Only 1,643 normal receipts exist, 0 fraudulent receipts
- ML-based approach not feasible → Rule-based + Zero-shot LLM approach
- 2-Stage Validation: Separate Score calculation (Stage 1) and Legal explanation (Stage 2)

**4. RAG-based Legal Explanation**

- Law crawling: Corporate Tax Act, Income Tax Act, Value Added Tax Act articles
- National Tax Service FAQ crawling: Expenditure evidence, qualified evidence
- Document preprocessing: JSON conversion, chunking, metadata organization
- FAISS index creation and Retriever implementation

## Technical Achievements

**1. OCR Fine-tuning**

- Detection Hmean improved 20% over Pretrained model (50% → 70%)
- Recognition Accuracy improved 11% (80% → 91%)
- Prevented Overfitting (Validation-Test difference < 3%)

**2. Hybrid OCR System**

- Balance between speed and accuracy: Average processing time <1 second, target accuracy 92-95%
- Cost efficiency: $0 achieved through PaddleOCR + Qwen2.5-VL local inference
- Maintained Vision-LLM usage rate <50% with Smart Fallback

**3. Data Shortage Problem Resolution**

- ML-based anomaly detection not feasible (0 fraudulent receipt data)
- Presented alternative with Rule-based + Zero-shot LLM
- Ensured reliability through domain knowledge and law-based validation
