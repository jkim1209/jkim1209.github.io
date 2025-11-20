---
tags: Python, LangChain, LangGraph, AI Agent, RAG, Finance
date: 2025
icon: 🤖
---

# AI Multi-Agent Project: 금융 챗봇

## 프로젝트 개요

LangChain/LangGraph를 기반으로 4개의 전문화된 에이전트가 협업하는 금융 분석 멀티 에이전트 시스템을 구축하는 프로젝트입니다. RAG(Retrieval-Augmented Generation)를 통한 금융 용어 검색과 실시간 금융 데이터 수집을 통합하여 주식 분석, 비교, 보고서 자동 생성 기능을 제공하며, LLM Judge 기반 품질 평가로 답변 신뢰성을 보장합니다.

GitHub Repository: [https://github.com/jkim1209/ai_agent_project](https://github.com/jkim1209/ai_agent_project)

Presentation Slides: [Google Drive](https://drive.google.com/file/d/1wtD5dT1Mg_HFqJGC8Mj2lduqjFYTcFIH/view)

## 나의 역할과 기여도

팀장으로서 프로젝트의 전체 시스템 아키텍처를 설계하고 총괄하며, 핵심 에이전트 및 도구 개발, 모듈 통합 및 배포를 담당했습니다.

- 시스템 아키텍처 설계: LangGraph 기반 State Machine 워크플로우를 설계하여 7개 노드가 유기적으로 연결되도록 구성했습니다.
- 에이전트 및 도구 개발: 4개 중 2개의 에이전트(Financial Analyst, Report Generator)를 직접 개발하고, 그들이 사용하는 8개 도구(주식 정보 조회, 과거 데이터, 애널리스트 추천, 웹 검색, 차트 생성, 보고서 저장 등)을 구현했습니다.
- 시스템 통합 및 배포: 팀원들이 개발한 모듈들을 하나의 통합 시스템으로 구성하고, 실시간 인터랙션을 위한 Streamlit UI 구축, Render.com을 통해 배포하였습니다.
- 팀 협업 및 이슈 관리: GitHub Project의 칸반 보드를 활용해 태스크를 할당하고 진행 상황을 추적했으며, Notion을 통해 KPI, 마일스톤, 기술 문서를 체계적으로 관리했습니다. 팀원들이 개발한 모듈을 전체 워크플로우에 통합하고 충돌을 조율했습니다.

## 주요 기술 및 구현 내용

### 사용 기술

- Python
- LLM & AI Agent: LangChain, LangGraph, Upstage Solar Pro 2
- RAG & Vector DB: ChromaDB, BAAI/bge-m3 (HuggingFace Embeddings)
- Data Sources: yfinance (금융 데이터), Tavily (웹 검색)
- Database: SQLite (대화 히스토리)
- Frontend & Deployment: Streamlit, Render.com

### 핵심 구현

- LangGraph 기반 4개 전문화 에이전트 설계 (Request Analyst, Supervisor, Financial Analyst, Report Generator)
- Structured Output 기반 에이전트 간 JSON 데이터 전달 및 스키마 강제
- ChromaDB + BAAI/bge-m3 임베딩 기반 RAG 시스템 구축 (금융 관련 3개 PDF 임베딩)
- 8개 금융 도구 개발 (티커 검색, 주식 정보, 과거 가격, 애널리스트 추천, 웹 검색, 차트 생성, 보고서 저장)
- LLM Judge 품질 평가 시스템 (정확성·완전성·관련성·명확성 평가 및 자동 재시도)
- SQLite 기반 대화 히스토리 관리 및 멀티턴 대화 지원
- Supervisor 패턴 기반 동적 라우팅 (RAG 검색 vs 실시간 금융 분석 자동 분기)
- Streamlit 웹 UI 및 Render.com 배포

## 멀티 에이전트 아키텍처

![멀티 에이전트 아키텍처](/projects/assets/images/06/01.png)

## Troubleshooting

### 문제 1: Tool 호출 파싱 오류

**문제 상황**

에이전트가 Tool을 호출할 때 마크다운 코드 블록을 사용하거나 정해진 포맷을 지키지 않아 파싱 오류가 빈번하게 발생했습니다. 이는 ReAct 패턴만으로는 LLM이 자유 형식 텍스트로 출력하므로 포맷 준수가 보장되지 않기 때문이었습니다.

**해결 과정**

1. LangChain의 `with_structured_output()` 메서드를 도입하여 JSON 스키마를 강제했습니다.
2. Pydantic 모델로 출력 형식을 정의하고, 에이전트가 반드시 해당 형식으로 응답하도록 제약했습니다.
3. 결과적으로 Tool 호출 파싱 성공률이 99%로 향상되었습니다.

### 문제 2: 한국 기업 티커 심볼 오인식

**문제 상황**

한국 기업 티커 심볼을 오인식하는 문제가 발생했습니다. 예를 들어 "카카오 주식 분석해줘"라는 요청에 대해 한국 카카오(035720.KS) 대신 아르헨티나 카카오 원료 종목(CACAO.BA)을 검색합니다. 이는 "카카오"를 영어로 번역하면 "cacao"가 되어 yfinance에서 카카오 원료 관련 티커를 우선시하기 때문입니다.

**해결 과정**

1. 한국 주요 기업 약 50개의 티커 매핑 테이블을 추가했습니다.
2. 매핑 테이블에 없는 경우에도 한글로 입력하면 한국 주식 거래소에서 우선 검색하도록 설정하였습니다.
3. 그 결과 주요 한국 기업에 대한 티커 검색 정확도가 95% 이상으로 향상되었습니다.

### 문제 3: Yahoo Finance YTD 차트 파싱 실패

**문제 상황**

Yahoo Finance의 YTD 주가 차트를 파싱 실패하는 문제가 발생했습니다. 이는 Financial Analyst가 `to_string()` 메서드로 가변 공백 형식의 데이터를 전달했으나, Report Generator는 `sep='\s+'`로 파싱을 시도하여 컬럼 수가 불일치하였기 때문입니다.

**해결 과정**

1. 해당 데이터의 전달 형식을 CSV로 표준화하여 Financial Analyst가 `to_csv()` 메서드로 데이터를 전달하고, Report Generator는 `pd.read_csv()`로 파싱하도록 수정했습니다.
2. 그 결과 모든 차트가 정상적으로 생성되었습니다.

### 문제 4: 비교 분석 로직의 컨텍스트 처리 문제

**문제 상황**

비교 분석 로직이 이전 대화의 맥락을 올바르게 구분하지 못했습니다. 예를 들어 "삼성전자 주식 분석해줘" 이후 "LG전자는 어때?"라고 물어도 두 종목을 비교하려 합니다. 즉, 이전 대화와 새로운 요청 간의 관계를 판단하는 로직이 불안정했습니다.

**해결 과정**

1. 이전 분석 결과를 새 요청에서도 활용할 수 있도록, 분석 함수가 이전 대화의 종목 정보를 명시적으로 인식하고 전달받는 구조를 추가했습니다.
2. 확실하지 않은 비교 요청 시, 이전 분석에서 이미 다뤘던 종목을 우선 제외한 뒤 LLM을 이용하여 비교 요청인지 판단 후 비교 요청인 경우에만 이전 종목을 다시 추가해 비교 분석이 이루어지도록 수정했습니다.
3. 그 결과 비교/차이 분석 요청에 대해 80% 이상 의도대로 답변하는 것을 확인했습니다.

## 성과 및 결과

실시간 금융 질의응답, 단일/다중 종목 비교 분석, 자동 보고서 생성(MD/PDF/TXT), 차트 시각화 기능을 갖춘 웹 서비스를 성공적으로 배포했습니다. 또한 LLM Judge 품질 평가를 통해 답변 품질을 보장하며, 대화 히스토리 기반 멀티턴 대화 및 후속 질문 처리도 가능하도록 구현했습니다.

![결과 1](/projects/assets/images/06/02.png)

![결과 2](/projects/assets/images/06/03.png)
