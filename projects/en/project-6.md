---
tags: Python, LangChain, LangGraph, AI Agent, RAG, Finance
date: 2025
icon: 🤖
---

# AI Multi-Agent Project: Financial Chatbot

## Project Overview

This project builds a multi-agent financial analysis system where 4 specialized agents collaborate using LangChain/LangGraph. It integrates RAG (Retrieval-Augmented Generation) for financial terminology retrieval and real-time financial data collection, providing stock analysis, comparison, and automated report generation features while ensuring answer reliability through LLM Judge-based quality evaluation.

GitHub Repository: [https://github.com/jkim1209/ai_agent_project](https://github.com/jkim1209/ai_agent_project)

Presentation Slides: [Google Drive](https://drive.google.com/file/d/1wtD5dT1Mg_HFqJGC8Mj2lduqjFYTcFIH/view)

## My Role and Contributions

As team leader, I designed and oversaw the entire system architecture, led core agent and tool development, and handled module integration and deployment.

- System Architecture Design: Designed a LangGraph-based State Machine workflow with 7 nodes organically connected.
- Agent & Tool Development: Directly developed 2 out of 4 agents (Financial Analyst, Report Generator) and implemented 8 tools they use (stock information retrieval, historical data, analyst recommendations, web search, chart generation, report saving, etc.).
- System Integration & Deployment: Integrated modules developed by team members into a unified system, built Streamlit UI for real-time interaction, and deployed via Render.com.
- Team Collaboration & Issue Management: Assigned tasks and tracked progress using GitHub Project's Kanban board, systematically managed KPIs, milestones, and technical documentation through Notion, and coordinated team members' modules into the overall workflow while resolving conflicts.

## Key Technologies and Implementation

### Technologies Used

- Python
- LLM & AI Agent: LangChain, LangGraph, Upstage Solar Pro 2
- RAG & Vector DB: ChromaDB, BAAI/bge-m3 (HuggingFace Embeddings)
- Data Sources: yfinance (financial data), Tavily (web search)
- Database: SQLite (conversation history)
- Frontend & Deployment: Streamlit, Render.com

### Core Implementation

- Designed 4 specialized LangGraph-based agents (Request Analyst, Supervisor, Financial Analyst, Report Generator)
- Structured Output-based JSON data transfer between agents with enforced schema
- Built ChromaDB + BAAI/bge-m3 embedding-based RAG system (embedded 3 finance-related PDFs)
- Developed 8 financial tools (ticker search, stock info, historical prices, analyst recommendations, web search, chart generation, report saving)
- LLM Judge quality evaluation system (evaluating accuracy, completeness, relevance, clarity with automatic retry)
- SQLite-based conversation history management with multi-turn dialogue support
- Supervisor pattern-based dynamic routing (automatic branching between RAG search vs real-time financial analysis)
- Streamlit web UI and Render.com deployment

## Multi-Agent Architecture

![Multi-Agent Architecture](/projects/assets/images/01/en-01.png)

## Troubleshooting

### Problem 1: Tool Call Parsing Errors

**Problem Description**

When agents invoked tools, they frequently used markdown code blocks or didn't follow the defined format, resulting in parsing errors. This was because the ReAct pattern alone couldn't guarantee format compliance as LLMs output in free-form text.

**Solution**

1. Introduced LangChain's `with_structured_output()` method to enforce JSON schema.
2. Defined output formats using Pydantic models, constraining agents to respond in that format.
3. As a result, tool call parsing success rate improved to 99%.

### Problem 2: Korean Company Ticker Symbol Misrecognition

**Problem Description**

Korean company ticker symbols were being misrecognized. For example, when asked "Analyze Kakao stock," it searched for Argentine cacao commodity ticker (CACAO.BA) instead of Korean Kakao (035720.KS). This occurred because "Kakao" translates to "cacao" in English, leading yfinance to prioritize cacao commodity-related tickers.

**Solution**

1. Added a ticker mapping table for approximately 50 major Korean companies.
2. Even for companies not in the mapping table, configured the system to prioritize searching in the Korean stock exchange when input is in Korean.
3. As a result, ticker search accuracy for major Korean companies improved to over 95%.

### Problem 3: Yahoo Finance YTD Chart Parsing Failure

**Problem Description**

Yahoo Finance's YTD stock price charts failed to parse. This occurred because the Financial Analyst passed data using the `to_string()` method in a variable-space format, but the Report Generator attempted to parse with `sep='\s+'`, resulting in column count mismatches.

**Solution**

1. Standardized the data transfer format to CSV, so the Financial Analyst passes data using `to_csv()` and the Report Generator parses with `pd.read_csv()`.
2. As a result, all charts were generated successfully.

### Problem 4: Comparison Analysis Logic Context Handling Issue

**Problem Description**

The comparison analysis logic couldn't properly distinguish context from previous conversations. For example, after "Analyze Samsung Electronics stock," asking "How about LG Electronics?" would try to compare both stocks. In other words, the logic for determining the relationship between previous conversations and new requests was unstable.

**Solution**

1. Added a structure where the analysis function explicitly recognizes and receives stock information from previous conversations.
2. For ambiguous comparison requests, first excluded stocks already covered in previous analysis, then used LLM to determine if it's a comparison request, and only then re-added the previous stock for comparative analysis.
3. As a result, confirmed that over 80% of comparison/difference analysis requests were answered as intended.

## Results and Achievements

Successfully deployed a web service featuring real-time financial Q&A, single/multi-stock comparative analysis, automated report generation (MD/PDF/TXT), and chart visualization. Additionally, ensured answer quality through LLM Judge quality evaluation, and implemented multi-turn conversations and follow-up question handling based on conversation history.

![Result 1](/projects/assets/images/01/02.png)

![Result 2](/projects/assets/images/01/03.png)
