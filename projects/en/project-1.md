---
tags: Python, BERT, NLP, Finance, Sentiment Analysis, LDA
date: 2024
icon: 📈
---

# Biodiversity Concern and Firm Performance

## Research Overview

This study examines how biodiversity-related news affects U.S. firm stock returns, bridging the gap between environmental science and finance by providing systematic evidence of how biodiversity concerns influence financial markets.

Research Paper Download: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5625611>

## Research Motivation

While climate finance has received extensive academic attention, the economic impacts of biodiversity have only recently begun to be studied systematically. Given that biodiversity loss is estimated to cause annual economic damages of $4-20 trillion, I wanted to understand how markets respond to biodiversity-related information.

## Methodology

### Data Collection

I collected 60,213 biodiversity-related articles from Dow Jones Factiva spanning January 2003 to December 2022. The data shows a clear upward trend in biodiversity coverage, reflecting growing awareness of environmental issues.

![Data Collection](/projects/assets/images/01/00.png)

### Sentiment Analysis with BERT

Using the Bidirectional Encoder Representations from Transformers (BERT) model, I classified each sentence on a scale from 1 (most negative) to 5 (most positive). Examples include:

Negative sentiment (score = 1): Articles discussing legal action against governmental oversight failures in environmental protection, highlighting consequences like major oil spills.

Positive sentiment (score = 5): Articles detailing company achievements in energy conservation and sustainability, such as ENERGY STAR® certification.

From this analysis, I constructed the Biodiversity Sentiment Index (BSI) by subtracting the daily count of negative articles from positive articles.

### Physical vs. Transition Risk Analysis

Using Latent Dirichlet Allocation (LDA), I identified five distinct topics and categorized them into:

Physical Risks (2 topics):

- Endangered Species
- Natural Resource Management

Transition Risks (3 topics):

- Conservation Policy
- Environmental News  
- Regulations and Permits

## Key Findings

### 1. Short-term Impact with Long-term Attenuation

Higher biodiversity sentiment significantly increases daily stock returns (coefficient ≈ 0.44), but this effect disappears at the monthly horizon. This suggests markets react to sentiment rather than fundamental changes.

### 2. Transition Risks Dominate Physical Risks

The empirical results strongly support my hypothesis that transition risks have greater impact on stock returns than physical risks.

### 3. Heterogeneous Channel Effects

Within transition risks, I find interesting patterns:

- Environmental news attention loads positively on returns, consistent with increased public awareness raising cost of capital
- Regulations and permits attention loads negatively, reflecting anticipated operating constraints

![Research Results 1](/projects/assets/images/01/01.png)

![Research Results 2](/projects/assets/images/01/02.png)

![Research Results 3](/projects/assets/images/01/03.png)

## Contributions and Implications

This research makes three main contributions:

1. Scalable Measurement: Introduces a comprehensive, news-based biodiversity measure from large-scale textual analysis
2. Risk Decomposition: Uses topic modeling to distinguish between physical and transition risk channels
3. Empirical Evidence: Provides robust evidence that transition channels dominate in biodiversity risk pricing

### Practical Implications

For Investors: Monitoring biodiversity-related news, particularly regulatory changes, can provide valuable signals for portfolio management and ESG investing strategies.

For Firms: Companies should proactively manage biodiversity-related transition risks, especially regulatory compliance and public perception.

For Policymakers: New biodiversity regulations have immediate market impacts that should be considered in policy design.

## Limitations and Future Research

The main limitation is the mismatch between time-series explanatory variables and firm-specific dependent variables. Future research could:

1. Construct firm-level biodiversity indices using 10-K filings
2. Develop biodiversity-based long-short portfolio strategies
3. Explore cross-sectional variation in firm exposure to biodiversity risks

## Robustness and Methods

The results are robust to standard asset-pricing controls (market, size, value, profitability, investment, momentum factors) and various specification checks including lagged effects, event-time dynamics, and endogeneity considerations.

This research demonstrates that biodiversity is not just an environmental issue but a material financial factor that markets actively price. The dominance of transition risks over physical risks has important implications for how we think about biodiversity in investment and risk management contexts.
