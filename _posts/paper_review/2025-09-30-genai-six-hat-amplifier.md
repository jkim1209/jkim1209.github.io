---
layout: post
title: "AI는 격차를 줄이는가 증폭시키는가: 인지 능력 발달에 대한 실험"
date: 2025-09-30
categories: [Research, Review, AI, Education]
tags: [GenAI, cognitive presence, creativity, educational technology, epistemic network analysis, six thinking hats, learning analytics, cognitive amplifier, educational inequality]
math: true
---

## AI는 격차를 확대하는가

현재 AI는 누구나 편리하게 사용할 수 있는 도구가 되었고, 찾고자 하면 손쉽게 정보를 찾을 수 있다.  
그렇다면 이러한 AI의 사용은 사용자 간의 격차를 줄여주는 도구가 될 것인가(equalizer), 아니면 오히려 그 반대일까(amplifier)?  
논문의 저자들은 인지 발달(Cognitive Presence) 능력에 대해 이를 살펴보고자 했다.

## 연구 설계

중국의 한 사범대학에서 108명의 학생들을 대상으로 14주간 실험을 진행한다. 절반의 학생들은 실험기간 동안 AI(IFlytek Spark)를 사용할 수 있었고(GSG 그룹), 나머지 절반은 그렇지 않았다(SG 그룹).
구체적인 실험 절차는 다음과 같다:

**1주~3주차:**
- 모든 학생에게 PBL 과제와 Six-Hat Thinking 기법 교육
- 3주차 말: TTCT와 SCCT 창의성 테스트 실시
- 테스트 결과로 학생들을 상위 20%(고창의성), 하위 20%(저창의성)로 분류

**4주~16주차:**
- 무작위로 나눠진 두 그룹(GSG와 SG)이 프로젝트 수행
- GSG: AI 사용 가능, SG: AI 사용 불가

따라서 실험 설계상 4~16주차에 학생들은 2×2로 분류된다:
- 고창의성 + AI 사용
- 고창의성 + AI 미사용
- 저창의성 + AI 사용
- 저창의성 + AI 미사용

<div align="center">
  <img src="/assets/images/paper_review/cognitive_exp_mech.png" width="600" alt="cognitive_exp_mech">  
  <br>
  <a href="https://educationaltechnologyjournal.springeropen.com/articles/10.1186/s41239-025-00545-x/figures/2">
    Exploring cognitive presence patterns in GenAI-integrated six-hat thinking technique scaffolded discussion: an epistemic network analysis
  </a>
</div>

### Six-Hat Thinking과 Cognitive Presence

학생들은 프로젝트를 수행하며 Edward de Bono가 개발한 Six-Hat Thinking 기법으로 구조화된 토론을 했다. 각 단계(모자)는 서로 다른 사고 방식을 나타낸다:

- **빨간 모자**: 직관과 감정
- **하얀 모자**: 객관적 정보와 데이터
- **노란 모자**: 긍정적 측면과 이점
- **검은 모자**: 위험과 문제점
- **초록 모자**: 창의적 아이디어
- **파란 모자**: 전체 프로세스 관리

학생들은 `빨간 모자 → 하얀 모자 → 노란 모자 → 검은 모자 → 초록 모자 → 검은 모자 → 초록 모자` 순서로 토론을 진행했다.  
AI 사용 그룹은 각 단계에서 GenAI를 활용했다. 예를 들어 하얀 모자 단계에서는 객관적 정보를 검색하고, 초록 모자 단계에서는 창의적 대안을 요청했다.  

그리고 각 단계에서 나온 게시물들을 MOOC-BERT로 분석해 Cognitive Presence의 어느 단계에 해당하는지 분류했다:

1. **Triggering**: 문제 인식 단계
2. **Exploration**: 아이디어 탐색 단계
3. **Integration**: 정보 통합 단계
4. **Resolution**: 실제 적용 단계

총 15,678개의 게시물이 분석되었고, 그 뒤 ENA(Epistemic Network Analysis)를 사용해 이 단계들 간의 연결 패턴을 시각화하고 그룹 간 차이를 통계적으로 비교했다.

## 주요 발견

### 1. AI는 고창의성 학습자를 더욱 강화했다

고창의성 학생들이 AI를 사용했을 때:

- 더 많은 게시물 생성 (1,673개 vs 1,074개)
- 탐색(Exploration) 단계에서 63.42%의 담화 (비AI 그룹 51.58%)
- 탐색-통합(Exploration-Integration) 연결 강도: 0.81
- 탐색-해결(Exploration-Resolution) 연결 강도: 0.36

이들은 AI를 "사고의 확장 도구"로 활용했다. AI가 제공한 정보를 바탕으로 더 깊은 분석에 도달하고, 아이디어를 정교화하며, 해결책으로 원활하게 이동했다.

### 2. 저창의성 학습자에게는 제한적 효과

저창의성 학생들의 경우:

- AI 사용 여부에 관계없이 최소한의 차이
- AI 사용 시에도 촉발-탐색(Triggering-Exploration) 경로에 머무름 (0.57 vs 0.56)
- 탐색-통합 연결은 더 강했지만(0.70), 통합-해결 연결은 더 약함(0.09)

이들에게 AI는 아이디어를 반복하고 다듬는 데는 도움을 줬지만, 실제 적용으로 이동하는 데는 거의 기여하지 못했다. 오히려 AI 의존성이 독립적 사고 능력 발달을 방해했을 가능성이 있다.

### 3. 인지 능력 증폭기로 작용하는 AI

논문은 AI가 "평등화 도구"가 아니라 "인지 증폭기(cognitive amplifier)"로 작동한다고 주장한다. 기존의 강점과 약점을 모두 더 크게 만든다는 것이다.

- 이미 강한 창의적 사고자에게는 인지 능력을 증폭시켜주는 도구
- 창의성에 어려움을 겪는 사람들에게는 독립적 능력 개발을 방해
- 결과적으로 고창의성-저창의성 간 인지 능력 격차가 확대됨

## 한계

이 연구의 설계에는 분명한 한계가 있다:

1. **표본의 제한성**: 단일 문화권(중국)의 단일 대학 학생들만을 대상으로 함
2. **소규모 샘플**: 각 조건당 약 10명, 중간 창의성 학생 60% 제외
3. **질적 데이터 부재**: 학생들의 AI 사용 경험에 대한 인터뷰나 성찰 데이터 미수집
4. **Resolution 단계 부족**: 모든 그룹에서 실제 적용 단계가 현저히 약했음. Six-Hat 기법과 AI가 아이디어 생성에는 효과적이지만, 고차원적 인지 과정을 충분히 촉진하지 못했을 가능성 존재

더 다양한 맥락과 더 큰 표본을 대상으로 한다면 다른 결과가 나올 수 있다.

## 시사점

어떻게 보면 예상되었던 결과다. AI를 저창의성 학습자를 위해 활용할 때는 맞춤형으로 활용하는 개입이 필요하다. AI가 "답을 주는 도구"가 아니라 "사고를 자극하는 파트너"가 되도록 설계해야 한다. 단순히 정보를 제공하는 것을 넘어, 질문을 던지고 대안적 관점을 제시하며 비판적 사고를 요구하는 방식으로. 모든 학생에게 동일한 방식으로 AI를 사용하게 하는 것은 격차를 확대할 뿐이다.

AI는 마법의 해결책이 아니다. 누구에게나 동일하게 작동하지 않으며, 개인의 특성에 따라 매우 다른 효과를 낸다.

우리가 AI를 평등화 도구로 쉽게 상상하는 것은, 기술이 본질적으로 중립적이며 모두에게 공평하게 기회를 제공한다는 믿음 때문이다. 하지만 이 실험은 그 믿음에 의문을 제기한다.

여느 기술이 그랬듯 AI도 격차를 줄이지 않는다. 증폭시킨다. 교육 현장에서 AI를 도입할 때는 이러한 차별적 효과를 인식하고, 모든 학습자가 혜택을 받을 수 있도록 설계해야 할 것이다.

---

## References

Yu, M., Liu, Z., Long, T., Li, D., Deng, L., Kong, X., & Sun, J. (2025). Exploring cognitive presence patterns in GenAI-integrated six-hat thinking technique scaffolded discussion: an epistemic network analysis. *International Journal of Educational Technology in Higher Education*, 22(48). [https://doi.org/10.1186/s41239-025-00545-x](https://doi.org/10.1186/s41239-025-00545-x)
