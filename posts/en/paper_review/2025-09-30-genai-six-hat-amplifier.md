---
layout: post
title: "Does AI Reduce or Amplify Gaps: An Experiment on Cognitive Development"
date: 2025-09-30
categories: [Research, Review]
tags: [GenAI, cognitive presence, creativity, educational technology, epistemic network analysis, six thinking hats, learning analytics, cognitive amplifier, educational inequality]
math: true
---

## Does AI Widen the Gap?

AI has become a tool anyone can conveniently use, and information can be easily found when sought.
So will AI use serve as a tool that reduces gaps between users (equalizer), or rather the opposite (amplifier)?
The paper's authors sought to examine this regarding cognitive presence development.

## Research Design

A 14-week experiment was conducted with 108 students at a teachers' college in China. Half of the students could use AI (IFlytek Spark) during the experiment period (GSG group), while the other half could not (SG group).
The specific experimental procedure was as follows:

**Weeks 1-3:**
- All students educated on PBL tasks and Six-Hat Thinking technique
- End of week 3: TTCT and SCCT creativity tests administered
- Test results used to classify students into top 20% (high creativity) and bottom 20% (low creativity)

**Weeks 4-16:**
- Two randomly divided groups (GSG and SG) conduct projects
- GSG: AI usage permitted, SG: AI usage not permitted

Therefore, in the experimental design during weeks 4-16, students are classified in a 2×2 manner:
- High creativity + AI use
- High creativity + No AI use
- Low creativity + AI use
- Low creativity + No AI use

<div align="center">
  <img src="/assets/images/paper_review/cognitive_exp_mech.png" width="600" alt="cognitive_exp_mech">
  <br>
  <a href="https://educationaltechnologyjournal.springeropen.com/articles/10.1186/s41239-025-00545-x/figures/2">
    Exploring cognitive presence patterns in GenAI-integrated six-hat thinking technique scaffolded discussion: an epistemic network analysis
  </a>
</div>

### Six-Hat Thinking and Cognitive Presence

Students conducted structured discussions using the Six-Hat Thinking technique developed by Edward de Bono while carrying out projects. Each stage (hat) represents a different thinking mode:

- **Red Hat**: Intuition and emotion
- **White Hat**: Objective information and data
- **Yellow Hat**: Positive aspects and benefits
- **Black Hat**: Risks and problems
- **Green Hat**: Creative ideas
- **Blue Hat**: Overall process management

Students proceeded with discussions in the order: `Red Hat → White Hat → Yellow Hat → Black Hat → Green Hat → Black Hat → Green Hat`.
The AI-using group utilized GenAI at each stage. For example, in the White Hat stage they searched for objective information, and in the Green Hat stage they requested creative alternatives.

Posts from each stage were analyzed with MOOC-BERT to classify which stage of Cognitive Presence they corresponded to:

1. **Triggering**: Problem recognition stage
2. **Exploration**: Idea exploration stage
3. **Integration**: Information integration stage
4. **Resolution**: Practical application stage

A total of 15,678 posts were analyzed, and then ENA (Epistemic Network Analysis) was used to visualize connection patterns between these stages and statistically compare differences between groups.

## Key Findings

### 1. AI Further Strengthened High-Creativity Learners

When high-creativity students used AI:

- More posts generated (1,673 vs 1,074)
- 63.42% of discourse in Exploration stage (non-AI group 51.58%)
- Exploration-Integration connection strength: 0.81
- Exploration-Resolution connection strength: 0.36

They used AI as a "thinking expansion tool." Based on information provided by AI, they reached deeper analysis, refined ideas, and moved smoothly to solutions.

### 2. Limited Effect on Low-Creativity Learners

For low-creativity students:

- Minimal difference regardless of AI usage
- Even with AI use, remained in Triggering-Exploration pathway (0.57 vs 0.56)
- Exploration-Integration connection was stronger (0.70), but Integration-Resolution connection was weaker (0.09)

For them, AI helped repeat and refine ideas, but contributed little to moving to practical application. There's a possibility that AI dependency hindered the development of independent thinking abilities.

### 3. AI Acting as a Cognitive Amplifier

The paper argues that AI operates not as an "equalizer tool" but as a "cognitive amplifier." It makes both existing strengths and weaknesses larger.

- For already strong creative thinkers, a tool that amplifies cognitive abilities
- For those struggling with creativity, hinders independent ability development
- Consequently, the cognitive ability gap between high-creativity and low-creativity widens

## Limitations

This study's design has clear limitations:

1. **Sample Limitations**: Only students from a single university in a single culture (China)
2. **Small Sample Size**: About 10 people per condition, 60% of medium-creativity students excluded
3. **Absence of Qualitative Data**: No interviews or reflection data collected on students' AI usage experiences
4. **Lack of Resolution Stage**: The practical application stage was notably weak across all groups. There's a possibility that while the Six-Hat technique and AI were effective for idea generation, they didn't sufficiently promote higher-order cognitive processes

Different results might emerge with more diverse contexts and larger samples.

## Implications

In a way, these were expected results. When utilizing AI for low-creativity learners, tailored interventions are necessary. AI should be designed to be a "thinking-stimulating partner" rather than an "answer-providing tool." Beyond simply providing information, it should ask questions, present alternative perspectives, and demand critical thinking. Having all students use AI in the same way only widens the gap.

AI is not a magic solution. It doesn't work the same for everyone and has very different effects depending on individual characteristics.

We easily imagine AI as an equalizer because we believe technology is inherently neutral and provides opportunities fairly to all. But this experiment challenges that belief.

Like any technology, AI doesn't reduce gaps. It amplifies them. When introducing AI in educational settings, we must recognize these differential effects and design so all learners can benefit.

---

## References

Yu, M., Liu, Z., Long, T., Li, D., Deng, L., Kong, X., & Sun, J. (2025). Exploring cognitive presence patterns in GenAI-integrated six-hat thinking technique scaffolded discussion: an epistemic network analysis. *International Journal of Educational Technology in Higher Education*, 22(48). [https://doi.org/10.1186/s41239-025-00545-x](https://doi.org/10.1186/s41239-025-00545-x)
