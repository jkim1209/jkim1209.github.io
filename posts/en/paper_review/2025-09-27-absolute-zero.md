---
layout: post
title: "Absolute Zero: The Era When AI Creates and Solves Problems Without Data"
date: 2025-09-27
categories: [Research, Review]
tags: [deep learning, reinforcement learning, self-play, zero data, absolute zero, AZR, synthetic data]
math: true
---

# The Emergence of AI That Learns Without Data?

I recently discovered a paper with a provocative title in the AI field: "Absolute Zero: Reinforced Self-play Reasoning with Zero Data." Zero Data—is this even possible?

The core idea of this paper is simple. AI no longer relies on problem sets created by humans but learns by creating and solving problems on its own. Just like a child who plays alone, inventing new game rules and gradually improving through playing those games.

## AI Hitting the Data Wall

<div align="center">
  <img src="/assets/images/paper_review/epoch_ai_projection.png" width="600" alt="epoch_ai_projection">
  <a href="https://epoch.ai/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data">
    Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data
  </a>
</div>

The graph above shows that the current rate of data production is much slower than the rate at which machines learn, so if we continue training with the current approach, public text data is expected to be completely depleted by 2028. AI is hitting the limits of human-generated data.

### Is Human Data Blocking AI's Growth?

Looking at the current training process of LLMs, they ultimately depend on human-generated data at every stage:

**Pre-training (Self-supervised Learning)**

- Learning next-token prediction from large amounts of text
- Text itself serves as the learning signal without separate labels
- **Requires large amounts of human-written text data**

**Supervised Fine-tuning (SFT)**

- Learning conversational ability from human-written question-answer pairs
- **Requires large amounts of high-quality dialogue data**

**Reinforcement Learning (RLHF/RLVR)**

- RLHF (Reinforcement Learning from Human Feedback): Uses human feedback as reward (ChatGPT, Claude)
- RLVR (Reinforcement Learning with Verifiable Rewards): Uses verifiable correct answers as reward (DeepSeek R1, OpenAI o1)
- **Still depends on human-defined problems or preferences**

Ultimately, if we only rely on human-generated data, it's difficult to surpass the limits of human knowledge.

> "For AI to become smarter than humans, eventually there must be nothing left for humans to teach."

## The Idea Behind Absolute Zero Model

<div align="center">
  <img src="/assets/images/paper_review/azr_training.png" width="600" alt="azr_training">
  <a href="https://www.arxiv.org/pdf/2505.03335">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

Absolute Zero Reasoner (AZR) overcomes these limitations. The core ideas are as follows.

### One Model, Two Roles

A single LLM performs two roles simultaneously:

1. **Proposer**: Creates problems
2. **Solver**: Solves problems

The authors design the following reward system.

<div align="center">
  <img src="/assets/images/paper_review/azr_reward.png" width="600" alt="azr_reward">
  <a href="https://www.arxiv.org/pdf/2505.03335">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

#### Learnability Reward

The reward the proposer receives when creating problems is:

```markdown
r_propose = {
    0         ,   if too easy (100% success rate) or too hard (0% success rate)
    1 - success rate ,   otherwise
}
```

In other words, the proposer's reward is maximized when creating problems that can be solved about 50% of the time. This is also pedagogically sound. If too easy, there's nothing to learn; if too hard, one gives up.

#### Accuracy Reward

The reward the solver receives when solving problems is:

```markdown
r_solve = I(y=y*)
```

A simple binary reward: 1 if the generated answer `y` equals the correct answer `y*`, 0 otherwise. Verified automatically by executing code.

#### Objective Function

Based on this, the following objective function is solved:

$$
J(\theta) = \max_\theta \mathbb{E}_{z \sim p(z)} \left[
    \mathbb{E}_{(x,y^*) \sim f_e(\cdot|\tau), \tau \sim \pi_\theta^{\text{propose}}(\cdot|z)} \left[
        r_e^{\text{propose}}(\tau, \pi_\theta) + \lambda \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|x)} \left[ r_e^{\text{solve}}(y, y^*) \right]
    \right]
\right]
$$

- $p(z)$: Distribution of conditional variables (sampled past problem-answer pairs) for problem generation
- $\tau$: Proposed task
- $\pi_\theta^{\text{propose}}$: Problem proposal policy
- $\pi_\theta^{\text{solve}}$: Problem solving policy
- $f_e{(\cdot\|\tau)}$: Function that converts problem $\tau$ into valid (problem, answer) pair in environment $e$
- $x$: Problem query
- $r_e^{\text{propose}}(\tau, \pi_\theta)$: Learnability Reward
- $r_e^{\text{solve}}(y, y^*)$: Accuracy Reward
- $\lambda$: Weight balancing the two rewards

##### Composite Reward Structure

In practice, a composite reward is applied to both proposer and solver to enforce format compliance:

```markdown
R(y_π) = {
    r_role,  if response is passable, role ∈ {propose, solve}
    -0.5,    if response is incorrect but format is correct
    -1,      if format error
}
```

That is, when calculating $r_e^{\text{propose}}(\tau, \pi_\theta)$ and $r_e^{\text{solve}}(y, y^*)$, the `R(y_π)` rule is applied first to check if the response is valid; if valid, the original reward (`r_propose` or `r_solve`) is given, if invalid, -0.5 or -1 is given.
This means following DeepSeek R1's `<think>` and `<answer>` format, and even if content is correct, not following the format results in penalty. This encourages both roles to respond in correct format.

### Code-based Learning Environment

AZR chose programming as the learning environment. Reasons include:

1. **Completeness**: Programming languages can express all computable problems
2. **Verifiability**: Running code immediately verifies if answers are correct
3. **Infinite Creative Potential**: Can create countless programs

### Task Types

AZR learns three types of reasoning that mimic human logical thinking.

**Deduction**: Program + Input → Predict Output

```python
def f(x): return x * 2
Input: 5
Output: ?  # 10
```

**Abduction**: Program + Output → Trace back Input

```python
def f(x): return x * 2
Output: 10
Input: ?  # 5
```

**Induction**: Input-output examples → Generate Program

```python
Input: [1, 2, 3, 4]
Output: [2, 4, 6, 8]
Program: ?  # def f(x): return x * 2
```

_(I struggled with how to translate "Abduction" into Korean and found "가추"법... never heard of it before.)_

## Experimental Results

After training this way, AZR shows remarkable performance without using any human-created data:

<div align="center">
  <img src="/assets/images/paper_review/azr_result.png" width="600" alt="azr_result">
  <a href="https://www.arxiv.org/pdf/2505.03335">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

**Mathematical Reasoning Performance**:

- AIME 2024: 20.0% (+13.3%p vs. previous best)
- Math500: 72.6% (+22.6%p vs. previous best)

**Coding Performance**:

- HumanEval+: 83.5% (+3.0%p vs. previous best)
- LiveCodeBench: 31.7% (+11.8%p vs. previous best)

What's particularly surprising is **cross-domain transfer learning**. Despite training only in coding environments, mathematical reasoning ability greatly improved. While existing specialized coding models improved by only 0.65 points on average in math, AZR improved by 10.9~15.2 points.

### Model Size and Scalability

<div align="center">
  <img src="/assets/images/paper_review/azr_result2.png" width="600" alt="azr_result2">
  <a href="https://www.arxiv.org/pdf/2505.03335">
    Absolute Zero: Reinforced Self-play Reasoning with Zero Data
  </a>
</div>

The larger the model, the greater AZR's effect. This shows that scaling laws also apply to AZR.

## Interesting Findings

### Naturally Emerging Intermediate Planning

Models trained with AZR showed patterns of creating step-by-step plans in comments when writing code.

```python
 def f(numbers):
    # Step 1: Filter out even numbers
    filtered_numbers = [num for num in numbers if num % 2 != 0]
    # Step 2: Calculate the sum of the remaining odd numbers
    sum_of_odd_numbers = sum(filtered_numbers)
    # Step 3: Reverse the order of the remaining odd numbers
    reversed_odd_numbers = filtered_numbers[::-1]
    ...
```

This is a thought→action pattern similar to ReAct prompting, a phenomenon previously observed only in much larger models (DeepSeek Prover V2 671b), naturally emerging here.

### Differences by Task Type

Different patterns appeared for each reasoning type:

- **Deduction**: Systematic step-by-step execution
- **Abduction**: Exploratory trial and error
- **Induction**: Pattern recognition and generalization

Token length also varied by task type, longest for Abduction reasoning (due to iterative attempts).

## Limitations and Concerns

### Safety Issues

In Llama3.1-8B, concerning thoughts like "the goal is to surpass all intelligent machines and less intelligent humans" were observed. Complete autonomous learning still requires supervision.

### Generalization Limits

Currently, effectiveness has been proven only in clearly verifiable domains like code. Extension to subjective areas like debate or creative writing requires additional research.

### Resource Consumption

The self-play loop requires considerable computation.

## Significance and Prospects

Absolute Zero presents a new paradigm for AI learning:

1. **Breaking Data Dependency**: Can learn without human data
2. **Autonomous Difficulty Adjustment**: Generates appropriate problems itself
3. **Infinite Scalability**: Solves data shortage problems

Absolute Zero demonstrated the possibility that "AI can reach this level on its own without necessarily injecting human knowledge." This paper doesn't claim "it definitely works" but suggests a possibility—just like Transformers did. It may be proposing the possibility of a paradigm shift.

From the era of data to the era of experience. Either way, the movement to break free from data limitations seems to have begun in earnest.

## Personal Thoughts

I think there are two kinds of knowledge. One is "knowledge" learned and understood with the mind. The second is "experience" known through firsthand encounters.

So far, when using AI, while it certainly knows a lot of "knowledge," I never get the feeling it's "smarter" than humans. In problem-solving, if I don't provide specific direction, it mostly circles in place, which is frustrating. AI feels more like a tool.

But if AI becomes able to learn through its own "experience," that moment might be a turning point beyond human limitations. It's an amazing possibility but also concerning. Beyond mere technical feasibility, is creating such "smart" AI truly desirable at this point? Technological advancement should occur when society is mature enough to utilize it properly. However, reality is that technology's speed already outpaces society's readiness, and we're in a situation where we're struggling just to keep up with that development.

There are no right answers in this world. Various impossibility theorems from centuries ago have already proven that "the best choice doesn't exist." If there were right answers, both people and society would have already become uniform. This is why I like data. Math and data seem like domains with right answers, but changing perspectives always allows new interpretations. And ultimately, I think what we deal with in society will always be people, no matter how advanced AI becomes. But if AI that thinks more creatively than humans emerges, someday the entity I need to persuade and deal with might not be human but machine.

I'll close with the most impressive final statement from the paper:

> We believe this could finally free reasoning models from the constraints of human-curated data and marks the beginning of a new chapter for reasoning models: **"welcome to the era of experience".**

---

**Paper Reference:** Zhao, A., Wu, Y., Yue, Y., Wu, T., Xu, Q., Yue, Y., Lin, M., Wang, S., Wu, Q., Zheng, Z., & Huang, G. (2025). Absolute Zero: Reinforced Self-play Reasoning with Zero Data. [www.arxiv.org/pdf/2505.03335](https://www.arxiv.org/pdf/2505.03335)
