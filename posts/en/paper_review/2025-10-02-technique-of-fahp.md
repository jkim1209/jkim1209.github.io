---
layout: post
title: "What is the 'Best' Coffee? How AI Understands Human's Ambiguous Preferences"
date: 2025-10-02
categories: [Research, Review]
tags: [expert system, fuzzy logic, AHP, MCDM, q-rung orthopair fuzzy sets, decision making]
math: true
---

## "ChatGPT, what's the best coffee in this city?"

The question "what's the best OO" we ask AI is actually one of the most difficult questions for AI. If AI simply calculates and tells us the average star rating from the internet, would we be satisfied? 'The place with the highest rating' and what we feel is 'the best place' are clearly different. The latter is a highly subjective domain where numerous factors like price, atmosphere, service, and personal memories are complexly intertwined.

This is a question that requires 'expert insight' beyond data. A recently published paper presents a new blueprint for precisely this problem—how AI can understand humans' ambiguous and multi-layered preferences and make optimal decisions.

## What Makes Decision-Making Complex?

The process of choosing 'the best coffee' isn't determined by taste alone. Some people prioritize price, others fair trade certification of beans, and yet others care about brands that signal social status. Finding the optimal alternative among criteria that conflict and are sometimes difficult to even compare is called Multi-Criteria Decision-Making (MCDM).

<div align="center">
    <img src="/assets/images/paper_review/ahp-hierarchy.png" width="600" alt="hierarchy">
    <br>
    <a href="https://www.1000minds.com/decision-making/analytic-hierarchy-process-ahp">
        Analytic Hierarchy Process example
    </a>
</div>

To solve this complex problem, researchers use a powerful tool called the Analytic Hierarchy Process (AHP). AHP creates a hierarchy of 'goals-criteria-alternatives' as shown above, and assigns weights by having experts compare each criterion pairwise.

However, there's a fatal limitation. If you ask an expert "Is price exactly 1.5 times more important than quality?" how many could answer confidently? Human judgment is inherently ambiguous like "Hmm... price seems a bit more important..." This "uncertainty" is precisely why traditional AHP hits its limits.

## AI Learns 'Ambiguity': q-ROFS

To solve this problem, Fuzzy Logic emerged. It's a way to mathematically express middle grounds like 'somewhat good' or 'very good' beyond black-and-white logic of 'good/bad'.

And this paper goes one step further with the latest theory: q-rung Orthopair Fuzzy Sets (q-ROFS) integrated with AHP.

### q-ROFS: Capturing Even Expert 'Hesitation'

While traditional fuzzy theory focused on 'to what degree does it belong' (membership degree), q-ROFS considers both 'to what degree does it belong' (membership degree, $\tilde{E}_S$) and 'to what degree does it not belong' (non-membership degree, $\dot{G}_S$).

The biggest innovation lies in the relationship between these two values. q-ROFS grants much greater freedom to membership and non-membership degrees while satisfying the following condition:

$$
0 \le \tilde{E}_{S}(r)^{q} + \dot{G}_{S}(r)^{q} \le 1, \quad (q \ge 1)
$$

Here, as parameter $q$ increases, the range of values membership and non-membership degrees can take widens. This enables mathematical expression even of the expert's hesitation or uncertainty when "it's ambiguous to say it's good, but can't say it's bad either."

<div align="center">
    <img src="/assets/images/paper_review/qrofs-comparison.png" width="500" alt="Comparison of fuzzy set spaces">
    <br>
    Membership degree(μ) means the "degree to which an object belongs" to a set, Non-membership degree(ν) means the "degree to which it doesn't belong."
    In IFS(q=1), the sum of two values is restricted not to exceed 1, but as q increases, it allows both (μ, ν) to be simultaneously larger, expressing even expert's hesitation or uncertainty.
</div>

### TR-q-ROFNS: Extension for Actual Calculation

If q-ROFS is the theoretical framework, what the paper actually used for decision-making problems is TR-q-ROFNS (Triangular q-rung orthopair fuzzy number set).
This extends membership and non-membership degrees as triangular fuzzy numbers $(a_1, a_2, a_3)$ instead of point values, allowing the handling of "approximately this much" ambiguous assessments by experts as ranges (lower–middle–upper).
This enables more realistic uncertainty reflection and easier application to actual decision-making problems like MCDM.

### Converting Ambiguity to Scores: Score & Accuracy Functions

To rank these expressed ambiguous fuzzy values, the paper uses Score Function and Accuracy Function:

$$
S(\tilde{a}) = \frac{(a_1+a_2+a_3)(1+E^q-G^q)}{6}, \quad
H(\tilde{a}) = \frac{(a_1+a_2+a_3)(1+E^q+G^q)}{6}
$$

These functions synthesize the expert's judgment (value range $a_{1,2,3}$), confidence in that judgment ($E$), and non-confidence ($G$) into a single clear score.

### Final Result: South Africa's Best Coffee Brand

The researchers applied TR-q-ROFNS-based FAHP to five main criteria (C1: Availability, C2: Effectiveness, C3: Price, C4: Quality, C5: Quantity) and three alternative brands (A1, A2, A3).

<div align="center">
    <img src="/assets/images/paper_review/qrofs-result.png" width="500" alt="qrofs-result">
    <br>
    <a href="https://peerj.com/articles/cs-2555/" target="_blank">
        Final ranking
    </a>
</div>

As a result of synthesizing all criteria for final priority calculation, A2 received the highest score and was selected as South Africa's best coffee brand.

This is meaningful in that it's "a result of mathematically systematizing expert judgment while reflecting uncertainty and ambiguity" rather than simply averaging ratings.

## Could This Open the Era of Explainable 'Expert AI'?

This paper is like a blueprint showing how AI can step into humans' subjective domains. It contains the entire process of systematically structuring complex problems, delicately translating humans' ambiguous judgments into mathematical language, and ultimately making transparent, explainable decisions.

Unlike 'black box' models that learn patterns based on data, such logic-based expert systems have the strong advantage of clearly tracing 'why' they made such decisions. The future where AI understands our preferences and makes better recommendations may not start from learning more data, but perhaps from understanding our ambiguity and uncertainty more deeply.

## Limitations and Future Research Directions

- **Computational Complexity**: q-rung fuzzy AHP requires many calculations, so computational cost is high for large-scale problems. The researchers also mention that software optimization is needed for actual service implementation.
- **Non-membership Value Justification Problem**: When experts are not confident that "this brand is bad," it's empirically difficult to determine how to set the non-membership grade.

---

## References

Huang Y, Gulistan M, Rafique A, Chammam W, Aurangzeb K, Rehman AU. 2025. *The technique of fuzzy analytic hierarchy process (FAHP) based on the triangular q-rung fuzzy numbers (TR-q-ROFNS) with applications in best African coffee brand selection.* PeerJ Computer Science 11:e2555 [https://doi.org/10.7717/peerj-cs.2555](https://doi.org/10.7717/peerj-cs.2555)
