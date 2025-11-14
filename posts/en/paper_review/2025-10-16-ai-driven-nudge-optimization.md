---
layout: post
title: "AI-Driven Nudge Optimization: Integration of Behavioral Economics, Two-Tower Networks, and Multi-Armed Bandits"
date: 2025-10-16
categories: [Research, Review]
tags: [nudge, two-tower network, multi-armed bandit, behavioral economics, deep learning, recommendation system]
math: true
mermaid: true
---

## Searching for the Perfect Nudge

A nudge refers to reminding, prompting, or gently pushing someone's back to make them do something they might have postponed or skipped entirely. One of the fields where nudge power works most powerfully is finance. A well-designed nudge can make the difference between credit card application and expiration, break the boundary between deposit product subscription and neglect, and determine the outcome between insurance contract renewal and cancellation.

These might be small steps for customers, but they are decisions that make enormous differences in a bank's bottom line. And in most cases, these decisions don't happen spontaneously without external intervention. This is precisely why 'nudge optimization' has become a huge business. If implemented properly, nudges are powerful tools that increase customer lifetime value, boost product adoption rates, prevent churn, and maintain high engagement.

The authors of today's paper AI-Driven Nudge Optimization built a new recommendation system based on behavioral economics principles. This system doesn't just predict what customers might want, but organizes and presents products in the most persuasive way, actively inducing action by capturing the moment when customers are most susceptible to influence.

In this article, we'll examine how the authors combined algorithms to create a system that learns and adapts in real-time.

## Limitations of Existing Recommendation Systems: Humans Are Not Rational

The digital finance industry faces one problem: existing recommendation systems aren't very effective. Most off-the-shelf systems suggest products to customers based on past patterns. These models assume user preferences don't change and individual psychological factors don't matter. As a result, when user behavior changes at individual or market levels, systems fail to adapt, ultimately leading to stagnant engagement rates and declining conversion rates.

Are users really rational beings who make consistent and logical decisions? Behavioral economics says no. We are influenced by how information is presented (framing effect), motivated more by fear of losing something (loss aversion) than the possibility of gaining something, and tend to stick with options set by default (default bias) even when better alternatives exist. These psychological biases play decisive roles in financial decisions, yet most recommendation systems completely ignore this reality.

## New Approach: Two-Tower Networks and Multi-Armed Bandits

In this research, the authors are building a very different system from the past: an adaptive system combining structured representation learning with real-time decision optimization. At the core of this system are two complementary components:

* **Two-Tower Network (TWN):** Generates structured rankings for users and financial products. This serves as the foundation for understanding who users are and which products suit them.
* **Multi-Armed Bandit (MAB):** Treats recommendations as sequential decision problems, making the system continuously learn and adapt based on user responses.

This entire system operates in the following structure: gaining insights from behavioral data (Behavioral Insight), TWN ranking users and products (TWN), MAB adjusting rankings through real-time feedback (MAB), and finally outputting optimized recommendations by applying nudge mechanisms (Deployment).

<div align="center">
    <img src="/assets/images/paper_review/nudge_optimization_architecture.gif" width="600" alt="AI-Driven Nudge Optimization">
    <br>
    <a href="https://ieeexplore.ieee.org/document/11059932">
        Optimized recommendation model Architecture
    </a>
</div>

<br>

Before learning how these elements work together, we first need to understand the multi-armed bandit framework itself, which is the core that differentiates this system from existing approaches.

## Understanding Multi-Armed Bandits (MAB) in Detail

The multi-armed bandit problem can be analogized to deciding which slot machine to pull. Imagine slot machines with different win rates lined up, and you don't know each machine's win rate. With limited budget to maximize prize money, you must decide which machine to pull next based on previous experience.

<div align="center">
    <img src="/assets/images/paper_review/nudge_optimization_MAB_analogy.png" width="500" alt="Multi-Armed Bandit Analogy with Slot Machines">
    <br>
        Multi-Armed Bandit Analogy
</div>

<br>

This situation creates tension between Exploration and Exploitation:

* **Exploration:** Trying new actions to learn about potential rewards. The process of pulling levers on machines at the end to see what results they produce.
* **Exploitation:** Choosing actions known to have the highest expected reward. The strategy of continuously pulling the lever of the machine that's shown the highest win rate so far.

The real challenge is balancing these two over time. In the context of recommendation systems, each "lever" represents different nudge strategies or financial products. The entire system must balance between exploring new approaches (gathering more information) and exploiting already proven successful strategies. This framework is particularly powerful because it can adapt in real-time while learning which nudges are most effective for which users.

The epsilon-greedy strategy used in this research is a representative method of achieving this balance. For example, trying random options with 1% probability (exploration) and selecting the best-performing option with remaining 99% probability (exploitation). This allows the system to continue exploiting existing successful strategies while not missing opportunities to find better alternatives.

## The System's Brain: Structure of Two-Tower Networks (TWN)

So how does the two-tower network fit into this framework? The TWN architecture consists of two independent neural network "towers" processing different types of information. This separation structure is important because it allows the system to learn representations for users and products separately, then combine them meaningfully.

<div align="center">
    <img src="/assets/images/paper_review/nudge_optimization_TWN_concept.png" width="500" alt="Conceptual Diagram of a Two-Tower Network">
    <br>
        Two-Tower Network Conceptual Diagram
</div>

<br>

* **User Tower**: Encodes everything about individual users. This includes not only demographic information like age, income, region, but also behavioral patterns extracted from transaction history, browsing behavior, interaction frequency, preference data like risk tolerance and investment goals, and even psychometric data like personality, values, and lifestyle. It even processes nudge mechanism preference data about which psychological stimuli are most effective for specific user types.
* **Product Tower**: Encodes information about financial products and related nudge strategies. Each product is represented not only by financial characteristics like return or risk level, but also by behavioral profiles. Various nudge strategies are processed in this tower, including message framing (whether products are presented from profit or loss perspectives), urgency level creating temporal pressure, and default bias exploiting users' default choice tendencies.

These two towers project their respective input data into a common embedding space to calculate similarity. This is where the magic happens. When recommendation for a specific user is needed, that user's embedding vector is compared with product embedding vectors to generate a ranked list of candidate products. This similarity calculation captures complex, nonlinear relationships between user characteristics and product features that are difficult to code manually.

In fact, after training the model, **'risk_tolerance', 'risky_investment', 'phone_active_time'** emerged as very important features for predicting users' purchase decisions. This shows the system fundamentally grasps users' financial tendencies and digital activity patterns.

<div align="center">
    <img src="/assets/images/paper_review/nudge_optimization_feature_importance.gif" width="500" alt="Top 40 Important Features in the Model">
    <br>
    <a href="https://ieeexplore.ieee.org/document/11059932">
        Top 40 Important Features in Model
    </a>
</div>

<br>

The model's training process was very stable. Training and validation data loss steadily decreased together, and accuracy showed simultaneous increase, indicating good generalization without overfitting.

Ultimately, the model's prediction accuracy on test data reached 91.25%. As can be seen from the confusion matrix below, the model classified users who would purchase (745 people) and users who wouldn't purchase (735 people) with high precision.

<div align="center">
    <img src="/assets/images/paper_review/nudge_optimization_confusion_matrix.gif" width="500" alt="Confusion Matrix and Classification Report for the TWN Model">
    <br>
    <a href="https://ieeexplore.ieee.org/document/11059932">
        Confusion Matrix for the TWN Model
    </a>
</div>

<br>

The ingenuity of this design lies in combining deep learning's expressiveness with reinforcement learning's adaptability. **TWN provides sophisticated understanding of user-product relationships**, and **MAB enables real-time adaptation based on continuous interactions.** Either alone would have difficulty building such an effective system.

## Integration of Behavioral Economics: 4 Nudge Mechanisms

So how is behavioral economics integrated into this system? The authors divided nudges that can influence financial decision-making into four main categories. Each category targets different aspects of human psychology:

* **Decision Framework**: Simplifies choices using default options and preset recommendations. This exploits our natural tendency to minimize cognitive effort.
* **Decision Information**: Influences how options are perceived using framing effects and loss aversion. The same financial product can be expressed differently, like "Don't miss growth opportunities" or "Protect your assets from inflation risk."
* **Decision Assistance**: Induces participation by integrating gamification and incentives. Progress bars showing savings goals or point systems for completing financial tasks fall here.
* **Social Decision Appeal**: Motivates behavior using social proof and peer influence. Showing what choices similar users made or displaying testimonials from people in similar financial situations.

Each financial product in the system is connected with specific nudge strategies. For example, retirement savings products might use loss aversion framing to emphasize what would be lost by not saving. Investment products might use social proof to show how many similar users already chose this product. The MAB algorithm learns which nudge strategies are most effective for which users in which situations.

## Real-Time Learning and Optimization

When the system recommends products, it's not simply predicting relevance but **actively selecting strategies most likely to induce positive user behavior, including behavioral economics nudge strategies**. This selection occurs in real-time and adapts based on observed outcomes.

The Multi-Armed Bandit (MAB) algorithm is the core of this real-time adaptation. The figure below shows how the MAB system learns which strategies are most effective through actual user interactions.

1. Interaction 1: System presents recommendations combining 5 products ("Healthcare Insurance", "Gold Investment", etc.) with respective nudge strategies.
2. User selects "Gold Investment Product" and "Healthcare Insurance".
3. System immediately updates Rewards and Policy weights for these two products. Learning which nudge strategies were effective through user responses.
4. Interaction 2: According to updated policy, system presents new recommendation list, and this process repeats.

Thus MAB continuously optimizes itself by learning which products and nudge strategies are more effective for users through every interaction.

<div align="center">
    <img src="/assets/images/paper_review/nudge_optimization_MAB_interaction_flow.gif" width="600" alt="User Interaction and Adaptive Learning in MAB System">
    <br>
    <a href="https://ieeexplore.ieee.org/document/11059932">
        User Interaction Flow in MAB
    </a>
</div>

<br>

This learning process operates through a continuous feedback loop. When users interact with recommendations, those interactions provide reward signals to the bandit algorithm. Positive interactions like clicks, time viewing details, actual purchases provide positive rewards, while immediate ignoring or lack of engagement provides negative or zero rewards. The algorithm uses these signals to update understanding of which strategies are most effective for which users.

In this research, the reward function was designed to capture multiple objectives simultaneously. It considers not only immediate engagement metrics like click-through rate, but also long-term outcomes like actual product adoption and user satisfaction. This multi-objective approach ensures the system optimizes for meaningful business outcomes rather than just vanity metrics.

## Conclusion: Success in Simulation and Real-World Challenges

To validate their approach, the authors conducted A/B testing comparing the existing rule-based recommendation system (Group A) with the new AI-based system (Group B). The results were stunning. There was a dramatic difference in Conversion Rate, the ratio of recommendations leading to actual purchases. Conversion rate is calculated as follows:

$$
CR = \dfrac{P_{rec}}{N_{user}} \times 100
$$

* $P_{rec}$: Number of purchases through recommendations
* $N_{user}$: Total number of users

Experimental results showed the rule-based system's conversion rate was only 10.5%, while the AI-based system's conversion rate reached 52.87%. This is a statistically highly significant difference (p < 0.0001), demonstrating the overwhelming effect of the AI-based nudge system. Thus, this research clearly showed the possibility of next-generation recommendation systems that combine sophisticated deep learning architecture with real-time reinforcement learning to go beyond simply recommending relevant products and positively guide user behavior. However, whether success in simulation environments can be equally reproduced within the complexity of actual financial markets remains an important challenge to be solved going forward.

-----

## References

Kristiana, I., Prabowo, H., Gaol, F. L., & Qomariyah, N. N. (2025). AI-Driven Nudge Optimization: Integrating Two-Tower Networks and Multi-Armed Bandit With Behavioral Economics for Digital Banking Campaign. *IEEE Access*, 13, 112948-112961. [https://doi.org/10.1109/ACCESS.2025.3584648](https://doi.org/10.1109/ACCESS.2025.3584648)
