---
layout: post
title: "Tabular Data Modeling Insights from Intrusion Detection Research: Applicability to CTR Prediction"
date: 2025-09-26
categories: [Research, Review]
tags: [deep learning, machine learning, intrusion detection, tabular data, SMOTE, CTR, FT transformer]
---

# Tabular Data: Machine Learning vs Deep Learning

While reading Ali et al.'s "Deep Learning vs. Machine Learning for Intrusion Detection in Computer Networks: A Comparative Study" (2025), I found many commonalities with the CTR (Click-Through Rate) prediction competition I'm currently participating in, which prompted me to organize these thoughts.

## Research Findings and Unexpected Discoveries

The researchers systematically compared traditional machine learning and deep learning approaches for network intrusion detection using the CIC-IDS2017 dataset.

**Traditional ML Results:**

- **Random Forest: 99.88% accuracy, 97.46% F1-score** (best performance)
- Decision Tree: 99.83% accuracy, 97.60% F1-score
- K-Nearest Neighbors: 99.88% accuracy, 97.46% F1-score
- Logistic Regression: 96.91% accuracy, 74.00% F1-score
- SVM: 97.32% accuracy, 73.33% F1-score
- Naive Bayes: 64.59% accuracy, 48.88% F1-score

**Deep Learning Results:**

- LSTM: 98% accuracy
- CNN: 98% accuracy
- MLP: 97% accuracy

The results show that **Random Forest achieved the highest performance**. This clearly demonstrates that deep learning is not a universal solution.

## Common Challenges

What was interesting about this research was its similarity to the data I've encountered in the CTR competition.

**Imbalanced Data Problem:** Just as my CTR data has a click rate below 2%, they also face the challenge of detecting minority classes like "Web Attacks" which constitute only about 1% compared to normal traffic. They used **SMOTE (Synthetic Minority Oversampling Technique)**, which reportedly improved the performance of all models.

**Feature Engineering:** The researchers performed correlation-based feature selection with a 0.85 threshold. I currently use 0.95 (0.90 for inter-group) as thresholds for correlation-based feature selection, so it was interesting that they also relied solely on correlation for feature selection.

## Random Forest's Superior Performance

The most impressive part of the paper was Random Forest's performance.

Reasons presented for Random Forest's superiority over deep learning models:

- Better capture of nonlinear relationships in high-dimensional data
- Better handling of class imbalance compared to other models
- Robustness due to ensemble characteristics
- Interpretability and computational efficiency

## Limitations of Deep Learning

Deep learning models showed several limitations:

**Difficulty in Minority Class Detection:** Misclassification still occurred, especially for classes 3 and 4. LSTM showed the best balance but wasn't perfect.

**Computational Cost:** The authors noted that deep learning models require significant computational resources. Latency can be problematic especially in environments requiring real-time detection.

**Overfitting Tendency:** Fluctuations in validation loss were observed in the MLP model, raising concerns about overfitting.

## How to Apply These Insights

After reading this paper, I thought about how to apply these findings.

**Applying SMOTE:** Since SMOTE's effectiveness on imbalanced data has been validated, I should explore whether it can be utilized in my preprocessing pipeline.

**Continuing Correlation Analysis:** While I was somewhat uneasy about selecting features solely based on correlation, it seems this approach won't significantly impact model performance even if other feature selection methods are used, so it can be deprioritized.

**Reconsidering FT-Transformer Use:** I should recognize that traditional methods like Random Forest are still powerful and try applying them.

## A Balanced Perspective on Tabular Data

This paper provides a realistic lesson that deep learning is not always the best choice for tabular data. Particularly:

- **The importance of model selection based on data characteristics**
- **The value of interpretability and computational efficiency**
- **The continued competitiveness of traditional methods**

Both network intrusion detection and click prediction deal with classification problems on tabular data, but I've confirmed once again that the optimal approach can vary depending on specific data characteristics and requirements.

## Conclusion

This research demonstrates that we should avoid hasty generalizations in machine learning model selection. Traditional methods like Random Forest can still show strong performance, and deep learning can bring excessive complexity depending on the situation.

For my CTR competition as well, it would be good to systematically compare Random Forest and other ensemble methods in addition to FT-Transformer. I've learned the lesson that sometimes simpler solutions can be more effective.

---

**Paper Reference:** Ali, M.L.; Thakur, K.; Schmeelk, S.; Debello, J.; Dragos, D. Deep Learning vs. Machine Learning for Intrusion Detection in Computer Networks: A Comparative Study. *Applied Sciences* 2025, 15, 1903. [https://doi.org/10.3390/app15041903](https://doi.org/10.3390/app15041903)
