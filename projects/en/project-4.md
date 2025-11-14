# CTR Prediction Project: Toss Ad Click Prediction

**Developed a model predicting ad click probability by combining user behavior logs and other available data. Integrated user sequence LSTM embeddings into FT-Transformer-based architecture to improve prediction stability and performance, achieving top 10% on the leaderboard.**

## Project Overview

This project develops a click-through rate prediction model analyzing user behavior patterns from complex tabular datasets.

GitHub Repository: [https://github.com/jkim1209/toss-ctr-prediction](https://github.com/jkim1209/toss-ctr-prediction)

## My Role and Contributions

Led data analysis and modeling, playing a key role in boosting prediction performance.

- Data Analysis & Preprocessing: Identified complex correlations among numerous features through initial data analysis, establishing and leading preprocessing and feature engineering strategies that became the core of the project.
- Model Implementation & Experimentation: Selected FT-Transformer as the main model and implemented a hybrid model structure that extracts latent features from sequence information by combining LSTM.

## Key Technologies and Implementation

### Technologies Used

- Framework: Python, PyTorch, Polars, Pandas, NumPy, Scikit-learn
- Models: FT-Transformer, LSTM

### Core Implementation

- Large-scale data processing in Parquet format and feature engineering pipeline
- FT-Transformer-based classification model training with 5-Fold cross-validation
- User behavior sequence embedding generation using LSTM

![Model Architecture](/projects/assets/images/04/01.png)

![Feature Engineering](/projects/assets/images/04/02.png)

![Training Process](/projects/assets/images/04/03.png)

## Troubleshooting

### Problem: Breaking Through Performance Plateau

**Problem Description**

In early model training, the evaluation metric score plateaued around 0.3450 with no further improvement. At the time, user behavior sequence information was only being utilized as simple statistics like sequence start, end, and length.

**Solution**

1. Determined that the model wasn't fully leveraging the sequential, dynamic information contained in the sequence data.
2. **Introduced LSTM model** to transform each user's entire behavior sequence into a single compressed embedding vector. This embedding vector encapsulates the sequence's order and pattern information and is input to the model along with other static features.
3. As a result, the evaluation metric score rose **above 0.3480**, breaking through the performance plateau.

## Results and Achievements

### Quantitative Results

Achieved **top 10%** (70th place) in Toss CTR prediction competition

- Private Score: 0.34814
- Baseline Score: 0.33185

*Score = Average of Average Precision and Weighted Log Loss

### Learnings and Insights

- Experienced the complete process of applying the state-of-the-art deep learning architecture FT-Transformer to large-scale tabular data and building a reproducible experimental pipeline for large-scale data processing.
- Directly confirmed that feature engineering using models like LSTM as feature extractors can significantly impact model performance on data where time series or sequential information is important.

![Final Results](/projects/assets/images/04/04.png)
