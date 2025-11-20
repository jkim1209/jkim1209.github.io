---
layout: post
title: "Cross-validation methods"
date: 2025-07-16
categories: [Machine Learning, Data Science, Model Evaluation]
tags: [cross validation, model validation, k-fold, time series, LOOCV, stratified kfold, group kfold, rolling window, expanding window, blocked cv, sklearn, python, model selection, hold out, panel data]
math: true
mermaid: true
---

## 0. Splitting the entire dataset into training and evaluation data is necessary for model training

### 0-1. The most basic **hold-out** method for splitting training/evaluation data

<p align="center">
  <img src="/assets/images/machinelearning/holdout.webp" width="600" alt="holdout">
  <a href="https://medium.com/@hahahumble/cross-validation-clearly-explained-in-5-graphs-9b83067bc696">Image Source</a>
</p>

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.arange(15)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)
print("TRAIN:", X_train)
print("TEST:", X_test)

# TRAIN: [ 0  5  9  8 11  3  1  6 12  2 13 14]
# TEST: [ 7 10  4]
```

> - The hold-out method can result in significantly different performance depending on data splitting, leading to unstable generalization performance.
> - Cross-validation enables more stable and reliable performance evaluation by training on multiple data splits.

## 1. Cross-validation methods for cross-sectional data

### 1-1. **Leave-One-Out CV (LOOCV)**

- Uses one data point as test set and the rest as training set. Repeats this process for all data points
- When you want to maximize learning capability on small datasets

<p align="center">
  <img src="/assets/images/machinelearning/LOOCV.webp" width="600" alt="LOOCV">
  <a href="https://medium.com/@hahahumble/cross-validation-clearly-explained-in-5-graphs-9b83067bc696">Image Source</a>
</p>

```python
import numpy as np
from sklearn.model_selection import LeaveOneOut

X = np.arange(5)

for train_idx, test_idx in LeaveOneOut().split(X):
    print("TRAIN:", train_idx, "/ Values:", X[train_idx])
    print("TEST: ", test_idx,  "/ Value: ", X[test_idx])
    print()

# TRAIN: [1 2 3 4] / Values: [1 2 3 4]
# TEST:  [0] / Value:  [0]

# TRAIN: [0 2 3 4] / Values: [0 2 3 4]
# TEST:  [1] / Value:  [1]

# TRAIN: [0 1 3 4] / Values: [0 1 3 4]
# TEST:  [2] / Value:  [2]

# TRAIN: [0 1 2 4] / Values: [0 1 2 4]
# TEST:  [3] / Value:  [3]

# TRAIN: [0 1 2 3] / Values: [0 1 2 3]
# TEST:  [4] / Value:  [4]
```

### 1-2. **Standard K-Fold CV**

- Divides data into k equal-sized folds, then uses each fold once as test set
- Commonly used for balanced datasets

<p align="center">
  <img src="/assets/images/machinelearning/kfold.png" width="600" alt="kfold">
  <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py">Image Source</a>
</p>

```python
import numpy as np
from sklearn.model_selection import KFold

X = np.arange(15)
kf = KFold(n_splits=3, shuffle=True, random_state=123)
for train_idx, test_idx in kf.split(X):
    print("TRAIN:", train_idx, "TEST:", test_idx)

# TRAIN: [ 1  2  3  6  8  9 11 12 13 14] TEST: [ 0  4  5  7 10]
# TRAIN: [ 0  2  4  5  6  7 10 12 13 14] TEST: [ 1  3  8  9 11]
# TRAIN: [ 0  1  3  4  5  7  8  9 10 11] TEST: [ 2  6 12 13 14]
```

### 1-3. **Stratified K-Fold CV**

- Maintains the same class proportions in each fold as in the original dataset
- When you want to preserve class proportions in classification problems with class imbalance

<p align="center">
  <img src="/assets/images/machinelearning/stratifiedkfold.png" width="600" alt="stratifiedkfold">
  <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py">Image Source</a>
</p>

```python
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold

X = np.arange(15)
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

for train_idx, test_idx in skf.split(X, y):
    y_train = np.array(y)[train_idx]
    y_test = np.array(y)[test_idx]

    print("TRAIN:", train_idx, " / Class count:", dict(Counter(y_train)))
    print("TEST: ", test_idx,  " / Class count:", dict(Counter(y_test)))
    print()

# TRAIN: [ 1  2  4  5  6  7  9 10 11 13]  / Class count: {0: 3, 1: 7}
# TEST:  [ 0  3  8 12 14]  / Class count: {0: 2, 1: 3}

# TRAIN: [ 0  2  3  5  6  8 10 11 12 14]  / Class count: {0: 3, 1: 7}
# TEST:  [ 1  4  7  9 13]  / Class count: {0: 2, 1: 3}

# TRAIN: [ 0  1  3  4  7  8  9 12 13 14]  / Class count: {0: 4, 1: 6}
# TEST:  [ 2  5  6 10 11]  / Class count: {0: 1, 1: 4}
```

### 1-4. **Group K-Fold CV**

- Splits data so that the same group is not included in both training and test sets simultaneously
- Suitable for group-level performance evaluation when data samples lack independence

<p align="center">
  <img src="/assets/images/machinelearning/groupkfold.png" width="600" alt="groupkfold">
  <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py">Image Source</a>
</p>

```python
import numpy as np
from sklearn.model_selection import GroupKFold

X = np.arange(10)
groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4])  # Same groups are separated
gkf = GroupKFold(n_splits=3, shuffle=True, random_state=123)

for train_idx, test_idx in gkf.split(X, groups=groups):
    print("TRAIN:", train_idx, " / Group:", np.unique(groups[train_idx]))
    print("TEST: ", test_idx,  " / Group:", np.unique(groups[test_idx]))
    print()

# TRAIN: [3 4 5 6 7]  / Class: [2 3]
# TEST:  [0 1 2 8 9]  / Class: [1 4]

# TRAIN: [0 1 2 5 6 7 8 9]  / Class: [1 3 4]
# TEST:  [3 4]  / Class: [2]

# TRAIN: [0 1 2 3 4 8 9]  / Class: [1 2 4]
# TEST:  [5 6 7]  / Class: [3]
```

## 2. Cross-validation methods for time series data

Cannot use general CV methods to prevent look-ahead bias

### 2-1.  **Rolling Window CV**

<p align="center">
  <img src="/assets/images/machinelearning/slidingwindowCV.png" width="600" alt="slidingwindowCV">
  <a href="https://www.kaggle.com/code/cworsnup/backtesting-cross-validation-for-timeseries/notebook">Image Source</a>
</p>

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

x = np.arange(15)
cv = TimeSeriesSplit(max_train_size=3, test_size=1)
for train_index, test_index in cv.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)

# TRAIN: [7 8 9] TEST: [10]
# TRAIN: [8 9 10] TEST: [11]
# TRAIN: [9 10 11] TEST: [12]
# TRAIN: [10 11 12] TEST: [13]
# TRAIN: [11 12 13] TEST: [14]
```

### 2-2.**Expanding Window CV**

<p align="center">
  <img src="/assets/images/machinelearning/expandingwindowCV.png" width="600" alt="expandingwindowCV">
  <a href="https://www.kaggle.com/code/cworsnup/backtesting-cross-validation-for-timeseries/notebook">Image Source</a>
</p>

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

x = np.arange(15)
cv = TimeSeriesSplit(n_splits=8, test_size=1)
for train_index, test_index in cv.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)

# TRAIN: [0 1 2 3 4 5 6] TEST: [7]
# TRAIN: [0 1 2 3 4 5 6 7] TEST: [8]
# TRAIN: [0 1 2 3 4 5 6 7 8] TEST: [9]
# TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10]
# TRAIN: [0 1 2 3 4 5 6 7 8 9 10] TEST: [11]
# TRAIN: [0 1 2 3 4 5 6 7 8 9 10 11] TEST: [12]
# TRAIN: [0 1 2 3 4 5 6 7 8 9 10 11 12] TEST: [13]
# TRAIN: [0 1 2 3 4 5 6 7 8 9 10 11 12 13] TEST: [14]
```

### 2-3. **Blocked CV**

<p align="center">
  <img src="/assets/images/machinelearning/blockedCV.jpg" width="600" alt="blockedCV">
  <a href="https://www.packtpub.com/en-us/learning/how-to-tutorials/cross-validation-strategies-for-time-series-forecasting-tutorial/">Image Source</a>
</p>

```python
import numpy as np

# Source code from: https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
class BlockedTimeSeriesSplit():
    def __init__(self, n_splits):
        # Define
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        # Split data using a blocked strategy
        # and return indices of where to split
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]



btscv = BlockedTimeSeriesSplit(n_splits=3)
X = np.arange(15)

for train_index, test_index in btscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

# TRAIN: [0 1 2 3] TEST: [4]
# TRAIN: [5 6 7 8] TEST: [9]
# TRAIN: [10 11 12 13] TEST: [14]
```

## 3. CV for panel data

[Time-series grouped cross-validation](https://datascience.stackexchange.com/questions/77684/time-series-grouped-cross-validation?answertab=modifieddesc#tab-top)

---

## 🔗 References

- [Time-series grouped cross-validation](https://datascience.stackexchange.com/questions/77684/time-series-grouped-cross-validation?answertab=modifieddesc#tab-top)
- [Visualizing cross-validation behavior in scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py)
- [Cross-Validation Clearly Explained in 5 Graphs](https://medium.com/@hahahumble/cross-validation-clearly-explained-in-5-graphs-9b83067bc696)
- [The Ultimate Guide to Time Series CV](https://www.numberanalytics.com/blog/ultimate-guide-time-series-cv)
- [Time Series Analysis: Walk-Forward Validation](https://www.linkedin.com/pulse/time-series-analysis-walk-forward-validation-rafi-ahmed-4v91c)
- [Forecast evaluation for data scientists: common pitfalls and best practices](https://www.researchgate.net/publication/365969672_Forecast_evaluation_for_data_scientists_common_pitfalls_and_best_practices)
- [Backtesting - Cross-Validation for TimeSeries](https://www.kaggle.com/code/cworsnup/backtesting-cross-validation-for-timeseries/notebook)
- [Add rolling window to sklearn.model_selection.TimeSeriesSplit #22523](https://github.com/scikit-learn/scikit-learn/issues/22523)
- [Cross-Validation strategies for Time Series forecasting [Tutorial]](https://www.packtpub.com/en-us/learning/how-to-tutorials/cross-validation-strategies-for-time-series-forecasting-tutorial/)
