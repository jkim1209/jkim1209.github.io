---
layout: post
title: "Cross-validation methods"
date: 2025-07-16
categories: [Machine Learning, Data Science, Model Evaluation]
tags: [cross validation, model validation, k-fold, time series, LOOCV, stratified kfold, group kfold, rolling window, expanding window, blocked cv, sklearn, python, model selection, hold out, panel data]
math: true
mermaid: true
---

## 학습/평가 데이터를 나누는 가장 기본적인 **hold-out** 방식

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

> - Hold-out 방식은 데이터 분할에 따라 성능이 크게 달라질 수 있어 일반화 성능이 불안정하다.  
> - Cross-validation은 데이터를 여러 번 나누어 학습함으로써 더 안정적이고 신뢰할 수 있는 성능 평가를 가능케 한다.

## 1. Cross-sectional data 에서 Cross-validation methods

### 1-1. **Leave-One-Out CV (LOOCV)**

- 하나의 데이터를 테스트셋으로, 나머지를 학습셋으로 사용. 이 과정을 모든 데이터에 반복
- 작은 데이터셋에서 학습 능력을 극대화하고 싶을 때

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

- 데이터를 k개의 동일한 크기의 폴드로 나눈 후 각 폴드를 한 번씩 테스트셋으로 사용
- 균형 잡힌 데이터셋에서 보편적으로 사용

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

- 원래 데이터셋의 클래스 비율을 각 폴드에서도 동일하게 유지함
- 클래스 불균형이 있는 분류 문제에서 클래스 비율을 유지하고자 할 때

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

- 동일 그룹이 학습셋과 테스트셋에 동시에 포함되지 않도록 나눔
- 데이터 샘플 간 독립성이 없는 경우 그룹 단위 성능 평가에 적합

<p align="center">
  <img src="/assets/images/machinelearning/groupkfold.png" width="600" alt="groupkfold">
  <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py">Image Source</a>
</p>

```python
import numpy as np
from sklearn.model_selection import GroupKFold

X = np.arange(10)
groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4])  # 같은 그룹끼리는 분리됨
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

## 2. Time series data 에서 Cross-validation methods

look-ahead bias 방지를 위해 일반적인 CV method 사용 불가

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

## 3. 패널 테이터에서의 CV  

[Time-series grouped cross-validation](https://datascience.stackexchange.com/questions/77684/time-series-grouped-cross-validation?answertab=modifieddesc#tab-top)

---

## 🔗 참고 자료  

- [Time-series grouped cross-validation](https://datascience.stackexchange.com/questions/77684/time-series-grouped-cross-validation?answertab=modifieddesc#tab-top)
- [Visualizing cross-validation behavior in scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py)
- [Cross-Validation Clearly Explained in 5 Graphs](https://medium.com/@hahahumble/cross-validation-clearly-explained-in-5-graphs-9b83067bc696)
- [The Ultimate Guide to Time Series CV](https://www.numberanalytics.com/blog/ultimate-guide-time-series-cv)
- [Time Series Analysis: Walk-Forward Validation](https://www.linkedin.com/pulse/time-series-analysis-walk-forward-validation-rafi-ahmed-4v91c)
- [Forecast evaluation for data scientists: common pitfalls and best practices](https://www.researchgate.net/publication/365969672_Forecast_evaluation_for_data_scientists_common_pitfalls_and_best_practices)
- [Backtesting - Cross-Validation for TimeSeries](https://www.kaggle.com/code/cworsnup/backtesting-cross-validation-for-timeseries/notebook)
- [Add rolling window to sklearn.model_selection.TimeSeriesSplit #22523](https://github.com/scikit-learn/scikit-learn/issues/22523)
- [Cross-Validation strategies for Time Series forecasting [Tutorial]](https://www.packtpub.com/en-us/learning/how-to-tutorials/cross-validation-strategies-for-time-series-forecasting-tutorial/)
