---
layout: post
title: "Residual Diagnostics in Linear Regression"
date: 2025-07-02
categories: [Machine Learning, Statistics, Regression Analysis]
tags: [linear regression, residual analysis, diagnostics, outlier detection, studentized residuals, hat matrix, leverage, model validation, statistical assumptions, homoscedasticity, cross validation, influence measures, regression diagnostics, data analysis, python, statsmodels]
math: true
---

## Residuals

Recall the linear regression model:

$$
Y_i = \beta_0 + \beta_1 x_{i1} + \beta_1 x_{i2} + \cdots + \beta_{p-1} x_{i,p-1} + \epsilon_i, \quad i = 1, 2, \ldots, n    \tag{1}
$$

with  

$$
\epsilon_i \overset{\text{iid}}{\sim} N(0, \sigma^2).
$$

* What major assumptions are we making?  
  * The response function  $E[Y]$ is linear.  
  * The errors $\epsilon$ are normally distributed.  
  * The errors $\epsilon$ have constant variance (**Homoscedasticity**).  
  * The errors $\epsilon$ are independent and identically distributed.  

* Other considerations
  * Does the model fit all but one or a few outlying observations?
  * Have one or several important predictor variables been omitted from the model?

Notice the true errors can be expressed as:

$$
\epsilon_i = Y_i - E[Y_i] = Y_i - (\beta_0 + \beta_1 x_i), \quad i = 1, 2, \ldots, n.
$$

> **Note**: The errors are unobservable.  
> **Note**: The *best guess* of the true errors are the residuals $\quad \hat{\epsilon}_i = e_i$, $\quad i = 1, \ldots, n$.

**Definition**  
Then the *i*th **residual** is defined by:

$$
e_i = Y_i - \hat{Y}_i, \quad i = 1, 2, \ldots, n.   \tag{2}
$$

Analyzing the residuals provides insight on whether or not the regression assumptions are satisfied.

* Sample mean and sample variance of the residuals

$$
\bar{e} = \frac{1}{n} \sum_{i=1}^{n} e_i = 0
\quad\text{and}\quad
\hat{\sigma}^2 = \frac{1}{n - p} \sum_{i=1}^{n} e_i^2 = MSE
$$

* Although the errors $\epsilon_i$ are independent random variables, the **residuals are not independent** random variables.  
This can be seen by the following two properties:

$$
\sum_{i=1}^{n} e_i = 0
\quad \text{and} \quad
\sum_{i=1}^{n} x_{ik} e_i = 0, \quad k = 1, 2, \ldots, p - 1.
$$

---

## **Semistudentized residuals**

**Definition**  
Let $e_i$ be the residual defined in $(2)$ and let $MSE$ be the mean square error.  
Then the *i*th **semistudentized residual** is defined by:  

$$
e_i^* = \frac{e_i - \bar{e}}{\hat{\sigma}} = \frac{e_i}{\hat{\sigma}}, \quad i = 1, 2, \ldots, n. \tag{3}
$$

---

## **Studentized residuals**

Recall that the residual vector can be expressed as

$$
\mathbf{e} = (\mathbf{I}_n - \mathbf{H})\mathbf{Y}
$$

where the hat matrix $\mathbf{H}$ is defined by

$$
\mathbf{H} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T.
$$

*The mean and variance of the residual vector* $\mathbf{e}$ *are respectively*

$$
E[\mathbf{e}]=\mathbf{0}
\quad\text{and}\quad
Var[\mathbf{e}]=\sigma^2(\mathbf{I}_n-\mathbf{H})
$$

Consequently, the *i*th **studentized residual** is defined by:

$$
t_i = \frac{e_i}{\sqrt{\hat{\sigma}^2 (1 - h_{ii})}}, \tag{4}
$$

where $h_{ii}$ is the *i*th diagonal element of the hat matrix $\mathbf{H}$.

> **Note**: If $h_{ii} \approx 1 \Rightarrow  t_i$ is big ($h_{ii}$ is close to 1 for an outlier)

---

## **Deleted residuals**

A useful refinement to make residuals more effective for detecting outlying $Y$ observations is to measure the *i*th residual $e_i$ when the fitted regression is based on all of the cases except the *i*th one.
Denote

* $\hat{Y}_{(i)}$: the fitted regression equation based on all cases except the *i*th one.  
* $\hat{Y_i\(i)}$ the *i*th fitted response value based on the predicted model $\hat{Y_\(i)}$

Consequently, the **deleted residual** denoted $d_i$ is defined by  

$$
d_i = Y_i - \hat{Y}_{i(i)}. \tag{5}
$$

<center><img src='/assets/images/machinelearning/Residual_diagnostics_deleted_residual.jpg' width = 600 alt="Residual_diagnostics_deleted_residual"></center>

> **Note**: This relates to  leave-one-out cross validation.

---

We want to studentize (or standardize) the deleted residuals, i.e., we want to find an expression for

$$
t_{(i)} = \frac{d_i}{\hat{\sigma}_{d_i}}.
$$

An algebraically equivalent expression for $d_i$ that does **not** require a recomputation of the fitted regression function omitting the *i*th case is:

$$
d_i = \frac{e_i}{1 - h_{ii}}. \tag{6}
$$

Define $\hat{\sigma}^2_{(i)}$ as the mean square error based on all cases except the *i*th one.  

The following equation analogous to $(6)$ relates $\hat{\sigma}^2_{(i)}$ with the regular
$\hat{\sigma}^2$:

$$
(n - p) \hat{\sigma}^2 = (n - p - 1) \hat{\sigma}^2_{(i)} + \frac{e_i^2}{1 - h_{ii}}.
$$

## **Studentized deleted residuals**

Using the above relation, the **studentized deleted residuals** can be expressed as:

$$
t_{(i)} = \frac{d_i}{\hat{\sigma}_{d_i}} = e_i \sqrt{ \frac{n - p - 1}{SSE(1 - h_{ii}) - e_i^2} }. \tag{7}
$$

Uses of studentized deleted residuals  

* Using the deleted studentized residuals in diagnostic plots is a common technique of validating the regression assumptions.  
* The deleted studentized residuals are particularly useful in identifying outlying $Y$ values.  

---

**Proof**. Derivation of the Studentized Deleted Residual

The **Studentized Deleted Residual** (also called the **Externally Studentized Residual**) is defined as:

$$
t_{(i)} = \frac{e_i}{\hat{\sigma}_{(i)} \sqrt{1 - h_{ii}}}
$$

Where:

* $e_i = y_i - \hat{y}_i$: the residual of observation $i$  
* $h_{ii}$: the leverage of observation $i$ (diagonal of the hat matrix)  
* $\hat{\sigma}_{(i)}^2$: the variance of residuals calculated by excluding observation $i$  

Step 1: Express $\hat{\sigma}_{(i)}^2$

We estimate the leave-one-out residual variance as:

$$
\hat{\sigma}_{(i)}^2 = \frac{\text{SSE}_{(i)}}{n - p - 1}
$$

We can approximate the leave-one-out sum of squared errors as:

$$
\text{SSE}_{(i)} = \text{SSE} - \frac{e_i^2}{1 - h_{ii}}
$$

Thus:

$$
\hat{\sigma}_{(i)}^2 = \frac{\text{SSE} - \frac{e_i^2}{1 - h_{ii}}}{n - p - 1}
$$

Step 2: Substitute into the definition of $t_i$

Plug into the original formula:

$$
t_i = \frac{e_i}{\sqrt{ \hat{\sigma}_{(i)}^2 (1 - h_{ii}) }}
= \frac{e_i}{\sqrt{ \left( \frac{\text{SSE} - \frac{e_i^2}{1 - h_{ii}}}{n - p - 1} \right)(1 - h_{ii}) }}
$$

Step 3: Simplify the expression

Multiply terms in the denominator:

$$
( \text{SSE} - \frac{e_i^2}{1 - h_{ii}} )(1 - h_{ii}) = \text{SSE}(1 - h_{ii}) - e_i^2
$$

So we arrive at:

$$
t_i = e_i \cdot \sqrt{ \frac{n - p - 1}{ \text{SSE}(1 - h_{ii}) - e_i^2 } }
$$

This formula allows you to compute the Studentized Deleted Residual **without explicitly re-fitting the model for each observation**, making it highly efficient for outlier diagnostics.

---

Use the studentized or deleted studentized residuals to construct residual plots.  
Some recommendations follow:

* **Scatter plot matrix of all variables.**  
  *Linearity, Constant Variance, General exploratory analysis*

* **Plot of the studentized residuals against all (or some) of the predictor variables $X$.**  
  *Linearity, Constant Variance, Normality, Independence*

* **Plot of the studentized residuals against fitted values.**  
  *Same as the previous plot*

* **Plot of the studentized residuals against time or other sequence.**  
  *Independence, Normality*

* **Plots of the studentized residuals against omitted predictor variables.**  
  *Model validation*

* **Box plot (or histogram) of the studentized residuals.**  
  *Normality*

* **Normal probability plot of studentized residuals.**  
  *Normality, Linearity*

---

## **Summary**

| Term                                                                                   | Symbol    | Formula                                            | Description                                                                                            |
| -------------------------------------------------------------------------------------- | --------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Residual**                                                                           | $e_i$     | $e_i = Y_i - \hat{Y}_i$                            | The raw error between the observed value and the predicted value.                                      |
| **Semistudentized Residual** <br> (Semistandardized Residual)                          | $e_i^*$   | $\dfrac{e_i}{\hat{\sigma}}$                        | Raw residual divided by the overall RMSE (ignores  leverage).                                          |
| **Studentized Residual** <br> (Internally Studentized Residual, Standardized Residual) | $t_i$     | $\dfrac{e_i}{\hat{\sigma} \sqrt{1 - h_{ii}}}$      | Adjusts residual by both RMSE and leverage                                                             |
| **Deleted Residual**                                                                   | $e_{(i)}$ | $Y_i - \hat{Y}_{(i)}$                              | The residual when observation $i$ is **left out** from model fitting (leave-one-out prediction error). |
| **Studentized Deleted Residual** <br> (Externally Studentized Residual)                | $t_{(i)}$ | $\dfrac{e_i}{\hat{\sigma_\(i)} \sqrt{1 - h_{ii}}}$ | Residual scaled by RMSE computed **excluding observation $i$**. Most reliable for outlier detection.   |

**Notes**
* $\hat{\sigma}$: The standard deviation of residuals computed using all observations (i.e., the RMSE).  
* $\hat{\sigma}_{(i)}$: The standard deviation of residuals computed by excluding observation $i$ from the model. This is used in externally studentized residuals.  
* $h_{ii}$: The leverage of observation $i$ — it represents how much influence observation $i$ has on its own predicted value (i.e., the diagonal element of the hat matrix $\mathbf{H} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$).  

**Practical Tips**
* In real-world applications, the Studentized Deleted Residual (also called Externally Studentized Residual) is most commonly used for outlier detection, because it accounts for the influence of the observation being tested.  
* The Deleted Residual (i.e., the prediction error when leaving out a point) is important for assessing predictive performance, especially in cross-validation contexts.  
* The Semistudentized Residual is less commonly used in practice, but it serves as a basic form of normalization (dividing residuals by the global RMSE).  

---

## **Example**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# Simulate new data with an internal outlier
np.random.seed(42)
n = 30
x = np.linspace(0, 10, n)
y = 3 * x + np.random.normal(0, 2, n)

# Add an outlier within the range of x but with a strange y value
x = np.append(x, 5.5)             # within x range
y = np.append(y, 30)              # y value inconsistent with trend at x=5.5

# Create new DataFrame
data_internal = pd.DataFrame({'x': x, 'y': y})
X_internal = sm.add_constant(data_internal['x'])
model_internal = sm.OLS(data_internal['y'], X_internal).fit()

# Get influence measures
influence_internal = OLSInfluence(model_internal)
studentized_residuals_internal = influence_internal.resid_studentized_external

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(studentized_residuals_internal)), studentized_residuals_internal, color='blue')
ax.axhline(y=3, color='red', linestyle='--', label='Threshold (+3)')
ax.axhline(y=-3, color='red', linestyle='--', label='Threshold (-3)')
ax.set_xlabel('Observation Index')
ax.set_ylabel('Studentized Deleted Residual')
ax.set_title('Studentized Deleted Residual Plot')

# Highlight potential outliers
outlier_indices_internal = np.where(np.abs(studentized_residuals_internal) > 3)[0]
for idx in outlier_indices_internal:
    ax.annotate(f"{idx}", (idx, studentized_residuals_internal[idx]), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
ax.legend()

plt.tight_layout()
plt.show()

# Return suspected outlier info
print('Outlier Info: \n', data_internal.iloc[outlier_indices_internal])
```

<img src='/assets/images/machinelearning/Studentized_Deleted_Residual_Plot.png' width = 600 alt="Studentized_Deleted_Residual_Plot">

```txt
Outlier Info: 
       x     y
30  5.5  30.0
```

## **Note**: Understanding Hat Matrix

In linear regression, the predicted values can be written as:

$$
\mathbf{\hat{Y}} = \mathbf{X} \hat{\mathbf{\beta}} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{H} \mathbf{Y}
$$

Here, $\mathbf{H} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$ is called the **hat matrix**. It maps the observed values $\mathbf{Y}$ to the predicted values $\mathbf{\hat{Y}}$:

$$
\hat{Y}_i = \sum_{j=1}^n h_{ij} Y_j
$$

The diagonal element $h_{ii}$ shows how much the prediction $\hat{Y}_i$ depends on its own observed value $Y_i$:

$$
\hat{Y}_i = h_{ii} Y_i + \sum_{j \ne i} h_{ij} Y_j
$$

### Why is $h_{ii}$ called "Leverage"?

* $h_{ii}$ tells us **how much influence observation $i$ has on its own predicted value**.  
* If $h_{ii}$ is **large**, then $y_i$ contributes a **large weight** to predicting $\hat{Y}_i$.  
* This usually happens when $x_i$ is **far from the center of the predictor space**.  
* Such points are said to have **high leverage**, because they can **pull the regression line** toward themselves.  

### Properties of $h_{ii}$

* $0 < h_{ii} < 1$  
* $\sum_{i=1}^n h_{ii} = p$  (where $p$ is the number of model parameters including the intercept)  
* Average leverage: $\displaystyle \bar{h} = \frac{p}{n}$  
* A rule of thumb: if $h_{ii} > 2\bar{h}$ or $3\bar{h}$, the point may be considered **high leverage**  

### High Leverage ≠ Outlier

* A high $h_{ii}$ doesn't necessarily mean the point is an outlier.  
* To detect **influential outliers**, we need to look at both:  
  * The **residual** $e_i$  
  * The **leverage** $h_{ii}$  
  * → This is what **studentized deleted residuals** are designed to detect.  

### **Summary**

* **$h_{ii}$ (leverage)** measures how much the predicted value $\hat{Y}_i$ is influenced by the actual observation $Y_i$.  
* High leverage points are far from the center of the $x$-space and can strongly affect the regression fit.  
