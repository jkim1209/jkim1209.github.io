---
layout: post
title: "Cointegration and Error Correction Model: Analyzing Variables That Move Together"
date: 2025-10-17
categories: [Econometrics, Macroeconomics]
tags: [cointegration, VECM, unit root test, error correction model, econometrics]
math: true
---

## 1. Cointegration and Error Correction Model

Most macroeconomic time series variables such as GDP, consumption, and prices show a **common trend** of increasing or decreasing together over time. However, when these non-stationary series variables are used directly in regression analysis, it's easy to fall into the problem of **'spurious regression'** where variables appear statistically significant even though there's actually no causal relationship.

So how should we analyze these variables? The answer can be found in **Cointegration**, which examines whether variables maintain a stable 'equilibrium relationship' and move together in the long run. For example, an individual's income and consumption may be inconsistent in the short term, but in the long term, they don't deviate far from each other.

Now let's assume we've confirmed that two variables are in such a cointegrating relationship. Then how are short-term imbalances adjusted in the future?

The **Error Correction Model (ECM)** answers precisely this question. ECM clearly analyzes the short-term dynamics of how fast variables 'correct' the 'error' and return to equilibrium when they temporarily deviate from long-run equilibrium.

### 1-1. Core Idea

Variables $(Y_t, X_t)$ that are individually non-stationary can have a stable equilibrium relationship in the long run, that is, a **cointegration relationship**. The Engle-Granger method first estimates this long-run equilibrium relationship, then measures the **'error (residual)'** of deviation from that relationship, and models the short-term adjustment process of this error returning to equilibrium.

### 1-2. Analysis

Let's specifically see how to confirm a cointegration relationship and apply the error correction model.

#### Stage 1: Estimate Long-Run Equilibrium Relationship and Calculate Residuals

First, run OLS regression of variable $Y_t$ on $X_t$ to estimate the long-run equilibrium relationship.

$$
Y_t = \beta_0 + \beta_1 X_t + u_t
$$

Calculate the residual $\hat{u}_t$ obtained here. This residual represents precisely the **'degree of deviation from long-run equilibrium (error)'**.

$$
\hat{u}_t = Y_t - \hat{\beta}_0 - \hat{\beta}_1 X_t
$$

#### Stage 2: Cointegration Test Using Residuals and ECM Construction

Now using the 'error' $\hat{u}_t$ obtained in stage 1, proceed with two important analyses.

##### Stage 2-1: Cointegration Test

If the two variables are truly in a cointegrating relationship, the error deviating from long-run equilibrium ($\hat{u}_t$) should be a **stationary series** that doesn't diverge forever but eventually returns to the mean (0). Therefore, perform a **Unit Root Test (e.g., ADF test)** on this residual $\hat{u}_t$.

* **Null hypothesis ($H_0$)**: A unit root exists in the residual. (The residual is non-stationary) $\implies$ **No cointegration relationship.**
* **Alternative hypothesis ($H_1$)**: No unit root in the residual. (The residual is stationary) $\implies$ **A cointegration relationship exists.**

If the null hypothesis is rejected here, we can conclude that a long-run equilibrium relationship exists between the two variables.

##### Stage 2-2: Error Correction Model (ECM) Construction

If the two variables are in a cointegrating relationship, short-term changes ($\Delta Y_t$) can be modeled as showing movements to 'correct' past errors that deviated from long-run equilibrium. At this time, **the past value of the residual obtained in stage 1 ($\hat{u}_{t-1}$)** is included in the regression as an 'error correction term'.

$$
\Delta Y_t = \alpha_0 + \gamma_1 \Delta X_t + \delta \hat{u}_{t-1} + \epsilon_t
$$

* $\Delta Y_t, \Delta X_t$ : First difference values of each variable, meaning **short-term changes**.
* $\hat{u}_{t-1}$ : **Error Correction Term**. That is, it represents the magnitude of imbalance at the previous time point $(t-1)$.
* $\delta$ : **Speed of adjustment**. This coefficient usually has a negative $(-)$ value. For example, if $\delta = -0.2$, it means 20% of the past imbalance is adjusted toward equilibrium at the current time point.

## 2. Extension to Multivariate: Vector Error Correction Model (VECM)

While the Error Correction Model (ECM) mainly deals with the relationship between two variables, **VECM** is a model that **extends this to a system with three or more variables**.

### 2-1. Core Idea

VECM is like a situation where several dogs are tied together on leashes and walking together.

* **Each dog (individual time series variable)**: Each variable such as GDP, consumption, investment can move freely in the short term.
* **Leashes (long-run equilibrium relationship / cointegration)**: However, there are long-run equilibrium relationships (cointegration) that bind variables to each other, so they can't deviate too far from each other.
* **Error Correction**: If one dog gets too far ahead (error occurs), the leash tightens and slows down, or other dogs catch up (error is corrected) to maintain an appropriate distance again.

Thus, VECM is a powerful analytical tool that comprehensively explains how multiple variables in a cointegrating relationship, while influencing each other's changes in the short term, maintain stable equilibrium in the long term.

### 2-2. Mathematical Structure of VECM

VECM is a transformed form of the Vector Autoregression (VAR) model for non-stationary time series vector $Y_t$ as follows.

$$
\Delta Y_t = \Pi Y_{t-1} + \Gamma_1 \Delta Y_{t-1} + \dots + \Gamma_{p-1} \Delta Y_{t-p+1} + \varepsilon_t
$$

* $\Delta Y_{t-i}$ : Represents the effect of **short-term changes** at past time points on current short-term changes ($\Delta Y_t$).
* $\Pi Y_{t-1}$ : The **error correction term**, which is the core of the model. It contains the effect that how much the level at past time point ($t-1$) deviated from long-run equilibrium affects current changes.

This $\Pi$ matrix can be decomposed into the product of two matrices, $\Pi = \alpha\beta'$.

$$
\Pi Y_{t-1} = \alpha(\beta' Y_{t-1})
$$

* $\beta$ : **Cointegrating vector**, which defines the **long-run equilibrium relationship** among variables. $\beta' Y_{t-1}$ is precisely the term representing the past imbalance (error).
* $\alpha$ : **Adjustment coefficient** vector, which represents the **adjustment speed** of each variable returning to equilibrium when an error occurs deviating from long-run equilibrium.

### 2-3. How to Confirm Cointegration Relationship? - Johansen Test

To use VECM, we must first confirm whether there is a cointegrating relationship among variables, and if so, how many. In systems with 3 or more variables, using the **Johansen Test** is the standard method.

The Johansen test statistically tests the number of independent long-run equilibrium relationships ($\beta$) existing in the system by analyzing the rank of the $\Pi$ matrix. If this test rejects the null hypothesis that the number of cointegrating relationships is 0, we secure the basis for using VECM.

## 3. Examining the Operating Principle of VECM in Detail

As mentioned earlier, VECM is expressed as the following matrix equation transforming VAR (Vector Autoregression model).

$$
\Delta Y_t = \Pi Y_{t-1} + \Gamma_1 \Delta Y_{t-1} + \dots + \Gamma_{p-1} \Delta Y_{t-p+1} + \varepsilon_t
$$

This matrix equation is actually **a system that expresses multiple regression equations at once**. Let's unpack this equation through a specific example where 3 variables (GDP, consumption C, investment I) are in a cointegrating relationship.

### 3-1. Definition of Vectors

First, the vector $Y_t$ used in VECM is a column vector consisting of variables included in the system.

$$
Y_t =
\begin{pmatrix}
GDP_t \\
C_t \\
I_t
\end{pmatrix},
\quad
\Delta Y_t =
\begin{pmatrix}
\Delta GDP_t \\
\Delta C_t \\
\Delta I_t
\end{pmatrix}
$$

### 3-2 Dissecting the Error Correction Term $(\Pi Y_{t-1})$

The core of VECM is the error correction term that corrects errors deviating from long-run equilibrium. This term is decomposed as $\Pi = \alpha\beta'$.

① Long-run equilibrium relationship $(\beta)$ and error $(\beta' Y_{t-1})$

* The cointegrating vector $(\beta)$ defines the long-run equilibrium relationship among variables. For example, if there is one long-run equilibrium relationship in the system, $\beta$ is as follows. (Normalize the first variable to 1)

$$
\beta =
\begin{pmatrix}
1 \\ -\beta_2 \\ -\beta_3
\end{pmatrix}
$$

* Multiplying this vector by past variable $Y_{t-1}$ calculates **a single value** $EC_{t-1}$ representing the 'error' deviating from long-run equilibrium.

$$
EC_{t-1} = \beta' Y_{t-1} \\
         = GDP_{t-1} - \beta_2 C_{t-1} - \beta_3 I_{t-1}
$$

Here $EC_{t-1}$ is precisely the **imbalance at past time point ($t-1$)**.

② Adjustment speed $(\alpha)$

* The adjustment coefficient vector $(\alpha)$ represents how quickly each variable reacts to this imbalance $(EC_{t-1})$ and returns to equilibrium.

$$
\alpha =
\begin{pmatrix}
\alpha_{GDP} \\ \alpha_C \\ \alpha_I
\end{pmatrix}
$$

### 3-3. Complete VECM Regression Equation System

Now combining the above elements, the VECM matrix equation actually means the following **system of 3 regression equations**. (Only 1 lag is shown for simplification)

**[Equation 1: Regression for $\Delta GDP_t$]**

$$
\Delta GDP_t = \alpha_{GDP} (EC_{t-1}) + \gamma_{11} \Delta GDP_{t-1} + \gamma_{12} \Delta C_{t-1} + \gamma_{13} \Delta I_{t-1} + \varepsilon_{GDP,t}
$$

**[Equation 2: Regression for $\Delta C_t$]**

$$
\Delta C_t = \alpha_C (EC_{t-1}) + \gamma_{21} \Delta GDP_{t-1} + \gamma_{22} \Delta C_{t-1} + \gamma_{23} \Delta I_{t-1} + \varepsilon_{C,t}
$$

**[Equation 3: Regression for $\Delta I_t$]**

$$
\Delta I_t = \alpha_I (EC_{t-1}) + \gamma_{31} \Delta GDP_{t-1} + \gamma_{32} \Delta C_{t-1} + \gamma_{33} \Delta I_{t-1} + \varepsilon_{I,t}
$$

When viewed this way, it becomes clear that VECM is a sophisticated model explaining the short-term change of each variable ($\Delta Y_t$) as **(1) response to past imbalance ($EC_{t-1}$)** and **(2) response to past short-term changes of other variables ($\Delta Y_{t-i}$)**.

In actual analysis, statistical packages estimate this complex system at once through the Johansen procedure, etc., to find $\alpha$ and $\beta$ values.

## 4. Integrated Analysis Encompassing Long-Run Equilibrium and Short-Run Adjustment

Cointegration and error correction models go beyond simply being technical techniques to solve the spurious regression problem of non-stationary time series data. This methodology gives mathematical rigor to the powerful economic intuition that 'variables find stable equilibrium in the long run'.

Through this framework, we can catch meaningful long-run equilibrium relationships ($\beta$) even in a sea of non-stationary data. Furthermore, ECM and VECM dynamically depict at what speed ($\alpha$) and through what interactions economic agents return to equilibrium when they deviate from it in response to short-term shocks.

Ultimately, these models are powerful narrative tools that unravel the **long-term directionality** and **short-term volatility** of complexly intertwined macroeconomic systems into one integrated story.

---

_**Source: My own Lecture Notes from Graduate School - Macroeconometrics**_
