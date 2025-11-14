---
layout: post
title: "Difference-in-Differences (DiD)"
date: 2025-11-11
categories: [Econometrics, Causal Inference]
tags: [DiD, Difference-in-Differences, Diff-in-Diff, causal inference, econometrics, microeconometrics, staggered DiD, synthetic control]
math: true
---

## Introduction: The Limitations of OLS and the Dilemma of Causal Inference

"Did the opening of a new subway line raise housing prices in Area A?"
"Did the minimum wage increase lower employment rates in State B?"

The simple comparison methods that first come to mind when analyzing policy effects often lead to errors.

- **Simple Before-After Comparison**: Comparing housing prices in Area A before and after the policy may mistake city-wide common trends for policy effects.
- **Simple Treated-Control Comparison**: Comparing Area A with a subway to Area C without one may misinterpret pre-existing inherent differences (time-invariant differences) as policy effects.

The method designed to simultaneously eliminate these two confounding factors—changes over time and inherent group differences—is precisely **Difference-in-Differences (DiD)**.

---

## 1. The Core Logic of DiD: Two Differences

DiD, as its name suggests, estimates the pure policy effect through two differences (subtractions).

### First Difference: Removing 'Time Effects' Within Each Group

- **Change in Treated Group ($\Delta Y_T$)**
  = [Post-treatment average of treated group] - [Pre-treatment average]
  = Policy effect + Time change effect
- **Change in Control Group ($\Delta Y_C$)**
  = [Post-treatment average of control group] - [Pre-treatment average]
  = Time change effect only

### Second Difference: Isolating 'Pure Effect' Through Group Differences

> **DiD Estimator = (Change in Treated Group) - (Change in Control Group)**
> $$
> \hat{\delta}_{DiD} = (Y_{T,After} - Y_{T,Before}) - (Y_{C,After} - Y_{C,Before})
> $$

By subtracting the control group's change (time effect), the time effect mixed in the treated group's change is removed, ultimately leaving only the pure policy effect (ATT: Average Treatment Effect on the Treated).

<div align="center">
  <img src="/assets/images/math/did_parallel_trends.png" width="800" alt="Difference-in-Differences Parallel Trends">
  <br>
  <a href="https://patrickthiel.com/the-great-regression-with-python-difference-in-differences-regressions/" target="_blank" rel="noopener noreferrer">
    Graphical representation of DiD idea
  </a>
</div>

---

## 2. Implementing DiD Through Regression Analysis

The logic of difference-in-differences can also be expressed through a simple linear regression equation.

$$
Y_{it} = \beta_0 + \beta_1 \text{Treated}_i + \beta_2 \text{After}_t + \delta (\text{Treated}_i \times \text{After}_t) + \mathbf{X}'_{it}\gamma + \epsilon_{it}
$$

| Coefficient | Meaning                                                      | Interpretation            |
| :---------- | :----------------------------------------------------------- | :------------------------ |
| $\beta_0$   | Control group average at baseline                            | Intercept                 |
| $\beta_1$   | Inherent difference between groups (time-invariant level difference) | time-invariant difference |
| $\beta_2$   | Common change over time                                      | Common time trend         |
| $\delta$    | Additional effect appearing only in treated group after policy | DiD estimator (ATT)       |

The estimated coefficient $\hat{\delta}$ of the interaction term $(\text{Treated}_i \times \text{After}_t)$ is precisely the DiD estimator, and its statistical significance can be tested using the p-value.

---

## 3. The Core Assumption of DiD: Parallel Trends Assumption

For DiD analysis to be valid, the following assumption must hold:

> "In the absence of treatment, the treated and control groups would have followed parallel trends over time."

If this assumption is violated, differences in trends between the two groups may be misinterpreted as policy effects.

- **Difficulty of Verification**: Perfect verification is impossible because we cannot directly observe the counterfactual situation.
- **Proxy Verification Method**: By obtaining data from multiple time points before policy implementation, we can check visually (graphs) or statistically whether the two groups actually showed parallel trends during that period.

---

## 4. Classic Case Study: Card & Krueger (1994)

A representative application of DiD is Card & Krueger's (1994) study analyzing minimum wage and employment.

- **Policy**: In 1992, New Jersey (NJ) raised its minimum wage from $4.25 to $5.05.
- **Treated Group**: Fast-food restaurants in New Jersey.
- **Control Group**: Pennsylvania (PA), a neighboring state that did not raise the minimum wage.
- **Outcome Variable**: Employment levels.

The analysis found that employment in New Jersey did not decrease, an important result that challenged the conventional wisdom that "minimum wage increases → employment decreases."

---

## 5. Advanced Topics: Limitations of DiD and Modern Alternatives

While the basic 2x2 DiD model is intuitive and powerful, it has limitations when applied to complex real-world data.
Recent econometrics has proposed various extension models to overcome these limitations.

### 5-1. Staggered Policy Adoption: Staggered DiD

#### (1) What's the Problem?

In reality, policies are often introduced sequentially over time (Staggered Adoption) rather than simultaneously across all regions. In the past, it was common to analyze such data using the following 2-way fixed effects (TWFE) model:

$$Y_{it} = \alpha_i + \gamma_t + \delta^{TWFE} D_{it} + \epsilon_{it}$$

- $\alpha_i$ is the unit fixed effect, $\gamma_t$ is the time fixed effect.
- $D_{it}$ is a dummy variable that equals 1 when unit $i$ receives treatment at time $t$.

Researchers long believed that $\delta^{TWFE}$ would be the average policy effect (ATT) we want. However, recent studies have proven that when policy effects are heterogeneous, this $\delta^{TWFE}$ can have serious bias.

#### (2) The Core of the "Bad Comparison" Problem: Goodman-Bacon Decomposition

Goodman-Bacon (2021) clearly decomposes and shows what 2x2 DiD estimates actually constitute the $\delta^{TWFE}$ estimator as a weighted average. According to this decomposition, $\delta^{TWFE}$ includes both of the following types of comparisons:

1. **Good Comparisons**: Clean comparisons using not-yet-treated units or never-treated units as control groups.
2. **Bad Comparisons**: Comparisons using already-treated units as control groups. For example, estimating the effect for a unit treated in 2012 while using a unit already treated in 2010 as the control group.

If policy effects change over time (dynamic treatment effects) or vary by treatment timing (heterogeneous treatment effects), these "bad comparisons" introduce unpredictable bias into $\delta^{TWFE}$, potentially even producing results with the opposite sign of the true effect.

#### (3) How to Solve It?: Decompose and Aggregate

Recent research by Callaway & Sant'Anna (2021) and Sun & Abraham (2021) proposes a new approach that fundamentally eliminates "bad comparisons." Their core idea is to decompose the data into meaningful units, estimate each effect separately, and then aggregate them according to the research purpose, rather than analyzing all data at once like TWFE.

##### ① Step 1: Estimating Group-Time Average Treatment Effects (GATT)

The most important concept is the Group-Time Average Treatment Effect (GATT).

> $$GATT(g, t) = E[Y_t(1) - Y_t(0) | G_g=1]$$

- $G_g=1$: Cohort first treated at specific time $g$.
- $Y_t(1), Y_t(0)$: Potential outcomes at time $t$ under treatment/non-treatment.
- **Meaning**: The average policy effect experienced at time $t$ by the cohort treated at time $g$.

For example, $GATT(2012, 2014)$ represents the average policy effect experienced in 2014 by cities where the policy was introduced in 2012. Callaway & Sant'Anna's methodology fundamentally blocks "bad comparisons" by using only clean control groups (not-yet-treated or never-treated units) when estimating each $GATT(g, t)$.

##### ② Step 2: Meaningful Aggregation of Effects

After obtaining numerous $GATT(g, t)$ estimates, researchers can aggregate them in various ways to answer their questions.

- **Group (Cohort)-Specific Average Effect**: Average effect experienced by a specific cohort after treatment.
    > $$ATT(g) = E_{t \ge g} [GATT(g,t)]$$
- **Event Study**: Dynamic analysis of how policy effects change according to relative time since treatment.
    > $$ATT(e) = E_{g} [GATT(g, g+e)] \quad (e \ge 0)$$
- **Overall ATT**: Weighted average across all treated groups and post-treatment periods.

This approach provides rich information that transparently and robustly shows the dynamics and heterogeneity of policy effects, instead of a single potentially biased $\delta^{TWFE}$ coefficient. With staggered adoption data, using these new methods instead of TWFE is the current standard.

### 5-2. When No Adequate Control Group Exists: Synthetic Control Method

#### What's the Problem?

It's often difficult to find a single control group that satisfies DiD's core parallel trends assumption. This is especially true when the treated unit is very large and has unique characteristics, like a state or country. For example, when asking "What was the effect when California passed a specific tobacco regulation law?" which state could serve as an appropriate 'twin' for California? Texas? Florida? Probably none.

#### Solution: Creating a "Virtual Twin"

The Synthetic Control Method (SCM) creates an "optimal virtual control group" that shows very similar movements to the treated group by taking a weighted average of multiple control groups (donor pool), rather than finding a single control group.

<div align="center">
  <img src="/assets/images/math/scm-graph.jpg" width="800" alt="Synthetic Control Method">
  <br>
  <a href="https://lost-stats.github.io/Model_Estimation/Research_Design/synthetic_control_method.html" target="_blank" rel="noopener noreferrer">
    Synthetic Control Method
  </a>
</div>

- **How It Works**: SCM finds optimal weights based on data that make the actual values of the treated group and the synthetic control group nearly perfectly match during the pre-treatment period.
- **Effect Estimation**: Maintaining these weights, we compare the actual values of the treated group with the synthetic control group values during the post-treatment period. The difference between the two paths represents the causal effect of the policy.
- **Advantages**: It prevents researchers from arbitrarily selecting control groups and transparently shows which control groups were combined with what weights, increasing confidence in the results. SCM is particularly powerful in case studies where there is only one treated unit (N=1).

#### Specific Methodology of Synthetic Control Method (SCM)

The core goal of SCM is to construct a data-driven virtual control group (synthetic control) that nearly perfectly matches the characteristics of the treated unit during the pre-treatment period. This process consists of the following clear steps:

##### ① Data Setup

First, we organize the data needed for analysis.

- **Treated Unit**: A single entity affected by the policy (i=1). (e.g., California)
- **Donor Pool**: Multiple control group candidates not affected by the policy (i=2, ..., J+1). (e.g., states other than California)
- **Time Period**: Divided into pre-treatment period ($t = 1, \dots, T_0$) and post-treatment period ($t = T_0+1, \dots, T$).

##### ② Step 2: Finding Optimal Weights (W) (The Optimization Problem)

The core of SCM is finding the optimal weight vector $W = (w_2, \dots, w_{J+1})'$ to assign to each control group in the donor pool. These weights have two important constraints:

1. **Non-negativity**: $w_j \ge 0$ for all $j \in \{2, \dots, J+1\}$
2. **Sum to one**: $\sum_{j=2}^{J+1} w_j = 1$

These constraints ensure that the synthetic control is created only as a convex combination of the donor pool, preventing arbitrary extrapolation.

Weights are chosen to minimize the distance between the treated unit's characteristics and the synthetic control's characteristics during the pre-treatment period ($t \le T_0$). Here, 'characteristics' include both past values of the outcome variable ($Y$) and other predictors ($X$) that affect the outcome.

Mathematically, we solve the following optimization problem:

> $$\min_{W} (X_1 - X_0 W)'V(X_1 - X_0 W)$$

- $X_1$: Pre-treatment characteristic vector of the treated unit.
- $X_0$: Pre-treatment characteristic matrix of all control units in the donor pool.
- $W$: The weight vector we seek to find.
- $V$: A weighting matrix representing the relative importance of each characteristic variable. Usually V is determined data-driven to minimize the mean squared prediction error (MSPE) during the pre-treatment period.

##### ③ Step 3: Constructing the Synthetic Control

Once we've found the optimal weight vector $$W^*=(w_2^*, \dots, w_{J+1}^*)'$$, we can calculate the outcome variable value of the synthetic control for all periods $(t=1, \dots, T)$:

$$Y_{synthetic, t} = \sum_{j=2}^{J+1} w_j^* Y_{j,t}$$

Thanks to the optimization process, during the pre-treatment period ($t \le T_0$), the actual treated unit's value ($Y_{1,t}$) and the synthetic control's value ($Y_{synthetic, t}$) follow nearly identical paths.

##### ④ Step 4: Estimating the Treatment Effect

The policy effect is simply calculated as the difference between the actual treated unit's value and the virtual control's value during the post-treatment period ($t > T_0$):

> $$\hat{\alpha}_t = Y_{1,t} - Y_{synthetic, t} = Y_{1,t} - \sum_{j=2}^{J+1} w_j^* Y_{j,t}$$

This difference ($\hat{\alpha}_t$) represents the policy effect, the difference between the actual path and the counterfactual path in the absence of the policy.

##### ⑤ Step 5: Statistical Inference

Determining whether the effect estimated by SCM is statistically significant, i.e., whether it occurred by chance, is somewhat more complex. Because calculating standard errors directly is difficult, we typically use a method called Placebo Test or Permutation Test:

1. **Set up Fake Treated Units**: Assume each control unit in the donor pool is a treated unit.
2. **Repeat SCM**: Repeat all steps 1-4 of SCM analysis for each of these 'fake' treated units and calculate 'fake' policy effects.
3. **Compare Effects**: Compare the policy effect calculated for the actual treated unit ($\hat{\alpha}_t$) with the distribution of numerous fake policy effects.
4. **Determine Significance**: If the actual treated unit's effect is substantially larger than most fake effects, we can conclude that this effect did not occur by chance.

Thus, SCM provides high confidence in single case studies through data-driven optimization and rigorous statistical inference processes, while maintaining an intuitive underlying idea.

---

## Summary: Key Points of DiD

| Category    | Content                                                      |
| ----------- | ------------------------------------------------------------ |
| Purpose     | Simultaneously control for time changes and group differences |
| Core Assumption | Parallel Trends Assumption                                 |
| Estimation Formula | $(Y_{T,After}-Y_{T,Before})-(Y_{C,After}-Y_{C,Before})$ |
| Representative Case | Card & Krueger (1994)                                   |
| Extension Methods | Staggered DiD, Synthetic Control Method                |

---

## References

- Card, D., & Krueger, A. (1994). *Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania*. **American Economic Review**, 84(4), 772–793.
- Goodman-Bacon, A. (2021). *Difference-in-Differences with Variation in Treatment Timing*. **Journal of Econometrics**, 225(2), 254–277.
- Callaway, B., & Sant'Anna, P. H. (2021). *Difference-in-Differences with multiple time periods*. **Journal of Econometrics**, 225(2), 200–230.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic control methods for comparative case studies*. **Journal of the American Statistical Association**, 105(490), 493–505.
- [The Great Regression — with Python: Difference-in-Differences Regressions](https://patrickthiel.com/the-great-regression-with-python-difference-in-differences-regressions/)
- [Understanding Synthetic Control and Causal Inference in A/B Testing](https://medium.com/@suraj_bansal/understanding-synthetic-control-and-causal-inference-in-a-b-testing-e10e67d570a0)
- [LOST-stats Synthetic Control Method (SCM)](https://lost-stats.github.io/Model_Estimation/Research_Design/synthetic_control_method.html)
