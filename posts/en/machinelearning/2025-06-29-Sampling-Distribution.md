---
layout: post
title: "Sampling Distribution"
date: 2025-06-29
categories: [Machine Learning, Statistics]
tags: [sampling distribution, central limit theorem, sample statistics, t-distribution, statistical inference]
math: true
---

## 1. Random Sample and Statistic

When extracting a random sample of size $n$ from population $f(x)$,
let's denote the $i$-th sample as random variable $X_i, i=1,2,3,\cdots,n$.
If we independently extracted $n$ samples under identical conditions from population $f(x)$,
random variables $X_1, X_2, \cdots,X_n$ become random samples, each having values $x_1,x_2,\cdots,x_n$.

- In the case of sampling without replacement, since each sample is not independent, random variables $X_i$ cannot be random samples.

> **Def. Random Sample**
> When $n$ random variables $X_1, X_2, \cdots,X_n$ that are independent of each other
> follow the same probability distribution $f(x)$,
> $X_1, X_2, \cdots,X_n$ are defined as a **random sample** of size $n$ from population $f(x)$.
> In this case, the joint probability distribution is
> $f(x_1,x_2,\cdots,x_n)=f(x_1)f(x_2)\cdots f(x_n)$.
>
> **Def. Statistic**
> A function of random variables composing a random sample is called a **statistic**.
>
>- When $X_1, X_2, \cdots,X_n$ is a random sample of size $n$, representative statistics are as follows:
>- Sample mean $\bar{X}=\dfrac{1}{n}\sum^n_{i=1}X_i$
>- Sample variance $S^2=\dfrac{1}{n-1}\sum^n_{i=1}(X_i-\bar{X})^2$
>- Sample standard deviation $S$

---

## 2. Sampling Distribution

> **Def. Sampling Distribution**
> The probability distribution of a statistic is called a sampling distribution.
>
>- Commonly used sampling distributions are as follows:
>- Distribution of sample mean $\bar{X}$ (sampling distribution of mean)
>- Distribution of sample variance (note: not the distribution of $S^2$, but $(n-1)S^2/\sigma^2$)
>- $t$-distribution
>- $F$-distribution

### 1) Distribution of Sample Mean $\bar{X}$ (Sampling Distribution of Mean)

- Regardless of the population distribution, for sample mean $\bar{X}$ defined by a random sample of size $n$, the mean is $\mu$ and variance is $\sigma^2/n$.

- **Central Limit Theorem (CLT)**
When extracting a random sample of size $n$ (with replacement) from a population with mean $\mu$ and variance $\sigma^2$,
if the sample mean is $\bar{X}$,
$Z=\dfrac{\bar{X}-\mu}{\sigma/\sqrt{n}}$ approaches
standard normal distribution $N(0,1)$ when $n$ is considerably large (typically $n \ge 30$).
That is, the Central Limit Theorem explains that even when the population distribution is unknown,
**if the sample size is large, the distribution of sample mean approximately follows normal distribution with mean $\mu$ and variance $\sigma^2/n$**.

  - Case of sampling without replacement
    For random variables $X_1, X_2, \cdots,X_n$ by sampling without replacement from a finite population with mean $\mu$, variance $\sigma^2$, and size $N$,
    the sample mean $\bar{X}$ has
    mean $\mu_{\bar{X}}=\mu$,
    variance $\sigma_{\bar{X}}^2=\dfrac{N-n}{N-1}\dfrac{\sigma^2}{n}$.
    (Finite population correction factor: $\dfrac{N-n}{N-1}$)
    If sample size $n$ is sufficiently large ($n \ge 30$),
    and population size $N$ is sufficiently large compared to $n$ ($n/N \le 0.05$), $\dfrac{N-n}{N-1}$ is close to 1, so
    the distribution of sample mean $\bar{X}$ of random variables $X_1, X_2, \cdots,X_n$ by sampling without replacement from finite population is also approximately
    treated as normal distribution with mean $\mu$ and variance $\sigma^2/n$.

  - Sample mean in binomial distribution (special case of Central Limit Theorem)
    For sample proportion $\hat{P}=X/n$ (where $X$ is the random variable representing number of successes) defined by a random sample of size $n$ from binomial population with population proportion $p$,
    as sample size $n$ increases, the distribution approaches normal distribution with mean $p$ and variance $\dfrac{p(1-p)}{n}$.
    That is, the distribution of random variable $Z=\dfrac{\hat{P}-p}{\sqrt{p(1-p)/n}}$ approaches standard normal distribution as sample size $n$ increases.
    (Typically known to hold when $np\ge5$ and $n(1-p)\ge5$.)

  - Distribution of difference between two sample means (application)
    When two populations are independent of each other and have means $\mu_1,\mu_2$ and variances $\sigma_1^2,\sigma_2^2$ respectively,
    the distribution of difference between sample means $\bar{X_1}-\bar{X_2}$ of two samples of sizes $n_1,n_2$ extracted from each population
    approximately follows normal distribution with mean $\mu_{\bar{X_1}-\bar{X_2}}=\mu_1-\mu_2$ and
    variance $\sigma_{\bar{X_1}-\bar{X_2}}^2=\dfrac{\sigma_1^2}{n_1} +\dfrac{\sigma_2^2}{n_2}$.
    Therefore, $Z=\dfrac{(\bar{X_1}-\bar{X_2})-(\mu_1-\mu_2)}{\sqrt{(\sigma_1^2/n_1)+(\sigma_2^2/n_2)}}$ approximately follows standard normal distribution.

### 2) Distribution of Sample Variance

- When extracting a random sample of size $n$ from normal population with variance $\sigma^2$, if sample variance is $S^2$,
  statistic $\chi^2=\dfrac{(n-1)S^2}{\sigma^2}=\sum^n_{i=1}\dfrac{(X_i-\bar{X})^2}{\sigma^2}$
  follows chi-squared distribution with degrees of freedom $v=n-1$.

### 3) $t$-distribution

The Central Limit Theorem uses $Z=\dfrac{\bar{X}-\mu}{\sqrt{\sigma^2/n}}$,
$Z$ can be used to estimate population mean or differences between population means, but the population variance ($\sigma^2$) must be known to estimate population mean or differences between means.
However, cases where population variance is known are extremely rare. Therefore, we utilize $t$-distribution using sample variance instead of population variance.

- When $X_1, X_2, \cdots,X_n$ are all independent random variables following normal distribution with mean $\mu$ and standard deviation $\sigma$,
  random variable $T=\dfrac{\bar{X}-\mu}{\sqrt{S^2/n}}$ follows $t$-distribution with degrees of freedom $v=n-1$.

### 4) $F$-distribution

A distribution widely used for comparing variances between samples. Utilized in statistical inference about the ratio of population variances of two normal populations and analysis of variance.
① Statistic $F$ is the ratio $F=\dfrac{U/v_1}{V/v_2}$ of two independent random variables $U,V$ following chi-squared distributions, each divided by their respective degrees of freedom $v_1,v_2$.
② Let's say we extracted samples of sizes $n_1,n_2$ from normal populations where population variances are known as $\sigma_1^2,\sigma_2^2$ respectively.
Then, as we saw in the distribution of sample variance above,
$\chi_1^2=\dfrac{(n_1-1)S_1^2}{\sigma_1^2}$ follows chi-squared distribution with degrees of freedom $v_1=n_1-1$,
$\chi_2^2=\dfrac{(n_2-1)S_2^2}{\sigma_2^2}$ follows chi-squared distribution with degrees of freedom $v_2=n_2-1$. Therefore, the following theorem holds.

- When samples of sizes $n_1,n_2$ are extracted independently from normal populations with population variances $\sigma_1^2,\sigma_2^2$ respectively,
  if sample variances are $S_1^2,S_2^2$,
  $F=\dfrac{\chi_1^2}{\chi_2^2}=\dfrac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}=\dfrac{\sigma_2^2S_1^2}{\sigma_1^2S_2^2}$
  follows $F$-distribution with degrees of freedom $v_1=n_1-1$ and $v_2=n_2-1$.

---

## Advanced Topics

- Sample moments
- Order statistics
- Approximation of sampling distributions

---
_Source: Lecture Notes from undergraduate and graduate courses - Mathematical Statistics, Statistical Methodology, Statistical Inference_
