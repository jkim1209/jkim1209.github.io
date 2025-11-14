---
layout: post
title: "Continuous Probability and Measure Theory: Why Is the Probability of a Single Point Zero?"
date: 2025-10-10
categories: [Math, Probability]
tags: [measure theory, probability, discrete probability, continuous probability, sigma algebra, sigma-field, lebesgue-integral, statistics]
math: true
---

## 1. Discrete Probability and Continuous Probability

In probability theory, we encounter two different worlds. The world of discrete probability, composed of clearly countable outcomes like rolling a die, and the world of continuous probability, composed of infinitely many outcome values like height or weight.

The world of discrete probability is intuitive. The probability of getting any single number when rolling a die is $1/6$, and adding up the probabilities of all numbers equals 1.

$$
\sum_{i=1}^{6}P(X=i) = P(X=1)+P(X=2)+P(X=3)+P(X=4)+P(X=5)+P(X=6) = 1
$$

But when we move to the world of continuous probability, the story changes. Now it becomes perplexing how to define the probability of "a single point".

For example, let's consider the probability that someone's height is 'exactly 175.125942...cm'. Intuitively, it shouldn't be 0, and we might think it's a very small positive number.
However, if the probability is even slightly greater than 0 for all real number points, when we add up the probabilities of all those infinitely many points, the total sum becomes infinite.

$$
\text{If } P(X=x)=\varepsilon>0 \text{ for all } x \in [a,b] \;\;\Rightarrow \;
\sum_{x\in[a,b]} P(X=x)=\infty
$$

This is clearly a contradiction. Therefore, the probability at a single point must be 0.

But here a new question arises.

"If infinitely many points with probability 0 are gathered together, how can the total probability be 1?"

$$
\text{If } P(X=x)=0 \text{ for all } x \in [a,b] \;\;\; \Rightarrow \;\;
\sum_{x \in [a,b]} P(X=x) = 1 \;?
$$

The mathematical framework introduced to resolve this paradox is precisely **Measure Theory**. Measure theory meticulously addresses this problem by assigning **size (Measure)** not to 'individual points', but to 'intervals' or 'entire sets'.

## 2. On the Method of Measuring 'Size (Measure)'

Before the full discussion, the most effective analogy is 'length'. Imagine a line segment with length from 0 to 1, that is, the interval `[0, 1]`.

* Total length of the line segment: The entire length of this line segment is `1`. This is equivalent to the total probability of a continuous probability distribution being 1.
* Length of a single point on the line segment: Now pick a single point `0.5` on it. What is the 'length' of this point itself? Mathematically, a point has no length, so its length is 0.

Right at this point, our intuition collides with mathematical reality. "How can the total length be 1 when countless points with length 0 are gathered?"

This question is essentially the same as "How can the total probability be 1 when infinitely many points with probability 0 are gathered?"

Measure theory requires a shift in perspective to solve this problem. Instead of focusing on individual 'points', it presents a general method of measuring the size of 'intervals' or 'sets'.

## 3. The World of Measure Theory: A Solid Foundation for Probability

Measure theory is a discipline for rigorously handling the concept of 'size' mathematically, encompassing 'length', 'area', 'volume', and even 'probability'. Probability theory is built on this solid foundation called measure theory.

### 3-1. σ-Field (Sigma-Field): Defining Measurable Events

Although it would be nice if we could measure the 'size' of all sets, mathematically very strange sets (example: [Vitali set](https://namu.wiki/w/%EB%B9%84%ED%83%88%EB%A6%AC%20%EC%A7%91%ED%95%A9)) exist that can cause contradictions. Therefore, we agree to collect 'well-defined measurable sets' into one set, which is precisely the **σ-Field (Sigma-Field, or σ-Algebra)** $\mathcal{F}$. Understanding the concept of sigma-field as a set of sets makes it easier.

For a collection of sets $\mathcal{F}$ to be a sigma-field, it must follow three rules (axioms) below.

> 1. Contains the entire set (Ω). $(\Omega \in \mathcal{F})$
>    * The most certain event, that is, the event that 'something happens', must be included in the list.
> 2. If an event A is included, its complement ($A^c$) is also included. $(A \in \mathcal{F} \implies A^c \in \mathcal{F})$
>    * If we can handle the event 'A happens', we should naturally be able to handle the event 'A doesn't happen' as well. This is a logically natural agreement.
> 3. The union of a countable number of events is also included. $(A_1, A_2, \dots \in \mathcal{F} \implies \bigcup_{i=1}^{\infty} A_i \in \mathcal{F})$
>     * This means that the event 'at least one of them happens' for several events in the list should also be in the list.

Let's consider a simple case of flipping a coin once. The set of all outcomes (sample space) is Ω = {H, T}.
What 'events' might we be interested in here?

* ∅ (nothing happens - empty set)
* {H} (event that heads comes up)
* {T} (event that tails comes up)
* {H, T} (event that heads or tails comes up - entire set)

Let's consider the set $\mathcal{F} = \{\emptyset,\{H\},\{T\},\{H,T\}\}$ that collects all these events. Does this set $\mathcal{F}$ satisfy the three conditions of a sigma-field?

1. It contains the entire set $\Omega = \{H, T\}$.
2. The complement of every element belonging to F also belongs to F.
   * $\emptyset^c = \Omega \in \mathcal{F}$
   * $\{H\}^c = \{T\} \in \mathcal{F}$
   * $\{T\}^c = \{H\} \in \mathcal{F}$
   * $\{H, T\}^c = \emptyset \in \mathcal{F}$
3. The union of elements belonging to F also belongs to F. (e.g., $\{H\} \cup \{T\} = \{H, T\} \in \mathcal{F}$)

Since all conditions are satisfied, F is a sigma-field.

In conclusion, a sigma-field is **a reliable list (set) that collects events (sets) qualified to be assigned a value called 'probability'**. (The rigorous definition is 'a system of events on which a probability measure can be defined consistently', but since we focus on understanding here, let's continue to explain it in simple terms.)

### 3-2. Measure: A Function that Assigns 'Size'

Once the sigma-field is prepared, we need a function that assigns 'size' to each event (set), which is the **Measure** $\mu$.

> 1. Non-negativity: The size of every set is 0 or greater. $(\mu(A) \ge 0)$
> 2. Empty set has measure 0: The size of an empty set is 0. $(\mu(\emptyset) = 0)$
> 3. Countable Additivity: The size of the union of non-overlapping sets equals the sum of the sizes of each set.
>    * For disjoint $A_1, A_2, \dots$, $\mu\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \mu(A_i)$

These three properties mathematically express what we intuitively expect from the concept of 'size'.

### 3-3. Probability Space: The Stage of Probability Theory

Synthesizing these concepts, the **Probability Space**, the stage of probability theory, is defined. This is expressed as a combination of three elements $(\Omega, \mathcal{F}, \mathcal{P})$.

* $\Omega$ (Sample Space): The set of all possible outcomes that can occur in an experiment.
* $\mathcal{F}$ (σ-Field): A set of measurable events consisting of subsets of $\Omega$.
* $\mathcal{P}$ (Probability Measure): A special measure that assigns a number between 0 and 1 to each element (event) of the sigma-field $\mathcal{F}$. This measure satisfies the key condition that the size of the entire space is 1: $P(\Omega) = 1$.

This is like setting the total length of a ruler to 1 meter and measuring other lengths within it. Thus probability has a value between 0 and 1, consistent with the definition of probability we know.

## 4. Resolution of the Paradox: Probability from a New Perspective

Now let's return to the original problem.

### 4-1. Probability of a Single Point: Why Is It 0?

Asking about $P(X=c)$ for a continuous random variable $X$ means, in measure-theoretic terms, calculating the probability measure of the singleton set `{c}`. This set can be viewed as the closed interval `[c, c]`, and the 'length (Lebesgue measure)' of this interval is $c-c=0$. Therefore, the probability measure value assigned to this set is also 0. This is exactly the same logic as a single point on a line segment having length 0.

> **Note.**
> In continuous probability, $P(X=c)=0$ doesn't mean 'that event is impossible', but simply that **the measure of that set is 0**.
>
> For example, in a mixture distribution (continuous + point mass distribution), positive probability mass can actually exist at a specific point. $P(X=0)>0$ is such an example.
> Therefore, $P(X=c)=0$ doesn't mean that value **absolutely never occurs**.

### 4-2. Probability Density Function (PDF): Where Does Probability Exist?

Then where does probability exist? It is distributed not at a specific point but in an interval. The concept that emerges here is the Probability Density Function (PDF) $f(x)$.

<div align="center">
  <img src="/assets/images/math/measure_theory_normal_distribution.jpg" width="800" alt="measure_theory_normal_distribution">
  <br>
    Standard normal distribution (Gauss distribution)
</div>

Understanding PDF is easy when compared to population density.

* No person lives on a single 'point' in Seoul (since people occupy volume).
* However, the population of 'Seoul' as a 'region (set)' is much larger than the population of the same 'area (set)' in the Gangwon mountain region.
* Population density indicates not the population of a specific point, but how densely people are gathered around that point.

Similarly, the value of PDF $f(x)$ is not a probability itself, but a 'density' indicating how densely probability is distributed around point $x$. Where $f(x)$ values are high is like a popular area where probability is concentrated.

### 4-3. Integration: Gathering Density to Create Probability

We gather this 'density' to calculate actual 'probability (size)'. This process is precisely integration (Integral).

$$
P(a \le X \le b) = \int_a^b f(x) dx
$$

The above formula means the process of sweeping up all probability density distributed over the interval `[a, b]` to calculate its total amount (probability measure).

> **Note.**
> Not all random variables have a probability density function (PDF).
> A representative example is the [Cantor distribution](https://ko.wikipedia.org/wiki/%EC%B9%B8%ED%86%A0%EC%96%B4_%ED%95%A8%EC%88%98), which is continuous but has no PDF.
>
> For this reason, in actual probability calculations, the cumulative distribution function (CDF) $F(x)$ is used more generally than the probability density function (PDF).
>
> $$
> P(a < X \le b) = F(b) - F(a)
> $$

### 4-4. Lebesgue Integral

In fact, the integration used above implies the concept of the more powerful Lebesgue Integral based on measure theory, beyond the Riemann Integral learned in high school or calculus.

The difference between the two integrals can be compared to counting money.

* Riemann Integral: A method of counting money in a wallet one bill at a time in the order it was received. That is, divide the domain (x-axis) into small pieces, multiply the height of each interval, and add them up.
* Lebesgue Integral: A method of first sorting all money in the wallet into 1000 won bills, 5000 won bills, and 10000 won bills, then counting how many bills of each denomination there are and summing them up. That is, based on the codomain (y-axis, function value), find the 'size (measure)' of the domain with the same value, multiply, then add them all up.

The Lebesgue integral can integrate more general functions and is perfectly compatible with measure theory. The general form of the Lebesgue integral is as follows.

$$
\int_A f\,d\mu
$$

This means integrating function $f$ over set $A$ with respect to measure $\mu$.
The probability integral $\int_a^b f(x)\,dx$ we commonly use can be seen as a special case of this general formula.

* Set $A$ is the interval `[a, b]`
* Function $f$ is the probability density function (PDF)
* Measure $\mu$ is the Lebesgue measure $\lambda$ that measures 'length', conventionally written as $dx$.

  * That is, in the Lebesgue integral, writing as follows on the right side is a more rigorous expression in measure-theoretic terms.
    $$
    \int_a^b f(x)\,dx = \int_{[a,b]} f(x)\,d\lambda(x)
    $$
  * Here $\lambda$ is the Lebesgue measure, a measure that measures length. For example, $\lambda([0,1]) = 1$, $\lambda([a,b]) = b - a$.
  * In the above formula, the integration region `[a,b]` is the probability interval we're interested in, $\lambda([a,b])$ is the length of that interval, and $f(x)$ indicates how densely probability is concentrated according to that length (probability density).
  * Therefore, $dx$ is essentially a conventional notation meaning "integrate with respect to Lebesgue measure $\lambda$".

Thus, the Lebesgue integral serves as the mathematical engine that calculates probability measure from the probability density function.

## 5. Conclusion: Resolving the Paradox Through a Shift in Perspective

In conclusion, the paradox of continuous probability is clearly resolved through measure theory as follows.

1. Shift of Object: Probability is a measure defined not for individual 'points' but for 'sets', that is, events belonging to the sigma-field.
2. Probability of a Point: In continuous space, a single point is a set with 'length 0', so its probability measure is 0.
3. Probability of an Interval: Meaningful positive probability appears in sets (intervals) with 'non-zero length', calculated by Lebesgue integration of the probability density function (PDF) over that interval.

In continuous probability, the proposition "the probability of a single point is 0" is no longer a paradox, but rather an inevitable result when viewed from the measure-theoretic perspective of probability as 'the size of sets'. Through this great shift in perspective, numerous theories in modern mathematics, statistics, and machine learning have finally obtained a solid logical foundation.

Intuition is a powerful starting point for understanding phenomena, but sometimes it also leads us into the trap of paradoxes. Mathematical tools like measure theory elevate our intuition to a higher level at precisely that point, allowing us to clearly see the structure of a world that was previously invisible.

---

_**Source: My own Lecture Notes from Graduate School - Probability, Statistical Inference**_
