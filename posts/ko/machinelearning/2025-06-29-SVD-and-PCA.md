---
layout: post
title: "SVD와 PCA"
date: 2025-06-29
categories: [Machine Learning, Statistics]
tags: [singular value decomposition, principal component analysis, orthogonal matrix, gram schmidt, pseudo inverse, truncated svd, covariance matrix]
math: true
---

## 1. 고유값과 고유벡터

### (1) 정의

$n\times n$행렬 $A$에 대하여 $A\vec{v}=\lambda \vec{v}$를 만족하는 $\lambda$를 고유값, $\vec{v}\neq\vec{0}$를 고유벡터라 한다.

### (2) 도출

$1.$ $Det(A-\lambda I)=0$을 만족하는 고유값 $\lambda$를 구한다.

$2.$ 구한 고유값을 $(A-\lambda I)\vec{v}=0$에 대입하여 고유벡터 $\vec{v}$를 구한다.

$3.$ $n\times n$행렬 $A$의 고유값 $\lambda$와 고유벡터 $\vec{v}$에 대하여 $E_\lambda=Span(\vec{v})$를 고유공간이라 한다.

### (3) 정리

$1.$ 삼각행렬의 고유값은 대각성분이다.

$2.$ $n\times n$행렬의 서로 다른 고유값을 $\lambda_1, \lambda_2, \cdots, \lambda_k$이라 하고 각각의 고유값에 해당하는 고유벡터를 $\vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k$ 라고 하면 $\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$는 일차독립이다.

---

## 2. Gram-Shmidt 직교화

### 1) 두 벡터의 직교

#### (1) 정의

내적공간 $V$의 원소 $\vec{u},\vec{v}$에 대하여 $\vec{u}$와 $\vec{v}$가 직교한다는 것은 내적이 $0$임을 의미한다.  
즉, $\vec{u}\perp \vec{v} \Leftrightarrow \ <\vec{u},\vec{v}>=0.$

### 2) 벡터의 크기(Norm)

#### (1) 정의

내적공간 $V$의 원소 $\vec{u}$에 대하여 벡터 $\vec{u}$의 크기는 $\|\vec{u}\|=\sqrt{<\vec{u},\vec{u}>}$ 로 정의한다.

#### (2) 정리

$1.\ \|\vec{u}+\vec{v}\|^2=\|\vec{u}\|^2+\|\vec{v}\|^2+2<\vec{u},\vec{v}>$

$2.\ \|\vec{u}+\vec{v}\|^2+\|\vec{u}-\vec{v}\|^2=2\|\vec{u}\|^2+2\|\vec{v}\|^2$

$3.\ <\vec{u},\vec{v}>=\dfrac{1}{4}\|\vec{u}+\vec{v}\|^2-\dfrac{1}{4}\|\vec{u}-\vec{v}\|^2$

$4.\ \vec{u}, \vec{v}$ 가 직교할 필요충분조건은 $\|\vec{u}+\vec{v}\|^2=\|\vec{u}\|^2+\|\vec{v}\|^2$이다.

$5.\ \|<\vec{u}, \vec{v}>\| \le \|\vec{u}\| \|\vec{v}\|$ (단, 등호는 $\vec{u}$와 $\vec{v}$가 평행한 경우에 성립한다.)

$6.\ \|\vec{u}+\vec{v}\| \le \|\vec{u}\| + \|\vec{v}\|$

> - $\mathbb{R}^n$ 에서 정의된 닷곱 $\vec{u}\cdot \vec{v}=\vec{u}^T\vec{v}$ 도 내적(Inner Product)이다.
> - 따라서 $\mathbb{R}^n$ 에서 정의된 닷곱에 대해서도 직교성을 논의할 수 있다.  
> - 이하의 내용은 일반적인 내적공간에 대해서도 성립하나 $\mathbb{R}^n$ 에서 정의된 닷곱에 한정하여 서술한다.

### 3) 직교집합(Orthogonal Set)

#### (1) 정의

$\vec{v}_1,\ \vec{v}_2,\ \cdots,\ \vec{v}_k \in \mathbb{R}^n$ 에서 정의된 닷곱에 대하여
$\{ \vec{v}_1,\ \vec{v}_2,\ \cdots,\ \vec{v}_k \}$가 직교집합이라는 것은 $\vec{v}_i \cdot \vec{v}_j=0, \quad i\neq j$ 임을 뜻한다.

#### (2) 정리

$1.$ 직교집합 $B=\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$ 에 의해 생성된 $\mathbb{R}^n$의 부분공간 $V$ (즉, $V=Span(B) \subseteq \mathbb{R}^n$)의 임의의 벡터 $\vec{u} \in V$ 에 대하여,

$\quad \vec{u}=\dfrac{\vec{u} \cdot \vec{v}_1}{\vec{v}_1 \cdot \vec{v}_1}\vec{v}_1   + \dfrac{\vec{u} \cdot \vec{v}_2}{\vec{v}_2 \cdot \vec{v}_2}\vec{v}_2  +  \cdots  + \dfrac{\vec{u} \cdot \vec{v}_k}{\vec{v}_k \cdot \vec{v}_k}\vec{v}_k$ 로 나타낼 수 있다.

$2.$ 직교집합 $\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$ 이 영벡터를 포함하지 않으면 $\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$는 일차독립이다.

### 4) 직교기저(Orthogonal Basis)

#### (1) 정의

$\mathbb{R}^n$의 부분공간 $V$에 대하여 $B=\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$ 가 $V$의 ①기저이며 ②직교집합이면 $B$는 $V$의 직교기저이다.

### 5) 정규직교집합(Orthonormal Set)

#### (1) 정의

$\{ \vec{v}_1,\ \vec{v}_2,\ \cdots,\ \vec{v}_k \}$가 정규직교집합이라는 것은
① $<\vec{v}_i, \vec{v}_j>=0, \quad i\neq j$
② $\|vec{v}_i\|=1, \quad \forall{i}$
임을 뜻한다.

> 참고: $\mathbb{R}^n$의 표준기저는 정규직교집합이다.

### 6) 정규직교기저(Orthonormal Basis)

#### (1) 정의

$\mathbb{R}^n$의 부분공간 $V$에 대하여 $B=\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$ 가 $V$의 ①기저이며 ②정규직교집합이면 $B$는 $W$의 직교기저이다.

#### (2) 정리

$1.$ 정규직교집합 $B=\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$ 에 의해 생성된 $\mathbb{R}^n$의 부분공간 $V$의 임의의 벡터 $\vec{u} \in V$ 에 대하여
$\quad \vec{u} \ = \ (\vec{u} ·t \vec{v}_1)\vec{v}_1 \ + \ (\vec{u} \cdot \vec{v}_2)\vec{v}_2 \ + \ \cdots \ + \ (\vec{u} \cdot \vec{v}_k)\vec{v}_k$ 로 나타낼 수 있다.

$2.$ $m\times n \ (m\ge n)$ 행렬 $A$의 열벡터들의 정규직교이면 $A^TA=I_n$이다.

### 7) 직교행렬(Orthogonal Matrix)

#### (1) 정의

$A$가 직교행렬이라는 것은 ① $A: n \times n$ 행렬, ② $A$의 열벡터들이 **정규직교** 집합일 때이다.

#### (2) 정리

$1.$ 정사각행렬 $A$가 직교행렬일 필요충분조건은 $A^TA=I$ 즉, $A^T=A^{-1}$이다.  
$2.$ $A^T=A^{-1}$ 으로부터 $AA^T=I$ 이기도 한데, 이는 정사각행렬 $A$가 직교행렬이라면 $A$의 행벡터들의 집합도 정규직교집합임을 의미한다.  

### 8) 직교사영(Orthogonal Projection)

#### (1) 도출

##### [1] $\mathbb{R}^2$에서의 직교사영

일차독립인 $\vec{u},\vec{v} \in \mathbb{R}^2$ 에 대하여 (즉, $\vec{u},\vec{v}$는 $\mathbb{R}^2$의 기저)

① $\vec{u_{pr}}=\dfrac{\vec{u} \cdot \vec{v}}{\vec{v} \cdot \vec{v}}\vec{v}\qquad$: $\vec{v}$ 위로 $\vec{u}$ 의 직교사영 (벡터사영)

② $\vec{u_c}=\vec{u}-\vec{u_{pr}} \qquad$ : $\vec{v}$ 에 직교하는 $\vec{u}$의 벡터성분

(i) $\vec{u_{pr}}+\vec{u_c}=\vec{u} \qquad$(ii) $\vec{u_{pr}} \parallel \vec{v} \qquad$ (iii) $\vec{u_c} \perp \vec{v}$

<center><img src='/assets/images/machinelearning/SVD_orthogonal_projection_r2.jpg' width = 600 alt="SVD_orthogonal_projection_r2"></center>

##### [2] $\mathbb{R}^3$에서의 직교사영

일차독립인 $\vec{u},\vec{v}_1,\vec{v}_2  \in \mathbb{R}^3$, 직교하는 벡터 $\vec{v}_1,\vec{v}_2$ 에 대하여

① $\vec{u_{1}}=\dfrac{\vec{u} \cdot \vec{v}_1}{\vec{v}_1 \cdot \vec{v}_1}\vec{v}_1\qquad$: $\vec{v}_1$ 위로 $\vec{u}$ 의 직교사영

② $\vec{u_{2}}=\dfrac{\vec{u} \cdot \vec{v}_2}{\vec{v}_2 \cdot \vec{v}_2}\vec{v}_2\qquad$: $\vec{v}_2$ 위로 $\vec{u}$ 의 직교사영

③ $\vec{u_{pr}}= \vec{u_{1}} + \vec{u_{2}} = \dfrac{\vec{u} \cdot \vec{v}_1}{\vec{v}_1 \cdot \vec{v}_1}\vec{v}_1 + \dfrac{\vec{u} \cdot \vec{v}_2}{\vec{v}_2 \cdot \vec{v}_2}\vec{v}_2 \qquad$ : $Span(\vec{v}_1,\vec{v}_2)$ 위로 $\vec{u}$ 의 직교사영

④ $\vec{u_c}=\vec{u}-\vec{u_{pr}}=\vec{u}-\dfrac{\vec{u} \cdot \vec{v}_1}{\vec{v}_1 \cdot \vec{v}_1}\vec{v}_1 - \dfrac{\vec{u} \cdot \vec{v}_2}{\vec{v}_2 \cdot \vec{v}_2}\vec{v}_2 \qquad$ : $Span(\vec{v}_1,\vec{v}_2)$ 에 직교하는 $\vec{u}$ 의 벡터성분

<center><img src='/assets/images/machinelearning/SVD_orthogonal_projection_r3.jpg' width = 600 alt="SVD_orthogonal_projection_r3"></center>

##### [3] $\mathbb{R}^n$에서의 직교사영

$\mathbb{R}^n$의 부분공간 $V$에 대한 직교기저 $B=\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$ 가 주어져 있을 때, $V$의 임의의 벡터 $\vec{u}$ 에 대하여  

① $\vec{u_{pr}}=\dfrac{\vec{u} \cdot \vec{v}_1}{\vec{v}_1 \cdot \vec{v}_1}\vec{v}_1   + \dfrac{\vec{u} \cdot \vec{v}_2}{\vec{v}_2 \cdot \vec{v}_2}\vec{v}_2   +   \cdots   + \dfrac{\vec{u} \cdot \vec{v}_k}{\vec{v}_k \cdot \vec{v}_k}\vec{v}_k \qquad$ : $V$위로의 $\vec{u}$의 직교사영

② $\vec{u_c}=\vec{u}-\vec{u_{pr}}=\vec{u}-\dfrac{\vec{u} \cdot \vec{v}_1}{\vec{v}_1 \cdot \vec{v}_1}\vec{v}_1   - \dfrac{\vec{u} \cdot \vec{v}_2}{\vec{v}_2 \cdot \vec{v}_2}\vec{v}_2   -   \cdots   - \dfrac{\vec{u} \cdot \vec{v}_k}{\vec{v}_k \cdot \vec{v}_k}\vec{v}_k \qquad$ : $V$에 직교하는 $\vec{u}$의 벡터성분

#### (2) 정리: Gram-Schmidt 직교화과정

$\mathbb{R}^n$의 부분공간 $V$에 대하여 $B=\{ \vec{v}_1, \vec{v}_2, \cdots, \vec{v}_k \}$ 가 $V$의 기저일 때, 직교기저 $C=\{ \vec{u}_1, \vec{u}_2, \cdots, \vec{u}_k \}$ 와  
정규직교기저 $D=\{ \vec{w}_1, \vec{w}_2, \cdots, \vec{w}_k \}$도 존재한다. 여기서 $C$와 $D$는 다음과 같이 구할 수 있다.  

$$
\vec{u}_1=\vec{v}_1
$$

$$
\vec{u}_2=\vec{v}_2-\dfrac{\vec{v}_2 \cdot \vec{u}_1}{\vec{u}_1 \cdot \vec{u}_1}\vec{u}_1
$$

$$
\vec{u}_3=\vec{v}_3-\dfrac{\vec{v}_3 \cdot \vec{v}_1}{\vec{u}_1 \cdot \vec{u}_1}\vec{u}_1-\dfrac{\vec{v}_3 \cdot \vec{u}_2}{\vec{u}_2 \cdot \vec{u}_2}\vec{u}_2
$$

$$
\cdots
$$

$$
\vec{u}_k=\vec{v}_k-\dfrac{\vec{v}_k \cdot \vec{u}_1}{\vec{u}_1 \cdot \vec{u}_1}\vec{u}_1-\dfrac{\vec{v}_k \cdot \vec{u}_2}{\vec{u}_2 \cdot \vec{u}_2}\vec{u}_2 - \cdots -\dfrac{\vec{v}_k \cdot \vec{u}_{k-1}}{\vec{u}_{k-1} \cdot \vec{u}_{k-1}}\vec{u}_{k-1}
$$

$$
\vec{w}_1=\dfrac{\vec{u}}{\|u_1\|}, \quad \vec{w}_2=\dfrac{\vec{u}}{\|u_2\|}, \quad \cdots, \quad \vec{w}_k=\dfrac{\vec{u}}{\|u_k\|}
$$

> ① $\vec{u}_1=\vec{v}_1\qquad$이제 $\vec{u}_1$를 만들었으니 $\vec{v}_1$는 사용하지 않는다.  
> ② $\vec{u}_1$ 와 직교하는 벡터 $\vec{u}_2$ 를 찾고 싶은 것이므로  
>
> $$
> \quad \vec{u}_2=\vec{v}_{2c}=\vec{v}_2-\vec{v}_{2pr}=\vec{v}_2-\dfrac{\vec{v}_2 \cdot \vec{u}_1}{\vec{u}_1 \cdot \vec{u}_1}\vec{u}_1
> $$
>
> <center><img src='/assets/images/machinelearning/SVD_gram_schmidt_r2.jpg' width = 600 alt="SVD_gram_schmidt_r2"></center>
>
>
> $\quad$ 이제 $\vec{u}_2$를 만들었으니 $\vec{v}_2$는 사용하지 않는다.
>
>
> ③ 위에서 $\vec{u}_1$과 $\vec{u}_2$는 직교하게 만들었으므로, $Span(\vec{u}_1,\vec{u}_2)$과 직교하는 $\vec{u}_3$을 찾으려면  
>
>
> $$
> \quad \vec{u}_2=\vec{v}_{3c}=\vec{v}_3-\vec{v}_{3pr}=\vec{v}_3-\dfrac{\vec{v}_3 \cdot \vec{u}_1}{\vec{u}_1 \cdot \vec{u}_1}\vec{u}_1-\dfrac{\vec{v}_3 \cdot \vec{u}_2}{\vec{u}_2 \cdot \vec{u}_2}\vec{u}_2
> $$
>
> <center><img src='/assets/images/machinelearning/SVD_gram_schmidt_r3.jpg' width = 600 alt="SVD_gram_schmidt_r3"></center>
>
>
> $\quad$ 이제 $\vec{u}_3$를 만들었으니 $\vec{v}_3$는 사용하지 않는다.
>
>
> ④ 이를 반복하면,
>
> $$
> \quad \vec{u}_k=\vec{v}_k-\dfrac{\vec{v}_k \cdot \vec{u}_1}{\vec{u}_1 \cdot \vec{u}_1}\vec{u}_1-\dfrac{\vec{v}_k \cdot \vec{u}_2}{\vec{u}_2 \cdot \vec{u}_2}\vec{u}_2 -   \cdots   -\dfrac{\vec{v}_k \cdot ?> \vec{u}_{k-1}}{\vec{u}_{k-1} \cdot \vec{u}_{k-1}}\vec{u}_{k-1}
> $$

---

## 3. 특이값 분해(SVD; Singular Value Decomposition)

#### (1) 정의

$m\times n$행렬 $A$에 대하여 $A=U\Sigma V^T$ 의 형태로 표현할 수 있다. 여기서
$V: n\times n$ 직교행렬, $\quad \Sigma: m\times n$ 행렬이다. 특히

$$
\Sigma=\begin{bmatrix} D & 0 \\ 0 & 0 \end{bmatrix}  
, D=\begin{bmatrix} \sigma_1 & 0 & \cdots & 0 \\ 0 & \sigma_2 &\cdots & 0 \\ \cdots & \cdots & \cdots &\cdots \\ 0 & 0 & \cdots & \sigma_r\end{bmatrix}
$$  

> <center><img src='/assets/images/machinelearning/SVD_SVD.webp' width = 900 alt="SVD_SVD"></center>
>
><div align="center">
>  <a href="https://towardsdatascience.com/singular-value-decomposition-svd-demystified-57fc44b802a0/">Singular Value Decomposition (SVD), Demystified</a>
>  <br>
></div>

#### (2) 도출

① 대칭행렬 $A^TA$를 구한다. (직교대각화 가능)

② $A^TA$의 고유값 $\lambda_1,\lambda_2,\cdots,\lambda_n$을 구한다.

③ 고유값 $\lambda_1,\lambda_2,\cdots,\lambda_n$에 해당하는 고유벡터를 구한 뒤, 이를 **정규직교** 고유벡터 $\vec{v}_1,\vec{v}_2,\cdots,\vec{v}_n$로 바꿔준다.

- 이 때 서로 다른 고유값으로부터 나온 고유벡터들은 서로 직교성이 보장되어 있지만 중근이 존재하여 하나의 고유값으로부터 나온 여러개의 고유벡터가 있다면 그들은 서로 직교성이 보장되어 있지 않으므로 Gram-Shmidt 직교화과정을 통해 직교성을 갖게끔 바꿔주어야 한다.

④ 특이값 $\quad \sigma_1=\sqrt{\lambda_1} \quad \ge \quad \sigma_2=\sqrt{\lambda_2} \quad \ge \quad \cdots \quad \ge \quad  \sigma_n=\sqrt{\lambda_n} \quad \ge \quad 0$ 을 구한다.
$\quad \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0, \quad \sigma_{r+1}=\sigma_{r+2}=\cdots=\sigma_n=0$ 이라 하자.

⑤ $\vec{u}_i=\dfrac{1}{\sigma_i}A\vec{v}_i \quad (i=1,2,\cdots,r)$ 을 구한다.

⑥ 정규직교집합 $\{\vec{u}_1,\vec{u}_2,\cdots,\vec{u}_r\} \subset{\mathbb{R}^n}$ 으로부터 $\mathbb{R}^m$ 의 정규직교기저

$$
\{\vec{u}_1, \vec{u}_2, \cdots, \vec{u}_r,\vec{u}_{r+1}, \cdots, \vec{u}_m\}
$$

을 구한다.

- $\mathbb{R}^n$의 표준기저들 중 정규직교집합 $\{\vec{u}_1,\vec{u}_2,\cdots,\vec{u}_r\}$과 일차독립인 표준기저들을 찾아 Gram-shmidt 직교화과정을 통해 구할 수 있다.

⑦ $U=[\vec{u}_1 \quad \vec{u}_2 \quad \cdots \quad \vec{u}_m], \qquad V=[\vec{v}_1 \quad \vec{v}_2 \quad \cdots \quad \vec{v}_n]$

#### (3) 정리

$m\times n$행렬 $A$에 대하여 $V, \Sigma, U$ 를 특이값분해행렬이라 하고, $\sigma_1, \sigma_2, \cdots, \sigma_r$  을 영이 아닌 $A$의 특이값 전체라고 하면

$$
V=\begin{bmatrix} \vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_r & | &\vec{v}_{r+1} & \vec{v}_{r+2} & \cdots & \vec{v}_n \end{bmatrix}
$$

$$
U=\begin{bmatrix} \vec{u}_1 & \vec{u}_2 & \cdots & \vec{u}_r & | &\vec{u}_{r+1} & \vec{u}_{r+2} & \cdots & \vec{u}_m \end{bmatrix}
$$  

(i) $ rank(A)=r$  

(ii) $\{\vec{u}_1,\vec{u}_2,\cdots,\vec{u}_r\}$ $\quad  \text{Col}(A)$ 의 정규직교기저

(iii) $\{\vec{u_{r+1}},\vec{u_{r+2}},\cdots,\vec{u}_m\}$ $\quad  \text{Null}(A^T)$ 의 정규직교기저

(iv) $\{\vec{v}_1,\vec{v}_2,\cdots,\vec{v}_r\}$ $\quad   \text{Row}(A)$ 의 정규직교기저

(v) $\{\vec{v_{r+1}},\vec{v_{r+2}},\cdots,\vec{v}_n\}$ $\quad  \text{Null}(A)$ 의 정규직교기저
  
#### (4) 유사역원(Pseudo Inverse, Moore-Penrose Inverse)

$m\times n$행렬 $A$에 대하여 $V, \Sigma, U$ 를 특이값분해행렬이라 하자. 즉, $A=U\Sigma V^T$ 이다.  
이 때  $A^+=U\Sigma^+ V^T$ 를 $A$의 유사역원이라 한다. 여기서 $\Sigma^+$는 다음과 같다.

$$
\Sigma^+=\begin{bmatrix} D^{-1} & 0 \\ 0 & 0 \end{bmatrix}_{n\times m}, \qquad

D^{-1}=\begin{bmatrix} \sigma_1^{-1} & 0 & \cdots & 0 \\ 0 & \sigma_2^{-1} &\cdots & 0 \\ \cdots & \cdots & \cdots &\cdots \\ 0 & 0 & \cdots & \sigma_r^{-1}\end{bmatrix}_{r\times r}
$$  

만약 $A$가 가역행렬이면 $A^{-1}=A^+$ 이다.

---

## 4. 주성분 분석(PCA; Principal Component Analysis)

### (1) 목적

- 고차원 데이터를 정보 손실을 최소화하면서 저차원으로 투영하는 것이다.  
- 분산이 클수록 데이터를 잘 설명할 수 있다.  
- 따라서, 데이터의 분산이 가장 큰 방향(주성분)을 찾는 것이 핵심이다.  

### (2) 방법

기존 입력 변수를 표현하는 좌표축들을 다른 좌표축으로 변환시킨 뒤 데이터를 가장 잘 설명할 수 있는(분산이 가장 큰) 몇몇 차원만 선택하여 차원을 줄인다.

> **예시**
>
> **Step 1.** 2개의 feature X, Y 존재
>
>
> <img src='/assets/images/machinelearning/PCA_pca1.webp' width="600" alt="PCA_pca1" style="display: block; margin: 0;">
>
> **Step 2.** 분산이 가장 큰 좌표축과 해당 좌표축과 서로 직교하는 새로운 축을 찾는다 → V1 (분산이 가장 큰 좌표축), V2
>
>
> <img src='/assets/images/machinelearning/PCA_pca2.webp' width="600" alt="PCA_pca2" style="display: block; margin: 0;">
>
> **Step 3.** V1, V2를 새로운 좌표축 PC1, PC2로 만들고, 데이터를 가장 잘 설명할 수 있는 좌표축 PC1만 남겨둔다.
>
>
> <img src='/assets/images/machinelearning/PCA_pca3.webp' width="600" alt="PCA_pca3" style="display: block; margin: 0;">
>
> [Principal Component Analysis Guide & Example](https://statisticsbyjim.com/basics/principal-component-analysis/)

### (2) 수식을 이용한 구체적인 단계 설명

#### [1] 데이터 정규화 (평균 제거)

$X$ 를 $n$개의 관측치와 $p$개의 feature를 가지고 있는 데이터라고 하자: $X \in \mathbb{R}^{n\times p}$  
우선 각 feature에 대해 중심화를 수행하여 각 특성의 평균이 0이 되게 한다.  
각 관측치 $i=1,2,\cdots,n$ 와 feature $j=1,2,\cdots,p$ 에 대해

① 열별 평균 계산

$\mu_j = \dfrac{1}{n} \sum_{i=1}^n X_{ij}, \quad \text{for } j = 1, 2, \dots, p$

이를 벡터로 표시하면,  

$\vec{\mu} = \dfrac{1}{n} X^\top \mathbf{1}_n$, $\quad \mathbf{1}_n \in \mathbb{R}^{n\times 1}$ 은 모든 성분이 1인 열벡터.

따라서 $\vec{\mu} = \begin{bmatrix} \mu_1 \\\ \mu_2 \\\ \vdots \\\ \mu_p \end{bmatrix} \in \mathbb{R}^{p\times 1}$ 인 p차원의 열벡터이다.

② 평균 제거

$\tilde{X_{ij}} = X_{ij} - \mu_j, \quad \text{for } i = 1, \dots, n,\ j = 1, \dots, p$

이를 벡터로 표시하면,

$\tilde{X} = X - \mathbf{1}_n \vec{\mu}^\top ,\quad \tilde{X} \in \mathbb{R}^{n\times p}$

#### [2] SVD 수행

$\tilde{X} = U \Sigma V^\top$

이 때  
① $V$는 직교행렬이므로 $V$의 열벡터들은 정규직교이며,  
② 각 특이값(표준편차)  
$\sigma_1=\sqrt{\lambda_1} \ge \sigma_2=\sqrt{\lambda_2} \ge \cdots \ge \sigma_n=\sqrt{\lambda_n} \ge 0$ 으로부터 도출된 정규직교 벡터들을 대응되게 순서대로 나열한 행렬 $V=[\vec{v}_1 \quad \vec{v}_2 \quad \cdots \quad \vec{v}_n]$ 이다.  
$\quad$ **즉, $V$의 열벡터들은 특이값의 제곱(=분산)의 내림차순 기준으로 정렬되어 있다.**

> **공분산 행렬과 SVD**  
>
> - 평균 0으로 중심화된 데이터에 대해 행렬 $\tilde{X} \in \mathbb{R}^{n\times p}$ 의 공분산 행렬은
> $Cov(\tilde{X}) = \frac{1}{n} \tilde{X}^\top \tilde{X}$ 인데 이 공분산행렬을 고유값 분해하면  
> $Cov(\tilde{X}) = V \Lambda V^\top$ ($\Lambda = diag(\lambda_1,\cdots,\lambda_p)$ : 고유값들(분산 크기, 내림차순), $V \in \mathbb{R}^{p\times p}$ : 각 $\lambda$ 에 대응되는 고유벡터들)이다.  
>
> - 이제 $\tilde{X}$ 를 SVD 분해한 $\tilde{X} = U \Sigma V^\top$ 을 대입하면 $U=[\vec{u}_1 \quad \vec{u}_2 \quad \cdots \quad \vec{u}_n]$ 은 직교행렬 $(\{\vec{u}_1, \cdots,\vec{u}_n\}$ 가 정규직교집합$)$ 이므로
>
> $$
> \tilde{X}^\top \tilde{X}
> = (U \Sigma V^\top)^\top (U \Sigma V^\top)
> = V \Sigma^\top U^\top U \Sigma V^\top
> = V \Sigma^2 V^\top
> $$
>
> $\qquad \therefore \tilde{X}^\top \tilde{X}$ 의 고유값: $\Sigma^2$ 의 대각 원소 (**즉, 분산**)  
> $\qquad \quad \tilde{X}^\top \tilde{X}$ 의 고유벡터: $V$ 의 열벡터

#### [3] 주성분 선택

$V=[\vec{v}_1 \quad \vec{v}_2 \quad \cdots \quad \vec{v}_n] \in \mathbb{R}^{p \times p}$  
상위 $k$개의 주성분 선택:  
$W = [\vec{v}_1 \quad \vec{v}_2 \quad \cdots \quad \vec{v}_k] \in \mathbb{R}^{p \times k}$

#### [4] 데이터 차원 축소

$Z = \tilde{X} W \in \mathbb{R}^{n \times k}$  
→ 이것이 원래 데이터 $X$를 $k$-차원 주성분 공간에 투영한 결과이다.

#### [5] 원래 차원으로 복원 (선택)

$\hat{X} = Z W^\top + \bar{X} \qquad \text{where} \quad \bar{X} = \mathbf{1}_n \vec{\mu}^\top \in \mathbb{R}^{n\times p}$ (데이터의 열평균 벡터, 즉 중심화 이전의 평균값)  
→ 복원된 데이터 $\hat{X} \in \mathbb{R}^{n \times p}$

> **참고: [1] 데이터 정규화에서 분산까지 1로 만드는 경우 (표준화: 평균 0, 분산 1)**  
> PCA에 반드시 필요한 과정은 아니지만, 각 feature의 단위(scale)이 매우 다를 때 수행한다:  
> $\qquad \qquad \qquad \qquad \text{standardized } \tilde{X} = \dfrac{X_{ij}-\mu_j}{\sigma_j}$  
> 벡터로 표현하는 경우: $X^{std} = (X - \vec{\mu}) \oslash \vec{\sigma}$  
> $\qquad \qquad \qquad \qquad \vec{\mu} \in \mathbb{R}^{1 \times p} :$ 평균 벡터  
> $\qquad \qquad \qquad \qquad \vec{\sigma} \in \mathbb{R}^{1 \times p} :$ 표준편차 벡터  
> $\qquad \qquad \qquad \qquad \oslash :$ 열 단위로 나누는 브로드캐스팅 연산 (element-wise division)  
>
> - 표준화하는 경우 이제 [2]에서 **공분산 행렬이 아니라 상관계수 행렬(correlation matrix)**의 고유분해를 수행하는 것과 같다.  
> - 결과적으로 찾는 주성분 방향은 동일한 방식으로 구해지지만, 단위가 정규화된 축에서의 최대 분산 방향이 될 뿐이다.  

### (3) 선형변환으로서의 PCA

PCA의 투영:  $Z = \tilde{X} W \qquad W = [\vec{v}_1, \dots, \vec{v}_k] \in \mathbb{R}^{p\times k}, \quad Z \in \mathbb{R}^{n\times k}$  

$\Rightarrow$ **따라서 PCA는 원점을 지나면서 기저를 회전 및 축소시키는 선형사상(linear transformation)이다.**

> $\tilde{X}$ 의 $i$번째 행벡터(샘플)의 전치를 $\tilde{x}_i$ 라고 하자. ($\vec{\tilde{x}_i} \in \mathbb{R}^{p}$ 는 $p$차원 열벡터)  
> 또한 $\vec{z}_i$ 를 $Z$ 의 $i$번째 행벡터(샘플)의 전치라 하면,  
> 각 샘플 $\vec{\tilde{x}_i}$ 에 대해 $\vec{z}_i = W^\top\vec{\tilde{x}_i} \in \mathbb{R}^{k}$ ($k$차원 열벡터) 이고,  
> 이는 곧 PCA가 $\vec{\tilde{x}_i}$ 를 $W^\top$ 라는 행렬을 곱해서 저차원 벡터 $\vec{z}_i$ 로 보내는 변환을 의미한다.  

---

## 5. Truncated SVD

기존 SVD를 다시 써 보면 다음과 같다.

**SVD:**

$$
A=U\Sigma V^T
$$

- $A \in \mathbb{R}^{m\times n}$
- $U \in \mathbb{R}^{n\times n}$
- $\Sigma \in \mathbb{R}^{n\times p}$
- $V \in \mathbb{R}^{p\times p}$

한편 PCA에서는 $V=[\vec{v}_1 \quad \vec{v}_2 \quad \cdots \quad \vec{v}_n] \in \mathbb{R}^{p \times p}$ 중 상위 $k$개의 주성분만 선택하는데, 이에 대응되는 상위 $k$ 개의 특이값(=정보량이 큰 주성분)을 $\Sigma$ 에서 선택하고, 또 $U$ 에서 대응되는 앞쪽의 $k$ 개의 열벡터만 선택하여 SVD를 다음과 같이 다시 써 볼 수 있다. 이를 Truncated SVD라 한다.

**Truncated SVD:**

$$
A\approx U_k\Sigma_k V_{k}^T
$$

- $A \in \mathbb{R}^{m\times n}$
- $U_k \in \mathbb{R}^{m\times k}$ : 앞쪽에서부터 대응되는 $k$ 개의 열벡터만 남긴 행렬
- $\Sigma_k \in \mathbb{R}^{k\times k}$ : 상위 $k$ 개의 특이값만 남긴 대각행렬
- $V_k \in \mathbb{R}^{n\times k}$ : 앞쪽에서부터 대응되는 $k$ 개의 열벡터만 남긴 행렬
  
**따라서 PCA를 위한 SVD를 할 것이라면 처음부터 Truncated SVD를 구해도 된다.**

$$
\tilde{X} \approx U_k \Sigma_k V_{k}^\top \qquad \Rightarrow \qquad Z = \tilde{X} W = \tilde{X} V_k (= U_k \Sigma_k \,\,\, V_{k}^\top V_k) = U_k \Sigma_K
$$

---

_**출처: 대학교 및 대학원에서 본인의 Lecture Note - 선형대수학, Linear Regression, Macroeconometrics, Advanced Machine Learning**_
