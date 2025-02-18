---
layout: pages
title: RL5 随机近似与随机梯度下降
date: 2025-02-13 14:52:05
index_img: /img/banner/rl.png 
tags:
- Reinforcement learning
---

## 回顾

在MC 算法中最重要的就是利用Mean Estimation 对 $q(s,a)$ 进行估计。

其本质上是一个利用独立且同分布（i.i.d.）$\{x_i\}_{i=1}^{n}$的sample来对随机变量$\mathcal{X}$ 的期望估计
$$
E(\mathcal{X}) \approx \bar{x} =  \frac{1}{n}\sum_{i=1}^{n}x_{i}
$$
如果我们不想等到全部$n$个到之后再估计，而是来$k$个就更新一次估计，将该方法转化为 **incremental update**的方式,应该怎么办呢?这就引入随机近似的思想

## 随机近似

假设
$$
w_{k} = \frac{1}{k-1}\sum_{i=1}^{k-1}x_i
$$
则
$$
w_{k+1} = \frac{1}{k}\sum_{i=1}^{k}x_i = \frac{1}{k}[(k-1)w_k + x_k] = w_k - \frac{1}{k}(w_k - x_k)
$$
**随机近似**将其写为更一般的形式
$$
w_{k+1} = w_k - \alpha_k(w_k - x_k)
$$

## RM算法 | Robbins-Monro algorithm

Robbins-Monro algorithm可以帮助我们解决形如
$$
g(w) = 0
$$
其中 $w$ 是未知量， $g$ 是一个未知表达式的black box函数

RM算法流程：
$$
g(w_k,  \eta_k) = g(w_k) + \eta_k
$$
$g(w_k,  \eta_k)$是对$g(w_k)$的一个观测，其中 $\eta_k$ 是观测误差 
$$
\begin{equation}
    w_{k+1} = w_k - a_k g(w_k,  \eta_k), \quad k = 1, 2, 3, \ldots
    \tag{6.5}
\end{equation}
$$
RM算法的收敛性依赖于以下条件：

(a) $0 < c_1 \leq \nabla_w g(w) \leq c_2$对所有 $w$ 都成立

* 假设$g(w) =\nabla_{w} J(w)$, 则$\nabla_w g(w)$实际上对应$J(w)$的Hessian 矩阵。Hessian 矩阵为正符合 $J(w)$ 的**凸函数**性质

(b) $\sum_{k=1}^{\infty} a_k = \infty$且  $\sum_{k=1}^{\infty} a_k^2 < \infty$

* $\sum_{k=1}^{\infty} a_k^2 < \infty$ 表明 $a_k$ 最终收敛到 0 当 $k \to \infty$ , 避免发散

* $\sum_{k=1}^{\infty} a_k = \infty$意味着 $a_k$ 最终收敛到 0 的速度**不能太快**，确保随着迭代次数增加，算法能够不断更新。
* 实践中 $a_k = \frac{1}{k}$ 符合上述两个条件

(c)  $\mathbb{E}[\eta_k | H_k] = 0$ 且 $\mathbb{E}[\eta_k^2 | H_k] < \infty$；

其中 $H_k = \{ w_k, w_{k-1}, \dots \}$，则 $w_k$ 几乎肯定收敛到满足 $g(w^*) = 0$ 的根 $w^*$ 。

**将RM算法应用到Mean Estimation** 

令$g(w)$为：
$$
g(w) = w -  E[X] 
$$
给定一个$w_k$, 我们可以得到一个噪声观察
$$
\begin{align*}
\tilde{g}(w_k, x) &= w_k - x \\
&= (w_k - \mathbb{E}[x]) + (\mathbb{E}[x] - x) \\
&= g(w_k) + \eta_k
\end{align*}
$$
则RM算法求解：
$$
w_{k+1} = w_k - a_k g(w_k,  \eta_k) =  w_k - \alpha_k(w_k - x_k)
$$
