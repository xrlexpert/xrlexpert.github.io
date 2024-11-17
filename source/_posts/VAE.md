---
title: VAE
index_img: /img/banner/VAE.png
date: 2024-11-17 13:36:38
tags:
- Deep Learning
- VAE
categories:
- Generative models
---

## 核心思想

- 已知输入数据$X$的样本$\{x_1, x_2, ......x_n\}$

- 假设一个隐式变量$z$服从常见的分布如正态分布等（先验知识）

- 希望训练一个生成器$\hat X = g(z)$使得$\hat X$尽可能逼近输入数据X的真实分布



## 从Auto Encoder到Variational Auto-Encoder

原始的AE思想很简单

- 用encoder原数据压缩，压缩后的特征可是视作隐式变量，之后再用decoder还原

![image.png](/img/VAE/AE.png)

但需要注意的是，AE压缩后的特征z是离散的（可以视作$\{z_1, z_2, ...z_n\}$）其能表示的空间有限，例如$z_1 = [1,20,0.5,19]$,如果这里第一维改成0.5，如果生成器在训练的时候没有见过，则生成出来的效果可能不佳。

能生成的X分布（离散的）：

$P(X) = \sum_z P(X|z)P(z)$

![P(x|m)即为图中P(x)中蓝色部分，P(m)则是多项式分布是离散的。](/img/VAE/AE_px.png)

- 基于该**离散**的方式所能生成的P(X)能力有限

ok，想到这里，要想获得好的**生成**, 我们想尽可能地扩大隐式变量z的空间，怎么做呢？

- 与其让神经网络生成基于样本x对应的特征z（一个向量），我们不如让神经网络学习基于样本x的**隐式z的分布**（一个分布）不就好了嘛

![difference.png](/img/VAE/difference.png)

接下来就是我们的VAE之旅

## VAE实现

- 回顾目标，学习$P(X)$分布

- VAE提出先验知识：假设隐式变量$z \sim N (0,I)$,  $x|z \sim N(u(z), \sigma (z))$

  - 为什么是假设是正态分布，这样有什么好处？

- 根据全概率公式转化为

$P(X) = \int_z P(X|z)P(z)dz$

我们采用对数最大似然估计的方式，求解生成器$g$的参数$\theta$

$$
L(\theta) = \sum _x logp_{\theta}(x) =\sum_x log\int_z p_{\theta}(x|z)q(z)dz
$$
其中：
$$
\begin{align*}
\log p_{\theta}(x) 
&= \int_z q(z) \log p_{\theta}(x|z) \, dz \\
&= \int_z q(z) \cdot \log \left[ \frac{p_{\theta}(x|z) p(z)}{p_{\theta}(z|x)} \cdot \frac{q(z)}{q(z)} \right] dz \\
&= \int_z q(z) \big[ \log p_{\theta}(x|z) + \log \frac{p(z)}{q(z)} + \log \frac{q(z)}{p_{\theta}(z|x)} \big] dz \\
&= \underbrace{\mathbb{E}_{z \sim q(z)} \log p_{\theta}(x|z) 
- D_{\text{KL}}(q(z) \| p(z))}_{\text{ELBO}} 
+ \underbrace{D_{\text{KL}}(q(z) \| p_{\theta}(z|x))}_{\text{KL}} \\
&\geq \big[ \mathbb{E}_{z \sim q(z)} \log p_{\theta}(x|z) - D_{\text{KL}}(q(z) \| p(z)) \big] \quad \text{(ELBO)}
\end{align*}
$$

### EM算法

学长已经说的很好了

[链接](https://xyfjason.top/blog-main/2022/08/23/EM%E7%AE%97%E6%B3%95/)

总结EM

- E-step：取$q(z) = p_\theta(z|x)$,此时KL等于0

- M-step: 固定q(z),优化$\theta$，最大化ELBO

### 从EM到VAE

但在VAE中，取$q(z) = p_\theta(z|x)$这一步是做不到的，因为$p_\theta(z|x)$的解析式我们无法得出

![explain.png](/img/VAE/gpt.png)

- 注意上图中$p_{\theta}(z) = p(z)$

- 但E-step很巧妙在于，当我们**固定$\theta$时**，$logp_{\theta }(x)$**是固定的**，即 $ELBO + KL$ 固定。

- 虽然我们无法直接求出令KL = 0的$p_\theta(z|x)$的解析解,但是我们可以**通过最大化ELBO来隐式地最小化KL**，从而使得我们的q(z)逼近$p_\theta(z|x)$的最优解。

### 最终的损失函数

故VAE中采取的做法是将原始的EM转化为

- E-step：固定$\theta$,  优化$q(z)$，直接最大化ELBO

- M-step：固定$q(z)$,优化$\theta$，最大化ELBO

我们将$q(z)$进一步写为用参数$\phi$ 表示的$q_{\phi} (z|x)$这就是VAE中的encoder

同时令$p_{\theta}(z)$为我们的先验分布$N(0,I)$

- 由于两个都是最大化ELBO，且在使用梯度下降法时每次更新都是基于上一次的参数做调整，与这里的固定异曲同工。故VAE的Loss函数可以写为-ELBO

$\mathcal{L}_{\theta, \phi}(x)= -ELBO = -E_{z\sim q_{\phi}(z|x)} logp_{\theta}(x|z) + D_{kl}(q_{\phi}(z)||p(z))$

- 重构项：

![reconstruction.png](/img/VAE/reconstruct.png)

- 正则项：

![regularization.png](/img/VAE/regular.png)



