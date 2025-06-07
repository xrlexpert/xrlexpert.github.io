---
layout: pages
title: Inverse Problems
index_img: /img/banner/inverse_problems.png
date: 2025-05-16 10:12:13
tags:
---

## ReCall

### Gaussians and Score Function

已知 $p(z) = N(0, I)$ , $x = u + \Sigma ^{\frac{1}{2}}z$

则 $p(x) = N(u, \Sigma)$ , $p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}e^{-\frac{1}{2}(x-u)^T\Sigma^{-1}(x-u)}$

$\log p(x) = -\frac{1}{2}(x-u)^T\Sigma^{-1}(x-u)$

$\nabla_x \log p(x) =  -\Sigma^{-1}(x-u)$

### Tweedie’s Formula

假设 $x = u + \Sigma ^{\frac{1}{2}}z$ ，即 $x$ 是 $u$ 的带噪声版本， 如何从这个带噪声的 $x$ (observation) 中恢复出原始的 $u$ 呢？ 
$$
\mathbb E[u|x] = x + \Sigma \nabla_x \log p(x)
$$
在diffusion model中, 有
$$
p(x_t|x_0) = \mathcal N(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
$$
观测的随机变量为$x_t$

则应用Tweedie公式，得
$$
\mathbb E_{x_0 \sim p(x), x_t \sim p_{\sigma}(x_t)}[\sqrt{\bar{\alpha}_t}x_0|x_t] = \sqrt{\bar{\alpha}_t} \mathbb E[x_0|x_t] =x_t + (1-\bar{\alpha}_t)\nabla_x \log p(x_t |x_0)
$$

* 这里化边缘分布为条件分布，和score function 类似

$$
\mathbb E[x_0|x_t] = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t + (1-\bar{\alpha}_t)\nabla_x \log p(x_t |x_0))
$$



## Inverse Problems

问题定义：

已知观测 $y$, 其符合 $y = \mathcal A(x) + \sigma z， z \sim N(0，I)$， 如何找到对应于该观测 $y$ 的原始数据点 $x$

* $\mathcal A$ 为观测算子
* $z$ 为观测噪声
* 去模糊，超分，inpaint，都可以看作是逆问题。要从观测 $y$ (一张模糊的图像，一张低分辨率的图像，一张带掩码的图像) 恢复到diffusion 学习到的原始正常图像 $x$

![Pseudoinverse-Guided Diffusion Models for Inverse Problems, ICLR 2023](/img/diffusion/inverse_applications.png)

## Methods

方法：已知观测 $y$ 以及diffusion model学习到的$\nabla_x \log p(x_t)$, 计算 $\nabla_x \log p(x_t|y)$ 来引导 $y$ 恢复为 $x$
$$
\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t)
$$
$\nabla_x \log p(x_t)$ 通过diffusion model可以得到，问题转化为如何不用训练就计算$\nabla_x \log p(y|x_t)$

结论：
$$
p(y|x_t) \approx N(Ax_{0|t},\Sigma)
$$

* 其中$\Sigma = r^2AA^T + \sigma^2$

{% note success %}

考虑特殊的情况，$\mathcal A(x)$是一个线性变换，

则$\mathcal A(x)$可以用一个线性矩阵$A$ 表示  

$y =\mathcal A(x_0) + \sigma z=Ax_0 + \sigma z$, 

* 其中 $z \sim N(0,I)$

则 
$$
p(y|x_0) = N(Ax_0, \sigma^2I)
$$
令 $x_{0|t} = E[x_0|x_t]$ , $p(x_{0}|x_t) \approx N(x_{0|t}, r^2I)$

又：
$$
p(y) = \int p(y|x_0) p(x_0) \, dx_0
$$
加上 $x_t$ 的条件：

$$
p(y|x_t) = \int p(y|x_0, x_t) p(x_0|x_t) \, dx_0
$$
因为 $x_t$ 是从 $x_0$ 采样得到，故对于y没有提供额外信息：
$$
= \int p(y|x_0) p(x_0|x_t) \, dx_0
$$
因为

*  $p(y|x_0) = N(Ax_0, \sigma^2I)$

*  令$p(x_{0}|x_t) \approx  N(x_{0|t}, r^2I)$

所以
$$
p(y|x_t) \approx N(Ax_{0|t}, (r^2AA^T + \sigma^2) I)
$$
令 $\Sigma = r^2AA^T + \sigma^2$

则
$$
p(y|x_t) \approx N(Ax_{0|t},\Sigma)
$$
{% endnote %}


$$
\log p(y|x_t) \approx -\frac{1}{2}(y-Ax_{0|t})^T\Sigma^{-1}(y-Ax_{0|t})
$$

$$
\begin{align*}
\nabla_{x_t} \log p(y|x_t) &= \frac{\partial \log p(y|x_t)}{\partial x_0|t} \cdot \frac{\partial x_0|t}{\partial x_t} \\
&\approx \frac{\partial}{\partial x_0|t} \left( -\frac{1}{2} (y - A x_{0|t})^T \Sigma^{-1} (y - A x_{0|t}) \right) \cdot \frac{\partial x_{0|t}}{\partial x_t} \\
&= A^T \Sigma^{-1} (y - A x_{0|t}) \cdot \frac{\partial x_{0|t}}{\partial x_t} \\
&= A^T (\sigma^2 I + r^2 A A^T)^{-1} (y - A x_{0|t}) \cdot \frac{\partial x_{0|t}}{\partial x_t}
\end{align*}
$$

### Pseudoinverse guidance (ΠG) [^1]

当 $\sigma = 0$ 时，表示 $y$ 没有观测误差
$$
\begin{align*}
\nabla_{x_t} \log p(y|x_t) &= \frac{1}{r^2}A^T (A A^T)^{-1} (y - A x_{0|t}) \cdot \frac{\partial x_{0|t}}{\partial x_t}
\end{align*}
$$

* 此时 $A^T (A A^T)^{-1}$ 是 $A$ 的一个伪逆矩阵
* 当非线性时，需要人工构造伪逆函数，保证 $\mathcal A\mathcal A^{-1}\mathcal A(x)=\mathcal A(x)$



### DPS [^2]

当 $r = 0$ 时，
$$
p(y|x_t) \approx N(Ax_{0|t}, \sigma^2 I)
$$

* 不需要 $AA^T$ 矩阵，因此可以去掉$\mathcal A(x)$是一个线性变换的前提, 适用于广泛的观测算子

$$
p(y|x_t) \approx N(\mathcal A(x_{0|t}), \sigma^2 I)
$$

$$
\nabla_{x_t} \log p(y|x_t) = \frac{1}{2 \sigma^2}\nabla_{x_t}||y-\mathcal A(x_{0|t})||^2
$$

![](/img/diffusion/dps.png)

## References

[^1]: Song et al., Pseudoinverse-Guided Diffusion Models for Inverse Problems, ICLR 2023
[^2]: Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR 2023
[^3]: [CS492(D) Diffusion Models and Their Applications (KAIST, Fall 2024)](https://mhsung.github.io/kaist-cs492d-fall-2024/)
