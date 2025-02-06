---
title: Machine Learning 3 逻辑回归
index_img: /img/banner/machine_learning.jpg 
date: 2024-05-30 22:21:47
tags:
categories:
- [Machine learning]
---

# 逻辑回归

## 逻辑回归的引入

考虑预测值 $y$ 不再连续，而是离散值。这时候线性回归不再适用。

对于二分类问题$y \in \{ 0, 1\}$ ,不妨使得假设函数$h_{\theta}(x)$ 预测$p(y=1|x)$,即$x$是种类$y=1$的概率

构造逻辑回归函数：

$$h_{\theta}(x) = g(z) = g(\theta x) = \frac{1}{1+e^{-\theta x}}$$

> $sigmoid$ 函数：
>
> $g(z) = \frac{1}{1 + e^{-z}}$
>
> 其导数 
>
> $g(z)^{’} = g(z)(1-g(z))$

则$p(y|x) = h_{\theta}(x)^{y}[1-h_{\theta}(x)]^{1-y}$

我们定义：

$if \space h_{\theta}(x) >= 0.5, y = 1 \\ else \space y = 0$ 

进而问题转化为

$z >= 0 => y = 1$

$z < 0 => y =0$

## 代价函数

$$J(\theta) = -\frac{1}{m}\sum y_{i}lnh_{\theta}(x) + (1-y_{i})ln(1-h_{\theta}(x))$$

> 可由极大似然估计推导得到
>
> $L(\theta) = \prod_{i=1}^{m} p(y_{i}|x_{i})$​
>
> $lnL(\theta) = \sum_{i=1}^{m}y_{i}lnh_{\theta}(x) + (1-y_{i})ln(1-h_{\theta}(x))$
>
> 极大似然求极大，而损失函数求极小，在似然函数前去负号 + 求个平均就得到了损失函数

对$\theta$ 求偏导：
$$
\frac{\alpha J_{i}}{\alpha \theta} = -[\frac{y_{i}}{h_{\theta(x_{i})}} g(\theta x)(1-g(\theta x_{i}))x_{i} + (-1)\frac{1-y_{i}}{1-h_{\theta(x_{i})}} g(\theta x_{i})(1-g(\theta x_{i}))x_{i} ]\\= -(y_{i} - g(\theta x_{i}) )x_{i} \\= -(y_{i} - h_{\theta}(x))x_{i}
$$

> 与线性回归的偏导形式一样，具体原因参考CS229指数族部分讲解

* 从概率角度推导逻辑二分类问题的代价函数
* 但为何不采用MSE [为什么分类问题的损失函数采用交叉熵而不是均方误差MSE？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/104130889)
