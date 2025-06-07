---
layout: pages
title: RL8 策略梯度
date: 2025-04-27 17:19:26
index_img: /img/banner/rl.png 
tags:
---

## 回顾

在这之前的方法，都是based on value，通过优化$q(s,a)$， 来进一步得到最优策略。我们能不能用神经网络直接对策略进行拟合并优化呢？

## Metircs

回顾最优策略的定义
$$
v_{\pi^{*}}(s) \ge v_{\pi}(s) \space {\forall \pi}
$$
一个自然而然想要最大化的目标即$\bar{v}_{\pi}$
$$
\begin{align*}
\bar{v}_{\pi} &= \mathbb E_{s\sim d_{\pi}(s)}[v_{\pi}(s)] \\
&= \sum_{s \in S}d(s)v_{\pi}(s) \\
&= \sum_{s \in S}d_{\pi}(s)E[\sum_{t=0}^{\infty}\gamma ^tR_{t+1}|S_0=s]\\
&= E[\sum_{t=0}^{\infty}\gamma ^tR_{t+1}] 
\end{align*}
$$
另外一个常见的最大化目标为$\bar{r}_{\pi}$
$$
\begin{align*}
\bar{r}_{\pi} &= \mathbb E_{s\sim d_{\pi}(s)}[r_{\pi}(s)] \\
&= \sum_{s\in S}d_{\pi}(s)r_{\pi}(s) \\
&= \sum_{s \in S}d_{\pi}(s)\sum_{a \in A}\pi(a|s)r(s,a) \\
&=\lim_{n \to \infty}\frac{1}{n} E[\sum_{k=1}^{n}R_{t+k}|S_t = s_0] \\
&= \lim_{n \to \infty}\frac{1}{n} E[\sum_{k=1}^{n}R_{t+k}] 
\end{align*}
$$
实际上二者是等价的，符合以下关系
$$
\bar{r}_{\pi} = (1-\gamma)\bar{v}_{\pi}
$$

## 梯度更新

由于我们想要最大化metrics，故采用梯度上升的方法
$$
\theta_{t+1} =  \theta_{t} + \alpha\nabla_{\theta}J(\theta)
$$
策略梯度定理告诉我们这些metrics的梯度都近似为下面的式子：
$$
\begin{align*}
\nabla_{\theta} J(\theta) &= \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi(a|s, \theta) q_{\pi}(s, a) \\
&= \mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)} \left[ \nabla_{\theta} \ln \pi(A|S, \theta) q_{\pi}(S, A) \right]
\end{align*}
$$
然而要写为这种形式，得保证 $\pi(A|S, \theta)>0$ 恒成立，故考虑做softmax
$$
\pi(a|s, \theta) = \frac{e^{h(s,a,\theta)}}{\sum_{a' \in \mathcal{A}} e^{h(s,a',\theta)}}, \quad a \in \mathcal{A}
$$


故梯度上升可以写为：
$$
\theta_{t+1} =  \theta_{t} + \alpha E_{S \sim \eta, A \sim \pi(S, \theta)}\left[ \nabla_{\theta} \ln \pi(A|S, \theta) q_{\pi}(S, A) \right]
$$
然而上面的期望由于真实的$\pi, S, A$分布无法得到，故采用随机梯度上升
$$
\theta_{t+1} =  \theta_{t} + \alpha_t \left[ \nabla_{\theta} \ln \pi(a_t|s_t, \theta_{t}) q_{\pi}(s_t, a_t) \right]
$$

## 另一个角度

因为
$$
\nabla_{\theta} \ln \pi(a_t|s_t, \theta_t) = \frac{\nabla_{\theta_t}(\pi(a_t|s_t,\theta_t))}{\pi(a_t|s_t, \theta_t)}
$$
进一步可以写为
$$
\theta_{t+1} =  \theta_{t} + \alpha_t \frac{q_{\pi}(s_t, a_t)}{\pi(a_t|s_t, \theta_t)}\nabla_{\theta}(\pi(a_t|s_t,\theta_t))
$$
如果我们令$\beta_t =\frac{q_{\pi}(s_t, a_t)}{\pi(a_t|s_t, \theta_t)}$
$$
\theta_{t+1} =  \theta_{t} + \alpha_t \beta_t\nabla_{\theta}(\pi(a_t|s_t,\theta_t))
$$
这实际上就在 **优化策略**!

$\beta_t$保证

* 当 $q_{\pi}(s_t, a_t) \le  0$时， $\pi(a_t|s_t,\theta_{t+1}) \le \pi(a_t|s_t,\theta_{t})$
* 当 $q_{\pi}(s_t, a_t) \ge 0$时， $\pi(a_t|s_t,\theta_{t+1}) \ge \pi(a_t|s_t,\theta_{t})$

## REINFORCE

由于真实的$q_{\pi}(s_t, a_t)$我们不知道，需要通过其他方式来获得。

特别的，通过 **MC方法** 从$(s_t,a_t)$开始采样一条轨迹计算$q_t(s_t, a_t)$用来近似$q_{\pi}(s_t, a_t)$， 来进行策略梯度，则该方法被称为**REINFORCE**

