---
layout: pages
title: RL9 Actor-critic算法
date: 2025-04-27 20:05:51
index_img: /img/banner/rl.png
tags:
---

## 回顾

Actor-critic算法本质上就是policy gradient算法的一种
$$
\theta_{t+1} =  \theta_{t} + \alpha_t \frac{\color{blue}q_{\pi}(s_t, a_t)}{\pi(a_t|s_t, \theta_t)}\nabla_{\theta}(\pi(a_t|s_t,\theta_t))
$$

* 如果 $q_{\pi}(s_t, a_t)$ 使用MC算法采样估计，则对应 **REINFORCE** 算法
* 如果 $q_{\pi}(s_t, a_t)$ 使用TD算法估计，则对应 **Actor-critic** 算法

## QAC

![](/img/rl/rl9_qac.png)

* 使用一个神经网络利用**值函数近似** 的Sara算法更新$q_{w}(s,a)$， 进行 **value upate**
* 使用一个神经网络利用**策略梯度**算法优化策略$\pi_{\theta}(a|s)$， 进行 **policy update**

该算法是on-policy的，因为Sara值函数近似时取出的 $a_{t+1} \sim \pi$ 



## Advantage actor-critic(A2C)

### 优势函数

在策略梯度算法的基础上引入一个baseline
$$
\nabla J(\theta)= \mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)} \left[ \nabla_{\theta} \ln \pi(A|S, \theta) q_{\pi}(S, A) \right] = \mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)} \left[ \nabla_{\theta} \ln \pi(A|S, \theta) (q_{\pi}(S, A) -\textcolor{red}{b(S)}) \right]
$$
为什么要引入它？

* 引入后该随机变量的期望不会改变，也就是策略梯度不变
  * 考察 $\mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)} \left[ \nabla_{\theta} \ln \pi(A|S, \theta){b(S)}) \right] = 0$
* 但该随机变量的方差会变化，我们想要构造一个$b(s)$使得$\nabla_{\theta} \ln \pi(A|S, \theta) (q_{\pi}(S, A) -{b(S)})$整体的 **方差最小** ，这样保证更新时的稳定性，加快模型收敛

下面直接给出最优解：
$$
b^*(s) = \frac{\mathbb{E}_{A \sim \pi} \left[ \left\| \nabla_{\theta} \ln \pi(A|s, \theta) \right\|^2 q_{\pi}(s, A) \right]}{\mathbb{E}_{A \sim \pi} \left[ \left\| \nabla_{\theta} \ln \pi(A|s, \theta) \right\|^2 \right]}, \quad s \in \mathcal{S}.
$$
实际使用中，考虑计算和效率均衡，我们采用 
$$
b(s) = E_{A\sim \pi}[q_{\pi}(s,A)]= v_{\pi}(s)
$$
由此我们得到**Advantage actor-critic**算法：
$$
\begin{align*}
\theta_{t+1} &= \theta_{t} + \alpha\mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)} \left[ \nabla_{\theta} \ln \pi(A|S, \theta) \textcolor{blue}{(q_{\pi}(S, A)-v_{\pi}(S))} \right] \\
&= \theta_{t} + \alpha\mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)} \left[ \nabla_{\theta} \ln \pi(A|S, \theta) \textcolor{blue}{(\delta_{\pi}(S, A)} \right] 
\end{align*}
$$

* $\delta_{\pi}(S, A) = q_{\pi}(S, A)-v_{\pi}(S)$ 被称为**优势函数**（Advantage function），体现每个动作状态价值对于平均动作状态价值的**相对值**

进一步地
$$
q_{\pi}(s_t, a_t)-v_{\pi}(s_t) \approx r_{t+1} + \gamma v_{\pi}(s_{t+1}) -  v_{\pi}(s_{t})
$$

* 这样就可以将需要训练的神经网络数量减少一个

![](/img/rl/rl9_a2c.png)



### Importance Sampling

由于策略梯度中$\nabla J(\theta) = E_{S\sim \eta, A\sim\pi}[*]$, 采样出的数据需要服从当前的策略， 因此都是**on-policy**的

**importance sampling** 就是一种将on-policy转化为off-policy的算法

假设我们已知两个分布$p_0, p_1$

问题简述：

已有的数据$x \sim p_1$,  $p_0(x)$ 和 $p_1(x)$ 两个神经网络，如果估计$\mathbb{E}_{x\sim x_0}[x]$

有读者会认为，诶，$p_0(x)$不是有吗，我们直接拿所有的$x$做 $\sum xp_0(x)$ 不就行了吗

* 经典错误，这里的数据$x$是服从$p_1$的， 实际上$p_1$和$p_0$ 数据的交集可能很少
* 例如，$p_1(x)$ 主要集中在 $x \in [0, 1]$ 区间内，而 $p_0(x)$ 则主要集中在 $x \in [2, 3]$ 区间内。如果你从 $p_1(x)$ 中采样数据，这些数据在 $[2, 3]$ 区间的概率非常小或几乎为零，因此你无法通过简单地使用 $\sum x p_0(x)$ 来计算 $\mathbb{E}_{x \sim p_0}[x]$，因为你的数据分布并没有覆盖到 $p_0(x)$中的那个重要区间。

正确做法：
$$
\mathbb{E}_{x \sim p_0}[x] = \mathbb{E}_{x \sim p_1} \left[ x \cdot \textcolor{red}{\frac{p_0(x)}{p_1(x)}} \right] \approx \frac{1}{n}\sum^{n}_{x=1}x_i\frac{p_0(x_i)}{p_1(x_i)}
$$

* $\frac{p_0(x_i)}{p_1(x_i)}$ 被称为重要性权重

假设behavior policy为$\beta$, 我们的目标是利用由$\beta$ 生成的数据最大化以下目标
$$
J(\theta) = \sum d_{\beta}v_{\pi}(s)
$$
off-policy 策略梯度理论：
$$
\begin{align*}
\nabla_{\theta} J(\theta) &= \mathbb{E}_{S \sim \rho, A \sim \pi(S, \theta)} \left[\textcolor{red}{\frac{\pi(A|S,\theta)}{\beta(A|S)}} \nabla_{\theta} \ln \pi(A|S, \theta) q_{\pi}(S, A) \right] \\
\end{align*}
$$

* 其中$\rho$是稳态分布

### off-policy版本

目标函数为
$$
J_{\theta} = \mathbb E_{s\sim d_{\beta}(s)}[v_{\pi}(s)]
$$

* 有读者会问，不是才讲重要性采样吗，我们的目标为什么不是$\mathbb E_{s\sim d_{\pi}(s)}[v_{\pi}(s)]$
* 因为事实上，我们现在要做off-policy，off-policy的定义就是数据$s$ 从另一个策略中获得
* 除此之外，一个直觉是数据中$s$服从什么分布对策略学习不太重要(因为无论什么分布大概率所有状态都覆盖得到），但$(a|s)$从状态中选取的动作对策略学习就很重要了！

那读者可能又要问了，那既然这样，${\frac{\pi(A|S,\theta)}{\beta(A|S)}}$又是哪里来的？

![](/img/rl/rl9_off_policy.png)

* 证明如上，我们要把从$\beta$ 采样的数据对$(s,a)$ 这个行为写为对应的表达式，S服从的分布通常不重要，重要的是动作A



## Deterministic actor-critic

 先前的神经网络的输出是每个动作的概率，且因为目标函数中ln的原因导致学习的策略输出概率一定都大于0，因此一定不是确定性的策略

我们将神经网络的输出改为动作是学习一个确定性的策略
$$
u(s,\theta) = a
$$
目标函数为
$$
\begin{align*}
J_{\theta} &= \sum_{s\in S}d_{u}(s)r_{u}(s) \\
&= \mathbb E_{s\sim d_{u}(s)}[r_{u}(s)] \\

\end{align*}
$$
计算得到其梯度：
$$
\begin{align*}
\nabla_{\theta} J(\theta) &= \sum_{s \in S} d_\mu(s) \nabla_\theta \mu(s) \left( \nabla_a q_\mu(s, a) \right) \Big|_{a = \mu(s)} \\

&= \mathbb{E}_{S \sim d_\mu} \left[ \nabla_\theta \mu(S) \left( \nabla_a q_\mu(S, a) \right) \Big|_{a = \mu(S)} \right]
\end{align*}
$$

* 先对a求导，之后再将a替换为$u(s)$


$$
\theta_{t+1} =  \theta_{t} + \alpha E_{S \sim d_u}\left[ \nabla_{\theta} u(S) (\nabla_a q_{u}(S,a) )|a=u(S)\right]
$$

* 随机梯度

$$
\theta_{t+1} =  \theta_{t} + \alpha_t\left[ \nabla_{\theta} u(s_t) (\nabla_a q_{u}(s_t,a) )|a=u(s_t)\right]
$$

天然的off-policy，不需要a服从特定的分布，理论上都可以

进一步可以使用动作状态函数对应的优势函数进行优化值函数估计

![](/img/rl/rl9_dpg.png)

