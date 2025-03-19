---
layout: pages
title: RL6 时序差分方法
date: 2025-03-04 14:13:16
index_img: /img/banner/rl.png 
tags:
---

## 朴素TD

* 相较于蒙特卡洛采样一整个episode后才对策略进行更新，时序差分每执行一步即可更新值函数

朴素的**时序差分方法**：依靠数据而非模型，遵循给定policy的计算特定状态的状态价值
$$
\begin{align}
v_{t+1}(s_t) &= v_t(s_t) - \alpha_t(s_t) \left[ v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1})) \right], \tag{1} \\
v_{t+1}(s) &= v_t(s), \quad \text{for all } s \neq s_t, \tag{2}
\end{align}
$$

* $s_t$ 是agent在 $t$ 时刻执行到的状态 

* $r_{t+1} + \gamma v_t(s_{t+1})$ 为TD Target，因为TD算法本质是希望$v(s_t)$ 朝 $\bar{v} = r_{t+1} + \gamma v(s_{t+ 1})$ 去逼近
* $v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1}))$ 为TD Error, 当策略为最优策略$\pi$ 时， $\delta_t = v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1})) = 0$

|                              TD                              |                              MC                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| **增量式**: 每次接收到一个经验样本（如状态转移和奖励）后立即更新状态价值 | **非增量式**: 必须等待整个 episode 结束后计算回报（discounted return），最后才能更新状态价值。 |
| 可以处理**连续任务 (continuing tasks)** 和**分段任务 (episodic tasks)** |            只能处理**分段任务 (episodic tasks)**             |
| 方差较低：估计所用到的随机变量少，例如Sara算法中估计$q_{\pi}(s_t,a_t)$，仅需要$R_{t+1}, S_{t+1}, A_{t+1}$ | 方差较大: 为了估计$q_{\pi}(s_t a_t)$, 需要采样$R_{t+1}, \gamma R_{t+2}...$, 假设每个episode长度为$L$, 每个状态下可执行的动作空间大小为$\mathbb A$, 则总共$\mathbb A ^{L}$的空间。如果我们sample出的episode数量太少，很容易方差过大，估计有偏差 |

## Sara

上节提及的朴素TD算法，只是用来在特定策略对状态价值。

而本节的Sara则是给定一个策略对动作价值估计

根据t时刻状态$s_t$, 根据策略$\pi$, 采样得到$(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$, 以此来对动作函数进行更新
$$
\begin{align}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) - \alpha_t(s_t, q_t) \left[ q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})) \right], \tag{1} \\
q_{t+1}(s, a) &= q_t(s, a), \quad \text{for all } s \neq s_t, a \neq a_t \tag{2}
\end{align}
$$
其原理也是平均估计
$$
q_{\pi}(s,a) = E[R+ q_{\pi}(S_{t+1},A_{t+1}|s,a)] 
$$


如何利用Sara进行最优策略的寻找呢？

![](/img/rl/rl6_sara.png)

* 第一步，使用sara算法更新动作函数
* 第二步，根据动作函数更新$\epsilon - greedy$策略



n-step Sara就是将采样步数扩展为n步，n步之后才更新一次
$$
\begin{align}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) - \alpha_t(s_t, q_t) \left[ q_t(s_t, a_t) - (r_{t+1} + ...\gamma^{n-1}r_{t+n} + \gamma^n q_t(s_{t+n}, a_{t+n})) \right], \tag{1} \\
q_{t+1}(s, a) &= q_t(s, a), \quad \text{for all } s \neq s_t, a \neq a_t \tag{2}
\end{align}
$$
当n值很大时，算法靠近MC算法

当n值很小时，算法靠近Sara算法

## Q-learning

相比Sara需要配合action value mean estimation和policy improvement两步来寻找最优策略，Q-learning直接通过以下公式便可直接推导出最优动作函数和动作策略
$$
\begin{align}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) - \alpha_t(s_t, q_t) \left[ q_t(s_t, a_t) - (r_{t+1} + \gamma \max_{a\in \mathcal A(s_t+1)}q_t(s_{t+1},a)) \right], \tag{1} \\
q_{t+1}(s, a) &= q_t(s, a), \quad \text{for all } s \neq s_t, a \neq a_t \tag{2}
\end{align}
$$

其实质为
$$
q(s,a) = \mathbb E[R + \gamma \max_{a \in \mathcal A(s_{t+1})}q(S_{t+1},a)|S_t = s,A_t = a]
$$
实际上这等同于求解**贝尔曼最优公式**

{% note info %}

证明如下

对该方程两边关于状态 s 中的动作 a 取最大值：

$$
\max_{a \in \mathcal{A}(s)} q(s, a) = \max_{a \in \mathcal{A}(s)} \left[ \sum_r p(r|s, a) r + \gamma \sum_{s'} p(s'|s, a) \max_{a \in \mathcal{A}(s')} q(s', a) \right].
$$
我们采用符号$v(s)$代替$\max_{a \in \mathcal{A}(s)} q(s, a) $

$v(s) \doteq \max_{a \in \mathcal{A}(s)} q(s, a).$

使用这种符号表示，方程变为：

$$
v(s) = \max_{a \in \mathcal{A}(s)} \left[ \sum_r p(r|s, a) r + \gamma \sum_{s'} p(s'|s, a) v(s') \right]  \\
 =\max_{\pi} \sum_{a \in \mathcal{A}(s)} \pi(a|s) \left[ \sum_r p(r|s, a) r + \gamma \sum_{s'} p(s'|s, a) v(s') \right]
$$
因此，这个方程用即时奖励和下一个状态 \( s' \) 的最优值来表达状态 \( s \) 的最优值。

这就是第3章介绍的状态值的贝尔曼最优方程。

$$
v_*(s) = \max_{\pi} \sum_{a \in \mathcal{A}(s)} \pi(a|s) \left[ \sum_r p(r|s, a) r + \gamma \sum_{s'} p(s'|s, a) v_*(s') \right]
$$
其中最大值是对所有可能策略 π 而言的。

{% endnote %}

## Off-policy vs on-policy

基于数据的强化学习通常具有两个policy：

* behavior policy: 用于生成经验样本的策略
* target policy: 通过不断更新以收敛到最优策略的策略。

*on-policy learning* 即behavior policy和target policy相同

*off-policy learning* 即behavior policy和target policy不同，此时behavior policy通常用一个探索能力更强的策略代替，例如人类

{% note warning %}

策略迭代和价值迭代，某种意义上来讲其实就是on-policy和off-policy

{% endnote %}

MC，TD，Sara算法本质都是对贝尔曼公式的求解，对应于**策略迭代算法**（如有遗忘详见[策略迭代算法](https://xrlexpert.github.io/2025/02/12/RL4/#策略迭代算法-Policy-iteration-algorithm)）的第一步policy evaluation

其本质
$$
q_{\pi}(s,a) = E_{s, R \sim environment, A_{t+1} \sim \pi(A|s)}[R+ q_{\pi}(S_{t+1},A_{t+1}|s,a)]
$$

* 其每一步得到的期望实质是在当前的策略下最优的，并非全局最优，只有当前策略达到了optimal policy，其值才是最优状态价值函数
* 对于策略$\pi_t$, 更新后变为$\pi_{t+n}$,这个时候$\pi_t$收集的数据，对于$\pi_{t+n}$是无效的，要求**“数据和策略同时进步”**。所以不能和off-policy一样，预先用策略收集然后可以使用另一个策略来训练。

所以上述算法第二步policy improvement采用的策略需要第一步保持一致，是on-policy的



而Q-learning本质上为对贝尔曼最优公式求解，且对应为**值迭代算法**

其实质
$$
q(s,a) = \mathbb E_{s, R\sim environment}[R + \gamma \max_{a \in \mathcal A(s_{t+1})}q(S_{t+1},a)|S_t = s,A_t = a]
$$

* 其期望与当前策略无关，因此可以离线收集数据，然后再进行更新，不要求**”数据和策略同时进步“**



## 统一理解

上述算法都可以写成如下形式：
$$
q_{t+1}(s,a) = q_t(s,a) - \alpha_t(q_t(s,a) - \bar{q_t}) \tag{3}
$$

| Algorithm    | Expression of the TD target $\bar{q}_t$ in (3)               |
| ------------ | ------------------------------------------------------------ |
| SARSA        | $\bar{q}_t = r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$         |
| n-step SARSA | $\bar{q}_t = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^n q_t(s_{t+n}, a_{t+n})$ |
| Q-learning   | $\bar{q}_t = r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)$        |
| Monte Carlo  | $\bar{q}_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$ |



| Algorithm    | Equation to be solved                                        |
| ------------ | ------------------------------------------------------------ |
| SARSA        | $\text{BE: } q_\pi(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) {\|}S_t = s, A_t = a \right]$ |
| n-step SARSA | $\text{BE: } q_\pi(s, a) = \mathbb{E}[ R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) {\|}S_t = s, A_t = a ]$ |
| Q-learning   | $\text{BOE: } q(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_a q(S_{t+1}, a) {\|}S_t = s, A_t = a \right]$ |
| Monte Carlo  | $\text{BE: } q_\pi(s, a) = \mathbb{E}[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots {\|}S_t = s, A_t = a ]$ |
