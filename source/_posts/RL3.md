---
layout: pages
title: RL3 贝尔曼最优公式
index_img: /img/banner/rl.png 
date: 2025-02-12 13:59:20
tags:
- Reinforcement learning
---

## 回顾

已知策略，我们可以通过上一节提到的贝尔曼公式计算出状态价值和行动价值。

然而实际上我们想要的是最优的策略，下面就仔细讲讲如何得到最优策略

##  最优策略

最优策略定义:使得每个状态的状态价值最高的策略
$$
\pi^* = \arg\max_{\pi} V_\pi(s), \quad \forall s \in \mathcal{S}
$$

## Bellman optimality equation 贝尔曼最优公式

### 公式

Elementwise form:
$$
\begin{align*}
v(s) &= \max_{\pi} \sum_a \pi(a|s) \left( \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v(s') \right), & \forall s \in \mathcal{S} \\
&= \max_{\pi} \sum_a \pi(a|s) q(s,a) & s \in \mathcal{S}
\end{align*}
$$
Matrix-vector form:
$$
\begin{align*}
V &= \max_{\pi}  \cdot (r_{\pi} + \gamma P_{\pi} \cdot V)
\end{align*}
$$


最优贝尔曼公式已知$r_{\pi}, P_{\pi}, \gamma$, 未知$\pi, V$

**理想解法**：

1. 求得使得$r_{\pi} + \gamma P_{\pi} \cdot V$最大的$\pi$
2. 带入$\pi$之后通过等式求解$V$

假设已知$q(s,a)$,又因为$\sum_{a} \pi(a|s) = 1$

则
$$
\begin{align*}\max_{\pi} \sum_a \pi(a|s) q(s,a)  = \max_{a \in \mathcal{A(s)} }q(s|a)\end{align*}
$$
即选择最大的动作价值作为确定动作（概率为1）

由上方的解法可知，第一步求解$\pi$时，是固定$v$不变的。故贝尔曼最优公式的右侧可以写为一个关于固定值v的函数$f(V) = \max_{\pi}  \cdot (r_{\pi} + \gamma P_{\pi} \cdot V)$

### 简化后的BOE(贝尔曼最优公式)

贝尔曼最优公式化简为
$$
v = f(v)
$$

* 故我们的最终目标就是求解该方程得到$v$，实际上这个$v$即是函数$f(v)$的一个不动点

* 然而该如何求解呢？，收缩函数理论为我们提供了解法

**收缩函数**(Contraction mapping)
$$
\|f(x_1) - f(x_2)\| \leq \alpha \|x_1 - x_2\|
$$

* $\alpha \in (0, 1)$ must be strictly less than 1 

**收缩函数理论**

**定理（压缩映射定理）：**

对于任何形如 $x = f(x)$ 的方程，如果 $f$ 是一个压缩映射，则：

- **存在性：** 存在一个固定点 $x^*$，满足 $f(x^*) = x^*$。
- **唯一性：** 该固定点 $x^*$ 是唯一的。
- **算法：** 设定一个序列 $\{x_k\}$，其中 $x_{k+1} = f(x_k)$，则当 $k \to \infty$ 时，$x_k \to x^*$。

此外，收敛速率是指数级的。

## BOE(贝尔曼最优公式)解法

**贝尔曼最优性方程求解过程：**

* 对于任何状态 $s$，当前估计的值 $v_k(s)$（在初始时刻需要自己初始化一个值）
* 对于任何动作 $a \in \mathcal{A}(s)$，计算：
  $$
  q_k(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_k(s')
  $$
* 计算贪心策略 $\pi_{k+1}$ 对于 $s$：
  $$
  \pi_{k+1}(a|s) =  \begin{cases} 
  1 & \text{当 } a = a^*_k(s) \\
  0 & \text{当 } a \neq a^*_k(s)  
  \end{cases}
  $$
  其中 $a^*_k(s) = \arg\max_a q_k(s, a)$。
* 计算 $v_{k+1}(s) = \max_a q_k(s, a)$。（为确定策略概率为1，因此直接等于该行动对应的行动价值）

上述算法实际上就是**值迭代算法**，将在下一讲中详细讨论。

{% note info %}

解法最优性证明：

对于任何策略$\pi$，我们有以下公式：
$$
v_{\pi} = r_{\pi} + \gamma P_{\pi} v_{\pi}.
$$

由于

$$
v^{*} = \max_{\pi} \left( r_{\pi} + \gamma P_{\pi} v^{*} \right) = r_{\pi^*} + \gamma P_{\pi^*} v^{*} \geq r_{\pi} + \gamma P_{\pi} v^{*},
$$

其中$\ge$ 由$max$的定义而来

通过反复应用上述不等式，我们得到：
$$
v^{*} - v_{\pi} \geq \gamma P_{\pi} \left( v^{*} - v_{\pi} \right) 
\geq \gamma^2 P_{\pi}^2 \left( v^{*} - v_{\pi} \right)
\geq \cdots \geq \gamma^n P_{\pi}^n \left( v^{*} - v_{\pi} \right).
$$

由此可得：

$$
v^{*} - v_{\pi} \geq \lim_{n \to \infty} \gamma^n P_{\pi}^n \left( v^{*} - v_{\pi} \right) = 0,
$$

最后的等式成立是因为$\gamma < 1$，而且$P_{\pi}^n$是一个非负矩阵，其所有元素都不超过1（因为$P_{\pi}^n \mathbf{1} = \mathbf{1}$）。因此，对于任何策略$\pi$，我们有$v^{*} \geq v_{\pi}$。

{% endnote %}



## 影响BOE结果的因素

1. **奖励设计 (Reward Design)**:表示为 $r$

2. **系统模型 (System Model)**: 状态转移概率: $p(s'|s,a)$ 和奖励概率 $p(r|s,a)$

3. **折扣率 (Discount Rate)**:表示为 $\gamma$

* 如果仅仅是将$R  \to aR + b$, 对BOE结果并没有影响，因为BOE本质考虑的是相对奖励而不是绝对奖励
