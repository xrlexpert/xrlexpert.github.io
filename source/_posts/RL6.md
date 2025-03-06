---
layout: pages
title: RL6 时序差分方法
date: 2025-03-04 14:13:16
index_img: /img/banner/rl.png 
tags:
---

## 时序差分方法(TD)

* 相较于蒙特卡洛采样一整个episode后才对策略进行更新，时序差分每执行一步即可更新值函数

朴素的时序差分方法：仅依靠数据而非模型来估计在给定policy的状态价值，相当于从数据的角度求解贝尔曼最优公式
$$
\begin{align}
v_{t+1}(s_t) &= v_t(s_t) - \alpha_t(s_t) \left[ v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1})) \right], \tag{1} \\
v_{t+1}(s) &= v_t(s), \quad \text{for all } s \neq s_t, \tag{2}
\end{align}
$$

* $s_t$ 是agent在 $t$ 时刻执行到的状态 

* $r_{t+1} + \gamma v_t(s_{t+1})$ 为TD Target，因为TD算法本质是希望$v(s_t)$ 朝 $\bar{v} = r_{t+1} + \gamma v(s_{t+ 1})$ 去逼近
* $v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1}))$ 为TD Error, 当策略为最优策略$\pi$ 时， $\delta_t = v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1})) = 0$

