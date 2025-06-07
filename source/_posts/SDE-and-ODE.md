---
layout: pages
title: SDE and ODE
index_img: /img/banner/sde_ode.png 
date: 2025-05-19 14:43:17
tags:
---

## SDE

在连续的时间域中,  扩散模型的 **前向过程** 可以通过 **随机微分方程(stochastic differential equation)** 来描述：
$$
\begin{equation}
dx(t) = f(t)x(t)dt + g(t)dw
\end{equation}
$$

* $f(t)$ 被称为漂移系数，表明系统确定性演化趋势
* $g(t)$ 被称为扩散系数， 表明随机噪声的强度
* $w$ 被称为维纳过程（Wiener Process，即布朗运动），是随机噪声的来源。 $dw∼N(0,dt)$

其 **反转过程(去噪过程)** 也可以用另一种形式的随机微分方程来描述：
$$
\begin{align}
dx(t) &= \left[ f(t)x(t) - g^2(t) \nabla_{x(t)} \log p_t(x(t)) \right] dt + g(t) dw \\
&= \left[ f(t)x(t) + \frac{g^2(t)}{\sqrt{1 - \bar{\alpha}_t}} \hat{\varepsilon}_\theta(x(t), t) \right] dt + g(t) dw \\
\end{align}
$$

* 其中 $x(T) \sim \mathcal{N}(0, I)$

### Wiener Process

**Wiener Process (Brownian Motion)** 是一个具有以下4个关键性质的一维连续时间随机过程：

1. 初始条件 $w(0) = 0$, 几乎必然，即概率为1
2. 正态增量：对于任意时间 $0 \leq s< t$ ，增量 $w(t) - w(s) \sim N(0, t-s)$
3. 独立增量：对于任意时间 $0 \leq s< t$ ，增量 $w(t) - w(s)$ 对于过去的时间独立
4. 连续性：w(t)是几乎必然连续的, 即概率为1



## Variational Diffusion Models

展示diffusion models一种通用的形式：
$$
q(x_t|x_0) = \mathcal N(x_t;\alpha(t)x_0, \sigma^2(t)I)
$$

* 其中 $\alpha(t)$ 和 $\sigma^2(t)$ 的选取需要满足信噪比(Signal-to-Noise Ratio) $SNR = \frac{\alpha^2(t)}{\sigma^2(t)}$ 随着t增大持续降低

## SDE 与 Diffusion model的转化

SDE：
$$
dx(t) = f(t)x(t)dt + g(t)dw
$$
前向传播的概率分布：
$$
q(x_t|x_0) = \mathcal N(x_t;\alpha(t)x_0, \sigma^2(t)I)
$$
根据SDE可以推导出二者之间的关系：

从SDE到Diffuson model：

* $\alpha(t) = e^{\int_0^t f(s)ds}$

* $\sigma^2(t) = \int_0^t \frac{g(s)^2}{\alpha(s)^2}ds$



从Diffusion model到SDE：

* $f(t)= \frac{d\log \alpha(t)}{dt} = \frac{d\log \alpha_t}{dt}$
* $g^2(t) = \frac{d\sigma^2(t)}{dt} - 2\frac{d\log \alpha(t)}{dt}\sigma^2(t) = \frac{d\sigma^2_t}{dt} - 2\frac{d\log \alpha_t}{dt}\sigma^2_t$

$$
\begin{align*}
g^2(t) &= \frac{d\sigma^2_t}{dt} - 2\frac{d\log \alpha_t}{dt}\sigma^2_t \\
&= 2\sigma_t (\sigma_t \frac{d \log \sigma_t}{dt}) - 2\frac{d\log \alpha_t}{dt}\sigma^2_t \\
&= 2\sigma_t^2(\frac{d \log \sigma_t}{dt} - \frac{d\log \alpha_t}{dt}) \\
&= 2\sigma_t^2( - \frac{d \lambda_t}{dt})
\end{align*}
$$

* 其中 $\lambda_t = \log \frac{\alpha_t}{\sigma_t}$ 随t增大而逐渐减小

## Probability Flow ODE

给定一个反向传播的SDE,
$$
dx(t) = \left[ f(t)x(t) + \frac{g^2(t)}{\sqrt{1 - \bar{\alpha}_t}} \hat{\varepsilon}_\theta(x(t), t) \right] dt + g(t) dw
$$
可以确定一个反向去噪的ODE：
$$
\begin{equation}
\frac{dx(t)}{dt} = f(t)x(t) + \frac{g^2(t)}{2\sigma(t)}\hat\epsilon_{\theta}(x(t),t)
\end{equation}
$$
* 概率流ODE保证在时间演化过程中，每个时刻 t 的样本分布 $p(x_t)$，与原始扩散过程SDE在相同时刻的分布 $q(x_t)$ **完全一致**
* 概率流ODE通过确定性的动力学方程描述样本演化过程，避免了SDE中的随机噪声项(SDE因为维纳过程，转化为离散形式计算时，一旦步长变大会出现很大误差)，**更适合采用大步长以加速采样**

## DPM-Solver

### Step1: 一阶半线性微分方程

针对以下ODE：
$$
\frac{dx(t)}{dt} = f(t)x(t) + \frac{g^2(t)}{2\sigma(t)}\hat\epsilon_{\theta}(x(t),t)
$$

* $f(t)x(t)$ 是关于 $x$ 的线性
* $\frac{g^2(t)}{2\sigma(t)}\hat\epsilon_{\theta}(x(t),t)$ 是用神经网络的非线性

该结构就是的一阶半线性微分方程(First-Order Semlinear Differential Equation)

{% note success %}

回忆一阶半线性微分方程的解法：
$$
x'(t) = p(t)x(t) + q(x_t,t)
$$

* 线性是指 $p(t)x(t)$ 对于 $x$ 是线性函数
* 非线性是指 $q(x_t,t)$ 可能针对 $x$ 不是线性的
* 方程是相对于t求导，$x$ 是关于 $t$ 的一个函数

考虑特殊情况 $\color{red}{q(x_t,t)}$
$$
x'(t) = p(t)x(t)
$$
对方程进行转化：
$$
\frac{x'(t)}{x(t)} = \frac{\ln x(t)}{dt} =p(t)
$$
将 $d(t)$ 移动到等式左边，再积分(注意积分变元要和 $t$ 区分)得：
$$
\ln x(t) - \ln x(s) = \int_{s}^{t} p(z)dz
$$
得出：
$$
x(t)  = e^{\ln x(s)}\cdot e^{\int_{s}^{t} p(z)dz} = C\cdot e^{\int_{s}^{t} p(z)dz}
$$


再回过头来审视 $\color{red}{一般情况}$：
$$
x'(t) = p(t)x(t) + q(t)
$$
方程两边同时乘上 $e^{-\int_{t_0}^{t} p(z)dz}$ :
$$
e^{-\int_{t_0}^{t} p(z)dz}[x'(t) - p(t)x(t)] = e^{-\int_{t_0}^{t} p(z)dz}q(t)
$$

* 注意到：  $\frac{d}{dt} \left( f(t)g(t) \right) = f'(t)g(t) + f(t)g'(t)$ 且 $\frac{de^{-\int_{t_0}^{t} p(z)dz}}{dt} = e^{-\int_{t_0}^{t} p(z)dz}\cdot (-p(t))$

所以：
$$
d\textcolor{red}{[e^{-\int_{t_0}^{t} p(z)dz}\cdot x(t)]} =  e^{-\int_{t_0}^{t} p(z)dz} q(t)
$$
由：
$$
\frac{d}{dt}\left(p(t)\right) = q(s) \quad \Rightarrow \quad p(t) = p(s) + \int_{s}^{t} q(\tau) \, d\tau
$$
得到：
$$
e^{-\int_{t_0}^{t} p(z)dz}\cdot x(t) = e^{-\int_{t_0}^{s} p(z)dz}\cdot x(s) + \int_{s}^{t} e^{-\int_{t_0}^{t} p(z)dz} q(\tau) d\tau
$$


化简得到：
$$
x(t) = e^{\int_s^t p(z)dz}x(s) + \int_{s}^{t} e^{-\int_{\tau}^{t} p(z)dz} q(\tau) d\tau
$$
{% endnote %}

将 $p(t) = f(t) = \frac{d\log \alpha_t}{dt}$, $q(t) = \frac{g^2(t)}{2\sigma(t)} = \sigma_t( - \frac{d \lambda_t}{dt})\hat\epsilon_{\theta}(x(t),t)$ 带入
$$
x(t) = \frac{\alpha_t}{\alpha_s}x(s) - \alpha_t \int_{s}^{t}(\frac{d\lambda_\tau}{d\tau}e^{-\lambda_\tau}\hat\epsilon_{\theta}(x(\tau), \tau))d\tau
$$

* $\lambda_{\tau} = \log \frac{\alpha_{\tau}}{\sigma_{\tau}}$

### Step2: 积分变量换元

由于 $\lambda_\tau = \lambda(\tau)$ 单调递减，存在逆函数 $t_{\lambda}(\lambda_t) = t$, $\lambda_t$ 和 $t$ 之间可以一一对应

对上式进行积分变量替换：
$$
\begin{align*}
x(t) &= \frac{\alpha_t}{\alpha_s}x(s) - \alpha_t \int_{\lambda_s}^{\lambda_t}(\frac{d\lambda_\tau}{d\tau}e^{-\lambda_\tau}\hat\epsilon_{\theta}(x(\lambda_\tau), \lambda_\tau))\frac{d\tau}{d\lambda_{\tau}}d\lambda_{\tau} \\
&= \frac{\alpha_t}{\alpha_s}x(s) - \alpha_t \int_{\lambda_s}^{\lambda_t}(e^{-\lambda_\tau}\hat\epsilon_{\theta}(x(\lambda_\tau), \lambda_\tau))d\lambda_\tau 
\end{align*}
$$
离散形式，$s=t+1$
$$
\begin{align}
x(t) &= \frac{\alpha_t}{\alpha_{t+1}}x(t+1) - \alpha_t \int_{\lambda_{t+1}}^{\lambda_t}(e^{-\lambda_\tau}\hat\epsilon_{\theta}(x(\lambda_\tau), \lambda_\tau))d\lambda_\tau 
\end{align}
$$

### Step3: 解析计算

作者进一步对 $\hat\epsilon_{\theta}(x(\lambda_\tau), \lambda_\tau))$ 进行泰勒展开
$$
\hat{\epsilon}_{\theta}(\hat{\mathbf{x}}_{\lambda}, \lambda) = \sum_{n=0}^{k-1} \frac{(\lambda - \lambda_{t+1})^n}{n!} \hat{\epsilon}_{\theta}^{(n)} (\hat{\mathbf{x}}_{\lambda_{t+1}}, \lambda_{t+1}) + O((\lambda - \lambda_{t+1})^k)
$$
带入得：
$$
\begin{align}
\mathbf{x}_{t} = \frac{\alpha_{t}}{\alpha_{t+1}} \mathbf{x}_{t+1} - \alpha_{t} \sum_{n=0}^{k-1} \underbrace{\hat{\epsilon}_{\theta}^{(n)} (\hat{\mathbf{x}}_{\lambda_{t+1}}, \lambda_{t+1})}_{\text{derivatives}} \underbrace{\int_{\lambda_{t+1}}^{\lambda_{t}} e^{-\lambda} \frac{(\lambda - \lambda_{t+1})^n}{n!} d\lambda}_{coefficients} + O(h_i^{k+1})
\end{align}
$$

* 其中$h_{t}= \lambda_t - \lambda_{t+1}$

系数 $C_n$ 的解析计算过程如下：
$$
\begin{align*}
C_n &= \int_{\lambda_{t+1}}^{\lambda_{t}} e^{-\lambda} \frac{(\lambda - \lambda_{t+1})^n}{n!} d\lambda \\
&= -\int_{\lambda_{t+1}}^{\lambda_{t}} \frac{(\lambda - \lambda_{t+1})^n}{n!} de^{-\lambda} \\
&= \left( -\frac{(\lambda - \lambda_{t+1})^n}{n!} e^{-\lambda} \right) \bigg|_{\lambda_{t+1}}^{\lambda_{t}} + \int_{\lambda_{t+1}}^{\lambda_{t}} e^{-\lambda} \frac{(\lambda - \lambda_{t+1})^{n-1}}{(n-1)!} d\lambda \\
&= -\frac{h_i^n}{n!} e^{-\lambda_{t}} + C_{n-1}
\end{align*}
$$


基础项 $C_0$ 的计算为：
$$
\begin{equation}
C_0 = \int_{\lambda_{t+1}}^{\lambda_{t}} e^{-\lambda} d\lambda = e^{-\lambda_{t+1}} - e^{-\lambda_{t}} = \frac{\sigma_{t}}{\alpha_{t}} (e^{h_t} - 1)
\end{equation}
$$


高阶系数的递推结果：
$$
\begin{align*}
C_1 &= e^{-\lambda_{t+1}} - (1 + h_t) e^{-\lambda_{t}} = \frac{\sigma_{t}}{\alpha_{t}} \left( e^{h_t} - 1 - h_t \right)  \\
C_2 &= e^{-\lambda_{t+1}} - \left( 1 + h_t + \frac{h_t^2}{2} \right) e^{-\lambda_{t}} = \frac{\sigma_{t}}{\alpha_{t}} \left( e^{h_t} - 1 - h_t - \frac{h_t^2}{2} \right) 
\end{align*}
$$


### DPM-Solver-1

令k = 1
$$
\begin{align*}
\mathbf{x}_{t} &= \frac{\alpha_{t}}{\alpha_{t+1}} \mathbf{x}_{t+1} - \alpha_{t}\hat{\epsilon}_{\theta}(\hat{\mathbf{x}}_{\lambda}, \lambda)C_0  + O(h_i^{2}) \\
&= \frac{\alpha_{t}}{\alpha_{t+1}} \mathbf{x}_{t+1} - \sigma_t(e^{h_t} - 1)\hat{\epsilon}_{\theta}(\hat{\mathbf{x}}_{\lambda}, \lambda) \\
& = \frac{\alpha_{t}}{\alpha_{t+1}} \mathbf{x}_{t+1} - \sigma_t(e^{h_t} - 1){\epsilon}_{\theta}({\mathbf{x}}_{t}, t) \\
& = \frac{\alpha_{t}}{\alpha_{t+1}} \mathbf{x}_{t+1} - (\alpha_t \frac{\sigma_{t+1}}{\alpha_{t+1}}-\sigma_{t+1}){\epsilon}_{\theta}({\mathbf{x}}_{t}, t) = DDIM
\end{align*}
$$

### Step4：数值估计导数项

对于更高阶的， 我们需要求解 $\hat{\epsilon}_{\theta}^{(n)} (\hat{\mathbf{x}}_{\lambda_{t+1}}, \lambda_{t+1})$ , 这一项作者进一步采用数值方式估计，具体略有复杂，感兴趣的可以参照原论文

![](/img/diffusion/dpm_solver1.png)

![](/img/diffusion/dpm_solver2.png)

![](/img/diffusion/dpm_solver3.png)
