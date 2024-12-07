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

- 用encoder原数据压缩，压缩后的特征可视作隐式变量z，之后再用decoder还原

![Auto Encoder](/img/VAE/AE.png)

AE的缺点：

* AE压缩后的特征z是离散的（可以视作$\{z_1, z_2, ...z_n\}$）其能表示的空间有限，例如$z_1 = [1,20,0.5,19]$,如果这里第一维改成0.5，如果生成器在训练的时候没有见过，则生成出来的效果可能不佳。

* 解释说明： $P(X) = \sum_z P(X|z)P(z)$

​	![P(x|m)即为图中P(x)中蓝色部分，P(m)则是多项式分布是离散的。](/img/VAE/AE_px.png)

- 基于该**离散**的方式所能生成的$P(X)$能力有限

ok，想到这里，要想获得好的**生成**, 我们想尽可能地扩大隐式变量z的空间，怎么做呢？

VAE:

- 与其让神经网络生成基于样本x对应的特征z（一个向量），我们不如让神经网络学习基于样本x的**隐式z的分布**（一个分布）不就好了嘛

![difference](/img/VAE/difference.png)

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

![explaination](/img/VAE/gpt.png)

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

![reconstruction](/img/VAE/reconstruct.png)

将高斯分布带入化简可得MSE：
$$
L2(x, u_{\theta}(z))
$$


- 正则项：

![regularization](/img/VAE/regular.png)

{% note success %}

两个高斯分布的KL散度公式推导：

注意:对于连续变量的最大似然估计时，我们采用的是概率密度而不是概率

假设$P \sim N(u_p, \sigma_p), Q\sim N(u_q, \sigma_q)$
$$
\begin{align*}
D_{KL}(P || Q) &= \int_z p(z) \log \frac{\frac{1}{\sqrt{2\pi \sigma_p^2}} e^{-\frac{(z-\mu_p)^2}{2\sigma_p^2}}}{\frac{1}{\sqrt{2\pi \sigma_q^2}} e^{-\frac{(z-\mu_q)^2}{2\sigma_q^2}}} dz \\
&= \int_z p(z) \left[ \frac{1}{2} \log\left(\frac{\sigma_q^2}{\sigma_p^2}\right) 
- \frac{(z-\mu_p)^2}{2\sigma_p^2} + \frac{(z-\mu_q)^2}{2\sigma_q^2} \right] dz.
\end{align*}
$$
逐项计算：

1. 第一项：
   $$
   \int_z p(z) \frac{1}{2} \log\left(\frac{\sigma_q^2}{\sigma_p^2}\right) dz 
   = \frac{1}{2} \log\left(\frac{\sigma_q^2}{\sigma_p^2}\right)
   $$

2. 第二项：
   $$
   \begin{align*}
   \int_z p(z) \frac{(z-\mu_p)^2}{2\sigma_p^2} dz 
   &= \frac{1}{2\sigma_p^2} \int_z p(z) (z-\mu_p)^2 dz \\
   &= \frac{1}{2\sigma_p^2} \operatorname{Var}(P) = \frac{1}{2}.
   \end{align*}
   $$


3. 第三项：
   $$
   \begin{align*}
   \int_z p(z) \frac{(z-\mu_q)^2}{2\sigma_q^2} dz 
   &= \frac{1}{2\sigma_q^2} \int_z p(z) \left[z^2 - 2\mu_q z + \mu_q^2 \right] dz \\
   &= \frac{1}{2\sigma_q^2} \left[\sigma_p^2 + \mu_p^2 - 2\mu_q \mu_p + \mu_q^2 \right].
   \end{align*}
   $$

综上：
$$
D_{KL}(P || Q) = \frac{1}{2} \left[ \log\left(\frac{\sigma_q^2}{\sigma_p^2}\right)  + \frac{\sigma_p^2}{\sigma_q^2}  + \frac{(\mu_p - \mu_q)^2}{\sigma_q^2} - 1 \right]
$$


如果P,Q服从多维高斯分布
$$
\begin{align*}
D_{\text{KL}}(P \| Q) = 
\frac{1}{2} \Big[ 
\log \frac{\det \Sigma_p}{\det \Sigma_q} 
+ \operatorname{tr}(\Sigma_q^{-1} \Sigma_p) 
+ (\mu_p - \mu_q)^\top \Sigma_q^{-1} (\mu_p - \mu_q)
- d
\Big]
\end{align*}
$$


{% endnote %}

将$P \sim N(u_{\phi}, \sigma_{\phi})$, $Q \sim N(0,I)$代入

* 在实际VAE实现中简化为不同维度是独立的，故协方差矩阵只有对角线上存在值。公式简化为d个一维高斯分布的散度之和

$$
D_{KL}(q_{\phi}(z|x)|| N(0,I)) = \sum^d_{i=1} \frac{1}{2}[log(\sigma_{\phi_i})^2 + \sigma_{\phi_i}^2 + u_{\phi_i}^2 -1]
$$



```python
class VAE(torch.nn.Module):
  def __init__(self, input_dim, hidden_dims, decode_dim=-1, use_sigmoid=True):
      '''
      input_dim: The dimensionality of the input data.
      hidden_dims: A list of hidden dimensions for the layers of the encoder and decoder.
      decode_dim: (Optional) Specifies the dimensions to decode, if different from input_dim.
      '''
      super().__init__()

      self.z_size = hidden_dims[-1] // 2

      encoder_layers = []
      decoder_layers = []
      counts = defaultdict(int)

      def add_encoder_layer(name: str, layer: torch.nn.Module) -> None:
        encoder_layers.append((f"{name}{counts[name]}", layer))
        counts[name] += 1
      def add_decoder_layer(name: str, layer: torch.nn.Module) -> None:
        decoder_layers.append((f"{name}{counts[name]}", layer))
        counts[name] += 1
      input_channel = input_dim
      encoder_dims = hidden_dims
      for x in hidden_dims:
        add_encoder_layer("mlp", torch.nn.Linear(input_channel, x))
        add_encoder_layer("relu",  torch.nn.LeakyReLU())
        input_channel = x

      decoder_dims = encoder_dims[::-1]
      input_channel = self.z_size
      for x in decoder_dims:
        add_decoder_layer("mlp", torch.nn.Linear(input_channel, x))
        add_decoder_layer("relu",  torch.nn.LeakyReLU())
        input_channel = x
      self.fc_mean = torch.nn.Sequential(
            torch.nn.Linear(encoder_dims[-1], self.z_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.z_size, self.z_size),
        )
      self.fc_var = torch.nn.Sequential(
            torch.nn.Linear(encoder_dims[-1], self.z_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.z_size, self.z_size),
        )
      self.encoder = torch.nn.Sequential(OrderedDict(encoder_layers))
      self.decoder = torch.nn.Sequential(OrderedDict(decoder_layers))
      self.out_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(decoder_dims[-1], input_dim),
            torch.nn.Tanh(),
      )

      ##################
      ### Problem 2(b): finish the implementation for encoder and decoder
      ##################

  def encode(self, x):
      res = self.encoder(x)
      mean = self.fc_mean(res)
      logvar = self.fc_var(res)
      return mean, logvar

  def reparameterize(self, mean, logvar, n_samples_per_z=1):
      ##################
      ### Problem 2(c): finish the implementation for reparameterization
      ##################
      d, latent_dim = mean.size()
      device = mean.device  # This will ensure the device of 'mean' is used for others

      # Ensure all tensors are on the same device
      std = torch.exp(0.5 * logvar).to(device)  # Move std to the same device as 'mean'
      epsilon = torch.randn(d, latent_dim, device=device)  # Move epsilon to the same device

      z = mean + std * epsilon  # Apply the reparameterization trick

      return z


  def decode(self, z):
      probs = self.decoder(z)
      out = self.out_layer(probs)
      return out

  def forward(self, x, n_samples_per_z=1):
      mean, logvar = self.encode(x)

      batch_size, latent_dim = mean.shape
      if n_samples_per_z > 1:
        mean = mean.unsqueeze(1).expand(batch_size, n_samples_per_z, latent_dim)
        logvar = logvar.unsqueeze(1).expand(batch_size, n_samples_per_z, latent_dim)

        mean = mean.contiguous().view(batch_size * n_samples_per_z, latent_dim)
        logvar = logvar.contiguous().view(batch_size * n_samples_per_z, latent_dim)

      z = self.reparameterize(mean, logvar, n_samples_per_z)
      x_probs = self.decode(z)

      x_probs = x_probs.reshape(batch_size, n_samples_per_z, -1)
      x_probs = torch.mean(x_probs, dim=[1])

      return {
          "imgs": x_probs,
          "z": z,
          "mean": mean,
          "logvar": logvar
      }

### Test
hidden_dims = [128, 64, 36, 18, 18]
input_dim = 256
test_tensor = torch.randn([1, input_dim]).to(device)

vae_test = VAE(input_dim, hidden_dims).to(device)

with torch.no_grad():
  test_out = vae_test(test_tensor)
  print(test_out)

```

重构损失就是MSE在此略，正则损失则为：

```python
##### Loss 2: KL w/o Estimation #####
def loss_KL_wo_E(output):
    var = torch.exp(output['logvar'])
    logvar = output['logvar']
    mean = output['mean']

    return -0.5 * torch.sum(torch.pow(mean, 2)
                            + var - 1.0 - logvar,
                            dim=[1])

```

