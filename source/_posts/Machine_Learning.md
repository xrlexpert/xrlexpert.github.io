---
index_img: /img/banner/machine_learning.jpg 
title: Machine Learning-一元线性回归
date: 2023-12-02 23:43:42
tags:
---

# 监督学习（supervised learning)

给定一个input x，给出x的正确答案，即标签 y。机器通过这些大量的例子训练学习后使得遇到一个崭新的x时，能够辨别出x对应的y是多少

![](/img/machine_learning/regression_exap.png)

专业术语：

**training set**：训练集
$x$:input variable  or input feature
$y$:output variable or target variable
$(x,y)$:single training example
$(x^{i},y^{i})$:$i^{th}$  training example
$m$：训练集数据大小

## 回归问题（Regression）

给定x，预测x对应的数字y(y的取值为无穷)

### 一元线性回归

$f(x)=wx+b$
$w.b$:coeffient,weights

**损失函数**：$J(w,b)= \frac{1}{2m} \sum _{1}^{m} (\hat{y}-y)^{2}$
![](/img/machine_learning/visual_cost.png)

$goal:minize(J(w,b))$

### 梯度下降法

#### 前置知识

首先需要复习一下方向导数和梯度的概念：
**方向导数**：若函数$z=f(x,y)$在$P(x,y)$ 某一邻域有定义，自P点沿周围360度任一方向引出有向直线L，定义方向导数为

$$ \frac{\alpha z}{\alpha l} |_{(x,y)} = \lim_{\rho \to 0} \frac{\Delta z}{\Delta 自变量}= \lim_{\rho \to 0} \frac{f(x+\Delta x,y+\Delta y)-f(x,y)}{\sqrt{x^{2}+y^{2}}} (\rho =\sqrt{x^{2}+y^{2}})$$  

特别地，当$z=f(x,y)$在$P(x,y)$ 可微分，则$z=f(x,y)$沿任一方向的方向导数都存在，且

$$\frac{\alpha z}{\alpha l} |_{(x,y)} = \frac{\alpha z}{\alpha x}cos{\alpha}+\frac{\alpha z}{\alpha y}cos{\beta} =\{\frac{\alpha z}{\alpha x},\frac{\alpha z}{\alpha y} \}\cdot \{cos{\alpha},cos{\beta}\}$$

故当方向$\vec {l}=\{cos{\alpha},cos{\beta}\}$ 和 $\{\frac{\alpha z}{\alpha x},\frac{\alpha z}{\alpha y}\}$ 同向时，方向导数最大，函数沿此方向增长得最快,
**梯度**：$gard f=\{\frac{\alpha z}{\alpha x},\frac{\alpha z}{\alpha y} \}$，表示函数在该点增长最快的方向

#### 算法描述

梯度下降法即每次迭代，都使得$(J(w,b))$中的参数沿着梯度的反方向移动，使得函数下降得最快。
$w=w-\alpha \cdot \frac{\alpha J}{\alpha w}$
$b=b-\alpha \cdot \frac{\alpha J}{\alpha b}$
其中$\alpha$ 称为学习率，$\in (0,1)$

* 如果$\alpha$太小， 收敛速度很慢
* 如果$\alpha$ 太大，最终可能在min附近反复横跳，永远达不到最小值

**但对于固定的学习率来讲梯度下降法得到的函数最小值和初值有关，每次得到是初值附近的局部最优解**
**然而对于一元线性回归来讲，其损失函数是一个凹函数（convex function）,其极小值点只有一个，故梯度下降法在此适用**

## 分类问题(Classfication)

给定x,预测x对应的类别y（取值范围为种类数）

# 无监督学习（unsupervised learning）

find things in unlabled data

the data comes only with inputs x but not output labels y, and the algorithm has to find some structure or some pattern or something interesting in the data

## 聚类问题

## 异常检测

## 降维
