---
title: 二分法
index_img: /img/banner/algorithm.jpg
date: 2023-07-12 10:29:58
tags:
- algorithm
categories:
- [算法学习,二分]
---

# 核心思想

设答案所在范围为$[l,r]$ ,所求为t则题目一定满足：

$\begin{cases}
  & \text true & \text{ if } x\ge t \\\\
  & \text flase  & \text{ if } x<t
\end{cases}$

**即定义域左端保持一个性质，右端保持另外一个性质**

# 题型

* 求最大，最小平均值
* 求最大中位数
* 最大和的最小值min(max)

# check函数方法

* 贪心（前提能保证结果正确，一般不设置选法就可以用贪心）
* dp(要限制选法)
* 直接函数：如求解一元三次方程的解

# 求最大(最小)平均值

**题目描述：** [P1570 KC 喝咖啡 - 洛谷 ](https://www.luogu.com.cn/problem/P1570)

现在有$ n$ 种调料，这杯咖啡只可以加入其中的 $m$ 种（当然 KC 一定会加入 m 种，不会加入少于 m 种的调料）根据加入的调料不同，制成这杯咖啡要用的时间也不同，得到的咖啡的美味度也不同。

KC 在得知所有的 *n* 种调料后，作为曾经的化竞之神的他，马上就知道了所有调料消耗的时间 $c_{i}$以及调料的美味度 $v_{i}$。由于 KC 急着回去刷题，所以他想尽快喝到这杯咖啡，但他又想喝到美味的咖啡，所以他想出了一个办法，他要喝到$\frac{\sum v_{i}}{\sum c_{i}}$最大的咖啡，请你帮他变成解决

**思想：**

* 普适结论：设枚举的数为x，如果x小于等于最大平均值，那么一定存在一种方案使得$x\le \frac{\sum_{n}^{m} a_{i}}{m-n+1}$，即存在$\sum_{n}^{m} (a_{i}-x) \ge 0$ ,即$\sum_{n}^{m} (a_{i}-x)的最大值 \ge 0$ ,进一步贪心得到最大值再与0进行比较

* 得到结论：
  
    $$
  \begin{cases}
  & \exists \text{ scheme }st.  \sum_{n}^{m} (a_{i}-x) \ge 0 & \text{ if } x\le ans \\\\
  & \nexists \text{ scheme }st.\sum_{n}^{m} (a_{i}-x) \ge 0  & \text{ if } x>ans
  \end{cases}
  $$

而这道题中分母为 $\sum_{n}^{m}c_{i}$  ,我们转化为存在一个方案 $x\sum_{n}^{m}c_{i}-\sum_{n}^{m} a_{i}<0$ ，贪心得到最小值再与0比较判断

**代码：**

```c++
#include<bits/stdc++.h>
using namespace std;
int n,m;
double s[201];
const double eps=1e-7;
struct coffee
{
    double v;
    double c;
}co[201];
int judge(double x)
{
    for(int i=1;i<=n;i++ )
    {
      s[i]=x*co[i].c-co[i].v;
    }
    sort(s+1,s+1+n);//排序再贪心得到最小的前m个和
    double sum=0;
    for(int i=1;i<=m;i++)
    {
        sum+=s[i];
    }
    return sum<0;
}
int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        cin>>co[i].v;
    }
    for(int i=1;i<=n;i++)
    {
        cin>>co[i].c;
    }
    double l=0;
    double r=co[1].v/co[1].c;
    for(int i=1;i<=n;i++)
    {
        r=max(r,co[i].v/co[i].c);
    }
    while(r-l>eps)
    {
        double mid=(l+r)/2.0;
        if(judge(mid))
        {
            l=mid;
        }
        else
        {
            r=mid;
        }
    }
    printf("%.3lf",l);
    
}
```



# 求最大中位数

**题目描述：**[ABC：E - Average and Median ](https://atcoder.jp/contests/abc236/tasks/abc236_e)

*n* 个数排成一列。现在要选出一些数，满足 **任意两个相邻的数中至少有一个数被选择**。

请求出：

- 所有选择方案中，被选中的数字平均值的最大值，误差在 $10^{-3}$以内视为正确；
- 所有选择方案中，被选中的数字中位数的的最大值。在这里，偶数 2k个数的中位数视作第 k 小的数。



**思想：**

dp+二分

设枚举的数为x，预处理题目序列  $\begin{cases}
  & \text b[i]=1 & \text{ if } a[i]>x\\
  & \text b[i]=0  & \text{ if } a[i]=x\\
  & \text b[i]=-1  & \text{ if } a[i]<x\end{cases}$  ,使得小于x的都为-1，大于x都为1,等于x都为0。
如果当前数x小于等于最大中位数，则存在一种方案使得 $\sum_{n}^{m}b[i] >=0$ ，即$max(\sum_{n}^{m}b[i])\ge0$

得到结论

$\begin{cases}
  & \exists \text{ scheme }st.max(\sum_{n}^{m}b[i])\ge0 & \text{ if } x\le ans \\\\
  & \nexists \text{ scheme }st.max(\sum_{n}^{m}b[i])\ge0  & \text{ if } x>ans
\end{cases}$

* 处理平均数：每个数减去x，转化为求和大于等于0
* 处理中位数：分为大于等于小于x来设定贡献值

**代码：**

```c++
int judge_average(double x)
{
    for(int i=1;i<=n;i++)
    {
        s[i]=a[i]-x;
    }
    for(int i=1;i<=n;i++)
    {
        dp[i][0]=dp[i-1][1];
        dp[i][1]=max(dp[i-1][0],dp[i-1][1])+s[i];
    }
    if(max(dp[n][0],dp[n][1])>=0) return 1;
    else return 0;
}
int judge_mid(int x)
{
    for(int i=1;i<=n;i++)
    {
        if(a[i]>x)
        {
            s[i]=1;
        }
        else if(a[i]<x)
        {
            s[i]=-1;
        }
        else
        {
            s[i]=0;
        }
    }
    for(int i=1;i<=n;i++)
    {
        dp[i][0]=dp[i-1][1];
        dp[i][1]=max(dp[i-1][0],dp[i-1][1])+s[i];
    }
    if(max(dp[n][0],dp[n][1])>=0) return 1;
    else return 0;
}
```



# 求最大值的最小值max（min）

**关键**：每次枚举k，

* 求最大值：若$k<= condition$，则存在方案  , k继续变大
* 求最小值：若$k>=condition$，则存在方案，k继续变小



**题目描述：**[P1182 数列分段 Section II](https://www.luogu.com.cn/problem/P1182)

对于给定的一个长度为N的正整数数列 ，现要将其分成 M段，并要求每段连续，求每段和的最大值最小。

**思想：**

设枚举的每段和的最大值为x，若$x \ge ans$ ,则存在分割方式，若$x < ans$,则不存在切割方式。
check函数的话我们尽可能让每段的和都接近x，因为如果某一段不够接近x，那么这之后的某一段和肯定要变大。如果划分了k>=M段，那么说明这个x还不够大，x需要变大来使得k更靠近M。如果k<M,说明x太大，需要变小。

**代码**：

```cpp
int judge(int x)
{
    int num=0;
    int k=1;
    int sum=0;
    while(k<=n)
    {
        if(sum+a[k]<=x)
        {
            sum+=a[k];
        }
        else
        {
            num+=1;
            sum=a[k];
        }
        k++;
    }
    num+=1;
    if(num<=m)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
```



**题目描述**：[Atcoder ABC 215 F - Dist Max 2 ](https://atcoder.jp/contests/abc215/tasks/abc215_f)

给定n个二维坐标点，规定两两坐标的距离$d=min(|x_{i}-x_{j}|,|{y_{i}-y_{j}|})$,求这些d中的最大值

**思路**:

对于max(min)问题，要首先想到二分法。针对这道题目。设答案为ans，则这些点中一定存在$min(|x_{i}-x_{j}|,|{y_{i}-y_{j}|})\ge ans$ 即 $|x_{i}-x_{j}|{\ge} ans 且|y_{i}-y_{j}|\ge ans$  

如果想要在$O(n)$的时间复杂度内判断，则我们需要对每个i来判断。不妨假设$x_{i}$ 固定不动，分析针对这个点需要满足什么条件才能判断ans。
首先对于$x_{i}$,不需要考虑$[x_{i}-ans+1,x_{i}+ans-1]$ 的位置，因为这些距离$x_{i}$已经小于ans，肯定不存在$min(|x_{i}-x_{j}|,|{y_{i}-y_{j}|})\ge ans$ 的情况。针对于需要考虑的j位置，因为$|x_{i}-x_{j}|已经满足\ge ans$,所以只要这些位置中有一个点的$|y_{i}-y_{j}|满足\ge ans$即可，而这可以通过$x_{i}-ans$及其以前的对应y的最大值和最小值来判断。又因为对称性，$x_{i}+ans=x_{j}$ 等价于$x_{j}-ans=x_{i}$,所以其实只需将x从小到大遍历，每次判断$0到x_{i}-ans$ 的范围对应的y的最大最小值即可，这里可以用**滑动窗口**的思想。

**代码**：

```c++
#include<bits/stdc++.h>
using namespace std;
#define N 200001
int n;
struct node
{
    int x;
    int y;
    bool operator < (const node &a)
    {
        return this->x < a.x;
    }
}a[N];
bool check(int k)
{ 
    queue<node>q;
    int min_y=1e+9+1;
    int max_y=0;
    for(int i=1;i<=n;i++)
    {
       while(!q.empty()&&q.front().x<=a[i].x-k) //队列用于存储x坐标与当前判断的x[i]距离大于k的点对应的y值
       {
            min_y=min(min_y,q.front().y);
            max_y=max(max_y,q.front().y);
            q.pop();
       }
       if(a[i].y>=min_y+k||a[i].y<=max_y-k)
       {
            return true;
       }
       q.push(a[i]);

    }
    return false;

}   
int main()
{
    cin>>n;
    int min_x=1e+9+1;
    int max_x=0;
    int min_y=1e+9+1;
    int max_y=0;
    for(int i=1;i<=n;i++)
    {
        cin>>a[i].x>>a[i].y;
        min_x=min(min_x,a[i].x);
        max_x=max(max_x,a[i].x);
        min_y=min(min_y,a[i].y);
        max_y=max(max_y,a[i].y);
    }
    sort(a+1,a+1+n);
    int r=min(max_x-min_x,max_y-min_y);
    int l=0;
    while(l<r)
    {
        int mid=(l+r+1)>>1;
        if(check(mid))
        {
            l=mid;
        }
        else
        {
            r=mid-1;
        }
    }
    cout<<l<<endl;
    

}
```

# 寻找第k大数

思路：二分答案
[CF448D](https://codeforces.com/problemset/problem/448/D)

```c++
#include<bits/stdc++.h>
using namespace std;
long long  n,m,k;
bool check(long long x)
{
    long long cnt=0;
    for(int i=1;i<=n;i++)
    {
        long long y=min(i*m,x);
        cnt+=y/i;
    }
    return cnt>=k;
}
int main()
{
    cin>>n>>m>>k;
    long long l=0,r=n*m;
    while(l<r)
    {
        long long mid=(l+r)>>1;
        if(check(mid))
        {
            r=mid;
        }
        else
        {
            l=mid+1;
        }
    }
    cout<<l<<endl;
}
```

