---
title: Python
index_img: /img/banner/python.jpg
date: 2023-06-28 22:16:14
tags:
- python
categories:
- [技能学习,Python]
---

# Python学习

## Interactive Sessions

python可以交互式编程

* type some Python *code* after the *prompt*, `>>>`. The Python *interpreter* reads and executes what you type, carrying out your various commands.

## Expressions

表达式：包含操作符和值 

```python
>>2+2  #(这里的2+2就为一个表达式)
4
```

*  

### 操作符

* `**`表示指数
* %表示取模
* //表示整除
* /表示除法
* *乘法

### 布尔操作符

* and
* or
* not

```Python
>>True and False
True
>> True or False
False
>>not not not False #not可以嵌套
True
```

### 数据类型

* 整型
* 浮点型
* 字符串：单引号双引号均可
  * “a“+”b“=”ab“
  * ”abc“*2=”abcabc“
* 布尔值：True False（必须大写开头）
* 强转用三个函数
  * int()
  * float()
  * str()

### 变量名

* 只能是一个词不能包含空格
* 只能包含字母，数字和下划线
* 不能以数字开头

## 代码块

**Python是按照缩进来区分代码块的**

* 当缩进增加，代码块开始
* 当缩进变为0，代码块结束

## 控制流

### if语句

```python
if 条件 :
	.....
elif 条件:
	.....
else 条件:
	....
```

### while循环

```python
while 条件:
	......
```

* break和continue同样使用

### for循环

for+一个变量名+in +range(n)+冒号：

```python
for i in range(5): #i从0到4
```

* range(a):  [0,a)
* range(a,b): 

## 字符串，字典，列表

```python
a='I love you'
```

**切片表达式**：通过下标访问

* $a[i:j]$ 若是从左往右，则表示从下标为i一直到j-1( 0-len-1）；若是从右往左，下标从-1开始
* $a[i:j:step]$ ：从i开始，步长为step取，直到j-1
* 
