---
title: Python
index_img: /img/banner/python.jpg
date: 2023-06-28 22:16:14
tags:
- python
categories:
- [技术学习,Python]
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

#### 布尔操作符

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

#### 成员操作符

* in：如果在指定的序列中找到值返回 True，否则返回 False
* not in：如果在指定的序列中没有找到值返回 True，否则返回 False。

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

* if
* elif
* else

```python
if 条件 :
	.....
elif 条件:
	.....
else 条件:
	....
```

### match语句

* 等价于c语言中的case

```python
match a:
    case 1:
        print("a=1")
	case 2:
        print("a=2")
    case _:
        print("a!=1&&a!=2")
    
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

**字符串**

```python
a='I love you'
```

**切片表达式**：通过下标访问

* $a[i:j]$ 若是从左往右，则表示从下标为i一直到j-1( 0-len-1）；若是从右往左，下标从-1开始
* $a[start:end:step]$ ：从start开始，步长为step取，直到end-1

**字典dict**
**字典的key可以是任何不可变的（也就是可以通过哈希每次得到的结果不变）对象**：如整数，字符串

但mutable tpye(可变类型)如列表list，set，dict都不能作为key

* 不可变对象：调用对象自身的任意方法，也不会改变该对象自身的内容。相反，这些方法会创建新的对象并返回，这样，就保证了不可变对象本身永远是不可变的。

```python
>>> a='abc'
>>>a.replace('a','A')
'Abc'
>>>a
'abc'
```

字符串a中的任何函数都是创建一个新的变量b使得b为更改后的值并且返回b，对应字符串a本身并未发生任何改变。a的地址没变

添加方式：

1.直接添加

```python
a['xiaoming']=0
```

2.变量添加

```python
x="hello"
y=5
a[x]=y #a["hello"]=5
```

内置函数

* dict.clear()删除字典内所有元素
* dict.get(key,default=None) 返回指定键的值，如果键不存在则返回default的默认值
* in成员运算符（not in），key $in$ dic:如果key在dic的键中，则返回True，否则返回False
* dic.pop(key):删除键值和对应的value

## 函数

规范：

```python
def FuntionName(arguement):
    return x,y
```

如果想要在函数内部给全局变量赋值，而不是新开一个变量，需要使用`global`语句告诉pyhton 该变量是全局变量中的a

* def关键字
* 函数名
* 参数名
* 返回值，可以返回多个值

举例：

```python
def change():
	global a
    a=9
a=10
change()
print(a)
```

> 输出结果应该为9

### 函数传参

python 函数的参数传递：

- **不可变类型：**类似 C++ 的值传递，如整数、字符串、元组。如 fun(a)，传递的只是 a 的值，没有影响 a 对象本身。如果在 fun(a) 内部修改 a 的值，则是新生成一个 a 的对象。
- **可变类型：**类似 C++ 的引用传递，如 列表，字典。如 fun(la)，则是将 la 真正的传过去，修改后 fun 外部的 la 也会受影响

python 中一切都是对象，严格意义我们不能说值传递还是引用传递，我们应该说传不可变对象和传可变对象。

### 函数内嵌

在函数中定义函数

```python
def A(arguement):
	......
	def B():
		return 
	
	return B
```

## 正则表达式

