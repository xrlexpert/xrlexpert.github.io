---
title: Solidity学习
date: 2023-04-08 18:57:51
tags:
- solidity
categories:
- 大一立项
index_img: /img/blockchain.jpg
---

# Solidity学习

由于大一立项的要求简洁的记录一下方便复习

<!--more-->

## 数据类型

* 数值类型：调用时按值传递
  * uint 无符号正整数(uint256的别名)
  * int  有符号整数(int256的别名)
  * int8 - int256(8到256位)
  * uint8 - uint256 (8到256位)
  * address 调用合约地址
    * `address`：保存一个20字节的值（以太坊地址的大小）。
    * `address payable` ：可支付地址，与 `address` 相同，不过有成员函数 `transfer` 和 `send`
      * ap.transfer(10) 表示合约向ap转账10wei
  * bool 布尔类型
* 引用类型：调用时按地址传递
  * 数组
  * 结构体
  * 函数
  * 映射

## 合约

个人理解相当于C++中的类

```solidity
// SPDX-License-Identifier: MIT  /*注释用来注明代码的软件许可*/
pragma solidity ^0.8.4;  //声明solidity的版本
contract HelloWeb3{
	struct Student
	{
		int data;
		string id;
	}
    string public _string = "Hello Web3!";  //合约的状态变量 相当于类的成员
    uint balance=0;
    uint []x=[1,2,3];
    function A()
    function B()
}
```

**引用类型**在函数中都需要一个额外的标注来表明数据存储的位置：

* storage：存储在链上。合约的状态变量都默认存储在链上。
* memory：函数内的临时变量
* calldata：相当于c++的const   函数参数的临时变量，且传进来不能被改变

注意：如果在函数内创建一个storage变量指向合约的状态变量（必须是引用类型），那么改变该变量的同时会改变合约的状态变量
是指针还是复本：$storage>memory>calldata$ 当被赋值的类型 ==赋值的类型时，才是指针

```solidity
function()public pure 
{
	uint []storage x_1=x;
	x_1[1]=1;//此时balance也会变成1
}
```



## 函数

```solidity
function <function name> (<parameter types>) {internal|external|public|private}   [pure|view|payable] [returns (<return types>)]

function returnName () external pure returns(uint index,bool _bool)
{
	return(12,true);
}
```



## 映射

```solidity
mapping(uint=>student)a; 
```

key只能为数值类型
value可以是任何类型·



## 构造函数与修饰器

构造函数

```solidity
constructor(参数)  //用来对状态变量进行初始化
{
	.....
}

```

修饰器（`modifier`）是`solidity`特有的语法，类似于面向对象编程中的`decorator`，声明函数拥有的特性
相当于执行函数前需要判断的条件

```solidity
   // 定义modifier
   modifier onlyOwner {
      require(msg.sender == owner); // 检查调用者是否为owner地址
      _; // 如果是的话，继续运行函数主体；否则报错并revert交易
   }
   
   //代有onlyOwner修饰符的函数只能被owner地址调用
   function changeOwner(address _newOwner) external onlyOwner{
      owner = _newOwner; // 只有owner地址运行这个函数，并改变owner
   }
```



## 继承

**继承格式**

```solidity
contract A is B,C,D...
{
		
}
```

* 如果B，C，D....之间仍然有继承关系，需要从辈分由高到低顺序排序



**函数继承**

* 父类中的函数如果希望被子类重写需要加上`virtual`关键字

```solidity
contract Yeye {
    event Log(string msg);

    // 定义3个function: hip(), pop(), man()，Log值为Yeye。
    function hip() public virtual{
        emit Log("Yeye");
    }

    function pop() public virtual{
        emit Log("Yeye");
    }

    function yeye() public virtual {
        emit Log("Yeye");
    }
}
```

* 子类重写父类的函数再加上`override`关键字

```solidity
contract Baba is Yeye{
    // 继承两个function: hip()和pop()，输出改为Baba。
    function hip() public virtual override{
        emit Log("Baba");
    }

    function pop() public virtual override{
        emit Log("Baba");
    }

    function baba() public virtual{
        emit Log("Baba");
    }
}
```



**构造函数的继承**

* 需要标明父类继承中的参数格式
* 然后再写该子类构造器参数与父类构造器参数之间的关系

```solidity
contract C is A {
    constructor(uint _c) A(_c * _c) {}
}
```

**调用父类的函数**

* 直接调用：父类名.函数名( )
* `super`关键字调用：super.函数名( )  调用继承关系最亲近的那个父类的函数



## 抽象合约和接口

**抽象合约**

如果一个智能合约里至少有一个未实现的函数，即某个函数缺少主体`{}`中的内容，则必须将该合约标为`abstract`，不然编译会报错；另外，未实现的函数需要加`virtual`，以便子合约重写。如果我们还没想好具体怎么实现某个函数，那么可以把合约标为`abstract`，之后让别人补写上。

```solidity
abstract contract InsertionSort{
    function insertionSort(uint[] memory a) public pure virtual returns(uint[] memory);
}
```

**接口**

1. 不能包含状态变量
2. 不能包含构造函数
3. 不能继承除接口外的其他合约
4. 所有函数都必须是external且不能有函数体
5. 继承接口的合约必须实现接口定义的所有功能

```
interface base
{
	function getFirstName()public pure retruns(string memory);
}
contract B is base
{
	function getFirstName()public pure retruns(string memory)
	{
		return "HAHA";
	}
}
```



