---
layout: pages
title: 服务器环境搭建
index_img: /img/banner/server.png 
date: 2025-01-10 15:58:22
categories:
- 环境配置
tags:
- server environment
---

## 远程桌面

由于种种原因，需要借助服务器的GPU，但同时也需要GUI界面。在此记录一下配置Ubuntu22.04服务器远程桌面的详细过程，避免后人踩坑。

### 服务器端

本人采用TigerVnc，按需也可以使用 Todesk(目前版本对终端用户不友好)

首先，检查服务器是否下载常见的图形化桌面。

```shell
dpkg -l | grep -E "gnome|kde|xfce|mate|cinnamon|lxde|lxqt"
```

若未下载，下载对应的图形化界面。

```shell
sudo apt install gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal ubuntu-desktop
```

下载Tigervnc

```shell
sudo apt install tigervnc-standalone-server
```

创建Vnc用户的密码

```shell
vncpasswd
```

* 输入6-8位密码，若超过8位会自动截断。
* 中途会问是否配置桌面为read-only，按照自己需要yes/no即可

用户的Vnc配置文件放在用户的`~/.vnc/xstartup`中

对xstartup文件配置：

```shell
sudo vim ~/.vnc/xstartup
```

若使用`gnome`图形化桌面（ubuntu最常见），添加如下内容

```bash
#!/bin/sh
export XKL_XMODMAP_DISABLE=1
export XDG_CURRENT_DESKTOP="GNOME-Flashback:GNOME"
export XDG_MENU_PREFIX="gnome-flashback-"
gnome-session --session=gnome-flashback-metacity --disable-acceleration-check
```

或者

```bash
cat xstartup 
#!/bin/sh
# Start Gnome 3 Desktop 
[ -x /etc/vnc/xstartup ] &amp;&amp; exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] &amp;&amp; xrdb $HOME/.Xresources
vncconfig -iconic &amp;
dbus-launch --exit-with-session gnome-session
```

* 注意这里末尾并没有添加&, 具体原因见 https://askubuntu.com/questions/1375111/vncserver-exited-too-early

```shell
sudo chmod a+x ~/.vnc/xstartup
```

之后便可以创建桌面啦

```shell
vncserver -localhost no
```

* vncserver默认创建时只有本地能连，对我们来讲这当然没用，所以添加参数`-localhost no`

顺利的话shell会弹出对应vncserver的编号信息。vnc采用端口号位`590*`, 若为1，则对应端口号`5901`,vpcserver地址就是`<服务器ip>:5901`,其中`<服务器ip>`填入ssh连接远程服务器时的ip即可

删除桌面

```shell
vncserver -kill :1
```



### 客户端

下载RealVNC: [Download VNC Viewer by RealVNC®](https://www.realvnc.com/en/connect/download/viewer/))

安装过程中一路默认next。完成后填入vpcserver地址，以及服务器创建vnc对应的密码即可。



## VPN

采用**反向端口转发**的方式

本地：如使用7897作为代理端口，具体在本地的clash中查看

![](/img/Server/clash.png)

```bash
ssh -NR 11804:127.0.0.1:7897 username@ip_address -v
```

服务器：

```bash
export http_proxy=http://127.0.0.1:11804/
export https_proxy=http://127.0.0.1:11804/
```

