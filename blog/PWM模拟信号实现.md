---
tags: [hardware, C]
---
# PWM模拟信号实现
Date:  2016-06-16 11:32

## 介绍
- PWM信号是一种用于硬件控制的通信信号.
- PWM信号按周期里更改高低电平各自占用时间的差异来进行控制.
- 一个周期的时间硬件是有说明文档的.
  - 例如我最近做的四轴飞行器的XXD牌子的电子调速器简称电调(diǎo)的操作频率是`50Hz`,`1/50=0.02秒/次`即一个周期的时间为`20毫秒`即`20000微妙`. 
  - C语言的微妙延时的函数是`usleep(unsigned int);`
  - PS:一个周期即是程序猿的一次loop.
    ![1 1](https://cloud.githubusercontent.com/assets/3397225/12373499/d0a7c7fc-bcb5-11e5-95d9-5a8e5ec0c75f.jpg)

_如图所示__T__pwm就是一个周期,__T_*on为一个周期的高电平时间,当高电平时间越短,XXD电调就越慢[其他硬件有不同的使用规则的详细问X宝店长或上网找文档].


### Example: Arduino的电平转换函数
- **xxd电调(diǎo)调速**
		xxd电调的调速是要在>500微秒的高电平后才开始的,其他参数你们试试我也说不准,我也没有文档,只是百度回来的~~

```
int pin=3;//表示信号针脚的地址
int speed=100;
while(true){//确保一个loop需要使用0.02秒运行一次
digitalWrite(pin,HIGH);//高电平
usleep(500+speed);//高电平存在时间越长加速率越高
digitalWrite(pin,LOW);//低电平
usleep(20000-500-speed);//填充延时时间
```
-  **SG90陀机角度**
 
  ```
  int pin = 7; 
  int angle = 80;//角度:0~180度.PS: SG90陀机的旋转角度只有180度
  digitalWrite(pin ,HIGH);//高电平
  usleep(angle*11+500);//将角度转化为500-2480的脉宽值,__SG90陀机的控制范围为500-2480
  digitalWrite(pin ,LOW); //低电平
  usleep(20000-500-angle*11);  //填充延时时间
  ```




---
Copyright © 2021, [Jialin Li](https://github.com/keyskull).  [![Copyright](https://i.creativecommons.org/l/by-nc/4.0/80x15.png)](/LICENSE)