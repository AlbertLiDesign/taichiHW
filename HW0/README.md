# 太极图形课S1-【作业0】BESO Topology Optimization

## 背景简介
在助教李喆昊的帮助下，我实现了基于BESO方法的拓扑优化程序。其中求解器部分依赖了scipy。

双向渐进结构优化法（Bi-directional Evolutionary Structural Optimization, BESO）是由黄晓东教授和谢亿民院士提出的拓扑优化方法，它能在指定的体积约束下，通过逐步移除设计区域中的低效材料，并且在需要的地方增加材料（双向渐进）来获得高效、美观的结构设计。 简而言之，就是在给定的体积下，能自动地找到高效的结构设计。常用于轻量化设计，主要用在工业设计、建筑设计和航空航天等领域。

## 效果展示
下图示例是一个悬臂梁的优化过程，梁的左端被固定，右端中央被施加了一个向下的力，要求减少到原体积的50%，算法自动地找到了柔顺度最小的设计。
![beso](./img/video.gif)

## 运行方式
运行环境：

```
[Taichi] version 0.8.0, llvm 10.0.0, commit 181c9039, osx, python 3.9.2
```

## 参考资料
1. [BESO2D.py](https://github.com/ToddyXuTao/BESO-for-2D)
2. [CISM_BESO_2D](https://www.cism.org.au/tools)
3. Zuo, Z.H. and Xie, Y.M., 2015. A simple and compact Python code for complex 3D topology optimization. Advances in Engineering Software, 85, pp.1-11.
4. Huang, X. and Xie, M., 2010. Evolutionary topology optimization of continuum structures: methods and applications. John Wiley & Sons.