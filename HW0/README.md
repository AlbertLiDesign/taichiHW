# 【作业0】BESO Topology Optimization

## 作业描述
在助教李喆昊的帮助下，我实现了基于BESO方法的拓扑优化。求解器、递归优化和单元刚度矩阵部分暂时没有使用taichi来完成，求解器依赖了scipy。

## 效果展示
![beso](./img/video.gif)

## 运行方式
运行环境：

```
[Taichi] version 0.8.0, llvm 10.0.0, commit 181c9039, osx, python 3.9.2
```

## Reference
1. [BESO2D.py](https://github.com/ToddyXuTao/BESO-for-2D)
2. [CISM_BESO_2D](https://www.cism.org.au/tools)
3. Zuo, Z.H. and Xie, Y.M., 2015. A simple and compact Python code for complex 3D topology optimization. Advances in Engineering Software, 85, pp.1-11.
4. Huang, X. and Xie, M., 2010. Evolutionary topology optimization of continuum structures: methods and applications. John Wiley & Sons.