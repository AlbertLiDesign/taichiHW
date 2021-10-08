# 太极图形课S1-作业1

## 作业描述
实现了一个[Marching Squares](http://jamie-wong.com/2014/08/19/metaballs-and-marching-squares/)类，并用它分别做了两个程序，第一个是Marching Squares的原理可视化，我在gui中创建了一个网格，求每个网格顶点到黄色点的距离（标记在每个点上），然后使用Marching Squares来提取等值轮廓，轮廓提取的iso-value由用户控制，可大可小。

第二个程序是用Marching Squares来可视化三个行星的平均距离，即求每个顶点到三个行星的平均距离来绘制轮廓。

## 效果展示
![ms](./ms_img/video.gif)
![ms_galaxy](./galaxy_img/video.gif)

## 运行方式
运行环境：

```
[Taichi] version 0.8.0, llvm 10.0.0, commit 181c9039, osx, python 3.9.2
```

按键

- `Space` : 暂停/继续
- `i` : 运行时导出图片/停止导出（导出的图片会存放在./images/目录下）
- `r` : 重置
- `q`：增大iso-value
- `e`：减小iso-value
- `w, a, s, d`：控制移动 （也可鼠标点击移动）