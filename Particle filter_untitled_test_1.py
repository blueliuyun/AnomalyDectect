# -*- coding=utf-8 -*-
 
# 直接运行代码可以看到跟踪效果
# 红色的小点代表粒子位置
# 蓝色的大点表示跟踪的结果
# 白色的方框表示要跟踪的目标
# 看懂下面两个函数即可
 
from numpy import *
from numpy.random import *
# 说实话，不太赞同这种 import 方法
# 第一是可能导致名字冲突，第二，很多函数方法都不知道是哪个模块里
 
 
def resample(weights):
    n = len(weights)
    indices = []
    # 求出离散累积密度函数(CDF)
    C = [0.] + [sum(weights[:i+1]) for i in range(n)] 
    # 选定一个随机初始点
    u0, j = random(), 0
    for u in [(u0+i)/n for i in range(n)]: # u 线性增长到 1
        while u > C[j]: # 碰到小粒子，跳过
            j+=1
        indices.append(j-1)  # 碰到大粒子，添加，u 增大，还有第二次被添加的可能
    return indices # 返回大粒子的下标
 
 
def particlefilter(sequence, pos, stepsize, n):
    ''' sequence: 表示图片序列
        pos： 第一帧目标位置
        stepsize： 采样范围
        n: 粒子数目
        '''
    seq = iter(sequence)
    x = ones((n, 2), int) * pos      # 100 个初始位置(中心)
    
    x=np.array(x,dtype=np.float) 
    
    f0 = seq.__next__()[tuple(pos)] * ones(n)  # 目标的颜色模型， 100 个 255
    yield pos, x, ones(n)/n # 返回第一帧期望位置(expectation pos)，粒子(x)和权重
    for im in seq:
        # 在上一帧的粒子周围撒点, 作为当前帧的粒子
        x += uniform(-stepsize, stepsize, x.shape) 
        # 去掉超出画面边框的粒子
        x  = x.clip(zeros(2), array(im.shape)-1).astype(int) 
        f  = im[tuple(x.T)]  # 得到每个粒子的像素值
        w  = 1./(1. + (f0-f)**2)  # 求与目标模型的差异, w 是与粒子一一对应的权重向量
                                  # 可以看到像素值为 255 的权重最大(1.0)
        w /= sum(w)      # 归一化 w
        yield sum(x.T*w, axis=1), x, w # 返回目标期望位置，粒子和对应的权重
        if 1./sum(w**2) < n/2.:    # 如果当前帧粒子退化:
            x  = x[resample(w),:]  # 根据权重重采样, 有利于后续帧有效采样
 
 
 
if __name__ == "__main__":
    from pylab import *
    #from itertools import izip
    izip = zip
    import time
    ion() # 打开交互模式
    seq = [ im for im in zeros((20,240,320), int)]      # 创建 20 帧全 0 图片
    x0 = array([120, 160])                              # 第一帧的框中心坐标
 
    # 为每张图片添加一个运动轨迹为 xs 的白色方块(像素值是255, 每帧横坐标加3,竖坐标加2)
 
    xs = vstack((arange(20)*3, arange(20)*2)).T + x0 # vstack: 竖直叠加
    for t, x in enumerate(xs):   # t 从 0 开始， x 从 xs[0] 开始
        # slice 的用法也很有意思，可以很方便用来表示被访问数组seq的下标范围
        xslice = slice(x[0]-8, x[0]+8) 
        yslice = slice(x[1]-8, x[1]+8)
        seq[t][xslice, yslice] = 255
 
    # 跟踪白框
    for im, p in izip(seq, particlefilter(seq, x0, 8, 100)): # 
        pos, xs, ws = p
        position_overlay = zeros_like(im)
        position_overlay[tuple(pos)] = 1
        particle_overlay = zeros_like(im)
        particle_overlay[tuple(xs.T)] = 1
        hold(True)
        draw()
        time.sleep(0.3)
        clf()                                           # Causes flickering, but without the spy plots aren't overwritten
        imshow(im,cmap=cm.gray)                         # Plot the image
        spy(position_overlay, marker='.', color='b')    # Plot the expected position
        spy(particle_overlay, marker=',', color='r')    # Plot the particles
    show()