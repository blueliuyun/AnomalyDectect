# -*- coding: utf-8 -*-   
from math import *
import random
 
#机器人四个参照物
landmarks  = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]
#地图大小
world_size = 100.0
class robot:
    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        #给机器人初试化一个坐标和方向
        self.forward_noise = 0.0;
        self.turn_noise    = 0.0;
        self.sense_noise   = 0.0;
    
    def set(self, new_x, new_y, new_orientation):
		#设定机器人的坐标　方向
        if new_x < 0 or new_x >= world_size:
            raise(ValueError, 'X coordinate out of bound')
        if new_y < 0 or new_y >= world_size:
            raise(ValueError, 'Y coordinate out of bound')
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise(ValueError, 'Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        #设定一下机器人的噪声
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);
        self.sense_noise   = float(new_s_noise);
    
    
    def sense(self):
		#测量机器人到四个参照物的距离　可以添加一些高斯噪声
        Z = []
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z
    
    
    def move(self, turn, forward):
        #机器人转向　前进　并返回更新后的机器人新的坐标和噪声大小
        if forward < 0:
            raise(ValueError, 'Robot cant move backwards')
        
        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi
        
        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= world_size    # cyclic truncate
        y %= world_size
        
        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res
    
    def Gaussian(self, mu, sigma, x):
        
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    
    def measurement_prob(self, measurement):
        
        # calculates how likely a measurement should be
        #计算出的距离相对于正确正确的概率　离得近肯定大　离得远就小
        prob = 1.0;
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob

###################################    
    
# 初始化一个机器人
myrobot = robot()
#设定噪声
myrobot.set_noise(5.0,0.1,5.0)
#设定初始位置
myrobot.set(30,50,0.5)
#打印位置方向
print(myrobot)
#打印与四个参照物的距离
Z=myrobot.sense()
print(Z)
 
#机器人移动　
myrobot=myrobot.move(pi/2,10.0)
print(myrobot)
Z=myrobot.sense()
print(Z)


#################

myrobot=robot()
myrobot.move(0.1,5.0)
Z=myrobot.sense()
N=1000
#初始化一千个粒子
p=[]
for i in range( N):
	x=robot()
	x.set_noise(0.05,0.05,5.0)
	p.append(x)
print(len(p))
p2=[]
for i in range(N):
	p2.append(p[i].move(0.1,5.0))
p=p2
#计算各个粒子的权重
w=[]
for i in range(N):
	w.append(p[i].measurement_prob(Z))
print(w)
