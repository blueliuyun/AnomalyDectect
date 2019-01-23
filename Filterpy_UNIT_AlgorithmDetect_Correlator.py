# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:18:03 2019

@author: tianye

 1. 相关系数计算。
 
"""

######################################################################
# func. 相关系数
# Return correlator: 浮点型，若是 >= 0.5 则说明当前 index 不合适，需继续找下一个。
u0_312 = u0[index-128*2-10:index-128*2+11]
u0_440 = u0[index-128-10:index-128+11]
u0_568 = u0[index-10:index+11]
u0_696 = u0[index+128-10:index+128+11]

#u0_440 = u0_440
#u0_568 = u0_696
print(u0_440)
print(u0_568)

#prod = map(lambda (a,b):a*b, zip(u0_440, u0_568))
u0_sum_up = 0.0
for i in range(len(u0_568)):
    u0_sum_up = u0_sum_up + u0_440[i]*u0_568[i]
print(u0_sum_up)

u0_sum_down = 0.0
u0_sum_down_a = 0.0
u0_sum_down_b = 0.0
for i in range(len(u0_568)):
    u0_sum_down_a = u0_sum_down_a + u0_440[i]*u0_440[i]
    u0_sum_down_b = u0_sum_down_b + u0_568[i]*u0_568[i]

# a 或 b list元素全是0值，并且 a != b
if((u0_sum_down_a == 0 or u0_sum_down_b == 0) and (u0_sum_down_a != u0_sum_down_b)):
    print("correlator : false")
else: 
    u0_sum_down = u0_sum_down_a * u0_sum_down_b
    u0_sum_down = u0_sum_down**0.5
    print(u0_sum_down)
    correlator = u0_sum_up/u0_sum_down
    print("correlator : %f \r\n" % correlator)