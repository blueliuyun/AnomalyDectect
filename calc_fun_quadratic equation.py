# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:15:38 2019

@author: tianye
 1. Python 计算一元二次函数的代码。
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
def quadratic(a,b,c):
    if a == 0:
        raise TypeError('a不能为0')
    if not isinstance(a,(int,float)) or  not isinstance(b,(int,float)) or not isinstance(c,(int,float)):
        raise TypeError('Bad operand type')
    delta = math.pow(b,2) - 4*a*c
    if delta < 0:
        return '无实根'
    x1= (math.sqrt(delta)-b)/(2*a)
    x2=-(math.sqrt(delta)+b)/(2*a)
    return x1,x2
print(quadratic(200, 3084, 45000))
print(quadratic(1, 5, 1))