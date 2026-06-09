# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:49:29 2021

@author: mc
"""

import numpy as np
#
a = np.array([1,2,3])

x = np.zeros([3,3,3])
for i in range(3):
    for j in range(3):
        for z in range(3):
            x[i,j,z] = 100*(i+1) + 10*(j+1) + z + 1
y = x.reshape(-1)
print(x.reshape(-1))
for i in range(27):
    print(f"{i+1}:{y[i]}")