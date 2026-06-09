# -*- coding: utf-8 -*-
"""
Created on Fri May  7 03:40:50 2021

@author: WANG Mingchuan
"""

import numpy as np
from fg.mxfft import *
import time
#
path_tou = "series_shear_compare/mix/"
subs = ["1/","2/","3/","4/","5/","6/","7/","8/"]
#
N = 63
ndim = 3
delta  = lambda i,j: np.float32(i==j)
freq   = np.arange(-(N-1)/2.,+(N+1)/2.)
Ghat4  = np.zeros([ndim,ndim,ndim,ndim,N,N,N])
# - compute
for i,j,l,m in itertools.product(range(ndim),repeat=4):
    for x,y,z    in itertools.product(range(N),   repeat=3):
        q = np.array([freq[x], freq[y], freq[z]])  # frequency vector
        if not q.dot(q) == 0:                      # zero freq. -> mean
            Ghat4[i,j,l,m,x,y,z] = delta(i,l)*q[j]*q[m]/(q.dot(q))
#
print("Ghat4 is ok")

for sub in subs:
    path = path_tou + sub
    print("{} ======================================================".format(path))
    t1 = time.time()
    prob = FFTSolver(path,N=63)
    #
    incre_list=[1.0]
    prob.calculate(incre_list=incre_list,savemodel="normal", give_Ghat=True, Ghat_given=Ghat4)
    t2 = time.time()
    print("finish!")
    print(t2-t1)