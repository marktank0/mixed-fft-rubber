# -*- coding: utf-8 -*-
"""
Created on Fri May  7 03:40:50 2021

@author: WANG Mingchuan
"""

import numpy as np
from fg.mxfft import *
import time
#
subs = ["3/"] #,"2/","3/","4/","5/","6/"]
for sub in subs:
    path = "own_charge/" + sub
#   
    print(path)
    t1 = time.time()
    prob = FFTSolver(path,N=31, charge_path=None)
    #
    incre_list=[0.125]*8
    prob.calculate(incre_list=incre_list,savemodel="normal", preconditioner="reference")
    t2 = time.time()
    print("finish!")
    print(t2-t1)
