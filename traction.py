# -*- coding: utf-8 -*-
"""
Created on Fri May  7 03:40:50 2021

@author: WANG Mingchuan
"""

import numpy as np
from fg.mxfft import *
from result_plots import save_result_plots
import time
#
SAVE_PLOTS = True
PLOT_DPI = 200

subs = ["7/"] #,"2/","3/","4/","5/","6/"]
for sub in subs:
    path = "own_charge/" + sub
#   
    print(path)
    t1 = time.time()
    prob = FFTSolver(path,N=31, charge_path=None)
    #
    incre_list=[0.1]*10
    prob.calculate(incre_list=incre_list,savemodel="normal", preconditioner="reference", diagnostics=False)
    if SAVE_PLOTS:
        plot_files = save_result_plots(path, dpi=PLOT_DPI)
        print("plots are saved...")
        for plot_file in plot_files:
            print(plot_file)
    t2 = time.time()
    print("finish!")
    print(t2-t1)