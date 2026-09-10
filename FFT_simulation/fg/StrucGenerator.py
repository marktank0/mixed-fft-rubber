# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 23:03:37 2021

@author: WANG Mingchuan

generate structure used in FFT method

"""

import numpy as np
import itertools
#import numba

#
class Voxels:
    """ N must be odd number"""
    def __init__(self,N=31,L=1.0):
        """ """
        self.N = N
        self.L = L
        assert (N+1)%2 == 0         #N must be odd number
        #phase value
        self.phase = np.zeros([N,N,N])
        #coordinates
        self.xyz   = np.zeros([3,N,N,N])
        self.__coords()
        #
        #
    #
    def __coords(self):
        """ middle is zero """
        #calculate the center coordinates of each voxel
        #each voxel has 1x1x1 3D or 1x1 2D
        begin = 0.5*(self.L/self.N-self.L)
        end   = 0.5*(self.L+self.L/self.N)
        seg   = self.L/self.N
        Xs = np.arange(begin,end,seg)
        #
        for x,y,z in itertools.product(range(self.N),repeat=3):
            self.xyz[:,x,y,z] = np.array([Xs[x],Xs[y],Xs[z]])
        #
    #
#
    def sphere(self,a,b,c,center):
        """ sphere phase 1 at center with a,b,c"""
        for x,y,z in itertools.product(range(self.N),repeat=3):
            pos = self.xyz[:,x,y,z]
            #judge if this point is in the sphere
            dis = ((pos[0] - center[0])/a)**2 + ((pos[1] - center[1])/b)**2 \
                + ((pos[2] - center[2])/c)**2
            if dis <= 1.0:
                self.phase[x,y,z] = 1.0

    def rdm(self,percentage):
        """random phase 2, with percentage"""
        assert percentage < 1.0
        Totalnum = self.N*self.N*self.N
        Voidnum  = int(percentage*Totalnum)
        #
        inds = np.arange(0,Totalnum)
        void_inds = np.random.choice(inds,Voidnum,replace=False)
        #
        phase = np.zeros(Totalnum)
        #
        phase[void_inds] = 1.0
        self.phase = phase.reshape([self.N,self.N,self.N])
        #
    def record(self,path):
        #
        phaselist = self.phase.reshape(-1)
        np.savetxt(path,phaselist,fmt="%.1f")
######-------------------------------------------------------------------------



if __name__ == "__main__":
    #
    hh = np.arange(0,100000)
    hoho = np.random.choice(hh,20000,replace=False)
    hihi = np.sort(hoho)
    print(hihi.shape[0])
    #print(hihi)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        