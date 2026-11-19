#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:07:59 2026

@author: C.Pritz
"""

# imports 
import numpy as np
import matplotlib.pyplot as plt


#Defining all the parameters
k = 0.9/100 #volume change  konstant, this constant is defined as change per dt (that is why it is divided by 100)
V0 = 3 #initial volume of the balloon 
dt = 0.01 # size of each time step
T = 10 # total time of simulation 

time = np.arange(0,T,dt)
V = np.zeros_like(time)
V[0] = V0

for i in range(1,time.size):
    dV_dt = -k*V[i-1]
    V[i] = V[i-1] + dV_dt
    
    
    
fig,ax = plt.subplots(figsize=(4,3),dpi=300)
ax.set_xlabel('time (s)')
ax.set_ylabel('Volume (l)')
plt.plot(time,V)
plt.show()






