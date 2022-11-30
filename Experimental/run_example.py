# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:35:10 2022

@author: sqd0001
"""
from operability_implicit_mapping import *
from DMA_MR_ss import *

# %% Test area! REMOVE BEFORE RELEASE!!!!
def shower_implicit(u,y):
    d = jnp.zeros(2)
    LHS1 = y[0] - (u[0]+u[1])
    # LHS2 = y[1] - (u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    if y[0]!=0:
        LHS2 = y[1] - (u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    else:
        LHS2 = y[1] - (60+120)/2
    
    return jnp.array([LHS1, LHS2])
def shower(u):
    y = np.zeros(2)
    d = jnp.zeros(2)
    y[0] = (u[0]+u[1])
    # LHS2 = y[1] - (u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    if y[0]!=0:
        y[1] = (u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    else:
        y[1] = (60+120)/2
    
    return jnp.array(y)

def FF1(u):
    y = np.zeros(2)
    # y[0] = u[1]**2*u[0]
    y[0] = u[0] - 2*u[1]
    y[1] = 3*u[0] + 4*u[1]
    return jnp.array(y)

def FF1_implicit(u,y):
    # LHS0 = y[0] - u[1]**2*u[0]
    LHS0 = y[0] - (u[0] - 2*u[1])
    LHS1 = y[1] - (3*u[0] + 4*u[1])
    return jnp.array([LHS0, LHS1])

#%% Test DMA-MR inverse
DOS_bound = np.array([[22.4, 22.8],
                    [39.4, 40.0]])

DOSresolution = [10, 10]

output_init = np.array([20.0, 0.9])

DOS, DIS, DOS_poly, DIS_poly = implicit_map(F_DMA_MR_eqn, 
                                            DOS_bound, 
                                            DOSresolution, 
                                            output_init, 
                                            direction = 'inverse')

# %% Test shower forward
AIS_bound = np.array([[10.0, 100.0],
                    [0.5, 2.0]])

AISresolution = [10, 10]

output_init = np.array([00.0, 10])

AIS, AOS, AIS_poly, AOS_poly = implicit_map(shower_implicit, 
                                            AIS_bound, 
                                            AISresolution, 
                                            output_init)