# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:48:37 2022

@author: sqd0001
"""
import jax.numpy as jnp

# %% Feasibility Test number 1
'''
In this example, the feasible regions is 9*(u[0] - 3)^2 + 2.5 - u[1] >= 0
'''
def FeasTest1_implicit(u,y):
    LHS0 = y[0] - jnp.sqrt(9*(u[0] - 3)**2 + 2.5 - u[1])
    LHS1 = y[1] - (u[0] + u[1]**2)
    return jnp.array([LHS0, LHS1])

def FeasTest1(u):
    y = jnp.zeros(2)
    y[0] = jnp.sqrt(9*(u[0] - 3)**2 + 2.5 - u[1])
    y[1] = (u[0] + u[1]**2)
    return jnp.array(y)

AIS_bound = jnp.array([[0.0, 6.0],
                    [0.0, 6.0]])

# This resolution does not reveal the infeasibility
AISresolution_feas = [4, 4]

# This resolution reveals the infeasibility
AISresolution_infeas = [7, 7]
