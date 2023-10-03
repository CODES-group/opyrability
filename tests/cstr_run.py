from opyrability import AIS2AOS_map
import jax.numpy as np
from cstr_bifurcation import cstr
from scipy.optimize import root

def m(u):
    
    # y = np.zeros(2)
    
    x10     = 1.00
    # x20     = 0.00
    # x30     = -0.05
    

    # Ray 1982
    # beta    = 0.35
    # xi      = 1.00
    gamma   = 20.00
    phi     = 0.11
    # sigma_s = 0.44
    # sigma_b = 0.06
    q0      = 1.00
    # qc      = 1.00
    
    x10     = 1.00
    # x20     = 0.00
    # x30     = -0.05
    
    # y[1] = root(cstr, x0 = 10, args=u).x
    
    solution = root(cstr, x0 = 0.1, args=u).x
    Temp_dimensionless = solution
    
    fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
    
    # y[0] = q0*x10/ (q0 + phi*fx1*u[1])
    
    xA_dimensionless =  (q0*x10/ (q0 + phi*fx1*u[1]))
    
    conversion = ((x10 - xA_dimensionless)/ x10)
    
    return  np.array([Temp_dimensionless,conversion]).reshape(2,)


AIS_bound =  np.array([[0   , 1.50],
                       [0.25, 1.00]])

AIS_resolution = [150, 150]
AOS = AIS2AOS_map(m, AIS_bound, AIS_resolution)

# x10     = 1.00
# x20     = 0.00
# x30     = -0.05

# jac = jacrev(cstr)

# AIS = np.array([0.5, 0.6])

# y  =  root(cstr, x0 = 10, args=(AIS))