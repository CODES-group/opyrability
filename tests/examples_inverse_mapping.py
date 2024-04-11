import numpy as np
from opyrability import nlp_based_approach

# ----------------------------------------------------------------------------
# Examples: Inverse mapping examples: Shower Problem and DMA-MR
# Author: Victor Alves
# Control, Optimization and Design for Energy and Sustainability,
# CODES Group, West Virginia University (2023)
# ----------------------------------------------------------------------------

"""
This script presents illustrative examples that demonstrate the utilization of 
the inverse mapping applied to the classic Shower Problem and a membrane
reactor (DMA-MR).

Each example explores diverse scenarios and dimensions, offering insights 
into the functionality of opyrability.

"""

# %% Shower problem inverse mapping - 2x2

from shower import shower2x2

# Defining initial estimate for DIS*, as well as lower and upper bounds.
u0 = np.array([0, 10])
lb = np.array([0, 0])
ub = np.array([100,100])

# DOS Bounds and resolution.
DOS_bound = np.array([[17.5, 21.0],
                      [80.0, 100.0]])

DOSresolution = [5, 5]

label = ['Cold water flow rate', 'Hot water flow rate', 'Total flow rate', 'Temperature']
    
# Obtaining DOS* and DIS*
fDIS, fDOS, message = nlp_based_approach(shower2x2,
                                         DOS_bound,
                                         DOSresolution,
                                         u0,
                                         lb,
                                         ub,
                                         method='ipopt',
                                         plot=True,
                                         ad=False,
                                         warmstart=False,
                                         labels=label)


norm_fDIS = np.linalg.norm(fDIS)
norm_fDOS = np.linalg.norm(fDOS)


# %% Shower problem inverse mapping - 3x3

from shower import shower3x3

# Defining initial estimate for DIS*, as well as lower and upper bounds.
u0 = np.array([10, 10, 5])
lb = np.array([0, 0, -10])
ub = np.array([20, 20, 10])

# DOS Bounds and resolution.
DOS_bound = np.array([[17.5, 21.0],
                    [80.0, 100.0],
                    [-10, 10]])

DOS_resolution = [5, 5, 5]

label = ['Cold water flow rate', 'Hot water flow rate', 'Disturbance',
         'Total flow rate', 'Temperature', 'Cold water flow rate']
    
# Obtaining DOS* and DIS*
fDIS, fDOS, message = nlp_based_approach(shower3x3,
                                          DOS_bound, 
                                          DOS_resolution, 
                                          u0, 
                                          lb,
                                          ub, 
                                          method='ipopt', 
                                          plot=True, 
                                          ad=False,
                                          warmstart=True,
                                          labels = label)
    
# %% Shower problem inverse mapping - 2x3 - Nonsquare

from shower import shower2x3

# Defining initial estimate for DIS*, as well as lower and upper bounds.
u0 = np.array([10, 10])
lb = np.array([0, 0])
ub = np.array([20, 20])

# DOS Bounds and resolution.
DOS_bound = np.array([[17.5, 21.0],
                    [80.0, 100.0],
                    [-10, 10]])

DOS_resolution = [5, 5, 5]

label = ['Cold water flow rate', 'Hot water flow rate', 
         'Total flow rate', 'Temperature', 'Cold water flow rate']
    
# Obtaining DOS* and DIS*
fDIS, fDOS, message = nlp_based_approach(shower2x3, 
                                          DOS_bound, 
                                          DOS_resolution, 
                                          u0, 
                                          lb,
                                          ub, 
                                          method='ipopt', 
                                          plot=True, 
                                          ad=False,
                                          warmstart=True)
    

# %% DMA-MR - Inverse mapping using JAX (AD) - 2x2
from dma_mr import dma_mr_design
import jax.numpy as np

# Lower and upper bounds for DOS definition
DOS_bounds = np.array([[15,25],
                        [35,45]])

# Discretization Grid - 10x10 grid for DOS.
DOS_resolution =  [3, 3]

# Lower and upper bounds of AIS (design)
lb = np.array([10,  0.5])
ub = np.array([100, 2])

# Initial estimate for NLP.
u0 = np.array([50, 1])

# Plug-flow constraint definition: Length/Diameter >= 30.
def plug_flow(u):
    return u[0] - 30.0*u[1]

con= {'type': 'ineq', 'fun': plug_flow}

legends = ['Length [cm]', 'Diameter [cm]', 
           '$Benzene \, production \, F_{C_{6}H_{6}} [mg/h]$',
           '$Methane \, conversion \, X_{CH_{4}} [\%]$']

# # Obtain inverse mapping: DOS* and DIS*                                             
fDIS, fDOS, convergence = nlp_based_approach(dma_mr_design,
                                              DOS_bounds, 
                                              DOS_resolution,
                                              u0, 
                                              lb,ub,
                                              constr=(con),
                                              method='ipopt', 
                                              plot=True,
                                              ad=True,
                                              warmstart=True,
                                              labels = legends)
                                
