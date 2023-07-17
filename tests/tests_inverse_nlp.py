import time
from   pypo import nlp_based_approach
# %% NLP-Based Approach - Shower inverse mapping test
import numpy as np
from shower import shower2x2

u0 = np.array([0, 10])
lb = np.array([0, 0])
ub = np.array([100,100])

DOS_bound = np.array([[17.5, 21.0],
                    [80.0, 100.0]])

DOSresolution = [10, 10]
    

t = time.time()
fDIS, fDOS, message = nlp_based_approach(DOS_bound, 
                                          DOSresolution, 
                                          shower2x2, 
                                          u0, 
                                          lb,
                                          ub, 
                                          method='ipopt', 
                                          plot=True, 
                                          ad=False,
                                          warmstart=False)
    
    
elapsed = time.time() - t
# %% DMA-MR - Inverse mapping test using JAX (AD)
# from dma_mr import *
# import jax.numpy as np



# # Lower and upper bounds for DOS definition
# DOS_bounds = np.array([[15,25],
#                         [35,45]])

# # Discretization Grid - 10x10 grid for DOS.
# DOS_resolution =  [10, 10]

# # Lower and upper bounds of AIS (design)
# lb = np.array([10,  0.1])
# ub = np.array([300, 2])

# # Initial estimate for NLP.
# u0 = np.array([50, 1])

# # Plug-flow constraint definition: Length/Diameter >= 30.
# def plug_flow(u):
#     return u[0] - 30.0*u[1]

# con= {'type': 'ineq', 'fun': plug_flow}

# # Model assignment: Design Problem - Inverse mapping
# # model          = dma_mr_design
# # Obtain inverse mapping.
# t = time.time()                                                  
# fDIS, fDOS, convergence = nlp_based_approach(DOS_bounds, DOS_resolution,
#                                 dma_mr_design, 
#                                 u0, 
#                                 lb,ub,
#                                 constr=(con),
#                                 method='ipopt', 
#                                 plot=True,
#                                 ad=True,
#                                 warmstart=True)
# elapsed = time.time() - t