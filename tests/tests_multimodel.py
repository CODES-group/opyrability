from dma_mr import dma_mr_design, dma_mr_mvs

from pypo import multimodel_rep, OI_eval

import numpy as np

# Tests for the multimodel approach - DMA-MR and shower problem


# %% DMA-MR - 2x2 System - Design Variables
    
# Defining DOS bounds

# DOS_bounds =  np.array([[20, 25], [35, 45]])

# AIS_bounds =  np.array([[10, 100],
#                     [0.5, 2]])

# AIS_resolution =  [6, 6]

# model  = dma_mr_design

# AOS_region  =  multimodel_rep(AIS_bounds, 
#                 AIS_resolution, model, polytopic_trace = 'simplices')

# OI = OI_calc(AOS_region,
#             DOS_bounds, hypervol_calc= 'robust')



# %% DMA-MR - 2x2 System - Manipulated Variables
    
# Defining DOS bounds

DOS_bounds = np.array([[15,25],
                        [35,45]])


AIS_bounds =  np.array([[450, 1500],
                    [450, 1500]])

AIS_resolution =  [10, 10]

model  = dma_mr_mvs

AOS_region  =  multimodel_rep(AIS_bounds,  AIS_resolution, model)

OI = OI_eval(AOS_region, DOS_bounds)





# %% Shower problem
# from shower import shower2x2
# DOS_bounds =  np.array([[10, 20], 
#                         [70, 100]])

# AIS_bounds =  np.array([[0.1, 10],
#                         [0.1, 10]])

# AIS_resolution =  [5, 5]

# model =  shower2x2

# AOS_region  =  multimodel_rep(AIS_bounds, 
#                 AIS_resolution, model, polytopic_trace = 'polyhedra')

# OI = OI_calc(AOS_region,
#             DOS_bounds, hypervol_calc= 'robust')
