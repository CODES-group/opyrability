from dma_mr import dma_mr_design, dma_mr_mvs
from shower import shower2x2
import sys
sys.path.append('../')

# from src.PolyhedraVolAprox import *
from src.pyprop import multimodel_rep, OI_calc

import numpy as np

# Tests for the multimodel approach - DMA-MR and shower problem


# %% DMA-MR - 2x2 System - Design Variables
    
# Defining DOS bounds

DOS_bounds =  np.array([[20, 25], [35, 45]])

AIS_bounds =  np.array([[10, 100],
                    [0.5, 2]])

AIS_resolution =  [6, 6]

model  = dma_mr_design

AOS_region  =  multimodel_rep(AIS_bounds, 
                AIS_resolution, model, polytopic_trace = 'simplices')

OI = OI_calc(AOS_region,
            DOS_bounds, hypervol_calc= 'robust')



# %% DMA-MR - 2x2 System - Manipulated Variables
    
# Defining DOS bounds

# DOS_bounds =  np.array([[70, 80], [30, 45]])

# AIS_bounds =  np.array([[600, 1800],
#                     [600, 1800]])

# AIS_resolution =  [5, 5]

# model  = dma_mr_mvs

# AOS_region  =  multimodel_rep(AIS_bounds, 
#                 AIS_resolution, model, polytopic_trace = 'polyhedra')

# OI = OI_calc(AOS_region,
#             DOS_bounds, hypervol_calc= 'robust')





# %% Shower problem

# DOS_bounds =  np.array([[10, 20], 
#                         [70, 100]])

# AIS_bounds =  np.array([[0.1, 10],
#                         [0.1, 10]])

# AIS_resolution =  [15, 15]

# model =  shower2x2

# AOS_region  =  multimodel_rep(AIS_bounds, 
#                 AIS_resolution, model, polytopic_trace = 'polyhedra')

# OI = OI_calc(AOS_region,
#             DOS_bounds, hypervol_calc= 'robust')
