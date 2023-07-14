import numpy as np
from pypo import multimodel_rep, OI_eval


# Tests for the multimodel approach - DMA-MR and shower problem


# %% DMA-MR - 2x2 System - Design Variables
from dma_mr import dma_mr_design, dma_mr_mvs
# Defining DOS bounds

DOS_bounds =  np.array([[20, 25], 
                        [35, 45]])

AIS_bounds =  np.array([[10, 100],
                        [0.5, 2]])

AIS_resolution =  [10, 10]

model  = dma_mr_design

AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, model)

OI = OI_eval(AOS_region, DOS_bounds)



# %% DMA-MR - 2x2 System - Manipulated Variables
    
# Defining DOS bounds

# DOS_bounds = np.array([[15,25],
#                         [35,45]])


# AIS_bounds =  np.array([[450, 1500],
#                     [450, 1500]])

# AIS_resolution =  [10, 10]

# model  = dma_mr_mvs

# AOS_region  =  multimodel_rep(AIS_bounds,  AIS_resolution, model)

# OI = OI_eval(AOS_region, DOS_bounds)





# %% Shower problem 2x2
# from shower import shower2x2
# from pypo import AIS2AOS_map

# DOS_bounds =  np.array([[10, 20], 
#                         [70, 100]])

# AIS_bounds =  np.array([[0.1, 10],
#                         [0.1, 10]])

# AIS_resolution =  [5, 5]

# model =  shower2x2


# AIS, AOS = AIS2AOS_map(model, AIS_bounds, AIS_resolution)


# AOS_region  =  multimodel_rep(AIS_bounds, 
#                 AIS_resolution, model, polytopic_trace = 'simplices')

# OI = OI_eval(AOS_region, DOS_bounds, hypervol_calc= 'robust')

# %% Shower problem 3x3
# from shower import shower3x3
# from pypo import AIS2AOS_map
# import matplotlib.pyplot as plt
# DOS_bounds =  np.array([[10.00, 20.00], 
#                         [70.00, 100.00],
#                         [-10.00, 10.00]])


# AIS_bounds =  np.array([[0.00, 10.00],
#                         [0.00, 10.00],
#                         [-10.00, 10.00]])

# AIS_resolution =  [5, 5, 5]

# model =  shower3x3

# AIS, AOS = AIS2AOS_map(model, AIS_bounds, AIS_resolution)

# AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, model)

# OI = OI_eval(AOS_region, DOS_bounds)

# %% Shower problem 2x2 - analytical inverse map.
# from shower import inv_shower2x2
# from pypo import AIS2AOS_map

# AOS_bounds =  np.array([[10, 20], 
#                         [70, 100]])

# # AIS_bounds =  np.array([[0.1, 10],
# #                         [0.1, 10]])

# AOS_resolution =  [5, 5]

# model =  inv_shower2x2

# DOS_bounds =  np.array([[0, 10.00],
#                         [0, 10.00]])


# AIS, AOS = AIS2AOS_map(model, AOS_bounds, AOS_resolution)


# AOS_region  =  multimodel_rep(AOS_bounds, AOS_resolution, model, 
#                               polytopic_trace = 'polyhedra', 
#                               perspective = 'inputs')

# OI = OI_eval(AOS_region, DOS_bounds, hypervol_calc= 'robust', 
#              perspective = 'inputs')


