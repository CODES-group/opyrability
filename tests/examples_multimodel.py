import numpy as np
from opyrability import multimodel_rep, OI_eval



# Tests for the multimodel approach - DMA-MR and shower problem


# %% DMA-MR - 2x2 System - Design Variables
from dma_mr import dma_mr_design, dma_mr_mvs
# Defining DOS bounds

DOS_bounds =  np.array([[20, 25], 
                        [35, 45]])

AIS_bounds =  np.array([[10, 150],
                        [0.5, 2]])

AIS_resolution =  [15, 15]

model  = dma_mr_design

AOS_region  =  multimodel_rep(model, AIS_bounds, AIS_resolution,
                              polytopic_trace='simplices')

OI = OI_eval(AOS_region, DOS_bounds)



# %% DMA-MR - 2x2 System - Manipulated Variables
# from dma_mr import dma_mr_design, dma_mr_mvs
# # Defining DOS bounds

# DOS_bounds = np.array([[15,25],
#                         [35,45]])


# AIS_bounds =  np.array([[450, 1500],
#                     [450, 1500]])

# AIS_resolution =  [5, 5]

# model  = dma_mr_mvs

# AOS_region  =  multimodel_rep(model, AIS_bounds,  AIS_resolution,)

# OI = OI_eval(AOS_region, DOS_bounds)





# %% Shower problem 2x2
# from shower import shower2x2
# from opyrability import AIS2AOS_map

# DOS_bounds =  np.array([[10, 20], 
#                         [70, 100]])

# AIS_bounds =  np.array([[0, 10],
#                         [0, 10]])

# AIS_resolution =  [5, 5]

# model =  shower2x2


# AIS, AOS = AIS2AOS_map(model, AIS_bounds, AIS_resolution)


# AOS_region  =  multimodel_rep(model, AIS_bounds, 
#                 AIS_resolution, polytopic_trace = 'simplices')

# OI = OI_eval(AOS_region, DOS_bounds, hypervol_calc= 'robust')

# %% Shower problem 3x3
# from shower import shower3x3
# from opyrability import AIS2AOS_map
# DOS_bounds =  np.array([[10.00, 20.00], 
#                         [70.00, 100.00],
#                         [-10.00, 10.00]])


# AIS_bounds =  np.array([[0.00, 10.00],
#                         [0.00, 10.00]])

# # AIS_bounds =  np.array([[0.00, 10.00]])

# # EDS_bounds = np.array([[-10.00, 10.00],
# #                        [-20, 50]])

# EDS_bounds = np.array([[-10.00, 10.00]])

# AIS_resolution = [5, 5]

# EDS_resolution = [5]

# model =  shower3x3

# # AIS, AOS = AIS2AOS_map(model, 
# #                         AIS_bounds, 
# #                         AIS_resolution, 
# #                         EDS_bound=EDS_bounds,
# #                         EDS_resolution=EDS_resolution)

# AOS_region  =  multimodel_rep(model,
#                               AIS_bounds, 
#                               AIS_resolution,
#                               EDS_bound=EDS_bounds,
#                               EDS_resolution=EDS_resolution,
#                               plot = False)

# OI = OI_eval(AOS_region, DOS_bounds, plot = True)



# %% Shower problem 2x3

# from shower import shower2x3
# from opyrability import AIS2AOS_map
# import matplotlib.pyplot as plt
# DOS_bounds =  np.array([[10.00, 20.00], 
#                         [70.00, 100.00],
#                         [70.00, 100.00]])


# AIS_bounds =  np.array([[0.00, 10.00],
#                         [0.00, 10.00]])

# AIS_resolution =  [5, 5]

# model =  shower2x3

# AIS, AOS = AIS2AOS_map(model, AIS_bounds, AIS_resolution)


# %% Shower problem 3x2


# from shower import shower3x2
# from opyrability import AIS2AOS_map


# DOS_bounds =  np.array([[10.00, 20.00], 
#                         [70.00, 100.00]])


# AIS_bounds =  np.array([[0.00, 10.00],
#                         [0.00, 10.00],
#                         [-10.00, 10.00]])

# AIS_resolution =  [5, 5, 5]

# model =  shower3x2

# AIS, AOS = AIS2AOS_map(model, AIS_bounds, AIS_resolution)

# AOS_region  =  multimodel_rep(model, AIS_bounds, AIS_resolution,
# plot=True, polytopic_trace='polyhedra')

# OI = OI_eval(AOS_region, DOS_bounds, plot = True)


# %% Shower inverse mapping - multimodel representation 3x3
# import numpy as np
# from shower import shower3x3, inv_shower3x3
# from opyrability import AIS2AOS_map, multimodel_rep, nlp_based_approach

# u0 = np.array([10, 10, 5])
# lb = np.array([0, 0, -10])
# ub = np.array([20, 20, 10])

# DOS_bounds = np.array([[17.5, 21.0],
#                     [80.0, 100.0],
#                     [-10, 10]])

# DOS_resolution = [6, 6, 6]
    
# # AIS, AOS = AIS2AOS_map(inv_shower3x3, DOS_bounds, DOS_resolution)
# # t = time.time()
# # fDIS, fDOS, message = nlp_based_approach(DOS_bounds, 
# #                                           DOS_resolution, 
# #                                           shower3x3, 
# #                                           u0, 
# #                                           lb,
# #                                           ub, 
# #                                           method='ipopt', 
# #                                           plot=True, 
# #                                           ad=False,
# #                                           warmstart=True)
    

# AOS_region  =  multimodel_rep(DOS_bounds, DOS_resolution, shower3x3, 
#                               perspective = 'inputs')
   
# elapsed = time.time() - t


# %% Shower problem 2x2 - Inverse using NLP + Multimodel
# from shower import shower2x2
# from opyrability import AIS2AOS_map

# DOS_bounds =  np.array([[10, 20], 
#                         [70, 100]])

# DIS_bounds =  np.array([[0, 10],
#                         [0, 10]])

# AIS_resolution =  [3, 3]

# DOS_resolution = [3, 3]

# model =  shower2x2


# # AIS, AOS = AIS2AOS_map(model, AIS_bounds, AIS_resolution)


# AIS_region  =  multimodel_rep(DOS_bounds, 
#                               DOS_resolution, 
#                               model, 
#                               perspective='inputs')

# OI = OI_eval(AIS_region, 
#              DIS_bounds, 
#              perspective='inputs')

