from dma_mr_grid_map import dma_mr_2x2, dma_mr_mvs


from multimodel import multimodel_rep, OI_calc

import numpy as np

# Tests for the multimodel approach - DMA-MR and shower problem


# %% DMA-MR - 2x2 System - Design Variables
    
# Defining DOS bounds

DOS_bounds =  np.array([[20, 25], [35, 45]])

AIS_bounds =  np.array([[10, 100],
                    [0.5, 2]])

AIS_resolution =  [5, 5]

model  = dma_mr_2x2

AOS_region  =  multimodel_rep(AIS_bounds, 
                AIS_resolution, model)

OI = OI_calc(AOS_region,
            DOS_bounds)



# %% DMA-MR - 2x2 System - Manipulated Variables
    
# Defining DOS bounds

DOS_bounds =  np.array([[70, 80], [30, 45]])

AIS_bounds =  np.array([[600, 1800],
                    [600, 1800]])

AIS_resolution =  [5, 5]

model  = dma_mr_mvs

AOS_region  =  multimodel_rep(AIS_bounds, 
                AIS_resolution, model)

OI = OI_calc(AOS_region,
            DOS_bounds)





# %% Shower problem

def shower(u):
    
    d = np.zeros(2)
    y = np.zeros(2)
    y[0]=u[0]+u[1]
    if y[0]!=0:
        y[1]=(u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    else:
        y[1]=(60+120)/2
        
    return y

DOS_bounds =  np.array([[10, 20], 
                        [70, 100]])

AIS_bounds =  np.array([[0, 10],
                        [0, 10]])

AIS_resolution =  [5, 5]

model =  shower

AOS_region  =  multimodel_rep(AIS_bounds, 
                AIS_resolution, model)

OI = OI_calc(AOS_region,
            DOS_bounds)
