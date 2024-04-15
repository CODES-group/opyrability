
import jax.numpy as np
from dma_mr import dma_mr_2x1
from opyrability import AIS2AOS_map
# Defining DOS/AIS bounds and resolution.

DOS_bounds =  np.array([[20, 25], 
                        [35, 45]])

AIS_bounds =  np.array([[10, 150],
                        [0.5, 2]])

AIS_resolution =  [5, 5]


model  = dma_mr_2x1


AIS, AOS = AIS2AOS_map(model, 
                       AIS_bounds, 
                       AIS_resolution, 
                       plot=True)

