
import line_profiler
import numpy as np
from pypo import multimodel_rep
from shower import shower2x2

DOS_bounds =  np.array([[10, 20], 
                         [70, 100]])

AIS_bounds =  np.array([[0, 10],
                         [0, 10]])

AIS_resolution =  [3, 3]

model =  shower2x2

@profile  # for line_profiler
def profiled_multimodel():
    return  multimodel_rep(model, AIS_bounds, AIS_resolution)


