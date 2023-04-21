from pypo import multimodel_rep, OI
import numpy as np


def shower_problem_2x2(u):
    y = np.zeros(2)
    y[0]=u[0]+u[1]
    if y[0]!=0:
        y[1]=(u[0]*60+u[1]*120)/(u[0]+u[1])
    else:
        y[1]=(60+120)/2
        
    return y


DOS_bounds =  np.array([[10, 20], 
                        [70, 100]])

AIS_bounds =  np.array([[0, 10],
                        [0, 10]])

AIS_resolution =  [5, 5]

model =  shower_problem_2x2


AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, model, plotting=True)


OI = OI(AOS_region, DOS_bounds, plotting=True)