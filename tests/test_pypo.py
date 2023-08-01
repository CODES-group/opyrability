import pytest
import numpy as np
from pypo import multimodel_rep, OI_eval, AIS2AOS_map, nlp_based_approach
from shower import shower2x2, shower3x3

def test_shower2x2():
    DOS_bounds =  np.array([[10, 20], 
                            [70, 100]])
    
    AIS_bounds =  np.array([[0, 10],
                            [0, 10]])
    
    AIS_resolution =  [5, 5]
    
    AIS, AOS = AIS2AOS_map(shower2x2, AIS_bounds, AIS_resolution, plot = False)
    
    AOS_region  =  multimodel_rep(AIS_bounds, AIS_resolution, shower2x2,
                                  plot = False)
    
    OI = OI_eval(AOS_region, DOS_bounds, plot = False)
    
    assert OI == pytest.approx(60.34, abs=1e-2, rel=1e-2)

    

def test_shower_analytical_inverse():
    
    from shower import inv_shower2x2
    from pypo import AIS2AOS_map

    AOS_bounds =  np.array([[10, 20], 
                            [70, 100]])


    AOS_resolution =  [5, 5]

    model =  inv_shower2x2

    DIS_bounds =  np.array([[0, 10.00],
                            [0, 10.00]])


    AIS, AOS = AIS2AOS_map(model, AOS_bounds, AOS_resolution)


    AIS_region  =  multimodel_rep(AOS_bounds, AOS_resolution, model, 
                                  polytopic_trace = 'polyhedra'
                                  , plot = False)

    OI = OI_eval(AIS_region, DIS_bounds, plot = False)
    
    assert OI == pytest.approx(40, abs=1e-2, rel=1e-2)
    
# def test_shower3x3():
    
#     DOS_bounds =  np.array([[ 10, 20], 
#                             [ 70, 100],
#                             [-10, 10]])


#     AIS_bounds =  np.array([[  0, 10],
#                             [  0, 10]])

#     EDS_bounds = np.array([[-10, 
#                             10]])

#     AIS_resolution = [5, 
#                       5]

#     EDS_resolution = [5]



#     AIS, AOS = AIS2AOS_map(shower3x3, 
#                            AIS_bounds, 
#                            AIS_resolution, 
#                            EDS_bound=EDS_bounds,
#                            EDS_resolution=EDS_resolution,
#                            plot = False)

#     AOS_region  =  multimodel_rep(AIS_bounds, 
#                                   AIS_resolution, 
#                                   shower3x3, 
#                                   EDS_bound=EDS_bounds,
#                                   EDS_resolution=EDS_resolution,
#                                   plot = False)

#     OI = OI_eval(AOS_region, DOS_bounds, plot = False)
    
#     OI == pytest.approx(69, abs=1e-1, rel=1e-1)

if __name__ == '__main__':
    pytest.main()
    