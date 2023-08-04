import pytest
import numpy as np
from pypo import multimodel_rep, OI_eval, AIS2AOS_map, nlp_based_approach
from shower import shower2x2, inv_shower2x2
from dma_mr import dma_mr_design, dma_mr_mvs

# Tolerances and to see or not to see the operability plots.
plot_flag = False
abs_tol = 1e-3
rel_tol = 1e-3


# Multimodel approach tests.
def test_shower2x2():
    DOS_bounds =  np.array([[10, 20], 
                            [70, 100]])
    
    AIS_bounds =  np.array([[0, 10],
                            [0, 10]])
    
    AIS_resolution =  [5, 5]
    
    AIS, AOS = AIS2AOS_map(shower2x2, AIS_bounds, AIS_resolution, plot = plot_flag)
    
    AOS_region  =  multimodel_rep(shower2x2, AIS_bounds, AIS_resolution,
                                  plot = plot_flag)
    
    OI = OI_eval(AOS_region, DOS_bounds, plot = plot_flag)
    
    assert OI == pytest.approx(60.33, abs=abs_tol, rel=rel_tol)

    

def test_shower_analytical_inverse():
    

    AOS_bounds =  np.array([[10, 20], 
                            [70, 100]])


    AOS_resolution =  [5, 5]

    model =  inv_shower2x2

    DIS_bounds =  np.array([[0, 10.00],
                            [0, 10.00]])


    AIS, AOS = AIS2AOS_map(model, 
                           AOS_bounds, 
                           AOS_resolution, 
                           plot = plot_flag)


    AIS_region  =  multimodel_rep(model, 
                                  AOS_bounds, 
                                  AOS_resolution, 
                                  polytopic_trace = 'polyhedra', 
                                  plot = plot_flag)

    OI = OI_eval(AIS_region, DIS_bounds, plot = plot_flag)
    
    assert OI == pytest.approx(40, abs=abs_tol, rel=rel_tol)
    
    
def test_dma_mr_design():
    
    DOS_bounds =  np.array([[20, 25], 
                            [35, 45]])

    AIS_bounds =  np.array([[10, 100],
                            [0.5, 2]])

    AIS_resolution =  [5, 5]


    AOS_region  =  multimodel_rep(dma_mr_design,
                                  AIS_bounds, 
                                  AIS_resolution,
                                  plot = plot_flag)

    OI = OI_eval(AOS_region, DOS_bounds, plot = plot_flag)
    
    assert OI == pytest.approx(22.11, abs=abs_tol, rel=rel_tol)


# NLP-based approach tests

def test_shower_inverse_nlp_2x2():
    
    u0 = np.array([0, 10])
    lb = np.array([0, 0])
    ub = np.array([100,100])

    DOS_bound = np.array([[17.5, 21.0],
                        [80.0, 100.0]])

    DOSresolution = [5, 5]
        


    fDIS, fDOS, message = nlp_based_approach(shower2x2,
                                             DOS_bound, 
                                             DOSresolution,
                                             u0,
                                             lb,
                                             ub,
                                             method='ipopt',
                                             plot=plot_flag,
                                             ad=False,
                                             warmstart=True)

    norm_fDIS = np.linalg.norm(fDIS)
    norm_fDOS = np.linalg.norm(fDOS)
    
    asserted_fDIS = 70.0683253828694
    asserted_fDOS = 461.57593383905106
    
    assert  norm_fDIS == pytest.approx(asserted_fDIS, abs=abs_tol, rel=rel_tol)
    assert  norm_fDOS == pytest.approx(asserted_fDOS, abs=abs_tol, rel=rel_tol)
    

if __name__ == '__main__':
    pytest.main()
    