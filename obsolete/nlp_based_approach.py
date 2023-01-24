import numpy as np
# import numpy.typing as npt
import scipy as sp
from scipy.optimize import differential_evolution as DE
from cyipopt import minimize_ipopt
from typing import Callable,Union
from pdfo import pdfo, Bounds, LinearConstraint, NonlinearConstraint

def nlp_based_approach(DOSPts: np.ndarray, 
                       model: Callable[...,Union[float,np.ndarray]], 
                       u0: np.ndarray, 
                       lb: np.ndarray,
                       ub: np.ndarray,
                       constr=None,
                       method: str ='DE') -> Union[np.ndarray,np.ndarray,list]:
    '''
    Inverse mapping for Process Operability calculations. From a Desired Output
    Set (DOS) defined by the user, this function calculates the closest
    Feasible Desired Ouput set (DOS*) from the AOS and its respective Feasible
    Desired Input Set (DIS*), which gives insight about potential changes in
    design and/or operations of a given process model.
    
    This function is part of Python-based Process Operability package.
    
    Control, Optimization and Design for Energy and Sustainability (CODES) 
    Group - West Virginia University - 2022
    
    Author: Victor Alves

    Parameters
    ----------
    DOSPts : np.ndarray
        Array containing the Desired Output set (DOS). Each row correspond to
        a scenario (discretized or sampled) and each column correspond to an
        output variable.
    model : Callable[...,Union[float,np.ndarray]]
        Process model that calculates the relationship between inputs (AIS-DIS) 
        and Outputs (AOS-DOS).
    u0 : np.ndarray
        Initial estimate for the inverse mapping at each point. Should have
        same dimension as model inputs.
    lb : np.ndarray
        Lower bound on inputs.
    ub : np.ndarray
        Upper bound on inputs.
    constr : list, optional
        List of functions containing constraints to the inverse mapping.
        The default is None. follows scipy.optimize synthax.
    method: str
        Optimization method used. The default is DE 
        (Differential evolution from scipy). Set 'ipopt' if CyIpopt is 
        installed and want to use gradient-based solver.

    Returns
    -------
    fDIS: np.ndarray
        Feasible Desired Input Set (DIS*). Array containing the solution for
        each point of the inverse-mapping.
    fDIS: np.ndarray
        Feasible Desired Output Set (DOS*). Array containing the feasible
        output for each feasible input calculated via inverse-mapping.
    message_list: list
        List containing the termination criteria for each optimization run
        performed for each DOS grid point.
        
    References
    ----------
    [1] J. C. Carrasco and F. V. Lima, “An optimization-based operability 
    framework for process design and intensification of modular natural 
    gas utilization systems,” Computers & Chemical Engineering, 
    vol. 105, pp. 246-258, 2017.

    '''
    
    # Initialization of variables
    m = len(u0)
    r,c = np.shape(DOSPts)
    fDOS = np.zeros((r,c))
    fDIS = np.zeros((r,m))
    message_list = []
    bounds =  np.column_stack((lb,ub)) 
    
    # Input sanitation 
    if u0.size !=c:
        raise ValueError("Initial estimate and DOS grid have"
                         " inconsistent sizes. Check the dimensions"
                         " of your problem.")
    if bounds.shape[0] != u0.size:
        raise ValueError("Initial estimate and given bounds have"
                         " inconsistent sizes."
                         " Check the dimensions"
                         " of your problem.")
    if bounds.shape[0] !=c:
        raise ValueError("Initial estimate and DOS grid have"
                         " inconsistent sizes."
                         " Check the dimensions"
                         " of your problem.")
    
    # If unbounded, set as +-inf.
    if lb.size == 0:
        lb = -np.inf
    if ub.size == 0:
        ub = np.inf
    
    # Inverse-mapping: Run for each DOS grid point
    for i in range(r):
        
        
        # This approach is useful for ipopt
        def obj(u):
            return p1(u, model, DOSPts[i,:])
        
        if constr is None:

            if method == 'trust-constr':
                sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                           args=(model, DOSPts[i, :]),
                                           method=method, 
                                           options={'xtol': 1e-12})

            elif method == 'Nelder-Mead':
                sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                           args=(model, DOSPts[i, :]), 
                                           method=method, 
                                           options={'fatol': 1e-10,
                                                    'xatol': 1e-10})

            elif method == 'ipopt':
                sol = minimize_ipopt(p1, x0=u0, bounds=bounds,
                                     args=(model, DOSPts[i, :]))

            elif method == 'DE':
                sol = DE(p1, bounds=bounds, x0=u0, strategy='best1bin',
                         maxiter=2000, workers=-1, updating='deferred',
                         init='sobol', args=(model, DOSPts[i, :]))

            elif method == 'COBYLA':
                sol = pdfo(p1, u0, (model, DOSPts[i, :]), method='COBYLA',
                           bounds=bounds)

        else:
            if method == 'ipopt':
                sol = minimize_ipopt(obj, x0=u0, bounds=bounds,
                                     constraints=(constr))

            elif method == 'DE':
                sol = DE(p1, bounds=bounds, x0=u0, strategy='best1bin',
                         maxiter=2000, workers=-1, updating='deferred',
                         init='sobol', constraints=(constr),
                         args=(model, DOSPts[i, :]))

            elif method == 'dfo':
                sol = pdfo(p1, u0, (model, DOSPts[i, :]), method='COBYLA',
                           bounds=bounds, constraints=constr)

        
        # Append results into fDOS, fDIS and message list for each iteration
        fDOS[i,:] = model(sol.x)
        fDIS[i,:] = sol.x
        message_list.append(sol.message)
        
    
    return fDIS, fDOS, message_list

# Auxiliary functions
# Objective function (Problem type I)
def p1(u: np.ndarray, 
       model:Callable[...,Union[float,np.ndarray]], 
       DOSpt: np.ndarray):
    
    y_found = np.zeros(DOSpt.shape)
    y_found = model(u)
    
    f =  np.sum(list(map(error, y_found, DOSpt)))
    
    return f

# Error minimization function
def error(yf,yd):
    return ((yf-yd)/yd)*((yf-yd)/yd)