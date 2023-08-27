import jax.numpy as np
from opyrability import multimodel_rep, OI_eval

v0 = 1
Ca0 = 1
H = 1
Fa0 = v0 *  Ca0

def CSTR(u):
    
    """
    CSTR Example from: 
    Alves, V, Kitchin, JR, Lima, FV. An inverse mapping approach for process 
    systems engineering using automatic differentiation and the implicit 
    function theorem. AIChE J. 2023; 69(9):e18119. doi:10.1002/aic.18119
    
    
    Parameters
    ----------
    u : np.ndarray
        Vector containing the AIS variables. First variable is the CSTR radius
        and the second variable is the dimensionless temperature, RT.

    Returns
    -------
    y: np.ndarray
        Vector containing the AOS/DOS variables. They are concentration of A 
        and B, respectively.

    """
    # Radius and dimensionless temperature.
    R  = u[0]
    RT = u[1]
    
    volume = np.pi * R **2  * H
    
    # Pre exp. factors.
    k1 = np.exp(-3 / RT)
    k2 = np.exp(-10 / RT)

    # Used in CA evaluation.
    a =  volume * k1
    b = v0 + volume*k2
    c = - v0 * Ca0
    
    # Concentrations of A and B.
    Ca = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    Cb = k1 * Ca**2 * volume / v0
    
    # AOS/DOS outputs as a vector.
    y = np.array([Ca, Cb])
    
    return y

# Obtaining OI and inverse mapping.

# AIS Bounds
AIS_bounds = np.array([[0.25,  3.00],
                       [1.00, 15.00]])
# Discretization Resolution.
AIS_resolution = [25, 25]

# Obtain AOS.
AOS_region = multimodel_rep(CSTR, 
                            AIS_bounds, 
                            AIS_resolution)

# DOS Bounds.
DOS_bounds = np.array([[0.10,   0.35],
                       [0.45,   0.65]])
# Obtain Operability Index (OI).
OI = OI_eval(AOS_region, DOS_bounds)


from opyrability import nlp_based_approach

# Lower and upper bounds on the AIS/DIS*, initial estimate as well.
u0 = np.array([3 , 10])

lb = np.array([0.25, 3])
                  
ub = np.array([50, 50])

# Discretization resolution.
DOS_resolution = [20, 20]
    
# Obtain DIS*/DOS*, since the model as built in JAX, AD can be used.
fDIS, fDOS, message = nlp_based_approach(CSTR, 
                                        DOS_bounds,
                                        DOS_resolution,
                                        u0,
                                        lb,
                                        ub,
                                        method='ipopt',
                                        ad=True,
                                        warmstart=True)