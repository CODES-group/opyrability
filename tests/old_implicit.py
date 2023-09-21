from typing import Callable,Union

import jax
from jax.config import config
from jax import jacrev
from jax.numpy.linalg import pinv
config.update("jax_enable_x64", True)


from numpy.linalg import norm 
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from scipy.optimize import root
from tqdm import tqdm
# %% REMOVE THIS BEFORE RELEASE
from DMA_MR_ss import *
import matplotlib.pyplot as plt

from opyrability import implicit_map as imap

# %% Functions
def implicit_map(model:             Callable[...,Union[float,np.ndarray]], 
                 image_init:        np.ndarray ,
                 domain_bound:      np.ndarray = None, 
                 domain_resolution: np.ndarray = None, 
                 direction:         str = 'forward', 
                 validation:        str = 'predictor-corrector', 
                 tol_cor:           float = 1e-4, 
                 continuation:      str = 'Explicit RK4',
                 derivative:        str = 'jax',
                 jit:               bool = True,
                 step_cutting:      bool = False,
                 domain_points:     np.ndarray = None):
    '''
    Performs implicit mapping of a implicitly defined process F(u,y) = 0. 
    F can be a vector-valued, multivariable function, which is typically the 
    case for chemical processes studied in Process Operability. 
    This method relies in the implicit function theorem and automatic
    differentiation in order to obtain the mapping of the required 
    input/output space. The
    mapping "direction" can be set by changing the 'direction' parameter.
    
    Author: San Dinh
    Coauthor: Victor Alves
    

    Parameters
    ----------
    implicit_model : Callable[...,Union[float,np.ndarray]]
        Process model that describes the relationship between the input and 
        output spaces. has to be written as a function in the following form:
            F(Input Vector, Output Vector) = 0
    domain_bound : np.ndarray
        Domains of the domain variables. Each row corresponds to the lower and
        upper bound of each variable. The domain terminology corresponds to the
        mathematical definition of the domain of a function: If the mapping is
        in the foward direction (direction = 'forward'), then the domain 
        corresponds to the process inputs. Otherwise, the mapping is in the 
        inverse direction and the domain corresponds to the process outputs.
    domain_resolution : np.ndarray
        Array containing the resolution of the discretization grid for the domain.
        Each element corresponds to the resolution of each variable. For a 
        resolution defined as k, it will generate d^k points (in which d is the
        dimensionality of the domain).
    direction : str, optional
        Mapping direction. this will define if a forward or inverse mapping is 
        performed. The default is 'forward'.
    validation : str, optional
        How the implicit mapping is obatined. The default is 'predictor-corrector'.
        The available options are:
            'predictor' => Numerical integration only.
            
            'corrector' => Solved using a nonlinear equation solver.
            
            'predictor-corrector' => Numerical integration + correction using 
            nonlinear equation solver if F(u, y) >= Tolerance.
        
    tol_cor : float, optional
        Tolerance for solution of the implicit function F(u,y) <= tol_cor. 
        The algorithm will keep solving the implicit mapping while
        F(u,y) >= tol_cor. The default is 1e-4.
    continuation : str, optional
        Numerical continuation method used in the implicit mapping.
        The default is 'Explicit RK4' .
        
        'Explicit Euler' - Euler's method. Good for simple (nonstiff) problems.
        
        'Explicit RK4'   - Fixed-step 4th order Runge-Kutta. Good for moderate,
        mildly stiff problems. Good balance between accuracy and complexity.
        
        'Odeint'         - Variable step Runge-Kutta. Suitable for challenging,
        high-dimensional and stiff problems. This is a fully-featured IVP solver.

    derivative: str, optional
        Derivative calculation method. If JAX is available, automatic
        differentiation can be performed. The default is 'jax'.
    jit:  bool True, optional
        JAX's Just-in-time compilation (JIT) of implicit function and its 
        respective multidimensional derivatives (Jacobians). JIT allows faster
        computation of the implicit map. The default is 'True'.
    step_cutting:      bool, False, optional
        Cutting step strategy to subdivide the domain/image in case of stifness.
        The default is 'False'.
    
    Returns
    -------
    domain_set: np.ndarray
        Set of values that corresponds to the domain of the implicit function.
    image_set: np.ndarray
        Set of values that corresponds to the calculated image of the implicit
        function., 
    domain_polyhedra:  list
        Set of coordinates for polytopic construction of the domain. Can be used
        to create polytopes for multimodel representation and/or OI evaluation
        (see multimodel_rep and oi_calc modules)
    image_polyhedra: list
        Set of coordinates for polytopic construction of the calculated image
        of the implicit function. Similarly to 'domain_polyhedra', this list can
        be used to construct polytopes for multimodel representation and OI
        evaluation.

    '''
    
    # Implicit function theorem and pre-configuration steps.
    if direction == 'forward':
        print('Forward Mapping Selected.')
        print('The given domain is recognized as an Available Input Set (AIS).')
        print('The result of this mapping is an Achievable Output Set(AOS)')

        def F(i, o): return model(i, o)
        def F_io(o, i): return model(i, o)
    elif direction == 'inverse':
        print('Inverse Mapping Selected.')
        print('The given domain is recognized as Desired Output Set (DOS).')
        print('The result of this mapping is an Desired Input Set(DIS)')

        def F(i, o): return model(o, i)
        def F_io(o, i): return model(o, i)
    else:
        print('Invalid Mapping Selected. Please select the direction \
              to be either "forward" or "inverse"')
        

    dFdi = jacrev(F, 0)
    dFdo = jacrev(F, 1)

    if jit:
        @jax.jit
        def dodi(ii,oo):
            return -pinv(dFdo(ii,oo)) @ dFdi(ii,oo)
        
        @jax.jit
        def dods(oo, s, s_length, i0, iplus):
            return dodi(i0 + (s/s_length)*(iplus - i0), oo) \
                @((iplus - i0)/s_length)
    else:
        def dodi(ii, oo): return -pinv(dFdo(ii, oo)) @ dFdi(ii, oo)
        def dods(oo, s, s_length, i0, iplus): return dodi(
            i0 + (s/s_length)*(iplus - i0), oo)@((iplus - i0)/s_length)

        
    # #  Initialization step: obtaining first solution
    # sol = root(F_io, image_init,args=domain_bound[:,0])
    
    #  Predictor scheme selection
    if continuation == 'Explicit RK4':
        print('Selected RK4')
        predict = predict_RK4
        do_predict = dodi
    elif continuation == 'Explicit Euler':
        print('Selected Euler')
        predict = predict_eEuler
        do_predict = dodi
    elif continuation == 'odeint':
        print('Selected odeint')
        predict = predict_odeint
        do_predict = dods
    else:
        print('Incorrect continuation method')
        
    
    
    if domain_points is not None:
        # Determine dimension
        d = domain_points.shape[1]
        
        # Calculate side length of grid (assumes cubic/square grid)
        side_length = int(domain_points.shape[0] ** (1/d))
        
        # Reshape into the desired shape
        domain_set = domain_points.reshape(*([side_length]*d), d)
    
        # Update other dependent parameters
        nInput = domain_set.shape[-1]
        numInput = np.prod([side_length]*d)
        domain_resolution = [side_length]*d  # inferred resolution
        image_set = np.zeros(domain_resolution + [nInput])*np.nan
        #  Initialization step: obtaining first solution
        sol = root(F_io, image_init,args=domain_points[:,0])
        image_set[0, 0] = sol.x
        nOutput = image_init.shape[0]
        
        for i in range(numInput):
            inputID = [0]*nInput
            inputID[0] = int(np.mod(i, domain_resolution[0]))
    
    else:
        # Pre-alocating the domain set
        numInput = np.prod(domain_resolution)
        nInput = domain_bound.shape[0]
        nOutput = image_init.shape[0]
        Input_u = []

        # Create discretized AIS based on bounds and resolution information.
        for i in range(nInput):
            Input_u.append(list(np.linspace(domain_bound[i, 0],
                                            domain_bound[i, 1],
                                            domain_resolution[i])))

        domain_set = np.zeros(domain_resolution + [nInput])
        image_set = np.zeros(domain_resolution + [nInput])*np.nan
        #  Initialization step: obtaining first solution
        sol = root(F_io, image_init,args=domain_bound[:,0])
        image_set[0, 0] = sol.x

        for i in range(numInput):
            inputID = [0]*nInput
            inputID[0] = int(np.mod(i, domain_resolution[0]))
            domain_val = [Input_u[0][inputID[0]]]

            for j in range(1, nInput):
                inputID[j] = int(np.mod(np.floor(i/np.prod(domain_resolution[0:j])),
                                        domain_resolution[j]))
                domain_val.append(Input_u[j][inputID[j]])

            domain_set[tuple(inputID)] = domain_val
        

    
    # # Pre-alocating the domain set
    # numInput = np.prod(domain_resolution)
    # nInput = domain_bound.shape[0]
    # nOutput = image_init.shape[0]
    # Input_u = []

    # # Create discretized AIS based on bounds and resolution information.
    # for i in range(nInput):
    #     Input_u.append(list(np.linspace(domain_bound[i, 0],
    #                                     domain_bound[i, 1],
    #                                     domain_resolution[i])))

    # domain_set = np.zeros(domain_resolution + [nInput])
    # image_set = np.zeros(domain_resolution + [nInput])*np.nan
    # image_set[0, 0] = sol.x

    # for i in range(numInput):
    #     inputID = [0]*nInput
    #     inputID[0] = int(np.mod(i, domain_resolution[0]))
    #     domain_val = [Input_u[0][inputID[0]]]

    #     for j in range(1, nInput):
    #         inputID[j] = int(np.mod(np.floor(i/np.prod(domain_resolution[0:j])),
    #                                 domain_resolution[j]))
    #         domain_val.append(Input_u[j][inputID[j]])

    #     domain_set[tuple(inputID)] = domain_val

    cube_vertices = list()
    for n in range(2**(nInput)):
        cube_vertices.append([int(x) for x in
                              np.binary_repr(n, width=nInput)])
    
    # Preallocate domain and image polyhedras
    domain_polyhedra = list()
    image_polyhedra = list()


    # domain_polyhedra = list()
    # image_polyhedra = list()

    for i in tqdm(range(numInput)):
        inputID[0] = int(np.mod(i, domain_resolution[0]))

        for j in range(1, nInput):
            inputID[j] = int(np.mod(np.floor(i/np.prod(domain_resolution[0:j])),
                                    domain_resolution[j]))

        if np.prod(inputID) != 0:
            inputID = [x-1 for x in inputID]
            ID = np.array([inputID]*(2**(nInput))) + np.array(cube_vertices)

            V_domain_id = np.zeros((nInput, 2**(nInput)))
            V_image_id = np.zeros((nOutput, 2**(nInput)))
            for k in range(2**(nInput)):
                ID_cell = tuple([int(x) for x in ID[k]])

                if k == 0:
                    domain_0 = domain_set[ID_cell]
                    image_0 = image_set[ID_cell]
                    V_domain_id[:, k] = domain_0
                    V_image_id[:, k] = image_0

                else:
                    if validation == 'predictor-corrector':
                        if (np.isnan(np.prod(image_set[ID_cell])) 
                            and np.isnan(np.prod(image_0)) == False):
                            
                            domain_k = domain_set[ID_cell]
                            V_domain_id[:, k] = domain_k
                            if step_cutting == True:
                                test_img_k = predict_eEuler(do_predict, 
                                                            domain_0,
                                                            domain_k, 
                                                            image_0)
                                domain_k_test = domain_k
                                count = 1
                                condition = max(
                                    abs(F(domain_k_test, test_img_k)))
                                while ((condition > tol_cor or 
                                        np.isnan(condition)) 
                                       and count < 10):

                                    count = count + 1
                                    test_img_k = predict_eEuler(do_predict, 
                                                                domain_0,
                                                                domain_k_test, 
                                                                image_0)
                                    domain_k_test = (domain_k_test*(count-1) \
                                                     + domain_0) / count

                                    condition = max(
                                        abs(F(domain_k_test, test_img_k)))

                                
                                domain_k_step_size = domain_k_test
                                domain_kk_minus = domain_0
                                image_kk_minus = image_0
                                if count < 10:
                                    for kk in range(count):
                                        
                                        domain_kk = domain_0 + \
                                        domain_k_step_size*count
                                        
                                        image_kk = predict(do_predict, 
                                                           domain_kk_minus, 
                                                           domain_kk, 
                                                           image_kk_minus)
                                        image_kk_minus = image_kk

                                image_k = image_kk_minus
                            else:
                                image_k = predict(do_predict, 
                                                  domain_0, 
                                                  domain_k, 
                                                  image_0)

                            max_residual = max(abs(F(domain_k, image_k)))
                            if (max_residual**2 > tol_cor or 
                                np.isnan(max_residual)):
                                
                                # Call the corrector:
                                sol = root(F_io, image_0, args=domain_k)
                                found_sol = sol.success
                                
                                # Treat case in which solution is not found:
                                if found_sol:
                                    image_k = sol.x
                                else:
                                    image_k = np.nan
                                    
                            
                            image_set[ID_cell] = image_k
                            V_image_id[:, k] = image_k
                        else:
                            domain_k = domain_set[ID_cell]
                            V_domain_id[:, k] = domain_k
                            image_k = image_set[ID_cell]
                            V_image_id[:, k] = image_k
                            

                    elif validation == 'predictor': 
                        if np.isnan(np.prod(image_set[ID_cell])):
                            domain_k = domain_set[ID_cell]
                            V_domain_id[:,k] = domain_k
                            
                            image_k = predict(do_predict,
                                              domain_0,
                                              domain_k,
                                              image_0)
                                                                                     
                            image_set[ID_cell] = image_k
                            V_image_id[:,k] = image_k
                            
                    elif validation == 'Corrector':     
                        domain_k = domain_set[ID_cell]
                        V_domain_id[:,k] = domain_k
                        
                        sol = root(F_io, image_0, args=domain_k)
                        image_k = sol.x
                        
                        image_set[ID_cell] = image_k
                        V_image_id[:,k]    = image_k
                        
            domain_polyhedra.append(V_domain_id)
            image_polyhedra.append(V_image_id)
            
    return domain_set, image_set, domain_polyhedra, image_polyhedra

# Numerical Integration (continuation) methods

def predict_odeint(dods, i0, iplus ,o0):
    s_length = norm(iplus - i0)
    s_span = jnp.linspace(0.0, s_length, 10)
    sol = odeint(dods, o0, s_span, s_length, i0, iplus)
    return sol[-1,:]


def predict_RK4(dodi,i0, iplus ,o0):
    h = iplus -i0
    k1 = dodi( i0          , o0           )
    k2 = dodi( i0 + (1/2)*h, o0 + (h/2)@k1)
    k3 = dodi( i0 + (1/2)*h, o0 + (h/2)@k1)
    k4 = dodi( jnp.array(i0 +       h), o0 +       (h)@k3)
    return o0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)@h

def predict_eEuler(dodi,i0, iplus ,o0):
    return o0 + dodi(i0,o0)@(iplus -i0)

# %% Test area! REMOVE BEFORE RELEASE!!!!
def shower_implicit(u,y):
    d = jnp.zeros(2)
    y0_aux = (u[0]+u[1])
    
    y0_aux= jnp.where(y0_aux <= 1e-9, 1e-9, y0_aux)
    
    
    LHS1 = y[0] - (u[0]+u[1])
    LHS2 = y[1] - (u[0]*(60+d[0])+u[1]*(120+d[1]))/(y0_aux)
    
    
    # LHS2_AX = y[1] - (60+120)/2
    LHS2_AX = y[1] - (60+120)/2
    LHS2 = jnp.where(y0_aux <= 1e-9, LHS2_AX, LHS2)
    
    # if y[0]!=0:
    #     LHS2 = y[1] - (u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    # else:
        
    
    return jnp.array([LHS1, LHS2])



def shower(u):
    y = np.zeros(2)
    d = jnp.zeros(2)
    y[0] = (u[0]+u[1])
    # LHS2 = y[1] - (u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    if y[0]!=0:
        y[1] = (u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    else:
        y[1] = (60+120)/2
    
    return jnp.array(y)

def FF1(u):
    y = np.zeros(2)
    # y[0] = u[1]**2*u[0]
    y[0] = u[0] - 2*u[1]
    y[1] = 3*u[0] + 4*u[1]
    return jnp.array(y)

def FF1_implicit(u,y):
    # LHS0 = y[0] - u[1]**2*u[0]
    LHS0 = y[0] - (u[0] - 2*u[1])
    LHS1 = y[1] - (3*u[0] + 4*u[1])
    return jnp.array([LHS0, LHS1])



from matplotlib.patches import Polygon

def makeplot(AOS_poly):
    fig, ax = plt.subplots()
    for i in range(len(AOS_poly)):
        x = AOS_poly[i][0]
        y = AOS_poly[i][1]
        order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
        A = AOS_poly[i].T
        polygon = Polygon(A[order] ,facecolor='tab:blue', edgecolor='black', linewidth=1)
        ax.add_patch(polygon)
    ax.autoscale_view()

#%% Test DMA-MR inverse
# DOS_bound = np.array([[22.4, 22.8],
#                     [39.4, 40.0]])

# DOSresolution = [10, 10]

# output_init = np.array([20.0, 0.9])

# DOS, DIS, DOS_poly, DIS_poly = imap(F_DMA_MR_eqn,
#                                     DOS_bound,
#                                     DOSresolution,
#                                     output_init,
#                                     direction='inverse')

# %% Test shower forward
# AIS_bound = np.array([[0.1, 10.0],
#                     [0.1, 10.0]])

# # AIS_bound =  np.array([[5.0, 10.0],
# #                       [80.0, 100.0]])

# AISresolution = [5, 5]

# output_init = np.array([0.0, 10.0])


# theta = np.linspace(0.01, 4 * np.pi, 400)

# phi = np.pi / 4
# a, b= 0.15, 1
# y1 = 5 + np.exp(-0.05 / theta) * (a * np.cos(theta) * np.cos(phi) + 
#                                   b * np.sin(theta) * np.sin(phi))  
# y2 = 5 + np.exp(-0.05 / theta) * (b * np.sin(theta) * np.cos(phi) - 
#                                   a * np.cos(theta) * np.cos(phi))


# AIS_PTS=np.array([y1,y2]).T
# AIS_PTS = data = np.array([
#     [0.1, 0.1],
#     [0.1, 2.575],
#     [0.1, 5.05],
#     [0.1, 7.525],
#     [0.1, 10],
#     [2.575, 0.1],
#     [2.575, 2.575],
#     [2.575, 5.05],
#     [2.575, 7.525],
#     [2.575, 10],
#     [5.05, 0.1],
#     [5.05, 2.575],
#     [5.05, 5.05],
#     [5.05, 7.525],
#     [5.05, 10],
#     [7.525, 0.1],
#     [7.525, 2.575],
#     [7.525, 5.05],
#     [7.525, 7.525],
#     [7.525, 10],
#     [10, 0.1],
#     [10, 2.575],
#     [10, 5.05],
#     [10, 7.525],
#     [10, 10]
# ])


# t1 = time.time()
# AIS, AOS, AIS_poly, AOS_poly = implicit_map(shower_implicit,  
#                                             output_init,
#                                             continuation='odeint',
#                                             domain_points=AIS_PTS)


# from opyrability import implicit_map as imap

# AIS, AOS, AIS_poly, AOS_poly = imap(shower_implicit, 
#                                             AIS_bound, 
#                                             AISresolution, 
#                                             output_init,
#                                             continuation='odeint')

# AIS_infeas_plot = np.reshape(AIS,(-1,2))
# AOS_infeas_plot = np.reshape(AOS,(-1,2))

# fig3, ax3 = plt.subplots()
# ax3.scatter(AIS_infeas_plot[:,0], AIS_infeas_plot[:,1])

# fig4, ax4 = plt.subplots()
# ax4.scatter(AOS_infeas_plot[:,0], AOS_infeas_plot[:,1])

# %% DMA-MR uncertainty - Forward
# phi = np.pi / 4
# a, b= 150, 500
# theta = np.linspace(0, 3 * np.pi, 100)
# y1 = 1173.15  + np.exp(-0.05 / theta) * (a * np.cos(theta) * np.cos(phi) + 
#                                   b * np.sin(theta) * np.sin(phi))  
# y2 = 1500.00   + np.exp(-0.05 / theta) * (b * np.sin(theta) * np.cos(phi) - 
#                                   a * np.cos(theta) * np.cos(phi))
# # 
# # # Center coordinates
# # h, k = 1173.15 , 1500
# # a, b = 500, 450  # Adjust a and b as needed

# # # # h, k = 1173.15 , 101325.0
# # # # a, b = 500, 450  # Adjust a and b as needed

# # # # # h, k = 4   , 56.38
# # # # # a, b = 2, 20  # Adjust a and b as needed
# # alpha = np.pi / 4  # 45 degree rotation, adjust as needed
# # theta = np.linspace(0, 2 * np.pi, 100)

# # Ellipse equations
# # x = a * np.cos(theta)
# # y = b * np.sin(theta)

# # # Rotated ellipse equations centered at (1500, 0.0036)
# # y1 = h + (x * np.sin(alpha) + y * np.cos(alpha))
# # y2 = k + (x * np.cos(alpha) - y * np.sin(alpha))


# AIS_PTS=np.array([y1,y2]).T
# # from matplotlib import pyplot as plt
# plt.plot(AIS_PTS[:,0], AIS_PTS[:,1])
# output_init = np.array([22.4, 39.4])
# AIS, AOS, AIS_poly, AOS_poly = implicit_map(dma_mr_uncertain,  
#                                             output_init,
#                                             continuation='Explicit RK4',
#                                             domain_points=AIS_PTS)


# AOS_PTS = AOS.reshape(-1,2)
# from matplotlib import pyplot
# pyplot.figure()
# plt.plot(AOS_PTS[:,0], AOS_PTS[:,1])

# %% DMA-MR uncertainty - Inverse
phi = np.pi / 4
a, b= 20265.00, 10132.50
theta = np.linspace(0, 3 * np.pi, 100)
y1 = 101325.00  + np.exp(-0.05 / theta) * (a * np.cos(theta) * np.cos(phi) + 
                                  b * np.sin(theta) * np.sin(phi))  
y2 = 101325.00   + np.exp(-0.05 / theta) * (b * np.sin(theta) * np.cos(phi) - 
                                  a * np.cos(theta) * np.cos(phi))


# phi = np.pi / 4
# a, b= 0.01, 0.085
# theta = np.linspace(0, 3 * np.pi, 100)
# y1 = 21.00  + np.exp(-0.05 / theta) * (a * np.cos(theta) * np.cos(phi) + 
#                                   b * np.sin(theta) * np.sin(phi))  
# y2 = 35.00  + np.exp(-0.05 / theta) * (b * np.sin(theta) * np.cos(phi) - 
#                                   a * np.cos(theta) * np.cos(phi))
# 
# # Center coordinates
# h, k = 1173.15 , 1500
# a, b = 500, 450  # Adjust a and b as needed

# # # h, k = 1173.15 , 101325.0
# # # a, b = 500, 450  # Adjust a and b as needed

# # # # h, k = 4   , 56.38
# # # # a, b = 2, 20  # Adjust a and b as needed
# alpha = np.pi / 4  # 45 degree rotation, adjust as needed
# theta = np.linspace(0, 2 * np.pi, 100)

# Ellipse equations
# x = a * np.cos(theta)
# y = b * np.sin(theta)

# # Rotated ellipse equations centered at (1500, 0.0036)
# y1 = h + (x * np.sin(alpha) + y * np.cos(alpha))
# y2 = k + (x * np.cos(alpha) - y * np.sin(alpha))


AIS_PTS=np.array([y1,y2]).T
# from matplotlib import pyplot as plt
plt.plot(AIS_PTS[:,0], AIS_PTS[:,1])
output_init = np.array([22.4, 39.4])
# output_init = np.array([101325.0 , 101325.0])
AIS, AOS, AIS_poly, AOS_poly = imap(dma_mr_uncertain_inv,  
                                            output_init,
                                            continuation='Explicit RK4',
                                            domain_points=AIS_PTS,
                                            direction = 'forward')


AOS_PTS = AOS.reshape(-1,2)
from matplotlib import pyplot
pyplot.figure()
plt.plot(AOS_PTS[:,0], AOS_PTS[:,1])

# %% DMA-MR inverse (Broken)

# DOS_bound = np.array([[15.0, 25.0],
#                     [35.0, 45.0]])


# DOS_bound = np.array([[22.4, 23],
#                     [39.4, 42.0]])

# DOSresolution = [5, 5]

# output_init = np.array([50.0, 2.0])

# t2 = time.time()
# AIS, AOS, AIS_poly, AOS_poly = imap(F_DMA_MR_eqn, 
#                                             DOS_bound, 
#                                             DOSresolution, 
#                                             output_init,
#                                             continuation='Explicit RK4',
#                                             direction='inverse')

# elapsed_RK4 = time.time() -  t2

# makeplot(AOS_poly)

# %%
# elapsed_odeint = time.time() -  t1

# print('ODEINT time (s)')
# print(elapsed_odeint)


# t2 = time.time()
# AIS, AOS, AIS_poly, AOS_poly = imap(shower_implicit, 
#                                             AIS_bound, 
#                                             AISresolution, 
#                                             output_init,
#                                             continuation='Explicit RK4')

# elapsed_RK4 = time.time() -  t2

# print('RK4 time (s)')
# print(elapsed_RK4)


# # %%
# DOS_bound = np.array([[22.4, 23],
#                     [39.4, 42.0]])

# DOS_bound = np.array([[22.0, 25.0],
#                     [39.5, 45.0]])

# DOS_bound = np.array([[22.0, 32.0],
#                     [39.5, 49.0]])

# DOS_bound = np.array([[22.4, 22.8],
#                     [39.4, 40.0]])



# DOSresolution =  [5, 5]

# output_init = np.array([20.0, 0.9])

# %%
# t2 = time.time()
# AIS, AOS, AIS_poly, AOS_poly = imap(F_DMA_MR_eqn, 
#                                             DOS_bound, 
#                                             DOSresolution , 
#                                             output_init,
#                                             continuation='Explicit RK4',
#                                             direction='inverse',
#                                             validation = 'predictor-corrector')
# elapsed_RK4 = time.time() -  t2

# print('RK4 time (s)')
# print(elapsed_RK4)

# AIS_plot = np.reshape(AIS,(-1,2))
# AOS_plot = np.reshape(AOS,(-1,2))

# fig1, ax1 = plt.subplots()
# ax1.scatter(AIS_plot[:,0], AIS_plot[:,1])

# fig2, ax2 = plt.subplots()
# ax2.scatter(AOS_plot[:,0], AOS_plot[:,1])

# %%
# # t1 = time.time()
# AIS, AOS, AIS_poly, AOS_poly = imap(F_DMA_MR_eqn, 
#                                             DOS_bound, 
#                                             DOSresolution , 
#                                             output_init,
#                                             continuation='odeint',
#                                             direction= 'inverse')

# elapsed_odeint = time.time() -  t1

# print('ODEINT time (s)')
# print(elapsed_odeint)

