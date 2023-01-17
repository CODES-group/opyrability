import polytope as pc
import numpy as np
import scipy as sp
from operability_grid_mapping import AIS2AOS_map, points2simplices, points2polyhedra
from polytope.polytope import region_diff
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from polytope.polytope import _get_patch
from polytope import solvers
import string
from PolyhedraVolAprox import VolumeApprox_fast as Dinh_volume


from scipy.optimize import differential_evolution as DE
from cyipopt import minimize_ipopt
from typing import Callable,Union
from tqdm import tqdm

# Setting default plot optins and default solver for multimodel approach.
plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
solvers.default_solver = 'scipy'



def multimodel_rep(AIS_bound: np.ndarray, 
                  resolution: np.ndarray, 
                  model: Callable[...,Union[float,np.ndarray]]):
    """
    

    Parameters
    ----------
    AIS_bound : np.ndarray
        Bounds on the Available Input Set (AIS). Each row corresponds to the lower
        and upper bound of each AIS variable.
    resolution : np.ndarray
        Array containing the resolution of the discretization grid for the AIS.
        Each element corresponds to the resolution of each variable. For a 
        resolution defined as k, it will generate d^k points (in which d is the
        dimensionality of the AIS).
    model : Callable[...,Union[float,np.ndarray]]
        Process model that calculates the relationship between inputs (AIS-DIS) 
        and Outputs (AOS-DOS).

    Returns
    -------
    finalpolytope : polytope.Region
        Convex polytope or collection of convex polytopes (named as Region) that
        describes the AOS. Can be used to calculate the Operability Index using
        OI_Calc.

    """
    
    # Map AOS from AIS setup.
    AIS, AOS =  AIS2AOS_map(model, AIS_bound, resolution)
    
    ## TODO: Add option to trace simplices or polyhedrons. 
    AIS_poly, AOS_poly = points2simplices(AIS,AOS)
    
    #AIS_poly, AOS_poly = points2polyhedra(AIS,AOS)
    
    
    
    # Define empty polyopes list
    Polytope = list()
    
    # Create convex hull using the vertices of the AOS
    for i in range(len(AOS_poly)):
        Vertices = AOS_poly[i].T
        
        Polytope.append(pc.qhull(Vertices))
        
    
    # Define the AOS as an (possibly) overlapped region. This will be fixed in
    # the next lines of code.    
    overlapped_region = pc.Region(Polytope[0:])
    
    # Create a bounding box of the region above:
    min_coord =  overlapped_region.bounding_box[0]
    max_coord =  overlapped_region.bounding_box[1]
    box_coord =  np.hstack([min_coord, max_coord])
    bound_box =  pc.box2poly(box_coord)

    # Remove overlapping (Gazzaneo's trick) - Remove one polytope at a time, create
    # void polytope using the bounding box and subtract from the original bounding
    # box itself.
    RemPoly = [bound_box]
    for i in range(len(Polytope)):
        # print(i)
        RemPolyList = []
        for j in range(len(RemPoly)+1):
            temp_diff = RemPoly[j-1].diff(Polytope[i])
            
            if str(type(temp_diff)) == "<class 'polytope.polytope.Polytope'>":
                temp_diff = [temp_diff]
            for k in range(len(temp_diff)):
                RemPolyList.append(temp_diff[k])
                
        RemPoly = RemPolyList
    
    RemU =  RemPoly[0]
    for p in range(len(RemPoly)):
        RemU = RemU.union(RemPoly[p])
        
    # Generate final (non-overlapped) polytope
    finalpolytope = region_diff(bound_box, RemU)

    return finalpolytope


def OI_calc(AS: pc.Region,
            DS: np.ndarray, perspective= 'outputs'):
    
    
    
    DS_region =  pc.box2poly(DS)
    AS = pc.reduce(AS)
    DS_region = pc.reduce(DS_region)
    
    
    intersection =  pc.intersect(AS, DS_region)
    
    ## TODO: Different volume approximations
    # if calc == 'polytope':
    #     OI =  (intersection.volume/DS_region.volume)*100
        
    # elif calc=='other':
        
    #     v_intersect = intersection.vertices
    #     A_intersect = intersection.A
    #     b_intersect = intersection.b
        
    #     v_DS =  DS.vertices
    #     A_DS        =  DS.A
    #     b_DS        =  DS.b
        
    #     OI = Dinh_volume(A_intersect, 
    #                      b_intersect, v_intersect) / Dinh_volume(A_DS, 
    #                                                              b_DS, v_DS)
        
        # VolumeApprox_fast(A, b, Vertices)
        
    OI =  (intersection.volume/DS_region.volume)*100
    
    
    
    if perspective == 'outputs':
        DS_label = 'Desired Output Set (DOS)'
        AS_label = 'Achievable Output Set (AOS)'
        int_label = r'$ AOS \cap DOS$'
        
    else:
        DS_label = 'Desired Input Set (DOS)'
        AS_label = 'Available Input Set (AOS)'
        int_label = r'$ AIS \cap DIS$'
            
    
    # Plotting if 2D/ 3D (Future implementation)
    # TODO: 3D plotting
    if DS_region.dim == 2:
        
        polyplot = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        DS_COLOR = "grey"
        INTERSECT_COLOR = "red"
        AS_COLOR = 'b'
        for i in range(len(AS)):
            
            
            polyplot = _get_patch(AS[i], linestyle="dashed", 
                                  edgecolor="black",linewidth=3, 
                                  facecolor =AS_COLOR)
            ax.add_patch(polyplot)
            
            
    
        for j in range(len(intersection)):
     
            
            interplot = _get_patch(intersection[j], linestyle="dashed", 
                                  edgecolor="black",linewidth=3, 
                                  facecolor = INTERSECT_COLOR)
            ax.add_patch(interplot)
    
    
    
        DSplot    = _get_patch(DS_region, linestyle="dashed", 
                              edgecolor="black", alpha=0.5,linewidth=3,
                              facecolor = DS_COLOR)    
        ax.add_patch(DSplot)
        ax.legend('DOS')
    
        lower_xaxis =  min(AS.bounding_box[0][0], DS_region.bounding_box[0][0])
        upper_xaxis =  max(AS.bounding_box[1][0], DS_region.bounding_box[1][0])
    
        lower_yaxis =  min(AS.bounding_box[0][1], DS_region.bounding_box[0][1])
        upper_yaxis =  max(AS.bounding_box[1][1], DS_region.bounding_box[1][1])
    
    
        ax.set_xlim(lower_xaxis - 0.05*lower_xaxis, 
                    upper_xaxis + 0.05*upper_xaxis) 
        ax.set_ylim(lower_yaxis - 0.05*lower_yaxis, 
                    upper_yaxis + 0.05*upper_yaxis)  
    
        
       
    
        DS_patch = mpatches.Patch(color=DS_COLOR, label=DS_label)
        AS_patch = mpatches.Patch(color=AS_COLOR, label=AS_label)
        INTERSECT_patch = mpatches.Patch(color=INTERSECT_COLOR, 
                                         label=int_label)
    
    
    
        OI_str =  'Operability Index = ' + str(round(OI,2)) + str('\%')
    
        extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", 
                                   fill=False, 
                                   edgecolor='none', 
                                   linewidth=0, label= OI_str)
    
        if perspective == 'outputs':
            str_title = string.capwords("Operability Index Evaluation - Outputs' perspective")
            ax.set_title(str_title)
        else:
            str_title = string.capwords("Operability Index Evaluation - Inputs' perspective")
            ax.set_title(str_title)
    
        ax.legend(handles=[DS_patch,AS_patch, INTERSECT_patch, extra])
    
        ax.set_xlabel('$y_{1}$')
        ax.set_ylabel('$y_{2}$')
        plt.show()
        
    else:
        print('Plotting not supported. Dimension different than 2.')
        
        
    return OI






cmap =  'rainbow'
lineweight = 1
edgecolors = 'k'
markersize =  128


def create_grid(region_bounds, region_resolution):
    # Mapping and indexing
    numInput = np.prod(region_resolution)
    nInput = region_bounds.shape[0]
    Input_u = []
    
    # Create discretized region based on bounds and resolution information.
    for i in range(nInput):
        Input_u.append(list(np.linspace(region_bounds[i,0],
                                        region_bounds[i,1],
                                        region_resolution[i])))
    
    
    # Create slack variables for preallocation purposes.    
    region_grid = np.zeros(region_resolution + [nInput])

    for i in range(numInput):
        inputID = [0]*nInput
        inputID[0] = int(np.mod(i, region_resolution[0]))
        region_val = [Input_u[0][inputID[0]]]
        
        for j in range(1,nInput):
            inputID[j] = int(np.mod( np.floor(i/np.prod(region_resolution[0:j])),
                                    region_resolution[j]))
            region_val.append(Input_u[j][inputID[j]])
        
        region_grid[tuple(inputID)] = region_val

    return region_grid
    
    



def nlp_based_approach(DOS_bounds: np.ndarray, 
                       DOS_resolution: np.ndarray,
                       model: Callable[...,Union[float,np.ndarray]], 
                       u0: np.ndarray, 
                       lb: np.ndarray,
                       ub: np.ndarray,
                       constr=None,
                       method: str ='DE', plot: bool =True, ad: bool =False) -> Union[np.ndarray,np.ndarray,list]:
    
   
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
    DOS_bounds : np.ndarray
        Array containing the bounds of the Desired Output set (DOS). Each row 
        corresponds to the lower and upper bound of each variable.
    DOS_resolution: np.ndarray
        Array containing the resolution of the discretization grid for the DOS.
        Each element corresponds to the resolution of each variable. For a 
        resolution defined as k, it will generate d^k points (in which d is the
        dimensionality of the problem).
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
        installed and want to use gradient-based solver. Options are:
            For unconstrained problems:
                -'trust-constr'
                
                -'Nelder-Mead'
                
                -'ipopt'
                
                -'DE'
                
            For constrained problems:
                -'ipopt'
                
                -'DE'
                
                -'SLSQP
                
    plot: bool
        Turn on/off plotting. If dimension is d<=3, plotting is available and
        both the Feasible Desired Output Set (DOS*) and Feasible Desired Input
        Set (DIS*) are plotted. Default is True.

    ad: bool
       Turn on/off use of Automatic Differentiation using JAX. If Jax is installed
       high-order data (jacobians, hessians) are obtained using AD. Default is False.

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
   
    # Use JAX.numpy if differentiable programming is available.
    if ad is False:
        import numpy as np
    else:
        from jax.config import config
        config.update("jax_enable_x64", True)
        import jax.numpy as np
        from jax import jit, grad, jacrev
        constr['fun'] = jit(constr['fun'])
        # con_jac =  jit(jacrev(constr['fun']))
        # constr['jac']  = jit(jacrev(constr['fun']))
        # constr['hess'] = jit(jacrev(jacrev(constr['fun'])))
    
    dimDOS =  DOS_bounds.shape[0]
    DOSPts =  create_grid(DOS_bounds,DOS_resolution)
    DOSPts = DOSPts.reshape(-1, dimDOS)
    
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
    
    
    
    def p1(u: np.ndarray, 
           model:Callable[...,Union[float,np.ndarray]], 
           DOSpt: np.ndarray):
        
        
        y_found = model(u)
        
        vector_of_f = np.array([y_found.T, DOSpt.T])
        f = np.sum(error(*vector_of_f))
        
        return f

    # Error minimization function
    def error(y_achieved,y_desired):
        return ((y_achieved-y_desired)/y_desired)**2
    
    
    # Inverse-mapping: Run for each DOS grid point
    for i in tqdm(range(r)):

        
        
        # This approach is useful for ipopt
        def obj(u):
            return p1(u, model, DOSPts[i,:])
        
        if constr is None:

            if method == 'trust-constr':
                sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                           args=(model, DOSPts[i, :]),
                                           method=method, 
                                           options={'xtol': 1e-10})

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


        else:
            if method == 'ipopt':
                if ad is True:
                    # constr['fun'] = jit(constr['fun'])
                    # obj_jit = jit(obj)
                    # obj_grad = jit(grad(obj_jit))
                    # obj_hess = jit(jacrev(jacrev(obj_jit)))
                    sol = minimize_ipopt(jit(obj), x0=u0, bounds=bounds,
                                        constraints=(constr))
                
                else:
                    sol = minimize_ipopt(obj, x0=u0, bounds=bounds,
                                        constraints=(constr))
                        
                
                

            elif method == 'DE':
                sol = DE(p1, bounds=bounds, x0=u0, strategy='best1bin',
                         maxiter=2000, workers=-1, updating='deferred',
                         init='sobol', constraints=(constr),
                         args=(model, DOSPts[i, :]))
                
            if method == 'SLSQP':
                sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                           args=(model, DOSPts[i, :]),
                                           method=method, constraints=(constr),
                                           options={'xtol': 1e-9, 'ftol': 1e-9})


        
        # Append results into fDOS, fDIS and message list for each iteration
        
        if ad is True:
            fDOS = fDOS.at[i,:].set(model(sol.x))
            fDIS = fDIS.at[i,:].set(sol.x)
        
        else:
            fDOS[i,:] = model(sol.x)
            fDIS[i,:] = sol.x
        
        message_list.append(sol.message)
        
    
    if fDIS.shape[1] > 2:
        plot is False
        print('Plotting not supported. Dimension different than 2.')
    else:
        
        if plot is True:
            plt.subplot(121)
            
           
            plt.rcParams['figure.facecolor'] = 'white'
            plt.scatter(fDIS[:,0],fDIS[:,1], s=16, 
                        c=np.sqrt(fDOS[:,0]**1 + fDOS[:,1]**1), 
                        cmap=cmap, antialiased=True,  
                        lw=lineweight, marker='s', 
                        edgecolors=edgecolors)    
            plt.ylabel('$u_{2}$')
            plt.xlabel('$u_{1}$')
            plt.title('DIS*')
            plt.show
            
            plt.subplot(122)
            
            plt.scatter(fDOS[:,0],fDOS[:,1], s=16, 
                        c=np.sqrt(fDOS[:,0]**1 + fDOS[:,1]**1), 
                        cmap=cmap, antialiased=True,  
                        lw=lineweight, marker='o', 
                        edgecolors=edgecolors)    
            plt.ylabel('$y_{2}$')
            plt.xlabel('$y_{1}$')
            plt.title('DOS*')
            
           
            plt.show
            
        
      
    
    return fDIS, fDOS, message_list
