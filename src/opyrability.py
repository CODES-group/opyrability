# Basic python tools
import sys
import warnings
from itertools import permutations as perms
import string
from typing import Callable,Union
from tqdm import tqdm

# Linear Algebra
import numpy as np
from   numpy.linalg import norm , pinv

# Optimization algorithms
import scipy as sp
from scipy.optimize import root
from scipy.optimize import differential_evolution as DE
from cyipopt import minimize_ipopt

# Polytopic calculations
import polytope as pc
from polytope.polytope import _get_patch

# Plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Setting default plot options
plt.rcParams['figure.dpi'] = 150
cmap =  'rainbow'
lineweight = 1
edgecolors = 'k'
markersize =  128
DS_COLOR = '#7f7f7f'
INTERSECT_COLOR = '#0000cc'
AS_COLOR = '#1cff1c'
EDGES_COLORS = '#000000'
EDGES_WIDTH = 0.25


def multimodel_rep(model: Callable[...,Union[float,np.ndarray]], 
                  bounds: np.ndarray, 
                  resolution: np.ndarray,
                  polytopic_trace: str = 'simplices',
                  perspective: str = 'outputs',
                  plot: str = True,
                  EDS_bound: str = None,
                  EDS_resolution: str = None):
    
    """
    Obtain a multimodel representation based on polytopes of Process Operability
    sets. This procedure is essential for evaluating the Operability Index (OI).
    
    This function is part of Python-based Process Operability package.
    
    Control, Optimization and Design for Energy and Sustainability (CODES) 
    Group - West Virginia University - 2023
    
    Author: Victor Alves

    Parameters
    ----------
    model : Callable[...,Union[float,np.ndarray]]
        Process model that calculates the relationship between inputs (AIS-DIS) 
        and Outputs (AOS-DOS).
    bounds : np.ndarray
        Bounds on the Available Input Set (AIS), or Desired Output Set (DOS) if
        an inverse mapping is chosen. Each row corresponds to the 
        lower and upper bound of each AIS or DOS variable.
    resolution : np.ndarray
        Array containing the resolution of the discretization grid for the AIS or
        DOS.Each element corresponds to the resolution of each variable. For a 
        resolution defined as k, it will generate d^k points (in which d is the
        dimensionality of the AIS or DOS).
    polytopic_trace: str, Optional.
        Determines if the polytopes will be constructed using simplices or
        polyhedrons. Default is 'simplices'. Additional option is 'polyhedra'.
    perspective: str, Optional.
        Defines if the calculation is to be done from the inputs/outputs
        perspective. Affects only labels in plots. Default is 'outputs'.
    plot: str, Optional.
        Defines if the plot of operability sets is desired (If the dimension
        is <= 3). Default is 'True'.
    EDS_bound : np.ndarray
        Lower and upper bounds for the Expected Disturbance Set (EDS). Default
        is 'None'.
    EDS_resolution : np.ndarray
        Resolution for the Expected Disturbance Set (EDS). This will be used to
        discretize the EDS, similar to the AIS_resolution   
        

    Returns
    -------
    mapped_region : list
        List with first argument being a convex polytope or collection of 
        convex polytopes (named as Region) 
        that describes the AOS. Can be used to calculate the Operability Index 
        using ``OI_eval``. Second argument is the coordinates of the AOS as a
        numpy array.
        
    
    References
    ----------
    [1]  D. R., Vinson and C. Georgakis.  New Measure of Process Output 
    Controllability. J. Process Control, 2000. 
    https://doi.org/10.1016/S0959-1524(99)00045-1
    
    [2]  C. Georgakis, D. Uztürk, S. Subramanian, and D. R. Vinson. On the 
    Operability of Continuous Processes. Control Eng. Pract., 2003. 
    https://doi.org/10.1016/S0967-0661(02)00217-4
    
    [3] V.Gazzaneo, J. C. Carrasco, D. R. Vinson, and F. V. Lima. Process 
    Operability Algorithms: Past, Present, and Future Developments.
    Ind. & Eng. Chem. Res. 59 (6), 2457-2470, 2020.
    https://doi.org/10.1021/acs.iecr.9b05181
                                 
    """
    
    # TODO: Add implicit mapping option to perform any multimodel rep 
    #(future release).
    # Map AOS from AIS setup.
    
    # AIS, AOS =  AIS2AOS_map(model, 
    #                         AIS_bound, 
    #                         resolution, 
    #                         EDS_bound=EDS_bound,
    #                         EDS_resolution=EDS_resolution,
    #                         plot = False)
    
    # If it is a forward map, using the forward mapping function (AIS2AOS_map).
    # If not, using the NLP-based approach for inverse mapping. In this case,
    # an initial estimate is needed and the user is prompted.
    if perspective == 'outputs':
        AIS, AOS =  AIS2AOS_map(model, 
                                bounds, 
                                resolution,
                                EDS_bound= EDS_bound,
                                EDS_resolution=EDS_resolution, 
                                plot= False)
    else:
        u0_input = input('Enter an initial estimate for your inverse model'
                         'separated only by commas (,) : ')
        
        input_list = [float(u0_input) for u0_input in u0_input.split(',')]
        
        u0 = np.array(input_list)
        
        AIS, AOS, _ = nlp_based_approach(model, bounds, resolution, 
                                                  u0, 
                                                  -np.inf*np.ones(u0.shape), 
                                                  +np.inf*np.ones(u0.shape), 
                                                  method='ipopt', 
                                                  plot=False, 
                                                  ad=False,
                                                  warmstart=True)
        
        # Reshape (n^k, k) vectors into (list[k*n, k]) multidimensional arrays. 
        # This makes polytopic tracing calculations more "clear". 
        AIS = AIS.reshape((resolution + [AIS.shape[-1]]))
        AOS = AIS.reshape((resolution + [AOS.shape[-1]]))
    
    # Switch in between for simplicial of polyhedra calculations.
    if  polytopic_trace  =='simplices':
        AIS_poly, AOS_poly = points2simplices(AIS,AOS)
    elif polytopic_trace =='polyhedra':
        AIS_poly, AOS_poly = points2polyhedra(AIS,AOS)
    else:
        print('Invalid option for polytopic tracing. Exiting algorithm.')
        sys.exit()
        
        
    # Define empty polyopes list.
    Polytopes = list()
    Vertices_list = list()
    # Create convex hull using the vertices of the AOS.
    for i in range(len(AOS_poly)):
        Vertices = AOS_poly[i]
        Vertices_list.append(Vertices)
        Polytopes.append(pc.qhull(Vertices))
        
    
    # Define the AOS as an (possibly) overlapped region. Here we don't need to
    # worry about this here for now, only when evaluating the OI.
    mapped_region = pc.Region(Polytopes[0:])
    
    
    # Perspective switch: This will only affect plot and legends.
    if perspective == 'outputs':
        AS_label = 'Achievable Output Set (AOS)'

    else:
        AS_label = 'Available Input Set (AIS)'

    # Plots (2D/ 3D), unfortunately humans can't see higher dimensions. :(
    if plot is True:
        if mapped_region.dim == 2:
            
            
            polyplot = []
            fig = plt.figure()
            ax = fig.add_subplot(111)
            AS_coords = np.concatenate(Vertices_list, axis=0)
            for i in range(len(mapped_region)):

                polyplot = _get_patch(mapped_region[i], linestyle="solid",
                                    edgecolor=EDGES_COLORS, linewidth=EDGES_WIDTH,
                                    facecolor=AS_COLOR)
                ax.add_patch(polyplot)


            lower_xaxis = mapped_region.bounding_box[0][0]
            upper_xaxis = mapped_region.bounding_box[1][0]

            lower_yaxis = mapped_region.bounding_box[0][1]
            upper_yaxis = mapped_region.bounding_box[1][1]

            ax.set_xlim(lower_xaxis - 0.05*lower_xaxis,
                        upper_xaxis + 0.05*upper_xaxis)
            ax.set_ylim(lower_yaxis - 0.05*lower_yaxis,
                        upper_yaxis + 0.05*upper_yaxis)

            
            AS_patch = mpatches.Patch(color=AS_COLOR, label=AS_label)

            extra = mpatches.Rectangle((0, 0), 1, 1, fc="w",
                                    fill=False,
                                    edgecolor='none',
                                    linewidth=0)

            if perspective == 'outputs':
                str_title = 'Achievable Output Set (AOS)'
                ax.set_title(str_title)
            else:
                str_title ='Available Input set (AIS)'
                ax.set_title(str_title)

            ax.legend(handles=[AS_patch, extra])

            ax.set_xlabel('$y_{1}$')
            ax.set_ylabel('$y_{2}$')
            plt.show()
            
        elif mapped_region.dim == 3:
            
            
            polyplot = []
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            AS_coords = np.concatenate(Vertices_list, axis=0)
            
            for cube, color in zip([AS_coords], [AS_COLOR]):
                hull = sp.spatial.ConvexHull(cube)
                # Draw the polygons of the convex hull
                for s in hull.simplices:
                    tri = Poly3DCollection([cube[s]])
                    tri.set_color(color)
                    tri.set_alpha(0.5)
                    tri.set_edgecolor(EDGES_COLORS)
                    ax.add_collection3d(tri)
                # Draw the vertices
                ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], marker='o', 
                           color='None')
            plt.show()

            
            AS_patch = mpatches.Patch(color=AS_COLOR, label=AS_label)

            extra = mpatches.Rectangle((0, 0), 1, 1, fc="w",
                                    fill=False,
                                    edgecolor='none',
                                    linewidth=0)

            if perspective == 'outputs':
                str_title = 'Achievable Output Set (AOS)'
                ax.set_title(str_title)
            else:
                str_title = 'Available Input Set (AIS)'
                ax.set_title(str_title)

            ax.legend(handles=[AS_patch, extra])

            ax.set_xlabel('$y_{1}$')
            ax.set_ylabel('$y_{2}$')
            ax.set_zlabel('$y_{3}$')
            plt.show()
            

        else:
            print('plot not supported. Dimension different greater than 3.')
            AS_coords = np.concatenate(Vertices_list, axis=0)

    else:
        print('Either plot is not possible (dimension > 3) or you have',
              'chosen plot=False. The operability set is still returned as',
              'a polytopic region of general dimension.')
        AS_coords = np.concatenate(Vertices_list, axis=0)
    
    
    # Small hack: Inject AS coordinates into return to be able to
    # plot 3D region effortlessly.
    mapped_region = [mapped_region, AS_coords]
    return mapped_region


def OI_eval(AS: pc.Region,
       DS: np.ndarray, perspective  = 'outputs',
       hypervol_calc:           str = 'robust',
       plot:                str = True):
    
    '''
    Operability Index (OI) calculation. From a Desired Output
    Set (DOS) defined by the user, this function calculates the intersection
    between achievable (AOS) and desired output operation (DOS). Similarly, the 
    OI can be also calculated from the inputs' perspective, as an intersection
    between desired input (DIS) and available input (AIS). This function is able
    to  evaluate the OI in any dimension, ranging from 1-d (length) up to
    higher dimensions (Hypervolumes, > 3-d), aided by the Polytope package.
    
    This function is part of Python-based Process Operability package.
    
    Control, Optimization and Design for Energy and Sustainability (CODES) 
    Group - West Virginia University - 2023
    
    Author: Victor Alves

    Parameters
    ----------
    AS : pc.Region
        Available input set (AIS) or Achievable output set (AOS) represented as
        a series of paired (and convex) polytopes. This region is easily 
        obtained using the multimodel_rep function.
    DS: np.ndarray
        Array containing the desired operation, either in the inputs (DIS) or
        outputs perspective (DOS).
    perspective: str, Optional.
        String that determines in which perspective the OI will be evaluated:
        inputs or outputs. default is 'outputs'.
    hypervol_calc: str, Optional.
        Determines how the approach when evaluating hypervolumes. Default is
        'robust', an implementation that switches from qhull's hypervolume
        evaluation or polytope's evaluation depending on the dimensionality of
        the problem. Additional option is 'polytope', Polytope's package own
        implementation of hypervolumes evaluation being used in problems of 
        any dimension.

    Returns
    -------
    OI: float
        Operability Index value. Ranges from 0 to 100%. A fully operable 
        process has OI = 100% and if not fully operable, OI < 100%.
    
        
    References
    ----------
    [1]  D. R., Vinson and C. Georgakis.  New Measure of Process Output 
    Controllability. J. Process Control, 2000. 
    https://doi.org/10.1016/S0959-1524(99)00045-1
    
    [2]  C. Georgakis, D. Uztürk, S. Subramanian, and D. R. Vinson. On the 
    Operability of Continuous Processes. Control Eng. Pract., 2003. 
    https://doi.org/10.1016/S0967-0661(02)00217-4
    
    [3] V.Gazzaneo, J. C. Carrasco, D. R. Vinson, and F. V. Lima. Process 
    Operability Algorithms: Past, Present, and Future Developments.
    Ind. & Eng. Chem. Res. 59 (6), 2457-2470, 2020.
    https://doi.org/10.1021/acs.iecr.9b05181

    
    '''
    
    # Defining Polytopes for manipulation. Obtaining polytopes in min-rep if
    # applicable.
    DS_region = pc.box2poly(DS)
    AS_region = pc.reduce(AS[0])
    DS_region = pc.reduce(DS_region)
    inter_list = list()
    
    # Obtain (possibly overlapped) intersection region of polytopes and
    # its respective bounding box.
    for i in range(len(AS_region)):
        intersection = pc.intersect(AS_region[i], DS_region)
        if intersection.fulldim is False or intersection.dim != DS_region.dim:
            pass
        else:
            inter_list.append(intersection)
            
    overlapped_intersection = pc.Region(inter_list)
    min_coord =  overlapped_intersection.bounding_box[0]
    max_coord =  overlapped_intersection.bounding_box[1]
    box_coord =  np.hstack([min_coord, max_coord])
    bound_box =  pc.box2poly(box_coord)
    
    # Remove overlapping (Vinson&Gazzaneo trick) - Remove one polytope at a time, 
    # create void polytope using the bounding box and subtract from the original 
    # bounding box itself. This avoids wrong calculations of the OI.
    intersection= process_overlapping_polytopes(bound_box, 
                                                overlapped_intersection)
    
    v_intersect_list = list()
    final_intersection = list()
    
    # Methods for OI evaluation.
    if hypervol_calc == 'polytope':
        OI = (intersection.volume/DS_region.volume)*100

    elif hypervol_calc == 'robust':
        
        if DS_region.dim < 7:
            intersect_i = []
            volumes_i = []
            
            for i in range(len(intersection)):
                intersect_i = intersection[i]
                # print(i)
                v_intersect = pc.extreme(intersect_i)
                
                if v_intersect is None:
                    continue
                else:
                    processed_intersection = pc.qhull(v_intersect)
                    final_intersection.append(processed_intersection)
                    v_intersect_list.append(v_intersect)
                    volumes_i.append(sp.spatial.ConvexHull(v_intersect).volume)
            
            each_polytope_volume = np.array(volumes_i)
            intersection_volume = each_polytope_volume[0:].sum()
            final_intersection = pc.Region(final_intersection)
            v_DS = pc.extreme(DS_region)
            
            # Evaluate OI
            OI = (intersection_volume / sp.spatial.ConvexHull(v_DS).volume)*100
        else:
            print("For higher dimensions (>7) polytope's hypervolume estimation \
                  is faster. Switching to polytope's calculation.")
            OI = (intersection.volume/DS_region.volume)*100
        
    else:
        print('Invalid hypervolume calculation option. Exiting algorithm.')
        sys.exit()



    # Perspective switch: This will only affect plot and legends.
    if perspective == 'outputs':
        DS_label = 'Desired Output Set (DOS)'
        AS_label = 'Achievable Output Set (AOS)'
        int_label = r'$ AOS \cap DOS$'

    else:
        DS_label = 'Desired Input Set (DIS)'
        AS_label = 'Available Input Set (AIS)'
        int_label = r'$ AIS \cap DIS$'

    # Plot if 2D / 3D
    if plot is True:
        if DS_region.dim == 2:
            
            
            polyplot = []
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(len(AS_region)):

                polyplot = _get_patch(AS_region[i], linestyle="solid",
                                    edgecolor=EDGES_COLORS, 
                                    linewidth=EDGES_WIDTH,
                                    facecolor=AS_COLOR)
                ax.add_patch(polyplot)

            
            for item in final_intersection:
                if item is None:
                    continue  # Skip None values
                else:
                    interplot = _get_patch(item, linestyle="solid",
                                           linewidth=EDGES_WIDTH,
                                           facecolor=INTERSECT_COLOR, 
                                           edgecolor=INTERSECT_COLOR)
                    ax.add_patch(interplot)
    
                    
            DSplot = _get_patch(DS_region, linestyle="dashed",
                                edgecolor=DS_COLOR, alpha=0.5, 
                                linewidth=EDGES_WIDTH,
                                facecolor=DS_COLOR)
            ax.add_patch(DSplot)
            ax.legend('DOS')

            lower_xaxis = min(AS_region.bounding_box[0][0], 
                              DS_region.bounding_box[0][0])
            upper_xaxis = max(AS_region.bounding_box[1][0], 
                              DS_region.bounding_box[1][0])

            lower_yaxis = min(AS_region.bounding_box[0][1], 
                              DS_region.bounding_box[0][1])
            upper_yaxis = max(AS_region.bounding_box[1][1], 
                              DS_region.bounding_box[1][1])

            ax.set_xlim(lower_xaxis - 0.05*lower_xaxis,
                        upper_xaxis + 0.05*upper_xaxis)
            ax.set_ylim(lower_yaxis - 0.05*lower_yaxis,
                        upper_yaxis + 0.05*upper_yaxis)

            DS_patch = mpatches.Patch(color=DS_COLOR, label=DS_label)
            AS_patch = mpatches.Patch(color=AS_COLOR, label=AS_label)
            INTERSECT_patch = mpatches.Patch(color=INTERSECT_COLOR,
                                            label=int_label)


            OI_str = 'Operability Index = ' + str(round(OI, 2)) + str('%')

            extra = mpatches.Rectangle((0, 0), 1, 1, fc="w",
                                    fill=False,
                                    edgecolor='none',
                                    linewidth=0, label=OI_str)

            if perspective == 'outputs':
                str_title = string.capwords(
                    "Operability Index Evaluation - Outputs' perspective")
                ax.set_title(str_title)
            else:
                str_title = string.capwords(
                    "Operability Index Evaluation - Inputs' perspective")
                ax.set_title(str_title)

            ax.legend(handles=[DS_patch, AS_patch, INTERSECT_patch, extra])

            ax.set_xlabel('$y_{1}$')
            ax.set_ylabel('$y_{2}$')
            plt.show()
        
        elif DS_region.dim == 3:
            
            AS_coords = AS[1]
            DS_coords = get_extreme_vertices(DS)
            polyplot = []
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            intersect_coords = np.concatenate(v_intersect_list,  axis=0)
           
            
            for cube, color in zip([AS_coords, DS_coords, intersect_coords], 
                                   [AS_COLOR, DS_COLOR, INTERSECT_COLOR]):
                hull = sp.spatial.ConvexHull(cube)
                # Draw the polygons of the convex hull
                for s in hull.simplices:
                    tri = Poly3DCollection([cube[s]])
                    tri.set_color(color)
                    tri.set_edgecolor(EDGES_COLORS)
                    tri.set_alpha(0.5)
                    ax.add_collection3d(tri)
                # Draw the vertices
                ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], 
                           marker='o', color='None')
            plt.show()

            
            AS_patch = mpatches.Patch(color=AS_COLOR, label=AS_label)

            extra = mpatches.Rectangle((0, 0), 1, 1, fc="w",
                                    fill=False,
                                    edgecolor='none',
                                    linewidth=0)

            if perspective == 'outputs':
                str_title = 'Achievable Output Set (AOS)'
                ax.set_title(str_title)
            else:
                str_title = 'Available Input set (AIS)'
                ax.set_title(str_title)

            ax.legend(handles=[AS_patch, extra])
            


            OI_str = 'Operability Index = ' + str(round(OI, 2)) + str('%')

            extra = mpatches.Rectangle((0, 0), 1, 1, fc="w",
                                    fill=False,
                                    edgecolor='none',
                                    linewidth=0, label=OI_str)

            if perspective == 'outputs':
                str_title = string.capwords(
                    "Operability Index Evaluation - Outputs' perspective")
                ax.set_title(str_title)
            else:
                str_title = string.capwords(
                    "Operability Index Evaluation - Inputs' perspective")
                ax.set_title(str_title)
            
            DS_patch = mpatches.Patch(color=DS_COLOR, label=DS_label)
            AS_patch = mpatches.Patch(color=AS_COLOR, label=AS_label)
            INTERSECT_patch = mpatches.Patch(color=INTERSECT_COLOR,
                                            label=int_label)

            ax.legend(handles=[DS_patch, AS_patch, INTERSECT_patch, extra])

            ax.set_xlabel('$y_{1}$')
            ax.set_ylabel('$y_{2}$')
            ax.set_zlabel('$y_{3}$')
            plt.show()

        elif DS_region.dim > 3:
            print('plot not supported. Dimension higher than 3d.',
                  'Nevertheless, the OI value is still available ', 
                  'for interpretation.')
    if plot is False:
        print('You have',
              'chosen plot=False.', 'Nevertheless, the OI value', 
              'is still available for interpretation.')
        

        
    return OI


def nlp_based_approach(model: Callable[..., Union[float, np.ndarray]],
                       DOS_bounds: np.ndarray,
                       DOS_resolution: np.ndarray,
                       u0: np.ndarray,
                       lb: np.ndarray,
                       ub: np.ndarray,
                       constr=None,
                       method: str = 'ipopt', 
                       plot: bool = True, 
                       ad: bool = False,
                       warmstart: bool = True) -> Union[np.ndarray, np.ndarray, list]:
    '''
    Inverse mapping for Process Operability calculations. From a Desired Output
    Set (DOS) defined by the user, this function calculates the closest
    Feasible Desired Ouput set (DOS*) from the AOS and its respective Feasible
    Desired Input Set (DIS*), which gives insight about potential changes in
    design and/or operations of a given process model.
    
    This function is part of Python-based Process Operability package.
    
    Control, Optimization and Design for Energy and Sustainability (CODES) 
    Group - West Virginia University - 2023
    
    Author: Victor Alves

    Parameters
    ----------
    model : Callable[...,Union[float,np.ndarray]]
        Process model that calculates the relationship from inputs (AIS-DIS) 
        to outputs (AOS-DOS).
    DOS_bounds : np.ndarray
        Array containing the bounds of the Desired Output set (DOS). Each row 
        corresponds to the lower and upper bound of each variable.
    DOS_resolution: np.ndarray
        Array containing the resolution of the discretization grid for the DOS.
        Each element corresponds to the resolution of each variable. For a 
        resolution defined as k, it will generate d^k points (in which d is the
        dimensionality of the problem).
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
                
                -'SLSQP'
                
    plot: bool
        Turn on/off plot. If dimension is d<=3, plot is available and
        both the Feasible Desired Output Set (DOS*) and Feasible Desired Input
        Set (DIS*) are plotted. Default is True.

    ad: bool
       Turn on/off use of Automatic Differentiation using JAX. If Jax is 
       installed, high-order data (jacobians, hessians) are obtained using AD.
       Default is False.

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
    gas utilization systems,” Comput. & Chem. Eng, 2017. 
    https://doi.org/10.1016/j.compchemeng.2016.12.010

    '''
    
    
    from scipy.optimize import NonlinearConstraint
    # Use JAX.numpy if differentiable programming is available.
    if ad is False:
        import numpy as np
        
        def p1(u: np.ndarray,
               model: Callable[..., Union[float, np.ndarray]],
               DOSpt: np.ndarray):

            y_found = model(u)

            vector_of_f = np.array([y_found.T, DOSpt.T])
            f = np.sum(error(*vector_of_f))

            return f

        # Error minimization function
        def error(y_achieved, y_desired):
            return ((y_achieved-y_desired)/y_desired)**2
        
    else:
        print(" You have selected automatic differentiation as a method for"
       " obtaining higher-order data (Jacobians/Hessian),")
        print(" Make sure your process model is JAX-compatible implementation-wise.")
        from jax.config import config
        config.update("jax_enable_x64", True)
        config.update('jax_platform_name', 'cpu')
        warnings.filterwarnings('ignore', module='jax._src.lib.xla_bridge')
        import jax.numpy as np
        from jax import jacrev, grad
        
        # Make sure that the arrays contain floats (Reviewer #2 bug @ JOSS)
        u0 = u0.astype(np.float64)
        lb = lb.astype(np.float64)
        ub = ub.astype(np.float64)
        DOS_bounds = DOS_bounds.astype(np.float64)
        
        def p1(u: np.ndarray,
               model: Callable[..., Union[float, np.ndarray]],
               DOSpt: np.ndarray):

            y_found = model(u)

            vector_of_f = np.array([y_found.T, DOSpt.T])
            f = np.sum(error(*vector_of_f))

            return f

        # Error minimization function
        
        def error(y_achieved, y_desired):
            return ((y_achieved-y_desired)/y_desired)**2
        
        # Take gradient and hessian of Objective function
        grad_ad = grad(p1)
        # hess_ad = jacrev(grad_ad)
        # BUG: AD-based hessians are deactivated in ipopt for now due to 
        # Cyipopt`s bug.
        # https://github.com/mechmotum/cyipopt/issues/175
        # https://github.com/mechmotum/cyipopt/pull/176
        
        if constr is not None:
            constr['jac']  = (jacrev(constr['fun']))
            # constr['fun'] = jit(constr['fun'])
            # BUG: AD-based hessians are deactivated for in ipopt now 
            # due to Cyipopt`s bug.
            # https://github.com/mechmotum/cyipopt/issues/175
            # https://github.com/mechmotum/cyipopt/pull/176
            
            # constr['hess'] = jit(jacrev(jacrev(constr['fun'])))
        else:
            pass

            
    if not isinstance(DOS_bounds, np.ndarray):
        DOS_bounds = np.array(DOS_bounds)
    else:
        pass
    
    dimDOS = DOS_bounds.shape[0]
    DOSPts = create_grid(DOS_bounds, DOS_resolution)
    DOSPts = DOSPts.reshape(-1, dimDOS)
    u00    = u0
    # Initialization of variables
    m = len(u0)
    r, c = np.shape(DOSPts)
    fDOS = np.zeros((r, c))
    fDIS = np.zeros((r, m))
    message_list = []
    bounds = np.column_stack((lb, ub))

    # Input sanitation
    if u0.size < c:
        warnings.warn("Your problem is non-square and you have "
                      "less degrees of freedom in the AIS/DIS "
                      "Than variables in the AIS.")
    if bounds.shape[0] != u0.size:
        raise ValueError("Initial estimate and given bounds have"
                         " inconsistent sizes."
                         " Check the dimensions"
                         " of your problem.")


    # If unbounded, set as +-inf.
    if lb.size == 0:
        lb = -np.inf
    if ub.size == 0:
        ub = np.inf

    # Inverse-mapping: Run for each DOS grid point
    for i in tqdm(range(r)):
        
        if constr is None:
            
            if ad is True:
                if method == 'trust-constr':
                    sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                               args=(model, DOSPts[i, :]),
                                               method=method,
                                               options={'xtol': 1e-10},
                                               jac=grad_ad)
                                                # , hess = hess_ad)

                elif method == 'Nelder-Mead':
                    sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                               args=(model, DOSPts[i, :]),
                                               method=method,
                                               options={'fatol': 1e-10,
                                                        'xatol': 1e-10},
                                               jac=grad_ad)
                                                # , hess = hess_ad)

                elif method == 'ipopt':
                    sol = minimize_ipopt(p1, x0=u0, bounds=bounds,
                                         args=(model, DOSPts[i, :]),
                                         jac=grad_ad)
                                         #, hess = hess_ad)

                elif method == 'DE':
                    sol = DE(p1, bounds=bounds, x0=u0, strategy='best1bin',
                             maxiter=2000, workers=-1, updating='deferred',
                             init='sobol', args=(model, DOSPts[i, :]))
            else:
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
                if ad==True:
                    sol = minimize_ipopt(p1, x0=u0, bounds=bounds,
                                         constraints=(constr),
                                         jac=grad_ad,
                                         args=(model, DOSPts[i, :]))

                else:
                    sol = minimize_ipopt(p1, x0=u0, bounds=bounds,
                                         constraints=(constr),
                                         args=(model, DOSPts[i, :]))

            elif method == 'DE':
                sol = DE(p1, bounds=bounds, x0=u0, strategy='best1bin',
                         maxiter=2000, workers=-1, updating='deferred',
                         init='sobol', constraints=(constr),
                         args=(model, DOSPts[i, :]))

            elif method == 'trust-constr':
                if ad is True:
                    con_fun =  constr['fun']
                    nlc = NonlinearConstraint((con_fun), -np.inf, 0,
                                              jac= (jacrev(con_fun)),
                                              hess=(jacrev(jacrev(con_fun))))
                    sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                               args=(model, DOSPts[i, :]),
                                               method=method, 
                                               constraints=(nlc),
                                               jac=grad_ad)
                                               # , hess=hess_ad)
                else:
                    sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                               args=(model, DOSPts[i, :]),
                                               method=method, 
                                               constraints=(nlc))

        
        
        # Append results into fDOS, fDIS and message list for each iteration
        
        if warmstart is True:
            if sol.success is True:
                # print(u0)
                u0 = sol.x
            else:
                u0 = u00 # Reboot to first initial estimate
        else:
            u0 = u00 # Reboot to first initial estimate
            
        
        if ad is True:
            fDOS = fDOS.at[i, :].set(model(sol.x))
            fDIS = fDIS.at[i, :].set(sol.x)

        elif ad is False:
            fDOS[i, :] = model(sol.x)
            fDIS[i, :] = sol.x

        message_list.append(sol.message)
    
    
    if fDIS.shape[1] > 3 and fDOS.shape[1] > 3:
        plot is False
        print('plot not supported. Dimension higher than 3.')
        pass
    else:

        if plot is True:
            
            if fDIS.shape[1] == 2 and fDOS.shape[1] == 2:
                _, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, 
                                              constrained_layout=True)
                ax1.scatter(fDIS[:, 0], fDIS[:, 1], s=16,
                            c=np.sqrt(fDOS[:, 0]**1 + fDOS[:, 1]**1),
                            cmap=cmap, antialiased=True,
                            lw=lineweight, marker='s',
                            edgecolors=edgecolors, label='DIS*')
                ax1.set_ylabel('$u_{2}$')
                ax1.set_xlabel('$u_{1}$')
                ax1.set_title('Feasible Desired Input Set (DIS*)', fontsize=10)
                
                vertices_DOS =  [(DOS_bounds[0, 0], DOS_bounds[1, 0]),
                                 (DOS_bounds[0, 0], DOS_bounds[1, 1]),
                                 (DOS_bounds[0, 1], DOS_bounds[1, 1]),
                                 (DOS_bounds[0, 1], DOS_bounds[1, 0])]
                
                
                vertices_DOS = np.array(vertices_DOS)
                
                ax2.fill(vertices_DOS[:, 0], vertices_DOS[:, 1],
                          facecolor='gray', edgecolor='gray',
                          alpha=0.5, label= 'DOS')
                
                
                ax2.scatter(fDOS[:, 0], fDOS[:, 1], s=16,
                            c=np.sqrt(fDOS[:, 0]**1 + fDOS[:, 1]**1),
                            cmap=cmap, antialiased=True,
                            lw=lineweight, marker='o',
                            edgecolors=edgecolors, label='DOS*')
                ax2.set_ylabel('$y_{2}$')
                ax2.set_xlabel('$y_{1}$')
                ax2.set_title('Feasible Desired Output Set (DOS*)', fontsize=10)
                plt.legend()
                
                
            elif fDIS.shape[1] == 3 and fDOS.shape[1] == 3:
                
                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax = fig.add_subplot(1,2,1, projection='3d')
                
                plt.rcParams['figure.facecolor'] = 'white'
                ax.scatter(fDIS[:, 0], fDIS[:, 1], fDIS[:,2], 
                           s=16, c=np.sqrt(fDOS[:, 0]**2 + 
                                           fDOS[:, 1]**2 +
                                           fDOS[:, 2]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='s',
                        edgecolors=edgecolors)
                
                ax.set_xlabel('$u_{1}$')
                ax.set_ylabel('$u_{2}$')
                ax.set_zlabel('$u_{3}$')
                
                ax.set_title('DIS*')
                
                ax = fig.add_subplot(1,2,2, projection='3d')
                
                ax.scatter(fDOS[:, 0], 
                           fDOS[:, 1], 
                           fDOS[:, 2], 
                           s=16,
                           c=np.sqrt(fDOS[:, 0]**2 + 
                                     fDOS[:, 1]**2 + 
                                     fDOS[:, 2]**2),
                           cmap=cmap, 
                           antialiased=True,
                           lw=lineweight, 
                           marker='o',
                           edgecolors=edgecolors)
                ax.set_ylabel('$y_{2}$')
                ax.set_xlabel('$y_{1}$')
                ax.set_zlabel('$y_{3}$')
                ax.set_title('$DOS*$')
                
                
            elif fDIS.shape[1] == 2 and fDOS.shape[1] == 3:
                
                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax = fig.add_subplot(1,2,1)
                
                plt.rcParams['figure.facecolor'] = 'white'
                ax.scatter(fDIS[:, 0], fDIS[:, 1], 
                           s=16, c=np.sqrt(fDOS[:, 0]**2 + 
                                           fDOS[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='s',
                        edgecolors=edgecolors)
                
                ax.set_xlabel('$u_{1}$')
                ax.set_ylabel('$u_{2}$')
                
                
                ax.set_title('DIS*')
                
                ax = fig.add_subplot(1,2,2, projection='3d')
                
                ax.scatter(fDOS[:, 0], 
                           fDOS[:, 1], 
                           fDOS[:, 2], 
                           s=16,
                           c=np.sqrt(fDOS[:, 0]**2 + 
                                     fDOS[:, 1]**2 + 
                                     fDOS[:, 2]**2),
                           cmap=cmap, 
                           antialiased=True,
                           lw=lineweight, 
                           marker='o',
                           edgecolors=edgecolors)
                ax.set_ylabel('$y_{2}$')
                ax.set_xlabel('$y_{1}$')
                ax.set_zlabel('$y_{3}$')
                ax.set_title('$DOS*$')
                
            elif fDIS.shape[1] == 3 and fDOS.shape[1] == 2:
                
                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax = fig.add_subplot(1,2,1, projection='3d')
                
                plt.rcParams['figure.facecolor'] = 'white'
                ax.scatter(fDIS[:, 0], fDIS[:, 1], fDIS[:, 2],
                           s=16, c=np.sqrt(fDOS[:, 0]**2 + 
                                           fDOS[:, 1]**2 +
                                           fDOS[:, 2]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='s',
                        edgecolors=edgecolors)
                
                ax.set_xlabel('$u_{1}$')
                ax.set_ylabel('$u_{2}$')
                ax.set_zlabel('$u_{3}$')
                
                
                ax.set_title('DIS*')
                
                ax = fig.add_subplot(1,2,2)
                
                ax.scatter(fDOS[:, 0], 
                           fDOS[:, 1],
                           s=16,
                           c=np.sqrt(fDOS[:, 0]**2 + 
                                     fDOS[:, 1]**2),
                           cmap=cmap, 
                           antialiased=True,
                           lw=lineweight, 
                           marker='o',
                           edgecolors=edgecolors)
                ax.set_ylabel('$y_{2}$')
                ax.set_xlabel('$y_{1}$')
                ax.set_title('$DOS*$')   
            else:
                print('plot not supported. Dimension higher than 3.')
                plot is False
                pass


    return fDIS, fDOS, message_list



def create_grid(region_bounds: np.ndarray, region_resolution: tuple):
    
    '''
    Create a multidimensional, discretized grid, given the bounds and the
    resolution.
    
    This function is part of Python-based Process Operability package.
    
    Control, Optimization and Design for Energy and Sustainability (CODES) 
    Group - West Virginia University - 2023
    
    Author: San Dinh

    Parameters
    ----------
    region_bounds : np.ndarray
        Numpy array that contains the bounds of a (possibily) multidimensional
        region. Each row corresponds to the lower and upper bound of each 
        variable that constitutes the hypercube.
    region_resolution: np.ndarray
        Array containing the resolution of the discretization grid for the 
        region. Each element corresponds to the resolution of each variable. 
        For a resolution defined as k, it will generate d^k points 
        (in which d is the dimensionality of the problem).

    Returns
    -------
    region_grid: np.ndarray
        Multidimensional array that represents the grid descritization of the
        region in Euclidean coordinates.
    
    
    '''
    # Indexing
    numInput = np.prod(region_resolution)
    nInput = region_bounds.shape[0]
    Input_u = []
    
    # Create discretized region based on bounds and resolution information.
    for i in range(nInput):
        Input_u.append(list(np.linspace(region_bounds[i, 0],
                                        region_bounds[i, 1],
                                        region_resolution[i])))

    # Create slack variables for preallocation purposes.
    region_grid = np.zeros(region_resolution + [nInput])

    for i in range(numInput):
        inputID = [0]*nInput
        inputID[0] = int(np.mod(i, region_resolution[0]))
        region_val = [Input_u[0][inputID[0]]]

        for j in range(1, nInput):
            inputID[j] = int(np.mod(np.floor(i/np.prod(region_resolution[0:j])),
                                    region_resolution[j]))
            region_val.append(Input_u[j][inputID[j]])

        region_grid[tuple(inputID)] = region_val

    return region_grid


def AIS2AOS_map(model: Callable[...,Union[float,np.ndarray]],
     AIS_bound: np.ndarray,
     AIS_resolution: np.ndarray, EDS_bound: np.ndarray = None,
     EDS_resolution: np.ndarray = None, plot: bool = True)-> Union[np.ndarray,np.ndarray]:
    '''
    Forward mapping for Process Operability calculations (From AIS to AOS). 
    From an Available Input Set (AIS) bounds and discretization resolution both
    defined by the user, 
    this function calculates the corresponding discretized 
    Available Output Set (AOS).
    
    This function is part of Python-based Process Operability package.

    Control, Optimization and Design for Energy and Sustainability 
    (CODES) Group - West Virginia University - 2023

    Authors: San Dinh & Victor Alves

    Parameters
    ----------
    model : Callable[...,Union[float,np.ndarray]]
        Process model that calculates the relationship between inputs (AIS)
        and Outputs (AOS).
    AIS_bound : np.ndarray
        Lower and upper bounds for the Available Input Set (AIS).
    AIS_resolution : np.ndarray
        Resolution for the Available Input Set (AIS). This will be used to
        discretize the AIS.
    EDS_bound : np.ndarray
        Lower and upper bounds for the Expected Disturbance Set (EDS). Default
        is 'None'.
    EDS_resolution : np.ndarray
        Resolution for the Expected Disturbance Set (EDS). This will be used to
        discretize the EDS.   
    plot: bool
        Turn on/off plot. If dimension is d<=3, plot is available and
        both the Achievable Output Set (AOS) and Available Input
        Set (AIS) are plotted. Default is True.

    Returns
    -------
    AIS : np.ndarray
        Discretized Available Input Set (AIS).
    AOS : np.ndarray
        Discretized Available Output Set (AIS).

    '''
    
    # Indexing
    if type(EDS_bound) and type(EDS_resolution) is type(None):
        numInput_map = np.prod(AIS_resolution)
        nInput_map = AIS_bound.shape[0]
        map_bounds = AIS_bound
        map_resolution = AIS_resolution
    else:
        numInput_map = np.prod(AIS_resolution + EDS_resolution)
        numInput_d = np.prod(EDS_resolution)
        nInput_map = AIS_bound.shape[0] + EDS_bound.shape[0]
        nInput_d = EDS_bound.shape[0]
        map_bounds = np.concatenate([AIS_bound, EDS_bound])
        map_resolution =  AIS_resolution + EDS_resolution
        EDS = np.zeros(EDS_resolution + [nInput_d])
        
        
    
    Input_map = []
    Input_d = []
    
    # Create discretized AIS based on bounds and resolution information.
    for i in range(nInput_map):
        Input_map.append(list(np.linspace(map_bounds[i, 0],
                                        map_bounds[i, 1],
                                        map_resolution[i])))
        
        
    if (type(EDS_bound) and type(EDS_resolution)) is not type(None):
        for i in range(nInput_d):
            Input_d.append(list(np.linspace(EDS_bound[i, 0],
                                            EDS_bound[i, 1],
                                            EDS_resolution[i])))
    else:
        pass
    
    
    # Create slack variables for preallocation purposes.
    u_d_slack = map_bounds[:, 0]
    y_slack = model(u_d_slack)
    nOutput = y_slack.shape[0]
    input_map = np.zeros(map_resolution + [nInput_map])
    AOS = np.zeros(map_resolution + [nOutput])
    

    # General map (AIS+EDS) multidimensional array
    for i in range(numInput_map):
        inputID = [0]*nInput_map
        inputID[0] = int(np.mod(i, map_resolution[0]))
        map_val = [Input_map[0][inputID[0]]]

        for j in range(1, nInput_map):
            inputID[j] = int(np.mod(np.floor(i/np.prod(map_resolution[0:j])),
                                    map_resolution[j]))
            map_val.append(Input_map[j][inputID[j]])

        input_map[tuple(inputID)] = map_val
        AOS[tuple(inputID)] = model(map_val)
        
    # EDS multidimensional array.
    if (type(EDS_bound) and type(EDS_resolution)) is not type(None):
        for i in range(numInput_d):
            inputID = [0]*nInput_d
            inputID[0] = int(np.mod(i, EDS_resolution[0]))
            map_val = [Input_d[0][inputID[0]]]

            for j in range(1, nInput_d):
                inputID[j] = int(np.mod(np.floor(i/np.prod(EDS_resolution[0:j])),
                                        map_resolution[j]))
                map_val.append(Input_d[j][inputID[j]])

            EDS[tuple(inputID)] = map_val
        
    else:
        pass
        
    
    # 2D / 3D Plots
    if plot is False:
        pass
    elif plot is True:
        if input_map.shape[-1]  == 2 and AOS.shape[-1] == 2:
            
            input_plot = input_map.reshape(np.prod(input_map.shape[0:-1]),
                                         input_map.shape[-1])
            
            AOS_plot = AOS.reshape(np.prod(AOS.shape[0:-1]), AOS.shape[-1])
            
            _, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, 
                                          constrained_layout=True)

            plt.rcParams['figure.facecolor'] = 'white'
            ax1.scatter(input_plot[:, 0], input_plot[:, 1], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='s',
                        edgecolors=edgecolors)
            
            ax1.set_xlabel('$u_{1}$')
            if (EDS_bound and EDS_resolution) is None:
                ax1.set_title('$AIS_{u}$')
                ax1.set_ylabel('$u_{2}$')
            else:
                ax1.set_title('$AIS_{u} \, and \, EDS_{d}$')
                ax1.set_ylabel('$d_{1}$')
           
            
            ax2.scatter(AOS_plot[:, 0], AOS_plot[:, 1], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax2.set_ylabel('$y_{2}$')
            plt.xlabel('$y_{1}$')
            
            ax2.set_title('$AOS$')

            
        elif input_map.shape[-1]== 3 and AOS.shape[-1] == 3:
            
            input_plot = input_map.reshape(np.prod(input_map.shape[0:-1]), 
                                           input_map.shape[-1])
            AOS_plot = AOS.reshape(np.prod(AOS.shape[0:-1]), AOS.shape[-1])
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1,2,1, projection='3d')

            plt.rcParams['figure.facecolor'] = 'white'
            ax.scatter(input_plot[:, 0], input_plot[:, 1], input_plot[:,2], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2 +
                                  AOS_plot[:, 2]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='s',
                        edgecolors=edgecolors)
            
            ax.set_xlabel('$u_{1}$')

            if (type(EDS_bound) and type(EDS_resolution)) is type(None):
                ax.set_title('$AIS_{u}$')
                ax.set_ylabel('$u_{2}$')
                ax.set_zlabel('$u_{3}$')
               
            elif EDS_bound.shape[0] == 2:
                ax.set_title('$AIS_{u} \, and  \, EDS_{d}$')
                
                ax.set_ylabel('$d_{1}$')
                ax.set_zlabel('$d_{2}$')
            elif EDS_bound.shape[0] == 1:
                ax.set_title('$AIS_{u} \, and \, EDS_{d}$')
                ax.set_ylabel('$u_{2}$')
                ax.set_zlabel('$d_{1}$')
                
            

            
            ax = fig.add_subplot(1,2,2, projection='3d')
            ax.scatter(AOS_plot[:, 0], AOS_plot[:, 1], AOS_plot[:, 2], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2 + 
                                  AOS_plot[:, 2]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax.set_ylabel('$y_{2}$')
            ax.set_xlabel('$y_{1}$')
            ax.set_zlabel('$y_{3}$')
            ax.set_title('$AOS$')
            
        elif input_map.shape[-1] == 2 and AOS.shape[-1] == 3:
            
            input_plot = input_map.reshape(np.prod(input_map.shape[0:-1]), 
                                     input_map.shape[-1])
            AOS_plot = AOS.reshape(np.prod(AOS.shape[0:-1]), AOS.shape[-1])
            fig = plt.figure(figsize=plt.figaspect(0.5))
            
            ax = fig.add_subplot(1,2,1)

            plt.rcParams['figure.facecolor'] = 'white'
            ax.scatter(input_plot[:, 0], input_plot[:, 1], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='s',
                        edgecolors=edgecolors)

            ax.set_xlabel('$u_{1}$')

            

            if (type(EDS_bound) and type(EDS_resolution)) is type(None):
                plt.title('$AIS_{u}$')
                plt.ylabel('$u_{2}$')
            else:
                plt.title('$AIS_{u} \, and \, EDS_{d}$')
                plt.ylabel('$d_{1}$')
            
            
            ax = fig.add_subplot(1,2,2, projection='3d')
            ax.scatter(AOS_plot[:, 0], AOS_plot[:, 1], AOS_plot[:, 2], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2 + 
                                  AOS_plot[:, 2]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax.set_ylabel('$y_{2}$')
            ax.set_xlabel('$y_{1}$')
            ax.set_zlabel('$y_{3}$')
            ax.set_title('$AOS$')
            
            
        elif input_map.shape[-1] == 3 and AOS.shape[-1] == 2:
            
            input_plot = input_map.reshape(np.prod(input_map.shape[0:-1]), 
                                           input_map.shape[-1])
            AOS_plot = AOS.reshape(np.prod(AOS.shape[0:-1]), AOS.shape[-1])
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1,2,1, projection='3d')

            plt.rcParams['figure.facecolor'] = 'white'
            ax.scatter(input_plot[:, 0], input_plot[:, 1], input_plot[:,2], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='s',
                        edgecolors=edgecolors)

            ax.set_xlabel('$u_{1}$')

            if (type(EDS_bound) and type(EDS_resolution)) is type(None):
                ax.set_title('$AIS_{u}$')
                ax.set_ylabel('$u_{2}$')
                ax.set_zlabel('$u_{3}$')

            elif EDS_bound.shape[0] == 2:
                ax.set_title('$AIS_{u} \, and \, EDS_{d}')
                
                ax.set_ylabel('$d_{1}$')
                ax.set_zlabel('$d_{2}$')
            elif EDS_bound.shape[0] == 1:
                ax.set_title('$AIS_{u} \, and \, EDS_{d}$')
                ax.set_ylabel('$u_{2}$')
                ax.set_zlabel('$d_{1}$')
            
            
            ax = fig.add_subplot(1,2,2)

            ax.scatter(AOS_plot[:, 0], AOS_plot[:, 1], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax.set_ylabel('$y_{2}$')
            ax.set_xlabel('$y_{1}$')
            
            ax.set_title('$AOS$')
                
            
        else:
            print('dimension greater than 3, plot not supported.')
            
    else:
        pass
            
    
    return input_map, AOS


def points2simplices(AIS: np.ndarray, AOS: np.ndarray) -> Union[np.ndarray,
                                                                np.ndarray]:
    '''
    Generation of connected simplices (k+1 convex hull of k+1 vertices) 
    based on the AIS/AOS points.
    
    
    This function is part of Python-based Process Operability package.
    
    Control, Optimization and Design for Energy and Sustainability 
    (CODES) Group - West Virginia University - 2023
    
    Author: San Dinh

    Parameters
    ----------
    AIS : np.ndarray
        Array containing the points that constitutes the AIS.
    AOS : np.ndarray
        Array containing the points that constitutes the AOS.

    Returns
    -------
    AIS_simplices : np.ndarray
        List of input (AIS) vertices for each simplex generated, in the form of
        an array.
    AOS_simplices : np.ndarray
        List of output (AOS) vertices for each simplex generated, in the form 
        of an array.

    '''

    # Additional parameters: Resolution, no. of inputs/outputs.
    resolution = AIS.shape[0:-1]
    nInputVar = AIS.shape[-1]
    nOutputVar = AOS.shape[-1]
    numInput = np.prod(resolution)
    nInput = nInputVar

    inputID = [1]*nInput
    permutations = list(perms(range(nOutputVar)))
    n_simplex = len(permutations)
    simplex_vertices = list()
    for n in range(n_simplex):

        ineq_choice = permutations[n]
        A_full = np.zeros((nOutputVar+1, nOutputVar))
        A_full[0, ineq_choice[0]] = -1
        for i in range(0, nOutputVar-1):
            A_full[i+1, ineq_choice[i]] = 1
            A_full[i+1, ineq_choice[i+1]] = -1
        A_full[-1, ineq_choice[-1]] = 1
        b_full = np.zeros((nOutputVar+1, 1))
        b_full[-1] = 1

        V_simplex = np.zeros((nOutputVar, nOutputVar+1))

        for i in range(nOutputVar+1):
            A = np.delete(A_full, i, axis=0)
            b = np.delete(b_full, i, axis=0)
            V_sol = np.linalg.solve(A, b)
            V_simplex[:, [i]] = V_sol.astype(int)

        simplex_vertices.append(list(V_simplex.T))

    AOS_simplices = list()
    AIS_simplices = list()
    for i in range(numInput):
        inputID[0] = int(np.mod(i, resolution[0]))

        for j in range(1, nInput):
            inputID[j] = int(np.mod(np.floor(i/np.prod(resolution[0:j])),
                                    resolution[j]))

        if np.prod(inputID) != 0:
            inputID = [x-1 for x in inputID]

            for k in range(n_simplex):

                ID = [a + b for a, b in zip([inputID]*(nInput + 1),
                                            simplex_vertices[k])]
                
                

                V_AOS_id = np.zeros((nOutputVar, nInput + 1))
                V_AIS_id = np.zeros((nInputVar, nInput + 1))
                for m in range(nInput + 1):
                    ID_cell = tuple([int(x) for x in ID[m]])
                    V_AOS_id[:, m] = AOS[ID_cell]
                    V_AIS_id[:, m] = AIS[ID_cell]

                AOS_simplices.append(V_AOS_id)
                AIS_simplices.append(V_AIS_id)
                
                
                
                
    # Putting polytopes together.
    for i, simplex in enumerate(AOS_simplices):
        poly = pc.qhull(simplex.T)
        AOS_simplices[i] = pc.extreme(poly)

    for i, simplex in enumerate(AIS_simplices):
        poly = pc.qhull(simplex.T)
        AIS_simplices[i] = pc.extreme(poly)


    
    return AIS_simplices, AOS_simplices


def points2polyhedra(AIS: np.ndarray, AOS: np.ndarray) -> Union[np.ndarray,
                                                                np.ndarray]:
    '''
    Generation of connected polyhedra based on the AIS/AOS points.
    
    This function is part of Python-based Process Operability package.
    
    Control, Optimization and Design for Energy and Sustainability 
    (CODES) Group - West Virginia University - 2023
    
    Author: San Dinh

    Parameters
    ----------
    AIS : np.ndarray
        Array containing the points that constitutes the AIS.
    AOS : np.ndarray
        Array containing the points that constitutes the AOS.

    Returns
    -------
    AIS_polytope : np.ndarray
        List of input (AIS) vertices for each polytope generated, in the form 
        of an array.
    AOS_polytope : np.ndarray
        List of output (AOS) vertices for each polyhedra generated, in the form 
        of an array.

    '''

    # Additional parameters: Resolution, no. of inputs/outputs.
    resolution = AIS.shape[0:-1]
    nInputVar = AIS.shape[-1]
    nOutputVar = AOS.shape[-1]
    numInput = np.prod(resolution)
    nInput = nInputVar

    inputID = [1]*nInput
    cube_vertices = list()
    for n in range(2**(nInput)):

        cube_vertices.append([int(x) for x in
                              np.binary_repr(n, width=nInput)])
    # Preallocation
    AOS_polytope = list()
    AIS_polytope = list()
    
    for i in range(numInput):
        inputID[0] = int(np.mod(i, resolution[0]))

        for j in range(1, nInput):
            inputID[j] = int(np.mod(np.floor(i/np.prod(resolution[0:j])),
                                    resolution[j]))

        if np.prod(inputID) != 0:
            inputID = [x-1 for x in inputID]
            ID = np.array([inputID]*(2**(nInput))) + np.array(cube_vertices)

            V_AOS_id = np.zeros((nOutputVar, 2**(nInput)))
            V_AIS_id = np.zeros((nInputVar, 2**(nInput)))
            for k in range(2**(nInput)):
                ID_cell = tuple([int(x) for x in ID[k]])
                V_AOS_id[:, k] = AOS[ID_cell]
                V_AIS_id[:, k] = AIS[ID_cell]

            AOS_polytope.append(V_AOS_id)
            AIS_polytope.append(V_AIS_id)
            
            
    # Putting polytopes together.
    for i, simplex in enumerate(AOS_polytope):
        poly = pc.qhull(simplex.T)
        AOS_polytope[i] = pc.extreme(poly)

    for i, simplex in enumerate(AIS_polytope):
        poly = pc.qhull(simplex.T)
        AIS_polytope[i] = pc.extreme(poly)


    return AIS_polytope, AOS_polytope


def implicit_map(model:             Callable[...,Union[float,np.ndarray]], 
                 domain_bound:      np.ndarray, 
                 domain_resolution: np.ndarray, 
                 image_init:        np.ndarray ,
                 direction:         str = 'forward', 
                 validation:        str = 'predictor-corrector', 
                 tol_cor:           float = 1e-4, 
                 continuation:      str = 'Explicit RK4',
                 derivative:        str = 'jax',
                 jit:               bool = True,
                 step_cutting:      bool = False):
    '''
    Performs implicit mapping of a implicitly defined process F(u,y) = 0. 
    F can be a vector-valued, multivariable function, which is typically the 
    case for chemical processes studied in Process Operability. 
    This method relies in the implicit function theorem and automatic
    differentiation in order to obtain the mapping of the required 
    input/output space. The
    mapping "direction" can be set by changing the 'direction' parameter.
    
    Authors: San Dinh & Victor Alves
    
    Control, Optimization and Design for Energy and Sustainability 
    (CODES) Group - West Virginia University - 2023
    

    Parameters
    ----------
    implicit_model : Callable[...,Union[float,np.ndarray]]
        Process model that describes the relationship between the input and 
        output spaces. Has to be written as a function in the following form:
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
        Cutting step strategy to subdivide the domain/image in case of stiffness.
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
        
    References
    ----------
    V. Alves, J. R. Kitchin, and F. V. Lima. "An inverse mapping approach for process
    systems engineering using automatic differentiation and the implicit 
    function theorem". AIChE Journal, 2023. 
    https://doi.org/10.1002/aic.18119

    '''
    
    # Implicit function theorem and pre-configuration steps.
    if direction == 'forward':
        print('Forward Mapping Selected.')
        print('The given domain is recognized as an Available Input Set (AIS).')
        print('The result of this mapping is an Achievable Output Set(AOS)')

        def F(i, o)   : return model(i, o)
        def F_io(o, i): return model(i, o)
        
    elif direction == 'inverse':
        print('Inverse Mapping Selected.')
        print('The given domain is recognized as Desired Output Set (DOS).')
        print('The result of this mapping is an Desired Input Set(DIS)')

        def F(i, o)   : return model(o, i)
        def F_io(o, i): return model(o, i)
    else:
        print('Invalid Mapping Selected. Please select the direction \
              to be either "forward" or "inverse"')


    # Use JAX.numpy if differentiable programming is available.
    if derivative == 'jax':
        from jax.config import config
        config.update("jax_enable_x64", True)
        import jax.numpy as np
        from jax import jit, jacrev
        from jax.experimental.ode import odeint as odeint
        dFdi = jacrev(F, 0)
        dFdo = jacrev(F, 1)
    else:
        print('Currently JAX is the only supported option for \
              calculating derivatives. Exiting code.')
        sys.exit()

                 

    if jit:
        @jit
        def dodi(ii,oo):
            return -pinv(dFdo(ii,oo)) @ dFdi(ii,oo)
        
        @jit
        def dods(oo, s, s_length, i0, iplus):
            return dodi(i0 + (s/s_length)*(iplus - i0), oo) \
                @((iplus - i0)/s_length)
    else:
        def dodi(ii, oo): return -pinv(dFdo(ii, oo)) @ dFdi(ii, oo)
        def dods(oo, s, s_length, i0, iplus): return dodi(
            i0 + (s/s_length)*(iplus - i0), oo)@((iplus - i0)/s_length)

        
    #  Initialization step: obtaining first solution
    sol = root(F_io, image_init,args=domain_bound[:,0])
    
    #  Predictor scheme selection
    if continuation == 'Explicit RK4':
        print('Selected RK4')
        
        def predict_RK4(dodi,i0, iplus ,o0):
            h = iplus -i0
            k1 = dodi( i0          ,  o0           )
            k2 = dodi( i0 + (1/2)*h,  o0 + (h/2) @ k1)
            k3 = dodi( i0 + (1/2)*h,  o0 + (h/2) @ k1)
            k4 = dodi(np.array(i0+h), o0 +      h@ k3)
            
            return o0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4) @ h
        
        predict = predict_RK4
        do_predict = dodi
        
    elif continuation == 'Explicit Euler':
        
        print('Selected Euler')
        
        def predict_eEuler(dodi,i0, iplus ,o0):
            return o0 + dodi(i0,o0)@(iplus -i0)
        
        predict = predict_eEuler
        do_predict = dodi
        
    elif continuation == 'odeint':
        
        print('Selected odeint')
        
        def predict_odeint(dods, i0, iplus ,o0):
            s_length = norm(iplus - i0)
            s_span = np.linspace(0.0, s_length, 10)
            sol = odeint(dods, o0, s_span, s_length, i0, iplus)
            return sol[-1,:]
        
        predict = predict_odeint
        do_predict = dods
        
    else:
        print('Ivalid continuation method. Exiting algorithm.')
        sys.exit()
        
    # Pre-allocating the domain set
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

    cube_vertices = list()
    for n in range(2**(nInput)):
        cube_vertices.append([int(x) for x in
                              np.binary_repr(n, width=nInput)])
    
    # Preallocate domain and image polyhedras
    domain_polyhedra = list()
    image_polyhedra = list()

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

# The functions below are fundamental for operability calculations, though  
# typical users won't need to directly interact with them. They play a crucial 
# role within 'opyrability' without requiring user intervention, but are 
# documented here nevertheless.
def get_extreme_vertices(bounds: np.ndarray) -> np.ndarray:
    """
    Gets the extreme vertices of any D-dimensional hypercube. This is used to
    plot the DOS in 3D.
    
    Author: Victor Alves
    
    Control, Optimization and Design for Energy and Sustainability 
    (CODES) Group - West Virginia University - 2023

    Parameters
    ----------
    bounds : np.ndarray
        Lower and upper bounds for the Desired Output Set (DOS).

    Returns
    -------
    extreme_vertices : np.ndarray
        Extreme vertices for the DOS.

    """
    num_dimensions = bounds.shape[0]
    num_points = 2 ** num_dimensions

    extreme_vertices = np.zeros((num_points, num_dimensions))

    for i in range(num_dimensions):
        indices = np.arange(num_points) // (2 ** i) % 2
        extreme_vertices[:, i] = bounds[i, indices]

    return extreme_vertices


def process_overlapping_polytopes(bound_box: pc.Polytope, 
                                  overlapped_intersection: pc.Region) -> pc.Region:
    """
   Eliminate overlaps between polytopes given a bounding box and a region of 
   potentially overlapping polytopes.

   The function aims to process a set of polytopes such that the resultant 
   polytopes within the specified bounding
   box do not overlap with each other.
   
   This was initially suggested by David Vinson in his Ph.D. dissertation 
   "A new measure of process operability for the improved steady-state design 
   of chemical processes. Ph.D. Thesis, Lehigh University, USA, 2000."
   
   This has been refined by me to perform the subraction of polytopes if and only
   if the polytopes are indeed overlapping. This is checked by 
   function 'are_overlapping'. Hence, computational time is largely reduced.
   In addition, this avoids numerical instability in the LP that could
   potentially lead to unbounded solutions (infesaible).
   
   Author: Victor Alves
   
   Control, Optimization and Design for Energy and Sustainability 
   (CODES) Group - West Virginia University - 2023

   Parameters
   ----------
   bound_box : pc.Polytope
       The bounding polytope within which all other polytopes are contained.
   
   overlapped_intersection : pc.Region
       A collection of polytopes (encapsulated in a pc.Region) that may 
       potentially overlap with each other.

   Returns
   -------
   pc.Region
       A region consisting of polytopes that are within the bounding box 
       and do not overlap with each other.


   """
    refined_polytopes = list(overlapped_intersection)
    flattened_polytopes = []
    for poly in refined_polytopes:
        if isinstance(poly, pc.Region):
            flattened_polytopes.extend(poly)
        else:
            flattened_polytopes.append(poly)

    refined_polytopes = flattened_polytopes
    non_overlapping_polytopes = []

    while refined_polytopes:
        current_poly = refined_polytopes.pop()
        overlapping = False

        for poly in non_overlapping_polytopes:
            try:
                if are_overlapping(current_poly, poly):
                    # Split the current polytope and add the non-overlapping
                    # parts back for consideration
                    diffs = current_poly.diff(poly)
                    if isinstance(diffs, pc.Region):
                        refined_polytopes.extend(diffs)
                    else:
                        refined_polytopes.append(diffs)
                    overlapping = True
                    break

            except RuntimeError as e:
                print('Some overlapping polytopes could not be resolved.')
                continue

        # If there was no overlap with existing non-overlapping polytopes, 
        # add it to the list
        if not overlapping:
            non_overlapping_polytopes.append(current_poly)

    # Intersect each polytope in the non_overlapping_polytopes list with the 
    # bounding box
    final_polytopes = []
    for poly in non_overlapping_polytopes:
        try:
            intersection = pc.intersect(bound_box, poly)

            if not pc.is_empty(intersection):
                if isinstance(intersection, pc.Region):
                    final_polytopes.extend(intersection)
                else:
                    final_polytopes.append(intersection)

        except RuntimeError as e:
            print('Some overlapping polytopes could not be resolved.')
            continue

    return pc.Region(final_polytopes)

def are_overlapping(poly1, poly2):
    """
   Check if two polytopes overlap.
   
   Author: Victor Alves
   
   Control, Optimization and Design for Energy and Sustainability 
   (CODES) Group - West Virginia University - 2023

   Parameters
   ----------
   poly1 : Polytope
       First polytope object.
   poly2 : Polytope
       Second polytope object.

   Returns
   -------
   bool
       True if the polytopes overlap, False otherwise.
   """
    return not pc.is_empty(pc.intersect(poly1, poly2))
