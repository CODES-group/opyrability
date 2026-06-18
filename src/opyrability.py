"""
Opyrability - Process Operability Analysis in Python
=====================================================

A Python package for process operability analysis: forward and inverse
mapping between input and output spaces, operability index evaluation,
nonlinear- and mixed-integer-linear-programming based operability
calculations, and dynamic operability, for steady-state and dynamic
process models.

Copyright (c) 2022-2026 Victor Alves -- Carnegie Mellon University.
Released under the MIT License.

See the acknowledgements on source code and documentation for the project's origins 
and current development.
"""
# ---------------------------------------------------------------------------- #
# Acknowledgements

# Opyrability began its development in late 2022 in the Control, Optimization and
# Design for Energy and Sustainability (CODES) Group at West Virginia
# University (WVU), by Victor Alves and San Dinh. 

# It is currently under active development and maintained by
# Victor Alves @ Carnegie Mellon University.
# ---------------------------------------------------------------------------- #


# Basic python tools
import sys
import warnings
import string
from itertools import permutations as perms
from typing import Callable, Union
from tqdm.auto import tqdm
# Linear Algebra
import numpy as np
from numpy.linalg import norm

# Optimization algorithms
import scipy as sp
from scipy.optimize import root
from scipy.optimize import differential_evolution as DE
# Pounce (pure-Rust IPOPT, bundled FERAL linear solver) is the default NLP
# solver and a required dependency. cyipopt is optional and imported lazily
# (see _import_cyipopt_minimize) because it ships compiled binaries that
# users typically install via conda.
from pounce import minimize as pounce_minimize

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
                  plot: bool = True,
                  EDS_bound = None,
                  EDS_resolution = None,
                  labels: str = None):
    
    """
    Obtain a multimodel representation based on polytopes of Process Operability
    sets. This procedure is essential for evaluating the Operability Index (OI).
    
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
        DOS. Each element corresponds to the resolution of each variable. For a
        resolution defined as k, it will generate d^k points (in which d is the
        dimensionality of the AIS or DOS).
    polytopic_trace: str, Optional.
        Determines if the polytopes will be constructed using simplices or
        polyhedrons. Default is 'simplices'. Additional option is 'polyhedra'.
    perspective: str, Optional.
        Defines if the calculation is to be done from the inputs/outputs
        perspective. Also affects labels in plots. Default is 'outputs'.
    plot: str, Optional.
        Defines if the plot of operability sets is desired (If the dimension
        is <= 3). Default is True.
    EDS_bound : np.ndarray
        Lower and upper bounds for the Expected Disturbance Set (EDS). Default
        is 'None'.
    EDS_resolution : np.ndarray
        Resolution for the Expected Disturbance Set (EDS). This will be used to
        discretize the EDS, similar to the AIS_resolution  
    labels: str, Optional.
        labels for axes. Accepts TeX math input as it uses matplotlib math
        rendering. Should be in order y1, y2, y3, and so on: 
        labels= ['first label','second label']. Default is None.
        
        

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
        u0_input = input('Enter an initial estimate for your inverse model '
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
        AOS = AOS.reshape((resolution + [AOS.shape[-1]]))
    
    # Switch in between for simplicial of polyhedra calculations.
    if  polytopic_trace  =='simplices':
        AIS_poly, AOS_poly = points2simplices(AIS,AOS)
    elif polytopic_trace =='polyhedra':
        AIS_poly, AOS_poly = points2polyhedra(AIS,AOS)
    else:
        raise ValueError('Invalid option for polytopic tracing. Choose '
                         '"simplices" or "polyhedra".')
        
        
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

            # Use range-based margins to avoid inverted padding on negative coordinates.
            x_range = upper_xaxis - lower_xaxis
            y_range = upper_yaxis - lower_yaxis
            ax.set_xlim(lower_xaxis - 0.05 * x_range,
                        upper_xaxis + 0.05 * x_range)
            ax.set_ylim(lower_yaxis - 0.05 * y_range,
                        upper_yaxis + 0.05 * y_range)

            
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
            
            if labels is not None:
                if len(labels) != 2:
                        raise ValueError('You need two entries for your custom '+
                                  'labels for your 2D system, but entered ' +
                                  'an incorrect number of labels.')
                else:
                    ax.set_xlabel(labels[0])
                    ax.set_ylabel(labels[1])
                            
            else:
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

            if labels is not None:
                if len(labels) != 3:
                        raise ValueError('You need three entries for your custom '+
                                  'labels for your 3D system, but entered ' +
                                  'an incorrect number of labels.')
                else:
                    ax.set_xlabel(labels[0])
                    ax.set_ylabel(labels[1])
                    ax.set_zlabel(labels[2])
                
            else:
                ax.set_xlabel('$y_{1}$')
                ax.set_ylabel('$y_{2}$')
                ax.set_zlabel('$y_{3}$')

            plt.show()
            

        else:
            print(f'Plotting is only supported for 2D and 3D. '
                  f'Your data has dimension {mapped_region.dim}. '
                  f'The operability set is still returned.')
            AS_coords = np.concatenate(Vertices_list, axis=0)

    else:
        print('plot=False selected. The operability set is still returned '
              'as a polytopic region of general dimension.')
        AS_coords = np.concatenate(Vertices_list, axis=0)
    
    
    # Small hack: Inject AS coordinates into return to be able to
    # plot 3D region effortlessly.
    mapped_region = [mapped_region, AS_coords]
    return mapped_region


def OI_eval(AS: pc.Region,
            DS: np.ndarray,
            perspective='outputs',
            hypervol_calc: str = 'robust',
            plot: bool = True,
            labels: str = None):
    
    '''
    Operability Index (OI) calculation. From a Desired Output
    Set (DOS) defined by the user, this function calculates the intersection
    between achievable (AOS) and desired output operation (DOS). Similarly, the 
    OI can be also calculated from the inputs' perspective, as an intersection
    between desired input (DIS) and available input (AIS). This function is able
    to  evaluate the OI in any dimension, ranging from 1-d (length) up to
    higher dimensions (Hypervolumes, > 3-d), aided by the Polytope package.
    
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
    plot: bool, Optional.
        Defines if the plot of operability sets is desired. Default is True.
    labels: str, Optional.
        labels for axes. Accepts TeX math input as it uses matplotlib math
        rendering. Should be in order y1, y2, y3, and so on: 
        labels= ['first label','second label']. Default is None.

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

    if len(inter_list) == 0:
        print('No intersection found between AS and DS. OI = 0.')
        return 0.0

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
                v_intersect = pc.extreme(intersect_i)
                
                if v_intersect is None:
                    continue
                else:
                    processed_intersection = pc.qhull(v_intersect)
                    final_intersection.append(processed_intersection)
                    v_intersect_list.append(v_intersect)

                    # ConvexHull requires >= 2D; for 1D use segment length.
                    if DS_region.dim == 1:
                        volumes_i.append(v_intersect.max() - v_intersect.min())
                    else:
                        volumes_i.append(sp.spatial.ConvexHull(v_intersect).volume)
            
            each_polytope_volume = np.array(volumes_i)
            intersection_volume = each_polytope_volume[0:].sum()
            final_intersection = pc.Region(final_intersection)
            v_DS = pc.extreme(DS_region)
            
            # Evaluate OI
            # ConvexHull requires >= 2D; for 1D use segment length.
            if DS_region.dim == 1:
                DS_volume = v_DS.max() - v_DS.min()
            else:
                DS_volume = sp.spatial.ConvexHull(v_DS).volume
            OI = (intersection_volume / DS_volume) * 100

        else:
            print("For higher dimensions (>7) polytope's hypervolume estimation \
                  is faster. Switching to polytope's calculation.")
            OI = (intersection.volume/DS_region.volume)*100
        
    else:
        raise ValueError('Invalid hypervolume calculation option. Choose '
                         '"robust" or "polytope".')



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
            ax.legend(['DOS']) # Pass label as a list to avoid matplotlib treating the string as individual characters.

            lower_xaxis = min(AS_region.bounding_box[0][0], 
                              DS_region.bounding_box[0][0])
            upper_xaxis = max(AS_region.bounding_box[1][0], 
                              DS_region.bounding_box[1][0])

            lower_yaxis = min(AS_region.bounding_box[0][1], 
                              DS_region.bounding_box[0][1])
            upper_yaxis = max(AS_region.bounding_box[1][1], 
                              DS_region.bounding_box[1][1])

            # Use range-based margins to avoid inverted padding on negative coordinates.
            x_range = upper_xaxis - lower_xaxis
            y_range = upper_yaxis - lower_yaxis
            ax.set_xlim(lower_xaxis - 0.05 * x_range,
                        upper_xaxis + 0.05 * x_range)
            ax.set_ylim(lower_yaxis - 0.05 * y_range,
                        upper_yaxis + 0.05 * y_range)

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

            
            if labels is not None:
                if len(labels) != 2:
                    raise ValueError('You need two entries for your custom '+
                                  'labels for your 2D system, but entered ' +
                                  'an incorrect number of labels.')
                else:   
                    ax.set_xlabel(labels[0])
                    ax.set_ylabel(labels[1])
            else:
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

            if labels is not None:
                    if len(labels) != 3:
                        raise ValueError('You need three entries for your custom '+
                                      'labels for your 3D system, but entered ' +
                                      'an incorrect number of labels.')
                        
                    else:
                        ax.set_xlabel(labels[0])
                        ax.set_ylabel(labels[1])
                        ax.set_zlabel(labels[2])
            else:
                ax.set_xlabel('$y_{1}$')
                ax.set_ylabel('$y_{2}$')
                ax.set_zlabel('$y_{3}$')
                
            plt.show()

        elif DS_region.dim > 3:
            print(f'Plotting is only supported for 2D and 3D. '
                  f'Your data has dimension {DS_region.dim}. '
                  f'The OI value is still available for interpretation.')
    return OI


def rank_designs(models,
                 AIS_bound,
                 DOS_bound,
                 resolution,
                 perspective='outputs',
                 polytopic_trace='simplices',
                 u0=None,
                 lb=None,
                 ub=None,
                 constr=None,
                 method='pounce',
                 ad=False,
                 hypervol_calc='robust',
                 plot=True):
    '''
    Rank two or more steady-state process designs by their Operability Index.

    Each design (a process model) is scored by its Operability Index (OI)
    against the same Desired Output Set (DOS), and the designs are returned
    sorted from most to least operable. The comparison can be made from two
    perspectives:

    - ``'outputs'`` (default): each model is forward-mapped from its Available
      Input Set (AIS) to its Achievable Output Set (AOS) with
      ``multimodel_rep``, and the OI is the output-space index
      ``mu(AOS ∩ DOS) / mu(DOS)``.
    - ``'inputs'``: the DOS is inverse-mapped to the feasible desired input set
      (DIS*) with ``nlp_based_approach``, and the OI is the input-space index
      ``mu(DIS* ∩ AIS) / mu(AIS)``, that is, the share of the available input
      envelope that achieves the desired outputs.

    Both indices are computed with the existing ``OI_eval`` routine, so the
    ranking inherits its geometry. Steady-state models only.

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    models : dict or list of Callable
        The designs to rank. A dict ``{label: model}`` names each design; a
        plain list is accepted and auto-labeled ``'Design 1'``,
        ``'Design 2'``, and so on. Every model maps an input vector ``u`` to an
        output vector ``y`` whose dimension matches ``DOS_bound``.
    AIS_bound : np.ndarray or dict
        Available Input Set bounds, shape ``(n_u, 2)``. A single array is
        shared by all designs; a dict ``{label: bounds}`` gives each design its
        own AIS, so models with different input spaces can be compared against
        the same ``DOS_bound``.
    DOS_bound : np.ndarray
        Desired Output Set bounds, shape ``(n_y, 2)``, shared by all designs.
    resolution : int, array-like or dict
        Discretization resolution passed to the mapping routine. A single value
        is shared by all designs; a dict ``{label: resolution}`` sets it per
        design.
    perspective : {'outputs', 'inputs'}, Optional.
        Space in which the OI is evaluated. Default is ``'outputs'``.
    polytopic_trace : {'simplices', 'polyhedra'}, Optional.
        Polytopic tracing passed to ``multimodel_rep`` for the output
        perspective. Default is ``'simplices'``.
    u0 : np.ndarray, Optional.
        Initial estimate for the inverse mapping (input perspective). Default
        is the midpoint of each design's AIS.
    lb : np.ndarray, Optional.
        Lower bounds for the inverse mapping (input perspective). Default is the
        lower column of each design's AIS, so DIS* lies inside the AIS.
    ub : np.ndarray, Optional.
        Upper bounds for the inverse mapping (input perspective). Default is the
        upper column of each design's AIS.
    constr : dict, Optional.
        Nonlinear constraint forwarded to ``nlp_based_approach`` (input
        perspective). Default is None.
    method : str, Optional.
        NLP solver for the inverse mapping (input perspective). Default is
        ``'pounce'``.
    ad : bool, Optional.
        Whether to use automatic differentiation in the inverse mapping (input
        perspective). Default is False.
    hypervol_calc : {'robust', 'polytope'}, Optional.
        Hypervolume method forwarded to ``OI_eval``. Default is ``'robust'``.
    plot : bool, Optional.
        Whether to draw a ranked bar chart of the OI of each design. Default is
        True.

    Returns
    -------
    ranking : list of dict
        One entry per design, sorted by OI from highest to lowest (the list
        order is the ranking). Each entry has keys ``'label'`` (the design
        name), ``'OI'`` (the Operability Index in percent) and ``'region'``
        (the ``pc.Region`` scored: the AOS for the output perspective, the
        DIS* for the input perspective).

    Notes
    -----
    For the input perspective, the feasible desired input set DIS* is taken as
    the convex hull of the inverse-mapping solution points (``pc.qhull`` over
    the ``nlp_based_approach`` result), so a strongly non-convex DIS* is
    over-approximated by its hull. Because ``nlp_based_approach`` bounds the
    inverse map by the AIS, DIS* lies inside the AIS and the input-space index
    reduces to ``mu(DIS*) / mu(AIS)``.

    References
    ----------
    [1] D. R. Vinson and C. Georgakis. New Measure of Process Output
        Controllability. J. Process Control, 2000.
        https://doi.org/10.1016/S0959-1524(99)00045-1
    '''
    # Normalize the models container into an ordered {label: model} dict.
    if isinstance(models, dict):
        model_map = dict(models)
    else:
        model_map = {f'Design {i + 1}': m for i, m in enumerate(models)}

    if len(model_map) < 2:
        warnings.warn('rank_designs compares two or more designs; only '
                      f'{len(model_map)} was given.')

    # Per-design AIS and resolution: a dict selects per design; anything else
    # is shared by all designs.
    def _per_design(value, key):
        return value[key] if isinstance(value, dict) else value

    ranking = []
    for key, model in model_map.items():
        AIS_i = np.asarray(_per_design(AIS_bound, key), dtype=float)
        res_i = _per_design(resolution, key)

        if perspective == 'outputs':
            # Forward map AIS -> AOS, then score the AOS against the DOS.
            region = multimodel_rep(model,
                                    AIS_i,
                                    res_i,
                                    polytopic_trace=polytopic_trace,
                                    perspective='outputs',
                                    plot=False)
            OI = OI_eval(region,
                         DOS_bound,
                         perspective='outputs',
                         hypervol_calc=hypervol_calc,
                         plot=False)
            scored_region = region[0]

        elif perspective == 'inputs':
            # Inverse map the DOS to DIS*, then score DIS* against the AIS.
            lb_i = AIS_i[:, 0] if lb is None else np.asarray(lb, dtype=float)
            ub_i = AIS_i[:, 1] if ub is None else np.asarray(ub, dtype=float)
            u0_i = (AIS_i.mean(axis=1) if u0 is None
                    else np.asarray(u0, dtype=float))
            fDIS, _, _ = nlp_based_approach(model,
                                            DOS_bound,
                                            res_i,
                                            u0_i,
                                            lb_i,
                                            ub_i,
                                            constr=constr,
                                            method=method,
                                            ad=ad,
                                            warmstart=True,
                                            plot=False)
            fDIS = np.asarray(fDIS, dtype=float)
            dis_region = [pc.Region([pc.qhull(fDIS)]), fDIS]
            OI = OI_eval(dis_region,
                         AIS_i,
                         perspective='inputs',
                         hypervol_calc=hypervol_calc,
                         plot=False)
            scored_region = dis_region[0]

        else:
            raise ValueError("perspective must be 'outputs' or 'inputs'.")

        ranking.append({'label': key, 'OI': float(OI),
                        'region': scored_region})

    # Sort from most to least operable.
    ranking.sort(key=lambda entry: entry['OI'], reverse=True)

    # Ranked table.
    space = 'output' if perspective == 'outputs' else 'input'
    print(f'Design ranking by {space}-space Operability Index:')
    for position, entry in enumerate(ranking, start=1):
        print(f'  {position}. {entry["label"]:<22s} '
              f'OI = {entry["OI"]:6.2f} %')

    # Ranked bar chart, colored with opyrability's default colormap.
    if plot and ranking:
        names = [entry['label'] for entry in ranking]
        ois = [entry['OI'] for entry in ranking]
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(ois)))
        fig, ax = plt.subplots()
        ax.bar(names, ois, color=colors, edgecolor='black')
        ax.set_ylabel(f'{space.capitalize()}-space OI [%]')
        ax.set_title('Design ranking by Operability Index')
        ax.set_ylim(0, max(100.0, max(ois) * 1.1))
        for position, value in enumerate(ois):
            ax.text(position, value, f'{value:.1f}',
                    ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.show()

    return ranking


def nlp_based_approach(model: Callable[..., Union[float, np.ndarray]],
                       DOS_bounds: np.ndarray,
                       DOS_resolution: np.ndarray,
                       u0: np.ndarray,
                       lb: np.ndarray,
                       ub: np.ndarray,
                       constr          = None,
                       method: str     = 'pounce',
                       plot: bool      = True,
                       ad: bool        = False,
                       warmstart: bool = True,
                       labels: str = None,
                       pyomo_solver: str = 'ipopt',
                       print_level: int = 5,
                       problem: str = 'P1',
                       PI_target: Callable[..., Union[float, np.ndarray]] = None,
                       PI_bounds: np.ndarray = None) -> Union[np.ndarray, np.ndarray, list]:
    '''
    Inverse mapping for Process Operability calculations. From a Desired Output
    Set (DOS) defined by the user, this function calculates the closest
    Feasible Desired Ouput set (DOS*) from the AOS and its respective Feasible
    Desired Input Set (DIS*), which gives insight about potential changes in
    design and/or operations of a given process model.
    
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
        Optimization method used. The default is 'pounce' (Pounce, a
        pure-Rust reimplementation of the Ipopt interior-point solver,
        installed as a core dependency). Set 'ipopt' to use cyipopt's
        IPOPT instead (an optional dependency, typically installed via
        conda). Options are:
            For unconstrained problems:

                -'trust-constr'

                -'Nelder-Mead'

                -'ipopt'

                -'pounce'

                -'DE'

            For constrained problems:

                -'ipopt'

                -'pounce'

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
    warmstart: bool
        Turn on/off warm-start of NLP. If 'on', the sucessful solution of the
        current iteration is used as an estimate to the next one. Default is
        True.
    labels: str, Optional.
        labels for axes. Accepts TeX math input as it uses matplotlib math
        rendering. Should be in order u1,u2,u3, y1, y2, y3, and so on:
        labels= ['first u label','second u label', 'first y label', 'second y label'].
        Default is False.
    pyomo_solver: str, Optional.
        NLP solver used when the model is a Pyomo/OMLT builder (see below).
        Either 'ipopt' or 'pounce'. Ignored for plain callable models.
        Default is 'ipopt'.
    print_level: int, Optional.
        Solver output verbosity for the Pyomo/OMLT path (Ipopt convention,
        0 = silent to 12 = most verbose). Ignored for plain callable
        models. Default is 5 (Ipopt's default).
    problem: str, Optional.
        Which operability optimization problem to solve. Options are:

            -'P1' (default): the classic inverse mapping. For each DOS
            grid point, minimize the relative error
            sum(((y - y*)/y)**2) to obtain the DIS*/DOS* pair. This is
            the original behavior of this function, unchanged.

            -'P2': process intensification. Solves
            P1 over the DOS grid, filters the DOS* by the level of
            performance given in PI_bounds (the DOS_PI subset), and
            selects the design among the DIS_PI points that minimizes
            PI_target: Omega = min PI_target subject to u in DIS*,
            y in DOS*. Since P2 is a postprocessing step on the P1
            output, the returned (fDIS, fDOS) can be ranked by any other
            PI_target directly, without re-solving P1.

            -'P3': the bilevel operability framework. The nested program

            Psi = min_{u in U} [PI_target]
            s.t. u*_k in argmin {sum_j ((y_jk - y*_jk)/y_jk)**2 :
            u*_k in U_k, y_k in DOS}, c_1(u*_k) <= 0

            with the inner argmin being problem P1 over the k = 1...p
            DOS grid points and c_1 the process constraints (constr).
            It is solved through its validated sequential equivalence:
            the outcome is the same as solving P1 and P2 in series over
            the full DOS grid (Carrasco dissertation, pp. 56 and 60).

        Default is 'P1'.
    PI_target: Callable, Optional.
        Process intensification target Omega(u) for problems P2/P3, a
        function of the input vector returning the metric to minimize
        (e.g. reactor volume, membrane area, cost). Required for
        P2/P3. Default is None.
    PI_bounds: np.ndarray, Optional.
        Level-of-performance box (the DOS_PI subset of the DOS), shape
        (n_y, 2). Only DOS* points inside this box are eligible for the
        intensified design selection. Default is None (the full
        DOS_bounds are used).

    Notes
    -----
    Pyomo/OMLT models (equation-oriented path): instead of a plain Python
    callable, the model may be a Pyomo builder function with the signature
    model(m, u_vars, y_vars) that adds the process constraints linking the
    input variables u_vars to the output variables y_vars on the Pyomo
    ConcreteModel m, flagged by setting an attribute on the function:
    model.build_pyomo_constraints = True (or is_omlt for OMLT objects).
    The inverse mapping is then solved algebraically with exact first and
    second derivatives from the Pyomo expression graph. The 'ad' and
    'constr' arguments are ignored in this path (define constraints in
    the builder). Pyomo/OMLT support contributed by Heitor F.
    (github @hfsf), PR #33, adapted.

    Returns
    -------
    fDIS: np.ndarray
        Feasible Desired Input Set (DIS*). Array containing the solution for
        each point of the inverse-mapping.
    fDOS: np.ndarray
        Feasible Desired Output Set (DOS*). Array containing the feasible
        output for each feasible input calculated via inverse-mapping.
    message_list: list
        List containing the termination criteria for each optimization run
        performed for each DOS grid point.
    pi_report: dict
        Returned ONLY for problem='P2' or 'P3' (the classic P1 return is
        unchanged). Contains the intensified design 'u_PI', its outputs
        'y_PI', the metric value 'PI_value', the level-of-performance
        subsets 'DIS_PI'/'DOS_PI' (eqs. 5.1-5.2 of Carrasco's
        dissertation), the metric evaluated over the subset 'PI_grid',
        the 'problem' string and a 'success' flag.

    References
    ----------
    [1] J. C. Carrasco and F. V. Lima, “An optimization-based operability 
    framework for process design and intensification of modular natural 
    gas utilization systems,” Comput. & Chem. Eng, 2017. 
    https://doi.org/10.1016/j.compchemeng.2016.12.010

    '''
    
    
    from scipy.optimize import NonlinearConstraint

    # Pyomo/OMLT equation-oriented path detection (PR #33, adapted).
    is_pyomo = _is_pyomo_model(model)
    if is_pyomo:
        warnings.warn("Pyomo/OMLT model detected. The inverse mapping "
                      "will be solved algebraically through Pyomo.",
                      UserWarning)
        if ad is True:
            warnings.warn("The 'ad=True' argument is ignored for "
                          "Pyomo/OMLT models, since exact derivatives "
                          "are obtained natively from the algebraic "
                          "model.", UserWarning)
            ad = False
        if constr is not None:
            warnings.warn("The 'constr' argument is ignored for "
                          "Pyomo/OMLT models. Define the constraints "
                          "inside the model builder function instead.",
                          UserWarning)
            constr = None

    # P1/P2/P3 problem selection validation (Carrasco and Lima).
    if problem not in ('P1', 'P2', 'P3'):
        raise ValueError("problem must be 'P1', 'P2' or 'P3', got "
                         f"'{problem}'.")
    if problem in ('P2', 'P3'):
        if PI_target is None:
            raise ValueError(
                f"problem='{problem}' requires the PI_target argument: "
                "a callable Omega(u) returning the process "
                "intensification metric to minimize (e.g. reactor "
                "volume, membrane area, cost).")
        if is_pyomo:
            raise NotImplementedError(
                "problem='P2'/'P3' with Pyomo/OMLT models is not "
                "supported yet. Use a plain callable model.")

    # cyipopt is optional; import its IPOPT interface only when selected.
    # Pounce (the default) is imported at module level.
    if method == 'ipopt':
        minimize_ipopt = _import_cyipopt_minimize()

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
        from jax import config
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
        
        # Take the gradient of the objective via AD. We deliberately do NOT
        # pass an exact AD Hessian: for the small problems typical here,
        # Ipopt's limited-memory (L-BFGS) approximation, used by both
        # Pounce and cyipopt (>=1.7.0) when no Hessian is supplied, is
        # faster than evaluating a dense AD Hessian every iteration. Both
        # solvers accept an exact Hessian if one is ever needed.
        grad_ad = grad(p1)

        if constr is not None:
            constr['jac']  = (jacrev(constr['fun']))
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

    if is_pyomo:
        # Equation-oriented inverse mapping: per DOS grid point, build a
        # Pyomo ConcreteModel with bounded inputs and free outputs, let the
        # user's builder add the process constraints, and minimize the
        # squared distance to the desired output point with exact
        # derivatives from the algebraic model (PR #33, adapted).
        pyo, pyomo_nlp_solver = _pyomo_solver_factory(pyomo_solver,
                                                      print_level)
        current_u0 = np.asarray(u0, dtype=float)
        for i in tqdm(range(r)):
            m_pyo = pyo.ConcreteModel()
            u_start = current_u0

            def _u_bounds(mm, j):
                return (float(lb[j]), float(ub[j]))

            def _u_init(mm, j):
                return float(u_start[j])

            m_pyo.u = pyo.Var(range(m), bounds=_u_bounds,
                              initialize=_u_init)
            m_pyo.y = pyo.Var(range(dimDOS))

            # The user's builder adds the process model constraints.
            model(m_pyo, m_pyo.u, m_pyo.y)

            DOS_pt = DOSPts[i, :]
            m_pyo.obj = pyo.Objective(
                expr=sum((m_pyo.y[j] - float(DOS_pt[j])) ** 2
                         for j in range(dimDOS)))

            results_pyo = pyomo_nlp_solver.solve(m_pyo, tee=False)
            term = results_pyo.solver.termination_condition
            message_list.append(str(term))

            fDIS[i, :] = [pyo.value(m_pyo.u[j]) for j in m_pyo.u]
            fDOS[i, :] = [pyo.value(m_pyo.y[j]) for j in m_pyo.y]

            # Warm-start: reuse the previous solution as the next initial
            # estimate, rebooting to the first estimate on failure.
            if warmstart and (term == pyo.TerminationCondition.optimal):
                current_u0 = np.asarray(fDIS[i, :], dtype=float)
            else:
                current_u0 = np.asarray(u00, dtype=float)
    else:
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

                    elif method == 'Nelder-Mead':
                        sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                                   args=(model, DOSPts[i, :]),
                                                   method=method,
                                                   options={'fatol': 1e-10,
                                                            'xatol': 1e-10},
                                                   jac=grad_ad)

                    elif method == 'ipopt':
                        sol = minimize_ipopt(p1, x0=u0, bounds=bounds,
                                             args=(model, DOSPts[i, :]),
                                             jac=grad_ad)

                    elif method == 'pounce':
                        # Pounce (>=0.5.0) mirrors the scipy/cyipopt
                        # interface, including args=. The AD gradient is
                        # supplied; the Hessian is left to Ipopt's
                        # limited-memory approximation (faster here).
                        sol = pounce_minimize(p1, x0=u0, bounds=bounds,
                                              args=(model, DOSPts[i, :]),
                                              jac=grad_ad)

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

                    elif method == 'pounce':
                        # Without AD, Pounce falls back to central finite
                        # differences internally (numerical derivatives).
                        sol = pounce_minimize(p1, x0=u0, bounds=bounds,
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

                elif method == 'pounce':
                    # Constrained problems: Pounce consumes the same
                    # constraint dict format as cyipopt; constraint
                    # Jacobians from AD are honored when ad=True, and the
                    # Hessian is approximated internally (limited-memory)
                    # since nonlinear constraints are present.
                    if ad is True:
                        sol = pounce_minimize(p1, x0=u0, bounds=bounds,
                                              args=(model, DOSPts[i, :]),
                                              constraints=constr,
                                              jac=grad_ad)
                    else:
                        sol = pounce_minimize(p1, x0=u0, bounds=bounds,
                                              args=(model, DOSPts[i, :]),
                                              constraints=constr)

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
                    else:
                        # Build NonlinearConstraint in both branches; without AD,
                        # scipy falls back to finite-difference Jacobians/Hessians.
                        con_fun = constr['fun']
                        nlc = NonlinearConstraint((con_fun), -np.inf, 0)
                        sol = sp.optimize.minimize(p1, x0=u0, bounds=bounds,
                                                   args=(model, DOSPts[i, :]),
                                                   method=method,
                                                   constraints=(nlc))

        
        
            # Append results into fDOS, fDIS and message list for each iteration
        
            if warmstart is True:
                if sol.success is True:
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
        plot = False # Assignment, not comparison: disable plot for high-dimensional cases.
        print(f'Plotting is only supported for 2D and 3D. '
              f'DIS* has dimension {fDIS.shape[1]}, DOS* has dimension {fDOS.shape[1]}.')
    else:
        if plot is False:
            pass
        elif plot is True:
            
            if fDIS.shape[1] == 2 and fDOS.shape[1] == 2:
                _, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, 
                                              constrained_layout=True)
                ax1.scatter(fDIS[:, 0], fDIS[:, 1], s=16,
                            c=np.sqrt(fDOS[:, 0]**1 + fDOS[:, 1]**1),
                            cmap=cmap, antialiased=True,
                            lw=lineweight, marker='s',
                            edgecolors=edgecolors, label='DIS*')
                
                
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
                
                
                if labels is not None:
                    if len(labels) != 4:
                        raise ValueError('You need four entries for your custom '+
                                  'labels for your 2x2 system, but entered ' +
                                  'an incorrect number of labels.')
                        
                    else:
                        ax1.set_xlabel(labels[0])
                        ax1.set_ylabel(labels[1])
                        ax2.set_xlabel(labels[2])
                        ax2.set_ylabel(labels[3])
                else:
                    ax1.set_ylabel('$u_{2}$')
                    ax1.set_xlabel('$u_{1}$')
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
                
                
                
                if labels is not None:
                    if len(labels) != 6:
                        raise ValueError('You need six entries for your custom '+
                                  'labels for your 3x3 system, but entered ' +
                                  'an incorrect number of labels.')
                    else:     
                        ax.set_xlabel(labels[0])
                        ax.set_ylabel(labels[1])
                        ax.set_zlabel(labels[2])

                else:
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
                
                
                if labels is not None:
                    ax.set_xlabel(labels[3])
                    ax.set_ylabel(labels[4])
                    ax.set_zlabel(labels[5])

                else:
                    ax.set_xlabel('$y_{1}$')
                    ax.set_ylabel('$y_{2}$')
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
                
                
                
                if labels is not None:
                    if len(labels) != 5:
                        raise ValueError('You need five entries for your custom '+
                                  'labels for your 2x3 system, but entered ' +
                                  'an incorrect number of labels.')
                    else:
                        ax.set_xlabel(labels[0])
                        ax.set_ylabel(labels[1])
                        
                else:
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
                
                
                if labels is not None:
                    ax.set_xlabel(labels[2])
                    ax.set_ylabel(labels[3])
                    ax.set_zlabel(labels[4])

                else:
                    ax.set_xlabel('$y_{1}$')
                    ax.set_ylabel('$y_{2}$')
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
                
                
                if labels is not None:
                    if len(labels) != 5:
                        raise ValueError('You need five entries for your custom '+
                                  'labels for your 3x2 system, but entered ' +
                                  'an incorrect number of labels.')
                    else:   
                        ax.set_xlabel(labels[0])
                        ax.set_ylabel(labels[1])
                        ax.set_zlabel(labels[2])
                        
                else:
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
                
                
                if labels is not None:
                    ax.set_xlabel(labels[3])
                    ax.set_ylabel(labels[4])

                else:
                    ax.set_xlabel('$y_{1}$')
                    ax.set_ylabel('$y_{2}$')
                
                ax.set_title('$DOS*$')
            else:
                print(f'Plotting is only supported for 2D and 3D. '
                      f'DIS* has dimension {fDIS.shape[1]}, DOS* has dimension {fDOS.shape[1]}.')
                plot = False # Assignment, not comparison: disable plot for high-dimensional cases.

    if problem == 'P1':
        return fDIS, fDOS, message_list

    # P2/P3 intensified design selection (Carrasco and Lima): filter the
    # DOS* by the level of performance (DOS_PI, eq. 5.1) and select the
    # design among the DIS_PI points that minimizes the PI target.
    fDIS_arr = np.asarray(fDIS, dtype=float)
    fDOS_arr = np.asarray(fDOS, dtype=float)
    perf = (np.asarray(PI_bounds, dtype=float) if PI_bounds is not None
            else np.asarray(DOS_bounds, dtype=float))
    inside = np.all((fDOS_arr >= perf[:, 0] - 1e-6)
                    & (fDOS_arr <= perf[:, 1] + 1e-6), axis=1)
    if not np.any(inside):
        raise ValueError(
            "No DOS* point satisfies the level of performance "
            "(PI_bounds). Loosen PI_bounds, refine DOS_resolution, or "
            "check that the performance subset overlaps the achievable "
            "outputs.")
    DIS_PI = fDIS_arr[inside]
    DOS_PI = fDOS_arr[inside]
    PI_grid = np.asarray([float(PI_target(u)) for u in DIS_PI])
    i_best = int(np.argmin(PI_grid))
    pi_report = {'u_PI': DIS_PI[i_best],
                 'y_PI': DOS_PI[i_best],
                 'PI_value': float(PI_grid[i_best]),
                 'problem': problem,
                 'DIS_PI': DIS_PI,
                 'DOS_PI': DOS_PI,
                 'PI_grid': PI_grid,
                 'success': True}

    if plot is True and fDIS_arr.shape[1] == 2 and fDOS_arr.shape[1] == 2:
        _plot_pi_sets(fDIS_arr, fDOS_arr, pi_report, DOS_bounds, perf,
                      labels=labels)

    return fDIS, fDOS, message_list, pi_report


def _plot_pi_sets(fDIS, fDOS, pi_report, DOS_bounds, perf, labels=None):
    '''
    Plot the process intensification sets for problems P2/P3 in the style
    of Carrasco's dissertation (Fig. 5.2): DIS* and DOS* as solid dots,
    the DIS_PI/DOS_PI level-of-performance subsets as hollow red circles,
    and the intensified design as a star. 2D only.
    '''
    DIS_PI = pi_report['DIS_PI']
    DOS_PI = pi_report['DOS_PI']
    u_PI = pi_report['u_PI']
    y_PI = pi_report['y_PI']

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    ax1.scatter(fDIS[:, 0], fDIS[:, 1], s=12, c='k', marker='.',
                label='DIS*')
    ax1.scatter(DIS_PI[:, 0], DIS_PI[:, 1], s=60, facecolors='none',
                edgecolors='r', label='$DIS_{PI}$')
    ax1.scatter(u_PI[0], u_PI[1], s=120, c='b', marker='*',
                label='$u_{PI}^*$')
    ax1.set_title('Intensified design selection (inputs)', fontsize=9)
    ax1.legend(fontsize=7)

    # DOS box and performance subset box for spatial reference.
    dos = np.asarray(DOS_bounds, dtype=float)
    ax2.add_patch(mpatches.Rectangle(
        (dos[0, 0], dos[1, 0]), dos[0, 1] - dos[0, 0],
        dos[1, 1] - dos[1, 0], fill=False, edgecolor='gray',
        linestyle='--', linewidth=0.8, label='DOS'))
    ax2.add_patch(mpatches.Rectangle(
        (perf[0, 0], perf[1, 0]), perf[0, 1] - perf[0, 0],
        perf[1, 1] - perf[1, 0], fill=False, edgecolor='g',
        linestyle=':', linewidth=1.0, label='$DOS_{PI}$ box'))
    ax2.scatter(fDOS[:, 0], fDOS[:, 1], s=12, c='k', marker='.',
                label='DOS*')
    ax2.scatter(DOS_PI[:, 0], DOS_PI[:, 1], s=60, facecolors='none',
                edgecolors='r', label='$DOS_{PI}$')
    ax2.scatter(y_PI[0], y_PI[1], s=120, c='b', marker='*',
                label='$y_{PI}^*$')
    ax2.set_title('Intensified design selection (outputs)', fontsize=9)
    ax2.legend(fontsize=7)

    if labels is not None and len(labels) >= 4:
        ax1.set_xlabel(labels[0])
        ax1.set_ylabel(labels[1])
        ax2.set_xlabel(labels[2])
        ax2.set_ylabel(labels[3])
    else:
        ax1.set_xlabel('$u_{1}$')
        ax1.set_ylabel('$u_{2}$')
        ax2.set_xlabel('$y_{1}$')
        ax2.set_ylabel('$y_{2}$')


def create_grid(region_bounds: np.ndarray, region_resolution: np.ndarray):
    
    
    '''
    Create a multidimensional, discretized grid, given the bounds and the
    resolution.
    
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
                AIS_resolution: np.ndarray,
                EDS_bound: np.ndarray = None,
                EDS_resolution: np.ndarray = None,
                plot: bool = True,
                labels: list = None,
                output_dim: int = None,
                pyomo_solver: str = 'ipopt')-> Union[np.ndarray,np.ndarray]:
    '''
    Forward mapping for Process Operability calculations (From AIS to AOS). 
    From an Available Input Set (AIS) bounds and discretization resolution both
    defined by the user, 
    this function calculates the corresponding discretized 
    Available Output Set (AOS).
    
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
    labels: list, Optional.
        Custom axis labels for the plots, input-side labels first and
        output-side labels afterwards (when an EDS is present, the
        disturbance labels follow the manipulated input labels). Accepts
        TeX math input as it uses matplotlib math rendering. Example for
        a 2x2 system: labels=['Flowrate [m3/h]', 'Temperature [K]',
        'Level [m]', 'Concentration [mol/L]']. Default is None, which
        reproduces the standard $u_{i}$/$d_{i}$/$y_{i}$ labels.
    output_dim: int, Optional.
        Number of output variables y. Required when the model is a
        Pyomo/OMLT builder (the output dimension cannot be inferred from
        an algebraic model). Ignored for plain callable models. Default
        is None.
    pyomo_solver: str, Optional.
        NLP solver used to simulate Pyomo/OMLT models in the forward
        mapping ('ipopt' or 'pounce'). Ignored for plain callable models.
        Default is 'ipopt'.

    Notes
    -----
    Pyomo/OMLT models: when the model is a Pyomo builder (a function
    model(m, u_vars, y_vars) flagged with a build_pyomo_constraints
    attribute, or an OMLT object), the forward mapping wraps it into a
    simulation proxy: inputs are fixed at each grid point and the square
    algebraic system is solved for the outputs. The proxy prefers the
    persistent appsi_ipopt interface when available, falling back to the
    standard executable interface. Pyomo/OMLT support contributed by
    Heitor F. (github @hfsf), PR #33, adapted.

    Returns
    -------
    AIS : np.ndarray
        Discretized Available Input Set (AIS).
    AOS : np.ndarray
        Discretized Available Output Set (AOS).
        Discretized Available Output Set (AOS).

    '''
    # Pyomo/OMLT simulation proxy (PR #33, adapted): wrap the algebraic
    # model into a plain callable so the grid loop below runs unchanged.
    if _is_pyomo_model(model):
        if output_dim is None:
            raise ValueError(
                "For Pyomo/OMLT models in the forward mapping you must "
                "provide the 'output_dim' argument (number of output "
                "variables y), since it cannot be inferred from the "
                "algebraic model.")
        warnings.warn("Pyomo/OMLT model detected in the forward mapping. "
                      "Creating a simulation proxy.", UserWarning)

        pyo, _ = _pyomo_solver_factory(pyomo_solver, print_level=0)

        m_sim = pyo.ConcreteModel()
        total_inputs = AIS_bound.shape[0]
        if EDS_bound is not None:
            total_inputs += EDS_bound.shape[0]
        m_sim.u = pyo.Var(range(total_inputs))
        m_sim.y = pyo.Var(range(output_dim))

        # The user's builder adds the process model constraints.
        model(m_sim, m_sim.u, m_sim.y)

        # Constant objective: each grid point is a square simulation.
        m_sim.dummy_obj = pyo.Objective(expr=0, sense=pyo.minimize)

        # Prefer the persistent APPSI interface for speed when available
        # (keeps the solver loaded in memory across grid points).
        sim_solver_name = pyomo_solver
        if pyomo_solver == 'ipopt':
            try:
                if pyo.SolverFactory('appsi_ipopt').available():
                    sim_solver_name = 'appsi_ipopt'
            except Exception:
                pass
        sim_solver = pyo.SolverFactory(sim_solver_name)
        try:
            sim_solver.options['print_level'] = 0
            if sim_solver_name == 'ipopt':
                sim_solver.options['sb'] = 'yes'
                sim_solver.options['warm_start_init_point'] = 'yes'
                sim_solver.options['mu_strategy'] = 'adaptive'
        except Exception:
            pass

        def _pyomo_simulation_proxy(u_values):
            for k_u in range(len(u_values)):
                m_sim.u[k_u].fix(float(u_values[k_u]))
            res_sim = sim_solver.solve(m_sim, tee=False)
            term_sim = res_sim.solver.termination_condition
            if term_sim != pyo.TerminationCondition.optimal:
                for k_y in m_sim.y:
                    m_sim.y[k_y].set_value(0)
                return np.full(output_dim, np.nan)
            return np.array([pyo.value(m_sim.y[k_y]) for k_y in m_sim.y])

        model = _pyomo_simulation_proxy

    # Indexing
    # Check if both EDS parameters are None using identity checks.
    # Avoid `type(x) and type(y) is type(None)` as type() is always truthy.
    if EDS_bound is None and EDS_resolution is None:
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

    # Only populate EDS arrays when both EDS parameters are provided.
    if EDS_bound is not None and EDS_resolution is not None:
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
    # Only populate EDS arrays when both EDS parameters are provided.
    if EDS_bound is not None and EDS_resolution is not None:
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
        # Effective axis labels: the defaults reproduce the standard
        # $u_{i}$/$d_{i}$/$y_{i}$ labels exactly; user-provided labels
        # replace them positionally (inputs first, then disturbances,
        # then outputs).
        n_in_plot = input_map.shape[-1]
        n_out_plot = AOS.shape[-1]
        n_d_plot = 0 if EDS_bound is None else EDS_bound.shape[0]
        in_labels = (['$u_{' + str(i + 1) + '}$'
                      for i in range(n_in_plot - n_d_plot)]
                     + ['$d_{' + str(i + 1) + '}$'
                        for i in range(n_d_plot)])
        out_labels = ['$y_{' + str(i + 1) + '}$'
                      for i in range(n_out_plot)]
        if labels is not None:
            if len(labels) != n_in_plot + n_out_plot:
                raise ValueError('You need ' +
                                 str(n_in_plot + n_out_plot) +
                                 ' entries for your custom labels '
                                 '(inputs/disturbances first, then '
                                 'outputs), but entered ' +
                                 str(len(labels)) + ' labels.')
            in_labels = list(labels[:n_in_plot])
            out_labels = list(labels[n_in_plot:])

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
            
            ax1.set_xlabel(in_labels[0])
            if (EDS_bound and EDS_resolution) is None:
                ax1.set_title('$AIS_{u}$')
            else:
                ax1.set_title(r'$AIS_{u} \, and \, EDS_{d}$')
            ax1.set_ylabel(in_labels[1])


            ax2.scatter(AOS_plot[:, 0], AOS_plot[:, 1], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax2.set_ylabel(out_labels[1])
            plt.xlabel(out_labels[0])
            
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
            
            ax.set_xlabel(in_labels[0])

            # Check if both EDS parameters are None using identity checks.
            # Avoid `type(x) and type(y) is type(None)` as type() is always truthy.
            if EDS_bound is None and EDS_resolution is None:
                ax.set_title('$AIS_{u}$')
            elif EDS_bound.shape[0] == 2:
                ax.set_title(r'$AIS_{u} \, and  \, EDS_{d}$')
            elif EDS_bound.shape[0] == 1:
                ax.set_title(r'$AIS_{u} \, and \, EDS_{d}$')
            ax.set_ylabel(in_labels[1])
            ax.set_zlabel(in_labels[2])



            ax = fig.add_subplot(1,2,2, projection='3d')
            ax.scatter(AOS_plot[:, 0], AOS_plot[:, 1], AOS_plot[:, 2], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2 +
                                  AOS_plot[:, 2]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax.set_ylabel(out_labels[1])
            ax.set_xlabel(out_labels[0])
            ax.set_zlabel(out_labels[2])
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

            ax.set_xlabel(in_labels[0])

            # Check if both EDS parameters are None using identity checks.
            # Avoid `type(x) and type(y) is type(None)` as type() is always truthy.
            if EDS_bound is None and EDS_resolution is None:
                plt.title('$AIS_{u}$')
            else:
                plt.title(r'$AIS_{u} \, and \, EDS_{d}$')
            plt.ylabel(in_labels[1])


            ax = fig.add_subplot(1,2,2, projection='3d')
            ax.scatter(AOS_plot[:, 0], AOS_plot[:, 1], AOS_plot[:, 2], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2 +
                                  AOS_plot[:, 2]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax.set_ylabel(out_labels[1])
            ax.set_xlabel(out_labels[0])
            ax.set_zlabel(out_labels[2])
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

            ax.set_xlabel(in_labels[0])

            # Check if both EDS parameters are None using identity checks.
            # Avoid `type(x) and type(y) is type(None)` as type() is always truthy.
            if EDS_bound is None and EDS_resolution is None:
                ax.set_title('$AIS_{u}$')
            elif EDS_bound.shape[0] == 2:
                # Closing $ required for valid LaTeX rendering.
                ax.set_title(r'$AIS_{u} \, and \, EDS_{d}$')
            elif EDS_bound.shape[0] == 1:
                ax.set_title(r'$AIS_{u} \, and \, EDS_{d}$')
            ax.set_ylabel(in_labels[1])
            ax.set_zlabel(in_labels[2])


            ax = fig.add_subplot(1,2,2)

            ax.scatter(AOS_plot[:, 0], AOS_plot[:, 1], s=16,
                        c=np.sqrt(AOS_plot[:, 0]**2 + AOS_plot[:, 1]**2),
                        cmap=cmap, antialiased=True,
                        lw=lineweight, marker='o',
                        edgecolors=edgecolors)
            ax.set_ylabel(out_labels[1])
            ax.set_xlabel(out_labels[0])

            ax.set_title('$AOS$')
                
            
        else:
            print(f'Plotting is only supported for 2D and 3D. '
                  f'Your data has dimension {input_map.shape[-1]} inputs '
                  f'and {AOS.shape[-1]} outputs.')
            
    else:
        pass
            
    
    return input_map, AOS


def _paired_simplices(AIS: np.ndarray, AOS: np.ndarray) -> Union[list, list]:
    '''
    Build the raw paired input/output simplices from gridded AIS/AOS data,
    PRESERVING the vertex correspondence: column m of an AIS simplex maps to
    column m of its paired AOS simplex (same grid point). This pairing is
    required by barycentric interpolation (e.g. the multilayer MILP of the
    Gazzaneo and Lima framework) and is destroyed by the qhull vertex
    reordering that points2simplices applies for its public output.

    Returns (AIS_simplices, AOS_simplices) as lists of (n_var, n + 1) arrays.
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

    return AIS_simplices, AOS_simplices


def points2simplices(AIS: np.ndarray, AOS: np.ndarray) -> Union[np.ndarray,
                                                                np.ndarray]:
    '''
    Generation of connected simplices (k+1 convex hull of k+1 vertices)
    based on the AIS/AOS points.


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
    AIS_simplices, AOS_simplices = _paired_simplices(AIS, AOS)

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


def _filter_polytope_pairs(AIS_raw, AOS_raw, DOS_bounds, input_constr,
                           AIS_bound):
    '''
    Layer 1, step 3 of the multilayer operability framework (Gazzaneo and
    Lima): from the paired input/output simplices, keep the subset S' of
    pairs whose output simplex intersects the DOS and (when given) whose
    input simplex intersects the linear input-constraint polytope
    A_u u <= b_u. Returns the list of kept pair indices; degenerate
    simplices are skipped. The input-constraint polytope is bounded by
    the AIS box (the polytope library mishandles intersections with
    unbounded polyhedra).
    '''
    DOS_poly = pc.box2poly(np.asarray(DOS_bounds, dtype=float))
    U_poly = None
    if input_constr is not None:
        A_u, b_u = input_constr
        A_u = np.asarray(A_u, dtype=float)
        b_u = np.asarray(b_u, dtype=float).reshape(-1)
        box = np.asarray(AIS_bound, dtype=float)
        n_u = box.shape[0]
        A_box = np.vstack([np.eye(n_u), -np.eye(n_u)])
        b_box = np.hstack([box[:, 1], -box[:, 0]])
        U_poly = pc.Polytope(np.vstack([A_u, A_box]),
                             np.hstack([b_u, b_box]))

    kept = []
    n_degenerate = 0
    for k in range(len(AIS_raw)):
        try:
            aos_poly = pc.qhull(AOS_raw[k].T)
            if pc.is_empty(aos_poly):
                n_degenerate += 1
                continue
            if not pc.is_fulldim(pc.intersect(aos_poly, DOS_poly)):
                continue
            if U_poly is not None:
                ais_poly = pc.qhull(AIS_raw[k].T)
                if pc.is_empty(ais_poly):
                    n_degenerate += 1
                    continue
                if not pc.is_fulldim(pc.intersect(ais_poly, U_poly)):
                    continue
        except Exception:
            n_degenerate += 1
            continue
        kept.append(k)
    if n_degenerate > 0:
        warnings.warn(f'{n_degenerate} degenerate polytope pair(s) were '
                      'skipped during the multimodel filtering.',
                      UserWarning)
    return kept


def _assemble_pi_milp(AIS_S, AOS_S, PI_target, DOS_bounds, input_constr):
    '''
    Assemble the MILP of Layer 1, step 4 of the multilayer operability
    framework (Gazzaneo and Lima): continuous barycentric weights w_{j,k}
    per vertex j of each kept polytope pair k, binaries b_k selecting
    exactly one pair, the inputs/outputs expressed as the weighted vertex
    combinations, and the linearized PI target as the objective. Returns
    (c, integrality, variable_bounds, constraint_list, U_mat, Y_mat).
    '''
    from scipy.optimize import LinearConstraint, Bounds

    K = len(AIS_S)
    n_vert = AIS_S[0].shape[1]
    n_u = AIS_S[0].shape[0]
    n_y = AOS_S[0].shape[0]
    Nw = n_vert * K
    n_var = Nw + K

    # Stacked vertex matrices: u = U_mat @ w, y = Y_mat @ w.
    U_mat = np.hstack([V for V in AIS_S])
    Y_mat = np.hstack([V for V in AOS_S])

    # Objective: barycentric linearization of the PI target (evaluated at
    # the input vertices); binaries carry zero cost.
    c = np.concatenate([
        np.array([float(PI_target(U_mat[:, j])) for j in range(Nw)]),
        np.zeros(K)])

    integrality = np.concatenate([np.zeros(Nw), np.ones(K)])
    variable_bounds = Bounds(np.zeros(n_var), np.ones(n_var))

    constraint_list = []

    # Linking: sum_j w_{j,k} - b_k = 0 for each pair k.
    A_link = np.zeros((K, n_var))
    for k in range(K):
        A_link[k, k * n_vert:(k + 1) * n_vert] = 1.0
        A_link[k, Nw + k] = -1.0
    constraint_list.append(LinearConstraint(A_link, 0.0, 0.0))

    # Exactly one pair is selected: sum_k b_k = 1.
    A_one = np.zeros((1, n_var))
    A_one[0, Nw:] = 1.0
    constraint_list.append(LinearConstraint(A_one, 1.0, 1.0))

    # Output box (the DOS/performance targets): y_lb <= Y w <= y_ub.
    dos = np.asarray(DOS_bounds, dtype=float)
    A_y = np.zeros((n_y, n_var))
    A_y[:, :Nw] = Y_mat
    constraint_list.append(LinearConstraint(A_y, dos[:, 0], dos[:, 1]))

    # Linear input constraints: A_u (U w) <= b_u.
    if input_constr is not None:
        A_u, b_u = input_constr
        A_u = np.asarray(A_u, dtype=float)
        b_u = np.asarray(b_u, dtype=float).reshape(-1)
        A_in = np.zeros((A_u.shape[0], n_var))
        A_in[:, :Nw] = A_u @ U_mat
        constraint_list.append(
            LinearConstraint(A_in, -np.inf, b_u))

    return c, integrality, variable_bounds, constraint_list, U_mat, Y_mat


def milp_based_approach(model: Callable[..., Union[float, np.ndarray]],
                        AIS_bound: np.ndarray,
                        PI_target: Callable[..., Union[float, np.ndarray]],
                        DOS_bounds: np.ndarray,
                        AIS_resolution=3,
                        input_constr: tuple = None,
                        tol: float = 1e-3,
                        max_iter: int = 20,
                        solver_options: dict = None,
                        plot: bool = True,
                        labels: list = None) -> Union[np.ndarray, np.ndarray,
                                                      float, list]:
    '''
    MILP-based iterative algorithm for optimal modular design: Layer 1 of
    the multilayer operability framework of Gazzaneo and Lima. At each
    iteration, the nonlinear process model is simulated on a coarse AIS
    grid, the input-output data are triangulated into paired multimodel
    polytopes (the linearized subsystems), the pairs satisfying the DOS
    and the input constraints are kept, and a mixed-integer linear
    program selects the design point minimizing the linearized process
    intensification target via barycentric interpolation: binaries b_k
    pick exactly one polytope pair and continuous weights w express the
    design as a convex combination of its vertices. The AIS bounds are
    then refined to the winning polytope and the procedure repeats until
    the relative change in the objective falls below tol.

    The MILP is solved with scipy.optimize.milp (the HiGHS solver bundled
    inside scipy, no additional installation needed; for advanced HiGHS
    controls the highspy package is the escalation path).

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    model : Callable
        Process model (forward mapping) y = M(u). Either a plain callable
        or a Pyomo/OMLT builder (auto-detected, simulated through the
        forward-mapping proxy).
    AIS_bound : np.ndarray
        Bounds of the Available Input Set for design, shape (n_u, 2).
    PI_target : Callable
        Process intensification target Omega(u) to minimize (e.g. reactor
        volume, membrane area, cost). Evaluated at the polytope vertices;
        the MILP minimizes its barycentric linearization.
    DOS_bounds : np.ndarray
        Desired Output Set bounds (including PI and sustainable
        manufacturing targets), shape (n_y, 2). The framework requires a
        square system (n_y equal to n_u) for the simplicial multimodel
        representation.
    AIS_resolution : int or array-like, Optional.
        Grid resolution per input dimension for the multimodel
        discretization. The paper uses 3 (a 3^n grid). Default is 3.
    input_constr : tuple, Optional.
        Linear input constraints as (A_u, b_u) with A_u u <= b_u, e.g.
        the plug-flow aspect ratio L/D >= 30 written as
        (np.array([[-1.0, 30.0]]), np.array([0.0])). Default is None.
    tol : float, Optional.
        Relative-improvement stopping tolerance between consecutive
        iterations. Default is 1e-3 (the paper's 0.001).
    max_iter : int, Optional.
        Maximum number of refinement iterations. Default is 20.
    solver_options : dict, Optional.
        Options forwarded to scipy.optimize.milp (e.g.
        {'mip_rel_gap': 1e-3, 'time_limit': 60}). Default is None.
    plot : bool, Optional.
        Plot the iteration history (2D systems only): the refined
        triangulations, the winning polytopes and the optimal design.
        Default is True.
    labels : list, Optional.
        Axis labels, input labels first then output labels, e.g.
        ['L [cm]', 'D [cm]', 'Benzene [mg/h]', 'Conversion [%]'].
        Default is None.

    Returns
    -------
    u_opt : np.ndarray
        The optimal (intensified/modular) design point.
    y_opt : np.ndarray
        The outputs of the optimal design from the barycentric
        interpolation (re-evaluate model(u_opt) for the nonlinear value).
    PI_linearized : float
        The linearized PI target at the optimum (phi), i.e. the quantity
        the MILP actually minimizes (the barycentric interpolation of
        PI_target over the winning simplex vertices).
    PI_true : float
        The true intensification metric, PI_target evaluated directly at
        the optimal design u_opt. Equals PI_linearized when PI_target is
        linear, and converges to it as the simplices shrink; their gap is
        the linearization error.
    history : list
        One dict per iteration with keys 'phi', 'u', 'y', 'PI_true',
        'AIS_simplex'/'AOS_simplex' (the winning input/output simplex),
        'AIS_simplices'/'AOS_simplices' (the full multimodel triangulation
        of the current box), 'AIS_simplices_kept'/'AOS_simplices_kept' (the
        subset reaching the DOS), 'AIS_bound', 'n_pairs', 'E_rel' and
        'milp_status'.

    References
    ----------
    [1] V. Gazzaneo and F. V. Lima. Multilayer Operability Framework for
        Process Design, Intensification, and Modularization of Nonlinear
        Energy Systems. Ind. Eng. Chem. Res., 58, 6069-6079, 2019.
        https://doi.org/10.1021/acs.iecr.8b05482

    [2] V. Gazzaneo, J. C. Carrasco, D. R. Vinson and F. V. Lima.
        Process Operability Algorithms: Past, Present, and Future
        Developments. Ind. Eng. Chem. Res., 59, 2457-2470, 2020.

    '''
    from scipy.optimize import milp

    AIS_bound = np.asarray(AIS_bound, dtype=float)
    DOS_bounds = np.asarray(DOS_bounds, dtype=float)
    n_u = AIS_bound.shape[0]
    n_y = DOS_bounds.shape[0]
    if n_u != n_y:
        raise ValueError(
            "milp_based_approach currently requires a square system "
            f"(n_u == n_y); got {n_u} inputs and {n_y} outputs. The "
            "simplicial multimodel representation of the framework is "
            "defined for square subsystems. Support for non-square "
            "systems is an upcoming feature of opyrability.")

    if np.isscalar(AIS_resolution):
        resolution = [int(AIS_resolution)] * n_u
    else:
        resolution = [int(rr) for rr in AIS_resolution]

    # Pyomo/OMLT models: the forward-mapping proxy needs the output
    # dimension, which here is known from the DOS.
    forward_kwargs = {}
    if _is_pyomo_model(model):
        forward_kwargs['output_dim'] = n_y

    original_range = AIS_bound[:, 1] - AIS_bound[:, 0]
    bounds_current = AIS_bound.copy()
    history = []
    phi_prev = None

    for iteration in range(max_iter):
        # (1) Simulate the model on the current coarse grid and (2) build
        # the paired multimodel simplices (vertex pairing preserved).
        AIS_grid, AOS_grid = AIS2AOS_map(model, bounds_current, resolution,
                                         plot=False, **forward_kwargs)
        AIS_raw, AOS_raw = _paired_simplices(AIS_grid, AOS_grid)

        # (3) Keep the subset S' satisfying output and input constraints.
        kept = _filter_polytope_pairs(AIS_raw, AOS_raw, DOS_bounds,
                                      input_constr, bounds_current)
        if len(kept) == 0:
            if iteration == 0:
                raise ValueError(
                    "No polytope pair satisfies the DOS and input "
                    "constraints (S' is empty). Refine AIS_resolution, "
                    "widen the DOS_bounds, or check the input_constr "
                    "definition.")
            warnings.warn(
                "S' became empty after bound refinement; returning the "
                "best design found so far.", UserWarning)
            break
        AIS_S = [AIS_raw[k] for k in kept]
        AOS_S = [AOS_raw[k] for k in kept]

        # (4) Assemble and solve the design MILP.
        c, integrality, var_bounds, constr_list, U_mat, Y_mat = \
            _assemble_pi_milp(AIS_S, AOS_S, PI_target, DOS_bounds,
                              input_constr)
        res = milp(c=c, constraints=constr_list, integrality=integrality,
                   bounds=var_bounds, options=solver_options)
        if not res.success:
            if iteration == 0:
                raise ValueError(
                    "The design MILP is infeasible at the first "
                    f"iteration (status: {res.message}). Refine "
                    "AIS_resolution or widen the DOS_bounds.")
            warnings.warn(
                f"The design MILP became infeasible ({res.message}); "
                "returning the best design found so far.", UserWarning)
            break

        Nw = U_mat.shape[1]
        w_sol = res.x[:Nw]
        b_sol = res.x[Nw:]
        k_win = int(np.argmax(b_sol))
        u_it = U_mat @ w_sol
        y_it = Y_mat @ w_sol
        phi_it = float(c @ res.x)

        # (5) Relative improvement check.
        if phi_prev is None:
            E_rel = np.inf
        else:
            E_rel = abs(phi_it - phi_prev) / max(abs(phi_prev), 1e-12)

        history.append({'phi': phi_it,
                        'u': u_it,
                        'y': y_it,
                        'PI_true': float(PI_target(u_it)),
                        'AIS_simplex': AIS_S[k_win],
                        'AOS_simplex': AOS_S[k_win],
                        # Full multimodel triangulation of the current box and
                        # the subset that reaches the DOS (for visualization).
                        'AIS_simplices': AIS_raw,
                        'AOS_simplices': AOS_raw,
                        'AIS_simplices_kept': AIS_S,
                        'AOS_simplices_kept': AOS_S,
                        'AIS_bound': bounds_current.copy(),
                        'n_pairs': len(kept),
                        'E_rel': E_rel,
                        'milp_status': res.message})

        if E_rel < tol:
            break
        phi_prev = phi_it

        # (6) New bounds: the bounding box of the winning input simplex,
        # guarded against collapsing to zero width.
        V_win = AIS_S[k_win]
        lo = V_win.min(axis=1)
        hi = V_win.max(axis=1)
        width_guard = 1e-9 * original_range
        hi = np.maximum(hi, lo + width_guard)
        bounds_current = np.column_stack((lo, hi))

    # Best iteration by the linearized objective (the quantity the MILP
    # actually minimizes). Both the linearized objective and the true
    # PI_target evaluated at the design are returned: they coincide as the
    # simplices shrink, and their gap measures the linearization error.
    i_best = int(np.argmin([h['phi'] for h in history]))
    u_opt = history[i_best]['u']
    y_opt = history[i_best]['y']
    PI_linearized = history[i_best]['phi']
    PI_true = history[i_best]['PI_true']

    if plot is True and n_u == 2 and n_y == 2:
        _plot_milp_iterations(history, DOS_bounds, u_opt, y_opt,
                              labels=labels)
    elif plot is True:
        print(f'Plotting is only supported for 2D systems. '
              f'Your design problem has {n_u} inputs and {n_y} outputs.')

    return u_opt, y_opt, PI_linearized, PI_true, history


def _plot_milp_iterations(history, DOS_bounds, u_opt, y_opt, labels=None):
    '''
    Plot the MILP-based iterative refinement (2D), reproducing the
    multimodel-triangulation view of the Gazzaneo and Lima framework
    (Figs. 5-6). For each iteration the full paired triangulation of the
    current AIS box is drawn faded (left = input space, right = output
    space), the simplices whose output reaches the DOS are shaded a little
    stronger, and the winning simplex is highlighted in the iteration
    color. The DOS box and the optimal design (star) are overlaid.
    '''
    cmap_func = plt.get_cmap(cmap, max(len(history), 2))
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    def _fill(ax, V, **kw):
        # V is a (2, 3) simplex; fill its triangle.
        ax.fill(V[0, :], V[1, :], **kw)

    for it, h in enumerate(history):
        color = cmap_func(it / max(len(history) - 1, 1))

        # Full multimodel triangulation of this iteration's box (faded),
        # so the whole partition of the AIS and the image fan over the AOS
        # are visible, not just the winner.
        for VA, VO in zip(h['AIS_simplices'], h['AOS_simplices']):
            _fill(ax1, VA, facecolor=color, alpha=0.05,
                  edgecolor=color, linewidth=0.3)
            _fill(ax2, VO, facecolor=color, alpha=0.05,
                  edgecolor=color, linewidth=0.3)

        # Candidate simplices that reach the DOS (the MILP search set).
        for VA, VO in zip(h['AIS_simplices_kept'], h['AOS_simplices_kept']):
            _fill(ax1, VA, facecolor=color, alpha=0.18,
                  edgecolor=color, linewidth=0.4)
            _fill(ax2, VO, facecolor=color, alpha=0.18,
                  edgecolor=color, linewidth=0.4)

        # Winning simplex of the iteration (highlighted, in the legend).
        _fill(ax1, h['AIS_simplex'], facecolor=color, alpha=0.55,
              edgecolor='k', linewidth=1.1, label=f'Iteration {it + 1}')
        _fill(ax2, h['AOS_simplex'], facecolor=color, alpha=0.55,
              edgecolor='k', linewidth=1.1)

    ax1.scatter(u_opt[0], u_opt[1], s=120, c='k', marker='*', zorder=5,
                label='$u^*$ (optimal design)')
    ax2.scatter(y_opt[0], y_opt[1], s=120, c='k', marker='*', zorder=5)

    dos = np.asarray(DOS_bounds, dtype=float)
    ax2.add_patch(mpatches.Rectangle(
        (dos[0, 0], dos[1, 0]), dos[0, 1] - dos[0, 0],
        dos[1, 1] - dos[1, 0], fill=False, edgecolor='gray',
        linestyle='--', linewidth=0.9, label='DOS'))

    ax1.set_title('AIS triangulation and refinement', fontsize=9)
    ax2.set_title('AOS triangulation and DOS', fontsize=9)
    ax1.legend(fontsize=6)
    ax2.legend(fontsize=7)

    if labels is not None and len(labels) >= 4:
        ax1.set_xlabel(labels[0])
        ax1.set_ylabel(labels[1])
        ax2.set_xlabel(labels[2])
        ax2.set_ylabel(labels[3])
    else:
        ax1.set_xlabel('$u_{1}$')
        ax1.set_ylabel('$u_{2}$')
        ax2.set_xlabel('$y_{1}$')
        ax2.set_ylabel('$y_{2}$')


def implicit_map(model:             Callable[...,Union[float,np.ndarray]], 
                 image_init:        np.ndarray, 
                 domain_bound:      np.ndarray = None, 
                 domain_resolution: np.ndarray = None, 
                 domain_points:     np.ndarray = None,
                 direction:         str = 'forward', 
                 validation:        str = 'predictor-corrector', 
                 tol_cor:           float = 1e-4, 
                 continuation:      str = 'Explicit RK4',
                 derivative:        str = 'jax',
                 jit:               bool = True,
                 step_cutting:      bool = False):
    '''
    Performs implicit mapping of an implicitly defined process F(u,y) = 0.
    F can be a vector-valued, multivariable function, which is typically the 
    case for chemical processes studied in Process Operability. 
    This method relies on the implicit function theorem and automatic
    differentiation in order to obtain the mapping of the required 
    input/output space. The
    mapping "direction" can be set by changing the 'direction' parameter.
    
    Authors: San Dinh & Victor Alves
    

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
        computation of the implicit map. The default is True.
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
    # Pyomo/OMLT models are incompatible with the JAX JIT compilation
    # this function relies on (PR #33, adapted).
    if _is_pyomo_model(model):
        raise NotImplementedError(
            "Implicit mapping is not supported for Pyomo/OMLT models due "
            "to its dependency on JAX JIT compilation. Use AIS2AOS_map "
            "(forward) or nlp_based_approach (inverse) instead.")

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
        from jax import config
        config.update("jax_enable_x64", True)
        config.update('jax_platform_name', 'cpu')
        warnings.filterwarnings('ignore', module='jax._src.lib.xla_bridge')
        import jax.numpy as jnp
        from jax import jit, jacrev
        from jax.experimental.ode import odeint as odeint
        dFdi = jacrev(F, 0)
        dFdo = jacrev(F, 1)
    else:
        raise ValueError('Currently JAX is the only supported option for '
                         'calculating derivatives.')

                 

    if jit is True:
        @jit
        def dodi(ii,oo):
            return -jnp.linalg.pinv(dFdo(ii,oo)) @ dFdi(ii,oo)
        
        @jit
        def dods(oo, s, s_length, i0, iplus):
            return dodi(i0 + (s/s_length)*(iplus - i0), oo) \
                @((iplus - i0)/s_length)
    else:
        def dodi(ii, oo): return -jnp.linalg.pinv(dFdo(ii, oo)) @ dFdi(ii, oo)
        def dods(oo, s, s_length, i0, iplus): return dodi(
            i0 + (s/s_length)*(iplus - i0), oo)@((iplus - i0)/s_length)

        
    #  Initialization step: obtaining first solution
    # sol = root(F_io, image_init,args=domain_bound[:,0])
    
    #  Predictor scheme selection
    if continuation == 'Explicit RK4':
        print('Selected RK4')
        
        
        def predict_RK4(dodi, i0, iplus, o0):
            h = iplus -i0
            k1 = dodi( i0          ,  o0           )
            k2 = dodi( i0 + (1/2)*h,  o0 + (h/2) @ k1)
            ## RK4 classical scheme: each stage uses the previous stage's estimate.
            # k1 → k2 → k3 → k4 (do not reuse k1 in k3).
            k3 = dodi( i0 + (1/2)*h,  o0 + (h/2) @ k2)
            k4 = dodi(np.array(i0+h), o0 +      h@ k3)
            
            return o0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4) @ h
        
        predict = predict_RK4
        do_predict = dodi
        
    elif continuation == 'Explicit Euler':
        
        print('Selected Euler')
        
        def predict_eEuler(dodi, i0, iplus, o0):
            return o0 + dodi(i0,o0)@(iplus -i0)
        
        predict = predict_eEuler
        do_predict = dodi
        
    elif continuation == 'odeint':
        
        print('Selected odeint')
        
        def predict_odeint(dods, i0, iplus, o0):
            s_length = norm(iplus - i0)
            s_span = np.linspace(0.0, s_length, 10)
            sol = odeint(dods, o0, s_span, s_length, i0, iplus)
            return sol[-1,:]
        
        predict = predict_odeint
        do_predict = dods
        
    else:
        raise ValueError('Invalid continuation method. Choose "Explicit RK4", '
                         '"Explicit Euler", or "odeint".')
      
    # This code below is a partial implementation of implicit mapping with a
    # closed path. It works for applications in which the meshgrid can be 
    # inferred from a discrete path.
    # TODO: Work in a definitive solution that can be generalized.
    solv_method =  'hybr'
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

        # Pre-allocate image set with nOutput dimensions (not nInput),
        # as input and output spaces may differ in non-square problems.
        nOutput = image_init.shape[0]
        image_set = np.zeros(domain_resolution + [nOutput]) * np.nan

        #  Initialization step: obtaining first solution
        sol = root(F_io, image_init,args=domain_points[0,:], method=solv_method)
        image_set[0, 0] = sol.x
        
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
        image_set = np.zeros(domain_resolution + [nOutput])*np.nan
        #  Initialization step: obtaining the first solution. The first grid
        #  cell (0, 0, ...) sits at the lower-bound corner of every input,
        #  i.e. domain_bound[:, 0]; image_init is the output guess there.
        sol = root(F_io, image_init, args=domain_bound[:, 0],
                   method=solv_method)
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
     
    # End of partial implementation. Below we have the OG code that I will leave
    # here for my own sake of sanity.
    
    # --------------------------- Previous strategy ------------------------- #
    # # Pre-allocating the domain set
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

    # --------------------------- End of Previous strategy ------------------ #
    
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
                                sol = root(F_io, image_0, args=domain_k, method=solv_method)
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
                            
                    elif validation == 'corrector':     
                        domain_k = domain_set[ID_cell]
                        V_domain_id[:,k] = domain_k
                        
                        sol = root(F_io, image_0, args=domain_k, method=solv_method)
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


# --------------------------------------------------------------------------- #
# Dynamic Operability Analysis
#
# State-space projection framework for time-varying systems. The public
# surface mirrors the steady-state pair (multimodel_rep, OI_eval):
#     dynamic_operability_mapping -- builds the AOS polytope at each k
#     dOI_eval                    -- evaluates the DOI at each k
# Supporting utilities (simulate_mc_trajectories, plot_dynamic_funnel,
# make_pyomo_step_model) follow. Private helpers are at the end.
#
# Based on: Dinh & Lima, Ind. Eng. Chem. Res., 2023
#           Dinh, PhD Dissertation, West Virginia University, 2023
#           Dinh & Lima, Comput. Chem. Eng., 2026
#
# Author: Victor Alves
# --------------------------------------------------------------------------- #

from inspect import signature as _inspect_signature
from scipy.spatial import ConvexHull


def dynamic_operability_mapping(step_model=None,
                                 x0=None,
                                 AIS_bound=None,
                                 AIS_resolution=5,
                                 k_max=10,
                                 EDS_bound=None,
                                 EDS_resolution=None,
                                 A=None, B=None, C=None, B_d=None,
                                 u_ref=None, y_ref=None, d_ref=None,
                                 convergence_tol=1e-4,
                                 plot=True,
                                 labels=None):
    '''
    Obtain the multimodel representation of the output-space Achievable
    Output Set (AOS) for a dynamic process as it evolves over k time steps.
    Propagates the AOS forward using state-space projection from an initial
    state x0, mirroring multimodel_rep in the steady-state case. When the
    system has disturbances, a worst-case propagation is used: the AOS at
    each step is the intersection of the reachable sets under every vertex
    of the Expected Disturbance Set (EDS).

    Two propagation modes are supported:
      -- Nonlinear mode (default): evaluates the user's step_model over
         every vertex of the current state polytope and every AIS grid
         point, taking the convex hull to obtain the next AOS. Works for
         arbitrary nonlinear dynamics.
      -- Linear mode (opt-in): if A, B, C are supplied, uses polytope
         affine transforms and Minkowski sums (Dinh & Lima, Comp. Chem.
         Eng., 2026).

    Author: Victor Alves

    Parameters
    ----------
    step_model : Callable, Optional.
        State transition of the form step_model(x, u) or step_model(x, u, d),
        returning the tuple (x_next, y). Arity is auto-detected via
        inspect.signature. Can be None ONLY when A, B, C matrices are
        supplied (linear fast path); in that case an internal callable
        is constructed for downstream simulation and MC sampling.
    x0 : np.ndarray
        Initial state vector, shape (n_x,). The state-space AOS at k=0
        collapses to this single point.
    AIS_bound : np.ndarray
        Bounds on the Available Input Set (AIS), shape (n_u, 2). Each row
        corresponds to the lower and upper bound of each input dimension.
    AIS_resolution : int or array-like, Optional.
        Discretization grid resolution on the AIS. A scalar applies the
        same resolution in every dimension; an array specifies the
        resolution per dimension. Resolution of 2 samples only the AIS
        corners; higher values capture nonlinear AOS-boundary curvature.
        Default is 5.
    k_max : int
        Maximum number of discrete time steps to propagate the AOS forward.
    EDS_bound : np.ndarray, Optional.
        Bounds on the Expected Disturbance Set (EDS), shape (n_d, 2).
        Required when step_model has arity 3 (or B_d is supplied in the
        linear fast path). Default is None.
    EDS_resolution : int or array-like, Optional.
        Resolution of the disturbance grid. Only the EDS corners are
        required for the worst-case intersection semantics, but denser
        grids are accepted. Default is None (corners only).
    A : np.ndarray, Optional.
        Linear state transition matrix, shape (n_x, n_x). Enables the
        linear fast path when A, B, and C are all given. Default is None.
    B : np.ndarray, Optional.
        Linear input matrix, shape (n_x, n_u). Default is None.
    C : np.ndarray, Optional.
        Linear output matrix, shape (n_y, n_x). Default is None.
    B_d : np.ndarray, Optional.
        Linear disturbance-input matrix, shape (n_x, n_d). Required for
        the linear fast path when disturbances are present. Default is
        None.
    u_ref : np.ndarray, Optional.
        Nominal (reference) input values, shape (n_u,). Linear path only.
        Identified LTI models are usually expressed in deviation
        variables; supplying u_ref lets the AIS_bound be given in
        absolute units, with the deviation u - u_ref fed to the model.
        Default is None (AIS_bound already in deviation form).
    y_ref : np.ndarray, Optional.
        Nominal (reference) output values, shape (n_y,). Linear path
        only. When supplied, the output equation becomes
        y = C x + y_ref so the AOS (and any DOS used downstream) are in
        absolute units. Default is None (outputs in deviation form).
    d_ref : np.ndarray, Optional.
        Nominal (reference) disturbance values, shape (n_d,). Linear
        path only; lets EDS_bound be given in absolute units. Default is
        None.
    convergence_tol : float, Optional.
        Relative change threshold on the AOS volume for early stopping:
        the loop exits when |vol(k) - vol(k-1)| / vol(k-1) falls below
        this tolerance, i.e., the dynamic funnel has reached steady
        shape. Default is 1e-4.
    plot : bool, Optional.
        Whether to generate the standard AOS evolution and volume plots
        (only for 2D or 3D output spaces), mirroring multimodel_rep's
        plot-by-default behavior. Default is True.
    labels : list, Optional.
        Axis labels for the output-space AOS plot. Accepts TeX math input
        as it uses matplotlib math rendering. Should be in order y1, y2,
        and so on: labels=['first label','second label']. Default is None.

    Returns
    -------
    results : dict
        Dictionary containing the following entries:

        'AOS_regions' : list of pc.Region
            Output-space AOS at each time step k = 1..K. Each Region
            contains one polytope and is directly compatible with
            OI_eval (used internally by dOI_eval).
        'AOS_x' : list of pc.Polytope
            State-space AOS at each time step, starting from k = 0.
        'volumes' : np.ndarray
            AOS_y Lebesgue volume per step, used for the convergence
            check and the standard diagnostic plot.
        'k_converged' : int or None
            Step at which the volume-change tolerance was first met;
            None if the loop ran to k_max.

        Additional private keys ('_step_model', '_arity', '_x0',
        '_AIS_bound', '_EDS_bound', '_matrices') are included so that
        downstream routines (dOI_eval, simulate_mc_trajectories,
        plot_dynamic_funnel) can reuse the system definition without
        requiring the user to re-specify it.


    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543

    [2] S. Dinh. Nonlinear Dynamic Analysis and Control of Chemical
        Processes Using Dynamic Operability Framework. PhD Dissertation,
        West Virginia University, 2023.

    [3] S. Dinh and F. V. Lima. Linear dynamic operability analysis with
        state-space projection for the online construction of achievable
        output funnels. Comput. Chem. Eng., 205:109428, 2026.
        https://doi.org/10.1016/j.compchemeng.2025.109428

    '''
    # Validate the two always-required arguments. Everything else is mode-
    # dependent (linear vs nonlinear, EDS present or not).
    if x0 is None:
        raise ValueError("x0 is required.")
    if AIS_bound is None:
        raise ValueError("AIS_bound is required.")
    x0 = np.asarray(x0, dtype=float)
    AIS_bound = np.asarray(AIS_bound, dtype=float)
    if EDS_bound is not None:
        EDS_bound = np.asarray(EDS_bound, dtype=float)

    # Decide between the linear (fast) and nonlinear (general) propagation
    # paths. Linear mode activates when the user supplies A, B, and C; the
    # step_model callable becomes optional since the same dynamics can be
    # constructed from the matrices for downstream simulation use.
    linear_mode = (A is not None and B is not None and C is not None)

    if not linear_mode and step_model is None:
        raise ValueError(
            "step_model must be provided unless A, B, C matrices are given."
        )

    # In linear mode we wrap the matrices into an equivalent callable so
    # that simulate_mc_trajectories and other downstream consumers never
    # have to branch on linear-vs-nonlinear. Arity follows the presence
    # of a disturbance-input matrix.
    A_arr = B_arr = C_arr = Bd_arr = None
    uref_arr = yref_arr = dref_arr = None
    if linear_mode:
        A_arr = np.asarray(A, dtype=float)
        B_arr = np.asarray(B, dtype=float)
        C_arr = np.asarray(C, dtype=float)
        Bd_arr = np.asarray(B_d, dtype=float) if B_d is not None else None

        # Reference (nominal) values let deviation-form identified models
        # work with AIS/EDS bounds and outputs expressed in absolute
        # units: deviations are formed internally and y_ref is added back
        # to the outputs.
        uref_arr = (np.zeros(B_arr.shape[1]) if u_ref is None
                    else np.asarray(u_ref, dtype=float))
        yref_arr = (np.zeros(C_arr.shape[0]) if y_ref is None
                    else np.asarray(y_ref, dtype=float))
        if Bd_arr is not None:
            dref_arr = (np.zeros(Bd_arr.shape[1]) if d_ref is None
                        else np.asarray(d_ref, dtype=float))

        # In linear mode the arity is dictated by whether the user asked
        # for a disturbance analysis on this particular run (EDS_bound
        # present), not by whether B_d is defined in the system. This lets
        # the same matrices be reused for both disturbance-free offline
        # and disturbance-aware robust runs.
        if EDS_bound is None or Bd_arr is None:
            def _linear_step(x, u):
                x_next = A_arr @ x + B_arr @ (u - uref_arr)
                y = C_arr @ x_next + yref_arr
                return x_next, y
            arity = 2
        else:
            def _linear_step(x, u, d):
                x_next = (A_arr @ x + B_arr @ (u - uref_arr)
                          + Bd_arr @ (d - dref_arr))
                y = C_arr @ x_next + yref_arr
                return x_next, y
            arity = 3

        step_callable = _linear_step
    elif u_ref is not None or y_ref is not None or d_ref is not None:
        raise ValueError(
            "u_ref/y_ref/d_ref are only supported in the linear path "
            "(A, B, C matrices); fold reference values into the "
            "step_model callable instead.")
    else:
        # Nonlinear mode: detect the arity of the user's step function.
        # 2 args => step(x, u); 3 args => step(x, u, d). Anything else
        # is ambiguous and raises.
        sig = _inspect_signature(step_model)
        n_params = len(sig.parameters)
        if n_params not in (2, 3):
            raise ValueError(
                "step_model must accept 2 or 3 arguments (x, u) or "
                "(x, u, d); got {} parameters.".format(n_params)
            )
        arity = n_params
        step_callable = step_model

    # Cross-check arity against the presence of EDS_bound so the user
    # gets a clear message instead of a silent mismatch at runtime.
    if arity == 2 and EDS_bound is not None:
        raise ValueError(
            "step_model has 2 arguments (no disturbance) but EDS_bound "
            "was provided. Use a 3-argument step(x, u, d) (or supply "
            "B_d in the linear path) if disturbances are relevant."
        )
    if arity == 3 and EDS_bound is None:
        raise ValueError(
            "step_model has 3 arguments (with disturbance) but "
            "EDS_bound was not provided."
        )

    # ----- Build the AIS sample grid ----- #
    # Resolution of 2 samples only the AIS corners (polygonal AOS
    # boundary). Higher values capture nonlinear curvature of the
    # reachable-set boundary, matching the discretization convention
    # used in multimodel_rep on the steady-state side.
    n_u = AIS_bound.shape[0]
    if np.isscalar(AIS_resolution):
        AIS_res_arr = np.full(n_u, int(AIS_resolution), dtype=int)
    else:
        AIS_res_arr = np.asarray(AIS_resolution, dtype=int)
    AIS_res_arr = np.maximum(AIS_res_arr, 2)

    input_linspaces = [np.linspace(AIS_bound[i, 0], AIS_bound[i, 1],
                                   AIS_res_arr[i])
                       for i in range(n_u)]
    input_grids = np.meshgrid(*input_linspaces, indexing='ij')
    input_vertices = np.column_stack([g.ravel() for g in input_grids])

    # ----- Build the EDS sample grid (only if disturbances present) ----- #
    # Worst-case semantics only require the EDS corners, but the user can
    # request a denser grid via EDS_resolution. The resulting vertex set
    # is looped over and the reachable polytopes are intersected below.
    disturbance_vertices = None
    if EDS_bound is not None:
        n_d = EDS_bound.shape[0]
        if EDS_resolution is None:
            EDS_res_arr = np.full(n_d, 2, dtype=int)
        elif np.isscalar(EDS_resolution):
            EDS_res_arr = np.full(n_d, int(EDS_resolution), dtype=int)
        else:
            EDS_res_arr = np.asarray(EDS_resolution, dtype=int)
        EDS_res_arr = np.maximum(EDS_res_arr, 2)

        dist_linspaces = [np.linspace(EDS_bound[i, 0], EDS_bound[i, 1],
                                      EDS_res_arr[i])
                          for i in range(n_d)]
        dist_grids = np.meshgrid(*dist_linspaces, indexing='ij')
        disturbance_vertices = np.column_stack([g.ravel()
                                                for g in dist_grids])

    # AIS polytope is only needed by the linear-mode Minkowski sum path.
    # The deviation-form dynamics consume u - u_ref, so the box is shifted
    # by the reference input before entering the Minkowski sums.
    AIS_poly = (pc.box2poly(AIS_bound - uref_arr[:, None])
                if linear_mode else None)

    # ----- Initialize AOS at k=0 ----- #
    # The state is fully known at k=0, so the initial state-space AOS is
    # a degenerate point at x0. The helper pads it with a tiny box
    # (eps=1e-6) so the polytope library treats it as full-dimensional.
    AOS_x_list = [_point_or_degenerate_polytope(x0.reshape(1, -1))]
    AOS_regions = []
    volumes = []
    k_converged = None

    # ----- Main propagation loop over k ----- #
    for k in range(k_max):

        # ----- State-space propagation: AOS_x(k) -> AOS_x(k+1) ----- #
        if linear_mode:

            if disturbance_vertices is None:
                # Single Minkowski sum: A X(k) (+) B AIS.
                AOS_x_next = _propagate_state_linear(
                    A_arr, B_arr, None, AOS_x_list[k], AIS_poly
                )
            else:
                # Achievable-set semantics: A X(k) (+) B AIS (+) B_d EDS.
                # The Minkowski sum with the full EDS polytope gives the
                # union of reachable states over every disturbance in EDS,
                # matching the convention used on the steady-state side.
                # The box is shifted by d_ref for deviation-form models.
                EDS_poly_full = pc.box2poly(EDS_bound - dref_arr[:, None])
                AOS_x_next = _propagate_state_linear(
                    A_arr, B_arr, Bd_arr, AOS_x_list[k],
                    AIS_poly, EDS_polytope=EDS_poly_full
                )
            AOS_y_points = None  # deferred until after reduction

        else:

            # Extract vertices of the current state polytope for
            # propagation. On the first iteration this is just x0 (the
            # degenerate-point polytope pads a tiny box around it).
            state_verts = pc.extreme(AOS_x_list[k])
            if state_verts is None or len(state_verts) == 0:
                state_verts = x0.reshape(1, -1)

            if arity == 2:
                # No disturbance: propagate all (state-vertex, input-
                # vertex) pairs through the step function and take the
                # convex hull of the resulting next states. Output points
                # are collected at the same time so we do not have to
                # call the step function a second time.
                next_pts, next_outputs = _propagate_state_nonlinear(
                    step_callable, state_verts, input_vertices
                )
                AOS_x_next = _hull_or_degenerate(next_pts)
                AOS_y_points = next_outputs
            else:
                # Achievable-set semantics under disturbances: propagate
                # every (state vertex, input vertex, disturbance vertex)
                # triplet, then take the convex hull. The AOS therefore
                # reflects the union of reachable states/outputs over all
                # admissible realizations in EDS, matching the steady-
                # state convention.
                all_next_pts = []
                all_outputs = []
                for dv in disturbance_vertices:
                    next_pts, next_outputs = _propagate_state_nonlinear(
                        step_callable, state_verts, input_vertices,
                        d_vec=dv
                    )
                    all_next_pts.append(next_pts)
                    all_outputs.append(next_outputs)
                all_next_pts = np.vstack(all_next_pts)
                AOS_x_next = _hull_or_degenerate(all_next_pts)
                AOS_y_points = np.vstack(all_outputs)

        # Reduce the state polytope to its minimal half-space
        # representation to keep the polytope representation compact
        # across iterations.
        AOS_x_next = pc.reduce(AOS_x_next)
        AOS_x_list.append(AOS_x_next)

        # ----- Output-space AOS ----- #
        # In linear mode, or whenever the nonlinear path did not already
        # build the output points, map state polytope vertices through
        # the output equation to obtain AOS_y.
        if AOS_y_points is None:
            out_verts = pc.extreme(AOS_x_next)
            if out_verts is None or len(out_verts) == 0:
                AOS_y_points = (C_arr @ x0 + yref_arr).reshape(1, -1)
            else:
                AOS_y_points = out_verts @ C_arr.T + yref_arr

        AOS_y_poly = _hull_or_degenerate(AOS_y_points)
        AOS_regions.append(pc.Region([AOS_y_poly]))

        # Record the Lebesgue volume of AOS_y; this drives the early-
        # stopping convergence heuristic and the volume-evolution plot.
        volumes.append(AOS_y_poly.volume)

        # ----- Early-stopping check ----- #
        # Stop once the relative change in AOS volume falls below the
        # user-specified tolerance, indicating the funnel has reached
        # a steady shape.
        if k > 0:
            prev_vol = volumes[-2]
            if prev_vol > 0:
                rel_change = abs(volumes[-1] - prev_vol) / prev_vol
            else:
                rel_change = abs(volumes[-1] - prev_vol)
            if rel_change < convergence_tol:
                k_converged = k
                break

    volumes = np.asarray(volumes)

    # Pack the results. Private keys carry the system definition forward
    # so that dOI_eval, simulate_mc_trajectories, and plot_dynamic_funnel
    # never require the user to redundantly respecify the system.
    results = {
        'AOS_regions': AOS_regions,
        'AOS_x':       AOS_x_list,
        'volumes':     volumes,
        'k_converged': k_converged,
        '_step_model': step_callable,
        '_arity':      arity,
        '_x0':         x0,
        '_AIS_bound':  AIS_bound,
        '_EDS_bound':  EDS_bound,
        '_matrices':   ((A_arr, B_arr, C_arr, Bd_arr)
                        if linear_mode else None),
        '_refs':       ((uref_arr, yref_arr, dref_arr)
                        if linear_mode else None),
    }

    # Auto-plot the AOS evolution and the volume curve, matching the
    # plot-by-default behavior of multimodel_rep on the steady-state side.
    if plot:
        _plot_AOS_evolution(AOS_regions, labels=labels)
        _plot_volume_evolution(volumes)

    return results


def dynamic_operability_nstep(step_model,
                              x0,
                              AIS_bound,
                              k_max,
                              AIS_resolution=3,
                              y0=None,
                              max_vertices=80,
                              convergence_tol=1e-3,
                              plot=False,
                              labels=None):
    '''
    Build the dynamic Achievable Output Set (AOS) funnel by direct n-step
    simulation of a nonlinear step model, reproducing the construction used
    in Dinh & Lima (IECR, 2023, Figures 8-9).

    Unlike dynamic_operability_mapping, which propagates a state-space
    polytope and is therefore limited to low-dimensional states, this
    routine works entirely in the output space and is suited to
    high-dimensional-state models (e.g. spatially discretized PDE/reactor
    models with hundreds of states). From a fixed initial state x0 it
    forward-simulates input sequences and takes the convex hull of the
    reachable outputs at each step k. The reachable set is propagated by
    vertex tracking: every retained boundary state is branched over the
    AIS grid, the resulting outputs are hulled to form AOS(k), and only the
    states whose outputs land on that hull are retained for the next step.
    This keeps the vertex count bounded while tracing the AOS boundary,
    avoiding the scenario-tree blow-up of full input-sequence enumeration.

    Disturbances are handled by the caller: wrap the disturbance into a
    fixed-d, two-argument closure step(x, u) and call this routine once per
    disturbance value, then intersect the resulting AOS regions step-by-step
    (pc.intersect) to obtain the disturbance-robust funnel (Figure 9).

    The returned dictionary follows the same contract as
    dynamic_operability_mapping, so dOI_eval, simulate_mc_trajectories, and
    plot_dynamic_funnel can be chained directly afterward.

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    step_model : Callable
        Two-argument state transition step_model(x, u) returning the tuple
        (x_next, y). x is the full (possibly high-dimensional) state and y
        is the output vector. The step integrates the dynamics over one
        discrete time step internally.
    x0 : np.ndarray
        Initial state vector. The funnel starts from this single point.
    AIS_bound : np.ndarray
        Bounds on the Available Input Set, shape (n_u, 2).
    k_max : int
        Maximum number of discrete time steps to propagate forward.
    AIS_resolution : int or array-like, Optional.
        Grid resolution on the AIS used to branch each retained state. A
        scalar applies the same resolution to every input dimension.
        Default is 3 (matching the MATLAB reference implementation).
    y0 : np.ndarray, Optional.
        Output at the initial state x0. When provided, a degenerate AOS
        point at y0 is prepended as the k=0 funnel slice so the funnel
        starts from the steady state. Default is None.
    max_vertices : int, Optional.
        Cap on the number of boundary states retained between steps. When
        the hull has more vertices, they are evenly subsampled. Default 80.
    convergence_tol : float, Optional.
        Relative change in AOS area below which the funnel is considered
        time-invariant and propagation stops early. Default is 1e-3.
    plot : bool, Optional.
        Whether to draw the AOS evolution and area curves. Default False.
    labels : list, Optional.
        Output-space axis labels. Default None.

    Returns
    -------
    results : dict
        Same contract as dynamic_operability_mapping: 'AOS_regions' (list of
        pc.Region per step), 'volumes', 'k_converged', plus the private keys
        '_step_model', '_arity', '_x0', '_AIS_bound', '_EDS_bound',
        '_matrices' used by the downstream dynamic-operability functions.

    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543
    '''
    from scipy.spatial import ConvexHull

    x0 = np.asarray(x0, dtype=float)
    AIS_bound = np.asarray(AIS_bound, dtype=float)
    n_u = AIS_bound.shape[0]

    # Build the AIS grid that every retained state is branched over.
    if np.isscalar(AIS_resolution):
        res = [int(AIS_resolution)] * n_u
    else:
        res = [int(r) for r in AIS_resolution]
    res = [max(r, 2) for r in res]
    axes = [np.linspace(AIS_bound[i, 0], AIS_bound[i, 1], res[i])
            for i in range(n_u)]
    grids = np.meshgrid(*axes, indexing='ij')
    input_grid = np.column_stack([g.ravel() for g in grids])

    def _hull_vertex_indices(points):
        # Indices of the convex-hull vertices of a 2D (or nD) point cloud.
        # Falls back to all points when the cloud is degenerate (collinear,
        # coincident, or too few points for qhull).
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < pts.shape[1] + 1:
            return list(range(pts.shape[0]))
        try:
            return list(ConvexHull(pts).vertices)
        except Exception:
            return list(range(pts.shape[0]))

    def _dedup(states, ys, decimals=6):
        # Collapse boundary states that map to the same output (rounded),
        # keeping the vertex set small without losing distinct boundary pts.
        seen = {}
        for s, y in zip(states, ys):
            key = tuple(np.round(np.atleast_1d(y), decimals))
            if key not in seen:
                seen[key] = s
        return list(seen.values())

    AOS_regions = []
    volumes = []
    k_converged = None

    # Optional k=0 slice: a degenerate point at the steady-state output.
    if y0 is not None:
        y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        AOS_regions.append(
            pc.Region([_point_or_degenerate_polytope(y0.reshape(1, -1))]))
        volumes.append(0.0)

    kept_states = [x0]

    for k in range(k_max):
        cand_states = []
        cand_y = []
        for s in kept_states:
            for u in input_grid:
                x_next, y_next = step_model(s, u)
                cand_states.append(np.asarray(x_next, dtype=float))
                cand_y.append(np.atleast_1d(np.asarray(y_next, dtype=float)))
        cand_y = np.asarray(cand_y, dtype=float)

        # AOS(k) is the convex hull of all reachable outputs this step.
        AOS_poly = _hull_or_degenerate(cand_y)
        AOS_regions.append(pc.Region([AOS_poly]))
        volumes.append(AOS_poly.volume)

        # Retain only the boundary states for the next step's branching.
        idx = _hull_vertex_indices(cand_y)
        kept_states = _dedup([cand_states[i] for i in idx],
                             [cand_y[i] for i in idx])
        if len(kept_states) > max_vertices:
            sel = np.linspace(0, len(kept_states) - 1,
                              max_vertices).astype(int)
            kept_states = [kept_states[i] for i in sel]

        # Early stop once the funnel area stops changing.
        if len(volumes) > 1 and volumes[-2] > 0:
            rel = abs(volumes[-1] - volumes[-2]) / volumes[-2]
            if rel < convergence_tol:
                k_converged = k
                break

    volumes = np.asarray(volumes, dtype=float)
    results = {
        'AOS_regions': AOS_regions,
        'volumes':     volumes,
        'k_converged': k_converged,
        '_step_model': step_model,
        '_arity':      2,
        '_x0':         x0,
        '_AIS_bound':  AIS_bound,
        '_EDS_bound':  None,
        '_matrices':   None,
        '_refs':       None,
    }

    if plot:
        _plot_AOS_evolution(AOS_regions, labels=labels)
        _plot_volume_evolution(volumes)

    return results


def dOI_eval(mapping_results,
             DOS,
             plot=True,
             labels=None):
    '''
    Evaluate the Dynamic Operability Index (dOI) at each time step of a
    dynamic operability mapping. For each step k, the output-space AOS
    region is intersected with the Desired Output Set (DOS) using the
    existing steady-state OI_eval routine, and the resulting scalar OI
    value is collected into a time-series. This function is the dynamic
    counterpart to OI_eval and is meant to be chained directly after
    dynamic_operability_mapping.

    Author: Victor Alves

    Parameters
    ----------
    mapping_results : dict
        Dictionary returned by dynamic_operability_mapping. The list of
        output-space AOS regions is read from the 'AOS_regions' key.
    DOS : np.ndarray
        Bounds on the Desired Output Set (DOS), shape (n_y, 2). Each row
        corresponds to the lower and upper bound of each output dimension.
        Same convention as the DS argument of OI_eval on the steady-state
        side.
    plot : bool, Optional.
        Whether to plot the dOI convergence curve (dOI vs time step k).
        Default is True.
    labels : list, Optional.
        Currently unused; reserved for symmetry with OI_eval's signature.
        Default is None.

    Returns
    -------
    dOI : np.ndarray
        Dynamic Operability Index at each time step, shape (K,), values
        in the range [0, 100].


    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543

    [2] S. Dinh. Nonlinear Dynamic Analysis and Control of Chemical
        Processes Using Dynamic Operability Framework. PhD Dissertation,
        West Virginia University, 2023.

    [3] S. Dinh and F. V. Lima. Linear dynamic operability analysis with
        state-space projection for the online construction of achievable
        output funnels. Comput. Chem. Eng., 205:109428, 2026.
        https://doi.org/10.1016/j.compchemeng.2025.109428

    '''
    # Pull the time-indexed AOS regions; each is a pc.Region wrapping the
    # output-space polytope at that step, as expected by OI_eval.
    AOS_regions = mapping_results['AOS_regions']
    DOS = np.asarray(DOS, dtype=float)

    # Compute the DOI per step by delegating to the steady-state OI_eval.
    # Plotting is suppressed there; the dynamic convergence curve is
    # drawn separately from the collected time-series below.
    dOI_values = []
    for region in AOS_regions:
        dOI_k = _dOI_at_step(region, DOS)
        dOI_values.append(dOI_k)

    dOI_array = np.asarray(dOI_values, dtype=float)

    # Plot the dOI convergence trajectory if requested.
    if plot and len(dOI_array) > 0:
        _plot_dOI_convergence(dOI_array)

    return dOI_array


def simulate_mc_trajectories(mapping_results,
                             n_steps=None,
                             n_trajectories=50,
                             seed=None):
    '''
    Simulate Monte Carlo output trajectories using randomly sampled AIS
    (and, if applicable, EDS) inputs at each time step. The step model
    and system definition are read directly from the dictionary returned
    by dynamic_operability_mapping, so no system parameters need to be
    repeated by the user. Trajectories are useful for verifying that
    arbitrary admissible input sequences yield outputs contained in the
    AOS funnel.

    Author: Victor Alves

    Parameters
    ----------
    mapping_results : dict
        Dictionary returned by dynamic_operability_mapping. The step
        callable, its arity, the initial state, the AIS bounds, and
        (optionally) the EDS bounds are read from the private keys
        '_step_model', '_arity', '_x0', '_AIS_bound', and '_EDS_bound'.
    n_steps : int, Optional.
        Number of time steps to simulate. If None (default), the number
        of steps is taken from the length of mapping_results['AOS_regions'].
    n_trajectories : int, Optional.
        Number of independent random trajectories to simulate. Default
        is 50.
    seed : int, Optional.
        Random seed for reproducibility. Default is None.

    Returns
    -------
    trajectories : np.ndarray
        Output trajectories, shape (n_trajectories, n_steps + 1, n_y).
        Index 0 along axis 1 corresponds to k=0 (the initial output y0
        obtained by evaluating the step model from x0 with the first
        sampled input).


    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543

    '''
    # Recover the system definition stored by dynamic_operability_mapping.
    step_model = mapping_results['_step_model']
    arity = mapping_results['_arity']
    x0 = np.asarray(mapping_results['_x0'])
    AIS_bound = np.asarray(mapping_results['_AIS_bound'])
    EDS_bound = mapping_results.get('_EDS_bound')

    # If the user did not provide n_steps, match the mapping horizon so
    # MC trajectories line up one-to-one with the funnel polytopes.
    if n_steps is None:
        n_steps = len(mapping_results['AOS_regions'])

    rng = np.random.default_rng(seed)
    n_u = AIS_bound.shape[0]
    n_d = EDS_bound.shape[0] if (arity == 3 and EDS_bound is not None) else 0

    # Probe the output dimension by running a single step from x0 with
    # input/disturbance at the AIS/EDS center. Using means avoids
    # triggering edge-case corner behavior during the probe.
    u_center = AIS_bound.mean(axis=1)
    if arity == 2:
        _, y_probe = step_model(x0, u_center)
    else:
        d_center = EDS_bound.mean(axis=1)
        _, y_probe = step_model(x0, u_center, d_center)
    y_probe = np.atleast_1d(np.asarray(y_probe, dtype=float))
    n_y = y_probe.shape[0]

    trajectories = np.zeros((n_trajectories, n_steps + 1, n_y))

    # Simulate each trajectory by drawing uniform samples from the AIS
    # (and EDS, when disturbances are present) at every step.
    for t in range(n_trajectories):
        x = x0.copy()

        # Initial output at k=0: a single sample to fill index 0.
        if arity == 2:
            u0 = (AIS_bound[:, 0]
                  + rng.random(n_u) * (AIS_bound[:, 1] - AIS_bound[:, 0]))
            _, y0_sample = step_model(x, u0)
        else:
            u0 = (AIS_bound[:, 0]
                  + rng.random(n_u) * (AIS_bound[:, 1] - AIS_bound[:, 0]))
            d0 = (EDS_bound[:, 0]
                  + rng.random(n_d) * (EDS_bound[:, 1] - EDS_bound[:, 0]))
            _, y0_sample = step_model(x, u0, d0)
        trajectories[t, 0, :] = np.asarray(y0_sample, dtype=float)

        # Propagate forward n_steps times and store each output.
        for k in range(n_steps):
            u = (AIS_bound[:, 0]
                 + rng.random(n_u) * (AIS_bound[:, 1] - AIS_bound[:, 0]))
            if arity == 2:
                x, y = step_model(x, u)
            else:
                d = (EDS_bound[:, 0]
                     + rng.random(n_d)
                     * (EDS_bound[:, 1] - EDS_bound[:, 0]))
                x, y = step_model(x, u, d)
            trajectories[t, k + 1, :] = np.asarray(y, dtype=float)

    return trajectories


def _plotly_camera(view_elev, view_azim, r=2.0):
    '''Convert matplotlib-style elevation/azimuth angles (degrees) to a
    plotly scene camera eye position.'''
    e = np.radians(view_elev)
    az = np.radians(view_azim)
    return dict(eye=dict(x=r * np.cos(e) * np.cos(az),
                         y=r * np.cos(e) * np.sin(az),
                         z=r * np.sin(e)))


def _plotly_colorscale(colormap, n=33):
    '''Build a plotly colorscale from a matplotlib colormap name.'''
    from matplotlib import colors as mcolors
    cmap = plt.get_cmap(colormap)
    return [[float(f), mcolors.to_hex(cmap(float(f)))]
            for f in np.linspace(0.0, 1.0, n)]


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "engine='plotly' requires the plotly package. Install it "
            "with: pip install plotly") from exc
    return go


def _plotly_funnel(AOS_regions, DOS=None, dOI=None, labels=None,
                   alpha=0.25, colormap='rainbow', view_elev=20,
                   view_azim=-60, orientation='landscape',
                   mc_trajectories=None, mc_color='green', mc_alpha=0.4,
                   mc_linewidth=0.6):
    '''Interactive plotly version of the dynamic operability funnel.
    Returns a plotly Figure that supports pan/rotate/zoom both in Jupyter
    and in static HTML exports (e.g. the Jupyter Book documentation).'''
    go = _import_plotly()
    from matplotlib import colors as mcolors

    landscape = orientation == 'landscape'
    n_steps = len(AOS_regions)
    use_dOI = dOI is not None and len(dOI) == n_steps
    dOI_arr = np.asarray(dOI, dtype=float) if use_dOI else None
    cmap = plt.get_cmap(colormap)

    def coords(V, k_val):
        kk = np.full(len(V), float(k_val))
        if landscape:
            return V[:, 0], kk, V[:, 1]
        return V[:, 0], V[:, 1], kk

    fig = go.Figure()

    # DOS reference: a rectangular column swept along the time axis, drawn
    # with darker end faces plus thick near-black outlines and connector edges
    # so it stands out clearly behind the funnel.
    if DOS is not None:
        dos = np.asarray(DOS, dtype=float)
        dos_edge = '#222222'
        rect = np.array([[dos[0, 0], dos[1, 0]], [dos[0, 1], dos[1, 0]],
                         [dos[0, 1], dos[1, 1]], [dos[0, 0], dos[1, 1]]])
        first_dos = True
        for k_val in (1, n_steps):
            # End face (darker fill).
            x, y, z = coords(rect, k_val)
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z, i=[0, 0], j=[1, 2], k=[2, 3],
                color=DS_COLOR, opacity=0.5, hoverinfo='skip',
                name='DOS', legendgroup='DOS', showlegend=False))
            # Thick dark outline of the end face (also the legend entry).
            loop = np.vstack([rect, rect[0:1]])
            xo, yo, zo = coords(loop, k_val)
            fig.add_trace(go.Scatter3d(
                x=xo, y=yo, z=zo, mode='lines',
                line=dict(color=dos_edge, width=6),
                name='DOS', legendgroup='DOS', showlegend=first_dos,
                hoverinfo='skip'))
            first_dos = False
        # Thick dark connector edges joining the two faces into a column.
        for v in rect:
            seg = np.array([v, v])
            x, y, z = coords(seg, 1)
            x2, y2, z2 = coords(seg, n_steps)
            fig.add_trace(go.Scatter3d(
                x=[x[0], x2[0]], y=[y[0], y2[0]], z=[z[0], z2[0]],
                mode='lines',
                line=dict(color=dos_edge, width=6),
                legendgroup='DOS', hoverinfo='skip', showlegend=False))

    # One filled slice (Mesh3d fan triangulation) + outline per step.
    for i, region in enumerate(AOS_regions):
        verts = pc.extreme(region.list_poly[0])
        k_val = i + 1
        if use_dOI:
            frac = float(np.clip(dOI_arr[i] / 100.0, 0.0, 1.0))
        else:
            frac = i / max(n_steps - 1, 1)
        color = mcolors.to_hex(cmap(frac))
        hover = (f'k = {k_val}' + (f'<br>dOI = {dOI_arr[i]:.1f} %'
                                   if use_dOI else ''))

        if verts is None or len(verts) < 3:
            # Degenerate slice (e.g. the k = 0 steady-state point).
            if verts is not None and len(verts) > 0:
                x, y, z = coords(np.atleast_2d(verts.mean(axis=0)), k_val)
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z, mode='markers',
                    marker=dict(size=4, color=color),
                    hovertext=hover, hoverinfo='text', showlegend=False))
            continue

        center = verts.mean(axis=0)
        order = np.argsort(np.arctan2(verts[:, 1] - center[1],
                                      verts[:, 0] - center[0]))
        ordered = verts[order]
        m = len(ordered)
        x, y, z = coords(ordered, k_val)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0] * (m - 2), j=list(range(1, m - 1)),
            k=list(range(2, m)),
            color=color, opacity=alpha + 0.1,
            hovertext=hover, hoverinfo='text', showlegend=False))
        loop = np.vstack([ordered, ordered[0:1]])
        x, y, z = coords(loop, k_val)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color=color, width=4),
            hovertext=hover, hoverinfo='text', showlegend=False))

    # Monte Carlo trajectories: one trace with None-separated segments.
    if mc_trajectories is not None:
        xs, ys_, zs = [], [], []
        n_traj, n_pts, _ = mc_trajectories.shape
        for t in range(n_traj):
            traj = mc_trajectories[t]
            ks = np.arange(n_pts, dtype=float)
            if landscape:
                xs += list(traj[:, 0]) + [None]
                ys_ += list(ks) + [None]
                zs += list(traj[:, 1]) + [None]
            else:
                xs += list(traj[:, 0]) + [None]
                ys_ += list(traj[:, 1]) + [None]
                zs += list(ks) + [None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys_, z=zs, mode='lines',
            line=dict(color=mc_color, width=max(1, int(mc_linewidth * 3))),
            opacity=mc_alpha, hoverinfo='skip', showlegend=False))

    # Invisible marker carrying the colorbar (dOI % or time step).
    cbar_title = 'dOI (%)' if use_dOI else 'Time step k'
    cmax = 100 if use_dOI else n_steps
    fig.add_trace(go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None], mode='markers',
        marker=dict(size=0.1, color=[0, cmax], cmin=0, cmax=cmax,
                    colorscale=_plotly_colorscale(colormap),
                    showscale=True,
                    colorbar=dict(title=cbar_title, len=0.6)),
        hoverinfo='skip', showlegend=False))

    y1_label = labels[0] if (labels is not None and len(labels) >= 2) \
        else 'y1'
    y2_label = labels[1] if (labels is not None and len(labels) >= 2) \
        else 'y2'
    if landscape:
        scene = dict(xaxis_title=y1_label, yaxis_title='Time step k',
                     zaxis_title=y2_label)
    else:
        scene = dict(xaxis_title=y1_label, yaxis_title=y2_label,
                     zaxis_title='Time step k')
    scene['aspectmode'] = 'cube'
    scene['camera'] = _plotly_camera(view_elev, view_azim)
    fig.update_layout(scene=scene, width=850, height=650,
                      title='Dynamic Operability Funnel',
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _plotly_scenarios(per, inter_regions, n_k, labels_list, labels=None,
                      colors=None, title=None, orientation='landscape',
                      view_elev=18, view_azim=-50):
    '''Interactive plotly version of the scenario funnels + robust
    intersection plot. Pan/rotate/zoom works in Jupyter and static HTML.'''
    go = _import_plotly()
    landscape = orientation == 'landscape'

    def _ordered(poly):
        V = pc.extreme(poly)
        if V is None or len(V) < 3:
            return V
        c = V.mean(axis=0)
        return V[np.argsort(np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))]

    def coords(V, k_val):
        kk = np.full(len(V), float(k_val))
        if landscape:
            return V[:, 0], kk, V[:, 1]
        return V[:, 0], V[:, 1], kk

    if colors is None:
        cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
            'color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        colors = [cycle[i % len(cycle)] for i in range(len(labels_list))]

    fig = go.Figure()

    # Scenario outlines, one None-separated trace per scenario so the
    # legend has exactly one entry per scenario.
    for j, lab in enumerate(labels_list):
        regions = per[lab]['AOS_regions']
        xs, ys_, zs = [], [], []
        for k in range(min(n_k, len(regions))):
            V = _ordered(regions[k].list_poly[0])
            if V is None:
                continue
            loop = np.vstack([V, V[0:1]])
            x, y, z = coords(loop, k)
            xs += list(x) + [None]
            ys_ += list(y) + [None]
            zs += list(z) + [None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys_, z=zs, mode='lines',
            line=dict(color=colors[j], width=3),
            name=lab, hoverinfo='name'))

    # Robust intersection, filled in blue.
    showleg = True
    for k in range(n_k):
        reg = inter_regions[k]
        if len(reg) == 0:
            continue
        V = _ordered(reg.list_poly[0])
        if V is None or len(V) < 3:
            continue
        m = len(V)
        x, y, z = coords(V, k)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0] * (m - 2), j=list(range(1, m - 1)),
            k=list(range(2, m)),
            color='blue', opacity=0.55,
            name='Robust intersection', showlegend=showleg,
            hovertext=f'k = {k} (robust)', hoverinfo='text'))
        showleg = False

    y1_label = labels[0] if (labels is not None and len(labels) >= 2) \
        else 'y1'
    y2_label = labels[1] if (labels is not None and len(labels) >= 2) \
        else 'y2'
    if landscape:
        scene = dict(xaxis_title=y1_label, yaxis_title='Time step k',
                     zaxis_title=y2_label)
    else:
        scene = dict(xaxis_title=y1_label, yaxis_title=y2_label,
                     zaxis_title='Time step k')
    scene['aspectmode'] = 'cube'
    scene['camera'] = _plotly_camera(view_elev, view_azim)
    fig.update_layout(scene=scene, width=850, height=650,
                      title=(title or 'Dynamic operability: scenarios '
                                      'and robust intersection'),
                      margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(x=0.01, y=0.99))
    return fig


def _show_if_notebook(fig):
    '''Display a plotly figure when running inside an IPython kernel,
    mirroring matplotlib's display-at-cell-end behavior; no-op in plain
    scripts and test runs.'''
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            fig.show()
    except Exception:
        pass


def plot_dynamic_funnel(mapping_results,
                        DOS=None,
                        dOI=None,
                        labels=None,
                        alpha=0.25,
                        colormap='rainbow',
                        view_elev=20,
                        view_azim=-60,
                        orientation='landscape',
                        engine='matplotlib',
                        mc_trajectories=None,
                        mc_color='green',
                        mc_alpha=0.4,
                        mc_linewidth=0.6,
                        show=True):
    '''
    Plot the 3D dynamic operability funnel by stacking the output-space
    AOS polytopes along the time axis, as in Dinh & Lima (IECR 2023 and
    Comput. Chem. Eng. 2026). In the default landscape orientation the
    time axis is horizontal and the funnel opens sideways, matching the
    figures of the publications; each time step k is rendered as a filled
    polygon at its time coordinate, producing the characteristic expanding
    or contracting funnel shape. Monte Carlo trajectories returned by
    simulate_mc_trajectories may be overlaid for visual containment
    verification.

    Author: Victor Alves

    Parameters
    ----------
    mapping_results : dict
        Dictionary returned by dynamic_operability_mapping.
    DOS : np.ndarray, Optional.
        Bounds on the Desired Output Set (DOS), shape (n_y, 2). When
        provided, the DOS box is drawn as a translucent reference at
        the first and last time steps. Default is None.
    dOI : array-like, Optional.
        Per-time-step Dynamic Operability Index values (as returned by
        dOI_eval), one per AOS region. When provided, each funnel slice is
        colored by its dOI value and the colorbar is rescaled to the
        operability index in [0, 100] % instead of the time step. This is
        how dynamic_operability shows the dOI directly on the funnel.
        Default is None (color slices by time step).
    labels : list, Optional.
        Axis labels for the two output variables, e.g. ['$y_1$','$y_2$'].
        Default is None.
    alpha : float, Optional.
        Transparency for the AOS polygon faces. Default is 0.25.
    colormap : str, Optional.
        Matplotlib colormap name for coloring time steps. Default is
        'rainbow' to match opyrability's default cmap.
    view_elev : float, Optional.
        Elevation angle for the 3D view in degrees. Default is 20.
    view_azim : float, Optional.
        Azimuth angle for the 3D view in degrees. Default is -60.
    orientation : {'landscape', 'vertical'}, Optional.
        Funnel orientation. 'landscape' (default) places time on a
        horizontal axis with the second output on the vertical axis, as
        in the figures of Dinh & Lima; 'vertical' stacks the time steps
        upward along the vertical axis.
    engine : {'matplotlib', 'plotly'}, Optional.
        Plotting engine. 'matplotlib' (default) produces a static 3D
        figure. 'plotly' produces an interactive WebGL figure that
        supports pan, rotate, and zoom both in Jupyter and in static
        HTML exports such as the Jupyter Book documentation (requires
        the plotly package); the second return value is then None.
    mc_trajectories : np.ndarray, Optional.
        Monte Carlo trajectories produced by simulate_mc_trajectories,
        shape (n_traj, n_steps + 1, n_y). When provided, trajectories
        are drawn as semi-transparent lines through the funnel. Default
        is None.
    mc_color : str, Optional.
        Color for Monte Carlo trajectory lines. Default is 'green'.
    mc_alpha : float, Optional.
        Transparency for Monte Carlo lines. Default is 0.4.
    mc_linewidth : float, Optional.
        Line width for Monte Carlo lines. Default is 0.6.
    show : bool, Optional.
        For the plotly engine, display the figure in a notebook before
        returning it. Default is True. High-level callers that set a title
        and display the figure themselves pass show=False to avoid a double
        display. Has no effect on the matplotlib engine.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D axes object.


    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543

    [2] S. Dinh. Nonlinear Dynamic Analysis and Control of Chemical
        Processes Using Dynamic Operability Framework. PhD Dissertation,
        West Virginia University, 2023.

    '''
    # Extract the output-space AOS regions from the mapping results.
    AOS_regions = mapping_results['AOS_regions']
    n_steps = len(AOS_regions)

    if n_steps == 0:
        warnings.warn("No AOS regions to plot.")
        return None, None

    # The funnel requires 2D output polytopes (x=y1, y=y2, z=k). Probe
    # the first region to confirm dimensionality before building axes.
    first_verts = pc.extreme(AOS_regions[0].list_poly[0])
    if first_verts is None or first_verts.shape[1] != 2:
        warnings.warn("Funnel plot requires 2D output polytopes.")
        return None, None

    if engine == 'plotly':
        fig = _plotly_funnel(
            AOS_regions, DOS=DOS, dOI=dOI, labels=labels, alpha=alpha,
            colormap=colormap, view_elev=view_elev, view_azim=view_azim,
            orientation=orientation, mc_trajectories=mc_trajectories,
            mc_color=mc_color, mc_alpha=mc_alpha,
            mc_linewidth=mc_linewidth)
        # Display the plotly figure in a notebook, unless a caller (e.g.
        # dynamic_operability) will set a title and show it itself.
        if show:
            _show_if_notebook(fig)
        return fig, None
    if engine != 'matplotlib':
        raise ValueError("engine must be 'matplotlib' or 'plotly'.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    cmap_func = plt.get_cmap(colormap, max(n_steps, 2))
    cmap_cont = plt.get_cmap(colormap)

    # When per-step dOI is supplied, color the slices by operability index
    # (continuous, 0-100 %) instead of by time step.
    use_dOI = dOI is not None and len(dOI) == n_steps
    dOI_arr = np.asarray(dOI, dtype=float) if use_dOI else None

    # Draw the DOS box at the first and last time steps as a translucent
    # spatial reference, and connect its vertical edges with dashed lines.
    if orientation not in ('landscape', 'vertical'):
        raise ValueError("orientation must be 'landscape' or 'vertical'.")
    landscape = orientation == 'landscape'

    def _p3d(v, k_val):
        # Map a 2D output vertex and its time coordinate to the 3D axes:
        # landscape puts time on the (horizontal) y-axis with the second
        # output vertical, matching the Dinh & Lima figures.
        return (v[0], k_val, v[1]) if landscape else (v[0], v[1], k_val)

    if DOS is not None:
        dos = np.asarray(DOS, dtype=float)
        dos_verts_2d = np.array([
            [dos[0, 0], dos[1, 0]],
            [dos[0, 1], dos[1, 0]],
            [dos[0, 1], dos[1, 1]],
            [dos[0, 0], dos[1, 1]],
        ])
        for k_val in [1, n_steps]:
            dos_verts_3d = [_p3d(v, k_val) for v in dos_verts_2d]
            poly_dos = Poly3DCollection(
                [dos_verts_3d], alpha=0.08, facecolor=DS_COLOR,
                edgecolor='gray', linewidth=0.5, linestyle='--'
            )
            ax.add_collection3d(poly_dos)
        # Dashed edges along the time axis connecting the two DOS faces
        # for visual continuity.
        for v in dos_verts_2d:
            if landscape:
                ax.plot([v[0], v[0]], [1, n_steps], [v[1], v[1]],
                        color='gray', linewidth=0.4, linestyle='--',
                        alpha=0.3)
            else:
                ax.plot([v[0], v[0]], [v[1], v[1]], [1, n_steps],
                        color='gray', linewidth=0.4, linestyle='--',
                        alpha=0.3)

    # Plot each AOS_y polytope as a filled 3D polygon at its time-step
    # height, with rainbow coloring mapped to k.
    for i, region in enumerate(AOS_regions):
        poly = region.list_poly[0]
        verts = pc.extreme(poly)
        if verts is None or len(verts) < 3:
            continue

        # Sort vertices by polar angle around the centroid so the polygon
        # is drawn in order (no self-intersections).
        center = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - center[1],
                            verts[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = verts[order]

        k_val = i + 1
        if use_dOI:
            color = cmap_cont(np.clip(dOI_arr[i] / 100.0, 0.0, 1.0))
        else:
            color = cmap_func(i / max(n_steps - 1, 1))

        # Filled polygon face.
        verts_3d = [_p3d(v, k_val) for v in ordered]
        poly_3d = Poly3DCollection(
            [verts_3d], alpha=alpha, facecolor=color,
            edgecolor=color, linewidth=1.2
        )
        ax.add_collection3d(poly_3d)

        # Stronger boundary line around the polygon for visual clarity.
        boundary = np.vstack([ordered, ordered[0:1]])
        if landscape:
            ax.plot(boundary[:, 0], [k_val] * len(boundary),
                    boundary[:, 1],
                    color=color, linewidth=1.0, alpha=0.8)
        else:
            ax.plot(boundary[:, 0], boundary[:, 1],
                    [k_val] * len(boundary),
                    color=color, linewidth=1.0, alpha=0.8)

    # Overlay Monte Carlo trajectories as lines through the funnel.
    if mc_trajectories is not None:
        n_traj, n_pts, _ = mc_trajectories.shape
        for t in range(n_traj):
            ys = mc_trajectories[t]
            ks = np.arange(n_pts)
            if landscape:
                ax.plot(ys[:, 0], ks, ys[:, 1],
                        color=mc_color, alpha=mc_alpha,
                        linewidth=mc_linewidth)
            else:
                ax.plot(ys[:, 0], ys[:, 1], ks,
                        color=mc_color, alpha=mc_alpha,
                        linewidth=mc_linewidth)

    # Axis labels and 3D view angle. In landscape orientation the time
    # axis is horizontal (y-axis) and the second output is vertical.
    y1_label = labels[0] if (labels is not None and len(labels) >= 2) \
        else '$y_1$'
    y2_label = labels[1] if (labels is not None and len(labels) >= 2) \
        else '$y_2$'
    if landscape:
        ax.set_xlabel(y1_label, fontsize=11, labelpad=8)
        ax.set_ylabel('Time step $k$', fontsize=11, labelpad=8)
        ax.set_zlabel(y2_label, fontsize=11, labelpad=8)
    else:
        ax.set_xlabel(y1_label, fontsize=11, labelpad=8)
        ax.set_ylabel(y2_label, fontsize=11, labelpad=8)
        ax.set_zlabel('Time step $k$', fontsize=11, labelpad=8)
    ax.set_title('Dynamic Operability Funnel', fontsize=13)

    # Colorbar: dOI (%) when slices are colored by operability index,
    # otherwise the time-step index.
    if use_dOI:
        sm = plt.cm.ScalarMappable(
            cmap=colormap, norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('dOI (%)')
    else:
        sm = plt.cm.ScalarMappable(
            cmap=colormap, norm=plt.Normalize(vmin=1, vmax=n_steps))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Time step $k$')

    ax.view_init(elev=view_elev, azim=view_azim)
    fig.tight_layout()

    return fig, ax


def dynamic_operability(model,
                        x0,
                        AIS_bound,
                        DOS=None,
                        k_max=10,
                        AIS_resolution=3,
                        method='auto',
                        y0=None,
                        EDS_bound=None,
                        monte_carlo=0,
                        plot=True,
                        labels=None,
                        title=None,
                        orientation='landscape',
                        engine='matplotlib',
                        seed=None):
    '''
    One-call dynamic operability analysis -- the recommended high-level entry
    point.

    Builds the achievable-output funnel over k_max time steps, evaluates the
    Dynamic Operability Index (dOI) against the Desired Output Set, and plots
    the funnel with each time slice colored by its dOI -- replacing the
    manual dynamic_operability_mapping/nstep -> dOI_eval -> plot_dynamic_funnel
    sequence with a single call. The propagation method is selected
    automatically so the caller never has to choose:

      -- linear state-space projection when ``model`` is a set of matrices,
      -- nonlinear state-space projection for low-dimensional states,
      -- n-step simulation for high-dimensional states (e.g. spatially
         discretized PDE/reactor models) where the state polytope cannot be
         enumerated.

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    model : Callable or dict
        Either a step function step(x, u) -> (x_next, y) (or step(x, u, d)
        for the projection path with disturbances), or a dict of linear
        system matrices {'A':.., 'B':.., 'C':.., 'B_d':.. (optional)}. The
        dict also accepts 'u_ref', 'y_ref', and 'd_ref' nominal values so
        that deviation-form identified models can be analyzed with AIS,
        DOS, and EDS bounds given in absolute units. For
        high-dimensional-state models bake any disturbance into a
        two-argument closure step(x, u).
    x0 : np.ndarray
        Initial state vector.
    AIS_bound : np.ndarray
        Available Input Set bounds, shape (n_u, 2).
    DOS : np.ndarray, Optional.
        Desired Output Set bounds, shape (n_y, 2). When given, the dOI is
        computed and the funnel is colored by it. Default None.
    k_max : int, Optional.
        Number of discrete time steps. Default 10.
    AIS_resolution : int or array-like, Optional.
        AIS grid resolution. Default 3.
    method : {'auto', 'projection', 'nstep'}, Optional.
        Propagation method. 'auto' (default) picks projection for matrices or
        low-dimensional states (<= 8) and n-step otherwise.
    y0 : np.ndarray, Optional.
        Output at x0; prepended as the k=0 funnel slice for the n-step
        method. Default None.
    EDS_bound : np.ndarray, Optional.
        Expected Disturbance Set bounds for the projection path with an
        arity-3 step or a B_d matrix. Default None.
    monte_carlo : int, Optional.
        If > 0, overlay this many Monte Carlo input-sequence trajectories on
        the funnel. Default 0.
    plot : bool, Optional.
        Whether to draw the dOI-colored funnel. Default True.
    labels : list, Optional.
        Output-space axis labels. Default None.
    title : str, Optional.
        Plot title. Default None.
    orientation : {'landscape', 'vertical'}, Optional.
        Funnel orientation, as in plot_dynamic_funnel. 'landscape'
        (default) places time on a horizontal axis with the second output
        vertical, matching the Dinh & Lima figures.
    engine : {'matplotlib', 'plotly'}, Optional.
        Plotting engine. 'plotly' produces an interactive figure
        (pan/rotate/zoom) that stays interactive in Jupyter and in static
        HTML exports; 'matplotlib' (default) produces a static figure.
    seed : int, Optional.
        Seed for the Monte Carlo trajectories. Default None.

    Returns
    -------
    results : dict
        The funnel results dict (see dynamic_operability_mapping /
        dynamic_operability_nstep) augmented with 'dOI' (when DOS is given),
        'method' (the method used), and 'fig'/'ax' (when plotted).

    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543
    '''
    x0 = np.asarray(x0, dtype=float)

    if callable(model):
        if method == 'auto':
            # State-space projection enumerates the state polytope, which is
            # only tractable for low-dimensional states; everything larger
            # falls back to n-step simulation.
            method = 'projection' if x0.size <= 8 else 'nstep'
        if method == 'projection':
            results = dynamic_operability_mapping(
                step_model=model, x0=x0, AIS_bound=AIS_bound,
                AIS_resolution=AIS_resolution, k_max=k_max,
                EDS_bound=EDS_bound, plot=False)
            results['method'] = 'state-space projection (nonlinear)'
        elif method == 'nstep':
            results = dynamic_operability_nstep(
                model, x0, AIS_bound, k_max,
                AIS_resolution=AIS_resolution, y0=y0, plot=False)
            results['method'] = 'n-step simulation'
        else:
            raise ValueError(
                "method must be 'auto', 'projection', or 'nstep'.")
    elif isinstance(model, dict):
        A, B = model.get('A'), model.get('B')
        C, B_d = model.get('C'), model.get('B_d')
        if A is None or B is None or C is None:
            raise ValueError(
                "matrix model must provide at least 'A', 'B' and 'C'.")
        results = dynamic_operability_mapping(
            x0=x0, AIS_bound=AIS_bound, AIS_resolution=AIS_resolution,
            k_max=k_max, A=A, B=B, C=C, B_d=B_d,
            u_ref=model.get('u_ref'), y_ref=model.get('y_ref'),
            d_ref=model.get('d_ref'),
            EDS_bound=EDS_bound, plot=False)
        results['method'] = 'state-space projection (linear)'
    else:
        raise TypeError(
            "model must be a step callable or a dict of {A, B, C, B_d}.")

    # Dynamic Operability Index against the DOS.
    dOI = None
    if DOS is not None:
        dOI = dOI_eval(results, DOS, plot=False)
        results['dOI'] = dOI

    # Optional Monte Carlo trajectory overlay.
    mc = None
    if monte_carlo and monte_carlo > 0:
        mc = simulate_mc_trajectories(
            results, n_trajectories=int(monte_carlo), seed=seed)

    if plot:
        fig, ax = plot_dynamic_funnel(
            results, DOS=DOS, dOI=dOI, labels=labels, mc_trajectories=mc,
            orientation=orientation, engine=engine, show=False)
        if fig is not None and title is not None:
            if engine == 'plotly':
                fig.update_layout(title_text=title)
            else:
                ax.set_title(title, fontsize=13)
        if engine == 'plotly' and fig is not None:
            _show_if_notebook(fig)
        results['fig'], results['ax'] = fig, ax

    return results


def dynamic_operability_scenarios(step_factory,
                                  x0_factory,
                                  AIS_bound,
                                  scenarios,
                                  DOS=None,
                                  k_max=10,
                                  AIS_resolution=3,
                                  method='auto',
                                  y0_factory=None,
                                  plot=True,
                                  labels=None,
                                  colors=None,
                                  title=None,
                                  orientation='landscape',
                                  engine='matplotlib'):
    '''
    Dynamic operability across several disturbance scenarios (or input
    sequences), together with their disturbance-robust intersection.

    For each scenario the achievable-output funnel is built with
    dynamic_operability; the scenarios are then intersected step by step to
    give the robust funnel -- the outputs achievable regardless of which
    scenario is realized (Dinh & Lima 2023, Figure 9). When the output space
    is two-dimensional the result is plotted: each scenario's funnel outline
    in its own color with the robust intersection filled and stacked along the
    time axis. For higher output dimensions the plot is skipped and the
    numerical results are returned.

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    step_factory : Callable
        Maps a scenario value d to a two-argument step closure
        step(x, u) -> (x_next, y) with that disturbance baked in.
    x0_factory : Callable
        Maps a scenario value d to the initial state x0 for that scenario
        (e.g. the steady state at that disturbance).
    AIS_bound : np.ndarray
        Available Input Set bounds, shape (n_u, 2).
    scenarios : dict or list
        Either {label: d} or a list of scenario values d (labels are then
        generated automatically). Each d is passed to step_factory and
        x0_factory.
    DOS : np.ndarray, Optional.
        Desired Output Set bounds, shape (n_y, 2). When given, the dOI of the
        robust intersection is computed per step. Default None.
    k_max : int, Optional.
        Number of discrete time steps. Default 10.
    AIS_resolution : int or array-like, Optional.
        AIS grid resolution. Default 3.
    method : {'auto', 'projection', 'nstep'}, Optional.
        Propagation method passed to dynamic_operability. Default 'auto'.
    y0_factory : Callable, Optional.
        Maps a scenario value d to the output at x0 (prepended as the k=0
        slice). Default None.
    plot : bool, Optional.
        Whether to draw the scenario funnels + intersection (2D output only).
        Default True.
    labels : list, Optional.
        Output-space axis labels. Default None.
    colors : list, Optional.
        One color per scenario for the funnel outlines. Default None (uses
        the matplotlib color cycle).
    title : str, Optional.
        Plot title. Default None.
    orientation : {'landscape', 'vertical'}, Optional.
        Funnel orientation, as in plot_dynamic_funnel. 'landscape'
        (default) places time on a horizontal axis with the second output
        vertical, matching the Dinh & Lima figures.
    engine : {'matplotlib', 'plotly'}, Optional.
        Plotting engine. 'plotly' produces an interactive figure
        (pan/rotate/zoom) that stays interactive in Jupyter and in static
        HTML exports; 'matplotlib' (default) produces a static figure.

    Returns
    -------
    results : dict
        {'scenarios': {label: per-scenario results dict},
         'intersection': {'AOS_regions', 'volumes', 'dOI' (if DOS given)},
         'fig', 'ax' (when plotted)}.

    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543
    '''
    # Normalize the scenario specification to an ordered list of (label, d).
    if isinstance(scenarios, dict):
        items = list(scenarios.items())
    else:
        items = [(f'scenario {i + 1}', d) for i, d in enumerate(scenarios)]
    labels_list = [lab for lab, _ in items]

    # Build each scenario's funnel (no per-scenario plotting).
    per = {}
    for label, d in items:
        y0 = y0_factory(d) if y0_factory is not None else None
        per[label] = dynamic_operability(
            step_factory(d), x0_factory(d), AIS_bound, DOS=DOS, k_max=k_max,
            AIS_resolution=AIS_resolution, method=method, y0=y0, plot=False)

    n_k = min(len(per[lab]['AOS_regions']) for lab in labels_list)

    # Step-by-step intersection of all scenario AOS polytopes.
    def _as_region(obj):
        if obj is None or pc.is_empty(obj):
            return pc.Region([])
        return obj if isinstance(obj, pc.Region) else pc.Region([obj])

    inter_regions, inter_vol = [], []
    for k in range(n_k):
        cur = per[labels_list[0]]['AOS_regions'][k].list_poly[0]
        for lab in labels_list[1:]:
            nxt = per[lab]['AOS_regions'][k].list_poly[0]
            inter = pc.intersect(cur, nxt)
            if pc.is_empty(inter):
                cur = inter
                break
            cur = (inter.list_poly[0] if isinstance(inter, pc.Region)
                   else inter)
        reg = _as_region(cur)
        inter_regions.append(reg)
        inter_vol.append(float(reg.volume) if len(reg) > 0 else 0.0)

    intersection = {
        'AOS_regions': inter_regions,
        'volumes': np.asarray(inter_vol, dtype=float),
        'k_converged': None,
    }
    if DOS is not None:
        DOS_arr = np.asarray(DOS, dtype=float)
        intersection['dOI'] = np.asarray(
            [_dOI_at_step(r, DOS_arr) if len(r) > 0 else 0.0
             for r in inter_regions], dtype=float)

    results = {'scenarios': per, 'intersection': intersection}

    # Plot only for 2D output spaces.
    probe = pc.extreme(per[labels_list[0]]['AOS_regions'][0].list_poly[0])
    plottable = probe is not None and probe.shape[1] == 2
    if plot and not plottable:
        warnings.warn(
            "Scenario plot requires a 2D output space; returning data only.")
    if plot and plottable:
        if engine == 'plotly':
            fig = _plotly_scenarios(per, inter_regions, n_k, labels_list,
                                    labels=labels, colors=colors,
                                    title=title, orientation=orientation)
            _show_if_notebook(fig)
            results['fig'], results['ax'] = fig, None
        elif engine == 'matplotlib':
            fig, ax = _plot_scenarios(per, inter_regions, n_k, labels_list,
                                      labels=labels, colors=colors,
                                      title=title, orientation=orientation)
            results['fig'], results['ax'] = fig, ax
        else:
            raise ValueError("engine must be 'matplotlib' or 'plotly'.")

    return results


def _plot_scenarios(per, inter_regions, n_k, labels_list,
                    labels=None, colors=None, title=None,
                    orientation='landscape'):
    '''3D plot of per-scenario funnel outlines (one color each) with the
    robust intersection filled in blue, stacked along the time axis. The
    default landscape orientation places time on a horizontal axis with
    the second output vertical, as in the Dinh & Lima figures.'''
    landscape = orientation == 'landscape'

    def _ordered(poly):
        V = pc.extreme(poly)
        if V is None or len(V) < 3:
            return V
        c = V.mean(axis=0)
        ang = np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0])
        return V[np.argsort(ang)]

    if colors is None:
        cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
            'color', ['C0', 'C1', 'C2', 'C3'])
        colors = [cycle[i % len(cycle)] for i in range(len(labels_list))]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scenario funnel outlines.
    for j, lab in enumerate(labels_list):
        regions = per[lab]['AOS_regions']
        for k in range(min(n_k, len(regions))):
            V = _ordered(regions[k].list_poly[0])
            if V is None:
                continue
            B = np.vstack([V, V[0:1]])
            if landscape:
                ax.plot(B[:, 0], [k] * len(B), B[:, 1], color=colors[j],
                        linewidth=1.2, label=lab if k == 0 else None)
            else:
                ax.plot(B[:, 0], B[:, 1], [k] * len(B), color=colors[j],
                        linewidth=1.2, label=lab if k == 0 else None)

    # Robust intersection, filled.
    for k in range(n_k):
        reg = inter_regions[k]
        if len(reg) == 0:
            continue
        V = _ordered(reg.list_poly[0])
        if V is None or len(V) < 3:
            continue
        if landscape:
            face_verts = [[(x, k, y) for x, y in V]]
        else:
            face_verts = [[(x, y, k) for x, y in V]]
        face = Poly3DCollection(face_verts,
                                facecolor='blue', alpha=0.55,
                                edgecolor='b', linewidth=0.8)
        ax.add_collection3d(face)

    y1_label = labels[0] if (labels is not None and len(labels) >= 2) \
        else '$y_1$'
    y2_label = labels[1] if (labels is not None and len(labels) >= 2) \
        else '$y_2$'
    if landscape:
        ax.set_xlabel(y1_label, fontsize=11, labelpad=8)
        ax.set_ylabel('Time step $k$', fontsize=11, labelpad=8)
        ax.set_zlabel(y2_label, fontsize=11, labelpad=8)
    else:
        ax.set_xlabel(y1_label, fontsize=11, labelpad=8)
        ax.set_ylabel(y2_label, fontsize=11, labelpad=8)
        ax.set_zlabel('Time step $k$', fontsize=11, labelpad=8)
    ax.set_title(title or 'Dynamic operability: scenarios and robust '
                          'intersection', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=18, azim=-50)
    fig.tight_layout()
    return fig, ax


def _project_state_regions(mapping_results, dims):
    '''Project the stored state-space funnel polytopes (AOS_x) onto two
    selected state dimensions, returning a list of 2D pc.Region slices.'''
    if 'AOS_x' not in mapping_results:
        raise ValueError(
            "plot_state_funnel requires results from the state-space "
            "projection path (dynamic_operability_mapping, or "
            "dynamic_operability with matrices or a low-dimensional "
            "state); the n-step simulation method does not store state "
            "polytopes.")
    d0, d1 = int(dims[0]), int(dims[1])
    regions = []
    for P in mapping_results['AOS_x']:
        V = pc.extreme(P)
        if V is None or len(V) == 0:
            continue
        pts = V[:, [d0, d1]]
        regions.append(pc.Region([_hull_or_degenerate(pts)]))
    return regions


def plot_state_funnel(mapping_results,
                      dims=(0, 1),
                      labels=None,
                      colors=None,
                      title=None,
                      alpha=0.25,
                      colormap='rainbow',
                      view_elev=20,
                      view_azim=-60,
                      orientation='landscape',
                      engine='matplotlib'):
    '''
    Plot the dynamic funnel in the STATE space, as in Figures 4 and 5 of
    Dinh & Lima (Comput. Chem. Eng., 2026): the achievable state-space
    polytopes AOS_x(k) are projected onto two selected state dimensions
    and stacked along the time axis. Passing a dictionary of several
    mapping results overlays their state funnels as outlines (one color
    each), which reproduces the nominal-versus-updated funnel comparison
    used for the online update of the operability funnel at a perturbed
    initial state.

    Only available for results produced by the state-space projection
    path (dynamic_operability_mapping, or dynamic_operability with
    matrices or a low-dimensional state), which stores the state
    polytopes under the 'AOS_x' key; the n-step simulation method works
    in the output space and does not retain them.

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    mapping_results : dict
        Either a single results dictionary (from the projection path) or
        a dictionary mapping legend labels to results dictionaries, e.g.
        {'Nominal initial state': res_a, 'Perturbed initial state':
        res_b}, in which case the state funnels are overlaid as outlines.
    dims : tuple of int, Optional.
        The two (zero-based) state dimensions to project onto. Default
        is (0, 1).
    labels : list, Optional.
        Axis labels for the two state variables. Default is None
        ('State variable <dims[0]+1>', 'State variable <dims[1]+1>').
    colors : list, Optional.
        One color per overlaid funnel (multi-results input only).
        Default is None (matplotlib color cycle).
    title : str, Optional.
        Plot title. Default is None.
    alpha : float, Optional.
        Transparency for the funnel slices (single-results input only).
        Default is 0.25.
    colormap : str, Optional.
        Colormap for the time-step coloring (single-results input only).
        Default is 'rainbow'.
    view_elev : float, Optional.
        Elevation angle for the 3D view in degrees. Default is 20.
    view_azim : float, Optional.
        Azimuth angle for the 3D view in degrees. Default is -60.
    orientation : {'landscape', 'vertical'}, Optional.
        Funnel orientation, as in plot_dynamic_funnel. Default is
        'landscape'.
    engine : {'matplotlib', 'plotly'}, Optional.
        Plotting engine, as in plot_dynamic_funnel. Default is
        'matplotlib'.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly Figure
        The figure object.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None
        The 3D axes object (None for the plotly engine).

    References
    ----------
    [1] S. Dinh and F. V. Lima. Linear dynamic operability analysis with
        state-space projection for the online construction of achievable
        output funnels. Comput. Chem. Eng., 205:109428, 2026.
        https://doi.org/10.1016/j.compchemeng.2025.109428
    '''
    if labels is None:
        labels = [f'State variable {int(dims[0]) + 1}',
                  f'State variable {int(dims[1]) + 1}']

    # A single results dict carries 'AOS_regions'/'AOS_x' keys; a
    # label -> results mapping does not.
    multi = not ('AOS_regions' in mapping_results
                 or 'AOS_x' in mapping_results)
    if not multi:
        # Single funnel: reuse the output-funnel renderer on the
        # projected state regions (time-step coloring, no DOS/dOI).
        regions = _project_state_regions(mapping_results, dims)
        fig, ax = plot_dynamic_funnel(
            {'AOS_regions': regions}, labels=labels, alpha=alpha,
            colormap=colormap, view_elev=view_elev, view_azim=view_azim,
            orientation=orientation, engine=engine, show=False)
        final_title = title or 'Dynamic state funnel'
        if fig is not None:
            if engine == 'plotly':
                fig.update_layout(title_text=final_title)
                _show_if_notebook(fig)
            else:
                ax.set_title(final_title, fontsize=13)
        return fig, ax

    # Multiple funnels: overlay outlines per label, reusing the scenario
    # renderers with an empty intersection.
    per = {lab: {'AOS_regions': _project_state_regions(res, dims)}
           for lab, res in mapping_results.items()}
    labels_list = list(per.keys())
    n_k = min(len(per[lab]['AOS_regions']) for lab in labels_list)
    empty_inter = [pc.Region([]) for _ in range(n_k)]
    final_title = title or 'Dynamic state funnels'
    if engine == 'plotly':
        fig = _plotly_scenarios(per, empty_inter, n_k, labels_list,
                                labels=labels, colors=colors,
                                title=final_title,
                                orientation=orientation,
                                view_elev=view_elev, view_azim=view_azim)
        _show_if_notebook(fig)
        return fig, None
    if engine == 'matplotlib':
        return _plot_scenarios(per, empty_inter, n_k, labels_list,
                               labels=labels, colors=colors,
                               title=final_title, orientation=orientation)
    raise ValueError("engine must be 'matplotlib' or 'plotly'.")


def plot_funnel_comparison(results_dict,
                           labels=None,
                           colors=None,
                           title=None,
                           orientation='landscape',
                           view_elev=20,
                           view_azim=-60,
                           engine='matplotlib'):
    '''
    Overlay the output-space funnels of several dynamic operability results
    as outlines, one color per result. Useful for comparing mapping methods
    (linear projection, nonlinear projection, n-step simulation) or model
    fidelities on the same process and operability sets.

    The funnels are aligned index-wise: slice i of every result is drawn at
    time step i+1. For a fair comparison all results should therefore start
    at the same time step (e.g. build the n-step funnel without the y0
    initial slice when comparing against projection results).

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    results_dict : dict
        Mapping of legend labels to results dictionaries (as returned by
        dynamic_operability and the lower-level mapping functions), e.g.
        {'Linear': res_lin, 'Nonlinear projection': res_nl,
        'N-step (full model)': res_ns}.
    labels : list, Optional.
        Axis labels for the two output variables. Default is None.
    colors : list, Optional.
        One color per result. Default is None (matplotlib color cycle).
    title : str, Optional.
        Plot title. Default is None.
    orientation : {'landscape', 'vertical'}, Optional.
        Funnel orientation, as in plot_dynamic_funnel. Default is
        'landscape'.
    view_elev : float, Optional.
        Elevation angle for the 3D view in degrees. Default is 20.
    view_azim : float, Optional.
        Azimuth angle for the 3D view in degrees. Default is -60.
    engine : {'matplotlib', 'plotly'}, Optional.
        Plotting engine, as in plot_dynamic_funnel. Default is
        'matplotlib'.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly Figure
        The figure object.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None
        The 3D axes object (None for the plotly engine).
    '''
    per = {lab: {'AOS_regions': res['AOS_regions']}
           for lab, res in results_dict.items()}
    labels_list = list(per.keys())
    n_k = min(len(per[lab]['AOS_regions']) for lab in labels_list)
    empty_inter = [pc.Region([]) for _ in range(n_k)]
    final_title = title or 'Dynamic funnel comparison'
    if engine == 'plotly':
        fig = _plotly_scenarios(per, empty_inter, n_k, labels_list,
                                labels=labels, colors=colors,
                                title=final_title, orientation=orientation,
                                view_elev=view_elev, view_azim=view_azim)
        _show_if_notebook(fig)
        return fig, None
    if engine == 'matplotlib':
        return _plot_scenarios(per, empty_inter, n_k, labels_list,
                               labels=labels, colors=colors,
                               title=final_title, orientation=orientation)
    raise ValueError("engine must be 'matplotlib' or 'plotly'.")


def update_dynamic_funnel(mapping_results, x0_new, DOS=None):
    '''
    Online update of a linear dynamic operability funnel for a new initial
    state, via the hyperplane right-hand-side shift of Dinh & Lima
    (Comput. Chem. Eng., 2026, Eq. 18). For an LTI system the funnel from
    any initial state is the offline funnel translated by A^k (x0_new -
    x0_offline), so both the state polytopes and the output regions are
    updated by pure linear algebra, with no re-simulation: the offline
    construction is performed once, and this update runs in microseconds
    at every sampling instant during operation.

    Only available for results produced by the linear matrices path of
    dynamic_operability / dynamic_operability_mapping.

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    mapping_results : dict
        Results dictionary from the linear path (must carry '_matrices'
        and 'AOS_x').
    x0_new : np.ndarray
        The new initial (estimated) state vector, in the same deviation
        coordinates as the offline x0.
    DOS : np.ndarray, Optional.
        Desired Output Set bounds, shape (n_y, 2). When given, the dOI of
        the updated funnel is computed and attached. Default is None.

    Returns
    -------
    results : dict
        A new results dictionary with the translated state polytopes and
        output regions (volumes are translation-invariant and reused),
        compatible with all downstream plotting and evaluation functions.

    References
    ----------
    [1] S. Dinh and F. V. Lima. Linear dynamic operability analysis with
        state-space projection for the online construction of achievable
        output funnels. Comput. Chem. Eng., 205:109428, 2026.
        https://doi.org/10.1016/j.compchemeng.2025.109428
    '''
    mats = mapping_results.get('_matrices')
    if mats is None or 'AOS_x' not in mapping_results:
        raise ValueError(
            "update_dynamic_funnel requires results from the linear "
            "matrices path of dynamic_operability_mapping.")
    A_arr, B_arr, C_arr, _ = mats
    x0_old = np.asarray(mapping_results['_x0'], dtype=float)
    x0_new = np.asarray(x0_new, dtype=float)
    dx0 = x0_new - x0_old

    # Translate the state polytopes by A^k dx0 and the output regions by
    # C A^k dx0: {H x <= b} + t = {H x <= b + H t}. Volumes are unchanged.
    AOS_x_new = []
    AOS_regions_new = []
    Ak = np.eye(A_arr.shape[0])
    for k, P in enumerate(mapping_results['AOS_x']):
        t_x = Ak @ dx0
        AOS_x_new.append(pc.Polytope(P.A, P.b + P.A @ t_x))
        if k >= 1:
            t_y = C_arr @ t_x
            reg = mapping_results['AOS_regions'][k - 1]
            polys = [pc.Polytope(Q.A, Q.b + Q.A @ t_y)
                     for Q in reg.list_poly]
            AOS_regions_new.append(pc.Region(polys))
        Ak = A_arr @ Ak

    results = dict(mapping_results)
    results['AOS_x'] = AOS_x_new
    results['AOS_regions'] = AOS_regions_new
    results['_x0'] = x0_new
    results['method'] = 'state-space projection (linear, online update)'
    results.pop('fig', None)
    results.pop('ax', None)
    if DOS is not None:
        results['dOI'] = dOI_eval(results, DOS, plot=False)
    return results


def propagate_output_covariance(A, G, C, Sigma_w, k_max, Sigma_v=None):
    '''
    Propagate Gaussian disturbance covariance through linear dynamics and
    return the output covariance at each time step:

        Sigma_x(k+1) = A Sigma_x(k) A^T + G Sigma_w G^T,  Sigma_x(0) = 0
        Sigma_y(k)   = C Sigma_x(k) C^T (+ Sigma_v)

    Companion to gaussian_robust_funnel, following the uncertainty
    propagation of Dinh & Lima (Comput. Chem. Eng., 2026, Eqs. 3-6).

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    A : np.ndarray
        State transition matrix of the disturbance-carrying dynamics,
        shape (n, n).
    G : np.ndarray
        Disturbance input matrix, shape (n, n_d).
    C : np.ndarray
        Output matrix, shape (n_y, n).
    Sigma_w : np.ndarray
        Disturbance covariance, shape (n_d, n_d).
    k_max : int
        Number of time steps.
    Sigma_v : np.ndarray, Optional.
        Measurement noise covariance, shape (n_y, n_y), added to every
        Sigma_y(k). Default is None.

    Returns
    -------
    list of np.ndarray
        Output covariance matrices Sigma_y(k) for k = 1, ..., k_max.
    '''
    A = np.asarray(A, dtype=float)
    G = np.asarray(G, dtype=float)
    C = np.asarray(C, dtype=float)
    Sigma_w = np.atleast_2d(np.asarray(Sigma_w, dtype=float))
    Sx = np.zeros((A.shape[0], A.shape[0]))
    out = []
    for _ in range(k_max):
        Sx = A @ Sx @ A.T + G @ Sigma_w @ G.T
        Sy = C @ Sx @ C.T
        if Sigma_v is not None:
            Sy = Sy + np.atleast_2d(np.asarray(Sigma_v, dtype=float))
        out.append(Sy)
    return out


def gaussian_robust_funnel(mapping_results, Sigma_y, confidence=0.95,
                           DOS=None):
    '''
    Disturbance-robust achievable output funnel for Gaussian uncertainty,
    by hyperplane shrinkage: each slice of the (mean) funnel is shrunk so
    that the remaining outputs stay achievable for every uncertainty
    realization inside the chosen highest-density region. Following the
    set-intersection theorem of Dinh & Lima (Comput. Chem. Eng., 2026,
    Theorem 1), the intersection of all translations of a polytope
    {H y <= b} over an ellipsoidal uncertainty with covariance Sigma is

        {H y <= b_hat},   b_hat_i = b_i - l * sqrt(H_i Sigma H_i^T),
        l^2 = Inv_chi2(confidence; n_y).

    This applies the shrinkage in the output space using the
    disturbance-induced output covariance (see
    propagate_output_covariance); the original publication composes a
    state-space shrinkage followed by a measurement-noise output
    shrinkage, of which this is the single-stage output-space variant.
    An empty slice means no output can be guaranteed at that time step
    regardless of the controller (transient inoperability).

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    mapping_results : dict
        Results dictionary whose 'AOS_regions' hold the mean (zero-
        disturbance) funnel, e.g. from the linear matrices path.
    Sigma_y : np.ndarray or list of np.ndarray
        Output covariance, either a single (n_y, n_y) matrix applied to
        every step or one matrix per funnel slice (as returned by
        propagate_output_covariance).
    confidence : float, Optional.
        Probability content of the uncertainty highest-density region.
        Default is 0.95.
    DOS : np.ndarray, Optional.
        Desired Output Set bounds. When given, the dOI of the robust
        funnel is computed and attached. Default is None.

    Returns
    -------
    results : dict
        A results dictionary with the shrunken 'AOS_regions' (empty
        regions where nothing can be guaranteed), updated 'volumes', and
        'dOI' when DOS is given; compatible with the plotting functions.

    References
    ----------
    [1] S. Dinh and F. V. Lima. Linear dynamic operability analysis with
        state-space projection for the online construction of achievable
        output funnels. Comput. Chem. Eng., 205:109428, 2026.
        https://doi.org/10.1016/j.compchemeng.2025.109428
    '''
    from scipy.stats import chi2

    regions = mapping_results['AOS_regions']
    n_k = len(regions)
    if isinstance(Sigma_y, (list, tuple)):
        if len(Sigma_y) < n_k:
            raise ValueError(
                f"Sigma_y has {len(Sigma_y)} entries but the funnel has "
                f"{n_k} slices.")
        Sy_list = [np.atleast_2d(np.asarray(S, dtype=float))
                   for S in Sigma_y]
    else:
        Sy_list = [np.atleast_2d(np.asarray(Sigma_y, dtype=float))] * n_k

    probe = pc.extreme(regions[0].list_poly[0])
    n_y = probe.shape[1] if probe is not None else Sy_list[0].shape[0]
    l_val = np.sqrt(chi2.ppf(confidence, df=n_y))

    regions_new = []
    volumes = []
    for k, reg in enumerate(regions):
        Sy = Sy_list[k]
        polys = []
        for Q in reg.list_poly:
            shrink = l_val * np.sqrt(
                np.einsum('ij,jk,ik->i', Q.A, Sy, Q.A))
            Pn = pc.reduce(pc.Polytope(Q.A, Q.b - shrink))
            if not pc.is_empty(Pn) and Pn.volume > 0:
                polys.append(Pn)
        reg_new = pc.Region(polys)
        regions_new.append(reg_new)
        volumes.append(float(sum(p.volume for p in polys)) if polys
                       else 0.0)

    results = dict(mapping_results)
    results['AOS_regions'] = regions_new
    results['volumes'] = np.asarray(volumes, dtype=float)
    results['method'] = (mapping_results.get('method', '')
                         + ' + Gaussian robust shrinkage').strip()
    results.pop('fig', None)
    results.pop('ax', None)
    if DOS is not None:
        DOS_arr = np.asarray(DOS, dtype=float)
        results['dOI'] = np.asarray(
            [_dOI_at_step(r, DOS_arr) if len(r) > 0 else 0.0
             for r in regions_new], dtype=float)
    return results


def identify_lti_step_tests(step_model, x0, u_ref, du, n_steps=25):
    '''
    Identify a discrete-time LTI model from step tests on a nonlinear
    step model, in the form consumed by dynamic_operability's matrices
    interface. Each input is stepped by du[j] from the reference point;
    every output response is fit with a first-order model
    y_i(k) = K_ij (1 - exp(-k / tau_ij)), and each fitted channel
    contributes one state (zero-order-hold discretization), mirroring the
    construction of identified models such as the HYPER process model of
    Dinh & Lima (Comput. Chem. Eng., 2026).

    Author: Victor Alves -- Carnegie Mellon University

    Parameters
    ----------
    step_model : Callable
        Two-argument step model step(x, u) -> (x_next, y) of the rigorous
        process (one call advances one sampling period).
    x0 : np.ndarray
        Steady-state state vector at the reference point (the step tests
        start here).
    u_ref : np.ndarray
        Reference (nominal) input vector, shape (n_u,).
    du : np.ndarray or float
        Step size per input (scalar applies to all inputs).
    n_steps : int, Optional.
        Number of sampling periods recorded per step test. Should cover
        the settling time of the slowest channel. Default is 25.

    Returns
    -------
    model : dict
        {'A', 'B', 'C', 'u_ref', 'y_ref'} ready for
        dynamic_operability(model, ...). The fitted gains and time
        constants are attached under 'K' and 'tau' (shape (n_y, n_u)) for
        inspection.
    '''
    x0 = np.asarray(x0, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    n_u = u_ref.shape[0]
    du_arr = (np.full(n_u, float(du)) if np.isscalar(du)
              else np.asarray(du, dtype=float))

    # Reference output at the steady state.
    _, y_ref = step_model(x0.copy(), u_ref)
    y_ref = np.atleast_1d(np.asarray(y_ref, dtype=float))
    n_y = y_ref.shape[0]

    # One step test per input.
    responses = np.zeros((n_u, n_steps, n_y))
    for j in range(n_u):
        u_test = u_ref.copy()
        u_test[j] += du_arr[j]
        x = x0.copy()
        for k in range(n_steps):
            x, y = step_model(x, u_test)
            responses[j, k, :] = np.asarray(y, dtype=float) - y_ref

    # First-order fit per channel via grid search on tau.
    t_grid = np.arange(1, n_steps + 1, dtype=float)
    taus = np.linspace(0.3, max(3.0, n_steps / 1.5), 300)
    K = np.zeros((n_y, n_u))
    tau = np.zeros((n_y, n_u))
    for j in range(n_u):
        for i in range(n_y):
            yk = responses[j, :, i]
            K_end = yk[-1]
            errors = [np.sum((K_end * (1 - np.exp(-t_grid / t_c))
                              - yk) ** 2) for t_c in taus]
            tau[i, j] = taus[int(np.argmin(errors))]
            K[i, j] = K_end / du_arr[j]

    # One state per channel, ZOH at the sampling period of step_model.
    n_states = n_y * n_u
    A = np.zeros((n_states, n_states))
    B = np.zeros((n_states, n_u))
    C = np.zeros((n_y, n_states))
    s = 0
    for i in range(n_y):
        for j in range(n_u):
            A[s, s] = np.exp(-1.0 / tau[i, j])
            B[s, j] = 1 - np.exp(-1.0 / tau[i, j])
            C[i, s] = K[i, j]
            s += 1

    return {'A': A, 'B': B, 'C': C, 'u_ref': u_ref, 'y_ref': y_ref,
            'K': K, 'tau': tau}


def make_pyomo_step_model(build_func, n_x, n_u):
    '''
    Wrap a Pyomo model builder function into a step_model callable that
    is compatible with dynamic_operability_mapping. At each call the
    returned callable builds and solves the Pyomo NLP to advance the
    state x(k) to x(k+1) given input u(k).

    The builder function must return a ConcreteModel with:
      -- m.x_current[i] : Param, indexed 0..n_x-1, for current state values.
      -- m.u[j]         : fixed Var, indexed 0..n_u-1, for input values.
      -- m.x_next[i]    : Var, indexed 0..n_x-1, for the next state values.
      -- Constraints linking x_current and u to x_next (e.g. via orthogonal
         collocation on finite elements over a single discretized time
         step).

    Author: Victor Alves

    Parameters
    ----------
    build_func : Callable
        Function that takes no arguments and returns a
        pyomo.environ.ConcreteModel with the structure described above.
    n_x : int
        Number of state variables.
    n_u : int
        Number of input variables.

    Returns
    -------
    step_model : Callable
        step_model(x, u) -> (x_next, y), compatible with
        dynamic_operability_mapping. The output y equals x_next; supply
        your own C matrix in the linear path or post-process results if
        a reduced output mapping is needed.


    References
    ----------
    [1] S. Dinh and F. V. Lima. Dynamic operability analysis for process
        design and control of modular natural gas utilization systems.
        Ind. Eng. Chem. Res., 2023.
        https://doi.org/10.1021/acs.iecr.2c03543

    [2] S. Dinh. Nonlinear Dynamic Analysis and Control of Chemical
        Processes Using Dynamic Operability Framework. PhD Dissertation,
        West Virginia University, 2023.

    '''
    import pyomo.environ as pyo

    def step_model(x, u):

        # Build a fresh Pyomo model instance for this (x, u) evaluation.
        m = build_func()

        # Fix the current state and input values as model parameters.
        for i in range(n_x):
            m.x_current[i].set_value(float(x[i]))
        for j in range(n_u):
            m.u[j].fix(float(u[j]))

        # Solve the NLP to obtain the next state vector.
        solver = pyo.SolverFactory('ipopt')
        solver.solve(m, tee=False)

        # Extract the next-state values and return with identity output.
        x_next = np.array([pyo.value(m.x_next[i]) for i in range(n_x)])
        return x_next, x_next

    return step_model


# --------------------------------------------------------------------------- #
# Private helpers - not part of the public API.
# --------------------------------------------------------------------------- #


def _propagate_state_nonlinear(step_model, state_vertices, input_vertices,
                                d_vec=None):
    '''
    Propagate a set of state vertices one time step forward by evaluating
    the nonlinear step model over all (state vertex, input vertex) pairs.
    Returns both the array of candidate next-state points (for state-space
    AOS reconstruction) and the array of corresponding output points.

    Parameters
    ----------
    step_model : Callable
        (x, u) -> (x_next, y) or (x, u, d) -> (x_next, y).
    state_vertices : np.ndarray
        Vertices of the current state polytope, shape (n_sv, n_x).
    input_vertices : np.ndarray
        Vertices or grid points of the AIS, shape (n_iv, n_u).
    d_vec : np.ndarray, Optional.
        A single fixed disturbance vector, shape (n_d,). The caller
        loops over EDS vertices to cover worst-case scenarios. Default
        is None (no disturbance argument passed to step_model).

    Returns
    -------
    next_pts : np.ndarray
        All candidate next-state points, shape (n_sv * n_iv, n_x).
    out_pts : np.ndarray
        Corresponding output points, shape (n_sv * n_iv, n_y).

    '''
    next_pts = []
    out_pts = []

    # Evaluate the step model for every (state vertex, input vertex)
    # pair, with the disturbance held fixed when supplied.
    if d_vec is None:
        for sv in state_vertices:
            for iv in input_vertices:
                x_next, y = step_model(np.asarray(sv), np.asarray(iv))
                next_pts.append(np.asarray(x_next, dtype=float))
                out_pts.append(np.atleast_1d(np.asarray(y, dtype=float)))
    else:
        d_arr = np.asarray(d_vec, dtype=float)
        for sv in state_vertices:
            for iv in input_vertices:
                x_next, y = step_model(np.asarray(sv), np.asarray(iv),
                                        d_arr)
                next_pts.append(np.asarray(x_next, dtype=float))
                out_pts.append(np.atleast_1d(np.asarray(y, dtype=float)))

    return np.array(next_pts), np.array(out_pts)


def _propagate_state_linear(A, B, B_d, state_polytope, AIS_polytope,
                             EDS_polytope=None):
    '''
    Propagate a state polytope one time step forward for a discrete-time
    linear system using polytope affine transforms and Minkowski sums:

        AOS_x(k+1) = A @ AOS_x(k) (+) B @ AIS

    If disturbances are present, an additional term is Minkowski-summed:

        AOS_x(k+1) = A @ AOS_x(k) (+) B @ AIS (+) B_d @ EDS

    Parameters
    ----------
    A : np.ndarray
        State transition matrix, shape (n_x, n_x).
    B : np.ndarray
        Input matrix, shape (n_x, n_u).
    B_d : np.ndarray or None
        Disturbance input matrix, shape (n_x, n_d). Required when
        EDS_polytope is supplied.
    state_polytope : pc.Polytope
        Current state polytope AOS_x(k).
    AIS_polytope : pc.Polytope
        Polytope representation of the Available Input Set.
    EDS_polytope : pc.Polytope or None, Optional.
        Polytope for the disturbance (typically a single-point degenerate
        polytope for one EDS vertex when the caller is handling worst-
        case intersection). Default is None.

    Returns
    -------
    pc.Polytope
        The propagated state polytope AOS_x(k+1).

    '''
    # Affine transform: A @ AOS_x(k).
    A_state = _affine_transform_polytope(A, state_polytope)

    # Affine transform: B @ AIS.
    B_input = _affine_transform_polytope(B, AIS_polytope)

    # Minkowski sum: A @ AOS_x(k) (+) B @ AIS.
    next_poly = _minkowski_sum_vrep(A_state, B_input)

    # Add the disturbance contribution B_d @ EDS via a second Minkowski
    # sum when EDS_polytope is supplied.
    if EDS_polytope is not None and B_d is not None:
        Bd_dist = _affine_transform_polytope(B_d, EDS_polytope)
        next_poly = _minkowski_sum_vrep(next_poly, Bd_dist)

    return next_poly


def _affine_transform_polytope(M, polytope_in):
    '''
    Apply the affine transformation y = M @ x to a polytope via vertex
    enumeration. The transformed vertex cloud is converted back to a
    polytope using convex hull (qhull), or to a degenerate bounding box
    when fewer vertices are present than required for a full-dimensional
    polytope.

    Parameters
    ----------
    M : np.ndarray
        Transformation matrix, shape (n_out, n_in).
    polytope_in : pc.Polytope
        Input polytope in n_in dimensions.

    Returns
    -------
    pc.Polytope
        Transformed polytope in n_out dimensions.

    '''
    # Extract the vertex representation of the input polytope.
    verts = pc.extreme(polytope_in)
    if verts is None or len(verts) == 0:
        raise ValueError(
            "Cannot extract vertices from the input polytope."
        )

    # Apply the linear transformation to each vertex row.
    transformed = (M @ verts.T).T

    # Fall back to a degenerate bounding box when qhull would need more
    # points than we have for a full-dimensional polytope.
    return _hull_or_degenerate(transformed)


def _minkowski_sum_vrep(poly_a, poly_b):
    '''
    Compute the Minkowski sum of two polytopes via vertex enumeration:

        P_a (+) P_b = ConvexHull{ a + b | a in V(P_a), b in V(P_b) }

    Parameters
    ----------
    poly_a : pc.Polytope
        First input polytope.
    poly_b : pc.Polytope
        Second input polytope.

    Returns
    -------
    pc.Polytope
        The Minkowski sum polytope.

    '''
    # Enumerate vertices of both operand polytopes.
    va = pc.extreme(poly_a)
    vb = pc.extreme(poly_b)
    if va is None or vb is None:
        raise ValueError("Cannot extract vertices for Minkowski sum.")

    # Form all pairwise sums and take the convex hull.
    pts = []
    for a in va:
        for b in vb:
            pts.append(a + b)
    pts = np.array(pts)

    return _hull_or_degenerate(pts)


def _hull_or_degenerate(points):
    '''
    Return pc.qhull(points) when the point cloud spans a full-dimensional
    polytope, or a small padded bounding box via _point_or_degenerate_polytope
    when it does not (e.g. single point or collinear set).

    Parameters
    ----------
    points : np.ndarray
        Array of points, shape (n_pts, n_dim).

    Returns
    -------
    pc.Polytope

    '''
    points = np.atleast_2d(points)
    # qhull needs at least n_dim + 1 points in general position.
    if points.shape[0] < points.shape[1] + 1:
        return _point_or_degenerate_polytope(points)
    return pc.qhull(points)


def _point_or_degenerate_polytope(points, eps=1e-6):
    '''
    Create a small full-dimensional polytope enclosing a potentially
    degenerate set of points (e.g. a single point or collinear set). A
    tiny axis-aligned bounding box of half-width eps is added around
    the point cloud so the resulting polytope is non-degenerate and
    admits a proper H-representation.

    Parameters
    ----------
    points : np.ndarray
        Array of points, shape (n_pts, n_dim).
    eps : float, Optional.
        Half-width of the padding added around each dimension. Default
        is 1e-6.

    Returns
    -------
    pc.Polytope
        A (possibly tiny) full-dimensional polytope enclosing all points.

    '''
    # Pad the bounding box of the point cloud by eps on all sides.
    points = np.atleast_2d(points)
    lo = points.min(axis=0) - eps
    hi = points.max(axis=0) + eps
    bounds = np.column_stack([lo, hi])
    return pc.box2poly(bounds)


def _dOI_at_step(AOS_region, DOS):
    '''
    Compute the Dynamic Operability Index (dOI) at a single time step
    by delegating to the steady-state OI_eval routine.

    Parameters
    ----------
    AOS_region : pc.Region
        The achievable output set region at this time step.
    DOS : np.ndarray
        Bounds on the Desired Output Set, shape (n_y, 2).

    Returns
    -------
    float
        Operability index value in the range [0, 100].

    '''
    # OI_eval expects AS as [pc.Region, None]; reuse the steady-state code.
    return OI_eval([AOS_region, None], DOS, plot=False)


def _plot_AOS_evolution(AOS_regions, labels=None):
    '''
    Plot the 2D output-space AOS polygon at each time step overlaid on
    a single axes, with a rainbow color map indicating time progression.

    Parameters
    ----------
    AOS_regions : list of pc.Region
        Output-space AOS regions at each time step.
    labels : list, Optional.
        Axis labels, e.g. ['$y_1$', '$y_2$']. Default is None.

    Returns
    -------
    None

    '''
    n_steps = len(AOS_regions)
    if n_steps == 0:
        return

    # Determine output dimensionality from the first region.
    first_verts = pc.extreme(AOS_regions[0].list_poly[0])
    if first_verts is None:
        return
    n_y = first_verts.shape[1] if first_verts.ndim == 2 else 1

    # AOS evolution plot is only generated for 2D output spaces.
    if n_y != 2:
        return

    fig, ax = plt.subplots(1, 1)
    cmap_func = plt.get_cmap(cmap, max(n_steps, 2))

    # Plot each AOS polygon with a time-indexed color and increasing
    # opacity from early to late steps.
    for i, region in enumerate(AOS_regions):
        color = cmap_func(i / max(n_steps - 1, 1))
        alpha = 0.15 + 0.6 * (i / max(n_steps - 1, 1))
        lbl = f'k={i+1}' if i == 0 or i == n_steps - 1 else None
        _plot_polytope_2d(ax, region.list_poly[0], facecolor=color,
                          edgecolor='k', alpha=alpha, label=lbl,
                          linewidth=0.5)

    ax.set_title('Dynamic AOS Evolution')
    if labels is not None and len(labels) >= 2:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    else:
        ax.set_xlabel('$y_1$')
        ax.set_ylabel('$y_2$')
    ax.legend(loc='best', fontsize='small')

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_steps)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time step k')
    fig.tight_layout()


def _plot_volume_evolution(volumes):
    '''
    Plot the evolution of the AOS_y Lebesgue volume (area in 2D) as a
    function of the time step k.

    Parameters
    ----------
    volumes : np.ndarray
        AOS_y volume per step.

    Returns
    -------
    None

    '''
    if len(volumes) == 0:
        return
    fig, ax = plt.subplots(1, 1)
    steps = np.arange(1, len(volumes) + 1)
    ax.plot(steps, volumes, 's-', color=AS_COLOR, linewidth=1.5,
            markersize=5, markeredgecolor='k')
    ax.set_xlabel('Time step k')
    ax.set_ylabel('AOS Volume (Lebesgue measure)')
    ax.set_title('AOS Volume Evolution')
    fig.tight_layout()


def _plot_dOI_convergence(dOI_values):
    '''
    Plot the Dynamic Operability Index convergence trajectory: dOI vs
    time step k, with a reference horizontal line at dOI = 100%.

    Parameters
    ----------
    dOI_values : np.ndarray
        Dynamic Operability Index at each time step, values in [0, 100].

    Returns
    -------
    None

    '''
    fig, ax = plt.subplots(1, 1)
    steps = np.arange(1, len(dOI_values) + 1)
    ax.plot(steps, dOI_values, 'o-', color=INTERSECT_COLOR,
            linewidth=1.5, markersize=5)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8,
               label='dOI = 100%')
    ax.set_xlabel('Time step k')
    ax.set_ylabel('Dynamic Operability Index (%)')
    ax.set_title('dOI Convergence')
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    fig.tight_layout()


def _plot_polytope_2d(ax, polytope_obj, facecolor='b', edgecolor='k',
                       alpha=0.3, label=None, linewidth=1.0):
    '''
    Plot a 2D convex polytope on an existing matplotlib axes by
    extracting its vertices and drawing a closed filled polygon.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the polygon.
    polytope_obj : pc.Polytope
        A 2D polytope whose vertices define the polygon.
    facecolor : str, Optional.
        Fill color of the polygon. Default is 'b'.
    edgecolor : str, Optional.
        Edge (boundary) color. Default is 'k'.
    alpha : float, Optional.
        Transparency of the polygon fill (0 transparent, 1 opaque).
        Default is 0.3.
    label : str, Optional.
        Legend label for the polygon patch. Default is None.
    linewidth : float, Optional.
        Width of the polygon boundary lines. Default is 1.0.

    Returns
    -------
    None

    '''
    # Extract vertices; skip degenerate or under-defined polytopes.
    verts = pc.extreme(polytope_obj)
    if verts is None or len(verts) < 3:
        return

    # Sort vertices by polar angle around the centroid for a proper
    # (non-self-intersecting) polygon.
    center = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - center[1],
                        verts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered_verts = verts[order]

    # Add the closed polygon patch and update axis limits.
    polygon = plt.Polygon(ordered_verts, closed=True, facecolor=facecolor,
                           edgecolor=edgecolor, alpha=alpha, label=label,
                           linewidth=linewidth)
    ax.add_patch(polygon)
    ax.autoscale_view()


# --------------------------------------------------------------------------- #
# Private helpers: optional-solver and Pyomo/OMLT support.
#
# Pounce (the default NLP solver) is a core dependency imported at module
# level. cyipopt and pyomo are optional and imported lazily here so that
# importing opyrability never requires them.
# --------------------------------------------------------------------------- #
def _is_pyomo_model(model) -> bool:
    '''
    Detect whether the supplied process model is a Pyomo/OMLT builder
    instead of a plain Python callable.

    The convention (contributed by Heitor F., github @hfsf, PR #33,
    adapted) is a builder function model(m, u_vars, y_vars) that adds the
    model constraints to a Pyomo ConcreteModel, flagged with a
    'build_pyomo_constraints' attribute (or 'is_omlt' for OMLT objects).

    Parameters
    ----------
    model : Callable or object
        The process model passed by the user: either a plain Python/JAX
        callable y = model(u), or a Pyomo/OMLT builder carrying the
        'build_pyomo_constraints' or 'is_omlt' attribute.

    Returns
    -------
    bool
        True if model is a Pyomo/OMLT builder, False if it is a plain
        callable.

    Author: Victor Alves -- Carnegie Mellon University
    '''
    return (hasattr(model, 'build_pyomo_constraints')
            or hasattr(model, 'is_omlt'))


def _pyomo_solver_factory(pyomo_solver: str, print_level: int = 5):
    '''
    Lazily import pyomo and build the requested NLP solver for the
    Pyomo/OMLT mapping paths.

    Supports 'ipopt' and 'pounce' (the pure-Rust Ipopt reimplementation
    with the bundled FERAL linear solver), the latter registered through
    the pyomo_pounce package.

    Parameters
    ----------
    pyomo_solver : str
        Name of the Pyomo solver to build (e.g. 'ipopt' or 'pounce').
    print_level : int, optional
        IPOPT print_level forwarded to the solver options when supported.
        The default is 5.

    Returns
    -------
    pyo : module
        The imported pyomo.environ module.
    solver : pyomo solver object
        The constructed Pyomo solver returned by SolverFactory.

    Raises
    ------
    ImportError
        If pyomo (or, when pyomo_solver='pounce', pyomo_pounce) is not
        installed.

    Author: Victor Alves -- Carnegie Mellon University
    '''
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError(
            "Pyomo/OMLT models require the pyomo package. Install it "
            "with: pip install pyomo") from exc
    if pyomo_solver == 'pounce':
        try:
            import pyomo_pounce  # noqa: F401 -- registers 'pounce'
        except ImportError as exc:
            raise ImportError(
                "pyomo_solver='pounce' requires the Pounce solver and "
                "its Pyomo plugin. Install them with: pip install "
                "pounce-solver pyomo-pounce") from exc
    solver = pyo.SolverFactory(pyomo_solver)
    try:
        solver.options['print_level'] = print_level
    except Exception:
        pass
    return pyo, solver


def _import_cyipopt_minimize():
    '''
    Lazily import cyipopt's scipy-style minimize interface.

    cyipopt is an optional dependency (it ships compiled IPOPT binaries
    that are usually installed through conda), so it is only imported when
    the user selects method='ipopt'. Pounce, the default solver, is a core
    dependency imported at module level instead.

    Returns
    -------
    minimize_ipopt : Callable
        cyipopt's scipy-style minimize_ipopt entry point.

    Raises
    ------
    ImportError
        If cyipopt (>=1.7.0) is not installed.

    Author: Victor Alves -- Carnegie Mellon University
    '''
    try:
        from cyipopt import minimize_ipopt
    except ImportError as exc:
        raise ImportError(
            "method='ipopt' requires cyipopt (>=1.7.0). Install it with: "
            "conda install -c conda-forge cyipopt") from exc
    return minimize_ipopt

