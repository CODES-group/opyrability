import polytope as pc

import numpy as np
from operability_grid_mapping import AIS2AOS_map, points2simplices, points2polyhedra
from polytope.polytope import region_diff
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from polytope.polytope import _get_patch


from PolyhedraVolAprox import VolumeApprox_fast as Dinh_volume


def multimodel_rep(AIS_bound: np.ndarray, 
               resolution: np.ndarray, model):
    
    
    AIS, AOS =  AIS2AOS_map(model, AIS_bound, resolution)
    
    ## TODO: Add option to trace simplices or polyhedras. 
    #AIS_poly, AOS_poly = points2simplices(AIS,AOS)
    
    AIS_poly, AOS_poly = points2polyhedra(AIS,AOS)
    
    
    
    Polytope = list()
  
    for i in range(len(AOS_poly)):
        Vertices = AOS_poly[i].T
        
        Polytope.append(pc.qhull(Vertices))
        
        
    overlapped_region = pc.Region(Polytope[0:])
    # Create a bounding box of the region above:
    min_coord =  overlapped_region.bounding_box[0]
    max_coord =  overlapped_region.bounding_box[1]
    box_coord =  np.hstack([min_coord, max_coord])
    bound_box =  pc.box2poly(box_coord)

    # Remove overlapping
    RemPoly = [bound_box]
    # for i in range(len(Polytope)-1):
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
    # RemU =  RemU.diff(overlapped_region[-1])
    finalpolytope = region_diff(bound_box, RemU, abs_tol = 1e-11, intersect_tol=1e-11)
    #finalpolytope =  pc.reduce(finalpolytope)
    # overlapped_region.plot()
    
    # RemU.plot()
    
    # finalpolytope.plot()
    return finalpolytope


def OI_calc(AS: pc.Region,
            DS: np.ndarray, perspective= 'outputs'):
    
    DS_region =  pc.box2poly(DS)
    # DS_region =  pc.Region([DS_region])
    
    # intersection =  pc.intersect(AS, DS_region, abs_tol=1e-10)
    
    # intersection =  pc.intersect(DS_region, AS)
    AS = pc.reduce(AS)
    DS_region = pc.reduce(DS_region)
    
    
    intersection =  pc.intersect(AS, DS_region)
    # intersection =  pc.reduce(intersection)
    
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
    
    
    
        OI_str =  'Operability Index = ' + str(round(OI,2)) + str('%')
    
        extra = mpatches.Rectangle((0, 0), 1, 1, fc="w", 
                                   fill=False, 
                                   edgecolor='none', 
                                   linewidth=0, label= OI_str)
    
        if perspective == 'outputs':
            ax.set_title('OI Evaluation - Outputs" perspective')
        else:
            ax.set_title('OI Evaluation - Inputs" perspective')
    
        ax.legend(handles=[DS_patch,AS_patch, INTERSECT_patch, extra])
    
    
        plt.show()
        
    else:
        print('Plotting not supported. Dimension different than 2.')
        
        
    return OI
