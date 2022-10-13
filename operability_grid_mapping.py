import numpy as np
from itertools import permutations as perms
from typing import Callable,Union

#%% Functions
def AIS2AOS_map(model: Callable[...,Union[float,np.ndarray]],
     AIS_bound: np.ndarray,
     AIS_resolution: np.ndarray)-> Union[np.ndarray,np.ndarray]:
    '''
    Forward mapping for Process Operability calculations (From AIS to AOS). 
    From a Available Input Set (AIS) bounds and discretization resolution both
    defined by the user, 
    this function calculates the corresponding discretized 
    Available Output Set (AIS).
    
    This function is part of Python-based Process Operability package.

    Control, Optimization and Design for Energy and Sustainability 
    (CODES) Group - West Virginia University - 2022

    Author: San Dinh

    Parameters
    ----------
    model : Callable[...,Union[float,np.ndarray]]
        Process model that calculates the relationship between inputs (AIS)
        and Outputs (AOS).
    AIS_bound : np.ndarray
        Lower and upper bounds for the Available Input Set (AIS).
    AIS_resolution : np.ndarray
        Resolution for the Available Input Set (AIS). This will be used to
        discretized the AIS.

    Returns
    -------
    AIS : np.ndarray
        Discretized Available Input Set (AIS).
    AOS : TYPE
        Discretized Available Output Set (AIS).

    '''
    
    
    # Mapping and indexing
    numInput = np.prod(AISresolution)
    nInput = AIS_bound.shape[0]
    Input_u = []
    
    # Create discretized AIS based on bounds and resolution information.
    for i in range(nInput):
        Input_u.append(list(np.linspace(AIS_bound[i,0],
                                        AIS_bound[i,1],
                                        AISresolution[i])))
    # Create slack variables for preallocation purposes.    
    u_slack = AIS_bound[:,0]
    y_slack = model(u_slack)
    nOutput = y_slack.shape[0]
    AIS = np.zeros(AISresolution + [nInput])
    AOS = np.zeros(AISresolution + [nOutput])
    
    # 
    for i in range(numInput):
        inputID = [0]*nInput
        inputID[0] = int(np.mod(i, AISresolution[0]))
        AIS_val = [Input_u[0][inputID[0]]]
        
        for j in range(1,nInput):
            inputID[j] = int(np.mod( np.floor(i/np.prod(AISresolution[0:j])),
                                    AISresolution[j]))
            AIS_val.append(Input_u[j][inputID[j]])
        
        AIS[tuple(inputID)] = AIS_val
        AOS[tuple(inputID)] = model(AIS_val)
    return AIS, AOS

def points2simplices(AIS: np.ndarray, AOS: np.ndarray) -> Union[np.ndarray,
                                                                np.ndarray]:
    '''
    Generation of connected simplices (k+1 convex hull of k+1 vertices) 
    based on the AIS/AOS points.
    
    
    This function is part of Python-based Process Operability package.
    
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
        A_full = np.zeros((nOutputVar+1,nOutputVar))
        A_full[0,ineq_choice[0]] = -1
        for i in range(0,nOutputVar-1):
            A_full[i+1, ineq_choice[i]] = 1
            A_full[i+1, ineq_choice[i+1]] = -1
        A_full[-1,ineq_choice[-1]] = 1
        b_full = np.zeros((nOutputVar+1,1))
        b_full[-1] = 1
        
        V_simplex = np.zeros((nOutputVar,nOutputVar+1))
        
        
        for i in range(nOutputVar+1):
            A = np.delete(A_full, i, axis = 0)
            b = np.delete(b_full, i, axis = 0)
            V_sol = np.linalg.solve(A, b) 
            V_simplex[:,[i]] = V_sol.astype(int)
        
        simplex_vertices.append(list(V_simplex.T))
        
    
    AOS_simplices = list()
    AIS_simplices = list()
    for i in range(numInput):
        inputID[0] = int(np.mod(i, resolution[0]))
        
        
        for j in range(1,nInput):
            inputID[j] = int(np.mod( np.floor(i/np.prod(resolution[0:j])), 
                                    resolution[j]))
        
        
        if np.prod(inputID) != 0:
            inputID = [x-1 for x in inputID]
            #print(inputID)
            for k in range(n_simplex):
                #ID = [inputID]*(nInput + 1) + simplex_vertices[k]
                ID = [a + b for a,b in zip([inputID]*(nInput + 1), 
                                           simplex_vertices[k])] 
                
                V_AOS_id = np.zeros((nOutputVar,nInput + 1))
                V_AIS_id = np.zeros((nInputVar,nInput + 1))
                for m in range(nInput + 1):
                    ID_cell = tuple([int(x) for x in ID[m]])
                    V_AOS_id[:,m] = AOS[ID_cell]
                    V_AIS_id[:,m] = AIS[ID_cell]
                
                AOS_simplices.append(V_AOS_id)
                AIS_simplices.append(V_AIS_id)
    return AIS_simplices, AOS_simplices

def points2polyhedra(AIS: np.ndarray, AOS: np.ndarray) -> Union[np.ndarray,
                                                                np.ndarray]:
    '''
    Generation of connected polyhedra based on the AIS/AOS points.
    
    This function is part of Python-based Process Operability package.
    
    Author: San Dinh

    Parameters
    ----------
    AIS : np.ndarray
        Array containing the points that constitutes the AIS.
    AOS : np.ndarray
        Array containing the points that constitutes the AOS.

    Returns
    -------
    AIS_polyhedra : np.ndarray
        List of input (AIS) vertices for each polyhedra generated, in the form 
        of an array.
    AOS_polyhedra : np.ndarray
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
        #print(n)
        cube_vertices.append( [int(x) for x in 
                               np.binary_repr(n, width=nInput)])
    
    AOS_polyhedra = list()
    AIS_polyhedra = list()
    for i in range(numInput):
        inputID[0] = int(np.mod(i, resolution[0]))
        
        
        for j in range(1,nInput):
            inputID[j] = int(np.mod( np.floor(i/np.prod(resolution[0:j])),
                                    resolution[j]))
        
        
        if np.prod(inputID) != 0:
            inputID = [x-1 for x in inputID]
            ID = np.array([inputID]*(2**(nInput)))+ np.array(cube_vertices)
            
            
            V_AOS_id = np.zeros((nOutputVar,2**(nInput)))
            V_AIS_id = np.zeros((nInputVar,2**(nInput)))
            for k in range(2**(nInput)):
                ID_cell = tuple([int(x) for x in ID[k]])
                V_AOS_id[:,k] = AOS[ID_cell]
                V_AIS_id[:,k] = AIS[ID_cell]
                
            AOS_polyhedra.append(V_AOS_id)
            AIS_polyhedra.append(V_AIS_id)
    return AIS_polyhedra, AOS_polyhedra
#%% Testing only, remove in the main package
# def shower(u):
#     # Shower Problem
#     d = [0,0]
#     y0 = u[0]+u[1] 
#     if y0 != 0:
#         y1 = ( u[0]*(60 + d[0]) + u[1]*(120 + d[1]) )/( u[0] + u[1] )
#     else:
#         y1 = (60+120)/2
    
#     return np.array([y0, y1])


def shower(u):
    
    d = np.zeros(2)
    y = np.zeros(2)
    y[0]=u[0]+u[1]
    if y[0]!=0:
        y[1]=(u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    else:
        y[1]=(60+120)/2
        
    return y

#%% Test run
AIS_bound = np.array([[0, 10],
                    [0, 10]])

AISresolution = [10, 10]
AIS, AOS =  AIS2AOS_map(shower, 
                        AIS_bound, 
                        AISresolution)
AIS_simp, AOS_simp = points2simplices(AIS=AIS,AOS=AOS)
AIS_poly, AOS_poly = points2polyhedra(AIS=AIS,AOS=AOS)