import numpy as np
import polytope as pc
import scipy.optimize as opt
from scipy.special import gamma
#%% Functions
def walkCDHR(A,b,point,nStep = 0):
    nDim = A.shape[1]
    
    if nStep == 0:
        nStep = nDim*10

    direction = np.zeros(nDim)
    direction_ID = np.random.randint(nDim)
    direction[direction_ID] = 1
    
    t = b - A@point
    A_col = A[:,direction_ID]
    negative_ID = np.where(A_col < 0)
    positive_ID = np.where(A_col > 0)
    
    step_min = min(-t[negative_ID]/A_col[negative_ID])
    step_max = min(t[positive_ID]/A_col[positive_ID])
    stepsize = np.random.uniform(low = -step_min, high = step_max)
    newpoint = point + stepsize*direction
    
    for stepcount in range(nStep-1):
        
        direction = np.zeros(nDim)
        direction_ID = np.random.randint(nDim)
        direction[direction_ID] = 1
        
        t = t - stepsize*A_col

        A_col = A[:,direction_ID]
        
        negative_ID = np.where(A_col < 0)
        positive_ID = np.where(A_col > 0)
        
        step_min = min(-t[negative_ID]/A_col[negative_ID])
        step_max = min(t[positive_ID]/A_col[positive_ID])
        stepsize = np.random.uniform(low = -step_min, high = step_max)
        newpoint = newpoint + stepsize*direction
       
    return newpoint

def walkCDHRinBall(A,b,center,radius,point,nStep = 0):
    nDim = A.shape[1]
    
    if nStep == 0:
        nStep = nDim*10

    direction = np.zeros(nDim)
    direction_ID = np.random.randint(nDim)
    direction[direction_ID] = 1
    
    t = b - A@point
    A_col = A[:,direction_ID]
    negative_ID = np.where(A_col < 0)
    positive_ID = np.where(A_col > 0)
    
    step_min = min(-t[negative_ID]/A_col[negative_ID])
    step_max = min(t[positive_ID]/A_col[positive_ID])
    
    b_quad = (point - center)@direction
    c_quad = (point - center).T@(point - center) - radius**2

    if step_max**2 + 2*step_max*b_quad + c_quad> 0:
        step_max = -b_quad + np.sqrt((b_quad)**2 - c_quad)

    if step_min**2 - 2*step_min*b_quad+ c_quad > 0:
        step_min = b_quad + np.sqrt((b_quad)**2 - c_quad)    

    stepsize = np.random.uniform(low = -step_min, high = step_max)
    newpoint = point + stepsize*direction

    for stepcount in range(nStep):
        
        direction = np.zeros(nDim)
        direction_ID = np.random.randint(nDim)
        direction[direction_ID] = 1
        
        t = t - stepsize*A_col

        A_col = A[:,direction_ID]
        
        negative_ID = np.where(A_col < 0)
        positive_ID = np.where(A_col > 0)
        
        step_min = min(-t[negative_ID]/A_col[negative_ID])
        step_max = min(t[positive_ID]/A_col[positive_ID])
        
        b_quad = (newpoint-center)@direction
        c_quad = (newpoint-center).T@(newpoint-center) - radius**2
        
        if step_max**2 + 2*step_max*b_quad + c_quad> 0:
            step_max = -b_quad + np.sqrt((b_quad)**2 - c_quad)

        if step_min**2 - 2*step_min*b_quad+ c_quad > 0:
            step_min = b_quad + np.sqrt((b_quad)**2 - c_quad)

        
        stepsize = np.random.uniform(low = -step_min, high = step_max)
        
        newpoint = newpoint + stepsize*direction
        
    return newpoint

def ChebychevBall(A,b):
    
    nDim = A.shape[1]
    
    rowNrmA = np.linalg.norm(A,axis=1)
    A_linprog = np.vstack((rowNrmA, A.T)).T
    
    b_linprog = b
    
    obj_linprog = np.zeros(nDim+1)
    obj_linprog[0] = -1
    
    sol = opt.linprog(obj_linprog, A_ub=A_linprog, b_ub=b_linprog)
    vecZ = sol.x
    radius, center = vecZ[0], vecZ[1:]
   
    return radius, center

def MinVolEllipsoid(points, tol = 1e-6):
    """
    Find the minimum volume ellipse.
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    
    while err > tol:
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * np.linalg.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u-u)
        u = new_u
    c = u*points
    A = np.linalg.inv(points.T*np.diag(u)*points - c.T*c)/d    
    
    return np.asarray(A), np.squeeze(np.asarray(c))

def RoundnSandwich(A, b, Vertices, tol_ellipsoid= 1e-8):
    A_ellipsoid, c_ellipsoid = MinVolEllipsoid(Vertices, tol_ellipsoid)
    L_transform = np.linalg.cholesky(A_ellipsoid).T
    
    A_new = A@np.linalg.inv(L_transform)
    b_new = b - A@c_ellipsoid
    
    Poly_new = pc.Polytope(A_new,b_new)
    r_in, center = ChebychevBall(A_new, b_new)
    r_out = max([np.linalg.norm(L_transform@(x - c_ellipsoid) - center) for x in Vertices])
    
    return A_new, b_new, center, r_in, r_out, L_transform

def VolumeApprox_Mulitphase(A, b, Vertices):
    nDim = A.shape[1]
    
    Nmonte = 10**(nDim+1)
    A_new, b_new, center, r_in, r_out, L_transform = RoundnSandwich(A, b, Vertices)
    WalkStep = int(np.floor(10 + nDim/10))
    point = center

    alpha = int(np.floor(nDim*np.log2(r_in)))
    beta = int(np.ceil(nDim*np.log2(r_out)))
    
    rad_seq = np.linspace(r_in,r_out,beta-alpha+2)

    Vol = (np.pi**(nDim/2))*(rad_seq[0]**nDim)/(gamma(nDim/2 + 1))

    i = beta
    for i in range(beta-alpha,0,-1):
        count = 0
        point = center
        for n in range(Nmonte):
            point = walkCDHRinBall(A_new,b_new,point,rad_seq[i],point,WalkStep)
            if (point-center).T@(point-center) <= rad_seq[i-1]**2:
                count = count + 1
        Vol = Vol*Nmonte/count
        i = i - 1
            
    return Vol/np.linalg.det(L_transform)

def VolumeApprox_fast(A, b, Vertices, Nsample = 0):
    seed = None
    n = A.shape[1]
    if Nsample ==0:        
        N = 10**(n+1)
    else:
        N = Nsample    
    
    A_new, b_new, center, r_in, r_out, L_transform = RoundnSandwich(A, b, Vertices)
    
    Poly_new = pc.Polytope(A_new,b_new)
    l_b, u_b = Poly_new.bounding_box
  
    x = (np.tile(l_b, (1, N))
         + np.random.default_rng(seed).random((n, N))
         * np.tile(u_b - l_b, (1, N)))
    aux = (np.dot(A_new, x)
           - np.tile(np.array([b_new]).T, (1, N)))
    aux = np.nonzero(np.all(aux < 0, 0))[0].shape[0]
    Vol = np.prod(u_b - l_b) * aux / N
    #Vol = Poly_new.volume
    return Vol/np.linalg.det(L_transform)
#%% Test the functions
# Vertices = np.array([[100, 100],
#               [1, 0],
#               [0, 0]])

# Poly = pc.qhull(Vertices)

# A = Poly.A
# b = Poly.b

# Volume_VolEsti = VolumeApprox_fast(A, b, Vertices)
# Volume_Polytope = pc.volume(Poly)

