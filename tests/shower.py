import numpy as np

def shower2x2(u):
    
    d = np.zeros(2)
    y = np.zeros(2)
    y[0]=u[0]+u[1]
    if y[0]!=0:
        y[1]=(u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    else:
        y[1]=(60+120)/2
        
    return y


def shower3x3(u):
    
    # d = np.zeros(2)
    d = u[2]
    y = np.zeros(3)
    y[0]=u[0] + u[1]
    if y[0]!=0:
        y[1]=(u[0] * ( 60 + d ) + u[1]*120 ) / (u[0] + u[1] )
    else:
        y[1]=(60+120)/2
        
        
    y[2] = d
    return y


def shower3x2(u):
    
    
    d = u[2]
    y = np.zeros(2)
    y[0]=u[0] + u[1]
    if y[0]!=0:
        y[1]=(u[0] * ( 60 + d ) + u[1]*120 ) / (u[0] + u[1] )
    else:
        y[1]=(60+120)/2
        
        
    #y[2] = d
    return y


def shower2x3(u):
    
    # d = np.zeros(2)
    d = u[1]
    y = np.zeros(3)
    y[0]=u[0] + u[1]
    if y[0]!=0:
        y[1]=(u[0] * ( 60 + d ) + u[1]*120 ) / (u[0] + u[1] )
    else:
        y[1]=(60+120)/2
        
        
    y[2] = d
    return y

def inv_shower2x2(y):
    
    u = np.zeros(2)
   
    u[0] = (y[0]*(y[1]-60))/60
    u[1] = y[0] - u[0]
    
    return u

def inv_shower3x3(y):
    
    u = np.zeros(3)
   
    u[0] = (y[0]*(y[1]-60))/60
    u[1] = y[0] - u[0]
    u[2] = y[2]
    
    return u
