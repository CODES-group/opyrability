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


