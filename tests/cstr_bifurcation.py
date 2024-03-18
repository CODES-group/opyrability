import jax.numpy as np
from jax import jacrev

def cstr(x, AIS):
    

    
    u = AIS
    
    # beta    = 1.33
    # xi      = 0.60
    # gamma   = 25.18
    # phi     = 19.26
    # sigma_s = 91.42
    # sigma_b = 11.43
    # q0      = 1.00
    # qc      = 6.56
    # Tr      = 599.67
    
    # Ray 1982
    beta    = 0.35
    xi      = 1.00
    gamma   = 20.00
    phi     = 0.11
    sigma_s = 0.44
    sigma_b = 0.06
    q0      = 1.00
    qc      = 1.00
    
    x10     = 1.00
    x20     = 0.00
    x30     = -0.05
    
    fx1 = np.exp((gamma*x)/(1+x))
    
    term1 =  q0*(x20 - x)
    term2 =  (beta*phi*fx1*q0*x10) / (q0 + phi*fx1*u[1])
    term3 = ((sigma_s*u[1] + sigma_b)*u[0]*qc)/(qc*u[0] + xi*(sigma_s*u[1] + sigma_b))
    term4 = (x - x30)
    
    g_x1 = term1 + term2 - term3*term4
    
    # g_x0 =  x[0] - q0*x10/(q0 + phi*fx1*u[1])
    # g2 =  x[2] - (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*x[1]) \
    # / (qc*u[0] + xi*(sigma_s*u[1] + sigma_b)) 
    
    return g_x1
    
# def cstr_jac(u, x10, x20, x30):
    
#     return jacrev(cstr(u, x10, x20,x30))

