import jax.numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from opyrability import AIS2AOS_map
from jax import jacrev


beta = 0.35
xi = 1.00
gamma = 20.00
phi = 0.11
sigma_s = 0.44
sigma_b = 0.06
q0 = 1.00
qc = 1.00
x10 = 1.00
x20 = 0.00
x30 = -0.05


# beta = 1.33
# xi = 0.6
# gamma = 25.18
# phi = 19.26
# sigma_s = 91.42
# sigma_b = 11.43
# q0 = 1.00
# x10 = 1.00
# x20 =  -0.12
# x30 = - 0.12
# qc  = 6.56



# Define cstr function
def cstr(x, AIS):
    

    fx1 = np.exp((gamma*x)/(1+x))
    
    term1 =  q0*(x20 - x)
    term2 =  (beta*phi*fx1*AIS[1]*q0*x10) / (q0 + phi*fx1*AIS[1])
    term3 = ((sigma_s*AIS[1] + sigma_b)*AIS[0]*qc)/(qc*AIS[0] + xi*(sigma_s*AIS[1] + sigma_b))
    term4 = (x - x30)
    
    g_x1 = term1 + term2 - (term3*term4)
    
    return g_x1


# jac_cstr_raw=jacrev(cstr)

# def jac_cstr(x, *args):
#     return np.array(jac_cstr_raw(x, *args))

# Define m function
# def m(u):
    
    
    
#     solution = root(cstr, x0 = 0.1, args=u, method='hybr', tol=1e-16)
 
#     if solution.success is True:
#         Temp_dimensionless = solution.x
#         fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
#         xA_dimensionless =  ((q0*x10)/ (q0 + phi*fx1*u[1]))
#         # conversion = ((x10 - xA_dimensionless)/ x10)
#         conversion = 1 - ( xA_dimensionless / x10)
#     else:
#         Temp_dimensionless = np.nan
#         fx1 = np.nan
#         xA_dimensionless = np.nan
#         conversion = np.nan
        
    
    
    
#     return np.array([Temp_dimensionless, conversion]).reshape(2,)


def m(u):
    try:
        solution = root(cstr, x0=0.1, args=u, method='hybr', tol=1e-16)
        
        if not solution.success:
            # If the first attempt failed, try with the last iteration as the initial guess
            solution = root(cstr, x0=solution.x, args=u, method='hybr', tol=1e-16)
        
        # If the second attempt is also unsuccessful, raise an exception to move to the except block
        if not solution.success:
            raise ValueError("Solver failed to converge.")
        
        Temp_dimensionless = solution.x
        fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
        xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
        conversion = 1 - (xA_dimensionless / x10)
        
    except ValueError:
        Temp_dimensionless = np.nan
        fx1 = np.nan
        xA_dimensionless = np.nan
        conversion = np.nan

    return np.array([Temp_dimensionless, conversion]).reshape(2,)

# # Initialize the u values
# u_values = np.array([
#     [0.553030303, 1.001104566],
#     [0.58030303, 1.001104566],
#     [0.604545455, 1.001104566],
#     [0.63030303, 1.001104566],
#     [0.654545455, 1.001104566],
#     [0.68030303, 1.001104566],
#     [0.707575758, 1.001104566],
#     [0.731818182, 0.999631811],
#     [0.706060606, 0.986377025],
#     [0.631818182, 0.940721649],
#     [0.557575758, 0.895066274],
#     [0.486363636, 0.849410898],
#     [0.342424242, 0.752209131],
#     [0.272727273, 0.700662739],
#     [0.206060606, 0.646170839],
#     [0.140909091, 0.590206186],
#     [0.08030303, 0.528350515],
#     [0.001515152, 0.412002946],
#     [0.001515152, 0.423784978],
#     [0, 0.439985272],
#     [0, 0.456185567],
#     [0, 0.47533137],
#     [0, 0.491531664],
#     [0, 0.509204713],
#     [0, 0.525405007],
#     [0, 0.543078056],
#     [0, 0.560751105],
#     [0, 0.578424153],
#     [0, 0.594624448],
#     [0, 0.612297496],
#     [0.016666667, 0.624079529],
#     [0.057575758, 0.671207658],
#     [0.131818182, 0.740427099],
#     [0.210606061, 0.803755523],
#     [0.295454545, 0.859720177],
#     [0.381818182, 0.911266568],
#     [0.471212121, 0.959867452],
#     [0.522727273, 0.984904271],
#     [0.413636364, 0.800810015]
# ])

# # Run the model
# Temps = []
# Conversions = []

# for u in u_values:
#     result = m(u)
#     Temps.append(result[0])
#     Conversions.append(result[1])


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(Temps, Conversions, marker='o', color='b', linestyle='-')
# plt.xlabel("Temperature (Dimensionless)")
# plt.ylabel("Conversion")
# plt.grid(True)
# plt.title("Conversion vs. Temperature (Dimensionless)")
# plt.show()


AIS_bound =  np.array([[0.00, 0.75],
                       [0.25, 1.00]])

AIS_resolution = [10, 10]
AOS, AIS = AIS2AOS_map(m, AIS_bound, AIS_resolution)

AIS = AIS.reshape(100,-1)

from opyrability import create_grid

AIS_grid = create_grid(AIS_bound, AIS_resolution)

AIS_grid = AIS_grid.reshape(100,2)

# AIS_nan =  AIS_grid[np.isnan(AIS)]
