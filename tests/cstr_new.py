import jax.numpy as np
from scipy.optimize import root, root_scalar
import matplotlib.pyplot as plt
from opyrability import AIS2AOS_map
from jax import jacrev, grad, jit, jacfwd, hessian

from jax.config import config
config.update("jax_enable_x64", True)


# Values from Ray #
# beta = 0.35
# xi = 1.00
# gamma = 20.00
# phi = 0.11
# sigma_s = 0.44
# sigma_b = 0.06
# q0 = 1.00
# qc = 1.00
# x10 = 1.00
# x20 = 0.00
# x30 = -0.05



# ------------------------ Values from Luyben ------------------------------ #

def findA(E,k):
    R,T = 1.99, 599.67
    
    
    return k / np.exp(-E/(R*T))

E1, E2 = 30000, 15000
k1, k2 = 0.5, 0.05

Aa =  findA(E1,k1)
Ab =  findA(E2, k2)


beta = 1.33
xi = 0.6
gamma = 25.18
phi = 19.26
sigma_s = 91.42
sigma_b = 11.43
q0 = 1.00
x10 = 1.00
x20 =  -0.12
x30 = - 0.12
qc  = 6.56

dH1 = 20000
dH2 = 20000
eta1 = E1/E2
eta2 = dH1/dH2
Tr  = 599.67
Tc0 = 529.67

x40 =  (Tc0 - Tr)/ Tr

phi2 = (Ab * np.exp(-gamma*(eta1 - 1 ))) / Aa



@jit
# Define cstr function
def cstr(x, AIS):
    

    fx1 = np.exp((gamma*x)/(1+x))
    
    term1 =  q0*(x20 - x)
    term2 =  (beta*phi*fx1*AIS[1]*q0*x10) / (q0 + phi*fx1*AIS[1])
    term3 = ((sigma_s*AIS[1] + sigma_b)*AIS[0]*qc)/(qc*AIS[0] + xi*(sigma_s*AIS[1] + sigma_b))
    term4 = (x - x30)
    
    g_x1 = term1 + term2 - (term3*term4)
    
    return g_x1



def cstr2(x, AIS):
    
    fx3 = np.exp((gamma*x)/(1+x))
    
    term1 = q0*(x30 - x)
    term2 = phi*fx3*AIS[1]*beta
    term3 = (q0*x10)/(q0 + phi*fx3*AIS[1])
    term4 = (eta2*phi2*(fx3**(eta1-1)))/(q0 + phi*phi2*(fx3**(eta1))*AIS[1])
    term5 = (q0*x20 + (phi*fx3*AIS[1]*q0*x10)/(q0 + phi*fx3*AIS[1]))
    term6 = (sigma_s*AIS[1] + sigma_b)*qc*AIS[0]*(x - x40)
    term7 = (qc*AIS[0] + (sigma_s*AIS[1] +  sigma_b)*xi)
    
    g = term1 + term2*(term3 + (term4*term5)) + (term6/term7)
    
    return g




jac_cstr1_raw=jacrev(cstr)
hess_cstr_raw = jacfwd(jacrev(cstr2))

grad_cstr2_raw =  jacrev(cstr2)

hess_cstr2_raw = jacfwd(jacrev(cstr2))



@jit
def jac_cstr(x, *args):
    return np.array(jac_cstr1_raw(x, *args))

@jit
def hess_cstr(x, *args):
    return np.array(hess_cstr_raw(x, *args))
@jit
def grad_cstr2(x, *args):
    return np.array(grad_cstr2_raw(x, *args))
@jit
def hess_cstr2(x, *args):
    return np.array(hess_cstr2_raw(x, *args))
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
ftol=1e-8

def m(u):
    try:
        options = {'xtol': 1e-8, 'ftol': ftol }
        solution = root(cstr, x0=0.1, args=u, method='lm', jac=jac_cstr, 
                        tol=ftol, options = options)
        
        if not solution.success:
            # If the first attempt failed, try with the last iteration as the initial guess
            solution = root(cstr, x0=solution.x, args=u, method='lm', jac=jac_cstr, 
                            tol=ftol, options = options)
        
        # If the second attempt is also unsuccessful, raise an exception to move to the except block
        if not solution.success:
            raise ValueError("Solver failed to converge.")
        
        Temp_dimensionless = solution.x
        fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
        xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
        conversion = 1 - (xA_dimensionless / x10)
        jacket_temp = (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*xA_dimensionless)/ \
            (qc*u[0] +  xi*(sigma_s*u[1] + sigma_b))
        
    except ValueError:
        Temp_dimensionless = np.nan
        fx1 = np.nan
        xA_dimensionless = np.nan
        conversion = np.nan
        jacket_temp = np.nan

    return np.array([Temp_dimensionless, conversion]).reshape(2,)


def m_cstr2(u):
    solution = root_scalar(cstr2, x0=0.1, args=u, method='halley',
                           fprime=grad_cstr2, fprime2 = hess_cstr2,
                           bracket= (0, 1))
    
    
    Temp_dimensionless = solution.root
    Temp = Tr*(1 + Temp_dimensionless)
    fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
    xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
    conversion = 1 - (xA_dimensionless / x10)
    # jacket_temp = (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*xA_dimensionless) / \
    #     (qc*u[0] + xi*(sigma_s*u[1] + sigma_b))
        
    return np.array([conversion, Temp]).reshape(2,)
        

def m_cstr22(u):
    x0=np.array([0.1])
    solver = NonlinearSystemSolver([cstr2],u)
    # x0 = np.array([0.5, 0.5])
    solution = solver.solve(x0)
    
    
    Temp_dimensionless = solution
    Temp = Tr*(1 + Temp_dimensionless)
    fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
    xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
    conversion = 1 - (xA_dimensionless / x10)
    # jacket_temp = (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*xA_dimensionless) / \
    #     (qc*u[0] + xi*(sigma_s*u[1] + sigma_b))
        
    return np.array([conversion, Temp]).reshape(2,)
    

def generate_edge_points(bounds, n_points):
    """
    Generates boundary points on the edges of a hypercube defined by the given bounds.
    
    Parameters:
    - bounds: numpy array with shape (n_dimensions, 2) where each row is [lower_bound, upper_bound] for a dimension.
    - n_points: total number of points to generate.
    
    Returns:
    - edge_points: numpy array with the boundary points.
    """
    
    n_dims = bounds.shape[0]
    points_per_dim = int(round(n_points ** (1/n_dims)))
    
    edge_sets = []
    for i in range(n_dims):
        for bound in bounds[i]:
            slice_points = [np.linspace(bounds[j, 0], bounds[j, 1], points_per_dim) if j != i else np.array([bound]) 
                            for j in range(n_dims)]
            mesh = np.array(np.meshgrid(*slice_points)).T.reshape(-1, n_dims)
            edge_sets.append(mesh)
    
    edge_points = np.vstack(edge_sets)
    return edge_points

# Initialize the u values
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

# AIS_bound =  np.array([[0.00, 1.00],
#                         [0.25, 1.00]])

# n_points = 100000

# u_values = generate_edge_points(AIS_bound, n_points)


# for u in u_values:
#     result = m(u)
#     Temps.append(result[0])
#     Conversions.append(result[1])


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(u_values[:,0], u_values[:,1], marker='o', color='b', linestyle='-')
# plt.xlabel("Normalized Coolant flow")
# plt.ylabel("Normalized Volume")
# plt.grid(True)
# plt.title("Normalized Coolant flow vs. Normalized Volume")
# plt.show()

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(Temps, Conversions, marker='o', color='b', linestyle='-')
# plt.xlabel("Temperature (Dimensionless)")
# plt.ylabel("Conversion")
# plt.grid(True)
# plt.title("Conversion vs. Temperature (Dimensionless)")
# plt.show()

import cyipopt


class NonlinearSystemSolver:
    def __init__(self, funcs, args):
        self.funcs = funcs
        self.args = args
        # Compute the Jacobian of the vector function using jacrev
        self.jac_func = jacrev(self.vector_func)
        self.hess_func = hessian(self.objective)
    
    def vector_func(self, x):
        """Vector function representing the system of equations."""
        return np.array([f(x, self.args) for f in self.funcs])
    
    def objective(self, x):
        """Objective function: sum of squares of residuals."""
        residuals = np.array([f(x, self.args) for f in self.funcs])
        return np.sum(residuals**2)
    
    def gradient(self, x):
        """Calculate gradient using the Jacobian."""
        # Evaluate the Jacobian at the current x
        J = self.jac_func(x)
        # Compute the gradient as J^T * f(x)
        return J.T @ self.vector_func(x)
    
    def hessian(self, x, lagrange, obj_factor):
        """Calculate Hessian of the objective function."""
        # For systems of equations without constraints, the Hessian of the Lagrangian 
        # is just obj_factor times the Hessian of the objective.
        H = self.hess_func(x)
        return obj_factor * H
    
    def solve(self, x0):
        """Solve the system using IPOPT."""
        n = len(x0)
        x_L = np.ones(n) * -1.0e5
        x_U = np.ones(n) * 1.0e5
        problem = cyipopt.Problem(n=n, m=0, problem_obj=self, lb=x_L, ub=x_U)
        problem.addOption('print_level', 0)
        x, info = problem.solve(x0)
        return x

def m2(u):
    x0=np.array([0.1])
    solver = NonlinearSystemSolver([cstr],u)
    # x0 = np.array([0.5, 0.5])
    solution = solver.solve(x0)
    # solution = root_scalar(cstr, x0=0.1, args=u, method='newton',
    #                        fprime=jac_cstr, fprime2 = hess_cstr,
    #                        bracket= [0, 1.00], maxiter=5000, xtol=1e-10,
    #                        rtol= 1e-11)
    
    # print(solution.converged)
    
    Temp_dimensionless = solution
    fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
    xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
    conversion = 1 - (xA_dimensionless / x10)
    jacket_temp = (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*xA_dimensionless)/ \
        (qc*u[0] +  xi*(sigma_s*u[1] + sigma_b))
        
    return np.array([Temp_dimensionless, conversion]).reshape(2,)
        
        
    





# AIS_bound =  np.array([[0.00,  0.75],
#                        [0.00,  1.00]])

# AIS_resolution = [20, 20]
# AOS, AIS = AIS2AOS_map(m2, AIS_bound, AIS_resolution)

AIS_bound2 =  np.array([[0.8,   16.00],
                        [0.2,   6.00]])

AIS_resolution2 = [5, 5]
AOS, AIS = AIS2AOS_map(m_cstr22, AIS_bound2, AIS_resolution2)


# # Run the model
# Temps = []
# Conversions = []

# AIS_bound =  np.array([[0.80, 4.00],
#                         [0.2, 8.00]])

# n_points = 100

# u_values = generate_edge_points(AIS_bound, n_points)


# for u in u_values:
#     result = m_cstr22(u)
#     Temps.append(result[0])
#     Conversions.append(result[1])


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(u_values[:,0], u_values[:,1], marker='o', color='b', linestyle='-')
# plt.xlabel("Normalized Coolant flow")
# plt.ylabel("Normalized Volume")
# plt.grid(True)
# plt.title("Normalized Coolant flow vs. Normalized Volume")
# plt.show()

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(Conversions,Temps, marker='o', color='b', linestyle='-')
# plt.xlabel("Temperature (Dimensionless)")
# plt.ylabel("Conversion")
# plt.grid(True)
# plt.title("Conversion vs. Temperature (Dimensionless)")
# plt.show()

# AIS = AIS.reshape(100,-1)

# from opyrability import create_grid

# AIS_grid = create_grid(AIS_bound, AIS_resolution)

# AIS_grid = AIS_grid.reshape(100,2)

# AIS_nan =  AIS_grid[np.isnan(AIS)]
