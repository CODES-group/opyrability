import jax.numpy as np
from scipy.optimize import root, root_scalar, fsolve
import matplotlib.pyplot as plt
from opyrability import AIS2AOS_map, create_grid
from jax import jacrev, grad, jit, jacfwd, hessian
import cyipopt
import pandas as pd

from jax.config import config
config.update("jax_enable_x64", True)


# ---------------------------- Set I of parameters -------------------------- #
# Values from Ray (1982) applicable for CSTR Model 1 
#                            from Subramanian (2003)

beta_1    = 0.35
xi_1      = 1.00
gamma_1   = 20.00
phi_1     = 0.11
sigma_s_1 = 0.44
sigma_b_1 = 0.06
q0_1      = 1.00
qc_1      = 1.00
x10_1     = 1.00
x20_1     = 0.00
x30_1     = -0.05


# ---------------------------- Set II of parameters ------------------------- #
# Values from Luyben (1993) applicable for CSTR Model 1 
#                            from Subramanian (2003)

def findA(E,k):
    R,T = 1.99, 599.67
    
    
    return k / np.exp(-E/(R*T))

E1, E2 = 30000, 15000
k1, k2 = 0.5, 0.05

Aa      =  findA(E1,k1)
Ab      =  findA(E2, k2)
beta    = 1.33
xi      = 0.6
gamma   = 25.18
phi     = 19.26
sigma_s = 91.42
sigma_b = 11.43
q0      = 1.00
x10     = 1.00
x20     = -0.12
x30     = - 0.12
qc      = 6.56
# I had to go to Luyben '93 to find these (original source).
dH1     = 20000
dH2     = 20000
eta1    = E1/E2
eta2    = dH1/dH2
Tr      = 599.67
Tc0     = 529.67

# x40, phi2 values were not given but it can be calculated.
x40     =  (Tc0 - Tr)/ Tr
phi2    = (Ab * np.exp(-gamma*(eta1 - 1 ))) / Aa


# ---------------------------- 1d CSTR functions ---------------------------- #
# @jit
# Define cstr functions
def cstr(x, AIS):
    
    beta_1    = 0.35
    xi_1      = 1.00
    gamma_1   = 20.00
    phi_1     = 0.11
    sigma_s_1 = 0.44
    sigma_b_1 = 0.06
    q0_1      = 1.00
    qc_1      = 1.00
    x10_1     = 1.00
    x20_1     = 0.00
    x30_1     = -0.05
    
    
    fx2 = np.exp((gamma_1*x)/(1+x)) # f(x2) Appendix B.1
    
    # Equation B.5 broken into terms
    A   = q0_1*(x20_1 - x)

    B   = (beta_1*phi_1*fx2*AIS[1]*q0_1*x10_1) / (q0_1 + phi_1*fx2*AIS[1])

    C   = (((sigma_s_1*AIS[1] + sigma_b_1)*AIS[0]*qc_1) /
         (qc_1*AIS[0] + xi_1*(sigma_s_1*AIS[1] + sigma_b_1)))

    D   = (x - x30_1)

    Eqn = A + B - (C*D)

    return Eqn


# @jit
def cstr2(x, AIS):
    
    fx3 = np.exp((gamma*x)/(1+x))
    
    # Equation B.13 broken into several terms (for the sake of sanity)
    A = q0*(x30 - x)
    
    B = phi*fx3*AIS[1]*beta
    
    C = (q0*x10)/(q0 + phi*fx3*AIS[1])
    
    D = (eta2*phi2*(fx3**(eta1-1)))/(q0 + phi*phi2*(fx3**(eta1))*AIS[1])
    
    E = (q0*x20 + (phi*fx3*AIS[1]*q0*x10)/(q0 + phi*fx3*AIS[1]))
    
    F = (sigma_s*AIS[1] + sigma_b)*qc*AIS[0]*(x - x40)
    
    G = (qc*AIS[0] + (sigma_s*AIS[1] + sigma_b)*xi)

    Eqn = A + B*(C + (D*E)) + (F/G)

    return Eqn



# High-order data (Jacobians/Hessians) for both models.
jac_cstr1_raw=grad(cstr)
hess_cstr1_raw = grad(grad(cstr))

grad_cstr2_raw =  grad(cstr2)
hess_cstr2_raw = grad(grad(cstr2))


# "Jiitize".
# @jit
def jac_cstr(x, *args):
    return np.array(jac_cstr1_raw(x, *args))

# @jit
def hess_cstr(x, *args):
    return np.array(hess_cstr1_raw(x, *args))
# @jit
def jac_cstr2(x, *args):
    return np.array(grad_cstr2_raw(x, *args))
# @jit
def hess_cstr2(x, *args):
    return np.array(hess_cstr2_raw(x, *args))


# Models based on 1d problem solution.
rtol=1e-11
xtol= 1e-10

def m(u):
        # 1st CSTR
        solution=root_scalar(cstr, args=u, method='secant',
                    bracket=[-0.5, 1], fprime=jac_cstr, fprime2=hess_cstr, 
                    x0=0.1, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
        
        if cstr(solution.root, u) > rtol:
            print('before corrector:')
            
            solution=root_scalar(cstr, args=u, method='secant',
                        bracket=[-0.5, 1], fprime=jac_cstr, fprime2=hess_cstr, 
                        x0=solution.root, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
            print('after corrector:')
            print(cstr(solution.root, u))
            print(solution.converged)
            
            
        if solution.converged is not True:
            solution=root_scalar(cstr, args=u, method='secant',
                        bracket=[-0.5, 1], fprime=jac_cstr, fprime2=hess_cstr, 
                        x0=0.25, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
            
       
        Temp_dimensionless = solution.root  # (x2)
        fx1 = np.exp((gamma_1*Temp_dimensionless)/(1 + Temp_dimensionless))
        xA_dimensionless = ((q0_1*x10_1) / (q0_1 + phi_1*fx1*u[1])) # Equation B.6 (x1)
        conversion = 1 - xA_dimensionless 
        
        # Jacket temperature (optional variable).
        jacket_temp = (qc_1*u[0]*x30_1 + xi_1*(sigma_s_1*u[1] + sigma_b_1)*xA_dimensionless)/ \
            (qc_1*u[0] +  xi_1*(sigma_s_1*u[1] + sigma_b_1))
        

        return np.array([Temp_dimensionless, conversion]).reshape(2,)
    


def m2(u):
    # 2nd CSTR.
    solution=root_scalar(cstr2, args=u, method='secant',
                bracket=[-0.5, 1], fprime=jac_cstr2, fprime2=hess_cstr2, 
                x0=0.1, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
    
    if cstr2(solution.root, u) > rtol:
        print('before corrector:')
        
        solution=root_scalar(cstr2, args=u, method='secant',
                    bracket=[-0.5, 1], fprime=jac_cstr2, fprime2=hess_cstr2, 
                    x0=solution.root, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
        print('after corrector:')
        print(cstr2(solution.root, u))
        print(solution.converged)
        
        
    if solution.converged is not True:
        solution=root_scalar(cstr2, args=u, method='secant',
                    bracket=[-0.5, 1], fprime=jac_cstr2, fprime2=hess_cstr2, 
                    x0=0.25, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
    
    
    Temp_dimensionless = solution.root
    Temp = Tr*(1 + Temp_dimensionless)
    fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
    xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
    conversion = 1 - (xA_dimensionless / x10)
    
    # Jacket temperature (optional variable)
    # jacket_temp = (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*xA_dimensionless) / \
    #     (qc*u[0] + xi*(sigma_s*u[1] + sigma_b))
        
    return np.array([conversion, Temp]).reshape(2,)



def m2_jacket(u):
    # 2nd CSTR.
    solution=root_scalar(cstr2, args=u, method='secant',
                bracket=[-0.5, 1], fprime=jac_cstr2, fprime2=hess_cstr2, 
                x0=0.1, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
    
    if cstr2(solution.root, u) > rtol:
        print('before corrector:')
        
        solution=root_scalar(cstr2, args=u, method='secant',
                    bracket=[-0.5, 1], fprime=jac_cstr2, fprime2=hess_cstr2, 
                    x0=solution.root, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
        print('after corrector:')
        print(cstr2(solution.root, u))
        print(solution.converged)
        
        
    if solution.converged is not True:
        solution=root_scalar(cstr2, args=u, method='secant',
                    bracket=[-0.5, 1], fprime=jac_cstr2, fprime2=hess_cstr2, 
                    x0=0.25, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
    
    
    Temp_dimensionless = solution.root
    Temp = Tr*(1 + Temp_dimensionless)
    fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
    xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
    conversion = 1 - (xA_dimensionless / x10)
    
    # Jacket temperature (optional variable)
    jacket_temp = (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*xA_dimensionless) / \
        (qc*u[0] + xi*(sigma_s*u[1] + sigma_b))
        
    return np.array([conversion, jacket_temp]).reshape(2,)
        

def m_cstr2_ipopt(u):
    
    # 2nd CSTR solved using IPOPT (Least-squares problem, similar to LM)
    x0=np.array([0.1])
    solver = NonlinearSystemSolver([cstr2],u)
    
    solution = solver.solve(x0)
    
    
    Temp_dimensionless = solution
    Temp = Tr*(1 + Temp_dimensionless)
    fx1 = np.exp((gamma*Temp_dimensionless)/(1 + Temp_dimensionless))
    xA_dimensionless = ((q0*x10) / (q0 + phi*fx1*u[1]))
    conversion = 1 - (xA_dimensionless / x10)
    # jacket_temp = (qc*u[0]*x30 + xi*(sigma_s*u[1] + sigma_b)*xA_dimensionless) / \
    #     (qc*u[0] + xi*(sigma_s*u[1] + sigma_b))
        
    return np.array([conversion, Temp]).reshape(2,)
    
# Generate edge points (boundary)
def generate_edge_points(bounds, n_points):
    """
    Generates boundary points on the edges of a hypercube defined by the given bounds.
    
    Parameters:
    - bounds: numpy array with shape (n_dimensions, 2) where each row is 
    [lower_bound, upper_bound] for a dimension.
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


# Cyipopt class for solving the problems if desired.
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
        x_L = np.ones(n) * -1.0e0
        x_U = np.ones(n) * 1.0e0
        problem = cyipopt.Problem(n=n, m=0, problem_obj=self, lb=x_L, ub=x_U)
        problem.addOption('print_level', 0)
        x, info = problem.solve(x0)
        return x


        
# %%  Model Run
# Run the models
# Temps = []
# Conversions = []

# AIS_bound =  np.array([[0.00,  1.42],
#                        [0.40,  1.40]])

# AIS_resolution = [100, 100]
# n_points = 10000

# # u_values = generate_edge_points(AIS_bound, n_points)

# u_values = create_grid(AIS_bound, AIS_resolution).reshape(AIS_resolution[0]**2, -1)

# # Multiplicity Region from  Subramanian (2003) - Digitized Using Webplotidigitizer.
# MR_data = pd.read_excel('mr_data_2.xlsx')
# u_values = np.vstack((np.array(MR_data), u_values))
# MR_reg =  np.array(MR_data)

# for u in u_values:
#     result = m(u)
#     Temps.append(result[0])
#     Conversions.append(result[1])


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(u_values[:,0], u_values[:,1], marker='o', linestyle='-',
#             c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2), cmap='rainbow')
# plt.scatter(MR_reg[:,0], MR_reg[:,1], color = 'black')
# plt.xlabel("Normalized Coolant flow")
# plt.ylabel("Normalized Volume")
# plt.grid(True)
# plt.title("Normalized Coolant flow vs. Normalized Volume - 1st CSTR")
# plt.show()

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(Temps, Conversions, marker='o', linestyle='-',
#             c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2) , cmap='rainbow')
# plt.scatter(Temps[0:MR_reg.shape[0]], Conversions[0:MR_reg.shape[0]], color = 'black')
# plt.xlabel("Temperature (Dimensionless)")
# plt.ylabel("Conversion")
# plt.grid(True)
# plt.title("Conversion vs. Temperature (Dimensionless) - 1st CSTR")
# plt.show()

# # CSTR 2nd model

# Temps = []
# Conversions = []

# AIS_bound =  np.array([[0.50, 4.00],
#                        [0.50, 1.00]])

# n_points = 10000

# u_values = generate_edge_points(AIS_bound, n_points)

# for u in u_values:
#     result = m2(u)
#     Temps.append(result[0])
#     Conversions.append(result[1])


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(u_values[:,0], u_values[:,1], marker='o', linestyle='-',
#             c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2))
# plt.xlabel("Normalized Coolant flow")
# plt.ylabel("Normalized Volume")
# plt.grid(True)
# plt.title("Normalized Coolant flow vs. Normalized Volume - 2nd CSTR")
# plt.show()

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(Temps, Conversions, marker='o', linestyle='-',
#             c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2))
# plt.xlabel("Temperature (Dimensionless)")
# plt.ylabel("Conversion")
# plt.grid(True)
# plt.title("Conversion vs. Temperature (Dimensionless) - 2nd CSTR")
# plt.show()


# # CSTR 2nd model - Jacket Temperature

# Temps = []
# Conversions = []

# AIS_bound =  np.array([[0.80, 20.00],
#                        [0.20, 10.00]])

# n_points = 10000

# u_values = generate_edge_points(AIS_bound, n_points)

# for u in u_values:
#     result = m2_jacket(u)
#     Temps.append(result[0])
#     Conversions.append(result[1])


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(u_values[:,0], u_values[:,1], marker='o', linestyle='-',
#             c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2))
# plt.xlabel("Normalized Coolant flow")
# plt.ylabel("Normalized Volume")
# plt.grid(True)
# plt.title("Normalized Coolant flow vs. Normalized Volume - 2nd CSTR")
# plt.show()

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(Temps, Conversions, marker='o', linestyle='-',
#             c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2))
# plt.xlabel("Temperature (Dimensionless)")
# plt.ylabel("Conversion")
# plt.grid(True)
# plt.title("Conversion vs. Jacket Temperature (Dimensionless) - 2nd CSTR")
# plt.show()




# %% Run models using opyrability

def m_implicit(input_vector, output_vector):
        # 1st CSTR
        
        
        # flowrate = u[0]
        # volume = u[1]
        
        T_adim =  output_vector[0]
        Conv   =  output_vector[1]
        
        x0=np.array([0.1])
        solver = NonlinearSystemSolver([cstr], input_vector)
        
        solution = solver.solve(x0)
        
        
        Temp_dimensionless = solution
        
        # solution=root_scalar(cstr, args=input_vector, method='newton',
        #             bracket=[-0.5, 1], fprime=jac_cstr, fprime2=hess_cstr, 
        #             x0=0.1, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
        
        # if cstr(solution.root, input_vector) > rtol:
        #     print('before corrector:')
            
        #     solution=root_scalar(cstr, args=input_vector, method='newton',
        #                 bracket=[-0.5, 1], fprime=jac_cstr, fprime2=hess_cstr, 
        #                 x0=solution.root, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
        #     print('after corrector:')
        #     print(cstr(solution.root, input_vector))
        #     print(solution.converged)
            
            
        # if solution.converged is not True:
        #     solution=root_scalar(cstr, args=input_vector, method='newton',
        #                 bracket=[-0.5, 1], fprime=jac_cstr, fprime2=hess_cstr, 
        #                 x0=0.25, x1=0.5, xtol=xtol, rtol=rtol, maxiter=10000)
            
       
        # Temp_dimensionless = solution.root  # (x2)
        fx1 = np.exp((gamma_1*Temp_dimensionless)/(1 + Temp_dimensionless))
        xA_dimensionless = ((q0_1*x10_1) / (q0_1 + phi_1*fx1*input_vector[1])) # Equation B.6 (x1)
        conversion = 1 - xA_dimensionless 
        
        # Jacket temperature (optional variable).
        jacket_temp = (qc_1*input_vector[0]*x30_1 + xi_1*(sigma_s_1*input_vector[1] + sigma_b_1)*xA_dimensionless)/ \
            (qc_1*input_vector[0] +  xi_1*(sigma_s_1*input_vector[1] + sigma_b_1))
        
        LHS1 = T_adim - Temp_dimensionless
        LHS2 = Conv - conversion
        
        
        
        return np.array([LHS1, LHS2]).reshape(2,)




from opyrability import implicit_map as imap

AIS_bound =  np.array([[0.00,  0.75],
                        [0.45,  1.00]])

# # AIS_bound =  np.array([[0.00,  1.5],
# #                         [0.00,  1.5]])

AIS_resolution = [5, 5]
# AOS, AIS = AIS2AOS_map(m, AIS_bound, AIS_resolution)

# AIS_bound2 =  np.array([[0.8,   16.00],
                        # [0.2,   6.00]])

# AIS_resolution2 = [150, 150]
# AOS, AIS = AIS2AOS_map(m, AIS_bound2, AIS_resolution2)






initial_estimate = np.array([0.5, 0.5])
AIS, AOS, AIS_poly, AOS_poly = imap(m_implicit, initial_estimate, 
                                    continuation='Explicit RK4', 
                                    domain_bound = AIS_bound, 
                                    domain_resolution = AIS_resolution,
                                    direction = 'forward',
                                    jit = False)



