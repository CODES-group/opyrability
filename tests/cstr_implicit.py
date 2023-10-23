import jax
import jax.numpy as np
from scipy.optimize import root, root_scalar, fsolve
import matplotlib.pyplot as plt
from opyrability import AIS2AOS_map, create_grid
from jax import jacrev, jit, vmap
import cyipopt
import pandas as pd
from jaxopt import ScipyRootFinding, Broyden
from opyrability import implicit_map as imap
import time

from jax.config import config
config.update("jax_enable_x64", True)

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

# x40, phi2 values were not given but they can be calculated.
x40     =  (Tc0 - Tr)/ Tr
phi2    = (Ab * np.exp(-gamma*(eta1 - 1 ))) / Aa


# ---------------------------- 1d CSTR functions ---------------------------- #
# Define cstr functions
# @jit
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
    # Eqn = Eqn.astype(float)
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





def cstr_equation(Temp_dimensionless, AIS):
    return cstr(Temp_dimensionless, AIS)

rtol=1e-6
xtol= 1e-6
gtol = 1e-6


# Using JaxOPT (Differentiable Root-finding from Google)
cstr_solver = Broyden(cstr_equation, tol=rtol, maxls=1000, jit=True)
cstr_solver_nojit = cstr_solver
# cstr_solver_nojit = ScipyRootFinding(optimality_fun=cstr_equation , method='hybr', jit=False)
# Other option is to use the Scipy.optimize.root wrapper written by Google:
# cstr_solver = ScipyRootFinding(method="hybr",
#                                optimality_fun=cstr_equation,
#                                jit=True,
#                                use_jacrev=True, tol=rtol,
#                                options={'ftol': rtol, 'xtol': xtol, 
#                                         "factor": 1.0})



# Models for Operability analysis (AIS-AOS maps)
# @jit
def m_jax(u):
        # 1st CSTR
        initial_estimate = np.array([0.1])
        solution = cstr_solver.run(initial_estimate, AIS=u)
        
        # This line avoids memory leak in JaxOpt. 
        # See https://github.com/google/jaxopt/issues/380 and
        # https://github.com/google/jaxopt/issues/548. Works well. :)
        jax.clear_caches()
        
        
        # This is the equivalent of a if statment (control flow), Jax-compatible.
        def true_fun(_):
            # print('before corrector:')
            initial_estimate = np.array([0.25])
            solution_corrected = cstr_solver.run(initial_estimate, AIS=u)
            # print('after corrector:')
            # print(cstr(solution_corrected.params, u))
            return solution_corrected
        
        def false_fun(_):
            return solution
        
        condition = cstr(solution.params, u)[0] > rtol
        solution = jax.lax.cond(condition, true_fun, false_fun, None)
        
        # condition_val = jax.device_get(cstr(solution.params, u)[0] > rtol)

        # if condition_val:
        #     solution = true_fun(None)
        # else:
        #     solution = false_fun(None)
        
       
            
        # Calculating Outputs from the nonlinear equation results.
        Temp_dimensionless = solution.params # x(2) 
        
        fx1 = np.exp((gamma_1*Temp_dimensionless)/(1 + Temp_dimensionless))
        
        xA_dimensionless = ((q0_1*x10_1) / (q0_1 + phi_1*fx1*u[1])) # Equation B.6 (x1)
        
        conversion = 1 - xA_dimensionless 
        
        # Jacket temperature (optional variable).
        # jacket_temp = (qc_1*u[0]*x30_1 + xi_1*(sigma_s_1*u[1] + sigma_b_1)*xA_dimensionless)/ \
        #     (qc_1*u[0] +  xi_1*(sigma_s_1*u[1] + sigma_b_1))
        
        # conversion = jacket_temp       
        
        return np.array([Temp_dimensionless, conversion]).reshape(2,)
    


# @jit
def m_implicit(input_vector, output_vector):
    
        # 1st CSTR in implicit form F(u,y) = 0
        T_adim, Conv =  output_vector
        u = input_vector
        
        # 1st CSTR
        initial_estimate = np.array([0.1])
        solution = cstr_solver_nojit.run(initial_estimate, AIS=u)
        
        # This line avoids memory leak in JaxOpt. 
        # See https://github.com/google/jaxopt/issues/380 and
        # https://github.com/google/jaxopt/issues/548. Works well. :)
        jax.clear_caches()
        
        
        # This is the equivalent of a if statment (control flow), Jax-compatible.
        def true_fun(_):
            # print('before corrector:')
            initial_estimate = np.array([0.25])
            solution_corrected = cstr_solver.run(initial_estimate, AIS=u)
            # print('after corrector:')
            # print(cstr(solution_corrected.params, u))
            return solution_corrected
        
        def false_fun(_):
            return solution
        
        condition = cstr(solution.params, u)[0] > rtol
        solution = jax.lax.cond(condition, true_fun, false_fun, None)
        
        # condition_val = jax.device_get(cstr(solution.params, u)[0] > rtol)

        # if condition_val:
        #     solution = true_fun(None)
        # else:
        #     solution = false_fun(None)
        
       
            
        # Calculating Outputs from the nonlinear equation results.
        Temp_dimensionless = solution.params # x(2) 
        
        fx1 = np.exp((gamma_1*Temp_dimensionless)/(1 + Temp_dimensionless))
        
        xA_dimensionless = ((q0_1*x10_1) / (q0_1 + phi_1*fx1*u[1])) # Equation B.6 (x1)
        
        conversion = 1 - xA_dimensionless 
        
        LHS1 = T_adim - Temp_dimensionless
        LHS2 = Conv - conversion
        
        
        # jacket_temp = (qc_1*input_vector[0]*x30_1 + xi_1*(sigma_s_1*input_vector[1] + sigma_b_1)*xA_dimensionless)/ \
        #     (qc_1*input_vector[0] +  xi_1*(sigma_s_1*input_vector[1] + sigma_b_1))
        
        
        
        return np.array([LHS1, LHS2]).reshape(2,)
    
# %% Implicit mapping
# AIS_bound =  np.array([[0.25,  1.2],
#                         [0.50,  1.2]])


# AIS_resolution = [5, 5]

# initial_estimate = np.array([0.1, 0.1])
# AIS, AOS, AIS_poly, AOS_poly = imap(m_implicit, initial_estimate, 
#                                     continuation='Explicit RK4', 
#                                     domain_bound = AIS_bound, 
#                                     domain_resolution = AIS_resolution,
#                                     direction = 'forward', jit= True,
#                                     validation= 'Corrector')    


# AIS_plot = np.reshape(AIS,(-1,2))
# AOS_plot = np.reshape(AOS,(-1,2))

# fig1, ax1 = plt.subplots()
# ax1.scatter(AIS_plot[:,0], AIS_plot[:,1])

# fig2, ax2 = plt.subplots()
# ax2.scatter(AOS_plot[:,0], AOS_plot[:,1])
# %%  Brute force enumeration
# Run the models

AIS_bound =  np.array([[0.25,  1.20],
                        [0.50, 1.20]])

AIS_resolution = [100, 100]
# 
# n_points = 10000

# u_values = generate_edge_points(AIS_bound, n_points)

u_values = create_grid(AIS_bound, AIS_resolution).reshape(AIS_resolution[0]**2, -1)
# 
# Multiplicity Region from  Subramanian (2003) - Digitized Using Webplotidigitizer.
# MR_data = pd.read_excel('mr_data.xlsx')
# u_values = np.vstack((np.array(MR_data), u_values))
# MR_reg =  np.array(MR_data)


# Vectorized map using Jax (fasssssssssssssst :) )
vectorized_map = vmap(m_jax)

start_time = time.time()
results = vectorized_map(u_values)
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

Temps = results[:, 0]
Conversions = results[:, 1]
    
    
# Plotting AIS
plt.figure(figsize=(10, 5))
plt.scatter(u_values[:,0], u_values[:,1], marker='o', linestyle='-',
            c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2), cmap='rainbow')
# plt.scatter(MR_reg[:,0], MR_reg[:,1], color = 'black')
plt.xlabel("Normalized Coolant flow")
plt.ylabel("Normalized Volume")
plt.grid(True)
plt.title("Normalized Coolant flow vs. Normalized Volume - 1st CSTR")
plt.show()

# Plotting AOS
plt.figure(figsize=(10, 5))
plt.scatter(Temps, Conversions, marker='o', linestyle='-',
            c=np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2) , cmap='rainbow')
# plt.scatter(Temps[0:MR_reg.shape[0]], Conversions[0:MR_reg.shape[0]], color = 'black')
plt.xlabel("Temperature (Dimensionless)")
plt.ylabel("Conversion")
plt.grid(True)
plt.title("Conversion vs. Temperature (Dimensionless) - 1st CSTR")
plt.show()


