import jax.numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
# Load data that was generated in the jupyter notebook session.
data = np.load('montecarlo.npz')

v_He_values = data['arr_0']
v0_values  = data['arr_1']
DIS_points =  data['arr_2']



theta = np.linspace(0, 2 * np.pi, 400)
phi = np.pi / 4
a, b= 0.15, 1
h, k = 22.4 , 39.4
y1 = h +  (a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi))  
y2 = k +  (b * np.sin(theta) * np.cos(phi) + a * np.cos(theta) * np.cos(phi))

from jax import random
# Set the key for random number generation
key = random.PRNGKey(0)

# Number of simulation points and center of ellipse.
num_simulations = 10000
a, b= 0.15, 1




# Scaling factor for 95% confidence interval in 2D
from scipy.stats import chi2
alpha = 0.95 # Confidence
dof   = 2    # Degrees of freedom
# scaling factor, 2.4477 for 95%
scaling_factor  = np.sqrt(chi2.ppf(alpha, dof))  # take the square root. :)



# Adjust the a and b values - here a and b are adjusted to be able to build 
# the covariance matrix to draw the multivariate normal distribution 
# that will be within 95% of the cloud of points.
a_adjusted = a / scaling_factor
b_adjusted = b / scaling_factor


# Constructing the covariance matrix using a, b, and rotation matrix

covariance_matrix_initial = np.array([[a_adjusted**2,  0], 
                                      [0,              b_adjusted**2]])


rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], 
                            [np.sin(phi), np.cos(phi)]])


covariance_matrix_constructed = rotation_matrix @ covariance_matrix_initial @ rotation_matrix.T


# Plotting the ellipses with constructed covariance matrix
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(y1, y2, 'r--', label='Confidence Ellipse from Parametric Equation', linewidth=3)
# Monte Carlo Sampling simulations
mean = np.array([h, k])
samples = random.multivariate_normal(key, mean, covariance_matrix_constructed, (num_simulations,))
hb=ax.hexbin(samples[:, 0], samples[:, 1], gridsize=50, cmap='viridis', bins='log', label='Monte Carlo (Hexagonal bins)')
fig.colorbar(hb, ax=ax, label='$log_{10}(N)$')
ax.set_xlabel('Benzene ($C_{6}H_{6}$) production [mg/h]')
ax.set_ylabel('Methane ($CH_4$) conversion [%]')
ax.set_title('Output Space Variables (Domain)')
ax.legend()
plt.tight_layout()
plt.show()




fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(DIS_points[1:,0], DIS_points[1:,1],'r--', label='Opyrability Inverse Implicit Mapping', linewidth=3)
hb2 = ax2.hexbin(v0_values, v_He_values, gridsize=50, cmap='viridis', bins='log', label='Monte Carlo (Hexagonal bins)')
fig2.colorbar(hb2, ax=ax2, label='$log_{10}(N)$')
ax2.set_title('Input Space Variables (Image)')
ax2.set_xlabel('Tube flow rate [$cm^3/h$]')
ax2.set_ylabel('Shell flow rate [$cm^3/h$]')
ax2.legend()
plt.tight_layout()
plt.show()


import matplotlib.path as mpath

path = mpath.Path(np.array(DIS_points))
points =  np.hstack([v0_values, v_He_values])

inside_points = path.contains_points(points)

points_in =  np.sum(inside_points)
points_out = len(points) - points_in

print('there are', points_in, 'points within the ellipse')
print('there are', points_out, 'points outside the ellipse')

print(points_in/num_simulations * 100, '%')

import seaborn as sns

g = sns.jointplot(x=v0_values.reshape(10000,), y=v_He_values.reshape(10000,), kind="hex", color="#002855")
g.fig.set_size_inches(10, 8)

g.set_axis_labels('Tube flow rate [$cm^3/h$]','Shell flow rate [$cm^3/h$]')
g.fig.suptitle('Hexagonal bin - Input variables s.t disturbances', fontsize=16, y=1.03)
# g.ax_joint.plot(DIS_points[1:, 0], DIS_points[1:, 1], 'r--', label="Opyrability Inverse Implicit Mapping") 
line = g.ax_joint.plot(DIS_points[1:, 0], DIS_points[1:, 1], 'r--', label="Opyrability Inverse Implicit Mapping")
plt.tight_layout()
g.ax_joint.legend(handles=line, loc='upper right')