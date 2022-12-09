# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:35:10 2022

@author: sqd0001
"""
from operability_implicit_mapping import *
from DMA_MR_ss import *
from Feasibility_tests_bank import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from ultility_set import *
# AIS_bound = jnp.array([[0.0, 6.0],
#                     [0.0, 6.0]])

# output_init = np.array([3,3])
# # This resolution does not reveal the infeasibility
# AISresolution_feas = [4, 4]

# AIS_feas, AOS_feas, AIS_feas_poly, AOS_feas_poly = implicit_map(FeasTest1_implicit, 
#                                                                 AIS_bound, 
#                                                                 AISresolution_feas, 
#                                                                 output_init,
#                                                                 continuation='Explicit RK4')

# AIS_feas_plot = np.reshape(AIS_feas,(-1,2))
# AOS_feas_plot = np.reshape(AOS_feas,(-1,2))

# fig1, ax1 = plt.subplots()
# ax1.scatter(AIS_feas_plot[:,0], AIS_feas_plot[:,1])

# fig2, ax2 = plt.subplots()
# ax2.scatter(AOS_feas_plot[:,0], AOS_feas_plot[:,1])

# # This resolution reveals the infeasibility
# AISresolution_infeas = [20, 20]

# AIS_infeas, AOS_infeas, AIS_poly_infeas, AOS_poly_infeas = implicit_map(FeasTest1_implicit, 
#                                                                         AIS_bound, 
#                                                                         AISresolution_infeas, 
#                                                                         output_init,
#                                                                         continuation='Explicit RK4')
# AIS_infeas_plot = np.reshape(AIS_infeas,(-1,2))
# AOS_infeas_plot = np.reshape(AOS_infeas,(-1,2))

# fig3, ax3 = plt.subplots()
# ax3.scatter(AIS_infeas_plot[:,0], AIS_infeas_plot[:,1])

# fig4, ax4 = plt.subplots()
# ax4.scatter(AOS_infeas_plot[:,0], AOS_infeas_plot[:,1])

AIS_bound = np.array([[0.1, 10.0],
                    [0.1, 10.0]])



AISresolution = [10, 10]

output_init = np.array([0.0, 10.0])

t1 = time.time()
AIS, AOS, AIS_poly, AOS_poly = implicit_map(shower_implicit, 
                                            AIS_bound, 
                                            AISresolution, 
                                            output_init,
                                            continuation='odeint')

makeplot(AOS_poly)