import numpy as np
from opyrability import multimodel_rep, OI_eval, AIS2AOS_map

# -----------------------------------------------------------------------------
# Examples: Multimodel examples: DMA-MR and Shower Problem
# Author: Victor Alves
# Control, Optimization and Design for Energy and Sustainability,
# CODES Group, West Virginia University (2023)
# -----------------------------------------------------------------------------

"""
This script presents illustrative examples that showcase the application of the
Multimodel approach, applied to a membrane reactor (DMA-MR), alongside the 
classic Shower Problem.

Each example explores different scenarios and dimensions, offering insights 
into the functionality of opyrability.

"""

# %% DMA-MR - 2x2 System - Design Variables - Tube length and diameter [cm]
from dma_mr import dma_mr_design, dma_mr_mvs
# Defining DOS/AIS bounds and resolution.

DOS_bounds =  np.array([[20, 25], 
                        [35, 45]])

AIS_bounds =  np.array([[10, 150],
                        [0.5, 2]])

AIS_resolution =  [5, 5]


model  = dma_mr_design

legends = ['$Benzene \, production \, F_{C_{6}H_{6}} [mg/h]$',
           '$Methane \, conversion \, X_{CH_{4}} [\%]$']

# Obtaining AOS and evaluating the OI
AOS_region  =  multimodel_rep(model, 
                              AIS_bounds, 
                              AIS_resolution,
                              polytopic_trace='simplices',
                              labels=legends)

OI = OI_eval(AOS_region, DOS_bounds, labels = legends)


# %% DMA-MR - 2x2 System - Manipulated Variables - Shell and tube flow rates
#                                                                    [cm3/h]

# Defining DOS/AIS bounds and resolution
DOS_bounds = np.array([[15,25],
                        [35,45]])


AIS_bounds =  np.array([[450, 1500],
                    [450, 1500]])

AIS_resolution =  [5, 5]

model  = dma_mr_mvs

legends = ['$Benzene \, production \, F_{C_{6}H_{6}} [mg/h]$',
           '$Methane \, conversion \, X_{CH_{4}} [\%]$']

# Obtaining AOS and evaluating the OI
AOS_region  =  multimodel_rep(model, 
                              AIS_bounds,  
                              AIS_resolution,
                              labels = legends)

OI = OI_eval(AOS_region, DOS_bounds, labels = legends)

# %% Shower problem 2x2 - Classic problem
from shower import shower2x2
from opyrability import AIS2AOS_map

# Defining DOS/AIS bounds and resolution
DOS_bounds =  np.array([[10, 20], 
                        [70, 100]])

AIS_bounds =  np.array([[0, 10],
                        [0, 10]])

AIS_resolution =  [10, 10]

model =  shower2x2

# Obtaining input-output mapping, AOS and evaluating the OI
AIS, AOS = AIS2AOS_map(model, 
                       AIS_bounds, 
                       AIS_resolution, 
                       plot=False)


AOS_region  =  multimodel_rep(model, 
                              AIS_bounds,
                              AIS_resolution, 
                              polytopic_trace = 'simplices', 
                              plot=True)

OI = OI_eval(AOS_region, DOS_bounds)

# %% Shower problem 3x3 - The third dimension is disturbance in the inlet temp.
# This example also has the EDS (Expected Disturbance Set)
from shower import shower3x3


# Defining DOS/AIS/EDS bounds and resolution
DOS_bounds =  np.array([[10.00, 20.00], 
                        [70.00, 100.00],
                        [-10.00, 10.00]])


AIS_bounds =  np.array([[0.00, 10.00],
                        [0.00, 10.00]])


EDS_bounds = np.array([[-10.00, 10.00]])

AIS_resolution = [5, 5]

EDS_resolution = [5]

model =  shower3x3

# Obtaining AOS and evaluating the OI
AOS_region  =  multimodel_rep(model,
                              AIS_bounds, 
                              AIS_resolution,
                              EDS_bound=EDS_bounds,
                              EDS_resolution=EDS_resolution)

OI = OI_eval(AOS_region, DOS_bounds)

# %% Shower problem 3x2 - To showcase a non-square example
from shower import shower3x2


# Defining DOS/AIS bounds and resolution
DOS_bounds =  np.array([[10.00, 20.00], 
                        [70.00, 100.00]])


AIS_bounds =  np.array([[0.00, 10.00],
                        [0.00, 10.00],
                        [-10.00, 10.00]])

AIS_resolution =  [5, 5, 5]

model =  shower3x2

# Obtaining input-output mapping, AOS and evaluating the OI
AIS, AOS = AIS2AOS_map(model, 
                       AIS_bounds, 
                       AIS_resolution, 
                       plot=False)

AOS_region  =  multimodel_rep(model, 
                              AIS_bounds, 
                              AIS_resolution,
                              plot=False, 
                              polytopic_trace='polyhedra')

OI = OI_eval(AOS_region, DOS_bounds)


# %% Shower inverse mapping - multimodel representation - 3x3

# In this case, the forward model of the shower problem is used to obtain
# the inverse multimodel map representation. In short, 'multimodel_rep' uses
# 'nlp_based_approach' if the user indicates perspective = 'inputs'. You will
# be asked an initial estimate at the terminal for the inverse mapping.

# In this example, it is 10,10,5.

from shower import shower3x3

model = shower3x3

# Defining DOS bounds and resolution - this is an inverse map example.
DOS_bounds = np.array([[17.5, 21.0],
                    [80.0, 100.0],
                    [-10, 10]])

DOS_resolution = [6, 6, 6]

DIS_bounds =  np.array([[0, 10],
                        [0, 10],
                        [-10, 10]])

# Obtaining input-output mapping, AIS and evaluating the OI from the inputs'
# perspective.


AIS_region  =  multimodel_rep(model, 
                              DOS_bounds, 
                              DOS_resolution, 
                              perspective = 'inputs')
   
OI = OI_eval(AIS_region, 
              DIS_bounds, 
              perspective='inputs')

# %% Shower inverse mapping - multimodel representation - 2x2

# Inverse mapping of the classic 2x2 shower problem. As the example above,
# we will obtain the OI from the inputs perspective, using 'multimodel_rep' and
# its native connectivity with 'nlp_based_approach'.

# You will be asked an initial estimate at the terminal for the inverse mapping.

# In this example, it is 5,5.
from shower import shower2x2
model =  shower2x2
# Defining DOS bounds and resolution - this is an inverse map example.
DOS_bounds =  np.array([[10, 20], 
                        [70, 100]])

DIS_bounds =  np.array([[0, 10],
                        [0, 10]])


DOS_resolution = [3, 3]



# Obtaining AIS and evaluating the OI from the inputs'
# perspective.
AIS_region  =  multimodel_rep(model,
                              DOS_bounds, 
                              DOS_resolution, 
                              perspective='inputs')

OI = OI_eval(AIS_region, 
              DIS_bounds, 
              perspective='inputs')

