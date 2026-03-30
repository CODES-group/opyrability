import numpy as np
import pyomo.environ as pyo

def shower2x2(u):
    d = np.zeros(2)
    y = np.zeros(2)
    y[0] = u[0] + u[1]
    if y[0] != 0:
        y[1] = (u[0]*(60+d[0]) + u[1]*(120+d[1])) / (u[0] + u[1])
    else:
        y[1] = (60+120)/2
    return y

def shower3x3(u):
    d = u[2]
    y = np.zeros(3)
    y[0] = u[0] + u[1]
    if y[0] != 0:
        y[1] = (u[0] * (60 + d) + u[1]*120) / (u[0] + u[1])
    else:
        y[1] = (60+120)/2
    y[2] = d
    return y

def shower3x2(u):
    d = u[2]
    y = np.zeros(2)
    y[0] = u[0] + u[1]
    if y[0] != 0:
        y[1] = (u[0] * (60 + d) + u[1]*120) / (u[0] + u[1])
    else:
        y[1] = (60+120)/2
    return y

def shower2x3(u):
    d = u[1]
    y = np.zeros(3)
    y[0] = u[0] + u[1]
    if y[0] != 0:
        y[1] = (u[0] * (60 + d) + u[1]*120) / (u[0] + u[1])
    else:
        y[1] = (60+120)/2
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

def build_shower2x2(m, u, y):
    m.mass_bal = pyo.Constraint(expr= y[0] == u[0] + u[1])
    m.energy_bal = pyo.Constraint(expr= y[1] * (u[0] + u[1]) == u[0]*60 + u[1]*120)

build_shower2x2.build_pyomo_constraints = True

def build_shower3x3(m, u, y):
    m.c1 = pyo.Constraint(expr= y[0] == u[0] + u[1])
    heat_in = u[0]*(60 + u[2]) + u[1]*120
    m.c2 = pyo.Constraint(expr= y[1] * (u[0] + u[1]) == heat_in)
    m.c3 = pyo.Constraint(expr= y[2] == u[2])

build_shower3x3.build_pyomo_constraints = True

def build_shower3x2(m, u, y):
    m.c1 = pyo.Constraint(expr= y[0] == u[0] + u[1])
    heat_in = u[0]*(60 + u[2]) + u[1]*120
    m.c2 = pyo.Constraint(expr= y[1] * (u[0] + u[1]) == heat_in)

build_shower3x2.build_pyomo_constraints = True
