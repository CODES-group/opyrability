"""Pyomo builder versions of the shower problem for the Pyomo/OMLT mapping
paths (equation-oriented counterparts of the callables in shower.py).

Builder contract: build_*(m, u, y) adds the model constraints linking the
input variables u to the output variables y on the Pyomo ConcreteModel m,
and the function is flagged with the build_pyomo_constraints attribute so
that opyrability auto-detects it. Adapted from PR #33 (Heitor F., @hfsf).
"""
import pyomo.environ as pyo


def build_shower2x2(m, u, y):
    """Shower problem, 2 inputs (hot/cold flowrates) x 2 outputs (total
    flowrate, temperature). Equation-oriented form of shower.shower2x2;
    valid for u[0] + u[1] > 0 (enforce via input bounds)."""
    m.flow_balance = pyo.Constraint(expr=y[0] == u[0] + u[1])
    # Energy balance written multiplied through by the total flowrate to
    # keep the expression polynomial (no division).
    m.energy_balance = pyo.Constraint(
        expr=y[1] * (u[0] + u[1]) == 60 * u[0] + 120 * u[1])


build_shower2x2.build_pyomo_constraints = True
