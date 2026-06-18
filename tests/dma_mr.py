from __future__ import annotations
from numpy import pi as pi
import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
import scipy.integrate as spint
from dataclasses import dataclass

def dma_mr(z,F,k1,k1_Inv,k2,k2_Inv,T,Q,Pt,v0,At,Ft0,Ps,v_He,F_He,dt,selec):
    
    
    # r = np.zeros(2)
    # dFdz = np.zeros(8)

    # Avoid negative flows that can happen in the first integration steps.
    # Consequently this avoids that any molar balance (^ 1/4 terms) generates
    # complex numbers.
    F[F<0]=0
    
    # Evaluate total flowrate in tube & shell.
    Ft = F[0] + F[1] + F[2] + F[3]
    Fs = F[4] + F[5] + F[6] + F[7] + F_He
    v  = v0*(Ft/Ft0)
    
    # Concentrations from molar flowrates [mol/cm3]
    C = np.array([])
    C = F[:4] / v

    # Partial pressures - Tube & Shell [mol/cm3]
    
    P0t = ((Pt/101325)*(F[0]/Ft))
    P1t = ((Pt/101325)*(F[1]/Ft))
    P2t = ((Pt/101325)*(F[2]/Ft))
    P3t = ((Pt/101325)*(F[3]/Ft))
    
    P0s=((Ps/101325)*(F[4]/Fs)) 
    P1s=((Ps/101325)*(F[5]/Fs)) 
    P2s=((Ps/101325)*(F[6]/Fs)) 
    P3s=((Ps/101325)*(F[7]/Fs))
    
    # Reaction rates [mol/(h.cm3)]
    
    if C[0]==0:
        r0  = 0
    else:
        r0 = (3600*k1*C[0]*(1-((k1_Inv*C[1]*C[2]**2)/(k1*C[0]**2))))
    
    if C[1]==0:
        # r[1] = 0
        r1 = 0
    else:
        r1 = (3600*k2*C[1]*(1-((k2_Inv*C[3]*C[2]**3)/(k2*C[1]**3))))
    
    # Molar balances adjustment with experimental data.
    eff = 0.9
    vb  = 0.5
    Cat = (1-vb)*eff
    
    # Molar balances dFdz - Tube (0 to 3) & Shell (4 to 7)
    dF0 = -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF1 = 1 / 2 * Cat * r0 * At - Cat * r1 * At - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF2 = Cat * r0 * At + Cat * r1 * At- (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF3 = (1 / 3) * Cat * r1 * At - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    dF4 = (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF5 = (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF6 = (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF7 = (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    
    dFdz = np.array([ dF0, dF1, dF2, dF3, dF4, dF5, dF6, dF7 ])
    
    return dFdz


def dma_mr_jax(F, z, dt):

    At = 0.25 * jnp.pi * (dt ** 2)  # Cross sectional area[cm³]
    
    # Avoid negative flows that can happen in the first integration steps.
    # Consequently this avoids that any molar balance (^ 1/4 terms) generates
    # complex numbers.
    F = jnp.where(F <= 1e-9, 1e-9, F)
    

    
    # Evaluate total flowrate in tube & shell.
    Ft = F[0:4].sum()
    Fs = F[4:].sum() + F_He
    v = v0 * (Ft / Ft0)

    # Concentrations from molar flowrates [mol/cm3]
    C = F[:4] / v
    # Partial pressures - Tube & Shell [mol/cm3]

    P0t = (Pt / 101325) * (F[0] / Ft)
    P1t = (Pt / 101325) * (F[1] / Ft)
    P2t = (Pt / 101325) * (F[2] / Ft)
    P3t = (Pt / 101325) * (F[3] / Ft)

    P0s = (Ps / 101325) * (F[4] / Fs)
    P1s = (Ps / 101325) * (F[5] / Fs)
    P2s = (Ps / 101325) * (F[6] / Fs)
    P3s = (Ps / 101325) * (F[7] / Fs)
    
    

    
    r0 = 3600 * k1 * C[0] * (1 - ((k1_Inv * C[1] * C[2] ** 2) / 
                                  (k1 * (C[0])**2 )))
    

    # This replicates an if statement whenever the concentrations are near zero
    C0_aux = C[0]
    r0 = jnp.where(C0_aux <= 1e-9, 0, r0)
    
    r1 = 3600 * k2 * C[1] * (1 - ((k2_Inv * C[3] * C[2] ** 3) / 
                                  (k2 * (C[1])**3 )))
    

    # Same as before
    C1_aux = C[1]
    r1 = jnp.where(C1_aux <= 1e-9 , 0, r1)  

    # Molar balances adjustment with experimental data.
    eff = 0.9
    vb = 0.5
    Cat = (1 - vb) * eff

    # Molar balances dFdz - Tube (0 to 3) & Shell (4 to 7)
    dF0 = -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF1 = 1 / 2 * Cat * r0 * At - Cat * r1 * At - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF2 = Cat * r0 * At + Cat * r1 * At- (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF3 = (1 / 3) * Cat * r1 * At - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    dF4 = (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF5 = (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF6 = (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF7 = (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    
    dFdz = jnp.array([ dF0, dF1, dF2, dF3, dF4, dF5, dF6, dF7 ])

    return dFdz



def dma_mr_design(u):
    

    L =  u[0]
    dt = u[1]

    # Initial conditions
    y0 = jnp.hstack((Ft0, jnp.zeros(7)))
    rtol, atol = 1e-10, 1e-10

    z = jnp.linspace(0, L, 2000)
    F = odeint(dma_mr_jax, y0, z, dt, rtol=rtol, atol=atol)

    
    F_C6H6 = ((F[-1, 3] * 1000) * MM_B)
    X_CH4  = (100 * (Ft0 - F[-1, 0] - F[-1, 4]) / Ft0)

    return jnp.array([F_C6H6, X_CH4])


def dma_mr_mvs(u):
    
    
    v0  = u[0]          # Vol. Flowrate [cm³ h-¹]
    v_He = u[1]         # Vol. flowrate[cm³/h]
    
    #Kinetic  and general parameters
    R = 8.314e+6     #[Pa.cm³/(K.mol.)]
    k1 = 0.04        #[s-¹]
    k1_Inv = 6.40e+6 #[cm³/s-mol]
    k2 = 4.20        #[s-¹]
    k2_Inv = 56.38   #[cm³/s-mol]
    
    # Molecular weights
    MM_B = 78.00     #[g/mol] 
    MM_H = 2.00      #[g/mol]
    
    L  = 17.00              # Tube length [cm]
    dt = 0.56             # Tube diameter [cm]
    # Fixed Reactor Values 
    T = 1173.15               # Temperature[K]  =900[°C] (Isothermal)
    Q = 3600*0.01e-4          # [mol/(h.cm².atm1/4)]
    selec = 1500              # Selectivity
    
    # Tube side   
    Pt = 101325               # Pressure [Pa](1atm)
    
    At = 0.25*np.pi*(dt**2)   # Cross sectional area[cm³]
    Ft0 = Pt*v0/(R*T)         # Initial molar flowrate[mol/h] - Pure CH4
    
    #Shell side 
    Ps = 101325               # Pressure [Pa](1atm)
    ds = 3                    # Diameter[cm]
    
    F_He = Ps*v_He/(R*T)      # Sweep gas molar flowrate [mol/h]
    
    z = np.asarray([0, L])
    
    
    initialflows = np.zeros(7)
    y0 = np.hstack((Ft0, initialflows))
    
    
    sol = spint.solve_ivp(dma_mr, z, y0 ,args=(k1,k1_Inv,k2,k2_Inv,T,Q,Pt,v0,At,
                                         Ft0,Ps,v_He,F_He,dt,selec))
    y = np.zeros(2)
    
    F = sol.y.T
    y[0]= (F[-1,3]*1000)*MM_B             #F_C6H6
    y[1]= 100*(Ft0-F[-1,0]-F[-1,4])/Ft0   #X_CH4
    
    return y

def dma_mr_mv(u):
    
    
    v0  = u[0]          # Vol. Flowrate [cm³ h-¹]
    v_He = u[1]         # Vol. flowrate[cm³/h]
    
    #Kinetic  and general parameters
    R = 8.314e+6     #[Pa.cm³/(K.mol.)]
    k1 = 0.04        #[s-¹]
    k1_Inv = 6.40e+6 #[cm³/s-mol]
    k2 = 4.20        #[s-¹]
    k2_Inv = 56.38   #[cm³/s-mol]
    
    # Molecular weights
    MM_B = 78.00     #[g/mol] 
    MM_H = 2.00      #[g/mol]
    
    L  = 30.00              # Tube length [cm]
    dt = 1.00               # Tube diameter [cm]
    # Fixed Reactor Values 
    T = 1173.15               # Temperature[K]  =900[°C] (Isothermal)
    Q = 3600*0.01e-4          # [mol/(h.cm².atm1/4)]
    selec = 1500              # Selectivity
    
    # Tube side   
    Pt = 101325               # Pressure [Pa](1atm)
    
    At = 0.25*np.pi*(dt**2)   # Cross sectional area[cm³]
    Ft0 = Pt*v0/(R*T)         # Initial molar flowrate[mol/h] - Pure CH4
    
    #Shell side 
    Ps = 101325               # Pressure [Pa](1atm)
    ds = 3                    # Diameter[cm]
    
    F_He = Ps*v_He/(R*T)      # Sweep gas molar flowrate [mol/h]
    
    z = np.asarray([0, L])
    
    initialflows = np.zeros(7)
    y0 = np.hstack((Ft0, initialflows))
    
    sol = spint.solve_ivp(dma_mr, z, y0 ,args=(k1,k1_Inv,k2,k2_Inv,T,Q,Pt,v0,At,
                                         Ft0,Ps,v_He,F_He,dt,selec))
    y = np.zeros(2)
    
    F = sol.y.T
    y[0]= (F[-1,3]*1000)*MM_B             #F_C6H6
    y[1]= 100*(Ft0-F[-1,0]-F[-1,4])/Ft0   #X_CH4
    
    return y



def dma_mr_QS(F, z, Q,selec):
    
    dt = 0.5634
    At = 0.25 * jnp.pi * (dt ** 2)  # Cross sectional area[cm³]
    
    # Avoid negative flows that can happen in the first integration steps.
    # Consequently this avoids that any molar balance (^ 1/4 terms) generates
    # complex numbers.
    F = jnp.where(F <= 1e-9, 1e-9, F)
    

    
    # Evaluate total flowrate in tube & shell.
    Ft = F[0:4].sum()
    Fs = F[4:].sum() + F_He
    v = v0 * (Ft / Ft0)

    # Concentrations from molar flowrates [mol/cm3]
    C = F[:4] / v
    # Partial pressures - Tube & Shell [mol/cm3]

    P0t = (Pt / 101325) * (F[0] / Ft)
    P1t = (Pt / 101325) * (F[1] / Ft)
    P2t = (Pt / 101325) * (F[2] / Ft)
    P3t = (Pt / 101325) * (F[3] / Ft)

    P0s = (Ps / 101325) * (F[4] / Fs)
    P1s = (Ps / 101325) * (F[5] / Fs)
    P2s = (Ps / 101325) * (F[6] / Fs)
    P3s = (Ps / 101325) * (F[7] / Fs)
    
    

    
    r0 = 3600 * k1 * C[0] * (1 - ((k1_Inv * C[1] * C[2] ** 2) / 
                                  (k1 * (C[0])**2 )))
    

    # This replicates an if statement whenever the concentrations are near zero
    C0_aux = C[0]
    r0 = jnp.where(C0_aux <= 1e-9, 0, r0)
    
    r1 = 3600 * k2 * C[1] * (1 - ((k2_Inv * C[3] * C[2] ** 3) / 
                                  (k2 * (C[1])**3 )))
    

    # Same as before
    C1_aux = C[1]
    r1 = jnp.where(C1_aux <= 1e-9 , 0, r1)  

    # Molar balances adjustment with experimental data.
    eff = 0.9
    vb = 0.5
    Cat = (1 - vb) * eff

    # Molar balances dFdz - Tube (0 to 3) & Shell (4 to 7)
    dF0 = -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF1 = 1 / 2 * Cat * r0 * At - Cat * r1 * At - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF2 = Cat * r0 * At + Cat * r1 * At- (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF3 = (1 / 3) * Cat * r1 * At - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    dF4 = (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt
    dF5 = (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt
    dF6 = (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
    dF7 = (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
    
    dFdz = jnp.array([ dF0, dF1, dF2, dF3, dF4, dF5, dF6, dF7 ])

    return dFdz

def dma_mr_uncertain(u):
    
    
    
    
    Q = u[0]
    selec =  u[1]
    

    # Initial conditions
    y0 = jnp.hstack((Ft0, jnp.zeros(7)))
    rtol, atol = 1e-10, 1e-10

    z = jnp.linspace(0, 17.2283, 2000)
    F = odeint(dma_mr_QS, y0, z, Q, selec, rtol=rtol, atol=atol)

    
    F_C6H6 = ((F[-1, 3] * 1000) * MM_B)
    X_CH4  = (100 * (Ft0 - F[-1, 0] - F[-1, 4]) / Ft0)

    return jnp.array([F_C6H6, X_CH4])

# Kinetic  and general parameters

R = 8.314e6                 # [Pa.cm³/(K.mol.)]
k1 = 0.04                   # [s-¹]
k1_Inv = 6.40e6             # [cm³/s-mol]
k2 = 4.20                   # [s-¹]
k2_Inv = 56.38              # [cm³/s-mol]

    
# Molecular weights
MM_B = 78.00     #[g/mol] 

# Fixed Reactor Values
T = 1173.15                 # Temperature[K]  =900[°C] (Isothermal)
Q = 3600 * 0.01e-4          # [mol/(h.cm².atm1/4)]
selec = 1500

# Tube side
Pt = 101325.0               # Pressure [Pa](1atm)
v0 = 3600 * (2 / 15)        # Vol. Flowrate [cm³ h-¹]
Ft0 = Pt * v0 / (R * T)     # Initial molar flowrate[mol/h] - Pure CH4

# Shell side
Ps = 101325.0               # Pressure [Pa](1atm)
ds = 3                      # Diameter[cm]
v_He = 3600 * (1 / 6)       # Vol. flowrate[cm³/h]
F_He = Ps * v_He / (R * T)  # Sweep gas molar flowrate [mol/h]


# ============================================================
# Countercurrent DMA-MR (from San Dinh's model)
# Pyomo/IPOPT implementation in dm units
# ============================================================

def dma_mr_counter_ss(u):
    """
    Steady-state countercurrent DMA-MR model (San Dinh's formulation).

    Uses Pyomo + IPOPT to solve the spatial BVP with 11 nodes (dz=3 dm,
    L=30 dm). Units: dm for lengths, dm^3/h for flowrates, mol/h for
    molar flows.

    Parameters
    ----------
    u : array-like, shape (2,)
        u[0] = Qtube -- tube inlet volumetric flowrate (dm^3/h)
        u[1] = Qshell -- shell inlet volumetric flowrate (dm^3/h)

    Returns
    -------
    y : ndarray, shape (2,)
        y[0] = benzene mole percent in tube outlet
        y[1] = H2 mole percent in shell outlet
    """
    import pyomo.environ as pyo
    from pyomo.environ import (ConcreteModel, Var, Constraint, Objective,
                               SolverFactory, value)
    import logging
    logging.getLogger('pyomo').setLevel(logging.ERROR)

    _Species = ['CH4', 'C2H4', 'H2', 'C6H6', 'inert']
    _dz = 3.0
    _z = np.arange(0, 11, 1)  # 11 nodes: 0..10

    _Ptube = 1.0; _Pshell = 1.0
    _Ctube = 0.010387973185676  # mol/dm^3 = P/(R*T) at 1173 K
    _Cshell = _Ctube
    _Dt = 1.5       # tube diameter (dm)
    _Ds = 0.2       # shell thickness (dm)
    _Vtube = (((_Dt / 2) ** 2) * pi * _dz)

    _vb, _eff = 0.5, 0.9
    _k1, _k2 = 144.0, 15120.0        # h^-1
    _k1eq = 6.25e-6
    _k2eq = 74.494
    _Qm = 0.3       # membrane permeability (mol/(h.dm^2.atm^{1/4}))
    _sel = 500.0     # selectivity

    Q_tube_in = float(u[0])
    Q_shell_in = float(u[1])

    MR = ConcreteModel()

    # Tube mole fractions
    MR.xtube = Var(_Species, _z, within=pyo.NonNegativeReals, initialize=0.1)
    MR.xtube['CH4', 0].fix(1.0)
    MR.xtube['C2H4', 0].fix(0.0)
    MR.xtube['H2', 0].fix(0.0)
    MR.xtube['C6H6', 0].fix(0.0)
    MR.xtube['inert', 0].fix(0.0)

    # Shell mole fractions (countercurrent: inlet at z=L)
    MR.xshell = Var(_Species, _z, within=pyo.NonNegativeReals, initialize=0.1)
    MR.xshell['CH4', _z[-1]].fix(0.0)
    MR.xshell['C2H4', _z[-1]].fix(0.0)
    MR.xshell['H2', _z[-1]].fix(0.0)
    MR.xshell['C6H6', _z[-1]].fix(0.0)
    MR.xshell['inert', _z[-1]].fix(1.0)

    # Total molar flows
    MR.Ftube = Var(_z, within=pyo.NonNegativeReals, initialize=100)
    MR.Ftube[0].fix(Q_tube_in * _Ctube)
    MR.Fshell = Var(_z, within=pyo.NonNegativeReals, initialize=100)
    MR.Fshell[_z[-1]].fix(Q_shell_in * _Cshell)

    # Tube mole fraction sum = 1
    def _tf(m, zk):
        if zk == 0:
            return Constraint.Skip
        return sum(m.xtube[s, zk] for s in _Species) == 1
    MR.tube_frac = Constraint(_z, rule=_tf)

    # Tube mass balance
    def _tmb(m, sp, zk):
        if zk == 0:
            return Constraint.Skip
        F_in = m.Ftube[zk - 1] * m.xtube[sp, zk - 1]
        F_out = m.Ftube[zk] * m.xtube[sp, zk]
        Ct0 = _Ctube * m.xtube['CH4', zk]
        Ct1 = _Ctube * m.xtube['C2H4', zk]
        Ct2 = _Ctube * m.xtube['H2', zk]
        Ct3 = _Ctube * m.xtube['C6H6', zk]
        _Cat = (1 - _vb) * _eff
        r1 = -_Cat * _k1 * Ct0 * (
            _k1eq * Ct0**2 - Ct1 * Ct2**2) / (_k1eq * Ct0**2)
        r2 = -_Cat * _k2 * Ct1 * (
            _k2eq * Ct1**3 - Ct3 * Ct2**3) / (_k2eq * Ct1**3)
        gmap = {'CH4': r1, 'C2H4': -r1/2 + r2, 'H2': -r1/2 - r2,
                'C6H6': -r2/3, 'inert': 0}
        generation = gmap[sp] * _Vtube
        s_f = 1.0 if sp == 'H2' else 1.0 / _sel
        diffusion = _Qm * s_f * (
            (_Ptube * m.xtube[sp, zk])**0.25
            - (_Pshell * m.xshell[sp, zk - 1])**0.25
        ) * pi * _Dt * _dz
        return 0 == F_in - F_out + generation - diffusion
    MR.tube_mb = Constraint(_Species, _z, rule=_tmb)

    # Shell mole fraction sum = 1
    def _sf(m, zk):
        if zk == _z[-1]:
            return Constraint.Skip
        return sum(m.xshell[s, zk] for s in _Species) == 1
    MR.shell_frac = Constraint(_z, rule=_sf)

    # Shell mass balance (countercurrent: F_in from higher z)
    def _smb(m, sp, zk):
        if zk == 0:
            return Constraint.Skip
        F_in = m.Fshell[zk] * m.xshell[sp, zk]
        F_out = m.Fshell[zk - 1] * m.xshell[sp, zk - 1]
        s_f = 1.0 if sp == 'H2' else 1.0 / _sel
        diffusion = _Qm * s_f * (
            (_Ptube * m.xtube[sp, zk])**0.25
            - (_Pshell * m.xshell[sp, zk - 1])**0.25
        ) * pi * _Dt * _dz
        return 0 == F_in - F_out + diffusion
    MR.shell_mb = Constraint(_Species, _z, rule=_smb)

    MR.obj = Objective(rule=Objective.Skip)
    solver = SolverFactory("ipopt")
    solver.options['print_level'] = 0
    solver.solve(MR, tee=False)

    benz_pct = value(MR.xtube['C6H6', _z[-1]]) * 100
    h2_pct = value(MR.xshell['H2', 0]) * 100

    return np.array([benz_pct, h2_pct])


# --------------------------------------------------------------------------- #
# Co-current dynamic DMA-MR model (Dinh and Lima, Ind. Eng. Chem. Res., 2023).
#
# Co-current shell-and-tube membrane reactor: the tube feed and the shell sweep
# gas both enter at z = 0 and flow in the same direction to z = L. The state is
# the spatial profile of molar holdups: n_elements spatial elements times 8
# species (4 tube and 4 shell), stored as a flat row-major vector. The
# achievable-output funnel is produced by stepping this transient model forward
# in time. Units: dm for lengths, dm^3/h for volumetric flowrates, mol for
# holdups. The outputs are the benzene mole percent at the tube outlet and the
# H2 mole percent at the shell outlet.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DMAMRParameters:
    """
    Physical and numerical parameters of the co-current dynamic DMA-MR model.

    The defaults reproduce the case study of Dinh and Lima (Ind. Eng. Chem.
    Res., 2023). Lengths are in dm, volumetric flowrates in dm^3/h and molar
    holdups in mol. Pass a customized instance to any of the model functions
    through their ``params`` argument to study a different reactor.

    Attributes
    ----------
    pressure : float
        Operating pressure [atm].
    total_concentration : float
        Total molar concentration P/(R T) [mol/dm^3].
    tube_diameter : float
        Tube diameter [dm].
    tube_volume, shell_volume : float
        Per-element tube and shell volumes [dm^3].
    element_length : float
        Axial length of one spatial element [dm]; the reactor length is
        ``element_length * n_elements``.
    bed_voidage : float
        Catalyst bed voidage (void fraction).
    catalyst_efficiency : float
        Catalyst effectiveness factor.
    rate_constant_1, rate_constant_2 : float
        Forward rate constants of the two reactions [1/h].
    equilibrium_1, equilibrium_2 : float
        Equilibrium constants of the two reactions.
    permeance : float
        Membrane permeance, applied internally as ``permeance * 36``.
    selectivity : float
        Membrane selectivity of H2 over the other species.
    n_elements : int
        Number of spatial discretization elements.
    timescale : float
        Time scaling so that integrating the transient model over [0, 1]
        advances the reactor by one minute.
    feed_CH4, feed_C2H4, feed_inert : float
        Nominal feed mole fractions of methane, ethylene and inert.
    """

    pressure: float = 1.0
    total_concentration: float = 0.010387973185676
    tube_diameter: float = 1.5
    tube_volume: float = 4.417864669110937
    shell_volume: float = 1.943860454408813
    element_length: float = 2.5
    bed_voidage: float = 0.5
    catalyst_efficiency: float = 0.9
    rate_constant_1: float = 144.0
    rate_constant_2: float = 15120.0
    equilibrium_1: float = 6.25e-06
    equilibrium_2: float = 74.494
    permeance: float = 0.01
    selectivity: float = 100.0
    n_elements: int = 20
    timescale: float = 1.0 / 60.0
    feed_CH4: float = 0.942
    feed_C2H4: float = 0.023
    feed_inert: float = 0.035


DMA_MR_PARAMS = DMAMRParameters()


def feed_fractions(d: float | None,
                   params: DMAMRParameters) -> tuple[float, float, float]:
    """
    Map a disturbance ``d`` (methane feed mole percent) to the feed mole
    fractions (CH4, C2H4, inert).

    Ethylene is held at its nominal value and the inert takes up the balance.
    ``d`` may be given as a fraction (<= 1) or as a percent (> 1); ``None``
    returns the nominal feed.

    Parameters
    ----------
    d : float or None
        Methane feed mole percent (or fraction). ``None`` selects the nominal
        feed defined in ``params``.
    params : DMAMRParameters
        Model parameters holding the nominal feed composition.

    Returns
    -------
    tuple of float
        The mole fractions ``(yCH4, yC2H4, yinert)``.
    """
    if d is None:
        return params.feed_CH4, params.feed_C2H4, params.feed_inert
    yCH4 = d / 100.0 if d > 1.0 else float(d)
    return yCH4, params.feed_C2H4, 1.0 - yCH4 - params.feed_C2H4


def steady_state_rhs(z: float, F: np.ndarray, Qt: float, Qs: float,
                     yCH4: float, yC2H4: float, yinert: float,
                     params: DMAMRParameters) -> list:
    """
    Spatial steady-state molar-flow derivative dF/dz used to build the initial
    holdup profile of the reactor.

    This is the right-hand side integrated by ``scipy.integrate.solve_ivp``
    along the reactor length to obtain the steady-state spatial profile of the
    eight molar flows (four tube and four shell species).

    Parameters
    ----------
    z : float
        Axial position [dm] (required by the solver, not used explicitly).
    F : numpy.ndarray
        The eight molar flows ``[tube CH4, C2H4, H2, C6H6, shell CH4, C2H4,
        H2, C6H6]`` [mol/h].
    Qt, Qs : float
        Tube and shell inlet volumetric flowrates [dm^3/h].
    yCH4, yC2H4, yinert : float
        Feed mole fractions (see :func:`feed_fractions`).
    params : DMAMRParameters
        Model parameters.

    Returns
    -------
    list
        The derivative dF/dz of the eight molar flows.
    """
    def quarter_root(pressure):
        # Guarded fourth root: the membrane driving force is P ** 0.25, and the
        # solver can momentarily probe slightly negative pressures, for which a
        # plain power would return NaN. Clipping keeps the integration real.
        return np.power(np.clip(pressure, 1e-12, None), 0.25)

    p = params

    # Unpack the eight molar flows (4 tube, 4 shell).
    FtCH4, FtC2H4, FtH2, FtC6H6, FsCH4, FsC2H4, FsH2, FsC6H6 = F

    # Tube cross-sectional area and total molar flows (the inert is the tube
    # tie component, the sweep gas the shell tie component).
    At = p.tube_diameter ** 2 / 4 * pi
    Ftinert = Qt * p.total_concentration * yinert
    Fshell0 = Qs * p.total_concentration
    Ftube = FtCH4 + FtC2H4 + FtH2 + FtC6H6 + Ftinert
    Fshell = FsCH4 + FsC2H4 + FsH2 + FsC6H6 + Fshell0

    # Tube concentrations and tube/shell partial pressures.
    CtCH4 = FtCH4 * p.total_concentration / Ftube
    CtC2H4 = FtC2H4 * p.total_concentration / Ftube
    CtH2 = FtH2 * p.total_concentration / Ftube
    CtC6H6 = FtC6H6 * p.total_concentration / Ftube
    Pt = (p.pressure * FtCH4 / Ftube, p.pressure * FtC2H4 / Ftube,
          p.pressure * FtH2 / Ftube, p.pressure * FtC6H6 / Ftube)
    Ps = (p.pressure * FsCH4 / Fshell, p.pressure * FsC2H4 / Fshell,
          p.pressure * FsH2 / Fshell, p.pressure * FsC6H6 / Fshell)

    # Membrane permeation from tube to shell. H2 permeates freely; the other
    # species are slowed by the membrane selectivity.
    perm_other = p.permeance * 36 / p.selectivity
    perm_H2 = p.permeance * 36
    T2S_CH4 = perm_other * pi * p.tube_diameter * (quarter_root(Pt[0]) - quarter_root(Ps[0]))
    T2S_C2H4 = perm_other * pi * p.tube_diameter * (quarter_root(Pt[1]) - quarter_root(Ps[1]))
    T2S_H2 = perm_H2 * pi * p.tube_diameter * (quarter_root(Pt[2]) - quarter_root(Ps[2]))
    T2S_C6H6 = perm_other * pi * p.tube_diameter * (quarter_root(Pt[3]) - quarter_root(Ps[3]))

    # Equilibrium-limited rates of the two reactions.
    r1 = -(1 - p.bed_voidage) * p.catalyst_efficiency * p.rate_constant_1 * CtCH4 * (
        (p.equilibrium_1 * CtCH4 ** 2) - CtC2H4 * CtH2 ** 2) / (p.equilibrium_1 * CtCH4 ** 2)
    r2 = -(1 - p.bed_voidage) * p.catalyst_efficiency * p.rate_constant_2 * CtC2H4 * (
        (p.equilibrium_2 * CtC2H4 ** 3) - CtC6H6 * CtH2 ** 3) / (p.equilibrium_2 * CtC2H4 ** 3)

    # dF/dz: four tube species (reaction generation minus permeation out) then
    # four shell species (permeation in).
    return [r1 * At - T2S_CH4,
            (-r1 / 2 + r2) * At - T2S_C2H4,
            (-r1 / 2 - r2) * At - T2S_H2,
            (-r2 / 3) * At - T2S_C6H6,
            T2S_CH4, T2S_C2H4, T2S_H2, T2S_C6H6]


def state_space_rhs(t: float, M: np.ndarray, Qt: float, Qs: float,
                    yCH4: float, yC2H4: float, yinert: float,
                    params: DMAMRParameters) -> np.ndarray:
    """
    Transient state-space derivative dM/dt for the molar holdups, vectorized
    over the spatial elements.

    This is the right-hand side integrated by ``scipy.integrate.solve_ivp`` to
    advance the reactor in time. Integrating it over one unit of time advances
    the reactor by one minute (see ``DMAMRParameters.timescale``).

    Parameters
    ----------
    t : float
        Scaled time (required by the solver, not used explicitly).
    M : numpy.ndarray
        Flat holdup state of length ``8 * n_elements`` [mol].
    Qt, Qs : float
        Tube and shell inlet volumetric flowrates [dm^3/h].
    yCH4, yC2H4, yinert : float
        Feed mole fractions (see :func:`feed_fractions`).
    params : DMAMRParameters
        Model parameters.

    Returns
    -------
    numpy.ndarray
        The flat derivative dM/dt, same length as ``M``.
    """
    def quarter_root(pressure):
        # Guarded fourth root (see steady_state_rhs); here applied to the
        # whole (n_elements, 4) partial-pressure arrays at once.
        return np.power(np.clip(pressure, 1e-12, None), 0.25)

    p = params
    Ct = p.total_concentration

    # Inlet molar flows of the tube feed and the shell sweep gas.
    Ft0 = Qt * Ct
    Fs0 = Qs * Ct
    Ftinert = Ft0 * yinert
    feed = np.array([Ft0 * yCH4, Ft0 * yC2H4, 0.0, 0.0])

    # Reshape the flat state into (element, species); split tube and shell.
    X = np.asarray(M).reshape(p.n_elements, 8)
    Mt, Ms = X[:, 0:4], X[:, 4:8]

    # Convective molar outflow of each element, with the non-permeating,
    # non-reacting inert/sweep acting as the tie component (fixed flow).
    denom_t = Mt.sum(axis=1) - Ct * p.tube_volume
    denom_s = Ms.sum(axis=1) - Ct * p.shell_volume
    Ft = -(Ftinert * Mt) / denom_t[:, None]
    Fs = -(Fs0 * Ms) / denom_s[:, None]

    # Tube concentrations and tube/shell partial pressures.
    Ct_sp = Mt / p.tube_volume
    Pt = Mt * p.pressure / (Ct * p.tube_volume)
    Ps = Ms * p.pressure / (Ct * p.shell_volume)

    # Membrane permeation from tube to shell (H2 unscaled, others divided by
    # the selectivity).
    sel = np.array([p.selectivity, p.selectivity, 1.0, p.selectivity])
    coef = (p.permeance * 36.0 / sel) * pi * p.tube_diameter * p.element_length
    T2S = coef * (quarter_root(Pt) - quarter_root(Ps))

    # Equilibrium-limited reaction rates per element.
    CtCH4, CtC2H4, CtH2, CtC6H6 = Ct_sp[:, 0], Ct_sp[:, 1], Ct_sp[:, 2], Ct_sp[:, 3]
    r1 = -(1 - p.bed_voidage) * p.catalyst_efficiency * p.rate_constant_1 * CtCH4 * (
        (p.equilibrium_1 * CtCH4 ** 2) - CtC2H4 * CtH2 ** 2) / (p.equilibrium_1 * CtCH4 ** 2)
    r2 = -(1 - p.bed_voidage) * p.catalyst_efficiency * p.rate_constant_2 * CtC2H4 * (
        (p.equilibrium_2 * CtC2H4 ** 3) - CtC6H6 * CtH2 ** 3) / (p.equilibrium_2 * CtC2H4 ** 3)

    # Net reaction generation of each tube species, scaled by the element
    # volume (stoichiometry of 2 CH4 -> C2H4 + 2 H2 and 3 C2H4 -> C6H6 + 3 H2).
    gen = np.empty((p.n_elements, 4))
    gen[:, 0] = r1
    gen[:, 1] = -r1 / 2 + r2
    gen[:, 2] = -r1 - r2
    gen[:, 3] = -r2 / 3
    gen *= p.tube_volume

    # Inflow to each element from its upstream neighbor (co-current): element 0
    # of the tube is fed by the inlet, element 0 of the shell has no upstream.
    Ft_in = np.vstack([feed, Ft[:-1, :]])
    Fs_in = np.vstack([np.zeros(4), Fs[:-1, :]])

    # Holdup balances: tube (convection + reaction - permeation out) and shell
    # (convection + permeation in), scaled to a per-minute time base.
    dM = np.empty((p.n_elements, 8))
    dM[:, 0:4] = (Ft_in - Ft + gen - T2S) * p.timescale
    dM[:, 4:8] = (Fs_in - Fs + T2S) * p.timescale
    return dM.reshape(-1)


def dma_mr_cocurrent_outputs(M: np.ndarray,
                             params: DMAMRParameters = DMA_MR_PARAMS) -> np.ndarray:
    """
    Reactor outputs from the holdup state ``M``.

    Returns ``[benzene mole % at the tube outlet, H2 mole % at the shell
    outlet]``, both evaluated at the reactor end (z = L), which is the last
    spatial element of the flat state.
    """
    M = np.asarray(M)
    p = params

    # The last element block is M[-8:]: its tube C6H6 is M[-5] and its shell H2
    # is M[-2]. Convert those holdups to mole percent.
    benzene_pct = 100.0 * M[-5] / (p.total_concentration * p.tube_volume)
    hydrogen_pct = 100.0 * M[-2] / (p.total_concentration * p.shell_volume)
    return np.array([benzene_pct, hydrogen_pct])


def dma_mr_cocurrent_x0(u: np.ndarray, d: float | None = None,
                        n_relax: int = 50,
                        params: DMAMRParameters = DMA_MR_PARAMS) -> np.ndarray:
    """
    Steady-state holdup state at the given operating point.

    A spatial-profile guess is first built by integrating the steady-state
    molar balances along the reactor, and then relaxed to steady state with the
    transient model (the reactor is Lyapunov-stable, so time-marching settles
    onto the steady state).

    Parameters
    ----------
    u : array-like, shape (2,)
        Inputs ``[tube flowrate, shell flowrate]`` in dm^3/h.
    d : float, optional
        Disturbance: methane feed mole percent. ``None`` uses the nominal feed.
    n_relax : int, optional
        Number of one-minute relaxation steps. The default is 50.
    params : DMAMRParameters, optional
        Model parameters.

    Returns
    -------
    numpy.ndarray
        Flat holdup state of length ``8 * n_elements``.
    """
    p = params
    yCH4, yC2H4, yinert = feed_fractions(d, p)
    Qt, Qs = float(u[0]), float(u[1])

    # Inlet molar flows and the spatial grid along the reactor length.
    Ft0 = Qt * p.total_concentration
    Ftinert = Ft0 * yinert
    Fs0 = Qs * p.total_concentration
    F0 = [Ft0 * yCH4, Ft0 * yC2H4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Lgrid = np.linspace(0, p.element_length * p.n_elements, p.n_elements + 1)

    # Integrate the steady-state molar balances along z for a profile guess.
    sol = spint.solve_ivp(steady_state_rhs, [0, Lgrid[-1]], F0, t_eval=Lgrid,
                          args=(Qt, Qs, yCH4, yC2H4, yinert, p),
                          rtol=1e-7, atol=1e-7, method='LSODA')
    Fss = sol.y.T

    # Convert the molar-flow profile to holdups (mole fractions times the
    # element holdup Ct * V), dropping the inlet node.
    M = np.zeros((Fss.shape[0], 8))
    st = Fss[:, 0:4].sum(axis=1) + Ftinert
    M[:, 0:4] = p.total_concentration * p.tube_volume * Fss[:, 0:4] / st[:, None]
    ss = Fss[:, 4:8].sum(axis=1) + Fs0
    M[:, 4:8] = p.total_concentration * p.shell_volume * Fss[:, 4:8] / ss[:, None]
    M = M[1:, :].reshape(-1)

    # Relax the transient model to steady state.
    for _ in range(n_relax):
        sol = spint.solve_ivp(state_space_rhs, [0, 1], M,
                              args=(Qt, Qs, yCH4, yC2H4, yinert, p),
                              rtol=1e-8, atol=1e-8, method='LSODA')
        M = sol.y[:, -1]
    return M


def dma_mr_cocurrent_step(M: np.ndarray, u: np.ndarray, d: float | None = None,
                          dt: float = 1.0,
                          params: DMAMRParameters = DMA_MR_PARAMS
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance the transient co-current DMA-MR by ``dt`` minutes.

    Integrates the transient model from holdup state ``M`` with inputs
    ``u = [tube flowrate, shell flowrate]`` held constant and disturbance ``d``
    (methane feed mole percent). Returns ``(M_next, y)``, the step contract
    expected by ``dynamic_operability``.
    """
    p = params
    yCH4, yC2H4, yinert = feed_fractions(d, p)
    Qt, Qs = float(u[0]), float(u[1])

    # Integrate the transient balances over the requested time interval.
    sol = spint.solve_ivp(state_space_rhs, [0, dt], np.asarray(M, dtype=float),
                          args=(Qt, Qs, yCH4, yC2H4, yinert, p),
                          rtol=1e-6, atol=1e-9, method='LSODA')
    M_next = sol.y[:, -1]
    return M_next, dma_mr_cocurrent_outputs(M_next, p)
