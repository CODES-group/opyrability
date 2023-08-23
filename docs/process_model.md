---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


## The Process Model (M) - Programming Aspects 

Let's recapitulate how the process model (M) is mathematically defined:

```{eval-rst}
.. math::
	M=\left\{\begin{array}{l}
	\dot{x}_s=f\left(x_s, u, d\right) \\
	y=g\left(x_s, u, d\right) \\
	h_1\left(\dot{x}_s, x_s, y, \dot{u}, u, d\right)=0 \\
	h_2\left(\dot{x}_s, x_s, y, \dot{u}, u, d\right) \geq 0
	\end{array}\right.
```
This might look straightforward in mathematical terms. However, when coding
your process model some questions regarding implementation in Python might arise.
This section seeks to illustrate different cases that you might need to deal with
when creating your process model for performing a process operability analysis
using opyrability.

opyrability is *model agnostic*. This means that irrespective of how you want to define
your process model, as long as you follow a simple syntax in terms of a Python
function, **opyrability will work**.

The following pseudocode illustrates the overall syntax that must be followed
in order to be opyrability-compatible:


```{code-cell}
    def process_model(u, d):

        # u is a vector with the AIS variables.
        # d is a vector with the EDS variables.

        # the AOS variables y are a function (f)
        # of the AIS(u) and EDS(d):
        
        y = f(u,d)
            
        return y
```

### Process Model Defined as a Set of Algebraic Equations

This might be the simplest case, one example being the [Shower Problem](examples_gallery/operability_index_shower.ipynb){cite}`vinson00, lima10b`. The model equations 
are explicit and algebraic, being solved directly. The process model Python function
would be, in this case:


```{code-cell}
    def shower_problem(u):
        y = np.zeros(2)
        y[0]=u[0]+u[1]
        if y[0]!=0:
            y[1]=(u[0]*60+u[1]*120)/(u[0]+u[1])
        else:
            y[1]=(60+120)/2
            
        return y
```

### Process Model Defined as a Set of Ordinary Differential (ODE) Equations:

A more complex case is when the process model is defined as a system of equations.
These can be nonlinear, algebraic and/or differential that need to be numerically
integrated. Irrespective of the increased complexity, as long as you follow the
syntax that is opyrability-compatible, the process operability calculations performed
by opyrability will work effortlessly.

One example is the 
[DMA-MR](examples_gallery/membrane_reactor.ipynb) {cite}`carrasco16, carrasco17`
in which the AIS variables are design parameters (tube length and tube diameter)
and AOS variables correspond to methane conversion and benzene production. Both
are calculated based on the calculated states of the following ODE system,
described by the `dma_mr_model` function below:

```{code-cell}

    import jax.numpy as np
    from jax.numpy import pi as pi
    from jax.experimental.ode import odeint

    # Kinetic and general parameters

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

    def dma_mr_model(F, z, dt,v_He,v0, F_He, Ft0):

        At = 0.25 * np.pi * (dt ** 2)  # Cross sectional area [cm2].
        
        # Avoid negative flows that can happen in the first integration steps.
        # Consequently this avoids that any molar balance (^ 1/4 terms) generates
        # complex numbers.
        F = np.where(F <= 1e-9, 1e-9, F)
        
        

        
        # Evaluate the total flow rate in the tube & shell.
        Ft = F[0:4].sum()
        Fs = F[4:].sum() + F_He
        v = v0 * (Ft / Ft0)

        # Concentrations from molar flow rates [mol/cm3].
        C = F[:4] / v
        # Partial pressures - Tube & Shell [mol/cm3].

        P0t = (Pt / 101325) * (F[0] / Ft)
        P1t = (Pt / 101325) * (F[1] / Ft)
        P2t = (Pt / 101325) * (F[2] / Ft)
        P3t = (Pt / 101325) * (F[3] / Ft)

        P0s = (Ps / 101325) * (F[4] / Fs)
        P1s = (Ps / 101325) * (F[5] / Fs)
        P2s = (Ps / 101325) * (F[6] / Fs)
        P3s = (Ps / 101325) * (F[7] / Fs)
        
        

        # First reaction rate.
        r0 = 3600 * k1 * C[0] * (1 - ((k1_Inv * C[1] * C[2] ** 2) / 
                                    (k1 * (C[0])**2 )))
        

        # This replicates an if statement to avoid division by zero, 
        # whenever the concentrations are near zero. JAX's syntax compatible.
        C0_aux = C[0]
        r0 = np.where(C0_aux <= 1e-9, 0, r0)
        

        # Second reaction rate.
        r1 = 3600 * k2 * C[1] * (1 - ((k2_Inv * C[3] * C[2] ** 3) / 
                                    (k2 * (C[1])**3 )))
        

        # Same as before
        C1_aux = C[1]
        r1 = np.where(C1_aux <= 1e-9 , 0, r1)  

        # Molar balances adjustment with experimental data.
        eff = 0.9
        vb = 0.5
        Cat = (1 - vb) * eff

        # Molar balances dFdz - Tube (0 to 3) & Shell (4 to 7)
        dF0 = -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt

        dF1 = 1 / 2 * Cat * r0 * At - Cat * r1 * At 
        - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt

        dF2 = Cat * r0 * At + Cat * r1 * At- (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt

        dF3 = (1 / 3) * Cat * r1 * At - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt

        dF4 = (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * pi * dt

        dF5 = (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * pi * dt

        dF6 = (Q) * ((P2t ** 0.25) - (P2s ** 0.25)) * pi * dt
        
        dF7 = (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * pi * dt
        
        dFdz = np.array([ dF0, dF1, dF2, dF3, dF4, dF5, dF6, dF7 ])

        return dFdz
```

This system needs to be numerically integrated. Afterward, the AOS variables
are obtained based on the calculated system states. 
Hence, the process model function that will be used in opyrability can be 
written in the following form:

```{code-cell}

    def dma_mr_design(u):


        L =  u[0]                   # Tube length   [cm2]
        dt = u[1]                   # Tube diameter [cm2]

        F_He = Ps * v_He / (R * T)  # Sweep gas molar flow rate [mol/h].

        # Initial conditions.
        y0 = np.hstack((Ft0, np.zeros(7)))
        rtol, atol = 1e-10, 1e-10

        # Integration of mol balances using Jax's Dormand Prince.
        z = np.linspace(0, L, 2000)
        F = odeint(dma_mr_model, y0, z, dt,v_He,v0, F_He, Ft0, rtol=rtol, atol=atol)
        
        # Calculating outputs (AOS/DOS) from states.
        F_C6H6 = ((F[-1, 3] * 1000) * MM_B)
        X_CH4  = (100 * (Ft0 - F[-1, 0] - F[-1, 4]) / Ft0)

        return np.array([F_C6H6, X_CH4])

```

In the function `dma_mr_design` above, the function input `u` is a two-dimensional vector
allocating the tube length and tube diameter values, respectively. The return
is also a two-dimensional array, allocating the benzene production and methane
conversion. Note that the system is numerically integrated using `jax.experimental.ode.odeint`.


