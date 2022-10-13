import numpy as np
import scipy.integrate as spint
from dma_mr import dma_mr

def dma_mr_2x2(u):
    #Kinetic  and general parameters
    R = 8.314e+6     #[Pa.cm³/(K.mol.)]
    k1 = 0.04        #[s-¹]
    k1_Inv = 6.40e+6 #[cm³/s-mol]
    k2 = 4.20        #[s-¹]
    k2_Inv = 56.38   #[cm³/s-mol]
    
    # Molecular weights
    MM_B = 78.00     #[g/mol] 
    MM_H = 2.00      #[g/mol]
    
    L  = u[0]              # Tube length [cm]
    dt = u[1]               # Tube diameter [cm]
    # Fixed Reactor Values 
    T = 1173.15               # Temperature[K]  =900[°C] (Isothermal)
    Q = 3600*0.01e-4          # [mol/(h.cm².atm1/4)]
    selec = 1500              # Selectivity
    
    # Tube side   
    Pt = 101325               # Pressure [Pa](1atm)
    v0 = 3600*(2/15)          # Vol. Flowrate [cm³ h-¹]
    At = 0.25*np.pi*(dt**2)   # Cross sectional area[cm³]
    Ft0 = Pt*v0/(R*T)         # Initial molar flowrate[mol/h] - Pure CH4
    
    #Shell side 
    Ps = 101325               # Pressure [Pa](1atm)
    ds = 3                    # Diameter[cm]
    v_He = 3600*(1/6)         # Vol. flowrate[cm³/h]
    F_He = Ps*v_He/(R*T)      # Sweep gas molar flowrate [mol/h]
    
    z = np.asarray([0, L])
    
    rtol, atol = 1e-8, 1e-8
    
    initialflows = np.zeros(7)
    y0 = np.hstack((Ft0, initialflows))
    
    sol = spint.solve_ivp(dma_mr, z, y0 ,args=(k1,k1_Inv,k2,k2_Inv,T,Q,Pt,v0,At,
                                         Ft0,Ps,v_He,F_He,dt,selec),method='RK45',
                    rtol=rtol,atol=atol)
    y = np.zeros(2)
    
    F = sol.y.T
    y[0]= (F[-1,3]*1000)*MM_B             #F_C6H6
    y[1]= 100*(Ft0-F[-1,0]-F[-1,4])/Ft0   #X_CH4
    
    return y



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
    
    L  = 30              # Tube length [cm]
    dt = 1               # Tube diameter [cm]
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
    
    rtol, atol = 1e-8, 1e-8
    
    initialflows = np.zeros(7)
    y0 = np.hstack((Ft0, initialflows))
    
    sol = spint.solve_ivp(dma_mr, z, y0 ,args=(k1,k1_Inv,k2,k2_Inv,T,Q,Pt,v0,At,
                                         Ft0,Ps,v_He,F_He,dt,selec),method='RK45',
                    rtol=rtol,atol=atol)
    y = np.zeros(2)
    
    F = sol.y.T
    y[0]= (F[-1,3]*1000)*MM_B             #F_C6H6
    y[1]= 100*(Ft0-F[-1,0]-F[-1,4])/Ft0   #X_CH4
    
    return y