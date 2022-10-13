from numpy import pi as pi
import numpy as np
def dma_pfr(z,F,k1,k1_Inv,k2,k2_Inv,T,Q,Pt,v0,At,Ft0,Ps,v_He,F_He,dt,selec):
    
    
    r = np.zeros(2)
    dFdz = np.zeros(4)

    # Avoid negative flows that can happen in the first integration steps.
    # Consequently this avoids that any molar balance (^ 1/4 terms) generates
    # complex numbers.
    #F[F<0]=0
    F = np.where(F < 0, 0, F)
    
    # Evaluate total flowrate in tube & shell.
    Ft = F[0] + F[1] + F[2] + F[3]
    #Fs = F[4] + F[5] + F[6] + F[7] + F_He
    v  = v0*(Ft/Ft0)
    
    # Concentrations from molar flowrates [mol/cm3]
    C = np.array([])
    C = F[:4] / v

    # Partial pressures - Tube & Shell [mol/cm3]
    
    # P0t = ((Pt/101325)*(F[0]/Ft))
    # P1t = ((Pt/101325)*(F[1]/Ft))
    # P2t = ((Pt/101325)*(F[2]/Ft))
    # P3t = ((Pt/101325)*(F[3]/Ft))
    
    # P0s=((Ps/101325)*(F[4]/Fs)) 
    # P1s=((Ps/101325)*(F[5]/Fs)) 
    # P2s=((Ps/101325)*(F[6]/Fs)) 
    # P3s=((Ps/101325)*(F[7]/Fs))
    
    # Reaction rates [mol/(h.cm3)]
    
    if C[0]==0:
        r[0] = 0
    else:
        r[0] = (3600*k1*C[0]*(1-((k1_Inv*C[1]*C[2]**2)/(k1*C[0]**2))))
    
    if C[1]==0:
        r[1] = 0
    else:
        r[1] = (3600*k2*C[1]*(1-((k2_Inv*C[3]*C[2]**3)/(k2*C[1]**3))))
    
    # Molar balances adjustment with experimental data.
    eff = 0.9
    vb  = 0.5
    Cat = (1-vb)*eff
    
    # Molar balances dFdz - Tube (0 to 3) & Shell (4 to 7)
    dFdz[0]=  - Cat*r[0]*At            
    #-(Q/selec)*((P0t**0.25)-(P0s**0.25))*pi*dt
    dFdz[1]=1/2*Cat*r[0]*At-Cat*r[1]*At
    #-(Q/selec)*((P1t**0.25)-(P1s**0.25))*pi*dt
    dFdz[2]=    Cat*r[0]*At+Cat*r[1]*At
    #-      (Q)*((P2t**0.25)-(P2s**0.25))*pi*dt
    dFdz[3]=          (1/3)*Cat*r[1]*At
    #-(Q/selec)*((P3t**0.25)-(P3s**0.25))*pi*dt
    
    #dFdz[4]=                            (Q/selec)*((P0t**0.25)-(P0s**0.25))*pi*dt
    #dFdz[5]=                            (Q/selec)*((P1t**0.25)-(P1s**0.25))*pi*dt
    #dFdz[6]=                                  (Q)*((P2t**0.25)-(P2s**0.25))*pi*dt
    #dFdz[7]=                            (Q/selec)*((P3t**0.25)-(P3s**0.25))*pi*dt
    
    return dFdz