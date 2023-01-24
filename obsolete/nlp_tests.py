import numpy as np
import scipy.io as sio
from nlp_based_approach import nlp_based_approach
import os
import time
import matplotlib.pyplot as plt

def shower2d(u):
    
    d = np.zeros(2)
    y = np.zeros(2)
    y[0]=u[0]+u[1]
    if y[0]!=0:
        y[1]=(u[0]*(60+d[0])+u[1]*(120+d[1]))/(u[0]+u[1])
    else:
        y[1]=(60+120)/2
        
    return y


def shower(u):
    return shower2d(u) 

if __name__ == "__main__":
    file_dir = os. getcwd()
    file_name = '\\mvs.mat'
    full_path = file_dir + file_name
    data = sio.loadmat(full_path)
    DOSPts =  data['dos']
    # lb = data['lb']
    # ub = data['ub']
    u0 = np.asarray([3, 3])
    lb = np.asarray([1e-4,1e-4])
    ub = np.asarray([10,10])
    

    t = time.time()
    fDIS_1, fDOS_1, message_list = nlp_based_approach(DOSPts, shower, u0, 
                                                  lb,ub, method='trust-constr')
    
    
    elapsed_1 = time.time() - t
    
    
    
    t = time.time()
    fDIS_2, fDOS_2, message_list = nlp_based_approach(DOSPts, shower, u0, 
                                                  lb,ub, method='Nelder-Mead')
    
    
    elapsed_2 = time.time() - t
    
    
    
    t = time.time()
    fDIS_3, fDOS_3, message_list = nlp_based_approach(DOSPts, shower, u0, 
                                                  lb,ub, method='ipopt')
    
    
    elapsed_3 = time.time() - t
    
    
    
    t = time.time()
    fDIS_4, fDOS_4, message_list = nlp_based_approach(DOSPts, shower, u0, 
                                                  lb,ub, method='DE')
    
    
    elapsed_4 = time.time() - t
    
    
    t = time.time()
    fDIS_5, fDOS_5, message_list = nlp_based_approach(DOSPts, shower, u0, 
                                                  lb,ub, method='COBYLA')
    
    
    elapsed_5 = time.time() - t
    
    plt.rcParams['text.usetex'] = True
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(fDIS_1[:,0],fDIS_1[:,1],'ro')    
    plt.ylabel('Cold water flowrate [gal/min]')
    plt.xlabel('Hot water flowrate [gal/min]')
    plt.title('fDIS - Nelder-Mead NLP algorithm')
    plt.subplot(1, 2, 2)
    plt.plot(fDOS_1[:,0],fDOS_1[:,1],'ro')    
    plt.ylabel('Total flowrate [gal/min]')
    plt.xlabel('Temperature [F]')
    plt.title('fDOS - Trust-constr NLP algorithm')
    plt.show
    
    
    
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(fDIS_2[:,0],fDIS_2[:,1],'ro')    
    plt.ylabel('Cold water flowrate [gal/min]')
    plt.xlabel('Hot water flowrate [gal/min]')
    plt.title('fDIS - Nelder-Mead NLP algorithm')
    plt.subplot(1, 2, 2)
    plt.plot(fDOS_2[:,0],fDOS_2[:,1],'ro')    
    plt.ylabel('Total flowrate [gal/min]')
    plt.xlabel('Temperature [F]')
    plt.title('fDOS - Nelder-Mead NLP algorithm')
    plt.show
    
    
    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.plot(fDIS_3[:,0],fDIS_3[:,1],'ro')    
    plt.ylabel('Cold water flowrate [gal/min]')
    plt.xlabel('Hot water flowrate [gal/min]')
    plt.title('fDIS - IPOPT NLP algorithm')
    plt.subplot(1, 2, 2)
    plt.plot(fDOS_3[:,0],fDOS_3[:,1],'ro')    
    plt.ylabel('Total flowrate [gal/min]')
    plt.xlabel('Temperature [F]')
    plt.title('fDOS - IPOPT NLP algorithm')
    plt.show
    
    
    plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.plot(fDIS_4[:,0],fDIS_4[:,1],'ro')    
    plt.ylabel('Cold water flowrate [gal/min]')
    plt.xlabel('Hot water flowrate [gal/min]')
    plt.title('fDIS - DE NLP algorithm')
    plt.subplot(1, 2, 2)
    plt.plot(fDOS_4[:,0],fDOS_4[:,1],'ro')    
    plt.ylabel('Total flowrate [gal/min]')
    plt.xlabel('Temperature [F]')
    plt.title('fDOS - DE NLP algorithm')
    plt.show
    
    plt.figure(5)
    plt.subplot(1, 2, 1)
    plt.plot(fDIS_5[:,0],fDIS_5[:,1],'ro')    
    plt.ylabel('Cold water flowrate [gal/min]')
    plt.xlabel('Hot water flowrate [gal/min]')
    plt.title('fDIS - COBYLA NLP algorithm')
    plt.subplot(1, 2, 2)
    plt.plot(fDOS_5[:,0],fDOS_5[:,1],'ro')    
    plt.ylabel('Total flowrate [gal/min]')
    plt.xlabel('Temperature [F]')
    plt.title('fDOS - COBYLA NLP algorithm')
    plt.show