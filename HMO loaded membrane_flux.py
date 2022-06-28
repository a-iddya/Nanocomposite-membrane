# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 08:52:15 2021

@author: arpit
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from membrane_toolkit.core import donnan_equilibrium
import statistics
import pandas as pd

# Constants
MW_water = 18            # g/mol, molecular weight of water
MW_phosphate = 94.9714   # g/mol, molecular wegiht of phosphate
F = 96487                # C/mol, Faraday constant
R = 8.314                # J/mol/K, universal gas constant  
T = 298                  # K, Temperature
pw = 998                 # g/L, water density
V =2                     # V, Voltage applied

# Membrane properties
delta = 30E-6                  # m, membrane thickness
WU_HMO = 0.237                  # water uptake of HMO membrane
IEC_HMO = 0.69                  # eq/g, ion exchange capacity
w_p = 1                         # water permeability
C_fix = IEC_HMO*pw/WU_HMO/1000  # eq/L, Concentration of fixed ions
alpha = 0.7                    # Structural parameter
beta = 1                      # Structural parameter
Wp = 0.11                       # Volume fraction of particle in membrane
fg = 0.741                        # Volume fraction of gel phase in membrane
fpin = 0.0237                    # Volume fraction of particle in inetrphase, assuming that the particles were formed by interacting with charged groups in gel
fint = 0.235                      # Volume fraction of intergel solution phase
fp = 0.1
fsin = 0.91
dphi = 0.5                      # V, voltage applied for membrane conductivity expt
I = 20.23                       # A/m2, current density for membrane conductivity expt
Ds = 2.989E-9                   # m2/s, Salt diffusion coefficient of membrane
Ds_hmo = 3.32E-10               # m2/s, Salt diffusion coefficient of membrane
dEdx = V/delta/25
increment = delta/25

# Solution Properties
C_feed = 0.1                # mol/L, feed concentration
D_solution_minus = 0.879E-9  # m2/s, Diffusion coefficient, H2PO4-
D_solution_plus = 1.33E-9   # m2/s, Diffusion coefficient, Na

J_p = []
J_m = []
C_m =[0]*26
C_p =[0]*25
C_OH =[0]*26
C_H =[0]*25
C_int = []
C_m[0] = donnan_equilibrium(C_feed, C_fix, z_counter=1, z_co=-1, nu_counter=1, nu_co=1, z_fix=-1, gamma=1)
C_p[0] = (C_m[0] + C_fix)
#C_OH[0] = donnan_equilibrium(1e-11, C_fix, z_counter=1, z_co=-1, nu_counter=1, nu_co=1, z_fix=-1, gamma=1)
#print(C_OH[0])
#C_H[0] = (1e-1 + C_fix)
#print(C_H[0])

D_g_plus = []
D_g_minus = []
D_g_H = []
D_g_OH = []
D_hmo_minus = []
D_hmo_plus = []
D_p_minus = []
D_p_plus = []
L_ghmo_plus = []
L_ghmo_minus = []
L_p_minus = []
L_p_plus = []
L_g_minus = []
L_g_plus = []
L_int_plus= []
L_int_minus = []
L_minus = []
L_plus = []
L_g_OH = []
L_g_H = []
L_int_H= []
L_int_OH = []
L_OH = []
L_H = []

t_plus =[]
t_minus = []

##############################################################################
def flux(C_feed, C_fix):
    for n in range(1,25):
        pos = increment*n
        C_m[n] = C_m[-1]+((C_m[0]-C_m[-1])*(((delta-pos)/delta)**5))
        C_p[n] = (C_m[n] + C_fix)
        #C_int.append(C_feed+((0-C_feed)*(((delta-pos)/delta)**5)))
        #C_OH[n] = C_OH[-1]+((C_OH[0]-0)*(((delta-pos)/delta)**1))
        #C_H[n] = (C_OH[n] + C_fix)

        k = 0.45      # S/m, conductivity of membrane

        def gelD(x):
            return [(k - ((F**2/(R*0.001*T))*((x[0]*C_p[n])+(x[1]*C_m[n])))),
                    (Ds-((x[0]*x[1]*(C_p[n]+C_m[n]))/((x[0]*C_p[n])+(x[1]*C_m[n]))))]

        Diffcoefficient = fsolve(gelD,[1e-9,1e-9])
        D_g_plus.append(Diffcoefficient [0])      # m2/s, Diffusion coefficient in membrane for co ion 
        D_g_minus.append(Diffcoefficient [1])     # m2/s, Diffusion coefficient in membrane for counter ion
        
        coffm = np.asarray(C_m[n])
        coffp = np.asarray(C_p[n])
                
        # Calculate effective conductance coefficient, for gel phase
        L_g_minus.append(D_g_minus[n-1]*coffm/(R*0.001*T))
        L_g_plus.append(D_g_plus[n-1]*coffp/(R*0.001*T))
        
        L_int_minus.append(D_solution_minus*C_feed/(R*0.001*T))
        L_int_plus.append(D_solution_plus*C_feed/(R*0.001*T))
        
# =============================================================================
 #       L_int_OH.append(4.56E-9*1e-9/(R*0.001*T))
 #       L_int_H.append(7.6E-9*1e-5/(R*0.001*T))  
# =============================================================================

        # Calculate total effective conductance
        #L_minus.append(((fg*(L_g_minus[n-1]**alpha))+((1-fg)*L_int_minus[n-1]**alpha))**(1/alpha))
        #L_plus.append(((fg*(L_g_plus[n-1]**alpha))+((1-fg)*L_int_plus[n-1]**alpha))**(1/alpha))
        L_minus.append(((fg*((L_g_minus[n-1])**alpha))+(fint*(fsin**(alpha/beta))*((L_int_minus[n-1])**alpha)))**(1/alpha))
        L_plus.append(((fg*(L_g_plus[n-1]**alpha))+(fint*(fsin**(alpha/beta))*(L_int_plus[n-1]**alpha)))**(1/alpha))
        
# =============================================================================
#        L_H.append(((fg*((L_g_H[n-1])**alpha))+(fint*((L_int_H[n-1])**alpha)))**(1/alpha))
#        L_OH.append(((fg*((L_g_OH[n-1])**alpha))+(fint*((L_int_OH[n-1])**alpha)))**(1/alpha))
# =============================================================================
       
        J_p.append(L_plus[n-1]*((R*T*coffp/0.001/(delta/25))+(F*dEdx))/10000)
        J_m.append(L_minus[n-1]*((R*T*coffm/0.001/(delta/25))+(F*dEdx))/10000)
        #print(L_minus[n-1],L_plus[n-1],L_H[n-1])
        
        t_minus.append((L_minus[n-1]*(R*0.001*T))/((L_minus[n-1]*(R*0.001*T))+(3.43e-9)))
        t_plus.append(L_plus[n-1]/(L_minus[n-1]+L_plus[n-1])) 
        
        if n==26:
            break
    return C_m, C_p, J_p, J_m, t_minus, t_plus, D_g_minus, D_g_plus, L_minus

Flux = flux(C_feed, C_fix)

Cminus = Flux[0]
Cplus = Flux[1]
Jp = (Flux[2])
Jm = (Flux[3])
tMINUS = Flux[4]
tPLUS = Flux[5]
Dgm = Flux[6]
Dgp = Flux[7]
Lmin = Flux[8]
Perm = ((Jm[-1])*delta)/(C_feed-Cminus[-1])
#print(Perm)
#print (tMINUS)
#print (Jm)
#print (Cminus)
#df = pd.DataFrame(Cminus)
#df.to_csv('n1.csv')
#print(statistics.mean(Cminus))
#print(statistics.mean(Cplus))
print(statistics.mean(tMINUS))
#print(statistics.mean(tPLUS))
print(statistics.mean(Jm))
#print(statistics.mean(Lmin))
#print(statistics.mean(Dgm))
#print(statistics.mean(Dgp))
#print(statistics.mean(Dphmop))
#print(statistics.mean(Dphmom))
X = []
def xrange(start, stop, step):
    for i in range(start,stop,step):
        X.append(i)
    return X
#xvalues = np.array(xrange(0,24,1))
#print(Jm)
#plt.plot(xvalues,Jm,'*') #, xvalues,Jm,'o')
#plt.show()