# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:03:52 2021

@author: arpit
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from membrane_toolkit.core import donnan_equilibrium
import statistics

# Constants
MW_water = 18            # g/mol, molecular weight of water
MW_phosphate = 94.9714   # g/mol, molecular wegiht of phosphate
F = 96487                # C/mol, Faraday constant
R = 8.314                # J/mol/K, universal gas constant  
T = 298                  # K, Temperature
pw = 998                 # g/L, water density
V =2                     # V, Voltage applied

# Membrane properties
delta = 110E-6                  # m, membrane thickness
WU_plain = 0.162                # water uptake of plain membrane
WU_HMO = 0.237                  # water uptake of HMO membrane
WU_low = 0.168                  # water uptake of HMO membrane
IEC_plain = 0.73                # meq/g, ion exchange capacity
IEC_HMO = 0.69                  # meq/g, ion exchange capacity
IEC_low = 0.72                  # meq/g, ion exchange capacity
w_p = 1                         # water permeability
C_fix = IEC_low*pw/WU_low/1000  # eq/L, Concentration of fixed ions
alpha = 0.3                    # Structural parameter
beta = 1                     # Structural parameter
Wp = 0.11                       # Volume fraction of particle in membrane
fg = 0.795                        # Volume fraction of gel phase in membrane
fpin = 0.013                    # Volume fraction of particle in inetrphase, assuming that the particles were formed by interacting with charged groups in gel
fint = 0.192                    # Volume fraction of intergel solution phase
fp = 0.08
fsin = 0.936
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
C_int = []
C_m =[0]*26
C_p =[0]*25
C_m[0] = donnan_equilibrium(C_feed, C_fix, z_counter=1, z_co=-1, nu_counter=1, nu_co=1, z_fix=-1, gamma=1)
C_p[0] = (C_m[0] + C_fix)

D_g_plus = []
D_g_minus = []
D_hmo_minus = []
D_hmo_plus = []
L_ghmo_plus = []
L_ghmo_minus = []
D_p_minus = []
D_p_plus = []
L_p_minus = []
L_p_plus = []
L_g_minus = []
L_g_plus = []
L_int_plus= []
L_int_minus = []
L_minus = []
L_plus = []
t_plus =[]
t_minus = []

##############################################################################
def flux(C_feed, C_fix):
    for n in range(1,25):
        pos = n*increment
        C_m[n] = C_m[-1]+((C_m[0]-C_m[-1])*(((delta-pos)/delta)**10))
        C_p[n] = (C_m[n] + C_fix)
        #C_int.append(C_feed+((0-C_feed)*(((delta-pos)/delta)**5)))

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
        
        def gelDhmo(x):
            return [(Ds_hmo-((x[0]*x[1]*(C_feed+C_feed))/((x[1]*C_feed)+(x[0]*C_feed)))),
                    (0.444 - ((F**2/(R*0.001*T))*((x[0]*C_feed)+(x[1]*C_feed))))]

        Diffcoefficient = fsolve(gelDhmo,[1e-9,1e-9])
        D_hmo_minus.append(Diffcoefficient [0])    # m2/s, Diffusion coefficient in membrane for co ion 
        D_hmo_plus.append(Diffcoefficient [1])     # m2/s, Diffusion coefficient in membrane for counter ion
        
        Aminus = (D_hmo_minus[n-1]-Ds)/(D_hmo_minus[n-1]+(2*Ds))
        Aplus = (D_hmo_plus[n-1]-Ds)/(D_hmo_plus[n-1]+(2*Ds))
        D_p_minus.append(Ds*(1-(2*Aminus/fpin))/(1-(Aminus/fpin)))
        D_p_plus.append(Ds*(1-(2*Aplus/fpin))/(1-(Aplus/fpin)))
        
        L_ghmo_plus.append((D_hmo_plus[n-1]*C_feed/(R*0.001*T)))         # Effective conductance of HMO for Na
        #L_ghmo_minus.append((D_hmo_minus[n-1]*C_feed/(R*0.001*T))) 
        
        L_p_minus.append(D_p_minus[n-1]*C_feed/(R*0.001*T)) 
        #L_p_plus.append(D_p_plus[n-1]*C_feed/(R*0.001*T))
        
        L_p_plus.append(L_ghmo_plus[n-1])
        #L_p_minus.append(L_ghmo_minus[n-1]-(2.76E-12))
        
        ######################################################################
        #lintminus = ((fp*((L_p_minus[n-1])**beta))+(fsin*((D_solution_minus*C_feed/(R*0.001*T))**beta)))**(1/beta)
        #lintplus = ((fp*((L_p_plus[n-1])**beta))+(fsin*((D_solution_plus*C_feed/(R*0.001*T))**beta)))**(1/beta)
        
        #L_int_minus.append(lintminus)
        #L_int_plus.append(lintplus)
        L_int_minus.append(D_solution_minus*C_feed/(R*0.001*T))
        L_int_plus.append(D_solution_plus*C_feed/(R*0.001*T))
        
        # Calculate total effective conductance
        #L_minus.append(((fg*(L_g_minus[n-1]**alpha))+((1-fg)*(L_int_minus[n-1])**alpha))**(1/alpha))
        #L_plus.append(((fg*(L_g_plus[n-1]**alpha))+((1-fg)*(L_int_plus[n-1])**alpha))**(1/alpha))
        L_minus.append(((fg*((L_g_minus[n-1])**alpha))+(fint*(fsin**(alpha/beta))*((L_int_minus[n-1])**alpha)))**(1/alpha))
        L_plus.append(((fg*(L_g_plus[n-1]**alpha))+(fint*(fsin**(alpha/beta))*(L_int_plus[n-1]**alpha)))**(1/alpha))
        
        J_p.append(L_plus[n-1]*((R*T*coffp/0.001/(delta/25))+(F*dEdx))/10000)
        J_m.append(L_minus[n-1]*((R*T*coffm/0.001/(delta/25))+(F*dEdx))/10000)
        
        t_minus.append((L_minus[n-1]*(R*0.001*T))/((L_minus[n-1]*(R*0.001*T))+(1.69e-9))) # change Li to Ci*Di [L-DC/RT]
        t_plus.append(L_plus[n-1]/(L_minus[n-1]+L_plus[n-1])) 
        
        if n==26:
            break
    return C_m, C_p, J_p, J_m, t_minus, t_plus, D_g_minus, D_g_plus, D_p_minus, D_p_plus, L_minus

Flux = flux(C_feed, C_fix)

Cminus = Flux[0]
Cplus = Flux[1]
Jp = (Flux[2])
Jm = (Flux[3])
tMINUS = Flux[4]
tPLUS = Flux[5]
Dgm = Flux[6]
Dgp = Flux[7]
Dghmom = Flux[8]
Dghmop = Flux[9]
Lmin = Flux[10]
#print (tMINUS)
#print (Jm)
#print(statistics.mean(Cminus))
#print(statistics.mean(Cplus))
Perm = ((Jm[-1])*delta)/(C_feed-Cminus[-1])
print(Perm)
print(statistics.mean(tMINUS))
#print(statistics.mean(tPLUS))
print(statistics.mean(Jm))
print(statistics.mean(Lmin))
#print(statistics.mean(Dgm))
#print(statistics.mean(Dgp))
#print(statistics.mean(Dghmop))
#print(statistics.mean(Dghmom))
X = []
def xrange(start, stop, step):
    for i in range(start,stop,step):
        X.append(i)
    return X
#xvalues = np.array(xrange(0,24,1))
#print(Jm)
#plt.plot(xvalues,Jm,'*') #, xvalues,Jm,'o')
#plt.show()