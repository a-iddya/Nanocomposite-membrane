# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 09:41:05 2021

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

# Membrane properties
delta = 110E-6                           # m, membrane thickness
WU_plain = 0.162                         # water uptake of plain membrane
WU_HMO = 0.228                           # water uptake of HMO membrane
IEC_plain = 0.73                         # meq/g, ion exchange capacity
IEC_HMO = 0.69                           # meq/g, ion exchange capacity
w_p = 1                                  # water permeability
alpha = 0.3                            # Structural parameter
fint = 0.128                               # Volume fraction of gel phase in membrane
fg = 0.872                               # Volume fraction of intergel solution phase
C_fix = IEC_plain*pw/1000/WU_plain       # eq/L, Concentration of fixed ions
dphi = 0.5                               # V, voltage applied for membrane conductivity expt
I = 20.23                                # A/m2, current density for membrane conductivity expt
Ds = 2.989E-9                            # m2/s, Salt diffusion coefficient of membrane
dEdx = 2/delta/25
increment = delta/25

# Solution Properties
C_feed = 0.1               # M, feed concentration
D_solution_minus = 0.879E-9  # m2/s, Diffusion coefficient, H2PO4-
D_solution_plus = 1.33E-9   # m2/s, Diffusion coefficient, Na


J_p = []
J_m = []
C_m =[0]*26
C_p =[0]*25
C_int = []
C_m[0] = donnan_equilibrium(C_feed, C_fix, z_counter=1, z_co=-1, nu_counter=1, nu_co=1, z_fix=-1, gamma=1)
C_p[0] = (C_m[0] + C_fix)

D_g_plus = []
D_g_minus = []
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
        C_m[n] = C_m[-1]+((C_m[0]-C_m[-1])*(((25-n)/25)**10))
        C_p[n] = (C_m[n] + C_fix)
        #C_int.append(C_feed+((0-C_feed)*(((delta-pos)/delta)**5)))
        
        k = 0.45      # S/m, conductivity of membrane
        def gelD(x):
            return [(k - ((F**2/(R*0.001*T))*((x[0]*C_p[n])+(x[1]*C_m[n])))),
                    (Ds-((x[0]*x[1]*(C_p[n]+C_m[n]))/((x[0]*C_p[n])+(x[1]*C_m[n]))))]
        Diffcoefficient = fsolve(gelD,[1e-10,1e-10])
        D_g_plus.append(Diffcoefficient [0])      # m2/s, Diffusion coefficient in membrane for co ion 
        D_g_minus.append(Diffcoefficient [1])     # m2/s, Diffusion coefficient in membrane for counter ion
        coffm = np.asarray(C_m[n])
        coffp = np.asarray(C_p[n])
        # Calculate effective conductance coefficient, for gel phase
        L_g_minus.append(D_g_minus[n-1]*coffm/(R*0.001*T))
        L_g_plus.append(D_g_plus[n-1]*coffp/(R*0.001*T))
        
        L_int_minus.append(D_solution_minus*C_feed/(R*0.001*T))
        L_int_plus.append(D_solution_plus*C_feed/(R*0.001*T))
        
        # Calculate total effective conductance
        L_minus.append(((fg*(L_g_minus[n-1]**alpha))+(fint*(L_int_minus[n-1]**alpha)))**(1/alpha))
        L_plus.append(((fg*(L_g_plus[n-1]**alpha))+(fint*(L_int_plus[n-1]**alpha)))**(1/alpha))
        J_p.append(L_plus[n-1]*((R*T*coffp/0.001/(delta/25))+(F*dEdx))/10000)
        J_m.append(L_minus[n-1]*((R*T*coffm/0.001/(delta/25))+(F*dEdx))/10000)
        
        t_minus.append((L_minus[n-1]*(R*0.001*T))/((L_minus[n-1]*(R*0.001*T))+(3.36e-9)))
        t_plus.append((L_plus[n-1]*(R*0.001*T))/((L_minus[n-1]*(R*0.001*T))+(L_plus[n-1]*(R*0.001*T))))
        if n==26:
            break
    return C_m, C_p, D_g_minus, D_g_plus, J_p, J_m, t_plus, t_minus, L_minus, L_plus

Flux = flux(C_feed, C_fix)

Cminus = Flux[0]
Cplus = Flux[1]

Dgminus = Flux[2]
Dgplus = Flux[3]
Jp = (Flux[4])
Jm = (Flux[5])

tPLUS = Flux[6]
tMINUS = Flux[7]
Lmin= Flux[8]
Lplus = Flux[9]
Perm = ((Jm[-1])*delta)/(C_feed-Cminus[-1])
print(Perm)
P8 = 2*(statistics.mean(tMINUS))*(statistics.mean(Lmin))*R*T/C_feed
print (P8)
#print(tMINUS)
#print(statistics.mean(Cminus))
#print(statistics.mean(Cplus))
print(statistics.mean(tMINUS))
print(statistics.mean(Jm))
print(statistics.mean(Lmin))
print(statistics.mean(Lplus))
#print(statistics.mean(Dgminus))
#print(statistics.mean(Dgplus))

def xrange(start, stop, step):
    X=[]
    for i in range(start,stop,step):
        X.append(i)
    return X
#xvalues = np.array(xrange(0,24,1))
#x1values = np.array(xrange(0,26,1))
#print(Jm)
#plt.plot(xvalues,tMINUS,"*") # x1values, Cminus, 'o')
#plt.show()