# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt



# note for jan: row 36, col 8 has 4,040 instead of 4.040, so i updated it in the .txt file.

# constants, cgs units
G = 6.67e-8
a = 7.5646e-15
c = 3e10

Msun = 1.989e33
Lsun = 3.9e33
Rsun = 6.96e10

kappa_R = 0.3

# stellar parameters in solar units

M = 1
R = 1.0763
L = 1.3727
Teff = 6034





numpy_data = np.loadtxt('novotny.txt', skiprows=1, usecols = (0, 1, 2, 4, 6, 7, 9, 10), unpack=True) # (0, 1, 2, 4, 6, 7, 9, 10))

m = numpy_data[0]
r = numpy_data[1]
P = numpy_data[2]
T = numpy_data[3]
l = numpy_data[4]
rho = numpy_data[5]
X = numpy_data[6]
He = numpy_data[7]


# need to get exponents


with open('novotny.txt', 'r') as f:
    data = f.readlines()
    n = len(data) - 1
    P_exp = np.zeros(n)
    T_exp = np.zeros(n)
    rho_exp = np.zeros(n)

    # first row is header
    for i in range(1, n+1):
        temp = data[i]
        exp = temp.split('(')
        
        exp1 = exp[1].split(')')
        exp2 = exp[2].split(')')
        exp3 = exp[3].split(')')
        
        P_exp[i-1] = int(exp1[0])
        T_exp[i-1] = int(exp2[0])
        rho_exp[i-1] = int(exp3[0])
        
# adding exponents and scaling constants to relevant arrays
    
P = P*10**(P_exp+17)
T = T*10**(T_exp+6)
rho = rho*10**(rho_exp)

# outer boundary stuff, part b

T_R = ((L*Lsun)/(np.pi*a*c*(R*Rsun)**2))**0.25

P_R = 2*G*Msun/(3*kappa_R*(R*Rsun)**2)





# want plot from m/M = 0.9 outwards

x = np.where(np.round(m, 2) == 0.9)[0][0]

P_data = P[x:]
T_data = T[x:]

plt.figure()
plt.plot(np.log10(P_data), np.log10(T_data), label='table values')

plt.scatter([np.log10(P_R)], [np.log10(T_R)], color='black', label='surface')
plt.title('Temperature vs Pressure, cgs units')
plt.legend()
plt.xlabel('log P')
plt.ylabel('log T')

plt.show()


# use lmfit

kappa0 = 6e15

B = 8.5*3*kappa0*(L*Lsun)/(2*16*np.pi*a*c*G*Msun)

import lmfit
fit_params = lmfit.create_params(C=1)


def residuals(params, P_data, T_data): # T_data log form, P_data not
    C = params['C'].value

    T_calc = (np.log10(B) + np.log10(P_data**2))/8.5 + C
    
    return T_data - T_calc



result = lmfit.minimize(residuals, fit_params, args=(P_data[:-5], np.log10(T_data[:-5])))
C = result.params['C'].value


fit_params = lmfit.create_params(const=1, slope=1)

def residuals2(params, P_data, T_data): # T_data log form, P_data not
    const = params['const'].value
    slope = params['slope'].value

    T_calc = slope*P_data + const
    
    return T_data - T_calc

result = lmfit.minimize(residuals2, fit_params, args=(np.log10(P_data[-6:]), np.log10(T_data[-6:])))
slope = result.params['slope'].value
const = result.params['const'].value

new_P = np.arange(4, 12, 0.1)

# have at logT=4, logP = 7.22

final_slope = (np.log10(T_R) - 4)/(np.log10(P_R) - 7.22)

plt.figure()
plt.plot(np.log10(P_data), np.log10(T_data), label='table values')
plt.plot(np.log10(P_data[:-5]), (np.log10(B) + np.log10(P_data[:-5]**2))/8.5 + C, color='blue', linestyle='dashed', label='rad fit')
plt.plot(new_P[34:], slope*new_P[34:] + const, color='red', linestyle='dashed', label='conv fit')

plt.plot(new_P, 4*np.ones(new_P.shape), color='green', label='T where conv invalid')
plt.plot(new_P[8:35], final_slope*(new_P[8:35] - np.log10(P_R)) + np.log10(T_R), color='orange', linestyle='dashed', label='H partial ionization fit')

plt.scatter([np.log10(P_R)], [np.log10(T_R)], color='black', label='surface')
plt.title('Temperature vs Pressure, cgs units')
plt.legend()
plt.xlabel('log P')
plt.ylabel('log T')

plt.show()





















