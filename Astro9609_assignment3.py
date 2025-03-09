# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:03:33 2025

@author: nickj
"""

import numpy as np
import matplotlib.pyplot as plt



'''
QUESTION 1
'''

# part a

# given values, everything in cgs (with energy units in eV)

temp = 50000.0 # K
pressure = 1.0e11 # dyne/cm^2

# constants
m_e = 9.11e-28 # g
planck = 6.626e-27 # erg s
boltzmann = 1.38e-16 # erg/K
boltzmann_eV = 8.617e-5 #eV/K since chi are given in eV, use this one for exponent

# told that log(n(He)/n(H)) = -1.07
# or, solar_Y/(4*solar_X) = 10^-1.07. neglecting metals, solar_X + solar_Y = 1
solar_X = 1/(1+4*(10**-1.07))
solar_Y = 1 - solar_X

mu0 = 1/(solar_X + solar_Y/4)

# note: E ranges from 0 for fully neutral H and He, to just under mu0 for fully ionized gas. 

# initial guess: T and P are high, so guess H half ionized,
# and He a split between neutral and 1st ionization
E_in_1 = (solar_X*0.5 + (solar_Y/4)*(0.5 + 2*0))*mu0
E_in_1 = 0.1

print('\n1.a\n')
print(f'have that the hydrogen and helium fractions are {solar_X} and {solar_Y} respectively')
print(f'have the mean molecular weight of the ions is {mu0}') 
print(f'the initial guess E_in_1 is {E_in_1}') 


# part b

# defining the system of equations

# constants; uses the convention u_i_r
# i is 1 for hydrogen, 2 for helium
# r is 0 for neutral, 1 for singly ionized, 2 for doubly ionized
u_1_0 = 2 
u_1_1 = 1
u_2_0 = 1
u_2_1 = 2
u_2_2 = 1

# chi in units of eV, has negative incorporated
chi_1_0 = -13.598
chi_2_0 = -24.587
chi_2_1 = -54.418

def K(r, i, T, P): 
    # 14.36 in the homework/textbook
    num = 2*((2*np.pi*m_e)**1.5)*(boltzmann*T)**2.5
    den = P*planck**3
    
    if r == 1:
        partition = u_1_1/u_1_0
        exp = np.exp(chi_1_0/(boltzmann_eV*T))
        
    elif r == 2 and i == 0:
        partition = u_2_1/u_2_0
        exp = np.exp(chi_2_0/(boltzmann_eV*T))
        
    elif r == 2 and i == 1:
        partition = u_2_2/u_2_1
        exp = np.exp(chi_2_1/(boltzmann_eV*T))
        
    else: # this is a typo failsafe that will cause a crash
        partition = 'silly'
        exp = 'goose'
        
    return partition*num*exp/den

# use _i_r convention for calculating ratios

ratio_1_0 = (1 + 1/E_in_1)*K(1, 0, temp, pressure)
ratio_2_0 = (1 + 1/E_in_1)*K(2, 0, temp, pressure)
ratio_2_1 = (1 + 1/E_in_1)*K(2, 1, temp, pressure)

# using 14.37, can turn ratios into the individual x values

x_1_0 = 1/(ratio_1_0 + 1)
x_1_1 = ratio_1_0*x_1_0

x_2_0 = 1/(ratio_2_0*(ratio_2_1 + 1) + 1)
x_2_1 = ratio_2_0*x_2_0
x_2_2 = ratio_2_1*x_2_1

# use the calculated x values to find a new E
def E(x_1_1, x_2_1, x_2_2):
    # 14.34 in the homework/textbook
    
    return (solar_X*x_1_1 + (solar_Y/4)*(x_2_1 + 2*x_2_2))*mu0

E_out_1 = E(x_1_1, x_2_1, x_2_2)

A_1 = E_out_1 - E_in_1
print('\n1.b\n')
print(f'E_out_1 is {E_out_1}, and A_1 is {A_1}')
print(f'hydrogen x are (ascending ionization): {x_1_0}, {x_1_1}') 
print(f'helium x are (ascending ionization): {x_2_0}, {x_2_1}, {x_2_2}')

# part c

# repeating the process, put it in a function

def E_iteration(E_in, T, P):
    # use _i_r convention for calculating ratios
    
    ratio_1_0 = (1 + 1/E_in)*K(1, 0, T, P)
    ratio_2_0 = (1 + 1/E_in)*K(2, 0, T, P)
    ratio_2_1 = (1 + 1/E_in)*K(2, 1, T, P)
    
    # using 14.37, can turn ratios into the individual x values
    
    x_1_0 = 1/(ratio_1_0 + 1)
    x_1_1 = ratio_1_0*x_1_0
    
    x_2_0 = 1/(ratio_2_0*(ratio_2_1 + 1) + 1)
    x_2_1 = ratio_2_0*x_2_0
    x_2_2 = ratio_2_1*x_2_1
    
    E_out = E(x_1_1, x_2_1, x_2_2)
    A = E_out - E_in
    
    # return the x in a list
    x_list = [x_1_0, x_1_1, x_2_0, x_2_1, x_2_2]
    
    return E_out, A, x_list

E_in_2 = E_in_1 + 0.01

E_out_2, A_2, x_list_2 = E_iteration(E_in_2, temp, pressure)

print('\n1.c\n')
print(f'the new E is {E_out_2}, and the new A is {A_2}')
print(f'the various x are {x_list_2}')

# part d

# now want to use the results of b and c to find a derivative of A with respect to E,
# to correct E by such that delta A = 0

def A_derivative(A_1, A_2, E_1, E_2):
    # here 1 is representative of the n th iteration, and 2 is representative
    # of the n+1 th iteration 
    
    return (A_2 - A_1)/(E_2 - E_1)

A_derivative_1 = A_derivative(A_1, A_2, E_in_1, E_in_2)

delta_E_1 = -1*A_1/A_derivative_1

E_in_old = E_in_1 + delta_E_1

print('\n1.d\n')
print(f'the updated E_in to use is {E_in_old}')

# part e, f

# using parts a-d to solve the system of equations. 
# instead of numbers use old and new 
A_old = np.copy(A_1)

# note: since A_old is computed at the beginning of the loop, once the condition is met
# the loop will run one more time and return a slightly better A_old
n = 0
while abs(A_old) > 1e-5 and n < 100:
    # checking E guess
    E_out_old, A_old, x_list_old = E_iteration(E_in_old, temp, pressure)
    
    # iterating E_out_old
    E_in_new = E_in_old + 0.01
    E_out_new, A_new, x_list_new = E_iteration(E_in_new, temp, pressure)
    
    A_derivative_old = A_derivative(A_old, A_new, E_in_old, E_in_new)
    
    delta_E_old = -1*A_old/A_derivative_old
    
    E_in_old = E_in_old + delta_E_old
    
    n+=1

print('\n1.e, 1.f\n')
print(f'the final E_in is {E_in_old}')
print(f'hydrogen x are (ascending ionization): {x_list_old[0]}, {x_list_old[1]}') 
print(f'helium x are (ascending ionization): {x_list_old[2]}, {x_list_old[3]}, {x_list_old[4]}')
print(f'this run stopped after {n} iterations')

# in order to more easily modify initial conditions, turning the above into a function

def E_finder(E_in_old, T, P):
    # checking E guess
    E_out_old, A_old, x_list_old = E_iteration(E_in_old, T, P)

    # iterating E_out_old, checking new guess
    E_in_new = E_in_old + 0.01
    E_out_new, A_new, x_list_new = E_iteration(E_in_new, T, P)
    
    # calculating new E guess
    A_derivative_old = A_derivative(A_old, A_new, E_in_old, E_in_new)
    delta_E_old = -1*A_old/A_derivative_old
    E_in_old = E_in_old + delta_E_old

    return E_in_old, A_old, x_list_old



# check for solar surface conditions as stated in the textbook.
# initial E guess: H 50% ionized, He 75% neutral, 25% singly ionized

E_guess_surface = (solar_X*0.5 + (solar_Y/4)*(0.25 + 2*0))*mu0
T_surface = 5779
P_surface = 1.01e5

E_surface, A_surface, x_list_surface = E_finder(E_guess_surface, T_surface, P_surface)

# defining a number to stop the loop in case of divergent behavior
n = 0
while abs(A_surface) > 1e-5 and n < 100:
    E_surface, A_surface, x_list_surface = E_finder(E_surface, T_surface, P_surface)
    n+=1
    
print('\ntextbook solar surface\n')
print(f'the final E_in corresponding to T={T_surface} and P={P_surface} is {E_surface}')
print(f'hydrogen x are (ascending ionization): {x_list_surface[0]}, {x_list_surface[1]}') 
print(f'helium x are (ascending ionization): {x_list_surface[2]}, {x_list_surface[3]}, {x_list_surface[4]}')  
print(f'this run stopped after {n} iterations')
print('\nnote for jan: my equations contain a 1/E, and so converge slowly as E approaches 0, hence this') 
print('run using the full n=100. without an n limit, this particular set of conditions reaches')
print('the required A limit after n=183 iterations.')

# check for solar interior conditions as stated in the textbook.
# use same initial guess as in 1.a

E_guess_interior = (solar_X*0.5 + (solar_Y/4)*(0.5 + 2*0))*mu0
T_interior = 7.17e5
P_interior = 3.35e12

E_interior, A_interior, x_list_interior = E_finder(E_guess_interior, T_interior, P_interior)

# defining a number to stop the loop in case of divergent behavior
n = 0
while abs(A_interior) > 1e-5 and n < 100:
    E_interior, A_interior, x_list_interior = E_finder(E_interior, T_interior, P_interior)
    n+=1

print('\ntextbook solar interior\n')
print(f'the final E_in corresponding to T={T_interior} and P={P_interior} is {E_interior}')
print(f'hydrogen x are (ascending ionization): {x_list_interior[0]}, {x_list_interior[1]}') 
print(f'helium x are (ascending ionization): {x_list_interior[2]}, {x_list_interior[3]}, {x_list_interior[4]}') 
print(f'this run stopped after {n} iterations')

# redo with 1a T and P, but with an unphysically large E=10

E_guess_interior = 10
T_interior = 50000.0 # K
P_interior = 1.0e11 # dyne/cm^2

E_interior, A_interior, x_list_interior = E_finder(E_guess_interior, T_interior, P_interior)

# defining a number to stop the loop in case of divergent behavior
n = 0
while abs(A_interior) > 1e-5 and n < 100:
    E_interior, A_interior, x_list_interior = E_finder(E_interior, T_interior, P_interior)
    n+=1

print(f'\nunphysically large E={E_guess_interior}\n')
print(f'the final E_in corresponding to T={T_interior} and P={P_interior} is {E_interior}')
print(f'hydrogen x are (ascending ionization): {x_list_interior[0]}, {x_list_interior[1]}') 
print(f'helium x are (ascending ionization): {x_list_interior[2]}, {x_list_interior[3]}, {x_list_interior[4]}') 
print(f'this run stopped after {n} iterations')

# now with a hilariously huge E=100000

E_guess_interior = 100000
T_interior = 50000.0 # K
P_interior = 1.0e11 # dyne/cm^2

E_interior, A_interior, x_list_interior = E_finder(E_guess_interior, T_interior, P_interior)

# defining a number to stop the loop in case of divergent behavior
n = 0
while abs(A_interior) > 1e-5 and n < 100:
    E_interior, A_interior, x_list_interior = E_finder(E_interior, T_interior, P_interior)
    n+=1

print(f'\nhilariously huge E={E_guess_interior}\n')
print(f'the final E_in corresponding to T={T_interior} and P={P_interior} is {E_interior}')
print(f'hydrogen x are (ascending ionization): {x_list_interior[0]}, {x_list_interior[1]}') 
print(f'helium x are (ascending ionization): {x_list_interior[2]}, {x_list_interior[3]}, {x_list_interior[4]}') 
print(f'this run stopped after {n} iterations')



'''
QUESTION 2
'''

# objective: code an RK2 method to solve Lane-Emden equation

# first, define the LE equations to be integrated

def w_function(z, v, n): # n isnt used but i put it in the function definitions for symmetry
    # w(z) as defined by homework. note this is an integrand
    w = v/z**2 
    return w

def v_function(z, w, n): 
    # v(z) as defined by homework. note this is an integrand
    v = -1*(w**n)*z**2
    return v

# these are for n>=5 

def w_function_special(y, v, n): # n isnt used but i put it in the function definitions for symmetry
    # w(z) with z= y/(1-y) (it simplifies to something simple here)
    w = v/y**2 
    return w

def v_function_special(y, w, n): 
    # v(z) with z= y/(1-y)
    v = -1*(w**n)*(y**2)/(1-y)**4
    return v

# i will need to run this for many n, so do it object oriented

class Polytrope:
    
    def __init__(self,
                 h,
                 n,
                 v, 
                 w, 
                 z, 
                 limit,
                 stop):
        
        self.h = h
        self.n = n
        self.v = v
        self.w = w
        self.z = z
        self.limit = limit
        self.stop = stop
    
    @staticmethod
    def RK2(h, n, limit):
        # note: we know the beginning value is at z=0. however, the endpoint value z_n such that w(z_n)=0
        # is not known. So, will need to set up RK2 to run until w(z) = 0.
        
        v = 0.0
        w = 1.0
        z = 0.0
        
        # want to save values of w, v, z into lists for plotting
        v_list = [0.0]
        w_list = [1.0]
        z_list = [0.0]
        
        k1_v = h*v
        k1_w = h*w
        
        k2_v = h*v_function(z+0.5*h, w+0.5*k1_w, n)
        k2_w = h*w_function(z+0.5*h, v+0.5*k1_v, n)
        
        v += k2_v
        w += k2_w
        z += h
        
        v_list.append(v)
        w_list.append(w)
        z_list.append(z)
        
        stop = 0
        
        # run until w small. If w goes negative, remove final entry from list
        
        while w > limit and stop < 100/h: # smaller h requires more iterations, so limit should scale
    
            k1_v = h*v_function(z, w, n)
            k1_w = h*w_function(z, v, n)
            
            k2_v = h*v_function(z+0.5*h, w+0.5*k1_w, n)
            k2_w = h*w_function(z+0.5*h, v+0.5*k1_v, n)
            
            v += k2_v
            w += k2_w
            z += h
            
            v_list.append(v)
            w_list.append(w)
            z_list.append(z)
        
            stop += 1
        
        
        # turning lsits into arrays
        v = np.array(v_list)
        w = np.array(w_list)
        z = np.array(z_list)
        
        return Polytrope(h, n, v, w, z, limit, stop)
    
    @staticmethod
    def RK2_special(h, n, limit):
        # this function is intended for n>=5, where z_n=inf. it makes a variable
        # substitution z=y/(1-y), where y has integration bounds 0 to 1. This substitution
        # adds a 1/(1-y)^2 when subbing dz for dy. h corresponds to y and not z in this case
        # these considerations are applied to w and v special defined above
        
        v = 0.0
        w = 1.0
        y = 0.0/(1-0.0)
        
        # want to save values of w, v, z into lists for plotting
        v_list = [0.0]
        w_list = [1.0]
        z_list = [0.0] # still calling this z_list for simplicity, though it contains y
        
        k1_v = h*v # technically has an inf_int_scaling of 1
        k1_w = h*w

        
        k2_v = h*v_function_special(y+0.5*h, w+0.5*k1_w, n)
        k2_w = h*w_function_special(y+0.5*h, v+0.5*k1_v, n)
        
        v += k2_v
        w += k2_w
        y += h
        
        v_list.append(v)
        w_list.append(w)
        z_list.append(y)
        
        stop = 0
        
        # run until w small. If w goes negative, remove final entry from list
        
        while w > limit and stop < 100/h: # smaller h requires more iterations, so limit should scale
            
            k1_v = h*v_function_special(y, w, n)
            k1_w = h*w_function_special(y, v, n)
            
            k2_v = h*v_function_special(y+0.5*h, w+0.5*k1_w, n)
            k2_w = h*w_function_special(y+0.5*h, v+0.5*k1_v, n)
            
            v += k2_v
            w += k2_w
            y += h
            
            v_list.append(v)
            w_list.append(w)
            z_list.append(y)
        
            stop += 1
        
        
        # turning lsits into arrays
        v = np.array(v_list)
        w = np.array(w_list)
        z = np.array(z_list)
        
        return Polytrope(h, n, v, w, z, limit, stop)
    
    
    
    # will need avg rho/rho_c
    
    # (19.20) rho_avg/rho_c = (-3/z)dw/dz at z=z_n
    # or, with dw/dz = v/z^2: rho_avg/rho_c = (-3v/z^3) at z=z_n
    
    # interested in rho_c/rho_avg, inverse of above value. 

    def rho_ratio(self):
        
        # since interested in comparing with table 19.1, which computer z_n to 
        # 5 decimal places, use 5 decimal places in this calculation
        
        # round num and den as well, want to try and emulate rounding of 
        # original calculation in table 19.1
        
        v = self.v[-1]
        z = self.z[-1]
        
        num = z**3
        den = -3*v
        
        self.rho_ratio = num/den
        

    
    # will need to be printing a lot of data for each polytrope, make a function
    
    def printer(self):
        
        # round z_n and the density ratio to 5 decimal places to match table 19.1
        z_round = round(self.z[-1], 5)
        rho_ratio_round = round(self.rho_ratio, 5) # this value should be 5 decimal points already
        
        print(f'for n={self.n}, z_n={z_round} and rho_ratio={rho_ratio_round}')


# distinguish between the class instances by their n

# first, a sanity check. for n=1, w=sin(z)/z, and z_1 = pi.
Polytrope1_test = Polytrope.RK2(0.01, 1, 1e-6)

plt.figure()
plt.title('n=1: sanity check plot with analytic solution')
plt.plot(Polytrope1_test.z, Polytrope1_test.w, color='black', linewidth=3, label='w, n=1')
plt.plot(Polytrope1_test.z[1:], np.sin(Polytrope1_test.z[1:])/Polytrope1_test.z[1:], 
         linestyle='dotted', color='red', linewidth=3, label='sin(z)/z')
plt.plot([np.pi, np.pi], [-2, 2], linestyle='dotted', color='green', linewidth=3, label='expected z_1')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()


#%%

# next, want to verify h is sufficiently small. do this by running the sanity check
# for smaller and smaller h, until the difference between z_1[-1] and pi is small.
# define small as agreeing to first 5 decimal places rounded, i.e. > 1e-6
h = 0.01
while abs(round(np.pi, 5) - Polytrope1_test.z[-1]) > 1e-5:
    h = h/10
    Polytrope1_test = Polytrope.RK2(h, 1, 1e-6)

print('\n')
print(f'for n=1, I find that h={h} is small enough for the difference between ')
print('z_1[-1] and pi to be less than 1e-5, accurate to 5 decimal places.')
print('going to use this h going forward.')
print('\n')



# next, want to find z_n and avg rho/rho_c for n corresponding to table 19.1

# define corresponding polytropes, calculate and print values

Polytrope0 = Polytrope.RK2(h, 0.0, 1e-6)
Polytrope0.rho_ratio()
Polytrope0.printer()

Polytrope1 = Polytrope.RK2(h, 1.0, 1e-6)
Polytrope1.rho_ratio()
Polytrope1.printer()

Polytrope1_5 = Polytrope.RK2(h, 1.5, 1e-6)
Polytrope1_5.rho_ratio()
Polytrope1_5.printer()

Polytrope2 = Polytrope.RK2(h, 2.0, 1e-6)
Polytrope2.rho_ratio()
Polytrope2.printer()

print('\nnote previous limit is not accurate to 5 decimal places for n>2, so make it smaller\n')

Polytrope3 = Polytrope.RK2(h, 3.0, 1e-7)
Polytrope3.rho_ratio()
Polytrope3.printer()

Polytrope4 = Polytrope.RK2(h, 4.0, 1e-8)
Polytrope4.rho_ratio()
Polytrope4.printer()

Polytrope4_5 = Polytrope.RK2(h, 4.5, 1e-8)
Polytrope4_5.rho_ratio()
Polytrope4_5.printer()

#%%
Polytrope5 = Polytrope.RK2_special(h, 5.0, 1e-8)
# here z is actually y so need to convert to z using z=y/(1-y)
y_5 = round(Polytrope5.z[-1], 5)
z_5 = y_5/(1 - y_5)
v_inv = round(Polytrope5.v[-1]**-1, 5)
print(f'for n=5.0, z_n={z_5} and since rho ratio is prop to z^3 and v^-1={v_inv}, rho_ratio is infinity also.')

#%%

plt.figure()
plt.title(f'polytropes with h={h}')
plt.plot(Polytrope1.z, Polytrope1.w, color='black', linewidth=3, label='n=1')
plt.plot(Polytrope1_5.z, Polytrope1_5.w, color='red', linewidth=3, label='n=3/2')
plt.plot(Polytrope3.z, Polytrope3.w, color='green', linewidth=3, label='n=3')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()




    