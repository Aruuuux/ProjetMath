__author__ = "ANTROPIUS Simon, AUGEY Louis"
__date__ = "2024-05-13"

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 100 # number of points
y0 = 5 
x0 = 0
t = np.linspace(0, 10, n) # time
x = np.linspace(0, 10, n) 
k = 0.3 # constant

# Functions
def f(k, y) :
    '''
    input : k (float) : constant
            y (float) : variable
    
    output : -k * y (float) : derived form of y
    '''
    return -k * y

def f_real(k, y0,x) :
    '''
    input : k (float) : constant
            y0 (float) : initial value
            x (float) : variable

    output : y0 * np.exp(-k * x) (float) : analytical solution
    '''
    return y0 * np.exp(-k * x)

def euler_method(t, f, y0):
    '''
    input : t (np.array) : time
            f (function) : derivative function
            y0 (float) : initial value
    output : y (np.array) : numerical solution of the euler method
    '''
    n = len(t) # number of points
    y = np.zeros(n)
    y[0] = y0
    h = 10/n # step size
    for i in range(n-1):
        y[i+1] = y[i] + h * f(k, y[i]) # Euler method
    return y


def runge_kutta_method(f_real, y0, x, k):
    '''
    input : f_real (function) : analytical solution
            y0 (float) : initial value
            x (np.array) : variable
            k (float) : constant
    output : y (np.array) : numerical solution of the runge-kutta method
    '''
    n = len(x)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        h = x[i+1] - x[i]
        k1 = h * f(k, y[i])
        k2 = h * f(k, y[i] + k1/2)
        k3 = h * f(k, y[i] + k2/2)
        k4 = h * f(k, y[i] + k3)
        y[i+1] = y[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

def error_euler_runge_kutta(f_real,f_euler, f_runge_kutta):
    '''
    input : f_euler (np.array) : numerical solution of the euler method
            f_runge_kutta (np.array) : numerical solution of the runge-kutta method
    output : np.abs(f_euler - f_runge_kutta) (np.array) : error between the two methods
    '''
    return np.abs(f_euler - f_real), np.abs(f_runge_kutta - f_real)

# Compute
y_analytic = f_real(k, y0, x)
y_euler = euler_method(x, f, y0)
y_runge_kutta = runge_kutta_method(f_real, y0, x, k)
y_error_euler, y_error_runge_kutta = error_euler_runge_kutta(y_analytic, y_euler, y_runge_kutta)

# Plot
plt.subplot(1,2,1)
plt.plot(t, y_analytic, label='Analytical')
plt.plot(t, y_euler, label='Euler')
plt.plot(t, y_runge_kutta, label='Runge-Kutta')
plt.title(f"Approximated and exact solutions of the ODE y\' = -{k}*y with y(0) = {y0}")
plt.grid()
plt.xlabel('t in seconds')
plt.ylabel('y(t)')  
plt.legend()

plt.subplot(1,2,2)
plt.plot(t, y_error_euler, label='Euler', color='orange')
plt.plot(t, y_error_runge_kutta, label='Runge-Kutta', color='green')
plt.title("Error between Euler and Runge-Kutta methods compare to the analytical solution")
plt.grid()
plt.xlabel('t in seconds')
plt.ylabel('y(t)')
plt.legend()
plt.show()