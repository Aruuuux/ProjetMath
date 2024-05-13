import numpy as np
import matplotlib.pyplot as plt

# Parameters

n = 1000
y0 = 5
x0 = 0
t = np.linspace(0, 10, n)
x = np.linspace(0, 10, n)
k = 0.3

# Functions

def f(k, y) :
    '''
    input : k (float) : constant
            y (float) : variable
    
    output : -k * y (float) : derivative of y
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
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    h = 10/n
    for i in range(n-1):
        y[i+1] = y[i] + h * f(k, y[i])
    return y


def runge_kutta_method(t, f, y0):
    '''
    input : t (np.array) : time
            f (function) : derivative function
            y0 (float) : initial value
    output : y (np.array) : numerical solution of the runge-kutta method
    '''
    n = len(t)
    y = np.zeros(n)
    x = np.zeros(n)
    y[0] = y0
    x[0]=x0
    h = 10/n
    for i in range(n-1):
        k1 = h * f(k, y[i])
        k2 = h * f(k + h/2, y[i] + k1/2)
        k3 = h * f(k + h/2, y[i] + k2/2)
        k4 = h * f(k + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        x[i+1] = x[i] + h
    return y

# Compute
analytic_function = f_real(k, y0, x)
y_euler = euler_method(t, f, y0)
y_runge_kutta = runge_kutta_method(t, f, y0)

# Plot
plt.plot(t, analytic_function, label='Analytical')
plt.plot(t, y_euler, label='Euler')
plt.plot(t, y_runge_kutta, label='Runge-Kutta')
plt.title(f"Approximated and exact solutions of the ODE y\' = -{k}*y with y(0) = 5")
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')  
plt.legend()
plt.show()