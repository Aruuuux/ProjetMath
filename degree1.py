__author__ = "ANTROPIUS Simon, AUGEY Louis"
__date__ = "2024-05-13"

import matplotlib.pyplot as plt
import numpy as np

##### Definition of the equation ##############################################

def f(x, y):
    """
    Right side of the ODE y'(x) = -a(x)*y(x) + b(x).
    Define the equation here.

    :param x: The value of x.
    :param y: The value of y(x).
    :return: The value of -a(x)*y(x) + b(x).
    """
    return -0.3*y

# Initial conditions
x0 = 0 # Initial value of x
y0 = 5 # Initial value of y (i.e. f(x0))

##### Euler's solution ########################################################

def euler_method(f, x, y0):
    """
    Approximates the solution of the ODE y' = f(x, y) using Euler's method.

    :param f: The function f(x, y) representing the right-hand side of the ODE.
    :param x: The array containing the values of x.
    :param y0: The initial value of y.
    :return: An array containing the approximation of y.
    """
    n = len(x) # Number of steps
    h = x[1] - x[0] # Step size
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        y[i] = y[i-1] + h * f(x[i-1], y[i-1])
    return y

##### Runge-Kutta's solution ##################################################

def runge_kutta_method(f, x, y0):
    """
    Approximates the solution of the ODE y' = f(x, y) using Runge-Kutta's
    method.

    :param f: The function f(x, y) representing the right-hand side of the ODE.
    :param x: The array containing the values of x.
    :param y0: The initial value of y.
    :return: An array containing the approximation of y.
    """
    n = len(x) # Number of steps
    y = np.zeros(n)
    y[0] = y0
    for i in range(1, n):
        h = x[i] - x[i-1]
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1], y[i-1] + k1/2)
        k3 = h * f(x[i-1], y[i-1] + k2/2)
        k4 = h * f(x[i-1], y[i-1] + k3)
        y[i] = y[i-1] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

##### Exact solution ##########################################################

def analytical_solution(x):
    """
    Solution of the ODE y'(x) = -a(x)*y(x) + b(x).
    Define the solution to the equation here.

    :param x: The array containing the values of x.
    :return: An array containing the exact values of y.
    """
    return y0*np.exp(-0.3*x)

##### Error ###################################################################

def error_euler_runge_kutta(f_analytical, f_euler, f_runge_kutta):
    """
    Computes the error between euler and the exact solution and runge-kutta
    and the exact solution.

    :param f_analytical: An array containing the exact values of y.
    :param f_euler: An array containing approximated values of y with euler.
    :param f_runge_kutta: An array containing approximated values of y with
    runge-kutta.
    :return: 2 arrays containing the the error between euler and the exact
    solution and runge-kutta and the exact solution.
    """
    return np.abs(f_euler - f_analytical), np.abs(f_runge_kutta - f_analytical)

##### Computations ############################################################

# Interval
a = x0 # Beginning of the interval
b = 10 # End of the interval
n = 1000  # Number of steps

x = np.linspace(a, b, n)
y_analytical = analytical_solution(x)
y_euler = euler_method(f, x, y0)
y_runge_kutta = runge_kutta_method(f, x, y0)

y_error_euler, y_error_runge_kutta = error_euler_runge_kutta(y_analytical,
                                                             y_euler,
                                                             y_runge_kutta)

##### Figure ##################################################################
plt.subplot(1, 2, 1)
plt.plot(x, y_analytical, label='Exact solution')
plt.plot(x, y_euler, label='Approximated solution (Euler)')
plt.plot(x, y_runge_kutta, label='Approximated solution (Runge-Kutta)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f"Approximated and exact solution of dy/dx = -0.3*y using Euler\'s "
          f"and Runge-Kutta\'s methods")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, y_error_euler, label='Euler')
plt.plot(x, y_error_runge_kutta, label='Runge-Kutta')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Error between Euler/Runge-Kutta methods and the analytical "
          "solution")
plt.legend()

plt.show()