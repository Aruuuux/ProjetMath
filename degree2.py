__author__ = "ANTROPIUS Simon, AUGEY Louis"
__date__ = "2024-05-13"

import matplotlib.pyplot as plt
import numpy as np

##### Definition of the equation ##############################################

# Euler method works for 1 degree ODE only. If we have to solve equations
# with a degree > 1, we need to decouple them. Here with a degree 2:

# y''(x) = f(x, y, y')
# -> with z(x) = y'(x):
# y'(x) = z(x) and
# z'(x) = f(x, y, z)
# We solve each one with Euler's method

def f(x, y, z):
    """
    Right side of the ODE y''(x) = -a(x)*y'(x) + b(x)y(x) + c(x).
    Define the equation here.

    :param x: The value of x.
    :param y: The value of y(x).
    :param z: The value of y'(x).
    :return: The value of -a(x)*y'(x) + b(x)y(x) + c(x).
    """
    g = 9.81
    l = 0.5
    return -(g/l)*np.sin(y)

# Initial conditions
x0 = 0 # Initial value of x
y0 = np.pi/6 # Initial value of y (i.e. f(x0))
z0 = np.pi/4 # Initial value of y' (i.e. f'(x0))

##### Euler's solution ########################################################

def euler_method_second_order(f, x, y0, z0):
    """
    Approximates the solution of the second-order ODE y'' = f(x, y,
    y') using Euler's method.

    :param f: The function f(x, y, y') representing the right-hand side of
    the second-order ODE.
    :param x: The array containing the values of x.
    :param y0: The initial value of y.
    :param z0: The initial value of dy/dx.
    :return: An array containing the approximation of y.
    """
    n = len(x)  # Number of steps
    h = x[1] - x[0]  # Step size
    y = np.zeros(n)
    z = np.zeros(n)
    y[0] = y0
    z[0] = z0
    for i in range(1, n):
        z[i] = z[i - 1] + h * f(x[i - 1], y[i - 1], z[i - 1])
        y[i] = y[i-1] + h * z[i]
    return y

##### Exact solution ##########################################################

def analytical_solution(x):
    """

    Solution of the ODE y''(x) = -a(x)*y'(x) + b(x)y(x) + c(x).
    Define the solution to the equation here.

    :param x: The array containing the values of x.
    :return: An array containing the exact values of y.
    """
    return y0*np.cos(np.sqrt(9.81/0.5)*x+np.arcsin(1/(y0*np.sqrt(9.81/0.5))))

##### Computations ############################################################

# Interval
a = x0 # Beginning of the interval
b = 10 # End of the interval
n = 1000  # Number of steps

x = np.linspace(a, b, n)
y_analytical = analytical_solution(x)
y_euler = euler_method_second_order(f, x, y0, z0)

##### Figure ##################################################################

plt.plot(x, y_analytical, label='Exact solution')
plt.plot(x, y_euler, label='Approximated solution (Euler)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f"Approximated and exact solution of d^2y/dx^2 = -(g/l)*sin(y)using "
          f"Euler\'s Method")
plt.legend()
plt.grid(True)

plt.show()