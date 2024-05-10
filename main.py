import matplotlib.pyplot as plt
import numpy as np

"""
y0 = 5
k = 3
x = np.linspace(0, 3, 10000)
y = y0*np.exp(-k*x)

fig = plt.figure(figsize = (20, 20))
plt.plot(x, y)
plt.show()
"""

def derive(y, t):
    g = 0
    return (g-y)/tau

def euler(derive, y0, t):
    h = t[1]-t[0]
    N = len(t)
    y = np.zeros(N)
    y[0] = y0
    for n in range(0,N-1):
        y[n+1] = y[n] + h*derive(y[n], t[n])
    return y

# PARAMETRES
tau = 1/3
T = 3
N = 1000

# RESOLUTION DE L'EQUADIFF
y0 = 5                 # Condition initiale
t = np.linspace(0,T,N)    # Tableau du temps
y = euler(derive, y0, t)  # Integration

# COURBES
plt.plot(t, y, ".", ms =2 ,label='y(t) pour N={}'.format(N))
plt.legend()
plt.xlabel('t')
plt.ylim(0,12)
plt.ylabel('y')
plt.grid()
plt.show()
