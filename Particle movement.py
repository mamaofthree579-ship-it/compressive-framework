import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
lambda_val = 1.0
C0 = 1.0
G = 1.0

def V(C):
    return 0.25 * lambda_val * (C**2 - C0**2)**2

def dV_dC(C):
    return lambda_val * C * (C**2 - C0**2)

def equations(t, y):
    C, Cdot = y
    rho = 0.5 * Cdot**2 + V(C)
    H = np.sqrt((8*np.pi*G/3) * rho)
    Cddot = -3*H*Cdot - dV_dC(C)
    return [Cdot, Cddot]

# Initial conditions
y0 = [2.0, 0.0]

sol = solve_ivp(equations, [0, 10], y0, t_eval=np.linspace(0,10,1000))

plt.plot(sol.t, sol.y[0])
plt.xlabel("Time")
plt.ylabel("C(t)")
plt.title("Compression Field Evolution")
plt.show()
