import numpy as np

# SHARED SOURCE CODE — same numbers for all scales
lambda_shared = 0.8
G_shared = 0.3
gamma_shared = 0.08
C0, C_init = 1.0, 2.0

t = np.linspace(0, 10, 1000)
dt = t[1]-t[0]

# 1. Bio envelope (gut*heart*brain)
f_gut, f_heart, f_brain = 0.12, 1.1, 38.0
K = lambda_shared
drive = 1 + K*np.sin(2*np.pi*0.2*t)
gut = np.exp(-gamma_shared*t)*drive*np.exp(1j*2*np.pi*f_gut*t)
heart = np.exp(-gamma_shared*0.5*t)*drive*np.exp(1j*2*np.pi*f_heart*t)
brain = np.exp(-gamma_shared*t)*drive*np.exp(1j*2*np.pi*f_brain*t)
env_norm = np.abs(gut*heart*brain)
env_norm /= env_norm.max()

# 2. Compression field C(t)
C, Cd = np.zeros_like(t), np.zeros_like(t)
C[0], Cd[0] = C_init, 0.0
def dVdC(Cv): return lambda_shared*Cv*(Cv**2 - C0**2)
for i in range(1, len(t)):
    rho = 0.5*Cd[i-1]**2 + 0.25*lambda_shared*((C[i-1]**2 - C0**2)**2)
    H = np.sqrt(8*np.pi*G_shared/3 * max(rho, 1e-8))
    Cdd = -3*H*Cd[i-1] - dVdC(C[i-1])
    Cd[i] = Cd[i-1] + Cdd*dt
    C[i] = C[i-1] + Cd[i]*dt
C_norm = (C - C0)/(C_init - C0)

# 3. Correlations
corr = np.corrcoef(env_norm, C_norm)[0,1]
print(f"Correlation Bio vs Cosmic: {corr:.3f}")
print(f"Peak bio at {t[np.argmax(env_norm)]:.2f}s, C settles near {t[np.where(C<=1.05)[0][0]]:.2f}s")
