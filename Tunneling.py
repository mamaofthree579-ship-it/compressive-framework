import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha, lam, beta = 0.1, 0.5, 0.4
omega_GR = 0.3737 - 0.08896j
def cgup_freqs(alpha, lam, N=4):
    return [omega_GR + (alpha**2)*omega_GR*(lam**n + beta*lam**(n+1)) for n in range(N)]

t = np.linspace(0, 0.2, 2000)
h = np.zeros_like(t)
for w in cgup_freqs(alpha, lam):
    h += np.exp(w.imag*t)*np.cos(w.real*t)

plt.figure()
plt.plot(t, h)
plt.xlabel('Time (s)')
plt.ylabel('Strain (arb.)')
plt.title('CGUP ringdown toy waveform')
plt.savefig('cgup_waveform.png')
