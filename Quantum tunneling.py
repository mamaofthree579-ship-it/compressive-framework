import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Wavefunction Visualization in Double Barrier")

hbar = 1.0
m = 1.0

# Controls
V0 = st.slider("Barrier Height V0", 0.5, 10.0, 5.0, 0.1)
a = st.slider("Barrier Width", 0.1, 3.0, 1.0, 0.1)
well_width = st.slider("Well Width", 0.1, 5.0, 1.0, 0.1)

E = st.slider("Select Energy", 0.01, V0*2, 1.0, 0.01)

# Spatial grid
x = np.linspace(-3, 3, 2000)

# Define potential
def V(x):
    if -a-well_width/2 <= x <= -well_width/2:
        return V0
    elif well_width/2 <= x <= a+well_width/2:
        return V0
    else:
        return 0

Vx = np.array([V(xi) for xi in x])

# Wave numbers
k1 = np.sqrt(2*m*E + 0j)/hbar
k2 = np.sqrt(2*m*(E - V0) + 0j)/hbar

psi = np.zeros_like(x, dtype=complex)

for i, xi in enumerate(x):
    if xi < -a-well_width/2:
        psi[i] = np.exp(1j*k1*xi) + 0.2*np.exp(-1j*k1*xi)
    elif -a-well_width/2 <= xi <= -well_width/2:
        psi[i] = np.exp(1j*k2*xi)
    elif -well_width/2 <= xi <= well_width/2:
        psi[i] = np.exp(1j*k1*xi)
    elif well_width/2 <= xi <= a+well_width/2:
        psi[i] = np.exp(1j*k2*xi)
    else:
        psi[i] = 0.8*np.exp(1j*k1*xi)

# Plot
fig, ax = plt.subplots()

ax.plot(x, np.real(psi), label="Re(ψ)")
ax.plot(x, np.abs(psi)**2, linestyle="--", label="|ψ|²")

# Shade barriers
ax.fill_between(x, 0, Vx/np.max(Vx), where=Vx>0, alpha=0.2)

ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.legend()

st.pyplot(fig)
