import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Physically Correct Double Barrier Scattering")

hbar = 1.0
m = 1.0

# --- Controls ---
V0 = st.slider("Barrier Height V0", 0.5, 10.0, 5.0, 0.1)
a = st.slider("Barrier Width", 0.1, 3.0, 1.0, 0.1)
well_width = st.slider("Well Width", 0.1, 5.0, 1.0, 0.1)
E = st.slider("Energy", 0.01, V0*2, 1.0, 0.01)

# Wave numbers
k1 = np.sqrt(2*m*E + 0j)/hbar
k2 = np.sqrt(2*m*(E - V0) + 0j)/hbar

# --- Transfer matrices ---
def interface(k_left, k_right):
    return 0.5 * np.array([
        [1 + k_right/k_left, 1 - k_right/k_left],
        [1 - k_right/k_left, 1 + k_right/k_left]
    ])

def propagation(k, L):
    return np.array([
        [np.exp(1j*k*L), 0],
        [0, np.exp(-1j*k*L)]
    ])

# Build total matrix
M = (
    interface(k1, k2) @
    propagation(k2, a) @
    interface(k2, k1) @
    propagation(k1, well_width) @
    interface(k1, k2) @
    propagation(k2, a) @
    interface(k2, k1)
)

# Reflection and transmission
r = M[1,0] / M[0,0]
t = 1 / M[0,0]
T = np.real((k1/k1) * np.abs(t)**2)

st.write(f"Transmission T = {T:.4f}")

# --- Reconstruct wavefunction ---
x = np.linspace(-3, 3, 2000)
psi = np.zeros_like(x, dtype=complex)

# Region boundaries
x1 = -a - well_width/2
x2 = -well_width/2
x3 = well_width/2
x4 = a + well_width/2

for i, xi in enumerate(x):
    if xi < x1:
        psi[i] = np.exp(1j*k1*xi) + r*np.exp(-1j*k1*xi)
    elif x1 <= xi < x2:
        A = 1
        B = r
        psi[i] = A*np.exp(1j*k2*(xi-x1)) + B*np.exp(-1j*k2*(xi-x1))
    elif x2 <= xi < x3:
        psi[i] = np.exp(1j*k1*(xi-x2)) + r*np.exp(-1j*k1*(xi-x2))
    elif x3 <= xi < x4:
        psi[i] = t*np.exp(1j*k2*(xi-x3))
    else:
        psi[i] = t*np.exp(1j*k1*xi)

# --- Plot ---
fig, ax = plt.subplots()

ax.plot(x, np.real(psi), label="Re(ψ)")
ax.plot(x, np.abs(psi)**2, linestyle="--", label="|ψ|²")

# Draw barriers
ax.axvspan(x1, x2, alpha=0.2)
ax.axvspan(x3, x4, alpha=0.2)

ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.legend()

st.pyplot(fig)
