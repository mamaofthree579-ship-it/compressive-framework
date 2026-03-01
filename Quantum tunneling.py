import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Quantum Tunneling Simulator (1D Schrödinger)")

# ─────────────────────────────────────────────
# Constants (natural units for simplicity)
# ─────────────────────────────────────────────
hbar = 1
m = 1

# ─────────────────────────────────────────────
# User Controls
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    V0 = st.slider("Barrier Height (V0)", 0.1, 10.0, 5.0, 0.1)

with col2:
    a = st.slider("Barrier Width (a)", 0.1, 5.0, 1.0, 0.1)

with col3:
    E = st.slider("Particle Energy (E)", 0.1, 9.9, 2.0, 0.1)

# ─────────────────────────────────────────────
# Spatial grid
# ─────────────────────────────────────────────
x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]

# Barrier potential
V = np.zeros_like(x)
V[(x >= 0) & (x <= a)] = V0

# ─────────────────────────────────────────────
# Wave numbers
# ─────────────────────────────────────────────
k1 = np.sqrt(2*m*E)/hbar
k2 = np.sqrt(2*m*(V0 - E))/hbar if V0 > E else np.sqrt(2*m*(E - V0))/hbar

# Approximate transmission coefficient
if E < V0:
    T = np.exp(-2 * k2 * a)
else:
    T = 1 / (1 + (V0**2 * np.sin(k2*a)**2)/(4*E*(E - V0)))

# ─────────────────────────────────────────────
# Construct approximate wavefunction
# ─────────────────────────────────────────────
psi = np.zeros_like(x, dtype=complex)

psi[x < 0] = np.exp(1j * k1 * x[x < 0])
psi[(x >= 0) & (x <= a)] = np.exp(-k2 * (x[(x >= 0) & (x <= a)]))
psi[x > a] = np.sqrt(T) * np.exp(1j * k1 * x[x > a])

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, ax = plt.subplots()

ax.plot(x, np.real(psi), label="Re(ψ)")
ax.plot(x, V/np.max(V0)*2 - 3, linestyle="--", label="Barrier (scaled)")

ax.set_title("Wavefunction and Potential Barrier")
ax.set_xlabel("Position x")
ax.set_ylabel("Amplitude")

ax.legend()
st.pyplot(fig)

st.markdown(f"### Transmission Probability T ≈ {T:.5f}")
