import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")
st.title("Advanced Quantum Tunneling Simulator")

# ─────────────────────────────────────────────
# Constants (natural units)
# ─────────────────────────────────────────────
hbar = 1.0
m = 1.0

# ─────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────
st.sidebar.header("Barrier Controls")

V0 = st.sidebar.slider("Barrier Height (V0)", 0.1, 10.0, 5.0, 0.1)
a = st.sidebar.slider("Barrier Width (a)", 0.1, 5.0, 1.0, 0.1)
E_target = st.sidebar.slider("Target Energy (Eigen Solver)", 0.1, 9.9, 2.0, 0.1)

barrier_type = st.sidebar.selectbox("Barrier Type",
                                     ["Single Barrier", "Double Barrier"])

st.sidebar.header("Gravitational Modification")

alpha = st.sidebar.slider("Coupling Strength (α)", 0.0, 5.0, 0.0, 0.1)
beta = st.sidebar.slider("Curvature Decay (β)", 0.1, 5.0, 1.0, 0.1)

# ─────────────────────────────────────────────
# Spatial Grid
# ─────────────────────────────────────────────
N = 600
x = np.linspace(-5, 5, N)
dx = x[1] - x[0]

# ─────────────────────────────────────────────
# Potential Definition
# ─────────────────────────────────────────────
V = np.zeros(N)

if barrier_type == "Single Barrier":
    V[(x >= 0) & (x <= a)] = V0
else:
    d = 1.5
    V[(x >= 0) & (x <= a)] = V0
    V[(x >= a + d) & (x <= 2*a + d)] = V0

# Gravitational modification
V += alpha * np.exp(-beta * np.abs(x))

# ─────────────────────────────────────────────
# Hamiltonian Construction
# ─────────────────────────────────────────────
diag = np.ones(N) * (1/dx**2)
offdiag = np.ones(N-1) * (-0.5/dx**2)

H = np.diag(2*diag + V) \
    + np.diag(offdiag, 1) \
    + np.diag(offdiag, -1)

# ─────────────────────────────────────────────
# Eigenvalue Solution
# ─────────────────────────────────────────────
eigvals, eigvecs = np.linalg.eig(H)

idx = np.argmin(np.abs(eigvals.real - E_target))
psi = eigvecs[:, idx]
psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

# ─────────────────────────────────────────────
# Plot Eigenstate
# ─────────────────────────────────────────────
st.subheader("Stationary Schrödinger Solution")

fig1, ax1 = plt.subplots()
ax1.plot(x, np.real(psi), label="Re(ψ)")
ax1.plot(x, V/np.max(V0+1e-6)*2 - 2, '--', label="Potential (scaled)")
ax1.set_title("Eigenstate Near Selected Energy")
ax1.legend()
st.pyplot(fig1)

st.markdown(f"**Closest Eigen Energy:** {eigvals[idx].real:.4f}")

# ─────────────────────────────────────────────
# Time-Dependent Wavepacket
# ─────────────────────────────────────────────
st.subheader("Time-Dependent Wavepacket Tunneling")

animate = st.checkbox("Animate Wavepacket")

if animate:
    placeholder = st.empty()

    x0 = -3
    sigma = 0.5
    k0 = 3.0

    psi_t = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
    psi_t /= np.sqrt(np.sum(np.abs(psi_t)**2) * dx)

    dt = 0.001
    steps = 200

    H_complex = H.astype(complex)

    for i in range(steps):
        psi_t = psi_t - 1j * dt * (H_complex @ psi_t)
        psi_t /= np.sqrt(np.sum(np.abs(psi_t)**2) * dx)

        fig2, ax2 = plt.subplots()
        ax2.plot(x, np.abs(psi_t)**2, label="|ψ|²")
        ax2.plot(x, V/np.max(V0+1e-6)*2 - 0.5, '--', label="Potential")
        ax2.set_ylim(0, 2)
        ax2.set_title("Wavepacket Evolution")
        ax2.legend()

        placeholder.pyplot(fig2)
        time.sleep(0.02)

# ─────────────────────────────────────────────
# Transmission Estimation
# ─────────────────────────────────────────────
st.subheader("Approximate Transmission Measurement")

if animate:
    transmitted_region = x > 2
    T_numeric = np.sum(np.abs(psi_t[transmitted_region])**2) * dx
    st.markdown(f"**Estimated Transmission Probability:** {T_numeric:.5f}")
