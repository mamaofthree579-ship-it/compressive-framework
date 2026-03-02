import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import solve
import time

st.set_page_config(layout="wide")
st.title("Quantum Tunneling Research Laboratory")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
hbar = 1.0
m = 1.0

# ─────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────
st.sidebar.header("Barrier Settings")

V0 = st.sidebar.slider("Barrier Height V0", 0.1, 10.0, 5.0, 0.1)
a = st.sidebar.slider("Barrier Width a", 0.1, 5.0, 1.0, 0.1)
barrier_type = st.sidebar.selectbox("Barrier Type",
                                    ["Single", "Double"])

st.sidebar.header("Curvature Term")

alpha = st.sidebar.slider("α (Coupling)", 0.0, 5.0, 0.0, 0.1)
beta = st.sidebar.slider("β (Decay)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.header("Simulation Controls")

dt = st.sidebar.slider("Time Step", 0.0005, 0.01, 0.002, 0.0005)
steps = st.sidebar.slider("Time Steps", 100, 600, 300, 50)

# ─────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────
N = 800
x = np.linspace(-10, 10, N)
dx = x[1] - x[0]

# ─────────────────────────────────────────────
# Potential
# ─────────────────────────────────────────────
V = np.zeros(N)

if barrier_type == "Single":
    V[(x >= 0) & (x <= a)] = V0
else:
    d = 1.5
    V[(x >= 0) & (x <= a)] = V0
    V[(x >= a + d) & (x <= 2*a + d)] = V0

# Gravitational curvature modification
V += alpha * np.exp(-beta * np.abs(x))

# ─────────────────────────────────────────────
# Absorbing Boundary (Complex Absorber)
# ─────────────────────────────────────────────
absorb_strength = 0.02
edge = 8
absorber = np.zeros(N)
absorber[np.abs(x) > edge] = absorb_strength * (np.abs(x[np.abs(x) > edge]) - edge)**2
V_complex = V - 1j * absorber

# ─────────────────────────────────────────────
# Hamiltonian (finite difference)
# ─────────────────────────────────────────────
diag = np.ones(N) * (-2)
off = np.ones(N-1)

laplacian = (np.diag(diag) + np.diag(off,1) + np.diag(off,-1)) / dx**2
H = -(hbar**2)/(2*m) * laplacian + np.diag(V_complex)

# ─────────────────────────────────────────────
# Crank–Nicolson Setup
# ─────────────────────────────────────────────
I = np.identity(N)
A = I + 1j*dt*H/2
B = I - 1j*dt*H/2

# ─────────────────────────────────────────────
# Initial Wavepacket
# ─────────────────────────────────────────────
x0 = -6
sigma = 0.7
k0 = 3.0

psi = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

# ─────────────────────────────────────────────
# Time Evolution
# ─────────────────────────────────────────────
st.subheader("Wavepacket Evolution (Crank–Nicolson)")

placeholder = st.empty()

for t in range(steps):
    psi = solve(A, B @ psi)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    if t % 5 == 0:
        fig, ax = plt.subplots()
        ax.plot(x, np.abs(psi)**2, label="|ψ|²")
        ax.plot(x, V/np.max(V0+1e-6)*2 - 0.5, '--', label="Barrier")
        ax.set_ylim(0, 2)
        ax.set_title("Quantum Tunneling Evolution")
        ax.legend()
        placeholder.pyplot(fig)

# ─────────────────────────────────────────────
# Transmission (flux-based)
# ─────────────────────────────────────────────
right_region = x > 5
T = np.sum(np.abs(psi[right_region])**2) * dx

st.markdown(f"### Transmission Probability ≈ {T:.5f}")

# ─────────────────────────────────────────────
# Heatmap Generator
# ─────────────────────────────────────────────
st.subheader("Transmission vs Energy Heatmap")

generate_map = st.checkbox("Generate Heatmap")

if generate_map:
    energies = np.linspace(0.5, 8, 40)
    widths = np.linspace(0.5, 3.0, 40)

    heatmap = np.zeros((len(energies), len(widths)))

    for i, E in enumerate(energies):
        for j, width in enumerate(widths):
            kappa = np.sqrt(max(2*m*(V0-E),0))
            heatmap[i,j] = np.exp(-2*kappa*width)

    fig2, ax2 = plt.subplots()
    c = ax2.imshow(heatmap,
                   extent=[widths.min(), widths.max(),
                           energies.min(), energies.max()],
                   origin='lower',
                   aspect='auto')
    fig2.colorbar(c, label="Transmission")
    ax2.set_xlabel("Barrier Width")
    ax2.set_ylabel("Energy")
    ax2.set_title("Tunneling Probability Map")
    st.pyplot(fig2)

    # Export
    df = pd.DataFrame(heatmap,
                      index=np.round(energies,2),
                      columns=np.round(widths,2))
    csv = df.to_csv().encode()
    st.download_button("Download Heatmap CSV", csv, "tunneling_map.csv", "text/csv")
