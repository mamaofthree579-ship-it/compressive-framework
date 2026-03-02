import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.linalg import eigh

st.title("Band Structure Solver (Bloch States)")

# Constants
hbar = 1.0
m = 1.0

# Controls
V0 = st.slider("Lattice Depth", 0.1, 10.0, 5.0, 0.1)
a = st.slider("Lattice Spacing", 0.5, 5.0, 2.0, 0.1)
Nk = st.slider("k-Points", 20, 200, 80)

# Spatial grid (one unit cell)
N = 300
x = np.linspace(0, a, N, endpoint=False)
dx = x[1] - x[0]

# Periodic potential (cosine lattice)
V = V0 * np.cos(2*np.pi*x/a)

# k-space
k_vals = np.linspace(-np.pi/a, np.pi/a, Nk)
bands = []

for k in k_vals:

    # Laplacian
    diag = np.ones(N)*(-2)
    off = np.ones(N-1)

    lap = (np.diag(diag) + np.diag(off,1) + np.diag(off,-1)) / dx**2

    # Bloch boundary condition
    lap[0,-1] = np.exp(-1j*k*a)/dx**2
    lap[-1,0] = np.exp(1j*k*a)/dx**2

    H = -(hbar**2)/(2*m)*lap + np.diag(V)

    eigvals = eigh(H, eigvals_only=True)
    bands.append(eigvals[:5])  # first 5 bands

bands = np.array(bands)

# Plot
fig, ax = plt.subplots()

for n in range(5):
    ax.plot(k_vals, bands[:,n])

ax.set_xlabel("k")
ax.set_ylabel("Energy E(k)")
ax.set_title("Band Structure")
st.pyplot(fig)
