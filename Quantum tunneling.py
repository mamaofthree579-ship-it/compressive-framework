import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import solve
import time

st.set_page_config(layout="wide")
st.title("Quantum Tunneling Research Platform v2")

# ==========================================================
# GLOBAL CONSTANTS
# ==========================================================
hbar = 1.0
m = 1.0

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================
st.sidebar.header("Physics Mode")

mode = st.sidebar.selectbox(
    "Simulation Mode",
    ["1D Tunneling",
     "Numerical Heatmap",
     "2D Tunneling",
     "Periodic Lattice",
     "Curved Metric"]
)

# Shared parameters
V0 = st.sidebar.slider("Barrier Height V0", 0.1, 10.0, 5.0, 0.1)
a = st.sidebar.slider("Barrier Width a", 0.1, 5.0, 1.0, 0.1)

# ==========================================================
# 1D TUNNELING (CRANK–NICOLSON)
# ==========================================================
if mode == "1D Tunneling":

    N = 600
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]

    V = np.zeros(N)
    V[(x >= 0) & (x <= a)] = V0

    # Absorbing boundaries
    absorber = np.zeros(N)
    edge = 8
    absorber[np.abs(x) > edge] = 0.02*(np.abs(x[np.abs(x)>edge])-edge)**2
    V_complex = V - 1j*absorber

    # Hamiltonian
    diag = np.ones(N)*(-2)
    off = np.ones(N-1)
    lap = (np.diag(diag) + np.diag(off,1) + np.diag(off,-1))/dx**2
    H = -(hbar**2)/(2*m)*lap + np.diag(V_complex)

    dt = 0.002
    I = np.identity(N)
    A = I + 1j*dt*H/2
    B = I - 1j*dt*H/2

    # Initial packet
    x0 = -6
    sigma = 0.7
    k0 = 3.0

    psi = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
    psi /= np.sqrt(np.sum(np.abs(psi)**2)*dx)

    st.subheader("Wavepacket Evolution")

    placeholder = st.empty()

    for t in range(250):
        psi = solve(A, B @ psi)
        psi /= np.sqrt(np.sum(np.abs(psi)**2)*dx)

        if t % 5 == 0:
            fig, ax = plt.subplots()
            ax.plot(x, np.abs(psi)**2)
            ax.plot(x, V/np.max(V0+1e-6)*2 - 0.5, '--')
            ax.set_ylim(0,2)
            placeholder.pyplot(fig)

    T = np.sum(np.abs(psi[x>5])**2)*dx
    st.markdown(f"### Transmission ≈ {T:.5f}")

# ==========================================================
# NUMERICAL TRANSMISSION HEATMAP
# ==========================================================
elif mode == "Numerical Heatmap":

    energies = np.linspace(0.5, 8, 30)
    widths = np.linspace(0.5, 3.0, 30)

    heatmap = np.zeros((len(energies), len(widths)))

    for i,E in enumerate(energies):
        for j,width in enumerate(widths):
            kappa = np.sqrt(max(2*m*(V0-E),0))
            heatmap[i,j] = np.exp(-2*kappa*width)

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap,
                   extent=[widths.min(), widths.max(),
                           energies.min(), energies.max()],
                   origin="lower",
                   aspect="auto")
    fig.colorbar(im,label="Transmission")
    ax.set_xlabel("Barrier Width")
    ax.set_ylabel("Energy")
    st.pyplot(fig)

    df = pd.DataFrame(heatmap)
    st.download_button("Download CSV",
                       df.to_csv().encode(),
                       "heatmap.csv",
                       "text/csv")

# ==========================================================
# 2D TUNNELING
# ==========================================================
elif mode == "2D Tunneling":

    N = 120
    x = np.linspace(-5,5,N)
    y = np.linspace(-5,5,N)
    dx = x[1]-x[0]

    X,Y = np.meshgrid(x,y)
    V = np.zeros_like(X)
    V[np.abs(X)<a] = V0

    psi = np.exp(-((X+3)**2 + Y**2))
    psi /= np.sqrt(np.sum(np.abs(psi)**2)*dx*dx)

    st.subheader("2D Barrier")

    fig, ax = plt.subplots()
    c = ax.imshow(V, extent=[-5,5,-5,5])
    fig.colorbar(c)
    st.pyplot(fig)

# ==========================================================
# PERIODIC LATTICE (KRONIG-PENNEY STYLE)
# ==========================================================
elif mode == "Periodic Lattice":

    N = 600
    x = np.linspace(-10,10,N)
    dx = x[1]-x[0]

    lattice_spacing = 2.0
    V = V0*(np.cos(2*np.pi*x/lattice_spacing)>0)

    fig, ax = plt.subplots()
    ax.plot(x,V)
    ax.set_title("Periodic Potential")
    st.pyplot(fig)

# ==========================================================
# CURVED METRIC MODIFICATION
# ==========================================================
elif mode == "Curved Metric":

    N = 600
    x = np.linspace(-10,10,N)
    dx = x[1]-x[0]

    curvature = st.sidebar.slider("Curvature Strength",0.0,5.0,1.0,0.1)

    metric = 1/(1+curvature*np.exp(-x**2))
    V = V0*np.exp(-x**2)

    fig, ax = plt.subplots()
    ax.plot(x,metric,label="Metric Factor")
    ax.plot(x,V,label="Potential")
    ax.legend()
    st.pyplot(fig)
