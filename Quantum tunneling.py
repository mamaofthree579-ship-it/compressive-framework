import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Finite Barrier Quantum Tunneling Simulator")

hbar = 1.0
m = 1.0

# --- Controls ---
V0 = st.slider("Barrier Height V0", 0.5, 10.0, 5.0, 0.1)
a = st.slider("Barrier Width", 0.1, 5.0, 1.0, 0.1)
well_width = st.slider("Well Width (for double barrier)", 0.1, 5.0, 1.0, 0.1)
double_barrier = st.checkbox("Enable Double Barrier (Resonant Tunneling)", True)

E_vals = np.linspace(0.01, V0*2, 400)

def k(E, V):
    return np.sqrt(2*m*(E - V) + 0j) / hbar

def transfer_matrix(E):
    if not double_barrier:
        # Single barrier
        k1 = k(E, 0)
        k2 = k(E, V0)
        
        M11 = np.cos(k2*a) + 0.5j*(k2/k1 + k1/k2)*np.sin(k2*a)
        M12 = 0.5j*(k2/k1 - k1/k2)*np.sin(k2*a)
        M21 = M12
        M22 = np.cos(k2*a) - 0.5j*(k2/k1 + k1/k2)*np.sin(k2*a)
        
        M = np.array([[M11, M12],[M21, M22]])
    else:
        # Double barrier (barrier-well-barrier)
        k1 = k(E, 0)
        k2 = k(E, V0)
        
        # Barrier matrix
        Mb = np.array([
            [np.cos(k2*a), 1j*np.sin(k2*a)/k2],
            [1j*k2*np.sin(k2*a), np.cos(k2*a)]
        ])
        
        # Well matrix
        Mw = np.array([
            [np.cos(k1*well_width), 1j*np.sin(k1*well_width)/k1],
            [1j*k1*np.sin(k1*well_width), np.cos(k1*well_width)]
        ])
        
        M = Mb @ Mw @ Mb

    return M

T_vals = []

for E in E_vals:
    M = transfer_matrix(E)
    T = 1 / np.abs(M[0,0])**2
    T_vals.append(np.real(T))

T_vals = np.array(T_vals)

# --- Plot ---
fig, ax = plt.subplots()
ax.plot(E_vals, T_vals)
ax.set_xlabel("Energy E")
ax.set_ylabel("Transmission T(E)")
ax.set_title("Quantum Tunneling Transmission Spectrum")
ax.set_ylim(0,1.1)
st.pyplot(fig)
