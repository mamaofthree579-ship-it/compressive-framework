import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Quantum Double-Barrier Tunneling Lab")

# -----------------------
# Constants
# -----------------------
hbar = 1.0
m = st.slider("Particle Mass (m)", 0.5, 5.0, 1.0, 0.1)

# -----------------------
# Barrier + Well Parameters
# -----------------------
V0 = st.slider("Barrier Height V0", 1.0, 12.0, 6.0, 0.1)
a = st.slider("Barrier Width (a)", 0.1, 4.0, 2.0, 0.1)
well_width = st.slider("Well Width", 0.5, 6.0, 3.0, 0.1)

E = st.slider("Energy (E)", 0.01, V0*1.5, 2.0, 0.01)

# -----------------------
# Derived Quantities
# -----------------------
k = np.sqrt(2*m*E + 0j)/hbar

if E < V0:
    kappa = np.sqrt(2*m*(V0 - E) + 0j)/hbar
    barrier_strength = np.real(kappa * a)
else:
    kappa = 0
    barrier_strength = 0

st.write(f"Barrier Strength κa = {barrier_strength:.4f}")

# -----------------------
# Transfer Matrix Functions
# -----------------------
def interface(kL, kR):
    return 0.5*np.array([
        [1 + kR/kL, 1 - kR/kL],
        [1 - kR/kL, 1 + kR/kL]
    ])

def propagation(k, L):
    return np.array([
        [np.exp(1j*k*L), 0],
        [0, np.exp(-1j*k*L)]
    ])

# Determine k2
if E < V0:
    k2 = 1j*kappa
else:
    k2 = np.sqrt(2*m*(E - V0) + 0j)/hbar

# -----------------------
# Build Total Transfer Matrix
# -----------------------
M = (
    interface(k, k2) @
    propagation(k2, a) @
    interface(k2, k) @
    propagation(k, well_width) @
    interface(k, k2) @
    propagation(k2, a) @
    interface(k2, k)
)

t = 1/M[0,0]
r = M[1,0]/M[0,0]
T = np.real(np.abs(t)**2)

st.write(f"Transmission T = {T:.8f}")

# -----------------------
# Wavefunction Construction
# -----------------------
x = np.linspace(-8, 8, 5000)
psi = np.zeros_like(x, dtype=complex)

x1 = -a - well_width/2
x2 = -well_width/2
x3 = well_width/2
x4 = a + well_width/2

for i, xi in enumerate(x):
    if xi < x1:
        psi[i] = np.exp(1j*k*xi) + r*np.exp(-1j*k*xi)
    elif x1 <= xi < x2:
        if E < V0:
            psi[i] = np.exp(-kappa*(xi-x1))
        else:
            psi[i] = np.exp(1j*k2*(xi-x1))
    elif x2 <= xi < x3:
        psi[i] = np.exp(1j*k*(xi-x2))
    elif x3 <= xi < x4:
        if E < V0:
            psi[i] = np.exp(-kappa*(xi-x3))
        else:
            psi[i] = np.exp(1j*k2*(xi-x3))
    else:
        psi[i] = t*np.exp(1j*k*xi)

# -----------------------
# Plot Wavefunction
# -----------------------
fig1, ax1 = plt.subplots()
ax1.plot(x, np.abs(psi)**2)
ax1.axvspan(x1, x2, alpha=0.2)
ax1.axvspan(x3, x4, alpha=0.2)
ax1.set_xlabel("x")
ax1.set_ylabel("|ψ|²")
ax1.set_title("Probability Density")
st.pyplot(fig1)

# -----------------------
# Log Scale Plot
# -----------------------
fig2, ax2 = plt.subplots()
ax2.plot(x, np.log10(np.abs(psi)**2 + 1e-15))
ax2.axvspan(x1, x2, alpha=0.2)
ax2.axvspan(x3, x4, alpha=0.2)
ax2.set_xlabel("x")
ax2.set_ylabel("log10(|ψ|²)")
ax2.set_title("Log Scale (Exponential Decay View)")
st.pyplot(fig2)

# ============================================================
# Automatic Resonance Finder
# ============================================================

st.subheader("Automatic Resonance Finder")
scan = st.checkbox("Scan Transmission vs Energy")

if scan:
    energies = np.linspace(0.01, V0*0.99, 800)
    transmissions = []

    for Es in energies:
        k_scan = np.sqrt(2*m*Es + 0j)/hbar

        if Es < V0:
            kappa_scan = np.sqrt(2*m*(V0-Es) + 0j)/hbar
            k2_scan = 1j*kappa_scan
        else:
            k2_scan = np.sqrt(2*m*(Es-V0) + 0j)/hbar

        M_scan = (
            interface(k_scan, k2_scan) @
            propagation(k2_scan, a) @
            interface(k2_scan, k_scan) @
            propagation(k_scan, well_width) @
            interface(k_scan, k2_scan) @
            propagation(k2_scan, a) @
            interface(k2_scan, k_scan)
        )

        t_scan = 1/M_scan[0,0]
        T_scan = np.real(np.abs(t_scan)**2)
        transmissions.append(T_scan)

    transmissions = np.array(transmissions)

    fig3, ax3 = plt.subplots()
    ax3.plot(energies, transmissions, label="T(E)")

    # Detect peaks
    peak_indices = np.where(
        (transmissions[1:-1] > transmissions[:-2]) &
        (transmissions[1:-1] > transmissions[2:]) &
        (transmissions[1:-1] > 0.01)
    )[0] + 1

    for idx in peak_indices:
        ax3.axvline(energies[idx], linestyle="--")

    # Predicted bound state energies
    n_max = 6
    for n in range(1, n_max+1):
        E_n = (n*np.pi)**2 / (2*m*well_width**2)
        if E_n < V0:
            ax3.axvline(E_n, linestyle=":", alpha=0.7)

    ax3.set_xlabel("Energy")
    ax3.set_ylabel("Transmission")
    ax3.set_title("Transmission Spectrum with Resonances")
    ax3.legend()

    st.pyplot(fig3)

    if len(peak_indices) > 0:
        st.write("Detected Resonance Energies:")
        for idx in peak_indices:
            st.write(f"E ≈ {energies[idx]:.5f}")
    else:
        st.write("No strong resonances detected.")
