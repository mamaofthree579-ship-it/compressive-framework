import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Quantum Tunneling Lab – Clear Diagnostic Version")

hbar = 1.0
m = st.slider("Particle Mass (m)", 0.5, 5.0, 1.0, 0.1)

# --- Barrier parameters ---
V0 = st.slider("Barrier Height V0", 1.0, 10.0, 6.0, 0.1)
a = st.slider("Barrier Width (a)", 0.1, 4.0, 2.0, 0.1)
well_width = st.slider("Well Width", 0.5, 5.0, 3.0, 0.1)

# --- Automatic Resonance Scan ---

st.subheader("Resonance Finder")

scan = st.checkbox("Scan for Resonances Automatically")

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

    # --- Plot Transmission vs Energy ---
    fig_res, ax_res = plt.subplots()
    ax_res.plot(energies, transmissions, label="T(E)")

    # --- Find Local Peaks ---
    peak_indices = np.where(
        (transmissions[1:-1] > transmissions[:-2]) &
        (transmissions[1:-1] > transmissions[2:]) &
        (transmissions[1:-1] > 0.01)
    )[0] + 1

    for idx in peak_indices:
        ax_res.axvline(energies[idx], linestyle="--")

    # --- Bound State Predictions ---
    st.subheader("Predicted Well Eigenenergies")

    n_max = 5
    bound_energies = []

    for n in range(1, n_max+1):
        E_n = (n*np.pi)**2 / (2*m*well_width**2)
        if E_n < V0:
            bound_energies.append(E_n)
            ax_res.axvline(E_n, linestyle=":", alpha=0.7)

    ax_res.set_xlabel("Energy")
    ax_res.set_ylabel("Transmission")
    ax_res.set_title("Transmission Spectrum with Resonances")
    ax_res.legend()

    st.pyplot(fig_res)

    if len(peak_indices) > 0:
        st.write("Detected Resonance Energies:")
        for idx in peak_indices:
            st.write(f"E ≈ {energies[idx]:.4f}")
    else:
        st.write("No strong resonances detected in this range.")

# --- Derived quantities ---
k = np.sqrt(2*m*E + 0j)/hbar
kappa = np.sqrt(2*m*(V0 - E) + 0j)/hbar if E < V0 else 0

if E < V0:
    barrier_strength = np.real(kappa * a)
else:
    barrier_strength = 0

st.write(f"Barrier Strength κa = {barrier_strength:.3f}")

# --- Transfer Matrix for Transmission ---
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

if E < V0:
    k2 = 1j*kappa
else:
    k2 = np.sqrt(2*m*(E - V0) + 0j)/hbar

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
T = np.real(np.abs(t)**2)

st.write(f"Transmission T = {T:.6f}")

# --- Spatial Grid ---
x = np.linspace(-6, 6, 4000)
psi = np.zeros_like(x, dtype=complex)

x1 = -a - well_width/2
x2 = -well_width/2
x3 = well_width/2
x4 = a + well_width/2

r = M[1,0]/M[0,0]

for i, xi in enumerate(x):
    if xi < x1:
        psi[i] = np.exp(1j*k*xi) + r*np.exp(-1j*k*xi)
    elif x1 <= xi < x2:
        psi[i] = np.exp(-kappa*(xi-x1)) if E < V0 else np.exp(1j*k2*(xi-x1))
    elif x2 <= xi < x3:
        psi[i] = np.exp(1j*k*(xi-x2))
    elif x3 <= xi < x4:
        psi[i] = np.exp(-kappa*(xi-x3)) if E < V0 else np.exp(1j*k2*(xi-x3))
    else:
        psi[i] = t*np.exp(1j*k*xi)

# --- Plot ---
fig, ax = plt.subplots()

ax.plot(x, np.abs(psi)**2, label="|ψ|²")
ax.set_xlabel("x")
ax.set_ylabel("Probability Density")

# Shade barriers
ax.axvspan(x1, x2, alpha=0.2)
ax.axvspan(x3, x4, alpha=0.2)

st.pyplot(fig)

# --- Log Scale Plot ---
fig2, ax2 = plt.subplots()
ax2.plot(x, np.log10(np.abs(psi)**2 + 1e-15))
ax2.set_xlabel("x")
ax2.set_ylabel("log10(|ψ|²)")
ax2.set_title("Log Scale – Shows Exponential Decay Clearly")

ax2.axvspan(x1, x2, alpha=0.2)
ax2.axvspan(x3, x4, alpha=0.2)

st.pyplot(fig2)
ax.plot(x, np.real(psi), label="Re(ψ)")
ax.plot(x, np.abs(psi)**2, linestyle="--", label="|ψ|²")

# Draw barriers
ax.axvspan(x1, x2, alpha=0.2)
ax.axvspan(x3, x4, alpha=0.2)

ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.legend()

st.pyplot(fig)
