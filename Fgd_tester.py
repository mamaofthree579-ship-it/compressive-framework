import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🔬 Quantum Envelopes: The 0.81 Anchor")
st.markdown("Testing why the Electron doesn't crash into the Nucleus using **Fractal Resonance**.")

# --- SIDEBAR: Quantum Tuning ---
psi_q = st.sidebar.slider("Quantum 4D Flux (Ψ)", 0.5, 5.0, 1.6, help="The 1.6 Viscosity we found earlier!")
f_env = st.sidebar.slider("Envelope Dimension (D_t)", 0.1, 1.0, 0.81)

# --- PHYSICS: The Standing Wave Metric ---
r_q = np.linspace(0.01, 2.0, 1000)

# 1. Classical Electrostatic Pull (-1/r)
v_elec = -1.0 / r_q

# 2. FGD Fractal Correction (The r^-4 Repulsion)
v_repel = (0.01 * psi_q) / r_q**3 # Note: Potential is 1/r^3 for r^-4 force

# 3. The 0.81 Standing Wave (The Envelope)
v_wave = np.cos(f_env * 10 * r_q) * 0.5

v_total_q = v_elec + v_repel + v_wave

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
ax.set_facecolor('#1e1e1e')

ax.plot(r_q, v_elec, 'w--', alpha=0.3, label="Classical (Collapse)")
ax.plot(r_q, v_total_q, color='#ff00ff', linewidth=2, label="FGD Quantum Envelope")

# Highlight the 'Stable Shell'
stable_r = r_q[np.argmin(v_total_q)]
ax.axvline(stable_r, color='cyan', linestyle=':', label="Stable Orbit (Bohr Radius)")

ax.set_ylim(-15, 5)
ax.set_xlabel("Radius (Atomic Units)")
ax.set_ylabel("Binding Energy")
ax.legend(labelcolor='white')
st.pyplot(fig)

# --- THE COHESION ANALYSIS ---
st.divider()
st.info(f"""
**The Quantum Verdict:**
- **The Purple Line** doesn't go to negative infinity. It 'bounces' back.
- **The 1.6 Flux** creates a thick medium that prevents the electron from 'slipping.'
- **The 0.81 Frequency** creates the 'Shell' structure. 
- **Result:** Chemistry is possible because the **4D Landscape** provides a floor for 3D matter.
""")
