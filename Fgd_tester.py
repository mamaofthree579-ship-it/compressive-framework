import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="FGD 4D Multi-Metric", layout="wide")
st.title("🌐 4-D Landscape & 3-D Material Resonance")
st.markdown("""
**Theory:** 3D Matter is a 'folded' state within a 4D Informational Flux. 
The 'Dark' fields are the 4D structures holding the 3D material in place.
""")

# --- SIDEBAR: Multi-Metric Inputs ---
st.sidebar.header("Field Metrics")
dim_flux = st.sidebar.slider("4D Flux Density (Ψ)", 0.5, 2.0, 1.2, 
                            help="The strength of the 4th dimensional background field.")
res_freq = st.sidebar.slider("Resonance Frequency", 0.1, 5.0, 1.0, 
                            help="The 'Micro-burst' frequency creating coherence.")
stability_zone = st.sidebar.slider("Goldilocks Zone (kpc)", 10, 100, (30, 70))

# --- PHYSICS: Multi-Field Integration ---
r = np.linspace(1, 120, 1000)
M_3d = 100.0 # Visible 3D Material

# 1. 3D Gravitational Pull (Classical)
v_3d = np.sqrt(M_3d / r)

# 2. 4D Flux 'Anchor' (The Multi-Metric Correction)
# This represents the 'Compiled' Dark Matter structure as a 4D pressure
v_4d_anchor = np.sqrt((dim_flux * M_3d / r) + (res_freq * np.sin(r/10) * 10))

# 3. Combined 'Stuck' Material Velocity
v_combined = np.sqrt(v_3d**2 + np.abs(v_4d_anchor**2))

# --- VISUALIZATION ---
st.subheader("3D Material Stability in a 4D Landscape")
fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
ax.set_facecolor('#1e1e1e')

# The 4D Field (Background)
ax.fill_between(r, 0, v_combined, color='cyan', alpha=0.1, label="4D Informational Flux")

# The Goldilocks Resonance Zone
ax.axvspan(stability_zone[0], stability_zone[1], color='gold', alpha=0.2, label="Goldilocks Resonance Zone")

# Plotting the Forces
ax.plot(r, v_3d, 'w--', alpha=0.5, label="3D Matter Only (Unstable)")
ax.plot(r, v_combined, color='cyan', linewidth=2, label="Resonant 3D Material (Stable)")

# Formatting
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.legend(facecolor='#1e1e1e', labelcolor='white')
st.pyplot(fig)

# --- THE RESONANCE ANALYSIS ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.write("### 💎 Multi-Field Analysis")
    st.info(f"""
    - **Coherence:** At current Psi ({dim_flux}), the 3D material is 'anchored' effectively.
    - **Micro-bursts:** The sine-wave fluctuations simulate the 'heartbeat' maintaining the system.
    """)

with col2:
    st.write("### 🚨 Collision Alert")
    if stability_zone[0] < 20:
        st.error("❌ CRASH RISK: Galaxies too close. Resonance becomes turbulent.")
    else:
        st.success("✅ HARMONIC STATE: Galaxies maintained by mutual 4D flux.")
