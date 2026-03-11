import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🌙 The Moon Anchor: 4D Resonance Tracking")
st.markdown("""
**The Moon Test:** Is the Moon just 'falling' around Earth, or is it 'nested' in a 4D Standing Wave? 
Jones suggests the Moon is locked into a **97.9% Cohesion Well**.
""")

# --- SIDEBAR: Lunar Tuning ---
psi_lunar = st.sidebar.slider("Earth's 4D Flux (Ψ)", 0.5, 3.0, 1.45)
f_lunar = st.sidebar.slider("Resonance Frequency (f)", 0.01, 0.10, 0.07, 
                           help="Your discovered stable limit!")

# --- PHYSICS: The Standing Wave Metric ---
# Distance in 10,000s of km (Moon is at ~38.4)
dist = np.linspace(10, 60, 1000)

# 1. Classical Gravity Potential (1/r)
v_grav = -1.0 / dist

# 2. 4D Standing Wave (Resonance Template)
# Wave = Ψ * cos(f * dist)
v_wave = psi_lunar * np.cos(f_lunar * dist)

# 3. The Combined 'Nested' Well
v_total = v_grav + (v_wave * 0.1) # The 4D flux guides the 3D gravity

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
ax.set_facecolor('#1e1e1e')

ax.plot(dist, v_grav, 'w--', alpha=0.3, label="Newtonian Path")
ax.plot(dist, v_total, color='cyan', linewidth=2, label="FGD 'Resonance Groove'")

# Mark the Actual Moon Distance
ax.axvline(38.4, color='gold', linestyle=':', label="Current Moon Distance")
ax.scatter(38.4, v_total[np.argmin(np.abs(dist - 38.4))], color='gold', s=100)

ax.set_xlabel("Distance (10,000 km)")
ax.set_ylabel("Binding Energy")
ax.legend(labelcolor='white')
st.pyplot(fig)

# --- COHESION ANALYSIS ---
st.divider()
current_cohesion = np.abs(np.cos(f_lunar * 38.4)) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("Lunar Anchor Score", f"{current_cohesion:.2f}%")
    if 97.0 <= current_cohesion <= 98.5:
        st.success("🎯 HARMONIC LOCK: The Moon is perfectly anchored in the 4D Groove!")
    else:
        st.warning("⚠️ DRIFT RISK: The Moon is not in its 'Jones Constant' well.")

with col2:
    st.info("""
    **The Discovery:**
    If the Moon is in the 'Lock' zone, it explains why it only shows us one face. 
    It's not just tidal friction; it's **Geometric Cohesion**.
    """)
