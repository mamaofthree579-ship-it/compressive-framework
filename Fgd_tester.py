import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🕳️ Cosmic Void Analysis: The 1.6 Threshold")
st.markdown("Testing orbital stability as a system moves from **High-Density Plasma** to a **Low-Density Void**.")

# --- SIDEBAR: Void Parameters ---
visc_env = st.sidebar.slider("Environment Viscosity", 0.5, 2.0, 0.8, 
                             help="Below 1.6, the system loses its 4D anchor.")
drift_time = st.sidebar.slider("Time in Void", 10, 100, 50)

# --- PHYSICS: Turbulence Calculation ---
t = np.linspace(0, drift_time, 1000)
# Stability Factor: If visc < 1.6, we add 'Turbulence' (Random Walk)
stability_gap = max(0, 1.6 - visc_env)
turbulence = np.cumsum(np.random.normal(0, stability_gap, 1000) * 0.1)

# The 'Tumbling' Orbit
orbit_path = np.sin(t) + turbulence

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
ax.set_facecolor('#1e1e1e')

ax.plot(t, orbit_path, color='#ff4b4b' if visc_env < 1.6 else '#00ffcc', 
        linewidth=2, label="Void Trajectory" if visc_env < 1.6 else "Stable Trajectory")
ax.axhline(0, color='white', alpha=0.2)

ax.set_title(f"Orbital Integrity at Viscosity: {visc_env}")
ax.legend(labelcolor='white')
st.pyplot(fig)

# --- THE VERDICT ---
st.divider()
if visc_env < 1.6:
    st.error(f"❌ TURBULENT STATE: The 4D Medium is too thin ({visc_env}). The 97.9% Cohesion is lost.")
    st.write("**Outcome:** System is prone to crashes, mergers, or total separation.")
else:
    st.success(f"✅ LAMINAR STATE: The 1.6 Threshold is met. Matter is 'Locked' in its groove.")
