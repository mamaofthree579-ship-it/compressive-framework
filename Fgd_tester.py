import streamlit as st
import numpy as np

st.title("🌊 The Cosmic Plasma: Fluid Stability")
viscosity = st.sidebar.slider("Plasma Viscosity (Medium Grip)", 0.1, 2.0, 1.0)

# Logic: High Viscosity = Lower Drift
# If Viscosity is high, the planet is 'held' by the water-like space.
drift_risk = (1.0 / viscosity) * (97.9 - 89.89)

st.metric("Drift Risk Level", f"{drift_risk:.2f}%")
if drift_risk > 5.0:
    st.error("🚨 VACUUM CHAOS: The system is too 'slippery.' Prepare for Collision.")
else:
    st.success("✅ FLUID STABILITY: The Plasma Medium is holding the orbits in place.")
