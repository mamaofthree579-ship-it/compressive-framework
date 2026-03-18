import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("🌊 Dugout vs QH-1 Adaptive Hull Simulation")

# -------------------------
# Environment Inputs
# -------------------------
wave_amp = st.slider("Wave Amplitude (m)", 0.1, 2.0, 1.0)
wave_freq = st.slider("Wave Frequency (Hz)", 0.5, 5.0, 1.5)
velocity = st.slider("Boat Velocity (m/s)", 0.5, 5.0, 2.0)
rho = 1000  # water density kg/m^3

wavelength = velocity / wave_freq

# -------------------------
# Dugout Canoe Properties
# -------------------------
dugout = {
    "length": 5.0,
    "beam": 0.75,
    "draft": 0.2,
    "Cd": 0.7,
    "A": 0.15,  # frontal area
    "Rw": 1.0,  # water resistance factor (resin-rich wood)
}

# -------------------------
# QH-1 Adaptive Hull Properties
# -------------------------
qh1 = {
    "length": wavelength/2,  # wave-matched
    "beam": 1.0,
    "tail_eff": 0.35,
    "tail_surface": 0.6,
    "draft": 0.25,
    "Cd_base": 0.7,
    "A": 0.2,
}

# Effective Cd for QH-1
qh1["Cd_eff"] = qh1["Cd_base"] * (1 - qh1["tail_eff"] * qh1["tail_surface"])

# -------------------------
# Calculations
# -------------------------
def drag_force(Cd, A):
    return 0.5 * rho * velocity**2 * Cd * A

dugout_drag = drag_force(dugout["Cd"], dugout["A"]) * dugout["Rw"]
qh1_drag = drag_force(qh1["Cd_eff"], qh1["A"])

# Stability Index (simplified)
dugout_stab = 0.5  # baseline, rudderless
qh1_stab = np.cos(np.radians(30)) * qh1["tail_surface"] * 1.0

# Wave matching factor
dugout_wave_match = 1 - abs(dugout["length"] - wavelength/2) / (wavelength/2 + 0.001)
qh1_wave_match = 1.0  # adaptive

# -------------------------
# Display Metrics
# -------------------------
st.subheader("Performance Metrics")
st.write(f"**Dugout Canoe:** Drag={dugout_drag:.2f} N, Stability={dugout_stab:.2f}, Wave Match={dugout_wave_match:.2f}")
st.write(f"**QH-1 Hull:** Drag={qh1_drag:.2f} N, Stability={qh1_stab:.2f}, Wave Match={qh1_wave_match:.2f}")

# -------------------------
# Visualization
# -------------------------
x = np.linspace(0, 10, 500)
dugout_y = wave_amp * np.exp(-0.1*x) * np.sin(wave_freq*x)
qh1_y = wave_amp * np.exp(-0.3*x) * np.sin(wave_freq*x)  # better damping

plt.figure(figsize=(10,4))
plt.plot(x, dugout_y, label="Dugout Canoe Wake")
plt.plot(x, qh1_y, label="QH-1 Adaptive Hull Wake")
plt.xlabel("Distance (m)")
plt.ylabel("Wave Amplitude (m)")
plt.title("Wake Comparison")
plt.legend()
st.pyplot(plt)
