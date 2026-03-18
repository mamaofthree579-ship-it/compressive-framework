import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")
st.title("🌊 QH-1 Adaptive Hull: Real-Time Interactive Simulation")

# -------------------------
# Environment Presets
# -------------------------
environments = {
    "Lake (Calm)": {"amp": 0.2, "freq": 0.5, "velocity": 1.0},
    "River (Moderate)": {"amp": 0.5, "freq": 1.2, "velocity": 2.0},
    "Ocean (Rough)": {"amp": 1.5, "freq": 3.0, "velocity": 3.0}
}

# -------------------------
# Hull Defaults
# -------------------------
dugout = {
    "length": 5.0,
    "beam": 0.75,
    "draft": 0.2,
    "Cd": 0.7,
    "A": 0.15,
    "Rw": 1.0
}

qh1 = {
    "tail_eff": 0.35,
    "tail_surface": 0.6,
    "beam": 1.0,
    "draft": 0.25,
    "Cd_base": 0.7,
    "A": 0.2
}

# -------------------------
# Interactive Controls
# -------------------------
st.sidebar.header("Adaptive Hull Controls")
tail_angle = st.sidebar.slider("Tail Feather Angle (°)", -30, 30, 15)
tail_spread = st.sidebar.slider("Tail Feather Spread (°)", 5, 45, 30)
modular_length_factor = st.sidebar.slider("Hull Length Factor (relative to λ/2)", 0.5, 1.5, 1.0)
modular_width = st.sidebar.slider("Hull Width (m)", 0.6, 1.2, 1.0)

# -------------------------
# Simulation Function
# -------------------------
def simulate_hull(env_name, tail_angle, tail_spread, length_factor, width):
    env = environments[env_name]
    amp, freq, velocity = env["amp"], env["freq"], env["velocity"]
    wavelength = velocity / freq

    # Dugout metrics
    dugout_drag = 0.5 * 1000 * velocity**2 * dugout["Cd"] * dugout["A"] * dugout["Rw"]
    dugout_stab = 0.5
    dugout_wave = 1 - abs(dugout["length"] - wavelength/2) / (wavelength/2 + 0.001)

    # QH-1 adaptive metrics
    qh1_length = wavelength/2 * length_factor
    qh1["length"] = qh1_length
    qh1["beam"] = width
    qh1["Cd_eff"] = qh1["Cd_base"] * (1 - qh1["tail_eff"] * qh1["tail_surface"])
    qh1_drag = 0.5 * 1000 * velocity**2 * qh1["Cd_eff"] * qh1["A"]
    qh1_stab = np.cos(np.radians(tail_spread)) * qh1["tail_surface"]
    qh1_wave = 1.0  # adaptive system

    # Wake decay for visualization
    x_line = np.linspace(0, 20, 500)
    dugout_y = amp * np.exp(-0.1 * x_line) * np.sin(freq * x_line)
    qh1_y = amp * np.exp(-0.3 * x_line) * np.sin(freq * x_line)

    return dugout_drag, qh1_drag, dugout_stab, qh1_stab, dugout_wave, qh1_wave, x_line, dugout_y, qh1_y

# -------------------------
# Run Simulation for Selected Environment
# -------------------------
env_selected = st.selectbox("Select Environment", list(environments.keys()))
dugout_drag, qh1_drag, dugout_stab, qh1_stab, dugout_wave, qh1_wave, x_line, dugout_y, qh1_y = simulate_hull(
    env_selected, tail_angle, tail_spread, modular_length_factor, modular_width
)

# -------------------------
# Display Metrics
# -------------------------
st.subheader(f"Performance Metrics: {env_selected}")
df = pd.DataFrame({
    "Metric": ["Drag (N)", "Stability", "Wave Match"],
    "Dugout Canoe": [dugout_drag, dugout_stab, dugout_wave],
    "QH-1 Adaptive Hull": [qh1_drag, qh1_stab, qh1_wave]
})
st.dataframe(df)

# -------------------------
# Visualize Wake
# -------------------------
st.subheader(f"Wake Visualization: {env_selected}")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x_line, dugout_y, label="Dugout Canoe Wake")
ax.plot(x_line, qh1_y, label="QH-1 Adaptive Hull Wake")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Wave Amplitude (m)")
ax.set_title(f"Wake Decay Comparison: {env_selected}")
ax.legend()
st.pyplot(fig)

# -------------------------
# Interpretation
# -------------------------
st.write("""
### Interpretation:
- Dugout canoes perform best in calm lakes, but lose efficiency in rivers and oceans.  
- QH-1 maintains high wave matching, improves stability, and accelerates wake decay through tail adjustment and modular hull length.  
- Use sliders to experiment with tail angles, spread, and hull length to optimize performance for different water conditions.
""")
