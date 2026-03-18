import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("🜂 Quetzal Hull Adaptive Simulation (QH-1 V4)")

# -------------------------
# CONSTANTS
# -------------------------
rho = 1000  # water density (kg/m^3)

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("🌊 Wave Conditions")

wave_amp = st.sidebar.slider("Wave Amplitude", 0.1, 2.0, 1.0)
wave_freq = st.sidebar.slider("Wave Frequency", 0.5, 5.0, 2.0)
velocity = st.sidebar.slider("Boat Velocity (m/s)", 0.1, 5.0, 2.0)

# Derived wavelength
wavelength = velocity / wave_freq if wave_freq != 0 else 1.0

st.sidebar.markdown(f"**Estimated Wavelength:** {wavelength:.2f} m")

# -------------------------
st.sidebar.header("🚤 Hull Parameters")

hull_length = st.sidebar.slider("Hull Length", 2.0, 10.0, 4.5)
Cd_base = st.sidebar.slider("Base Drag Coefficient", 0.2, 1.5, 0.8)
area = st.sidebar.slider("Frontal Area", 0.1, 2.0, 0.5)

# -------------------------
st.sidebar.header("🪶 Tail System")

tail_angle = st.sidebar.slider("Fin Angle (°)", -30, 30, 0)
tail_spread = st.sidebar.slider("Spread (°)", 5, 45, 20)
tail_length = st.sidebar.slider("Fin Length", 0.5, 2.0, 1.2)
tail_efficiency = st.sidebar.slider("Efficiency (α)", 0.0, 0.5, 0.25)
tail_surface = st.sidebar.slider("Surface Factor", 0.0, 1.0, 0.6)

# -------------------------
# PHYSICS CALCULATIONS
# -------------------------
Cd_eff = Cd_base * (1 - tail_efficiency * tail_surface)
drag_force = 0.5 * rho * velocity**2 * Cd_eff * area

stability_index = tail_surface * np.cos(np.radians(tail_spread)) * tail_length

# Hull-wave matching score
wave_match = 1 - abs(hull_length - wavelength/2) / (wavelength/2 + 0.001)

# -------------------------
# AI OPTIMIZATION
# -------------------------
if st.sidebar.button("🤖 Auto Optimize Tail"):
    best_score = -np.inf
    best = None

    for angle in np.linspace(-30, 30, 10):
        for spread in np.linspace(5, 45, 10):
            for length in np.linspace(0.5, 2.0, 10):

                stability = np.cos(np.radians(spread)) * length
                drag_gain = spread * length * 0.3
                score = stability + drag_gain

                if score > best_score:
                    best_score = score
                    best = (angle, spread, length)

    tail_angle, tail_spread, tail_length = best

    st.sidebar.success(
        f"Optimal Tail → Angle: {best[0]:.1f}, Spread: {best[1]:.1f}, Length: {best[2]:.2f}"
    )

# -------------------------
# DISPLAY METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Drag Force (N)", f"{drag_force:.2f}")
col2.metric("Stability Index", f"{stability_index:.3f}")
col3.metric("Wave Match Score", f"{wave_match:.3f}")

# -------------------------
# HULL VISUALIZATION
# -------------------------
st.subheader("🚤 Hull Geometry (Top View)")

fig1, ax1 = plt.subplots()

# Hull lines
ax1.plot([-0.3, -0.3], [0, hull_length], linewidth=3)
ax1.plot([0.3, 0.3], [0, hull_length], linewidth=3)

# Bow
ax1.plot([-0.3, 0, 0.3], [hull_length, hull_length + 0.5, hull_length])

# Tail fins
angles = np.linspace(-tail_spread, tail_spread, 5) + tail_angle

for angle in angles:
    x = tail_length * np.sin(np.radians(angle))
    y = -tail_length * np.cos(np.radians(angle))
    ax1.plot([0, x], [0, y], linewidth=2)

ax1.set_aspect('equal')
ax1.axis('off')

st.pyplot(fig1)

# -------------------------
# WAKE SIMULATION
# -------------------------
st.subheader("🌊 Wake Simulation")

x = np.linspace(0, 10, 500)

# Tail increases damping
k = 0.3 + tail_efficiency * tail_surface

y = wave_amp * np.exp(-k * x) * np.sin(wave_freq * x)

fig2, ax2 = plt.subplots()
ax2.plot(x, y)
ax2.set_title("Wave Decay Behind Hull")
ax2.set_xlabel("Distance")
ax2.set_ylabel("Amplitude")

st.pyplot(fig2)

# -------------------------
# INTERPRETATION PANEL
# -------------------------
st.subheader("🧠 System Interpretation")

st.write(f"""
- **Lower Drag** achieved via tail efficiency → {tail_efficiency:.2f}
- **Stability** improves with spread and length
- **Wave Matching** currently at {wave_match:.2f}

### Insights:
- Match hull length ≈ half wavelength for best glide
- Increase spread for rough water
- Increase length for stronger damping
""")
