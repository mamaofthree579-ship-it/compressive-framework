import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🜂 Quetzal Hull Physics Simulator (QH-1 V3)")

# -------------------------
# CONSTANTS
# -------------------------
rho = 1000  # water density kg/m^3

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("Hull Parameters")

velocity = st.sidebar.slider("Velocity (m/s)", 0.1, 5.0, 2.0)
Cd_base = st.sidebar.slider("Base Drag Coefficient", 0.2, 1.5, 0.8)
area = st.sidebar.slider("Frontal Area (m^2)", 0.1, 2.0, 0.5)

st.sidebar.header("Tail System")

tail_efficiency = st.sidebar.slider("Tail Efficiency Factor (α)", 0.0, 0.5, 0.25)
tail_surface = st.sidebar.slider("Tail Surface Factor (Sf)", 0.0, 1.0, 0.6)

tail_spread = st.sidebar.slider("Tail Spread Angle", 5, 45, 20)
fin_length = st.sidebar.slider("Tail Length", 0.5, 2.0, 1.2)

# -------------------------
# DRAG CALCULATION
# -------------------------
Cd_eff = Cd_base * (1 - tail_efficiency * tail_surface)
drag_force = 0.5 * rho * velocity**2 * Cd_eff * area

# -------------------------
# STABILITY ESTIMATION
# -------------------------
stability_index = tail_surface * np.cos(np.radians(tail_spread)) * fin_length

# -------------------------
# DISPLAY RESULTS
# -------------------------
st.subheader("Physics Output")

st.write(f"Effective Drag Coefficient: {Cd_eff:.3f}")
st.write(f"Drag Force: {drag_force:.2f} N")
st.write(f"Stability Index: {stability_index:.3f}")

# -------------------------
# HULL VISUALIZATION
# -------------------------
st.subheader("Hull Geometry")

fig1, ax1 = plt.subplots()

# Hull
ax1.plot([-0.3, -0.3], [0, 4], linewidth=3)
ax1.plot([0.3, 0.3], [0, 4], linewidth=3)

# Bow
ax1.plot([-0.3, 0, 0.3], [4, 4.5, 4])

# Tail
angles = np.linspace(-tail_spread, tail_spread, 5)
for angle in angles:
    x = fin_length * np.sin(np.radians(angle))
    y = -fin_length * np.cos(np.radians(angle))
    ax1.plot([0, x], [0, y], linewidth=2)

ax1.set_aspect('equal')
ax1.axis('off')

st.pyplot(fig1)

# -------------------------
# WAKE SIMULATION
# -------------------------
st.subheader("Wake Decay")

x = np.linspace(0, 10, 500)
k = 0.3 + tail_efficiency * tail_surface
y = np.exp(-k * x) * np.sin(4 * x)

fig2, ax2 = plt.subplots()
ax2.plot(x, y)
ax2.set_title("Wave Decay")
st.pyplot(fig2)
