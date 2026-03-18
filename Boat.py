import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🜂 Quetzal Hull Simulation (QH-1)")

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("Simulation Controls")

A = st.sidebar.slider("Wave Amplitude (A)", 0.1, 2.0, 1.0)
k = st.sidebar.slider("Damping Factor (k)", 0.1, 2.0, 0.5)
omega = st.sidebar.slider("Wave Frequency (ω)", 1.0, 10.0, 4.0)
phi = st.sidebar.slider("Phase Shift (φ)", 0.0, np.pi, 0.0)

tail_spread = st.sidebar.slider("Tail Spread Angle", 5, 45, 20)
fin_length = st.sidebar.slider("Tail Length", 0.5, 2.0, 1.2)

# -------------------------
# HULL VISUALIZATION
# -------------------------
st.subheader("Hull Geometry (Top View)")

fig1, ax1 = plt.subplots()

# Hull lines
ax1.plot([-0.3, -0.3], [0, 4], linewidth=3)
ax1.plot([0.3, 0.3], [0, 4], linewidth=3)

# Bow
ax1.plot([-0.3, 0, 0.3], [4, 4.5, 4])

# Tail fins
angles = np.linspace(-tail_spread, tail_spread, 5)

for angle in angles:
    x = fin_length * np.sin(np.radians(angle))
    y = -fin_length * np.cos(np.radians(angle))
    ax1.plot([0, x], [0, y], linewidth=2)

ax1.set_aspect('equal')
ax1.set_title("Quetzal-Inspired Hull + Tail")
ax1.axis('off')

st.pyplot(fig1)

# -------------------------
# WAVE SIMULATION
# -------------------------
st.subheader("Wake Behavior Simulation")

x = np.linspace(0, 10, 500)
y = A * np.exp(-k * x) * np.sin(omega * x - phi)

fig2, ax2 = plt.subplots()
ax2.plot(x, y)
ax2.set_title("Wave Decay Behind Hull")
ax2.set_xlabel("Distance")
ax2.set_ylabel("Wave Amplitude")

st.pyplot(fig2)

# -------------------------
# INTERPRETATION
# -------------------------
st.subheader("Interpretation")

st.write("""
- Faster decay = more efficient wake control  
- Tail system increases damping (k)  
- Wider tail spread improves lateral stability  
""")
