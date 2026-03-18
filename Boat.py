import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🜂 QH-1 V5 Adaptive Hull Intelligence System")

rho = 1000

# -------------------------
# CONTROLS
# -------------------------
auto_mode = st.sidebar.checkbox("Enable Auto Adaptation", True)

st.sidebar.header("🌊 Wave Conditions")
A = st.sidebar.slider("Amplitude", 0.1, 2.0, 1.0)
freq = st.sidebar.slider("Frequency", 0.5, 5.0, 2.0)
velocity = st.sidebar.slider("Velocity", 0.1, 5.0, 2.0)

wavelength = velocity / freq if freq != 0 else 1.0
roughness = A * freq

# -------------------------
# DEFAULTS
# -------------------------
hull_length = st.sidebar.slider("Hull Length", 2.0, 10.0, 4.5)
spread = st.sidebar.slider("Spread", 5, 45, 20)
length = st.sidebar.slider("Fin Length", 0.5, 2.0, 1.2)
angle = st.sidebar.slider("Angle", -30, 30, 0)

# -------------------------
# AUTO SYSTEM
# -------------------------
if auto_mode:
    # Rule 1: Hull match
    hull_length = wavelength / 2

    # Rule 2: Spread
    spread = 10 + 20 * roughness

    # Rule 3: Length
    length = 0.5 + 0.8 * A

    # Clamp values
    spread = np.clip(spread, 5, 45)
    length = np.clip(length, 0.5, 2.0)

# -------------------------
# PHYSICS
# -------------------------
Cd_base = 0.8
area = 0.5

tail_eff = 0.3
tail_surface = spread / 45

Cd_eff = Cd_base * (1 - tail_eff * tail_surface)
drag = 0.5 * rho * velocity**2 * Cd_eff * area

stability = tail_surface * np.cos(np.radians(spread)) * length
wave_match = 1 - abs(hull_length - wavelength/2) / (wavelength/2 + 0.001)

# -------------------------
# DISPLAY
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Drag", f"{drag:.2f}")
col2.metric("Stability", f"{stability:.3f}")
col3.metric("Wave Match", f"{wave_match:.3f}")

# -------------------------
# HULL VISUAL
# -------------------------
fig, ax = plt.subplots()

ax.plot([-0.3, -0.3], [0, hull_length], linewidth=3)
ax.plot([0.3, 0.3], [0, hull_length], linewidth=3)
ax.plot([-0.3, 0, 0.3], [hull_length, hull_length+0.5, hull_length])

angles = np.linspace(-spread, spread, 5) + angle

for a in angles:
    x = length * np.sin(np.radians(a))
    y = -length * np.cos(np.radians(a))
    ax.plot([0, x], [0, y], linewidth=2)

ax.set_aspect('equal')
ax.axis('off')

st.pyplot(fig)

# -------------------------
# WAVE GRAPH
# -------------------------
x = np.linspace(0, 10, 500)
k = 0.3 + tail_surface

y = A * np.exp(-k * x) * np.sin(freq * x)

fig2, ax2 = plt.subplots()
ax2.plot(x, y)
ax2.set_title("Wake Decay")

st.pyplot(fig2)

# -------------------------
# INTERPRETATION
# -------------------------
st.write(f"""
### Auto System Active: {auto_mode}

- Hull auto-matched to wavelength
- Spread responding to roughness ({roughness:.2f})
- Tail length scaling with amplitude

This is now a **self-adjusting hydrodynamic system**.
""")
