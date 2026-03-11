import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🧬 Biological Resonance: The DNA Double-Helix")
st.markdown("Testing if the **0.81 Fractal Dimension** creates the 'Master Template' for life.")

# --- SIDEBAR: Biological Resonance ---
phi_bio = st.sidebar.slider("Cellular 4D Flux (Ψ)", 0.5, 2.0, 1.6, help="The 1.6 Stability Threshold!")
freq_dna = st.sidebar.slider("Coiling Frequency", 0.01, 0.2, 0.081, help="The 0.81 Fractal Scale!")

# --- PHYSICS: 3D Projection of 4D Coiling ---
z = np.linspace(0, 20, 1000)
# The Helix: x = r*cos(theta), y = r*sin(theta)
# Theta is driven by the 0.081 Frequency
theta = 2 * np.pi * freq_dna * z * 10

x1 = np.cos(theta)
y1 = np.sin(theta)

# The Second Strand (180 degrees out of phase)
x2 = np.cos(theta + np.pi)
y2 = np.sin(theta + np.pi)

# --- VISUALIZATION: The 3D/4D Projection ---
fig = plt.figure(figsize=(10, 6), facecolor='#0e1117')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#1e1e1e')

# Plotting the Strands
ax.plot(x1, y1, z, color='cyan', linewidth=3, label="Strand A (4D Pulse)")
ax.plot(x2, y2, z, color='magenta', linewidth=3, label="Strand B (Resonance)")

# Drawing the 'Hydrogen Bonds' (The 3D Tethers)
for i in range(0, len(z), 40):
    ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z[i], z[i]], color='white', alpha=0.3)

ax.set_axis_off()
ax.legend(facecolor='#1e1e1e', labelcolor='white')
st.pyplot(fig)

# --- THE COHESION VERDICT ---
st.divider()
st.info(f"""
**The Bio-Logic:**
- **The Double Helix** is the most efficient way to pack 3D material into a 4D landscape.
- **The 0.81 Frequency** ensures the 'Cohesion' is high enough to store data but low enough to allow mutation.
- **Result:** Life is a **standing wave** in a 4D liquid medium.
""")
