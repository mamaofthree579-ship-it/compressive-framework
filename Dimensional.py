import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")
st.title("Galaxy Formation Simulator")

# ---------------------------
# Settings
# ---------------------------
num_particles = st.slider("Number of particles", 50, 300, 120)
gravity_strength = st.slider("Attraction strength", 0.001, 0.05, 0.01)
noise_level = st.slider("Initial noise", 0.0, 0.5, 0.1)
steps_per_frame = 2

# ---------------------------
# Session state
# ---------------------------
if "positions" not in st.session_state:
    st.session_state.positions = np.random.uniform(-1, 1, (num_particles, 3)) * noise_level
    st.session_state.velocities = np.zeros((num_particles, 3))

# ---------------------------
# Physics step
# ---------------------------
def update():
    pos = st.session_state.positions
    vel = st.session_state.velocities
    
    forces = np.zeros_like(pos)

    for i in range(len(pos)):
        diff = pos - pos[i]
        dist = np.linalg.norm(diff, axis=1) + 0.01
        
        attraction = (diff.T / dist**3).T
        forces[i] += np.sum(attraction, axis=0)
    
    vel += gravity_strength * forces
    pos += vel

    st.session_state.positions = pos
    st.session_state.velocities = vel

# ---------------------------
# Run simulation
# ---------------------------
if st.button("Run Simulation"):
    for _ in range(steps_per_frame):
        update()

# ---------------------------
# Plot
# ---------------------------
pos = st.session_state.positions
x, y, z = pos[:,0], pos[:,1], pos[:,2]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers'
))

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Coherence Index
# ---------------------------
center = np.mean(pos, axis=0)
distances = np.linalg.norm(pos - center, axis=1)
coherence_index = 1 / (np.std(distances) + 0.001)

st.metric("Coherence Index", f"{coherence_index:.3f}")
