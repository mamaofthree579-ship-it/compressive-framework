import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Burst-Seeded Galaxy Bubble Simulator")

# ---------------------------
# Controls
# ---------------------------
num_particles = st.slider("Particles", 50, 400, 150)
burst_strength = st.slider("Burst Strength", 0.1, 3.0, 1.0)
expansion_rate = st.slider("Expansion Rate", 0.01, 0.2, 0.05)
rotation_strength = st.slider("Rotation", 0.0, 0.1, 0.02)
compression_strength = st.slider("Compression", 0.0, 0.05, 0.01)
steps = st.slider("Steps per run", 1, 10, 3)

# ---------------------------
# Initialize state
# ---------------------------
if "pos" not in st.session_state:
    st.session_state.pos = np.random.uniform(-0.1, 0.1, (num_particles, 3))
    st.session_state.vel = np.zeros((num_particles, 3))
    st.session_state.radius = 0.1

# ---------------------------
# Simulation Step
# ---------------------------
def update():
    pos = st.session_state.pos
    vel = st.session_state.vel
    radius = st.session_state.radius
    
    center = np.array([0.0, 0.0, 0.0])
    
    # Expand boundary (bubble growth)
    radius += expansion_rate
    
    for i in range(len(pos)):
        direction = pos[i] - center
        dist = np.linalg.norm(direction) + 1e-5
        
        # Normalize
        dir_norm = direction / dist
        
        # ---------------------------
        # Burst outward push (initial)
        # ---------------------------
        vel[i] += burst_strength * dir_norm * 0.01
        
        # ---------------------------
        # Boundary trapping
        # ---------------------------
        if dist > radius:
            # push back inward if escaping
            vel[i] -= dir_norm * 0.05
        
        # ---------------------------
        # Compression toward center
        # ---------------------------
        vel[i] -= compression_strength * dir_norm
        
        # ---------------------------
        # Rotation (around Z axis)
        # ---------------------------
        rot = np.array([-pos[i][1], pos[i][0], 0])
        vel[i] += rotation_strength * rot
    
    # Update positions
    pos += vel
    
    # Save back
    st.session_state.pos = pos
    st.session_state.vel = vel
    st.session_state.radius = radius

# ---------------------------
# Run Simulation
# ---------------------------
if st.button("Run Simulation"):
    for _ in range(steps):
        update()

# ---------------------------
# Plot
# ---------------------------
pos = st.session_state.pos
x, y, z = pos[:,0], pos[:,1], pos[:,2]

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers'
))

# Bubble boundary (visual sphere)
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 15)
r = st.session_state.radius

xs = r * np.outer(np.cos(u), np.sin(v))
ys = r * np.outer(np.sin(u), np.sin(v))
zs = r * np.outer(np.ones(np.size(u)), np.cos(v))

fig.add_trace(go.Surface(
    x=xs, y=ys, z=zs,
    opacity=0.1,
    showscale=False
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
# Metrics
# ---------------------------
center = np.mean(pos, axis=0)
distances = np.linalg.norm(pos - center, axis=1)

coherence = 1 / (np.std(distances) + 1e-5)

st.metric("Bubble Radius", f"{st.session_state.radius:.2f}")
st.metric("Coherence Index", f"{coherence:.3f}")
