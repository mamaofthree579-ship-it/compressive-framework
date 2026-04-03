import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Multi-Bubble Galaxy + Filament Simulator")

# ---------------------------
# Controls
# ---------------------------
num_bubbles = st.slider("Number of Bubbles", 2, 5, 3)
particles_per_bubble = st.slider("Particles per Bubble", 50, 200, 100)

burst_strength = st.slider("Burst Strength", 0.1, 2.0, 1.0)
expansion_rate = st.slider("Expansion Rate", 0.01, 0.1, 0.03)
rotation_strength = st.slider("Rotation", 0.0, 0.1, 0.02)
compression_strength = st.slider("Compression", 0.0, 0.05, 0.01)

shell_strength = st.slider("Shell Strength", 0.0, 0.1, 0.03)
num_shells = st.slider("Shell Layers", 2, 8, 4)

filament_strength = st.slider("Filament Strength", 0.0, 0.05, 0.01)

steps = st.slider("Steps per run", 1, 10, 3)

# ---------------------------
# Initialize state
# ---------------------------
if "bubbles" not in st.session_state:
    bubbles = []
    for i in range(num_bubbles):
        center = np.random.uniform(-2, 2, 3)
        pos = center + np.random.uniform(-0.1, 0.1, (particles_per_bubble, 3))
        vel = np.zeros((particles_per_bubble, 3))
        bubbles.append({"center": center, "pos": pos, "vel": vel, "radius": 0.2})
    
    st.session_state.bubbles = bubbles

# ---------------------------
# Shell function
# ---------------------------
def shell_force(distance, max_radius):
    spacing = max_radius / num_shells
    nearest = round(distance / spacing) * spacing
    return nearest - distance

# ---------------------------
# Simulation Step
# ---------------------------
def update():
    bubbles = st.session_state.bubbles
    
    for b in bubbles:
        b["radius"] += expansion_rate

    # Bubble interactions (centers attract slightly)
    for i in range(len(bubbles)):
        for j in range(i+1, len(bubbles)):
            dir_vec = bubbles[j]["center"] - bubbles[i]["center"]
            dist = np.linalg.norm(dir_vec) + 1e-5
            force = dir_vec / dist * filament_strength

            bubbles[i]["center"] += force * 0.01
            bubbles[j]["center"] -= force * 0.01

    # Particle updates
    for b in bubbles:
        center = b["center"]
        pos = b["pos"]
        vel = b["vel"]
        radius = b["radius"]

        for i in range(len(pos)):
            direction = pos[i] - center
            dist = np.linalg.norm(direction) + 1e-5
            dir_norm = direction / dist

            # Burst
            vel[i] += burst_strength * dir_norm * 0.01

            # Containment
            if dist > radius:
                vel[i] -= dir_norm * 0.05

            # Compression
            vel[i] -= compression_strength * dir_norm

            # Rotation
            rot = np.array([-direction[1], direction[0], 0])
            vel[i] += rotation_strength * rot

            # Shells
            shell_adj = shell_force(dist, radius)
            vel[i] += shell_strength * shell_adj * dir_norm

            # ---------------------------
            # Filament pull toward other bubbles
            # ---------------------------
            for other in bubbles:
                if not np.array_equal(other["center"], center):
                    link = other["center"] - pos[i]
                    d = np.linalg.norm(link) + 1e-5
                    vel[i] += filament_strength * link / d * 0.01

        pos += vel

    st.session_state.bubbles = bubbles

# ---------------------------
# Run
# ---------------------------
if st.button("Run Simulation"):
    for _ in range(steps):
        update()

# ---------------------------
# Plot
# ---------------------------
fig = go.Figure()

for b in st.session_state.bubbles:
    pos = b["pos"]
    x, y, z = pos[:,0], pos[:,1], pos[:,2]

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
# Metrics
# ---------------------------
total_particles = sum(len(b["pos"]) for b in st.session_state.bubbles)
st.metric("Total Particles", total_particles)
