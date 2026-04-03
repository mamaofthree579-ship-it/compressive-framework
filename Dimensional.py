import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Galaxy Formation: Burst + Harmonic Shell + Filament Model")

# ---------------------------
# CONTROLS
# ---------------------------
colA, colB, colC = st.columns(3)

with colA:
    num_bubbles = st.slider("Bubbles", 2, 5, 3)
    particles_per_bubble = st.slider("Particles per Bubble", 50, 200, 100)

with colB:
    burst_strength = st.slider("Burst", 0.1, 2.0, 1.0)
    expansion_rate = st.slider("Expansion", 0.01, 0.1, 0.03)
    compression_strength = st.slider("Compression", 0.0, 0.05, 0.01)

with colC:
    rotation_strength = st.slider("Rotation", 0.0, 0.1, 0.02)
    shell_strength = st.slider("Shell Strength", 0.0, 0.1, 0.03)
    filament_strength = st.slider("Filament", 0.0, 0.05, 0.01)

num_shells = st.slider("Shell Layers", 2, 8, 4)
steps = st.slider("Steps per Run", 1, 10, 3)

# ---------------------------
# INITIALIZE
# ---------------------------
def init_sim():
    bubbles = []
    for _ in range(num_bubbles):
        center = np.random.uniform(-2, 2, 3)
        pos = center + np.random.uniform(-0.1, 0.1, (particles_per_bubble, 3))
        vel = np.zeros((particles_per_bubble, 3))
        bubbles.append({
            "center": center,
            "pos": pos,
            "vel": vel,
            "radius": 0.2
        })
    return bubbles

if "bubbles" not in st.session_state:
    st.session_state.bubbles = init_sim()

# ---------------------------
# SHELL FUNCTION
# ---------------------------
def shell_force(distance, max_radius):
    spacing = max_radius / num_shells
    nearest = np.round(distance / spacing) * spacing
    return nearest - distance

# ---------------------------
# SIMULATION STEP
# ---------------------------
def update():
    bubbles = st.session_state.bubbles

    # Expand bubbles
    for b in bubbles:
        b["radius"] += expansion_rate

    # Bubble-to-bubble attraction (filament seeds)
    for i in range(len(bubbles)):
        for j in range(i + 1, len(bubbles)):
            diff = bubbles[j]["center"] - bubbles[i]["center"]
            dist = np.linalg.norm(diff) + 1e-5
            force = filament_strength * diff / dist

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

            # Harmonic shells
            shell_adj = shell_force(dist, radius)
            vel[i] += shell_strength * shell_adj * dir_norm

            # Filament pull
            for other in bubbles:
                if not np.array_equal(other["center"], center):
                    link = other["center"] - pos[i]
                    d = np.linalg.norm(link) + 1e-5
                    vel[i] += filament_strength * link / d * 0.01

        pos += vel

# ---------------------------
# METRICS
# ---------------------------
def compute_metrics(bubbles):
    all_pos = np.vstack([b["pos"] for b in bubbles])

    # Clustering
    dists = []
    for i in range(len(all_pos)):
        d = np.linalg.norm(all_pos - all_pos[i], axis=1)
        dists.append(np.mean(np.sort(d)[1:6]))
    clustering = 1 / (np.mean(dists) + 1e-5)

    # Shell alignment
    shell_vals = []
    for b in bubbles:
        center = b["center"]
        radius = b["radius"]
        spacing = radius / num_shells

        d = np.linalg.norm(b["pos"] - center, axis=1)
        deviation = np.abs((d / spacing) - np.round(d / spacing))
        shell_vals.append(np.mean(deviation))

    shell_score = 1 / (np.mean(shell_vals) + 1e-5)

    # Filament strength
    centers = np.array([b["center"] for b in bubbles])
    filament = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            filament += 1 / (d + 1e-5)
    filament /= len(centers)

    # Coherence
    center_global = np.mean(all_pos, axis=0)
    dist_global = np.linalg.norm(all_pos - center_global, axis=1)
    coherence = 1 / (np.std(dist_global) + 1e-5)

    return clustering, shell_score, filament, coherence

# ---------------------------
# RUN BUTTONS
# ---------------------------
col1, col2 = st.columns(2)

if col1.button("Run Simulation"):
    for _ in range(steps):
        update()

if col2.button("Reset"):
    st.session_state.bubbles = init_sim()

# ---------------------------
# PLOT
# ---------------------------
fig = go.Figure()

for b in st.session_state.bubbles:
    pos = b["pos"]
    fig.add_trace(go.Scatter3d(
        x=pos[:,0],
        y=pos[:,1],
        z=pos[:,2],
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
# METRICS DISPLAY
# ---------------------------
c, s, f, coh = compute_metrics(st.session_state.bubbles)

st.subheader("System Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clustering", f"{c:.3f}")
m2.metric("Shell Alignment", f"{s:.3f}")
m3.metric("Filament Strength", f"{f:.3f}")
m4.metric("Coherence", f"{coh:.3f}")
