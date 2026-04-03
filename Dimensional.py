import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Galaxy Formation Discovery Engine")

# ---------------------------
# CONTROLS
# ---------------------------
colA, colB, colC = st.columns(3)

with colA:
    num_bubbles = st.slider("Bubbles", 2, 4, 3)
    particles_per_bubble = st.slider("Particles per Bubble", 50, 150, 80)

with colB:
    burst_strength = st.slider("Burst", 0.1, 2.0, 1.0)
    expansion_rate = st.slider("Expansion", 0.01, 0.1, 0.03)
    compression_strength = st.slider("Compression", 0.0, 0.05, 0.01)

with colC:
    rotation_strength = st.slider("Rotation", 0.0, 0.1, 0.02)
    shell_strength = st.slider("Shell Strength", 0.0, 0.1, 0.03)
    filament_strength = st.slider("Filament", 0.0, 0.05, 0.01)

num_shells = st.slider("Shell Layers", 2, 6, 4)
steps = st.slider("Steps per Run", 1, 8, 3)

# ---------------------------
# INIT
# ---------------------------
def init_sim():
    bubbles = []
    for _ in range(num_bubbles):
        center = np.random.uniform(-2, 2, 3)
        pos = center + np.random.uniform(-0.1, 0.1, (particles_per_bubble, 3))
        vel = np.zeros((particles_per_bubble, 3))
        bubbles.append({"center": center, "pos": pos, "vel": vel, "radius": 0.2})
    return bubbles

if "bubbles" not in st.session_state:
    st.session_state.bubbles = init_sim()

if "history" not in st.session_state:
    st.session_state.history = {"coherence": []}

# ---------------------------
# SHELL FORCE
# ---------------------------
def shell_force(distance, max_radius):
    spacing = max_radius / num_shells
    nearest = np.round(distance / spacing) * spacing
    return nearest - distance

# ---------------------------
# UPDATE
# ---------------------------
def update(bubbles):
    for b in bubbles:
        b["radius"] += expansion_rate

    for i in range(len(bubbles)):
        for j in range(i+1, len(bubbles)):
            diff = bubbles[j]["center"] - bubbles[i]["center"]
            dist = np.linalg.norm(diff) + 1e-5
            force = filament_strength * diff / dist
            bubbles[i]["center"] += force * 0.01
            bubbles[j]["center"] -= force * 0.01

    for b in bubbles:
        center = b["center"]
        pos = b["pos"]
        vel = b["vel"]
        radius = b["radius"]

        for i in range(len(pos)):
            direction = pos[i] - center
            dist = np.linalg.norm(direction) + 1e-5
            dir_norm = direction / dist

            vel[i] += burst_strength * dir_norm * 0.01

            if dist > radius:
                vel[i] -= dir_norm * 0.05

            vel[i] -= compression_strength * dir_norm

            rot = np.array([-direction[1], direction[0], 0])
            vel[i] += rotation_strength * rot

            shell_adj = shell_force(dist, radius)
            vel[i] += shell_strength * shell_adj * dir_norm

        pos += vel

    return bubbles

# ---------------------------
# COHERENCE
# ---------------------------
def compute_coherence(bubbles):
    all_pos = np.vstack([b["pos"] for b in bubbles])
    center = np.mean(all_pos, axis=0)
    dist = np.linalg.norm(all_pos - center, axis=1)
    return 1 / (np.std(dist) + 1e-5)

# ---------------------------
# PHASE DETECTION
# ---------------------------
def detect_phase(series):
    if len(series) < 5:
        return "Initializing"
    recent = series[-5:]
    diffs = np.diff(recent)
    avg = np.mean(np.abs(diffs))

    if avg > 0.1:
        return "Chaotic"
    elif avg > 0.02:
        return "Forming"
    else:
        return "Stable"

# ---------------------------
# CLASSIFICATION
# ---------------------------
def classify_structure(bubbles):
    all_pos = np.vstack([b["pos"] for b in bubbles])
    all_vel = np.vstack([b["vel"] for b in bubbles])

    center = np.mean(all_pos, axis=0)
    rel_pos = all_pos - center

    distances = np.linalg.norm(rel_pos, axis=1)
    spread = np.std(distances)

    angular = np.cross(rel_pos, all_vel)
    ang_mag = np.mean(np.linalg.norm(angular, axis=1))

    cov = np.cov(rel_pos.T)
    eigvals = np.linalg.eigvals(cov)
    eigvals = np.sort(np.abs(eigvals))
    planarity = eigvals[0] / (eigvals[-1] + 1e-5)

    if ang_mag > 0.05 and planarity < 0.3:
        return 2  # Spiral
    elif spread < 0.5:
        return 1  # Cluster
    else:
        return 0  # Void

def label_structure(val):
    return ["Void / Diffuse", "Clustered", "Spiral-like"][val]

# ---------------------------
# RUN
# ---------------------------
col1, col2 = st.columns(2)

if col1.button("Run Simulation"):
    for _ in range(steps):
        st.session_state.bubbles = update(st.session_state.bubbles)
        coh = compute_coherence(st.session_state.bubbles)
        st.session_state.history["coherence"].append(coh)

if col2.button("Reset"):
    st.session_state.bubbles = init_sim()
    st.session_state.history = {"coherence": []}

# ---------------------------
# 3D VISUAL
# ---------------------------
fig = go.Figure()

for b in st.session_state.bubbles:
    pos = b["pos"]
    fig.add_trace(go.Scatter3d(
        x=pos[:,0], y=pos[:,1], z=pos[:,2],
        mode='markers'
    ))

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TIME EVOLUTION
# ---------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    y=st.session_state.history["coherence"],
    mode='lines',
    name="Coherence"
))
st.plotly_chart(fig2, use_container_width=True)

phase = detect_phase(st.session_state.history["coherence"])
st.write(f"**Phase:** {phase}")

structure = label_structure(classify_structure(st.session_state.bubbles))
st.write(f"**Detected Structure:** {structure}")

# ---------------------------
# PARAMETER SWEEP + CLASS MAP
# ---------------------------
st.subheader("Structure Map (Shell vs Filament)")

if st.button("Run Sweep"):

    shell_vals = np.linspace(0.0, 0.1, 8)
    filament_vals = np.linspace(0.0, 0.05, 8)

    heatmap = np.zeros((len(shell_vals), len(filament_vals)))

    for i, s_val in enumerate(shell_vals):
        for j, f_val in enumerate(filament_vals):

            bubbles = init_sim()

            for _ in range(4):
                for b in bubbles:
                    b["radius"] += expansion_rate

                    for p in b["pos"]:
                        direction = p - b["center"]
                        dist = np.linalg.norm(direction) + 1e-5
                        dir_norm = direction / dist

                        p += dir_norm * burst_strength * 0.01
                        p -= dir_norm * compression_strength

                        spacing = b["radius"] / num_shells
                        nearest = round(dist / spacing) * spacing
                        p += s_val * (nearest - dist) * dir_norm

                        p += f_val * (np.random.randn(3)) * 0.01

            heatmap[i, j] = classify_structure(bubbles)

    fig3 = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=filament_vals,
        y=shell_vals,
        colorbar=dict(
            tickvals=[0,1,2],
            ticktext=["Void", "Cluster", "Spiral"]
        )
    ))

    fig3.update_layout(
        xaxis_title="Filament Strength",
        yaxis_title="Shell Strength"
    )

    st.plotly_chart(fig3, use_container_width=True)
