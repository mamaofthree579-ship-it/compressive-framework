import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Galaxy Formation Discovery Engine — Full System")

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
# INITIALIZATION
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
        return 2
    elif spread < 0.5:
        return 1
    else:
        return 0

def label_structure(val):
    return ["Void / Diffuse", "Clustered", "Spiral-like"][val]

# ---------------------------
# REFERENCE COSMIC WEB
# ---------------------------
def generate_reference_web():
    points = []

    for _ in range(5):
        start = np.random.uniform(-2, 2, 3)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        for t in np.linspace(-2, 2, 100):
            p = start + direction * t + np.random.normal(0, 0.05, 3)
            points.append(p)

    for _ in range(5):
        center = np.random.uniform(-2, 2, 3)
        cluster = center + np.random.normal(0, 0.2, (50, 3))
        points.extend(cluster)

    return np.array(points)

# ---------------------------
# STRUCTURE COMPARISON
# ---------------------------
def compare_structures(sim_bubbles, ref_points):
    sim_points = np.vstack([b["pos"] for b in sim_bubbles])

    sim_dist = np.linalg.norm(sim_points - np.mean(sim_points, axis=0), axis=1)
    ref_dist = np.linalg.norm(ref_points - np.mean(ref_points, axis=0), axis=1)

    density = 1 / (abs(np.std(sim_dist) - np.std(ref_dist)) + 1e-5)
    cluster = 1 / (abs(np.mean(sim_dist) - np.mean(ref_dist)) + 1e-5)

    sim_cov = np.cov(sim_points.T)
    ref_cov = np.cov(ref_points.T)

    sim_eig = np.sort(np.linalg.eigvals(sim_cov))
    ref_eig = np.sort(np.linalg.eigvals(ref_cov))

    filament = 1 / (np.linalg.norm(sim_eig - ref_eig) + 1e-5)

    return density, cluster, filament

# ---------------------------
# RUN / RESET
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

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TIME EVOLUTION
# ---------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=st.session_state.history["coherence"], mode='lines'))
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# STATUS
# ---------------------------
phase = detect_phase(st.session_state.history["coherence"])
structure = label_structure(classify_structure(st.session_state.bubbles))

st.write(f"**Phase:** {phase}")
st.write(f"**Detected Structure:** {structure}")

# ---------------------------
# PARAMETER SWEEP MAP
# ---------------------------
st.subheader("Structure Map")

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

                        p += f_val * np.random.randn(3) * 0.01

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

    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# REALITY COMPARISON
# ---------------------------
st.subheader("Reality Comparison")

if st.button("Compare to Reference"):
    ref = generate_reference_web()
    d, c, f = compare_structures(st.session_state.bubbles, ref)

    col1, col2, col3 = st.columns(3)
    col1.metric("Density Match", f"{d:.3f}")
    col2.metric("Cluster Match", f"{c:.3f}")
    col3.metric("Filament Match", f"{f:.3f}")

    fig_compare = go.Figure()

    sim_points = np.vstack([b["pos"] for b in st.session_state.bubbles]

    )

    fig_compare.add_trace(go.Scatter3d(
        x=sim_points[:,0], y=sim_points[:,1], z=sim_points[:,2],
        mode='markers',
        name='Simulation'
    ))

    fig_compare.add_trace(go.Scatter3d(
        x=ref[:,0], y=ref[:,1], z=ref[:,2],
        mode='markers',
        name='Reference',
        opacity=0.3
    ))

    st.plotly_chart(fig_compare, use_container_width=True)
