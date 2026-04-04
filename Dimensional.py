import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout="wide")
st.title("Galaxy Formation Discovery Engine — Real Data Integrated")

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
# UPDATE STEP
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
# METRICS
# ---------------------------
def compute_coherence(bubbles):
    all_pos = np.vstack([b["pos"] for b in bubbles])
    center = np.mean(all_pos, axis=0)
    dist = np.linalg.norm(all_pos - center, axis=1)
    return 1 / (np.std(dist) + 1e-5)

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
    eigvals = np.sort(np.abs(np.linalg.eigvals(cov)))
    planarity = eigvals[0] / (eigvals[-1] + 1e-5)

    if ang_mag > 0.05 and planarity < 0.3:
        return "Spiral-like"
    elif spread < 0.5:
        return "Clustered"
    else:
        return "Void / Diffuse"

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
structure = classify_structure(st.session_state.bubbles)

st.write(f"**Phase:** {phase}")
st.write(f"**Detected Structure:** {structure}")

# ---------------------------
# PARAMETER SWEEP
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

            heatmap[i, j] = ["Void / Diffuse","Clustered","Spiral-like"].index(classify_structure(bubbles))

    fig3 = go.Figure(data=go.Heatmap(z=heatmap, x=filament_vals, y=shell_vals))
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# REAL DATA IMPORT
# ---------------------------
st.subheader("Import Real Galaxy Data")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def normalize_points(points):
    points = points - np.mean(points, axis=0)
    scale = np.max(np.linalg.norm(points, axis=1)) + 1e-5
    return points / scale

def convert_spherical_to_cartesian(ra, dec, z):
    ra = np.radians(ra)
    dec = np.radians(dec)

    x = z * np.cos(dec) * np.cos(ra)
    y = z * np.cos(dec) * np.sin(ra)
    z = z * np.sin(dec)

    return np.vstack((x, y, z)).T

real_data = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if {"x","y","z"}.issubset(df.columns):
        real_data = df[["x","y","z"]].values

    elif {"ra","dec","redshift"}.issubset(df.columns):
        real_data = convert_spherical_to_cartesian(
            df["ra"].values,
            df["dec"].values,
            df["redshift"].values
        )

    if real_data is not None:
        real_data = normalize_points(real_data)
        st.success("Dataset loaded")

# ---------------------------
# OVERLAY + METRICS
# ---------------------------
if real_data is not None:

    sim_points = np.vstack([b["pos"] for b in st.session_state.bubbles])
    sim_points = normalize_points(sim_points)

    fig_real = go.Figure()

    fig_real.add_trace(go.Scatter3d(
        x=sim_points[:,0], y=sim_points[:,1], z=sim_points[:,2],
        mode='markers',
        name='Simulation'
    ))

    fig_real.add_trace(go.Scatter3d(
        x=real_data[:,0], y=real_data[:,1], z=real_data[:,2],
        mode='markers',
        name='Real Data',
        opacity=0.3
    ))

    st.plotly_chart(fig_real, use_container_width=True)

    def compare_real(sim, real):
        sim_dist = np.linalg.norm(sim, axis=1)
        real_dist = np.linalg.norm(real, axis=1)

        density = 1 / (abs(np.std(sim_dist) - np.std(real_dist)) + 1e-5)
        spread = 1 / (abs(np.mean(sim_dist) - np.mean(real_dist)) + 1e-5)

        sim_cov = np.cov(sim.T)
        real_cov = np.cov(real.T)

        sim_eig = np.sort(np.linalg.eigvals(sim_cov))
        real_eig = np.sort(np.linalg.eigvals(real_cov))

        structure = 1 / (np.linalg.norm(sim_eig - real_eig) + 1e-5)

        return density, spread, structure

    d, s, stc = compare_real(sim_points, real_data)

    c1, c2, c3 = st.columns(3)
    c1.metric("Density Match", f"{d:.3f}")
    c2.metric("Spread Match", f"{s:.3f}")
    c3.metric("Structure Match", f"{stc:.3f}")

# ---------------------------
# CORRELATION FUNCTION
# ---------------------------
st.subheader("Two-Point Correlation Function")

def compute_correlation(points, bins=20):
    from scipy.spatial import distance_matrix

    dists = distance_matrix(points, points)
    dists = dists[np.triu_indices_from(dists, k=1)]

    hist, edges = np.histogram(dists, bins=bins)
    random = np.random.uniform(0, np.max(points), points.shape)
    rand_dists = distance_matrix(random, random)
    rand_dists = rand_dists[np.triu_indices_from(rand_dists, k=1)]

    rand_hist, _ = np.histogram(rand_dists, bins=edges)

    correlation = (hist / (rand_hist + 1e-5)) - 1

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, correlation

if st.button("Compute Correlation"):

    sim_points = np.vstack([b["pos"] for b in st.session_state.bubbles])
    sim_points = normalize_points(sim_points)

    x_sim, y_sim = compute_correlation(sim_points)

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=x_sim, y=y_sim, mode='lines', name='Simulation'))

    if real_data is not None:
        x_real, y_real = compute_correlation(real_data)
        fig_corr.add_trace(go.Scatter(x=x_real, y=y_real, mode='lines', name='Real Data'))

    st.plotly_chart(fig_corr, use_container_width=True)
