import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout="wide")
st.title("Galaxy Formation Discovery Engine — Coherence Enabled")

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

if "ci_history" not in st.session_state:
    st.session_state.ci_history = []

if "last_positions" not in st.session_state:
    st.session_state.last_positions = None

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
# COHERENCE INDEX
# ---------------------------
def compute_coherence_index(bubbles):

    all_pos = np.vstack([b["pos"] for b in bubbles])
    all_vel = np.vstack([b["vel"] for b in bubbles])

    center = np.mean(all_pos, axis=0)
    rel_pos = all_pos - center

    # Radial order
    distances = np.linalg.norm(rel_pos, axis=1)
    radial_score = 1 / (np.std(distances) + 1e-5)

    # Velocity alignment
    vel_norms = np.linalg.norm(all_vel, axis=1) + 1e-5
    norm_vel = all_vel / vel_norms[:, None]
    alignment = np.mean(np.dot(norm_vel, norm_vel.T))
    alignment_score = np.clip(alignment, 0, 1)

    # Temporal stability
    if st.session_state.last_positions is not None:
        delta = all_pos - st.session_state.last_positions
        stability = 1 / (np.mean(np.linalg.norm(delta, axis=1)) + 1e-5)
    else:
        stability = 0.0

    st.session_state.last_positions = all_pos.copy()

    CI = (0.4 * radial_score +
          0.3 * alignment_score +
          0.3 * stability)

    return CI

# ---------------------------
# RUN / RESET
# ---------------------------
col1, col2 = st.columns(2)

if col1.button("Run Simulation"):
    for _ in range(steps):
        st.session_state.bubbles = update(st.session_state.bubbles)

        ci = compute_coherence_index(st.session_state.bubbles)
        st.session_state.ci_history.append(ci)

if col2.button("Reset"):
    st.session_state.bubbles = init_sim()
    st.session_state.history = {"coherence": []}
    st.session_state.ci_history = []
    st.session_state.last_positions = None

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
# COHERENCE DASHBOARD
# ---------------------------
st.subheader("Coherence Index Monitor")

if len(st.session_state.ci_history) > 0:

    current_ci = st.session_state.ci_history[-1]
    threshold = st.slider("Coherence Threshold", 0.0, 5.0, 1.5)

    c1, c2 = st.columns(2)

    c1.metric("Current CI", f"{current_ci:.3f}")

    if current_ci > threshold:
        c2.success("COHERENCE LOCK DETECTED")
    else:
        c2.warning("Below Threshold")

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        y=st.session_state.ci_history,
        mode='lines'
    ))
    fig_ci.add_hline(y=threshold)

    st.plotly_chart(fig_ci, use_container_width=True)

# ---------------------------
# REAL DATA IMPORT
# ---------------------------
st.subheader("Import Real Galaxy Data")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def normalize_points(points):
    points = points - np.mean(points, axis=0)
    scale = np.max(np.linalg.norm(points, axis=1)) + 1e-5
    return points / scale

real_data = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if {"x","y","z"}.issubset(df.columns):
        real_data = normalize_points(df[["x","y","z"]].values)
        st.success("Dataset loaded")

# ---------------------------
# CORRELATION FUNCTION
# ---------------------------
st.subheader("Two-Point Correlation")

from scipy.spatial import distance_matrix

def compute_correlation(points):
    dists = distance_matrix(points, points)
    dists = dists[np.triu_indices_from(dists, k=1)]

    hist, edges = np.histogram(dists, bins=20)
    rand = np.random.uniform(points.min(), points.max(), points.shape)
    rand_dists = distance_matrix(rand, rand)
    rand_dists = rand_dists[np.triu_indices_from(rand_dists, k=1)]
    rand_hist, _ = np.histogram(rand_dists, bins=edges)

    corr = (hist / (rand_hist + 1e-5)) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, corr

if st.button("Compute Correlation"):
    sim = normalize_points(np.vstack([b["pos"] for b in st.session_state.bubbles]))
    x, y = compute_correlation(sim)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Sim'))

    if real_data is not None:
        xr, yr = compute_correlation(real_data)
        fig.add_trace(go.Scatter(x=xr, y=yr, mode='lines', name='Real'))

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# POWER SPECTRUM
# ---------------------------
st.subheader("Power Spectrum")

def compute_power(points):
    grid_size = 32
    grid = np.zeros((grid_size,)*3)

    pts = (points - points.min(axis=0)) / (points.max(axis=0)-points.min(axis=0)+1e-5)
    idx = (pts * (grid_size-1)).astype(int)

    for i in idx:
        grid[i[0], i[1], i[2]] += 1

    fft = np.fft.fftn(grid)
    power = np.abs(fft)**2

    k = np.fft.fftfreq(grid_size)
    kx, ky, kz = np.meshgrid(k, k, k)
    kmag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
    power = power.flatten()

    bins = np.linspace(0, np.max(kmag), 20)
    vals = []

    for i in range(len(bins)-1):
        mask = (kmag >= bins[i]) & (kmag < bins[i+1])
        vals.append(np.mean(power[mask]) if np.any(mask) else 0)

    centers = 0.5*(bins[:-1]+bins[1:])
    return centers, vals

if st.button("Compute Power Spectrum"):
    sim = normalize_points(np.vstack([b["pos"] for b in st.session_state.bubbles]))
    k, p = compute_power(sim)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=p, mode='lines', name='Sim'))

    if real_data is not None:
        kr, pr = compute_power(real_data)
        fig.add_trace(go.Scatter(x=kr, y=pr, mode='lines', name='Real'))

    st.plotly_chart(fig, use_container_width=True)
