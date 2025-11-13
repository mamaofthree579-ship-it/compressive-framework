#!/usr/bin/env python3
"""
parameter_sweep_double_slit_3d_time_recorder_v2.py

Compressive Framework Quantum Double-Slit Simulator
- 3D parameter sweep
- Time evolution visualizer
- GIF + MP4 export with timestamp overlays
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage
import imageio
import os
from moviepy.editor import ImageSequenceClip

# ───────────────────────────────────────────────
# Streamlit setup
# ───────────────────────────────────────────────
st.set_page_config(page_title="Quantum Double-Slit — 3D Sweep + Recorder v2", layout="wide")

st.title("🌌 Quantum Double-Slit — 3D Sweep + Temporal Recorder (v2)")
st.markdown(
"""
This simulator explores **observer-induced decoherence** in the **quantum double-slit experiment**,  
and now includes **MP4 & GIF recording with timestamp overlays**.
"""
)

# ───────────────────────────────────────────────
# Sidebar controls
# ───────────────────────────────────────────────
st.sidebar.header("🎛 Simulation Controls")
nx = st.sidebar.slider("Grid Resolution", 80, 200, 120, step=20)
frames = st.sidebar.slider("Frames per Run", 10, 60, 30, step=10)
observer_type = st.sidebar.selectbox("Observer Type", ["detector", "human", "instrument"])
run_sweep = st.sidebar.button("Run 3D Sweep")
record_gif = st.sidebar.button("🎥 Record Simulation (GIF)")
record_mp4 = st.sidebar.button("🎬 Record Simulation (MP4)")

# ───────────────────────────────────────────────
# Parameters
# ───────────────────────────────────────────────
obs_strengths = np.linspace(0.0, 2.5, 6)
mem_couplings = np.linspace(0.0, 1.0, 6)
residual_decays = np.linspace(0.90, 0.99, 4)

detector_sigma = 0.8
human_noise_amp = 0.15
human_bias_x = -0.5

# ───────────────────────────────────────────────
# Spatial grid
# ───────────────────────────────────────────────
x = np.linspace(-6, 6, nx)
y = np.linspace(0, 6, nx // 2)
X, Y = np.meshgrid(x, y)

# ───────────────────────────────────────────────
# Field definitions
# ───────────────────────────────────────────────
def base_field(X, Y, sep=1.0, k=2.2):
    slit1 = np.exp(-((X + sep)**2 + (Y - 0.5)**2) / 0.25)
    slit2 = np.exp(-((X - sep)**2 + (Y - 0.5)**2) / 0.25)
    phase = np.exp(1j * (k * np.sqrt((X**2 + Y**2) + 1e-9)))
    return phase * (slit1 + slit2)

def observer_field(X, Y, obs_type, strength, t):
    if obs_type == "detector":
        gauss = np.exp(-((X)**2 + (Y - 0.4)**2) / (2 * detector_sigma**2))
        return strength * gauss * np.sin(6.0 * X + 1.5 * np.sin(t))
    if obs_type == "instrument":
        return strength * 0.6 * np.sin(0.8 * X + 0.3 * np.cos(0.5 * t)) * np.exp(-0.15 * (X**2 + Y**2))
    if obs_type == "human":
        rand_comp = np.sin(1.8 * X + 2.5 * Y + 0.9 * t) + 0.5 * np.cos(2.2 * X - 1.2 * Y + 0.3 * t)
        bias = np.exp(-((X - human_bias_x)**2 + Y**2) / 2.2)
        noise = human_noise_amp * np.random.randn(*X.shape)
        return strength * (rand_comp * bias + noise)
    return np.zeros_like(X)

def normalize(arr):
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    if amax - amin < 1e-12:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)

def detect_particles(field, sensitivity=1.2, min_area=6):
    thresh = np.mean(field) * sensitivity + 1e-6
    mask = field > thresh
    labeled, n = ndimage.label(mask)
    objs = ndimage.find_objects(labeled)
    count = 0
    for i, slc in enumerate(objs):
        if slc is None:
            continue
        region = (labeled[slc] == (i + 1))
        area = int(region.sum())
        if area >= min_area:
            count += 1
    return count

def run_simulation(observer_strength, memory_coupling, residual_decay, frames=frames):
    psi_base = base_field(X, Y)
    residual = np.zeros_like(psi_base.real)
    particle_counts = []
    frames_data = []

    for f in range(frames):
        t = 2 * np.pi * f / frames
        psi_t = psi_base * np.cos(0.4 * t)
        obs = observer_field(X, Y, observer_type, observer_strength, t)
        mem_feedback = memory_coupling * (residual - np.mean(residual))
        psi_total = psi_t + (obs + mem_feedback)
        energy = np.abs(psi_total)**2
        energy_n = normalize(energy)
        residual = residual * residual_decay + 0.02 * energy_n
        count = detect_particles(residual)
        particle_counts.append(count)
        frames_data.append(residual.copy())

    return np.mean(particle_counts), frames_data

# ───────────────────────────────────────────────
# Parameter sweep visualization
# ───────────────────────────────────────────────
if run_sweep:
    st.info("Running 3D parameter sweep... please wait ⏳")

    total_points = len(obs_strengths) * len(mem_couplings) * len(residual_decays)
    results = []
    progress = st.progress(0)
    done = 0

    for obs in obs_strengths:
        for mem in mem_couplings:
            for decay in residual_decays:
                avg_particles, _ = run_simulation(obs, mem, decay)
                results.append((obs, mem, decay, avg_particles))
                done += 1
                progress.progress(done / total_points)

    st.success("✅ Sweep completed")

    data = np.array(results)
    obs_vals, mem_vals, decay_vals, particle_counts = data.T

    fig = go.Figure(data=[go.Scatter3d(
        x=obs_vals, y=mem_vals, z=decay_vals,
        mode='markers',
        marker=dict(
            size=6, color=particle_counts,
            colorscale='Viridis', colorbar=dict(title='Avg Particle Count'),
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Observer Strength (s_obs)',
            yaxis_title='Memory Coupling (λ_mem)',
            zaxis_title='Residual Decay (γ)',
        ),
        title=f"3D Parameter Influence — Observer: {observer_type.capitalize()}",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────
# Time evolution + Recorder
# ───────────────────────────────────────────────
st.header("🌀 Temporal Field Evolution")

obs_choice = st.select_slider("Observer Strength", obs_strengths.tolist(), value=1.0)
mem_choice = st.select_slider("Memory Coupling", mem_couplings.tolist(), value=0.5)
decay_choice = st.select_slider("Residual Decay", residual_decays.tolist(), value=0.95)

avg_count, frames_data = run_simulation(obs_choice, mem_choice, decay_choice)
show_time = st.slider("Frame (Time Step)", 0, frames - 1, 0)

frame_to_show = frames_data[show_time]

fig2, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(frame_to_show, cmap='plasma', extent=[-6, 6, 0, 6], origin='lower')
ax.set_title(f"Frame {show_time+1}/{frames} | s={obs_choice}, λ={mem_choice}, γ={decay_choice}")
plt.colorbar(im, ax=ax, label="Curvature Intensity")
st.pyplot(fig2)

# ───────────────────────────────────────────────
# GIF / MP4 Recording
# ───────────────────────────────────────────────
def record_simulation(frames_data, fmt="gif"):
    st.info(f"Recording simulation as {fmt.upper()} — please wait...")
    images = []
    os.makedirs("frames_tmp", exist_ok=True)

    for i, frame in enumerate(frames_data):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(frame, cmap='plasma', extent=[-6, 6, 0, 6], origin='lower')
        ax.set_title(f"t={i+1}/{len(frames_data)} | s={obs_choice}, λ={mem_choice}, γ={decay_choice}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        plt.tight_layout()
        filename = f"frames_tmp/frame_{i:03d}.png"
        plt.savefig(filename, dpi=100)
        plt.close(fig)
        images.append(imageio.imread(filename))

    if fmt == "gif":
        gif_name = "quantum_double_slit_sim.gif"
        imageio.mimsave(gif_name, images, fps=8)
        out_file = gif_name
    else:
        clip = ImageSequenceClip("frames_tmp", fps=8)
        mp4_name = "quantum_double_slit_sim.mp4"
        clip.write_videofile(mp4_name, codec="libx264", audio=False, verbose=False, logger=None)
        out_file = mp4_name

    for f in os.listdir("frames_tmp"):
        os.remove(os.path.join("frames_tmp", f))
    os.rmdir("frames_tmp")

    return out_file

if record_gif:
    out_file = record_simulation(frames_data, fmt="gif")
    st.success("✅ GIF recording complete!")
    with open(out_file, "rb") as f:
        st.download_button("📥 Download Simulation GIF", f, file_name=out_file)

if record_mp4:
    out_file = record_simulation(frames_data, fmt="mp4")
    st.success("✅ MP4 recording complete!")
    with open(out_file, "rb") as f:
        st.download_button("🎬 Download Simulation MP4", f, file_name=out_file)

# ───────────────────────────────────────────────
# Interpretation
# ───────────────────────────────────────────────
st.markdown(
"""
### 🔬 Interpretation
- **Detectors** dampen interference amplitude.
- **Instruments** produce structured interference modulation.
- **Humans** introduce stochastic nonlinear decoherence.
- **GIF/MP4 recording** captures the temporal buildup of interference collapse.
"""
)
