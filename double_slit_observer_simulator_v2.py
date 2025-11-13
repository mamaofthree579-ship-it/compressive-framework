#!/usr/bin/env python3
"""
parameter_sweep_double_slit_3d.py

3D parameter sweep for the Compressive Framework double-slit simulation.
Explores how observer strength, memory coupling, and residual decay
jointly influence emergent particle formation patterns.
"""

import streamlit as st
import numpy as np
from scipy import ndimage
import plotly.graph_objects as go

st.set_page_config(page_title="3D Parameter Sweep — Double Slit", layout="wide")
st.title("🌌 3D Parameter Sweep — Double-Slit Simulation")
st.markdown(
    """
    This simulator explores how *observer strength*, *memory coupling*, and *residual decay*
    collectively shape the emergence of particles in the Compressive Framework model of
    the double-slit experiment.

    Each point in the 3D space represents one simulation run.
    Color = average number of detected particles.
    """
)

# Sidebar controls
st.sidebar.header("Simulation Controls")
nx = st.sidebar.slider("Grid Resolution", 80, 200, 120, step=20)
frames = st.sidebar.slider("Frames per Run", 10, 60, 30, step=10)
observer_type = st.sidebar.selectbox("Observer Type", ["detector", "human", "instrument"])
run_sweep = st.sidebar.button("Run 3D Sweep")

# Parameter ranges
obs_strengths = np.linspace(0.0, 2.5, 6)
mem_couplings = np.linspace(0.0, 1.0, 6)
residual_decays = np.linspace(0.90, 0.99, 4)

detector_sigma = 0.8
human_noise_amp = 0.15
human_bias_x = -0.5

# Grid setup
x = np.linspace(-6, 6, nx)
y = np.linspace(0, 6, nx // 2)
X, Y = np.meshgrid(x, y)

# Base wave interference
def base_field(X, Y, sep=1.0, k=2.2):
    slit1 = np.exp(-((X + sep)**2 + (Y-0.5)**2) / 0.25)
    slit2 = np.exp(-((X - sep)**2 + (Y-0.5)**2) / 0.25)
    phase = np.exp(1j * (k * np.sqrt((X**2 + Y**2) + 1e-9)))
    return phase * (slit1 + slit2)

# Observer modulation
def observer_field(X, Y, obs_type, strength, t):
    if obs_type == "detector":
        gauss = np.exp(-((X)**2 + (Y-0.4)**2) / (2 * detector_sigma**2))
        return strength * gauss * np.sin(6.0 * X + 1.5 * np.sin(t))
    if obs_type == "instrument":
        return strength * 0.6 * np.sin(0.8 * X + 0.3 * np.cos(0.5 * t)) * np.exp(-0.15*(X**2 + Y**2))
    if obs_type == "human":
        rand_comp = np.sin(1.8*X + 2.5*Y + 0.9*t) + 0.5*np.cos(2.2*X - 1.2*Y + 0.3*t)
        bias = np.exp(-((X - human_bias_x)**2 + Y**2) / 2.2)
        noise = human_noise_amp * np.random.randn(*X.shape)
        return strength * (rand_comp * bias + noise)
    return np.zeros_like(X)

# Helpers
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

# Core simulation
def run_simulation(observer_strength, memory_coupling, residual_decay):
    psi_base = base_field(X, Y)
    residual = np.zeros_like(psi_base.real)
    particle_counts = []
    for f in range(frames):
        t = 2*np.pi*f/frames
        psi_t = psi_base * np.cos(0.4 * t)
        obs = observer_field(X, Y, observer_type, observer_strength, t)
        mem_feedback = memory_coupling * (residual - np.mean(residual))
        psi_total = psi_t + (obs + mem_feedback)
        energy = np.abs(psi_total)**2
        energy_n = normalize(energy)
        residual = residual * residual_decay + 0.02 * energy_n
        count = detect_particles(residual)
        particle_counts.append(count)
    return np.mean(particle_counts)

# Main run
if run_sweep:
    st.info("Running 3D parameter sweep... please wait ⏳")

    total_points = len(obs_strengths) * len(mem_couplings) * len(residual_decays)
    results = []
    progress = st.progress(0)
    done = 0

    for obs in obs_strengths:
        for mem in mem_couplings:
            for decay in residual_decays:
                avg_particles = run_simulation(obs, mem, decay)
                results.append((obs, mem, decay, avg_particles))
                done += 1
                progress.progress(done / total_points)

    st.success("✅ Sweep completed")

    # Convert to numpy arrays for plotting
    data = np.array(results)
    obs_vals, mem_vals, decay_vals, particle_counts = data.T

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=obs_vals,
        y=mem_vals,
        z=decay_vals,
        mode='markers',
        marker=dict(
            size=6,
            color=particle_counts,
            colorscale='Viridis',
            colorbar=dict(title='Avg Particle Count'),
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

    # Interpretive summary
    st.markdown(
        """
        ### 🔍 Interpretation
        - **Bright regions:** strong particle collapse due to feedback resonance.
        - **Dark regions:** coherent interference retained (wave-dominant).
        - **High λ_mem & γ:** energy builds slowly but persists — sustained observation bias.
        - **Human observers:** introduce higher entropy due to stochastic noise, leading to non-linear collapse dynamics.
        """
    )
