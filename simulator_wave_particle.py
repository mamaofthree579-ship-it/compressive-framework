#!/usr/bin/env python3
"""
streamlit_simulator.py

Streamlit front-end for visualizing the Compressive Framework process:
quantum waveform movement, fluctuations, residual leakage, and particle formation.

Run with:
    streamlit run streamlit_simulator.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ---------------------------
# Simulation core functions
# ---------------------------

def build_grid(nx, xlim=(-6, 6)):
    x = np.linspace(xlim[0], xlim[1], nx)
    X = np.tile(x, (nx, 1))
    return x, X

def compute_wavefields(X, t_phase, params):
    alpha, beta, gamma, omega, env_sigma = (
        params["alpha"], params["beta"], params["gamma"],
        params["omega"], params["env_sigma"]
    )
    env = np.exp(-(X ** 2) / (2.0 * env_sigma ** 2))
    graviton = np.sin(alpha * (omega * X - t_phase))
    chronon = np.sin(beta * (omega * X - 0.7 * t_phase) + 0.3)
    cognon = np.sin(gamma * (omega * X - 1.3 * t_phase) + 0.7)
    combined = (graviton + chronon + cognon) * env
    return combined

def normalize(A):
    amin, amax = np.nanmin(A), np.nanmax(A)
    return (A - amin) / (amax - amin + 1e-8)

def detect_particles(residual, threshold, min_pixels=3):
    mask = residual > threshold
    labeled, n = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    particles = []
    for i, slc in enumerate(objects):
        if slc is None:
            continue
        region = labeled[slc] == (i + 1)
        area = int(region.sum())
        if area >= min_pixels:
            cy, cx = ndimage.center_of_mass(region)
            row0, col0 = slc[0].start, slc[1].start
            centroid = (row0 + cy, col0 + cx)
            particles.append(centroid)
    return particles

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Quantum Compressive Simulator", layout="wide")
st.title("🌀 Quantum Wave–Particle Formation Simulator")
st.markdown("""
This interactive simulator visualizes **wave superposition**, **quantum fluctuations**,  
**residual buildup (leakage)**, and **particle nucleation** within the **Compressive Framework**.

Adjust parameters in the sidebar and press **▶ Run Simulation**.
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
nx = st.sidebar.slider("Grid resolution (nx)", 100, 600, 300, 50)
frames = st.sidebar.slider("Frames", 50, 600, 200, 50)
alpha = st.sidebar.slider("Graviton scale (α)", 0.1, 1.5, 0.7, 0.1)
beta = st.sidebar.slider("Chronon scale (β)", 0.1, 1.5, 0.5, 0.1)
gamma = st.sidebar.slider("Cognon scale (γ)", 0.1, 1.5, 0.6, 0.1)
omega = st.sidebar.slider("Base frequency (ω)", 0.5, 6.0, 3.0, 0.1)
env_sigma = st.sidebar.slider("Envelope σ", 0.5, 5.0, 3.0, 0.1)
fluct = st.sidebar.slider("Quantum fluctuation amplitude", 0.0, 0.2, 0.05, 0.01)
decay = st.sidebar.slider("Residual decay per frame", 0.90, 1.00, 0.98, 0.005)
leak = st.sidebar.slider("Residual leak rate", 0.01, 0.10, 0.03, 0.005)
thresh = st.sidebar.slider("Particle threshold", 0.05, 0.3, 0.12, 0.01)
minpix = st.sidebar.slider("Minimum particle pixels", 1, 10, 3)

run_button = st.sidebar.button("▶ Run Simulation")

# ---------------------------
# Simulation logic
# ---------------------------

if run_button:
    x, X = build_grid(nx)
    residual = np.zeros_like(X)
    params = dict(alpha=alpha, beta=beta, gamma=gamma, omega=omega, env_sigma=env_sigma)

    frame_list = np.linspace(0, 2*np.pi, frames)
    counts = []
    last_particles = None

    progress = st.progress(0)
    status = st.empty()
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1])
    plt.subplots_adjust(hspace=0.35)

    for i, t_phase in enumerate(frame_list):
        combined = compute_wavefields(X, t_phase, params)
        combined += np.random.randn(*combined.shape) * fluct
        energy = combined ** 2
        energy_n = normalize(energy)
        residual = residual * decay + leak * energy_n
        particles = detect_particles(residual, thresh, minpix)
        counts.append(len(particles))

        # Plot compressed field
        axs[0].cla()
        axs[0].imshow(energy_n, extent=[x[0], x[-1], 0, 2*np.pi],
                      origin="lower", cmap="plasma", aspect="auto")
        axs[0].set_title(f"Frame {i+1}/{frames} | Particles: {len(particles)}")
        axs[0].set_xlabel("Spatial Coordinate x")
        axs[0].set_ylabel("Phase t")
        if particles:
            coords = np.array([[np.interp(p[1], [0, X.shape[1]-1], [x[0], x[-1]]),
                                 np.interp(p[0], [0, X.shape[0]-1], [0, 2*np.pi])] for p in particles])
            axs[0].scatter(coords[:,0], coords[:,1], s=60, edgecolors="white", facecolors="none")

        # Plot particle count timeline
        axs[1].cla()
        axs[1].plot(counts, '-o', color='black', markersize=3)
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Particle count")
        axs[1].set_xlim(0, frames)
        axs[1].set_ylim(0, max(5, max(counts) + 2))

        progress.progress((i + 1) / frames)
        status.text(f"Simulating frame {i + 1}/{frames}")
        st.pyplot(fig, clear_figure=False)

    progress.progress(1.0)
    status.success("✅ Simulation complete!")
    st.balloons()

    # Summary metrics
    st.markdown("### Simulation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Max particle count", max(counts))
    col2.metric("Average count", f"{np.mean(counts):.2f}")
    col3.metric("Frames simulated", frames)

else:
    st.info("Adjust parameters in the sidebar and click **▶ Run Simulation** to begin.")
