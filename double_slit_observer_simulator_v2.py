#!/usr/bin/env python3
"""
parameter_sweep_double_slit.py

Runs a systematic sweep of the double-slit residual memory simulator
across observer strength, memory coupling, and residual decay parameters.

Outputs: heatmap of average particle yield.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import io
import csv

st.set_page_config(page_title="Parameter Sweep — Double Slit", layout="wide")
st.title("🧩 Double-Slit Parameter Sweep")
st.markdown(
    """
    This app runs multiple simulated double-slit scenarios varying observer and memory parameters.
    Each cell in the heatmap shows the **average detected particle count** across frames.

    Parameters explored:
    - **Observer strength (s_obs)**
    - **Memory coupling (λ_mem)**
    - **Residual decay (γ)**

    The simulation model is simplified from the full app for speed but preserves wave interference,
    observer influence, and residual memory behavior.
    """
)

# -----------------------------------
# Sidebar configuration
# -----------------------------------
st.sidebar.header("Sweep configuration")

# Grid controls
nx = st.sidebar.slider("Grid resolution", 80, 200, 120, step=20)
frames = st.sidebar.slider("Frames per run", 10, 100, 40, step=10)

# Sweep range controls
s_obs_range = np.linspace(
    st.sidebar.number_input("Observer strength min", 0.0, 3.0, 0.0, step=0.1),
    st.sidebar.number_input("Observer strength max", 0.0, 3.0, 2.0, step=0.1),
    st.sidebar.slider("Observer strength steps", 2, 10, 5),
)
mem_coupling_range = np.linspace(
    st.sidebar.number_input("Memory coupling min", 0.0, 1.0, 0.0, step=0.05),
    st.sidebar.number_input("Memory coupling max", 0.0, 1.0, 1.0, step=0.05),
    st.sidebar.slider("Memory coupling steps", 2, 10, 5),
)
residual_decay_range = np.linspace(
    st.sidebar.number_input("Residual decay min", 0.8, 1.0, 0.9, step=0.005),
    st.sidebar.number_input("Residual decay max", 0.8, 1.0, 0.99, step=0.005),
    st.sidebar.slider("Residual decay steps", 2, 10, 4),
)

# Observer type
observer_type = st.sidebar.selectbox("Observer type", ["detector", "human", "instrument"])
detector_sigma = 0.8
human_noise_amp = 0.15
human_bias_x = -0.5

# Run button
run_sweep = st.sidebar.button("Run parameter sweep")
export_csv = st.sidebar.checkbox("Enable CSV export", value=True)

# -----------------------------------
# Core model
# -----------------------------------
x = np.linspace(-6, 6, nx)
y = np.linspace(0, 6, nx // 2)
X, Y = np.meshgrid(x, y)

def base_field(X, Y, sep=1.0, k=2.2):
    slit1 = np.exp(-((X + sep)**2 + (Y-0.5)**2) / 0.25)
    slit2 = np.exp(-((X - sep)**2 + (Y-0.5)**2) / 0.25)
    phase = np.exp(1j * (k * np.sqrt((X**2 + Y**2) + 1e-9)))
    return phase * (slit1 + slit2)

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

def run_single_sim(observer_strength, memory_coupling, residual_decay):
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
    return np.mean(particle_counts), np.std(particle_counts)

# -----------------------------------
# Parameter sweep
# -----------------------------------
if run_sweep:
    st.info("Running parameter sweep — this may take several minutes depending on grid size.")
    progress = st.progress(0)
    results = []
    total_runs = len(s_obs_range) * len(mem_coupling_range) * len(residual_decay_range)
    completed = 0

    # For now: vary observer_strength vs. memory_coupling, fix residual_decay to median
    decay_fixed = np.median(residual_decay_range)
    z_data = np.zeros((len(s_obs_range), len(mem_coupling_range)))

    for i, s_obs in enumerate(s_obs_range):
        for j, mem_c in enumerate(mem_coupling_range):
            avg, std = run_single_sim(s_obs, mem_c, decay_fixed)
            z_data[i, j] = avg
            results.append({
                "observer_strength": s_obs,
                "memory_coupling": mem_c,
                "residual_decay": decay_fixed,
                "avg_particles": avg,
                "std_particles": std
            })
            completed += 1
            progress.progress(completed / total_runs)

    # -----------------------------------
    # Heatmap output
    # -----------------------------------
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(z_data, origin="lower", cmap="plasma", aspect="auto",
                   extent=[mem_coupling_range[0], mem_coupling_range[-1],
                           s_obs_range[0], s_obs_range[-1]])
    fig.colorbar(im, ax=ax, label="Avg. particle detections")
    ax.set_xlabel("Memory coupling (λ_mem)")
    ax.set_ylabel("Observer strength (s_obs)")
    ax.set_title(f"Average particle count — observer: {observer_type}, residual_decay={decay_fixed:.3f}")
    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------
    # Optional CSV export
    # -----------------------------------
    if export_csv and results:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        csv_bytes = csv_buf.getvalue().encode("utf-8")
        st.download_button("📥 Download Sweep Data (CSV)", csv_bytes, "parameter_sweep_results.csv", mime="text/csv")

    st.success("Sweep completed.")
    st.markdown("**Interpretation tip:** Brighter areas = higher particle yield → stronger measurement/feedback effects.")
