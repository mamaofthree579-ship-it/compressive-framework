#!/usr/bin/env python3
"""
CF-DPF Simulator (Streamlit)
Interactive exploration of Grav/Chron/CoG (graviton, chronon, cognon) field compression
and particle nucleation via constructive interference.

Save as: simulator/app.py
Run: streamlit run simulator/app.py
"""

import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import ndimage

st.set_page_config(page_title="CF-DPF Simulator", layout="wide", initial_sidebar_state="expanded")

# ---- Sidebar: controls ----
st.sidebar.header("Simulation Controls")

# Grid resolution
nx = st.sidebar.selectbox("Spatial resolution (grid points)", [200, 300, 400, 600], index=2)

# Field toggles
st.sidebar.subheader("Fields")
show_graviton = st.sidebar.checkbox("Graviton (γ)", True)
show_chronon = st.sidebar.checkbox("Chronon (χ)", True)
show_cognon  = st.sidebar.checkbox("Cognon (κ)", True)

st.sidebar.subheader("Field weights & params")
alpha = st.sidebar.slider("α — graviton scale", 0.0, 2.0, 0.7, 0.05)
beta  = st.sidebar.slider("β — chronon scale", 0.0, 2.0, 0.5, 0.05)
gamma = st.sidebar.slider("γ — cognon scale", 0.0, 2.0, 0.6, 0.05)

st.sidebar.subheader("Waveform & compression")
omega = st.sidebar.slider("Fundamental frequency ω", 0.5, 8.0, 3.0, 0.1)
envelope_width = st.sidebar.slider("Envelope width (σ for exp(-x²/σ))", 0.5, 10.0, 3.0, 0.1)

st.sidebar.subheader("Particle detection")
detection_threshold = st.sidebar.slider("Nucleation threshold (0..1)", 0.01, 0.5, 0.12, 0.01)
min_area = st.sidebar.slider("Min particle area (pixels)", 1, 50, 5, 1)

# Animation controls (time)
st.sidebar.subheader("Time / animation")
t_val = st.sidebar.slider("Time (phase)", 0.0, 2*np.pi, 0.0, 0.01)
animate = st.sidebar.button("Animate (quick 40 frames)")

# Quick presets
preset = st.sidebar.selectbox("Presets", ["Default", "High Compression", "High Coupling", "Long Envelope"])
if preset == "High Compression":
    alpha, beta, gamma, omega, envelope_width = 1.2, 0.7, 1.0, 4.0, 2.0
elif preset == "High Coupling":
    alpha, beta, gamma, omega, envelope_width = 0.9, 0.9, 0.9, 3.5, 3.0
elif preset == "Long Envelope":
    envelope_width = 8.0

# ---- Helpers: compute fields ----
def build_grid(nx):
    x = np.linspace(-6, 6, nx)
    t = np.linspace(0, 2*np.pi, nx)
    X, T = np.meshgrid(x, t)
    return x, X, T

def compute_fields(X, T, alpha, beta, gamma, omega, t_phase, envelope_width,
                   show_graviton=True, show_chronon=True, show_cognon=True):
    # Basic waveforms (you can replace with more advanced operators)
    graviton = np.sin(alpha * (omega * X - t_phase)) if show_graviton else 0.0
    chronon  = np.sin(beta  * (omega * X - 0.7*t_phase)) if show_chronon else 0.0
    cognon   = np.sin(gamma * (omega * X - 1.3*t_phase)) if show_cognon else 0.0

    # Spatial envelope (compressive region)
    envelope = np.exp(- (X**2) / (2.0 * (envelope_width**2)))

    # Combined amplitude and energy density
    combined = (graviton + chronon + cognon) * envelope
    energy = combined**2  # proxy for local compression energy

    # simple curvature proxy (2nd spatial derivative)
    curvature = ndimage.gaussian_filter1d(combined, sigma=1.0, axis=1, order=2)
    return graviton, chronon, cognon, combined, energy, curvature

def normalize_to_unit(A):
    amin, amax = np.nanmin(A), np.nanmax(A)
    if amax - amin <= 1e-12:
        return np.zeros_like(A)
    return (A - amin) / (amax - amin)

def detect_particles(energy_map, threshold, min_area=3):
    # energy_map assumed normalized [0,1]
    mask = energy_map > threshold
    labeled, n = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    particles = []
    for i, slc in enumerate(objects):
        if slc is None:
            continue
        region = (labeled[slc] == (i+1))
        area = region.sum()
        if area >= min_area:
            # compute centroid in grid coords (row, col)
            cy, cx = ndimage.center_of_mass(region)
            # center_of_mass returns coords relative to slice; convert to full indices:
            row0, col0 = slc[0].start, slc[1].start
            centroid = (row0 + cy, col0 + cx)
            particles.append({"label": i+1, "area": int(area), "centroid": centroid})
    return particles, labeled

# ---- Build grid and compute initial fields ----
x, X, T = build_grid(nx)

gr, ch, co, combined, energy, curvature = compute_fields(
    X, T, alpha, beta, gamma, omega, t_val, envelope_width,
    show_graviton, show_chronon, show_cognon
)

# Normalize energy and curvature for display
energy_norm = normalize_to_unit(energy)
curv_norm = normalize_to_unit(curvature)

# Detect particles
particles, labeled = detect_particles(energy_norm, detection_threshold, min_area=min_area)

# ---- Layout: two-column main area ----
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Compressive Energy Field (Ψ²) — heatmap")
    fig_heat = px.imshow(
        energy_norm,
        origin="lower",
        labels={"x": "x index", "y":"t index", "color":"energy"},
        color_continuous_scale="plasma",
        aspect="auto",
    )
    # mark detected particle centroids
    if particles:
        ys = [p["centroid"][0] for p in particles]
        xs = [p["centroid"][1] for p in particles]
        fig_heat.add_scatter(x=xs, y=ys, mode="markers+text",
                             marker=dict(size=8, color="white", line=dict(width=1, color="black")),
                             text=[f"P{p['label']}" for p in particles],
                             textposition="top center", showlegend=False)
    fig_heat.update_layout(height=640, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(f"**Detected particles:** {len(particles)}  — threshold = {detection_threshold:.3f}, min_area = {min_area}")

with col2:
    st.subheader("Field slices & diagnostics")

    # Show a slice (middle time row)
    mid_row = energy_norm.shape[0] // 2
    fig1, ax1 = plt.subplots(figsize=(6, 2.2))
    ax1.plot(x, combined[mid_row, :], label="Ψ (combined)", color="#2b2b2b")
    ax1.plot(x, energy_norm[mid_row, :], label="normalized energy", color="#ff7f0e", alpha=0.9)
    ax1.set_xlabel("Spatial x")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Spatial slice at mid time (t)")
    st.pyplot(fig1)

    st.markdown("**Curvature intensity (normalized)**")
    fig2, ax2 = plt.subplots(figsize=(6,2))
    ax2.plot(x, curv_norm[mid_row, :], color="#6a5acd")
    ax2.set_xlabel("Spatial x")
    ax2.set_title("Curvature proxy (2nd derivative)")
    st.pyplot(fig2)

    # Show particle table if any
    if particles:
        st.markdown("### Particle list")
        table_rows = []
        for p in particles:
            # convert centroid indices to x,t coordinates for display
            row_idx, col_idx = p["centroid"]
            t_coord = np.interp(row_idx, [0, energy.shape[0]-1], [0, 2*np.pi])
            x_coord = np.interp(col_idx, [0, energy.shape[1]-1], [x[0], x[-1]])
            table_rows.append([p["label"], p["area"], f"{x_coord:.2f}", f"{t_coord:.2f}"])
        st.table({"Label":[r[0] for r in table_rows], "Area":[r[1] for r in table_rows],
                  "x":[r[2] for r in table_rows], "t":[r[3] for r in table_rows]})

# ---- Animation (quick) ----
if animate:
    placeholder = st.empty()
    frames = 40
    for i in range(frames):
        phase = 2*np.pi * i / frames
        _, _, _, _, energy_frame, _ = compute_fields(
            X, T, alpha, beta, gamma, omega, phase, envelope_width,
            show_graviton, show_chronon, show_cognon
        )
        energy_norm_frame = normalize_to_unit(energy_frame)
        fig_anim = px.imshow(energy_norm_frame, origin="lower", color_continuous_scale="plasma", aspect="auto")
        fig_anim.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10))
        placeholder.plotly_chart(fig_anim, use_container_width=True)
    placeholder.empty()

# ---- Footer / tips ----
st.markdown("---")
st.markdown("**Tips:** Try increasing `alpha`, `beta`, or `gamma` to see stronger constructive regions. "
            "Reduce the envelope width to localize compression. Use the detection threshold to tune particle nucleation sensitivity.")
