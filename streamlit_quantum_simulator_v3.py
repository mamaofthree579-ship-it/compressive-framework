import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

st.set_page_config(page_title="Quantum Wave–Particle Simulator", layout="wide")

st.title("🌀 Quantum Wave–Particle Evolution Simulator")

# --- Sidebar controls ---
st.sidebar.header("Simulation Parameters")
nx = st.sidebar.slider("Grid Resolution", 100, 400, 200, step=50)
frames = st.sidebar.slider("Time Steps", 50, 500, 150, step=50)
fluct = st.sidebar.slider("Quantum Fluctuation Amplitude", 0.0, 0.2, 0.05, step=0.01)
leak = st.sidebar.slider("Leakage Rate", 0.0, 0.1, 0.02, step=0.005)
decay = st.sidebar.slider("Residual Decay", 0.90, 0.999, 0.985)
thresh = st.sidebar.slider("Particle Threshold", 0.05, 0.3, 0.12, step=0.01)
min_area = st.sidebar.slider("Minimum Particle Area", 3, 20, 6, step=1)

st.sidebar.markdown("---")
start_sim = st.sidebar.button("▶ Run Simulation")

# --- Core simulation setup ---
x = np.linspace(-8, 8, nx)
y = np.linspace(0, 8, nx // 2)
X, Y = np.meshgrid(x, y)

source_sep = 2.0
r1 = np.sqrt((X + source_sep)**2 + Y**2 + 1e-6)
r2 = np.sqrt((X - source_sep)**2 + Y**2 + 1e-6)
k = 2.0
psi_base = np.sin(k * r1) / (r1 + 1e-6) + np.sin(k * r2) / (r2 + 1e-6)
obs_amp = 0.35
obs_k = 0.6
obs_field = obs_amp * np.sin(obs_k * X)

residual = np.zeros_like(psi_base)

# --- Helper: detect particle nucleation sites ---
def detect_particles(resid, thresh, min_area):
    mask = resid > thresh
    labeled, n = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    centers = []
    for i, slc in enumerate(objects):
        if slc is None:
            continue
        region = (labeled[slc] == (i + 1))
        area = region.sum()
        if area >= min_area:
            cy, cx = ndimage.center_of_mass(region)
            row0, col0 = slc[0].start, slc[1].start
            centers.append((row0 + cy, col0 + cx))
    return centers

# --- Streamlit simulation display ---
placeholder = st.empty()

if start_sim:
    st.info("Simulating wave evolution... please wait.")
    for frame in range(frames):
        t = 2 * np.pi * (frame / frames)
        psi_time = psi_base * np.cos(0.8 * t) + obs_field * (0.5 * np.sin(0.7 * t))
        noise = fluct * np.random.randn(*psi_time.shape)
        psi_noisy = psi_time + noise
        energy = psi_noisy**2
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-12)
        residual = residual * decay + leak * energy_norm

        centers = detect_particles(residual, thresh, min_area)

        # Prepare plot
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            energy_norm,
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=1
        )

        if centers:
            coords = []
            for r, c in centers:
                yy = np.interp(r, [0, residual.shape[0]-1], [y[0], y[-1]])
                xx = np.interp(c, [0, residual.shape[1]-1], [x[0], x[-1]])
                coords.append((xx, yy))
            coords = np.array(coords)
            ax.scatter(coords[:, 0], coords[:, 1], s=60, facecolors='none', edgecolors='white')

        ax.set_title(f"Frame {frame+1}/{frames} — Particles: {len(centers)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, label="Energy Density |ψ|²")

        placeholder.pyplot(fig)
        plt.close(fig)

    st.success("✅ Simulation complete!")
else:
    st.warning("Press ▶ **Run Simulation** to start.")
