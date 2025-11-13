#!/usr/bin/env python3
"""
double_slit_observer_simulator_v4.py

Quantum double-slit simulator with observer types + temporal residual memory.
Run:
    streamlit run double_slit_observer_simulator_v4.py
"""
import io
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used by mpl)
import streamlit as st
from scipy import ndimage
import imageio.v2 as imageio

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Double-Slit w/ Residual Memory", layout="wide")
st.title("🧠 Double-Slit Simulator — Residual Memory & Observation")
st.markdown(
    "This simulator shows double-slit wave interference, how observation perturbs the field, "
    "and how a temporal **residual (memory) field** accumulates past activity and feeds back on the wave."
)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Simulation controls")

# Grid & frames
nx = st.sidebar.select_slider("Grid resolution (nx)", options=[120, 160, 200, 240, 320], value=200)
frames = st.sidebar.slider("Frames", 10, 200, 60, step=10)

# Observer controls
observer_present = st.sidebar.checkbox("Enable observation", value=True)
observer_type = st.sidebar.selectbox("Observer type", ["detector", "instrument", "human"])
observer_strength = st.sidebar.slider("Observer strength", 0.0, 2.0, 1.0, 0.05)

# Residual memory controls
st.sidebar.markdown("**Residual (memory) parameters**")
residual_enabled = st.sidebar.checkbox("Enable residual memory", value=True)
residual_leak = st.sidebar.slider("Residual leak (add per frame)", 0.0, 0.2, 0.02, step=0.005)
residual_decay = st.sidebar.slider("Residual decay per frame", 0.85, 0.999, 0.98, step=0.005)
memory_coupling = st.sidebar.slider("Memory → wave coupling (λ_mem)", 0.0, 1.0, 0.25, step=0.01)

# Particle detection controls
particle_sensitivity = st.sidebar.slider("Detection sensitivity (multiplier)", 0.2, 3.0, 1.2, step=0.05)
min_area = st.sidebar.slider("Min particle area (pixels)", 1, 40, 6, step=1)

# Visual options & export
enable_3d = st.sidebar.checkbox("3D field view", value=False)
compare_mode = st.sidebar.checkbox("Side-by-side comparison", value=False)
export_metrics = st.sidebar.checkbox("Enable metrics CSV export", value=True)
gif_export = st.sidebar.button("Export GIF (all frames)")

# Misc controls
global_noise = st.sidebar.slider("Global fluctuation amplitude", 0.0, 0.2, 0.05, step=0.01)

# Advanced observer tuning
st.sidebar.markdown("Observer tuning (advanced)")
detector_sigma = st.sidebar.slider("Detector sigma", 0.2, 2.0, 0.8, step=0.05)
human_noise_amp = st.sidebar.slider("Human noise amplitude", 0.0, 0.4, 0.12, step=0.01)
human_bias_x = st.sidebar.slider("Human bias x", -2.0, 2.0, -0.5, step=0.1)

# ----------------------------
# Grid & base wave
# ----------------------------
x_grid = np.linspace(-6, 6, nx)
y_grid = np.linspace(0, 6, nx // 2)
X, Y = np.meshgrid(x_grid, y_grid)

def double_slit_base(X, Y, sep=1.0, k=2.2):
    """Return a complex-valued base field for two slits (amplitude-modulated)."""
    # Gaussian slits in Y with slight X localization
    slit1 = np.exp(-((X + sep)**2 + (Y-0.5)**2) / 0.25)
    slit2 = np.exp(-((X - sep)**2 + (Y-0.5)**2) / 0.25)
    phase = np.exp(1j * (k * np.sqrt((X**2 + Y**2) + 1e-9)))
    psi = phase * (slit1 + slit2)
    return psi

psi_base = double_slit_base(X, Y, sep=1.0, k=2.2)

# ----------------------------
# Observer field generator
# ----------------------------
def observer_field(X, Y, obs_type, strength, t):
    if obs_type == "detector":
        # localized gaussian near slits with oscillatory phase effect
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

# ----------------------------
# Residual & detection helpers
# ----------------------------
def normalize(arr):
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    if amax - amin < 1e-12:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)

def detect_particles_from_residual(resid, sensitivity, min_area):
    # use threshold relative to resid stats
    thresh = np.mean(resid) * sensitivity + 1e-6
    mask = resid > thresh
    labeled, n = ndimage.label(mask)
    objs = ndimage.find_objects(labeled)
    centers = []
    for i, slc in enumerate(objs):
        if slc is None:
            continue
        region = (labeled[slc] == (i + 1))
        area = int(region.sum())
        if area >= min_area:
            cy, cx = ndimage.center_of_mass(region)
            row0, col0 = slc[0].start, slc[1].start
            centers.append({"centroid_idx": (row0 + cy, col0 + cx), "area": area})
    return centers

def idx_to_coords(idx, grid_shape, xg, yg):
    r, c = idx
    y_coord = np.interp(r, [0, grid_shape[0]-1], [yg[0], yg[-1]])
    x_coord = np.interp(c, [0, grid_shape[1]-1], [xg[0], xg[-1]])
    return x_coord, y_coord

# ----------------------------
# UI placeholders & state
# ----------------------------
plot_ph = st.empty()
residual_ph = st.empty()
stats_ph = st.empty()

if "residual" not in st.session_state:
    st.session_state.residual = np.zeros_like(psi_base.real)
if "tracked" not in st.session_state:
    st.session_state.tracked = {}
if "next_pid" not in st.session_state:
    st.session_state.next_pid = 1
if "metrics" not in st.session_state:
    st.session_state.metrics = []

# ----------------------------
# Per-frame compute + render
# ----------------------------
def compute_frame(fi, residual_state):
    t = 2 * np.pi * (fi / max(1, frames))
    # base time modulation
    psi_t = psi_base * np.cos(0.4 * t)
    # observer contribution
    obs = observer_field(X, Y, observer_type if observer_present else "none", observer_strength, t)
    # optionally include memory feedback as an additive real-valued potential
    memory_feedback = np.zeros_like(obs)
    if residual_enabled and memory_coupling > 0.0:
        # scale & smooth residual to use as potential
        resid_smooth = ndimage.gaussian_filter(residual_state, sigma=1.0)
        memory_feedback = memory_coupling * (resid_smooth - np.mean(resid_smooth))
    # total psi (complex base + real-valued observer + memory feedback + noise)
    psi_total = psi_t + (obs + memory_feedback) * 1.0  # treat obs and memory as real perturbations
    # add small stochastic fluctuations
    psi_total = psi_total + global_noise * np.random.randn(*psi_total.shape)
    energy = np.abs(psi_total)**2
    energy_n = normalize(energy)

    # update residual: decay + leak*energy
    new_residual = residual_state * residual_decay + residual_leak * energy_n

    # detect particle candidates using residual (memory-enhanced)
    centers = detect_particles_from_residual(new_residual, particle_sensitivity, min_area)

    return psi_total, energy_n, new_residual, centers

def render_energy_and_residual(frame_idx):
    psi_total, energy_n, new_resid, centers = compute_frame(frame_idx, st.session_state.residual)
    st.session_state.residual = new_resid  # commit updated residual into state

    # Update tracked registry (simple nearest matching)
    for c in centers:
        cid_idx = c["centroid_idx"]
        matched = None
        for pid, p in st.session_state.tracked.items():
            prev_idx = p["centroid_idx"]
            dist = np.hypot(prev_idx[0] - cid_idx[0], prev_idx[1] - cid_idx[1])
            if dist <= max(4, min_area):
                matched = pid
                break
        if matched is None:
            pid = st.session_state.next_pid
            st.session_state.tracked[pid] = {"id": pid, "centroid_idx": cid_idx, "first_frame": frame_idx}
            st.session_state.next_pid += 1
        else:
            st.session_state.tracked[matched]["centroid_idx"] = cid_idx
            st.session_state.tracked[matched]["last_frame"] = frame_idx

    # Render main plot (2D or 3D depending on toggle)
    if enable_3d:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, energy_n, cmap="inferno", linewidth=0, antialiased=False)
        ax.set_title(f"3D |ψ|² — Frame {frame_idx+1}/{frames}")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("|ψ|²")
        fig.colorbar(surf, ax=ax, shrink=0.5)
        plot_ph.pyplot(fig)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(energy_n, origin="lower", cmap="inferno", aspect="auto",
                       extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
        # overlay detected centers as cyan rings and tracked as white x
        if centers:
            pts = [idx_to_coords(c["centroid_idx"], energy_n.shape, x_grid, y_grid) for c in centers]
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=80, facecolors="none", edgecolors="cyan", linewidths=1.2, label="Detections")
        if st.session_state.tracked:
            tx = [idx_to_coords(p["centroid_idx"], energy_n.shape, x_grid, y_grid)[0] for p in st.session_state.tracked.values()]
            ty = [idx_to_coords(p["centroid_idx"], energy_n.shape, x_grid, y_grid)[1] for p in st.session_state.tracked.values()]
            ax.scatter(tx, ty, s=40, c="white", marker="x", label="Tracked")
        ax.set_title(f"|ψ|² — Frame {frame_idx+1}/{frames} — Observer: {observer_type if observer_present else 'none'}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized |ψ|²")
        plot_ph.pyplot(fig)
        plt.close(fig)

    # Render residual map
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    res_norm = normalize(st.session_state.residual)
    im2 = ax2.imshow(res_norm, origin="lower", cmap="viridis", aspect="auto",
                     extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax2.set_title("Residual (memory) field")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="residual (norm)")
    residual_ph.pyplot(fig2)
    plt.close(fig2)

    # stats
    stats = {
        "frame": frame_idx,
        "detections_this_frame": len(centers),
        "unique_tracked": len(st.session_state.tracked),
        "residual_mean": float(np.mean(st.session_state.residual)),
        "residual_max": float(np.max(st.session_state.residual))
    }
    st.session_state.metrics.append(stats)
    stats_md = (
        f"**Frame:** {frame_idx+1}/{frames}  \n"
        f"Detections this frame: **{stats['detections_this_frame']}**  \n"
        f"Unique tracked particles: **{stats['unique_tracked']}**  \n"
        f"Residual mean: {stats['residual_mean']:.4f}  \n"
        f"Residual max: {stats['residual_max']:.4f}"
    )
    stats_ph.markdown(stats_md)

# ----------------------------
# Control flow / execution
# ----------------------------
st.sidebar.markdown("---")
run_mode = st.sidebar.radio("Run mode", ["Manual frame", "Play frames"], index=0)
if run_mode == "Manual frame":
    slider_frame = st.sidebar.slider("Frame index", 0, max(0, frames-1), 0)
    if st.sidebar.button("Render frame"):
        render_energy_and_residual(int(slider_frame))
else:
    if st.sidebar.button("▶ Play (run through frames)"):
        # reset residual and tracking if user wants a fresh run
        st.session_state.residual = np.zeros_like(st.session_state.residual)
        st.session_state.tracked = {}
        st.session_state.next_pid = 1
        st.session_state.metrics = []
        # run frames
        progress = st.sidebar.progress(0)
        for fi in range(frames):
            render_energy_and_residual(fi)
            progress.progress((fi + 1) / frames)
            time.sleep(0.03)
        st.success("Run complete.")

# ----------------------------
# Side-by-side compare (optional)
# ----------------------------
if compare_mode:
    # show baseline (no observer, no memory) next to current observer+memory at frame 0 for quick check
    figc, axes = plt.subplots(1, 2, figsize=(11, 4))
    # baseline
    psi_b = psi_base * np.cos(0.0)
    energy_b = normalize(np.abs(psi_b)**2)
    axes[0].imshow(energy_b, origin="lower", cmap="inferno", aspect="auto")
    axes[0].set_title("Baseline (no observer/no memory)")
    # current residual-influenced at last frame if exists
    if st.session_state.metrics:
        last_frame = st.session_state.metrics[-1]["frame"]
        psi_t, energy_t, _, _ = compute_frame(last_frame, st.session_state.residual)
        axes[1].imshow(normalize(np.abs(energy_t)**2 if np.iscomplexobj(energy_t) else energy_t), origin="lower", cmap="inferno", aspect="auto")
    else:
        axes[1].imshow(normalize(np.abs(psi_base)**2), origin="lower", cmap="inferno", aspect="auto")
    axes[1].set_title("Observed / Memory-influenced")
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    st.pyplot(figc)
    plt.close(figc)

# ----------------------------
# GIF export (render using current settings)
# ----------------------------
if gif_export:
    st.info("Rendering GIF — this may take some time.")
    frames_imgs = []
    # temporary residual for export
    tmp_res = np.zeros_like(st.session_state.residual)
    for fi in range(frames):
        psi_total, energy_n, tmp_res, centers = compute_frame(fi, tmp_res)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(energy_n, origin="lower", cmap="inferno", aspect="auto")
        ax.set_axis_off()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frames_imgs.append(imageio.imread(buf))
    tmp_gif = "/tmp/double_slit_residual.gif"
    imageio.mimsave(tmp_gif, frames_imgs, fps=12)
    with open(tmp_gif, "rb") as f:
        st.download_button("📥 Download GIF", f, file_name="double_slit_residual.gif", mime="image/gif")
    st.success("GIF export ready.")

# ----------------------------
# Export metrics CSV
# ----------------------------
if export_metrics and st.session_state.metrics:
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=list(st.session_state.metrics[0].keys()))
    writer.writeheader()
    writer.writerows(st.session_state.metrics)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    st.download_button("📥 Download per-frame metrics CSV", data=csv_bytes, file_name="double_slit_metrics.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: change residual parameters (leak, decay, λ_mem) to explore how memory strengthens or weakens particle registration over time.")
