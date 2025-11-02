#!/usr/bin/env python3
"""
streamlit_simulator.py
Streamlit simulator for wave → residual → particle nucleation with:
 - Play / Pause / Reset
 - Persistent particle tracking (unique ids)
 - Color-shifting visualization
 - 3D surface + particle bubbles
 - Final particle count (tracked)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------
# Utilities
# ---------------------------
def build_grid(nx, xlim=(-6, 6)):
    x = np.linspace(xlim[0], xlim[1], nx)
    X = np.tile(x, (nx, 1))
    T = np.linspace(0, 2 * np.pi, nx)
    return x, X, T

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
    if amax - amin < 1e-12:
        return np.zeros_like(A)
    return (A - amin) / (amax - amin)

def detect_regions(mask, min_pixels=3):
    labeled, n = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    regions = []
    for i, slc in enumerate(objects):
        if slc is None:
            continue
        region = (labeled[slc] == (i + 1))
        area = int(region.sum())
        if area >= min_pixels:
            cy, cx = ndimage.center_of_mass(region)
            row0, col0 = slc[0].start, slc[1].start
            centroid = (row0 + cy, col0 + cx)
            regions.append({"label": i + 1, "area": area, "centroid": centroid})
    return regions

def idx_to_coords(centroid_idx, X_shape, x_grid):
    r, c = centroid_idx
    t_coord = np.interp(r, [0, X_shape[0]-1], [0, 2*np.pi])
    x_coord = np.interp(c, [0, X_shape[1]-1], [x_grid[0], x_grid[-1]])
    return x_coord, t_coord

def nearest_match(existing_particles, centroid_idx, max_dist_px=8):
    """Return existing particle id matched to centroid_idx or None."""
    best_id = None
    best_dist = None
    for pid, p in existing_particles.items():
        ex_r, ex_c = p["centroid_idx"]
        dist = np.hypot(ex_r - centroid_idx[0], ex_c - centroid_idx[1])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = pid
    if best_dist is not None and best_dist <= max_dist_px:
        return best_id
    return None

# ---------------------------
# Session state initialization
# ---------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "time_phase" not in st.session_state:
    st.session_state.time_phase = 0.0
if "particles" not in st.session_state:
    # dict of pid -> {centroid_idx, x,t, area, age, track:list}
    st.session_state.particles = {}
if "next_pid" not in st.session_state:
    st.session_state.next_pid = 1

# ---------------------------
# UI: controls
# ---------------------------
st.set_page_config(page_title="CF Compressive Simulator", layout="wide")
st.title("CF — Quantum Wave → Particle Simulator")

st.sidebar.header("Simulation Controls")
nx = st.sidebar.select_slider("Grid resolution", options=[100, 150, 200, 300, 400], value=250)
frames = st.sidebar.slider("Frames", min_value=40, max_value=600, value=180, step=10)
alpha = st.sidebar.slider("α — graviton", 0.0, 2.0, 0.7, 0.05)
beta = st.sidebar.slider("β — chronon", 0.0, 2.0, 0.5, 0.05)
gamma = st.sidebar.slider("γ — cognon", 0.0, 2.0, 0.6, 0.05)
omega = st.sidebar.slider("ω — frequency", 0.5, 8.0, 3.0, 0.1)
env_sigma = st.sidebar.slider("Envelope σ", 0.5, 6.0, 3.0, 0.1)
fluct = st.sidebar.slider("Fluctuation amplitude", 0.0, 0.25, 0.06, 0.01)
decay = st.sidebar.slider("Residual decay per frame", 0.9, 0.999, 0.985, 0.001)
leak = st.sidebar.slider("Residual leak rate", 0.0, 0.15, 0.02, 0.005)
thresh = st.sidebar.slider("Nucleation threshold", 0.01, 0.5, 0.12, 0.01)
minpix = st.sidebar.slider("Min particle area (px)", 1, 40, 4, 1)
merge_px = st.sidebar.slider("Merge distance (px)", 1, 40, 8, 1)
speed = st.sidebar.slider("Frame delay (sec)", 0.01, 0.2, 0.03, 0.01)

# color-shift controls
colormap_list = ["plasma", "viridis", "inferno", "magma", "cividis"]
colormap_idx = st.sidebar.selectbox("Base colormap", options=list(range(len(colormap_list))),
                                    format_func=lambda i: colormap_list[i], index=0)
cycle_colors = st.sidebar.checkbox("Color shift (cycle palettes)", True)

# 3D view controls
elev = st.sidebar.slider("3D elev", 5, 70, 30, 5)
azim = st.sidebar.slider("3D azim", 0, 360, 35, 5)
bubble_scale = st.sidebar.slider("Bubble scale", 20, 200, 80, 10)

# control buttons
col1, col2, col3 = st.sidebar.columns([1,1,1])
with col1:
    if st.button("▶ Play / Pause"):
        st.session_state.running = not st.session_state.running
with col2:
    if st.button("⏮ Reset"):
        st.session_state.running = False
        st.session_state.time_phase = 0.0
        st.session_state.particles = {}
        st.session_state.next_pid = 1
with col3:
    if st.button("Step"):
        # advance a single frame
        st.session_state.running = False
        st.session_state.time_phase += 2*np.pi/frames

st.markdown("Adjust parameters then press **Play** or **Step**. Reset clears tracked particles.")

# ---------------------------
# Build grid + fields
# ---------------------------
x, X, Tgrid = build_grid(nx)
params = {"alpha": alpha, "beta": beta, "gamma": gamma, "omega": omega, "env_sigma": env_sigma}

# placeholders
placeholder_2d = st.empty()
placeholder_3d = st.empty()
progress_bar = st.progress(0)
status = st.empty()

# internal runtime lists
counts = []  # store number of detections each frame

# Simulation function (single frame update)
def step_frame(frame_idx):
    t_phase = 2*np.pi * (frame_idx / frames)
    combined = compute_wavefields(X, t_phase, params)
    combined += np.random.randn(*combined.shape) * fluct
    energy = combined**2
    energy_n = normalize(energy)
    # accumulate residual in session (persist)
    if "residual" not in st.session_state:
        st.session_state.residual = np.zeros_like(energy_n)
    st.session_state.residual = st.session_state.residual * decay + leak * energy_n

    # detect regions in residual
    mask = st.session_state.residual > thresh
    regions = detect_regions(mask, min_pixels=minpix)

    # match regions to existing tracked particles
    existing = st.session_state.particles
    matched_new = set()
    # try to match each detected region to an existing particle
    for reg in regions:
        centroid_idx = reg["centroid"]
        match_id = nearest_match(existing, centroid_idx, max_dist_px=merge_px)
        if match_id is not None:
            # update existing particle
            existing[match_id]["centroid_idx"] = centroid_idx
            xcoord, tcoord = idx_to_coords(centroid_idx, X.shape, x)
            existing[match_id]["x"] = xcoord
            existing[match_id]["t"] = tcoord
            existing[match_id]["area"] = reg["area"]
            existing[match_id]["age"] = existing[match_id].get("age", 0) + 1
            existing[match_id]["track"].append((frame_idx, xcoord, tcoord))
            matched_new.add(match_id)
        else:
            # create new particle
            pid = st.session_state.next_pid
            st.session_state.next_pid += 1
            xcoord, tcoord = idx_to_coords(centroid_idx, X.shape, x)
            existing[pid] = {
                "id": pid,
                "centroid_idx": centroid_idx,
                "x": xcoord,
                "t": tcoord,
                "area": reg["area"],
                "age": 0,
                "track": [(frame_idx, xcoord, tcoord)]
            }
            matched_new.add(pid)

    # decay/age-out particles not matched recently: track age and prune if needed
    to_remove = []
    for pid, p in list(existing.items()):
        # if last track frame is too old relative to current, decrement age
        if p["track"]:
            last_frame = p["track"][-1][0]
            if frame_idx - last_frame > 2 * max(1, int(frames/50)):
                p["age"] -= 1
                if p["age"] < -10:
                    to_remove.append(pid)
    for pid in to_remove:
        existing.pop(pid, None)

    st.session_state.particles = existing
    counts.append(len(regions))

    # ---- Visualization building ----
    # Choose colormap (optionally cycle)
    cmap_name = colormap_list[colormap_idx]
    if cycle_colors:
        # rotate choice based on frame index
        cmap_name = colormap_list[(colormap_idx + (frame_idx//8)) % len(colormap_list)]

    # 2D heatmap
    fig2d, axs = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios":[2, 1]})
    im = axs[0].imshow(energy_n, origin="lower", extent=[x[0], x[-1], 0, 2*np.pi],
                       cmap=cmap_name, aspect="auto")
    axs[0].set_title(f"Frame {frame_idx+1}/{frames} — Detected this frame: {len(regions)} — Tracked total: {len(st.session_state.particles)}")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("t (phase)")
    # overlay particle markers from tracked particles
    if st.session_state.particles:
        pxs = [p["x"] for p in st.session_state.particles.values()]
        pts = [p["t"] for p in st.session_state.particles.values()]
        axs[0].scatter(pxs, pts, s=80, facecolors="none", edgecolors="white", linewidths=1.2)
        # annotate with ids
        for p in st.session_state.particles.values():
            axs[0].text(p["x"], p["t"], f"P{p['id']}", color="white", fontsize=8)

    # count timeline
    axs[1].plot(counts, "-o", color="#222", markersize=3)
    axs[1].set_xlabel("Frame"); axs[1].set_ylabel("Detections/frame")
    axs[1].set_xlim(0, frames); axs[1].set_ylim(0, max(5, max(counts)+2))
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    # 3D surface with bubbles
    fig3d = plt.figure(figsize=(8, 4))
    ax3d = fig3d.add_subplot(111, projection="3d")
    # create mesh for 3D: X grid (x) vs Tgrid (t)
    Tvals = np.linspace(0, 2*np.pi, nx)
    Xgrid, Tgrid_mesh = np.meshgrid(x, Tvals)
    surf = ax3d.plot_surface(Xgrid, Tgrid_mesh, combined, cmap=cmap_name, linewidth=0, antialiased=True, alpha=0.9)
    # potential overlay (scaled)
    pot = -energy_n * 0.35
    ax3d.plot_surface(Xgrid, Tgrid_mesh, pot, cmap="plasma", alpha=0.4)

    # draw particle bubbles from tracked particles
    if st.session_state.particles:
        xs = np.array([p["x"] for p in st.session_state.particles.values()])
        ts = np.array([p["t"] for p in st.session_state.particles.values()])
        # sample z amplitude at nearest index for bubble height
        zs = []
        for p in st.session_state.particles.values():
            r,c = int(round(p["centroid_idx"][0])), int(round(p["centroid_idx"][1]))
            rr = max(0, min(energy.shape[0]-1, r))
            cc = max(0, min(energy.shape[1]-1, c))
            zs.append(combined[rr, cc])
        zs = np.array(zs)
        sizes = np.array([p["area"] for p in st.session_state.particles.values()]) * (bubble_scale/10.0)
        ax3d.scatter(xs, ts, zs, s=sizes, c="white", edgecolors="black", alpha=0.9)

    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("t"); ax3d.set_zlabel("Amplitude")
    # render to placeholders
    placeholder_2d.pyplot(fig2d)
    placeholder_3d.pyplot(fig3d)

# ---------------------------
# Playback loop
# ---------------------------
# If running, iterate frames from current time_phase
if st.session_state.running:
    # compute starting frame index from time_phase
    start_fraction = (st.session_state.time_phase % (2*np.pi)) / (2*np.pi)
    start_frame = int(start_fraction * frames)
    for fi in range(start_frame, frames):
        step_frame(fi)
        progress_bar.progress((fi + 1) / frames)
        status.text(f"Simulating frame {fi+1}/{frames}")
        # increment time_phase
        st.session_state.time_phase += 2*np.pi/frames
        time.sleep(speed)
        # break early if paused by user
        if not st.session_state.running:
            break
    # after run complete, stop running
    st.session_state.running = False

# if not playing, still draw current frame without advancing (so UI shows something)
if not st.session_state.running and "residual" in st.session_state:
    # draw most recent visuals (no advance)
    # build a quick figure for display using current residual + last combined if available
    # we attempt to use last frame index = int((st.session_state.time_phase % (2*np.pi)) / (2*np.pi) * frames)
    frame_idx = int(((st.session_state.time_phase % (2*np.pi)) / (2*np.pi)) * frames)
    # compute combined at that frame for display
    combined = compute_wavefields(X, 2*np.pi*(frame_idx/frames), params)
    combined += np.random.randn(*combined.shape) * 0.0
    energy_n = normalize(combined**2)
    # quick 2D
    fig2d, ax = plt.subplots(1,1, figsize=(8,4))
    ax.imshow(energy_n, origin="lower", extent=[x[0], x[-1], 0, 2*np.pi], cmap=colormap_list[colormap_idx], aspect="auto")
    if st.session_state.particles:
        pxs = [p["x"] for p in st.session_state.particles.values()]
        pts = [p["t"] for p in st.session_state.particles.values()]
        ax.scatter(pxs, pts, s=80, facecolors='none', edgecolors='white')
    placeholder_2d.pyplot(fig2d)

# ---------------------------
# Final summary / download
# ---------------------------
st.markdown("---")
total_tracked = len(st.session_state.particles)
st.metric("Total unique tracked particles", total_tracked)
st.write("Tracked particle IDs:", sorted(list(st.session_state.particles.keys())))
st.write("Tip: Press Reset to clear tracked particles and rerun.")
