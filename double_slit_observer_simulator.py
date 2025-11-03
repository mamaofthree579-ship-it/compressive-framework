#!/usr/bin/env python3
"""
Double-slit + Observer Simulator (Streamlit)

Simulates classical double-slit interference and then shows how adding an
observer (detector/instrument/human) perturbs the field and changes particle
nucleation patterns according to the Compressive Framework idea.

Run:
    streamlit run double_slit_observer_simulator.py
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import io
import time
import csv

# ---------------------------
# Utilities
# ---------------------------
def build_grid(nx_x=300, nx_y_factor=0.5, xlim=(-8, 8), ylim=(0, 8)):
    nx_x = int(nx_x)
    nx_y = int(max(32, nx_x * nx_y_factor))
    x = np.linspace(xlim[0], xlim[1], nx_x)
    y = np.linspace(ylim[0], ylim[1], nx_y)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

def two_slit_base(X, Y, sep=2.0, k=2.0, phase_offset=0.0):
    r1 = np.sqrt((X + sep)**2 + Y**2 + 1e-9)
    r2 = np.sqrt((X - sep)**2 + Y**2 + 1e-9)
    # Spherical-ish waves (damped amplitude by radius)
    psi = (np.sin(k * r1 + phase_offset) / (r1 + 1e-6)
           + np.sin(k * r2 + phase_offset) / (r2 + 1e-6))
    return psi

def observer_field(X, Y, type_, strength, t, params):
    """
    Generate additional field from observer.
    type_ in {'none','detector','instrument','human'}
    strength: global multiplier [0..1]
    t: normalized phase/time [0..2pi]
    params: dict of extra parameters
    """
    if type_ == 'none' or strength <= 0:
        return np.zeros_like(X)
    if type_ == 'detector':
        # Localized near the slit region(s) — resets phase / adds local phase noise
        x0 = params.get("detector_x", 0.0)
        sigma = params.get("detector_sigma", 0.7)
        # Gaussian localized influence centered at x0 across Y near slit plane
        gauss = np.exp(-((X - x0)**2)/(2*sigma**2)) * np.exp(-((Y-0.5)**2)/(2*(sigma*2)**2))
        phase_k = params.get("detector_k", 6.0)
        local = strength * gauss * np.sin(phase_k * X + 2.0*np.sin(t))
        return local
    if type_ == 'instrument':
        # A low-frequency global harmonic that modulates amplitude/phase
        freq = params.get("inst_freq", 0.9)
        global_h = strength * 0.6 * np.sin(freq * X + 0.5*np.cos(t))
        return global_h
    if type_ == 'human':
        # Complex nonstationary influence:
        # multiple harmonics, amplitude modulation, and stochastic component
        a1 = 0.6 * np.sin(0.9 * X + 0.2*np.sin(1.3*t))
        a2 = 0.4 * np.cos(1.7 * Y + 0.8*np.cos(0.5*t))
        mod = 0.3 * np.sin(0.13 * X * Y + 0.7 * np.sin(0.4*t))
        noise = params.get("human_noise_amp", 0.08) * np.random.randn(*X.shape)
        combined = (a1 + a2 + mod + noise) * strength
        # optionally add an asymmetric bias to represent human positioning
        bias_x = params.get("human_bias_x", -1.0)
        bias = np.exp(-((X-bias_x)**2)/(2*1.8**2))
        return combined * (1 + 0.3 * bias)
    return np.zeros_like(X)

def normalize(A):
    amin, amax = np.nanmin(A), np.nanmax(A)
    if amax - amin < 1e-12:
        return np.zeros_like(A)
    return (A - amin) / (amax - amin)

def detect_particles_from_residual(residual, thresh, min_area):
    mask = residual > thresh
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

def idx_to_coords(centroid_idx, grid_shape, x_grid, y_grid):
    r, c = centroid_idx
    y_coord = np.interp(r, [0, grid_shape[0]-1], [y_grid[0], y_grid[-1]])
    x_coord = np.interp(c, [0, grid_shape[1]-1], [x_grid[0], x_grid[-1]])
    return x_coord, y_coord

# ---------------------------
# Streamlit UI + Controls
# ---------------------------
st.set_page_config(page_title="Double-Slit Observer Simulator", layout="wide")
st.title("🕳️ Double-Slit + Observer Simulator — Compressive Framework")
st.markdown(
    "Baseline: two coherent sources form interference fringes. "
    "Toggle **Observer Present** to add a perturbing harmonic (Detector / Instrument / Human). "
    "Observe how interference changes and particles (residual nucleation) appear."
)

# Layout: sidebar for parameters, main for visuals
with st.sidebar:
    st.header("Simulation Controls")
    nx = st.slider("Grid width resolution (nx)", 120, 480, 320, step=40)
    frames = st.slider("Time steps / frames", 10, 300, 80, step=10)
    source_sep = st.slider("Slit separation (sep)", 0.5, 4.0, 2.0, step=0.1)
    k = st.slider("Wave number k (wavelength control)", 0.5, 6.0, 2.0, step=0.1)
    global_noise = st.slider("Quantum fluctuation amplitude", 0.0, 0.20, 0.04, step=0.01)
    leak = st.slider("Residual leak rate", 0.0, 0.08, 0.02, step=0.005)
    decay = st.slider("Residual decay per frame", 0.90, 0.999, 0.985, step=0.001)
    thresh = st.slider("Particle nucleation threshold", 0.01, 0.5, 0.12, step=0.01)
    min_area = st.slider("Min particle area (px)", 1, 80, 6, step=1)
    observer_present = st.checkbox("Observer present (adds perturbation)", value=False)
    observer_type = st.selectbox("Observer type", ["detector", "instrument", "human"]) if observer_present else st.selectbox("Observer type", ["none"], index=0)
    observer_strength = st.slider("Observer strength", 0.0, 1.0, 0.5 if observer_present else 0.0, step=0.05)
    autoplay = st.checkbox("Autoplay (run through all frames)", value=False)
    show_phase = st.checkbox("Show phase overlay / phase view (toggle)", value=False)
    export_csv = st.checkbox("Enable CSV export of per-frame metrics", value=True)
    st.markdown("---")
    st.markdown("Observer tuning (advanced):")
    detector_sigma = st.slider("Detector sigma (localization)", 0.2, 2.5, 0.7, step=0.1)
    human_noise_amp = st.slider("Human noise amp", 0.0, 0.2, 0.08, step=0.01)
    human_bias_x = st.slider("Human bias (x position)", -4.0, 4.0, -1.0, step=0.2)

# Build grid
x_grid, y_grid, X, Y = build_grid(nx_x=nx, nx_y_factor=0.5, xlim=(-8, 8), ylim=(0, 8))
psi_base = two_slit_base(X, Y, sep=source_sep, k=k)

# Prepare residual accumulator and tracking
residual = np.zeros_like(psi_base)
tracked_particles = {}  # simple registry: id -> dict
next_pid = 1

# per-frame metrics
metrics_rows = []

# UI placeholders
left_col, right_col = st.columns([1.1, 0.9])
plot_placeholder = left_col.empty()
stats_placeholder = right_col.empty()
controls_col = right_col

# Playback controls if not autoplay
frame_slider = None
if not autoplay:
    frame_slider = controls_col.slider("Frame", 0, frames - 1, 0)

run_button = controls_col.button("▶ Run Simulation (single pass)" if autoplay else "▶ Render Frame / Play")
pause_button = controls_col.button("⏸ Pause (stop autoplay)")

# internal run state stored in session state
if "playing" not in st.session_state:
    st.session_state.playing = False
if autoplay and run_button:
    st.session_state.playing = True
elif pause_button:
    st.session_state.playing = False
elif not autoplay and run_button:
    st.session_state.playing = False  # single render

# Helper for one step
def compute_frame(frame_index, tphase):
    global residual, next_pid
    # time-dependent base + observer
    psi_time = psi_base * np.cos(0.5 * tphase)  # slow modulation to visualize motion

    obs = observer_field(X, Y, observer_type if observer_present else 'none',
                         observer_strength, tphase,
                         params={"detector_x": 0.0, "detector_sigma": detector_sigma,
                                 "detector_k": 6.0,
                                 "inst_freq": 0.9,
                                 "human_noise_amp": human_noise_amp,
                                 "human_bias_x": human_bias_x})
    psi_total = psi_time + obs

    # add quantum fluctuations
    psi_total = psi_total + global_noise * np.random.randn(*psi_total.shape)

    # compute intensity and normalized energy
    energy = np.abs(psi_total)**2
    energy_n = normalize(energy)

    # update residual accumulation
    residual = residual * decay + leak * energy_n

    # detect particles
    centers = detect_particles_from_residual(residual, thresh, min_area)
    # register unique tracked particles by nearest neighbor matching
    # simple algorithm: if new center within 6 px of existing centroid, treat as same
    for c in centers:
        cid_idx = c["centroid_idx"]
        matched = None
        for pid, p in tracked_particles.items():
            prev_idx = p["centroid_idx"]
            dist = np.hypot(prev_idx[0] - cid_idx[0], prev_idx[1] - cid_idx[1])
            if dist <= max(6, min_area):
                matched = pid
                break
        if matched is None:
            # new particle
            tracked_particles[next_pid] = {"id": next_pid, "centroid_idx": cid_idx, "area": c["area"], "first_frame": frame_index}
            next_pid += 1
        else:
            # update existing
            tracked_particles[matched]["centroid_idx"] = cid_idx
            tracked_particles[matched]["area"] = c["area"]

    return psi_total, energy_n, centers

# Run/render logic
def render_frame_and_stats(frame_index):
    tphase = 2 * np.pi * (frame_index / max(1, frames))
    psi_total, energy_n, centers = compute_frame(frame_index, tphase)

    # Build figure (left)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.imshow(energy_n, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                   cmap='inferno', aspect='auto', vmin=0, vmax=1)
    if show_phase:
        phase = np.angle(psi_total)
        # overlay phase as contour lines (subtle)
        cs = ax.contour(np.linspace(x_grid[0], x_grid[-1], energy_n.shape[1]),
                        np.linspace(y_grid[0], y_grid[-1], energy_n.shape[0]),
                        phase, levels=12, linewidths=0.5, colors='white', alpha=0.4)
    # show detected centers from this frame
    if centers:
        pts = []
        for c in centers:
            x_c, y_c = idx_to_coords(c["centroid_idx"], energy_n.shape, x_grid, y_grid)
            pts.append((x_c, y_c))
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=90, facecolors='none', edgecolors='cyan', linewidths=1.2, label="Detected nucleation")
    # show tracked particles (persistent)
    if tracked_particles:
        tx = [idx_to_coords(p["centroid_idx"], energy_n.shape, x_grid, y_grid)[0] for p in tracked_particles.values()]
        ty = [idx_to_coords(p["centroid_idx"], energy_n.shape, x_grid, y_grid)[1] for p in tracked_particles.values()]
        ax.scatter(tx, ty, s=40, c='white', marker='x', label='Tracked (unique)')
    ax.set_title(f"Frame {frame_index+1}/{frames} — Observer: {observer_type if observer_present else 'none'} (strength={observer_strength:.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized energy |ψ|²')

    plot_placeholder.pyplot(fig)
    plt.close(fig)

    # Stats
    total_unique = len(tracked_particles)
    stats_md = f"""
**Frame:** {frame_index+1}/{frames}  
**Detections (this frame):** {len(centers)}  
**Total unique tracked particles:** {total_unique}  
**Residual mean:** {float(np.mean(residual)):.4f}  
**Residual max:** {float(np.max(residual)):.4f}
"""
    stats_placeholder.markdown(stats_md)

    # record metrics
    if export_csv:
        metrics_rows.append({
            "frame": frame_index,
            "detections_this_frame": len(centers),
            "total_unique_tracked": total_unique,
            "residual_mean": float(np.mean(residual)),
            "residual_max": float(np.max(residual))
        })

# Execution modes
if st.session_state.playing and autoplay:
    # autoplay through all frames
    prog = st.sidebar.progress(0)
    for fi in range(frames):
        render_frame_and_stats(fi)
        prog.progress((fi + 1) / frames)
        time.sleep(0.02)  # small pause to allow UI update; adjust for speed
    st.session_state.playing = False
else:
    # manual single-frame render or single-pass run
    if autoplay and run_button:
        # run once through frames (non-interactive), similar to autoplay but triggered by run_button
        prog = st.sidebar.progress(0)
        for fi in range(frames):
            render_frame_and_stats(fi)
            prog.progress((fi + 1) / frames)
            time.sleep(0.02)
    else:
        # render frame from slider
        frame_to_show = frame_slider if frame_slider is not None else 0
        render_frame_and_stats(int(frame_to_show))

# Export CSV if requested and metrics exist
if export_csv and metrics_rows:
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=list(metrics_rows[0].keys()))
    writer.writeheader()
    writer.writerows(metrics_rows)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    right_col.download_button("📥 Download per-frame metrics (CSV)", data=csv_bytes, file_name="double_slit_metrics.csv", mime="text/csv")

# Final summary
right_col.markdown("---")
right_col.markdown(f"**Simulation complete — unique tracked particles:** {len(tracked_particles)}")
right_col.markdown("Tip: adjust `Observer type` and `strength` and re-run to compare unobserved vs observed behaviour.")
