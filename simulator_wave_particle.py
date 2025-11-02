#!/usr/bin/env python3
"""
simulator_wave_particle.py

A compact simulator that shows quantum waveform movement with quantum fluctuations,
residual buildup (leakage), and particle nucleation when residual concentration
exceeds a threshold.

Usage:
    python simulator_wave_particle.py          # run with defaults
    python simulator_wave_particle.py --frames 600 --nx 400 --save sim.mp4

Requirements:
    numpy
    matplotlib
    scipy

Optional (for saving animation):
    imageio or ffmpeg available to matplotlib

Install:
    pip install numpy matplotlib scipy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import ndimage

# ---------------------------
# Helper / physics functions
# ---------------------------

def build_grid(nx, xlim=(-6, 6), tlim=(0, 2*np.pi)):
    x = np.linspace(xlim[0], xlim[1], nx)
    t = np.linspace(tlim[0], tlim[1], nx)
    X, T = np.meshgrid(x, t)
    return x, X, T

def compute_wavefields(X, t_phase, params):
    """
    Returns component fields and their combined amplitude on grid X at phase t_phase.
    params: dict containing alpha,beta,gamma,omega,envelope_sigma and toggles
    """
    alpha = params['alpha']     # graviton weight
    beta = params['beta']       # chronon weight
    gamma = params['gamma']     # cognon weight
    omega = params['omega']
    env = np.exp(-(X**2) / (2.0 * params['envelope_sigma']**2))

    # three wave components with small phase offsets
    graviton = params['show_graviton'] * np.sin(alpha * (omega * X - t_phase))
    chronon  = params['show_chronon']  * np.sin(beta  * (omega * X - 0.7*t_phase) + 0.3)
    cognon   = params['show_cognon']   * np.sin(gamma * (omega * X - 1.3*t_phase) + 0.7)

    combined = (graviton + chronon + cognon) * env
    return graviton, chronon, cognon, combined

def normalize(A):
    amin, amax = np.nanmin(A), np.nanmax(A)
    if amax - amin < 1e-12:
        return np.zeros_like(A)
    return (A - amin) / (amax - amin)

def detect_particles(residual, threshold, min_pixels=3):
    """
    Detect connected regions in residual above threshold. Return list of centroids and labeled mask.
    """
    mask = residual > threshold
    labeled, n = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    particles = []
    for i, slc in enumerate(objects):
        if slc is None:
            continue
        region = (labeled[slc] == (i + 1))
        area = int(region.sum())
        if area >= min_pixels:
            cy, cx = ndimage.center_of_mass(region)
            # convert to global indices:
            row0, col0 = slc[0].start, slc[1].start
            centroid_idx = (row0 + cy, col0 + cx)
            particles.append({"id": i+1, "area": area, "centroid_idx": centroid_idx})
    return particles, labeled

# ---------------------------
# Simulator / animation
# ---------------------------

def run_simulation(
    nx=300,
    frames=300,
    alpha=0.7, beta=0.5, gamma=0.6,
    omega=3.0,
    envelope_sigma=3.0,
    fluct_amp=0.06,
    residual_decay=0.98,
    residual_leak=0.02,
    nucleation_threshold=0.12,
    min_particle_area=4,
    save_path=None
):
    x, X, Tgrid = build_grid(nx)
    # residual field stores accumulated energy where compression persists
    residual = np.zeros_like(X, dtype=float)

    params = {
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'omega': omega, 'envelope_sigma': envelope_sigma,
        'show_graviton': 1.0, 'show_chronon': 1.0, 'show_cognon': 1.0
    }

    # For visualization: we'll animate a single row (time-phase across rows)
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    ax_heat = fig.add_subplot(gs[0, :2])
    ax_slice = fig.add_subplot(gs[0, 2])
    ax_count = fig.add_subplot(gs[1, :])

    heat_im = ax_heat.imshow(np.zeros_like(X), origin='lower', extent=[x[0], x[-1], 0, 2*np.pi],
                             cmap='plasma', aspect='auto', vmin=0, vmax=1)
    ax_heat.set_title("Compression Energy (Ψ²) — time vs x")
    ax_heat.set_xlabel("x")
    ax_heat.set_ylabel("phase (t)")

    line_comb, = ax_slice.plot([], [], lw=1.5, label="Ψ (combined)")
    line_energy, = ax_slice.plot([], [], lw=1, label="energy (normalized)", color='orange')
    ax_slice.set_xlim(x[0], x[-1])
    ax_slice.set_ylim(-1.2, 1.2)
    ax_slice.set_title("Spatial slice (middle phase)")
    ax_slice.legend(loc='upper right', fontsize=8)

    count_plot, = ax_count.plot([], [], '-o')
    ax_count.set_xlim(0, frames)
    ax_count.set_ylim(0, 30)
    ax_count.set_title("Particle count over time")
    ax_count.set_xlabel("frame")
    ax_count.set_ylabel("count")

    particle_scat = ax_heat.scatter([], [], s=60, facecolors='none', edgecolors='white', linewidths=1.2)

    counts = []
    particle_tracks = []  # list of lists of (frame, x_coord, t_coord)
    detected_ids = 0

    def init():
        heat_im.set_data(np.zeros_like(X))
        line_comb.set_data([], [])
        line_energy.set_data([], [])
        count_plot.set_data([], [])
        particle_scat.set_offsets(np.empty((0,2)))
        return heat_im, line_comb, line_energy, count_plot, particle_scat

    def update(frame):
        nonlocal residual, detected_ids, particle_tracks

        t_phase = 2*np.pi * (frame / frames)  # rotate through a full cycle over frames
        # compute fields
        gr, ch, co, combined = compute_wavefields(X, t_phase, params)

        # add small quantum fluctuations (spatio-temporal white noise)
        noise = fluct_amp * np.random.randn(*combined.shape)
        combined_noisy = combined + noise

        # local energy (compression) proxy
        energy = combined_noisy**2
        energy_norm = normalize(energy)

        # accumulate residual where energy exists (leakage model)
        # residual decays each frame, then accrues a portion of current energy
        residual = residual * residual_decay + residual_leak * energy_norm

        # detect nucleation (particles) on residual
        particles, labeled = detect_particles(residual, nucleation_threshold, min_pixels=min_particle_area)

        # map centroids indices -> coordinates for plotting
        centroids = []
        for p in particles:
            r, c = p['centroid_idx']
            # convert row index (phase) to t coordinate and column index to x coordinate
            t_coord = np.interp(r, [0, X.shape[0]-1], [0, 2*np.pi])
            x_coord = np.interp(c, [0, X.shape[1]-1], [x[0], x[-1]])
            centroids.append((x_coord, t_coord))

            # simple tracking: append to particle_tracks if near an existing track
            matched = False
            for track in particle_tracks:
                last_frame, lx, lt = track[-1]
                if abs(frame - last_frame) <= 3 and np.hypot(lx - x_coord, lt - t_coord) < 0.3:
                    track.append((frame, x_coord, t_coord))
                    matched = True
                    break
            if not matched:
                detected_ids += 1
                particle_tracks.append([(frame, x_coord, t_coord)])

        # Limit particle_tracks length for clarity
        particle_tracks = [trk for trk in particle_tracks if (frame - trk[-1][0]) < 50]

        # Update heatmap / images
        heat_im.set_data(energy_norm)

        # Update spatial slice (middle row)
        midrow = energy_norm.shape[0] // 2
        line_comb.set_data(x, combined_noisy[midrow, :])
        line_energy.set_data(x, energy_norm[midrow, :])

        # update particle scatter
        if centroids:
            particle_scat.set_offsets(np.array(centroids))
        else:
            particle_scat.set_offsets(np.empty((0,2)))

        # update count plot
        counts.append(len(centroids))
        count_plot.set_data(np.arange(len(counts)), counts)
        ax_count.set_xlim(0, max(50, len(counts)))
        ax_count.set_ylim(0, max(8, max(counts)+2))

        # draw tracks on heatmap
        # clear previous track lines
        # we redraw tracks each frame for simplicity
        for coll in ax_heat.collections[:]:
            # preserve heatmap (first image) and scatter; remove older line collections if any
            pass
        # overlay simple track points/lines
        for trk in particle_tracks:
            coords = [(pt[1], pt[2]) for pt in trk]  # (x,t)
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            ax_heat.plot(xs, ys, color='white', linewidth=1.0, alpha=0.6)

        ax_heat.set_title(f"Compression Energy (frame {frame}) — particles: {len(centroids)}")

        return heat_im, line_comb, line_energy, count_plot, particle_scat

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=50, repeat=False)

    if save_path:
        # try to save with writer if available (ffmpeg)
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=20, metadata=dict(artist='CF-DPF'), bitrate=1800)
            ani.save(save_path, writer=writer)
            print(f"Saved animation to {save_path}")
        except Exception as e:
            print("Unable to save via ffmpeg writer:", e)
            print("Attempting to save via imagemagick or fallback...")
            try:
                ani.save(save_path, writer='imagemagick')
                print(f"Saved animation to {save_path}")
            except Exception as e2:
                print("Final save attempt failed:", e2)
    else:
        plt.show()

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Wave-particle compressive simulator")
    p.add_argument("--nx", type=int, default=300, help="grid resolution")
    p.add_argument("--frames", type=int, default=400, help="animation frames")
    p.add_argument("--alpha", type=float, default=0.7, help="graviton scale")
    p.add_argument("--beta", type=float, default=0.5, help="chronon scale")
    p.add_argument("--gamma", type=float, default=0.6, help="cognon scale")
    p.add_argument("--omega", type=float, default=3.0, help="base frequency")
    p.add_argument("--env", type=float, default=3.0, help="envelope sigma")
    p.add_argument("--fluct", type=float, default=0.06, help="fluctuation amplitude")
    p.add_argument("--decay", type=float, default=0.985, help="residual decay per frame")
    p.add_argument("--leak", type=float, default=0.02, help="residual leak/accrual factor")
    p.add_argument("--thresh", type=float, default=0.12, help="nucleation threshold (residual)")
    p.add_argument("--minarea", type=int, default=4, help="minimum pixels for particle")
    p.add_argument("--save", type=str, default=None, help="optional output file (MP4/GIF) to save animation")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        nx=args.nx,
        frames=args.frames,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        omega=args.omega,
        envelope_sigma=args.env,
        fluct_amp=args.fluct,
        residual_decay=args.decay,
        residual_leak=args.leak,
        nucleation_threshold=args.thresh,
        min_particle_area=args.minarea,
        save_path=args.save
    )
