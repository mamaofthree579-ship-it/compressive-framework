import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Guarantees headless execution
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

ANIM_DIR = 'animations'
if not os.path.exists(ANIM_DIR):
    os.makedirs(ANIM_DIR)

def safe_animation_save(ani, final_filepath, fps, dpi):
    """
    Saves an animation atomically to prevent Streamlit from reading 
    a partial/corrupted file during the write process.
    """
    # 1. Create a temporary file in the same directory to ensure fast renaming
    dir_name = os.path.dirname(final_filepath)
    with tempfile.NamedTemporaryFile(suffix='.gif', dir=dir_name, delete=False) as tmpfile:
        tmp_path = tmpfile.name
    
    try:
        # 2. Save the animation fully to the temporary location
        ani.save(tmp_path, writer='pillow', fps=fps, dpi=dpi)
        # 3. Atomically swap it to the destination (thread-safe OS operation)
        os.replace(tmp_path, final_filepath)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

def render_plasma_vortex(temp, magnetic_flux):
    filepath = os.path.join(ANIM_DIR, f'plasma_{temp}_{magnetic_flux}.gif')
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0: 
        return filepath

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    FRAMES = 40

    def update(frame):
        ax.clear()
        num_particles = 120
        t = np.linspace(0, 4 * np.pi, num_particles)
        radius = (300 / magnetic_flux) + 0.2 * np.sin(5 * t + (frame / FRAMES) * 2 * np.pi)
        x = radius * np.cos(t + (frame / FRAMES) * 2 * np.pi)
        y = radius * np.sin(t + (frame / FRAMES) * 2 * np.pi)
        z = (temp / 6000.0) * (t / (4 * np.pi)) - (temp / 12000.0)
        ax.scatter(x, y, z, c=cm.plasma(np.linspace(0, 1, num_particles)), s=12, alpha=0.7)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-2, 2)
        ax.set_axis_off(); fig.patch.set_facecolor('black')

    ani = FuncAnimation(fig, update, frames=FRAMES)
    safe_animation_save(ani, filepath, fps=20, dpi=70)
    plt.close(fig)
    return filepath

def render_tesseract(voltage):
    filepath = os.path.join(ANIM_DIR, f'tesseract_{voltage:.1f}.gif')
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0: 
        return filepath

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    FRAMES = 40
    vertices_4d = np.array([[x, y, z, w] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1] for w in [-1, 1]])
    edges = []
    for i in range(len(vertices_4d)):
        for j in range(i + 1, len(vertices_4d)):
            if np.sum(np.abs(vertices_4d[i] - vertices_4d[j]) == 2) == 1: edges.append((i, j))

    def update(frame):
        ax.clear()
        theta = (frame / FRAMES) * 2 * np.pi
        phi = (voltage / 150.0) * np.pi
        R_xw = np.array([[np.cos(theta), 0, 0, -np.sin(theta)],,, [np.sin(theta), 0, 0, np.cos(theta)]])
        R_zw = np.array([,, [0, 0, np.cos(phi), -np.sin(phi)], [0, 0, np.sin(phi), np.cos(phi)]])
        rotated = vertices_4d @ R_xw @ R_zw
        D = 2.5
        projected = np.array([[v*(D/(D-v)), v*(D/(D-v)), v*(D/(D-v))] for v in rotated])
        for edge in edges:
            ax.plot([projected[edge,0], projected[edge,0]], [projected[edge,1], projected[edge,1]], [projected[edge,2], projected[edge,2]], color='cyan', alpha=0.4, lw=1)
        ax.scatter(projected[:,0], projected[:,1], projected[:,2], color='magenta', s=12)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_zlim(-2.5, 2.5)
        ax.set_axis_off(); fig.patch.set_facecolor('black')

    ani = FuncAnimation(fig, update, frames=FRAMES)
    safe_animation_save(ani, filepath, fps=20, dpi=70)
    plt.close(fig)
    return filepath

def render_propulsion(depth, skew):
    filepath = os.path.join(ANIM_DIR, f'propulsion_{depth}_{skew}.gif')
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0: 
        return filepath

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    FRAMES = 60
    x, y = np.arange(-60, 60, 4.0), np.arange(-60, 60, 4.0)
    X, Y = np.meshgrid(x, y)
    craft_pos = np.array([0.0, 0.0])

    def update(frame):
        nonlocal craft_pos
        if frame == 0: craft_pos = np.array([0.0, 0.0])
        ax.clear()
        if 10 < frame <= 40: craft_pos += skew * 1.5
        X_s, Y_s = X - craft_pos, Y - craft_pos
        R = np.sqrt(X_s**2 + Y_s**2)
        Z = -depth * np.exp(-(R**2)/(2*25.0**2)) + (skew * X_s)
        ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, rcount=30, ccount=30)
        ax.scatter(craft_pos, craft_pos, np.min(Z)-5, marker='o', s=80, c='cyan')
        ax.set_zlim(-180, 80); ax.set_axis_off(); fig.patch.set_facecolor('black')

    ani = FuncAnimation(fig, update, frames=FRAMES)
    safe_animation_save(ani, filepath, fps=15, dpi=70)
    plt.close(fig)
    return filepath

def render_invisibility(radius, intensity):
    filepath = os.path.join(ANIM_DIR, f'invisibility_{radius}_{intensity}.gif')
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0: 
        return filepath

    fig, ax = plt.subplots(figsize=(6, 4.5))
    FRAMES = 40
    def update(frame):
        ax.clear()
        active = frame > 10
        strength = max(0, min(1, (frame - 10) / 15.0)) * intensity
        for y_pos in np.linspace(-10, 10, 20):
            x = np.linspace(-20, 20, 100)
            y = np.ones_like(x) * y_pos
            if active and abs(y_pos) < radius:
                y -= np.sign(y_pos) * ((strength / (abs(y_pos) + 0.5)) * np.exp(-(x**2) / 30.0))
            ax.plot(x, y, color='yellow', lw=1)
        ax.add_artist(plt.Circle((0,0), 1.5, color='cyan', zorder=10))
        if active: ax.add_artist(plt.Circle((0,0), radius, color='purple', alpha=0.15, zorder=5))
        ax.set_xlim(-20, 20); ax.set_ylim(-12, 12); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')

    ani = FuncAnimation(fig, update, frames=FRAMES)
    safe_animation_save(ani, filepath, fps=15, dpi=70)
    plt.close(fig)
    return filepath

def render_weapon(intensity):
    filepath = os.path.join(ANIM_DIR, f'weapon_{intensity}.gif')
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0: 
        return filepath

    fig, ax = plt.subplots(figsize=(6, 4.5))
    FRAMES = 60
    def update(frame):
        ax.clear()
        ax.set_xlim(-15, 15); ax.set_ylim(-10, 15); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
        ax.axhline(y=-9.5, color='gray'); ax.plot(0, 8, 'o', markersize=12, color='cyan') 
        if 15 < frame <= 45:
            p = (frame - 15) / 30.0
            ax.add_artist(Polygon([(-1*intensity, 7), (1*intensity, 7), (1-p, -9), (-1+p, -9)], color='red', alpha=0.4 + 0.5*p))
        if frame <= 45: ax.add_artist(plt.Rectangle((-1, -10), 2, 1, color='gray')) 
        else: ax.add_artist(plt.Circle((0, -9), 4.0 * (frame-45)/15.0 * intensity, color='yellow', alpha=1.0 - (frame-45)/15.0))

    ani = FuncAnimation(fig, update, frames=FRAMES)
    safe_animation_save(ani, filepath, fps=15, dpi=70)
    plt.close(fig)
    return filepath
