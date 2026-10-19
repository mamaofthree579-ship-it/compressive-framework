import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-blocking headless driver
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from PIL import Image

def compile_in_memory_gif(frames, duration=50):
    """Compiles list of PIL images into an animated GIF byte stream."""
    if not frames:
        return None
    out_buf = io.BytesIO()
    frames[0].save(out_buf, format='GIF', save_all=True, append_images=frames[1:], duration=duration, loop=0)
    out_buf.seek(0)
    return out_buf

def render_plasma_vortex(temp, magnetic_flux):
    frames = []
    num_particles = 120
    t = np.linspace(0, 4 * np.pi, num_particles)
    colors = cm.plasma(np.linspace(0, 1, num_particles))
    ring_t = np.linspace(0, 2 * np.pi, 50)
    
    for f in range(25):
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(projection='3d')
        radius = (300 / magnetic_flux) + 0.2 * np.sin(5 * t + (f / 25.0) * 2 * np.pi)
        x = radius * np.cos(t + (f / 25.0) * 2 * np.pi)
        y = radius * np.sin(t + (f / 25.0) * 2 * np.pi)
        z = (temp / 6000.0) * (t / (4 * np.pi)) - (temp / 12000.0)
        
        ax.scatter(x, y, z, c=colors, s=12, alpha=0.7)
        for ring_z in [-0.5, 0.0, 0.5]:
            ax.plot(2.0 * np.cos(ring_z + ring_t), 2.0 * np.sin(ring_z + ring_t), ring_z, color='cyan', lw=1, alpha=0.2)
            
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-2, 2)
        ax.set_axis_off(); fig.patch.set_facecolor('black')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=70)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        
    return compile_in_memory_gif(frames, duration=50)

def render_tesseract(voltage):
    frames = []
    # Build standard 4D Hypercube vertex coordinate parameters cleanly
    v4d = np.array([[x, y, z, w] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1] for w in [-1, 1]])
    edges = [(i, j) for i in range(16) for j in range(i+1, 16) if np.sum(np.abs(v4d[i] - v4d[j]) == 2) == 1]
    
    for f in range(25):
        fig = plt.figure(figsize=(4.5, 4.5))
        ax = fig.add_subplot(projection='3d')
        theta = (f / 25.0) * 2 * np.pi
        phi = (voltage / 150.0) * np.pi
        
        # Explicit flattened matrix declarations to completely avoid parsing crashes
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        Rxw = np.array([[ct, 0, 0, -st], [0, 1, 0, 0], [0, 0, 1, 0], [st, 0, 0, ct]])
        Rzw = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, cp, -sp], [0, 0, sp, cp]])
        
        rotated = v4d @ Rxw @ Rzw
        D = 2.5
        proj = np.array([[v[0]*(D/(D-v[3])), v[1]*(D/(D-v[3])), v[2]*(D/(D-v[3]))] for v in rotated])
        
        for edge in edges:
            ax.plot([proj[edge[0],0], proj[edge[1],0]], [proj[edge[0],1], proj[edge[1],1]], [proj[edge[0],2], proj[edge[1],2]], color='cyan', alpha=0.4, lw=1)
        ax.scatter(proj[:,0], proj[:,1], proj[:,2], color='magenta', s=12)
        
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_zlim(-2.5, 2.5)
        ax.set_axis_off(); fig.patch.set_facecolor('black')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=70)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        
    return compile_in_memory_gif(frames, duration=50)

def render_propulsion(depth, skew):
    frames = []
    x, y = np.arange(-50, 50, 4.0), np.arange(-50, 50, 4.0)
    X, Y = np.meshgrid(x, y)
    
    for f in range(25):
        fig = plt.figure(figsize=(5, 4.5))
        ax = fig.add_subplot(projection='3d')
        craft_pos = skew * 1.5 * min(f, 15)
        
        X_s, Y_s = X - craft_pos, Y - craft_pos
        R = np.sqrt(X_s**2 + Y_s**2)
        Z = -depth * np.exp(-(R**2)/(2*25.0**2)) + (skew * X_s)
        
        ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, rcount=25, ccount=25, antialiased=False)
        ax.scatter(craft_pos, craft_pos, np.min(Z)-5, marker='o', s=80, c='cyan', zorder=10)
        
        ax.set_zlim(-180, 80); ax.set_axis_off(); fig.patch.set_facecolor('black')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=70)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        
    return compile_in_memory_gif(frames, duration=60)

def render_invisibility(radius, intensity):
    frames = []
    rays_y = np.linspace(-10, 10, 20)
    x_coords = np.linspace(-20, 20, 100)
    
    for f in range(20):
        fig, ax = plt.subplots(figsize=(5, 4))
        strength = (f / 20.0) * intensity
        
        for y_pos in rays_y:
            y_coords = np.ones_like(x_coords) * y_pos
            if abs(y_pos) < radius:
                y_coords -= np.sign(y_pos) * ((strength / (abs(y_pos) + 0.5)) * np.exp(-(x_coords**2) / 30.0))
            ax.plot(x_coords, y_coords, color='yellow', lw=1)
            
        ax.add_artist(plt.Circle((0,0), 1.5, color='cyan', zorder=10))
        ax.add_artist(plt.Circle((0,0), radius, color='purple', alpha=0.12, zorder=5))
        
        ax.set_xlim(-20, 20); ax.set_ylim(-12, 12)
        ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=70)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        
    return compile_in_memory_gif(frames, duration=60)

def render_weapon(intensity):
    frames = []
    for f in range(25):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_xlim(-15, 15); ax.set_ylim(-10, 15); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
        ax.axhline(y=-9.5, color='gray'); ax.plot(0, 8, 'o', markersize=12, color='cyan') 
        
        if 5 < f <= 18:
            p = (f - 5) / 13.0
            ax.add_artist(Polygon([(-1*intensity, 7), (1*intensity, 7), (1-p, -9), (-1+p, -9)], color='red', alpha=0.4 + 0.5*p))
        if f <= 18: 
            ax.add_artist(plt.Rectangle((-1, -10), 2, 1, color='gray')) 
        else: 
            p_exp = (f - 18) / 7.0
            ax.add_artist(plt.Circle((0, -9), 4.0 * p_exp * intensity, color='yellow', alpha=1.0 - p_exp))
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=70)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        
    return compile_in_memory_gif(frames, duration=60)
