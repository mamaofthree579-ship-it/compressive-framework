import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import os

# --- UTILITY FUNCTION ---
def create_output_directory():
    """Creates a directory to save the animations if it doesn't exist."""
    if not os.path.exists('manas_yantra_simulations'):
        os.makedirs('manas_yantra_simulations')

# --- SIMULATION 1: PROPULSION & ADVANCED MANEUVERS ---
def generate_propulsion_maneuver_animation():
    """
    Simulates the core flight model: forward thrust, instantaneous stop,
    and a 90-degree non-inertial turn by manipulating the gravity well.
    """
    print("Generating: Propulsion & Maneuvers Simulation...")
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection='3d')
    FRAMES = 160
    GRID_SIZE = 80
    
    x = np.arange(-GRID_SIZE, GRID_SIZE, 2.5)
    y = np.arange(-GRID_SIZE, GRID_SIZE, 2.5)
    X, Y = np.meshgrid(x, y)
    craft_pos = [0.0, 0.0]

    def update(frame):
        nonlocal craft_pos
        if frame == 0: craft_pos = [0.0, 0.0] # Reset for looping
        ax.clear()
        
        depth, width = -40, 30.0
        pulse = 0.05 * np.sin(2 * np.pi * frame / 20.0)
        gradient_x, gradient_y, flash_color = 0, 0, None

        if 10 < frame <= 60:
            gradient_x = 0.5
            craft_pos[0] += gradient_x * 2.0
        elif 60 < frame <= 100:
            if frame == 61: flash_color = cm.coolwarm
        elif frame > 100:
            if frame == 101: flash_color = cm.spring
            gradient_y = 0.5
            craft_pos[1] += gradient_y * 2.0
        
        X_shifted, Y_shifted = X - craft_pos[0], Y - craft_pos[1]
        R = np.sqrt(X_shifted**2 + Y_shifted**2)
        
        Z_well = (depth * (1 + pulse)) * np.exp(- (R**2) / (2 * width**2))
        Z_gradient = (gradient_x * X_shifted) + (gradient_y * Y_shifted)
        Z = Z_well + Z_gradient

        cmap = flash_color if flash_color else cm.magma
        ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False, rcount=100, ccount=100)
        ax.scatter(craft_pos[0], craft_pos[1], np.min(Z) - 5, marker='o', s=120, c='cyan', depthshade=True)
        ax.set_zlim(-60, 20); ax.set_axis_off(); fig.patch.set_facecolor('black')
        ax.set_title("Propulsion & Maneuvers", color='white')

    ani = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    ani.save('manas_yantra_simulations/1_propulsion_maneuvers.gif', writer='imagemagick', fps=20, dpi=80)
    plt.close(fig)
    print("...Done: 1_propulsion_maneuvers.gif")

# --- SIMULATION 2: INVISIBILITY VIA LIGHT DIFFRACTION ---
def generate_invisibility_animation():
    """
    Simulates the cloaking system, where the gravity field acts as a
    diffraction grating to bend light around the craft.
    """
    print("Generating: Invisibility Simulation...")
    fig, ax = plt.subplots(figsize=(8, 6))
    FRAMES = 100
    NUM_RAYS = 25
    
    rays_y = np.linspace(-10, 10, NUM_RAYS)
    craft_pos, craft_radius = (0, 0), 1.5

    def update(frame):
        ax.clear()
        
        field_active = frame > 20
        diffraction_strength = max(0, min(1, (frame - 20) / 20.0)) * 5.0

        for y_pos in rays_y:
            x = np.linspace(-20, 20, 200)
            y_path = np.ones_like(x) * y_pos
            if field_active and abs(y_pos) < 6:
                bend = (diffraction_strength / (abs(y_pos) + 0.5)) * np.exp(-(x**2) / 30.0)
                y_path -= np.sign(y_pos) * bend
            ax.plot(x, y_path, color='yellow', lw=1)

        ax.add_artist(plt.Circle(craft_pos, craft_radius, color='cyan', zorder=10))
        if field_active:
            alpha_pulse = 0.1 + 0.1 * np.sin(2 * np.pi * frame / 10.0)
            ax.add_artist(plt.Circle(craft_pos, 6.0 * (diffraction_strength/5.0), color='purple', alpha=alpha_pulse, zorder=5))
        
        ax.set_xlim(-20, 20); ax.set_ylim(-12, 12); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
        ax.set_title("Invisibility: Light Diffraction Field", color='white')

    ani = FuncAnimation(fig, update, frames=FRAMES, blit=False)
    
