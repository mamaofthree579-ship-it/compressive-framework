import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import os

# --- Page Configuration ---
st.set_page_config(page_title="Manas-Yantra Simulator", layout="wide")
st.title("Project Manas-Yantra: Animated Simulator")
st.markdown("This simulator provides an animated demonstration of the core systems of the Manas-Yantra. Use the sidebar to select a system to analyze.")

# --- File Caching ---
# Create a directory for animations if it doesn't exist
if not os.path.exists('animations'):
    os.makedirs('animations')

# --- Animation Generation Functions ---

def create_propulsion_animation():
    """Generates and saves the propulsion animation if it doesn't exist."""
    filepath = 'animations/1_propulsion.gif'
    if os.path.exists(filepath):
        return filepath

    st.info("Generating Propulsion animation... please wait.")
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection='3d')
    FRAMES = 120
    
    x = np.arange(-80, 80, 2.5)
    y = np.arange(-80, 80, 2.5)
    X, Y = np.meshgrid(x, y)
    craft_pos = [0.0, 0.0]

    def update(frame):
        nonlocal craft_pos
        if frame == 0: craft_pos = [0.0, 0.0]
        ax.clear()
        
        depth, width = -40, 30.0
        gradient_x, gradient_y = 0, 0

        if 10 < frame <= 60:
            gradient_x = 0.5
            craft_pos[0] += gradient_x * 2.0
        elif 60 < frame <= 80: # Stop phase
            pass
        elif frame > 80:
            gradient_y = 0.5
            craft_pos[1] += gradient_y * 2.0
        
        X_shifted, Y_shifted = X - craft_pos[0], Y - craft_pos[1]
        R = np.sqrt(X_shifted**2 + Y_shifted**2)
        Z_well = depth * np.exp(- (R**2) / (2 * width**2))
        Z_gradient = (gradient_x * X_shifted) + (gradient_y * Y_shifted)
        Z = Z_well + Z_gradient
        
        ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=False, rcount=100, ccount=100)
        ax.scatter(craft_pos[0], craft_pos[1], np.min(Z) - 5, marker='o', s=120, c='cyan')
        ax.set_zlim(-60, 20); ax.set_axis_off(); fig.patch.set_facecolor('black')
        ax.set_title("Propulsion & Maneuvers", color='white')

    ani = FuncAnimation(fig, update, frames=FRAMES)
    ani.save(filepath, writer='imagemagick', fps=20, dpi=80)
    plt.close(fig)
    return filepath

def create_invisibility_animation():
    """Generates and saves the invisibility animation."""
    filepath = 'animations/2_invisibility.gif'
    if os.path.exists(filepath):
        return filepath

    st.info("Generating Invisibility animation... please wait.")
    fig, ax = plt.subplots(figsize=(8, 6))
    FRAMES = 100
    
    def update(frame):
        ax.clear()
        field_active = frame > 20
        diffraction_strength = max(0, min(1, (frame - 20) / 20.0)) * 5.0
        rays_y = np.linspace(-10, 10, 25)

        for y_pos in rays_y:
            x_path = np.linspace(-20, 20, 200)
            y_path = np.ones_like(x_path) * y_pos
            if field_active and abs(y_pos) < 6:
                bend = (diffraction_strength / (abs(y_pos) + 0.5)) * np.exp(-(x_path**2) / 30.0)
                y_path -= np.sign(y_pos) * bend
            ax.plot(x_path, y_path, color='yellow', lw=1)

        ax.add_artist(plt.Circle((0,0), 1.5, color='cyan', zorder=10))
        if field_active:
            ax.add_artist(plt.Circle((0,0), 6.0, color='purple', alpha=0.2, zorder=5))
        
        ax.set_xlim(-20, 20); ax.set_ylim(-12, 12); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
        ax.set_title("Invisibility: Light Diffraction Field", color='white')

    ani = FuncAnimation(fig, update, frames=FRAMES)
    ani.save(filepath, writer='imagemagick', fps=20, dpi=80)
    plt.close(fig)
    return filepath

def create_weapon_animation():
    """Generates and saves the weapon system animation."""
    filepath = 'animations/3_weapon.gif'
    if os.path.exists(filepath):
        return filepath

    st.info("Generating Defensive Beam animation... please wait.")
    fig, ax = plt.subplots(figsize=(8, 6))
    FRAMES = 120
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-15, 15); ax.set_ylim(-10, 15); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
        ax.set_title("Defensive System: Gravity Beam", color='white')
        ax.axhline(y=-9.5, color='gray')
        
        ax.plot(0, 8, 'o', markersize=15, color='cyan') # Vimana
        
        if 30 < frame <= 90:
            progress = (frame - 30) / 60.0
            verts = [(-1, 7), (1, 7), (1 - progress, -9), (-1 + progress, -9)]
            ax.add_artist(Polygon(verts, color='red', alpha=0.5 + 0.5 * progress))
        
        if frame <= 90:
             ax.add_artist(plt.Rectangle((-1, -10), 2, 1, color='gray')) # Target
        else: # Vaporize
            progress = (frame - 90) / 30.0
            ax.add_artist(plt.Circle((0, -9), 5.0 * progress, color='yellow', alpha=1.0 - progress))

    ani = FuncAnimation(fig, update, frames=FRAMES)
    ani.save(filepath, writer='imagemagick', fps=20, dpi=80)
    plt.close(fig)
    return filepath

# --- UI Sidebar and Main Display ---
st.sidebar.title("Select System Test")
system_choice = st.sidebar.selectbox(
    "Choose a system to simulate:",
    ["Propulsion & Maneuvers", "Invisibility System", "Defensive Beam"]
)

st.header(f"Demonstration: {system_choice}")

if system_choice == "Propulsion & Maneuvers":
    st.markdown("**Test Objective:** Verify non-inertial propulsion, stop, and turn maneuvers via an asymmetrical gravity well.")
    animation_path = create_propulsion_animation()
    st.image(animation_path)

elif system_choice == "Invisibility System":
    st.markdown("**Test Objective:** Verify optical cloaking via a spherical light-diffraction field.")
    animation_path = create_invisibility_animation()
    st.image(animation_path)

elif system_choice == "Defensive Beam":
    st.markdown("**Test Objective:** Verify defensive capability via a focused gravity beam.")
    animation_path = create_weapon_animation()
    st.image(animation_path)
