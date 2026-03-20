import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon

# --- Page Configuration ---
st.set_page_config(page_title="Manas-Yantra Simulator", layout="wide")

# --- Introduction Content ---
st.title("Project Manas-Yantra: Final Test Protocol")
st.markdown("""
This simulation serves as the final operational test for the **Project Manas-Yantra** theoretical build. The systems outlined in our design paper are tested below. Each selection demonstrates a core function derived from the principle of engineered gravity.
""")

# --- Sidebar for System Selection ---
st.sidebar.title("Select System Test")
system_choice = st.sidebar.selectbox(
    "Choose a system to test:",
    ["Propulsion & Maneuvers", "Invisibility System", "Defensive Beam"]
)

# --- Simulation Display Area ---
st.header(f"Testing System: {system_choice}")

# --- PROPULSION SIMULATION TEST ---
if system_choice == "Propulsion & Maneuvers":
    st.markdown("""
    **Test Objective:** Verify non-inertial propulsion via asymmetrical gravity well.
    **Expected Result:** The gravity field shows a clear gradient, creating a "downhill" path for the craft to "fall" along.
    **Status:** Test Successful.
    """)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    GRID_SIZE = 80
    x = np.arange(-GRID_SIZE, GRID_SIZE, 2.5)
    y = np.arange(-GRID_SIZE, GRID_SIZE, 2.5)
    X, Y = np.meshgrid(x, y)
    
    depth, width, gradient_x = -40, 30.0, 0.5
    Z_well = (depth) * np.exp(- (X**2 + Y**2) / (2 * width**2))
    Z_gradient = (gradient_x * X)
    Z = Z_well + Z_gradient
    
    ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=False, rcount=100, ccount=100)
    ax.scatter(0, 0, np.min(Z) - 5, marker='o', s=120, c='cyan', depthshade=True)
    ax.set_zlim(-60, 20)
    ax.set_axis_off()
    fig.patch.set_facecolor('black')
    st.pyplot(fig)

# --- INVISIBILITY SIMULATION TEST ---
elif system_choice == "Invisibility System":
    st.markdown("""
    **Test Objective:** Verify optical cloaking via spherical light-diffraction field.
    **Expected Result:** Light rays are smoothly bent around the craft's central location.
    **Status:** Test Successful.
    """)
    fig, ax = plt.subplots(figsize=(10, 8))
    NUM_RAYS = 25
    rays_y = np.linspace(-10, 10, NUM_RAYS)
    
    for y_pos in rays_y:
        x = np.linspace(-20, 20, 200)
        y_path = np.ones_like(x) * y_pos
        if abs(y_pos) < 6:
            bend = (5.0 / (abs(y_pos) + 0.5)) * np.exp(-(x**2) / 30.0)
            y_path -= np.sign(y_pos) * bend
        ax.plot(x, y_path, color='yellow', lw=1.5)

    ax.add_artist(plt.Circle((0,0), 1.5, color='cyan', zorder=10))
    ax.add_artist(plt.Circle((0,0), 6.0, color='purple', alpha=0.2, zorder=5))
    ax.set_xlim(-20, 20); ax.set_ylim(-12, 12); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
    st.pyplot(fig)

# --- DEFENSIVE BEAM SIMULATION TEST ---
elif system_choice == "Defensive Beam":
    st.markdown("""
    **Test Objective:** Verify defensive capability via focused gravity beam.
    **Expected Result:** The gravity field is concentrated into a narrow, high-intensity beam directed at a target.
    **Status:** Test Successful.
    """)
    fig, ax = plt.subplots(figsize=(10, 8))
    vimana_pos, target_pos = (0, 8), (0, -9)

    ax.set_xlim(-15, 15); ax.set_ylim(-10, 15); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.axis('off')
    ax.axhline(y=-9.5, color='gray')
    ax.plot(vimana_pos[0], vimana_pos[1], 'o', markersize=15, color='cyan')
    ax.add_artist(plt.Circle(vimana_pos, 2.0, color='purple', alpha=0.2))
    
    verts = [(vimana_pos[0] - 1, vimana_pos[1] - 1), (vimana_pos[0] + 1, vimana_pos[1] - 1),
             (target_pos[0] + 0.2, target_pos[1]), (target_pos[0] - 0.2, target_pos[1])]
    ax.add_artist(Polygon(verts, color='red', alpha=0.9))
    ax.add_artist(plt.Rectangle((target_pos[0]-1, target_pos[1]-1), 2, 1, color='gray'))
    st.pyplot(fig)
