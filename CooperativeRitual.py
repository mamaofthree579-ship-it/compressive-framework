import streamlit as st
import numpy as np
import matplotlib

# Set the non-GUI backend for Streamlit compatibility
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import uuid

# --- Core Simulation Function (Upgraded for multiple sources) ---
def run_simulation(params, filename, progress_bar):
    """Runs a wave interference simulation with one or two chant sources."""
    GRID_SIZE = 100
    TIME_STEPS = 120

    x = np.arange(0, GRID_SIZE)
    y = np.arange(0, GRID_SIZE)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))

    def update(frame):
        progress_bar.progress((frame + 1) / TIME_STEPS)
        ax.clear()

        # 1. Calculate the Storm Wave
        storm_wave = params["storm_amp"] * np.sin(params["storm_freq"] * X - 0.5 * frame)

        # 2. Calculate the Chant Wave from the first source
        chant_pos_1 = params["chant_pos_1"]
        dist_1 = np.sqrt((X - chant_pos_1[0])**2 + (Y - chant_pos_1[1])**2)
        chant_wave_1 = params["chant_amp"] * np.sin(params["chant_freq"] * dist_1 - 0.5 * frame + params["chant_phase"])
        chant_wave_1 *= np.exp(-(dist_1 / (GRID_SIZE / 5))**2)

        # --- New in v2: Add a second chant source ---
        chant_wave_2 = 0 # Initialize to zero
        if params.get("use_second_shaman", False):
            chant_pos_2 = params["chant_pos_2"]
            dist_2 = np.sqrt((X - chant_pos_2[0])**2 + (Y - chant_pos_2[1])**2)
            chant_wave_2 = params["chant_amp"] * np.sin(params["chant_freq"] * dist_2 - 0.5 * frame + params["chant_phase"])
            chant_wave_2 *= np.exp(-(dist_2 / (GRID_SIZE / 5))**2)
            # Mark the second shaman's position
            ax.plot(chant_pos_2[0], chant_pos_2[1], 'w+', markersize=10, alpha=0.8)

        # 3. Combine all waves
        combined_field = storm_wave + chant_wave_1 + chant_wave_2

        # Display the resulting field
        vmin, vmax = params.get("v_range", (-2.5, 2.5))
        im = ax.imshow(combined_field, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(params["title"], fontsize=12)

        # Mark the first shaman's position
        ax.plot(chant_pos_1[0], chant_pos_1[1], 'r+', markersize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, blit=True)
    ani.save(filename, writer='imagemagick', fps=20)
    plt.close(fig)
    return filename

# --- Streamlit App Layout ---

st.set_page_config(page_title="Cooperative Ritual Simulator", layout="wide")

st.title("The Cooperative Ritual: A Multi-Source Simulation")
st.write("An interactive simulation based on the paper by **Hope Jones & Meta AI**.")

# --- Sidebar Controls ---
st.sidebar.title("Simulation Controls")

scenario_explanations = {
    "Single Shaman (vs. Strong Storm)": "A single shaman attempts to cancel a storm with double the energy. Observe that it is not powerful enough.",
    "Cooperative Ritual (Two Shamans)": "Two shamans work in unison against the strong storm. Their combined power creates a large, stable zone of calm.",
    "Custom": "Set all the parameters manually to explore."
}

scenario = st.sidebar.selectbox("Choose a Scenario:", list(scenario_explanations.keys()))

st.sidebar.markdown("---")

# --- New in v2: Controls for multiple shamans ---
use_second_shaman_default = scenario == "Cooperative Ritual (Two Shamans)"
use_second_shaman = st.sidebar.checkbox("Enable Second Shaman", value=use_second_shaman_default)

st.sidebar.subheader("Shaman 1 Position")
s1_x = st.sidebar.slider("Shaman 1 (X-axis)", 0, 100, 70)
s1_y = st.sidebar.slider("Shaman 1 (Y-axis)", 0, 100, 60)

s2_x, s2_y = 0, 0
if use_second_shaman:
    st.sidebar.subheader("Shaman 2 Position")
    s2_x = st.sidebar.slider("Shaman 2 (X-axis)", 0, 100, 70)
    s2_y = st.sidebar.slider("Shaman 2 (Y-axis)", 0, 100, 40)

st.sidebar.markdown("---")
st.sidebar.subheader("Shared Chant Parameters")
chant_freq = st.sidebar.slider("Chant Frequency", 0.1, 2.0, 0.5, 0.1)
chant_phase_deg = st.sidebar.slider("Chant Phase (Timing)", 0, 360, 180, 10)
chant_amp = st.sidebar.slider("Chant Amplitude (Power)", 0.0, 2.0, 1.2, 0.1)

# --- Main App Area ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Scenario")
    st.info(f"**{scenario}**")
    st.markdown(scenario_explanations[scenario])
