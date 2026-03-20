import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import uuid

# --- Core Simulation Function ---
def run_simulation(params, filename, progress_bar):
    """A function to run a single wave interference simulation and save it as a GIF."""
    GRID_SIZE = 100
    TIME_STEPS = 120

    x = np.arange(0, GRID_SIZE)
    y = np.arange(0, GRID_SIZE)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))

    def update(frame):
        progress_bar.progress((frame + 1) / TIME_STEPS)
        ax.clear()

        # Calculate storm wave
        storm_wave = params["storm_amp"] * np.sin(params["storm_freq"] * X - 0.5 * frame)

        # Determine the position of the chant source
        chant_pos = params.get("chant_pos", (0, 0))
        if params.get("moving", False):
            dance_x = chant_pos[0] + params["dance_radius"] * np.cos(params["dance_speed"] * frame)
            dance_y = chant_pos[1] + params["dance_radius"] * np.sin(params["dance_speed"] * frame)
            current_pos = (dance_x, dance_y)
        else:
            current_pos = chant_pos

        # Calculate chant wave
        dist_from_chant = np.sqrt((X - current_pos[0])**2 + (Y - current_pos[1])**2)
        chant_wave = params["chant_amp"] * np.sin(params["chant_freq"] * dist_from_chant - 0.5 * frame + params["chant_phase"])
        chant_wave *= np.exp(-(dist_from_chant / (GRID_SIZE / 5))**2)

        combined_field = storm_wave + chant_wave

        vmin, vmax = params.get("v_range", (-2.5, 2.5))
        im = ax.imshow(combined_field, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(params["title"], fontsize=12)

        if params.get("moving", False):
            ax.plot(current_pos[0], current_pos[1], 'r+', markersize=12)

        ax.set_xticks([])
        ax.set_yticks([])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, blit=True)
    ani.save(filename, writer='imagemagick', fps=20)
    plt.close(fig)
    return filename

# --- Streamlit App Layout ---

st.set_page_config(page_title="Shamanic Physics Simulator", layout="wide")

st.title("Wave Interference as a Model for Shamanic Rituals")
st.write("An interactive simulation based on the paper by **Hope Jones & Meta AI**.")

# --- Sidebar Controls ---
st.sidebar.title("Simulation Controls")

scenario_explanations = {
    "Destructive Interference (Ideal)": "The 'chant' frequency and phase are perfectly set to cancel the 'storm' wave. This is the ideal success case.",
    "Frequency Mismatch": "The 'chant' frequency is wrong. This demonstrates what happens when the ritual is performed with an incorrect pitch or rhythm.",
    "Constructive Interference (Wrong Phase)": "The timing is wrong, causing the 'chant' to amplify the 'storm' instead of canceling it. This represents a catastrophic failure.",
    "Custom": "Set all the parameters manually to explore different possibilities."
}

# Scenario presets
scenario = st.sidebar.selectbox("Choose a Scenario:", list(scenario_explanations.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Wave Parameters")

# Default parameters
params = {
    "Destructive Interference (Ideal)": {"chant_freq": 0.5, "chant_phase_deg": 180, "moving": False},
    "Frequency Mismatch": {"chant_freq": 0.8, "chant_phase_deg": 180, "moving": False},
    "Constructive Interference (Wrong Phase)": {"chant_freq": 0.5, "chant_phase_deg": 0, "moving": False},
    "Custom": {"chant_freq": 0.5, "chant_phase_deg": 180, "moving": False}
}

preset = params[scenario]

# Sliders and checkboxes for user input
chant_freq = st.sidebar.slider("Chant Frequency", 0.1, 2.0, preset["chant_freq"], 0.1)
chant_phase_deg = st.sidebar.slider("Chant Phase (Timing)", 0, 360, preset["chant_phase_deg"], 10)
chant_amp = st.sidebar.slider("Chant Amplitude (Power)", 0.0, 2.0, 1.2, 0.1)
is_moving = st.sidebar.checkbox("Enable Moving Source (Dance)", value=preset["moving"])

dance_radius = 0
dance_speed = 0
if is_moving:
    st.sidebar.subheader("Dance Parameters")
    dance_radius = st.sidebar.slider("Dance Radius", 5, 25, 15)
    dance_speed = st.sidebar.slider("Dance Speed", 0.05, 0.5, 0.1, 0.05)

# --- Main App Area ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Scenario")
    st.info(f"**{scenario}**")
    st.markdown(scenario_explanations[scenario])
    
    st.subheader("What to Look For:")
    if scenario == "Destructive Interference (Ideal)":
        st.markdown("- A stable, dark blue/green 'pocket of calm' appears.")
        st.markdown("- This shows the storm's energy being successfully neutralized.")
    elif scenario == "Frequency Mismatch":
        st.markdown("- Note the chaotic, unpredictable patterns.")
        st.markdown("- No stable calm zone is formed, showing the ritual has failed.")
    elif scenario == "Constructive Interference (Wrong Phase)":
        st.markdown("- Watch for bright yellow 'rogue waves'.")
        st.markdown("- This shows the storm being dangerously amplified.")
    else:
        st.markdown("Experiment with the settings to see what you can discover!")

if st.button("▶️ Generate Simulation"):
    # Build the params dictionary for the simulation function
    user_params = {
        "storm_amp": 1.0,
        "storm_freq": 0.5,
        "chant_amp": chant_amp,
        "chant_freq": chant_freq,
        "chant_phase": np.deg2rad(chant_phase_deg),
        "chant_pos": (70, 50),
        "moving": is_moving,
        "dance_radius": dance_radius,
        "dance_speed": dance_speed,
        "v_range": (-2.5, 2.5),
        "title": scenario
    }

    # Unique filename for each run to avoid caching issues
    output_filename = f"simulation_{uuid.uuid4().hex}.gif"
    
    # Placeholder for the animation and progress bar
    with col2:
        st.subheader("Simulation Output")
        progress_bar = st.progress(0)
        st_image = st.empty()
        st_image.info("⚙️ Generating animation... this may take a minute.")

        # Run the simulation
        run_simulation(user_params, output_filename, progress_bar)
        
        # Display the animation
        st_image.image(output_filename)

        # Provide download button
        with open(output_filename, "rb") as file:
            st.download_button(
                label="⬇️ Download GIF",
                data=file,
                file_name=f"{scenario.replace(' ', '_')}.gif",
                mime="image/gif"
            )
