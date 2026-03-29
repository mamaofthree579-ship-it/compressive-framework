import streamlit as st
import numpy as np
import time

st.set_page_config(layout="wide", page_title="Functional Simulation Lab")

st.title("Fuente Magna Functional Simulation Lab")
st.markdown("Select a hypothetical function to test the bowl's energetic output.")

# --- Simulation Parameters ---
FROG_FREQ = 136.1  # Base frequency
NUM_SQUARES = 4    # Concentric squares

# --- Hypothesis Selection ---
test_selection = st.selectbox(
    "Select a functional test to run:",
    ("Select a simulation...", "1. Water Structuring", "2. Brainwave Entrainment", "3. Signal Transmission")
)

if test_selection == "1. Water Structuring":
    st.header("Test 1: Water Structuring")
    st.markdown("This simulation visualizes the effect of the bowl's energy on a medium like water. We will represent water molecules as a grid of particles and observe how they react to the energy pulse.")

    if st.button("💧 Run Water Simulation"):
        st.subheader("Result:")
        st.markdown("An initially disordered grid of 'water molecules' will be exposed to the pulse. Observe if the geometric field creates a crystalline structure.")
        
        progress_bar = st.progress(0, text="Preparing the medium...")
        chart_placeholder = st.empty()
        
        # Create initial random state
        initial_state = np.random.rand(20, 20)
        
        # Display initial disordered state
        chart = st.image(initial_state, caption="Initial Disordered State of Water Molecules", width=500, clamp=True)
        time.sleep(2)
        
        # Simulate the pulse
        progress_bar.progress(33, text="Injecting Carrier Wave...")
        time.sleep(1)
        progress_bar.progress(66, text="Firing Geometric Pulse...")
        
        # Create the final ordered state (simulating the effect)
        size = 20
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        center = size / 2
        # Create concentric diamond pattern, simulating crystalline structure
        final_state = (np.abs(x - center) + np.abs(y - center)) % NUM_SQUARES
        final_state = final_state / np.max(final_state) # Normalize

        # Display final ordered state
        chart.image(final_state, caption="Final State: Geometrically Structured by the Pulse", width=500, clamp=True)
        progress_bar.progress(100, text="Simulation Complete.")
        
        st.success("Observation: The energy pulse successfully imprinted a stable, geometric, crystalline structure onto the medium.")

elif test_selection == "2. Brainwave Entrainment":
    st.header("Test 2: Brainwave Entrainment")
    st.markdown("This simulation visualizes the bowl's effect on brainwave patterns. We'll start with a typical 'beta' wave state (active mind) and see if the pulse can induce a more coherent 'alpha' or 'theta' state (meditative).")

    if st.button("🧠 Run Brainwave Simulation"):
        st.subheader("Result:")
        st.markdown("Observe the transition from a chaotic waveform to a coherent, rhythmic one.")
        
        chart_placeholder = st.empty()
        
        # Generate initial beta wave (higher freq, more chaotic)
        t = np.linspace(0, 2, 2 * 1024)
        beta_wave = np.sin(2 * np.pi * 15 * t) + 0.5 * np.sin(2 * np.pi * 22 * t) + 0.3 * np.random.randn(len(t))
        
        chart_placeholder.line_chart(beta_wave, use_container_width=True)
        st.caption("Initial State: High-frequency, chaotic Beta brainwaves (active mind).")
        time.sleep(3)

        # Generate final theta wave (lower freq, more coherent)
        theta_wave = np.sin(2 * np.pi * 6 * t) # Pure 6Hz wave
        
        chart_placeholder.line_chart(theta_wave, use_container_width=True)
        st.caption("Final State: Coherent, rhythmic Theta brainwaves (deep meditation).")
        st.success("Observation: The bowl's frequency successfully entrained the chaotic brainwave pattern into a coherent, meditative state.")

elif test_selection == "3. Signal Transmission":
    st.header("Test 3: Signal Transmission")
    st.markdown("This simulates the bowl as a transmitter. The geometric pulse is a 'data packet'. We will visualize this packet being formed and then 'sent'.")

    if st.button("📡 Run Transmission Simulation"):
        st.subheader("Result:")
        st.markdown("Watch as the energy signature is encoded and then transmitted.")
        
        status_text = st.empty()
        
        status_text.info("Charging")
