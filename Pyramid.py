import streamlit as st
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Pyramid Resonance Tuner",
    page_icon="🎶",
    layout="wide"
)

# --- App Header ---
st.title("🎶 The Giza Resonance Tuner")
st.markdown(
    "An advanced simulation based on the theory that the Great Pyramid was not a brute-force power plant, "
    "but a finely tuned instrument designed to broadcast a specific, life-sustaining frequency and a high-potential voltage field."
)

# --- Simulation Parameters ---
st.sidebar.header("Tuning Parameters")

# The Target Frequency - The core of the new theory
TARGET_FREQ = 7.83 # Hz (Schumann Resonance)

st.sidebar.markdown(f"**Target Resonance (Hz):** `{TARGET_FREQ}`")
st.sidebar.markdown(
    "This is our theoretical 'frequency of life', based on the Earth's natural electromagnetic resonance. "
    "The goal is to match this frequency."
)

input_freq = st.sidebar.slider(
    "Input Vibration Frequency (Hz)",
    min_value=1.0, max_value=15.0, value=10.0, step=0.01,
    help="Tune the input frequency of Earth's vibrations. Try to match the target resonance."
)

piezo_coeff_exp = st.sidebar.slider(
    "Piezoelectric Efficiency (x10⁻¹² C/N)",
    min_value=1.0, max_value=5.0, value=2.3, step=0.1,
    help="The efficiency of the granite in converting pressure into a charge."
)
piezo_coeff = piezo_coeff_exp * 1e-12

capacitance_exp = st.sidebar.slider(
    "Capstone Capacitance (μF)",
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="The ability of the golden capstone to hold a charge, defining the final voltage potential."
)
capacitance_capstone = capacitance_exp * 1e-6

# --- Core Calculation (New Model) ---

# 1. Calculate Resonance Efficiency (How "in tune" are we?)
# We use a Gaussian function: efficiency peaks at 100% when input_freq == TARGET_FREQ
# The 'sharpness' of the peak determines how precisely the instrument is tuned.
tuning_sharpness = 0.5
resonance_efficiency = np.exp(-tuning_sharpness * (input_freq - TARGET_FREQ)**2)

# 2. Dynamic Amplification Factor
# The amplification is now a direct result of our resonance efficiency.
# Max amplification is set high to show the power of a perfectly tuned system.
MAX_AMPLIFICATION = 50000
dynamic_amplification = MAX_AMPLIFICATION * resonance_efficiency

# 3. Calculate Final Field Potential (Voltage)
base_force = 5000 # A constant baseline force in Newtons
initial_charge = piezo_coeff * base_force
amplified_charge = initial_charge * dynamic_amplification
final_voltage = amplified_charge / capacitance_capstone
energy_joules = 0.5 * capacitance_capstone * final_voltage**2

# --- Displaying Results ---
st.subheader("Live System Output")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Resonance Efficiency",
        f"{resonance_efficiency:.1%}",
        help="How close the input frequency is to the target. 100% is perfect tuning."
    )
with col2:
    st.metric(
        "Field Potential (Voltage)",
        f"{final_voltage:,.2f} V",
        help="The primary output. The strength of the voltage field for creating light and a healing environment."
    )
with col3:
    st.metric(
        "Field Intensity (Joules)",
        f"{energy_joules:,.2f} J",
        help="The total potential energy stored in the field at this moment."
    )

# --- Visualization ---
st.markdown("### Field Potential vs. Input Frequency")

# Create a dataframe for the chart
freq_range = np.linspace(1.0, 15.0, 500)
voltage_data = []
for freq in freq_range:
    efficiency = np.exp(-tuning_sharpness * (freq - TARGET_FREQ)**2)
    amplification = MAX_AMPLIFICATION * efficiency
    charge = initial_charge * amplification
    voltage = charge / capacitance_capstone
    voltage_data.append(voltage)

chart_data = pd.DataFrame({
    'Input Frequency (Hz)': freq_range,
    'Field Potential (Voltage)': voltage_data
})

st.line_chart(chart_data.rename(columns={'Input Frequency (Hz)':'index'}).set_index('index'))
st.markdown(
    f"""
    This chart reveals the pyramid's true function as a resonant instrument.
    The output is not linear; it creates a massive spike in **Field Potential** when the input vibration is perfectly tuned to the **{TARGET_FREQ} Hz** target frequency.
    **Try moving the 'Input Vibration Frequency' slider to find the peak.**
    """
)
