import streamlit as st
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Pyramid Resonance Simulator",
    page_icon="🔺",
    layout="wide"
)

# --- App Header ---
st.title("🔺 The Giza Resonance Simulator")
st.markdown(
    "This tool provides a conceptual simulation of the 'Pyramid Power Plant' theory. "
    "It models how the Great Pyramid could act as a machine to convert ambient vibrations "
    "into stored electrical energy, focusing on four key stages."
)

# --- Simulation Parameters (User Controls) ---
st.sidebar.header("Simulation Parameters")

st.sidebar.markdown("**1. The Engine: Piezoelectric Granite**")
piezo_coeff_exp = st.sidebar.slider(
    "Piezoelectric Coefficient (x10⁻¹²)",
    min_value=1.0, max_value=5.0, value=2.3, step=0.1,
    help="Efficiency of the King's Chamber granite in converting pressure to charge. Higher is more efficient."
)
piezo_coeff = piezo_coeff_exp * 1e-12

force_n = st.sidebar.slider(
    "Vibrational Force (Newtons)",
    min_value=1000, max_value=10000, value=5000, step=500,
    help="The baseline force from Earth's vibrations acting on the granite."
)

st.sidebar.markdown("**2. The Amplifier: Pyramid Shape**")
amplification_factor = st.sidebar.slider(
    "Resonant Amplification Factor",
    min_value=1, max_value=10000, value=1000, step=10,
    help="How much the pyramid's shape amplifies the initial charge. This is the most critical variable."
)

st.sidebar.markdown("**3. The Collector: Benben Stone**")
capacitance_exp = st.sidebar.slider(
    "Capstone Capacitance (microfarads, μF)",
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="The ability of the golden capstone to store electrical charge. Higher means more storage capacity."
)
capacitance_capstone = capacitance_exp * 1e-6

# --- Core Calculation ---
initial_charge = piezo_coeff * force_n
amplified_charge = initial_charge * amplification_factor
voltage = amplified_charge / capacitance_capstone
energy_stored_joules = 0.5 * capacitance_capstone * voltage**2

# --- Displaying Results ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Amplified Charge on Capstone (Coulombs)",
        f"{amplified_charge:.6f}"
    )
with col2:
    st.metric(
        "Final Voltage on Capstone (Volts)",
        f"{voltage:,.2f} V"
    )
with col3:
    st.metric(
        "Total Energy Stored (Joules)",
        f"{energy_stored_joules:,.2f} J",
        delta=f"{energy_stored_joules / 3600:.4f} Watt-hours"
    )

# --- Visualization ---
st.markdown("### Energy Stored vs. Resonant Amplification")

amp_range = np.linspace(1, 10000, 200)
energy_data = []
for amp in amp_range:
    temp_charge = initial_charge * amp
    temp_volt = temp_charge / capacitance_capstone
    temp_energy = 0.5 * capacitance_capstone * temp_volt**2
    energy_data.append(temp_energy)

chart_data = pd.DataFrame({
    'Resonant Amplification Factor': amp_range,
    'Stored Energy (Joules)': energy_data
})

st.line_chart(chart_data.rename(columns={'Resonant Amplification Factor':'index'}).set_index('index'))

st.markdown(
    "This chart shows the most important concept: the relationship between the pyramid's shape (amplification) "
    "and the energy it can store is **exponential**. A poorly shaped pyramid stores nothing. A perfectly "
    "tuned resonant structure stores immense power. Play with the **Resonant Amplification Factor** slider to see this effect."
)

# --- How to Install and Run (Corrected and Simplified) ---
with st.expander("How to Run This Simulation Yourself"):
    st.markdown("1. **Install Python:** If you don't have it, download and install Python from [python.org](https://www.python.org/downloads/).")
    st.markdown("2. **Install Streamlit & Pandas:** Open your computer's terminal or command prompt and type:")
    st.code("pip install streamlit pandas", language="bash")
    st.markdown("3. **Save the Code:** Copy all the code from this page and save it in a file named `pyramid_sim.py`.")
    st.markdown("4. **Run the App:** In your terminal, navigate to the folder where you saved the file and type:")
    st.code("streamlit run pyramid_sim.py", language="bash")
    st.markdown("Your web browser will open with your new simulation running.")
