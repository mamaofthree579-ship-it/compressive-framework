import streamlit as st
import numpy as np
import pandas as pd

# 1. Setup the Page
st.title("The Guardians: Planetary Regulation Simulator")
st.write("Based on the theories of Hope Jones")

# 2. Input Parameters
st.sidebar.header("Variable Inputs")
planetary_stress = st.sidebar.slider("Planetary Stress (Sp) - e.g., Seismic/Magmatic", 0.0, 10.0, 5.0)
human_intent = st.sidebar.slider("Human Coherence (Ih) - Z-Score from REG", 0.0, 5.0, 1.0)
resonance_threshold = 7.5 # Jones's suggested activation constant (Kr)

# 3. The Algorithm Logic
# Probability of Guardian Response (Pg) = f(Sp * Ih)
# We apply the 0.3548 correlation weight to the planetary interaction
def calculate_activation(sp, ih):
    weight = 0.3548
    score = (sp * weight) + (ih * (1 - weight))
    return np.clip(score * 1.5, 0, 10) # Normalized to a 10-point scale

activation_score = calculate_activation(planetary_stress, human_intent)

# 4. Display Results
st.subheader("Simulation Results")
col1, col2 = st.columns(2)
col1.metric("Activation Score", f"{activation_score:.2f} / 10")
col2.metric("Threshold (Kr)", f"{resonance_threshold}")

if activation_score >= resonance_threshold:
    st.success("CONDITION MET: Guardian System Active (Non-Dormant State)")
    st.write("The system has shifted from 'Standby' to 'Active Monitoring'. Expect UAP clusters.")
else:
    st.info("STATUS: System Dormant (Standby Mode)")
    st.write("Planetary stress or human coherence is too low to trigger a response.")

# 5. Data Visualization
chart_data = pd.DataFrame({
    'Metric': ['Planetary Stress', 'Human Intent', 'Total Activation'],
    'Value': [planetary_stress, human_intent, activation_score]
})
st.bar_chart(chart_data.set_index('Metric'))
