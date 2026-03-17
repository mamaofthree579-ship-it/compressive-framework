import streamlit as st
import numpy as np

st.set_page_config(page_title="Khipu Acoustic Processor", layout="wide")

st.title("🧶 Khipu Node v5: The Vibrational Tuner")
st.write("Processing Fibonacci (1, 2, 3, 5) ratios into acoustic frequencies.")

# --- SIDEBAR: FREQUENCY CONTROLS ---
st.sidebar.header("🔊 Oscillator Controls")
base_freq = st.sidebar.slider("Base Frequency (Hz)", 100, 1000, 432) # Using the 'Harmonious' 432Hz

# --- FIBONACCI RATIO CALCULATIONS ---
# Ratios: 1/1 (Unison), 2/1 (Octave), 3/2 (Fifth), 5/3 (Major Sixth)
ratios = {"1/1": 1.0, "2/1": 2.0, "3/2": 1.5, "5/3": 1.66}
selected_ratio = st.sidebar.selectbox("Fibonacci Interval", list(ratios.keys()))

target_freq = base_freq * ratios[selected_ratio]

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎵 Interval Analysis")
    st.metric("Base Frequency", f"{base_freq} Hz")
    st.metric(f"Fibonacci {selected_ratio} Frequency", f"{target_freq:.2f} Hz")
    st.write(f"**Interpretation:** This ratio represents the '{selected_ratio}' interval found in the Southern Guild's pentatonic scale.")

with col2:
    st.subheader("🏺 Whistling Jar Simulation")
    if selected_ratio == "3/2":
        st.success("🌊 RESONANCE DETECTED: Perfect Fifth (The 'Sturgeon' Call)")
    elif selected_ratio == "5/3":
        st.info("🐦 RESONANCE DETECTED: Major Sixth (The 'Spirit Bird' Call)")
    else:
        st.warning("🔄 RESONANCE: Fundamental Octave Harmony")

st.divider()
st.subheader("🧬 Cymatic Geometry")
st.write(f"At {target_freq:.2f} Hz, the 'String' is vibrating in a stable node of the maritime database.")
st.write("In String Theory terms, this frequency creates a stable 'vibrational fingerprint' for the guild.")
